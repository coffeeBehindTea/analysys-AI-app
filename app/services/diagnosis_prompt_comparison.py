"""执行基础Prompt与证据优先Prompt的同输入对照。

本模块负责：

1. 调用一个带版本号的诊断草稿Provider；
2. 将成功或预期的LLM失败转换成实验结果；
3. 对相同请求和证据依次运行两个Prompt版本；
4. 组装DiagnosisPromptComparisonCase；
5. 计算两个Prompt组的汇总指标；
6. 组装完整DiagnosisPromptComparisonReport。

本模块不执行检索、不创建OpenAI客户端，
也不写JSON或Markdown报告。
"""

# Callable描述“可以像函数一样调用”的对象。
#
# 当前Clock契约为：
#
# 不接收参数，返回float时间点。
from collections.abc import Callable

# perf_counter()提供单调递增的高精度计时器。
#
# 它适合计算经过时间，不用于记录现实日期时间。
from time import perf_counter

from typing import Literal, Protocol

from app.errors import (
    InvalidLLMResponseError,
    LLMTimeoutError,
    LLMUpstreamError,
)
from app.schemas.diagnosis_llm import (
    DiagnosisLLMDraft,
)
from app.schemas.diagnosis_prompt_comparison import (
    DiagnosisPromptComparisonCase,
    DiagnosisPromptComparisonReport,
    DiagnosisPromptRunResult,
    DiagnosisPromptVariantMetrics,
)
from app.schemas.diagnostics import (
    DiagnosisRequest,
)
from app.schemas.retrieval import (
    HybridRetrievedChunk,
)
from app.services.diagnosis_service import (
    DiagnosisDraftProvider,
)


# Clock是一个类型别名。
#
# 正式运行使用perf_counter；
# 单元测试可以注入FakeClock。
Clock = Callable[[], float]

# 指标计算时只能选择两个明确的实验组。
DiagnosisPromptGroup = Literal[
    "basic",
    "evidence_first",
]

class VersionedDiagnosisDraftProvider(
    DiagnosisDraftProvider,
    Protocol,
):
    """Prompt实验需要的带版本号草稿Provider。"""

    # DiagnosisDraftProvider已经规定了：
    #
    # async generate_draft(
    #     request,
    #     evidence,
    # ) -> DiagnosisLLMDraft
    #
    # 当前Protocol只额外增加版本读取能力，
    # 不需要重复声明generate_draft()。
    @property
    def prompt_version(self) -> str:
        """返回该Provider实际使用的Prompt版本。"""

        ...


async def run_diagnosis_prompt_variant(
    *,
    provider: VersionedDiagnosisDraftProvider,
    request: DiagnosisRequest,
    evidence: tuple[
        HybridRetrievedChunk,
        ...,
    ],

    # 依赖注入计时函数。
    #
    # 普通调用不传时使用真实高精度计时器；
    # 测试传FakeClock以获得固定耗时。
    clock: Clock = perf_counter,
) -> DiagnosisPromptRunResult:
    """运行一个Prompt变体并把结果转换为实验数据。"""

    # 先读取版本号，再开始计时。
    #
    # 如果Provider没有正确实现版本契约，
    # 应作为配置或编程错误直接抛出，
    # 不能记录成LLM运行失败。
    prompt_version = provider.prompt_version

    started_at = clock()

    try:
        draft = await provider.generate_draft(
            request=request,
            evidence=evidence,
        )
    except InvalidLLMResponseError as exc:
        # LLM返回了空内容、非法JSON，
        # 或结构化JSON没有通过Pydantic契约。
        elapsed_ms = (
            clock() - started_at
        ) * 1000.0

        return DiagnosisPromptRunResult(
            prompt_version=prompt_version,
            outcome="invalid_response",
            generation_ms=elapsed_ms,
            draft=None,

            # 项目异常已经把原始响应转换成
            # 不含完整LLM输入的安全摘要。
            error_detail=str(exc),
        )
    except LLMTimeoutError as exc:
        elapsed_ms = (
            clock() - started_at
        ) * 1000.0

        return DiagnosisPromptRunResult(
            prompt_version=prompt_version,
            outcome="timeout",
            generation_ms=elapsed_ms,
            draft=None,
            error_detail=str(exc),
        )
    except LLMUpstreamError as exc:
        elapsed_ms = (
            clock() - started_at
        ) * 1000.0

        return DiagnosisPromptRunResult(
            prompt_version=prompt_version,
            outcome="upstream_error",
            generation_ms=elapsed_ms,
            draft=None,
            error_detail=str(exc),
        )

    # 这里只捕获了三类已经预期的外部失败。
    #
    # TypeError、ValueError、AttributeError和RuntimeError
    # 等异常会继续向上传播，因为它们更可能代表：
    #
    # 1. 程序错误；
    # 2. 错误的实验数据；
    # 3. 不完整的Provider实现。
    #
    # 不能把这些错误伪装成“Prompt表现不好”。

    elapsed_ms = (
        clock() - started_at
    ) * 1000.0

    return DiagnosisPromptRunResult(
        prompt_version=prompt_version,
        outcome="success",
        generation_ms=elapsed_ms,
        draft=draft,
        error_detail=None,
    )


async def compare_diagnosis_prompt_case(
    *,
    case_id: str,
    request: DiagnosisRequest,
    evidence: tuple[
        HybridRetrievedChunk,
        ...,
    ],
    basic_provider: (
        VersionedDiagnosisDraftProvider
    ),
    evidence_first_provider: (
        VersionedDiagnosisDraftProvider
    ),
    clock: Clock = perf_counter,
) -> DiagnosisPromptComparisonCase:
    """对同一请求和证据运行两个Prompt版本。"""

    # 在产生任何付费LLM调用前检查版本。
    #
    # 如果两个Provider版本相同，
    # 这不是一次有效的对照实验。
    if (
        basic_provider.prompt_version
        == evidence_first_provider.prompt_version
    ):
        raise ValueError(
            "两个Provider必须使用不同Prompt版本"
        )

    # 两组顺序执行，避免同时请求时相互争抢：
    #
    # 1. 上游并发配额；
    # 2. 本地连接池；
    # 3. 服务端吞吐资源。
    #
    # generation_ms只作为描述信息，
    # 不把单次耗时差异解释成Prompt质量差异。
    basic_result = (
        await run_diagnosis_prompt_variant(
            provider=basic_provider,
            request=request,
            evidence=evidence,
            clock=clock,
        )
    )

    evidence_first_result = (
        await run_diagnosis_prompt_variant(
            provider=evidence_first_provider,
            request=request,
            evidence=evidence,
            clock=clock,
        )
    )

    # 两次调用收到的是完全相同的request和evidence对象。
    #
    # 到了报告边界再将tuple转换为list，
    # 便于Pydantic序列化成JSON数组。
    return DiagnosisPromptComparisonCase(
        case_id=case_id,
        request=request,
        evidence=list(evidence),
        basic_result=basic_result,
        evidence_first_result=(
            evidence_first_result
        ),
    )


def collect_diagnosis_evidence_ids(
    draft: DiagnosisLLMDraft,
) -> tuple[str, ...]:
    """按首次出现顺序收集草稿引用的证据编号。"""

    # 类型注解不会自动进行运行时检查。
    if not isinstance(
        draft,
        DiagnosisLLMDraft,
    ):
        raise TypeError(
            "draft必须是DiagnosisLLMDraft"
        )

    ordered_ids: list[str] = []
    seen_ids: set[str] = set()

    # possible_causes和next_checks中的元素
    # 都具有evidence_ids字段。
    evidence_bearing_items = [
        *draft.possible_causes,
        *draft.next_checks,
    ]

    for item in evidence_bearing_items:
        for evidence_id in item.evidence_ids:
            # 同一个编号可能同时支持原因和检查，
            # 指标中只计一次。
            if evidence_id in seen_ids:
                continue

            seen_ids.add(evidence_id)
            ordered_ids.append(evidence_id)

    # 返回tuple表示这是完成收集后的不可变快照。
    return tuple(ordered_ids)


def calculate_diagnosis_prompt_metrics(
    *,
    cases: list[
        DiagnosisPromptComparisonCase
    ],
    group: DiagnosisPromptGroup,
) -> DiagnosisPromptVariantMetrics:
    """计算一个Prompt实验组在全部案例上的指标。"""

    if not cases:
        raise ValueError(
            "cases不能为空"
        )

    # Literal只是静态类型约束，
    # 所以这里仍建立运行时边界。
    if group not in (
        "basic",
        "evidence_first",
    ):
        raise ValueError(
            "group必须是basic或evidence_first"
        )

    results: list[
        DiagnosisPromptRunResult
    ]

    if group == "basic":
        results = [
            case.basic_result
            for case in cases
        ]
    else:
        results = [
            case.evidence_first_result
            for case in cases
        ]

    prompt_versions = {
        result.prompt_version
        for result in results
    }

    # 一组实验中途切换Prompt版本后，
    # 汇总指标将无法对应一个确定的Prompt。
    if len(prompt_versions) != 1:
        raise ValueError(
            "同一实验组必须使用同一个Prompt版本"
        )

    # set中已经确认只有一个元素。
    prompt_version = next(
        iter(prompt_versions)
    )

    success_count = sum(
        1
        for result in results
        if result.outcome == "success"
    )
    invalid_response_count = sum(
        1
        for result in results
        if result.outcome
        == "invalid_response"
    )
    timeout_count = sum(
        1
        for result in results
        if result.outcome == "timeout"
    )
    upstream_error_count = sum(
        1
        for result in results
        if result.outcome
        == "upstream_error"
    )

    abstained_count = 0
    whitelist_passed_count = 0
    unknown_evidence_id_count = 0

    # strict=True要求cases和results长度完全相等。
    #
    # 如果长度意外不同，zip不会静默截断，
    # 而是抛出ValueError。
    for case, result in zip(
        cases,
        results,
        strict=True,
    ):
        if result.outcome != "success":
            # 结构化草稿没有成功生成，
            # 因而不能获得白名单通过计数。
            continue

        draft = result.draft

        # DiagnosisPromptRunResult已经保证：
        #
        # outcome=success时draft一定存在。
        #
        # 这里仍显式检查，避免未来Schema变化后
        # 指标函数悄悄出现错误结果。
        if draft is None:
            raise ValueError(
                "success结果缺少draft"
            )

        if draft.abstained:
            abstained_count += 1

        referenced_ids = set(
            collect_diagnosis_evidence_ids(
                draft
            )
        )

        # Prompt Builder根据候选rank建立：
        #
        # 第1条候选 -> E1
        # 第2条候选 -> E2
        # ...
        #
        # 因此白名单只由本案例证据数量决定，
        # 不能相信LLM自己声明允许哪些编号。
        allowed_ids = {
            f"E{index}"
            for index in range(
                1,
                len(case.evidence) + 1,
            )
        }

        unknown_ids = (
            referenced_ids - allowed_ids
        )

        # 同一案例中的重复E999已经通过set去重。
        unknown_evidence_id_count += len(
            unknown_ids
        )

        if not unknown_ids:
            whitelist_passed_count += 1

    case_count = len(cases)

    average_generation_ms = (
        sum(
            result.generation_ms
            for result in results
        )
        / case_count
    )

    # Pydantic会再次检查：
    #
    # 1. 四种结果计数之和；
    # 2. abstained和白名单计数上限；
    # 3. 两个比率是否与整数计数一致。
    return DiagnosisPromptVariantMetrics(
        prompt_version=prompt_version,
        case_count=case_count,
        success_count=success_count,
        invalid_response_count=(
            invalid_response_count
        ),
        timeout_count=timeout_count,
        upstream_error_count=(
            upstream_error_count
        ),
        abstained_count=abstained_count,
        whitelist_passed_count=(
            whitelist_passed_count
        ),
        unknown_evidence_id_count=(
            unknown_evidence_id_count
        ),
        schema_success_rate=(
            success_count / case_count
        ),
        evidence_whitelist_pass_rate=(
            whitelist_passed_count
            / case_count
        ),
        average_generation_ms=(
            average_generation_ms
        ),
    )


def build_diagnosis_prompt_comparison_report(
    *,
    llm_model: str,
    temperature: float,
    cases: list[
        DiagnosisPromptComparisonCase
    ],
) -> DiagnosisPromptComparisonReport:
    """计算两组指标并组装完整Prompt对照报告。"""

    # 分别从同一份cases计算两组指标。
    basic_metrics = (
        calculate_diagnosis_prompt_metrics(
            cases=cases,
            group="basic",
        )
    )
    evidence_first_metrics = (
        calculate_diagnosis_prompt_metrics(
            cases=cases,
            group="evidence_first",
        )
    )

    # list(cases)建立新的列表容器，
    # 避免调用方之后append()直接改变报告的案例集合。
    return DiagnosisPromptComparisonReport(
        llm_model=llm_model,
        temperature=temperature,
        basic_metrics=basic_metrics,
        evidence_first_metrics=(
            evidence_first_metrics
        ),
        cases=list(cases),
    )
