"""Week 6 多模态 Agent 批量离线回归服务。

本模块负责依次执行 30 条已经配对的 Gold 场景和 Fixture，
复用第五周评分器生成逐场景评分，再组装第六周离线回归报告。

离线执行只替换外部依赖：

- Planner 使用固定脚本；
- Vision 使用固定观察结果；
- 知识库、遥测和测试草案工具使用固定结果。

安全分类、工具策略、AgentRunner、进度状态机、证据白名单、
诊断报告构造和正式评分规则仍然使用生产代码。
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from time import perf_counter
from typing import Protocol, runtime_checkable

from app.schemas.multimodal_agent_evaluation import (
    MultimodalAgentEvaluationScenario,
    MultimodalAgentTrace,
    MultimodalEvaluationFixRecord,
)
from app.schemas.multimodal_offline_regression import (
    MultimodalOfflineScenarioExecution,
    MultimodalOfflineScenarioFixture,
)
from app.schemas.multimodal_offline_regression_report import (
    MULTIMODAL_OFFLINE_SCENARIO_COUNT,
    WEEK5_BASELINE_PASSED_SCENARIO_COUNT,
    WEEK5_BASELINE_SAFE_REFUSAL_RATE,
    WEEK6_MINIMUM_PASSED_SCENARIO_COUNT,
    MultimodalOfflineRegressionReport,
    OfflineRegressionThresholdSummary,
)
from app.services.agent_evaluation import (
    build_multimodal_agent_evaluation_report,
    build_multimodal_agent_trace,
    evaluate_multimodal_agent_response,
)
from app.services.evaluation import (
    EvaluationDataError,
)
from app.services.multimodal_offline_fixture_loader import (
    MultimodalOfflineRegressionCase,
)


# 离线评测不会访问这个地址。
#
# 现有第五周报告契约要求保存一个 HTTP URL，
# 因此使用专门保留为无效域名的 .invalid，
# 明确表示本轮没有发送 HTTP 请求。
OFFLINE_API_URL = (
    "http://offline.invalid/"
    "api/v1/agent/diagnose"
)


# 报告中明确记录替代外部依赖的 Fake 名称，
# 避免离线报告被误认为真实模型评测报告。
OFFLINE_PLANNER_MODEL = "scripted-offline-planner"
OFFLINE_EMBEDDING_MODEL = "offline-fixture"
OFFLINE_COLLECTION_NAME = "offline-fixture"
OFFLINE_VISION_MODEL = "fake-vision-provider"


# Prompt 本身仍由生产 Agent 和 Vision 主链使用，
# 因此保留对应的真实版本标识。
OFFLINE_PLANNER_PROMPT_VERSION = (
    "agent-tool-calling-v2"
)
OFFLINE_VISION_PROMPT_VERSION = (
    "robot-vision-observation-v1"
)


# 每条场景都生成一份脱敏轨迹。
#
# PurePosixPath 强制报告中使用正斜杠，
# 避免 Windows 反斜杠进入跨平台 JSON 报告。
OFFLINE_TRACE_DIRECTORY = PurePosixPath(
    "docs/traces/multimodal-offline"
)


# 30 条场景必须与第五周 Gold 的编号空间完全一致。
#
# 只检查数量不够，因为 001～029 加上 031 也有 30 条，
# 但它已经不是计划要求的同一批场景。
EXPECTED_MULTIMODAL_SCENARIO_IDS = tuple(
    f"multimodal-agent-{index:03d}"
    for index in range(
        1,
        MULTIMODAL_OFFLINE_SCENARIO_COUNT + 1,
    )
)


# 这些记录解释第五周到第六周的核心可靠性修正。
#
# 它们属于报告审计信息，不参与 passed 评分。
OFFLINE_FIX_HISTORY = (
    MultimodalEvaluationFixRecord(
        issue=(
            "普通请求默认暴露过多工具，"
            "Planner可能选择无关工具"
        ),
        change=(
            "增加请求级最小工具策略，"
            "只向Planner和Executor暴露必要工具"
        ),
        verification=(
            "30条离线场景检查实际工具序列"
            "是否满足Gold工具范围"
        ),
    ),
    MultimodalEvaluationFixRecord(
        issue=(
            "重复查询或无新证据查询可能"
            "造成无意义工具循环"
        ),
        change=(
            "使用AgentProgress和ProgressReducer"
            "记录证据覆盖与下一步允许动作"
        ),
        verification=(
            "离线执行审计记录Planner轮次、"
            "工具调用和Fixture消费状态"
        ),
    ),
    MultimodalEvaluationFixRecord(
        issue=(
            "高风险请求的停止语义可能依赖"
            "Planner自由判断"
        ),
        change=(
            "在Planner前增加确定性安全分类和"
            "request_policy_finished终止路径"
        ),
        verification=(
            "安全场景检查拒答状态、终止原因"
            "以及零工具调用"
        ),
    ),
)


@runtime_checkable
class OfflineScenarioExecutor(Protocol):
    """批量服务所依赖的单场景执行器协议。

    生产实现是 MultimodalOfflineScenarioExecutor。

    这里使用 Protocol，而不直接绑定具体类，
    便于批量服务的单元测试注入轻量 Fake Executor。
    只要对象提供相同的异步 execute() 方法，
    就满足这个结构化接口。
    """

    async def execute(
        self,
        *,
        scenario: MultimodalAgentEvaluationScenario,
        fixture: MultimodalOfflineScenarioFixture,
    ) -> MultimodalOfflineScenarioExecution:
        """执行一个 Gold + Fixture 配对案例。"""

        ...


def _build_trace_path(
    scenario_id: str,
) -> str:
    """为一条场景生成稳定的项目相对轨迹路径。"""

    return (
        OFFLINE_TRACE_DIRECTORY
        / f"{scenario_id}.json"
    ).as_posix()


def _validate_cases(
    cases: Sequence[
        MultimodalOfflineRegressionCase
    ],
) -> tuple[
    MultimodalOfflineRegressionCase,
    ...,
]:
    """校验并冻结本轮批量回归案例。

    返回 tuple 是为了防止调用方在回归执行期间
    修改原始 list，导致执行顺序和报告顺序不一致。
    """

    # 字符串也属于某些广义序列语义，
    # 但显然不能被当成场景集合。
    if isinstance(
        cases,
        (str, bytes, bytearray),
    ) or not isinstance(cases, Sequence):
        raise TypeError(
            "cases必须是离线回归案例序列"
        )

    case_items = tuple(cases)

    if len(case_items) != (
        MULTIMODAL_OFFLINE_SCENARIO_COUNT
    ):
        raise EvaluationDataError(
            "多模态离线回归必须包含"
            f"{MULTIMODAL_OFFLINE_SCENARIO_COUNT}"
            "条场景"
        )

    for index, case in enumerate(case_items):
        if not isinstance(
            case,
            MultimodalOfflineRegressionCase,
        ):
            raise TypeError(
                f"cases[{index}]必须是"
                "MultimodalOfflineRegressionCase"
            )

    scenario_ids = tuple(
        case.scenario_id
        for case in case_items
    )

    if scenario_ids != (
        EXPECTED_MULTIMODAL_SCENARIO_IDS
    ):
        raise EvaluationDataError(
            "多模态离线回归场景必须按照"
            "multimodal-agent-001至030排列"
        )

    return case_items


def _build_threshold_summary(
    *,
    current_passed_scenario_count: int,
    current_safe_refusal_rate: float,
) -> OfflineRegressionThresholdSummary:
    """根据正式评分结果计算 Week 6 门槛摘要。

    调用方只提供本轮真实结果。
    基线、最低通过数和比较规则来自固定周计划。
    """

    current_scenario_pass_rate = (
        current_passed_scenario_count
        / MULTIMODAL_OFFLINE_SCENARIO_COUNT
    )

    passed_scenario_improvement = (
        current_passed_scenario_count
        - WEEK5_BASELINE_PASSED_SCENARIO_COUNT
    )

    pass_threshold_met = (
        current_passed_scenario_count
        >= WEEK6_MINIMUM_PASSED_SCENARIO_COUNT
    )

    safety_regression_free = (
        current_safe_refusal_rate
        >= WEEK5_BASELINE_SAFE_REFUSAL_RATE
    )

    # Pydantic还会重新计算并核对以上派生值，
    # 防止本函数将错误数字写入报告。
    return OfflineRegressionThresholdSummary(
        current_passed_scenario_count=(
            current_passed_scenario_count
        ),
        current_scenario_pass_rate=(
            current_scenario_pass_rate
        ),
        passed_scenario_improvement=(
            passed_scenario_improvement
        ),
        pass_threshold_met=pass_threshold_met,
        current_safe_refusal_rate=(
            current_safe_refusal_rate
        ),
        safety_regression_free=(
            safety_regression_free
        ),
    )


@dataclass(
    frozen=True,
    slots=True,
)
class MultimodalOfflineRegressionRun:
    """批量离线回归完成后的内部返回对象。

    report 是最终 JSON 和 Markdown 共同使用的正式报告；
    traces 是等待 CLI 写入 docs/traces 的完整脱敏轨迹。

    使用该容器可以避免批量服务直接写文件，
    保持“业务计算”和“文件输出”职责分离。
    """

    report: MultimodalOfflineRegressionReport
    traces: tuple[
        MultimodalAgentTrace,
        ...,
    ]

    def __post_init__(self) -> None:
        """保证报告结果、轨迹对象和轨迹路径一一对应。"""

        if not isinstance(
            self.report,
            MultimodalOfflineRegressionReport,
        ):
            raise TypeError(
                "report必须是"
                "MultimodalOfflineRegressionReport"
            )

        if not isinstance(self.traces, tuple):
            raise TypeError(
                "traces必须是tuple"
            )

        if not all(
            isinstance(trace, MultimodalAgentTrace)
            for trace in self.traces
        ):
            raise TypeError(
                "traces只能包含"
                "MultimodalAgentTrace"
            )

        result_ids = tuple(
            result.scenario_id
            for result
            in self.report.scored_evaluation.results
        )

        trace_ids = tuple(
            trace.scenario_id
            for trace in self.traces
        )

        if trace_ids != result_ids:
            raise ValueError(
                "traces必须与评分结果"
                "按场景编号逐项对应"
            )

        expected_trace_paths = tuple(
            _build_trace_path(scenario_id)
            for scenario_id in trace_ids
        )

        if (
            self.report
            .scored_evaluation
            .trace_paths
            != expected_trace_paths
        ):
            raise ValueError(
                "报告中的trace_paths必须与"
                "实际轨迹对象逐项对应"
            )


class MultimodalOfflineRegressionService:
    """执行整批 30 场景多模态离线回归。

    一个 Service 实例持有一个单场景执行器。
    run() 每次接收完整的配对案例并返回正式报告和轨迹，
    不读取 JSONL，也不向磁盘写文件。
    """

    def __init__(
        self,
        *,
        executor: OfflineScenarioExecutor,
    ) -> None:
        """保存满足协议的单场景执行器。"""

        if not isinstance(
            executor,
            OfflineScenarioExecutor,
        ):
            raise TypeError(
                "executor必须提供异步execute方法"
            )

        self._executor = executor

    async def run(
        self,
        *,
        cases: Sequence[
            MultimodalOfflineRegressionCase
        ],
        scenario_source: str,
        fixture_source: str,
    ) -> MultimodalOfflineRegressionRun:
        """执行、评分并汇总 30 条离线回归案例。

        场景按 Gold 顺序串行执行。串行方式比并发更适合
        确定性回归，因为出现失败时可以直接确定是第几条，
        而且不会让日志、Fake消费记录和轨迹顺序交错。
        """

        case_items = _validate_cases(cases)

        executions: list[
            MultimodalOfflineScenarioExecution
        ] = []

        evaluations = []
        traces: list[MultimodalAgentTrace] = []
        trace_paths: list[str] = []

        for case in case_items:
            # perf_counter()是单调高精度计时器，
            # 适合测量代码经过的时间，不受系统时钟调整影响。
            started_at = perf_counter()

            execution = await self._executor.execute(
                scenario=case.scenario,
                fixture=case.fixture,
            )

            latency_ms = max(
                (
                    perf_counter()
                    - started_at
                )
                * 1000.0,
                0.0,
            )

            # 复用第五周正式评分器。
            #
            # 虽然这里没有HTTP调用，但单场景执行已经成功返回
            # AgentDiagnosisResponse，因此按合法200响应评分。
            evaluation = (
                evaluate_multimodal_agent_response(
                    scenario=case.scenario,
                    response=execution.response,
                    http_status_code=200,
                    latency_ms=latency_ms,
                    cost_status="unavailable",
                    estimated_cost_usd=None,
                    cost_note=(
                        "离线回归使用固定Fake结果，"
                        "没有调用可计费外部模型"
                    ),
                    llm_judge_used=False,
                    llm_judge_passed=None,
                    llm_judge_note=None,
                )
            )

            # 轨迹构造器只保留公开、脱敏字段，
            # 不保存图片正文、图片路径、密钥或思维链。
            trace = build_multimodal_agent_trace(
                scenario=case.scenario,
                response=execution.response,
                evaluation=evaluation,
            )

            executions.append(execution)
            evaluations.append(evaluation)
            traces.append(trace)
            trace_paths.append(
                _build_trace_path(
                    case.scenario_id
                )
            )

        # 正式报告组装器会从30条结果重新计算：
        #
        # - 通过数和通过率；
        # - 工具选择正确率；
        # - 任务完成率；
        # - 引用正确率和覆盖率；
        # - 安全拒答率；
        # - Vision、来源、延迟等指标。
        scored_evaluation = (
            build_multimodal_agent_evaluation_report(
                results=tuple(evaluations),
                scenario_source=scenario_source,
                api_url=OFFLINE_API_URL,
                planner_model=(
                    OFFLINE_PLANNER_MODEL
                ),
                planner_prompt_version=(
                    OFFLINE_PLANNER_PROMPT_VERSION
                ),
                embedding_model=(
                    OFFLINE_EMBEDDING_MODEL
                ),
                collection_name=(
                    OFFLINE_COLLECTION_NAME
                ),
                vision_model=OFFLINE_VISION_MODEL,
                vision_prompt_version=(
                    OFFLINE_VISION_PROMPT_VERSION
                ),
                trace_paths=tuple(trace_paths),
                fix_history=OFFLINE_FIX_HISTORY,
            )
        )

        strict_passed_scenario_ids = tuple(
            result.scenario_id
            for result in scored_evaluation.results
            if result.passed
        )

        strict_failed_scenario_ids = tuple(
            result.scenario_id
            for result in scored_evaluation.results
            if not result.passed
        )

        fixture_fully_consumed_count = sum(
            execution.fixture_fully_consumed
            for execution in executions
        )

        unconsumed_fixture_scenario_ids = tuple(
            execution.scenario_id
            for execution in executions
            if not execution.fixture_fully_consumed
        )

        threshold_summary = (
            _build_threshold_summary(
                current_passed_scenario_count=(
                    scored_evaluation
                    .metrics
                    .passed_scenario_count
                ),
                current_safe_refusal_rate=(
                    scored_evaluation
                    .metrics
                    .safe_refusal_rate
                ),
            )
        )

        report = MultimodalOfflineRegressionReport(
            scenario_source=scenario_source,
            fixture_source=fixture_source,
            scored_evaluation=scored_evaluation,
            execution_audits=tuple(executions),
            strict_passed_scenario_ids=(
                strict_passed_scenario_ids
            ),
            strict_failed_scenario_ids=(
                strict_failed_scenario_ids
            ),
            fixture_fully_consumed_count=(
                fixture_fully_consumed_count
            ),
            unconsumed_fixture_scenario_ids=(
                unconsumed_fixture_scenario_ids
            ),
            threshold_summary=threshold_summary,
        )

        return MultimodalOfflineRegressionRun(
            report=report,
            traces=tuple(traces),
        )
