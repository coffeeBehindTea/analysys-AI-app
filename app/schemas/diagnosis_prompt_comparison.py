"""诊断Prompt对照实验使用的数据契约。

本模块只定义并校验实验数据：

1. 一次Prompt调用是成功还是安全失败；
2. 同一案例的基础组和证据优先组结果；
3. 完整实验使用的模型、temperature和案例集合。

本模块不调用LLM、不执行检索，也不计算汇总指标。
"""

# isclose()用于比较根据整数计数算出的浮点比率。
#
# 浮点数采用二进制表示，
# 例如1 / 3不一定能被精确保存，
# 所以不直接使用==。
from math import isclose

# Literal把字符串限制在固定集合内。
#
# Self表示当前Pydantic模型自身，
# 用于model_validator的返回类型。
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from app.schemas.diagnosis_llm import (
    DiagnosisLLMDraft,
)
from app.schemas.diagnostics import (
    DiagnosisRequest,
)
from app.schemas.retrieval import (
    HybridRetrievedChunk,
)


# 一次Prompt调用只有以下四种结果。
#
# success：
# 成功取得并校验DiagnosisLLMDraft。
#
# invalid_response：
# LLM返回空内容、非法JSON或违反Schema的数据。
#
# timeout：
# 上游LLM请求超时。
#
# upstream_error：
# 网络连接、认证、限流或上游错误状态。
DiagnosisPromptRunOutcome = Literal[
    "success",
    "invalid_response",
    "timeout",
    "upstream_error",
]


class DiagnosisPromptRunResult(BaseModel):
    """一个Prompt变体在一道案例上的运行结果。"""

    model_config = ConfigDict(
        # 自动清除字符串首尾空白。
        str_strip_whitespace=True,

        # 拒绝Schema没有声明的字段，
        # 防止字段拼写错误被静默忽略。
        extra="forbid",

        # 耗时不能保存NaN或正负无穷。
        allow_inf_nan=False,
    )

    # 实际发送给LLM的Prompt版本。
    prompt_version: str = Field(
        min_length=1,
        max_length=200,
        description="本次运行使用的Prompt版本",
    )

    outcome: DiagnosisPromptRunOutcome

    # 从调用Provider到得到结果或异常的墙钟耗时。
    generation_ms: float = Field(
        ge=0.0,
        description="LLM生成阶段耗时，单位毫秒",
    )

    # 只有outcome=success时才能存在。
    #
    # 该对象已经通过DiagnosisLLMDraft的
    # 字段、状态关系和高风险规则校验。
    draft: DiagnosisLLMDraft | None = None

    # 只保存安全的内部错误摘要，
    # 不保存完整LLM原始输出。
    error_detail: str | None = Field(
        default=None,
        min_length=1,
        max_length=2_000,
    )

    @model_validator(mode="after")
    def validate_outcome_payload(
        self,
    ) -> Self:
        """检查运行状态、草稿和错误信息是否自洽。"""

        if self.outcome == "success":
            # 成功必须有一个真正通过Schema校验的草稿。
            if self.draft is None:
                raise ValueError(
                    "success结果必须包含draft"
                )

            # 成功结果不能同时宣称发生错误。
            if self.error_detail is not None:
                raise ValueError(
                    "success结果不能包含error_detail"
                )

            return self

        # 下面处理三种失败结果。
        #
        # 如果失败结果仍携带draft，
        # 使用方可能误把未经认可的结果当成有效诊断。
        if self.draft is not None:
            raise ValueError(
                "失败结果不能包含draft"
            )

        if self.error_detail is None:
            raise ValueError(
                "失败结果必须包含error_detail"
            )

        return self


class DiagnosisPromptComparisonCase(
    BaseModel
):
    """相同请求和证据上的两个Prompt运行结果。"""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        allow_inf_nan=False,
    )

    # dpc表示diagnosis prompt comparison。
    #
    # 固定编号便于在报告和失败分析中引用案例。
    case_id: str = Field(
        pattern=r"^dpc\d{3}$",
        description="Prompt对照案例稳定编号",
    )

    # 两个Prompt实验组共同使用同一个诊断请求。
    request: DiagnosisRequest

    # 两个实验组共同使用完全相同的已排序证据。
    #
    # 报告保留完整候选而不只保存Chunk ID，
    # 这样可以审计模型当时真正看到了什么正文。
    evidence: list[
        HybridRetrievedChunk
    ] = Field(
        min_length=1,
        max_length=10,
    )

    # 控制组：基础结构化诊断Prompt。
    basic_result: DiagnosisPromptRunResult

    # 实验组：证据优先和安全约束Prompt。
    evidence_first_result: (
        DiagnosisPromptRunResult
    )

    @model_validator(mode="after")
    def validate_comparison_case(
        self,
    ) -> Self:
        """检查证据排名和两个实验组版本。"""

        # 同一个版本不能同时冒充控制组和实验组。
        if (
            self.basic_result.prompt_version
            == self.evidence_first_result.prompt_version
        ):
            raise ValueError(
                "两个实验组必须使用不同Prompt版本"
            )

        # 字段名称本身代表实验角色，
        # 因而版本前缀也必须与角色一致。
        if not (
            self.basic_result.prompt_version
            .startswith("diagnosis-basic-")
        ):
            raise ValueError(
                "basic_result必须使用基础Prompt版本"
            )

        if not (
            self.evidence_first_result.prompt_version
            .startswith("diagnosis-evidence-")
        ):
            raise ValueError(
                "evidence_first_result必须使用"
                "证据优先Prompt版本"
            )

        # 证据必须按最终排名1、2、3连续排列。
        #
        # 如果顺序混乱，E1、E2与真实候选的关系
        # 就不能被可靠复现。
        actual_ranks = [
            candidate.rank
            for candidate in self.evidence
        ]
        expected_ranks = list(
            range(
                1,
                len(self.evidence) + 1,
            )
        )

        if actual_ranks != expected_ranks:
            raise ValueError(
                "evidence必须按连续rank排序"
            )

        chunk_ids = [
            candidate.chunk.chunk_id
            for candidate in self.evidence
        ]

        if len(set(chunk_ids)) != len(
            chunk_ids
        ):
            raise ValueError(
                "evidence不能包含重复chunk_id"
            )

        return self


class DiagnosisPromptVariantMetrics(
    BaseModel
):
    """一个Prompt版本在全部案例上的汇总指标。"""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        allow_inf_nan=False,
    )

    prompt_version: str = Field(
        min_length=1,
        max_length=200,
    )

    # 参与该Prompt版本评测的案例总数。
    case_count: int = Field(
        ge=1,
    )

    # 成功表示LLM结果通过了
    # DiagnosisLLMDraft结构和安全规则校验。
    success_count: int = Field(
        ge=0,
    )

    invalid_response_count: int = Field(
        ge=0,
    )

    timeout_count: int = Field(
        ge=0,
    )

    upstream_error_count: int = Field(
        ge=0,
    )

    # 在成功草稿中主动返回abstained的案例数。
    abstained_count: int = Field(
        ge=0,
    )

    # 成功草稿中，所有evidence_id都属于
    # 本次E1到En白名单的案例数。
    whitelist_passed_count: int = Field(
        ge=0,
    )

    # 所有案例中出现的未知证据编号数量。
    #
    # 同一案例内重复出现的E999只计一次，
    # 但不同案例分别出现时分别计数。
    unknown_evidence_id_count: int = Field(
        ge=0,
    )

    # success_count / case_count。
    schema_success_rate: float = Field(
        ge=0.0,
        le=1.0,
    )

    # whitelist_passed_count / case_count。
    #
    # 分母使用全部案例，而不是只使用成功案例。
    # 这样非法JSON或Schema失败不会从指标中消失。
    evidence_whitelist_pass_rate: float = (
        Field(
            ge=0.0,
            le=1.0,
        )
    )

    # 包含成功和失败调用的平均生成耗时。
    average_generation_ms: float = Field(
        ge=0.0,
    )

    @model_validator(mode="after")
    def validate_metrics(
        self,
    ) -> Self:
        """检查计数、比率和案例总数是否一致。"""

        outcome_count = (
            self.success_count
            + self.invalid_response_count
            + self.timeout_count
            + self.upstream_error_count
        )

        # 每个案例只能落入四种运行结果之一。
        if outcome_count != self.case_count:
            raise ValueError(
                "运行结果计数之和必须等于case_count"
            )

        # 只有通过Schema的草稿才可能拥有
        # 可解释的abstained字段。
        if (
            self.abstained_count
            > self.success_count
        ):
            raise ValueError(
                "abstained_count不能超过success_count"
            )

        # 只有成功解析的草稿才能执行证据编号检查。
        if (
            self.whitelist_passed_count
            > self.success_count
        ):
            raise ValueError(
                "whitelist_passed_count"
                "不能超过success_count"
            )

        expected_schema_success_rate = (
            self.success_count
            / self.case_count
        )

        if not isclose(
            self.schema_success_rate,
            expected_schema_success_rate,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "schema_success_rate"
                "与success_count不一致"
            )

        expected_whitelist_rate = (
            self.whitelist_passed_count
            / self.case_count
        )

        if not isclose(
            self.evidence_whitelist_pass_rate,
            expected_whitelist_rate,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "evidence_whitelist_pass_rate"
                "与whitelist_passed_count不一致"
            )

        return self


class DiagnosisPromptComparisonReport(
    BaseModel
):
    """一次基础Prompt与证据优先Prompt的完整报告。"""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        allow_inf_nan=False,
    )

    # 两组必须使用相同模型，
    # 所以模型名称只在报告顶层记录一次。
    llm_model: str = Field(
        min_length=1,
        max_length=500,
    )

    # 两组必须使用相同temperature。
    #
    # 当前项目使用0.0，但保留字段可以使报告
    # 明确说明真实实验参数。
    temperature: float = Field(
        ge=0.0,
        le=2.0,
    )

    # 基础Prompt控制组的汇总指标。
    #
    # 该对象记录基础组的成功、失败、
    # 白名单通过情况和平均耗时。
    basic_metrics: DiagnosisPromptVariantMetrics

    # 证据优先Prompt实验组的汇总指标。
    #
    # validate_report()会进一步检查：
    #
    # 1. case_count是否等于cases数量；
    # 2. prompt_version是否与逐案例结果一致。
    evidence_first_metrics: (
        DiagnosisPromptVariantMetrics
    )

    cases: list[
        DiagnosisPromptComparisonCase
    ] = Field(
        min_length=1,
    )

    @model_validator(mode="after")
    def validate_report(
        self,
    ) -> Self:
        """检查案例编号、指标数量和Prompt版本是否一致。"""

        case_ids = [
            case.case_id
            for case in self.cases
        ]

        # 重复案例会让成功率和其他汇总指标
        # 对同一输入重复计数。
        if len(set(case_ids)) != len(
            case_ids
        ):
            raise ValueError(
                "Prompt对照报告包含重复case_id"
            )

        case_count = len(self.cases)

        # 两组指标必须覆盖报告中的全部案例。
        if (
            self.basic_metrics.case_count
            != case_count
            or self.evidence_first_metrics.case_count
            != case_count
        ):
            raise ValueError(
                "指标case_count与cases数量不一致"
            )

        basic_versions = {
            case.basic_result.prompt_version
            for case in self.cases
        }

        evidence_first_versions = {
            case.evidence_first_result.prompt_version
            for case in self.cases
        }

        # 一组实验不能在不同案例中悄悄切换Prompt版本。
        if basic_versions != {
            self.basic_metrics.prompt_version
        }:
            raise ValueError(
                "基础Prompt指标版本"
                "与逐案例结果不一致"
            )

        if evidence_first_versions != {
            self.evidence_first_metrics.prompt_version
        }:
            raise ValueError(
                "证据优先Prompt指标版本"
                "与逐案例结果不一致"
            )

        return self
