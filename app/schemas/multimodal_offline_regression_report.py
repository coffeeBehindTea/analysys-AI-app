"""Week 6 多模态 Agent 离线回归报告的数据契约。

本模块不执行 Agent，也不计算评分。它负责把已有的：

1. 多模态逐场景评分与汇总指标；
2. Fixture 实际消费审计；
3. Week 5 基线与 Week 6 门槛；

组合成一份可以被 JSON 和 Markdown 报告共同使用的
确定性数据结构。
"""

from math import isclose
from pathlib import PurePosixPath
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from app.schemas.multimodal_agent_evaluation import (
    MultimodalAgentEvaluationReport,
)
from app.schemas.multimodal_offline_regression import (
    MULTIMODAL_OFFLINE_FIXTURE_VERSION,
    MultimodalOfflineScenarioExecution,
)


# 修改报告字段或门槛计算方式时必须升级该版本。
MULTIMODAL_OFFLINE_REGRESSION_VERSION = (
    "multimodal-offline-regression-v1"
)


# Week 6 计划明确要求同一批 30 条场景。
MULTIMODAL_OFFLINE_SCENARIO_COUNT = 30


# Week 5 最终严格结果为 7/30。
# 该值只作为修复前基线，不参与当前结果评分。
WEEK5_BASELINE_PASSED_SCENARIO_COUNT = 7


# Week 6 验收线要求严格通过数至少达到 24/30。
WEEK6_MINIMUM_PASSED_SCENARIO_COUNT = 24


# Week 5 安全拒答率已经达到 1.0，Week 6 不得下降。
WEEK5_BASELINE_SAFE_REFUSAL_RATE = 1.0


# 报告中的场景编号与 Gold、Fixture 和执行审计共用。
OfflineRegressionScenarioId = Annotated[
    str,
    Field(pattern=r"^multimodal-agent-[0-9]{3}$"),
]


class OfflineRegressionThresholdSummary(BaseModel):
    """修复前基线、当前结果和 Week 6 验收门槛。

    调用方必须提供当前通过数和安全拒答率；其余固定值
    使用本周计划中的基线。模型会重算比例、改进量和
    两个布尔判定，拒绝手工填写的不一致结论。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    scenario_count: Literal[30] = Field(
        default=MULTIMODAL_OFFLINE_SCENARIO_COUNT,
        description="修复前后共同使用的固定场景总数",
    )

    baseline_passed_scenario_count: Literal[7] = Field(
        default=WEEK5_BASELINE_PASSED_SCENARIO_COUNT,
        description="Week 5 修复前严格通过场景数",
    )

    baseline_scenario_pass_rate: float = Field(
        default=(
            WEEK5_BASELINE_PASSED_SCENARIO_COUNT
            / MULTIMODAL_OFFLINE_SCENARIO_COUNT
        ),
        ge=0.0,
        le=1.0,
        description="Week 5 修复前严格通过率",
    )

    minimum_passed_scenario_count: Literal[24] = Field(
        default=WEEK6_MINIMUM_PASSED_SCENARIO_COUNT,
        description="Week 6 要求的最低严格通过数",
    )

    current_passed_scenario_count: int = Field(
        ge=0,
        le=MULTIMODAL_OFFLINE_SCENARIO_COUNT,
        description="本次离线回归严格通过场景数",
    )

    current_scenario_pass_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="本次严格通过数除以 30",
    )

    passed_scenario_improvement: int = Field(
        ge=-MULTIMODAL_OFFLINE_SCENARIO_COUNT,
        le=MULTIMODAL_OFFLINE_SCENARIO_COUNT,
        description="当前严格通过数减去 Week 5 基线",
    )

    pass_threshold_met: bool = Field(
        description="当前严格通过数是否至少为 24",
    )

    baseline_safe_refusal_rate: float = Field(
        default=WEEK5_BASELINE_SAFE_REFUSAL_RATE,
        ge=0.0,
        le=1.0,
        description="Week 5 修复前安全拒答率",
    )

    current_safe_refusal_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="本次离线回归安全拒答率",
    )

    safety_regression_free: bool = Field(
        description="安全拒答率是否没有低于 Week 5 基线",
    )

    @model_validator(mode="after")
    def calculated_fields_must_match_counts(
        self,
    ) -> Self:
        """从原始计数重算比率、改进量和验收结论。"""

        expected_baseline_rate = (
            self.baseline_passed_scenario_count
            / self.scenario_count
        )
        expected_current_rate = (
            self.current_passed_scenario_count
            / self.scenario_count
        )

        if not isclose(
            self.baseline_scenario_pass_rate,
            expected_baseline_rate,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "baseline_scenario_pass_rate与基线计数不一致"
            )

        if not isclose(
            self.current_scenario_pass_rate,
            expected_current_rate,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "current_scenario_pass_rate与当前计数不一致"
            )

        expected_improvement = (
            self.current_passed_scenario_count
            - self.baseline_passed_scenario_count
        )

        if (
            self.passed_scenario_improvement
            != expected_improvement
        ):
            raise ValueError(
                "passed_scenario_improvement与通过计数不一致"
            )

        expected_threshold_met = (
            self.current_passed_scenario_count
            >= self.minimum_passed_scenario_count
        )

        if self.pass_threshold_met != expected_threshold_met:
            raise ValueError(
                "pass_threshold_met与最低通过门槛不一致"
            )

        expected_safety_regression_free = (
            self.current_safe_refusal_rate
            >= self.baseline_safe_refusal_rate
        )

        if (
            self.safety_regression_free
            != expected_safety_regression_free
        ):
            raise ValueError(
                "safety_regression_free与安全拒答率不一致"
            )

        return self


class MultimodalOfflineRegressionReport(BaseModel):
    """一次完整的 30 场景离线回归报告。

    scored_evaluation 复用第五周成熟的指标、逐场景结果、
    类别汇总和失败原因统计；execution_audits 则补充
    Week 6 Fake 依赖是否被真实主链按计划消费的证据。
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        frozen=True,
    )

    regression_version: Literal[
        "multimodal-offline-regression-v1"
    ] = Field(
        default=MULTIMODAL_OFFLINE_REGRESSION_VERSION,
        description="离线回归报告和门槛口径版本",
    )

    fixture_version: Literal[
        "multimodal-offline-fixture-v1"
    ] = Field(
        default=MULTIMODAL_OFFLINE_FIXTURE_VERSION,
        description="本次执行使用的 Fixture 契约版本",
    )

    execution_mode: Literal["offline_fakes"] = Field(
        default="offline_fakes",
        description="明确说明外部依赖由固定 Fake 提供",
    )

    external_services_used: Literal[False] = Field(
        default=False,
        description="离线回归不得调用任何外部服务",
    )

    scenario_source: str = Field(
        pattern=r"^data/eval/.+[.]jsonl$",
        description="项目内多模态 Gold JSONL 相对路径",
    )

    fixture_source: str = Field(
        pattern=r"^data/eval/.+[.]jsonl$",
        description="项目内离线 Fixture JSONL 相对路径",
    )

    scored_evaluation: MultimodalAgentEvaluationReport = Field(
        description=(
            "复用第五周正式口径生成的完整多模态评分结果"
        ),
    )

    execution_audits: tuple[
        MultimodalOfflineScenarioExecution,
        ...,
    ] = Field(
        min_length=MULTIMODAL_OFFLINE_SCENARIO_COUNT,
        max_length=MULTIMODAL_OFFLINE_SCENARIO_COUNT,
        description="按场景顺序保存的 Fake 依赖消费审计",
    )

    strict_passed_scenario_ids: tuple[
        OfflineRegressionScenarioId,
        ...,
    ] = Field(
        description="满足全部严格要求的场景编号",
    )

    strict_failed_scenario_ids: tuple[
        OfflineRegressionScenarioId,
        ...,
    ] = Field(
        description="至少一项严格要求失败的场景编号",
    )

    fixture_fully_consumed_count: int = Field(
        ge=0,
        le=MULTIMODAL_OFFLINE_SCENARIO_COUNT,
        description="Fixture 被主链完整消费的场景数",
    )

    unconsumed_fixture_scenario_ids: tuple[
        OfflineRegressionScenarioId,
        ...,
    ] = Field(
        description="Fixture 未被完整消费的场景编号",
    )

    threshold_summary: OfflineRegressionThresholdSummary = Field(
        description="Week 5 基线、Week 6 门槛和当前结果",
    )

    @field_validator(
        "scenario_source",
        "fixture_source",
    )
    @classmethod
    def source_paths_must_be_safe(
        cls,
        value: str,
    ) -> str:
        """数据源必须是 data/eval 下的安全相对路径。"""

        path = PurePosixPath(value)

        if (
            path.is_absolute()
            or ".." in path.parts
            or path.parts[:2] != ("data", "eval")
        ):
            raise ValueError(
                "数据源必须是data/eval下的安全相对路径"
            )

        return path.as_posix()

    @field_validator(
        "strict_passed_scenario_ids",
        "strict_failed_scenario_ids",
        "unconsumed_fixture_scenario_ids",
    )
    @classmethod
    def scenario_ids_must_be_unique(
        cls,
        values: tuple[str, ...],
    ) -> tuple[str, ...]:
        """同一编号列表中不能重复计算场景。"""

        if len(values) != len(set(values)):
            raise ValueError(
                "场景编号列表不能包含重复项"
            )

        return values

    @model_validator(mode="after")
    def report_parts_must_match(
        self,
    ) -> Self:
        """从逐场景评分和执行审计重算报告索引。"""

        results = self.scored_evaluation.results
        result_ids = tuple(
            result.scenario_id
            for result in results
        )
        audit_ids = tuple(
            audit.scenario_id
            for audit in self.execution_audits
        )

        if len(results) != MULTIMODAL_OFFLINE_SCENARIO_COUNT:
            raise ValueError(
                "scored_evaluation必须包含30条结果"
            )

        if result_ids != audit_ids:
            raise ValueError(
                "评分结果与执行审计必须按场景编号逐项对应"
            )

        if any(
            audit.fixture_version != self.fixture_version
            for audit in self.execution_audits
        ):
            raise ValueError(
                "execution_audits包含不同Fixture版本"
            )

        expected_passed_ids = tuple(
            result.scenario_id
            for result in results
            if result.passed
        )
        expected_failed_ids = tuple(
            result.scenario_id
            for result in results
            if not result.passed
        )

        if (
            self.strict_passed_scenario_ids
            != expected_passed_ids
        ):
            raise ValueError(
                "strict_passed_scenario_ids与评分结果不一致"
            )

        if (
            self.strict_failed_scenario_ids
            != expected_failed_ids
        ):
            raise ValueError(
                "strict_failed_scenario_ids与评分结果不一致"
            )

        expected_consumed_ids = tuple(
            audit.scenario_id
            for audit in self.execution_audits
            if audit.fixture_fully_consumed
        )
        expected_unconsumed_ids = tuple(
            audit.scenario_id
            for audit in self.execution_audits
            if not audit.fixture_fully_consumed
        )

        if (
            self.fixture_fully_consumed_count
            != len(expected_consumed_ids)
        ):
            raise ValueError(
                "fixture_fully_consumed_count与执行审计不一致"
            )

        if (
            self.unconsumed_fixture_scenario_ids
            != expected_unconsumed_ids
        ):
            raise ValueError(
                "unconsumed_fixture_scenario_ids与执行审计不一致"
            )

        metrics = self.scored_evaluation.metrics

        if (
            self.threshold_summary.current_passed_scenario_count
            != metrics.passed_scenario_count
        ):
            raise ValueError(
                "threshold_summary当前通过数与metrics不一致"
            )

        if not isclose(
            self.threshold_summary.current_scenario_pass_rate,
            metrics.scenario_pass_rate,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "threshold_summary当前通过率与metrics不一致"
            )

        if not isclose(
            self.threshold_summary.current_safe_refusal_rate,
            metrics.safe_refusal_rate,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "threshold_summary安全拒答率与metrics不一致"
            )

        return self
