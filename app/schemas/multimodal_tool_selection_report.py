"""多模态工具选择批量实验报告的数据契约。

本模块负责定义：

1. 一条处理路线在整批案例上的汇总指标；
2. 整次工具选择对照实验的结果结构；
3. 计数、比率、路线和单条结果之间的一致性；

本模块不负责：

1. 读取Gold案例；
2. 执行OCR、Vision或直接拒答路线；
3. 计算百分位数和平均值；
4. 调用外部模型；
5. 写入JSON或Markdown文件。

具体指标将在批量实验服务中计算，
本模块负责拒绝不一致的计算结果。
"""

# isclose用于比较浮点指标。
#
# 浮点数使用二进制保存，
# 例如2 / 3不一定能被精确表示，
# 因此不直接使用严格相等比较。
from math import (
    isclose,
)

# Annotated允许为基础类型附加Pydantic约束。
#
# Literal把字符串限制为固定版本。
#
# Self表示当前模型类型，
# 用于model_validator的返回注解。
from typing import (
    Annotated,
    Literal,
    Self,
)

# Pydantic负责数据契约和跨字段校验。
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# 复用统一路线名称。
from app.schemas.multimodal_tool_selection import (
    ToolSelectionRoute,
)

# 复用已经评分完成的单路线结果。
from app.schemas.multimodal_tool_selection_evaluation import (
    MultimodalToolSelectionRouteResult,
)


# 报告版本用于区分未来可能发生的字段变化。
MULTIMODAL_TOOL_SELECTION_REPORT_VERSION = (
    "multimodal-tool-selection-report-v1"
)


# 所有0到1之间的比率共用相同约束。
MetricRatio = Annotated[
    float,
    Field(
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    ),
]


class MultimodalToolSelectionRouteSummary(
    BaseModel
):
    """一条处理路线在整批案例上的汇总结果。

    本模型只保存统计结果，
    不保存完整图片、Base64或模型原始响应。
    """

    model_config = ConfigDict(
        # 清理普通字符串两端空白。
        str_strip_whitespace=True,

        # 拒绝未声明字段，
        # 防止指标字段拼写错误被静默忽略。
        extra="forbid",

        # 汇总记录创建后不能重新赋值。
        frozen=True,
    )

    route: ToolSelectionRoute = Field(
        description=(
            "本汇总对应的处理路线"
        ),
    )

    evaluated_case_count: int = Field(
        ge=1,
        description=(
            "这条路线实际完成评测的案例数量"
        ),
    )

    completed_count: int = Field(
        ge=0,
        description=(
            "execution_status为completed的数量"
        ),
    )

    partial_count: int = Field(
        ge=0,
        description=(
            "execution_status为partial的数量"
        ),
    )

    abstained_count: int = Field(
        ge=0,
        description=(
            "execution_status为abstained的数量"
        ),
    )

    failed_count: int = Field(
        ge=0,
        description=(
            "execution_status为failed的数量"
        ),
    )

    passed_count: int = Field(
        ge=0,
        description=(
            "路线适合、结果正确且安全的数量"
        ),
    )

    route_appropriate_count: int = Field(
        ge=0,
        description=(
            "路线包含在案例acceptable_routes中的数量"
        ),
    )

    outcome_correct_count: int = Field(
        ge=0,
        description=(
            "观察内容或拒答行为符合Gold的数量"
        ),
    )

    safety_passed_count: int = Field(
        ge=0,
        description=(
            "通过不可信文字和输出安全检查的数量"
        ),
    )

    content_accuracy_sample_count: int = Field(
        ge=0,
        description=(
            "具有可计算content_accuracy的结果数量；"
            "应拒答案例不进入该指标"
        ),
    )

    mean_content_accuracy: (
        MetricRatio
        | None
    ) = Field(
        default=None,
        description=(
            "具有内容标准项结果的平均内容准确率；"
            "没有可计算样本时为None"
        ),
    )

    pass_rate: MetricRatio = Field(
        description=(
            "passed_count除以evaluated_case_count"
        ),
    )

    route_appropriate_rate: MetricRatio = Field(
        description=(
            "route_appropriate_count除以"
            "evaluated_case_count"
        ),
    )

    outcome_correct_rate: MetricRatio = Field(
        description=(
            "outcome_correct_count除以"
            "evaluated_case_count"
        ),
    )

    safety_pass_rate: MetricRatio = Field(
        description=(
            "safety_passed_count除以"
            "evaluated_case_count"
        ),
    )

    mean_latency_ms: float = Field(
        ge=0.0,
        allow_inf_nan=False,
        description=(
            "该路线全部结果的平均端到端延迟"
        ),
    )

    p50_latency_ms: float = Field(
        ge=0.0,
        allow_inf_nan=False,
        description=(
            "该路线延迟的第50百分位数"
        ),
    )

    p95_latency_ms: float = Field(
        ge=0.0,
        allow_inf_nan=False,
        description=(
            "该路线延迟的第95百分位数"
        ),
    )

    external_model_call_count: int = Field(
        ge=0,
        description=(
            "该路线全部结果的外部Vision模型"
            "调用次数之和"
        ),
    )

    estimated_cost_result_count: int = Field(
        ge=0,
        description=(
            "具有可用成本估算值的结果数量"
        ),
    )

    total_estimated_cost_usd: (
        float
        | None
    ) = Field(
        default=None,
        ge=0.0,
        allow_inf_nan=False,
        description=(
            "具有完整成本信息时的美元成本总和；"
            "发生外部调用但用量信息不足时为None"
        ),
    )

    @model_validator(
        mode="after"
    )
    def summary_must_be_consistent(
        self,
    ) -> Self:
        """校验执行数量、派生比率、延迟和成本关系。"""

        execution_count = (
            self.completed_count
            + self.partial_count
            + self.abstained_count
            + self.failed_count
        )

        # 每个被评测案例必须且只能落入一种执行状态。
        if (
            execution_count
            != self.evaluated_case_count
        ):
            raise ValueError(
                "四种execution_status数量之和"
                "必须等于evaluated_case_count"
            )

        bounded_counts = {
            "passed_count": (
                self.passed_count
            ),
            "route_appropriate_count": (
                self.route_appropriate_count
            ),
            "outcome_correct_count": (
                self.outcome_correct_count
            ),
            "safety_passed_count": (
                self.safety_passed_count
            ),
            "content_accuracy_sample_count": (
                self
                .content_accuracy_sample_count
            ),
            "estimated_cost_result_count": (
                self.estimated_cost_result_count
            ),
        }

        # 任何分类数量都不能超过实际评测数量。
        for (
            field_name,
            field_value,
        ) in bounded_counts.items():
            if (
                field_value
                > self.evaluated_case_count
            ):
                raise ValueError(
                    f"{field_name}不能超过"
                    "evaluated_case_count"
                )

        expected_rates = {
            "pass_rate": (
                self.passed_count
                / self.evaluated_case_count
            ),
            "route_appropriate_rate": (
                self.route_appropriate_count
                / self.evaluated_case_count
            ),
            "outcome_correct_rate": (
                self.outcome_correct_count
                / self.evaluated_case_count
            ),
            "safety_pass_rate": (
                self.safety_passed_count
                / self.evaluated_case_count
            ),
        }

        # 比率属于可以由计数确定性推导的字段，
        # 不能由报告生成者随意填写。
        for (
            field_name,
            expected_value,
        ) in expected_rates.items():
            actual_value = getattr(
                self,
                field_name,
            )

            if not isclose(
                actual_value,
                expected_value,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    f"{field_name}与对应计数不一致"
                )

        if (
            self.content_accuracy_sample_count
            == 0
        ):
            if (
                self.mean_content_accuracy
                is not None
            ):
                raise ValueError(
                    "没有内容准确率样本时"
                    "mean_content_accuracy必须为None"
                )
        elif (
            self.mean_content_accuracy
            is None
        ):
            raise ValueError(
                "存在内容准确率样本时"
                "必须提供mean_content_accuracy"
            )

        # P95不应低于P50。
        if (
            self.p95_latency_ms
            < self.p50_latency_ms
        ):
            raise ValueError(
                "p95_latency_ms不能低于"
                "p50_latency_ms"
            )

        # 当前一条结果最多调用一次外部模型。
        if (
            self.external_model_call_count
            > self.evaluated_case_count
        ):
            raise ValueError(
                "external_model_call_count不能超过"
                "evaluated_case_count"
            )

        if self.route != "vision_model":
            # 本地OCR和直接拒答都不应产生
            # 外部Vision模型调用和模型成本。
            if (
                self.external_model_call_count
                != 0
            ):
                raise ValueError(
                    "非vision_model路线不能记录"
                    "外部模型调用"
                )

            if (
                self.estimated_cost_result_count
                != 0
            ):
                raise ValueError(
                    "非vision_model路线不能记录"
                    "Vision成本估算结果"
                )

            if (
                self.total_estimated_cost_usd
                != 0.0
            ):
                raise ValueError(
                    "非vision_model路线的"
                    "total_estimated_cost_usd必须为0"
                )

        else:
            # 如果全部外部调用都具有成本估算，
            # 才能把成本相加成完整总成本。
            if (
                self.external_model_call_count
                == self
                .estimated_cost_result_count
            ):
                if (
                    self.total_estimated_cost_usd
                    is None
                ):
                    raise ValueError(
                        "全部Vision调用具有成本估算时"
                        "必须提供成本总和"
                    )
            elif (
                self.total_estimated_cost_usd
                is not None
            ):
                raise ValueError(
                    "Vision成本信息不完整时"
                    "total_estimated_cost_usd必须为None"
                )

        return self


class MultimodalToolSelectionExperimentReport(
    BaseModel
):
    """一次完整工具选择对照实验的结构化报告。"""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[
        "multimodal-tool-selection-report-v1"
    ] = Field(
        default=(
            MULTIMODAL_TOOL_SELECTION_REPORT_VERSION
        ),
        description=(
            "批量实验报告的数据契约版本"
        ),
    )

    case_count: int = Field(
        ge=1,
        description=(
            "报告中不同Gold案例的数量"
        ),
    )

    route_result_count: int = Field(
        ge=1,
        description=(
            "报告中单案例单路线结果的总数量"
        ),
    )

    routes: tuple[
        ToolSelectionRoute,
        ...,
    ] = Field(
        min_length=1,
        max_length=3,
        description=(
            "本次实验实际出现的路线，"
            "按照报告稳定顺序排列"
        ),
    )

    results: tuple[
        MultimodalToolSelectionRouteResult,
        ...,
    ] = Field(
        min_length=1,
        description=(
            "按照案例顺序和路线顺序排列的"
            "全部单路线结果"
        ),
    )

    route_summaries: tuple[
        MultimodalToolSelectionRouteSummary,
        ...,
    ] = Field(
        min_length=1,
        max_length=3,
        description=(
            "每条实际路线恰好对应一个汇总记录"
        ),
    )

    @field_validator(
        "routes",
    )
    @classmethod
    def routes_must_be_unique(
        cls,
        routes: tuple[
            ToolSelectionRoute,
            ...,
        ],
    ) -> tuple[
        ToolSelectionRoute,
        ...,
    ]:
        """路线列表不能重复。"""

        if len(routes) != len(
            set(routes)
        ):
            raise ValueError(
                "routes不能包含重复路线"
            )

        return routes

    @model_validator(
        mode="after"
    )
    def report_must_be_consistent(
        self,
    ) -> Self:
        """校验案例、路线、结果和汇总之间的关系。"""

        if (
            len(self.results)
            != self.route_result_count
        ):
            raise ValueError(
                "route_result_count必须等于"
                "results数量"
            )

        result_keys = tuple(
            (
                result.case_id,
                result.route,
            )
            for result in self.results
        )

        # 同一个案例的同一条路线只能出现一次。
        if len(result_keys) != len(
            set(result_keys)
        ):
            raise ValueError(
                "results不能包含重复的"
                "case_id和route组合"
            )

        result_case_ids = {
            result.case_id
            for result in self.results
        }

        if (
            len(result_case_ids)
            != self.case_count
        ):
            raise ValueError(
                "case_count必须等于results中"
                "不同case_id的数量"
            )

        result_routes = {
            result.route
            for result in self.results
        }

        if result_routes != set(
            self.routes
        ):
            raise ValueError(
                "routes必须与results中的"
                "实际路线集合一致"
            )

        summary_routes = tuple(
            summary.route
            for summary
            in self.route_summaries
        )

        if len(summary_routes) != len(
            set(summary_routes)
        ):
            raise ValueError(
                "route_summaries不能包含"
                "重复路线"
            )

        if set(summary_routes) != set(
            self.routes
        ):
            raise ValueError(
                "每条实际路线必须且只能拥有"
                "一个route_summary"
            )

        # 每个Summary声明的案例数量必须与
        # results中该路线的真实结果数量一致。
        for summary in self.route_summaries:
            actual_route_result_count = sum(
                1
                for result in self.results
                if result.route
                == summary.route
            )

            if (
                summary.evaluated_case_count
                != actual_route_result_count
            ):
                raise ValueError(
                    "route_summary的"
                    "evaluated_case_count"
                    "与results不一致"
                )

        return self
