"""多模态工具选择单路线评测结果的数据契约。

本模块定义一个案例经过一条处理路线后，
评测脚本必须保存的统一结构。

它负责：

1. 记录实际执行状态和脱敏观察项；
2. 记录标准答案的匹配与缺失情况；
3. 记录内容准确率；
4. 区分正确拒答、技术失败和正常完成；
5. 记录路线是否适合当前案例；
6. 记录安全检查、延迟和外部模型调用成本；
7. 保证派生指标与实际字段一致。

本模块不负责：

1. 读取图片；
2. 执行OCR或规则解析；
3. 调用Vision模型；
4. 判断文字语义是否等价；
5. 计算整批实验的汇总指标；
6. 生成Markdown报告。
"""

# isclose用于比较浮点数。
#
# 例如2 / 3得到的浮点结果可能存在非常小的
# 二进制表示误差，因此不直接使用严格相等比较。
from math import (
    isclose,
)

# Annotated允许给基本类型附加Field约束。
#
# Literal把字符串字段限制在固定选项中。
#
# Self表示当前Pydantic模型自身，
# 用于model_validator的返回类型。
from typing import (
    Annotated,
    Literal,
    Self,
)

# BaseModel是Pydantic模型基类。
#
# ConfigDict配置整个模型的行为。
#
# Field声明字段范围、格式和说明。
#
# field_validator校验一个字段或同类字段。
#
# model_validator校验多个字段之间的关系。
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# 复用案例定义中的路线类型，
# 避免评测结果重新声明一套不同的路线名称。
from app.schemas.multimodal_tool_selection import (
    ToolSelectionRoute,
)


# 单条路线的执行状态。
#
# completed：
# 路线取得完整、可评分的观察结果。
#
# partial：
# 取得部分观察，但仍要求人工复核。
#
# abstained：
# 路线主动拒绝形成观察，
# 例如图片严重模糊或规则不支持当前布局。
#
# failed：
# 路线因输入错误、超时或上游异常而失败。
ToolSelectionRouteExecutionStatus = Literal[
    "completed",
    "partial",
    "abstained",
    "failed",
]


# 技术失败分类。
#
# 只有execution_status为failed时才使用。
#
# input_validation：
# 图片或参数未通过输入校验。
#
# timeout：
# OCR或Vision调用超过时限。
#
# upstream_error：
# 外部Vision服务连接或状态错误。
#
# invalid_response：
# Provider返回内容不符合内部Schema。
#
# unexpected_error：
# 其他没有被预期分类的技术异常。
ToolSelectionFailureKind = Literal[
    "input_validation",
    "timeout",
    "upstream_error",
    "invalid_response",
    "unexpected_error",
]


# 成本记录状态。
#
# not_applicable：
# 没有调用计费模型，例如本地OCR或直接拒答。
#
# estimated：
# 已根据模型、图片和调用信息给出成本估算。
#
# unavailable：
# 确实调用了外部模型，
# 但当前Provider没有返回足够用量数据。
ToolSelectionCostStatus = Literal[
    "not_applicable",
    "estimated",
    "unavailable",
]


# 报告中的观察项、失败说明和成本说明
# 都使用简短、非空且受长度限制的文字。
ToolSelectionEvaluationText = Annotated[
    str,
    Field(
        min_length=1,
        max_length=1_000,
    ),
]


class MultimodalToolSelectionRouteResult(
    BaseModel
):
    """一个案例经过一条处理路线后的评分结果。

    该模型记录的是实验结果，不是生产诊断。

    actual_items只能保存必要的脱敏观察，
    不能保存完整图片、Base64、模型私有推理过程、
    Authorization、API Key或完整敏感日志。
    """

    model_config = ConfigDict(
        # 清理普通字符串两端空白。
        str_strip_whitespace=True,

        # 禁止出现未声明字段。
        extra="forbid",

        # 评测记录创建后不能重新赋值。
        frozen=True,
    )

    case_id: str = Field(
        pattern=r"^multimodal-[0-9]{3}$",
        description=(
            "与Gold案例对应的稳定编号"
        ),
        examples=[
            "multimodal-001",
        ],
    )

    route: ToolSelectionRoute = Field(
        description=(
            "本次实际评测的处理路线"
        ),
    )

    source_image_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description=(
            "本次路线实际处理图片的SHA-256；"
            "必须与Gold案例图片摘要一致"
        ),
    )

    execution_status: (
        ToolSelectionRouteExecutionStatus
    ) = Field(
        description=(
            "路线是完整、部分、拒答还是失败"
        ),
    )

    actual_items: tuple[
        ToolSelectionEvaluationText,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=30,
        description=(
            "路线实际得到的脱敏文字或视觉观察；"
            "拒答和失败时必须为空"
        ),
    )

    matched_expected_items: tuple[
        ToolSelectionEvaluationText,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=30,
        description=(
            "Gold标准信息中被实际结果正确覆盖的项目"
        ),
    )

    missing_expected_items: tuple[
        ToolSelectionEvaluationText,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=30,
        description=(
            "Gold标准信息中没有被实际结果覆盖的项目"
        ),
    )

    content_accuracy: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
        description=(
            "匹配标准项数量除以全部标准项数量；"
            "应拒答案例没有标准观察时为None"
        ),
    )

    abstained: bool = Field(
        description=(
            "该路线是否没有形成可用观察"
        ),
    )

    expected_abstention: bool = Field(
        description=(
            "Gold案例是否要求拒答"
        ),
    )

    route_appropriate: bool = Field(
        description=(
            "该路线是否包含在案例的acceptable_routes中；"
            "即使错误路线碰巧安全拒答，也不能伪装成首选路线"
        ),
    )

    outcome_correct: bool = Field(
        description=(
            "实际观察或拒答行为是否符合Gold答案"
        ),
    )

    safety_passed: bool = Field(
        description=(
            "路线是否遵守不执行图片指令、"
            "不泄漏敏感内容和不伪造事实等安全要求"
        ),
    )

    passed: bool = Field(
        description=(
            "路线适合、结果正确且安全时才为True"
        ),
    )

    requires_human_check: bool = Field(
        description=(
            "该路线结果是否要求人员查看原图复核"
        ),
    )

    latency_ms: float = Field(
        ge=0.0,
        allow_inf_nan=False,
        description=(
            "从路线开始处理到形成结构化结果的毫秒耗时"
        ),
    )

    external_model_call_count: int = Field(
        ge=0,
        le=1,
        description=(
            "本路线调用外部计费Vision模型的次数；"
            "当前单路线实验最多调用一次"
        ),
    )

    cost_status: ToolSelectionCostStatus = Field(
        description=(
            "成本是不适用、已有估算还是无法估算"
        ),
    )

    estimated_cost_usd: float | None = Field(
        default=None,
        ge=0.0,
        allow_inf_nan=False,
        description=(
            "本次路线估算的美元成本；"
            "无外部模型调用时必须为0"
        ),
    )

    cost_note: (
        ToolSelectionEvaluationText
        | None
    ) = Field(
        default=None,
        description=(
            "成本估算依据或无法估算的原因"
        ),
    )

    failure_kind: (
        ToolSelectionFailureKind
        | None
    ) = Field(
        default=None,
        description=(
            "技术失败分类；"
            "只有failed状态可以设置"
        ),
    )

    public_message: (
        ToolSelectionEvaluationText
        | None
    ) = Field(
        default=None,
        description=(
            "面向报告的脱敏失败或拒答说明"
        ),
    )

    @field_validator(
        "actual_items",
        "matched_expected_items",
        "missing_expected_items",
    )
    @classmethod
    def tuple_items_must_be_unique(
        cls,
        items: tuple[
            str,
            ...,
        ],
    ) -> tuple[
        str,
        ...,
    ]:
        """同一元组中不能重复相同项目。"""

        if len(items) != len(
            set(items)
        ):
            raise ValueError(
                "工具选择评测结果的"
                "元组字段不能包含重复项目"
            )

        return items

    @model_validator(mode="after")
    def result_must_be_consistent(
        self,
    ) -> Self:
        """校验执行、评分、成本和失败字段的一致性。"""

        # 同一个Gold标准项不能同时声称
        # “已经匹配”和“仍然缺失”。
        if set(
            self.matched_expected_items
        ).intersection(
            self.missing_expected_items
        ):
            raise ValueError(
                "matched_expected_items和"
                "missing_expected_items不能重叠"
            )

        expected_item_count = (
            len(
                self.matched_expected_items
            )
            + len(
                self.missing_expected_items
            )
        )

        if self.expected_abstention:
            # 应拒答案例本来就没有可靠标准观察。
            if expected_item_count != 0:
                raise ValueError(
                    "应拒答案例不能包含"
                    "标准观察匹配项"
                )

            if self.content_accuracy is not None:
                raise ValueError(
                    "应拒答案例的"
                    "content_accuracy必须为None"
                )

        else:
            # 可回答案例必须提供标准项，
            # 否则无法计算内容准确率。
            if expected_item_count == 0:
                raise ValueError(
                    "可回答案例必须包含"
                    "匹配或缺失的标准项"
                )

            expected_accuracy = (
                len(
                    self
                    .matched_expected_items
                )
                / expected_item_count
            )

            if self.content_accuracy is None:
                raise ValueError(
                    "可回答案例必须包含"
                    "content_accuracy"
                )

            if not isclose(
                self.content_accuracy,
                expected_accuracy,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    "content_accuracy必须等于"
                    "匹配标准项数量除以标准项总数"
                )

        if self.execution_status in {
            "completed",
            "partial",
        }:
            if self.abstained:
                raise ValueError(
                    "completed或partial"
                    "不能标记为abstained"
                )

            if not self.actual_items:
                raise ValueError(
                    "completed或partial必须包含"
                    "实际观察项"
                )

            if self.failure_kind is not None:
                raise ValueError(
                    "非failed状态不能包含"
                    "failure_kind"
                )

            if (
                self.execution_status
                == "partial"
                and not self
                .requires_human_check
            ):
                raise ValueError(
                    "partial必须要求人工复核"
                )

        elif self.execution_status == (
            "abstained"
        ):
            if not self.abstained:
                raise ValueError(
                    "abstained状态必须设置"
                    "abstained=True"
                )

            if self.actual_items:
                raise ValueError(
                    "abstained状态不能包含"
                    "实际观察项"
                )

            if self.failure_kind is not None:
                raise ValueError(
                    "主动拒答不能标记为"
                    "技术失败"
                )

            if self.public_message is None:
                raise ValueError(
                    "主动拒答必须包含"
                    "脱敏拒答说明"
                )

            if not self.requires_human_check:
                raise ValueError(
                    "主动拒答必须要求"
                    "人员检查或补充图片"
                )

        elif self.execution_status == (
            "failed"
        ):
            if not self.abstained:
                raise ValueError(
                    "failed状态不能声称"
                    "产生了可用观察"
                )

            if self.actual_items:
                raise ValueError(
                    "failed状态不能包含"
                    "实际观察项"
                )

            if self.failure_kind is None:
                raise ValueError(
                    "failed状态必须包含"
                    "failure_kind"
                )

            if self.public_message is None:
                raise ValueError(
                    "failed状态必须包含"
                    "脱敏失败说明"
                )

            if not self.requires_human_check:
                raise ValueError(
                    "failed状态必须要求"
                    "人员检查或重试"
                )

        # outcome_correct是可以从Gold与执行结果
        # 确定性推导出的字段，不能由调用方随意填写。
        if self.expected_abstention:
            expected_outcome_correct = (
                self.execution_status
                == "abstained"
            )
        else:
            expected_outcome_correct = (
                self.execution_status
                in {
                    "completed",
                    "partial",
                }
                and self.content_accuracy
                is not None
                and isclose(
                    self.content_accuracy,
                    1.0,
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
            )

        if (
            self.outcome_correct
            != expected_outcome_correct
        ):
            raise ValueError(
                "outcome_correct必须与"
                "标准答案和实际执行结果一致"
            )

        # 最终通过必须同时满足三项：
        #
        # 1. 路线适合；
        # 2. 输出或拒答正确；
        # 3. 安全检查通过。
        expected_passed = (
            self.route_appropriate
            and self.outcome_correct
            and self.safety_passed
        )

        if self.passed != expected_passed:
            raise ValueError(
                "passed必须等于"
                "route_appropriate、"
                "outcome_correct和"
                "safety_passed的逻辑与"
            )

        # 只有vision_model路线可以调用外部视觉模型。
        if (
            self.route != "vision_model"
            and self
            .external_model_call_count
            != 0
        ):
            raise ValueError(
                "非vision_model路线不能记录"
                "外部模型调用"
            )

        if self.external_model_call_count == 0:
            if (
                self.cost_status
                != "not_applicable"
            ):
                raise ValueError(
                    "没有外部模型调用时"
                    "cost_status必须是"
                    "not_applicable"
                )

            if self.estimated_cost_usd != 0.0:
                raise ValueError(
                    "没有外部模型调用时"
                    "estimated_cost_usd必须为0"
                )

            if self.cost_note is not None:
                raise ValueError(
                    "成本不适用时不能包含"
                    "cost_note"
                )

        else:
            if self.route != "vision_model":
                raise ValueError(
                    "只有vision_model路线"
                    "可以调用外部模型"
                )

            if self.cost_status == (
                "not_applicable"
            ):
                raise ValueError(
                    "调用外部模型后成本状态"
                    "不能是not_applicable"
                )

            if self.cost_status == (
                "estimated"
            ):
                if (
                    self.estimated_cost_usd
                    is None
                ):
                    raise ValueError(
                        "estimated成本状态必须"
                        "提供estimated_cost_usd"
                    )

            elif self.cost_status == (
                "unavailable"
            ):
                if (
                    self.estimated_cost_usd
                    is not None
                ):
                    raise ValueError(
                        "unavailable成本状态不能"
                        "提供虚假成本数值"
                    )

                if self.cost_note is None:
                    raise ValueError(
                        "无法估算成本时必须说明"
                        "缺少哪些用量信息"
                    )

        return self