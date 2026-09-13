"""OCR确定性规则解析结果的数据契约。

本模块定义OCR文字经过固定规则解析后的结构化结果。

它负责：

1. 区分状态面板、遥测面板和未知面板；
2. 保存规则成功提取出的字段；
3. 保存缺失字段和失败原因；
4. 区分完成、部分完成、不支持和空结果；
5. 关联原始图片和OCR文字摘要；
6. 保证字段、状态和人工复核要求一致。

本模块不负责：

1. 读取图片；
2. 调用Tesseract；
3. 编写或执行正则表达式；
4. 解释故障码的工程含义；
5. 检索知识库；
6. 调用LLM或Vision模型；
7. 生成最终诊断。
"""

# Annotated允许给基本类型附加Field约束。
#
# Literal把字符串限制为固定值。
#
# Self表示当前Pydantic模型自身。
from typing import (
    Annotated,
    Literal,
    Self,
)

# BaseModel是Pydantic数据模型基类。
#
# ConfigDict配置模型级行为。
#
# Field声明字段范围和格式。
#
# field_validator校验同类元组字段。
#
# model_validator校验多个字段之间的关系。
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# 复用OCR层已经定义的分析状态，
# 避免在两个模块中定义不同状态集合。
from app.schemas.ocr import (
    OcrAnalysisStatus,
)


# 当前规则解析支持两种已知面板。
#
# unknown表示OCR有文字，
# 但当前规则无法识别这是哪种面板。
OcrRulePanelType = Literal[
    "robot_status",
    "robot_telemetry",
    "unknown",
]


# 规则解析自身的处理状态。
#
# completed：
# 已识别面板类型，并提取出全部必要字段。
#
# partial：
# 已识别面板类型，但字段不完整，
# 或OCR自身置信度不足。
#
# unsupported：
# OCR识别到了文字，
# 但文字布局不属于当前支持的面板规则。
#
# empty：
# 上游OCR没有识别到可解析文字。
OcrRuleParseStatus = Literal[
    "completed",
    "partial",
    "unsupported",
    "empty",
]


# 当前规则解析器能够返回的结构化字段。
OcrRuleFieldName = Literal[
    "fault_code",
    "network_state",
    "task_state",
    "battery_percent",
    "speed_mps",
    "temperature_celsius",
]


# 网络状态只接受规则明确支持的值。
#
# 不使用UNKNOWN字符串：
# 无法确认时应保持字段为None，
# 并写入missing_fields。
OcrRuleNetworkState = Literal[
    "CONNECTED",
    "DISCONNECTED",
]


# 当前模拟面板支持的任务状态。
OcrRuleTaskState = Literal[
    "PAUSED",
    "RUNNING",
    "STOPPED",
    "IDLE",
    "CHARGING",
]


# 规则解析器的版本号。
#
# 修改标签、正则或数值范围时，
# 应创建新版本而不是静默改变旧实验含义。
OCR_RULE_VERSION = (
    "robot-panel-rules-v1"
)


# 所有字段的固定顺序。
#
# 规则解析器、Schema和报告都使用这个顺序，
# 避免集合转换导致输出顺序不稳定。
OCR_RULE_FIELD_ORDER: tuple[
    OcrRuleFieldName,
    ...,
] = (
    "fault_code",
    "network_state",
    "task_state",
    "battery_percent",
    "speed_mps",
    "temperature_celsius",
)


# 两类面板分别要求的必要字段。
STATUS_PANEL_REQUIRED_FIELDS: tuple[
    OcrRuleFieldName,
    ...,
] = (
    "fault_code",
    "network_state",
    "task_state",
)


TELEMETRY_PANEL_REQUIRED_FIELDS: tuple[
    OcrRuleFieldName,
    ...,
] = (
    "battery_percent",
    "speed_mps",
    "temperature_celsius",
)


# 多个失败原因共用的简短文字类型。
OcrRuleFailureReason = Annotated[
    str,
    Field(
        min_length=1,
        max_length=500,
    ),
]


class OcrRuleParseResult(
    BaseModel
):
    """一次OCR规则解析后的结构化结果。

    本模型保存的是“图片文字中提取出的字段”，
    不是经过知识库确认的工程事实。

    例如fault_code等于ERR-NET-4001，
    只能表示OCR和规则提取到了这个故障码，
    不能直接证明机器人当前确实存在该故障。
    """

    model_config = ConfigDict(
        # 清理普通字符串两端空白。
        str_strip_whitespace=True,

        # 禁止增加未声明字段。
        extra="forbid",

        # 校验后不允许重新赋值。
        frozen=True,
    )

    status: OcrRuleParseStatus = Field(
        description=(
            "规则解析是完成、部分完成、"
            "不支持还是空结果"
        ),
    )

    panel_type: OcrRulePanelType = Field(
        description=(
            "识别出的面板类型；"
            "无法识别时为unknown"
        ),
    )

    ocr_status: OcrAnalysisStatus = Field(
        description=(
            "生成本结果的上游OCR质量状态"
        ),
    )

    source_image_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description=(
            "与上游VisionInput一致的图片摘要"
        ),
    )

    source_text_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description=(
            "与上游OcrObservation识别文字一致的摘要"
        ),
    )

    source: Literal[
        "ocr_rule"
    ] = Field(
        default="ocr_rule",
        description=(
            "明确标记字段来自OCR规则解析"
        ),
    )

    rules_version: Literal[
        "robot-panel-rules-v1"
    ] = Field(
        default=OCR_RULE_VERSION,
        description=(
            "实际使用的确定性规则版本"
        ),
    )

    fault_code: str | None = Field(
        default=None,
        pattern=(
            r"^ERR-[A-Z0-9]+-[0-9]{4}$"
        ),
        max_length=100,
        description=(
            "从状态面板提取出的故障码"
        ),
        examples=[
            "ERR-NET-4001",
        ],
    )

    network_state: (
        OcrRuleNetworkState
        | None
    ) = Field(
        default=None,
        description=(
            "从状态面板提取出的网络状态"
        ),
    )

    task_state: (
        OcrRuleTaskState
        | None
    ) = Field(
        default=None,
        description=(
            "从状态面板提取出的任务状态"
        ),
    )

    battery_percent: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
        allow_inf_nan=False,
        description=(
            "从遥测面板提取出的电量百分比"
        ),
    )

    speed_mps: float | None = Field(
        default=None,
        ge=-20.0,
        le=20.0,
        allow_inf_nan=False,
        description=(
            "从遥测面板提取出的速度；"
            "负值可以表示反向运动"
        ),
    )

    temperature_celsius: float | None = Field(
        default=None,
        ge=-100.0,
        le=200.0,
        allow_inf_nan=False,
        description=(
            "从遥测面板提取出的摄氏温度"
        ),
    )

    matched_fields: tuple[
        OcrRuleFieldName,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=6,
        description=(
            "本次成功解析出的字段；"
            "顺序必须符合固定字段顺序"
        ),
    )

    missing_fields: tuple[
        OcrRuleFieldName,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=3,
        description=(
            "已识别面板中缺少或无法确认的"
            "必要字段"
        ),
    )

    failure_reasons: tuple[
        OcrRuleFailureReason,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=10,
        description=(
            "部分、不支持或空结果的原因；"
            "不保存完整图片或完整敏感文字"
        ),
    )

    requires_human_check: bool = Field(
        description=(
            "规则结果是否需要人员查看原图确认"
        ),
    )

    @field_validator(
        "matched_fields",
        "missing_fields",
        "failure_reasons",
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
        """同一元组不能重复相同项目。"""

        if len(items) != len(
            set(items)
        ):
            raise ValueError(
                "OCR规则结果的元组字段"
                "不能包含重复项目"
            )

        return items

    @model_validator(mode="after")
    def parsed_fields_must_be_consistent(
        self,
    ) -> Self:
        """校验字段值、面板类型和状态之间的关系。"""

        # 根据真正非None的字段，
        # 按固定顺序计算实际匹配字段。
        actual_matched_fields = tuple(
            field_name
            for field_name in (
                OCR_RULE_FIELD_ORDER
            )
            if getattr(
                self,
                field_name,
            )
            is not None
        )

        if (
            self.matched_fields
            != actual_matched_fields
        ):
            raise ValueError(
                "matched_fields必须与"
                "实际非空解析字段一致"
            )

        if set(
            self.matched_fields
        ).intersection(
            self.missing_fields
        ):
            raise ValueError(
                "matched_fields和"
                "missing_fields不能重叠"
            )

        if self.panel_type == (
            "robot_status"
        ):
            required_fields = (
                STATUS_PANEL_REQUIRED_FIELDS
            )

            # 状态面板不能携带遥测面板字段。
            if (
                self.battery_percent
                is not None
                or self.speed_mps
                is not None
                or self.temperature_celsius
                is not None
            ):
                raise ValueError(
                    "robot_status不能包含"
                    "遥测面板字段"
                )

        elif self.panel_type == (
            "robot_telemetry"
        ):
            required_fields = (
                TELEMETRY_PANEL_REQUIRED_FIELDS
            )

            # 遥测面板不能携带状态面板字段。
            if (
                self.fault_code is not None
                or self.network_state
                is not None
                or self.task_state
                is not None
            ):
                raise ValueError(
                    "robot_telemetry不能包含"
                    "状态面板字段"
                )

        else:
            required_fields = ()

            # unknown表示当前规则不能确认面板类型。
            if self.matched_fields:
                raise ValueError(
                    "unknown面板不能包含"
                    "已匹配字段"
                )

        expected_missing_fields = tuple(
            field_name
            for field_name in required_fields
            if getattr(
                self,
                field_name,
            )
            is None
        )

        if (
            self.missing_fields
            != expected_missing_fields
        ):
            raise ValueError(
                "missing_fields必须与"
                "面板实际缺失字段一致"
            )

        if self.ocr_status == "empty":
            if self.status != "empty":
                raise ValueError(
                    "上游OCR为空时"
                    "规则状态必须是empty"
                )

        elif self.status == "empty":
            raise ValueError(
                "规则状态为empty时"
                "上游OCR也必须为空"
            )

        if self.status == "completed":
            if self.panel_type == "unknown":
                raise ValueError(
                    "completed不能使用"
                    "unknown面板类型"
                )

            if self.ocr_status != "completed":
                raise ValueError(
                    "completed要求上游OCR"
                    "状态也是completed"
                )

            if self.missing_fields:
                raise ValueError(
                    "completed不能包含"
                    "missing_fields"
                )

            if self.failure_reasons:
                raise ValueError(
                    "completed不能包含"
                    "failure_reasons"
                )

            if self.requires_human_check:
                raise ValueError(
                    "completed不能要求"
                    "OCR质量人工复核"
                )

        elif self.status == "partial":
            if self.panel_type == "unknown":
                raise ValueError(
                    "partial必须已经识别"
                    "面板类型"
                )

            if self.ocr_status == "empty":
                raise ValueError(
                    "partial不能来自空OCR结果"
                )

            if not self.matched_fields:
                raise ValueError(
                    "partial必须至少包含"
                    "一个已匹配字段"
                )

            # 部分结果要么缺字段，
            # 要么因为OCR低置信度等原因不能直接完成。
            if (
                not self.missing_fields
                and not self.failure_reasons
            ):
                raise ValueError(
                    "partial必须说明缺失字段"
                    "或失败原因"
                )

            if not self.requires_human_check:
                raise ValueError(
                    "partial必须要求人工复核"
                )

        elif self.status == "unsupported":
            if self.panel_type != "unknown":
                raise ValueError(
                    "unsupported必须使用"
                    "unknown面板类型"
                )

            if self.ocr_status == "empty":
                raise ValueError(
                    "unsupported不能来自"
                    "空OCR结果"
                )

            if not self.failure_reasons:
                raise ValueError(
                    "unsupported必须说明"
                    "不支持原因"
                )

            if not self.requires_human_check:
                raise ValueError(
                    "unsupported必须要求"
                    "人工复核"
                )

        elif self.status == "empty":
            if self.panel_type != "unknown":
                raise ValueError(
                    "empty必须使用"
                    "unknown面板类型"
                )

            if self.matched_fields:
                raise ValueError(
                    "empty不能包含"
                    "已匹配字段"
                )

            if not self.failure_reasons:
                raise ValueError(
                    "empty必须说明"
                    "没有解析结果的原因"
                )

            if not self.requires_human_check:
                raise ValueError(
                    "empty必须要求人工复核"
                )

        return self