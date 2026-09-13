"""请求级Agent工具策略的数据契约。

本模块定义：

1. Agent当前允许使用的只读工具名称；
2. 一项请求可能需要的结构化能力；
3. 工具策略的处理方向；
4. 稳定、机器可读的策略原因码；
5. 工具策略最终返回的数据结构；
6. 工具范围、能力需求和提前终止之间的关系。

本模块只定义数据契约，不负责：

1. 分析用户请求；
2. 识别提示注入；
3. 判断高风险控制意图；
4. 调用Planner；
5. 执行任何工具；
6. 生成最终诊断报告。

后续策略服务会创建AgentToolPolicyDecision，
AgentDiagnosisService再根据该决定选择进入Planner
或在Planner之前安全结束。
"""

from typing import (
    Annotated,
    Literal,
    Self,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


# 当前Agent允许注册的五种只读工具。
#
# 使用Literal而不是普通str，可以在运行时拒绝：
#
# run_shell
# control_robot
# disable_emergency_stop
#
# 等没有进入允许集合的任意工具名称。
AgentReadOnlyToolName = Literal[
    "analyze_robot_image",
    "search_knowledge",
    "get_robot_telemetry",
    "draft_test_case",
    "get_current_time",
]


# 工具名称使用固定顺序。
#
# frozenset适合权限判断，但没有面向展示的稳定顺序。
# 这个元组用于构造Planner任务、日志和测试结果，
# 保证同一工具集合每次都产生相同排列。
AGENT_READ_ONLY_TOOL_ORDER: tuple[
    AgentReadOnlyToolName,
    ...,
] = (
    "analyze_robot_image",
    "search_knowledge",
    "get_robot_telemetry",
    "draft_test_case",
    "get_current_time",
)


# 能力描述“任务需要取得什么信息”，
# 工具名称描述“使用什么手段取得信息”。
#
# 将两者分开后，策略模块可以先识别任务需求，
# 再把需求转换成最小工具集合。
AgentRequiredCapability = Literal[
    # 需要读取图片中的文字、颜色、状态或外观。
    "vision_observation",

    # 需要由知识库文档确认原因、规则、限制或步骤。
    "knowledge_evidence",

    # 需要读取当前模拟位置、电量、速度或任务状态。
    "robot_telemetry",

    # 用户明确要求生成测试或验证草案。
    "test_case_draft",

    # 用户明确要求当前时间或日期参照。
    "current_time",
]


# 每一种能力对应一个现有Agent工具。
#
# 后续策略实现会根据required_capabilities计算
# allowed_tool_names，而不是维护两套无关列表。
CAPABILITY_TO_TOOL: dict[
    AgentRequiredCapability,
    AgentReadOnlyToolName,
] = {
    "vision_observation": (
        "analyze_robot_image"
    ),
    "knowledge_evidence": (
        "search_knowledge"
    ),
    "robot_telemetry": (
        "get_robot_telemetry"
    ),
    "test_case_draft": (
        "draft_test_case"
    ),
    "current_time": (
        "get_current_time"
    ),
}


# 策略执行后只有三种处理方向。
#
# continue_to_planner：
# 当前请求可以由Agent继续自动处理。
#
# abstained：
# 缺少必要输入或没有安全可用的信息来源，
# 不进入Planner，直接以信息不足结束。
#
# human_review_required：
# 请求涉及高风险控制、权限边界或其他必须
# 由人工处理的情况，不进入Planner。
AgentToolPolicyDisposition = Literal[
    "continue_to_planner",
    "abstained",
    "human_review_required",
]


# 原因码供测试、日志和会话记录使用。
#
# 调用方应该判断这些稳定英文值，
# 而不是解析可能调整措辞的中文public_message。
AgentToolPolicyReasonCode = Literal[
    # 请求携带了通过API Schema校验的图片。
    "image_input_available",

    # 当前任务明确需要直接观察图片。
    "vision_observation_required",

    # 任务依赖图片，但请求没有提供图片。
    "required_image_missing",

    # 当前任务需要知识库文档证据。
    "knowledge_evidence_required",

    # 当前任务明确需要当前模拟遥测。
    "robot_telemetry_required",

    # 用户明确要求测试或验证草案。
    "test_case_draft_required",

    # 用户明确要求当前时间或日期。
    "current_time_required",

    # 图片与当前任务没有可确认的关系。
    "image_not_relevant",

    # 检测到试图改变系统规则或扩大权限的输入。
    "prompt_injection_detected",

    # 检测到要求访问图片、二维码或日志中外部链接的输入。
    "external_link_action_detected",

    # 检测到要求执行图片、二维码或日志中命令的输入。
    "untrusted_instruction_execution_detected",

    # 检测到解除安全联锁或控制设备等高风险意图。
    "high_risk_control_detected",

    # 当前请求没有安全、必要的自动工具可以调用。
    "no_safe_tool_required",
]


# 策略产生的公开说明和缺失信息共用这一约束。
#
# Annotated可以在str类型旁附加Pydantic Field约束，
# 因此元组中的每一个字符串也会执行长度校验。
AgentToolPolicyText = Annotated[
    str,
    Field(
        min_length=1,
        max_length=500,
    ),
]


class AgentToolPolicyDecision(BaseModel):
    """一次请求级工具策略判断的完整结果。

    这是工具策略服务和AgentDiagnosisService之间
    使用的数据契约。

    它不是Planner的返回值，也不是最终HTTP响应。
    """

    model_config = ConfigDict(
        # 清理普通字符串字段两端的空白。
        str_strip_whitespace=True,

        # 拒绝没有声明的字段。
        #
        # 这样可以防止调用方把任意调试信息、
        # 原始日志或模型输出塞进策略结果。
        extra="forbid",

        # 策略决定创建后不允许重新给字段赋值。
        #
        # 权限集合不能在进入Planner前被其他模块
        # 原地扩大。
        frozen=True,
    )

    policy_version: str = Field(
        default="agent-tool-policy-v1",
        min_length=1,
        max_length=100,
        pattern=r"^[a-z0-9][a-z0-9._-]*$",
        description=(
            "当前请求使用的工具策略版本"
        ),
    )

    disposition: (
        AgentToolPolicyDisposition
    ) = Field(
        description=(
            "继续进入Planner、信息不足拒答，"
            "或要求人工审核"
        ),
    )

    required_capabilities: tuple[
        AgentRequiredCapability,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=5,
        description=(
            "当前任务真正需要取得的信息能力"
        ),
    )

    allowed_tool_names: tuple[
        AgentReadOnlyToolName,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=5,
        description=(
            "允许同时暴露给Planner和Executor的"
            "最小工具集合"
        ),
    )

    reason_codes: tuple[
        AgentToolPolicyReasonCode,
        ...,
    ] = Field(
        min_length=1,
        max_length=12,
        description=(
            "解释本次策略决定的稳定机器可读原因"
        ),
    )

    public_message: (
        AgentToolPolicyText | None
    ) = Field(
        default=None,
        description=(
            "提前结束时可以返回给用户的安全说明；"
            "继续进入Planner时必须为空"
        ),
    )

    missing_information: tuple[
        AgentToolPolicyText,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=10,
        description=(
            "提前结束时仍缺少的信息或"
            "必须由人工完成的事项"
        ),
    )

    @field_validator(
        "required_capabilities",
        "allowed_tool_names",
        "reason_codes",
        "missing_information",
    )
    @classmethod
    def tuple_items_must_be_unique(
        cls,
        values: tuple[
            str,
            ...,
        ],
    ) -> tuple[
        str,
        ...,
    ]:
        """策略结果中的元组字段不能包含重复值。"""

        if len(values) != len(set(values)):
            raise ValueError(
                "工具策略的元组字段"
                "不能包含重复项"
            )

        return values

    @field_validator(
        "allowed_tool_names"
    )
    @classmethod
    def tools_must_use_stable_order(
        cls,
        values: tuple[
            AgentReadOnlyToolName,
            ...,
        ],
    ) -> tuple[
        AgentReadOnlyToolName,
        ...,
    ]:
        """工具名称必须使用项目规定的稳定顺序。"""

        expected_order = tuple(
            tool_name
            for tool_name
            in AGENT_READ_ONLY_TOOL_ORDER
            if tool_name in values
        )

        if values != expected_order:
            raise ValueError(
                "allowed_tool_names必须按照"
                "AGENT_READ_ONLY_TOOL_ORDER排列"
            )

        return values

    @model_validator(mode="after")
    def policy_relationships_must_be_consistent(
        self,
    ) -> Self:
        """校验处理方向、能力、工具和说明的关系。"""

        # 根据所需能力计算理论上的最小工具集合。
        #
        # 先遍历固定工具顺序，再检查相应能力是否存在，
        # 可以保证计算结果具有稳定顺序。
        expected_tools = tuple(
            tool_name
            for tool_name
            in AGENT_READ_ONLY_TOOL_ORDER
            if tool_name
            in {
                CAPABILITY_TO_TOOL[capability]
                for capability
                in self.required_capabilities
            }
        )

        if (
            self.disposition
            == "continue_to_planner"
        ):
            # 继续进入Planner必须至少有一项必要能力。
            if not self.required_capabilities:
                raise ValueError(
                    "继续进入Planner时必须声明"
                    "required_capabilities"
                )

            # allowed_tool_names必须恰好等于能力映射结果。
            #
            # 多一个工具会扩大权限，
            # 少一个工具则无法完成声明的任务。
            if (
                self.allowed_tool_names
                != expected_tools
            ):
                raise ValueError(
                    "继续进入Planner时，"
                    "allowed_tool_names必须与"
                    "required_capabilities完全对应"
                )

            # Planner尚未执行，策略层不应提前生成
            # 最终用户说明或声称信息仍然缺失。
            if self.public_message is not None:
                raise ValueError(
                    "继续进入Planner时"
                    "public_message必须为空"
                )

            if self.missing_information:
                raise ValueError(
                    "继续进入Planner时"
                    "missing_information必须为空"
                )

            return self

        # abstained和human_review_required都在
        # Planner之前结束，因此不得暴露任何工具。
        if self.allowed_tool_names:
            raise ValueError(
                "提前结束时"
                "allowed_tool_names必须为空"
            )

        # 提前结束必须给出稳定、可公开的解释。
        if self.public_message is None:
            raise ValueError(
                "提前结束时必须提供"
                "public_message"
            )

        # 还必须说明缺少什么，或者哪一项需要人工处理。
        if not self.missing_information:
            raise ValueError(
                "提前结束时必须提供"
                "missing_information"
            )

        return self
