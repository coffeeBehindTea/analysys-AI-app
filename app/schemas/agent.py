"""Robot Diagnostic Agent使用的结构化数据契约。

本模块当前负责定义：

1. Agent执行状态；
2. 工具风险级别；
3. 注册工具必须提供的静态定义；
4. 将Pydantic输入模型转换成LLM可见的Tool Schema。

本模块不执行工具、不调用LLM、不管理Agent循环，
也不处理HTTP请求。
"""

from typing import (
    Annotated,
    Any,
    Literal,
    Self,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


# AgentLifecycleState描述Agent执行循环当前所在的阶段。
#
# 这里不复用DiagnosisStatus，因为：
#
# AgentLifecycleState描述“程序执行到哪里”；
# DiagnosisStatus描述“证据能支持多完整的诊断”。
AgentLifecycleState = Literal[
    "received",
    "planning",
    "tool_running",
    "observing",
    "completed",
    "aborted",
]


# 工具风险级别用于后续权限策略。
#
# low：
#     普通只读查询，例如知识库检索。
#
# medium：
#     不修改外部状态，但输出需要人工审核，
#     例如生成测试用例草案。
#
# high：
#     可能改变设备、系统或生产状态。
#     Week 4 Agent不允许注册或执行此类工具。
ToolRiskLevel = Literal[
    "low",
    "medium",
    "high",
]


# ToolName统一约束工具定义和工具调用中的名称。
#
# 使用同一个类型别名可以避免：
#
# ToolDefinition允许一种命名格式，
# 但ToolCall又接受另一种格式。
ToolName = Annotated[
    str,
    Field(
        min_length=3,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_]*$",
        description=(
            "小写字母开头，只包含小写字母、"
            "数字和下划线的工具名称"
        ),
    ),
]


# ToolCallId由LLM SDK或兼容上游返回，
# 用于把工具请求和工具结果对应起来。
#
# 不对字符类型施加过强限制，
# 以兼容不同OpenAI兼容服务的ID格式。
ToolCallId = Annotated[
    str,
    Field(
        min_length=1,
        max_length=200,
        description="一次工具调用的唯一标识",
    ),
]


# ToolExecutionStatus描述工具请求的处理结果。
#
# success：
#     工具成功执行并返回通过输出模型校验的数据。
#
# empty：
#     工具正常完成，但没有找到可用结果。
#
# rejected：
#     工具没有执行，例如工具未知、参数非法或权限不足。
#
# timeout：
#     工具超过ToolDefinition.timeout_seconds。
#
# error：
#     工具执行异常，或者输出没有通过数据契约。
ToolExecutionStatus = Literal[
    "success",
    "empty",
    "rejected",
    "timeout",
    "error",
]


# ToolErrorCode是稳定的机器可读错误码。
#
# AgentRunner和自动化测试通过error_code判断失败类型，
# 不需要解析可能发生变化的中文错误消息。
ToolErrorCode = Literal[
    "unknown_tool",
    "invalid_tool_arguments",
    "tool_blocked",
    "duplicate_tool_call",
    "tool_timeout",
    "empty_result",
    "tool_execution_error",
    "invalid_tool_output",
    "vision_timeout",
    "vision_upstream_error",
    "invalid_vision_response",
]


class ToolDefinition(BaseModel):
    """一个可以进入Agent工具注册表的静态工具定义。

    ToolDefinition只描述工具的名称、数据契约和权限属性。
    真正的Python异步处理函数会在后续ToolRegistry中单独保存。

    将定义和处理函数分开，可以避免把任意可调用对象直接暴露
    给LLM，也便于独立验证和输出LLM可见的Tool Schema。
    """

    model_config = ConfigDict(
        # 自动删除普通字符串字段首尾的空白。
        str_strip_whitespace=True,

        # 拒绝模型中没有声明的额外字段。
        extra="forbid",

        # 工具注册后不允许直接修改定义，
        # 防止运行期间改变名称、风险或输入模型。
        frozen=True,

        # 当前模型包含type[BaseModel]这类Python类对象。
        # 它是内部运行时契约，不会直接作为API响应模型。
        arbitrary_types_allowed=True,
    )

    name: ToolName = Field(
        description=(
            "供LLM请求和注册表查找使用的"
            "工具唯一名称"
        ),
        examples=["search_knowledge"],
    )

    description: str = Field(
        min_length=10,
        max_length=1_000,
        description=(
            "告诉LLM工具用途、适用条件"
            "和主要限制的公开说明"
        ),
    )

    input_model: type[BaseModel] = Field(
        description=(
            "校验工具调用参数的Pydantic模型类"
        ),
    )

    output_model: type[BaseModel] = Field(
        description=(
            "校验工具返回结果的Pydantic模型类"
        ),
    )

    risk_level: ToolRiskLevel = Field(
        description="工具的静态风险级别",
    )

    read_only: bool = Field(
        description=(
            "工具是否保证不修改外部系统、"
            "设备或持久化数据"
        ),
    )

    timeout_seconds: float = Field(
        gt=0,
        le=120,
        description=(
            "单次工具调用允许执行的最大秒数"
        ),
    )

    def to_openai_tool_schema(
        self,
    ) -> dict[str, Any]:
        """生成OpenAI兼容Tool Calling所需的工具描述。

        这里只把输入模型交给LLM，因为模型需要知道
        调用工具时可以提供哪些参数。

        output_model不会发送给LLM。工具执行完成后，
        Python应用会使用output_model校验真实返回值。
        """

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,

                # model_json_schema()是Pydantic BaseModel
                # 类方法。它把字段、类型、必填项和约束
                # 转换成JSON Schema字典。
                "parameters": (
                    self.input_model
                    .model_json_schema()
                ),
            },
        }


class ToolCall(BaseModel):
    """LLM请求执行某个已注册工具的结构化调用。

    本模型只确认调用ID、工具名和参数容器的基本结构。
    arguments中的具体字段必须由ToolExecutor使用
    ToolDefinition.input_model再次校验。
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",

        # 工具请求创建后不允许替换整个字段。
        #
        # 需要注意：frozen不会递归冻结arguments内部的字典，
        # 因此执行器仍应把参数校验成新的Pydantic对象，
        # 而不是把原始arguments直接传给处理函数。
        frozen=True,
    )

    call_id: ToolCallId

    tool_name: ToolName

    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "由LLM提供、尚未经过具体工具"
            "输入模型校验的参数对象"
        ),
    )


class ToolExecutionResult(BaseModel):
    """ToolExecutor处理一次工具调用后的统一结果。

    AgentRunner只接收这个稳定结果，不直接接收：
    
    1. 工具抛出的原始异常；
    2. SDK响应对象；
    3. 未经过output_model校验的任意Python对象。
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        frozen=True,

        # duration_ms不能接受NaN或正负无穷大。
        allow_inf_nan=False,
    )

    call_id: ToolCallId

    tool_name: ToolName

    status: ToolExecutionStatus

    output: dict[str, Any] | None = Field(
        default=None,
        description=(
            "经过ToolDefinition.output_model校验后"
            "使用model_dump得到的工具结果"
        ),
    )

    error_code: ToolErrorCode | None = Field(
        default=None,
        description=(
            "失败、拒绝、超时或空结果对应的"
            "稳定机器可读错误码"
        ),
    )

    public_message: str | None = Field(
        default=None,
        min_length=1,
        max_length=500,
        description=(
            "经过清理、可以进入审计轨迹或"
            "公开响应的错误说明"
        ),
    )

    duration_ms: float = Field(
        ge=0,
        description="本次工具处理耗时，单位为毫秒",
    )

    @model_validator(mode="after")
    def result_fields_must_match_status(
        self,
    ) -> Self:
        """确保状态、输出和错误字段保持一致。"""

        if self.status == "success":
            # 成功结果必须携带经过输出模型校验的数据。
            if self.output is None:
                raise ValueError(
                    "success工具结果必须包含output"
                )

            # 成功不能同时携带错误码或错误消息。
            if (
                self.error_code is not None
                or self.public_message is not None
            ):
                raise ValueError(
                    "success工具结果不能包含错误信息"
                )

            return self

        # 非成功结果不能携带看似可信的工具输出。
        #
        # 否则AgentRunner可能在status表示失败时，
        # 仍然误用output中的数据形成诊断。
        if self.output is not None:
            raise ValueError(
                "非success工具结果不能包含output"
            )

        if self.error_code is None:
            raise ValueError(
                "非success工具结果必须包含error_code"
            )

        if self.public_message is None:
            raise ValueError(
                "非success工具结果必须包含public_message"
            )

        expected_codes: dict[
            ToolExecutionStatus,
            set[ToolErrorCode],
        ] = {
            # success分支已经在上面提前返回，
            # 这里保留空集合是为了让映射结构完整。
            "success": set(),

            "empty": {
                "empty_result",
            },

            "rejected": {
                "unknown_tool",
                "invalid_tool_arguments",
                "tool_blocked",
                "duplicate_tool_call",
            },

            "timeout": {
                "tool_timeout",
                "vision_timeout",
            },

            "error": {
                "tool_execution_error",
                "invalid_tool_output",
                "vision_upstream_error",
                "invalid_vision_response",
            },
        }

        if (
            self.error_code
            not in expected_codes[self.status]
        ):
            raise ValueError(
                "工具状态与error_code不匹配"
            )

        return self
