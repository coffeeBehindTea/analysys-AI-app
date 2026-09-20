"""RobotOps Copilot Agent SSE公开事件的数据契约。

本模块定义Agent执行过程中允许通过SSE公开的结构化事件。

本模块负责：

1. 定义所有SSE事件共有的字段；
2. 定义每一种事件允许携带的专属字段；
3. 校验事件类型、生命周期状态和负载之间的关系；
4. 使用event_type组成Pydantic判别联合；
5. 阻止未声明字段进入公开事件流。

本模块不负责：

1. 执行Agent；
2. 调用Planner、LLM或工具；
3. 保存诊断会话；
4. 把事件编码成text/event-stream；
5. 保存或返回完整图片、完整日志和模型私有思维链。
"""

from typing import (
    Annotated,
    Literal,
    Self,
    TypeAlias,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from app.schemas.agent import (
    AgentLifecycleState,
    ToolName,
)
from app.schemas.agent_api import (
    AgentDiagnosisResponse,
    AgentToolTraceEvent,
    AgentTraceInputSummary,
    MAX_AGENT_DIAGNOSIS_IMAGES,
)
from app.schemas.agent_progress import (
    AgentProgress,
)
from app.schemas.agent_tool_policy import (
    AgentReadOnlyToolName,
    AgentRequiredCapability,
    AgentToolPolicyDisposition,
    AgentToolPolicyReasonCode,
)
from app.schemas.error import (
    ErrorDetail,
)


# 当前公开SSE事件契约的版本。
#
# 如果将来修改事件字段含义、删除字段或改变事件顺序，
# 应创建新版本，而不是静默改变v1行为。
AGENT_SSE_EVENT_SCHEMA_VERSION = (
    "agent-sse-event-v1"
)


# event_type是SSE客户端选择事件处理逻辑的稳定机器字段。
#
# Web控制台应判断这些值，
# 而不是解析可能调整措辞的message。
AgentSseEventType = Literal[
    "request_received",
    "safety_classified",
    "tool_scope_decided",
    "planning_started",
    "tool_started",
    "tool_finished",
    "progress_updated",
    "diagnosis_finished",
    "stream_error",
]


class AgentSseEventBase(BaseModel):
    """所有Agent SSE公开事件的公共字段。

    具体事件会继承本模型，并把event_type和state
    收窄成该事件允许使用的Literal值。
    """

    model_config = ConfigDict(
        # 清理普通字符串首尾空白。
        str_strip_whitespace=True,

        # 禁止内部对象、原始参数或调试字段
        # 意外进入SSE公开响应。
        extra="forbid",

        # 事件创建后不允许被修改。
        #
        # 同一个事件可能先进入会话记录，
        # 再进入HTTP编码器；不可变对象可以避免
        # 两个消费者看到不同内容。
        frozen=True,

        # 禁止NaN和无穷大进入JSON。
        allow_inf_nan=False,
    )

    schema_version: Literal[
        "agent-sse-event-v1"
    ] = Field(
        default=AGENT_SSE_EVENT_SCHEMA_VERSION,
        description="公开SSE事件的数据契约版本",
    )

    sequence: int = Field(
        ge=1,
        le=10_000,
        description=(
            "当前请求内从1开始连续递增的事件序号；"
            "同时作为SSE的id"
        ),
    )

    event_type: AgentSseEventType = Field(
        description=(
            "用于Pydantic判别联合和客户端分发的事件类型"
        ),
    )

    request_id: str = Field(
        min_length=1,
        max_length=200,
        description=(
            "用于关联HTTP响应、日志、事件和会话的请求ID"
        ),
    )

    state: AgentLifecycleState = Field(
        description=(
            "事件产生时Agent对外公开的生命周期状态"
        ),
    )

    message: str = Field(
        min_length=1,
        max_length=500,
        description=(
            "经过清理、可以直接展示给用户的简短说明"
        ),
    )


class AgentRequestReceivedEvent(
    AgentSseEventBase
):
    """请求已经通过FastAPI基础结构校验。

    该事件不表示安全策略已经允许执行工具，
    也不表示图片内容已经通过VisionInputAdapter校验。
    """

    event_type: Literal[
        "request_received"
    ] = "request_received"

    state: Literal[
        "received"
    ] = "received"

    image_count: int = Field(
        ge=0,
        le=MAX_AGENT_DIAGNOSIS_IMAGES,
        description=(
            "请求中提交的图片数量；"
            "不包含图片Base64或图片内容"
        ),
    )

    has_task_goal: bool = Field(
        description=(
            "请求是否显式提供了可选task_goal"
        ),
    )


class AgentSafetyClassifiedEvent(
    AgentSseEventBase
):
    """确定性请求安全分类和工具策略已经完成。

    continue_to_planner表示可以继续规划；
    abstained和human_review_required表示策略层
    已经形成终止处置，不需要把请求交给Planner。
    """

    event_type: Literal[
        "safety_classified"
    ] = "safety_classified"

    state: Literal[
        "planning",
        "completed",
    ]

    disposition: AgentToolPolicyDisposition = Field(
        description=(
            "请求策略决定继续、拒答还是转人工审核"
        ),
    )

    reason_codes: tuple[
        AgentToolPolicyReasonCode,
        ...,
    ] = Field(
        min_length=1,
        max_length=20,
        description=(
            "安全分类和工具策略产生的稳定机器原因码"
        ),
    )

    @model_validator(mode="after")
    def disposition_must_match_state(
        self,
    ) -> Self:
        """继续规划和策略终止必须使用对应生命周期状态。"""

        if (
            self.disposition
            == "continue_to_planner"
            and self.state != "planning"
        ):
            raise ValueError(
                "continue_to_planner事件"
                "必须使用planning状态"
            )

        if (
            self.disposition
            in {
                "abstained",
                "human_review_required",
            }
            and self.state != "completed"
        ):
            raise ValueError(
                "策略层终止事件必须使用"
                "completed生命周期状态"
            )

        return self


class AgentToolScopeDecidedEvent(
    AgentSseEventBase
):
    """请求允许使用的最小工具范围已经确定。

    本事件只在策略决定continue_to_planner时产生。
    客户端不能通过该事件或原始请求修改工具范围。
    """

    event_type: Literal[
        "tool_scope_decided"
    ] = "tool_scope_decided"

    state: Literal[
        "planning"
    ] = "planning"

    required_capabilities: tuple[
        AgentRequiredCapability,
        ...,
    ] = Field(
        min_length=1,
        max_length=5,
        description=(
            "完成当前请求所必需的能力集合"
        ),
    )

    allowed_tool_names: tuple[
        AgentReadOnlyToolName,
        ...,
    ] = Field(
        min_length=1,
        max_length=5,
        description=(
            "当前请求允许Planner选择的最小工具集合"
        ),
    )


class AgentPlanningStartedEvent(
    AgentSseEventBase
):
    """Agent开始规划下一步。

    ready_to_finish阶段也可能再次进入Planner，
    此时allowed_tool_names可以为空，
    因为Planner只允许形成最终草稿。
    """

    event_type: Literal[
        "planning_started"
    ] = "planning_started"

    state: Literal[
        "planning"
    ] = "planning"

    step_number: int = Field(
        ge=1,
        le=20,
        description=(
            "即将规划的Agent步骤编号"
        ),
    )

    allowed_tool_names: tuple[
        AgentReadOnlyToolName,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=5,
        description=(
            "本轮Planner仍然允许选择的工具；"
            "准备结束时可以为空"
        ),
    )


class AgentToolStartedEvent(
    AgentSseEventBase
):
    """一个已经通过策略检查的工具开始执行。

    input_summary必须是脱敏摘要，
    不能使用Planner提供的完整arguments。
    """

    event_type: Literal[
        "tool_started"
    ] = "tool_started"

    state: Literal[
        "tool_running"
    ] = "tool_running"

    step_id: int = Field(
        ge=1,
        le=20,
        description=(
            "当前工具执行对应的Agent步骤编号"
        ),
    )

    tool_name: ToolName = Field(
        description=(
            "本步骤实际开始执行的注册工具名称"
        ),
    )

    input_summary: AgentTraceInputSummary = Field(
        description=(
            "经过脱敏和长度限制的工具输入摘要"
        ),
    )


class AgentToolFinishedEvent(
    AgentSseEventBase
):
    """一次工具执行已经结束。

    工具成功、空结果、被拒绝、超时和执行错误
    都使用本事件，通过trace.status区分。
    """

    event_type: Literal[
        "tool_finished"
    ] = "tool_finished"

    state: Literal[
        "observing"
    ] = "observing"

    trace: AgentToolTraceEvent = Field(
        description=(
            "已经通过公开轨迹契约校验的工具执行结果"
        ),
    )


class AgentProgressUpdatedEvent(
    AgentSseEventBase
):
    """确定性AgentProgress状态机已经完成一次归约。

    progress不是Planner自行声明的进度，
    而是Python根据工具结果和安全策略计算的快照。
    """

    event_type: Literal[
        "progress_updated"
    ] = "progress_updated"

    state: Literal[
        "planning",
        "observing",
        "completed",
        "aborted",
    ]

    progress: AgentProgress = Field(
        description=(
            "本次工具结果处理后的完整脱敏进度快照"
        ),
    )

    @model_validator(mode="after")
    def progress_must_match_state(
        self,
    ) -> Self:
        """内部进度状态必须映射到正确的公开生命周期。"""

        expected_state_by_progress = {
            "collecting": "observing",
            "ready_to_finish": "planning",
            "completed": "completed",
            "partial": "completed",
            "abstained": "completed",
            "human_review_required": "completed",
            "failed": "aborted",
        }

        expected_state = (
            expected_state_by_progress[
                self.progress.state
            ]
        )

        if self.state != expected_state:
            raise ValueError(
                "公开生命周期状态与"
                "AgentProgress.state不一致"
            )

        return self


class AgentDiagnosisFinishedEvent(
    AgentSseEventBase
):
    """最终诊断已经通过校验并完成会话持久化。

    response是普通JSON诊断接口使用的同一响应模型，
    因此SSE和非SSE接口不会产生两套不同的最终结果。
    """

    event_type: Literal[
        "diagnosis_finished"
    ] = "diagnosis_finished"

    state: Literal[
        "completed",
        "aborted",
    ]

    response: AgentDiagnosisResponse = Field(
        description=(
            "经过最终报告、轨迹和会话关系校验的诊断响应"
        ),
    )

    @model_validator(mode="after")
    def response_must_match_event(
        self,
    ) -> Self:
        """终止事件必须与嵌套诊断响应保持一致。"""

        if (
            self.request_id
            != self.response.request_id
        ):
            raise ValueError(
                "事件request_id必须与"
                "响应request_id一致"
            )

        if (
            self.state
            != self.response.execution.state
        ):
            raise ValueError(
                "事件state必须与"
                "response.execution.state一致"
            )

        return self


class AgentStreamErrorEvent(
    AgentSseEventBase
):
    """SSE响应开始后发生的公开流错误。

    HTTP响应开始发送后已经不能再修改状态码，
    因此必须把错误编码成事件并关闭连接。
    """

    event_type: Literal[
        "stream_error"
    ] = "stream_error"

    state: Literal[
        "aborted"
    ] = "aborted"

    error: ErrorDetail = Field(
        description=(
            "稳定机器错误码和安全公开信息"
        ),
    )

    retryable: bool = Field(
        description=(
            "客户端是否可以在不修改输入的情况下重试"
        ),
    )


# 这是Pydantic判别联合。
#
# Field(discriminator="event_type")告诉Pydantic：
#
# 1. 先读取输入对象的event_type；
# 2. 根据Literal值选择唯一事件模型；
# 3. 只执行该模型的字段和关系校验；
# 4. 未知event_type直接拒绝。
#
# 这样比包含大量Optional字段的单一事件模型更严格。
AgentSseEvent: TypeAlias = Annotated[
    (
        AgentRequestReceivedEvent
        | AgentSafetyClassifiedEvent
        | AgentToolScopeDecidedEvent
        | AgentPlanningStartedEvent
        | AgentToolStartedEvent
        | AgentToolFinishedEvent
        | AgentProgressUpdatedEvent
        | AgentDiagnosisFinishedEvent
        | AgentStreamErrorEvent
    ),
    Field(
        discriminator="event_type"
    ),
]