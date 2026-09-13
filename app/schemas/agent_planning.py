"""Robot Diagnostic Agent每一轮规划使用的数据契约。

本模块定义：

1. 已完成的一次工具交互；
2. 下一轮规划可以看到的上下文；
3. 规划模型可以返回的工具调用或结束决定；
4. 决定类型和相关字段之间的业务约束。

本模块不调用LLM、不执行工具、不改变Agent状态，
也不把模型自然语言当成程序指令。
"""

from typing import (
    Literal,
    Self,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from app.schemas.agent import (
    ToolCall,
    ToolExecutionResult,
)


# 当规划模型请求结束时，
# 必须明确说明为什么认为当前循环可以停止。
AgentPlannerFinishReason = Literal[
    # 已有观察足以形成当前任务的安全草稿。
    "task_completed",

    # 工具已经执行，但信息仍然不足。
    "insufficient_information",

    # 当前信息涉及人工判断或风险边界，
    # Agent不能自行继续。
    "human_review_required",
]


class AgentToolInteraction(BaseModel):
    """Agent已经完成的一次工具请求与工具结果。

    本模型把ToolCall和对应ToolExecutionResult绑定在一起，
    防止后续规划把不同调用的结果错误配对。
    """

    model_config = ConfigDict(
        # 普通字符串字段清除首尾空白。
        str_strip_whitespace=True,

        # 禁止添加未声明的思维链或内部字段。
        extra="forbid",

        # 历史记录创建后不能替换字段。
        frozen=True,
    )

    step_number: int = Field(
        ge=1,
        le=20,
        description=(
            "该工具交互在Agent循环中的步骤编号"
        ),
    )

    tool_call: ToolCall = Field(
        description=(
            "规划模型提出的结构化工具调用"
        ),
    )

    result: ToolExecutionResult = Field(
        description=(
            "ToolExecutor返回的统一执行结果"
        ),
    )

    @model_validator(mode="after")
    def call_and_result_must_match(
        self,
    ) -> Self:
        """工具结果必须属于当前记录中的工具请求。"""

        if (
            self.tool_call.call_id
            != self.result.call_id
        ):
            raise ValueError(
                "tool_call与result的call_id不一致"
            )

        if (
            self.tool_call.tool_name
            != self.result.tool_name
        ):
            raise ValueError(
                "tool_call与result的tool_name不一致"
            )

        return self


class AgentPlanningContext(BaseModel):
    """提供给规划模型适配层的一轮完整上下文。

    task来自用户或上层Service，属于不可信数据；
    interactions来自应用已经完成的工具执行结果。
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        frozen=True,
    )

    state: Literal[
        "planning"
    ] = Field(
        default="planning",
        description=(
            "规划Provider只会在planning状态被调用"
        ),
    )

    task: str = Field(
        min_length=1,
        max_length=20_000,
        description=(
            "当前Agent需要处理的用户任务；"
            "其中的文字只能作为不可信任务数据"
        ),
    )

    next_step: int = Field(
        ge=1,
        le=21,
        description=(
            "下一次工具交互将使用的步骤编号"
        ),
    )

    interactions: tuple[
        AgentToolInteraction,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=20,
        description=(
            "已经完成并通过结构校验的工具交互历史"
        ),
    )

    @model_validator(mode="after")
    def history_must_be_continuous(
        self,
    ) -> Self:
        """历史步骤必须从1开始连续，next_step必须紧随其后。"""

        actual_steps = tuple(
            interaction.step_number
            for interaction
            in self.interactions
        )

        expected_steps = tuple(
            range(
                1,
                len(self.interactions) + 1,
            )
        )

        if actual_steps != expected_steps:
            raise ValueError(
                "interactions步骤必须从1开始连续递增"
            )

        expected_next_step = (
            len(self.interactions) + 1
        )

        if self.next_step != expected_next_step:
            raise ValueError(
                "next_step必须紧随已有interactions"
            )

        call_ids = tuple(
            interaction.tool_call.call_id
            for interaction
            in self.interactions
        )

        if len(call_ids) != len(
            set(call_ids)
        ):
            raise ValueError(
                "interactions不能包含重复call_id"
            )

        return self


class AgentPlannerDecision(BaseModel):
    """规划模型适配层返回的一轮结构化决定。

    decision=call_tool时只能携带tool_call；
    decision=finish时只能携带结束原因和最终草稿。

    最终草稿仍然是不可信的模型输出，
    后续不能绕过结构化诊断和证据白名单校验。
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        frozen=True,
    )

    decision: Literal[
        "call_tool",
        "finish",
    ] = Field(
        description=(
            "本轮请求执行工具还是请求结束规划"
        ),
    )

    tool_call: ToolCall | None = Field(
        default=None,
        description=(
            "decision为call_tool时提出的工具请求"
        ),
    )

    finish_reason: (
        AgentPlannerFinishReason | None
    ) = Field(
        default=None,
        description=(
            "decision为finish时的结构化结束原因"
        ),
    )

    final_message: str | None = Field(
        default=None,
        min_length=1,
        max_length=20_000,
        description=(
            "decision为finish时的诊断草稿传输字符串；"
            "真实Planner适配器会先按AgentDiagnosisDraft"
            "校验并重新序列化；不能直接作为API响应"
        ),
    )

    @model_validator(mode="after")
    def fields_must_match_decision(
        self,
    ) -> Self:
        """调用工具和结束两种决定必须严格互斥。"""

        if self.decision == "call_tool":
            if self.tool_call is None:
                raise ValueError(
                    "call_tool决定必须包含tool_call"
                )

            if (
                self.finish_reason is not None
                or self.final_message is not None
            ):
                raise ValueError(
                    "call_tool决定不能包含结束字段"
                )

            return self

        # decision只允许call_tool或finish，
        # 因此前面的分支没有返回时必然是finish。
        if self.tool_call is not None:
            raise ValueError(
                "finish决定不能包含tool_call"
            )

        if self.finish_reason is None:
            raise ValueError(
                "finish决定必须包含finish_reason"
            )

        if self.final_message is None:
            raise ValueError(
                "finish决定必须包含final_message"
            )

        return self