"""AgentRunner完成或中止一次循环后的结果契约。

本模块定义：

1. Agent循环终止原因；
2. 正常完成和安全中止的字段关系；
3. 已完成工具交互的不可变历史；
4. Planner最终草稿与应用终止说明的区别；
5. 循环中止后仍未解决的信息。

本模块不运行Agent、不调用工具，也不处理HTTP请求。
"""

from typing import (
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

from app.schemas.agent_planning import (
    AgentPlannerFinishReason,
    AgentToolInteraction,
)
from app.schemas.agent_progress import (
    AgentMissingInformation,
    AgentProgress,
)


# AgentRunner当前支持的结构化终止原因。
#
# 调用方应该判断termination_reason，
# 而不是解析termination_message中的中文文本。
AgentRunTerminationReason = Literal[
    # Planner主动返回finish决定。
    "planner_finished",

    # 请求级安全分类或工具策略在Planner启动前
    # 已经确定必须拒答或转人工审核。
    "request_policy_finished",

    # Planner在已经执行max_steps个工具后，
    # 仍然请求继续调用工具。
    "max_steps_reached",

    # Planner重复请求相同call_id，
    # 或使用新call_id重复相同工具和参数。
    "duplicate_tool_call",

    # Planner没有在规定时间内返回决定。
    "planner_timeout",

    # Planner调用过程中出现了非超时异常。
    "planner_error",

    # Planner返回的对象、结构或工具参数
    # 不符合AgentRunner要求的数据契约。
    "invalid_planner_response",

    # 可恢复工具失败连续达到配置上限。
    "tool_failure_limit_reached",

    # 工具请求触碰只读或风险权限边界。
    "tool_policy_violation",
]


class AgentRunResult(BaseModel):
    """AgentRunner一次完整运行的结构化结果。

    state描述循环是否正常结束，
    不代表最终机器人诊断一定完整。

    interactions只保存已经实际执行完成的工具调用。
    尚未执行、被重复检测阻止或超过步数限制的调用，
    不能伪造为已完成历史。
    """

    model_config = ConfigDict(
        # 清除普通字符串字段的首尾空白。
        str_strip_whitespace=True,

        # 拒绝未声明字段，避免模型私有推理、
        # SDK对象或任意调试数据进入公开结果。
        extra="forbid",

        # 运行结果创建后不允许直接替换字段。
        frozen=True,
    )

    state: Literal[
        "completed",
        "aborted",
    ] = Field(
        description=(
            "Agent循环正常完成还是安全中止"
        ),
    )

    termination_reason: (
        AgentRunTerminationReason
    ) = Field(
        description=(
            "由Python控制逻辑决定的"
            "结构化终止原因"
        ),
    )

    termination_message: str = Field(
        min_length=1,
        max_length=500,
        description=(
            "由应用生成、可以公开记录的终止说明"
        ),
    )

    interactions: tuple[
        AgentToolInteraction,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=20,
        description=(
            "实际完成的工具调用与执行结果历史"
        ),
    )

    progress: AgentProgress | None = Field(
        default=None,
        description=(
            "启用证据驱动状态机时生成的最终进度快照；"
            "None用于兼容尚未迁移的内部调用方"
        ),
    )

    missing_information: tuple[
        AgentMissingInformation,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=20,
        description=(
            "循环结束时仍然没有解决的信息；"
            "aborted结果必须至少包含一项"
        ),
    )

    finish_reason: (
        AgentPlannerFinishReason | None
    ) = Field(
        default=None,
        description=(
            "正常结束时Planner提供的"
            "结构化结束原因"
        ),
    )

    final_message: str | None = Field(
        default=None,
        min_length=1,
        max_length=20_000,
        description=(
            "正常结束时保存的已完成结构校验、"
            "但尚未通过证据白名单校验的"
            "Agent诊断草稿JSON字符串"
        ),
    )

    @field_validator(
        "missing_information"
    )
    @classmethod
    def missing_information_must_be_unique(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        """同一项缺失信息不能在结果中重复出现。"""

        # set会去除重复值。
        #
        # 如果转换成set后数量减少，
        # 说明原始tuple中存在重复字符串。
        if len(value) != len(set(value)):
            raise ValueError(
                "missing_information"
                "不能包含重复项"
            )

        return value

    @model_validator(mode="after")
    def fields_must_match_state(
        self,
    ) -> Self:
        """正常完成和中止结果必须使用不同字段组合。"""

        if self.progress is not None:
            # 一条AgentProgress工具记录必须对应一条真实
            # AgentToolInteraction，不能增加或隐藏工具步骤。
            if len(
                self.progress.tool_records
            ) != len(self.interactions):
                raise ValueError(
                    "progress.tool_records必须与"
                    "interactions数量一致"
                )

            for record, interaction in zip(
                self.progress.tool_records,
                self.interactions,
                strict=True,
            ):
                if (
                    record.step_number
                    != interaction.step_number
                    or record.call_id
                    != interaction.tool_call.call_id
                    or record.tool_name
                    != interaction.tool_call.tool_name
                    or record.status
                    != interaction.result.status
                ):
                    raise ValueError(
                        "progress.tool_records必须与"
                        "interactions逐项对应"
                    )

            if (
                self.state == "completed"
                and self.progress.state
                not in {
                    "completed",
                    "partial",
                    "abstained",
                    "human_review_required",
                }
            ):
                raise ValueError(
                    "正常结束的Runner必须包含"
                    "终态AgentProgress"
                )

            if (
                self.state == "aborted"
                and self.progress.state != "failed"
            ):
                raise ValueError(
                    "中止的Runner必须包含failed进度"
                )

        if self.state == "completed":
            if (
                self.termination_reason
                != "planner_finished"
            ):
                raise ValueError(
                    "completed结果必须由"
                    "planner_finished结束"
                )

            if self.finish_reason is None:
                raise ValueError(
                    "completed结果必须包含"
                    "finish_reason"
                )

            if self.final_message is None:
                raise ValueError(
                    "completed结果必须包含"
                    "final_message"
                )

            # task_completed表示Planner声称任务已经完成。
            #
            # 这种情况下不能同时声明仍有缺失信息，
            # 否则结果内部含义互相矛盾。
            if (
                self.finish_reason
                == "task_completed"
                and self.missing_information
            ):
                raise ValueError(
                    "task_completed结果不能包含"
                    "missing_information"
                )

            return self

        # state只能是completed或aborted，
        # 因此前面的分支未返回时必然是aborted。
        if (
            self.termination_reason
            == "planner_finished"
        ):
            raise ValueError(
                "aborted结果不能使用"
                "planner_finished"
            )

        if (
            self.finish_reason is not None
            or self.final_message is not None
        ):
            raise ValueError(
                "aborted结果不能包含"
                "Planner最终草稿"
            )

        # 中止结果不能只说“失败了”，
        # 还要告诉上层哪些内容尚未完成。
        #
        # Agent Service后续可以把这些内容转换成
        # partial或abstained诊断中的缺失信息。
        if not self.missing_information:
            raise ValueError(
                "aborted结果必须包含"
                "missing_information"
            )

        return self
