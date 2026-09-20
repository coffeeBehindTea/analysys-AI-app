"""Agent诊断生命周期通知到SSE公开事件的适配器。

本模块实现AgentDiagnosisObserver协议，把诊断服务和
AgentRunner产生的内部通知转换成经过校验的公开SSE事件。

本模块负责：

1. 发布请求接收、安全分类和最小工具范围事件；
2. 把规划开始通知转换成planning_started事件；
3. 把工具开始通知转换成tool_started事件；
4. 把工具结束通知转换成tool_finished事件；
5. 把确定性进度快照转换成progress_updated事件；
6. 在响应校验和会话保存完成后发布最终诊断；
7. 使用公共轨迹构造器生成脱敏输入和结果摘要；
8. 把结构化事件交给AgentEventPublisher发布。

本模块不负责：

1. 调用Planner或LLM；
2. 执行Agent工具；
3. 决定请求允许使用哪些工具；
4. 计算新的AgentProgress；
5. 构造最终DiagnosisReport；
6. 发布stream_error异常终端事件；
7. 编码text/event-stream文本。
"""

from app.agent.diagnosis_observer import (
    AgentDiagnosisObserver,
)
from app.schemas.agent import (
    AgentLifecycleState,
    ToolCall,
    ToolExecutionResult,
)
from app.schemas.agent_api import (
    AgentDiagnosisResponse,
)
from app.schemas.agent_planning import (
    AgentToolInteraction,
)
from app.schemas.agent_progress import (
    AgentProgress,
    AgentProgressState,
)
from app.schemas.agent_tool_policy import (
    AgentReadOnlyToolName,
    AgentRequiredCapability,
    AgentToolPolicyDisposition,
    AgentToolPolicyReasonCode,
)
from app.services.agent_event_publisher import (
    AgentEventPublisher,
)
from app.services.agent_tool_trace_builder import (
    build_agent_tool_input_summary,
    build_agent_tool_trace,
)


# AgentProgress描述业务证据进度，
# AgentLifecycleState描述对客户端公开的执行阶段。
#
# 两者不是同一个概念，因此必须显式映射，
# 不能直接把progress.state当成SSE事件state。
_PROGRESS_LIFECYCLE_STATE: dict[
    AgentProgressState,
    AgentLifecycleState,
] = {
    # 工具刚刚产生结果，但必要能力还没有收集完成。
    "collecting": "observing",

    # 必要能力已经完成，下一轮只允许Planner形成最终草稿。
    "ready_to_finish": "planning",

    # 下列四种都是业务终态，对外统一属于completed生命周期。
    "completed": "completed",
    "partial": "completed",
    "abstained": "completed",
    "human_review_required": "completed",

    # 不可恢复的内部进度失败映射为aborted。
    "failed": "aborted",
}


# 这些是可以直接展示给浏览器用户的固定公开说明。
#
# 消息不包含原始请求、日志、证据正文或Planner输出，
# 客户端的程序判断仍应使用state和progress.state，
# 不应解析这些中文说明。
_PROGRESS_PUBLIC_MESSAGES: dict[
    AgentProgressState,
    str,
] = {
    "collecting": (
        "工具结果已经处理，"
        "Agent仍在收集必要信息"
    ),
    "ready_to_finish": (
        "必要能力已经完成，"
        "Agent正在准备最终诊断"
    ),
    "completed": (
        "Agent已经取得完成任务所需的信息"
    ),
    "partial": (
        "Agent已经取得部分可信信息，"
        "但仍存在缺失项"
    ),
    "abstained": (
        "现有信息不足以形成可信诊断"
    ),
    "human_review_required": (
        "当前请求需要由具备权限的人员审核"
    ),
    "failed": (
        "Agent执行未能形成可用进度"
    ),
}


class AgentRunnerSseObserver(
    AgentDiagnosisObserver
):
    """把完整诊断生命周期通知发布成结构化SSE事件。

    每个Agent诊断请求必须创建独立的Observer实例，
    并绑定该请求自己的AgentEventPublisher。

    本类不会缓存另一份事件历史。
    Publisher负责事件序号、顺序、工具配对和终端状态。
    """

    def __init__(
        self,
        *,
        publisher: AgentEventPublisher,
    ) -> None:
        """创建绑定单次请求Publisher的观察者。

        publisher：
            当前诊断请求独享的事件发布器。

            Observer只通过publisher.publish()提交事件，
            不直接访问Publisher内部Queue。
        """

        if not isinstance(
            publisher,
            AgentEventPublisher,
        ):
            raise TypeError(
                "publisher必须是"
                "AgentEventPublisher"
            )

        self._publisher = publisher

    @property
    def publisher(
        self,
    ) -> AgentEventPublisher:
        """返回当前Observer绑定的事件发布器。

        该属性主要供应用编排层检查对象关系。

        Publisher本身仍负责保护事件顺序，
        调用方不能绕过publish()修改内部历史。
        """

        return self._publisher

    async def on_request_received(
        self,
        *,
        image_count: int,
        has_task_goal: bool,
    ) -> None:
        """发布请求已经通过API基础结构校验的事件。

        本方法只接收图片数量和任务目标存在标记，
        不接收完整请求、图片Base64或日志正文。
        """

        await self._publisher.publish(
            event_type="request_received",
            state="received",
            message="诊断请求已经接收",
            payload={
                "image_count": image_count,
                "has_task_goal": (
                    has_task_goal
                ),
            },
        )

    async def on_safety_classified(
        self,
        *,
        disposition: (
            AgentToolPolicyDisposition
        ),
        reason_codes: tuple[
            AgentToolPolicyReasonCode,
            ...,
        ],
    ) -> None:
        """发布确定性请求安全与工具策略处置。

        继续规划使用planning生命周期；
        安全拒答和转人工属于正常业务终态，使用completed。
        """

        lifecycle_state: (
            AgentLifecycleState
        ) = (
            "planning"
            if disposition
            == "continue_to_planner"
            else "completed"
        )

        await self._publisher.publish(
            event_type="safety_classified",
            state=lifecycle_state,
            message=(
                "请求安全分类和工具策略"
                "已经完成"
            ),
            payload={
                "disposition": disposition,
                "reason_codes": reason_codes,
            },
        )

    async def on_tool_scope_decided(
        self,
        *,
        required_capabilities: tuple[
            AgentRequiredCapability,
            ...,
        ],
        allowed_tool_names: tuple[
            AgentReadOnlyToolName,
            ...,
        ],
    ) -> None:
        """发布策略和Runner注册表交集形成的最小工具范围。"""

        await self._publisher.publish(
            event_type="tool_scope_decided",
            state="planning",
            message="最小工具范围已经确定",
            payload={
                "required_capabilities": (
                    required_capabilities
                ),
                "allowed_tool_names": (
                    allowed_tool_names
                ),
            },
        )

    async def on_diagnosis_finished(
        self,
        *,
        response: AgentDiagnosisResponse,
    ) -> None:
        """发布已经校验并完成会话持久化的最终诊断。

        事件直接嵌入普通JSON接口使用的同一个响应模型，
        不根据工具轨迹重新生成另一份诊断结果。
        """

        if not isinstance(
            response,
            AgentDiagnosisResponse,
        ):
            raise TypeError(
                "response必须是"
                "AgentDiagnosisResponse"
            )

        await self._publisher.publish(
            event_type="diagnosis_finished",
            state=(
                response.execution.state
            ),
            message="最终诊断已经生成并保存",
            payload={
                "response": response,
            },
        )

    async def on_planning_started(
        self,
        *,
        step_number: int,
        allowed_tool_names: tuple[
            str,
            ...,
        ],
    ) -> None:
        """发布Runner即将调用Planner的事件。

        Runner会在每轮Planner调用前执行本方法。

        allowed_tool_names来自Runner本轮实际传给Planner的
        工具Schema，而不是来自用户请求或Planner自行声明。
        """

        await self._publisher.publish(
            event_type="planning_started",
            state="planning",
            message=(
                "Agent正在规划"
                f"第{step_number}步"
            ),
            payload={
                "step_number": step_number,
                "allowed_tool_names": (
                    allowed_tool_names
                ),
            },
        )

    async def on_tool_started(
        self,
        *,
        step_id: int,
        tool_call: ToolCall,
    ) -> None:
        """发布已经通过Runner检查的工具开始事件。

        Runner只会在工具通过以下检查后通知本方法：

        1. 工具位于当前允许范围；
        2. 没有违反只读安全策略；
        3. 不是重复工具调用；
        4. 参数已经通过对应输入模型校验。

        本方法使用公共摘要函数，不能直接公开arguments。
        """

        input_summary = (
            build_agent_tool_input_summary(
                tool_call
            )
        )

        await self._publisher.publish(
            event_type="tool_started",
            state="tool_running",
            message=(
                "Agent工具开始执行："
                f"{tool_call.tool_name}"
            ),
            payload={
                "step_id": step_id,
                "tool_name": (
                    tool_call.tool_name
                ),
                "input_summary": (
                    input_summary
                ),
            },
        )

    async def on_tool_finished(
        self,
        *,
        step_id: int,
        tool_call: ToolCall,
        result: ToolExecutionResult,
    ) -> None:
        """发布一次工具执行的确定结果。

        success、empty、rejected、timeout和error
        都会进入本方法。

        本方法先构造AgentToolInteraction，让Pydantic检查
        ToolCall与ToolExecutionResult的call_id和tool_name
        必须属于同一次调用。

        然后使用公共轨迹构造器完成输出二次校验和脱敏。
        """

        interaction = AgentToolInteraction(
            step_number=step_id,
            tool_call=tool_call,
            result=result,
        )

        built_trace = build_agent_tool_trace(
            interaction
        )

        await self._publisher.publish(
            event_type="tool_finished",
            state="observing",
            message=(
                "Agent工具执行结束："
                f"{tool_call.tool_name}；"
                f"status={result.status}"
            ),
            payload={
                "trace": built_trace.trace,
            },
        )

    async def on_progress_updated(
        self,
        *,
        progress: AgentProgress,
    ) -> None:
        """发布Progress Reducer计算出的权威进度快照。

        progress不是LLM声明的状态。

        它由Python根据请求所需能力、真实工具结果、
        已确认来源、证据覆盖和安全规则计算得到。
        """

        if not isinstance(
            progress,
            AgentProgress,
        ):
            raise TypeError(
                "progress必须是AgentProgress"
            )

        lifecycle_state = (
            _PROGRESS_LIFECYCLE_STATE[
                progress.state
            ]
        )

        public_message = (
            _PROGRESS_PUBLIC_MESSAGES[
                progress.state
            ]
        )

        await self._publisher.publish(
            event_type="progress_updated",
            state=lifecycle_state,
            message=public_message,
            payload={
                # 深复制能够避免Publisher和其他调用方
                # 共享嵌套对象引用。
                "progress": (
                    progress.model_copy(
                        deep=True
                    )
                ),
            },
        )
