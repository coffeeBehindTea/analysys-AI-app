"""单次Agent诊断请求的SSE事件发布器。

本模块位于Agent诊断业务流程和HTTP SSE编码层之间。

本模块负责：

1. 为单次请求自动分配连续事件序号；
2. 使用AgentSseEvent判别联合校验公开事件；
3. 检查不同事件之间的先后顺序；
4. 检查tool_started和tool_finished是否配对；
5. 保证一个请求只能产生一个终端事件；
6. 通过异步迭代器向SSE Router提供事件；
7. 保存当前请求内已经公开的不可变事件快照。

本模块不负责：

1. 运行Planner或AgentRunner；
2. 执行任何Agent工具；
3. 构造最终诊断结论；
4. 保存诊断会话；
5. 把事件编码成text/event-stream；
6. 处理HTTP客户端断开。
"""

import asyncio

from collections.abc import (
    AsyncIterator,
    Mapping,
)
from typing import (
    Any,
)

from pydantic import (
    TypeAdapter,
    ValidationError,
)

from app.schemas.agent import (
    AgentLifecycleState,
)
from app.schemas.agent_events import (
    AgentDiagnosisFinishedEvent,
    AgentPlanningStartedEvent,
    AgentSafetyClassifiedEvent,
    AgentSseEvent,
    AgentSseEventType,
    AgentToolFinishedEvent,
    AgentToolStartedEvent,
)


# TypeAdapter让普通联合类型也能够使用Pydantic校验。
#
# AgentSseEvent不是单个BaseModel类，
# 而是带event_type判别字段的Annotated联合类型，
# 因此不能直接调用AgentSseEvent.model_validate()。
_AGENT_SSE_EVENT_ADAPTER = TypeAdapter(
    AgentSseEvent
)


# 这些字段由Publisher统一控制。
#
# 调用方只能通过publish()的明确参数提供它们，
# 不能再把它们放进payload中覆盖序号、请求ID或状态。
_RESERVED_PAYLOAD_FIELDS = frozenset({
    "schema_version",
    "sequence",
    "event_type",
    "request_id",
    "state",
    "message",
})


# 一次事件流只能由以下两种事件结束。
_TERMINAL_EVENT_TYPES = frozenset({
    "diagnosis_finished",
    "stream_error",
})


# 每种非首事件允许跟在哪些事件之后。
#
# 这里描述的是公开协议顺序，
# 不是Agent内部所有状态转换细节。
_ALLOWED_PREDECESSORS: dict[
    AgentSseEventType,
    frozenset[AgentSseEventType],
] = {
    "request_received": frozenset(),

    "safety_classified": frozenset({
        "request_received",
    }),

    "tool_scope_decided": frozenset({
        "safety_classified",
    }),

    "planning_started": frozenset({
        "tool_scope_decided",
        "progress_updated",
    }),

    "tool_started": frozenset({
        "planning_started",
    }),

    "tool_finished": frozenset({
        "tool_started",
    }),

    "progress_updated": frozenset({
        "tool_finished",
    }),

    # 请求策略可以直接产生拒答或转人工结果；
    # 正常Planner也可以在规划或进度更新后结束。
    "diagnosis_finished": frozenset({
        "safety_classified",
        "planning_started",
        "progress_updated",
    }),

    # 事件流开始后的任意非终端阶段都可能失败，
    # 因此stream_error由单独逻辑处理。
    "stream_error": frozenset(),
}


class AgentEventPublicationError(
    RuntimeError
):
    """事件无法安全进入公开事件流。

    这是内部编排错误，不是直接面向客户端的HTTP错误。

    SSE Router或上层Service捕获它后，
    应生成安全的stream_error事件或终止当前请求，
    不能把异常堆栈直接发送给浏览器。
    """


class AgentEventPublisher:
    """管理一次Agent请求的结构化公开事件流。

    每个HTTP诊断请求都必须创建独立实例。

    Publisher内部保存sequence、上一事件、
    活动工具步骤和终端状态，因此不能作为跨请求单例复用。
    """

    def __init__(
        self,
        *,
        request_id: str,
        max_events: int = 200,
    ) -> None:
        """创建一个尚未发布任何事件的Publisher。

        request_id：
            当前HTTP请求的追踪标识。

        max_events：
            单次请求最多允许公开的事件数量，
            用于限制异常Planner循环造成的内存增长。
        """

        if not isinstance(
            request_id,
            str,
        ):
            raise TypeError(
                "request_id必须是字符串"
            )

        cleaned_request_id = (
            request_id.strip()
        )

        if not cleaned_request_id:
            raise ValueError(
                "request_id不能为空"
            )

        if len(cleaned_request_id) > 200:
            raise ValueError(
                "request_id不能超过200个字符"
            )

        # bool是int的子类，所以需要显式拒绝，
        # 防止max_events=True被当成1。
        if (
            isinstance(max_events, bool)
            or not isinstance(max_events, int)
        ):
            raise TypeError(
                "max_events必须是整数"
            )

        if not 1 <= max_events <= 10_000:
            raise ValueError(
                "max_events必须在1到10000之间"
            )

        self._request_id = (
            cleaned_request_id
        )

        self._max_events = max_events

        # sequence保存最后一次成功发布的序号。
        #
        # 初始为0，因此第一条成功事件使用1。
        self._sequence = 0

        # 保存已经成功公开的不可变事件。
        self._events: list[
            AgentSseEvent
        ] = []

        # Queue连接生产事件的业务协程
        # 和消费事件的SSE Router协程。
        #
        # 单次事件总数已经由max_events限制，
        # 因此这里使用不阻塞的内存Queue。
        self._queue: asyncio.Queue[
            AgentSseEvent
        ] = asyncio.Queue()

        # Lock保证多个异步生产方同时调用publish()时，
        # 序号分配和顺序检查仍然是原子操作。
        self._publish_lock = (
            asyncio.Lock()
        )

        # 当前正在执行的工具步骤。
        #
        # None表示现在没有等待tool_finished的工具。
        self._active_tool_step: tuple[
            int,
            str,
        ] | None = None

        # diagnosis_finished或stream_error发布后变为True。
        self._closed = False

        # 当前Queue只支持一个SSE消费者。
        self._stream_started = False

    @property
    def request_id(self) -> str:
        """返回当前事件流所属的请求ID。"""

        return self._request_id

    @property
    def next_sequence(self) -> int:
        """返回下一条成功事件应使用的序号。"""

        return self._sequence + 1

    @property
    def closed(self) -> bool:
        """返回当前事件流是否已经发布终端事件。"""

        return self._closed

    @property
    def events(
        self,
    ) -> tuple[AgentSseEvent, ...]:
        """返回已经发布事件的不可变快照。

        返回tuple而不是内部list，
        防止调用方append、remove或重新排序历史记录。
        """

        return tuple(self._events)

    async def publish(
        self,
        *,
        event_type: AgentSseEventType,
        state: AgentLifecycleState,
        message: str,
        payload: Mapping[
            str,
            Any,
        ] | None = None,
    ) -> AgentSseEvent:
        """校验并发布一条公开事件。

        调用方提供事件类型、生命周期、公开消息
        和该事件特有的payload字段。

        Publisher负责补充：

        1. sequence；
        2. request_id；
        3. schema_version默认值；
        4. 事件顺序检查；
        5. 工具配对检查；
        6. 终端状态检查。
        """

        if (
            payload is not None
            and not isinstance(
                payload,
                Mapping,
            )
        ):
            raise TypeError(
                "payload必须是Mapping或None"
            )

        # 转成普通dict，避免后续依赖调用方
        # 自定义Mapping对象的可变行为。
        event_payload = dict(
            payload or {}
        )

        conflicting_fields = sorted(
            _RESERVED_PAYLOAD_FIELDS
            & set(event_payload)
        )

        if conflicting_fields:
            raise AgentEventPublicationError(
                "payload不能覆盖Publisher控制字段："
                + ", ".join(
                    conflicting_fields
                )
            )

        async with self._publish_lock:
            if self._closed:
                raise AgentEventPublicationError(
                    "终端事件之后不能继续发布事件"
                )

            if (
                len(self._events)
                >= self._max_events
            ):
                raise AgentEventPublicationError(
                    "当前请求的SSE事件数量"
                    "已经达到上限"
                )

            candidate_data = {
                "sequence": (
                    self._sequence + 1
                ),
                "event_type": event_type,
                "request_id": (
                    self._request_id
                ),
                "state": state,
                "message": message,
                **event_payload,
            }

            try:
                # 判别联合会先根据event_type
                # 选择具体事件模型，再校验专属字段。
                event = (
                    _AGENT_SSE_EVENT_ADAPTER
                    .validate_python(
                        candidate_data
                    )
                )
            except ValidationError as exc:
                # 不把candidate_data放入异常信息，
                # 防止上层日志意外记录敏感payload。
                raise AgentEventPublicationError(
                    "SSE事件未通过公开数据契约："
                    f"{event_type}"
                ) from exc

            self._validate_transition(
                event
            )

            # 只有Schema和跨事件规则都通过后，
            # 才提交序号和历史记录。
            self._sequence = event.sequence
            self._events.append(event)

            if isinstance(
                event,
                AgentToolStartedEvent,
            ):
                self._active_tool_step = (
                    event.step_id,
                    event.tool_name,
                )

            if isinstance(
                event,
                AgentToolFinishedEvent,
            ):
                self._active_tool_step = None

            if (
                event.event_type
                in _TERMINAL_EVENT_TYPES
            ):
                self._closed = True

            # put_nowait不会在持有发布锁时等待消费者，
            # 单次事件数量由max_events负责限制。
            self._queue.put_nowait(event)

            return event

    def _validate_transition(
        self,
        event: AgentSseEvent,
    ) -> None:
        """检查当前事件与已发布历史之间的关系。"""

        if not self._events:
            if (
                event.event_type
                != "request_received"
            ):
                raise AgentEventPublicationError(
                    "第一条SSE事件必须是"
                    "request_received"
                )

            return

        previous_event = self._events[-1]

        if (
            event.event_type
            == "request_received"
        ):
            raise AgentEventPublicationError(
                "request_received只能发布一次"
            )

        # stream_error允许在任意非终端阶段结束事件流。
        if event.event_type == "stream_error":
            return

        allowed_predecessors = (
            _ALLOWED_PREDECESSORS[
                event.event_type
            ]
        )

        if (
            previous_event.event_type
            not in allowed_predecessors
        ):
            raise AgentEventPublicationError(
                "非法SSE事件顺序："
                f"{previous_event.event_type}"
                "之后不能发布"
                f"{event.event_type}"
            )

        if (
            event.event_type
            == "tool_scope_decided"
        ):
            if not isinstance(
                previous_event,
                AgentSafetyClassifiedEvent,
            ):
                raise AgentEventPublicationError(
                    "工具范围事件前必须存在"
                    "安全分类事件"
                )

            if (
                previous_event.disposition
                != "continue_to_planner"
            ):
                raise AgentEventPublicationError(
                    "已经拒答或转人工的请求"
                    "不能继续发布工具范围"
                )

        if isinstance(
            event,
            AgentToolStartedEvent,
        ):
            if (
                self._active_tool_step
                is not None
            ):
                raise AgentEventPublicationError(
                    "前一个工具尚未结束，"
                    "不能开始新的工具"
                )

            if not isinstance(
                previous_event,
                AgentPlanningStartedEvent,
            ):
                raise AgentEventPublicationError(
                    "工具开始事件前必须存在"
                    "规划开始事件"
                )

            if (
                event.step_id
                != previous_event.step_number
            ):
                raise AgentEventPublicationError(
                    "工具步骤编号必须与"
                    "规划步骤编号一致"
                )

        if isinstance(
            event,
            AgentToolFinishedEvent,
        ):
            if (
                self._active_tool_step
                is None
            ):
                raise AgentEventPublicationError(
                    "没有活动工具时不能发布"
                    "tool_finished"
                )

            active_step_id, active_tool_name = (
                self._active_tool_step
            )

            if (
                event.trace.step_id
                != active_step_id
            ):
                raise AgentEventPublicationError(
                    "tool_finished步骤编号"
                    "与活动工具不一致"
                )

            if (
                event.trace.tool_name
                != active_tool_name
            ):
                raise AgentEventPublicationError(
                    "tool_finished工具名称"
                    "与活动工具不一致"
                )

        if isinstance(
            event,
            AgentDiagnosisFinishedEvent,
        ):
            if (
                self._active_tool_step
                is not None
            ):
                raise AgentEventPublicationError(
                    "工具尚未结束时不能发布"
                    "最终诊断"
                )

            # 如果安全策略直接结束请求，
            # 最终诊断状态必须与策略处置一致。
            if isinstance(
                previous_event,
                AgentSafetyClassifiedEvent,
            ):
                if (
                    previous_event.disposition
                    == "continue_to_planner"
                ):
                    raise AgentEventPublicationError(
                        "允许继续规划的请求"
                        "不能跳过工具范围和规划"
                    )

                if (
                    event
                    .response
                    .diagnosis
                    .status
                    != previous_event.disposition
                ):
                    raise AgentEventPublicationError(
                        "策略处置与最终诊断状态"
                        "不一致"
                    )

    async def stream(
        self,
    ) -> AsyncIterator[AgentSseEvent]:
        """按照发布顺序异步产生事件。

        这是供未来SSE Router使用的异步生成器。

        读取到diagnosis_finished或stream_error后，
        生成器自动结束。
        """

        if self._stream_started:
            raise AgentEventPublicationError(
                "同一个Publisher只能创建"
                "一个事件消费者"
            )

        self._stream_started = True

        while True:
            event = await self._queue.get()

            yield event

            if (
                event.event_type
                in _TERMINAL_EVENT_TYPES
            ):
                return