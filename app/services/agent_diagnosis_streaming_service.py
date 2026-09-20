"""Agent诊断业务与SSE事件消费之间的异步编排服务。

本模块负责：

1. 为每次流式诊断创建独立的AgentEventPublisher；
2. 创建绑定该Publisher的AgentRunnerSseObserver；
3. 在后台异步任务中运行完整AgentDiagnosisService；
4. 通过异步迭代器实时返回结构化Agent SSE事件；
5. 把流开始后的应用异常转换成安全的stream_error；
6. 在客户端停止消费时取消仍在运行的诊断任务；
7. 防止诊断服务无终端事件返回而导致SSE永久等待。

本模块不负责：

1. 执行Planner或Agent工具；
2. 决定安全策略和工具权限；
3. 构造最终诊断报告；
4. 保存诊断会话；
5. 把结构化事件编码成SSE文本；
6. 创建FastAPI StreamingResponse。
"""

import asyncio
import inspect
import logging

from collections.abc import (
    AsyncIterator,
)
from contextlib import (
    suppress,
)
from typing import (
    Protocol,
)

from app.agent.diagnosis_observer import (
    AgentDiagnosisObserver,
)
from app.errors import (
    ApplicationError,
)
from app.exception_handlers import (
    get_error_metadata,
)
from app.schemas.agent_api import (
    AgentDiagnosisRequest,
    AgentDiagnosisResponse,
)
from app.schemas.agent_events import (
    AgentSseEvent,
)
from app.schemas.error import (
    ErrorDetail,
)
from app.services.agent_event_publisher import (
    AgentEventPublisher,
)
from app.services.agent_runner_sse_observer import (
    AgentRunnerSseObserver,
)


logger = logging.getLogger(__name__)


# 这些错误通常来自暂时性的上游服务状态。
#
# retryable=True只表示客户端可以在不改变请求内容的情况下
# 尝试重新建立一次诊断，不表示重试一定成功。
#
# 配置缺失、图片输入错误和内部程序错误不在该集合中，
# 因为重复发送相同请求通常不能修复这些问题。
_RETRYABLE_STREAM_ERROR_CODES = frozenset({
    "llm_timeout",
    "llm_upstream_error",
    "invalid_llm_response",
    "embedding_timeout",
    "embedding_upstream_error",
    "invalid_embedding_response",
    "vision_timeout",
    "vision_upstream_error",
    "invalid_vision_response",
    "ocr_timeout",
    "ocr_execution_error",
    "invalid_ocr_response",
    "vector_store_error",
    "diagnostic_session_store_error",
})


class AgentDiagnosisStreamingProvider(
    Protocol
):
    """流式编排服务依赖的最小诊断服务协议。

    使用Protocol而不是把构造参数绑定到唯一具体类，
    可以在离线测试中注入Fake诊断服务。

    真实运行时由AgentDiagnosisService满足这个协议。
    """

    async def diagnose(
        self,
        request: AgentDiagnosisRequest,
        request_id: str,
        *,
        observer: (
            AgentDiagnosisObserver | None
        ) = None,
    ) -> AgentDiagnosisResponse:
        """执行完整诊断并向Observer发送生命周期通知。"""

        ...


class AgentDiagnosisStreamingService:
    """协调诊断事件生产和SSE事件消费。

    一个实例可以保存无状态的诊断服务依赖，
    但每次stream()调用都会创建新的Publisher和Observer。

    Publisher不能跨请求复用，因为它内部保存：

    1. 当前请求ID；
    2. 事件sequence；
    3. 上一条事件；
    4. 当前活动工具；
    5. 是否已经发布终端事件；
    6. 当前请求的异步事件队列。
    """

    def __init__(
        self,
        *,
        diagnosis_service: (
            AgentDiagnosisStreamingProvider
        ),
        max_events: int = 200,
    ) -> None:
        """注入真实诊断服务并设置单次事件数量上限。

        max_events至少为2，因为异常发生在第一条事件之前时，
        编排服务需要先发布request_received，再发布stream_error。
        """

        diagnose_method = getattr(
            diagnosis_service,
            "diagnose",
            None,
        )

        if not inspect.iscoroutinefunction(
            diagnose_method
        ):
            raise TypeError(
                "diagnosis_service.diagnose"
                "必须是异步方法"
            )

        # bool是int的子类，因此需要显式拒绝。
        if (
            isinstance(max_events, bool)
            or not isinstance(max_events, int)
        ):
            raise TypeError(
                "max_events必须是整数"
            )

        if not 2 <= max_events <= 10_000:
            raise ValueError(
                "max_events必须在2到10000之间"
            )

        self._diagnosis_service = (
            diagnosis_service
        )
        self._max_events = max_events

    async def stream(
        self,
        request: AgentDiagnosisRequest,
        request_id: str,
    ) -> AsyncIterator[AgentSseEvent]:
        """运行一次诊断并实时产生结构化公开事件。

        调用本方法不会一次性返回list。

        调用方应使用：

        async for event in service.stream(...)

        每次循环只取得当前已经产生的一条事件。
        后续FastAPI Router会把每个event交给
        encode_agent_sse_event()编码成SSE文本。
        """

        if not isinstance(
            request,
            AgentDiagnosisRequest,
        ):
            raise TypeError(
                "request必须是"
                "AgentDiagnosisRequest"
            )

        # Publisher构造器会清理和验证request_id。
        #
        # 每次stream()调用创建独立Publisher，
        # 保证不同HTTP请求不会共享序号和事件队列。
        publisher = AgentEventPublisher(
            request_id=request_id,
            max_events=self._max_events,
        )

        observer = AgentRunnerSseObserver(
            publisher=publisher,
        )

        # create_task()把协程登记到当前事件循环中运行。
        #
        # 诊断任务负责产生事件；
        # 当前stream()异步生成器负责消费事件。
        producer_task = asyncio.create_task(
            self._produce_events(
                request=request,
                request_id=publisher.request_id,
                publisher=publisher,
                observer=observer,
            ),
            name=(
                "agent-diagnosis-stream:"
                f"{publisher.request_id}"
            ),
        )

        try:
            # publisher.stream()是异步生成器。
            #
            # 队列暂时为空时，async for会等待新事件，
            # 但不会阻塞整个FastAPI事件循环。
            async for event in publisher.stream():
                yield event

        finally:
            # 客户端断开、Router取消响应或调用方提前停止
            # async for时，异步生成器会进入finally。
            #
            # 如果诊断仍在运行，就取消不再需要的下游等待。
            if not producer_task.done():
                producer_task.cancel()

            # await确保后台任务完成资源清理。
            #
            # 客户端断开导致的CancelledError属于正常清理路径，
            # 不应再向已经断开的连接发布stream_error。
            with suppress(
                asyncio.CancelledError
            ):
                await producer_task

    async def _produce_events(
        self,
        *,
        request: AgentDiagnosisRequest,
        request_id: str,
        publisher: AgentEventPublisher,
        observer: AgentRunnerSseObserver,
    ) -> None:
        """运行诊断生产者并保证事件流出现终端事件。

        正常情况下，AgentDiagnosisService会通过Observer
        发布diagnosis_finished。

        已知应用异常会转换为保留稳定错误码的stream_error。
        未知程序异常只公开统一内部错误，不公开异常详情。
        """

        try:
            response = await (
                self._diagnosis_service.diagnose(
                    request,
                    request_id,
                    observer=observer,
                )
            )

            if not isinstance(
                response,
                AgentDiagnosisResponse,
            ):
                raise TypeError(
                    "diagnosis_service.diagnose"
                    "必须返回AgentDiagnosisResponse"
                )

            # 完整诊断Service应该已经通过Observer发布
            # diagnosis_finished，并让Publisher进入closed状态。
            #
            # 如果Service返回了响应却没有终端事件，
            # 不能让消费者永久等待，也不能在这里擅自补造
            # diagnosis_finished，因为这表示Observer接线存在错误。
            if not publisher.closed:
                raise RuntimeError(
                    "诊断服务返回时没有发布"
                    "SSE终端事件"
                )

        except asyncio.CancelledError:
            # 客户端断开时，stream()会取消生产任务。
            #
            # 必须继续抛出CancelledError，
            # 让下游Planner、网络请求和工具等待真正停止。
            raise

        except ApplicationError as exc:
            # 复用普通JSON异常处理器使用的稳定错误映射，
            # 避免JSON接口和SSE接口产生两套错误码。
            (
                _http_status,
                error_code,
                public_message,
            ) = get_error_metadata(exc)

            logger.warning(
                "agent_stream_application_error "
                "request_id=%s type=%s",
                request_id,
                type(exc).__name__,
            )

            await self._publish_stream_error(
                request=request,
                publisher=publisher,
                observer=observer,
                error=ErrorDetail(
                    code=error_code,
                    message=public_message,
                ),
                retryable=(
                    error_code
                    in _RETRYABLE_STREAM_ERROR_CODES
                ),
            )

        except Exception as exc:
            # 未知异常属于程序错误。
            #
            # 服务端日志保留异常类型和堆栈，
            # 但SSE事件不能公开str(exc)，防止泄露路径、
            # SDK内容、请求数据或其他内部实现。
            logger.exception(
                "agent_stream_unexpected_error "
                "request_id=%s type=%s",
                request_id,
                type(exc).__name__,
            )

            await self._publish_stream_error(
                request=request,
                publisher=publisher,
                observer=observer,
                error=ErrorDetail(
                    code="internal_stream_error",
                    message=(
                        "诊断流处理过程中发生内部错误"
                    ),
                ),
                retryable=False,
            )

    async def _publish_stream_error(
        self,
        *,
        request: AgentDiagnosisRequest,
        publisher: AgentEventPublisher,
        observer: AgentRunnerSseObserver,
        error: ErrorDetail,
        retryable: bool,
    ) -> None:
        """发布唯一的安全错误终端事件。

        如果最终诊断已经发布，Publisher已经关闭，
        此方法不会再尝试发布第二个终端事件。

        如果异常发生在第一条事件发布之前，
        会先补发真实的request_received事件，
        从而继续满足公开事件协议的首事件约束。
        """

        if publisher.closed:
            return

        if not publisher.events:
            await observer.on_request_received(
                image_count=len(request.images),
                has_task_goal=(
                    request.task_goal is not None
                ),
            )

        await publisher.publish(
            event_type="stream_error",
            state="aborted",
            message="诊断事件流已经安全终止",
            payload={
                # model_copy(deep=True)避免调用方和事件
                # 共享同一个可变Pydantic对象引用。
                "error": error.model_copy(
                    deep=True
                ),
                "retryable": retryable,
            },
        )