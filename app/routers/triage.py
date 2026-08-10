"""机器人故障分诊的 HTTP 路由。"""

import logging

# AsyncIterator 表示异步事件生成器会逐个产生 ServerSentEvent。
from collections.abc import AsyncIterator

# Annotated 用于同时描述参数真实类型和 FastAPI Depends 元数据。
from typing import Annotated

# APIRouter 管理相关 route；Depends 声明依赖注入；Request 表示完整 HTTP 请求。
from fastapi import APIRouter, Depends, Request

# EventSourceResponse 把异步迭代器包装为 text/event-stream HTTP 响应；
# ServerSentEvent 负责把 event 和 data 编码成标准 SSE 文本格式。
from sse_starlette import EventSourceResponse, ServerSentEvent

from app.dependencies import get_triage_service
from app.errors import ApplicationError
from app.exception_handlers import get_error_metadata
from app.schemas.error import ErrorResponse
from app.schemas.error import ErrorDetail
from app.schemas.stream import (
    StreamDeltaData,
    StreamDoneData,
    StreamErrorData,
    StreamMetaData,
)
from app.schemas.triage import TriageRequest, TriageResponse
from app.services.triage import TriageService

# __name__ 是当前模块名，即 app.routers.triage。
# 使用模块名创建日志记录器，便于定位日志来自哪个文件。
logger = logging.getLogger(__name__)

# prefix 会添加到本 Router 每条 route 前；tags 用于 Swagger 分组。
router = APIRouter(
    prefix="/api/v1/triage",
    tags=["triage"],
)


# @router.post() 将下面函数登记为 POST route。
@router.post(
    "",
    # response_model 让 FastAPI 校验成功响应并生成 OpenAPI Schema。
    response_model=TriageResponse,

    # responses 只负责描述可能出现的错误，
    # 不负责真正捕获或处理异常。
    responses={
        502: {
            "model": ErrorResponse,
            "description": "LLM 上游失败或返回无效响应",
        },
        503: {
            "model": ErrorResponse,
            "description": "LLM 配置缺失",
        },
        504: {
            "model": ErrorResponse,
            "description": "LLM 请求超时",
        },
    },
)
async def create_triage(
    # payload 来自 JSON 请求体，FastAPI 会先用 TriageRequest 校验。
    payload: TriageRequest,
    # http_request 是完整 Request，用于读取 Middleware 写入的 state。
    http_request: Request,
    # Depends(get_triage_service) 告诉 FastAPI 创建并注入 TriageService。
    service: Annotated[
        TriageService,
        Depends(get_triage_service),
    ],
) -> TriageResponse:
    """接收故障信息并返回 LLM 分析结果。"""

    # Middleware 已经在 request.state 中保存了请求 ID。
    # 这里将同一个 ID 继续传给 Service。
    # await 会等待异步 LLM 调用完成，而不阻塞事件循环处理其他请求。
    return await service.analyze(
        payload,
        request_id=http_request.state.request_id,
    )


@router.post(
    "/stream",

    # 这个接口返回持续的 SSE 数据流，不是一个完整 JSON 对象。
    # 因此不使用 response_model，而是指定响应类。
    response_class=EventSourceResponse,

    # 这里只描述 SSE 响应开始前可能产生的普通 JSON 错误。
    # 例如依赖注入阶段发现 LLM 配置缺失。
    responses={
        503: {
            "model": ErrorResponse,
            "description": "在 SSE 响应开始前发现 LLM 配置缺失",
        },
    },
)
async def stream_triage(
    # FastAPI 从 JSON 请求体构造并校验 TriageRequest。
    payload: TriageRequest,

    # 完整 HTTP 请求对象，用于读取 request_id 和检查客户端断开。
    http_request: Request,

    # FastAPI 调用 get_triage_service()，
    # 将得到的 TriageService 注入这个参数。
    service: Annotated[
        TriageService,
        Depends(get_triage_service),
    ],
) -> EventSourceResponse:
    """以 meta、delta、done 或 error 事件流式返回故障分析。"""

    # RequestIdMiddleware 已经把 request_id 写入 request.state。
    request_id = http_request.state.request_id

    async def event_generator() -> AsyncIterator[ServerSentEvent]:
        """把 Service 的文本增量转换成客户端可消费的 SSE 事件。"""

        # meta 是流建立后的第一个业务事件。
        # model_dump_json() 把 Pydantic 模型序列化成 JSON 字符串。
        yield ServerSentEvent(
            event="meta",
            data=StreamMetaData(
                request_id=request_id,
            ).model_dump_json(),
        )

        try:
            # stream_analysis() 是异步生成器。
            # async for 在每个文本增量到达后立即继续执行。
            async for text in service.stream_analysis(payload):
                # is_disconnected() 异步检查客户端是否已经关闭连接。
                # 如果客户端断开，就结束生成器，不再继续发送事件。
                if await http_request.is_disconnected():
                    logger.info(
                        "triage_stream_disconnected request_id=%s",
                        request_id,
                    )
                    return

                # 把 Service 产生的纯文本包装成 delta 事件。
                yield ServerSentEvent(
                    event="delta",
                    data=StreamDeltaData(
                        text=text,
                    ).model_dump_json(),
                )

        except ApplicationError as exc:
            # 流开始后不能再修改 HTTP 状态码，
            # 因此把应用异常转换成 SSE error 事件。
            _status_code, error_code, public_message = (
                get_error_metadata(exc)
            )

            # 内部异常详情只写入服务端日志，不发送给客户端。
            logger.warning(
                "triage_stream_error "
                "request_id=%s type=%s detail=%s",
                request_id,
                type(exc).__name__,
                str(exc),
            )

            yield ServerSentEvent(
                event="error",
                data=StreamErrorData(
                    request_id=request_id,
                    error=ErrorDetail(
                        code=error_code,
                        message=public_message,
                    ),
                ).model_dump_json(),
            )

            # error 是本次流的终止事件，后面不能再发送 done。
            return

        # 只有 LLM 流正常结束并通过完整结果校验后，
        # Service 的异步生成器才会正常结束并执行到这里。
        logger.info(
            "triage_stream_completed request_id=%s",
            request_id,
        )

        yield ServerSentEvent(
            event="done",
            data=StreamDoneData().model_dump_json(),
        )

    # 调用 event_generator() 只创建异步生成器对象，
    # 此时生成器函数体尚未执行。
    #
    # EventSourceResponse 开始发送响应时，
    # 才会逐次迭代生成器并发送其中的事件。
    return EventSourceResponse(event_generator())
