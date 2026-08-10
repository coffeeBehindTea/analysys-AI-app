"""将应用异常统一转换成安全、稳定的 JSON HTTP 响应。"""

# logging 是 Python 标准库，用于记录服务端诊断信息。
import logging

# Request 提供 request.state 等当前 HTTP 请求上下文。
from fastapi import Request

# JSONResponse 是 Response 子类，会把 Python 数据编码为 UTF-8 JSON。
from fastapi.responses import JSONResponse

# status 模块提供带名称的 HTTP 状态码常量，例如 HTTP_502_BAD_GATEWAY == 502。
from starlette import status

from app.errors import (
    ApplicationError,
    InvalidLLMResponseError,
    LLMConfigurationError,
    LLMTimeoutError,
    LLMUpstreamError,
)
from app.schemas.error import ErrorDetail, ErrorResponse


# 创建当前模块使用的日志记录器。
logger = logging.getLogger(__name__)


# 每项元数据的结构依次是：
# HTTP 状态码、稳定错误码、向客户端公开的安全信息。
ErrorMetadata = tuple[int, str, str]


# 将业务层的应用异常转换为 HTTP 层可以理解的信息。
ERROR_METADATA: dict[type[ApplicationError], ErrorMetadata] = {
    LLMConfigurationError: (
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "llm_configuration_error",
        "LLM 服务未正确配置",
    ),
    LLMTimeoutError: (
        status.HTTP_504_GATEWAY_TIMEOUT,
        "llm_timeout",
        "LLM 服务响应超时",
    ),
    LLMUpstreamError: (
        status.HTTP_502_BAD_GATEWAY,
        "llm_upstream_error",
        "LLM 上游服务暂时不可用",
    ),
    InvalidLLMResponseError: (
        status.HTTP_502_BAD_GATEWAY,
        "invalid_llm_response",
        "LLM 返回了无法处理的响应",
    ),
}


def get_error_metadata(exc: ApplicationError) -> ErrorMetadata:
    """把应用异常转换为 HTTP 状态码、公开错误码和安全信息。"""

    # dict.get(key, default) 在异常类型未登记时返回安全的 500 兜底值。
    # 单独封装成函数后，JSON 异常处理器和 SSE error 事件可以复用同一映射。
    return ERROR_METADATA.get(
        type(exc),
        (
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "application_error",
            "应用处理请求时发生错误",
        ),
    )


async def application_error_handler(
    request: Request,
    exc: ApplicationError,
) -> JSONResponse:
    """将可预期的应用异常转换成统一 JSON 响应。"""

    # 根据具体异常类型查找对应的 HTTP 状态和公开信息。
    #
    # type(exc) 可能是：
    # LLMTimeoutError、LLMUpstreamError 等具体异常类。
    # dict.get(key, default) 查找异常类型；没有映射时使用安全的 500 兜底值。
    status_code, error_code, public_message = get_error_metadata(exc)

    # 正常情况下，RequestIdMiddleware 已经设置了 request_id。
    # getattr 提供兜底，防止请求 ID 尚未设置时再次触发异常。
    # getattr(对象, 属性名, 默认值) 可以避免属性不存在时抛出 AttributeError。
    request_id = getattr(
        request.state,
        "request_id",
        "unavailable",
    )

    # 服务端记录相对详细的内部错误。
    # str(exc) 是 Service 抛出的错误信息，不直接返回给客户端。
    # logger.warning() 写入 WARNING 级别日志；%s 由日志模块延迟格式化。
    logger.warning(
        "application_error request_id=%s type=%s detail=%s",
        request_id,
        type(exc).__name__,
        str(exc),
    )

    # 使用 Pydantic 构造符合错误契约的对象。
    error_response = ErrorResponse(
        request_id=request_id,
        error=ErrorDetail(
            code=error_code,
            message=public_message,
        ),
    )

    # JSONResponse 将错误对象转换成真正的 HTTP 响应。
    # model_dump(mode="json") 把 Pydantic 模型转换成适合 JSON 序列化的 dict。
    # JSONResponse 再将 dict 编码为响应体，并设置 application/json 媒体类型。
    return JSONResponse(
        status_code=status_code,
        content=error_response.model_dump(mode="json"),
    )
