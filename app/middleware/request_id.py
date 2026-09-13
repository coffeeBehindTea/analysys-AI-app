"""为每个 HTTP 请求生成和传递唯一 request_id。"""

# uuid4() 是 Python 标准库函数，返回随机 UUID 对象。
from uuid import uuid4

# Request 表示完整 HTTP 请求；FastAPI 重新导出了 Starlette 的 Request。
from fastapi import Request

# BaseHTTPMiddleware 是 Starlette 的 HTTP Middleware 基类。
# RequestResponseEndpoint 是类型别名：
# Callable[[Request], Awaitable[Response]]，即接收 Request、异步返回 Response 的函数。
from starlette.middleware.base import (
    BaseHTTPMiddleware,
    RequestResponseEndpoint,
)

# Response 表示准备发送给客户端的 HTTP 响应。
from starlette.responses import Response


class RequestIdMiddleware(BaseHTTPMiddleware):
    """在请求进入时生成 ID，并在响应离开时写入响应头。"""

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """处理一次 HTTP 请求并返回添加了追踪头的 Response。

        call_next 由 BaseHTTPMiddleware 提供。
        调用它会继续执行后续 Middleware、依赖、Router 和 Service。
        """

        # str() 把 UUID 对象变成可写入 JSON 与 Header 的 ASCII 字符串。
        request_id = str(uuid4())

        # request.state 是当前请求专属的动态 State 容器。
        # request_id 不是 Request 预定义字段，而是在 state 中新增的属性。
        request.state.request_id = request_id

        # call_next(request) 返回 Awaitable[Response]；await 后得到最终 Response。
        response = await call_next(request)

        # response.headers 是 Starlette 的 MutableHeaders 可修改容器。
        # 此赋值会新增或覆盖 X-Request-ID 响应头。
        response.headers["X-Request-ID"] = request_id

        # 返回后由 Starlette/Uvicorn 将响应真正发送给客户端。
        return response
