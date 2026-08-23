"""FastAPI 应用的装配入口。

这个模块只负责创建应用、注册异常处理器、Middleware 和 Router，
不在这里实现具体的机器人故障分析业务。
"""

# FastAPI 是应用主类。实例化后得到整个 Web 应用对象。
from fastapi import FastAPI

from app.errors import ApplicationError
from app.exception_handlers import application_error_handler
from app.middleware.request_id import RequestIdMiddleware
from app.routers.health import router as health_router
from app.routers.triage import router as triage_router
from app.routers.knowledge import (
    router as knowledge_router,
)
from app.routers.diagnostics import (
    router as diagnostics_router,
)

def create_app() -> FastAPI:
    """创建并装配完整的 FastAPI 应用。"""

    # FastAPI(...) 创建应用对象；title 和 version 会显示在 Swagger 文档中。
    application = FastAPI(
        title="Robot Incident Triage API",
        version="0.1.0",
    )

    # 注册应用异常的统一处理器。
    #
    # ApplicationError 是所有可预期应用异常的父类，
    # 所以它可以统一处理 LLMTimeoutError 等子类。
    # add_exception_handler(异常类型, 处理函数) 建立异常到响应函数的映射。
    application.add_exception_handler(
        ApplicationError,
        application_error_handler,
    )

    # 注册请求 ID Middleware。
    # 它会在请求进入时生成 ID，在响应离开时添加响应头。
    # add_middleware() 将 Middleware 类加入应用的请求处理链。
    application.add_middleware(RequestIdMiddleware)

    # 将各业务 Router 安装到主应用。
    # include_router() 把 APIRouter 中登记的路由合并到主应用路由表。
    application.include_router(health_router)
    application.include_router(triage_router)
    # 注册知识库文档和 RAG 查询接口。
    application.include_router(knowledge_router)
    # 注册证据约束结构化诊断接口。
    application.include_router(
        diagnostics_router
    )

    return application


# Uvicorn 通过 main:app 查找这个应用对象。
app = create_app()
