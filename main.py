"""FastAPI 应用的装配入口。

这个模块只负责创建应用、注册异常处理器、Middleware 和 Router，
不在这里实现具体的机器人故障分析业务。
"""

from pathlib import Path

# FastAPI 是应用主类。实例化后得到整个 Web 应用对象。
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

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
# 导入Robot Diagnostic Agent路由。
#
# 此处只是取得已经定义好的APIRouter对象，
# 不会执行Agent、工具或LLM请求。
from app.routers.agent import (
    router as agent_router,
)
from app.routers.diagnostic_sessions import (
    router as diagnostic_sessions_router,
)


# 使用main.py所在目录作为项目根目录，构造Web静态文件绝对路径。
#
# 不依赖启动命令的当前工作目录，原因是：
#
# 1. 本地可能从项目根目录执行Uvicorn；
# 2. Docker中的WORKDIR可能与本机路径不同；
# 3. IDE和测试工具也可能使用不同的当前目录。
#
# resolve()把路径转换成规范绝对路径，parent取得main.py所在目录，
# /运算符由pathlib.Path重载，用于安全拼接子目录。
WEB_CONSOLE_DIRECTORY = (
    Path(__file__).resolve().parent
    / "app"
    / "web"
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
    # 把Agent Router中的路径合并到FastAPI主应用。
    #
    # 注册后主应用才能识别：
    # POST /api/v1/agent/diagnose
    application.include_router(
        agent_router
    )

    # 注册诊断会话只读查询接口：
    #
    # GET /api/v1/diagnostic-sessions
    # GET /api/v1/diagnostic-sessions/{session_id}
    #
    # Router本身不读取文件，真实Store由FastAPI依赖注入。
    application.include_router(
        diagnostic_sessions_router
    )

    # 将app/web挂载到/console。
    #
    # StaticFiles只负责读取和返回静态文件，不会执行其中的
    # JavaScript，也不会获得Agent Service或其他业务依赖。
    #
    # html=True使/console/自动查找index.html。
    # 浏览器随后会继续请求/console/styles.css和/console/app.js。
    application.mount(
        "/console",
        StaticFiles(
            directory=WEB_CONSOLE_DIRECTORY,
            html=True,
        ),
        name="web_console",
    )

    return application


# Uvicorn 通过 main:app 查找这个应用对象。
app = create_app()
