"""诊断会话与脱敏运行轨迹的只读HTTP接口。

本模块负责：

1. 声明最近会话列表接口；
2. 声明按session_id查询完整会话接口；
3. 让FastAPI校验limit和UUID4路径参数；
4. 通过依赖注入取得DiagnosticSessionStore；
5. 将不存在的合法会话ID转换成应用级404异常。

本模块不负责：

1. 执行Agent诊断；
2. 构造或保存诊断会话；
3. 直接读取JSON文件；
4. 返回完整图片、原始日志、密钥或模型私有思维链。
"""

from typing import (
    Annotated,
)

from fastapi import (
    APIRouter,
    Depends,
    Query,
)
from pydantic import (
    UUID4,
)

from app.dependencies import (
    get_diagnostic_session_store,
)
from app.errors import (
    DiagnosticSessionNotFoundError,
)
from app.schemas.diagnostic_session import (
    DiagnosticSessionListResponse,
    DiagnosticSessionRecord,
)
from app.schemas.error import (
    ErrorResponse,
)
from app.services.diagnostic_session_store import (
    DiagnosticSessionStore,
)


# prefix会自动添加到本文件声明的所有路由前面。
#
# 因此：
#
# @router.get("")
# 对应GET /api/v1/diagnostic-sessions
#
# @router.get("/{session_id}")
# 对应GET /api/v1/diagnostic-sessions/{session_id}
router = APIRouter(
    prefix="/api/v1/diagnostic-sessions",
    tags=["diagnostic-sessions"],
)


@router.get(
    "",

    # response_model要求FastAPI：
    #
    # 1. 校验Router返回的数据；
    # 2. 按照Schema过滤响应字段；
    # 3. 序列化datetime和嵌套Pydantic模型；
    # 4. 为OpenAPI和Swagger生成响应文档。
    response_model=DiagnosticSessionListResponse,

    # 这里声明的是可能发生的存储错误。
    #
    # responses只生成API文档，
    # 真正的异常捕获仍由application_error_handler完成。
    responses={
        500: {
            "model": ErrorResponse,
            "description": (
                "存储中的会话记录损坏"
            ),
        },
        503: {
            "model": ErrorResponse,
            "description": (
                "诊断会话存储暂时不可用"
            ),
        },
    },
)
async def list_diagnostic_sessions(
    # Query表示limit来自URL查询字符串：
    #
    # GET /api/v1/diagnostic-sessions?limit=20
    #
    # ge=1表示至少为1；
    # le=100表示最多为100；
    # FastAPI会在调用函数前完成类型转换和范围校验。
    limit: Annotated[
        int,
        Query(
            ge=1,
            le=100,
            description=(
                "最多返回多少条最近会话摘要"
            ),
        ),
    ] = 20,

    # Depends告诉FastAPI：
    #
    # store不是客户端提交的参数，
    # 而是由get_diagnostic_session_store()
    # 根据Settings装配并注入。
    store: Annotated[
        DiagnosticSessionStore,
        Depends(
            get_diagnostic_session_store
        ),
    ] = None,
) -> DiagnosticSessionListResponse:
    """按照创建时间从新到旧查询会话摘要。

    列表只返回DiagnosticSessionSummary，
    不返回完整工具轨迹、引用正文和视觉观察。
    """

    # Store的list_recent()是异步方法。
    #
    # 它会把同步文件读取放入工作线程，
    # 因此这里必须使用await。
    return await store.list_recent(
        limit=limit
    )


@router.get(
    "/{session_id}",

    # 详情接口返回完整的脱敏会话记录。
    #
    # “完整”是相对于列表摘要而言，
    # 仍然不包含原始图片、完整日志、密钥
    # 和Planner私有思维链。
    response_model=DiagnosticSessionRecord,

    responses={
        404: {
            "model": ErrorResponse,
            "description": (
                "指定的诊断会话不存在"
            ),
        },
        500: {
            "model": ErrorResponse,
            "description": (
                "诊断会话记录损坏"
            ),
        },
        503: {
            "model": ErrorResponse,
            "description": (
                "诊断会话存储暂时不可用"
            ),
        },
    },
)
async def get_diagnostic_session(
    # UUID4让FastAPI在进入函数前验证路径参数。
    #
    # 合法URL示例：
    #
    # /api/v1/diagnostic-sessions/
    # 123e4567-e89b-42d3-a456-426614174000
    #
    # 不合法的普通字符串会直接得到422，
    # 不会传给文件存储层。
    session_id: UUID4,

    store: Annotated[
        DiagnosticSessionStore,
        Depends(
            get_diagnostic_session_store
        ),
    ],
) -> DiagnosticSessionRecord:
    """根据session_id读取一条完整脱敏会话。"""

    # Pydantic把路径字符串转换成UUID对象。
    #
    # Store的公开契约接收字符串，
    # str()会把UUID转换成规范的小写带连字符形式。
    record = await store.get(
        str(session_id)
    )

    if record is None:
        # 这里不直接构造JSONResponse。
        #
        # 抛出ApplicationError子类后，
        # 全局异常处理器会统一添加request_id、
        # 稳定错误码和安全公开信息。
        raise DiagnosticSessionNotFoundError(
            "诊断会话不存在："
            f"{session_id}"
        )

    return record