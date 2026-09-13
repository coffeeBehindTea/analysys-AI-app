"""证据约束结构化诊断的HTTP路由。

本模块负责：

1. 声明POST /api/v1/diagnostics；
2. 让FastAPI校验DiagnosisRequest请求体；
3. 通过依赖注入取得DiagnosisService；
4. 从Request.state读取Middleware生成的request_id；
5. 调用Service并返回DiagnosisReport。

本模块不实现检索、Embedding、门控、
Prompt、LLM调用或报告组装。
"""

# Annotated把参数的真实Python类型
# 与FastAPI Depends元数据组合在一起。
from typing import Annotated

# APIRouter用于组织相关接口；
# Depends声明由FastAPI提供依赖；
# Request表示当前完整HTTP请求。
from fastapi import (
    APIRouter,
    Depends,
    Request,
)

from app.dependencies import (
    get_diagnosis_service,
)
from app.schemas.diagnostics import (
    DiagnosisReport,
    DiagnosisRequest,
)
from app.schemas.error import (
    ErrorResponse,
)
from app.services.diagnosis_service import (
    DiagnosisService,
)


# prefix会自动添加到本文件中的所有接口路径前。
#
# 当前接口装饰器使用空字符串路径，
# 因此最终路径就是：
#
# POST /api/v1/diagnostics
#
# tags控制Swagger UI中的接口分组。
router = APIRouter(
    prefix="/api/v1/diagnostics",
    tags=["diagnostics"],
)


@router.post(
    "",

    # FastAPI使用DiagnosisReport：
    #
    # 1. 校验Router的成功返回值；
    # 2. 将Pydantic对象序列化成JSON；
    # 3. 生成OpenAPI响应Schema；
    # 4. 防止返回未声明的内部字段。
    response_model=DiagnosisReport,

    # responses只描述可能发生的非成功响应，
    # 不负责捕获或转换异常。
    #
    # 已知的LLM生成失败会由DiagnosisService
    # 转换成200状态的abstained报告，
    # 因此这里的错误主要来自依赖装配、
    # Embedding检索或VectorStore。
    responses={
        502: {
            "model": ErrorResponse,
            "description": (
                "Embedding上游失败"
                "或返回无效响应"
            ),
        },
        503: {
            "model": ErrorResponse,
            "description": (
                "Embedding或LLM配置缺失，"
                "或者知识库暂时不可用"
            ),
        },
        504: {
            "model": ErrorResponse,
            "description": (
                "Embedding请求超时"
            ),
        },
    },
)
async def create_diagnosis(
    # 没有Query、Path或Header等来源标记的
    # Pydantic模型会被FastAPI解析为JSON请求体。
    payload: DiagnosisRequest,

    # Request由FastAPI直接提供，
    # 不属于客户端JSON请求体。
    http_request: Request,

    # FastAPI解析Depends后会：
    #
    # 1. 创建Settings；
    # 2. 创建Embedding和LLM客户端；
    # 3. 打开VectorStore；
    # 4. 调用get_diagnosis_service()；
    # 5. 将得到的DiagnosisService注入此参数。
    service: Annotated[
        DiagnosisService,
        Depends(get_diagnosis_service),
    ],
) -> DiagnosisReport:
    """根据请求和真实知识库证据生成结构化诊断。"""

    # RequestIdMiddleware在Router执行之前，
    # 已经向当前请求的state容器写入request_id。
    #
    # diagnose()是异步方法，因为下游可能执行
    # Embedding和LLM网络请求，所以必须await。
    return await service.diagnose(
        payload,
        request_id=(
            http_request.state.request_id
        ),
    )