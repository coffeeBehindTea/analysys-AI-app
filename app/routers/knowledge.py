"""知识库文档与问答 HTTP 路由。"""

# Annotated 将真实参数类型与 Depends 元数据组合在一起。
from typing import Annotated

# APIRouter 用于组织一组相关接口；
# Depends 声明参数由 FastAPI 依赖系统提供。
from fastapi import (
    APIRouter,
    Depends,

    # File 声明参数来自 multipart 文件字段，
    # 而不是 JSON 请求体。
    File,

    # 使用别名避免与 pathlib.Path 混淆。
    Path as PathParameter,

    # UploadFile 表示 FastAPI 接收到的上传文件。
    UploadFile,

    # status 提供具名 HTTP 状态码。
    status,
)

from app.config import Settings, get_settings
from app.dependencies import (
    get_document_catalog_service,
    get_document_service,
    get_knowledge_query_service,
)

from app.schemas.error import ErrorResponse
from app.schemas.knowledge import (
    DocumentDeleteResponse,
    DocumentListResponse,
    DocumentRecord,
)
from app.schemas.knowledge_query import (
    KnowledgeQueryRequest,
    KnowledgeQueryResponse,
)

from app.services.document_catalog import (
    DocumentCatalogService,
)
from app.services.document_service import (
    DocumentService,
)
from app.services.upload_staging import (
    stage_uploaded_file,
)
from app.services.knowledge_query import (
    KnowledgeQueryService,
)

# prefix 会添加到本文件中所有接口路径前。
#
# tags 控制 Swagger UI 中的接口分组名称。
router = APIRouter(
    prefix="/api/v1/knowledge",
    tags=["knowledge"],
)


@router.post(
    "/documents",

    # 文档成功完成全部摄取后返回 201 Created。
    #
    # 如果摄取中途失败，不会返回 201，
    # 也不会产生 ready 状态的 DocumentRecord。
    status_code=status.HTTP_201_CREATED,

    # FastAPI 使用 DocumentRecord：
    # 1. 校验 Router 的返回结果；
    # 2. 序列化响应 JSON；
    # 3. 生成 Swagger 响应文档。
    response_model=DocumentRecord,

    # responses 只描述可能出现的应用错误。
    #
    # 真正的异常捕获和 JSON 转换仍由
    # application_error_handler() 完成。
    responses={
        409: {
            "model": ErrorResponse,
            "description": "相同文档已经存在",
        },
        502: {
            "model": ErrorResponse,
            "description": "Embedding 上游失败或响应无效",
        },
        503: {
            "model": ErrorResponse,
            "description": "Embedding 配置缺失或知识库不可用",
        },
        504: {
            "model": ErrorResponse,
            "description": "Embedding 请求超时",
        },
    },
)
async def upload_knowledge_document(
    # File() 告诉 FastAPI：
    # file 来自 multipart/form-data 中名为 file 的字段。
    #
    # UploadFile 提供 filename、content_type、
    # read()、seek() 和 close() 等成员。
    file: Annotated[
        UploadFile,
        File(
            description=(
                "待摄取的 PDF、Markdown 或 TXT 文档"
            ),
        ),
    ],

    # FastAPI 调用 get_document_service()，
    # 装配真实摄取服务并注入该参数。
    service: Annotated[
        DocumentService,
        Depends(get_document_service),
    ],

    # 暂存层需要知道允许上传的最大字节数。
    #
    # 它与 DocumentService 使用同一份 Settings，
    # 避免两处大小限制不一致。
    settings: Annotated[
        Settings,
        Depends(get_settings),
    ],
) -> DocumentRecord:
    """上传并同步摄取一份知识库文档。"""

    # async with 的有效期覆盖整个摄取过程。
    #
    # 因此调用 ingest_document() 时临时文件一定存在；
    # 摄取成功或失败后，临时文件都会被删除。
    async with stage_uploaded_file(
        file,
        max_file_size_bytes=(
            settings.max_document_size_bytes
        ),
    ) as staged_path:
        # ingest_document() 是异步方法，因为内部需要
        # await Embedding Provider 的网络请求。
        record = await service.ingest_document(
            staged_path
        )

    # 离开 async with 时临时文件已经被清理。
    #
    # DocumentRecord 只保存文件名、哈希和统计信息，
    # 不依赖临时文件继续存在。
    return record



@router.get(
    "/documents",

    # FastAPI 会用 DocumentListResponse：
    # 1. 校验 Router 返回的数据；
    # 2. 序列化 JSON；
    # 3. 生成 OpenAPI/Swagger Schema。
    response_model=DocumentListResponse,
)
def list_knowledge_documents(
    service: Annotated[
        DocumentCatalogService,
        Depends(get_document_catalog_service),
    ],
) -> DocumentListResponse:
    """列出当前知识库中已经完成摄取的文档。"""

    # list_documents() 返回 list[DocumentRecord]。
    records = service.list_documents()

    # 使用明确的响应模型包裹记录，
    # 防止 Router 直接返回存储层内部对象结构。
    return DocumentListResponse(
        documents=records,
    )


@router.delete(
    "/documents/{document_id}",

    # 成功响应必须满足 DocumentDeleteResponse。
    response_model=DocumentDeleteResponse,

    # responses 只为 OpenAPI 描述可能的错误响应。
    # 真正的异常转换由 application_error_handler 完成。
    responses={
        404: {
            "model": ErrorResponse,
            "description": "指定文档不存在",
        },
    },
)
def delete_knowledge_document(
    # document_id 来自 URL 路径。
    #
    # PathParameter 为它添加 SHA-256 格式校验。
    # 不符合格式时，FastAPI 会在调用 Service 前返回 422。
    document_id: Annotated[
        str,
        PathParameter(
            pattern=r"^[0-9a-f]{64}$",
            description="待删除文档的 SHA-256 标识",
        ),
    ],

    # FastAPI 注入只依赖本地 Chroma 的目录服务。
    service: Annotated[
        DocumentCatalogService,
        Depends(get_document_catalog_service),
    ],
) -> DocumentDeleteResponse:
    """删除指定文档产生的全部 Chunk。"""

    # 成功时该方法返回 None；
    # 不存在时抛出 DocumentNotFoundError。
    service.delete_document(document_id)

    return DocumentDeleteResponse(
        document_id=document_id,
        deleted=True,
    )


@router.post(
    "/query",

    # 成功时返回知识库问答响应。
    response_model=KnowledgeQueryResponse,

    # responses只用于OpenAPI文档；
    # 真正异常转换由application_error_handler完成。
    responses={
        502: {
            "model": ErrorResponse,
            "description": (
                "Embedding或LLM上游失败，"
                "或者返回无效响应"
            ),
        },
        503: {
            "model": ErrorResponse,
            "description": (
                "Embedding、LLM配置缺失，"
                "或者知识库暂时不可用"
            ),
        },
        504: {
            "model": ErrorResponse,
            "description": (
                "Embedding或LLM请求超时"
            ),
        },
    },
)
async def query_knowledge(
    # Pydantic模型且没有其他来源标记，
    # 因此FastAPI将它解析为JSON请求体。
    query_request: KnowledgeQueryRequest,

    # FastAPI通过依赖图装配真实RAG Service。
    service: Annotated[
        KnowledgeQueryService,
        Depends(get_knowledge_query_service),
    ],
) -> KnowledgeQueryResponse:
    """根据知识库证据回答自然语言问题。"""

    return await service.answer_query(
        query_request
    )