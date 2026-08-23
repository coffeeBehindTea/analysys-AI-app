"""知识库 HTTP API 的离线契约测试。"""

from pathlib import Path

import pytest

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.config import Settings, get_settings
from app.dependencies import (
    get_document_catalog_service,
    get_document_service,
    get_knowledge_query_service,
)
from app.errors import (
    DocumentNotFoundError,
    DuplicateDocumentError,
    LLMTimeoutError,
)
from app.schemas.knowledge import DocumentRecord
from app.schemas.knowledge_query import (
    KnowledgeCitation,
    KnowledgeQueryRequest,
    KnowledgeQueryResponse,
)
from main import create_app


class FakeDocumentCatalogService:
    """返回固定文档记录的假目录服务。"""

    def __init__(
        self,
        records: list[DocumentRecord],
        missing_document_ids: set[str] | None = None,
    ) -> None:
        """保存列表结果和模拟不存在的文档 ID。"""

        self._records = list(records)
        self._missing_document_ids = set(
            missing_document_ids or set()
        )

        self.list_documents_calls = 0

        # 保存 Router 传给 delete_document() 的所有 ID。
        self.deleted_document_ids: list[str] = []

    def list_documents(
        self,
    ) -> list[DocumentRecord]:
        """返回测试预先设置的文档列表。"""

        # 记录该方法被 Router 调用的次数。
        self.list_documents_calls += 1

        # 返回列表副本，避免 Router 或测试代码
        # 意外修改 Fake 内部保存的原列表。
        return list(self._records)

    def delete_document(
        self,
        document_id: str,
    ) -> None:
        """模拟成功删除或文档不存在。"""

        self.deleted_document_ids.append(
            document_id
        )

        if document_id in self._missing_document_ids:
            raise DocumentNotFoundError(
                f"知识库文档不存在：{document_id}"
            )


class FakeDocumentService:
    """模拟文档摄取服务，不调用真实 Embedding 和 Chroma。"""

    def __init__(
        self,
        *,
        record: DocumentRecord | None = None,
        error: Exception | None = None,
    ) -> None:
        """保存预期响应或预期异常。"""

        self._record = record
        self._error = error

        # 记录 ingest_document() 被调用的次数。
        self.ingest_calls = 0

        # 保存 Router 传入的临时路径。
        self.received_path: Path | None = None

        # 保存暂存文件中的原始字节。
        self.received_content: bytes | None = None

        # 记录调用期间临时文件是否存在。
        self.path_existed_during_call = False

    async def ingest_document(
        self,
        path: Path,
    ) -> DocumentRecord:
        """记录调用参数并返回预设结果。"""

        self.ingest_calls += 1
        self.received_path = path

        # exists() 用于证明 Router 没有过早清理临时文件。
        self.path_existed_during_call = (
            path.exists()
        )

        # read_bytes() 读取完整原始字节，
        # 用于验证 multipart 内容正确写入了暂存文件。
        self.received_content = (
            path.read_bytes()
        )

        if self._error is not None:
            raise self._error

        if self._record is None:
            raise AssertionError(
                "FakeDocumentService 没有配置返回记录"
            )

        return self._record


class FakeKnowledgeQueryService:
    """模拟RAG问答Service，不访问任何外部依赖。"""

    def __init__(
        self,
        *,
        response: KnowledgeQueryResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        """保存预期响应或预期异常。"""

        self._response = response
        self._error = error

        # 保存Router传入的所有请求模型。
        self.requests: list[
            KnowledgeQueryRequest
        ] = []

    async def answer_query(
        self,
        request: KnowledgeQueryRequest,
    ) -> KnowledgeQueryResponse:
        """记录请求并返回预设结果。"""

        self.requests.append(request)

        if self._error is not None:
            raise self._error

        if self._response is None:
            raise AssertionError(
                "FakeKnowledgeQueryService"
                "没有配置返回响应"
            )

        return self._response


def create_test_client(
    application: FastAPI,
) -> AsyncClient:
    """创建不访问真实网络的异步 HTTP 客户端。"""

    # ASGITransport 会把请求直接发送给内存中的应用，
    # 不启动 Uvicorn，也不访问 TCP 端口。
    transport = ASGITransport(
        app=application,
    )

    return AsyncClient(
        transport=transport,

        # base_url 只用于拼接相对路径，
        # 不会进行真实 DNS 或网络请求。
        base_url="http://testserver",
    )


@pytest.mark.asyncio
async def test_list_documents_returns_records() -> None:
    """GET 接口应返回目录服务提供的文档记录。"""

    record = DocumentRecord(
        document_id="a" * 64,
        source_file="manual.pdf",
        file_type="pdf",
        file_size_bytes=1024,
        page_or_section_count=10,
        chunk_count=20,
        embedding_model="embedding-3",
    )

    fake_service = FakeDocumentCatalogService(
        records=[record],
    )

    # 每个测试创建独立应用，避免共享依赖覆盖状态。
    application = create_app()

    def override_catalog_service(
    ) -> FakeDocumentCatalogService:
        """替换真实 Chroma 依赖。"""

        return fake_service

    application.dependency_overrides[
        get_document_catalog_service
    ] = override_catalog_service

    try:
        async with create_test_client(
            application
        ) as client:
            response = await client.get(
                "/api/v1/knowledge/documents"
            )
    finally:
        # 清理依赖覆盖，避免影响后续测试。
        application.dependency_overrides.clear()

    assert response.status_code == 200

    assert response.json() == {
        "documents": [
            record.model_dump(mode="json"),
        ]
    }

    # Router 应且只应调用一次 Service。
    assert fake_service.list_documents_calls == 1

    # RequestIdMiddleware 也必须处理知识库接口。
    assert response.headers["x-request-id"]


@pytest.mark.asyncio
async def test_list_documents_returns_empty_list() -> None:
    """空知识库应返回空列表，而不是 null 或 404。"""

    fake_service = FakeDocumentCatalogService(
        records=[],
    )
    application = create_app()

    def override_catalog_service(
    ) -> FakeDocumentCatalogService:
        """返回代表空知识库的 Fake Service。"""

        return fake_service

    application.dependency_overrides[
        get_document_catalog_service
    ] = override_catalog_service

    try:
        async with create_test_client(
            application
        ) as client:
            response = await client.get(
                "/api/v1/knowledge/documents"
            )
    finally:
        application.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "documents": [],
    }
    assert fake_service.list_documents_calls == 1


@pytest.mark.asyncio
async def test_delete_document_returns_success() -> None:
    """合法文档 ID 删除成功时应返回 200。"""

    document_id = "c" * 64
    fake_service = FakeDocumentCatalogService(
        records=[],
    )
    application = create_app()

    def override_catalog_service(
    ) -> FakeDocumentCatalogService:
        """使用不会访问真实 Chroma 的 Fake Service。"""

        return fake_service

    application.dependency_overrides[
        get_document_catalog_service
    ] = override_catalog_service

    try:
        async with create_test_client(
            application
        ) as client:
            response = await client.delete(
                f"/api/v1/knowledge/documents/"
                f"{document_id}"
            )
    finally:
        application.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "document_id": document_id,
        "deleted": True,
    }

    # 确认 Router 将 URL 中的 ID 原样交给了 Service。
    assert fake_service.deleted_document_ids == [
        document_id
    ]

    assert response.headers["x-request-id"]


@pytest.mark.asyncio
async def test_delete_missing_document_returns_404() -> None:
    """Service 报告文档不存在时应转换为统一 404 JSON。"""

    document_id = "d" * 64
    fake_service = FakeDocumentCatalogService(
        records=[],
        missing_document_ids={document_id},
    )
    application = create_app()

    def override_catalog_service(
    ) -> FakeDocumentCatalogService:
        """返回会对指定 ID 抛出不存在异常的 Fake。"""

        return fake_service

    application.dependency_overrides[
        get_document_catalog_service
    ] = override_catalog_service

    try:
        async with create_test_client(
            application
        ) as client:
            response = await client.delete(
                f"/api/v1/knowledge/documents/"
                f"{document_id}"
            )
    finally:
        application.dependency_overrides.clear()

    assert response.status_code == 404

    response_data = response.json()

    assert response_data["error"] == {
        "code": "document_not_found",
        "message": "知识库文档不存在",
    }

    # 错误响应体和响应头必须使用同一个请求 ID。
    assert response_data["request_id"] == (
        response.headers["x-request-id"]
    )

    assert fake_service.deleted_document_ids == [
        document_id
    ]


@pytest.mark.asyncio
async def test_delete_invalid_document_id_returns_422() -> None:
    """非法路径参数应在调用 Service 前被 FastAPI 拒绝。"""

    fake_service = FakeDocumentCatalogService(
        records=[],
    )
    application = create_app()

    def override_catalog_service(
    ) -> FakeDocumentCatalogService:
        """提供用于确认未被调用的 Fake Service。"""

        return fake_service

    application.dependency_overrides[
        get_document_catalog_service
    ] = override_catalog_service

    try:
        async with create_test_client(
            application
        ) as client:
            response = await client.delete(
                "/api/v1/knowledge/documents/not-a-sha256"
            )
    finally:
        application.dependency_overrides.clear()

    assert response.status_code == 422

    # 路径校验失败发生在调用 Router 业务逻辑之前。
    assert fake_service.deleted_document_ids == []

    assert response.headers["x-request-id"]


@pytest.mark.asyncio
async def test_upload_document_returns_created_record(
) -> None:
    """POST 应暂存文件、调用 Service 并返回 201。"""

    upload_content = (
        b"# Robot Manual\n"
        b"Check the emergency stop."
    )

    record = DocumentRecord(
        document_id="e" * 64,
        source_file="manual.md",
        file_type="md",
        file_size_bytes=len(upload_content),
        page_or_section_count=1,
        chunk_count=1,
        embedding_model="test-embedding-model",
    )

    fake_service = FakeDocumentService(
        record=record,
    )

    # 不读取真实 .env，也不连接正式 V4。
    settings = Settings(
        _env_file=None,
        max_document_size_bytes=1024,
    )

    application = create_app()

    def override_document_service(
    ) -> FakeDocumentService:
        """替换真实摄取服务。"""

        return fake_service

    def override_settings() -> Settings:
        """返回本测试专用配置。"""

        return settings

    application.dependency_overrides[
        get_document_service
    ] = override_document_service

    application.dependency_overrides[
        get_settings
    ] = override_settings

    try:
        async with create_test_client(
            application
        ) as client:
            response = await client.post(
                "/api/v1/knowledge/documents",

                # files 会让 HTTPX 构造
                # multipart/form-data 请求。
                files={
                    "file": (
                        "manual.md",
                        upload_content,
                        "text/markdown",
                    ),
                },
            )
    finally:
        application.dependency_overrides.clear()

    assert response.status_code == 201

    assert response.json() == (
        record.model_dump(mode="json")
    )

    assert fake_service.ingest_calls == 1
    assert (
        fake_service.path_existed_during_call
        is True
    )
    assert fake_service.received_content == (
        upload_content
    )

    # HTTP 请求处理完成后，
    # stage_uploaded_file() 应删除临时文件。
    assert fake_service.received_path is not None
    assert (
        fake_service.received_path.exists()
        is False
    )

    assert response.headers["x-request-id"]


@pytest.mark.asyncio
async def test_upload_duplicate_document_returns_409(
) -> None:
    """重复文档异常应转换成统一的 409 JSON。"""

    fake_service = FakeDocumentService(
        error=DuplicateDocumentError(
            "文档已经存在：manual.txt"
        ),
    )

    settings = Settings(
        _env_file=None,
        max_document_size_bytes=1024,
    )

    application = create_app()

    application.dependency_overrides[
        get_document_service
    ] = lambda: fake_service

    application.dependency_overrides[
        get_settings
    ] = lambda: settings

    try:
        async with create_test_client(
            application
        ) as client:
            response = await client.post(
                "/api/v1/knowledge/documents",
                files={
                    "file": (
                        "manual.txt",
                        b"robot manual",
                        "text/plain",
                    ),
                },
            )
    finally:
        application.dependency_overrides.clear()

    assert response.status_code == 409

    response_data = response.json()

    assert response_data["error"] == {
        "code": "duplicate_document",
        "message": (
            "相同文档已经存在于知识库中"
        ),
    }

    assert response_data["request_id"] == (
        response.headers["x-request-id"]
    )

    assert fake_service.ingest_calls == 1

    # 即使 Service 抛出异常，
    # async with 仍然必须清理临时文件。
    assert fake_service.received_path is not None
    assert (
        fake_service.received_path.exists()
        is False
    )


@pytest.mark.asyncio
async def test_upload_oversize_document_returns_422(
) -> None:
    """超限文件应在调用 DocumentService 前被拒绝。"""

    fake_service = FakeDocumentService()

    settings = Settings(
        _env_file=None,

        # 测试中只允许最多 5 字节。
        max_document_size_bytes=5,
    )

    application = create_app()

    application.dependency_overrides[
        get_document_service
    ] = lambda: fake_service

    application.dependency_overrides[
        get_settings
    ] = lambda: settings

    try:
        async with create_test_client(
            application
        ) as client:
            response = await client.post(
                "/api/v1/knowledge/documents",
                files={
                    "file": (
                        "large.txt",

                        # 这里是6字节，超过5字节限制。
                        b"123456",
                        "text/plain",
                    ),
                },
            )
    finally:
        application.dependency_overrides.clear()

    assert response.status_code == 422

    response_data = response.json()

    assert response_data["error"] == {
        "code": "document_validation_error",
        "message": "上传文档不符合摄取要求",
    }

    assert response_data["request_id"] == (
        response.headers["x-request-id"]
    )

    # 超限发生在暂存阶段，
    # 所以 DocumentService 不应该被调用。
    assert fake_service.ingest_calls == 0


@pytest.mark.asyncio
async def test_query_knowledge_returns_answer_and_citations(
) -> None:
    """查询接口应返回Service提供的回答和引用。"""

    citation = KnowledgeCitation(
        chunk_id=f"{'a' * 64}:000003",
        document_id="a" * 64,
        source_file="robot-faults.txt",
        page_or_section=(
            "section: ERR-NET-4001"
        ),
        chunk_index=3,
        rank=2,

        # 模拟一个只被关键词路径召回的融合候选。
        # 没有向量测量值时必须返回None，
        # JSON序列化后会成为null。
        similarity=None,
        rrf_score=0.016129,
        excerpt=(
            "ERR-NET-4001表示心跳包超时。"
        ),
    )

    expected_response = KnowledgeQueryResponse(
        answer=(
            "网络恢复后需要先核对任务状态。"
        ),
        citations=[citation],
        retrieval_ms=35.6,
        abstained=False,
    )

    fake_service = FakeKnowledgeQueryService(
        response=expected_response,
    )

    application = create_app()

    def override_query_service(
    ) -> FakeKnowledgeQueryService:
        """替换真实RAG依赖图。"""

        return fake_service

    application.dependency_overrides[
        get_knowledge_query_service
    ] = override_query_service

    try:
        async with create_test_client(
            application
        ) as client:
            response = await client.post(
                "/api/v1/knowledge/query",

                # json参数会：
                # 1. 把dict序列化为JSON；
                # 2. 设置application/json请求头。
                json={
                    "question": (
                        "急停复位后应检查什么？"
                    ),
                    "top_k": 3,
                },
            )
    finally:
        application.dependency_overrides.clear()

    assert response.status_code == 200

    assert response.json() == (
        expected_response.model_dump(
            mode="json"
        )
    )

    # Router应且只应调用一次Service。
    assert len(fake_service.requests) == 1

    received_request = (
        fake_service.requests[0]
    )

    assert received_request.question == (
        "急停复位后应检查什么？"
    )
    assert received_request.top_k == 3

    assert response.headers["x-request-id"]


@pytest.mark.asyncio
async def test_query_knowledge_returns_abstained_response(
) -> None:
    """证据不足时接口应返回200和明确拒答。"""

    expected_response = KnowledgeQueryResponse(
        answer="知识库没有足够证据回答该问题。",
        citations=[],
        retrieval_ms=20.4,
        abstained=True,
    )

    fake_service = FakeKnowledgeQueryService(
        response=expected_response,
    )

    application = create_app()

    application.dependency_overrides[
        get_knowledge_query_service
    ] = lambda: fake_service

    try:
        async with create_test_client(
            application
        ) as client:
            response = await client.post(
                "/api/v1/knowledge/query",
                json={
                    "question": "知识库之外的问题",
                },
            )
    finally:
        application.dependency_overrides.clear()

    assert response.status_code == 200

    assert response.json() == {
        "answer": (
            "知识库没有足够证据回答该问题。"
        ),
        "citations": [],
        "retrieval_ms": 20.4,
        "abstained": True,
    }

    # 没有提交top_k时，
    # Pydantic应使用默认值3。
    assert fake_service.requests[0].top_k == 3


@pytest.mark.asyncio
async def test_query_knowledge_rejects_blank_question(
) -> None:
    """空白问题应在调用Service前返回422。"""

    fake_service = FakeKnowledgeQueryService()

    application = create_app()

    application.dependency_overrides[
        get_knowledge_query_service
    ] = lambda: fake_service

    try:
        async with create_test_client(
            application
        ) as client:
            response = await client.post(
                "/api/v1/knowledge/query",
                json={
                    "question": "   ",
                    "top_k": 3,
                },
            )
    finally:
        application.dependency_overrides.clear()

    assert response.status_code == 422

    # str_strip_whitespace先把问题变成""，
    # min_length=1随后拒绝它。
    #
    # Router函数没有开始执行，
    # 所以Fake Service不应收到请求。
    assert fake_service.requests == []

    assert response.headers["x-request-id"]


@pytest.mark.asyncio
async def test_query_knowledge_timeout_returns_504(
) -> None:
    """RAG Service超时应转换成统一504错误。"""

    fake_service = FakeKnowledgeQueryService(
        error=LLMTimeoutError(
            "知识库回答服务响应超时"
        ),
    )

    application = create_app()

    application.dependency_overrides[
        get_knowledge_query_service
    ] = lambda: fake_service

    try:
        async with create_test_client(
            application
        ) as client:
            response = await client.post(
                "/api/v1/knowledge/query",
                json={
                    "question": "测试超时处理",
                },
            )
    finally:
        application.dependency_overrides.clear()

    assert response.status_code == 504

    response_data = response.json()

    assert response_data["error"] == {
        "code": "llm_timeout",
        "message": "LLM 服务响应超时",
    }

    # Middleware响应头和错误体必须使用同一个ID。
    assert response_data["request_id"] == (
        response.headers["x-request-id"]
    )

    assert len(fake_service.requests) == 1
