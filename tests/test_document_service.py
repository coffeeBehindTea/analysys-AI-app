"""DocumentService 完整摄取编排的离线测试。"""

from pathlib import Path

import pytest

from app.errors import (
    DocumentValidationError,
    DuplicateDocumentError,
)
from app.schemas.knowledge import DocumentRecord
from app.schemas.retrieval import (
    EmbeddedChunk,
    EmbeddingVector,
    RetrievedChunk,
)
from app.services.chunking import TextChunker
from app.services.document_preparation import (
    TextDocumentPreparationService,
    calculate_file_sha256,
)
from app.services.document_service import (
    DocumentService,
)
from app.services.vector_store import VectorStore


class FakeEmbeddingProvider:
    """不联网的测试 Embedding 提供者。"""

    def __init__(self) -> None:
        # 每一项记录一次 embed_texts()调用收到的文本批次。
        self.calls: list[list[str]] = []

    async def embed_texts(
        self,
        texts: list[str],
    ) -> list[EmbeddingVector]:
        """为每段文本生成简单、非空的二维假向量。"""

        # list(texts) 创建副本，
        # 避免外部之后修改原列表影响测试记录。
        self.calls.append(list(texts))

        return [
            # 向量只用于离线验证编排，
            # 不表示真实语义。
            [1.0, float(len(text))]
            for text in texts
        ]


class FakeVectorStore:
    """只保存在内存中的测试 VectorStore。"""

    def __init__(self) -> None:
        self.records: dict[
            str,
            DocumentRecord,
        ] = {}

        self.chunks_by_document: dict[
            str,
            list[EmbeddedChunk],
        ] = {}

    def has_document(
        self,
        document_id: str,
    ) -> bool:
        """根据 document_id 检查内存字典。"""

        return document_id in self.records

    def add_document(
        self,
        *,
        record: DocumentRecord,
        chunks: list[EmbeddedChunk],
    ) -> None:
        """保存文档记录及其向量 Chunk。"""

        if self.has_document(record.document_id):
            raise DuplicateDocumentError(
                f"文档已经存在：{record.source_file}"
            )

        self.records[record.document_id] = record

        # list(chunks) 创建列表副本。
        self.chunks_by_document[
            record.document_id
        ] = list(chunks)

    def list_documents(
        self,
    ) -> list[DocumentRecord]:
        """返回按文件名排序的文档记录。"""

        return sorted(
            self.records.values(),
            key=lambda record: (
                record.source_file.casefold()
            ),
        )

    def retrieve(
        self,
        query_embedding: EmbeddingVector,
        *,
        query_embedding_model: str,
        top_k: int = 3,
    ) -> list[RetrievedChunk]:
        """当前测试不需要检索，返回空列表。"""

        return []

    def delete_document(
        self,
        document_id: str,
    ) -> bool:
        """删除内存中的文档记录和 Chunk。"""

        if document_id not in self.records:
            return False

        del self.records[document_id]
        del self.chunks_by_document[document_id]

        return True


def make_service(
    *,
    provider: FakeEmbeddingProvider,
    vector_store: VectorStore,
    max_file_size_bytes: int = (
        20 * 1024 * 1024
    ),
) -> DocumentService:
    """使用真实解析/切分和两个 Fake 创建服务。"""

    preparation_service = (
        TextDocumentPreparationService(
            chunker=TextChunker(
                chunk_size=1000,
                overlap=100,
            )
        )
    )

    return DocumentService(
        preparation_service=preparation_service,
        embedding_provider=provider,
        vector_store=vector_store,
        embedding_model="fake-embedding",
        batch_size=2,
        max_file_size_bytes=max_file_size_bytes,
    )


@pytest.mark.asyncio
async def test_ingest_document_runs_complete_pipeline(
    tmp_path: Path,
) -> None:
    """摄取应完成解析、切分、向量化和持久化。"""

    path = tmp_path / "faults.txt"

    path.write_text(
        """故障代码 (ErrorCode)
ERR-NAV-2001
故障名称
二维码定位丢失

故障代码 (ErrorCode)
ERR-SAF-1002
故障名称
机械防撞条触发
""",
        encoding="utf-8",
    )

    provider = FakeEmbeddingProvider()
    vector_store = FakeVectorStore()

    service = make_service(
        provider=provider,
        vector_store=vector_store,
    )

    record = await service.ingest_document(path)

    assert record.document_id == (
        calculate_file_sha256(path)
    )
    assert record.source_file == "faults.txt"
    assert record.file_type == "txt"
    assert record.page_or_section_count == 2
    assert record.chunk_count == 2
    assert record.embedding_model == "fake-embedding"
    assert record.status == "ready"

    # 两个 Chunk 小于 batch_size=2，
    # 所以 EmbeddingProvider 应只被调用一次。
    assert len(provider.calls) == 1
    assert len(provider.calls[0]) == 2

    saved_chunks = (
        vector_store.chunks_by_document[
            record.document_id
        ]
    )

    assert len(saved_chunks) == 2

    # 确认保存的不只是 Chunk 信息，
    # 而是包含实际向量的 EmbeddedChunk。
    assert saved_chunks[0].embedding
    assert (
        saved_chunks[0].embedding_model
        == "fake-embedding"
    )


@pytest.mark.asyncio
async def test_duplicate_is_rejected_before_embedding(
    tmp_path: Path,
) -> None:
    """重复文档不应再次调用 Embedding 服务。"""

    path = tmp_path / "manual.txt"
    path.write_text(
        "机器人急停后必须人工确认。",
        encoding="utf-8",
    )

    provider = FakeEmbeddingProvider()
    vector_store = FakeVectorStore()

    document_id = calculate_file_sha256(path)

    # 预先放入一条同 document_id 的记录，
    # 模拟这份文件已经完成摄取。
    vector_store.records[document_id] = (
        DocumentRecord(
            document_id=document_id,
            source_file="existing.txt",
            file_type="txt",
            file_size_bytes=100,
            page_or_section_count=1,
            chunk_count=1,
            embedding_model="fake-embedding",
        )
    )

    service = make_service(
        provider=provider,
        vector_store=vector_store,
    )

    with pytest.raises(
        DuplicateDocumentError,
        match="已经存在",
    ):
        await service.ingest_document(path)

    # 空列表说明 embed_texts()一次也没有被调用。
    assert provider.calls == []

@pytest.mark.asyncio
async def test_unsupported_file_type_is_rejected(
    tmp_path: Path,
) -> None:
    """未支持的文件类型不能进入解析和向量化。"""

    path = tmp_path / "manual.docx"
    path.write_bytes(b"fake docx")

    provider = FakeEmbeddingProvider()
    vector_store = FakeVectorStore()

    service = make_service(
        provider=provider,
        vector_store=vector_store,
    )

    with pytest.raises(
        DocumentValidationError,
        match="只支持",
    ):
        await service.ingest_document(path)

    assert provider.calls == []
    assert vector_store.list_documents() == []


@pytest.mark.asyncio
async def test_oversized_file_is_rejected(
    tmp_path: Path,
) -> None:
    """超过配置上限的文件不能进入 Embedding。"""

    path = tmp_path / "large.txt"
    path.write_bytes(b"12345")

    provider = FakeEmbeddingProvider()
    vector_store = FakeVectorStore()

    service = make_service(
        provider=provider,
        vector_store=vector_store,

        # 测试中把限制缩小为 4 字节，
        # 不需要真的创建 20 MiB 文件。
        max_file_size_bytes=4,
    )

    with pytest.raises(
        DocumentValidationError,
        match="超过大小限制",
    ):
        await service.ingest_document(path)

    assert provider.calls == []
