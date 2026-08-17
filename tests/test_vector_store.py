"""ChromaDB 向量存储的离线测试。"""

from hashlib import sha256
from pathlib import Path

import pytest

from app.errors import DuplicateDocumentError
from app.schemas.knowledge import DocumentRecord
from app.schemas.retrieval import (
    DocumentChunk,
    EmbeddedChunk,
)
from app.services.vector_store import (
    ChromaVectorStore,
)


def make_test_data() -> tuple[
    DocumentRecord,
    list[EmbeddedChunk],
]:
    """创建不联网的文档记录和假向量。"""

    document_id = "a" * 64
    source_file = "manual.txt"

    contents = [
        "机器人急停后必须人工确认。",
        "恢复任务前需要重新检查周围环境。",
    ]

    vectors = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    chunks: list[EmbeddedChunk] = []

    for index, (content, vector) in enumerate(
        zip(contents, vectors)
    ):
        content_hash = sha256(
            content.encode("utf-8")
        ).hexdigest()

        chunks.append(
            EmbeddedChunk(
                chunk=DocumentChunk(
                    chunk_id=(
                        f"{document_id}:{index:06d}"
                    ),
                    document_id=document_id,
                    source_file=source_file,
                    page_or_section=(
                        f"section: test-{index + 1}"
                    ),
                    chunk_index=index,
                    content_hash=content_hash,
                    content=content,
                ),
                embedding_model="fake-embedding",
                embedding=vector,
            )
        )

    record = DocumentRecord(
        document_id=document_id,
        source_file=source_file,
        file_type="txt",
        file_size_bytes=100,
        page_or_section_count=2,
        chunk_count=2,
        embedding_model="fake-embedding",
    )

    return record, chunks


def make_store(
    tmp_path: Path,
) -> ChromaVectorStore:
    """在 pytest 临时目录中创建持久化 ChromaDB。"""

    return ChromaVectorStore(
        persist_directory=tmp_path / "chroma",
        collection_name="test_knowledge",
        embedding_model="fake-embedding",
    )


def test_add_and_list_document(
    tmp_path: Path,
) -> None:
    """写入后应能恢复整份文档记录。"""

    store = make_store(tmp_path)
    record, chunks = make_test_data()

    assert store.has_document(
        record.document_id
    ) is False

    store.add_document(
        record=record,
        chunks=chunks,
    )

    assert store.has_document(
        record.document_id
    ) is True

    assert store.list_documents() == [record]


def test_data_survives_new_store_instance(
    tmp_path: Path,
) -> None:
    """重新创建客户端后，持久化数据仍应存在。"""

    first_store = make_store(tmp_path)
    record, chunks = make_test_data()

    first_store.add_document(
        record=record,
        chunks=chunks,
    )

    # 使用相同目录和 Collection 创建另一个实例。
    second_store = make_store(tmp_path)

    assert second_store.has_document(
        record.document_id
    ) is True

    assert second_store.list_documents() == [
        record
    ]


def test_duplicate_document_is_rejected(
    tmp_path: Path,
) -> None:
    """相同文件哈希不能重复写入。"""

    store = make_store(tmp_path)
    record, chunks = make_test_data()

    store.add_document(
        record=record,
        chunks=chunks,
    )

    with pytest.raises(
        DuplicateDocumentError,
        match="已经存在",
    ):
        store.add_document(
            record=record,
            chunks=chunks,
        )


def test_delete_document_removes_all_chunks(
    tmp_path: Path,
) -> None:
    """删除文档后，其所有 Chunk 都应消失。"""

    store = make_store(tmp_path)
    record, chunks = make_test_data()

    store.add_document(
        record=record,
        chunks=chunks,
    )

    assert store.delete_document(
        record.document_id
    ) is True

    assert store.has_document(
        record.document_id
    ) is False

    assert store.list_documents() == []

    # 第二次删除时已经不存在，所以返回 False。
    assert store.delete_document(
        record.document_id
    ) is False


def test_retrieve_returns_ranked_traceable_chunks(
    tmp_path: Path,
) -> None:
    """查询应按相似度返回完整 Chunk 和来源元数据。"""

    store = make_store(tmp_path)
    record, chunks = make_test_data()

    store.add_document(
        record=record,
        chunks=chunks,
    )

    # 测试数据中的两个向量分别是：
    #
    # Chunk 0：[1, 0]
    # Chunk 1：[0, 1]
    #
    # 查询向量 [1, 0] 应与 Chunk 0 完全相同。
    results = store.retrieve(
        [1.0, 0.0],
        query_embedding_model="fake-embedding",
        top_k=2,
    )

    assert len(results) == 2

    assert [
        result.rank
        for result in results
    ] == [1, 2]

    assert [
        result.chunk.chunk_index
        for result in results
    ] == [0, 1]

    # 第一个结果与查询向量完全同向，
    # 余弦相似度应接近 1。
    assert results[0].similarity == pytest.approx(
        1.0,
        abs=1e-6,
    )

    # 第二个结果与查询向量正交，
    # 余弦相似度应接近 0。
    assert results[1].similarity == pytest.approx(
        0.0,
        abs=1e-6,
    )

    # 检索结果必须保留可追溯证据。
    assert (
        results[0].chunk.source_file
        == "manual.txt"
    )
    assert (
        results[0].chunk.page_or_section
        == "section: test-1"
    )
    assert (
        results[0].chunk.content
        == "机器人急停后必须人工确认。"
    )


def test_retrieve_respects_top_k(
    tmp_path: Path,
) -> None:
    """top_k 应限制最终返回的结果数量。"""

    store = make_store(tmp_path)
    record, chunks = make_test_data()

    store.add_document(
        record=record,
        chunks=chunks,
    )

    results = store.retrieve(
        [1.0, 0.0],
        query_embedding_model="fake-embedding",
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].rank == 1


def test_retrieve_empty_store_returns_empty_list(
    tmp_path: Path,
) -> None:
    """空知识库查询应返回空结果，而不是伪造证据。"""

    store = make_store(tmp_path)

    results = store.retrieve(
        [1.0, 0.0],
        query_embedding_model="fake-embedding",
        top_k=3,
    )

    assert results == []


def test_retrieve_rejects_different_model(
    tmp_path: Path,
) -> None:
    """查询和文档不能使用不同的 Embedding 模型。"""

    store = make_store(tmp_path)

    with pytest.raises(
        ValueError,
        match="同一个 Embedding 模型",
    ):
        store.retrieve(
            [1.0, 0.0],
            query_embedding_model="another-model",
            top_k=3,
        )


def test_retrieve_rejects_invalid_top_k(
    tmp_path: Path,
) -> None:
    """Top-0 没有检索语义，应被拒绝。"""

    store = make_store(tmp_path)

    with pytest.raises(
        ValueError,
        match="top_k",
    ):
        store.retrieve(
            [1.0, 0.0],
            query_embedding_model="fake-embedding",
            top_k=0,
        )