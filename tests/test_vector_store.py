"""ChromaDB 向量存储的离线测试。"""

from hashlib import sha256
from pathlib import Path

import pytest

from app.errors import (
    DuplicateDocumentError,
    VectorStoreError,
)
from app.schemas.knowledge import DocumentRecord
from app.schemas.retrieval import (
    DocumentChunk,
    EmbeddedChunk,
)
from app.services.vector_store import (
    ChromaVectorStore,
)
from app.services.retrieval_scope import (
    RetrievalScopeFilter,
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


def make_single_chunk_document(
    *,
    document_id: str,
    source_file: str,
    content: str,
    vector: list[float],
) -> tuple[
    DocumentRecord,
    list[EmbeddedChunk],
]:
    """构造一份只有一个Chunk的范围检索测试文档。"""

    chunk = DocumentChunk(
        chunk_id=f"{document_id}:000000",
        document_id=document_id,
        source_file=source_file,
        page_or_section="section: scoped-test",
        chunk_index=0,
        content_hash=sha256(
            content.encode("utf-8")
        ).hexdigest(),
        content=content,
    )

    embedded_chunk = EmbeddedChunk(
        chunk=chunk,
        embedding_model="fake-embedding",
        embedding=vector,
    )

    record = DocumentRecord(
        document_id=document_id,
        source_file=source_file,
        file_type="txt",
        file_size_bytes=len(
            content.encode("utf-8")
        ),
        page_or_section_count=1,
        chunk_count=1,
        embedding_model="fake-embedding",
    )

    return record, [embedded_chunk]


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


def test_list_chunks_empty_store_returns_empty_list(
    tmp_path: Path,
) -> None:
    """空Collection应返回空Chunk列表。"""

    store = make_store(tmp_path)

    # 空知识库是合法状态，
    # 不能伪造Chunk或抛出存储结构异常。
    assert store.list_chunks() == []


def test_list_chunks_restores_complete_sorted_chunks(
    tmp_path: Path,
) -> None:
    """全部正文和来源字段应恢复，并按Chunk序号稳定排序。"""

    store = make_store(tmp_path)
    record, chunks = make_test_data()

    # 故意按1、0的反向顺序写入，
    # 验证list_chunks()不依赖Chroma内部返回顺序。
    store.add_document(
        record=record,
        chunks=list(reversed(chunks)),
    )

    restored_chunks = store.list_chunks()

    # 返回值必须是项目内部DocumentChunk契约。
    assert all(
        isinstance(chunk, DocumentChunk)
        for chunk in restored_chunks
    )

    # make_test_data()中的原始顺序是chunk_index 0、1。
    # Pydantic模型相等比较会比较各字段值，
    # 因而同时覆盖ID、来源、哈希和正文恢复。
    assert restored_chunks == [
        embedded_chunk.chunk
        for embedded_chunk in chunks
    ]

    assert [
        chunk.chunk_index
        for chunk in restored_chunks
    ] == [0, 1]


def test_list_chunks_survives_new_store_instance(
    tmp_path: Path,
) -> None:
    """新建ChromaVectorStore后仍应读到持久化Chunk。"""

    first_store = make_store(tmp_path)
    record, chunks = make_test_data()

    first_store.add_document(
        record=record,
        chunks=chunks,
    )

    # 使用相同目录、Collection名称和Embedding模型
    # 模拟应用进程重启后重新创建存储适配器。
    second_store = make_store(tmp_path)

    assert second_store.list_chunks() == [
        embedded_chunk.chunk
        for embedded_chunk in chunks
    ]


def test_list_chunks_wraps_incomplete_metadata(
    tmp_path: Path,
) -> None:
    """缺少来源字段的存储记录应转换成统一VectorStoreError。"""

    store = make_store(tmp_path)

    # 测试故意绕过add_document()，
    # 直接向底层Collection写入损坏记录。
    # 正常业务代码不应访问_collection私有属性。
    store._collection.add(  # noqa: SLF001
        ids=["broken-chunk"],
        embeddings=[[1.0, 0.0]],
        documents=["损坏的测试正文"],
        metadatas=[
            {
                # 模型字段合法，但缺少document_id、
                # source_file等DocumentChunk必要元数据。
                "embedding_model": "fake-embedding",
            }
        ],
    )

    with pytest.raises(
        VectorStoreError,
        match="Chunk数据结构无效",
    ) as exc_info:
        store.list_chunks()

    # from exc会保留最初的KeyError作为__cause__，
    # 便于服务端日志诊断，同时对外只暴露统一异常。
    assert isinstance(
        exc_info.value.__cause__,
        KeyError,
    )


def test_list_chunks_rejects_mismatched_embedding_model(
    tmp_path: Path,
) -> None:
    """Chunk元数据模型与Collection不一致时必须拒绝读取。"""

    store = make_store(tmp_path)
    content = "模型不一致的测试正文"

    # 同样绕过正常写入服务，模拟数据库被外部工具
    # 写入了模型元数据不一致的记录。
    store._collection.add(  # noqa: SLF001
        ids=["wrong-model-chunk"],
        embeddings=[[1.0, 0.0]],
        documents=[content],
        metadatas=[
            {
                "document_id": "b" * 64,
                "source_file": "wrong.txt",
                "page_or_section": "section: wrong",
                "chunk_index": 0,
                "content_hash": sha256(
                    content.encode("utf-8")
                ).hexdigest(),
                "embedding_model": "another-model",
            }
        ],
    )

    with pytest.raises(
        VectorStoreError,
        match="Chunk数据结构无效",
    ) as exc_info:
        store.list_chunks()

    assert isinstance(
        exc_info.value.__cause__,
        ValueError,
    )


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

    # 文档删除后，供关键词索引读取的Chunk正文
    # 也必须同步消失，不能残留“幽灵证据”。
    assert store.list_chunks() == []

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


def test_retrieve_source_scope_filters_before_top_k(
    tmp_path: Path,
) -> None:
    """数据库应在向量Top-K之前应用来源文件范围。"""

    store = make_store(tmp_path)

    outside_record, outside_chunks = (
        make_single_chunk_document(
            document_id="b" * 64,
            source_file="outside.txt",
            content="范围外但向量完全相同",

            # 与查询向量[1, 0]完全相同，
            # 不加范围时它应获得最高相似度。
            vector=[1.0, 0.0],
        )
    )
    allowed_record, allowed_chunks = (
        make_single_chunk_document(
            document_id="c" * 64,
            source_file="allowed.txt",
            content="范围内但相似度略低",
            vector=[0.8, 0.6],
        )
    )

    store.add_document(
        record=outside_record,
        chunks=outside_chunks,
    )
    store.add_document(
        record=allowed_record,
        chunks=allowed_chunks,
    )

    results = store.retrieve(
        [1.0, 0.0],
        query_embedding_model="fake-embedding",
        top_k=1,
        retrieval_scope=(
            RetrievalScopeFilter(
                source_files=(
                    "allowed.txt",
                ),
            )
        ),
    )

    # 如果先做全库Top-1再过滤，
    # outside.txt会占据唯一名额。
    #
    # 返回allowed.txt证明where条件在
    # Chroma近邻搜索阶段已经生效。
    assert len(results) == 1
    assert results[0].chunk.source_file == (
        "allowed.txt"
    )
    assert results[0].chunk.document_id == (
        allowed_record.document_id
    )
    assert results[0].rank == 1


def test_retrieve_document_id_scope_filters_vectors(
    tmp_path: Path,
) -> None:
    """Chroma应只从允许document_id中搜索向量。"""

    store = make_store(tmp_path)
    allowed_record, allowed_chunks = (
        make_single_chunk_document(
            document_id="d" * 64,
            source_file="same-name.txt",
            content="允许文档",
            vector=[0.7, 0.7],
        )
    )
    other_record, other_chunks = (
        make_single_chunk_document(
            document_id="e" * 64,
            source_file="same-name.txt",
            content="其他文档",
            vector=[1.0, 0.0],
        )
    )

    store.add_document(
        record=allowed_record,
        chunks=allowed_chunks,
    )
    store.add_document(
        record=other_record,
        chunks=other_chunks,
    )

    results = store.retrieve(
        [1.0, 0.0],
        query_embedding_model="fake-embedding",
        top_k=2,
        retrieval_scope=(
            RetrievalScopeFilter(
                document_ids=(
                    allowed_record.document_id,
                ),
            )
        ),
    )

    assert [
        result.chunk.document_id
        for result in results
    ] == [allowed_record.document_id]


def test_retrieve_combined_scope_uses_chroma_and(
    tmp_path: Path,
) -> None:
    """文档ID和文件名应在真实Chroma查询中同时生效。"""

    store = make_store(tmp_path)

    matching_record, matching_chunks = (
        make_single_chunk_document(
            document_id="6" * 64,
            source_file="allowed.txt",
            content="同时满足两项条件",
            vector=[0.6, 0.8],
        )
    )
    document_only_record, document_only_chunks = (
        make_single_chunk_document(
            document_id="6" * 64,
            source_file="other.txt",
            content="只满足文档ID",
            vector=[1.0, 0.0],
        )
    )
    file_only_record, file_only_chunks = (
        make_single_chunk_document(
            document_id="7" * 64,
            source_file="allowed.txt",
            content="只满足文件名",
            vector=[0.9, 0.1],
        )
    )

    # 相同document_id代表同一份文档内容，
    # ChromaVectorStore会拒绝把它作为另一份文档写入。
    # 因此“只满足document_id”的测试记录需要直接写入
    # 当前Collection，模拟同一文档ID下的异常文件名元数据。
    store.add_document(
        record=matching_record,
        chunks=matching_chunks,
    )
    store.add_document(
        record=file_only_record,
        chunks=file_only_chunks,
    )

    document_only_chunk = (
        document_only_chunks[0]
    )
    store._collection.add(  # noqa: SLF001
        ids=[
            document_only_chunk.chunk.chunk_id
            + "-other-file"
        ],
        embeddings=[
            document_only_chunk.embedding
        ],
        documents=[
            document_only_chunk.chunk.content
        ],
        metadatas=[
            {
                "document_id": (
                    document_only_record.document_id
                ),
                "source_file": (
                    document_only_record.source_file
                ),
                "page_or_section": (
                    "section: scoped-test"
                ),
                "chunk_index": 1,
                "content_hash": (
                    document_only_chunk
                    .chunk
                    .content_hash
                ),
                "embedding_model": (
                    "fake-embedding"
                ),
            }
        ],
    )

    results = store.retrieve(
        [1.0, 0.0],
        query_embedding_model="fake-embedding",
        top_k=3,
        retrieval_scope=(
            RetrievalScopeFilter(
                document_ids=(
                    matching_record.document_id,
                ),
                source_files=(
                    matching_record.source_file,
                ),
            )
        ),
    )

    assert len(results) == 1
    assert results[0].chunk.document_id == (
        matching_record.document_id
    )
    assert results[0].chunk.source_file == (
        matching_record.source_file
    )


def test_retrieve_scope_without_matches_returns_empty(
    tmp_path: Path,
) -> None:
    """Collection非空但范围无匹配记录时应返回空列表。"""

    store = make_store(tmp_path)
    record, chunks = make_single_chunk_document(
        document_id="8" * 64,
        source_file="outside.txt",
        content="范围外向量",
        vector=[1.0, 0.0],
    )
    store.add_document(
        record=record,
        chunks=chunks,
    )

    results = store.retrieve(
        [1.0, 0.0],
        query_embedding_model="fake-embedding",
        top_k=3,
        retrieval_scope=(
            RetrievalScopeFilter(
                source_files=(
                    "missing.txt",
                ),
            )
        ),
    )

    assert results == []


def test_retrieve_rejects_invalid_scope_type(
    tmp_path: Path,
) -> None:
    """普通dict不能直接成为Chroma元数据过滤条件。"""

    store = make_store(tmp_path)

    with pytest.raises(
        TypeError,
        match=(
            "retrieval_scope必须是"
            "RetrievalScopeFilter或None"
        ),
    ):
        store.retrieve(
            [1.0, 0.0],
            query_embedding_model=(
                "fake-embedding"
            ),
            retrieval_scope={  # type: ignore[arg-type]
                "source_file": "allowed.txt"
            },
        )
