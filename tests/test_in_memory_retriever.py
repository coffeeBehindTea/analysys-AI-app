"""内存 Top-K 检索器的单元测试。"""

import pytest

from app.schemas.retrieval import (
    DocumentChunk,
    EmbeddedChunk,
)
from app.services.retrieval import InMemoryRetriever


def make_embedded_chunk(
    chunk_id: str,
    embedding: list[float],
    *,
    model: str = "test-embedding-model",
) -> EmbeddedChunk:
    """构造测试使用的带向量 Chunk。"""

    return EmbeddedChunk(
        chunk=DocumentChunk(
            chunk_id=chunk_id,
            document_id="test-document",
            source_file="test.txt",
            page_or_section=f"section:{chunk_id}",
            chunk_index=0,

            # "a" * 64 会生成一个长度为 64 的小写十六进制字符串，
            # 满足当前 content_hash 的格式约束。
            content_hash="a" * 64,
            content=f"{chunk_id} 的测试内容",
        ),
        embedding_model=model,
        embedding=embedding,
    )


def test_retrieve_returns_top_k_in_similarity_order() -> None:
    """检索器应按相似度从高到低返回指定数量结果。"""

    retriever = InMemoryRetriever(
        [
            make_embedded_chunk(
                "chunk-c",
                [0.0, 1.0],
            ),
            make_embedded_chunk(
                "chunk-a",
                [1.0, 0.0],
            ),
            make_embedded_chunk(
                "chunk-b",
                [0.8, 0.2],
            ),
        ]
    )

    results = retriever.retrieve(
        [1.0, 0.0],
        query_embedding_model="test-embedding-model",
        top_k=2,
    )

    assert [
        result.chunk.chunk_id
        for result in results
    ] == [
        "chunk-a",
        "chunk-b",
    ]

    assert [
        result.rank
        for result in results
    ] == [1, 2]

    assert results[0].similarity == pytest.approx(1.0)
    assert (
        results[0].similarity
        > results[1].similarity
    )


def test_equal_scores_use_chunk_id_as_tiebreaker() -> None:
    """相同分数使用 chunk_id 产生稳定顺序。"""

    retriever = InMemoryRetriever(
        [
            # 故意先放 chunk-b。
            make_embedded_chunk(
                "chunk-b",
                [1.0, 0.0],
            ),
            make_embedded_chunk(
                "chunk-a",
                [1.0, 0.0],
            ),
        ]
    )

    results = retriever.retrieve(
        [1.0, 0.0],
        query_embedding_model="test-embedding-model",
        top_k=2,
    )

    # 两项相似度相同，因此按照 chunk_id 升序排列。
    assert [
        result.chunk.chunk_id
        for result in results
    ] == [
        "chunk-a",
        "chunk-b",
    ]


def test_top_k_larger_than_corpus_returns_all_chunks() -> None:
    """top_k 超过文档数量时应返回全部已有结果。"""

    retriever = InMemoryRetriever(
        [
            make_embedded_chunk(
                "chunk-a",
                [1.0, 0.0],
            ),
            make_embedded_chunk(
                "chunk-b",
                [0.0, 1.0],
            ),
        ]
    )

    results = retriever.retrieve(
        [1.0, 0.0],
        query_embedding_model="test-embedding-model",
        top_k=10,
    )

    assert len(results) == 2


def test_empty_corpus_returns_empty_results() -> None:
    """空知识库应返回空结果，而不是伪造证据。"""

    retriever = InMemoryRetriever([])

    results = retriever.retrieve(
        [1.0, 0.0],
        query_embedding_model="test-embedding-model",
        top_k=3,
    )

    assert results == []


def test_non_positive_top_k_is_rejected() -> None:
    """top_k 必须至少为 1。"""

    retriever = InMemoryRetriever([])

    with pytest.raises(
        ValueError,
        match="top_k 必须",
    ):
        retriever.retrieve(
            [1.0, 0.0],
            query_embedding_model="test-embedding-model",
            top_k=0,
        )


def test_query_model_must_match_document_model() -> None:
    """查询和文档不能来自不同的向量空间。"""

    retriever = InMemoryRetriever(
        [
            make_embedded_chunk(
                "chunk-a",
                [1.0, 0.0],
                model="document-model",
            )
        ]
    )

    with pytest.raises(
        ValueError,
        match="同一个 Embedding 模型",
    ):
        retriever.retrieve(
            [1.0, 0.0],
            query_embedding_model="query-model",
            top_k=1,
        )


def test_mixed_document_models_are_rejected() -> None:
    """一个内存索引不能混合不同 Embedding 模型。"""

    chunks = [
        make_embedded_chunk(
            "chunk-a",
            [1.0, 0.0],
            model="model-a",
        ),
        make_embedded_chunk(
            "chunk-b",
            [0.0, 1.0],
            model="model-b",
        ),
    ]

    with pytest.raises(
        ValueError,
        match="不能混合不同的 Embedding 模型",
    ):
        InMemoryRetriever(chunks)


def test_mixed_vector_dimensions_are_rejected() -> None:
    """一个内存索引不能混合不同向量维度。"""

    chunks = [
        make_embedded_chunk(
            "chunk-a",
            [1.0, 0.0],
        ),
        make_embedded_chunk(
            "chunk-b",
            [1.0, 0.0, 0.0],
        ),
    ]

    with pytest.raises(
        ValueError,
        match="不能混合不同维度",
    ):
        InMemoryRetriever(chunks)