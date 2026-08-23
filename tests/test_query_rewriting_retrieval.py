"""查询改写混合检索编排器的异步离线测试。

本模块使用Fake Embedding和Fake混合检索器，
不访问网络、真实Embedding API、Chroma或LLM。

测试重点：
1. Embedding和关键词路径使用同一个改写查询；
2. 编排器正确转发查询向量与top_k；
3. 可选检索范围原样传给下游混合检索器；
4. 返回结果包含查询改写审计信息和候选快照；
5. 非法参数在调用外部依赖前被拒绝；
6. 依赖对象违反Protocol返回约定时能够被发现。
"""

from dataclasses import FrozenInstanceError

import pytest

from app.schemas.retrieval import (
    DocumentChunk,
    EmbeddingVector,
    HybridRetrievedChunk,
)
from app.services.query_rewriting_retrieval import (
    QueryRewritingHybridRetriever,
    QueryRewritingRetrievalResult,
)
from app.services.retrieval_scope import (
    RetrievalScopeFilter,
)


TEST_CONTENT_HASH = "b" * 64


def make_hybrid_result(
    *,
    chunk_id: str = "chunk-001",
) -> HybridRetrievedChunk:
    """构造一条合法且同时来自两路召回的融合结果。"""

    chunk = DocumentChunk(
        chunk_id=chunk_id,
        document_id="document-001",
        source_file="reader-manual.pdf",
        page_or_section="page: 56",
        chunk_index=0,
        content_hash=TEST_CONTENT_HASH,
        content="DataMan 260 cable precautions.",
    )

    return HybridRetrievedChunk(
        chunk=chunk,
        rrf_score=0.032,
        rank=1,
        vector_rank=1,
        vector_similarity=0.72,
        keyword_rank=1,
        keyword_score=8.0,
        matched_model_aliases=(
            "dataman-260",
        ),
    )


class FakeEmbeddingProvider:
    """记录收到的查询，并返回固定测试向量。"""

    def __init__(
        self,
        *,
        vector: EmbeddingVector | None = None,
        error: Exception | None = None,
    ) -> None:
        """保存返回向量、可选异常和调用记录。"""

        self.vector = (
            vector
            if vector is not None
            else [0.6, 0.8]
        )
        self.error = error
        self.received_queries: list[str] = []

    async def embed_query(
        self,
        query: str,
    ) -> EmbeddingVector:
        """模拟异步Embedding调用。"""

        self.received_queries.append(query)

        if self.error is not None:
            raise self.error

        # 返回新列表，避免测试调用方意外修改
        # Fake内部保存的原始向量。
        return list(self.vector)


class FakeHybridRetriever:
    """记录混合检索参数，并返回预设结果。"""

    def __init__(
        self,
        *,
        results: object,
    ) -> None:
        """保存预设返回值并初始化调用记录。"""

        self.results = results
        self.received_queries: list[str] = []
        self.received_embeddings: list[
            EmbeddingVector
        ] = []
        self.received_top_k_values: list[int] = []
        # 每次调用对应一个可选范围；
        # None明确表示该次调用使用全库检索。
        self.received_scopes: list[
            RetrievalScopeFilter | None
        ] = []

    def retrieve(
        self,
        *,
        query: str,
        query_embedding: EmbeddingVector,
        top_k: int = 3,
        retrieval_scope: (
            RetrievalScopeFilter | None
        ) = None,
    ) -> list[HybridRetrievedChunk]:
        """记录查询、向量、深度和范围并返回预设对象。"""

        self.received_queries.append(query)
        self.received_embeddings.append(
            list(query_embedding)
        )
        self.received_top_k_values.append(top_k)
        self.received_scopes.append(
            retrieval_scope
        )

        # 部分测试会故意配置非法返回值，
        # 以验证生产编排器的运行时防御。
        return self.results  # type: ignore[return-value]


@pytest.mark.asyncio
async def test_rewritten_query_is_used_by_both_paths(
) -> None:
    """Embedding与混合检索必须收到同一改写查询。"""

    source_result = make_hybrid_result()
    embedding_provider = FakeEmbeddingProvider(
        vector=[0.25, 0.75],
    )
    hybrid_retriever = FakeHybridRetriever(
        results=[source_result],
    )

    service = QueryRewritingHybridRetriever(
        embedding_provider=embedding_provider,
        hybrid_retriever=hybrid_retriever,
    )

    result = await service.retrieve(
        query=(
            "DM260的线缆靠近高压电源时"
            "有什么风险？"
        ),
        top_k=2,
    )

    assert isinstance(
        result,
        QueryRewritingRetrievalResult,
    )

    rewritten_query = (
        result.query_rewrite.rewritten_query
    )

    assert "DataMan 260" in rewritten_query
    assert "cable" in rewritten_query
    assert (
        "high-voltage power sources"
        in rewritten_query
    )

    # 向量路径接收改写查询。
    assert embedding_provider.received_queries == [
        rewritten_query
    ]

    # HybridRetriever内部的关键词路径也接收
    # 完全相同的改写查询。
    assert hybrid_retriever.received_queries == [
        rewritten_query
    ]

    assert (
        hybrid_retriever.received_embeddings
        == [[0.25, 0.75]]
    )
    assert (
        hybrid_retriever.received_top_k_values
        == [2]
    )
    assert hybrid_retriever.received_scopes == [
        None
    ]

    # 编排器返回tuple候选快照。
    assert isinstance(
        result.retrieved_chunks,
        tuple,
    )
    assert len(result.retrieved_chunks) == 1
    assert result.retrieved_chunks[0] == source_result

    # model_copy(deep=True)应复制外层结果和内部Chunk。
    assert result.retrieved_chunks[0] is not source_result
    assert (
        result.retrieved_chunks[0].chunk
        is not source_result.chunk
    )


@pytest.mark.asyncio
async def test_query_without_expansion_still_uses_normalization(
) -> None:
    """没有扩展规则时仍应使用归一化后的查询。"""

    embedding_provider = FakeEmbeddingProvider()
    hybrid_retriever = FakeHybridRetriever(
        results=[],
    )

    service = QueryRewritingHybridRetriever(
        embedding_provider=embedding_provider,
        hybrid_retriever=hybrid_retriever,
    )

    result = await service.retrieve(
        query="机器人当前位于哪里？",
    )

    assert result.query_rewrite.added_terms == ()
    assert (
        result.query_rewrite.rewritten_query
        == result.query_rewrite.normalized_query
    )
    assert embedding_provider.received_queries == [
        result.query_rewrite.normalized_query
    ]
    assert hybrid_retriever.received_queries == [
        result.query_rewrite.normalized_query
    ]


@pytest.mark.parametrize(
    "invalid_top_k",
    [
        True,
        1.5,
        "3",
        None,
    ],
    ids=[
        "boolean",
        "float",
        "string",
        "none",
    ],
)
@pytest.mark.asyncio
async def test_non_integer_top_k_is_rejected_before_embedding(
    invalid_top_k: object,
) -> None:
    """非整数top_k不能触发Embedding或检索调用。"""

    embedding_provider = FakeEmbeddingProvider()
    hybrid_retriever = FakeHybridRetriever(
        results=[],
    )
    service = QueryRewritingHybridRetriever(
        embedding_provider=embedding_provider,
        hybrid_retriever=hybrid_retriever,
    )

    with pytest.raises(
        TypeError,
        match="top_k必须是整数",
    ):
        await service.retrieve(
            query="测试问题",
            top_k=invalid_top_k,  # type: ignore[arg-type]
        )

    assert embedding_provider.received_queries == []
    assert hybrid_retriever.received_queries == []


@pytest.mark.parametrize(
    "invalid_top_k",
    [
        0,
        -1,
    ],
    ids=[
        "zero",
        "negative",
    ],
)
@pytest.mark.asyncio
async def test_non_positive_top_k_is_rejected_before_embedding(
    invalid_top_k: int,
) -> None:
    """零或负数top_k不能触发外部依赖。"""

    embedding_provider = FakeEmbeddingProvider()
    hybrid_retriever = FakeHybridRetriever(
        results=[],
    )
    service = QueryRewritingHybridRetriever(
        embedding_provider=embedding_provider,
        hybrid_retriever=hybrid_retriever,
    )

    with pytest.raises(
        ValueError,
        match="top_k必须大于或等于1",
    ):
        await service.retrieve(
            query="测试问题",
            top_k=invalid_top_k,
        )

    assert embedding_provider.received_queries == []
    assert hybrid_retriever.received_queries == []


@pytest.mark.asyncio
async def test_blank_query_is_rejected_before_embedding(
) -> None:
    """空白问题应由查询改写边界拒绝。"""

    embedding_provider = FakeEmbeddingProvider()
    hybrid_retriever = FakeHybridRetriever(
        results=[],
    )
    service = QueryRewritingHybridRetriever(
        embedding_provider=embedding_provider,
        hybrid_retriever=hybrid_retriever,
    )

    with pytest.raises(
        ValueError,
        match="query不能为空",
    ):
        await service.retrieve(
            query="   ",
        )

    assert embedding_provider.received_queries == []
    assert hybrid_retriever.received_queries == []


@pytest.mark.asyncio
async def test_non_list_retrieval_result_is_rejected(
) -> None:
    """混合检索器违反列表返回契约时应立即报错。"""

    embedding_provider = FakeEmbeddingProvider()
    hybrid_retriever = FakeHybridRetriever(
        results=(),
    )
    service = QueryRewritingHybridRetriever(
        embedding_provider=embedding_provider,
        hybrid_retriever=hybrid_retriever,
    )

    with pytest.raises(
        TypeError,
        match="hybrid_retriever必须返回列表",
    ):
        await service.retrieve(
            query="测试问题",
        )


@pytest.mark.asyncio
async def test_invalid_retrieval_item_is_rejected(
) -> None:
    """列表中的非HybridRetrievedChunk对象应被拒绝。"""

    embedding_provider = FakeEmbeddingProvider()
    hybrid_retriever = FakeHybridRetriever(
        results=[object()],
    )
    service = QueryRewritingHybridRetriever(
        embedding_provider=embedding_provider,
        hybrid_retriever=hybrid_retriever,
    )

    with pytest.raises(
        TypeError,
        match=(
            "混合检索结果中的第 0 项必须是"
            "HybridRetrievedChunk"
        ),
    ):
        await service.retrieve(
            query="测试问题",
        )


@pytest.mark.asyncio
async def test_empty_retrieval_result_is_allowed(
) -> None:
    """空知识库或无候选时允许返回空候选元组。"""

    embedding_provider = FakeEmbeddingProvider()
    hybrid_retriever = FakeHybridRetriever(
        results=[],
    )
    service = QueryRewritingHybridRetriever(
        embedding_provider=embedding_provider,
        hybrid_retriever=hybrid_retriever,
    )

    result = await service.retrieve(
        query="测试问题",
    )

    assert result.retrieved_chunks == ()


@pytest.mark.asyncio
async def test_embedding_error_propagates_without_retrieval(
) -> None:
    """Embedding失败后不能继续调用混合检索器。"""

    embedding_provider = FakeEmbeddingProvider(
        error=RuntimeError(
            "fake embedding failure"
        ),
    )
    hybrid_retriever = FakeHybridRetriever(
        results=[],
    )
    service = QueryRewritingHybridRetriever(
        embedding_provider=embedding_provider,
        hybrid_retriever=hybrid_retriever,
    )

    with pytest.raises(
        RuntimeError,
        match="fake embedding failure",
    ):
        await service.retrieve(
            query="测试问题",
        )

    assert len(
        embedding_provider.received_queries
    ) == 1
    assert hybrid_retriever.received_queries == []


@pytest.mark.asyncio
async def test_retrieval_result_container_is_immutable(
) -> None:
    """已返回结果不能被替换为另一份候选集合。"""

    service = QueryRewritingHybridRetriever(
        embedding_provider=FakeEmbeddingProvider(),
        hybrid_retriever=FakeHybridRetriever(
            results=[make_hybrid_result()],
        ),
    )

    result = await service.retrieve(
        query="测试问题",
    )

    with pytest.raises(
        FrozenInstanceError,
    ):
        result.retrieved_chunks = (  # type: ignore[misc]
        )


@pytest.mark.asyncio
async def test_retrieval_scope_is_forwarded_after_query_rewrite(
) -> None:
    """合法范围应越过改写和Embedding并原样到达混合检索器。"""

    embedding_provider = FakeEmbeddingProvider()
    hybrid_retriever = FakeHybridRetriever(
        results=[],
    )
    service = QueryRewritingHybridRetriever(
        embedding_provider=embedding_provider,
        hybrid_retriever=hybrid_retriever,
    )

    retrieval_scope = RetrievalScopeFilter(
        document_ids=("a" * 64,),
        source_files=("reader-manual.pdf",),
    )

    result = await service.retrieve(
        query="DM260线缆有什么注意事项？",
        top_k=2,
        retrieval_scope=retrieval_scope,
    )

    # Fake混合检索器返回空列表，编排器应将它转换成
    # 不可变的空tuple，而不是伪造候选。
    assert result.retrieved_chunks == ()

    # 合法请求必须先生成改写查询的Embedding。
    assert len(
        embedding_provider.received_queries
    ) == 1

    # 使用is验证没有复制、替换或重新解释范围对象。
    assert len(
        hybrid_retriever.received_scopes
    ) == 1
    assert (
        hybrid_retriever.received_scopes[0]
        is retrieval_scope
    )


@pytest.mark.asyncio
async def test_invalid_retrieval_scope_is_rejected_before_embedding(
) -> None:
    """普通dict范围必须在外部Embedding调用前被拒绝。"""

    embedding_provider = FakeEmbeddingProvider()
    hybrid_retriever = FakeHybridRetriever(
        results=[],
    )
    service = QueryRewritingHybridRetriever(
        embedding_provider=embedding_provider,
        hybrid_retriever=hybrid_retriever,
    )

    with pytest.raises(
        TypeError,
        match=(
            "retrieval_scope必须是"
            "RetrievalScopeFilter或None"
        ),
    ):
        await service.retrieve(
            query="测试非法检索范围",
            # dict没有执行RetrievalScopeFilter的
            # 文档ID、文件名、重复值和组合语义校验。
            retrieval_scope={  # type: ignore[arg-type]
                "source_files": ["reader-manual.pdf"]
            },
        )

    # 非法本地参数不能产生外部API费用，
    # 也不能让下游执行一半业务流程。
    assert embedding_provider.received_queries == []
    assert hybrid_retriever.received_queries == []
    assert hybrid_retriever.received_scopes == []
