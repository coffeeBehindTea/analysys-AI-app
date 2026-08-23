"""查询改写混合检索与证据门控编排器的离线测试。

本模块使用Fake查询改写检索器，不访问真实Embedding、
Chroma、关键词索引、LLM或HTTP服务。

测试范围：
1. 原始问题和top_k是否正确传给下游检索器；
2. 可选检索范围是否原样传给下游检索器；
3. 精确错误码证据是否触发放行；
4. 弱证据和空候选是否触发拒绝；
5. 下游返回错误类型时是否在边界处失败；
6. 下游异常是否保留原始异常类型向上传播；
7. 返回候选是否形成独立深复制快照；
8. 构造器是否拒绝非法门控策略对象。
"""

import pytest

from app.schemas.retrieval import (
    DocumentChunk,
    HybridRetrievedChunk,
)
from app.services.gated_hybrid_retrieval import (
    GatedHybridRetrievalResult,
    GatedHybridRetriever,
)
from app.services.hybrid_retrieval_gate import (
    HybridEvidenceGatePolicy,
)
from app.services.query_rewriting import (
    RewrittenRetrievalQuery,
)
from app.services.query_rewriting_retrieval import (
    QueryRewritingRetrievalResult,
)
from app.services.retrieval_scope import (
    RetrievalScopeFilter,
)


# DocumentChunk要求正文哈希为64位小写十六进制文本。
TEST_CONTENT_HASH = "d" * 64


def make_hybrid_candidate(
    *,
    vector_similarity: float,
    matched_identifiers: tuple[str, ...] = (),
    matched_lexical_terms: tuple[str, ...] = (),
) -> HybridRetrievedChunk:
    """构造一条同时来自向量和关键词路径的候选。"""

    chunk = DocumentChunk(
        chunk_id="gated-document:000001",
        document_id="gated-document",
        source_file="robot-faults.txt",
        page_or_section="section: ERR-NET-4001",
        chunk_index=1,
        content_hash=TEST_CONTENT_HASH,
        content=(
            "ERR-NET-4001网络恢复后，"
            "需要等待调度系统确认。"
        ),
    )

    return HybridRetrievedChunk(
        chunk=chunk,
        rrf_score=0.032,
        rank=1,
        vector_rank=1,
        vector_similarity=vector_similarity,
        keyword_rank=1,
        keyword_score=10.0,
        matched_identifiers=matched_identifiers,
        matched_lexical_terms=matched_lexical_terms,
    )


def make_retrieval_result(
    candidates: tuple[
        HybridRetrievedChunk,
        ...,
    ],
) -> QueryRewritingRetrievalResult:
    """构造带查询改写审计信息的下游检索结果。"""

    query_rewrite = RewrittenRetrievalQuery(
        original_query=(
            "ERR-NET-4001网络恢复后如何继续任务？"
        ),
        normalized_query=(
            "err-net-4001网络恢复后如何继续任务?"
        ),
        rewritten_query=(
            "err-net-4001网络恢复后如何继续任务? "
            "network recovery resume task"
        ),
        added_terms=(
            "network recovery",
            "resume task",
        ),
        applied_rule_ids=(
            "identifier-err-net-4001",
        ),
        rewrite_version="deterministic-v1",
    )

    return QueryRewritingRetrievalResult(
        query_rewrite=query_rewrite,
        retrieved_chunks=candidates,
    )


class FakeQueryRewritingRetriever:
    """记录调用参数并返回预设对象或异常。"""

    def __init__(
        self,
        *,
        result: object | None = None,
        error: Exception | None = None,
    ) -> None:
        """保存预设结果、异常和调用记录。"""

        self.result = result
        self.error = error
        self.received_queries: list[str] = []
        self.received_top_k_values: list[int] = []
        # 记录每次调用使用的范围；
        # None表示该次调用使用全库检索。
        self.received_scopes: list[
            RetrievalScopeFilter | None
        ] = []

    async def retrieve(
        self,
        *,
        query: str,
        top_k: int = 3,
        retrieval_scope: (
            RetrievalScopeFilter | None
        ) = None,
    ) -> QueryRewritingRetrievalResult:
        """模拟异步查询改写混合检索并记录范围。"""

        self.received_queries.append(query)
        self.received_top_k_values.append(top_k)
        self.received_scopes.append(
            retrieval_scope
        )

        if self.error is not None:
            raise self.error

        # 部分测试故意返回错误类型，
        # 用于验证编排器的运行时边界检查。
        return self.result  # type: ignore[return-value]


@pytest.mark.asyncio
async def test_exact_identifier_candidate_is_accepted(
) -> None:
    """精确错误码加足够向量支持应通过组合门控。"""

    source_candidate = make_hybrid_candidate(
        vector_similarity=0.46,
        matched_identifiers=(
            "err-net-4001",
        ),
    )
    fake_retriever = FakeQueryRewritingRetriever(
        result=make_retrieval_result(
            (source_candidate,)
        )
    )
    service = GatedHybridRetriever(
        retrieval_provider=fake_retriever,
        policy=HybridEvidenceGatePolicy(),
    )

    question = (
        "ERR-NET-4001网络恢复后如何继续任务？"
    )
    result = await service.retrieve(
        query=question,
        top_k=3,
    )

    assert isinstance(
        result,
        GatedHybridRetrievalResult,
    )
    assert fake_retriever.received_queries == [
        question
    ]
    assert (
        fake_retriever.received_top_k_values
        == [3]
    )
    assert fake_retriever.received_scopes == [
        None
    ]
    assert result.decision.accepted is True
    assert result.decision.reason == (
        "exact_identifier_support"
    )
    assert result.decision.supporting_chunk_ids == (
        source_candidate.chunk.chunk_id,
    )
    assert result.query_rewrite.rewrite_version == (
        "deterministic-v1"
    )


@pytest.mark.asyncio
async def test_weak_candidate_is_rejected(
) -> None:
    """只有弱向量与普通词命中的候选不能进入回答阶段。"""

    weak_candidate = make_hybrid_candidate(
        vector_similarity=0.30,
        matched_lexical_terms=("网络",),
    )
    fake_retriever = FakeQueryRewritingRetriever(
        result=make_retrieval_result(
            (weak_candidate,)
        )
    )
    service = GatedHybridRetriever(
        retrieval_provider=fake_retriever,
        policy=HybridEvidenceGatePolicy(),
    )

    result = await service.retrieve(
        query="普通网络问题",
        top_k=1,
    )

    assert result.decision.accepted is False
    assert result.decision.reason == (
        "insufficient_combined_support"
    )
    assert result.decision.supporting_chunk_ids == ()


@pytest.mark.asyncio
async def test_empty_candidates_return_no_candidates(
) -> None:
    """空知识库结果必须形成稳定拒绝决定。"""

    fake_retriever = FakeQueryRewritingRetriever(
        result=make_retrieval_result(())
    )
    service = GatedHybridRetriever(
        retrieval_provider=fake_retriever,
        policy=HybridEvidenceGatePolicy(),
    )

    result = await service.retrieve(
        query="知识库中不存在的问题",
        top_k=3,
    )

    assert result.retrieved_chunks == ()
    assert result.decision.accepted is False
    assert result.decision.reason == (
        "no_candidates"
    )
    assert (
        result.decision.evaluated_candidate_count
        == 0
    )


@pytest.mark.asyncio
async def test_invalid_provider_result_is_rejected(
) -> None:
    """下游违反Protocol返回约定时必须立即失败。"""

    fake_retriever = FakeQueryRewritingRetriever(
        result={"retrieved_chunks": []}
    )
    service = GatedHybridRetriever(
        retrieval_provider=fake_retriever,
        policy=HybridEvidenceGatePolicy(),
    )

    with pytest.raises(
        TypeError,
        match=(
            "retrieval_provider必须返回"
            "QueryRewritingRetrievalResult"
        ),
    ):
        await service.retrieve(
            query="测试非法返回值",
            top_k=3,
        )


@pytest.mark.asyncio
async def test_provider_exception_is_not_hidden(
) -> None:
    """Embedding或检索异常应保留原始类型交给上层处理。"""

    expected_error = RuntimeError(
        "模拟下游检索失败"
    )
    fake_retriever = FakeQueryRewritingRetriever(
        error=expected_error
    )
    service = GatedHybridRetriever(
        retrieval_provider=fake_retriever,
        policy=HybridEvidenceGatePolicy(),
    )

    with pytest.raises(RuntimeError) as exc_info:
        await service.retrieve(
            query="测试异常传播",
            top_k=3,
        )

    assert exc_info.value is expected_error


@pytest.mark.asyncio
async def test_result_owns_deep_candidate_snapshot(
) -> None:
    """下游候选后续被修改时不能污染已返回结果。"""

    source_candidate = make_hybrid_candidate(
        vector_similarity=0.55,
        matched_lexical_terms=("网络",),
    )
    fake_retriever = FakeQueryRewritingRetriever(
        result=make_retrieval_result(
            (source_candidate,)
        )
    )
    service = GatedHybridRetriever(
        retrieval_provider=fake_retriever,
        policy=HybridEvidenceGatePolicy(),
    )

    result = await service.retrieve(
        query="网络恢复要求",
        top_k=1,
    )

    # Pydantic模型默认允许字段赋值。
    # 修改源对象，用来证明Service返回的是深复制快照。
    source_candidate.chunk.content = (
        "调用完成后被修改的正文"
    )

    assert result.retrieved_chunks[0] is not (
        source_candidate
    )
    assert (
        result.retrieved_chunks[0].chunk.content
        != source_candidate.chunk.content
    )


def test_constructor_rejects_invalid_policy(
) -> None:
    """普通字典不能冒充版本化门控策略。"""

    fake_retriever = FakeQueryRewritingRetriever(
        result=make_retrieval_result(())
    )

    with pytest.raises(
        TypeError,
        match="policy必须是HybridEvidenceGatePolicy",
    ):
        GatedHybridRetriever(
            retrieval_provider=fake_retriever,
            policy={},  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_retrieval_scope_is_forwarded_before_gate_evaluation(
) -> None:
    """合法范围应原样下传，门控只评估范围内返回的候选。"""

    fake_retriever = FakeQueryRewritingRetriever(
        result=make_retrieval_result(())
    )
    service = GatedHybridRetriever(
        retrieval_provider=fake_retriever,
        policy=HybridEvidenceGatePolicy(),
    )

    retrieval_scope = RetrievalScopeFilter(
        document_ids=("a" * 64,),
        source_files=("robot-faults.txt",),
    )

    result = await service.retrieve(
        query="限定文档内的故障问题",
        top_k=3,
        retrieval_scope=retrieval_scope,
    )

    # is验证下游收到同一个不可变范围对象，
    # 而不是由门控层自行重建或改变过滤规则。
    assert len(fake_retriever.received_scopes) == 1
    assert (
        fake_retriever.received_scopes[0]
        is retrieval_scope
    )

    # Fake下游返回空候选，门控仍需生成稳定拒绝决定。
    assert result.retrieved_chunks == ()
    assert result.decision.accepted is False
    assert result.decision.reason == "no_candidates"


@pytest.mark.asyncio
async def test_invalid_retrieval_scope_is_rejected_before_provider(
) -> None:
    """非法范围必须在异步检索提供者被调用前拒绝。"""

    fake_retriever = FakeQueryRewritingRetriever(
        result=make_retrieval_result(())
    )
    service = GatedHybridRetriever(
        retrieval_provider=fake_retriever,
        policy=HybridEvidenceGatePolicy(),
    )

    with pytest.raises(
        TypeError,
        match=(
            "retrieval_scope必须是"
            "RetrievalScopeFilter或None"
        ),
    ):
        await service.retrieve(
            query="测试非法范围",
            retrieval_scope={  # type: ignore[arg-type]
                "document_ids": ["a" * 64]
            },
        )

    # 门控层必须快速失败，不能先启动查询改写、
    # Embedding或任何候选检索步骤。
    assert fake_retriever.received_queries == []
    assert fake_retriever.received_top_k_values == []
    assert fake_retriever.received_scopes == []
