"""混合检索知识问答编排Service的离线测试。

本模块只使用Fake门控检索器和Fake回答Provider，
不会访问真实Embedding、Chroma、LLM或HTTP服务。

测试范围：
1. 门控放行后调用LLM并构造可追溯引用；
2. 关键词单路候选不伪造向量相似度；
3. 门控拒绝后直接返回稳定拒答且跳过LLM；
4. 未知临时证据编号不能变成虚构引用；
5. 检索依赖和回答依赖必须遵守运行时返回契约；
6. 混合引用必须按最终排名稳定输出。
"""

from collections.abc import Sequence

import logging

import pytest

from app.errors import InvalidLLMResponseError
from app.schemas.knowledge_query import (
    KnowledgeAnswerDraft,
    KnowledgeQueryRequest,
)
from app.schemas.retrieval import (
    DocumentChunk,
    HybridRetrievedChunk,
)
from app.services.gated_hybrid_retrieval import (
    GatedHybridRetrievalResult,
)
from app.services.hybrid_knowledge_query import (
    HybridKnowledgeQueryService,
    build_hybrid_citations,
)
from app.services.hybrid_retrieval_gate import (
    HybridEvidenceGateDecision,
)
from app.services.knowledge_query import (
    INSUFFICIENT_EVIDENCE_ANSWER,
)
from app.services.query_rewriting import (
    RewrittenRetrievalQuery,
)
from app.services.retrieval_scope import (
    RetrievalScopeFilter,
)


# 测试使用固定门控版本，证明Service返回结果
# 来自当前已评测的组合证据门控策略。
TEST_GATE_VERSION = "hybrid-evidence-gate-v1"

# 两个文档ID都使用64位小写十六进制文本，
# 因为最终KnowledgeCitation会使用SHA-256格式校验。
PRIMARY_DOCUMENT_ID = "b" * 64
SECONDARY_DOCUMENT_ID = "c" * 64

# DocumentChunk要求content_hash为
# 64位小写十六进制文本。
TEST_CONTENT_HASH = "d" * 64


def make_dual_path_candidate(
) -> HybridRetrievedChunk:
    """创建向量和关键词双路命中的Top-1候选。"""

    return HybridRetrievedChunk(
        chunk=DocumentChunk(
            chunk_id=(
                f"{PRIMARY_DOCUMENT_ID}:000000"
            ),
            document_id=PRIMARY_DOCUMENT_ID,
            source_file="network-test-standard.md",
            page_or_section=(
                "section: TEST-NET-001"
            ),
            chunk_index=0,
            content_hash=TEST_CONTENT_HASH,
            content=(
                "通信恢复后不得自动继续旧任务；"
                "调度系统核对状态后才能恢复。"
            ),
        ),
        rrf_score=0.032787,
        rank=1,
        vector_rank=1,
        vector_similarity=0.62,
        keyword_rank=1,
        keyword_score=15.0,
        matched_identifiers=(
            "err-net-4001",
        ),
    )


def make_keyword_only_candidate(
) -> HybridRetrievedChunk:
    """创建只进入关键词候选集合的Top-2候选。"""

    return HybridRetrievedChunk(
        chunk=DocumentChunk(
            chunk_id=(
                f"{SECONDARY_DOCUMENT_ID}:000003"
            ),
            document_id=SECONDARY_DOCUMENT_ID,
            source_file="robot-faults.txt",
            page_or_section=(
                "section: ERR-NET-4001"
            ),
            chunk_index=3,
            content_hash="e" * 64,
            content=(
                "ERR-NET-4001表示心跳包超时。"
            ),
        ),
        rrf_score=0.016129,
        rank=2,

        # None明确表示该Chunk没有进入
        # 本次向量候选集合。
        vector_rank=None,
        vector_similarity=None,
        keyword_rank=2,
        keyword_score=12.0,
        matched_identifiers=(
            "err-net-4001",
        ),
    )


def make_query_rewrite(
) -> RewrittenRetrievalQuery:
    """创建门控结果携带的查询改写审计快照。"""

    return RewrittenRetrievalQuery(
        original_query=(
            "ERR-NET-4001网络恢复后"
            "如何继续任务？"
        ),
        normalized_query=(
            "err-net-4001网络恢复后"
            "如何继续任务?"
        ),
        rewritten_query=(
            "err-net-4001网络恢复后"
            "如何继续任务? "
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


def make_gated_result(
    *,
    accepted: bool,
) -> GatedHybridRetrievalResult:
    """创建门控放行或拒绝的完整检索结果。"""

    if accepted:
        # 放行样本保留精确错误码和足够向量支持。
        candidates = (
            make_dual_path_candidate(),
            make_keyword_only_candidate(),
        )
    else:
        # 拒答样本只有低向量相似度和普通主题词，
        # 不包含足以触发精确编号规则的标识符。
        weak_dual_path = (
            make_dual_path_candidate().model_copy(
                update={
                    "vector_similarity": 0.30,
                    "matched_identifiers": (),
                    "matched_lexical_terms": (
                        "网络",
                    ),
                }
            )
        )
        weak_keyword_only = (
            make_keyword_only_candidate().model_copy(
                update={
                    "matched_identifiers": (),
                    "matched_lexical_terms": (
                        "网络",
                    ),
                }
            )
        )
        candidates = (
            weak_dual_path,
            weak_keyword_only,
        )

    decision = HybridEvidenceGateDecision(
        accepted=accepted,
        reason=(
            "exact_identifier_support"
            if accepted
            else "insufficient_combined_support"
        ),
        policy_version=TEST_GATE_VERSION,
        evaluated_candidate_count=len(
            candidates
        ),
        top_rrf_score=(
            candidates[0].rrf_score
        ),
        top_vector_similarity=(
            candidates[0].vector_similarity
        ),
        max_vector_similarity=(
            candidates[0].vector_similarity
        ),
        dual_path_candidate_count=1,
        identifier_candidate_count=(
            2
            if accepted
            else 0
        ),
        model_lexical_candidate_count=0,
        supporting_chunk_ids=(
            (
                candidates[0].chunk.chunk_id,
            )
            if accepted
            else ()
        ),
    )

    return GatedHybridRetrievalResult(
        query_rewrite=make_query_rewrite(),
        retrieved_chunks=candidates,
        decision=decision,
    )


class FakeGatedHybridRetriever:
    """记录Service传入的查询参数并返回预设对象。"""

    def __init__(
        self,
        *,
        result: object,
    ) -> None:
        """保存预设结果并初始化调用记录。"""

        self.result = result
        self.received_queries: list[str] = []
        self.received_top_k_values: list[int] = []
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
    ) -> GatedHybridRetrievalResult:
        """模拟异步门控检索并记录收到的参数。"""

        self.received_queries.append(query)
        self.received_top_k_values.append(top_k)
        self.received_scopes.append(
            retrieval_scope
        )

        # 个别测试故意返回dict，
        # 验证Service不会只相信Protocol类型标注。
        return self.result  # type: ignore[return-value]


class FakeKnowledgeAnswerProvider:
    """记录进入LLM层的证据并返回预设回答草稿。"""

    def __init__(
        self,
        *,
        result: object,
    ) -> None:
        """保存预设草稿并初始化调用记录。"""

        self.result = result
        self.received_questions: list[str] = []
        self.received_evidence: list[
            tuple[HybridRetrievedChunk, ...]
        ] = []

    async def generate_answer(
        self,
        *,
        question: str,
        evidence: Sequence[
            HybridRetrievedChunk
        ],
    ) -> KnowledgeAnswerDraft:
        """模拟异步LLM回答并保存证据快照。"""

        self.received_questions.append(question)
        self.received_evidence.append(
            tuple(evidence)
        )

        # 个别测试故意返回错误类型，
        # 验证Service的运行时返回契约检查。
        return self.result  # type: ignore[return-value]


def make_answer_draft(
    *,
    evidence_id: str = "E2",
) -> KnowledgeAnswerDraft:
    """创建选择一条临时证据编号的合法LLM草稿。"""

    return KnowledgeAnswerDraft(
        answer=(
            "网络恢复后不能自动继续旧任务，"
            "需要先由调度系统核对状态。"
        ),
        used_evidence_ids=[
            evidence_id,
        ],
    )


def test_build_hybrid_citations_preserves_real_scores(
) -> None:
    """引用必须按rank排序并保留真实检索分数。"""

    dual_path = make_dual_path_candidate()
    keyword_only = (
        make_keyword_only_candidate()
    )

    # 故意按Top-2、Top-1的逆序传入，
    # 验证引用构造器会根据最终rank稳定排序。
    citations = build_hybrid_citations(
        [keyword_only, dual_path]
    )

    assert [
        citation.rank
        for citation in citations
    ] == [1, 2]

    # 双路候选同时保留向量相似度和RRF分数。
    assert citations[0].similarity == 0.62
    assert citations[0].rrf_score == (
        dual_path.rrf_score
    )

    # 关键词单路候选没有向量测量值，
    # 但仍保留真实RRF融合分数。
    assert citations[1].similarity is None
    assert citations[1].rrf_score == (
        keyword_only.rrf_score
    )
    assert citations[1].chunk_id == (
        keyword_only.chunk.chunk_id
    )
    assert citations[1].excerpt == (
        keyword_only.chunk.content
    )


@pytest.mark.asyncio
async def test_accepted_gate_calls_llm_and_maps_e2_to_real_chunk(
) -> None:
    """门控放行后应调用LLM并把E2映射成真实引用。"""

    request = KnowledgeQueryRequest(
        question=(
            "ERR-NET-4001网络恢复后"
            "如何继续任务？"
        ),
        top_k=2,
    )
    gated_result = make_gated_result(
        accepted=True
    )
    retriever = FakeGatedHybridRetriever(
        result=gated_result
    )
    answer_provider = (
        FakeKnowledgeAnswerProvider(
            result=make_answer_draft(
                evidence_id="E2"
            )
        )
    )
    service = HybridKnowledgeQueryService(
        retrieval_provider=retriever,
        answer_provider=answer_provider,
    )

    response = await service.answer_query(
        request
    )

    # Service必须使用原始用户问题和请求top_k
    # 调用门控混合检索器。
    assert retriever.received_queries == [
        request.question
    ]
    assert retriever.received_top_k_values == [
        2
    ]
    assert retriever.received_scopes == [None]

    # 门控放行后，全部Top-K候选进入LLM层，
    # 由模型选择回答真正使用的最小证据集合。
    assert answer_provider.received_questions == [
        request.question
    ]
    assert len(
        answer_provider.received_evidence
    ) == 1
    assert (
        answer_provider.received_evidence[0]
        == gated_result.retrieved_chunks
    )

    assert response.abstained is False
    assert response.answer == (
        make_answer_draft().answer
    )
    assert response.retrieval_ms >= 0.0

    # LLM只返回E2，Python必须把它映射为
    # 当前检索结果中rank=2的真实Chunk。
    assert len(response.citations) == 1
    citation = response.citations[0]
    keyword_only = (
        gated_result.retrieved_chunks[1]
    )
    assert citation.chunk_id == (
        keyword_only.chunk.chunk_id
    )
    assert citation.rank == 2
    assert citation.similarity is None
    assert citation.rrf_score == (
        keyword_only.rrf_score
    )


@pytest.mark.asyncio
async def test_rejected_gate_abstains_without_calling_llm(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """门控拒绝时必须直接拒答且不能调用LLM。"""

    request = KnowledgeQueryRequest(
        question="证据不足的问题",
        top_k=3,
    )
    retriever = FakeGatedHybridRetriever(
        result=make_gated_result(
            accepted=False
        )
    )
    answer_provider = (
        FakeKnowledgeAnswerProvider(
            result=make_answer_draft()
        )
    )
    service = HybridKnowledgeQueryService(
        retrieval_provider=retriever,
        answer_provider=answer_provider,
    )

    # caplog是pytest提供的日志捕获器。
    #
    # pytest默认不保留INFO日志，
    # 所以只为当前Service的logger开启INFO捕获。
    caplog.set_level(
        logging.INFO,
        logger=(
            "app.services."
            "hybrid_knowledge_query"
        ),
    )

    response = await service.answer_query(request)

    assert response.abstained is True
    assert response.answer == (
        INSUFFICIENT_EVIDENCE_ANSWER
    )
    assert response.citations == []
    assert response.retrieval_ms >= 0.0

    # 即使检索器返回了候选，
    # 只要组合证据门控拒绝，LLM调用次数仍必须为0。
    assert (
        answer_provider.received_questions
        == []
    )
    assert (
        answer_provider.received_evidence
        == []
    )

    # 预期日志说明拒答发生在gate阶段，
    # 并保留确定性门控的原因代码。
    assert "stage=gate" in caplog.text
    assert (
        "gate_reason="
        "insufficient_combined_support"
        in caplog.text
    )


@pytest.mark.asyncio
async def test_llm_secondary_abstention_returns_stable_response(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """门控误放行后，LLM拒答应转换成稳定的200响应数据。"""

    # 模拟门控已经放行候选证据。
    # 这对应q007当前出现的真实边界：
    # 形式信号足够通过门控，但正文并不支持问题答案。
    retriever = FakeGatedHybridRetriever(
        result=make_gated_result(
            accepted=True
        )
    )

    # 模拟LLM逐条阅读候选后发现证据不足。
    # 拒答草稿不能选择任何临时证据编号。
    answer_provider = (
        FakeKnowledgeAnswerProvider(
            result=KnowledgeAnswerDraft(
                answer=(
                    "知识库没有足够证据回答该问题。"
                ),
                used_evidence_ids=[],
                abstained=True,
            )
        )
    )
    service = HybridKnowledgeQueryService(
        retrieval_provider=retriever,
        answer_provider=answer_provider,
    )

    caplog.set_level(
        logging.INFO,
        logger=(
            "app.services."
            "hybrid_knowledge_query"
        ),
    )

    response = await service.answer_query(
        KnowledgeQueryRequest(
            question=(
                "DataMan 260的线缆为什么应远离"
                "大电流线路和高压电源？"
            ),
            top_k=3,
        )
    )

    # 第二道拒答必须复用应用定义的固定公开文案，
    # 不能直接暴露模型生成的任意拒答文字。
    assert response.answer == (
        INSUFFICIENT_EVIDENCE_ANSWER
    )
    assert response.abstained is True
    assert response.citations == []
    assert response.retrieval_ms >= 0.0

    # 与第一道门控拒答不同，第二道拒答发生在LLM之后，
    # 因此这里应当恰好留下一个问题和一组证据快照。
    assert len(
        answer_provider.received_questions
    ) == 1
    assert len(
        answer_provider.received_evidence
    ) == 1

    # 门控已经放行且Fake LLM确实被调用，
    # 所以日志必须标记为llm_secondary，
    # 不能误记为gate拒答。
    assert (
        "stage=llm_secondary"
        in caplog.text
    )
    assert "stage=gate " not in caplog.text


@pytest.mark.asyncio
async def test_unknown_evidence_id_is_rejected(
) -> None:
    """LLM选择不存在的E9时不能构造虚假引用。"""

    retriever = FakeGatedHybridRetriever(
        result=make_gated_result(
            accepted=True
        )
    )
    answer_provider = (
        FakeKnowledgeAnswerProvider(
            result=make_answer_draft(
                evidence_id="E9"
            )
        )
    )
    service = HybridKnowledgeQueryService(
        retrieval_provider=retriever,
        answer_provider=answer_provider,
    )

    with pytest.raises(
        InvalidLLMResponseError,
        match="未提供的证据编号.*E9",
    ):
        await service.answer_query(
            KnowledgeQueryRequest(
                question="测试未知引用"
            )
        )


@pytest.mark.asyncio
async def test_invalid_retrieval_result_stops_before_llm(
) -> None:
    """门控检索器返回dict时必须在LLM前停止。"""

    retriever = FakeGatedHybridRetriever(
        result={
            "retrieved_chunks": [],
        }
    )
    answer_provider = (
        FakeKnowledgeAnswerProvider(
            result=make_answer_draft()
        )
    )
    service = HybridKnowledgeQueryService(
        retrieval_provider=retriever,
        answer_provider=answer_provider,
    )

    with pytest.raises(
        TypeError,
        match=(
            "retrieval_provider必须返回"
            "GatedHybridRetrievalResult"
        ),
    ):
        await service.answer_query(
            KnowledgeQueryRequest(
                question="测试非法检索结果"
            )
        )

    assert (
        answer_provider.received_questions
        == []
    )


@pytest.mark.asyncio
async def test_invalid_answer_result_is_rejected(
) -> None:
    """回答Provider返回dict时应转换成统一无效响应异常。"""

    retriever = FakeGatedHybridRetriever(
        result=make_gated_result(
            accepted=True
        )
    )
    answer_provider = (
        FakeKnowledgeAnswerProvider(
            result={
                "answer": "缺少数据契约",
            }
        )
    )
    service = HybridKnowledgeQueryService(
        retrieval_provider=retriever,
        answer_provider=answer_provider,
    )

    with pytest.raises(
        InvalidLLMResponseError,
        match=(
            "知识库回答服务没有返回"
            "KnowledgeAnswerDraft"
        ),
    ):
        await service.answer_query(
            KnowledgeQueryRequest(
                question="测试非法回答结果"
            )
        )
