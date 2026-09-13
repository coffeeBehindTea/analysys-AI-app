"""search_knowledge纯检索边界的回归测试。

这些测试使用内存Fake门控检索器，不调用Embedding、Chroma、
生成式LLM或HTTP。它们专门防止search_knowledge重新退化成
“工具内部再调用一次知识问答LLM”的嵌套生成链。
"""

import pytest
from pydantic import ValidationError

from app.agent.evidence_store import ConfirmedEvidenceStore
from app.agent.tools.search_knowledge import (
    SearchKnowledgeToolHandler,
)
from app.schemas.agent_tools import (
    SearchKnowledgeToolInput,
    SearchKnowledgeToolOutput,
)
from app.schemas.retrieval import (
    DocumentChunk,
    HybridRetrievedChunk,
)
from app.services.gated_hybrid_retrieval import (
    GatedHybridRetrievalResult,
)
from app.services.hybrid_retrieval_gate import (
    HybridEvidenceGateDecision,
)
from app.services.query_rewriting import (
    RewrittenRetrievalQuery,
)
from app.services.retrieval_scope import (
    RetrievalScopeFilter,
)


# 固定的SHA-256格式值保证测试可重复且不依赖真实语料。
TEST_DOCUMENT_ID = "c" * 64
TEST_CONTENT_HASH = "d" * 64
TEST_QUERY = "ERR-NET-4001网络恢复后如何继续任务？"


def make_candidate(
    *,
    chunk_index: int,
    rank: int,
    content: str,
) -> HybridRetrievedChunk:
    """构造一条可追溯的双路融合检索候选。"""

    return HybridRetrievedChunk(
        chunk=DocumentChunk(
            chunk_id=(
                f"{TEST_DOCUMENT_ID}:"
                f"{chunk_index:06d}"
            ),
            document_id=TEST_DOCUMENT_ID,
            source_file="仓储机器人故障说明.txt",
            page_or_section="section: ERR-NET-4001",
            chunk_index=chunk_index,
            content_hash=TEST_CONTENT_HASH,
            content=content,
        ),
        rrf_score=0.032,
        rank=rank,
        vector_rank=rank,
        vector_similarity=0.72,
        keyword_rank=rank,
        keyword_score=10.0,
        matched_identifiers=(
            "err-net-4001",
        ),
    )


def make_result(
    *,
    accepted: bool,
    candidates: tuple[
        HybridRetrievedChunk,
        ...,
    ],
    supporting_chunk_ids: tuple[str, ...],
) -> GatedHybridRetrievalResult:
    """构造包含候选和可审计门控决定的结果。"""

    return GatedHybridRetrievalResult(
        query_rewrite=RewrittenRetrievalQuery(
            original_query=TEST_QUERY,
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
        ),
        retrieved_chunks=candidates,
        decision=HybridEvidenceGateDecision(
            accepted=accepted,
            reason=(
                "exact_identifier_support"
                if accepted
                else "insufficient_combined_support"
            ),
            policy_version=(
                "hybrid-evidence-gate-v1"
            ),
            evaluated_candidate_count=(
                len(candidates)
            ),
            top_rrf_score=(
                candidates[0].rrf_score
                if candidates
                else None
            ),
            top_vector_similarity=(
                candidates[0].vector_similarity
                if candidates
                else None
            ),
            max_vector_similarity=(
                max(
                    candidate.vector_similarity
                    for candidate in candidates
                    if candidate.vector_similarity
                    is not None
                )
                if candidates
                else None
            ),
            dual_path_candidate_count=(
                len(candidates)
            ),
            identifier_candidate_count=(
                len(candidates)
            ),
            model_lexical_candidate_count=0,
            supporting_chunk_ids=(
                supporting_chunk_ids
            ),
        ),
    )


class FakeGatedRetrievalProvider:
    """只提供异步retrieve()的Fake，故意没有answer_query()。"""

    def __init__(
        self,
        *,
        result: object,
    ) -> None:
        self.result = result
        self.calls: list[
            tuple[
                str,
                int,
                RetrievalScopeFilter | None,
            ]
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
        """记录参数并返回预设结果。"""

        self.calls.append(
            (query, top_k, retrieval_scope)
        )
        return self.result  # type: ignore[return-value]


def test_output_contract_is_evidence_only(
) -> None:
    """输出契约必须删除嵌套知识回答LLM产生的answer字段。"""

    candidate = make_candidate(
        chunk_index=3,
        rank=1,
        content="网络恢复后必须由调度系统核对状态。",
    )

    with pytest.raises(ValidationError) as exc_info:
        SearchKnowledgeToolOutput.model_validate(
            {
                "answer": "检索工具不应生成最终答案",
                "citations": [
                    {
                        "chunk_id": (
                            candidate.chunk.chunk_id
                        ),
                        "document_id": (
                            candidate.chunk.document_id
                        ),
                        "source_file": (
                            candidate.chunk.source_file
                        ),
                        "page_or_section": (
                            candidate.chunk.page_or_section
                        ),
                        "chunk_index": (
                            candidate.chunk.chunk_index
                        ),
                        "rank": candidate.rank,
                        "similarity": (
                            candidate.vector_similarity
                        ),
                        "rrf_score": (
                            candidate.rrf_score
                        ),
                        "excerpt": (
                            candidate.chunk.content
                        ),
                    }
                ],
                "retrieval_ms": 1.0,
            }
        )

    assert "extra_forbidden" in {
        error["type"]
        for error in exc_info.value.errors()
    }


@pytest.mark.asyncio
async def test_handler_accepts_retrieval_only_provider(
) -> None:
    """Handler应依赖retrieve()，不能再要求answer_query()。"""

    candidate = make_candidate(
        chunk_index=3,
        rank=1,
        content="网络恢复后必须由调度系统核对状态。",
    )
    provider = FakeGatedRetrievalProvider(
        result=make_result(
            accepted=True,
            candidates=(candidate,),
            supporting_chunk_ids=(
                candidate.chunk.chunk_id,
            ),
        )
    )
    evidence_store = ConfirmedEvidenceStore()
    handler = SearchKnowledgeToolHandler(
        retrieval_provider=provider,
        evidence_store=evidence_store,
    )

    output = await handler(
        SearchKnowledgeToolInput(
            query=TEST_QUERY,
            top_k=5,
        )
    )

    assert provider.calls == [
        (TEST_QUERY, 5, None)
    ]
    assert isinstance(
        output,
        SearchKnowledgeToolOutput,
    )
    assert "answer" not in output.model_dump()
    assert output.citations[0].chunk_id == (
        candidate.chunk.chunk_id
    )


@pytest.mark.asyncio
async def test_handler_records_only_gate_supporting_chunks(
) -> None:
    """门控放行后只允许真正支持放行的Chunk进入白名单。"""

    supported = make_candidate(
        chunk_index=3,
        rank=1,
        content="ERR-NET-4001的恢复条件。",
    )
    background = make_candidate(
        chunk_index=4,
        rank=2,
        content="同一文档中的无关背景信息。",
    )
    provider = FakeGatedRetrievalProvider(
        result=make_result(
            accepted=True,
            candidates=(supported, background),
            supporting_chunk_ids=(
                supported.chunk.chunk_id,
            ),
        )
    )
    evidence_store = ConfirmedEvidenceStore()
    handler = SearchKnowledgeToolHandler(
        retrieval_provider=provider,
        evidence_store=evidence_store,
    )

    output = await handler(
        SearchKnowledgeToolInput(
            query=TEST_QUERY
        )
    )

    assert output is not None
    assert [
        citation.chunk_id
        for citation in output.citations
    ] == [supported.chunk.chunk_id]
    assert supported.chunk.chunk_id in (
        evidence_store
    )
    assert background.chunk.chunk_id not in (
        evidence_store
    )


@pytest.mark.asyncio
async def test_handler_maps_gate_rejection_to_empty_without_recording(
) -> None:
    """确定性门控拒绝时应返回None且不写入证据白名单。"""

    candidate = make_candidate(
        chunk_index=3,
        rank=1,
        content="只有弱相关性的候选。",
    )
    provider = FakeGatedRetrievalProvider(
        result=make_result(
            accepted=False,
            candidates=(candidate,),
            supporting_chunk_ids=(),
        )
    )
    evidence_store = ConfirmedEvidenceStore()
    handler = SearchKnowledgeToolHandler(
        retrieval_provider=provider,
        evidence_store=evidence_store,
    )

    output = await handler(
        SearchKnowledgeToolInput(
            query="证据不足的问题"
        )
    )

    assert output is None
    assert len(evidence_store) == 0


@pytest.mark.asyncio
async def test_handler_rejects_unknown_supporting_chunk_id(
) -> None:
    """门控引用不存在的候选ID时不能产生部分白名单写入。"""

    candidate = make_candidate(
        chunk_index=3,
        rank=1,
        content="真实存在的候选。",
    )
    provider = FakeGatedRetrievalProvider(
        result=make_result(
            accepted=True,
            candidates=(candidate,),
            supporting_chunk_ids=(
                "unknown:000001",
            ),
        )
    )
    evidence_store = ConfirmedEvidenceStore()
    handler = SearchKnowledgeToolHandler(
        retrieval_provider=provider,
        evidence_store=evidence_store,
    )

    with pytest.raises(
        ValueError,
        match="门控支持的Chunk",
    ):
        await handler(
            SearchKnowledgeToolInput(
                query=TEST_QUERY
            )
        )

    assert len(evidence_store) == 0
