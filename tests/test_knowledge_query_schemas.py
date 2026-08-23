"""知识库问答请求、引用和响应Schema测试。"""

import pytest

# ValidationError表示Pydantic数据校验失败。
from pydantic import ValidationError

from app.schemas.knowledge_query import (
    KnowledgeCitation,
    KnowledgeQueryRequest,
    KnowledgeQueryResponse,
    KnowledgeAnswerDraft,
)


def make_citation() -> KnowledgeCitation:
    """创建供多个测试复用的合法引用。"""

    document_id = "a" * 64

    return KnowledgeCitation(
        chunk_id=f"{document_id}:000003",
        document_id=document_id,
        source_file="工业移动机器人安全标准.pdf",
        page_or_section="page: 12",
        chunk_index=3,
        rank=1,
        similarity=0.82,
        excerpt="急停装置复位后，应检查安全控制系统状态。",
    )


def test_query_request_strips_question_and_uses_default_top_k(
) -> None:
    """请求应清理问题空白并默认检索Top-3。"""

    request = KnowledgeQueryRequest(
        question="  急停复位后应检查什么？  ",
    )

    assert request.question == (
        "急停复位后应检查什么？"
    )
    assert request.top_k == 3


def test_query_request_rejects_invalid_top_k() -> None:
    """top_k超过接口上限时应校验失败。"""

    with pytest.raises(
        ValidationError,
        match="top_k",
    ):
        KnowledgeQueryRequest(
            question="测试问题",
            top_k=11,
        )


def test_answered_response_requires_citation(
) -> None:
    """非拒答响应必须包含至少一条真实引用。"""

    response = KnowledgeQueryResponse(
        answer="应检查安全控制系统状态。",
        citations=[make_citation()],
        retrieval_ms=25.5,
        abstained=False,
    )

    assert response.abstained is False
    assert len(response.citations) == 1
    assert response.citations[0].rank == 1
    assert response.citations[0].similarity == 0.82

    # 纯向量基线引用没有RRF融合分数。
    # 新字段必须保持可选，避免破坏Week 2响应构造。
    assert response.citations[0].rrf_score is None


def test_hybrid_citation_accepts_keyword_only_candidate(
) -> None:
    """关键词独占候选应保留RRF分数，并允许没有向量相似度。"""

    document_id = "b" * 64

    citation = KnowledgeCitation(
        chunk_id=f"{document_id}:000007",
        document_id=document_id,
        source_file="仓储机器人故障说明.txt",
        page_or_section="section: ERR-NET-4001",
        chunk_index=7,
        rank=2,

        # None表示该Chunk没有进入本次向量候选集合，
        # 不能伪造成余弦相似度0.0。
        similarity=None,

        # RRF分数来自混合检索的最终融合结果，
        # 只用于解释排名，不表示概率或置信度。
        rrf_score=0.031754,
        excerpt="ERR-NET-4001表示调度心跳超时。",
    )

    assert citation.similarity is None
    assert citation.rrf_score == 0.031754


def test_hybrid_citation_rejects_non_positive_rrf_score(
) -> None:
    """RRF候选至少来自一条检索路径，因此分数必须大于零。"""

    citation_data = make_citation().model_dump()

    # 纯向量引用本来没有RRF分数，
    # 这里故意写入非法的0.0验证数值边界。
    citation_data["rrf_score"] = 0.0

    with pytest.raises(
        ValidationError,
        match="rrf_score",
    ) as exc_info:
        KnowledgeCitation.model_validate(
            citation_data
        )

    # 字段已经存在但数值越界时，
    # Pydantic应给出greater_than，
    # 不能只是因为字段未声明而给出extra_forbidden。
    error_types = {
        item["type"]
        for item in exc_info.value.errors(
            include_url=False
        )
    }

    assert error_types == {
        "greater_than",
    }


def test_citation_requires_at_least_one_retrieval_score(
) -> None:
    """引用必须保留向量相似度或RRF分数中的至少一种。"""

    citation_data = make_citation().model_dump()
    citation_data["similarity"] = None
    citation_data["rrf_score"] = None

    with pytest.raises(
        ValidationError,
        match="至少需要一种检索分数",
    ):
        KnowledgeCitation.model_validate(
            citation_data
        )


def test_abstained_response_accepts_empty_citations(
) -> None:
    """证据不足时应允许固定说明和空引用。"""

    response = KnowledgeQueryResponse(
        answer="知识库没有足够证据回答该问题。",
        citations=[],
        retrieval_ms=12.3,
        abstained=True,
    )

    assert response.abstained is True
    assert response.citations == []


def test_abstained_response_rejects_citations(
) -> None:
    """拒答响应携带引用属于矛盾状态。"""

    with pytest.raises(
        ValidationError,
        match="拒答响应不能包含引用",
    ):
        KnowledgeQueryResponse(
            answer="知识库没有足够证据。",
            citations=[make_citation()],
            retrieval_ms=10.0,
            abstained=True,
        )


def test_answered_response_rejects_empty_citations(
) -> None:
    """声称已经回答但没有证据时应校验失败。"""

    with pytest.raises(
        ValidationError,
        match="非拒答响应必须包含",
    ):
        KnowledgeQueryResponse(
            answer="这是一个没有证据的回答。",
            citations=[],
            retrieval_ms=10.0,
            abstained=False,
        )


def test_answer_draft_accepts_selected_evidence(
) -> None:
    """内部LLM契约应保存回答和证据编号。"""

    draft = KnowledgeAnswerDraft(
        answer="应检查安全控制系统状态。",
        used_evidence_ids=["E2"],
    )

    assert draft.answer == (
        "应检查安全控制系统状态。"
    )
    assert draft.used_evidence_ids == [
        "E2"
    ]
    assert draft.abstained is False


def test_answer_draft_accepts_secondary_abstention(
) -> None:
    """LLM发现候选证据不足时应能返回合法拒答草稿。"""

    # 这个草稿模拟第二道拒答出口：
    # 检索门控已经放行，但是LLM逐条阅读候选后
    # 发现这些候选仍不足以支持问题要求的事实。
    draft = KnowledgeAnswerDraft(
        answer="知识库没有足够证据回答该问题。",
        used_evidence_ids=[],
        abstained=True,
    )

    # 拒答草稿不能声称使用了任何证据。
    assert draft.used_evidence_ids == []
    assert draft.abstained is True


def test_answer_draft_rejects_answer_without_evidence(
) -> None:
    """非拒答草稿没有证据编号时必须校验失败。"""

    # abstained=False表示模型声称已经回答。
    # 在这种状态下，空证据列表无法支撑回答，
    # 因而必须由跨字段校验器拒绝。
    with pytest.raises(
        ValidationError,
        match="非拒答草稿必须选择至少一条证据",
    ):
        KnowledgeAnswerDraft(
            answer="这是一个没有证据支撑的回答。",
            used_evidence_ids=[],
            abstained=False,
        )


def test_answer_draft_rejects_abstention_with_evidence(
) -> None:
    """拒答草稿携带证据编号时必须校验失败。"""

    # abstained=True和非空used_evidence_ids表达了
    # “既拒答又引用证据”的矛盾状态。
    with pytest.raises(
        ValidationError,
        match="拒答草稿不能选择证据",
    ):
        KnowledgeAnswerDraft(
            answer="知识库没有足够证据回答该问题。",
            used_evidence_ids=["E1"],
            abstained=True,
        )


def test_answer_draft_rejects_duplicate_evidence_ids(
) -> None:
    """LLM不能重复选择同一证据。"""

    with pytest.raises(
        ValidationError,
        match="不能包含重复编号",
    ):
        KnowledgeAnswerDraft(
            answer="测试回答",
            used_evidence_ids=[
                "E2",
                "E2",
            ],
        )
