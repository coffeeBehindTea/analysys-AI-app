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