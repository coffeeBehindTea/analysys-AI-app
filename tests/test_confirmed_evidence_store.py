"""当前Agent请求已确认证据Store的离线测试。

测试验证白名单记录、深复制、冲突拒绝、批量原子性和
多证据解析；不访问知识库、Chroma、Embedding或LLM。
"""

import pytest

from app.agent.evidence_store import (
    ConflictingConfirmedEvidenceError,
    ConfirmedEvidenceStore,
)
from app.schemas.knowledge_query import (
    KnowledgeCitation,
)


# 固定文档ID满足KnowledgeCitation的SHA-256格式契约。
DOCUMENT_ID_A = "a" * 64
DOCUMENT_ID_B = "b" * 64


def make_citation(
    *,
    document_id: str = DOCUMENT_ID_A,
    chunk_index: int = 1,
    source_file: str = "故障说明.txt",
    excerpt: str = "ERR-NET-4001恢复条件说明。",
) -> KnowledgeCitation:
    """创建可按参数区分的固定真实引用结构。"""

    return KnowledgeCitation(
        chunk_id=(
            f"{document_id}:"
            f"{chunk_index:06d}"
        ),
        document_id=document_id,
        source_file=source_file,
        page_or_section=(
            f"section: {chunk_index}"
        ),
        chunk_index=chunk_index,
        rank=chunk_index,
        similarity=0.7,
        rrf_score=0.03,
        excerpt=excerpt,
    )


def test_new_store_is_empty(
) -> None:
    """新请求必须从空白名单开始。"""

    store = ConfirmedEvidenceStore()

    assert len(store) == 0
    assert "unknown:000001" not in store
    assert store.get("unknown:000001") is None


def test_record_preserves_input_order_and_indexes_citations(
) -> None:
    """成功批次应按输入顺序返回ID并可精确查询。"""

    first = make_citation()
    second = make_citation(
        document_id=DOCUMENT_ID_B,
        chunk_index=2,
    )
    store = ConfirmedEvidenceStore()

    recorded_ids = store.record(
        [first, second]
    )

    assert recorded_ids == (
        first.chunk_id,
        second.chunk_id,
    )
    assert len(store) == 2
    assert first.chunk_id in store
    assert second.chunk_id in store
    assert store.get(first.chunk_id) == first


def test_record_deep_copies_input_citation(
) -> None:
    """记录后修改调用方对象不能改变Store内部白名单。"""

    citation = make_citation()
    original_source = citation.source_file
    store = ConfirmedEvidenceStore()
    store.record([citation])

    # KnowledgeCitation不是frozen模型，调用方可以改字段。
    citation.source_file = "被外部修改.txt"

    stored = store.get(citation.chunk_id)
    assert stored is not None
    assert stored.source_file == original_source


def test_get_returns_deep_copy(
) -> None:
    """修改get()返回值也不能反向污染Store。"""

    citation = make_citation()
    store = ConfirmedEvidenceStore()
    store.record([citation])

    first_read = store.get(citation.chunk_id)
    assert first_read is not None
    first_read.excerpt = "被读取方修改的正文"

    second_read = store.get(citation.chunk_id)
    assert second_read is not None
    assert second_read.excerpt == citation.excerpt


def test_recording_identical_citation_is_idempotent(
) -> None:
    """重复记录完全相同引用不应制造重复白名单项。"""

    citation = make_citation()
    store = ConfirmedEvidenceStore()

    store.record([citation])
    store.record([
        citation.model_copy(deep=True)
    ])

    assert len(store) == 1
    assert store.get(citation.chunk_id) == citation


def test_same_chunk_with_new_retrieval_scores_is_idempotent(
) -> None:
    """再次检索导致排名变化时不能误判为证据正文冲突。"""

    first_query_citation = make_citation()
    second_query_citation = (
        first_query_citation.model_copy(
            deep=True,
            update={
                "rank": 3,
                "similarity": 0.41,
                "rrf_score": 0.021,
            },
        )
    )
    store = ConfirmedEvidenceStore()
    store.record([first_query_citation])

    store.record([second_query_citation])

    # 白名单仍只保存一个稳定Chunk，并保留首次确认快照。
    assert len(store) == 1
    assert store.get(
        first_query_citation.chunk_id
    ) == first_query_citation


def test_conflicting_existing_citation_is_rejected(
) -> None:
    """相同Chunk ID不能对应两份不同正文或元数据。"""

    original = make_citation()
    conflicting = make_citation(
        excerpt="同一ID对应的冲突正文。"
    )
    store = ConfirmedEvidenceStore()
    store.record([original])

    with pytest.raises(
        ConflictingConfirmedEvidenceError,
        match="同一chunk_id出现冲突引用",
    ):
        store.record([conflicting])

    assert len(store) == 1
    assert store.get(original.chunk_id) == original


def test_conflicting_batch_is_atomic(
) -> None:
    """批次后项冲突时，前面的新引用也不能被部分写入。"""

    existing = make_citation()
    new_citation = make_citation(
        document_id=DOCUMENT_ID_B,
        chunk_index=2,
    )
    conflicting = make_citation(
        excerpt="冲突正文"
    )
    store = ConfirmedEvidenceStore()
    store.record([existing])

    with pytest.raises(
        ConflictingConfirmedEvidenceError
    ):
        store.record([
            new_citation,
            conflicting,
        ])

    assert len(store) == 1
    assert store.get(
        new_citation.chunk_id
    ) is None


def test_invalid_batch_item_is_atomic(
) -> None:
    """批次包含普通dict时不能留下之前处理的有效引用。"""

    citation = make_citation()
    store = ConfirmedEvidenceStore()

    with pytest.raises(
        TypeError,
        match=(
            "citations中的第 1 项必须是"
            "KnowledgeCitation"
        ),
    ):
        store.record([
            citation,
            {"chunk_id": "forged"},
        ])

    assert len(store) == 0
    assert store.get(citation.chunk_id) is None


def test_resolve_many_preserves_requested_order(
) -> None:
    """草案工具应按LLM提交的合法ID顺序取得真实引用。"""

    first = make_citation()
    second = make_citation(
        document_id=DOCUMENT_ID_B,
        chunk_index=2,
    )
    store = ConfirmedEvidenceStore()
    store.record([first, second])

    resolved = store.resolve_many([
        second.chunk_id,
        first.chunk_id,
    ])

    assert resolved is not None
    assert tuple(
        item.chunk_id
        for item in resolved
    ) == (
        second.chunk_id,
        first.chunk_id,
    )


def test_resolve_many_is_all_or_nothing(
) -> None:
    """一条未知ID应使整组解析返回None。"""

    citation = make_citation()
    store = ConfirmedEvidenceStore()
    store.record([citation])

    resolved = store.resolve_many([
        citation.chunk_id,
        "unknown:000001",
    ])

    assert resolved is None


def test_resolve_many_rejects_duplicate_ids(
) -> None:
    """同一证据不能在一次草案请求中重复计数。"""

    citation = make_citation()
    store = ConfirmedEvidenceStore()
    store.record([citation])

    with pytest.raises(
        ValueError,
        match="chunk_ids不能包含重复项",
    ):
        store.resolve_many([
            citation.chunk_id,
            citation.chunk_id,
        ])


def test_get_requires_string_chunk_id(
) -> None:
    """直接调用Store时也不能使用非字符串证据ID。"""

    store = ConfirmedEvidenceStore()

    with pytest.raises(
        TypeError,
        match="chunk_id必须是字符串",
    ):
        store.get(1)
