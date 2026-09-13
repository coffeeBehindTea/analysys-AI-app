"""头部候选保留重排器的纯离线测试。

这些测试只使用手工构造的HybridRetrievedChunk，
不调用Embedding、Chroma、关键词索引、LLM或网络。

被测试模块预期位于：
app.services.head_preserving_reranking

测试目标：
1. RRF头部候选不会因重排而丢失；
2. 关键词头部可以从原始Top-K之外被救回；
3. 向量头部和关键词头部可以提供互补证据；
4. 必选候选多于Top-K时仍按原RRF顺序确定取舍；
5. 返回排名连续，且输入对象不会被修改；
6. 非法策略参数和非法Top-K会在重排前被拒绝。
"""

from hashlib import sha256

import pytest

from app.schemas.retrieval import (
    DocumentChunk,
    HybridRetrievedChunk,
)
from app.services.head_preserving_reranking import (
    HEAD_PRESERVING_RERANK_VERSION,
    HeadPreservationPolicy,
    rerank_hybrid_candidates,
)


def make_chunk(
    chunk_id: str,
) -> DocumentChunk:
    """创建一份稳定、合法且可区分的测试Chunk。"""

    content = f"{chunk_id}测试正文"

    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=sha256(
            chunk_id.encode("utf-8")
        ).hexdigest(),
        source_file=f"{chunk_id}.txt",
        page_or_section=(
            f"section: {chunk_id}"
        ),
        chunk_index=0,
        content_hash=sha256(
            content.encode("utf-8")
        ).hexdigest(),
        content=content,
    )


def make_candidate(
    chunk_id: str,
    *,
    rank: int,
    vector_rank: int | None = None,
    keyword_rank: int | None = None,
) -> HybridRetrievedChunk:
    """创建包含指定来源排名的RRF候选。"""

    # HybridRetrievedChunk要求：
    # vector_rank和vector_similarity必须同时存在；
    # keyword_rank和keyword_score必须同时存在。
    return HybridRetrievedChunk(
        chunk=make_chunk(chunk_id),

        # 测试只比较候选顺序，
        # 因此使用随rank递减的稳定正数即可。
        rrf_score=1.0 / (60 + rank),
        rank=rank,
        vector_rank=vector_rank,
        vector_similarity=(
            0.5
            if vector_rank is not None
            else None
        ),
        keyword_rank=keyword_rank,
        keyword_score=(
            1.0
            if keyword_rank is not None
            else None
        ),
        matched_lexical_terms=(
            ("test-term",)
            if keyword_rank is not None
            else ()
        ),
    )


def test_default_policy_and_version_are_stable(
) -> None:
    """默认策略和版本号必须可以写入评测审计信息。"""

    policy = HeadPreservationPolicy()

    assert HEAD_PRESERVING_RERANK_VERSION == (
        "head-preserving-v1"
    )
    assert policy.fused_head_k == 1
    assert policy.vector_head_k == 1
    assert policy.keyword_head_k == 2


def test_keyword_head_outside_top_k_is_rescued(
) -> None:
    """关键词第一名不能因只命中一条路径而被RRF完全丢弃。"""

    candidates = [
        make_candidate(
            "fused-head",
            rank=1,
            vector_rank=2,
            keyword_rank=3,
        ),
        make_candidate(
            "rrf-second",
            rank=2,
            vector_rank=3,
            keyword_rank=4,
        ),
        make_candidate(
            "replaceable-tail",
            rank=3,
            vector_rank=4,
            keyword_rank=5,
        ),
        make_candidate(
            "keyword-head",
            rank=4,
            keyword_rank=1,
        ),
    ]

    results = rerank_hybrid_candidates(
        candidates=candidates,
        top_k=3,
        policy=HeadPreservationPolicy(),
    )

    # 原RRF第一名和第二名继续保留；
    # 原第三名被关键词第一名替换。
    assert [
        item.chunk.chunk_id
        for item in results
    ] == [
        "fused-head",
        "rrf-second",
        "keyword-head",
    ]

    # 重排后的公开排名必须重新从1开始连续编号。
    assert [item.rank for item in results] == [
        1,
        2,
        3,
    ]

    # 输入对象仍保留原RRF排名，证明函数没有原地修改输入。
    assert candidates[3].rank == 4


def test_vector_and_keyword_heads_preserve_complementary_evidence(
) -> None:
    """向量头部与关键词头部应能共同进入最终候选。"""

    candidates = [
        # 模拟q027中的手册清洁证据：
        # 已经是RRF第一名和关键词第一名。
        make_candidate(
            "maintenance-evidence",
            rank=1,
            vector_rank=5,
            keyword_rank=1,
        ),
        make_candidate(
            "duplicate-topic-one",
            rank=2,
            vector_rank=3,
            keyword_rank=11,
        ),
        make_candidate(
            "duplicate-topic-two",
            rank=3,
            vector_rank=6,
            keyword_rank=10,
        ),
        # 模拟另一条路径独立找到的验收测试证据。
        make_candidate(
            "acceptance-test-evidence",
            rank=4,
            vector_rank=1,
        ),
        make_candidate(
            "keyword-second",
            rank=5,
            keyword_rank=2,
        ),
    ]

    results = rerank_hybrid_candidates(
        candidates=candidates,
        top_k=3,
        policy=HeadPreservationPolicy(),
    )

    result_ids = {
        item.chunk.chunk_id
        for item in results
    }

    assert result_ids == {
        "maintenance-evidence",
        "acceptance-test-evidence",
        "keyword-second",
    }


def test_existing_heads_keep_original_rrf_order(
) -> None:
    """所有头部已经位于Top-K时不得进行无意义换位。"""

    candidates = [
        make_candidate(
            "shared-head",
            rank=1,
            vector_rank=1,
            keyword_rank=1,
        ),
        make_candidate(
            "keyword-second",
            rank=2,
            vector_rank=4,
            keyword_rank=2,
        ),
        make_candidate(
            "ordinary-third",
            rank=3,
            vector_rank=2,
            keyword_rank=3,
        ),
    ]

    results = rerank_hybrid_candidates(
        candidates=candidates,
        top_k=3,
        policy=HeadPreservationPolicy(),
    )

    assert [
        item.chunk.chunk_id
        for item in results
    ] == [
        "shared-head",
        "keyword-second",
        "ordinary-third",
    ]


def test_too_many_heads_use_original_fused_order(
) -> None:
    """保护候选超过Top-K时应按原RRF顺序选择，不能随机截取。"""

    candidates = [
        make_candidate(
            "fused-head",
            rank=1,
            vector_rank=3,
            keyword_rank=4,
        ),
        make_candidate(
            "keyword-head",
            rank=2,
            keyword_rank=1,
        ),
        make_candidate(
            "vector-head",
            rank=3,
            vector_rank=1,
        ),
        make_candidate(
            "keyword-second",
            rank=4,
            keyword_rank=2,
        ),
    ]

    results = rerank_hybrid_candidates(
        candidates=candidates,
        top_k=2,
        policy=HeadPreservationPolicy(),
    )

    # 四个候选都具有某种头部资格，
    # 但Top-K只有2，因此保留原RRF顺序最靠前的两个。
    assert [
        item.chunk.chunk_id
        for item in results
    ] == [
        "fused-head",
        "keyword-head",
    ]


def test_returned_chunks_are_deep_copies(
) -> None:
    """修改返回Chunk不能污染调用方持有的原始候选。"""

    candidates = [
        make_candidate(
            "isolated",
            rank=1,
            vector_rank=1,
            keyword_rank=1,
        )
    ]

    results = rerank_hybrid_candidates(
        candidates=candidates,
        top_k=1,
        policy=HeadPreservationPolicy(),
    )

    results[0].chunk.content = "修改后的正文"

    assert candidates[0].chunk.content == (
        "isolated测试正文"
    )


@pytest.mark.parametrize(
    (
        "field_name",
        "invalid_value",
        "expected_exception",
    ),
    [
        ("fused_head_k", True, TypeError),
        ("vector_head_k", -1, ValueError),
        ("keyword_head_k", False, TypeError),
    ],
)
def test_policy_rejects_invalid_head_depths(
    field_name: str,
    invalid_value: object,
    expected_exception: type[Exception],
) -> None:
    """各路径头部深度必须是真正的非负整数。"""

    arguments: dict[str, object] = {
        "fused_head_k": 1,
        "vector_head_k": 1,
        "keyword_head_k": 2,
    }
    arguments[field_name] = invalid_value

    with pytest.raises(expected_exception):
        HeadPreservationPolicy(
            **arguments,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    (
        "invalid_top_k",
        "expected_exception",
    ),
    [
        (True, TypeError),
        (0, ValueError),
    ],
)
def test_reranker_rejects_invalid_top_k(
    invalid_top_k: object,
    expected_exception: type[Exception],
) -> None:
    """非法Top-K必须在读取候选前被拒绝。"""

    with pytest.raises(expected_exception):
        rerank_hybrid_candidates(
            candidates=[],
            top_k=invalid_top_k,  # type: ignore[arg-type]
            policy=HeadPreservationPolicy(),
        )
