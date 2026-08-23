"""RRF融合结果Pydantic数据契约的离线测试。

本文件只验证HybridRetrievedChunk本身：
1. 向量独占、关键词独占和双路命中是否合法；
2. 排名与原始分数是否必须成对出现；
3. 关键词来源与命中解释是否保持一致；
4. RRF分数、排名、有限浮点数和额外字段约束。

本文件不执行RRF算法、向量检索、关键词检索或网络请求。
"""

from hashlib import sha256

from pydantic import ValidationError
import pytest

from app.schemas.retrieval import (
    DocumentChunk,
    HybridRetrievedChunk,
)


def make_chunk() -> DocumentChunk:
    """创建所有契约测试共用的合法Chunk。"""

    content = "ERR-NET-4001网络恢复测试正文"

    return DocumentChunk(
        chunk_id="document-a:000001",
        document_id="a" * 64,
        source_file="network-manual.txt",
        page_or_section=(
            "section: ERR-NET-4001"
        ),
        chunk_index=1,
        content_hash=sha256(
            content.encode("utf-8")
        ).hexdigest(),
        content=content,
    )


def test_vector_only_hybrid_result_is_valid(
) -> None:
    """只进入向量候选的Chunk应保留排名和余弦相似度。"""

    result = HybridRetrievedChunk(
        chunk=make_chunk(),
        rrf_score=(1.0 / 61.0),
        rank=2,
        vector_rank=1,
        vector_similarity=0.72,
    )

    assert result.vector_rank == 1
    assert result.vector_similarity == (
        pytest.approx(0.72)
    )
    assert result.keyword_rank is None
    assert result.keyword_score is None

    # 没有关键词来源时，四类命中解释必须为空。
    assert result.matched_identifiers == ()
    assert result.matched_model_aliases == ()
    assert result.matched_numeric_terms == ()
    assert result.matched_lexical_terms == ()


def test_keyword_only_hybrid_result_is_valid(
) -> None:
    """只进入关键词候选的Chunk应保留分数和命中解释。"""

    result = HybridRetrievedChunk(
        chunk=make_chunk(),
        rrf_score=(1.0 / 61.0),
        rank=2,
        keyword_rank=1,
        keyword_score=12.0,
        matched_identifiers=(
            "err-net-4001",
        ),
    )

    assert result.vector_rank is None
    assert result.vector_similarity is None
    assert result.keyword_rank == 1
    assert result.keyword_score == (
        pytest.approx(12.0)
    )
    assert result.matched_identifiers == (
        "err-net-4001",
    )


def test_dual_source_hybrid_result_is_valid(
) -> None:
    """同一Chunk被两条路径召回时应保留两侧完整审计字段。"""

    result = HybridRetrievedChunk(
        chunk=make_chunk(),
        rrf_score=(
            (1.0 / 62.0)
            + (1.0 / 61.0)
        ),
        rank=1,
        vector_rank=2,
        vector_similarity=0.68,
        keyword_rank=1,
        keyword_score=12.5,
        matched_identifiers=(
            "err-net-4001",
        ),
        matched_lexical_terms=(
            "网络",
        ),
    )

    assert result.rank == 1
    assert result.vector_rank == 2
    assert result.keyword_rank == 1
    assert result.rrf_score == (
        pytest.approx(
            (1.0 / 62.0)
            + (1.0 / 61.0)
        )
    )


@pytest.mark.parametrize(
    "incomplete_vector_fields",
    [
        {
            "vector_rank": 1,
        },
        {
            "vector_similarity": 0.72,
        },
    ],
    ids=[
        "rank-without-similarity",
        "similarity-without-rank",
    ],
)
def test_vector_fields_must_appear_together(
    incomplete_vector_fields: dict[
        str,
        int | float,
    ],
) -> None:
    """向量排名和相似度缺少任何一项都应校验失败。"""

    with pytest.raises(
        ValidationError,
        match=(
            "vector_rank与vector_similarity"
            "必须同时存在或同时为空"
        ),
    ):
        HybridRetrievedChunk(
            chunk=make_chunk(),
            rrf_score=(1.0 / 61.0),
            rank=1,
            **incomplete_vector_fields,
        )


@pytest.mark.parametrize(
    "incomplete_keyword_fields",
    [
        {
            "keyword_rank": 1,
        },
        {
            "keyword_score": 12.0,
        },
    ],
    ids=[
        "rank-without-score",
        "score-without-rank",
    ],
)
def test_keyword_fields_must_appear_together(
    incomplete_keyword_fields: dict[
        str,
        int | float,
    ],
) -> None:
    """关键词排名和分数缺少任何一项都应校验失败。"""

    with pytest.raises(
        ValidationError,
        match=(
            "keyword_rank与keyword_score"
            "必须同时存在或同时为空"
        ),
    ):
        HybridRetrievedChunk(
            chunk=make_chunk(),
            rrf_score=(1.0 / 61.0),
            rank=1,
            matched_identifiers=(
                "err-net-4001",
            ),
            **incomplete_keyword_fields,
        )


def test_result_without_any_retrieval_source_is_rejected(
) -> None:
    """两条路径都没有召回时不能伪造融合结果。"""

    with pytest.raises(
        ValidationError,
        match="至少来自一条检索路径",
    ):
        HybridRetrievedChunk(
            chunk=make_chunk(),
            rrf_score=(1.0 / 61.0),
            rank=1,
        )


def test_keyword_source_without_explanation_is_rejected(
) -> None:
    """关键词候选必须说明具体命中了什么。"""

    with pytest.raises(
        ValidationError,
        match="关键词融合结果必须包含命中解释",
    ):
        HybridRetrievedChunk(
            chunk=make_chunk(),
            rrf_score=(1.0 / 61.0),
            rank=1,
            keyword_rank=1,
            keyword_score=12.0,
        )


def test_vector_only_result_cannot_claim_keyword_matches(
) -> None:
    """没有关键词来源时不能携带伪造的关键词解释。"""

    with pytest.raises(
        ValidationError,
        match="没有关键词来源时不能包含关键词命中解释",
    ):
        HybridRetrievedChunk(
            chunk=make_chunk(),
            rrf_score=(1.0 / 61.0),
            rank=1,
            vector_rank=1,
            vector_similarity=0.72,
            matched_identifiers=(
                "err-net-4001",
            ),
        )


@pytest.mark.parametrize(
    "invalid_fields",
    [
        {
            "rrf_score": 0.0,
        },
        {
            "rank": 0,
        },
        {
            "rrf_score": float("nan"),
        },
        {
            "vector_similarity": 1.01,
        },
    ],
    ids=[
        "zero-rrf-score",
        "zero-final-rank",
        "nan-rrf-score",
        "similarity-above-one",
    ],
)
def test_numeric_constraints_are_enforced(
    invalid_fields: dict[
        str,
        int | float,
    ],
) -> None:
    """融合分数、最终排名和余弦相似度必须在合法范围内。"""

    valid_fields: dict[
        str,
        int | float,
    ] = {
        "rrf_score": (1.0 / 61.0),
        "rank": 1,
        "vector_rank": 1,
        "vector_similarity": 0.72,
    }

    # update()用测试参数覆盖一个合法字段，
    # 每个参数用例只制造一种无效输入。
    valid_fields.update(
        invalid_fields
    )

    with pytest.raises(
        ValidationError,
    ):
        HybridRetrievedChunk(
            chunk=make_chunk(),
            **valid_fields,
        )


def test_extra_field_is_rejected(
) -> None:
    """未声明字段不能被Pydantic静默忽略。"""

    with pytest.raises(
        ValidationError,
        match="rrf_scores",
    ):
        HybridRetrievedChunk(
            chunk=make_chunk(),
            rrf_score=(1.0 / 61.0),
            rank=1,
            vector_rank=1,
            vector_similarity=0.72,
            rrf_scores=999.0,  # type: ignore[call-arg]
        )
