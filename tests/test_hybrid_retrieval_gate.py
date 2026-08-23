"""混合检索证据门控的纯离线测试。

本模块不访问Embedding、Chroma、LLM或真实报告文件。

测试范围：
1. 门控策略参数及不可变性；
2. 空候选的稳定拒答；
3. 向量与关键词联合支持的普通放行；
4. 只有高相似度或只有型号命中时不能误放行；
5. Top-K中的错误码精确命中和型号语义命中；
6. 阈值边界、非法参数及非法候选列表；
7. 门控决定必须保留可审计原因和支持Chunk ID。
"""

from dataclasses import FrozenInstanceError

import pytest

from app.schemas.retrieval import (
    DocumentChunk,
    HybridRetrievedChunk,
)
from app.services.hybrid_retrieval_gate import (
    HYBRID_EVIDENCE_GATE_VERSION,
    HybridEvidenceGateDecision,
    HybridEvidenceGatePolicy,
    evaluate_hybrid_evidence_gate,
)


# DocumentChunk要求哈希为64位十六进制文本。
TEST_CONTENT_HASH = "a" * 64


def make_chunk(
    index: int,
) -> DocumentChunk:
    """构造来源稳定且Chunk ID唯一的测试证据。"""

    return DocumentChunk(
        chunk_id=f"gate-document:{index:06d}",
        document_id="gate-document",
        source_file="gate-manual.pdf",
        page_or_section=f"page: {index + 1}",
        chunk_index=index,
        content_hash=TEST_CONTENT_HASH,
        content=f"门控测试正文 {index}",
    )


def make_result(
    *,
    rank: int,
    chunk_index: int,
    vector_similarity: float | None,
    identifiers: tuple[str, ...] = (),
    model_aliases: tuple[str, ...] = (),
    numeric_terms: tuple[str, ...] = (),
    lexical_terms: tuple[str, ...] = (),
) -> HybridRetrievedChunk:
    """构造字段组合自洽的混合检索候选。"""

    has_keyword_support = any(
        (
            identifiers,
            model_aliases,
            numeric_terms,
            lexical_terms,
        )
    )

    return HybridRetrievedChunk(
        chunk=make_chunk(chunk_index),
        rrf_score=0.04 - rank * 0.001,
        rank=rank,
        vector_rank=(
            rank
            if vector_similarity is not None
            else None
        ),
        vector_similarity=vector_similarity,
        keyword_rank=(
            rank
            if has_keyword_support
            else None
        ),
        keyword_score=(
            8.0
            if has_keyword_support
            else None
        ),
        matched_identifiers=identifiers,
        matched_model_aliases=model_aliases,
        matched_numeric_terms=numeric_terms,
        matched_lexical_terms=lexical_terms,
    )


def test_default_policy_records_calibrated_parameters(
) -> None:
    """默认策略必须保存版本和所有可审计阈值。"""

    policy = HybridEvidenceGatePolicy()

    assert policy.version == (
        HYBRID_EVIDENCE_GATE_VERSION
    )
    assert policy.general_min_vector_similarity == 0.50
    assert policy.identifier_min_vector_similarity == 0.45
    assert policy.model_min_vector_similarity == 0.48
    assert policy.min_model_lexical_matches == 2


def test_policy_is_immutable_after_creation(
) -> None:
    """一次评测中不能静默修改门控阈值。"""

    policy = HybridEvidenceGatePolicy()

    with pytest.raises(FrozenInstanceError):
        policy.general_min_vector_similarity = 0.10  # type: ignore[misc]


def test_empty_candidates_are_rejected(
) -> None:
    """没有任何候选时必须拒答且不声明支持Chunk。"""

    decision = evaluate_hybrid_evidence_gate(
        retrieved_chunks=[],
        policy=HybridEvidenceGatePolicy(),
    )

    assert isinstance(
        decision,
        HybridEvidenceGateDecision,
    )
    assert decision.accepted is False
    assert decision.reason == "no_candidates"
    assert decision.evaluated_candidate_count == 0
    assert decision.top_vector_similarity is None
    assert decision.supporting_chunk_ids == ()


def test_general_dual_path_support_is_accepted(
) -> None:
    """达到通用阈值且有内容词支持的Top-1应放行。"""

    top_result = make_result(
        rank=1,
        chunk_index=1,
        vector_similarity=0.55,
        lexical_terms=(
            "网络",
            "恢复",
        ),
    )

    decision = evaluate_hybrid_evidence_gate(
        retrieved_chunks=[top_result],
        policy=HybridEvidenceGatePolicy(),
    )

    assert decision.accepted is True
    assert decision.reason == (
        "general_dual_path_support"
    )
    assert decision.top_vector_similarity == 0.55
    assert decision.supporting_chunk_ids == (
        top_result.chunk.chunk_id,
    )


def test_high_similarity_with_only_model_name_is_rejected(
) -> None:
    """类似q028的纯型号命中不能证明具体序列号存在。"""

    result = make_result(
        rank=1,
        chunk_index=1,
        vector_similarity=0.70,
        model_aliases=("dataman-260",),
    )

    decision = evaluate_hybrid_evidence_gate(
        retrieved_chunks=[result],
        policy=HybridEvidenceGatePolicy(),
    )

    assert decision.accepted is False
    assert decision.reason == (
        "insufficient_combined_support"
    )
    assert decision.supporting_chunk_ids == ()


def test_exact_identifier_in_top_k_can_use_lower_threshold(
) -> None:
    """Top-2中的精确错误码可作为高精度结构化支持。"""

    ordinary_top = make_result(
        rank=1,
        chunk_index=1,
        vector_similarity=0.30,
        lexical_terms=("故障",),
    )
    identifier_result = make_result(
        rank=2,
        chunk_index=2,
        vector_similarity=0.46,
        identifiers=("err-net-4001",),
    )

    decision = evaluate_hybrid_evidence_gate(
        retrieved_chunks=[
            ordinary_top,
            identifier_result,
        ],
        policy=HybridEvidenceGatePolicy(),
    )

    assert decision.accepted is True
    assert decision.reason == (
        "exact_identifier_support"
    )
    assert decision.supporting_chunk_ids == (
        identifier_result.chunk.chunk_id,
    )


def test_model_and_lexical_support_can_use_lower_threshold(
) -> None:
    """型号加具体主题词可以放行低于通用阈值的证据。"""

    model_result = make_result(
        rank=1,
        chunk_index=1,
        vector_similarity=0.49,
        model_aliases=("dataman-260",),
        lexical_terms=(
            "cable",
            "high-voltage",
        ),
    )

    decision = evaluate_hybrid_evidence_gate(
        retrieved_chunks=[model_result],
        policy=HybridEvidenceGatePolicy(),
    )

    assert decision.accepted is True
    assert decision.reason == (
        "model_and_lexical_support"
    )
    assert decision.supporting_chunk_ids == (
        model_result.chunk.chunk_id,
    )


def test_vector_only_candidate_is_rejected(
) -> None:
    """只有向量高相似度而无关键词解释时不能放行。"""

    vector_only = make_result(
        rank=1,
        chunk_index=1,
        vector_similarity=0.90,
    )

    decision = evaluate_hybrid_evidence_gate(
        retrieved_chunks=[vector_only],
        policy=HybridEvidenceGatePolicy(),
    )

    assert decision.accepted is False
    assert decision.reason == (
        "insufficient_combined_support"
    )


@pytest.mark.parametrize(
    (
        "result",
        "expected_reason",
    ),
    [
        (
            make_result(
                rank=1,
                chunk_index=1,
                vector_similarity=0.50,
                lexical_terms=("安全",),
            ),
            "general_dual_path_support",
        ),
        (
            make_result(
                rank=1,
                chunk_index=2,
                vector_similarity=0.45,
                identifiers=("err-test-1001",),
            ),
            "exact_identifier_support",
        ),
        (
            make_result(
                rank=1,
                chunk_index=3,
                vector_similarity=0.48,
                model_aliases=("dataman-260",),
                lexical_terms=("reader", "cable"),
            ),
            "model_and_lexical_support",
        ),
    ],
    ids=[
        "general-equality",
        "identifier-equality",
        "model-equality",
    ],
)
def test_similarity_equal_to_threshold_is_accepted(
    result: HybridRetrievedChunk,
    expected_reason: str,
) -> None:
    """门控阈值使用大于等于而不是严格大于。"""

    decision = evaluate_hybrid_evidence_gate(
        retrieved_chunks=[result],
        policy=HybridEvidenceGatePolicy(),
    )

    assert decision.accepted is True
    assert decision.reason == expected_reason


@pytest.mark.parametrize(
    (
        "field_name",
        "invalid_value",
        "expected_message",
    ),
    [
        (
            "general_min_vector_similarity",
            1.1,
            "general_min_vector_similarity必须在0到1之间",
        ),
        (
            "identifier_min_vector_similarity",
            -0.1,
            "identifier_min_vector_similarity必须在0到1之间",
        ),
        (
            "model_min_vector_similarity",
            float("nan"),
            "model_min_vector_similarity必须是有限数值",
        ),
        (
            "min_model_lexical_matches",
            0,
            "min_model_lexical_matches必须大于或等于1",
        ),
    ],
    ids=[
        "general-above-one",
        "identifier-below-zero",
        "model-nan",
        "zero-model-terms",
    ],
)
def test_invalid_policy_parameters_are_rejected(
    field_name: str,
    invalid_value: object,
    expected_message: str,
) -> None:
    """非法阈值不能形成看似有效的门控策略。"""

    with pytest.raises(
        ValueError,
        match=expected_message,
    ):
        HybridEvidenceGatePolicy(
            **{
                field_name: invalid_value,
            }
        )


def test_gate_rejects_invalid_container_and_items(
) -> None:
    """公共门控函数必须检查列表和元素运行时类型。"""

    policy = HybridEvidenceGatePolicy()

    with pytest.raises(
        TypeError,
        match="retrieved_chunks必须是列表",
    ):
        evaluate_hybrid_evidence_gate(
            retrieved_chunks=(),  # type: ignore[arg-type]
            policy=policy,
        )

    with pytest.raises(
        TypeError,
        match="必须是HybridRetrievedChunk",
    ):
        evaluate_hybrid_evidence_gate(
            retrieved_chunks=[object()],  # type: ignore[list-item]
            policy=policy,
        )


def test_gate_rejects_rank_gaps_and_duplicate_chunks(
) -> None:
    """有排名缺口或重复Chunk的候选不能进入门控。"""

    policy = HybridEvidenceGatePolicy()

    rank_one = make_result(
        rank=1,
        chunk_index=1,
        vector_similarity=0.60,
        lexical_terms=("安全",),
    )
    rank_three = make_result(
        rank=3,
        chunk_index=3,
        vector_similarity=0.60,
        lexical_terms=("检查",),
    )

    with pytest.raises(
        ValueError,
        match="rank必须从1开始连续",
    ):
        evaluate_hybrid_evidence_gate(
            retrieved_chunks=[
                rank_one,
                rank_three,
            ],
            policy=policy,
        )

    duplicate_chunk = HybridRetrievedChunk(
        chunk=rank_one.chunk.model_copy(
            deep=True
        ),
        rrf_score=0.038,
        rank=2,
        vector_rank=2,
        vector_similarity=0.55,
        keyword_rank=2,
        keyword_score=7.0,
        matched_lexical_terms=("安全",),
    )

    with pytest.raises(
        ValueError,
        match="不能包含重复的chunk_id",
    ):
        evaluate_hybrid_evidence_gate(
            retrieved_chunks=[
                rank_one,
                duplicate_chunk,
            ],
            policy=policy,
        )


def test_gate_requires_policy_instance(
) -> None:
    """门控函数不能接受未经校验的字典代替策略。"""

    with pytest.raises(
        TypeError,
        match="policy必须是HybridEvidenceGatePolicy",
    ):
        evaluate_hybrid_evidence_gate(
            retrieved_chunks=[],
            policy={  # type: ignore[arg-type]
                "general_min_vector_similarity": 0.5,
            },
        )
