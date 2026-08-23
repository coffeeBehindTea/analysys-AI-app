"""混合证据门控逐题评价与指标汇总的离线测试。

本模块不访问Embedding、Chroma、LLM或真实报告文件。

测试范围：
1. 可回答题放行且证据完整；
2. 可回答题放行但证据不完整；
3. 可回答题被门控拒绝；
4. 无答案题被错误放行或正确拒绝；
5. 五种结果的计数和比例汇总；
6. 候选报告、门控策略和评测结果的快照构造；
7. 非混合报告、重复题号及非法输入的防御。
"""

import pytest

from app.schemas.evaluation import (
    ExpectedEvidence,
    GoldQuestion,
)
from app.schemas.retrieval import (
    DocumentChunk,
    HybridRetrievedChunk,
    RetrievedChunk,
)
from app.schemas.retrieval_strategy import (
    CandidateQuestionEvaluation,
    CandidateStrategyParameters,
)
from app.services.hybrid_gate_evaluation import (
    HybridGateEvaluationReport,
    HybridGateMetrics,
    HybridGateQuestionEvaluation,
    build_hybrid_gate_evaluation_report,
    calculate_hybrid_gate_metrics,
    evaluate_hybrid_gate_question,
)
from app.services.hybrid_retrieval_gate import (
    HybridEvidenceGatePolicy,
)
from app.services.retrieval_strategy_evaluation import (
    build_candidate_strategy_report,
    evaluate_candidate_question,
)


TEST_CONTENT_HASH = "b" * 64


def make_chunk(
    *,
    index: int,
    source_file: str,
    page_or_section: str,
) -> DocumentChunk:
    """构造具有稳定Gold位置的测试Chunk。"""

    return DocumentChunk(
        chunk_id=f"gate-eval-document:{index:06d}",
        document_id="gate-eval-document",
        source_file=source_file,
        page_or_section=page_or_section,
        chunk_index=index,
        content_hash=TEST_CONTENT_HASH,
        content=f"门控评测正文 {index}",
    )


def make_hybrid_result(
    *,
    rank: int,
    chunk: DocumentChunk,
    vector_similarity: float,
    lexical_terms: tuple[str, ...] = (),
) -> HybridRetrievedChunk:
    """构造向量与可选关键词路径共同产生的候选。"""

    has_keyword_support = bool(
        lexical_terms
    )

    return HybridRetrievedChunk(
        chunk=chunk,
        rrf_score=0.04 - rank * 0.001,
        rank=rank,
        vector_rank=rank,
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
        matched_lexical_terms=lexical_terms,
    )


def make_gold_question(
    *,
    question_id: str,
    question_type: str,
    expected_evidence: list[
        ExpectedEvidence
    ],
) -> GoldQuestion:
    """按可回答性构造合法Gold问题。"""

    answerable = (
        question_type != "unanswerable"
    )

    return GoldQuestion(
        question_id=question_id,
        question=f"测试问题 {question_id}",
        question_type=question_type,
        answerable=answerable,
        reference_answer=(
            f"测试答案 {question_id}"
            if answerable
            else None
        ),
        expected_evidence=expected_evidence,
        tags=["gate-evaluation"],
        notes="门控评测测试数据",
    )


def make_candidate_evaluation(
    *,
    question: GoldQuestion,
    retrieved_chunks: list[
        HybridRetrievedChunk
    ],
) -> CandidateQuestionEvaluation:
    """调用真实候选评分函数形成逐题输入。"""

    return evaluate_candidate_question(
        question=question,
        retrieved_chunks=retrieved_chunks,
    )


def make_five_gate_evaluations(
) -> list[HybridGateQuestionEvaluation]:
    """构造覆盖五种门控结果的评测集合。"""

    evidence_a = make_chunk(
        index=1,
        source_file="manual-a.pdf",
        page_or_section="page: 1",
    )
    evidence_b = make_chunk(
        index=2,
        source_file="manual-b.pdf",
        page_or_section="page: 2",
    )
    evidence_c = make_chunk(
        index=3,
        source_file="procedure-c.md",
        page_or_section="section: C",
    )
    distractor = make_chunk(
        index=4,
        source_file="distractor.pdf",
        page_or_section="page: 4",
    )

    questions_and_results = [
        # q001：放行且单跳证据完整。
        (
            make_gold_question(
                question_id="q001",
                question_type="single_hop",
                expected_evidence=[
                    ExpectedEvidence(
                        source_file="manual-a.pdf",
                        page_or_section="page: 1",
                    )
                ],
            ),
            [
                make_hybrid_result(
                    rank=1,
                    chunk=evidence_a,
                    vector_similarity=0.60,
                    lexical_terms=("安全",),
                )
            ],
        ),
        # q002：多跳题只找回一半证据，但门控仍放行。
        (
            make_gold_question(
                question_id="q002",
                question_type="multi_hop",
                expected_evidence=[
                    ExpectedEvidence(
                        source_file="manual-b.pdf",
                        page_or_section="page: 2",
                    ),
                    ExpectedEvidence(
                        source_file="procedure-c.md",
                        page_or_section="section: C",
                    ),
                ],
            ),
            [
                make_hybrid_result(
                    rank=1,
                    chunk=evidence_b,
                    vector_similarity=0.60,
                    lexical_terms=("检查",),
                )
            ],
        ),
        # q003：可回答题只有弱候选，被门控拒绝。
        (
            make_gold_question(
                question_id="q003",
                question_type="single_hop",
                expected_evidence=[
                    ExpectedEvidence(
                        source_file="manual-a.pdf",
                        page_or_section="page: 1",
                    )
                ],
            ),
            [
                make_hybrid_result(
                    rank=1,
                    chunk=distractor,
                    vector_similarity=0.30,
                    lexical_terms=("背景",),
                )
            ],
        ),
        # q004：无答案题却有高分表面相关候选，错误放行。
        (
            make_gold_question(
                question_id="q004",
                question_type="unanswerable",
                expected_evidence=[],
            ),
            [
                make_hybrid_result(
                    rank=1,
                    chunk=distractor,
                    vector_similarity=0.70,
                    lexical_terms=("设备",),
                )
            ],
        ),
        # q005：无答案题只有弱候选，正确拒答。
        (
            make_gold_question(
                question_id="q005",
                question_type="unanswerable",
                expected_evidence=[],
            ),
            [
                make_hybrid_result(
                    rank=1,
                    chunk=distractor,
                    vector_similarity=0.20,
                    lexical_terms=("设备",),
                )
            ],
        ),
    ]

    policy = HybridEvidenceGatePolicy()

    return [
        evaluate_hybrid_gate_question(
            question_result=(
                make_candidate_evaluation(
                    question=question,
                    retrieved_chunks=results,
                )
            ),
            policy=policy,
        )
        for question, results
        in questions_and_results
    ]


@pytest.mark.parametrize(
    (
        "evaluation_index",
        "expected_outcome",
        "expected_accepted",
    ),
    [
        (
            0,
            "accepted_with_full_evidence",
            True,
        ),
        (
            1,
            "accepted_with_incomplete_evidence",
            True,
        ),
        (
            2,
            "rejected_answerable",
            False,
        ),
        (
            3,
            "false_accept_unanswerable",
            True,
        ),
        (
            4,
            "correct_abstain_unanswerable",
            False,
        ),
    ],
    ids=[
        "answerable-full",
        "answerable-incomplete",
        "answerable-rejected",
        "unanswerable-false-accept",
        "unanswerable-correct-abstain",
    ],
)
def test_gate_question_outcomes_are_classified(
    evaluation_index: int,
    expected_outcome: str,
    expected_accepted: bool,
) -> None:
    """逐题评价必须区分五种业务结果。"""

    evaluation = (
        make_five_gate_evaluations()[
            evaluation_index
        ]
    )

    assert evaluation.outcome == expected_outcome
    assert (
        evaluation.decision.accepted
        is expected_accepted
    )


def test_gate_metrics_count_all_five_outcomes(
) -> None:
    """汇总指标必须同时体现覆盖率和拒答安全性。"""

    metrics = calculate_hybrid_gate_metrics(
        make_five_gate_evaluations()
    )

    assert isinstance(
        metrics,
        HybridGateMetrics,
    )
    assert metrics.answerable_question_count == 3
    assert metrics.unanswerable_question_count == 2
    assert metrics.accepted_answerable_count == 2
    assert metrics.rejected_answerable_count == 1
    assert (
        metrics.accepted_with_full_evidence_count
        == 1
    )
    assert (
        metrics.accepted_with_incomplete_evidence_count
        == 1
    )
    assert (
        metrics.false_accepted_unanswerable_count
        == 1
    )
    assert (
        metrics.correctly_abstained_unanswerable_count
        == 1
    )
    assert metrics.answerable_acceptance_rate == (
        pytest.approx(2 / 3)
    )
    assert metrics.full_evidence_eligible_rate == (
        pytest.approx(1 / 3)
    )
    assert (
        metrics.unanswerable_false_acceptance_rate
        == pytest.approx(0.5)
    )
    assert metrics.unanswerable_abstention_rate == (
        pytest.approx(0.5)
    )


def test_report_preserves_candidate_and_policy_snapshots(
) -> None:
    """正式报告应记录候选策略、门控版本和逐题结果。"""

    evaluations = make_five_gate_evaluations()

    # 用逐题评价对应的候选结果重建候选报告。
    candidate_evaluations = [
        CandidateQuestionEvaluation(
            question_id=evaluation.question_id,
            question=evaluation.question,
            question_type=(
                evaluation.question_type
            ),
            answerable=evaluation.answerable,
            expected_evidence=[
                item.model_copy(deep=True)
                for item in (
                    evaluation.expected_evidence
                )
            ],
            retrieved_chunks=[
                item.model_copy(deep=True)
                for item in (
                    evaluation.retrieved_chunks
                )
            ],
            matched_evidence_at_1=(
                evaluation.matched_evidence_at_1
            ),
            matched_evidence_at_3=(
                evaluation.matched_evidence_at_3
            ),
            fully_recalled_at_1=(
                evaluation.fully_recalled_at_1
            ),
            fully_recalled_at_3=(
                evaluation.fully_recalled_at_3
            ),
        )
        for evaluation in evaluations
    ]

    candidate_report = (
        build_candidate_strategy_report(
            embedding_model="embedding-3",
            collection_name=(
                "robot_knowledge_gate_test"
            ),
            parameters=(
                CandidateStrategyParameters(
                    strategy=(
                        "hybrid_rrf_rewrite"
                    ),
                    top_k=3,
                    candidate_k=20,
                    rank_constant=60,
                    rewrite_version=(
                        "deterministic-v1"
                    ),
                )
            ),
            evaluations=candidate_evaluations,
        )
    )

    policy = HybridEvidenceGatePolicy()
    report = build_hybrid_gate_evaluation_report(
        candidate_report=candidate_report,
        policy=policy,
    )

    assert isinstance(
        report,
        HybridGateEvaluationReport,
    )
    assert report.embedding_model == "embedding-3"
    assert report.collection_name == (
        "robot_knowledge_gate_test"
    )
    assert report.candidate_parameters.strategy == (
        "hybrid_rrf_rewrite"
    )
    assert report.gate_policy.version == policy.version
    assert len(report.results) == 5
    assert report.metrics.answerable_question_count == 3
    assert not hasattr(report, "generated_at")


def test_metrics_reject_empty_duplicate_and_invalid_items(
) -> None:
    """空列表、重复题号和非法对象不能生成汇总指标。"""

    with pytest.raises(
        ValueError,
        match="evaluations不能为空",
    ):
        calculate_hybrid_gate_metrics([])

    evaluations = make_five_gate_evaluations()

    with pytest.raises(
        ValueError,
        match="不能包含重复的question_id",
    ):
        calculate_hybrid_gate_metrics(
            [
                evaluations[0],
                evaluations[0],
            ]
        )

    with pytest.raises(
        TypeError,
        match="必须是HybridGateQuestionEvaluation",
    ):
        calculate_hybrid_gate_metrics(
            [object()]  # type: ignore[list-item]
        )


def test_question_evaluation_rejects_wrong_types(
) -> None:
    """逐题评价不能接受普通字典冒充契约对象。"""

    with pytest.raises(
        TypeError,
        match="question_result必须是CandidateQuestionEvaluation",
    ):
        evaluate_hybrid_gate_question(
            question_result={},  # type: ignore[arg-type]
            policy=HybridEvidenceGatePolicy(),
        )

    question = make_gold_question(
        question_id="q001",
        question_type="single_hop",
        expected_evidence=[
            ExpectedEvidence(
                source_file="manual.pdf",
                page_or_section="page: 1",
            )
        ],
    )
    chunk = make_chunk(
        index=1,
        source_file="manual.pdf",
        page_or_section="page: 1",
    )
    candidate = make_candidate_evaluation(
        question=question,
        retrieved_chunks=[
            make_hybrid_result(
                rank=1,
                chunk=chunk,
                vector_similarity=0.60,
                lexical_terms=("安全",),
            )
        ],
    )

    with pytest.raises(
        TypeError,
        match="policy必须是HybridEvidenceGatePolicy",
    ):
        evaluate_hybrid_gate_question(
            question_result=candidate,
            policy={},  # type: ignore[arg-type]
        )


def test_report_rejects_vector_candidate_report(
) -> None:
    """纯向量报告没有混合审计字段，不能送入混合门控。"""

    question = make_gold_question(
        question_id="q001",
        question_type="single_hop",
        expected_evidence=[
            ExpectedEvidence(
                source_file="manual.pdf",
                page_or_section="page: 1",
            )
        ],
    )
    chunk = make_chunk(
        index=1,
        source_file="manual.pdf",
        page_or_section="page: 1",
    )
    vector_evaluation = (
        evaluate_candidate_question(
            question=question,
            retrieved_chunks=[
                RetrievedChunk(
                    chunk=chunk,
                    similarity=0.80,
                    rank=1,
                )
            ],
        )
    )
    vector_report = (
        build_candidate_strategy_report(
            embedding_model="embedding-3",
            collection_name="vector-only",
            parameters=(
                CandidateStrategyParameters(
                    strategy="vector_baseline",
                    top_k=3,
                )
            ),
            evaluations=[vector_evaluation],
        )
    )

    with pytest.raises(
        ValueError,
        match="只接受混合候选策略报告",
    ):
        build_hybrid_gate_evaluation_report(
            candidate_report=vector_report,
            policy=HybridEvidenceGatePolicy(),
        )


def test_report_contract_has_no_generated_at_field(
) -> None:
    """门控报告的数据类不得重新引入生成时间字段。"""

    assert (
        "generated_at"
        not in HybridGateEvaluationReport.__dataclass_fields__
    )
