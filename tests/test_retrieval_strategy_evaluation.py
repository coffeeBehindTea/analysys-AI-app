"""候选检索策略评分与报告构造的离线测试。

本测试模块不访问网络、Embedding API、Chroma或LLM。

它通过人工构造Gold问题和检索候选，验证：
1. 单题证据命中是否计算正确；
2. Top-1和Top-3完整召回是否区分正确；
3. 无答案问题是否不会被误判为完整召回；
4. 多道题的Recall指标是否计算正确；
5. 最终策略报告是否保存正确参数和结果；
6. 非法候选排名、重复Chunk和重复题号是否被拒绝。
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
from app.services.retrieval_strategy_evaluation import (
    build_candidate_strategy_report,
    calculate_candidate_strategy_metrics,
    evaluate_candidate_question,
)


# DocumentChunk要求content_hash是64位小写十六进制字符串。
#
# 本测试不验证SHA-256计算功能，只需要提供一个符合
# DocumentChunk数据契约的稳定测试值。
TEST_CONTENT_HASH = "a" * 64


def make_chunk(
    *,
    chunk_id: str,
    source_file: str,
    page_or_section: str,
) -> DocumentChunk:
    """构造一个只包含测试所需字段的DocumentChunk。"""

    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=f"document-{chunk_id}",
        source_file=source_file,
        page_or_section=page_or_section,
        chunk_index=0,
        content_hash=TEST_CONTENT_HASH,
        content=f"{chunk_id}对应的测试正文",
    )


def make_vector_result(
    *,
    chunk_id: str,
    source_file: str,
    page_or_section: str,
    rank: int,
) -> RetrievedChunk:
    """构造纯向量策略返回的候选结果。"""

    return RetrievedChunk(
        chunk=make_chunk(
            chunk_id=chunk_id,
            source_file=source_file,
            page_or_section=page_or_section,
        ),

        # 测试重点是证据位置与排名，不是相似度计算。
        # 这里为不同排名提供合法、稳定的相似度。
        similarity=0.90 - rank * 0.01,
        rank=rank,
    )


def make_hybrid_result(
    *,
    chunk_id: str,
    source_file: str,
    page_or_section: str,
    rank: int,
) -> HybridRetrievedChunk:
    """构造RRF混合策略返回的候选结果。"""

    return HybridRetrievedChunk(
        chunk=make_chunk(
            chunk_id=chunk_id,
            source_file=source_file,
            page_or_section=page_or_section,
        ),

        # 使用与RRF相同形式的合法正数。
        #
        # 评测器不会重新计算RRF，
        # 它只消费已经排好名的候选结果。
        rrf_score=1.0 / (60 + rank),
        rank=rank,

        # 该测试候选来自向量路径。
        #
        # HybridRetrievedChunk允许候选只进入一条路径，
        # 因为真实RRF候选不一定同时被两条路径召回。
        vector_rank=rank,
        vector_similarity=0.90 - rank * 0.01,
    )


def make_answerable_question(
    *,
    question_id: str,
    expected_evidence: list[ExpectedEvidence],
) -> GoldQuestion:
    """根据预期证据数量构造单跳或多跳问题。"""

    question_type = (
        "multi_hop"
        if len(expected_evidence) > 1
        else "single_hop"
    )

    return GoldQuestion(
        question_id=question_id,
        question=f"{question_id}的测试问题",
        question_type=question_type,
        answerable=True,
        reference_answer="测试参考答案",
        expected_evidence=expected_evidence,
        tags=["test"],
        notes="候选策略评测器的离线测试数据",
    )


def make_unanswerable_question(
    *,
    question_id: str,
) -> GoldQuestion:
    """构造一条知识库中没有答案的问题。"""

    return GoldQuestion(
        question_id=question_id,
        question=f"{question_id}的无答案问题",
        question_type="unanswerable",
        answerable=False,
        reference_answer=None,
        expected_evidence=[],
        tags=["test", "unanswerable"],
        notes="验证无答案问题不会被标记为完整召回",
    )


def make_strategy_evaluations(
) -> list[CandidateQuestionEvaluation]:
    """构造汇总指标和报告测试共用的三道题结果。"""

    first_question = make_answerable_question(
        question_id="q101",
        expected_evidence=[
            ExpectedEvidence(
                source_file="target-a.md",
                page_or_section="section: A",
            ),
        ],
    )

    # q101的正确证据排在第二名：
    #
    # Top-1命中0项；
    # Top-3命中1项；
    # Top-1不完整；
    # Top-3完整。
    first_evaluation = evaluate_candidate_question(
        question=first_question,
        retrieved_chunks=[
            make_hybrid_result(
                chunk_id="q101-wrong",
                source_file="wrong.md",
                page_or_section="section: Wrong",
                rank=1,
            ),
            make_hybrid_result(
                chunk_id="q101-correct",
                source_file="target-a.md",
                page_or_section="section: A",
                rank=2,
            ),
        ],
    )

    second_question = make_answerable_question(
        question_id="q102",
        expected_evidence=[
            ExpectedEvidence(
                source_file="target-b.md",
                page_or_section="section: B",
            ),
            ExpectedEvidence(
                source_file="target-c.md",
                page_or_section="section: C",
            ),
        ],
    )

    # q102需要两项证据。
    #
    # 第一项位于Top-1；
    # 第二项位于Top-3；
    # 因此Top-1只命中1/2，Top-3命中2/2。
    second_evaluation = evaluate_candidate_question(
        question=second_question,
        retrieved_chunks=[
            make_hybrid_result(
                chunk_id="q102-correct-b",
                source_file="target-b.md",
                page_or_section="section: B",
                rank=1,
            ),
            make_hybrid_result(
                chunk_id="q102-wrong",
                source_file="wrong.md",
                page_or_section="section: Wrong",
                rank=2,
            ),
            make_hybrid_result(
                chunk_id="q102-correct-c",
                source_file="target-c.md",
                page_or_section="section: C",
                rank=3,
            ),
        ],
    )

    third_question = make_unanswerable_question(
        question_id="q103",
    )

    third_evaluation = evaluate_candidate_question(
        question=third_question,
        retrieved_chunks=[
            make_hybrid_result(
                chunk_id="q103-unrelated",
                source_file="unrelated.md",
                page_or_section="section: Unrelated",
                rank=1,
            ),
        ],
    )

    return [
        first_evaluation,
        second_evaluation,
        third_evaluation,
    ]


def test_vector_question_is_scored_by_rank(
) -> None:
    """纯向量候选应先按rank排序，再计算Top-1和Top-3。"""

    question = make_answerable_question(
        question_id="q201",
        expected_evidence=[
            ExpectedEvidence(
                source_file="correct.md",
                page_or_section="section: Correct",
            ),
        ],
    )

    correct_result = make_vector_result(
        chunk_id="correct",
        source_file="correct.md",
        page_or_section="section: Correct",
        rank=2,
    )

    wrong_result = make_vector_result(
        chunk_id="wrong",
        source_file="wrong.md",
        page_or_section="section: Wrong",
        rank=1,
    )

    # 故意按照rank=2、rank=1的顺序传入，
    # 验证评测器不会依赖列表的偶然顺序。
    evaluation = evaluate_candidate_question(
        question=question,
        retrieved_chunks=[
            correct_result,
            wrong_result,
        ],
    )

    assert [
        result.rank
        for result in evaluation.retrieved_chunks
    ] == [1, 2]

    assert evaluation.matched_evidence_at_1 == 0
    assert evaluation.matched_evidence_at_3 == 1
    assert evaluation.fully_recalled_at_1 is False
    assert evaluation.fully_recalled_at_3 is True


def test_multi_hop_question_requires_all_evidence(
) -> None:
    """多跳题必须找全两项证据，才能标记为完整召回。"""

    evaluations = make_strategy_evaluations()
    evaluation = evaluations[1]

    assert evaluation.question_id == "q102"

    # Top-1只包含target-b.md，
    # 尚未找全target-c.md。
    assert evaluation.matched_evidence_at_1 == 1
    assert evaluation.fully_recalled_at_1 is False

    # Top-3包含两项预期证据，
    # 所以此时才算完整召回。
    assert evaluation.matched_evidence_at_3 == 2
    assert evaluation.fully_recalled_at_3 is True


def test_unanswerable_question_is_not_fully_recalled(
) -> None:
    """无答案题不能因为0项预期证据而出现0等于0的误判。"""

    evaluations = make_strategy_evaluations()
    evaluation = evaluations[2]

    assert evaluation.answerable is False
    assert evaluation.expected_evidence == []
    assert evaluation.matched_evidence_at_1 == 0
    assert evaluation.matched_evidence_at_3 == 0
    assert evaluation.fully_recalled_at_1 is False
    assert evaluation.fully_recalled_at_3 is False


def test_candidate_metrics_are_calculated_from_questions(
) -> None:
    """汇总指标应根据全部逐题结果重新计算。"""

    evaluations = make_strategy_evaluations()

    metrics = calculate_candidate_strategy_metrics(
        evaluations
    )

    assert metrics.answerable_question_count == 2
    assert metrics.unanswerable_question_count == 1

    # q101有1项预期证据；
    # q102有2项预期证据；
    # 总计3项。
    assert metrics.expected_evidence_count == 3

    # Top-1只有q102命中一项证据。
    assert metrics.matched_evidence_at_1 == 1

    # Top-3找全q101的一项和q102的两项。
    assert metrics.matched_evidence_at_3 == 3

    # pytest.approx()允许极小的浮点表示误差。
    assert metrics.recall_at_1 == pytest.approx(
        1 / 3
    )
    assert metrics.recall_at_3 == pytest.approx(
        1.0
    )

    # 两道可回答题在Top-1都没有找全。
    assert metrics.fully_recalled_at_1_count == 0

    # 两道可回答题在Top-3都已经找全。
    assert metrics.fully_recalled_at_3_count == 2


def test_strategy_report_contains_parameters_and_metrics(
) -> None:
    """报告构造器应组合参数、逐题结果和汇总指标。"""

    evaluations = make_strategy_evaluations()

    parameters = CandidateStrategyParameters(
        strategy="hybrid_rrf",
        top_k=3,
        candidate_k=20,
        rank_constant=60,
    )

    report = build_candidate_strategy_report(
        embedding_model="embedding-3",
        collection_name="robot_knowledge_v4",
        parameters=parameters,
        evaluations=evaluations,
    )

    assert "generated_at" not in report.model_dump()
    assert report.embedding_model == "embedding-3"
    assert report.collection_name == "robot_knowledge_v4"

    assert report.parameters.strategy == "hybrid_rrf"
    assert report.parameters.top_k == 3
    assert report.parameters.candidate_k == 20
    assert report.parameters.rank_constant == 60
    assert report.parameters.rewrite_version is None

    assert report.metrics.expected_evidence_count == 3
    assert report.metrics.matched_evidence_at_3 == 3
    assert len(report.results) == 3

    # 报告保存的是深复制快照，
    # 不应直接复用调用方传入的对象。
    assert report.parameters is not parameters
    assert report.results[0] is not evaluations[0]


def test_duplicate_chunk_ids_are_rejected(
) -> None:
    """同一道题的最终候选中不能重复保存同一个Chunk。"""

    question = make_answerable_question(
        question_id="q301",
        expected_evidence=[
            ExpectedEvidence(
                source_file="target.md",
                page_or_section="section: Target",
            ),
        ],
    )

    duplicate_results = [
        make_hybrid_result(
            chunk_id="duplicate",
            source_file="target.md",
            page_or_section="section: Target",
            rank=1,
        ),
        make_hybrid_result(
            chunk_id="duplicate",
            source_file="target.md",
            page_or_section="section: Target",
            rank=2,
        ),
    ]

    with pytest.raises(
        ValueError,
        match="不能包含重复的chunk_id",
    ):
        evaluate_candidate_question(
            question=question,
            retrieved_chunks=duplicate_results,
        )


def test_non_continuous_ranks_are_rejected(
) -> None:
    """最终候选排名必须是从1开始的连续整数。"""

    question = make_answerable_question(
        question_id="q302",
        expected_evidence=[
            ExpectedEvidence(
                source_file="target.md",
                page_or_section="section: Target",
            ),
        ],
    )

    results_with_rank_gap = [
        make_hybrid_result(
            chunk_id="rank-one",
            source_file="target.md",
            page_or_section="section: Target",
            rank=1,
        ),
        make_hybrid_result(
            chunk_id="rank-three",
            source_file="other.md",
            page_or_section="section: Other",
            rank=3,
        ),
    ]

    with pytest.raises(
        ValueError,
        match="必须从1开始连续且不能重复",
    ):
        evaluate_candidate_question(
            question=question,
            retrieved_chunks=results_with_rank_gap,
        )


def test_empty_evaluations_are_rejected(
) -> None:
    """空评测列表无法形成有意义的策略指标。"""

    with pytest.raises(
        ValueError,
        match="evaluations不能为空",
    ):
        calculate_candidate_strategy_metrics([])


def test_duplicate_question_ids_are_rejected(
) -> None:
    """同一道Gold题不能在一份策略指标中重复计数。"""

    evaluations = make_strategy_evaluations()

    duplicate_evaluation = (
        evaluations[0].model_copy(
            deep=True
        )
    )

    with pytest.raises(
        ValueError,
        match="不能包含重复的question_id",
    ):
        calculate_candidate_strategy_metrics(
            [
                evaluations[0],
                duplicate_evaluation,
            ]
        )
