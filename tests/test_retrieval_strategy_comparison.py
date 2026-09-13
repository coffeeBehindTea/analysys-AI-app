"""候选检索策略对比服务的离线测试。

本测试模块不访问Embedding API、Chroma数据库或真实报告文件。

测试范围：
1. 三份可比较报告能够形成统一对比结果；
2. 候选层胜者按照完整证据召回率优先选择；
3. 指标完全相同时优先选择结构更简单的策略；
4. 不同模型、Collection、Top-K或Gold题集不能混在一起比较；
5. 重复、缺失或非法策略报告必须被拒绝；
6. Markdown必须包含指标、逐题变化和候选阶段安全声明。
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
    CandidateStrategyParameters,
    CandidateStrategyReport,
    RetrievalStrategyName,
)
from app.services.retrieval_strategy_comparison import (
    CandidateStrategyComparison,
    compare_candidate_strategy_reports,
    render_candidate_strategy_comparison,
)
from app.services.retrieval_strategy_evaluation import (
    build_candidate_strategy_report,
    evaluate_candidate_question,
)


# DocumentChunk要求content_hash是64位十六进制字符串。
# 测试只关心来源位置，不需要真的计算文件哈希，
# 因此使用稳定的测试值。
TEST_CONTENT_HASH = "e" * 64

# 三种策略必须使用完全相同的实验上下文，
# 这些常量用于构造可比较的离线报告。
TEST_EMBEDDING_MODEL = "embedding-3"
TEST_COLLECTION_NAME = "robot_knowledge_test"
TEST_TOP_K = 3


def make_chunk(
    *,
    chunk_index: int,
    source_file: str,
    page_or_section: str,
) -> DocumentChunk:
    """构造一条带稳定来源位置的测试Chunk。"""

    return DocumentChunk(
        chunk_id=(
            f"document-{chunk_index:03d}:"
            f"{chunk_index:06d}"
        ),
        document_id=f"document-{chunk_index:03d}",
        source_file=source_file,
        page_or_section=page_or_section,
        chunk_index=chunk_index,
        content_hash=TEST_CONTENT_HASH,
        content=f"测试正文 {chunk_index}",
    )


def make_questions() -> list[GoldQuestion]:
    """构造单跳、多跳和无答案三种Gold问题。"""

    return [
        GoldQuestion(
            question_id="q001",
            question="单跳证据在哪里？",
            question_type="single_hop",
            answerable=True,
            reference_answer="在manual-a.pdf第1页。",
            expected_evidence=[
                ExpectedEvidence(
                    source_file="manual-a.pdf",
                    page_or_section="page: 1",
                )
            ],
            tags=["single-hop"],
            notes="测试单跳召回",
        ),
        GoldQuestion(
            question_id="q002",
            question="跨文档流程需要哪两份证据？",
            question_type="multi_hop",
            answerable=True,
            reference_answer="需要B和C两份证据。",
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
            tags=["multi-hop"],
            notes="测试完整证据召回",
        ),
        GoldQuestion(
            question_id="q003",
            question="知识库中不存在的真实设备编号是什么？",
            question_type="unanswerable",
            answerable=False,
            reference_answer=None,
            expected_evidence=[],
            tags=["unanswerable"],
            notes="测试无答案题保持在实验集合中",
        ),
    ]


def make_retrieved_chunks(
    *,
    strategy: RetrievalStrategyName,
    chunks: list[DocumentChunk],
) -> list[RetrievedChunk | HybridRetrievedChunk]:
    """根据策略构造类型正确、排名连续的候选结果。"""

    results: list[
        RetrievedChunk | HybridRetrievedChunk
    ] = []

    for rank, chunk in enumerate(chunks, start=1):
        if strategy == "vector_baseline":
            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    similarity=0.80 - rank * 0.01,
                    rank=rank,
                )
            )
            continue

        results.append(
            HybridRetrievedChunk(
                chunk=chunk,
                rrf_score=0.04 - rank * 0.001,
                rank=rank,
                vector_rank=rank,
                vector_similarity=(
                    0.80 - rank * 0.01
                ),
                keyword_rank=rank,
                keyword_score=10.0 - rank,

                # 现有HybridRetrievedChunk契约要求：
                # 只要声明该候选来自关键词路径，
                # 就必须至少保留一项真实命中解释。
                #
                # 本测试不评价分词算法，
                # 因此使用稳定的模拟命中词。
                matched_lexical_terms=(
                    "测试",
                ),
            )
        )

    return results


def make_parameters(
    strategy: RetrievalStrategyName,
) -> CandidateStrategyParameters:
    """构造与三种策略各自能力一致的参数。"""

    if strategy == "vector_baseline":
        return CandidateStrategyParameters(
            strategy=strategy,
            top_k=TEST_TOP_K,
        )

    return CandidateStrategyParameters(
        strategy=strategy,
        top_k=TEST_TOP_K,
        candidate_k=20,
        rank_constant=60,
        rewrite_version=(
            "deterministic-v1"
            if strategy == "hybrid_rrf_rewrite"
            else None
        ),
    )


def make_report(
    *,
    strategy: RetrievalStrategyName,
    q001_chunks: list[DocumentChunk],
    q002_chunks: list[DocumentChunk],
    collection_name: str = TEST_COLLECTION_NAME,
) -> CandidateStrategyReport:
    """使用真实评分函数构造一份自洽的候选报告。"""

    questions = make_questions()

    # 无答案题仍会拥有“最相近候选”，
    # 但候选阶段不会据此判断是否允许回答。
    unanswerable_candidate = make_chunk(
        chunk_index=99,
        source_file="distractor.pdf",
        page_or_section="page: 99",
    )

    evaluations = [
        evaluate_candidate_question(
            question=questions[0],
            retrieved_chunks=(
                make_retrieved_chunks(
                    strategy=strategy,
                    chunks=q001_chunks,
                )
            ),
        ),
        evaluate_candidate_question(
            question=questions[1],
            retrieved_chunks=(
                make_retrieved_chunks(
                    strategy=strategy,
                    chunks=q002_chunks,
                )
            ),
        ),
        evaluate_candidate_question(
            question=questions[2],
            retrieved_chunks=(
                make_retrieved_chunks(
                    strategy=strategy,
                    chunks=[unanswerable_candidate],
                )
            ),
        ),
    ]

    return build_candidate_strategy_report(
        embedding_model=TEST_EMBEDDING_MODEL,
        collection_name=collection_name,
        parameters=make_parameters(strategy),
        evaluations=evaluations,
    )


def make_three_reports(
) -> list[CandidateStrategyReport]:
    """构造逐步改善的三种候选策略报告。"""

    evidence_a = make_chunk(
        chunk_index=1,
        source_file="manual-a.pdf",
        page_or_section="page: 1",
    )
    evidence_b = make_chunk(
        chunk_index=2,
        source_file="manual-b.pdf",
        page_or_section="page: 2",
    )
    evidence_c = make_chunk(
        chunk_index=3,
        source_file="procedure-c.md",
        page_or_section="section: C",
    )
    distractor = make_chunk(
        chunk_index=4,
        source_file="distractor.pdf",
        page_or_section="page: 4",
    )

    return [
        # 基线只完整解决q001。
        make_report(
            strategy="vector_baseline",
            q001_chunks=[evidence_a],
            q002_chunks=[distractor],
        ),
        # 普通混合检索为q002找到一半证据。
        make_report(
            strategy="hybrid_rrf",
            q001_chunks=[evidence_a],
            q002_chunks=[evidence_b],
        ),
        # 查询改写方案为q002找齐两份证据。
        make_report(
            strategy="hybrid_rrf_rewrite",
            q001_chunks=[evidence_a],
            q002_chunks=[evidence_b, evidence_c],
        ),
    ]


def test_comparison_selects_best_candidate_strategy(
) -> None:
    """完整证据召回更高的改写策略应成为候选层胜者。"""

    comparison = compare_candidate_strategy_reports(
        make_three_reports()
    )

    assert isinstance(
        comparison,
        CandidateStrategyComparison,
    )
    assert comparison.candidate_winner == (
        "hybrid_rrf_rewrite"
    )
    assert comparison.question_count == 3
    assert comparison.answerable_question_count == 2
    assert comparison.unanswerable_question_count == 1
    assert comparison.expected_evidence_count == 3

    rewrite_row = comparison.strategy_rows[2]

    assert rewrite_row.strategy == (
        "hybrid_rrf_rewrite"
    )
    assert rewrite_row.recall_at_3 == 1.0
    assert (
        rewrite_row.fully_recalled_at_3_rate
        == 1.0
    )
    assert rewrite_row.recall_at_3_delta == (
        pytest.approx(2 / 3)
    )
    assert (
        rewrite_row.fully_recalled_at_3_rate_delta
        == pytest.approx(0.5)
    )

    # q002从0/2提升为2/2，因此属于明确改进。
    assert comparison.improved_question_ids == (
        "q002",
    )
    assert comparison.regressed_question_ids == ()
    assert comparison.unresolved_question_ids == ()


def test_equal_metrics_prefer_simpler_strategy(
) -> None:
    """所有指标相同时应优先保留纯向量基线。"""

    reports = make_three_reports()
    baseline = reports[0]

    # 使用与基线相同的逐题命中情况，
    # 但保留混合策略自己的结果类型和参数。
    evidence_a = baseline.results[0].retrieved_chunks[0].chunk
    distractor = baseline.results[1].retrieved_chunks[0].chunk

    tied_reports = [
        baseline,
        make_report(
            strategy="hybrid_rrf",
            q001_chunks=[evidence_a],
            q002_chunks=[distractor],
        ),
        make_report(
            strategy="hybrid_rrf_rewrite",
            q001_chunks=[evidence_a],
            q002_chunks=[distractor],
        ),
    ]

    comparison = compare_candidate_strategy_reports(
        tied_reports
    )

    assert comparison.candidate_winner == (
        "vector_baseline"
    )


@pytest.mark.parametrize(
    ("field_name", "changed_value", "expected_message"),
    [
        (
            "embedding_model",
            "embedding-other",
            "Embedding模型必须一致",
        ),
        (
            "collection_name",
            "other_collection",
            "Collection名称必须一致",
        ),
    ],
    ids=[
        "embedding-model",
        "collection-name",
    ],
)
def test_incompatible_experiment_context_is_rejected(
    field_name: str,
    changed_value: str,
    expected_message: str,
) -> None:
    """不同向量空间或语料快照的报告不能横向比较。"""

    reports = make_three_reports()
    reports[1] = reports[1].model_copy(
        update={field_name: changed_value},
        deep=True,
    )

    with pytest.raises(
        ValueError,
        match=expected_message,
    ):
        compare_candidate_strategy_reports(reports)


def test_different_top_k_is_rejected(
) -> None:
    """不同候选展示深度的Recall不能作为公平实验比较。"""

    reports = make_three_reports()
    changed_parameters = (
        reports[1].parameters.model_copy(
            update={"top_k": 4},
        )
    )
    reports[1] = reports[1].model_copy(
        update={
            "parameters": changed_parameters,
        },
        deep=True,
    )

    with pytest.raises(
        ValueError,
        match="top_k必须一致",
    ):
        compare_candidate_strategy_reports(reports)


def test_different_gold_question_contract_is_rejected(
) -> None:
    """题目正文或Gold证据改变后不能伪装成同一实验。"""

    reports = make_three_reports()
    changed_result = reports[2].results[0].model_copy(
        update={
            "question": "已经被修改的问题",
        },
        deep=True,
    )
    changed_results = [
        changed_result,
        *reports[2].results[1:],
    ]
    reports[2] = reports[2].model_copy(
        update={"results": changed_results},
        deep=True,
    )

    with pytest.raises(
        ValueError,
        match="Gold问题契约必须一致",
    ):
        compare_candidate_strategy_reports(reports)


def test_missing_strategy_report_is_rejected(
) -> None:
    """缺少任意一种规定策略时不能生成三策略报告。"""

    with pytest.raises(
        ValueError,
        match="必须恰好包含三种候选策略",
    ):
        compare_candidate_strategy_reports(
            make_three_reports()[:2]
        )


def test_duplicate_strategy_report_is_rejected(
) -> None:
    """同一种策略出现两次时不能覆盖先前报告。"""

    reports = make_three_reports()

    with pytest.raises(
        ValueError,
        match="不能包含重复策略",
    ):
        compare_candidate_strategy_reports(
            [
                reports[0],
                reports[0].model_copy(deep=True),
                reports[1],
            ]
        )


def test_non_list_and_invalid_items_are_rejected(
) -> None:
    """公共函数必须执行运行时容器和元素类型检查。"""

    with pytest.raises(
        TypeError,
        match="reports必须是列表",
    ):
        compare_candidate_strategy_reports(
            ()  # type: ignore[arg-type]
        )

    with pytest.raises(
        TypeError,
        match="必须是CandidateStrategyReport",
    ):
        compare_candidate_strategy_reports(
            [
                make_three_reports()[0],
                object(),
                make_three_reports()[2],
            ]  # type: ignore[list-item]
        )


def test_markdown_contains_metrics_changes_and_safety_boundary(
) -> None:
    """正式文档必须解释结果，也必须说明尚未完成门控评测。"""

    comparison = compare_candidate_strategy_reports(
        make_three_reports()
    )

    markdown = render_candidate_strategy_comparison(
        comparison
    )

    assert (
        "# Week 3 候选检索策略对比"
        in markdown
    )
    assert "关键词 + 向量 + RRF + 查询改写" in markdown
    assert "Recall@3" in markdown
    assert "1.000" in markdown
    assert "+0.667" in markdown
    assert "q002" in markdown
    assert "候选阶段，不代表已经获准接入在线 API" in markdown
    assert "无答案错误召回率尚未计算" in markdown


def test_renderer_rejects_wrong_object_type(
) -> None:
    """渲染器只能接收已经比较完成的结果对象。"""

    with pytest.raises(
        TypeError,
        match="comparison必须是CandidateStrategyComparison",
    ):
        render_candidate_strategy_comparison(
            object()  # type: ignore[arg-type]
        )
