"""候选检索策略评测Schema的纯离线测试。

这些测试只构造Pydantic对象，不访问Embedding、
Chroma、关键词索引、LLM或网络。

测试目标是验证：

1. 四种候选策略只能携带各自允许的参数；
2. 纯向量和混合检索结果都能保留真实审计字段；
3. 逐题排名、命中数量和答案类型保持一致；
4. Recall必须由证据命中数量正确计算；
5. 报告声明的策略、结果类型和汇总指标一致。
"""

from hashlib import sha256

import pytest
from pydantic import ValidationError

from app.schemas.evaluation import (
    ExpectedEvidence,
)
from app.schemas.retrieval import (
    DocumentChunk,
    HybridRetrievedChunk,
    RetrievedChunk,
)
from app.schemas.retrieval_strategy import (
    CandidateQuestionEvaluation,
    CandidateStrategyMetrics,
    CandidateStrategyParameters,
    CandidateStrategyReport,
    HeadPreservingRerankParameters,
)


def make_chunk(
    chunk_name: str,
) -> DocumentChunk:
    """创建具有稳定哈希和来源位置的测试Chunk。"""

    content = (
        f"{chunk_name}的测试证据正文"
    )

    return DocumentChunk(
        chunk_id=(
            sha256(
                chunk_name.encode("utf-8")
            ).hexdigest()
            + ":000000"
        ),
        document_id=sha256(
            (
                "document-"
                + chunk_name
            ).encode("utf-8")
        ).hexdigest(),
        source_file=f"{chunk_name}.txt",
        page_or_section=(
            f"section: {chunk_name}"
        ),
        chunk_index=0,
        content_hash=sha256(
            content.encode("utf-8")
        ).hexdigest(),
        content=content,
    )


def make_vector_result(
    *,
    rank: int = 1,
    chunk_name: str = "vector-evidence",
) -> RetrievedChunk:
    """创建保留余弦相似度的纯向量候选。"""

    return RetrievedChunk(
        chunk=make_chunk(chunk_name),
        similarity=(
            0.80 - (rank - 1) * 0.05
        ),
        rank=rank,
    )


def make_hybrid_result(
    *,
    rank: int = 1,
    chunk_name: str = "hybrid-evidence",
) -> HybridRetrievedChunk:
    """创建同时带向量和关键词审计字段的融合候选。"""

    return HybridRetrievedChunk(
        chunk=make_chunk(chunk_name),
        rrf_score=(
            0.04 - (rank - 1) * 0.005
        ),
        rank=rank,
        vector_rank=rank,
        vector_similarity=(
            0.75 - (rank - 1) * 0.05
        ),
        keyword_rank=rank,
        keyword_score=(
            12.0 - (rank - 1)
        ),
        matched_identifiers=(
            "err-test-1001",
        ),
    )


def make_expected_evidence(
    evidence_name: str = "expected",
) -> ExpectedEvidence:
    """创建一项人工标注的预期证据位置。"""

    return ExpectedEvidence(
        source_file=f"{evidence_name}.txt",
        page_or_section=(
            f"section: {evidence_name}"
        ),
    )


def make_answerable_evaluation(
    retrieved_result: (
        RetrievedChunk
        | HybridRetrievedChunk
    ),
    *,
    question_id: str = "q001",
) -> CandidateQuestionEvaluation:
    """创建一项Top-1和Top-3均完整召回的可回答结果。"""

    return CandidateQuestionEvaluation(
        question_id=question_id,
        question="测试可回答问题",
        question_type="single_hop",
        answerable=True,
        expected_evidence=[
            make_expected_evidence()
        ],
        retrieved_chunks=[
            retrieved_result
        ],
        matched_evidence_at_1=1,
        matched_evidence_at_3=1,
        fully_recalled_at_1=True,
        fully_recalled_at_3=True,
    )


def make_unanswerable_evaluation(
    retrieved_result: (
        RetrievedChunk
        | HybridRetrievedChunk
    ),
    *,
    question_id: str = "q002",
) -> CandidateQuestionEvaluation:
    """创建保留候选、但没有Gold证据的无答案结果。"""

    return CandidateQuestionEvaluation(
        question_id=question_id,
        question="测试无答案问题",
        question_type="unanswerable",
        answerable=False,
        expected_evidence=[],
        retrieved_chunks=[
            retrieved_result
        ],
        matched_evidence_at_1=0,
        matched_evidence_at_3=0,
        fully_recalled_at_1=False,
        fully_recalled_at_3=False,
    )


def make_valid_metrics(
) -> CandidateStrategyMetrics:
    """创建与一项可回答和一项无答案结果一致的指标。"""

    return CandidateStrategyMetrics(
        answerable_question_count=1,
        unanswerable_question_count=1,
        expected_evidence_count=1,
        matched_evidence_at_1=1,
        matched_evidence_at_3=1,
        recall_at_1=1.0,
        recall_at_3=1.0,
        fully_recalled_at_1_count=1,
        fully_recalled_at_3_count=1,
    )


def make_rerank_parameters(
) -> HeadPreservingRerankParameters:
    """创建一组可审计的头部保留重排参数。"""

    return HeadPreservingRerankParameters(
        version="head-preserving-v1",
        fused_head_k=1,
        vector_head_k=1,
        keyword_head_k=2,
    )


def make_strategy_parameters(
    strategy: str,
) -> CandidateStrategyParameters:
    """根据策略名称创建一组合法实验参数。"""

    if strategy == "vector_baseline":
        return CandidateStrategyParameters(
            strategy="vector_baseline",
            top_k=3,
        )

    if strategy == "hybrid_rrf":
        return CandidateStrategyParameters(
            strategy="hybrid_rrf",
            top_k=3,
            candidate_k=20,
            rank_constant=60,
        )

    if strategy == "hybrid_rrf_rewrite":
        return CandidateStrategyParameters(
            strategy="hybrid_rrf_rewrite",
            top_k=3,
            candidate_k=20,
            rank_constant=60,
            rewrite_version="deterministic-v1",
        )

    if (
        strategy
        == "hybrid_rrf_rewrite_rerank"
    ):
        return CandidateStrategyParameters(
            strategy=(
                "hybrid_rrf_rewrite_rerank"
            ),
            top_k=3,
            candidate_k=20,
            rank_constant=60,
            rewrite_version="deterministic-v2",
            rerank_parameters=(
                make_rerank_parameters()
            ),
        )

    raise AssertionError(
        f"测试未处理策略：{strategy}"
    )


@pytest.mark.parametrize(
    "strategy",
    [
        "vector_baseline",
        "hybrid_rrf",
        "hybrid_rrf_rewrite",
        "hybrid_rrf_rewrite_rerank",
    ],
)
def test_strategy_parameters_accept_each_supported_strategy(
    strategy: str,
) -> None:
    """四种声明过的策略都应接受各自完整的参数组合。"""

    parameters = make_strategy_parameters(
        strategy
    )

    assert parameters.strategy == strategy
    assert parameters.top_k == 3

    if strategy == "vector_baseline":
        assert parameters.candidate_k is None
        assert parameters.rank_constant is None
        assert parameters.rewrite_version is None
        assert parameters.rerank_parameters is None

        # 未使用重排的旧策略不应在JSON中增加无意义的null字段，
        # 从而保持历史报告的序列化结构稳定。
        assert (
            "rerank_parameters"
            not in parameters.model_dump()
        )
    else:
        assert parameters.candidate_k == 20
        assert parameters.rank_constant == 60

        if (
            strategy
            == "hybrid_rrf_rewrite_rerank"
        ):
            assert parameters.rewrite_version == (
                "deterministic-v2"
            )
            assert (
                parameters.rerank_parameters
                == make_rerank_parameters()
            )
            assert parameters.model_dump()[
                "rerank_parameters"
            ] == {
                "version": "head-preserving-v1",
                "fused_head_k": 1,
                "vector_head_k": 1,
                "keyword_head_k": 2,
            }
        else:
            assert (
                parameters.rerank_parameters
                is None
            )
            assert (
                "rerank_parameters"
                not in parameters.model_dump()
            )


@pytest.mark.parametrize(
    (
        "arguments",
        "expected_message",
    ),
    [
        (
            {
                "strategy": "vector_baseline",
                "top_k": 3,
                "candidate_k": 20,
            },
            "vector_baseline不能设置candidate_k",
        ),
        (
            {
                "strategy": "vector_baseline",
                "top_k": 3,
                "rank_constant": 60,
            },
            "vector_baseline不能设置rank_constant",
        ),
        (
            {
                "strategy": "vector_baseline",
                "top_k": 3,
                "rewrite_version": "v1",
            },
            "vector_baseline不能设置rewrite_version",
        ),
        (
            {
                "strategy": "hybrid_rrf",
                "top_k": 3,
                "rank_constant": 60,
            },
            "混合检索必须设置candidate_k",
        ),
        (
            {
                "strategy": "hybrid_rrf",
                "top_k": 3,
                "candidate_k": 20,
            },
            "混合检索必须设置rank_constant",
        ),
        (
            {
                "strategy": "hybrid_rrf",
                "top_k": 21,
                "candidate_k": 20,
                "rank_constant": 60,
            },
            "top_k不能大于candidate_k",
        ),
        (
            {
                "strategy": "hybrid_rrf",
                "top_k": 3,
                "candidate_k": 20,
                "rank_constant": 60,
                "rewrite_version": "v1",
            },
            "hybrid_rrf不能设置rewrite_version",
        ),
        (
            {
                "strategy": (
                    "hybrid_rrf_rewrite"
                ),
                "top_k": 3,
                "candidate_k": 20,
                "rank_constant": 60,
            },
            "hybrid_rrf_rewrite必须设置rewrite_version",
        ),
        (
            {
                "strategy": "vector_baseline",
                "top_k": 3,
                "rerank_parameters": (
                    make_rerank_parameters()
                ),
            },
            "vector_baseline不能设置rerank_parameters",
        ),
        (
            {
                "strategy": "hybrid_rrf",
                "top_k": 3,
                "candidate_k": 20,
                "rank_constant": 60,
                "rerank_parameters": (
                    make_rerank_parameters()
                ),
            },
            "hybrid_rrf不能设置rerank_parameters",
        ),
        (
            {
                "strategy": "hybrid_rrf_rewrite",
                "top_k": 3,
                "candidate_k": 20,
                "rank_constant": 60,
                "rewrite_version": "deterministic-v2",
                "rerank_parameters": (
                    make_rerank_parameters()
                ),
            },
            "hybrid_rrf_rewrite不能设置rerank_parameters",
        ),
        (
            {
                "strategy": (
                    "hybrid_rrf_rewrite_rerank"
                ),
                "top_k": 3,
                "candidate_k": 20,
                "rank_constant": 60,
                "rerank_parameters": (
                    make_rerank_parameters()
                ),
            },
            (
                "hybrid_rrf_rewrite_rerank"
                "必须设置rewrite_version"
            ),
        ),
        (
            {
                "strategy": (
                    "hybrid_rrf_rewrite_rerank"
                ),
                "top_k": 3,
                "candidate_k": 20,
                "rank_constant": 60,
                "rewrite_version": "deterministic-v2",
            },
            (
                "hybrid_rrf_rewrite_rerank"
                "必须设置rerank_parameters"
            ),
        ),
    ],
    ids=[
        "vector-with-candidate-k",
        "vector-with-rank-constant",
        "vector-with-rewrite-version",
        "hybrid-without-candidate-k",
        "hybrid-without-rank-constant",
        "top-k-larger-than-candidate-k",
        "plain-hybrid-with-rewrite-version",
        "rewrite-strategy-without-version",
        "vector-with-rerank-parameters",
        "plain-hybrid-with-rerank-parameters",
        "rewrite-with-rerank-parameters",
        "rerank-strategy-without-rewrite-version",
        "rerank-strategy-without-rerank-parameters",
    ],
)
def test_strategy_parameters_reject_inconsistent_combinations(
    arguments: dict[str, object],
    expected_message: str,
) -> None:
    """策略名称和不属于该策略的参数组合必须被拒绝。"""

    with pytest.raises(
        ValidationError,
        match=expected_message,
    ):
        CandidateStrategyParameters(
            **arguments,  # type: ignore[arg-type]
        )


def test_rerank_parameters_require_at_least_one_preserved_head(
) -> None:
    """全零策略没有重排行为，必须在报告契约边界被拒绝。"""

    with pytest.raises(
        ValidationError,
        match="至少保留一个检索头部",
    ):
        HeadPreservingRerankParameters(
            version="head-preserving-v1",
            fused_head_k=0,
            vector_head_k=0,
            keyword_head_k=0,
        )


@pytest.mark.parametrize(
    "retrieved_result",
    [
        pytest.param(
            make_vector_result(),
            id="vector-result",
        ),
        pytest.param(
            make_hybrid_result(),
            id="hybrid-result",
        ),
    ],
)
def test_question_evaluation_accepts_both_result_contracts(
    retrieved_result: (
        RetrievedChunk
        | HybridRetrievedChunk
    ),
) -> None:
    """逐题契约应保留纯向量或混合结果的具体类型。"""

    evaluation = make_answerable_evaluation(
        retrieved_result
    )

    assert (
        evaluation.retrieved_chunks[0]
        is retrieved_result
    )
    assert evaluation.fully_recalled_at_1 is True
    assert evaluation.fully_recalled_at_3 is True


@pytest.mark.parametrize(
    (
        "overrides",
        "expected_message",
    ),
    [
        (
            {
                "retrieved_chunks": [
                    make_vector_result(rank=2)
                ],
            },
            "必须按连续rank排序",
        ),
        (
            {
                "question_type": "unanswerable",
                "answerable": True,
            },
            "answerable必须为false",
        ),
        (
            {
                "answerable": False,
            },
            "可回答问题的answerable必须为true",
        ),
        (
            {
                "question_type": "multi_hop",
            },
            "multi_hop问题至少需要两项预期证据",
        ),
        (
            {
                "expected_evidence": [],
            },
            "可回答问题必须包含预期证据",
        ),
        (
            {
                "matched_evidence_at_1": 2,
                "matched_evidence_at_3": 2,
            },
            "Top-1命中数不能超过预期证据数",
        ),
        (
            {
                "matched_evidence_at_1": 1,
                "matched_evidence_at_3": 0,
            },
            "Top-1命中数不能大于Top-3命中数",
        ),
        (
            {
                "fully_recalled_at_1": False,
            },
            "fully_recalled_at_1与命中数量不一致",
        ),
        (
            {
                "question_type": "unanswerable",
                "answerable": False,
                "expected_evidence": [
                    make_expected_evidence()
                ],
                "matched_evidence_at_1": 0,
                "matched_evidence_at_3": 0,
                "fully_recalled_at_1": False,
                "fully_recalled_at_3": False,
            },
            "无答案问题不能包含预期证据",
        ),
        (
            {
                "question_type": "unanswerable",
                "answerable": False,
                "expected_evidence": [],
                "matched_evidence_at_1": 1,
                "matched_evidence_at_3": 1,
                "fully_recalled_at_1": False,
                "fully_recalled_at_3": False,
            },
            "无答案问题的证据命中数必须为0",
        ),
        (
            {
                "question_type": "unanswerable",
                "answerable": False,
                "expected_evidence": [],
                "matched_evidence_at_1": 0,
                "matched_evidence_at_3": 0,
                "fully_recalled_at_1": True,
                "fully_recalled_at_3": True,
            },
            "无答案问题不能标记为完整召回",
        ),
    ],
    ids=[
        "non-continuous-rank",
        "unanswerable-marked-answerable",
        "answerable-type-marked-false",
        "multi-hop-with-one-evidence",
        "answerable-without-evidence",
        "top-one-exceeds-expected",
        "top-one-exceeds-top-three",
        "wrong-fully-recalled-flag",
        "unanswerable-with-evidence",
        "unanswerable-with-match-count",
        "unanswerable-marked-fully-recalled",
    ],
)
def test_question_evaluation_rejects_inconsistent_fields(
    overrides: dict[str, object],
    expected_message: str,
) -> None:
    """逐题结果中的排名、类型、证据和命中字段必须一致。"""

    arguments: dict[str, object] = {
        "question_id": "q001",
        "question": "测试可回答问题",
        "question_type": "single_hop",
        "answerable": True,
        "expected_evidence": [
            make_expected_evidence()
        ],
        "retrieved_chunks": [
            make_vector_result()
        ],
        "matched_evidence_at_1": 1,
        "matched_evidence_at_3": 1,
        "fully_recalled_at_1": True,
        "fully_recalled_at_3": True,
    }
    arguments.update(overrides)

    with pytest.raises(
        ValidationError,
        match=expected_message,
    ):
        CandidateQuestionEvaluation(
            **arguments,  # type: ignore[arg-type]
        )


def test_metrics_accept_counts_and_matching_recall_values(
) -> None:
    """证据计数与除法结果一致时应构造成功。"""

    metrics = CandidateStrategyMetrics(
        answerable_question_count=2,
        unanswerable_question_count=1,
        expected_evidence_count=3,
        matched_evidence_at_1=1,
        matched_evidence_at_3=2,
        recall_at_1=(1 / 3),
        recall_at_3=(2 / 3),
        fully_recalled_at_1_count=0,
        fully_recalled_at_3_count=1,
    )

    assert metrics.recall_at_1 == pytest.approx(
        1 / 3
    )
    assert metrics.recall_at_3 == pytest.approx(
        2 / 3
    )


@pytest.mark.parametrize(
    (
        "overrides",
        "expected_message",
    ),
    [
        (
            {
                "matched_evidence_at_3": 4,
                "recall_at_3": 1.0,
            },
            "证据命中数不能超过预期证据总数",
        ),
        (
            {
                "matched_evidence_at_1": 3,
                "matched_evidence_at_3": 2,
                "recall_at_1": 1.0,
                "recall_at_3": (2 / 3),
            },
            "Top-1命中数不能大于Top-3命中数",
        ),
        (
            {
                "fully_recalled_at_1_count": 3,
            },
            "完整召回题数不能超过可回答题数",
        ),
        (
            {
                "fully_recalled_at_1_count": 2,
                "fully_recalled_at_3_count": 1,
            },
            "Top-1完整召回题数不能大于",
        ),
        (
            {
                "recall_at_1": 0.90,
            },
            "recall_at_1与证据命中数量不一致",
        ),
        (
            {
                "recall_at_3": 0.90,
            },
            "recall_at_3与证据命中数量不一致",
        ),
    ],
    ids=[
        "matches-exceed-expected",
        "top-one-matches-exceed-top-three",
        "fully-recalled-exceeds-answerable",
        "top-one-full-count-exceeds-top-three",
        "wrong-recall-at-one",
        "wrong-recall-at-three",
    ],
)
def test_metrics_reject_inconsistent_counts_or_recall(
    overrides: dict[str, object],
    expected_message: str,
) -> None:
    """汇总计数和Recall公式出现矛盾时必须拒绝报告。"""

    arguments: dict[str, object] = {
        "answerable_question_count": 2,
        "unanswerable_question_count": 1,
        "expected_evidence_count": 3,
        "matched_evidence_at_1": 1,
        "matched_evidence_at_3": 2,
        "recall_at_1": (1 / 3),
        "recall_at_3": (2 / 3),
        "fully_recalled_at_1_count": 0,
        "fully_recalled_at_3_count": 1,
    }
    arguments.update(overrides)

    with pytest.raises(
        ValidationError,
        match=expected_message,
    ):
        CandidateStrategyMetrics(
            **arguments,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "strategy",
    [
        "vector_baseline",
        "hybrid_rrf",
        "hybrid_rrf_rewrite",
    ],
)
def test_report_accepts_matching_strategy_results_and_metrics(
    strategy: str,
) -> None:
    """三种策略都应接受类型正确且汇总一致的完整报告。"""

    if strategy == "vector_baseline":
        first_result = make_vector_result()
        second_result = make_vector_result(
            chunk_name="vector-unanswerable"
        )
    else:
        first_result = make_hybrid_result()
        second_result = make_hybrid_result(
            chunk_name="hybrid-unanswerable"
        )

    report = CandidateStrategyReport(
        embedding_model="embedding-test-model",
        collection_name="test_collection",
        parameters=make_strategy_parameters(
            strategy
        ),
        metrics=make_valid_metrics(),
        results=[
            make_answerable_evaluation(
                first_result,
                question_id="q001",
            ),
            make_unanswerable_evaluation(
                second_result,
                question_id="q002",
            ),
        ],
    )

    assert report.parameters.strategy == strategy
    assert len(report.results) == 2
    assert report.metrics.recall_at_3 == 1.0
    assert "generated_at" not in report.model_dump()


def test_report_rejects_duplicate_question_ids(
) -> None:
    """同一道Gold题不能在一份策略报告中重复计分。"""

    with pytest.raises(
        ValidationError,
        match="不能包含重复question_id",
    ):
        CandidateStrategyReport(
            embedding_model="embedding-test-model",
            collection_name="test_collection",
            parameters=(
                make_strategy_parameters(
                    "vector_baseline"
                )
            ),
            metrics=make_valid_metrics(),
            results=[
                make_answerable_evaluation(
                    make_vector_result(),
                    question_id="q001",
                ),
                make_unanswerable_evaluation(
                    make_vector_result(
                        chunk_name="other"
                    ),
                    question_id="q001",
                ),
            ],
        )


@pytest.mark.parametrize(
    (
        "strategy",
        "retrieved_result",
    ),
    [
        (
            "vector_baseline",
            make_hybrid_result(),
        ),
        (
            "hybrid_rrf",
            make_vector_result(),
        ),
    ],
    ids=[
        "hybrid-result-in-vector-report",
        "vector-result-in-hybrid-report",
    ],
)
def test_report_rejects_result_type_that_does_not_match_strategy(
    strategy: str,
    retrieved_result: (
        RetrievedChunk
        | HybridRetrievedChunk
    ),
) -> None:
    """报告名称不能与内部候选结果类型互相矛盾。"""

    with pytest.raises(
        ValidationError,
        match="报告策略与检索结果类型不一致",
    ):
        CandidateStrategyReport(
            embedding_model="embedding-test-model",
            collection_name="test_collection",
            parameters=make_strategy_parameters(
                strategy
            ),
            metrics=CandidateStrategyMetrics(
                answerable_question_count=1,
                unanswerable_question_count=0,
                expected_evidence_count=1,
                matched_evidence_at_1=1,
                matched_evidence_at_3=1,
                recall_at_1=1.0,
                recall_at_3=1.0,
                fully_recalled_at_1_count=1,
                fully_recalled_at_3_count=1,
            ),
            results=[
                make_answerable_evaluation(
                    retrieved_result
                )
            ],
        )


def test_report_rejects_summary_that_disagrees_with_questions(
) -> None:
    """metrics题数即使自身合法，也必须与逐题结果一致。"""

    inconsistent_metrics = (
        CandidateStrategyMetrics(
            answerable_question_count=2,
            unanswerable_question_count=0,
            expected_evidence_count=1,
            matched_evidence_at_1=1,
            matched_evidence_at_3=1,
            recall_at_1=1.0,
            recall_at_3=1.0,
            fully_recalled_at_1_count=1,
            fully_recalled_at_3_count=1,
        )
    )

    with pytest.raises(
        ValidationError,
        match="汇总指标与逐题结果不一致",
    ):
        CandidateStrategyReport(
            embedding_model="embedding-test-model",
            collection_name="test_collection",
            parameters=(
                make_strategy_parameters(
                    "vector_baseline"
                )
            ),
            metrics=inconsistent_metrics,
            results=[
                make_answerable_evaluation(
                    make_vector_result(),
                    question_id="q001",
                ),
                make_unanswerable_evaluation(
                    make_vector_result(
                        chunk_name="unanswerable"
                    ),
                    question_id="q002",
                ),
            ],
        )


def test_report_rejects_more_candidates_than_top_k(
) -> None:
    """逐题保存的最终候选数量不能超过实验声明的Top-K。"""

    retrieved_chunks = [
        make_vector_result(
            rank=rank,
            chunk_name=f"vector-{rank}",
        )
        for rank in range(1, 5)
    ]

    evaluation = CandidateQuestionEvaluation(
        question_id="q001",
        question="测试Top-K边界",
        question_type="single_hop",
        answerable=True,
        expected_evidence=[
            make_expected_evidence()
        ],
        retrieved_chunks=retrieved_chunks,
        matched_evidence_at_1=1,
        matched_evidence_at_3=1,
        fully_recalled_at_1=True,
        fully_recalled_at_3=True,
    )

    with pytest.raises(
        ValidationError,
        match="逐题候选数量不能超过top_k",
    ):
        CandidateStrategyReport(
            embedding_model="embedding-test-model",
            collection_name="test_collection",
            parameters=(
                make_strategy_parameters(
                    "vector_baseline"
                )
            ),
            metrics=CandidateStrategyMetrics(
                answerable_question_count=1,
                unanswerable_question_count=0,
                expected_evidence_count=1,
                matched_evidence_at_1=1,
                matched_evidence_at_3=1,
                recall_at_1=1.0,
                recall_at_3=1.0,
                fully_recalled_at_1_count=1,
                fully_recalled_at_3_count=1,
            ),
            results=[evaluation],
        )


def test_serialization_keeps_rrf_and_similarity_as_different_fields(
) -> None:
    """混合分数不能在JSON数据中伪装成余弦相似度。"""

    vector_evaluation = (
        make_answerable_evaluation(
            make_vector_result()
        )
    )
    hybrid_evaluation = (
        make_answerable_evaluation(
            make_hybrid_result()
        )
    )

    vector_data = (
        vector_evaluation.model_dump()
        ["retrieved_chunks"][0]
    )
    hybrid_data = (
        hybrid_evaluation.model_dump()
        ["retrieved_chunks"][0]
    )

    assert "similarity" in vector_data
    assert "rrf_score" not in vector_data

    assert "rrf_score" in hybrid_data
    assert "vector_similarity" in hybrid_data
    assert "similarity" not in hybrid_data
