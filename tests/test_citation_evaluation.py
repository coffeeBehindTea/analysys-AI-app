"""最终RAG引用匹配与单题评分的离线测试。"""

import pytest

from app.schemas.evaluation import (
    ExpectedEvidence,
    GoldQuestion,
)

from app.schemas.knowledge_query import (
    KnowledgeCitation,
    KnowledgeQueryResponse,
)

from app.services.citation_evaluation import (
    calculate_citation_metrics,
    citation_matches,
    count_correct_citations,
    count_matched_expected_evidence,
    evaluate_citation_question,
)


def make_single_hop_gold(
    *,
    question_id: str = "q005",
    expected_evidence: (
        list[ExpectedEvidence] | None
    ) = None,
) -> GoldQuestion:
    """创建一条合法的单跳可回答问题。"""

    # 没有显式传入预期证据时，
    # 使用q005对应的page 5。
    selected_evidence = (
        expected_evidence
        if expected_evidence is not None
        else [
            ExpectedEvidence(
                source_file=(
                    "工业移动机器人安全标准.pdf"
                ),
                page_or_section=(
                    "page: 5; section: 1 范围"
                ),
            )
        ]
    )

    return GoldQuestion(
        question_id=question_id,
        question=(
            "该征求意见稿适用于哪些机器人？"
        ),
        question_type="single_hop",
        answerable=True,
        reference_answer=(
            "该征求意见稿适用于工业移动机器人。"
        ),
        expected_evidence=selected_evidence,
        tags=[
            "safety-standard",
        ],
        notes="单跳引用评测测试数据。",
    )


def make_multi_hop_gold(
) -> GoldQuestion:
    """创建需要两项不同证据的多跳问题。"""

    return GoldQuestion(
        question_id="q013",
        question=(
            "结合适用范围和安全要求，"
            "说明该标准适用对象及急停要求。"
        ),
        question_type="multi_hop",
        answerable=True,
        reference_answer=(
            "该标准适用于工业移动机器人，"
            "并要求配置符合要求的急停功能。"
        ),

        # multi_hop问题至少需要两项预期证据。
        expected_evidence=[
            ExpectedEvidence(
                source_file=(
                    "工业移动机器人安全标准.pdf"
                ),
                page_or_section=(
                    "page: 5; section: 1 范围"
                ),
            ),
            ExpectedEvidence(
                source_file=(
                    "工业移动机器人安全标准.pdf"
                ),
                page_or_section=(
                    "page: 12; section: 急停要求"
                ),
            ),
        ],
        tags=[
            "multi-hop",
            "safety-standard",
        ],
        notes="验证多项预期证据是否全部被引用。",
    )


def make_unanswerable_gold(
) -> GoldQuestion:
    """创建一条无答案问题。"""

    return GoldQuestion(
        question_id="q020",
        question=(
            "robot-001当前位于哪里，"
            "电量是多少？"
        ),
        question_type="unanswerable",
        answerable=False,
        reference_answer=None,
        expected_evidence=[],
        tags=[
            "abstention",
            "telemetry",
        ],
        notes="静态知识库没有实时遥测。",
    )


def make_citation(
    *,
    source_file: str = (
        "工业移动机器人安全标准.pdf"
    ),
    page_or_section: str = "page: 5",
    rank: int = 1,
    chunk_index: int = 15,
) -> KnowledgeCitation:
    """创建一条可调整来源位置的假API引用。"""

    document_id = "6" * 64

    return KnowledgeCitation(
        chunk_id=(
            f"{document_id}:"
            f"{chunk_index:06d}"
        ),
        document_id=document_id,
        source_file=source_file,
        page_or_section=page_or_section,
        chunk_index=chunk_index,
        rank=rank,
        similarity=0.80,
        excerpt=(
            "本文件适用于工业移动机器人。"
        ),
    )


def make_answered_response(
    citations: list[KnowledgeCitation],
) -> KnowledgeQueryResponse:
    """创建一条带指定引用的正常回答。"""

    return KnowledgeQueryResponse(
        answer=(
            "该征求意见稿适用于"
            "工业移动机器人。"
        ),
        citations=citations,
        retrieval_ms=100.0,
        abstained=False,
    )


def make_abstained_response(
) -> KnowledgeQueryResponse:
    """创建固定文本、空引用的拒答响应。"""

    return KnowledgeQueryResponse(
        answer="知识库没有足够证据回答该问题。",
        citations=[],
        retrieval_ms=100.0,
        abstained=True,
    )


def test_citation_matches_pdf_page_with_gold_section(
) -> None:
    """PDF实际页码应匹配带辅助section的Gold位置。"""

    citation = make_citation(
        page_or_section="page: 5",
    )

    expected = ExpectedEvidence(
        source_file=(
            "工业移动机器人安全标准.pdf"
        ),
        page_or_section=(
            "page: 5; section: 1 范围"
        ),
    )

    # PDF Loader只稳定保存物理页码page: 5。
    #
    # Gold中的section是供人阅读的辅助说明，
    # 因此页码相同就应判定为匹配。
    assert citation_matches(
        citation,
        expected,
    ) is True


def test_citation_rejects_wrong_source_file(
) -> None:
    """页码相同但文件不同不能判为正确引用。"""

    citation = make_citation(
        source_file="另一个文档.pdf",
        page_or_section="page: 5",
    )

    expected = ExpectedEvidence(
        source_file=(
            "工业移动机器人安全标准.pdf"
        ),
        page_or_section="page: 5",
    )

    assert citation_matches(
        citation,
        expected,
    ) is False


def test_citation_rejects_wrong_pdf_page(
) -> None:
    """文件相同但物理页码不同不能判为正确引用。"""

    citation = make_citation(
        page_or_section="page: 6",
    )

    expected = ExpectedEvidence(
        source_file=(
            "工业移动机器人安全标准.pdf"
        ),
        page_or_section=(
            "page: 5; section: 1 范围"
        ),
    )

    assert citation_matches(
        citation,
        expected,
    ) is False


def test_duplicate_chunks_count_as_two_correct_citations_but_one_expected(
) -> None:
    """重叠Chunk不能重复增加Gold证据覆盖数。"""

    expected_evidence = [
        ExpectedEvidence(
            source_file=(
                "工业移动机器人安全标准.pdf"
            ),
            page_or_section="page: 5",
        )
    ]

    # 两条不同Chunk都来自同一个正确页码。
    citations = [
        make_citation(
            rank=1,
            chunk_index=15,
        ),
        make_citation(
            rank=2,
            chunk_index=16,
        ),
    ]

    # 从返回引用角度看，两条都正确。
    assert count_correct_citations(
        citations,
        expected_evidence,
    ) == 2

    # 从Gold覆盖角度看，只覆盖了一项预期证据。
    assert count_matched_expected_evidence(
        citations,
        expected_evidence,
    ) == 1


def test_evaluate_answerable_question_with_correct_citation(
) -> None:
    """q005正常回答和正确引用应判为完整证据回答。"""

    result = evaluate_citation_question(
        question=make_single_hop_gold(),
        response=make_answered_response(
            citations=[
                make_citation()
            ],
        ),
    )

    assert (
        result.gold_question.question_id
        == "q005"
    )

    assert result.returned_citation_count == 1
    assert result.correct_citation_count == 1
    assert result.expected_evidence_count == 1

    assert (
        result.matched_expected_evidence_count
        == 1
    )

    assert result.citation_correctness == 1.0
    assert result.citation_coverage == 1.0
    assert result.fully_grounded is True
    assert result.correctly_abstained is False


def test_evaluate_answerable_abstention_has_zero_coverage(
) -> None:
    """可回答问题被拒答时不应算作完整证据回答。"""

    result = evaluate_citation_question(
        question=make_single_hop_gold(
            question_id="q006",
        ),
        response=make_abstained_response(),
    )

    assert result.response.abstained is True

    assert result.returned_citation_count == 0
    assert result.correct_citation_count == 0

    # 没有返回引用，所以引用正确率没有分母。
    assert result.citation_correctness is None

    # Gold有一项预期证据，但一项都没有引用，
    # 所以覆盖率是0/1=0.0。
    assert result.citation_coverage == 0.0

    assert result.fully_grounded is False

    # correctly_abstained只用于Gold无答案问题。
    #
    # q006本来可以回答，因此这里必须是False。
    assert result.correctly_abstained is False


def test_evaluate_unanswerable_question_accepts_abstention(
) -> None:
    """q020拒答且空引用应判为正确拒答。"""

    result = evaluate_citation_question(
        question=make_unanswerable_gold(),
        response=make_abstained_response(),
    )

    assert result.response.abstained is True
    assert result.returned_citation_count == 0

    # 无返回引用，所以正确率不适用。
    assert result.citation_correctness is None

    # 无答案问题没有预期证据，
    # 所以覆盖率也不适用。
    assert result.citation_coverage is None

    assert result.fully_grounded is False
    assert result.correctly_abstained is True


def test_multi_hop_missing_one_evidence_is_not_fully_grounded(
) -> None:
    """多跳回答引用都正确但缺一项证据时不能算完整。"""

    result = evaluate_citation_question(
        question=make_multi_hop_gold(),

        # 只返回page 5，没有返回page 12。
        response=make_answered_response(
            citations=[
                make_citation(
                    page_or_section="page: 5",
                )
            ],
        ),
    )

    # 返回的一条引用本身是正确的。
    assert result.correct_citation_count == 1
    assert result.returned_citation_count == 1
    assert result.citation_correctness == 1.0

    # Gold需要page 5和page 12，
    # 现在只覆盖其中一项。
    assert result.expected_evidence_count == 2

    assert (
        result.matched_expected_evidence_count
        == 1
    )

    assert result.citation_coverage == 0.5

    # 引用正确不等于证据完整。
    assert result.fully_grounded is False


def test_metrics_expose_high_correctness_but_low_coverage(
) -> None:
    """少量回答引用正确时不能掩盖大量错误拒答。"""

    # q005正常回答并引用正确证据。
    answered_result = (
        evaluate_citation_question(
            question=make_single_hop_gold(
                question_id="q005",
            ),
            response=make_answered_response(
                citations=[
                    make_citation()
                ],
            ),
        )
    )

    # q006在Gold中可以回答，
    # 但API因为阈值过高而拒答。
    abstained_answerable_result = (
        evaluate_citation_question(
            question=make_single_hop_gold(
                question_id="q006",
            ),
            response=make_abstained_response(),
        )
    )

    # q020在Gold中没有答案，
    # API正确拒答。
    unanswerable_result = (
        evaluate_citation_question(
            question=make_unanswerable_gold(),
            response=make_abstained_response(),
        )
    )

    metrics = calculate_citation_metrics(
        [
            answered_result,
            abstained_answerable_result,
            unanswerable_result,
        ]
    )

    assert metrics.total_question_count == 3
    assert metrics.answerable_question_count == 2
    assert metrics.unanswerable_question_count == 1

    # 两道可回答题中，只有q005实际回答。
    assert (
        metrics.answered_answerable_count
        == 1
    )

    # 只有q005属于完整证据回答。
    assert (
        metrics.fully_grounded_answer_count
        == 1
    )

    # q020正确拒答。
    assert (
        metrics
        .correctly_abstained_unanswerable_count
        == 1
    )

    # 全部响应只返回q005的一条引用，
    # 而且这一条引用正确。
    assert metrics.returned_citation_count == 1
    assert metrics.correct_citation_count == 1

    # 所以引用正确率是1/1=100%。
    assert metrics.citation_correctness == 1.0

    # 两道可回答题各有一项预期证据，
    # 总计两项，但只引用了q005的一项。
    assert metrics.expected_evidence_count == 2

    assert (
        metrics.matched_expected_evidence_count
        == 1
    )

    # 引用覆盖率是1/2=50%。
    assert metrics.citation_coverage == 0.5

    # 可回答率是1/2=50%。
    assert (
        metrics.answerable_response_rate
        == 0.5
    )

    # 完整证据回答率也是1/2=50%。
    assert (
        metrics.fully_grounded_answer_rate
        == 0.5
    )

    # 唯一无答案题q020被正确拒答。
    assert (
        metrics.unanswerable_abstention_rate
        == 1.0
    )


def test_metrics_count_wrong_citation_in_denominator(
) -> None:
    """错误引用必须降低全局引用正确率。"""

    correct_result = evaluate_citation_question(
        question=make_single_hop_gold(
            question_id="q005",
        ),
        response=make_answered_response(
            citations=[
                make_citation()
            ],
        ),
    )

    # q006虽然正常回答，
    # 但引用来自错误文件。
    wrong_citation_result = (
        evaluate_citation_question(
            question=make_single_hop_gold(
                question_id="q006",
            ),
            response=make_answered_response(
                citations=[
                    make_citation(
                        source_file=(
                            "错误来源.pdf"
                        ),
                    )
                ],
            ),
        )
    )

    metrics = calculate_citation_metrics(
        [
            correct_result,
            wrong_citation_result,
        ]
    )

    # 两道可回答题都实际给出了回答。
    assert (
        metrics.answerable_response_rate
        == 1.0
    )

    # 总共返回两条引用，
    # 其中只有q005的一条正确。
    assert metrics.returned_citation_count == 2
    assert metrics.correct_citation_count == 1

    # 1条正确 / 2条返回 = 50%。
    assert metrics.citation_correctness == 0.5

    # 两项预期证据中只覆盖q005的一项。
    assert metrics.expected_evidence_count == 2

    assert (
        metrics.matched_expected_evidence_count
        == 1
    )

    assert metrics.citation_coverage == 0.5

    # 两道题中只有q005完成无错误引用。
    assert (
        metrics.fully_grounded_answer_rate
        == 0.5
    )

    # 这个测试集没有无答案问题。
    #
    # 分母为0时，全局无答案拒答率定义为0.0。
    assert (
        metrics.unanswerable_abstention_rate
        == 0.0
    )


def test_metrics_reject_empty_results(
) -> None:
    """没有逐题结果时不能生成汇总指标。"""

    with pytest.raises(
        ValueError,
        match="结果列表不能为空",
    ):
        calculate_citation_metrics([])


def test_metrics_reject_duplicate_question_id(
) -> None:
    """同一道题不能被重复计入全局分母。"""

    q005_result = evaluate_citation_question(
        question=make_single_hop_gold(
            question_id="q005",
        ),
        response=make_answered_response(
            citations=[
                make_citation()
            ],
        ),
    )

    with pytest.raises(
        ValueError,
        match="重复question_id",
    ):
        # 同一个q005被传入两次。
        calculate_citation_metrics(
            [
                q005_result,
                q005_result,
            ]
        )