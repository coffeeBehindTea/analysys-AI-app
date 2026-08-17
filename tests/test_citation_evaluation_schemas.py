"""最终RAG引用评测Schema的离线测试。"""

from datetime import datetime, timezone

import pytest

# ValidationError表示传入数据没有通过
# Pydantic字段约束或model_validator校验。
from pydantic import ValidationError

# 引用评测的单题、汇总和报告数据契约。
from app.schemas.citation_evaluation import (
    CitationEvaluationReport,
    CitationMetrics,
    CitationQuestionEvaluation,
)

# ExpectedEvidence和GoldQuestion表示
# 人工标注的标准问题与预期证据。
from app.schemas.evaluation import (
    ExpectedEvidence,
    GoldQuestion,
)

# KnowledgeCitation表示API返回的一条引用；
# KnowledgeQueryResponse表示完整知识库回答。
from app.schemas.knowledge_query import (
    KnowledgeCitation,
    KnowledgeQueryResponse,
)


def make_answerable_gold(
) -> GoldQuestion:
    """创建一条与真实q005同结构的可回答问题。"""

    return GoldQuestion(
        question_id="q005",
        question=(
            "《工业移动机器人安全标准》"
            "征求意见稿适用于哪些生命周期阶段、"
            "行业和机器人类型？"
        ),
        question_type="single_hop",
        answerable=True,
        reference_answer=(
            "该征求意见稿适用于新设计、制造、"
            "安装、运行、维护和报废阶段，"
            "涉及仓储业、制造业和交通运输业，"
            "包括AGV、AMR等机器人类型。"
        ),
        expected_evidence=[
            ExpectedEvidence(
                source_file=(
                    "工业移动机器人安全标准.pdf"
                ),
                page_or_section=(
                    "page: 5; section: 1 范围"
                ),
            )
        ],
        tags=[
            "safety-standard",
            "scope",
        ],
        notes=(
            "验证回答是否引用标准草案的适用范围。"
        ),
    )


def make_unanswerable_gold(
) -> GoldQuestion:
    """创建一条与真实q020同结构的无答案问题。"""

    return GoldQuestion(
        question_id="q020",
        question=(
            "当前正在运行的robot-001"
            "位于仓库哪个位置，剩余电量是多少，"
            "正在执行什么订单？"
        ),
        question_type="unanswerable",
        answerable=False,

        # 无答案问题不能提供参考答案。
        reference_answer=None,

        # 无答案问题也不能标注预期证据。
        expected_evidence=[],
        tags=[
            "abstention",
            "real-time-data",
        ],
        notes=(
            "静态知识库不包含实时遥测和订单状态。"
        ),
    )


def make_correct_citation(
) -> KnowledgeCitation:
    """创建与q005预期位置匹配的一条真实结构引用。"""

    document_id = "6" * 64

    return KnowledgeCitation(
        chunk_id=(
            f"{document_id}:000015"
        ),
        document_id=document_id,
        source_file=(
            "工业移动机器人安全标准.pdf"
        ),
        page_or_section="page: 5",
        chunk_index=15,

        # 保留其原始检索排名。
        rank=2,

        similarity=0.6616196036338806,
        excerpt=(
            "1 范围\n"
            "本文件适用于新设计、制造、安装、"
            "运行、维护和报废的工业移动机器人。"
        ),
    )


def make_answered_response(
) -> KnowledgeQueryResponse:
    """创建q005对应的正常知识库响应。"""

    return KnowledgeQueryResponse(
        answer=(
            "该征求意见稿适用于新设计、制造、"
            "安装、运行、维护和报废的"
            "工业移动机器人。"
        ),
        citations=[
            make_correct_citation()
        ],
        retrieval_ms=178.5,
        abstained=False,
    )


def make_answered_evaluation(
) -> CitationQuestionEvaluation:
    """创建计数、比例和状态完全一致的q005评测。"""

    return CitationQuestionEvaluation(
        gold_question=make_answerable_gold(),
        response=make_answered_response(),

        # API返回了一条引用。
        returned_citation_count=1,

        # 这一条引用与Gold的文件和页码匹配。
        correct_citation_count=1,

        # q005只有一项预期证据。
        expected_evidence_count=1,

        # 该预期证据已被返回引用覆盖。
        matched_expected_evidence_count=1,

        # 1条正确引用 / 1条返回引用。
        citation_correctness=1.0,

        # 1项已覆盖证据 / 1项预期证据。
        citation_coverage=1.0,

        # 系统正常回答、所有引用正确、
        # 并且找全了全部预期证据。
        fully_grounded=True,

        # q005是可回答问题，
        # 所以不属于正确拒答。
        correctly_abstained=False,
    )


def test_answerable_evaluation_accepts_consistent_data(
) -> None:
    """q005形态的完整证据回答应通过校验。"""

    evaluation = make_answered_evaluation()

    assert (
        evaluation.gold_question.question_id
        == "q005"
    )

    assert evaluation.response.abstained is False

    assert evaluation.returned_citation_count == 1
    assert evaluation.correct_citation_count == 1
    assert evaluation.citation_correctness == 1.0
    assert evaluation.citation_coverage == 1.0
    assert evaluation.fully_grounded is True

    # q005不是无答案问题。
    assert evaluation.correctly_abstained is False


def test_unanswerable_evaluation_accepts_correct_abstention(
) -> None:
    """q020形态的固定拒答和空引用应通过校验。"""

    gold_question = make_unanswerable_gold()

    response = KnowledgeQueryResponse(
        answer="知识库没有足够证据回答该问题。",
        citations=[],
        retrieval_ms=178.5,
        abstained=True,
    )

    evaluation = CitationQuestionEvaluation(
        gold_question=gold_question,
        response=response,

        # 拒答响应没有返回引用。
        returned_citation_count=0,
        correct_citation_count=0,

        # 无答案问题没有预期证据。
        expected_evidence_count=0,
        matched_expected_evidence_count=0,

        # 两个比例都没有分母，因此使用None。
        citation_correctness=None,
        citation_coverage=None,

        # 无答案问题不属于完整证据回答。
        fully_grounded=False,

        # Gold标注无答案，API也确实拒答。
        correctly_abstained=True,
    )

    assert (
        evaluation.gold_question.question_id
        == "q020"
    )
    assert evaluation.response.abstained is True
    assert evaluation.response.citations == []

    # None是Python中的单例对象，
    # 因此使用is None检查。
    assert evaluation.citation_correctness is None
    assert evaluation.citation_coverage is None
    assert evaluation.correctly_abstained is True


def test_question_evaluation_rejects_wrong_returned_count(
) -> None:
    """声明的返回引用数必须等于响应中的实际数量。"""

    # model_dump()把Pydantic模型转换成普通字典。
    #
    # 嵌套的GoldQuestion、Response和Citation
    # 也会转换成可供model_validate()重新读取的数据。
    evaluation_data = (
        make_answered_evaluation().model_dump()
    )

    # response.citations中实际有一条引用，
    # 这里故意错误地声明为0。
    evaluation_data[
        "returned_citation_count"
    ] = 0

    with pytest.raises(
        ValidationError,
        match="returned_citation_count",
    ):
        # model_validate()用修改后的字典
        # 重新创建并校验模型。
        CitationQuestionEvaluation.model_validate(
            evaluation_data
        )


def test_question_evaluation_rejects_wrong_correctness(
) -> None:
    """引用正确率必须由正确数和返回总数计算。"""

    evaluation_data = (
        make_answered_evaluation().model_dump()
    )

    # 当前计数是1条正确引用/1条返回引用，
    # 正确率应为1.0。
    #
    # 这里故意填成0.5。
    evaluation_data[
        "citation_correctness"
    ] = 0.5

    with pytest.raises(
        ValidationError,
        match="citation_correctness",
    ):
        CitationQuestionEvaluation.model_validate(
            evaluation_data
        )


def test_question_evaluation_rejects_wrong_grounded_flag(
) -> None:
    """完整证据回答状态必须由真实响应和计数决定。"""

    evaluation_data = (
        make_answered_evaluation().model_dump()
    )

    # 当前响应没有拒答、引用全部正确，
    # 而且找全了全部预期证据，
    # 所以fully_grounded必须是True。
    #
    # 这里故意改成False。
    evaluation_data[
        "fully_grounded"
    ] = False

    with pytest.raises(
        ValidationError,
        match="fully_grounded",
    ):
        CitationQuestionEvaluation.model_validate(
            evaluation_data
        )


def make_abstained_evaluation_for_report(
) -> CitationQuestionEvaluation:
    """创建供汇总报告使用的合法q020评测结果。"""

    response = KnowledgeQueryResponse(
        answer="知识库没有足够证据回答该问题。",
        citations=[],
        retrieval_ms=178.5,
        abstained=True,
    )

    return CitationQuestionEvaluation(
        gold_question=make_unanswerable_gold(),
        response=response,
        returned_citation_count=0,
        correct_citation_count=0,
        expected_evidence_count=0,
        matched_expected_evidence_count=0,

        # 没有返回引用，也没有预期证据，
        # 两个比例都没有分母。
        citation_correctness=None,
        citation_coverage=None,

        fully_grounded=False,
        correctly_abstained=True,
    )


def make_consistent_metrics(
) -> CitationMetrics:
    """创建q005正常回答加q020正确拒答的汇总指标。"""

    return CitationMetrics(
        # 这个最小测试数据集一共两道题。
        total_question_count=2,

        # q005可以回答，q020无答案。
        answerable_question_count=1,
        unanswerable_question_count=1,

        # q005实际给出了答案。
        answered_answerable_count=1,

        # q005返回的引用完全匹配Gold。
        fully_grounded_answer_count=1,

        # q020正确拒答。
        correctly_abstained_unanswerable_count=1,

        # q005返回一条引用，q020没有引用。
        returned_citation_count=1,

        # q005的这一条引用正确。
        correct_citation_count=1,

        # q005只有一项预期证据。
        expected_evidence_count=1,

        # 这一项预期证据已经被引用。
        matched_expected_evidence_count=1,

        # 1条正确引用 / 1条返回引用。
        citation_correctness=1.0,

        # 1项命中证据 / 1项预期证据。
        citation_coverage=1.0,

        # 1道已回答可回答题 / 1道可回答题。
        answerable_response_rate=1.0,

        # 1道完整证据回答 / 1道可回答题。
        fully_grounded_answer_rate=1.0,

        # 1道正确拒答 / 1道无答案题。
        unanswerable_abstention_rate=1.0,
    )


def make_valid_report(
) -> CitationEvaluationReport:
    """创建包含q005和q020的合法评测报告。"""

    return CitationEvaluationReport(
        # datetime.now(timezone.utc)返回带UTC时区的当前时间。
        #
        # 带时区时间比普通datetime.now()更明确，
        # 不会让报告读取者猜测时间属于哪个时区。
        generated_at=datetime.now(
            timezone.utc
        ),

        api_url=(
            "http://127.0.0.1:8000"
            "/api/v1/knowledge/query"
        ),
        llm_model="test-llm",
        embedding_model="embedding-3",
        collection_name="robot_knowledge_v4",
        top_k=3,
        similarity_threshold=0.70,
        metrics=make_consistent_metrics(),

        # 汇总指标声明一共两道题，
        # 所以results中也必须有两项。
        results=[
            make_answered_evaluation(),
            make_abstained_evaluation_for_report(),
        ],
    )


def test_metrics_accept_consistent_counts_and_rates(
) -> None:
    """汇总计数与五项比例一致时应通过校验。"""

    metrics = make_consistent_metrics()

    assert metrics.total_question_count == 2

    assert (
        metrics.answerable_question_count
        == 1
    )
    assert (
        metrics.unanswerable_question_count
        == 1
    )

    assert (
        metrics.citation_correctness
        == 1.0
    )
    assert metrics.citation_coverage == 1.0

    assert (
        metrics.answerable_response_rate
        == 1.0
    )
    assert (
        metrics.fully_grounded_answer_rate
        == 1.0
    )
    assert (
        metrics.unanswerable_abstention_rate
        == 1.0
    )


def test_metrics_reject_wrong_citation_correctness(
) -> None:
    """全局引用正确率必须由全局计数计算。"""

    metrics_data = (
        make_consistent_metrics().model_dump()
    )

    # 当前计数是：
    #
    # 1条正确引用 / 1条返回引用 = 1.0。
    #
    # 这里故意把结果改成0.5。
    metrics_data[
        "citation_correctness"
    ] = 0.5

    with pytest.raises(
        ValidationError,
        match="citation_correctness",
    ):
        CitationMetrics.model_validate(
            metrics_data
        )


def test_metrics_reject_inconsistent_question_total(
) -> None:
    """问题总数必须等于可回答数加无答案数。"""

    metrics_data = (
        make_consistent_metrics().model_dump()
    )

    # 可回答数1 + 无答案数1 = 总数2。
    #
    # 这里故意把总数改成3。
    metrics_data[
        "total_question_count"
    ] = 3

    with pytest.raises(
        ValidationError,
        match="问题总数",
    ):
        CitationMetrics.model_validate(
            metrics_data
        )


def test_report_accepts_matching_result_count(
) -> None:
    """逐题结果数与汇总问题数一致时报告应合法。"""

    report = make_valid_report()

    assert report.metrics.total_question_count == 2
    assert len(report.results) == 2

    assert (
        report.results[0]
        .gold_question
        .question_id
        == "q005"
    )

    assert (
        report.results[1]
        .gold_question
        .question_id
        == "q020"
    )

    # datetime对象带有时区时，
    # tzinfo不会是None。
    assert report.generated_at.tzinfo is not None


def test_report_rejects_result_count_mismatch(
) -> None:
    """报告不能用两道题的指标配一条逐题结果。"""

    report_data = (
        make_valid_report().model_dump()
    )

    # metrics.total_question_count仍然是2，
    # 但这里故意只保留q005一项结果。
    report_data["results"] = [
        report_data["results"][0]
    ]

    with pytest.raises(
        ValidationError,
        match="逐题结果数量",
    ):
        CitationEvaluationReport.model_validate(
            report_data
        )