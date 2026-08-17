"""最终RAG回答的引用匹配、单题评分和汇总计算。"""

from app.schemas.citation_evaluation import (
    CitationMetrics,
    CitationQuestionEvaluation,
)

# ExpectedEvidence和GoldQuestion来自人工标注数据。
from app.schemas.evaluation import (
    ExpectedEvidence,
    GoldQuestion,
)

# KnowledgeCitation和KnowledgeQueryResponse
# 来自真实知识库问答结果。
from app.schemas.knowledge_query import (
    KnowledgeCitation,
    KnowledgeQueryResponse,
)

# 复用已经过测试的位置匹配逻辑。
#
# 它能处理：
# - PDF的物理页码；
# - Markdown和TXT的章节路径；
# - 大小写和多余空白；
# - Gold位置中的辅助section说明。
from app.services.retrieval_evaluation import (
    location_matches,
)


def citation_matches(
    citation: KnowledgeCitation,
    expected: ExpectedEvidence,
) -> bool:
    """判断一条API引用是否匹配一项Gold预期证据。"""

    # 文件名必须相同。
    #
    # casefold()执行适合无视大小写比较的标准化。
    # 它比lower()更适合处理Unicode文本。
    if (
        citation.source_file.casefold()
        != expected.source_file.casefold()
    ):
        return False

    # 文件名相同后，继续比较页码或章节。
    return location_matches(
        actual_location=(
            citation.page_or_section
        ),
        expected_location=(
            expected.page_or_section
        ),
    )


def count_correct_citations(
    citations: list[KnowledgeCitation],
    expected_evidence: list[ExpectedEvidence],
) -> int:
    """统计返回引用中有多少条属于正确引用。"""

    # 这里从“返回引用”的角度统计。
    #
    # 每条citation最多计数一次：
    # 只要它能匹配任意一项expected_evidence，
    # 就属于正确引用。
    return sum(
        1
        for citation in citations
        if any(
            citation_matches(
                citation,
                expected,
            )
            for expected in expected_evidence
        )
    )


def count_matched_expected_evidence(
    citations: list[KnowledgeCitation],
    expected_evidence: list[ExpectedEvidence],
) -> int:
    """统计有多少项不同Gold证据被返回引用覆盖。"""

    # 这里从“Gold预期证据”的角度统计。
    #
    # 即使两个重叠Chunk都来自同一个正确页码，
    # 该expected也只能被计数一次。
    return sum(
        1
        for expected in expected_evidence
        if any(
            citation_matches(
                citation,
                expected,
            )
            for citation in citations
        )
    )


def evaluate_citation_question(
    *,
    question: GoldQuestion,
    response: KnowledgeQueryResponse,
) -> CitationQuestionEvaluation:
    """评价一道Gold Question的最终回答和引用。"""

    # 创建列表副本，避免后续逻辑意外修改
    # Pydantic模型内部保存的原始列表。
    citations = list(
        response.citations
    )
    expected_evidence = list(
        question.expected_evidence
    )

    returned_citation_count = len(
        citations
    )
    expected_evidence_count = len(
        expected_evidence
    )

    correct_citation_count = (
        count_correct_citations(
            citations,
            expected_evidence,
        )
    )

    matched_expected_evidence_count = (
        count_matched_expected_evidence(
            citations,
            expected_evidence,
        )
    )

    # 没有返回引用时，分母为0。
    #
    # 单题级Schema使用None表示“无法计算”，
    # 而不是错误地把它解释成0%或100%。
    citation_correctness = (
        correct_citation_count
        / returned_citation_count
        if returned_citation_count
        else None
    )

    # 无答案问题没有预期证据，
    # 所以单题引用覆盖率不适用。
    citation_coverage = (
        matched_expected_evidence_count
        / expected_evidence_count
        if expected_evidence_count
        else None
    )

    # 完整证据回答必须同时满足：
    #
    # 1. Gold认为问题可以回答；
    # 2. API没有拒答；
    # 3. 至少返回一条引用；
    # 4. 所有返回引用都正确；
    # 5. 找全全部预期证据。
    fully_grounded = (
        question.answerable
        and not response.abstained
        and returned_citation_count > 0
        and (
            correct_citation_count
            == returned_citation_count
        )
        and (
            matched_expected_evidence_count
            == expected_evidence_count
        )
    )

    # 正确拒答必须同时满足：
    #
    # 1. Gold认为问题无答案；
    # 2. API明确返回abstained=true；
    # 3. 引用列表为空。
    correctly_abstained = (
        not question.answerable
        and response.abstained
        and returned_citation_count == 0
    )

    # CitationQuestionEvaluation会再次校验
    # 上述计数、比例和布尔状态是否一致。
    return CitationQuestionEvaluation(
        gold_question=question,
        response=response,
        returned_citation_count=(
            returned_citation_count
        ),
        correct_citation_count=(
            correct_citation_count
        ),
        expected_evidence_count=(
            expected_evidence_count
        ),
        matched_expected_evidence_count=(
            matched_expected_evidence_count
        ),
        citation_correctness=(
            citation_correctness
        ),
        citation_coverage=citation_coverage,
        fully_grounded=fully_grounded,
        correctly_abstained=(
            correctly_abstained
        ),
    )


def calculate_citation_metrics(
    results: list[CitationQuestionEvaluation],
) -> CitationMetrics:
    """根据全部逐题结果计算引用质量汇总指标。"""

    if not results:
        raise ValueError(
            "引用评测结果列表不能为空"
        )

    # 防止同一道题被重复计入分母。
    question_ids = [
        result.gold_question.question_id
        for result in results
    ]

    # set会删除重复值。
    #
    # 如果set长度变小，就表示原列表中存在重复ID。
    if len(set(question_ids)) != len(
        question_ids
    ):
        raise ValueError(
            "引用评测结果包含重复question_id"
        )

    answerable_results = [
        result
        for result in results
        if result.gold_question.answerable
    ]

    unanswerable_results = [
        result
        for result in results
        if not result.gold_question.answerable
    ]

    # 可回答问题中，API没有拒答的数量。
    answered_answerable_count = sum(
        1
        for result in answerable_results
        if not result.response.abstained
    )

    fully_grounded_answer_count = sum(
        1
        for result in answerable_results
        if result.fully_grounded
    )

    correctly_abstained_count = sum(
        1
        for result in unanswerable_results
        if result.correctly_abstained
    )

    returned_citation_count = sum(
        result.returned_citation_count
        for result in results
    )

    correct_citation_count = sum(
        result.correct_citation_count
        for result in results
    )

    expected_evidence_count = sum(
        result.expected_evidence_count
        for result in answerable_results
    )

    matched_expected_evidence_count = sum(
        result.matched_expected_evidence_count
        for result in answerable_results
    )

    # 全局指标没有使用None。
    #
    # 如果没有任何返回引用，
    # 引用正确率统一记为0.0，
    # 并结合可回答率一起解释。
    citation_correctness = (
        correct_citation_count
        / returned_citation_count
        if returned_citation_count
        else 0.0
    )

    citation_coverage = (
        matched_expected_evidence_count
        / expected_evidence_count
        if expected_evidence_count
        else 0.0
    )

    answerable_count = len(
        answerable_results
    )
    unanswerable_count = len(
        unanswerable_results
    )

    answerable_response_rate = (
        answered_answerable_count
        / answerable_count
        if answerable_count
        else 0.0
    )

    fully_grounded_answer_rate = (
        fully_grounded_answer_count
        / answerable_count
        if answerable_count
        else 0.0
    )

    unanswerable_abstention_rate = (
        correctly_abstained_count
        / unanswerable_count
        if unanswerable_count
        else 0.0
    )

    # CitationMetrics会再次验证：
    #
    # - 总数是否一致；
    # - 正确数是否超过总数；
    # - 五个比例是否和计数相符。
    return CitationMetrics(
        total_question_count=len(results),
        answerable_question_count=(
            answerable_count
        ),
        unanswerable_question_count=(
            unanswerable_count
        ),
        answered_answerable_count=(
            answered_answerable_count
        ),
        fully_grounded_answer_count=(
            fully_grounded_answer_count
        ),
        correctly_abstained_unanswerable_count=(
            correctly_abstained_count
        ),
        returned_citation_count=(
            returned_citation_count
        ),
        correct_citation_count=(
            correct_citation_count
        ),
        expected_evidence_count=(
            expected_evidence_count
        ),
        matched_expected_evidence_count=(
            matched_expected_evidence_count
        ),
        citation_correctness=(
            citation_correctness
        ),
        citation_coverage=citation_coverage,
        answerable_response_rate=(
            answerable_response_rate
        ),
        fully_grounded_answer_rate=(
            fully_grounded_answer_rate
        ),
        unanswerable_abstention_rate=(
            unanswerable_abstention_rate
        ),
    )