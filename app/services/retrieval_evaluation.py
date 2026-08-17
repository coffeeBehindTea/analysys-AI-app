"""检索结果的证据匹配、单题评分和汇总指标计算。"""

import re

from app.schemas.evaluation import (
    ExpectedEvidence,
    GoldQuestion,
    RetrievalMetrics,
    RetrievalQuestionEvaluation,
)
from app.schemas.retrieval import RetrievedChunk


# 匹配类似下面的位置：
#
# page: 55
# Page : 55
# page: 55; section: Cleaning
#
# re.IGNORECASE 表示不区分 page 的大小写。
PAGE_LOCATION_PATTERN = re.compile(
    r"\bpage\s*:\s*(\d+)\b",
    re.IGNORECASE,
)


def normalize_location(location: str) -> str:
    """统一页码或章节字符串的大小写与空白。"""

    # casefold() 执行适合无视大小写比较的标准化。
    normalized = location.casefold()

    # split() 按连续空白切分；
    # " ".join(...) 再用单个空格重新连接。
    #
    # 例如：
    # "section:   Safety\nCheck"
    # 变成：
    # "section: safety check"
    return " ".join(normalized.split())


def location_matches(
    *,
    actual_location: str,
    expected_location: str,
) -> bool:
    """判断实际位置是否符合人工标注的预期位置。"""

    # search() 在整个字符串中查找页码模式。
    actual_page_match = PAGE_LOCATION_PATTERN.search(
        actual_location
    )
    expected_page_match = PAGE_LOCATION_PATTERN.search(
        expected_location
    )

    # PDF Loader 当前只保存物理页码，例如 page: 55。
    #
    # 人工标注可能写成：
    # page: 55; section: Cleaning and Maintenance
    #
    # 其中 section 是方便人阅读的补充说明，
    # 数据库中真正稳定保存的位置是 page: 55，
    # 因此 PDF 使用物理页码进行判定。
    if expected_page_match is not None:
        if actual_page_match is None:
            return False

        # group(1) 取得正则中第一个括号捕获的数字。
        #
        # 转成 int 后，page: 055 和 page: 55
        # 也会被视为相同页码。
        return (
            int(actual_page_match.group(1))
            == int(expected_page_match.group(1))
        )

    actual_normalized = normalize_location(
        actual_location
    )

    # removeprefix() 只在字符串确实以指定内容开头时删除。
    #
    # 数据库中的 Markdown 位置通常是：
    # section: 一级标题 > 二级标题 > 三级标题
    actual_normalized = (
        actual_normalized
        .removeprefix("section:")
        .strip()
    )

    # 标注中的 "/" 表示同一个章节路径中
    # 必须同时出现这些部分。
    #
    # 例如：
    # section: 10.1 TEST-NET-001/通过标准
    expected_parts = [
        normalize_location(part)
        .removeprefix("section:")
        .strip()
        for part in expected_location.split("/")
        if part.strip()
    ]

    if not expected_parts:
        return False

    # all(...) 要求每一个预期章节片段
    # 都出现在实际章节路径中。
    return all(
        part in actual_normalized
        for part in expected_parts
    )


def evidence_matches(
    result: RetrievedChunk,
    expected: ExpectedEvidence,
) -> bool:
    """判断一条检索结果是否命中一项预期证据。"""

    # 文件名必须相同。
    #
    # 不能因为两个文件都有 page: 4，
    # 就把错误文件的第四页判定为正确证据。
    if (
        result.chunk.source_file.casefold()
        != expected.source_file.casefold()
    ):
        return False

    return location_matches(
        actual_location=(
            result.chunk.page_or_section
        ),
        expected_location=(
            expected.page_or_section
        ),
    )


def count_matched_evidence(
    results: list[RetrievedChunk],
    expected_evidence: list[ExpectedEvidence],
) -> int:
    """统计一组结果命中了多少项不同的预期证据。"""

    # 对每项 expected_evidence 最多计数一次。
    #
    # 即使两个重叠 Chunk 都来自同一个正确章节，
    # 也只能算命中了一项预期证据。
    return sum(
        1
        for expected in expected_evidence
        if any(
            evidence_matches(result, expected)
            for result in results
        )
    )


def evaluate_question(
    *,
    question: GoldQuestion,
    retrieved_chunks: list[RetrievedChunk],
    similarity_threshold: float,
) -> RetrievalQuestionEvaluation:
    """根据 Top-K 结果评价一道问题。"""

    if not -1.0 <= similarity_threshold <= 1.0:
        raise ValueError(
            "similarity_threshold 必须在 -1 到 1 之间"
        )

    # VectorStore 正常情况下已经按排名返回。
    #
    # 这里再次按 rank 排序，使评分函数不依赖
    # 调用方传入列表时的偶然顺序。
    ordered_chunks = sorted(
        retrieved_chunks,
        key=lambda result: result.rank,
    )

    # 切片不会在结果不足时抛出异常。
    #
    # 一个结果时：
    # ordered_chunks[:3]
    # 仍然只返回这一项。
    top_1 = ordered_chunks[:1]
    top_3 = ordered_chunks[:3]

    top_similarity = (
        ordered_chunks[0].similarity
        if ordered_chunks
        else None
    )

    if question.answerable:
        matched_at_1 = count_matched_evidence(
            top_1,
            question.expected_evidence,
        )
        matched_at_3 = count_matched_evidence(
            top_3,
            question.expected_evidence,
        )

        expected_count = len(
            question.expected_evidence
        )

        fully_recalled_at_1 = (
            matched_at_1 == expected_count
        )
        fully_recalled_at_3 = (
            matched_at_3 == expected_count
        )

        # 可回答问题不属于“无答案错误召回”的统计对象。
        false_recall = False

    else:
        # 无答案问题没有 expected_evidence，
        # 因此不能用 0 == 0 判断“完整召回”。
        matched_at_1 = 0
        matched_at_3 = 0
        fully_recalled_at_1 = False
        fully_recalled_at_3 = False

        # 向量数据库即使面对无答案问题，
        # 通常也会返回最相似的若干结果。
        #
        # 只有最高相似度达到接受阈值，
        # 才认为系统可能错误接受了无关证据。
        false_recall = (
            top_similarity is not None
            and top_similarity
            >= similarity_threshold
        )

    return RetrievalQuestionEvaluation(
        question_id=question.question_id,
        question=question.question,
        question_type=question.question_type,
        answerable=question.answerable,
        expected_evidence=(
            question.expected_evidence
        ),
        retrieved_chunks=ordered_chunks,
        matched_evidence_at_1=matched_at_1,
        matched_evidence_at_3=matched_at_3,
        fully_recalled_at_1=(
            fully_recalled_at_1
        ),
        fully_recalled_at_3=(
            fully_recalled_at_3
        ),
        top_similarity=top_similarity,
        false_recall=false_recall,
    )


def calculate_metrics(
    results: list[RetrievalQuestionEvaluation],
) -> RetrievalMetrics:
    """根据全部逐题结果计算检索汇总指标。"""

    if not results:
        raise ValueError(
            "评测结果列表不能为空"
        )

    answerable_results = [
        result
        for result in results
        if result.answerable
    ]
    unanswerable_results = [
        result
        for result in results
        if not result.answerable
    ]

    expected_evidence_count = sum(
        len(result.expected_evidence)
        for result in answerable_results
    )

    matched_evidence_at_1 = sum(
        result.matched_evidence_at_1
        for result in answerable_results
    )
    matched_evidence_at_3 = sum(
        result.matched_evidence_at_3
        for result in answerable_results
    )

    recall_at_1 = (
        matched_evidence_at_1
        / expected_evidence_count
        if expected_evidence_count
        else 0.0
    )
    recall_at_3 = (
        matched_evidence_at_3
        / expected_evidence_count
        if expected_evidence_count
        else 0.0
    )

    fully_recalled_at_1_count = sum(
        1
        for result in answerable_results
        if result.fully_recalled_at_1
    )
    fully_recalled_at_3_count = sum(
        1
        for result in answerable_results
        if result.fully_recalled_at_3
    )

    false_recall_count = sum(
        1
        for result in unanswerable_results
        if result.false_recall
    )

    false_recall_rate = (
        false_recall_count
        / len(unanswerable_results)
        if unanswerable_results
        else 0.0
    )

    return RetrievalMetrics(
        answerable_question_count=len(
            answerable_results
        ),
        unanswerable_question_count=len(
            unanswerable_results
        ),
        expected_evidence_count=(
            expected_evidence_count
        ),
        matched_evidence_at_1=(
            matched_evidence_at_1
        ),
        matched_evidence_at_3=(
            matched_evidence_at_3
        ),
        recall_at_1=recall_at_1,
        recall_at_3=recall_at_3,
        fully_recalled_at_1_count=(
            fully_recalled_at_1_count
        ),
        fully_recalled_at_3_count=(
            fully_recalled_at_3_count
        ),
        false_recall_count=false_recall_count,
        false_recall_rate=false_recall_rate,
    )