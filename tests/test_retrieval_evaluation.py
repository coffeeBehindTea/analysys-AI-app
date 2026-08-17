"""检索证据匹配与指标计算的离线测试。"""

from app.schemas.evaluation import (
    ExpectedEvidence,
    GoldQuestion,
)
from app.schemas.retrieval import (
    DocumentChunk,
    RetrievedChunk,
)
from app.services.retrieval_evaluation import (
    calculate_metrics,
    evidence_matches,
    evaluate_question,
)


def make_result(
    *,
    source_file: str,
    page_or_section: str,
    rank: int,
    similarity: float = 0.80,
) -> RetrievedChunk:
    """创建不依赖 ChromaDB 的假检索结果。"""

    return RetrievedChunk(
        chunk=DocumentChunk(
            chunk_id=f"document-{rank}:000000",
            document_id=f"document-{rank}",
            source_file=source_file,
            page_or_section=page_or_section,
            chunk_index=rank - 1,
            content_hash="a" * 64,
            content="用于测试的 Chunk 正文。",
        ),
        similarity=similarity,
        rank=rank,
    )


def test_pdf_evidence_matches_physical_page() -> None:
    """PDF 标注带章节说明时，仍应按物理页码命中。"""

    result = make_result(
        source_file="manual.pdf",
        page_or_section="page: 55",
        rank=1,
    )
    expected = ExpectedEvidence(
        source_file="manual.pdf",
        page_or_section=(
            "page: 55; section: Cleaning and Maintenance"
        ),
    )

    assert evidence_matches(result, expected) is True


def test_markdown_evidence_matches_heading_path() -> None:
    """Markdown 实际路径应包含标注中的全部章节片段。"""

    result = make_result(
        source_file="test-standard.md",
        page_or_section=(
            "section: 仓储测试 > "
            "10.1 TEST-NET-001 调度心跳中断 > "
            "通过标准"
        ),
        rank=1,
    )
    expected = ExpectedEvidence(
        source_file="test-standard.md",
        page_or_section=(
            "section: 10.1 TEST-NET-001 "
            "调度心跳中断/通过标准"
        ),
    )

    assert evidence_matches(result, expected) is True


def test_evidence_requires_same_source_file() -> None:
    """页码相同但文件错误时不能判定为命中。"""

    result = make_result(
        source_file="wrong.pdf",
        page_or_section="page: 55",
        rank=1,
    )
    expected = ExpectedEvidence(
        source_file="manual.pdf",
        page_or_section="page: 55",
    )

    assert evidence_matches(result, expected) is False


def test_multi_hop_question_is_fully_recalled_at_three() -> None:
    """Top-3 找到两项证据时，多跳问题应完整召回。"""

    question = GoldQuestion(
        question_id="q013",
        question="机器人应如何停止并恢复定位？",
        question_type="multi_hop",
        answerable=True,
        reference_answer="先安全停车，再进行重定位。",
        expected_evidence=[
            ExpectedEvidence(
                source_file="test-standard.md",
                page_or_section="section: 停车",
            ),
            ExpectedEvidence(
                source_file="fault.txt",
                page_or_section="section: 恢复",
            ),
        ],
        tags=["navigation"],
        notes="测试多跳召回。",
    )

    retrieved_chunks = [
        make_result(
            source_file="test-standard.md",
            page_or_section="section: 停车",
            rank=1,
            similarity=0.90,
        ),
        make_result(
            source_file="fault.txt",
            page_or_section="section: 恢复",
            rank=2,
            similarity=0.85,
        ),
    ]

    evaluation = evaluate_question(
        question=question,
        retrieved_chunks=retrieved_chunks,
        similarity_threshold=0.70,
    )

    assert evaluation.matched_evidence_at_1 == 1
    assert evaluation.matched_evidence_at_3 == 2
    assert evaluation.fully_recalled_at_1 is False
    assert evaluation.fully_recalled_at_3 is True
    assert evaluation.top_similarity == 0.90


def test_unanswerable_question_can_false_recall() -> None:
    """无答案问题超过阈值时应记录错误召回。"""

    question = GoldQuestion(
        question_id="q020",
        question="robot-001 当前在哪里？",
        question_type="unanswerable",
        answerable=False,
        reference_answer=None,
        expected_evidence=[],
        tags=["realtime"],
        notes="静态知识库没有实时状态。",
    )

    evaluation = evaluate_question(
        question=question,
        retrieved_chunks=[
            make_result(
                source_file="manual.md",
                page_or_section="section: robot",
                rank=1,
                similarity=0.75,
            )
        ],
        similarity_threshold=0.70,
    )

    assert evaluation.false_recall is True
    assert evaluation.matched_evidence_at_3 == 0
    assert evaluation.fully_recalled_at_3 is False


def test_calculate_metrics() -> None:
    """汇总指标应正确计算 Recall 和错误召回率。"""

    answerable_question = GoldQuestion(
        question_id="q013",
        question="需要哪两项证据？",
        question_type="multi_hop",
        answerable=True,
        reference_answer="需要证据 A 和证据 B。",
        expected_evidence=[
            ExpectedEvidence(
                source_file="a.md",
                page_or_section="section: A",
            ),
            ExpectedEvidence(
                source_file="b.md",
                page_or_section="section: B",
            ),
        ],
        tags=["multi-hop"],
        notes="测试两项预期证据。",
    )

    answerable_evaluation = evaluate_question(
        question=answerable_question,
        retrieved_chunks=[
            make_result(
                source_file="a.md",
                page_or_section="section: A",
                rank=1,
                similarity=0.90,
            ),
            make_result(
                source_file="b.md",
                page_or_section="section: B",
                rank=2,
                similarity=0.80,
            ),
        ],
        similarity_threshold=0.70,
    )

    unanswerable_question = GoldQuestion(
        question_id="q020",
        question="当前实时状态是什么？",
        question_type="unanswerable",
        answerable=False,
        reference_answer=None,
        expected_evidence=[],
        tags=["realtime"],
        notes="知识库没有实时状态。",
    )

    unanswerable_evaluation = evaluate_question(
        question=unanswerable_question,
        retrieved_chunks=[
            make_result(
                source_file="a.md",
                page_or_section="section: A",
                rank=1,
                similarity=0.75,
            )
        ],
        similarity_threshold=0.70,
    )

    metrics = calculate_metrics(
        [
            answerable_evaluation,
            unanswerable_evaluation,
        ]
    )

    # 可回答问题有两项预期证据：
    # Top-1 命中一项，所以 Recall@1 = 1/2；
    # Top-3 命中两项，所以 Recall@3 = 2/2。
    assert metrics.recall_at_1 == 0.5
    assert metrics.recall_at_3 == 1.0

    assert metrics.fully_recalled_at_1_count == 0
    assert metrics.fully_recalled_at_3_count == 1

    # 唯一一道无答案问题超过阈值：
    # false_recall_rate = 1/1。
    assert metrics.false_recall_count == 1
    assert metrics.false_recall_rate == 1.0