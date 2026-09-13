"""真实检索基线脚本中纯函数部分的离线测试。"""

from hashlib import sha256

from app.schemas.evaluation import (
    ExpectedEvidence,
    GoldQuestion,
)
from app.schemas.retrieval import (
    DocumentChunk,
    RetrievedChunk,
)
from scripts.run_retrieval_baseline import (
    count_matched_evidence,
    evidence_matches,
    normalize_location,
    select_supported_questions,
)


def make_question(
    *,
    question_id: str,
    source_file: str,
    answerable: bool = True,
) -> GoldQuestion:
    """创建测试问题。"""

    if answerable:
        return GoldQuestion(
            question_id=question_id,
            question="测试问题",
            question_type="single_hop",
            answerable=True,
            reference_answer="测试答案",
            expected_evidence=[
                ExpectedEvidence(
                    source_file=source_file,
                    page_or_section=(
                        "section: 10.1 网络中断/通过标准"
                    ),
                )
            ],
            tags=["test"],
            notes="测试说明",
        )

    return GoldQuestion(
        question_id=question_id,
        question="当前实时状态是什么？",
        question_type="unanswerable",
        answerable=False,
        reference_answer=None,
        expected_evidence=[],
        tags=["test"],
        notes="知识库没有实时状态。",
    )


def make_result(
    *,
    source_file: str,
    page_or_section: str,
    rank: int = 1,
) -> RetrievedChunk:
    """创建测试检索结果。"""

    content = "网络中断后的通过标准"

    return RetrievedChunk(
        chunk=DocumentChunk(
            chunk_id=f"chunk-{rank}",
            document_id="document-001",
            source_file=source_file,
            page_or_section=page_or_section,
            chunk_index=rank - 1,
            content_hash=sha256(
                content.encode("utf-8")
            ).hexdigest(),
            content=content,
        ),
        similarity=0.9,
        rank=rank,
    )


def test_normalize_location_removes_prefix_and_spaces() -> None:
    """位置归一化应清除固定前缀和多余空白。"""

    assert normalize_location(
        "  section:  10.1   网络中断  "
    ) == "10.1 网络中断"


def test_evidence_matches_hierarchical_location() -> None:
    """父章节和子章节都存在时应判定命中。"""

    result = make_result(
        source_file="manual.md",
        page_or_section=(
            "section: 测试规程"
            " > 10. 网络测试"
            " > 10.1 网络中断"
            " > 通过标准"
        ),
    )

    expected = ExpectedEvidence(
        source_file="manual.md",
        page_or_section=(
            "section: 10.1 网络中断/通过标准"
        ),
    )

    assert evidence_matches(
        result,
        expected,
    ) is True


def test_evidence_requires_matching_source_file() -> None:
    """章节相同但文件不同不能算证据命中。"""

    result = make_result(
        source_file="wrong.md",
        page_or_section=(
            "section: 10.1 网络中断 > 通过标准"
        ),
    )

    expected = ExpectedEvidence(
        source_file="manual.md",
        page_or_section=(
            "section: 10.1 网络中断/通过标准"
        ),
    )

    assert evidence_matches(
        result,
        expected,
    ) is False


def test_duplicate_results_count_one_expected_evidence() -> None:
    """两个重叠 Chunk 命中同一证据时只能计数一次。"""

    results = [
        make_result(
            source_file="manual.md",
            page_or_section=(
                "section: 10.1 网络中断"
                " > 通过标准"
            ),
            rank=1,
        ),
        make_result(
            source_file="manual.md",
            page_or_section=(
                "section: 10.1 网络中断"
                " > 通过标准"
            ),
            rank=2,
        ),
    ]

    expected = [
        ExpectedEvidence(
            source_file="manual.md",
            page_or_section=(
                "section: 10.1 网络中断/通过标准"
            ),
        )
    ]

    assert count_matched_evidence(
        results,
        expected,
    ) == 1


def test_select_supported_questions() -> None:
    """只选择当前语料可以完整评测的可回答问题。"""

    supported = make_question(
        question_id="q001",
        source_file="manual.md",
    )

    pdf_question = make_question(
        question_id="q002",
        source_file="manual.pdf",
    )

    unanswerable = make_question(
        question_id="q003",
        source_file="unused",
        answerable=False,
    )

    selected = select_supported_questions(
        [
            supported,
            pdf_question,
            unanswerable,
        ],
        available_source_files={
            "manual.md",
            "faults.txt",
        },
    )

    assert [
        question.question_id
        for question in selected
    ] == ["q001"]