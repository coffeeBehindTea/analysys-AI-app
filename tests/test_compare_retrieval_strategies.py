"""三策略对比CLI中读取、写入和命令行部分的离线测试。

本模块只使用pytest提供的临时目录，不读取或覆盖
docs目录中的真实策略报告。

测试范围：
1. CLI默认路径与显式路径解析；
2. CandidateStrategyReport JSON反序列化；
3. 参数位置与报告内部策略名称一致性；
4. 三份报告的完整读取、比较和Markdown写入流程；
5. 终端摘要中候选胜者与安全边界的显示。
"""

from pathlib import Path
import sys

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
)
from app.services.retrieval_strategy_evaluation import (
    build_candidate_strategy_report,
    evaluate_candidate_question,
)
from scripts.compare_retrieval_strategies import (
    DEFAULT_HYBRID_REPORT,
    DEFAULT_OUTPUT,
    DEFAULT_REWRITE_REPORT,
    DEFAULT_VECTOR_REPORT,
    load_strategy_report,
    parse_args,
    print_summary,
    run_comparison,
    write_comparison_report,
)


# 测试Chunk使用稳定的64位十六进制哈希。
TEST_CONTENT_HASH = "f" * 64


@pytest.fixture
def project_tmp_path(
    request: pytest.FixtureRequest,
) -> Path:
    """在已忽略的项目tmp目录中创建测试专属路径。

    不使用pytest内置tmp_path，避免Windows系统临时目录
    被其他安全主体创建后产生ACL权限冲突。
    """

    # request.node.name是当前测试函数的稳定名称。
    # 每项测试使用不同子目录，不会互相覆盖。
    temporary_directory = (
        Path("tmp")
        / "pytest_compare_retrieval_strategies"
        / request.node.name
    )

    # tmp/已经由.gitignore忽略，
    # 重复执行时exist_ok=True允许复用该目录。
    temporary_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return temporary_directory


def make_chunk(
    *,
    chunk_index: int,
    source_file: str,
    page_or_section: str,
) -> DocumentChunk:
    """构造一条CLI测试使用的文档Chunk。"""

    return DocumentChunk(
        chunk_id=(
            f"cli-document:"
            f"{chunk_index:06d}"
        ),
        document_id="cli-document",
        source_file=source_file,
        page_or_section=page_or_section,
        chunk_index=chunk_index,
        content_hash=TEST_CONTENT_HASH,
        content=f"CLI测试正文 {chunk_index}",
    )


def make_questions() -> list[GoldQuestion]:
    """构造一条可回答题和一条无答案题。"""

    return [
        GoldQuestion(
            question_id="q001",
            question="正确证据位于哪里？",
            question_type="single_hop",
            answerable=True,
            reference_answer="位于manual.pdf第1页。",
            expected_evidence=[
                ExpectedEvidence(
                    source_file="manual.pdf",
                    page_or_section="page: 1",
                )
            ],
            tags=["cli-test"],
            notes="测试CLI报告读取",
        ),
        GoldQuestion(
            question_id="q002",
            question="不存在的生产设备序列号是什么？",
            question_type="unanswerable",
            answerable=False,
            reference_answer=None,
            expected_evidence=[],
            tags=["unanswerable"],
            notes="测试无答案题数量被保留",
        ),
    ]


def make_parameters(
    strategy: RetrievalStrategyName,
) -> CandidateStrategyParameters:
    """构造一种候选策略的合法参数。"""

    if strategy == "vector_baseline":
        return CandidateStrategyParameters(
            strategy=strategy,
            top_k=3,
        )

    return CandidateStrategyParameters(
        strategy=strategy,
        top_k=3,
        candidate_k=20,
        rank_constant=60,
        rewrite_version=(
            "deterministic-v1"
            if strategy == "hybrid_rrf_rewrite"
            else None
        ),
    )


def make_retrieved_chunk(
    *,
    strategy: RetrievalStrategyName,
    chunk: DocumentChunk,
) -> RetrievedChunk | HybridRetrievedChunk:
    """构造与报告策略相匹配的候选结果类型。"""

    if strategy == "vector_baseline":
        return RetrievedChunk(
            chunk=chunk,
            similarity=0.75,
            rank=1,
        )

    return HybridRetrievedChunk(
        chunk=chunk,
        rrf_score=0.032,
        rank=1,
        vector_rank=1,
        vector_similarity=0.75,
        keyword_rank=1,
        keyword_score=8.0,

        # 关键词路径存在时必须保留命中解释。
        matched_lexical_terms=("测试",),
    )


def make_report(
    *,
    strategy: RetrievalStrategyName,
    answerable_hit: bool,
) -> CandidateStrategyReport:
    """构造能够序列化和重新校验的候选报告。"""

    questions = make_questions()

    correct_chunk = make_chunk(
        chunk_index=1,
        source_file="manual.pdf",
        page_or_section="page: 1",
    )
    distractor_chunk = make_chunk(
        chunk_index=2,
        source_file="distractor.pdf",
        page_or_section="page: 2",
    )

    answerable_candidate = (
        correct_chunk
        if answerable_hit
        else distractor_chunk
    )

    evaluations = [
        evaluate_candidate_question(
            question=questions[0],
            retrieved_chunks=[
                make_retrieved_chunk(
                    strategy=strategy,
                    chunk=answerable_candidate,
                )
            ],
        ),
        evaluate_candidate_question(
            question=questions[1],
            retrieved_chunks=[
                make_retrieved_chunk(
                    strategy=strategy,
                    chunk=distractor_chunk,
                )
            ],
        ),
    ]

    return build_candidate_strategy_report(
        embedding_model="embedding-3",
        collection_name="robot_knowledge_cli_test",
        parameters=make_parameters(strategy),
        evaluations=evaluations,
    )


def make_three_reports(
) -> list[CandidateStrategyReport]:
    """构造改写策略优于另外两种策略的报告集合。"""

    return [
        make_report(
            strategy="vector_baseline",
            answerable_hit=False,
        ),
        make_report(
            strategy="hybrid_rrf",
            answerable_hit=False,
        ),
        make_report(
            strategy="hybrid_rrf_rewrite",
            answerable_hit=True,
        ),
    ]


def write_json_report(
    *,
    report: CandidateStrategyReport,
    path: Path,
) -> None:
    """把测试报告写入pytest临时目录。"""

    path.write_text(
        report.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )


def test_parse_args_uses_documented_default_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """无额外参数时应使用三份标准报告和标准输出路径。"""

    monkeypatch.setattr(
        sys,
        "argv",
        ["compare_retrieval_strategies.py"],
    )

    args = parse_args()

    assert args.vector_report == DEFAULT_VECTOR_REPORT
    assert args.hybrid_report == DEFAULT_HYBRID_REPORT
    assert args.rewrite_report == DEFAULT_REWRITE_REPORT
    assert args.output == DEFAULT_OUTPUT


def test_parse_args_preserves_explicit_paths(
    monkeypatch: pytest.MonkeyPatch,
    project_tmp_path: Path,
) -> None:
    """用户显式给出的文件路径不能被默认值覆盖。"""

    vector_path = project_tmp_path / "vector.json"
    hybrid_path = project_tmp_path / "hybrid.json"
    rewrite_path = project_tmp_path / "rewrite.json"
    output_path = project_tmp_path / "comparison.md"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_retrieval_strategies.py",
            "--vector-report",
            str(vector_path),
            "--hybrid-report",
            str(hybrid_path),
            "--rewrite-report",
            str(rewrite_path),
            "--output",
            str(output_path),
        ],
    )

    args = parse_args()

    assert args.vector_report == vector_path
    assert args.hybrid_report == hybrid_path
    assert args.rewrite_report == rewrite_path
    assert args.output == output_path


def test_load_strategy_report_round_trips_valid_json(
    project_tmp_path: Path,
) -> None:
    """合法JSON应重新构造成完整CandidateStrategyReport。"""

    expected_report = make_three_reports()[0]
    report_path = project_tmp_path / "vector.json"

    write_json_report(
        report=expected_report,
        path=report_path,
    )

    actual_report = load_strategy_report(
        path=report_path,
        expected_strategy="vector_baseline",
    )

    assert actual_report == expected_report
    assert isinstance(
        actual_report,
        CandidateStrategyReport,
    )


def test_load_strategy_report_rejects_wrong_argument_position(
    project_tmp_path: Path,
) -> None:
    """混合报告不能冒充--vector-report对应的基线报告。"""

    hybrid_report = make_three_reports()[1]
    report_path = (
        project_tmp_path
        / "wrong-position.json"
    )

    write_json_report(
        report=hybrid_report,
        path=report_path,
    )

    with pytest.raises(
        ValueError,
        match="报告策略与参数位置不一致",
    ):
        load_strategy_report(
            path=report_path,
            expected_strategy="vector_baseline",
        )


def test_load_strategy_report_requires_path_object(
) -> None:
    """公共读取函数应拒绝没有Path方法的字符串对象。"""

    with pytest.raises(
        TypeError,
        match="path必须是Path",
    ):
        load_strategy_report(
            path="report.json",  # type: ignore[arg-type]
            expected_strategy="vector_baseline",
        )


def test_write_comparison_report_creates_parent_directory(
    project_tmp_path: Path,
) -> None:
    """写入函数应创建父目录并保存UTF-8 Markdown。"""

    comparison = compare_candidate_strategy_reports(
        make_three_reports()
    )
    output_path = (
        project_tmp_path
        / "nested"
        / "comparison.md"
    )

    write_comparison_report(
        comparison=comparison,
        output_path=output_path,
    )

    assert output_path.is_file()

    markdown = output_path.read_text(
        encoding="utf-8"
    )

    assert "# Week 3 候选检索策略对比" in markdown
    assert "混合 RRF + 改写" in markdown
    assert "无答案错误召回率尚未计算" in markdown


def test_run_comparison_executes_complete_local_flow(
    project_tmp_path: Path,
) -> None:
    """CLI业务入口应读取三份JSON、比较并写出Markdown。"""

    reports = make_three_reports()
    vector_path = project_tmp_path / "vector.json"
    hybrid_path = project_tmp_path / "hybrid.json"
    rewrite_path = project_tmp_path / "rewrite.json"
    output_path = project_tmp_path / "result.md"

    for report, path in zip(
        reports,
        (
            vector_path,
            hybrid_path,
            rewrite_path,
        ),
        strict=True,
    ):
        write_json_report(
            report=report,
            path=path,
        )

    comparison = run_comparison(
        vector_report_path=vector_path,
        hybrid_report_path=hybrid_path,
        rewrite_report_path=rewrite_path,
        output_path=output_path,
    )

    assert isinstance(
        comparison,
        CandidateStrategyComparison,
    )
    assert comparison.candidate_winner == (
        "hybrid_rrf_rewrite"
    )
    assert output_path.is_file()


def test_print_summary_reports_winner_and_pending_safety(
    capsys: pytest.CaptureFixture[str],
    project_tmp_path: Path,
) -> None:
    """终端摘要必须区分候选胜者和生产接入结论。"""

    comparison = compare_candidate_strategy_reports(
        make_three_reports()
    )
    output_path = project_tmp_path / "comparison.md"

    print_summary(
        comparison=comparison,
        output_path=output_path,
    )

    output = capsys.readouterr().out

    assert "候选检索策略对比完成" in output
    assert "候选层胜者：hybrid_rrf_rewrite" in output
    assert "无答案安全评测：尚未完成" in output
    assert str(output_path.resolve()) in output
