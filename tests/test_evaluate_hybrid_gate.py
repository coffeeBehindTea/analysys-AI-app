"""混合证据门控评测命令行脚本的离线测试。

本模块只使用合成候选报告和项目内临时目录，
不访问真实语料、Embedding、Chroma或LLM。

测试范围：
1. 命令行默认路径和显式路径；
2. 候选JSON报告的Pydantic反序列化；
3. 门控报告到JSON安全字典的转换；
4. Markdown只输出审计摘要，不泄露Chunk正文；
5. JSON和Markdown文件写入；
6. 从输入报告到两种输出文件的完整本地流程；
7. 终端摘要区分门控安全结论与最终回答结论。
"""

import json
from pathlib import Path

import pytest

from app.schemas.evaluation import (
    ExpectedEvidence,
)
from app.schemas.retrieval import (
    DocumentChunk,
    HybridRetrievedChunk,
)
from app.schemas.retrieval_strategy import (
    CandidateQuestionEvaluation,
    CandidateStrategyMetrics,
    CandidateStrategyParameters,
    CandidateStrategyReport,
    HeadPreservingRerankParameters,
)
from app.services.hybrid_gate_evaluation import (
    HybridGateEvaluationReport,
    build_hybrid_gate_evaluation_report,
)
from app.services.hybrid_retrieval_gate import (
    HybridEvidenceGatePolicy,
)
from scripts.evaluate_hybrid_gate import (
    DEFAULT_CANDIDATE_REPORT,
    DEFAULT_JSON_OUTPUT,
    DEFAULT_MARKDOWN_OUTPUT,
    gate_report_to_dict,
    load_candidate_report,
    parse_args,
    print_summary,
    render_markdown_report,
    run_evaluation,
    write_evaluation_reports,
)


# DocumentChunk要求content_hash是64位十六进制文本。
TEST_CONTENT_HASH = "c" * 64

# 这段内容模拟不应进入可提交Markdown的原始语料。
PRIVATE_CHUNK_CONTENT = (
    "PRIVATE-CHUNK-CONTENT-"
    "只允许出现在本地JSON中"
)


@pytest.fixture
def project_tmp_path(
    request: pytest.FixtureRequest,
) -> Path:
    """在.gitignore已忽略的项目tmp目录中创建测试路径。"""

    # 不使用pytest内置tmp_path，避免Windows系统临时目录
    # 被其他安全主体创建后出现ACL权限冲突。
    temporary_directory = (
        Path("tmp")
        / "pytest_evaluate_hybrid_gate"
        / request.node.name
    )
    temporary_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return temporary_directory


def make_hybrid_candidate(
    *,
    rank: int,
    source_file: str,
    page_or_section: str,
    vector_similarity: float,
    lexical_terms: tuple[str, ...],
) -> HybridRetrievedChunk:
    """构造同时带向量信号和关键词审计字段的候选。"""

    chunk = DocumentChunk(
        chunk_id=(
            "gate-cli-document:"
            f"{rank:06d}"
        ),
        document_id="gate-cli-document",
        source_file=source_file,
        page_or_section=page_or_section,
        chunk_index=rank,
        content_hash=TEST_CONTENT_HASH,
        content=PRIVATE_CHUNK_CONTENT,
    )

    return HybridRetrievedChunk(
        chunk=chunk,
        rrf_score=0.04 - rank * 0.001,
        rank=rank,
        vector_rank=rank,
        vector_similarity=vector_similarity,
        keyword_rank=rank,
        keyword_score=5.0,
        matched_lexical_terms=lexical_terms,
    )


def make_candidate_report(
) -> CandidateStrategyReport:
    """构造一题正确放行、一题正确拒答的混合候选报告。"""

    answerable_candidate = make_hybrid_candidate(
        rank=1,
        source_file="manual.md",
        page_or_section="section: safe-check",
        vector_similarity=0.60,
        lexical_terms=("安全",),
    )

    unanswerable_candidate = make_hybrid_candidate(
        rank=1,
        source_file="background.md",
        page_or_section="section: background",
        vector_similarity=0.20,
        lexical_terms=("背景",),
    )

    results = [
        CandidateQuestionEvaluation(
            question_id="q001",
            question="安全检查是什么？",
            question_type="single_hop",
            answerable=True,
            expected_evidence=[
                ExpectedEvidence(
                    source_file="manual.md",
                    page_or_section=(
                        "section: safe-check"
                    ),
                )
            ],
            retrieved_chunks=[
                answerable_candidate
            ],
            matched_evidence_at_1=1,
            matched_evidence_at_3=1,
            fully_recalled_at_1=True,
            fully_recalled_at_3=True,
        ),
        CandidateQuestionEvaluation(
            question_id="q002",
            question="知识库没有的编号是什么？",
            question_type="unanswerable",
            answerable=False,
            expected_evidence=[],
            retrieved_chunks=[
                unanswerable_candidate
            ],
            matched_evidence_at_1=0,
            matched_evidence_at_3=0,
            fully_recalled_at_1=False,
            fully_recalled_at_3=False,
        ),
    ]

    return CandidateStrategyReport(
        embedding_model="embedding-3",
        collection_name="gate-cli-test",
        parameters=CandidateStrategyParameters(
            strategy=(
                "hybrid_rrf_rewrite_rerank"
            ),
            top_k=3,
            candidate_k=20,
            rank_constant=60,
            rewrite_version="deterministic-v2",
            # 使用真实评测相同的头部深度，
            # 验证它们能够从候选报告
            # 完整传递到门控JSON和Markdown。
            rerank_parameters=(
                HeadPreservingRerankParameters(
                    version="head-preserving-v1",
                    fused_head_k=1,
                    vector_head_k=1,
                    keyword_head_k=2,
                )
            ),
        ),
        metrics=CandidateStrategyMetrics(
            answerable_question_count=1,
            unanswerable_question_count=1,
            expected_evidence_count=1,
            matched_evidence_at_1=1,
            matched_evidence_at_3=1,
            recall_at_1=1.0,
            recall_at_3=1.0,
            fully_recalled_at_1_count=1,
            fully_recalled_at_3_count=1,
        ),
        results=results,
    )


def make_gate_report(
) -> HybridGateEvaluationReport:
    """调用真实Service生成命令行层使用的门控报告。"""

    return build_hybrid_gate_evaluation_report(
        candidate_report=make_candidate_report(),
        policy=HybridEvidenceGatePolicy(),
    )


def test_parse_args_uses_documented_default_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """无额外参数时必须使用约定的三条报告路径。"""

    monkeypatch.setattr(
        "sys.argv",
        ["evaluate_hybrid_gate.py"],
    )

    args = parse_args()

    assert (
        args.candidate_report
        == DEFAULT_CANDIDATE_REPORT
    )
    assert args.json_output == DEFAULT_JSON_OUTPUT
    assert (
        args.markdown_output
        == DEFAULT_MARKDOWN_OUTPUT
    )


def test_parse_args_preserves_explicit_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """显式命令行路径不能被默认值覆盖。"""

    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_hybrid_gate.py",
            "--candidate-report",
            "input.json",
            "--json-output",
            "output.json",
            "--markdown-output",
            "output.md",
        ],
    )

    args = parse_args()

    assert args.candidate_report == Path(
        "input.json"
    )
    assert args.json_output == Path(
        "output.json"
    )
    assert args.markdown_output == Path(
        "output.md"
    )


def test_load_candidate_report_validates_json(
    project_tmp_path: Path,
) -> None:
    """输入JSON必须重新构造成完整Pydantic报告。"""

    path = project_tmp_path / "candidate.json"
    expected = make_candidate_report()
    path.write_text(
        expected.model_dump_json(indent=2),
        encoding="utf-8",
    )

    loaded = load_candidate_report(path=path)

    assert isinstance(
        loaded,
        CandidateStrategyReport,
    )
    assert loaded == expected


def test_gate_report_to_dict_is_json_serializable(
) -> None:
    """数据类和Pydantic对象必须转换成JSON兼容结构。"""

    payload = gate_report_to_dict(
        make_gate_report()
    )

    serialized = json.dumps(
        payload,
        ensure_ascii=False,
    )

    assert payload["gate_policy"]["version"] == (
        "hybrid-evidence-gate-v1"
    )
    assert payload["metrics"][
        "unanswerable_abstention_rate"
    ] == pytest.approx(1.0)
    assert "generated_at" not in payload
    assert PRIVATE_CHUNK_CONTENT in serialized


def test_markdown_is_a_redacted_audit_summary(
) -> None:
    """可提交Markdown不得复制候选Chunk原文。"""

    markdown = render_markdown_report(
        make_gate_report()
    )

    assert "# Week 3 混合证据门控评测" in markdown
    assert "无答案正确拒答率" in markdown
    assert "q001" in markdown
    assert "q002" in markdown
    assert "`head-preserving-v1`" in markdown
    assert "| RRF头部保留数 | 1 |" in markdown
    assert "| 向量头部保留数 | 1 |" in markdown
    assert "| 关键词头部保留数 | 2 |" in markdown
    assert "生成时间" not in markdown
    assert PRIVATE_CHUNK_CONTENT not in markdown


def test_write_evaluation_reports_creates_both_files(
    project_tmp_path: Path,
) -> None:
    """写入函数应创建父目录及JSON、Markdown两份文件。"""

    json_path = (
        project_tmp_path
        / "nested"
        / "gate.json"
    )
    markdown_path = (
        project_tmp_path
        / "nested"
        / "gate.md"
    )

    write_evaluation_reports(
        report=make_gate_report(),
        json_output_path=json_path,
        markdown_output_path=markdown_path,
    )

    payload = json.loads(
        json_path.read_text(
            encoding="utf-8"
        )
    )

    assert json_path.is_file()
    assert markdown_path.is_file()
    assert payload["metrics"][
        "answerable_acceptance_rate"
    ] == pytest.approx(1.0)


def test_run_evaluation_executes_complete_local_flow(
    project_tmp_path: Path,
) -> None:
    """真实本地调用链应从候选JSON走到两份门控报告。"""

    candidate_path = (
        project_tmp_path / "candidate.json"
    )
    json_output_path = (
        project_tmp_path / "gate.json"
    )
    markdown_output_path = (
        project_tmp_path / "gate.md"
    )

    candidate_path.write_text(
        make_candidate_report().model_dump_json(
            indent=2
        ),
        encoding="utf-8",
    )

    report = run_evaluation(
        candidate_report_path=candidate_path,
        json_output_path=json_output_path,
        markdown_output_path=(
            markdown_output_path
        ),
        policy=HybridEvidenceGatePolicy(),
    )

    assert isinstance(
        report,
        HybridGateEvaluationReport,
    )
    assert (
        report.metrics
        .false_accepted_unanswerable_count
        == 0
    )
    assert json_output_path.is_file()
    assert markdown_output_path.is_file()


def test_print_summary_reports_safety_and_scope(
    capsys: pytest.CaptureFixture[str],
    project_tmp_path: Path,
) -> None:
    """终端摘要必须说明门控通过但最终回答仍待评测。"""

    report = make_gate_report()
    markdown_path = project_tmp_path / "gate.md"

    print_summary(
        report=report,
        markdown_output_path=markdown_path,
    )

    output = capsys.readouterr().out

    assert "混合证据门控评测完成" in output
    assert "无答案错误放行率：0.000" in output
    assert "无答案安全检查：通过" in output
    assert "最终回答与引用评测：尚未完成" in output
    assert str(markdown_path.resolve()) in output
