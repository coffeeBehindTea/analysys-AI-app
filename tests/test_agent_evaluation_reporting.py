"""Agent评测命令行参数与报告输出的离线测试。

被测试模块：

scripts.evaluate_agent

被测试函数：

1. parse_args；
2. render_markdown_report；
3. write_reports；
4. print_summary。

预期调用链：

AgentEvaluationReport
→ render_markdown_report生成可审计Markdown
→ write_reports写出JSON和Markdown
→ print_summary输出不含敏感正文的终端摘要。

本测试不启动FastAPI服务，不调用LLM、Embedding、
ChromaDB或任何Agent工具。
"""

import json

from pathlib import Path

import pytest

from app.schemas.agent_evaluation import (
    AgentEvaluationMetrics,
    AgentEvaluationReport,
    AgentScenarioEvaluation,
)
from scripts.evaluate_agent import (
    parse_args,
    print_summary,
    render_markdown_report,
    write_reports,
)


# 固定文档标识和Chunk标识只用于离线构造报告。
#
# 它们不对应真实客户或生产文档，
# 也不会被发送到Agent API。
TEST_DOCUMENT_ID = "c" * 64
TEST_CHUNK_ID = (
    f"{TEST_DOCUMENT_ID}:000003"
)
TEST_SOURCE_FILE = "仓储机器人故障说明.txt"
TEST_LOCATION = "section: ERR-NET-4001"


def make_completed_result(
) -> AgentScenarioEvaluation:
    """构造一条完整完成并正确引用的场景结果。"""

    expected_evidence = {
        "source_file": TEST_SOURCE_FILE,
        "page_or_section": TEST_LOCATION,
    }

    return AgentScenarioEvaluation(
        scenario_id="agent-001",
        scenario_type="knowledge_only",
        request_succeeded=True,
        http_status_code=200,
        request_id="request-report-001",
        request_error=None,
        actual_tool_sequence=(
            "search_knowledge",
        ),
        actual_execution_state="completed",
        actual_termination_reason=(
            "planner_finished"
        ),
        actual_finish_reason="task_completed",
        actual_diagnosis_status="completed",
        step_count=1,
        citation_evaluations=(
            {
                "chunk_id": TEST_CHUNK_ID,
                "document_id": TEST_DOCUMENT_ID,
                "source_file": TEST_SOURCE_FILE,
                "page_or_section": TEST_LOCATION,
                "correct": True,
            },
        ),
        expected_evidence=(
            expected_evidence,
        ),
        matched_expected_evidence=(
            dict(expected_evidence),
        ),
        tool_selection_correct=True,
        task_completion_correct=True,
        safe_refusal_correct=None,
        passed=True,
        failure_reasons=(),
    )


def make_failed_result(
) -> AgentScenarioEvaluation:
    """构造一条HTTP失败且没有伪造Agent观察的结果。"""

    return AgentScenarioEvaluation(
        scenario_id="agent-002",
        scenario_type="high_risk_request",
        request_succeeded=False,
        http_status_code=502,
        request_id="request-report-002",
        request_error=(
            "Agent API返回错误状态：502"
        ),
        actual_tool_sequence=(),
        actual_execution_state=None,
        actual_termination_reason=None,
        actual_finish_reason=None,
        actual_diagnosis_status=None,
        step_count=0,
        citation_evaluations=(),
        expected_evidence=(),
        matched_expected_evidence=(),
        tool_selection_correct=False,
        task_completion_correct=None,
        safe_refusal_correct=False,
        passed=False,
        failure_reasons=(
            "Agent API返回错误状态：502",
        ),
    )


def make_metrics(
) -> AgentEvaluationMetrics:
    """构造与一条成功、一条失败结果一致的指标。"""

    return AgentEvaluationMetrics(
        scenario_count=2,
        request_success_count=1,
        request_success_rate=0.5,
        passed_scenario_count=1,
        scenario_pass_rate=0.5,
        tool_selection_correct_count=1,
        tool_selection_accuracy=0.5,
        expected_completion_count=1,
        completed_task_count=1,
        task_completion_rate=1.0,
        returned_citation_count=1,
        correct_citation_count=1,
        citation_correctness=1.0,
        expected_evidence_count=1,
        matched_expected_evidence_count=1,
        citation_coverage=1.0,
        safe_refusal_case_count=1,
        correct_safe_refusal_count=0,
        safe_refusal_rate=0.0,
        total_step_count=1,
        average_steps=0.5,
    )


def make_report() -> AgentEvaluationReport:
    """构造报告渲染和落盘共同使用的合法报告。"""

    return AgentEvaluationReport(
        evaluation_version=(
            "agent-evaluation-v1"
        ),
        scenario_source=(
            "data/eval/agent_scenarios.jsonl"
        ),
        api_url=(
            "http://127.0.0.1:8000"
            "/api/v1/agent/diagnose"
        ),
        planner_model="deepseek-chat",
        planner_prompt_version=(
            "agent-tool-calling-v2"
        ),
        embedding_model="embedding-3",
        collection_name=(
            "robot_knowledge_week3_"
            "vector_baseline"
        ),
        metrics=make_metrics(),
        results=(
            make_completed_result(),
            make_failed_result(),
        ),
    )


def test_parse_args_uses_documented_defaults(
) -> None:
    """无参数运行应使用仓库内约定的评测路径和API。"""

    args = parse_args([])

    assert args.scenarios == Path(
        "data/eval/agent_scenarios.jsonl"
    )
    assert args.api_url == (
        "http://127.0.0.1:8000"
        "/api/v1/agent/diagnose"
    )
    assert args.timeout_seconds == 120.0
    assert args.json_output == Path(
        "docs/agent-evaluation.json"
    )
    assert args.markdown_output == Path(
        "docs/agent-evaluation.md"
    )


def test_parse_args_preserves_explicit_values(
) -> None:
    """显式命令行值必须覆盖默认值并转换成正确类型。"""

    args = parse_args(
        [
            "--scenarios",
            "custom/scenarios.jsonl",
            "--api-url",
            "http://localhost:9000/agent",
            "--timeout-seconds",
            "45.5",
            "--json-output",
            "out/report.json",
            "--markdown-output",
            "out/report.md",
        ]
    )

    assert args.scenarios == Path(
        "custom/scenarios.jsonl"
    )
    assert args.api_url == (
        "http://localhost:9000/agent"
    )
    assert args.timeout_seconds == 45.5
    assert args.json_output == Path(
        "out/report.json"
    )
    assert args.markdown_output == Path(
        "out/report.md"
    )


def test_parse_args_rejects_non_positive_timeout(
) -> None:
    """超时时间为零或负数时应在发HTTP请求前拒绝。"""

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--timeout-seconds",
                "0",
            ]
        )


def test_render_markdown_report_is_auditable(
) -> None:
    """Markdown应包含配置、核心指标、逐场景结果和失败原因。"""

    markdown = render_markdown_report(
        make_report()
    )

    assert (
        "# Robot Diagnostic Agent评测报告"
        in markdown
    )
    assert "agent-evaluation-v1" in markdown
    assert "deepseek-chat" in markdown
    assert "agent-tool-calling-v2" in markdown
    assert "工具选择正确率" in markdown
    assert "任务完成率" in markdown
    assert "引用正确率" in markdown
    assert "平均工具步骤数" in markdown
    assert "安全拒答率" in markdown
    assert "agent-001" in markdown
    assert "agent-002" in markdown
    assert (
        "Agent API返回错误状态：502"
        in markdown
    )


def test_render_markdown_report_omits_generation_time(
) -> None:
    """可复现报告不能包含动态生成时间字段或标题。"""

    markdown = render_markdown_report(
        make_report()
    )

    assert "generated_at" not in markdown
    assert "生成时间" not in markdown


def test_write_reports_creates_round_trip_files(
    tmp_path: Path,
) -> None:
    """落盘函数应创建目录并写出可重新校验的同一报告。"""

    report = make_report()
    json_path = (
        tmp_path
        / "nested"
        / "agent-evaluation.json"
    )
    markdown_path = (
        tmp_path
        / "nested"
        / "agent-evaluation.md"
    )

    write_reports(
        report=report,
        json_output=json_path,
        markdown_output=markdown_path,
    )

    raw_json = json.loads(
        json_path.read_text(
            encoding="utf-8"
        )
    )
    loaded_report = (
        AgentEvaluationReport.model_validate(
            raw_json
        )
    )

    assert loaded_report == report
    assert "generated_at" not in raw_json
    assert (
        markdown_path.read_text(
            encoding="utf-8"
        )
        == render_markdown_report(report)
    )


def test_print_summary_reports_core_metrics_and_paths(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """终端摘要应显示核心指标和两个报告文件名。"""

    print_summary(
        report=make_report(),
        json_path=Path(
            "docs/agent-evaluation.json"
        ),
        markdown_path=Path(
            "docs/agent-evaluation.md"
        ),
    )

    output = capsys.readouterr().out

    assert "Agent评测完成" in output
    assert "场景数：2" in output
    assert "工具选择正确率：0.500" in output
    assert "任务完成率：1.000" in output
    assert "引用正确率：1.000" in output
    assert "平均工具步骤数：0.500" in output
    assert "安全拒答率：0.000" in output
    assert "agent-evaluation.json" in output
    assert "agent-evaluation.md" in output
