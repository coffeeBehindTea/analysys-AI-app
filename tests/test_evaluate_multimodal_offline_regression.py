"""Week 6 多模态离线回归CLI的测试。

被测试模块：

scripts.evaluate_multimodal_offline_regression

本文件覆盖命令行参数、路径安全、Markdown渲染、
原子化写入、正式30场景离线执行和验收门槛退出逻辑。
"""

import argparse
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.evaluate_multimodal_offline_regression as script
from app.services.multimodal_offline_regression_service import (
    MultimodalOfflineRegressionRun,
)


# 使用测试文件位置定位真实项目根目录。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# 这两个输入就是CLI默认使用的正式数据。
SCENARIO_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "eval"
    / "multimodal_scenarios.jsonl"
)

FIXTURE_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "eval"
    / "multimodal_offline_fixtures.jsonl"
)


@pytest.fixture(scope="module")
def completed_offline_run(
) -> MultimodalOfflineRegressionRun:
    """执行一次正式30场景离线回归并在本模块复用。

    fixture的scope="module"表示整个测试文件只运行一次，
    避免渲染测试和写入测试重复执行30条Agent主链。
    """

    return asyncio.run(
        script.run_offline_regression(
            scenarios_path=SCENARIO_DATA_PATH,
            fixtures_path=FIXTURE_DATA_PATH,
        )
    )


def test_parse_args_uses_documented_defaults(
) -> None:
    """不传参数时应使用文档中的四个默认路径。"""

    args = script.parse_args([])

    assert args.scenarios == (
        script.DEFAULT_SCENARIOS_PATH
    )
    assert args.fixtures == (
        script.DEFAULT_FIXTURES_PATH
    )
    assert args.json_output == (
        script.DEFAULT_JSON_OUTPUT_PATH
    )
    assert args.markdown_output == (
        script.DEFAULT_MARKDOWN_OUTPUT_PATH
    )


def test_parse_args_preserves_explicit_paths(
) -> None:
    """显式命令行路径应覆盖默认值并解析为Path。"""

    args = script.parse_args(
        [
            "--scenarios",
            "data/eval/custom-scenarios.jsonl",
            "--fixtures",
            "data/eval/custom-fixtures.jsonl",
            "--json-output",
            "docs/custom-report.json",
            "--markdown-output",
            "docs/custom-report.md",
        ]
    )

    assert args.scenarios == Path(
        "data/eval/custom-scenarios.jsonl"
    )
    assert args.fixtures == Path(
        "data/eval/custom-fixtures.jsonl"
    )
    assert args.json_output == Path(
        "docs/custom-report.json"
    )
    assert args.markdown_output == Path(
        "docs/custom-report.md"
    )


def test_resolve_docs_output_rejects_path_outside_docs(
    tmp_path: Path,
) -> None:
    """输出路径不能通过父目录跳转逃离docs。"""

    with pytest.raises(
        ValueError,
        match="只能写入项目docs目录",
    ):
        script.resolve_docs_output_path(
            Path("docs/../secret.json"),
            expected_suffix=".json",
        )


def test_resolve_docs_output_rejects_wrong_suffix(
) -> None:
    """JSON输出不能伪装成其他扩展名。"""

    with pytest.raises(
        ValueError,
        match="输出文件扩展名必须为.json",
    ):
        script.resolve_docs_output_path(
            Path("docs/report.txt"),
            expected_suffix=".json",
        )


def test_real_offline_run_reaches_week7_closed_baseline(
    completed_offline_run: (
        MultimodalOfflineRegressionRun
    ),
) -> None:
    """真实30场景离线入口应全部通过且无安全回退。

    被测试模块是脚本层run_real_evaluation()所复用的完整离线
    回归执行结果。

    测试方法是读取模块级Fixture已经执行完成的30场景报告，
    检查严格通过数、安全拒答率、验收门槛和轨迹数量。

    预期流程是30条Gold与30条Fixture逐一配对，经过生产Agent
    主链、正式评分器和报告组装器，且不访问外部服务。

    预期结果是30/30严格通过、安全拒答率1.0、验收门槛通过、
    没有安全回退，并保留30份完整脱敏轨迹。
    """

    report = completed_offline_run.report
    metrics = report.scored_evaluation.metrics
    threshold = report.threshold_summary

    assert metrics.scenario_count == 30
    assert metrics.passed_scenario_count == 30
    assert metrics.safe_refusal_rate == 1.0
    assert threshold.pass_threshold_met is True
    assert threshold.safety_regression_free is True
    assert len(completed_offline_run.traces) == 30


def test_markdown_contains_auditable_sections_without_time(
    completed_offline_run: (
        MultimodalOfflineRegressionRun
    ),
) -> None:
    """Markdown应展示新基线、空失败集和Fixture审计。

    被测试模块是render_markdown_report()。

    测试方法是把30/30正式离线结果渲染成内存字符串，并检查
    报告标题、前后对比、失败分类、Fixture审计和时间字段。

    预期流程是渲染器只读取已校验报告，不重新执行场景；最终
    文本应记录30/30的新基线，且不再把008列为失败场景。

    预期结果是审计章节齐全、包含30/30、不包含旧失败编号，
    并且不出现generated_at或报告生成时间。
    """

    markdown = script.render_markdown_report(
        completed_offline_run
    )

    assert "# 多模态 Agent 离线回归报告" in markdown
    assert "## 2. 修复前后与验收门槛" in markdown
    assert "## 5. 失败分类统计" in markdown
    assert "## 6. Fixture消费审计" in markdown
    assert "30/30" in markdown
    assert "multimodal-agent-008" in markdown
    assert (
        "### 4.2 失败场景\n\n无\n"
        in markdown
    )
    assert "generated_at" not in markdown
    assert "报告生成时间" not in markdown


def test_writer_creates_reports_and_thirty_traces(
    completed_offline_run: (
        MultimodalOfflineRegressionRun
    ),
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """写入器应在临时项目docs中生成全32个文件。

    32个文件包括：

    - 1个JSON正式报告；
    - 1个Markdown正式报告；
    - 30个脱敏场景轨迹。
    """

    # 只改变当前脚本模块中的项目根目录，
    # 使测试不会覆盖真实docs报告。
    monkeypatch.setattr(
        script,
        "PROJECT_ROOT",
        tmp_path,
    )

    script.write_regression_artifacts(
        execution=completed_offline_run,
        json_output_path=Path(
            "docs/offline-report.json"
        ),
        markdown_output_path=Path(
            "docs/offline-report.md"
        ),
    )

    json_path = (
        tmp_path
        / "docs"
        / "offline-report.json"
    )
    markdown_path = (
        tmp_path
        / "docs"
        / "offline-report.md"
    )
    trace_directory = (
        tmp_path
        / "docs"
        / "traces"
        / "multimodal-offline"
    )

    assert json_path.is_file()
    assert markdown_path.is_file()
    assert len(
        list(trace_directory.glob("*.json"))
    ) == 30

    report_data = json.loads(
        json_path.read_text(encoding="utf-8")
    )

    assert report_data[
        "external_services_used"
    ] is False
    assert "generated_at" not in report_data

    # 原子写入完成后不应留下.tmp文件。
    assert not list(
        (tmp_path / "docs").rglob("*.tmp")
    )


def test_atomic_writer_rejects_duplicate_targets(
    tmp_path: Path,
) -> None:
    """同一目标不能被两份内容在同一批次重复写入。"""

    target = tmp_path / "duplicate.json"

    with pytest.raises(
        ValueError,
        match="输出目标路径不能重复",
    ):
        script.write_text_artifacts_atomically(
            (
                (target, "first"),
                (target, "second"),
            )
        )


@pytest.mark.parametrize(
    (
        "pass_threshold_met",
        "safety_regression_free",
        "expected_error",
    ),
    [
        (True, True, None),
        (
            False,
            True,
            "未达到24/30严格通过门槛",
        ),
        (
            True,
            False,
            "安全拒答率低于第五周基线",
        ),
    ],
)
def test_main_checks_threshold_after_writing_reports(
    pass_threshold_met: bool,
    safety_regression_free: bool,
    expected_error: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """main()应先保存失败证据，再用非零状态退出。"""

    execution = SimpleNamespace(
        report=SimpleNamespace(
            threshold_summary=SimpleNamespace(
                pass_threshold_met=(
                    pass_threshold_met
                ),
                safety_regression_free=(
                    safety_regression_free
                ),
            )
        )
    )

    async def fake_run_offline_regression(
        *,
        scenarios_path: Path,
        fixtures_path: Path,
    ) -> object:
        """返回当前参数化门槛结果。"""

        return execution

    call_order: list[str] = []

    monkeypatch.setattr(
        script,
        "parse_args",
        lambda: argparse.Namespace(
            scenarios=Path("scenarios.jsonl"),
            fixtures=Path("fixtures.jsonl"),
            json_output=Path("report.json"),
            markdown_output=Path("report.md"),
        ),
    )
    monkeypatch.setattr(
        script,
        "resolve_project_input_path",
        lambda path: path,
    )
    monkeypatch.setattr(
        script,
        "resolve_docs_output_path",
        lambda path, expected_suffix: path,
    )
    monkeypatch.setattr(
        script,
        "run_offline_regression",
        fake_run_offline_regression,
    )
    monkeypatch.setattr(
        script,
        "write_regression_artifacts",
        lambda **kwargs: call_order.append(
            "write"
        ),
    )
    monkeypatch.setattr(
        script,
        "print_summary",
        lambda **kwargs: call_order.append(
            "print"
        ),
    )

    if expected_error is None:
        script.main()
    else:
        with pytest.raises(
            SystemExit,
            match=expected_error,
        ):
            script.main()

    # 不论门槛是否通过，都必须先写报告并打印摘要。
    assert call_order == ["write", "print"]
