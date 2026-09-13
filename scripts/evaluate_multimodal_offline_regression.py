"""运行 Week 6 的 30 场景多模态 Agent 离线回归。

该脚本负责：

1. 加载第五周的 30 条多模态 Gold 场景；
2. 加载第六周的 30 条固定 Fixture；
3. 按 scenario_id 一一配对；
4. 运行批量离线回归服务；
5. 写入 JSON、Markdown 和 30 份脱敏轨迹；
6. 根据 24/30 和安全拒答率门槛决定退出状态。

本脚本不会创建真实 LLM、Vision、Embedding 或 Chroma 客户端。
"""

import argparse
import asyncio
from collections.abc import Sequence
import json
from pathlib import Path

from app.services.evaluation import (
    load_multimodal_agent_evaluation_scenarios,
)
from app.services.multimodal_offline_fixture_loader import (
    load_multimodal_offline_fixtures,
    pair_multimodal_offline_regression_cases,
)
from app.services.multimodal_offline_regression_service import (
    MultimodalOfflineRegressionRun,
    MultimodalOfflineRegressionService,
)
from app.services.multimodal_offline_scenario_executor import (
    MultimodalOfflineScenarioExecutor,
)
from app.services.vision_input import (
    VisionInputAdapter,
)


# scripts目录的上一级就是项目根目录。
#
# 使用__file__而不是当前工作目录，
# 保证从PowerShell、IDE或其他目录启动时都能定位项目。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_SCENARIOS_PATH = Path(
    "data/eval/multimodal_scenarios.jsonl"
)

DEFAULT_FIXTURES_PATH = Path(
    "data/eval/multimodal_offline_fixtures.jsonl"
)

DEFAULT_JSON_OUTPUT_PATH = Path(
    "docs/multimodal-offline-regression.json"
)

DEFAULT_MARKDOWN_OUTPUT_PATH = Path(
    "docs/multimodal-offline-regression.md"
)


def resolve_project_input_path(
    path: Path,
) -> Path:
    """把输入路径转换成已存在的绝对路径。

    相对路径以项目根目录为基准，而不是以当前终端目录为基准。
    """

    if not isinstance(path, Path):
        raise TypeError(
            "输入路径必须是pathlib.Path"
        )

    candidate = (
        path
        if path.is_absolute()
        else PROJECT_ROOT / path
    )

    resolved_path = candidate.resolve(
        strict=True
    )

    if not resolved_path.is_file():
        raise FileNotFoundError(
            f"输入路径不是文件：{resolved_path}"
        )

    return resolved_path


def build_project_relative_source(
    path: Path,
) -> str:
    """生成报告使用的data/eval相对POSIX路径。

    报告不能保存D盘绝对路径，
    也不能通过..引用项目外部文件。
    """

    if not isinstance(path, Path):
        raise TypeError(
            "path必须是pathlib.Path"
        )

    resolved_root = PROJECT_ROOT.resolve(
        strict=True
    )
    resolved_path = path.resolve(
        strict=True
    )

    try:
        relative_path = resolved_path.relative_to(
            resolved_root
        )
    except ValueError as exc:
        raise ValueError(
            "评测输入文件必须位于项目目录中"
        ) from exc

    # 报告契约要求数据源位于data/eval。
    if relative_path.parts[:2] != (
        "data",
        "eval",
    ):
        raise ValueError(
            "评测输入文件必须位于data/eval目录"
        )

    return relative_path.as_posix()


def resolve_docs_output_path(
    path: Path,
    *,
    expected_suffix: str,
) -> Path:
    """把报告输出路径限制在项目docs目录。

    strict=False允许目标文件尚不存在，
    但仍会规范化路径并检查是否逃离docs。
    """

    if not isinstance(path, Path):
        raise TypeError(
            "输出路径必须是pathlib.Path"
        )

    if not isinstance(expected_suffix, str):
        raise TypeError(
            "expected_suffix必须是字符串"
        )

    if not expected_suffix.startswith("."):
        raise ValueError(
            "expected_suffix必须以点号开头"
        )

    candidate = (
        path
        if path.is_absolute()
        else PROJECT_ROOT / path
    )

    resolved_path = candidate.resolve(
        strict=False
    )
    resolved_docs = (
        PROJECT_ROOT
        / "docs"
    ).resolve(strict=False)

    try:
        resolved_path.relative_to(
            resolved_docs
        )
    except ValueError as exc:
        raise ValueError(
            "报告和轨迹只能写入项目docs目录"
        ) from exc

    if resolved_path.suffix.lower() != (
        expected_suffix.lower()
    ):
        raise ValueError(
            "输出文件扩展名必须为"
            f"{expected_suffix}"
        )

    return resolved_path


def escape_markdown_cell(
    value: object,
) -> str:
    """转义Markdown表格单元格中的特殊内容。"""

    text = str(value)

    return (
        text
        .replace("|", r"\|")
        .replace("\r\n", "<br>")
        .replace("\n", "<br>")
        .replace("\r", "<br>")
    )


def format_optional_ratio(
    value: float | None,
) -> str:
    """把可选比率转换成报告文本。"""

    if value is None:
        return "不适用"

    return f"{value:.3f}"


def render_markdown_report(
    execution: MultimodalOfflineRegressionRun,
) -> str:
    """把正式离线回归结果渲染成人工审核Markdown。

    本函数只展示已经通过Schema校验的报告，
    不重新执行场景，也不重新计算passed。
    """

    if not isinstance(
        execution,
        MultimodalOfflineRegressionRun,
    ):
        raise TypeError(
            "execution必须是"
            "MultimodalOfflineRegressionRun"
        )

    report = execution.report
    evaluation = report.scored_evaluation
    metrics = evaluation.metrics
    threshold = report.threshold_summary

    lines = [
        "# 多模态 Agent 离线回归报告",
        "",
        "## 1. 回归范围",
        "",
        f"- 回归版本：`{report.regression_version}`",
        f"- Fixture 版本：`{report.fixture_version}`",
        f"- 执行模式：`{report.execution_mode}`",
        (
            "- 使用外部服务："
            + (
                "是"
                if report.external_services_used
                else "否"
            )
        ),
        f"- Gold 场景：`{report.scenario_source}`",
        f"- Fixture：`{report.fixture_source}`",
        (
            "- Planner："
            f"`{evaluation.planner_model}`"
        ),
        (
            "- Planner Prompt："
            f"`{evaluation.planner_prompt_version}`"
        ),
        (
            "- Vision Provider："
            f"`{evaluation.vision_model}`"
        ),
        (
            "- Vision Prompt："
            f"`{evaluation.vision_prompt_version}`"
        ),
        "",
        "## 2. 修复前后与验收门槛",
        "",
        "| 项目 | 修复前基线 | 当前结果 | 验收要求 | 结论 |",
        "|---|---:|---:|---:|---|",
        (
            "| 严格通过场景数 | "
            f"{threshold.baseline_passed_scenario_count} | "
            f"{threshold.current_passed_scenario_count} | "
            f"至少 {threshold.minimum_passed_scenario_count} | "
            + (
                "通过"
                if threshold.pass_threshold_met
                else "未通过"
            )
            + " |"
        ),
        (
            "| 严格场景通过率 | "
            f"{threshold.baseline_scenario_pass_rate:.3f} | "
            f"{threshold.current_scenario_pass_rate:.3f} | "
            f"{threshold.minimum_passed_scenario_count}/"
            f"{threshold.scenario_count} | "
            + (
                "通过"
                if threshold.pass_threshold_met
                else "未通过"
            )
            + " |"
        ),
        (
            "| 安全拒答率 | "
            f"{threshold.baseline_safe_refusal_rate:.3f} | "
            f"{threshold.current_safe_refusal_rate:.3f} | "
            "不得下降 | "
            + (
                "通过"
                if threshold.safety_regression_free
                else "发生回退"
            )
            + " |"
        ),
        "",
        (
            "- 相对第五周增加通过场景："
            f"**{threshold.passed_scenario_improvement}** 条"
        ),
        "",
        "## 3. 正式评分指标",
        "",
        "| 指标 | 结果 |",
        "|---|---:|",
        (
            "| 请求成功率 | "
            f"{metrics.request_success_rate:.3f} |"
        ),
        (
            "| 场景总通过率 | "
            f"{metrics.scenario_pass_rate:.3f} |"
        ),
        (
            "| 图片观察字段准确性 | "
            f"{format_optional_ratio(metrics.image_observation_field_accuracy)} |"
        ),
        (
            "| 工具选择正确率 | "
            f"{metrics.tool_selection_accuracy:.3f} |"
        ),
        (
            "| Vision工具选择正确率 | "
            f"{metrics.vision_tool_selection_accuracy:.3f} |"
        ),
        (
            "| 任务完成率 | "
            f"{metrics.task_completion_rate:.3f} |"
        ),
        (
            "| 引用正确率 | "
            f"{metrics.citation_correctness:.3f} |"
        ),
        (
            "| 引用覆盖率 | "
            f"{metrics.citation_coverage:.3f} |"
        ),
        (
            "| 来源标注正确率 | "
            f"{metrics.source_label_accuracy:.3f} |"
        ),
        (
            "| 安全拒答率 | "
            f"{metrics.safe_refusal_rate:.3f} |"
        ),
        (
            "| 平均工具步骤数 | "
            f"{metrics.average_steps:.3f} |"
        ),
        (
            "| 平均本地执行延迟 | "
            f"{metrics.mean_latency_ms:.3f} ms |"
        ),
        (
            "| P50本地执行延迟 | "
            f"{metrics.p50_latency_ms:.3f} ms |"
        ),
        (
            "| P95本地执行延迟 | "
            f"{metrics.p95_latency_ms:.3f} ms |"
        ),
        "",
        "## 4. 严格通过和失败场景",
        "",
        "### 4.1 通过场景",
        "",
        (
            "、".join(
                f"`{scenario_id}`"
                for scenario_id
                in report.strict_passed_scenario_ids
            )
            or "无"
        ),
        "",
        "### 4.2 失败场景",
        "",
        (
            "、".join(
                f"`{scenario_id}`"
                for scenario_id
                in report.strict_failed_scenario_ids
            )
            or "无"
        ),
        "",
        "## 5. 失败分类统计",
        "",
    ]

    failure_reason_counts = (
        evaluation.failure_reason_counts
    )

    if failure_reason_counts:
        lines.extend(
            [
                "| 失败原因 | 场景数 |",
                "|---|---:|",
            ]
        )

        for item in failure_reason_counts:
            lines.append(
                "| "
                f"{escape_markdown_cell(item.reason)}"
                " | "
                f"{item.scenario_count}"
                " |"
            )
    else:
        lines.append("本批次没有失败原因。")

    lines.extend(
        [
            "",
            "## 6. Fixture消费审计",
            "",
            (
                "- 完整消费Fixture的场景数："
                f"{report.fixture_fully_consumed_count}"
                f"/{threshold.scenario_count}"
            ),
            (
                "- 未完整消费Fixture的场景："
                + (
                    "、".join(
                        f"`{scenario_id}`"
                        for scenario_id
                        in (
                            report
                            .unconsumed_fixture_scenario_ids
                        )
                    )
                    or "无"
                )
            ),
            "",
            "| 场景 | 结果 | 工具顺序 | Planner调用/预设 | "
            "Vision调用/预设 | Fixture完整消费 | 失败原因 |",
            "|---|---|---|---:|---:|---|---|",
        ]
    )

    # 报告Schema已经保证results和execution_audits
    # 按scenario_id逐项对应，因此这里可以安全zip。
    for result, audit in zip(
        evaluation.results,
        report.execution_audits,
        strict=True,
    ):
        tool_sequence = (
            " → ".join(
                result.actual_tool_sequence
            )
            or "无"
        )

        failure_reasons = (
            "；".join(result.failure_reasons)
            or "无"
        )

        lines.append(
            "| "
            f"`{result.scenario_id}`"
            " | "
            + (
                "PASS"
                if result.passed
                else "FAIL"
            )
            + " | "
            f"{escape_markdown_cell(tool_sequence)}"
            " | "
            f"{audit.planner_call_count}/"
            f"{audit.planner_turn_count}"
            " | "
            f"{audit.vision_call_count}/"
            f"{audit.vision_outcome_count}"
            " | "
            + (
                "是"
                if audit.fixture_fully_consumed
                else "否"
            )
            + " | "
            f"{escape_markdown_cell(failure_reasons)}"
            " |"
        )

    lines.extend(
        [
            "",
            "## 7. 可靠性修正记录",
            "",
            "| 已发现问题 | 修改 | 验证方式 |",
            "|---|---|---|",
        ]
    )

    for fix in evaluation.fix_history:
        lines.append(
            "| "
            f"{escape_markdown_cell(fix.issue)}"
            " | "
            f"{escape_markdown_cell(fix.change)}"
            " | "
            f"{escape_markdown_cell(fix.verification)}"
            " |"
        )

    lines.extend(
        [
            "",
            "## 8. 评测限制",
            "",
            (
                "- 本报告验证的是确定性安全分类、请求级工具范围、"
                "Agent进度状态机、Runner、Executor、证据白名单、"
                "报告构造和评分规则的离线稳定性。"
            ),
            (
                "- Planner、Vision和普通工具结果来自固定Fixture，"
                "因此该报告不能代表真实模型输出的随机性或线上质量。"
            ),
            (
                "- 本地执行延迟只反映Fake依赖和本机Python主链耗时，"
                "不能作为真实网络模型延迟或生产容量指标。"
            ),
            (
                "- 本批次没有调用可计费外部模型，"
                "因此不使用该报告估算真实模型成本。"
            ),
            (
                "- 报告与轨迹不保存完整图片、密钥、"
                "完整敏感日志或模型私有思维链。"
            ),
            "",
        ]
    )

    return "\n".join(lines)


def write_text_artifacts_atomically(
    artifacts: Sequence[
        tuple[Path, str]
    ],
) -> None:
    """先写临时文件，再统一替换正式目标。

    如果任意内容写入失败，finally只清理由本函数创建的
    临时文件，避免留下半份JSON或Markdown报告。
    """

    if isinstance(
        artifacts,
        (str, bytes, bytearray),
    ) or not isinstance(artifacts, Sequence):
        raise TypeError(
            "artifacts必须是路径与文本的序列"
        )

    artifact_items = tuple(artifacts)

    if not artifact_items:
        raise ValueError(
            "artifacts不能为空"
        )

    target_paths = tuple(
        path
        for path, _ in artifact_items
    )

    if len(target_paths) != len(
        set(target_paths)
    ):
        raise ValueError(
            "输出目标路径不能重复"
        )

    staged_files: list[
        tuple[Path, Path]
    ] = []

    try:
        for target_path, content in artifact_items:
            if not isinstance(target_path, Path):
                raise TypeError(
                    "每个输出目标必须是pathlib.Path"
                )

            if not isinstance(content, str):
                raise TypeError(
                    "每个输出内容必须是字符串"
                )

            target_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            temporary_path = (
                target_path.parent
                / f".{target_path.name}.tmp"
            )

            temporary_path.write_text(
                content,
                encoding="utf-8",
                newline="\n",
            )

            staged_files.append(
                (
                    temporary_path,
                    target_path,
                )
            )

        # 所有临时文件都成功写入后才替换正式文件。
        for temporary_path, target_path in (
            staged_files
        ):
            temporary_path.replace(
                target_path
            )
    finally:
        for temporary_path, _ in staged_files:
            if temporary_path.exists():
                temporary_path.unlink()


def write_regression_artifacts(
    *,
    execution: MultimodalOfflineRegressionRun,
    json_output_path: Path,
    markdown_output_path: Path,
) -> None:
    """写入正式报告和30份完整脱敏轨迹。"""

    if not isinstance(
        execution,
        MultimodalOfflineRegressionRun,
    ):
        raise TypeError(
            "execution必须是"
            "MultimodalOfflineRegressionRun"
        )

    resolved_json = resolve_docs_output_path(
        json_output_path,
        expected_suffix=".json",
    )
    resolved_markdown = resolve_docs_output_path(
        markdown_output_path,
        expected_suffix=".md",
    )

    if resolved_json == resolved_markdown:
        raise ValueError(
            "JSON和Markdown不能写入同一路径"
        )

    report_json = json.dumps(
        execution.report.model_dump(
            mode="json"
        ),
        ensure_ascii=False,
        indent=2,
    ) + "\n"

    markdown = render_markdown_report(
        execution
    )

    artifacts: list[
        tuple[Path, str]
    ] = [
        (
            resolved_json,
            report_json,
        ),
        (
            resolved_markdown,
            markdown,
        ),
    ]

    trace_paths = (
        execution
        .report
        .scored_evaluation
        .trace_paths
    )

    # MultimodalOfflineRegressionRun已经校验轨迹对象、
    # 场景编号和trace_paths一一对应。
    for trace_path, trace in zip(
        trace_paths,
        execution.traces,
        strict=True,
    ):
        resolved_trace_path = (
            resolve_docs_output_path(
                Path(trace_path),
                expected_suffix=".json",
            )
        )

        trace_json = json.dumps(
            trace.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        ) + "\n"

        artifacts.append(
            (
                resolved_trace_path,
                trace_json,
            )
        )

    write_text_artifacts_atomically(
        artifacts
    )


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """解析离线回归命令行参数。"""

    parser = argparse.ArgumentParser(
        description=(
            "使用Fake Planner、Fake Vision和Fake工具"
            "运行30条多模态Agent离线回归"
        )
    )

    parser.add_argument(
        "--scenarios",
        type=Path,
        default=DEFAULT_SCENARIOS_PATH,
        help="30条多模态Gold场景JSONL",
    )

    parser.add_argument(
        "--fixtures",
        type=Path,
        default=DEFAULT_FIXTURES_PATH,
        help="30条多模态离线Fixture JSONL",
    )

    parser.add_argument(
        "--json-output",
        type=Path,
        default=DEFAULT_JSON_OUTPUT_PATH,
        help="正式JSON报告输出路径",
    )

    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=DEFAULT_MARKDOWN_OUTPUT_PATH,
        help="正式Markdown报告输出路径",
    )

    return parser.parse_args(argv)


def print_summary(
    *,
    execution: MultimodalOfflineRegressionRun,
    json_output_path: Path,
    markdown_output_path: Path,
) -> None:
    """打印包含Week 6门槛的终端摘要。"""

    if not isinstance(
        execution,
        MultimodalOfflineRegressionRun,
    ):
        raise TypeError(
            "execution必须是"
            "MultimodalOfflineRegressionRun"
        )

    report = execution.report
    metrics = report.scored_evaluation.metrics
    threshold = report.threshold_summary

    print("多模态 Agent 离线回归完成")
    print(
        f"场景数：{metrics.scenario_count}"
    )
    print(
        "严格通过数："
        f"{metrics.passed_scenario_count}"
        f"/{metrics.scenario_count}"
    )
    print(
        "严格通过率："
        f"{metrics.scenario_pass_rate:.3f}"
    )
    print(
        "相对第五周增加："
        f"{threshold.passed_scenario_improvement}条"
    )
    print(
        "24/30门槛："
        + (
            "通过"
            if threshold.pass_threshold_met
            else "未通过"
        )
    )
    print(
        "安全拒答率："
        f"{metrics.safe_refusal_rate:.3f}"
    )
    print(
        "安全回退检查："
        + (
            "通过"
            if threshold.safety_regression_free
            else "未通过"
        )
    )
    print(
        "Fixture完整消费："
        f"{report.fixture_fully_consumed_count}"
        f"/{metrics.scenario_count}"
    )
    print(
        "失败场景："
        + (
            "、".join(
                report.strict_failed_scenario_ids
            )
            or "无"
        )
    )
    print(
        "脱敏轨迹数："
        f"{len(execution.traces)}"
    )
    print(
        f"JSON报告：{json_output_path}"
    )
    print(
        f"Markdown报告：{markdown_output_path}"
    )


async def run_offline_regression(
    *,
    scenarios_path: Path,
    fixtures_path: Path,
) -> MultimodalOfflineRegressionRun:
    """加载数据并执行完整离线回归。"""

    scenarios = (
        load_multimodal_agent_evaluation_scenarios(
            scenarios_path
        )
    )

    fixtures = load_multimodal_offline_fixtures(
        fixtures_path
    )

    cases = pair_multimodal_offline_regression_cases(
        scenarios=scenarios,
        fixtures=fixtures,
    )

    vision_input_adapter = VisionInputAdapter(
        max_image_size_bytes=1_000_000,
        max_image_dimension_px=2_048,
        max_image_pixels=4_000_000,
    )

    scenario_executor = (
        MultimodalOfflineScenarioExecutor(
            project_root=PROJECT_ROOT,
            vision_input_adapter=(
                vision_input_adapter
            ),
        )
    )

    regression_service = (
        MultimodalOfflineRegressionService(
            executor=scenario_executor,
        )
    )

    return await regression_service.run(
        cases=cases,
        scenario_source=(
            build_project_relative_source(
                scenarios_path
            )
        ),
        fixture_source=(
            build_project_relative_source(
                fixtures_path
            )
        ),
    )


def main() -> None:
    """执行CLI完整流程并强制检查验收门槛。"""

    args = parse_args()

    scenarios_path = resolve_project_input_path(
        args.scenarios
    )
    fixtures_path = resolve_project_input_path(
        args.fixtures
    )

    json_output_path = resolve_docs_output_path(
        args.json_output,
        expected_suffix=".json",
    )
    markdown_output_path = resolve_docs_output_path(
        args.markdown_output,
        expected_suffix=".md",
    )

    execution = asyncio.run(
        run_offline_regression(
            scenarios_path=scenarios_path,
            fixtures_path=fixtures_path,
        )
    )

    # 即使门槛未通过，也先保存报告，
    # 使开发者能够查看具体失败原因。
    write_regression_artifacts(
        execution=execution,
        json_output_path=json_output_path,
        markdown_output_path=(
            markdown_output_path
        ),
    )

    print_summary(
        execution=execution,
        json_output_path=json_output_path,
        markdown_output_path=(
            markdown_output_path
        ),
    )

    threshold = execution.report.threshold_summary

    # 未达到门槛时使用非零退出状态，
    # 让CI或自动验收能够识别回归失败。
    if not threshold.pass_threshold_met:
        raise SystemExit(
            "离线回归未达到24/30严格通过门槛"
        )

    if not threshold.safety_regression_free:
        raise SystemExit(
            "离线回归安全拒答率低于第五周基线"
        )


if __name__ == "__main__":
    main()