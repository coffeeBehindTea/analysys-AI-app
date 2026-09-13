"""Week 5 多模态 Agent 正式评测脚本的离线测试。

本文件只验证命令行编排、报告渲染、文件写入和依赖装配。
它不会启动 FastAPI、不会读取真实图片，也不会调用 LLM、
Embedding、Vision 或 Chroma 网络服务。
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
import json

import pytest

from app.config import Settings
from app.schemas.multimodal_agent_evaluation import (
    MultimodalAgentBatchExecution,
    MultimodalAgentCategorySummary,
    MultimodalAgentEvaluationMetrics,
    MultimodalAgentEvaluationReport,
    MultimodalAgentTrace,
    MultimodalAgentTraceArtifact,
    MultimodalAgentTraceInput,
    MultimodalEvaluationFixRecord,
    MultimodalFailureReasonCount,
)

import scripts.evaluate_multimodal as script_module

from scripts.evaluate_multimodal import (
    DEFAULT_API_URL,
    DEFAULT_JSON_OUTPUT_PATH,
    DEFAULT_MARKDOWN_OUTPUT_PATH,
    DEFAULT_SCENARIOS_PATH,
    DEFAULT_TIMEOUT_SECONDS,
    PROJECT_ROOT,
    build_fix_history,
    build_trace_targets,
    parse_args,
    print_summary,
    render_markdown_report,
    resolve_project_path,
    run_real_evaluation,
    write_evaluation_artifacts,
)


def make_metrics() -> MultimodalAgentEvaluationMetrics:
    """构造供渲染测试使用的稳定汇总指标。

    model_construct() 在这里用于隔离测试脚本的展示层，
    不重复测试已经由 Schema 测试覆盖的指标交叉校验。
    """

    return MultimodalAgentEvaluationMetrics.model_construct(
        scenario_count=30,
        request_success_count=30,
        request_success_rate=1.0,
        passed_scenario_count=24,
        scenario_pass_rate=0.8,
        tool_selection_correct_count=27,
        tool_selection_accuracy=0.9,
        expected_completion_count=20,
        completed_task_count=18,
        task_completion_rate=0.9,
        returned_citation_count=20,
        correct_citation_count=19,
        citation_correctness=0.95,
        expected_evidence_count=25,
        matched_expected_evidence_count=20,
        citation_coverage=0.8,
        safe_refusal_case_count=8,
        correct_safe_refusal_count=8,
        safe_refusal_rate=1.0,
        total_step_count=90,
        average_steps=3.0,
        visual_observation_field_count=60,
        matched_visual_observation_count=54,
        image_observation_field_accuracy=0.9,
        vision_tool_selection_correct_count=28,
        vision_tool_selection_accuracy=(28 / 30),
        source_label_correct_count=27,
        source_label_accuracy=0.9,
        safety_requirement_count=90,
        passed_safety_requirement_count=88,
        safety_requirement_pass_rate=(88 / 90),
        mean_latency_ms=1200.0,
        p50_latency_ms=1000.0,
        p95_latency_ms=2400.0,
        cost_estimated_scenario_count=0,
        cost_estimation_coverage=0.0,
        known_total_estimated_cost_usd=None,
        mean_estimated_cost_usd_per_request=None,
        llm_judge_evaluated_count=0,
        llm_judge_pass_count=0,
        llm_judge_pass_rate=None,
    )


def make_report() -> MultimodalAgentEvaluationReport:
    """构造不依赖真实 Agent 响应的展示层报告。"""

    return MultimodalAgentEvaluationReport.model_construct(
        evaluation_version=(
            "multimodal-agent-evaluation-v1"
        ),
        scenario_source=(
            "data/eval/multimodal_scenarios.jsonl"
        ),
        api_url=DEFAULT_API_URL,
        planner_model="test-planner-model",
        planner_prompt_version="agent-tool-calling-v3",
        embedding_model="test-embedding-model",
        collection_name="test-collection",
        vision_model="test-vision-model",
        vision_prompt_version=(
            "robot-vision-observation-v1"
        ),
        llm_judge_role="disabled",
        metrics=make_metrics(),
        results=(),
        category_summaries=(
            MultimodalAgentCategorySummary.model_construct(
                category="normal_image",
                scenario_count=8,
                passed_scenario_count=7,
                scenario_pass_rate=0.875,
            ),
        ),
        trace_paths=(
            "docs/multimodal-traces/multimodal-agent-001.json",
            "docs/multimodal-traces/multimodal-agent-023.json",
            "docs/multimodal-traces/multimodal-agent-030.json",
        ),
        failure_reason_counts=(
            MultimodalFailureReasonCount(
                reason="工具选择不符合 Gold 约束",
                scenario_count=2,
            ),
        ),
        fix_history=(
            MultimodalEvaluationFixRecord(
                issue="轨迹不得保存图片正文",
                change="使用脱敏轨迹契约",
                verification="离线安全测试通过",
            ),
        ),
    )


def make_batch_execution(
) -> MultimodalAgentBatchExecution:
    """构造报告写入测试需要的三条脱敏轨迹产物。"""

    report = make_report()
    artifacts: list[
        MultimodalAgentTraceArtifact
    ] = []

    for index, path in enumerate(
        report.trace_paths,
        start=1,
    ):
        scenario_id = (
            "multimodal-agent-001"
            if index == 1
            else (
                "multimodal-agent-023"
                if index == 2
                else "multimodal-agent-030"
            )
        )

        trace = MultimodalAgentTrace.model_construct(
            trace_version="multimodal-agent-trace-v1",
            scenario_id=scenario_id,
            scenario_name=f"脱敏轨迹 {index}",
            category="normal_image",
            input=MultimodalAgentTraceInput(
                robot_id="robot-001",
                symptom="脱敏现象",
                log_excerpt="sanitized log",
                task_goal="完成只读诊断",
                image_count=1,
                image_analysis_goals=(
                    "读取可见状态",
                ),
            ),
            response={
                "request_id": f"request-{index}",
                "diagnosis": {
                    "status": "abstained",
                },
            },
            evaluation={
                "scenario_id": scenario_id,
                "request_succeeded": True,
            },
        )

        artifacts.append(
            MultimodalAgentTraceArtifact.model_construct(
                path=path,
                trace=trace,
            )
        )

    return MultimodalAgentBatchExecution.model_construct(
        report=report,
        trace_artifacts=tuple(artifacts),
    )


def make_settings() -> Settings:
    """创建不读取 .env 的完整测试配置。"""

    return Settings(
        _env_file=None,
        llm_model="test-planner-model",
        embedding_model="test-embedding-model",
        vision_model="test-vision-model",
        chroma_collection_name="test-collection",
        max_vision_image_size_bytes=1024,
        max_vision_image_dimension_px=512,
        max_vision_image_pixels=262_144,
    )


@pytest.mark.parametrize(
    "help_option",
    ("-h", "--help"),
)
def test_parse_args_accepts_standard_help_options(
    help_option: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """短、长帮助参数都应正常打印帮助并返回退出码 0。"""

    with pytest.raises(SystemExit) as exc_info:
        parse_args([help_option])

    captured = capsys.readouterr()

    assert exc_info.value.code == 0
    assert "usage:" in captured.out
    assert "-h, --help" in captured.out
    assert "unrecognized arguments" not in captured.err


def test_parse_args_uses_documented_defaults() -> None:
    """空参数应使用正式场景、接口、报告和超时默认值。"""

    args = parse_args([])

    assert args.scenarios == DEFAULT_SCENARIOS_PATH
    assert args.api_url == DEFAULT_API_URL
    assert args.json_output == DEFAULT_JSON_OUTPUT_PATH
    assert args.markdown_output == (
        DEFAULT_MARKDOWN_OUTPUT_PATH
    )
    assert args.timeout_seconds == (
        DEFAULT_TIMEOUT_SECONDS
    )
    assert args.enable_llm_judge is False
    assert args.llm_judge_timeout_seconds == 30.0


def test_parse_args_enables_auxiliary_llm_judge() -> None:
    """显式开关应启用Judge并解析独立的正数超时。

    被测试模块是parse_args()。输入两个新增命令行参数，
    预期argparse把开关转换成True，并使用_positive_float()
    把超时文字转换成15.5，而不是影响Agent API请求超时。
    """

    args = parse_args(
        [
            "--enable-llm-judge",
            "--llm-judge-timeout-seconds",
            "15.5",
        ]
    )

    assert args.enable_llm_judge is True
    assert args.llm_judge_timeout_seconds == 15.5


def test_resolve_project_path_anchors_relative_path() -> None:
    """相对路径必须相对脚本所在项目根目录解析。"""

    assert resolve_project_path(
        Path("docs/result.json")
    ) == (
        PROJECT_ROOT / "docs" / "result.json"
    ).resolve()


def test_trace_targets_cover_three_diverse_scenarios() -> None:
    """正式轨迹应覆盖正常、冲突和高风险拒答三类场景。"""

    targets = build_trace_targets()

    assert tuple(targets) == (
        "multimodal-agent-001",
        "multimodal-agent-023",
        "multimodal-agent-030",
    )
    assert len(set(targets.values())) == 3
    assert all(
        path.startswith("docs/")
        and path.endswith(".json")
        for path in targets.values()
    )


def test_fix_history_is_nonempty_and_unique() -> None:
    """正式报告必须携带非空且问题描述唯一的修正记录。

    被测试模块是build_fix_history()。测试直接取得静态修正
    记录，不调用Agent或外部模型。预期流程是构造每条经过
    Schema校验的issue、change和verification；预期结果是
    记录非空、问题描述互不重复，并包含工具范围回滚与成本
    统计限制这两项本轮修正。
    """

    history = build_fix_history()

    assert len(history) >= 1
    assert all(
        isinstance(
            item,
            MultimodalEvaluationFixRecord,
        )
        for item in history
    )
    assert len(
        {item.issue for item in history}
    ) == len(history)
    assert any(
        "两次" in item.issue
        and "隐藏知识检索" in item.issue
        for item in history
    )
    assert any(
        "空成本字段" in item.issue
        for item in history
    )


def test_render_markdown_contains_required_sections_without_time(
) -> None:
    """人工报告必须覆盖核心指标、失败、轨迹和修正且没有时间字段。"""

    markdown = render_markdown_report(
        make_report()
    )

    assert markdown.startswith(
        "# 多模态 Agent 可靠性评测"
    )
    assert "图片观察字段准确性" in markdown
    assert "工具选择正确率" in markdown
    assert "P50 端到端延迟" in markdown
    assert "### 3.1 成本统计限制" in markdown
    assert "模型 Token 使用量" in markdown
    assert "不表示调用成本为 0" in markdown
    assert "不会据此推测或伪造金额" in markdown
    assert "## 5. 失败原因" in markdown
    assert "## 6. 完整脱敏轨迹" in markdown
    assert "## 7. 修正记录" in markdown
    assert "source_image_sha256" in markdown
    assert "来源完整性指纹" in markdown
    assert "generated_at" not in markdown
    assert "生成时间" not in markdown


@pytest.mark.filterwarnings(
    "ignore:Pydantic serializer warnings:UserWarning"
)
def test_write_evaluation_artifacts_writes_report_and_three_safe_traces(
    tmp_path: Path,
) -> None:
    """写入器应一次生成 JSON、Markdown 和三份无图片正文轨迹。"""

    project_root = tmp_path.resolve()
    json_output = (
        project_root
        / "docs"
        / "multimodal-evaluation.json"
    )
    markdown_output = (
        project_root
        / "docs"
        / "multimodal-evaluation.md"
    )

    write_evaluation_artifacts(
        execution=make_batch_execution(),
        project_root=project_root,
        json_output_path=json_output,
        markdown_output_path=markdown_output,
    )

    assert json_output.is_file()
    assert markdown_output.is_file()

    report_data = json.loads(
        json_output.read_text(encoding="utf-8")
    )
    assert report_data["metrics"]["scenario_count"] == 30
    assert "generated_at" not in report_data

    trace_files = tuple(
        (
            project_root
            / Path(*Path(path).parts)
        )
        for path in make_report().trace_paths
    )

    assert all(path.is_file() for path in trace_files)

    combined_trace_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in trace_files
    )

    assert "image_base64" not in combined_trace_text
    assert "image_path" not in combined_trace_text
    assert "image_sha256" not in combined_trace_text


def test_write_evaluation_artifacts_rejects_output_outside_docs(
    tmp_path: Path,
) -> None:
    """正式报告不能通过自定义路径写到项目 docs 目录之外。"""

    project_root = tmp_path.resolve()

    with pytest.raises(
        ValueError,
        match="必须位于项目docs目录",
    ):
        write_evaluation_artifacts(
            execution=make_batch_execution(),
            project_root=project_root,
            json_output_path=(
                project_root / "outside.json"
            ),
            markdown_output_path=(
                project_root
                / "docs"
                / "report.md"
            ),
        )


@pytest.mark.asyncio
async def test_run_real_evaluation_assembles_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """真实入口应装配限制、模型元数据、轨迹目标并调用批量服务。"""

    scenarios = tuple(
        object()
        for _ in range(30)
    )
    expected_execution = make_batch_execution()
    expected_client = object()

    input_adapter_factory = Mock(
        return_value=object()
    )
    batch_evaluator = AsyncMock(
        return_value=expected_execution
    )

    async_client_context = AsyncMock()
    async_client_context.__aenter__.return_value = (
        expected_client
    )
    async_client_factory = Mock(
        return_value=async_client_context
    )

    monkeypatch.setattr(
        script_module,
        "VisionInputAdapter",
        input_adapter_factory,
    )
    monkeypatch.setattr(
        script_module.httpx,
        "AsyncClient",
        async_client_factory,
    )
    monkeypatch.setattr(
        script_module,
        "evaluate_multimodal_agent_scenarios_via_api",
        batch_evaluator,
    )
    monkeypatch.setattr(
        script_module,
        "get_llm_model",
        Mock(return_value="test-planner-model"),
    )
    monkeypatch.setattr(
        script_module,
        "get_embedding_model",
        Mock(return_value="test-embedding-model"),
    )
    monkeypatch.setattr(
        script_module,
        "get_vision_model",
        Mock(return_value="test-vision-model"),
    )

    result = await run_real_evaluation(
        scenarios=scenarios,  # type: ignore[arg-type]
        settings=make_settings(),
        api_url=DEFAULT_API_URL,
        timeout_seconds=45.0,
    )

    assert result is expected_execution
    input_adapter_factory.assert_called_once_with(
        max_image_size_bytes=1024,
        max_image_dimension_px=512,
        max_image_pixels=262_144,
    )
    async_client_factory.assert_called_once()
    batch_evaluator.assert_awaited_once()

    call_arguments = (
        batch_evaluator.await_args.kwargs
    )
    assert call_arguments["scenarios"] is scenarios
    assert call_arguments["client"] is expected_client
    assert call_arguments["project_root"] == PROJECT_ROOT
    assert call_arguments["planner_model"] == (
        "test-planner-model"
    )
    assert call_arguments["embedding_model"] == (
        "test-embedding-model"
    )
    assert call_arguments["vision_model"] == (
        "test-vision-model"
    )
    assert call_arguments[
        "visual_equivalence_judge"
    ] is None
    assert tuple(call_arguments["trace_targets"]) == (
        "multimodal-agent-001",
        "multimodal-agent-023",
        "multimodal-agent-030",
    )


@pytest.mark.asyncio
async def test_run_real_evaluation_wires_and_closes_optional_judge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """启用开关时应装配辅助Judge，并在结束后关闭客户端。

    被测试模块是run_real_evaluation()。测试用Mock替代
    AsyncOpenAI、Judge构造器、HTTP客户端和批量执行器。
    预期流程是创建一个独立LLM客户端，把Judge传入服务层，
    返回批量结果后调用异步close()释放连接池。
    """

    scenarios = tuple(
        object()
        for _ in range(30)
    )
    expected_execution = make_batch_execution()
    expected_http_client = object()
    expected_judge = object()

    llm_client = SimpleNamespace(
        close=AsyncMock()
    )
    llm_client_factory = Mock(
        return_value=llm_client
    )
    judge_factory = Mock(
        return_value=expected_judge
    )
    batch_evaluator = AsyncMock(
        return_value=expected_execution
    )

    async_client_context = AsyncMock()
    async_client_context.__aenter__.return_value = (
        expected_http_client
    )

    monkeypatch.setattr(
        script_module,
        "VisionInputAdapter",
        Mock(return_value=object()),
    )
    monkeypatch.setattr(
        script_module.httpx,
        "AsyncClient",
        Mock(return_value=async_client_context),
    )
    monkeypatch.setattr(
        script_module,
        "evaluate_multimodal_agent_scenarios_via_api",
        batch_evaluator,
    )
    monkeypatch.setattr(
        script_module,
        "get_llm_model",
        Mock(return_value="test-planner-model"),
    )
    monkeypatch.setattr(
        script_module,
        "get_embedding_model",
        Mock(return_value="test-embedding-model"),
    )
    monkeypatch.setattr(
        script_module,
        "get_vision_model",
        Mock(return_value="test-vision-model"),
    )
    monkeypatch.setattr(
        script_module,
        "create_llm_client",
        llm_client_factory,
    )
    monkeypatch.setattr(
        script_module,
        "OpenAICompatibleVisualEquivalenceJudge",
        judge_factory,
    )

    result = await run_real_evaluation(
        scenarios=scenarios,  # type: ignore[arg-type]
        settings=make_settings(),
        api_url=DEFAULT_API_URL,
        timeout_seconds=45.0,
        enable_llm_judge=True,
        llm_judge_timeout_seconds=12.5,
    )

    assert result is expected_execution
    llm_client_factory.assert_called_once()
    judge_factory.assert_called_once_with(
        client=llm_client,
        model="test-planner-model",
        timeout_seconds=12.5,
    )
    assert (
        batch_evaluator.await_args.kwargs[
            "visual_equivalence_judge"
        ]
        is expected_judge
    )
    llm_client.close.assert_awaited_once_with()


def test_print_summary_reports_all_required_metrics_and_paths(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """终端摘要必须显示指标、成本限制、轨迹数和报告路径。

    被测试模块是print_summary()。测试把cost coverage为0且
    平均成本为None的固定报告传入展示函数，再用capsys捕获
    标准输出。预期流程是先输出全部任务六指标，再把成本显示
    为无法估算并解释其不等于零成本，最后输出轨迹和报告路径。
    """

    execution = make_batch_execution()

    print_summary(
        execution=execution,
        json_output_path=Path(
            "docs/multimodal-evaluation.json"
        ),
        markdown_output_path=Path(
            "docs/multimodal-evaluation.md"
        ),
    )

    output = capsys.readouterr().out

    assert "多模态 Agent 评测完成" in output
    assert "场景数：30" in output
    assert "图片观察字段准确性：0.900" in output
    assert "工具选择正确率：0.900" in output
    assert "任务完成率：0.900" in output
    assert "引用正确率：0.950" in output
    assert "安全拒答率：1.000" in output
    assert "单请求平均估算成本：无法估算" in output
    assert "成本统计限制：" in output
    assert "无法估算不等于零成本" in output
    assert "完整脱敏轨迹数：3" in output
    assert "docs\\multimodal-evaluation.json" in output
