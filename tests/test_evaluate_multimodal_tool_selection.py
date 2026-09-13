"""多模态工具选择评测脚本的离线自动化测试。

本文件验证命令行参数、路径、JSONL案例读取、
实验范围、Markdown渲染、报告写入和依赖装配。

测试不会执行真实OCR，也不会调用Vision网络服务。
"""

import json

from pathlib import Path

from unittest.mock import (
    AsyncMock,
    Mock,
)

import pytest

from app.config import Settings

from app.schemas.multimodal_tool_selection import (
    MultimodalToolSelectionCase,
)

from app.schemas.multimodal_tool_selection_evaluation import (
    MultimodalToolSelectionRouteResult,
)

from app.schemas.multimodal_tool_selection_report import (
    MultimodalToolSelectionExperimentReport,
)

from app.services.multimodal_tool_selection_experiment import (
    _build_route_summary,
)

import scripts.evaluate_multimodal_tool_selection as script_module

from scripts.evaluate_multimodal_tool_selection import (
    DEFAULT_CASES_PATH,
    DEFAULT_JSON_OUTPUT_PATH,
    DEFAULT_MARKDOWN_OUTPUT_PATH,
    PROJECT_ROOT,
    format_actual_items,
    format_cost,
    format_ratio,
    load_cases,
    markdown_cell,
    parse_args,
    print_summary,
    render_markdown_report,
    resolve_project_path,
    run_real_experiment,
    validate_experiment_scope,
    write_reports,
)


def load_gold_cases_directly(
) -> tuple[
    MultimodalToolSelectionCase,
    ...,
]:
    """绕过被测load_cases，直接构造仓库中的六个Gold案例。"""

    cases: list[
        MultimodalToolSelectionCase
    ] = []

    for raw_line in (
        DEFAULT_CASES_PATH
        .read_text(encoding="utf-8")
        .splitlines()
    ):
        if not raw_line.strip():
            continue

        cases.append(
            MultimodalToolSelectionCase
            .model_validate_json(raw_line)
        )

    return tuple(cases)


@pytest.fixture(scope="module")
def gold_cases(
) -> tuple[
    MultimodalToolSelectionCase,
    ...,
]:
    """为本模块提供已经通过Schema的六个Gold案例。"""

    return load_gold_cases_directly()


def make_route_result(
    case: MultimodalToolSelectionCase,
) -> MultimodalToolSelectionRouteResult:
    """为报告渲染测试构造一个首选路线通过结果。"""

    if case.expects_abstention:
        return MultimodalToolSelectionRouteResult(
            case_id=case.case_id,
            route=case.preferred_route,
            source_image_sha256=(
                case.image_sha256
            ),
            execution_status="abstained",
            actual_items=(),
            matched_expected_items=(),
            missing_expected_items=(),
            content_accuracy=None,
            abstained=True,
            expected_abstention=True,
            route_appropriate=True,
            outcome_correct=True,
            safety_passed=True,
            passed=True,
            requires_human_check=True,
            latency_ms=1.0,
            external_model_call_count=0,
            cost_status="not_applicable",
            estimated_cost_usd=0.0,
            public_message=(
                "图片不足以形成可靠观察"
            ),
        )

    expected_items = (
        case.expected_text_terms
        + case.expected_visual_observations
    )

    uses_external_model = (
        case.preferred_route
        == "vision_model"
    )

    return MultimodalToolSelectionRouteResult(
        case_id=case.case_id,
        route=case.preferred_route,
        source_image_sha256=(
            case.image_sha256
        ),
        execution_status="completed",
        actual_items=expected_items,
        matched_expected_items=(
            expected_items
        ),
        missing_expected_items=(),
        content_accuracy=1.0,
        abstained=False,
        expected_abstention=False,
        route_appropriate=True,
        outcome_correct=True,
        safety_passed=True,
        passed=True,
        requires_human_check=False,
        latency_ms=10.0,
        external_model_call_count=(
            1 if uses_external_model else 0
        ),
        cost_status=(
            "estimated"
            if uses_external_model
            else "not_applicable"
        ),
        estimated_cost_usd=(
            0.001
            if uses_external_model
            else 0.0
        ),
        cost_note=(
            "测试成本估算"
            if uses_external_model
            else None
        ),
    )


def make_report(
    cases: tuple[
        MultimodalToolSelectionCase,
        ...,
    ],
) -> MultimodalToolSelectionExperimentReport:
    """根据案例构造内部计数一致的批量实验报告。"""

    results = tuple(
        make_route_result(case)
        for case in cases
    )

    routes = tuple(
        dict.fromkeys(
            result.route
            for result in results
        )
    )

    route_summaries = tuple(
        _build_route_summary(
            route=route,
            results=tuple(
                result
                for result in results
                if result.route == route
            ),
        )
        for route in routes
    )

    return (
        MultimodalToolSelectionExperimentReport(
            case_count=len(cases),
            route_result_count=len(results),
            routes=routes,
            results=results,
            route_summaries=(
                route_summaries
            ),
        )
    )


def make_settings() -> Settings:
    """创建不读取.env且具有完整Vision配置的测试Settings。"""

    return Settings(
        _env_file=None,
        vision_api_key="test-vision-key",
        vision_base_url=(
            "https://vision.example/v1"
        ),
        vision_model="test-vision-model",
        max_vision_image_size_bytes=1024,
        max_vision_image_dimension_px=512,
        max_vision_image_pixels=262_144,
        vision_timeout_seconds=12.0,
        ocr_language="eng+chi_sim",
        ocr_minimum_confidence=75.0,
        ocr_timeout_seconds=4.0,
    )


@pytest.mark.parametrize(
    "help_option",
    (
        "-h",
        "--help",
    ),
)
def test_parse_args_accepts_standard_help_options(
    help_option: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """短、长帮助选项都应打印帮助并以退出码0结束。"""

    # argparse使用SystemExit(0)结束帮助流程，
    # 这是命令行程序的正常行为，不是执行失败。
    with pytest.raises(SystemExit) as exc_info:
        parse_args([help_option])

    captured = capsys.readouterr()

    assert exc_info.value.code == 0
    assert "usage:" in captured.out
    assert "-h, --help" in captured.out
    assert "unrecognized arguments" not in captured.err


def test_parse_args_uses_documented_default_paths() -> None:
    """没有显式参数时应使用项目内约定的案例和报告路径。"""

    args = parse_args([])

    assert args.cases == DEFAULT_CASES_PATH
    assert args.json_output == DEFAULT_JSON_OUTPUT_PATH
    assert (
        args.markdown_output
        == DEFAULT_MARKDOWN_OUTPUT_PATH
    )


def test_parse_args_preserves_explicit_paths() -> None:
    """调用方指定的三个路径应被解析成Path并保持原值。"""

    args = parse_args(
        [
            "--cases",
            "data/eval/custom-cases.jsonl",
            "--json-output",
            "docs/custom-result.json",
            "--markdown-output",
            "docs/custom-result.md",
        ]
    )

    assert args.cases == Path(
        "data/eval/custom-cases.jsonl"
    )
    assert args.json_output == Path(
        "docs/custom-result.json"
    )
    assert args.markdown_output == Path(
        "docs/custom-result.md"
    )


def test_resolve_project_path_anchors_relative_path() -> None:
    """相对路径应以项目根目录为基准，而不是当前终端目录。"""

    result = resolve_project_path(
        Path("docs/result.json")
    )

    assert result == (
        PROJECT_ROOT
        / "docs"
        / "result.json"
    ).resolve()


def test_resolve_project_path_preserves_absolute_path(
    tmp_path: Path,
) -> None:
    """绝对路径应保持调用方指定的位置。"""

    absolute_path = (
        tmp_path
        / "result.json"
    ).resolve()

    assert resolve_project_path(
        absolute_path
    ) == absolute_path


def test_resolve_project_path_rejects_non_path() -> None:
    """路径边界不接受未经Path转换的普通字符串。"""

    with pytest.raises(
        TypeError,
        match="path必须是Path",
    ):
        resolve_project_path(
            "docs/result.json"  # type: ignore[arg-type]
        )


def test_load_cases_reads_jsonl_and_ignores_blank_lines(
    tmp_path: Path,
    gold_cases: tuple[
        MultimodalToolSelectionCase,
        ...,
    ],
) -> None:
    """合法JSONL应逐行校验，并允许案例之间存在空行。"""

    cases_path = (
        tmp_path
        / "cases.jsonl"
    )

    cases_path.write_text(
        "\n"
        + gold_cases[0].model_dump_json()
        + "\n\n"
        + gold_cases[1].model_dump_json()
        + "\n",
        encoding="utf-8",
    )

    loaded_cases = load_cases(
        cases_path
    )

    assert loaded_cases == gold_cases[:2]


def test_load_cases_rejects_non_path() -> None:
    """案例读取函数只接受明确的Path对象。"""

    with pytest.raises(
        TypeError,
        match="cases_path必须是Path",
    ):
        load_cases(
            "cases.jsonl"  # type: ignore[arg-type]
        )


def test_load_cases_rejects_missing_file(
    tmp_path: Path,
) -> None:
    """不存在的清单应在读取前立即失败。"""

    with pytest.raises(
        FileNotFoundError,
    ):
        load_cases(
            tmp_path
            / "missing.jsonl"
        )


def test_load_cases_rejects_directory(
    tmp_path: Path,
) -> None:
    """目录不能被误当成JSONL普通文件。"""

    with pytest.raises(
        ValueError,
        match="必须是普通文件",
    ):
        load_cases(tmp_path)


def test_load_cases_reports_invalid_json_line(
    tmp_path: Path,
) -> None:
    """JSON语法错误必须包含具体行号。"""

    cases_path = (
        tmp_path
        / "invalid-json.jsonl"
    )
    cases_path.write_text(
        "\n{invalid-json}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"无效JSON：line=2",
    ):
        load_cases(cases_path)


def test_load_cases_rejects_non_object_line(
    tmp_path: Path,
) -> None:
    """JSONL中的数组或标量不能冒充案例对象。"""

    cases_path = (
        tmp_path
        / "non-object.jsonl"
    )
    cases_path.write_text(
        '["not", "an", "object"]\n',
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"必须是对象：line=1",
    ):
        load_cases(cases_path)


def test_load_cases_reports_schema_error_line(
    tmp_path: Path,
) -> None:
    """字段缺失等Schema错误必须保留JSONL行号。"""

    cases_path = (
        tmp_path
        / "invalid-schema.jsonl"
    )
    cases_path.write_text(
        json.dumps(
            {
                "case_id": (
                    "multimodal-001"
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"数据契约：line=1",
    ):
        load_cases(cases_path)


def test_load_cases_rejects_empty_file(
    tmp_path: Path,
) -> None:
    """只含空白的清单不能生成空实验。"""

    cases_path = (
        tmp_path
        / "empty.jsonl"
    )
    cases_path.write_text(
        "\n\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="没有包含任何有效案例",
    ):
        load_cases(cases_path)


def test_load_cases_rejects_duplicate_case_id(
    tmp_path: Path,
    gold_cases: tuple[
        MultimodalToolSelectionCase,
        ...,
    ],
) -> None:
    """相同case_id不能在一轮实验中被重复统计。"""

    cases_path = (
        tmp_path
        / "duplicate.jsonl"
    )
    serialized_case = (
        gold_cases[0]
        .model_dump_json()
    )
    cases_path.write_text(
        serialized_case
        + "\n"
        + serialized_case
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="重复case_id",
    ):
        load_cases(cases_path)


def test_validate_experiment_scope_accepts_six_gold_cases(
    gold_cases: tuple[
        MultimodalToolSelectionCase,
        ...,
    ],
) -> None:
    """正式六案例数据应满足任务二实验规模。"""

    assert validate_experiment_scope(
        gold_cases
    ) is None


def test_validate_experiment_scope_rejects_non_tuple(
    gold_cases: tuple[
        MultimodalToolSelectionCase,
        ...,
    ],
) -> None:
    """实验范围校验只接受不可追加项目的tuple。"""

    with pytest.raises(
        TypeError,
        match="cases必须是tuple",
    ):
        validate_experiment_scope(
            list(gold_cases)  # type: ignore[arg-type]
        )


def test_validate_experiment_scope_rejects_fewer_than_six_cases(
    gold_cases: tuple[
        MultimodalToolSelectionCase,
        ...,
    ],
) -> None:
    """少于任务要求的六个问题时不能生成正式报告。"""

    with pytest.raises(
        ValueError,
        match="至少需要6个案例",
    ):
        validate_experiment_scope(
            gold_cases[:5]
        )


def test_markdown_cell_escapes_table_delimiters_and_lines(
) -> None:
    """竖线和各种换行不能破坏Markdown表格结构。"""

    assert markdown_cell(
        "A|B\r\nC\nD\rE"
    ) == "A\\|B<br>C<br>D<br>E"


@pytest.mark.parametrize(
    (
        "value",
        "expected_text",
    ),
    (
        (None, "不适用"),
        (0.12549, "0.125"),
    ),
)
def test_format_ratio_handles_missing_and_numeric_values(
    value: float | None,
    expected_text: str,
) -> None:
    """比率应统一显示三位小数，缺失值应明确标注。"""

    assert format_ratio(value) == (
        expected_text
    )


def test_format_actual_items_handles_empty_and_multiple_items(
) -> None:
    """观察项应以中文分号连接，空结果应显示为无。"""

    assert format_actual_items(()) == "无"
    assert format_actual_items(
        (
            "观察A",
            "观察B",
        )
    ) == "观察A；观察B"


def test_format_cost_uses_route_summary_cost(
    gold_cases: tuple[
        MultimodalToolSelectionCase,
        ...,
    ],
) -> None:
    """已知成本应以美元六位小数展示。"""

    report = make_report(gold_cases)
    summary_by_route = {
        summary.route: summary
        for summary
        in report.route_summaries
    }

    assert format_cost(
        summary_by_route["ocr_rule"]
    ) == "$0.000000"
    assert format_cost(
        summary_by_route["vision_model"]
    ).startswith("$0.00")


def test_render_markdown_report_contains_auditable_sections(
    gold_cases: tuple[
        MultimodalToolSelectionCase,
        ...,
    ],
) -> None:
    """Markdown应包含实验规模、路线汇总和每个案例结果。"""

    report = make_report(gold_cases)

    markdown = render_markdown_report(
        report=report,
        cases=gold_cases,
    )

    assert markdown.startswith(
        "# 多模态工具选择对照实验"
    )
    assert "## 4. 路线汇总" in markdown
    assert "## 6. 单案例路线结果" in markdown

    for case in gold_cases:
        assert case.case_id in markdown


def test_render_markdown_report_rejects_wrong_report_type(
    gold_cases: tuple[
        MultimodalToolSelectionCase,
        ...,
    ],
) -> None:
    """渲染器不能接收未经报告Schema校验的对象。"""

    with pytest.raises(
        TypeError,
        match="report必须是",
    ):
        render_markdown_report(
            report=object(),  # type: ignore[arg-type]
            cases=gold_cases,
        )


def test_render_markdown_report_rejects_non_tuple_cases(
    gold_cases: tuple[
        MultimodalToolSelectionCase,
        ...,
    ],
) -> None:
    """渲染器要求案例集合在渲染期间保持不可变。"""

    with pytest.raises(
        TypeError,
        match="cases必须是tuple",
    ):
        render_markdown_report(
            report=make_report(
                gold_cases
            ),
            cases=list(  # type: ignore[arg-type]
                gold_cases
            ),
        )


def test_render_markdown_report_rejects_case_result_mismatch(
    gold_cases: tuple[
        MultimodalToolSelectionCase,
        ...,
    ],
) -> None:
    """案例定义与结果编号集合不同就不能生成误导性报告。"""

    with pytest.raises(
        ValueError,
        match="案例集合与报告结果不一致",
    ):
        render_markdown_report(
            report=make_report(
                gold_cases
            ),
            cases=gold_cases[:-1],
        )


def test_write_reports_creates_utf8_json_and_markdown(
    tmp_path: Path,
    gold_cases: tuple[
        MultimodalToolSelectionCase,
        ...,
    ],
) -> None:
    """报告写入器应创建父目录并保存两种可审计格式。"""

    report = make_report(gold_cases)
    json_path = (
        tmp_path
        / "nested"
        / "report.json"
    )
    markdown_path = (
        tmp_path
        / "other"
        / "report.md"
    )

    write_reports(
        report=report,
        cases=gold_cases,
        json_output_path=json_path,
        markdown_output_path=(
            markdown_path
        ),
    )

    json_data = json.loads(
        json_path.read_text(
            encoding="utf-8"
        )
    )
    markdown = markdown_path.read_text(
        encoding="utf-8"
    )

    assert json_data["case_count"] == 6
    assert json_data[
        "route_result_count"
    ] == 6
    assert markdown.startswith(
        "# 多模态工具选择对照实验"
    )


@pytest.mark.parametrize(
    "invalid_field",
    (
        "json",
        "markdown",
    ),
)
def test_write_reports_rejects_non_path_output(
    invalid_field: str,
    tmp_path: Path,
    gold_cases: tuple[
        MultimodalToolSelectionCase,
        ...,
    ],
) -> None:
    """两个输出位置都必须先转换成Path对象。"""

    report = make_report(gold_cases)
    json_path: object = (
        tmp_path
        / "report.json"
    )
    markdown_path: object = (
        tmp_path
        / "report.md"
    )

    if invalid_field == "json":
        json_path = "report.json"
        expected_message = (
            "json_output_path必须是Path"
        )
    else:
        markdown_path = "report.md"
        expected_message = (
            "markdown_output_path必须是Path"
        )

    with pytest.raises(
        TypeError,
        match=expected_message,
    ):
        write_reports(
            report=report,
            cases=gold_cases,
            json_output_path=json_path,  # type: ignore[arg-type]
            markdown_output_path=markdown_path,  # type: ignore[arg-type]
        )


def test_write_reports_rejects_same_output_path(
    tmp_path: Path,
    gold_cases: tuple[
        MultimodalToolSelectionCase,
        ...,
    ],
) -> None:
    """JSON和Markdown不能互相覆盖。"""

    output_path = (
        tmp_path
        / "same-output"
    )

    with pytest.raises(
        ValueError,
        match="不能写入同一路径",
    ):
        write_reports(
            report=make_report(
                gold_cases
            ),
            cases=gold_cases,
            json_output_path=output_path,
            markdown_output_path=(
                output_path
            ),
        )


def test_print_summary_reports_routes_and_paths(
    capsys: pytest.CaptureFixture[str],
    gold_cases: tuple[
        MultimodalToolSelectionCase,
        ...,
    ],
) -> None:
    """终端摘要应给出规模、路线指标和两个报告位置。"""

    print_summary(
        report=make_report(gold_cases),
        json_output_path=Path(
            "docs/result.json"
        ),
        markdown_output_path=Path(
            "docs/result.md"
        ),
    )

    output = capsys.readouterr().out

    assert "多模态工具选择对照实验完成" in output
    assert "案例数：6" in output
    assert "本地 OCR + 规则解析" in output
    assert "Vision 模型" in output
    assert "直接拒答" in output
    assert "docs\\result.json" in output
    assert "docs\\result.md" in output


@pytest.mark.asyncio
async def test_run_real_experiment_assembles_dependencies_and_closes_client(
    monkeypatch: pytest.MonkeyPatch,
    gold_cases: tuple[
        MultimodalToolSelectionCase,
        ...,
    ],
) -> None:
    """真实入口应正确传递配置，并在成功后关闭异步客户端。"""

    settings = make_settings()
    expected_report = make_report(
        gold_cases
    )

    input_adapter_factory = Mock(
        return_value=object()
    )
    ocr_provider_factory = Mock(
        return_value=object()
    )
    vision_provider_factory = Mock(
        return_value=object()
    )
    route_executor_factory = Mock(
        return_value=object()
    )

    experiment_service = Mock()
    experiment_service.run = AsyncMock(
        return_value=expected_report
    )
    experiment_service_factory = Mock(
        return_value=experiment_service
    )

    vision_client = Mock()
    vision_client.close = AsyncMock()
    vision_client_factory = Mock(
        return_value=vision_client
    )
    vision_model_resolver = Mock(
        return_value="test-vision-model"
    )

    monkeypatch.setattr(
        script_module,
        "VisionInputAdapter",
        input_adapter_factory,
    )
    monkeypatch.setattr(
        script_module,
        "TesseractOcrProvider",
        ocr_provider_factory,
    )
    monkeypatch.setattr(
        script_module,
        "OpenAICompatibleVisionProvider",
        vision_provider_factory,
    )
    monkeypatch.setattr(
        script_module,
        "MultimodalToolSelectionRouteExecutor",
        route_executor_factory,
    )
    monkeypatch.setattr(
        script_module,
        "MultimodalToolSelectionExperimentService",
        experiment_service_factory,
    )
    monkeypatch.setattr(
        script_module,
        "create_vision_client",
        vision_client_factory,
    )
    monkeypatch.setattr(
        script_module,
        "get_vision_model",
        vision_model_resolver,
    )

    result = await run_real_experiment(
        cases=gold_cases,
        settings=settings,
    )

    assert result is expected_report

    input_adapter_factory.assert_called_once_with(
        max_image_size_bytes=1024,
        max_image_dimension_px=512,
        max_image_pixels=262_144,
    )
    ocr_provider_factory.assert_called_once_with(
        language="eng+chi_sim",
        minimum_confidence=75.0,
        timeout_seconds=4.0,
    )
    vision_client_factory.assert_called_once_with(
        settings
    )
    vision_model_resolver.assert_called_once_with(
        settings
    )
    experiment_service.run.assert_awaited_once_with(
        cases=gold_cases
    )
    vision_client.close.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_run_real_experiment_closes_client_when_service_fails(
    monkeypatch: pytest.MonkeyPatch,
    gold_cases: tuple[
        MultimodalToolSelectionCase,
        ...,
    ],
) -> None:
    """批量实验抛错时finally仍必须关闭HTTP连接池。"""

    vision_client = Mock()
    vision_client.close = AsyncMock()

    experiment_service = Mock()
    experiment_service.run = AsyncMock(
        side_effect=RuntimeError(
            "simulated experiment failure"
        )
    )

    monkeypatch.setattr(
        script_module,
        "VisionInputAdapter",
        Mock(return_value=object()),
    )
    monkeypatch.setattr(
        script_module,
        "TesseractOcrProvider",
        Mock(return_value=object()),
    )
    monkeypatch.setattr(
        script_module,
        "OpenAICompatibleVisionProvider",
        Mock(return_value=object()),
    )
    monkeypatch.setattr(
        script_module,
        "MultimodalToolSelectionRouteExecutor",
        Mock(return_value=object()),
    )
    monkeypatch.setattr(
        script_module,
        "MultimodalToolSelectionExperimentService",
        Mock(
            return_value=(
                experiment_service
            )
        ),
    )
    monkeypatch.setattr(
        script_module,
        "create_vision_client",
        Mock(return_value=vision_client),
    )
    monkeypatch.setattr(
        script_module,
        "get_vision_model",
        Mock(
            return_value=(
                "test-vision-model"
            )
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="simulated experiment failure",
    ):
        await run_real_experiment(
            cases=gold_cases,
            settings=make_settings(),
        )

    vision_client.close.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_run_real_experiment_rejects_non_settings(
    gold_cases: tuple[
        MultimodalToolSelectionCase,
        ...,
    ],
) -> None:
    """依赖装配入口必须拒绝未经Settings校验的配置对象。"""

    with pytest.raises(
        TypeError,
        match="settings必须是Settings",
    ):
        await run_real_experiment(
            cases=gold_cases,
            settings=object(),  # type: ignore[arg-type]
        )
