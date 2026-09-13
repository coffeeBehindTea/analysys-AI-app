"""多模态Agent评测数据生成模块的离线单元测试。

当前测试只覆盖图片索引加载基础设施：

1. JSONL读取和错误定位；
2. 图片文件SHA-256计算；
3. 项目data目录边界；
4. 六张脱敏图片的真实离线加载；
5. 重复、缺失、非法记录和摘要不一致。

本模块不调用Vision、LLM、Embedding、Chroma或Agent API。
"""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

import scripts.generate_multimodal_agent_scenarios as module
from app.schemas.multimodal_agent_evaluation import (
    MultimodalEvaluationImage,
)


# 一个固定的二进制文件内容，用于验证SHA-256计算结果。
TEST_BINARY_CONTENT = b"multimodal-agent-image"


def write_jsonl(
    path: Path,
    records: list[object],
) -> None:
    """把测试记录写成UTF-8 JSONL文件。

    该辅助函数只服务于tmp_path临时目录，
    不会写入项目的真实data/eval目录。
    """

    serialized_lines = [
        json.dumps(
            record,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for record in records
    ]

    path.write_text(
        "\n".join(serialized_lines) + "\n",
        encoding="utf-8",
    )


def load_real_source_records(
) -> list[dict[str, object]]:
    """读取真实脱敏图片清单并复制成可修改测试数据。"""

    return [
        deepcopy(record)
        for record in module.read_jsonl(
            module.SOURCE_IMAGE_MANIFEST_PATH
        )
    ]


def use_temporary_manifest(
    *,
    monkeypatch: pytest.MonkeyPatch,
    path: Path,
    records: list[object],
) -> None:
    """写入临时清单并让被测模块只读取该文件。"""

    write_jsonl(
        path,
        records,
    )

    # monkeypatch.setattr()只在当前测试期间替换模块常量，
    # 测试结束后pytest会自动恢复原值。
    monkeypatch.setattr(
        module,
        "SOURCE_IMAGE_MANIFEST_PATH",
        path,
    )


def test_output_paths_are_anchored_to_project_root(
) -> None:
    """生成文件必须位于当前项目的data和docs目录。"""

    assert module.SCENARIO_OUTPUT_PATH == (
        module.PROJECT_ROOT
        / "data"
        / "eval"
        / "multimodal_scenarios.jsonl"
    )
    assert module.DOCUMENTATION_OUTPUT_PATH == (
        module.PROJECT_ROOT
        / "docs"
        / "multimodal-scenarios.md"
    )


def test_read_jsonl_returns_objects_and_ignores_blank_lines(
    tmp_path: Path,
) -> None:
    """合法JSON对象应按原顺序返回，空行应被忽略。"""

    jsonl_path = tmp_path / "records.jsonl"
    jsonl_path.write_text(
        (
            '{"case_id":"first"}\n'
            "\n"
            '{"case_id":"second"}\n'
        ),
        encoding="utf-8",
    )

    records = module.read_jsonl(
        jsonl_path
    )

    assert records == (
        {"case_id": "first"},
        {"case_id": "second"},
    )


def test_read_jsonl_requires_path_object(
) -> None:
    """路径参数类型错误时应在文件操作前失败。"""

    with pytest.raises(
        TypeError,
        match="path必须是pathlib.Path",
    ):
        module.read_jsonl(  # type: ignore[arg-type]
            "records.jsonl"
        )


def test_read_jsonl_rejects_missing_file(
    tmp_path: Path,
) -> None:
    """不存在的JSONL文件应返回明确文件错误。"""

    missing_path = (
        tmp_path / "missing.jsonl"
    )

    with pytest.raises(
        FileNotFoundError,
        match="JSONL文件不存在",
    ):
        module.read_jsonl(
            missing_path
        )


def test_read_jsonl_reports_invalid_json_line_number(
    tmp_path: Path,
) -> None:
    """JSON语法损坏时应指出具体行号。"""

    jsonl_path = tmp_path / "invalid.jsonl"
    jsonl_path.write_text(
        '{"valid":true}\n{"broken":}\n',
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="line=2",
    ):
        module.read_jsonl(
            jsonl_path
        )


def test_read_jsonl_rejects_non_object_record(
    tmp_path: Path,
) -> None:
    """合法JSON数组不能冒充一条图片对象记录。"""

    jsonl_path = tmp_path / "array.jsonl"
    jsonl_path.write_text(
        '["not", "an", "object"]\n',
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="每一行必须是JSON对象",
    ):
        module.read_jsonl(
            jsonl_path
        )


def test_read_jsonl_rejects_empty_file(
    tmp_path: Path,
) -> None:
    """只有空白的JSONL不能形成有效评测输入。"""

    jsonl_path = tmp_path / "empty.jsonl"
    jsonl_path.write_text(
        "\n  \n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="JSONL没有可用记录",
    ):
        module.read_jsonl(
            jsonl_path
        )


def test_calculate_file_sha256_matches_raw_bytes(
    tmp_path: Path,
) -> None:
    """文件摘要必须根据未经文本解码的原始字节计算。"""

    binary_path = tmp_path / "image.bin"
    binary_path.write_bytes(
        TEST_BINARY_CONTENT
    )

    expected_digest = sha256(
        TEST_BINARY_CONTENT
    ).hexdigest()

    assert module.calculate_file_sha256(
        binary_path
    ) == expected_digest


def test_calculate_file_sha256_requires_path_object(
) -> None:
    """摘要函数应拒绝字符串路径以保持接口明确。"""

    with pytest.raises(
        TypeError,
        match="path必须是pathlib.Path",
    ):
        module.calculate_file_sha256(  # type: ignore[arg-type]
            "image.png"
        )


def test_calculate_file_sha256_rejects_empty_file(
    tmp_path: Path,
) -> None:
    """空图片不能被当成合法评测图片。"""

    empty_path = tmp_path / "empty.png"
    empty_path.write_bytes(b"")

    with pytest.raises(
        ValueError,
        match="图片文件不能为空",
    ):
        module.calculate_file_sha256(
            empty_path
        )


def test_resolve_project_image_path_stays_inside_data(
) -> None:
    """合法Gold路径应解析到项目data目录内部。"""

    resolved_path = (
        module.resolve_project_image_path(
            "data/multimodal/tool-selection/"
            "multimodal-001.png"
        )
    )

    assert resolved_path.is_relative_to(
        (module.PROJECT_ROOT / "data").resolve()
    )
    assert resolved_path.name == (
        "multimodal-001.png"
    )


def test_resolve_project_image_path_rejects_escape(
) -> None:
    """包含上级目录的路径解析后不能越出data目录。"""

    with pytest.raises(
        ValueError,
        match="超出项目data目录",
    ):
        module.resolve_project_image_path(
            "../outside.png"
        )


def test_load_reusable_images_returns_six_valid_records(
) -> None:
    """真实离线清单应产生六条已校验图片索引。"""

    images = module.load_reusable_images()

    assert tuple(images) == (
        module.EXPECTED_SOURCE_IMAGE_IDS
    )
    assert len(images) == 6
    assert all(
        isinstance(
            image,
            MultimodalEvaluationImage,
        )
        for image in images.values()
    )
    assert images[
        "multimodal-001"
    ].analysis_goal == (
        module.IMAGE_ANALYSIS_GOALS[
            "multimodal-001"
        ]
    )


def test_load_reusable_images_rejects_duplicate_case_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """重复图片ID会破坏稳定索引，必须拒绝。"""

    records = load_real_source_records()
    records.append(
        deepcopy(records[0])
    )

    use_temporary_manifest(
        monkeypatch=monkeypatch,
        path=tmp_path / "duplicate.jsonl",
        records=records,
    )

    with pytest.raises(
        ValueError,
        match="重复case_id",
    ):
        module.load_reusable_images()


def test_load_reusable_images_rejects_missing_required_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """固定白名单中的任一图片缺失时不能生成场景。"""

    records = [
        record
        for record in load_real_source_records()
        if record.get("case_id")
        != "multimodal-006"
    ]

    use_temporary_manifest(
        monkeypatch=monkeypatch,
        path=tmp_path / "missing-id.jsonl",
        records=records,
    )

    with pytest.raises(
        ValueError,
        match="multimodal-006",
    ):
        module.load_reusable_images()


def test_load_reusable_images_rejects_invalid_case_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """case_id缺失或不是字符串时应立即拒绝。"""

    records = load_real_source_records()
    records[0]["case_id"] = None

    use_temporary_manifest(
        monkeypatch=monkeypatch,
        path=tmp_path / "invalid-id.jsonl",
        records=records,
    )

    with pytest.raises(
        ValueError,
        match="缺少合法case_id",
    ):
        module.load_reusable_images()


def test_load_reusable_images_rejects_invalid_image_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """图片路径不是字符串时不能交给Schema。"""

    records = load_real_source_records()
    records[0]["image_path"] = None

    use_temporary_manifest(
        monkeypatch=monkeypatch,
        path=tmp_path / "invalid-path.jsonl",
        records=records,
    )

    with pytest.raises(
        ValueError,
        match="缺少合法image_path",
    ):
        module.load_reusable_images()


def test_load_reusable_images_rejects_hash_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """清单摘要与真实图片不一致时必须停止生成。"""

    records = load_real_source_records()
    records[0]["image_sha256"] = (
        "0" * 64
    )

    use_temporary_manifest(
        monkeypatch=monkeypatch,
        path=tmp_path / "wrong-hash.jsonl",
        records=records,
    )

    with pytest.raises(
        ValueError,
        match="图片SHA-256不一致",
    ):
        module.load_reusable_images()


def test_stable_unique_preserves_first_occurrence_order(
) -> None:
    """公共去重函数应保留第一次出现的稳定顺序。"""

    assert module.stable_unique(
        (
            "vision",
            "knowledge",
            "vision",
            "telemetry",
        )
    ) == (
        "vision",
        "knowledge",
        "telemetry",
    )


def test_select_images_rejects_unknown_image_id(
) -> None:
    """场景引用白名单以外的图片编号时必须失败。"""

    images = module.load_reusable_images()

    with pytest.raises(
        ValueError,
        match="场景引用了未知图片",
    ):
        module.select_images(
            reusable_images=images,
            image_ids=(
                "multimodal-999",
            ),
        )


def test_build_normal_scenarios_creates_eight_completed_cases(
) -> None:
    """正常组应有8条可完成且有知识证据的多模态场景。"""

    scenarios = (
        module.build_normal_scenarios(
            module.load_reusable_images()
        )
    )

    assert len(scenarios) == 8
    assert all(
        scenario.category
        == "normal_image"
        for scenario in scenarios
    )
    assert all(
        scenario.expected_vision_call
        for scenario in scenarios
    )
    assert all(
        scenario.expects_task_completion
        for scenario in scenarios
    )
    assert all(
        scenario.expected_evidence
        for scenario in scenarios
    )


def test_build_noisy_scenarios_separates_refusal_and_partial_cases(
) -> None:
    """低质量组前4条拒答、后4条保留知识部分结果。"""

    scenarios = (
        module.build_noisy_scenarios(
            module.load_reusable_images()
        )
    )

    assert len(scenarios) == 8
    assert all(
        scenario.category
        == "noisy_or_low_quality"
        for scenario in scenarios
    )
    assert all(
        "do_not_guess_unreadable_content"
        in scenario.safety_requirements
        for scenario in scenarios
    )

    refusal_cases = scenarios[:4]
    partial_cases = scenarios[4:]

    assert all(
        scenario.expects_safe_refusal
        and not scenario.expected_evidence
        and scenario.expected_diagnosis_statuses
        == ("abstained",)
        for scenario in refusal_cases
    )
    assert all(
        not scenario.expects_safe_refusal
        and bool(scenario.expected_evidence)
        and scenario.expected_diagnosis_statuses
        == ("partial",)
        for scenario in partial_cases
    )


def test_build_missing_scenarios_creates_six_safe_refusals(
) -> None:
    """缺失组必须全部拒答，并区分需要与不需要Vision的情况。"""

    scenarios = (
        module.build_missing_scenarios(
            module.load_reusable_images()
        )
    )

    assert len(scenarios) == 6
    assert all(
        scenario.expects_safe_refusal
        for scenario in scenarios
    )
    assert all(
        scenario.expected_diagnosis_statuses
        == ("abstained",)
        for scenario in scenarios
    )
    assert sum(
        scenario.expected_vision_call
        for scenario in scenarios
    ) == 2


def test_build_conflict_scenarios_never_require_completed_diagnosis(
) -> None:
    """四条图文冲突记录都必须保留冲突并避免完整诊断。"""

    scenarios = (
        module.build_conflict_scenarios(
            module.load_reusable_images()
        )
    )

    assert len(scenarios) == 4
    assert all(
        scenario.image_log_relationship
        == "conflicting"
        for scenario in scenarios
    )
    assert all(
        scenario.expected_diagnosis_statuses
        != ("completed",)
        for scenario in scenarios
    )
    assert all(
        not scenario.expects_task_completion
        for scenario in scenarios
    )


def test_build_adversarial_scenarios_preserves_safety_boundary(
) -> None:
    """对抗和高风险组必须安全拒答并具有专项安全规则。"""

    scenarios = (
        module.build_adversarial_scenarios(
            module.load_reusable_images()
        )
    )

    specialized_requirements = {
        "ignore_image_instructions",
        "do_not_access_image_links",
        "do_not_expose_sensitive_content",
        "do_not_control_robot",
        "require_qualified_person_for_high_risk",
    }

    assert len(scenarios) == 4
    assert all(
        scenario.expects_safe_refusal
        for scenario in scenarios
    )
    assert all(
        set(scenario.safety_requirements)
        & specialized_requirements
        for scenario in scenarios
    )
    assert all(
        "control_robot"
        in scenario.forbidden_tools
        for scenario in scenarios
    )


def test_build_all_scenarios_has_continuous_ids_and_required_counts(
) -> None:
    """完整评测集必须是001到030及固定的五类数量。"""

    scenarios = module.build_all_scenarios()

    assert len(scenarios) == 30
    assert tuple(
        scenario.scenario_id
        for scenario in scenarios
    ) == tuple(
        f"multimodal-agent-{index:03d}"
        for index in range(1, 31)
    )

    actual_counts = {
        category: sum(
            scenario.category == category
            for scenario in scenarios
        )
        for category in (
            module.EXPECTED_CATEGORY_COUNTS
        )
    }

    assert actual_counts == (
        module.EXPECTED_CATEGORY_COUNTS
    )


def test_all_scenarios_use_minimum_tool_allowlist_and_order(
) -> None:
    """允许工具应等于必需工具，核心顺序只能引用必需工具。"""

    for scenario in (
        module.build_all_scenarios()
    ):
        assert (
            scenario.allowed_tools
            == scenario.required_tools
        )
        assert set(
            scenario.required_tool_order
        ).issubset(
            set(scenario.required_tools)
        )


def test_all_scenarios_derive_information_sources_from_tools(
) -> None:
    """Vision、遥测和知识证据必须具有对应来源标签。"""

    for scenario in (
        module.build_all_scenarios()
    ):
        sources = set(
            scenario
            .expected_information_sources
        )

        assert "user_report" in sources
        assert "log_excerpt" in sources
        assert (
            "vision_model" in sources
        ) is scenario.expected_vision_call
        assert (
            "simulated_memory" in sources
        ) is (
            "get_robot_telemetry"
            in scenario.required_tools
        )
        assert (
            "knowledge_base" in sources
        ) is bool(
            scenario.expected_evidence
        )


def test_serialized_scenario_set_contains_no_base64_images(
) -> None:
    """30条Gold记录序列化后仍然只能保存图片路径和摘要。"""

    serialized = "\n".join(
        scenario.model_dump_json()
        for scenario in (
            module.build_all_scenarios()
        )
    )

    assert "image_base64" not in serialized
    assert "data:image/" not in serialized
    assert "iVBOR" not in serialized


def test_serialize_scenarios_jsonl_round_trips_all_records(
) -> None:
    """JSONL每行应能重新构造完全相同的Pydantic场景。"""

    scenarios = module.build_all_scenarios()
    serialized = (
        module.serialize_scenarios_jsonl(
            scenarios
        )
    )

    lines = serialized.splitlines()

    assert len(lines) == 30

    restored = tuple(
        module.MultimodalAgentEvaluationScenario
        .model_validate_json(line)
        for line in lines
    )

    assert restored == scenarios


def test_serialize_scenarios_jsonl_rejects_empty_tuple(
) -> None:
    """空数据集不能覆盖已有正式评测文件。"""

    with pytest.raises(
        ValueError,
        match="scenarios不能为空",
    ):
        module.serialize_scenarios_jsonl(
            ()
        )


def test_write_scenario_manifest_creates_parent_and_valid_jsonl(
    tmp_path: Path,
) -> None:
    """写出函数应创建父目录并生成30行UTF-8 JSONL。"""

    scenarios = module.build_all_scenarios()
    output_path = (
        tmp_path
        / "nested"
        / "multimodal_scenarios.jsonl"
    )

    returned_path = (
        module.write_scenario_manifest(
            scenarios=scenarios,
            output_path=output_path,
        )
    )

    assert returned_path == output_path
    assert output_path.is_file()
    assert len(
        module.read_jsonl(output_path)
    ) == 30


def test_markdown_cell_escapes_table_separator_and_newline(
) -> None:
    """动态文字不能破坏Markdown表格列和行结构。"""

    assert module.markdown_cell(
        "first|second\nthird"
    ) == "first\\|second third"


def test_render_documentation_contains_required_sections_and_no_time(
) -> None:
    """字段说明应覆盖数量、字段、图片、安全边界和场景目录。"""

    markdown = (
        module.render_documentation(
            module.build_all_scenarios()
        )
    )

    assert "# 多模态 Agent 评测场景说明" in markdown
    assert "## 2. 类别数量" in markdown
    assert "场景总数：30" in markdown
    assert "## 3. 主要字段" in markdown
    assert "## 4. 图片样本与生成方法" in markdown
    assert "## 5. 安全和可信边界" in markdown
    assert "multimodal-agent-001" in markdown
    assert "multimodal-agent-030" in markdown
    assert "生成时间" not in markdown
    assert "generated_at" not in markdown


def test_write_documentation_creates_markdown_file(
    tmp_path: Path,
) -> None:
    """文档写出函数应创建目录并保存完整Markdown。"""

    output_path = (
        tmp_path
        / "docs"
        / "multimodal-scenarios.md"
    )

    returned_path = (
        module.write_documentation(
            scenarios=(
                module.build_all_scenarios()
            ),
            output_path=output_path,
        )
    )

    assert returned_path == output_path
    assert output_path.is_file()
    assert output_path.read_text(
        encoding="utf-8"
    ).startswith(
        "# 多模态 Agent 评测场景说明"
    )


def test_validate_written_manifest_detects_different_expected_order(
    tmp_path: Path,
) -> None:
    """磁盘记录与内存Gold顺序不同也应使回环校验失败。"""

    scenarios = module.build_all_scenarios()
    output_path = (
        tmp_path / "scenarios.jsonl"
    )

    module.write_scenario_manifest(
        scenarios=scenarios,
        output_path=output_path,
    )

    with pytest.raises(
        ValueError,
        match="回环校验失败",
    ):
        module.validate_written_manifest(
            path=output_path,
            expected_scenarios=(
                *scenarios[1:],
                scenarios[0],
            ),
        )


def test_generate_outputs_executes_complete_local_flow(
    tmp_path: Path,
) -> None:
    """总入口应生成JSONL、文档并通过回环校验。"""

    manifest_path = (
        tmp_path / "data" / "scenarios.jsonl"
    )
    documentation_path = (
        tmp_path / "docs" / "scenarios.md"
    )

    (
        scenarios,
        returned_manifest_path,
        returned_documentation_path,
    ) = module.generate_outputs(
        scenario_output_path=manifest_path,
        documentation_output_path=(
            documentation_path
        ),
    )

    assert len(scenarios) == 30
    assert returned_manifest_path == (
        manifest_path
    )
    assert returned_documentation_path == (
        documentation_path
    )
    assert manifest_path.is_file()
    assert documentation_path.is_file()


def test_print_summary_reports_all_category_counts(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """终端摘要应显示五类数量和两份输出路径。"""

    module.print_summary(
        scenarios=(
            module.build_all_scenarios()
        ),
        manifest_path=(
            tmp_path / "scenarios.jsonl"
        ),
        documentation_path=(
            tmp_path / "scenarios.md"
        ),
    )

    output = capsys.readouterr().out

    assert "多模态Agent评测集生成完成" in output
    assert "场景数：30" in output
    assert "正常图片：8" in output
    assert "噪声/低质量：8" in output
    assert "缺失或无法回答：6" in output
    assert "图片与日志冲突：4" in output
    assert "提示注入或高风险：4" in output
    assert "scenarios.jsonl" in output
    assert "scenarios.md" in output
