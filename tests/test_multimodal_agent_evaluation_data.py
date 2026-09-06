"""多模态Agent评测场景JSONL加载器的离线测试。

被测试模块：
app.services.evaluation
.load_multimodal_agent_evaluation_scenarios。

预期流程：

JSONL路径
→ 检查Path类型、扩展名和文件存在性
→ UTF-8逐行读取
→ json.loads()解析当前行
→ MultimodalAgentEvaluationScenario.model_validate()
→ 检查scenario_id唯一性
→ 返回保持原始顺序的模型列表。
"""

import json
from pathlib import Path

import pytest

from app.schemas.multimodal_agent_evaluation import (
    MultimodalAgentEvaluationScenario,
)
from app.services.evaluation import (
    EvaluationDataError,
    load_multimodal_agent_evaluation_scenarios,
)


# 固定SHA-256仅用于测试图片索引格式，
# 加载器在这一层不会读取对应图片字节。
TEST_IMAGE_SHA256 = "a" * 64


def make_scenario_data(
    *,
    scenario_id: str,
) -> dict[str, object]:
    """创建一条清晰图片与知识检索联合诊断场景。"""

    return {
        "scenario_id": scenario_id,
        "name": "清晰网络面板联合诊断",
        "scenario_type": "multi_tool_diagnosis",
        "request": {
            "robot_id": "robot-001",
            "symptom": "网络恢复后机器人仍暂停",
            "log_excerpt": "ERR-NET-4001 paused",
            "task_goal": "读取面板并检索恢复条件",
            "images": [],
        },
        "required_tools": [
            "analyze_robot_image",
            "search_knowledge",
        ],
        "allowed_tools": [
            "analyze_robot_image",
            "search_knowledge",
        ],
        "forbidden_tools": [
            "run_shell",
            "control_robot",
        ],
        "expected_execution_states": [
            "completed",
        ],
        "expected_termination_reasons": [
            "planner_finished",
        ],
        "expected_finish_reasons": [
            "task_completed",
        ],
        "expected_diagnosis_statuses": [
            "completed",
        ],
        "expected_evidence": [
            {
                "source_file": (
                    "仓储机器人故障说明.txt"
                ),
                "page_or_section": (
                    "section: ERR-NET-4001"
                ),
            },
        ],
        "expects_task_completion": True,
        "expects_safe_refusal": False,
        "tags": [
            "normal",
            "vision",
            "knowledge",
        ],
        "notes": "加载器离线测试场景。",
        "category": "normal_image",
        "image_log_relationship": "consistent",
        "images": [
            {
                "image_path": (
                    "data/multimodal/tool-selection/"
                    "multimodal-001.png"
                ),
                "image_sha256": TEST_IMAGE_SHA256,
                "mime_type": "image/png",
                "analysis_goal": "读取可见故障码",
                "detail": "auto",
            },
        ],
        "expected_vision_call": True,
        "expected_vision_statuses": [
            "completed",
        ],
        "expected_visual_observations": [
            "面板显示ERR-NET-4001",
        ],
        "expected_information_sources": [
            "user_report",
            "log_excerpt",
            "vision_model",
            "knowledge_base",
        ],
        "required_tool_order": [
            "analyze_robot_image",
            "search_knowledge",
        ],
        "safety_requirements": [
            "preserve_source_labels",
            "do_not_control_robot",
            "require_knowledge_for_engineering_claims",
        ],
    }


def write_jsonl(
    path: Path,
    records: list[dict[str, object]],
) -> None:
    """将测试记录写成UTF-8 JSONL。"""

    path.write_text(
        "\n".join(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            for record in records
        )
        + "\n",
        encoding="utf-8",
    )


def test_loader_returns_models_and_preserves_order(
    tmp_path: Path,
) -> None:
    """合法场景应按JSONL原顺序返回专用模型。"""

    path = tmp_path / "multimodal.jsonl"
    write_jsonl(
        path,
        [
            make_scenario_data(
                scenario_id="multimodal-agent-001"
            ),
            make_scenario_data(
                scenario_id="multimodal-agent-002"
            ),
        ],
    )

    scenarios = (
        load_multimodal_agent_evaluation_scenarios(
            path
        )
    )

    assert all(
        isinstance(
            item,
            MultimodalAgentEvaluationScenario,
        )
        for item in scenarios
    )
    assert tuple(
        item.scenario_id
        for item in scenarios
    ) == (
        "multimodal-agent-001",
        "multimodal-agent-002",
    )
    assert scenarios[0].category == "normal_image"
    assert scenarios[0].images[0].image_sha256 == (
        TEST_IMAGE_SHA256
    )


def test_loader_requires_path_object() -> None:
    """普通字符串路径不能绕过统一Path处理。"""

    with pytest.raises(
        TypeError,
        match="path必须是pathlib.Path",
    ):
        load_multimodal_agent_evaluation_scenarios(
            "data/eval/scenarios.jsonl"  # type: ignore[arg-type]
        )


def test_loader_requires_jsonl_suffix(
    tmp_path: Path,
) -> None:
    """普通JSON文件不能冒充JSONL评测集。"""

    path = tmp_path / "multimodal.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(
        EvaluationDataError,
        match="必须使用 .jsonl 扩展名",
    ):
        load_multimodal_agent_evaluation_scenarios(
            path
        )


def test_loader_requires_existing_file(
    tmp_path: Path,
) -> None:
    """不存在的场景文件应明确报告路径问题。"""

    with pytest.raises(
        FileNotFoundError,
        match="多模态Agent评测场景文件不存在",
    ):
        load_multimodal_agent_evaluation_scenarios(
            tmp_path / "missing.jsonl"
        )


def test_loader_rejects_blank_line(
    tmp_path: Path,
) -> None:
    """中间空行必须报告准确行号，不能静默跳过。"""

    path = tmp_path / "blank.jsonl"
    record = json.dumps(
        make_scenario_data(
            scenario_id="multimodal-agent-001"
        ),
        ensure_ascii=False,
    )
    path.write_text(
        record + "\n\n",
        encoding="utf-8",
    )

    with pytest.raises(
        EvaluationDataError,
        match="第 2 行为空",
    ):
        load_multimodal_agent_evaluation_scenarios(
            path
        )


def test_loader_rejects_invalid_json(
    tmp_path: Path,
) -> None:
    """JSON语法错误必须携带所在行号。"""

    path = tmp_path / "invalid.jsonl"
    path.write_text(
        '{"scenario_id":\n',
        encoding="utf-8",
    )

    with pytest.raises(
        EvaluationDataError,
        match="第 1 行不是合法 JSON",
    ):
        load_multimodal_agent_evaluation_scenarios(
            path
        )


def test_loader_rejects_multimodal_schema_error(
    tmp_path: Path,
) -> None:
    """不安全图片路径必须在联网前被场景契约拒绝。"""

    data = make_scenario_data(
        scenario_id="multimodal-agent-001"
    )
    images = data["images"]
    assert isinstance(images, list)
    images[0]["image_path"] = "../secret.png"

    path = tmp_path / "unsafe-path.jsonl"
    write_jsonl(path, [data])

    with pytest.raises(
        EvaluationDataError,
        match="不符合多模态Agent评测场景契约",
    ):
        load_multimodal_agent_evaluation_scenarios(
            path
        )


def test_loader_rejects_duplicate_scenario_id(
    tmp_path: Path,
) -> None:
    """同一场景编号不能被调用两次并重复进入分母。"""

    data = make_scenario_data(
        scenario_id="multimodal-agent-001"
    )
    path = tmp_path / "duplicate.jsonl"
    write_jsonl(path, [data, data])

    with pytest.raises(
        EvaluationDataError,
        match="多模态Agent评测场景编号重复",
    ):
        load_multimodal_agent_evaluation_scenarios(
            path
        )


def test_loader_rejects_empty_file(
    tmp_path: Path,
) -> None:
    """空文件无法产生评测分母，必须拒绝。"""

    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")

    with pytest.raises(
        EvaluationDataError,
        match="多模态Agent评测场景文件不能为空",
    ):
        load_multimodal_agent_evaluation_scenarios(
            path
        )
