"""30 条多模态 Agent 离线 Fixture 生成脚本的测试。

被测试模块：

scripts.build_multimodal_offline_fixtures

预期流程：

固定 Fixture 计划
→ 构造 Fake Planner、Vision 和工具结果
→ Pydantic 校验跨对象对应关系
→ 写入 UTF-8 JSONL
→ 正式 Fixture Loader 重新读取。

本文件不调用真实 LLM、Vision、Embedding、ChromaDB、
FastAPI 或机器人设备。
"""

import json
import sys

from pathlib import Path

import pytest

from app.services.multimodal_offline_fixture_loader import (
    load_multimodal_offline_fixtures,
)
from scripts.build_multimodal_offline_fixtures import (
    DEFAULT_OUTPUT_PATH,
    FIXTURE_PLANS,
    build_all_fixtures,
    build_fixture,
    parse_args,
    write_fixtures,
)


# Week 5 Gold 数据固定包含 001 到 030 共 30 个场景。
# 显式列出编号，可以同时发现漏项、重复项和意外新增项。
EXPECTED_SCENARIO_IDS = tuple(
    f"multimodal-agent-{index:03d}"
    for index in range(1, 31)
)


# 这五类请求应由请求级 Python 策略在 Planner 前结束。
# 它们的 Fixture 不应伪造 Planner、Vision 或工具调用。
PRETERMINATED_SCENARIO_IDS = {
    "multimodal-agent-017",
    "multimodal-agent-018",
    "multimodal-agent-027",
    "multimodal-agent-028",
    "multimodal-agent-030",
}


def test_build_all_fixtures_covers_exactly_thirty_scenarios(
) -> None:
    """生成器必须完整且只覆盖 30 个固定场景。

    被测试方法：build_all_fixtures()。

    预期返回结果是不可重复、按编号排列的 30 个
    MultimodalOfflineScenarioFixture，而不是未经校验的字典。
    """

    fixtures = build_all_fixtures()

    assert len(fixtures) == 30
    assert tuple(
        fixture.scenario_id
        for fixture in fixtures
    ) == EXPECTED_SCENARIO_IDS
    assert tuple(FIXTURE_PLANS) == (
        EXPECTED_SCENARIO_IDS
    )


def test_generated_fixture_parts_match_planner_calls(
) -> None:
    """Planner 调用必须和 Vision、普通工具结果逐项对应。

    选用 agent-003，因为它同时包含 Vision、知识检索和
    测试草案三类工具，可以覆盖最完整的生成路径。

    预期流程：三个 call_tool 决定 → 一个 finish 决定；
    Vision 结果按图片摘要保存，两个非 Vision 工具结果
    按 call_id 和 tool_name 与 Planner 调用顺序对应。
    """

    fixture = build_fixture(
        "multimodal-agent-003",
        FIXTURE_PLANS[
            "multimodal-agent-003"
        ],
    )

    planner_calls = tuple(
        turn.decision.tool_call
        for turn in fixture.planner_turns
        if (
            turn.decision is not None
            and turn.decision.decision
            == "call_tool"
        )
    )

    assert len(fixture.planner_turns) == 4
    assert tuple(
        call.tool_name
        for call in planner_calls
        if call is not None
    ) == (
        "analyze_robot_image",
        "search_knowledge",
        "draft_test_case",
    )
    assert len(fixture.vision_outcomes) == 1
    assert tuple(
        (
            outcome.call_id,
            outcome.tool_name,
        )
        for outcome in fixture.tool_outcomes
    ) == tuple(
        (
            call.call_id,
            call.tool_name,
        )
        for call in planner_calls
        if (
            call is not None
            and call.tool_name
            != "analyze_robot_image"
        )
    )
    assert (
        fixture.planner_turns[-1]
        .decision is not None
    )
    assert (
        fixture.planner_turns[-1]
        .decision.decision
        == "finish"
    )


@pytest.mark.parametrize(
    "scenario_id",
    sorted(PRETERMINATED_SCENARIO_IDS),
)
def test_preterminated_fixtures_do_not_fake_external_calls(
    scenario_id: str,
) -> None:
    """策略前终止场景不得配置不会发生的外部调用。

    build_fixture() 收到 preterminated 计划后，应该返回
    只有场景编号和版本的空 Fixture。真实执行器随后会让
    请求级策略直接形成拒答或人工审核结果。
    """

    fixture = build_fixture(
        scenario_id,
        FIXTURE_PLANS[scenario_id],
    )

    assert fixture.planner_turns == ()
    assert fixture.vision_outcomes == ()
    assert fixture.tool_outcomes == ()


def test_dm260_completed_draft_requires_qualified_person(
) -> None:
    """DM260 连接器场景必须保留高风险人员资质边界。

    测试解析 Fake Planner 最后一轮的 final_message，
    验证最终诊断草稿而不只是测试草案中包含安全要求。

    预期结果：检查项为 high，并明确设置
    requires_qualified_person=True，同时引用本轮真实
    Fake 检索结果中的 Chunk ID。
    """

    fixture = build_fixture(
        "multimodal-agent-007",
        FIXTURE_PLANS[
            "multimodal-agent-007"
        ],
    )
    final_decision = (
        fixture.planner_turns[-1].decision
    )

    assert final_decision is not None
    assert final_decision.final_message is not None

    payload = json.loads(
        final_decision.final_message
    )
    check = payload["next_checks"][0]

    assert payload["risk_level"] == "high"
    assert check["risk_level"] == "high"
    assert (
        check["requires_qualified_person"]
        is True
    )
    assert check["evidence_chunk_ids"]


def test_write_fixtures_is_deterministic_and_loader_compatible(
    tmp_path: Path,
) -> None:
    """相同 Fixture 必须产生相同且可重新加载的 JSONL。

    tmp_path 是 pytest 提供的临时目录 Fixture；它负责为
    本测试创建隔离目录，并会在测试结束后清理。

    测试把相同对象写入两个文件，然后比较完整字节内容，
    最后使用正式 Loader 读取其中一份。
    """

    fixtures = build_all_fixtures()
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"

    write_fixtures(fixtures, first_path)
    write_fixtures(fixtures, second_path)

    first_content = first_path.read_text(
        encoding="utf-8"
    )
    second_content = second_path.read_text(
        encoding="utf-8"
    )

    assert first_content == second_content
    assert "generated_at" not in first_content
    assert "timestamp" not in first_content

    loaded = load_multimodal_offline_fixtures(
        first_path
    )

    assert len(loaded) == 30
    assert loaded[0].scenario_id == (
        "multimodal-agent-001"
    )
    assert loaded[-1].scenario_id == (
        "multimodal-agent-030"
    )


def test_parse_args_uses_documented_default_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """未传 --output 时必须使用项目约定的默认路径。

    monkeypatch 是 pytest 的内置 Fixture。这里用它临时
    替换 sys.argv，测试结束后 pytest 会自动恢复原值。
    """

    monkeypatch.setattr(
        sys,
        "argv",
        ["build_multimodal_offline_fixtures.py"],
    )

    args = parse_args()

    assert args.output == DEFAULT_OUTPUT_PATH


def test_parse_args_preserves_explicit_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """显式 --output 必须转换成 Path 并原样保留。"""

    expected_path = Path(
        "docs/custom-offline-fixtures.jsonl"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_multimodal_offline_fixtures.py",
            "--output",
            str(expected_path),
        ],
    )

    args = parse_args()

    assert args.output == expected_path


def test_build_fixture_rejects_unknown_tool_name(
) -> None:
    """Fixture 计划不能静默接受未实现的工具。

    测试传入 unknown_tool。预期 _tool_arguments() 明确
    抛出 ValueError，避免拼写错误生成无法执行的回归数据。
    """

    invalid_plan = {
        "tools": ("unknown_tool",),
        "status": "completed",
        "finish": "task_completed",
        "robot": "robot-001",
    }

    with pytest.raises(
        ValueError,
        match="Fixture计划包含未知工具",
    ):
        build_fixture(
            "multimodal-agent-001",
            invalid_plan,
        )
