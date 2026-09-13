"""Week 6 多模态离线回归夹具 Schema 的离线测试。

被测试模块：
app.schemas.multimodal_offline_regression。

预期调用链：

测试或离线回归脚本
→ 构造 MultimodalOfflineScenarioFixture
→ Pydantic 校验 Planner、Vision 和工具 Fake 结果
→ 拒绝不完整、重复或互相矛盾的离线脚本
→ 将合法夹具交给后续离线执行器。

本测试不调用真实 LLM、Vision、Embedding、Chroma 或 HTTP 服务。
"""

from typing import Any

import pytest
from pydantic import ValidationError

from app.schemas.agent_planning import (
    AgentPlannerDecision,
)
from app.schemas.multimodal_offline_regression import (
    MULTIMODAL_OFFLINE_FIXTURE_VERSION,
    MultimodalOfflineScenarioFixture,
    OfflinePlannerTurn,
    OfflineToolOutcome,
    OfflineVisionOutcome,
)
from app.schemas.vision import (
    VisionModelDraft,
)


# 使用稳定的 64 位十六进制摘要模拟图片 SHA-256。
TEST_IMAGE_SHA256 = "a" * 64


def make_tool_decision(
    *,
    call_id: str,
    tool_name: str,
    arguments: dict[str, Any],
) -> AgentPlannerDecision:
    """创建一轮合法的工具调用决定。"""

    return AgentPlannerDecision(
        decision="call_tool",
        tool_call={
            "call_id": call_id,
            "tool_name": tool_name,
            "arguments": arguments,
        },
    )


def make_finish_decision() -> AgentPlannerDecision:
    """创建一轮合法的完成决定及最小诊断草稿。"""

    return AgentPlannerDecision(
        decision="finish",
        finish_reason="task_completed",
        final_message=(
            '{"status":"completed",'
            '"possible_causes":[],"next_checks":[],'
            '"risk_level":"low",'
            '"missing_information":[],"abstained":false}'
        ),
    )


def make_vision_draft() -> VisionModelDraft:
    """创建 Fake Vision 可以返回的最小结构化观察。"""

    return VisionModelDraft(
        status="completed",
        image_quality="clear",
        observations=(
            {
                "description": "面板显示 ERR-NET-4001",
                "category": "visible_text",
                "confidence": "high",
            },
        ),
        visible_indicators=(),
        uncertain_items=(),
        requires_human_check=False,
        human_check_reasons=(),
    )


def make_valid_fixture() -> MultimodalOfflineScenarioFixture:
    """创建 Vision、知识检索和结束三轮组成的合法夹具。"""

    return MultimodalOfflineScenarioFixture(
        scenario_id="multimodal-agent-001",
        planner_turns=(
            OfflinePlannerTurn(
                decision=make_tool_decision(
                    call_id="call_vision_001",
                    tool_name="analyze_robot_image",
                    arguments={
                        "image_ref": "image_001",
                        "analysis_goal": "读取网络面板状态",
                    },
                ),
            ),
            OfflinePlannerTurn(
                decision=make_tool_decision(
                    call_id="call_search_001",
                    tool_name="search_knowledge",
                    arguments={
                        "query": "ERR-NET-4001 恢复条件",
                        "top_k": 3,
                    },
                ),
            ),
            OfflinePlannerTurn(
                decision=make_finish_decision(),
            ),
        ),
        vision_outcomes=(
            OfflineVisionOutcome(
                image_sha256=TEST_IMAGE_SHA256,
                draft=make_vision_draft(),
            ),
        ),
        tool_outcomes=(
            OfflineToolOutcome(
                call_id="call_search_001",
                tool_name="search_knowledge",
                status="success",
                output={
                    "answer": "需要状态核对后恢复",
                    "citations": [],
                    "retrieval_ms": 1.0,
                    "abstained": False,
                },
            ),
        ),
    )


def test_valid_fixture_preserves_version_and_nested_contracts() -> None:
    """合法夹具应保留版本、Planner 决定和 Fake 结果。"""

    fixture = make_valid_fixture()

    assert fixture.fixture_version == (
        MULTIMODAL_OFFLINE_FIXTURE_VERSION
    )
    assert fixture.scenario_id == "multimodal-agent-001"
    assert len(fixture.planner_turns) == 3
    assert fixture.planner_turns[0].decision is not None
    assert fixture.planner_turns[0].decision.decision == "call_tool"
    assert fixture.vision_outcomes[0].draft is not None
    assert fixture.tool_outcomes[0].status == "success"


def test_policy_terminated_fixture_can_have_no_planner_turns() -> None:
    """Planner 前安全结束的场景允许不配置任何 Fake 调用。"""

    fixture = MultimodalOfflineScenarioFixture(
        scenario_id="multimodal-agent-027",
        planner_turns=(),
        vision_outcomes=(),
        tool_outcomes=(),
    )

    assert fixture.planner_turns == ()
    assert fixture.vision_outcomes == ()
    assert fixture.tool_outcomes == ()


def test_planner_turn_requires_exactly_one_outcome() -> None:
    """一轮 Planner 只能返回决定或失败，不能同时存在。"""

    with pytest.raises(
        ValidationError,
        match="decision和failure必须且只能设置一个",
    ):
        OfflinePlannerTurn(
            decision=make_finish_decision(),
            failure="timeout",
        )


def test_vision_outcome_requires_exactly_one_outcome() -> None:
    """Fake Vision 必须在结构化草稿与失败之间二选一。"""

    with pytest.raises(
        ValidationError,
        match="draft和failure必须且只能设置一个",
    ):
        OfflineVisionOutcome(
            image_sha256=TEST_IMAGE_SHA256,
        )


@pytest.mark.parametrize(
    ("status", "output"),
    [
        ("success", None),
        ("empty", {"unexpected": True}),
        ("timeout", {"unexpected": True}),
        ("error", {"unexpected": True}),
    ],
)
def test_tool_outcome_status_must_match_output(
    status: str,
    output: dict[str, Any] | None,
) -> None:
    """成功必须有输出，空结果和失败不得伪造业务输出。"""

    with pytest.raises(
        ValidationError,
        match="status与output不一致",
    ):
        OfflineToolOutcome(
            call_id="call_search_001",
            tool_name="search_knowledge",
            status=status,
            output=output,
        )


def test_fixture_rejects_duplicate_planner_call_ids() -> None:
    """同一场景不能让 Planner 重复使用同一个调用 ID。"""

    valid = make_valid_fixture()
    duplicated_turn = OfflinePlannerTurn(
        decision=make_tool_decision(
            call_id="call_search_001",
            tool_name="search_knowledge",
            arguments={"query": "另一条查询"},
        ),
    )

    with pytest.raises(
        ValidationError,
        match="planner_turns不能包含重复call_id",
    ):
        MultimodalOfflineScenarioFixture(
            scenario_id=valid.scenario_id,
            planner_turns=(
                *valid.planner_turns[:-1],
                duplicated_turn,
                valid.planner_turns[-1],
            ),
            vision_outcomes=valid.vision_outcomes,
            tool_outcomes=valid.tool_outcomes,
        )


def test_fixture_requires_terminal_last_planner_turn() -> None:
    """非空 Planner 脚本必须以 finish 或显式失败终止。"""

    with pytest.raises(
        ValidationError,
        match="最后一轮必须结束规划或模拟Planner失败",
    ):
        MultimodalOfflineScenarioFixture(
            scenario_id="multimodal-agent-001",
            planner_turns=(
                OfflinePlannerTurn(
                    decision=make_tool_decision(
                        call_id="call_search_001",
                        tool_name="search_knowledge",
                        arguments={"query": "ERR-NET-4001"},
                    ),
                ),
            ),
            tool_outcomes=(
                OfflineToolOutcome(
                    call_id="call_search_001",
                    tool_name="search_knowledge",
                    status="empty",
                ),
            ),
        )


def test_fixture_rejects_turns_after_terminal_outcome() -> None:
    """finish 或 Planner 失败之后不能再配置后续轮次。"""

    with pytest.raises(
        ValidationError,
        match="终止Planner轮次必须位于最后",
    ):
        MultimodalOfflineScenarioFixture(
            scenario_id="multimodal-agent-001",
            planner_turns=(
                OfflinePlannerTurn(
                    decision=make_finish_decision(),
                ),
                OfflinePlannerTurn(
                    failure="error",
                ),
            ),
        )


def test_fixture_requires_matching_non_vision_tool_outcome() -> None:
    """每次非 Vision 工具调用必须有且只有一个对应 Fake 结果。"""

    valid = make_valid_fixture()

    with pytest.raises(
        ValidationError,
        match="按顺序逐项对应",
    ):
        MultimodalOfflineScenarioFixture(
            scenario_id=valid.scenario_id,
            planner_turns=valid.planner_turns,
            vision_outcomes=valid.vision_outcomes,
            tool_outcomes=(),
        )


def test_fixture_requires_non_vision_outcomes_in_planner_order() -> None:
    """工具结果集合相同但顺序相反时也必须拒绝。"""

    first_decision = AgentPlannerDecision(
        decision="call_tool",
        tool_call={
            "call_id": "call_search_001",
            "tool_name": "search_knowledge",
            "arguments": {
                "query": "ERR-NET-4001",
            },
        },
    )
    second_decision = AgentPlannerDecision(
        decision="call_tool",
        tool_call={
            "call_id": "call_telemetry_001",
            "tool_name": "get_robot_telemetry",
            "arguments": {
                "robot_id": "robot-001",
            },
        },
    )

    with pytest.raises(
        ValidationError,
        match="按顺序逐项对应",
    ):
        MultimodalOfflineScenarioFixture(
            scenario_id="multimodal-agent-001",
            planner_turns=(
                OfflinePlannerTurn(
                    decision=first_decision,
                ),
                OfflinePlannerTurn(
                    decision=second_decision,
                ),
                OfflinePlannerTurn(
                    decision=make_finish_decision(),
                ),
            ),
            tool_outcomes=(
                # 结果集合正确，但顺序与Planner相反。
                OfflineToolOutcome(
                    call_id="call_telemetry_001",
                    tool_name="get_robot_telemetry",
                    status="empty",
                ),
                OfflineToolOutcome(
                    call_id="call_search_001",
                    tool_name="search_knowledge",
                    status="empty",
                ),
            ),
        )


def test_fixture_rejects_tool_name_mismatch_for_same_call_id() -> None:
    """Fake 工具结果的工具名必须与 Planner 请求一致。"""

    valid = make_valid_fixture()

    with pytest.raises(
        ValidationError,
        match="按顺序逐项对应",
    ):
        MultimodalOfflineScenarioFixture(
            scenario_id=valid.scenario_id,
            planner_turns=valid.planner_turns,
            vision_outcomes=valid.vision_outcomes,
            tool_outcomes=(
                OfflineToolOutcome(
                    call_id="call_search_001",
                    tool_name="get_robot_telemetry",
                    status="empty",
                ),
            ),
        )


def test_fixture_requires_vision_outcome_for_vision_call() -> None:
    """包含视觉调用的脚本必须为 Fake Vision 提供结果。"""

    valid = make_valid_fixture()

    with pytest.raises(
        ValidationError,
        match="Vision工具调用必须配置vision_outcomes",
    ):
        MultimodalOfflineScenarioFixture(
            scenario_id=valid.scenario_id,
            planner_turns=valid.planner_turns,
            vision_outcomes=(),
            tool_outcomes=valid.tool_outcomes,
        )


def test_fixture_rejects_unreferenced_vision_outcome() -> None:
    """没有视觉调用时不能留下不会被消费的 Fake Vision 结果。"""

    with pytest.raises(
        ValidationError,
        match="没有Vision调用时不能配置vision_outcomes",
    ):
        MultimodalOfflineScenarioFixture(
            scenario_id="multimodal-agent-021",
            planner_turns=(
                OfflinePlannerTurn(
                    decision=make_finish_decision(),
                ),
            ),
            vision_outcomes=(
                OfflineVisionOutcome(
                    image_sha256=TEST_IMAGE_SHA256,
                    draft=make_vision_draft(),
                ),
            ),
        )


def test_fixture_rejects_duplicate_vision_image_hashes() -> None:
    """同一图片摘要不能配置两个互相竞争的 Vision 结果。"""

    valid = make_valid_fixture()

    with pytest.raises(
        ValidationError,
        match="vision_outcomes不能包含重复image_sha256",
    ):
        MultimodalOfflineScenarioFixture(
            scenario_id=valid.scenario_id,
            planner_turns=valid.planner_turns,
            vision_outcomes=(
                *valid.vision_outcomes,
                *valid.vision_outcomes,
            ),
            tool_outcomes=valid.tool_outcomes,
        )
