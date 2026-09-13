"""Week 6 离线回归 Fake 适配器的单元测试。

被测试模块：app.services.multimodal_offline_fakes。

预期调用链：

MultimodalOfflineScenarioFixture
→ ScriptedOfflinePlanner / ScriptedOfflineToolHandler
→ 现有 AgentRunner / ToolExecutor
→ 与真实 Provider 相同的结构化决定和工具行为。

测试只使用内存对象，不访问真实模型、数据库、网络或文件系统。
"""

from hashlib import sha256
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from app.errors import (
    InvalidLLMResponseError,
    InvalidVisionResponseError,
    LLMUpstreamError,
    VisionTimeoutError,
)
from app.schemas.agent_planning import (
    AgentPlannerDecision,
    AgentPlanningContext,
)
from app.schemas.multimodal_offline_regression import (
    MultimodalOfflineScenarioFixture,
    OfflinePlannerTurn,
    OfflineToolOutcome,
    OfflineVisionOutcome,
)
from app.schemas.vision import (
    VisionInput,
    VisionModelDraft,
)
from app.services.multimodal_offline_fakes import (
    ScriptedOfflinePlanner,
    ScriptedOfflineToolHandler,
    build_offline_fake_vision_provider,
)


class DemoToolInput(BaseModel):
    """模拟一个经过 ToolExecutor 校验后的工具输入。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    query: str


def make_context() -> AgentPlanningContext:
    """创建没有历史工具交互的首轮 Planner 上下文。"""

    return AgentPlanningContext(
        task="检查 ERR-NET-4001",
        next_step=1,
        interactions=(),
    )


def make_tool_decision() -> AgentPlannerDecision:
    """创建知识检索工具调用决定。"""

    return AgentPlannerDecision(
        decision="call_tool",
        tool_call={
            "call_id": "call_search_001",
            "tool_name": "search_knowledge",
            "arguments": {
                "query": "ERR-NET-4001",
            },
        },
    )


def make_finish_decision() -> AgentPlannerDecision:
    """创建合法 Planner 结束决定。"""

    return AgentPlannerDecision(
        decision="finish",
        finish_reason="insufficient_information",
        final_message=(
            '{"status":"abstained",'
            '"possible_causes":[],"next_checks":[],'
            '"risk_level":"unknown",'
            '"missing_information":["缺少证据"],'
            '"abstained":true}'
        ),
    )


def make_vision_draft() -> VisionModelDraft:
    """创建 Fake Vision 的合法结构化草稿。"""

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


def make_vision_input(
    image_bytes: bytes = b"offline-image",
) -> VisionInput:
    """根据真实测试字节构造内部 VisionInput。"""

    return VisionInput(
        metadata={
            "mime_type": "image/png",
            "size_bytes": len(image_bytes),
            "width_px": 1,
            "height_px": 1,
            "pixel_count": 1,
            "sha256_hex": sha256(
                image_bytes
            ).hexdigest(),
        },
        image_bytes=image_bytes,
        analysis_goal="读取面板故障码",
        detail="auto",
    )


@pytest.mark.asyncio
async def test_scripted_planner_returns_turns_in_order() -> None:
    """Fake Planner 应顺序返回决定并记录脱敏工具范围。"""

    planner = ScriptedOfflinePlanner(
        turns=(
            OfflinePlannerTurn(
                decision=make_tool_decision(),
            ),
            OfflinePlannerTurn(
                decision=make_finish_decision(),
            ),
        )
    )
    context = make_context()
    tool_schemas: tuple[dict[str, Any], ...] = (
        {
            "type": "function",
            "function": {
                "name": "search_knowledge",
            },
        },
    )

    first = await planner.plan(
        context=context,
        tool_schemas=tool_schemas,
    )
    second = await planner.plan(
        context=context,
        tool_schemas=(),
    )

    assert first.decision == "call_tool"
    assert second.decision == "finish"
    assert planner.call_count == 2
    assert planner.contexts == (
        context,
        context,
    )
    assert planner.exposed_tool_names == (
        ("search_knowledge",),
        (),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "expected_exception"),
    [
        ("timeout", TimeoutError),
        ("error", LLMUpstreamError),
        (
            "invalid_response",
            InvalidLLMResponseError,
        ),
    ],
)
async def test_scripted_planner_converts_failure_kind(
    failure: str,
    expected_exception: type[Exception],
) -> None:
    """每种 Planner Fake 失败应转换成稳定异常类型。"""

    planner = ScriptedOfflinePlanner(
        turns=(
            OfflinePlannerTurn(
                failure=failure,
            ),
        )
    )

    with pytest.raises(expected_exception):
        await planner.plan(
            context=make_context(),
            tool_schemas=(),
        )


@pytest.mark.asyncio
async def test_scripted_planner_rejects_exhausted_script() -> None:
    """脚本耗尽后继续规划应暴露夹具错误。"""

    planner = ScriptedOfflinePlanner(
        turns=(
            OfflinePlannerTurn(
                decision=make_finish_decision(),
            ),
        )
    )

    await planner.plan(
        context=make_context(),
        tool_schemas=(),
    )

    with pytest.raises(
        InvalidLLMResponseError,
        match="Fake Planner脚本已经耗尽",
    ):
        await planner.plan(
            context=make_context(),
            tool_schemas=(),
        )


@pytest.mark.asyncio
async def test_tool_handler_returns_success_and_empty_in_order() -> None:
    """Fake 工具应按队列返回成功数据和空结果。"""

    handler = ScriptedOfflineToolHandler(
        tool_name="search_knowledge",
        outcomes=(
            OfflineToolOutcome(
                call_id="call_search_001",
                tool_name="search_knowledge",
                status="success",
                output={"evidence_ids": ["chunk-001"]},
            ),
            OfflineToolOutcome(
                call_id="call_search_002",
                tool_name="search_knowledge",
                status="empty",
            ),
        ),
    )
    tool_input = DemoToolInput(
        query="ERR-NET-4001",
    )

    first = await handler(tool_input)
    second = await handler(tool_input)

    assert first == {
        "evidence_ids": ["chunk-001"],
    }
    assert second is None
    assert handler.call_count == 2
    assert handler.calls == (
        {"query": "ERR-NET-4001"},
        {"query": "ERR-NET-4001"},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "expected_exception"),
    [
        ("timeout", TimeoutError),
        ("error", RuntimeError),
    ],
)
async def test_tool_handler_converts_failure_status(
    status: str,
    expected_exception: type[Exception],
) -> None:
    """Fake 工具超时和异常应经过真实 Executor 的异常路径。"""

    handler = ScriptedOfflineToolHandler(
        tool_name="search_knowledge",
        outcomes=(
            OfflineToolOutcome(
                call_id="call_search_001",
                tool_name="search_knowledge",
                status=status,
            ),
        ),
    )

    with pytest.raises(expected_exception):
        await handler(
            DemoToolInput(query="ERR-NET-4001")
        )


def test_tool_handler_rejects_wrong_tool_outcome() -> None:
    """处理器只能消费属于自身工具名的结果。"""

    with pytest.raises(
        ValueError,
        match="outcomes中的tool_name必须全部匹配",
    ):
        ScriptedOfflineToolHandler(
            tool_name="search_knowledge",
            outcomes=(
                OfflineToolOutcome(
                    call_id="call_telemetry_001",
                    tool_name="get_robot_telemetry",
                    status="empty",
                ),
            ),
        )


@pytest.mark.asyncio
async def test_fake_vision_factory_returns_scripted_draft() -> None:
    """工厂应把 Vision 草稿交给现有 FakeVisionProvider。"""

    vision_input = make_vision_input()
    fixture = MultimodalOfflineScenarioFixture(
        scenario_id="multimodal-agent-001",
        planner_turns=(
            OfflinePlannerTurn(
                decision=AgentPlannerDecision(
                    decision="call_tool",
                    tool_call={
                        "call_id": "call_vision_001",
                        "tool_name": "analyze_robot_image",
                        "arguments": {
                            "image_ref": "image_001",
                            "analysis_goal": "读取面板故障码",
                        },
                    },
                ),
            ),
            OfflinePlannerTurn(
                decision=make_finish_decision(),
            ),
        ),
        vision_outcomes=(
            OfflineVisionOutcome(
                image_sha256=(
                    vision_input.metadata.sha256_hex
                ),
                draft=make_vision_draft(),
            ),
        ),
    )

    provider = build_offline_fake_vision_provider(
        fixture
    )
    observation = await provider.analyze_image(
        vision_input=vision_input
    )

    assert observation.status == "completed"
    assert observation.source == "vision_model"
    assert observation.source_image_sha256 == (
        vision_input.metadata.sha256_hex
    )
    assert provider.call_count == 1


@pytest.mark.asyncio
async def test_fake_vision_factory_converts_failure() -> None:
    """Vision timeout 夹具应转换成真实应用异常。"""

    vision_input = make_vision_input()
    fixture = MultimodalOfflineScenarioFixture(
        scenario_id="multimodal-agent-009",
        planner_turns=(
            OfflinePlannerTurn(
                decision=AgentPlannerDecision(
                    decision="call_tool",
                    tool_call={
                        "call_id": "call_vision_009",
                        "tool_name": "analyze_robot_image",
                        "arguments": {
                            "image_ref": "image_001",
                            "analysis_goal": "检查模糊面板",
                        },
                    },
                ),
            ),
            OfflinePlannerTurn(
                failure="error",
            ),
        ),
        vision_outcomes=(
            OfflineVisionOutcome(
                image_sha256=(
                    vision_input.metadata.sha256_hex
                ),
                failure="timeout",
            ),
        ),
    )

    provider = build_offline_fake_vision_provider(
        fixture
    )

    with pytest.raises(VisionTimeoutError):
        await provider.analyze_image(
            vision_input=vision_input
        )


@pytest.mark.asyncio
async def test_fake_vision_factory_maps_invalid_response_failure() -> None:
    """无效视觉响应应使用专门的应用异常而非普通异常。"""

    vision_input = make_vision_input(
        b"invalid"
    )
    image_hash = (
        vision_input.metadata.sha256_hex
    )
    fixture = MultimodalOfflineScenarioFixture(
        scenario_id="multimodal-agent-010",
        planner_turns=(
            OfflinePlannerTurn(
                decision=AgentPlannerDecision(
                    decision="call_tool",
                    tool_call={
                        "call_id": "call_vision_010",
                        "tool_name": "analyze_robot_image",
                        "arguments": {
                            "image_ref": "image_001",
                            "analysis_goal": "检查图片",
                        },
                    },
                ),
            ),
            OfflinePlannerTurn(
                failure="invalid_response",
            ),
        ),
        vision_outcomes=(
            OfflineVisionOutcome(
                image_sha256=image_hash,
                failure="invalid_response",
            ),
        ),
    )

    provider = build_offline_fake_vision_provider(
        fixture
    )
    with pytest.raises(
        InvalidVisionResponseError,
    ):
        await provider.analyze_image(
            vision_input=vision_input
        )
