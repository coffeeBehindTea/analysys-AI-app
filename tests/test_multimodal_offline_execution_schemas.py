"""Week 6 单场景离线执行审计契约的单元测试。

被测试模块：app.schemas.multimodal_offline_regression。

这些测试只校验执行结果中的计数关系，
不启动 Agent、不调用任何 Provider，也不访问文件系统。
"""

import pytest
from pydantic import ValidationError

from app.schemas.agent_api import (
    AgentDiagnosisResponse,
    AgentExecutionSummary,
)
from app.schemas.diagnostics import (
    DiagnosisReport,
    DiagnosisSymptom,
)
from app.schemas.multimodal_offline_regression import (
    MultimodalOfflineScenarioExecution,
    OfflineToolCallAudit,
)


# 固定请求ID用于校验顶层响应与诊断报告的关联。
TEST_REQUEST_ID = "offline-multimodal-agent-001"


# 固定图片摘要只用于验证Vision结果按图片身份消费。
TEST_IMAGE_SHA256 = "a" * 64


def make_abstained_response() -> AgentDiagnosisResponse:
    """创建一份不依赖证据和工具输出的合法拒答响应。"""

    missing_information = ["缺少可确认的知识库证据"]

    diagnosis = DiagnosisReport(
        request_id=TEST_REQUEST_ID,
        prompt_version="offline-agent-planner-v1",
        gate_version="agent-confirmed-evidence-v1",
        status="abstained",
        symptoms=[
            DiagnosisSymptom(
                description="机器人面板状态无法确认",
                source="user_report",
            )
        ],
        evidence=[],
        possible_causes=[],
        next_checks=[],
        risk_level="unknown",
        missing_information=missing_information,
        abstained=True,
    )

    execution = AgentExecutionSummary(
        planner_prompt_version=(
            "offline-agent-planner-v1"
        ),
        state="completed",
        termination_reason="planner_finished",
        termination_message="Agent规划正常结束",
        finish_reason="insufficient_information",
        step_count=0,
        steps=[],
        missing_information=missing_information,
    )

    return AgentDiagnosisResponse(
        request_id=TEST_REQUEST_ID,
        diagnosis=diagnosis,
        execution=execution,
    )


def test_tool_call_audit_reports_fully_consumed() -> None:
    """预设数与实际数相等时，工具夹具应视为完整消费。"""

    audit = OfflineToolCallAudit(
        tool_name="search_knowledge",
        planned_outcome_count=2,
        actual_call_count=2,
    )

    assert audit.fully_consumed is True


def test_tool_call_audit_reports_unconsumed_outcome() -> None:
    """实际调用较少时，审计必须保留未消费事实。"""

    audit = OfflineToolCallAudit(
        tool_name="search_knowledge",
        planned_outcome_count=2,
        actual_call_count=1,
    )

    assert audit.fully_consumed is False


def test_execution_accepts_consistent_audit_counts() -> None:
    """全部调用计数一致时应创建可序列化执行结果。"""

    execution = MultimodalOfflineScenarioExecution(
        scenario_id="multimodal-agent-001",
        response=make_abstained_response(),
        planner_turn_count=2,
        planner_call_count=2,
        planner_exposed_tool_names=(
            ("analyze_robot_image",),
            ("search_knowledge",),
        ),
        vision_outcome_count=1,
        planned_vision_image_sha256s=(
            TEST_IMAGE_SHA256,
        ),
        # 同一图片被调用两次表示Vision内部进行了一次重试。
        vision_call_count=2,
        vision_called_image_sha256s=(
            TEST_IMAGE_SHA256,
            TEST_IMAGE_SHA256,
        ),
        tool_call_audits=(
            OfflineToolCallAudit(
                tool_name="search_knowledge",
                planned_outcome_count=1,
                actual_call_count=1,
            ),
        ),
        fixture_fully_consumed=True,
    )

    dumped = execution.model_dump(
        mode="json"
    )

    assert dumped["scenario_id"] == (
        "multimodal-agent-001"
    )
    assert dumped["fixture_fully_consumed"] is True
    assert dumped["response"]["diagnosis"][
        "abstained"
    ] is True


@pytest.mark.parametrize(
    (
        "field_overrides",
        "expected_message",
    ),
    [
        (
            {
                "planner_call_count": 2,
                "planner_exposed_tool_names": (
                    ("search_knowledge",),
                ),
            },
            "planner_exposed_tool_names数量",
        ),
        (
            {
                "tool_call_audits": (
                    OfflineToolCallAudit(
                        tool_name="search_knowledge",
                        planned_outcome_count=1,
                        actual_call_count=1,
                    ),
                    OfflineToolCallAudit(
                        tool_name="search_knowledge",
                        planned_outcome_count=1,
                        actual_call_count=1,
                    ),
                ),
            },
            "不能包含重复tool_name",
        ),
        (
            {
                "planner_turn_count": 2,
                "fixture_fully_consumed": True,
            },
            "与审计计数不一致",
        ),
    ],
)
def test_execution_rejects_inconsistent_audit_data(
    field_overrides: dict[str, object],
    expected_message: str,
) -> None:
    """错误计数、重复工具和错误汇总结论都必须被拒绝。"""

    data: dict[str, object] = {
        "scenario_id": "multimodal-agent-001",
        "response": make_abstained_response(),
        "planner_turn_count": 1,
        "planner_call_count": 1,
        "planner_exposed_tool_names": (
            ("search_knowledge",),
        ),
        "vision_outcome_count": 0,
        "planned_vision_image_sha256s": (),
        "vision_call_count": 0,
        "vision_called_image_sha256s": (),
        "tool_call_audits": (),
        "fixture_fully_consumed": True,
    }
    data.update(field_overrides)

    with pytest.raises(
        ValidationError,
        match=expected_message,
    ):
        MultimodalOfflineScenarioExecution(
            **data,
        )
