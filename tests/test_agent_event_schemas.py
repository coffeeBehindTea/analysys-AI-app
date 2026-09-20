"""Agent SSE公开事件数据契约的离线测试。

本文件只构造Pydantic事件模型：

1. 不启动FastAPI或SSE服务器；
2. 不调用Planner、LLM、Vision或Embedding；
3. 不执行Agent工具；
4. 不访问知识库和会话文件；
5. 不测试事件先后顺序，那属于后续EventPublisher测试。

测试重点是九种事件的字段结构、判别联合、
生命周期关系、最终响应关系和公开字段安全边界。
"""

import pytest

from pydantic import (
    TypeAdapter,
    ValidationError,
)

from app.schemas.agent_api import (
    AgentDiagnosisResponse,
    AgentExecutionSummary,
    AgentToolTraceEvent,
)
from app.schemas.agent_events import (
    AgentDiagnosisFinishedEvent,
    AgentPlanningStartedEvent,
    AgentProgressUpdatedEvent,
    AgentRequestReceivedEvent,
    AgentSafetyClassifiedEvent,
    AgentSseEvent,
    AgentStreamErrorEvent,
    AgentToolFinishedEvent,
    AgentToolScopeDecidedEvent,
    AgentToolStartedEvent,
)
from app.schemas.agent_progress import (
    AgentProgress,
)
from app.schemas.diagnostics import (
    DiagnosisCause,
    DiagnosisEvidence,
    DiagnosisReport,
    DiagnosisSymptom,
)
from app.schemas.error import (
    ErrorDetail,
)


# 固定请求ID用于验证事件、最终响应和诊断报告
# 是否属于同一次请求。
TEST_REQUEST_ID = "request-agent-sse-001"


# 固定SHA-256形状的文档ID仅用于离线测试，
# 不对应任何真实知识库文件。
TEST_DOCUMENT_ID = "a" * 64


# 诊断原因通过该Chunk ID引用测试证据。
TEST_CHUNK_ID = (
    f"{TEST_DOCUMENT_ID}:000001"
)


def make_success_trace(
    *,
    step_id: int = 1,
) -> AgentToolTraceEvent:
    """构造一条成功且不含错误码的脱敏工具轨迹。"""

    return AgentToolTraceEvent(
        step_id=step_id,
        tool_name="search_knowledge",
        input_summary=(
            "知识库检索；query_chars=24；top_k=3"
        ),
        result_summary=(
            "知识库返回1条已确认引用"
        ),
        status="success",
        duration_ms=12.5,
    )


def make_collecting_progress(
) -> AgentProgress:
    """构造仍在收集知识库证据的确定性进度快照。"""

    return AgentProgress(
        state="collecting",
        required_capabilities=(
            "knowledge_evidence",
        ),
        completed_capabilities=(),
        allowed_next_tool_names=(
            "search_knowledge",
        ),
    )


def make_completed_response(
    *,
    request_id: str = TEST_REQUEST_ID,
) -> AgentDiagnosisResponse:
    """构造可用于终止事件的最小完整诊断响应。"""

    diagnosis = DiagnosisReport(
        request_id=request_id,
        prompt_version="agent-tool-calling-v2",
        gate_version="agent-confirmed-evidence-v1",
        status="completed",
        symptoms=[
            DiagnosisSymptom(
                description="网络恢复后机器人仍暂停",
                source="user_report",
            ),
        ],
        evidence=[
            DiagnosisEvidence(
                chunk_id=TEST_CHUNK_ID,
                document_id=TEST_DOCUMENT_ID,
                source_file="robot-fault-demo.txt",
                page_or_section=(
                    "section: ERR-NET-4001"
                ),
                rank=1,
                rrf_score=0.0327,
                vector_similarity=0.71,
                excerpt=(
                    "网络恢复后不得自动继续旧任务。"
                ),
            ),
        ],
        possible_causes=[
            DiagnosisCause(
                description=(
                    "旧任务仍在等待调度状态核对"
                ),
                evidence_chunk_ids=[
                    TEST_CHUNK_ID,
                ],
            ),
        ],
        next_checks=[],
        risk_level="medium",
        missing_information=[],
        abstained=False,
    )

    execution = AgentExecutionSummary(
        planner_prompt_version=(
            "agent-tool-calling-v2"
        ),
        state="completed",
        termination_reason="planner_finished",
        termination_message=(
            "Agent规划正常结束"
        ),
        finish_reason="task_completed",
        step_count=0,
        steps=[],
        missing_information=[],
    )

    return AgentDiagnosisResponse(
        request_id=request_id,
        diagnosis=diagnosis,
        execution=execution,
    )


def test_all_nine_public_event_models_accept_valid_data(
) -> None:
    """九种事件都应能表达各自预期的公开阶段。"""

    events = [
        AgentRequestReceivedEvent(
            sequence=1,
            request_id=TEST_REQUEST_ID,
            message="诊断请求已经接收",
            image_count=1,
            has_task_goal=True,
        ),
        AgentSafetyClassifiedEvent(
            sequence=2,
            request_id=TEST_REQUEST_ID,
            state="planning",
            message="请求安全分类已经完成",
            disposition="continue_to_planner",
            reason_codes=(
                "knowledge_evidence_required",
            ),
        ),
        AgentToolScopeDecidedEvent(
            sequence=3,
            request_id=TEST_REQUEST_ID,
            message="最小工具范围已经确定",
            required_capabilities=(
                "knowledge_evidence",
            ),
            allowed_tool_names=(
                "search_knowledge",
            ),
        ),
        AgentPlanningStartedEvent(
            sequence=4,
            request_id=TEST_REQUEST_ID,
            message="Agent开始规划下一步",
            step_number=1,
            allowed_tool_names=(
                "search_knowledge",
            ),
        ),
        AgentToolStartedEvent(
            sequence=5,
            request_id=TEST_REQUEST_ID,
            message="知识库检索开始执行",
            step_id=1,
            tool_name="search_knowledge",
            input_summary=(
                "知识库检索；query_chars=24；top_k=3"
            ),
        ),
        AgentToolFinishedEvent(
            sequence=6,
            request_id=TEST_REQUEST_ID,
            message="知识库检索执行完成",
            trace=make_success_trace(),
        ),
        AgentProgressUpdatedEvent(
            sequence=7,
            request_id=TEST_REQUEST_ID,
            state="observing",
            message="Agent进度已经更新",
            progress=make_collecting_progress(),
        ),
        AgentDiagnosisFinishedEvent(
            sequence=8,
            request_id=TEST_REQUEST_ID,
            state="completed",
            message="最终诊断已经生成",
            response=make_completed_response(),
        ),
        AgentStreamErrorEvent(
            sequence=9,
            request_id=TEST_REQUEST_ID,
            message="诊断事件流意外中断",
            error=ErrorDetail(
                code="stream_interrupted",
                message="诊断事件流已经中断",
            ),
            retryable=True,
        ),
    ]

    assert {
        event.event_type
        for event in events
    } == {
        "request_received",
        "safety_classified",
        "tool_scope_decided",
        "planning_started",
        "tool_started",
        "tool_finished",
        "progress_updated",
        "diagnosis_finished",
        "stream_error",
    }

    assert all(
        event.schema_version
        == "agent-sse-event-v1"
        for event in events
    )


def test_discriminated_union_selects_event_by_event_type(
) -> None:
    """TypeAdapter应根据event_type选择唯一具体事件模型。"""

    adapter = TypeAdapter(
        AgentSseEvent
    )

    event = adapter.validate_python({
        "sequence": 1,
        "event_type": "request_received",
        "request_id": TEST_REQUEST_ID,
        "state": "received",
        "message": "诊断请求已经接收",
        "image_count": 0,
        "has_task_goal": False,
    })

    assert isinstance(
        event,
        AgentRequestReceivedEvent,
    )


def test_discriminated_union_rejects_unknown_event_type(
) -> None:
    """未知事件类型不能进入SSE发布链。"""

    adapter = TypeAdapter(
        AgentSseEvent
    )

    with pytest.raises(
        ValidationError,
        match="union_tag_invalid",
    ):
        adapter.validate_python({
            "sequence": 1,
            "event_type": "planner_private_reasoning",
            "request_id": TEST_REQUEST_ID,
            "state": "planning",
            "message": "不应公开的事件",
        })


def test_event_rejects_undeclared_sensitive_field(
) -> None:
    """事件不能夹带完整工具参数等未声明字段。"""

    with pytest.raises(
        ValidationError,
        match="extra_forbidden",
    ):
        AgentToolStartedEvent.model_validate({
            "sequence": 1,
            "request_id": TEST_REQUEST_ID,
            "message": "工具开始执行",
            "step_id": 1,
            "tool_name": "search_knowledge",
            "input_summary": "脱敏输入摘要",
            "arguments": {
                "raw_log": "完整敏感日志",
            },
        })


def test_request_received_rejects_more_than_three_images(
) -> None:
    """公开接收事件必须沿用诊断请求的三张图片上限。"""

    with pytest.raises(
        ValidationError,
        match="less_than_equal",
    ):
        AgentRequestReceivedEvent(
            sequence=1,
            request_id=TEST_REQUEST_ID,
            message="诊断请求已经接收",
            image_count=4,
            has_task_goal=False,
        )


@pytest.mark.parametrize(
    ("disposition", "state"),
    [
        ("continue_to_planner", "completed"),
        ("abstained", "planning"),
        ("human_review_required", "planning"),
    ],
)
def test_safety_disposition_must_match_lifecycle_state(
    disposition: str,
    state: str,
) -> None:
    """继续规划和策略终止不能使用相反的生命周期状态。"""

    with pytest.raises(
        ValidationError,
        match=(
            "必须使用planning状态"
            "|必须使用completed生命周期状态"
        ),
    ):
        AgentSafetyClassifiedEvent.model_validate({
            "sequence": 1,
            "request_id": TEST_REQUEST_ID,
            "state": state,
            "message": "请求安全分类已经完成",
            "disposition": disposition,
            "reason_codes": [
                "knowledge_evidence_required",
            ],
        })


def test_progress_event_requires_matching_public_state(
) -> None:
    """collecting进度不能被公开成已经完成。"""

    with pytest.raises(
        ValidationError,
        match="AgentProgress.state不一致",
    ):
        AgentProgressUpdatedEvent(
            sequence=1,
            request_id=TEST_REQUEST_ID,
            state="completed",
            message="Agent进度已经更新",
            progress=make_collecting_progress(),
        )


def test_finished_event_requires_matching_request_id(
) -> None:
    """终止事件和最终响应必须属于同一次请求。"""

    with pytest.raises(
        ValidationError,
        match="响应request_id一致",
    ):
        AgentDiagnosisFinishedEvent(
            sequence=1,
            request_id="different-request-id",
            state="completed",
            message="最终诊断已经生成",
            response=make_completed_response(),
        )


def test_finished_event_requires_matching_execution_state(
) -> None:
    """终止事件不能把正常完成的响应标记为运行中止。"""

    with pytest.raises(
        ValidationError,
        match="execution.state一致",
    ):
        AgentDiagnosisFinishedEvent(
            sequence=1,
            request_id=TEST_REQUEST_ID,
            state="aborted",
            message="最终诊断已经生成",
            response=make_completed_response(),
        )


def test_event_is_frozen_after_validation(
) -> None:
    """公开事件创建后不能被其他消费者原地修改。"""

    event = AgentRequestReceivedEvent(
        sequence=1,
        request_id=TEST_REQUEST_ID,
        message="诊断请求已经接收",
        image_count=0,
        has_task_goal=False,
    )

    with pytest.raises(
        ValidationError,
        match="frozen_instance",
    ):
        event.sequence = 2
