"""Agent SSE文本编码器的离线测试。

本文件只测试结构化事件到SSE文本帧的转换：

1. 不启动FastAPI或HTTP服务器；
2. 不运行Agent、Planner或工具；
3. 不访问LLM、Vision、Embedding和知识库；
4. 不重复验证事件先后顺序；
5. 通过解析编码结果验证id、event、data和结束空行。
"""

import json

import pytest

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
from app.services.agent_sse_encoder import (
    AGENT_SSE_CACHE_CONTROL,
    AGENT_SSE_CONTENT_TYPE,
    AgentSseEncodingError,
    encode_agent_sse_event,
)


# 固定请求ID用于检查事件JSON在编码前后没有发生变化。
TEST_REQUEST_ID = "request-agent-encoder-001"


# 固定测试文档和Chunk ID只用于构造合法的最终诊断响应。
TEST_DOCUMENT_ID = "c" * 64
TEST_CHUNK_ID = (
    f"{TEST_DOCUMENT_ID}:000001"
)


def make_success_trace(
) -> AgentToolTraceEvent:
    """构造可以嵌入tool_finished和最终响应的脱敏轨迹。"""

    return AgentToolTraceEvent(
        step_id=1,
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
    """构造可以嵌入progress_updated事件的进度快照。"""

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
) -> AgentDiagnosisResponse:
    """构造用于验证深层嵌套JSON编码的最终响应。"""

    trace = make_success_trace()

    diagnosis = DiagnosisReport(
        request_id=TEST_REQUEST_ID,
        prompt_version="agent-tool-calling-v2",
        gate_version="agent-confirmed-evidence-v1",
        status="completed",
        symptoms=[
            DiagnosisSymptom(
                description=(
                    "网络恢复后机器人仍暂停"
                ),
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
        step_count=1,
        steps=[trace],
        missing_information=[],
    )

    return AgentDiagnosisResponse(
        request_id=TEST_REQUEST_ID,
        diagnosis=diagnosis,
        execution=execution,
    )


def make_all_public_events(
) -> tuple[AgentSseEvent, ...]:
    """构造九种能够独立进行编码测试的公开事件。"""

    trace = make_success_trace()

    return (
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
            trace=trace,
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
    )


def parse_sse_frame(
    frame: str,
) -> tuple[int, str, dict[str, object]]:
    """把单个测试帧拆成id、事件名和JSON对象。

    该函数不是生产SSE解析器，只用于让测试明确检查
    编码器是否生成三行字段和一个结束空行。
    """

    assert isinstance(frame, str)
    assert frame.endswith("\n\n")

    content_lines = frame[:-2].split("\n")

    assert len(content_lines) == 3
    assert content_lines[0].startswith("id: ")
    assert content_lines[1].startswith("event: ")
    assert content_lines[2].startswith("data: ")

    sequence = int(
        content_lines[0].removeprefix(
            "id: "
        )
    )
    event_type = (
        content_lines[1].removeprefix(
            "event: "
        )
    )
    event_data = json.loads(
        content_lines[2].removeprefix(
            "data: "
        )
    )

    return (
        sequence,
        event_type,
        event_data,
    )


def test_sse_response_constants_match_public_contract(
) -> None:
    """HTTP层应获得明确的UTF-8媒体类型和禁用缓存值。"""

    assert AGENT_SSE_CONTENT_TYPE == (
        "text/event-stream; charset=utf-8"
    )
    assert AGENT_SSE_CACHE_CONTROL == (
        "no-cache"
    )


@pytest.mark.parametrize(
    "event",
    make_all_public_events(),
    ids=lambda event: event.event_type,
)
def test_all_public_event_types_encode_to_matching_sse_frame(
    event: AgentSseEvent,
) -> None:
    """九种公开事件都应保留序号、类型和完整JSON数据。"""

    frame = encode_agent_sse_event(
        event
    )
    sequence, event_type, event_data = (
        parse_sse_frame(frame)
    )

    assert sequence == event.sequence
    assert event_type == event.event_type
    assert event_data == event.model_dump(
        mode="json",
    )


def test_encoder_preserves_chinese_and_escapes_message_newline(
) -> None:
    """中文应保持可读，消息换行不能破坏SSE三行结构。"""

    event = AgentRequestReceivedEvent(
        sequence=1,
        request_id=TEST_REQUEST_ID,
        message="第一行\n第二行：诊断请求已接收",
        image_count=0,
        has_task_goal=False,
    )

    frame = encode_agent_sse_event(
        event
    )
    _, _, event_data = parse_sse_frame(
        frame
    )

    assert "诊断请求已接收" in frame
    assert "\\n" in frame
    assert event_data["message"] == (
        "第一行\n第二行：诊断请求已接收"
    )


def test_encoder_uses_compact_single_line_json(
) -> None:
    """data字段应是没有格式化空格和物理换行的紧凑JSON。"""

    event = AgentRequestReceivedEvent(
        sequence=1,
        request_id=TEST_REQUEST_ID,
        message="诊断请求已经接收",
        image_count=0,
        has_task_goal=True,
    )

    frame = encode_agent_sse_event(
        event
    )
    data_line = frame.splitlines()[2]

    assert data_line.startswith("data: {")
    assert '": "' not in data_line
    assert '", "' not in data_line
    assert "\\u8bca" not in data_line


def test_encoder_rejects_non_event_input(
) -> None:
    """普通字典不能绕过Pydantic事件契约直接进入SSE。"""

    with pytest.raises(
        TypeError,
        match="AgentSseEventBase",
    ):
        encode_agent_sse_event({
            "sequence": 1,
            "event_type": "request_received",
        })


def test_encoder_converts_json_serialization_failure(
) -> None:
    """内部JSON转换异常应变成稳定的编码器异常。"""

    class SerializationFailingEvent(
        AgentRequestReceivedEvent
    ):
        """只在本测试中模拟Pydantic序列化失败。"""

        def model_dump(
            self,
            *args: object,
            **kwargs: object,
        ) -> dict[str, object]:
            """用稳定ValueError代替真实序列化过程。"""

            raise ValueError(
                "simulated serialization failure"
            )

    event = SerializationFailingEvent(
        sequence=1,
        request_id=TEST_REQUEST_ID,
        message="诊断请求已经接收",
        image_count=0,
        has_task_goal=False,
    )

    with pytest.raises(
        AgentSseEncodingError,
        match="无法编码成标准JSON",
    ):
        encode_agent_sse_event(event)
