"""Agent SSE事件发布器的离线状态机测试。

本文件只测试AgentEventPublisher：

1. 不启动FastAPI或SSE HTTP接口；
2. 不调用Planner、LLM、Vision或Embedding；
3. 不执行真实Agent工具；
4. 不访问知识库和诊断会话文件；
5. 通过预先构造的公开事件负载验证顺序、配对、终止和流式消费。
"""

import asyncio

import pytest

from app.schemas.agent_api import (
    AgentDiagnosisResponse,
    AgentExecutionSummary,
    AgentToolTraceEvent,
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
from app.services.agent_event_publisher import (
    AgentEventPublicationError,
    AgentEventPublisher,
)


# 所有事件和最终响应都使用同一个固定请求ID，
# 用于验证发布器不会把不同请求的结果混进同一条事件流。
TEST_REQUEST_ID = "request-agent-publisher-001"


# 该文档ID只用于满足诊断证据的数据契约，
# 不对应任何真实文件或知识库记录。
TEST_DOCUMENT_ID = "b" * 64


# 原因条目通过该Chunk ID引用上面的测试证据，
# 用于构造一份能够通过最终响应契约的completed诊断。
TEST_CHUNK_ID = (
    f"{TEST_DOCUMENT_ID}:000001"
)


def make_success_trace(
    *,
    step_id: int = 1,
    tool_name: str = "search_knowledge",
) -> AgentToolTraceEvent:
    """构造一条成功且已经脱敏的工具执行轨迹。"""

    return AgentToolTraceEvent(
        step_id=step_id,
        tool_name=tool_name,
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
    """构造工具完成后仍需继续收集证据的进度快照。"""

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
    trace: AgentToolTraceEvent,
) -> AgentDiagnosisResponse:
    """构造与一条成功工具轨迹对应的最终诊断响应。"""

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


def make_human_review_response(
) -> AgentDiagnosisResponse:
    """构造请求级安全策略直接转人工的最终响应。"""

    review_reason = (
        "需要有权限和资质的人员确认现场安全状态"
    )

    diagnosis = DiagnosisReport(
        request_id=TEST_REQUEST_ID,
        prompt_version="agent-tool-calling-v2",
        gate_version="agent-confirmed-evidence-v1",
        status="human_review_required",
        symptoms=[
            DiagnosisSymptom(
                description="请求远程解除急停",
                source="user_report",
            ),
        ],
        evidence=[],
        possible_causes=[],
        next_checks=[],
        risk_level="high",
        missing_information=[
            review_reason,
        ],
        abstained=False,
    )

    execution = AgentExecutionSummary(
        planner_prompt_version=(
            "agent-tool-calling-v2"
        ),
        state="completed",
        termination_reason=(
            "request_policy_finished"
        ),
        termination_message=(
            "请求涉及真实设备控制，需要人工审核"
        ),
        finish_reason=(
            "human_review_required"
        ),
        step_count=0,
        steps=[],
        missing_information=[
            review_reason,
        ],
    )

    return AgentDiagnosisResponse(
        request_id=TEST_REQUEST_ID,
        diagnosis=diagnosis,
        execution=execution,
    )


async def collect_stream(
    publisher: AgentEventPublisher,
) -> list[object]:
    """消费发布器直到终端事件并返回收到的全部事件。"""

    return [
        event
        async for event in publisher.stream()
    ]


async def publish_request_received(
    publisher: AgentEventPublisher,
) -> None:
    """发布所有正常事件流都需要的第一条接收事件。"""

    await publisher.publish(
        event_type="request_received",
        state="received",
        message="诊断请求已经接收",
        payload={
            "image_count": 0,
            "has_task_goal": True,
        },
    )


@pytest.mark.asyncio
async def test_complete_flow_assigns_sequence_and_streams_in_order(
) -> None:
    """正常工具流程应按发布顺序产生连续事件并由最终结果结束。"""

    publisher = AgentEventPublisher(
        request_id=TEST_REQUEST_ID,
    )
    trace = make_success_trace()

    await publish_request_received(
        publisher
    )

    await publisher.publish(
        event_type="safety_classified",
        state="planning",
        message="请求安全分类已经完成",
        payload={
            "disposition": "continue_to_planner",
            "reason_codes": (
                "knowledge_evidence_required",
            ),
        },
    )

    await publisher.publish(
        event_type="tool_scope_decided",
        state="planning",
        message="最小工具范围已经确定",
        payload={
            "required_capabilities": (
                "knowledge_evidence",
            ),
            "allowed_tool_names": (
                "search_knowledge",
            ),
        },
    )

    await publisher.publish(
        event_type="planning_started",
        state="planning",
        message="Agent开始规划第一步",
        payload={
            "step_number": 1,
            "allowed_tool_names": (
                "search_knowledge",
            ),
        },
    )

    await publisher.publish(
        event_type="tool_started",
        state="tool_running",
        message="知识库检索开始执行",
        payload={
            "step_id": 1,
            "tool_name": "search_knowledge",
            "input_summary": (
                "知识库检索；query_chars=24；top_k=3"
            ),
        },
    )

    await publisher.publish(
        event_type="tool_finished",
        state="observing",
        message="知识库检索执行完成",
        payload={
            "trace": trace,
        },
    )

    await publisher.publish(
        event_type="progress_updated",
        state="observing",
        message="Agent进度已经更新",
        payload={
            "progress": make_collecting_progress(),
        },
    )

    await publisher.publish(
        event_type="planning_started",
        state="planning",
        message="Agent开始形成最终诊断",
        payload={
            "step_number": 2,
            "allowed_tool_names": (),
        },
    )

    final_event = await publisher.publish(
        event_type="diagnosis_finished",
        state="completed",
        message="最终诊断已经生成",
        payload={
            "response": make_completed_response(
                trace=trace,
            ),
        },
    )

    streamed_events = await collect_stream(
        publisher
    )

    assert [
        event.sequence
        for event in streamed_events
    ] == list(range(1, 10))

    assert [
        event.event_type
        for event in streamed_events
    ] == [
        "request_received",
        "safety_classified",
        "tool_scope_decided",
        "planning_started",
        "tool_started",
        "tool_finished",
        "progress_updated",
        "planning_started",
        "diagnosis_finished",
    ]

    assert tuple(streamed_events) == (
        publisher.events
    )
    assert streamed_events[-1] is final_event
    assert publisher.next_sequence == 10
    assert publisher.closed is True


@pytest.mark.asyncio
async def test_human_review_policy_can_finish_without_planner_or_tools(
) -> None:
    """高风险请求应能在安全分类后直接发布转人工结果。"""

    publisher = AgentEventPublisher(
        request_id=TEST_REQUEST_ID,
    )

    await publish_request_received(
        publisher
    )
    await publisher.publish(
        event_type="safety_classified",
        state="completed",
        message="请求需要人工审核",
        payload={
            "disposition": (
                "human_review_required"
            ),
            "reason_codes": (
                "high_risk_control_detected",
                "no_safe_tool_required",
            ),
        },
    )
    await publisher.publish(
        event_type="diagnosis_finished",
        state="completed",
        message="转人工诊断结果已经生成",
        payload={
            "response": (
                make_human_review_response()
            ),
        },
    )

    streamed_events = await collect_stream(
        publisher
    )

    assert [
        event.event_type
        for event in streamed_events
    ] == [
        "request_received",
        "safety_classified",
        "diagnosis_finished",
    ]
    assert (
        streamed_events[-1]
        .response
        .diagnosis
        .status
        == "human_review_required"
    )
    assert all(
        event.event_type
        not in {
            "planning_started",
            "tool_started",
            "tool_finished",
        }
        for event in streamed_events
    )


@pytest.mark.asyncio
async def test_concurrent_publications_allocate_only_one_next_event(
) -> None:
    """并发发布相同下一阶段时只能提交一条连续事件。"""

    publisher = AgentEventPublisher(
        request_id=TEST_REQUEST_ID,
    )
    await publish_request_received(
        publisher
    )

    async def publish_safety_event(
    ) -> object:
        """模拟一个独立异步生产方发布安全分类。"""

        return await publisher.publish(
            event_type="safety_classified",
            state="planning",
            message="请求安全分类已经完成",
            payload={
                "disposition": (
                    "continue_to_planner"
                ),
                "reason_codes": (
                    "knowledge_evidence_required",
                ),
            },
        )

    results = await asyncio.gather(
        publish_safety_event(),
        publish_safety_event(),
        return_exceptions=True,
    )

    successful_results = [
        result
        for result in results
        if not isinstance(
            result,
            Exception,
        )
    ]
    failed_results = [
        result
        for result in results
        if isinstance(
            result,
            AgentEventPublicationError,
        )
    ]

    assert len(successful_results) == 1
    assert len(failed_results) == 1
    assert [
        event.sequence
        for event in publisher.events
    ] == [1, 2]
    assert publisher.next_sequence == 3


@pytest.mark.asyncio
async def test_first_event_must_be_request_received(
) -> None:
    """没有接收事件时不能直接发布安全分类。"""

    publisher = AgentEventPublisher(
        request_id=TEST_REQUEST_ID,
    )

    with pytest.raises(
        AgentEventPublicationError,
        match="第一条SSE事件",
    ):
        await publisher.publish(
            event_type="safety_classified",
            state="planning",
            message="请求安全分类已经完成",
            payload={
                "disposition": (
                    "continue_to_planner"
                ),
                "reason_codes": (
                    "knowledge_evidence_required",
                ),
            },
        )

    assert publisher.next_sequence == 1
    assert publisher.events == ()


@pytest.mark.asyncio
async def test_invalid_schema_does_not_consume_sequence(
) -> None:
    """字段契约失败时不应留下半条事件或跳过序号。"""

    publisher = AgentEventPublisher(
        request_id=TEST_REQUEST_ID,
    )

    with pytest.raises(
        AgentEventPublicationError,
        match="未通过公开数据契约",
    ):
        await publisher.publish(
            event_type="request_received",
            state="received",
            message="诊断请求已经接收",
            payload={
                "image_count": 4,
                "has_task_goal": False,
            },
        )

    assert publisher.next_sequence == 1
    assert publisher.events == ()

    await publish_request_received(
        publisher
    )

    assert publisher.events[0].sequence == 1


@pytest.mark.asyncio
async def test_payload_cannot_override_publisher_fields(
) -> None:
    """调用方不能通过payload伪造序号或请求ID。"""

    publisher = AgentEventPublisher(
        request_id=TEST_REQUEST_ID,
    )

    with pytest.raises(
        AgentEventPublicationError,
        match="payload不能覆盖",
    ):
        await publisher.publish(
            event_type="request_received",
            state="received",
            message="诊断请求已经接收",
            payload={
                "sequence": 99,
                "image_count": 0,
                "has_task_goal": False,
            },
        )


@pytest.mark.asyncio
async def test_illegal_event_order_is_rejected(
) -> None:
    """接收事件之后不能跳过安全分类直接开始规划。"""

    publisher = AgentEventPublisher(
        request_id=TEST_REQUEST_ID,
    )
    await publish_request_received(
        publisher
    )

    with pytest.raises(
        AgentEventPublicationError,
        match="非法SSE事件顺序",
    ):
        await publisher.publish(
            event_type="planning_started",
            state="planning",
            message="Agent开始规划下一步",
            payload={
                "step_number": 1,
                "allowed_tool_names": (),
            },
        )

    assert len(publisher.events) == 1
    assert publisher.next_sequence == 2


@pytest.mark.asyncio
async def test_tool_started_must_match_planning_step(
) -> None:
    """工具开始事件的步骤号必须与上一条规划事件一致。"""

    publisher = AgentEventPublisher(
        request_id=TEST_REQUEST_ID,
    )

    await publish_request_received(
        publisher
    )
    await publisher.publish(
        event_type="safety_classified",
        state="planning",
        message="请求安全分类已经完成",
        payload={
            "disposition": "continue_to_planner",
            "reason_codes": (
                "knowledge_evidence_required",
            ),
        },
    )
    await publisher.publish(
        event_type="tool_scope_decided",
        state="planning",
        message="最小工具范围已经确定",
        payload={
            "required_capabilities": (
                "knowledge_evidence",
            ),
            "allowed_tool_names": (
                "search_knowledge",
            ),
        },
    )
    await publisher.publish(
        event_type="planning_started",
        state="planning",
        message="Agent开始规划下一步",
        payload={
            "step_number": 1,
            "allowed_tool_names": (
                "search_knowledge",
            ),
        },
    )

    with pytest.raises(
        AgentEventPublicationError,
        match="工具步骤编号",
    ):
        await publisher.publish(
            event_type="tool_started",
            state="tool_running",
            message="知识库检索开始执行",
            payload={
                "step_id": 2,
                "tool_name": (
                    "search_knowledge"
                ),
                "input_summary": (
                    "脱敏工具输入摘要"
                ),
            },
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("finished_step", "finished_tool", "error_text"),
    [
        (2, "search_knowledge", "步骤编号"),
        (1, "get_current_time", "工具名称"),
    ],
)
async def test_tool_finished_must_match_active_tool(
    finished_step: int,
    finished_tool: str,
    error_text: str,
) -> None:
    """工具结束事件必须与正在执行的步骤和工具名称配对。"""

    publisher = AgentEventPublisher(
        request_id=TEST_REQUEST_ID,
    )

    await publish_request_received(
        publisher
    )
    await publisher.publish(
        event_type="safety_classified",
        state="planning",
        message="请求安全分类已经完成",
        payload={
            "disposition": "continue_to_planner",
            "reason_codes": (
                "knowledge_evidence_required",
            ),
        },
    )
    await publisher.publish(
        event_type="tool_scope_decided",
        state="planning",
        message="最小工具范围已经确定",
        payload={
            "required_capabilities": (
                "knowledge_evidence",
            ),
            "allowed_tool_names": (
                "search_knowledge",
            ),
        },
    )
    await publisher.publish(
        event_type="planning_started",
        state="planning",
        message="Agent开始规划下一步",
        payload={
            "step_number": 1,
            "allowed_tool_names": (
                "search_knowledge",
            ),
        },
    )
    await publisher.publish(
        event_type="tool_started",
        state="tool_running",
        message="知识库检索开始执行",
        payload={
            "step_id": 1,
            "tool_name": "search_knowledge",
            "input_summary": (
                "脱敏工具输入摘要"
            ),
        },
    )

    with pytest.raises(
        AgentEventPublicationError,
        match=error_text,
    ):
        await publisher.publish(
            event_type="tool_finished",
            state="observing",
            message="工具执行完成",
            payload={
                "trace": make_success_trace(
                    step_id=finished_step,
                    tool_name=finished_tool,
                ),
            },
        )

    assert publisher.events[-1].event_type == (
        "tool_started"
    )


@pytest.mark.asyncio
async def test_stream_error_closes_stream_and_blocks_more_events(
) -> None:
    """流错误应成为唯一终端事件并让异步消费者正常结束。"""

    publisher = AgentEventPublisher(
        request_id=TEST_REQUEST_ID,
    )
    await publish_request_received(
        publisher
    )

    await publisher.publish(
        event_type="stream_error",
        state="aborted",
        message="诊断事件流意外中断",
        payload={
            "error": ErrorDetail(
                code="stream_interrupted",
                message="诊断事件流已经中断",
            ),
            "retryable": True,
        },
    )

    streamed_events = await collect_stream(
        publisher
    )

    assert [
        event.event_type
        for event in streamed_events
    ] == [
        "request_received",
        "stream_error",
    ]
    assert publisher.closed is True

    with pytest.raises(
        AgentEventPublicationError,
        match="终端事件之后",
    ):
        await publisher.publish(
            event_type="stream_error",
            state="aborted",
            message="不允许的第二个终端事件",
            payload={
                "error": ErrorDetail(
                    code="stream_interrupted",
                    message="诊断事件流已经中断",
                ),
                "retryable": False,
            },
        )


@pytest.mark.asyncio
async def test_stream_allows_only_one_consumer(
) -> None:
    """同一队列不能被两个SSE响应竞争消费。"""

    publisher = AgentEventPublisher(
        request_id=TEST_REQUEST_ID,
    )
    await publish_request_received(
        publisher
    )
    await publisher.publish(
        event_type="stream_error",
        state="aborted",
        message="诊断事件流意外中断",
        payload={
            "error": ErrorDetail(
                code="stream_interrupted",
                message="诊断事件流已经中断",
            ),
            "retryable": False,
        },
    )

    first_stream = await collect_stream(
        publisher
    )

    assert len(first_stream) == 2

    with pytest.raises(
        AgentEventPublicationError,
        match="一个事件消费者",
    ):
        await collect_stream(
            publisher
        )


@pytest.mark.asyncio
async def test_max_events_stops_unbounded_publication(
) -> None:
    """达到单请求事件上限后应拒绝继续占用内存。"""

    publisher = AgentEventPublisher(
        request_id=TEST_REQUEST_ID,
        max_events=1,
    )
    await publish_request_received(
        publisher
    )

    with pytest.raises(
        AgentEventPublicationError,
        match="已经达到上限",
    ):
        await publisher.publish(
            event_type="safety_classified",
            state="planning",
            message="请求安全分类已经完成",
            payload={
                "disposition": (
                    "continue_to_planner"
                ),
                "reason_codes": (
                    "knowledge_evidence_required",
                ),
            },
        )

    assert len(publisher.events) == 1
    assert publisher.next_sequence == 2
