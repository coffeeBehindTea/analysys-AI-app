"""AgentRunner到SSE事件适配器的离线测试。

本文件验证：

1. Observer实现的八个异步方法会发布正确事件；
2. Publisher分配的sequence保持连续；
3. 工具输入与结果使用公共构造器完成脱敏；
4. 七种AgentProgress状态映射成正确生命周期；
5. 不匹配的工具调用和结果不能产生结束事件；
6. 伪造的success输出不能进入公开SSE轨迹；
7. 构造参数、进度参数和最终响应会在边界接受类型检查；
8. 请求策略提前终止时也能发布完整且有序的最终事件。

测试使用真实AgentEventPublisher，但不启动FastAPI，
也不调用真实Planner、LLM、工具、网络或知识库。
"""

from inspect import (
    iscoroutinefunction,
)

import pytest

from pydantic import (
    ValidationError,
)

from app.schemas.agent import (
    ToolCall,
    ToolExecutionResult,
)
from app.schemas.agent_api import (
    AgentDiagnosisResponse,
    AgentExecutionSummary,
)
from app.schemas.agent_events import (
    AgentDiagnosisFinishedEvent,
    AgentPlanningStartedEvent,
    AgentProgressUpdatedEvent,
    AgentRequestReceivedEvent,
    AgentSafetyClassifiedEvent,
    AgentToolScopeDecidedEvent,
    AgentToolFinishedEvent,
    AgentToolStartedEvent,
)
from app.schemas.agent_progress import (
    AgentProgress,
    AgentProgressState,
)
from app.schemas.agent_tools import (
    SearchKnowledgeToolOutput,
)
from app.schemas.diagnostics import (
    DiagnosisReport,
    DiagnosisSymptom,
)
from app.schemas.knowledge_query import (
    KnowledgeCitation,
)
from app.services.agent_event_publisher import (
    AgentEventPublisher,
)
from app.services.agent_runner_sse_observer import (
    AgentRunnerSseObserver,
)


# 单次测试事件流使用固定请求ID，便于检查所有事件
# 都属于同一个请求，不对应任何真实HTTP请求。
TEST_REQUEST_ID = (
    "request-runner-sse-observer-001"
)


# 固定文档和Chunk标识用于构造合法知识引用。
# 值只用于离线测试，不对应真实知识库内容。
TEST_DOCUMENT_ID = "f" * 64
TEST_CHUNK_ID = (
    TEST_DOCUMENT_ID + ":000001"
)


async def publish_runner_prefix(
    observer: AgentRunnerSseObserver,
) -> None:
    """发布进入Runner前必须存在的三个请求级事件。

    三个事件必须通过完整诊断Observer发布，测试不能
    绕过Observer直接操作Publisher，否则无法验证适配逻辑。
    """

    await observer.on_request_received(
        image_count=0,
        has_task_goal=True,
    )

    await observer.on_safety_classified(
        disposition="continue_to_planner",
        reason_codes=(
            "knowledge_evidence_required",
        ),
    )

    await observer.on_tool_scope_decided(
        required_capabilities=(
            "knowledge_evidence",
        ),
        allowed_tool_names=(
            "search_knowledge",
        ),
    )


def make_human_review_response(
) -> AgentDiagnosisResponse:
    """构造请求策略直接转人工的合法最终响应。

    该响应没有运行Planner或工具，因此执行摘要使用
    request_policy_finished，并明确给出人工审核原因。
    """

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


def make_search_call(
    *,
    call_id: str = "call-search-001",
    query: str = "PRIVATE_QUERY",
) -> ToolCall:
    """创建一条尚未执行的知识检索调用。"""

    return ToolCall(
        call_id=call_id,
        tool_name="search_knowledge",
        arguments={
            "query": query,
            "top_k": 3,
        },
    )


def make_search_output(
) -> SearchKnowledgeToolOutput:
    """创建含一条敏感占位正文的合法检索输出。"""

    return SearchKnowledgeToolOutput(
        citations=[
            KnowledgeCitation(
                chunk_id=TEST_CHUNK_ID,
                document_id=TEST_DOCUMENT_ID,
                source_file="robot-demo.txt",
                page_or_section=(
                    "section: ERR-DEMO-1001"
                ),
                chunk_index=1,
                rank=1,
                similarity=0.81,
                rrf_score=0.032,
                excerpt=(
                    "PRIVATE_EVIDENCE_BODY"
                ),
            ),
        ],
        retrieval_ms=7.5,
    )


def make_success_result(
    tool_call: ToolCall,
) -> ToolExecutionResult:
    """把合法检索输出包装成统一success执行结果。"""

    return ToolExecutionResult(
        call_id=tool_call.call_id,
        tool_name=tool_call.tool_name,
        status="success",
        output=(
            make_search_output().model_dump(
                mode="python"
            )
        ),
        duration_ms=12.5,
    )


def make_empty_result(
    tool_call: ToolCall,
) -> ToolExecutionResult:
    """创建没有取得可用知识证据的empty结果。"""

    return ToolExecutionResult(
        call_id=tool_call.call_id,
        tool_name=tool_call.tool_name,
        status="empty",
        error_code="empty_result",
        public_message=(
            "工具没有返回可用结果"
        ),
        duration_ms=3.0,
    )


def make_progress(
    state: AgentProgressState,
) -> AgentProgress:
    """为七种业务状态构造满足关系约束的进度快照。"""

    if state == "collecting":
        return AgentProgress(
            state=state,
            required_capabilities=(
                "knowledge_evidence",
            ),
            allowed_next_tool_names=(
                "search_knowledge",
            ),
        )

    if state == "ready_to_finish":
        return AgentProgress(
            state=state,
            required_capabilities=(
                "knowledge_evidence",
            ),
            completed_capabilities=(
                "knowledge_evidence",
            ),
        )

    if state == "completed":
        return AgentProgress(
            state=state,
            required_capabilities=(
                "knowledge_evidence",
            ),
            completed_capabilities=(
                "knowledge_evidence",
            ),
            stop_reason=(
                "sufficient_evidence"
            ),
        )

    if state == "partial":
        return AgentProgress(
            state=state,
            required_capabilities=(
                "knowledge_evidence",
                "current_time",
            ),
            completed_capabilities=(
                "knowledge_evidence",
            ),
            missing_information=(
                "尚未取得当前时间观察",
            ),
            stop_reason="partial_evidence",
        )

    if state == "abstained":
        return AgentProgress(
            state=state,
            required_capabilities=(
                "knowledge_evidence",
            ),
            missing_information=(
                "知识库没有返回可用证据",
            ),
            stop_reason=(
                "insufficient_information"
            ),
        )

    if state == "human_review_required":
        return AgentProgress(
            state=state,
            required_capabilities=(
                "knowledge_evidence",
            ),
            missing_information=(
                "需要由有权限人员审核",
            ),
            stop_reason=(
                "human_review_required"
            ),
        )

    return AgentProgress(
        state="failed",
        required_capabilities=(
            "knowledge_evidence",
        ),
        missing_information=(
            "运行时处理未完成",
        ),
        stop_reason="runtime_failure",
    )


async def make_observer_with_active_tool(
) -> tuple[
    AgentEventPublisher,
    AgentRunnerSseObserver,
    ToolCall,
]:
    """创建已经发布到tool_started阶段的测试事件流。"""

    publisher = AgentEventPublisher(
        request_id=TEST_REQUEST_ID,
    )
    observer = AgentRunnerSseObserver(
        publisher=publisher,
    )
    tool_call = make_search_call()

    await publish_runner_prefix(observer)
    await observer.on_planning_started(
        step_number=1,
        allowed_tool_names=(
            "search_knowledge",
        ),
    )
    await observer.on_tool_started(
        step_id=1,
        tool_call=tool_call,
    )

    return publisher, observer, tool_call


def test_observer_constructor_preserves_publisher(
) -> None:
    """构造器应保存单次请求Publisher并拒绝错误类型。

    被测试模块是AgentRunnerSseObserver.__init__()和
    publisher属性。合法实例应返回同一个Publisher对象，
    非Publisher对象应在Runner开始前被TypeError拒绝。
    """

    publisher = AgentEventPublisher(
        request_id=TEST_REQUEST_ID,
    )
    observer = AgentRunnerSseObserver(
        publisher=publisher,
    )

    assert observer.publisher is publisher

    with pytest.raises(
        TypeError,
        match=(
            "publisher必须是"
            "AgentEventPublisher"
        ),
    ):
        AgentRunnerSseObserver(
            publisher=object(),  # type: ignore[arg-type]
        )


def test_observer_implements_complete_async_protocol(
) -> None:
    """Observer必须满足诊断服务和Runner的八个异步回调。

    AgentDiagnosisService与AgentRunner都会await这些方法，
    因此使用inspect.iscoroutinefunction检查完整协议边界。
    """

    assert iscoroutinefunction(
        AgentRunnerSseObserver
        .on_request_received
    )
    assert iscoroutinefunction(
        AgentRunnerSseObserver
        .on_safety_classified
    )
    assert iscoroutinefunction(
        AgentRunnerSseObserver
        .on_tool_scope_decided
    )
    assert iscoroutinefunction(
        AgentRunnerSseObserver
        .on_planning_started
    )
    assert iscoroutinefunction(
        AgentRunnerSseObserver
        .on_tool_started
    )
    assert iscoroutinefunction(
        AgentRunnerSseObserver
        .on_tool_finished
    )
    assert iscoroutinefunction(
        AgentRunnerSseObserver
        .on_progress_updated
    )
    assert iscoroutinefunction(
        AgentRunnerSseObserver
        .on_diagnosis_finished
    )


@pytest.mark.asyncio
async def test_observer_publishes_complete_runner_sequence(
) -> None:
    """四类Runner通知应形成连续且结构正确的公开事件。

    测试先发布三个请求级前缀事件，再模拟一次成功检索和
    ready_to_finish进度。预期Observer追加第4至第7条事件，
    并由Publisher保持严格顺序。
    """

    private_query = (
        "PRIVATE_QUERY ERR-NET-4001"
    )
    publisher = AgentEventPublisher(
        request_id=TEST_REQUEST_ID,
    )
    observer = AgentRunnerSseObserver(
        publisher=publisher,
    )
    tool_call = make_search_call(
        query=private_query,
    )

    await publish_runner_prefix(observer)
    await observer.on_planning_started(
        step_number=1,
        allowed_tool_names=(
            "search_knowledge",
        ),
    )
    await observer.on_tool_started(
        step_id=1,
        tool_call=tool_call,
    )
    await observer.on_tool_finished(
        step_id=1,
        tool_call=tool_call,
        result=make_success_result(
            tool_call
        ),
    )
    await observer.on_progress_updated(
        progress=make_progress(
            "ready_to_finish"
        ),
    )

    events = publisher.events

    assert [
        event.sequence
        for event in events
    ] == list(range(1, 8))
    assert [
        event.event_type
        for event in events
    ] == [
        "request_received",
        "safety_classified",
        "tool_scope_decided",
        "planning_started",
        "tool_started",
        "tool_finished",
        "progress_updated",
    ]

    planning_event = events[3]
    started_event = events[4]
    finished_event = events[5]
    progress_event = events[6]

    assert isinstance(
        planning_event,
        AgentPlanningStartedEvent,
    )
    assert planning_event.step_number == 1

    assert isinstance(
        started_event,
        AgentToolStartedEvent,
    )
    assert started_event.step_id == 1
    assert private_query not in (
        started_event.input_summary
    )
    assert started_event.input_summary == (
        "知识库检索；"
        f"query_chars={len(private_query)}；"
        "top_k=3"
    )

    assert isinstance(
        finished_event,
        AgentToolFinishedEvent,
    )
    assert finished_event.trace.status == (
        "success"
    )
    assert finished_event.trace.result_summary == (
        "知识库返回1条已确认引用；"
        "retrieval_ms=7.500"
    )
    assert "PRIVATE_EVIDENCE_BODY" not in (
        finished_event.trace.result_summary
    )

    assert isinstance(
        progress_event,
        AgentProgressUpdatedEvent,
    )
    assert progress_event.state == "planning"
    assert progress_event.progress.state == (
        "ready_to_finish"
    )

    # 前三个事件也必须来自Observer的请求级回调，
    # 不能只是满足Publisher顺序的测试占位数据。
    assert isinstance(
        events[0],
        AgentRequestReceivedEvent,
    )
    assert events[0].image_count == 0
    assert events[0].has_task_goal is True

    assert isinstance(
        events[1],
        AgentSafetyClassifiedEvent,
    )
    assert events[1].disposition == (
        "continue_to_planner"
    )

    assert isinstance(
        events[2],
        AgentToolScopeDecidedEvent,
    )
    assert events[2].allowed_tool_names == (
        "search_knowledge",
    )


@pytest.mark.asyncio
async def test_observer_publishes_policy_terminal_response(
) -> None:
    """请求策略直接转人工也必须形成完整终态事件流。

    测试不运行Runner：请求接收后，安全策略直接返回
    human_review_required，Observer随后发布同一份最终响应。
    预期顺序为request_received、safety_classified和
    diagnosis_finished，且不会伪造工具范围或工具事件。
    """

    publisher = AgentEventPublisher(
        request_id=TEST_REQUEST_ID,
    )
    observer = AgentRunnerSseObserver(
        publisher=publisher,
    )
    response = make_human_review_response()

    await observer.on_request_received(
        image_count=1,
        has_task_goal=False,
    )
    await observer.on_safety_classified(
        disposition=(
            "human_review_required"
        ),
        reason_codes=(
            "high_risk_control_detected",
            "no_safe_tool_required",
        ),
    )
    await observer.on_diagnosis_finished(
        response=response,
    )

    events = publisher.events

    assert [
        event.sequence
        for event in events
    ] == [1, 2, 3]
    assert [
        event.event_type
        for event in events
    ] == [
        "request_received",
        "safety_classified",
        "diagnosis_finished",
    ]
    assert events[1].state == "completed"

    final_event = events[2]

    assert isinstance(
        final_event,
        AgentDiagnosisFinishedEvent,
    )
    assert final_event.response is response
    assert final_event.state == "completed"
    assert publisher.closed is True


@pytest.mark.asyncio
async def test_diagnosis_finished_rejects_wrong_type(
) -> None:
    """最终响应回调必须拒绝非AgentDiagnosisResponse对象。

    类型检查发生在Publisher之前，所以错误对象不会关闭
    事件流，也不会产生一条内容不可信的终端事件。
    """

    publisher = AgentEventPublisher(
        request_id=TEST_REQUEST_ID,
    )
    observer = AgentRunnerSseObserver(
        publisher=publisher,
    )

    await observer.on_request_received(
        image_count=0,
        has_task_goal=True,
    )

    with pytest.raises(
        TypeError,
        match=(
            "response必须是"
            "AgentDiagnosisResponse"
        ),
    ):
        await observer.on_diagnosis_finished(
            response=object(),  # type: ignore[arg-type]
        )

    assert len(publisher.events) == 1
    assert publisher.closed is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "progress_state",
        "expected_lifecycle_state",
    ),
    [
        ("collecting", "observing"),
        ("ready_to_finish", "planning"),
        ("completed", "completed"),
        ("partial", "completed"),
        ("abstained", "completed"),
        (
            "human_review_required",
            "completed",
        ),
        ("failed", "aborted"),
    ],
)
async def test_progress_states_map_to_public_lifecycle(
    progress_state: AgentProgressState,
    expected_lifecycle_state: str,
) -> None:
    """七种业务进度必须映射到正确的公开生命周期。

    每个参数化案例都会建立合法工具事件前缀，再发布一份
    经过AgentProgress契约校验的快照。预期事件state与
    AgentProgressUpdatedEvent的关系校验完全一致。
    """

    (
        publisher,
        observer,
        tool_call,
    ) = await make_observer_with_active_tool()

    await observer.on_tool_finished(
        step_id=1,
        tool_call=tool_call,
        result=make_empty_result(
            tool_call
        ),
    )
    await observer.on_progress_updated(
        progress=make_progress(
            progress_state
        ),
    )

    event = publisher.events[-1]

    assert isinstance(
        event,
        AgentProgressUpdatedEvent,
    )
    assert event.state == (
        expected_lifecycle_state
    )
    assert event.progress.state == (
        progress_state
    )


@pytest.mark.asyncio
async def test_tool_finished_rejects_mismatched_result(
) -> None:
    """不同call_id的结果不能与当前工具调用错误配对。

    on_tool_finished()预期先构造AgentToolInteraction，
    由其关系校验拒绝不匹配结果，Publisher事件历史不能
    出现伪造的tool_finished。
    """

    (
        publisher,
        observer,
        tool_call,
    ) = await make_observer_with_active_tool()
    event_count_before = len(
        publisher.events
    )

    mismatched_result = ToolExecutionResult(
        call_id="different-call-id",
        tool_name=tool_call.tool_name,
        status="empty",
        error_code="empty_result",
        public_message=(
            "工具没有返回可用结果"
        ),
        duration_ms=1.0,
    )

    with pytest.raises(
        ValidationError,
        match=(
            "tool_call与result的"
            "call_id不一致"
        ),
    ):
        await observer.on_tool_finished(
            step_id=1,
            tool_call=tool_call,
            result=mismatched_result,
        )

    assert len(publisher.events) == (
        event_count_before
    )
    assert publisher.events[-1].event_type == (
        "tool_started"
    )


@pytest.mark.asyncio
async def test_tool_finished_rejects_invalid_success_output(
) -> None:
    """已知工具的非法success字典不能进入SSE事件流。

    测试绕过ToolExecutor构造错误输出。公共轨迹构造器应
    使用SearchKnowledgeToolOutput重新校验并抛出TypeError，
    Publisher不能提交tool_finished事件。
    """

    (
        publisher,
        observer,
        tool_call,
    ) = await make_observer_with_active_tool()
    event_count_before = len(
        publisher.events
    )

    invalid_result = ToolExecutionResult(
        call_id=tool_call.call_id,
        tool_name=tool_call.tool_name,
        status="success",
        output={
            "private_invalid_output": (
                "DO_NOT_EXPOSE"
            ),
        },
        duration_ms=1.0,
    )

    with pytest.raises(
        TypeError,
        match=(
            "search_knowledge"
            "成功结果不符合输出契约"
        ),
    ) as exception_info:
        await observer.on_tool_finished(
            step_id=1,
            tool_call=tool_call,
            result=invalid_result,
        )

    assert "DO_NOT_EXPOSE" not in str(
        exception_info.value
    )
    assert len(publisher.events) == (
        event_count_before
    )


@pytest.mark.asyncio
async def test_progress_callback_rejects_wrong_type(
) -> None:
    """进度回调应在Publisher之前拒绝错误对象类型。

    这样错误会稳定表现为TypeError，而不是在读取state时
    偶然出现AttributeError，也不会提交半条公开事件。
    """

    (
        publisher,
        observer,
        tool_call,
    ) = await make_observer_with_active_tool()

    await observer.on_tool_finished(
        step_id=1,
        tool_call=tool_call,
        result=make_empty_result(
            tool_call
        ),
    )
    event_count_before = len(
        publisher.events
    )

    with pytest.raises(
        TypeError,
        match="progress必须是AgentProgress",
    ):
        await observer.on_progress_updated(
            progress=object(),  # type: ignore[arg-type]
        )

    assert len(publisher.events) == (
        event_count_before
    )
