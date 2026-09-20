"""Agent诊断SSE流式编排服务的离线测试。

本文件验证：

1. 正常诊断事件会按发布顺序流出并以diagnosis_finished结束；
2. 已知ApplicationError会转换成稳定的stream_error；
3. 输入错误不会被错误标记为可重试；
4. 未知异常不会向公开事件泄露内部详情；
5. 诊断服务漏发终端事件时不会让消费者永久等待；
6. 调用方提前停止消费时会取消后台诊断任务；
7. 不同请求使用独立的sequence和request_id；
8. 构造参数和请求参数会在明确边界接受校验。

所有测试都使用Fake诊断服务，不启动FastAPI，
也不调用LLM、Vision、Embedding、Chroma或真实Agent工具。
"""

import asyncio

from collections.abc import (
    AsyncIterator,
)

import pytest

from app.agent.diagnosis_observer import (
    AgentDiagnosisObserver,
)
from app.errors import (
    LLMTimeoutError,
    VisionInputValidationError,
)
from app.schemas.agent_api import (
    AgentDiagnosisRequest,
    AgentDiagnosisResponse,
    AgentExecutionSummary,
)
from app.schemas.agent_events import (
    AgentDiagnosisFinishedEvent,
    AgentRequestReceivedEvent,
    AgentSseEvent,
    AgentStreamErrorEvent,
)
from app.schemas.diagnostics import (
    DiagnosisReport,
    DiagnosisSymptom,
)
from app.services.agent_diagnosis_streaming_service import (
    AgentDiagnosisStreamingService,
)


# 两个固定请求ID用于验证单请求Publisher隔离。
# 它们只属于离线测试，不对应真实HTTP请求。
TEST_REQUEST_ID = "request-streaming-service-001"
SECOND_REQUEST_ID = "request-streaming-service-002"


def make_request(
) -> AgentDiagnosisRequest:
    """创建不包含真实日志、图片或设备数据的固定请求。"""

    return AgentDiagnosisRequest(
        robot_id="robot-demo-001",
        symptom="模拟网络恢复后仍处于暂停状态",
        log_excerpt=(
            "ERR-DEMO-1001 simulated heartbeat timeout"
        ),
        task_goal="仅检查现有证据是否足够",
        images=[],
    )


def make_abstained_response(
    request_id: str,
) -> AgentDiagnosisResponse:
    """构造Planner正常结束但证据不足的合法公开响应。

    该响应用于验证流式编排，不表示运行时异常。
    因此execution.state仍然是completed，业务诊断状态是abstained。
    """

    missing_information = (
        "知识库没有返回足够的模拟证据"
    )

    diagnosis = DiagnosisReport(
        request_id=request_id,
        prompt_version="agent-tool-calling-v2",
        gate_version="agent-confirmed-evidence-v1",
        status="abstained",
        symptoms=[
            DiagnosisSymptom(
                description="模拟网络故障",
                source="user_report",
            ),
        ],
        evidence=[],
        possible_causes=[],
        next_checks=[],
        risk_level="unknown",
        missing_information=[
            missing_information,
        ],
        abstained=True,
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
        finish_reason=(
            "insufficient_information"
        ),
        step_count=0,
        steps=[],
        missing_information=[
            missing_information,
        ],
    )

    return AgentDiagnosisResponse(
        request_id=request_id,
        diagnosis=diagnosis,
        execution=execution,
    )


async def publish_safe_planning_prefix(
    observer: AgentDiagnosisObserver,
) -> None:
    """通过Observer发布进入Planner前的合法公开事件前缀。"""

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
    await observer.on_planning_started(
        step_number=1,
        allowed_tool_names=(
            "search_knowledge",
        ),
    )


async def collect_events(
    stream: AsyncIterator[AgentSseEvent],
) -> list[AgentSseEvent]:
    """消费一个结构化事件流直到其终端事件。"""

    return [
        event
        async for event in stream
    ]


class CompletedDiagnosisService:
    """发布完整事件并返回合法响应的异步Fake。"""

    def __init__(self) -> None:
        self.request_ids: list[str] = []

    async def diagnose(
        self,
        request: AgentDiagnosisRequest,
        request_id: str,
        *,
        observer: (
            AgentDiagnosisObserver | None
        ) = None,
    ) -> AgentDiagnosisResponse:
        """模拟安全分类、规划和最终诊断的正常流程。"""

        assert isinstance(
            request,
            AgentDiagnosisRequest,
        )
        assert observer is not None

        self.request_ids.append(request_id)

        await publish_safe_planning_prefix(
            observer
        )

        response = make_abstained_response(
            request_id
        )

        await observer.on_diagnosis_finished(
            response=response,
        )

        return response


class RaisingDiagnosisService:
    """在发布任何事件前抛出预设异常的异步Fake。"""

    def __init__(
        self,
        error: Exception,
    ) -> None:
        self._error = error

    async def diagnose(
        self,
        request: AgentDiagnosisRequest,
        request_id: str,
        *,
        observer: (
            AgentDiagnosisObserver | None
        ) = None,
    ) -> AgentDiagnosisResponse:
        """抛出异常，让流式编排服务负责公开错误转换。"""

        raise self._error


class MissingTerminalDiagnosisService:
    """返回响应但故意漏发最终事件的错误Fake实现。"""

    async def diagnose(
        self,
        request: AgentDiagnosisRequest,
        request_id: str,
        *,
        observer: (
            AgentDiagnosisObserver | None
        ) = None,
    ) -> AgentDiagnosisResponse:
        """发布合法前缀后直接返回，用于模拟错误接线。"""

        assert observer is not None

        await publish_safe_planning_prefix(
            observer
        )

        return make_abstained_response(
            request_id
        )


class BlockingDiagnosisService:
    """发布首事件后持续等待，直到生产任务被取消。"""

    def __init__(self) -> None:
        self.cancelled = asyncio.Event()

    async def diagnose(
        self,
        request: AgentDiagnosisRequest,
        request_id: str,
        *,
        observer: (
            AgentDiagnosisObserver | None
        ) = None,
    ) -> AgentDiagnosisResponse:
        """模拟仍在等待慢速上游服务的诊断任务。"""

        assert observer is not None

        await observer.on_request_received(
            image_count=0,
            has_task_goal=True,
        )

        try:
            # 一个未设置的Event会一直等待，
            # 直到流式服务取消当前生产任务。
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


class SynchronousDiagnosisService:
    """故意提供同步diagnose方法的非法构造依赖。"""

    def diagnose(
        self,
        request: AgentDiagnosisRequest,
        request_id: str,
    ) -> object:
        """返回普通对象，不满足异步Provider协议。"""

        return object()


@pytest.mark.asyncio
async def test_stream_yields_normal_sequence_and_terminal_response(
) -> None:
    """正常诊断必须逐条返回事件并以最终响应结束。

    被测试模块是AgentDiagnosisStreamingService.stream()。
    CompletedDiagnosisService通过Observer发布五条事件。
    预期stream()保持顺序、sequence连续，最后一条事件
    是携带同一request_id的AgentDiagnosisFinishedEvent。
    """

    fake = CompletedDiagnosisService()
    service = AgentDiagnosisStreamingService(
        diagnosis_service=fake,
    )

    events = await collect_events(
        service.stream(
            make_request(),
            TEST_REQUEST_ID,
        )
    )

    assert [
        event.sequence
        for event in events
    ] == [1, 2, 3, 4, 5]
    assert [
        event.event_type
        for event in events
    ] == [
        "request_received",
        "safety_classified",
        "tool_scope_decided",
        "planning_started",
        "diagnosis_finished",
    ]

    final_event = events[-1]

    assert isinstance(
        final_event,
        AgentDiagnosisFinishedEvent,
    )
    assert final_event.request_id == (
        TEST_REQUEST_ID
    )
    assert final_event.response.diagnosis.status == (
        "abstained"
    )
    assert fake.request_ids == [
        TEST_REQUEST_ID
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "error",
        "expected_code",
        "expected_retryable",
    ),
    [
        (
            LLMTimeoutError(
                "PRIVATE_LLM_TIMEOUT_DETAIL"
            ),
            "llm_timeout",
            True,
        ),
        (
            VisionInputValidationError(
                "PRIVATE_INVALID_IMAGE_DETAIL"
            ),
            "vision_input_validation_error",
            False,
        ),
    ],
)
async def test_application_error_becomes_safe_stream_error(
    error: Exception,
    expected_code: str,
    expected_retryable: bool,
) -> None:
    """已知应用异常必须复用统一错误码并正确标记重试性。

    Fake在首事件前抛错。预期编排服务先补发一条真实的
    request_received，再发布stream_error并结束；异常内部
    文本不能进入公开事件JSON。
    """

    service = AgentDiagnosisStreamingService(
        diagnosis_service=(
            RaisingDiagnosisService(error)
        ),
    )

    events = await collect_events(
        service.stream(
            make_request(),
            TEST_REQUEST_ID,
        )
    )

    assert len(events) == 2
    assert isinstance(
        events[0],
        AgentRequestReceivedEvent,
    )

    error_event = events[1]

    assert isinstance(
        error_event,
        AgentStreamErrorEvent,
    )
    assert error_event.error.code == (
        expected_code
    )
    assert error_event.retryable is (
        expected_retryable
    )
    assert str(error) not in (
        error_event.model_dump_json()
    )


@pytest.mark.asyncio
async def test_unknown_error_is_redacted_and_not_retryable(
) -> None:
    """未知程序异常只能返回统一内部错误。

    Fake抛出含敏感占位内容的RuntimeError。预期公开事件
    使用internal_stream_error，retryable=False，并且事件
    序列化内容中不存在原始异常详情。
    """

    private_detail = (
        "PRIVATE_INTERNAL_PATH_AND_SECRET"
    )
    service = AgentDiagnosisStreamingService(
        diagnosis_service=(
            RaisingDiagnosisService(
                RuntimeError(private_detail)
            )
        ),
    )

    events = await collect_events(
        service.stream(
            make_request(),
            TEST_REQUEST_ID,
        )
    )

    error_event = events[-1]

    assert isinstance(
        error_event,
        AgentStreamErrorEvent,
    )
    assert error_event.error.code == (
        "internal_stream_error"
    )
    assert error_event.retryable is False
    assert private_detail not in (
        error_event.model_dump_json()
    )


@pytest.mark.asyncio
async def test_missing_terminal_event_is_converted_without_hanging(
) -> None:
    """诊断服务漏发终端事件时必须安全结束而不能挂起。

    MissingTerminalDiagnosisService返回合法响应，但没有调用
    on_diagnosis_finished()。asyncio.wait_for设置一秒测试上限。
    预期编排服务检测publisher.closed仍为False，发布内部
    stream_error，并在超时前结束异步迭代。
    """

    service = AgentDiagnosisStreamingService(
        diagnosis_service=(
            MissingTerminalDiagnosisService()
        ),
    )

    events = await asyncio.wait_for(
        collect_events(
            service.stream(
                make_request(),
                TEST_REQUEST_ID,
            )
        ),
        timeout=1.0,
    )

    assert events[-2].event_type == (
        "planning_started"
    )
    assert isinstance(
        events[-1],
        AgentStreamErrorEvent,
    )
    assert events[-1].error.code == (
        "internal_stream_error"
    )


@pytest.mark.asyncio
async def test_closing_stream_cancels_background_diagnosis(
) -> None:
    """调用方提前关闭事件流时必须取消后台生产任务。

    BlockingDiagnosisService发布首事件后永久等待。测试读取
    第一条事件后调用异步生成器的aclose()。预期stream()
    进入finally、调用producer_task.cancel()，Fake收到
    CancelledError并设置cancelled事件。
    """

    fake = BlockingDiagnosisService()
    service = AgentDiagnosisStreamingService(
        diagnosis_service=fake,
    )
    stream = service.stream(
        make_request(),
        TEST_REQUEST_ID,
    )

    first_event = await anext(stream)

    assert isinstance(
        first_event,
        AgentRequestReceivedEvent,
    )

    await stream.aclose()

    await asyncio.wait_for(
        fake.cancelled.wait(),
        timeout=1.0,
    )

    assert fake.cancelled.is_set()


@pytest.mark.asyncio
async def test_two_streams_use_isolated_publishers(
) -> None:
    """同一Service的不同请求不能共享事件序号或请求ID。

    测试依次运行两个请求。预期两个事件流都从sequence=1
    开始，并且每条事件只携带所属请求的request_id。
    """

    fake = CompletedDiagnosisService()
    service = AgentDiagnosisStreamingService(
        diagnosis_service=fake,
    )

    first_events = await collect_events(
        service.stream(
            make_request(),
            TEST_REQUEST_ID,
        )
    )
    second_events = await collect_events(
        service.stream(
            make_request(),
            SECOND_REQUEST_ID,
        )
    )

    assert first_events[0].sequence == 1
    assert second_events[0].sequence == 1
    assert {
        event.request_id
        for event in first_events
    } == {TEST_REQUEST_ID}
    assert {
        event.request_id
        for event in second_events
    } == {SECOND_REQUEST_ID}
    assert fake.request_ids == [
        TEST_REQUEST_ID,
        SECOND_REQUEST_ID,
    ]


def test_constructor_rejects_synchronous_service(
) -> None:
    """构造器必须在运行前拒绝同步diagnose依赖。

    流式编排必须await诊断服务；同步方法会阻塞事件循环，
    因此预期构造阶段直接抛出TypeError。
    """

    with pytest.raises(
        TypeError,
        match=(
            "diagnosis_service.diagnose"
            "必须是异步方法"
        ),
    ):
        AgentDiagnosisStreamingService(
            diagnosis_service=(
                SynchronousDiagnosisService()
            ),
        )


@pytest.mark.parametrize(
    (
        "invalid_value",
        "expected_exception",
    ),
    [
        (True, TypeError),
        ("200", TypeError),
        (1, ValueError),
        (10_001, ValueError),
    ],
)
def test_constructor_rejects_invalid_event_limit(
    invalid_value: object,
    expected_exception: type[Exception],
) -> None:
    """max_events必须是允许至少两个事件的有限整数。

    参数化案例分别覆盖bool、字符串、过小和过大数值。
    预期类型错误与范围错误在创建Publisher之前被明确拒绝。
    """

    with pytest.raises(
        expected_exception,
    ):
        AgentDiagnosisStreamingService(
            diagnosis_service=(
                CompletedDiagnosisService()
            ),
            max_events=invalid_value,  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_stream_rejects_wrong_request_type(
) -> None:
    """stream()必须在创建后台任务前拒绝普通对象请求。

    异步生成器的函数体在首次anext()时才开始执行，
    因此测试对anext(stream)断言TypeError，而不是只调用
    stream()后立即断言。
    """

    service = AgentDiagnosisStreamingService(
        diagnosis_service=(
            CompletedDiagnosisService()
        ),
    )
    stream = service.stream(
        object(),  # type: ignore[arg-type]
        TEST_REQUEST_ID,
    )

    with pytest.raises(
        TypeError,
        match=(
            "request必须是"
            "AgentDiagnosisRequest"
        ),
    ):
        await anext(stream)
