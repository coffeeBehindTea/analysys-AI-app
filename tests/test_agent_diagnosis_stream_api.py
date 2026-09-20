"""Robot Diagnostic Agent SSE接口的离线HTTP测试。

本文件验证Router这一层的职责：

1. 接收并校验AgentDiagnosisRequest；
2. 把Middleware生成的request_id交给流式Service；
3. 把结构化事件编码成SSE帧；
4. 设置流式响应所需的媒体类型和缓存控制头；
5. 在流开始后使用stream_error表达错误；
6. 在OpenAPI中公开稳定的POST接口。

测试使用HTTPX ASGITransport直接调用内存FastAPI应用，
不启动Uvicorn，也不调用真实LLM、Vision、Embedding或Agent工具。
"""

import json

from collections.abc import (
    AsyncIterator,
)

import pytest

from fastapi import FastAPI
from httpx import (
    ASGITransport,
    AsyncClient,
)

from app.dependencies import (
    get_agent_diagnosis_streaming_service,
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
from app.schemas.error import ErrorDetail
from main import create_app


# 测试请求体不包含真实机器人、客户或生产日志。
VALID_PAYLOAD = {
    "robot_id": "robot-demo-001",
    "symptom": "模拟网络恢复后仍处于暂停状态",
    "log_excerpt": "ERR-DEMO-1001 simulated timeout",
    "task_goal": "仅检查现有证据是否足够",
}

# 固定公开消息用于确认编码器保留UTF-8中文。
RECEIVED_MESSAGE = "诊断请求已经接收"
FINISHED_MESSAGE = "最终诊断已经生成"


def make_abstained_response(
    request_id: str,
) -> AgentDiagnosisResponse:
    """构造证据不足但业务上正常结束的最终响应。"""

    missing_information = (
        "知识库没有返回足够的模拟证据"
    )

    return AgentDiagnosisResponse(
        request_id=request_id,
        diagnosis=DiagnosisReport(
            request_id=request_id,
            prompt_version="agent-tool-calling-v2",
            gate_version="agent-confirmed-evidence-v1",
            status="abstained",
            symptoms=[
                DiagnosisSymptom(
                    description=VALID_PAYLOAD["symptom"],
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
        ),
        execution=AgentExecutionSummary(
            planner_prompt_version=(
                "agent-tool-calling-v2"
            ),
            state="completed",
            termination_reason="planner_finished",
            termination_message="Agent规划正常结束",
            finish_reason="insufficient_information",
            step_count=0,
            steps=[],
            missing_information=[
                missing_information,
            ],
        ),
    )


class CompletedStreamingService:
    """记录Router输入并返回两个合法事件的Fake Service。"""

    def __init__(self) -> None:
        # 每项保存Pydantic请求对象和Middleware请求ID。
        self.calls: list[
            tuple[AgentDiagnosisRequest, str]
        ] = []

    async def stream(
        self,
        request: AgentDiagnosisRequest,
        request_id: str,
    ) -> AsyncIterator[AgentSseEvent]:
        """模拟请求接收后形成证据不足的最终诊断。"""

        self.calls.append(
            (request, request_id)
        )

        yield AgentRequestReceivedEvent(
            sequence=1,
            request_id=request_id,
            message=RECEIVED_MESSAGE,
            image_count=len(request.images),
            has_task_goal=(
                request.task_goal is not None
            ),
        )

        yield AgentDiagnosisFinishedEvent(
            sequence=2,
            request_id=request_id,
            state="completed",
            message=FINISHED_MESSAGE,
            response=make_abstained_response(
                request_id
            ),
        )


class ErrorStreamingService:
    """模拟HTTP流开始后发生可重试上游错误的Fake。"""

    def __init__(self) -> None:
        self.call_count = 0

    async def stream(
        self,
        request: AgentDiagnosisRequest,
        request_id: str,
    ) -> AsyncIterator[AgentSseEvent]:
        """先发布接收事件，再以stream_error安全结束。"""

        self.call_count += 1

        yield AgentRequestReceivedEvent(
            sequence=1,
            request_id=request_id,
            message=RECEIVED_MESSAGE,
            image_count=len(request.images),
            has_task_goal=(
                request.task_goal is not None
            ),
        )

        yield AgentStreamErrorEvent(
            sequence=2,
            request_id=request_id,
            message="诊断事件流意外中断",
            error=ErrorDetail(
                code="llm_timeout",
                message="LLM服务响应超时",
            ),
            retryable=True,
        )


def create_test_client(
    application: FastAPI,
) -> AsyncClient:
    """创建不经过真实网络的异步HTTP客户端。"""

    return AsyncClient(
        transport=ASGITransport(
            app=application,
        ),
        base_url="http://testserver",
    )


def parse_sse_frames(
    response_text: str,
) -> list[dict[str, object]]:
    """把Router返回的SSE文本还原成可断言的事件字典。"""

    frames: list[dict[str, object]] = []

    for raw_frame in response_text.strip().split(
        "\n\n"
    ):
        fields: dict[str, str] = {}

        for line in raw_frame.splitlines():
            field_name, field_value = line.split(
                ": ",
                maxsplit=1,
            )
            fields[field_name] = field_value

        frames.append(
            {
                "id": int(fields["id"]),
                "event": fields["event"],
                "data": json.loads(fields["data"]),
            }
        )

    return frames


@pytest.mark.asyncio
async def test_stream_route_returns_ordered_sse_frames(
) -> None:
    """合法请求应得到同一request_id下的有序SSE事件。"""

    fake_service = CompletedStreamingService()
    application = create_app()
    application.dependency_overrides[
        get_agent_diagnosis_streaming_service
    ] = lambda: fake_service

    try:
        async with create_test_client(
            application
        ) as client:
            response = await client.post(
                "/api/v1/agent/diagnose/stream",
                json=VALID_PAYLOAD,
            )
    finally:
        application.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.headers[
        "content-type"
    ].startswith("text/event-stream")
    assert response.headers["cache-control"] == (
        "no-cache"
    )
    assert response.headers[
        "x-accel-buffering"
    ] == "no"

    assert len(fake_service.calls) == 1
    received_request, received_request_id = (
        fake_service.calls[0]
    )
    assert isinstance(
        received_request,
        AgentDiagnosisRequest,
    )
    assert received_request.robot_id == (
        VALID_PAYLOAD["robot_id"]
    )
    assert received_request_id == (
        response.headers["x-request-id"]
    )

    frames = parse_sse_frames(response.text)
    assert [frame["id"] for frame in frames] == [
        1,
        2,
    ]
    assert [
        frame["event"]
        for frame in frames
    ] == [
        "request_received",
        "diagnosis_finished",
    ]
    assert all(
        frame["data"]["request_id"]
        == received_request_id
        for frame in frames
    )
    assert frames[0]["data"]["message"] == (
        RECEIVED_MESSAGE
    )
    assert frames[-1]["data"]["response"][
        "diagnosis"
    ]["abstained"] is True


@pytest.mark.asyncio
async def test_stream_route_keeps_http_200_for_stream_error(
) -> None:
    """流开始后的失败应使用终端事件，不能伪造成功诊断。"""

    fake_service = ErrorStreamingService()
    application = create_app()
    application.dependency_overrides[
        get_agent_diagnosis_streaming_service
    ] = lambda: fake_service

    try:
        async with create_test_client(
            application
        ) as client:
            response = await client.post(
                "/api/v1/agent/diagnose/stream",
                json=VALID_PAYLOAD,
            )
    finally:
        application.dependency_overrides.clear()

    assert response.status_code == 200
    assert fake_service.call_count == 1

    frames = parse_sse_frames(response.text)
    assert [frame["event"] for frame in frames] == [
        "request_received",
        "stream_error",
    ]
    assert frames[-1]["data"]["error"] == {
        "code": "llm_timeout",
        "message": "LLM服务响应超时",
    }
    assert frames[-1]["data"]["retryable"] is True
    assert "diagnosis_finished" not in {
        frame["event"]
        for frame in frames
    }


@pytest.mark.asyncio
async def test_stream_route_rejects_invalid_body_before_stream(
) -> None:
    """请求体缺少robot_id时应返回422且不消费事件流。"""

    fake_service = ErrorStreamingService()
    application = create_app()
    application.dependency_overrides[
        get_agent_diagnosis_streaming_service
    ] = lambda: fake_service

    try:
        async with create_test_client(
            application
        ) as client:
            response = await client.post(
                "/api/v1/agent/diagnose/stream",
                json={
                    "symptom": "模拟故障",
                    "log_excerpt": "ERR-DEMO-1001",
                },
            )
    finally:
        application.dependency_overrides.clear()

    assert response.status_code == 422
    assert response.headers["x-request-id"]
    assert fake_service.call_count == 0


@pytest.mark.asyncio
async def test_stream_route_rejects_get_method(
) -> None:
    """流式诊断只允许POST，GET必须返回405。"""

    fake_service = ErrorStreamingService()
    application = create_app()
    application.dependency_overrides[
        get_agent_diagnosis_streaming_service
    ] = lambda: fake_service

    try:
        async with create_test_client(
            application
        ) as client:
            response = await client.get(
                "/api/v1/agent/diagnose/stream"
            )
    finally:
        application.dependency_overrides.clear()

    assert response.status_code == 405
    assert fake_service.call_count == 0


def test_stream_route_is_present_in_openapi_contract(
) -> None:
    """OpenAPI必须公开POST路径和text/event-stream响应。"""

    application = create_app()
    operation = application.openapi()["paths"][
        "/api/v1/agent/diagnose/stream"
    ]["post"]

    assert "get" not in application.openapi()["paths"][
        "/api/v1/agent/diagnose/stream"
    ]
    assert "text/event-stream" in operation[
        "responses"
    ]["200"]["content"]

