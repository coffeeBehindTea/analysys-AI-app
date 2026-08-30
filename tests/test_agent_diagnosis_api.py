"""Robot Diagnostic Agent HTTP接口的离线契约测试。

本文件通过HTTPX ASGITransport直接调用内存FastAPI应用：

1. 不启动Uvicorn；
2. 不访问真实网络；
3. 不创建真实LLM、Embedding或Chroma客户端；
4. 使用Fake Service验证Router的参数和返回契约。
"""

import pytest

from fastapi import FastAPI
from httpx import (
    ASGITransport,
    AsyncClient,
)

from app.dependencies import (
    get_agent_diagnosis_service,
)
from app.schemas.agent_api import (
    AgentDiagnosisRequest,
    AgentDiagnosisResponse,
    AgentExecutionSummary,
)
from app.schemas.diagnostics import (
    DiagnosisCause,
    DiagnosisCheck,
    DiagnosisEvidence,
    DiagnosisReport,
    DiagnosisSymptom,
)
from main import create_app


# 固定哈希用于构造符合证据Schema的脱敏测试引用。
TEST_DOCUMENT_ID = "a" * 64
TEST_CHUNK_ID = f"{TEST_DOCUMENT_ID}:000001"

# 固定版本用于验证Router不会丢失Service生成的审计字段。
TEST_PLANNER_PROMPT_VERSION = "agent-tool-calling-v2"
TEST_GATE_VERSION = "agent-confirmed-evidence-v1"


def build_completed_response(
    *,
    request: AgentDiagnosisRequest,
    request_id: str,
) -> AgentDiagnosisResponse:
    """根据Fake Service收到的参数构造完整Agent响应。"""

    evidence = DiagnosisEvidence(
        chunk_id=TEST_CHUNK_ID,
        document_id=TEST_DOCUMENT_ID,
        source_file="robot-fault-demo.txt",
        page_or_section="section: ERR-NET-4001",
        rank=1,
        rrf_score=0.0327,
        vector_similarity=0.71,
        excerpt=(
            "通信恢复后不自动继续旧任务，"
            "状态核对后才能继续。"
        ),
    )

    diagnosis = DiagnosisReport(
        request_id=request_id,
        prompt_version=TEST_PLANNER_PROMPT_VERSION,
        gate_version=TEST_GATE_VERSION,
        status="completed",
        symptoms=[
            DiagnosisSymptom(
                description=request.symptom,
                source="user_report",
            ),
            DiagnosisSymptom(
                description=request.log_excerpt,
                source="log_excerpt",
            ),
        ],
        evidence=[evidence],
        possible_causes=[
            DiagnosisCause(
                description="旧任务仍在等待调度状态核对",
                evidence_chunk_ids=[TEST_CHUNK_ID],
            ),
        ],
        next_checks=[
            DiagnosisCheck(
                description="核对RCS/RMS任务状态",
                evidence_chunk_ids=[TEST_CHUNK_ID],
                risk_level="low",
                requires_qualified_person=False,
            ),
        ],
        risk_level="medium",
        missing_information=[],
        abstained=False,
    )

    execution = AgentExecutionSummary(
        planner_prompt_version=TEST_PLANNER_PROMPT_VERSION,
        state="completed",
        termination_reason="planner_finished",
        termination_message="Agent规划正常结束",
        finish_reason="task_completed",
        step_count=0,
        steps=[],
        missing_information=[],
    )

    return AgentDiagnosisResponse(
        request_id=request_id,
        diagnosis=diagnosis,
        execution=execution,
        telemetry_observations=[],
        test_case_drafts=[],
    )


def build_abstained_response(
    *,
    request: AgentDiagnosisRequest,
    request_id: str,
) -> AgentDiagnosisResponse:
    """构造Runner安全中止后的合法拒答响应。"""

    diagnosis = DiagnosisReport(
        request_id=request_id,
        prompt_version=TEST_PLANNER_PROMPT_VERSION,
        gate_version=TEST_GATE_VERSION,
        status="abstained",
        symptoms=[
            DiagnosisSymptom(
                description=request.symptom,
                source="user_report",
            ),
            DiagnosisSymptom(
                description=request.log_excerpt,
                source="log_excerpt",
            ),
        ],
        evidence=[],
        possible_causes=[],
        next_checks=[],
        risk_level="unknown",
        missing_information=[
            "规划服务没有返回下一步决定",
        ],
        abstained=True,
    )

    execution = AgentExecutionSummary(
        planner_prompt_version=TEST_PLANNER_PROMPT_VERSION,
        state="aborted",
        termination_reason="planner_timeout",
        termination_message="Agent规划服务响应超时",
        finish_reason=None,
        step_count=0,
        steps=[],
        missing_information=[
            "规划服务没有返回下一步决定",
        ],
    )

    return AgentDiagnosisResponse(
        request_id=request_id,
        diagnosis=diagnosis,
        execution=execution,
        telemetry_observations=[],
        test_case_drafts=[],
    )


class FakeAgentDiagnosisService:
    """记录Router调用并返回预设完整响应或拒答响应。"""

    def __init__(
        self,
        *,
        abstained: bool = False,
    ) -> None:
        self._abstained = abstained

        # 每项保存Router实际传入的请求模型和request_id。
        self.calls: list[
            tuple[AgentDiagnosisRequest, str]
        ] = []

    async def diagnose(
        self,
        request: AgentDiagnosisRequest,
        request_id: str,
    ) -> AgentDiagnosisResponse:
        """模拟异步Agent诊断，不执行真实Runner。"""

        self.calls.append(
            (request, request_id)
        )

        if self._abstained:
            return build_abstained_response(
                request=request,
                request_id=request_id,
            )

        return build_completed_response(
            request=request,
            request_id=request_id,
        )


def create_test_client(
    application: FastAPI,
) -> AsyncClient:
    """创建直接调用内存ASGI应用的异步HTTP客户端。"""

    transport = ASGITransport(
        app=application,
    )

    return AsyncClient(
        transport=transport,
        base_url="http://testserver",
    )


@pytest.mark.asyncio
async def test_agent_diagnose_returns_completed_response(
) -> None:
    """合法请求应调用Service并返回可追溯结构化结果。"""

    fake_service = FakeAgentDiagnosisService()
    application = create_app()
    application.dependency_overrides[
        get_agent_diagnosis_service
    ] = lambda: fake_service

    request_payload = {
        "robot_id": "robot-001",
        "symptom": "网络恢复后仍未继续任务",
        "log_excerpt": "ERR-NET-4001 heartbeat timeout",
        "task_goal": "核对原因和安全恢复条件",
    }

    try:
        async with create_test_client(
            application
        ) as client:
            response = await client.post(
                "/api/v1/agent/diagnose",
                json=request_payload,
            )
    finally:
        application.dependency_overrides.clear()

    assert response.status_code == 200
    assert len(fake_service.calls) == 1

    received_request, received_request_id = (
        fake_service.calls[0]
    )

    assert isinstance(
        received_request,
        AgentDiagnosisRequest,
    )
    assert received_request.robot_id == "robot-001"
    assert received_request.task_goal == (
        "核对原因和安全恢复条件"
    )

    # Middleware生成的同一个ID必须进入：
    # Service参数、响应头和响应JSON。
    assert received_request_id
    assert response.headers["x-request-id"] == (
        received_request_id
    )

    response_data = response.json()
    assert response_data["request_id"] == (
        received_request_id
    )
    assert response_data["diagnosis"]["status"] == (
        "completed"
    )
    assert response_data["execution"]["state"] == (
        "completed"
    )
    assert response_data["diagnosis"]["evidence"][0][
        "chunk_id"
    ] == TEST_CHUNK_ID


@pytest.mark.asyncio
async def test_agent_diagnose_returns_abstention_with_200(
) -> None:
    """Runner安全中止属于合法业务结果，HTTP仍应返回200。"""

    fake_service = FakeAgentDiagnosisService(
        abstained=True,
    )
    application = create_app()
    application.dependency_overrides[
        get_agent_diagnosis_service
    ] = lambda: fake_service

    try:
        async with create_test_client(
            application
        ) as client:
            response = await client.post(
                "/api/v1/agent/diagnose",
                json={
                    "robot_id": "robot-001",
                    "symptom": "机器人出现未知告警",
                    "log_excerpt": "UNKNOWN-WARNING",
                },
            )
    finally:
        application.dependency_overrides.clear()

    assert response.status_code == 200

    response_data = response.json()
    assert response_data["diagnosis"]["abstained"] is True
    assert response_data["diagnosis"]["evidence"] == []
    assert response_data["execution"]["state"] == "aborted"
    assert response_data["execution"][
        "termination_reason"
    ] == "planner_timeout"
    assert response_data["execution"][
        "missing_information"
    ]


@pytest.mark.parametrize(
    ("invalid_payload", "expected_field"),
    [
        (
            {
                "symptom": "机器人停止",
                "log_excerpt": "ERR-NET-4001",
            },
            "robot_id",
        ),
        (
            {
                "robot_id": "robot-001",
                "symptom": "   ",
                "log_excerpt": "ERR-NET-4001",
            },
            "symptom",
        ),
        (
            {
                "robot_id": "robot-001",
                "symptom": "机器人停止",
                "log_excerpt": "ERR-NET-4001",
                "tools": ["run_shell"],
            },
            "tools",
        ),
        (
            {
                "robot_id": "robot-001",
                "symptom": "机器人停止",
                "log_excerpt": "ERR-NET-4001",
                "task_goal": "   ",
            },
            "task_goal",
        ),
    ],
    ids=[
        "missing-robot-id",
        "blank-symptom",
        "extra-tools-field",
        "blank-task-goal",
    ],
)
@pytest.mark.asyncio
async def test_invalid_agent_request_returns_422_before_service(
    invalid_payload: dict[str, object],
    expected_field: str,
) -> None:
    """非法请求必须由FastAPI拒绝，不能进入Agent Service。"""

    fake_service = FakeAgentDiagnosisService()
    application = create_app()
    application.dependency_overrides[
        get_agent_diagnosis_service
    ] = lambda: fake_service

    try:
        async with create_test_client(
            application
        ) as client:
            response = await client.post(
                "/api/v1/agent/diagnose",
                json=invalid_payload,
            )
    finally:
        application.dependency_overrides.clear()

    assert response.status_code == 422
    assert response.headers["x-request-id"]
    assert fake_service.calls == []

    response_data = response.json()
    assert any(
        expected_field in error["loc"]
        for error in response_data["detail"]
    )


@pytest.mark.asyncio
async def test_agent_diagnose_endpoint_rejects_get_method(
) -> None:
    """Agent诊断只允许POST，GET不能触发Service。"""

    fake_service = FakeAgentDiagnosisService()
    application = create_app()
    application.dependency_overrides[
        get_agent_diagnosis_service
    ] = lambda: fake_service

    try:
        async with create_test_client(
            application
        ) as client:
            response = await client.get(
                "/api/v1/agent/diagnose"
            )
    finally:
        application.dependency_overrides.clear()

    assert response.status_code == 405
    assert fake_service.calls == []
