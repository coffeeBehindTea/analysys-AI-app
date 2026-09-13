"""结构化诊断HTTP接口的离线契约测试。

本文件通过HTTPX的ASGITransport直接调用内存中的
FastAPI应用，不启动Uvicorn，也不访问真实网络、
Embedding、Chroma或LLM。

测试目标：

1. POST /api/v1/diagnostics能够调用DiagnosisService；
2. Middleware生成的request_id同时进入报告和响应头；
3. 完整报告中的真实证据与引用可以正确序列化；
4. 证据不足的拒答报告仍使用200和相同响应Schema；
5. 非法JSON请求在进入Service前由FastAPI/Pydantic拒绝。
"""

import pytest

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.dependencies import (
    get_diagnosis_service,
)
from app.schemas.diagnostics import (
    DiagnosisCause,
    DiagnosisCheck,
    DiagnosisEvidence,
    DiagnosisReport,
    DiagnosisRequest,
    DiagnosisSymptom,
)
from main import create_app


# 固定哈希只用于构造满足Schema的脱敏测试证据。
TEST_DOCUMENT_ID = "a" * 64
TEST_CHUNK_ID = (
    TEST_DOCUMENT_ID
    + ":000000"
)

# 这两个版本号用于验证Router没有丢失
# Service已经写入的可审计版本信息。
TEST_PROMPT_VERSION = "diagnosis-evidence-v1"
TEST_GATE_VERSION = "hybrid-evidence-gate-v1"


def build_completed_report(
    *,
    request: DiagnosisRequest,
    request_id: str,
) -> DiagnosisReport:
    """根据Service收到的参数构造可追溯完整报告。"""

    evidence = DiagnosisEvidence(
        chunk_id=TEST_CHUNK_ID,
        document_id=TEST_DOCUMENT_ID,
        source_file="robot-fault-demo.txt",
        page_or_section=(
            "section: ERR-DEMO-1001"
        ),
        rank=1,
        rrf_score=0.032,
        vector_similarity=0.72,
        excerpt=(
            "ERR-DEMO-1001定位质量下降后，"
            "机器人进入受控停止状态。"
        ),
    )

    return DiagnosisReport(
        request_id=request_id,
        prompt_version=TEST_PROMPT_VERSION,
        gate_version=TEST_GATE_VERSION,
        status="completed",
        symptoms=[
            DiagnosisSymptom(
                description=request.symptom,
                source="user_report",
            ),
            DiagnosisSymptom(
                description=(
                    request.log_excerpt
                ),
                source="log_excerpt",
            ),
        ],
        evidence=[evidence],
        possible_causes=[
            DiagnosisCause(
                description=(
                    "可能存在定位质量下降"
                ),
                evidence_chunk_ids=[
                    TEST_CHUNK_ID
                ],
            )
        ],
        next_checks=[
            DiagnosisCheck(
                description=(
                    "检查模拟定位标记是否被遮挡"
                ),
                evidence_chunk_ids=[
                    TEST_CHUNK_ID
                ],
                risk_level="low",
                requires_qualified_person=False,
            )
        ],
        risk_level="low",
        missing_information=[],
        abstained=False,
    )


def build_abstained_report(
    *,
    request: DiagnosisRequest,
    request_id: str,
) -> DiagnosisReport:
    """构造门控拒绝时使用的稳定拒答报告。"""

    return DiagnosisReport(
        request_id=request_id,
        prompt_version="not-invoked",
        gate_version=TEST_GATE_VERSION,
        status="abstained",
        symptoms=[
            DiagnosisSymptom(
                description=request.symptom,
                source="user_report",
            ),
            DiagnosisSymptom(
                description=(
                    request.log_excerpt
                ),
                source="log_excerpt",
            ),
        ],
        evidence=[],
        possible_causes=[],
        next_checks=[],
        risk_level="unknown",
        missing_information=[
            "知识库未检索到可用于诊断的候选证据"
        ],
        abstained=True,
    )


class FakeDiagnosisService:
    """记录Router调用并返回完整报告或拒答报告。"""

    def __init__(
        self,
        *,
        abstained: bool = False,
    ) -> None:
        """保存预设分支并初始化调用记录。"""

        self._abstained = abstained

        # 每项保存Service实际收到的请求模型和请求ID。
        self.calls: list[
            tuple[DiagnosisRequest, str]
        ] = []

    async def diagnose(
        self,
        request: DiagnosisRequest,
        request_id: str,
    ) -> DiagnosisReport:
        """模拟异步诊断，不访问任何外部服务。"""

        self.calls.append(
            (request, request_id)
        )

        if self._abstained:
            return build_abstained_report(
                request=request,
                request_id=request_id,
            )

        return build_completed_report(
            request=request,
            request_id=request_id,
        )


def create_test_client(
    application: FastAPI,
) -> AsyncClient:
    """创建直接调用ASGI应用的异步HTTP客户端。"""

    # ASGITransport把HTTPX请求直接交给FastAPI，
    # 不需要端口、DNS或真实TCP连接。
    transport = ASGITransport(
        app=application
    )

    # base_url只用于拼接相对路径，
    # 不代表测试会访问该网络地址。
    return AsyncClient(
        transport=transport,
        base_url="http://testserver",
    )


@pytest.mark.asyncio
async def test_diagnostics_returns_traceable_report(
) -> None:
    """合法请求应返回证据、引用和同一个request_id。"""

    fake_service = FakeDiagnosisService()
    application = create_app()

    def override_diagnosis_service(
    ) -> FakeDiagnosisService:
        """替换真实依赖，阻止测试访问外部服务。"""

        return fake_service

    application.dependency_overrides[
        get_diagnosis_service
    ] = override_diagnosis_service

    request_payload = {
        "robot_id": "robot-demo-001",
        "symptom": "机器人定位质量下降并停止",
        "log_excerpt": (
            "ERR-DEMO-1001 localization confidence low"
        ),
        "retrieval_scope": {
            "document_ids": [
                TEST_DOCUMENT_ID
            ],
        },
    }

    try:
        async with create_test_client(
            application
        ) as client:
            response = await client.post(
                "/api/v1/diagnostics",
                json=request_payload,
            )
    finally:
        # dependency_overrides属于应用级可变字典，
        # 必须清理以免影响其他测试。
        application.dependency_overrides.clear()

    assert response.status_code == 200
    assert len(fake_service.calls) == 1

    received_request, received_request_id = (
        fake_service.calls[0]
    )

    # FastAPI应先把JSON解析成DiagnosisRequest，
    # 再调用Service。
    assert isinstance(
        received_request,
        DiagnosisRequest,
    )
    assert received_request.robot_id == (
        request_payload["robot_id"]
    )
    assert (
        received_request.retrieval_scope
        is not None
    )
    assert (
        received_request
        .retrieval_scope
        .document_ids
        == [TEST_DOCUMENT_ID]
    )

    # Router传给Service的ID来自Middleware，
    # Middleware又把同一个ID写入响应头。
    assert received_request_id
    assert response.headers[
        "x-request-id"
    ] == received_request_id

    response_data = response.json()

    assert response_data["request_id"] == (
        received_request_id
    )
    assert response_data["status"] == (
        "completed"
    )
    assert response_data["abstained"] is False
    assert response_data["prompt_version"] == (
        TEST_PROMPT_VERSION
    )
    assert response_data["gate_version"] == (
        TEST_GATE_VERSION
    )

    # 原因和检查项必须引用响应evidence中
    # 同一个真实Chunk ID。
    assert response_data["evidence"][0][
        "chunk_id"
    ] == TEST_CHUNK_ID
    assert response_data[
        "possible_causes"
    ][0]["evidence_chunk_ids"] == [
        TEST_CHUNK_ID
    ]
    assert response_data[
        "next_checks"
    ][0]["evidence_chunk_ids"] == [
        TEST_CHUNK_ID
    ]


@pytest.mark.asyncio
async def test_diagnostics_returns_abstained_report_with_200(
) -> None:
    """安全拒答是合法业务结果，仍应返回200。"""

    fake_service = FakeDiagnosisService(
        abstained=True
    )
    application = create_app()
    application.dependency_overrides[
        get_diagnosis_service
    ] = lambda: fake_service

    try:
        async with create_test_client(
            application
        ) as client:
            response = await client.post(
                "/api/v1/diagnostics",
                json={
                    "robot_id": "robot-demo-002",
                    "symptom": "机器人出现未知告警",
                    "log_excerpt": "UNKNOWN-WARNING",
                },
            )
    finally:
        application.dependency_overrides.clear()

    assert response.status_code == 200

    response_data = response.json()
    assert response_data["status"] == (
        "abstained"
    )
    assert response_data["abstained"] is True
    assert response_data["evidence"] == []
    assert response_data[
        "possible_causes"
    ] == []
    assert response_data["next_checks"] == []
    assert response_data[
        "missing_information"
    ]


@pytest.mark.parametrize(
    (
        "invalid_payload",
        "expected_location_part",
    ),
    [
        (
            {
                "symptom": "机器人停止",
                "log_excerpt": "ERR-DEMO-1001",
            },
            "robot_id",
        ),
        (
            {
                "robot_id": "robot-demo-003",
                "symptom": "   ",
                "log_excerpt": "ERR-DEMO-1001",
            },
            "symptom",
        ),
        (
            {
                "robot_id": "robot-demo-003",
                "symptom": "机器人停止",
                "log_excerpt": "ERR-DEMO-1001",
                "untrusted_command": "disable safety",
            },
            "untrusted_command",
        ),
        (
            {
                "robot_id": "robot-demo-003",
                "symptom": "机器人停止",
                "log_excerpt": "ERR-DEMO-1001",
                "retrieval_scope": {},
            },
            "retrieval_scope",
        ),
    ],
    ids=[
        "missing-robot-id",
        "blank-symptom",
        "extra-field",
        "empty-retrieval-scope",
    ],
)
@pytest.mark.asyncio
async def test_invalid_diagnostics_request_returns_422_before_service(
    invalid_payload: dict[str, object],
    expected_location_part: str,
) -> None:
    """非法请求必须在调用诊断Service前返回422。"""

    fake_service = FakeDiagnosisService()
    application = create_app()
    application.dependency_overrides[
        get_diagnosis_service
    ] = lambda: fake_service

    try:
        async with create_test_client(
            application
        ) as client:
            response = await client.post(
                "/api/v1/diagnostics",
                json=invalid_payload,
            )
    finally:
        application.dependency_overrides.clear()

    assert response.status_code == 422
    assert response.headers["x-request-id"]
    assert fake_service.calls == []

    response_data = response.json()
    assert isinstance(
        response_data["detail"],
        list,
    )

    # loc可能是["body", "symptom"]，
    # 也可能包含嵌套模型路径；
    # 因此检查目标字段是否出现在任一loc中。
    assert any(
        expected_location_part
        in error["loc"]
        for error in response_data["detail"]
    )
