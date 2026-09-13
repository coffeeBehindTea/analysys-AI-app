"""诊断会话查询API的离线集成测试。

测试通过FastAPI dependency_overrides注入Fake Store，
不会读取正式data/diagnostic_sessions目录，
也不会执行Agent、LLM、Vision、知识库或网络请求。
"""

from contextlib import (
    asynccontextmanager,
)
from datetime import (
    datetime,
    timezone,
)
from typing import (
    AsyncIterator,
)

import httpx
import pytest

from app.dependencies import (
    get_diagnostic_session_store,
)
from app.errors import (
    DiagnosticSessionCorruptedError,
    DiagnosticSessionStoreError,
)
from app.schemas.diagnostic_session import (
    DiagnosticSessionDiagnosisSnapshot,
    DiagnosticSessionEvidenceCoverage,
    DiagnosticSessionListResponse,
    DiagnosticSessionMetrics,
    DiagnosticSessionRecord,
    DiagnosticSessionRequestSummary,
)
from app.services.diagnostic_session_builder import (
    build_diagnostic_session_summary,
)
from main import create_app


# 固定的小写UUID4用于URL、响应和Fake Store断言。
TEST_SESSION_ID = (
    "123e4567-e89b-42d3-a456-426614174000"
)
TEST_MISSING_SESSION_ID = (
    "223e4567-e89b-42d3-a456-426614174001"
)
TEST_REQUEST_ID = "request-session-api-001"


def make_record() -> DiagnosticSessionRecord:
    """创建一条可以由详情接口公开的脱敏拒答会话。"""

    return DiagnosticSessionRecord(
        session_id=TEST_SESSION_ID,
        request_id=TEST_REQUEST_ID,
        created_at=datetime(
            2026,
            9,
            9,
            10,
            30,
            tzinfo=timezone.utc,
        ),
        request_summary=(
            DiagnosticSessionRequestSummary(
                robot_id="robot-001",
                symptom_summary=(
                    "未知故障代码需要补充资料"
                ),
                task_goal_summary=(
                    "确认故障原因"
                ),
                log_excerpt_sha256="a" * 64,
                log_excerpt_char_count=21,
                image_count=0,
            )
        ),
        status="abstained",
        execution_state="completed",
        termination_reason="planner_finished",
        finish_reason="insufficient_information",
        diagnosis=(
            DiagnosticSessionDiagnosisSnapshot(
                prompt_version=(
                    "agent-tool-calling-v2"
                ),
                gate_version=(
                    "agent-confirmed-evidence-v1"
                ),
                status="abstained",
                evidence=(),
                possible_causes=(),
                next_checks=(),
                risk_level="unknown",
                missing_information=(
                    "知识库未收录该故障代码",
                ),
                abstained=True,
            )
        ),
        tool_events=(),
        vision_observations=(),
        telemetry_observations=(),
        test_case_drafts=(),
        evidence_coverage=(
            DiagnosticSessionEvidenceCoverage()
        ),
        metrics=DiagnosticSessionMetrics(
            total_duration_ms=18.5,
            tool_step_count=0,
            failed_tool_count=0,
            failed_tool_names=(),
            confirmed_source_count=0,
            covered_evidence_count=0,
            citation_count=0,
        ),
    )


class FakeDiagnosticSessionStore:
    """为Router返回预设结果并记录调用参数的异步Fake。

    Fake实现与Router实际使用的Store方法具有相同签名，
    但数据只保存在内存属性中，不访问文件系统。
    """

    def __init__(
        self,
        *,
        record: DiagnosticSessionRecord | None = None,
        get_error: Exception | None = None,
        list_error: Exception | None = None,
    ) -> None:
        self.record = record
        self.get_error = get_error
        self.list_error = list_error
        self.get_calls: list[str] = []
        self.list_limits: list[int] = []

    async def get(
        self,
        session_id: str,
        /,
    ) -> DiagnosticSessionRecord | None:
        """记录查询ID，并返回记录、None或预设异常。"""

        self.get_calls.append(session_id)

        if self.get_error is not None:
            raise self.get_error

        if (
            self.record is not None
            and self.record.session_id
            == session_id
        ):
            return self.record

        return None

    async def list_recent(
        self,
        *,
        limit: int = 20,
    ) -> DiagnosticSessionListResponse:
        """记录limit并动态构造与该limit一致的列表响应。"""

        self.list_limits.append(limit)

        if self.list_error is not None:
            raise self.list_error

        summaries = (
            (
                build_diagnostic_session_summary(
                    self.record
                ),
            )
            if self.record is not None
            else ()
        )

        return DiagnosticSessionListResponse(
            sessions=summaries[:limit],
            total=len(summaries),
            limit=limit,
        )


@asynccontextmanager
async def create_test_client(
    fake_store: FakeDiagnosticSessionStore,
) -> AsyncIterator[httpx.AsyncClient]:
    """创建使用Fake Store的进程内异步HTTP客户端。

    ASGITransport直接调用FastAPI应用，
    不打开端口，也不需要启动Uvicorn。
    """

    application = create_app()

    # FastAPI在解析Depends时改用当前Fake，
    # 其他应用依赖不会因为查询接口而被创建。
    application.dependency_overrides[
        get_diagnostic_session_store
    ] = lambda: fake_store

    transport = httpx.ASGITransport(
        app=application
    )

    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            yield client
    finally:
        application.dependency_overrides.clear()


def test_main_registers_diagnostic_session_routes() -> None:
    """主应用必须同时注册列表和详情两条GET路由。"""

    application = create_app()
    openapi_paths = application.openapi()[
        "paths"
    ]

    # 当前FastAPI版本可能在app.routes中使用
    # 没有path属性的_IncludedRouter包装对象。
    # OpenAPI是应用真正公开的HTTP契约，
    # 因此用它验证最终路径和方法更稳定。
    assert "get" in openapi_paths[
        "/api/v1/diagnostic-sessions"
    ]
    assert "get" in openapi_paths[
        (
            "/api/v1/diagnostic-sessions/"
            "{session_id}"
        )
    ]


@pytest.mark.asyncio
async def test_list_sessions_returns_lightweight_summaries(
) -> None:
    """列表接口应传递limit并返回摘要而非完整轨迹。"""

    fake_store = FakeDiagnosticSessionStore(
        record=make_record()
    )

    async with create_test_client(
        fake_store
    ) as client:
        response = await client.get(
            "/api/v1/diagnostic-sessions",
            params={"limit": 5},
        )

    response_data = response.json()

    assert response.status_code == 200
    assert fake_store.list_limits == [5]
    assert response_data["total"] == 1
    assert response_data["limit"] == 5
    assert response_data["sessions"][0][
        "session_id"
    ] == TEST_SESSION_ID
    assert "tool_events" not in (
        response_data["sessions"][0]
    )


@pytest.mark.asyncio
async def test_list_sessions_uses_default_limit() -> None:
    """客户端省略limit时Router应使用文档化默认值20。"""

    fake_store = FakeDiagnosticSessionStore()

    async with create_test_client(
        fake_store
    ) as client:
        response = await client.get(
            "/api/v1/diagnostic-sessions"
        )

    assert response.status_code == 200
    assert fake_store.list_limits == [20]
    assert response.json() == {
        "sessions": [],
        "total": 0,
        "limit": 20,
    }


@pytest.mark.asyncio
async def test_get_session_returns_full_record() -> None:
    """合法且存在的ID应返回完整脱敏会话详情。"""

    fake_store = FakeDiagnosticSessionStore(
        record=make_record()
    )

    async with create_test_client(
        fake_store
    ) as client:
        response = await client.get(
            (
                "/api/v1/diagnostic-sessions/"
                f"{TEST_SESSION_ID}"
            )
        )

    response_data = response.json()

    assert response.status_code == 200
    assert fake_store.get_calls == [
        TEST_SESSION_ID
    ]
    assert response_data["session_id"] == (
        TEST_SESSION_ID
    )
    assert response_data["request_summary"][
        "robot_id"
    ] == "robot-001"
    assert "tool_events" in response_data
    assert "log_excerpt" not in (
        response_data["request_summary"]
    )


@pytest.mark.asyncio
async def test_get_missing_session_returns_unified_404(
) -> None:
    """合法但不存在的UUID4应转换成统一404错误。"""

    fake_store = FakeDiagnosticSessionStore()

    async with create_test_client(
        fake_store
    ) as client:
        response = await client.get(
            (
                "/api/v1/diagnostic-sessions/"
                f"{TEST_MISSING_SESSION_ID}"
            )
        )

    response_data = response.json()

    assert response.status_code == 404
    assert response.headers.get(
        "x-request-id"
    )
    assert response_data["error"] == {
        "code": "diagnostic_session_not_found",
        "message": "诊断会话不存在",
    }


@pytest.mark.asyncio
async def test_get_invalid_session_id_returns_422_before_store(
) -> None:
    """非法路径参数必须由FastAPI拒绝且不能调用Store。"""

    fake_store = FakeDiagnosticSessionStore(
        record=make_record()
    )

    async with create_test_client(
        fake_store
    ) as client:
        response = await client.get(
            (
                "/api/v1/diagnostic-sessions/"
                "not-a-uuid"
            )
        )

    assert response.status_code == 422
    assert fake_store.get_calls == []
    assert response.json()["detail"]


@pytest.mark.asyncio
async def test_corrupted_session_returns_stable_500(
) -> None:
    """存储完整性异常应转换成安全且稳定的500响应。"""

    fake_store = FakeDiagnosticSessionStore(
        get_error=(
            DiagnosticSessionCorruptedError(
                "测试损坏会话"
            )
        )
    )

    async with create_test_client(
        fake_store
    ) as client:
        response = await client.get(
            (
                "/api/v1/diagnostic-sessions/"
                f"{TEST_SESSION_ID}"
            )
        )

    assert response.status_code == 500
    assert response.json()["error"] == {
        "code": "diagnostic_session_corrupted",
        "message": "诊断会话记录无法读取",
    }


@pytest.mark.asyncio
async def test_list_store_failure_returns_stable_503(
) -> None:
    """列表文件系统失败应转换成统一503响应。"""

    fake_store = FakeDiagnosticSessionStore(
        list_error=(
            DiagnosticSessionStoreError(
                "测试存储不可用"
            )
        )
    )

    async with create_test_client(
        fake_store
    ) as client:
        response = await client.get(
            "/api/v1/diagnostic-sessions"
        )

    assert response.status_code == 503
    assert response.json()["error"] == {
        "code": "diagnostic_session_store_error",
        "message": "诊断会话存储暂时不可用",
    }
