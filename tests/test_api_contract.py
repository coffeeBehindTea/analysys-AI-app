"""验证健康检查和请求数据契约。"""

# pytest 提供测试收集、异步标记和参数化能力。
import pytest

# ASGITransport 直接调用内存中的 FastAPI 应用；
# AsyncClient 提供异步 HTTP 请求方法。
from httpx import ASGITransport, AsyncClient

# 测试中需要覆盖这个真实依赖，避免创建 LLM 客户端。
from app.dependencies import get_triage_service

# Fake Service 使用与真实 Router 一致的方法参数和返回类型。
from app.schemas.triage import TriageRequest, TriageResponse

# 导入已经装配 Router、Middleware 和异常处理器的应用。
from main import app


class NeverCalledTriageService:
    """用于确认无效请求不会进入业务层的 Fake Service。"""

    def __init__(self) -> None:
        """记录 analyze() 是否曾经被调用。"""

        self.analyze_called = False

    async def analyze(
        self,
        request: TriageRequest,
        request_id: str,
    ) -> TriageResponse:
        """如果该方法被调用，就主动让测试失败。"""

        # 如果执行到这里，说明无效请求错误地进入了业务层。
        self.analyze_called = True

        raise AssertionError(
            "请求校验失败时不应该调用 TriageService.analyze()"
        )


def create_test_client() -> AsyncClient:
    """创建不访问真实网络的异步测试客户端。"""

    # app=app 表示所有请求直接交给当前 FastAPI 应用。
    transport = ASGITransport(app=app)

    # testserver 只是组合相对 URL 所需的占位地址，
    # 不会进行 DNS 查询或真实网络连接。
    return AsyncClient(
        transport=transport,
        base_url="http://testserver",
    )


@pytest.mark.asyncio
async def test_health_check() -> None:
    """健康检查应该返回稳定的 200 JSON 响应。"""

    async with create_test_client() as client:
        # get() 发送 GET 请求；因为是异步客户端，所以需要 await。
        response = await client.get("/health")

    # status_code 是 HTTPX Response 保存的 HTTP 状态码属性。
    assert response.status_code == 200

    # json() 把 JSON 响应体解析为 Python 对象。
    assert response.json() == {
        "status": "ok",
    }

    # RequestIdMiddleware 应该处理健康检查请求。
    # headers 是不区分大小写的响应头容器。
    assert response.headers["x-request-id"]


@pytest.mark.parametrize(
    (
        "invalid_payload",
        "expected_error_field",
    ),
    [
        pytest.param(
            # 缺少必填 robot_id。
            {
                "symptom": "机器人无法移动",
                "log_excerpt": "WARN wheel feedback unavailable",
            },
            "robot_id",
            id="missing-robot-id",
        ),
        pytest.param(
            # str_strip_whitespace 会先把纯空格清理成空字符串，
            # 随后 min_length=1 校验失败。
            {
                "robot_id": "test-robot-001",
                "symptom": "   ",
                "log_excerpt": "WARN wheel feedback unavailable",
            },
            "symptom",
            id="blank-symptom",
        ),
        pytest.param(
            # TriageRequest 只允许最多 10_000 个日志字符。
            {
                "robot_id": "test-robot-001",
                "symptom": "机器人无法移动",
                "log_excerpt": "L" * 10_001,
            },
            "log_excerpt",
            id="log-too-long",
        ),
        pytest.param(
            # extra="forbid" 应该拒绝 Schema 中没有声明的字段。
            {
                "robot_id": "test-robot-001",
                "symptom": "机器人无法移动",
                "log_excerpt": "WARN wheel feedback unavailable",
                "api_key": "must-not-be-accepted",
            },
            "api_key",
            id="extra-field",
        ),
    ],
)
@pytest.mark.asyncio
async def test_invalid_triage_request_returns_422(
    invalid_payload: dict[str, object],
    expected_error_field: str,
) -> None:
    """无效请求应该返回 422，并且不能执行业务 Service。"""

    fake_service = NeverCalledTriageService()

    def override_get_triage_service() -> NeverCalledTriageService:
        """替代真实依赖，避免测试读取 API Key 或创建 LLM 客户端。"""

        return fake_service

    app.dependency_overrides[
        get_triage_service
    ] = override_get_triage_service

    try:
        async with create_test_client() as client:
            response = await client.post(
                "/api/v1/triage",
                json=invalid_payload,
            )

    finally:
        # dependency_overrides 属于全局 app 对象，
        # 每次测试后必须清除，避免影响其他测试。
        app.dependency_overrides.clear()

    # 422 表示 HTTP 请求格式可以理解，
    # 但请求内容不符合接口的数据契约。
    assert response.status_code == 422

    response_data = response.json()

    # FastAPI 把字段校验错误放在 detail 列表中，
    # 因为一次请求可能同时存在多个字段错误。
    assert isinstance(response_data["detail"], list)

    validation_errors = response_data["detail"]

    # 每项错误的 loc 表示错误位置。
    # 例如 ["body", "robot_id"] 表示请求体中的 robot_id。
    #
    # any() 只要找到一项满足条件，就返回 True。
    assert any(
        error["loc"][-1] == expected_error_field
        for error in validation_errors
    )

    # 即使发生 422，Middleware 仍然应该提供追踪 ID。
    assert response.headers["x-request-id"]

    # 最关键的分层断言：
    # Pydantic/FastAPI 应该在 Router 调用 Service 之前拒绝请求。
    assert fake_service.analyze_called is False