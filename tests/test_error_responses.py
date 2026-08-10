"""验证成功响应和统一应用错误响应。"""

# pytest 提供测试发现、参数化和断言结果报告。
import pytest

# httpx 提供异步 HTTP 客户端和内存 ASGI Transport。
from httpx import ASGITransport, AsyncClient

# 导入 Router 当前使用的原始依赖函数。
# dependency_overrides 必须用这个原始函数作为字典键。
from app.dependencies import get_triage_service

# 导入需要模拟的应用异常。
from app.errors import (
    ApplicationError,
    InvalidLLMResponseError,
    LLMConfigurationError,
    LLMTimeoutError,
    LLMUpstreamError,
)

# Fake Service 的方法仍然使用真实 Schema，
# 从而与 Router 期待的接口保持一致。
from app.schemas.triage import TriageRequest, TriageResponse

# 导入已经装配好 Router、Middleware 和异常处理器的 FastAPI 应用。
from main import app


# 测试使用的合法请求体。
VALID_PAYLOAD = {
    "robot_id": "test-robot-001",
    "symptom": "机器人无法移动",
    "log_excerpt": "WARN wheel feedback unavailable",
}


class FakeSuccessTriageService:
    """不访问 LLM，直接返回固定成功结果的假 Service。"""

    async def analyze(
        self,
        request: TriageRequest,
        request_id: str,
    ) -> TriageResponse:
        """返回与真实 Service 相同类型的成功响应。"""

        return TriageResponse(
            # 使用 Router 传入的 request_id，
            # 用于验证响应头和响应体是否保持一致。
            request_id=request_id,
            summary=f"已收到 {request.robot_id} 的测试数据",
            recommended_actions=["检查连接"],
        )


class FakeErrorTriageService:
    """调用 analyze() 时抛出指定应用异常的假 Service。"""

    def __init__(self, error: ApplicationError) -> None:
        # 保存本次测试希望触发的异常对象。
        self._error = error

    async def analyze(
        self,
        request: TriageRequest,
        request_id: str,
    ) -> TriageResponse:
        """不访问 LLM，直接抛出预先指定的异常。"""

        raise self._error


def create_test_client() -> AsyncClient:
    """创建直接调用 FastAPI 应用的异步测试客户端。"""

    # ASGITransport 将 HTTPX 请求直接交给 FastAPI 应用，
    # 不需要启动 Uvicorn，也不会访问真实网络。
    transport = ASGITransport(app=app)

    # base_url 是 HTTPX 组合相对路径时需要的基础地址。
    # 这里的 testserver 只是测试占位地址，不会进行 DNS 访问。
    return AsyncClient(
        transport=transport,
        base_url="http://testserver",
    )


@pytest.mark.asyncio
async def test_triage_success_contains_same_request_id() -> None:
    """成功响应头和响应体应该使用同一个 request_id。"""

    fake_service = FakeSuccessTriageService()

    def override_get_triage_service() -> FakeSuccessTriageService:
        """让 FastAPI 注入 Fake Service，而不是真实 Service。"""

        return fake_service

    # dependency_overrides 是 FastAPI 应用提供的依赖覆盖字典。
    #
    # 原本：
    # get_triage_service → 真实 TriageService
    #
    # 测试中：
    # get_triage_service → FakeSuccessTriageService
    app.dependency_overrides[
        get_triage_service
    ] = override_get_triage_service

    try:
        # AsyncClient 实现异步上下文管理器。
        # async with 结束时会调用异步关闭方法，释放客户端资源。
        async with create_test_client() as client:
            # post() 构造 POST 请求。
            # json= 会把 Python dict 自动序列化为 JSON 请求体，
            # 并设置 Content-Type: application/json。
            response = await client.post(
                "/api/v1/triage",
                json=VALID_PAYLOAD,
            )

        # status_code 是响应的 HTTP 状态码。
        assert response.status_code == 200

        # json() 把 JSON 响应体解析成 Python dict。
        response_data = response.json()

        # headers 是不区分大小写的响应头容器。
        header_request_id = response.headers["x-request-id"]
        body_request_id = response_data["request_id"]

        assert header_request_id == body_request_id
        assert response_data["summary"] == (
            "已收到 test-robot-001 的测试数据"
        )
    finally:
        # dependency_overrides 属于全局 app 对象。
        # 测试完成后必须清空，避免影响其他测试。
        app.dependency_overrides.clear()


# parametrize 会使用下面四组参数重复运行同一个测试函数。
@pytest.mark.parametrize(
    (
        "application_error",
        "expected_status_code",
        "expected_error_code",
        "expected_public_message",
    ),
    [
        (
            LLMConfigurationError("LLM_API_KEY 未配置"),
            503,
            "llm_configuration_error",
            "LLM 服务未正确配置",
        ),
        (
            LLMTimeoutError("等待上游超过 30 秒"),
            504,
            "llm_timeout",
            "LLM 服务响应超时",
        ),
        (
            LLMUpstreamError("无法连接 LLM 服务"),
            502,
            "llm_upstream_error",
            "LLM 上游服务暂时不可用",
        ),
        (
            InvalidLLMResponseError("模型返回的内容不是 JSON"),
            502,
            "invalid_llm_response",
            "LLM 返回了无法处理的响应",
        ),
    ],
)
@pytest.mark.asyncio
async def test_application_error_response(
    application_error: ApplicationError,
    expected_status_code: int,
    expected_error_code: str,
    expected_public_message: str,
) -> None:
    """不同应用异常应该转换成对应的安全 HTTP 响应。"""

    fake_service = FakeErrorTriageService(application_error)

    def override_get_triage_service() -> FakeErrorTriageService:
        """向 Router 注入本次参数对应的错误 Service。"""

        return fake_service

    app.dependency_overrides[
        get_triage_service
    ] = override_get_triage_service

    try:
        async with create_test_client() as client:
            response = await client.post(
                "/api/v1/triage",
                json=VALID_PAYLOAD,
            )

        response_data = response.json()

        # 验证应用异常被转换成了正确 HTTP 状态码。
        assert response.status_code == expected_status_code

        # 验证公开错误码和信息稳定、安全。
        assert (
            response_data["error"]["code"]
            == expected_error_code
        )
        assert (
            response_data["error"]["message"]
            == expected_public_message
        )

        # 验证错误响应头和响应体使用同一 request_id。
        assert (
            response.headers["x-request-id"]
            == response_data["request_id"]
        )

        # 验证内部异常详情没有直接泄漏给客户端。
        assert str(application_error) not in response.text
    finally:
        app.dependency_overrides.clear()