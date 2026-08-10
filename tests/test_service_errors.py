"""验证 LLM 客户端配置和 Service 异常转换。"""

# MagicMock 模拟普通对象；
# AsyncMock 模拟会被 await 的异步方法。
from unittest.mock import AsyncMock, MagicMock

# httpx.Request 只用于构造 SDK 异常所需的请求上下文。
import httpx

# pytest 提供异步测试标记和异常断言。
import pytest

# APITimeoutError 是 OpenAI SDK 在请求超时时抛出的异常。
from openai import APITimeoutError

from app.config import Settings
from app.errors import (
    LLMConfigurationError,
    LLMTimeoutError,
)
from app.schemas.triage import TriageRequest
from app.services.llm_client import create_llm_client
from app.services.triage import TriageService


def test_create_llm_client_without_api_key() -> None:
    """没有 API Key 时，客户端工厂应该拒绝创建客户端。"""

    # _env_file=None 表示这次构造 Settings 时不读取真实 .env。
    # 所有 LLM 配置都显式设为 None，使测试不依赖本机环境。
    settings = Settings(
        _env_file=None,
        llm_api_key=None,
        llm_base_url=None,
        llm_model=None,
    )

    # pytest.raises() 要求代码块必须抛出指定异常。
    # as exc_info 保存捕获到的异常信息，供后续断言。
    with pytest.raises(
        LLMConfigurationError
    ) as exc_info:
        create_llm_client(settings)

    # exc_info.value 是实际捕获到的异常对象。
    assert "LLM_API_KEY" in str(exc_info.value)


@pytest.mark.asyncio
async def test_service_converts_sdk_timeout() -> None:
    """Service 应该把 SDK 超时转换成应用自己的超时异常。"""

    # APITimeoutError 的构造函数需要一个 httpx.Request。
    # 创建 Request 对象不会发送网络请求，只是在内存中描述请求。
    sdk_request = httpx.Request(
        method="POST",
        url="https://llm.test/chat/completions",
    )

    sdk_timeout_error = APITimeoutError(
        request=sdk_request,
    )

    # AsyncMock 模拟一个可以被 await 的 create() 方法。
    #
    # side_effect 设置为异常对象后，
    # 每次 await mock_create(...) 都会抛出该异常。
    mock_create = AsyncMock(
        side_effect=sdk_timeout_error,
    )

    # MagicMock 创建一个通用的模拟对象。
    # 它允许构造与 AsyncOpenAI 相同的属性访问链：
    # client.chat.completions.create
    mock_client = MagicMock()

    # 用 AsyncMock 替换最终会被 await 的 create()。
    mock_client.chat.completions.create = mock_create

    service = TriageService(
        client=mock_client,
        model="test-model",
    )

    request = TriageRequest(
        robot_id="test-robot-001",
        symptom="机器人无法移动",
        log_excerpt="WARN wheel feedback unavailable",
    )

    # Service 不应该把 SDK 的 APITimeoutError 原样泄漏出去，
    # 而应该转换成应用层的 LLMTimeoutError。
    with pytest.raises(
        LLMTimeoutError
    ) as exc_info:
        await service.analyze(
            request,
            request_id="test-request-id",
        )

    assert str(exc_info.value) == "LLM 服务响应超时"

    # 验证 SDK create() 方法只被 await 了一次。
    mock_create.assert_awaited_once()

    # await_args 保存最近一次 await 调用的参数。
    # kwargs 是其中通过关键字传入的参数字典。
    awaited_kwargs = mock_create.await_args.kwargs

    # 确认 Service 把构造时注入的模型名称传给了 SDK。
    assert awaited_kwargs["model"] == "test-model"

    messages = awaited_kwargs["messages"]

    # build_triage_messages() 应该产生 system 和 user 两条消息。
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"