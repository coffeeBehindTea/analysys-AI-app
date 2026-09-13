"""裸LLM对照组Provider的离线测试。"""

# AsyncMock模拟会被await的异步方法；
# MagicMock模拟普通对象、属性链和同步函数。
from unittest.mock import (
    AsyncMock,
    MagicMock,
)

# httpx.Request用于构造OpenAI SDK异常所需的请求上下文。
import httpx

import pytest

from openai import (
    APIConnectionError,
    APITimeoutError,
)

from app.errors import (
    InvalidLLMResponseError,
    LLMTimeoutError,
    LLMUpstreamError,
)

# 导入模块对象是为了替换该模块内部使用的perf_counter。
import app.services.bare_llm as bare_llm_module

from app.services.bare_llm import (
    BARE_LLM_SYSTEM_PROMPT,
    OpenAIBareLLMProvider,
    build_bare_llm_messages,
)


def create_mock_client_with_content(
    content: str | None,
) -> tuple[MagicMock, AsyncMock]:
    """创建返回指定正文的假OpenAI异步客户端。"""

    # 模拟completion.choices[0]。
    mock_choice = MagicMock()

    # 模拟choices[0].message.content。
    mock_choice.message.content = content

    # 模拟OpenAI ChatCompletion对象。
    mock_completion = MagicMock()
    mock_completion.choices = [
        mock_choice,
    ]

    # create()在生产代码中会被await，
    # 所以这里必须使用AsyncMock。
    mock_create = AsyncMock(
        return_value=mock_completion,
    )

    # 构造client.chat.completions.create属性链。
    mock_client = MagicMock()
    mock_client.chat.completions.create = (
        mock_create
    )

    return mock_client, mock_create


def test_messages_contain_only_rules_and_question(
) -> None:
    """裸LLM消息只能包含System规则和原始问题。"""

    messages = build_bare_llm_messages(
        "  robot-001当前电量是多少？  "
    )

    assert len(messages) == 2

    assert messages[0] == {
        "role": "system",
        "content": BARE_LLM_SYSTEM_PROMPT,
    }

    # question会执行strip()，
    # 因此首尾空格不应进入模型输入。
    assert messages[1] == {
        "role": "user",
        "content": (
            "robot-001当前电量是多少？"
        ),
    }

    # System Prompt必须明确说明没有实时工具，
    # 否则裸LLM可能假装知道当前机器人状态。
    assert "实时工具" in (
        messages[0]["content"]
    )

    # User Message只能有role和content，
    # 不能附带evidence、reference_answer等实验答案。
    assert set(messages[1]) == {
        "role",
        "content",
    }


def test_messages_reject_blank_question(
) -> None:
    """只包含空白字符的问题不能调用裸LLM。"""

    with pytest.raises(
        ValueError,
        match="问题不能为空",
    ):
        build_bare_llm_messages(
            "   "
        )


@pytest.mark.asyncio
async def test_provider_returns_answer_and_latency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider应返回清理后的正文和可重复验证的耗时。"""

    mock_client, mock_create = (
        create_mock_client_with_content(
            "  无法访问robot-001的实时电量。  "
        )
    )

    # 用可控的假时钟替换bare_llm模块中的perf_counter。
    #
    # 第一次调用返回100.0秒；
    # 第二次调用返回100.25秒；
    # 因此预期耗时为250毫秒。
    mock_clock = MagicMock(
        side_effect=[
            100.0,
            100.25,
        ]
    )

    # 必须替换“被测模块实际查找的名字”。
    #
    # generate_answer()运行时查找的是：
    # app.services.bare_llm.perf_counter
    monkeypatch.setattr(
        bare_llm_module,
        "perf_counter",
        mock_clock,
    )

    provider = OpenAIBareLLMProvider(
        client=mock_client,
        model="test-model",
    )

    result = await provider.generate_answer(
        question=(
            "robot-001当前电量是多少？"
        )
    )

    # Provider应清理模型正文首尾空白。
    assert result.answer == (
        "无法访问robot-001的实时电量。"
    )

    # pytest.approx()用于比较浮点计算结果，
    # 避免二进制浮点微小误差导致测试失败。
    assert result.generation_ms == pytest.approx(
        250.0
    )

    # create()必须恰好被await一次。
    mock_create.assert_awaited_once()

    awaited_kwargs = (
        mock_create.await_args.kwargs
    )

    assert awaited_kwargs["model"] == (
        "test-model"
    )

    messages = awaited_kwargs["messages"]

    assert messages[0]["role"] == "system"
    assert messages[1] == {
        "role": "user",
        "content": (
            "robot-001当前电量是多少？"
        ),
    }

    # 开始和结束时各读取一次时钟。
    assert mock_clock.call_count == 2


def test_provider_rejects_blank_model(
) -> None:
    """模型名称为空时应在调用SDK前失败。"""

    mock_client = MagicMock()

    with pytest.raises(
        ValueError,
        match="模型名称不能为空",
    ):
        OpenAIBareLLMProvider(
            client=mock_client,
            model="   ",
        )


@pytest.mark.asyncio
async def test_provider_converts_sdk_timeout(
) -> None:
    """SDK超时应转换成项目的LLMTimeoutError。"""

    sdk_request = httpx.Request(
        method="POST",
        url=(
            "https://llm.test"
            "/chat/completions"
        ),
    )

    sdk_timeout = APITimeoutError(
        request=sdk_request,
    )

    mock_create = AsyncMock(
        side_effect=sdk_timeout,
    )

    mock_client = MagicMock()
    mock_client.chat.completions.create = (
        mock_create
    )

    provider = OpenAIBareLLMProvider(
        client=mock_client,
        model="test-model",
    )

    with pytest.raises(
        LLMTimeoutError,
        match="响应超时",
    ):
        await provider.generate_answer(
            question="测试问题"
        )

    mock_create.assert_awaited_once()


@pytest.mark.asyncio
async def test_provider_converts_connection_error(
) -> None:
    """SDK连接失败应转换成项目的LLMUpstreamError。"""

    sdk_request = httpx.Request(
        method="POST",
        url=(
            "https://llm.test"
            "/chat/completions"
        ),
    )

    sdk_error = APIConnectionError(
        request=sdk_request,
    )

    mock_create = AsyncMock(
        side_effect=sdk_error,
    )

    mock_client = MagicMock()
    mock_client.chat.completions.create = (
        mock_create
    )

    provider = OpenAIBareLLMProvider(
        client=mock_client,
        model="test-model",
    )

    with pytest.raises(
        LLMUpstreamError,
        match="无法连接",
    ):
        await provider.generate_answer(
            question="测试问题"
        )

    mock_create.assert_awaited_once()


@pytest.mark.asyncio
async def test_provider_rejects_empty_choices(
) -> None:
    """SDK没有候选结果时应产生无效响应异常。"""

    mock_completion = MagicMock()
    mock_completion.choices = []

    mock_create = AsyncMock(
        return_value=mock_completion,
    )

    mock_client = MagicMock()
    mock_client.chat.completions.create = (
        mock_create
    )

    provider = OpenAIBareLLMProvider(
        client=mock_client,
        model="test-model",
    )

    with pytest.raises(
        InvalidLLMResponseError,
        match="没有候选结果",
    ):
        await provider.generate_answer(
            question="测试问题"
        )

    mock_create.assert_awaited_once()


@pytest.mark.asyncio
async def test_provider_rejects_empty_content(
) -> None:
    """候选正文为空时应产生无效响应异常。"""

    mock_client, mock_create = (
        create_mock_client_with_content(
            "   "
        )
    )

    provider = OpenAIBareLLMProvider(
        client=mock_client,
        model="test-model",
    )

    with pytest.raises(
        InvalidLLMResponseError,
        match="空内容",
    ):
        await provider.generate_answer(
            question="测试问题"
        )

    mock_create.assert_awaited_once()