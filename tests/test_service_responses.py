"""验证 TriageService 对 LLM 成功响应和无效响应的处理。"""

# AsyncMock 模拟 SDK 异步 create()；
# MagicMock 模拟 SDK 返回的嵌套响应对象。
from unittest.mock import AsyncMock, MagicMock

# pytest 提供异步标记、参数化和异常断言。
import pytest

from app.errors import InvalidLLMResponseError
from app.schemas.triage import TriageRequest
from app.services.triage import TriageService


def create_mock_client_with_content(
    content: str | None,
) -> tuple[MagicMock, AsyncMock]:
    """创建一个返回指定 message.content 的 Mock LLM 客户端。"""

    # 模拟 completion.choices[0]。
    mock_choice = MagicMock()

    # 模拟 completion.choices[0].message.content。
    # content 可以是正常字符串、空字符串或 None。
    mock_choice.message.content = content

    # 模拟 SDK 返回的 ChatCompletion 对象。
    mock_completion = MagicMock()

    # Service 会读取 completion.choices，
    # 因此把唯一候选结果放入列表。
    mock_completion.choices = [
        mock_choice,
    ]

    # create() 是被 await 的异步方法。
    # return_value 表示 await 完成后返回 mock_completion。
    mock_create = AsyncMock(
        return_value=mock_completion,
    )

    # 模拟 AsyncOpenAI 客户端的属性结构。
    mock_client = MagicMock()
    mock_client.chat.completions.create = mock_create

    # 同时返回客户端和 create Mock：
    # 客户端用于注入 Service；
    # create Mock 用于检查调用情况。
    return mock_client, mock_create


def create_test_request() -> TriageRequest:
    """创建多个 Service 测试共用的合法请求模型。"""

    return TriageRequest(
        robot_id="test-robot-001",
        symptom="机器人无法移动",
        log_excerpt="WARN wheel feedback unavailable",
    )


@pytest.mark.asyncio
async def test_service_parses_valid_llm_response() -> None:
    """合法 LLM JSON 应该被转换成 TriageResponse。"""

    llm_content = (
        "{"
        '"summary":"根据现有信息，只能确认机器人无法移动。",'
        '"recommended_actions":['
        '"检查急停状态",'
        '"补充信息：确认电机是否通电"'
        "]"
        "}"
    )

    mock_client, mock_create = (
        create_mock_client_with_content(llm_content)
    )

    service = TriageService(
        client=mock_client,
        model="test-model",
    )

    response = await service.analyze(
        create_test_request(),
        request_id="test-request-id",
    )

    # model_dump() 把 Pydantic 模型转换成普通 Python 字典。
    assert response.model_dump() == {
        "request_id": "test-request-id",
        "summary": "根据现有信息，只能确认机器人无法移动。",
        "recommended_actions": [
            "检查急停状态",
            "补充信息：确认电机是否通电",
        ],
    }

    # 确认 Service 只请求了一次 LLM。
    mock_create.assert_awaited_once()


@pytest.mark.parametrize(
    "invalid_content",
    [
        pytest.param(
            None,
            id="none-content",
        ),
        pytest.param(
            "",
            id="empty-content",
        ),
        pytest.param(
            "这不是 JSON",
            id="invalid-json",
        ),
        pytest.param(
            # JSON 语法合法，但缺少 recommended_actions。
            '{"summary":"缺少建议字段"}',
            id="wrong-json-structure",
        ),
    ],
)
@pytest.mark.asyncio
async def test_service_rejects_invalid_llm_content(
    invalid_content: str | None,
) -> None:
    """不同类型的无效模型内容都应转换成统一应用异常。"""

    mock_client, mock_create = (
        create_mock_client_with_content(
            invalid_content
        )
    )

    service = TriageService(
        client=mock_client,
        model="test-model",
    )

    with pytest.raises(
        InvalidLLMResponseError
    ):
        await service.analyze(
            create_test_request(),
            request_id="test-request-id",
        )

    # 即使响应内容无效，也应该确认 SDK 请求确实只执行了一次。
    mock_create.assert_awaited_once()


@pytest.mark.asyncio
async def test_service_rejects_empty_choices() -> None:
    """LLM 响应没有候选结果时应该被识别为无效响应。"""

    mock_completion = MagicMock()

    # choices 是 SDK 返回的候选回答列表。
    # 空列表表示没有任何可用回答。
    mock_completion.choices = []

    mock_create = AsyncMock(
        return_value=mock_completion,
    )

    mock_client = MagicMock()
    mock_client.chat.completions.create = mock_create

    service = TriageService(
        client=mock_client,
        model="test-model",
    )

    with pytest.raises(
        InvalidLLMResponseError
    ) as exc_info:
        await service.analyze(
            create_test_request(),
            request_id="test-request-id",
        )

    assert "没有候选结果" in str(exc_info.value)

    mock_create.assert_awaited_once()