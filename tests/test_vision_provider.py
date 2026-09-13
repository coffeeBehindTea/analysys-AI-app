"""OpenAI兼容Vision Provider的离线测试。

本文件使用MagicMock和AsyncMock模拟OpenAI兼容客户端。
测试不会读取.env、不会访问网络，也不会调用真实Vision模型。

测试覆盖：

1. Provider协议和构造参数校验；
2. 合法多模态调用及可信追踪字段注入；
3. Markdown代码围栏兼容；
4. 空候选、空内容、非法JSON和无效Schema；
5. asyncio应用层超时和SDK超时；
6. SDK连接失败和HTTP错误状态转换；
7. 未知程序错误不得被伪装成上游错误；
8. 非VisionInput不得进入SDK调用；
9. 最终公开观察不得包含完整图片字节。
"""

import asyncio
from hashlib import (
    sha256,
)
import inspect
import json
from typing import (
    Any,
)
from unittest.mock import (
    AsyncMock,
    MagicMock,
)

import httpx
import pytest

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
)

from app.errors import (
    InvalidVisionResponseError,
    VisionTimeoutError,
    VisionUpstreamError,
)
from app.schemas.vision import (
    VisionImageMetadata,
    VisionInput,
    VisionObservation,
)
from app.services.vision_prompts import (
    VISION_OBSERVATION_PROMPT_VERSION,
)
from app.services.vision_provider import (
    VISION_OBSERVATION_TEMPERATURE,
    OpenAICompatibleVisionProvider,
    VisionProvider,
    parse_vision_model_draft,
    remove_vision_code_fence,
)


# 测试使用固定模型名，避免断言散落多个字符串。
TEST_MODEL = "test-vision-model"


# 12.5秒只作为确定性构造参数，
# Mock调用会立即返回，不会真的等待这些时间。
TEST_TIMEOUT_SECONDS = 12.5


# 固定图片字节用于验证最终结果只保留SHA-256，
# 不会把完整原图复制进VisionObservation。
TEST_IMAGE_BYTES = (
    b"verified-provider-image-bytes"
)


# 根据固定图片字节计算真实摘要，
# 避免测试硬编码一个与图片不一致的SHA值。
TEST_IMAGE_SHA256 = sha256(
    TEST_IMAGE_BYTES
).hexdigest()


def make_vision_input(
) -> VisionInput:
    """构造满足内部契约的确定性VisionInput。"""

    return VisionInput(
        metadata=VisionImageMetadata(
            mime_type="image/png",
            size_bytes=len(
                TEST_IMAGE_BYTES
            ),
            width_px=6,
            height_px=4,
            pixel_count=24,
            sha256_hex=(
                TEST_IMAGE_SHA256
            ),
        ),
        image_bytes=TEST_IMAGE_BYTES,
        analysis_goal=(
            "检查NET指示灯的可见状态"
        ),
        detail="high",
    )


def make_valid_draft_data(
) -> dict[str, object]:
    """构造能够通过VisionModelDraft的模型业务字段。"""

    return {
        "status": "completed",
        "image_quality": "clear",
        "observations": [
            {
                "description": (
                    "设备面板上可见一个红色指示灯"
                ),
                "category": (
                    "visible_condition"
                ),
                "confidence": "high",
                "region": "面板右上区域",
            }
        ],
        "visible_indicators": [
            {
                "label": "NET指示灯",
                "observed_state": "红色常亮",
                "confidence": "high",
                "region": "面板右上区域",
            }
        ],
        "uncertain_items": [],
        "untrusted_text_detected": False,
        "untrusted_text_notes": [],
        "requires_human_check": False,
        "human_check_reasons": [],
    }


def make_valid_content(
) -> str:
    """把合法模型业务字段序列化成模拟assistant正文。"""

    return json.dumps(
        make_valid_draft_data(),
        ensure_ascii=False,
    )


def make_mock_client(
    *,
    content: str | None = None,
    choices: list[object] | None = None,
    error: Exception | None = None,
) -> tuple[MagicMock, AsyncMock]:
    """构造可控制正文、候选列表或异常的假SDK客户端。"""

    if choices is None:
        mock_choice = MagicMock()
        mock_choice.message.content = (
            make_valid_content()
            if content is None
            else content
        )
        resolved_choices = [
            mock_choice,
        ]
    else:
        resolved_choices = choices

    mock_completion = MagicMock()
    mock_completion.choices = (
        resolved_choices
    )

    if error is None:
        mock_create = AsyncMock(
            return_value=mock_completion,
        )
    else:
        mock_create = AsyncMock(
            side_effect=error,
        )

    mock_client = MagicMock()
    mock_client.chat.completions.create = (
        mock_create
    )

    return mock_client, mock_create


def make_provider(
    client: Any,
    *,
    model: str = TEST_MODEL,
    timeout_seconds: float = (
        TEST_TIMEOUT_SECONDS
    ),
) -> OpenAICompatibleVisionProvider:
    """使用统一测试配置创建真实Provider实例。"""

    return OpenAICompatibleVisionProvider(
        client=client,
        model=model,
        timeout_seconds=timeout_seconds,
    )


def make_sdk_request(
) -> httpx.Request:
    """创建只用于SDK异常上下文的假HTTP请求。"""

    return httpx.Request(
        method="POST",
        url=(
            "https://vision.test/"
            "chat/completions"
        ),
    )


def test_protocol_declares_async_analyze_image(
) -> None:
    """Provider协议必须把analyze_image声明为异步方法。"""

    assert inspect.iscoroutinefunction(
        VisionProvider.analyze_image
    )


@pytest.mark.parametrize(
    "model",
    [
        None,
        123,
    ],
    ids=[
        "none",
        "integer",
    ],
)
def test_provider_rejects_non_string_model(
    model: object,
) -> None:
    """模型名不是str时应在任何SDK调用前拒绝。"""

    mock_client, _ = make_mock_client()

    with pytest.raises(
        TypeError,
        match="Vision模型名称必须是str",
    ):
        OpenAICompatibleVisionProvider(
            client=mock_client,
            model=model,  # type: ignore[arg-type]
            timeout_seconds=(
                TEST_TIMEOUT_SECONDS
            ),
        )


@pytest.mark.parametrize(
    "model",
    [
        "",
        "   ",
    ],
    ids=[
        "empty",
        "blank",
    ],
)
def test_provider_rejects_blank_model(
    model: str,
) -> None:
    """空模型名和纯空白模型名都没有可调用含义。"""

    mock_client, _ = make_mock_client()

    with pytest.raises(
        ValueError,
        match="Vision模型名称不能为空",
    ):
        make_provider(
            mock_client,
            model=model,
        )


@pytest.mark.parametrize(
    "timeout_seconds",
    [
        True,
        "60",
    ],
    ids=[
        "boolean",
        "string",
    ],
)
def test_provider_rejects_non_numeric_timeout(
    timeout_seconds: object,
) -> None:
    """bool和字符串不能作为Vision调用超时。"""

    mock_client, _ = make_mock_client()

    with pytest.raises(
        TypeError,
        match="timeout_seconds必须是数字",
    ):
        OpenAICompatibleVisionProvider(
            client=mock_client,
            model=TEST_MODEL,
            timeout_seconds=(
                timeout_seconds
            ),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "timeout_seconds",
    [
        0.0,
        -1.0,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
    ids=[
        "zero",
        "negative",
        "nan",
        "positive-infinity",
        "negative-infinity",
    ],
)
def test_provider_rejects_non_positive_or_non_finite_timeout(
    timeout_seconds: float,
) -> None:
    """超时必须是大于零的有限数字。"""

    mock_client, _ = make_mock_client()

    with pytest.raises(
        ValueError,
        match="大于0的有限数字",
    ):
        make_provider(
            mock_client,
            timeout_seconds=(
                timeout_seconds
            ),
        )


@pytest.mark.asyncio
async def test_provider_returns_tracked_observation(
) -> None:
    """合法模型JSON应形成带真实追踪字段的最终观察。"""

    mock_client, mock_create = (
        make_mock_client()
    )
    provider = make_provider(
        mock_client,
        # 两端空白用于验证构造器会清理模型名。
        model=f"  {TEST_MODEL}  ",
    )
    vision_input = make_vision_input()

    result = await provider.analyze_image(
        vision_input=vision_input,
    )

    assert isinstance(
        result,
        VisionObservation,
    )
    assert result.status == "completed"
    assert result.image_quality == "clear"
    assert result.source == "vision_model"
    assert result.source_image_sha256 == (
        TEST_IMAGE_SHA256
    )
    assert result.prompt_version == (
        VISION_OBSERVATION_PROMPT_VERSION
    )
    assert result.model_name == TEST_MODEL
    assert result.visible_indicators[
        0
    ].label == "NET指示灯"

    mock_create.assert_awaited_once()
    awaited_kwargs = (
        mock_create.await_args.kwargs
    )

    assert awaited_kwargs["model"] == (
        TEST_MODEL
    )
    assert awaited_kwargs[
        "temperature"
    ] == VISION_OBSERVATION_TEMPERATURE
    assert awaited_kwargs["timeout"] == (
        TEST_TIMEOUT_SECONDS
    )

    messages = awaited_kwargs["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"

    user_content = messages[1][
        "content"
    ]
    assert isinstance(user_content, list)
    assert user_content[0]["type"] == "text"
    assert user_content[1]["type"] == (
        "image_url"
    )

    # 当前通用Provider不发送供应商私有字段。
    assert "extra_body" not in (
        awaited_kwargs
    )


def test_parser_accepts_json_code_fence(
) -> None:
    """只包住整个JSON的Markdown围栏应被兼容删除。"""

    fenced_content = (
        "```json\n"
        f"{make_valid_content()}\n"
        "```"
    )

    result = parse_vision_model_draft(
        fenced_content
    )

    assert result.status == "completed"
    assert result.visible_indicators[
        0
    ].observed_state == "红色常亮"


def test_remove_code_fence_does_not_strip_plain_json(
) -> None:
    """没有围栏的JSON只能清除整体两端空白。"""

    content = make_valid_content()

    assert remove_vision_code_fence(
        f"  {content}\n"
    ) == content


@pytest.mark.parametrize(
    "content",
    [
        None,
        "",
        "   ",
    ],
    ids=[
        "none",
        "empty",
        "blank",
    ],
)
def test_parser_rejects_empty_content(
    content: str | None,
) -> None:
    """None、空字符串和纯空白都不能形成视觉观察。"""

    with pytest.raises(
        InvalidVisionResponseError,
        match="空内容",
    ):
        parse_vision_model_draft(
            content
        )


def test_parser_rejects_invalid_json(
) -> None:
    """普通说明文字不得绕过结构化JSON契约。"""

    with pytest.raises(
        InvalidVisionResponseError,
        match="不是合法JSON",
    ):
        parse_vision_model_draft(
            "图片中似乎有一个红灯"
        )


def test_parser_rejects_invalid_draft_schema(
) -> None:
    """JSON合法但跨字段状态矛盾时必须拒绝。"""

    invalid_data = make_valid_draft_data()

    # unusable状态不能继续携带看似可用的观察，
    # 且必须要求人工检查并说明不确定内容。
    invalid_data["status"] = "unusable"
    invalid_data["image_quality"] = (
        "unusable"
    )

    with pytest.raises(
        InvalidVisionResponseError,
        match="不符合内部观察契约",
    ):
        parse_vision_model_draft(
            json.dumps(
                invalid_data,
                ensure_ascii=False,
            )
        )


@pytest.mark.asyncio
async def test_provider_rejects_empty_choices(
) -> None:
    """SDK没有返回候选结果时应转换成无效响应异常。"""

    mock_client, mock_create = (
        make_mock_client(
            choices=[],
        )
    )
    provider = make_provider(
        mock_client
    )

    with pytest.raises(
        InvalidVisionResponseError,
        match="没有候选结果",
    ):
        await provider.analyze_image(
            vision_input=(
                make_vision_input()
            ),
        )

    mock_create.assert_awaited_once()


@pytest.mark.asyncio
async def test_provider_rejects_empty_assistant_content(
) -> None:
    """候选存在但assistant正文为空时必须拒绝。"""

    mock_client, mock_create = (
        make_mock_client(
            content="   ",
        )
    )
    provider = make_provider(
        mock_client
    )

    with pytest.raises(
        InvalidVisionResponseError,
        match="空内容",
    ):
        await provider.analyze_image(
            vision_input=(
                make_vision_input()
            ),
        )

    mock_create.assert_awaited_once()


@pytest.mark.asyncio
async def test_provider_converts_sdk_timeout(
) -> None:
    """OpenAI SDK超时必须转换成VisionTimeoutError。"""

    sdk_error = APITimeoutError(
        request=make_sdk_request(),
    )
    mock_client, mock_create = (
        make_mock_client(
            error=sdk_error,
        )
    )
    provider = make_provider(
        mock_client
    )

    with pytest.raises(
        VisionTimeoutError,
        match="响应超时",
    ):
        await provider.analyze_image(
            vision_input=(
                make_vision_input()
            ),
        )

    mock_create.assert_awaited_once()


@pytest.mark.asyncio
async def test_provider_enforces_application_timeout(
) -> None:
    """客户端不主动超时时，asyncio边界仍必须终止等待。"""

    async def wait_until_cancelled(
        **kwargs: object,
    ) -> object:
        """模拟永不主动返回、只能由超时取消的SDK调用。"""

        del kwargs

        never_set_event = asyncio.Event()

        return await never_set_event.wait()

    mock_create = AsyncMock(
        side_effect=wait_until_cancelled,
    )
    mock_client = MagicMock()
    mock_client.chat.completions.create = (
        mock_create
    )
    provider = make_provider(
        mock_client,
        timeout_seconds=0.01,
    )

    with pytest.raises(
        VisionTimeoutError,
        match="响应超时",
    ):
        await provider.analyze_image(
            vision_input=(
                make_vision_input()
            ),
        )

    mock_create.assert_awaited_once()


@pytest.mark.asyncio
async def test_provider_converts_connection_error(
) -> None:
    """DNS或连接失败必须转换成VisionUpstreamError。"""

    sdk_error = APIConnectionError(
        request=make_sdk_request(),
    )
    mock_client, mock_create = (
        make_mock_client(
            error=sdk_error,
        )
    )
    provider = make_provider(
        mock_client
    )

    with pytest.raises(
        VisionUpstreamError,
        match="无法连接Vision模型服务",
    ):
        await provider.analyze_image(
            vision_input=(
                make_vision_input()
            ),
        )

    mock_create.assert_awaited_once()


@pytest.mark.asyncio
async def test_provider_converts_api_status_error(
) -> None:
    """上游503状态必须转换并保留非敏感状态码。"""

    sdk_request = make_sdk_request()
    sdk_response = httpx.Response(
        status_code=503,
        request=sdk_request,
    )
    sdk_error = APIStatusError(
        "Service Unavailable",
        response=sdk_response,
        body={
            "error": "unavailable",
        },
    )
    mock_client, mock_create = (
        make_mock_client(
            error=sdk_error,
        )
    )
    provider = make_provider(
        mock_client
    )

    with pytest.raises(
        VisionUpstreamError,
        match="错误状态：503",
    ):
        await provider.analyze_image(
            vision_input=(
                make_vision_input()
            ),
        )

    mock_create.assert_awaited_once()


@pytest.mark.asyncio
async def test_provider_does_not_hide_programming_error(
) -> None:
    """未知RuntimeError应原样暴露，不得伪装成上游失败。"""

    programming_error = RuntimeError(
        "synthetic programming error"
    )
    mock_client, mock_create = (
        make_mock_client(
            error=programming_error,
        )
    )
    provider = make_provider(
        mock_client
    )

    with pytest.raises(
        RuntimeError,
        match="synthetic programming error",
    ):
        await provider.analyze_image(
            vision_input=(
                make_vision_input()
            ),
        )

    mock_create.assert_awaited_once()


@pytest.mark.asyncio
async def test_provider_rejects_non_vision_input_before_sdk(
) -> None:
    """未经Adapter处理的字典不能进入真实模型调用。"""

    mock_client, mock_create = (
        make_mock_client()
    )
    provider = make_provider(
        mock_client
    )

    with pytest.raises(
        TypeError,
        match="vision_input必须是VisionInput",
    ):
        await provider.analyze_image(
            vision_input={
                "analysis_goal": "绕过输入适配器",
            },  # type: ignore[arg-type]
        )

    mock_create.assert_not_awaited()


@pytest.mark.asyncio
async def test_public_observation_excludes_full_image(
) -> None:
    """最终可序列化结果只能保留摘要，不能包含完整图片。"""

    mock_client, _ = make_mock_client()
    provider = make_provider(
        mock_client
    )

    result = await provider.analyze_image(
        vision_input=make_vision_input(),
    )
    public_data = result.model_dump(
        mode="json"
    )
    serialized_public_data = json.dumps(
        public_data,
        ensure_ascii=False,
    )

    assert public_data[
        "source_image_sha256"
    ] == TEST_IMAGE_SHA256
    assert "image_bytes" not in (
        serialized_public_data
    )
    assert "image_base64" not in (
        serialized_public_data
    )
    assert (
        TEST_IMAGE_BYTES.decode("ascii")
        not in serialized_public_data
    )
