"""Vision独立客户端工厂的离线单元测试。

本文件不会访问真实Vision服务，也不会读取项目.env。

测试覆盖：

1. Vision必须使用自己的API Key，不能隐式复用文本LLM Key；
2. 空白API Key必须在创建SDK客户端前被拒绝；
3. Vision模型名缺失或为空时必须返回明确配置错误；
4. Vision客户端必须取得自己的Base URL和超时配置；
5. 未配置Base URL时必须把None交给SDK，由SDK选择默认地址；
6. 两个公开函数都必须拒绝错误的Settings参数类型。
"""

from unittest.mock import (
    patch,
)

import pytest

from app.config import Settings
from app.errors import (
    VisionConfigurationError,
)
from app.services.vision_client import (
    create_vision_client,
    get_vision_model,
)


# 这些常量只用于离线测试。
#
# 它们不是任何真实服务的密钥、地址或模型名，
# 集中声明可以让每个测试使用相同且易识别的输入。
TEST_VISION_API_KEY = "vision-test-key"
TEST_VISION_BASE_URL = (
    "https://vision.example/v1/"
)
TEST_VISION_MODEL = "vision-test-model"
TEST_VISION_TIMEOUT_SECONDS = 42.5


def test_create_vision_client_requires_own_api_key(
) -> None:
    """存在LLM Key时，Vision Key缺失仍必须报错。"""

    settings = Settings(
        # 禁止读取开发者本地.env，
        # 确保缺失的Vision Key不会被外部配置补上。
        _env_file=None,

        # 故意提供完整的文本LLM配置。
        # 如果Vision工厂错误复用了它，测试就会失败。
        llm_api_key="llm-test-key",
        llm_base_url="https://llm.example/v1",
        llm_model="llm-test-model",

        vision_api_key=None,
        vision_base_url=TEST_VISION_BASE_URL,
        vision_model=TEST_VISION_MODEL,
    )

    # patch()只在with代码块内替换
    # vision_client模块引用的AsyncOpenAI。
    #
    # 这里同时验证配置错误发生在SDK客户端创建之前。
    with patch(
        "app.services.vision_client.AsyncOpenAI"
    ) as client_constructor:
        with pytest.raises(
            VisionConfigurationError,
            match="VISION_API_KEY",
        ):
            create_vision_client(
                settings
            )

        client_constructor.assert_not_called()


def test_create_vision_client_rejects_blank_api_key(
) -> None:
    """只包含空白的Vision Key不能通过配置边界。"""

    settings = Settings(
        _env_file=None,
        vision_api_key="   ",
        vision_base_url=TEST_VISION_BASE_URL,
        vision_model=TEST_VISION_MODEL,
    )

    with pytest.raises(
        VisionConfigurationError,
        match="VISION_API_KEY",
    ):
        create_vision_client(
            settings
        )


def test_get_vision_model_requires_own_model_name(
) -> None:
    """文本LLM模型存在时，Vision模型缺失仍必须报错。"""

    settings = Settings(
        _env_file=None,
        llm_api_key="llm-test-key",
        llm_base_url="https://llm.example/v1",
        llm_model="llm-test-model",
        vision_api_key=TEST_VISION_API_KEY,
        vision_base_url=TEST_VISION_BASE_URL,
        vision_model=None,
    )

    with pytest.raises(
        VisionConfigurationError,
        match="VISION_MODEL",
    ):
        get_vision_model(
            settings
        )


def test_get_vision_model_rejects_blank_name(
) -> None:
    """只包含空白的Vision模型名必须被拒绝。"""

    settings = Settings(
        _env_file=None,
        vision_api_key=TEST_VISION_API_KEY,
        vision_base_url=TEST_VISION_BASE_URL,
        vision_model="   ",
    )

    with pytest.raises(
        VisionConfigurationError,
        match="VISION_MODEL",
    ):
        get_vision_model(
            settings
        )


def test_get_vision_model_returns_cleaned_name(
) -> None:
    """合法Vision模型名应以清理后的字符串返回。"""

    settings = Settings(
        _env_file=None,
        vision_api_key=TEST_VISION_API_KEY,
        vision_base_url=TEST_VISION_BASE_URL,
        vision_model=(
            f"  {TEST_VISION_MODEL}  "
        ),
    )

    assert (
        get_vision_model(settings)
        == TEST_VISION_MODEL
    )


def test_create_vision_client_passes_vision_configuration(
) -> None:
    """客户端工厂应把Vision专用配置完整传给SDK。"""

    settings = Settings(
        _env_file=None,

        # 两组配置故意使用不同值，
        # 用于证明工厂没有读取文本LLM字段。
        llm_api_key="llm-test-key",
        llm_base_url="https://llm.example/v1",
        llm_model="llm-test-model",

        vision_api_key=TEST_VISION_API_KEY,
        vision_base_url=TEST_VISION_BASE_URL,
        vision_model=TEST_VISION_MODEL,
        vision_timeout_seconds=(
            TEST_VISION_TIMEOUT_SECONDS
        ),
    )

    with patch(
        "app.services.vision_client.AsyncOpenAI"
    ) as client_constructor:
        # return_value是Mock构造器被调用后
        # 应该返回的伪客户端对象。
        fake_client = object()
        client_constructor.return_value = (
            fake_client
        )

        result = create_vision_client(
            settings
        )

    # 工厂必须把SDK构造器的返回值原样返回。
    assert result is fake_client

    # assert_called_once_with()同时验证：
    #
    # 1. SDK构造器只调用一次；
    # 2. 参数名和值完全符合预期；
    # 3. 没有误用LLM配置。
    client_constructor.assert_called_once_with(
        api_key=TEST_VISION_API_KEY,
        base_url=TEST_VISION_BASE_URL,
        timeout=TEST_VISION_TIMEOUT_SECONDS,
    )


def test_create_vision_client_allows_default_base_url(
) -> None:
    """Base URL缺失时应让SDK选择自己的默认地址。"""

    settings = Settings(
        _env_file=None,
        vision_api_key=TEST_VISION_API_KEY,
        vision_base_url=None,
        vision_model=TEST_VISION_MODEL,
        vision_timeout_seconds=(
            TEST_VISION_TIMEOUT_SECONDS
        ),
    )

    with patch(
        "app.services.vision_client.AsyncOpenAI"
    ) as client_constructor:
        create_vision_client(
            settings
        )

    # base_url=None表示应用没有指定兼容服务地址。
    # SDK随后使用自己的默认API地址。
    client_constructor.assert_called_once_with(
        api_key=TEST_VISION_API_KEY,
        base_url=None,
        timeout=TEST_VISION_TIMEOUT_SECONDS,
    )


@pytest.mark.parametrize(
    "operation",
    [
        create_vision_client,
        get_vision_model,
    ],
    ids=[
        "create-client",
        "get-model",
    ],
)
def test_vision_client_functions_reject_wrong_settings_type(
    operation: object,
) -> None:
    """公开函数都必须拒绝普通dict代替Settings。"""

    # callable()是Python内置函数，
    # 用于确认参数确实是可以调用的函数对象。
    assert callable(operation)

    with pytest.raises(
        TypeError,
        match="settings必须是Settings",
    ):
        # 本测试有意在运行时传入错误类型，
        # 因此不使用静态类型检查推断其合法性。
        operation(  # type: ignore[operator]
            {}
        )
