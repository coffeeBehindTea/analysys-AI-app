"""Vision FastAPI依赖装配函数的离线单元测试。

本文件不读取真实.env、不解码图片，也不访问Vision服务。

测试覆盖：

1. get_vision_client把Settings交给客户端工厂；
2. 正常结束依赖时关闭异步客户端；
3. 下游抛出异常时仍关闭异步客户端；
4. get_vision_input_adapter传递全部图片资源限制；
5. get_vision_provider传递客户端、模型和超时；
6. Provider组装不会错误读取文本LLM模型。
"""

from unittest.mock import (
    AsyncMock,
    MagicMock,
    patch,
)

import pytest

from app import dependencies
from app.config import Settings


# 以下常量只用于依赖装配测试，
# 不包含真实服务凭证或真实供应商地址。
TEST_VISION_API_KEY = "vision-dependency-key"
TEST_VISION_BASE_URL = (
    "https://vision-dependency.example/v1/"
)
TEST_VISION_MODEL = (
    "vision-dependency-model"
)
TEST_VISION_TIMEOUT_SECONDS = 37.5


def make_settings() -> Settings:
    """构造不读取本地.env的完整Vision测试配置。"""

    return Settings(
        _env_file=None,

        # 文本LLM使用不同模型，
        # 用于发现Vision依赖是否发生错误串用。
        llm_api_key="llm-dependency-key",
        llm_base_url=(
            "https://llm-dependency.example/v1"
        ),
        llm_model="llm-dependency-model",

        vision_api_key=TEST_VISION_API_KEY,
        vision_base_url=TEST_VISION_BASE_URL,
        vision_model=TEST_VISION_MODEL,
        vision_timeout_seconds=(
            TEST_VISION_TIMEOUT_SECONDS
        ),
        max_vision_image_size_bytes=1_500_000,
        max_vision_image_dimension_px=2_048,
        max_vision_image_pixels=3_000_000,
    )


@pytest.mark.asyncio
async def test_get_vision_client_yields_factory_result_and_closes_it(
) -> None:
    """正常消费依赖后应返回并关闭同一个客户端。"""

    settings = make_settings()

    # AsyncMock模拟需要await的close()方法，
    # 并记录它被等待了多少次。
    fake_client = MagicMock()
    fake_client.close = AsyncMock()

    with patch(
        "app.dependencies.create_vision_client",
        return_value=fake_client,
    ) as client_factory:
        # 调用含yield的async def会返回异步生成器，
        # 此时函数体还没有执行。
        dependency_iterator = (
            dependencies.get_vision_client(
                settings
            )
        )

        # anext()执行到yield并取得依赖提供的客户端。
        yielded_client = await anext(
            dependency_iterator
        )

        assert yielded_client is fake_client

        # aclose()让异步生成器从yield处继续进入finally。
        await dependency_iterator.aclose()

    client_factory.assert_called_once_with(
        settings
    )
    fake_client.close.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_get_vision_client_closes_client_when_downstream_fails(
) -> None:
    """下游异常进入依赖时也必须执行资源清理。"""

    settings = make_settings()
    fake_client = MagicMock()
    fake_client.close = AsyncMock()

    with patch(
        "app.dependencies.create_vision_client",
        return_value=fake_client,
    ):
        dependency_iterator = (
            dependencies.get_vision_client(
                settings
            )
        )

        await anext(
            dependency_iterator
        )

        # athrow()把异常送回异步生成器的yield位置，
        # 模拟使用该依赖的下游Service或Router失败。
        with pytest.raises(
            RuntimeError,
            match="downstream failed",
        ):
            await dependency_iterator.athrow(
                RuntimeError(
                    "downstream failed"
                )
            )

    # finally必须在原异常继续向外传播前关闭客户端。
    fake_client.close.assert_awaited_once_with()


def test_get_vision_input_adapter_passes_resource_limits(
) -> None:
    """输入适配器必须取得Settings中的三项图片限制。"""

    settings = make_settings()
    fake_adapter = object()

    with patch(
        "app.dependencies.VisionInputAdapter",
        return_value=fake_adapter,
    ) as adapter_constructor:
        result = (
            dependencies
            .get_vision_input_adapter(
                settings
            )
        )

    assert result is fake_adapter

    adapter_constructor.assert_called_once_with(
        max_image_size_bytes=1_500_000,
        max_image_dimension_px=2_048,
        max_image_pixels=3_000_000,
    )


def test_get_vision_provider_passes_client_model_and_timeout(
) -> None:
    """Provider必须由Vision客户端、Vision模型和超时组装。"""

    settings = make_settings()
    fake_client = object()
    fake_provider = object()

    # MagicMock(return_value=...)创建可调用测试替身，
    # 调用后返回指定对象并保存调用参数。
    model_reader = MagicMock(
        return_value=TEST_VISION_MODEL
    )

    with (
        patch(
            "app.dependencies.get_vision_model",
            model_reader,
        ),
        patch(
            "app.dependencies.OpenAICompatibleVisionProvider",
            return_value=fake_provider,
        ) as provider_constructor,
    ):
        result = (
            dependencies.get_vision_provider(
                client=(
                    fake_client  # type: ignore[arg-type]
                ),
                settings=settings,
            )
        )

    assert result is fake_provider

    # 模型名必须由Vision专用读取函数取得，
    # 而不是使用settings.llm_model。
    model_reader.assert_called_once_with(
        settings
    )

    provider_constructor.assert_called_once_with(
        client=fake_client,
        model=TEST_VISION_MODEL,
        timeout_seconds=(
            TEST_VISION_TIMEOUT_SECONDS
        ),
    )
