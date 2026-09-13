"""Embedding 独立客户端工厂的离线测试。"""

import pytest

from app.config import Settings
from app.errors import EmbeddingConfigurationError
from app.services.embedding_client import (
    create_embedding_client,
    get_embedding_model,
)


def test_create_embedding_client_requires_own_api_key() -> None:
    """Embedding 不能错误复用 LLM API Key。"""

    settings = Settings(
        _env_file=None,

        # 即使 LLM Key 存在，
        # Embedding 自己的 Key 缺失也必须报错。
        llm_api_key="llm-test-key",
        llm_base_url="https://llm.example/v1",
        llm_model="llm-model",

        embedding_api_key=None,
        embedding_base_url=(
            "https://embedding.example/v1"
        ),
        embedding_model="embedding-model",
    )

    with pytest.raises(
        EmbeddingConfigurationError,
        match="EMBEDDING_API_KEY",
    ):
        create_embedding_client(settings)


def test_get_embedding_model_rejects_blank_name() -> None:
    """Embedding 模型名不能只包含空白。"""

    settings = Settings(
        _env_file=None,
        embedding_api_key="embedding-test-key",
        embedding_base_url=(
            "https://embedding.example/v1"
        ),
        embedding_model="   ",
    )

    with pytest.raises(
        EmbeddingConfigurationError,
        match="EMBEDDING_MODEL",
    ):
        get_embedding_model(settings)


@pytest.mark.asyncio
async def test_create_embedding_client_uses_embedding_base_url() -> None:
    """客户端应使用 Embedding 地址，而不是 LLM 地址。"""

    settings = Settings(
        _env_file=None,
        llm_api_key="llm-test-key",
        llm_base_url="https://llm.example/v1",
        llm_model="llm-model",

        embedding_api_key="embedding-test-key",
        embedding_base_url=(
            "https://embedding.example/v1/"
        ),
        embedding_model="embedding-model",
    )

    client = create_embedding_client(settings)

    try:
        # base_url 是 AsyncOpenAI 客户端提供的属性，
        # str() 把 URL 对象转成普通字符串。
        assert str(client.base_url) == (
            "https://embedding.example/v1/"
        )
    finally:
        # close() 是异步资源清理方法。
        await client.close()