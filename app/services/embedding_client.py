"""创建 OpenAI 兼容的异步 Embedding 客户端并读取模型配置。"""

from openai import AsyncOpenAI

from app.config import Settings
from app.errors import EmbeddingConfigurationError


def create_embedding_client(
    settings: Settings,
) -> AsyncOpenAI:
    """使用独立的 Embedding 配置创建异步客户端。"""

    # SecretStr 对象存在，不代表其中一定有真实内容，
    # 所以还需要检查去除空白后的密钥。
    if (
        settings.embedding_api_key is None
        or not settings.embedding_api_key
        .get_secret_value()
        .strip()
    ):
        raise EmbeddingConfigurationError(
            "EMBEDDING_API_KEY 未配置"
        )

    # 没有配置 Base URL 时传入 None，
    # AsyncOpenAI 会使用其默认官方地址。
    #
    # 使用兼容服务时，必须显式配置对应地址。
    base_url = (
        str(settings.embedding_base_url)
        if settings.embedding_base_url is not None
        else None
    )

    # 创建客户端不会立即联网。
    # 真正的网络请求发生在 embeddings.create()。
    return AsyncOpenAI(
        api_key=(
            settings.embedding_api_key
            .get_secret_value()
        ),
        base_url=base_url,
        timeout=30.0,
    )


def get_embedding_model(
    settings: Settings,
) -> str:
    """取得并校验 Embedding 模型名称。"""

    if (
        settings.embedding_model is None
        or not settings.embedding_model.strip()
    ):
        raise EmbeddingConfigurationError(
            "EMBEDDING_MODEL 未配置"
        )

    return settings.embedding_model