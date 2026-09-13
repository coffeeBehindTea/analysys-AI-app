"""创建 OpenAI 兼容的异步 LLM 客户端并读取模型名称。"""

# AsyncOpenAI 是 OpenAI Python SDK 的异步客户端类。
from openai import AsyncOpenAI

from app.config import Settings
from app.errors import LLMConfigurationError


def create_llm_client(settings: Settings) -> AsyncOpenAI:
    """根据 Settings 创建异步 LLM 客户端。"""

    # SecretStr 对象存在，不代表它的内容一定非空。
    # 因此同时检查 None 和去除空格后的真实内容。
    if (
        settings.llm_api_key is None
        or not settings.llm_api_key.get_secret_value().strip()
    ):
        raise LLMConfigurationError("LLM_API_KEY 未配置")

    # 官方 OpenAI 服务可以使用 SDK 默认地址；
    # 兼容服务则通过 LLM_BASE_URL 指定自己的地址。
    base_url = (
        str(settings.llm_base_url)
        if settings.llm_base_url is not None
        else None
    )

    # AsyncOpenAI(...) 只创建客户端；真正请求在 chat.completions.create() 时发生。
    return AsyncOpenAI(
        # api_key 用于上游 Bearer 身份验证。
        api_key=settings.llm_api_key.get_secret_value(),
        # base_url 允许 SDK 连接实现 OpenAI 兼容协议的其他服务商。
        base_url=base_url,
        # timeout 是单次 SDK 请求允许等待的秒数。
        timeout=30.0,
    )


def get_llm_model(settings: Settings) -> str:
    """取得模型名称，并检查是否为空。"""

    # 同时处理 None、空字符串和只包含空格的情况。
    if (
        settings.llm_model is None
        or not settings.llm_model.strip()
    ):
        raise LLMConfigurationError("LLM_MODEL 未配置")

    return settings.llm_model
