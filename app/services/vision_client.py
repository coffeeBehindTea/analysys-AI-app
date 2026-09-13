"""创建OpenAI兼容的异步Vision客户端并读取模型名称。

本模块只负责把Settings中的Vision配置转换成：

1. 可以交给Vision Provider使用的AsyncOpenAI客户端；
2. 经过非空校验的Vision模型名称。

本模块不负责读取图片、构造Prompt、调用模型或解析模型响应。
"""

# AsyncOpenAI是OpenAI Python SDK提供的异步客户端。
#
# 创建这个对象时不会立即访问网络。
# 真正的网络请求发生在Provider调用：
#
# await client.chat.completions.create(...)
from openai import AsyncOpenAI

# Settings集中保存从环境变量和.env读取的应用配置。
from app.config import Settings

# VisionConfigurationError表示视觉模型配置缺失，
# 它与网络超时、上游错误等运行阶段异常分开。
from app.errors import VisionConfigurationError


def create_vision_client(
    settings: Settings,
) -> AsyncOpenAI:
    """根据应用配置创建Vision专用异步客户端。"""

    # 先检查调用方传入的对象类型。
    #
    # 如果误传dict或其他对象，就在配置边界给出明确错误，
    # 而不是等访问settings.vision_api_key时才出现AttributeError。
    if not isinstance(
        settings,
        Settings,
    ):
        raise TypeError(
            "settings必须是Settings"
        )

    # SecretStr对象存在不代表内部一定有有效内容，
    # 因此需要同时处理：
    #
    # 1. 字段为None；
    # 2. 内容为空字符串；
    # 3. 内容只包含空格。
    if settings.vision_api_key is None:
        raise VisionConfigurationError(
            "VISION_API_KEY 未配置"
        )

    # get_secret_value()取得SecretStr内部真实字符串。
    #
    # 只在即将把密钥交给SDK的边界解包，
    # 可以减少密钥在其他业务代码中传播的范围。
    api_key = (
        settings
        .vision_api_key
        .get_secret_value()
        .strip()
    )

    if not api_key:
        raise VisionConfigurationError(
            "VISION_API_KEY 未配置"
        )

    # VISION_BASE_URL是可选配置。
    #
    # 未配置时传入None，SDK使用自己的默认服务地址；
    # 配置后，HttpUrl对象需要转换成普通str再交给SDK。
    base_url = (
        str(settings.vision_base_url)
        if settings.vision_base_url is not None
        else None
    )

    # AsyncOpenAI(...)只构造客户端，不会立即发送请求。
    return AsyncOpenAI(
        # SDK会使用这个值生成身份验证请求头。
        api_key=api_key,

        # 允许连接实现OpenAI兼容协议的视觉服务。
        base_url=base_url,

        # 这是SDK底层HTTP请求的超时设置。
        #
        # VisionProvider还会使用asyncio.timeout()限制
        # 整个异步调用过程，二者构成两层超时保护。
        timeout=settings.vision_timeout_seconds,
    )


def get_vision_model(
    settings: Settings,
) -> str:
    """取得并校验Vision模型名称。"""

    if not isinstance(
        settings,
        Settings,
    ):
        raise TypeError(
            "settings必须是Settings"
        )

    # 同时拒绝None、空字符串和纯空格字符串。
    if (
        settings.vision_model is None
        or not settings.vision_model.strip()
    ):
        raise VisionConfigurationError(
            "VISION_MODEL 未配置"
        )

    # 返回去除两端空白后的模型名称。
    #
    # 这个字符串会在Provider中作为
    # chat.completions.create(model=...)的model参数。
    return settings.vision_model.strip()