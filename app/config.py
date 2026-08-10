"""集中读取和校验应用配置。"""

# lru_cache 会缓存无参数函数的返回值，避免每次请求都重新读取 .env。
from functools import lru_cache

# HttpUrl 校验 HTTP/HTTPS 地址；SecretStr 避免密钥在打印时直接显示。
from pydantic import HttpUrl, SecretStr

# BaseSettings 是专门读取环境变量的 Pydantic 模型基类；
# SettingsConfigDict 用于配置 .env、编码、大小写等读取规则。
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用支持的 LLM 环境变量及其类型。"""

    # model_config 是 Pydantic v2 的模型级配置。
    model_config = SettingsConfigDict(
        # Settings() 创建时自动读取项目工作目录下的 .env。
        env_file=".env",
        # 使用 UTF-8 解析 .env 文件。
        env_file_encoding="utf-8",
        # 环境变量名不区分大小写，llm_model 可匹配 LLM_MODEL。
        case_sensitive=False,
        # 自动删除字符串配置首尾空格。
        str_strip_whitespace=True,
        # .env 中存在当前模型未声明的其他配置时不报错。
        extra="ignore",
    )

    # None 表示配置可以在应用启动时缺失，在真正调用 LLM 时再校验。
    llm_api_key: SecretStr | None = None
    llm_base_url: HttpUrl | None = None
    llm_model: str | None = None


@lru_cache
def get_settings() -> Settings:
    """返回当前进程缓存的 Settings 对象。"""

    # 第一次调用会读取环境变量和 .env；后续调用返回同一缓存对象。
    return Settings()
