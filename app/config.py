"""集中读取和校验应用配置。"""

# lru_cache 缓存无参数函数的返回值，
# 避免每次请求都重新读取 .env。
from functools import lru_cache

# Path 表示文件系统路径。
from pathlib import Path

# Self 表示当前类 Settings 自身的类型。
from typing import Self

# Field 为字段增加数值、长度等约束；
# HttpUrl 校验 HTTP/HTTPS 地址；
# SecretStr 降低密钥被意外打印的风险；
# model_validator 校验多个字段之间的关系。
from pydantic import (
    Field,
    HttpUrl,
    SecretStr,
    model_validator,
)

# BaseSettings 从环境变量和 .env 中读取配置；
# SettingsConfigDict 配置其读取行为。
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)

from app.schemas.vision import (
    ABSOLUTE_MAX_VISION_IMAGE_BYTES,
    ABSOLUTE_MAX_VISION_IMAGE_DIMENSION_PX,
    ABSOLUTE_MAX_VISION_IMAGE_PIXELS,
)


class Settings(BaseSettings):
    """集中读取 LLM、Embedding、Vision 与应用策略配置。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        str_strip_whitespace=True,
        extra="ignore",
    )

    # ---------- 生成式 LLM 配置 ----------
    #
    # 用于故障摘要、建议生成等文本生成任务。
    llm_api_key: SecretStr | None = None
    llm_base_url: HttpUrl | None = None
    llm_model: str | None = None

    # ---------- Embedding 配置 ----------
    #
    # Embedding 可能与 LLM 来自不同厂商，
    # 因此必须拥有独立的 API Key 和 Base URL。
    embedding_api_key: SecretStr | None = None
    embedding_base_url: HttpUrl | None = None
    embedding_model: str | None = None

        # ---------- 知识库与摄取配置 ----------

    # ChromaDB 数据持久化目录。
    #
    # Path 会把环境变量中的字符串转换为路径对象，
    # 但创建 Settings 时不会自动创建这个目录。
    # 真正创建目录的是 Chroma PersistentClient。
    chroma_persist_directory: Path = Path(
        "chroma_data"
    )

    # Chroma Collection 类似关系数据库中的表。
    #
    # 同一个 Collection 内的向量必须来自
    # 同一个 Embedding 模型和维度。
    chroma_collection_name: str = Field(
        default="robot_knowledge",
        min_length=3,
        max_length=512,
    )

    # Day 1–2 的两套参数在当前语料上打平。
    #
    # 这里暂时选择 Chunk 数量略少的 800/120，
    # PDF 加入评测后仍需要重新比较。
    rag_chunk_size: int = Field(
        default=800,
        ge=1,
        le=20_000,
    )

    rag_chunk_overlap: int = Field(
        default=120,
        ge=0,
    )

    # ---------- RAG在线问答策略 ----------

    # 只有Top-1相似度达到该阈值，
    # KnowledgeQueryService才会调用LLM。
    #
    # 0.60来自robot_knowledge_v4上的20题校准结果：
    # - 7道可回答题被放行；
    # - 这些题的Top-3均包含完整Gold证据；
    # - 4道无答案题全部被拒答；
    # - 低于0.60会开始放行已知错误检索q008。
    #
    # 这是当前语料、Embedding模型和评测集上的
    # 保守基线，不是适用于所有知识库的通用阈值。
    rag_similarity_threshold: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
    )
    # 一次 Embedding 请求最多包含多少段文本。
    embedding_batch_size: int = Field(
        default=64,
        ge=1,
    )

    # 单个上传文件最多允许 20 MiB。
    #
    # 这是字节数，不是字符数。
    max_document_size_bytes: int = Field(
        default=20 * 1024 * 1024,
        ge=1,
    )

    # ---------- Vision模型与输入策略 ----------

    # Vision服务可能与文本LLM来自不同厂商，
    # 也可能使用不同权限和不同计费账户。
    # 因此不自动复用LLM_API_KEY和LLM_BASE_URL。
    vision_api_key: SecretStr | None = None
    vision_base_url: HttpUrl | None = None
    vision_model: str | None = None

    # 单张图片的业务字节上限默认为5 MiB。
    #
    # 业务上限比Schema层20 MiB绝对边界更严格，
    # 可以通过环境变量调小，但不能超过绝对边界。
    max_vision_image_size_bytes: int = Field(
        default=5 * 1024 * 1024,
        ge=1,
        le=ABSOLUTE_MAX_VISION_IMAGE_BYTES,
    )

    # 图片任意一边允许的最大像素尺寸。
    #
    # 这可以在完整解码像素前拒绝异常宽图或长图。
    max_vision_image_dimension_px: int = Field(
        default=4_096,
        ge=1,
        le=(
            ABSOLUTE_MAX_VISION_IMAGE_DIMENSION_PX
        ),
    )

    # 图片总像素数的业务上限。
    #
    # 文件体积较小不代表解码后的像素较少，
    # 单独限制总像素有助于防御解压缩炸弹。
    max_vision_image_pixels: int = Field(
        default=16_000_000,
        ge=1,
        le=ABSOLUTE_MAX_VISION_IMAGE_PIXELS,
    )

    # Vision Provider单次上游调用的超时时间。
    #
    # 输入适配器本身不访问网络；
    # 该配置将在异步Provider阶段交给超时控制使用。
    vision_timeout_seconds: float = Field(
        default=60.0,
        gt=0.0,
        le=300.0,
        allow_inf_nan=False,
    )

    # ---------- 本地OCR基线策略 ----------

    # Tesseract语言可以使用单一模型，
    # 也可以使用eng+chi_sim这样的组合。
    ocr_language: str = Field(
        default="eng",
        min_length=1,
        max_length=100,
        pattern=r"^[a-z0-9_+.-]+$",
    )

    # Tesseract置信度范围是0到100。
    #
    # 低于该门槛的结果仍可保留，
    # 但必须标为low_confidence并要求人工复核。
    ocr_minimum_confidence: float = Field(
        default=70.0,
        ge=0.0,
        le=100.0,
        allow_inf_nan=False,
    )

    # pytesseract会把该值传给Tesseract子进程。
    # 超时后应终止本次OCR，而不是无限占用工作线程。
    ocr_timeout_seconds: float = Field(
        default=10.0,
        gt=0.0,
        le=60.0,
        allow_inf_nan=False,
    )

    @model_validator(mode="after")
    def validate_chunking_configuration(
        self,
    ) -> Self:
        """检查 Chunk 大小与重叠长度之间的关系。"""

        # overlap 必须小于 chunk_size。
        #
        # 如果二者相等，例如 800/800，
        # TextChunker 的下一次起点将无法前进，
        # 可能导致无限循环。
        if (
            self.rag_chunk_overlap
            >= self.rag_chunk_size
        ):
            raise ValueError(
                "RAG_CHUNK_OVERLAP "
                "必须小于 RAG_CHUNK_SIZE"
            )

        # mode="after" 的模型校验器必须返回
        # 校验完成后的模型实例。
        return self


@lru_cache
def get_settings() -> Settings:
    """返回当前进程缓存的 Settings 对象。"""

    # 第一次调用会读取环境变量和 .env；后续调用返回同一缓存对象。
    return Settings()
