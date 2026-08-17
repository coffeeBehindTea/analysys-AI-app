"""知识库文档级数据契约。"""

# SupportedDocumentType 只能是：
# "pdf"、"md"、"txt"
from typing import Literal

# BaseModel 提供数据校验和序列化；
# ConfigDict 配置整个模型；
# Field 为单个字段添加范围、格式和说明。
from pydantic import BaseModel, ConfigDict, Field



# 当前知识库明确支持的文件类型。
#
# 使用 Literal 后，如果意外传入 "docx"，
# Pydantic 会立即产生 ValidationError。
SupportedDocumentType = Literal[
    "pdf",
    "md",
    "txt",
]


class DocumentRecord(BaseModel):
    """一份已经成功摄取到知识库的文档记录。"""

    model_config = ConfigDict(
        # 自动删除字符串首尾空白。
        str_strip_whitespace=True,

        # 拒绝没有声明的字段，
        # 防止字段拼写错误后被静默忽略。
        extra="forbid",
    )
    # 整份原始文件的 SHA-256。
    #
    # 相同内容即使使用不同文件名，
    # 也会产生相同 document_id。
    document_id: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="原始文件的 SHA-256 标识",
    )

    # 只保存文件名，不保存服务器绝对路径，
    # 防止 API 响应泄漏本机目录结构。
    source_file: str = Field(
        min_length=1,
        max_length=255,
        description="原始文件名",
    )

    # 不保存开头的点，例如使用 "pdf" 而不是 ".pdf"。
    file_type: SupportedDocumentType = Field(
        description="文档文件类型",
    )

    # 原始上传文件的字节数。
    #
    # ge 是 greater than or equal，
    # ge=1 表示至少包含一个字节。
    file_size_bytes: int = Field(
        ge=1,
        description="原始文件大小，单位为字节",
    )

    # PDF 表示有可提取文字的页面数量；
    # Markdown/TXT 表示 Loader 产生的章节数量。
    #
    # 它不一定等于 Chunk 数，因为一个长页面
    # 可能继续被切成多个 Chunk。
    page_or_section_count: int = Field(
        ge=1,
        description="文档中包含有效文本的位置数量",
    )

    # 这份文档最终写入向量库的 Chunk 数量。
    chunk_count: int = Field(
        ge=1,
        description="文档产生的 Chunk 数量",
    )

    # 记录向量由哪个模型生成。
    #
    # 不同模型产生的向量不能放在同一空间中比较，
    # 所以这个字段也属于重要的可追溯元数据。
    embedding_model: str = Field(
        min_length=1,
        max_length=200,
        description="文档向量使用的 Embedding 模型",
    )

    # 当前摄取是同步过程：
    # 只有全部 Chunk 成功写入后才产生 DocumentRecord，
    # 因而现阶段持久化记录只有 ready 状态。
    #
    # 以后如果改成后台异步摄取，
    # 可以扩展为 processing、ready、failed。
    status: Literal["ready"] = Field(
        default="ready",
        description="文档当前状态",
    )


class DocumentListResponse(BaseModel):
    """知识库文档列表接口的响应契约。"""

    model_config = ConfigDict(
        # 拒绝没有在响应契约中声明的字段，
        # 防止 Service 或 Router 意外泄漏内部信息。
        extra="forbid",
    )

    # list[DocumentRecord] 表示列表中的每一项
    # 都必须符合 DocumentRecord 数据契约。
    #
    # default_factory=list 会为每个模型实例创建独立的新列表，
    # 避免多个响应对象意外共享同一个可变列表。
    documents: list[DocumentRecord] = Field(
        default_factory=list,
        description="知识库中已经完成摄取的文档",
    )


class DocumentDeleteResponse(BaseModel):
    """成功删除知识库文档后的响应契约。"""

    model_config = ConfigDict(
        # 清理字符串首尾空白。
        str_strip_whitespace=True,

        # 拒绝响应中没有声明的字段。
        extra="forbid",
    )

    # 返回被删除文档的稳定标识，
    # 方便客户端确认删除的是哪个文档。
    document_id: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="被删除文档的 SHA-256 标识",
    )

    # Literal[True] 表示成功响应中该字段只能是 True。
    #
    # 文档不存在时返回 404，不返回 deleted=False。
    deleted: Literal[True] = Field(
        default=True,
        description="文档是否已经成功删除",
    )
