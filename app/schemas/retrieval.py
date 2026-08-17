"""RAG 文档分块、向量和检索结果的数据契约。"""

# Annotated 可以在真实类型上附加 Pydantic Field 约束。
from typing import Annotated

# BaseModel 提供运行时数据校验和序列化；
# ConfigDict 配置整个模型的行为；
# Field 为字段声明长度、范围和格式限制。
from pydantic import BaseModel, ConfigDict, Field


# 一个 Embedding 向量本质上是浮点数列表。
#
# min_length=1 表示向量至少包含一个数值，
# 防止空向量进入相似度计算。
EmbeddingVector = Annotated[
    list[float],
    Field(min_length=1),
]

class SourceTextSegment(BaseModel):
    """解析器从原始文档中提取的一页或一个章节。"""

    model_config = ConfigDict(
        # 删除页码、章节名和正文首尾的空白。
        str_strip_whitespace=True,

        # 拒绝没有声明的字段，避免解析器字段拼写错误。
        extra="forbid",
    )

    # PDF 可以使用 page: 12；
    # Markdown/TXT 可以使用 section: 故障处理。
    page_or_section: str = Field(
        min_length=1,
        max_length=500,
        description="该文本段在原始文档中的稳定位置",
    )

    # 这是尚未执行 Chunk 切分的页面或章节全文。
    #
    # 这里不设置很小的 max_length，
    # 因为一个原始章节可能明显长于最终 Chunk。
    content: str = Field(
        min_length=1,
        description="从一页或一个章节中提取的原始文本",
    )

class DocumentChunk(BaseModel):
    """从原始文档中切分出的一段可追溯文本。"""

    model_config = ConfigDict(
        # 自动清理字符串首尾空格。
        str_strip_whitespace=True,

        # 拒绝模型中没有声明的字段，
        # 防止元数据字段拼写错误后被静默忽略。
        extra="forbid",
    )

    # 唯一标识一个 Chunk。
    #
    # 后续可以使用：
    # <document_id>:<chunk_index>
    #
    # 例如：
    # 7f21...a90:0003
    chunk_id: str = Field(
        min_length=1,
        max_length=200,
        description="Chunk 的唯一标识",
    )

    # 标识这个 Chunk 属于哪一份完整文档。
    #
    # 删除文档时，可以根据 document_id
    # 删除它产生的全部 Chunk 和向量。
    document_id: str = Field(
        min_length=1,
        max_length=128,
        description="原始文档的唯一标识",
    )

    # 只保存文件名，不保存服务器绝对路径。
    #
    # 这样可以避免在 API 引用中泄漏：
    # D:\\secret-project\\internal-manual.pdf
    source_file: str = Field(
        min_length=1,
        max_length=255,
        description="原始文档文件名",
    )

    # PDF 使用物理页码，Markdown/TXT 使用稳定章节。
    #
    # 示例：
    # page: 55
    # section: ERR-NAV-2001
    page_or_section: str = Field(
        min_length=1,
        max_length=500,
        description="Chunk 在原始文档中的稳定位置",
    )

    # Chunk 在当前文档中的顺序，从 0 开始。
    #
    # ge 是 greater than or equal 的缩写，
    # ge=0 表示必须大于或等于 0。
    chunk_index: int = Field(
        ge=0,
        description="Chunk 在文档中的零基索引",
    )

    # Chunk 正文的 SHA-256 十六进制摘要。
    #
    # pattern 使用正则表达式约束：
    # ^           字符串开头
    # [0-9a-f]    一个小写十六进制字符
    # {64}        必须正好出现 64 次
    # $           字符串结尾
    content_hash: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="Chunk 正文的 SHA-256",
    )

    # 真正用于 Embedding 和作为 RAG 证据的文本。
    content: str = Field(
        min_length=1,
        max_length=20_000,
        description="Chunk 的原始文本",
    )


class EmbeddedChunk(BaseModel):
    """已经生成 Embedding 的文档 Chunk。"""

    model_config = ConfigDict(
        # 拒绝意外字段。
        extra="forbid",

        # 向量中不允许出现 NaN、正无穷或负无穷。
        #
        # 这些特殊浮点数会污染余弦相似度排序。
        allow_inf_nan=False,
    )

    # 使用嵌套模型保存完整 Chunk。
    # Pydantic 会递归校验 DocumentChunk。
    chunk: DocumentChunk

    # 记录生成该向量的模型。
    #
    # 不同 Embedding 模型产生的向量不能混合比较，
    # 即使它们恰好具有相同维度。
    embedding_model: str = Field(
        min_length=1,
        max_length=200,
        description="生成向量的 Embedding 模型",
    )

    # 文档 Chunk 对应的向量。
    embedding: EmbeddingVector


class RetrievedChunk(BaseModel):
    """一次查询中召回的 Chunk 及其排序信息。"""

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
    )

    # 检索结果保留完整来源和正文。
    #
    # 不向结果中返回原始 Embedding，
    # 因为回答和引用不需要看到几千个浮点数。
    chunk: DocumentChunk

    # 当前使用余弦相似度。
    #
    # 理论范围是 -1 到 1：
    # 1  表示方向完全相同；
    # 0  表示方向正交；
    # -1 表示方向完全相反。
    similarity: float = Field(
        ge=-1.0,
        le=1.0,
        description="查询与 Chunk 的余弦相似度",
    )

    # 检索排名从 1 开始。
    #
    # 第一名 rank=1，比 Python 列表的零基索引
    # 更符合面向用户的 Top-1、Top-3 表达。
    rank: int = Field(
        ge=1,
        description="当前 Chunk 在检索结果中的排名",
    )