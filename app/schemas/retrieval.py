"""RAG 文档分块、向量和检索结果的数据契约。"""

# Annotated 可以在真实类型上附加 Pydantic Field 约束。
from typing import Annotated, Self

# BaseModel 提供运行时数据校验和序列化；
# ConfigDict 配置整个模型的行为；
# Field 为字段声明长度、范围和格式限制。
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


# 一个 Embedding 向量本质上是浮点数列表。
#
# min_length=1 表示向量至少包含一个数值，
# 防止空向量进入相似度计算。
EmbeddingVector = Annotated[
    list[float],
    Field(min_length=1),
]

# 关键词检索返回的每一个命中词都必须是非空字符串。
#
# max_length=200防止异常长文本被错误地当成一个关键词，
# 也使后续日志和评测报告保持可控。
RetrievalTerm = Annotated[
    str,
    Field(
        min_length=1,
        max_length=200,
    ),
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


class KeywordRetrievedChunk(BaseModel):
    """关键词检索召回的Chunk、得分和可解释命中依据。"""

    model_config = ConfigDict(
        # 拒绝调用方传入没有声明的字段，
        # 防止keyword_scores等拼写错误被静默忽略。
        extra="forbid",

        # keyword_score不允许出现NaN或正负无穷，
        # 否则排序结果将失去确定性。
        allow_inf_nan=False,
    )

    # 保留完整DocumentChunk，
    # 使后续融合结果仍可追溯到文件、页码或章节。
    chunk: DocumentChunk

    # 关键词分数不是余弦相似度，
    # 因此不限制到[-1, 1]。
    #
    # 关键词检索只返回真正发生命中的Chunk，
    # 所以分数必须严格大于0。
    keyword_score: float = Field(
        gt=0.0,
        description="当前Chunk的关键词匹配分数",
    )

    # 关键词排名从1开始。
    #
    # RRF后续主要使用排名，而不是直接比较
    # 关键词分数和向量相似度。
    rank: int = Field(
        ge=1,
        description="当前Chunk在关键词结果中的排名",
    )

    # 精确命中的错误码或测试编号。
    #
    # 示例：
    # ("err-net-4001",)
    matched_identifiers: tuple[
        RetrievalTerm,
        ...,
    ] = ()

    # 精确命中的产品型号规范名称。
    #
    # 示例：
    # ("dataman-260",)
    matched_model_aliases: tuple[
        RetrievalTerm,
        ...,
    ] = ()

    # 精确命中的数值和单位。
    #
    # 示例：
    # ("1500ms", "-10°c")
    matched_numeric_terms: tuple[
        RetrievalTerm,
        ...,
    ] = ()

    # 普通英文词或中文字符词组的交集。
    #
    # 这些词的权重低于错误码、型号和数值，
    # 但能为没有结构化编号的问题提供候选。
    matched_lexical_terms: tuple[
        RetrievalTerm,
        ...,
    ] = ()

    @model_validator(mode="after")
    def validate_match_explanation(
        self,
    ) -> Self:
        """要求每条关键词结果至少说明一项命中依据。"""

        # any()只要发现一个非空元组就返回True。
        has_match_explanation = any(
            (
                self.matched_identifiers,
                self.matched_model_aliases,
                self.matched_numeric_terms,
                self.matched_lexical_terms,
            )
        )

        if not has_match_explanation:
            raise ValueError(
                "关键词检索结果必须至少包含一项命中依据"
            )

        # mode="after"的模型校验器必须返回
        # 已经校验完成的当前对象。
        return self


class HybridRetrievedChunk(BaseModel):
    """向量和关键词排名经过RRF融合后的可审计结果。"""

    model_config = ConfigDict(
        # 拒绝没有声明的字段，
        # 防止rrf_scores等拼写错误被静默忽略。
        extra="forbid",

        # 融合分数、余弦相似度和关键词分数
        # 都不能包含NaN或正负无穷。
        allow_inf_nan=False,
    )

    # 保存真实Chunk正文和完整来源信息。
    chunk: DocumentChunk

    # RRF分数只用于融合结果内部排序。
    #
    # 它不是余弦相似度、概率或回答置信度。
    # 每个融合候选至少来自一条检索路径，
    # 所以分数必须严格大于0。
    rrf_score: float = Field(
        gt=0.0,
        description="倒数排名融合分数",
    )

    # 当前Chunk在RRF融合结果中的最终排名。
    rank: int = Field(
        ge=1,
        description="当前Chunk在融合结果中的排名",
    )

    # 如果Chunk出现在向量候选列表中，
    # 记录它原来的向量排名。
    #
    # None表示它没有进入本次向量候选集合。
    vector_rank: int | None = Field(
        default=None,
        ge=1,
        description="融合前的向量检索排名",
    )

    # 保留真正的余弦相似度。
    #
    # 只有vector_rank不为None时才允许存在。
    vector_similarity: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description="融合前的向量余弦相似度",
    )

    # 如果Chunk出现在关键词候选列表中，
    # 记录它原来的关键词排名。
    #
    # None表示它没有进入本次关键词候选集合。
    keyword_rank: int | None = Field(
        default=None,
        ge=1,
        description="融合前的关键词检索排名",
    )

    # 保留关键词检索器计算的原始分数。
    #
    # 它只用于审计关键词排序，
    # RRF不会直接把它与余弦相似度相加。
    keyword_score: float | None = Field(
        default=None,
        gt=0.0,
        description="融合前的关键词匹配分数",
    )

    # 以下四个字段保留关键词命中解释。
    #
    # 如果当前Chunk只来自向量检索，
    # 它们应保持为空元组。
    matched_identifiers: tuple[
        RetrievalTerm,
        ...,
    ] = ()

    matched_model_aliases: tuple[
        RetrievalTerm,
        ...,
    ] = ()

    matched_numeric_terms: tuple[
        RetrievalTerm,
        ...,
    ] = ()

    matched_lexical_terms: tuple[
        RetrievalTerm,
        ...,
    ] = ()

    @model_validator(mode="after")
    def validate_retrieval_sources(
        self,
    ) -> Self:
        """校验两条检索路径的字段组合是否自洽。"""

        # 向量排名和向量相似度必须同时存在，
        # 或者同时为None。
        has_vector_rank = (
            self.vector_rank is not None
        )
        has_vector_similarity = (
            self.vector_similarity is not None
        )

        if (
            has_vector_rank
            != has_vector_similarity
        ):
            raise ValueError(
                "vector_rank与vector_similarity"
                "必须同时存在或同时为空"
            )

        # 关键词排名和关键词分数也必须成对出现。
        has_keyword_rank = (
            self.keyword_rank is not None
        )
        has_keyword_score = (
            self.keyword_score is not None
        )

        if (
            has_keyword_rank
            != has_keyword_score
        ):
            raise ValueError(
                "keyword_rank与keyword_score"
                "必须同时存在或同时为空"
            )

        # 一个融合结果必须至少来自一条检索路径。
        #
        # 两条路径都不存在时，
        # rrf_score就没有合法来源。
        if not (
            has_vector_rank
            or has_keyword_rank
        ):
            raise ValueError(
                "融合结果必须至少来自一条检索路径"
            )

        has_keyword_explanation = any(
            (
                self.matched_identifiers,
                self.matched_model_aliases,
                self.matched_numeric_terms,
                self.matched_lexical_terms,
            )
        )

        # 来自关键词检索的结果必须保留至少一项
        # 真实命中词，否则keyword_score无法审计。
        if (
            has_keyword_rank
            and not has_keyword_explanation
        ):
            raise ValueError(
                "关键词融合结果必须包含命中解释"
            )

        # 如果没有进入关键词候选列表，
        # 就不能携带来源不明的关键词命中解释。
        if (
            not has_keyword_rank
            and has_keyword_explanation
        ):
            raise ValueError(
                "没有关键词来源时不能包含关键词命中解释"
            )

        return self


    