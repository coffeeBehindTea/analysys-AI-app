"""知识库证据约束问答接口的数据契约。"""

# Self 表示当前模型类自身的实例类型。
# 它主要用于 model_validator 返回值注解。
from typing import Annotated, Self

# BaseModel 提供校验和序列化；
# ConfigDict 配置整个模型；
# Field 为字段添加范围、长度和文档说明；
# model_validator 用于校验多个字段之间的关系。
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class KnowledgeQueryRequest(BaseModel):
    """知识库自然语言查询请求。"""

    model_config = ConfigDict(
        # 自动去掉问题首尾空白。
        str_strip_whitespace=True,

        # 拒绝未声明字段，避免字段拼写错误被忽略。
        extra="forbid",
    )

    question: str = Field(
        min_length=1,
        max_length=2_000,
        description="需要根据知识库证据回答的问题",
    )

    # top_k 表示最多召回多少个 Chunk。
    #
    # 默认使用评测指标中的 Top-3；
    # 上限10用于控制响应长度和Prompt大小。
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="最多召回的知识库Chunk数量",
    )


class KnowledgeCitation(BaseModel):
    """由真实检索结果构造的一条可追溯引用。"""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",

        # 禁止 similarity 出现 NaN 或无穷大。
        allow_inf_nan=False,
    )

    # 引用对应的唯一Chunk。
    chunk_id: str = Field(
        min_length=1,
        max_length=200,
        description="被引用Chunk的唯一标识",
    )

    # 引用所属原始文档的SHA-256。
    document_id: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="引用所属文档的SHA-256标识",
    )

    source_file: str = Field(
        min_length=1,
        max_length=255,
        description="引用所属原始文件名",
    )

    page_or_section: str = Field(
        min_length=1,
        max_length=500,
        description="引用在原始文档中的页码或章节",
    )

    # 保留零基Chunk顺序，方便工程定位和诊断。
    chunk_index: int = Field(
        ge=0,
        description="Chunk在文档中的零基索引",
    )

    # 检索排名面向用户，从1开始。
    rank: int = Field(
        ge=1,
        description="该Chunk在本次检索中的排名",
    )

    # 纯向量候选和进入向量路径的混合候选
    # 都具有真实余弦相似度。
    #
    # 关键词独占候选没有进入本次向量候选集合，
    # 因此必须使用None表达“没有该测量值”，
    # 不能伪造成0.0。
    similarity: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description=(
            "问题与该Chunk的余弦相似度；"
            "关键词独占候选为空"
        ),
    )

    # rrf_score是倒数排名融合分数。
    #
    # 它只解释混合检索的最终排序，
    # 不是余弦相似度、概率或答案置信度。
    #
    # 纯向量基线没有执行RRF，因此使用None。
    rrf_score: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "混合检索的RRF融合分数；"
            "纯向量引用为空"
        ),
    )

    # excerpt直接来自召回Chunk的正文。
    #
    # 它不能使用LLM自己生成或改写的文字，
    # 否则引用将无法证明回答依据。
    excerpt: str = Field(
        min_length=1,
        max_length=20_000,
        description="被引用Chunk的原始文本",
    )

    @model_validator(mode="after")
    def require_at_least_one_retrieval_score(
        self,
    ) -> Self:
        """引用必须保留至少一种真实检索分数。"""

        # similarity=0.0是合法余弦相似度，
        # 因此不能使用：
        #
        # if not self.similarity
        #
        # 必须使用is None准确区分：
        #
        # 0.0  → 有真实分数；
        # None → 没有该分数。
        if (
            self.similarity is None
            and self.rrf_score is None
        ):
            raise ValueError(
                "引用至少需要一种检索分数"
            )

        # mode="after"校验器必须返回
        # 已经完成检查的当前模型对象。
        return self


class KnowledgeQueryResponse(BaseModel):
    """知识库证据约束问答的成功响应。"""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        allow_inf_nan=False,
    )

    # 正常情况下是LLM根据证据生成的回答；
    # 拒答时是固定的证据不足说明。
    answer: str = Field(
        min_length=1,
        max_length=20_000,
        description="基于知识库证据生成的回答",
    )

    # default_factory=list为每个响应创建独立列表。
    citations: list[KnowledgeCitation] = Field(
        default_factory=list,
        description="由实际召回Chunk构造的引用",
    )

    # 这里只统计：
    # 查询Embedding + Chroma Top-K检索。
    #
    # 不包括LLM回答生成耗时，
    # 这样指标可以单独反映检索层性能。
    retrieval_ms: float = Field(
        ge=0.0,
        description="查询向量化与检索耗时，单位毫秒",
    )

    # True表示证据不足，未调用LLM。
    abstained: bool = Field(
        description="系统是否因证据不足而拒答",
    )

    @model_validator(mode="after")
    def validate_evidence_contract(
        self,
    ) -> Self:
        """检查拒答状态和引用之间的业务关系。"""

        # 任务要求：低于阈值时直接拒答，
        # 并返回空引用。
        if self.abstained and self.citations:
            raise ValueError(
                "拒答响应不能包含引用"
            )

        # 如果系统声称已经回答，
        # 就必须提供至少一条真实证据。
        if (
            not self.abstained
            and not self.citations
        ):
            raise ValueError(
                "非拒答响应必须包含至少一条引用"
            )

        # mode="after"校验器必须返回校验后的模型。
        return self


# LLM只能返回E1、E2、E3这类临时证据编号。
#
# E0无效，因为检索rank从1开始。
EvidenceId = Annotated[
    str,
    Field(
        pattern=r"^E[1-9][0-9]*$",
        description="本次查询中的临时证据编号",
    ),
]


class KnowledgeAnswerDraft(BaseModel):
    """LLM返回给编排Service的内部结构。"""

    model_config = ConfigDict(
        # 自动清理回答和证据编号的首尾空白。
        str_strip_whitespace=True,

        # 拒绝模型自行添加文件名、页码等字段。
        extra="forbid",
    )

    # 这里只允许LLM生成回答正文。
    answer: str = Field(
        min_length=1,
        max_length=20_000,
        description="只根据证据生成的回答正文",
    )

    # LLM只能选择本次Prompt中提供的临时证据编号。
    #
    # 正常回答时至少需要一条证据；
    # 二次拒答时必须为空列表。
    #
    # 因为是否允许空列表取决于abstained字段，
    # 所以这里不能继续使用min_length=1，
    # 而要交给下面的model_validator执行跨字段校验。
    used_evidence_ids: list[EvidenceId] = Field(
        max_length=10,
        description=(
            "实际用于回答的证据编号；"
            "拒答时必须为空列表"
        ),
    )

    # abstained表示LLM在阅读门控放行的候选后，
    # 是否仍然判断证据不足。
    #
    # default=False用于兼容现有正常回答构造代码。
    abstained: bool = Field(
        default=False,
        description=(
            "LLM是否因候选证据不足而拒答"
        ),
    )

    @field_validator("used_evidence_ids")
    @classmethod
    def evidence_ids_must_be_unique(
        cls,
        evidence_ids: list[str],
    ) -> list[str]:
        """同一证据编号不能重复出现。"""

        if len(set(evidence_ids)) != len(
            evidence_ids
        ):
            raise ValueError(
                "used_evidence_ids不能包含重复编号"
            )

        return evidence_ids

    @model_validator(mode="after")
    def validate_abstention_contract(
        self,
    ) -> Self:
        """检查回答状态与证据编号之间的关系。"""

        # 拒答表示没有证据足以支持答案，
        # 因此不能同时声称使用了某条证据。
        if (
            self.abstained
            and self.used_evidence_ids
        ):
            raise ValueError(
                "拒答草稿不能选择证据"
            )

        # 非拒答表示模型生成了正式答案，
        # 因此必须至少选择一条真实候选证据。
        if (
            not self.abstained
            and not self.used_evidence_ids
        ):
            raise ValueError(
                "非拒答草稿必须选择至少一条证据"
            )

        # mode="after"校验器必须返回当前模型实例。
        return self
