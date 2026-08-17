"""检索评测问题、预期证据和评测报告的数据契约。"""

# datetime 表示评测报告的生成时间。
from datetime import datetime

# Literal 限制字段只能取指定字符串；
# Self 表示当前模型自身的类型。
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

# RetrievedChunk 表示 ChromaDB 返回的一条检索结果，
# 包含正文、来源、相似度和排名。
from app.schemas.retrieval import RetrievedChunk

class ExpectedEvidence(BaseModel):
    """一个问题预期召回的来源位置。"""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    source_file: str = Field(
        min_length=1,
        max_length=255,
        description="预期召回的来源文件名",
    )

    page_or_section: str = Field(
        min_length=1,
        max_length=500,
        description="预期召回的页码或章节",
    )


class GoldQuestion(BaseModel):
    """一条带标准答案和预期证据的检索评测问题。"""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    # 例如 q001、q020。
    question_id: str = Field(
        pattern=r"^q\d{3}$",
        description="评测问题的稳定编号",
    )

    question: str = Field(
        min_length=1,
        max_length=2000,
        description="交给检索系统的自然语言问题",
    )

    # Literal 会拒绝其他拼写，
    # 例如 single-hop 或 multihop。
    question_type: Literal[
        "single_hop",
        "multi_hop",
        "unanswerable",
    ]

    answerable: bool

    # 无答案问题必须使用 None。
    reference_answer: str | None = Field(
        default=None,
        min_length=1,
        description="人工标注的参考答案",
    )

    expected_evidence: list[ExpectedEvidence]

    tags: list[str] = Field(
        min_length=1,
    )

    notes: str = Field(
        min_length=1,
    )

    @model_validator(mode="after")
    def validate_answerability(self) -> Self:
        """检查问题类型、答案和证据之间是否一致。"""

        # unanswerable 问题必须明确不可回答，
        # 不能包含参考答案或预期证据。
        if self.question_type == "unanswerable":
            if self.answerable:
                raise ValueError(
                    "unanswerable 问题的 answerable 必须为 false"
                )

            if self.reference_answer is not None:
                raise ValueError(
                    "unanswerable 问题不能包含 reference_answer"
                )

            if self.expected_evidence:
                raise ValueError(
                    "unanswerable 问题不能包含 expected_evidence"
                )

            return self

        # single_hop 和 multi_hop 都属于可回答问题。
        if not self.answerable:
            raise ValueError(
                "可回答问题的 answerable 必须为 true"
            )

        if self.reference_answer is None:
            raise ValueError(
                "可回答问题必须包含 reference_answer"
            )

        if not self.expected_evidence:
            raise ValueError(
                "可回答问题必须包含 expected_evidence"
            )

        # 跨段落问题至少需要两项预期证据，
        # 否则无法验证是否真正完成多跳召回。
        if (
            self.question_type == "multi_hop"
            and len(self.expected_evidence) < 2
        ):
            raise ValueError(
                "multi_hop 问题至少需要两项预期证据"
            )

        return self


class RetrievalQuestionEvaluation(BaseModel):
    """一道问题的完整检索评测结果。"""

    model_config = ConfigDict(
        # 拒绝未声明字段，
        # 防止生成报告时因字段拼写错误而静默丢失数据。
        extra="forbid",
    )

    # 对应 GoldQuestion.question_id。
    question_id: str = Field(
        pattern=r"^q\d{3}$",
        description="评测问题的稳定编号",
    )

    # 保留原始问题，方便直接阅读 JSON 报告。
    question: str = Field(
        min_length=1,
        max_length=2000,
        description="发送给检索系统的问题",
    )

    question_type: Literal[
        "single_hop",
        "multi_hop",
        "unanswerable",
    ]

    # 表示标注数据认为这道题是否可以
    # 根据当前知识库回答。
    answerable: bool

    # 可回答问题对应的人工标注证据。
    #
    # 无答案问题的列表为空。
    expected_evidence: list[ExpectedEvidence]

    # ChromaDB 实际返回的 Top-K 结果。
    #
    # RetrievedChunk 中已经包含：
    # - Chunk 正文
    # - 文件名
    # - 页码或章节
    # - 相似度
    # - 排名
    retrieved_chunks: list[RetrievedChunk]

    # Top-1 中命中的预期证据数量。
    matched_evidence_at_1: int = Field(
        ge=0,
        description="Top-1 命中的预期证据数",
    )

    # Top-3 中命中的预期证据数量。
    matched_evidence_at_3: int = Field(
        ge=0,
        description="Top-3 命中的预期证据数",
    )

    # 是否在 Top-1 中找全了这道题的全部预期证据。
    #
    # 多跳问题通常有两项预期证据，
    # 因而 Top-1 一般不可能完整召回。
    fully_recalled_at_1: bool

    # 是否在 Top-3 中找全了全部预期证据。
    fully_recalled_at_3: bool

    # 第一名结果的相似度。
    #
    # 知识库为空时没有第一名，因此允许为 None。
    top_similarity: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description="Top-1 检索结果的余弦相似度",
    )

    # 只对无答案问题有意义。
    #
    # 如果无答案问题的最高相似度仍然达到接受阈值，
    # 系统可能错误地把无关 Chunk 当作证据。
    false_recall: bool = Field(
        description="无答案问题是否错误召回了可接受证据",
    )


class RetrievalMetrics(BaseModel):
    """整套评测问题的汇总指标。"""

    model_config = ConfigDict(
        extra="forbid",
    )

    # 可回答问题数量。
    answerable_question_count: int = Field(
        ge=0,
    )

    # 无答案问题数量。
    unanswerable_question_count: int = Field(
        ge=0,
    )

    # 所有可回答问题包含的预期证据总数。
    #
    # 单跳题一般贡献 1 项；
    # 多跳题一般贡献 2 项。
    expected_evidence_count: int = Field(
        ge=0,
    )

    # Top-1 和 Top-3 命中的证据总数。
    matched_evidence_at_1: int = Field(
        ge=0,
    )
    matched_evidence_at_3: int = Field(
        ge=0,
    )

    # 证据级 Recall：
    #
    # 命中的预期证据数 / 预期证据总数。
    recall_at_1: float = Field(
        ge=0.0,
        le=1.0,
    )
    recall_at_3: float = Field(
        ge=0.0,
        le=1.0,
    )

    # 在 Top-K 中找全全部证据的问题数量。
    #
    # 这能避免多跳问题只找到一半证据，
    # 但总体 Recall 看起来仍然不错。
    fully_recalled_at_1_count: int = Field(
        ge=0,
    )
    fully_recalled_at_3_count: int = Field(
        ge=0,
    )

    # 无答案问题中，有多少道题错误地超过相似度阈值。
    false_recall_count: int = Field(
        ge=0,
    )

    # false_recall_count / unanswerable_question_count。
    false_recall_rate: float = Field(
        ge=0.0,
        le=1.0,
    )


class RetrievalEvaluationReport(BaseModel):
    """一次完整的 Chroma 检索评测报告。"""

    model_config = ConfigDict(
        extra="forbid",
    )

    # 使用 datetime，而不是普通字符串，
    # 让 Pydantic 验证它确实是合法时间。
    generated_at: datetime

    # 记录评测使用的向量模型。
    #
    # 更换模型后，旧评测结果不能直接与新结果混为一谈。
    embedding_model: str = Field(
        min_length=1,
        max_length=200,
    )

    # 记录被评测的 Chroma Collection。
    collection_name: str = Field(
        min_length=1,
        max_length=200,
    )

    # 本次检索至少要取得前三名，
    # 因为报告需要计算 Recall@3。
    top_k: int = Field(
        ge=3,
    )

    # 判断无答案问题是否出现错误召回的相似度阈值。
    similarity_threshold: float = Field(
        ge=-1.0,
        le=1.0,
    )

    metrics: RetrievalMetrics

    # 保存每道问题的详细结果，
    # 便于从汇总指标继续追查失败原因。
    results: list[RetrievalQuestionEvaluation] = Field(
        min_length=1,
    )