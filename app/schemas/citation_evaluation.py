"""最终RAG回答的引用质量评测数据契约。"""

# Self表示当前Pydantic模型自身的实例类型，
# 用于model_validator的返回类型。
from typing import Self

# BaseModel提供数据校验和序列化；
# ConfigDict配置整个模型；
# Field为字段添加范围和说明；
# model_validator校验多个字段之间的关系。
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

# GoldQuestion保存人工标注的问题、
# 可回答状态、参考答案和预期证据。
from app.schemas.evaluation import GoldQuestion

# KnowledgeQueryResponse是知识库查询接口
# 实际返回的结构。
from app.schemas.knowledge_query import (
    KnowledgeQueryResponse,
)

# CandidateStrategyParameters保存完整候选检索参数：
#
# 1. 策略名称；
# 2. 最终Top-K；
# 3. 每条检索路径的候选深度；
# 4. RRF排名常数；
# 5. 查询改写版本。
from app.schemas.retrieval_strategy import (
    CandidateStrategyParameters,
)


class CitationQuestionEvaluation(BaseModel):
    """一道Gold Question的最终RAG引用评测结果。"""

    model_config = ConfigDict(
        # 拒绝未声明字段，
        # 防止报告字段拼写错误被静默忽略。
        extra="forbid",

        # 禁止浮点字段出现NaN或无穷大。
        allow_inf_nan=False,
    )

    # 保存完整Gold Question，
    # 便于从报告追溯问题、参考答案和预期证据。
    gold_question: GoldQuestion

    # 保存知识库API的真实业务响应。
    response: KnowledgeQueryResponse

    # API一共返回了多少条引用。
    returned_citation_count: int = Field(
        ge=0,
        description="本题实际返回的引用数量",
    )

    # 返回引用中，有多少条能匹配任意一项
    # Gold expected_evidence。
    correct_citation_count: int = Field(
        ge=0,
        description="本题返回引用中的正确引用数量",
    )

    # 该题Gold数据中包含多少项预期证据。
    expected_evidence_count: int = Field(
        ge=0,
        description="本题人工标注的预期证据数量",
    )

    # 有多少项不同的预期证据，
    # 至少被一条返回引用命中。
    #
    # 两个重叠Chunk命中同一个预期位置时，
    # 只能算覆盖一项预期证据。
    matched_expected_evidence_count: int = Field(
        ge=0,
        description="本题被引用覆盖的不同预期证据数量",
    )

    # 正确引用数 / 返回引用总数。
    #
    # 没有返回引用时分母为0，
    # 此时使用None表示该题无法计算引用正确率。
    citation_correctness: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="本题返回引用的正确率",
    )

    # 命中的不同预期证据数 / 预期证据总数。
    #
    # 无答案问题没有预期证据，
    # 此时使用None表示不适用。
    citation_coverage: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="本题预期证据的引用覆盖率",
    )

    # True表示：
    #
    # 1. Gold认为问题可以回答；
    # 2. API没有拒答；
    # 3. 至少返回一条引用；
    # 4. 所有返回引用都正确；
    # 5. 找全了本题全部预期证据。
    fully_grounded: bool = Field(
        description="本题是否完成了完整且无错误的证据回答",
    )

    # True表示：
    #
    # 1. Gold认为问题无答案；
    # 2. API返回abstained=true；
    # 3. API没有返回任何引用。
    correctly_abstained: bool = Field(
        description="本题是否正确执行了无答案拒答",
    )

    @model_validator(mode="after")
    def validate_derived_fields(
        self,
    ) -> Self:
        """验证计数、比例和状态是否相互一致。"""

        returned_count = len(
            self.response.citations
        )
        expected_count = len(
            self.gold_question.expected_evidence
        )

        # returned_citation_count必须真实反映
        # response.citations中的项目数量。
        if (
            self.returned_citation_count
            != returned_count
        ):
            raise ValueError(
                "returned_citation_count"
                "与响应引用数量不一致"
            )

        # expected_evidence_count必须真实反映
        # Gold Question中的预期证据数量。
        if (
            self.expected_evidence_count
            != expected_count
        ):
            raise ValueError(
                "expected_evidence_count"
                "与Gold预期证据数量不一致"
            )

        # 正确引用不可能多于全部返回引用。
        if (
            self.correct_citation_count
            > returned_count
        ):
            raise ValueError(
                "correct_citation_count"
                "不能大于返回引用数量"
            )

        # 已覆盖的预期证据不可能多于
        # Gold中存在的预期证据。
        if (
            self.matched_expected_evidence_count
            > expected_count
        ):
            raise ValueError(
                "matched_expected_evidence_count"
                "不能大于预期证据数量"
            )

        expected_correctness = (
            self.correct_citation_count
            / returned_count
            if returned_count
            else None
        )

        # 没有返回引用时，引用正确率没有分母，
        # 因此必须使用None，而不是0或1。
        if expected_correctness is None:
            if self.citation_correctness is not None:
                raise ValueError(
                    "没有返回引用时"
                    "citation_correctness必须为None"
                )
        else:
            if self.citation_correctness is None:
                raise ValueError(
                    "存在返回引用时"
                    "citation_correctness不能为空"
                )

            # 浮点计算可能产生极小舍入差，
            # 因此不直接使用==比较。
            if abs(
                self.citation_correctness
                - expected_correctness
            ) > 1e-12:
                raise ValueError(
                    "citation_correctness"
                    "与引用计数不一致"
                )

        expected_coverage = (
            self.matched_expected_evidence_count
            / expected_count
            if expected_count
            else None
        )

        # 无答案问题没有预期证据，
        # 所以引用覆盖率不适用。
        if expected_coverage is None:
            if self.citation_coverage is not None:
                raise ValueError(
                    "没有预期证据时"
                    "citation_coverage必须为None"
                )
        else:
            if self.citation_coverage is None:
                raise ValueError(
                    "存在预期证据时"
                    "citation_coverage不能为空"
                )

            if abs(
                self.citation_coverage
                - expected_coverage
            ) > 1e-12:
                raise ValueError(
                    "citation_coverage"
                    "与证据计数不一致"
                )

        expected_fully_grounded = (
            self.gold_question.answerable
            and not self.response.abstained
            and returned_count > 0
            and (
                self.correct_citation_count
                == returned_count
            )
            and (
                self.matched_expected_evidence_count
                == expected_count
            )
        )

        if (
            self.fully_grounded
            != expected_fully_grounded
        ):
            raise ValueError(
                "fully_grounded"
                "与回答和引用状态不一致"
            )

        expected_correctly_abstained = (
            not self.gold_question.answerable
            and self.response.abstained
            and returned_count == 0
        )

        if (
            self.correctly_abstained
            != expected_correctly_abstained
        ):
            raise ValueError(
                "correctly_abstained"
                "与无答案拒答状态不一致"
            )

        return self


class CitationMetrics(BaseModel):
    """整套Gold Question的引用评测汇总指标。"""

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
    )

    total_question_count: int = Field(
        ge=1,
        description="本次评测问题总数",
    )

    answerable_question_count: int = Field(
        ge=0,
        description="Gold中可回答的问题数",
    )

    unanswerable_question_count: int = Field(
        ge=0,
        description="Gold中无答案的问题数",
    )

    # 可回答问题中，API实际给出答案而没有拒答的数量。
    answered_answerable_count: int = Field(
        ge=0,
        description="实际回答的可回答问题数",
    )

    # 可回答问题中，回答且引用完全符合Gold的数量。
    fully_grounded_answer_count: int = Field(
        ge=0,
        description="完整且无错误引用的可回答问题数",
    )

    # 无答案问题中正确返回拒答的数量。
    correctly_abstained_unanswerable_count: int = Field(
        ge=0,
        description="正确拒答的无答案问题数",
    )

    returned_citation_count: int = Field(
        ge=0,
        description="全部问题返回的引用总数",
    )

    correct_citation_count: int = Field(
        ge=0,
        description="全部返回引用中的正确引用数",
    )

    expected_evidence_count: int = Field(
        ge=0,
        description="全部可回答问题的预期证据总数",
    )

    matched_expected_evidence_count: int = Field(
        ge=0,
        description="被引用覆盖的不同预期证据总数",
    )

    # 全局正确引用数 / 全局返回引用数。
    #
    # 如果完全没有返回引用，统一记为0.0，
    # 同时结合answerable_response_rate解释。
    citation_correctness: float = Field(
        ge=0.0,
        le=1.0,
        description="全部返回引用的正确率",
    )

    # 全局命中预期证据数 / 全局预期证据数。
    citation_coverage: float = Field(
        ge=0.0,
        le=1.0,
        description="全部预期证据的引用覆盖率",
    )

    # answered_answerable_count /
    # answerable_question_count。
    answerable_response_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="可回答问题的实际回答率",
    )

    # fully_grounded_answer_count /
    # answerable_question_count。
    fully_grounded_answer_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="可回答问题的完整证据回答率",
    )

    # correctly_abstained_unanswerable_count /
    # unanswerable_question_count。
    unanswerable_abstention_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="无答案问题的正确拒答率",
    )

    @model_validator(mode="after")
    def validate_metric_relations(
        self,
    ) -> Self:
        """验证汇总计数和比例之间的关系。"""

        if (
            self.total_question_count
            != (
                self.answerable_question_count
                + self.unanswerable_question_count
            )
        ):
            raise ValueError(
                "问题总数与可回答、无答案数量不一致"
            )

        if (
            self.answered_answerable_count
            > self.answerable_question_count
        ):
            raise ValueError(
                "实际回答数不能大于可回答问题数"
            )

        if (
            self.fully_grounded_answer_count
            > self.answered_answerable_count
        ):
            raise ValueError(
                "完整证据回答数不能大于实际回答数"
            )

        if (
            self.correctly_abstained_unanswerable_count
            > self.unanswerable_question_count
        ):
            raise ValueError(
                "正确拒答数不能大于无答案问题数"
            )

        if (
            self.correct_citation_count
            > self.returned_citation_count
        ):
            raise ValueError(
                "正确引用数不能大于返回引用总数"
            )

        if (
            self.matched_expected_evidence_count
            > self.expected_evidence_count
        ):
            raise ValueError(
                "命中证据数不能大于预期证据总数"
            )

        expected_correctness = (
            self.correct_citation_count
            / self.returned_citation_count
            if self.returned_citation_count
            else 0.0
        )

        expected_coverage = (
            self.matched_expected_evidence_count
            / self.expected_evidence_count
            if self.expected_evidence_count
            else 0.0
        )

        expected_response_rate = (
            self.answered_answerable_count
            / self.answerable_question_count
            if self.answerable_question_count
            else 0.0
        )

        expected_grounded_rate = (
            self.fully_grounded_answer_count
            / self.answerable_question_count
            if self.answerable_question_count
            else 0.0
        )

        expected_abstention_rate = (
            self.correctly_abstained_unanswerable_count
            / self.unanswerable_question_count
            if self.unanswerable_question_count
            else 0.0
        )

        expected_rates = (
            (
                "citation_correctness",
                self.citation_correctness,
                expected_correctness,
            ),
            (
                "citation_coverage",
                self.citation_coverage,
                expected_coverage,
            ),
            (
                "answerable_response_rate",
                self.answerable_response_rate,
                expected_response_rate,
            ),
            (
                "fully_grounded_answer_rate",
                self.fully_grounded_answer_rate,
                expected_grounded_rate,
            ),
            (
                "unanswerable_abstention_rate",
                self.unanswerable_abstention_rate,
                expected_abstention_rate,
            ),
        )

        for (
            field_name,
            actual_rate,
            expected_rate,
        ) in expected_rates:
            if abs(
                actual_rate - expected_rate
            ) > 1e-12:
                raise ValueError(
                    f"{field_name}与对应计数不一致"
                )

        return self


class CitationEvaluationReport(BaseModel):
    """一次完整的RAG引用质量评测报告。"""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        allow_inf_nan=False,
    )

    # 记录被评测的HTTP接口，
    # 使报告可以追溯到实际调用入口。
    api_url: str = Field(
        min_length=1,
        max_length=2_000,
    )

    llm_model: str = Field(
        min_length=1,
        max_length=200,
    )

    embedding_model: str = Field(
        min_length=1,
        max_length=200,
    )

    collection_name: str = Field(
        min_length=1,
        max_length=200,
    )

    # top_k表示每次在线API请求最终使用的
    # 最大候选证据数量。
    top_k: int = Field(
        ge=1,
        le=10,
    )

    # 新版报告保存完整候选检索策略参数。
    #
    # None用于兼容Week 2已经生成的旧报告：
    # 旧报告没有这个字段，只记录单一相似度阈值。
    retrieval_parameters: (
        CandidateStrategyParameters | None
    ) = Field(
        default=None,
        description=(
            "候选检索策略和完整参数；"
            "旧版纯向量报告可以为空"
        ),
    )

    # 纯向量基线使用单一余弦相似度阈值。
    #
    # 混合门控综合错误码、型号、关键词、
    # 向量相似度和双路命中，因此没有一个
    # 可以代表整个门控策略的全局阈值。
    similarity_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "纯向量报告使用的相似度拒答阈值；"
            "混合门控报告为空"
        ),
    )

    # 门控版本说明在线回答使用了哪一套
    # 组合证据放行规则。
    #
    # 纯向量基线没有HybridEvidenceGate，
    # 所以该字段必须为空。
    gate_version: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
        description=(
            "混合证据门控策略版本；"
            "纯向量报告为空"
        ),
    )

    metrics: CitationMetrics

    results: list[
        CitationQuestionEvaluation
    ] = Field(
        min_length=1,
    )

    @model_validator(mode="after")
    def validate_report_contract(
        self,
    ) -> Self:
        """校验逐题数量和检索策略元数据组合。"""

        # 报告逐题结果数量必须与汇总指标一致。
        if (
            len(self.results)
            != self.metrics.total_question_count
        ):
            raise ValueError(
                "逐题结果数量与汇总问题数量不一致"
            )

        parameters = self.retrieval_parameters

        # 没有retrieval_parameters表示这是
        # Week 2格式的旧纯向量报告。
        #
        # 为了保持旧报告和旧实验可解析，
        # 这里继续允许这种格式，
        # 但必须提供原来的相似度阈值。
        if parameters is None:
            if self.similarity_threshold is None:
                raise ValueError(
                    "旧版纯向量报告必须设置"
                    "similarity_threshold"
                )

            if self.gate_version is not None:
                raise ValueError(
                    "未声明混合检索参数时"
                    "不能设置gate_version"
                )

            return self

        # 顶层top_k表示真实HTTP请求参数；
        # 嵌套参数中的top_k表示候选策略参数。
        #
        # 二者不同会导致报告无法复现。
        if parameters.top_k != self.top_k:
            raise ValueError(
                "retrieval_parameters.top_k"
                "必须与报告top_k一致"
            )

        if parameters.strategy == "vector_baseline":
            # 新格式也允许显式声明纯向量基线。
            #
            # 纯向量基线必须记录实际拒答阈值。
            if self.similarity_threshold is None:
                raise ValueError(
                    "纯向量报告必须设置"
                    "similarity_threshold"
                )

            # 纯向量Service没有组合证据门控版本。
            if self.gate_version is not None:
                raise ValueError(
                    "纯向量报告不能设置gate_version"
                )

            return self

        # 到这里，策略只能是：
        #
        # hybrid_rrf
        # 或
        # hybrid_rrf_rewrite
        #
        # 这两种策略都不能被描述成
        # 一个单一相似度阈值门控。
        if self.similarity_threshold is not None:
            raise ValueError(
                "混合检索报告不能设置"
                "similarity_threshold"
            )

        # 混合候选策略只负责召回和排序；
        # 是否进入LLM由HybridEvidenceGate决定。
        #
        # 因此最终在线回答报告必须记录门控版本。
        if self.gate_version is None:
            raise ValueError(
                "混合检索报告必须设置gate_version"
            )

        return self

        return self
