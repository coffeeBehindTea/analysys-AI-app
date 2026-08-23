"""候选检索策略参数、逐题结果和汇总报告契约。

本模块只描述候选召回和排序结果。

它不把RRF分数解释成余弦相似度，
也不负责决定是否调用LLM。
门控和在线回答指标将在后续独立评测。
"""

# isclose()用于比较根据整数计数计算出的浮点Recall。
#
# 不直接使用==，避免某些除法结果受到
# 浮点数二进制表示误差影响。
from math import isclose

# Literal限制字符串只能从明确集合中选择；
# Self表示当前Pydantic模型自身的类型。
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from app.schemas.evaluation import (
    ExpectedEvidence,
)
from app.schemas.retrieval import (
    HybridRetrievedChunk,
    RetrievedChunk,
)


# 四种候选策略的稳定名称。
#
# 新策略不能继续沿用hybrid_rrf_rewrite，
# 否则报告无法区分“只有查询改写”和
# “查询改写后又执行头部保留重排”。
RetrievalStrategyName = Literal[
    "vector_baseline",
    "hybrid_rrf",
    "hybrid_rrf_rewrite",
    "hybrid_rrf_rewrite_rerank",
]


# 纯向量和混合检索结果具有不同的审计字段。
#
# 使用联合类型保留各自的真实数据，
# 不把rrf_score伪装成similarity。
StrategyRetrievedChunk = (
    RetrievedChunk
    | HybridRetrievedChunk
)


class HeadPreservingRerankParameters(
    BaseModel
):
    """一次头部保留重排实验使用的可审计参数。"""

    model_config = ConfigDict(
        # 清理version字符串首尾空白。
        str_strip_whitespace=True,

        # 拒绝未声明字段，防止参数拼写错误被静默忽略。
        extra="forbid",

        # 实验参数创建后不可修改，
        # 避免执行中途与最终报告记录不一致。
        frozen=True,
    )

    # 重排规则的稳定版本。
    #
    # 修改保护条件、替换顺序或默认深度时，
    # 必须产生新版本。
    version: str = Field(
        min_length=1,
        max_length=100,
        description="头部保留重排规则版本",
    )

    # 需要保留的原RRF头部深度。
    fused_head_k: int = Field(
        ge=0,
        description="需要保护的RRF融合头部深度",
    )

    # 需要保留的向量检索头部深度。
    vector_head_k: int = Field(
        ge=0,
        description="需要保护的向量头部深度",
    )

    # 需要保留的关键词检索头部深度。
    keyword_head_k: int = Field(
        ge=0,
        description="需要保护的关键词头部深度",
    )

    @model_validator(
        mode="after"
    )
    def require_at_least_one_head(
        self,
    ) -> Self:
        """重排实验必须实际保护至少一类头部。"""

        if (
            self.fused_head_k == 0
            and self.vector_head_k == 0
            and self.keyword_head_k == 0
        ):
            raise ValueError(
                "头部保留重排至少保留一个检索头部"
            )

        return self


class CandidateStrategyParameters(
    BaseModel
):
    """一次候选检索实验使用的固定策略参数。"""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    strategy: RetrievalStrategyName

    # 报告至少需要计算Recall@3，
    # 所以最终结果数量必须不小于3。
    top_k: int = Field(
        ge=3,
        description="最终用于评测的候选数量",
    )

    # 纯向量基线直接取top_k，
    # 因而不需要candidate_k。
    #
    # 混合检索需要先从两条路径分别取候选，
    # 再融合成最终top_k。
    candidate_k: int | None = Field(
        default=None,
        ge=1,
        description="每条召回路径的候选深度",
    )

    # 只有RRF策略需要排名常数。
    rank_constant: int | None = Field(
        default=None,
        ge=1,
        description="RRF排名平滑常数",
    )

    # 查询改写策略必须记录版本。
    #
    # 后面即使修改一条中英文映射，
    # 旧报告仍能说明自己使用的是哪个版本。
    rewrite_version: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
        description="确定性查询改写规则版本",
    )

    # 只有包含头部保留重排的策略才能设置该字段。
    #
    # 使用嵌套模型集中保存版本和三个保护深度，
    # 避免多个零散字段与策略名称不一致。
    rerank_parameters: (
        HeadPreservingRerankParameters | None
    ) = Field(
        default=None,
        description="可选头部保留重排参数",

        # 字段为None时不写入model_dump()和JSON。
        #
        # 这样旧策略报告不会增加无意义的：
        #
        # "rerank_parameters": null
        exclude_if=lambda value: value is None,
    )

    @model_validator(
        mode="after"
    )
    def validate_strategy_parameters(
        self,
    ) -> Self:
        """检查策略名称和可选参数组合是否一致。"""

        if self.strategy == "vector_baseline":
            # 纯向量基线不使用候选扩展、RRF、
            # 查询改写或头部保留重排。
            if self.candidate_k is not None:
                raise ValueError(
                    "vector_baseline不能设置candidate_k"
                )

            if self.rank_constant is not None:
                raise ValueError(
                    "vector_baseline不能设置rank_constant"
                )

            if self.rewrite_version is not None:
                raise ValueError(
                    "vector_baseline不能设置rewrite_version"
                )

            if self.rerank_parameters is not None:
                raise ValueError(
                    "vector_baseline不能设置"
                    "rerank_parameters"
                )

            return self

        # 所有混合策略都需要两路候选深度和RRF常数。
        if self.candidate_k is None:
            raise ValueError(
                "混合检索必须设置candidate_k"
            )

        if self.rank_constant is None:
            raise ValueError(
                "混合检索必须设置rank_constant"
            )

        if self.top_k > self.candidate_k:
            raise ValueError(
                "top_k不能大于candidate_k"
            )

        if self.strategy == "hybrid_rrf":
            # 普通混合检索既不改写，也不重排。
            if self.rewrite_version is not None:
                raise ValueError(
                    "hybrid_rrf不能设置rewrite_version"
                )

            if self.rerank_parameters is not None:
                raise ValueError(
                    "hybrid_rrf不能设置"
                    "rerank_parameters"
                )

            return self

        # 剩余两种策略都包含查询改写。
        if self.rewrite_version is None:
            raise ValueError(
                f"{self.strategy}"
                "必须设置rewrite_version"
            )

        if (
            self.strategy
            == "hybrid_rrf_rewrite"
        ):
            # 原查询改写策略不能偷偷携带重排参数，
            # 否则它会与新重排策略无法区分。
            if self.rerank_parameters is not None:
                raise ValueError(
                    "hybrid_rrf_rewrite不能设置"
                    "rerank_parameters"
                )

            return self

        # Literal已经限制了全部策略名称，
        # 所以执行到这里时只能是：
        #
        # hybrid_rrf_rewrite_rerank
        if self.rerank_parameters is None:
            raise ValueError(
                "hybrid_rrf_rewrite_rerank"
                "必须设置rerank_parameters"
            )

        return self


class CandidateQuestionEvaluation(
    BaseModel
):
    """一道Gold问题在一种候选策略下的评测结果。"""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    question_id: str = Field(
        pattern=r"^q\d{3}$",
        description="Gold问题稳定编号",
    )

    question: str = Field(
        min_length=1,
        max_length=2000,
    )

    question_type: Literal[
        "single_hop",
        "multi_hop",
        "unanswerable",
    ]

    answerable: bool

    expected_evidence: list[
        ExpectedEvidence
    ]

    # 可以保存纯向量结果，
    # 也可以保存混合检索结果。
    retrieved_chunks: list[
        StrategyRetrievedChunk
    ]

    matched_evidence_at_1: int = Field(
        ge=0,
    )

    matched_evidence_at_3: int = Field(
        ge=0,
    )

    fully_recalled_at_1: bool
    fully_recalled_at_3: bool

    @model_validator(mode="after")
    def validate_question_result(
        self,
    ) -> Self:
        """校验排名、答案类型和命中数量是否自洽。"""

        ranks = [
            result.rank
            for result in self.retrieved_chunks
        ]

        expected_ranks = list(
            range(
                1,
                len(self.retrieved_chunks) + 1,
            )
        )

        # 报告必须直接保存最终排序，
        # 不能保存3、1、2之类的无序结果。
        if ranks != expected_ranks:
            raise ValueError(
                "retrieved_chunks必须按连续rank排序"
            )

        expected_count = len(
            self.expected_evidence
        )

        # question_type和answerable必须表达同一含义。
        #
        # 不能出现：
        #
        # question_type="unanswerable"
        # answerable=True
        if (
            self.question_type == "unanswerable"
            and self.answerable
        ):
            raise ValueError(
                "unanswerable问题的"
                "answerable必须为false"
            )

        # single_hop和multi_hop都属于
        # 当前知识库可以回答的问题。
        if (
            self.question_type != "unanswerable"
            and not self.answerable
        ):
            raise ValueError(
                "可回答问题的answerable必须为true"
            )

        # 多跳问题必须至少标注两项预期证据。
        #
        # 否则无法验证系统是否真正完成了
        # 跨Chunk或跨文档召回。
        if (
            self.question_type == "multi_hop"
            and expected_count < 2
        ):
            raise ValueError(
                "multi_hop问题至少需要两项预期证据"
            )

        if self.answerable:
            if expected_count == 0:
                raise ValueError(
                    "可回答问题必须包含预期证据"
                )

            if (
                self.matched_evidence_at_1
                > expected_count
            ):
                raise ValueError(
                    "Top-1命中数不能超过预期证据数"
                )

            if (
                self.matched_evidence_at_3
                > expected_count
            ):
                raise ValueError(
                    "Top-3命中数不能超过预期证据数"
                )

            if (
                self.matched_evidence_at_1
                > self.matched_evidence_at_3
            ):
                raise ValueError(
                    "Top-1命中数不能大于Top-3命中数"
                )

            expected_full_at_1 = (
                self.matched_evidence_at_1
                == expected_count
            )
            expected_full_at_3 = (
                self.matched_evidence_at_3
                == expected_count
            )

            if (
                self.fully_recalled_at_1
                != expected_full_at_1
            ):
                raise ValueError(
                    "fully_recalled_at_1"
                    "与命中数量不一致"
                )

            if (
                self.fully_recalled_at_3
                != expected_full_at_3
            ):
                raise ValueError(
                    "fully_recalled_at_3"
                    "与命中数量不一致"
                )

            return self

        # 无答案问题没有Gold证据，
        # 不能利用0 == 0把它标成完整召回。
        if self.expected_evidence:
            raise ValueError(
                "无答案问题不能包含预期证据"
            )

        if (
            self.matched_evidence_at_1 != 0
            or self.matched_evidence_at_3 != 0
        ):
            raise ValueError(
                "无答案问题的证据命中数必须为0"
            )

        if (
            self.fully_recalled_at_1
            or self.fully_recalled_at_3
        ):
            raise ValueError(
                "无答案问题不能标记为完整召回"
            )

        return self


class CandidateStrategyMetrics(
    BaseModel
):
    """一种候选策略在整套Gold上的汇总指标。"""

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
    )

    answerable_question_count: int = Field(
        ge=0,
    )

    unanswerable_question_count: int = Field(
        ge=0,
    )

    expected_evidence_count: int = Field(
        ge=0,
    )

    matched_evidence_at_1: int = Field(
        ge=0,
    )

    matched_evidence_at_3: int = Field(
        ge=0,
    )

    recall_at_1: float = Field(
        ge=0.0,
        le=1.0,
    )

    recall_at_3: float = Field(
        ge=0.0,
        le=1.0,
    )

    fully_recalled_at_1_count: int = Field(
        ge=0,
    )

    fully_recalled_at_3_count: int = Field(
        ge=0,
    )

    @model_validator(mode="after")
    def validate_metric_counts(
        self,
    ) -> Self:
        """检查汇总计数和Recall计算是否一致。"""

        if (
            self.matched_evidence_at_1
            > self.expected_evidence_count
            or self.matched_evidence_at_3
            > self.expected_evidence_count
        ):
            raise ValueError(
                "证据命中数不能超过预期证据总数"
            )

        # Top-1是Top-3的子集，
        # 所以Top-1命中数量不能更多。
        if (
            self.matched_evidence_at_1
            > self.matched_evidence_at_3
        ):
            raise ValueError(
                "Top-1命中数不能大于Top-3命中数"
            )

        if (
            self.fully_recalled_at_1_count
            > self.answerable_question_count
            or self.fully_recalled_at_3_count
            > self.answerable_question_count
        ):
            raise ValueError(
                "完整召回题数不能超过可回答题数"
            )

        # 在Top-1中完整召回的问题，
        # 到Top-3时也必须仍然完整召回。
        if (
            self.fully_recalled_at_1_count
            > self.fully_recalled_at_3_count
        ):
            raise ValueError(
                "Top-1完整召回题数不能大于"
                "Top-3完整召回题数"
            )

        expected_recall_at_1 = (
            self.matched_evidence_at_1
            / self.expected_evidence_count
            if self.expected_evidence_count
            else 0.0
        )

        expected_recall_at_3 = (
            self.matched_evidence_at_3
            / self.expected_evidence_count
            if self.expected_evidence_count
            else 0.0
        )

        # rel_tol=0表示不使用相对误差；
        # abs_tol允许极小的浮点表示误差。
        if not isclose(
            self.recall_at_1,
            expected_recall_at_1,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "recall_at_1与证据命中数量不一致"
            )

        if not isclose(
            self.recall_at_3,
            expected_recall_at_3,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "recall_at_3与证据命中数量不一致"
            )

        return self


class CandidateStrategyReport(
    BaseModel
):
    """一种候选检索策略的完整离线报告。"""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    embedding_model: str = Field(
        min_length=1,
        max_length=200,
    )

    collection_name: str = Field(
        min_length=1,
        max_length=512,
    )

    parameters: CandidateStrategyParameters

    metrics: CandidateStrategyMetrics

    results: list[
        CandidateQuestionEvaluation
    ] = Field(
        min_length=1,
    )

    @model_validator(mode="after")
    def validate_report(
        self,
    ) -> Self:
        """检查策略、逐题结果和汇总指标是否一致。"""

        question_ids = [
            result.question_id
            for result in self.results
        ]

        if (
            len(question_ids)
            != len(set(question_ids))
        ):
            raise ValueError(
                "报告不能包含重复question_id"
            )

        # 每道题最多只能保存本策略声明的top_k项。
        if any(
            len(result.retrieved_chunks)
            > self.parameters.top_k
            for result in self.results
        ):
            raise ValueError(
                "逐题候选数量不能超过top_k"
            )

        # 报告声明的策略必须与内部结果类型一致。
        #
        # 纯向量报告只能保存RetrievedChunk；
        # 两种混合报告只能保存HybridRetrievedChunk。
        expected_result_type: type[
            RetrievedChunk
        ] | type[
            HybridRetrievedChunk
        ]

        if (
            self.parameters.strategy
            == "vector_baseline"
        ):
            expected_result_type = (
                RetrievedChunk
            )
        else:
            expected_result_type = (
                HybridRetrievedChunk
            )

        for question_result in self.results:
            for retrieved in (
                question_result.retrieved_chunks
            ):
                if not isinstance(
                    retrieved,
                    expected_result_type,
                ):
                    raise ValueError(
                        "报告策略与检索结果类型不一致"
                    )

        answerable_results = [
            result
            for result in self.results
            if result.answerable
        ]

        unanswerable_results = [
            result
            for result in self.results
            if not result.answerable
        ]

        # 从逐题结果重新计算所有整数指标。
        #
        # 不能只相信调用方传入的metrics，
        # 否则报告正文和汇总表可能互相矛盾。
        actual_answerable_count = len(
            answerable_results
        )

        actual_unanswerable_count = len(
            unanswerable_results
        )

        actual_expected_evidence_count = sum(
            len(result.expected_evidence)
            for result in answerable_results
        )

        actual_matched_at_1 = sum(
            result.matched_evidence_at_1
            for result in answerable_results
        )

        actual_matched_at_3 = sum(
            result.matched_evidence_at_3
            for result in answerable_results
        )

        actual_fully_recalled_at_1 = sum(
            1
            for result in answerable_results
            if result.fully_recalled_at_1
        )

        actual_fully_recalled_at_3 = sum(
            1
            for result in answerable_results
            if result.fully_recalled_at_3
        )

        reported_counts = (
            self.metrics.answerable_question_count,
            self.metrics.unanswerable_question_count,
            self.metrics.expected_evidence_count,
            self.metrics.matched_evidence_at_1,
            self.metrics.matched_evidence_at_3,
            self.metrics.fully_recalled_at_1_count,
            self.metrics.fully_recalled_at_3_count,
        )

        actual_counts = (
            actual_answerable_count,
            actual_unanswerable_count,
            actual_expected_evidence_count,
            actual_matched_at_1,
            actual_matched_at_3,
            actual_fully_recalled_at_1,
            actual_fully_recalled_at_3,
        )

        if reported_counts != actual_counts:
            raise ValueError(
                "汇总指标与逐题结果不一致"
            )

        return self
