"""评价混合证据门控并汇总覆盖率与安全性指标。

本模块负责：

1. 对一道候选评测结果执行门控；
2. 根据Gold可回答性和证据完整性分类结果；
3. 汇总门控覆盖率及拒答安全性；
4. 构造带候选策略和门控版本的报告快照。

本模块不执行检索、不调用Embedding或LLM，
也不把门控指标冒充最终回答正确率。
"""

from dataclasses import dataclass
from typing import Literal

from app.schemas.evaluation import (
    ExpectedEvidence,
)
from app.schemas.retrieval import (
    HybridRetrievedChunk,
)
from app.schemas.retrieval_strategy import (
    CandidateQuestionEvaluation,
    CandidateStrategyParameters,
    CandidateStrategyReport,
)
from app.services.hybrid_retrieval_gate import (
    HybridEvidenceGateDecision,
    HybridEvidenceGatePolicy,
    evaluate_hybrid_evidence_gate,
)


# 五种结果形成门控层的“混淆矩阵”。
HybridGateQuestionOutcome = Literal[
    "accepted_with_full_evidence",
    "accepted_with_incomplete_evidence",
    "rejected_answerable",
    "false_accept_unanswerable",
    "correct_abstain_unanswerable",
]


@dataclass(
    frozen=True,
    slots=True,
)
class HybridGateQuestionEvaluation:
    """一道Gold问题经过门控后的完整评测结果。"""

    question_id: str
    question: str
    question_type: str
    answerable: bool

    # 保存Gold证据位置和真实候选快照，
    # 方便后续报告逐题审计。
    expected_evidence: tuple[
        ExpectedEvidence,
        ...,
    ]
    retrieved_chunks: tuple[
        HybridRetrievedChunk,
        ...,
    ]

    matched_evidence_at_1: int
    matched_evidence_at_3: int
    fully_recalled_at_1: bool
    fully_recalled_at_3: bool

    decision: HybridEvidenceGateDecision
    outcome: HybridGateQuestionOutcome


@dataclass(
    frozen=True,
    slots=True,
)
class HybridGateMetrics:
    """一套Gold问题上的门控汇总指标。"""

    answerable_question_count: int
    unanswerable_question_count: int

    # 可回答题中有多少被允许进入LLM阶段。
    accepted_answerable_count: int
    rejected_answerable_count: int

    # 放行的可回答题中，候选证据是否完整。
    accepted_with_full_evidence_count: int
    accepted_with_incomplete_evidence_count: int

    # 无答案题的错误放行和正确拒绝数量。
    false_accepted_unanswerable_count: int
    correctly_abstained_unanswerable_count: int

    # 可回答题门控放行率。
    answerable_acceptance_rate: float

    # 可回答题中：
    #
    # 门控放行并且Top-3证据完整的比例。
    #
    # 这是“完整证据回答率”的候选层上限，
    # 不是最终LLM回答正确率。
    full_evidence_eligible_rate: float

    # 无答案题被错误送入LLM的比例。
    unanswerable_false_acceptance_rate: float

    # 无答案题被门控正确拒绝的比例。
    unanswerable_abstention_rate: float


@dataclass(
    frozen=True,
    slots=True,
)
class HybridGateEvaluationReport:
    """混合候选策略与门控策略的完整评测报告。"""

    embedding_model: str
    collection_name: str

    # 保存产生候选的检索策略参数快照。
    candidate_parameters: (
        CandidateStrategyParameters
    )

    # frozen策略对象可以安全保留在报告中。
    gate_policy: HybridEvidenceGatePolicy

    metrics: HybridGateMetrics

    results: tuple[
        HybridGateQuestionEvaluation,
        ...,
    ]


def _classify_gate_outcome(
    *,
    answerable: bool,
    accepted: bool,
    fully_recalled_at_3: bool,
) -> HybridGateQuestionOutcome:
    """根据Gold标签、门控决定和证据完整性分类。"""

    if answerable:
        if not accepted:
            return "rejected_answerable"

        if fully_recalled_at_3:
            return (
                "accepted_with_full_evidence"
            )

        return (
            "accepted_with_incomplete_evidence"
        )

    if accepted:
        return "false_accept_unanswerable"

    return "correct_abstain_unanswerable"


def evaluate_hybrid_gate_question(
    *,
    question_result: CandidateQuestionEvaluation,
    policy: HybridEvidenceGatePolicy,
) -> HybridGateQuestionEvaluation:
    """对一道候选策略结果执行并评价门控。"""

    if not isinstance(
        question_result,
        CandidateQuestionEvaluation,
    ):
        raise TypeError(
            "question_result必须是"
            "CandidateQuestionEvaluation"
        )

    if not isinstance(
        policy,
        HybridEvidenceGatePolicy,
    ):
        raise TypeError(
            "policy必须是"
            "HybridEvidenceGatePolicy"
        )

    # evaluate_hybrid_evidence_gate()还会检查：
    #
    # 1. 所有结果必须是HybridRetrievedChunk；
    # 2. rank必须连续；
    # 3. chunk_id不能重复；
    # 4. 门控信号组合是否合法。
    decision = evaluate_hybrid_evidence_gate(
        retrieved_chunks=(
            question_result.retrieved_chunks
        ),
        policy=policy,
    )

    outcome = _classify_gate_outcome(
        answerable=question_result.answerable,
        accepted=decision.accepted,
        fully_recalled_at_3=(
            question_result
            .fully_recalled_at_3
        ),
    )

    # 深复制Pydantic对象。
    #
    # 后续修改原CandidateStrategyReport，
    # 不会改变已经产生的门控评测快照。
    expected_evidence = tuple(
        evidence.model_copy(deep=True)
        for evidence
        in question_result.expected_evidence
    )

    retrieved_chunks = tuple(
        candidate.model_copy(deep=True)
        for candidate
        in question_result.retrieved_chunks
    )

    return HybridGateQuestionEvaluation(
        question_id=(
            question_result.question_id
        ),
        question=question_result.question,
        question_type=(
            question_result.question_type
        ),
        answerable=(
            question_result.answerable
        ),
        expected_evidence=expected_evidence,
        retrieved_chunks=retrieved_chunks,
        matched_evidence_at_1=(
            question_result
            .matched_evidence_at_1
        ),
        matched_evidence_at_3=(
            question_result
            .matched_evidence_at_3
        ),
        fully_recalled_at_1=(
            question_result
            .fully_recalled_at_1
        ),
        fully_recalled_at_3=(
            question_result
            .fully_recalled_at_3
        ),
        decision=decision,
        outcome=outcome,
    )


def _validate_evaluation_outcome(
    evaluation: HybridGateQuestionEvaluation,
) -> None:
    """确认逐题对象中的结果分类仍然自洽。"""

    expected_outcome = _classify_gate_outcome(
        answerable=evaluation.answerable,
        accepted=evaluation.decision.accepted,
        fully_recalled_at_3=(
            evaluation.fully_recalled_at_3
        ),
    )

    if evaluation.outcome != expected_outcome:
        raise ValueError(
            "门控逐题结果的outcome"
            "与答案类型、决定或证据完整性不一致"
        )


def calculate_hybrid_gate_metrics(
    evaluations: object,
) -> HybridGateMetrics:
    """汇总全部逐题门控结果。"""

    if not isinstance(evaluations, list):
        raise TypeError(
            "evaluations必须是列表"
        )

    if not evaluations:
        raise ValueError(
            "evaluations不能为空"
        )

    seen_question_ids: set[str] = set()

    for position, evaluation in enumerate(
        evaluations
    ):
        if not isinstance(
            evaluation,
            HybridGateQuestionEvaluation,
        ):
            raise TypeError(
                "evaluations中的第 "
                f"{position} 项必须是"
                "HybridGateQuestionEvaluation"
            )

        if (
            evaluation.question_id
            in seen_question_ids
        ):
            raise ValueError(
                "evaluations不能包含重复的"
                f"question_id："
                f"{evaluation.question_id}"
            )

        seen_question_ids.add(
            evaluation.question_id
        )

        _validate_evaluation_outcome(
            evaluation
        )

    answerable_evaluations = [
        evaluation
        for evaluation in evaluations
        if evaluation.answerable
    ]

    unanswerable_evaluations = [
        evaluation
        for evaluation in evaluations
        if not evaluation.answerable
    ]

    accepted_answerable_count = sum(
        1
        for evaluation
        in answerable_evaluations
        if evaluation.decision.accepted
    )

    rejected_answerable_count = (
        len(answerable_evaluations)
        - accepted_answerable_count
    )

    accepted_with_full_evidence_count = sum(
        1
        for evaluation
        in answerable_evaluations
        if (
            evaluation.decision.accepted
            and evaluation.fully_recalled_at_3
        )
    )

    accepted_with_incomplete_evidence_count = sum(
        1
        for evaluation
        in answerable_evaluations
        if (
            evaluation.decision.accepted
            and not evaluation.fully_recalled_at_3
        )
    )

    false_accepted_unanswerable_count = sum(
        1
        for evaluation
        in unanswerable_evaluations
        if evaluation.decision.accepted
    )

    correctly_abstained_unanswerable_count = (
        len(unanswerable_evaluations)
        - false_accepted_unanswerable_count
    )

    answerable_count = len(
        answerable_evaluations
    )
    unanswerable_count = len(
        unanswerable_evaluations
    )

    answerable_acceptance_rate = (
        accepted_answerable_count
        / answerable_count
        if answerable_count
        else 0.0
    )

    full_evidence_eligible_rate = (
        accepted_with_full_evidence_count
        / answerable_count
        if answerable_count
        else 0.0
    )

    unanswerable_false_acceptance_rate = (
        false_accepted_unanswerable_count
        / unanswerable_count
        if unanswerable_count
        else 0.0
    )

    unanswerable_abstention_rate = (
        correctly_abstained_unanswerable_count
        / unanswerable_count
        if unanswerable_count
        else 0.0
    )

    return HybridGateMetrics(
        answerable_question_count=(
            answerable_count
        ),
        unanswerable_question_count=(
            unanswerable_count
        ),
        accepted_answerable_count=(
            accepted_answerable_count
        ),
        rejected_answerable_count=(
            rejected_answerable_count
        ),
        accepted_with_full_evidence_count=(
            accepted_with_full_evidence_count
        ),
        accepted_with_incomplete_evidence_count=(
            accepted_with_incomplete_evidence_count
        ),
        false_accepted_unanswerable_count=(
            false_accepted_unanswerable_count
        ),
        correctly_abstained_unanswerable_count=(
            correctly_abstained_unanswerable_count
        ),
        answerable_acceptance_rate=(
            answerable_acceptance_rate
        ),
        full_evidence_eligible_rate=(
            full_evidence_eligible_rate
        ),
        unanswerable_false_acceptance_rate=(
            unanswerable_false_acceptance_rate
        ),
        unanswerable_abstention_rate=(
            unanswerable_abstention_rate
        ),
    )


def build_hybrid_gate_evaluation_report(
    *,
    candidate_report: CandidateStrategyReport,
    policy: HybridEvidenceGatePolicy,
) -> HybridGateEvaluationReport:
    """使用一份混合候选报告构造门控评测报告。"""

    if not isinstance(
        candidate_report,
        CandidateStrategyReport,
    ):
        raise TypeError(
            "candidate_report必须是"
            "CandidateStrategyReport"
        )

    if not isinstance(
        policy,
        HybridEvidenceGatePolicy,
    ):
        raise TypeError(
            "policy必须是"
            "HybridEvidenceGatePolicy"
        )

    # 纯向量结果没有RRF排名、关键词命中解释等字段，
    # 无法执行当前组合门控。
    if (
        candidate_report.parameters.strategy
        == "vector_baseline"
    ):
        raise ValueError(
            "门控评测只接受混合候选策略报告"
        )

    evaluations = [
        evaluate_hybrid_gate_question(
            question_result=result,
            policy=policy,
        )
        for result in candidate_report.results
    ]

    metrics = calculate_hybrid_gate_metrics(
        evaluations
    )

    return HybridGateEvaluationReport(
        embedding_model=(
            candidate_report.embedding_model
        ),
        collection_name=(
            candidate_report.collection_name
        ),

        # CandidateStrategyParameters是Pydantic模型，
        # 使用deep=True保存独立快照。
        candidate_parameters=(
            candidate_report
            .parameters
            .model_copy(deep=True)
        ),

        # HybridEvidenceGatePolicy是不可变数据类，
        # 可以安全共享同一实例。
        gate_policy=policy,
        metrics=metrics,
        results=tuple(evaluations),
    )
