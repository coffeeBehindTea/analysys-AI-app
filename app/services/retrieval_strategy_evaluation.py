"""候选检索策略的逐题评分、指标汇总和报告构造。

本模块只负责“判卷”：

1. 接收已经完成排序的候选结果；
2. 根据Gold证据位置计算命中；
3. 生成逐题评测结果；
4. 汇总Recall和完整召回题数；
5. 构造经过Pydantic校验的策略报告。

本模块不生成Embedding、不访问Chroma、
不执行关键词检索、不调用LLM，也不负责门控。
"""

from app.schemas.evaluation import (
    GoldQuestion,
)
from app.schemas.retrieval import (
    HybridRetrievedChunk,
    RetrievedChunk,
)
from app.schemas.retrieval_strategy import (
    CandidateQuestionEvaluation,
    CandidateStrategyMetrics,
    CandidateStrategyParameters,
    CandidateStrategyReport,
    StrategyRetrievedChunk,
)
from app.services.retrieval_evaluation import (
    count_matched_evidence,
)


def _validate_and_order_candidates(
    retrieved_chunks: object,
) -> list[StrategyRetrievedChunk]:
    """校验候选列表并按照rank返回深复制快照。"""

    # 当前公共契约明确要求list。
    #
    # 字符串、元组、None和单个结果对象
    # 都不能代替一份完整候选列表。
    if not isinstance(
        retrieved_chunks,
        list,
    ):
        raise TypeError(
            "retrieved_chunks必须是列表"
        )

    seen_chunk_ids: set[str] = set()

    for position, result in enumerate(
        retrieved_chunks
    ):
        # 联合类型的类型注解不会自动执行运行时校验，
        # 所以公共函数仍需使用isinstance()检查。
        if not isinstance(
            result,
            (
                RetrievedChunk,
                HybridRetrievedChunk,
            ),
        ):
            raise TypeError(
                "retrieved_chunks中的第 "
                f"{position} 项必须是"
                "RetrievedChunk或"
                "HybridRetrievedChunk"
            )

        chunk_id = result.chunk.chunk_id

        # 同一Chunk不能在最终排名中出现两次。
        #
        # 重复结果会浪费Top-K位置，
        # 并使不同策略的比较失去公平性。
        if chunk_id in seen_chunk_ids:
            raise ValueError(
                "retrieved_chunks不能包含重复的"
                f"chunk_id：{chunk_id}"
            )

        seen_chunk_ids.add(chunk_id)

    # sorted()返回新列表，不修改调用方列表。
    ordered_candidates = sorted(
        retrieved_chunks,
        key=lambda result: result.rank,
    )

    actual_ranks = [
        result.rank
        for result in ordered_candidates
    ]

    expected_ranks = list(
        range(
            1,
            len(ordered_candidates) + 1,
        )
    )

    # 一份最终候选排名必须是：
    #
    # 1、2、3……
    #
    # 不能存在重复排名或排名缺口。
    if actual_ranks != expected_ranks:
        raise ValueError(
            "retrieved_chunks的rank"
            "必须从1开始连续且不能重复"
        )

    # 深复制每个Pydantic结果对象。
    #
    # 评测报告形成后，调用方再修改原候选正文，
    # 不能改变已经生成的评测证据快照。
    return [
        result.model_copy(deep=True)
        for result in ordered_candidates
    ]


def evaluate_candidate_question(
    *,
    question: GoldQuestion,
    retrieved_chunks: list[
        StrategyRetrievedChunk
    ],
) -> CandidateQuestionEvaluation:
    """使用统一证据规则评价一道Gold问题。"""

    if not isinstance(
        question,
        GoldQuestion,
    ):
        raise TypeError(
            "question必须是GoldQuestion"
        )

    ordered_candidates = (
        _validate_and_order_candidates(
            retrieved_chunks
        )
    )

    # 切片在结果不足时不会报错。
    #
    # 空列表：
    # ordered_candidates[:1] → []
    #
    # 只有两项：
    # ordered_candidates[:3] → 两项
    top_1 = ordered_candidates[:1]
    top_3 = ordered_candidates[:3]

    if question.answerable:
        matched_at_1 = (
            count_matched_evidence(
                top_1,
                question.expected_evidence,
            )
        )

        matched_at_3 = (
            count_matched_evidence(
                top_3,
                question.expected_evidence,
            )
        )

        expected_count = len(
            question.expected_evidence
        )

        fully_recalled_at_1 = (
            matched_at_1 == expected_count
        )

        fully_recalled_at_3 = (
            matched_at_3 == expected_count
        )

    else:
        # 无答案问题没有Gold证据，
        # 不能因为0 == 0就标成完整召回。
        matched_at_1 = 0
        matched_at_3 = 0
        fully_recalled_at_1 = False
        fully_recalled_at_3 = False

    return CandidateQuestionEvaluation(
        question_id=question.question_id,
        question=question.question,
        question_type=question.question_type,
        answerable=question.answerable,

        # 保存人工证据标注的深复制快照。
        expected_evidence=[
            expected.model_copy(deep=True)
            for expected
            in question.expected_evidence
        ],

        # ordered_candidates已经完成深复制。
        retrieved_chunks=ordered_candidates,
        matched_evidence_at_1=matched_at_1,
        matched_evidence_at_3=matched_at_3,
        fully_recalled_at_1=(
            fully_recalled_at_1
        ),
        fully_recalled_at_3=(
            fully_recalled_at_3
        ),
    )


def calculate_candidate_strategy_metrics(
    evaluations: list[
        CandidateQuestionEvaluation
    ],
) -> CandidateStrategyMetrics:
    """根据全部逐题结果计算候选召回指标。"""

    if not isinstance(
        evaluations,
        list,
    ):
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
            CandidateQuestionEvaluation,
        ):
            raise TypeError(
                "evaluations中的第 "
                f"{position} 项必须是"
                "CandidateQuestionEvaluation"
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

    expected_evidence_count = sum(
        len(evaluation.expected_evidence)
        for evaluation
        in answerable_evaluations
    )

    matched_evidence_at_1 = sum(
        evaluation.matched_evidence_at_1
        for evaluation
        in answerable_evaluations
    )

    matched_evidence_at_3 = sum(
        evaluation.matched_evidence_at_3
        for evaluation
        in answerable_evaluations
    )

    recall_at_1 = (
        matched_evidence_at_1
        / expected_evidence_count
        if expected_evidence_count
        else 0.0
    )

    recall_at_3 = (
        matched_evidence_at_3
        / expected_evidence_count
        if expected_evidence_count
        else 0.0
    )

    fully_recalled_at_1_count = sum(
        1
        for evaluation
        in answerable_evaluations
        if evaluation.fully_recalled_at_1
    )

    fully_recalled_at_3_count = sum(
        1
        for evaluation
        in answerable_evaluations
        if evaluation.fully_recalled_at_3
    )

    return CandidateStrategyMetrics(
        answerable_question_count=len(
            answerable_evaluations
        ),
        unanswerable_question_count=len(
            unanswerable_evaluations
        ),
        expected_evidence_count=(
            expected_evidence_count
        ),
        matched_evidence_at_1=(
            matched_evidence_at_1
        ),
        matched_evidence_at_3=(
            matched_evidence_at_3
        ),
        recall_at_1=recall_at_1,
        recall_at_3=recall_at_3,
        fully_recalled_at_1_count=(
            fully_recalled_at_1_count
        ),
        fully_recalled_at_3_count=(
            fully_recalled_at_3_count
        ),
    )


def build_candidate_strategy_report(
    *,
    embedding_model: str,
    collection_name: str,
    parameters: CandidateStrategyParameters,
    evaluations: list[
        CandidateQuestionEvaluation
    ],
) -> CandidateStrategyReport:
    """计算汇总指标并构造完整候选策略报告。"""

    if not isinstance(
        parameters,
        CandidateStrategyParameters,
    ):
        raise TypeError(
            "parameters必须是"
            "CandidateStrategyParameters"
        )

    # calculate_candidate_strategy_metrics()
    # 会校验列表、元素类型、非空和重复题号。
    metrics = (
        calculate_candidate_strategy_metrics(
            evaluations
        )
    )

    # 保存逐题结果的深复制快照，
    # 避免调用方随后修改原评测对象。
    evaluation_snapshots = [
        evaluation.model_copy(deep=True)
        for evaluation in evaluations
    ]

    return CandidateStrategyReport(
        embedding_model=embedding_model,
        collection_name=collection_name,
        parameters=parameters.model_copy(
            deep=True
        ),
        metrics=metrics,
        results=evaluation_snapshots,
    )
