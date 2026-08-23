"""比较三种候选检索策略并生成Markdown内容。

本模块负责：

1. 检查三份策略报告是否来自同一实验条件；
2. 计算完整证据召回率和相对基线变化；
3. 按固定规则选出候选阶段胜者；
4. 识别改进题、退步题和未解决题；
5. 生成可审计的Markdown文本。

本模块不读取或写入文件，不访问Embedding、
Chroma或LLM，也不负责证据门控。
"""

from dataclasses import dataclass

from app.schemas.retrieval_strategy import (
    CandidateStrategyReport,
    RetrievalStrategyName,
)


# 固定策略顺序同时承担两个职责：
#
# 1. 控制报告中的展示顺序；
# 2. 指标完全相同时，优先选择结构更简单的策略。
STRATEGY_ORDER: tuple[
    RetrievalStrategyName,
    ...
] = (
    "vector_baseline",
    "hybrid_rrf",
    "hybrid_rrf_rewrite",
)


# 程序内部使用稳定英文名称，
# 面向读者的Markdown使用中文名称。
STRATEGY_DISPLAY_NAMES: dict[
    RetrievalStrategyName,
    str,
] = {
    "vector_baseline": "纯向量基线",
    "hybrid_rrf": "关键词 + 向量 + RRF",
    "hybrid_rrf_rewrite": (
        "关键词 + 向量 + RRF + 查询改写"
    ),
}


@dataclass(
    frozen=True,
    slots=True,
)
class StrategyMetricComparison:
    """一种策略在对比报告中的指标快照。"""

    strategy: RetrievalStrategyName
    display_name: str

    # 固定实验参数。
    top_k: int
    candidate_k: int | None
    rank_constant: int | None
    rewrite_version: str | None

    # 证据级指标。
    matched_evidence_at_1: int
    matched_evidence_at_3: int
    recall_at_1: float
    recall_at_3: float

    # 问题级完整召回指标。
    fully_recalled_at_1_count: int
    fully_recalled_at_3_count: int
    fully_recalled_at_1_rate: float
    fully_recalled_at_3_rate: float

    # 所有delta都以纯向量基线为参照。
    recall_at_1_delta: float
    recall_at_3_delta: float
    fully_recalled_at_3_rate_delta: float


@dataclass(
    frozen=True,
    slots=True,
)
class StrategyQuestionOutcome:
    """一道题在一种策略下的候选召回结果。"""

    strategy: RetrievalStrategyName
    matched_evidence_at_1: int
    matched_evidence_at_3: int
    fully_recalled_at_1: bool
    fully_recalled_at_3: bool


@dataclass(
    frozen=True,
    slots=True,
)
class QuestionStrategyComparison:
    """一道Gold问题在三种策略下的横向对比。"""

    question_id: str
    question: str
    answerable: bool
    expected_evidence_count: int

    # 顺序始终遵循STRATEGY_ORDER。
    outcomes: tuple[
        StrategyQuestionOutcome,
        ...,
    ]

    # 保存候选胜者的Top-1位置，
    # 用于分析尚未完整召回的问题。
    winner_top_1_source_file: str | None
    winner_top_1_page_or_section: str | None


@dataclass(
    frozen=True,
    slots=True,
)
class CandidateStrategyComparison:
    """三种候选策略的完整对比结果。"""

    embedding_model: str
    collection_name: str
    top_k: int

    question_count: int
    answerable_question_count: int
    unanswerable_question_count: int
    expected_evidence_count: int

    strategy_rows: tuple[
        StrategyMetricComparison,
        ...,
    ]

    question_rows: tuple[
        QuestionStrategyComparison,
        ...,
    ]

    # 这里只能叫candidate_winner，
    # 不能叫production_strategy。
    #
    # 因为无答案门控和最终回答指标尚未评测。
    candidate_winner: RetrievalStrategyName

    improved_question_ids: tuple[str, ...]
    regressed_question_ids: tuple[str, ...]
    unresolved_question_ids: tuple[str, ...]


def _validate_comparable_reports(
    reports: object,
) -> dict[
    RetrievalStrategyName,
    CandidateStrategyReport,
]:
    """校验并按照策略名称索引三份报告。"""

    # 类型注解不会在普通Python调用时自动生效，
    # 因此公共边界仍然需要运行时检查。
    if not isinstance(reports, list):
        raise TypeError(
            "reports必须是列表"
        )

    # 当前Week 3实验明确要求比较三种策略。
    if len(reports) != len(STRATEGY_ORDER):
        raise ValueError(
            "reports必须恰好包含三种候选策略"
        )

    reports_by_strategy: dict[
        RetrievalStrategyName,
        CandidateStrategyReport,
    ] = {}

    for position, report in enumerate(reports):
        if not isinstance(
            report,
            CandidateStrategyReport,
        ):
            raise TypeError(
                "reports中的第 "
                f"{position} 项必须是"
                "CandidateStrategyReport"
            )

        strategy = report.parameters.strategy

        # 如果直接赋值而不检查，
        # 后出现的同名策略会覆盖前一个报告。
        if strategy in reports_by_strategy:
            raise ValueError(
                "reports不能包含重复策略："
                f"{strategy}"
            )

        reports_by_strategy[strategy] = report

    expected_strategies = set(STRATEGY_ORDER)
    actual_strategies = set(
        reports_by_strategy
    )

    if actual_strategies != expected_strategies:
        raise ValueError(
            "reports必须恰好包含三种候选策略"
        )

    baseline = reports_by_strategy[
        "vector_baseline"
    ]

    baseline_question_by_id = {
        result.question_id: result
        for result in baseline.results
    }

    for strategy in STRATEGY_ORDER[1:]:
        candidate = reports_by_strategy[
            strategy
        ]

        # Embedding模型不同意味着向量空间不同，
        # 召回分数和排名不能进行公平对比。
        if (
            candidate.embedding_model
            != baseline.embedding_model
        ):
            raise ValueError(
                "三份报告的Embedding模型必须一致"
            )

        # Collection名称用于确认三次实验
        # 使用的是同一份语料与向量索引。
        if (
            candidate.collection_name
            != baseline.collection_name
        ):
            raise ValueError(
                "三份报告的Collection名称必须一致"
            )

        # Recall@3必须来自相同的最终候选深度。
        if (
            candidate.parameters.top_k
            != baseline.parameters.top_k
        ):
            raise ValueError(
                "三份报告的top_k必须一致"
            )

        candidate_question_by_id = {
            result.question_id: result
            for result in candidate.results
        }

        if (
            set(candidate_question_by_id)
            != set(baseline_question_by_id)
        ):
            raise ValueError(
                "三份报告的Gold问题契约必须一致"
            )

        for question_id, baseline_result in (
            baseline_question_by_id.items()
        ):
            candidate_result = (
                candidate_question_by_id[
                    question_id
                ]
            )

            # 报告中没有保存reference_answer，
            # 但用于候选召回评分的以下字段必须一致：
            #
            # 1. 问题正文；
            # 2. 问题类型；
            # 3. 是否可回答；
            # 4. 预期证据位置。
            if (
                candidate_result.question
                != baseline_result.question
                or candidate_result.question_type
                != baseline_result.question_type
                or candidate_result.answerable
                != baseline_result.answerable
                or candidate_result.expected_evidence
                != baseline_result.expected_evidence
            ):
                raise ValueError(
                    "三份报告的Gold问题契约必须一致"
                )

    return reports_by_strategy


def _calculate_complete_rate(
    *,
    complete_count: int,
    answerable_count: int,
) -> float:
    """计算问题级完整召回率。"""

    if answerable_count == 0:
        return 0.0

    return complete_count / answerable_count


def _build_strategy_rows(
    reports_by_strategy: dict[
        RetrievalStrategyName,
        CandidateStrategyReport,
    ],
) -> tuple[
    StrategyMetricComparison,
    ...,
]:
    """构造按固定顺序排列的指标对比行。"""

    baseline = reports_by_strategy[
        "vector_baseline"
    ]
    baseline_metrics = baseline.metrics

    baseline_full_at_3_rate = (
        _calculate_complete_rate(
            complete_count=(
                baseline_metrics
                .fully_recalled_at_3_count
            ),
            answerable_count=(
                baseline_metrics
                .answerable_question_count
            ),
        )
    )

    rows: list[
        StrategyMetricComparison
    ] = []

    for strategy in STRATEGY_ORDER:
        report = reports_by_strategy[strategy]
        parameters = report.parameters
        metrics = report.metrics

        full_at_1_rate = (
            _calculate_complete_rate(
                complete_count=(
                    metrics
                    .fully_recalled_at_1_count
                ),
                answerable_count=(
                    metrics
                    .answerable_question_count
                ),
            )
        )

        full_at_3_rate = (
            _calculate_complete_rate(
                complete_count=(
                    metrics
                    .fully_recalled_at_3_count
                ),
                answerable_count=(
                    metrics
                    .answerable_question_count
                ),
            )
        )

        rows.append(
            StrategyMetricComparison(
                strategy=strategy,
                display_name=(
                    STRATEGY_DISPLAY_NAMES[
                        strategy
                    ]
                ),
                top_k=parameters.top_k,
                candidate_k=(
                    parameters.candidate_k
                ),
                rank_constant=(
                    parameters.rank_constant
                ),
                rewrite_version=(
                    parameters.rewrite_version
                ),
                matched_evidence_at_1=(
                    metrics.matched_evidence_at_1
                ),
                matched_evidence_at_3=(
                    metrics.matched_evidence_at_3
                ),
                recall_at_1=(
                    metrics.recall_at_1
                ),
                recall_at_3=(
                    metrics.recall_at_3
                ),
                fully_recalled_at_1_count=(
                    metrics
                    .fully_recalled_at_1_count
                ),
                fully_recalled_at_3_count=(
                    metrics
                    .fully_recalled_at_3_count
                ),
                fully_recalled_at_1_rate=(
                    full_at_1_rate
                ),
                fully_recalled_at_3_rate=(
                    full_at_3_rate
                ),
                recall_at_1_delta=(
                    metrics.recall_at_1
                    - baseline_metrics.recall_at_1
                ),
                recall_at_3_delta=(
                    metrics.recall_at_3
                    - baseline_metrics.recall_at_3
                ),
                fully_recalled_at_3_rate_delta=(
                    full_at_3_rate
                    - baseline_full_at_3_rate
                ),
            )
        )

    return tuple(rows)


def _select_candidate_winner(
    strategy_rows: tuple[
        StrategyMetricComparison,
        ...,
    ],
) -> RetrievalStrategyName:
    """按照稳定优先级选择候选层胜者。"""

    winner = max(
        strategy_rows,
        key=lambda row: (
            # 第一优先级：
            # 一道题需要的证据是否全部进入Top-3。
            row.fully_recalled_at_3_rate,

            # 第二优先级：
            # 所有预期证据在Top-3中的覆盖程度。
            row.recall_at_3,

            # 第三优先级：
            # 最重要的第一名是否正确。
            row.recall_at_1,

            # 指标完全相同时优先简单策略。
            #
            # baseline索引为0，得到0；
            # hybrid索引为1，得到-1；
            # rewrite索引为2，得到-2。
            #
            # max()会选择数值更大的0。
            -STRATEGY_ORDER.index(
                row.strategy
            ),
        ),
    )

    return winner.strategy


def _build_question_rows(
    *,
    reports_by_strategy: dict[
        RetrievalStrategyName,
        CandidateStrategyReport,
    ],
    candidate_winner: RetrievalStrategyName,
) -> tuple[
    QuestionStrategyComparison,
    ...,
]:
    """构造逐题横向对比数据。"""

    result_maps = {
        strategy: {
            result.question_id: result
            for result in (
                reports_by_strategy[
                    strategy
                ].results
            )
        }
        for strategy in STRATEGY_ORDER
    }

    # 以基线报告中的顺序作为稳定展示顺序。
    baseline_results = reports_by_strategy[
        "vector_baseline"
    ].results

    question_rows: list[
        QuestionStrategyComparison
    ] = []

    for baseline_result in baseline_results:
        question_id = (
            baseline_result.question_id
        )

        outcomes = tuple(
            StrategyQuestionOutcome(
                strategy=strategy,
                matched_evidence_at_1=(
                    result_maps[strategy][
                        question_id
                    ].matched_evidence_at_1
                ),
                matched_evidence_at_3=(
                    result_maps[strategy][
                        question_id
                    ].matched_evidence_at_3
                ),
                fully_recalled_at_1=(
                    result_maps[strategy][
                        question_id
                    ].fully_recalled_at_1
                ),
                fully_recalled_at_3=(
                    result_maps[strategy][
                        question_id
                    ].fully_recalled_at_3
                ),
            )
            for strategy in STRATEGY_ORDER
        )

        winner_result = result_maps[
            candidate_winner
        ][question_id]

        if winner_result.retrieved_chunks:
            winner_top_1 = (
                winner_result.retrieved_chunks[0]
            )
            winner_top_1_source_file = (
                winner_top_1.chunk.source_file
            )
            winner_top_1_page_or_section = (
                winner_top_1
                .chunk
                .page_or_section
            )
        else:
            winner_top_1_source_file = None
            winner_top_1_page_or_section = None

        question_rows.append(
            QuestionStrategyComparison(
                question_id=question_id,
                question=baseline_result.question,
                answerable=(
                    baseline_result.answerable
                ),
                expected_evidence_count=len(
                    baseline_result
                    .expected_evidence
                ),
                outcomes=outcomes,
                winner_top_1_source_file=(
                    winner_top_1_source_file
                ),
                winner_top_1_page_or_section=(
                    winner_top_1_page_or_section
                ),
            )
        )

    return tuple(question_rows)


def _get_question_outcome(
    *,
    question_row: QuestionStrategyComparison,
    strategy: RetrievalStrategyName,
) -> StrategyQuestionOutcome:
    """取得一道题在指定策略下的结果。"""

    for outcome in question_row.outcomes:
        if outcome.strategy == strategy:
            return outcome

    # 正常情况下不会进入这里，
    # 因为_build_question_rows()始终构造三种结果。
    raise ValueError(
        f"问题{question_row.question_id}"
        f"缺少策略结果：{strategy}"
    )


def compare_candidate_strategy_reports(
    reports: object,
) -> CandidateStrategyComparison:
    """比较三份报告并返回稳定的对比结果。"""

    reports_by_strategy = (
        _validate_comparable_reports(
            reports
        )
    )

    baseline_report = reports_by_strategy[
        "vector_baseline"
    ]
    baseline_metrics = (
        baseline_report.metrics
    )

    strategy_rows = _build_strategy_rows(
        reports_by_strategy
    )

    candidate_winner = (
        _select_candidate_winner(
            strategy_rows
        )
    )

    question_rows = _build_question_rows(
        reports_by_strategy=reports_by_strategy,
        candidate_winner=candidate_winner,
    )

    improved_question_ids: list[str] = []
    regressed_question_ids: list[str] = []
    unresolved_question_ids: list[str] = []

    for question_row in question_rows:
        if not question_row.answerable:
            continue

        baseline_outcome = (
            _get_question_outcome(
                question_row=question_row,
                strategy="vector_baseline",
            )
        )
        winner_outcome = (
            _get_question_outcome(
                question_row=question_row,
                strategy=candidate_winner,
            )
        )

        # 改进和退步按照Top-3命中的
        # 预期证据数量判断。
        if (
            winner_outcome
            .matched_evidence_at_3
            > baseline_outcome
            .matched_evidence_at_3
        ):
            improved_question_ids.append(
                question_row.question_id
            )

        elif (
            winner_outcome
            .matched_evidence_at_3
            < baseline_outcome
            .matched_evidence_at_3
        ):
            regressed_question_ids.append(
                question_row.question_id
            )

        # 即使比基线有所改善，
        # 没有找齐全部证据仍属于未解决。
        if not winner_outcome.fully_recalled_at_3:
            unresolved_question_ids.append(
                question_row.question_id
            )

    return CandidateStrategyComparison(
        embedding_model=(
            baseline_report.embedding_model
        ),
        collection_name=(
            baseline_report.collection_name
        ),
        top_k=(
            baseline_report.parameters.top_k
        ),
        question_count=len(
            baseline_report.results
        ),
        answerable_question_count=(
            baseline_metrics
            .answerable_question_count
        ),
        unanswerable_question_count=(
            baseline_metrics
            .unanswerable_question_count
        ),
        expected_evidence_count=(
            baseline_metrics
            .expected_evidence_count
        ),
        strategy_rows=strategy_rows,
        question_rows=question_rows,
        candidate_winner=candidate_winner,
        improved_question_ids=tuple(
            improved_question_ids
        ),
        regressed_question_ids=tuple(
            regressed_question_ids
        ),
        unresolved_question_ids=tuple(
            unresolved_question_ids
        ),
    )


def _format_optional_value(
    value: int | str | None,
) -> str:
    """把没有使用的参数显示为破折号。"""

    if value is None:
        return "—"

    return str(value)


def _format_delta(value: float) -> str:
    """把差值格式化为带正负号的三位小数。"""

    return f"{value:+.3f}"


def _escape_markdown_cell(value: object) -> str:
    """转义Markdown表格中的特殊字符和换行。"""

    return (
        str(value)
        .replace("|", "\\|")
        .replace("\r", " ")
        .replace("\n", " ")
    )


def render_candidate_strategy_comparison(
    comparison: CandidateStrategyComparison,
) -> str:
    """把结构化对比结果渲染成Markdown。"""

    if not isinstance(
        comparison,
        CandidateStrategyComparison,
    ):
        raise TypeError(
            "comparison必须是"
            "CandidateStrategyComparison"
        )

    winner_name = STRATEGY_DISPLAY_NAMES[
        comparison.candidate_winner
    ]

    lines = [
        "# Week 3 候选检索策略对比",
        "",
        "## 1. 实验范围与可比性",
        "",
        (
            f"- Embedding 模型："
            f"`{comparison.embedding_model}`"
        ),
        (
            f"- Chroma Collection："
            f"`{comparison.collection_name}`"
        ),
        f"- 最终候选深度：Top-{comparison.top_k}",
        (
            f"- Gold 问题总数："
            f"{comparison.question_count}"
        ),
        (
            f"- 可回答问题数："
            f"{comparison.answerable_question_count}"
        ),
        (
            f"- 无答案问题数："
            f"{comparison.unanswerable_question_count}"
        ),
        (
            f"- 预期证据总数："
            f"{comparison.expected_evidence_count}"
        ),
        (
            "- 本报告评价的是候选阶段，"
            "不代表已经获准接入在线 API。"
        ),
        "",
        "## 2. 策略参数",
        "",
        (
            "| 策略 | top_k | candidate_k | "
            "RRF 常数 | 改写版本 |"
        ),
        "|---|---:|---:|---:|---|",
    ]

    for row in comparison.strategy_rows:
        lines.append(
            "| "
            f"{row.display_name} | "
            f"{row.top_k} | "
            f"{_format_optional_value(row.candidate_k)} | "
            f"{_format_optional_value(row.rank_constant)} | "
            f"{_format_optional_value(row.rewrite_version)} |"
        )

    lines.extend(
        [
            "",
            "## 3. 汇总指标",
            "",
            (
                "| 策略 | Recall@1 | Recall@3 | "
                "Top-3 完整证据召回率 | "
                "Recall@1 Δ | Recall@3 Δ | "
                "完整召回率 Δ |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for row in comparison.strategy_rows:
        lines.append(
            "| "
            f"{row.display_name} | "
            f"{row.recall_at_1:.3f} "
            f"({row.matched_evidence_at_1}/"
            f"{comparison.expected_evidence_count}) | "
            f"{row.recall_at_3:.3f} "
            f"({row.matched_evidence_at_3}/"
            f"{comparison.expected_evidence_count}) | "
            f"{row.fully_recalled_at_3_rate:.3f} "
            f"({row.fully_recalled_at_3_count}/"
            f"{comparison.answerable_question_count}) | "
            f"{_format_delta(row.recall_at_1_delta)} | "
            f"{_format_delta(row.recall_at_3_delta)} | "
            f"{_format_delta(row.fully_recalled_at_3_rate_delta)} |"
        )

    lines.extend(
        [
            "",
            "## 4. 候选阶段结论",
            "",
            f"- 候选层胜者：{winner_name}。",
            (
                "- 选择优先级依次为："
                "Top-3 完整证据召回率、"
                "Recall@3、Recall@1。"
            ),
            (
                "- 如果上述指标完全相同，"
                "优先选择依赖更少、结构更简单的策略。"
            ),
            "",
            "## 5. 逐题变化",
            "",
            (
                "- 相对纯向量基线得到更多"
                "Top-3 正确证据的问题："
                + (
                    "、".join(
                        comparison
                        .improved_question_ids
                    )
                    if comparison
                    .improved_question_ids
                    else "无"
                )
                + "。"
            ),
            (
                "- 相对纯向量基线丢失"
                "Top-3 正确证据的问题："
                + (
                    "、".join(
                        comparison
                        .regressed_question_ids
                    )
                    if comparison
                    .regressed_question_ids
                    else "无"
                )
                + "。"
            ),
            "",
            (
                "| 问题 | 预期证据数 | "
                "纯向量 Top-3 | "
                "混合 RRF Top-3 | "
                "混合 RRF + 改写 Top-3 |"
            ),
            "|---|---:|---:|---:|---:|",
        ]
    )

    for question_row in comparison.question_rows:
        if not question_row.answerable:
            continue

        outcomes_by_strategy = {
            outcome.strategy: outcome
            for outcome in question_row.outcomes
        }

        lines.append(
            "| "
            f"{question_row.question_id}："
            f"{_escape_markdown_cell(question_row.question)} | "
            f"{question_row.expected_evidence_count} | "
            f"{outcomes_by_strategy['vector_baseline'].matched_evidence_at_3} | "
            f"{outcomes_by_strategy['hybrid_rrf'].matched_evidence_at_3} | "
            f"{outcomes_by_strategy['hybrid_rrf_rewrite'].matched_evidence_at_3} |"
        )

    lines.extend(
        [
            "",
            "## 6. 候选胜者尚未解决的问题",
            "",
        ]
    )

    unresolved_rows = [
        row
        for row in comparison.question_rows
        if (
            row.question_id
            in comparison.unresolved_question_ids
        )
    ]

    if not unresolved_rows:
        lines.append(
            "- 候选胜者在当前测试数据中"
            "已完整召回所有可回答问题的Top-3证据。"
        )
    else:
        lines.extend(
            [
                (
                    "| 问题 | 候选胜者 Top-1 来源 | "
                    "Top-1 位置 |"
                ),
                "|---|---|---|",
            ]
        )

        for row in unresolved_rows:
            source_file = (
                row.winner_top_1_source_file
                or "无候选"
            )
            location = (
                row.winner_top_1_page_or_section
                or "无候选"
            )

            lines.append(
                "| "
                f"{row.question_id}："
                f"{_escape_markdown_cell(row.question)} | "
                f"{_escape_markdown_cell(source_file)} | "
                f"{_escape_markdown_cell(location)} |"
            )

    lines.extend(
        [
            "",
            "## 7. 尚未完成的安全评测",
            "",
            (
                "- 当前报告只判断正确证据是否"
                "进入Top-1或Top-3。"
            ),
            (
                "- 无答案问题也可能返回表面相关的候选，"
                "候选报告本身不会决定是否调用LLM。"
            ),
            "- 无答案错误召回率尚未计算。",
            "- 可回答响应率尚未计算。",
            "- 完整证据回答率尚未计算。",
            "- 无答案正确拒答率尚未计算。",
            (
                "- 必须完成混合结果门控校准后，"
                "才能决定是否接入在线API。"
            ),
            "",
        ]
    )

    return "\n".join(lines)