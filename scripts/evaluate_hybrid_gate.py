"""读取候选策略报告并评测混合证据门控。

本脚本负责文件和命令行边界：

1. 读取当前候选胜者的JSON报告；
2. 使用Pydantic重新校验报告；
3. 调用混合门控评测Service；
4. 生成本地完整JSON报告；
5. 生成不包含Chunk正文的Markdown审计报告；
6. 在终端输出覆盖率和无答案安全结论。

本脚本不重新执行Embedding、Chroma检索或LLM调用。
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from app.schemas.retrieval_strategy import (
    CandidateStrategyReport,
)
from app.services.hybrid_gate_evaluation import (
    HybridGateEvaluationReport,
    build_hybrid_gate_evaluation_report,
)
from app.services.hybrid_retrieval_gate import (
    HybridEvidenceGatePolicy,
)


# 默认读取当前候选层的胜者：
#
# 混合检索 + 确定性查询改写
# + 头部保留重排。
DEFAULT_CANDIDATE_REPORT = Path(
    "docs/retrieval-strategy-"
    "hybrid-rrf-rewrite-rerank.json"
)

# JSON报告包含完整候选Chunk，
# 只应保存在本地。
#
# 当前.gitignore中的/docs/*.json
# 已经阻止它进入Git。
DEFAULT_JSON_OUTPUT = Path(
    "docs/hybrid-gate-evaluation.json"
)

# Markdown报告不包含Chunk正文，
# 可用于人工检查和后续项目交付。
DEFAULT_MARKDOWN_OUTPUT = Path(
    "docs/hybrid-gate-evaluation.md"
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(
        description=(
            "读取候选策略JSON报告，"
            "评测混合证据门控的覆盖率和无答案安全性"
        )
    )

    parser.add_argument(
        "--candidate-report",
        type=Path,
        default=DEFAULT_CANDIDATE_REPORT,
        help=(
            "混合候选策略JSON报告路径；"
            "默认使用混合RRF、查询改写"
            "和头部保留重排报告"
        ),
    )

    parser.add_argument(
        "--json-output",
        type=Path,
        default=DEFAULT_JSON_OUTPUT,
        help="包含完整逐题结果的本地JSON报告路径",
    )

    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=DEFAULT_MARKDOWN_OUTPUT,
        help="不包含Chunk正文的Markdown摘要路径",
    )

    return parser.parse_args()


def load_candidate_report(
    *,
    path: Path,
) -> CandidateStrategyReport:
    """读取并校验候选策略JSON报告。"""

    # 类型注解不会在普通Python运行时
    # 自动阻止调用方传入字符串等非法对象，
    # 所以公共边界仍需显式检查。
    if not isinstance(path, Path):
        raise TypeError("path必须是Path")

    # read_text()会完成：
    #
    # 1. 打开文件；
    # 2. 读取完整文本；
    # 3. 自动关闭文件。
    #
    # 显式指定UTF-8，避免Windows中文乱码。
    json_text = path.read_text(
        encoding="utf-8"
    )

    # model_validate_json()同时完成：
    #
    # 1. JSON文本解析；
    # 2. 字段类型校验；
    # 3. 嵌套模型构造；
    # 4. 策略参数与候选类型一致性校验；
    # 5. 汇总指标与逐题结果一致性校验。
    return (
        CandidateStrategyReport
        .model_validate_json(json_text)
    )


def gate_report_to_dict(
    report: HybridGateEvaluationReport,
) -> dict[str, object]:
    """将门控报告转换成JSON兼容字典。

    HybridGateEvaluationReport及其部分内部对象
    是dataclass，而候选参数和Chunk是Pydantic模型。

    因此不能直接对report调用json.dumps()，
    必须分别转换这些对象。
    """

    if not isinstance(
        report,
        HybridGateEvaluationReport,
    ):
        raise TypeError(
            "report必须是HybridGateEvaluationReport"
        )

    results: list[dict[str, object]] = []

    for result in report.results:
        results.append(
            {
                "question_id": result.question_id,
                "question": result.question,
                "question_type": (
                    result.question_type
                ),
                "answerable": result.answerable,

                # model_dump(mode="json")把Pydantic模型
                # 转换成JSON兼容的Python字典。
                "expected_evidence": [
                    evidence.model_dump(
                        mode="json"
                    )
                    for evidence
                    in result.expected_evidence
                ],
                "retrieved_chunks": [
                    candidate.model_dump(
                        mode="json"
                    )
                    for candidate
                    in result.retrieved_chunks
                ],

                "matched_evidence_at_1": (
                    result.matched_evidence_at_1
                ),
                "matched_evidence_at_3": (
                    result.matched_evidence_at_3
                ),
                "fully_recalled_at_1": (
                    result.fully_recalled_at_1
                ),
                "fully_recalled_at_3": (
                    result.fully_recalled_at_3
                ),

                # asdict()递归地把dataclass
                # 转换成普通字典。
                "decision": asdict(
                    result.decision
                ),
                "outcome": result.outcome,
            }
        )

    return {
        "embedding_model": (
            report.embedding_model
        ),
        "collection_name": (
            report.collection_name
        ),
        "candidate_parameters": (
            report.candidate_parameters.model_dump(
                mode="json"
            )
        ),
        "gate_policy": asdict(
            report.gate_policy
        ),
        "metrics": asdict(
            report.metrics
        ),
        "results": results,
    }


def markdown_cell(
    value: str,
) -> str:
    """将文本转换成Markdown表格安全单元格。"""

    # split()再join()把换行和连续空白
    # 压缩成单个空格。
    compact = " ".join(
        value.split()
    )

    # 竖线是Markdown表格的列分隔符，
    # 因此正文中的竖线必须转义。
    return compact.replace(
        "|",
        r"\|",
    )


def format_optional_float(
    value: float | None,
) -> str:
    """格式化可能不存在的浮点指标。"""

    if value is None:
        return "无"

    return f"{value:.6f}"


def format_question_ids(
    question_ids: list[str],
) -> str:
    """把题号列表转换成人工可读文本。"""

    if not question_ids:
        return "无"

    return "、".join(question_ids)


def render_markdown_report(
    report: HybridGateEvaluationReport,
) -> str:
    """生成不包含Chunk正文的门控审计报告。"""

    if not isinstance(
        report,
        HybridGateEvaluationReport,
    ):
        raise TypeError(
            "report必须是HybridGateEvaluationReport"
        )

    metrics = report.metrics
    policy = report.gate_policy
    parameters = report.candidate_parameters
    rerank_parameters = (
        parameters.rerank_parameters
    )

    # “通过”只表示：
    #
    # 当前Gold中的无答案题没有被错误放行。
    #
    # 如果Gold中根本没有无答案题，
    # 就不能声称完成了安全验证。
    safety_passed = (
        metrics.unanswerable_question_count > 0
        and (
            metrics
            .false_accepted_unanswerable_count
            == 0
        )
    )

    rejected_answerable_ids = [
        result.question_id
        for result in report.results
        if result.outcome == "rejected_answerable"
    ]

    incomplete_evidence_ids = [
        result.question_id
        for result in report.results
        if (
            result.outcome
            == "accepted_with_incomplete_evidence"
        )
    ]

    false_accept_ids = [
        result.question_id
        for result in report.results
        if (
            result.outcome
            == "false_accept_unanswerable"
        )
    ]

    lines: list[str] = [
        "# Week 3 混合证据门控评测",
        "",
        "## 1. 评测配置",
        "",
        "| 项目 | 值 |",
        "|---|---|",
        (
            "| Embedding模型 | "
            f"`{markdown_cell(report.embedding_model)}` |"
        ),
        (
            "| Chroma Collection | "
            f"`{markdown_cell(report.collection_name)}` |"
        ),
        (
            "| 候选策略 | "
            f"`{parameters.strategy}` |"
        ),
        (
            "| 最终Top-K | "
            f"{parameters.top_k} |"
        ),
        (
            "| 单路候选深度 | "
            f"{parameters.candidate_k} |"
        ),
        (
            "| RRF排名常数 | "
            f"{parameters.rank_constant} |"
        ),
        (
            "| 查询改写版本 | "
            f"`{parameters.rewrite_version}` |"
        ),
        (
            "| 重排版本 | "
            + (
                f"`{rerank_parameters.version}` |"
                if rerank_parameters is not None
                else "无 |"
            )
        ),
        (
            "| RRF头部保留数 | "
            + (
                f"{rerank_parameters.fused_head_k} |"
                if rerank_parameters is not None
                else "无 |"
            )
        ),
        (
            "| 向量头部保留数 | "
            + (
                f"{rerank_parameters.vector_head_k} |"
                if rerank_parameters is not None
                else "无 |"
            )
        ),
        (
            "| 关键词头部保留数 | "
            + (
                f"{rerank_parameters.keyword_head_k} |"
                if rerank_parameters is not None
                else "无 |"
            )
        ),
        (
            "| 门控版本 | "
            f"`{markdown_cell(policy.version)}` |"
        ),
        "",
        "## 2. 门控参数",
        "",
        "| 参数 | 值 |",
        "|---|---:|",
        (
            "| 通用向量阈值 | "
            f"{policy.general_min_vector_similarity:.3f} |"
        ),
        (
            "| 错误码向量阈值 | "
            f"{policy.identifier_min_vector_similarity:.3f} |"
        ),
        (
            "| 型号向量阈值 | "
            f"{policy.model_min_vector_similarity:.3f} |"
        ),
        (
            "| 型号所需最少主题词数 | "
            f"{policy.min_model_lexical_matches} |"
        ),
        "",
        "## 3. 汇总指标",
        "",
        "| 指标 | 结果 |",
        "|---|---:|",
        (
            "| 可回答问题数 | "
            f"{metrics.answerable_question_count} |"
        ),
        (
            "| 无答案问题数 | "
            f"{metrics.unanswerable_question_count} |"
        ),
        (
            "| 可回答题门控放行数 | "
            f"{metrics.accepted_answerable_count} |"
        ),
        (
            "| 可回答题门控拒绝数 | "
            f"{metrics.rejected_answerable_count} |"
        ),
        (
            "| 放行且证据完整数 | "
            f"{metrics.accepted_with_full_evidence_count} |"
        ),
        (
            "| 放行但证据不完整数 | "
            f"{metrics.accepted_with_incomplete_evidence_count} |"
        ),
        (
            "| 无答案题错误放行数 | "
            f"{metrics.false_accepted_unanswerable_count} |"
        ),
        (
            "| 无答案题正确拒绝数 | "
            f"{metrics.correctly_abstained_unanswerable_count} |"
        ),
        (
            "| 可回答题门控放行率 | "
            f"{metrics.answerable_acceptance_rate:.3f} |"
        ),
        (
            "| 完整证据可用率 | "
            f"{metrics.full_evidence_eligible_rate:.3f} |"
        ),
        (
            "| 无答案错误放行率 | "
            f"{metrics.unanswerable_false_acceptance_rate:.3f} |"
        ),
        (
            "| 无答案正确拒答率 | "
            f"{metrics.unanswerable_abstention_rate:.3f} |"
        ),
        "",
        "## 4. 逐题门控结果",
        "",
        (
            "| 题号 | Gold类型 | 门控决定 | "
            "门控原因 | 结果分类 | "
            "Top-1向量相似度 | 支持放行的Chunk |"
        ),
        (
            "|---|---|---|---|---|---:|---|"
        ),
    ]

    for result in report.results:
        decision = result.decision

        supporting_chunks = (
            "<br>".join(
                markdown_cell(chunk_id)
                for chunk_id
                in decision.supporting_chunk_ids
            )
            if decision.supporting_chunk_ids
            else "无"
        )

        lines.append(
            "| "
            f"{markdown_cell(result.question_id)}"
            " | "
            + (
                "可回答"
                if result.answerable
                else "无答案"
            )
            + " | "
            + (
                "放行"
                if decision.accepted
                else "拒绝"
            )
            + " | "
            f"`{decision.reason}`"
            " | "
            f"`{result.outcome}`"
            " | "
            f"{format_optional_float(decision.top_vector_similarity)}"
            " | "
            f"{supporting_chunks}"
            " |"
        )

    lines.extend(
        [
            "",
            "## 5. 失败与风险清单",
            "",
            (
                "- 被错误拒绝的可回答题："
                f"{format_question_ids(rejected_answerable_ids)}"
            ),
            (
                "- 被放行但Top-3证据不完整的题："
                f"{format_question_ids(incomplete_evidence_ids)}"
            ),
            (
                "- 被错误放行的无答案题："
                f"{format_question_ids(false_accept_ids)}"
            ),
            "",
            "## 6. 安全结论",
            "",
            (
                "- 无答案安全检查："
                + (
                    "**通过**"
                    if safety_passed
                    else "**不通过或样本不足**"
                )
            ),
            (
                "- 当前结论只评价候选检索与门控，"
                "不代表LLM已经生成了正确答案。"
            ),
            (
                "- `完整证据可用率`是最终完整证据回答率的"
                "候选层上限，不是最终回答率。"
            ),
            (
                "- 仍需完成在线回答、结构化输出、"
                "引用白名单和降级路径评测。"
            ),
            "",
            "## 7. 数据边界",
            "",
            (
                "本Markdown仅保存题号、指标、门控原因和"
                "Chunk ID，不复制问题正文或Chunk正文。"
            ),
            (
                "包含完整候选内容的JSON报告只应保存在本地，"
                "并继续由`.gitignore`排除。"
            ),
            "",
        ]
    )

    return "\n".join(lines)


def write_evaluation_reports(
    *,
    report: HybridGateEvaluationReport,
    json_output_path: Path,
    markdown_output_path: Path,
) -> None:
    """写入本地JSON报告和脱敏Markdown报告。"""

    if not isinstance(
        report,
        HybridGateEvaluationReport,
    ):
        raise TypeError(
            "report必须是HybridGateEvaluationReport"
        )

    if not isinstance(json_output_path, Path):
        raise TypeError(
            "json_output_path必须是Path"
        )

    if not isinstance(
        markdown_output_path,
        Path,
    ):
        raise TypeError(
            "markdown_output_path必须是Path"
        )

    # 先在内存中完成转换和渲染。
    #
    # 如果报告结构非法，不应提前创建空文件。
    payload = gate_report_to_dict(report)
    markdown = render_markdown_report(report)

    json_text = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    ) + "\n"

    json_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    markdown_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_output_path.write_text(
        json_text,
        encoding="utf-8",
    )
    markdown_output_path.write_text(
        markdown,
        encoding="utf-8",
    )


def run_evaluation(
    *,
    candidate_report_path: Path,
    json_output_path: Path,
    markdown_output_path: Path,
    policy: HybridEvidenceGatePolicy,
) -> HybridGateEvaluationReport:
    """执行完整的本地门控评测流程。"""

    if not isinstance(
        policy,
        HybridEvidenceGatePolicy,
    ):
        raise TypeError(
            "policy必须是HybridEvidenceGatePolicy"
        )

    candidate_report = load_candidate_report(
        path=candidate_report_path
    )

    report = build_hybrid_gate_evaluation_report(
        candidate_report=candidate_report,
        policy=policy,
    )

    write_evaluation_reports(
        report=report,
        json_output_path=json_output_path,
        markdown_output_path=(
            markdown_output_path
        ),
    )

    return report


def print_summary(
    *,
    report: HybridGateEvaluationReport,
    markdown_output_path: Path,
) -> None:
    """在终端输出门控评测的核心结论。"""

    if not isinstance(
        report,
        HybridGateEvaluationReport,
    ):
        raise TypeError(
            "report必须是HybridGateEvaluationReport"
        )

    if not isinstance(
        markdown_output_path,
        Path,
    ):
        raise TypeError(
            "markdown_output_path必须是Path"
        )

    metrics = report.metrics

    safety_passed = (
        metrics.unanswerable_question_count > 0
        and (
            metrics
            .false_accepted_unanswerable_count
            == 0
        )
    )

    print("混合证据门控评测完成")
    print(
        "门控版本："
        f"{report.gate_policy.version}"
    )
    print(
        "可回答题门控放行率："
        f"{metrics.answerable_acceptance_rate:.3f}"
    )
    print(
        "完整证据可用率："
        f"{metrics.full_evidence_eligible_rate:.3f}"
    )
    print(
        "无答案错误放行率："
        f"{metrics.unanswerable_false_acceptance_rate:.3f}"
    )
    print(
        "无答案正确拒答率："
        f"{metrics.unanswerable_abstention_rate:.3f}"
    )
    print(
        "无答案安全检查："
        + (
            "通过"
            if safety_passed
            else "不通过或样本不足"
        )
    )

    # 即使门控安全检查通过，
    # 也不能把它描述成最终RAG回答已经通过。
    print("最终回答与引用评测：尚未完成")
    print(
        "Markdown报告："
        f"{markdown_output_path.resolve()}"
    )


def main() -> None:
    """同步命令行入口。"""

    args = parse_args()

    # 门控参数集中定义在版本化Policy中。
    #
    # CLI不单独开放无版本阈值覆盖，
    # 避免同一个版本号对应多套不同阈值。
    policy = HybridEvidenceGatePolicy()

    report = run_evaluation(
        candidate_report_path=(
            args.candidate_report
        ),
        json_output_path=args.json_output,
        markdown_output_path=(
            args.markdown_output
        ),
        policy=policy,
    )

    print_summary(
        report=report,
        markdown_output_path=(
            args.markdown_output
        ),
    )


if __name__ == "__main__":
    # 只有运行：
    #
    # python -m scripts.evaluate_hybrid_gate
    #
    # 才执行文件读取和报告写入。
    #
    # 普通import只会声明函数与常量，
    # 便于测试单独调用它们。
    main()
