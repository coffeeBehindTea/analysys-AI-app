"""读取三份候选策略报告并生成正式对比文档。

本脚本只负责文件和命令行边界：

1. 解析三份输入JSON报告路径；
2. 使用Pydantic重新校验报告；
3. 调用候选策略对比Service；
4. 把Markdown写入目标文件；
5. 在终端打印简短结果。

本脚本不访问Embedding、Chroma或LLM，
也不会重新运行检索评测。
"""

import argparse
from pathlib import Path

from app.schemas.retrieval_strategy import (
    CandidateStrategyReport,
    RetrievalStrategyName,
)
from app.services.retrieval_strategy_comparison import (
    CandidateStrategyComparison,
    STRATEGY_ORDER,
    compare_candidate_strategy_reports,
    render_candidate_strategy_comparison,
)


# 三份已经生成的候选策略JSON报告。
#
# 使用Path而不是普通字符串，
# 方便后续调用read_text()、resolve()等路径方法。
DEFAULT_VECTOR_REPORT = Path(
    "docs/retrieval-strategy-vector-baseline.json"
)
DEFAULT_HYBRID_REPORT = Path(
    "docs/retrieval-strategy-hybrid-rrf.json"
)
DEFAULT_REWRITE_REPORT = Path(
    "docs/retrieval-strategy-hybrid-rrf-rewrite.json"
)

# Week 3验收清单要求的正式对比报告路径。
DEFAULT_OUTPUT = Path(
    "docs/retrieval-strategy-comparison.md"
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(
        description=(
            "读取三份候选检索策略JSON报告，"
            "校验可比性并生成Markdown对比报告"
        )
    )

    parser.add_argument(
        "--vector-report",
        type=Path,
        default=DEFAULT_VECTOR_REPORT,
        help="纯向量基线JSON报告路径",
    )

    parser.add_argument(
        "--hybrid-report",
        type=Path,
        default=DEFAULT_HYBRID_REPORT,
        help="关键词、向量和RRF策略JSON报告路径",
    )

    parser.add_argument(
        "--rewrite-report",
        type=Path,
        default=DEFAULT_REWRITE_REPORT,
        help="混合检索加查询改写策略JSON报告路径",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="生成的Markdown对比报告路径",
    )

    return parser.parse_args()


def load_strategy_report(
    *,
    path: Path,
    expected_strategy: RetrievalStrategyName,
) -> CandidateStrategyReport:
    """读取并校验一份候选策略JSON报告。"""

    # 类型注解不会在普通Python调用时自动执行，
    # 所以公共函数仍然需要运行时检查。
    if not isinstance(path, Path):
        raise TypeError(
            "path必须是Path"
        )

    # expected_strategy虽然使用Literal类型注解，
    # 但运行时仍可能收到普通非法字符串。
    if expected_strategy not in STRATEGY_ORDER:
        raise ValueError(
            "expected_strategy不是受支持的候选策略"
        )

    # read_text()打开文件、读取全部文本并关闭文件。
    #
    # 明确指定UTF-8，避免Windows使用系统默认编码
    # 读取中文报告时出现乱码。
    json_text = path.read_text(
        encoding="utf-8"
    )

    # model_validate_json()完成两个步骤：
    #
    # 1. 把JSON文本解析成Python数据；
    # 2. 根据CandidateStrategyReport及其嵌套模型，
    #    校验字段、类型、策略参数和汇总指标。
    #
    # 因此不能直接使用json.loads()后就相信文件内容。
    report = (
        CandidateStrategyReport
        .model_validate_json(
            json_text
        )
    )

    # 参数名称不仅是路径标签，也属于调用契约。
    #
    # 例如传给--vector-report的文件，
    # 内部必须确实声明vector_baseline，
    # 不能只因为文件名像基线就相信它。
    if (
        report.parameters.strategy
        != expected_strategy
    ):
        raise ValueError(
            "报告策略与参数位置不一致："
            f"{path}声明为"
            f"{report.parameters.strategy}，"
            f"预期为{expected_strategy}"
        )

    return report


def write_comparison_report(
    *,
    comparison: CandidateStrategyComparison,
    output_path: Path,
) -> None:
    """把比较结果写成UTF-8 Markdown文件。"""

    if not isinstance(output_path, Path):
        raise TypeError(
            "output_path必须是Path"
        )

    # 先完成渲染和类型检查。
    #
    # 如果comparison非法，
    # 不应该提前创建一个空的输出文件。
    markdown = (
        render_candidate_strategy_comparison(
            comparison
        )
    )

    # parent表示目标文件所在目录。
    #
    # parents=True：
    # 父目录的上级目录不存在时也一起创建。
    #
    # exist_ok=True：
    # docs目录已经存在时不抛出异常。
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        markdown,
        encoding="utf-8",
    )


def run_comparison(
    *,
    vector_report_path: Path,
    hybrid_report_path: Path,
    rewrite_report_path: Path,
    output_path: Path,
) -> CandidateStrategyComparison:
    """执行完整的本地报告读取、比较和写入流程。"""

    vector_report = load_strategy_report(
        path=vector_report_path,
        expected_strategy="vector_baseline",
    )

    hybrid_report = load_strategy_report(
        path=hybrid_report_path,
        expected_strategy="hybrid_rrf",
    )

    rewrite_report = load_strategy_report(
        path=rewrite_report_path,
        expected_strategy=(
            "hybrid_rrf_rewrite"
        ),
    )

    # compare_candidate_strategy_reports()负责：
    #
    # 1. 检查三份报告的实验条件；
    # 2. 计算相对基线变化；
    # 3. 选择候选层胜者；
    # 4. 识别改进、退步和未解决题。
    comparison = (
        compare_candidate_strategy_reports(
            [
                vector_report,
                hybrid_report,
                rewrite_report,
            ]
        )
    )

    write_comparison_report(
        comparison=comparison,
        output_path=output_path,
    )

    return comparison


def print_summary(
    *,
    comparison: CandidateStrategyComparison,
    output_path: Path,
) -> None:
    """向终端打印最重要的对比结论。"""

    if not isinstance(
        comparison,
        CandidateStrategyComparison,
    ):
        raise TypeError(
            "comparison必须是"
            "CandidateStrategyComparison"
        )

    if not isinstance(output_path, Path):
        raise TypeError(
            "output_path必须是Path"
        )

    # 从三行策略指标中找到候选胜者对应的那一行。
    #
    # next()返回生成器产生的第一项。
    winner_row = next(
        row
        for row in comparison.strategy_rows
        if (
            row.strategy
            == comparison.candidate_winner
        )
    )

    print("候选检索策略对比完成")
    print(
        "候选层胜者："
        f"{comparison.candidate_winner}"
    )
    print(
        "Recall@1："
        f"{winner_row.recall_at_1:.3f}"
    )
    print(
        "Recall@3："
        f"{winner_row.recall_at_3:.3f}"
    )
    print(
        "Top-3完整证据召回率："
        f"{winner_row.fully_recalled_at_3_rate:.3f}"
    )
    print(
        "相对基线改进题："
        + (
            "、".join(
                comparison.improved_question_ids
            )
            if comparison.improved_question_ids
            else "无"
        )
    )
    print(
        "相对基线退步题："
        + (
            "、".join(
                comparison.regressed_question_ids
            )
            if comparison.regressed_question_ids
            else "无"
        )
    )
    print(
        "候选胜者未解决题："
        + (
            "、".join(
                comparison.unresolved_question_ids
            )
            if comparison.unresolved_question_ids
            else "无"
        )
    )

    # 明确阻止读者把候选召回胜者
    # 误解为可以直接接入生产环境。
    print("无答案安全评测：尚未完成")
    print(
        "Markdown报告："
        f"{output_path.resolve()}"
    )


def main() -> None:
    """同步命令行入口。"""

    args = parse_args()

    comparison = run_comparison(
        vector_report_path=(
            args.vector_report
        ),
        hybrid_report_path=(
            args.hybrid_report
        ),
        rewrite_report_path=(
            args.rewrite_report
        ),
        output_path=args.output,
    )

    print_summary(
        comparison=comparison,
        output_path=args.output,
    )


if __name__ == "__main__":
    # 导入本模块时只声明函数和常量。
    #
    # 只有执行：
    #
    # python -m scripts.compare_retrieval_strategies
    #
    # 才会读取JSON并写入Markdown。
    main()