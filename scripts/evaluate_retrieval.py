"""对持久化 ChromaDB 执行完整的检索评测。"""

# argparse 负责解析命令行参数。
import argparse

# asyncio.run() 用于从同步命令行入口
# 启动异步 Embedding 调用。
import asyncio

# json 用于生成机器可读取的 JSON 报告。
import json

# datetime 记录评测执行时间。
from datetime import datetime

# Path 提供面向对象的路径操作。
from pathlib import Path

from app.config import (
    Settings,
    get_settings,
)
from app.schemas.evaluation import (
    GoldQuestion,
    RetrievalEvaluationReport,
)
from app.services.embedding import EmbeddingService
from app.services.embedding_client import (
    create_embedding_client,
    get_embedding_model,
)
from app.services.evaluation import (
    load_gold_questions,
)
from app.services.retrieval_baseline import (
    EmbeddingProvider,
)
from app.services.retrieval_evaluation import (
    calculate_metrics,
    evaluate_question,
)
from app.services.vector_store import (
    ChromaVectorStore,
    VectorStore,
)


def parse_args() -> argparse.Namespace:
    """解析检索评测脚本的命令行参数。"""

    parser = argparse.ArgumentParser(
        description=(
            "使用真实 Embedding 和 ChromaDB "
            "评测 20 条标注问题"
        )
    )

    parser.add_argument(
        "--questions",
        type=Path,
        default=Path(
            "data/eval/gold_questions.jsonl"
        ),
        help="评测问题 JSONL 文件",
    )

    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path(
            "docs/retrieval-evaluation.json"
        ),
        help="JSON 评测报告输出位置",
    )

    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path(
            "docs/retrieval-evaluation.md"
        ),
        help="Markdown 评测报告输出位置",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="每道问题从 Chroma 召回的结果数量",
    )

    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.70,
        help="判断无答案问题错误召回的初始相似度阈值",
    )

    return parser.parse_args()


async def evaluate_retrieval(
    *,
    questions: list[GoldQuestion],
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    embedding_model: str,
    collection_name: str,
    top_k: int,
    similarity_threshold: float,
) -> RetrievalEvaluationReport:
    """执行完整评测并返回经过校验的报告对象。"""

    if not questions:
        raise ValueError(
            "评测问题列表不能为空"
        )

    # 本报告必须计算 Recall@3，
    # 因此至少需要召回三个结果。
    if top_k < 3:
        raise ValueError(
            "top_k 必须大于或等于 3"
        )

    if not -1.0 <= similarity_threshold <= 1.0:
        raise ValueError(
            "similarity_threshold 必须在 -1 到 1 之间"
        )

    # 一次性收集 20 道问题的文本。
    question_texts = [
        question.question
        for question in questions
    ]

    # 使用一个批量请求生成全部查询向量。
    #
    # 这样比循环执行 20 次 embed_query()
    # 网络开销更小，返回顺序仍由
    # EmbeddingService 根据 response.data.index 校验。
    query_vectors = (
        await embedding_provider.embed_texts(
            question_texts
        )
    )

    # 即使实际 EmbeddingService 已经检查过数量，
    # 这里仍然保护服务边界，方便 Fake 或其他实现接入。
    if len(query_vectors) != len(questions):
        raise ValueError(
            "查询向量数量与评测问题数量不一致"
        )

    question_evaluations = []

    # strict=True 要求两个序列长度完全相同。
    #
    # 如果问题有 20 个但向量只有 19 个，
    # zip 不会静默忽略最后一个问题。
    for question, query_vector in zip(
        questions,
        query_vectors,
        strict=True,
    ):
        # retrieve() 只执行本地向量检索，
        # 不会再次调用 Embedding API。
        retrieved_chunks = vector_store.retrieve(
            query_vector,
            query_embedding_model=embedding_model,
            top_k=top_k,
        )

        # 使用已经通过离线测试的“判卷器”
        # 计算该题 Top-1、Top-3 和错误召回结果。
        question_evaluation = evaluate_question(
            question=question,
            retrieved_chunks=retrieved_chunks,
            similarity_threshold=(
                similarity_threshold
            ),
        )

        question_evaluations.append(
            question_evaluation
        )

    # 把全部逐题结果汇总成 Recall 等指标。
    metrics = calculate_metrics(
        question_evaluations
    )

    return RetrievalEvaluationReport(
        # astimezone() 返回带本地时区信息的当前时间。
        generated_at=datetime.now().astimezone(),
        embedding_model=embedding_model,
        collection_name=collection_name,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
        metrics=metrics,
        results=question_evaluations,
    )


def markdown_cell(value: str) -> str:
    """把文本整理成可安全放入 Markdown 表格的单行内容。"""

    # split() 按连续空白切分，
    # join() 将换行和多余空格压缩为一个空格。
    compact = " ".join(value.split())

    # Markdown 表格使用 | 分隔列，
    # 所以正文里的 | 必须转义。
    return compact.replace("|", r"\|")


def compact_preview(
    content: str,
    *,
    max_length: int = 220,
) -> str:
    """生成报告中使用的 Chunk 正文预览。"""

    compact = markdown_cell(content)

    if len(compact) <= max_length:
        return compact

    return compact[:max_length].rstrip() + "……"


def render_markdown_report(
    report: RetrievalEvaluationReport,
) -> str:
    """把结构化评测报告转换成便于阅读的 Markdown。"""

    metrics = report.metrics

    target_reached = (
        metrics.recall_at_3 >= 0.80
    )
    target_text = (
        "达到本周目标"
        if target_reached
        else "未达到本周目标"
    )

    lines: list[str] = [
        "# Day 3–4 ChromaDB 检索评测",
        "",
        (
            "本报告由 "
            "`python -m scripts.evaluate_retrieval` "
            "自动生成。"
        ),
        "",
        f"- 生成时间：`{report.generated_at.isoformat()}`",
        f"- Embedding 模型：`{report.embedding_model}`",
        f"- Chroma Collection：`{report.collection_name}`",
        f"- Top-K：{report.top_k}",
        (
            "- 无答案错误召回相似度阈值："
            f"{report.similarity_threshold:.3f}"
        ),
        "",
        "## 汇总指标",
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
            "| 预期证据总数 | "
            f"{metrics.expected_evidence_count} |"
        ),
        (
            "| Recall@1 | "
            f"{metrics.recall_at_1:.3f} |"
        ),
        (
            "| Recall@3 | "
            f"{metrics.recall_at_3:.3f} |"
        ),
        (
            "| Top-1 完整召回问题数 | "
            f"{metrics.fully_recalled_at_1_count}/"
            f"{metrics.answerable_question_count} |"
        ),
        (
            "| Top-3 完整召回问题数 | "
            f"{metrics.fully_recalled_at_3_count}/"
            f"{metrics.answerable_question_count} |"
        ),
        (
            "| 无答案错误召回数 | "
            f"{metrics.false_recall_count}/"
            f"{metrics.unanswerable_question_count} |"
        ),
        (
            "| 无答案错误召回率 | "
            f"{metrics.false_recall_rate:.3f} |"
        ),
        "",
        "## 目标判定",
        "",
        (
            f"- 本周目标：`Recall@3 >= 0.80`"
        ),
        (
            f"- 当前结果：`{metrics.recall_at_3:.3f}`，"
            f"**{target_text}**。"
        ),
        "",
        "## 逐题结果",
        "",
        (
            "| 问题 | 类型 | Top-1 命中 | "
            "Top-3 命中 | Top-3 完整召回 | "
            "最高相似度 | 错误召回 |"
        ),
        "|---|---|---:|---:|---|---:|---|",
    ]

    for result in report.results:
        expected_count = len(
            result.expected_evidence
        )

        top_similarity = (
            f"{result.top_similarity:.3f}"
            if result.top_similarity is not None
            else "—"
        )

        lines.append(
            "| "
            f"`{result.question_id}` | "
            f"{result.question_type} | "
            f"{result.matched_evidence_at_1}/"
            f"{expected_count} | "
            f"{result.matched_evidence_at_3}/"
            f"{expected_count} | "
            f"{'是' if result.fully_recalled_at_3 else '否'} | "
            f"{top_similarity} | "
            f"{'是' if result.false_recall else '否'} |"
        )

    # 失败包括：
    # 1. 可回答问题没有在 Top-3 找全证据；
    # 2. 无答案问题超过阈值，被错误接受。
    failures = [
        result
        for result in report.results
        if (
            (
                result.answerable
                and not result.fully_recalled_at_3
            )
            or (
                not result.answerable
                and result.false_recall
            )
        )
    ]

    lines.extend(
        [
            "",
            "## 失败案例",
            "",
        ]
    )

    if not failures:
        lines.append(
            "本次评测没有发现符合当前定义的失败案例。"
        )
    else:
        for result in failures:
            lines.extend(
                [
                    f"### {result.question_id}",
                    "",
                    (
                        "- 问题："
                        f"{markdown_cell(result.question)}"
                    ),
                ]
            )

            if result.answerable:
                lines.append(
                    "- 失败原因：Top-3 未找全全部预期证据。"
                )

                lines.append("- 预期证据：")

                for expected in result.expected_evidence:
                    lines.append(
                        "  - "
                        f"`{expected.source_file}`；"
                        f"`{expected.page_or_section}`"
                    )
            else:
                lines.append(
                    "- 失败原因：知识库无答案，但最高相似度"
                    "达到当前接受阈值。"
                )

            lines.append("- 实际 Top-K：")

            for retrieved in result.retrieved_chunks:
                chunk = retrieved.chunk

                lines.append(
                    "  - "
                    f"Top-{retrieved.rank}；"
                    f"相似度 `{retrieved.similarity:.3f}`；"
                    f"`{chunk.source_file}`；"
                    f"`{chunk.page_or_section}`；"
                    f"Chunk `{chunk.chunk_id}`；"
                    f"预览：{compact_preview(chunk.content)}"
                )

            lines.append("")

    lines.extend(
        [
            "## 指标解释",
            "",
            (
                "- Recall@K = Top-K 中命中的预期证据数"
                " / 全部预期证据数。"
            ),
            (
                "- 完整召回表示一道问题需要的全部证据"
                "都出现在 Top-K 中。"
            ),
            (
                "- 无答案错误召回率表示无答案问题中，"
                "最高相似度达到阈值的比例。"
            ),
            (
                "- 相似度阈值 `0.70` 目前只是初始值，"
                "后续应结合可回答和无答案问题的分数分布调整。"
            ),
        ]
    )

    return "\n".join(lines) + "\n"


def write_reports(
    *,
    report: RetrievalEvaluationReport,
    json_output: Path,
    markdown_output: Path,
) -> None:
    """将同一报告分别写为 JSON 和 Markdown。"""

    # parents=True 会创建缺失的父目录；
    # exist_ok=True 表示目录已存在时不报错。
    json_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    markdown_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # model_dump(mode="json") 将 datetime 等对象
    # 转换成 JSON 可以表示的字符串或基本类型。
    json_data = report.model_dump(
        mode="json"
    )

    json_output.write_text(
        json.dumps(
            json_data,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    markdown_output.write_text(
        render_markdown_report(report),
        encoding="utf-8",
    )


async def run_real_evaluation(
    *,
    settings: Settings,
    questions_path: Path,
    json_output: Path,
    markdown_output: Path,
    top_k: int,
    similarity_threshold: float,
) -> RetrievalEvaluationReport:
    """创建真实依赖，运行评测并保存报告。"""

    questions = load_gold_questions(
        questions_path
    )

    embedding_model = get_embedding_model(
        settings
    )

    client = create_embedding_client(
        settings
    )

    try:
        embedding_service = EmbeddingService(
            client=client,
            model=embedding_model,
        )

        vector_store = ChromaVectorStore(
            persist_directory=(
                settings.chroma_persist_directory
            ),
            collection_name=(
                settings.chroma_collection_name
            ),
            embedding_model=embedding_model,
        )

        report = await evaluate_retrieval(
            questions=questions,
            embedding_provider=embedding_service,
            vector_store=vector_store,
            embedding_model=embedding_model,
            collection_name=(
                settings.chroma_collection_name
            ),
            top_k=top_k,
            similarity_threshold=(
                similarity_threshold
            ),
        )

        write_reports(
            report=report,
            json_output=json_output,
            markdown_output=markdown_output,
        )

        return report

    finally:
        # 即使中途发生异常，也关闭 HTTP 连接池。
        await client.close()


def print_summary(
    report: RetrievalEvaluationReport,
) -> None:
    """在终端打印最重要的评测结果。"""

    metrics = report.metrics

    print("检索评测完成")
    print(
        "可回答问题数："
        f"{metrics.answerable_question_count}"
    )
    print(
        "无答案问题数："
        f"{metrics.unanswerable_question_count}"
    )
    print(
        f"Recall@1：{metrics.recall_at_1:.3f}"
    )
    print(
        f"Recall@3：{metrics.recall_at_3:.3f}"
    )
    print(
        "无答案错误召回率："
        f"{metrics.false_recall_rate:.3f}"
    )


def main() -> None:
    """命令行同步入口。"""

    args = parse_args()
    settings = get_settings()

    report = asyncio.run(
        run_real_evaluation(
            settings=settings,
            questions_path=args.questions,
            json_output=args.json_output,
            markdown_output=args.markdown_output,
            top_k=args.top_k,
            similarity_threshold=(
                args.similarity_threshold
            ),
        )
    )

    print_summary(report)


if __name__ == "__main__":
    main()