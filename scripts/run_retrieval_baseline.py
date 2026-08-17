"""运行 TXT/Markdown 的真实 Embedding 内存检索基线。"""

# argparse 解析 --source-dir、--top-k 等命令行参数。
import argparse

# asyncio.run() 负责从同步命令行入口启动异步代码。
import asyncio

from pathlib import Path

from app.config import get_settings
from app.schemas.evaluation import (
    ExpectedEvidence,
    GoldQuestion,
)
from app.schemas.retrieval import RetrievedChunk
from app.services.chunking import TextChunker
from app.services.document_preparation import (
    TextDocumentPreparationService,
)
from app.services.embedding import EmbeddingService
from app.services.evaluation import load_gold_questions
from app.services.embedding_client import (
    create_embedding_client,
    get_embedding_model,
)
from app.services.retrieval import InMemoryRetriever
from app.services.retrieval_baseline import (
    EmbeddingCache,
    embed_document_chunks,
)


# Day 1-2 要对比的两套切分参数。
CHUNKING_CONFIGURATIONS = [
    ("450/80", 450, 80),
    ("800/120", 800, 120),
]


# 当前基线只使用已经实现 Loader 的文件类型。
SUPPORTED_SUFFIXES = {
    ".txt",
    ".md",
}


# 类型别名：
#
# 第一层 key 是配置名，例如 "450/80"；
# 第二层 key 是 question_id，例如 "q001"；
# 最终 value 是该问题的 Top-K 检索结果。
BaselineResults = dict[
    str,
    dict[str, list[RetrievedChunk]],
]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(
        description=(
            "运行两套 Chunk 参数的真实 Embedding "
            "内存检索基线"
        )
    )

    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/source"),
        help="原始语料目录",
    )

    parser.add_argument(
        "--eval-file",
        type=Path,
        default=Path(
            "data/eval/gold_questions.jsonl"
        ),
        help="检索评测问题 JSONL 文件",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/retrieval-baseline.md"
        ),
        help="Markdown 报告输出路径",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="每个问题返回的最大结果数",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="每次 Embedding 请求包含的最大文本数",
    )

    return parser.parse_args()


def find_supported_files(
    source_dir: Path,
) -> list[Path]:
    """查找并稳定排序当前支持的文本语料。"""

    if not source_dir.is_dir():
        raise NotADirectoryError(
            f"语料目录不存在：{source_dir}"
        )

    files = [
        path
        for path in source_dir.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_SUFFIXES
        )
    ]

    return sorted(
        files,
        key=lambda path: path.name.casefold(),
    )


def select_supported_questions(
    questions: list[GoldQuestion],
    *,
    available_source_files: set[str],
) -> list[GoldQuestion]:
    """选择当前 TXT/Markdown 语料能够评测的问题。"""

    selected: list[GoldQuestion] = []

    for question in questions:
        # 本阶段先评测有明确证据的可回答问题。
        #
        # 无答案问题需要相似度阈值和拒答策略，
        # 属于后续 RAG 问答阶段。
        if not question.answerable:
            continue

        # all(...) 表示每一项预期证据都必须来自
        # 当前已经加载的 TXT/Markdown 文件。
        #
        # PDF 问题会暂时被过滤掉。
        if all(
            evidence.source_file
            in available_source_files
            for evidence in question.expected_evidence
        ):
            selected.append(question)

    return selected


def normalize_location(location: str) -> str:
    """统一章节位置格式，便于比较标注与实际结果。"""

    # casefold() 是比 lower() 更完整的大小写归一化方法，
    # 适用于无视大小写的文本比较。
    normalized = location.casefold()

    # split() 不传参数时按照连续空白切分，
    # join() 再使用一个空格连接。
    normalized = " ".join(
        normalized.split()
    )

    # 标注和实际结果可能都包含 section:，
    # 比较章节正文时不需要该固定前缀。
    if normalized.startswith("section:"):
        normalized = normalized.removeprefix(
            "section:"
        ).strip()

    return normalized


def evidence_matches(
    result: RetrievedChunk,
    expected: ExpectedEvidence,
) -> bool:
    """判断一个检索结果是否命中一项预期证据。"""

    # 文件名必须相同。
    if (
        result.chunk.source_file.casefold()
        != expected.source_file.casefold()
    ):
        return False

    actual_location = normalize_location(
        result.chunk.page_or_section
    )

    # 部分标注使用：
    #
    # section: 10.1 TEST-NET-001/通过标准
    #
    # “/”表示父章节和子章节都必须出现在实际章节路径中。
    expected_parts = [
        normalize_location(part)
        for part in expected.page_or_section.split("/")
    ]

    return all(
        part in actual_location
        for part in expected_parts
    )


def count_matched_evidence(
    results: list[RetrievedChunk],
    expected_evidence: list[ExpectedEvidence],
) -> int:
    """统计一组 Top-K 结果命中了多少项预期证据。"""

    # 每项预期证据最多计数一次。
    #
    # 即使两个重叠 Chunk 都命中同一章节，
    # 也不能把一项证据计算成两项。
    return sum(
        1
        for expected in expected_evidence
        if any(
            evidence_matches(result, expected)
            for result in results
        )
    )


def compact_result_preview(
    content: str,
    *,
    max_length: int = 180,
) -> str:
    """生成适合报告展示的单行正文预览。"""

    compacted = " ".join(
        content.split()
    )

    # 避免正文中的反引号破坏 Markdown 行内代码。
    compacted = compacted.replace(
        "`",
        "'",
    )

    if len(compacted) <= max_length:
        return compacted

    return (
        compacted[:max_length]
        + "……"
    )


def build_report(
    *,
    questions: list[GoldQuestion],
    results_by_configuration: BaselineResults,
    chunk_counts: dict[str, int],
    embedding_model: str,
    top_k: int,
) -> str:
    """把两套配置的检索结果转换成 Markdown 报告。"""

    lines: list[str] = [
        "# Day 1–2 内存检索基线",
        "",
        "本报告由 `scripts.run_retrieval_baseline` 自动生成。",
        "",
        "当前语料范围仅包含已经实现 Loader 的 "
        "TXT 和 Markdown；PDF 尚未进入本轮结果。",
        "",
        f"- Embedding 模型：`{embedding_model}`",
        f"- 评测问题数：{len(questions)}",
        f"- Top-K：{top_k}",
        "",
        "## 汇总",
        "",
        (
            "| 切分参数 | Chunk 数 | 命中证据数"
            " | 预期证据数 | Evidence Recall"
            f"@{top_k} | 完整召回问题数 |"
        ),
        "|---|---:|---:|---:|---:|---:|",
    ]

    for (
        configuration_name,
        question_results,
    ) in results_by_configuration.items():
        matched_total = 0
        expected_total = 0
        fully_recalled_questions = 0

        for question in questions:
            results = question_results[
                question.question_id
            ]

            matched = count_matched_evidence(
                results,
                question.expected_evidence,
            )

            expected_count = len(
                question.expected_evidence
            )

            matched_total += matched
            expected_total += expected_count

            if matched == expected_count:
                fully_recalled_questions += 1

        recall = (
            matched_total / expected_total
            if expected_total
            else 0.0
        )

        lines.append(
            "| "
            f"{configuration_name} | "
            f"{chunk_counts[configuration_name]} | "
            f"{matched_total} | "
            f"{expected_total} | "
            f"{recall:.3f} | "
            f"{fully_recalled_questions}"
            f"/{len(questions)} |"
        )

    lines.extend(
        [
            "",
            "## 逐题 Top-K 结果",
            "",
        ]
    )

    for question in questions:
        lines.extend(
            [
                (
                    f"### {question.question_id}："
                    f"{question.question}"
                ),
                "",
                f"- 问题类型：`{question.question_type}`",
                "- 预期证据：",
            ]
        )

        for expected in question.expected_evidence:
            lines.append(
                "  - "
                f"`{expected.source_file}`；"
                f"`{expected.page_or_section}`"
            )

        lines.append("")

        for (
            configuration_name,
            question_results,
        ) in results_by_configuration.items():
            results = question_results[
                question.question_id
            ]

            matched_count = count_matched_evidence(
                results,
                question.expected_evidence,
            )

            lines.extend(
                [
                    (
                        f"#### 参数 {configuration_name}"
                    ),
                    "",
                    (
                        f"证据命中：{matched_count}/"
                        f"{len(question.expected_evidence)}"
                    ),
                    "",
                ]
            )

            for result in results:
                matched = any(
                    evidence_matches(result, expected)
                    for expected
                    in question.expected_evidence
                )

                match_text = (
                    "命中预期证据"
                    if matched
                    else "未命中预期证据"
                )

                preview = compact_result_preview(
                    result.chunk.content
                )

                lines.append(
                    "- "
                    f"Top-{result.rank}；"
                    f"相似度：{result.similarity:.6f}；"
                    f"{match_text}；"
                    f"来源：`{result.chunk.source_file}`；"
                    f"位置：`{result.chunk.page_or_section}`；"
                    f"Chunk：`{result.chunk.chunk_id}`；"
                    f"预览：{preview}"
                )

            lines.append("")

    lines.extend(
        [
            "## 失败问题与人工分析",
            "",
        ]
    )

    for (
        configuration_name,
        question_results,
    ) in results_by_configuration.items():
        failed_question_ids = [
            question.question_id
            for question in questions
            if count_matched_evidence(
                question_results[
                    question.question_id
                ],
                question.expected_evidence,
            )
            < len(question.expected_evidence)
        ]

        failed_text = (
            "、".join(failed_question_ids)
            if failed_question_ids
            else "无"
        )

        lines.extend(
            [
                f"### 参数 {configuration_name}",
                "",
                f"- 未完整召回的问题：{failed_text}",
                "- 失败原因分析：待结合 Top-K 正文人工填写。",
                "",
            ]
        )

    lines.extend(
        [
            "## 阶段结论",
            "",
            "- 哪套参数的证据召回更高：待填写。",
            "- 哪套参数更容易返回完整操作步骤：待填写。",
            "- 是否存在过长 Chunk 混合多个主题：待填写。",
            "- PDF 加入后是否需要重新选择参数：是。",
            "",
        ]
    )

    return "\n".join(lines)


async def run_baseline(
    args: argparse.Namespace,
) -> str:
    """执行真实 Embedding 和两套内存检索实验。"""

    if args.top_k < 1:
        raise ValueError(
            "top_k 必须大于或等于 1"
        )

    if args.batch_size < 1:
        raise ValueError(
            "batch_size 必须大于或等于 1"
        )

    source_files = find_supported_files(
        args.source_dir
    )

    if not source_files:
        raise ValueError(
            "语料目录中没有可处理的 TXT 或 Markdown"
        )

    all_questions = load_gold_questions(
        args.eval_file
    )

    available_source_files = {
        path.name
        for path in source_files
    }

    questions = select_supported_questions(
        all_questions,
        available_source_files=available_source_files,
    )

    if not questions:
        raise ValueError(
            "没有适用于当前 TXT/Markdown 语料的评测问题"
        )

    # Settings() 从 .env 或系统环境变量读取配置。
    settings = get_settings()

    # get_embedding_model() 检查 EMBEDDING_MODEL。
    embedding_model = get_embedding_model(
        settings
    )

    # 使用 Embedding 独立的 API Key 和 Base URL
    client = create_embedding_client(settings)

    try:
        provider = EmbeddingService(
            client=client,
            model=embedding_model,
        )

        # 问题文本与 Chunk 参数无关，
        # 所以只生成一次查询向量，
        # 两套参数共享这些查询向量。
        query_vectors = await provider.embed_texts(
            [
                question.question
                for question in questions
            ]
        )

        # 文档向量缓存由两套配置共同使用。
        #
        # 完全相同的 Chunk 正文不会重复调用 Embedding。
        cache: EmbeddingCache = {}

        results_by_configuration: BaselineResults = {}
        chunk_counts: dict[str, int] = {}

        for (
            configuration_name,
            chunk_size,
            overlap,
        ) in CHUNKING_CONFIGURATIONS:
            preparation_service = (
                TextDocumentPreparationService(
                    chunker=TextChunker(
                        chunk_size=chunk_size,
                        overlap=overlap,
                    )
                )
            )

            document_chunks = []

            for source_file in source_files:
                document_chunks.extend(
                    preparation_service.prepare(
                        source_file
                    )
                )

            chunk_counts[
                configuration_name
            ] = len(document_chunks)

            embedded_chunks = await embed_document_chunks(
                document_chunks,
                provider=provider,
                embedding_model=embedding_model,
                cache=cache,
                batch_size=args.batch_size,
            )

            retriever = InMemoryRetriever(
                embedded_chunks
            )

            question_results: dict[
                str,
                list[RetrievedChunk],
            ] = {}

            # strict=True 保证问题和查询向量严格一一对应。
            for question, query_vector in zip(
                questions,
                query_vectors,
                strict=True,
            ):
                question_results[
                    question.question_id
                ] = retriever.retrieve(
                    query_vector,
                    query_embedding_model=(
                        embedding_model
                    ),
                    top_k=args.top_k,
                )

            results_by_configuration[
                configuration_name
            ] = question_results

        return build_report(
            questions=questions,
            results_by_configuration=(
                results_by_configuration
            ),
            chunk_counts=chunk_counts,
            embedding_model=embedding_model,
            top_k=args.top_k,
        )
    finally:
        # 无论实验成功或发生异常，
        # 都关闭 AsyncOpenAI 内部的异步 HTTP 连接。
        await client.close()


def main() -> None:
    """命令行同步入口。"""

    args = parse_args()

    # asyncio.run() 创建事件循环，
    # 执行异步 run_baseline()，
    # 完成后关闭该事件循环。
    report = asyncio.run(
        run_baseline(args)
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        report,
        encoding="utf-8",
    )

    print(
        f"检索基线报告已生成：{args.output}"
    )


if __name__ == "__main__":
    main()