"""使用真实Gold、Embedding和Chroma评测一种候选策略。

支持四种策略：

1. vector_baseline
2. hybrid_rrf
3. hybrid_rrf_rewrite
4. hybrid_rrf_rewrite_rerank

每次运行只评测一种策略，并生成对应的
JSON机器报告和Markdown人工审计报告。

后续比较脚本会读取多份JSON报告，
在相同Gold、Collection和参数约束下进行横向比较。
"""

import argparse
import asyncio
import json
from pathlib import Path

from app.config import (
    Settings,
    get_settings,
)
from app.schemas.retrieval import (
    HybridRetrievedChunk,
    RetrievedChunk,
)
from app.schemas.retrieval_strategy import (
    CandidateStrategyParameters,
    CandidateStrategyReport,
    HeadPreservingRerankParameters,
)
from app.services.embedding import (
    EmbeddingService,
)
from app.services.embedding_client import (
    create_embedding_client,
    get_embedding_model,
)
from app.services.evaluation import (
    load_gold_questions,
)
from app.services.hybrid_retrieval import (
    DEFAULT_HYBRID_CANDIDATE_K,
    DEFAULT_RRF_RANK_CONSTANT,
    HybridRetriever,
)
from app.services.head_preserving_reranking import (
    HEAD_PRESERVING_RERANK_VERSION,
    HeadPreservationPolicy,
)
from app.services.keyword_retrieval import (
    KeywordRetriever,
)
from app.services.query_rewriting import (
    DETERMINISTIC_QUERY_REWRITE_VERSION,
)
from app.services.retrieval_strategy_runner import (
    run_candidate_strategy,
)
from app.services.vector_store import (
    ChromaVectorStore,
)


# argparse的choices会把命令行输入限制为这四个值。
STRATEGY_CHOICES = (
    "vector_baseline",
    "hybrid_rrf",
    "hybrid_rrf_rewrite",
    "hybrid_rrf_rewrite_rerank",
)


def parse_positive_integer(
    raw_value: str,
) -> int:
    """把命令行文本转换成正整数。"""

    try:
        value = int(raw_value)
    except ValueError as exc:
        # ArgumentTypeError会让argparse输出规范的
        # usage信息，而不是显示普通Python堆栈。
        raise argparse.ArgumentTypeError(
            "必须是整数"
        ) from exc

    if value < 1:
        raise argparse.ArgumentTypeError(
            "必须大于或等于1"
        )

    return value


def parse_top_k(
    raw_value: str,
) -> int:
    """解析能够支持Recall@3的Top-K。"""

    value = parse_positive_integer(
        raw_value
    )

    if value < 3:
        raise argparse.ArgumentTypeError(
            "必须大于或等于3"
        )

    return value


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(
        description=(
            "使用真实Embedding、Chroma和Gold数据，"
            "运行一种Week 3候选检索策略"
        ),
    )

    parser.add_argument(
        "--strategy",
        choices=STRATEGY_CHOICES,
        required=True,
        help=(
            "vector_baseline、hybrid_rrf、"
            "hybrid_rrf_rewrite或"
            "hybrid_rrf_rewrite_rerank"
        ),
    )

    parser.add_argument(
        "--questions",
        type=Path,
        default=Path(
            "data/eval/gold_questions.jsonl"
        ),
        help="Gold Question JSONL文件",
    )

    parser.add_argument(
        "--top-k",
        type=parse_top_k,
        default=3,
        help="最终用于评测的候选数量，至少为3",
    )

    parser.add_argument(
        "--candidate-k",
        type=parse_positive_integer,
        default=(
            DEFAULT_HYBRID_CANDIDATE_K
        ),
        help="混合策略中每条召回路径的候选深度",
    )

    parser.add_argument(
        "--rank-constant",
        type=parse_positive_integer,
        default=(
            DEFAULT_RRF_RANK_CONSTANT
        ),
        help="RRF公式中的排名平滑常数",
    )

    # 默认值使用None，因为实际文件名需要根据
    # --strategy动态生成。
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="JSON报告路径；省略时根据策略自动生成",
    )

    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=None,
        help="Markdown报告路径；省略时根据策略自动生成",
    )

    return parser.parse_args()


def build_strategy_parameters(
    *,
    strategy: str,
    top_k: int,
    candidate_k: int,
    rank_constant: int,
) -> CandidateStrategyParameters:
    """根据策略构造合法且不可混用的参数契约。"""

    if strategy == "vector_baseline":
        # 纯向量策略不使用RRF，也不执行查询改写。
        #
        # 因此不能把candidate_k和rank_constant
        # 写进这份策略参数。
        return CandidateStrategyParameters(
            strategy="vector_baseline",
            top_k=top_k,
        )

    if strategy == "hybrid_rrf":
        return CandidateStrategyParameters(
            strategy="hybrid_rrf",
            top_k=top_k,
            candidate_k=candidate_k,
            rank_constant=rank_constant,
        )

    if strategy == "hybrid_rrf_rewrite":
        return CandidateStrategyParameters(
            strategy="hybrid_rrf_rewrite",
            top_k=top_k,
            candidate_k=candidate_k,
            rank_constant=rank_constant,
            rewrite_version=(
                DETERMINISTIC_QUERY_REWRITE_VERSION
            ),
        )

    if (
        strategy
        == "hybrid_rrf_rewrite_rerank"
    ):
        # 从运行时策略类取得默认深度，
        # 再写入Pydantic报告契约。
        #
        # 这样实际执行参数和报告声明参数
        # 不需要维护两套独立默认值。
        default_policy = (
            HeadPreservationPolicy()
        )

        return CandidateStrategyParameters(
            strategy=(
                "hybrid_rrf_rewrite_rerank"
            ),
            top_k=top_k,
            candidate_k=candidate_k,
            rank_constant=rank_constant,
            rewrite_version=(
                DETERMINISTIC_QUERY_REWRITE_VERSION
            ),
            rerank_parameters=(
                HeadPreservingRerankParameters(
                    version=(
                        HEAD_PRESERVING_RERANK_VERSION
                    ),
                    fused_head_k=(
                        default_policy.fused_head_k
                    ),
                    vector_head_k=(
                        default_policy.vector_head_k
                    ),
                    keyword_head_k=(
                        default_policy.keyword_head_k
                    ),
                )
            ),
        )

    # 正常命令行输入已被argparse choices限制，
    # 这个分支保护直接调用当前函数的其他代码。
    raise ValueError(
        f"不支持的候选策略：{strategy}"
    )


def build_runtime_rerank_policy(
    *,
    parameters: CandidateStrategyParameters,
) -> HeadPreservationPolicy | None:
    """把报告参数转换成HybridRetriever使用的运行时策略。"""

    if not isinstance(
        parameters,
        CandidateStrategyParameters,
    ):
        raise TypeError(
            "parameters必须是"
            "CandidateStrategyParameters"
        )

    rerank_parameters = (
        parameters.rerank_parameters
    )

    # 三种旧策略没有重排配置，
    # HybridRetriever应继续使用原始RRF截断行为。
    if rerank_parameters is None:
        return None

    # 报告可以从历史JSON恢复，
    # 因而必须确认报告声明的规则版本
    # 与当前代码真正实现的版本一致。
    if (
        rerank_parameters.version
        != HEAD_PRESERVING_RERANK_VERSION
    ):
        raise ValueError(
            "实际头部保留重排版本与"
            "策略参数声明不一致"
        )

    # Schema对象负责JSON验证和审计；
    # dataclass策略对象负责真正控制重排算法。
    return HeadPreservationPolicy(
        fused_head_k=(
            rerank_parameters.fused_head_k
        ),
        vector_head_k=(
            rerank_parameters.vector_head_k
        ),
        keyword_head_k=(
            rerank_parameters.keyword_head_k
        ),
    )


def resolve_output_paths(
    *,
    strategy: str,
    json_output: Path | None,
    markdown_output: Path | None,
) -> tuple[Path, Path]:
    """根据策略名称生成默认报告路径。"""

    # 文件名使用连字符，使其更适合普通文档命名。
    filename_strategy = strategy.replace(
        "_",
        "-",
    )

    resolved_json_output = (
        json_output
        if json_output is not None
        else Path(
            "docs/"
            f"retrieval-strategy-"
            f"{filename_strategy}.json"
        )
    )

    resolved_markdown_output = (
        markdown_output
        if markdown_output is not None
        else Path(
            "docs/"
            f"retrieval-strategy-"
            f"{filename_strategy}.md"
        )
    )

    return (
        resolved_json_output,
        resolved_markdown_output,
    )


def markdown_cell(
    value: str,
) -> str:
    """将文本转换成Markdown表格中的安全单行内容。"""

    compact = " ".join(
        value.split()
    )

    # Markdown表格使用竖线分列，
    # 正文中的竖线必须转义。
    return compact.replace(
        "|",
        r"\|",
    )


def compact_preview(
    content: str,
    *,
    max_length: int = 240,
) -> str:
    """生成候选正文的短预览。"""

    compact = markdown_cell(
        content
    )

    if len(compact) <= max_length:
        return compact

    return (
        compact[:max_length].rstrip()
        + "……"
    )


def yes_or_no(
    value: bool,
) -> str:
    """将布尔值转换成报告中的中文。"""

    return "是" if value else "否"


def strategy_display_name(
    strategy: str,
) -> str:
    """返回便于阅读的策略名称。"""

    names = {
        "vector_baseline": "纯向量基线",
        "hybrid_rrf": "关键词 + 向量 + RRF",
        "hybrid_rrf_rewrite": (
            "关键词 + 向量 + RRF + 查询改写"
        ),
        "hybrid_rrf_rewrite_rerank": (
            "关键词 + 向量 + RRF + 查询改写"
            " + 头部保留重排"
        ),
    }

    return names.get(
        strategy,
        strategy,
    )


def render_markdown_report(
    report: CandidateStrategyReport,
) -> str:
    """将结构化候选策略报告渲染成Markdown。"""

    parameters = report.parameters
    metrics = report.metrics

    full_recall_at_3_rate = (
        metrics.fully_recalled_at_3_count
        / metrics.answerable_question_count
        if metrics.answerable_question_count
        else 0.0
    )

    candidate_k_text = (
        str(parameters.candidate_k)
        if parameters.candidate_k is not None
        else "不适用"
    )

    rank_constant_text = (
        str(parameters.rank_constant)
        if parameters.rank_constant is not None
        else "不适用"
    )

    rewrite_version_text = (
        parameters.rewrite_version
        if parameters.rewrite_version is not None
        else "不适用"
    )

    rerank_parameters = (
        parameters.rerank_parameters
    )

    if rerank_parameters is None:
        rerank_version_text = "不适用"
        fused_head_k_text = "不适用"
        vector_head_k_text = "不适用"
        keyword_head_k_text = "不适用"
    else:
        rerank_version_text = (
            rerank_parameters.version
        )
        fused_head_k_text = str(
            rerank_parameters.fused_head_k
        )
        vector_head_k_text = str(
            rerank_parameters.vector_head_k
        )
        keyword_head_k_text = str(
            rerank_parameters.keyword_head_k
        )

    lines: list[str] = [
        "# Week 3 候选检索策略评测",
        "",
        "## 1. 运行配置",
        "",
        "| 项目 | 值 |",
        "|---|---|",
        (
            "| 策略 | "
            f"{strategy_display_name(parameters.strategy)} |"
        ),
        (
            "| 策略代码 | "
            f"`{parameters.strategy}` |"
        ),
        (
            "| Embedding模型 | "
            f"`{report.embedding_model}` |"
        ),
        (
            "| Chroma Collection | "
            f"`{report.collection_name}` |"
        ),
        (
            "| 最终Top-K | "
            f"{parameters.top_k} |"
        ),
        (
            "| 单路候选深度 | "
            f"{candidate_k_text} |"
        ),
        (
            "| RRF排名常数 | "
            f"{rank_constant_text} |"
        ),
        (
            "| 查询改写版本 | "
            f"{rewrite_version_text} |"
        ),
        (
            "| 头部保留重排版本 | "
            f"{rerank_version_text} |"
        ),
        (
            "| RRF头部保护深度 | "
            f"{fused_head_k_text} |"
        ),
        (
            "| 向量头部保护深度 | "
            f"{vector_head_k_text} |"
        ),
        (
            "| 关键词头部保护深度 | "
            f"{keyword_head_k_text} |"
        ),
        "",
        "## 2. 候选召回指标",
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
            "| Top-1命中证据数 | "
            f"{metrics.matched_evidence_at_1} |"
        ),
        (
            "| Top-3命中证据数 | "
            f"{metrics.matched_evidence_at_3} |"
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
            "| Top-1完整召回题数 | "
            f"{metrics.fully_recalled_at_1_count} |"
        ),
        (
            "| Top-3完整召回题数 | "
            f"{metrics.fully_recalled_at_3_count} |"
        ),
        (
            "| Top-3完整证据召回率 | "
            f"{full_recall_at_3_rate:.3f} |"
        ),
        "",
        (
            "> 本报告只评价候选召回和排序。"
            "RRF分数不是余弦相似度，"
            "因此这里暂不使用旧的单一相似度阈值"
            "计算无答案错误放行率。"
        ),
        "",
        "## 3. 逐题结果",
        "",
        (
            "| ID | 类型 | 可回答 | "
            "Top-1命中 | Top-3命中 | "
            "Top-1完整 | Top-3完整 | Top-1来源 |"
        ),
        (
            "|---|---|---|---:|---:|"
            "---|---|---|"
        ),
    ]

    for result in report.results:
        top_result = (
            result.retrieved_chunks[0]
            if result.retrieved_chunks
            else None
        )

        if top_result is None:
            top_source = "无候选"
        else:
            top_source = markdown_cell(
                f"{top_result.chunk.source_file} / "
                f"{top_result.chunk.page_or_section}"
            )

        lines.append(
            "| "
            f"{result.question_id} | "
            f"{result.question_type} | "
            f"{yes_or_no(result.answerable)} | "
            f"{result.matched_evidence_at_1} | "
            f"{result.matched_evidence_at_3} | "
            f"{yes_or_no(result.fully_recalled_at_1)} | "
            f"{yes_or_no(result.fully_recalled_at_3)} | "
            f"{top_source} |"
        )

    lines.extend(
        [
            "",
            "## 4. 逐题候选审计",
            "",
        ]
    )

    for result in report.results:
        lines.extend(
            [
                f"### {result.question_id}",
                "",
                f"- 问题：{result.question}",
                (
                    "- 问题类型："
                    f"`{result.question_type}`"
                ),
                (
                    "- Top-1完整召回："
                    f"{yes_or_no(result.fully_recalled_at_1)}"
                ),
                (
                    "- Top-3完整召回："
                    f"{yes_or_no(result.fully_recalled_at_3)}"
                ),
                "- 预期证据：",
            ]
        )

        if result.expected_evidence:
            for expected in result.expected_evidence:
                lines.append(
                    "  - "
                    f"{expected.source_file} / "
                    f"{expected.page_or_section}"
                )
        else:
            lines.append(
                "  - 无；该题被标注为无答案问题"
            )

        lines.extend(
            [
                "",
                "候选结果：",
                "",
            ]
        )

        if not result.retrieved_chunks:
            lines.extend(
                [
                    "- 无候选结果",
                    "",
                ]
            )
            continue

        for retrieved in result.retrieved_chunks:
            chunk = retrieved.chunk

            lines.extend(
                [
                    (
                        f"#### Top-{retrieved.rank}"
                    ),
                    "",
                    (
                        f"- Chunk ID：`{chunk.chunk_id}`"
                    ),
                    (
                        f"- 来源文件：{chunk.source_file}"
                    ),
                    (
                        "- 位置："
                        f"{chunk.page_or_section}"
                    ),
                ]
            )

            if isinstance(
                retrieved,
                RetrievedChunk,
            ):
                lines.append(
                    "- 余弦相似度："
                    f"{retrieved.similarity:.6f}"
                )
            elif isinstance(
                retrieved,
                HybridRetrievedChunk,
            ):
                lines.append(
                    "- RRF分数："
                    f"{retrieved.rrf_score:.6f}"
                )

                if (
                    retrieved.vector_rank is not None
                    and retrieved.vector_similarity
                    is not None
                ):
                    lines.append(
                        "- 向量路径："
                        f"Top-{retrieved.vector_rank}，"
                        "相似度 "
                        f"{retrieved.vector_similarity:.6f}"
                    )
                else:
                    lines.append(
                        "- 向量路径：未进入候选"
                    )

                if (
                    retrieved.keyword_rank is not None
                    and retrieved.keyword_score
                    is not None
                ):
                    lines.append(
                        "- 关键词路径："
                        f"Top-{retrieved.keyword_rank}，"
                        "关键词分数 "
                        f"{retrieved.keyword_score:.3f}"
                    )
                else:
                    lines.append(
                        "- 关键词路径：未进入候选"
                    )

                matched_terms = (
                    *retrieved.matched_identifiers,
                    *retrieved.matched_model_aliases,
                    *retrieved.matched_numeric_terms,
                    *retrieved.matched_lexical_terms,
                )

                lines.append(
                    "- 关键词命中："
                    + (
                        "、".join(matched_terms)
                        if matched_terms
                        else "无"
                    )
                )

            lines.extend(
                [
                    (
                        "- 正文预览："
                        f"{compact_preview(chunk.content)}"
                    ),
                    "",
                ]
            )

    lines.extend(
        [
            "## 5. 指标边界",
            "",
            (
                "- Recall@1/@3按命中的预期证据数量"
                "除以预期证据总数计算。"
            ),
            (
                "- 多跳题只有在Top-K找全全部预期证据时，"
                "才计为完整召回。"
            ),
            (
                "- 无答案问题没有Gold证据，"
                "不会因为0项命中而被标记为完整召回。"
            ),
            (
                "- 门控、可回答响应率、完整证据回答率和"
                "正确拒答率将在候选策略确定后单独评测。"
            ),
            "",
        ]
    )

    return "\n".join(lines)


def write_reports(
    *,
    report: CandidateStrategyReport,
    json_output: Path,
    markdown_output: Path,
) -> None:
    """把同一份报告写成JSON和Markdown。"""

    json_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    markdown_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # mode="json"会把datetime转换成JSON字符串。
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
        render_markdown_report(
            report
        ),
        encoding="utf-8",
    )


async def run_real_strategy(
    *,
    settings: Settings,
    questions_path: Path,
    json_output: Path,
    markdown_output: Path,
    parameters: CandidateStrategyParameters,
) -> CandidateStrategyReport:
    """创建真实依赖、执行策略并保存报告。"""

    # 在读取Gold、Chroma或调用Embedding前，
    # 将报告参数转换成真实运行时策略并校验版本。
    #
    # 配置无效时应尽早失败，避免产生外部调用费用。
    rerank_policy = (
        build_runtime_rerank_policy(
            parameters=parameters
        )
    )

    questions = load_gold_questions(
        questions_path
    )

    embedding_model = get_embedding_model(
        settings
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

    # 关键词检索需要当前Collection中的正文和元数据，
    # 不需要把全部Embedding向量加载到内存。
    all_chunks = vector_store.list_chunks()

    if not all_chunks:
        raise RuntimeError(
            "Chroma知识库中没有可评测的Chunk"
        )

    keyword_retriever = KeywordRetriever(
        chunks=all_chunks,
    )

    # 纯向量分支不会实际调用HybridRetriever，
    # 但统一执行器需要一个满足Protocol的对象。
    #
    # 此时使用默认混合参数构造即可；
    # 它不会影响纯向量策略结果。
    candidate_k = (
        parameters.candidate_k
        if parameters.candidate_k is not None
        else DEFAULT_HYBRID_CANDIDATE_K
    )

    rank_constant = (
        parameters.rank_constant
        if parameters.rank_constant is not None
        else DEFAULT_RRF_RANK_CONSTANT
    )

    hybrid_retriever = HybridRetriever(
        vector_retriever=vector_store,
        keyword_retriever=keyword_retriever,
        embedding_model=embedding_model,
        candidate_k=candidate_k,
        rank_constant=rank_constant,
        # None保持普通RRF；策略对象启用
        # 完整候选融合和头部保留重排。
        rerank_policy=rerank_policy,
    )

    client = create_embedding_client(
        settings
    )

    try:
        embedding_service = EmbeddingService(
            client=client,
            model=embedding_model,
        )

        report = await run_candidate_strategy(
            questions=questions,
            embedding_provider=embedding_service,
            vector_retriever=vector_store,
            hybrid_retriever=hybrid_retriever,
            embedding_model=embedding_model,
            collection_name=(
                settings.chroma_collection_name
            ),
            parameters=parameters,
        )

        write_reports(
            report=report,
            json_output=json_output,
            markdown_output=markdown_output,
        )

        return report

    finally:
        # 正常完成或发生异常时都关闭HTTP连接池。
        await client.close()


def print_summary(
    *,
    report: CandidateStrategyReport,
    json_output: Path,
    markdown_output: Path,
) -> None:
    """在终端输出最重要的候选评测指标。"""

    metrics = report.metrics
    parameters = report.parameters

    full_recall_at_3_rate = (
        metrics.fully_recalled_at_3_count
        / metrics.answerable_question_count
        if metrics.answerable_question_count
        else 0.0
    )

    print("候选检索策略评测完成")
    print(
        "策略："
        f"{strategy_display_name(parameters.strategy)}"
    )
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
        "Top-3完整证据召回率："
        f"{full_recall_at_3_rate:.3f}"
    )
    print(
        "JSON报告："
        f"{json_output.resolve()}"
    )
    print(
        "Markdown报告："
        f"{markdown_output.resolve()}"
    )


def main() -> None:
    """同步命令行入口。"""

    args = parse_args()

    parameters = build_strategy_parameters(
        strategy=args.strategy,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        rank_constant=args.rank_constant,
    )

    (
        json_output,
        markdown_output,
    ) = resolve_output_paths(
        strategy=args.strategy,
        json_output=args.json_output,
        markdown_output=args.markdown_output,
    )

    report = asyncio.run(
        run_real_strategy(
            settings=get_settings(),
            questions_path=args.questions,
            json_output=json_output,
            markdown_output=markdown_output,
            parameters=parameters,
        )
    )

    print_summary(
        report=report,
        json_output=json_output,
        markdown_output=markdown_output,
    )


if __name__ == "__main__":
    # 直接导入本模块不会访问网络或Chroma。
    #
    # 只有执行：
    #
    # python -m scripts.evaluate_retrieval_strategies
    #
    # 才会进入main()。
    main()
