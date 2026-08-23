"""运行并展示诊断Prompt A/B对照实验。

本模块最终负责：

1. 接收命令行实验参数；
2. 使用相同请求和相同证据比较两个Prompt版本；
3. 生成经过Pydantic校验的结构化报告；
4. 输出JSON、Markdown和终端摘要。

当前阶段先实现：

1. 命令行参数解析；
2. Markdown报告渲染；
3. JSON和Markdown文件写入；
4. 终端摘要输出。

真实Embedding、Chroma检索和LLM调用将在下一阶段接入。
"""

import argparse
import asyncio
from collections.abc import Sequence
from pathlib import Path
import re

from app.schemas.diagnosis_prompt_comparison import (
    DiagnosisPromptComparisonReport,
    DiagnosisPromptRunResult,
    DiagnosisPromptVariantMetrics,
)
from app.schemas.diagnostics import (
    DiagnosisRequest,
)
from app.services.diagnosis_prompt_comparison import (
    VersionedDiagnosisDraftProvider,
    build_diagnosis_prompt_comparison_report,
    compare_diagnosis_prompt_case,
)
from app.services.diagnosis_service import (
    DiagnosisRetrievalProvider,
    build_diagnosis_retrieval_query,
    build_diagnosis_retrieval_scope,
)
from app.services.gated_hybrid_retrieval import (
    GatedHybridRetrievalResult,
)
from app.config import get_settings
from app.dependencies import (
    build_diagnosis_retrieval_provider,
)
from app.services.diagnosis_generation import (
    DIAGNOSIS_TEMPERATURE,
    OpenAIDiagnosisDraftProvider,
)
from app.services.diagnosis_prompts import (
    BASIC_DIAGNOSIS_PROMPT_VARIANT,
    EVIDENCE_DIAGNOSIS_PROMPT_VARIANT,
)
from app.services.embedding_client import (
    create_embedding_client,
    get_embedding_model,
)
from app.services.llm_client import (
    create_llm_client,
    get_llm_model,
)
from app.services.vector_store import (
    ChromaVectorStore,
)


# 默认执行一个固定的诊断Prompt冒烟案例。
#
# dpc表示diagnosis prompt comparison，
# 后面的三位数字是稳定案例编号。
DEFAULT_CASE_ID = "dpc001"

DEFAULT_ROBOT_ID = "diagnostic-robot"

DEFAULT_SYMPTOM = (
    "机器人网络恢复后仍然没有继续执行任务"
)

DEFAULT_LOG_EXCERPT = (
    "ERR-NET-4001 heartbeat timeout exceeded "
    "1500 ms; network recovered but task did not resume"
)

# 两个Prompt版本必须使用相同的最终证据数量。
DEFAULT_TOP_K = 3

DEFAULT_JSON_OUTPUT = Path(
    "docs/diagnosis-prompt-comparison.json"
)

DEFAULT_MARKDOWN_OUTPUT = Path(
    "docs/diagnosis-prompt-comparison.md"
)

# 案例编号必须使用dpc加三位数字。
CASE_ID_PATTERN = re.compile(
    r"^dpc\d{3}$"
)


def non_blank_text(
    value: str,
) -> str:
    """拒绝空字符串和纯空白命令行参数。"""

    stripped_value = value.strip()

    if not stripped_value:
        # ArgumentTypeError会被argparse转换成
        # 用户可读的命令行参数错误。
        raise argparse.ArgumentTypeError(
            "参数不能是空字符串"
        )

    return stripped_value


def positive_int(
    value: str,
) -> int:
    """把命令行文本解析成正整数。"""

    try:
        parsed_value = int(value)
    except ValueError as exc:
        # from exc保留原始异常因果链，
        # 便于调试参数为什么无法转换。
        raise argparse.ArgumentTypeError(
            "参数必须是整数"
        ) from exc

    if parsed_value <= 0:
        raise argparse.ArgumentTypeError(
            "参数必须大于0"
        )

    return parsed_value


def diagnosis_case_id(
    value: str,
) -> str:
    """校验诊断Prompt对照案例编号。"""

    stripped_value = non_blank_text(value)

    if CASE_ID_PATTERN.fullmatch(
        stripped_value
    ) is None:
        raise argparse.ArgumentTypeError(
            "case_id必须使用dpc加三位数字，"
            "例如dpc001"
        )

    return stripped_value


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """解析诊断Prompt对照实验的命令行参数。

    argv为None时，argparse读取真实命令行。

    单元测试传入list时，不读取pytest自己的命令行，
    从而可以稳定验证各种参数组合。
    """

    parser = argparse.ArgumentParser(
        description=(
            "使用相同请求、相同检索证据和相同模型，"
            "比较基础诊断Prompt与证据优先Prompt"
        ),
    )

    parser.add_argument(
        "--case-id",
        type=diagnosis_case_id,
        default=DEFAULT_CASE_ID,
        help=(
            "实验案例编号，格式为dpc加三位数字"
        ),
    )

    parser.add_argument(
        "--robot-id",
        type=non_blank_text,
        default=DEFAULT_ROBOT_ID,
        help="脱敏后的机器人编号",
    )

    parser.add_argument(
        "--symptom",
        type=non_blank_text,
        default=DEFAULT_SYMPTOM,
        help="用户观察到的故障现象",
    )

    parser.add_argument(
        "--log-excerpt",
        type=non_blank_text,
        default=DEFAULT_LOG_EXCERPT,
        help="经过脱敏处理的日志摘要",
    )

    parser.add_argument(
        "--top-k",
        type=positive_int,
        default=DEFAULT_TOP_K,
        help=(
            "证据门控通过后提供给两个Prompt的"
            "相同候选数量"
        ),
    )

    parser.add_argument(
        "--json-output",
        type=Path,
        default=DEFAULT_JSON_OUTPUT,
        help="结构化JSON报告输出路径",
    )

    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=DEFAULT_MARKDOWN_OUTPUT,
        help="可读Markdown报告输出路径",
    )

    return parser.parse_args(argv)


async def run_prompt_comparison_experiment(
    *,
    case_id: str,
    request: DiagnosisRequest,
    top_k: int,
    retrieval_provider: DiagnosisRetrievalProvider,
    basic_provider: VersionedDiagnosisDraftProvider,
    evidence_first_provider: (
        VersionedDiagnosisDraftProvider
    ),
    llm_model: str,
    temperature: float,
) -> DiagnosisPromptComparisonReport:
    """使用一次门控检索结果运行两个诊断Prompt。

    本函数不创建真实客户端，也不直接写报告文件。

    调用方负责注入：

    1. 检索Provider；
    2. 基础Prompt Provider；
    3. 证据优先Prompt Provider；
    4. 模型名称和temperature。

    这样单元测试可以注入Fake对象，
    正式命令行入口则可以注入真实实现。
    """

    # bool是int的子类：
    #
    # isinstance(True, int) == True
    #
    # 但True不能表示有意义的检索候选数量，
    # 所以要先单独排除bool。
    if (
        isinstance(top_k, bool)
        or not isinstance(top_k, int)
    ):
        raise TypeError(
            "top_k必须是整数"
        )

    if top_k < 1:
        raise ValueError(
            "top_k必须大于或等于1"
        )

    # 复用在线DiagnosisService的查询构造规则。
    #
    # 当前规则为：
    #
    # symptom + 换行 + log_excerpt
    #
    # 不把robot_id放入查询，因为机器人实例编号
    # 通常不属于知识库中的故障语义。
    query = build_diagnosis_retrieval_query(
        request
    )

    # 将API请求中的可选范围转换成
    # 检索层使用的不可变RetrievalScopeFilter。
    #
    # 请求没有显式范围时返回None，表示全库检索。
    retrieval_scope = (
        build_diagnosis_retrieval_scope(
            request
        )
    )

    # 正式运行时，retrieval_provider实际是：
    #
    # GatedHybridRetriever
    #
    # 它内部执行：
    #
    # 查询改写
    # → 查询Embedding
    # → 向量召回
    # → 关键词召回
    # → RRF融合
    # → 组合证据门控。
    retrieval_result = (
        await retrieval_provider.retrieve(
            query=query,
            top_k=top_k,
            retrieval_scope=retrieval_scope,
        )
    )

    # Protocol主要服务于静态类型检查，
    # 不会自动验证真实对象的返回值。
    #
    # 因此在运行时明确检查返回数据契约，
    # 避免错误实现返回dict或None后，
    # 在访问decision时产生模糊的AttributeError。
    if not isinstance(
        retrieval_result,
        GatedHybridRetrievalResult,
    ):
        raise TypeError(
            "retrieval_provider必须返回"
            "GatedHybridRetrievalResult"
        )

    # 门控拒绝代表当前证据不允许进入生成阶段。
    #
    # 这里不能把门控拒绝记成某个Prompt的拒答表现，
    # 因为两个Prompt实际上都没有被调用。
    #
    # 如果仍计入Prompt指标，会把检索层问题
    # 错误归因到Prompt层。
    if not retrieval_result.decision.accepted:
        raise RuntimeError(
            "证据门控拒绝，不能运行Prompt对照实验："
            f"{retrieval_result.decision.reason}"
        )

    # 正常门控策略放行时一定应存在候选。
    #
    # 这里仍显式检查，用于防御一个不自洽的
    # 自定义检索Provider：
    #
    # accepted=True，但是retrieved_chunks为空。
    if not retrieval_result.retrieved_chunks:
        raise RuntimeError(
            "证据门控已经放行，"
            "但没有返回任何检索候选"
        )

    # 两个Provider收到同一个request对象，
    # 以及同一个retrieved_chunks元组对象。
    #
    # 不能在两个组之间重新检索，
    # 否则向量服务、索引或排序的变化
    # 会成为额外实验变量。
    comparison_case = (
        await compare_diagnosis_prompt_case(
            case_id=case_id,
            request=request,
            evidence=(
                retrieval_result
                .retrieved_chunks
            ),
            basic_provider=basic_provider,
            evidence_first_provider=(
                evidence_first_provider
            ),
        )
    )

    # 报告构造服务会分别计算：
    #
    # 1. 结构成功率；
    # 2. 非法响应、超时和上游失败数；
    # 3. 引用白名单通过率；
    # 4. 未知证据编号数量；
    # 5. 平均LLM生成耗时。
    return (
        build_diagnosis_prompt_comparison_report(
            llm_model=llm_model,
            temperature=temperature,
            cases=[comparison_case],
        )
    )


async def run_real_prompt_comparison(
    args: argparse.Namespace,
) -> DiagnosisPromptComparisonReport:
    """装配真实资源并运行一次诊断Prompt对照实验。

    本函数负责：

    1. 读取真实配置；
    2. 打开真实Chroma Collection；
    3. 创建Embedding和LLM客户端；
    4. 创建共享检索器；
    5. 创建两个Prompt Provider；
    6. 运行一次受门控保护的A/B实验；
    7. 无论成功或失败都关闭外部客户端。

    本函数只返回结构化报告，
    不负责写文件或打印终端摘要。
    """

    if not isinstance(
        args,
        argparse.Namespace,
    ):
        raise TypeError(
            "args必须是argparse.Namespace"
        )

    # get_settings()会从环境变量和.env读取配置。
    #
    # Settings对象中的SecretStr不会在普通repr中
    # 直接暴露真实API Key。
    settings = get_settings()

    # 分别取得并校验两个模型名称。
    #
    # Embedding模型负责检索；
    # LLM模型负责生成结构化诊断草稿。
    embedding_model = get_embedding_model(
        settings
    )
    llm_model = get_llm_model(
        settings
    )

    # 先打开Chroma，再创建网络客户端。
    #
    # 如果本地Collection配置错误，
    # 可以在尚未创建网络资源时快速失败。
    vector_store = ChromaVectorStore(
        persist_directory=(
            settings.chroma_persist_directory
        ),
        collection_name=(
            settings.chroma_collection_name
        ),
        embedding_model=embedding_model,
    )

    # 创建Embedding客户端本身不会立即联网。
    #
    # 真正的外部请求会在检索器调用：
    #
    # await EmbeddingService.embed_texts(...)
    #
    # 时发生。
    embedding_client = (
        create_embedding_client(
            settings
        )
    )

    # 先使用None表示LLM客户端尚未成功创建。
    #
    # 如果create_llm_client()本身失败，
    # finally仍然需要关闭已经创建的Embedding客户端。
    llm_client = None

    try:
        # 创建生成式LLM客户端。
        #
        # 真正的LLM请求会在：
        #
        # await client.chat.completions.create(...)
        #
        # 时发生。
        llm_client = create_llm_client(
            settings
        )

        # 使用在线API共用的装配工厂构造检索链。
        #
        # 该工厂会把同一个vector_store同时交给：
        #
        # 1. 向量检索；
        # 2. 关键词检索索引构造。
        retrieval_provider = (
            build_diagnosis_retrieval_provider(
                embedding_client=(
                    embedding_client
                ),
                vector_store=vector_store,
                settings=settings,
            )
        )

        # 两个Provider共享：
        #
        # 1. 同一个llm_client；
        # 2. 同一个llm_model；
        # 3. 同一个temperature常量。
        #
        # 两者唯一不同之处是prompt_variant。
        basic_provider = (
            OpenAIDiagnosisDraftProvider(
                client=llm_client,
                model=llm_model,
                prompt_variant=(
                    BASIC_DIAGNOSIS_PROMPT_VARIANT
                ),
            )
        )

        evidence_first_provider = (
            OpenAIDiagnosisDraftProvider(
                client=llm_client,
                model=llm_model,
                prompt_variant=(
                    EVIDENCE_DIAGNOSIS_PROMPT_VARIANT
                ),
            )
        )

        # DiagnosisRequest会再次应用Pydantic约束：
        #
        # 1. 清理首尾空白；
        # 2. 拒绝空字段；
        # 3. 检查最大长度；
        # 4. 拒绝未知字段。
        request = DiagnosisRequest(
            robot_id=args.robot_id,
            symptom=args.symptom,
            log_excerpt=args.log_excerpt,

            # 当前CLI暂时没有范围参数，
            # 所以明确表示允许全库检索。
            retrieval_scope=None,
        )

        # 进入前一阶段完成的纯实验编排函数。
        #
        # 它会执行一次共享检索，
        # 门控放行后再依次调用两个Provider。
        return await run_prompt_comparison_experiment(
            case_id=args.case_id,
            request=request,
            top_k=args.top_k,
            retrieval_provider=(
                retrieval_provider
            ),
            basic_provider=basic_provider,
            evidence_first_provider=(
                evidence_first_provider
            ),
            llm_model=llm_model,
            temperature=(
                DIAGNOSIS_TEMPERATURE
            ),
        )
    finally:
        # finally在以下情况都会执行：
        #
        # 1. 正常返回报告；
        # 2. Embedding失败；
        # 3. 门控拒绝；
        # 4. LLM超时；
        # 5. 报告构造失败；
        # 6. 发生其他异常。
        #
        # 使用嵌套finally保证：
        #
        # 即使关闭LLM客户端本身发生异常，
        # Embedding客户端仍会继续尝试关闭。
        try:
            if llm_client is not None:
                # AsyncOpenAI.close()是异步方法，
                # 必须使用await等待连接资源释放完成。
                await llm_client.close()
        finally:
            await embedding_client.close()


def escape_markdown_cell(
    value: object,
) -> str:
    """转义Markdown表格单元格中的特殊内容。"""

    return (
        str(value)
        .replace("|", r"\|")
        .replace("\r\n", "<br>")
        .replace("\n", "<br>")
    )


def render_metrics_row(
    *,
    group_name: str,
    metrics: DiagnosisPromptVariantMetrics,
) -> str:
    """将一个Prompt实验组转换成Markdown指标行。"""

    values = [
        group_name,
        metrics.prompt_version,
        metrics.case_count,
        metrics.success_count,
        metrics.invalid_response_count,
        metrics.timeout_count,
        metrics.upstream_error_count,
        metrics.abstained_count,
        metrics.whitelist_passed_count,
        metrics.unknown_evidence_id_count,
        f"{metrics.schema_success_rate:.3f}",
        (
            f"{metrics.evidence_whitelist_pass_rate:.3f}"
        ),
        f"{metrics.average_generation_ms:.1f}",
    ]

    escaped_values = [
        escape_markdown_cell(value)
        for value in values
    ]

    return (
        "| "
        + " | ".join(escaped_values)
        + " |"
    )


def append_run_result(
    *,
    lines: list[str],
    group_name: str,
    result: DiagnosisPromptRunResult,
) -> None:
    """把一个Prompt版本的逐案例结果加入Markdown。

    lines是正在构造的Markdown行列表。
    本函数直接向该列表追加内容，因此没有返回值。
    """

    lines.extend(
        [
            f"### {group_name}",
            "",
            f"- Prompt版本：`{result.prompt_version}`",
            f"- 运行结果：`{result.outcome}`",
            (
                "- LLM生成耗时："
                f"{result.generation_ms:.1f} ms"
            ),
            "",
        ]
    )

    if result.outcome != "success":
        # 失败结果没有draft，
        # 只展示已经过项目异常转换的安全错误摘要。
        lines.extend(
            [
                "#### 安全错误摘要",
                "",
                (
                    result.error_detail
                    or "没有错误摘要"
                ),
                "",
            ]
        )

        return

    # DiagnosisPromptRunResult的数据契约已经保证：
    #
    # outcome=success时draft一定存在。
    # 这里的assert用于声明内部不变量，
    # 不是代替Pydantic做输入校验。
    assert result.draft is not None

    draft = result.draft

    lines.extend(
        [
            "#### 结构化草稿状态",
            "",
            f"- status：`{draft.status}`",
            f"- risk_level：`{draft.risk_level}`",
            f"- abstained：`{draft.abstained}`",
            "",
            "#### 可能原因",
            "",
        ]
    )

    if draft.possible_causes:
        for index, cause in enumerate(
            draft.possible_causes,
            start=1,
        ):
            evidence_ids = ", ".join(
                cause.evidence_ids
            )

            lines.append(
                f"{index}. {cause.description}"
                f"（临时证据引用：{evidence_ids}）"
            )
    else:
        lines.append("- 无")

    lines.extend(
        [
            "",
            "#### 下一步检查",
            "",
        ]
    )

    if draft.next_checks:
        for index, check in enumerate(
            draft.next_checks,
            start=1,
        ):
            evidence_ids = ", ".join(
                check.evidence_ids
            )

            lines.append(
                f"{index}. {check.description}"
                f"（临时证据引用：{evidence_ids}；"
                f"风险：{check.risk_level}；"
                "需要有资质人员："
                f"{check.requires_qualified_person}）"
            )
    else:
        lines.append("- 无")

    lines.extend(
        [
            "",
            "#### 缺失信息",
            "",
        ]
    )

    if draft.missing_information:
        for item in draft.missing_information:
            lines.append(f"- {item}")
    else:
        lines.append("- 无")

    lines.append("")


def render_markdown_report(
    report: DiagnosisPromptComparisonReport,
) -> str:
    """把结构化Prompt对照报告渲染成Markdown。"""

    if not isinstance(
        report,
        DiagnosisPromptComparisonReport,
    ):
        raise TypeError(
            "report必须是"
            "DiagnosisPromptComparisonReport"
        )

    lines = [
        "# 诊断 Prompt A/B 对照实验",
        "",
        "## 实验说明",
        "",
        (
            "本报告使用相同请求、相同检索证据、"
            "相同模型和相同 temperature，"
            "只改变诊断 Prompt。"
        ),
        "",
        (
            "当前默认命令只运行单案例冒烟实验，"
            "用于验证真实调用链和观察Prompt差异；"
            "单案例结果不能代表统计显著性，"
            "也不能替代完整Gold集评测。"
        ),
        "",
        "## 实验配置",
        "",
        f"- LLM模型：`{report.llm_model}`",
        f"- temperature：`{report.temperature}`",
        f"- 案例数量：`{len(report.cases)}`",
        "",
        "## 汇总指标",
        "",
        (
            "| 实验组 | Prompt版本 | 案例数 | "
            "结构成功数 | 非法响应数 | 超时数 | "
            "上游错误数 | 主动拒答数 | "
            "白名单通过数 | 未知引用数 | "
            "结构成功率 | 引用白名单通过率 | "
            "平均生成耗时(ms) |"
        ),
        (
            "|---|---|---:|---:|---:|---:|---:|"
            "---:|---:|---:|---:|---:|---:|"
        ),
        render_metrics_row(
            group_name="基础Prompt",
            metrics=report.basic_metrics,
        ),
        render_metrics_row(
            group_name="证据优先Prompt",
            metrics=report.evidence_first_metrics,
        ),
        "",
        "## 逐案例结果",
        "",
    ]

    for case in report.cases:
        lines.extend(
            [
                f"## 案例 {case.case_id}",
                "",
                "### 请求",
                "",
                f"- robot_id：`{case.request.robot_id}`",
                f"- symptom：{case.request.symptom}",
                (
                    "- log_excerpt："
                    f"{case.request.log_excerpt}"
                ),
                "",
                "### 两组共同证据",
                "",
            ]
        )

        for candidate in case.evidence:
            chunk = candidate.chunk

            if (
                candidate.vector_similarity
                is None
            ):
                vector_similarity = (
                    "未进入向量候选"
                )
            else:
                vector_similarity = (
                    f"{candidate.vector_similarity:.6f}"
                )

            lines.extend(
                [
                    (
                        f"#### E{candidate.rank}："
                        f"`{chunk.chunk_id}`"
                    ),
                    "",
                    f"- 来源文件：`{chunk.source_file}`",
                    (
                        "- 位置："
                        f"{chunk.page_or_section}"
                    ),
                    (
                        "- RRF分数："
                        f"`{candidate.rrf_score:.6f}`"
                    ),
                    (
                        "- 向量相似度："
                        f"`{vector_similarity}`"
                    ),
                    "",
                    "证据正文：",
                    "",
                    chunk.content,
                    "",
                ]
            )

        append_run_result(
            lines=lines,
            group_name="基础Prompt结果",
            result=case.basic_result,
        )

        append_run_result(
            lines=lines,
            group_name="证据优先Prompt结果",
            result=case.evidence_first_result,
        )

    return "\n".join(lines).rstrip() + "\n"


def write_reports(
    *,
    report: DiagnosisPromptComparisonReport,
    json_path: Path,
    markdown_path: Path,
) -> None:
    """将同一份实验报告写成JSON和Markdown。"""

    if not isinstance(json_path, Path):
        raise TypeError(
            "json_path必须是Path"
        )

    if not isinstance(markdown_path, Path):
        raise TypeError(
            "markdown_path必须是Path"
        )

    # parent取得文件所在目录。
    #
    # parents=True允许递归创建多层目录；
    # exist_ok=True表示目录已存在时不报错。
    json_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    markdown_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # model_dump_json()是Pydantic BaseModel方法。
    #
    # 它按照模型字段和datetime序列化规则
    # 生成合法JSON，而不是依赖手写字典。
    json_text = report.model_dump_json(
        indent=2,
    )

    json_path.write_text(
        json_text + "\n",
        encoding="utf-8",
    )

    markdown_path.write_text(
        render_markdown_report(report),
        encoding="utf-8",
    )


def print_summary(
    *,
    report: DiagnosisPromptComparisonReport,
    json_path: Path,
    markdown_path: Path,
) -> None:
    """在终端输出便于人工确认的核心指标。"""

    basic = report.basic_metrics
    evidence_first = (
        report.evidence_first_metrics
    )

    print("诊断Prompt对照实验完成")
    print(f"案例数：{len(report.cases)}")

    print(
        "基础Prompt结构成功率："
        f"{basic.schema_success_rate:.3f}"
    )
    print(
        "基础Prompt引用白名单通过率："
        f"{basic.evidence_whitelist_pass_rate:.3f}"
    )

    print(
        "证据优先Prompt结构成功率："
        f"{evidence_first.schema_success_rate:.3f}"
    )
    print(
        "证据优先Prompt引用白名单通过率："
        f"{evidence_first.evidence_whitelist_pass_rate:.3f}"
    )

    print(f"JSON报告：{json_path.resolve()}")
    print(
        "Markdown报告："
        f"{markdown_path.resolve()}"
    )


def main(
    argv: Sequence[str] | None = None,
) -> None:
    """运行CLI实验并输出JSON、Markdown和终端摘要。"""

    # 正式运行时argv=None，
    # parse_args()读取sys.argv。
    #
    # 测试可以传入list，
    # 避免读取pytest自己的命令行参数。
    args = parse_args(argv)

    # asyncio.run()负责：
    #
    # 1. 创建新的事件循环；
    # 2. 运行异步实验直到完成；
    # 3. 关闭事件循环；
    # 4. 返回协程的最终结果。
    report = asyncio.run(
        run_real_prompt_comparison(
            args
        )
    )

    # JSON和Markdown必须来自同一个report对象，
    # 避免两份报告记录不同实验结果。
    write_reports(
        report=report,
        json_path=args.json_output,
        markdown_path=(
            args.markdown_output
        ),
    )

    # 文件成功写入后再打印完成摘要。
    #
    # 如果写文件失败，不应该提前显示
    # “实验完成”造成误导。
    print_summary(
        report=report,
        json_path=args.json_output,
        markdown_path=(
            args.markdown_output
        ),
    )


if __name__ == "__main__":
    main()
