"""编排同一道问题上的RAG与裸LLM对照实验。"""

# argparse负责定义和解析命令行参数。
import argparse

# asyncio.run()负责从同步命令行入口
# 启动并等待异步实验流程。
import asyncio

# json负责把Pydantic报告转换成格式化JSON文本。
import json

# datetime用于记录实验报告生成时间。
from datetime import datetime

# Path提供跨平台的文件路径对象，
# 后面使用它创建目录并写入JSON/Markdown报告。
from pathlib import Path

# perf_counter用于测量RAG HTTP请求的端到端耗时。
from time import perf_counter

# httpx提供异步HTTP客户端、响应对象和异常类型。
import httpx

# ValidationError表示RAG API返回的数据
# 不符合KnowledgeQueryResponse契约。
from pydantic import ValidationError

# Settings是经过Pydantic校验的配置对象；
# get_settings()读取环境变量和.env并缓存结果。
from app.config import (
    Settings,
    get_settings,
)

from app.schemas.evaluation import (
    GoldQuestion,
)
from app.schemas.knowledge_query import (
    KnowledgeQueryResponse,
)

from app.schemas.rag_comparison import (
    RagVsBareLLMCase,
    RagVsBareLLMReport,
)
# get_embedding_model()只取得并验证模型名称。
#
# 实验脚本不会自己调用Embedding，
# 真正的Query Embedding由RAG API内部执行。
from app.services.embedding_client import (
    get_embedding_model,
)

# load_gold_questions()逐行读取JSONL，
# 并把每行验证成GoldQuestion。
from app.services.evaluation import (
    load_gold_questions,
)

# create_llm_client()创建裸LLM控制组使用的
# OpenAI兼容异步客户端；
# get_llm_model()取得并验证生成模型名称。
from app.services.llm_client import (
    create_llm_client,
    get_llm_model,
)

from app.services.bare_llm import (
    BARE_LLM_SYSTEM_PROMPT,
    BareLLMProvider,
    OpenAIBareLLMProvider,
)


class RagComparisonRequestError(
    RuntimeError
):
    """RAG对比实验访问知识库API失败。"""


def select_gold_questions(
    *,
    questions: list[GoldQuestion],
    question_ids: list[str],
) -> list[GoldQuestion]:
    """按照指定题号和顺序选择Gold Question。"""

    if not questions:
        raise ValueError(
            "Gold Question列表不能为空"
        )

    if not question_ids:
        raise ValueError(
            "RAG对比实验题号列表不能为空"
        )

    cleaned_ids: list[str] = []

    for question_id in question_ids:
        cleaned_id = question_id.strip()

        if not cleaned_id:
            raise ValueError(
                "RAG对比实验题号不能为空"
            )

        cleaned_ids.append(
            cleaned_id
        )

    if len(set(cleaned_ids)) != len(
        cleaned_ids
    ):
        raise ValueError(
            "RAG对比实验题号不能重复"
        )

    # 将问题列表转换成以question_id为键的索引，
    # 便于按用户指定顺序取得问题。
    question_index = {
        question.question_id: question
        for question in questions
    }

    missing_ids = [
        question_id
        for question_id in cleaned_ids
        if question_id not in question_index
    ]

    if missing_ids:
        raise ValueError(
            "RAG对比实验题号不存在："
            + ", ".join(missing_ids)
        )

    # 不按照Gold文件原始顺序返回，
    # 而是保留question_ids参数声明的实验顺序。
    return [
        question_index[question_id]
        for question_id in cleaned_ids
    ]


async def request_rag_answer(
    *,
    question: GoldQuestion,
    client: httpx.AsyncClient,
    rag_api_url: str,
    top_k: int,
) -> tuple[
    KnowledgeQueryResponse,
    float,
]:
    """调用一次正式RAG API并返回响应与端到端耗时。"""

    request_body = {
        "question": question.question,
        "top_k": top_k,
    }

    started_at = perf_counter()

    try:
        # json=让httpx自动：
        # 1. 将Python字典序列化为JSON；
        # 2. 使用UTF-8编码；
        # 3. 设置application/json请求头。
        http_response = await client.post(
            rag_api_url,
            json=request_body,
        )
    except httpx.TimeoutException as exc:
        raise RagComparisonRequestError(
            "RAG对比实验请求超时："
            f"question_id={question.question_id}"
        ) from exc
    except httpx.RequestError as exc:
        raise RagComparisonRequestError(
            "RAG对比实验无法连接知识库API："
            f"question_id={question.question_id}"
        ) from exc

    rag_total_ms = (
        perf_counter() - started_at
    ) * 1000.0

    # Request ID用于把实验失败与服务端日志关联起来。
    request_id = http_response.headers.get(
        "X-Request-ID",
        "missing",
    )

    try:
        # 2xx响应正常返回；
        # 4xx、5xx会产生HTTPStatusError。
        http_response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise RagComparisonRequestError(
            "知识库API返回错误状态："
            f"question_id={question.question_id}，"
            f"status={http_response.status_code}，"
            f"request_id={request_id}"
        ) from exc

    try:
        # response.json()将HTTP响应正文解析为Python对象。
        raw_response = http_response.json()
    except ValueError as exc:
        raise RagComparisonRequestError(
            "知识库API返回了非法JSON："
            f"question_id={question.question_id}，"
            f"request_id={request_id}"
        ) from exc

    try:
        # 即使HTTP状态是200，也必须验证业务响应。
        #
        # 这里会检查：
        # - answer
        # - citations
        # - retrieval_ms
        # - abstained
        # - 拒答与引用之间的业务关系
        response = (
            KnowledgeQueryResponse
            .model_validate(
                raw_response
            )
        )
    except ValidationError as exc:
        raise RagComparisonRequestError(
            "知识库API响应不符合"
            "KnowledgeQueryResponse契约："
            f"question_id={question.question_id}，"
            f"request_id={request_id}"
        ) from exc

    return response, rag_total_ms


async def compare_rag_and_bare(
    *,
    questions: list[GoldQuestion],
    rag_client: httpx.AsyncClient,
    bare_provider: BareLLMProvider,
    rag_api_url: str,
    llm_model: str,
    embedding_model: str,
    collection_name: str,
    top_k: int,
    similarity_threshold: float,
) -> RagVsBareLLMReport:
    """顺序执行RAG与裸LLM实验并生成结构化报告。"""

    if not questions:
        raise ValueError(
            "RAG对比实验问题列表不能为空"
        )

    if not 1 <= top_k <= 10:
        raise ValueError(
            "top_k必须在1到10之间"
        )

    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError(
            "similarity_threshold"
            "必须在0到1之间"
        )

    # 在发送任何请求之前验证报告元数据，
    # 避免实验完成后才因配置缺失无法创建报告。
    metadata_values = {
        "rag_api_url": rag_api_url,
        "llm_model": llm_model,
        "embedding_model": embedding_model,
        "collection_name": collection_name,
    }

    cleaned_metadata: dict[str, str] = {}

    for field_name, value in (
        metadata_values.items()
    ):
        cleaned_value = value.strip()

        if not cleaned_value:
            raise ValueError(
                f"{field_name}不能为空"
            )

        cleaned_metadata[field_name] = (
            cleaned_value
        )

    question_ids = [
        question.question_id
        for question in questions
    ]

    if len(set(question_ids)) != len(
        question_ids
    ):
        raise ValueError(
            "RAG对比实验问题包含重复question_id"
        )

    cases: list[RagVsBareLLMCase] = []

    # 有意顺序执行，而不使用asyncio.gather()。
    #
    # 每道题先执行RAG，再执行裸LLM：
    # - 避免同时触发多个真实模型请求；
    # - 降低上游限流风险；
    # - 保持报告顺序稳定；
    # - 出错时容易定位题号。
    for question in questions:
        rag_response, rag_total_ms = (
            await request_rag_answer(
                question=question,
                client=rag_client,
                rag_api_url=(
                    cleaned_metadata[
                        "rag_api_url"
                    ]
                ),
                top_k=top_k,
            )
        )

        # 裸LLM只取得同一道问题的文本，
        # 不会收到Gold参考答案、预期证据或RAG响应。
        bare_result = (
            await bare_provider.generate_answer(
                question=question.question,
            )
        )

        cases.append(
            RagVsBareLLMCase(
                gold_question=question,
                rag_response=rag_response,
                rag_total_ms=rag_total_ms,
                bare_result=bare_result,
            )
        )

    return RagVsBareLLMReport(
        generated_at=(
            datetime.now().astimezone()
        ),
        rag_api_url=(
            cleaned_metadata["rag_api_url"]
        ),
        llm_model=(
            cleaned_metadata["llm_model"]
        ),
        embedding_model=(
            cleaned_metadata[
                "embedding_model"
            ]
        ),
        collection_name=(
            cleaned_metadata[
                "collection_name"
            ]
        ),
        top_k=top_k,
        similarity_threshold=(
            similarity_threshold
        ),

        # 把裸LLM真实使用的System Prompt
        # 原样保存进报告。
        bare_system_prompt=(
            BARE_LLM_SYSTEM_PROMPT
        ),
        cases=cases,
    )


def render_markdown_blockquote(
    value: str,
) -> str:
    """把多行文本渲染成Markdown引用块。"""

    # 去掉整段文本首尾的空白，
    # 避免报告产生多余的空行。
    cleaned_value = value.strip()

    if not cleaned_value:
        raise ValueError(
            "Markdown引用内容不能为空"
        )

    quoted_lines: list[str] = []

    # splitlines()按照不同平台的换行符切分文本，
    # 能同时处理Windows的\r\n和Unix的\n。
    for line in cleaned_value.splitlines():
        # 只清理每行右侧空格，
        # 不破坏正文可能具有意义的左侧缩进。
        cleaned_line = line.rstrip()

        if cleaned_line:
            quoted_lines.append(
                f"> {cleaned_line}"
            )
        else:
            # Markdown引用块中的空行使用单独的>表示，
            # 保证多段文字仍属于同一个引用区域。
            quoted_lines.append(">")

    return "\n".join(quoted_lines)


def render_markdown_report(
    report: RagVsBareLLMReport,
) -> str:
    """把结构化RAG对照报告渲染成Markdown文本。"""

    # 先写入整次实验共享的配置。
    #
    # 这些配置决定不同实验结果能否公平比较，
    # 所以不能只保存回答正文。
    lines: list[str] = [
        "# RAG 与裸 LLM 对照实验",
        "",
        (
            "本报告使用同一组Gold Question，"
            "对比正式RAG API与不提供知识库证据的"
            "裸LLM控制组。"
        ),
        "",
        "## 实验配置",
        "",
        (
            "- 生成时间："
            f"`{report.generated_at.isoformat()}`"
        ),
        f"- RAG API：`{report.rag_api_url}`",
        f"- LLM模型：`{report.llm_model}`",
        (
            "- Embedding模型："
            f"`{report.embedding_model}`"
        ),
        (
            "- Chroma Collection："
            f"`{report.collection_name}`"
        ),
        f"- RAG Top-K：{report.top_k}",
        (
            "- RAG相似度拒答阈值："
            f"{report.similarity_threshold:.3f}"
        ),
        "",
        "## 对照原则",
        "",
        (
            "- 两组收到完全相同的自然语言问题，"
            "并使用相同的生成模型。"
        ),
        (
            "- RAG组通过正式知识库API检索证据，"
            "回答可以返回可追溯引用。"
        ),
        (
            "- 裸LLM组只收到问题和下面记录的"
            "System Prompt，不会收到Gold答案、"
            "预期证据或RAG结果。"
        ),
        (
            "- 本报告保存事实结果，不根据回答长度"
            "自动判断哪一组更好；最终需要结合"
            "事实正确性、可追溯性和拒答行为进行观察。"
        ),
        "",
        "## 裸 LLM 控制组 System Prompt",
        "",
        render_markdown_blockquote(
            report.bare_system_prompt
        ),
        "",
        "## 逐题结果",
        "",
    ]

    # 每个case表示同一道题上的一组对照结果。
    for case in report.cases:
        question = case.gold_question
        rag_response = case.rag_response
        bare_result = case.bare_result

        gold_status = (
            "可回答"
            if question.answerable
            else "知识库无答案"
        )

        rag_status = (
            "拒答"
            if rag_response.abstained
            else "已回答"
        )

        lines.extend(
            [
                (
                    f"### {question.question_id}"
                    f" — {question.question_type}"
                ),
                "",
                f"- Gold状态：{gold_status}",
                f"- RAG状态：{rag_status}",
                "",
                "#### 问题",
                "",
                render_markdown_blockquote(
                    question.question
                ),
                "",
                "#### Gold参考答案",
                "",
            ]
        )

        if question.reference_answer is None:
            lines.append(
                "该题被标注为知识库无答案，"
                "因此没有参考答案。"
            )
        else:
            lines.append(
                render_markdown_blockquote(
                    question.reference_answer
                )
            )

        lines.extend(
            [
                "",
                "#### Gold预期证据",
                "",
            ]
        )

        if not question.expected_evidence:
            lines.append(
                "- 无；该题预期系统拒答。"
            )
        else:
            # enumerate(..., start=1)在遍历证据的同时，
            # 生成从1开始的可读编号。
            for evidence_number, evidence in enumerate(
                question.expected_evidence,
                start=1,
            ):
                lines.append(
                    f"- 证据 {evidence_number}："
                    f"`{evidence.source_file}`；"
                    f"`{evidence.page_or_section}`"
                )

        lines.extend(
            [
                "",
                "#### RAG组结果",
                "",
                f"- 状态：{rag_status}",
                (
                    "- API端到端耗时："
                    f"{case.rag_total_ms:.3f} ms"
                ),
                (
                    "- 响应记录的检索耗时："
                    f"{rag_response.retrieval_ms:.3f} ms"
                ),
                "",
                "**回答正文**",
                "",
                render_markdown_blockquote(
                    rag_response.answer
                ),
                "",
                "**代码组装的引用**",
                "",
            ]
        )

        if not rag_response.citations:
            lines.append("- 无")
        else:
            for citation_number, citation in enumerate(
                rag_response.citations,
                start=1,
            ):
                lines.extend(
                    [
                        (
                            f"- 引用 {citation_number}："
                            f"Top-{citation.rank}；"
                            f"`{citation.source_file}`；"
                            f"`{citation.page_or_section}`"
                        ),
                        (
                            "  - Chunk ID："
                            f"`{citation.chunk_id}`"
                        ),
                        (
                            "  - 相似度："
                            f"{citation.similarity:.3f}"
                        ),
                        "  - 原始片段：",
                        "",
                        render_markdown_blockquote(
                            citation.excerpt
                        ),
                        "",
                    ]
                )

        lines.extend(
            [
                "#### 裸 LLM 组结果",
                "",
                (
                    "- 生成耗时："
                    f"{bare_result.generation_ms:.3f} ms"
                ),
                "",
                "**回答正文**",
                "",
                render_markdown_blockquote(
                    bare_result.answer
                ),
                "",
                "#### 人工观察（实验运行后填写）",
                "",
                "- RAG事实是否符合Gold参考答案：",
                "- 裸LLM事实是否符合Gold参考答案：",
                "- RAG引用是否覆盖必要证据：",
                "- 系统是否在应当拒答时正确拒答：",
                "- 两组最主要的差异：",
                "",
            ]
        )

    lines.extend(
        [
            "## 耗时指标解释",
            "",
            (
                "- `rag_total_ms`是实验脚本从发出HTTP请求"
                "到收到完整API响应的端到端耗时，包含HTTP、"
                "问题向量化、检索、LLM生成和响应序列化。"
            ),
            (
                "- `retrieval_ms`是知识库响应内部记录的"
                "检索阶段耗时，不代表整个RAG请求耗时。"
            ),
            (
                "- `generation_ms`是裸LLM Provider从发出"
                "模型请求到收到完整回答的耗时。"
            ),
            (
                "- 因为三个指标覆盖的处理阶段不同，"
                "它们可以用于观察工程成本，但不能被当作"
                "严格相同口径的模型速度基准。"
            ),
            "",
            "## 结论边界",
            "",
            (
                "该实验用于展示知识库证据对回答正确性、"
                "可追溯性和安全拒答的影响。一次模型输出"
                "具有随机性，因此单次结果只能作为案例分析，"
                "不能替代完整的检索评测和引用评测。"
            ),
        ]
    )

    # 统一在文件末尾保留一个换行符，
    # 符合常见文本文件约定。
    return "\n".join(lines) + "\n"


def write_reports(
    *,
    report: RagVsBareLLMReport,
    json_output: Path,
    markdown_output: Path,
) -> None:
    """把同一份结构化实验结果写成JSON和Markdown。"""

    # parent表示输出文件所在的父目录。
    #
    # parents=True：
    # 如果中间有多级目录不存在，一并创建。
    #
    # exist_ok=True：
    # 目录已经存在时不抛出FileExistsError。
    json_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    markdown_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # model_dump(mode="json")将Pydantic模型转换为
    # 只包含JSON兼容值的Python字典。
    #
    # 例如datetime会转换成ISO 8601时间字符串。
    json_data = report.model_dump(
        mode="json"
    )

    # json.dumps()只生成字符串，不直接写文件。
    #
    # ensure_ascii=False：
    # 中文保持为可读文字，而不是\u673a\u5668形式。
    #
    # indent=2：
    # 使用两个空格缩进，使报告适合人工检查。
    json_text = json.dumps(
        json_data,
        ensure_ascii=False,
        indent=2,
    ) + "\n"

    # write_text()负责真正写入文件。
    #
    # encoding="utf-8"明确指定编码，
    # 避免Windows默认编码造成中文乱码。
    json_output.write_text(
        json_text,
        encoding="utf-8",
    )

    markdown_output.write_text(
        render_markdown_report(report),
        encoding="utf-8",
    )


def parse_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """解析RAG与裸LLM对照实验的命令行参数。"""

    parser = argparse.ArgumentParser(
        description=(
            "在同一组Gold Question上对比"
            "正式RAG API与裸LLM回答"
        )
    )

    parser.add_argument(
        "--questions",
        type=Path,
        default=Path(
            "data/eval/gold_questions.jsonl"
        ),
        help="Gold Question JSONL文件路径",
    )

    parser.add_argument(
        "--question-id",
        dest="question_ids",
        action="append",
        default=None,
        help=(
            "参加实验的题号；"
            "需要多题时重复使用该参数。"
            "默认选择q014和q020"
        ),
    )

    parser.add_argument(
        "--api-url",
        default=(
            "http://127.0.0.1:8000"
            "/api/v1/knowledge/query"
        ),
        help="正式知识库查询API的完整地址",
    )

    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path(
            "docs/rag-vs-bare-llm.json"
        ),
        help="JSON实验报告输出路径",
    )

    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path(
            "docs/rag-vs-bare-llm.md"
        ),
        help="Markdown实验报告输出路径",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="RAG查询使用的Top-K数量",
    )

    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=120.0,
        help=(
            "每次RAG HTTP请求允许等待的"
            "最长秒数"
        ),
    )

    # argv=None时，argparse会读取真实命令行。
    #
    # 自动化测试可以传入一个普通列表，
    # 从而不修改进程级sys.argv。
    args = parser.parse_args(argv)

    # action="append"让参数可以重复出现：
    #
    # --question-id q014 --question-id q020
    #
    # 如果用户没有提供任何题号，
    # 使用预先设计的一道可回答题和一道无答案题。
    if args.question_ids is None:
        args.question_ids = [
            "q014",
            "q020",
        ]

    return args


async def run_real_comparison(
    *,
    settings: Settings,
    questions_path: Path,
    question_ids: list[str],
    api_url: str,
    json_output: Path,
    markdown_output: Path,
    top_k: int,
    timeout_seconds: float,
) -> RagVsBareLLMReport:
    """执行真实RAG与裸LLM实验并写出报告。"""

    # 先验证普通参数，
    # 避免创建客户端后才发现输入无效。
    if timeout_seconds <= 0:
        raise ValueError(
            "timeout_seconds必须大于0"
        )

    if not 1 <= top_k <= 10:
        raise ValueError(
            "top_k必须在1到10之间"
        )

    if not api_url.strip():
        raise ValueError(
            "api_url不能为空"
        )

    # load_gold_questions()会完成：
    #
    # 1. 打开JSONL文件；
    # 2. 逐行解析JSON；
    # 3. 验证GoldQuestion字段；
    # 4. 检查问题编号是否重复。
    #
    # 任何数据错误都会在联网前失败。
    all_questions = load_gold_questions(
        questions_path
    )

    # 只选择本次实验需要的问题，
    # 并保留question_ids指定的顺序。
    selected_questions = (
        select_gold_questions(
            questions=all_questions,
            question_ids=question_ids,
        )
    )

    # 取得两种模型名称。
    #
    # LLM模型会被实际用于裸LLM调用；
    # Embedding模型只作为RAG实验配置写进报告。
    llm_model = get_llm_model(
        settings
    )

    embedding_model = (
        get_embedding_model(
            settings
        )
    )

    # 创建真实的OpenAI兼容异步客户端。
    #
    # create_llm_client()只创建客户端对象，
    # 此时还没有向上游发送模型请求。
    llm_client = create_llm_client(
        settings
    )

    # 把真实SDK客户端包装成符合
    # BareLLMProvider Protocol的实现。
    bare_provider = OpenAIBareLLMProvider(
        client=llm_client,
        model=llm_model,
    )

    # httpx.Timeout控制RAG HTTP请求的：
    #
    # - 建立连接等待时间；
    # - 发送请求等待时间；
    # - 读取响应等待时间；
    # - 连接池等待时间。
    rag_timeout = httpx.Timeout(
        timeout_seconds
    )

    try:
        # async with保证无论实验成功还是异常，
        # RAG HTTP连接池都会正确关闭。
        async with httpx.AsyncClient(
            timeout=rag_timeout,
        ) as rag_client:
            report = await compare_rag_and_bare(
                questions=selected_questions,
                rag_client=rag_client,
                bare_provider=bare_provider,
                rag_api_url=api_url,
                llm_model=llm_model,
                embedding_model=embedding_model,
                collection_name=(
                    settings.chroma_collection_name
                ),
                top_k=top_k,

                # 记录本地配置中的正式拒答阈值。
                #
                # FastAPI服务应当使用同一项目和.env启动，
                # 才能保证这里记录的值与服务实际值一致。
                similarity_threshold=(
                    settings
                    .rag_similarity_threshold
                ),
            )
    finally:
        # AsyncOpenAI内部也维护HTTP连接池。
        #
        # 即使RAG请求、裸LLM请求或响应验证失败，
        # finally仍然会执行，避免连接资源泄漏。
        await llm_client.close()

    # 只有两组实验全部完成并通过契约校验后，
    # 才写出正式报告。
    #
    # 这样不会把一份不完整报告误认为成功结果。
    write_reports(
        report=report,
        json_output=json_output,
        markdown_output=markdown_output,
    )

    return report


def print_summary(
    report: RagVsBareLLMReport,
) -> None:
    """在终端打印实验完成状态与逐题概况。"""

    print("RAG与裸LLM对照实验完成")
    print(
        "实验问题数："
        f"{len(report.cases)}"
    )

    for case in report.cases:
        question_id = (
            case.gold_question.question_id
        )

        rag_status = (
            "拒答"
            if case.rag_response.abstained
            else "已回答"
        )

        citation_count = len(
            case.rag_response.citations
        )

        print(
            f"{question_id}："
            f"RAG={rag_status}，"
            f"引用数={citation_count}，"
            "RAG端到端耗时="
            f"{case.rag_total_ms:.1f}ms，"
            "裸LLM耗时="
            f"{case.bare_result.generation_ms:.1f}ms"
        )


def main() -> None:
    """RAG与裸LLM实验的同步命令行入口。"""

    # 解析真实命令行参数。
    args = parse_args()

    # 读取环境变量和.env。
    #
    # get_settings()带有lru_cache，
    # 同一进程内重复调用会复用同一个Settings对象。
    settings = get_settings()

    # asyncio.run()负责创建事件循环，
    # 执行完整的异步实验，然后关闭事件循环。
    report = asyncio.run(
        run_real_comparison(
            settings=settings,
            questions_path=args.questions,
            question_ids=args.question_ids,
            api_url=args.api_url,
            json_output=args.json_output,
            markdown_output=(
                args.markdown_output
            ),
            top_k=args.top_k,
            timeout_seconds=(
                args.timeout_seconds
            ),
        )
    )

    print_summary(report)

    # resolve()把相对路径转换成绝对路径，
    # 方便确认报告实际写入位置。
    print(
        "JSON报告："
        f"{args.json_output.resolve()}"
    )

    print(
        "Markdown报告："
        f"{args.markdown_output.resolve()}"
    )


# 使用下面的命令直接执行本模块时：
#
# python -m scripts.compare_rag_vs_bare_llm
#
# __name__等于"__main__"，因此调用main()。
#
# pytest只是import这个模块时，
# __name__是完整模块名，不会意外运行真实实验。
if __name__ == "__main__":
    main()