"""通过知识库HTTP API执行最终回答引用评测。"""

# argparse负责定义和解析命令行参数。
#
# 例如：
# python -m scripts.evaluate_citations --top-k 3
import argparse

# asyncio.run()负责从同步命令行入口启动异步HTTP评测流程。
import asyncio

# json负责生成机器可读取的JSON报告。
import json

from pathlib import Path

# httpx提供异步HTTP客户端、响应对象和异常类型。
import httpx

# ValidationError表示API返回的JSON
# 不符合KnowledgeQueryResponse契约。
from pydantic import ValidationError

# Settings是应用配置的数据契约；
# get_settings()从环境变量和.env读取并缓存配置。
from app.config import (
    Settings,
    get_settings,
)

# 引用评测的数据契约：
# CitationEvaluationReport表示完整评测报告；
# CitationQuestionEvaluation表示一道题的评测结果。
from app.schemas.citation_evaluation import (
    CitationEvaluationReport,
    CitationQuestionEvaluation,
)

# GoldQuestion表示评测集中的标准问题及其Gold标注。
from app.schemas.evaluation import (
    GoldQuestion,
)

# KnowledgeQueryResponse表示知识库查询API的响应契约。
from app.schemas.knowledge_query import (
    KnowledgeQueryResponse,
)

# CandidateStrategyParameters保存候选检索策略的
# 完整、经过Pydantic校验的参数。
from app.schemas.retrieval_strategy import (
    CandidateStrategyParameters,
)

# 直接从真实混合检索实现中读取默认参数，
# 避免评测脚本另外硬编码20和60后发生漂移。
from app.services.hybrid_retrieval import (
    DEFAULT_HYBRID_CANDIDATE_K,
    DEFAULT_RRF_RANK_CONSTANT,
)

# 门控版本必须与在线API实际使用的规则一致。
from app.services.hybrid_retrieval_gate import (
    HYBRID_EVIDENCE_GATE_VERSION,
)

# 查询改写版本直接来自确定性改写实现。
from app.services.query_rewriting import (
    DETERMINISTIC_QUERY_REWRITE_VERSION,
)

# get_embedding_model()读取并验证EMBEDDING_MODEL。
from app.services.embedding_client import (
    get_embedding_model,
)

# load_gold_questions()逐行读取并验证Gold Question JSONL。
from app.services.evaluation import (
    load_gold_questions,
)

# get_llm_model()读取并验证LLM_MODEL。
from app.services.llm_client import (
    get_llm_model,
)

# 引用评测服务中的评分函数。
#
# evaluate_citation_question负责：
# 比较一道Gold Question和一次API响应，生成单题评测结果。
#
# calculate_citation_metrics负责：
# 汇总所有单题评测结果，计算整体评测指标。
from app.services.citation_evaluation import (
    calculate_citation_metrics,
    evaluate_citation_question,
)

class CitationEvaluationRequestError(
    RuntimeError
):
    """引用评测调用知识库API失败。"""


async def evaluate_citations_via_api(
    *,
    questions: list[GoldQuestion],
    client: httpx.AsyncClient,
    api_url: str,
    llm_model: str,
    embedding_model: str,
    collection_name: str,
    top_k: int,

    # None表示兼容旧版纯向量报告。
    #
    # Week 3真实评测会传入
    # hybrid_rrf_rewrite完整参数。
    retrieval_parameters: (
        CandidateStrategyParameters | None
    ) = None,

    # 纯向量报告填写实际阈值；
    # 混合门控报告必须填写None。
    similarity_threshold: float | None = None,

    # 混合门控报告必须记录版本；
    # 纯向量报告保持None。
    gate_version: str | None = None,
) -> CitationEvaluationReport:
    """调用知识库API并生成结构化引用评测报告。"""

    if not questions:
        raise ValueError(
            "引用评测问题列表不能为空"
        )

    # API Schema允许top_k位于1到10之间。
    if not 1 <= top_k <= 10:
        raise ValueError(
            "top_k必须在1到10之间"
        )

    # Protocol和类型标注不会执行运行时检查。
    #
    # 在发送28次HTTP请求之前检查策略参数类型，
    # 避免全部请求结束后才发现报告无法创建。
    if (
        retrieval_parameters is not None
        and not isinstance(
            retrieval_parameters,
            CandidateStrategyParameters,
        )
    ):
        raise TypeError(
            "retrieval_parameters必须是"
            "CandidateStrategyParameters或None"
        )

    # None表示当前策略没有单一全局相似度阈值。
    #
    # 如果调用方确实提供阈值，
    # 就必须检查类型和值域。
    if similarity_threshold is not None:
        # bool是int的子类，但True不能代表
        # 一个有意义的检索阈值。
        if (
            isinstance(
                similarity_threshold,
                bool,
            )
            or not isinstance(
                similarity_threshold,
                (int, float),
            )
        ):
            raise TypeError(
                "similarity_threshold必须是"
                "数值或None"
            )

        if not (
            0.0
            <= float(similarity_threshold)
            <= 1.0
        ):
            raise ValueError(
                "similarity_threshold"
                "必须在0到1之间"
            )

    if (
        gate_version is not None
        and not isinstance(
            gate_version,
            str,
        )
    ):
        raise TypeError(
            "gate_version必须是字符串或None"
        )

    if retrieval_parameters is None:
        # 没有策略参数时，按照Week 2旧版
        # 纯向量报告处理。
        if similarity_threshold is None:
            raise ValueError(
                "旧版纯向量报告必须设置"
                "similarity_threshold"
            )

        if gate_version is not None:
            raise ValueError(
                "未声明混合检索参数时"
                "不能设置gate_version"
            )
    else:
        # HTTP请求中的top_k和策略参数中的top_k
        # 必须描述同一次实验。
        if (
            retrieval_parameters.top_k
            != top_k
        ):
            raise ValueError(
                "retrieval_parameters.top_k"
                "必须与报告top_k一致"
            )

        if (
            retrieval_parameters.strategy
            == "vector_baseline"
        ):
            if similarity_threshold is None:
                raise ValueError(
                    "纯向量报告必须设置"
                    "similarity_threshold"
                )

            if gate_version is not None:
                raise ValueError(
                    "纯向量报告不能设置"
                    "gate_version"
                )
        else:
            # hybrid_rrf和hybrid_rrf_rewrite
            # 都使用组合证据门控。
            if similarity_threshold is not None:
                raise ValueError(
                    "混合检索报告不能设置"
                    "similarity_threshold"
                )

            if (
                gate_version is None
                or not gate_version.strip()
            ):
                raise ValueError(
                    "混合检索报告必须设置"
                    "gate_version"
                )

    # 在发出任何HTTP请求前检查实验元数据。
    #
    # 如果配置缺失，应立即失败，
    # 不能调用完20道题后才发现报告无法创建。
    metadata_values = {
        "api_url": api_url,
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

    # 在访问API前拒绝重复question_id，
    # 避免同一道题被重复调用并计入分母。
    question_ids = [
        question.question_id
        for question in questions
    ]

    if len(set(question_ids)) != len(
        question_ids
    ):
        raise ValueError(
            "引用评测问题包含重复question_id"
        )

    question_evaluations = []

    # 有意顺序执行，而不使用asyncio.gather()并发。
    #
    # 原因：
    # 1. 避免同时触发大量LLM请求；
    # 2. 降低上游限流风险；
    # 3. 保持报告顺序与Gold文件一致；
    # 4. 服务端日志更容易按题目追踪。
    for question in questions:
        request_body = {
            "question": question.question,
            "top_k": top_k,
        }

        try:
            # json=会让httpx自动：
            #
            # 1. 把Python字典序列化成JSON；
            # 2. 使用UTF-8编码中文；
            # 3. 设置application/json请求头。
            http_response = await client.post(
                cleaned_metadata["api_url"],
                json=request_body,
            )
        except httpx.TimeoutException as exc:
            raise CitationEvaluationRequestError(
                "引用评测请求超时："
                f"{question.question_id}"
            ) from exc
        except httpx.RequestError as exc:
            raise CitationEvaluationRequestError(
                "引用评测无法连接知识库API："
                f"{question.question_id}"
            ) from exc

        # Request ID用于把评测错误与服务端日志对应起来。
        request_id = http_response.headers.get(
            "X-Request-ID",
            "missing",
        )

        try:
            # 2xx响应不会抛异常。
            #
            # 4xx、5xx以及意外重定向会抛出
            # httpx.HTTPStatusError。
            http_response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise CitationEvaluationRequestError(
                "知识库API返回错误状态："
                f"question_id={question.question_id}，"
                f"status={http_response.status_code}，"
                f"request_id={request_id}"
            ) from exc

        try:
            # response.json()把HTTP响应正文
            # 从JSON解析成Python对象。
            raw_response = http_response.json()
        except ValueError as exc:
            # JSONDecodeError继承自ValueError，
            # 因此捕获ValueError可以兼容httpx的解析实现。
            raise CitationEvaluationRequestError(
                "知识库API返回了非法JSON："
                f"question_id={question.question_id}，"
                f"request_id={request_id}"
            ) from exc

        try:
            # 即使HTTP状态是200，也不能直接相信响应结构。
            #
            # model_validate()会检查：
            # - answer
            # - citations
            # - retrieval_ms
            # - abstained
            # - 拒答与引用之间的关系
            api_response = (
                KnowledgeQueryResponse
                .model_validate(
                    raw_response
                )
            )
        except ValidationError as exc:
            raise CitationEvaluationRequestError(
                "知识库API响应不符合"
                "KnowledgeQueryResponse契约："
                f"question_id={question.question_id}，"
                f"request_id={request_id}"
            ) from exc

        # 把真实API响应与当前Gold Question比较。
        question_evaluation = (
            evaluate_citation_question(
                question=question,
                response=api_response,
            )
        )

        question_evaluations.append(
            question_evaluation
        )

    metrics = calculate_citation_metrics(
        question_evaluations
    )

    return CitationEvaluationReport(
        api_url=cleaned_metadata["api_url"],
        llm_model=cleaned_metadata["llm_model"],
        embedding_model=(
            cleaned_metadata["embedding_model"]
        ),
        collection_name=(
            cleaned_metadata["collection_name"]
        ),
        top_k=top_k,

        # 保存完整候选策略参数。
        retrieval_parameters=(
            retrieval_parameters
        ),

        # 纯向量报告为float；
        # 混合报告为None。
        similarity_threshold=(
            similarity_threshold
        ),

        # 混合报告保存门控版本。
        gate_version=gate_version,

        metrics=metrics,
        results=question_evaluations,
    )


def markdown_cell(
    value: str,
) -> str:
    """把文本整理成Markdown表格中的安全单行内容。"""

    # split()按连续空白切分文本，
    # join()再用一个空格重新连接。
    #
    # 这会把换行、制表符和多余空格
    # 统一压缩成单个空格。
    compact = " ".join(
        value.split()
    )

    # Markdown表格使用竖线分隔列，
    # 因此正文中的竖线必须转义。
    return compact.replace(
        "|",
        r"\|",
    )


def compact_preview(
    value: str,
    *,
    max_length: int = 260,
) -> str:
    """生成报告中使用的单行文本预览。"""

    if max_length < 1:
        raise ValueError(
            "max_length必须大于或等于1"
        )

    compact = markdown_cell(
        value
    )

    if len(compact) <= max_length:
        return compact

    # 超出长度时保留开头并添加省略号。
    return (
        compact[:max_length]
        .rstrip()
        + "……"
    )


def format_optional_rate(
    value: float | None,
) -> str:
    """把可选比例转换成Markdown显示文本。"""

    # None表示该题没有可计算的分母。
    if value is None:
        return "—"

    return f"{value:.3f}"


def build_failure_reasons(
    result: CitationQuestionEvaluation,
) -> list[str]:
    """解释一道题为什么没有通过引用评测。"""

    reasons: list[str] = []

    question = result.gold_question
    response = result.response

    if question.answerable:
        if response.abstained:
            reasons.append(
                "Gold标注为可回答，"
                "但API返回了拒答"
            )

        if (
            result.correct_citation_count
            < result.returned_citation_count
        ):
            reasons.append(
                "返回引用中包含"
                "未匹配Gold位置的引用"
            )

        if (
            result.matched_expected_evidence_count
            < result.expected_evidence_count
        ):
            reasons.append(
                "没有引用覆盖全部Gold预期证据"
            )

    else:
        if not response.abstained:
            reasons.append(
                "Gold标注为无答案，"
                "但API仍然生成了回答"
            )

        if result.returned_citation_count > 0:
            reasons.append(
                "无答案问题返回了引用"
            )

    if not reasons:
        # 理论上失败结果至少应有一个原因。
        #
        # 这里保留兜底文字，
        # 防止未来新增评测条件后报告出现空列表。
        reasons.append(
            "结果未满足当前完整通过条件"
        )

    return reasons


def render_markdown_report(
    report: CitationEvaluationReport,
) -> str:
    """把结构化引用评测报告转换成Markdown。"""

    metrics = report.metrics

    # 根据报告实际策略动态生成元数据行。
    #
    # 不能对None执行：
    #
    # f"{None:.3f}"
    #
    # 否则会产生当前测试发现的TypeError。
    strategy_lines: list[str] = []

    parameters = report.retrieval_parameters

    if parameters is None:
        # 没有嵌套参数表示兼容旧版纯向量报告。
        strategy_lines.append(
            "- 检索策略："
            "`vector_baseline`（兼容旧报告）"
        )
    else:
        strategy_lines.append(
            "- 检索策略："
            f"`{parameters.strategy}`"
        )

        # 混合策略必须保存单路候选深度。
        if parameters.candidate_k is not None:
            strategy_lines.append(
                "- 单路候选深度："
                f"{parameters.candidate_k}"
            )

        # 混合策略必须保存RRF排名常数。
        if parameters.rank_constant is not None:
            strategy_lines.append(
                "- RRF排名常数："
                f"{parameters.rank_constant}"
            )

        # 只有查询改写策略具有改写版本。
        if parameters.rewrite_version is not None:
            strategy_lines.append(
                "- 查询改写版本："
                f"`{parameters.rewrite_version}`"
            )

    # 纯向量报告显示单一相似度阈值。
    if report.similarity_threshold is not None:
        strategy_lines.append(
            "- 相似度拒答阈值："
            f"{report.similarity_threshold:.3f}"
        )

    # 混合门控报告显示门控版本。
    if report.gate_version is not None:
        strategy_lines.append(
            "- 证据门控版本："
            f"`{report.gate_version}`"
        )

    lines: list[str] = [
        "# Day 5–7 RAG回答引用评测",
        "",
        (
            "本报告由 "
            "`python -m scripts.evaluate_citations` "
            "调用真实知识库HTTP API生成。"
        ),
        "",
        f"- API：`{report.api_url}`",
        f"- LLM模型：`{report.llm_model}`",
        f"- Embedding模型：`{report.embedding_model}`",
        f"- Chroma Collection：`{report.collection_name}`",
        f"- Top-K：{report.top_k}",

        # 星号把strategy_lines中的每一个字符串
        # 分别展开成lines列表中的独立元素。
        *strategy_lines,

        "",
        "## 汇总指标",
        "",
        "| 指标 | 结果 |",
        "|---|---:|",
        (
            "| 问题总数 | "
            f"{metrics.total_question_count} |"
        ),
        (
            "| 可回答问题数 | "
            f"{metrics.answerable_question_count} |"
        ),
        (
            "| 无答案问题数 | "
            f"{metrics.unanswerable_question_count} |"
        ),
        (
            "| 实际回答的可回答问题数 | "
            f"{metrics.answered_answerable_count}/"
            f"{metrics.answerable_question_count} |"
        ),
        (
            "| 完整证据回答数 | "
            f"{metrics.fully_grounded_answer_count}/"
            f"{metrics.answerable_question_count} |"
        ),
        (
            "| 正确拒答的无答案问题数 | "
            f"{metrics.correctly_abstained_unanswerable_count}/"
            f"{metrics.unanswerable_question_count} |"
        ),
        (
            "| 返回引用总数 | "
            f"{metrics.returned_citation_count} |"
        ),
        (
            "| 正确引用总数 | "
            f"{metrics.correct_citation_count} |"
        ),
        (
            "| 引用正确率 | "
            f"{metrics.citation_correctness:.3f} |"
        ),
        (
            "| 预期证据总数 | "
            f"{metrics.expected_evidence_count} |"
        ),
        (
            "| 已覆盖预期证据数 | "
            f"{metrics.matched_expected_evidence_count} |"
        ),
        (
            "| 引用覆盖率 | "
            f"{metrics.citation_coverage:.3f} |"
        ),
        (
            "| 可回答率 | "
            f"{metrics.answerable_response_rate:.3f} |"
        ),
        (
            "| 完整证据回答率 | "
            f"{metrics.fully_grounded_answer_rate:.3f} |"
        ),
        (
            "| 无答案正确拒答率 | "
            f"{metrics.unanswerable_abstention_rate:.3f} |"
        ),
        "",
        "## 逐题结果",
        "",
        (
            "| ID | 类型 | Gold可回答 | API拒答 | "
            "正确引用/返回引用 | 已覆盖/预期证据 | "
            "引用正确率 | 引用覆盖率 | 结果 |"
        ),
        (
            "|---|---|---|---|---:|---:|"
            "---:|---:|---|"
        ),
    ]

    for result in report.results:
        question = result.gold_question
        response = result.response

        passed = (
            result.fully_grounded
            or result.correctly_abstained
        )

        lines.append(
            "| "
            f"`{question.question_id}` | "
            f"{question.question_type} | "
            f"{'是' if question.answerable else '否'} | "
            f"{'是' if response.abstained else '否'} | "
            f"{result.correct_citation_count}/"
            f"{result.returned_citation_count} | "
            f"{result.matched_expected_evidence_count}/"
            f"{result.expected_evidence_count} | "
            f"{format_optional_rate(result.citation_correctness)} | "
            f"{format_optional_rate(result.citation_coverage)} | "
            f"{'通过' if passed else '失败'} |"
        )

    failures = [
        result
        for result in report.results
        if not (
            result.fully_grounded
            or result.correctly_abstained
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
            "本次评测没有发现失败案例。"
        )
    else:
        for result in failures:
            question = result.gold_question
            response = result.response

            lines.extend(
                [
                    f"### {question.question_id}",
                    "",
                    (
                        "- 问题："
                        f"{markdown_cell(question.question)}"
                    ),
                    (
                        "- Gold状态："
                        f"{'可回答' if question.answerable else '无答案'}"
                    ),
                    (
                        "- API状态："
                        f"{'拒答' if response.abstained else '已回答'}"
                    ),
                    (
                        "- API回答："
                        f"{compact_preview(response.answer, max_length=500)}"
                    ),
                    "- 失败原因：",
                ]
            )

            for reason in build_failure_reasons(
                result
            ):
                lines.append(
                    f"  - {reason}"
                )

            if (
                question.reference_answer
                is not None
            ):
                lines.append(
                    "- 参考答案："
                    + compact_preview(
                        question.reference_answer,
                        max_length=500,
                    )
                )

            lines.append("- Gold预期证据：")

            if not question.expected_evidence:
                lines.append("  - 无")
            else:
                for expected in (
                    question.expected_evidence
                ):
                    lines.append(
                        "  - "
                        f"`{expected.source_file}`；"
                        f"`{expected.page_or_section}`"
                    )

            lines.append("- API返回引用：")

            if not response.citations:
                lines.append("  - 无")
            else:
                for citation in response.citations:
                    lines.append(
                        "  - "
                        f"Top-{citation.rank}；"
                        f"`{citation.source_file}`；"
                        f"`{citation.page_or_section}`；"
                        f"Chunk `{citation.chunk_id}`；"
                        f"相似度 `{citation.similarity:.3f}`；"
                        "预览："
                        f"{compact_preview(citation.excerpt)}"
                    )

            lines.append("")

    lines.extend(
        [
            "## 指标解释",
            "",
            (
                "- 引用正确率 = 匹配Gold文件和位置的"
                "返回引用数 / 全部返回引用数。"
            ),
            (
                "- 引用覆盖率 = 被引用命中的不同Gold证据数"
                " / 全部Gold预期证据数。"
            ),
            (
                "- 可回答率 = 实际回答的可回答问题数"
                " / Gold可回答问题数。"
            ),
            (
                "- 完整证据回答率要求问题被回答、"
                "所有返回引用正确，并覆盖全部预期证据。"
            ),
            (
                "- 无答案正确拒答率要求无答案问题返回"
                "`abstained=true`且引用为空。"
            ),
            (
                "- 本报告评价引用位置与拒答行为，"
                "不等同于对回答全部语义事实进行自动判定。"
            ),
        ]
    )

    return "\n".join(lines) + "\n"


def write_reports(
    *,
    report: CitationEvaluationReport,
    json_output: Path,
    markdown_output: Path,
) -> None:
    """把同一结构化报告写成JSON和Markdown。"""

    # parents=True创建缺失的父目录；
    # exist_ok=True表示目录已存在时不报错。
    json_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    markdown_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # mode="json"把datetime等类型转换成
    # JSON能够表示的字符串或基础类型。
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


def parse_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """解析引用评测脚本的命令行参数。"""

    parser = argparse.ArgumentParser(
        description=(
            "通过知识库HTTP API对Gold Question"
            "执行真实回答与引用评测"
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
        "--api-url",
        default=(
            "http://127.0.0.1:8000"
            "/api/v1/knowledge/query"
        ),
        help="知识库查询API的完整地址",
    )

    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path(
            "docs/citation-evaluation.json"
        ),
        help="JSON评测报告输出路径",
    )

    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path(
            "docs/citation-evaluation.md"
        ),
        help="Markdown评测报告输出路径",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help=(
            "每道问题请求知识库API时使用的"
            "Top-K检索结果数量"
        ),
    )

    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=120.0,
        help=(
            "每次知识库HTTP请求允许等待的"
            "最长秒数"
        ),
    )

    # argv=None时，argparse自动读取真实命令行参数。
    #
    # 测试时可以传入列表，例如：
    # parse_args(["--top-k", "5"])
    #
    # 这样测试不必修改全局sys.argv。
    return parser.parse_args(argv)


async def run_real_evaluation(
    *,
    settings: Settings,
    questions_path: Path,
    api_url: str,
    json_output: Path,
    markdown_output: Path,
    top_k: int,
    timeout_seconds: float,
) -> CitationEvaluationReport:
    """调用真实知识库API并写出引用评测报告。"""

    if timeout_seconds <= 0:
        raise ValueError(
            "timeout_seconds必须大于0"
        )

    # 逐行读取JSONL，并将每行验证成GoldQuestion。
    #
    # 文件不存在、JSON非法、字段不符合契约或题号重复时，
    # 都会在发送HTTP请求前失败。
    questions = load_gold_questions(
        questions_path
    )

    # 取得模型名称，用于验证配置并记录报告元数据。
    #
    # 这里不创建LLM或Embedding客户端；
    # 真正的模型调用发生在FastAPI服务内部。
    llm_model = get_llm_model(settings)
    embedding_model = get_embedding_model(
        settings
    )

    # 这些参数必须与在线API依赖工厂
    # 实际装配的默认策略完全一致。
    retrieval_parameters = (
        CandidateStrategyParameters(
            strategy="hybrid_rrf_rewrite",
            top_k=top_k,
            candidate_k=(
                DEFAULT_HYBRID_CANDIDATE_K
            ),
            rank_constant=(
                DEFAULT_RRF_RANK_CONSTANT
            ),
            rewrite_version=(
                DETERMINISTIC_QUERY_REWRITE_VERSION
            ),
        )
    )

    # httpx.Timeout为每次HTTP请求设置超时约束。
    #
    # 设置120秒，是因为知识库API内部可能依次执行：
    # Query Embedding、Chroma检索和LLM回答。
    timeout = httpx.Timeout(
        timeout_seconds
    )

    # async with保证评测成功或失败后，
    # AsyncClient及其连接池都会被关闭。
    async with httpx.AsyncClient(
        timeout=timeout,
    ) as client:
        report = await evaluate_citations_via_api(
            questions=questions,
            client=client,
            api_url=api_url,
            llm_model=llm_model,
            embedding_model=embedding_model,
            collection_name=(
                settings.chroma_collection_name
            ),
            top_k=top_k,

            # 保存真实在线候选策略参数。
            retrieval_parameters=(
                retrieval_parameters
            ),

            # 混合门控不使用单一全局相似度阈值。
            similarity_threshold=None,

            # 保存在线API实际使用的门控规则版本。
            gate_version=(
                HYBRID_EVIDENCE_GATE_VERSION
            ),
        )

    # HTTP客户端关闭后，把同一份结构化报告分别写成：
    # 1. JSON：供程序继续分析；
    # 2. Markdown：供人工阅读和项目交付。
    write_reports(
        report=report,
        json_output=json_output,
        markdown_output=markdown_output,
    )

    return report


def print_summary(
    report: CitationEvaluationReport,
) -> None:
    """在终端打印最重要的引用评测指标。"""

    metrics = report.metrics

    print("引用评测完成")
    print(
        "问题总数："
        f"{metrics.total_question_count}"
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
        "实际回答的可回答问题数："
        f"{metrics.answered_answerable_count}"
    )
    print(
        "引用正确率："
        f"{metrics.citation_correctness:.3f}"
    )
    print(
        "引用覆盖率："
        f"{metrics.citation_coverage:.3f}"
    )
    print(
        "可回答问题响应率："
        f"{metrics.answerable_response_rate:.3f}"
    )
    print(
        "完整证据回答率："
        f"{metrics.fully_grounded_answer_rate:.3f}"
    )
    print(
        "无答案问题正确拒答率："
        f"{metrics.unanswerable_abstention_rate:.3f}"
    )


def main() -> None:
    """引用评测脚本的同步命令行入口。"""

    # 先解析命令行参数。
    args = parse_args()

    # get_settings()读取环境变量和.env；
    # 同一进程内重复调用会复用缓存的Settings对象。
    settings = get_settings()

    # asyncio.run()创建事件循环，执行异步评测，
    # 等待全部20道题完成后关闭事件循环。
    report = asyncio.run(
        run_real_evaluation(
            settings=settings,
            questions_path=args.questions,
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
    # 方便从终端直接确认报告写到了哪里。
    print(
        "JSON报告："
        f"{args.json_output.resolve()}"
    )
    print(
        "Markdown报告："
        f"{args.markdown_output.resolve()}"
    )


# 使用python -m scripts.evaluate_citations执行本文件时，
# __name__等于"__main__"，因此调用main()。
#
# 测试代码只是import本模块时，
# __name__等于"scripts.evaluate_citations"，
# 不会意外运行真实API评测。
if __name__ == "__main__":
    main()
