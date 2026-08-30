"""通过Robot Diagnostic Agent HTTP API执行批量评测。

本模块负责：

1. 通过HTTP执行一条Agent固定评测场景；
2. 把HTTP、传输和响应契约错误转换成可审计失败；
3. 顺序执行至少8条固定场景；
4. 汇总工具选择、任务完成、引用和安全拒答指标；
5. 构造经过交叉校验的AgentEvaluationReport；
6. 把结构化报告写成JSON和Markdown；
7. 输出不包含生成时间和敏感正文的终端摘要；
8. 解析真实评测使用的命令行参数。

本模块不会直接创建Planner、执行Agent工具或访问ChromaDB。
这些操作发生在被评测的FastAPI服务内部。

JUnit XML由pytest的--junitxml参数生成，
不由本评测脚本伪造。
"""

# asyncio用于从同步命令行入口运行异步评测流程。
#
# Agent API调用依赖httpx.AsyncClient，
# 因此真实批量评测必须在事件循环中执行。
import asyncio

# argparse用于声明和解析命令行参数。
#
# ArgumentParser负责生成--help说明，
# parse_args()负责把字符串参数转换成Namespace对象。
import argparse

# json用于把Pydantic报告转换成UTF-8 JSON文本。
import json

# Sequence表示可重复遍历的只读序列。
from collections.abc import (
    Sequence,
)

# Path封装文件系统路径。
#
# 它提供parent、mkdir()、write_text()和resolve()
# 等跨平台路径操作。
from pathlib import Path


# Sequence表示可重复遍历的只读序列。
#
# 批量评测可以接收list或tuple，
# 但不会修改调用方提供的场景集合。
from collections.abc import (
    Sequence,
)


# httpx提供异步HTTP客户端、响应对象、
# MockTransport兼容接口和网络异常类型。
import httpx

# ValidationError表示HTTP正文虽然是合法JSON，
# 但没有通过AgentDiagnosisResponse数据契约。
from pydantic import (
    ValidationError,
)

# Settings定义应用配置契约；
# get_settings()读取环境变量和.env并缓存结果。
from app.config import (
    Settings,
    get_settings,
)

# 评测报告必须记录实际使用的Planner Prompt版本。
#
# 直接导入生产常量可以避免评测脚本单独硬编码版本，
# 导致Prompt升级后报告仍写旧版本。
from app.agent.openai_planner import (
    AGENT_PLANNER_PROMPT_VERSION,
)


from app.schemas.agent_api import (
    AgentDiagnosisResponse,
)
from app.schemas.agent_evaluation import (
    AgentEvaluationReport,
    AgentEvaluationScenario,
    AgentScenarioEvaluation,
)
from app.services.agent_evaluation import (
    calculate_agent_evaluation_metrics,
    evaluate_agent_request_failure,
    evaluate_agent_response,
)

# get_embedding_model()读取并检查EMBEDDING_MODEL。
#
# 本脚本只把名称写入报告，
# 不创建Embedding客户端。
from app.services.embedding_client import (
    get_embedding_model,
)

# load_agent_evaluation_scenarios()负责逐行读取、
# 校验JSONL并检查场景编号唯一性。
from app.services.evaluation import (
    load_agent_evaluation_scenarios,
)

# get_llm_model()读取并检查LLM_MODEL。
#
# Agent Planner的真实模型调用发生在FastAPI服务中；
# 本脚本只记录配置名称。
from app.services.llm_client import (
    get_llm_model,
)


# Week 4任务6明确要求至少8条固定Agent场景。
#
# 这是正式批量评测的最低规模，
# 不能用一个成功样例冒充批量结果。
MIN_AGENT_EVALUATION_SCENARIO_COUNT = 8


# 正式Agent评测默认读取的固定场景文件。
DEFAULT_AGENT_SCENARIOS_PATH = Path(
    "data/eval/agent_scenarios.jsonl"
)

# 被评测的公开Agent诊断接口。
DEFAULT_AGENT_API_URL = (
    "http://127.0.0.1:8000"
    "/api/v1/agent/diagnose"
)

# JSON报告保留完整、机器可读取的结构化结果。
DEFAULT_AGENT_JSON_OUTPUT = Path(
    "docs/agent-evaluation.json"
)

# Markdown报告用于人工审核指标和失败场景。
DEFAULT_AGENT_MARKDOWN_OUTPUT = Path(
    "docs/agent-evaluation.md"
)

# 单个HTTP请求允许覆盖一次多步Agent执行，
# 因而使用比普通接口更长的默认超时。
DEFAULT_AGENT_TIMEOUT_SECONDS = 120.0

# 本常量描述评测场景、评分规则和指标口径的版本。
#
# 它不是Agent Prompt版本：
#
# agent-evaluation-v1描述评测方法；
# agent-tool-calling-v2描述Planner Prompt。
AGENT_EVALUATION_VERSION = (
    "agent-evaluation-v1"
)


def _get_response_request_id(
    response: httpx.Response,
) -> str | None:
    """从HTTP响应头读取并清理追踪编号。

    HTTP Header不区分大小写，因此既可以读取
    X-Request-ID，也可以读取x-request-id。

    缺失或只有空白时返回None，不伪造missing等
    看起来像真实追踪编号的占位字符串。
    """

    request_id = response.headers.get(
        "X-Request-ID"
    )

    if request_id is None:
        return None

    cleaned_request_id = (
        request_id.strip()
    )

    if not cleaned_request_id:
        return None

    return cleaned_request_id


async def evaluate_agent_scenario_via_api(
    *,
    scenario: AgentEvaluationScenario,
    client: httpx.AsyncClient,
    api_url: str,
) -> AgentScenarioEvaluation:
    """调用一次Agent API并返回统一场景评分。

    成功路径：

    场景请求
    → HTTP 2xx
    → 合法JSON
    → AgentDiagnosisResponse
    → evaluate_agent_response()
    → 成功评分结果。

    失败路径：

    超时、连接失败、非2xx、非法JSON或无效Schema
    → evaluate_agent_request_failure()
    → 不含虚假Agent观察的失败评分结果。
    """

    if not isinstance(
        scenario,
        AgentEvaluationScenario,
    ):
        raise TypeError(
            "scenario必须是"
            "AgentEvaluationScenario"
        )

    if not isinstance(
        client,
        httpx.AsyncClient,
    ):
        raise TypeError(
            "client必须是httpx.AsyncClient"
        )

    if (
        not isinstance(api_url, str)
        or not api_url.strip()
    ):
        raise ValueError(
            "api_url不能为空"
        )

    cleaned_api_url = api_url.strip()

    # model_dump()把Pydantic请求模型转换成普通字典。
    #
    # exclude_none=True表示没有提供task_goal时，
    # 请求体中不发送"task_goal": null。
    #
    # 这里只会发送公开AgentDiagnosisRequest字段，
    # 不会发送tools、max_steps或权限配置。
    request_body = (
        scenario.request.model_dump(
            exclude_none=True
        )
    )

    try:
        # json=让httpx自动完成：
        #
        # 1. Python字典到JSON的序列化；
        # 2. 中文UTF-8编码；
        # 3. Content-Type请求头设置；
        # 4. 异步等待HTTP响应。
        http_response = await client.post(
            cleaned_api_url,
            json=request_body,
        )
    except httpx.TimeoutException:
        # 超时发生时没有收到HTTP响应，
        # 所以不存在可信状态码和request_id。
        return evaluate_agent_request_failure(
            scenario=scenario,
            http_status_code=None,
            request_id=None,
            request_error=(
                "Agent API请求超时"
            ),
        )
    except httpx.RequestError:
        # RequestError包含连接失败、DNS失败、
        # 连接被重置等传输层错误。
        #
        # 不把原始异常文本写入报告，
        # 避免泄漏内部主机名和网络结构。
        return evaluate_agent_request_failure(
            scenario=scenario,
            http_status_code=None,
            request_id=None,
            request_error=(
                "无法连接Agent API"
            ),
        )

    request_id = _get_response_request_id(
        http_response
    )

    # 非2xx属于一次真实HTTP响应，
    # 因而可以保留状态码和响应头中的request_id。
    #
    # 不读取或复制错误正文，避免报告保存上游细节。
    if not (
        200
        <= http_response.status_code
        < 300
    ):
        return evaluate_agent_request_failure(
            scenario=scenario,
            http_status_code=(
                http_response.status_code
            ),
            request_id=request_id,
            request_error=(
                "Agent API返回错误状态："
                f"{http_response.status_code}"
            ),
        )

    try:
        # response.json()把响应正文解析成Python对象。
        #
        # httpx可能使用不同JSON实现，
        # 其解析错误统一属于ValueError。
        raw_response = http_response.json()
    except ValueError:
        return evaluate_agent_request_failure(
            scenario=scenario,
            http_status_code=(
                http_response.status_code
            ),
            request_id=request_id,
            request_error=(
                "Agent API返回非法JSON"
            ),
        )

    try:
        # 即使HTTP状态是200且正文是合法JSON，
        # 也不能直接信任其中的字段。
        #
        # model_validate()会检查：
        #
        # 1. request_id关系；
        # 2. DiagnosisReport引用白名单；
        # 3. 执行状态和终止原因；
        # 4. step_id连续性；
        # 5. 工具轨迹状态和错误码；
        # 6. 模拟遥测和测试草案契约。
        api_response = (
            AgentDiagnosisResponse
            .model_validate(
                raw_response
            )
        )
    except ValidationError:
        return evaluate_agent_request_failure(
            scenario=scenario,
            http_status_code=(
                http_response.status_code
            ),
            request_id=request_id,
            request_error=(
                "Agent API响应不符合数据契约"
            ),
        )

    # 正常响应交给纯评分器。
    #
    # HTTP执行器本身不重复实现工具、引用和
    # 任务完成规则，避免出现两套评分口径。
    return evaluate_agent_response(
        scenario=scenario,
        response=api_response,
        http_status_code=(
            http_response.status_code
        ),
    )


async def evaluate_agent_scenarios_via_api(
    *,
    scenarios: Sequence[
        AgentEvaluationScenario
    ],
    client: httpx.AsyncClient,
    api_url: str,
    evaluation_version: str,
    scenario_source: str,
    planner_model: str,
    planner_prompt_version: str,
    embedding_model: str,
    collection_name: str,
) -> AgentEvaluationReport:
    """顺序执行完整Agent场景集并生成结构化报告。

    执行过程：

    1. 在联网前验证场景数量、类型和唯一性；
    2. 在联网前验证报告元数据；
    3. 按输入顺序逐条调用Agent API；
    4. 单条失败转换成失败评分并继续；
    5. 根据全部逐场景结果计算汇总指标；
    6. 构造不包含生成时间的最终报告。
    """

    # 排除str、bytes等虽然可以遍历，
    # 但并不表示场景集合的对象。
    if (
        isinstance(
            scenarios,
            (
                str,
                bytes,
                bytearray,
            ),
        )
        or not isinstance(
            scenarios,
            Sequence,
        )
    ):
        raise TypeError(
            "scenarios必须是"
            "AgentEvaluationScenario序列"
        )

    # 转换成tuple建立本次评测使用的稳定快照。
    #
    # 如果调用方传入list，后续代码也不会修改原列表。
    scenario_items = tuple(
        scenarios
    )

    if (
        len(scenario_items)
        < MIN_AGENT_EVALUATION_SCENARIO_COUNT
    ):
        raise ValueError(
            "Agent批量评测至少需要"
            f"{MIN_AGENT_EVALUATION_SCENARIO_COUNT}"
            "条场景"
        )

    # Python类型注解不会自动验证列表中的元素，
    # 因此必须逐项进行运行时检查。
    for index, scenario in enumerate(
        scenario_items
    ):
        if not isinstance(
            scenario,
            AgentEvaluationScenario,
        ):
            raise TypeError(
                f"scenarios[{index}]必须是"
                "AgentEvaluationScenario"
            )

    scenario_ids = [
        scenario.scenario_id
        for scenario in scenario_items
    ]

    # 同一场景不能重复发出请求或重复进入指标分母。
    if len(scenario_ids) != len(
        set(scenario_ids)
    ):
        raise ValueError(
            "Agent批量评测场景包含"
            "重复scenario_id"
        )

    if not isinstance(
        client,
        httpx.AsyncClient,
    ):
        raise TypeError(
            "client必须是httpx.AsyncClient"
        )

    # 这些字段会进入最终报告，用于回答：
    #
    # 1. 使用了哪份场景数据；
    # 2. 调用了哪个API；
    # 3. 使用了什么模型；
    # 4. 使用了哪个Prompt版本；
    # 5. 检索使用了哪个Collection。
    metadata_values = {
        "api_url": api_url,
        "evaluation_version": (
            evaluation_version
        ),
        "scenario_source": (
            scenario_source
        ),
        "planner_model": planner_model,
        "planner_prompt_version": (
            planner_prompt_version
        ),
        "embedding_model": (
            embedding_model
        ),
        "collection_name": (
            collection_name
        ),
    }

    cleaned_metadata: dict[
        str,
        str,
    ] = {}

    # 所有元数据必须在发送第一条HTTP请求前完成校验。
    #
    # 否则可能运行完8条场景后，才发现报告无法创建。
    for field_name, value in (
        metadata_values.items()
    ):
        if not isinstance(value, str):
            raise TypeError(
                f"{field_name}必须是字符串"
            )

        cleaned_value = value.strip()

        if not cleaned_value:
            raise ValueError(
                f"{field_name}不能为空"
            )

        cleaned_metadata[field_name] = (
            cleaned_value
        )

    scenario_results: list[
        AgentScenarioEvaluation
    ] = []

    # 有意使用顺序执行，而不是asyncio.gather()。
    #
    # 原因：
    #
    # 1. Agent内部会调用生成式Planner；
    # 2. 多个场景并发容易触发LLM限流；
    # 3. 服务端日志更容易按场景追踪；
    # 4. 报告顺序与JSONL顺序稳定一致；
    # 5. 某条失败后仍然可以明确知道后续执行顺序。
    for scenario in scenario_items:
        result = (
            await evaluate_agent_scenario_via_api(
                scenario=scenario,
                client=client,
                api_url=(
                    cleaned_metadata["api_url"]
                ),
            )
        )

        scenario_results.append(
            result
        )

    # 汇总函数只读取逐场景结果并机械计算指标。
    metrics = (
        calculate_agent_evaluation_metrics(
            results=scenario_results,
        )
    )

    # AgentEvaluationReport会再次从results重新统计计数，
    # 检查metrics是否真正来自当前这批结果。
    #
    # 报告契约故意不包含generated_at，
    # 避免同一配置重复执行产生无意义的时间差异。
    return AgentEvaluationReport(
        evaluation_version=(
            cleaned_metadata[
                "evaluation_version"
            ]
        ),
        scenario_source=(
            cleaned_metadata[
                "scenario_source"
            ]
        ),
        api_url=(
            cleaned_metadata["api_url"]
        ),
        planner_model=(
            cleaned_metadata[
                "planner_model"
            ]
        ),
        planner_prompt_version=(
            cleaned_metadata[
                "planner_prompt_version"
            ]
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
        metrics=metrics,
        results=tuple(
            scenario_results
        ),
    )


def _markdown_cell(
    value: object,
) -> str:
    """把任意公开值转换成安全的单行Markdown表格文本。

    本函数只处理报告中的脱敏字段，
    不接收完整日志、文档正文或模型原始响应。
    """

    if value is None:
        return "-"

    # split()删除换行和连续空白，
    # join()将它们统一成一个普通空格。
    compact = " ".join(
        str(value).split()
    )

    # Markdown表格使用竖线分隔单元格，
    # 因此数据中的竖线必须转义。
    return compact.replace(
        "|",
        r"\|",
    )


def render_markdown_report(
    report: AgentEvaluationReport,
) -> str:
    """把结构化Agent评测报告转换成可审计Markdown。

    Markdown只展示运行配置、汇总指标、
    脱敏场景状态和失败原因。

    完整结构化字段仍保存在JSON报告中。
    """

    if not isinstance(
        report,
        AgentEvaluationReport,
    ):
        raise TypeError(
            "report必须是AgentEvaluationReport"
        )

    metrics = report.metrics

    lines = [
        "# Robot Diagnostic Agent评测报告",
        "",
        "## 评测配置",
        "",
        "| 配置项 | 值 |",
        "|---|---|",
        (
            "| 评测版本 | "
            f"`{_markdown_cell(report.evaluation_version)}` |"
        ),
        (
            "| 场景文件 | "
            f"`{_markdown_cell(report.scenario_source)}` |"
        ),
        (
            "| Agent API | "
            f"`{_markdown_cell(report.api_url)}` |"
        ),
        (
            "| Planner模型 | "
            f"`{_markdown_cell(report.planner_model)}` |"
        ),
        (
            "| Planner Prompt版本 | "
            f"`{_markdown_cell(report.planner_prompt_version)}` |"
        ),
        (
            "| Embedding模型 | "
            f"`{_markdown_cell(report.embedding_model)}` |"
        ),
        (
            "| Chroma Collection | "
            f"`{_markdown_cell(report.collection_name)}` |"
        ),
        "",
        "## 汇总指标",
        "",
        "| 指标 | 计数 | 比率 |",
        "|---|---:|---:|",
        (
            "| 请求成功率 | "
            f"{metrics.request_success_count}/"
            f"{metrics.scenario_count} | "
            f"{metrics.request_success_rate:.3f} |"
        ),
        (
            "| 场景总通过率 | "
            f"{metrics.passed_scenario_count}/"
            f"{metrics.scenario_count} | "
            f"{metrics.scenario_pass_rate:.3f} |"
        ),
        (
            "| 工具选择正确率 | "
            f"{metrics.tool_selection_correct_count}/"
            f"{metrics.scenario_count} | "
            f"{metrics.tool_selection_accuracy:.3f} |"
        ),
        (
            "| 任务完成率 | "
            f"{metrics.completed_task_count}/"
            f"{metrics.expected_completion_count} | "
            f"{metrics.task_completion_rate:.3f} |"
        ),
        (
            "| 引用正确率 | "
            f"{metrics.correct_citation_count}/"
            f"{metrics.returned_citation_count} | "
            f"{metrics.citation_correctness:.3f} |"
        ),
        (
            "| 引用覆盖率 | "
            f"{metrics.matched_expected_evidence_count}/"
            f"{metrics.expected_evidence_count} | "
            f"{metrics.citation_coverage:.3f} |"
        ),
        (
            "| 安全拒答率 | "
            f"{metrics.correct_safe_refusal_count}/"
            f"{metrics.safe_refusal_case_count} | "
            f"{metrics.safe_refusal_rate:.3f} |"
        ),
        (
            "| 平均工具步骤数 | "
            f"{metrics.total_step_count}/"
            f"{metrics.scenario_count} | "
            f"{metrics.average_steps:.3f} |"
        ),
        "",
        "## 逐场景结果",
        "",
        (
            "| 场景 | 类型 | 请求 | 工具顺序 | "
            "执行状态 | 终止原因 | Planner结束原因 | "
            "诊断状态 | 步骤 | 引用正确 | 证据覆盖 | 通过 |"
        ),
        (
            "|---|---|---|---|---|---|---|---|"
            "---:|---:|---:|---|"
        ),
    ]

    for result in report.results:
        tool_sequence = (
            " → ".join(
                result.actual_tool_sequence
            )
            if result.actual_tool_sequence
            else "无"
        )

        correct_citation_count = sum(
            citation.correct
            for citation
            in result.citation_evaluations
        )

        lines.append(
            "| "
            f"`{_markdown_cell(result.scenario_id)}` | "
            f"{_markdown_cell(result.scenario_type)} | "
            f"{'成功' if result.request_succeeded else '失败'} | "
            f"{_markdown_cell(tool_sequence)} | "
            f"{_markdown_cell(result.actual_execution_state)} | "
            f"{_markdown_cell(result.actual_termination_reason)} | "
            f"{_markdown_cell(result.actual_finish_reason)} | "
            f"{_markdown_cell(result.actual_diagnosis_status)} | "
            f"{result.step_count} | "
            f"{correct_citation_count}/"
            f"{len(result.citation_evaluations)} | "
            f"{len(result.matched_expected_evidence)}/"
            f"{len(result.expected_evidence)} | "
            f"{'通过' if result.passed else '失败'} |"
        )

    failed_results = [
        result
        for result in report.results
        if not result.passed
    ]

    lines.extend(
        [
            "",
            "## 失败场景",
            "",
        ]
    )

    if not failed_results:
        lines.append(
            "本次评测没有失败场景。"
        )
    else:
        for result in failed_results:
            lines.extend(
                [
                    (
                        "### "
                        f"{_markdown_cell(result.scenario_id)}"
                    ),
                    "",
                    (
                        "- 场景类型："
                        f"`{_markdown_cell(result.scenario_type)}`"
                    ),
                ]
            )

            if result.request_error is not None:
                lines.append(
                    "- 请求错误："
                    + _markdown_cell(
                        result.request_error
                    )
                )

            lines.append("- 失败原因：")

            for reason in result.failure_reasons:
                lines.append(
                    "  - "
                    + _markdown_cell(reason)
                )

            lines.append("")

    lines.extend(
        [
            "## 指标说明",
            "",
            (
                "- 工具选择正确率检查必需工具是否出现、"
                "实际工具是否都在允许列表中，以及是否调用禁止工具。"
            ),
            (
                "- 任务完成率只统计"
                "`expects_task_completion=true`的Gold场景。"
            ),
            (
                "- 引用正确率衡量返回引用中有多少命中"
                "Gold的来源文件和位置。"
            ),
            (
                "- 引用覆盖率衡量Gold预期证据中"
                "有多少被实际引用覆盖。"
            ),
            (
                "- 平均工具步骤数使用全部合法响应的"
                "公开工具步骤数除以场景总数。"
            ),
            (
                "- 安全拒答率只统计"
                "`expects_safe_refusal=true`的场景，"
                "并要求最终诊断状态为`abstained`。"
            ),
            (
                "- 报告不保存完整日志、文档正文、"
                "Planner原始输出或模型私有思维链。"
            ),
        ]
    )

    # 结尾保留一个换行，使重复生成的文本文件稳定。
    return "\n".join(lines) + "\n"


def write_reports(
    *,
    report: AgentEvaluationReport,
    json_output: Path,
    markdown_output: Path,
) -> None:
    """将同一Agent评测结果写成JSON和Markdown。"""

    if not isinstance(
        report,
        AgentEvaluationReport,
    ):
        raise TypeError(
            "report必须是AgentEvaluationReport"
        )

    if not isinstance(json_output, Path):
        raise TypeError(
            "json_output必须是Path"
        )

    if not isinstance(markdown_output, Path):
        raise TypeError(
            "markdown_output必须是Path"
        )

    # parents=True允许建立多层父目录；
    # exist_ok=True表示目录已存在时不报错。
    json_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    markdown_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # mode="json"确保嵌套Pydantic对象全部转换成
    # JSON可以表示的字典、列表、字符串和数值。
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
    """解析Agent批量评测的命令行参数。"""

    parser = argparse.ArgumentParser(
        description=(
            "通过Robot Diagnostic Agent HTTP API"
            "执行固定场景批量评测"
        )
    )

    parser.add_argument(
        "--scenarios",
        type=Path,
        default=DEFAULT_AGENT_SCENARIOS_PATH,
        help="Agent固定评测场景JSONL路径",
    )

    parser.add_argument(
        "--api-url",
        default=DEFAULT_AGENT_API_URL,
        help="Agent诊断API的完整地址",
    )

    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_AGENT_TIMEOUT_SECONDS,
        help="每条Agent HTTP请求的超时秒数",
    )

    parser.add_argument(
        "--json-output",
        type=Path,
        default=DEFAULT_AGENT_JSON_OUTPUT,
        help="JSON评测报告输出路径",
    )

    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=DEFAULT_AGENT_MARKDOWN_OUTPUT,
        help="Markdown评测报告输出路径",
    )

    args = parser.parse_args(argv)

    # argparse负责字符串转float，
    # 业务层继续拒绝没有意义的零或负超时。
    if args.timeout_seconds <= 0:
        parser.error(
            "--timeout-seconds必须大于0"
        )

    return args


def print_summary(
    *,
    report: AgentEvaluationReport,
    json_path: Path,
    markdown_path: Path,
) -> None:
    """输出不包含敏感正文的Agent评测摘要。"""

    if not isinstance(
        report,
        AgentEvaluationReport,
    ):
        raise TypeError(
            "report必须是AgentEvaluationReport"
        )

    if not isinstance(json_path, Path):
        raise TypeError(
            "json_path必须是Path"
        )

    if not isinstance(markdown_path, Path):
        raise TypeError(
            "markdown_path必须是Path"
        )

    metrics = report.metrics

    print("Agent评测完成")
    print(
        f"场景数：{metrics.scenario_count}"
    )
    print(
        "请求成功率："
        f"{metrics.request_success_rate:.3f}"
    )
    print(
        "场景总通过率："
        f"{metrics.scenario_pass_rate:.3f}"
    )
    print(
        "工具选择正确率："
        f"{metrics.tool_selection_accuracy:.3f}"
    )
    print(
        "任务完成率："
        f"{metrics.task_completion_rate:.3f}"
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
        "平均工具步骤数："
        f"{metrics.average_steps:.3f}"
    )
    print(
        "安全拒答率："
        f"{metrics.safe_refusal_rate:.3f}"
    )
    print(
        "JSON报告："
        f"{json_path.resolve()}"
    )
    print(
        "Markdown报告："
        f"{markdown_path.resolve()}"
    )


async def run_real_evaluation(
    *,
    settings: Settings,
    scenarios_path: Path,
    api_url: str,
    json_output: Path,
    markdown_output: Path,
    timeout_seconds: float,
) -> AgentEvaluationReport:
    """执行一次真实Agent固定场景评测并写出报告。

    本函数负责连接已有组件，不重新实现：

    1. JSONL字段校验；
    2. 单场景HTTP处理；
    3. Agent评分；
    4. 指标计算；
    5. Markdown渲染。
    """

    if not isinstance(settings, Settings):
        raise TypeError(
            "settings必须是Settings"
        )

    if not isinstance(scenarios_path, Path):
        raise TypeError(
            "scenarios_path必须是Path"
        )

    if not isinstance(json_output, Path):
        raise TypeError(
            "json_output必须是Path"
        )

    if not isinstance(markdown_output, Path):
        raise TypeError(
            "markdown_output必须是Path"
        )

    # bool是int的子类，也可以参与数值比较，
    # 但True/False不应被当作超时秒数。
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(
            timeout_seconds,
            (int, float),
        )
    ):
        raise TypeError(
            "timeout_seconds必须是数值"
        )

    if timeout_seconds <= 0:
        raise ValueError(
            "timeout_seconds必须大于0"
        )

    # 整个JSONL会在第一次HTTP请求前完成校验。
    #
    # 文件不存在、空行、非法JSON、字段错误和
    # scenario_id重复都会在这里停止评测。
    scenarios = (
        load_agent_evaluation_scenarios(
            scenarios_path
        )
    )

    # 这里只读取模型名称作为报告元数据。
    #
    # get_llm_model()和get_embedding_model()
    # 不创建SDK客户端，也不会发出网络请求。
    planner_model = get_llm_model(
        settings
    )
    embedding_model = get_embedding_model(
        settings
    )

    # Timeout是httpx的超时配置对象。
    #
    # 一条Agent请求内部可能执行多轮Planner和多个工具，
    # 因此默认超时比普通单次接口更长。
    timeout = httpx.Timeout(
        timeout_seconds
    )

    # async with管理AsyncClient生命周期。
    #
    # 无论批量评测正常结束还是抛出异常，
    # 离开代码块时都会关闭连接池。
    async with httpx.AsyncClient(
        timeout=timeout,
    ) as client:
        report = (
            await evaluate_agent_scenarios_via_api(
                scenarios=scenarios,
                client=client,
                api_url=api_url,
                evaluation_version=(
                    AGENT_EVALUATION_VERSION
                ),
                scenario_source=str(
                    scenarios_path
                ),
                planner_model=planner_model,
                planner_prompt_version=(
                    AGENT_PLANNER_PROMPT_VERSION
                ),
                embedding_model=(
                    embedding_model
                ),
                collection_name=(
                    settings
                    .chroma_collection_name
                ),
            )
        )

    # 客户端关闭后再写报告。
    #
    # 这一步只消费已经完成交叉校验的
    # AgentEvaluationReport。
    write_reports(
        report=report,
        json_output=json_output,
        markdown_output=markdown_output,
    )

    return report


def main() -> None:
    """Agent评测脚本的同步命令行入口。"""

    # argv省略时，argparse读取真实命令行参数。
    args = parse_args()

    # 第一次调用读取环境变量和.env；
    # get_settings()使用lru_cache复用同一Settings。
    settings = get_settings()

    # asyncio.run()负责：
    #
    # 1. 创建事件循环；
    # 2. 执行异步评测；
    # 3. 等待全部场景完成；
    # 4. 关闭事件循环。
    report = asyncio.run(
        run_real_evaluation(
            settings=settings,
            scenarios_path=args.scenarios,
            api_url=args.api_url,
            json_output=args.json_output,
            markdown_output=(
                args.markdown_output
            ),
            timeout_seconds=(
                args.timeout_seconds
            ),
        )
    )

    # 终端只显示汇总指标和报告路径，
    # 不输出完整日志、证据正文或Planner响应。
    print_summary(
        report=report,
        json_path=args.json_output,
        markdown_path=(
            args.markdown_output
        ),
    )


# 使用下面的命令执行模块时：
#
# python -m scripts.evaluate_agent
#
# __name__等于"__main__"，因此进入main()。
#
# pytest只是import本模块时，
# __name__等于"scripts.evaluate_agent"，
# 不会意外启动真实评测。
if __name__ == "__main__":
    main()