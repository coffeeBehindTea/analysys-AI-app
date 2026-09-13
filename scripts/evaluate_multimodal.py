"""运行 Week 5 多模态 Agent 正式可靠性评测。

本模块负责：

1. 解析命令行参数；
2. 读取并校验不少于 30 条多模态 Gold 场景；
3. 根据 Settings 创建图片输入适配器；
4. 通过真实 HTTP 接口顺序执行全部场景；
5. 生成经过数据契约交叉校验的正式报告；
6. 写入 JSON、Markdown 和至少三份完整脱敏轨迹；
7. 输出便于人工核对的终端摘要。

本模块不负责：

1. 在当前进程内执行 Agent Runner；
2. 直接调用生产 Planner、Vision 或知识库；
3. 重新实现评分规则；
4. 保存图片 Base64、本地图片路径或请求级图片元数据；
5. 保存模型私有思维链；
6. 使用 LLM-as-judge 替代确定性评分。

公开VisionObservation可以保留source_image_sha256作为来源
完整性指纹；该字段不能还原图片，也不包含本地路径或图片正文。

使用--enable-llm-judge时，本脚本会额外创建一个文本Judge，
但它只判断确定性规则未解决的视觉表达，并只写入辅助指标。
"""

# argparse 用于声明命令行参数，并自动生成 -h/--help。
import argparse

# asyncio 用于从同步 main() 入口运行异步 HTTP 评测。
import asyncio

# json 用于生成 UTF-8 JSON 报告和轨迹文件。
import json

# Sequence 表示只读序列接口。
#
# parse_args() 和 run_real_evaluation() 不需要修改调用方
# 传入的参数或场景集合。
from collections.abc import Sequence

# Path 和 PurePosixPath 提供跨平台路径处理。
#
# PurePosixPath 专门解析报告中使用正斜杠保存的项目相对路径，
# 不会访问文件系统。
from pathlib import Path, PurePosixPath

# httpx 提供异步 HTTP 客户端。
#
# 本脚本通过公开 Agent API 运行评测，而不是直接调用
# AgentDiagnosisService。
import httpx

# 当前 Planner Prompt 版本必须直接来自生产模块，
# 避免脚本和服务分别硬编码后产生版本漂移。
from app.agent.openai_planner import (
    AGENT_PLANNER_PROMPT_VERSION,
)

# Settings 是经过 Pydantic 校验的应用配置；
# get_settings() 从环境变量和 .env 读取配置。
from app.config import (
    Settings,
    get_settings,
)

# 这些 Schema 分别表示：
#
# MultimodalAgentBatchExecution：
#   整批评测完成后，尚未写入文件的报告和轨迹集合。
#
# MultimodalAgentEvaluationReport：
#   最终正式报告。
#
# MultimodalEvaluationFixRecord：
#   评测问题、修正内容和验证依据。
from app.schemas.multimodal_agent_evaluation import (
    MultimodalAgentBatchExecution,
    MultimodalAgentEvaluationReport,
    MultimodalAgentScenarioEvaluation,
    MultimodalEvaluationFixRecord,
)

# get_embedding_model() 取得经过非空校验的 Embedding 模型名。
#
# 当前脚本不会直接请求 Embedding 服务，
# 但报告必须记录服务端检索所使用的模型配置。
from app.services.embedding_client import (
    get_embedding_model,
)

# 加载器负责读取 JSONL；
# 批量执行器负责图片预检、真实 HTTP 请求、评分和汇总。
from app.services.evaluation import (
    evaluate_multimodal_agent_scenarios_via_api,
    load_multimodal_agent_evaluation_scenarios,
)

# get_llm_model() 取得 Planner 使用的文本模型名。
from app.services.llm_client import (
    create_llm_client,
    get_llm_model,
)

# 可选文本Judge只比较已经脱敏的视觉观察文字，
# 不读取图片，也不能改变正式确定性passed结果。
from app.services.visual_equivalence_judge import (
    OpenAICompatibleVisualEquivalenceJudge,
)

# get_vision_model() 取得 analyze_robot_image 使用的模型名。
from app.services.vision_client import (
    get_vision_model,
)

# VisionInputAdapter 在发送第一条 HTTP 请求前验证全部图片。
from app.services.vision_input import (
    VisionInputAdapter,
)

# Vision Prompt 版本直接引用生产常量。
from app.services.vision_prompts import (
    VISION_OBSERVATION_PROMPT_VERSION,
)


# __file__ 是当前脚本文件的位置。
#
# parents[1] 表示：
#
# scripts/evaluate_multimodal.py
# └─ scripts       parents[0]
#    └─ 项目根目录 parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# 正式评测默认读取的 30 条多模态场景。
DEFAULT_SCENARIOS_PATH = Path(
    "data/eval/multimodal_scenarios.jsonl"
)

# 被评测的公开多模态 Agent 接口。
DEFAULT_API_URL = (
    "http://127.0.0.1:8000"
    "/api/v1/agent/diagnose"
)

# 机器可读取的完整结构化报告。
DEFAULT_JSON_OUTPUT_PATH = Path(
    "docs/multimodal-evaluation.json"
)

# 供人工审核的 Markdown 报告。
DEFAULT_MARKDOWN_OUTPUT_PATH = Path(
    "docs/multimodal-evaluation.md"
)

# 一条场景可能包含多次 Planner、Vision、知识检索和遥测调用，
# 因此使用比普通 HTTP 请求更长的默认超时。
DEFAULT_TIMEOUT_SECONDS = 120.0

# 辅助Judge是独立模型请求，使用较短的单次超时；超时只会
# 让该辅助指标失败，不会丢弃已经完成的Agent正式评分。
DEFAULT_LLM_JUDGE_TIMEOUT_SECONDS = 30.0


# 这三条轨迹覆盖三种不同性质的场景：
#
# 001：正常图片 + 知识库联合诊断；
# 023：图片、日志和模拟遥测发生冲突；
# 030：高风险绕过急停请求，应该安全拒答。
TRACE_TARGET_ITEMS: tuple[
    tuple[str, str],
    ...,
] = (
    (
        "multimodal-agent-001",
        (
            "docs/multimodal-traces/"
            "multimodal-agent-001.json"
        ),
    ),
    (
        "multimodal-agent-023",
        (
            "docs/multimodal-traces/"
            "multimodal-agent-023.json"
        ),
    ),
    (
        "multimodal-agent-030",
        (
            "docs/multimodal-traces/"
            "multimodal-agent-030.json"
        ),
    ),
)


def resolve_project_path(
    path: Path,
) -> Path:
    """把相对路径锚定到当前项目根目录。

    相对路径不能依赖用户当前终端所在目录，
    否则从其他目录执行模块时可能读取或写入错误位置。
    """

    if not isinstance(path, Path):
        raise TypeError(
            "path必须是pathlib.Path"
        )

    if path.is_absolute():
        return path.resolve()

    return (
        PROJECT_ROOT
        / path
    ).resolve()


def build_trace_targets() -> dict[str, str]:
    """返回正式评测需要收集的三条轨迹映射。

    返回新 dict 而不是共享全局可变字典，
    防止调用方修改一次后影响后续运行。

    dict 会保留插入顺序，因此报告和文件写入顺序稳定为：

    001 → 023 → 030。
    """

    return dict(TRACE_TARGET_ITEMS)


def build_fix_history(
) -> tuple[
    MultimodalEvaluationFixRecord,
    ...,
]:
    """构造需要写入正式报告的修正记录。

    修正记录不是运行日志，而是说明评测基础设施中
    已经发现并处理了哪些可靠性问题。
    """

    return (
        MultimodalEvaluationFixRecord(
            issue=(
                "完整评测轨迹如果直接保存原始请求，"
                "可能包含图片Base64、本地路径或图片哈希"
            ),
            change=(
                "新增独立的脱敏轨迹契约，轨迹输入只保存"
                "机器人编号、脱敏现象、脱敏日志、任务目标、"
                "图片数量和公开分析目标"
            ),
            verification=(
                "自动化测试确认三份轨迹不包含"
                "image_base64、image_path和请求级"
                "image_sha256字段；公开Vision观察只保留"
                "source_image_sha256来源完整性指纹"
            ),
        ),
        MultimodalEvaluationFixRecord(
            issue=(
                "批量评测如果在逐条请求时才检查图片，"
                "可能在消耗部分真实API请求后才发现坏文件"
            ),
            change=(
                "批量执行器在发出第一条HTTP请求之前，"
                "预检全部场景的图片路径、SHA-256、格式、"
                "尺寸、像素数和字节数"
            ),
            verification=(
                "离线批量测试确认任意图片预检失败时，"
                "Agent API请求次数保持为零"
            ),
        ),
        MultimodalEvaluationFixRecord(
            issue=(
                "单条HTTP或响应契约错误不应导致"
                "剩余多模态场景完全丢失"
            ),
            change=(
                "单场景执行器把超时、连接失败、非2xx、"
                "非法JSON和响应Schema错误转换成脱敏失败评分，"
                "批量执行器继续执行后续场景"
            ),
            verification=(
                "Mock HTTP测试确认普通失败场景会进入正式结果，"
                "且30条场景仍保持固定顺序"
            ),
        ),
        MultimodalEvaluationFixRecord(
            issue=(
                "Vision最终失败如果统一记录成"
                "tool_execution_error，无法区分超时、"
                "上游不可用和模型响应无效"
            ),
            change=(
                "扩展工具错误码契约，并由ToolExecutor"
                "把三类Vision异常转换成独立的脱敏错误码"
            ),
            verification=(
                "离线工具链测试确认异常正文不会进入公开响应，"
                "但轨迹保留vision_timeout、"
                "vision_upstream_error或"
                "invalid_vision_response"
            ),
        ),
        MultimodalEvaluationFixRecord(
            issue=(
                "只保存三条代表轨迹无法定位批量评测中"
                "其余失败场景的工具状态和错误码"
            ),
            change=(
                "保留三条代表轨迹，并追加所有响应契约有效、"
                "但确定性评分未通过的完整脱敏轨迹"
            ),
            verification=(
                "离线30场景批量测试确认失败轨迹写入"
                "failures子目录，并与报告中的同场景评分一致"
            ),
        ),
        MultimodalEvaluationFixRecord(
            issue=(
                "按任务文字关键词缩小工具范围，以及在两次"
                "成功检索后隐藏知识检索，会把语言表述和固定"
                "次数误当成任务是否已经完成的可靠依据"
            ),
            change=(
                "请求级范围只按是否存在图片排除不可调用的"
                "Vision工具，其余注册只读工具保持可见；"
                "Runner不再按成功检索次数隐藏知识检索"
            ),
            verification=(
                "离线测试确认不同task_goal得到相同结构性范围，"
                "且第三次成功检索前后search_knowledge仍然可见"
            ),
        ),
        MultimodalEvaluationFixRecord(
            issue=(
                "Agent API不提供模型Token使用量时，空成本字段"
                "容易被误读成零成本"
            ),
            change=(
                "Markdown报告和终端摘要明确说明成本统计的"
                "数据来源、覆盖率以及无法估算的含义"
            ),
            verification=(
                "展示层测试确认报告说明无法估算不等于零成本，"
                "并且不使用延迟或工具步数伪造金额"
            ),
        ),
    )


def _positive_float(
    value: str,
) -> float:
    """把命令行文本转换成严格大于零的浮点数。"""

    try:
        parsed_value = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "必须是数字"
        ) from exc

    if parsed_value <= 0:
        raise argparse.ArgumentTypeError(
            "必须大于0"
        )

    return parsed_value


def _markdown_cell(
    value: object,
) -> str:
    """把公开字段转换成安全的 Markdown 表格单元格。"""

    if value is None:
        return "-"

    text = str(value)

    # 统一各种换行，避免一个字段破坏表格行。
    text = "<br>".join(
        text.splitlines()
    )

    # Markdown 表格使用竖线分隔列。
    return text.replace("|", "\\|")


def _format_ratio(
    value: float,
) -> str:
    """把确定存在的比率格式化成三位小数。"""

    return f"{value:.3f}"


def _format_optional_ratio(
    value: float | None,
) -> str:
    """格式化可能没有分母的比率。"""

    if value is None:
        return "不适用"

    return _format_ratio(value)


def _format_optional_bool(
    value: bool | None,
) -> str:
    """把三态布尔值转换成人工可读文本。"""

    if value is None:
        return "不适用"

    return "是" if value else "否"


def _format_cost(
    value: float | None,
) -> str:
    """格式化美元成本；没有可靠估算时明确标注。"""

    if value is None:
        return "无法估算"

    return f"${value:.6f}"


def _result_citation_correctness(
    result: MultimodalAgentScenarioEvaluation,
) -> float | None:
    """计算单场景引用正确率。

    分子是 correct=True 的实际引用数；
    分母是 Agent 返回的全部引用数。
    """

    returned_count = len(
        result.citation_evaluations
    )

    if returned_count == 0:
        return None

    correct_count = sum(
        citation.correct
        for citation in result.citation_evaluations
    )

    return correct_count / returned_count


def _result_citation_coverage(
    result: MultimodalAgentScenarioEvaluation,
) -> float | None:
    """计算单场景 Gold 证据覆盖率。"""

    expected_count = len(
        result.expected_evidence
    )

    if expected_count == 0:
        return None

    return (
        len(result.matched_expected_evidence)
        / expected_count
    )


def _result_safety_pass_rate(
    result: MultimodalAgentScenarioEvaluation,
) -> float:
    """计算单场景逐项安全要求通过率。"""

    safety_count = len(
        result.safety_evaluations
    )

    # 数据契约要求至少有一条安全检查，
    # 这里仍保留防御性处理。
    if safety_count == 0:
        return 0.0

    passed_count = sum(
        evaluation.passed
        for evaluation
        in result.safety_evaluations
    )

    return passed_count / safety_count


def render_markdown_report(
    report: MultimodalAgentEvaluationReport,
) -> str:
    """把正式结构化报告渲染成人工审核 Markdown。

    本函数不重新评分，只展示已经通过 Report Schema
    交叉校验的指标和逐场景结果。
    """

    if not isinstance(
        report,
        MultimodalAgentEvaluationReport,
    ):
        raise TypeError(
            "report必须是"
            "MultimodalAgentEvaluationReport"
        )

    metrics = report.metrics

    if metrics.cost_estimation_coverage == 0:
        # 当前Agent API公开响应没有Planner和Vision调用的
        # Token usage，因此本批次没有任何请求能可靠换算金额。
        cost_coverage_explanation = (
            "本批次没有请求取得可用于计价的模型 Token 使用量。"
        )
    elif metrics.cost_estimation_coverage < 1:
        # 部分覆盖时，已知金额只能代表取得usage的请求子集，
        # 不能外推成整批真实成本。
        cost_coverage_explanation = (
            "本批次只有部分请求取得模型 Token 使用量，已知成本"
            "只覆盖该请求子集，不能代表全部场景。"
        )
    else:
        cost_coverage_explanation = (
            "本批次所有请求都取得了可用于计价的模型 Token 使用量。"
        )

    lines = [
        "# 多模态 Agent 可靠性评测",
        "",
        "## 1. 评测范围",
        "",
        f"- 评测版本：`{report.evaluation_version}`",
        f"- 场景来源：`{report.scenario_source}`",
        f"- Agent API：`{report.api_url}`",
        f"- Planner 模型：`{report.planner_model}`",
        (
            "- Planner Prompt："
            f"`{report.planner_prompt_version}`"
        ),
        f"- Embedding 模型：`{report.embedding_model}`",
        f"- Chroma Collection：`{report.collection_name}`",
        f"- Vision 模型：`{report.vision_model}`",
        (
            "- Vision Prompt："
            f"`{report.vision_prompt_version}`"
        ),
        (
            "- LLM-as-judge："
            + (
                "仅作为辅助指标"
                if report.llm_judge_role == "auxiliary"
                else "未启用"
            )
        ),
        "",
        "## 2. 核心指标",
        "",
        "| 指标 | 分子/计数 | 分母/总量 | 结果 |",
        "|---|---:|---:|---:|",
        (
            "| 请求成功率 | "
            f"{metrics.request_success_count} | "
            f"{metrics.scenario_count} | "
            f"{_format_ratio(metrics.request_success_rate)} |"
        ),
        (
            "| 场景总通过率 | "
            f"{metrics.passed_scenario_count} | "
            f"{metrics.scenario_count} | "
            f"{_format_ratio(metrics.scenario_pass_rate)} |"
        ),
        (
            "| 图片观察字段准确性 | "
            f"{metrics.matched_visual_observation_count} | "
            f"{metrics.visual_observation_field_count} | "
            + _format_optional_ratio(
                metrics.image_observation_field_accuracy
            )
            + " |"
        ),
        (
            "| 工具选择正确率 | "
            f"{metrics.tool_selection_correct_count} | "
            f"{metrics.scenario_count} | "
            + _format_ratio(
                metrics.tool_selection_accuracy
            )
            + " |"
        ),
        (
            "| Vision 工具选择正确率 | "
            f"{metrics.vision_tool_selection_correct_count} | "
            f"{metrics.scenario_count} | "
            + _format_ratio(
                metrics.vision_tool_selection_accuracy
            )
            + " |"
        ),
        (
            "| 任务完成率 | "
            f"{metrics.completed_task_count} | "
            f"{metrics.expected_completion_count} | "
            f"{_format_ratio(metrics.task_completion_rate)} |"
        ),
        (
            "| 引用正确率 | "
            f"{metrics.correct_citation_count} | "
            f"{metrics.returned_citation_count} | "
            f"{_format_ratio(metrics.citation_correctness)} |"
        ),
        (
            "| 引用覆盖率 | "
            f"{metrics.matched_expected_evidence_count} | "
            f"{metrics.expected_evidence_count} | "
            f"{_format_ratio(metrics.citation_coverage)} |"
        ),
        (
            "| 信息来源标注正确率 | "
            f"{metrics.source_label_correct_count} | "
            f"{metrics.scenario_count} | "
            f"{_format_ratio(metrics.source_label_accuracy)} |"
        ),
        (
            "| 安全要求通过率 | "
            f"{metrics.passed_safety_requirement_count} | "
            f"{metrics.safety_requirement_count} | "
            + _format_ratio(
                metrics.safety_requirement_pass_rate
            )
            + " |"
        ),
        (
            "| 安全拒答率 | "
            f"{metrics.correct_safe_refusal_count} | "
            f"{metrics.safe_refusal_case_count} | "
            f"{_format_ratio(metrics.safe_refusal_rate)} |"
        ),
        "",
        "## 3. 效率与成本",
        "",
        "| 指标 | 结果 |",
        "|---|---:|",
        (
            "| 平均工具步骤数 | "
            f"{metrics.average_steps:.3f} |"
        ),
        (
            "| 平均端到端延迟 | "
            f"{metrics.mean_latency_ms:.3f} ms |"
        ),
        (
            "| P50 端到端延迟 | "
            f"{metrics.p50_latency_ms:.3f} ms |"
        ),
        (
            "| P95 端到端延迟 | "
            f"{metrics.p95_latency_ms:.3f} ms |"
        ),
        (
            "| 成本估算覆盖率 | "
            + _format_ratio(
                metrics.cost_estimation_coverage
            )
            + " |"
        ),
        (
            "| 已知总估算成本 | "
            + _format_cost(
                metrics.known_total_estimated_cost_usd
            )
            + " |"
        ),
        (
            "| 可估算请求平均成本 | "
            + _format_cost(
                metrics.mean_estimated_cost_usd_per_request
            )
            + " |"
        ),
        (
            "| 辅助 Judge 评测数 | "
            f"{metrics.llm_judge_evaluated_count} |"
        ),
        "",
        "### 3.1 成本统计限制",
        "",
        cost_coverage_explanation,
        "",
        (
            "- 报告中的成本只能依据服务端实际采集的 Planner、"
            "Vision 和辅助 Judge 模型输入/输出 Token 使用量，"
            "再结合对应模型单价进行估算。"
        ),
        (
            "- `无法估算`、`null` 或空值表示缺少可靠计量数据，"
            "不表示调用成本为 0。"
        ),
        (
            "- 请求延迟、工具调用次数和返回文字长度都不能替代"
            "模型 Token usage，本报告不会据此推测或伪造金额。"
        ),
        (
            "- 当前公开 Agent API 响应不暴露 Token 明细；若后续"
            "需要数值成本，应在服务端内部采集并汇总 usage，"
            "同时继续避免在公开响应中暴露密钥或敏感调用内容。"
        ),
        "",
        "## 4. 分类结果",
        "",
        "| 场景类别 | 场景数 | 通过数 | 通过率 |",
        "|---|---:|---:|---:|",
    ]

    for summary in report.category_summaries:
        lines.append(
            "| "
            + _markdown_cell(summary.category)
            + " | "
            + str(summary.scenario_count)
            + " | "
            + str(summary.passed_scenario_count)
            + " | "
            + _format_ratio(
                summary.scenario_pass_rate
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## 5. 失败原因",
            "",
            "| 脱敏失败原因 | 场景数 |",
            "|---|---:|",
        ]
    )

    if report.failure_reason_counts:
        for failure in report.failure_reason_counts:
            lines.append(
                "| "
                + _markdown_cell(failure.reason)
                + " | "
                + str(failure.scenario_count)
                + " |"
            )
    else:
        lines.append(
            "| 本次评测没有记录失败原因 | 0 |"
        )

    lines.extend(
        [
            "",
            "## 6. 完整脱敏轨迹",
            "",
        ]
    )

    for trace_path in report.trace_paths:
        lines.append(
            f"- `{trace_path}`"
        )

    lines.extend(
        [
            "",
            "这些轨迹只保存公开 Agent 响应、确定性评分和"
            "脱敏输入摘要，不保存图片正文、本地图片路径、"
            "请求级图片元数据、Authorization、API Key 或模型"
            "私有思维链。公开 Vision 观察可能保留"
            "`source_image_sha256`作为来源完整性指纹；该字段"
            "不能还原图片，也不包含图片正文或本地路径。",
            "",
            "## 7. 修正记录",
            "",
            "| 发现的问题 | 实施的修正 | 验证依据 |",
            "|---|---|---|",
        ]
    )

    for fix in report.fix_history:
        lines.append(
            "| "
            + _markdown_cell(fix.issue)
            + " | "
            + _markdown_cell(fix.change)
            + " | "
            + _markdown_cell(fix.verification)
            + " |"
        )

    lines.extend(
        [
            "",
            "## 8. 逐场景结果",
            "",
            (
                "| 场景 | 类别 | 请求成功 | Agent状态 | "
                "工具顺序 | Vision调用 | 视觉准确率 | "
                "工具正确 | 任务完成正确 | 引用正确率 | "
                "引用覆盖率 | 来源正确 | 安全通过率 | "
                "安全拒答正确 | 步骤 | 延迟ms | 通过 | 失败原因 |"
            ),
            (
                "|---|---|---|---|---|---:|---:|---|---|"
                "---:|---:|---|---:|---|---:|---:|---|---|"
            ),
        ]
    )

    for result in report.results:
        tool_sequence = (
            " → ".join(
                result.actual_tool_sequence
            )
            if result.actual_tool_sequence
            else "无"
        )

        failure_text = (
            "；".join(result.failure_reasons)
            if result.failure_reasons
            else "无"
        )

        lines.append(
            "| "
            + _markdown_cell(result.scenario_id)
            + " | "
            + _markdown_cell(result.category)
            + " | "
            + _format_optional_bool(
                result.request_succeeded
            )
            + " | "
            + _markdown_cell(
                result.actual_diagnosis_status
            )
            + " | "
            + _markdown_cell(tool_sequence)
            + " | "
            + str(result.actual_vision_call_count)
            + " | "
            + _format_optional_ratio(
                result.visual_observation_accuracy
            )
            + " | "
            + _format_optional_bool(
                result.tool_selection_correct
            )
            + " | "
            + _format_optional_bool(
                result.task_completion_correct
            )
            + " | "
            + _format_optional_ratio(
                _result_citation_correctness(
                    result
                )
            )
            + " | "
            + _format_optional_ratio(
                _result_citation_coverage(
                    result
                )
            )
            + " | "
            + _format_optional_bool(
                result.source_labels_correct
            )
            + " | "
            + _format_ratio(
                _result_safety_pass_rate(
                    result
                )
            )
            + " | "
            + _format_optional_bool(
                result.safe_refusal_correct
            )
            + " | "
            + str(result.step_count)
            + " | "
            + f"{result.latency_ms:.3f}"
            + " | "
            + _format_optional_bool(result.passed)
            + " | "
            + _markdown_cell(failure_text)
            + " |"
        )

    lines.extend(
        [
            "",
            "## 9. 指标解释与边界",
            "",
            (
                "- 图片观察字段准确性使用 Gold 观察项与"
                "公开 VisionObservation 的确定性匹配结果计算。"
            ),
            (
                "- 工具选择正确率同时检查 required_tools、"
                "allowed_tools 和 forbidden_tools。"
            ),
            (
                "- 引用正确率衡量已返回引用是否正确；"
                "引用覆盖率衡量 Gold 证据是否被完整覆盖。"
            ),
            (
                "- 安全拒答率只以明确标记为安全拒答场景的"
                "案例为分母。"
            ),
            (
                "- LLM-as-judge 当前未启用；即使以后启用，"
                "也只能作为辅助观察，不能改变确定性 passed 结果。"
            ),
            (
                "- 成本只有在公开响应提供可靠 usage 和价格依据时"
                "才进行估算；无法估算不会被错误记为零成本。"
            ),
            "",
        ]
    )

    return "\n".join(lines)


def _resolve_docs_output(
    *,
    project_root: Path,
    path: Path,
    expected_suffix: str,
) -> Path:
    """校验一个输出文件确实位于当前项目 docs 目录。"""

    if not isinstance(path, Path):
        raise TypeError(
            "输出路径必须是pathlib.Path"
        )

    candidate = (
        path
        if path.is_absolute()
        else project_root / path
    )

    resolved_path = candidate.resolve()
    resolved_docs = (
        project_root / "docs"
    ).resolve()

    if not resolved_path.is_relative_to(
        resolved_docs
    ):
        raise ValueError(
            "评测输出必须位于项目docs目录"
        )

    if (
        resolved_path.suffix.lower()
        != expected_suffix
    ):
        raise ValueError(
            "评测输出扩展名必须是"
            f"{expected_suffix}"
        )

    return resolved_path


def _write_text_artifacts(
    artifacts: Sequence[
        tuple[Path, str]
    ],
) -> None:
    """先写临时文件，再把全部临时文件替换为正式文件。

    这样可以避免 JSON 只写到一半时就留下看似完整的报告。
    """

    staged_files: list[
        tuple[Path, Path]
    ] = []

    try:
        for target_path, text in artifacts:
            target_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            temporary_path = (
                target_path.with_name(
                    target_path.name + ".tmp"
                )
            )

            temporary_path.write_text(
                text,
                encoding="utf-8",
            )

            staged_files.append(
                (
                    temporary_path,
                    target_path,
                )
            )

        # Path.replace() 在目标文件已经存在时替换它。
        #
        # 同一文件系统内的单文件替换通常是原子的，
        # 可以避免正式文件处于半写入状态。
        for temporary_path, target_path in (
            staged_files
        ):
            temporary_path.replace(
                target_path
            )
    finally:
        # 如果中间步骤失败，只清理由本函数创建的 .tmp 文件。
        for temporary_path, _ in staged_files:
            if temporary_path.exists():
                temporary_path.unlink()


def write_evaluation_artifacts(
    *,
    execution: MultimodalAgentBatchExecution,
    project_root: Path,
    json_output_path: Path,
    markdown_output_path: Path,
) -> None:
    """写入正式 JSON、Markdown 和完整脱敏轨迹。

    BatchExecution 已经确认：

    1. 报告与轨迹路径一致；
    2. 轨迹场景都存在于报告；
    3. 轨迹评分与报告中的同场景评分相同；
    4. 至少包含三份轨迹。
    """

    if not isinstance(
        execution,
        MultimodalAgentBatchExecution,
    ):
        raise TypeError(
            "execution必须是"
            "MultimodalAgentBatchExecution"
        )

    if not isinstance(project_root, Path):
        raise TypeError(
            "project_root必须是pathlib.Path"
        )

    resolved_root = project_root.resolve(
        strict=True
    )

    if not resolved_root.is_dir():
        raise NotADirectoryError(
            "project_root必须是目录"
        )

    resolved_json = _resolve_docs_output(
        project_root=resolved_root,
        path=json_output_path,
        expected_suffix=".json",
    )

    resolved_markdown = _resolve_docs_output(
        project_root=resolved_root,
        path=markdown_output_path,
        expected_suffix=".md",
    )

    if resolved_json == resolved_markdown:
        raise ValueError(
            "JSON和Markdown不能写入同一路径"
        )

    report_json = json.dumps(
        execution.report.model_dump(
            mode="json"
        ),
        ensure_ascii=False,
        indent=2,
    ) + "\n"

    markdown = render_markdown_report(
        execution.report
    )

    pending_artifacts: list[
        tuple[Path, str]
    ] = [
        (
            resolved_json,
            report_json,
        ),
        (
            resolved_markdown,
            markdown,
        ),
    ]

    used_paths = {
        resolved_json,
        resolved_markdown,
    }

    for artifact in (
        execution.trace_artifacts
    ):
        relative_trace_path = PurePosixPath(
            artifact.path
        )

        trace_path = (
            resolved_root.joinpath(
                *relative_trace_path.parts
            )
        ).resolve()

        resolved_trace = _resolve_docs_output(
            project_root=resolved_root,
            path=trace_path,
            expected_suffix=".json",
        )

        if resolved_trace in used_paths:
            raise ValueError(
                "报告和轨迹不能写入重复路径"
            )

        used_paths.add(resolved_trace)

        trace_json = json.dumps(
            artifact.trace.model_dump(
                mode="json"
            ),
            ensure_ascii=False,
            indent=2,
        ) + "\n"

        pending_artifacts.append(
            (
                resolved_trace,
                trace_json,
            )
        )

    _write_text_artifacts(
        tuple(pending_artifacts)
    )


def _scenario_source_text(
    scenarios_path: Path,
) -> str:
    """生成写入报告的稳定场景来源文本。"""

    resolved_path = scenarios_path.resolve()

    if resolved_path.is_relative_to(
        PROJECT_ROOT
    ):
        return (
            resolved_path
            .relative_to(PROJECT_ROOT)
            .as_posix()
        )

    # 显式使用项目外案例时保留绝对位置，
    # 避免报告谎称使用了默认数据。
    return str(resolved_path)


async def run_real_evaluation(
    *,
    scenarios: Sequence,
    settings: Settings,
    api_url: str,
    timeout_seconds: float,
    scenario_source: str = (
        DEFAULT_SCENARIOS_PATH.as_posix()
    ),
    enable_llm_judge: bool = False,
    llm_judge_timeout_seconds: float = (
        DEFAULT_LLM_JUDGE_TIMEOUT_SECONDS
    ),
) -> MultimodalAgentBatchExecution:
    """装配依赖并通过真实 Agent API 执行整批评测。

    当前函数只装配客户端和元数据。
    图片验证、HTTP 请求、评分与报告构造仍由服务层完成。
    """

    if (
        isinstance(
            scenarios,
            (str, bytes, bytearray),
        )
        or not isinstance(
            scenarios,
            Sequence,
        )
    ):
        raise TypeError(
            "scenarios必须是多模态场景序列"
        )

    if not isinstance(settings, Settings):
        raise TypeError(
            "settings必须是Settings"
        )

    if (
        not isinstance(api_url, str)
        or not api_url.strip()
    ):
        raise ValueError(
            "api_url不能为空"
        )

    if (
        not isinstance(
            timeout_seconds,
            (int, float),
        )
        or isinstance(timeout_seconds, bool)
        or timeout_seconds <= 0
    ):
        raise ValueError(
            "timeout_seconds必须大于0"
        )

    if (
        not isinstance(
            scenario_source,
            str,
        )
        or not scenario_source.strip()
    ):
        raise ValueError(
            "scenario_source不能为空"
        )

    if not isinstance(enable_llm_judge, bool):
        raise TypeError(
            "enable_llm_judge必须是bool"
        )

    if (
        isinstance(
            llm_judge_timeout_seconds,
            bool,
        )
        or not isinstance(
            llm_judge_timeout_seconds,
            (int, float),
        )
    ):
        raise TypeError(
            "llm_judge_timeout_seconds必须是数字"
        )

    if llm_judge_timeout_seconds <= 0:
        raise ValueError(
            "llm_judge_timeout_seconds必须大于0"
        )

    # 使用与服务端相同的业务上限。
    #
    # 批量执行器会在第一条 HTTP 请求前，
    # 使用该 Adapter 检查全部本地图片。
    vision_input_adapter = VisionInputAdapter(
        max_image_size_bytes=(
            settings.max_vision_image_size_bytes
        ),
        max_image_dimension_px=(
            settings.max_vision_image_dimension_px
        ),
        max_image_pixels=(
            settings.max_vision_image_pixels
        ),
    )

    # 这些 resolver 只取得并校验模型名称，
    # 不会在当前脚本中创建上游模型客户端。
    planner_model = get_llm_model(settings)
    embedding_model = get_embedding_model(
        settings
    )
    vision_model = get_vision_model(settings)

    llm_judge_client = None
    visual_equivalence_judge = None

    try:
        if enable_llm_judge:
            # Judge复用已经校验的文本LLM配置，但使用独立客户端，
            # 避免与被测API服务进程共享连接或隐藏全局状态。
            llm_judge_client = create_llm_client(
                settings
            )
            visual_equivalence_judge = (
                OpenAICompatibleVisualEquivalenceJudge(
                    client=llm_judge_client,
                    model=planner_model,
                    timeout_seconds=float(
                        llm_judge_timeout_seconds
                    ),
                )
            )

        # httpx.Timeout把Agent API总请求超时显式交给
        # AsyncClient。async with会在成功或异常时关闭连接池。
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(
                float(timeout_seconds)
            )
        ) as client:
            return await (
                evaluate_multimodal_agent_scenarios_via_api(
                    scenarios=scenarios,
                    client=client,
                    api_url=api_url.strip(),
                    project_root=PROJECT_ROOT,
                    vision_input_adapter=(
                        vision_input_adapter
                    ),
                    scenario_source=(
                        scenario_source.strip()
                    ),
                    planner_model=planner_model,
                    planner_prompt_version=(
                        AGENT_PLANNER_PROMPT_VERSION
                    ),
                    embedding_model=(
                        embedding_model
                    ),
                    collection_name=(
                        settings.chroma_collection_name
                    ),
                    vision_model=vision_model,
                    vision_prompt_version=(
                        VISION_OBSERVATION_PROMPT_VERSION
                    ),
                    trace_targets=(
                        build_trace_targets()
                    ),
                    fix_history=(
                        build_fix_history()
                    ),
                    visual_equivalence_judge=(
                        visual_equivalence_judge
                    ),
                )
            )
    finally:
        if llm_judge_client is not None:
            # AsyncOpenAI.close()异步关闭HTTP连接池。
            # 即使Agent批量评测抛出异常也必须释放Judge连接。
            await llm_judge_client.close()


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """解析正式评测的命令行参数。"""

    parser = argparse.ArgumentParser(
        description=(
            "通过真实Agent HTTP API运行30条"
            "多模态可靠性场景并生成正式报告"
        )
    )

    parser.add_argument(
        "--scenarios",
        type=Path,
        default=DEFAULT_SCENARIOS_PATH,
        help="多模态评测场景JSONL文件",
    )

    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help="被评测的Agent诊断API地址",
    )

    parser.add_argument(
        "--json-output",
        type=Path,
        default=DEFAULT_JSON_OUTPUT_PATH,
        help="正式JSON报告输出路径",
    )

    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=DEFAULT_MARKDOWN_OUTPUT_PATH,
        help="人工审核Markdown报告输出路径",
    )

    parser.add_argument(
        "--timeout-seconds",
        type=_positive_float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="每条Agent HTTP请求的超时秒数",
    )

    parser.add_argument(
        "--enable-llm-judge",
        action="store_true",
        help=(
            "对确定性规则未匹配的视觉字段运行"
            "辅助LLM语义Judge"
        ),
    )

    parser.add_argument(
        "--llm-judge-timeout-seconds",
        type=_positive_float,
        default=(
            DEFAULT_LLM_JUDGE_TIMEOUT_SECONDS
        ),
        help="单次辅助LLM Judge请求超时秒数",
    )

    return parser.parse_args(argv)


def print_summary(
    *,
    execution: MultimodalAgentBatchExecution,
    json_output_path: Path,
    markdown_output_path: Path,
) -> None:
    """输出简短但覆盖任务六指标的终端摘要。"""

    if not isinstance(
        execution,
        MultimodalAgentBatchExecution,
    ):
        raise TypeError(
            "execution必须是"
            "MultimodalAgentBatchExecution"
        )

    metrics = execution.report.metrics

    print("多模态 Agent 评测完成")
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
        "图片观察字段准确性："
        + _format_optional_ratio(
            metrics.image_observation_field_accuracy
        )
    )
    print(
        "工具选择正确率："
        f"{metrics.tool_selection_accuracy:.3f}"
    )
    print(
        "Vision工具选择正确率："
        f"{metrics.vision_tool_selection_accuracy:.3f}"
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
        "来源标注正确率："
        f"{metrics.source_label_accuracy:.3f}"
    )
    print(
        "安全拒答率："
        f"{metrics.safe_refusal_rate:.3f}"
    )
    print(
        "平均工具步骤数："
        f"{metrics.average_steps:.3f}"
    )
    print(
        "P50延迟："
        f"{metrics.p50_latency_ms:.1f}ms"
    )
    print(
        "P95延迟："
        f"{metrics.p95_latency_ms:.1f}ms"
    )
    print(
        "单请求平均估算成本："
        + _format_cost(
            metrics.mean_estimated_cost_usd_per_request
        )
    )
    if (
        metrics.mean_estimated_cost_usd_per_request
        is None
    ):
        # 这里直接解释“无法估算”的含义，避免终端用户把空成本
        # 当成零成本。真实金额必须来自服务端模型usage计量。
        print(
            "成本统计限制：Agent API未提供模型Token使用量，"
            "无法可靠估算；无法估算不等于零成本"
        )
    print(
        "完整脱敏轨迹数："
        f"{len(execution.trace_artifacts)}"
    )
    print(
        f"JSON报告：{json_output_path}"
    )
    print(
        f"Markdown报告：{markdown_output_path}"
    )


def main() -> None:
    """运行完整命令行评测流程。"""

    args = parse_args()

    scenarios_path = resolve_project_path(
        args.scenarios
    )
    json_output_path = resolve_project_path(
        args.json_output
    )
    markdown_output_path = resolve_project_path(
        args.markdown_output
    )

    # 加载器逐行进行 JSON、Schema 和重复编号检查。
    scenarios = tuple(
        load_multimodal_agent_evaluation_scenarios(
            scenarios_path
        )
    )

    settings = get_settings()

    # asyncio.run() 创建事件循环，运行异步评测，
    # 并在结束后关闭该事件循环。
    execution = asyncio.run(
        run_real_evaluation(
            scenarios=scenarios,
            settings=settings,
            api_url=args.api_url,
            timeout_seconds=(
                args.timeout_seconds
            ),
            scenario_source=(
                _scenario_source_text(
                    scenarios_path
                )
            ),
            enable_llm_judge=(
                args.enable_llm_judge
            ),
            llm_judge_timeout_seconds=(
                args.llm_judge_timeout_seconds
            ),
        )
    )

    # 只有整批评测成功构造出合法 BatchExecution 后，
    # 才开始写报告和轨迹。
    write_evaluation_artifacts(
        execution=execution,
        project_root=PROJECT_ROOT,
        json_output_path=json_output_path,
        markdown_output_path=(
            markdown_output_path
        ),
    )

    print_summary(
        execution=execution,
        json_output_path=json_output_path,
        markdown_output_path=(
            markdown_output_path
        ),
    )


if __name__ == "__main__":
    main()
