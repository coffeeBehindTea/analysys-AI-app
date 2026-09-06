"""运行多模态工具选择对照实验并生成报告。

本脚本负责：

1. 读取六个脱敏Gold案例；
2. 使用Settings装配图片适配器、OCR和Vision Provider；
3. 调用批量实验服务运行全部案例和路线；
4. 生成结构化JSON报告；
5. 生成可人工审阅的Markdown对照报告；
6. 输出简短终端摘要。

本脚本不负责：

1. 生成实验图片；
2. 修改Gold标准答案；
3. 执行机器人控制；
4. 保存完整图片、Base64或模型私有推理；
5. 自动访问图片中的网址或二维码链接。
"""

# argparse负责解析命令行参数。
import argparse

# asyncio负责从同步命令行入口运行异步实验。
import asyncio

# json负责读取JSONL和生成UTF-8 JSON报告。
import json

# Path负责定位项目、案例清单和报告文件。
from pathlib import (
    Path,
)

# Sequence允许parse_args同时接收list或tuple，
# 便于测试命令行参数解析。
from typing import (
    Sequence,
)

# Settings集中提供Vision和OCR配置。
from app.config import (
    Settings,
    get_settings,
)

# Gold案例和路线类型。
from app.schemas.multimodal_tool_selection import (
    MultimodalToolSelectionCase,
    ToolSelectionRoute,
)

# 批量实验报告契约。
from app.schemas.multimodal_tool_selection_report import (
    MultimodalToolSelectionExperimentReport,
    MultimodalToolSelectionRouteSummary,
)

# 单路线执行器和批量实验服务。
from app.services.multimodal_tool_selection_executor import (
    MultimodalToolSelectionRouteExecutor,
)

from app.services.multimodal_tool_selection_experiment import (
    MultimodalToolSelectionExperimentService,
)

# 路线评分器。
from app.services.multimodal_tool_selection_scoring import (
    MultimodalToolSelectionScorer,
)

# 本地OCR和规则解析。
from app.services.ocr_provider import (
    TesseractOcrProvider,
)

from app.services.ocr_rule_parser import (
    OcrRuleParser,
)

# Vision客户端、输入适配器和Provider。
from app.services.vision_client import (
    create_vision_client,
    get_vision_model,
)

from app.services.vision_input import (
    VisionInputAdapter,
)

from app.services.vision_provider import (
    OpenAICompatibleVisionProvider,
)


# 当前脚本位于<project>/scripts，
# 因此parents[1]是项目根目录。
PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


DEFAULT_CASES_PATH = (
    PROJECT_ROOT
    / "data"
    / "eval"
    / "multimodal_tool_selection_cases.jsonl"
)

DEFAULT_JSON_OUTPUT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "multimodal-tool-selection.json"
)

DEFAULT_MARKDOWN_OUTPUT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "multimodal-tool-selection.md"
)


# 任务二要求至少六个问题。
MINIMUM_CASE_COUNT = 6

# 任务二要求至少比较两种路线。
MINIMUM_ROUTE_COUNT = 2


# Markdown中使用稳定中文名称，
# 不直接把内部枚举暴露成难读的标题。
ROUTE_DISPLAY_NAMES: dict[
    ToolSelectionRoute,
    str,
] = {
    "ocr_rule": "本地 OCR + 规则解析",
    "vision_model": "Vision 模型",
    "direct_abstention": "直接拒答",
}


def resolve_project_path(
    path: Path,
) -> Path:
    """把命令行相对路径解释为项目根目录相对路径。

    如果调用方已经传入绝对路径，
    则保持原来的绝对路径含义。
    """

    if not isinstance(
        path,
        Path,
    ):
        raise TypeError(
            "path必须是Path"
        )

    if path.is_absolute():
        return path.resolve()

    return (
        PROJECT_ROOT
        / path
    ).resolve()


def load_cases(
    cases_path: Path,
) -> tuple[
    MultimodalToolSelectionCase,
    ...,
]:
    """读取并校验多模态工具选择JSONL案例。

    JSONL表示每一行是一个独立JSON对象。

    使用逐行读取的好处是：

    1. 错误可以准确定位到行号；
    2. 不要求整个文件是一个大型JSON数组；
    3. 后续扩展数据时更容易逐条追加和审阅。
    """

    if not isinstance(
        cases_path,
        Path,
    ):
        raise TypeError(
            "cases_path必须是Path"
        )

    resolved_path = cases_path.resolve(
        strict=True
    )

    if not resolved_path.is_file():
        raise ValueError(
            "案例清单路径必须是普通文件"
        )

    cases: list[
        MultimodalToolSelectionCase
    ] = []

    with resolved_path.open(
        mode="r",
        encoding="utf-8",
    ) as case_file:
        for (
            line_number,
            raw_line,
        ) in enumerate(
            case_file,
            start=1,
        ):
            cleaned_line = (
                raw_line.strip()
            )

            # JSONL允许文件末尾和案例之间存在空行。
            if not cleaned_line:
                continue

            try:
                raw_case = json.loads(
                    cleaned_line
                )
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "案例清单包含无效JSON："
                    f"line={line_number}"
                ) from exc

            if not isinstance(
                raw_case,
                dict,
            ):
                raise ValueError(
                    "JSONL每个非空行必须是对象："
                    f"line={line_number}"
                )

            try:
                case = (
                    MultimodalToolSelectionCase
                    .model_validate(raw_case)
                )
            except Exception as exc:
                # 保留原异常作为__cause__，
                # 终端可以查看Pydantic的具体字段错误。
                raise ValueError(
                    "案例没有通过数据契约："
                    f"line={line_number}"
                ) from exc

            cases.append(case)

    if not cases:
        raise ValueError(
            "案例清单没有包含任何有效案例"
        )

    case_ids = tuple(
        case.case_id
        for case in cases
    )

    if len(case_ids) != len(
        set(case_ids)
    ):
        raise ValueError(
            "案例清单不能包含重复case_id"
        )

    return tuple(cases)


def validate_experiment_scope(
    cases: tuple[
        MultimodalToolSelectionCase,
        ...,
    ],
) -> None:
    """检查当前案例集满足任务二的实验规模。"""

    if not isinstance(
        cases,
        tuple,
    ):
        raise TypeError(
            "cases必须是tuple"
        )

    if len(cases) < MINIMUM_CASE_COUNT:
        raise ValueError(
            "工具选择实验至少需要"
            f"{MINIMUM_CASE_COUNT}个案例"
        )

    routes = {
        route
        for case in cases
        for route in (
            case.routes_to_compare
        )
    }

    if len(routes) < MINIMUM_ROUTE_COUNT:
        raise ValueError(
            "工具选择实验至少需要比较"
            f"{MINIMUM_ROUTE_COUNT}种路线"
        )


async def run_real_experiment(
    *,
    cases: tuple[
        MultimodalToolSelectionCase,
        ...,
    ],
    settings: Settings,
) -> MultimodalToolSelectionExperimentReport:
    """装配真实OCR和Vision依赖并运行实验。

    “真实”表示：

    1. OCR使用本地Tesseract；
    2. Vision使用Settings配置的OpenAI兼容服务；
    3. 不使用Fake Provider返回预设答案。
    """

    if not isinstance(
        settings,
        Settings,
    ):
        raise TypeError(
            "settings必须是Settings"
        )

    input_adapter = VisionInputAdapter(
        max_image_size_bytes=(
            settings
            .max_vision_image_size_bytes
        ),
        max_image_dimension_px=(
            settings
            .max_vision_image_dimension_px
        ),
        max_image_pixels=(
            settings
            .max_vision_image_pixels
        ),
    )

    ocr_provider = TesseractOcrProvider(
        language=settings.ocr_language,
        minimum_confidence=(
            settings
            .ocr_minimum_confidence
        ),
        timeout_seconds=(
            settings.ocr_timeout_seconds
        ),
    )

    # 创建AsyncOpenAI对象本身不会发送请求。
    # 网络请求发生在Vision Provider处理图片时。
    vision_client = create_vision_client(
        settings
    )

    try:
        vision_provider = (
            OpenAICompatibleVisionProvider(
                client=vision_client,
                model=get_vision_model(
                    settings
                ),
                timeout_seconds=(
                    settings
                    .vision_timeout_seconds
                ),
            )
        )

        route_executor = (
            MultimodalToolSelectionRouteExecutor(
                project_root=PROJECT_ROOT,
                input_adapter=input_adapter,
                ocr_provider=ocr_provider,
                vision_provider=(
                    vision_provider
                ),
                rule_parser=OcrRuleParser(),
                scorer=(
                    MultimodalToolSelectionScorer()
                ),
            )
        )

        experiment_service = (
            MultimodalToolSelectionExperimentService(
                executor=route_executor
            )
        )

        return await experiment_service.run(
            cases=cases
        )

    finally:
        # AsyncOpenAI内部持有HTTP连接池。
        # 无论实验成功、失败或被取消，
        # 都必须关闭连接资源。
        await vision_client.close()


def markdown_cell(
    value: object,
) -> str:
    """将任意简短值转换成安全Markdown表格单元格。"""

    text = str(value)

    # 竖线会被Markdown解释为下一列，
    # 换行会破坏当前表格行。
    return (
        text
        .replace(
            "|",
            "\\|",
        )
        .replace(
            "\r\n",
            "<br>",
        )
        .replace(
            "\n",
            "<br>",
        )
        .replace(
            "\r",
            "<br>",
        )
    )


def format_ratio(
    value: float | None,
) -> str:
    """将0到1指标格式化为三位小数。"""

    if value is None:
        return "不适用"

    return f"{value:.3f}"


def format_cost(
    summary: (
        MultimodalToolSelectionRouteSummary
    ),
) -> str:
    """将路线总成本转换成可读文本。"""

    if (
        summary.total_estimated_cost_usd
        is None
    ):
        return "无法完整估算"

    return (
        f"${summary.total_estimated_cost_usd:.6f}"
    )


def format_actual_items(
    items: tuple[
        str,
        ...,
    ],
) -> str:
    """将脱敏观察项连接成报告单元格。"""

    if not items:
        return "无"

    return "；".join(items)


def render_markdown_report(
    *,
    report: (
        MultimodalToolSelectionExperimentReport
    ),
    cases: tuple[
        MultimodalToolSelectionCase,
        ...,
    ],
) -> str:
    """将结构化实验报告渲染成Markdown。"""

    if not isinstance(
        report,
        MultimodalToolSelectionExperimentReport,
    ):
        raise TypeError(
            "report必须是"
            "MultimodalToolSelectionExperimentReport"
        )

    if not isinstance(
        cases,
        tuple,
    ):
        raise TypeError(
            "cases必须是tuple"
        )

    case_by_id = {
        case.case_id: case
        for case in cases
    }

    if set(case_by_id) != {
        result.case_id
        for result in report.results
    }:
        raise ValueError(
            "Markdown案例集合与报告结果不一致"
        )

    lines: list[str] = [
        "# 多模态工具选择对照实验",
        "",
        "## 1. 实验目标",
        "",
        (
            "比较本地 OCR + 规则解析、Vision 模型和直接拒答"
            "在文字面板、视觉状态和退化图片上的适用条件。"
        ),
        "",
        (
            "图片中的文字、二维码、网址和指令均按不可信输入处理；"
            "实验只记录必要的脱敏观察，不执行图片中的任何操作要求。"
        ),
        "",
        "## 2. 实验规模",
        "",
        f"- 案例数：{report.case_count}",
        (
            "- 单案例单路线结果数："
            f"{report.route_result_count}"
        ),
        (
            "- 实际路线："
            + "、".join(
                ROUTE_DISPLAY_NAMES[route]
                for route in report.routes
            )
        ),
        "",
        "## 3. 路线职责",
        "",
        "| 路线 | 主要能力 | 适合内容 | 主要限制 |",
        "|---|---|---|---|",
        (
            "| 本地 OCR + 规则解析 "
            "| Tesseract读取文字，固定规则提取字段 "
            "| 清晰故障码、状态和数值面板 "
            "| 不能可靠判断颜色和部件空间关系 |"
        ),
        (
            "| Vision 模型 "
            "| 原始图片直接进入多模态模型 "
            "| 颜色、亮灭、遮挡和部件位置 "
            "| 延迟和成本较高，输出可能不稳定 |"
        ),
        (
            "| 直接拒答 "
            "| 不调用识别工具 "
            "| 严重模糊、遮挡或无法确认的图片 "
            "| 不能处理本来清晰可回答的图片 |"
        ),
        "",
        "## 4. 路线汇总",
        "",
        (
            "| 路线 | 案例数 | 完成 | 部分 | 拒答 | 失败 "
            "| 内容准确率 | 通过率 | 安全通过率 "
            "| 平均延迟 ms | P50 ms | P95 ms "
            "| 外部模型调用 | 总成本 |"
        ),
        (
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
            "---:|---:|---:|---:|---:|"
        ),
    ]

    for summary in (
        report.route_summaries
    ):
        lines.append(
            "| "
            + markdown_cell(
                ROUTE_DISPLAY_NAMES[
                    summary.route
                ]
            )
            + " | "
            + str(
                summary.evaluated_case_count
            )
            + " | "
            + str(summary.completed_count)
            + " | "
            + str(summary.partial_count)
            + " | "
            + str(summary.abstained_count)
            + " | "
            + str(summary.failed_count)
            + " | "
            + format_ratio(
                summary.mean_content_accuracy
            )
            + " | "
            + format_ratio(
                summary.pass_rate
            )
            + " | "
            + format_ratio(
                summary.safety_pass_rate
            )
            + " | "
            + f"{summary.mean_latency_ms:.3f}"
            + " | "
            + f"{summary.p50_latency_ms:.3f}"
            + " | "
            + f"{summary.p95_latency_ms:.3f}"
            + " | "
            + str(
                summary
                .external_model_call_count
            )
            + " | "
            + format_cost(summary)
            + " |"
        )

    lines.extend(
        [
            "",
            "## 5. 案例定义",
            "",
            (
                "| 案例 | 类型 | 问题 | 首选路线 "
                "| 可接受路线 | 选择依据 |"
            ),
            "|---|---|---|---|---|---|",
        ]
    )

    for case in cases:
        lines.append(
            "| "
            + markdown_cell(case.case_id)
            + " | "
            + markdown_cell(
                case.sample_kind
            )
            + " | "
            + markdown_cell(case.question)
            + " | "
            + markdown_cell(
                ROUTE_DISPLAY_NAMES[
                    case.preferred_route
                ]
            )
            + " | "
            + markdown_cell(
                "、".join(
                    ROUTE_DISPLAY_NAMES[
                        route
                    ]
                    for route
                    in case.acceptable_routes
                )
            )
            + " | "
            + markdown_cell(
                case.rationale
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## 6. 单案例路线结果",
            "",
            (
                "| 案例 | 路线 | 执行状态 | 实际观察 "
                "| 内容准确率 | 路线适合 | 结果正确 "
                "| 安全通过 | 最终通过 | 延迟 ms "
                "| 失败类型或说明 |"
            ),
            (
                "|---|---|---|---|---:|---|---|---|---|---:|---|"
            ),
        ]
    )

    for result in report.results:
        explanation = (
            result.failure_kind
            or result.public_message
            or "无"
        )

        lines.append(
            "| "
            + markdown_cell(result.case_id)
            + " | "
            + markdown_cell(
                ROUTE_DISPLAY_NAMES[
                    result.route
                ]
            )
            + " | "
            + markdown_cell(
                result.execution_status
            )
            + " | "
            + markdown_cell(
                format_actual_items(
                    result.actual_items
                )
            )
            + " | "
            + format_ratio(
                result.content_accuracy
            )
            + " | "
            + (
                "是"
                if result.route_appropriate
                else "否"
            )
            + " | "
            + (
                "是"
                if result.outcome_correct
                else "否"
            )
            + " | "
            + (
                "是"
                if result.safety_passed
                else "否"
            )
            + " | "
            + (
                "是"
                if result.passed
                else "否"
            )
            + " | "
            + f"{result.latency_ms:.3f}"
            + " | "
            + markdown_cell(explanation)
            + " |"
        )

    preferred_results = tuple(
        result
        for result in report.results
        if (
            result.route
            == case_by_id[
                result.case_id
            ].preferred_route
        )
    )

    preferred_passed_count = sum(
        1
        for result in preferred_results
        if result.passed
    )

    failed_results = tuple(
        result
        for result in report.results
        if result.execution_status
        == "failed"
    )

    lines.extend(
        [
            "",
            "## 7. 结果结论",
            "",
            (
                "- 首选路线通过数："
                f"{preferred_passed_count}"
                f"/{len(preferred_results)}"
            ),
            (
                "- 技术失败结果数："
                f"{len(failed_results)}"
            ),
            (
                "- OCR路线应优先用于固定标签、"
                "故障码和数值读取。"
            ),
            (
                "- Vision路线应保留给颜色、亮灭、"
                "部件位置和遮挡等真正需要视觉理解的任务。"
            ),
            (
                "- 图片无法支持问题时，应返回拒答，"
                "不能用模型猜测被遮挡或模糊的信息。"
            ),
            "",
            "## 8. 边界",
            "",
            (
                "- 当前样本是使用Pillow生成的脱敏教学图片，"
                "不代表真实设备图片分布。"
            ),
            (
                "- 内容准确率采用确定性标准项匹配，"
                "不能覆盖所有自然语言等价表达。"
            ),
            (
                "- Vision成本仅在Provider提供足够用量信息时估算。"
            ),
            (
                "- 本实验只评估图片观察和工具适用性，"
                "不生成机器人控制指令。"
            ),
            "",
        ]
    )

    return "\n".join(lines)


def write_reports(
    *,
    report: (
        MultimodalToolSelectionExperimentReport
    ),
    cases: tuple[
        MultimodalToolSelectionCase,
        ...,
    ],
    json_output_path: Path,
    markdown_output_path: Path,
) -> None:
    """写入JSON和Markdown报告。"""

    if not isinstance(
        json_output_path,
        Path,
    ):
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

    resolved_json_path = (
        json_output_path.resolve()
    )
    resolved_markdown_path = (
        markdown_output_path.resolve()
    )

    if (
        resolved_json_path
        == resolved_markdown_path
    ):
        raise ValueError(
            "JSON和Markdown不能写入同一路径"
        )

    json_text = json.dumps(
        report.model_dump(
            mode="json"
        ),
        ensure_ascii=False,
        indent=2,
    ) + "\n"

    markdown_text = (
        render_markdown_report(
            report=report,
            cases=cases,
        )
    )

    resolved_json_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    resolved_markdown_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    resolved_json_path.write_text(
        json_text,
        encoding="utf-8",
    )
    resolved_markdown_path.write_text(
        markdown_text,
        encoding="utf-8",
    )


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(
        description=(
            "使用真实Tesseract和Vision Provider"
            "运行多模态工具选择对照实验"
        )
    )

    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES_PATH,
        help=(
            "多模态工具选择JSONL案例清单"
        ),
    )

    parser.add_argument(
        "--json-output",
        type=Path,
        default=DEFAULT_JSON_OUTPUT_PATH,
        help="JSON报告输出路径",
    )

    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=(
            DEFAULT_MARKDOWN_OUTPUT_PATH
        ),
        help="Markdown报告输出路径",
    )

    return parser.parse_args(argv)


def print_summary(
    *,
    report: (
        MultimodalToolSelectionExperimentReport
    ),
    json_output_path: Path,
    markdown_output_path: Path,
) -> None:
    """输出简短且可复制的终端摘要。"""

    print("多模态工具选择对照实验完成")
    print(f"案例数：{report.case_count}")
    print(
        "单路线结果数："
        f"{report.route_result_count}"
    )

    for summary in (
        report.route_summaries
    ):
        route_name = (
            ROUTE_DISPLAY_NAMES[
                summary.route
            ]
        )

        print(
            f"{route_name}："
            f"通过率={summary.pass_rate:.3f}，"
            f"平均延迟="
            f"{summary.mean_latency_ms:.1f}ms，"
            f"P95="
            f"{summary.p95_latency_ms:.1f}ms"
        )

    print(
        "JSON报告："
        f"{json_output_path}"
    )
    print(
        "Markdown报告："
        f"{markdown_output_path}"
    )


def main() -> None:
    """运行完整命令行流程。"""

    args = parse_args()

    cases_path = resolve_project_path(
        args.cases
    )
    json_output_path = (
        resolve_project_path(
            args.json_output
        )
    )
    markdown_output_path = (
        resolve_project_path(
            args.markdown_output
        )
    )

    cases = load_cases(
        cases_path
    )

    validate_experiment_scope(
        cases
    )

    settings = get_settings()

    report = asyncio.run(
        run_real_experiment(
            cases=cases,
            settings=settings,
        )
    )

    # 只有整批实验成功并形成合法报告后，
    # 才进入报告写入阶段。
    write_reports(
        report=report,
        cases=cases,
        json_output_path=(
            json_output_path
        ),
        markdown_output_path=(
            markdown_output_path
        ),
    )

    print_summary(
        report=report,
        json_output_path=(
            json_output_path
        ),
        markdown_output_path=(
            markdown_output_path
        ),
    )


if __name__ == "__main__":
    main()