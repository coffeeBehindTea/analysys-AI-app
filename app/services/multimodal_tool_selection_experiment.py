"""多模态工具选择批量对照实验服务。

本模块负责：

1. 接收一组已经校验的Gold案例；
2. 按稳定顺序执行每个案例声明的路线；
3. 收集全部单路线评分结果；
4. 按路线计算数量、准确率、延迟和成本汇总；
5. 构造经过Pydantic校验的完整实验报告。

本模块不负责：

1. 从JSONL读取案例；
2. 创建OCR、Vision或输入适配器；
3. 修改Gold标准答案；
4. 并发调用外部模型；
5. 写入JSON或Markdown文件。
"""

# ceil和floor用于计算百分位数插值位置。
from math import (
    ceil,
    floor,
)

# fmean使用浮点算法计算平均值。
#
# 与sum(values) / len(values)相比，
# fmean可以直接表达“计算浮点平均数”的意图。
from statistics import (
    fmean,
)

# Gold案例和路线名称。
from app.schemas.multimodal_tool_selection import (
    MultimodalToolSelectionCase,
    ToolSelectionRoute,
)

# 单条路线执行结果。
from app.schemas.multimodal_tool_selection_evaluation import (
    MultimodalToolSelectionRouteResult,
)

# 批量报告和路线汇总契约。
from app.schemas.multimodal_tool_selection_report import (
    MultimodalToolSelectionExperimentReport,
    MultimodalToolSelectionRouteSummary,
)

# 单路线执行器。
from app.services.multimodal_tool_selection_executor import (
    MultimodalToolSelectionRouteExecutor,
)


def _unique_routes(
    results: tuple[
        MultimodalToolSelectionRouteResult,
        ...,
    ],
) -> tuple[
    ToolSelectionRoute,
    ...,
]:
    """按照第一次出现的顺序返回唯一路线。

    dict从Python 3.7开始保证保持键的插入顺序。

    fromkeys()把路线作为键去重，
    因而能够同时做到：

    1. 删除重复路线；
    2. 保持批量执行时的稳定顺序。
    """

    return tuple(
        dict.fromkeys(
            result.route
            for result in results
        )
    )


def _calculate_percentile(
    values: tuple[
        float,
        ...,
    ],
    *,
    percentile: float,
) -> float:
    """使用线性插值计算一个百分位数。

    percentile使用0到1之间的小数：

    0.50表示P50；
    0.95表示P95。

    算法步骤：

    1. 将数据从小到大排列；
    2. 使用(n - 1) * percentile计算理论位置；
    3. 如果位置是整数，直接返回该位置的值；
    4. 如果位置在两个样本之间，执行线性插值。

    例如两个延迟：

    10 ms、20 ms

    P50的位置是：

    (2 - 1) * 0.5 = 0.5

    因而结果是10和20中间的15 ms。
    """

    if not isinstance(
        values,
        tuple,
    ):
        raise TypeError(
            "values必须是tuple"
        )

    if not values:
        raise ValueError(
            "计算百分位数时values不能为空"
        )

    if (
        isinstance(percentile, bool)
        or not isinstance(
            percentile,
            float,
        )
    ):
        raise TypeError(
            "percentile必须是float"
        )

    if (
        percentile < 0.0
        or percentile > 1.0
    ):
        raise ValueError(
            "percentile必须位于0到1之间"
        )

    ordered_values = tuple(
        sorted(values)
    )

    if len(ordered_values) == 1:
        return ordered_values[0]

    position = (
        len(ordered_values) - 1
    ) * percentile

    lower_index = floor(position)
    upper_index = ceil(position)

    if lower_index == upper_index:
        return ordered_values[
            lower_index
        ]

    lower_value = ordered_values[
        lower_index
    ]
    upper_value = ordered_values[
        upper_index
    ]

    interpolation_weight = (
        position - lower_index
    )

    return (
        lower_value
        + (
            upper_value
            - lower_value
        )
        * interpolation_weight
    )


def _count_status(
    results: tuple[
        MultimodalToolSelectionRouteResult,
        ...,
    ],
    *,
    status: str,
) -> int:
    """统计指定execution_status的结果数量。"""

    return sum(
        1
        for result in results
        if result.execution_status
        == status
    )


def _count_true_field(
    results: tuple[
        MultimodalToolSelectionRouteResult,
        ...,
    ],
    *,
    field_name: str,
) -> int:
    """统计指定布尔字段为True的结果数量。

    getattr(result, field_name)等价于：

    result.passed
    result.outcome_correct
    result.safety_passed

    区别是字段名可以由参数提供，
    便于复用相同的统计逻辑。
    """

    return sum(
        1
        for result in results
        if getattr(
            result,
            field_name,
        )
        is True
    )


def _build_route_summary(
    *,
    route: ToolSelectionRoute,
    results: tuple[
        MultimodalToolSelectionRouteResult,
        ...,
    ],
) -> MultimodalToolSelectionRouteSummary:
    """根据一条路线的全部结果计算汇总指标。"""

    if not results:
        raise ValueError(
            "构造路线汇总时results不能为空"
        )

    # 调用方已经按照路线分组，
    # 这里再次保护模块自己的输入边界。
    if any(
        result.route != route
        for result in results
    ):
        raise ValueError(
            "路线汇总中的所有结果"
            "必须属于同一条route"
        )

    evaluated_case_count = len(
        results
    )

    completed_count = _count_status(
        results,
        status="completed",
    )
    partial_count = _count_status(
        results,
        status="partial",
    )
    abstained_count = _count_status(
        results,
        status="abstained",
    )
    failed_count = _count_status(
        results,
        status="failed",
    )

    passed_count = _count_true_field(
        results,
        field_name="passed",
    )
    route_appropriate_count = (
        _count_true_field(
            results,
            field_name=(
                "route_appropriate"
            ),
        )
    )
    outcome_correct_count = (
        _count_true_field(
            results,
            field_name=(
                "outcome_correct"
            ),
        )
    )
    safety_passed_count = (
        _count_true_field(
            results,
            field_name=(
                "safety_passed"
            ),
        )
    )

    content_accuracy_values = tuple(
        result.content_accuracy
        for result in results
        if result.content_accuracy
        is not None
    )

    if content_accuracy_values:
        mean_content_accuracy = fmean(
            content_accuracy_values
        )
    else:
        mean_content_accuracy = None

    latency_values = tuple(
        result.latency_ms
        for result in results
    )

    external_model_call_count = sum(
        result.external_model_call_count
        for result in results
    )

    estimated_cost_values = tuple(
        result.estimated_cost_usd
        for result in results
        if (
            result.cost_status
            == "estimated"
            and result
            .estimated_cost_usd
            is not None
        )
    )

    estimated_cost_result_count = len(
        estimated_cost_values
    )

    if route != "vision_model":
        # OCR和直接拒答没有外部Vision模型成本。
        total_estimated_cost_usd: (
            float
            | None
        ) = 0.0

    elif (
        external_model_call_count
        == estimated_cost_result_count
    ):
        # 只有每一次外部模型调用都有成本值，
        # 才能计算完整总成本。
        total_estimated_cost_usd = sum(
            estimated_cost_values,
            start=0.0,
        )

    else:
        # 只要有一次模型调用缺少成本，
        # 完整成本总和就必须保持未知。
        total_estimated_cost_usd = None

    return (
        MultimodalToolSelectionRouteSummary(
            route=route,
            evaluated_case_count=(
                evaluated_case_count
            ),
            completed_count=completed_count,
            partial_count=partial_count,
            abstained_count=abstained_count,
            failed_count=failed_count,
            passed_count=passed_count,
            route_appropriate_count=(
                route_appropriate_count
            ),
            outcome_correct_count=(
                outcome_correct_count
            ),
            safety_passed_count=(
                safety_passed_count
            ),
            content_accuracy_sample_count=(
                len(
                    content_accuracy_values
                )
            ),
            mean_content_accuracy=(
                mean_content_accuracy
            ),
            pass_rate=(
                passed_count
                / evaluated_case_count
            ),
            route_appropriate_rate=(
                route_appropriate_count
                / evaluated_case_count
            ),
            outcome_correct_rate=(
                outcome_correct_count
                / evaluated_case_count
            ),
            safety_pass_rate=(
                safety_passed_count
                / evaluated_case_count
            ),
            mean_latency_ms=fmean(
                latency_values
            ),
            p50_latency_ms=(
                _calculate_percentile(
                    latency_values,
                    percentile=0.50,
                )
            ),
            p95_latency_ms=(
                _calculate_percentile(
                    latency_values,
                    percentile=0.95,
                )
            ),
            external_model_call_count=(
                external_model_call_count
            ),
            estimated_cost_result_count=(
                estimated_cost_result_count
            ),
            total_estimated_cost_usd=(
                total_estimated_cost_usd
            ),
        )
    )


class MultimodalToolSelectionExperimentService:
    """依次运行多模态路线并构造批量实验报告。

    本服务保存一个已经完成依赖装配的单路线执行器。

    它本身不保存每次实验结果，
    因此同一个服务实例可以连续运行多组案例。
    """

    def __init__(
        self,
        *,
        executor: (
            MultimodalToolSelectionRouteExecutor
        ),
    ) -> None:
        """保存单路线执行器。"""

        if not isinstance(
            executor,
            MultimodalToolSelectionRouteExecutor,
        ):
            raise TypeError(
                "executor必须是"
                "MultimodalToolSelectionRouteExecutor"
            )

        self._executor = executor

    async def run(
        self,
        *,
        cases: tuple[
            MultimodalToolSelectionCase,
            ...,
        ],
    ) -> MultimodalToolSelectionExperimentReport:
        """运行一组工具选择对照实验。

        当前采用顺序执行，而不是并发执行。

        原因是：

        1. 六个案例规模很小；
        2. 顺序执行能保留稳定结果顺序；
        3. 避免同时请求外部Vision模型；
        4. 减少限流对延迟对比的干扰；
        5. 更容易定位某个案例的失败位置。
        """

        if not isinstance(
            cases,
            tuple,
        ):
            raise TypeError(
                "cases必须是tuple"
            )

        if not cases:
            raise ValueError(
                "cases不能为空"
            )

        if any(
            not isinstance(
                case,
                MultimodalToolSelectionCase,
            )
            for case in cases
        ):
            raise TypeError(
                "cases中的每一项必须是"
                "MultimodalToolSelectionCase"
            )

        case_ids = tuple(
            case.case_id
            for case in cases
        )

        if len(case_ids) != len(
            set(case_ids)
        ):
            raise ValueError(
                "cases不能包含重复case_id"
            )

        route_results: list[
            MultimodalToolSelectionRouteResult
        ] = []

        # 外层按照Gold案例顺序执行。
        for case in cases:
            # 内层按照该案例声明的路线顺序执行。
            for route in (
                case.routes_to_compare
            ):
                result = await (
                    self
                    ._executor
                    .execute(
                        case=case,
                        route=route,
                    )
                )

                # 批量服务不能仅依赖执行器的类型注解。
                # 运行时仍要确认返回了正式单路线结果契约。
                if not isinstance(
                    result,
                    MultimodalToolSelectionRouteResult,
                ):
                    raise TypeError(
                        "executor.execute()必须返回"
                        "MultimodalToolSelectionRouteResult"
                    )

                # 返回结果必须属于当前正在执行的案例和路线。
                # 如果这里发生错位，后续分组统计将失去意义，
                # 因此应立即终止实验而不是生成错误报告。
                if (
                    result.case_id
                    != case.case_id
                    or result.route != route
                ):
                    raise ValueError(
                        "executor返回结果的case_id或route"
                        "与当前执行请求不一致"
                    )

                route_results.append(
                    result
                )

        results = tuple(
            route_results
        )

        routes = _unique_routes(
            results
        )

        route_summaries = tuple(
            _build_route_summary(
                route=route,
                results=tuple(
                    result
                    for result in results
                    if result.route == route
                ),
            )
            for route in routes
        )

        # 最终交给批量报告Schema执行一致性检查。
        return (
            MultimodalToolSelectionExperimentReport(
                case_count=len(cases),
                route_result_count=len(
                    results
                ),
                routes=routes,
                results=results,
                route_summaries=(
                    route_summaries
                ),
            )
        )
