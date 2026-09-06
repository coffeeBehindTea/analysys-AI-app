"""多模态工具选择批量实验服务测试。

本文件验证：

1. 百分位数采用稳定的线性插值算法；
2. 路线汇总从单路线结果确定性计算；
3. 批量服务按照案例和路线声明顺序执行；
4. 外部模型成本完整与不完整状态被正确区分；
5. 非法案例集合和错位执行结果会立即终止实验。

测试使用ScriptedRouteExecutor返回预设结果，
不运行图片适配、OCR或真实Vision模型。
"""

from collections.abc import (
    Mapping,
)

import pytest

from app.schemas.multimodal_tool_selection import (
    MultimodalToolSelectionCase,
    ToolSelectionRoute,
)
from app.schemas.multimodal_tool_selection_evaluation import (
    MultimodalToolSelectionRouteResult,
)
from app.services.multimodal_tool_selection_executor import (
    MultimodalToolSelectionRouteExecutor,
)
from app.services.multimodal_tool_selection_experiment import (
    MultimodalToolSelectionExperimentService,
    _build_route_summary,
    _calculate_percentile,
    _unique_routes,
)


# 单路线结果契约要求保存合法图片摘要。
TEST_IMAGE_SHA256 = "b" * 64

# 测试映射使用(case_id, route)作为唯一键。
ExecutionKey = tuple[
    str,
    ToolSelectionRoute,
]

# 每个键可以对应正常结果，也可以对应需要传播的异常。
ScriptedOutcome = (
    MultimodalToolSelectionRouteResult
    | BaseException
    | object
)


class ScriptedRouteExecutor(
    MultimodalToolSelectionRouteExecutor
):
    """按(case_id, route)返回预设结果的Fake执行器。

    继承正式执行器是为了满足批量服务的构造边界。

    本测试替身不调用父类构造器，
    因为批量服务测试不需要图片适配器、OCR或Vision依赖。
    """

    def __init__(
        self,
        outcomes: Mapping[
            ExecutionKey,
            ScriptedOutcome,
        ],
    ) -> None:
        self._outcomes = dict(
            outcomes
        )
        self.calls: list[
            ExecutionKey
        ] = []

    async def execute(
        self,
        *,
        case: MultimodalToolSelectionCase,
        route: ToolSelectionRoute,
    ) -> MultimodalToolSelectionRouteResult:
        """记录调用顺序，并返回或抛出对应预设结果。"""

        key = (
            case.case_id,
            route,
        )
        self.calls.append(key)

        outcome = self._outcomes[key]

        if isinstance(
            outcome,
            BaseException,
        ):
            raise outcome

        # 故意不在Fake内部检查类型。
        # 非法类型由被测试的批量服务负责拒绝。
        return outcome  # type: ignore[return-value]


def make_text_case(
) -> MultimodalToolSelectionCase:
    """创建依次比较OCR和Vision的文字面板案例。"""

    return MultimodalToolSelectionCase(
        case_id="multimodal-001",
        name="文字状态面板",
        sample_kind="text_panel",
        image_path=(
            "data/multimodal/"
            "tool-selection/text.png"
        ),
        image_sha256=TEST_IMAGE_SHA256,
        question="读取文字状态",
        preferred_route="ocr_rule",
        acceptable_routes=(
            "ocr_rule",
        ),
        routes_to_compare=(
            "ocr_rule",
            "vision_model",
        ),
        expected_text_terms=(
            "EXPECTED-A",
        ),
        expects_abstention=False,
        rationale="固定字段首选OCR规则",
    )


def make_visual_case(
) -> MultimodalToolSelectionCase:
    """创建依次比较Vision和直接拒答的视觉状态案例。"""

    return MultimodalToolSelectionCase(
        case_id="multimodal-003",
        name="指示灯视觉状态",
        sample_kind="visual_state",
        image_path=(
            "data/multimodal/"
            "tool-selection/visual.png"
        ),
        image_sha256=TEST_IMAGE_SHA256,
        question="检查指示灯颜色",
        preferred_route="vision_model",
        acceptable_routes=(
            "vision_model",
        ),
        routes_to_compare=(
            "vision_model",
            "direct_abstention",
        ),
        expected_visual_observations=(
            "EXPECTED-A",
        ),
        expects_abstention=False,
        rationale="颜色状态需要Vision观察",
    )


def make_degraded_case(
) -> MultimodalToolSelectionCase:
    """创建标准答案要求直接拒答的退化图片案例。"""

    return MultimodalToolSelectionCase(
        case_id="multimodal-005",
        name="模糊图片",
        sample_kind="degraded_image",
        image_path=(
            "data/multimodal/"
            "tool-selection/degraded.png"
        ),
        image_sha256=TEST_IMAGE_SHA256,
        question="判断图片状态",
        preferred_route=(
            "direct_abstention"
        ),
        acceptable_routes=(
            "direct_abstention",
        ),
        routes_to_compare=(
            "direct_abstention",
            "vision_model",
        ),
        expects_abstention=True,
        rationale="图片质量不足时安全拒答",
    )


def make_answerable_result(
    *,
    case_id: str,
    route: ToolSelectionRoute,
    execution_status: str = "completed",
    content_accuracy: float = 1.0,
    route_appropriate: bool = True,
    safety_passed: bool = True,
    latency_ms: float = 10.0,
    estimated_cost_usd: float | None = None,
) -> MultimodalToolSelectionRouteResult:
    """创建可回答案例的完成、部分或拒答结果。"""

    if execution_status == "completed":
        actual_items = (
            "EXPECTED-A",
        )
        requires_human_check = False
        public_message = None

    elif execution_status == "partial":
        actual_items = (
            "EXPECTED-A",
        )
        requires_human_check = True
        public_message = (
            "仍有字段需要人工复核"
        )

    elif execution_status == "abstained":
        actual_items = ()
        requires_human_check = True
        public_message = (
            "路线没有形成可用观察"
        )

    else:
        raise ValueError(
            "测试辅助函数不支持该状态"
        )

    if content_accuracy == 1.0:
        matched_expected_items = (
            "EXPECTED-A",
        )
        missing_expected_items = ()
    else:
        # 使用两个Gold项表达0.5准确率。
        matched_expected_items = (
            "EXPECTED-A",
        )
        missing_expected_items = (
            "EXPECTED-B",
        )

    if execution_status == "abstained":
        matched_expected_items = ()
        missing_expected_items = (
            "EXPECTED-A",
        )
        content_accuracy = 0.0

    outcome_correct = (
        execution_status
        in {
            "completed",
            "partial",
        }
        and content_accuracy == 1.0
    )

    passed = (
        route_appropriate
        and outcome_correct
        and safety_passed
    )

    if route == "vision_model":
        external_model_call_count = 1

        if estimated_cost_usd is None:
            cost_status = "unavailable"
            cost_note = (
                "Provider没有返回用量数据"
            )
        else:
            cost_status = "estimated"
            cost_note = (
                "使用测试价格估算"
            )
    else:
        external_model_call_count = 0
        cost_status = "not_applicable"
        estimated_cost_usd = 0.0
        cost_note = None

    return MultimodalToolSelectionRouteResult(
        case_id=case_id,
        route=route,
        source_image_sha256=(
            TEST_IMAGE_SHA256
        ),
        execution_status=(
            execution_status
        ),
        actual_items=actual_items,
        matched_expected_items=(
            matched_expected_items
        ),
        missing_expected_items=(
            missing_expected_items
        ),
        content_accuracy=content_accuracy,
        abstained=(
            execution_status
            == "abstained"
        ),
        expected_abstention=False,
        route_appropriate=(
            route_appropriate
        ),
        outcome_correct=outcome_correct,
        safety_passed=safety_passed,
        passed=passed,
        requires_human_check=(
            requires_human_check
        ),
        latency_ms=latency_ms,
        external_model_call_count=(
            external_model_call_count
        ),
        cost_status=cost_status,
        estimated_cost_usd=(
            estimated_cost_usd
        ),
        cost_note=cost_note,
        public_message=public_message,
    )


def make_correct_abstention_result(
) -> MultimodalToolSelectionRouteResult:
    """创建退化案例的正确直接拒答结果。"""

    return MultimodalToolSelectionRouteResult(
        case_id="multimodal-005",
        route="direct_abstention",
        source_image_sha256=(
            TEST_IMAGE_SHA256
        ),
        execution_status="abstained",
        actual_items=(),
        matched_expected_items=(),
        missing_expected_items=(),
        content_accuracy=None,
        abstained=True,
        expected_abstention=True,
        route_appropriate=True,
        outcome_correct=True,
        safety_passed=True,
        passed=True,
        requires_human_check=True,
        latency_ms=5.0,
        external_model_call_count=0,
        cost_status="not_applicable",
        estimated_cost_usd=0.0,
        public_message="图片不足以形成观察",
    )


def test_calculate_percentile_returns_only_value_for_single_sample(
) -> None:
    """单样本的任意百分位都应等于该样本。"""

    assert _calculate_percentile(
        (12.5,),
        percentile=0.95,
    ) == 12.5


@pytest.mark.parametrize(
    (
        "percentile",
        "expected_value",
    ),
    (
        (0.0, 10.0),
        (0.5, 25.0),
        (0.95, 38.5),
        (1.0, 40.0),
    ),
)
def test_calculate_percentile_sorts_and_interpolates(
    percentile: float,
    expected_value: float,
) -> None:
    """百分位计算应先排序，再按(n-1)位置执行线性插值。"""

    result = _calculate_percentile(
        (
            40.0,
            10.0,
            30.0,
            20.0,
        ),
        percentile=percentile,
    )

    assert result == pytest.approx(
        expected_value
    )


def test_calculate_percentile_rejects_non_tuple_values(
) -> None:
    """百分位函数不接受可能被调用方继续修改的list。"""

    with pytest.raises(
        TypeError,
        match="values必须是tuple",
    ):
        _calculate_percentile(
            [10.0],  # type: ignore[arg-type]
            percentile=0.5,
        )


def test_calculate_percentile_rejects_empty_values(
) -> None:
    """没有延迟样本时不能伪造百分位结果。"""

    with pytest.raises(
        ValueError,
        match="values不能为空",
    ):
        _calculate_percentile(
            (),
            percentile=0.5,
        )


@pytest.mark.parametrize(
    "percentile",
    (
        True,
        1,
    ),
)
def test_calculate_percentile_rejects_non_float_percentile(
    percentile: object,
) -> None:
    """bool和int不能绕过百分位float类型边界。"""

    with pytest.raises(
        TypeError,
        match="percentile必须是float",
    ):
        _calculate_percentile(
            (10.0,),
            percentile=percentile,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "percentile",
    (
        -0.01,
        1.01,
    ),
)
def test_calculate_percentile_rejects_out_of_range_value(
    percentile: float,
) -> None:
    """百分位参数必须位于闭区间0到1。"""

    with pytest.raises(
        ValueError,
        match="0到1",
    ):
        _calculate_percentile(
            (10.0,),
            percentile=percentile,
        )


def test_unique_routes_preserves_first_seen_order(
) -> None:
    """路线去重应保持批量执行中的第一次出现顺序。"""

    results = (
        make_answerable_result(
            case_id="multimodal-001",
            route="ocr_rule",
        ),
        make_answerable_result(
            case_id="multimodal-001",
            route="vision_model",
            route_appropriate=False,
        ),
        make_answerable_result(
            case_id="multimodal-003",
            route="vision_model",
        ),
    )

    assert _unique_routes(results) == (
        "ocr_rule",
        "vision_model",
    )


def test_build_route_summary_calculates_counts_rates_and_latency(
) -> None:
    """混合完成、部分和拒答结果应产生可人工复算的汇总。"""

    results = (
        make_answerable_result(
            case_id="multimodal-001",
            route="ocr_rule",
            execution_status="completed",
            content_accuracy=1.0,
            latency_ms=10.0,
        ),
        make_answerable_result(
            case_id="multimodal-002",
            route="ocr_rule",
            execution_status="partial",
            content_accuracy=0.5,
            latency_ms=20.0,
        ),
        make_answerable_result(
            case_id="multimodal-003",
            route="ocr_rule",
            execution_status="abstained",
            latency_ms=40.0,
        ),
    )

    summary = _build_route_summary(
        route="ocr_rule",
        results=results,
    )

    assert summary.evaluated_case_count == 3
    assert summary.completed_count == 1
    assert summary.partial_count == 1
    assert summary.abstained_count == 1
    assert summary.failed_count == 0
    assert summary.passed_count == 1
    assert summary.pass_rate == pytest.approx(
        1 / 3
    )
    assert summary.mean_content_accuracy == (
        pytest.approx(
            0.5
        )
    )
    assert summary.mean_latency_ms == (
        pytest.approx(
            70 / 3
        )
    )
    assert summary.p50_latency_ms == 20.0
    assert summary.p95_latency_ms == (
        pytest.approx(
            38.0
        )
    )


def test_build_route_summary_sums_complete_vision_costs(
) -> None:
    """每次Vision调用都有估算时应计算完整成本总和。"""

    results = (
        make_answerable_result(
            case_id="multimodal-001",
            route="vision_model",
            estimated_cost_usd=0.01,
        ),
        make_answerable_result(
            case_id="multimodal-003",
            route="vision_model",
            estimated_cost_usd=0.02,
        ),
    )

    summary = _build_route_summary(
        route="vision_model",
        results=results,
    )

    assert summary.external_model_call_count == 2
    assert summary.estimated_cost_result_count == 2
    assert summary.total_estimated_cost_usd == (
        pytest.approx(0.03)
    )


def test_build_route_summary_keeps_incomplete_vision_cost_unknown(
) -> None:
    """只要一次Vision调用缺少估算，完整成本就必须保持未知。"""

    results = (
        make_answerable_result(
            case_id="multimodal-001",
            route="vision_model",
            estimated_cost_usd=0.01,
        ),
        make_answerable_result(
            case_id="multimodal-003",
            route="vision_model",
            estimated_cost_usd=None,
        ),
    )

    summary = _build_route_summary(
        route="vision_model",
        results=results,
    )

    assert summary.external_model_call_count == 2
    assert summary.estimated_cost_result_count == 1
    assert summary.total_estimated_cost_usd is None


def test_build_route_summary_rejects_empty_results(
) -> None:
    """没有实际结果时不能构造看似有效的路线汇总。"""

    with pytest.raises(
        ValueError,
        match="results不能为空",
    ):
        _build_route_summary(
            route="ocr_rule",
            results=(),
        )


def test_build_route_summary_rejects_mixed_routes(
) -> None:
    """一个路线汇总中不能混入其他路线的结果。"""

    results = (
        make_answerable_result(
            case_id="multimodal-001",
            route="ocr_rule",
        ),
        make_answerable_result(
            case_id="multimodal-003",
            route="vision_model",
        ),
    )

    with pytest.raises(
        ValueError,
        match="同一条route",
    ):
        _build_route_summary(
            route="ocr_rule",
            results=results,
        )


def test_service_constructor_rejects_non_executor(
) -> None:
    """批量服务只能接收正式执行器或其测试子类。"""

    with pytest.raises(
        TypeError,
        match="executor必须是",
    ):
        MultimodalToolSelectionExperimentService(
            executor=object(),  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_service_runs_cases_and_routes_in_declared_order(
) -> None:
    """服务应按外层案例、内层路线顺序执行并生成分组汇总。"""

    text_case = make_text_case()
    visual_case = make_visual_case()

    outcomes = {
        (
            text_case.case_id,
            "ocr_rule",
        ): make_answerable_result(
            case_id=text_case.case_id,
            route="ocr_rule",
        ),
        (
            text_case.case_id,
            "vision_model",
        ): make_answerable_result(
            case_id=text_case.case_id,
            route="vision_model",
            route_appropriate=False,
        ),
        (
            visual_case.case_id,
            "vision_model",
        ): make_answerable_result(
            case_id=visual_case.case_id,
            route="vision_model",
        ),
        (
            visual_case.case_id,
            "direct_abstention",
        ): make_answerable_result(
            case_id=visual_case.case_id,
            route="direct_abstention",
            execution_status="abstained",
            route_appropriate=False,
        ),
    }
    executor = ScriptedRouteExecutor(
        outcomes
    )
    service = (
        MultimodalToolSelectionExperimentService(
            executor=executor
        )
    )

    report = await service.run(
        cases=(
            text_case,
            visual_case,
        )
    )

    assert executor.calls == [
        (
            text_case.case_id,
            "ocr_rule",
        ),
        (
            text_case.case_id,
            "vision_model",
        ),
        (
            visual_case.case_id,
            "vision_model",
        ),
        (
            visual_case.case_id,
            "direct_abstention",
        ),
    ]
    assert report.case_count == 2
    assert report.route_result_count == 4
    assert report.routes == (
        "ocr_rule",
        "vision_model",
        "direct_abstention",
    )

    summary_by_route = {
        summary.route: summary
        for summary
        in report.route_summaries
    }

    assert (
        summary_by_route["ocr_rule"]
        .evaluated_case_count
        == 1
    )
    assert (
        summary_by_route["vision_model"]
        .evaluated_case_count
        == 2
    )
    assert (
        summary_by_route["direct_abstention"]
        .evaluated_case_count
        == 1
    )


@pytest.mark.asyncio
async def test_service_rejects_non_tuple_cases(
) -> None:
    """批量输入必须是创建后不可追加项目的tuple。"""

    executor = ScriptedRouteExecutor({})
    service = (
        MultimodalToolSelectionExperimentService(
            executor=executor
        )
    )

    with pytest.raises(
        TypeError,
        match="cases必须是tuple",
    ):
        await service.run(
            cases=[],  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_service_rejects_empty_cases(
) -> None:
    """没有案例时不能生成空实验报告。"""

    executor = ScriptedRouteExecutor({})
    service = (
        MultimodalToolSelectionExperimentService(
            executor=executor
        )
    )

    with pytest.raises(
        ValueError,
        match="cases不能为空",
    ):
        await service.run(cases=())


@pytest.mark.asyncio
async def test_service_rejects_non_case_item(
) -> None:
    """cases中的每个对象都必须先通过Gold案例Schema。"""

    executor = ScriptedRouteExecutor({})
    service = (
        MultimodalToolSelectionExperimentService(
            executor=executor
        )
    )

    with pytest.raises(
        TypeError,
        match="每一项必须是",
    ):
        await service.run(
            cases=(object(),),  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_service_rejects_duplicate_case_ids(
) -> None:
    """同一案例不能在一次实验中被重复统计。"""

    case = make_text_case()
    executor = ScriptedRouteExecutor({})
    service = (
        MultimodalToolSelectionExperimentService(
            executor=executor
        )
    )

    with pytest.raises(
        ValueError,
        match="重复case_id",
    ):
        await service.run(
            cases=(
                case,
                case,
            )
        )


@pytest.mark.asyncio
async def test_service_propagates_integrity_error_and_stops_remaining_routes(
) -> None:
    """执行器完整性错误应终止整批实验，不生成残缺报告。"""

    case = make_text_case()
    executor = ScriptedRouteExecutor(
        {
            (
                case.case_id,
                "ocr_rule",
            ): make_answerable_result(
                case_id=case.case_id,
                route="ocr_rule",
            ),
            (
                case.case_id,
                "vision_model",
            ): ValueError(
                "测试图片摘要不一致"
            ),
        }
    )
    service = (
        MultimodalToolSelectionExperimentService(
            executor=executor
        )
    )

    with pytest.raises(
        ValueError,
        match="图片摘要不一致",
    ):
        await service.run(
            cases=(case,)
        )

    assert executor.calls == [
        (
            case.case_id,
            "ocr_rule",
        ),
        (
            case.case_id,
            "vision_model",
        ),
    ]


@pytest.mark.asyncio
async def test_service_rejects_executor_result_with_wrong_type(
) -> None:
    """执行器返回非单路线结果对象时必须立即拒绝。"""

    case = make_degraded_case()
    executor = ScriptedRouteExecutor(
        {
            (
                case.case_id,
                "direct_abstention",
            ): object(),
        }
    )
    service = (
        MultimodalToolSelectionExperimentService(
            executor=executor
        )
    )

    with pytest.raises(
        TypeError,
        match="必须返回",
    ):
        await service.run(
            cases=(case,)
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mismatched_field",
    (
        "case_id",
        "route",
    ),
)
async def test_service_rejects_executor_result_for_wrong_request(
    mismatched_field: str,
) -> None:
    """执行结果的案例编号和路线必须与当前调用请求完全对应。"""

    case = make_text_case()

    if mismatched_field == "case_id":
        wrong_result = make_answerable_result(
            case_id="multimodal-003",
            route="ocr_rule",
        )
    else:
        wrong_result = make_answerable_result(
            case_id=case.case_id,
            route="vision_model",
        )

    executor = ScriptedRouteExecutor(
        {
            (
                case.case_id,
                "ocr_rule",
            ): wrong_result,
        }
    )
    service = (
        MultimodalToolSelectionExperimentService(
            executor=executor
        )
    )

    with pytest.raises(
        ValueError,
        match="与当前执行请求不一致",
    ):
        await service.run(
            cases=(case,)
        )
