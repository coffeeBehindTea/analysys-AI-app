"""多模态工具选择批量实验报告Schema测试。

本文件验证：

1. 单路线汇总的状态数量和总数保持一致；
2. 通过率等派生比率必须与原始计数一致；
3. 内容准确率、延迟百分位和成本字段符合语义；
4. 整份报告中的案例、路线、详细结果和汇总能够互相对应。

这里只测试Pydantic数据契约，
不执行OCR、Vision、路线执行器或批量实验。
"""

from copy import (
    deepcopy,
)

import pytest
from pydantic import (
    ValidationError,
)

from app.schemas.multimodal_tool_selection_evaluation import (
    MultimodalToolSelectionRouteResult,
)
from app.schemas.multimodal_tool_selection_report import (
    MULTIMODAL_TOOL_SELECTION_REPORT_VERSION,
    MultimodalToolSelectionExperimentReport,
    MultimodalToolSelectionRouteSummary,
)


# 三个稳定案例编号分别代表文字、视觉和退化图片。
TEXT_CASE_ID = "multimodal-001"
VISION_CASE_ID = "multimodal-003"
DEGRADED_CASE_ID = "multimodal-005"

# 测试不读取图片，但单路线结果契约要求保存合法摘要。
TEST_IMAGE_SHA256 = "a" * 64

# 固定延迟使平均值和百分位字段容易人工核对。
TEST_LATENCY_MS = 10.0


def make_completed_route_result(
    *,
    case_id: str,
    route: str,
) -> MultimodalToolSelectionRouteResult:
    """创建内容完全正确的OCR或Vision路线结果。"""

    if route == "vision_model":
        external_model_call_count = 1
        cost_status = "unavailable"
        estimated_cost_usd = None
        cost_note = (
            "Provider没有返回token usage"
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
        execution_status="completed",
        actual_items=(
            "已确认观察",
        ),
        matched_expected_items=(
            "已确认观察",
        ),
        missing_expected_items=(),
        content_accuracy=1.0,
        abstained=False,
        expected_abstention=False,
        route_appropriate=True,
        outcome_correct=True,
        safety_passed=True,
        passed=True,
        requires_human_check=False,
        latency_ms=TEST_LATENCY_MS,
        external_model_call_count=(
            external_model_call_count
        ),
        cost_status=cost_status,
        estimated_cost_usd=(
            estimated_cost_usd
        ),
        cost_note=cost_note,
    )


def make_abstained_route_result(
) -> MultimodalToolSelectionRouteResult:
    """创建退化图片的正确直接拒答结果。"""

    return MultimodalToolSelectionRouteResult(
        case_id=DEGRADED_CASE_ID,
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
        latency_ms=TEST_LATENCY_MS,
        external_model_call_count=0,
        cost_status="not_applicable",
        estimated_cost_usd=0.0,
        public_message=(
            "图片质量不足，未形成观察"
        ),
    )


def make_route_summary(
    result: MultimodalToolSelectionRouteResult,
) -> MultimodalToolSelectionRouteSummary:
    """根据一个结果构造对应的单案例路线汇总。"""

    execution_counts = {
        "completed_count": 0,
        "partial_count": 0,
        "abstained_count": 0,
        "failed_count": 0,
    }
    execution_counts[
        f"{result.execution_status}_count"
    ] = 1

    has_content_accuracy = (
        result.content_accuracy
        is not None
    )

    if result.external_model_call_count == 0:
        total_estimated_cost_usd = 0.0
    else:
        # 当前测试Vision结果的成本状态是unavailable。
        total_estimated_cost_usd = None

    return MultimodalToolSelectionRouteSummary(
        route=result.route,
        evaluated_case_count=1,
        **execution_counts,
        passed_count=int(result.passed),
        route_appropriate_count=int(
            result.route_appropriate
        ),
        outcome_correct_count=int(
            result.outcome_correct
        ),
        safety_passed_count=int(
            result.safety_passed
        ),
        content_accuracy_sample_count=int(
            has_content_accuracy
        ),
        mean_content_accuracy=(
            result.content_accuracy
        ),
        pass_rate=float(result.passed),
        route_appropriate_rate=float(
            result.route_appropriate
        ),
        outcome_correct_rate=float(
            result.outcome_correct
        ),
        safety_pass_rate=float(
            result.safety_passed
        ),
        mean_latency_ms=(
            result.latency_ms
        ),
        p50_latency_ms=(
            result.latency_ms
        ),
        p95_latency_ms=(
            result.latency_ms
        ),
        external_model_call_count=(
            result
            .external_model_call_count
        ),
        estimated_cost_result_count=(
            int(
                result.cost_status
                == "estimated"
            )
        ),
        total_estimated_cost_usd=(
            total_estimated_cost_usd
        ),
    )


def make_report(
) -> MultimodalToolSelectionExperimentReport:
    """创建包含三类路线的最小合法批量报告。"""

    results = (
        make_completed_route_result(
            case_id=TEXT_CASE_ID,
            route="ocr_rule",
        ),
        make_completed_route_result(
            case_id=VISION_CASE_ID,
            route="vision_model",
        ),
        make_abstained_route_result(),
    )

    return MultimodalToolSelectionExperimentReport(
        case_count=3,
        route_result_count=3,
        routes=(
            "ocr_rule",
            "vision_model",
            "direct_abstention",
        ),
        results=results,
        route_summaries=tuple(
            make_route_summary(result)
            for result in results
        ),
    )


def rebuild_summary(
    *,
    summary: MultimodalToolSelectionRouteSummary,
    updates: dict[str, object],
) -> MultimodalToolSelectionRouteSummary:
    """修改序列化字段并重新触发完整Pydantic校验。"""

    data = summary.model_dump()
    data.update(updates)

    return (
        MultimodalToolSelectionRouteSummary
        .model_validate(data)
    )


def rebuild_report(
    *,
    report: MultimodalToolSelectionExperimentReport,
    updates: dict[str, object],
) -> MultimodalToolSelectionExperimentReport:
    """修改报告字段并重新触发完整Pydantic校验。"""

    data = deepcopy(
        report.model_dump()
    )
    data.update(updates)

    return (
        MultimodalToolSelectionExperimentReport
        .model_validate(data)
    )


def test_route_summary_accepts_consistent_local_route_metrics(
) -> None:
    """本地OCR汇总应接受一致的状态、比率、延迟和零成本。"""

    result = make_completed_route_result(
        case_id=TEXT_CASE_ID,
        route="ocr_rule",
    )

    summary = make_route_summary(result)

    assert summary.route == "ocr_rule"
    assert summary.evaluated_case_count == 1
    assert summary.completed_count == 1
    assert summary.pass_rate == 1.0
    assert summary.mean_content_accuracy == 1.0
    assert summary.external_model_call_count == 0
    assert summary.total_estimated_cost_usd == 0.0


def test_route_summary_accepts_vision_with_unavailable_cost(
) -> None:
    """Vision已调用但缺少用量数据时，完整成本应保持未知。"""

    result = make_completed_route_result(
        case_id=VISION_CASE_ID,
        route="vision_model",
    )

    summary = make_route_summary(result)

    assert summary.external_model_call_count == 1
    assert summary.estimated_cost_result_count == 0
    assert summary.total_estimated_cost_usd is None


def test_route_summary_accepts_abstention_without_content_accuracy(
) -> None:
    """应拒答案例不应伪造内容准确率样本。"""

    summary = make_route_summary(
        make_abstained_route_result()
    )

    assert summary.abstained_count == 1
    assert summary.content_accuracy_sample_count == 0
    assert summary.mean_content_accuracy is None
    assert summary.pass_rate == 1.0


def test_route_summary_rejects_execution_count_mismatch(
) -> None:
    """四种执行状态数量之和必须等于评测案例数。"""

    summary = make_route_summary(
        make_completed_route_result(
            case_id=TEXT_CASE_ID,
            route="ocr_rule",
        )
    )

    with pytest.raises(
        ValidationError,
        match="execution_status",
    ):
        rebuild_summary(
            summary=summary,
            updates={
                "completed_count": 0,
            },
        )


def test_route_summary_rejects_count_above_evaluated_count(
) -> None:
    """通过数量等分类计数不能超过实际评测数量。"""

    summary = make_route_summary(
        make_completed_route_result(
            case_id=TEXT_CASE_ID,
            route="ocr_rule",
        )
    )

    with pytest.raises(
        ValidationError,
        match="passed_count",
    ):
        rebuild_summary(
            summary=summary,
            updates={
                "passed_count": 2,
            },
        )


@pytest.mark.parametrize(
    "rate_field",
    (
        "pass_rate",
        "route_appropriate_rate",
        "outcome_correct_rate",
        "safety_pass_rate",
    ),
)
def test_route_summary_rejects_rate_not_derived_from_count(
    rate_field: str,
) -> None:
    """四个汇总比率都必须由对应计数确定性推导。"""

    summary = make_route_summary(
        make_completed_route_result(
            case_id=TEXT_CASE_ID,
            route="ocr_rule",
        )
    )

    with pytest.raises(
        ValidationError,
        match=rate_field,
    ):
        rebuild_summary(
            summary=summary,
            updates={
                rate_field: 0.5,
            },
        )


@pytest.mark.parametrize(
    (
        "sample_count",
        "mean_accuracy",
    ),
    (
        (0, 1.0),
        (1, None),
    ),
)
def test_route_summary_rejects_inconsistent_content_accuracy_presence(
    sample_count: int,
    mean_accuracy: float | None,
) -> None:
    """平均内容准确率是否存在必须与可计算样本数一致。"""

    summary = make_route_summary(
        make_abstained_route_result()
    )

    with pytest.raises(
        ValidationError,
        match="mean_content_accuracy",
    ):
        rebuild_summary(
            summary=summary,
            updates={
                "content_accuracy_sample_count": (
                    sample_count
                ),
                "mean_content_accuracy": (
                    mean_accuracy
                ),
            },
        )


def test_route_summary_rejects_p95_below_p50(
) -> None:
    """延迟P95不能低于同一组数据的P50。"""

    summary = make_route_summary(
        make_completed_route_result(
            case_id=TEXT_CASE_ID,
            route="ocr_rule",
        )
    )

    with pytest.raises(
        ValidationError,
        match="p95_latency_ms",
    ):
        rebuild_summary(
            summary=summary,
            updates={
                "p50_latency_ms": 20.0,
                "p95_latency_ms": 10.0,
            },
        )


@pytest.mark.parametrize(
    "route",
    (
        "ocr_rule",
        "direct_abstention",
    ),
)
def test_non_vision_summary_rejects_external_model_call(
    route: str,
) -> None:
    """OCR和直接拒答汇总都不能记录外部Vision模型调用。"""

    if route == "ocr_rule":
        result = make_completed_route_result(
            case_id=TEXT_CASE_ID,
            route=route,
        )
    else:
        result = make_abstained_route_result()

    summary = make_route_summary(result)

    with pytest.raises(
        ValidationError,
        match="非vision_model",
    ):
        rebuild_summary(
            summary=summary,
            updates={
                "external_model_call_count": 1,
            },
        )


def test_vision_summary_rejects_total_when_cost_coverage_is_incomplete(
) -> None:
    """Vision部分调用缺少成本时不能给出虚假的完整总成本。"""

    summary = make_route_summary(
        make_completed_route_result(
            case_id=VISION_CASE_ID,
            route="vision_model",
        )
    )

    with pytest.raises(
        ValidationError,
        match="成本信息不完整",
    ):
        rebuild_summary(
            summary=summary,
            updates={
                "total_estimated_cost_usd": 0.01,
            },
        )


def test_vision_summary_requires_total_when_every_call_has_cost(
) -> None:
    """Vision全部调用均有估算时必须保存成本总和。"""

    summary = make_route_summary(
        make_completed_route_result(
            case_id=VISION_CASE_ID,
            route="vision_model",
        )
    )

    with pytest.raises(
        ValidationError,
        match="必须提供成本总和",
    ):
        rebuild_summary(
            summary=summary,
            updates={
                "estimated_cost_result_count": 1,
                "total_estimated_cost_usd": None,
            },
        )


def test_experiment_report_accepts_unique_results_and_summaries(
) -> None:
    """合法报告应保留三条路线、三个案例和三个单路线结果。"""

    report = make_report()

    assert report.schema_version == (
        MULTIMODAL_TOOL_SELECTION_REPORT_VERSION
    )
    assert report.case_count == 3
    assert report.route_result_count == 3
    assert len(report.results) == 3
    assert len(report.route_summaries) == 3


def test_experiment_report_rejects_result_count_mismatch(
) -> None:
    """route_result_count必须等于详细结果数量。"""

    report = make_report()

    with pytest.raises(
        ValidationError,
        match="route_result_count",
    ):
        rebuild_report(
            report=report,
            updates={
                "route_result_count": 4,
            },
        )


def test_experiment_report_rejects_duplicate_case_route_pair(
) -> None:
    """同一案例的同一路线不能在报告中重复出现。"""

    report = make_report()
    duplicate_result = (
        report.results[0]
    )

    with pytest.raises(
        ValidationError,
        match="case_id和route",
    ):
        rebuild_report(
            report=report,
            updates={
                "route_result_count": 4,
                "results": (
                    *report.results,
                    duplicate_result,
                ),
            },
        )


def test_experiment_report_rejects_wrong_case_count(
) -> None:
    """case_count必须来自详细结果中的不同案例编号。"""

    report = make_report()

    with pytest.raises(
        ValidationError,
        match="case_count",
    ):
        rebuild_report(
            report=report,
            updates={
                "case_count": 4,
            },
        )


def test_experiment_report_rejects_route_set_not_matching_results(
) -> None:
    """报告路线集合必须与详细结果中的实际路线完全相同。"""

    report = make_report()

    with pytest.raises(
        ValidationError,
        match="实际路线集合",
    ):
        rebuild_report(
            report=report,
            updates={
                "routes": (
                    "ocr_rule",
                    "vision_model",
                ),
            },
        )


def test_experiment_report_rejects_duplicate_routes(
) -> None:
    """顶层routes字段不能重复相同路线。"""

    report = make_report()

    with pytest.raises(
        ValidationError,
        match="routes不能包含重复路线",
    ):
        rebuild_report(
            report=report,
            updates={
                "routes": (
                    "ocr_rule",
                    "ocr_rule",
                    "vision_model",
                ),
            },
        )


def test_experiment_report_rejects_duplicate_route_summaries(
) -> None:
    """同一条路线不能拥有两份互相冲突的汇总。"""

    report = make_report()

    with pytest.raises(
        ValidationError,
        match="重复路线",
    ):
        rebuild_report(
            report=report,
            updates={
                "route_summaries": (
                    report.route_summaries[0],
                    report.route_summaries[0],
                    report.route_summaries[2],
                ),
            },
        )


def test_experiment_report_requires_summary_for_every_route(
) -> None:
    """每条实际路线必须恰好对应一个路线汇总。"""

    report = make_report()

    with pytest.raises(
        ValidationError,
        match="每条实际路线",
    ):
        rebuild_report(
            report=report,
            updates={
                "route_summaries": (
                    report.route_summaries[0],
                    report.route_summaries[1],
                ),
            },
        )


def test_experiment_report_rejects_summary_evaluated_count_mismatch(
) -> None:
    """路线汇总的案例数必须等于该路线详细结果数。"""

    report = make_report()
    ocr_summary_data = (
        report
        .route_summaries[0]
        .model_dump()
    )
    ocr_summary_data.update(
        {
            "evaluated_case_count": 2,
            "completed_count": 2,
            "passed_count": 2,
            "route_appropriate_count": 2,
            "outcome_correct_count": 2,
            "safety_passed_count": 2,
            "content_accuracy_sample_count": 2,
        }
    )
    inconsistent_summary = (
        MultimodalToolSelectionRouteSummary
        .model_validate(ocr_summary_data)
    )

    with pytest.raises(
        ValidationError,
        match="evaluated_case_count",
    ):
        rebuild_report(
            report=report,
            updates={
                "route_summaries": (
                    inconsistent_summary,
                    *report.route_summaries[1:],
                ),
            },
        )
