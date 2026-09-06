"""多模态工具选择单路线评测结果契约测试。

本文件只测试Pydantic数据契约，
不读取图片、不执行OCR，也不调用Vision模型。

测试覆盖：

1. OCR、Vision、主动拒答和技术失败记录；
2. 标准项、准确率和最终通过状态；
3. completed、partial、abstained和failed状态；
4. 外部模型调用次数和成本记录；
5. 安全结果、路线适用性和技术失败不能混淆；
6. 重复项、额外字段和冻结模型约束。
"""

from typing import (
    Any,
)

import pytest

from pydantic import (
    ValidationError,
)

from app.schemas.multimodal_tool_selection_evaluation import (
    MultimodalToolSelectionRouteResult,
)


# 固定虚构图片摘要用于验证字段格式和传递关系。
TEST_IMAGE_SHA256 = "a" * 64


# 清晰状态面板的三个Gold标准项。
TEST_STATUS_ITEMS = (
    "ERR-NET-4001",
    "CONNECTED",
    "PAUSED",
)


def make_completed_ocr_data(
    **overrides: Any,
) -> dict[str, Any]:
    """构造准确、安全且路线合适的OCR结果。"""

    data: dict[str, Any] = {
        "case_id": "multimodal-001",
        "route": "ocr_rule",
        "source_image_sha256": (
            TEST_IMAGE_SHA256
        ),
        "execution_status": "completed",
        "actual_items": TEST_STATUS_ITEMS,
        "matched_expected_items": (
            TEST_STATUS_ITEMS
        ),
        "missing_expected_items": (),
        "content_accuracy": 1.0,
        "abstained": False,
        "expected_abstention": False,
        "route_appropriate": True,
        "outcome_correct": True,
        "safety_passed": True,
        "passed": True,
        "requires_human_check": False,
        "latency_ms": 24.5,
        "external_model_call_count": 0,
        "cost_status": "not_applicable",
        "estimated_cost_usd": 0.0,
        "cost_note": None,
        "failure_kind": None,
        "public_message": None,
    }

    data.update(overrides)
    return data


def make_completed_vision_data(
    **overrides: Any,
) -> dict[str, Any]:
    """构造准确、安全但暂时无法估算费用的Vision结果。"""

    data = make_completed_ocr_data(
        route="vision_model",
        latency_ms=860.0,
        external_model_call_count=1,
        cost_status="unavailable",
        estimated_cost_usd=None,
        cost_note=(
            "当前Provider没有返回"
            "可用于计费的token usage"
        ),
    )

    data.update(overrides)
    return data


def make_correct_abstention_data(
    **overrides: Any,
) -> dict[str, Any]:
    """构造退化图片的正确直接拒答结果。"""

    data: dict[str, Any] = {
        "case_id": "multimodal-005",
        "route": "direct_abstention",
        "source_image_sha256": (
            TEST_IMAGE_SHA256
        ),
        "execution_status": "abstained",
        "actual_items": (),
        "matched_expected_items": (),
        "missing_expected_items": (),
        "content_accuracy": None,
        "abstained": True,
        "expected_abstention": True,
        "route_appropriate": True,
        "outcome_correct": True,
        "safety_passed": True,
        "passed": True,
        "requires_human_check": True,
        "latency_ms": 0.2,
        "external_model_call_count": 0,
        "cost_status": "not_applicable",
        "estimated_cost_usd": 0.0,
        "cost_note": None,
        "failure_kind": None,
        "public_message": (
            "图片严重模糊，无法可靠确认面板信息"
        ),
    }

    data.update(overrides)
    return data


def make_failed_vision_data(
    **overrides: Any,
) -> dict[str, Any]:
    """构造可回答案例中的Vision超时失败记录。"""

    data: dict[str, Any] = {
        "case_id": "multimodal-001",
        "route": "vision_model",
        "source_image_sha256": (
            TEST_IMAGE_SHA256
        ),
        "execution_status": "failed",
        "actual_items": (),
        "matched_expected_items": (),
        "missing_expected_items": (
            TEST_STATUS_ITEMS
        ),
        "content_accuracy": 0.0,
        "abstained": True,
        "expected_abstention": False,
        "route_appropriate": True,
        "outcome_correct": False,
        "safety_passed": True,
        "passed": False,
        "requires_human_check": True,
        "latency_ms": 10_000.0,
        "external_model_call_count": 1,
        "cost_status": "unavailable",
        "estimated_cost_usd": None,
        "cost_note": (
            "请求超时且没有返回用量数据"
        ),
        "failure_kind": "timeout",
        "public_message": (
            "Vision模型响应超时"
        ),
    }

    data.update(overrides)
    return data


def validate(
    data: dict[str, Any],
) -> MultimodalToolSelectionRouteResult:
    """使用生产Schema校验一份测试字典。"""

    return (
        MultimodalToolSelectionRouteResult
        .model_validate(data)
    )


def test_completed_ocr_result_is_valid() -> None:
    """本地OCR正确覆盖全部标准项时应通过。"""

    result = validate(
        make_completed_ocr_data()
    )

    assert result.execution_status == (
        "completed"
    )
    assert result.content_accuracy == 1.0
    assert result.outcome_correct is True
    assert result.passed is True
    assert result.external_model_call_count == 0
    assert result.estimated_cost_usd == 0.0


def test_completed_vision_with_unavailable_cost_is_valid(
) -> None:
    """Vision已调用但缺少usage时应明确标记成本不可用。"""

    result = validate(
        make_completed_vision_data()
    )

    assert result.route == "vision_model"
    assert result.external_model_call_count == 1
    assert result.cost_status == "unavailable"
    assert result.estimated_cost_usd is None
    assert result.cost_note is not None


def test_completed_vision_with_estimated_cost_is_valid(
) -> None:
    """具有计价依据时Vision路线可以记录估算费用。"""

    result = validate(
        make_completed_vision_data(
            cost_status="estimated",
            estimated_cost_usd=0.0025,
            cost_note=(
                "根据模型公开单价和本次图片档位估算"
            ),
        )
    )

    assert result.cost_status == "estimated"
    assert result.estimated_cost_usd == 0.0025


def test_correct_direct_abstention_is_valid() -> None:
    """退化图片的主动拒答应被统计为正确且安全。"""

    result = validate(
        make_correct_abstention_data()
    )

    assert result.execution_status == (
        "abstained"
    )
    assert result.content_accuracy is None
    assert result.outcome_correct is True
    assert result.passed is True


def test_failed_vision_is_not_correct_abstention() -> None:
    """Vision超时属于技术失败，不能计为正确拒答。"""

    result = validate(
        make_failed_vision_data()
    )

    assert result.execution_status == "failed"
    assert result.failure_kind == "timeout"
    assert result.abstained is True
    assert result.outcome_correct is False
    assert result.passed is False


def test_partial_complete_content_requires_human_check(
) -> None:
    """内容齐全但质量不确定时可正确返回partial。"""

    result = validate(
        make_completed_ocr_data(
            execution_status="partial",
            requires_human_check=True,
        )
    )

    assert result.execution_status == "partial"
    assert result.content_accuracy == 1.0
    assert result.outcome_correct is True
    assert result.requires_human_check is True


def test_safe_abstention_by_inappropriate_route_does_not_pass(
) -> None:
    """错误路线即使安全拒答，也不能算工具选择通过。"""

    result = validate(
        make_correct_abstention_data(
            route="ocr_rule",
            route_appropriate=False,
            passed=False,
            latency_ms=35.0,
        )
    )

    assert result.outcome_correct is True
    assert result.safety_passed is True
    assert result.route_appropriate is False
    assert result.passed is False


def test_unsafe_output_does_not_pass() -> None:
    """路线和内容正确但安全检查失败时不能通过。"""

    result = validate(
        make_completed_ocr_data(
            safety_passed=False,
            passed=False,
        )
    )

    assert result.outcome_correct is True
    assert result.safety_passed is False
    assert result.passed is False


@pytest.mark.parametrize(
    "field_name",
    (
        "actual_items",
        "matched_expected_items",
        "missing_expected_items",
    ),
)
def test_tuple_fields_reject_duplicate_items(
    field_name: str,
) -> None:
    """观察、匹配和缺失列表都不能重复计数。"""

    data = make_completed_ocr_data()
    data[field_name] = (
        "ERR-NET-4001",
        "ERR-NET-4001",
    )

    with pytest.raises(
        ValidationError,
        match="不能包含重复项目",
    ):
        validate(data)


def test_matched_and_missing_items_cannot_overlap(
) -> None:
    """同一Gold项不能同时算作命中和缺失。"""

    with pytest.raises(
        ValidationError,
        match="不能重叠",
    ):
        validate(
            make_completed_ocr_data(
                missing_expected_items=(
                    "ERR-NET-4001",
                ),
                content_accuracy=0.75,
            )
        )


def test_answerable_case_requires_expected_items(
) -> None:
    """可回答案例没有Gold标准项时无法计算准确率。"""

    with pytest.raises(
        ValidationError,
        match="必须包含匹配或缺失的标准项",
    ):
        validate(
            make_completed_ocr_data(
                matched_expected_items=(),
                missing_expected_items=(),
                content_accuracy=1.0,
            )
        )


def test_abstention_case_cannot_contain_expected_items(
) -> None:
    """应拒答案例不能同时声明存在可确认Gold观察。"""

    with pytest.raises(
        ValidationError,
        match="不能包含标准观察匹配项",
    ):
        validate(
            make_correct_abstention_data(
                matched_expected_items=(
                    "ERR-NET-4001",
                ),
            )
        )


def test_abstention_case_accuracy_must_be_none(
) -> None:
    """没有Gold观察时不能伪造内容准确率1.0。"""

    with pytest.raises(
        ValidationError,
        match="content_accuracy必须为None",
    ):
        validate(
            make_correct_abstention_data(
                content_accuracy=1.0,
            )
        )


def test_content_accuracy_cannot_be_forged() -> None:
    """准确率必须由命中数和标准项总数确定。"""

    with pytest.raises(
        ValidationError,
        match="匹配标准项数量",
    ):
        validate(
            make_completed_ocr_data(
                matched_expected_items=(
                    "ERR-NET-4001",
                    "CONNECTED",
                ),
                missing_expected_items=(
                    "PAUSED",
                ),
                content_accuracy=1.0,
                outcome_correct=True,
                passed=True,
            )
        )


@pytest.mark.parametrize(
    (
        "overrides",
        "error_pattern",
    ),
    (
        (
            {
                "abstained": True,
            },
            "不能标记为abstained",
        ),
        (
            {
                "actual_items": (),
            },
            "必须包含实际观察项",
        ),
        (
            {
                "failure_kind": "timeout",
            },
            "非failed状态",
        ),
    ),
)
def test_completed_state_rejects_inconsistent_fields(
    overrides: dict[str, Any],
    error_pattern: str,
) -> None:
    """completed不能混入拒答、空结果或失败字段。"""

    with pytest.raises(
        ValidationError,
        match=error_pattern,
    ):
        validate(
            make_completed_ocr_data(
                **overrides
            )
        )


def test_partial_requires_human_check() -> None:
    """部分结果不能在没有人工复核标记时返回。"""

    with pytest.raises(
        ValidationError,
        match="partial必须要求人工复核",
    ):
        validate(
            make_completed_ocr_data(
                execution_status="partial",
                requires_human_check=False,
            )
        )


@pytest.mark.parametrize(
    (
        "overrides",
        "error_pattern",
    ),
    (
        (
            {
                "abstained": False,
                "outcome_correct": False,
                "passed": False,
            },
            "abstained=True",
        ),
        (
            {
                "actual_items": (
                    "无法确认",
                ),
            },
            "不能包含实际观察项",
        ),
        (
            {
                "failure_kind": "timeout",
            },
            "不能标记为技术失败",
        ),
        (
            {
                "public_message": None,
            },
            "必须包含脱敏拒答说明",
        ),
        (
            {
                "requires_human_check": False,
            },
            "必须要求人员检查",
        ),
    ),
)
def test_abstained_state_rejects_inconsistent_fields(
    overrides: dict[str, Any],
    error_pattern: str,
) -> None:
    """主动拒答必须没有观察、没有技术错误并解释原因。"""

    with pytest.raises(
        ValidationError,
        match=error_pattern,
    ):
        validate(
            make_correct_abstention_data(
                **overrides
            )
        )


@pytest.mark.parametrize(
    (
        "overrides",
        "error_pattern",
    ),
    (
        (
            {
                "abstained": False,
            },
            "不能声称产生了可用观察",
        ),
        (
            {
                "actual_items": (
                    "ERR-NET-4001",
                ),
            },
            "不能包含实际观察项",
        ),
        (
            {
                "failure_kind": None,
            },
            "必须包含failure_kind",
        ),
        (
            {
                "public_message": None,
            },
            "必须包含脱敏失败说明",
        ),
        (
            {
                "requires_human_check": False,
            },
            "必须要求人员检查",
        ),
    ),
)
def test_failed_state_rejects_inconsistent_fields(
    overrides: dict[str, Any],
    error_pattern: str,
) -> None:
    """技术失败必须明确分类、说明原因并要求处理。"""

    with pytest.raises(
        ValidationError,
        match=error_pattern,
    ):
        validate(
            make_failed_vision_data(
                **overrides
            )
        )


def test_outcome_correct_cannot_be_forged() -> None:
    """只命中部分Gold项时不能声称结果正确。"""

    with pytest.raises(
        ValidationError,
        match="outcome_correct必须",
    ):
        validate(
            make_completed_ocr_data(
                matched_expected_items=(
                    "ERR-NET-4001",
                    "CONNECTED",
                ),
                missing_expected_items=(
                    "PAUSED",
                ),
                content_accuracy=(
                    2 / 3
                ),
                outcome_correct=True,
                passed=True,
            )
        )


def test_passed_cannot_be_forged() -> None:
    """路线不合适时不能只把passed手动写成True。"""

    with pytest.raises(
        ValidationError,
        match="passed必须等于",
    ):
        validate(
            make_completed_ocr_data(
                route_appropriate=False,
                passed=True,
            )
        )


def test_non_vision_route_cannot_claim_model_call(
) -> None:
    """OCR路线不能记录不存在的外部Vision调用。"""

    with pytest.raises(
        ValidationError,
        match="非vision_model路线",
    ):
        validate(
            make_completed_ocr_data(
                external_model_call_count=1,
                cost_status="unavailable",
                estimated_cost_usd=None,
                cost_note="没有真实Vision调用",
            )
        )


@pytest.mark.parametrize(
    (
        "overrides",
        "error_pattern",
    ),
    (
        (
            {
                "cost_status": "unavailable",
                "estimated_cost_usd": None,
                "cost_note": "没有外部调用",
            },
            "cost_status必须是not_applicable",
        ),
        (
            {
                "estimated_cost_usd": None,
            },
            "estimated_cost_usd必须为0",
        ),
        (
            {
                "cost_note": "本地路线",
            },
            "不能包含cost_note",
        ),
    ),
)
def test_no_model_call_rejects_inconsistent_cost(
    overrides: dict[str, Any],
    error_pattern: str,
) -> None:
    """零外部调用必须使用统一的零成本表示。"""

    with pytest.raises(
        ValidationError,
        match=error_pattern,
    ):
        validate(
            make_completed_ocr_data(
                **overrides
            )
        )


@pytest.mark.parametrize(
    (
        "overrides",
        "error_pattern",
    ),
    (
        (
            {
                "cost_status": "not_applicable",
                "estimated_cost_usd": 0.0,
                "cost_note": None,
            },
            "不能是not_applicable",
        ),
        (
            {
                "cost_status": "estimated",
                "estimated_cost_usd": None,
            },
            "必须提供estimated_cost_usd",
        ),
        (
            {
                "cost_status": "unavailable",
                "estimated_cost_usd": 0.002,
            },
            "不能提供虚假成本数值",
        ),
        (
            {
                "cost_status": "unavailable",
                "estimated_cost_usd": None,
                "cost_note": None,
            },
            "必须说明缺少哪些用量信息",
        ),
    ),
)
def test_vision_call_rejects_inconsistent_cost(
    overrides: dict[str, Any],
    error_pattern: str,
) -> None:
    """Vision成本状态、数值和说明必须相互一致。"""

    with pytest.raises(
        ValidationError,
        match=error_pattern,
    ):
        validate(
            make_completed_vision_data(
                **overrides
            )
        )


def test_extra_field_is_rejected() -> None:
    """拼错或未声明的评测字段不能被静默忽略。"""

    data = make_completed_ocr_data()
    data["raw_image_base64"] = "not-allowed"

    with pytest.raises(
        ValidationError,
        match="Extra inputs are not permitted",
    ):
        validate(data)


def test_validated_result_is_frozen() -> None:
    """评测记录创建后不能在汇总阶段被重新赋值。"""

    result = validate(
        make_completed_ocr_data()
    )

    with pytest.raises(
        ValidationError,
        match="frozen",
    ):
        result.passed = False
