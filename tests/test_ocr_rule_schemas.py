"""OCR确定性规则解析结果数据契约的单元测试。"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError
import pytest

from app.schemas.ocr_rule import (
    OCR_RULE_VERSION,
    OcrRuleParseResult,
)


# 两个固定摘要分别代表图片和OCR文字。
TEST_IMAGE_SHA256 = "a" * 64
TEST_TEXT_SHA256 = "b" * 64


def make_completed_status_data(
    **overrides: Any,
) -> dict[str, Any]:
    """构造字段完整且OCR质量合格的状态面板结果。"""

    data: dict[str, Any] = {
        "status": "completed",
        "panel_type": "robot_status",
        "ocr_status": "completed",
        "source_image_sha256": TEST_IMAGE_SHA256,
        "source_text_sha256": TEST_TEXT_SHA256,
        "source": "ocr_rule",
        "rules_version": OCR_RULE_VERSION,
        "fault_code": "ERR-NET-4001",
        "network_state": "CONNECTED",
        "task_state": "PAUSED",
        "battery_percent": None,
        "speed_mps": None,
        "temperature_celsius": None,
        "matched_fields": (
            "fault_code",
            "network_state",
            "task_state",
        ),
        "missing_fields": (),
        "failure_reasons": (),
        "requires_human_check": False,
    }

    data.update(overrides)
    return data


def make_completed_telemetry_data(
    **overrides: Any,
) -> dict[str, Any]:
    """构造字段完整且OCR质量合格的遥测面板结果。"""

    data: dict[str, Any] = {
        "status": "completed",
        "panel_type": "robot_telemetry",
        "ocr_status": "completed",
        "source_image_sha256": TEST_IMAGE_SHA256,
        "source_text_sha256": TEST_TEXT_SHA256,
        "battery_percent": 42.5,
        "speed_mps": 0.0,
        "temperature_celsius": 36.0,
        "matched_fields": (
            "battery_percent",
            "speed_mps",
            "temperature_celsius",
        ),
        "missing_fields": (),
        "failure_reasons": (),
        "requires_human_check": False,
    }

    data.update(overrides)
    return data


def make_partial_status_data(
    **overrides: Any,
) -> dict[str, Any]:
    """构造缺少任务状态的部分状态面板结果。"""

    data = make_completed_status_data(
        status="partial",
        task_state=None,
        matched_fields=(
            "fault_code",
            "network_state",
        ),
        missing_fields=(
            "task_state",
        ),
        failure_reasons=(
            "没有匹配到TASK字段的受支持状态",
        ),
        requires_human_check=True,
    )

    data.update(overrides)
    return data


def make_unsupported_data(
    **overrides: Any,
) -> dict[str, Any]:
    """构造OCR有文字但规则不支持其布局的结果。"""

    data: dict[str, Any] = {
        "status": "unsupported",
        "panel_type": "unknown",
        "ocr_status": "completed",
        "source_image_sha256": TEST_IMAGE_SHA256,
        "source_text_sha256": TEST_TEXT_SHA256,
        "matched_fields": (),
        "missing_fields": (),
        "failure_reasons": (
            "没有识别到受支持的面板标签组合",
        ),
        "requires_human_check": True,
    }

    data.update(overrides)
    return data


def make_empty_data(
    **overrides: Any,
) -> dict[str, Any]:
    """构造上游OCR没有识别到文字的结果。"""

    data: dict[str, Any] = {
        "status": "empty",
        "panel_type": "unknown",
        "ocr_status": "empty",
        "source_image_sha256": TEST_IMAGE_SHA256,
        "source_text_sha256": TEST_TEXT_SHA256,
        "matched_fields": (),
        "missing_fields": (),
        "failure_reasons": (
            "上游OCR没有识别到可解析文字",
        ),
        "requires_human_check": True,
    }

    data.update(overrides)
    return data


def test_completed_status_panel_is_valid() -> None:
    """完整状态面板应返回completed和三个状态字段。"""

    result = OcrRuleParseResult.model_validate(
        make_completed_status_data()
    )

    assert result.status == "completed"
    assert result.panel_type == "robot_status"
    assert result.fault_code == "ERR-NET-4001"
    assert result.network_state == "CONNECTED"
    assert result.task_state == "PAUSED"
    assert result.matched_fields == (
        "fault_code",
        "network_state",
        "task_state",
    )
    assert result.requires_human_check is False


def test_completed_telemetry_panel_is_valid() -> None:
    """完整遥测面板应保留三个带单位语义的数值。"""

    result = OcrRuleParseResult.model_validate(
        make_completed_telemetry_data()
    )

    assert result.status == "completed"
    assert result.panel_type == "robot_telemetry"
    assert result.battery_percent == 42.5
    assert result.speed_mps == 0.0
    assert result.temperature_celsius == 36.0


def test_partial_status_panel_is_valid() -> None:
    """缺少任务状态时应返回partial并列出缺失字段。"""

    result = OcrRuleParseResult.model_validate(
        make_partial_status_data()
    )

    assert result.status == "partial"
    assert result.task_state is None
    assert result.missing_fields == (
        "task_state",
    )
    assert result.requires_human_check is True


def test_low_confidence_complete_fields_can_be_partial() -> None:
    """字段齐全但OCR低置信度时仍应进入人工复核。"""

    result = OcrRuleParseResult.model_validate(
        make_completed_status_data(
            status="partial",
            ocr_status="low_confidence",
            failure_reasons=(
                "上游OCR平均置信度低于门槛",
            ),
            requires_human_check=True,
        )
    )

    assert result.status == "partial"
    assert result.missing_fields == ()
    assert result.requires_human_check is True


def test_unsupported_result_is_valid() -> None:
    """有OCR文字但未知布局时应返回unsupported。"""

    result = OcrRuleParseResult.model_validate(
        make_unsupported_data()
    )

    assert result.status == "unsupported"
    assert result.panel_type == "unknown"
    assert result.matched_fields == ()


def test_empty_result_is_valid() -> None:
    """OCR空结果应稳定转换成规则empty结果。"""

    result = OcrRuleParseResult.model_validate(
        make_empty_data()
    )

    assert result.status == "empty"
    assert result.ocr_status == "empty"
    assert result.requires_human_check is True


def test_result_round_trips_through_json() -> None:
    """规则结果应能稳定序列化并重新校验。"""

    original = OcrRuleParseResult.model_validate(
        make_completed_status_data()
    )

    restored = OcrRuleParseResult.model_validate_json(
        original.model_dump_json()
    )

    assert restored == original


def test_result_is_frozen() -> None:
    """规则结果创建后不能被调用方原地修改。"""

    result = OcrRuleParseResult.model_validate(
        make_completed_status_data()
    )

    with pytest.raises(
        ValidationError,
        match="Instance is frozen",
    ):
        result.task_state = "RUNNING"


def test_result_rejects_extra_field() -> None:
    """未声明字段不能绕过规则结果契约。"""

    with pytest.raises(
        ValidationError,
        match="Extra inputs are not permitted",
    ):
        OcrRuleParseResult.model_validate(
            make_completed_status_data(
                diagnosis="network failure",
            )
        )


@pytest.mark.parametrize(
    "fault_code",
    [
        "err-net-4001",
        "ERR_NET_4001",
        "ERR-NET-401",
        "CONNECTED",
    ],
)
def test_result_rejects_invalid_fault_code(
    fault_code: str,
) -> None:
    """故障码必须符合当前标准化格式。"""

    with pytest.raises(ValidationError):
        OcrRuleParseResult.model_validate(
            make_completed_status_data(
                fault_code=fault_code,
            )
        )


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        (
            "network_state",
            "ONLINE",
        ),
        (
            "task_state",
            "UNKNOWN",
        ),
    ],
)
def test_result_rejects_unsupported_enum_values(
    field_name: str,
    invalid_value: str,
) -> None:
    """状态字段只能使用当前规则明确支持的枚举。"""

    with pytest.raises(ValidationError):
        OcrRuleParseResult.model_validate(
            make_completed_status_data(
                **{
                    field_name: invalid_value,
                }
            )
        )


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        (
            "battery_percent",
            -0.1,
        ),
        (
            "battery_percent",
            100.1,
        ),
        (
            "speed_mps",
            -20.1,
        ),
        (
            "speed_mps",
            20.1,
        ),
        (
            "temperature_celsius",
            -100.1,
        ),
        (
            "temperature_celsius",
            200.1,
        ),
        (
            "temperature_celsius",
            float("nan"),
        ),
    ],
)
def test_result_rejects_out_of_range_numeric_values(
    field_name: str,
    invalid_value: float,
) -> None:
    """数值字段必须满足有限范围并拒绝NaN。"""

    with pytest.raises(ValidationError):
        OcrRuleParseResult.model_validate(
            make_completed_telemetry_data(
                **{
                    field_name: invalid_value,
                }
            )
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "matched_fields",
        "missing_fields",
        "failure_reasons",
    ],
)
def test_result_rejects_duplicate_tuple_items(
    field_name: str,
) -> None:
    """匹配、缺失和失败说明均不能重复计数。"""

    if field_name == "failure_reasons":
        value: tuple[str, ...] = (
            "重复原因",
            "重复原因",
        )
    else:
        value = (
            "task_state",
            "task_state",
        )

    with pytest.raises(
        ValidationError,
        match="不能包含重复项目",
    ):
        OcrRuleParseResult.model_validate(
            make_partial_status_data(
                **{
                    field_name: value,
                }
            )
        )


def test_matched_fields_must_equal_actual_values() -> None:
    """已有字段不能从matched_fields中漏掉。"""

    with pytest.raises(
        ValidationError,
        match=(
            "matched_fields必须与"
            "实际非空解析字段一致"
        ),
    ):
        OcrRuleParseResult.model_validate(
            make_completed_status_data(
                matched_fields=(
                    "fault_code",
                    "network_state",
                ),
            )
        )


def test_matched_fields_must_use_canonical_order() -> None:
    """matched_fields顺序必须与固定字段顺序一致。"""

    with pytest.raises(
        ValidationError,
        match=(
            "matched_fields必须与"
            "实际非空解析字段一致"
        ),
    ):
        OcrRuleParseResult.model_validate(
            make_completed_status_data(
                matched_fields=(
                    "network_state",
                    "fault_code",
                    "task_state",
                ),
            )
        )


def test_matched_and_missing_fields_cannot_overlap() -> None:
    """同一字段不能同时声称成功和缺失。"""

    with pytest.raises(
        ValidationError,
        match=(
            "matched_fields和"
            "missing_fields不能重叠"
        ),
    ):
        OcrRuleParseResult.model_validate(
            make_completed_status_data(
                missing_fields=(
                    "task_state",
                ),
            )
        )


def test_status_panel_rejects_telemetry_fields() -> None:
    """状态面板不能混入电量等遥测字段。"""

    with pytest.raises(
        ValidationError,
        match=(
            "robot_status不能包含"
            "遥测面板字段"
        ),
    ):
        OcrRuleParseResult.model_validate(
            make_completed_status_data(
                battery_percent=42.5,
                matched_fields=(
                    "fault_code",
                    "network_state",
                    "task_state",
                    "battery_percent",
                ),
            )
        )


def test_telemetry_panel_rejects_status_fields() -> None:
    """遥测面板不能混入故障码等状态字段。"""

    with pytest.raises(
        ValidationError,
        match=(
            "robot_telemetry不能包含"
            "状态面板字段"
        ),
    ):
        OcrRuleParseResult.model_validate(
            make_completed_telemetry_data(
                fault_code="ERR-NET-4001",
                matched_fields=(
                    "fault_code",
                    "battery_percent",
                    "speed_mps",
                    "temperature_celsius",
                ),
            )
        )


def test_unknown_panel_rejects_matched_fields() -> None:
    """未知布局中的偶然文字不能被解释成正式字段。"""

    with pytest.raises(
        ValidationError,
        match=(
            "unknown面板不能包含"
            "已匹配字段"
        ),
    ):
        OcrRuleParseResult.model_validate(
            make_unsupported_data(
                fault_code="ERR-NET-4001",
                matched_fields=(
                    "fault_code",
                ),
            )
        )


def test_missing_fields_must_equal_absent_required_fields() -> None:
    """状态面板缺失字段必须根据真实None值机械计算。"""

    with pytest.raises(
        ValidationError,
        match=(
            "missing_fields必须与"
            "面板实际缺失字段一致"
        ),
    ):
        OcrRuleParseResult.model_validate(
            make_partial_status_data(
                # task_state实际为None，
                # 这里故意不把它列入missing_fields。
                #
                # 使用空元组可以避免先触发
                # “matched与missing重叠”的另一条规则。
                missing_fields=(),
            )
        )


def test_empty_ocr_requires_empty_rule_status() -> None:
    """上游没有文字时规则不能声称unsupported。"""

    with pytest.raises(
        ValidationError,
        match=(
            "上游OCR为空时"
            "规则状态必须是empty"
        ),
    ):
        OcrRuleParseResult.model_validate(
            make_unsupported_data(
                ocr_status="empty",
            )
        )


def test_empty_rule_status_requires_empty_ocr() -> None:
    """OCR有文字时规则不能返回empty。"""

    with pytest.raises(
        ValidationError,
        match=(
            "规则状态为empty时"
            "上游OCR也必须为空"
        ),
    ):
        OcrRuleParseResult.model_validate(
            make_empty_data(
                ocr_status="completed",
            )
        )


def test_completed_rejects_low_confidence_ocr() -> None:
    """低置信度OCR不能被升级成无需复核的completed。"""

    with pytest.raises(
        ValidationError,
        match=(
            "completed要求上游OCR"
            "状态也是completed"
        ),
    ):
        OcrRuleParseResult.model_validate(
            make_completed_status_data(
                ocr_status="low_confidence",
            )
        )


def test_completed_rejects_failure_reasons() -> None:
    """完成结果不能同时声明存在解析失败。"""

    with pytest.raises(
        ValidationError,
        match=(
            "completed不能包含"
            "failure_reasons"
        ),
    ):
        OcrRuleParseResult.model_validate(
            make_completed_status_data(
                failure_reasons=(
                    "不应存在的失败原因",
                ),
            )
        )


def test_completed_rejects_human_check() -> None:
    """OCR质量完成结果不能因规则质量要求人工复核。"""

    with pytest.raises(
        ValidationError,
        match=(
            "completed不能要求"
            "OCR质量人工复核"
        ),
    ):
        OcrRuleParseResult.model_validate(
            make_completed_status_data(
                requires_human_check=True,
            )
        )


def test_partial_requires_at_least_one_matched_field() -> None:
    """完全没有匹配字段的已知面板不能称为部分结果。"""

    with pytest.raises(
        ValidationError,
        match=(
            "partial必须至少包含"
            "一个已匹配字段"
        ),
    ):
        OcrRuleParseResult.model_validate(
            make_partial_status_data(
                fault_code=None,
                network_state=None,
                task_state=None,
                matched_fields=(),
                missing_fields=(
                    "fault_code",
                    "network_state",
                    "task_state",
                ),
            )
        )


def test_partial_requires_missing_field_or_failure_reason() -> None:
    """字段完整的partial必须说明为何不能完成。"""

    with pytest.raises(
        ValidationError,
        match=(
            "partial必须说明缺失字段"
            "或失败原因"
        ),
    ):
        OcrRuleParseResult.model_validate(
            make_completed_status_data(
                status="partial",
                requires_human_check=True,
            )
        )


def test_partial_requires_human_check() -> None:
    """部分结果必须提醒调用方检查原图。"""

    with pytest.raises(
        ValidationError,
        match=(
            "partial必须要求人工复核"
        ),
    ):
        OcrRuleParseResult.model_validate(
            make_partial_status_data(
                requires_human_check=False,
            )
        )


def test_unsupported_requires_unknown_panel() -> None:
    """已识别面板类型时不能使用unsupported状态。"""

    with pytest.raises(
        ValidationError,
        match=(
            "unsupported必须使用"
            "unknown面板类型"
        ),
    ):
        OcrRuleParseResult.model_validate(
            make_partial_status_data(
                status="unsupported",
                failure_reasons=(
                    "布局不受支持",
                ),
            )
        )


def test_unsupported_requires_failure_reason() -> None:
    """不支持结果必须解释当前规则缺少什么能力。"""

    with pytest.raises(
        ValidationError,
        match=(
            "unsupported必须说明"
            "不支持原因"
        ),
    ):
        OcrRuleParseResult.model_validate(
            make_unsupported_data(
                failure_reasons=(),
            )
        )


def test_unsupported_requires_human_check() -> None:
    """未知面板布局不能在无人复核时作为可靠结果。"""

    with pytest.raises(
        ValidationError,
        match=(
            "unsupported必须要求"
            "人工复核"
        ),
    ):
        OcrRuleParseResult.model_validate(
            make_unsupported_data(
                requires_human_check=False,
            )
        )


def test_empty_requires_failure_reason() -> None:
    """空结果也必须说明没有产生字段的原因。"""

    with pytest.raises(
        ValidationError,
        match=(
            "empty必须说明"
            "没有解析结果的原因"
        ),
    ):
        OcrRuleParseResult.model_validate(
            make_empty_data(
                failure_reasons=(),
            )
        )


def test_empty_requires_human_check() -> None:
    """OCR空结果必须要求补图或人工查看原图。"""

    with pytest.raises(
        ValidationError,
        match=(
            "empty必须要求人工复核"
        ),
    ):
        OcrRuleParseResult.model_validate(
            make_empty_data(
                requires_human_check=False,
            )
        )


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        (
            "source",
            "vision_model",
        ),
        (
            "rules_version",
            "robot-panel-rules-v2",
        ),
        (
            "source_image_sha256",
            "a" * 63,
        ),
        (
            "source_text_sha256",
            "B" * 64,
        ),
    ],
)
def test_result_rejects_invalid_tracking_fields(
    field_name: str,
    invalid_value: str,
) -> None:
    """来源、规则版本和两个摘要不能由调用方伪造格式。"""

    with pytest.raises(ValidationError):
        OcrRuleParseResult.model_validate(
            make_completed_status_data(
                **{
                    field_name: invalid_value,
                }
            )
        )
