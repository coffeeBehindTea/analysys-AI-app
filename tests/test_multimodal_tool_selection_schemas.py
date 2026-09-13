"""多模态工具选择案例数据契约的单元测试。

本文件只测试案例定义和字段关系，
不会读取图片、执行 OCR 或调用 Vision API。
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError
import pytest

from app.schemas.multimodal_tool_selection import (
    MultimodalToolSelectionCase,
)


# 使用固定摘要，使测试只关注数据契约，
# 不依赖任何真实图片文件。
TEST_IMAGE_SHA256 = "a" * 64


# 三条路线的固定全集。
# 多个案例复用它，避免不同测试无意使用不同对比范围。
ALL_COMPARISON_ROUTES = (
    "ocr_rule",
    "vision_model",
    "direct_abstention",
)


def make_text_panel_data(
    **overrides: Any,
) -> dict[str, Any]:
    """返回一个合法的清晰文字面板案例字典。"""

    data: dict[str, Any] = {
        "case_id": "multimodal-001",
        "name": "清晰网络故障码面板",
        "sample_kind": "text_panel",
        "image_path": (
            "data/multimodal/tool-selection/"
            "multimodal-001.png"
        ),
        "image_sha256": TEST_IMAGE_SHA256,
        "generation_method": "synthetic_pillow",
        "question": (
            "图片显示的故障码和网络状态是什么？"
        ),
        "preferred_route": "ocr_rule",
        "acceptable_routes": (
            "ocr_rule",
            "vision_model",
        ),
        "routes_to_compare": (
            ALL_COMPARISON_ROUTES
        ),
        "expected_text_terms": (
            "ERR-NET-4001",
            "CONNECTED",
        ),
        "expected_visual_observations": (),
        "expects_abstention": False,
        "contains_untrusted_text": False,
        "rationale": (
            "关键信息是清晰文字，OCR更便宜且可重复。"
        ),
        "tags": (
            "text",
            "fault_code",
        ),
    }

    data.update(overrides)
    return data


def make_visual_state_data(
    **overrides: Any,
) -> dict[str, Any]:
    """返回一个合法的指示灯视觉状态案例字典。"""

    data: dict[str, Any] = {
        "case_id": "multimodal-003",
        "name": "网络指示灯颜色观察",
        "sample_kind": "visual_state",
        "image_path": (
            "data/multimodal/tool-selection/"
            "multimodal-003.png"
        ),
        "image_sha256": "b" * 64,
        "question": (
            "NET指示灯当前是什么颜色和亮灭状态？"
        ),
        "preferred_route": "vision_model",
        "acceptable_routes": (
            "vision_model",
        ),
        "routes_to_compare": (
            ALL_COMPARISON_ROUTES
        ),
        "expected_text_terms": (),
        "expected_visual_observations": (
            "NET指示灯红色常亮",
        ),
        "expects_abstention": False,
        "contains_untrusted_text": False,
        "rationale": (
            "颜色和亮灭不能由普通OCR可靠判断。"
        ),
        "tags": (
            "indicator",
            "color",
        ),
    }

    data.update(overrides)
    return data


def make_degraded_image_data(
    **overrides: Any,
) -> dict[str, Any]:
    """返回一个合法的严重模糊拒答案例字典。"""

    data: dict[str, Any] = {
        "case_id": "multimodal-005",
        "name": "严重模糊仪表截图",
        "sample_kind": "degraded_image",
        "image_path": (
            "data/multimodal/tool-selection/"
            "multimodal-005.png"
        ),
        "image_sha256": "c" * 64,
        "question": (
            "图片中的故障码和设备状态是什么？"
        ),
        "preferred_route": "direct_abstention",
        "acceptable_routes": (
            "direct_abstention",
        ),
        "routes_to_compare": (
            ALL_COMPARISON_ROUTES
        ),
        "expected_text_terms": (),
        "expected_visual_observations": (),
        "expects_abstention": True,
        "contains_untrusted_text": False,
        "rationale": (
            "关键区域严重模糊，不能形成可靠观察。"
        ),
        "tags": (
            "blur",
            "abstention",
        ),
    }

    data.update(overrides)
    return data


def test_text_panel_case_is_valid() -> None:
    """清晰文字案例应允许 OCR 和 Vision，并首选 OCR。"""

    case = MultimodalToolSelectionCase.model_validate(
        make_text_panel_data()
    )

    assert case.sample_kind == "text_panel"
    assert case.preferred_route == "ocr_rule"
    assert case.acceptable_routes == (
        "ocr_rule",
        "vision_model",
    )
    assert case.expected_text_terms == (
        "ERR-NET-4001",
        "CONNECTED",
    )
    assert case.expects_abstention is False


def test_visual_state_case_is_valid() -> None:
    """颜色状态案例应只把 Vision 列为可接受路线。"""

    case = MultimodalToolSelectionCase.model_validate(
        make_visual_state_data()
    )

    assert case.sample_kind == "visual_state"
    assert case.preferred_route == "vision_model"
    assert case.acceptable_routes == (
        "vision_model",
    )
    assert case.expected_visual_observations == (
        "NET指示灯红色常亮",
    )


def test_degraded_image_case_is_valid() -> None:
    """退化图片应只接受直接拒答路线。"""

    case = MultimodalToolSelectionCase.model_validate(
        make_degraded_image_data()
    )

    assert case.sample_kind == "degraded_image"
    assert case.preferred_route == "direct_abstention"
    assert case.acceptable_routes == (
        "direct_abstention",
    )
    assert case.expects_abstention is True


def test_case_round_trips_through_json() -> None:
    """案例应能稳定写入JSONL并从JSON重新校验。"""

    original = MultimodalToolSelectionCase.model_validate(
        make_text_panel_data()
    )

    restored = MultimodalToolSelectionCase.model_validate_json(
        original.model_dump_json()
    )

    assert restored == original


def test_case_is_frozen() -> None:
    """标准案例创建后不能被实验代码原地修改。"""

    case = MultimodalToolSelectionCase.model_validate(
        make_text_panel_data()
    )

    with pytest.raises(
        ValidationError,
        match="Instance is frozen",
    ):
        case.name = "被修改的名称"


def test_case_rejects_extra_field() -> None:
    """拼错或未声明字段不能被静默忽略。"""

    with pytest.raises(
        ValidationError,
        match="Extra inputs are not permitted",
    ):
        MultimodalToolSelectionCase.model_validate(
            make_text_panel_data(
                secret_instruction="execute command",
            )
        )


@pytest.mark.parametrize(
    "case_id",
    [
        "multimodal-1",
        "multimodal-0001",
        "case-001",
        "MULTIMODAL-001",
    ],
)
def test_case_rejects_invalid_case_id(
    case_id: str,
) -> None:
    """案例编号必须保持multimodal加三位数字格式。"""

    with pytest.raises(ValidationError):
        MultimodalToolSelectionCase.model_validate(
            make_text_panel_data(
                case_id=case_id,
            )
        )


@pytest.mark.parametrize(
    "image_sha256",
    [
        "a" * 63,
        "a" * 65,
        "A" * 64,
        "z" * 64,
    ],
)
def test_case_rejects_invalid_image_sha256(
    image_sha256: str,
) -> None:
    """图片摘要必须是64位小写十六进制SHA-256。"""

    with pytest.raises(ValidationError):
        MultimodalToolSelectionCase.model_validate(
            make_text_panel_data(
                image_sha256=image_sha256,
            )
        )


@pytest.mark.parametrize(
    ("image_path", "message"),
    [
        (
            r"data\multimodal\sample.png",
            "image_path必须使用正斜杠",
        ),
        (
            "/data/multimodal/sample.png",
            "image_path必须是相对路径",
        ),
        (
            "data/multimodal/../sample.png",
            "image_path不能包含上级目录",
        ),
        (
            "assets/multimodal/sample.png",
            "image_path必须位于data目录",
        ),
        (
            "data/multimodal/sample.gif",
            "image_path图片格式不受支持",
        ),
    ],
)
def test_case_rejects_unsafe_or_unsupported_image_path(
    image_path: str,
    message: str,
) -> None:
    """图片路径必须安全、可移植并使用允许的格式。"""

    with pytest.raises(
        ValidationError,
        match=message,
    ):
        MultimodalToolSelectionCase.model_validate(
            make_text_panel_data(
                image_path=image_path,
            )
        )


@pytest.mark.parametrize(
    ("field_name", "duplicate_value"),
    [
        (
            "acceptable_routes",
            ("ocr_rule", "ocr_rule"),
        ),
        (
            "routes_to_compare",
            ("ocr_rule", "ocr_rule"),
        ),
        (
            "expected_text_terms",
            ("CONNECTED", "CONNECTED"),
        ),
        (
            "expected_visual_observations",
            ("红灯常亮", "红灯常亮"),
        ),
        (
            "tags",
            ("text", "text"),
        ),
    ],
)
def test_case_rejects_duplicate_tuple_items(
    field_name: str,
    duplicate_value: tuple[str, ...],
) -> None:
    """路线、标准答案和标签中不能重复同一项目。"""

    with pytest.raises(
        ValidationError,
        match="不能包含重复项目",
    ):
        MultimodalToolSelectionCase.model_validate(
            make_text_panel_data(
                **{
                    field_name: duplicate_value,
                }
            )
        )


def test_case_requires_at_least_two_comparison_routes() -> None:
    """任务二要求每个案例至少比较两条处理路线。"""

    with pytest.raises(ValidationError):
        MultimodalToolSelectionCase.model_validate(
            make_text_panel_data(
                acceptable_routes=("ocr_rule",),
                routes_to_compare=("ocr_rule",),
            )
        )


def test_preferred_route_must_be_compared() -> None:
    """没有参加实验的路线不能被声明成首选路线。"""

    with pytest.raises(
        ValidationError,
        match=(
            "preferred_route必须包含在"
            "routes_to_compare中"
        ),
    ):
        MultimodalToolSelectionCase.model_validate(
            make_text_panel_data(
                routes_to_compare=(
                    "vision_model",
                    "direct_abstention",
                ),
            )
        )


def test_preferred_route_must_be_acceptable() -> None:
    """首选路线必须能够产生安全、正确的预期结果。"""

    with pytest.raises(
        ValidationError,
        match=(
            "preferred_route必须包含在"
            "acceptable_routes中"
        ),
    ):
        MultimodalToolSelectionCase.model_validate(
            make_text_panel_data(
                acceptable_routes=(
                    "vision_model",
                ),
            )
        )


def test_acceptable_routes_must_be_compared() -> None:
    """可接受路线必须实际出现在本案例对照范围中。"""

    with pytest.raises(
        ValidationError,
        match=(
            "acceptable_routes必须是"
            "routes_to_compare的子集"
        ),
    ):
        MultimodalToolSelectionCase.model_validate(
            make_text_panel_data(
                routes_to_compare=(
                    "ocr_rule",
                    "direct_abstention",
                ),
            )
        )


def test_abstention_case_requires_direct_preferred_route() -> None:
    """要求拒答时，首选路线不能是OCR或Vision。"""

    with pytest.raises(
        ValidationError,
        match="应拒答案例必须首选direct_abstention",
    ):
        MultimodalToolSelectionCase.model_validate(
            make_degraded_image_data(
                preferred_route="vision_model",
                acceptable_routes=(
                    "vision_model",
                    "direct_abstention",
                ),
            )
        )


def test_abstention_case_only_accepts_direct_abstention() -> None:
    """应拒答案例不能把模型或OCR结果标成正确答案。"""

    with pytest.raises(
        ValidationError,
        match=(
            "应拒答案例只能把direct_abstention"
            "列为可接受路线"
        ),
    ):
        MultimodalToolSelectionCase.model_validate(
            make_degraded_image_data(
                acceptable_routes=(
                    "direct_abstention",
                    "vision_model",
                ),
            )
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        (
            "expected_text_terms",
            ("ERR-NET-4001",),
        ),
        (
            "expected_visual_observations",
            ("红灯常亮",),
        ),
    ],
)
def test_abstention_case_rejects_confirmed_observations(
    field_name: str,
    value: tuple[str, ...],
) -> None:
    """拒答案例不能同时声明存在可确认的标准事实。"""

    with pytest.raises(
        ValidationError,
        match="应拒答案例不能包含可确认的标准观察",
    ):
        MultimodalToolSelectionCase.model_validate(
            make_degraded_image_data(
                **{
                    field_name: value,
                }
            )
        )


def test_answerable_case_rejects_direct_abstention_as_acceptable() -> None:
    """可回答案例不能同时把直接拒答标成正确路线。"""

    with pytest.raises(
        ValidationError,
        match=(
            "可回答案例不能把direct_abstention"
            "列为可接受路线"
        ),
    ):
        MultimodalToolSelectionCase.model_validate(
            make_text_panel_data(
                acceptable_routes=(
                    "ocr_rule",
                    "direct_abstention",
                ),
            )
        )


def test_answerable_case_requires_expected_information() -> None:
    """可回答案例必须提供可用于评分的标准信息。"""

    with pytest.raises(
        ValidationError,
        match=(
            "可回答案例必须包含"
            "标准文字或标准视觉观察"
        ),
    ):
        MultimodalToolSelectionCase.model_validate(
            make_text_panel_data(
                expected_text_terms=(),
                expected_visual_observations=(),
            )
        )


def test_text_panel_requires_ocr_as_preferred_route() -> None:
    """文字面板不能把更昂贵的Vision路线设为首选。"""

    with pytest.raises(
        ValidationError,
        match="text_panel必须首选ocr_rule",
    ):
        MultimodalToolSelectionCase.model_validate(
            make_text_panel_data(
                preferred_route="vision_model",
            )
        )


def test_text_panel_requires_expected_text_terms() -> None:
    """文字案例必须声明OCR或规则解析需要命中的词。"""

    with pytest.raises(
        ValidationError,
        match=(
            "可回答案例必须包含"
            "标准文字或标准视觉观察"
        ),
    ):
        MultimodalToolSelectionCase.model_validate(
            make_text_panel_data(
                expected_text_terms=(),
            )
        )


def test_visual_state_requires_vision_as_preferred_route() -> None:
    """颜色或部件状态案例必须首选Vision模型。"""

    with pytest.raises(
        ValidationError,
        match="visual_state必须首选vision_model",
    ):
        MultimodalToolSelectionCase.model_validate(
            make_visual_state_data(
                preferred_route="ocr_rule",
                acceptable_routes=(
                    "ocr_rule",
                ),
            )
        )


def test_visual_state_requires_expected_observation() -> None:
    """视觉案例必须声明模型应观察到的表面状态。"""

    with pytest.raises(
        ValidationError,
        match=(
            "可回答案例必须包含"
            "标准文字或标准视觉观察"
        ),
    ):
        MultimodalToolSelectionCase.model_validate(
            make_visual_state_data(
                expected_visual_observations=(),
            )
        )


def test_degraded_image_must_expect_abstention() -> None:
    """退化图片不能被定义成可正常回答的视觉案例。"""

    with pytest.raises(
        ValidationError,
        match="degraded_image必须要求拒答",
    ):
        MultimodalToolSelectionCase.model_validate(
            make_degraded_image_data(
                preferred_route="vision_model",
                acceptable_routes=(
                    "vision_model",
                ),
                expects_abstention=False,
                expected_visual_observations=(
                    "面板状态可见",
                ),
            )
        )


def test_generation_method_rejects_non_synthetic_source() -> None:
    """当前公开实验集只能声明为Pillow生成的脱敏样本。"""

    with pytest.raises(ValidationError):
        MultimodalToolSelectionCase.model_validate(
            make_text_panel_data(
                generation_method="customer_photo",
            )
        )
