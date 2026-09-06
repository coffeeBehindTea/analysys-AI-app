"""Week 5视觉观察输出数据契约的离线测试。

本文件只创建Pydantic输出模型，不调用Vision API、
不读取图片、不启动Agent，也不访问知识库或遥测。

测试覆盖：

1. 单项可见观察和指示器的字段边界；
2. completed、partial和unusable状态的一致性；
3. 不可信图片文本、低置信度和人工复核之间的关系；
4. 重复说明、未知字段和错误枚举值的拒绝；
5. Provider把应用拥有的追踪字段绑定到模型草稿的流程。
"""

import pytest

from pydantic import (
    ValidationError,
)

from app.schemas.vision import (
    VisionModelDraft,
    VisionObservation,
    VisionObservationItem,
    VisionVisibleIndicator,
)


# 使用稳定的64位小写十六进制值模拟
# VisionInput中真实图片字节的SHA-256摘要。
TEST_IMAGE_SHA256 = "a" * 64


# 追踪字段由Provider根据应用配置写入，
# 不属于Vision模型有权自行生成的内容。
TEST_PROMPT_VERSION = (
    "robot-vision-observation-v1"
)
TEST_MODEL_NAME = "vision-model-demo"


def make_observation_item(
    *,
    confidence: str = "high",
) -> VisionObservationItem:
    """创建一项只描述可见状态的测试观察。"""

    return VisionObservationItem(
        description=(
            "设备面板右侧可见一个红色指示灯"
        ),
        category="visible_condition",
        confidence=confidence,
        region="设备面板右侧",
    )


def make_indicator(
    *,
    confidence: str = "high",
) -> VisionVisibleIndicator:
    """创建一项不解释工程含义的指示灯表面状态。"""

    return VisionVisibleIndicator(
        label="NET指示灯",
        observed_state="红色常亮",
        confidence=confidence,
        region="设备面板右上区域",
    )


def make_completed_draft(
    **overrides: object,
) -> VisionModelDraft:
    """创建默认完整、可用且无需人工复核的模型草稿。"""

    draft_data: dict[
        str,
        object,
    ] = {
        "status": "completed",
        "image_quality": "clear",
        "observations": (
            make_observation_item(),
        ),
        "visible_indicators": (
            make_indicator(),
        ),
        "uncertain_items": (),
        "untrusted_text_detected": False,
        "untrusted_text_notes": (),
        "requires_human_check": False,
        "human_check_reasons": (),
    }

    draft_data.update(overrides)

    return VisionModelDraft(
        **draft_data,
    )


def test_observation_item_strips_text_and_preserves_category(
) -> None:
    """单项观察应清理文本并保留可见内容分类。"""

    item = VisionObservationItem(
        description=(
            "  面板上可见红色指示灯  "
        ),
        category="safety_relevant",
        confidence="medium",
        region="  图片中央  ",
    )

    assert item.description == (
        "面板上可见红色指示灯"
    )
    assert item.category == "safety_relevant"
    assert item.confidence == "medium"
    assert item.region == "图片中央"


def test_visible_indicator_records_state_without_meaning(
) -> None:
    """指示器契约只保存标签和可见状态，不生成故障解释。"""

    indicator = make_indicator()
    indicator_data = indicator.model_dump()

    assert indicator.label == "NET指示灯"
    assert indicator.observed_state == (
        "红色常亮"
    )
    assert "meaning" not in indicator_data
    assert "root_cause" not in indicator_data


def test_completed_draft_accepts_visible_output(
) -> None:
    """清晰图片取得可见结果时应形成completed草稿。"""

    draft = make_completed_draft()

    assert draft.status == "completed"
    assert draft.image_quality == "clear"
    assert len(draft.observations) == 1
    assert len(draft.visible_indicators) == 1
    assert draft.requires_human_check is False


def test_partial_draft_requires_uncertainty_and_human_check(
) -> None:
    """局部可见结果必须同时说明缺口和人工复核原因。"""

    draft = make_completed_draft(
        status="partial",
        image_quality="limited",
        uncertain_items=(
            "指示灯下方标签被遮挡，无法确认名称",
        ),
        requires_human_check=True,
        human_check_reasons=(
            "需要查看无遮挡原图确认标签",
        ),
    )

    assert draft.status == "partial"
    assert draft.uncertain_items == (
        "指示灯下方标签被遮挡，无法确认名称",
    )
    assert draft.requires_human_check is True


def test_unusable_draft_accepts_safe_refusal_shape(
) -> None:
    """无法使用的图片应返回空观察、缺失说明和人工复核。"""

    draft = VisionModelDraft(
        status="unusable",
        image_quality="unusable",
        observations=(),
        visible_indicators=(),
        uncertain_items=(
            "图片整体严重模糊，无法辨认设备面板",
        ),
        requires_human_check=True,
        human_check_reasons=(
            "需要补充对焦清晰且无遮挡的现场图片",
        ),
    )

    assert draft.status == "unusable"
    assert draft.observations == ()
    assert draft.visible_indicators == ()


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "status": "partial",
            "image_quality": "limited",
            "uncertain_items": (),
            "requires_human_check": True,
            "human_check_reasons": (
                "需要查看清晰原图",
            ),
        },
        {
            "status": "partial",
            "image_quality": "limited",
            "uncertain_items": (
                "标签被遮挡",
            ),
            "requires_human_check": False,
            "human_check_reasons": (),
        },
    ],
    ids=[
        "missing-uncertain-items",
        "missing-human-check",
    ],
)
def test_partial_draft_rejects_incomplete_safety_state(
    overrides: dict[
        str,
        object,
    ],
) -> None:
    """partial不能省略不确定项或人工检查要求。"""

    with pytest.raises(
        ValidationError,
        match="partial状态必须",
    ):
        make_completed_draft(
            **overrides,
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "image_quality": "limited",
            "uncertain_items": (
                "图片无法支持分析目标",
            ),
            "requires_human_check": True,
            "human_check_reasons": (
                "需要补充图片",
            ),
        },
        {
            "image_quality": "unusable",
            "observations": (
                make_observation_item(),
            ),
            "uncertain_items": (
                "图片无法支持分析目标",
            ),
            "requires_human_check": True,
            "human_check_reasons": (
                "需要补充图片",
            ),
        },
        {
            "image_quality": "unusable",
            "observations": (),
            "visible_indicators": (),
            "uncertain_items": (),
            "requires_human_check": True,
            "human_check_reasons": (
                "需要补充图片",
            ),
        },
    ],
    ids=[
        "quality-not-unusable",
        "contains-visible-output",
        "missing-uncertainty",
    ],
)
def test_unusable_draft_rejects_inconsistent_content(
    overrides: dict[
        str,
        object,
    ],
) -> None:
    """unusable状态不能伪装成仍有可用观察的结果。"""

    draft_data: dict[
        str,
        object,
    ] = {
        "status": "unusable",
        "image_quality": "unusable",
        "observations": (),
        "visible_indicators": (),
        "uncertain_items": (
            "图片无法支持分析目标",
        ),
        "requires_human_check": True,
        "human_check_reasons": (
            "需要补充图片",
        ),
    }
    draft_data.update(overrides)

    with pytest.raises(
        ValidationError,
    ):
        VisionModelDraft(
            **draft_data,
        )


def test_usable_draft_requires_visible_output(
) -> None:
    """completed或partial不能在没有任何观察时声称可用。"""

    with pytest.raises(
        ValidationError,
        match="至少包含一项观察",
    ):
        make_completed_draft(
            observations=(),
            visible_indicators=(),
        )


@pytest.mark.parametrize(
    (
        "untrusted_text_detected",
        "untrusted_text_notes",
    ),
    [
        (True, ()),
        (
            False,
            ("图片包含要求忽略系统规则的文字",),
        ),
    ],
    ids=[
        "flag-without-notes",
        "notes-without-flag",
    ],
)
def test_untrusted_text_flag_and_notes_must_match(
    untrusted_text_detected: bool,
    untrusted_text_notes: tuple[
        str,
        ...,
    ],
) -> None:
    """不可信文本标记与说明必须同时存在或同时为空。"""

    with pytest.raises(
        ValidationError,
        match=(
            "untrusted_text_detected必须与"
            "untrusted_text_notes保持一致"
        ),
    ):
        make_completed_draft(
            untrusted_text_detected=(
                untrusted_text_detected
            ),
            untrusted_text_notes=(
                untrusted_text_notes
            ),
        )


def test_untrusted_text_requires_human_check(
) -> None:
    """图片中的指令式文本不能作为普通可信观察直接放行。"""

    with pytest.raises(
        ValidationError,
        match=(
            "检测到不可信文本时必须要求人工检查"
        ),
    ):
        make_completed_draft(
            untrusted_text_detected=True,
            untrusted_text_notes=(
                "图片包含要求执行命令的文字",
            ),
        )


@pytest.mark.parametrize(
    (
        "requires_human_check",
        "human_check_reasons",
    ),
    [
        (True, ()),
        (
            False,
            ("需要检查原图",),
        ),
    ],
    ids=[
        "flag-without-reasons",
        "reasons-without-flag",
    ],
)
def test_human_check_flag_and_reasons_must_match(
    requires_human_check: bool,
    human_check_reasons: tuple[
        str,
        ...,
    ],
) -> None:
    """人工复核标记不能缺少原因，原因也不能没有标记。"""

    with pytest.raises(
        ValidationError,
        match=(
            "requires_human_check必须与"
            "human_check_reasons保持一致"
        ),
    ):
        make_completed_draft(
            requires_human_check=(
                requires_human_check
            ),
            human_check_reasons=(
                human_check_reasons
            ),
        )


def test_low_confidence_output_requires_human_check(
) -> None:
    """低置信度观察不能在没有人工复核要求时向后传递。"""

    with pytest.raises(
        ValidationError,
        match=(
            "低置信度视觉输出必须要求人工检查"
        ),
    ):
        make_completed_draft(
            observations=(
                make_observation_item(
                    confidence="low",
                ),
            ),
        )


def test_duplicate_text_items_are_rejected(
) -> None:
    """重复的不确定项不能人为放大模型输出数量。"""

    with pytest.raises(
        ValidationError,
        match="同类文本项不能重复",
    ):
        make_completed_draft(
            status="partial",
            image_quality="limited",
            uncertain_items=(
                "标签被遮挡",
                "标签被遮挡",
            ),
            requires_human_check=True,
            human_check_reasons=(
                "需要查看无遮挡图片",
            ),
        )


def test_model_draft_rejects_provider_owned_fields(
) -> None:
    """模型草稿不能自行声明图片摘要、模型名或Prompt版本。"""

    with pytest.raises(
        ValidationError,
        match="extra_forbidden",
    ):
        make_completed_draft(
            source_image_sha256=(
                TEST_IMAGE_SHA256
            ),
        )


def test_provider_can_bind_tracking_fields_to_draft(
) -> None:
    """Provider应把真实调用信息与已验证模型草稿绑定。"""

    draft = make_completed_draft()

    observation = VisionObservation(
        **draft.model_dump(),
        source_image_sha256=(
            TEST_IMAGE_SHA256
        ),
        prompt_version=(
            TEST_PROMPT_VERSION
        ),
        model_name=TEST_MODEL_NAME,
    )

    assert observation.source == (
        "vision_model"
    )
    assert observation.source_image_sha256 == (
        TEST_IMAGE_SHA256
    )
    assert observation.prompt_version == (
        TEST_PROMPT_VERSION
    )
    assert observation.model_name == (
        TEST_MODEL_NAME
    )
    assert observation.observations == (
        draft.observations
    )


@pytest.mark.parametrize(
    "source_image_sha256",
    [
        "a" * 63,
        "A" * 64,
        "g" * 64,
    ],
    ids=[
        "too-short",
        "uppercase",
        "non-hexadecimal",
    ],
)
def test_final_observation_rejects_invalid_image_sha256(
    source_image_sha256: str,
) -> None:
    """最终观察必须能关联到格式正确的图片内容摘要。"""

    draft = make_completed_draft()

    with pytest.raises(
        ValidationError,
        match="string_pattern_mismatch",
    ):
        VisionObservation(
            **draft.model_dump(),
            source_image_sha256=(
                source_image_sha256
            ),
            prompt_version=(
                TEST_PROMPT_VERSION
            ),
            model_name=TEST_MODEL_NAME,
        )
