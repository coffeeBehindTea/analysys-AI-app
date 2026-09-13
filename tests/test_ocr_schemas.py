"""OCR数据契约的离线单元测试。

本文件不读取图片、不调用Tesseract，也不访问网络。

测试覆盖：

1. 边界框、文字行和完整结果可以正常构造；
2. OCR原文不会被repr或JSON直接暴露；
3. completed、low_confidence和empty状态保持一致；
4. 行号、字符数和文字SHA-256不能被伪造；
5. 置信度门槛和人工复核标记必须相互匹配；
6. Pydantic字段范围、枚举和extra=forbid保持有效。
"""

from hashlib import (
    sha256,
)

import pytest

from pydantic import (
    ValidationError,
)

from app.schemas.ocr import (
    OcrBoundingBox,
    OcrObservation,
    OcrTextLine,
)


# 测试只使用固定的虚构图片摘要，
# 不关联任何真实图片。
TEST_IMAGE_SHA256 = "a" * 64


# 模拟当前实验使用的OCR引擎版本。
TEST_ENGINE_VERSION = "5.5.3"


# 70分是本测试使用的示例置信度门槛。
# 它不是所有OCR任务都适用的通用结论。
TEST_MINIMUM_CONFIDENCE = 70.0


def make_bounding_box(
    *,
    left_px: int = 10,
    top_px: int = 20,
    width_px: int = 200,
    height_px: int = 30,
) -> OcrBoundingBox:
    """构造一项合法的测试边界框。"""

    return OcrBoundingBox(
        left_px=left_px,
        top_px=top_px,
        width_px=width_px,
        height_px=height_px,
    )


def make_text_line(
    *,
    line_index: int,
    text: str,
    confidence: float = 92.0,
    token_count: int = 2,
) -> OcrTextLine:
    """构造一行合法的测试OCR文字。"""

    return OcrTextLine(
        line_index=line_index,
        text=text,
        confidence=confidence,
        token_count=token_count,
        bounding_box=make_bounding_box(
            top_px=(
                20 + line_index * 40
            )
        ),
    )


def make_completed_lines(
) -> tuple[OcrTextLine, ...]:
    """返回两行虚构的清晰仪表文字。"""

    return (
        make_text_line(
            line_index=0,
            text=(
                "FAULT CODE: ERR-NET-4001"
            ),
            token_count=3,
        ),
        make_text_line(
            line_index=1,
            text="NETWORK: CONNECTED",
            confidence=94.0,
            token_count=2,
        ),
    )


def text_metadata(
    lines: tuple[OcrTextLine, ...],
) -> tuple[int, str]:
    """根据实际行文字计算字符数和SHA-256。"""

    recognized_text = "\n".join(
        line.text.get_secret_value()
        for line in lines
    )

    return (
        len(recognized_text),
        sha256(
            recognized_text.encode("utf-8")
        ).hexdigest(),
    )


def make_completed_data(
) -> dict[str, object]:
    """返回一份完全合法的completed输入字典。"""

    lines = make_completed_lines()
    (
        character_count,
        text_sha256,
    ) = text_metadata(lines)

    return {
        "status": "completed",
        "lines": lines,
        "mean_confidence": 93.0,
        "minimum_confidence": (
            TEST_MINIMUM_CONFIDENCE
        ),
        "language": "eng",
        "source": "ocr_engine",
        "source_image_sha256": (
            TEST_IMAGE_SHA256
        ),
        "engine_name": "tesseract",
        "engine_version": (
            TEST_ENGINE_VERSION
        ),
        "duration_ms": 15.5,
        "text_character_count": (
            character_count
        ),
        "recognized_text_sha256": (
            text_sha256
        ),
        "requires_human_check": False,
        "uncertain_items": (),
    }


def make_low_confidence_data(
) -> dict[str, object]:
    """返回一份完全合法的low_confidence输入字典。"""

    data = make_completed_data()
    data.update({
        "status": "low_confidence",
        "mean_confidence": 45.0,
        "requires_human_check": True,
        "uncertain_items": (
            "OCR平均置信度低于实验门槛",
        ),
    })
    return data


def make_empty_data(
) -> dict[str, object]:
    """返回一份完全合法的empty输入字典。"""

    empty_text_sha256 = sha256(
        b""
    ).hexdigest()

    return {
        "status": "empty",
        "lines": (),
        "mean_confidence": None,
        "minimum_confidence": (
            TEST_MINIMUM_CONFIDENCE
        ),
        "language": "eng",
        "source": "ocr_engine",
        "source_image_sha256": (
            TEST_IMAGE_SHA256
        ),
        "engine_name": "tesseract",
        "engine_version": (
            TEST_ENGINE_VERSION
        ),
        "duration_ms": 12.0,
        "text_character_count": 0,
        "recognized_text_sha256": (
            empty_text_sha256
        ),
        "requires_human_check": True,
        "uncertain_items": (
            "图片中没有识别到可用文字",
        ),
    }


def test_completed_observation_preserves_text_and_metadata(
) -> None:
    """合法completed结果应保留顺序、摘要和质量信息。"""

    observation = (
        OcrObservation.model_validate(
            make_completed_data()
        )
    )

    assert observation.status == "completed"
    assert observation.mean_confidence == 93.0
    assert observation.requires_human_check is False
    assert len(observation.lines) == 2
    assert observation.recognized_text == (
        "FAULT CODE: ERR-NET-4001\n"
        "NETWORK: CONNECTED"
    )
    assert observation.text_character_count == len(
        observation.recognized_text
    )
    assert observation.recognized_text_sha256 == (
        sha256(
            observation
            .recognized_text
            .encode("utf-8")
        ).hexdigest()
    )


def test_ocr_text_is_hidden_from_repr_and_json(
) -> None:
    """普通显示和JSON不能直接暴露OCR原文。"""

    observation = (
        OcrObservation.model_validate(
            make_completed_data()
        )
    )

    private_text = "ERR-NET-4001"

    # OcrTextLine的text字段设置了repr=False。
    assert private_text not in repr(
        observation
    )

    # SecretStr在JSON序列化时只输出掩码。
    serialized = observation.model_dump_json()
    assert private_text not in serialized
    assert "**********" in serialized

    # recognized_text是普通property，
    # 不会自动成为Pydantic序列化字段。
    assert (
        "recognized_text"
        not in observation.model_dump()
    )

    # 只有显式访问property时才能取得原文。
    assert private_text in (
        observation.recognized_text
    )


def test_low_confidence_observation_requires_review(
) -> None:
    """低于门槛的文字应保留，但必须要求人工复核。"""

    observation = (
        OcrObservation.model_validate(
            make_low_confidence_data()
        )
    )

    assert observation.status == "low_confidence"
    assert observation.mean_confidence == 45.0
    assert observation.requires_human_check is True
    assert observation.uncertain_items == (
        "OCR平均置信度低于实验门槛",
    )


def test_empty_observation_has_no_text_or_confidence(
) -> None:
    """没有识别结果时应返回可解释的安全空结果。"""

    observation = (
        OcrObservation.model_validate(
            make_empty_data()
        )
    )

    assert observation.status == "empty"
    assert observation.lines == ()
    assert observation.recognized_text == ""
    assert observation.mean_confidence is None
    assert observation.requires_human_check is True
    assert observation.text_character_count == 0


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("left_px", -1),
        ("top_px", -1),
        ("width_px", 0),
        ("height_px", 0),
    ],
    ids=[
        "negative-left",
        "negative-top",
        "zero-width",
        "zero-height",
    ],
)
def test_bounding_box_rejects_invalid_coordinates(
    field_name: str,
    field_value: int,
) -> None:
    """边界框不能使用负坐标或零尺寸。"""

    data = {
        "left_px": 10,
        "top_px": 20,
        "width_px": 200,
        "height_px": 30,
    }
    data[field_name] = field_value

    with pytest.raises(
        ValidationError,
        match=field_name,
    ):
        OcrBoundingBox.model_validate(
            data
        )


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("confidence", -0.1),
        ("confidence", 100.1),
        ("token_count", 0),
    ],
    ids=[
        "confidence-below-zero",
        "confidence-above-one-hundred",
        "zero-token-count",
    ],
)
def test_text_line_rejects_invalid_quality_fields(
    field_name: str,
    field_value: int | float,
) -> None:
    """文字行置信度和词语数量必须在合法范围内。"""

    data: dict[str, object] = {
        "line_index": 0,
        "text": "NETWORK: CONNECTED",
        "confidence": 92.0,
        "token_count": 2,
        "bounding_box": make_bounding_box(),
    }
    data[field_name] = field_value

    with pytest.raises(
        ValidationError,
        match=field_name,
    ):
        OcrTextLine.model_validate(
            data
        )


def test_observation_rejects_non_contiguous_line_indices(
) -> None:
    """OCR行号必须从零开始连续排列。"""

    lines = (
        make_text_line(
            line_index=0,
            text="FAULT CODE",
        ),
        make_text_line(
            line_index=2,
            text="ERR-NET-4001",
        ),
    )
    character_count, text_sha256 = (
        text_metadata(lines)
    )

    data = make_completed_data()
    data.update({
        "lines": lines,
        "text_character_count": character_count,
        "recognized_text_sha256": text_sha256,
    })

    with pytest.raises(
        ValidationError,
        match="行号必须从0开始连续",
    ):
        OcrObservation.model_validate(
            data
        )


@pytest.mark.parametrize(
    "metadata_field",
    [
        "text_character_count",
        "recognized_text_sha256",
    ],
    ids=[
        "character-count",
        "text-sha256",
    ],
)
def test_observation_rejects_forged_text_metadata(
    metadata_field: str,
) -> None:
    """字符数和文字摘要必须与实际SecretStr内容一致。"""

    data = make_completed_data()

    if metadata_field == "text_character_count":
        data[metadata_field] = (
            int(data[metadata_field]) + 1
        )
    else:
        data[metadata_field] = "b" * 64

    with pytest.raises(
        ValidationError,
        match=metadata_field,
    ):
        OcrObservation.model_validate(
            data
        )


def test_empty_status_rejects_text_lines(
) -> None:
    """empty状态不能隐藏实际存在的OCR文字。"""

    lines = make_completed_lines()
    character_count, text_sha256 = (
        text_metadata(lines)
    )
    data = make_empty_data()
    data.update({
        "lines": lines,
        "text_character_count": character_count,
        "recognized_text_sha256": text_sha256,
    })

    with pytest.raises(
        ValidationError,
        match="empty状态不能包含OCR文字行",
    ):
        OcrObservation.model_validate(
            data
        )


@pytest.mark.parametrize(
    ("field_name", "field_value", "message"),
    [
        (
            "mean_confidence",
            10.0,
            "mean_confidence必须为None",
        ),
        (
            "requires_human_check",
            False,
            "empty状态必须要求人工检查",
        ),
        (
            "uncertain_items",
            (),
            "empty状态必须说明",
        ),
    ],
    ids=[
        "has-mean-confidence",
        "no-human-check",
        "no-uncertainty",
    ],
)
def test_empty_status_requires_safe_metadata(
    field_name: str,
    field_value: object,
    message: str,
) -> None:
    """empty结果必须没有置信度，并解释人工复核原因。"""

    data = make_empty_data()
    data[field_name] = field_value

    with pytest.raises(
        ValidationError,
        match=message,
    ):
        OcrObservation.model_validate(
            data
        )


@pytest.mark.parametrize(
    ("field_name", "field_value", "message"),
    [
        (
            "mean_confidence",
            69.9,
            "不能低于门槛",
        ),
        (
            "requires_human_check",
            True,
            "completed状态不能因为",
        ),
        (
            "uncertain_items",
            ("不应存在的不确定项",),
            "completed状态不能包含",
        ),
    ],
    ids=[
        "below-threshold",
        "human-check",
        "uncertainty",
    ],
)
def test_completed_status_rejects_inconsistent_quality(
    field_name: str,
    field_value: object,
    message: str,
) -> None:
    """completed必须达到门槛且不带OCR质量拒答标记。"""

    data = make_completed_data()
    data[field_name] = field_value

    with pytest.raises(
        ValidationError,
        match=message,
    ):
        OcrObservation.model_validate(
            data
        )


@pytest.mark.parametrize(
    ("field_name", "field_value", "message"),
    [
        (
            "mean_confidence",
            70.0,
            "必须低于门槛",
        ),
        (
            "requires_human_check",
            False,
            "必须要求人工检查",
        ),
        (
            "uncertain_items",
            (),
            "必须说明",
        ),
    ],
    ids=[
        "at-threshold",
        "no-human-check",
        "no-uncertainty",
    ],
)
def test_low_confidence_status_requires_safe_review(
    field_name: str,
    field_value: object,
    message: str,
) -> None:
    """低置信度结果必须低于门槛并进入人工复核。"""

    data = make_low_confidence_data()
    data[field_name] = field_value

    with pytest.raises(
        ValidationError,
        match=message,
    ):
        OcrObservation.model_validate(
            data
        )


def test_uncertain_items_must_be_unique(
) -> None:
    """同一个OCR质量问题不能在结果中重复出现。"""

    data = make_low_confidence_data()
    data["uncertain_items"] = (
        "文字模糊",
        "文字模糊",
    )

    with pytest.raises(
        ValidationError,
        match="不确定项不能重复",
    ):
        OcrObservation.model_validate(
            data
        )


def test_observation_rejects_unknown_fields(
) -> None:
    """OCR结果不能携带未声明的命令或控制字段。"""

    data = make_completed_data()
    data["execute_command"] = (
        "resume robot"
    )

    with pytest.raises(
        ValidationError,
        match="execute_command",
    ):
        OcrObservation.model_validate(
            data
        )
