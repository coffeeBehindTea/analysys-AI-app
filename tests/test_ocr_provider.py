"""Tesseract OCR Provider 的离线单元测试。

这些测试不会启动真实 Tesseract 子进程。

测试通过替换 pytesseract.image_to_data()，向 Provider 提供稳定、
可重复的原始列数据，从而只验证本项目负责的逻辑：参数校验、
异步包装、词语分组、边界框合并、置信度门控和异常转换。
"""

from __future__ import annotations

import os

from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image
import pytest

from app.errors import (
    InvalidOcrResponseError,
    OcrConfigurationError,
    OcrExecutionError,
    OcrTimeoutError,
)
from app.schemas.vision import (
    VisionImageMetadata,
    VisionInput,
)
from app.services.ocr_provider import (
    TESSERACT_ENGINE_MODE,
    TESSERACT_PAGE_SEGMENTATION_MODE,
    TesseractOcrProvider,
    TesseractRuntime,
    _build_tesseract_config,
)


# 所有测试共用的稳定运行时信息。
#
# 单元测试不会真的访问这些路径；它们只是模拟已经被运行时发现函数
# 验证过的 Tesseract 配置，避免测试依赖某台电脑的安装位置。
TEST_ENGINE_PATH = Path("test-runtime/tesseract.exe")
TEST_TESSDATA_DIRECTORY = Path("test-runtime/tessdata")
TEST_ENGINE_VERSION = "5.5.3-test"
TEST_IMAGE_WIDTH_PX = 80
TEST_IMAGE_HEIGHT_PX = 40


def make_runtime(
    *,
    available_languages: tuple[str, ...] = (
        "chi_sim",
        "eng",
    ),
) -> TesseractRuntime:
    """构造不访问真实文件系统的已验证运行时替身。"""

    return TesseractRuntime(
        engine_path=TEST_ENGINE_PATH,
        tessdata_directory=TEST_TESSDATA_DIRECTORY,
        engine_version=TEST_ENGINE_VERSION,
        available_languages=available_languages,
    )


def make_vision_input() -> VisionInput:
    """在内存中创建一张合法 PNG，并构造内部 VisionInput。"""

    image_buffer = BytesIO()

    with Image.new(
        "RGB",
        (
            TEST_IMAGE_WIDTH_PX,
            TEST_IMAGE_HEIGHT_PX,
        ),
        color="white",
    ) as image:
        image.save(
            image_buffer,
            format="PNG",
        )

    image_bytes = image_buffer.getvalue()

    return VisionInput(
        metadata=VisionImageMetadata(
            mime_type="image/png",
            size_bytes=len(image_bytes),
            width_px=TEST_IMAGE_WIDTH_PX,
            height_px=TEST_IMAGE_HEIGHT_PX,
            pixel_count=(
                TEST_IMAGE_WIDTH_PX
                * TEST_IMAGE_HEIGHT_PX
            ),
            sha256_hex=sha256(
                image_bytes
            ).hexdigest(),
        ),
        image_bytes=image_bytes,
        analysis_goal=(
            "读取设备面板上的故障码和网络状态"
        ),
        detail="auto",
    )


def make_raw_ocr_data() -> dict[str, list[object]]:
    """构造两行文字和一个布局占位行的 Tesseract 原始列。"""

    return {
        # 第一项是 Tesseract 常见的空布局行，应当被忽略。
        "text": [
            "",
            "FAULT",
            "CODE",
            "CONNECTED",
        ],
        "conf": [
            "-1",
            "90",
            "80",
            "70",
        ],
        "left": [0, 10, 35, 10],
        "top": [0, 5, 5, 25],
        "width": [1, 20, 25, 60],
        "height": [1, 10, 10, 12],
        "page_num": [1, 1, 1, 1],
        "block_num": [0, 1, 1, 1],
        "par_num": [0, 1, 1, 1],
        "line_num": [0, 1, 1, 2],
    }


def make_provider(
    *,
    minimum_confidence: float = 75.0,
    timeout_seconds: float = 10.0,
) -> TesseractOcrProvider:
    """使用稳定默认值构造待测试 Provider。"""

    return TesseractOcrProvider(
        language="eng",
        minimum_confidence=minimum_confidence,
        timeout_seconds=timeout_seconds,
        runtime=make_runtime(),
    )


@pytest.mark.asyncio
async def test_analyze_image_returns_completed_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """有效列应被分组为两行，并生成 completed 观察。"""

    captured_arguments: dict[str, Any] = {}

    def fake_image_to_data(
        image: Image.Image,
        **kwargs: Any,
    ) -> dict[str, list[object]]:
        """记录 Provider 传入 pytesseract 的关键参数。"""

        captured_arguments["mode"] = image.mode
        captured_arguments["tessdata_prefix"] = (
            os.environ.get("TESSDATA_PREFIX")
        )
        captured_arguments.update(kwargs)
        return make_raw_ocr_data()

    monkeypatch.setattr(
        "app.services.ocr_provider.pytesseract.image_to_data",
        fake_image_to_data,
    )

    vision_input = make_vision_input()
    observation = await make_provider().analyze_image(
        vision_input=vision_input,
    )

    assert observation.status == "completed"
    assert observation.recognized_text == (
        "FAULT CODE\nCONNECTED"
    )
    assert observation.mean_confidence == pytest.approx(
        80.0
    )
    assert observation.minimum_confidence == 75.0
    assert observation.requires_human_check is False
    assert observation.uncertain_items == ()
    assert observation.source_image_sha256 == (
        vision_input.metadata.sha256_hex
    )
    assert observation.engine_name == "tesseract"
    assert observation.engine_version == TEST_ENGINE_VERSION
    assert observation.duration_ms >= 0.0

    # FAULT 位于 x=10..30，CODE 位于 x=35..60，
    # 合并后的第一行边界框应覆盖 x=10..60。
    first_line = observation.lines[0]
    assert first_line.line_index == 0
    assert first_line.text.get_secret_value() == "FAULT CODE"
    assert first_line.confidence == pytest.approx(85.0)
    assert first_line.token_count == 2
    assert first_line.bounding_box.model_dump() == {
        "left_px": 10,
        "top_px": 5,
        "width_px": 50,
        "height_px": 10,
    }

    assert captured_arguments["mode"] == "RGB"
    assert captured_arguments["lang"] == "eng"
    assert captured_arguments["timeout"] == 10.0
    assert captured_arguments["tessdata_prefix"] == str(
        TEST_TESSDATA_DIRECTORY
    )
    assert (
        f"--oem {TESSERACT_ENGINE_MODE}"
        in captured_arguments["config"]
    )
    assert (
        f"--psm {TESSERACT_PAGE_SEGMENTATION_MODE}"
        in captured_arguments["config"]
    )


@pytest.mark.asyncio
async def test_analyze_image_marks_low_confidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """有文字但平均置信度低于门槛时应要求人工检查。"""

    monkeypatch.setattr(
        "app.services.ocr_provider.pytesseract.image_to_data",
        lambda *args, **kwargs: make_raw_ocr_data(),
    )

    observation = await make_provider(
        minimum_confidence=90.0,
    ).analyze_image(
        vision_input=make_vision_input(),
    )

    assert observation.status == "low_confidence"
    assert observation.mean_confidence == pytest.approx(80.0)
    assert observation.requires_human_check is True
    assert observation.uncertain_items == (
        "OCR平均置信度低于实验门槛",
    )


@pytest.mark.asyncio
async def test_analyze_image_returns_empty_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """只有空布局行时应返回可审计的 empty 结果。"""

    raw_data = make_raw_ocr_data()

    for column_name in raw_data:
        raw_data[column_name] = raw_data[column_name][:1]

    monkeypatch.setattr(
        "app.services.ocr_provider.pytesseract.image_to_data",
        lambda *args, **kwargs: raw_data,
    )

    observation = await make_provider().analyze_image(
        vision_input=make_vision_input(),
    )

    assert observation.status == "empty"
    assert observation.lines == ()
    assert observation.mean_confidence is None
    assert observation.recognized_text == ""
    assert observation.text_character_count == 0
    assert observation.requires_human_check is True
    assert observation.uncertain_items == (
        "图片中没有识别到可用文字",
    )


@pytest.mark.asyncio
async def test_analyze_image_rejects_wrong_input_type() -> None:
    """公开入口只能接收已经验证的 VisionInput。"""

    with pytest.raises(
        TypeError,
        match="vision_input必须是VisionInput",
    ):
        await make_provider().analyze_image(
            vision_input="raw-base64",  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_analyze_image_converts_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pytesseract 的超时 RuntimeError 应转换为应用超时异常。"""

    def raise_timeout(*args: Any, **kwargs: Any) -> object:
        raise RuntimeError("Tesseract process timeout")

    monkeypatch.setattr(
        "app.services.ocr_provider.pytesseract.image_to_data",
        raise_timeout,
    )

    with pytest.raises(
        OcrTimeoutError,
        match="本地OCR执行超时",
    ):
        await make_provider().analyze_image(
            vision_input=make_vision_input(),
        )


@pytest.mark.asyncio
async def test_analyze_image_converts_runtime_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """非超时 RuntimeError 应转换为通用 OCR 执行异常。"""

    def raise_runtime_error(
        *args: Any,
        **kwargs: Any,
    ) -> object:
        raise RuntimeError("worker failed")

    monkeypatch.setattr(
        "app.services.ocr_provider.pytesseract.image_to_data",
        raise_runtime_error,
    )

    with pytest.raises(
        OcrExecutionError,
        match="本地OCR执行失败",
    ):
        await make_provider().analyze_image(
            vision_input=make_vision_input(),
        )


@pytest.mark.asyncio
async def test_analyze_image_converts_os_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """图片读取或进程启动 OSError 应转换为 OCR 执行异常。"""

    def raise_os_error(*args: Any, **kwargs: Any) -> object:
        raise OSError("cannot start process")

    monkeypatch.setattr(
        "app.services.ocr_provider.pytesseract.image_to_data",
        raise_os_error,
    )

    with pytest.raises(
        OcrExecutionError,
        match="无法读取图片或启动OCR",
    ):
        await make_provider().analyze_image(
            vision_input=make_vision_input(),
        )


def test_provider_rejects_missing_language_model() -> None:
    """请求未安装语言时应在执行 OCR 前拒绝配置。"""

    with pytest.raises(
        OcrConfigurationError,
        match="Tesseract缺少请求的语言模型：eng",
    ):
        TesseractOcrProvider(
            language="eng",
            minimum_confidence=70.0,
            timeout_seconds=10.0,
            runtime=make_runtime(
                available_languages=("chi_sim",),
            ),
        )


def test_provider_accepts_multiple_installed_languages() -> None:
    """用加号连接且均已安装的语言组合应能够构造 Provider。"""

    provider = TesseractOcrProvider(
        language="eng+chi_sim",
        minimum_confidence=70.0,
        timeout_seconds=10.0,
        runtime=make_runtime(),
    )

    assert isinstance(provider, TesseractOcrProvider)


def test_provider_rejects_repeated_language() -> None:
    """重复语言代码没有意义，应拒绝含糊配置。"""

    with pytest.raises(
        OcrConfigurationError,
        match="OCR language不能重复语言代码",
    ):
        TesseractOcrProvider(
            language="eng+eng",
            minimum_confidence=70.0,
            timeout_seconds=10.0,
            runtime=make_runtime(),
        )


@pytest.mark.parametrize(
    "language",
    [
        "",
        "   ",
        "eng chi_sim",
        "ENG",
        "eng/chi_sim",
    ],
)
def test_provider_rejects_invalid_language_format(
    language: str,
) -> None:
    """语言字段必须满足稳定、可传递给 Tesseract 的格式。"""

    with pytest.raises(
        OcrConfigurationError,
        match="OCR language格式无效",
    ):
        TesseractOcrProvider(
            language=language,
            minimum_confidence=70.0,
            timeout_seconds=10.0,
            runtime=make_runtime(),
        )


@pytest.mark.parametrize(
    "minimum_confidence",
    [
        -0.1,
        100.1,
        float("nan"),
        float("inf"),
    ],
)
def test_provider_rejects_invalid_minimum_confidence(
    minimum_confidence: float,
) -> None:
    """置信度门槛必须是 0 到 100 的有限数字。"""

    with pytest.raises(
        ValueError,
        match="minimum_confidence",
    ):
        make_provider(
            minimum_confidence=minimum_confidence,
        )


@pytest.mark.parametrize(
    "timeout_seconds",
    [
        0.0,
        -1.0,
        60.1,
        float("nan"),
        float("inf"),
    ],
)
def test_provider_rejects_invalid_timeout(
    timeout_seconds: float,
) -> None:
    """本地 OCR 超时必须位于大于 0 且不超过 60 秒的范围。"""

    with pytest.raises(
        ValueError,
        match="timeout_seconds",
    ):
        make_provider(
            timeout_seconds=timeout_seconds,
        )


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "message"),
    [
        (
            "text",
            123,
            "Tesseract text列必须包含str",
        ),
        (
            "conf",
            "not-a-number",
            "Tesseract置信度不是有效数字",
        ),
        (
            "conf",
            "nan",
            "Tesseract置信度必须是有限数字",
        ),
        (
            "conf",
            "101",
            "Tesseract置信度不能超过100",
        ),
        (
            "left",
            -1,
            "Tesseract坐标不能为负数",
        ),
        (
            "width",
            0,
            "Tesseract文字框尺寸必须大于0",
        ),
        (
            "line_num",
            "invalid",
            "Tesseract整数字段无效",
        ),
    ],
)
def test_parser_rejects_invalid_cell_values(
    field_name: str,
    invalid_value: object,
    message: str,
) -> None:
    """原始列中的无效单元格不得进入 OcrObservation。"""

    raw_data = make_raw_ocr_data()
    raw_data[field_name][1] = invalid_value

    with pytest.raises(
        InvalidOcrResponseError,
        match=message,
    ):
        make_provider()._parse_ocr_data(raw_data)


def test_parser_rejects_non_mapping_result() -> None:
    """image_to_data() 返回值必须是按列组织的 Mapping。"""

    with pytest.raises(
        InvalidOcrResponseError,
        match="Tesseract返回值必须是Mapping",
    ):
        make_provider()._parse_ocr_data(
            ["not", "a", "mapping"]
        )


def test_parser_rejects_missing_column() -> None:
    """缺少任何必需列都不能继续组装文字行。"""

    raw_data = make_raw_ocr_data()
    del raw_data["line_num"]

    with pytest.raises(
        InvalidOcrResponseError,
        match="Tesseract缺少必需列：line_num",
    ):
        make_provider()._parse_ocr_data(raw_data)


def test_parser_rejects_string_as_column() -> None:
    """单个字符串虽然是 Sequence，但不能冒充列数组。"""

    raw_data: dict[str, object] = make_raw_ocr_data()
    raw_data["text"] = "FAULT"

    with pytest.raises(
        InvalidOcrResponseError,
        match="Tesseract列必须是序列：text",
    ):
        make_provider()._parse_ocr_data(raw_data)


def test_parser_rejects_unequal_column_lengths() -> None:
    """列长度不一致会让同行字段错位，因此必须拒绝。"""

    raw_data = make_raw_ocr_data()
    raw_data["conf"].pop()

    with pytest.raises(
        InvalidOcrResponseError,
        match="Tesseract返回列长度不一致",
    ):
        make_provider()._parse_ocr_data(raw_data)


def test_build_tesseract_config_contains_fixed_strategy() -> None:
    """配置字符串只应包含固定的 OEM 和 PSM 策略。"""

    config = _build_tesseract_config()

    assert config == (
        f"--oem {TESSERACT_ENGINE_MODE} "
        f"--psm {TESSERACT_PAGE_SEGMENTATION_MODE}"
    )
