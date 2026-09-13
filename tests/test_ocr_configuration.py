"""本地OCR策略配置和异常层级的离线测试。

本文件只构造Settings和检查异常类型，
不读取真实.env、不打开图片，也不启动Tesseract。
"""

import pytest

from pydantic import (
    ValidationError,
)

from app.config import Settings
from app.errors import (
    ApplicationError,
    InvalidOcrResponseError,
    OcrConfigurationError,
    OcrExecutionError,
    OcrTimeoutError,
)


def test_ocr_configuration_is_parsed(
) -> None:
    """显式OCR配置应转换成正确的Python值。"""

    settings = Settings(
        # 禁止读取开发者本地.env，
        # 保证结果只来自本次显式输入。
        _env_file=None,
        ocr_language="eng+chi_sim",
        ocr_minimum_confidence=75.5,
        ocr_timeout_seconds=8.5,
    )

    assert settings.ocr_language == (
        "eng+chi_sim"
    )
    assert (
        settings.ocr_minimum_confidence
        == 75.5
    )
    assert settings.ocr_timeout_seconds == 8.5
    assert isinstance(
        settings.ocr_minimum_confidence,
        float,
    )
    assert isinstance(
        settings.ocr_timeout_seconds,
        float,
    )


def test_ocr_configuration_uses_documented_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """没有外部配置时应使用可复现的英文OCR基线。"""

    environment_names = (
        "OCR_LANGUAGE",
        "OCR_MINIMUM_CONFIDENCE",
        "OCR_TIMEOUT_SECONDS",
    )

    for environment_name in environment_names:
        # raising=False表示环境变量原本不存在时
        # 也不抛出KeyError。
        monkeypatch.delenv(
            environment_name,
            raising=False,
        )

    settings = Settings(
        _env_file=None,
    )

    assert settings.ocr_language == "eng"
    assert (
        settings.ocr_minimum_confidence
        == 70.0
    )
    assert settings.ocr_timeout_seconds == 10.0


@pytest.mark.parametrize(
    "language",
    [
        "",
        "eng chi_sim",
        "eng/chi_sim",
        "ENG+CHI_SIM",
    ],
    ids=[
        "empty",
        "contains-space",
        "contains-slash",
        "uppercase",
    ],
)
def test_ocr_language_rejects_invalid_format(
    language: str,
) -> None:
    """语言配置只允许Tesseract语言代码组合格式。"""

    with pytest.raises(
        ValidationError,
        match="ocr_language",
    ):
        Settings(
            _env_file=None,
            ocr_language=language,
        )


@pytest.mark.parametrize(
    "minimum_confidence",
    [
        -0.1,
        100.1,
        float("inf"),
        float("nan"),
    ],
    ids=[
        "below-zero",
        "above-one-hundred",
        "infinity",
        "not-a-number",
    ],
)
def test_ocr_minimum_confidence_rejects_invalid_values(
    minimum_confidence: float,
) -> None:
    """OCR置信度门槛必须是0到100之间的有限数字。"""

    with pytest.raises(
        ValidationError,
        match="ocr_minimum_confidence",
    ):
        Settings(
            _env_file=None,
            ocr_minimum_confidence=(
                minimum_confidence
            ),
        )


@pytest.mark.parametrize(
    "timeout_seconds",
    [
        0.0,
        60.1,
        float("inf"),
        float("nan"),
    ],
    ids=[
        "zero",
        "above-sixty",
        "infinity",
        "not-a-number",
    ],
)
def test_ocr_timeout_rejects_invalid_values(
    timeout_seconds: float,
) -> None:
    """OCR超时必须是大于0且不超过60秒的有限数字。"""

    with pytest.raises(
        ValidationError,
        match="ocr_timeout_seconds",
    ):
        Settings(
            _env_file=None,
            ocr_timeout_seconds=(
                timeout_seconds
            ),
        )


@pytest.mark.parametrize(
    "error_type",
    [
        OcrConfigurationError,
        OcrTimeoutError,
        OcrExecutionError,
        InvalidOcrResponseError,
    ],
    ids=[
        "configuration",
        "timeout",
        "execution",
        "invalid-response",
    ],
)
def test_ocr_errors_are_application_errors(
    error_type: type[ApplicationError],
) -> None:
    """所有OCR可预期异常都必须属于ApplicationError。"""

    error = error_type(
        "test OCR error"
    )

    assert isinstance(
        error,
        ApplicationError,
    )
