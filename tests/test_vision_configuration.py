"""Week 5 Vision输入与调用策略配置的离线测试。

本文件只创建Settings对象，不读取真实.env、
不解码图片，也不调用Vision模型。

测试覆盖：

1. Vision策略配置可以转换成正确的Python数值；
2. 未提供环境变量时使用明确的安全默认值；
3. 字节、尺寸、像素和超时不能为零；
4. 三项图片限制不能超过Schema层绝对边界。
"""

import pytest

from pydantic import (
    ValidationError,
)

from app.config import Settings
from app.schemas.vision import (
    ABSOLUTE_MAX_VISION_IMAGE_BYTES,
    ABSOLUTE_MAX_VISION_IMAGE_DIMENSION_PX,
    ABSOLUTE_MAX_VISION_IMAGE_PIXELS,
)


def test_vision_configuration_is_parsed(
) -> None:
    """显式Vision配置应保存为正确数值类型。"""

    settings = Settings(
        # 禁止读取开发者本地.env，
        # 保证测试只依赖本次显式输入。
        _env_file=None,
        max_vision_image_size_bytes=(
            1_000_000
        ),
        max_vision_image_dimension_px=2_048,
        max_vision_image_pixels=4_000_000,
        vision_timeout_seconds=45.5,
    )

    assert (
        settings.max_vision_image_size_bytes
        == 1_000_000
    )
    assert (
        settings.max_vision_image_dimension_px
        == 2_048
    )
    assert (
        settings.max_vision_image_pixels
        == 4_000_000
    )
    assert settings.vision_timeout_seconds == 45.5
    assert isinstance(
        settings.vision_timeout_seconds,
        float,
    )


def test_vision_configuration_uses_safe_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """没有外部配置时应使用文档记录的业务默认值。"""

    environment_names = (
        "MAX_VISION_IMAGE_SIZE_BYTES",
        "MAX_VISION_IMAGE_DIMENSION_PX",
        "MAX_VISION_IMAGE_PIXELS",
        "VISION_TIMEOUT_SECONDS",
    )

    for environment_name in environment_names:
        # raising=False表示变量原本不存在时
        # 也不抛出KeyError。
        monkeypatch.delenv(
            environment_name,
            raising=False,
        )

    settings = Settings(
        _env_file=None,
    )

    assert (
        settings.max_vision_image_size_bytes
        == 5 * 1024 * 1024
    )
    assert (
        settings.max_vision_image_dimension_px
        == 4_096
    )
    assert (
        settings.max_vision_image_pixels
        == 16_000_000
    )
    assert settings.vision_timeout_seconds == 60.0


@pytest.mark.parametrize(
    "field_name",
    [
        "max_vision_image_size_bytes",
        "max_vision_image_dimension_px",
        "max_vision_image_pixels",
        "vision_timeout_seconds",
    ],
    ids=[
        "byte-limit",
        "dimension-limit",
        "pixel-limit",
        "timeout",
    ],
)
def test_vision_configuration_rejects_zero(
    field_name: str,
) -> None:
    """所有Vision资源和时间限制都必须大于零。"""

    with pytest.raises(
        ValidationError,
        match=field_name,
    ):
        Settings(
            _env_file=None,
            **{
                field_name: 0,
            },
        )


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        (
            "max_vision_image_size_bytes",
            ABSOLUTE_MAX_VISION_IMAGE_BYTES
            + 1,
        ),
        (
            "max_vision_image_dimension_px",
            ABSOLUTE_MAX_VISION_IMAGE_DIMENSION_PX
            + 1,
        ),
        (
            "max_vision_image_pixels",
            ABSOLUTE_MAX_VISION_IMAGE_PIXELS
            + 1,
        ),
        (
            "vision_timeout_seconds",
            300.1,
        ),
    ],
    ids=[
        "byte-absolute-limit",
        "dimension-absolute-limit",
        "pixel-absolute-limit",
        "timeout-upper-limit",
    ],
)
def test_vision_configuration_rejects_upper_limit_overflow(
    field_name: str,
    field_value: int | float,
) -> None:
    """业务配置不能突破Schema绝对边界或超时上限。"""

    with pytest.raises(
        ValidationError,
        match=field_name,
    ):
        Settings(
            _env_file=None,
            **{
                field_name: field_value,
            },
        )
