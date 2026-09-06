"""VisionInputAdapter图片解码与真实性校验的离线测试。

本文件使用Pillow在内存中生成脱敏合成图片。
不读取磁盘图片、不启动FastAPI、不访问网络，
也不调用任何真实Vision模型。

测试覆盖：

1. JPEG、PNG和静态WebP的合法适配流程；
2. Adapter构造参数的类型、正数和绝对上限；
3. 非法Base64、空载荷和字节大小限制；
4. 真实格式白名单和声明MIME一致性；
5. 单边尺寸、总像素和动态图限制；
6. 截断图片和Pillow解压缩炸弹异常转换；
7. 原始图片字节、SHA-256、分析目标和detail保持一致；
8. Adapter不会修改Pillow进程级MAX_IMAGE_PIXELS设置。
"""

from base64 import (
    b64encode,
)
from hashlib import (
    sha256,
)
from io import (
    BytesIO,
)

import pytest

from PIL import Image
from pydantic import ValidationError

from app.errors import (
    VisionInputValidationError,
)
from app.schemas.vision import (
    ABSOLUTE_MAX_VISION_IMAGE_BYTES,
    ABSOLUTE_MAX_VISION_IMAGE_DIMENSION_PX,
    ABSOLUTE_MAX_VISION_IMAGE_PIXELS,
    SupportedVisionMimeType,
    VisionImagePayload,
    VisionInput,
)
from app.services.vision_input import (
    VisionInputAdapter,
)


# 合法图片测试统一使用12×8像素。
# 图片足够小，可以快速生成和解码，
# 同时宽、高和总像素都不是相同数字，
# 便于发现元数据字段写反等问题。
TEST_IMAGE_SIZE = (12, 8)


# 默认Adapter限制明显高于合成测试图片，
# 但远低于生产绝对边界，便于单独覆盖超限场景。
DEFAULT_MAX_IMAGE_SIZE_BYTES = 100_000
DEFAULT_MAX_IMAGE_DIMENSION_PX = 100
DEFAULT_MAX_IMAGE_PIXELS = 10_000


# 每组依次是Pillow保存格式和对应HTTP MIME类型。
# pytest会把它展开成三个独立的合法格式用例。
SUPPORTED_FORMAT_CASES: tuple[
    tuple[
        str,
        SupportedVisionMimeType,
    ],
    ...,
] = (
    ("JPEG", "image/jpeg"),
    ("PNG", "image/png"),
    ("WEBP", "image/webp"),
)


def make_adapter(
    **overrides: object,
) -> VisionInputAdapter:
    """创建使用小型测试限制的VisionInputAdapter。"""

    adapter_arguments: dict[
        str,
        object,
    ] = {
        "max_image_size_bytes": (
            DEFAULT_MAX_IMAGE_SIZE_BYTES
        ),
        "max_image_dimension_px": (
            DEFAULT_MAX_IMAGE_DIMENSION_PX
        ),
        "max_image_pixels": (
            DEFAULT_MAX_IMAGE_PIXELS
        ),
    }

    adapter_arguments.update(overrides)

    return VisionInputAdapter(
        **adapter_arguments,
    )


def make_static_image_bytes(
    *,
    image_format: str = "PNG",
    size: tuple[int, int] = TEST_IMAGE_SIZE,
) -> bytes:
    """在内存中创建一张单帧RGB合成图片。"""

    output = BytesIO()

    # with确保Pillow Image对象在保存后关闭。
    with Image.new(
        "RGB",
        size,
        color=(220, 20, 60),
    ) as image:
        image.save(
            output,
            format=image_format,
        )

    return output.getvalue()


def make_animated_webp_bytes() -> bytes:
    """在内存中创建两帧颜色不同的动画WebP。"""

    output = BytesIO()

    with Image.new(
        "RGB",
        TEST_IMAGE_SIZE,
        color="red",
    ) as first_frame:
        with Image.new(
            "RGB",
            TEST_IMAGE_SIZE,
            color="blue",
        ) as second_frame:
            first_frame.save(
                output,
                format="WEBP",
                save_all=True,
                append_images=[
                    second_frame,
                ],
                duration=100,
                loop=0,
            )

    return output.getvalue()


def make_payload(
    *,
    image_bytes: bytes,
    mime_type: SupportedVisionMimeType = (
        "image/png"
    ),
    analysis_goal: str = (
        "检查设备面板上的指示灯"
    ),
    detail: str = "high",
) -> VisionImagePayload:
    """把测试图片字节编码成合法外部载荷。"""

    return VisionImagePayload(
        mime_type=mime_type,
        image_base64=(
            b64encode(
                image_bytes
            ).decode("ascii")
        ),
        analysis_goal=analysis_goal,
        detail=detail,
    )


@pytest.mark.parametrize(
    ("image_format", "mime_type"),
    SUPPORTED_FORMAT_CASES,
    ids=[
        "jpeg",
        "png",
        "webp",
    ],
)
def test_adapter_accepts_supported_static_images(
    image_format: str,
    mime_type: SupportedVisionMimeType,
) -> None:
    """三种白名单静态格式应形成完整一致的VisionInput。"""

    image_bytes = make_static_image_bytes(
        image_format=image_format,
    )
    payload = make_payload(
        image_bytes=image_bytes,
        mime_type=mime_type,
    )

    result = make_adapter().adapt(
        payload
    )

    assert isinstance(result, VisionInput)
    assert result.metadata.mime_type == mime_type
    assert result.metadata.size_bytes == len(
        image_bytes
    )
    assert result.metadata.width_px == 12
    assert result.metadata.height_px == 8
    assert result.metadata.pixel_count == 96
    assert result.metadata.sha256_hex == (
        sha256(image_bytes).hexdigest()
    )

    # Adapter不重编码或修改原图，
    # Provider收到的必须仍是同一组原始字节。
    assert (
        result.image_bytes.get_secret_value()
        == image_bytes
    )
    assert result.analysis_goal == (
        "检查设备面板上的指示灯"
    )
    assert result.detail == "high"


def test_adapter_requires_payload_contract(
) -> None:
    """Adapter不能跳过外部Schema直接接收字典。"""

    with pytest.raises(
        TypeError,
        match=(
            "payload必须是"
            "VisionImagePayload"
        ),
    ):
        make_adapter().adapt(
            {
                "mime_type": "image/png",
            }
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "max_image_size_bytes",
        "max_image_dimension_px",
        "max_image_pixels",
    ],
    ids=[
        "byte-limit",
        "dimension-limit",
        "pixel-limit",
    ],
)
def test_adapter_rejects_zero_limits(
    field_name: str,
) -> None:
    """直接构造Adapter时三项限制都必须大于零。"""

    with pytest.raises(
        ValueError,
        match=(
            f"{field_name}必须大于0"
        ),
    ):
        make_adapter(
            **{
                field_name: 0,
            },
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "max_image_size_bytes",
        "max_image_dimension_px",
        "max_image_pixels",
    ],
    ids=[
        "byte-limit",
        "dimension-limit",
        "pixel-limit",
    ],
)
def test_adapter_rejects_boolean_limits(
    field_name: str,
) -> None:
    """True虽然是int子类，但不能作为资源限制。"""

    with pytest.raises(
        TypeError,
        match=(
            f"{field_name}必须是int"
        ),
    ):
        make_adapter(
            **{
                field_name: True,
            },
        )


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        (
            "max_image_size_bytes",
            ABSOLUTE_MAX_VISION_IMAGE_BYTES
            + 1,
        ),
        (
            "max_image_dimension_px",
            ABSOLUTE_MAX_VISION_IMAGE_DIMENSION_PX
            + 1,
        ),
        (
            "max_image_pixels",
            ABSOLUTE_MAX_VISION_IMAGE_PIXELS
            + 1,
        ),
    ],
    ids=[
        "byte-limit",
        "dimension-limit",
        "pixel-limit",
    ],
)
def test_adapter_rejects_absolute_limit_overflow(
    field_name: str,
    field_value: int,
) -> None:
    """直接调用Adapter也不能突破Schema绝对边界。"""

    with pytest.raises(
        ValueError,
        match=(
            f"{field_name}不能超过"
        ),
    ):
        make_adapter(
            **{
                field_name: field_value,
            },
        )


def test_adapter_rejects_invalid_base64(
) -> None:
    """字符和填充不合法的Base64必须在Pillow之前被拒绝。"""

    payload = VisionImagePayload(
        mime_type="image/png",
        image_base64="!!!!",
        analysis_goal="检查指示灯",
    )

    with pytest.raises(
        VisionInputValidationError,
        match="Base64不是严格合法编码",
    ):
        make_adapter().adapt(payload)


def test_empty_base64_is_rejected_before_adapter(
) -> None:
    """空图片应由外部Payload契约直接拒绝。"""

    with pytest.raises(
        ValidationError,
        # SecretStr长度约束在Pydantic 2.13中
        # 使用稳定错误类型too_short。
        match="too_short",
    ):
        VisionImagePayload(
            mime_type="image/png",
            image_base64="",
            analysis_goal="检查指示灯",
        )


def test_byte_limit_is_checked_before_pillow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """超大字节内容不得先进入Image.open进行解析。"""

    image_bytes = make_static_image_bytes()
    payload = make_payload(
        image_bytes=image_bytes,
    )

    def image_open_must_not_run(
        *args: object,
        **kwargs: object,
    ) -> None:
        """如果Pillow被提前调用就使测试立即失败。"""

        raise AssertionError(
            "超限字节不应进入Pillow"
        )

    monkeypatch.setattr(
        Image,
        "open",
        image_open_must_not_run,
    )

    adapter = make_adapter(
        max_image_size_bytes=(
            len(image_bytes) - 1
        ),
    )

    with pytest.raises(
        VisionInputValidationError,
        match="超过业务大小限制",
    ):
        adapter.adapt(payload)


def test_adapter_rejects_real_format_outside_allowlist(
) -> None:
    """BMP即使可被Pillow读取，也不属于当前Vision白名单。"""

    bmp_bytes = make_static_image_bytes(
        image_format="BMP",
    )
    payload = make_payload(
        image_bytes=bmp_bytes,
        mime_type="image/png",
    )

    with pytest.raises(
        VisionInputValidationError,
        match="真实格式不在",
    ):
        make_adapter().adapt(payload)


def test_adapter_rejects_declared_mime_mismatch(
) -> None:
    """用户声明PNG但真实字节是JPEG时必须拒绝。"""

    jpeg_bytes = make_static_image_bytes(
        image_format="JPEG",
    )
    payload = make_payload(
        image_bytes=jpeg_bytes,
        mime_type="image/png",
    )

    with pytest.raises(
        VisionInputValidationError,
        match="声明MIME类型与真实图片格式不一致",
    ):
        make_adapter().adapt(payload)


def test_adapter_rejects_dimension_limit_overflow(
) -> None:
    """任意一边超过业务上限时应在像素加载前拒绝。"""

    image_bytes = make_static_image_bytes(
        size=(11, 5),
    )
    payload = make_payload(
        image_bytes=image_bytes,
    )

    with pytest.raises(
        VisionInputValidationError,
        match="单边尺寸超过业务上限",
    ):
        make_adapter(
            max_image_dimension_px=10,
        ).adapt(payload)


def test_adapter_rejects_pixel_limit_overflow(
) -> None:
    """宽高都合法但乘积超限时仍必须拒绝。"""

    image_bytes = make_static_image_bytes(
        size=(5, 5),
    )
    payload = make_payload(
        image_bytes=image_bytes,
    )

    with pytest.raises(
        VisionInputValidationError,
        match="总像素超过业务上限",
    ):
        make_adapter(
            max_image_dimension_px=10,
            max_image_pixels=24,
        ).adapt(payload)


def test_adapter_rejects_truncated_image(
) -> None:
    """能识别JPEG头但像素数据截断时必须拒绝。"""

    complete_jpeg = make_static_image_bytes(
        image_format="JPEG",
        size=(20, 10),
    )

    # 删除JPEG末尾10字节，保留足够头部，
    # 让Pillow在verify()或load()阶段发现损坏。
    truncated_jpeg = complete_jpeg[:-10]

    payload = make_payload(
        image_bytes=truncated_jpeg,
        mime_type="image/jpeg",
    )

    with pytest.raises(
        VisionInputValidationError,
        match="内容损坏或不完整",
    ):
        make_adapter().adapt(payload)


def test_adapter_rejects_animated_webp(
) -> None:
    """MIME合法但包含多帧的WebP不属于当前静态图片契约。"""

    animated_webp = (
        make_animated_webp_bytes()
    )
    payload = make_payload(
        image_bytes=animated_webp,
        mime_type="image/webp",
    )

    with pytest.raises(
        VisionInputValidationError,
        match="只支持单帧静态图片",
    ):
        make_adapter().adapt(payload)


def test_adapter_converts_pillow_decompression_bomb_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pillow资源保护异常应转换成统一应用异常。"""

    image_bytes = make_static_image_bytes()
    payload = make_payload(
        image_bytes=image_bytes,
    )

    def raise_decompression_bomb(
        *args: object,
        **kwargs: object,
    ) -> None:
        """模拟Pillow在读取异常图片头时触发资源保护。"""

        raise Image.DecompressionBombError(
            "synthetic decompression bomb"
        )

    # 构造真实超大图片会浪费大量内存。
    # monkeypatch只替换当前测试中的Image.open，
    # 用于验证Adapter的异常转换分支。
    monkeypatch.setattr(
        Image,
        "open",
        raise_decompression_bomb,
    )

    with pytest.raises(
        VisionInputValidationError,
        match="解压缩炸弹保护",
    ):
        make_adapter().adapt(payload)


def test_adapter_does_not_change_global_pillow_pixel_limit(
) -> None:
    """局部图片检查不能修改Pillow进程级全局配置。"""

    original_max_image_pixels = (
        Image.MAX_IMAGE_PIXELS
    )
    image_bytes = make_static_image_bytes()
    payload = make_payload(
        image_bytes=image_bytes,
    )

    make_adapter().adapt(payload)

    assert Image.MAX_IMAGE_PIXELS == (
        original_max_image_pixels
    )
