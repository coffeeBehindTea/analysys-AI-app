"""把外部Base64图片载荷转换成经过验证的VisionInput。

本模块负责：

1. 严格解码标准Base64；
2. 在图片解析前检查原始字节大小；
3. 使用Pillow确认真实图片格式；
4. 检查声明MIME类型与真实格式是否一致；
5. 检查宽度、高度和总像素数；
6. 拒绝当前v1不支持的动态图；
7. 验证图片文件结构并强制加载像素；
8. 计算SHA-256并构造验证后的VisionInput。

本模块不负责：

1. 调用Vision模型；
2. 生成图片观察结果；
3. 保存图片文件；
4. 访问图片中的网址或二维码；
5. 解释指示灯的工程含义；
6. 生成最终机器人诊断。
"""

# b64decode把Base64文本还原成原始字节。
from base64 import (
    b64decode,
)

# Base64格式、字符或填充不合法时，
# b64decode(validate=True)会抛出binascii.Error。
from binascii import (
    Error as BinasciiError,
)

# sha256用于给验证后的图片字节生成稳定摘要。
from hashlib import (
    sha256,
)

# BytesIO把内存中的bytes包装成类似文件的对象，
# Pillow可以直接从中读取图片。
from io import (
    BytesIO,
)

# catch_warnings创建局部警告处理范围。
#
# simplefilter把Pillow的解压缩炸弹警告
# 转换成可捕获异常。
from warnings import (
    catch_warnings,
    simplefilter,
)

# Image提供open()等图片读取方法。
#
# UnidentifiedImageError表示Pillow无法识别
# 输入字节对应的图片格式。
from PIL import (
    Image,
    UnidentifiedImageError,
)

from app.errors import (
    VisionInputValidationError,
)
from app.schemas.vision import (
    ABSOLUTE_MAX_VISION_IMAGE_BYTES,
    ABSOLUTE_MAX_VISION_IMAGE_DIMENSION_PX,
    ABSOLUTE_MAX_VISION_IMAGE_PIXELS,
    SupportedVisionMimeType,
    VisionImageMetadata,
    VisionImagePayload,
    VisionInput,
)


# Pillow使用大写格式名称，
# HTTP和Data URL使用标准MIME类型。
#
# 只在这张白名单中的真实格式
# 才能进入当前v1 Vision链路。
PILLOW_FORMAT_TO_MIME_TYPE: dict[
    str,
    SupportedVisionMimeType,
] = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}


class VisionInputAdapter:
    """把未验证图片载荷转换成Provider内部输入。

    调用方必须先用VisionImagePayload完成
    外部字段级校验，再调用adapt()。

    Vision Provider只能接收本类返回的VisionInput，
    不应直接接收外部Base64字符串。
    """

    def __init__(
        self,
        *,
        max_image_size_bytes: int,
        max_image_dimension_px: int,
        max_image_pixels: int,
    ) -> None:
        """保存经过检查的图片业务限制。"""

        # Settings通常已经通过Pydantic检查这些值，
        # 但Adapter也可能在测试或脚本中被直接构造。
        #
        # 因此模块仍需保护自己的公开构造边界。
        self._max_image_size_bytes = (
            self._validate_limit(
                name="max_image_size_bytes",
                value=max_image_size_bytes,
                absolute_maximum=(
                    ABSOLUTE_MAX_VISION_IMAGE_BYTES
                ),
            )
        )

        self._max_image_dimension_px = (
            self._validate_limit(
                name="max_image_dimension_px",
                value=max_image_dimension_px,
                absolute_maximum=(
                    ABSOLUTE_MAX_VISION_IMAGE_DIMENSION_PX
                ),
            )
        )

        self._max_image_pixels = (
            self._validate_limit(
                name="max_image_pixels",
                value=max_image_pixels,
                absolute_maximum=(
                    ABSOLUTE_MAX_VISION_IMAGE_PIXELS
                ),
            )
        )

    @staticmethod
    def _validate_limit(
        *,
        name: str,
        value: int,
        absolute_maximum: int,
    ) -> int:
        """检查Adapter构造参数是合法正整数。"""

        # bool是int的子类：
        #
        # isinstance(True, int) == True
        #
        # 但True显然不应该成为图片字节上限，
        # 因此要显式排除bool。
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
        ):
            raise TypeError(
                f"{name}必须是int"
            )

        if value < 1:
            raise ValueError(
                f"{name}必须大于0"
            )

        if value > absolute_maximum:
            raise ValueError(
                f"{name}不能超过"
                f"{absolute_maximum}"
            )

        return value

    def adapt(
        self,
        payload: VisionImagePayload,
    ) -> VisionInput:
        """验证外部图片载荷并返回Provider内部输入。

        返回成功意味着：

        1. Base64严格合法；
        2. 图片字节非空且未超过业务上限；
        3. Pillow可以识别和完整加载图片；
        4. 真实格式与声明MIME一致；
        5. 尺寸和总像素未超过业务上限；
        6. 图片不是当前v1不支持的动态图；
        7. 元数据与原始图片字节保持一致。

        这不意味着图片内容中的陈述是真实的，
        也不意味着图片已经足以支持工程诊断。
        """

        if not isinstance(
            payload,
            VisionImagePayload,
        ):
            raise TypeError(
                "payload必须是"
                "VisionImagePayload"
            )

        image_bytes = self._decode_base64(
            payload
        )

        (
            actual_mime_type,
            width_px,
            height_px,
            pixel_count,
        ) = self._inspect_image(
            image_bytes
        )

        # mime_type来自外部用户声明，
        # actual_mime_type来自Pillow真实识别结果。
        #
        # 二者不一致时必须拒绝，
        # 不能只修改元数据后继续处理。
        if (
            actual_mime_type
            != payload.mime_type
        ):
            raise VisionInputValidationError(
                "图片声明MIME类型与"
                "真实图片格式不一致；"
                f"declared={payload.mime_type}；"
                f"actual={actual_mime_type}"
            )

        metadata = VisionImageMetadata(
            mime_type=actual_mime_type,
            size_bytes=len(image_bytes),
            width_px=width_px,
            height_px=height_px,
            pixel_count=pixel_count,
            sha256_hex=(
                sha256(
                    image_bytes
                ).hexdigest()
            ),
        )

        # VisionInput会再次检查图片字节长度和
        # SHA-256是否与metadata一致。
        #
        # 这是Adapter与Provider之间的最后一道
        # 数据一致性保护。
        return VisionInput(
            metadata=metadata,
            image_bytes=image_bytes,
            analysis_goal=(
                payload.analysis_goal
            ),
            detail=payload.detail,
        )

    def _decode_base64(
        self,
        payload: VisionImagePayload,
    ) -> bytes:
        """严格解码Base64并检查字节大小。"""

        raw_base64 = (
            payload.image_base64
            .get_secret_value()
        )

        try:
            # validate=True禁止Base64字母表之外的字符，
            # 并严格检查填充格式。
            image_bytes = b64decode(
                raw_base64,
                validate=True,
            )
        except (
            BinasciiError,
            ValueError,
        ) as exc:
            raise VisionInputValidationError(
                "图片Base64不是严格合法编码"
            ) from exc

        if not image_bytes:
            raise VisionInputValidationError(
                "图片Base64解码后为空"
            )

        # 必须在交给Pillow之前检查大小。
        #
        # 否则超大字节内容可能先进入图片解析器，
        # 增加内存和CPU消耗。
        if (
            len(image_bytes)
            > self._max_image_size_bytes
        ):
            raise VisionInputValidationError(
                "图片超过业务大小限制；"
                f"最大允许"
                f"{self._max_image_size_bytes}"
                "字节"
            )

        return image_bytes

    def _inspect_image(
        self,
        image_bytes: bytes,
    ) -> tuple[
        SupportedVisionMimeType,
        int,
        int,
        int,
    ]:
        """使用Pillow确认格式、尺寸和图片完整性。"""

        try:
            # Pillow对异常大图片可能先发出Warning。
            #
            # 这里只在当前代码块内把该Warning转成异常，
            # 不修改Image.MAX_IMAGE_PIXELS等进程级全局设置。
            with catch_warnings():
                simplefilter(
                    "error",
                    Image.DecompressionBombWarning,
                )

                # Image.open()主要读取图片头部，
                # 此时像素通常仍是延迟加载状态。
                with Image.open(
                    BytesIO(image_bytes)
                ) as image:
                    pillow_format = (
                        image.format
                    )

                    actual_mime_type = (
                        PILLOW_FORMAT_TO_MIME_TYPE
                        .get(
                            pillow_format or ""
                        )
                    )

                    if actual_mime_type is None:
                        raise (
                            VisionInputValidationError(
                                "图片真实格式不在"
                                "JPEG、PNG、WebP白名单中"
                            )
                        )

                    (
                        width_px,
                        height_px,
                    ) = image.size

                    pixel_count = (
                        self._validate_dimensions(
                            width_px=width_px,
                            height_px=height_px,
                        )
                    )

                    # 当前v1只接受单帧静态图片。
                    #
                    # WebP可以是动画格式，
                    # 因此不能仅根据MIME类型判断。
                    is_animated = bool(
                        getattr(
                            image,
                            "is_animated",
                            False,
                        )
                    )
                    frame_count = getattr(
                        image,
                        "n_frames",
                        1,
                    )

                    if (
                        is_animated
                        or frame_count != 1
                    ):
                        raise (
                            VisionInputValidationError(
                                "当前视觉输入只支持"
                                "单帧静态图片"
                            )
                        )

                    # verify()检查文件结构，
                    # 但不会为后续使用保留已解码像素。
                    image.verify()

                # verify()之后必须重新open()。
                #
                # load()强制读取并解码实际像素，
                # 用于发现只读取头部时无法识别的
                # 截断或损坏内容。
                with Image.open(
                    BytesIO(image_bytes)
                ) as decoded_image:
                    decoded_image.load()

        except VisionInputValidationError:
            # 本模块主动产生的业务校验异常
            # 应保持原始类型和内部原因。
            raise

        except (
            Image.DecompressionBombError,
            Image.DecompressionBombWarning,
        ) as exc:
            raise VisionInputValidationError(
                "图片触发Pillow"
                "解压缩炸弹保护"
            ) from exc

        except UnidentifiedImageError as exc:
            raise VisionInputValidationError(
                "Pillow无法识别图片格式"
            ) from exc

        except (
            OSError,
            SyntaxError,
            ValueError,
        ) as exc:
            # Pillow插件可能针对截断、损坏或非法图片
            # 抛出不同的底层异常。
            #
            # 对外统一转换成应用异常，
            # 不让底层文件解析细节泄漏到HTTP响应。
            raise VisionInputValidationError(
                "图片内容损坏或不完整"
            ) from exc

        return (
            actual_mime_type,
            width_px,
            height_px,
            pixel_count,
        )

    def _validate_dimensions(
        self,
        *,
        width_px: int,
        height_px: int,
    ) -> int:
        """检查宽高和总像素并返回pixel_count。"""

        if (
            width_px < 1
            or height_px < 1
        ):
            raise VisionInputValidationError(
                "图片宽度和高度必须大于0"
            )

        if (
            width_px
            > self._max_image_dimension_px
            or height_px
            > self._max_image_dimension_px
        ):
            raise VisionInputValidationError(
                "图片单边尺寸超过业务上限；"
                f"最大允许"
                f"{self._max_image_dimension_px}"
                "像素"
            )

        pixel_count = (
            width_px * height_px
        )

        if (
            pixel_count
            > self._max_image_pixels
        ):
            raise VisionInputValidationError(
                "图片总像素超过业务上限；"
                f"最大允许"
                f"{self._max_image_pixels}"
                "像素"
            )

        return pixel_count