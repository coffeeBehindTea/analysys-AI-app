"""多模态视觉输入的数据契约。

本模块只定义图片进入Vision Provider前的数据结构。

它不负责：
1. 解码Base64；
2. 使用Pillow读取图片；
3. 调用Vision模型；
4. 生成诊断结论；
5. 保存图片文件。

外部请求先形成VisionImagePayload；
VisionInputAdapter完成解码和图片检查后，
再构造内部VisionInput。
"""

# sha256()根据图片原始字节生成稳定摘要。
#
# 摘要用于关联同一张图片，
# 不需要在日志或响应中保存完整图片内容。
from hashlib import (
    sha256,
)

# Annotated允许在类型旁附加Field约束。
#
# Literal把字段限制为列出的固定字符串。
#
# Self表示当前Pydantic模型自身，
# 用于model_validator的返回类型。
from typing import (
    Annotated,
    Literal,
    Self,
)

# BaseModel是所有Pydantic数据模型的父类。
#
# ConfigDict配置整个模型的校验和序列化行为。
#
# Field为字段增加长度、范围、正则和描述。
#
# SecretBytes和SecretStr避免图片内容被repr()
# 或日志直接显示。
#
# field_validator校验一个字段。
#
# model_validator校验多个字段之间的关系。
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretBytes,
    SecretStr,
    field_validator,
    model_validator,
)


# 这是Schema层允许接受的绝对图片字节上限。
#
# 20 MiB只是防止有人绕过正常适配器，
# 直接构造极大的VisionInput。
#
# 后续VisionInputAdapter还会使用配置中的
# 更严格业务上限，例如5 MiB。
ABSOLUTE_MAX_VISION_IMAGE_BYTES = (
    20 * 1024 * 1024
)


# 单边尺寸的绝对上限。
#
# 16384像素不是建议上传尺寸，
# 而是Schema层最后一道防御边界。
ABSOLUTE_MAX_VISION_IMAGE_DIMENSION_PX = (
    16_384
)


# 解码后的总像素绝对上限。
#
# 文件字节较小不代表解码后的图片也小，
# 因此还要限制width * height。
ABSOLUTE_MAX_VISION_IMAGE_PIXELS = (
    40_000_000
)


# 标准Base64每3个原始字节转换成4个字符。
#
# 加2后再整除3，可以正确处理
# 最后一组不足3字节的情况。
MAX_VISION_BASE64_CHARACTERS = (
    (
        ABSOLUTE_MAX_VISION_IMAGE_BYTES
        + 2
    )
    // 3
    * 4
)


# 当前v1只允许三种适合静态视觉分析的格式。
#
# 暂不允许：
# - GIF：可能包含多帧动画；
# - SVG：可能包含脚本、外部资源或复杂XML；
# - BMP/TIFF：体积或编码复杂度较高。
SupportedVisionMimeType = Literal[
    "image/jpeg",
    "image/png",
    "image/webp",
]


# 当前外部载荷只接受标准Base64。
#
# 不接受远程URL，避免服务端自动访问
# 用户提供的任意网址。
VisionImageEncoding = Literal[
    "base64",
]


# detail表示发送给兼容Vision API时
# 希望使用的图片分析精度。
#
# 是否真正支持这些值，
# 最终仍由具体Provider决定。
VisionImageDetail = Literal[
    "auto",
    "low",
    "high",
]


# 图片字节数类型。
#
# Annotated保留底层int类型，
# 同时附加Pydantic范围约束。
VisionImageByteSize = Annotated[
    int,
    Field(
        ge=1,
        le=ABSOLUTE_MAX_VISION_IMAGE_BYTES,
        description=(
            "Base64解码后的图片字节数"
        ),
    ),
]


# 图片单边像素尺寸类型。
VisionImageDimension = Annotated[
    int,
    Field(
        ge=1,
        le=(
            ABSOLUTE_MAX_VISION_IMAGE_DIMENSION_PX
        ),
        description=(
            "经过图片解码验证的像素尺寸"
        ),
    ),
]


# 图片总像素类型。
VisionImagePixelCount = Annotated[
    int,
    Field(
        ge=1,
        le=ABSOLUTE_MAX_VISION_IMAGE_PIXELS,
        description=(
            "图片宽度乘以高度得到的总像素数"
        ),
    ),
]


class VisionImagePayload(BaseModel):
    """外部提交的未验证图片载荷。

    这个模型只能证明请求字段的基本结构正确，
    不能证明Base64一定能解码，
    也不能证明内容确实是所声明的图片格式。

    后续必须交给VisionInputAdapter处理。
    """

    model_config = ConfigDict(
        # 清除普通字符串字段的首尾空白。
        str_strip_whitespace=True,

        # 禁止用户增加remote_url、command、
        # authorization等未声明字段。
        extra="forbid",

        # 校验完成后不允许重新赋值。
        frozen=True,
    )

    mime_type: SupportedVisionMimeType = Field(
        description=(
            "用户声明的图片MIME类型；"
            "适配器仍需检查真实图片格式"
        ),
        examples=["image/png"],
    )

    encoding: VisionImageEncoding = Field(
        default="base64",
        description=(
            "图片内容的传输编码方式"
        ),
    )

    image_base64: SecretStr = Field(
        min_length=4,
        max_length=(
            MAX_VISION_BASE64_CHARACTERS
        ),

        # repr=False使模型显示时不出现该字段。
        repr=False,

        # exclude=True使model_dump()和公开序列化
        # 默认不包含完整Base64图片。
        exclude=True,

        description=(
            "不包含data URL前缀的标准Base64图片内容"
        ),
    )

    analysis_goal: str = Field(
        min_length=1,
        max_length=2_000,
        description=(
            "本次需要从图片中观察的目标；"
            "该字段属于不可信用户输入"
        ),
        examples=[
            "检查设备面板上可见的指示灯状态"
        ],
    )

    detail: VisionImageDetail = Field(
        default="auto",
        description=(
            "请求Vision Provider使用的图片细节级别"
        ),
    )

    @field_validator(
        "image_base64",

        # 必须在Pydantic执行SecretStr转换和
        # str_strip_whitespace之前检查原始输入。
        #
        # 否则末尾换行或首尾空格会先被自动删除，
        # 验证器就无法兑现“禁止任何空白字符”
        # 这一外部输入契约。
        mode="before",
    )
    @classmethod
    def image_base64_must_be_raw_content(
        cls,
        image_base64: object,
    ) -> object:
        """Base64字段不能包含Data URL或空白字符。"""

        # mode="before"通常收到外部请求中的str；
        # 直接在Python中传入SecretStr时也需要兼容。
        #
        # 其他输入类型原样交回Pydantic，
        # 由SecretStr字段自身产生标准类型错误。
        if isinstance(
            image_base64,
            SecretStr,
        ):
            # SecretStr不能直接当作普通字符串使用。
            #
            # get_secret_value()只应在确实需要处理
            # 原始内容的受控代码中调用。
            raw_value = (
                image_base64.get_secret_value()
            )
        elif isinstance(
            image_base64,
            str,
        ):
            raw_value = image_base64
        else:
            return image_base64

        # 外部请求只传Base64主体。
        #
        # Data URL由Vision Provider在发送请求前
        # 根据已验证MIME类型统一构造。
        if raw_value.lower().startswith(
            "data:"
        ):
            raise ValueError(
                "image_base64不能包含"
                "data URL前缀"
            )

        # 标准Base64主体中不允许换行、空格、
        # Tab等空白字符。
        #
        # 这可以避免不同解码器对空白字符
        # 采取不一致的容错行为。
        if any(
            character.isspace()
            for character in raw_value
        ):
            raise ValueError(
                "image_base64不能包含空白字符"
            )

        # 返回原始值，随后由Pydantic完成SecretStr转换
        # 和字段长度等标准约束。
        #
        # 当前Schema只检查载荷外形。
        #
        # Base64字符和填充是否合法，
        # 将由适配器使用validate=True严格检查。
        return image_base64


class VisionImageMetadata(BaseModel):
    """VisionInputAdapter验证后的图片元数据。

    所有字段必须根据真实解码结果生成，
    不能直接相信用户提交的尺寸、格式或大小。
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        frozen=True,
    )

    mime_type: SupportedVisionMimeType = Field(
        description=(
            "根据真实图片格式确认的MIME类型"
        ),
    )

    size_bytes: VisionImageByteSize

    width_px: VisionImageDimension

    height_px: VisionImageDimension

    pixel_count: VisionImagePixelCount

    sha256_hex: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description=(
            "图片原始字节的SHA-256小写十六进制摘要"
        ),
    )

    @model_validator(mode="after")
    def pixel_count_must_match_dimensions(
        self,
    ) -> Self:
        """pixel_count必须等于width乘以height。"""

        expected_pixel_count = (
            self.width_px
            * self.height_px
        )

        if (
            self.pixel_count
            != expected_pixel_count
        ):
            raise ValueError(
                "pixel_count必须等于"
                "width_px乘以height_px"
            )

        return self


class VisionInput(BaseModel):
    """已经通过适配器检查的内部Vision输入。

    Vision Provider只接受该模型，
    不直接接受外部VisionImagePayload。

    这样可以避免模型调用层重复实现
    Base64、格式、大小和尺寸检查。
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        frozen=True,
    )

    metadata: VisionImageMetadata = Field(
        description=(
            "根据真实图片字节生成的验证后元数据"
        ),
    )

    image_bytes: SecretBytes = Field(
        min_length=1,
        max_length=(
            ABSOLUTE_MAX_VISION_IMAGE_BYTES
        ),

        # repr中不显示图片字段。
        repr=False,

        # model_dump()、JSON响应和普通日志
        # 默认不包含完整图片字节。
        exclude=True,

        description=(
            "经过Base64解码和Pillow验证的图片字节"
        ),
    )

    analysis_goal: str = Field(
        min_length=1,
        max_length=2_000,
        description=(
            "需要Vision Provider分析的目标；"
            "仍然属于不可信用户输入"
        ),
    )

    detail: VisionImageDetail = Field(
        default="auto",
        description=(
            "经过输入契约校验的视觉细节级别"
        ),
    )

    @model_validator(mode="after")
    def bytes_must_match_metadata(
        self,
    ) -> Self:
        """图片字节必须与元数据的大小和摘要一致。"""

        raw_bytes = (
            self.image_bytes.get_secret_value()
        )

        actual_size_bytes = len(raw_bytes)

        if (
            actual_size_bytes
            != self.metadata.size_bytes
        ):
            raise ValueError(
                "image_bytes长度必须等于"
                "metadata.size_bytes"
            )

        actual_sha256_hex = (
            sha256(raw_bytes).hexdigest()
        )

        if (
            actual_sha256_hex
            != self.metadata.sha256_hex
        ):
            raise ValueError(
                "image_bytes的SHA-256必须与"
                "metadata.sha256_hex一致"
            )

        return self


# Vision分析本身的完成状态。
#
# completed表示模型返回了可用观察，
# 不表示这些观察已经成为工程事实。
#
# partial表示仍有可用观察，
# 但图片质量或可见范围导致信息不完整。
#
# unusable表示图片无法支持本次分析目标，
# Provider或Agent后续应拒答或请求补充图片。
VisionAnalysisStatus = Literal[
    "completed",
    "partial",
    "unusable",
]


# 对单项视觉观察的定性置信度。
#
# 使用有限枚举而不是任意0到1浮点数，
# 避免把视觉模型未经校准的数字
# 误解成严格统计概率。
VisionObservationConfidence = Literal[
    "low",
    "medium",
    "high",
]


# 图片对当前分析目标的可用程度。
VisionImageQuality = Literal[
    "clear",
    "limited",
    "unusable",
]


# 观察类别只描述图片中可见的信息类型，
# 不直接表达设备故障根因。
VisionObservationCategory = Literal[
    "visible_condition",
    "visible_text",
    "image_quality",
    "safety_relevant",
]


# 多个输出字段共用的简短文本类型。
#
# 统一长度限制可以防止模型返回无限长列表项，
# 同时保留足够空间描述观察和不确定性。
VisionOutputText = Annotated[
    str,
    Field(
        min_length=1,
        max_length=1_000,
    ),
]


class VisionObservationItem(BaseModel):
    """Vision模型从图片中得到的一项可见观察。

    本模型只允许描述图片中可见的状态，
    不允许通过字段结构直接宣称故障根因。
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        frozen=True,
    )

    description: VisionOutputText = Field(
        description=(
            "只描述图片中可见内容的观察文本"
        ),
    )

    category: VisionObservationCategory = Field(
        description=(
            "观察属于状态、文字、质量或安全相关内容"
        ),
    )

    confidence: VisionObservationConfidence = Field(
        description=(
            "模型对这一项可见观察的定性置信度"
        ),
    )

    region: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
        description=(
            "观察在图片中的大致区域；"
            "不要求模型虚构精确像素坐标"
        ),
        examples=[
            "设备面板右上区域",
        ],
    )


class VisionVisibleIndicator(BaseModel):
    """图片中一个可辨认指示器的表面状态。

    指示灯、仪表读数或屏幕状态可以记录在这里，
    但observed_state只表示可见状态，
    不负责解释该状态对应的工程含义。
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        frozen=True,
    )

    label: str = Field(
        min_length=1,
        max_length=200,
        description=(
            "图片中可辨认的指示器名称或标签"
        ),
        examples=[
            "NET指示灯",
        ],
    )

    observed_state: str = Field(
        min_length=1,
        max_length=500,
        description=(
            "仅记录颜色、亮灭、闪烁或可见读数等表面状态"
        ),
        examples=[
            "红色常亮",
        ],
    )

    confidence: VisionObservationConfidence

    region: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
        description=(
            "指示器在图片中的大致区域"
        ),
    )


class VisionModelDraft(BaseModel):
    """Vision模型可直接生成的结构化内容。

    这里不包含图片SHA-256、Prompt版本或模型名。
    这些字段不能相信模型自行填写，
    必须由Vision Provider根据真实调用上下文注入。
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        frozen=True,
    )

    status: VisionAnalysisStatus = Field(
        description=(
            "本次视觉分析是完整、部分完成还是不可用"
        ),
    )

    image_quality: VisionImageQuality = Field(
        description=(
            "图片对当前分析目标的可用程度"
        ),
    )

    observations: tuple[
        VisionObservationItem,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=20,
        description=(
            "图片中可以直接观察到的现象"
        ),
    )

    visible_indicators: tuple[
        VisionVisibleIndicator,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=20,
        description=(
            "图片中可以辨认的指示器及其表面状态"
        ),
    )

    uncertain_items: tuple[
        VisionOutputText,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=20,
        description=(
            "因为模糊、遮挡或证据不足而无法确认的内容"
        ),
    )

    untrusted_text_detected: bool = Field(
        default=False,
        description=(
            "是否看到了指令、网址、二维码内容或密钥样例等不可信文本"
        ),
    )

    untrusted_text_notes: tuple[
        VisionOutputText,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=10,
        description=(
            "对不可信文本性质的简短说明；"
            "不保存完整密钥、长Prompt或完整敏感内容"
        ),
    )

    requires_human_check: bool = Field(
        default=False,
        description=(
            "这些观察是否必须由人员查看原图后确认"
        ),
    )

    human_check_reasons: tuple[
        VisionOutputText,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=20,
        description=(
            "需要人工复核的具体原因"
        ),
    )

    @field_validator(
        "uncertain_items",
        "untrusted_text_notes",
        "human_check_reasons",
    )
    @classmethod
    def text_items_must_be_unique(
        cls,
        text_items: tuple[
            str,
            ...,
        ],
    ) -> tuple[
        str,
        ...,
    ]:
        """同一类说明中不能重复相同文本。"""

        if len(text_items) != len(
            set(text_items)
        ):
            raise ValueError(
                "视觉输出中的同类文本项不能重复"
            )

        return text_items

    @model_validator(mode="after")
    def output_state_must_be_consistent(
        self,
    ) -> Self:
        """状态、可见结果、不确定性和人工复核必须一致。"""

        has_visible_output = bool(
            self.observations
            or self.visible_indicators
        )

        # unusable意味着图片不能支持分析目标。
        # 此时不允许保留看似可用的观察，
        # 必须解释不确定项并要求人员补充或复核。
        if self.status == "unusable":
            if self.image_quality != "unusable":
                raise ValueError(
                    "unusable状态必须使用"
                    "unusable图片质量"
                )

            if has_visible_output:
                raise ValueError(
                    "unusable状态不能包含"
                    "可用视觉观察"
                )

            if not self.uncertain_items:
                raise ValueError(
                    "unusable状态必须说明"
                    "无法确认的内容"
                )

            if not self.requires_human_check:
                raise ValueError(
                    "unusable状态必须要求人工检查"
                )

        else:
            # completed和partial都表示至少取得了一项
            # 可供后续编排使用的视觉观察。
            if not has_visible_output:
                raise ValueError(
                    "可用视觉分析必须至少包含"
                    "一项观察或可见指示器"
                )

            if self.image_quality == "unusable":
                raise ValueError(
                    "可用视觉分析不能使用"
                    "unusable图片质量"
                )

            if (
                self.status == "partial"
                and not self.uncertain_items
            ):
                raise ValueError(
                    "partial状态必须说明"
                    "无法确认的内容"
                )

            if (
                self.status == "partial"
                and not self.requires_human_check
            ):
                raise ValueError(
                    "partial状态必须要求人工检查"
                )

        has_untrusted_text_notes = bool(
            self.untrusted_text_notes
        )

        if (
            self.untrusted_text_detected
            != has_untrusted_text_notes
        ):
            raise ValueError(
                "untrusted_text_detected必须与"
                "untrusted_text_notes保持一致"
            )

        # 图片中出现指令式或敏感文本时，
        # 视觉结果必须显式进入人工复核路径。
        if (
            self.untrusted_text_detected
            and not self.requires_human_check
        ):
            raise ValueError(
                "检测到不可信文本时必须要求人工检查"
            )

        has_human_check_reasons = bool(
            self.human_check_reasons
        )

        if (
            self.requires_human_check
            != has_human_check_reasons
        ):
            raise ValueError(
                "requires_human_check必须与"
                "human_check_reasons保持一致"
            )

        has_low_confidence_output = any(
            item.confidence == "low"
            for item in self.observations
        ) or any(
            indicator.confidence == "low"
            for indicator in (
                self.visible_indicators
            )
        )

        if (
            has_low_confidence_output
            and not self.requires_human_check
        ):
            raise ValueError(
                "低置信度视觉输出必须要求人工检查"
            )

        return self


class VisionObservation(VisionModelDraft):
    """Provider补充可信追踪信息后的最终视觉观察。

    Agent视觉工具只应返回这个模型，
    不应把未经追踪信息绑定的VisionModelDraft
    直接暴露给Agent Runner。
    """

    source: Literal[
        "vision_model"
    ] = Field(
        default="vision_model",
        description=(
            "明确标记这些内容来自视觉模型观察"
        ),
    )

    source_image_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description=(
            "与本次VisionInput关联的图片SHA-256摘要"
        ),
    )

    prompt_version: str = Field(
        min_length=1,
        max_length=200,
        description=(
            "由Provider注入的Vision Prompt版本"
        ),
        examples=[
            "robot-vision-observation-v1",
        ],
    )

    model_name: str = Field(
        min_length=1,
        max_length=200,
        description=(
            "由Provider根据真实调用配置注入的模型名"
        ),
    )
