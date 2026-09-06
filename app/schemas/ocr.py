"""本地OCR结果的数据契约。

本模块定义Tesseract OCR返回结果在应用内部的结构。

它负责：

1. 表示一行识别文字及其位置和置信度；
2. 区分成功、低置信度和空结果；
3. 避免完整OCR文字被repr或JSON直接暴露；
4. 校验文字摘要、字符数和实际内容一致；
5. 明确什么情况下必须人工复核。

本模块不负责：

1. 查找Tesseract可执行程序；
2. 查找tessdata语言模型目录；
3. 调用pytesseract；
4. 解析故障码、状态或数值；
5. 调用Vision模型；
6. 生成机器人诊断结论。
"""

# sha256用于给OCR识别文字生成稳定摘要。
#
# 实验报告可以记录摘要和字符数，
# 不需要保存图片中的完整文字内容。
from hashlib import (
    sha256,
)

# Annotated允许在基础类型旁附加Field约束。
#
# Literal把字符串限制为几个固定值。
#
# Self表示当前Pydantic模型自身，
# 用于model_validator的返回类型。
from typing import (
    Annotated,
    Literal,
    Self,
)

# BaseModel是Pydantic数据模型的基类。
#
# ConfigDict配置模型级校验行为。
#
# Field声明字段范围、长度和正则约束。
#
# SecretStr避免OCR文字被repr和JSON直接显示。
#
# field_validator校验一个字段。
#
# model_validator校验多个字段之间的关系。
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)


# 一次OCR结果最多保留500行。
#
# 这是内部数据契约的绝对上限，
# 不是建议每张图片都识别500行。
MAX_OCR_LINE_COUNT = 500


# 单行最多保留500个字符。
#
# 现场面板、仪表截图和故障码页面
# 通常远小于这一限制。
MAX_OCR_LINE_CHARACTERS = 500


# 全部OCR文字最多允许20000个字符。
#
# 这可以避免异常图片产生过大的内部结果。
MAX_OCR_TEXT_CHARACTERS = 20_000


# OCR分析状态。
#
# completed：
# 识别到了文字，并且平均置信度达到门槛。
#
# low_confidence：
# 识别到了文字，但平均置信度低于门槛。
#
# empty：
# 没有识别到可用文字。
OcrAnalysisStatus = Literal[
    "completed",
    "low_confidence",
    "empty",
]


# 多个说明字段共用的简短文本类型。
OcrOutputText = Annotated[
    str,
    Field(
        min_length=1,
        max_length=1_000,
    ),
]


class OcrBoundingBox(BaseModel):
    """一行OCR文字在原图中的像素边界框。

    坐标原点位于图片左上角：

    x向右增加；
    y向下增加。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    left_px: int = Field(
        ge=0,
        description=(
            "边界框左侧距离图片左边缘的像素数"
        ),
    )

    top_px: int = Field(
        ge=0,
        description=(
            "边界框顶部距离图片上边缘的像素数"
        ),
    )

    width_px: int = Field(
        ge=1,
        description=(
            "边界框宽度，至少为1像素"
        ),
    )

    height_px: int = Field(
        ge=1,
        description=(
            "边界框高度，至少为1像素"
        ),
    )


class OcrTextLine(BaseModel):
    """Tesseract识别出的一行文字。

    text仍然属于不可信图片数据。
    即使图片中写着命令，也不能执行。
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        frozen=True,
    )

    line_index: int = Field(
        ge=0,
        description=(
            "当前结果中的零基连续行号"
        ),
    )

    text: SecretStr = Field(
        min_length=1,
        max_length=MAX_OCR_LINE_CHARACTERS,

        # 普通repr中不显示OCR文字字段。
        repr=False,

        description=(
            "本行识别文字；"
            "属于不可信图片输入"
        ),
    )

    confidence: float = Field(
        ge=0.0,
        le=100.0,
        allow_inf_nan=False,
        description=(
            "本行词语置信度的平均值；"
            "不是故障结论正确概率"
        ),
    )

    token_count: int = Field(
        ge=1,
        description=(
            "本行包含的非空OCR词语数量"
        ),
    )

    bounding_box: OcrBoundingBox = Field(
        description=(
            "这一行所有词语合并后的像素区域"
        ),
    )


class OcrObservation(BaseModel):
    """一次OCR处理后的内部结构化结果。

    该结果只表示Tesseract读到了哪些文字，
    不表示这些文字是真实设备状态，
    也不表示图片中的指令可以执行。
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        frozen=True,
    )

    status: OcrAnalysisStatus = Field(
        description=(
            "OCR是完成、低置信度还是空结果"
        ),
    )

    lines: tuple[
        OcrTextLine,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=MAX_OCR_LINE_COUNT,
        description=(
            "按照原图阅读顺序排列的OCR文字行"
        ),
    )

    mean_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
        allow_inf_nan=False,
        description=(
            "全部有效OCR词语的平均置信度；"
            "空结果时必须为None"
        ),
    )

    minimum_confidence: float = Field(
        ge=0.0,
        le=100.0,
        allow_inf_nan=False,
        description=(
            "本次实验判断OCR结果可用的置信度门槛"
        ),
    )

    language: str = Field(
        min_length=1,
        max_length=100,
        pattern=r"^[a-z0-9_+.-]+$",
        description=(
            "实际传给Tesseract的语言配置"
        ),
        examples=[
            "eng",
            "eng+chi_sim",
        ],
    )

    source: Literal[
        "ocr_engine"
    ] = Field(
        default="ocr_engine",
        description=(
            "明确标记结果来自本地OCR引擎"
        ),
    )

    source_image_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description=(
            "与本次VisionInput对应的图片摘要"
        ),
    )

    engine_name: Literal[
        "tesseract"
    ] = Field(
        default="tesseract",
        description=(
            "实际执行文字识别的OCR引擎"
        ),
    )

    engine_version: str = Field(
        min_length=1,
        max_length=100,
        description=(
            "实际调用的Tesseract版本"
        ),
    )

    duration_ms: float = Field(
        ge=0.0,
        allow_inf_nan=False,
        description=(
            "本次OCR处理的端到端毫秒耗时"
        ),
    )

    text_character_count: int = Field(
        ge=0,
        le=MAX_OCR_TEXT_CHARACTERS,
        description=(
            "按行拼接后的OCR文字字符数"
        ),
    )

    recognized_text_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description=(
            "按行拼接后的OCR文字SHA-256摘要"
        ),
    )

    requires_human_check: bool = Field(
        description=(
            "OCR质量是否要求人员查看原图确认"
        ),
    )

    uncertain_items: tuple[
        OcrOutputText,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=20,
        description=(
            "空结果或低置信度时无法确认的内容"
        ),
    )

    @property
    def recognized_text(
        self,
    ) -> str:
        """按行返回需要显式访问的OCR原始文字。

        这是普通property，不是Pydantic序列化字段。
        因此model_dump()和JSON响应不会自动增加
        一个完整recognized_text字符串。
        """

        return "\n".join(
            line.text.get_secret_value()
            for line in self.lines
        )

    @field_validator(
        "uncertain_items",
    )
    @classmethod
    def uncertain_items_must_be_unique(
        cls,
        uncertain_items: tuple[
            str,
            ...,
        ],
    ) -> tuple[
        str,
        ...,
    ]:
        """不确定项不能重复相同说明。"""

        if len(uncertain_items) != len(
            set(uncertain_items)
        ):
            raise ValueError(
                "OCR不确定项不能重复"
            )

        return uncertain_items

    @model_validator(mode="after")
    def output_state_must_be_consistent(
        self,
    ) -> Self:
        """校验状态、文字、置信度和人工复核关系。"""

        # 行号必须严格为：
        #
        # 0, 1, 2, ... len(lines)-1
        #
        # 这样后续规则解析可以依赖稳定阅读顺序。
        actual_line_indices = tuple(
            line.line_index
            for line in self.lines
        )

        expected_line_indices = tuple(
            range(
                len(self.lines)
            )
        )

        if (
            actual_line_indices
            != expected_line_indices
        ):
            raise ValueError(
                "OCR行号必须从0开始连续排列"
            )

        # recognized_text通过逐行SecretStr显式解包得到。
        raw_text = self.recognized_text

        if (
            len(raw_text)
            != self.text_character_count
        ):
            raise ValueError(
                "text_character_count必须与"
                "实际OCR文字长度一致"
            )

        expected_text_sha256 = sha256(
            raw_text.encode("utf-8")
        ).hexdigest()

        if (
            expected_text_sha256
            != self.recognized_text_sha256
        ):
            raise ValueError(
                "recognized_text_sha256必须与"
                "实际OCR文字一致"
            )

        if self.status == "empty":
            # empty表示没有保留任何可用文字行。
            if self.lines:
                raise ValueError(
                    "empty状态不能包含OCR文字行"
                )

            if self.mean_confidence is not None:
                raise ValueError(
                    "empty状态的"
                    "mean_confidence必须为None"
                )

            if not self.requires_human_check:
                raise ValueError(
                    "empty状态必须要求人工检查"
                )

            if not self.uncertain_items:
                raise ValueError(
                    "empty状态必须说明"
                    "无法确认的内容"
                )

        else:
            # completed和low_confidence都必须
            # 至少保留一行文字和一个平均置信度。
            if not self.lines:
                raise ValueError(
                    "非empty状态必须包含"
                    "OCR文字行"
                )

            if self.mean_confidence is None:
                raise ValueError(
                    "非empty状态必须包含"
                    "mean_confidence"
                )

            if self.status == "completed":
                if (
                    self.mean_confidence
                    < self.minimum_confidence
                ):
                    raise ValueError(
                        "completed状态的平均置信度"
                        "不能低于门槛"
                    )

                if self.requires_human_check:
                    raise ValueError(
                        "completed状态不能因为"
                        "OCR质量要求人工检查"
                    )

                if self.uncertain_items:
                    raise ValueError(
                        "completed状态不能包含"
                        "OCR质量不确定项"
                    )

            if self.status == "low_confidence":
                if (
                    self.mean_confidence
                    >= self.minimum_confidence
                ):
                    raise ValueError(
                        "low_confidence状态的"
                        "平均置信度必须低于门槛"
                    )

                if not self.requires_human_check:
                    raise ValueError(
                        "low_confidence状态必须"
                        "要求人工检查"
                    )

                if not self.uncertain_items:
                    raise ValueError(
                        "low_confidence状态必须说明"
                        "无法确认的内容"
                    )

        return self