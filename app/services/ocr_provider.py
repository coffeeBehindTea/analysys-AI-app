"""使用本地Tesseract执行异步OCR基线。

本模块负责：

1. 动态定位当前Python环境中的Tesseract引擎；
2. 动态定位tessdata语言模型目录；
3. 检查请求的OCR语言是否已经安装；
4. 在线程中执行同步pytesseract调用；
5. 校验image_to_data()返回的列；
6. 把词语按照页、块、段落和行分组；
7. 合并文字行边界框并计算置信度；
8. 返回经过Pydantic校验的OcrObservation；
9. 转换配置、超时、执行和无效输出异常。

本模块不负责：

1. 解码外部Base64；
2. 判断图片MIME类型；
3. 解释故障码的工程含义；
4. 执行图片中的命令；
5. 调用Vision模型；
6. 生成最终机器人诊断。
"""

# asyncio.to_thread()把同步阻塞函数交给工作线程，
# 避免直接阻塞FastAPI异步事件循环。
import asyncio

# os.environ用于读取可选TESSDATA_PREFIX。
import os

# re用于校验直接构造Provider时传入的语言格式。
import re

# sys.prefix表示当前Python或Conda环境根目录。
import sys

# Mapping和Sequence用于检查pytesseract返回的数据结构。
from collections.abc import (
    Mapping,
    Sequence,
)

# dataclass用于定义内部不可变运行时和词语记录。
from dataclasses import (
    dataclass,
)

# lru_cache缓存不带参数的运行时解析结果，
# 避免每次请求都重新启动Tesseract查询版本和语言。
from functools import (
    lru_cache,
)

# sha256为拼接后的OCR文字生成摘要。
from hashlib import (
    sha256,
)

# BytesIO把VisionInput中的图片bytes包装成内存文件。
from io import (
    BytesIO,
)

# isfinite用于拒绝NaN和正负无穷。
from math import (
    isfinite,
)

# Path用于跨平台组合和检查文件路径。
from pathlib import (
    Path,
)

# which按照当前PATH查找tesseract可执行程序。
from shutil import (
    which,
)

# perf_counter提供适合测量短时间间隔的高精度时钟。
from time import (
    perf_counter,
)

# Protocol定义OCR Provider的行为契约。
from typing import (
    Protocol,
)

# Pillow重新打开已经验证的图片字节，
# 并转换成Tesseract稳定支持的RGB模式。
from PIL import (
    Image,
)

# pytesseract是Python到Tesseract可执行程序的包装器。
import pytesseract

# Output.DICT要求image_to_data()返回字典和列数组。
from pytesseract import (
    Output,
)

# TesseractError表示引擎返回非零退出状态。
#
# TesseractNotFoundError表示找不到可执行程序。
from pytesseract.pytesseract import (
    TesseractError,
    TesseractNotFoundError,
)

# ValidationError用于转换内部Schema构造失败。
from pydantic import (
    ValidationError,
)

from app.errors import (
    InvalidOcrResponseError,
    OcrConfigurationError,
    OcrExecutionError,
    OcrTimeoutError,
)
from app.schemas.ocr import (
    OcrBoundingBox,
    OcrObservation,
    OcrTextLine,
)
from app.schemas.vision import (
    VisionInput,
)


# OEM 3表示让Tesseract使用默认可用OCR引擎模式。
TESSERACT_ENGINE_MODE = 3


# PSM 6表示把图片看作一个统一文字块。
#
# 任务2首轮样本是仪表截图，
# 固定该值有利于保持对照实验可重复。
TESSERACT_PAGE_SEGMENTATION_MODE = 6


# 直接构造Provider时也要保护语言格式边界，
# 不能完全依赖Settings。
OCR_LANGUAGE_PATTERN = re.compile(
    r"^[a-z0-9_+.-]+$"
)


# image_to_data()必须返回这些列。
#
# page_num、block_num、par_num和line_num
# 用于确定词语属于哪一行。
REQUIRED_TESSERACT_COLUMNS = (
    "text",
    "conf",
    "left",
    "top",
    "width",
    "height",
    "page_num",
    "block_num",
    "par_num",
    "line_num",
)


@dataclass(
    frozen=True,
    slots=True,
)
class TesseractRuntime:
    """已经解析并验证的Tesseract运行环境。

    frozen=True表示字段创建后不能重新赋值。

    slots=True避免为每个实例创建动态__dict__，
    同时阻止随意增加未声明属性。
    """

    engine_path: Path

    tessdata_directory: Path

    engine_version: str

    available_languages: tuple[
        str,
        ...,
    ]


@dataclass(
    frozen=True,
    slots=True,
)
class _OcrToken:
    """从Tesseract一行数据中解析出的内部词语。

    以下划线开头表示它只供本模块内部使用，
    不是对外公开的数据契约。
    """

    text: str

    confidence: float

    left_px: int

    top_px: int

    width_px: int

    height_px: int

    @property
    def right_px(
        self,
    ) -> int:
        """返回词语边界框右侧坐标。"""

        return (
            self.left_px
            + self.width_px
        )

    @property
    def bottom_px(
        self,
    ) -> int:
        """返回词语边界框底部坐标。"""

        return (
            self.top_px
            + self.height_px
        )


class OcrProvider(Protocol):
    """OCR Provider必须满足的异步行为契约。

    后续工具和实验代码依赖这个Protocol，
    而不是直接依赖TesseractOcrProvider。

    测试中的Fake只要实现同名异步方法，
    就可以替换真实OCR实现。
    """

    async def analyze_image(
        self,
        *,
        vision_input: VisionInput,
    ) -> OcrObservation:
        """分析已验证图片并返回结构化OCR结果。"""

        ...


@lru_cache(maxsize=1)
def resolve_tesseract_runtime(
) -> TesseractRuntime:
    """定位并验证当前进程可使用的Tesseract运行时。

    结果会在当前Python进程中缓存。
    安装或修改Tesseract后应重启应用进程。
    """

    engine_path = (
        _resolve_tesseract_engine_path()
    )

    tessdata_directory = (
        _resolve_tessdata_directory(
            engine_path=engine_path,
        )
    )

    # pytesseract使用这个模块变量确定
    # 应启动哪个Tesseract可执行程序。
    #
    # 当前应用只使用一个Tesseract运行时，
    # 因此在运行时初始化阶段统一设置。
    pytesseract.pytesseract.tesseract_cmd = (
        str(engine_path)
    )

    # pytesseract在Windows中使用
    # shlex.split(..., posix=False)拆分config。
    #
    # 如果把带双引号的tessdata路径放入config，
    # 双引号可能作为路径字符原样传给Tesseract。
    # 因此语言目录通过子进程继承的环境变量传递。
    os.environ["TESSDATA_PREFIX"] = str(
        tessdata_directory
    )

    try:
        engine_version = str(
            pytesseract.get_tesseract_version()
        )

        available_languages = tuple(
            sorted(
                pytesseract.get_languages()
            )
        )

    except TesseractNotFoundError as exc:
        raise OcrConfigurationError(
            "无法启动Tesseract可执行程序"
        ) from exc

    except TesseractError as exc:
        raise OcrConfigurationError(
            "无法读取Tesseract运行时信息"
        ) from exc

    except OSError as exc:
        raise OcrConfigurationError(
            "读取Tesseract运行时失败"
        ) from exc

    if not engine_version.strip():
        raise OcrConfigurationError(
            "Tesseract没有返回版本信息"
        )

    if not available_languages:
        raise OcrConfigurationError(
            "Tesseract没有可用语言模型"
        )

    return TesseractRuntime(
        engine_path=engine_path,
        tessdata_directory=(
            tessdata_directory
        ),
        engine_version=engine_version,
        available_languages=(
            available_languages
        ),
    )


def _resolve_tesseract_engine_path(
) -> Path:
    """按稳定顺序查找Tesseract可执行程序。"""

    candidates: list[Path] = []

    # 优先使用当前终端PATH找到的可执行程序。
    discovered_path = which(
        "tesseract"
    )

    if discovered_path is not None:
        candidates.append(
            Path(discovered_path)
        )

    # Windows Conda环境通常使用：
    #
    # <sys.prefix>/Library/bin/tesseract.exe
    candidates.append(
        Path(sys.prefix)
        / "Library"
        / "bin"
        / "tesseract.exe"
    )

    # Linux和部分Conda环境通常使用：
    #
    # <sys.prefix>/bin/tesseract
    candidates.append(
        Path(sys.prefix)
        / "bin"
        / "tesseract"
    )

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise OcrConfigurationError(
        "没有找到Tesseract可执行程序"
    )


def _resolve_tessdata_directory(
    *,
    engine_path: Path,
) -> Path:
    """查找包含.traineddata文件的语言目录。"""

    candidates: list[Path] = []

    # 如果运行环境显式提供TESSDATA_PREFIX，
    # 优先检查它。
    configured_prefix = os.environ.get(
        "TESSDATA_PREFIX"
    )

    if configured_prefix:
        candidates.append(
            Path(configured_prefix)
        )

    # 当前Windows Conda安装实际使用：
    #
    # <sys.prefix>/share/tessdata
    candidates.append(
        Path(sys.prefix)
        / "share"
        / "tessdata"
    )

    # 兼容另一种Conda布局：
    #
    # <sys.prefix>/Library/share/tessdata
    candidates.append(
        Path(sys.prefix)
        / "Library"
        / "share"
        / "tessdata"
    )

    # 根据引擎路径继续尝试相邻share目录。
    #
    # 对Library/bin/tesseract.exe而言：
    #
    # engine_path.parent.parent
    # → Library
    candidates.append(
        engine_path
        .parent
        .parent
        / "share"
        / "tessdata"
    )

    for candidate in candidates:
        if not candidate.is_dir():
            continue

        # 目录存在还不够，
        # 必须至少包含一个语言模型文件。
        if any(
            candidate.glob(
                "*.traineddata"
            )
        ):
            return candidate.resolve()

    raise OcrConfigurationError(
        "没有找到包含语言模型的"
        "Tesseract tessdata目录"
    )


def _build_tesseract_config(
) -> str:
    """构造与文件系统路径无关的固定OCR策略参数。

    tessdata目录不放入这个字符串，
    而是在运行Tesseract前通过TESSDATA_PREFIX传递。
    """

    return (
        f"--oem {TESSERACT_ENGINE_MODE} "
        f"--psm "
        f"{TESSERACT_PAGE_SEGMENTATION_MODE}"
    )


class TesseractOcrProvider:
    """使用本地Tesseract执行异步OCR。"""

    def __init__(
        self,
        *,
        language: str,
        minimum_confidence: float,
        timeout_seconds: float,
        runtime: TesseractRuntime | None = None,
    ) -> None:
        """保存并校验OCR策略和运行时。"""

        if not isinstance(
            language,
            str,
        ):
            raise TypeError(
                "OCR language必须是str"
            )

        cleaned_language = (
            language.strip()
        )

        if (
            not cleaned_language
            or OCR_LANGUAGE_PATTERN.fullmatch(
                cleaned_language
            )
            is None
        ):
            raise OcrConfigurationError(
                "OCR language格式无效"
            )

        if (
            isinstance(
                minimum_confidence,
                bool,
            )
            or not isinstance(
                minimum_confidence,
                (int, float),
            )
        ):
            raise TypeError(
                "minimum_confidence必须是数字"
            )

        if (
            not isfinite(
                minimum_confidence
            )
            or minimum_confidence < 0.0
            or minimum_confidence > 100.0
        ):
            raise ValueError(
                "minimum_confidence必须是"
                "0到100之间的有限数字"
            )

        if (
            isinstance(
                timeout_seconds,
                bool,
            )
            or not isinstance(
                timeout_seconds,
                (int, float),
            )
        ):
            raise TypeError(
                "timeout_seconds必须是数字"
            )

        if (
            not isfinite(
                timeout_seconds
            )
            or timeout_seconds <= 0.0
            or timeout_seconds > 60.0
        ):
            raise ValueError(
                "timeout_seconds必须是"
                "大于0且不超过60的有限数字"
            )

        if runtime is None:
            resolved_runtime = (
                resolve_tesseract_runtime()
            )
        elif isinstance(
            runtime,
            TesseractRuntime,
        ):
            resolved_runtime = runtime
        else:
            raise TypeError(
                "runtime必须是"
                "TesseractRuntime"
            )

        # eng+chi_sim拆分成：
        #
        # ("eng", "chi_sim")
        requested_languages = tuple(
            cleaned_language.split("+")
        )

        if len(requested_languages) != len(
            set(requested_languages)
        ):
            raise OcrConfigurationError(
                "OCR language不能重复语言代码"
            )

        missing_languages = tuple(
            sorted(
                set(requested_languages)
                - set(
                    resolved_runtime
                    .available_languages
                )
            )
        )

        if missing_languages:
            raise OcrConfigurationError(
                "Tesseract缺少请求的语言模型："
                + "、".join(
                    missing_languages
                )
            )

        self._language = (
            cleaned_language
        )
        self._minimum_confidence = float(
            minimum_confidence
        )
        self._timeout_seconds = float(
            timeout_seconds
        )
        self._runtime = (
            resolved_runtime
        )
        self._tesseract_config = (
            _build_tesseract_config()
        )

    async def analyze_image(
        self,
        *,
        vision_input: VisionInput,
    ) -> OcrObservation:
        """执行OCR并返回结构化质量结果。"""

        if not isinstance(
            vision_input,
            VisionInput,
        ):
            raise TypeError(
                "vision_input必须是VisionInput"
            )

        started_at = perf_counter()

        try:
            # pytesseract会启动同步子进程。
            #
            # asyncio.to_thread()把整个同步操作
            # 交给工作线程，当前事件循环可以继续
            # 处理其他异步任务。
            (
                lines,
                mean_confidence,
            ) = await asyncio.to_thread(
                self._run_ocr_sync,
                vision_input,
            )

        except (
            OcrConfigurationError,
            OcrTimeoutError,
            OcrExecutionError,
            InvalidOcrResponseError,
        ):
            # 已经转换过的应用异常保持原类型。
            raise

        except ValidationError as exc:
            raise InvalidOcrResponseError(
                "OCR文字行不符合内部数据契约"
            ) from exc

        duration_ms = (
            (
                perf_counter()
                - started_at
            )
            * 1_000
        )

        if not lines:
            status = "empty"
            requires_human_check = True
            uncertain_items = (
                "图片中没有识别到可用文字",
            )

        elif (
            mean_confidence is not None
            and mean_confidence
            >= self._minimum_confidence
        ):
            status = "completed"
            requires_human_check = False
            uncertain_items = ()

        else:
            status = "low_confidence"
            requires_human_check = True
            uncertain_items = (
                "OCR平均置信度低于实验门槛",
            )

        recognized_text = "\n".join(
            line.text.get_secret_value()
            for line in lines
        )

        observation_data = {
            "status": status,
            "lines": lines,
            "mean_confidence": (
                mean_confidence
            ),
            "minimum_confidence": (
                self._minimum_confidence
            ),
            "language": self._language,
            "source": "ocr_engine",
            "source_image_sha256": (
                vision_input
                .metadata
                .sha256_hex
            ),
            "engine_name": "tesseract",
            "engine_version": (
                self._runtime
                .engine_version
            ),
            "duration_ms": duration_ms,
            "text_character_count": len(
                recognized_text
            ),
            "recognized_text_sha256": (
                sha256(
                    recognized_text.encode(
                        "utf-8"
                    )
                ).hexdigest()
            ),
            "requires_human_check": (
                requires_human_check
            ),
            "uncertain_items": (
                uncertain_items
            ),
        }

        try:
            return (
                OcrObservation
                .model_validate(
                    observation_data
                )
            )
        except ValidationError as exc:
            raise InvalidOcrResponseError(
                "OCR结果不符合内部数据契约"
            ) from exc

    def _run_ocr_sync(
        self,
        vision_input: VisionInput,
    ) -> tuple[
        tuple[OcrTextLine, ...],
        float | None,
    ]:
        """在线程内同步调用Tesseract并解析结果。"""

        # 每次调用前再次设置引擎路径。
        #
        # pytesseract把它保存在模块级变量中。
        pytesseract.pytesseract.tesseract_cmd = (
            str(
                self._runtime.engine_path
            )
        )

        # Tesseract子进程从当前Python进程继承环境变量。
        #
        # 使用环境变量而不是把Windows路径拼入config，
        # 可以避免双引号被当作路径正文的一部分。
        os.environ["TESSDATA_PREFIX"] = str(
            self._runtime.tessdata_directory
        )

        image_bytes = (
            vision_input
            .image_bytes
            .get_secret_value()
        )

        try:
            # VisionInput中的图片已经由Pillow校验过。
            # 这里重新打开是为了构造Tesseract输入对象。
            with Image.open(
                BytesIO(image_bytes)
            ) as source_image:
                rgb_image = (
                    source_image.convert(
                        "RGB"
                    )
                )

                try:
                    raw_data = (
                        pytesseract
                        .image_to_data(
                            rgb_image,
                            lang=self._language,
                            config=(
                                self
                                ._tesseract_config
                            ),
                            output_type=(
                                Output.DICT
                            ),

                            # pytesseract超时后会终止
                            # Tesseract子进程并抛出
                            # RuntimeError。
                            timeout=(
                                self
                                ._timeout_seconds
                            ),
                        )
                    )
                finally:
                    rgb_image.close()

        except TesseractNotFoundError as exc:
            raise OcrConfigurationError(
                "无法启动Tesseract可执行程序"
            ) from exc

        except TesseractError as exc:
            raise OcrExecutionError(
                "Tesseract返回执行错误"
            ) from exc

        except RuntimeError as exc:
            # pytesseract目前使用RuntimeError
            # 表示子进程超时。
            if "timeout" in str(
                exc
            ).lower():
                raise OcrTimeoutError(
                    "本地OCR执行超时"
                ) from exc

            raise OcrExecutionError(
                "本地OCR执行失败"
            ) from exc

        except OSError as exc:
            raise OcrExecutionError(
                "无法读取图片或启动OCR"
            ) from exc

        return self._parse_ocr_data(
            raw_data
        )

    def _parse_ocr_data(
        self,
        raw_data: object,
    ) -> tuple[
        tuple[OcrTextLine, ...],
        float | None,
    ]:
        """校验Tesseract列并合并成文字行。"""

        if not isinstance(
            raw_data,
            Mapping,
        ):
            raise InvalidOcrResponseError(
                "Tesseract返回值必须是Mapping"
            )

        columns: dict[
            str,
            Sequence[object],
        ] = {}

        expected_length: int | None = None

        for column_name in (
            REQUIRED_TESSERACT_COLUMNS
        ):
            column = _require_column(
                raw_data=raw_data,
                column_name=column_name,
            )

            if expected_length is None:
                expected_length = len(
                    column
                )
            elif len(column) != expected_length:
                raise InvalidOcrResponseError(
                    "Tesseract返回列长度不一致"
                )

            columns[column_name] = column

        if expected_length is None:
            raise InvalidOcrResponseError(
                "Tesseract没有返回列数据"
            )

        # dict保持键的首次插入顺序。
        #
        # 因此最终文字行顺序与Tesseract
        # 输出行首次出现的顺序一致。
        grouped_tokens: dict[
            tuple[int, int, int, int],
            list[_OcrToken],
        ] = {}

        all_confidences: list[float] = []

        for row_index in range(
            expected_length
        ):
            raw_text = columns[
                "text"
            ][row_index]

            if not isinstance(
                raw_text,
                str,
            ):
                raise InvalidOcrResponseError(
                    "Tesseract text列必须包含str"
                )

            cleaned_text = (
                raw_text.strip()
            )

            # 空文字行和布局层级行不进入结果。
            if not cleaned_text:
                continue

            confidence = _parse_confidence(
                columns["conf"][row_index],
                row_index=row_index,
            )

            # Tesseract使用负置信度表示
            # 不是实际词语的布局行。
            if confidence < 0.0:
                continue

            if confidence > 100.0:
                raise InvalidOcrResponseError(
                    "Tesseract置信度不能超过100"
                )

            left_px = _parse_integer(
                columns["left"][row_index],
                field_name="left",
                row_index=row_index,
            )
            top_px = _parse_integer(
                columns["top"][row_index],
                field_name="top",
                row_index=row_index,
            )
            width_px = _parse_integer(
                columns["width"][row_index],
                field_name="width",
                row_index=row_index,
            )
            height_px = _parse_integer(
                columns["height"][row_index],
                field_name="height",
                row_index=row_index,
            )

            if left_px < 0 or top_px < 0:
                raise InvalidOcrResponseError(
                    "Tesseract坐标不能为负数"
                )

            if (
                width_px < 1
                or height_px < 1
            ):
                raise InvalidOcrResponseError(
                    "Tesseract文字框尺寸必须大于0"
                )

            grouping_key = (
                _parse_integer(
                    columns[
                        "page_num"
                    ][row_index],
                    field_name="page_num",
                    row_index=row_index,
                ),
                _parse_integer(
                    columns[
                        "block_num"
                    ][row_index],
                    field_name="block_num",
                    row_index=row_index,
                ),
                _parse_integer(
                    columns[
                        "par_num"
                    ][row_index],
                    field_name="par_num",
                    row_index=row_index,
                ),
                _parse_integer(
                    columns[
                        "line_num"
                    ][row_index],
                    field_name="line_num",
                    row_index=row_index,
                ),
            )

            token = _OcrToken(
                text=cleaned_text,
                confidence=confidence,
                left_px=left_px,
                top_px=top_px,
                width_px=width_px,
                height_px=height_px,
            )

            grouped_tokens.setdefault(
                grouping_key,
                [],
            ).append(
                token
            )

            all_confidences.append(
                confidence
            )

        if not grouped_tokens:
            return (), None

        lines: list[OcrTextLine] = []

        for (
            line_index,
            tokens,
        ) in enumerate(
            grouped_tokens.values()
        ):
            line_text = " ".join(
                token.text
                for token in tokens
            )

            line_confidence = (
                sum(
                    token.confidence
                    for token in tokens
                )
                / len(tokens)
            )

            left_px = min(
                token.left_px
                for token in tokens
            )
            top_px = min(
                token.top_px
                for token in tokens
            )
            right_px = max(
                token.right_px
                for token in tokens
            )
            bottom_px = max(
                token.bottom_px
                for token in tokens
            )

            lines.append(
                OcrTextLine(
                    line_index=line_index,
                    text=line_text,
                    confidence=(
                        line_confidence
                    ),
                    token_count=len(tokens),
                    bounding_box=(
                        OcrBoundingBox(
                            left_px=left_px,
                            top_px=top_px,
                            width_px=(
                                right_px
                                - left_px
                            ),
                            height_px=(
                                bottom_px
                                - top_px
                            ),
                        )
                    ),
                )
            )

        mean_confidence = (
            sum(all_confidences)
            / len(all_confidences)
        )

        return (
            tuple(lines),
            mean_confidence,
        )


def _require_column(
    *,
    raw_data: Mapping[
        object,
        object,
    ],
    column_name: str,
) -> Sequence[object]:
    """取得一列并拒绝字符串伪装成序列。"""

    if column_name not in raw_data:
        raise InvalidOcrResponseError(
            "Tesseract缺少必需列："
            f"{column_name}"
        )

    column = raw_data[
        column_name
    ]

    # str和bytes虽然也是Sequence，
    # 但不能作为Tesseract列数组使用。
    if (
        not isinstance(
            column,
            Sequence,
        )
        or isinstance(
            column,
            (
                str,
                bytes,
                bytearray,
            ),
        )
    ):
        raise InvalidOcrResponseError(
            "Tesseract列必须是序列："
            f"{column_name}"
        )

    return column


def _parse_integer(
    value: object,
    *,
    field_name: str,
    row_index: int,
) -> int:
    """把Tesseract整数字段转换成int。"""

    if isinstance(
        value,
        bool,
    ):
        raise InvalidOcrResponseError(
            "Tesseract整数字段不能是bool；"
            f"field={field_name}；"
            f"row={row_index}"
        )

    try:
        return int(
            str(value)
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise InvalidOcrResponseError(
            "Tesseract整数字段无效；"
            f"field={field_name}；"
            f"row={row_index}"
        ) from exc


def _parse_confidence(
    value: object,
    *,
    row_index: int,
) -> float:
    """把Tesseract置信度转换成有限float。"""

    if isinstance(
        value,
        bool,
    ):
        raise InvalidOcrResponseError(
            "Tesseract置信度不能是bool；"
            f"row={row_index}"
        )

    try:
        confidence = float(
            str(value)
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise InvalidOcrResponseError(
            "Tesseract置信度不是有效数字；"
            f"row={row_index}"
        ) from exc

    if not isfinite(
        confidence
    ):
        raise InvalidOcrResponseError(
            "Tesseract置信度必须是有限数字；"
            f"row={row_index}"
        )

    return confidence
