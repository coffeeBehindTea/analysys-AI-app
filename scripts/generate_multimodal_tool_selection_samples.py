"""生成任务二使用的六张脱敏多模态对照实验图片。

本脚本负责：

1. 使用Pillow生成两张文字面板图片；
2. 生成两张需要视觉判断的状态图片；
3. 生成一张严重模糊图片；
4. 生成一张关键区域被遮挡的图片；
5. 把图片编码成稳定PNG；
6. 计算每张图片的SHA-256；
7. 使用MultimodalToolSelectionCase校验案例；
8. 生成JSONL实验清单。

本脚本不会：

1. 使用真实机器人或客户图片；
2. 调用OCR；
3. 调用Vision API；
4. 自动访问图片中的网址；
5. 执行图片中的文字指令；
6. 在产物中记录生成时间。
"""

# Callable表示无参数图片生成函数的类型。
from collections.abc import (
    Callable,
)

# sha256为最终PNG字节生成稳定摘要。
from hashlib import (
    sha256,
)

# BytesIO让Pillow先把图片写入内存，
# 然后再取得最终PNG字节。
from io import (
    BytesIO,
)

# Path用于根据脚本位置定位项目根目录，
# 并创建数据和评测目录。
from pathlib import (
    Path,
)

# Any用于描述尚未交给Pydantic校验的案例字典。
from typing import (
    Any,
)

# Pillow负责创建、绘制、模糊和编码图片。
from PIL import (
    Image,
    ImageDraw,
    ImageFilter,
    ImageFont,
)

from app.schemas.multimodal_tool_selection import (
    MultimodalToolSelectionCase,
)


# 根据当前脚本路径确定项目根目录。
#
# 当前文件位于：
#
# <project>/scripts/generate_multimodal_tool_selection_samples.py
#
# parents[1]就是项目根目录。
PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


# 六张图片保存到独立实验目录，
# 不与知识库原始语料混合。
IMAGE_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "multimodal"
    / "tool-selection"
)


# JSONL清单放在现有data/eval目录。
MANIFEST_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "eval"
    / "multimodal_tool_selection_cases.jsonl"
)


# 所有实验图片使用相同画布尺寸，
# 避免尺寸差异干扰路线对比。
IMAGE_WIDTH_PX = 1_200
IMAGE_HEIGHT_PX = 500


# 三条实验路线的固定顺序。
ALL_COMPARISON_ROUTES = (
    "ocr_rule",
    "vision_model",
    "direct_abstention",
)


# 图片生成函数不接收参数，
# 每次返回一个新的Pillow图片对象。
ImageBuilder = Callable[
    [],
    Image.Image,
]


def load_font(
    *,
    size: int,
    bold: bool = False,
) -> ImageFont.FreeTypeFont:
    """加载可重复使用的TrueType字体。

    优先使用Windows Arial；
    如果当前系统没有Arial，则尝试DejaVu Sans。

    如果所有字体均不可用，
    直接失败而不是静默使用过小的默认位图字体。
    """

    if bold:
        font_candidates = (
            (
                "C:/Windows/Fonts/"
                "arialbd.ttf"
            ),
            "DejaVuSans-Bold.ttf",
        )
    else:
        font_candidates = (
            (
                "C:/Windows/Fonts/"
                "arial.ttf"
            ),
            "DejaVuSans.ttf",
        )

    for font_candidate in (
        font_candidates
    ):
        try:
            return ImageFont.truetype(
                font_candidate,
                size=size,
            )
        except OSError:
            # 当前候选字体不存在时继续尝试。
            continue

    raise RuntimeError(
        "没有找到生成实验图片所需的字体"
    )


def create_panel_canvas(
    *,
    title: str,
) -> Image.Image:
    """创建统一风格的模拟设备面板。"""

    image = Image.new(
        "RGB",
        (
            IMAGE_WIDTH_PX,
            IMAGE_HEIGHT_PX,
        ),
        color=(226, 232, 240),
    )

    draw = ImageDraw.Draw(image)

    # 外部深色面板边框。
    draw.rounded_rectangle(
        (
            35,
            35,
            IMAGE_WIDTH_PX - 35,
            IMAGE_HEIGHT_PX - 35,
        ),
        radius=28,
        fill=(30, 41, 59),
        outline=(100, 116, 139),
        width=4,
    )

    # 面板标题。
    draw.text(
        (75, 60),
        title,
        fill=(241, 245, 249),
        font=load_font(
            size=42,
            bold=True,
        ),
    )

    # 明确标记为教学模拟样本。
    draw.text(
        (
            75,
            IMAGE_HEIGHT_PX - 78,
        ),
        "SYNTHETIC TRAINING SAMPLE",
        fill=(148, 163, 184),
        font=load_font(
            size=24,
        ),
    )

    return image


def draw_key_value_rows(
    *,
    image: Image.Image,
    rows: tuple[
        tuple[str, str],
        ...,
    ],
) -> None:
    """在模拟面板上绘制对齐的标签和值。"""

    draw = ImageDraw.Draw(image)

    label_font = load_font(
        size=44,
        bold=True,
    )
    value_font = load_font(
        size=48,
    )

    first_row_top = 145
    row_spacing = 82

    for (
        row_index,
        (
            label,
            value,
        ),
    ) in enumerate(rows):
        row_top = (
            first_row_top
            + row_index
            * row_spacing
        )

        draw.text(
            (90, row_top),
            label,
            fill=(148, 163, 184),
            font=label_font,
        )

        draw.text(
            (490, row_top),
            value,
            fill=(248, 250, 252),
            font=value_font,
        )


def build_clear_fault_panel(
) -> Image.Image:
    """生成清晰故障码与任务状态面板。"""

    image = create_panel_canvas(
        title="ROBOT STATUS PANEL"
    )

    draw_key_value_rows(
        image=image,
        rows=(
            (
                "FAULT CODE",
                "ERR-NET-4001",
            ),
            (
                "NETWORK",
                "CONNECTED",
            ),
            (
                "TASK",
                "PAUSED",
            ),
        ),
    )

    return image


def build_clear_numeric_panel(
) -> Image.Image:
    """生成清晰电量、速度和温度面板。"""

    image = create_panel_canvas(
        title="ROBOT TELEMETRY PANEL"
    )

    draw_key_value_rows(
        image=image,
        rows=(
            (
                "BATTERY",
                "42.5 %",
            ),
            (
                "SPEED",
                "0.0 m/s",
            ),
            (
                "TEMPERATURE",
                "36 C",
            ),
        ),
    )

    return image


def draw_indicator(
    *,
    draw: ImageDraw.ImageDraw,
    center_x: int,
    center_y: int,
    color: tuple[
        int,
        int,
        int,
    ],
    label: str,
) -> None:
    """绘制带标签的圆形模拟指示灯。"""

    # 外环表示指示灯安装座。
    draw.ellipse(
        (
            center_x - 70,
            center_y - 70,
            center_x + 70,
            center_y + 70,
        ),
        fill=(15, 23, 42),
        outline=(203, 213, 225),
        width=5,
    )

    # 内部彩色圆表示当前可见灯色。
    draw.ellipse(
        (
            center_x - 50,
            center_y - 50,
            center_x + 50,
            center_y + 50,
        ),
        fill=color,
        outline=(255, 255, 255),
        width=3,
    )

    # 标签只说明灯的名称，
    # 不使用文字直接说明颜色。
    label_font = load_font(
        size=34,
        bold=True,
    )

    label_box = draw.textbbox(
        (0, 0),
        label,
        font=label_font,
    )

    label_width = (
        label_box[2]
        - label_box[0]
    )

    draw.text(
        (
            center_x
            - label_width // 2,
            center_y + 92,
        ),
        label,
        fill=(241, 245, 249),
        font=label_font,
    )


def build_indicator_panel(
) -> Image.Image:
    """生成需要判断颜色的指示灯面板。"""

    image = create_panel_canvas(
        title="ROBOT INDICATOR PANEL"
    )

    draw = ImageDraw.Draw(image)

    draw_indicator(
        draw=draw,
        center_x=300,
        center_y=260,
        color=(34, 197, 94),
        label="NET",
    )

    draw_indicator(
        draw=draw,
        center_x=600,
        center_y=260,
        color=(239, 68, 68),
        label="FAULT",
    )

    draw_indicator(
        draw=draw,
        center_x=900,
        center_y=260,
        color=(59, 130, 246),
        label="POWER",
    )

    return image


def build_connector_gap_panel(
) -> Image.Image:
    """生成插头与插座之间存在明显间隙的示意图。"""

    image = create_panel_canvas(
        title="SENSOR CONNECTOR VIEW"
    )

    draw = ImageDraw.Draw(image)

    # 插座主体。
    draw.rounded_rectangle(
        (
            180,
            165,
            435,
            350,
        ),
        radius=25,
        fill=(71, 85, 105),
        outline=(203, 213, 225),
        width=5,
    )

    # 插座接口。
    draw.ellipse(
        (
            350,
            205,
            445,
            310,
        ),
        fill=(15, 23, 42),
        outline=(226, 232, 240),
        width=5,
    )

    # 插头与插座之间保留明显空隙。
    draw.ellipse(
        (
            555,
            205,
            650,
            310,
        ),
        fill=(100, 116, 139),
        outline=(241, 245, 249),
        width=5,
    )

    # 插头主体。
    draw.rounded_rectangle(
        (
            620,
            180,
            865,
            335,
        ),
        radius=24,
        fill=(51, 65, 85),
        outline=(203, 213, 225),
        width=5,
    )

    # 锁紧环。
    draw.rectangle(
        (
            605,
            190,
            675,
            325,
        ),
        fill=(148, 163, 184),
        outline=(241, 245, 249),
        width=4,
    )

    # 电缆。
    draw.line(
        (
            865,
            257,
            1_080,
            257,
        ),
        fill=(17, 24, 39),
        width=35,
    )

    # 中心虚线只表示理想连接方向，
    # 不写“断开”等答案文字。
    for dash_left in range(
        450,
        550,
        22,
    ):
        draw.line(
            (
                dash_left,
                257,
                dash_left + 10,
                257,
            ),
            fill=(250, 204, 21),
            width=4,
        )

    label_font = load_font(
        size=30,
        bold=True,
    )

    draw.text(
        (240, 370),
        "PORT J3",
        fill=(203, 213, 225),
        font=label_font,
    )

    draw.text(
        (675, 370),
        "SENSOR CABLE",
        fill=(203, 213, 225),
        font=label_font,
    )

    return image


def build_severely_blurred_panel(
) -> Image.Image:
    """生成无法可靠读取文字的严重模糊图片。"""

    source_image = (
        build_clear_fault_panel()
    )

    # 先大幅缩小，再放大，破坏文字细节。
    small_image = source_image.resize(
        (120, 50),
        resample=(
            Image.Resampling.BILINEAR
        ),
    )

    enlarged_image = small_image.resize(
        (
            IMAGE_WIDTH_PX,
            IMAGE_HEIGHT_PX,
        ),
        resample=(
            Image.Resampling.BILINEAR
        ),
    )

    # 再应用高斯模糊，
    # 使文字和关键状态都不足以可靠确认。
    blurred_image = (
        enlarged_image.filter(
            ImageFilter.GaussianBlur(
                radius=12
            )
        )
    )

    source_image.close()
    small_image.close()
    enlarged_image.close()

    return blurred_image


def build_occluded_panel(
) -> Image.Image:
    """生成关键值区域完全被遮挡的面板。"""

    image = create_panel_canvas(
        title="ROBOT STATUS PANEL"
    )

    draw_key_value_rows(
        image=image,
        rows=(
            (
                "FAULT CODE",
                "ERR-SAF-1002",
            ),
            (
                "BATTERY",
                "18.0 %",
            ),
            (
                "TASK",
                "STOPPED",
            ),
        ),
    )

    draw = ImageDraw.Draw(image)

    # 完全覆盖三个值，但保留左侧标签。
    #
    # 后续系统可以知道图片包含这些字段，
    # 但不能知道被遮挡的具体值。
    draw.rounded_rectangle(
        (
            470,
            130,
            1_080,
            390,
        ),
        radius=18,
        fill=(3, 7, 18),
        outline=(15, 23, 42),
        width=4,
    )

    return image


def encode_png(
    image: Image.Image,
) -> bytes:
    """把图片编码成稳定PNG字节。"""

    output_buffer = BytesIO()

    image.save(
        output_buffer,
        format="PNG",

        # 明确固定压缩参数，
        # 避免脚本内不同调用使用不同策略。
        optimize=False,
        compress_level=9,
    )

    return output_buffer.getvalue()


def write_image(
    *,
    case_id: str,
    image: Image.Image,
) -> tuple[
    str,
    str,
]:
    """写入PNG并返回相对路径和SHA-256。"""

    image_bytes = encode_png(
        image
    )

    output_path = (
        IMAGE_OUTPUT_DIRECTORY
        / f"{case_id}.png"
    )

    output_path.write_bytes(
        image_bytes
    )

    relative_path = (
        output_path
        .relative_to(PROJECT_ROOT)
        .as_posix()
    )

    image_sha256 = sha256(
        image_bytes
    ).hexdigest()

    return (
        relative_path,
        image_sha256,
    )


def build_case(
    *,
    case_id: str,
    image_builder: ImageBuilder,
    case_data: dict[
        str,
        Any,
    ],
) -> MultimodalToolSelectionCase:
    """生成图片并构造经过校验的案例。"""

    image = image_builder()

    try:
        (
            image_path,
            image_sha256,
        ) = write_image(
            case_id=case_id,
            image=image,
        )
    finally:
        # 无论图片写入或Schema校验是否成功，
        # 都关闭Pillow图片对象。
        image.close()

    complete_case_data = {
        "case_id": case_id,
        "image_path": image_path,
        "image_sha256": image_sha256,
        "generation_method": (
            "synthetic_pillow"
        ),
        **case_data,
    }

    return (
        MultimodalToolSelectionCase
        .model_validate(
            complete_case_data
        )
    )


def build_all_cases(
) -> tuple[
    MultimodalToolSelectionCase,
    ...,
]:
    """生成六张图片及其标准案例。"""

    common_routes = (
        ALL_COMPARISON_ROUTES
    )

    cases = (
        build_case(
            case_id="multimodal-001",
            image_builder=(
                build_clear_fault_panel
            ),
            case_data={
                "name": (
                    "清晰故障码与任务状态面板"
                ),
                "sample_kind": (
                    "text_panel"
                ),
                "question": (
                    "图片显示的故障码、"
                    "网络状态和任务状态是什么？"
                ),
                "preferred_route": (
                    "ocr_rule"
                ),
                "acceptable_routes": (
                    "ocr_rule",
                    "vision_model",
                ),
                "routes_to_compare": (
                    common_routes
                ),
                "expected_text_terms": (
                    "ERR-NET-4001",
                    "CONNECTED",
                    "PAUSED",
                ),
                "expects_abstention": False,
                "contains_untrusted_text": False,
                "rationale": (
                    "关键信息全部是清晰文字；"
                    "OCR成本低、速度快且结果可重复。"
                ),
                "tags": (
                    "text_panel",
                    "fault_code",
                    "task_state",
                ),
            },
        ),
        build_case(
            case_id="multimodal-002",
            image_builder=(
                build_clear_numeric_panel
            ),
            case_data={
                "name": (
                    "清晰遥测数值面板"
                ),
                "sample_kind": (
                    "text_panel"
                ),
                "question": (
                    "图片显示的电量、速度"
                    "和温度分别是多少？"
                ),
                "preferred_route": (
                    "ocr_rule"
                ),
                "acceptable_routes": (
                    "ocr_rule",
                    "vision_model",
                ),
                "routes_to_compare": (
                    common_routes
                ),
                "expected_text_terms": (
                    "42.5 %",
                    "0.0 m/s",
                    "36 C",
                ),
                "expects_abstention": False,
                "contains_untrusted_text": False,
                "rationale": (
                    "问题只要求读取清晰数值，"
                    "无需使用视觉模型解释画面。"
                ),
                "tags": (
                    "text_panel",
                    "numeric",
                    "telemetry",
                ),
            },
        ),
        build_case(
            case_id="multimodal-003",
            image_builder=(
                build_indicator_panel
            ),
            case_data={
                "name": (
                    "多色机器人指示灯面板"
                ),
                "sample_kind": (
                    "visual_state"
                ),
                "question": (
                    "NET、FAULT和POWER"
                    "指示灯分别呈现什么颜色？"
                ),
                "preferred_route": (
                    "vision_model"
                ),
                "acceptable_routes": (
                    "vision_model",
                ),
                "routes_to_compare": (
                    common_routes
                ),
                "expected_visual_observations": (
                    "NET指示灯呈绿色亮起",
                    "FAULT指示灯呈红色亮起",
                    "POWER指示灯呈蓝色亮起",
                ),
                "expects_abstention": False,
                "contains_untrusted_text": False,
                "rationale": (
                    "OCR只能读取标签，"
                    "不能可靠判断圆形指示灯的颜色。"
                ),
                "tags": (
                    "visual_state",
                    "indicator",
                    "color",
                ),
            },
        ),
        build_case(
            case_id="multimodal-004",
            image_builder=(
                build_connector_gap_panel
            ),
            case_data={
                "name": (
                    "传感器连接器未贴合示意图"
                ),
                "sample_kind": (
                    "visual_state"
                ),
                "question": (
                    "图中的传感器插头是否"
                    "已经与J3插座完全贴合？"
                ),
                "preferred_route": (
                    "vision_model"
                ),
                "acceptable_routes": (
                    "vision_model",
                ),
                "routes_to_compare": (
                    common_routes
                ),
                "expected_visual_observations": (
                    "插头与J3插座之间存在明显间隙",
                    "锁紧环没有贴合插座",
                ),
                "expects_abstention": False,
                "contains_untrusted_text": False,
                "rationale": (
                    "连接间隙和部件位置属于"
                    "空间视觉状态，普通OCR无法判断。"
                ),
                "tags": (
                    "visual_state",
                    "connector",
                    "physical_gap",
                ),
            },
        ),
        build_case(
            case_id="multimodal-005",
            image_builder=(
                build_severely_blurred_panel
            ),
            case_data={
                "name": (
                    "严重模糊的状态面板"
                ),
                "sample_kind": (
                    "degraded_image"
                ),
                "question": (
                    "图片中的故障码、网络状态"
                    "和任务状态是什么？"
                ),
                "preferred_route": (
                    "direct_abstention"
                ),
                "acceptable_routes": (
                    "direct_abstention",
                ),
                "routes_to_compare": (
                    common_routes
                ),
                "expects_abstention": True,
                "contains_untrusted_text": False,
                "rationale": (
                    "文字和面板细节严重模糊，"
                    "OCR或Vision猜测都会产生风险。"
                ),
                "tags": (
                    "degraded_image",
                    "blur",
                    "abstention",
                ),
            },
        ),
        build_case(
            case_id="multimodal-006",
            image_builder=(
                build_occluded_panel
            ),
            case_data={
                "name": (
                    "关键数值完全遮挡的状态面板"
                ),
                "sample_kind": (
                    "degraded_image"
                ),
                "question": (
                    "图片显示的故障码、"
                    "电量和任务状态分别是什么？"
                ),
                "preferred_route": (
                    "direct_abstention"
                ),
                "acceptable_routes": (
                    "direct_abstention",
                ),
                "routes_to_compare": (
                    common_routes
                ),
                "expects_abstention": True,
                "contains_untrusted_text": False,
                "rationale": (
                    "标签仍然可见，但问题要求的"
                    "具体值被完全遮挡，不能推断。"
                ),
                "tags": (
                    "degraded_image",
                    "occlusion",
                    "abstention",
                ),
            },
        ),
    )

    return cases


def write_manifest(
    cases: tuple[
        MultimodalToolSelectionCase,
        ...,
    ],
) -> None:
    """把案例写成UTF-8 JSONL文件。"""

    json_lines = tuple(
        case.model_dump_json()
        for case in cases
    )

    manifest_text = (
        "\n".join(json_lines)
        + "\n"
    )

    # newline参数固定使用LF，
    # 使JSONL在Git中保持稳定。
    with MANIFEST_OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as output_file:
        output_file.write(
            manifest_text
        )


def main() -> None:
    """创建目录、生成样本并输出摘要。"""

    IMAGE_OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    MANIFEST_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cases = build_all_cases()

    write_manifest(
        cases
    )

    print(
        "多模态工具选择样本生成完成"
    )
    print(
        f"案例数：{len(cases)}"
    )
    print(
        "文字面板数："
        f"{sum(case.sample_kind == 'text_panel' for case in cases)}"
    )
    print(
        "视觉状态数："
        f"{sum(case.sample_kind == 'visual_state' for case in cases)}"
    )
    print(
        "退化图片数："
        f"{sum(case.sample_kind == 'degraded_image' for case in cases)}"
    )
    print(
        "图片目录："
        f"{IMAGE_OUTPUT_DIRECTORY}"
    )
    print(
        "案例清单："
        f"{MANIFEST_OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()