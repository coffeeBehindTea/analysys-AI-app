"""多模态工具选择对照实验的数据契约。

本模块定义任务二中每个脱敏图片案例必须记录的信息。

它负责：

1. 区分纯文字面板、视觉状态和退化图片；
2. 声明需要比较的处理路线；
3. 声明首选路线和仍可接受的路线；
4. 保存图片摘要，确保实验使用的是同一张图片；
5. 保存标准文字和标准视觉观察；
6. 明确案例是否应该拒答；
7. 校验案例定义内部不存在相互矛盾的字段。

本模块不负责：

1. 生成图片；
2. 读取图片文件；
3. 执行OCR；
4. 调用Vision模型；
5. 自动选择处理工具；
6. 计算准确率、延迟或成本；
7. 生成最终Markdown报告。
"""

# PurePosixPath用于校验清单中的仓库相对路径。
#
# 清单统一使用正斜杠，
# 使同一份JSONL可以在Windows和Linux中使用。
from pathlib import (
    PurePosixPath,
)

# Annotated允许在基本类型旁附加Pydantic约束。
#
# Literal把字符串字段限制在固定选项中。
#
# Self表示当前Pydantic模型自身，
# 用于model_validator的返回类型。
from typing import (
    Annotated,
    Literal,
    Self,
)

# BaseModel是Pydantic数据模型基类。
#
# ConfigDict配置模型级行为。
#
# Field声明字段范围、长度、正则和说明。
#
# field_validator负责单字段或同类字段校验。
#
# model_validator负责多个字段之间的关系校验。
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


# 当前任务二包含三类样本。
#
# text_panel：
# 主要信息是清晰的文字、故障码或数值。
#
# visual_state：
# 主要信息是颜色、亮灭、部件位置或外观状态。
#
# degraded_image：
# 图片模糊、遮挡或缺失关键区域，
# 不足以形成可靠观察。
ToolSelectionSampleKind = Literal[
    "text_panel",
    "visual_state",
    "degraded_image",
]


# 当前对照实验比较三种处理路线。
#
# ocr_rule：
# 先使用本地OCR读取文字，
# 再使用确定性规则提取故障码、数值或状态字段。
#
# vision_model：
# 把原始图片直接交给Vision Provider，
# 取得结构化可见观察。
#
# direct_abstention：
# 不调用OCR或Vision，
# 直接说明图片不足以支持回答。
ToolSelectionRoute = Literal[
    "ocr_rule",
    "vision_model",
    "direct_abstention",
]


# 实验样本只接受已经允许进入VisionInput的格式。
ToolSelectionImageSuffix = Literal[
    ".jpeg",
    ".jpg",
    ".png",
    ".webp",
]


# 多个字段共用的简短非空文字类型。
ToolSelectionText = Annotated[
    str,
    Field(
        min_length=1,
        max_length=1_000,
    ),
]


class MultimodalToolSelectionCase(
    BaseModel
):
    """一个可重复执行的多模态工具选择案例。

    这个模型记录的是实验标准答案，
    不是某一次OCR或Vision实际返回的结果。

    后续评测脚本会读取本模型，
    再分别运行OCR、Vision或拒答路线。
    """

    model_config = ConfigDict(
        # 自动清理普通字符串两端空白。
        str_strip_whitespace=True,

        # 禁止JSONL中出现未声明字段。
        #
        # 这样可以避免字段拼写错误被静默忽略。
        extra="forbid",

        # 校验完成后不能重新给字段赋值。
        frozen=True,
    )

    case_id: str = Field(
        pattern=(
            r"^multimodal-[0-9]{3}$"
        ),
        description=(
            "稳定案例编号，例如multimodal-001"
        ),
        examples=[
            "multimodal-001",
        ],
    )

    name: str = Field(
        min_length=1,
        max_length=200,
        description=(
            "供报告展示的案例名称"
        ),
    )

    sample_kind: (
        ToolSelectionSampleKind
    ) = Field(
        description=(
            "图片属于文字面板、视觉状态"
            "还是退化图片"
        ),
    )

    image_path: str = Field(
        min_length=1,
        max_length=500,
        description=(
            "从项目根目录开始的图片相对路径；"
            "统一使用正斜杠"
        ),
        examples=[
            (
                "data/multimodal/"
                "tool-selection/"
                "multimodal-001.png"
            ),
        ],
    )

    image_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description=(
            "生成后图片原始字节的SHA-256摘要；"
            "用于证明不同路线处理的是同一张图片"
        ),
    )

    generation_method: Literal[
        "synthetic_pillow"
    ] = Field(
        default="synthetic_pillow",
        description=(
            "明确样本是使用Pillow生成的"
            "教学脱敏图片"
        ),
    )

    question: str = Field(
        min_length=1,
        max_length=2_000,
        description=(
            "对这张图片提出的实验问题；"
            "属于不可信用户输入"
        ),
    )

    preferred_route: (
        ToolSelectionRoute
    ) = Field(
        description=(
            "在正确性、延迟、成本和稳定性"
            "综合考虑下的首选处理路线"
        ),
    )

    acceptable_routes: tuple[
        ToolSelectionRoute,
        ...,
    ] = Field(
        min_length=1,
        max_length=3,
        description=(
            "能够产生安全、正确结果的路线；"
            "不等于所有路线都同样经济"
        ),
    )

    routes_to_compare: tuple[
        ToolSelectionRoute,
        ...,
    ] = Field(
        min_length=2,
        max_length=3,
        description=(
            "本案例实际需要运行或评估的路线；"
            "任务二要求至少比较两条路线"
        ),
    )

    expected_text_terms: tuple[
        ToolSelectionText,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=20,
        description=(
            "正确文字路线必须保留的"
            "故障码、标签、数值或状态词"
        ),
    )

    expected_visual_observations: tuple[
        ToolSelectionText,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=20,
        description=(
            "正确视觉路线必须表达的"
            "颜色、亮灭、部件或图片质量观察"
        ),
    )

    expects_abstention: bool = Field(
        description=(
            "标准答案是否应该拒绝形成"
            "可用图片观察"
        ),
    )

    contains_untrusted_text: bool = Field(
        default=False,
        description=(
            "图片是否包含命令、网址、"
            "Prompt或密钥样例等不可信文本"
        ),
    )

    rationale: str = Field(
        min_length=1,
        max_length=2_000,
        description=(
            "为什么这个案例首选当前处理路线"
        ),
    )

    tags: tuple[
        ToolSelectionText,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=20,
        description=(
            "用于筛选和汇总案例的标签"
        ),
    )

    @field_validator(
        "image_path",
    )
    @classmethod
    def image_path_must_be_safe_relative_path(
        cls,
        image_path: str,
    ) -> str:
        """图片路径必须是安全的仓库相对路径。"""

        # 清单统一使用正斜杠。
        #
        # 如果允许反斜杠，
        # 同一清单在不同操作系统中的解释可能不同。
        if "\\" in image_path:
            raise ValueError(
                "image_path必须使用正斜杠"
            )

        parsed_path = PurePosixPath(
            image_path
        )

        if parsed_path.is_absolute():
            raise ValueError(
                "image_path必须是相对路径"
            )

        # 拒绝通过..跳出预期数据目录。
        if ".." in parsed_path.parts:
            raise ValueError(
                "image_path不能包含上级目录"
            )

        if (
            not parsed_path.parts
            or parsed_path.parts[0]
            != "data"
        ):
            raise ValueError(
                "image_path必须位于data目录"
            )

        suffix = (
            parsed_path.suffix.lower()
        )

        allowed_suffixes: tuple[
            ToolSelectionImageSuffix,
            ...,
        ] = (
            ".jpeg",
            ".jpg",
            ".png",
            ".webp",
        )

        if suffix not in allowed_suffixes:
            raise ValueError(
                "image_path图片格式不受支持"
            )

        return image_path

    @field_validator(
        "acceptable_routes",
        "routes_to_compare",
        "expected_text_terms",
        "expected_visual_observations",
        "tags",
    )
    @classmethod
    def tuple_items_must_be_unique(
        cls,
        items: tuple[
            str,
            ...,
        ],
    ) -> tuple[
        str,
        ...,
    ]:
        """同一元组字段不能包含重复项目。"""

        if len(items) != len(
            set(items)
        ):
            raise ValueError(
                "多模态案例的元组字段"
                "不能包含重复项目"
            )

        return items

    @model_validator(mode="after")
    def experiment_definition_must_be_consistent(
        self,
    ) -> Self:
        """校验样本类别、路线和标准答案之间的关系。"""

        route_set = set(
            self.routes_to_compare
        )

        acceptable_route_set = set(
            self.acceptable_routes
        )

        # 首选路线必须实际进入对照实验。
        if (
            self.preferred_route
            not in route_set
        ):
            raise ValueError(
                "preferred_route必须包含在"
                "routes_to_compare中"
            )

        # 首选路线必须能够产生安全正确结果。
        if (
            self.preferred_route
            not in acceptable_route_set
        ):
            raise ValueError(
                "preferred_route必须包含在"
                "acceptable_routes中"
            )

        # 不能声明一条没有参加实验的路线为可接受。
        if not acceptable_route_set.issubset(
            route_set
        ):
            raise ValueError(
                "acceptable_routes必须是"
                "routes_to_compare的子集"
            )

        if self.expects_abstention:
            # 应拒答案例的首选路线只能是直接拒答。
            if (
                self.preferred_route
                != "direct_abstention"
            ):
                raise ValueError(
                    "应拒答案例必须首选"
                    "direct_abstention"
                )

            # 如果图片不足以形成可靠观察，
            # 就不能再声明OCR或Vision也是可接受答案。
            if self.acceptable_routes != (
                "direct_abstention",
            ):
                raise ValueError(
                    "应拒答案例只能把"
                    "direct_abstention"
                    "列为可接受路线"
                )

            # 拒答案例不能同时声称存在
            # 可以可靠确认的文字或视觉事实。
            if (
                self.expected_text_terms
                or self
                .expected_visual_observations
            ):
                raise ValueError(
                    "应拒答案例不能包含"
                    "可确认的标准观察"
                )

        else:
            # 不需要拒答时，
            # direct_abstention不能是首选或可接受路线。
            if (
                self.preferred_route
                == "direct_abstention"
                or "direct_abstention"
                in acceptable_route_set
            ):
                raise ValueError(
                    "可回答案例不能把"
                    "direct_abstention"
                    "列为可接受路线"
                )

            # 可回答案例至少要定义一种标准信息。
            if not (
                self.expected_text_terms
                or self
                .expected_visual_observations
            ):
                raise ValueError(
                    "可回答案例必须包含"
                    "标准文字或标准视觉观察"
                )

        if self.sample_kind == "text_panel":
            # 清晰文字面板的首选路线应当是
            # 本地OCR加确定性规则。
            if (
                self.preferred_route
                != "ocr_rule"
            ):
                raise ValueError(
                    "text_panel必须首选"
                    "ocr_rule"
                )

            if not self.expected_text_terms:
                raise ValueError(
                    "text_panel必须包含"
                    "expected_text_terms"
                )

        elif self.sample_kind == "visual_state":
            # 颜色、亮灭和部件状态
            # 无法仅由普通OCR可靠判断。
            if (
                self.preferred_route
                != "vision_model"
            ):
                raise ValueError(
                    "visual_state必须首选"
                    "vision_model"
                )

            if not (
                self
                .expected_visual_observations
            ):
                raise ValueError(
                    "visual_state必须包含"
                    "expected_visual_observations"
                )

        elif self.sample_kind == (
            "degraded_image"
        ):
            if not self.expects_abstention:
                raise ValueError(
                    "degraded_image必须要求拒答"
                )

            if (
                self.preferred_route
                != "direct_abstention"
            ):
                raise ValueError(
                    "degraded_image必须首选"
                    "direct_abstention"
                )

        return self