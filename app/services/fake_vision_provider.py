"""用于离线测试和可重复评测的Fake Vision Provider。

FakeVisionProvider完整实现VisionProvider的异步方法，
但不会访问OpenAI兼容SDK或任何外部网络。

调用方需要提前按照图片SHA-256配置：

1. VisionModelDraft：
   表示本次图片分析成功；

2. Exception：
   表示本次图片分析需要模拟失败。

Fake只接受经过VisionInputAdapter验证的VisionInput。
图片格式、大小、尺寸和完整性错误仍由输入适配器负责。
"""

from collections.abc import (
    Mapping,
)
from dataclasses import (
    dataclass,
)
from hashlib import (
    sha256,
)
import re

from app.schemas.vision import (
    VisionImageDetail,
    VisionInput,
    VisionModelDraft,
    VisionObservation,
)


# Fake模型名必须明确表明结果并非来自真实模型。
#
# 这样测试轨迹和真实轨迹不会被混淆。
FAKE_VISION_MODEL_NAME = (
    "fake-vision-provider"
)


# Fake没有真正发送真实Prompt。
#
# 因此使用独立版本名，
# 不能伪装成真实Vision Prompt版本。
FAKE_VISION_PROMPT_VERSION = (
    "fake-vision-observation-v1"
)


# 图片摘要必须是64位小写SHA-256十六进制字符串。
IMAGE_SHA256_PATTERN = re.compile(
    r"^[0-9a-f]{64}$"
)


# Fake脚本中的一项结果可以是：
#
# 1. VisionModelDraft：
#    模拟模型成功返回结构化观察；
#
# 2. Exception：
#    模拟超时、上游失败或无效响应。
FakeVisionOutcome = (
    VisionModelDraft
    | Exception
)


@dataclass(
    frozen=True,
    slots=True,
)
class FakeVisionCall:
    """Fake Provider记录的一次脱敏调用摘要。

    记录中不保存完整图片，
    也不保存完整analysis_goal。
    """

    source_image_sha256: str

    analysis_goal_sha256: str

    analysis_goal_chars: int

    detail: VisionImageDetail


class FakeVisionProvider:
    """根据图片SHA返回预设结果的异步Fake Provider。"""

    def __init__(
        self,
        *,
        scripted_results: Mapping[
            str,
            FakeVisionOutcome,
        ],
    ) -> None:
        """复制并校验调用方提供的Fake结果脚本。"""

        # Mapping是只读映射接口。
        #
        # dict、MappingProxyType等映射对象
        # 都可以作为输入。
        if not isinstance(
            scripted_results,
            Mapping,
        ):
            raise TypeError(
                "scripted_results必须是Mapping"
            )

        copied_results: dict[
            str,
            FakeVisionOutcome,
        ] = {}

        for (
            image_sha256,
            outcome,
        ) in scripted_results.items():
            # 每个脚本键都必须对应一张图片的
            # 真实小写SHA-256摘要。
            if (
                not isinstance(
                    image_sha256,
                    str,
                )
                or IMAGE_SHA256_PATTERN.fullmatch(
                    image_sha256
                )
                is None
            ):
                raise ValueError(
                    "Fake Vision脚本键必须是"
                    "64位小写SHA-256"
                )

            # Fake只允许返回合法Draft，
            # 或抛出明确的Exception。
            #
            # 普通dict、字符串或None都不是合法结果。
            if not isinstance(
                outcome,
                (
                    VisionModelDraft,
                    Exception,
                ),
            ):
                raise TypeError(
                    "Fake Vision脚本结果必须是"
                    "VisionModelDraft或Exception"
                )

            copied_results[
                image_sha256
            ] = outcome

        # 复制成新的dict，避免调用方在Fake创建后
        # 修改原始映射并改变测试行为。
        self._scripted_results = (
            copied_results
        )

        # 调用记录只存在于当前Fake对象内存中。
        self._calls: list[
            FakeVisionCall
        ] = []

    @property
    def calls(
        self,
    ) -> tuple[
        FakeVisionCall,
        ...,
    ]:
        """返回不可变的脱敏调用记录快照。"""

        # tuple()创建当前列表的不可变快照。
        #
        # 调用方不能通过返回值修改
        # Fake内部保存的_calls列表。
        return tuple(
            self._calls
        )

    @property
    def call_count(
        self,
    ) -> int:
        """返回Fake Provider已经收到的调用次数。"""

        return len(
            self._calls
        )

    def reset_calls(
        self,
    ) -> None:
        """清空调用记录，但保留预设脚本结果。"""

        self._calls.clear()

    async def analyze_image(
        self,
        *,
        vision_input: VisionInput,
    ) -> VisionObservation:
        """按照图片SHA返回结果或抛出预设异常。"""

        # 与真实Provider保持相同输入边界。
        #
        # Fake不能因为用于测试，
        # 就允许普通dict绕过VisionInputAdapter。
        if not isinstance(
            vision_input,
            VisionInput,
        ):
            raise TypeError(
                "vision_input必须是VisionInput"
            )

        image_sha256 = (
            vision_input
            .metadata
            .sha256_hex
        )

        # analysis_goal仍属于不可信输入。
        #
        # Fake调用记录只保存UTF-8摘要和字符数，
        # 不保存目标全文。
        analysis_goal_bytes = (
            vision_input
            .analysis_goal
            .encode("utf-8")
        )

        call = FakeVisionCall(
            source_image_sha256=(
                image_sha256
            ),
            analysis_goal_sha256=(
                sha256(
                    analysis_goal_bytes
                ).hexdigest()
            ),
            analysis_goal_chars=len(
                vision_input.analysis_goal
            ),
            detail=vision_input.detail,
        )

        # 即使脚本没有对应结果，
        # 也要记录这次真实发生的调用，
        # 方便测试判断工具是否进行了意外调用。
        self._calls.append(
            call
        )

        if (
            image_sha256
            not in self._scripted_results
        ):
            # 未配置结果表示测试数据或测试编排有误。
            #
            # 使用KeyError暴露测试设置错误，
            # 不把它伪装成真实模型上游故障。
            raise KeyError(
                "FakeVisionProvider没有为图片"
                f"{image_sha256}配置结果"
            )

        outcome = self._scripted_results[
            image_sha256
        ]

        # 如果脚本结果是异常，
        # 直接从VisionProvider边界抛出。
        #
        # 例如可以预设：
        # VisionTimeoutError(...)
        # VisionUpstreamError(...)
        if isinstance(
            outcome,
            Exception,
        ):
            raise outcome

        # outcome已经是经过Pydantic验证的
        # VisionModelDraft。
        #
        # Fake与真实Provider一样，
        # 由Provider层注入追踪字段。
        observation_data = {
            **outcome.model_dump(
                mode="python"
            ),
            "source": "vision_model",
            "source_image_sha256": (
                image_sha256
            ),
            "prompt_version": (
                FAKE_VISION_PROMPT_VERSION
            ),
            "model_name": (
                FAKE_VISION_MODEL_NAME
            ),
        }

        return VisionObservation.model_validate(
            observation_data
        )