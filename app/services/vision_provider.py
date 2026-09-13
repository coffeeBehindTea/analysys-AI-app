"""通过OpenAI兼容Chat API执行异步图片观察。

本模块负责：

1. 接收经过VisionInputAdapter验证的VisionInput；
2. 使用vision_prompts构造原生多模态消息；
3. 异步调用OpenAI兼容Vision模型；
4. 转换超时、连接和HTTP状态异常；
5. 解析并校验模型返回的VisionModelDraft；
6. 注入可信图片摘要、Prompt版本和模型名称；
7. 返回最终VisionObservation。

本模块不负责：

1. 解码外部Base64；
2. 使用Pillow验证图片；
3. 生成机器人故障根因；
4. 执行设备控制；
5. 保存完整图片或模型私有推理过程。
"""

import asyncio
import json
from math import (
    isfinite,
)
from typing import (
    Protocol,
)

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
)
from pydantic import (
    ValidationError,
)

from app.errors import (
    InvalidVisionResponseError,
    VisionTimeoutError,
    VisionUpstreamError,
)
from app.schemas.vision import (
    VisionInput,
    VisionModelDraft,
    VisionObservation,
)
from app.services.vision_prompts import (
    VISION_OBSERVATION_PROMPT_VERSION,
    build_vision_observation_messages,
)


# 视觉观察任务要求输出稳定的结构化JSON。
#
# 0.0用于减少不必要的随机表达，
# 但不表示模型输出在数学意义上绝对确定。
VISION_OBSERVATION_TEMPERATURE = 0.0


class VisionProvider(Protocol):
    """Vision Provider必须满足的异步行为契约。

    Agent工具和业务Service应依赖这个协议，
    不需要知道底层使用哪家SDK或哪个模型。

    测试中的FakeVisionProvider只要实现同名方法，
    就可以代替真实Provider。
    """

    async def analyze_image(
        self,
        *,
        vision_input: VisionInput,
    ) -> VisionObservation:
        """分析一张经过验证的图片并返回结构化观察。"""

        ...


class OpenAICompatibleVisionProvider:
    """使用OpenAI兼容Chat接口进行视觉观察。"""

    def __init__(
        self,
        *,
        client: AsyncOpenAI,
        model: str,
        timeout_seconds: float,
    ) -> None:
        """保存异步客户端、模型名称和单次调用超时。"""

        # model必须是字符串。
        #
        # 先检查类型，可以避免对None或其他对象调用strip()
        # 时产生含义不清楚的AttributeError。
        if not isinstance(
            model,
            str,
        ):
            raise TypeError(
                "Vision模型名称必须是str"
            )

        # strip()只清理模型名称两端空白，
        # 不修改模型名称内部的字符。
        cleaned_model = model.strip()

        if not cleaned_model:
            raise ValueError(
                "Vision模型名称不能为空"
            )

        # bool是int的子类：
        #
        # isinstance(True, int) == True
        #
        # 但True不应该被当作1秒超时，
        # 因此必须单独拒绝bool。
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

        # isfinite()拒绝NaN、正无穷和负无穷。
        #
        # 超时还必须大于0。
        if (
            not isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError(
                "timeout_seconds必须是"
                "大于0的有限数字"
            )

        # 客户端通过构造器注入，
        # Provider内部不读取环境变量或创建全局客户端。
        #
        # 这使测试能够注入AsyncMock，
        # 也允许后续更换兼容供应商。
        self._client = client

        # 保存清理后的真实调用模型名称。
        self._model = cleaned_model

        # 统一保存成float，
        # 例如输入60后内部保存为60.0。
        self._timeout_seconds = float(
            timeout_seconds
        )

    async def analyze_image(
        self,
        *,
        vision_input: VisionInput,
    ) -> VisionObservation:
        """调用Vision模型并返回带可信来源信息的观察。"""

        # Prompt Builder还会检查vision_input类型。
        #
        # 它会把图片字节编码成Data URL，
        # 并与分析目标一起放入多模态User Message。
        messages = (
            build_vision_observation_messages(
                vision_input
            )
        )

        try:
            # asyncio.timeout()限制整个await代码块的时间。
            #
            # 即使某个Fake客户端或兼容SDK没有正确执行
            # 自己的timeout参数，应用层超时仍然有效。
            async with asyncio.timeout(
                self._timeout_seconds
            ):
                completion = await (
                    self
                    ._client
                    .chat
                    .completions
                    .create(
                        model=self._model,
                        messages=messages,

                        # 降低结构化观察中的随机表达。
                        temperature=(
                            VISION_OBSERVATION_TEMPERATURE
                        ),

                        # 同时把超时传给SDK的HTTP请求层。
                        #
                        # asyncio.timeout负责整个协程边界，
                        # SDK timeout负责底层网络请求边界。
                        timeout=(
                            self._timeout_seconds
                        ),
                    )
                )

        # asyncio.timeout到期后，
        # 会在上下文管理器外抛出TimeoutError。
        except TimeoutError as exc:
            raise VisionTimeoutError(
                "Vision模型响应超时"
            ) from exc

        # APITimeoutError表示OpenAI SDK自己的
        # HTTP超时机制已经触发。
        except APITimeoutError as exc:
            raise VisionTimeoutError(
                "Vision模型响应超时"
            ) from exc

        # APIConnectionError表示尚未取得正常HTTP响应，
        # 例如DNS失败、连接失败或连接中断。
        except APIConnectionError as exc:
            raise VisionUpstreamError(
                "无法连接Vision模型服务"
            ) from exc

        # APIStatusError表示上游已经返回HTTP响应，
        # 但状态码是4xx或5xx。
        except APIStatusError as exc:
            raise VisionUpstreamError(
                "Vision模型服务返回错误状态："
                f"{exc.status_code}"
            ) from exc

        # 正常Chat Completion至少应包含一个候选结果。
        if not completion.choices:
            raise InvalidVisionResponseError(
                "Vision模型响应中没有候选结果"
            )

        # 当前Provider只请求一个普通非流式结果，
        # 因此读取第一个候选的assistant消息正文。
        content = (
            completion
            .choices[0]
            .message
            .content
        )

        # 将模型文字解析成VisionModelDraft。
        #
        # 这个Draft只包含模型有权生成的观察字段，
        # 不允许模型生成图片SHA、模型名和Prompt版本。
        draft = parse_vision_model_draft(
            content
        )

        # model_dump(mode="python")把Pydantic模型转换为
        # 适合继续构造另一个Pydantic模型的Python数据。
        #
        # 然后由Provider注入可信追踪字段。
        observation_data = {
            **draft.model_dump(
                mode="python"
            ),
            "source": "vision_model",
            "source_image_sha256": (
                vision_input
                .metadata
                .sha256_hex
            ),
            "prompt_version": (
                VISION_OBSERVATION_PROMPT_VERSION
            ),
            "model_name": self._model,
        }

        # VisionObservation继承VisionModelDraft，
        # 并额外要求来源、图片摘要、Prompt版本和模型名。
        #
        # 这里再次执行完整Pydantic校验，
        # 保证最终输出仍满足所有跨字段约束。
        return VisionObservation.model_validate(
            observation_data
        )


def parse_vision_model_draft(
    content: str | None,
) -> VisionModelDraft:
    """解析并校验Vision模型返回的JSON正文。"""

    # assistant message.content可能是None、
    # 空字符串或只包含空白。
    if (
        not isinstance(
            content,
            str,
        )
        or not content.strip()
    ):
        raise InvalidVisionResponseError(
            "Vision模型服务返回了空内容"
        )

    # Prompt已要求模型不要输出Markdown围栏。
    #
    # 这里仍兼容模型偶尔返回的：
    #
    # ```json
    # {...}
    # ```
    #
    # 兼容层只删除最外层围栏，
    # 不接受JSON之外的解释性文字。
    cleaned_content = (
        remove_vision_code_fence(
            content
        )
    )

    try:
        # json.loads()把JSON字符串解析成Python对象。
        raw_data = json.loads(
            cleaned_content
        )
    except json.JSONDecodeError as exc:
        raise InvalidVisionResponseError(
            "Vision模型返回的内容不是合法JSON"
        ) from exc

    try:
        # model_validate()检查：
        #
        # 1. 字段是否齐全；
        # 2. 是否存在额外字段；
        # 3. 枚举值是否合法；
        # 4. 数组长度和文本长度；
        # 5. status、图片质量和人工检查是否一致；
        # 6. 不可信文字标志和说明是否一致。
        return VisionModelDraft.model_validate(
            raw_data
        )
    except ValidationError as exc:
        raise InvalidVisionResponseError(
            "Vision模型JSON不符合内部观察契约"
        ) from exc


def remove_vision_code_fence(
    content: str,
) -> str:
    """删除包住整个Vision JSON的Markdown代码围栏。"""

    # strip()删除响应整体两端的空白。
    cleaned_content = content.strip()

    # splitlines()按行拆分，
    # 同时兼容Windows和Unix换行符。
    lines = cleaned_content.splitlines()

    if (
        len(lines) >= 3
        and lines[0].strip().startswith(
            "```"
        )
        and lines[-1].strip() == "```"
    ):
        # 第一行可能是```或```json。
        #
        # 最后一行必须是独立的```，
        # 中间所有行重新拼接成JSON正文。
        return "\n".join(
            lines[1:-1]
        ).strip()

    return cleaned_content