"""RAG对照实验使用的裸LLM回答服务。"""

# perf_counter()提供单调递增的高精度计时器，
# 适合测量一次函数调用经过了多长时间。
from time import perf_counter

# Protocol用于定义裸LLM Provider的结构契约。
#
# Fake Provider只要实现相同方法，
# 就可以在离线测试中代替真实SDK Provider。
from typing import Protocol

# AsyncOpenAI是OpenAI Python SDK的异步客户端；
# 其余类型对应不同的SDK异常。
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
)

from app.errors import (
    InvalidLLMResponseError,
    LLMTimeoutError,
    LLMUpstreamError,
)
from app.schemas.rag_comparison import (
    BareLLMResult,
)


# 这段Prompt属于实验条件，
# 后续会被原样记录到实验报告中。
#
# 裸LLM没有接收检索Chunk，
# 也没有访问知识库或实时机器人状态的工具。
BARE_LLM_SYSTEM_PROMPT = """
你是RAG对照实验中的普通语言模型回答助手。

本次没有向你提供检索文档、知识库片段、数据库结果或实时工具。

请直接回答用户问题，但必须遵守以下规则：
- 只能依据模型自身已有知识作答。
- 如果问题依赖组织内部规程、自定义故障代码或未提供的设备参数，应明确说明无法确认。
- 如果问题依赖机器人当前状态、位置、电量或其他实时遥测，应明确说明无法访问这些信息。
- 不得声称已经查看某份知识库文档、内部手册或实时系统。
- 不得虚构文件名、页码、Chunk ID、引用或数据来源。
- 不确定时应说明不确定，不得把推测写成已经确认的事实。
- 只输出回答正文，不要输出JSON或Markdown代码围栏。
""".strip()


class BareLLMProvider(Protocol):
    """裸LLM回答能力的结构契约。"""

    async def generate_answer(
        self,
        *,
        question: str,
    ) -> BareLLMResult:
        """在没有检索证据时回答一道问题。"""

        ...


def build_bare_llm_messages(
    question: str,
) -> list[dict[str, str]]:
    """把一道问题转换成裸LLM使用的Chat消息。"""

    cleaned_question = question.strip()

    if not cleaned_question:
        raise ValueError(
            "裸LLM实验问题不能为空"
        )

    return [
        {
            # System Message定义实验组不可被普通问题覆盖的规则。
            "role": "system",
            "content": BARE_LLM_SYSTEM_PROMPT,
        },
        {
            # User Message只包含问题，
            # 不包含任何检索Chunk或Gold参考答案。
            "role": "user",
            "content": cleaned_question,
        },
    ]


class OpenAIBareLLMProvider:
    """通过OpenAI兼容Chat API执行裸LLM对照回答。"""

    def __init__(
        self,
        *,
        client: AsyncOpenAI,
        model: str,
    ) -> None:
        """保存依赖注入的异步客户端和模型名。"""

        cleaned_model = model.strip()

        if not cleaned_model:
            raise ValueError(
                "裸LLM模型名称不能为空"
            )

        self._client = client
        self._model = cleaned_model

    async def generate_answer(
        self,
        *,
        question: str,
    ) -> BareLLMResult:
        """调用裸LLM并返回回答正文与生成耗时。"""

        messages = build_bare_llm_messages(
            question
        )

        # perf_counter()返回当前单调时钟值。
        #
        # 单调时钟不会因为系统时间被人工修改、
        # 网络对时或时区变化而向后跳动。
        started_at = perf_counter()

        try:
            completion = (
                await self._client
                .chat
                .completions
                .create(
                    model=self._model,
                    messages=messages,
                )
            )
        except APITimeoutError as exc:
            raise LLMTimeoutError(
                "裸LLM对照请求响应超时"
            ) from exc
        except APIConnectionError as exc:
            raise LLMUpstreamError(
                "无法连接裸LLM上游服务"
            ) from exc
        except APIStatusError as exc:
            raise LLMUpstreamError(
                "裸LLM上游服务返回错误状态："
                f"{exc.status_code}"
            ) from exc

        # 用请求结束时刻减去开始时刻，
        # 再乘1000转换成毫秒。
        generation_ms = (
            perf_counter() - started_at
        ) * 1000.0

        if not completion.choices:
            raise InvalidLLMResponseError(
                "裸LLM响应中没有候选结果"
            )

        content = (
            completion
            .choices[0]
            .message
            .content
        )

        if (
            not isinstance(content, str)
            or not content.strip()
        ):
            raise InvalidLLMResponseError(
                "裸LLM返回了空内容"
            )

        return BareLLMResult(
            # 去掉模型回答首尾多余空白，
            # 不改写模型生成的正文。
            answer=content.strip(),
            generation_ms=generation_ms,
        )