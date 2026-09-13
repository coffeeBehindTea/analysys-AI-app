"""调用 LLM、分类上游异常并将模型文本转换成稳定业务响应。"""

# json.loads() 将 LLM 返回的 JSON 字符串解析成 Python 数据。
import json

# AsyncIterator[str] 表示可以通过 async for 逐个取得字符串的异步迭代器。
from collections.abc import AsyncIterator

# 这些类型来自 OpenAI Python SDK：
# AsyncOpenAI 是异步客户端；其余三个是不同上游失败类型。
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
)

# ValidationError 表示数据无法通过 Pydantic 模型校验。
from pydantic import ValidationError

from app.errors import (
    InvalidLLMResponseError,
    LLMTimeoutError,
    LLMUpstreamError,
)
from app.schemas.triage import (
    LLMAnalysis,
    TriageRequest,
    TriageResponse,
)
from app.services.prompts import build_triage_messages


class TriageService:
    """机器人故障分诊业务服务。"""

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
    ) -> None:
        """保存依赖注入提供的异步客户端和模型名称。"""

        self._client = client
        self._model = model

    async def analyze(
        self,
        request: TriageRequest,
        request_id: str,
    ) -> TriageResponse:
        """异步调用 LLM，验证结果，并构造 API 成功响应。"""

        # 将 TriageRequest 转换成 SDK 所需的 messages 列表。
        messages = build_triage_messages(request)

        try:
            # self._client.chat 选择 Chat API；.completions 选择补全资源；
            # .create(...) 发起 POST /chat/completions 网络请求。
            # await 等待异步网络响应，completion 是 SDK 的 ChatCompletion 对象。
            completion = await self._client.chat.completions.create(
                # model 指定本次请求使用的上游模型。
                model=self._model,
                # messages 包含 system 规则和 user 数据。
                messages=messages,
            )
        except APITimeoutError as exc:
            # from exc 保留底层异常作为异常链，便于服务端调试。
            raise LLMTimeoutError("LLM 服务响应超时") from exc
        except APIConnectionError as exc:
            raise LLMUpstreamError("无法连接 LLM 服务") from exc
        except APIStatusError as exc:
            raise LLMUpstreamError(
                f"LLM 服务返回错误状态：{exc.status_code}"
            ) from exc

        # choices 是 SDK 返回的候选回答列表；为空表示上游响应无可用答案。
        if not completion.choices:
            raise InvalidLLMResponseError("LLM 响应中没有候选结果")

        # 选择第一个候选结果；message.content 通常是模型生成的文本或 None。
        content = completion.choices[0].message.content

        # 解析 JSON 并用 LLMAnalysis 做第二次结构校验。
        analysis = parse_llm_analysis(content)

        return TriageResponse(
            request_id=request_id,
            summary=analysis.summary,
            recommended_actions=analysis.recommended_actions,
        )

    async def stream_analysis(
        self,
        request: TriageRequest,
    ) -> AsyncIterator[str]:
        """逐段产生 LLM 文本，并在结束前验证拼接后的完整 JSON。"""

        messages = build_triage_messages(request)

        # 保存所有增量，流结束时会重新拼接并验证最终内容。
        # 这样只有完整结果符合 LLMAnalysis 契约时，Router 才发送 done 事件。
        content_parts: list[str] = []

        try:
            # stream=True 告诉 OpenAI 兼容接口不要等待完整回答，
            # 而是返回一个 AsyncStream[ChatCompletionChunk] 异步流对象。
            stream = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                stream=True,
            )

            # AsyncStream 是异步上下文管理器。
            # async with 退出时会关闭上游 HTTP 流，即使中途发生异常或客户端断开。
            async with stream:
                # async for 每次等待并读取一个 ChatCompletionChunk，
                # 等待网络数据时不会阻塞 Python 事件循环。
                async for chunk in stream:
                    # choices 为空时没有可用文本增量，例如某些仅含元数据的分块。
                    if not chunk.choices:
                        continue

                    # delta 表示相对于上一个分块新增的内容；
                    # content 可能是 None，因此只有非空字符串才向下游传递。
                    text = chunk.choices[0].delta.content
                    if text:
                        content_parts.append(text)
                        # yield 暂停本方法，把当前文本片段交给 Router；
                        # 下一次 async for 请求数据时再从这里继续运行。
                        yield text
        except APITimeoutError as exc:
            raise LLMTimeoutError("LLM 服务响应超时") from exc
        except APIConnectionError as exc:
            raise LLMUpstreamError("无法连接 LLM 服务") from exc
        except APIStatusError as exc:
            raise LLMUpstreamError(
                f"LLM 服务返回错误状态：{exc.status_code}"
            ) from exc

        # join() 按原顺序拼回完整模型输出。
        # parse_llm_analysis() 会检查非空、JSON 语法和业务字段；
        # 验证失败会抛出 InvalidLLMResponseError，由 Router 转成 error 事件。
        parse_llm_analysis("".join(content_parts))


def parse_llm_analysis(content: str | None) -> LLMAnalysis:
    """把可能为空的 LLM 文本解析并校验为 LLMAnalysis。"""

    # strip() 去掉首尾空白；先检查 None 可避免对 None 调用方法。
    if content is None or not content.strip():
        raise InvalidLLMResponseError("LLM 返回了空响应")

    # 某些模型会违反提示词，用 Markdown 代码围栏包裹 JSON。
    cleaned_content = remove_code_fence(content)

    try:
        # json.loads() 将 JSON 字符串变成 dict/list 等 Python 对象。
        raw_data = json.loads(cleaned_content)
    except json.JSONDecodeError as exc:
        raise InvalidLLMResponseError(
            "LLM 返回的内容不是合法 JSON"
        ) from exc

    try:
        # model_validate() 让 Pydantic 校验 raw_data 的字段、类型和长度。
        return LLMAnalysis.model_validate(raw_data)
    except ValidationError as exc:
        raise InvalidLLMResponseError(
            "LLM 返回的 JSON 不符合响应结构"
        ) from exc


def remove_code_fence(content: str) -> str:
    """移除包裹完整 JSON 的 Markdown 三反引号代码围栏。"""

    # strip() 清理整体空白；splitlines() 按行拆成 list[str]。
    cleaned = content.strip()
    lines = cleaned.splitlines()

    if (
        len(lines) >= 3
        and lines[0].strip().startswith("```")
        and lines[-1].strip() == "```"
    ):
        # lines[1:-1] 去掉第一行 ```json 和最后一行 ```；
        # join() 再把中间 JSON 行组合成字符串。
        return "\n".join(lines[1:-1]).strip()

    return cleaned
