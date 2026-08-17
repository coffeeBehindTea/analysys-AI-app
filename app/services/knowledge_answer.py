"""使用OpenAI兼容Chat API生成结构化证据回答。"""

# json.loads()将模型JSON字符串解析成Python对象。
import json

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
)

# ValidationError表示模型JSON不符合Pydantic契约。
from pydantic import ValidationError

from app.errors import (
    InvalidLLMResponseError,
    LLMTimeoutError,
    LLMUpstreamError,
)
from app.schemas.knowledge_query import (
    KnowledgeAnswerDraft,
)
from app.schemas.retrieval import RetrievedChunk
from app.services.knowledge_prompts import (
    build_knowledge_answer_messages,
)


class OpenAIKnowledgeAnswerProvider:
    """使用OpenAI兼容接口生成回答与证据选择。"""

    def __init__(
        self,
        *,
        client: AsyncOpenAI,
        model: str,
    ) -> None:
        """保存依赖注入的客户端和模型名称。"""

        cleaned_model = model.strip()

        if not cleaned_model:
            raise ValueError(
                "知识库回答模型名称不能为空"
            )

        self._client = client
        self._model = cleaned_model

    async def generate_answer(
        self,
        *,
        question: str,
        evidence: list[RetrievedChunk],
    ) -> KnowledgeAnswerDraft:
        """生成回答正文和模型实际使用的证据编号。"""

        messages = build_knowledge_answer_messages(
            question=question,
            evidence=evidence,
        )

        try:
            completion = (
                await self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                )
            )
        except APITimeoutError as exc:
            raise LLMTimeoutError(
                "知识库回答服务响应超时"
            ) from exc
        except APIConnectionError as exc:
            raise LLMUpstreamError(
                "无法连接知识库回答服务"
            ) from exc
        except APIStatusError as exc:
            raise LLMUpstreamError(
                "知识库回答服务返回错误状态："
                f"{exc.status_code}"
            ) from exc

        if not completion.choices:
            raise InvalidLLMResponseError(
                "知识库回答响应中没有候选结果"
            )

        content = (
            completion
            .choices[0]
            .message
            .content
        )

        return parse_knowledge_answer(content)


def parse_knowledge_answer(
    content: str | None,
) -> KnowledgeAnswerDraft:
    """解析并校验LLM返回的知识库回答JSON。"""

    if (
        not isinstance(content, str)
        or not content.strip()
    ):
        raise InvalidLLMResponseError(
            "知识库回答服务返回了空内容"
        )

    cleaned_content = remove_code_fence(
        content
    )

    try:
        # 将JSON文本解析成dict。
        raw_data = json.loads(
            cleaned_content
        )
    except json.JSONDecodeError as exc:
        raise InvalidLLMResponseError(
            "知识库回答服务返回的内容不是合法JSON"
        ) from exc

    try:
        # 检查answer、used_evidence_ids、
        # 字段类型、编号格式和重复编号。
        return KnowledgeAnswerDraft.model_validate(
            raw_data
        )
    except ValidationError as exc:
        raise InvalidLLMResponseError(
            "知识库回答JSON不符合内部响应契约"
        ) from exc


def remove_code_fence(
    content: str,
) -> str:
    """兼容模型偶尔添加的Markdown代码围栏。"""

    cleaned_content = content.strip()
    lines = cleaned_content.splitlines()

    if (
        len(lines) >= 3
        and lines[0].strip().startswith("```")
        and lines[-1].strip() == "```"
    ):
        # 去掉第一行```json和最后一行```。
        return "\n".join(
            lines[1:-1]
        ).strip()

    return cleaned_content