"""调用OpenAI兼容LLM生成结构化诊断草稿。

本模块负责：

1. 调用诊断Prompt Builder；
2. 通过异步OpenAI兼容接口发送请求；
3. 将SDK异常转换成应用层异常；
4. 解析模型返回的JSON；
5. 使用DiagnosisLLMDraft执行结构校验。

本模块不执行检索、证据门控、真实Chunk映射，
也不生成最终DiagnosisReport。
"""

# json.loads()把模型返回的JSON字符串
# 解析成Python对象。
import json

# 这些类来自OpenAI Python SDK：
#
# AsyncOpenAI：
# 异步OpenAI兼容客户端。
#
# APITimeoutError：
# SDK请求超过配置的等待时间。
#
# APIConnectionError：
# DNS、网络连接或TLS连接等失败。
#
# APIStatusError：
# 上游返回4xx或5xx状态码。
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
)

# ValidationError表示数据没有通过
# Pydantic模型的数据契约。
from pydantic import ValidationError

from app.errors import (
    InvalidLLMResponseError,
    LLMTimeoutError,
    LLMUpstreamError,
)
from app.schemas.diagnosis_llm import (
    DiagnosisLLMDraft,
)
from app.schemas.diagnostics import (
    DiagnosisRequest,
)
from app.schemas.retrieval import (
    HybridRetrievedChunk,
)
from app.services.diagnosis_prompts import (
    EVIDENCE_DIAGNOSIS_PROMPT_VARIANT,
    DiagnosisPromptVariant,
    build_diagnosis_messages,
)


# temperature控制模型采样的随机程度。
#
# 结构化诊断更重视稳定性和可复现性，
# 因而使用0.0。
#
# 但temperature=0.0不代表数学意义上的
# 绝对确定；上游模型版本或基础设施变化
# 仍可能使结果发生变化。
DIAGNOSIS_TEMPERATURE = 0.0

# 最多把10项Pydantic校验错误写入内部日志。
#
# 设置上限可以避免模型一次返回大量错误字段时，
# 产生过长日志或造成日志放大。
MAX_DIAGNOSIS_VALIDATION_ERRORS = 10


def _format_diagnosis_validation_errors(
    error: ValidationError,
) -> str:
    """把Pydantic校验错误转换成安全、简洁的内部日志摘要。"""

    # ValidationError.errors()返回一个错误字典列表。
    #
    # 每个错误通常包含：
    #
    # loc：错误字段的位置；
    # type：错误类型，例如missing或value_error；
    # msg：便于阅读的错误说明；
    # input：导致错误的原始输入；
    # ctx：校验器附加的上下文；
    # url：Pydantic错误帮助网址。
    #
    # 日志只需要loc、type和msg。
    # 不记录input，防止把完整LLM响应或潜在敏感信息写进日志。
    error_items = error.errors(
        include_url=False,
        include_context=False,
        include_input=False,
    )

    summaries: list[str] = []

    # 只处理前MAX_DIAGNOSIS_VALIDATION_ERRORS项，
    # 防止异常信息无限增长。
    for item in error_items[
        :MAX_DIAGNOSIS_VALIDATION_ERRORS
    ]:
        # loc通常是元组，例如：
        #
        # ("risk_level",)
        # ("next_checks", 0)
        #
        # 第二种位置会被转换成next_checks.0，
        # 从而指出是列表中的第几项出错。
        location_items = item["loc"]

        location = (
            ".".join(
                str(part)
                for part in location_items
            )
            if location_items
            else "<root>"
        )

        summaries.append(
            f"{location} [{item['type']}]: "
            f"{item['msg']}"
        )

    # 如果实际错误超过上限，
    # 只记录还有多少项没有展开。
    omitted_count = (
        len(error_items)
        - len(summaries)
    )

    if omitted_count > 0:
        summaries.append(
            f"... 另有{omitted_count}项校验错误"
        )

    return "; ".join(summaries)


def _safe_text_presence_and_length(
    value: object,
) -> tuple[bool, int]:
    """只返回文本是否非空及字符长度，不返回正文。

    该函数用于诊断兼容接口中的：

    1. reasoning_content；
    2. refusal。

    它不能用于取得或记录这些字段的实际内容。
    """

    # OpenAI兼容接口的扩展字段不一定严格遵守类型。
    #
    # 只有真正的str才按照文本处理；
    # None、MagicMock、list和其他对象都视为无文本。
    if not isinstance(value, str):
        return False, 0

    # present使用strip()判断是否包含真实内容。
    #
    # length保留原字符串字符长度，
    # 但不返回任何原始字符。
    return bool(value.strip()), len(value)


def _safe_token_count(
    value: object,
) -> str:
    """将合法Token计数转换成文本，否则返回unknown。"""

    # bool是int的子类，但True和False不能表示Token数量。
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        return "unknown"

    return str(value)


def _build_empty_diagnosis_response_detail(
    *,
    completion: object,
    choice: object,
) -> str:
    """为LLM空最终答案生成不泄露正文的错误摘要。

    本函数只保存：

    1. 停止原因；
    2. reasoning_content是否存在及字符长度；
    3. 是否存在refusal；
    4. 工具调用数量；
    5. 总生成Token和推理Token数量。

    不保存Prompt、证据正文、最终内容或隐藏推理正文。
    """

    raw_finish_reason = getattr(
        choice,
        "finish_reason",
        None,
    )

    # finish_reason按照SDK契约应为字符串或None。
    #
    # 不能对未知对象直接使用str()，
    # 因为自定义兼容对象的repr可能包含额外数据。
    if (
        isinstance(raw_finish_reason, str)
        and raw_finish_reason.strip()
    ):
        finish_reason = (
            raw_finish_reason.strip()
        )
    else:
        finish_reason = "unknown"

    message = getattr(
        choice,
        "message",
        None,
    )

    reasoning_content = getattr(
        message,
        "reasoning_content",
        None,
    )
    (
        reasoning_content_present,
        reasoning_content_length,
    ) = _safe_text_presence_and_length(
        reasoning_content
    )

    refusal = getattr(
        message,
        "refusal",
        None,
    )
    refusal_present, _ = (
        _safe_text_presence_and_length(
            refusal
        )
    )

    tool_calls = getattr(
        message,
        "tool_calls",
        None,
    )

    # SDK正常情况下使用list或None。
    #
    # 未知对象不执行len()，
    # 防止兼容接口返回奇怪对象时再次产生异常。
    if isinstance(
        tool_calls,
        (list, tuple),
    ):
        tool_call_count = len(tool_calls)
    else:
        tool_call_count = 0

    usage = getattr(
        completion,
        "usage",
        None,
    )
    completion_tokens = _safe_token_count(
        getattr(
            usage,
            "completion_tokens",
            None,
        )
    )

    completion_token_details = getattr(
        usage,
        "completion_tokens_details",
        None,
    )
    reasoning_tokens = _safe_token_count(
        getattr(
            completion_token_details,
            "reasoning_tokens",
            None,
        )
    )

    return (
        "结构化诊断服务返回了空内容"
        "（"
        f"finish_reason={finish_reason}; "
        "reasoning_content_present="
        f"{reasoning_content_present}; "
        "reasoning_content_length="
        f"{reasoning_content_length}; "
        f"refusal_present={refusal_present}; "
        f"tool_call_count={tool_call_count}; "
        f"completion_tokens={completion_tokens}; "
        f"reasoning_tokens={reasoning_tokens}"
        "）"
    )


class OpenAIDiagnosisDraftProvider:
    """使用OpenAI兼容接口生成诊断草稿。"""

    def __init__(
        self,
        *,
        client: AsyncOpenAI,
        model: str,

        # 正式依赖注入不传此参数时，
        # 继续使用证据优先Prompt。
        #
        # Prompt对照实验可以显式传入基础变体。
        prompt_variant: DiagnosisPromptVariant = (
            EVIDENCE_DIAGNOSIS_PROMPT_VARIANT
        ),
    ) -> None:
        """保存依赖注入的客户端和模型名称。"""

        # 类型注解不会在Python运行时自动校验。
        #
        # 因此除了检查空字符串，
        # 还检查model是否真的是str。
        if (
            not isinstance(model, str)
            or not model.strip()
        ):
            raise ValueError(
                "结构化诊断模型名称不能为空"
            )

        # Prompt版本号和正文必须由
        # DiagnosisPromptVariant绑定。
        #
        # 在构造器中检查可以让错误配置尽早暴露，
        # 不必等到异步LLM请求开始时才失败。
        if not isinstance(
            prompt_variant,
            DiagnosisPromptVariant,
        ):
            raise TypeError(
                "prompt_variant必须是"
                "DiagnosisPromptVariant"
            )

        # client由依赖注入层创建。
        #
        # 测试时可以注入MagicMock；
        # 正式运行时注入AsyncOpenAI。
        self._client = client

        # strip()去除模型名称首尾空白，
        # 避免向SDK发送"  deepseek-chat  "。
        self._model = model.strip()

        # DiagnosisPromptVariant是frozen dataclass，
        # 保存后不会被调用方改写版本号或正文。
        self._prompt_variant = prompt_variant

    @property
    def prompt_version(self) -> str:
        """返回Provider实际配置的可审计Prompt版本。"""

        return self._prompt_variant.version


    async def generate_draft(
        self,
        *,
        request: DiagnosisRequest,
        evidence: tuple[
            HybridRetrievedChunk,
            ...,
        ],
    ) -> DiagnosisLLMDraft:
        """调用LLM并返回经过校验的诊断草稿。"""

        # Prompt在产生外部请求之前构造。
        #
        # 如果request类型错误、evidence为空、
        # rank断裂或Chunk重复，会在这里直接失败，
        # 不会消耗API调用。
        messages = build_diagnosis_messages(
            request=request,
            evidence=evidence,

            # Prompt Builder使用Provider构造时绑定的变体，
            # 确保发送正文和记录版本来自同一个对象。
            prompt_variant=self._prompt_variant,
        )

        try:
            # self._client.chat：
            # 选择SDK中的Chat资源。
            #
            # .completions：
            # 选择聊天补全接口。
            #
            # .create(...)：
            # 创建一次聊天补全请求。
            #
            # 因为使用AsyncOpenAI，
            # create()返回可等待对象，所以必须await。
            completion = (
                await self._client
                .chat
                .completions
                .create(
                    model=self._model,
                    messages=messages,
                    temperature=(
                        DIAGNOSIS_TEMPERATURE
                    ),

                    # DeepSeek V4默认启用思考模式。
                    #
                    # 当前任务是根据已经通过门控的证据
                    # 填充严格的结构化诊断JSON，
                    # 不需要长时间的隐藏推理。
                    #
                    # extra_body会把DeepSeek专有参数
                    # 合并到最终发送的HTTP JSON请求体中。
                    extra_body={
                        "thinking": {
                            "type": "disabled",
                        },
                    },
                )
            )
        except APITimeoutError as exc:
            # 把SDK异常转换成项目自己的异常。
            #
            # 上层异常处理器不需要依赖
            # OpenAI SDK的具体异常类型。
            raise LLMTimeoutError(
                "结构化诊断服务响应超时"
            ) from exc
        except APIConnectionError as exc:
            raise LLMUpstreamError(
                "无法连接结构化诊断服务"
            ) from exc
        except APIStatusError as exc:
            raise LLMUpstreamError(
                "结构化诊断服务返回错误状态："
                f"{exc.status_code}"
            ) from exc

        # choices是上游返回的候选回答列表。
        #
        # 当前系统只使用第一个候选；
        # 如果列表为空，就不存在可解析的响应。
        if not completion.choices:
            raise InvalidLLMResponseError(
                "结构化诊断响应中没有候选结果"
            )

        # 保存第一项Choice，后面既要读取最终内容，
        # 也要在内容为空时读取安全响应元数据。
        choice = completion.choices[0]

        # message是SDK解析后的助手消息对象。
        #
        # OpenAI SDK声明了content、refusal和tool_calls；
        # DeepSeek兼容接口还可能附带reasoning_content。
        message = choice.message

        # content是模型提供给应用程序的最终答案。
        #
        # reasoning_content属于思考过程，
        # 不能代替最终答案进行JSON解析。
        content = message.content

        # 在丢失Choice上下文之前处理空内容。
        #
        # 如果直接调用parse_diagnosis_draft(content)，
        # 解析器只能知道content为空，
        # 无法再报告finish_reason和Token统计。
        if (
            not isinstance(content, str)
            or not content.strip()
        ):
            raise InvalidLLMResponseError(
                _build_empty_diagnosis_response_detail(
                    completion=completion,
                    choice=choice,
                )
            )

        # 非空content继续进入原有可信边界：
        #
        # JSON语法解析
        # → DiagnosisLLMDraft Pydantic校验
        # → 返回内部结构化对象。
        return parse_diagnosis_draft(
            content
        )


def parse_diagnosis_draft(
    content: str | None,
) -> DiagnosisLLMDraft:
    """解析并校验LLM返回的诊断JSON。"""

    # 依次拦截：
    #
    # 1. None；
    # 2. 非字符串对象；
    # 3. 空字符串；
    # 4. 只包含空格或换行的字符串。
    if (
        not isinstance(content, str)
        or not content.strip()
    ):
        raise InvalidLLMResponseError(
            "结构化诊断服务返回了空内容"
        )

    cleaned_content = (
        remove_diagnosis_code_fence(
            content
        )
    )

    try:
        # json.loads()执行的是JSON语法解析。
        #
        # 它只能证明文本是合法JSON，
        # 不能证明字段符合诊断业务契约。
        raw_data = json.loads(
            cleaned_content
        )
    except json.JSONDecodeError as exc:
        raise InvalidLLMResponseError(
            "结构化诊断服务返回的"
            "内容不是合法JSON"
        ) from exc

    try:
        # model_validate()继续校验：
        #
        # 1. 必填字段；
        # 2. 字段类型；
        # 3. E1、E2编号格式；
        # 4. completed/partial/abstained关系；
        # 5. 高风险操作的资质人员标记；
        # 6. 未声明字段。
        return DiagnosisLLMDraft.model_validate(
            raw_data
        )
    except ValidationError as exc:
        # 将Pydantic的结构化错误转换成安全摘要。
        #
        # 摘要会沿着InvalidLLMResponseError传到
        # DiagnosisService的内部日志，
        # 但不会作为API错误详情直接返回给客户端。
        validation_summary = (
            _format_diagnosis_validation_errors(
                exc
            )
        )

        raise InvalidLLMResponseError(
            "结构化诊断JSON"
            "不符合内部响应契约："
            + validation_summary
        ) from exc


def remove_diagnosis_code_fence(
    content: str,
) -> str:
    """兼容完整包裹JSON的Markdown代码围栏。"""

    # 尽管System Prompt禁止代码围栏，
    # 某些模型仍可能返回：
    #
    # ```json
    # {...}
    # ```
    #
    # 这里仅兼容“围栏完整包裹整个响应”的情况，
    # 不会从任意自然语言中搜索或猜测JSON。
    cleaned_content = content.strip()

    # splitlines()按换行拆成字符串列表，
    # 并自动兼容不同平台的换行格式。
    lines = cleaned_content.splitlines()

    if (
        len(lines) >= 3
        and lines[0].strip().startswith(
            "```"
        )
        and lines[-1].strip() == "```"
    ):
        # 切片[1:-1]排除第一行和最后一行。
        #
        # "\n".join(...)再把中间的JSON
        # 按原有顺序拼接起来。
        return "\n".join(
            lines[1:-1]
        ).strip()

    return cleaned_content