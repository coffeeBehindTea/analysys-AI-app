"""结构化诊断LLM Provider的离线测试。

本模块使用AsyncMock模拟OpenAI兼容SDK，
不会访问真实LLM、Embedding、Chroma或HTTP服务。

测试范围：
1. 合法JSON是否解析为DiagnosisLLMDraft；
2. Provider是否发送正确模型、温度和Prompt变体消息；
3. 空内容是否携带不泄露正文的安全响应元数据；
4. 非法JSON、缺字段和高风险违规是否被拒绝；
5. Markdown JSON代码围栏是否被兼容处理；
6. 超时、连接失败和错误状态是否转换成应用异常；
7. Prompt参数错误是否在SDK调用前失败；
8. 未知但格式合法的E编号是否保留给业务层执行白名单检查。
"""

import json
from unittest.mock import (
    AsyncMock,
    MagicMock,
)

import httpx
import pytest
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
)

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
    DocumentChunk,
    HybridRetrievedChunk,
)
from app.services.diagnosis_generation import (
    DIAGNOSIS_TEMPERATURE,
    OpenAIDiagnosisDraftProvider,
    parse_diagnosis_draft,
)
from app.services.diagnosis_prompts import (
    BASIC_DIAGNOSIS_PROMPT_VARIANT,
    EVIDENCE_DIAGNOSIS_PROMPT_VARIANT,
)


# 测试模型名用于检查Provider是否正确转发构造参数。
TEST_MODEL = "test-diagnosis-model"

# DocumentChunk中的document_id和content_hash
# 都使用符合真实契约的64位小写十六进制文本。
TEST_DOCUMENT_ID = "c" * 64
TEST_CONTENT_HASH = "d" * 64


def make_request() -> DiagnosisRequest:
    """构造一份通过Pydantic校验的诊断请求。"""

    return DiagnosisRequest(
        robot_id="robot-001",
        symptom="网络恢复后机器人仍未继续任务",
        log_excerpt=(
            "ERR-NET-4001 heartbeat recovered"
        ),
    )


def make_evidence(
) -> tuple[HybridRetrievedChunk, ...]:
    """构造一条已经通过混合检索门控的候选。"""

    candidate = HybridRetrievedChunk(
        chunk=DocumentChunk(
            chunk_id=(
                f"{TEST_DOCUMENT_ID}:000001"
            ),
            document_id=TEST_DOCUMENT_ID,
            source_file="robot-faults.txt",
            page_or_section=(
                "section: ERR-NET-4001"
            ),
            chunk_index=1,
            content_hash=TEST_CONTENT_HASH,
            content=(
                "ERR-NET-4001网络恢复后，"
                "应先确认调度任务状态。"
            ),
        ),
        rrf_score=0.032,
        rank=1,
        vector_rank=1,
        vector_similarity=0.68,
        keyword_rank=1,
        keyword_score=12.0,
        matched_identifiers=(
            "err-net-4001",
        ),
    )

    return (candidate,)


def make_valid_draft_data(
) -> dict[str, object]:
    """构造符合DiagnosisLLMDraft契约的Python数据。"""

    return {
        "status": "completed",
        "possible_causes": [
            {
                "description": (
                    "调度通信可能曾经中断"
                ),
                "evidence_ids": ["E1"],
            }
        ],
        "next_checks": [
            {
                "description": (
                    "核对机器人与调度任务状态"
                ),
                "evidence_ids": ["E1"],
                "risk_level": "low",
                "requires_qualified_person": False,
            }
        ],
        "risk_level": "low",
        "missing_information": [],
        "abstained": False,
    }


def create_mock_client_with_content(
    content: str | None,
) -> tuple[MagicMock, AsyncMock]:
    """创建返回指定message.content的假异步客户端。"""

    # 模拟completion.choices[0].message.content。
    mock_choice = MagicMock()
    mock_choice.message.content = content

    # 显式设置真实SDK可能提供的诊断字段，
    # 避免未配置MagicMock属性被误判为有效数据。
    mock_choice.finish_reason = "stop"
    mock_choice.message.reasoning_content = None
    mock_choice.message.refusal = None
    mock_choice.message.tool_calls = None

    # 模拟SDK返回的ChatCompletion对象。
    mock_completion = MagicMock()
    mock_completion.choices = [
        mock_choice,
    ]

    # chat.completions.create()需要被await，
    # 所以使用AsyncMock而不是普通MagicMock。
    mock_create = AsyncMock(
        return_value=mock_completion,
    )

    # 构造client.chat.completions.create属性链。
    mock_client = MagicMock()
    mock_client.chat.completions.create = (
        mock_create
    )

    return mock_client, mock_create


def make_sdk_request() -> httpx.Request:
    """构造SDK异常所需的内存HTTP请求对象。"""

    # 创建Request对象不会发起网络请求。
    return httpx.Request(
        method="POST",
        url=(
            "https://llm.test/chat/completions"
        ),
    )


def test_constructor_rejects_blank_model(
) -> None:
    """空白模型名不能延迟到真实SDK请求时才失败。"""

    with pytest.raises(
        ValueError,
        match="结构化诊断模型名称不能为空",
    ):
        OpenAIDiagnosisDraftProvider(
            client=MagicMock(),
            model="   ",
        )


def test_constructor_rejects_invalid_prompt_variant(
) -> None:
    """Provider配置阶段必须拒绝未绑定版本的普通字典。"""

    # 在构造器中失败表示配置错误会在启动或装配阶段暴露，
    # 而不是等到真实异步SDK调用时才出现。
    with pytest.raises(
        TypeError,
        match=(
            "prompt_variant必须是"
            "DiagnosisPromptVariant"
        ),
    ):
        OpenAIDiagnosisDraftProvider(
            client=MagicMock(),
            model=TEST_MODEL,
            prompt_variant={  # type: ignore[arg-type]
                "version": "untracked",
                "system_prompt": "未审计提示",
            },
        )


@pytest.mark.asyncio
async def test_provider_returns_validated_draft(
) -> None:
    """合法模型JSON应完成Prompt、SDK和Pydantic流程。"""

    fake_content = json.dumps(
        make_valid_draft_data(),
        ensure_ascii=False,
    )
    mock_client, mock_create = (
        create_mock_client_with_content(
            fake_content
        )
    )
    provider = OpenAIDiagnosisDraftProvider(
        client=mock_client,
        model=f"  {TEST_MODEL}  ",
    )

    # 生产依赖没有显式传Prompt变体时，
    # Provider必须继续使用证据优先正式版本。
    assert provider.prompt_version == (
        EVIDENCE_DIAGNOSIS_PROMPT_VARIANT.version
    )

    draft = await provider.generate_draft(
        request=make_request(),
        evidence=make_evidence(),
    )

    assert isinstance(
        draft,
        DiagnosisLLMDraft,
    )
    assert draft.status == "completed"
    assert draft.possible_causes[
        0
    ].evidence_ids == ["E1"]

    mock_create.assert_awaited_once()
    awaited_kwargs = (
        mock_create.await_args.kwargs
    )

    # 构造器应去除模型名称首尾空白。
    assert awaited_kwargs["model"] == (
        TEST_MODEL
    )
    assert awaited_kwargs["temperature"] == (
        DIAGNOSIS_TEMPERATURE
    )

    # DeepSeek V4默认启用思考模式。
    #
    # 当前诊断生成属于“依据现有证据填充严格JSON契约”，
    # 应显式关闭隐藏推理，避免结构化抽取请求
    # 因长时间思考占满客户端超时窗口。
    assert awaited_kwargs["extra_body"] == {
        "thinking": {
            "type": "disabled",
        },
    }

    messages = awaited_kwargs["messages"]

    assert [
        message["role"]
        for message in messages
    ] == ["system", "user"]
    assert messages[0]["content"] == (
        EVIDENCE_DIAGNOSIS_PROMPT_VARIANT.system_prompt
    )
    assert "robot-001" in (
        messages[1]["content"]
    )
    assert "ERR-NET-4001" in (
        messages[1]["content"]
    )
    assert "output_schema" in (
        messages[1]["content"]
    )


@pytest.mark.asyncio
async def test_provider_uses_explicit_basic_prompt_variant(
) -> None:
    """实验Provider必须把基础Prompt及其版本发送给SDK。"""

    fake_content = json.dumps(
        make_valid_draft_data(),
        ensure_ascii=False,
    )
    mock_client, mock_create = (
        create_mock_client_with_content(
            fake_content
        )
    )
    provider = OpenAIDiagnosisDraftProvider(
        client=mock_client,
        model=TEST_MODEL,
        prompt_variant=(
            BASIC_DIAGNOSIS_PROMPT_VARIANT
        ),
    )

    draft = await provider.generate_draft(
        request=make_request(),
        evidence=make_evidence(),
    )

    # 相同解析器仍应把合法基础组JSON
    # 校验成DiagnosisLLMDraft。
    assert isinstance(
        draft,
        DiagnosisLLMDraft,
    )
    assert provider.prompt_version == (
        BASIC_DIAGNOSIS_PROMPT_VARIANT.version
    )

    awaited_kwargs = (
        mock_create.await_args.kwargs
    )
    messages = awaited_kwargs["messages"]

    assert messages[0]["content"] == (
        BASIC_DIAGNOSIS_PROMPT_VARIANT.system_prompt
    )

    # User Message中的审计版本也必须与
    # 实际发送的System Prompt属于同一变体。
    assert (
        '"prompt_version": "diagnosis-basic-v1"'
        in messages[1]["content"]
    )


def test_parser_accepts_json_code_fence(
) -> None:
    """完整包裹JSON的Markdown围栏应被兼容移除。"""

    fenced_content = (
        "```json\n"
        + json.dumps(
            make_valid_draft_data(),
            ensure_ascii=False,
        )
        + "\n```"
    )

    draft = parse_diagnosis_draft(
        fenced_content
    )

    assert draft.status == "completed"
    assert draft.abstained is False


@pytest.mark.parametrize(
    "empty_content",
    [
        None,
        "",
        "   ",
    ],
)
def test_parser_rejects_empty_content(
    empty_content: str | None,
) -> None:
    """None、空字符串和纯空白都属于无效LLM响应。"""

    with pytest.raises(
        InvalidLLMResponseError,
        match="结构化诊断服务返回了空内容",
    ):
        parse_diagnosis_draft(
            empty_content
        )


def test_parser_rejects_invalid_json(
) -> None:
    """普通自然语言不能冒充结构化诊断JSON。"""

    with pytest.raises(
        InvalidLLMResponseError,
        match="内容不是合法JSON",
    ):
        parse_diagnosis_draft(
            "我认为可能是网络故障"
        )


def test_parser_rejects_missing_required_field(
) -> None:
    """缺少risk_level的JSON必须被Pydantic拒绝。"""

    invalid_data = make_valid_draft_data()
    del invalid_data["risk_level"]

    with pytest.raises(
        InvalidLLMResponseError,
        match="JSON不符合内部响应契约",
    ) as exc_info:
        parse_diagnosis_draft(
            json.dumps(
                invalid_data,
                ensure_ascii=False,
            )
        )

    error_message = str(exc_info.value)

    # 内部异常需要指出失败字段和Pydantic错误类型，
    # 这样Service日志才能定位真实模型输出问题。
    assert (
        "risk_level [missing]: Field required"
        in error_message
    )

    # 不能使用str(ValidationError)，
    # 否则可能把模型完整输入值和帮助URL写入日志。
    assert "input_value" not in error_message
    assert "errors.pydantic.dev" not in error_message


def test_parser_rejects_unsafe_high_risk_check(
) -> None:
    """高风险检查不能被标记为无需资质人员。"""

    invalid_data = make_valid_draft_data()
    next_check = invalid_data[
        "next_checks"
    ][0]
    next_check["risk_level"] = "high"
    next_check[
        "requires_qualified_person"
    ] = False

    with pytest.raises(
        InvalidLLMResponseError,
        match="JSON不符合内部响应契约",
    ) as exc_info:
        parse_diagnosis_draft(
            json.dumps(
                invalid_data,
                ensure_ascii=False,
            )
        )

    error_message = str(exc_info.value)

    # 嵌套列表中的第一项使用路径next_checks.0定位。
    assert "next_checks.0 [value_error]" in (
        error_message
    )
    assert (
        "高风险检查必须标记为需要有资质人员确认"
        in error_message
    )

    # 服务端诊断信息只包含规则说明，
    # 不包含原始字段值或Pydantic帮助URL。
    assert "input_value" not in error_message
    assert "errors.pydantic.dev" not in error_message


def test_parser_preserves_well_formed_unknown_evidence_id(
) -> None:
    """格式合法的E999留给业务Service执行白名单检查。"""

    draft_data = make_valid_draft_data()
    cause = draft_data[
        "possible_causes"
    ][0]
    cause["evidence_ids"] = ["E999"]

    draft = parse_diagnosis_draft(
        json.dumps(
            draft_data,
            ensure_ascii=False,
        )
    )

    # Schema只知道E999格式合法，
    # 并不知道本次Prompt实际只有E1。
    assert draft.possible_causes[
        0
    ].evidence_ids == ["E999"]


@pytest.mark.asyncio
async def test_provider_rejects_empty_choices(
) -> None:
    """SDK没有返回候选消息时应产生稳定应用异常。"""

    mock_completion = MagicMock()
    mock_completion.choices = []

    mock_create = AsyncMock(
        return_value=mock_completion,
    )
    mock_client = MagicMock()
    mock_client.chat.completions.create = (
        mock_create
    )

    provider = OpenAIDiagnosisDraftProvider(
        client=mock_client,
        model=TEST_MODEL,
    )

    with pytest.raises(
        InvalidLLMResponseError,
        match="响应中没有候选结果",
    ):
        await provider.generate_draft(
            request=make_request(),
            evidence=make_evidence(),
        )


@pytest.mark.asyncio
async def test_provider_empty_content_reports_safe_metadata(
) -> None:
    """空最终答案应报告停止原因和长度，但不得泄露隐藏推理。"""

    mock_client, mock_create = (
        create_mock_client_with_content("")
    )
    mock_completion = (
        mock_create.return_value
    )
    mock_choice = (
        mock_completion.choices[0]
    )

    # 模拟思考内容存在，但最终content为空的上游响应。
    # 使用重复字符便于精确检查长度和泄漏。
    hidden_reasoning = "x" * 37
    mock_choice.finish_reason = "length"
    mock_choice.message.reasoning_content = (
        hidden_reasoning
    )
    mock_choice.message.refusal = None
    mock_choice.message.tool_calls = []

    # usage只记录Token计数，不包含Prompt或回答正文。
    mock_completion.usage.completion_tokens = 128
    mock_completion.usage.completion_tokens_details.reasoning_tokens = 96

    provider = OpenAIDiagnosisDraftProvider(
        client=mock_client,
        model=TEST_MODEL,
    )

    with pytest.raises(
        InvalidLLMResponseError,
        match="结构化诊断服务返回了空内容",
    ) as exc_info:
        await provider.generate_draft(
            request=make_request(),
            evidence=make_evidence(),
        )

    error_message = str(exc_info.value)

    assert "finish_reason=length" in error_message
    assert (
        "reasoning_content_present=True"
        in error_message
    )
    assert (
        "reasoning_content_length=37"
        in error_message
    )
    assert "refusal_present=False" in error_message
    assert "tool_call_count=0" in error_message
    assert "completion_tokens=128" in error_message
    assert "reasoning_tokens=96" in error_message

    # 错误摘要只能记录长度，不能保存隐藏推理正文。
    assert hidden_reasoning not in error_message


@pytest.mark.asyncio
async def test_provider_converts_timeout(
) -> None:
    """SDK超时必须转换成应用层LLMTimeoutError。"""

    sdk_error = APITimeoutError(
        request=make_sdk_request(),
    )
    mock_create = AsyncMock(
        side_effect=sdk_error,
    )
    mock_client = MagicMock()
    mock_client.chat.completions.create = (
        mock_create
    )
    provider = OpenAIDiagnosisDraftProvider(
        client=mock_client,
        model=TEST_MODEL,
    )

    with pytest.raises(
        LLMTimeoutError,
        match="结构化诊断服务响应超时",
    ) as exc_info:
        await provider.generate_draft(
            request=make_request(),
            evidence=make_evidence(),
        )

    assert exc_info.value.__cause__ is sdk_error


@pytest.mark.asyncio
async def test_provider_converts_connection_error(
) -> None:
    """SDK连接失败必须转换成应用层上游异常。"""

    sdk_error = APIConnectionError(
        request=make_sdk_request(),
    )
    mock_create = AsyncMock(
        side_effect=sdk_error,
    )
    mock_client = MagicMock()
    mock_client.chat.completions.create = (
        mock_create
    )
    provider = OpenAIDiagnosisDraftProvider(
        client=mock_client,
        model=TEST_MODEL,
    )

    with pytest.raises(
        LLMUpstreamError,
        match="无法连接结构化诊断服务",
    ) as exc_info:
        await provider.generate_draft(
            request=make_request(),
            evidence=make_evidence(),
        )

    assert exc_info.value.__cause__ is sdk_error


@pytest.mark.asyncio
async def test_provider_converts_status_error(
) -> None:
    """SDK错误状态必须保留状态码并转为应用异常。"""

    sdk_request = make_sdk_request()
    sdk_response = httpx.Response(
        status_code=503,
        request=sdk_request,
    )
    sdk_error = APIStatusError(
        "Service Unavailable",
        response=sdk_response,
        body={"error": "unavailable"},
    )
    mock_create = AsyncMock(
        side_effect=sdk_error,
    )
    mock_client = MagicMock()
    mock_client.chat.completions.create = (
        mock_create
    )
    provider = OpenAIDiagnosisDraftProvider(
        client=mock_client,
        model=TEST_MODEL,
    )

    with pytest.raises(
        LLMUpstreamError,
        match=(
            "结构化诊断服务返回错误状态：503"
        ),
    ) as exc_info:
        await provider.generate_draft(
            request=make_request(),
            evidence=make_evidence(),
        )

    assert exc_info.value.__cause__ is sdk_error


@pytest.mark.asyncio
async def test_prompt_validation_happens_before_sdk_call(
) -> None:
    """空证据必须在产生外部LLM请求之前失败。"""

    mock_client, mock_create = (
        create_mock_client_with_content(
            json.dumps(
                make_valid_draft_data(),
                ensure_ascii=False,
            )
        )
    )
    provider = OpenAIDiagnosisDraftProvider(
        client=mock_client,
        model=TEST_MODEL,
    )

    with pytest.raises(
        ValueError,
        match="evidence不能为空",
    ):
        await provider.generate_draft(
            request=make_request(),
            evidence=(),
        )

    # Prompt Builder先失败后，SDK方法一次也不能调用。
    mock_create.assert_not_awaited()
