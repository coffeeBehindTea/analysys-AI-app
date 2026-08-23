"""知识库LLM回答组件的离线测试。"""

# json.dumps()把Python字典转换成JSON字符串，
# 用于模拟LLM返回的message.content。
#
# json.loads()把JSON字符串转换回Python对象，
# 用于验证Prompt中携带的结构化数据。
import json

# AsyncMock模拟会被await的SDK方法；
# MagicMock模拟客户端和响应对象。
from unittest.mock import AsyncMock, MagicMock

# httpx.Request只用于构造SDK超时异常上下文，
# 不会发送网络请求。
import httpx

import pytest

from openai import APITimeoutError

from app.errors import (
    InvalidLLMResponseError,
    LLMTimeoutError,
)
from app.schemas.retrieval import (
    DocumentChunk,
    HybridRetrievedChunk,
    RetrievedChunk,
)
from app.services.knowledge_answer import (
    OpenAIKnowledgeAnswerProvider,
)
from app.services.knowledge_prompts import (
    build_knowledge_answer_messages,
)


def make_evidence(
    *,
    content: str = (
        "急停装置复位后，应检查"
        "安全控制系统状态。"
    ),
) -> list[RetrievedChunk]:
    """创建正文可定制的真实结构测试证据。"""

    document_id = "d" * 64

    return [
        RetrievedChunk(
            chunk=DocumentChunk(
                chunk_id=(
                    f"{document_id}:000002"
                ),
                document_id=document_id,
                source_file="safety-manual.pdf",
                page_or_section="page: 12",
                chunk_index=2,
                content_hash="e" * 64,

                # 默认使用原来的急停证据；
                # 个别测试可以通过content参数
                # 传入特定的安全约束正文。
                content=content,
            ),
            similarity=0.82,
            rank=1,
        )
    ]


def make_hybrid_evidence(
    *,
    keyword_only: bool = False,
) -> list[HybridRetrievedChunk]:
    """创建双路命中或关键词单路命中的混合检索证据。"""

    # 使用固定的64位十六进制文档ID，
    # 让测试数据稳定并符合真实Chunk ID的结构。
    document_id = "f" * 64

    # keyword_only=True模拟一个重要边界：
    # Chunk被精确错误码检索找到，
    # 但没有进入向量检索候选列表。
    vector_rank = (
        None
        if keyword_only
        else 1
    )
    vector_similarity = (
        None
        if keyword_only
        else 0.61
    )

    return [
        HybridRetrievedChunk(
            chunk=DocumentChunk(
                chunk_id=(
                    f"{document_id}:000003"
                ),
                document_id=document_id,
                source_file=(
                    "warehouse-faults.txt"
                ),
                page_or_section=(
                    "section: ERR-NET-4001"
                ),
                chunk_index=3,
                content_hash="a" * 64,
                content=(
                    "ERR-NET-4001表示调度心跳超时；"
                    "网络恢复后不得自动继续旧任务。"
                ),
            ),

            # 关键词单路命中只有一个倒数排名贡献；
            # 双路命中同时获得向量和关键词贡献。
            rrf_score=(
                0.016393
                if keyword_only
                else 0.032787
            ),
            rank=1,
            vector_rank=vector_rank,
            vector_similarity=(
                vector_similarity
            ),
            keyword_rank=1,
            keyword_score=15.0,
            matched_identifiers=(
                "err-net-4001",
            ),
        )
    ]


def create_mock_client_with_content(
    content: str | None,
) -> tuple[MagicMock, AsyncMock]:
    """创建返回指定文本的假OpenAI客户端。"""

    # 模拟completion.choices[0]。
    mock_choice = MagicMock()

    # 模拟choices[0].message.content。
    mock_choice.message.content = content

    # 模拟ChatCompletion对象。
    mock_completion = MagicMock()
    mock_completion.choices = [
        mock_choice,
    ]

    # create()会被await，因此使用AsyncMock。
    mock_create = AsyncMock(
        return_value=mock_completion,
    )

    # 模拟client.chat.completions.create属性链。
    mock_client = MagicMock()
    mock_client.chat.completions.create = (
        mock_create
    )

    return mock_client, mock_create


def test_prompt_serializes_question_and_evidence(
) -> None:
    """提示词应包含经过JSON序列化的真实证据。"""

    messages = build_knowledge_answer_messages(
        question="急停复位后应检查什么？",
        evidence=make_evidence(),
    )

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"

    # user消息第一行是说明，
    # 后续内容是完整JSON。
    _, payload_json = (
        messages[1]["content"].split(
            "\n",
            1,
        )
    )

    payload = json.loads(payload_json)

    assert payload["question"] == (
        "急停复位后应检查什么？"
    )

    assert payload["evidence"] == [
        {
            "evidence_id": "E1",
            "rank": 1,
            "similarity": 0.82,
            "source_file": "safety-manual.pdf",
            "page_or_section": "page: 12",
            "content": (
                "急停装置复位后，应检查"
                "安全控制系统状态。"
            ),
        }
    ]


def test_prompt_serializes_dual_path_hybrid_evidence(
) -> None:
    """Prompt应接收向量和关键词双路命中的融合证据。"""

    messages = build_knowledge_answer_messages(
        question=(
            "ERR-NET-4001网络恢复后"
            "能否自动继续旧任务？"
        ),
        evidence=make_hybrid_evidence(),
    )

    # User Message的第一行是边界说明，
    # 第一行之后才是可由json.loads()解析的载荷。
    _, payload_json = (
        messages[1]["content"].split(
            "\n",
            1,
        )
    )
    payload = json.loads(payload_json)
    evidence_item = payload["evidence"][0]

    # 双路命中的Chunk有真实向量相似度，
    # 因此可以诚实地把0.61交给Prompt。
    assert evidence_item["similarity"] == 0.61
    assert evidence_item["evidence_id"] == "E1"
    assert evidence_item["rank"] == 1
    assert evidence_item["source_file"] == (
        "warehouse-faults.txt"
    )

    # RRF分数只用于Python侧融合排序和API审计，
    # 不是事实证据或回答置信度，
    # 所以不应发送给LLM解释。
    assert "rrf_score" not in evidence_item


def test_prompt_keeps_keyword_only_similarity_null(
) -> None:
    """关键词单路证据不能伪造向量相似度。"""

    messages = build_knowledge_answer_messages(
        question=(
            "ERR-NET-4001网络恢复后"
            "能否自动继续旧任务？"
        ),
        evidence=make_hybrid_evidence(
            keyword_only=True,
        ),
    )

    _, payload_json = (
        messages[1]["content"].split(
            "\n",
            1,
        )
    )
    payload = json.loads(payload_json)
    evidence_item = payload["evidence"][0]

    # Python中的None经过json.dumps()后会成为JSON null。
    # 这明确表示“本次没有向量召回分数”，
    # 而不是把0或RRF分数冒充成余弦相似度。
    assert evidence_item["similarity"] is None
    assert evidence_item["evidence_id"] == "E1"
    assert "rrf_score" not in evidence_item


def test_prompt_requires_status_qualifiers_to_be_preserved(
) -> None:
    """System Prompt必须要求保留文档状态限定词。"""

    # 使用与真实q005相同性质的问题。
    #
    # “征求意见稿”是文件状态的一部分，
    # 不能在回答中被压缩成已经正式生效的标准。
    question = (
        "《工业移动机器人安全标准》"
        "征求意见稿适用于哪些机器人？"
    )

    # 构造完整的System和User消息。
    #
    # make_evidence()提供一条合法RetrievedChunk，
    # 使Prompt Builder可以正常运行。
    messages = build_knowledge_answer_messages(
        question=question,
        evidence=make_evidence(),
    )

    # build_knowledge_answer_messages()的契约是：
    #
    # messages[0]为System Message；
    # messages[1]为User Message。
    system_prompt = messages[0]["content"]
    user_message = messages[1]["content"]

    # System Prompt必须明确识别
    # “征求意见稿”这一类状态限定词。
    assert "征求意见稿" in system_prompt


    # 必须明确禁止把草案改写成正式发布状态。
    assert (
        "不得改写成正式发布"
        in system_prompt
    )

    # 用户问题必须原样进入User Message，
    # 否则即使System Prompt要求保留限定，
    # 模型也无法知道本次问题包含什么限定词。
    assert "征求意见稿" in user_message

    # 对“保留限定”给出可机械检查的最低要求：
    # answer中至少原样出现一次状态限定词。
    assert (
        "answer中必须至少原样出现一次"
        in system_prompt
    )

def test_prompt_requires_safety_constraints_to_be_preserved(
) -> None:
    """System Prompt必须要求保留相关安全约束。"""

    question = (
        "应怎样安全验证驱动电机过温故障，"
        "从触发到恢复需要满足哪些条件？"
    )

    safety_evidence = make_evidence(
        content=(
            "应通过测试接口注入高于85°C的"
            "模拟温度信号。"
            "不允许为了测试而真实堵转电机。"
            "高温持续10秒后触发故障，"
            "完成人工检查前不得复位。"
        ),
    )

    messages = build_knowledge_answer_messages(
        question=question,
        evidence=safety_evidence,
    )

    # 第一条消息是不可被普通用户输入覆盖的
    # System Prompt。
    system_prompt = messages[0]["content"]

    # 第二条消息包含本次真实问题和检索证据。
    user_message = messages[1]["content"]

    # System Prompt必须明确要求逐条检查
    # 禁止、强制和前置条件等安全约束。
    assert (
        "安全约束词"
        in system_prompt
    )

    # 不能只要求“注意安全”，
    # 而要明确规定最终answer需要保留哪些语义。
    assert (
        "answer必须明确保留其禁止、强制、"
        "前置条件和数值阈值"
        in system_prompt
    )

    # 原来的实际缺陷是：
    #
    # 模型保留了“使用测试接口”这一肯定建议，
    # 却省略了“不得真实堵转”这一禁止项。
    #
    # 这条断言专门防止Prompt规则退化。
    assert (
        "不得仅用肯定式建议替代禁止项"
        in system_prompt
    )

    # 还必须禁止模型把证据明令禁止的动作
    # 重新包装成“必要时可以尝试”的可选操作。
    assert (
        "不得把被禁止的操作改写为可选方案"
        in system_prompt
    )

    # Prompt Builder必须把实际安全证据
    # 完整序列化进User Message。
    assert (
        "不允许为了测试而真实堵转电机"
        in user_message
    )
    assert (
        "完成人工检查前不得复位"
        in user_message
    )

    # 用户问题也必须进入同一User Message，
    # 模型才能判断哪些安全约束与问题相关。
    assert (
        question
        in user_message
    )

def test_prompt_requires_minimal_sufficient_evidence(
) -> None:
    """System Prompt必须限制模型选择最小充分证据集。"""

    messages = build_knowledge_answer_messages(
        question=(
            "机器人电量达到什么条件时"
            "应进入充电队列？"
        ),
        evidence=make_evidence(),
    )

    # 第一条消息是System Message，
    # 其中保存所有不可被用户输入覆盖的回答规则。
    system_prompt = messages[0]["content"]

    # 必须明确要求“最小充分证据集合”，
    # 不能只要求证据编号真实存在。
    assert (
        "最小充分证据集合"
        in system_prompt
    )

    # 如果已有一条证据完整支持回答，
    # 必须禁止选择只重复相同事实的证据。
    assert (
        "不得再选择只重复相同事实"
        in system_prompt
    )

    # 只有出现尚未得到支持的必要事实时，
    # 才允许增加另一条证据。
    assert (
        "某个必要事实无法由已经选择的证据支持"
        in system_prompt
    )

@pytest.mark.asyncio
async def test_provider_returns_cleaned_content(
) -> None:
    """合法LLM JSON应解析成经过校验的回答Draft。"""

    # 构造模拟LLM返回的JSON字符串。
    #
    # 真实OpenAI兼容SDK中的
    # completion.choices[0].message.content
    # 也是字符串，而不是Python字典。
    fake_llm_content = json.dumps(
        {
            "answer": (
                "应检查安全控制系统状态。"
            ),
            "used_evidence_ids": [
                "E1"
            ],
        },

        # 默认情况下，json.dumps()会把中文转换成
        # "\\u5e94\\u68c0..."形式的Unicode转义。
        #
        # ensure_ascii=False让模拟响应保留可读中文。
        # 两种形式都能被json.loads()解析，
        # 这里保留中文主要是为了方便阅读和排错。
        ensure_ascii=False,
    )

    # create_mock_client_with_content()返回一个二元组：
    #
    # 第一个元素mock_client：
    # 模拟完整的OpenAI客户端，供Provider调用。
    #
    # 第二个元素mock_create：
    # 模拟client.chat.completions.create()方法，
    # 供测试检查它是否被await以及收到了哪些参数。
    #
    # 左侧两个变量会按照位置接收右侧二元组中的值，
    # 这种写法叫作“序列解包”或“元组解包”。
    mock_client, mock_create = (
        create_mock_client_with_content(
            fake_llm_content
        )
    )

    # 创建被测试的回答Provider。
    #
    # 通过构造函数把Mock客户端注入Provider，
    # 所以Provider不会访问真实LLM或外部网络。
    provider = OpenAIKnowledgeAnswerProvider(
        client=mock_client,
        model="test-model",
    )

    # 调用本测试真正验证的异步方法。
    #
    # 内部预期流程是：
    #
    # 1. 构造system和user消息；
    # 2. await mock_client.chat.completions.create()；
    # 3. 取得message.content；
    # 4. json.loads()解析JSON；
    # 5. KnowledgeAnswerDraft.model_validate()校验；
    # 6. 返回KnowledgeAnswerDraft。
    draft = await provider.generate_answer(
        question="急停复位后应检查什么？",
        evidence=make_evidence(),
    )

    # 回答正文必须来自模型JSON中的answer字段。
    assert draft.answer == (
        "应检查安全控制系统状态。"
    )

    # 模型声明使用的证据编号必须被保留下来，
    # 供KnowledgeQueryService之后执行白名单验证。
    assert draft.used_evidence_ids == [
        "E1"
    ]

    # create()是异步方法，所以必须使用
    # AsyncMock的assert_awaited_once()检查。
    #
    # 它同时验证：
    # 1. create()确实被await；
    # 2. create()只被await了一次。
    mock_create.assert_awaited_once()

    # await_args保存AsyncMock最近一次被await时
    # 收到的位置参数和关键字参数。
    #
    # kwargs属性取得通过“参数名=值”
    # 传入的关键字参数字典。
    awaited_kwargs = (
        mock_create.await_args.kwargs
    )

    # Provider必须把构造时保存的模型名称
    # 传给OpenAI兼容SDK的model参数。
    assert awaited_kwargs["model"] == (
        "test-model"
    )

    # 知识问答要求模型返回短小、严格的JSON，
    # 因此使用0温度降低不必要的采样波动。
    assert awaited_kwargs["temperature"] == 0.0

    # DeepSeek思考模式可能只产生reasoning_content，
    # 而让最终message.content为空。
    # Provider必须显式关闭思考模式，确保结构化JSON
    # 出现在message.content中供后续解析。
    assert awaited_kwargs["extra_body"] == {
        "thinking": {
            "type": "disabled",
        },
    }

    # messages是Provider发送给模型的聊天消息列表。
    messages = awaited_kwargs["messages"]

    # 第一条消息必须是System Prompt，
    # 用于定义不可被普通用户消息覆盖的回答规则。
    assert messages[0]["role"] == "system"

    # 第二条消息必须是User Message，
    # 用于携带本次问题和检索证据。
    assert messages[1]["role"] == "user"

    # 检索证据中的真实文件名必须进入User Message，
    # 证明Provider确实把召回证据传给了LLM。
    assert "safety-manual.pdf" in (
        messages[1]["content"]
    )

    # System Prompt必须明确要求模型返回
    # used_evidence_ids结构化字段。
    assert "used_evidence_ids" in (
        messages[0]["content"]
    )

    # System Prompt还必须禁止模型把E1、E2等
    # 内部临时编号直接写进最终回答正文。
    assert "answer正文中不得出现" in (
        messages[0]["content"]
    )


@pytest.mark.asyncio
async def test_provider_converts_sdk_timeout(
) -> None:
    """SDK超时应转换为应用层LLMTimeoutError。"""

    sdk_request = httpx.Request(
        method="POST",
        url="https://llm.test/chat/completions",
    )

    sdk_timeout = APITimeoutError(
        request=sdk_request,
    )

    mock_create = AsyncMock(
        side_effect=sdk_timeout,
    )

    mock_client = MagicMock()
    mock_client.chat.completions.create = (
        mock_create
    )

    provider = OpenAIKnowledgeAnswerProvider(
        client=mock_client,
        model="test-model",
    )

    with pytest.raises(
        LLMTimeoutError,
        match="响应超时",
    ):
        await provider.generate_answer(
            question="测试问题",
            evidence=make_evidence(),
        )

    mock_create.assert_awaited_once()


@pytest.mark.asyncio
async def test_provider_rejects_empty_choices(
) -> None:
    """LLM没有候选结果时应产生无效响应异常。"""

    mock_completion = MagicMock()
    mock_completion.choices = []

    mock_create = AsyncMock(
        return_value=mock_completion,
    )

    mock_client = MagicMock()
    mock_client.chat.completions.create = (
        mock_create
    )

    provider = OpenAIKnowledgeAnswerProvider(
        client=mock_client,
        model="test-model",
    )

    with pytest.raises(
        InvalidLLMResponseError,
        match="没有候选结果",
    ):
        await provider.generate_answer(
            question="测试问题",
            evidence=make_evidence(),
        )


@pytest.mark.asyncio
async def test_provider_rejects_empty_content(
) -> None:
    """候选结果正文为空时应产生无效响应异常。"""

    mock_client, mock_create = (
        create_mock_client_with_content("   ")
    )

    provider = OpenAIKnowledgeAnswerProvider(
        client=mock_client,
        model="test-model",
    )

    with pytest.raises(
        InvalidLLMResponseError,
        match="空内容",
    ):
        await provider.generate_answer(
            question="测试问题",
            evidence=make_evidence(),
        )

    mock_create.assert_awaited_once()


@pytest.mark.asyncio
async def test_provider_rejects_invalid_json(
) -> None:
    """模型返回普通文本时应拒绝进入编排层。"""

    mock_client, mock_create = (
        create_mock_client_with_content(
            "这不是合法JSON"
        )
    )

    provider = OpenAIKnowledgeAnswerProvider(
        client=mock_client,
        model="test-model",
    )

    with pytest.raises(
        InvalidLLMResponseError,
        match="不是合法JSON",
    ):
        await provider.generate_answer(
            question="测试问题",
            evidence=make_evidence(),
        )

    mock_create.assert_awaited_once()


def test_prompt_defines_secondary_abstention_contract(
) -> None:
    """Prompt必须告诉LLM如何返回可校验的二次拒答。"""

    messages = build_knowledge_answer_messages(
        question="候选证据没有覆盖问题时怎么办？",
        evidence=make_evidence(),
    )

    # 第一条消息是约束模型行为的System Prompt。
    system_prompt = messages[0]["content"]

    # Prompt必须区分两种互斥结果：
    # 证据足够时正常回答；证据不足时明确拒答。
    assert "证据足够时" in system_prompt
    assert "证据不足时" in system_prompt

    # 拒答结构必须明确包含状态字段和空证据列表，
    # 否则模型仍可能返回当前Schema无法表达的结果。
    assert '"abstained": true' in system_prompt
    assert '"used_evidence_ids": []' in system_prompt


@pytest.mark.asyncio
async def test_provider_accepts_secondary_abstention(
) -> None:
    """合法拒答JSON应解析成KnowledgeAnswerDraft。"""

    fake_llm_content = json.dumps(
        {
            "answer": (
                "知识库没有足够证据回答该问题。"
            ),
            "used_evidence_ids": [],
            "abstained": True,
        },
        ensure_ascii=False,
    )

    # Mock替代真实OpenAI兼容服务，
    # 使测试只验证Provider的解析与校验流程。
    mock_client, mock_create = (
        create_mock_client_with_content(
            fake_llm_content
        )
    )
    provider = OpenAIKnowledgeAnswerProvider(
        client=mock_client,
        model="test-model",
    )

    # 预期流程：构造消息 -> await Mock接口 ->
    # json.loads() -> Pydantic校验 -> 返回拒答草稿。
    draft = await provider.generate_answer(
        question="候选证据是否足够？",
        evidence=make_evidence(),
    )

    assert draft.abstained is True
    assert draft.used_evidence_ids == []
    mock_create.assert_awaited_once()
