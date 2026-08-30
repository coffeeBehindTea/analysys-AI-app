"""search_knowledge纯检索工具的数据契约、Handler和执行链测试。

本文件完全使用内存Fake门控检索器，不会连接Embedding、
Chroma、生成式LLM或真实HTTP接口，也不会产生外部API费用。

这里重点锁定一个架构边界：search_knowledge只取得门控确认的
真实Chunk，不得在Agent工具内部再次调用知识回答LLM。
"""

from typing import (
    Any,
)

import pytest

from pydantic import (
    BaseModel,
    ValidationError,
)

from app.agent.executor import (
    ToolExecutor,
)
from app.agent.evidence_store import (
    ConfirmedEvidenceStore,
)
from app.agent.registry import (
    ToolRegistry,
)
from app.agent.tools.search_knowledge import (
    SEARCH_KNOWLEDGE_TOOL_DEFINITION,
    SearchKnowledgeToolHandler,
)
from app.schemas.agent import (
    ToolCall,
)
from app.schemas.agent_tools import (
    SearchKnowledgeToolInput,
    SearchKnowledgeToolOutput,
)
from app.schemas.retrieval import (
    DocumentChunk,
    HybridRetrievedChunk,
)
from app.services.gated_hybrid_retrieval import (
    GatedHybridRetrievalResult,
)
from app.services.hybrid_retrieval_gate import (
    HybridEvidenceGateDecision,
)
from app.services.query_rewriting import (
    RewrittenRetrievalQuery,
)
from app.services.retrieval_scope import (
    RetrievalScopeFilter,
)


# 测试引用使用固定的合法SHA-256格式文档标识。
# 固定值让断言可重复，不依赖真实知识库内容。
TEST_DOCUMENT_ID = "a" * 64

# DocumentChunk要求正文哈希是64位小写十六进制文本。
TEST_CONTENT_HASH = "b" * 64


class FakeGatedRetrievalProvider:
    """记录检索参数并返回预设门控结果的异步Fake。"""

    def __init__(
        self,
        *,
        result: object,
        error: Exception | None = None,
    ) -> None:
        # result使用object，允许错误返回值测试传入dict。
        self.result = result
        self.error = error
        self.calls: list[
            tuple[
                str,
                int,
                RetrievalScopeFilter | None,
            ]
        ] = []

    async def retrieve(
        self,
        *,
        query: str,
        top_k: int = 3,
        retrieval_scope: (
            RetrievalScopeFilter | None
        ) = None,
    ) -> GatedHybridRetrievalResult:
        """记录调用，然后返回结果或抛出预设异常。"""

        self.calls.append(
            (query, top_k, retrieval_scope)
        )

        if self.error is not None:
            raise self.error

        return self.result  # type: ignore[return-value]


class SyncGatedRetrievalProvider:
    """故意使用同步retrieve()的错误依赖。"""

    def retrieve(
        self,
        *,
        query: str,
        top_k: int = 3,
        retrieval_scope: (
            RetrievalScopeFilter | None
        ) = None,
    ) -> GatedHybridRetrievalResult:
        del query, top_k, retrieval_scope
        return make_accepted_result()


class UnrelatedInput(BaseModel):
    """用于验证Handler不会接受其他Pydantic模型。"""

    value: str


def make_candidate(
) -> HybridRetrievedChunk:
    """创建一条门控可确认的真实结构化候选。"""

    return HybridRetrievedChunk(
        chunk=DocumentChunk(
            chunk_id=(
                f"{TEST_DOCUMENT_ID}:000003"
            ),
            document_id=TEST_DOCUMENT_ID,
            source_file=(
                "仓储机器人故障说明.txt"
            ),
            page_or_section=(
                "section: ERR-NET-4001"
            ),
            chunk_index=3,
            content_hash=TEST_CONTENT_HASH,
            content=(
                "ERR-NET-4001表示调度心跳中断，"
                "恢复网络后需要由调度系统核对状态。"
            ),
        ),
        rrf_score=0.0327,
        rank=1,
        vector_rank=1,
        vector_similarity=0.72,
        keyword_rank=1,
        keyword_score=10.0,
        matched_identifiers=(
            "err-net-4001",
        ),
    )


def make_result(
    *,
    accepted: bool,
) -> GatedHybridRetrievalResult:
    """创建放行或拒绝的完整门控检索结果。"""

    candidate = make_candidate()

    return GatedHybridRetrievalResult(
        query_rewrite=(
            RewrittenRetrievalQuery(
                original_query=(
                    "ERR-NET-4001如何恢复？"
                ),
                normalized_query=(
                    "err-net-4001如何恢复?"
                ),
                rewritten_query=(
                    "err-net-4001如何恢复? "
                    "network recovery"
                ),
                added_terms=(
                    "network recovery",
                ),
                applied_rule_ids=(
                    "identifier-err-net-4001",
                ),
                rewrite_version=(
                    "deterministic-v1"
                ),
            )
        ),
        retrieved_chunks=(candidate,),
        decision=(
            HybridEvidenceGateDecision(
                accepted=accepted,
                reason=(
                    "exact_identifier_support"
                    if accepted
                    else (
                        "insufficient_combined_support"
                    )
                ),
                policy_version=(
                    "hybrid-evidence-gate-v1"
                ),
                evaluated_candidate_count=1,
                top_rrf_score=0.0327,
                top_vector_similarity=0.72,
                max_vector_similarity=0.72,
                dual_path_candidate_count=1,
                identifier_candidate_count=1,
                model_lexical_candidate_count=0,
                supporting_chunk_ids=(
                    (
                        candidate.chunk.chunk_id,
                    )
                    if accepted
                    else ()
                ),
            )
        ),
    )


def make_accepted_result(
) -> GatedHybridRetrievalResult:
    """创建精确错误码证据触发放行的结果。"""

    return make_result(accepted=True)


def make_rejected_result(
) -> GatedHybridRetrievalResult:
    """创建候选存在但组合支持不足的拒绝结果。"""

    return make_result(accepted=False)


def make_handler_and_provider(
    *,
    result: object,
    error: Exception | None = None,
) -> tuple[
    SearchKnowledgeToolHandler,
    FakeGatedRetrievalProvider,
]:
    """创建共享同一个Fake检索器的Handler测试组合。"""

    provider = FakeGatedRetrievalProvider(
        result=result,
        error=error,
    )
    handler = SearchKnowledgeToolHandler(
        retrieval_provider=provider,
        evidence_store=(
            ConfirmedEvidenceStore()
        ),
    )

    return handler, provider


def make_registered_executor(
    *,
    result: object,
) -> tuple[
    ToolExecutor,
    FakeGatedRetrievalProvider,
]:
    """把真实工具定义和Fake Handler接入完整执行链。"""

    handler, provider = (
        make_handler_and_provider(
            result=result
        )
    )
    registry = ToolRegistry()
    registry.register(
        definition=(
            SEARCH_KNOWLEDGE_TOOL_DEFINITION
        ),
        handler=handler,
    )

    return (
        ToolExecutor(registry=registry),
        provider,
    )


def test_search_input_strips_query_and_uses_default_top_k(
) -> None:
    """输入契约应清理查询空白并默认使用Top-3。"""

    tool_input = SearchKnowledgeToolInput(
        query="  ERR-NET-4001如何恢复？  "
    )

    assert tool_input.query == (
        "ERR-NET-4001如何恢复？"
    )
    assert tool_input.top_k == 3


@pytest.mark.parametrize(
    ("arguments", "expected_error_type"),
    [
        (
            {
                "query": "有效问题",
                "shell_command": "whoami",
            },
            "extra_forbidden",
        ),
        (
            {"query": "有效问题", "top_k": 0},
            "greater_than_equal",
        ),
        (
            {"query": "有效问题", "top_k": 11},
            "less_than_equal",
        ),
    ],
)
def test_search_input_rejects_extra_or_out_of_range_values(
    arguments: dict[str, Any],
    expected_error_type: str,
) -> None:
    """输入契约应拒绝额外字段和越界Top-K。"""

    with pytest.raises(
        ValidationError
    ) as exc_info:
        SearchKnowledgeToolInput.model_validate(
            arguments
        )

    error_types = {
        error["type"]
        for error in exc_info.value.errors()
    }

    assert expected_error_type in error_types


def test_search_output_requires_at_least_one_citation(
) -> None:
    """成功输出不能缺少门控确认的真实引用。"""

    with pytest.raises(
        ValidationError
    ) as exc_info:
        SearchKnowledgeToolOutput(
            citations=[],
            retrieval_ms=1.0,
        )

    assert (
        exc_info.value.errors()[0]["type"]
        == "too_short"
    )


def test_definition_exposes_safe_openai_schema(
) -> None:
    """工具定义应保持只读、低风险和受限参数Schema。"""

    definition = (
        SEARCH_KNOWLEDGE_TOOL_DEFINITION
    )
    openai_schema = (
        definition.to_openai_tool_schema()
    )

    assert definition.name == (
        "search_knowledge"
    )
    assert definition.risk_level == "low"
    assert definition.read_only is True
    assert definition.input_model is (
        SearchKnowledgeToolInput
    )
    assert definition.output_model is (
        SearchKnowledgeToolOutput
    )

    parameters = (
        openai_schema["function"]["parameters"]
    )
    assert set(parameters["properties"]) == {
        "query",
        "top_k",
    }
    assert parameters[
        "additionalProperties"
    ] is False


@pytest.mark.parametrize(
    "invalid_provider",
    [
        object(),
        SyncGatedRetrievalProvider(),
    ],
)
def test_handler_requires_async_retrieve(
    invalid_provider: object,
) -> None:
    """Handler构造时应拒绝缺少异步方法的依赖。"""

    with pytest.raises(
        TypeError,
        match=(
            "retrieval_provider.retrieve"
            "必须是异步方法"
        ),
    ):
        SearchKnowledgeToolHandler(
            retrieval_provider=(
                invalid_provider
            ),
            evidence_store=(
                ConfirmedEvidenceStore()
            ),
        )


@pytest.mark.asyncio
async def test_handler_converts_input_and_accepted_retrieval_result(
) -> None:
    """正常路径应把工具输入转成检索参数和证据输出。"""

    handler, provider = (
        make_handler_and_provider(
            result=make_accepted_result()
        )
    )

    result = await handler(
        SearchKnowledgeToolInput(
            query="ERR-NET-4001如何恢复？",
            top_k=5,
        )
    )

    assert provider.calls == [
        (
            "ERR-NET-4001如何恢复？",
            5,
            None,
        )
    ]

    assert isinstance(
        result,
        SearchKnowledgeToolOutput,
    )
    assert len(result.citations) == 1
    assert result.citations[0].chunk_id == (
        f"{TEST_DOCUMENT_ID}:000003"
    )
    assert result.citations[0].excerpt == (
        make_candidate().chunk.content
    )
    assert result.retrieval_ms >= 0.0


@pytest.mark.asyncio
async def test_accepted_result_records_confirmed_citations(
) -> None:
    """门控确认的真实引用应进入当前请求证据白名单。"""

    provider = FakeGatedRetrievalProvider(
        result=make_accepted_result()
    )
    evidence_store = ConfirmedEvidenceStore()
    handler = SearchKnowledgeToolHandler(
        retrieval_provider=provider,
        evidence_store=evidence_store,
    )

    output = await handler(
        SearchKnowledgeToolInput(
            query="ERR-NET-4001如何恢复？"
        )
    )

    assert output is not None
    citation = output.citations[0]
    assert len(evidence_store) == 1
    assert citation.chunk_id in evidence_store
    assert evidence_store.get(
        citation.chunk_id
    ) == citation


@pytest.mark.asyncio
async def test_handler_maps_gate_rejection_to_none(
) -> None:
    """确定性门控拒绝应转换成工具层的空结果。"""

    handler, provider = (
        make_handler_and_provider(
            result=make_rejected_result()
        )
    )

    result = await handler(
        SearchKnowledgeToolInput(
            query="知识库没有记录的问题"
        )
    )

    assert result is None
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_rejected_gate_does_not_record_evidence(
) -> None:
    """门控拒绝不能向当前请求白名单写入任何内容。"""

    provider = FakeGatedRetrievalProvider(
        result=make_rejected_result()
    )
    evidence_store = ConfirmedEvidenceStore()
    handler = SearchKnowledgeToolHandler(
        retrieval_provider=provider,
        evidence_store=evidence_store,
    )

    result = await handler(
        SearchKnowledgeToolInput(
            query="没有足够证据的问题"
        )
    )

    assert result is None
    assert len(evidence_store) == 0


@pytest.mark.asyncio
async def test_handler_rejects_unrelated_input_before_retrieval(
) -> None:
    """绕过Executor直接传错模型时不能调用检索器。"""

    handler, provider = (
        make_handler_and_provider(
            result=make_accepted_result()
        )
    )

    with pytest.raises(
        TypeError,
        match=(
            "tool_input必须是"
            "SearchKnowledgeToolInput"
        ),
    ):
        await handler(
            UnrelatedInput(value="wrong")
        )

    assert provider.calls == []


@pytest.mark.asyncio
async def test_handler_rejects_invalid_retrieval_result_type(
) -> None:
    """错误依赖返回dict时不能伪装成门控检索结果。"""

    handler, provider = (
        make_handler_and_provider(
            result={
                "retrieved_chunks": []
            }
        )
    )

    with pytest.raises(
        TypeError,
        match=(
            "retrieval_provider.retrieve"
            "必须返回"
            "GatedHybridRetrievalResult"
        ),
    ):
        await handler(
            SearchKnowledgeToolInput(
                query="测试错误返回值"
            )
        )

    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_handler_does_not_hide_retrieval_exception(
) -> None:
    """Handler应把检索异常交给ToolExecutor统一转换。"""

    upstream_error = RuntimeError(
        "仅用于离线测试的上游异常"
    )
    handler, provider = (
        make_handler_and_provider(
            result=make_accepted_result(),
            error=upstream_error,
        )
    )

    with pytest.raises(
        RuntimeError,
        match="仅用于离线测试的上游异常",
    ):
        await handler(
            SearchKnowledgeToolInput(
                query="测试异常传播"
            )
        )

    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_registered_executor_returns_success_output(
) -> None:
    """注册表到执行器的完整成功路径应返回可序列化结果。"""

    executor, provider = (
        make_registered_executor(
            result=make_accepted_result()
        )
    )

    result = await executor.execute(
        ToolCall(
            call_id="call_search_001",
            tool_name="search_knowledge",
            arguments={
                "query": "ERR-NET-4001如何恢复？",
                "top_k": 3,
            },
        )
    )

    assert result.status == "success"
    assert result.error_code is None
    assert result.output is not None
    assert "answer" not in result.output
    assert result.output["citations"][0][
        "chunk_id"
    ] == f"{TEST_DOCUMENT_ID}:000003"
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_registered_executor_maps_gate_rejection_to_empty(
) -> None:
    """门控拒绝应穿过完整执行链成为稳定empty结果。"""

    executor, provider = (
        make_registered_executor(
            result=make_rejected_result()
        )
    )

    result = await executor.execute(
        ToolCall(
            call_id="call_search_002",
            tool_name="search_knowledge",
            arguments={
                "query": "没有证据的问题"
            },
        )
    )

    assert result.status == "empty"
    assert result.error_code == "empty_result"
    assert result.output is None
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_registered_executor_rejects_invalid_arguments_early(
) -> None:
    """非法参数应在Handler和检索器执行前被拒绝。"""

    executor, provider = (
        make_registered_executor(
            result=make_accepted_result()
        )
    )

    result = await executor.execute(
        ToolCall(
            call_id="call_search_003",
            tool_name="search_knowledge",
            arguments={
                "query": "有效问题",
                "top_k": 0,
            },
        )
    )

    assert result.status == "rejected"
    assert result.error_code == (
        "invalid_tool_arguments"
    )
    assert result.output is None
    assert provider.calls == []
