"""结构化诊断应用服务的异步离线测试。

本文件使用Fake门控检索器和Fake诊断草稿提供者，
不访问真实Embedding API、Chroma、关键词索引或LLM。

测试目标：
1. 检索查询只组合真正携带故障语义的请求字段；
2. API范围被转换成不可变的内部范围；
3. 门控拒绝时跳过LLM并返回稳定拒答报告；
4. 门控放行时调用LLM并映射真实Chunk引用；
5. 非法本地参数和非法依赖返回值在外部调用前后正确停止；
6. 已知生成失败被安全降级，未知程序错误不会被掩盖。
"""

import pytest

from app.errors import (
    InvalidLLMResponseError,
    LLMTimeoutError,
    LLMUpstreamError,
)
from app.schemas.diagnosis_llm import (
    DiagnosisDraftCause,
    DiagnosisDraftCheck,
    DiagnosisLLMDraft,
)
from app.schemas.diagnostics import (
    DiagnosisRequest,
    DiagnosisRetrievalScope,
)
from app.schemas.retrieval import (
    DocumentChunk,
    HybridRetrievedChunk,
)
from app.services.diagnosis_report_builder import (
    NO_LLM_PROMPT_VERSION,
)
from app.services.diagnosis_prompts import (
    DIAGNOSIS_PROMPT_VERSION,
)
from app.services.diagnosis_service import (
    DiagnosisService,
    build_diagnosis_retrieval_query,
    build_diagnosis_retrieval_scope,
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


# 固定64位十六进制值，分别模拟文档哈希和正文哈希。
# 它们只服务于Schema校验，不对应真实语料。
TEST_DOCUMENT_ID = "a" * 64
TEST_CONTENT_HASH = "b" * 64

# 诊断Service应把这个服务端请求标识写入最终报告。
TEST_REQUEST_ID = "request-test-001"


def make_request(
    *,
    with_scope: bool = False,
) -> DiagnosisRequest:
    """创建一份合法诊断请求，可选加入文档范围。"""

    scope = (
        DiagnosisRetrievalScope(
            document_ids=[TEST_DOCUMENT_ID],
            source_files=["robot-faults.txt"],
        )
        if with_scope
        else None
    )

    return DiagnosisRequest(
        robot_id="robot-runtime-001",
        symptom="机器人网络恢复后仍未继续任务",
        log_excerpt=(
            "ERR-NET-4001 heartbeat recovered"
        ),
        retrieval_scope=scope,
    )


def make_candidate() -> HybridRetrievedChunk:
    """创建一条合法且可被LLM通过E1引用的融合候选。"""

    chunk = DocumentChunk(
        chunk_id=(
            TEST_DOCUMENT_ID
            + ":000000"
        ),
        document_id=TEST_DOCUMENT_ID,
        source_file="robot-faults.txt",
        page_or_section=(
            "section: ERR-NET-4001"
        ),
        chunk_index=0,
        content_hash=TEST_CONTENT_HASH,
        content=(
            "ERR-NET-4001网络恢复后，"
            "需要等待调度系统确认后再继续任务。"
        ),
    )

    return HybridRetrievedChunk(
        chunk=chunk,
        rrf_score=0.032,
        rank=1,
        vector_rank=1,
        vector_similarity=0.72,
        keyword_rank=1,
        keyword_score=12.0,
        matched_identifiers=(
            "err-net-4001",
        ),
    )


def make_query_rewrite() -> RewrittenRetrievalQuery:
    """创建门控结果中需要保留的查询改写审计数据。"""

    return RewrittenRetrievalQuery(
        original_query=(
            "机器人网络恢复后仍未继续任务 "
            "ERR-NET-4001"
        ),
        normalized_query=(
            "机器人网络恢复后仍未继续任务 "
            "err-net-4001"
        ),
        rewritten_query=(
            "机器人网络恢复后仍未继续任务 "
            "err-net-4001 network recovery resume task"
        ),
        added_terms=(
            "network recovery",
            "resume task",
        ),
        applied_rule_ids=(
            "identifier-err-net-4001",
        ),
        rewrite_version="deterministic-v1",
    )


def make_gated_result(
    *,
    accepted: bool,
) -> GatedHybridRetrievalResult:
    """创建门控放行或门控拒绝结果。"""

    if accepted:
        candidate = make_candidate()
        candidates = (candidate,)
        decision = HybridEvidenceGateDecision(
            accepted=True,
            reason="exact_identifier_support",
            policy_version=(
                "hybrid-evidence-gate-v1"
            ),
            evaluated_candidate_count=1,
            top_rrf_score=candidate.rrf_score,
            top_vector_similarity=(
                candidate.vector_similarity
            ),
            max_vector_similarity=(
                candidate.vector_similarity
            ),
            dual_path_candidate_count=1,
            identifier_candidate_count=1,
            model_lexical_candidate_count=0,
            supporting_chunk_ids=(
                candidate.chunk.chunk_id,
            ),
        )
    else:
        candidates = ()
        decision = HybridEvidenceGateDecision(
            accepted=False,
            reason="no_candidates",
            policy_version=(
                "hybrid-evidence-gate-v1"
            ),
            evaluated_candidate_count=0,
            top_rrf_score=None,
            top_vector_similarity=None,
            max_vector_similarity=None,
            dual_path_candidate_count=0,
            identifier_candidate_count=0,
            model_lexical_candidate_count=0,
            supporting_chunk_ids=(),
        )

    return GatedHybridRetrievalResult(
        query_rewrite=make_query_rewrite(),
        retrieved_chunks=candidates,
        decision=decision,
    )


def make_completed_draft() -> DiagnosisLLMDraft:
    """创建引用E1且满足安全约束的LLM诊断草稿。"""

    return DiagnosisLLMDraft(
        status="completed",
        possible_causes=[
            DiagnosisDraftCause(
                description=(
                    "可能仍在等待调度系统确认"
                ),
                evidence_ids=["E1"],
            )
        ],
        next_checks=[
            DiagnosisDraftCheck(
                description=(
                    "检查调度系统是否已经确认网络恢复"
                ),
                evidence_ids=["E1"],
                risk_level="low",
                requires_qualified_person=False,
            )
        ],
        risk_level="low",
        missing_information=[],
        abstained=False,
    )


class FakeGatedRetriever:
    """记录诊断Service传入的查询、深度和范围。"""

    def __init__(
        self,
        *,
        result: object,
    ) -> None:
        """保存预设下游返回值和调用记录。"""

        self.result = result
        self.calls: list[
            dict[str, object]
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
        """模拟完整的查询改写、混合检索和门控。"""

        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "retrieval_scope": retrieval_scope,
            }
        )

        # 有测试会故意提供错误返回类型，
        # 用于验证Service的运行时边界检查。
        return self.result  # type: ignore[return-value]


class FakeDiagnosisDraftProvider:
    """记录LLM输入，并返回草稿或抛出预设异常。"""

    def __init__(
        self,
        *,
        draft: DiagnosisLLMDraft | None = None,
        error: Exception | None = None,
    ) -> None:
        """保存预设结果、预设异常和调用记录。"""

        self.draft = draft
        self.error = error
        self.calls: list[
            dict[str, object]
        ] = []

    async def generate_draft(
        self,
        *,
        request: DiagnosisRequest,
        evidence: tuple[
            HybridRetrievedChunk,
            ...,
        ],
    ) -> DiagnosisLLMDraft:
        """模拟LLM结构化诊断，不执行网络请求。"""

        self.calls.append(
            {
                "request": request,
                "evidence": evidence,
            }
        )

        # error用于离线模拟超时、上游错误等失败。
        # 异常在记录调用后抛出，便于测试确认：
        # Service确实进入过LLM生成阶段。
        if self.error is not None:
            raise self.error

        # 没有异常时必须配置一份草稿。
        # AssertionError表示测试替身本身配置错误，
        # 不是需要生产代码降级的业务异常。
        if self.draft is None:
            raise AssertionError(
                "FakeDiagnosisDraftProvider"
                "没有配置draft"
            )

        return self.draft


def test_retrieval_query_uses_symptom_and_log_but_not_robot_id(
) -> None:
    """检索文本应保留故障语义，不加入无关实例编号。"""

    request = make_request()

    query = build_diagnosis_retrieval_query(
        request
    )

    assert query == (
        request.symptom
        + "\n"
        + request.log_excerpt
    )
    assert request.robot_id not in query


def test_unrestricted_request_builds_no_internal_scope(
) -> None:
    """API未提供范围时必须使用None表示全库检索。"""

    assert (
        build_diagnosis_retrieval_scope(
            make_request()
        )
        is None
    )


def test_api_scope_is_copied_into_immutable_internal_scope(
) -> None:
    """Pydantic列表范围应转换成独立的tuple范围。"""

    request = make_request(
        with_scope=True
    )

    scope = build_diagnosis_retrieval_scope(
        request
    )

    assert isinstance(
        scope,
        RetrievalScopeFilter,
    )
    assert scope.document_ids == (
        TEST_DOCUMENT_ID,
    )
    assert scope.source_files == (
        "robot-faults.txt",
    )

    # 修改请求模型中的可变list，不能改变已经创建的
    # frozen内部范围对象。
    assert request.retrieval_scope is not None
    request.retrieval_scope.source_files.append(
        "later-added.txt"
    )
    assert scope.source_files == (
        "robot-faults.txt",
    )


@pytest.mark.asyncio
async def test_gate_rejection_skips_llm_and_returns_stable_report(
) -> None:
    """证据门控拒绝时不得调用LLM。"""

    request = make_request()
    gated_retriever = FakeGatedRetriever(
        result=make_gated_result(
            accepted=False
        )
    )
    draft_provider = FakeDiagnosisDraftProvider(
        draft=make_completed_draft()
    )
    service = DiagnosisService(
        retrieval_provider=gated_retriever,
        draft_provider=draft_provider,
    )

    report = await service.diagnose(
        request,
        TEST_REQUEST_ID,
    )

    assert len(gated_retriever.calls) == 1
    assert draft_provider.calls == []
    assert report.request_id == TEST_REQUEST_ID
    assert report.status == "abstained"
    assert report.abstained is True
    assert report.prompt_version == (
        NO_LLM_PROMPT_VERSION
    )
    assert report.evidence == []
    assert report.possible_causes == []
    assert report.next_checks == []
    assert report.missing_information


@pytest.mark.asyncio
async def test_accepted_evidence_calls_llm_and_builds_traceable_report(
) -> None:
    """门控放行后应调用LLM并把E1映射为真实Chunk ID。"""

    request = make_request(
        with_scope=True
    )
    retrieval_result = make_gated_result(
        accepted=True
    )
    gated_retriever = FakeGatedRetriever(
        result=retrieval_result
    )
    draft_provider = FakeDiagnosisDraftProvider(
        draft=make_completed_draft()
    )
    service = DiagnosisService(
        retrieval_provider=gated_retriever,
        draft_provider=draft_provider,
        retrieval_top_k=4,
    )

    report = await service.diagnose(
        request,
        f"  {TEST_REQUEST_ID}  ",
    )

    assert len(gated_retriever.calls) == 1
    retrieval_call = gated_retriever.calls[0]
    assert retrieval_call["query"] == (
        request.symptom
        + "\n"
        + request.log_excerpt
    )
    assert retrieval_call["top_k"] == 4

    internal_scope = retrieval_call[
        "retrieval_scope"
    ]
    assert isinstance(
        internal_scope,
        RetrievalScopeFilter,
    )
    assert internal_scope.document_ids == (
        TEST_DOCUMENT_ID,
    )

    assert len(draft_provider.calls) == 1
    assert (
        draft_provider.calls[0]["request"]
        is request
    )
    assert (
        draft_provider.calls[0]["evidence"]
        == retrieval_result.retrieved_chunks
    )

    assert report.request_id == TEST_REQUEST_ID
    assert report.status == "completed"
    assert report.abstained is False
    assert len(report.evidence) == 1

    real_chunk_id = (
        retrieval_result
        .retrieved_chunks[0]
        .chunk.chunk_id
    )
    assert report.evidence[0].chunk_id == (
        real_chunk_id
    )
    assert (
        report.possible_causes[0]
        .evidence_chunk_ids
        == [real_chunk_id]
    )
    assert (
        report.next_checks[0]
        .evidence_chunk_ids
        == [real_chunk_id]
    )


@pytest.mark.parametrize(
    (
        "invalid_top_k",
        "expected_exception",
        "expected_message",
    ),
    [
        (
            True,
            TypeError,
            "retrieval_top_k必须是整数",
        ),
        (
            0,
            ValueError,
            "retrieval_top_k必须大于或等于1",
        ),
    ],
    ids=[
        "boolean",
        "zero",
    ],
)
def test_constructor_rejects_invalid_retrieval_top_k(
    invalid_top_k: object,
    expected_exception: type[Exception],
    expected_message: str,
) -> None:
    """非法固定候选深度必须在服务构造阶段被拒绝。"""

    with pytest.raises(
        expected_exception,
        match=expected_message,
    ):
        DiagnosisService(
            retrieval_provider=(
                FakeGatedRetriever(
                    result=make_gated_result(
                        accepted=False
                    )
                )
            ),
            draft_provider=(
                FakeDiagnosisDraftProvider(
                    draft=make_completed_draft()
                )
            ),
            retrieval_top_k=(
                invalid_top_k  # type: ignore[arg-type]
            ),
        )


@pytest.mark.asyncio
async def test_blank_request_id_is_rejected_before_retrieval(
) -> None:
    """空白追踪标识不能触发检索或LLM调用。"""

    gated_retriever = FakeGatedRetriever(
        result=make_gated_result(
            accepted=False
        )
    )
    draft_provider = FakeDiagnosisDraftProvider(
        draft=make_completed_draft()
    )
    service = DiagnosisService(
        retrieval_provider=gated_retriever,
        draft_provider=draft_provider,
    )

    with pytest.raises(
        ValueError,
        match="request_id不能为空",
    ):
        await service.diagnose(
            make_request(),
            "   ",
        )

    assert gated_retriever.calls == []
    assert draft_provider.calls == []


@pytest.mark.asyncio
async def test_invalid_retrieval_result_stops_before_llm(
) -> None:
    """下游违反结果契约时不能继续调用诊断LLM。"""

    gated_retriever = FakeGatedRetriever(
        result={"accepted": True}
    )
    draft_provider = FakeDiagnosisDraftProvider(
        draft=make_completed_draft()
    )
    service = DiagnosisService(
        retrieval_provider=gated_retriever,
        draft_provider=draft_provider,
    )

    with pytest.raises(
        TypeError,
        match=(
            "retrieval_provider必须返回"
            "GatedHybridRetrievalResult"
        ),
    ):
        await service.diagnose(
            make_request(),
            TEST_REQUEST_ID,
        )

    assert len(gated_retriever.calls) == 1
    assert draft_provider.calls == []


@pytest.mark.parametrize(
    (
        "generation_error",
        "expected_failure_reason",
        "expected_missing_information",
    ),
    [
        (
            InvalidLLMResponseError(
                "内部原始响应包含非法字段"
            ),
            "invalid_llm_response",
            "结构化诊断结果未通过数据和引用校验，"
            "需要重试或由人工复核",
        ),
        (
            LLMTimeoutError(
                "内部上游请求等待超时"
            ),
            "llm_timeout",
            "结构化诊断生成服务响应超时，"
            "需要稍后重试或由人工复核",
        ),
        (
            LLMUpstreamError(
                "内部上游地址返回502"
            ),
            "llm_upstream_error",
            "结构化诊断生成服务暂时不可用，"
            "需要稍后重试或由人工复核",
        ),
    ],
    ids=[
        "invalid-llm-response",
        "llm-timeout",
        "llm-upstream-error",
    ],
)
@pytest.mark.asyncio
async def test_known_generation_error_returns_stable_report(
    generation_error: Exception,
    expected_failure_reason: str,
    expected_missing_information: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """已知LLM生成失败应转换成不泄露细节的拒答。"""

    # 只收集DiagnosisService产生的WARNING及以上日志。
    # caplog由pytest自动注入，不需要手动创建。
    caplog.set_level(
        "WARNING",
        logger=(
            "app.services.diagnosis_service"
        ),
    )

    retrieval_result = make_gated_result(
        accepted=True
    )
    gated_retriever = FakeGatedRetriever(
        result=retrieval_result
    )
    draft_provider = FakeDiagnosisDraftProvider(
        error=generation_error
    )
    service = DiagnosisService(
        retrieval_provider=gated_retriever,
        draft_provider=draft_provider,
    )

    report = await service.diagnose(
        make_request(),
        TEST_REQUEST_ID,
    )

    # 检索和LLM都应各执行一次，
    # 证明这是门控放行后的生成失败分支。
    assert len(gated_retriever.calls) == 1
    assert len(draft_provider.calls) == 1

    assert report.status == "abstained"
    assert report.abstained is True
    assert report.prompt_version == (
        DIAGNOSIS_PROMPT_VERSION
    )

    # 已通过门控的真实候选仍可用于审计，
    # 但失败的LLM不能产生原因和检查建议。
    assert len(report.evidence) == 1
    assert report.possible_causes == []
    assert report.next_checks == []
    assert report.risk_level == "unknown"
    assert report.missing_information == [
        expected_missing_information
    ]

    # 客户端只能看到稳定公开消息，
    # 不能看到异常中模拟的内部实现细节。
    assert str(generation_error) not in (
        report.missing_information
    )

    # 客户端报告不暴露内部异常，
    # 但服务端必须保留请求ID、稳定原因代码
    # 和项目异常已经清理过的诊断信息。
    matching_records = [
        record
        for record in caplog.records
        if record.name == (
            "app.services.diagnosis_service"
        )
        and (
            "diagnosis_generation_degraded"
            in record.getMessage()
        )
    ]

    assert len(matching_records) == 1

    log_message = (
        matching_records[0].getMessage()
    )
    assert (
        f"request_id={TEST_REQUEST_ID}"
        in log_message
    )
    assert (
        f"reason={expected_failure_reason}"
        in log_message
    )
    assert str(generation_error) in log_message


@pytest.mark.asyncio
async def test_unknown_draft_reference_is_degraded(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """报告组装时发现虚构E编号也应安全降级。"""

    caplog.set_level(
        "WARNING",
        logger=(
            "app.services.diagnosis_service"
        ),
    )

    invalid_reference_draft = DiagnosisLLMDraft(
        status="completed",
        possible_causes=[
            DiagnosisDraftCause(
                description="可能仍在等待调度确认",
                # E999满足编号格式，
                # 但不在本次只含E1的证据白名单中。
                evidence_ids=["E999"],
            )
        ],
        next_checks=[
            DiagnosisDraftCheck(
                description="检查调度确认状态",
                evidence_ids=["E1"],
                risk_level="low",
                requires_qualified_person=False,
            )
        ],
        risk_level="low",
        missing_information=[],
        abstained=False,
    )
    retrieval_result = make_gated_result(
        accepted=True
    )
    service = DiagnosisService(
        retrieval_provider=FakeGatedRetriever(
            result=retrieval_result
        ),
        draft_provider=FakeDiagnosisDraftProvider(
            draft=invalid_reference_draft
        ),
    )

    report = await service.diagnose(
        make_request(),
        TEST_REQUEST_ID,
    )

    # generate_draft()本身成功，
    # 但Report Builder会把未知E999识别成
    # InvalidLLMResponseError。
    assert report.status == "abstained"
    assert report.possible_causes == []
    assert report.next_checks == []
    assert report.missing_information == [
        "结构化诊断结果未通过数据和引用校验，"
        "需要重试或由人工复核"
    ]

    # 组装阶段发生的未知引用也必须留下
    # 与本次request_id关联的内部诊断日志。
    assert any(
        (
            f"request_id={TEST_REQUEST_ID}"
            in record.getMessage()
        )
        and (
            "reason=invalid_llm_response"
            in record.getMessage()
        )
        and "E999" in record.getMessage()
        for record in caplog.records
        if record.name == (
            "app.services.diagnosis_service"
        )
    )


@pytest.mark.asyncio
async def test_unexpected_generation_error_is_not_hidden(
) -> None:
    """未列入白名单的程序错误必须继续抛出。"""

    unexpected_error = RuntimeError(
        "unexpected programming defect"
    )
    service = DiagnosisService(
        retrieval_provider=FakeGatedRetriever(
            result=make_gated_result(
                accepted=True
            )
        ),
        draft_provider=FakeDiagnosisDraftProvider(
            error=unexpected_error
        ),
    )

    # 不能使用except Exception把所有Bug伪装成拒答，
    # 否则真正的程序缺陷会失去日志和告警信号。
    with pytest.raises(
        RuntimeError,
        match="unexpected programming defect",
    ):
        await service.diagnose(
            make_request(),
            TEST_REQUEST_ID,
        )
