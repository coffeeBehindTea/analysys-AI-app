"""可信结构化诊断报告组装器的离线测试。

本模块不调用LLM、Embedding、Chroma或HTTP服务。

测试范围：
1. 请求中的症状与日志是否保留真实来源；
2. HybridRetrievedChunk是否转换成可信DiagnosisEvidence；
3. E1、E2是否映射到本次检索的真实Chunk ID；
4. 未引用候选是否不会出现在非拒答报告中；
5. E999等未知引用是否被白名单拒绝；
6. 模型拒答与门控拒答是否形成不同的稳定报告；
7. LLM非法输出、超时和上游失败是否安全降级；
8. partial和高风险字段是否被完整保留；
9. 错误运行时类型和错误调用路径是否被拒绝。
"""

import pytest

from app.errors import (
    InvalidLLMResponseError,
)
from app.schemas.diagnosis_llm import (
    DiagnosisDraftCause,
    DiagnosisDraftCheck,
    DiagnosisLLMDraft,
)
from app.schemas.diagnostics import (
    DiagnosisRequest,
)
from app.schemas.retrieval import (
    DocumentChunk,
    HybridRetrievedChunk,
)
from app.services.diagnosis_report_builder import (
    NO_LLM_PROMPT_VERSION,
    build_diagnosis_evidence,
    build_diagnosis_report_from_draft,
    build_diagnosis_symptoms,
    build_gate_abstained_report,
    build_generation_abstained_report,
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


# 所有候选来自同一份测试文档。
# 64位小写十六进制文本满足DocumentId契约。
TEST_DOCUMENT_ID = "e" * 64
TEST_CONTENT_HASH = "f" * 64

# 报告必须同时记录Prompt版本和门控版本。
TEST_GATE_VERSION = "hybrid-evidence-gate-v1"
TEST_REQUEST_ID = "request-diagnosis-001"


def make_request() -> DiagnosisRequest:
    """构造包含用户现象和脱敏日志的诊断请求。"""

    return DiagnosisRequest(
        robot_id="robot-001",
        symptom="网络恢复后机器人仍未继续任务",
        log_excerpt=(
            "ERR-NET-4001 heartbeat recovered"
        ),
    )


def make_candidate(
    *,
    rank: int,
) -> HybridRetrievedChunk:
    """构造排名和真实来源可区分的混合检索候选。"""

    chunk_id = (
        f"{TEST_DOCUMENT_ID}:{rank:06d}"
    )

    return HybridRetrievedChunk(
        chunk=DocumentChunk(
            chunk_id=chunk_id,
            document_id=TEST_DOCUMENT_ID,
            source_file="robot-faults.txt",
            page_or_section=(
                f"section: evidence-{rank}"
            ),
            chunk_index=rank - 1,
            content_hash=TEST_CONTENT_HASH,
            content=f"第{rank}条真实测试证据",
        ),
        rrf_score=0.04 - rank / 1_000,
        rank=rank,
        vector_rank=rank,
        vector_similarity=0.70 - rank / 100,
        keyword_rank=rank,
        keyword_score=10.0 - rank / 10,
        matched_identifiers=(
            "err-net-4001",
        ),
    )


def make_retrieval_result(
    *,
    candidates: tuple[
        HybridRetrievedChunk,
        ...,
    ],
    accepted: bool,
    reason: str,
) -> GatedHybridRetrievalResult:
    """构造带查询改写和门控审计信息的检索结果。"""

    query_rewrite = RewrittenRetrievalQuery(
        original_query=(
            "ERR-NET-4001网络恢复后如何继续任务？"
        ),
        normalized_query=(
            "err-net-4001网络恢复后如何继续任务?"
        ),
        rewritten_query=(
            "err-net-4001网络恢复后如何继续任务? "
            "network recovery resume task"
        ),
        added_terms=(
            "network recovery",
            "resume task",
        ),
        applied_rule_ids=(
            "zh-network-recovery",
            "zh-resume-task",
        ),
        rewrite_version="deterministic-v1",
    )

    vector_similarities = [
        candidate.vector_similarity
        for candidate in candidates
        if candidate.vector_similarity is not None
    ]

    decision = HybridEvidenceGateDecision(
        accepted=accepted,
        reason=reason,  # type: ignore[arg-type]
        policy_version=TEST_GATE_VERSION,
        evaluated_candidate_count=len(
            candidates
        ),
        top_rrf_score=(
            candidates[0].rrf_score
            if candidates
            else None
        ),
        top_vector_similarity=(
            candidates[0].vector_similarity
            if candidates
            else None
        ),
        max_vector_similarity=(
            max(vector_similarities)
            if vector_similarities
            else None
        ),
        dual_path_candidate_count=len(
            candidates
        ),
        identifier_candidate_count=len(
            candidates
        ),
        model_lexical_candidate_count=0,
        supporting_chunk_ids=(
            tuple(
                candidate.chunk.chunk_id
                for candidate in candidates
            )
            if accepted
            else ()
        ),
    )

    return GatedHybridRetrievalResult(
        query_rewrite=query_rewrite,
        retrieved_chunks=candidates,
        decision=decision,
    )


def make_completed_draft(
    *,
    cause_evidence_ids: list[str] | None = None,
    check_evidence_ids: list[str] | None = None,
) -> DiagnosisLLMDraft:
    """构造可指定临时引用编号的完整诊断草稿。"""

    return DiagnosisLLMDraft(
        status="completed",
        possible_causes=[
            DiagnosisDraftCause(
                description=(
                    "调度通信可能曾经中断"
                ),
                evidence_ids=(
                    cause_evidence_ids
                    if cause_evidence_ids is not None
                    else ["E2"]
                ),
            )
        ],
        next_checks=[
            DiagnosisDraftCheck(
                description=(
                    "核对机器人与调度任务状态"
                ),
                evidence_ids=(
                    check_evidence_ids
                    if check_evidence_ids is not None
                    else ["E1"]
                ),
                risk_level="low",
                requires_qualified_person=False,
            )
        ],
        risk_level="low",
        missing_information=[],
        abstained=False,
    )


def test_symptoms_preserve_request_sources(
) -> None:
    """用户描述和日志必须分别标记其真实来源。"""

    symptoms = build_diagnosis_symptoms(
        make_request()
    )

    assert [
        symptom.model_dump()
        for symptom in symptoms
    ] == [
        {
            "description": (
                "网络恢复后机器人仍未继续任务"
            ),
            "source": "user_report",
        },
        {
            "description": (
                "ERR-NET-4001 heartbeat recovered"
            ),
            "source": "log_excerpt",
        },
    ]


def test_evidence_copies_trusted_candidate_fields(
) -> None:
    """最终证据字段必须全部复制自真实候选。"""

    candidate = make_candidate(rank=1)
    evidence = build_diagnosis_evidence(
        candidate
    )

    assert evidence.chunk_id == (
        candidate.chunk.chunk_id
    )
    assert evidence.document_id == (
        candidate.chunk.document_id
    )
    assert evidence.source_file == (
        candidate.chunk.source_file
    )
    assert evidence.page_or_section == (
        candidate.chunk.page_or_section
    )
    assert evidence.rank == candidate.rank
    assert evidence.rrf_score == (
        candidate.rrf_score
    )
    assert evidence.vector_similarity == (
        candidate.vector_similarity
    )
    assert evidence.excerpt == (
        candidate.chunk.content
    )


def test_completed_draft_maps_ids_to_real_chunks(
) -> None:
    """E1和E2必须映射成对应排名的真实Chunk ID。"""

    rank_one = make_candidate(rank=1)
    rank_two = make_candidate(rank=2)
    retrieval_result = make_retrieval_result(
        candidates=(rank_one, rank_two),
        accepted=True,
        reason="exact_identifier_support",
    )

    report = build_diagnosis_report_from_draft(
        request_id=TEST_REQUEST_ID,
        request=make_request(),
        retrieval_result=retrieval_result,
        draft=make_completed_draft(),
    )

    assert report.request_id == TEST_REQUEST_ID
    assert report.prompt_version == (
        "diagnosis-evidence-v1"
    )
    assert report.gate_version == (
        TEST_GATE_VERSION
    )
    assert report.status == "completed"
    assert report.abstained is False

    # 证据列表始终按真实RRF排名排列。
    assert [
        item.chunk_id
        for item in report.evidence
    ] == [
        rank_one.chunk.chunk_id,
        rank_two.chunk.chunk_id,
    ]

    # 草稿中的原因引用E2，
    # 最终报告必须引用rank=2的真实Chunk ID。
    assert report.possible_causes[
        0
    ].evidence_chunk_ids == [
        rank_two.chunk.chunk_id
    ]

    # 草稿中的检查项引用E1，
    # 最终报告必须引用rank=1的真实Chunk ID。
    assert report.next_checks[
        0
    ].evidence_chunk_ids == [
        rank_one.chunk.chunk_id
    ]


def test_unreferenced_candidate_is_not_exposed(
) -> None:
    """非拒答报告只返回模型真正引用的证据。"""

    candidates = tuple(
        make_candidate(rank=rank)
        for rank in (1, 2, 3)
    )
    retrieval_result = make_retrieval_result(
        candidates=candidates,
        accepted=True,
        reason="exact_identifier_support",
    )

    report = build_diagnosis_report_from_draft(
        request_id=TEST_REQUEST_ID,
        request=make_request(),
        retrieval_result=retrieval_result,
        draft=make_completed_draft(),
    )

    # 草稿只引用E1和E2，因此rank=3不进入最终证据列表。
    assert [
        item.rank
        for item in report.evidence
    ] == [1, 2]


def test_unknown_evidence_id_is_rejected(
) -> None:
    """格式合法但不在本次白名单中的E999必须失败。"""

    retrieval_result = make_retrieval_result(
        candidates=(make_candidate(rank=1),),
        accepted=True,
        reason="exact_identifier_support",
    )

    with pytest.raises(
        InvalidLLMResponseError,
        match=(
            "结构化诊断引用了未提供的"
            "证据编号：E999"
        ),
    ):
        build_diagnosis_report_from_draft(
            request_id=TEST_REQUEST_ID,
            request=make_request(),
            retrieval_result=retrieval_result,
            draft=make_completed_draft(
                cause_evidence_ids=["E999"],
                check_evidence_ids=["E1"],
            ),
        )


def test_model_abstention_preserves_retrieved_evidence(
) -> None:
    """模型因冲突拒答时应保留候选供审计。"""

    candidates = (
        make_candidate(rank=1),
        make_candidate(rank=2),
    )
    retrieval_result = make_retrieval_result(
        candidates=candidates,
        accepted=True,
        reason="exact_identifier_support",
    )
    abstained_draft = DiagnosisLLMDraft(
        status="abstained",
        possible_causes=[],
        next_checks=[],
        risk_level="unknown",
        missing_information=[
            "两条证据对恢复条件的描述存在冲突"
        ],
        abstained=True,
    )

    report = build_diagnosis_report_from_draft(
        request_id=TEST_REQUEST_ID,
        request=make_request(),
        retrieval_result=retrieval_result,
        draft=abstained_draft,
    )

    assert report.abstained is True
    assert report.possible_causes == []
    assert report.next_checks == []
    assert len(report.evidence) == 2
    assert report.missing_information == [
        "两条证据对恢复条件的描述存在冲突"
    ]


def test_partial_draft_preserves_missing_information(
) -> None:
    """partial草稿的已知结论和信息缺口都应保留。"""

    retrieval_result = make_retrieval_result(
        candidates=(make_candidate(rank=1),),
        accepted=True,
        reason="general_dual_path_support",
    )
    partial_draft = DiagnosisLLMDraft(
        status="partial",
        possible_causes=[
            DiagnosisDraftCause(
                description="可能发生过通信中断",
                evidence_ids=["E1"],
            )
        ],
        next_checks=[],
        risk_level="unknown",
        missing_information=[
            "仍需确认当前调度任务状态"
        ],
        abstained=False,
    )

    report = build_diagnosis_report_from_draft(
        request_id=TEST_REQUEST_ID,
        request=make_request(),
        retrieval_result=retrieval_result,
        draft=partial_draft,
    )

    assert report.status == "partial"
    assert len(report.possible_causes) == 1
    assert report.missing_information == [
        "仍需确认当前调度任务状态"
    ]


def test_high_risk_check_preserves_qualification_boundary(
) -> None:
    """高风险标记和资质人员要求不能在映射时丢失。"""

    retrieval_result = make_retrieval_result(
        candidates=(make_candidate(rank=1),),
        accepted=True,
        reason="general_dual_path_support",
    )
    high_risk_draft = DiagnosisLLMDraft(
        status="completed",
        possible_causes=[],
        next_checks=[
            DiagnosisDraftCheck(
                description=(
                    "由有资质人员确认高压连接状态"
                ),
                evidence_ids=["E1"],
                risk_level="high",
                requires_qualified_person=True,
            )
        ],
        risk_level="high",
        missing_information=[],
        abstained=False,
    )

    report = build_diagnosis_report_from_draft(
        request_id=TEST_REQUEST_ID,
        request=make_request(),
        retrieval_result=retrieval_result,
        draft=high_risk_draft,
    )

    assert report.risk_level == "high"
    assert report.next_checks[0].risk_level == (
        "high"
    )
    assert report.next_checks[
        0
    ].requires_qualified_person is True


def test_no_candidate_gate_rejection_builds_stable_report(
) -> None:
    """没有候选时由代码生成拒答，不调用任何Prompt。"""

    retrieval_result = make_retrieval_result(
        candidates=(),
        accepted=False,
        reason="no_candidates",
    )

    report = build_gate_abstained_report(
        request_id=TEST_REQUEST_ID,
        request=make_request(),
        retrieval_result=retrieval_result,
    )

    assert report.status == "abstained"
    assert report.abstained is True
    assert report.prompt_version == (
        NO_LLM_PROMPT_VERSION
    )
    assert report.gate_version == (
        TEST_GATE_VERSION
    )
    assert report.evidence == []
    assert report.possible_causes == []
    assert report.next_checks == []
    assert report.risk_level == "unknown"
    assert report.missing_information == [
        "知识库未检索到可用于诊断的候选证据"
    ]


def test_weak_gate_rejection_does_not_expose_candidates(
) -> None:
    """未通过门控的候选不能作为正式证据返回。"""

    retrieval_result = make_retrieval_result(
        candidates=(make_candidate(rank=1),),
        accepted=False,
        reason="insufficient_combined_support",
    )

    report = build_gate_abstained_report(
        request_id=TEST_REQUEST_ID,
        request=make_request(),
        retrieval_result=retrieval_result,
    )

    assert report.evidence == []
    assert report.missing_information == [
        "候选证据未达到组合门控要求，"
        "需要补充更具体的故障代码、"
        "脱敏日志或现场状态"
    ]


def test_draft_path_requires_accepted_gate(
) -> None:
    """门控拒绝后不能绕过流程组装LLM报告。"""

    retrieval_result = make_retrieval_result(
        candidates=(make_candidate(rank=1),),
        accepted=False,
        reason="insufficient_combined_support",
    )

    with pytest.raises(
        ValueError,
        match="只有门控放行结果才能组装LLM诊断报告",
    ):
        build_diagnosis_report_from_draft(
            request_id=TEST_REQUEST_ID,
            request=make_request(),
            retrieval_result=retrieval_result,
            draft=make_completed_draft(
                cause_evidence_ids=["E1"],
                check_evidence_ids=["E1"],
            ),
        )


def test_gate_rejection_path_rejects_accepted_gate(
) -> None:
    """门控放行后不能伪装成代码控制的门控拒答。"""

    retrieval_result = make_retrieval_result(
        candidates=(make_candidate(rank=1),),
        accepted=True,
        reason="exact_identifier_support",
    )

    with pytest.raises(
        ValueError,
        match="门控已经放行，不能生成门控拒答报告",
    ):
        build_gate_abstained_report(
            request_id=TEST_REQUEST_ID,
            request=make_request(),
            retrieval_result=retrieval_result,
        )


@pytest.mark.parametrize(
    (
        "field_name",
        "invalid_value",
        "expected_message",
    ),
    [
        (
            "request",
            {"robot_id": "robot-001"},
            "request必须是DiagnosisRequest",
        ),
        (
            "retrieval_result",
            {"decision": {"accepted": True}},
            "retrieval_result必须是"
            "GatedHybridRetrievalResult",
        ),
        (
            "draft",
            {"status": "completed"},
            "draft必须是DiagnosisLLMDraft",
        ),
    ],
    ids=[
        "invalid-request",
        "invalid-retrieval-result",
        "invalid-draft",
    ],
)
def test_draft_builder_checks_runtime_types(
    field_name: str,
    invalid_value: object,
    expected_message: str,
) -> None:
    """类型注解不能替代报告边界的运行时检查。"""

    kwargs: dict[str, object] = {
        "request_id": TEST_REQUEST_ID,
        "request": make_request(),
        "retrieval_result": (
            make_retrieval_result(
                candidates=(
                    make_candidate(rank=1),
                ),
                accepted=True,
                reason="exact_identifier_support",
            )
        ),
        "draft": make_completed_draft(
            cause_evidence_ids=["E1"],
            check_evidence_ids=["E1"],
        ),
    }
    kwargs[field_name] = invalid_value

    with pytest.raises(
        TypeError,
        match=expected_message,
    ):
        build_diagnosis_report_from_draft(
            **kwargs,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    (
        "failure_reason",
        "expected_missing_information",
    ),
    [
        (
            "invalid_llm_response",
            "结构化诊断结果未通过数据和引用校验，"
            "需要重试或由人工复核",
        ),
        (
            "llm_timeout",
            "结构化诊断生成服务响应超时，"
            "需要稍后重试或由人工复核",
        ),
        (
            "llm_upstream_error",
            "结构化诊断生成服务暂时不可用，"
            "需要稍后重试或由人工复核",
        ),
    ],
    ids=[
        "invalid-output",
        "timeout",
        "upstream-error",
    ],
)
def test_generation_failure_builds_stable_abstained_report(
    failure_reason: str,
    expected_missing_information: str,
) -> None:
    """门控放行后的已知生成失败应转换成安全拒答报告。"""

    candidate = make_candidate(rank=1)
    retrieval_result = make_retrieval_result(
        candidates=(candidate,),
        accepted=True,
        reason="exact_identifier_support",
    )

    report = build_generation_abstained_report(
        request_id=TEST_REQUEST_ID,
        request=make_request(),
        retrieval_result=retrieval_result,
        failure_reason=(
            failure_reason  # type: ignore[arg-type]
        ),
    )

    assert report.status == "abstained"
    assert report.abstained is True

    # Prompt已经构造或准备调用，不能记录not-invoked。
    assert report.prompt_version == (
        "diagnosis-evidence-v1"
    )
    assert report.gate_version == (
        TEST_GATE_VERSION
    )

    # 门控已经确认候选具备组合支持，
    # 所以可以保留真实证据用于审计；
    # 但生成失败时不能给出原因或操作建议。
    assert len(report.evidence) == 1
    assert report.evidence[0].chunk_id == (
        candidate.chunk.chunk_id
    )
    assert report.possible_causes == []
    assert report.next_checks == []
    assert report.risk_level == "unknown"
    assert report.missing_information == [
        expected_missing_information
    ]


def test_generation_failure_path_requires_accepted_gate(
) -> None:
    """门控拒绝不能错误进入“LLM已尝试但失败”的分支。"""

    retrieval_result = make_retrieval_result(
        candidates=(),
        accepted=False,
        reason="no_candidates",
    )

    with pytest.raises(
        ValueError,
        match=(
            "只有门控放行后"
            "才能生成诊断生成降级报告"
        ),
    ):
        build_generation_abstained_report(
            request_id=TEST_REQUEST_ID,
            request=make_request(),
            retrieval_result=retrieval_result,
            failure_reason="llm_timeout",
        )


def test_generation_failure_rejects_unknown_reason(
) -> None:
    """任意字符串不能绕过版本化的降级原因白名单。"""

    retrieval_result = make_retrieval_result(
        candidates=(make_candidate(rank=1),),
        accepted=True,
        reason="exact_identifier_support",
    )

    with pytest.raises(
        ValueError,
        match="不支持的诊断生成降级原因",
    ):
        build_generation_abstained_report(
            request_id=TEST_REQUEST_ID,
            request=make_request(),
            retrieval_result=retrieval_result,
            failure_reason=(
                "database_error"  # type: ignore[arg-type]
            ),
        )
