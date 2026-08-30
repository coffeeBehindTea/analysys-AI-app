"""Agent诊断可信报告构造器的离线测试。

本文件使用固定Pydantic对象和请求级内存证据Store：

1. 不启动FastAPI；
2. 不调用AgentRunner或LLM；
3. 不执行知识检索；
4. 不读取真实文档；
5. 重点验证Chunk ID白名单和可信元数据重建。
"""

import pytest

from app.agent.evidence_store import ConfirmedEvidenceStore
from app.errors import InvalidLLMResponseError
from app.schemas.agent_api import AgentDiagnosisRequest
from app.schemas.agent_diagnosis import AgentDiagnosisDraft
from app.schemas.diagnostics import DiagnosisCause, DiagnosisCheck
from app.schemas.knowledge_query import KnowledgeCitation
from app.services.agent_diagnosis_report_builder import (
    AGENT_EVIDENCE_GATE_VERSION,
    build_agent_abstained_report,
    build_agent_diagnosis_report_from_draft,
)


# 固定值用于检查报告追踪信息和Planner版本是否原样保留。
TEST_REQUEST_ID = "request-agent-builder-001"
TEST_PROMPT_VERSION = "agent-tool-calling-v2"
TEST_DOCUMENT_ID = "a" * 64


def make_chunk_id(index: int) -> str:
    """根据零基序号创建稳定、互不重复的测试Chunk ID。"""

    return f"{TEST_DOCUMENT_ID}:{index:06d}"


def make_request() -> AgentDiagnosisRequest:
    """创建带用户现象和脱敏日志的Agent请求。"""

    return AgentDiagnosisRequest(
        robot_id="robot-001",
        symptom="网络恢复后仍未继续任务",
        log_excerpt="ERR-NET-4001 heartbeat timeout",
        task_goal="核对原因和安全恢复条件",
    )


def make_citation(
    *,
    index: int,
    rank: int = 1,
    rrf_score: float | None = 0.0327,
    similarity: float | None = 0.71,
) -> KnowledgeCitation:
    """创建一条模拟由真实知识库工具确认的引用。"""

    return KnowledgeCitation(
        chunk_id=make_chunk_id(index),
        document_id=TEST_DOCUMENT_ID,
        source_file="robot-fault-demo.txt",
        page_or_section=f"section: DEMO-{index}",
        chunk_index=index,
        rank=rank,
        similarity=similarity,
        rrf_score=rrf_score,
        excerpt=f"第{index}条固定测试证据正文",
    )


def make_store(
    *citations: KnowledgeCitation,
) -> ConfirmedEvidenceStore:
    """创建请求级证据Store并记录指定真实引用。"""

    store = ConfirmedEvidenceStore()
    store.record(citations)
    return store


def make_completed_draft(
    *,
    cause_chunk_id: str,
    check_chunk_id: str,
) -> AgentDiagnosisDraft:
    """创建分别引用原因和检查证据的完整草稿。"""

    return AgentDiagnosisDraft(
        status="completed",
        possible_causes=(
            DiagnosisCause(
                description="任务仍在等待调度状态核对",
                evidence_chunk_ids=[cause_chunk_id],
            ),
        ),
        next_checks=(
            DiagnosisCheck(
                description="核对RCS/RMS恢复命令",
                evidence_chunk_ids=[check_chunk_id],
                risk_level="medium",
                requires_qualified_person=False,
            ),
        ),
        risk_level="medium",
        abstained=False,
    )


def test_builder_reconstructs_real_metadata_and_global_ranks() -> None:
    """多次检索的重复局部rank应被转换成唯一报告顺序。"""

    first = make_citation(index=1, rank=1)
    second = make_citation(index=2, rank=1)
    store = make_store(first, second)
    draft = make_completed_draft(
        cause_chunk_id=first.chunk_id,
        check_chunk_id=second.chunk_id,
    )

    report = build_agent_diagnosis_report_from_draft(
        request_id=TEST_REQUEST_ID,
        request=make_request(),
        planner_prompt_version=TEST_PROMPT_VERSION,
        draft=draft,
        evidence_store=store,
    )

    assert [item.chunk_id for item in report.evidence] == [
        first.chunk_id,
        second.chunk_id,
    ]
    assert [item.rank for item in report.evidence] == [1, 2]
    assert report.evidence[0].source_file == first.source_file
    assert report.evidence[0].excerpt == first.excerpt
    assert report.evidence[1].rrf_score == second.rrf_score


def test_builder_deduplicates_chunk_referenced_by_cause_and_check() -> None:
    """同一证据支持多个结论时最终evidence只能出现一次。"""

    citation = make_citation(index=1)
    store = make_store(citation)
    draft = make_completed_draft(
        cause_chunk_id=citation.chunk_id,
        check_chunk_id=citation.chunk_id,
    )

    report = build_agent_diagnosis_report_from_draft(
        request_id=TEST_REQUEST_ID,
        request=make_request(),
        planner_prompt_version=TEST_PROMPT_VERSION,
        draft=draft,
        evidence_store=store,
    )

    assert len(report.evidence) == 1
    assert report.evidence[0].chunk_id == citation.chunk_id


def test_builder_rejects_unconfirmed_chunk_id() -> None:
    """格式合法但不在当前请求Store中的ID必须使整份报告失败。"""

    confirmed = make_citation(index=1)
    store = make_store(confirmed)
    unconfirmed_id = make_chunk_id(999)
    draft = make_completed_draft(
        cause_chunk_id=confirmed.chunk_id,
        check_chunk_id=unconfirmed_id,
    )

    with pytest.raises(
        InvalidLLMResponseError,
        match="未确认的Chunk ID",
    ):
        build_agent_diagnosis_report_from_draft(
            request_id=TEST_REQUEST_ID,
            request=make_request(),
            planner_prompt_version=TEST_PROMPT_VERSION,
            draft=draft,
            evidence_store=store,
        )


def test_builder_rejects_citation_without_rrf_score() -> None:
    """Agent报告不能为缺失的真实RRF分数编造替代数值。"""

    citation = make_citation(
        index=1,
        rrf_score=None,
        similarity=0.71,
    )
    store = make_store(citation)
    draft = make_completed_draft(
        cause_chunk_id=citation.chunk_id,
        check_chunk_id=citation.chunk_id,
    )

    with pytest.raises(
        InvalidLLMResponseError,
        match="缺少RRF分数",
    ):
        build_agent_diagnosis_report_from_draft(
            request_id=TEST_REQUEST_ID,
            request=make_request(),
            planner_prompt_version=TEST_PROMPT_VERSION,
            draft=draft,
            evidence_store=store,
        )


def test_builder_rejects_more_than_ten_unique_evidence_items() -> None:
    """Builder不能静默截断超过DiagnosisReport上限的引用。"""

    citations = tuple(
        make_citation(index=index)
        for index in range(11)
    )
    store = make_store(*citations)
    draft = AgentDiagnosisDraft(
        status="completed",
        possible_causes=(
            DiagnosisCause(
                description="固定多证据原因",
                evidence_chunk_ids=[
                    item.chunk_id
                    for item in citations[:10]
                ],
            ),
        ),
        next_checks=(
            DiagnosisCheck(
                description="固定第十一条证据检查",
                evidence_chunk_ids=[
                    citations[10].chunk_id,
                ],
                risk_level="low",
                requires_qualified_person=False,
            ),
        ),
        risk_level="medium",
        abstained=False,
    )

    with pytest.raises(
        InvalidLLMResponseError,
        match="最多只能引用10条",
    ):
        build_agent_diagnosis_report_from_draft(
            request_id=TEST_REQUEST_ID,
            request=make_request(),
            planner_prompt_version=TEST_PROMPT_VERSION,
            draft=draft,
            evidence_store=store,
        )


def test_builder_preserves_partial_missing_information() -> None:
    """部分诊断应保留已确认结论和模型声明的信息缺口。"""

    citation = make_citation(index=1)
    store = make_store(citation)
    draft = AgentDiagnosisDraft(
        status="partial",
        possible_causes=(
            DiagnosisCause(
                description="可能存在调度状态未核对",
                evidence_chunk_ids=[citation.chunk_id],
            ),
        ),
        risk_level="medium",
        missing_information=("缺少当前任务状态",),
        abstained=False,
    )

    report = build_agent_diagnosis_report_from_draft(
        request_id=TEST_REQUEST_ID,
        request=make_request(),
        planner_prompt_version=TEST_PROMPT_VERSION,
        draft=draft,
        evidence_store=store,
    )

    assert report.status == "partial"
    assert report.missing_information == ["缺少当前任务状态"]
    assert len(report.evidence) == 1


def test_builder_converts_abstained_draft_without_exposing_evidence() -> None:
    """拒答草稿不能把未用于结论的Store内容包装成正式证据。"""

    store = make_store(make_citation(index=1))
    draft = AgentDiagnosisDraft(
        status="abstained",
        risk_level="unknown",
        missing_information=("当前证据仍不足",),
        abstained=True,
    )

    report = build_agent_diagnosis_report_from_draft(
        request_id=TEST_REQUEST_ID,
        request=make_request(),
        planner_prompt_version=TEST_PROMPT_VERSION,
        draft=draft,
        evidence_store=store,
    )

    assert report.status == "abstained"
    assert report.evidence == []
    assert report.possible_causes == []
    assert report.next_checks == []


def test_abstained_builder_includes_explicit_test_draft_evidence() -> None:
    """拒答报告可以审计实际测试草案使用的已确认证据。"""

    selected = make_citation(index=1)
    unused = make_citation(index=2)
    store = make_store(selected, unused)
    draft = AgentDiagnosisDraft(
        status="abstained",
        risk_level="unknown",
        missing_information=("Agent后续规划安全中止",),
        abstained=True,
    )

    report = build_agent_diagnosis_report_from_draft(
        request_id=TEST_REQUEST_ID,
        request=make_request(),
        planner_prompt_version=TEST_PROMPT_VERSION,
        draft=draft,
        evidence_store=store,
        additional_evidence_chunk_ids=(
            selected.chunk_id,
        ),
    )

    # 只公开测试草案实际引用的selected，
    # 不把Store中所有曾经检索到的引用全部暴露。
    assert [item.chunk_id for item in report.evidence] == [
        selected.chunk_id,
    ]
    assert report.possible_causes == []
    assert report.next_checks == []


def test_builder_rejects_unconfirmed_additional_evidence() -> None:
    """测试草案提供的附加证据ID也必须通过同一请求白名单。"""

    draft = AgentDiagnosisDraft(
        status="abstained",
        risk_level="unknown",
        missing_information=("Agent后续规划安全中止",),
        abstained=True,
    )

    with pytest.raises(
        InvalidLLMResponseError,
        match="未确认的Chunk ID",
    ):
        build_agent_diagnosis_report_from_draft(
            request_id=TEST_REQUEST_ID,
            request=make_request(),
            planner_prompt_version=TEST_PROMPT_VERSION,
            draft=draft,
            evidence_store=ConfirmedEvidenceStore(),
            additional_evidence_chunk_ids=(
                make_chunk_id(999),
            ),
        )


def test_builder_marks_symptom_and_log_sources() -> None:
    """用户现象和日志必须保留各自来源而不能冒充知识证据。"""

    draft = AgentDiagnosisDraft(
        status="abstained",
        risk_level="unknown",
        missing_information=("当前证据不足",),
        abstained=True,
    )

    report = build_agent_diagnosis_report_from_draft(
        request_id=TEST_REQUEST_ID,
        request=make_request(),
        planner_prompt_version=TEST_PROMPT_VERSION,
        draft=draft,
        evidence_store=ConfirmedEvidenceStore(),
    )

    assert [item.source for item in report.symptoms] == [
        "user_report",
        "log_excerpt",
    ]
    assert report.symptoms[0].description == "网络恢复后仍未继续任务"
    assert report.symptoms[1].description == "ERR-NET-4001 heartbeat timeout"


def test_builder_records_agent_versions() -> None:
    """最终报告必须记录真实Planner版本和Agent证据门控版本。"""

    draft = AgentDiagnosisDraft(
        status="abstained",
        risk_level="unknown",
        missing_information=("当前证据不足",),
        abstained=True,
    )

    report = build_agent_diagnosis_report_from_draft(
        request_id=TEST_REQUEST_ID,
        request=make_request(),
        planner_prompt_version=TEST_PROMPT_VERSION,
        draft=draft,
        evidence_store=ConfirmedEvidenceStore(),
    )

    assert report.prompt_version == TEST_PROMPT_VERSION
    assert report.gate_version == AGENT_EVIDENCE_GATE_VERSION


def test_abstained_builder_creates_stable_runner_failure_report() -> None:
    """Runner中止时应生成无结论、带缺失信息的稳定拒答报告。"""

    report = build_agent_abstained_report(
        request_id=TEST_REQUEST_ID,
        request=make_request(),
        planner_prompt_version=TEST_PROMPT_VERSION,
        missing_information=("Agent规划服务响应超时",),
    )

    assert report.status == "abstained"
    assert report.abstained is True
    assert report.evidence == []
    assert report.missing_information == ["Agent规划服务响应超时"]


def test_abstained_builder_rejects_empty_missing_information() -> None:
    """降级报告也必须解释Agent没有完成的内容。"""

    with pytest.raises(
        ValueError,
        match="缺失信息不能为空",
    ):
        build_agent_abstained_report(
            request_id=TEST_REQUEST_ID,
            request=make_request(),
            planner_prompt_version=TEST_PROMPT_VERSION,
            missing_information=(),
        )


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("request_id", "   "),
        ("planner_prompt_version", "   "),
    ],
)
def test_builder_rejects_blank_tracking_fields(
    field_name: str,
    field_value: str,
) -> None:
    """报告构造前必须拒绝空请求ID或空Planner版本。"""

    arguments = {
        "request_id": TEST_REQUEST_ID,
        "request": make_request(),
        "planner_prompt_version": TEST_PROMPT_VERSION,
        "missing_information": ("固定失败原因",),
    }
    arguments[field_name] = field_value

    with pytest.raises(
        ValueError,
        match=field_name,
    ):
        build_agent_abstained_report(
            **arguments,
        )
