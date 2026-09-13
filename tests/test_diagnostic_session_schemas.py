"""诊断会话数据契约的离线单元测试。

这些测试不运行Agent，也不访问文件、网络或模型。
它们只验证持久化会话允许保存的数据形状，
以及摘要、轨迹、证据和状态之间的一致性约束。
"""

from datetime import (
    datetime,
    timezone,
)

import pytest
from pydantic import (
    ValidationError,
)

from app.schemas.agent_api import (
    AgentToolTraceEvent,
)
from app.schemas.diagnostic_session import (
    DiagnosticSessionDiagnosisSnapshot,
    DiagnosticSessionEvidenceCoverage,
    DiagnosticSessionListResponse,
    DiagnosticSessionMetrics,
    DiagnosticSessionRecord,
    DiagnosticSessionRequestSummary,
    DiagnosticSessionSummary,
)
from app.schemas.diagnostics import (
    DiagnosisCause,
    DiagnosisEvidence,
)


# 固定UUID4使测试结果可重复，且满足session_id正则。
TEST_SESSION_ID = (
    "123e4567-e89b-42d3-a456-426614174000"
)

# request_id只负责关联HTTP响应和日志，
# 当前既有契约没有要求它必须是UUID。
TEST_REQUEST_ID = "request-session-001"

# 使用符合既有DocumentId和Chunk ID格式的稳定测试标识。
TEST_DOCUMENT_ID = "a" * 64
TEST_CHUNK_ID = f"{TEST_DOCUMENT_ID}:000001"

# 日志正文不会进入会话；这里只使用固定SHA-256摘要。
TEST_LOG_SHA256 = "b" * 64


def make_request_summary(
    **overrides: object,
) -> DiagnosticSessionRequestSummary:
    """创建不包含完整日志和图片载荷的请求摘要。"""

    values: dict[str, object] = {
        "robot_id": "robot-001",
        "symptom_summary": "网络恢复后机器人仍暂停",
        "task_goal_summary": "核对恢复条件",
        "log_excerpt_sha256": TEST_LOG_SHA256,
        "log_excerpt_char_count": 42,
        "image_count": 1,
    }
    values.update(overrides)
    return DiagnosticSessionRequestSummary(**values)


def make_evidence() -> DiagnosisEvidence:
    """创建一条符合正式诊断引用契约的知识证据。"""

    return DiagnosisEvidence(
        chunk_id=TEST_CHUNK_ID,
        document_id=TEST_DOCUMENT_ID,
        source_file="robot-fault-demo.txt",
        page_or_section="section: ERR-NET-4001",
        rank=1,
        rrf_score=0.0327,
        vector_similarity=0.71,
        excerpt=(
            "通信恢复后不自动继续旧任务，"
            "需要调度系统完成状态核对。"
        ),
    )


def make_completed_diagnosis(
) -> DiagnosticSessionDiagnosisSnapshot:
    """创建有真实引用和可能原因的完成诊断快照。"""

    return DiagnosticSessionDiagnosisSnapshot(
        prompt_version="agent-tool-calling-v2",
        gate_version="agent-confirmed-evidence-v1",
        status="completed",
        evidence=(make_evidence(),),
        possible_causes=(
            DiagnosisCause(
                description=(
                    "系统仍在等待调度状态核对"
                ),
                evidence_chunk_ids=[
                    TEST_CHUNK_ID,
                ],
            ),
        ),
        next_checks=(),
        risk_level="medium",
        missing_information=(),
        abstained=False,
    )


def make_abstained_diagnosis(
) -> DiagnosticSessionDiagnosisSnapshot:
    """创建不输出原因和检查项的合法拒答快照。"""

    return DiagnosticSessionDiagnosisSnapshot(
        prompt_version="agent-tool-calling-v2",
        gate_version="agent-confirmed-evidence-v1",
        status="abstained",
        evidence=(),
        possible_causes=(),
        next_checks=(),
        risk_level="unknown",
        missing_information=(
            "当前请求没有取得足够证据",
        ),
        abstained=True,
    )


def make_success_trace() -> AgentToolTraceEvent:
    """创建一次成功且已经脱敏的知识工具轨迹。"""

    return AgentToolTraceEvent(
        step_id=1,
        tool_name="search_knowledge",
        input_summary=(
            "知识库检索；query_chars=18；top_k=5"
        ),
        result_summary=(
            "知识库返回1条已确认引用"
        ),
        status="success",
        duration_ms=12.5,
        error_code=None,
    )


def make_timeout_trace() -> AgentToolTraceEvent:
    """创建一次使用稳定错误码的超时工具轨迹。"""

    return AgentToolTraceEvent(
        step_id=1,
        tool_name="search_knowledge",
        input_summary="知识库检索；query_chars=18",
        result_summary="知识库工具执行超时",
        status="timeout",
        duration_ms=30_000.0,
        error_code="tool_timeout",
    )


def make_coverage(
    *,
    cited_chunk_ids: tuple[str, ...] = (
        TEST_CHUNK_ID,
    ),
) -> DiagnosticSessionEvidenceCoverage:
    """创建与完成诊断引用一致的证据覆盖快照。"""

    return DiagnosticSessionEvidenceCoverage(
        confirmed_source_ids=(
            "robot-fault-demo.txt",
        ),
        covered_evidence_ids=(
            TEST_CHUNK_ID,
        ),
        cited_chunk_ids=cited_chunk_ids,
    )


def make_metrics(
    **overrides: object,
) -> DiagnosticSessionMetrics:
    """创建与一条成功轨迹对应的运行统计。"""

    values: dict[str, object] = {
        "total_duration_ms": 25.0,
        "tool_step_count": 1,
        "failed_tool_count": 0,
        "failed_tool_names": (),
        "confirmed_source_count": 1,
        "covered_evidence_count": 1,
        "citation_count": 1,
    }
    values.update(overrides)
    return DiagnosticSessionMetrics(**values)


def make_record(
    **overrides: object,
) -> DiagnosticSessionRecord:
    """创建字段互相一致的完成会话。"""

    values: dict[str, object] = {
        "session_id": TEST_SESSION_ID,
        "request_id": TEST_REQUEST_ID,
        "created_at": datetime(
            2026,
            9,
            9,
            8,
            0,
            tzinfo=timezone.utc,
        ),
        "request_summary": make_request_summary(),
        "status": "completed",
        "execution_state": "completed",
        "termination_reason": "planner_finished",
        "finish_reason": "task_completed",
        "diagnosis": make_completed_diagnosis(),
        "tool_events": (make_success_trace(),),
        "vision_observations": (),
        "telemetry_observations": (),
        "test_case_drafts": (),
        "evidence_coverage": make_coverage(),
        "metrics": make_metrics(),
    }
    values.update(overrides)
    return DiagnosticSessionRecord(**values)


def make_summary(
    **overrides: object,
) -> DiagnosticSessionSummary:
    """创建最近会话列表使用的轻量摘要。"""

    values: dict[str, object] = {
        "session_id": TEST_SESSION_ID,
        "request_id": TEST_REQUEST_ID,
        "created_at": datetime(
            2026,
            9,
            9,
            8,
            0,
            tzinfo=timezone.utc,
        ),
        "robot_id": "robot-001",
        "symptom_summary": "网络恢复后机器人仍暂停",
        "status": "completed",
        "termination_reason": "planner_finished",
        "tool_step_count": 1,
        "failed_tool_count": 0,
        "citation_count": 1,
        "total_duration_ms": 25.0,
    }
    values.update(overrides)
    return DiagnosticSessionSummary(**values)


def test_completed_session_record_accepts_consistent_data(
) -> None:
    """完整且互相一致的脱敏会话应通过全部契约。"""

    record = make_record()

    assert record.status == "completed"
    assert record.metrics.tool_step_count == 1
    assert record.metrics.citation_count == 1
    assert record.request_summary.image_count == 1


def test_request_summary_rejects_raw_log_field() -> None:
    """额外的完整日志字段不得混入持久化请求摘要。"""

    with pytest.raises(
        ValidationError,
        match="log_excerpt",
    ):
        DiagnosticSessionRequestSummary(
            **make_request_summary().model_dump(),
            log_excerpt="完整敏感日志不应被保存",
        )


def test_session_id_requires_uuid4() -> None:
    """任意字符串不能冒充服务端生成的会话标识。"""

    with pytest.raises(
        ValidationError,
        match="session_id",
    ):
        make_record(session_id="session-001")


def test_record_rejects_naive_created_at() -> None:
    """没有时区的时间不能用于可靠的最近会话排序。"""

    with pytest.raises(
        ValidationError,
        match="明确时区",
    ):
        make_record(
            created_at=datetime(2026, 9, 9, 8, 0),
        )


def test_abstained_snapshot_requires_matching_boolean() -> None:
    """拒答状态和abstained布尔值不能互相矛盾。"""

    with pytest.raises(
        ValidationError,
        match="status必须与abstained一致",
    ):
        DiagnosticSessionDiagnosisSnapshot(
            prompt_version="agent-tool-calling-v2",
            gate_version="agent-confirmed-evidence-v1",
            status="abstained",
            risk_level="unknown",
            missing_information=("证据不足",),
            abstained=False,
        )


def test_abstained_snapshot_rejects_possible_causes() -> None:
    """拒答记录不能同时向用户输出未经支持的原因。"""

    with pytest.raises(
        ValidationError,
        match="拒答快照不能包含原因",
    ):
        DiagnosticSessionDiagnosisSnapshot(
            prompt_version="agent-tool-calling-v2",
            gate_version="agent-confirmed-evidence-v1",
            status="abstained",
            possible_causes=(
                DiagnosisCause(
                    description="未经证据确认的原因",
                    evidence_chunk_ids=[
                        TEST_CHUNK_ID,
                    ],
                ),
            ),
            risk_level="unknown",
            missing_information=("证据不足",),
            abstained=True,
        )


def test_metrics_rejects_failure_count_above_step_count(
) -> None:
    """失败工具次数不可能大于全部工具执行次数。"""

    with pytest.raises(
        ValidationError,
        match="不能超过",
    ):
        make_metrics(
            tool_step_count=1,
            failed_tool_count=2,
            failed_tool_names=(
                "search_knowledge",
            ),
        )


def test_metrics_require_failed_tool_names() -> None:
    """存在失败步骤时必须说明哪个工具发生了失败。"""

    with pytest.raises(
        ValidationError,
        match="必须记录",
    ):
        make_metrics(
            failed_tool_count=1,
            failed_tool_names=(),
        )


def test_record_requires_step_count_to_match_trace() -> None:
    """会话统计的步骤数必须来自实际脱敏轨迹。"""

    with pytest.raises(
        ValidationError,
        match="tool_step_count",
    ):
        make_record(
            metrics=make_metrics(
                tool_step_count=0,
            ),
        )


def test_record_derives_failed_tools_from_trace() -> None:
    """失败次数和失败工具名必须与非成功轨迹一致。"""

    record = make_record(
        status="failed",
        execution_state="aborted",
        termination_reason="planner_error",
        finish_reason=None,
        diagnosis=make_abstained_diagnosis(),
        tool_events=(make_timeout_trace(),),
        evidence_coverage=(
            DiagnosticSessionEvidenceCoverage()
        ),
        metrics=DiagnosticSessionMetrics(
            total_duration_ms=30_010.0,
            tool_step_count=1,
            failed_tool_count=1,
            failed_tool_names=(
                "search_knowledge",
            ),
            confirmed_source_count=0,
            covered_evidence_count=0,
            citation_count=0,
        ),
    )

    assert record.metrics.failed_tool_count == 1
    assert record.metrics.failed_tool_names == (
        "search_knowledge",
    )


def test_record_rejects_wrong_failed_tool_name() -> None:
    """统计不能把一次知识工具失败登记成另一个工具。"""

    with pytest.raises(
        ValidationError,
        match="failed_tool_names",
    ):
        make_record(
            status="failed",
            execution_state="aborted",
            termination_reason="planner_error",
            finish_reason=None,
            diagnosis=make_abstained_diagnosis(),
            tool_events=(make_timeout_trace(),),
            evidence_coverage=(
                DiagnosticSessionEvidenceCoverage()
            ),
            metrics=DiagnosticSessionMetrics(
                total_duration_ms=30_010.0,
                tool_step_count=1,
                failed_tool_count=1,
                failed_tool_names=(
                    "get_robot_telemetry",
                ),
                confirmed_source_count=0,
                covered_evidence_count=0,
                citation_count=0,
            ),
        )


def test_record_requires_citations_to_match_diagnosis(
) -> None:
    """覆盖摘要不能隐藏或伪造最终诊断引用。"""

    with pytest.raises(
        ValidationError,
        match="cited_chunk_ids",
    ):
        make_record(
            evidence_coverage=make_coverage(
                cited_chunk_ids=(),
            ),
        )


def test_aborted_execution_requires_failed_status() -> None:
    """Runner异常中止不能被会话列表展示成普通拒答。"""

    with pytest.raises(
        ValidationError,
        match="aborted执行必须记录为failed",
    ):
        make_record(
            status="abstained",
            execution_state="aborted",
            termination_reason="planner_error",
            finish_reason=None,
        )


def test_completed_execution_rejects_failed_status() -> None:
    """Runner正常结束的会话不能被错误标记为运行失败。"""

    with pytest.raises(
        ValidationError,
        match="正常结束不能记录为failed",
    ):
        make_record(status="failed")


def test_human_review_finish_requires_human_review_status(
) -> None:
    """需要人工审核的结束原因必须映射到独立业务状态。"""

    with pytest.raises(
        ValidationError,
        match="人工审核结束",
    ):
        make_record(
            status="abstained",
            finish_reason="human_review_required",
            diagnosis=make_abstained_diagnosis(),
            tool_events=(),
            evidence_coverage=(
                DiagnosticSessionEvidenceCoverage()
            ),
            metrics=DiagnosticSessionMetrics(
                total_duration_ms=5.0,
                tool_step_count=0,
                failed_tool_count=0,
                failed_tool_names=(),
                confirmed_source_count=0,
                covered_evidence_count=0,
                citation_count=0,
            ),
        )


def test_recent_session_response_accepts_bounded_page() -> None:
    """最近会话响应应接受不超过limit的轻量摘要。"""

    response = DiagnosticSessionListResponse(
        sessions=(make_summary(),),
        total=5,
        limit=10,
    )

    assert len(response.sessions) == 1
    assert response.total == 5
    assert response.limit == 10


def test_recent_session_response_rejects_total_below_items(
) -> None:
    """列表返回的记录数量不能大于存储层声明的总数。"""

    with pytest.raises(
        ValidationError,
        match="不能超过total",
    ):
        DiagnosticSessionListResponse(
            sessions=(make_summary(),),
            total=0,
            limit=10,
        )
