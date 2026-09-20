"""诊断会话构造器的确定性单元测试。

测试使用固定UUID、固定UTC时间和本地Pydantic对象，
不调用Planner、Vision、知识库、网络或文件存储。
"""

from datetime import (
    datetime,
    timezone,
)
from hashlib import (
    sha256,
)
from uuid import (
    UUID,
)

import pytest
from pydantic import (
    ValidationError,
)

from app.schemas.agent_api import (
    AgentDiagnosisRequest,
    AgentDiagnosisResponse,
    AgentExecutionSummary,
    AgentToolTraceEvent,
)
from app.schemas.agent_progress import (
    AgentProgress,
    AgentToolProgressRecord,
)
from app.schemas.diagnostics import (
    DiagnosisCause,
    DiagnosisEvidence,
    DiagnosisReport,
    DiagnosisSymptom,
)
from app.services.diagnostic_session_builder import (
    DiagnosticSessionBuilder,
    build_diagnostic_session_summary,
)


# 构造器通过依赖注入使用这些固定值，
# 使每次测试都生成完全相同的会话追踪字段。
TEST_SESSION_UUID = UUID(
    "123e4567-e89b-42d3-a456-426614174000"
)
TEST_CREATED_AT = datetime(
    2026,
    9,
    9,
    9,
    30,
    tzinfo=timezone.utc,
)
TEST_REQUEST_ID = "request-builder-001"

# 以下标识满足现有知识证据和Progress契约。
TEST_DOCUMENT_ID = "a" * 64
TEST_CHUNK_ID = f"{TEST_DOCUMENT_ID}:000001"
TEST_CALL_SIGNATURE = "c" * 64

# 该字符串只存在于请求日志，测试会验证它不会进入会话JSON。
TEST_PRIVATE_LOG_TEXT = (
    "PRIVATE-LOG-CONTENT ERR-NET-4001 token=do-not-store"
)


def make_builder() -> DiagnosticSessionBuilder:
    """创建使用固定ID和固定时钟的会话构造器。"""

    return DiagnosticSessionBuilder(
        session_id_factory=(
            lambda: TEST_SESSION_UUID
        ),
        clock=lambda: TEST_CREATED_AT,
    )


def make_request(
    *,
    symptom: str = "网络恢复后机器人仍暂停",
    task_goal: str | None = "核对安全恢复条件",
) -> AgentDiagnosisRequest:
    """创建不带图片的合法Agent诊断请求。"""

    return AgentDiagnosisRequest(
        robot_id="robot-001",
        symptom=symptom,
        log_excerpt=TEST_PRIVATE_LOG_TEXT,
        task_goal=task_goal,
    )


def make_evidence() -> DiagnosisEvidence:
    """创建一条经过正式DiagnosisEvidence校验的引用。"""

    return DiagnosisEvidence(
        chunk_id=TEST_CHUNK_ID,
        document_id=TEST_DOCUMENT_ID,
        source_file="robot-fault-demo.txt",
        page_or_section="section: ERR-NET-4001",
        rank=1,
        rrf_score=0.0327,
        vector_similarity=0.71,
        excerpt=(
            "通信恢复后不能自动继续旧任务，"
            "需要调度系统完成状态核对。"
        ),
    )


def make_diagnosis_report(
    *,
    status: str = "completed",
) -> DiagnosisReport:
    """创建完成或拒答两种合法诊断报告。"""

    symptoms = [
        DiagnosisSymptom(
            description="网络恢复后机器人仍暂停",
            source="user_report",
        ),
        DiagnosisSymptom(
            # 故意把完整测试日志放进原始响应，
            # 用于验证Builder不会把symptoms持久化。
            description=TEST_PRIVATE_LOG_TEXT,
            source="log_excerpt",
        ),
    ]

    if status == "abstained":
        return DiagnosisReport(
            request_id=TEST_REQUEST_ID,
            prompt_version="agent-tool-calling-v2",
            gate_version="agent-confirmed-evidence-v1",
            status="abstained",
            symptoms=symptoms,
            evidence=[],
            possible_causes=[],
            next_checks=[],
            risk_level="unknown",
            missing_information=[
                "当前请求没有取得足够证据",
            ],
            abstained=True,
        )

    if status == "human_review_required":
        return DiagnosisReport(
            request_id=TEST_REQUEST_ID,
            prompt_version="agent-tool-calling-v8",
            gate_version="agent-confirmed-evidence-v1",
            status="human_review_required",
            symptoms=symptoms,
            evidence=[],
            possible_causes=[],
            next_checks=[],
            risk_level="unknown",
            missing_information=[
                "请求需要具备资质的人员审核",
            ],
            abstained=False,
        )

    return DiagnosisReport(
        request_id=TEST_REQUEST_ID,
        prompt_version="agent-tool-calling-v2",
        gate_version="agent-confirmed-evidence-v1",
        status="completed",
        symptoms=symptoms,
        evidence=[make_evidence()],
        possible_causes=[
            DiagnosisCause(
                description=(
                    "系统仍在等待调度状态核对"
                ),
                evidence_chunk_ids=[
                    TEST_CHUNK_ID,
                ],
            ),
        ],
        next_checks=[],
        risk_level="medium",
        missing_information=[],
        abstained=False,
    )


def make_trace(
    *,
    status: str = "success",
) -> AgentToolTraceEvent:
    """创建成功、空结果或超时的公开工具轨迹。"""

    if status == "success":
        return AgentToolTraceEvent(
            step_id=1,
            tool_name="search_knowledge",
            input_summary="知识库检索；query_chars=18",
            result_summary="返回1条已确认引用",
            status="success",
            duration_ms=12.5,
            error_code=None,
        )

    if status == "empty":
        return AgentToolTraceEvent(
            step_id=1,
            tool_name="search_knowledge",
            input_summary="知识库检索；query_chars=18",
            result_summary="工具没有返回可用结果",
            status="empty",
            duration_ms=10.0,
            error_code="empty_result",
        )

    return AgentToolTraceEvent(
        step_id=1,
        tool_name="search_knowledge",
        input_summary="知识库检索；query_chars=18",
        result_summary="知识库工具执行超时",
        status="timeout",
        duration_ms=30_000.0,
        error_code="tool_timeout",
    )


def make_progress(
    *,
    outcome: str = "completed",
) -> AgentProgress:
    """创建与三种Runner结果匹配的最终进度快照。"""

    if outcome == "human_review_required":
        return AgentProgress(
            state="human_review_required",
            required_capabilities=(
                "knowledge_evidence",
            ),
            completed_capabilities=(),
            tool_records=(),
            confirmed_source_ids=(),
            covered_evidence_ids=(),
            evidence_conflicts=(),
            missing_information=(
                "请求需要具备资质的人员审核",
            ),
            allowed_next_tool_names=(),
            stop_reason=(
                "human_review_required"
            ),
        )

    trace_status = (
        "success"
        if outcome == "completed"
        else (
            "empty"
            if outcome == "abstained"
            else "timeout"
        )
    )

    progress_record = AgentToolProgressRecord(
        step_number=1,
        call_id="call_session_001",
        tool_name="search_knowledge",
        call_signature=TEST_CALL_SIGNATURE,
        status=trace_status,
        produced_new_information=(
            outcome == "completed"
        ),
        new_source_ids=(
            ("robot-fault-demo.txt",)
            if outcome == "completed"
            else ()
        ),
        new_evidence_ids=(
            (TEST_CHUNK_ID,)
            if outcome == "completed"
            else ()
        ),
        error_code=(
            None
            if outcome == "completed"
            else (
                "empty_result"
                if outcome == "abstained"
                else "tool_timeout"
            )
        ),
    )

    if outcome == "completed":
        return AgentProgress(
            state="completed",
            required_capabilities=(
                "knowledge_evidence",
            ),
            completed_capabilities=(
                "knowledge_evidence",
            ),
            tool_records=(progress_record,),
            confirmed_source_ids=(
                "robot-fault-demo.txt",
            ),
            covered_evidence_ids=(
                TEST_CHUNK_ID,
            ),
            evidence_conflicts=(),
            missing_information=(),
            allowed_next_tool_names=(),
            stop_reason="sufficient_evidence",
        )

    if outcome == "abstained":
        return AgentProgress(
            state="abstained",
            required_capabilities=(
                "knowledge_evidence",
            ),
            completed_capabilities=(),
            tool_records=(progress_record,),
            confirmed_source_ids=(),
            covered_evidence_ids=(),
            evidence_conflicts=(),
            missing_information=(
                "知识工具没有返回可用证据",
            ),
            allowed_next_tool_names=(),
            stop_reason=(
                "insufficient_information"
            ),
        )

    return AgentProgress(
        state="failed",
        required_capabilities=(
            "knowledge_evidence",
        ),
        completed_capabilities=(),
        tool_records=(progress_record,),
        confirmed_source_ids=(),
        covered_evidence_ids=(),
        evidence_conflicts=(),
        missing_information=(
            "知识工具执行超时",
        ),
        allowed_next_tool_names=(),
        stop_reason="runtime_failure",
    )


def make_response(
    *,
    outcome: str = "completed",
) -> AgentDiagnosisResponse:
    """创建完成、空结果、超时或人工审核响应。"""

    if outcome == "human_review_required":
        execution = AgentExecutionSummary(
            planner_prompt_version=(
                "agent-tool-calling-v2"
            ),
            state="completed",
            termination_reason=(
                "request_policy_finished"
            ),
            termination_message=(
                "请求级策略要求人工审核"
            ),
            finish_reason=(
                "human_review_required"
            ),
            step_count=0,
            steps=[],
            progress=make_progress(
                outcome=(
                    "human_review_required"
                )
            ),
            missing_information=[
                "请求需要具备资质的人员审核",
            ],
        )

        return AgentDiagnosisResponse(
            request_id=TEST_REQUEST_ID,
            diagnosis=make_diagnosis_report(
                status="human_review_required"
            ),
            execution=execution,
        )

    if outcome == "failed":
        execution = AgentExecutionSummary(
            planner_prompt_version=(
                "agent-tool-calling-v2"
            ),
            state="aborted",
            termination_reason="planner_timeout",
            termination_message=(
                "Agent规划服务响应超时"
            ),
            finish_reason=None,
            step_count=1,
            steps=[make_trace(status="timeout")],
            progress=make_progress(
                outcome="failed"
            ),
            missing_information=[
                "知识工具执行超时",
            ],
        )

        return AgentDiagnosisResponse(
            request_id=TEST_REQUEST_ID,
            diagnosis=make_diagnosis_report(
                status="abstained"
            ),
            execution=execution,
        )

    if outcome == "abstained":
        execution = AgentExecutionSummary(
            planner_prompt_version=(
                "agent-tool-calling-v2"
            ),
            state="completed",
            termination_reason="planner_finished",
            termination_message=(
                "Agent规划正常结束"
            ),
            finish_reason=(
                "insufficient_information"
            ),
            step_count=1,
            steps=[make_trace(status="empty")],
            progress=make_progress(
                outcome="abstained"
            ),
            missing_information=[
                "知识工具没有返回可用证据",
            ],
        )

        return AgentDiagnosisResponse(
            request_id=TEST_REQUEST_ID,
            diagnosis=make_diagnosis_report(
                status="abstained"
            ),
            execution=execution,
        )

    execution = AgentExecutionSummary(
        planner_prompt_version=(
            "agent-tool-calling-v2"
        ),
        state="completed",
        termination_reason="planner_finished",
        termination_message="Agent规划正常结束",
        finish_reason="task_completed",
        step_count=1,
        steps=[make_trace(status="success")],
        progress=make_progress(
            outcome="completed"
        ),
        missing_information=[],
    )

    return AgentDiagnosisResponse(
        request_id=TEST_REQUEST_ID,
        diagnosis=make_diagnosis_report(),
        execution=execution,
    )


def test_builder_creates_completed_session_from_public_response(
) -> None:
    """完成响应应转换成带证据统计的完成会话。"""

    record = make_builder().build(
        request=make_request(),
        response=make_response(),
        total_duration_ms=45.5,
    )

    assert record.session_id == str(
        TEST_SESSION_UUID
    )
    assert record.created_at == TEST_CREATED_AT
    assert record.status == "completed"
    assert record.metrics.tool_step_count == 1
    assert record.metrics.failed_tool_count == 0
    assert record.metrics.citation_count == 1
    # evidence_coverage是Pydantic模型，
    # model_dump()将它转换成普通字典后再比较字段值。
    assert record.evidence_coverage.model_dump() == {
        "confirmed_source_ids": (
            "robot-fault-demo.txt",
        ),
        "covered_evidence_ids": (
            TEST_CHUNK_ID,
        ),
        "cited_chunk_ids": (
            TEST_CHUNK_ID,
        ),
    }


def test_builder_does_not_persist_complete_log_or_symptoms(
) -> None:
    """请求日志和DiagnosisReport症状不得进入会话JSON。"""

    record = make_builder().build(
        request=make_request(),
        response=make_response(),
        total_duration_ms=45.5,
    )

    serialized = record.model_dump_json()

    assert TEST_PRIVATE_LOG_TEXT not in serialized
    assert "symptoms" not in serialized
    assert record.request_summary.log_excerpt_sha256 == (
        sha256(
            TEST_PRIVATE_LOG_TEXT.encode("utf-8")
        ).hexdigest()
    )


def test_builder_compacts_and_redacts_summary_text() -> None:
    """摘要应压缩空白并遮盖常见密钥和Bearer令牌。"""

    request = make_request(
        symptom=(
            "面板异常\n\tapi_key=private-value"
        ),
        task_goal=(
            "使用 Bearer abc.def-123 核对状态"
        ),
    )

    record = make_builder().build(
        request=request,
        response=make_response(),
        total_duration_ms=45.5,
    )

    assert record.request_summary.symptom_summary == (
        "面板异常 api_key=[REDACTED]"
    )
    assert record.request_summary.task_goal_summary == (
        "使用 Bearer [REDACTED] 核对状态"
    )


def test_builder_truncates_long_symptom_summary() -> None:
    """长现象只保存500字符摘要而不是完整输入。"""

    record = make_builder().build(
        request=make_request(
            symptom="故" * 600,
        ),
        response=make_response(),
        total_duration_ms=45.5,
    )

    summary = record.request_summary.symptom_summary

    assert len(summary) == 500
    assert summary.endswith("…")


def test_builder_maps_empty_result_to_normal_abstention(
) -> None:
    """工具空结果是正常信息不足，不应计为工具失败。"""

    record = make_builder().build(
        request=make_request(),
        response=make_response(
            outcome="abstained"
        ),
        total_duration_ms=20.0,
    )

    assert record.status == "abstained"
    assert record.execution_state == "completed"
    assert record.metrics.tool_step_count == 1
    assert record.metrics.failed_tool_count == 0
    assert record.metrics.failed_tool_names == ()


def test_builder_maps_aborted_run_to_failed_session(
) -> None:
    """Runner异常中止必须转换为failed会话。"""

    record = make_builder().build(
        request=make_request(),
        response=make_response(outcome="failed"),
        total_duration_ms=30_020.0,
    )

    assert record.status == "failed"
    assert record.execution_state == "aborted"
    assert record.metrics.failed_tool_count == 1
    assert record.metrics.failed_tool_names == (
        "search_knowledge",
    )


def test_builder_preserves_human_review_as_distinct_status(
) -> None:
    """人工审核不能被压缩成普通abstained状态。"""

    record = make_builder().build(
        request=make_request(),
        response=make_response(
            outcome="human_review_required"
        ),
        total_duration_ms=5.0,
    )

    assert record.status == (
        "human_review_required"
    )
    assert record.execution_state == "completed"
    assert record.finish_reason == (
        "human_review_required"
    )
    assert record.diagnosis.status == (
        "human_review_required"
    )
    assert record.diagnosis.abstained is False
    assert record.diagnosis.evidence == ()


def test_builder_rejects_negative_total_duration() -> None:
    """总耗时不可能为负数。"""

    with pytest.raises(
        ValidationError,
        match="total_duration_ms",
    ):
        make_builder().build(
            request=make_request(),
            response=make_response(),
            total_duration_ms=-1.0,
        )


def test_builder_rejects_boolean_duration() -> None:
    """bool虽是int子类，但不能冒充毫秒数。"""

    with pytest.raises(
        TypeError,
        match="total_duration_ms必须是数值",
    ):
        make_builder().build(
            request=make_request(),
            response=make_response(),
            total_duration_ms=True,
        )


def test_builder_rejects_invalid_session_id_factory_output(
) -> None:
    """ID生成器返回非UUID4结果时应由最终Schema拒绝。"""

    builder = DiagnosticSessionBuilder(
        session_id_factory=(
            lambda: "not-a-uuid"  # type: ignore[return-value]
        ),
        clock=lambda: TEST_CREATED_AT,
    )

    with pytest.raises(
        ValidationError,
        match="session_id",
    ):
        builder.build(
            request=make_request(),
            response=make_response(),
            total_duration_ms=10.0,
        )


def test_builder_rejects_clock_without_timezone() -> None:
    """注入时钟仍必须返回带时区时间。"""

    builder = DiagnosticSessionBuilder(
        session_id_factory=(
            lambda: TEST_SESSION_UUID
        ),
        clock=lambda: datetime(2026, 9, 9, 9, 30),
    )

    with pytest.raises(
        ValidationError,
        match="明确时区",
    ):
        builder.build(
            request=make_request(),
            response=make_response(),
            total_duration_ms=10.0,
        )


def test_builder_requires_agent_request_model() -> None:
    """普通字典不能绕过AgentDiagnosisRequest契约。"""

    with pytest.raises(
        TypeError,
        match="AgentDiagnosisRequest",
    ):
        make_builder().build(
            request={},  # type: ignore[arg-type]
            response=make_response(),
            total_duration_ms=10.0,
        )


def test_builder_requires_agent_response_model() -> None:
    """未经公开响应契约验证的字典不能被持久化。"""

    with pytest.raises(
        TypeError,
        match="AgentDiagnosisResponse",
    ):
        make_builder().build(
            request=make_request(),
            response={},  # type: ignore[arg-type]
            total_duration_ms=10.0,
        )


def test_summary_builder_returns_lightweight_fields() -> None:
    """列表摘要应保留索引字段但不复制详细轨迹。"""

    record = make_builder().build(
        request=make_request(),
        response=make_response(),
        total_duration_ms=45.5,
    )

    summary = build_diagnostic_session_summary(
        record
    )

    assert summary.session_id == record.session_id
    assert summary.robot_id == "robot-001"
    assert summary.status == "completed"
    assert summary.tool_step_count == 1
    assert summary.citation_count == 1
    assert "tool_events" not in summary.model_dump()
    assert "evidence" not in summary.model_dump()


def test_summary_builder_rejects_wrong_input_type() -> None:
    """摘要函数只接受已经校验的完整会话记录。"""

    with pytest.raises(
        TypeError,
        match="DiagnosticSessionRecord",
    ):
        build_diagnostic_session_summary(  # type: ignore[arg-type]
            {}
        )
