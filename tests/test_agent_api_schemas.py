"""Robot Diagnostic Agent公开API数据契约的离线测试。

本文件只创建Pydantic模型，不启动FastAPI、不调用LLM、
不执行Agent工具，也不读取真实机器人或知识库数据。

测试覆盖：

1. Agent诊断请求的输入边界；
2. 脱敏工具轨迹的状态与错误码；
3. 执行摘要的步骤连续性和终止关系；
4. 最终响应的request_id、遥测和草案证据白名单。
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
    AgentDiagnosisRequest,
    AgentDiagnosisResponse,
    AgentExecutionSummary,
    AgentToolTraceEvent,
    AgentVisionObservation,
    MAX_AGENT_DIAGNOSIS_IMAGES,
)
from app.schemas.agent_tools import (
    DraftTestCaseEvidence,
    DraftTestCaseStep,
    DraftTestCaseToolOutput,
    RobotTelemetryToolOutput,
)
from app.schemas.diagnostics import (
    DiagnosisCause,
    DiagnosisEvidence,
    DiagnosisReport,
    DiagnosisSymptom,
)
from app.schemas.vision import (
    VisionObservation,
    VisionObservationItem,
)


# 固定请求ID用于验证顶层响应与DiagnosisReport
# 是否指向同一次HTTP请求。
TEST_REQUEST_ID = (
    "request-agent-api-001"
)


# 使用符合SHA-256格式的固定文档ID。
# 测试数据不对应任何真实文档。
TEST_DOCUMENT_ID = "a" * 64


# Chunk ID与固定文档ID关联，
# 供诊断证据和测试草案交叉引用。
TEST_CHUNK_ID = (
    f"{TEST_DOCUMENT_ID}:000001"
)


# 固定图片摘要用于测试视觉观察与图片引用的关系。
# 它只是满足SHA-256格式的脱敏测试值，不对应真实图片。
TEST_IMAGE_SHA256 = "b" * 64


def make_success_trace(
    *,
    step_id: int = 1,
    tool_name: str = "search_knowledge",
) -> AgentToolTraceEvent:
    """创建一条成功、无错误码的公开工具轨迹。"""

    return AgentToolTraceEvent(
        step_id=step_id,
        tool_name=tool_name,
        input_summary=(
            "检索与ERR-NET-4001相关的恢复证据"
        ),
        result_summary=(
            "返回1条经过知识库服务确认的引用"
        ),
        status="success",
        duration_ms=12.5,
    )


def make_completed_execution(
    *,
    steps: list[
        AgentToolTraceEvent
    ] | None = None,
) -> AgentExecutionSummary:
    """创建正常完成的Agent公开执行摘要。"""

    actual_steps = (
        [make_success_trace()]
        if steps is None
        else steps
    )

    return AgentExecutionSummary(
        planner_prompt_version=(
            "agent-tool-calling-v1"
        ),
        state="completed",
        termination_reason=(
            "planner_finished"
        ),
        termination_message=(
            "Agent规划正常结束"
        ),
        finish_reason="task_completed",
        step_count=len(actual_steps),
        steps=actual_steps,
        missing_information=[],
    )


def make_diagnosis_report(
    *,
    request_id: str = TEST_REQUEST_ID,
) -> DiagnosisReport:
    """创建引用一条真实结构测试证据的完整诊断。"""

    return DiagnosisReport(
        request_id=request_id,
        prompt_version=(
            "diagnosis-evidence-v1"
        ),
        gate_version=(
            "hybrid-evidence-gate-v1"
        ),
        status="completed",
        symptoms=[
            DiagnosisSymptom(
                description=(
                    "网络恢复后未继续任务"
                ),
                source="user_report",
            ),
            DiagnosisSymptom(
                description=(
                    "ERR-NET-4001"
                ),
                source="log_excerpt",
            ),
        ],
        evidence=[
            DiagnosisEvidence(
                chunk_id=TEST_CHUNK_ID,
                document_id=(
                    TEST_DOCUMENT_ID
                ),
                source_file=(
                    "robot-fault-demo.txt"
                ),
                page_or_section=(
                    "section: ERR-NET-4001"
                ),
                rank=1,
                rrf_score=0.0327,
                vector_similarity=0.71,
                excerpt=(
                    "网络恢复后不得自动继续旧任务。"
                ),
            )
        ],
        possible_causes=[
            DiagnosisCause(
                description=(
                    "任务仍在等待调度系统状态核对"
                ),
                evidence_chunk_ids=[
                    TEST_CHUNK_ID,
                ],
            )
        ],
        next_checks=[],
        risk_level="medium",
        missing_information=[],
        abstained=False,
    )


def make_telemetry(
) -> RobotTelemetryToolOutput:
    """创建明确标记为内存模拟来源的遥测快照。"""

    return RobotTelemetryToolOutput(
        robot_id="robot-001",
        observed_at=datetime(
            2026,
            8,
            26,
            8,
            0,
            0,
            tzinfo=timezone.utc,
        ),
        location=(
            "warehouse-demo/aisle-07/node-14"
        ),
        battery_percent=42.5,
        operational_state="paused",
        current_task_id=(
            "task-demo-1001"
        ),
        active_fault_codes=(
            "ERR-NET-4001",
        ),
        speed_mps=0.0,
        network_connected=True,
    )


def make_test_case_draft(
    *,
    chunk_id: str = TEST_CHUNK_ID,
) -> DraftTestCaseToolOutput:
    """创建引用指定Chunk ID的只读测试草案。"""

    return DraftTestCaseToolOutput(
        title=(
            "测试用例草案：网络恢复后任务状态核对"
        ),
        objective=(
            "验证网络恢复后旧任务不会自动继续"
        ),
        preconditions=(
            "仅在隔离的教学环境中使用。",
            "执行前由有权限人员审核。",
        ),
        steps=(
            DraftTestCaseStep(
                order=1,
                action=(
                    "核对知识库引用和适用范围。"
                ),
                expected_observation=(
                    "引用可以定位到真实Chunk。"
                ),
            ),
            DraftTestCaseStep(
                order=2,
                action=(
                    "准备隔离的模拟网络中断条件。"
                ),
                expected_observation=(
                    "模拟条件已经获得人工批准。"
                ),
            ),
            DraftTestCaseStep(
                order=3,
                action=(
                    "记录网络恢复后的任务状态。"
                ),
                expected_observation=(
                    "旧任务没有自动继续。"
                ),
            ),
        ),
        evidence=(
            DraftTestCaseEvidence(
                chunk_id=chunk_id,
                document_id=(
                    TEST_DOCUMENT_ID
                ),
                source_file=(
                    "robot-fault-demo.txt"
                ),
                page_or_section=(
                    "section: ERR-NET-4001"
                ),
                chunk_index=1,
                excerpt=(
                    "网络恢复后不得自动继续旧任务。"
                ),
                excerpt_truncated=False,
            ),
        ),
        limitations=(
            "本结果只是测试草案。",
            "投入使用前必须经过人工批准。",
        ),
    )


def make_vision_observation(
) -> VisionObservation:
    """创建一项明确标记为Vision模型来源的测试观察。"""

    return VisionObservation(
        status="completed",
        image_quality="clear",
        observations=(
            VisionObservationItem(
                description="NET指示灯红色常亮",
                category="visible_condition",
                confidence="high",
                region="设备面板右上区域",
            ),
        ),
        visible_indicators=(),
        uncertain_items=(),
        untrusted_text_detected=False,
        untrusted_text_notes=(),
        requires_human_check=False,
        human_check_reasons=(),
        source_image_sha256=(
            TEST_IMAGE_SHA256
        ),
        prompt_version=(
            "robot-vision-observation-v1"
        ),
        model_name="fake-vision-provider",
    )


def make_agent_vision_observation(
    *,
    step_id: int = 1,
    image_ref: str = "image_001",
) -> AgentVisionObservation:
    """把测试视觉输出关联到指定Agent步骤和图片引用。"""

    return AgentVisionObservation(
        step_id=step_id,
        image_ref=image_ref,
        observation=make_vision_observation(),
    )


def test_request_strips_fields_and_accepts_optional_goal(
) -> None:
    """请求契约应清理空白并保留合法可选任务目标。"""

    request = AgentDiagnosisRequest(
        robot_id="  robot-001  ",
        symptom=(
            "  网络恢复后仍未继续任务  "
        ),
        log_excerpt=(
            "  ERR-NET-4001  "
        ),
        task_goal=(
            "  核对恢复条件  "
        ),
    )

    assert request.model_dump() == {
        "robot_id": "robot-001",
        "symptom": (
            "网络恢复后仍未继续任务"
        ),
        "log_excerpt": "ERR-NET-4001",
        "task_goal": "核对恢复条件",
        "images": [],
    }


def test_request_allows_missing_task_goal(
) -> None:
    """没有显式目标时应保存None供Service使用默认目标。"""

    request = AgentDiagnosisRequest(
        robot_id="robot-001",
        symptom="机器人无法继续任务",
        log_excerpt="ERR-NET-4001",
    )

    assert request.task_goal is None
    assert request.images == []


def test_request_accepts_images_without_dumping_base64(
) -> None:
    """多模态请求应保留图片对象但默认序列化必须隐藏Base64。"""

    request = AgentDiagnosisRequest(
        robot_id="robot-001",
        symptom="设备面板指示灯状态异常",
        log_excerpt="NET indicator warning",
        images=[{
            "mime_type": "image/png",
            "encoding": "base64",
            "image_base64": "aGVsbG8=",
            "analysis_goal": (
                "检查设备面板上可见的指示灯状态"
            ),
            "detail": "auto",
        }],
    )

    assert len(request.images) == 1
    assert (
        request.images[0]
        .image_base64
        .get_secret_value()
        == "aGVsbG8="
    )

    dumped_image = (
        request.model_dump()["images"][0]
    )

    assert "image_base64" not in dumped_image
    assert dumped_image["mime_type"] == (
        "image/png"
    )
    assert dumped_image["analysis_goal"] == (
        "检查设备面板上可见的指示灯状态"
    )


def test_request_rejects_too_many_images(
) -> None:
    """请求图片数超过服务上限时应在进入Service前失败。"""

    image_payload = {
        "mime_type": "image/png",
        "encoding": "base64",
        "image_base64": "aGVsbG8=",
        "analysis_goal": "检查指示灯",
        "detail": "auto",
    }

    with pytest.raises(
        ValidationError,
        match="images",
    ):
        AgentDiagnosisRequest(
            robot_id="robot-001",
            symptom="设备面板状态异常",
            log_excerpt="panel warning",
            images=[
                image_payload
                for _ in range(
                    MAX_AGENT_DIAGNOSIS_IMAGES
                    + 1
                )
            ],
        )


def test_request_rejects_client_controlled_agent_boundaries(
) -> None:
    """客户端不能通过额外字段修改步数或工具白名单。"""

    with pytest.raises(
        ValidationError,
        match="extra_forbidden",
    ):
        AgentDiagnosisRequest.model_validate({
            "robot_id": "robot-001",
            "symptom": "机器人无法继续任务",
            "log_excerpt": "ERR-NET-4001",
            "max_steps": 100,
            "tools": [
                "run_shell",
            ],
        })


def test_request_rejects_blank_explicit_task_goal(
) -> None:
    """显式提供task_goal时不能使用纯空白文本。"""

    with pytest.raises(
        ValidationError,
        match="task_goal",
    ):
        AgentDiagnosisRequest(
            robot_id="robot-001",
            symptom="机器人无法继续任务",
            log_excerpt="ERR-NET-4001",
            task_goal="   ",
        )


def test_success_trace_has_no_error_code(
) -> None:
    """成功工具事件应包含摘要、耗时且没有错误码。"""

    trace = make_success_trace()

    assert trace.step_id == 1
    assert trace.event_type == (
        "tool_execution"
    )
    assert trace.status == "success"
    assert trace.error_code is None
    assert trace.duration_ms == 12.5


def test_timeout_trace_accepts_matching_error_code(
) -> None:
    """通用工具超时应接受tool_timeout错误码。"""

    trace = AgentToolTraceEvent(
        step_id=1,
        tool_name="search_knowledge",
        input_summary="检索故障恢复条件",
        result_summary="知识库工具执行超时",
        status="timeout",
        duration_ms=120_000.0,
        error_code="tool_timeout",
    )

    assert trace.status == "timeout"
    assert trace.error_code == (
        "tool_timeout"
    )


@pytest.mark.parametrize(
    (
        "status",
        "error_code",
    ),
    [
        (
            "timeout",
            "vision_timeout",
        ),
        (
            "error",
            "vision_upstream_error",
        ),
        (
            "error",
            "invalid_vision_response",
        ),
    ],
)
def test_trace_accepts_specific_vision_error_code(
    status: str,
    error_code: str,
) -> None:
    """公开轨迹应保留Vision最终失败的稳定分类。"""

    trace = AgentToolTraceEvent.model_validate({
        "step_id": 1,
        "tool_name": "analyze_robot_image",
        "input_summary": (
            "图片分析；image_ref=image_primary"
        ),
        "result_summary": (
            f"status={status}；"
            f"error_code={error_code}"
        ),
        "status": status,
        "duration_ms": 1.0,
        "error_code": error_code,
    })

    assert trace.status == status
    assert trace.error_code == error_code


@pytest.mark.parametrize(
    ("status", "error_code"),
    [
        (
            "success",
            "tool_timeout",
        ),
        (
            "timeout",
            None,
        ),
        (
            "timeout",
            "empty_result",
        ),
    ],
)
def test_trace_rejects_mismatched_status_and_error_code(
    status: str,
    error_code: str | None,
) -> None:
    """公开轨迹不能组合互相矛盾的状态和错误码。"""

    with pytest.raises(
        ValidationError,
    ):
        AgentToolTraceEvent.model_validate({
            "step_id": 1,
            "tool_name": "search_knowledge",
            "input_summary": "检索故障证据",
            "result_summary": "固定测试结果",
            "status": status,
            "duration_ms": 10.0,
            "error_code": error_code,
        })


def test_completed_execution_accepts_continuous_steps(
) -> None:
    """正常结束摘要应接受从1开始连续的工具步骤。"""

    steps = [
        make_success_trace(
            step_id=1,
            tool_name="search_knowledge",
        ),
        make_success_trace(
            step_id=2,
            tool_name=(
                "get_robot_telemetry"
            ),
        ),
    ]

    summary = make_completed_execution(
        steps=steps
    )

    assert summary.state == "completed"
    assert summary.step_count == 2
    assert [
        step.step_id
        for step in summary.steps
    ] == [
        1,
        2,
    ]


def test_aborted_execution_requires_missing_information(
) -> None:
    """安全中止摘要必须说明当前没有完成的内容。"""

    summary = AgentExecutionSummary(
        planner_prompt_version=(
            "agent-tool-calling-v1"
        ),
        state="aborted",
        termination_reason=(
            "planner_timeout"
        ),
        termination_message=(
            "Agent规划服务响应超时"
        ),
        finish_reason=None,
        step_count=0,
        steps=[],
        missing_information=[
            "规划服务没有返回下一步决定"
        ],
    )

    assert summary.state == "aborted"
    assert summary.finish_reason is None
    assert summary.missing_information == [
        "规划服务没有返回下一步决定"
    ]


def test_execution_rejects_step_count_mismatch(
) -> None:
    """声明的step_count必须等于实际轨迹数量。"""

    with pytest.raises(
        ValidationError,
        match="step_count",
    ):
        AgentExecutionSummary(
            planner_prompt_version=(
                "agent-tool-calling-v1"
            ),
            state="completed",
            termination_reason=(
                "planner_finished"
            ),
            termination_message=(
                "Agent规划正常结束"
            ),
            finish_reason="task_completed",
            step_count=2,
            steps=[
                make_success_trace(),
            ],
            missing_information=[],
        )


def test_execution_rejects_non_continuous_step_ids(
) -> None:
    """轨迹不能从2开始或跳过中间步骤。"""

    with pytest.raises(
        ValidationError,
        match="连续递增",
    ):
        make_completed_execution(
            steps=[
                make_success_trace(
                    step_id=1
                ),
                make_success_trace(
                    step_id=3,
                    tool_name=(
                        "get_robot_telemetry"
                    ),
                ),
            ]
        )


def test_non_completed_finish_requires_missing_information(
) -> None:
    """Planner声明信息不足时公开摘要必须写明缺口。"""

    with pytest.raises(
        ValidationError,
        match="missing_information",
    ):
        AgentExecutionSummary(
            planner_prompt_version=(
                "agent-tool-calling-v1"
            ),
            state="completed",
            termination_reason=(
                "planner_finished"
            ),
            termination_message=(
                "Agent规划正常结束"
            ),
            finish_reason=(
                "insufficient_information"
            ),
            step_count=0,
            steps=[],
            missing_information=[],
        )


def test_valid_agent_response_combines_all_public_outputs(
) -> None:
    """最终响应应组合诊断、执行摘要、模拟遥测和草案。"""

    response = AgentDiagnosisResponse(
        request_id=TEST_REQUEST_ID,
        diagnosis=(
            make_diagnosis_report()
        ),
        execution=(
            make_completed_execution()
        ),
        telemetry_observations=[
            make_telemetry(),
        ],
        test_case_drafts=[
            make_test_case_draft(),
        ],
    )

    assert response.request_id == (
        TEST_REQUEST_ID
    )
    assert response.diagnosis.status == (
        "completed"
    )
    assert response.execution.state == (
        "completed"
    )
    assert response.vision_observations == []
    assert response.telemetry_observations[
        0
    ].source == "simulated_memory"
    assert response.test_case_drafts[
        0
    ].draft_only is True


def test_response_accepts_visual_observation_for_successful_step(
) -> None:
    """视觉观察必须绑定到一次真实成功的视觉工具轨迹。"""

    vision_step = make_success_trace(
        step_id=1,
        tool_name="analyze_robot_image",
    )

    response = AgentDiagnosisResponse(
        request_id=TEST_REQUEST_ID,
        diagnosis=make_diagnosis_report(),
        execution=make_completed_execution(
            steps=[vision_step]
        ),
        vision_observations=[
            make_agent_vision_observation()
        ],
    )

    public_observation = (
        response.vision_observations[0]
    )

    assert public_observation.image_ref == (
        "image_001"
    )
    assert public_observation.observation.source == (
        "vision_model"
    )
    assert (
        public_observation
        .observation
        .source_image_sha256
        == TEST_IMAGE_SHA256
    )


def test_response_rejects_visual_observation_without_tool_step(
) -> None:
    """响应不能凭空加入没有执行轨迹支持的视觉观察。"""

    with pytest.raises(
        ValidationError,
        match="一一对应",
    ):
        AgentDiagnosisResponse(
            request_id=TEST_REQUEST_ID,
            diagnosis=make_diagnosis_report(),
            execution=make_completed_execution(
                steps=[]
            ),
            vision_observations=[
                make_agent_vision_observation()
            ],
        )


def test_response_rejects_missing_visual_observation_for_tool_step(
) -> None:
    """成功视觉步骤的结构化观察不能从公开响应中消失。"""

    with pytest.raises(
        ValidationError,
        match="一一对应",
    ):
        AgentDiagnosisResponse(
            request_id=TEST_REQUEST_ID,
            diagnosis=make_diagnosis_report(),
            execution=make_completed_execution(
                steps=[
                    make_success_trace(
                        tool_name=(
                            "analyze_robot_image"
                        )
                    )
                ]
            ),
            vision_observations=[],
        )


def test_response_requires_matching_request_ids(
) -> None:
    """顶层响应与诊断报告必须属于同一次HTTP请求。"""

    with pytest.raises(
        ValidationError,
        match="request_id",
    ):
        AgentDiagnosisResponse(
            request_id=TEST_REQUEST_ID,
            diagnosis=make_diagnosis_report(
                request_id=(
                    "different-request-id"
                )
            ),
            execution=(
                make_completed_execution()
            ),
        )


def test_response_rejects_duplicate_telemetry_robot_ids(
) -> None:
    """同一机器人快照不能在响应中重复出现。"""

    telemetry = make_telemetry()

    with pytest.raises(
        ValidationError,
        match="重复robot_id",
    ):
        AgentDiagnosisResponse(
            request_id=TEST_REQUEST_ID,
            diagnosis=(
                make_diagnosis_report()
            ),
            execution=(
                make_completed_execution()
            ),
            telemetry_observations=[
                telemetry,
                telemetry,
            ],
        )


def test_response_rejects_draft_evidence_outside_diagnosis(
) -> None:
    """测试草案不能引用最终诊断未公开的Chunk。"""

    unknown_chunk_id = (
        f"{'b' * 64}:000999"
    )

    with pytest.raises(
        ValidationError,
        match="不存在的Chunk ID",
    ):
        AgentDiagnosisResponse(
            request_id=TEST_REQUEST_ID,
            diagnosis=(
                make_diagnosis_report()
            ),
            execution=(
                make_completed_execution()
            ),
            test_case_drafts=[
                make_test_case_draft(
                    chunk_id=(
                        unknown_chunk_id
                    )
                )
            ],
        )
