"""AgentProgressReducer的确定性状态转换测试。

本文件验证：

1. 只有继续进入Planner的工具策略才能创建初始进度；
2. 当前进度会在工具执行前阻止范围外和重复调用；
3. 五种只读工具的成功输出会生成正确来源与证据标识；
4. 空结果会留下失败记录，但不会伪造完成能力；
5. 多能力任务会按工具结果逐步收窄下一步工具；
6. 全部能力完成后进入ready_to_finish；
7. Reducer返回新对象，不修改旧进度。

测试不调用真实LLM、网络、Chroma、Vision Provider或机器人。
"""

from datetime import (
    datetime,
    timezone,
)

import pytest

from pydantic import (
    ValidationError,
)

from app.agent.progress_reducer import (
    AgentProgressReducer,
    AgentProgressTransitionError,
)
from app.schemas.agent import (
    ToolCall,
    ToolExecutionResult,
)
from app.schemas.agent_planning import (
    AgentToolInteraction,
)
from app.schemas.agent_progress import (
    AgentEvidenceConflict,
    AgentProgress,
)
from app.schemas.agent_tool_policy import (
    AgentToolPolicyDecision,
)
from app.schemas.agent_tools import (
    DraftTestCaseEvidence,
    DraftTestCaseStep,
    DraftTestCaseToolOutput,
    GetCurrentTimeToolOutput,
    RobotTelemetryToolOutput,
    SearchKnowledgeToolOutput,
)
from app.schemas.knowledge_query import (
    KnowledgeCitation,
)
from app.schemas.vision import (
    VisionObservation,
    VisionObservationItem,
)


# 固定测试文档、Chunk和图片摘要。
# 所有值均为教学占位数据，不对应真实设备或生产文档。
TEST_DOCUMENT_ID = "d" * 64
TEST_CHUNK_ID = (
    TEST_DOCUMENT_ID + ":000001"
)
TEST_IMAGE_SHA256 = "e" * 64


# 使用固定UTC时间验证遥测和系统时钟来源标识。
TEST_OBSERVED_AT = datetime(
    2026,
    9,
    7,
    8,
    30,
    tzinfo=timezone.utc,
)


def make_knowledge_policy(
) -> AgentToolPolicyDecision:
    """创建只允许知识库检索的请求级策略。"""

    return AgentToolPolicyDecision(
        disposition="continue_to_planner",
        required_capabilities=(
            "knowledge_evidence",
        ),
        allowed_tool_names=(
            "search_knowledge",
        ),
        reason_codes=(
            "knowledge_evidence_required",
        ),
    )


def make_call(
    *,
    call_id: str,
    tool_name: str,
    arguments: dict[str, object],
) -> ToolCall:
    """创建经过通用ToolCall契约校验的测试工具请求。"""

    return ToolCall(
        call_id=call_id,
        tool_name=tool_name,
        arguments=arguments,
    )


def make_success_interaction(
    *,
    step_number: int,
    tool_call: ToolCall,
    output: object,
) -> AgentToolInteraction:
    """把一个Pydantic工具输出包装成成功交互。"""

    # 所有真实工具输出都继承Pydantic BaseModel并提供
    # model_dump()。测试保留mode="python"，使datetime
    # 继续保持datetime对象，而不是提前变成字符串。
    output_data = output.model_dump(
        mode="python"
    )

    return AgentToolInteraction(
        step_number=step_number,
        tool_call=tool_call,
        result=ToolExecutionResult(
            call_id=tool_call.call_id,
            tool_name=tool_call.tool_name,
            status="success",
            output=output_data,
            duration_ms=12.5,
        ),
    )


def make_empty_interaction(
    *,
    step_number: int,
    tool_call: ToolCall,
) -> AgentToolInteraction:
    """创建没有找到可用内容的合法empty工具交互。"""

    return AgentToolInteraction(
        step_number=step_number,
        tool_call=tool_call,
        result=ToolExecutionResult(
            call_id=tool_call.call_id,
            tool_name=tool_call.tool_name,
            status="empty",
            error_code="empty_result",
            public_message=(
                "工具没有返回可用结果"
            ),
            duration_ms=5.0,
        ),
    )


def make_knowledge_output(
) -> SearchKnowledgeToolOutput:
    """创建一条经过检索分数和引用字段校验的知识输出。"""

    return SearchKnowledgeToolOutput(
        citations=[
            KnowledgeCitation(
                chunk_id=TEST_CHUNK_ID,
                document_id=TEST_DOCUMENT_ID,
                source_file="robot-demo.txt",
                page_or_section=(
                    "section: ERR-DEMO-1001"
                ),
                chunk_index=1,
                rank=1,
                similarity=0.82,
                rrf_score=0.032,
                excerpt=(
                    "模拟证据要求人工确认后恢复任务。"
                ),
            ),
        ],
        retrieval_ms=8.5,
    )


def make_telemetry_output(
) -> RobotTelemetryToolOutput:
    """创建一份确定性的脱敏模拟遥测快照。"""

    return RobotTelemetryToolOutput(
        robot_id="robot-001",
        observed_at=TEST_OBSERVED_AT,
        location=(
            "warehouse-demo/aisle-01/node-01"
        ),
        battery_percent=42.5,
        operational_state="paused",
        current_task_id="task-demo-001",
        active_fault_codes=(
            "ERR-NET-4001",
        ),
        speed_mps=0.0,
        network_connected=True,
    )


def make_vision_output(
) -> VisionObservation:
    """创建一份只包含可见现象的确定性视觉观察。"""

    return VisionObservation(
        status="completed",
        image_quality="clear",
        observations=(
            VisionObservationItem(
                description=(
                    "面板上的NET标签清晰可见"
                ),
                category="visible_text",
                confidence="high",
                region="面板中央",
            ),
        ),
        source_image_sha256=(
            TEST_IMAGE_SHA256
        ),
        prompt_version=(
            "robot-vision-observation-v1"
        ),
        model_name="fake-vision-model",
    )


def make_draft_output(
) -> DraftTestCaseToolOutput:
    """创建明确要求人工批准的只读测试草案。"""

    return DraftTestCaseToolOutput(
        title="测试用例草案：网络恢复核对",
        objective=(
            "验证网络恢复后的任务状态"
        ),
        preconditions=(
            "仅在隔离教学环境中使用。",
            "执行前必须由有权限人员批准。",
        ),
        steps=(
            DraftTestCaseStep(
                order=1,
                action="核对知识库证据。",
                expected_observation=(
                    "证据来源可以定位。"
                ),
            ),
            DraftTestCaseStep(
                order=2,
                action="准备经过批准的模拟条件。",
                expected_observation=(
                    "模拟条件已经记录。"
                ),
            ),
            DraftTestCaseStep(
                order=3,
                action="记录恢复后的任务状态。",
                expected_observation=(
                    "旧任务没有自动继续。"
                ),
            ),
        ),
        evidence=(
            DraftTestCaseEvidence(
                chunk_id=TEST_CHUNK_ID,
                document_id=TEST_DOCUMENT_ID,
                source_file="robot-demo.txt",
                page_or_section=(
                    "section: ERR-DEMO-1001"
                ),
                chunk_index=1,
                excerpt=(
                    "恢复前必须进行状态核对。"
                ),
                excerpt_truncated=False,
            ),
        ),
        limitations=(
            "草案不代表测试已经执行。",
            "投入使用前必须完成人工审核。",
        ),
    )


def test_create_initial_progress_from_continue_policy() -> None:
    """继续执行策略应生成collecting初始状态。

    被测试模块是AgentProgressReducer.create_initial_progress()。
    预期能力和工具范围原样来自已经验证的策略决定。
    """

    reducer = AgentProgressReducer()

    progress = reducer.create_initial_progress(
        make_knowledge_policy()
    )

    assert progress.state == "collecting"
    assert progress.required_capabilities == (
        "knowledge_evidence",
    )
    assert progress.allowed_next_tool_names == (
        "search_knowledge",
    )


def test_create_initial_progress_rejects_early_finish_policy() -> None:
    """Planner前已经拒答的策略不能进入collecting循环。

    被测试模块是create_initial_progress()。
    预期abstained策略被ValueError拒绝。
    """

    policy = AgentToolPolicyDecision(
        disposition="abstained",
        reason_codes=(
            "required_image_missing",
        ),
        public_message="缺少任务所需图片",
        missing_information=(
            "需要提供相关现场图片",
        ),
    )

    with pytest.raises(
        ValueError,
        match="continue_to_planner",
    ):
        AgentProgressReducer().create_initial_progress(
            policy
        )


def test_signature_is_stable_across_argument_order() -> None:
    """相同JSON参数的键顺序不能改变调用摘要。

    被测试模块是validate_tool_call()及其摘要函数。
    两个ToolCall使用不同call_id和不同字典键顺序，预期得到
    相同SHA-256摘要。
    """

    reducer = AgentProgressReducer()
    progress = reducer.create_initial_progress(
        make_knowledge_policy()
    )

    first = make_call(
        call_id="call_order_a",
        tool_name="search_knowledge",
        arguments={
            "query": "ERR-NET-4001",
            "top_k": 3,
        },
    )
    second = make_call(
        call_id="call_order_b",
        tool_name="search_knowledge",
        arguments={
            "top_k": 3,
            "query": "ERR-NET-4001",
        },
    )

    assert reducer.validate_tool_call(
        progress,
        first,
    ) == reducer.validate_tool_call(
        progress,
        second,
    )


def test_validate_tool_call_rejects_tool_outside_progress() -> None:
    """当前进度不能调用未获准的已注册工具。

    被测试模块是validate_tool_call()。
    知识任务故意请求遥测，预期得到tool_not_allowed。
    """

    reducer = AgentProgressReducer()
    progress = reducer.create_initial_progress(
        make_knowledge_policy()
    )
    call = make_call(
        call_id="call_forbidden_telemetry",
        tool_name="get_robot_telemetry",
        arguments={
            "robot_id": "robot-001",
        },
    )

    with pytest.raises(
        AgentProgressTransitionError,
        match="允许范围",
    ) as exc_info:
        reducer.validate_tool_call(
            progress,
            call,
        )

    assert exc_info.value.reason == (
        "tool_not_allowed"
    )


def test_empty_result_is_recorded_without_completion() -> None:
    """空结果应保留审计记录，但不能完成知识能力。

    被测试模块是record_interaction()。
    预期状态仍为collecting、工具仍可使用不同参数重试，
    工具记录保存empty_result且没有来源或证据。
    """

    reducer = AgentProgressReducer()
    initial = reducer.create_initial_progress(
        make_knowledge_policy()
    )
    call = make_call(
        call_id="call_empty",
        tool_name="search_knowledge",
        arguments={
            "query": "UNKNOWN-CODE",
            "top_k": 3,
        },
    )

    updated = reducer.record_interaction(
        initial,
        make_empty_interaction(
            step_number=1,
            tool_call=call,
        ),
    )

    assert updated.state == "collecting"
    assert updated.completed_capabilities == ()
    assert updated.allowed_next_tool_names == (
        "search_knowledge",
    )
    assert updated.tool_records[0].status == (
        "empty"
    )
    assert updated.tool_records[0].error_code == (
        "empty_result"
    )
    assert updated.confirmed_source_ids == ()
    assert updated.covered_evidence_ids == ()


def test_duplicate_call_is_rejected_after_empty_result() -> None:
    """更换call_id不能绕过相同工具参数的重复检测。

    被测试模块是record_interaction()和validate_tool_call()。
    第一次检索返回empty，第二次使用新call_id但相同参数。
    预期第二次在记录前被duplicate_tool_call拒绝。
    """

    reducer = AgentProgressReducer()
    initial = reducer.create_initial_progress(
        make_knowledge_policy()
    )
    first_call = make_call(
        call_id="call_duplicate_a",
        tool_name="search_knowledge",
        arguments={
            "query": "UNKNOWN-CODE",
            "top_k": 3,
        },
    )
    progress = reducer.record_interaction(
        initial,
        make_empty_interaction(
            step_number=1,
            tool_call=first_call,
        ),
    )
    repeated_call = make_call(
        call_id="call_duplicate_b",
        tool_name="search_knowledge",
        arguments={
            "top_k": 3,
            "query": "UNKNOWN-CODE",
        },
    )

    with pytest.raises(
        AgentProgressTransitionError,
        match="已经尝试过",
    ) as exc_info:
        reducer.record_interaction(
            progress,
            make_empty_interaction(
                step_number=2,
                tool_call=repeated_call,
            ),
        )

    assert exc_info.value.reason == (
        "duplicate_tool_call"
    )


def test_step_number_must_follow_existing_progress() -> None:
    """Reducer不能接受跳号的工具交互。

    被测试模块是record_interaction()。
    初始进度下一步应为1，测试传入步骤2，预期得到稳定的
    step_number_mismatch原因。
    """

    reducer = AgentProgressReducer()
    progress = reducer.create_initial_progress(
        make_knowledge_policy()
    )
    call = make_call(
        call_id="call_wrong_step",
        tool_name="search_knowledge",
        arguments={
            "query": "ERR-NET-4001",
        },
    )

    with pytest.raises(
        AgentProgressTransitionError,
        match="步骤",
    ) as exc_info:
        reducer.record_interaction(
            progress,
            make_empty_interaction(
                step_number=2,
                tool_call=call,
            ),
        )

    assert exc_info.value.reason == (
        "step_number_mismatch"
    )


def test_knowledge_success_extracts_document_and_chunk() -> None:
    """知识检索成功应分别记录文档来源和Chunk证据。

    被测试模块是record_interaction()中的知识输出适配。
    单能力完成后预期进入ready_to_finish并清空工具范围。
    """

    reducer = AgentProgressReducer()
    initial = reducer.create_initial_progress(
        make_knowledge_policy()
    )
    call = make_call(
        call_id="call_knowledge_success",
        tool_name="search_knowledge",
        arguments={
            "query": "ERR-NET-4001",
            "top_k": 3,
        },
    )

    updated = reducer.record_interaction(
        initial,
        make_success_interaction(
            step_number=1,
            tool_call=call,
            output=make_knowledge_output(),
        ),
    )

    assert updated.state == "ready_to_finish"
    assert updated.completed_capabilities == (
        "knowledge_evidence",
    )
    assert updated.confirmed_source_ids == (
        f"knowledge:{TEST_DOCUMENT_ID}",
    )
    assert updated.covered_evidence_ids == (
        TEST_CHUNK_ID,
    )
    assert updated.allowed_next_tool_names == ()


def test_telemetry_success_extracts_snapshot_source() -> None:
    """遥测成功应区分数据源和带时间的快照证据。

    被测试模块是Reducer的遥测输出适配。
    预期来源包含simulated_memory和robot_id，证据还包含时间。
    """

    policy = AgentToolPolicyDecision(
        disposition="continue_to_planner",
        required_capabilities=(
            "robot_telemetry",
        ),
        allowed_tool_names=(
            "get_robot_telemetry",
        ),
        reason_codes=(
            "robot_telemetry_required",
        ),
    )
    reducer = AgentProgressReducer()
    initial = reducer.create_initial_progress(
        policy
    )
    call = make_call(
        call_id="call_telemetry_success",
        tool_name="get_robot_telemetry",
        arguments={
            "robot_id": "robot-001",
        },
    )

    updated = reducer.record_interaction(
        initial,
        make_success_interaction(
            step_number=1,
            tool_call=call,
            output=make_telemetry_output(),
        ),
    )

    assert updated.state == "ready_to_finish"
    assert updated.confirmed_source_ids == (
        "telemetry:simulated_memory:robot-001",
    )
    assert updated.covered_evidence_ids == (
        "telemetry:robot-001:"
        "2026-09-07T08:30:00+00:00",
    )


def test_vision_success_uses_image_and_call_reference() -> None:
    """视觉成功应使用图片摘要标识来源并区分观察调用。

    被测试模块是Reducer的VisionObservation适配。
    预期不保存图片字节或analysis_goal正文。
    """

    policy = AgentToolPolicyDecision(
        disposition="continue_to_planner",
        required_capabilities=(
            "vision_observation",
        ),
        allowed_tool_names=(
            "analyze_robot_image",
        ),
        reason_codes=(
            "image_input_available",
            "vision_observation_required",
        ),
    )
    reducer = AgentProgressReducer()
    initial = reducer.create_initial_progress(
        policy
    )
    call = make_call(
        call_id="call_vision_success",
        tool_name="analyze_robot_image",
        arguments={
            "image_ref": "image-001",
            "analysis_goal": "检查NET标签",
        },
    )

    updated = reducer.record_interaction(
        initial,
        make_success_interaction(
            step_number=1,
            tool_call=call,
            output=make_vision_output(),
        ),
    )

    source_id = (
        f"vision:{TEST_IMAGE_SHA256}"
    )

    assert updated.confirmed_source_ids == (
        source_id,
    )
    assert updated.covered_evidence_ids == (
        source_id + ":call_vision_success",
    )
    assert "检查NET标签" not in str(
        updated.model_dump()
    )


def test_draft_success_does_not_duplicate_knowledge_evidence() -> None:
    """测试草案是派生产物，不能重复增加已有Chunk覆盖。

    被测试模块是Reducer的DraftTestCaseToolOutput适配。
    预期只增加test_draft来源，covered_evidence_ids保持为空。
    """

    policy = AgentToolPolicyDecision(
        disposition="continue_to_planner",
        required_capabilities=(
            "test_case_draft",
        ),
        allowed_tool_names=(
            "draft_test_case",
        ),
        reason_codes=(
            "test_case_draft_required",
        ),
    )
    reducer = AgentProgressReducer()
    initial = reducer.create_initial_progress(
        policy
    )
    call = make_call(
        call_id="call_draft_success",
        tool_name="draft_test_case",
        arguments={
            "objective": "验证网络恢复状态",
            "evidence_chunk_ids": [
                TEST_CHUNK_ID,
            ],
        },
    )

    updated = reducer.record_interaction(
        initial,
        make_success_interaction(
            step_number=1,
            tool_call=call,
            output=make_draft_output(),
        ),
    )

    assert updated.confirmed_source_ids == (
        "test_draft:call_draft_success",
    )
    assert updated.covered_evidence_ids == ()
    assert updated.state == "ready_to_finish"


def test_current_time_success_records_utc_evidence() -> None:
    """时间工具成功应记录系统时钟来源和UTC时间证据。

    被测试模块是Reducer的GetCurrentTimeToolOutput适配。
    """

    policy = AgentToolPolicyDecision(
        disposition="continue_to_planner",
        required_capabilities=(
            "current_time",
        ),
        allowed_tool_names=(
            "get_current_time",
        ),
        reason_codes=(
            "current_time_required",
        ),
    )
    reducer = AgentProgressReducer()
    initial = reducer.create_initial_progress(
        policy
    )
    call = make_call(
        call_id="call_time_success",
        tool_name="get_current_time",
        arguments={},
    )

    updated = reducer.record_interaction(
        initial,
        make_success_interaction(
            step_number=1,
            tool_call=call,
            output=GetCurrentTimeToolOutput(
                current_time=TEST_OBSERVED_AT,
            ),
        ),
    )

    assert updated.confirmed_source_ids == (
        "system_clock",
    )
    assert updated.covered_evidence_ids == (
        "system_clock:2026-09-07T08:30:00+00:00",
    )


def test_multiple_capabilities_narrow_after_each_success() -> None:
    """多能力任务应完成一项后只保留尚未完成的工具。

    被测试模块是record_interaction()的状态转换主流程。
    知识成功后只剩遥测；遥测成功后进入ready_to_finish。
    """

    policy = AgentToolPolicyDecision(
        disposition="continue_to_planner",
        required_capabilities=(
            "knowledge_evidence",
            "robot_telemetry",
        ),
        allowed_tool_names=(
            "search_knowledge",
            "get_robot_telemetry",
        ),
        reason_codes=(
            "knowledge_evidence_required",
            "robot_telemetry_required",
        ),
    )
    reducer = AgentProgressReducer()
    initial = reducer.create_initial_progress(
        policy
    )
    knowledge_call = make_call(
        call_id="call_multi_knowledge",
        tool_name="search_knowledge",
        arguments={
            "query": "ERR-NET-4001",
        },
    )
    after_knowledge = reducer.record_interaction(
        initial,
        make_success_interaction(
            step_number=1,
            tool_call=knowledge_call,
            output=make_knowledge_output(),
        ),
    )

    assert after_knowledge.state == "collecting"
    assert after_knowledge.allowed_next_tool_names == (
        "get_robot_telemetry",
    )

    telemetry_call = make_call(
        call_id="call_multi_telemetry",
        tool_name="get_robot_telemetry",
        arguments={
            "robot_id": "robot-001",
        },
    )
    finished = reducer.record_interaction(
        after_knowledge,
        make_success_interaction(
            step_number=2,
            tool_call=telemetry_call,
            output=make_telemetry_output(),
        ),
    )

    assert finished.state == "ready_to_finish"
    assert finished.completed_capabilities == (
        "knowledge_evidence",
        "robot_telemetry",
    )
    assert finished.allowed_next_tool_names == ()
    assert len(finished.tool_records) == 2


def test_success_output_is_revalidated_by_specific_model() -> None:
    """success中的任意dict不能直接成为可信进度来源。

    被测试模块是Reducer的工具输出适配。
    测试构造通用ToolExecutionResult允许的dict，但它不符合
    SearchKnowledgeToolOutput。预期Pydantic重新校验时失败。
    """

    reducer = AgentProgressReducer()
    progress = reducer.create_initial_progress(
        make_knowledge_policy()
    )
    call = make_call(
        call_id="call_invalid_output",
        tool_name="search_knowledge",
        arguments={
            "query": "ERR-NET-4001",
        },
    )
    interaction = AgentToolInteraction(
        step_number=1,
        tool_call=call,
        result=ToolExecutionResult(
            call_id=call.call_id,
            tool_name=call.tool_name,
            status="success",
            output={
                "invented": "not trusted",
            },
            duration_ms=1.0,
        ),
    )

    with pytest.raises(ValidationError):
        reducer.record_interaction(
            progress,
            interaction,
        )


def test_record_interaction_does_not_mutate_old_progress() -> None:
    """Reducer必须返回新快照，不能修改旧状态对象。

    被测试模块是record_interaction()。
    预期旧进度仍无工具记录，新进度包含一步并已准备finish。
    """

    reducer = AgentProgressReducer()
    initial = reducer.create_initial_progress(
        make_knowledge_policy()
    )
    call = make_call(
        call_id="call_immutable",
        tool_name="search_knowledge",
        arguments={
            "query": "ERR-NET-4001",
        },
    )

    updated = reducer.record_interaction(
        initial,
        make_success_interaction(
            step_number=1,
            tool_call=call,
            output=make_knowledge_output(),
        ),
    )

    assert updated is not initial
    assert initial.state == "collecting"
    assert initial.tool_records == ()
    assert updated.state == "ready_to_finish"
    assert len(updated.tool_records) == 1


def make_two_capability_policy(
) -> AgentToolPolicyDecision:
    """创建同时需要知识证据和模拟遥测的测试策略。"""

    return AgentToolPolicyDecision(
        disposition="continue_to_planner",
        required_capabilities=(
            "knowledge_evidence",
            "robot_telemetry",
        ),
        allowed_tool_names=(
            "search_knowledge",
            "get_robot_telemetry",
        ),
        reason_codes=(
            "knowledge_evidence_required",
            "robot_telemetry_required",
        ),
    )


def make_progress_after_knowledge(
) -> AgentProgress:
    """创建只完成知识能力、仍缺遥测能力的进度。"""

    reducer = AgentProgressReducer()
    initial = reducer.create_initial_progress(
        make_two_capability_policy()
    )
    call = make_call(
        call_id="call_terminal_knowledge",
        tool_name="search_knowledge",
        arguments={
            "query": "ERR-NET-4001",
        },
    )

    return reducer.record_interaction(
        initial,
        make_success_interaction(
            step_number=1,
            tool_call=call,
            output=make_knowledge_output(),
        ),
    )


def make_ready_two_capability_progress(
) -> AgentProgress:
    """创建知识和遥测能力都已完成的待结束进度。"""

    reducer = AgentProgressReducer()
    after_knowledge = (
        make_progress_after_knowledge()
    )
    telemetry_call = make_call(
        call_id="call_terminal_telemetry",
        tool_name="get_robot_telemetry",
        arguments={
            "robot_id": "robot-001",
        },
    )

    return reducer.record_interaction(
        after_knowledge,
        make_success_interaction(
            step_number=2,
            tool_call=telemetry_call,
            output=make_telemetry_output(),
        ),
    )


def test_ready_progress_accepts_task_completed() -> None:
    """能力全部完成后，Planner的完成决定应成为completed。

    被测试方法是finalize_planner_decision()。
    输入是ready_to_finish进度和task_completed决定；预期返回
    sufficient_evidence终态，并且原进度对象保持不变。
    """

    reducer = AgentProgressReducer()
    ready = make_ready_two_capability_progress()

    completed = reducer.finalize_planner_decision(
        ready,
        finish_reason="task_completed",
    )

    assert ready.state == "ready_to_finish"
    assert completed.state == "completed"
    assert completed.stop_reason == (
        "sufficient_evidence"
    )
    assert completed.missing_information == ()
    assert completed.allowed_next_tool_names == ()


def test_early_task_completed_without_evidence_abstains() -> None:
    """没有完成任何能力时，Planner不能自行宣布任务完成。

    被测试方法是finalize_planner_decision()的防越权分支。
    输入是初始collecting状态；预期Reducer拒绝completed声明，
    转为abstained并根据未完成能力生成稳定缺失信息。
    """

    reducer = AgentProgressReducer()
    initial = reducer.create_initial_progress(
        make_knowledge_policy()
    )

    result = reducer.finalize_planner_decision(
        initial,
        finish_reason="task_completed",
    )

    assert result.state == "abstained"
    assert result.stop_reason == (
        "insufficient_information"
    )
    assert result.missing_information == (
        "尚未取得任务所需的知识库证据",
    )


def test_early_task_completed_with_some_evidence_is_partial() -> None:
    """完成部分必要能力后提前结束，应保留为partial。

    被测试方法是finalize_planner_decision()。
    输入已完成知识证据但尚未完成遥测的进度；预期状态为
    partial，并明确记录缺少模拟遥测快照。
    """

    reducer = AgentProgressReducer()
    progress = make_progress_after_knowledge()

    result = reducer.finalize_planner_decision(
        progress,
        finish_reason="task_completed",
    )

    assert result.state == "partial"
    assert result.stop_reason == "partial_evidence"
    assert result.completed_capabilities == (
        "knowledge_evidence",
    )
    assert result.missing_information == (
        "尚未取得任务所需的模拟遥测快照",
    )


def test_insufficient_information_merges_declared_and_computed_gaps() -> None:
    """信息不足终止应合并Planner说明和确定性能力缺口。

    被测试方法是finalize_planner_decision()的信息不足分支。
    预期用户可读说明保留首次出现顺序，而且不会重复同一项。
    """

    reducer = AgentProgressReducer()
    progress = make_progress_after_knowledge()

    result = reducer.finalize_planner_decision(
        progress,
        finish_reason=(
            "insufficient_information"
        ),
        missing_information=(
            "需要核对实时遥测来源",
            "需要核对实时遥测来源",
        ),
    )

    assert result.state == "partial"
    assert result.missing_information == (
        "需要核对实时遥测来源",
        "尚未取得任务所需的模拟遥测快照",
    )


def test_ready_progress_can_still_report_insufficient_information() -> None:
    """工具能力齐全不代表Planner一定形成了可信最终草稿。

    被测试方法是finalize_planner_decision()。
    ready_to_finish只说明输入资料已经收齐；如果Planner仍声明
    信息不足，Reducer应返回partial并生成明确复核事项。
    """

    reducer = AgentProgressReducer()
    ready = make_ready_two_capability_progress()

    result = reducer.finalize_planner_decision(
        ready,
        finish_reason=(
            "insufficient_information"
        ),
    )

    assert result.state == "partial"
    assert result.stop_reason == "partial_evidence"
    assert result.missing_information == (
        "Planner声明现有信息仍不足，需要复核最终诊断草稿",
    )


def test_planner_can_request_human_review() -> None:
    """Planner识别到风险边界时应进入人工审核终态。

    被测试方法是finalize_planner_decision()的人工审核分支。
    预期保留具体审核原因，清空下一步工具，并使用稳定的
    human_review_required停止原因。
    """

    reducer = AgentProgressReducer()
    progress = make_progress_after_knowledge()

    result = reducer.finalize_planner_decision(
        progress,
        finish_reason="human_review_required",
        missing_information=(
            "需要现场人员确认安全保护状态",
        ),
    )

    assert result.state == (
        "human_review_required"
    )
    assert result.stop_reason == (
        "human_review_required"
    )
    assert result.allowed_next_tool_names == ()
    assert result.missing_information == (
        "需要现场人员确认安全保护状态",
        "尚未取得任务所需的模拟遥测快照",
    )


def test_confirmed_conflict_requires_human_review() -> None:
    """两个已确认来源发生冲突时不能自动选择其中一个。

    被测试方法是record_conflicts()。
    输入引用已写入Progress的知识和遥测来源；预期状态转为
    human_review_required，并保留结构化冲突供审计。
    """

    reducer = AgentProgressReducer()
    ready = make_ready_two_capability_progress()
    conflict = AgentEvidenceConflict(
        description=(
            "知识规则与模拟遥测状态需要人工核对"
        ),
        source_ids=(
            "knowledge:" + TEST_DOCUMENT_ID,
            "telemetry:simulated_memory:robot-001",
        ),
        evidence_ids=(
            TEST_CHUNK_ID,
            (
                "telemetry:robot-001:"
                "2026-09-07T08:30:00+00:00"
            ),
        ),
    )

    result = reducer.record_conflicts(
        ready,
        conflicts=(conflict,),
    )

    assert result.state == (
        "human_review_required"
    )
    assert result.stop_reason == (
        "conflicting_evidence"
    )
    assert result.evidence_conflicts == (
        conflict,
    )
    assert result.missing_information == (
        "已确认来源之间存在冲突，需要人工核对原始证据",
    )


def test_record_conflicts_rejects_empty_collection() -> None:
    """没有冲突内容时不能伪造conflicting_evidence终态。

    被测试方法是record_conflicts()的输入边界。
    预期空元组在构造终态前被ValueError拒绝。
    """

    reducer = AgentProgressReducer()
    ready = make_ready_two_capability_progress()

    with pytest.raises(
        ValueError,
        match="至少一条冲突",
    ):
        reducer.record_conflicts(
            ready,
            conflicts=(),
        )


def test_runtime_failure_requires_explicit_missing_information() -> None:
    """不可恢复运行错误必须形成有解释的failed终态。

    被测试方法是fail_progress()。
    第一次调用验证正常失败转换；第二次验证空说明会被拒绝，
    避免API只返回一个没有上下文的failed标签。
    """

    reducer = AgentProgressReducer()
    initial = reducer.create_initial_progress(
        make_knowledge_policy()
    )

    failed = reducer.fail_progress(
        initial,
        missing_information=(
            "Planner服务连续失败，任务未形成可信结果",
        ),
    )

    assert failed.state == "failed"
    assert failed.stop_reason == "runtime_failure"
    assert failed.missing_information == (
        "Planner服务连续失败，任务未形成可信结果",
    )

    with pytest.raises(
        ValueError,
        match="必须说明未完成内容",
    ):
        reducer.fail_progress(
            initial,
            missing_information=(),
        )


def test_terminal_progress_cannot_transition_again() -> None:
    """同一进度到达终态后不能被再次改写。

    被测试方法是所有终态方法共享的活动状态校验。
    预期第二次转换抛出带稳定reason的专用异常，使Runner
    无需解析中文异常消息也能识别状态机违规。
    """

    reducer = AgentProgressReducer()
    ready = make_ready_two_capability_progress()
    completed = (
        reducer.finalize_planner_decision(
            ready,
            finish_reason="task_completed",
        )
    )

    with pytest.raises(
        AgentProgressTransitionError,
    ) as exc_info:
        reducer.fail_progress(
            completed,
            missing_information=(
                "不应写入的后续失败",
            ),
        )

    assert exc_info.value.reason == (
        "progress_already_terminal"
    )
