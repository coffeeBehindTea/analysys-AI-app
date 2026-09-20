"""Agent工具公开轨迹统一构造器的离线测试。

本文件验证：

1. 五种现有只读工具使用各自的脱敏输入摘要；
2. 五种success输出会恢复成正确的Pydantic模型；
3. 公开结果摘要不会复制知识库正文或完整内部数据；
4. empty等非成功结果会保留状态、错误码和公开消息；
5. 尚未专门适配的新工具会使用安全的通用摘要；
6. 伪造的success输出会在公开前被拒绝；
7. 构造函数会拒绝错误的输入对象类型。

测试不调用真实LLM、网络、Chroma、Vision Provider或Agent工具。
"""

from datetime import (
    datetime,
    timezone,
)

import pytest

from app.schemas.agent import (
    ToolCall,
    ToolExecutionResult,
)
from app.schemas.agent_planning import (
    AgentToolInteraction,
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
from app.services.agent_tool_trace_builder import (
    AgentToolTraceBuildResult,
    build_agent_tool_input_summary,
    build_agent_tool_trace,
)


# 固定文档、Chunk和图片摘要只用于离线测试。
# 这些值不对应真实设备、客户数据或生产知识库。
TEST_DOCUMENT_ID = "d" * 64
TEST_CHUNK_ID = (
    TEST_DOCUMENT_ID + ":000001"
)
TEST_IMAGE_SHA256 = "e" * 64


# 固定时间避免测试依赖当前系统时钟。
TEST_OBSERVED_AT = datetime(
    2026,
    9,
    7,
    8,
    30,
    tzinfo=timezone.utc,
)


def make_call(
    *,
    call_id: str,
    tool_name: str,
    arguments: dict[str, object],
) -> ToolCall:
    """创建经过通用ToolCall契约校验的工具调用。"""

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
    """把一个Pydantic输出包装成已完成的成功交互。"""

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


def make_search_output(
) -> SearchKnowledgeToolOutput:
    """创建包含一条可定位引用的知识检索输出。"""

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
                    "PRIVATE_EVIDENCE_CONTENT"
                ),
            ),
        ],
        retrieval_ms=8.5,
    )


def make_telemetry_output(
) -> RobotTelemetryToolOutput:
    """创建明确标记为模拟来源的遥测输出。"""

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


def make_draft_output(
) -> DraftTestCaseToolOutput:
    """创建需要人工批准的只读测试草案。"""

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


def make_time_output(
) -> GetCurrentTimeToolOutput:
    """创建固定的应用服务器UTC时间观察。"""

    return GetCurrentTimeToolOutput(
        current_time=TEST_OBSERVED_AT,
    )


def make_vision_output(
) -> VisionObservation:
    """创建一份确定性的结构化视觉观察。"""

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
        model_name="fake-vision-model",
    )


def test_search_input_summary_redacts_query_text(
) -> None:
    """知识检索摘要应公开长度和top_k，不公开原问题。

    被测试模块是build_agent_tool_input_summary()。
    测试向ToolCall放入可识别的敏感占位文本，预期摘要只含
    query_chars和top_k，不能复制原始query。
    """

    private_query = (
        "PRIVATE_QUERY ERR-NET-4001"
    )
    tool_call = make_call(
        call_id="call-search-input",
        tool_name="search_knowledge",
        arguments={
            "query": private_query,
            "top_k": 5,
        },
    )

    summary = (
        build_agent_tool_input_summary(
            tool_call
        )
    )

    assert summary == (
        "知识库检索；"
        f"query_chars={len(private_query)}；"
        "top_k=5"
    )
    assert private_query not in summary


@pytest.mark.parametrize(
    (
        "tool_name",
        "arguments",
        "expected_summary",
        "private_value",
    ),
    [
        (
            "get_robot_telemetry",
            {"robot_id": "robot-001"},
            (
                "读取脱敏模拟遥测；"
                "robot_id=robot-001"
            ),
            None,
        ),
        (
            "draft_test_case",
            {
                "objective": (
                    "PRIVATE_TEST_OBJECTIVE"
                ),
                "evidence_chunk_ids": [
                    TEST_CHUNK_ID,
                ],
            },
            (
                "生成测试草案；"
                "objective_chars=22；"
                "evidence_count=1"
            ),
            "PRIVATE_TEST_OBJECTIVE",
        ),
        (
            "get_current_time",
            {},
            (
                "读取应用服务器UTC时间；"
                "无输入参数"
            ),
            None,
        ),
        (
            "analyze_robot_image",
            {
                "image_ref": "image_001",
                "analysis_goal": (
                    "PRIVATE_ANALYSIS_GOAL"
                ),
            },
            (
                "分析请求级图片；"
                "image_ref=image_001；"
                "analysis_goal_chars=21"
            ),
            "PRIVATE_ANALYSIS_GOAL",
        ),
    ],
)
def test_known_tool_input_summaries_are_minimal(
    tool_name: str,
    arguments: dict[str, object],
    expected_summary: str,
    private_value: str | None,
) -> None:
    """其余四种工具应只公开各自所需的最小输入信息。

    被测试模块仍是build_agent_tool_input_summary()。
    参数化案例覆盖模拟遥测、测试草案、系统时间和图片分析。
    草案目标与图片分析目标只允许公开字符数。
    """

    tool_call = make_call(
        call_id=f"call-{tool_name}",
        tool_name=tool_name,
        arguments=arguments,
    )

    summary = (
        build_agent_tool_input_summary(
            tool_call
        )
    )

    assert summary == expected_summary

    if private_value is not None:
        assert private_value not in summary


def test_search_trace_returns_typed_output_without_excerpt(
) -> None:
    """知识成功输出应恢复类型，但轨迹不能泄漏证据正文。

    build_agent_tool_trace()预期重新校验SearchKnowledgeToolOutput，
    返回一条success轨迹，并只公开引用数量和检索耗时。
    """

    interaction = make_success_interaction(
        step_number=1,
        tool_call=make_call(
            call_id="call-search",
            tool_name="search_knowledge",
            arguments={
                "query": "PRIVATE_QUERY",
                "top_k": 3,
            },
        ),
        output=make_search_output(),
    )

    built = build_agent_tool_trace(
        interaction
    )

    assert isinstance(
        built,
        AgentToolTraceBuildResult,
    )
    assert isinstance(
        built.typed_output,
        SearchKnowledgeToolOutput,
    )
    assert built.trace.step_id == 1
    assert built.trace.tool_name == (
        "search_knowledge"
    )
    assert built.trace.status == "success"
    assert built.trace.result_summary == (
        "知识库返回1条已确认引用；"
        "retrieval_ms=8.500"
    )
    assert (
        "PRIVATE_EVIDENCE_CONTENT"
        not in built.trace.result_summary
    )


@pytest.mark.parametrize(
    (
        "tool_name",
        "arguments",
        "output",
        "expected_type",
        "expected_result_summary",
    ),
    [
        (
            "get_robot_telemetry",
            {"robot_id": "robot-001"},
            make_telemetry_output(),
            RobotTelemetryToolOutput,
            (
                "取得脱敏模拟遥测；"
                "robot_id=robot-001；"
                "state=paused；fault_count=1"
            ),
        ),
        (
            "draft_test_case",
            {
                "objective": "恢复验证",
                "evidence_chunk_ids": [
                    TEST_CHUNK_ID,
                ],
            },
            make_draft_output(),
            DraftTestCaseToolOutput,
            (
                "生成待人工批准测试草案；"
                "step_count=3；evidence_count=1；"
                "draft_only=true；"
                "requires_human_approval=true"
            ),
        ),
        (
            "get_current_time",
            {},
            make_time_output(),
            GetCurrentTimeToolOutput,
            "已读取应用服务器UTC系统时间",
        ),
        (
            "analyze_robot_image",
            {
                "image_ref": "image_001",
                "analysis_goal": "检查NET指示灯",
            },
            make_vision_output(),
            VisionObservation,
            (
                "取得结构化视觉观察；"
                "status=completed；"
                "image_quality=clear；"
                "observation_count=1；"
                "indicator_count=0；"
                "requires_human_check=false"
            ),
        ),
    ],
)
def test_known_success_outputs_use_specific_contracts(
    tool_name: str,
    arguments: dict[str, object],
    output: object,
    expected_type: type[object],
    expected_result_summary: str,
) -> None:
    """四种非检索success输出应使用专属模型和结果摘要。

    参数化案例验证遥测、测试草案、时间和Vision输出的
    二次Pydantic校验，并核对对应的公开摘要字段。
    """

    interaction = make_success_interaction(
        step_number=2,
        tool_call=make_call(
            call_id=f"call-{tool_name}",
            tool_name=tool_name,
            arguments=arguments,
        ),
        output=output,
    )

    built = build_agent_tool_trace(
        interaction
    )

    assert isinstance(
        built.typed_output,
        expected_type,
    )
    assert built.trace.step_id == 2
    assert built.trace.tool_name == tool_name
    assert built.trace.result_summary == (
        expected_result_summary
    )


def test_non_success_trace_preserves_public_failure_fields(
) -> None:
    """empty结果应进入公开轨迹，但不能产生typed_output。

    预期流程是跳过success输出校验，保留empty状态、
    empty_result错误码、公开消息和耗时。
    """

    tool_call = make_call(
        call_id="call-empty",
        tool_name="search_knowledge",
        arguments={
            "query": "不存在的故障代码",
            "top_k": 3,
        },
    )
    interaction = AgentToolInteraction(
        step_number=3,
        tool_call=tool_call,
        result=ToolExecutionResult(
            call_id=tool_call.call_id,
            tool_name=tool_call.tool_name,
            status="empty",
            error_code="empty_result",
            public_message=(
                "工具没有返回可用结果"
            ),
            duration_ms=5.25,
        ),
    )

    built = build_agent_tool_trace(
        interaction
    )

    assert built.typed_output is None
    assert built.trace.status == "empty"
    assert built.trace.error_code == (
        "empty_result"
    )
    assert built.trace.duration_ms == 5.25
    assert built.trace.result_summary == (
        "status=empty；"
        "error_code=empty_result；"
        "message=工具没有返回可用结果"
    )


def test_unknown_success_tool_uses_generic_summaries(
) -> None:
    """未来新工具未适配时应退化成字段计数摘要。

    ToolCall允许符合命名规则的新工具名。本测试确认构造器
    不会公开未知参数和值，也不会把未知输出标为可信类型。
    """

    tool_call = make_call(
        call_id="call-future",
        tool_name="future_read_only_tool",
        arguments={
            "private_a": "secret-a",
            "private_b": "secret-b",
        },
    )
    interaction = AgentToolInteraction(
        step_number=4,
        tool_call=tool_call,
        result=ToolExecutionResult(
            call_id=tool_call.call_id,
            tool_name=tool_call.tool_name,
            status="success",
            output={
                "private_result": "secret-result",
            },
            duration_ms=1.0,
        ),
    )

    built = build_agent_tool_trace(
        interaction
    )

    assert built.typed_output is None
    assert built.trace.input_summary == (
        "调用注册工具；argument_fields=2"
    )
    assert built.trace.result_summary == (
        "注册工具执行成功；output_fields=1"
    )
    assert "secret" not in (
        built.trace.input_summary
        + built.trace.result_summary
    )


def test_invalid_known_success_output_is_rejected(
) -> None:
    """已知工具伪造success但输出非法时必须拒绝公开。

    测试绕过ToolExecutor直接构造一条错误结果。构造器预期
    使用SearchKnowledgeToolOutput.model_validate()发现错误，
    抛出不包含原始敏感输出的TypeError。
    """

    tool_call = make_call(
        call_id="call-invalid-success",
        tool_name="search_knowledge",
        arguments={
            "query": "测试非法成功结果",
        },
    )
    interaction = AgentToolInteraction(
        step_number=5,
        tool_call=tool_call,
        result=ToolExecutionResult(
            call_id=tool_call.call_id,
            tool_name=tool_call.tool_name,
            status="success",
            output={
                "secret_payload": (
                    "PRIVATE_INVALID_OUTPUT"
                ),
            },
            duration_ms=1.0,
        ),
    )

    with pytest.raises(
        TypeError,
        match=(
            "search_knowledge"
            "成功结果不符合输出契约"
        ),
    ) as exception_info:
        build_agent_tool_trace(
            interaction
        )

    assert (
        "PRIVATE_INVALID_OUTPUT"
        not in str(exception_info.value)
    )


def test_public_builders_reject_wrong_input_types(
) -> None:
    """两个公开构造函数都应在边界拒绝错误对象。

    预期错误在访问对象属性前产生，从而返回稳定、可定位的
    TypeError，而不是偶然出现AttributeError。
    """

    with pytest.raises(
        TypeError,
        match="tool_call必须是ToolCall",
    ):
        build_agent_tool_input_summary(
            object(),  # type: ignore[arg-type]
        )

    with pytest.raises(
        TypeError,
        match=(
            "interaction必须是"
            "AgentToolInteraction"
        ),
    ):
        build_agent_tool_trace(
            object(),  # type: ignore[arg-type]
        )
