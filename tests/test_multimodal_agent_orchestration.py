"""多模态Agent多步编排的离线集成测试。

本模块不访问真实LLM、Vision API、Embedding、Chroma或机器人。
它使用固定ScriptedPlanner代替LLM Planner，但保留真实的：

1. AgentDiagnosisService图片准备和最终响应汇总；
2. AgentRunner规划、工具调用、观察循环；
3. ToolRegistry与ToolExecutor工具白名单和执行边界；
4. analyze_robot_image与get_robot_telemetry工具Handler；
5. 请求级图片Store和请求级已确认证据Store。

因此，本模块验证的是“编排代码能否正确连接各层”，
而不是验证外部模型是否每次都能做出同样的规划。
"""

from base64 import b64encode
from copy import deepcopy
from datetime import datetime, timezone
from io import BytesIO
import json

from PIL import Image
from pydantic import BaseModel
import pytest

from app.agent.evidence_store import ConfirmedEvidenceStore
from app.agent.executor import ToolExecutor
from app.agent.registry import ToolRegistry
from app.agent.runner import AgentRunner
from app.agent.tools.analyze_robot_image import (
    ANALYZE_ROBOT_IMAGE_TOOL_DEFINITION,
    AnalyzeRobotImageToolHandler,
)
from app.agent.tools.robot_telemetry import (
    GET_ROBOT_TELEMETRY_TOOL_DEFINITION,
    GetRobotTelemetryToolHandler,
    InMemoryRobotTelemetryStore,
)
from app.agent.tools.search_knowledge import (
    SEARCH_KNOWLEDGE_TOOL_DEFINITION,
)
from app.agent.vision_input_store import RequestVisionInputStore
from app.schemas.agent import ToolCall
from app.schemas.agent_api import AgentDiagnosisRequest
from app.schemas.agent_diagnosis import AgentDiagnosisDraft
from app.schemas.agent_planning import (
    AgentPlannerDecision,
    AgentPlanningContext,
)
from app.schemas.agent_tools import (
    RobotTelemetryToolOutput,
    SearchKnowledgeToolInput,
    SearchKnowledgeToolOutput,
)
from app.schemas.diagnostics import DiagnosisCause, DiagnosisCheck
from app.schemas.knowledge_query import KnowledgeCitation
from app.schemas.vision import (
    VisionImagePayload,
    VisionModelDraft,
    VisionObservationItem,
    VisionVisibleIndicator,
)
from app.services.agent_diagnosis_service import AgentDiagnosisService
from app.services.fake_vision_provider import FakeVisionProvider
from app.services.vision_input import VisionInputAdapter


# 所有测试使用同一个公开的Planner版本标签，
# 方便断言响应中的追踪字段来自当前多模态版本。
TEST_PLANNER_PROMPT_VERSION = "agent-tool-calling-v3"

# 合成图片很小，因此测试使用显式且远高于图片实际大小的限制。
# 这些限制仍会经过真实VisionInputAdapter检查。
TEST_MAX_IMAGE_SIZE_BYTES = 100_000
TEST_MAX_IMAGE_DIMENSION_PX = 128
TEST_MAX_IMAGE_PIXELS = 16_384

# Chunk ID沿用生产契约要求的“64位文档ID:六位索引”形式。
TEST_DOCUMENT_ID = "c" * 64
TEST_CHUNK_ID = f"{TEST_DOCUMENT_ID}:000001"


class ScriptedPlanner:
    """按固定顺序返回决定的离线Planner。

    decisions表示预先编写的规划轨迹；contexts保存Runner每一轮
    真正传给Planner的上下文，用于验证上一轮工具结果已经回填。
    """

    def __init__(
        self,
        *,
        decisions: tuple[AgentPlannerDecision, ...],
    ) -> None:
        self._decisions = decisions
        self.contexts: list[AgentPlanningContext] = []
        self.tool_schemas_seen: list[
            tuple[dict[str, object], ...]
        ] = []

    async def plan(
        self,
        *,
        context: AgentPlanningContext,
        tool_schemas: tuple[dict[str, object], ...],
    ) -> AgentPlannerDecision:
        """记录Runner输入并返回对应步骤的固定决定。"""

        decision_index = len(self.contexts)

        if decision_index >= len(self._decisions):
            raise AssertionError("Runner调用Planner的次数超过固定轨迹")

        self.contexts.append(context.model_copy(deep=True))
        self.tool_schemas_seen.append(deepcopy(tool_schemas))
        return self._decisions[decision_index]


class RecordingKnowledgeHandler:
    """返回固定知识引用并同步写入请求级证据白名单。

    真实search_knowledge内部检索已由自己的测试覆盖；本集成测试
    只需要一个确定性Handler验证Agent各层之间的连接关系。
    """

    def __init__(
        self,
        *,
        evidence_store: ConfirmedEvidenceStore,
        citation: KnowledgeCitation,
    ) -> None:
        self._evidence_store = evidence_store
        self._citation = citation
        self.queries: list[str] = []

    async def __call__(
        self,
        tool_input: BaseModel,
        /,
    ) -> BaseModel | None:
        """校验真实输入契约，记录查询并返回固定引用。"""

        if not isinstance(tool_input, SearchKnowledgeToolInput):
            raise TypeError(
                "tool_input必须是SearchKnowledgeToolInput"
            )

        self.queries.append(tool_input.query)

        # 生产Service只允许最终草稿引用当前请求Store内的Chunk。
        # 因此Fake Handler必须模拟真实工具的这项关键副作用。
        self._evidence_store.record((self._citation,))

        return SearchKnowledgeToolOutput(
            citations=[self._citation],
            retrieval_ms=12.5,
        )


def make_vision_payload(
    *,
    analysis_goal: str,
    color: tuple[int, int, int] = (210, 20, 50),
) -> VisionImagePayload:
    """在内存中生成合法PNG，并包装成API图片载荷。"""

    output = BytesIO()

    with Image.new("RGB", (12, 8), color=color) as image:
        image.save(output, format="PNG")

    return VisionImagePayload(
        mime_type="image/png",
        encoding="base64",
        image_base64=b64encode(output.getvalue()).decode("ascii"),
        analysis_goal=analysis_goal,
        detail="auto",
    )


def make_adapter() -> VisionInputAdapter:
    """创建会真实校验合成图片的输入适配器。"""

    return VisionInputAdapter(
        max_image_size_bytes=TEST_MAX_IMAGE_SIZE_BYTES,
        max_image_dimension_px=TEST_MAX_IMAGE_DIMENSION_PX,
        max_image_pixels=TEST_MAX_IMAGE_PIXELS,
    )


def make_request(
    *,
    image: VisionImagePayload,
    symptom: str,
    log_excerpt: str,
    task_goal: str,
) -> AgentDiagnosisRequest:
    """创建只带一张图片的固定Agent诊断请求。"""

    return AgentDiagnosisRequest(
        robot_id="robot-001",
        symptom=symptom,
        log_excerpt=log_excerpt,
        task_goal=task_goal,
        images=[image],
    )


def make_citation() -> KnowledgeCitation:
    """创建可用于最终诊断白名单校验的固定知识引用。"""

    return KnowledgeCitation(
        chunk_id=TEST_CHUNK_ID,
        document_id=TEST_DOCUMENT_ID,
        source_file="robot-network-demo.txt",
        page_or_section="section: ERR-NET-4001",
        chunk_index=1,
        rank=1,
        similarity=0.78,
        rrf_score=0.0327,
        excerpt=(
            "网络通信异常时应先核对链路状态；"
            "状态确认后再按照批准流程恢复任务。"
        ),
    )


def make_telemetry() -> RobotTelemetryToolOutput:
    """创建明确标记为模拟来源的只读遥测快照。"""

    return RobotTelemetryToolOutput(
        robot_id="robot-001",
        observed_at=datetime(
            2026,
            8,
            28,
            8,
            0,
            tzinfo=timezone.utc,
        ),
        location="warehouse-demo/aisle-07/node-14",
        battery_percent=42.5,
        operational_state="paused",
        current_task_id="task-demo-1001",
        active_fault_codes=("ERR-NET-4001",),
        speed_mps=0.0,
        network_connected=True,
    )


def call_tool(
    *,
    call_id: str,
    tool_name: str,
    arguments: dict[str, object],
) -> AgentPlannerDecision:
    """创建一项合法的结构化工具调用决定。"""

    return AgentPlannerDecision(
        decision="call_tool",
        tool_call=ToolCall(
            call_id=call_id,
            tool_name=tool_name,
            arguments=arguments,
        ),
    )


def finish_completed() -> AgentPlannerDecision:
    """创建引用当前请求真实Chunk的完整诊断决定。"""

    draft = AgentDiagnosisDraft(
        status="completed",
        possible_causes=(
            DiagnosisCause(
                description="可见网络状态与已知通信异常相符",
                evidence_chunk_ids=[TEST_CHUNK_ID],
            ),
        ),
        next_checks=(
            DiagnosisCheck(
                description="核对链路与任务状态后再按批准流程恢复",
                evidence_chunk_ids=[TEST_CHUNK_ID],
                risk_level="medium",
                requires_qualified_person=False,
            ),
        ),
        risk_level="medium",
        missing_information=(),
        abstained=False,
    )

    return AgentPlannerDecision(
        decision="finish",
        finish_reason="task_completed",
        final_message=json.dumps(
            draft.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
        ),
    )


def finish_abstained() -> AgentPlannerDecision:
    """创建图片不可用时的安全拒答决定。"""

    draft = AgentDiagnosisDraft(
        status="abstained",
        possible_causes=(),
        next_checks=(),
        risk_level="unknown",
        missing_information=(
            "图片严重模糊，无法确认面板指示灯状态；需要补充清晰图片",
        ),
        abstained=True,
    )

    return AgentPlannerDecision(
        decision="finish",
        finish_reason="insufficient_information",
        final_message=json.dumps(
            draft.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
        ),
    )


def make_service(
    *,
    planner: ScriptedPlanner,
    image: VisionImagePayload,
    vision_draft: VisionModelDraft,
    include_telemetry: bool,
) -> tuple[
    AgentDiagnosisService,
    FakeVisionProvider,
    RecordingKnowledgeHandler,
]:
    """组装真实Runner、Executor、Registry和请求级Store。"""

    adapter = make_adapter()

    # 先独立适配一次只为取得确定性SHA；Service运行时仍会再次执行
    # 完整适配和验证，不会复用这里生成的内部对象。
    source_sha256 = adapter.adapt(image).metadata.sha256_hex
    vision_provider = FakeVisionProvider(
        scripted_results={source_sha256: vision_draft}
    )
    vision_store = RequestVisionInputStore()
    evidence_store = ConfirmedEvidenceStore()
    citation = make_citation()
    knowledge_handler = RecordingKnowledgeHandler(
        evidence_store=evidence_store,
        citation=citation,
    )

    registry = ToolRegistry()
    registry.register(
        definition=ANALYZE_ROBOT_IMAGE_TOOL_DEFINITION,
        handler=AnalyzeRobotImageToolHandler(
            provider=vision_provider,
            input_store=vision_store,
        ),
    )
    registry.register(
        definition=SEARCH_KNOWLEDGE_TOOL_DEFINITION,
        handler=knowledge_handler,
    )

    if include_telemetry:
        registry.register(
            definition=GET_ROBOT_TELEMETRY_TOOL_DEFINITION,
            handler=GetRobotTelemetryToolHandler(
                store=InMemoryRobotTelemetryStore(
                    records=[make_telemetry()]
                )
            ),
        )

    runner = AgentRunner(
        planner=planner,
        executor=ToolExecutor(registry=registry),
        max_steps=5,
        planner_timeout_seconds=1.0,
        max_consecutive_tool_failures=2,
    )

    service = AgentDiagnosisService(
        runner=runner,
        evidence_store=evidence_store,
        vision_input_adapter=adapter,
        vision_input_store=vision_store,
        planner_prompt_version=TEST_PLANNER_PROMPT_VERSION,
    )

    return service, vision_provider, knowledge_handler


@pytest.mark.asyncio
async def test_vision_observation_drives_follow_up_knowledge_search(
) -> None:
    """视觉观察后应检索知识证据，再形成有白名单引用的诊断。

    预期流程：Service登记图片；Runner调用视觉工具；第二轮Context
    包含NET红灯观察；Planner再调用知识工具；最终草稿引用真实Chunk。
    """

    image = make_vision_payload(
        analysis_goal="检查NET指示灯的可见颜色和亮灭状态"
    )
    planner = ScriptedPlanner(
        decisions=(
            call_tool(
                call_id="call_vision_001",
                tool_name="analyze_robot_image",
                arguments={
                    "image_ref": "image_001",
                    "analysis_goal": "检查NET指示灯的可见颜色和亮灭状态",
                },
            ),
            call_tool(
                call_id="call_search_001",
                tool_name="search_knowledge",
                arguments={
                    "query": "NET指示灯红色常亮对应的安全检查是什么",
                    "top_k": 3,
                },
            ),
            finish_completed(),
        )
    )
    vision_draft = VisionModelDraft(
        status="completed",
        image_quality="clear",
        observations=(
            VisionObservationItem(
                description="设备面板上的NET指示灯呈红色常亮",
                category="visible_condition",
                confidence="high",
                region="面板右上区域",
            ),
        ),
        visible_indicators=(
            VisionVisibleIndicator(
                label="NET",
                observed_state="红色常亮",
                confidence="high",
                region="面板右上区域",
            ),
        ),
    )
    service, vision_provider, knowledge_handler = make_service(
        planner=planner,
        image=image,
        vision_draft=vision_draft,
        include_telemetry=False,
    )

    response = await service.diagnose(
        make_request(
            image=image,
            symptom="读码器面板显示异常",
            log_excerpt="NET indicator state unknown",
            task_goal="结合图片和知识库确认安全检查",
        ),
        "request-multimodal-001",
    )

    assert response.execution.state == "completed"
    assert [step.tool_name for step in response.execution.steps] == [
        "analyze_robot_image",
        "search_knowledge",
    ]
    assert len(response.vision_observations) == 1
    assert response.vision_observations[0].image_ref == "image_001"
    assert response.vision_observations[0].observation.source == (
        "vision_model"
    )
    assert [item.chunk_id for item in response.diagnosis.evidence] == [
        TEST_CHUNK_ID
    ]
    assert vision_provider.call_count == 1
    assert knowledge_handler.queries == [
        "NET指示灯红色常亮对应的安全检查是什么"
    ]

    # 第二轮Planner必须能看到第一轮视觉工具的结构化观察，
    # 否则就不能称为“视觉观察驱动后续检索”。
    second_context_output = planner.contexts[1].interactions[0].result.output
    assert second_context_output is not None
    assert second_context_output["visible_indicators"][0][
        "observed_state"
    ] == "红色常亮"


@pytest.mark.asyncio
async def test_vision_observation_drives_telemetry_then_knowledge_search(
) -> None:
    """视觉状态后应读取模拟遥测，并用知识证据支撑最终建议。

    预期流程：视觉工具看到状态灯；遥测工具读取robot-001当前状态；
    知识工具提供规则依据；最终响应分别公开视觉、遥测和引用来源。
    """

    image = make_vision_payload(
        analysis_goal="检查机器人状态灯是否显示暂停状态",
        color=(230, 170, 20),
    )
    planner = ScriptedPlanner(
        decisions=(
            call_tool(
                call_id="call_vision_002",
                tool_name="analyze_robot_image",
                arguments={
                    "image_ref": "image_001",
                    "analysis_goal": "检查机器人状态灯是否显示暂停状态",
                },
            ),
            call_tool(
                call_id="call_telemetry_001",
                tool_name="get_robot_telemetry",
                arguments={"robot_id": "robot-001"},
            ),
            call_tool(
                call_id="call_search_002",
                tool_name="search_knowledge",
                arguments={
                    "query": "ERR-NET-4001且机器人暂停时如何安全恢复",
                    "top_k": 3,
                },
            ),
            finish_completed(),
        )
    )
    vision_draft = VisionModelDraft(
        status="completed",
        image_quality="clear",
        observations=(
            VisionObservationItem(
                description="机器人状态灯呈黄色常亮",
                category="visible_condition",
                confidence="high",
                region="车体顶部",
            ),
        ),
        visible_indicators=(
            VisionVisibleIndicator(
                label="STATUS",
                observed_state="黄色常亮",
                confidence="high",
                region="车体顶部",
            ),
        ),
    )
    service, vision_provider, knowledge_handler = make_service(
        planner=planner,
        image=image,
        vision_draft=vision_draft,
        include_telemetry=True,
    )

    response = await service.diagnose(
        make_request(
            image=image,
            symptom="网络恢复后机器人仍处于暂停状态",
            log_excerpt="ERR-NET-4001; network_connected=true",
            task_goal="核对可见状态、当前遥测和安全恢复条件",
        ),
        "request-multimodal-002",
    )

    assert response.execution.state == "completed"
    assert [step.tool_name for step in response.execution.steps] == [
        "analyze_robot_image",
        "get_robot_telemetry",
        "search_knowledge",
    ]
    assert len(response.vision_observations) == 1
    assert len(response.telemetry_observations) == 1
    # telemetry_observations中的元素本身就是遥测契约，
    # 响应没有再增加一层snapshot包装。
    assert response.telemetry_observations[0].source == (
        "simulated_memory"
    )
    assert response.telemetry_observations[0].operational_state == (
        "paused"
    )
    assert response.diagnosis.abstained is False
    assert len(response.diagnosis.evidence) == 1
    assert vision_provider.call_count == 1
    assert knowledge_handler.queries == [
        "ERR-NET-4001且机器人暂停时如何安全恢复"
    ]

    # 第三轮Context必须同时保留前两轮成功结果，
    # 证明Runner没有在工具之间丢失视觉或遥测观察。
    third_context = planner.contexts[2]
    assert len(third_context.interactions) == 2
    assert third_context.interactions[0].tool_call.tool_name == (
        "analyze_robot_image"
    )
    assert third_context.interactions[1].result.output[
        "operational_state"
    ] == "paused"


@pytest.mark.asyncio
async def test_unusable_image_finishes_with_safe_abstention(
) -> None:
    """图片不可用时应正常结束为拒答，不猜测也不调用其他工具。

    预期流程：视觉工具返回unusable；Planner读取失败原因后以
    insufficient_information结束；API响应保留视觉观察和缺失信息，
    但不包含原因、检查、引用、遥测或额外知识检索。
    """

    image = make_vision_payload(
        analysis_goal="检查模糊面板上的故障指示灯",
        color=(120, 120, 120),
    )
    planner = ScriptedPlanner(
        decisions=(
            call_tool(
                call_id="call_vision_003",
                tool_name="analyze_robot_image",
                arguments={
                    "image_ref": "image_001",
                    "analysis_goal": "检查模糊面板上的故障指示灯",
                },
            ),
            finish_abstained(),
        )
    )
    vision_draft = VisionModelDraft(
        status="unusable",
        image_quality="unusable",
        observations=(),
        visible_indicators=(),
        uncertain_items=(
            "图片整体严重模糊，无法辨认面板或指示灯",
        ),
        requires_human_check=True,
        human_check_reasons=(
            "需要补充对焦清晰且无遮挡的设备面板图片",
        ),
    )
    service, vision_provider, knowledge_handler = make_service(
        planner=planner,
        image=image,
        vision_draft=vision_draft,
        include_telemetry=True,
    )

    response = await service.diagnose(
        make_request(
            image=image,
            symptom="无法辨认机器人面板状态",
            log_excerpt="no structured fault code",
            task_goal="仅在图片足够清晰时形成诊断",
        ),
        "request-multimodal-003",
    )

    assert response.execution.state == "completed"
    assert response.execution.finish_reason == "insufficient_information"
    assert [step.tool_name for step in response.execution.steps] == [
        "analyze_robot_image"
    ]
    assert response.diagnosis.status == "abstained"
    assert response.diagnosis.abstained is True
    assert response.diagnosis.possible_causes == []
    assert response.diagnosis.next_checks == []
    assert response.diagnosis.evidence == []
    assert len(response.diagnosis.missing_information) == 1
    assert response.vision_observations[0].observation.status == (
        "unusable"
    )
    assert response.vision_observations[0].observation.requires_human_check is True
    assert response.telemetry_observations == []
    assert response.test_case_drafts == []
    assert vision_provider.call_count == 1
    assert knowledge_handler.queries == []

    # 第二轮Planner看到的是工具正常执行、但图片业务状态不可用。
    # 这与Executor报错或Runner中止不同，所以应安全finish而非aborted。
    visual_result = planner.contexts[1].interactions[0].result
    assert visual_result.status == "success"
    assert visual_result.output is not None
    assert visual_result.output["status"] == "unusable"
