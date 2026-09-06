"""模拟遥测工厂和Agent工具依赖装配的离线测试。

本文件验证应用如何创建模拟数据、缓存内存Store、注册工具并
构造ToolExecutor。所有外部知识库能力均使用Fake替代。
"""

from unittest.mock import (
    MagicMock,
)

from datetime import (
    datetime,
    timedelta,
    timezone,
)

import pytest

from pydantic import (
    ValidationError,
)

from app.agent.demo_telemetry import (
    build_demo_robot_telemetry_records,
    build_demo_robot_telemetry_store,
)
from app.agent.executor import (
    ToolExecutor,
)
from app.agent.evidence_store import (
    ConfirmedEvidenceStore,
)
from app.agent.openai_planner import (
    AGENT_PLANNER_PROMPT_VERSION,
    OpenAICompatibleAgentPlanner,
)
from app.agent.registry import (
    ToolRegistry,
)
from app.agent.runner import (
    AgentRunner,
)
from app.agent.vision_input_store import (
    RequestVisionInputStore,
)
from app.agent.tools.current_time import (
    SystemUtcClock,
)
from app.config import (
    Settings,
)
from app.dependencies import (
    get_agent_diagnosis_service,
    get_agent_planner,
    get_agent_runner,
    get_agent_tool_executor,
    get_agent_tool_registry,
    get_confirmed_evidence_store,
    get_request_vision_input_store,
    get_robot_telemetry_store,
    get_system_utc_clock,
    get_vision_input_adapter,
)
from app.schemas.agent import (
    ToolCall,
)
from app.services.agent_diagnosis_service import (
    AgentDiagnosisService,
)
from app.services.gated_hybrid_retrieval import (
    GatedHybridRetrievalResult,
)
from app.services.fake_vision_provider import (
    FakeVisionProvider,
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


# 工厂纯函数测试使用固定时间，确保断言可重复。
FIXED_OBSERVED_AT = datetime(
    2026,
    8,
    24,
    8,
    0,
    0,
    tzinfo=timezone.utc,
)


class FakeGatedRetrievalProvider:
    """记录参数并固定返回门控拒绝的Fake检索器。

    本Fake只有retrieve()，没有answer_query()，
    用于锁定Agent搜索工具不能重新依赖完整RAG问答服务。
    """

    def __init__(
        self,
    ) -> None:
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
        """记录检索参数，不调用Embedding、Chroma或LLM。"""

        self.calls.append(
            (query, top_k, retrieval_scope)
        )

        # 返回一个结构完整但没有候选的门控拒绝结果。
        # Handler应把它转换成None，Executor再转换成empty。
        return GatedHybridRetrievalResult(
            query_rewrite=(
                RewrittenRetrievalQuery(
                    original_query=query,
                    normalized_query=query,
                    rewritten_query=query,
                    added_terms=(),
                    applied_rule_ids=(),
                    rewrite_version=(
                        "deterministic-v1"
                    ),
                )
            ),
            retrieved_chunks=(),
            decision=(
                HybridEvidenceGateDecision(
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
            ),
        )


def build_test_registry(
) -> tuple[
    ToolRegistry,
    FakeGatedRetrievalProvider,
]:
    """使用固定Store和Fake门控检索器调用依赖工厂。"""

    telemetry_store = (
        build_demo_robot_telemetry_store(
            observed_at=FIXED_OBSERVED_AT
        )
    )
    retrieval_provider = (
        FakeGatedRetrievalProvider()
    )

    registry = get_agent_tool_registry(
        retrieval_provider,
        telemetry_store,
        ConfirmedEvidenceStore(),
        SystemUtcClock(),
        FakeVisionProvider(
            scripted_results={},
        ),
        RequestVisionInputStore(),
    )

    return registry, retrieval_provider


def test_demo_record_factory_requires_datetime(
) -> None:
    """工厂不能接受字符串冒充观测时间。"""

    with pytest.raises(
        TypeError,
        match="observed_at必须是datetime",
    ):
        build_demo_robot_telemetry_records(
            observed_at=(
                "2026-08-24T08:00:00Z"
            )
        )


def test_demo_record_factory_rejects_naive_datetime(
) -> None:
    """即使类型是datetime，没有时区也不能创建快照。"""

    with pytest.raises(
        ValidationError,
        match="observed_at必须包含时区",
    ):
        build_demo_robot_telemetry_records(
            observed_at=datetime(
                2026,
                8,
                24,
                8,
                0,
                0,
            )
        )


def test_demo_records_have_unique_ids_and_shared_observation_time(
) -> None:
    """三条快照应可精确区分并属于同一次模拟采样。"""

    records = (
        build_demo_robot_telemetry_records(
            observed_at=FIXED_OBSERVED_AT
        )
    )

    assert tuple(
        record.robot_id
        for record in records
    ) == (
        "robot-001",
        "robot-002",
        "robot-003",
    )
    assert len({
        record.robot_id
        for record in records
    }) == 3
    assert all(
        record.observed_at
        == FIXED_OBSERVED_AT
        for record in records
    )
    assert all(
        record.source
        == "simulated_memory"
        for record in records
    )


def test_demo_records_preserve_documented_scenario_semantics(
) -> None:
    """固定场景应支持故障、充电和正常任务三类演示。"""

    records = {
        record.robot_id: record
        for record in (
            build_demo_robot_telemetry_records(
                observed_at=FIXED_OBSERVED_AT
            )
        )
    }

    network_case = records["robot-001"]
    assert network_case.operational_state == (
        "paused"
    )
    assert network_case.network_connected is True
    assert network_case.active_fault_codes == (
        "ERR-NET-4001",
    )
    assert network_case.speed_mps == 0.0

    charging_case = records["robot-002"]
    assert charging_case.operational_state == (
        "charging"
    )
    assert charging_case.current_task_id is None
    assert charging_case.active_fault_codes == ()

    normal_case = records["robot-003"]
    assert normal_case.operational_state == (
        "executing_task"
    )
    assert normal_case.current_task_id == (
        "task-demo-3001"
    )
    assert normal_case.active_fault_codes == ()


def test_demo_store_factory_indexes_all_records(
) -> None:
    """Store工厂应把三条记录转换成精确robot_id索引。"""

    store = build_demo_robot_telemetry_store(
        observed_at=FIXED_OBSERVED_AT
    )

    assert len(store) == 3
    assert store.get("robot-001") is not None
    assert store.get("robot-002") is not None
    assert store.get("robot-003") is not None
    assert store.get("robot-999") is None


def test_telemetry_dependency_reuses_one_process_local_store(
) -> None:
    """lru_cache应复用同一个Store及同一批观测时间。"""

    # 测试开始前移除其他测试可能留下的缓存。
    get_robot_telemetry_store.cache_clear()

    try:
        first = get_robot_telemetry_store()
        second = get_robot_telemetry_store()

        assert first is second
        assert len(first) == 3

        observed_times = {
            first.get(robot_id).observed_at
            for robot_id in (
                "robot-001",
                "robot-002",
                "robot-003",
            )
        }

        assert len(observed_times) == 1
        observed_at = observed_times.pop()
        assert observed_at.tzinfo is not None
        assert observed_at.utcoffset() is not None
    finally:
        # 不把当前测试创建的Store缓存泄漏给其他测试。
        get_robot_telemetry_store.cache_clear()


def test_agent_registry_dependency_registers_exact_allowlist(
) -> None:
    """当前依赖装配只能暴露五个明确允许的工具。"""

    registry, retrieval_provider = (
        build_test_registry()
    )

    assert isinstance(registry, ToolRegistry)
    assert len(registry) == 5
    assert tuple(
        definition.name
        for definition
        in registry.list_definitions()
    ) == (
        "search_knowledge",
        "get_robot_telemetry",
        "draft_test_case",
        "get_current_time",
        "analyze_robot_image",
    )
    assert "search_knowledge" in registry
    assert "get_robot_telemetry" in registry
    assert "draft_test_case" in registry
    assert "get_current_time" in registry
    assert "analyze_robot_image" in registry
    assert "run_shell" not in registry
    assert retrieval_provider.calls == []


def test_registry_shares_request_evidence_store_between_tools(
) -> None:
    """检索工具和草案工具必须共享同一个请求级Store。"""

    # 显式创建当前请求使用的Store，方便通过对象身份
    # 验证依赖装配没有意外创建第二个空Store。
    evidence_store = ConfirmedEvidenceStore()
    telemetry_store = (
        build_demo_robot_telemetry_store(
            observed_at=FIXED_OBSERVED_AT
        )
    )
    retrieval_provider = (
        FakeGatedRetrievalProvider()
    )

    registry = get_agent_tool_registry(
        retrieval_provider,
        telemetry_store,
        evidence_store,
        SystemUtcClock(),
        FakeVisionProvider(
            scripted_results={},
        ),
        RequestVisionInputStore(),
    )

    search_tool = registry.get(
        "search_knowledge"
    )
    draft_tool = registry.get(
        "draft_test_case"
    )

    assert search_tool is not None
    assert draft_tool is not None

    # 这里有意检查Handler内部保存的对象身份。
    # 本测试关注的是依赖装配关系，而不是Handler业务逻辑；
    # 两者都必须持有传给工厂的同一个Store实例。
    assert getattr(
        search_tool.handler,
        "_evidence_store",
    ) is evidence_store
    assert getattr(
        draft_tool.handler,
        "_evidence_store",
    ) is evidence_store


def test_registry_binds_request_vision_dependencies(
) -> None:
    """视觉工具必须复用传入的Provider和请求级图片Store。"""

    # 显式创建依赖对象，随后用对象身份检查
    # get_agent_tool_registry()没有在内部替换它们。
    vision_provider = FakeVisionProvider(
        scripted_results={},
    )
    vision_input_store = (
        RequestVisionInputStore()
    )

    registry = get_agent_tool_registry(
        FakeGatedRetrievalProvider(),
        build_demo_robot_telemetry_store(
            observed_at=FIXED_OBSERVED_AT,
        ),
        ConfirmedEvidenceStore(),
        SystemUtcClock(),
        vision_provider,
        vision_input_store,
    )

    vision_tool = registry.get(
        "analyze_robot_image"
    )

    assert vision_tool is not None

    # 当前测试验证的是依赖装配关系，
    # 所以有意检查Handler保存的内部依赖身份。
    assert getattr(
        vision_tool.handler,
        "_provider",
    ) is vision_provider
    assert getattr(
        vision_tool.handler,
        "_input_store",
    ) is vision_input_store


def test_confirmed_evidence_dependency_is_request_scoped(
) -> None:
    """直接调用依赖两次必须得到两个互不共享的空Store。"""

    first = get_confirmed_evidence_store()
    second = get_confirmed_evidence_store()

    assert first is not second
    assert len(first) == 0
    assert len(second) == 0


def test_vision_input_dependency_is_request_scoped(
) -> None:
    """直接调用图片Store依赖两次必须得到两个空实例。"""

    first = get_request_vision_input_store()
    second = get_request_vision_input_store()

    assert isinstance(
        first,
        RequestVisionInputStore,
    )
    assert isinstance(
        second,
        RequestVisionInputStore,
    )
    assert first is not second
    assert len(first) == 0
    assert len(second) == 0


def test_system_clock_dependency_returns_fresh_utc_clock(
) -> None:
    """时间依赖应返回能读取带时区UTC时间的正式Clock。"""

    first = get_system_utc_clock()
    second = get_system_utc_clock()

    # 当前依赖没有使用lru_cache，因此直接调用两次会
    # 得到两个无状态Clock；它们读取的都是真实系统时间。
    assert isinstance(first, SystemUtcClock)
    assert isinstance(second, SystemUtcClock)
    assert first is not second

    observed = first.now_utc()
    assert observed.tzinfo is not None
    assert observed.utcoffset() == (
        timedelta(0)
    )


def test_agent_registry_builds_five_openai_tool_schemas(
) -> None:
    """规划模型只能看到注册表生成的五项Tool Schema。"""

    registry, _ = build_test_registry()

    schemas = (
        registry.build_openai_tool_schemas()
    )

    assert [
        schema["function"]["name"]
        for schema in schemas
    ] == [
        "search_knowledge",
        "get_robot_telemetry",
        "draft_test_case",
        "get_current_time",
        "analyze_robot_image",
    ]


def test_agent_executor_dependency_wraps_supplied_registry(
) -> None:
    """执行器依赖应返回可以使用该注册表的ToolExecutor。"""

    registry, _ = build_test_registry()

    executor = get_agent_tool_executor(
        registry
    )

    assert isinstance(executor, ToolExecutor)


@pytest.mark.asyncio
async def test_dependency_assembled_executor_reads_demo_telemetry(
) -> None:
    """装配后的执行器应真正读取工厂生成的robot-001。"""

    registry, _ = build_test_registry()
    executor = get_agent_tool_executor(
        registry
    )

    result = await executor.execute(
        ToolCall(
            call_id="call_dependency_001",
            tool_name="get_robot_telemetry",
            arguments={
                "robot_id": "robot-001"
            },
        )
    )

    assert result.status == "success"
    assert result.output is not None
    assert result.output["robot_id"] == (
        "robot-001"
    )
    assert result.output[
        "operational_state"
    ] == "paused"
    assert result.output[
        "active_fault_codes"
    ] == ["ERR-NET-4001"]
    assert result.output["source"] == (
        "simulated_memory"
    )


@pytest.mark.asyncio
async def test_dependency_assembled_executor_reads_current_time(
) -> None:
    """装配后的时间工具应返回系统UTC时间而非模拟遥测。"""

    registry, _ = build_test_registry()
    executor = get_agent_tool_executor(
        registry
    )

    result = await executor.execute(
        ToolCall(
            call_id="call_dependency_time_001",
            tool_name="get_current_time",
            arguments={},
        )
    )

    assert result.status == "success"
    assert result.output is not None
    assert result.output["timezone"] == "UTC"
    assert result.output["source"] == (
        "system_clock"
    )

    # Executor使用model_dump(mode="json")序列化datetime，
    # 因此这里重新解析字符串并确认UTC偏移为零。
    observed = datetime.fromisoformat(
        result.output["current_time"]
        .replace("Z", "+00:00")
    )
    assert observed.utcoffset() == (
        timedelta(0)
    )


@pytest.mark.asyncio
async def test_dependency_assembled_search_tool_uses_fake_service(
) -> None:
    """知识库工具应使用依赖注入对象而不是自行联网。"""

    registry, retrieval_provider = (
        build_test_registry()
    )
    executor = get_agent_tool_executor(
        registry
    )

    result = await executor.execute(
        ToolCall(
            call_id="call_dependency_002",
            tool_name="search_knowledge",
            arguments={
                "query": "ERR-NET-4001如何恢复？",
                "top_k": 3,
            },
        )
    )

    # Fake返回确定性门控拒绝，因此Handler返回None，
    # Executor再把它转换成稳定empty结果。
    assert result.status == "empty"
    assert result.error_code == "empty_result"
    assert retrieval_provider.calls == [
        (
            "ERR-NET-4001如何恢复？",
            3,
            None,
        )
    ]


def test_agent_planner_dependency_uses_injected_client_and_model(
) -> None:
    """Planner依赖应完成装配且不能在构造阶段调用LLM。"""

    # MagicMock模拟FastAPI的get_llm_client依赖
    # 已经创建好的AsyncOpenAI对象。
    mock_client = MagicMock()

    # _env_file=None防止本测试读取开发者本机.env。
    # 显式模型名包含首尾空格，用于验证Provider会清理它。
    settings = Settings(
        _env_file=None,
        llm_model="  planner-test-model  ",
    )

    planner = get_agent_planner(
        mock_client,
        settings,
    )

    assert isinstance(
        planner,
        OpenAICompatibleAgentPlanner,
    )

    # 当前测试关注依赖装配关系，
    # 因此有意检查Provider保存的内部对象身份。
    assert getattr(
        planner,
        "_client",
    ) is mock_client
    assert getattr(
        planner,
        "_model",
    ) == "planner-test-model"
    assert planner.prompt_version == (
        AGENT_PLANNER_PROMPT_VERSION
    )

    # 创建Provider只保存依赖；
    # 真正的SDK调用只能发生在await planner.plan()。
    mock_client.chat.completions.create.assert_not_called()


def test_agent_runner_dependency_connects_real_planner_and_executor(
) -> None:
    """Runner依赖应连接真实Planner和当前请求Executor。"""

    registry, _ = build_test_registry()
    executor = get_agent_tool_executor(
        registry
    )
    planner = get_agent_planner(
        MagicMock(),
        Settings(
            _env_file=None,
            llm_model="planner-test-model",
        ),
    )

    runner = get_agent_runner(
        planner,
        executor,
    )

    assert isinstance(
        runner,
        AgentRunner,
    )

    # 依赖装配不能偷偷创建第二个Planner或Executor。
    assert getattr(
        runner,
        "_planner",
    ) is planner
    assert getattr(
        runner,
        "_executor",
    ) is executor

    # Runner构造时从Executor取得可信工具Schema快照。
    tool_schemas = getattr(
        runner,
        "_tool_schemas",
    )
    assert tuple(
        schema["function"]["name"]
        for schema in tool_schemas
    ) == (
        "search_knowledge",
        "get_robot_telemetry",
        "draft_test_case",
        "get_current_time",
        "analyze_robot_image",
    )

    # 依赖装配没有从HTTP请求读取这些安全边界；
    # 当前继续使用Runner构造器中经过测试的默认值。
    assert getattr(
        runner,
        "_max_steps",
    ) == 5
    assert getattr(
        runner,
        "_planner_timeout_seconds",
    ) == 30.0
    assert getattr(
        runner,
        "_max_consecutive_tool_failures",
    ) == 2


def test_agent_diagnosis_service_dependency_connects_request_components(
) -> None:
    """Service依赖应复用当前Runner、证据Store和图片Store。"""

    # 显式创建请求级Store。
    #
    # 依赖函数不能再创建第二个空Store，否则search_knowledge
    # 记录的证据不会出现在最终报告构建器的白名单中。
    evidence_store = ConfirmedEvidenceStore()
    vision_input_store = (
        RequestVisionInputStore()
    )

    settings = Settings(
        _env_file=None,
        llm_model="planner-test-model",
    )
    vision_input_adapter = (
        get_vision_input_adapter(
            settings
        )
    )

    registry = get_agent_tool_registry(
        FakeGatedRetrievalProvider(),
        build_demo_robot_telemetry_store(
            observed_at=FIXED_OBSERVED_AT,
        ),
        evidence_store,
        SystemUtcClock(),
        FakeVisionProvider(
            scripted_results={},
        ),
        vision_input_store,
    )
    executor = get_agent_tool_executor(registry)

    # Planner使用MagicMock客户端，因此构造阶段不会联网。
    planner = get_agent_planner(
        MagicMock(),
        settings,
    )
    runner = get_agent_runner(
        planner,
        executor,
    )

    service = get_agent_diagnosis_service(
        runner,
        planner,
        evidence_store,
        vision_input_adapter,
        vision_input_store,
    )

    assert isinstance(
        service,
        AgentDiagnosisService,
    )

    # 当前测试验证依赖装配的对象身份，
    # 因此有意检查Service保存的内部依赖。
    assert getattr(
        service,
        "_runner",
    ) is runner
    assert getattr(
        service,
        "_evidence_store",
    ) is evidence_store
    assert getattr(
        service,
        "_vision_input_adapter",
    ) is vision_input_adapter
    assert getattr(
        service,
        "_vision_input_store",
    ) is vision_input_store
    assert getattr(
        service,
        "_planner_prompt_version",
    ) == planner.prompt_version
