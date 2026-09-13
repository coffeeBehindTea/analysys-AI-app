"""AgentRunner与AgentProgressReducer集成的离线测试。

本文件验证：

1. Runner只接受与请求级工具范围一致的初始进度；
2. 工具成功后，真实来源和证据会进入最终进度；
3. Planner提前声称完成时，Reducer会修正业务状态；
4. 重复工具调用会在Executor前被阻止；
5. 当前进度范围外的工具不会被执行；
6. Planner运行失败时，已有进度会转换成failed终态。

测试使用Fake Planner、Fake Handler、内存Registry和真实
ToolExecutor，不调用LLM、Embedding、Chroma或机器人设备。
"""

from copy import (
    deepcopy,
)

import pytest

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from app.agent.executor import (
    ToolExecutor,
)
from app.agent.progress_reducer import (
    AgentProgressReducer,
)
from app.agent.registry import (
    ToolRegistry,
)
from app.agent.runner import (
    AgentRunner,
)
from app.schemas.agent import (
    ToolCall,
    ToolDefinition,
)
from app.schemas.agent_planning import (
    AgentPlannerDecision,
    AgentPlanningContext,
)
from app.schemas.agent_tool_policy import (
    AgentToolPolicyDecision,
)
from app.schemas.agent_tools import (
    SearchKnowledgeToolOutput,
)
from app.schemas.knowledge_query import (
    KnowledgeCitation,
)


# 固定文档和Chunk ID用于检查Reducer是否从工具输出中
# 提取了真实来源，而不是使用Planner生成的自然语言。
TEST_DOCUMENT_ID = "f" * 64
TEST_CHUNK_ID = TEST_DOCUMENT_ID + ":000007"


class SearchInput(BaseModel):
    """集成测试使用的最小知识检索输入契约。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    query: str = Field(
        min_length=1,
        max_length=500,
    )


class SuccessfulKnowledgeHandler:
    """记录调用并返回一条固定、合法知识引用。"""

    def __init__(self) -> None:
        self.call_count = 0

    async def __call__(
        self,
        tool_input: SearchInput,
    ) -> SearchKnowledgeToolOutput:
        """模拟一次成功知识检索，不访问外部服务。"""

        # 显式读取字段，确认Executor传入的是已经通过
        # SearchInput校验的Pydantic对象。
        assert tool_input.query
        self.call_count += 1

        return SearchKnowledgeToolOutput(
            citations=(
                KnowledgeCitation(
                    chunk_id=TEST_CHUNK_ID,
                    document_id=(
                        TEST_DOCUMENT_ID
                    ),
                    source_file=(
                        "robot-progress-demo.txt"
                    ),
                    page_or_section=(
                        "section: progress-demo"
                    ),
                    chunk_index=7,
                    rank=1,
                    similarity=0.82,
                    rrf_score=0.032,
                    excerpt=(
                        "固定离线证据要求人工确认后恢复。"
                    ),
                ),
            ),
            retrieval_ms=4.5,
        )


class EmptyKnowledgeHandler:
    """记录调用并用None模拟没有找到可用知识。"""

    def __init__(self) -> None:
        self.call_count = 0

    async def __call__(
        self,
        tool_input: SearchInput,
    ) -> None:
        """返回None，让Executor产生empty_result。"""

        assert tool_input.query
        self.call_count += 1
        return None


class ScriptedProgressPlanner:
    """按固定顺序返回决定并保存每轮工具Schema的Fake。"""

    def __init__(
        self,
        *,
        decisions: tuple[
            AgentPlannerDecision,
            ...,
        ],
    ) -> None:
        self._decisions = decisions
        self.contexts: list[
            AgentPlanningContext
        ] = []
        self.tool_schemas_seen: list[
            tuple[dict[str, object], ...]
        ] = []

    async def plan(
        self,
        *,
        context: AgentPlanningContext,
        tool_schemas: tuple[
            dict[str, object],
            ...,
        ],
    ) -> AgentPlannerDecision:
        """返回与当前调用轮次对应的预设决定。"""

        self.contexts.append(
            context.model_copy(deep=True)
        )
        self.tool_schemas_seen.append(
            deepcopy(tool_schemas)
        )

        index = len(self.contexts) - 1
        return self._decisions[index]


class FailingPlanner:
    """每次规划都抛出固定异常的异步Fake Planner。"""

    async def plan(
        self,
        *,
        context: AgentPlanningContext,
        tool_schemas: tuple[
            dict[str, object],
            ...,
        ],
    ) -> AgentPlannerDecision:
        """模拟Planner服务在第一轮发生不可恢复错误。"""

        del context, tool_schemas
        raise RuntimeError(
            "private-upstream-detail"
        )


def make_policy(
) -> AgentToolPolicyDecision:
    """创建只需要知识证据的请求级工具策略。"""

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


def make_initial_progress():
    """根据固定策略创建一份新的collecting进度。"""

    return (
        AgentProgressReducer()
        .create_initial_progress(
            make_policy()
        )
    )


def make_tool_decision(
    *,
    call_id: str,
    query: str,
    tool_name: str = "search_knowledge",
) -> AgentPlannerDecision:
    """创建一条经过AgentPlannerDecision校验的工具决定。"""

    return AgentPlannerDecision(
        decision="call_tool",
        tool_call=ToolCall(
            call_id=call_id,
            tool_name=tool_name,
            arguments={
                "query": query,
            },
        ),
    )


def make_finish_decision(
    *,
    finish_reason: str = "task_completed",
) -> AgentPlannerDecision:
    """创建一条固定Planner结束决定。"""

    return AgentPlannerDecision.model_validate({
        "decision": "finish",
        "finish_reason": finish_reason,
        "final_message": (
            "固定离线Agent诊断草稿"
        ),
    })


def make_executor(
    handler: object,
) -> ToolExecutor:
    """用指定Fake Handler构造真实白名单Executor。"""

    registry = ToolRegistry()
    registry.register(
        definition=ToolDefinition(
            name="search_knowledge",
            description=(
                "从离线测试知识中检索固定证据。"
            ),
            input_model=SearchInput,
            output_model=(
                SearchKnowledgeToolOutput
            ),
            risk_level="low",
            read_only=True,
            timeout_seconds=1.0,
        ),
        handler=handler,
    )

    return ToolExecutor(registry=registry)


@pytest.mark.asyncio
async def test_runner_records_success_and_finishes_progress() -> None:
    """成功工具结果应形成completed最终进度。

    被测试模块是AgentRunner.run()的完整进度路径。Planner先
    请求知识工具，再返回task_completed。预期Executor执行一
    次，Reducer确认文档和Chunk，第二轮不再暴露工具，最后
    AgentRunResult同时保留真实交互与completed进度。
    """

    handler = SuccessfulKnowledgeHandler()
    planner = ScriptedProgressPlanner(
        decisions=(
            make_tool_decision(
                call_id="call_success",
                query="ERR-NET-4001",
            ),
            make_finish_decision(),
        ),
    )
    runner = AgentRunner(
        planner=planner,
        executor=make_executor(handler),
    )

    result = await runner.run(
        "核对网络恢复规则",
        allowed_tool_names=frozenset({
            "search_knowledge",
        }),
        initial_progress=(
            make_initial_progress()
        ),
    )

    assert handler.call_count == 1
    assert result.state == "completed"
    assert result.progress is not None
    assert result.progress.state == "completed"
    assert result.progress.confirmed_source_ids == (
        "knowledge:" + TEST_DOCUMENT_ID,
    )
    assert result.progress.covered_evidence_ids == (
        TEST_CHUNK_ID,
    )
    assert len(result.interactions) == 1

    first_round_names = {
        schema["function"]["name"]
        for schema
        in planner.tool_schemas_seen[0]
    }
    assert first_round_names == {
        "search_knowledge"
    }
    assert planner.tool_schemas_seen[1] == ()


@pytest.mark.asyncio
async def test_runner_corrects_premature_completed_claim() -> None:
    """Planner未执行工具就声称完成时，业务进度应为拒答。

    被测试模块是Runner调用finalize_planner_decision()的分支。
    外层state仍表示循环正常结束，但权威progress必须是
    abstained，并说明知识证据尚未取得。
    """

    handler = SuccessfulKnowledgeHandler()
    planner = ScriptedProgressPlanner(
        decisions=(
            make_finish_decision(),
        ),
    )
    runner = AgentRunner(
        planner=planner,
        executor=make_executor(handler),
    )

    result = await runner.run(
        "核对网络恢复规则",
        initial_progress=(
            make_initial_progress()
        ),
    )

    assert handler.call_count == 0
    assert result.state == "completed"
    assert result.finish_reason == "task_completed"
    assert result.progress is not None
    assert result.progress.state == "abstained"
    assert result.progress.missing_information == (
        "尚未取得任务所需的知识库证据",
    )


@pytest.mark.asyncio
async def test_runner_blocks_duplicate_query_before_executor() -> None:
    """相同工具参数的第二次调用必须在执行前被阻止。

    被测试模块是Runner与Reducer的重复调用边界。第一次检索
    返回empty并被记录；Planner更换call_id但重复同一query。
    预期Handler只执行一次，Runner以duplicate_tool_call中止，
    最终failed进度保留第一次工具记录。
    """

    handler = EmptyKnowledgeHandler()
    planner = ScriptedProgressPlanner(
        decisions=(
            make_tool_decision(
                call_id="call_empty_1",
                query="unknown-code",
            ),
            make_tool_decision(
                call_id="call_empty_2",
                query="unknown-code",
            ),
        ),
    )
    runner = AgentRunner(
        planner=planner,
        executor=make_executor(handler),
    )

    result = await runner.run(
        "检索未知故障码",
        initial_progress=(
            make_initial_progress()
        ),
    )

    assert handler.call_count == 1
    assert result.state == "aborted"
    assert result.termination_reason == (
        "duplicate_tool_call"
    )
    assert len(result.interactions) == 1
    assert result.progress is not None
    assert result.progress.state == "failed"
    assert len(result.progress.tool_records) == 1


@pytest.mark.asyncio
async def test_runner_blocks_tool_outside_progress_scope() -> None:
    """Planner请求非必要工具时不能到达Executor。

    被测试模块是Runner执行前调用validate_tool_call()的路径。
    初始进度只允许search_knowledge，Fake Planner故意请求
    get_robot_telemetry。预期工具调用数为零，并返回稳定的
    tool_policy_violation和failed进度。
    """

    handler = SuccessfulKnowledgeHandler()
    planner = ScriptedProgressPlanner(
        decisions=(
            make_tool_decision(
                call_id="call_out_of_scope",
                query="ignored",
                tool_name=(
                    "get_robot_telemetry"
                ),
            ),
        ),
    )
    runner = AgentRunner(
        planner=planner,
        executor=make_executor(handler),
    )

    result = await runner.run(
        "只允许检索知识",
        initial_progress=(
            make_initial_progress()
        ),
    )

    assert handler.call_count == 0
    assert result.termination_reason == (
        "tool_policy_violation"
    )
    assert result.interactions == ()
    assert result.progress is not None
    assert result.progress.state == "failed"


@pytest.mark.asyncio
async def test_runner_failure_converts_active_progress_to_failed() -> None:
    """Planner异常应保留为结构化failed进度。

    被测试模块是Runner的普通Planner异常分支和
    _build_aborted_result()。预期原始异常文本不会进入结果，
    Runner返回planner_error，Progress返回runtime_failure。
    """

    handler = SuccessfulKnowledgeHandler()
    runner = AgentRunner(
        planner=FailingPlanner(),
        executor=make_executor(handler),
    )

    result = await runner.run(
        "核对网络恢复规则",
        initial_progress=(
            make_initial_progress()
        ),
    )

    assert result.state == "aborted"
    assert result.termination_reason == (
        "planner_error"
    )
    assert result.progress is not None
    assert result.progress.state == "failed"
    assert result.progress.stop_reason == (
        "runtime_failure"
    )
    assert "private-upstream-detail" not in (
        result.model_dump_json()
    )


@pytest.mark.asyncio
async def test_runner_rejects_scope_progress_mismatch() -> None:
    """调用参数不能让工具范围与初始进度表达不同权限。

    被测试模块是AgentRunner.run()入口校验。输入中的Progress
    允许知识检索，但显式allowed_tool_names为空。预期在调用
    Planner和Executor前抛出ValueError，避免两套权限源分叉。
    """

    handler = SuccessfulKnowledgeHandler()
    planner = ScriptedProgressPlanner(
        decisions=(
            make_finish_decision(),
        ),
    )
    runner = AgentRunner(
        planner=planner,
        executor=make_executor(handler),
    )

    with pytest.raises(
        ValueError,
        match="工具范围一致",
    ):
        await runner.run(
            "核对网络恢复规则",
            allowed_tool_names=frozenset(),
            initial_progress=(
                make_initial_progress()
            ),
        )

    assert planner.contexts == []
    assert handler.call_count == 0


@pytest.mark.asyncio
async def test_runner_blocks_reused_call_id_with_new_arguments() -> None:
    """相同call_id即使参数变化也属于无效重复请求。

    被测试模块是Reducer的已调用工具记录检查。第一次empty后
    Planner复用call_id并修改query；预期第二次仍在Executor前
    被阻止，避免一个调用标识对应两组不同参数。
    """

    handler = EmptyKnowledgeHandler()
    planner = ScriptedProgressPlanner(
        decisions=(
            make_tool_decision(
                call_id="call_reused",
                query="first-query",
            ),
            make_tool_decision(
                call_id="call_reused",
                query="second-query",
            ),
        ),
    )
    runner = AgentRunner(
        planner=planner,
        executor=make_executor(handler),
    )

    result = await runner.run(
        "检索未知故障码",
        initial_progress=(
            make_initial_progress()
        ),
    )

    assert handler.call_count == 1
    assert result.termination_reason == (
        "duplicate_tool_call"
    )
    assert result.progress is not None
    assert len(result.progress.tool_records) == 1
