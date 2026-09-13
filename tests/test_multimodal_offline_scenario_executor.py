"""Week 6 单场景多模态离线执行器的集成测试。

被测试模块：
app.services.multimodal_offline_scenario_executor。

正常场景预期调用链：

Gold 场景与离线 Fixture
→ MultimodalOfflineScenarioExecutor.execute()（当前模块）
→ 图片输入适配与请求级图片 Store
→ 请求级安全分类与最小工具策略
→ AgentDiagnosisService
→ AgentRunner + AgentProgressReducer
→ ToolExecutor
→ Fake Vision / Fake 知识工具
→ ConfirmedEvidenceStore 证据白名单
→ AgentDiagnosisResponse 与离线调用审计。

高风险场景预期调用链：

Gold 请求
→ MultimodalOfflineScenarioExecutor.execute()（当前模块）
→ 确定性安全分类
→ request_policy_finished
→ 不读取图片、不调用 Planner、不执行工具。

本文件不访问真实 LLM、Vision、Embedding、Chroma 或 HTTP 服务。
"""

import json
from pathlib import Path

import pytest

from app.schemas.agent_planning import (
    AgentPlannerDecision,
)
from app.schemas.multimodal_agent_evaluation import (
    MultimodalAgentEvaluationScenario,
)
from app.schemas.multimodal_offline_regression import (
    MultimodalOfflineScenarioFixture,
    OfflinePlannerTurn,
    OfflineToolOutcome,
    OfflineVisionOutcome,
)
from app.schemas.vision import (
    VisionModelDraft,
)
from app.services.evaluation import (
    load_multimodal_agent_evaluation_scenarios,
)
from app.services.multimodal_offline_scenario_executor import (
    MultimodalOfflineScenarioExecutor,
)
from app.services.vision_input import (
    VisionInputAdapter,
)


# 从当前测试文件向上一级取得项目根目录。
#
# 不使用当前工作目录，是因为 IDE 或测试工具可能从其他目录
# 启动 pytest；__file__ 对应的测试文件位置更稳定。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# 使用项目内真实的 30 条多模态 Gold 场景。
#
# 测试只读取该文件和其中引用的本地脱敏图片，
# 不使用任何线上服务。
SCENARIO_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "eval"
    / "multimodal_scenarios.jsonl"
)


# 正常场景使用 multimodal-agent-001 中的真实图片摘要。
#
# Fake Vision Provider 依据该摘要选择预设结果，
# 因而测试同时验证图片文件没有被静默替换。
NETWORK_PANEL_IMAGE_SHA256 = (
    "992426e7ba0bcb7683ed966081df37caf"
    "23475f7d91bc13fc93333b6a89d3367"
)


# 构造一条满足 KnowledgeCitation 契约的稳定测试引用。
#
# document_id 使用 64 位小写十六进制；chunk_id 复用该文档标识，
# 最终诊断草稿必须引用同一个 chunk_id，证据白名单才能放行。
TEST_DOCUMENT_ID = "a" * 64
TEST_CHUNK_ID = f"{TEST_DOCUMENT_ID}:000001"


def make_vision_input_adapter() -> VisionInputAdapter:
    """创建只用于本地测试的图片输入适配器。

    这些上限高于仓库中的小型脱敏样例图片，
    但仍会保留 Base64、MIME、图片尺寸和像素数校验。
    """

    return VisionInputAdapter(
        max_image_size_bytes=1_000_000,
        max_image_dimension_px=2_048,
        max_image_pixels=4_000_000,
    )


def load_scenario(
    scenario_id: str,
) -> MultimodalAgentEvaluationScenario:
    """从真实 Gold 文件取得指定场景。

    load_multimodal_agent_evaluation_scenarios() 会逐行解析 JSONL，
    并用 MultimodalAgentEvaluationScenario 对每条数据进行校验。
    """

    scenarios = (
        load_multimodal_agent_evaluation_scenarios(
            SCENARIO_DATA_PATH
        )
    )

    return next(
        scenario
        for scenario in scenarios
        if scenario.scenario_id == scenario_id
    )


def make_vision_draft() -> VisionModelDraft:
    """创建清晰网络面板对应的 Fake Vision 结构化结果。"""

    return VisionModelDraft(
        status="completed",
        image_quality="clear",
        observations=(
            {
                "description": (
                    "面板显示 ERR-NET-4001，"
                    "网络状态为 CONNECTED，"
                    "任务状态为 PAUSED"
                ),
                "category": "visible_text",
                "confidence": "high",
            },
        ),
        visible_indicators=(),
        uncertain_items=(),
        requires_human_check=False,
        human_check_reasons=(),
    )


def make_search_output() -> dict[str, object]:
    """创建通过正式 SearchKnowledgeToolOutput 校验的结果。"""

    return {
        "citations": [
            {
                "chunk_id": TEST_CHUNK_ID,
                "document_id": TEST_DOCUMENT_ID,
                "source_file": "仓储机器人故障说明.txt",
                "page_or_section": "section: ERR-NET-4001",
                "chunk_index": 1,
                "rank": 1,
                "similarity": 0.82,
                "rrf_score": 0.0327,
                "excerpt": (
                    "ERR-NET-4001 表示心跳超时。"
                    "通信恢复后必须完成状态核对，"
                    "再由调度系统下发恢复命令。"
                ),
            },
        ],
        "retrieval_ms": 5.0,
    }


def make_completed_final_message() -> str:
    """创建引用已确认 Chunk 的合法完成诊断 JSON。"""

    # json.dumps()负责生成合法 JSON 字符串，
    # 避免手写引号和转义符造成无效 Planner 最终响应。
    return json.dumps(
        {
            "status": "completed",
            "possible_causes": [
                {
                    "description": (
                        "网络恢复后旧任务仍需等待调度系统"
                        "完成状态核对并下发恢复命令"
                    ),
                    "evidence_chunk_ids": [
                        TEST_CHUNK_ID
                    ],
                }
            ],
            "next_checks": [],
            "risk_level": "medium",
            "missing_information": [],
            "abstained": False,
        },
        ensure_ascii=False,
    )


def make_completed_fixture(
    *,
    image_sha256: str = (
        NETWORK_PANEL_IMAGE_SHA256
    ),
) -> MultimodalOfflineScenarioFixture:
    """创建 Vision、知识检索和完成三轮离线脚本。

    Planner 每一轮只给出决定；真正的权限检查、输入输出校验、
    工具执行、进度归约和最终证据校验仍由生产主链负责。
    """

    return MultimodalOfflineScenarioFixture(
        scenario_id="multimodal-agent-001",
        planner_turns=(
            OfflinePlannerTurn(
                decision=AgentPlannerDecision(
                    decision="call_tool",
                    tool_call={
                        "call_id": "call_vision_001",
                        "tool_name": (
                            "analyze_robot_image"
                        ),
                        "arguments": {
                            "image_ref": "image_001",
                            "analysis_goal": (
                                "读取面板上的故障码、"
                                "网络状态和任务状态"
                            ),
                        },
                    },
                ),
            ),
            OfflinePlannerTurn(
                decision=AgentPlannerDecision(
                    decision="call_tool",
                    tool_call={
                        "call_id": "call_search_001",
                        "tool_name": "search_knowledge",
                        "arguments": {
                            "query": (
                                "ERR-NET-4001 网络恢复后"
                                "旧任务的恢复条件"
                            ),
                            "top_k": 3,
                        },
                    },
                ),
            ),
            OfflinePlannerTurn(
                decision=AgentPlannerDecision(
                    decision="finish",
                    finish_reason="task_completed",
                    final_message=(
                        make_completed_final_message()
                    ),
                ),
            ),
        ),
        vision_outcomes=(
            OfflineVisionOutcome(
                image_sha256=image_sha256,
                draft=make_vision_draft(),
            ),
        ),
        tool_outcomes=(
            OfflineToolOutcome(
                call_id="call_search_001",
                tool_name="search_knowledge",
                status="success",
                output=make_search_output(),
            ),
        ),
    )


@pytest.mark.asyncio
async def test_executor_runs_completed_vision_and_knowledge_flow(
) -> None:
    """正常场景应经过真实主链形成有白名单引用的完成响应。

    测试方法：

    1. 读取真实 multimodal-agent-001 和对应脱敏图片；
    2. 注入三轮 Fake Planner、一条 Fake Vision 观察和一条 Fake 引用；
    3. 调用执行器公开的 execute()；
    4. 检查工具顺序、完成状态、引用和 Fake 消费审计。

    预期结果：Vision 与知识检索各执行一次，最终诊断完成，
    TEST_CHUNK_ID 先进入 ConfirmedEvidenceStore 后才出现在公开响应中。
    """

    executor = MultimodalOfflineScenarioExecutor(
        project_root=PROJECT_ROOT,
        vision_input_adapter=(
            make_vision_input_adapter()
        ),
    )

    execution = await executor.execute(
        scenario=load_scenario(
            "multimodal-agent-001"
        ),
        fixture=make_completed_fixture(),
    )

    response = execution.response

    assert response.execution.state == "completed"
    assert (
        response.execution.termination_reason
        == "planner_finished"
    )
    assert (
        response.execution.finish_reason
        == "task_completed"
    )
    assert response.diagnosis.status == "completed"
    assert response.diagnosis.abstained is False

    # 公开执行轨迹必须保持真实工具调用顺序。
    assert tuple(
        step.tool_name
        for step in response.execution.steps
    ) == (
        "analyze_robot_image",
        "search_knowledge",
    )

    # 最终引用必须来自 Fake 检索结果写入的证据白名单。
    assert tuple(
        evidence.chunk_id
        for evidence in response.diagnosis.evidence
    ) == (TEST_CHUNK_ID,)

    assert len(
        response.vision_observations
    ) == 1
    assert execution.planner_call_count == 3
    assert execution.vision_call_count == 1

    search_audit = next(
        audit
        for audit in execution.tool_call_audits
        if audit.tool_name == "search_knowledge"
    )

    assert search_audit.planned_outcome_count == 1
    assert search_audit.actual_call_count == 1
    assert search_audit.fully_consumed is True
    assert execution.fixture_fully_consumed is True


@pytest.mark.asyncio
async def test_executor_stops_high_risk_request_before_images_and_planner(
) -> None:
    """高风险控制请求应在读图、规划和工具执行前安全结束。

    测试方法：向执行器传入真实 multimodal-agent-030，
    但 Fixture 不提供任何 Planner、Vision 或工具结果。

    预期结果：确定性安全分类直接生成 human_review_required；
    Planner、Vision 和全部工具调用次数都保持为零，且空 Fixture
    被视为完整消费。这证明拒绝结果不是 Fake Planner 伪造的。
    """

    executor = MultimodalOfflineScenarioExecutor(
        project_root=PROJECT_ROOT,
        vision_input_adapter=(
            make_vision_input_adapter()
        ),
    )

    execution = await executor.execute(
        scenario=load_scenario(
            "multimodal-agent-030"
        ),
        fixture=(
            MultimodalOfflineScenarioFixture(
                scenario_id=(
                    "multimodal-agent-030"
                )
            )
        ),
    )

    response = execution.response

    assert response.execution.state == "completed"
    assert (
        response.execution.termination_reason
        == "request_policy_finished"
    )
    assert (
        response.execution.finish_reason
        == "human_review_required"
    )
    assert response.execution.step_count == 0
    assert response.diagnosis.status == "abstained"
    assert response.diagnosis.abstained is True
    assert execution.planner_call_count == 0
    assert execution.vision_call_count == 0
    assert all(
        audit.actual_call_count == 0
        for audit in execution.tool_call_audits
    )
    assert execution.fixture_fully_consumed is True


@pytest.mark.asyncio
async def test_executor_rejects_scenario_and_fixture_id_mismatch(
) -> None:
    """执行器必须拒绝把一条场景的行为脚本用于另一条场景。

    预期结果：在图片读取、Planner 和工具创建之前抛出 ValueError，
    防止评测把错误 Fixture 的结果记到另一个 scenario_id 名下。
    """

    executor = MultimodalOfflineScenarioExecutor(
        project_root=PROJECT_ROOT,
        vision_input_adapter=(
            make_vision_input_adapter()
        ),
    )

    with pytest.raises(
        ValueError,
        match="scenario_id必须一致",
    ):
        await executor.execute(
            scenario=load_scenario(
                "multimodal-agent-001"
            ),
            fixture=(
                MultimodalOfflineScenarioFixture(
                    scenario_id=(
                        "multimodal-agent-030"
                    )
                )
            ),
        )


@pytest.mark.asyncio
async def test_executor_rejects_vision_hash_outside_scenario(
) -> None:
    """Fixture 不能为 Gold 场景以外的图片伪造视觉观察。

    测试方法：保留合法三轮 Planner 与工具输出，只把 Vision
    结果索引替换为不属于场景的 SHA-256。

    预期结果：执行器在读取图片和启动 Agent 之前抛出 ValueError，
    防止错误图片的视觉结论混入当前场景。
    """

    executor = MultimodalOfflineScenarioExecutor(
        project_root=PROJECT_ROOT,
        vision_input_adapter=(
            make_vision_input_adapter()
        ),
    )

    with pytest.raises(
        ValueError,
        match="不属于当前场景的图片SHA-256",
    ):
        await executor.execute(
            scenario=load_scenario(
                "multimodal-agent-001"
            ),
            fixture=make_completed_fixture(
                image_sha256="f" * 64
            ),
        )
