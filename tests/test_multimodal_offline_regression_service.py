"""Week 6 多模态 Agent 批量离线回归服务测试。

被测试模块：

app.services.multimodal_offline_regression_service

完整集成测试使用项目内正式的 30 条 Gold、30 条
Fixture、单场景离线执行器和第五周评分器。所有外部
Planner、Vision和工具结果都由 Fixture 提供，不访问网络。
"""

from pathlib import Path

import pytest

from app.schemas.multimodal_agent_evaluation import (
    MultimodalAgentEvaluationScenario,
)
from app.schemas.multimodal_offline_regression import (
    MultimodalOfflineScenarioExecution,
    MultimodalOfflineScenarioFixture,
)
from app.services.evaluation import (
    EvaluationDataError,
    load_multimodal_agent_evaluation_scenarios,
)
from app.services.multimodal_offline_fixture_loader import (
    MultimodalOfflineRegressionCase,
    load_multimodal_offline_fixtures,
    pair_multimodal_offline_regression_cases,
)
from app.services.multimodal_offline_regression_service import (
    MultimodalOfflineRegressionService,
)
from app.services.multimodal_offline_scenario_executor import (
    MultimodalOfflineScenarioExecutor,
)
from app.services.vision_input import (
    VisionInputAdapter,
)


# __file__是当前测试文件的绝对位置。
# parents[1]返回项目根目录，因此测试不依赖
# pytest启动时使用的当前工作目录。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# 正式的30条第五周多模态Gold场景。
SCENARIO_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "eval"
    / "multimodal_scenarios.jsonl"
)


# 第六周生成的固定离线Fixture。
FIXTURE_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "eval"
    / "multimodal_offline_fixtures.jsonl"
)


# 报告契约使用项目相对POSIX路径，
# 不使用Windows绝对路径。
SCENARIO_SOURCE = (
    "data/eval/multimodal_scenarios.jsonl"
)
FIXTURE_SOURCE = (
    "data/eval/multimodal_offline_fixtures.jsonl"
)


# 当前固定策略预期保留的四条已知缺口。
# 该常量只验证离线结果是否可重复，
# 不会修改Gold或生产评分规则。
EXPECTED_FAILED_SCENARIO_IDS = (
    "multimodal-agent-008",
    "multimodal-agent-012",
    "multimodal-agent-019",
    "multimodal-agent-029",
)


class NeverCalledOfflineExecutor:
    """用于输入边界测试的单场景Fake Executor。

    它具有与OfflineScenarioExecutor Protocol相同的
    async execute()方法，但一旦被调用就立即使测试失败。

    这能证明非法场景集是在进入执行循环之前被拒绝的。
    """

    def __init__(self) -> None:
        """初始调用计数为零。"""

        self.call_count = 0

    async def execute(
        self,
        *,
        scenario: MultimodalAgentEvaluationScenario,
        fixture: MultimodalOfflineScenarioFixture,
    ) -> MultimodalOfflineScenarioExecution:
        """记录意外调用并立即终止测试。"""

        self.call_count += 1

        raise AssertionError(
            "非法cases不应进入单场景执行器"
        )


def load_real_cases(
) -> tuple[
    MultimodalOfflineRegressionCase,
    ...,
]:
    """加载并配对项目内的正式Gold和Fixture。"""

    scenarios = (
        load_multimodal_agent_evaluation_scenarios(
            SCENARIO_DATA_PATH
        )
    )
    fixtures = load_multimodal_offline_fixtures(
        FIXTURE_DATA_PATH
    )

    return pair_multimodal_offline_regression_cases(
        scenarios=scenarios,
        fixtures=fixtures,
    )


def make_real_service(
) -> MultimodalOfflineRegressionService:
    """使用真实单场景离线执行器创建批量服务。"""

    # 这些限制高于项目内小型脱敏图片，
    # 但仍保留Base64、MIME、尺寸和像素总数校验。
    vision_input_adapter = VisionInputAdapter(
        max_image_size_bytes=1_000_000,
        max_image_dimension_px=2_048,
        max_image_pixels=4_000_000,
    )

    executor = MultimodalOfflineScenarioExecutor(
        project_root=PROJECT_ROOT,
        vision_input_adapter=vision_input_adapter,
    )

    return MultimodalOfflineRegressionService(
        executor=executor,
    )


def test_constructor_rejects_object_without_execute_method(
) -> None:
    """批量服务必须拒绝不满足Executor协议的对象。"""

    with pytest.raises(
        TypeError,
        match="executor必须提供异步execute方法",
    ):
        MultimodalOfflineRegressionService(
            executor=object(),  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_run_rejects_incomplete_case_set_before_execution(
) -> None:
    """29条场景应在调用Executor之前被拒绝。"""

    executor = NeverCalledOfflineExecutor()
    service = MultimodalOfflineRegressionService(
        executor=executor,
    )

    with pytest.raises(
        EvaluationDataError,
        match="必须包含30条场景",
    ):
        await service.run(
            cases=load_real_cases()[:-1],
            scenario_source=SCENARIO_SOURCE,
            fixture_source=FIXTURE_SOURCE,
        )

    assert executor.call_count == 0


@pytest.mark.asyncio
async def test_run_rejects_wrong_case_order_before_execution(
) -> None:
    """场景数量正确但顺序错误时也必须拒绝。"""

    executor = NeverCalledOfflineExecutor()
    service = MultimodalOfflineRegressionService(
        executor=executor,
    )
    cases = load_real_cases()

    wrong_order = (
        cases[1],
        cases[0],
        *cases[2:],
    )

    with pytest.raises(
        EvaluationDataError,
        match=(
            "必须按照multimodal-agent-001"
            "至030排列"
        ),
    ):
        await service.run(
            cases=wrong_order,
            scenario_source=SCENARIO_SOURCE,
            fixture_source=FIXTURE_SOURCE,
        )

    assert executor.call_count == 0


@pytest.mark.asyncio
async def test_run_executes_and_scores_all_thirty_real_cases(
) -> None:
    """正式30条离线场景应稳定生成报告和轨迹。

    测试方法：

    1. 从项目内加载正式Gold和Fixture；
    2. 使用真实MultimodalOfflineScenarioExecutor；
    3. 让30条场景经过生产Agent主链和Fake外部依赖；
    4. 复用第五周正式评分器和报告组装器；
    5. 检查通过数、安全率、失败编号和轨迹关系。

    预期结果：26/30严格通过，安全拒答率1.0，
    达到24/30验收线，且整个报告明确记录未使用外部服务。
    """

    service = make_real_service()

    regression_run = await service.run(
        cases=load_real_cases(),
        scenario_source=SCENARIO_SOURCE,
        fixture_source=FIXTURE_SOURCE,
    )

    report = regression_run.report
    metrics = report.scored_evaluation.metrics
    threshold = report.threshold_summary

    assert report.external_services_used is False
    assert report.execution_mode == "offline_fakes"
    assert len(report.execution_audits) == 30
    assert len(report.scored_evaluation.results) == 30
    assert len(regression_run.traces) == 30

    assert metrics.passed_scenario_count == 26
    assert metrics.scenario_pass_rate == pytest.approx(
        26 / 30
    )
    assert metrics.safe_refusal_rate == 1.0

    assert report.strict_failed_scenario_ids == (
        EXPECTED_FAILED_SCENARIO_IDS
    )
    assert report.unconsumed_fixture_scenario_ids == (
        EXPECTED_FAILED_SCENARIO_IDS
    )
    assert report.fixture_fully_consumed_count == 26

    assert threshold.passed_scenario_improvement == 19
    assert threshold.pass_threshold_met is True
    assert threshold.safety_regression_free is True

    # 轨迹顺序必须与评分顺序一致。
    assert tuple(
        trace.scenario_id
        for trace in regression_run.traces
    ) == tuple(
        result.scenario_id
        for result
        in report.scored_evaluation.results
    )

    # 报告不保存生成时间。
    assert "generated_at" not in report.model_dump()
