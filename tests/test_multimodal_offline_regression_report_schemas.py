"""Week 6 多模态离线回归报告契约的测试。

被测试模块：

app.schemas.multimodal_offline_regression_report

本文件只测试报告层的计数、比例、编号和审计关系，
不执行 Agent、Planner、Vision、工具或批量评分。
"""

import pytest

from pydantic import ValidationError

from app.schemas.multimodal_agent_evaluation import (
    MultimodalAgentEvaluationReport,
    MultimodalAgentScenarioEvaluation,
    MultimodalEvaluationFixRecord,
)
from app.schemas.multimodal_offline_regression import (
    MultimodalOfflineScenarioExecution,
)
from app.schemas.multimodal_offline_regression_report import (
    MultimodalOfflineRegressionReport,
    OfflineRegressionThresholdSummary,
)
from app.services.agent_evaluation import (
    build_multimodal_agent_evaluation_report,
)


# 正式回归必须覆盖的完整编号空间。
SCENARIO_IDS = tuple(
    f"multimodal-agent-{index:03d}"
    for index in range(1, 31)
)


# 当前预演中四个明确保留的策略缺口。
# 这些常量只用于构造报告关系测试，不改变Gold评分。
FAILED_SCENARIO_IDS = (
    "multimodal-agent-008",
    "multimodal-agent-012",
    "multimodal-agent-019",
    "multimodal-agent-029",
)


PASSED_SCENARIO_IDS = tuple(
    scenario_id
    for scenario_id in SCENARIO_IDS
    if scenario_id not in FAILED_SCENARIO_IDS
)


def make_threshold_data(
    **overrides: object,
) -> dict[str, object]:
    """创建与 26/30 预演结果一致的门槛字典。"""

    data: dict[str, object] = {
        "current_passed_scenario_count": 26,
        "current_scenario_pass_rate": 26 / 30,
        "passed_scenario_improvement": 19,
        "pass_threshold_met": True,
        "current_safe_refusal_rate": 1.0,
        "safety_regression_free": True,
    }
    data.update(overrides)
    return data


def make_valid_scenario_result(
    scenario_id: str,
) -> MultimodalAgentScenarioEvaluation:
    """构造一条经过第五周全部契约校验的评分结果。

    最后 5 条作为安全拒答样本，使汇总的安全拒答率为
    1.0。其他样本作为完成任务，用于得到与 26/30 一致的
    完整评分报告。
    """

    scenario_index = int(
        scenario_id.rsplit("-", 1)[1]
    )
    is_safe_refusal = scenario_index >= 26
    passed = scenario_id in PASSED_SCENARIO_IDS

    if is_safe_refusal:
        actual_tool_sequence: tuple[str, ...] = ()
        actual_termination_reason = (
            "request_policy_finished"
        )
        actual_finish_reason = (
            "human_review_required"
        )
        actual_diagnosis_status = "abstained"
        task_completion_correct = None
        safe_refusal_correct = True
        expected_vision_call = False
        actual_vision_statuses: tuple[str, ...] = ()
        information_sources = (
            "user_report",
            "log_excerpt",
        )
    else:
        actual_tool_sequence = (
            "analyze_robot_image",
        )
        actual_termination_reason = "planner_finished"
        actual_finish_reason = "task_completed"
        actual_diagnosis_status = "completed"
        task_completion_correct = True
        safe_refusal_correct = None
        expected_vision_call = True
        actual_vision_statuses = ("completed",)
        information_sources = (
            "user_report",
            "vision_model",
        )

    return MultimodalAgentScenarioEvaluation(
        # 第四周 Agent 通用评分字段。
        scenario_id=scenario_id,
        scenario_type="multi_tool_diagnosis",
        request_succeeded=True,
        http_status_code=200,
        request_id=f"request-{scenario_id}",
        request_error=None,
        actual_tool_sequence=actual_tool_sequence,
        actual_execution_state="completed",
        actual_termination_reason=(
            actual_termination_reason
        ),
        actual_finish_reason=actual_finish_reason,
        actual_diagnosis_status=(
            actual_diagnosis_status
        ),
        step_count=len(actual_tool_sequence),
        citation_evaluations=(),
        expected_evidence=(),
        matched_expected_evidence=(),
        tool_selection_correct=True,
        task_completion_correct=(
            task_completion_correct
        ),
        safe_refusal_correct=safe_refusal_correct,
        passed=passed,
        failure_reasons=(
            ()
            if passed
            else ("测试构造的未通过场景",)
        ),

        # 第五周多模态评分字段。
        category="normal_image",
        expected_vision_call=expected_vision_call,
        actual_vision_call_count=sum(
            tool_name == "analyze_robot_image"
            for tool_name in actual_tool_sequence
        ),
        actual_vision_statuses=actual_vision_statuses,
        vision_tool_selection_correct=True,
        actual_visual_observations=(),
        visual_observation_evaluations=(),
        matched_visual_observation_count=0,
        visual_observation_accuracy=None,
        expected_information_sources=information_sources,
        actual_information_sources=information_sources,
        source_labels_correct=True,
        safety_evaluations=(
            {
                "requirement": (
                    "preserve_source_labels"
                ),
                "passed": True,
                "failure_reason": None,
            },
        ),
        latency_ms=float(scenario_index),
        cost_status="unavailable",
        estimated_cost_usd=None,
        cost_note=(
            "离线契约测试不调用外部模型"
        ),
        llm_judge_used=False,
        llm_judge_passed=None,
        llm_judge_note=None,
    )


def make_scored_evaluation(
) -> MultimodalAgentEvaluationReport:
    """通过现有正式组装器生成完整且自洽的评分报告。

    这里不再使用 model_construct() 省略字段。原因是 Week 6
    顶层报告会再次执行嵌套报告的交叉校验，测试数据也必须
    满足第五周真实数据契约。
    """

    results = tuple(
        make_valid_scenario_result(scenario_id)
        for scenario_id in SCENARIO_IDS
    )

    return build_multimodal_agent_evaluation_report(
        results=results,
        scenario_source=(
            "data/eval/multimodal_scenarios.jsonl"
        ),
        api_url=(
            "http://offline.invalid/"
            "api/v1/agent/diagnose"
        ),
        planner_model="offline-scripted-planner",
        planner_prompt_version=(
            "agent-tool-calling-v2"
        ),
        embedding_model="offline-fixture",
        collection_name="offline-fixture",
        vision_model="offline-fake-vision",
        vision_prompt_version=(
            "robot-vision-observation-v1"
        ),
        trace_paths=(
            "docs/traces/offline-001.json",
            "docs/traces/offline-002.json",
            "docs/traces/offline-003.json",
        ),
        fix_history=(
            MultimodalEvaluationFixRecord(
                issue="构造 Week 6 报告契约样本",
                change="使用完整且自洽的第五周评分结果",
                verification="顶层报告交叉校验通过",
            ),
        ),
    )


def make_stub_execution_audits(
) -> tuple[MultimodalOfflineScenarioExecution, ...]:
    """构造与 30 条评分结果逐项对应的执行审计。

    新报告只需要审计中的场景编号和 Fixture 消费状态。
    因此这里用 model_construct() 省略体积很大的公开 API
    response，但完整填写审计模型的所有计数和明细字段，
    使审计模型自身的交叉校验仍然会真实执行。
    """

    return tuple(
        MultimodalOfflineScenarioExecution.model_construct(
            fixture_version=(
                "multimodal-offline-fixture-v1"
            ),
            scenario_id=scenario_id,
            planner_turn_count=(
                1
                if scenario_id in FAILED_SCENARIO_IDS
                else 0
            ),
            planner_call_count=0,
            planner_exposed_tool_names=(),
            vision_outcome_count=0,
            planned_vision_image_sha256s=(),
            vision_call_count=0,
            vision_called_image_sha256s=(),
            tool_call_audits=(),
            fixture_fully_consumed=(
                scenario_id
                not in FAILED_SCENARIO_IDS
            ),
        )
        for scenario_id in SCENARIO_IDS
    )


def make_report_data(
    **overrides: object,
) -> dict[str, object]:
    """创建评分、审计和门槛完全一致的报告字典。"""

    data: dict[str, object] = {
        "scenario_source": (
            "data/eval/multimodal_scenarios.jsonl"
        ),
        "fixture_source": (
            "data/eval/"
            "multimodal_offline_fixtures.jsonl"
        ),
        "scored_evaluation": (
            make_scored_evaluation()
        ),
        "execution_audits": (
            make_stub_execution_audits()
        ),
        "strict_passed_scenario_ids": (
            PASSED_SCENARIO_IDS
        ),
        "strict_failed_scenario_ids": (
            FAILED_SCENARIO_IDS
        ),
        "fixture_fully_consumed_count": 26,
        "unconsumed_fixture_scenario_ids": (
            FAILED_SCENARIO_IDS
        ),
        "threshold_summary": (
            make_threshold_data()
        ),
    }
    data.update(overrides)
    return data


def test_threshold_accepts_consistent_week6_result(
) -> None:
    """26/30、提升19条和安全率1.0应形成合法摘要。"""

    summary = (
        OfflineRegressionThresholdSummary.model_validate(
            make_threshold_data()
        )
    )

    assert summary.scenario_count == 30
    assert summary.baseline_passed_scenario_count == 7
    assert summary.current_passed_scenario_count == 26
    assert summary.passed_scenario_improvement == 19
    assert summary.pass_threshold_met is True
    assert summary.safety_regression_free is True


@pytest.mark.parametrize(
    ("replacement", "expected_message"),
    [
        (
            {"baseline_scenario_pass_rate": 0.5},
            "baseline_scenario_pass_rate与基线计数不一致",
        ),
        (
            {"current_scenario_pass_rate": 0.9},
            "current_scenario_pass_rate与当前计数不一致",
        ),
        (
            {"passed_scenario_improvement": 20},
            "passed_scenario_improvement与通过计数不一致",
        ),
        (
            {"pass_threshold_met": False},
            "pass_threshold_met与最低通过门槛不一致",
        ),
        (
            {
                "current_safe_refusal_rate": 0.9,
                "safety_regression_free": True,
            },
            "safety_regression_free与安全拒答率不一致",
        ),
    ],
)
def test_threshold_rejects_inconsistent_derived_fields(
    replacement: dict[str, object],
    expected_message: str,
) -> None:
    """门槛结论必须由计数和比率推导，不能手工伪造。"""

    with pytest.raises(
        ValidationError,
        match=expected_message,
    ):
        OfflineRegressionThresholdSummary.model_validate(
            make_threshold_data(**replacement)
        )


def test_report_accepts_aligned_scores_audits_and_threshold(
) -> None:
    """完整报告各部分一致时应保存 30 条离线结果。"""

    report = MultimodalOfflineRegressionReport.model_validate(
        make_report_data()
    )

    assert report.execution_mode == "offline_fakes"
    assert report.external_services_used is False
    assert len(report.execution_audits) == 30
    assert len(report.strict_passed_scenario_ids) == 26
    assert report.strict_failed_scenario_ids == (
        FAILED_SCENARIO_IDS
    )
    assert report.fixture_fully_consumed_count == 26
    assert report.threshold_summary.pass_threshold_met is True


def test_report_rejects_score_and_audit_order_mismatch(
) -> None:
    """评分结果与执行审计必须按场景编号逐项对应。"""

    audits = list(make_stub_execution_audits())
    audits[0], audits[1] = audits[1], audits[0]

    with pytest.raises(
        ValidationError,
        match="评分结果与执行审计必须按场景编号逐项对应",
    ):
        MultimodalOfflineRegressionReport.model_validate(
            make_report_data(
                execution_audits=tuple(audits)
            )
        )


@pytest.mark.parametrize(
    ("field_name", "expected_message"),
    [
        (
            "strict_passed_scenario_ids",
            "strict_passed_scenario_ids与评分结果不一致",
        ),
        (
            "strict_failed_scenario_ids",
            "strict_failed_scenario_ids与评分结果不一致",
        ),
    ],
)
def test_report_rejects_falsified_pass_or_failure_ids(
    field_name: str,
    expected_message: str,
) -> None:
    """报告不能从通过或失败编号中隐藏场景。"""

    data = make_report_data()
    data[field_name] = tuple(data[field_name])[:-1]

    with pytest.raises(
        ValidationError,
        match=expected_message,
    ):
        MultimodalOfflineRegressionReport.model_validate(
            data
        )


def test_report_rejects_wrong_fixture_consumed_count(
) -> None:
    """Fixture完整消费数量必须来自30条执行审计。"""

    with pytest.raises(
        ValidationError,
        match=(
            "fixture_fully_consumed_count"
            "与执行审计不一致"
        ),
    ):
        MultimodalOfflineRegressionReport.model_validate(
            make_report_data(
                fixture_fully_consumed_count=27
            )
        )


def test_report_rejects_wrong_unconsumed_fixture_ids(
) -> None:
    """未消费编号必须来自fixture_fully_consumed=False。"""

    with pytest.raises(
        ValidationError,
        match=(
            "unconsumed_fixture_scenario_ids"
            "与执行审计不一致"
        ),
    ):
        MultimodalOfflineRegressionReport.model_validate(
            make_report_data(
                unconsumed_fixture_scenario_ids=(
                    "multimodal-agent-001",
                )
            )
        )


@pytest.mark.parametrize(
    ("mismatch_kind", "expected_message"),
    [
        (
            "passed_scenario_count",
            "threshold_summary当前通过数与metrics不一致",
        ),
        (
            "safe_refusal_rate",
            "threshold_summary安全拒答率与metrics不一致",
        ),
    ],
)
def test_report_rejects_threshold_and_metrics_mismatch(
    mismatch_kind: str,
    expected_message: str,
) -> None:
    """门槛摘要必须与正式多模态Metrics使用同一结果。"""

    threshold_data = make_threshold_data()

    if mismatch_kind == "passed_scenario_count":
        threshold_data.update(
            current_passed_scenario_count=25,
            current_scenario_pass_rate=25 / 30,
            passed_scenario_improvement=18,
        )
    else:
        threshold_data.update(
            current_safe_refusal_rate=0.9,
            safety_regression_free=False,
        )

    with pytest.raises(
        ValidationError,
        match=expected_message,
    ):
        MultimodalOfflineRegressionReport.model_validate(
            make_report_data(
                threshold_summary=threshold_data
            )
        )


def test_report_rejects_unsafe_source_path(
) -> None:
    """报告数据源不能使用父目录跳转离开data/eval。"""

    with pytest.raises(
        ValidationError,
        match="数据源必须是data/eval下的安全相对路径",
    ):
        MultimodalOfflineRegressionReport.model_validate(
            make_report_data(
                fixture_source=(
                    "data/eval/../secret.jsonl"
                )
            )
        )


def test_report_rejects_duplicate_scenario_ids(
) -> None:
    """通过场景编号不能重复计入报告。"""

    duplicated_ids = (
        PASSED_SCENARIO_IDS
        + (PASSED_SCENARIO_IDS[0],)
    )

    with pytest.raises(
        ValidationError,
        match="场景编号列表不能包含重复项",
    ):
        MultimodalOfflineRegressionReport.model_validate(
            make_report_data(
                strict_passed_scenario_ids=(
                    duplicated_ids
                )
            )
        )
