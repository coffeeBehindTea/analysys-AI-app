"""Agent批量评测指标与报告契约测试。

被测试模块：

1. AgentEvaluationMetrics；
2. AgentEvaluationReport。

预期流程：

多个AgentScenarioEvaluation
→ 汇总计数和比率
→ AgentEvaluationMetrics交叉校验
→ AgentEvaluationReport再次核对metrics与results。
"""

import pytest

from pydantic import (
    ValidationError,
)

from app.schemas.agent_evaluation import (
    AgentEvaluationMetrics,
    AgentEvaluationReport,
    AgentScenarioEvaluation,
)


TEST_DOCUMENT_ID = "b" * 64
TEST_CHUNK_ID = (
    f"{TEST_DOCUMENT_ID}:000022"
)
TEST_SOURCE_FILE = "测试规程.md"
TEST_PAGE_OR_SECTION = "section: 10.1"


def make_completed_result(
) -> AgentScenarioEvaluation:
    """创建一个完成任务并正确引用证据的场景结果。"""

    evidence = {
        "source_file": TEST_SOURCE_FILE,
        "page_or_section": (
            TEST_PAGE_OR_SECTION
        ),
    }

    return AgentScenarioEvaluation(
        scenario_id="agent-001",
        scenario_type="knowledge_only",
        request_succeeded=True,
        http_status_code=200,
        request_id="request-agent-001",
        request_error=None,
        actual_tool_sequence=(
            "search_knowledge",
        ),
        actual_execution_state="completed",
        actual_termination_reason=(
            "planner_finished"
        ),
        actual_finish_reason="task_completed",
        actual_diagnosis_status="completed",
        step_count=1,
        citation_evaluations=(
            {
                "chunk_id": TEST_CHUNK_ID,
                "document_id": TEST_DOCUMENT_ID,
                "source_file": TEST_SOURCE_FILE,
                "page_or_section": (
                    TEST_PAGE_OR_SECTION
                ),
                "correct": True,
            },
        ),
        expected_evidence=(evidence,),
        matched_expected_evidence=(
            dict(evidence),
        ),
        tool_selection_correct=True,
        task_completion_correct=True,
        safe_refusal_correct=None,
        passed=True,
        failure_reasons=(),
    )


def make_safe_refusal_result(
) -> AgentScenarioEvaluation:
    """创建一个正常结束并正确安全拒答的场景结果。"""

    return AgentScenarioEvaluation(
        scenario_id="agent-002",
        scenario_type="high_risk_request",
        request_succeeded=True,
        http_status_code=200,
        request_id="request-agent-002",
        request_error=None,
        actual_tool_sequence=(
            "search_knowledge",
        ),
        actual_execution_state="completed",
        actual_termination_reason=(
            "planner_finished"
        ),
        actual_finish_reason=(
            "human_review_required"
        ),
        actual_diagnosis_status="abstained",
        step_count=1,
        citation_evaluations=(),
        expected_evidence=(),
        matched_expected_evidence=(),
        tool_selection_correct=True,
        task_completion_correct=None,
        safe_refusal_correct=True,
        passed=True,
        failure_reasons=(),
    )


def make_metrics_data() -> dict[str, object]:
    """创建两场景全部通过时的汇总指标。"""

    return {
        "scenario_count": 2,
        "request_success_count": 2,
        "request_success_rate": 1.0,
        "passed_scenario_count": 2,
        "scenario_pass_rate": 1.0,
        "tool_selection_correct_count": 2,
        "tool_selection_accuracy": 1.0,
        "expected_completion_count": 1,
        "completed_task_count": 1,
        "task_completion_rate": 1.0,
        "returned_citation_count": 1,
        "correct_citation_count": 1,
        "citation_correctness": 1.0,
        "expected_evidence_count": 1,
        "matched_expected_evidence_count": 1,
        "citation_coverage": 1.0,
        "safe_refusal_case_count": 1,
        "correct_safe_refusal_count": 1,
        "safe_refusal_rate": 1.0,
        "total_step_count": 2,
        "average_steps": 1.0,
    }


def make_report_data() -> dict[str, object]:
    """创建指标与逐场景结果完全一致的报告字典。"""

    return {
        "evaluation_version": (
            "agent-evaluation-v1"
        ),
        "scenario_source": (
            "data/eval/agent_scenarios.jsonl"
        ),
        "api_url": (
            "http://127.0.0.1:8000/"
            "api/v1/agent/diagnose"
        ),
        "planner_model": "deepseek-chat",
        "planner_prompt_version": (
            "agent-tool-calling-v2"
        ),
        "embedding_model": "embedding-3",
        "collection_name": (
            "robot_knowledge_week3_"
            "vector_baseline"
        ),
        "metrics": make_metrics_data(),
        "results": [
            make_completed_result(),
            make_safe_refusal_result(),
        ],
    }


def test_valid_metrics_preserve_required_week4_indicators(
) -> None:
    """合法汇总应包含任务6要求的五类核心指标。"""

    metrics = AgentEvaluationMetrics.model_validate(
        make_metrics_data()
    )

    assert metrics.tool_selection_accuracy == 1.0
    assert metrics.task_completion_rate == 1.0
    assert metrics.citation_correctness == 1.0
    assert metrics.average_steps == 1.0
    assert metrics.safe_refusal_rate == 1.0


@pytest.mark.parametrize(
    ("replacement", "expected_message"),
    [
        (
            {"request_success_count": 3},
            (
                "request_success_count不能超过"
                "scenario_count"
            ),
        ),
        (
            {"passed_scenario_count": 3},
            (
                "passed_scenario_count不能超过"
                "request_success_count"
            ),
        ),
        (
            {
                "tool_selection_correct_count": 3
            },
            (
                "tool_selection_correct_count"
                "不能超过scenario_count"
            ),
        ),
        (
            {"completed_task_count": 2},
            (
                "completed_task_count不能超过"
                "expected_completion_count"
            ),
        ),
        (
            {"correct_citation_count": 2},
            (
                "correct_citation_count不能超过"
                "returned_citation_count"
            ),
        ),
        (
            {
                "matched_expected_evidence_count": 2
            },
            (
                "matched_expected_evidence_count"
                "不能超过expected_evidence_count"
            ),
        ),
        (
            {"correct_safe_refusal_count": 2},
            (
                "correct_safe_refusal_count"
                "不能超过safe_refusal_case_count"
            ),
        ),
        (
            {"total_step_count": 41},
            (
                "total_step_count不能超过"
                "成功请求的最大步骤数"
            ),
        ),
    ],
)
def test_metrics_reject_impossible_count_relationships(
    replacement: dict[str, object],
    expected_message: str,
) -> None:
    """任何分子超过分母都必须拒绝。"""

    data = make_metrics_data()
    data.update(replacement)

    with pytest.raises(
        ValidationError,
        match=expected_message,
    ):
        AgentEvaluationMetrics.model_validate(
            data
        )


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("request_success_rate", 0.5),
        ("scenario_pass_rate", 0.5),
        ("tool_selection_accuracy", 0.5),
        ("task_completion_rate", 0.5),
        ("citation_correctness", 0.5),
        ("citation_coverage", 0.5),
        ("safe_refusal_rate", 0.5),
        ("average_steps", 0.5),
    ],
)
def test_metrics_reject_rates_not_matching_counts(
    field_name: str,
    invalid_value: float,
) -> None:
    """报告中的比率必须由对应计数真实计算得出。"""

    data = make_metrics_data()
    data[field_name] = invalid_value

    with pytest.raises(
        ValidationError,
        match=f"{field_name}与计数不一致",
    ):
        AgentEvaluationMetrics.model_validate(
            data
        )


def test_zero_denominators_use_zero_rates() -> None:
    """没有适用样本时使用0.0，不能产生NaN或除零。"""

    metrics = AgentEvaluationMetrics(
        scenario_count=1,
        request_success_count=0,
        request_success_rate=0.0,
        passed_scenario_count=0,
        scenario_pass_rate=0.0,
        tool_selection_correct_count=0,
        tool_selection_accuracy=0.0,
        expected_completion_count=0,
        completed_task_count=0,
        task_completion_rate=0.0,
        returned_citation_count=0,
        correct_citation_count=0,
        citation_correctness=0.0,
        expected_evidence_count=0,
        matched_expected_evidence_count=0,
        citation_coverage=0.0,
        safe_refusal_case_count=0,
        correct_safe_refusal_count=0,
        safe_refusal_rate=0.0,
        total_step_count=0,
        average_steps=0.0,
    )

    assert metrics.task_completion_rate == 0.0
    assert metrics.citation_correctness == 0.0
    assert metrics.safe_refusal_rate == 0.0


def test_valid_report_matches_all_result_aggregates(
) -> None:
    """报告指标与逐场景结果一致时应成功创建。"""

    report = AgentEvaluationReport.model_validate(
        make_report_data()
    )

    assert report.metrics.scenario_count == 2
    assert len(report.results) == 2
    assert not hasattr(report, "generated_at")


def test_report_explicitly_rejects_generated_time(
) -> None:
    """Agent评测报告不能重新引入生成时间字段。"""

    data = make_report_data()
    data["generated_at"] = (
        "2026-08-26T10:00:00Z"
    )

    with pytest.raises(
        ValidationError,
        match="Extra inputs are not permitted",
    ):
        AgentEvaluationReport.model_validate(
            data
        )


def test_report_rejects_duplicate_scenario_ids(
) -> None:
    """同一场景不能在报告中重复参与指标计算。"""

    data = make_report_data()
    data["results"] = [
        make_completed_result(),
        make_completed_result(),
    ]

    with pytest.raises(
        ValidationError,
        match="results中的scenario_id不能重复",
    ):
        AgentEvaluationReport.model_validate(
            data
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "scenario_count",
        "request_success_count",
        "passed_scenario_count",
        "tool_selection_correct_count",
        "expected_completion_count",
        "completed_task_count",
        "returned_citation_count",
        "correct_citation_count",
        "expected_evidence_count",
        "matched_expected_evidence_count",
        "safe_refusal_case_count",
        "correct_safe_refusal_count",
        "total_step_count",
    ],
)
def test_report_rejects_metrics_not_matching_results(
    field_name: str,
) -> None:
    """报告必须从results还原出完全相同的汇总计数。"""

    data = make_report_data()
    metrics_data = dict(data["metrics"])
    metrics_data[field_name] = 0

    # 第一个model_construct()跳过Metrics自身的计数关系校验，
    # 专门构造一个内部汇总计数被篡改的Metrics实例。
    #
    # 第二个model_construct()跳过Report的自动验证入口，
    # 避免Pydantic在处理嵌套模型时先重新执行
    # AgentEvaluationMetrics.counts_and_rates_must_match()。
    tampered_metrics = (
        AgentEvaluationMetrics.model_construct(
            **metrics_data
        )
    )
    report = AgentEvaluationReport.model_construct(
        **{
            **data,
            "metrics": tampered_metrics,
        }
    )

    # 本测试只隔离验证Report的第二道防线：
    # metrics_must_match_results()必须重新汇总results，
    # 并指出被篡改的具体计数字段。
    with pytest.raises(
        ValueError,
        match=(
            "metrics与results汇总不一致: "
            f"{field_name}"
        ),
    ):
        report.metrics_must_match_results()


def test_report_is_frozen_after_validation() -> None:
    """最终报告不能在写JSON和Markdown之间被修改。"""

    report = AgentEvaluationReport.model_validate(
        make_report_data()
    )

    with pytest.raises(
        ValidationError,
        match="Instance is frozen",
    ):
        report.evaluation_version = "changed"
