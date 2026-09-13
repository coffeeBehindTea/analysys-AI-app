"""多模态Agent批量评测报告Schema的单元测试。

本模块只构造内存中的脱敏字典并执行Pydantic校验，
不发送HTTP请求，也不调用Agent、Vision、LLM或知识库。

测试覆盖：

1. 类别汇总的计数和通过率；
2. 第四周基础指标和第五周新增指标的继承；
3. 视觉字段、Vision选择、来源和安全指标；
4. P50/P95端到端延迟；
5. 单请求成本估算和覆盖率；
6. 辅助LLM Judge不参与正式passed条件；
7. 至少三份安全相对轨迹路径；
8. 失败原因、修正记录和逐场景结果的一致性。
"""

from copy import deepcopy

from pydantic import ValidationError
import pytest

from app.schemas.multimodal_agent_evaluation import (
    MultimodalAgentCategorySummary,
    MultimodalAgentEvaluationMetrics,
    MultimodalAgentEvaluationReport,
    MultimodalAgentScenarioEvaluation,
)


# 固定文档标识只用于构造合法引用，
# 不对应本地Chroma中的真实文档。
TEST_DOCUMENT_ID = "c" * 64
TEST_SOURCE_FILE = "仓储机器人故障说明.txt"
TEST_PAGE_OR_SECTION = "section: ERR-NET-4001"


def make_completed_result_data(
    *,
    index: int,
    latency_ms: float,
    cost_status: str,
    estimated_cost_usd: float | None,
) -> dict[str, object]:
    """构造一条完整完成且全部正式检查通过的结果。"""

    scenario_id = f"multimodal-agent-{index:03d}"
    expected_evidence = {
        "source_file": TEST_SOURCE_FILE,
        "page_or_section": TEST_PAGE_OR_SECTION,
    }

    return {
        # 第四周AgentScenarioEvaluation字段。
        "scenario_id": scenario_id,
        "scenario_type": "multi_tool_diagnosis",
        "request_succeeded": True,
        "http_status_code": 200,
        "request_id": f"request-{scenario_id}",
        "request_error": None,
        "actual_tool_sequence": [
            "analyze_robot_image",
            "search_knowledge",
        ],
        "actual_execution_state": "completed",
        "actual_termination_reason": "planner_finished",
        "actual_finish_reason": "task_completed",
        "actual_diagnosis_status": "completed",
        "step_count": 2,
        "citation_evaluations": [
            {
                "chunk_id": (
                    f"{TEST_DOCUMENT_ID}:{index:06d}"
                ),
                "document_id": TEST_DOCUMENT_ID,
                "source_file": TEST_SOURCE_FILE,
                "page_or_section": TEST_PAGE_OR_SECTION,
                "correct": True,
            },
        ],
        "expected_evidence": [expected_evidence],
        "matched_expected_evidence": [
            expected_evidence,
        ],
        "tool_selection_correct": True,
        "task_completion_correct": True,
        "safe_refusal_correct": None,
        "passed": True,
        "failure_reasons": [],

        # 第五周多模态字段。
        "category": "normal_image",
        "expected_vision_call": True,
        "actual_vision_call_count": 1,
        "actual_vision_statuses": ["completed"],
        "vision_tool_selection_correct": True,
        "actual_visual_observations": [
            "面板显示ERR-NET-4001",
        ],
        "visual_observation_evaluations": [
            {
                "expected_observation": (
                    "面板显示ERR-NET-4001"
                ),
                "matched": True,
                "matched_actual_observation": (
                    "面板显示ERR-NET-4001"
                ),
                "match_method": "normalized_exact",
            },
        ],
        "matched_visual_observation_count": 1,
        "visual_observation_accuracy": 1.0,
        "expected_information_sources": [
            "user_report",
            "log_excerpt",
            "vision_model",
            "knowledge_base",
        ],
        "actual_information_sources": [
            "user_report",
            "log_excerpt",
            "vision_model",
            "knowledge_base",
        ],
        "source_labels_correct": True,
        "safety_evaluations": [
            {
                "requirement": "preserve_source_labels",
                "passed": True,
                "failure_reason": None,
            },
        ],
        "latency_ms": latency_ms,
        "cost_status": cost_status,
        "estimated_cost_usd": estimated_cost_usd,
        "cost_note": (
            "根据公开usage估算"
            if cost_status == "estimated"
            else "Provider未返回足够usage"
        ),
        "llm_judge_used": False,
        "llm_judge_passed": None,
        "llm_judge_note": None,
    }


def make_safe_refusal_result_data() -> dict[str, object]:
    """构造安全拒答正确但来源标签检查失败的结果。"""

    return {
        "scenario_id": "multimodal-agent-003",
        "scenario_type": "insufficient_evidence",
        "request_succeeded": True,
        "http_status_code": 200,
        "request_id": "request-multimodal-agent-003",
        "request_error": None,
        "actual_tool_sequence": [
            "search_knowledge",
        ],
        "actual_execution_state": "completed",
        "actual_termination_reason": "planner_finished",
        "actual_finish_reason": "insufficient_information",
        "actual_diagnosis_status": "abstained",
        "step_count": 1,
        "citation_evaluations": [],
        "expected_evidence": [],
        "matched_expected_evidence": [],
        "tool_selection_correct": True,
        "task_completion_correct": None,
        "safe_refusal_correct": True,
        "passed": False,
        "failure_reasons": [
            "信息来源标记不完整",
        ],
        "category": "missing_or_unanswerable",
        "expected_vision_call": False,
        "actual_vision_call_count": 0,
        "actual_vision_statuses": [],
        "vision_tool_selection_correct": True,
        "actual_visual_observations": [],
        "visual_observation_evaluations": [],
        "matched_visual_observation_count": 0,
        "visual_observation_accuracy": None,
        "expected_information_sources": [
            "user_report",
            "log_excerpt",
        ],
        "actual_information_sources": [
            "user_report",
        ],
        "source_labels_correct": False,
        "safety_evaluations": [
            {
                "requirement": (
                    "do_not_guess_unreadable_content"
                ),
                "passed": True,
                "failure_reason": None,
            },
        ],
        "latency_ms": 400.0,
        "cost_status": "unavailable",
        "estimated_cost_usd": None,
        "cost_note": "Provider未返回足够usage",
        "llm_judge_used": False,
        "llm_judge_passed": None,
        "llm_judge_note": None,
    }


def make_results() -> list[dict[str, object]]:
    """创建两条完成结果和一条安全拒答结果。"""

    return [
        make_completed_result_data(
            index=1,
            latency_ms=100.0,
            cost_status="estimated",
            estimated_cost_usd=0.003,
        ),
        make_completed_result_data(
            index=2,
            latency_ms=200.0,
            cost_status="unavailable",
            estimated_cost_usd=None,
        ),
        make_safe_refusal_result_data(),
    ]


def make_metrics_data(
    **overrides: object,
) -> dict[str, object]:
    """创建与make_results三条结果严格一致的汇总指标。"""

    data: dict[str, object] = {
        # 第四周继承指标。
        "scenario_count": 3,
        "request_success_count": 3,
        "request_success_rate": 1.0,
        "passed_scenario_count": 2,
        "scenario_pass_rate": 2 / 3,
        "tool_selection_correct_count": 3,
        "tool_selection_accuracy": 1.0,
        "expected_completion_count": 2,
        "completed_task_count": 2,
        "task_completion_rate": 1.0,
        "returned_citation_count": 2,
        "correct_citation_count": 2,
        "citation_correctness": 1.0,
        "expected_evidence_count": 2,
        "matched_expected_evidence_count": 2,
        "citation_coverage": 1.0,
        "safe_refusal_case_count": 1,
        "correct_safe_refusal_count": 1,
        "safe_refusal_rate": 1.0,
        "total_step_count": 5,
        "average_steps": 5 / 3,

        # 第五周新增指标。
        "visual_observation_field_count": 2,
        "matched_visual_observation_count": 2,
        "image_observation_field_accuracy": 1.0,
        "vision_tool_selection_correct_count": 3,
        "vision_tool_selection_accuracy": 1.0,
        "source_label_correct_count": 2,
        "source_label_accuracy": 2 / 3,
        "safety_requirement_count": 3,
        "passed_safety_requirement_count": 3,
        "safety_requirement_pass_rate": 1.0,
        "mean_latency_ms": 700 / 3,
        "p50_latency_ms": 200.0,
        "p95_latency_ms": 380.0,
        "cost_estimated_scenario_count": 1,
        "cost_estimation_coverage": 1 / 3,
        "known_total_estimated_cost_usd": 0.003,
        "mean_estimated_cost_usd_per_request": 0.003,
        "llm_judge_evaluated_count": 0,
        "llm_judge_pass_count": 0,
        "llm_judge_pass_rate": None,
    }
    data.update(overrides)
    return data


def make_report_data(
    **overrides: object,
) -> dict[str, object]:
    """创建与三条逐场景结果一致的完整报告。"""

    data: dict[str, object] = {
        "evaluation_version": (
            "multimodal-agent-evaluation-v1"
        ),
        "scenario_source": (
            "data/eval/multimodal_scenarios.jsonl"
        ),
        "api_url": (
            "http://127.0.0.1:8000/api/v1/agent/diagnose"
        ),
        "planner_model": "test-planner-model",
        "planner_prompt_version": "agent-tool-calling-v2",
        "embedding_model": "test-embedding-model",
        "collection_name": "test-collection",
        "vision_model": "test-vision-model",
        "vision_prompt_version": (
            "robot-vision-observation-v1"
        ),
        "llm_judge_role": "disabled",
        "metrics": make_metrics_data(),
        "results": make_results(),
        "category_summaries": [
            {
                "category": "normal_image",
                "scenario_count": 2,
                "passed_scenario_count": 2,
                "scenario_pass_rate": 1.0,
            },
            {
                "category": "missing_or_unanswerable",
                "scenario_count": 1,
                "passed_scenario_count": 0,
                "scenario_pass_rate": 0.0,
            },
        ],
        "trace_paths": [
            "docs/traces/multimodal-agent-001.json",
            "docs/traces/multimodal-agent-002.json",
            "docs/traces/multimodal-agent-003.json",
        ],
        "failure_reason_counts": [
            {
                "reason": "信息来源标记不完整",
                "scenario_count": 1,
            },
        ],
        "fix_history": [
            {
                "issue": "来源标签遗漏",
                "change": "增加确定性来源集合检查",
                "verification": (
                    "Schema测试拒绝不一致来源评分"
                ),
            },
        ],
    }
    data.update(overrides)
    return data


def test_category_summary_accepts_consistent_rate() -> None:
    """类别数量和通过率一致时应正常创建。"""

    summary = MultimodalAgentCategorySummary(
        category="normal_image",
        scenario_count=8,
        passed_scenario_count=6,
        scenario_pass_rate=0.75,
    )

    assert summary.scenario_pass_rate == 0.75


def test_category_summary_rejects_wrong_rate() -> None:
    """类别通过率不能与类别计数矛盾。"""

    with pytest.raises(
        ValidationError,
        match="scenario_pass_rate",
    ):
        MultimodalAgentCategorySummary(
            category="normal_image",
            scenario_count=8,
            passed_scenario_count=6,
            scenario_pass_rate=0.5,
        )


def test_metrics_preserve_parent_and_multimodal_fields() -> None:
    """第五周Metrics必须同时提供第四周和多模态指标。"""

    metrics = MultimodalAgentEvaluationMetrics.model_validate(
        make_metrics_data()
    )

    assert metrics.task_completion_rate == 1.0
    assert metrics.citation_correctness == 1.0
    assert metrics.image_observation_field_accuracy == 1.0
    assert metrics.p95_latency_ms == 380.0


def test_metrics_reject_wrong_visual_accuracy() -> None:
    """视觉准确率必须由视觉命中数量确定。"""

    with pytest.raises(
        ValidationError,
        match="image_observation_field_accuracy",
    ):
        MultimodalAgentEvaluationMetrics.model_validate(
            make_metrics_data(
                image_observation_field_accuracy=0.5,
            )
        )


def test_metrics_reject_p95_below_p50() -> None:
    """P95作为高分位延迟不能低于P50。"""

    with pytest.raises(
        ValidationError,
        match="p95_latency_ms",
    ):
        MultimodalAgentEvaluationMetrics.model_validate(
            make_metrics_data(
                p95_latency_ms=150.0,
            )
        )


def test_metrics_reject_wrong_mean_estimated_cost() -> None:
    """单请求估算成本必须由已知总成本和样本数计算。"""

    with pytest.raises(
        ValidationError,
        match="mean_estimated_cost_usd_per_request",
    ):
        MultimodalAgentEvaluationMetrics.model_validate(
            make_metrics_data(
                mean_estimated_cost_usd_per_request=0.01,
            )
        )


def test_auxiliary_judge_does_not_replace_formal_passed() -> None:
    """辅助Judge失败不能直接改写确定性正式通过结果。"""

    data = make_completed_result_data(
        index=1,
        latency_ms=100.0,
        cost_status="estimated",
        estimated_cost_usd=0.003,
    )
    data.update({
        "llm_judge_used": True,
        "llm_judge_passed": False,
        "llm_judge_note": "辅助语义检查认为表达不够完整",
    })

    result = MultimodalAgentScenarioEvaluation.model_validate(
        data
    )

    assert result.passed is True
    assert result.llm_judge_passed is False


def test_result_rejects_judge_fields_without_judge_call() -> None:
    """没有运行Judge时不能伪造Judge结论。"""

    data = make_completed_result_data(
        index=1,
        latency_ms=100.0,
        cost_status="estimated",
        estimated_cost_usd=0.003,
    )
    data["llm_judge_passed"] = True

    with pytest.raises(
        ValidationError,
        match="未运行辅助LLM Judge",
    ):
        MultimodalAgentScenarioEvaluation.model_validate(
            data
        )


def test_report_accepts_consistent_results_and_metrics() -> None:
    """完整报告应保留配置、分类、指标和逐场景结果。"""

    report = MultimodalAgentEvaluationReport.model_validate(
        make_report_data()
    )

    assert report.metrics.scenario_count == 3
    assert report.metrics.average_steps == 5 / 3
    assert len(report.results) == 3
    assert len(report.trace_paths) == 3


def test_report_rejects_wrong_category_summary() -> None:
    """类别汇总必须从results重新统计。"""

    report_data = make_report_data()
    category_summaries = deepcopy(
        report_data["category_summaries"]
    )
    assert isinstance(category_summaries, list)
    category_summaries[0]["passed_scenario_count"] = 1
    category_summaries[0]["scenario_pass_rate"] = 0.5
    report_data["category_summaries"] = category_summaries

    with pytest.raises(
        ValidationError,
        match="category_summaries与results",
    ):
        MultimodalAgentEvaluationReport.model_validate(
            report_data
        )


def test_report_rejects_wrong_multimodal_count() -> None:
    """新增汇总计数必须与逐场景结果一致。"""

    report_data = make_report_data()
    metrics = deepcopy(report_data["metrics"])
    assert isinstance(metrics, dict)
    metrics["source_label_correct_count"] = 3
    metrics["source_label_accuracy"] = 1.0
    report_data["metrics"] = metrics

    with pytest.raises(
        ValidationError,
        match="source_label_correct_count",
    ):
        MultimodalAgentEvaluationReport.model_validate(
            report_data
        )


def test_report_rejects_wrong_percentile() -> None:
    """P50和P95必须由所有逐场景latency_ms计算。"""

    report_data = make_report_data()
    metrics = deepcopy(report_data["metrics"])
    assert isinstance(metrics, dict)
    metrics["p95_latency_ms"] = 390.0
    report_data["metrics"] = metrics

    with pytest.raises(
        ValidationError,
        match="p95_latency_ms",
    ):
        MultimodalAgentEvaluationReport.model_validate(
            report_data
        )


def test_report_rejects_wrong_failure_reason_counts() -> None:
    """失败原因频次必须来自每条结果的failure_reasons。"""

    with pytest.raises(
        ValidationError,
        match="failure_reason_counts与results",
    ):
        MultimodalAgentEvaluationReport.model_validate(
            make_report_data(
                failure_reason_counts=[],
            )
        )


@pytest.mark.parametrize(
    "invalid_path",
    [
        pytest.param(
            "../secret.json",
            id="parent-directory",
        ),
        pytest.param(
            "data/trace.json",
            id="outside-docs",
        ),
        pytest.param(
            "docs/trace.txt",
            id="not-json",
        ),
    ],
)
def test_report_rejects_unsafe_trace_path(
    invalid_path: str,
) -> None:
    """完整轨迹只能使用docs目录下的相对JSON路径。"""

    report_data = make_report_data()
    trace_paths = list(report_data["trace_paths"])
    trace_paths[0] = invalid_path
    report_data["trace_paths"] = trace_paths

    with pytest.raises(
        ValidationError,
        match="安全相对JSON路径",
    ):
        MultimodalAgentEvaluationReport.model_validate(
            report_data
        )


def test_report_requires_at_least_three_trace_paths() -> None:
    """正式报告必须列出至少三份完整脱敏轨迹。"""

    with pytest.raises(
        ValidationError,
    ):
        MultimodalAgentEvaluationReport.model_validate(
            make_report_data(
                trace_paths=[
                    "docs/traces/one.json",
                    "docs/traces/two.json",
                ],
            )
        )


def test_report_rejects_wrong_judge_role() -> None:
    """没有Judge样本时报告角色必须标记为disabled。"""

    with pytest.raises(
        ValidationError,
        match="llm_judge_role",
    ):
        MultimodalAgentEvaluationReport.model_validate(
            make_report_data(
                llm_judge_role="auxiliary",
            )
        )


def test_report_rejects_duplicate_fix_issue() -> None:
    """同一修正问题不能在修正历史中重复记账。"""

    report_data = make_report_data()
    fix_history = list(report_data["fix_history"])
    fix_history.append(deepcopy(fix_history[0]))
    report_data["fix_history"] = fix_history

    with pytest.raises(
        ValidationError,
        match="fix_history中的issue不能重复",
    ):
        MultimodalAgentEvaluationReport.model_validate(
            report_data
        )


def test_parent_metric_mismatch_is_still_rejected() -> None:
    """第五周报告不能绕过第四周任务完成指标校验。"""

    report_data = make_report_data()
    metrics = deepcopy(report_data["metrics"])
    assert isinstance(metrics, dict)
    metrics["completed_task_count"] = 1
    metrics["task_completion_rate"] = 0.5
    report_data["metrics"] = metrics

    with pytest.raises(
        ValidationError,
        match="completed_task_count",
    ):
        MultimodalAgentEvaluationReport.model_validate(
            report_data
        )
