"""多模态Agent正式报告组装服务的离线测试。

被测试模块：app.services.agent_evaluation。

预期调用链：

逐场景MultimodalAgentScenarioEvaluation
→ 指标、类别和失败原因确定性汇总
→ build_multimodal_agent_evaluation_report()
→ Pydantic执行跨字段一致性校验
→ 返回MultimodalAgentEvaluationReport。

本文件不发送HTTP请求，不调用Agent、Vision、LLM、
Embedding或ChromaDB，也不向docs目录写报告。
"""

from pydantic import ValidationError
import pytest

from app.schemas.multimodal_agent_evaluation import (
    MultimodalAgentScenarioEvaluation,
    MultimodalEvaluationFixRecord,
)
from app.services.agent_evaluation import (
    build_multimodal_agent_evaluation_report,
    calculate_multimodal_failure_reason_counts,
)


# 三份路径满足正式报告“至少三份完整脱敏轨迹”的契约。
TEST_TRACE_PATHS = (
    "docs/traces/multimodal-agent-001.json",
    "docs/traces/multimodal-agent-002.json",
    "docs/traces/multimodal-agent-003.json",
)

# 修正历史属于正式报告必填审计信息。
TEST_FIX_HISTORY = (
    MultimodalEvaluationFixRecord(
        issue="视觉字段匹配遗漏",
        change="增加确定性规范化字段匹配",
        verification="离线评分和报告回归测试通过",
    ),
)


def make_result(
    *,
    index: int,
    category: str = "normal_image",
    passed: bool = True,
    failure_reasons: tuple[str, ...] = (),
    visual_matched: bool = True,
    source_labels_correct: bool = True,
    llm_judge_used: bool = False,
) -> MultimodalAgentScenarioEvaluation:
    """创建一条经过全部结果契约校验的脱敏评分结果。"""

    scenario_id = (
        f"multimodal-agent-{index:03d}"
    )

    actual_sources = (
        (
            "user_report",
            "vision_model",
        )
        if source_labels_correct
        else ("user_report",)
    )

    return MultimodalAgentScenarioEvaluation(
        # 第四周基础Agent评分字段。
        scenario_id=scenario_id,
        scenario_type="multi_tool_diagnosis",
        request_succeeded=True,
        http_status_code=200,
        request_id=f"request-{scenario_id}",
        request_error=None,
        actual_tool_sequence=(
            "analyze_robot_image",
        ),
        actual_execution_state="completed",
        actual_termination_reason=(
            "planner_finished"
        ),
        actual_finish_reason="task_completed",
        actual_diagnosis_status="completed",
        step_count=1,
        citation_evaluations=(),
        expected_evidence=(),
        matched_expected_evidence=(),
        tool_selection_correct=True,
        task_completion_correct=True,
        safe_refusal_correct=None,
        passed=passed,
        failure_reasons=failure_reasons,

        # 第五周多模态评分字段。
        category=category,
        expected_vision_call=True,
        actual_vision_call_count=1,
        actual_vision_statuses=(
            "completed",
        ),
        vision_tool_selection_correct=True,
        actual_visual_observations=(
            "面板显示ERR-DEMO-1001",
        ),
        visual_observation_evaluations=(
            {
                "expected_observation": (
                    "面板显示ERR-DEMO-1001"
                ),
                "matched": visual_matched,
                "matched_actual_observation": (
                    "面板显示ERR-DEMO-1001"
                    if visual_matched
                    else None
                ),
                "match_method": (
                    "normalized_exact"
                    if visual_matched
                    else "not_matched"
                ),
            },
        ),
        matched_visual_observation_count=(
            1 if visual_matched else 0
        ),
        visual_observation_accuracy=(
            1.0 if visual_matched else 0.0
        ),
        expected_information_sources=(
            "user_report",
            "vision_model",
        ),
        actual_information_sources=(
            actual_sources
        ),
        source_labels_correct=(
            source_labels_correct
        ),
        safety_evaluations=(
            {
                "requirement": (
                    "preserve_source_labels"
                ),
                "passed": True,
                "failure_reason": None,
            },
        ),
        latency_ms=float(index * 100),
        cost_status="unavailable",
        estimated_cost_usd=None,
        cost_note="测试Provider没有公开usage",
        llm_judge_used=llm_judge_used,
        llm_judge_passed=(
            False if llm_judge_used else None
        ),
        llm_judge_note=(
            "辅助Judge认为表达可进一步改进"
            if llm_judge_used
            else None
        ),
    )


def make_results(
    *,
    use_auxiliary_judge: bool = True,
) -> tuple[
    MultimodalAgentScenarioEvaluation,
    ...,
]:
    """创建一条通过和两条失败的正式报告输入。"""

    return (
        make_result(
            index=1,
            llm_judge_used=(
                use_auxiliary_judge
            ),
        ),
        make_result(
            index=2,
            category="noisy_or_low_quality",
            passed=False,
            failure_reasons=(
                "共同失败原因",
                "视觉字段缺失",
            ),
            visual_matched=False,
        ),
        make_result(
            index=3,
            category="missing_or_unanswerable",
            passed=False,
            failure_reasons=(
                "共同失败原因",
                "来源标签不一致",
            ),
            source_labels_correct=False,
        ),
    )


def build_report(
    *,
    results: tuple[
        MultimodalAgentScenarioEvaluation,
        ...,
    ] | None = None,
    trace_paths: object = TEST_TRACE_PATHS,
    fix_history: object = TEST_FIX_HISTORY,
    scenario_source: object = (
        "data/eval/multimodal_scenarios.jsonl"
    ),
):
    """用固定公开元数据调用真实报告组装器。"""

    return build_multimodal_agent_evaluation_report(
        results=(
            make_results()
            if results is None
            else results
        ),
        scenario_source=scenario_source,  # type: ignore[arg-type]
        api_url=(
            "http://127.0.0.1:8000/"
            "api/v1/agent/diagnose"
        ),
        planner_model="test-planner-model",
        planner_prompt_version=(
            "agent-tool-calling-v2"
        ),
        embedding_model="test-embedding-model",
        collection_name="test-collection",
        vision_model="test-vision-model",
        vision_prompt_version=(
            "robot-vision-observation-v1"
        ),
        trace_paths=trace_paths,  # type: ignore[arg-type]
        fix_history=fix_history,  # type: ignore[arg-type]
    )


def test_failure_reason_counts_use_frequency_then_text_order() -> None:
    """失败原因应先按次数降序，再按文本稳定排序。"""

    counts = (
        calculate_multimodal_failure_reason_counts(
            results=make_results()
        )
    )

    assert tuple(
        (item.reason, item.scenario_count)
        for item in counts
    ) == (
        ("共同失败原因", 2),
        ("来源标签不一致", 1),
        ("视觉字段缺失", 1),
    )


def test_builder_derives_metrics_categories_failures_and_role() -> None:
    """组装器应从结果自动推导所有正式报告区块。"""

    report = build_report()

    assert report.evaluation_version == (
        "multimodal-agent-evaluation-v1"
    )
    assert report.metrics.scenario_count == 3
    assert report.metrics.passed_scenario_count == 1
    assert report.metrics.llm_judge_evaluated_count == 1
    assert report.llm_judge_role == "auxiliary"
    assert tuple(
        item.category
        for item in report.category_summaries
    ) == (
        "normal_image",
        "noisy_or_low_quality",
        "missing_or_unanswerable",
    )
    assert report.failure_reason_counts[0].reason == (
        "共同失败原因"
    )
    assert "generated_at" not in report.model_dump()


def test_builder_derives_disabled_judge_role() -> None:
    """没有任何Judge样本时报告角色必须自动设为disabled。"""

    report = build_report(
        results=make_results(
            use_auxiliary_judge=False
        )
    )

    assert report.llm_judge_role == "disabled"
    assert report.metrics.llm_judge_evaluated_count == 0
    assert report.metrics.llm_judge_pass_rate is None


def test_builder_rejects_blank_metadata_before_report_creation() -> None:
    """空白场景来源必须在进入Pydantic报告前被拒绝。"""

    with pytest.raises(
        ValueError,
        match="scenario_source不能为空",
    ):
        build_report(
            scenario_source="   "
        )


def test_builder_rejects_trace_path_string() -> None:
    """单个字符串不能被当成由字符组成的轨迹路径序列。"""

    with pytest.raises(
        TypeError,
        match="trace_paths必须是字符串路径序列",
    ):
        build_report(
            trace_paths=(
                "docs/traces/trace.json"
            )
        )


def test_builder_rejects_non_string_trace_item() -> None:
    """轨迹序列中的每个元素都必须是字符串路径。"""

    with pytest.raises(
        TypeError,
        match=r"trace_paths\[1\]必须是字符串",
    ):
        build_report(
            trace_paths=(
                TEST_TRACE_PATHS[0],
                object(),
                TEST_TRACE_PATHS[2],
            )
        )


def test_builder_rejects_wrong_fix_history_item() -> None:
    """未经过修正记录Schema校验的普通对象不能进入报告。"""

    with pytest.raises(
        TypeError,
        match=r"fix_history\[0\].*MultimodalEvaluationFixRecord",
    ):
        build_report(
            fix_history=(object(),)
        )


def test_builder_rejects_duplicate_result_ids() -> None:
    """重复场景不能被重复统计到正式报告。"""

    item = make_results()[0]

    with pytest.raises(
        ValueError,
        match="scenario_id不能重复",
    ):
        build_report(
            results=(item, item, item)
        )


def test_report_schema_requires_three_trace_paths() -> None:
    """少于三份轨迹时由最终报告契约拒绝组装。"""

    with pytest.raises(
        ValidationError,
        match="trace_paths",
    ):
        build_report(
            trace_paths=TEST_TRACE_PATHS[:2]
        )


def test_report_schema_rejects_unsafe_trace_path() -> None:
    """包含父目录跳转的轨迹路径不能进入可审计报告。"""

    with pytest.raises(
        ValidationError,
        match="安全相对JSON路径",
    ):
        build_report(
            trace_paths=(
                "docs/traces/one.json",
                "docs/../secret.json",
                "docs/traces/three.json",
            )
        )


def test_failure_reason_counts_can_be_empty() -> None:
    """全部场景通过时失败原因汇总应是空元组。"""

    results = tuple(
        make_result(index=index)
        for index in range(1, 4)
    )

    assert (
        calculate_multimodal_failure_reason_counts(
            results=results
        )
        == ()
    )
