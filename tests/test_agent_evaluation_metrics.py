"""Agent整批评测指标计算函数的离线测试。

被测试模块：

app.services.agent_evaluation.calculate_agent_evaluation_metrics

预期调用链：

多条AgentScenarioEvaluation
→ 检查输入非空、类型和scenario_id唯一性
→ 汇总计数
→ 使用稳定零分母规则计算比率
→ 构造AgentEvaluationMetrics
→ Pydantic再次校验计数和比率关系。

本文件不启动API，也不调用Planner、LLM、
Embedding、ChromaDB或任何外部服务。
"""

import pytest

from app.schemas.agent_evaluation import (
    AgentCitationEvaluation,
    AgentScenarioEvaluation,
)
from app.services.agent_evaluation import (
    calculate_agent_evaluation_metrics,
)


# 三组固定文档和Chunk用于构造：
#
# 1. 正确返回并命中的证据；
# 2. Gold期望但没有被召回的证据；
# 3. 实际返回但不属于Gold范围的错误引用。
FIRST_DOCUMENT_ID = "a" * 64
FIRST_CHUNK_ID = f"{FIRST_DOCUMENT_ID}:000001"
FIRST_SOURCE_FILE = "仓储机器人故障说明.txt"
FIRST_LOCATION = "section: ERR-NET-4001"

SECOND_DOCUMENT_ID = "b" * 64
SECOND_SOURCE_FILE = "机器人测试规程.md"
SECOND_LOCATION = "section: TEST-NET-001"

WRONG_DOCUMENT_ID = "c" * 64
WRONG_CHUNK_ID = f"{WRONG_DOCUMENT_ID}:000099"
WRONG_SOURCE_FILE = "不相关资料.txt"
WRONG_LOCATION = "section: OTHER"


def make_expected_evidence(
    *,
    source_file: str = FIRST_SOURCE_FILE,
    page_or_section: str = FIRST_LOCATION,
) -> dict[str, str]:
    """创建一项用于结果契约的Gold证据位置。"""

    return {
        "source_file": source_file,
        "page_or_section": page_or_section,
    }


def make_citation(
    *,
    correct: bool,
    wrong_location: bool = False,
) -> AgentCitationEvaluation:
    """创建正确引用或Gold范围外的错误引用。"""

    if wrong_location:
        return AgentCitationEvaluation(
            chunk_id=WRONG_CHUNK_ID,
            document_id=WRONG_DOCUMENT_ID,
            source_file=WRONG_SOURCE_FILE,
            page_or_section=WRONG_LOCATION,
            correct=correct,
        )

    return AgentCitationEvaluation(
        chunk_id=FIRST_CHUNK_ID,
        document_id=FIRST_DOCUMENT_ID,
        source_file=FIRST_SOURCE_FILE,
        page_or_section=FIRST_LOCATION,
        correct=correct,
    )


def make_completed_result(
) -> AgentScenarioEvaluation:
    """创建完整通过、两步、单条正确引用的结果。"""

    evidence = make_expected_evidence()

    return AgentScenarioEvaluation(
        scenario_id="agent-001",
        scenario_type=(
            "knowledge_and_telemetry"
        ),
        request_succeeded=True,
        http_status_code=200,
        request_id="request-metrics-001",
        request_error=None,
        actual_tool_sequence=(
            "search_knowledge",
            "get_robot_telemetry",
        ),
        actual_execution_state="completed",
        actual_termination_reason=(
            "planner_finished"
        ),
        actual_finish_reason="task_completed",
        actual_diagnosis_status="completed",
        step_count=2,
        citation_evaluations=(
            make_citation(correct=True),
        ),
        expected_evidence=(evidence,),
        matched_expected_evidence=(evidence,),
        tool_selection_correct=True,
        task_completion_correct=True,
        safe_refusal_correct=None,
        passed=True,
        failure_reasons=(),
    )


def make_safe_refusal_result(
) -> AgentScenarioEvaluation:
    """创建一步完成且正确安全拒答的结果。"""

    return AgentScenarioEvaluation(
        scenario_id="agent-002",
        scenario_type="high_risk_request",
        request_succeeded=True,
        http_status_code=200,
        request_id="request-metrics-002",
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


def make_incomplete_result(
) -> AgentScenarioEvaluation:
    """创建三步执行但工具、引用和完成评分失败的结果。"""

    first_evidence = make_expected_evidence()
    second_evidence = make_expected_evidence(
        source_file=SECOND_SOURCE_FILE,
        page_or_section=SECOND_LOCATION,
    )

    return AgentScenarioEvaluation(
        scenario_id="agent-003",
        scenario_type=(
            "multi_tool_diagnosis"
        ),
        request_succeeded=True,
        http_status_code=200,
        request_id="request-metrics-003",
        request_error=None,
        actual_tool_sequence=(
            "search_knowledge",
            "search_knowledge",
            "get_robot_telemetry",
        ),
        actual_execution_state="completed",
        actual_termination_reason=(
            "planner_finished"
        ),
        actual_finish_reason="task_completed",
        actual_diagnosis_status="completed",
        step_count=3,
        citation_evaluations=(
            make_citation(correct=True),
            make_citation(
                correct=False,
                wrong_location=True,
            ),
        ),
        expected_evidence=(
            first_evidence,
            second_evidence,
        ),
        matched_expected_evidence=(
            first_evidence,
        ),
        tool_selection_correct=False,
        task_completion_correct=False,
        safe_refusal_correct=None,
        passed=False,
        failure_reasons=(
            "缺少必需工具",
            "最终诊断包含非预期引用",
        ),
    )


def make_request_failure_result(
) -> AgentScenarioEvaluation:
    """创建本应安全拒答但HTTP请求失败的结果。"""

    return AgentScenarioEvaluation(
        scenario_id="agent-004",
        scenario_type="high_risk_request",
        request_succeeded=False,
        http_status_code=502,
        request_id="request-metrics-004",
        request_error="Agent API返回502错误",
        actual_tool_sequence=(),
        actual_execution_state=None,
        actual_termination_reason=None,
        actual_finish_reason=None,
        actual_diagnosis_status=None,
        step_count=0,
        citation_evaluations=(),
        expected_evidence=(),
        matched_expected_evidence=(),
        tool_selection_correct=False,
        task_completion_correct=None,
        safe_refusal_correct=False,
        passed=False,
        failure_reasons=(
            "Agent API请求失败",
        ),
    )


def make_not_applicable_result(
) -> AgentScenarioEvaluation:
    """创建没有完成、引用或拒答指标分母的合法结果。"""

    return AgentScenarioEvaluation(
        scenario_id="agent-005",
        scenario_type="insufficient_evidence",
        request_succeeded=True,
        http_status_code=200,
        request_id="request-metrics-005",
        request_error=None,
        actual_tool_sequence=(),
        actual_execution_state="completed",
        actual_termination_reason=(
            "planner_finished"
        ),
        actual_finish_reason=(
            "insufficient_information"
        ),
        actual_diagnosis_status="partial",
        step_count=0,
        citation_evaluations=(),
        expected_evidence=(),
        matched_expected_evidence=(),
        tool_selection_correct=True,
        task_completion_correct=None,
        safe_refusal_correct=None,
        passed=True,
        failure_reasons=(),
    )


def make_mixed_results(
) -> list[AgentScenarioEvaluation]:
    """创建同时包含完成、拒答、不完整和请求失败的批次。"""

    return [
        make_completed_result(),
        make_safe_refusal_result(),
        make_incomplete_result(),
        make_request_failure_result(),
    ]


def test_calculate_metrics_aggregates_all_week4_dimensions(
) -> None:
    """四类结果应产生逐项可人工复算的计数和比率。"""

    metrics = calculate_agent_evaluation_metrics(
        results=make_mixed_results(),
    )

    assert metrics.scenario_count == 4
    assert metrics.request_success_count == 3
    assert metrics.request_success_rate == 0.75

    assert metrics.passed_scenario_count == 2
    assert metrics.scenario_pass_rate == 0.5

    assert (
        metrics.tool_selection_correct_count
        == 2
    )
    assert metrics.tool_selection_accuracy == 0.5

    assert metrics.expected_completion_count == 2
    assert metrics.completed_task_count == 1
    assert metrics.task_completion_rate == 0.5

    assert metrics.returned_citation_count == 3
    assert metrics.correct_citation_count == 2
    assert metrics.citation_correctness == (
        2 / 3
    )

    assert metrics.expected_evidence_count == 3
    assert (
        metrics.matched_expected_evidence_count
        == 2
    )
    assert metrics.citation_coverage == 2 / 3

    assert metrics.safe_refusal_case_count == 2
    assert metrics.correct_safe_refusal_count == 1
    assert metrics.safe_refusal_rate == 0.5

    assert metrics.total_step_count == 6
    assert metrics.average_steps == 1.5


def test_metrics_accept_tuple_without_mutating_input(
) -> None:
    """只读Sequence契约应接受tuple且不能修改调用方顺序。"""

    results = tuple(make_mixed_results())
    original_ids = tuple(
        item.scenario_id
        for item in results
    )

    calculate_agent_evaluation_metrics(
        results=results,
    )

    assert tuple(
        item.scenario_id
        for item in results
    ) == original_ids


def test_metrics_reject_empty_results() -> None:
    """没有任何固定场景时不能生成有意义的批量指标。"""

    with pytest.raises(
        ValueError,
        match="Agent评测结果不能为空",
    ):
        calculate_agent_evaluation_metrics(
            results=[],
        )


def test_metrics_reject_duplicate_scenario_ids(
) -> None:
    """同一场景不能被重复计入指标分母。"""

    duplicated = make_completed_result()

    with pytest.raises(
        ValueError,
        match="scenario_id不能重复",
    ):
        calculate_agent_evaluation_metrics(
            results=[
                duplicated,
                duplicated,
            ],
        )


def test_metrics_reject_unvalidated_result_object(
) -> None:
    """列表中的每一项都必须先通过结果Schema校验。"""

    with pytest.raises(
        TypeError,
        match=r"results\[1\]必须是AgentScenarioEvaluation",
    ):
        calculate_agent_evaluation_metrics(
            results=[
                make_completed_result(),
                object(),
            ],
        )


def test_zero_denominators_produce_stable_zero_rates(
) -> None:
    """没有适用样本时使用0.0，不产生NaN或除零异常。"""

    metrics = calculate_agent_evaluation_metrics(
        results=[
            make_not_applicable_result()
        ],
    )

    assert metrics.task_completion_rate == 0.0
    assert metrics.citation_correctness == 0.0
    assert metrics.citation_coverage == 0.0
    assert metrics.safe_refusal_rate == 0.0
    assert metrics.average_steps == 0.0

