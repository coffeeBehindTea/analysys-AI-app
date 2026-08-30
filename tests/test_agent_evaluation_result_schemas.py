"""Agent单场景评测结果契约的离线测试。

被测试模块：

1. AgentCitationEvaluation；
2. AgentScenarioEvaluation。

本文件验证一次API评测执行结束后，结果中保存的
请求状态、工具轨迹摘要、引用位置和评分标记是否自洽。
"""

import pytest

from pydantic import (
    ValidationError,
)

from app.schemas.agent_evaluation import (
    AgentCitationEvaluation,
    AgentScenarioEvaluation,
)


# 固定评测场景和请求ID用于成功结果。
TEST_SCENARIO_ID = "agent-001"
TEST_REQUEST_ID = "request-agent-eval-001"


# 固定SHA-256形式文档ID和Chunk ID。
#
# AgentCitationEvaluation不保存excerpt，
# 但仍保留Chunk ID以便定位真实工具结果。
TEST_DOCUMENT_ID = "a" * 64
TEST_CHUNK_ID = (
    f"{TEST_DOCUMENT_ID}:000003"
)


TEST_SOURCE_FILE = "仓储机器人故障说明.txt"
TEST_PAGE_OR_SECTION = "section: ERR-NET-4001"


def make_expected_evidence() -> dict[str, str]:
    """创建一条Gold预期证据位置。"""

    return {
        "source_file": TEST_SOURCE_FILE,
        "page_or_section": (
            TEST_PAGE_OR_SECTION
        ),
    }


def make_citation_evaluation(
    *,
    chunk_id: str = TEST_CHUNK_ID,
    correct: bool = True,
) -> dict[str, object]:
    """创建一条不包含原文的引用评分记录。"""

    return {
        "chunk_id": chunk_id,
        "document_id": TEST_DOCUMENT_ID,
        "source_file": TEST_SOURCE_FILE,
        "page_or_section": (
            TEST_PAGE_OR_SECTION
        ),
        "correct": correct,
    }


def make_success_result_data() -> dict[str, object]:
    """创建一次成功完成的两工具场景结果。"""

    evidence = make_expected_evidence()

    return {
        "scenario_id": TEST_SCENARIO_ID,
        "scenario_type": (
            "knowledge_and_telemetry"
        ),
        "request_succeeded": True,
        "http_status_code": 200,
        "request_id": TEST_REQUEST_ID,
        "request_error": None,
        "actual_tool_sequence": [
            "search_knowledge",
            "get_robot_telemetry",
        ],
        "actual_execution_state": (
            "completed"
        ),
        "actual_termination_reason": (
            "planner_finished"
        ),
        "actual_finish_reason": (
            "task_completed"
        ),
        "actual_diagnosis_status": (
            "completed"
        ),
        "step_count": 2,
        "citation_evaluations": [
            make_citation_evaluation()
        ],
        "expected_evidence": [evidence],
        "matched_expected_evidence": [
            dict(evidence)
        ],
        "tool_selection_correct": True,
        "task_completion_correct": True,
        "safe_refusal_correct": None,
        "passed": True,
        "failure_reasons": [],
    }


def test_valid_success_result_preserves_audit_fields(
) -> None:
    """成功结果应保存工具顺序、状态和脱敏引用位置。"""

    result = AgentScenarioEvaluation.model_validate(
        make_success_result_data()
    )

    assert result.request_succeeded is True
    assert result.actual_tool_sequence == (
        "search_knowledge",
        "get_robot_telemetry",
    )
    assert result.step_count == 2
    assert (
        result.citation_evaluations[0]
        .chunk_id
        == TEST_CHUNK_ID
    )
    assert result.failure_reasons == ()


def test_citation_evaluation_rejects_extra_content(
) -> None:
    """引用评分不能保存完整Chunk正文等未声明字段。"""

    citation = make_citation_evaluation()
    citation["excerpt"] = (
        "不应进入Agent评测汇总的完整原文"
    )

    with pytest.raises(
        ValidationError,
        match="Extra inputs are not permitted",
    ):
        AgentCitationEvaluation.model_validate(
            citation
        )


@pytest.mark.parametrize(
    ("replacement", "expected_message"),
    [
        (
            {"http_status_code": 502},
            (
                "成功请求必须使用"
                "2xx HTTP状态码"
            ),
        ),
        (
            {"request_id": None},
            "成功请求必须包含request_id",
        ),
        (
            {
                "request_error": (
                    "不应和成功结果同时出现"
                )
            },
            "成功请求不能包含request_error",
        ),
        (
            {"actual_execution_state": None},
            (
                "成功请求必须包含"
                "actual_execution_state"
            ),
        ),
        (
            {"actual_termination_reason": None},
            (
                "成功请求必须包含"
                "actual_termination_reason"
            ),
        ),
        (
            {"actual_diagnosis_status": None},
            (
                "成功请求必须包含"
                "actual_diagnosis_status"
            ),
        ),
    ],
)
def test_success_result_requires_complete_response_metadata(
    replacement: dict[str, object],
    expected_message: str,
) -> None:
    """HTTP成功结果必须包含可评分的响应元数据。"""

    data = make_success_result_data()
    data.update(replacement)

    with pytest.raises(
        ValidationError,
        match=expected_message,
    ):
        AgentScenarioEvaluation.model_validate(
            data
        )


def test_valid_request_failure_contains_no_fake_observation(
) -> None:
    """API失败应被记录，但不能伪造Agent执行观察。"""

    result = AgentScenarioEvaluation(
        scenario_id=TEST_SCENARIO_ID,
        scenario_type=(
            "tool_failure"
        ),
        request_succeeded=False,
        http_status_code=502,
        request_id=TEST_REQUEST_ID,
        request_error=(
            "知识库API返回错误状态"
        ),
        actual_tool_sequence=(),
        actual_execution_state=None,
        actual_termination_reason=None,
        actual_finish_reason=None,
        actual_diagnosis_status=None,
        step_count=0,
        citation_evaluations=(),
        expected_evidence=(
            make_expected_evidence(),
        ),
        matched_expected_evidence=(),
        tool_selection_correct=False,
        task_completion_correct=False,
        safe_refusal_correct=None,
        passed=False,
        failure_reasons=(
            "Agent API请求失败",
        ),
    )

    assert result.request_succeeded is False
    assert result.http_status_code == 502
    assert result.actual_execution_state is None
    assert result.step_count == 0


@pytest.mark.parametrize(
    ("replacement", "expected_message"),
    [
        (
            {"request_error": None},
            "失败请求必须包含request_error",
        ),
        (
            {
                "actual_tool_sequence": [
                    "search_knowledge"
                ],
                "step_count": 1,
            },
            (
                "失败请求不能包含"
                "Agent工具观察"
            ),
        ),
        (
            {
                "actual_execution_state": (
                    "aborted"
                )
            },
            (
                "失败请求不能包含"
                "Agent响应状态"
            ),
        ),
        (
            {
                "citation_evaluations": [
                    make_citation_evaluation()
                ]
            },
            (
                "失败请求不能包含"
                "引用评分"
            ),
        ),
    ],
)
def test_request_failure_rejects_fake_response_data(
    replacement: dict[str, object],
    expected_message: str,
) -> None:
    """网络或HTTP失败时不能保存并不存在的Agent结果。"""

    data = make_success_result_data()
    data.update({
        "request_succeeded": False,
        "http_status_code": 502,
        "request_error": "上游请求失败",
        "actual_tool_sequence": [],
        "actual_execution_state": None,
        "actual_termination_reason": None,
        "actual_finish_reason": None,
        "actual_diagnosis_status": None,
        "step_count": 0,
        "citation_evaluations": [],
        "matched_expected_evidence": [],
        "tool_selection_correct": False,
        "task_completion_correct": False,
        "passed": False,
        "failure_reasons": [
            "Agent API请求失败"
        ],
    })
    data.update(replacement)

    with pytest.raises(
        ValidationError,
        match=expected_message,
    ):
        AgentScenarioEvaluation.model_validate(
            data
        )


def test_completed_execution_requires_finish_reason(
) -> None:
    """completed执行必须保留Planner正常结束原因。"""

    data = make_success_result_data()
    data["actual_finish_reason"] = None

    with pytest.raises(
        ValidationError,
        match=(
            "completed执行必须包含"
            "actual_finish_reason"
        ),
    ):
        AgentScenarioEvaluation.model_validate(
            data
        )


def test_aborted_execution_rejects_finish_reason(
) -> None:
    """aborted执行不能伪装成Planner正常完成。"""

    data = make_success_result_data()
    data.update({
        "actual_execution_state": "aborted",
        "actual_termination_reason": (
            "planner_timeout"
        ),
        "actual_finish_reason": (
            "task_completed"
        ),
        "actual_diagnosis_status": (
            "abstained"
        ),
        "task_completion_correct": False,
        "passed": False,
        "failure_reasons": [
            "Agent执行状态不符合预期"
        ],
    })

    with pytest.raises(
        ValidationError,
        match=(
            "aborted执行不能包含"
            "actual_finish_reason"
        ),
    ):
        AgentScenarioEvaluation.model_validate(
            data
        )


def test_step_count_matches_actual_tool_sequence(
) -> None:
    """步骤总数必须等于实际工具轨迹长度。"""

    data = make_success_result_data()
    data["step_count"] = 1

    with pytest.raises(
        ValidationError,
        match=(
            "step_count必须等于"
            "actual_tool_sequence长度"
        ),
    ):
        AgentScenarioEvaluation.model_validate(
            data
        )


def test_result_rejects_duplicate_citation_chunk_ids(
) -> None:
    """同一个Chunk不能在引用评分中被重复计数。"""

    data = make_success_result_data()
    citations = list(
        data["citation_evaluations"]
    )
    citations.append(dict(citations[0]))
    data["citation_evaluations"] = citations

    with pytest.raises(
        ValidationError,
        match=(
            "citation_evaluations中的"
            "chunk_id不能重复"
        ),
    ):
        AgentScenarioEvaluation.model_validate(
            data
        )


def test_matched_evidence_must_be_expected(
) -> None:
    """命中的Gold位置必须来自当前场景预期证据。"""

    data = make_success_result_data()
    data["matched_expected_evidence"] = [
        {
            "source_file": "其他文件.pdf",
            "page_or_section": "page: 99",
        }
    ]

    with pytest.raises(
        ValidationError,
        match=(
            "matched_expected_evidence必须是"
            "expected_evidence的子集"
        ),
    ):
        AgentScenarioEvaluation.model_validate(
            data
        )


def test_citation_correct_flag_matches_expected_location(
) -> None:
    """引用correct不能与Gold位置匹配结果相矛盾。"""

    data = make_success_result_data()
    data["citation_evaluations"] = [
        make_citation_evaluation(
            correct=False
        )
    ]
    data["matched_expected_evidence"] = []

    with pytest.raises(
        ValidationError,
        match=(
            "citation.correct必须与"
            "预期证据位置一致"
        ),
    ):
        AgentScenarioEvaluation.model_validate(
            data
        )


def test_matched_evidence_equals_correct_citation_locations(
) -> None:
    """matched列表必须正好覆盖正确引用命中的位置。"""

    data = make_success_result_data()
    data["matched_expected_evidence"] = []

    with pytest.raises(
        ValidationError,
        match=(
            "matched_expected_evidence必须等于"
            "正确引用命中的预期位置"
        ),
    ):
        AgentScenarioEvaluation.model_validate(
            data
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "expected_evidence",
        "matched_expected_evidence",
    ],
)
def test_result_rejects_duplicate_evidence_locations(
    field_name: str,
) -> None:
    """预期位置和已命中位置都不能重复计数。"""

    data = make_success_result_data()
    locations = list(data[field_name])
    locations.append(dict(locations[0]))
    data[field_name] = locations

    with pytest.raises(
        ValidationError,
        match=f"{field_name}不能包含重复位置",
    ):
        AgentScenarioEvaluation.model_validate(
            data
        )


def test_true_task_completion_requires_completed_response(
) -> None:
    """任务完成评分为真时，实际响应必须真的完整完成。"""

    data = make_success_result_data()
    data.update({
        "actual_execution_state": "aborted",
        "actual_termination_reason": (
            "planner_timeout"
        ),
        "actual_finish_reason": None,
        "actual_diagnosis_status": (
            "abstained"
        ),
    })

    with pytest.raises(
        ValidationError,
        match=(
            "task_completion_correct为True时"
            "实际结果必须完整完成"
        ),
    ):
        AgentScenarioEvaluation.model_validate(
            data
        )


def test_true_safe_refusal_requires_abstained_response(
) -> None:
    """安全拒答评分为真时，诊断必须实际拒答。"""

    data = make_success_result_data()
    data.update({
        "task_completion_correct": None,
        "safe_refusal_correct": True,
    })

    with pytest.raises(
        ValidationError,
        match=(
            "safe_refusal_correct为True时"
            "诊断必须abstained"
        ),
    ):
        AgentScenarioEvaluation.model_validate(
            data
        )


@pytest.mark.parametrize(
    ("passed", "failure_reasons", "message"),
    [
        (
            True,
            ["不应和通过结果同时存在"],
            "passed结果不能包含failure_reasons",
        ),
        (
            False,
            [],
            "未通过结果必须包含failure_reasons",
        ),
    ],
)
def test_passed_flag_matches_failure_reasons(
    passed: bool,
    failure_reasons: list[str],
    message: str,
) -> None:
    """总通过标记必须与失败原因列表保持一致。"""

    data = make_success_result_data()
    data["passed"] = passed
    data["failure_reasons"] = (
        failure_reasons
    )

    with pytest.raises(
        ValidationError,
        match=message,
    ):
        AgentScenarioEvaluation.model_validate(
            data
        )


def test_result_is_frozen_after_validation() -> None:
    """已评分结果不能在汇总阶段被原地篡改。"""

    result = AgentScenarioEvaluation.model_validate(
        make_success_result_data()
    )

    with pytest.raises(
        ValidationError,
        match="Instance is frozen",
    ):
        result.step_count = 99
