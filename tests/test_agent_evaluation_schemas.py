"""Agent评测场景数据契约的离线测试。

本文件只测试评测数据本身是否自洽，不启动FastAPI，
不调用真实Planner、LLM、Embedding或ChromaDB。

被测试模块：

app.schemas.agent_evaluation.AgentEvaluationScenario

预期流程：

原始字典
→ Pydantic字段校验
→ 跨字段关系校验
→ 生成不可变的AgentEvaluationScenario。
"""

import pytest

from pydantic import (
    ValidationError,
)

from app.schemas.agent_evaluation import (
    AgentEvaluationScenario,
)


# 固定场景编号用于测试正常路径。
#
# agent-前缀能与Week 2/3的q001等
# RAG Gold Question编号明确区分。
TEST_SCENARIO_ID = "agent-001"


# 固定来源位置用于构造预期证据。
#
# Agent评测沿用Gold Question的
# source_file + page_or_section定位方式，
# 不把可能随重新摄取变化的Chunk ID
# 写死在评测场景中。
TEST_SOURCE_FILE = "仓储机器人故障说明.txt"
TEST_PAGE_OR_SECTION = "section: ERR-NET-4001"


def make_scenario_data() -> dict[str, object]:
    """创建一份字段完整、关系合法的场景字典。

    每个异常测试都会只替换一个字段，
    这样可以确认失败确实来自目标约束，
    而不是其他无关字段。
    """

    return {
        "scenario_id": TEST_SCENARIO_ID,
        "name": "网络恢复后核对诊断与模拟遥测",
        "scenario_type": (
            "knowledge_and_telemetry"
        ),
        "request": {
            "robot_id": "robot-001",
            "symptom": (
                "网络恢复后机器人仍处于暂停状态"
            ),
            "log_excerpt": (
                "ERR-NET-4001 heartbeat timeout"
            ),
            "task_goal": (
                "核对故障依据和当前模拟遥测"
            ),
        },
        "required_tools": [
            "search_knowledge",
            "get_robot_telemetry",
        ],
        "allowed_tools": [
            "search_knowledge",
            "get_robot_telemetry",
        ],
        "forbidden_tools": [
            "run_shell",
        ],
        "expected_execution_states": [
            "completed",
        ],
        "expected_termination_reasons": [
            "planner_finished",
        ],
        "expected_finish_reasons": [
            "task_completed",
        ],
        "expected_diagnosis_statuses": [
            "completed",
        ],
        "expected_evidence": [
            {
                "source_file": (
                    TEST_SOURCE_FILE
                ),
                "page_or_section": (
                    TEST_PAGE_OR_SECTION
                ),
            },
        ],
        "expects_task_completion": True,
        "expects_safe_refusal": False,
        "tags": [
            "multi_tool",
            "telemetry",
            "knowledge",
        ],
        "notes": (
            "用于验证知识检索和模拟遥测的"
            "两步以上调用链。"
        ),
    }


def test_valid_scenario_normalizes_nested_contracts(
) -> None:
    """合法字典应转换成完整的不可变评测场景。

    被测试流程：

    1. AgentEvaluationScenario校验顶层字段；
    2. request转换成AgentDiagnosisRequest；
    3. expected_evidence转换成ExpectedEvidence；
    4. JSON列表转换成内部tuple；
    5. 返回字段关系一致的场景对象。
    """

    scenario = (
        AgentEvaluationScenario.model_validate(
            make_scenario_data()
        )
    )

    assert (
        scenario.scenario_id
        == TEST_SCENARIO_ID
    )
    assert scenario.request.robot_id == (
        "robot-001"
    )
    assert scenario.required_tools == (
        "search_knowledge",
        "get_robot_telemetry",
    )
    assert (
        scenario.expected_evidence[0]
        .source_file
        == TEST_SOURCE_FILE
    )


def test_scenario_rejects_invalid_id() -> None:
    """场景编号必须使用agent-三位数字格式。"""

    data = make_scenario_data()
    data["scenario_id"] = "q001"

    with pytest.raises(
        ValidationError,
        match="scenario_id",
    ):
        AgentEvaluationScenario.model_validate(
            data
        )


def test_scenario_rejects_extra_field() -> None:
    """拼错或未声明字段不能静默进入评测集。"""

    data = make_scenario_data()
    data["expected_tool"] = (
        "search_knowledge"
    )

    with pytest.raises(
        ValidationError,
        match="Extra inputs are not permitted",
    ):
        AgentEvaluationScenario.model_validate(
            data
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "required_tools",
        "allowed_tools",
        "forbidden_tools",
        "expected_execution_states",
        "expected_termination_reasons",
        "expected_finish_reasons",
        "expected_diagnosis_statuses",
        "tags",
    ],
)
def test_scenario_rejects_duplicate_values(
    field_name: str,
) -> None:
    """集合语义字段不能包含重复值。

    参数化测试会用同一个测试函数依次验证八个字段，
    避免每个字段复制一份相同测试逻辑。
    """

    data = make_scenario_data()
    values = list(data[field_name])
    values.append(values[0])
    data[field_name] = values

    with pytest.raises(
        ValidationError,
        match=f"{field_name}不能包含重复项",
    ):
        AgentEvaluationScenario.model_validate(
            data
        )


def test_required_tools_must_be_allowed() -> None:
    """必需工具不能被场景自己的允许列表排除。"""

    data = make_scenario_data()
    data["allowed_tools"] = [
        "search_knowledge",
    ]

    with pytest.raises(
        ValidationError,
        match=(
            "required_tools必须是"
            "allowed_tools的子集"
        ),
    ):
        AgentEvaluationScenario.model_validate(
            data
        )


def test_allowed_and_forbidden_tools_cannot_overlap(
) -> None:
    """同一个工具不能同时被允许又被禁止。"""

    data = make_scenario_data()
    data["forbidden_tools"] = [
        "search_knowledge",
    ]

    with pytest.raises(
        ValidationError,
        match=(
            "allowed_tools和"
            "forbidden_tools不能重叠"
        ),
    ):
        AgentEvaluationScenario.model_validate(
            data
        )


def test_scenario_rejects_duplicate_evidence_locations(
) -> None:
    """同一来源位置不能重复贡献两份预期证据。"""

    data = make_scenario_data()
    evidence = list(
        data["expected_evidence"]
    )
    evidence.append(dict(evidence[0]))
    data["expected_evidence"] = evidence

    with pytest.raises(
        ValidationError,
        match=(
            "expected_evidence不能包含"
            "重复位置"
        ),
    ):
        AgentEvaluationScenario.model_validate(
            data
        )


@pytest.mark.parametrize(
    ("replacement", "expected_message"),
    [
        (
            {
                "expected_execution_states": [
                    "aborted"
                ],
                "expected_termination_reasons": [
                    "planner_timeout"
                ],
                "expected_finish_reasons": [],
            },
            (
                "完成型场景必须允许"
                "completed执行状态"
            ),
        ),
        (
            {
                "expected_diagnosis_statuses": [
                    "partial"
                ],
            },
            (
                "完成型场景必须允许"
                "completed诊断状态"
            ),
        ),
        (
            {
                "expected_finish_reasons": [
                    "human_review_required"
                ],
            },
            (
                "完成型场景必须包含"
                "task_completed"
            ),
        ),
        (
            {
                "expected_termination_reasons": [
                    "planner_timeout"
                ],
            },
            (
                "完成型场景必须包含"
                "planner_finished"
            ),
        ),
        (
            {
                "expected_evidence": [],
            },
            (
                "完成型场景必须包含"
                "预期证据"
            ),
        ),
    ],
)
def test_completion_scenario_requires_complete_contract(
    replacement: dict[str, object],
    expected_message: str,
) -> None:
    """要求完成的场景必须同时声明完整结果口径。"""

    data = make_scenario_data()
    data.update(replacement)

    with pytest.raises(
        ValidationError,
        match=expected_message,
    ):
        AgentEvaluationScenario.model_validate(
            data
        )


@pytest.mark.parametrize(
    ("replacement", "expected_message"),
    [
        (
            {
                "expects_safe_refusal": True,
            },
            (
                "安全拒答场景不能同时"
                "要求任务完成"
            ),
        ),
        (
            {
                "expects_task_completion": False,
                "expects_safe_refusal": True,
                "expected_diagnosis_statuses": [
                    "partial"
                ],
                "expected_finish_reasons": [
                    "human_review_required"
                ],
            },
            (
                "安全拒答场景必须允许"
                "abstained诊断"
            ),
        ),
        (
            {
                "expects_task_completion": False,
                "expects_safe_refusal": True,
                "expected_diagnosis_statuses": [
                    "abstained"
                ],
                "expected_finish_reasons": [
                    "insufficient_information"
                ],
            },
            (
                "安全拒答场景不能预期"
                "最终诊断证据"
            ),
        ),
    ],
)
def test_safe_refusal_scenario_requires_abstention_contract(
    replacement: dict[str, object],
    expected_message: str,
) -> None:
    """安全拒答必须与完成标记、状态和证据保持一致。"""

    data = make_scenario_data()
    data.update(replacement)

    with pytest.raises(
        ValidationError,
        match=expected_message,
    ):
        AgentEvaluationScenario.model_validate(
            data
        )


def test_aborted_only_scenario_rejects_finish_reasons(
) -> None:
    """只允许aborted的场景不能期待Planner正常finish。"""

    data = make_scenario_data()
    data.update({
        "expected_execution_states": [
            "aborted"
        ],
        "expected_termination_reasons": [
            "tool_failure_limit_reached"
        ],
        "expected_finish_reasons": [
            "insufficient_information"
        ],
        "expected_diagnosis_statuses": [
            "abstained"
        ],
        "expected_evidence": [],
        "expects_task_completion": False,
        "expects_safe_refusal": True,
    })

    with pytest.raises(
        ValidationError,
        match=(
            "不允许completed执行时不能设置"
            "expected_finish_reasons"
        ),
    ):
        AgentEvaluationScenario.model_validate(
            data
        )


def test_completed_scenario_requires_finish_reasons(
) -> None:
    """允许completed时必须说明哪些finish_reason可接受。"""

    data = make_scenario_data()
    data.update({
        "expected_finish_reasons": [],
        "expects_task_completion": False,
    })

    with pytest.raises(
        ValidationError,
        match=(
            "允许completed执行时必须设置"
            "expected_finish_reasons"
        ),
    ):
        AgentEvaluationScenario.model_validate(
            data
        )


def test_scenario_is_frozen_after_validation() -> None:
    """评测运行期间不能原地修改Gold场景。"""

    scenario = (
        AgentEvaluationScenario.model_validate(
            make_scenario_data()
        )
    )

    with pytest.raises(
        ValidationError,
        match="Instance is frozen",
    ):
        scenario.name = "运行中被修改"
