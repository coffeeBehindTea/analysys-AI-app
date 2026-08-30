"""AgentRunner最终运行结果数据契约的离线测试。

本文件只验证AgentRunResult字段之间的关系：

1. 正常完成必须来自Planner主动结束；
2. 正常完成必须保留Planner的结束原因和草稿；
3. 安全中止不能伪装成Planner正常完成；
4. 中止结果只能保留已经实际完成的工具历史；
5. 未声明字段和过长历史必须被拒绝。

测试不调用Planner、工具、LLM或任何外部服务。
"""

import pytest

from pydantic import (
    ValidationError,
)

from app.schemas.agent import (
    ToolCall,
    ToolExecutionResult,
)
from app.schemas.agent_planning import (
    AgentToolInteraction,
)
from app.schemas.agent_runtime import (
    AgentRunResult,
)


def make_interaction(
    *,
    step_number: int = 1,
    call_id: str = "call_001",
) -> AgentToolInteraction:
    """创建一条成功且请求、结果能够正确配对的工具历史。"""

    tool_call = ToolCall(
        call_id=call_id,
        tool_name="search_knowledge",
        arguments={
            "query": "ERR-NET-4001",
        },
    )

    result = ToolExecutionResult(
        call_id=call_id,
        tool_name="search_knowledge",
        status="success",
        output={
            "evidence_ids": [
                "document-id:000001",
            ],
        },
        duration_ms=12.5,
    )

    return AgentToolInteraction(
        step_number=step_number,
        tool_call=tool_call,
        result=result,
    )


def test_completed_result_accepts_planner_finish_fields(
) -> None:
    """Planner主动结束时应形成包含最终草稿的completed结果。"""

    result = AgentRunResult(
        state="completed",
        termination_reason="planner_finished",
        termination_message=(
            "Agent规划正常结束"
        ),
        interactions=(
            make_interaction(),
        ),
        finish_reason="task_completed",
        final_message=(
            "已完成知识检索并形成诊断草稿。"
        ),
    )

    assert result.state == "completed"
    assert (
        result.termination_reason
        == "planner_finished"
    )
    assert result.finish_reason == (
        "task_completed"
    )
    assert len(result.interactions) == 1


def test_insufficient_information_can_still_finish_loop(
) -> None:
    """信息不足可以正常结束循环，但不等于诊断内容完整。"""

    result = AgentRunResult(
        state="completed",
        termination_reason="planner_finished",
        termination_message=(
            "Agent规划正常结束"
        ),
        finish_reason=(
            "insufficient_information"
        ),
        final_message=(
            "现有证据不足，需要补充机器人遥测。"
        ),
    )

    assert result.state == "completed"
    assert result.finish_reason == (
        "insufficient_information"
    )


@pytest.mark.parametrize(
    "termination_reason",
    [
        "max_steps_reached",
        "duplicate_tool_call",
    ],
)
def test_completed_result_rejects_abort_reason(
    termination_reason: str,
) -> None:
    """completed不能与Python安全中止原因组合。"""

    with pytest.raises(
        ValidationError,
        match=(
            "completed结果必须由"
            "planner_finished结束"
        ),
    ):
        AgentRunResult.model_validate({
            "state": "completed",
            "termination_reason": (
                termination_reason
            ),
            "termination_message": (
                "错误组合"
            ),
            "finish_reason": (
                "task_completed"
            ),
            "final_message": "草稿",
        })


def test_completed_result_requires_finish_reason(
) -> None:
    """正常完成必须说明Planner为什么停止。"""

    with pytest.raises(
        ValidationError,
        match="finish_reason",
    ):
        AgentRunResult(
            state="completed",
            termination_reason=(
                "planner_finished"
            ),
            termination_message=(
                "Agent规划正常结束"
            ),
            final_message="草稿",
        )


def test_completed_result_requires_final_message(
) -> None:
    """正常完成必须携带Planner生成的非空草稿。"""

    with pytest.raises(
        ValidationError,
        match="final_message",
    ):
        AgentRunResult(
            state="completed",
            termination_reason=(
                "planner_finished"
            ),
            termination_message=(
                "Agent规划正常结束"
            ),
            finish_reason="task_completed",
        )


@pytest.mark.parametrize(
    "termination_reason",
    [
        "max_steps_reached",
        "duplicate_tool_call",
        "planner_timeout",
        "planner_error",
        "invalid_planner_response",
        "tool_failure_limit_reached",
        "tool_policy_violation",
    ],
)
def test_aborted_result_accepts_supported_reason(
    termination_reason: str,
) -> None:
    """两种Python保护条件都应产生稳定的aborted结果。"""

    result = AgentRunResult.model_validate({
        "state": "aborted",
        "termination_reason": (
            termination_reason
        ),
        "termination_message": (
            "Agent已安全中止"
        ),
        "interactions": [
            make_interaction(),
        ],
        "missing_information": [
            "Agent尚未形成可信最终草稿",
        ],
    })

    assert result.state == "aborted"
    assert result.finish_reason is None
    assert result.final_message is None
    assert len(result.interactions) == 1
    assert result.missing_information == (
        "Agent尚未形成可信最终草稿",
    )


def test_aborted_result_rejects_planner_finished_reason(
) -> None:
    """aborted不能声称循环由Planner正常结束。"""

    with pytest.raises(
        ValidationError,
        match=(
            "aborted结果不能使用"
            "planner_finished"
        ),
    ):
        AgentRunResult(
            state="aborted",
            termination_reason=(
                "planner_finished"
            ),
            termination_message=(
                "错误组合"
            ),
            missing_information=(
                "当前任务尚未完成",
            ),
        )


def test_aborted_result_rejects_planner_final_fields(
) -> None:
    """中止时不能把尚未可信完成的Planner草稿包装成最终结果。"""

    with pytest.raises(
        ValidationError,
        match=(
            "aborted结果不能包含"
            "Planner最终草稿"
        ),
    ):
        AgentRunResult(
            state="aborted",
            termination_reason=(
                "max_steps_reached"
            ),
            termination_message=(
                "达到最大工具步骤限制"
            ),
            missing_information=(
                "还需要更多工具步骤",
            ),
            finish_reason="task_completed",
            final_message="不应被接受的草稿",
        )


def test_aborted_result_requires_missing_information(
) -> None:
    """中止结果必须说明仍未解决的内容。"""

    with pytest.raises(
        ValidationError,
        match=(
            "aborted结果必须包含"
            "missing_information"
        ),
    ):
        AgentRunResult(
            state="aborted",
            termination_reason=(
                "planner_timeout"
            ),
            termination_message=(
                "Agent规划服务响应超时"
            ),
        )


def test_missing_information_must_be_unique(
) -> None:
    """同一条缺失信息不能在结构化结果中重复出现。"""

    with pytest.raises(
        ValidationError,
        match=(
            "missing_information"
            "不能包含重复项"
        ),
    ):
        AgentRunResult(
            state="aborted",
            termination_reason=(
                "planner_error"
            ),
            termination_message=(
                "Agent规划服务暂时不可用"
            ),
            missing_information=(
                "规划服务未返回下一步决定",
                "规划服务未返回下一步决定",
            ),
        )


def test_task_completed_rejects_missing_information(
) -> None:
    """声称任务完成时不能同时声明仍有缺失信息。"""

    with pytest.raises(
        ValidationError,
        match=(
            "task_completed结果不能包含"
            "missing_information"
        ),
    ):
        AgentRunResult(
            state="completed",
            termination_reason=(
                "planner_finished"
            ),
            termination_message=(
                "Agent规划正常结束"
            ),
            finish_reason="task_completed",
            final_message="任务已经完成",
            missing_information=(
                "仍然缺少遥测",
            ),
        )


def test_insufficient_information_can_preserve_missing_items(
) -> None:
    """信息不足结束时可以结构化保留尚缺少的内容。"""

    result = AgentRunResult(
        state="completed",
        termination_reason=(
            "planner_finished"
        ),
        termination_message=(
            "Agent规划正常结束"
        ),
        finish_reason=(
            "insufficient_information"
        ),
        final_message=(
            "需要补充机器人遥测。"
        ),
        missing_information=(
            "缺少robot-001当前遥测",
        ),
    )

    assert result.missing_information == (
        "缺少robot-001当前遥测",
    )


def test_result_rejects_undeclared_field(
) -> None:
    """结果契约不能接收未声明的私有推理或任意附加字段。"""

    with pytest.raises(
        ValidationError,
        match="chain_of_thought",
    ):
        AgentRunResult.model_validate({
            "state": "aborted",
            "termination_reason": (
                "max_steps_reached"
            ),
            "termination_message": (
                "达到最大工具步骤限制"
            ),
            "missing_information": [
                "当前任务尚未完成",
            ],
            "chain_of_thought": (
                "不允许进入公开结果"
            ),
        })


def test_result_rejects_more_than_twenty_interactions(
) -> None:
    """运行结果不能绕过Agent最多20步的全局结构上限。"""

    # 每个内层AgentToolInteraction都必须先是合法对象，
    # 才能精确验证外层AgentRunResult的集合长度限制。
    #
    # 因此这里重复使用合法step_number=1，
    # 而不是创建本身就非法的step_number=21。
    interactions = tuple(
        make_interaction(
            step_number=1,
            call_id=f"call_{index:03d}",
        )
        for index in range(1, 22)
    )

    with pytest.raises(
        ValidationError,
        match="interactions",
    ):
        AgentRunResult(
            state="aborted",
            termination_reason=(
                "max_steps_reached"
            ),
            termination_message=(
                "达到最大工具步骤限制"
            ),
            interactions=interactions,
            missing_information=(
                "当前任务尚未完成",
            ),
        )
