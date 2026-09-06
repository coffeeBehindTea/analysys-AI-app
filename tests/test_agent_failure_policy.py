"""Agent工具结果失败分类策略的纯函数单元测试。

本文件验证：

1. success结果分类为成功；
2. empty、普通rejected、timeout和error可受限恢复；
3. tool_blocked属于致命权限失败；
4. 普通dict不能绕过ToolExecutionResult契约。

测试不运行工具、不调用Planner，也不访问外部服务。
"""

import pytest

from app.agent.failure_policy import (
    classify_tool_execution_result,
)
from app.schemas.agent import (
    ToolErrorCode,
    ToolExecutionResult,
    ToolExecutionStatus,
)


def make_result(
    *,
    status: ToolExecutionStatus,
    error_code: ToolErrorCode | None = None,
) -> ToolExecutionResult:
    """创建与status、error_code关系一致的工具结果。"""

    if status == "success":
        return ToolExecutionResult(
            call_id="call_001",
            tool_name="search_knowledge",
            status="success",
            output={
                "evidence_ids": [
                    "document-id:000001",
                ],
            },
            duration_ms=1.0,
        )

    if error_code is None:
        raise ValueError(
            "非success测试结果必须提供error_code"
        )

    return ToolExecutionResult(
        call_id="call_001",
        tool_name="search_knowledge",
        status=status,
        error_code=error_code,
        public_message=(
            "测试使用的稳定公开错误说明"
        ),
        duration_ms=1.0,
    )


def test_success_result_is_classified_as_success(
) -> None:
    """经过输入、执行和输出校验的结果应清零失败计数。"""

    result = make_result(
        status="success"
    )

    disposition = (
        classify_tool_execution_result(
            result
        )
    )

    assert disposition == "success"


@pytest.mark.parametrize(
    (
        "status",
        "error_code",
    ),
    [
        (
            "empty",
            "empty_result",
        ),
        (
            "rejected",
            "unknown_tool",
        ),
        (
            "rejected",
            "invalid_tool_arguments",
        ),
        (
            "rejected",
            "duplicate_tool_call",
        ),
        (
            "timeout",
            "tool_timeout",
        ),
        (
            "error",
            "tool_execution_error",
        ),
        (
            "error",
            "invalid_tool_output",
        ),
        (
            "timeout",
            "vision_timeout",
        ),
        (
            "error",
            "vision_upstream_error",
        ),
        (
            "error",
            "invalid_vision_response",
        ),
    ],
)
def test_non_policy_failures_are_recoverable(
    status: ToolExecutionStatus,
    error_code: ToolErrorCode,
) -> None:
    """普通工具失败应允许Planner在连续失败预算内调整。"""

    result = make_result(
        status=status,
        error_code=error_code,
    )

    disposition = (
        classify_tool_execution_result(
            result
        )
    )

    assert disposition == (
        "recoverable_failure"
    )


def test_tool_blocked_is_fatal_policy_violation(
) -> None:
    """触碰只读或风险边界后不能允许Planner继续试探。"""

    result = make_result(
        status="rejected",
        error_code="tool_blocked",
    )

    disposition = (
        classify_tool_execution_result(
            result
        )
    )

    assert disposition == (
        "fatal_policy_violation"
    )


def test_classification_requires_execution_result(
) -> None:
    """普通字典不能绕过ToolExecutionResult字段关系校验。"""

    with pytest.raises(
        TypeError,
        match=(
            "result必须是"
            "ToolExecutionResult"
        ),
    ):
        classify_tool_execution_result({})
