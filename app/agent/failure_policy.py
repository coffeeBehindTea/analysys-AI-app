"""Agent工具执行结果的失败恢复分类策略。

本模块负责：

1. 识别成功工具结果；
2. 识别允许Planner调整方案的可恢复失败；
3. 识别必须立即中止的工具权限失败；
4. 为AgentRunner提供稳定的分类结果。

本模块不执行工具、不调用Planner、不维护失败次数，
也不直接创建AgentRunResult。
"""

from typing import (
    Literal,
)

from app.schemas.agent import (
    ToolExecutionResult,
)


# AgentRunner处理一项工具结果时使用的分类。
ToolResultDisposition = Literal[
    # 工具成功执行并返回经过校验的数据。
    "success",

    # 当前工具没有产生成功结果，
    # 但Planner可以在受限次数内调整参数、
    # 改写查询或选择其他已注册工具。
    "recoverable_failure",

    # 工具请求触碰了明确权限边界，
    # 当前Agent运行不能继续尝试。
    "fatal_policy_violation",
]


def classify_tool_execution_result(
    result: ToolExecutionResult,
    /,
) -> ToolResultDisposition:
    """将一项工具执行结果分类给AgentRunner。

    分类规则：

    1. status=success：
       返回success。

    2. status=rejected且error_code=tool_blocked：
       返回fatal_policy_violation。

    3. 其他所有非成功状态：
       返回recoverable_failure。

    该函数不解析public_message中的自然语言，
    只使用结构化status和error_code。
    """

    if not isinstance(
        result,
        ToolExecutionResult,
    ):
        # Python类型注解不会执行运行时检查。
        #
        # 普通dict或SDK响应对象不能绕过
        # ToolExecutionResult数据契约进入失败策略。
        raise TypeError(
            "result必须是ToolExecutionResult"
        )

    if result.status == "success":
        return "success"

    if (
        result.status == "rejected"
        and result.error_code
        == "tool_blocked"
    ):
        # tool_blocked表示工具违反只读或风险策略。
        #
        # 这种结果不是普通参数错误，
        # 也不允许Planner通过换一种参数再次尝试。
        return "fatal_policy_violation"

    # 其余非成功结果暂时都允许Planner在
    # 连续失败预算内调整一次方案。
    return "recoverable_failure"