"""Agent规划模型适配层的最小异步协议。

AgentRunner只依赖本协议，
不直接依赖AsyncOpenAI、具体模型或测试Fake。

正式运行时可以由OpenAI兼容Provider满足；
自动化测试时可以由固定轨迹Fake满足。
"""

from typing import (
    Any,
    Protocol,
)

from app.schemas.agent_planning import (
    AgentPlannerDecision,
    AgentPlanningContext,
)


# ToolRegistry生成的OpenAI兼容Tool Schema。
#
# 这些字典由应用代码根据可信ToolDefinition构造，
# 不是由用户或LLM提供。
OpenAIToolSchema = dict[
    str,
    Any,
]


class AgentPlanner(Protocol):
    """AgentRunner所依赖的最小规划能力。"""

    async def plan(
        self,
        *,
        context: AgentPlanningContext,
        tool_schemas: tuple[
            OpenAIToolSchema,
            ...,
        ],
    ) -> AgentPlannerDecision:
        """根据任务、历史观察和允许工具返回下一轮决定。

        Provider只能返回结构化决定：

        1. 请求调用一个工具；
        2. 请求结束并给出模型草稿。

        Provider本身不执行工具，也不改变Agent状态。
        """

        ...