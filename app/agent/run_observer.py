"""AgentRunner实时执行观察者协议。

本模块定义AgentRunner能够向外部公开的内部执行节点。

本模块负责：

1. 规定开始规划时需要通知的数据；
2. 规定工具开始时需要通知的数据；
3. 规定工具完成时需要通知的数据；
4. 规定确定性进度更新时需要通知的数据；
5. 让Runner不直接依赖SSE、FastAPI或具体Publisher。

本模块不负责：

1. 执行Planner；
2. 执行Agent工具；
3. 生成脱敏公开摘要；
4. 发布SSE事件；
5. 编码text/event-stream；
6. 保存诊断会话。
"""

from typing import (
    Protocol,
)

from app.schemas.agent import (
    ToolCall,
    ToolExecutionResult,
)
from app.schemas.agent_progress import (
    AgentProgress,
)


class AgentRunObserver(
    Protocol
):
    """AgentRunner执行节点的异步观察者契约。

    AgentRunner会按照实际执行顺序等待这些方法完成。

    具体实现必须快速完成通知，不应在方法中执行
    LLM调用、真实工具调用或其他耗时业务操作。

    SSE实现只需把事件放入AgentEventPublisher的
    asyncio.Queue，不需要等待浏览器实际读取事件。
    """

    async def on_planning_started(
        self,
        *,
        step_number: int,
        allowed_tool_names: tuple[
            str,
            ...,
        ],
    ) -> None:
        """通知观察者：Runner即将调用Planner。

        step_number：
            当前即将规划的步骤编号，从1开始。

        allowed_tool_names：
            本轮Planner实际能够看到和选择的工具名称。
            如果为空，表示Planner只能形成最终结果，
            不能继续请求工具。
        """

        ...

    async def on_tool_started(
        self,
        *,
        step_id: int,
        tool_call: ToolCall,
    ) -> None:
        """通知观察者：一个工具已经通过检查并准备执行。

        step_id：
            当前工具步骤编号。

        tool_call：
            已经通过基本结构校验的内部工具调用。

        tool_call可能包含内部参数，因此具体观察者不能
        直接记录或公开arguments，必须先转换成脱敏摘要。
        """

        ...

    async def on_tool_finished(
        self,
        *,
        step_id: int,
        tool_call: ToolCall,
        result: ToolExecutionResult,
    ) -> None:
        """通知观察者：工具执行已经得到确定结果。

        success、empty、rejected、timeout和error
        都必须产生该通知。

        具体观察者只能公开经过脱敏的结果摘要和稳定错误码，
        不能直接公开result.output中的完整内部内容。
        """

        ...

    async def on_progress_updated(
        self,
        *,
        progress: AgentProgress,
    ) -> None:
        """通知观察者：确定性Progress Reducer已生成新快照。

        该进度不是Planner自行声称的进度，而是Python根据
        工具实际结果、所需能力和安全规则计算的权威状态。
        """

        ...