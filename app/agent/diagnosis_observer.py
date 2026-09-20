"""Agent诊断服务完整生命周期的异步观察者协议。

本协议在AgentRunObserver的Runner内部通知基础上，
增加AgentDiagnosisService才能确定的请求级和终端通知。

本模块负责规定通知方法的数据契约，不负责：

1. 执行诊断；
2. 调用Planner或工具；
3. 发布SSE事件；
4. 编码text/event-stream；
5. 保存诊断会话；
6. 捕获或转换运行异常。
"""

from typing import (
    Protocol,
)

from app.agent.run_observer import (
    AgentRunObserver,
)
from app.schemas.agent_api import (
    AgentDiagnosisResponse,
)
from app.schemas.agent_tool_policy import (
    AgentReadOnlyToolName,
    AgentRequiredCapability,
    AgentToolPolicyDisposition,
    AgentToolPolicyReasonCode,
)


class AgentDiagnosisObserver(
    AgentRunObserver,
    Protocol,
):
    """观察一次Agent诊断的完整公开生命周期。

    本协议继承AgentRunObserver，因此实现者除了请求级方法，
    还必须实现规划、工具开始、工具结束和进度更新通知。

    AgentDiagnosisService负责调用本协议；
    AgentRunner只会看到继承自AgentRunObserver的四个方法。
    """

    async def on_request_received(
        self,
        *,
        image_count: int,
        has_task_goal: bool,
    ) -> None:
        """通知观察者：诊断请求已经通过API Schema校验。

        image_count：
            当前请求携带的图片数量。

        has_task_goal：
            用户是否显式提供了task_goal。

        不传递完整request，是为了避免Observer意外公开
        原始日志、图片Base64或其他用户输入。
        """

        ...

    async def on_safety_classified(
        self,
        *,
        disposition: (
            AgentToolPolicyDisposition
        ),
        reason_codes: tuple[
            AgentToolPolicyReasonCode,
            ...,
        ],
    ) -> None:
        """通知观察者：确定性安全和工具策略已经形成处置。

        disposition可能是：

        1. continue_to_planner；
        2. abstained；
        3. human_review_required。

        本通知不包含Planner判断，处置结果来自Python安全规则。
        """

        ...

    async def on_tool_scope_decided(
        self,
        *,
        required_capabilities: tuple[
            AgentRequiredCapability,
            ...,
        ],
        allowed_tool_names: tuple[
            AgentReadOnlyToolName,
            ...,
        ],
    ) -> None:
        """通知观察者：请求级最小工具范围已经确定。

        required_capabilities描述完成任务需要哪些能力。

        allowed_tool_names是策略允许工具与Runner真实注册工具
        求交集后的最终集合，不是用户或Planner自行指定的集合。
        """

        ...

    async def on_diagnosis_finished(
        self,
        *,
        response: AgentDiagnosisResponse,
    ) -> None:
        """通知观察者：最终响应已经校验并完成会话保存。

        response必须是普通JSON接口使用的同一个
        AgentDiagnosisResponse。

        这样SSE最终事件、普通响应和持久化会话不会分别
        构造三份可能不一致的诊断结果。
        """

        ...