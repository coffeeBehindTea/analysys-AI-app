"""Week 6 多模态 Agent 离线回归的 Fake 执行适配器。

本模块把声明式离线夹具转换成满足现有生产接口的对象：

1. ScriptedOfflinePlanner 满足 AgentPlanner Protocol；
2. ScriptedOfflineToolHandler 满足工具 Handler 的异步调用形式；
3. build_offline_fake_vision_provider 复用现有 FakeVisionProvider。

本模块不运行完整场景、不计算评测指标，也不读取 Gold 文件。
后续离线回归执行器负责把这些 Fake 注入真实 Agent 主链路。
"""

from copy import (
    deepcopy,
)
from typing import (
    Any,
)

from pydantic import (
    BaseModel,
    TypeAdapter,
)

from app.agent.planner import (
    OpenAIToolSchema,
)
from app.errors import (
    InvalidLLMResponseError,
    InvalidVisionResponseError,
    LLMUpstreamError,
    VisionTimeoutError,
    VisionUpstreamError,
)
from app.schemas.agent import (
    ToolName,
)
from app.schemas.agent_planning import (
    AgentPlannerDecision,
    AgentPlanningContext,
)
from app.schemas.multimodal_offline_regression import (
    MultimodalOfflineScenarioFixture,
    OfflinePlannerTurn,
    OfflineToolOutcome,
)
from app.services.fake_vision_provider import (
    FakeVisionOutcome,
    FakeVisionProvider,
)


# ToolName是Annotated类型，不是可以直接实例化的类。
#
# TypeAdapter允许普通代码复用ToolName中的Pydantic约束，
# 例如名称长度、首字符和允许字符。
_TOOL_NAME_ADAPTER = TypeAdapter(
    ToolName
)


def _extract_tool_names(
    tool_schemas: tuple[
        OpenAIToolSchema,
        ...,
    ],
    /,
) -> tuple[str, ...]:
    """从OpenAI兼容工具Schema提取工具名称。

    这里只记录工具名，不保存完整Schema，
    避免离线轨迹积累不必要的Prompt和参数描述。
    """

    tool_names: list[str] = []

    for schema in tool_schemas:
        if not isinstance(schema, dict):
            raise TypeError(
                "tool_schemas只能包含dict"
            )

        function_schema = schema.get(
            "function"
        )

        if not isinstance(
            function_schema,
            dict,
        ):
            raise ValueError(
                "工具Schema缺少function对象"
            )

        tool_name = function_schema.get(
            "name"
        )

        if not isinstance(tool_name, str):
            raise ValueError(
                "工具Schema缺少function.name"
            )

        # 再次使用ToolName规则检查名称，
        # 防止测试Fake接受生产Planner不会收到的非法Schema。
        validated_tool_name = (
            _TOOL_NAME_ADAPTER
            .validate_python(tool_name)
        )

        tool_names.append(
            validated_tool_name
        )

    return tuple(tool_names)


class ScriptedOfflinePlanner:
    """按顺序消费OfflinePlannerTurn的Fake Planner。

    本类满足AgentPlanner Protocol要求的异步plan()方法。
    AgentRunner不需要知道当前使用的是真实模型还是Fake。
    """

    def __init__(
        self,
        *,
        turns: tuple[
            OfflinePlannerTurn,
            ...,
        ],
    ) -> None:
        """复制并保存经过Schema校验的Planner轮次。"""

        if not isinstance(turns, tuple):
            raise TypeError(
                "turns必须是tuple"
            )

        if not all(
            isinstance(
                turn,
                OfflinePlannerTurn,
            )
            for turn in turns
        ):
            raise TypeError(
                "turns只能包含OfflinePlannerTurn"
            )

        self._turns = tuple(turns)
        self._next_turn_index = 0

        # 只保存经过结构化校验的Planner上下文。
        self._contexts: list[
            AgentPlanningContext
        ] = []

        # 每一轮只记录暴露的工具名称，
        # 不记录完整工具Schema。
        self._exposed_tool_names: list[
            tuple[str, ...]
        ] = []

    @property
    def call_count(self) -> int:
        """返回Fake Planner已经被调用的次数。"""

        return len(self._contexts)

    @property
    def contexts(
        self,
    ) -> tuple[
        AgentPlanningContext,
        ...,
    ]:
        """返回不可变的Planner上下文快照。"""

        return tuple(self._contexts)

    @property
    def exposed_tool_names(
        self,
    ) -> tuple[
        tuple[str, ...],
        ...,
    ]:
        """返回每轮Planner实际看到的工具名称。"""

        return tuple(
            self._exposed_tool_names
        )

    async def plan(
        self,
        *,
        context: AgentPlanningContext,
        tool_schemas: tuple[
            OpenAIToolSchema,
            ...,
        ],
    ) -> AgentPlannerDecision:
        """返回下一轮决定或抛出夹具指定的稳定异常。

        AgentRunner会像处理真实Planner一样处理这些结果：

        - TimeoutError转换成planner_timeout；
        - LLMUpstreamError转换成planner_error；
        - InvalidLLMResponseError转换成
          invalid_planner_response；
        - 正常决定进入工具调用或结束分支。
        """

        if not isinstance(
            context,
            AgentPlanningContext,
        ):
            raise TypeError(
                "context必须是AgentPlanningContext"
            )

        if not isinstance(
            tool_schemas,
            tuple,
        ):
            raise TypeError(
                "tool_schemas必须是tuple"
            )

        exposed_names = _extract_tool_names(
            tool_schemas
        )

        # 即使后续发现脚本耗尽，
        # 这次Planner调用也是真实发生的，
        # 因此先保存脱敏调用记录。
        self._contexts.append(context)
        self._exposed_tool_names.append(
            exposed_names
        )

        if (
            self._next_turn_index
            >= len(self._turns)
        ):
            raise InvalidLLMResponseError(
                "Fake Planner脚本已经耗尽"
            )

        turn = self._turns[
            self._next_turn_index
        ]
        self._next_turn_index += 1

        if turn.failure == "timeout":
            # Python 3.11中asyncio.TimeoutError与
            # 内置TimeoutError使用统一异常语义。
            raise TimeoutError(
                "Fake Planner模拟超时"
            )

        if turn.failure == "error":
            raise LLMUpstreamError(
                "Fake Planner模拟上游错误"
            )

        if turn.failure == "invalid_response":
            raise InvalidLLMResponseError(
                "Fake Planner模拟无效响应"
            )

        # OfflinePlannerTurn契约已经保证：
        # 没有failure时一定存在decision。
        if turn.decision is None:
            raise RuntimeError(
                "OfflinePlannerTurn内部状态无效"
            )

        return turn.decision


class ScriptedOfflineToolHandler:
    """按调用顺序返回预设结果的异步Fake工具处理器。

    ToolExecutor调用Handler时只传入已经校验的输入模型，
    不会把ToolCall或call_id传给Handler。

    因此：

    1. Schema层负责校验call_id与Planner脚本对应；
    2. 离线执行器按工具名和调用顺序建立结果队列；
    3. 本Handler按队列顺序消费结果。
    """

    def __init__(
        self,
        *,
        tool_name: str,
        outcomes: tuple[
            OfflineToolOutcome,
            ...,
        ],
    ) -> None:
        """保存一个工具对应的结果队列。"""

        validated_tool_name = (
            _TOOL_NAME_ADAPTER
            .validate_python(tool_name)
        )

        if not isinstance(outcomes, tuple):
            raise TypeError(
                "outcomes必须是tuple"
            )

        if not all(
            isinstance(
                outcome,
                OfflineToolOutcome,
            )
            for outcome in outcomes
        ):
            raise TypeError(
                "outcomes只能包含OfflineToolOutcome"
            )

        if any(
            outcome.tool_name
            != validated_tool_name
            for outcome in outcomes
        ):
            raise ValueError(
                "outcomes中的tool_name必须全部匹配"
            )

        self._tool_name = (
            validated_tool_name
        )
        self._outcomes = tuple(outcomes)
        self._next_outcome_index = 0

        # 这里只记录验证后的结构化输入，
        # 不记录未经校验的Planner参数。
        self._calls: list[
            dict[str, Any]
        ] = []

    @property
    def tool_name(self) -> str:
        """返回当前Handler负责的工具名称。"""

        return self._tool_name

    @property
    def call_count(self) -> int:
        """返回Handler已经执行的次数。"""

        return len(self._calls)

    @property
    def calls(
        self,
    ) -> tuple[
        dict[str, Any],
        ...,
    ]:
        """返回工具输入的独立快照。

        deepcopy避免调用方修改嵌套字典或列表，
        从而改变Fake内部保存的审计记录。
        """

        return tuple(
            deepcopy(call)
            for call in self._calls
        )

    async def __call__(
        self,
        validated_input: BaseModel,
    ) -> dict[str, Any] | None:
        """返回下一项结果，或抛出预设失败。

        该方法使用__call__，因此对象可以像异步函数一样调用：

        result = await handler(validated_input)
        """

        if not isinstance(
            validated_input,
            BaseModel,
        ):
            raise TypeError(
                "validated_input必须是BaseModel"
            )

        self._calls.append(
            validated_input.model_dump(
                mode="python"
            )
        )

        if (
            self._next_outcome_index
            >= len(self._outcomes)
        ):
            raise RuntimeError(
                "Fake工具结果队列已经耗尽"
            )

        outcome = self._outcomes[
            self._next_outcome_index
        ]
        self._next_outcome_index += 1

        if outcome.status == "success":
            # Schema已经保证success一定包含output。
            if outcome.output is None:
                raise RuntimeError(
                    "OfflineToolOutcome内部状态无效"
                )

            # 返回副本，防止ToolExecutor或输出模型
            # 修改夹具中保存的原始字典。
            return deepcopy(
                outcome.output
            )

        if outcome.status == "empty":
            # ToolExecutor把None转换成：
            # status=empty、error_code=empty_result。
            return None

        if outcome.status == "timeout":
            # ToolExecutor捕获TimeoutError并生成
            # 稳定的tool_timeout结果。
            raise TimeoutError(
                "Fake工具模拟超时"
            )

        # OfflineToolOutcome只剩error。
        #
        # 异常正文不会进入公开API；
        # ToolExecutor只输出稳定的tool_execution_error。
        raise RuntimeError(
            "Fake工具模拟执行错误"
        )


def build_offline_fake_vision_provider(
    fixture: MultimodalOfflineScenarioFixture,
    /,
) -> FakeVisionProvider:
    """把声明式Vision结果转换成现有FakeVisionProvider。

    复用现有Provider可以继续验证：

    1. VisionInput必须经过输入适配器；
    2. 使用真实图片SHA-256匹配结果；
    3. 自动注入source、模型名和Prompt版本；
    4. 只保存脱敏调用摘要；
    5. Vision异常由ToolExecutor转换成稳定状态。
    """

    if not isinstance(
        fixture,
        MultimodalOfflineScenarioFixture,
    ):
        raise TypeError(
            "fixture必须是"
            "MultimodalOfflineScenarioFixture"
        )

    scripted_results: dict[
        str,
        FakeVisionOutcome,
    ] = {}

    for outcome in fixture.vision_outcomes:
        if outcome.draft is not None:
            scripted_result: FakeVisionOutcome = (
                outcome.draft
            )

        elif outcome.failure == "timeout":
            scripted_result = VisionTimeoutError(
                "Fake Vision模拟超时"
            )

        elif (
            outcome.failure
            == "upstream_error"
        ):
            scripted_result = VisionUpstreamError(
                "Fake Vision模拟上游错误"
            )

        elif (
            outcome.failure
            == "invalid_response"
        ):
            scripted_result = (
                InvalidVisionResponseError(
                    "Fake Vision模拟无效响应"
                )
            )

        else:
            # OfflineVisionOutcome自身已经执行过
            # 二选一校验；该分支只是纵深防御。
            raise RuntimeError(
                "OfflineVisionOutcome内部状态无效"
            )

        scripted_results[
            outcome.image_sha256
        ] = scripted_result

    return FakeVisionProvider(
        scripted_results=scripted_results
    )