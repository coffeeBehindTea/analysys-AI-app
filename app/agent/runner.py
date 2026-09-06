"""Robot Diagnostic Agent的规划与受控工具执行循环。

当前版本负责：

1. 创建每一轮AgentPlanningContext；
2. 调用AgentPlanner；
3. 为单轮Planner调用设置超时；
4. 将Planner异常转换成稳定中止结果；
5. 将结构化ToolCall交给ToolExecutor；
6. 保存AgentToolInteraction；
7. 处理Planner主动结束；
8. 限制最大工具调用步数；
9. 拒绝重复工具调用；
10. 根据已完成进度收窄后续可见工具；
11. 拒绝不符合契约的Planner响应。

工具失败会经过独立策略模块分类；
连续可恢复失败达到限制或出现权限失败时，
Runner会保留已完成步骤并安全中止。
"""

import asyncio
import json
import logging

from copy import (
    deepcopy,
)
from inspect import (
    iscoroutinefunction,
)

from app.agent.executor import (
    ToolExecutor,
)
from app.agent.failure_policy import (
    classify_tool_execution_result,
)
from app.agent.planner import (
    AgentPlanner,
    OpenAIToolSchema,
)
# Provider把SDK超时转换成LLMTimeoutError，
# 把无法解析的模型响应转换成InvalidLLMResponseError。
#
# Runner根据异常类型生成稳定的结构化终止原因，
# 不解析异常消息中的自然语言。
from app.errors import (
    InvalidLLMResponseError,
    LLMTimeoutError,
)
from app.schemas.agent import (
    ToolCall,
)
from app.schemas.agent_planning import (
    AgentPlannerDecision,
    AgentPlanningContext,
    AgentToolInteraction,
)
from app.schemas.agent_runtime import (
    AgentRunResult,
    AgentRunTerminationReason,
)


# 该logger只记录稳定事件、异常类型和已完成步数。
#
# 不记录：
#
# 1. 用户完整任务；
# 2. Planner原始响应；
# 3. 异常消息原文；
# 4. 工具输入和输出正文。
#
# 这些内容可能包含日志、文档正文或上游敏感信息。
logger = logging.getLogger(__name__)


def _canonical_tool_call_signature(
    tool_call: ToolCall,
    /,
) -> tuple[str, str]:
    """生成不包含call_id的稳定工具调用签名。

    两次调用即使call_id不同，
    只要工具名和JSON参数完全相同，
    就会得到相同签名。
    """

    if not isinstance(
        tool_call,
        ToolCall,
    ):
        raise TypeError(
            "tool_call必须是ToolCall"
        )

    try:
        # sort_keys=True消除字典键顺序差异。
        #
        # 例如：
        #
        # {"query": "x", "top_k": 3}
        # {"top_k": 3, "query": "x"}
        #
        # 会产生相同的规范字符串。
        canonical_arguments = json.dumps(
            tool_call.arguments,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    except TypeError as exc:
        # OpenAI Tool Calling参数应来自JSON对象。
        # 不能使用repr()掩盖非JSON参数。
        raise TypeError(
            "tool_call.arguments必须是"
            "JSON可序列化数据"
        ) from exc

    return (
        tool_call.tool_name,
        canonical_arguments,
    )


def _narrow_tool_schemas_after_progress(
    *,
    active_tool_schemas: tuple[
        OpenAIToolSchema,
        ...,
    ],
    interactions: tuple[
        AgentToolInteraction,
        ...,
    ],
) -> tuple[
    OpenAIToolSchema,
    ...,
]:
    """根据成功工具历史收窄下一轮Planner可见工具。

    请求级工具范围回答“本次任务最多允许哪些能力”；本函数
    回答“根据已经完成的步骤，下一轮还需要哪些能力”。它只会
    从active_tool_schemas中删除工具，绝不会增加新工具。

    当前进度策略只有一条：draft_test_case已经成功时，
    隐藏该工具以及知识检索工具，避免生成多份内容不同但
    目的相同的草案，或在草案完成后重新检索并改写草案。

    在草案成功之前不会因为成功检索次数达到固定数字而隐藏
    search_knowledge。不同诊断任务需要的证据数量并不相同，
    Runner不应使用统一次数替代Planner对证据充分性的判断。

    本函数不读取用户任务正文、场景编号、Gold答案、文件名、
    页码或Chunk正文，因此不会把评测案例规则写进生产流程。
    """

    active_tool_names = frozenset(
        schema["function"]["name"]
        for schema in active_tool_schemas
    )

    # 不需要草案的诊断任务保持原有多轮检索能力。
    # 这样不会误伤需要从多个子问题收集证据的普通诊断。
    if "draft_test_case" not in active_tool_names:
        return active_tool_schemas

    draft_already_completed = any(
        interaction.tool_call.tool_name
        == "draft_test_case"
        and interaction.result.status
        == "success"
        for interaction in interactions
    )

    hidden_tool_names: set[str] = set()

    if draft_already_completed:
        # 草案已经形成后，不再重新检索并改写草案。
        # Planner仍可使用尚未完成的遥测、视觉或时间工具，
        # 也可以直接返回finish。
        hidden_tool_names.update({
            "search_knowledge",
            "draft_test_case",
        })

    if not hidden_tool_names:
        return active_tool_schemas

    return tuple(
        schema
        for schema in active_tool_schemas
        if schema["function"]["name"]
        not in hidden_tool_names
    )


def _build_aborted_result(
    *,
    termination_reason: (
        AgentRunTerminationReason
    ),
    termination_message: str,
    missing_information: str,
    interactions: list[
        AgentToolInteraction
    ],
) -> AgentRunResult:
    """统一构造保留已完成步骤的安全中止结果。

    使用一个辅助函数集中生成aborted结果，
    可以避免不同失败分支遗漏：

    1. interactions；
    2. missing_information；
    3. state；
    4. 稳定终止原因。
    """

    return AgentRunResult(
        state="aborted",
        termination_reason=(
            termination_reason
        ),
        termination_message=(
            termination_message
        ),

        # 深复制已经完成的历史。
        #
        # AgentToolInteraction虽然是frozen模型，
        # 但内部arguments和output仍可能包含可变dict。
        interactions=tuple(
            interaction.model_copy(
                deep=True
            )
            for interaction
            in interactions
        ),

        # 当前每种Runner中止分支只产生一条
        # 面向上层Service的结构化缺失说明。
        missing_information=(
            missing_information,
        ),
    )


class AgentRunner:
    """控制Planner与ToolExecutor之间的循环。"""

    def __init__(
        self,
        *,
        planner: AgentPlanner,
        executor: ToolExecutor,
        max_steps: int = 5,
        planner_timeout_seconds: float = 30.0,
        max_consecutive_tool_failures: int = 2,
    ) -> None:
        """保存Planner、Executor和运行边界配置。"""

        plan_method = getattr(
            planner,
            "plan",
            None,
        )

        if not iscoroutinefunction(
            plan_method
        ):
            raise TypeError(
                "planner.plan必须是异步方法"
            )

        if not isinstance(
            executor,
            ToolExecutor,
        ):
            raise TypeError(
                "executor必须是ToolExecutor"
            )

        # bool是int的子类：
        #
        # isinstance(True, int) == True
        #
        # 但True不应该被解释为max_steps=1，
        # 所以必须单独拒绝bool。
        if (
            isinstance(max_steps, bool)
            or not isinstance(max_steps, int)
        ):
            raise TypeError(
                "max_steps必须是整数"
            )

        if not 1 <= max_steps <= 20:
            raise ValueError(
                "max_steps必须在1到20之间"
            )

        # 连续失败限制必须使用真正的整数。
        #
        # bool是int的子类，但True不能被解释为
        # 允许连续失败1次。
        if (
            isinstance(
                max_consecutive_tool_failures,
                bool,
            )
            or not isinstance(
                max_consecutive_tool_failures,
                int,
            )
        ):
            raise TypeError(
                "max_consecutive_tool_failures"
                "必须是整数"
            )

        # 最多允许配置20次连续失败，
        # 与Agent工具交互历史的绝对上限保持一致。
        if not (
            1
            <= max_consecutive_tool_failures
            <= 20
        ):
            raise ValueError(
                "max_consecutive_tool_failures"
                "必须在1到20之间"
            )

        # Planner超时允许使用int或float，
        # 例如30秒或0.1秒。
        #
        # bool同样不能被解释为1秒或0秒。
        if (
            isinstance(
                planner_timeout_seconds,
                bool,
            )
            or not isinstance(
                planner_timeout_seconds,
                (int, float),
            )
        ):
            raise TypeError(
                "planner_timeout_seconds"
                "必须是数字"
            )

        # 单轮Planner调用最多等待120秒。
        #
        # NaN与任何有序范围比较都不会得到True，
        # 因此也会被这个条件拒绝。
        if not (
            0
            < planner_timeout_seconds
            <= 120
        ):
            raise ValueError(
                "planner_timeout_seconds"
                "必须大于0且不超过120"
            )

        self._planner = planner
        self._executor = executor
        self._max_steps = max_steps

        # 保存连续工具失败上限。
        #
        # 它只统计ToolExecutionResult中的非success结果，
        # 不统计Planner调用次数。
        self._max_consecutive_tool_failures = (
            max_consecutive_tool_failures
        )

        # 统一保存成float，
        # 例如输入30后内部保存为30.0。
        self._planner_timeout_seconds = float(
            planner_timeout_seconds
        )

        # 从Executor取得它实际执行白名单的Schema。
        #
        # tuple固定外层顺序；
        # 每轮调用Planner时还会进行deepcopy，
        # 防止Provider修改内部字典。
        self._tool_schemas: tuple[
            OpenAIToolSchema,
            ...,
        ] = (
            executor
            .build_openai_tool_schemas()
        )

        # 注册表Schema是Executor真实执行白名单的快照。
        # 保存名称集合后，每个请求只能在这个全集内继续收窄，
        # 不能凭空扩展一个未注册工具。
        self._registered_tool_names = frozenset(
            schema["function"]["name"]
            for schema in self._tool_schemas
        )

    @property
    def registered_tool_names(
        self,
    ) -> frozenset[str]:
        """返回本Runner对应Executor的不可变工具名快照。

        AgentDiagnosisService用这个只读属性把请求的结构性工具
        范围与实际注册表求交集。这样测试或其他部署只注册部分
        工具时，Service不会把不存在的工具名交给Runner；同时
        返回frozenset可防止调用方修改Runner内部权限快照。
        """

        return self._registered_tool_names

    async def run(
        self,
        task: str,
        /,
        *,
        allowed_tool_names: (
            frozenset[str] | None
        ) = None,
    ) -> AgentRunResult:
        """在当前请求工具范围内运行规划和观察循环。"""

        if not isinstance(task, str):
            raise TypeError(
                "task必须是字符串"
            )

        # None表示使用Executor注册表中的全部只读工具，
        # 从而保持旧的内部调用语义。
        # API Service会明确传入frozenset以启用请求级最小权限。
        if allowed_tool_names is None:
            active_tool_names = (
                self._registered_tool_names
            )
        else:
            if not isinstance(
                allowed_tool_names,
                frozenset,
            ):
                raise TypeError(
                    "allowed_tool_names必须是"
                    "frozenset或None"
                )

            if not all(
                isinstance(name, str)
                and bool(name.strip())
                for name in allowed_tool_names
            ):
                raise ValueError(
                    "allowed_tool_names只能包含"
                    "非空工具名"
                )

            unknown_tool_names = (
                allowed_tool_names
                - self._registered_tool_names
            )

            if unknown_tool_names:
                raise ValueError(
                    "allowed_tool_names包含"
                    "未注册工具"
                )

            active_tool_names = (
                allowed_tool_names
            )

        # 保留注册表原有Schema顺序，只删除本次请求不需要的工具。
        # 这是请求开始时的最大允许范围；循环内还会根据已完成
        # 的工具进度继续收窄，但不会重新扩大。
        #
        # 空tuple是合法值：Planner仍可直接finish或安全拒答，
        # 但不能请求任何工具。
        active_tool_schemas = tuple(
            schema
            for schema in self._tool_schemas
            if schema["function"]["name"]
            in active_tool_names
        )

        # 第一次构造Context同时完成：
        #
        # 1. 首尾空白清理；
        # 2. 空文本检查；
        # 3. 最大长度检查。
        initial_context = (
            AgentPlanningContext(
                task=task,
                next_step=1,
            )
        )
        normalized_task = (
            initial_context.task
        )

        interactions: list[
            AgentToolInteraction
        ] = []

        # call_id重复表示模型重复使用调用标识。
        seen_call_ids: set[str] = set()

        # 调用签名重复表示模型使用新call_id，
        # 但再次请求相同工具和相同参数。
        seen_call_signatures: set[
            tuple[str, str]
        ] = set()

        # 只统计连续出现的非success工具结果。
        #
        # 任意一次success都会把它重新设为0。
        consecutive_tool_failures = 0

        while True:
            # 请求级范围是静态权限上限；round_tool_schemas是
            # 本轮基于成功历史得到的最小剩余能力集合。
            #
            # 例如测试草案已经成功生成后，本轮不再暴露
            # search_knowledge和draft_test_case，避免重新检索后
            # 生成另一份相互冲突的草案。单纯的检索次数不会触发
            # 这个收窄过程。
            round_tool_schemas = (
                _narrow_tool_schemas_after_progress(
                    active_tool_schemas=(
                        active_tool_schemas
                    ),
                    interactions=tuple(
                        interactions
                    ),
                )
            )
            round_tool_names = frozenset(
                schema["function"]["name"]
                for schema in round_tool_schemas
            )

            # 保持旧的内部调用语义：调用方没有显式传入请求级
            # 范围，且本轮也没有发生动态收窄时，Executor继续
            # 接收None。这样注册表外名称仍被分类为unknown_tool，
            # Planner可以观察该可恢复失败后安全结束。
            #
            # 只要公开请求给出了范围，或本轮已经隐藏了完成的
            # 工具，就必须传入实际round_tool_names，确保执行期
            # 边界与Planner本轮看到的Schema完全一致。
            if (
                allowed_tool_names is None
                and round_tool_names
                == self._registered_tool_names
            ):
                executor_tool_names = None
            else:
                executor_tool_names = (
                    round_tool_names
                )

            # next_step根据已经实际完成的工具交互计算，
            # 不使用Planner返回的自然语言编号。
            context = AgentPlanningContext(
                task=normalized_task,
                next_step=(
                    len(interactions) + 1
                ),
                interactions=tuple(
                    interactions
                ),
            )

            # Context虽然是frozen模型，
            # 但内部arguments和output仍包含可变dict。
            #
            # deep=True确保Provider不能通过修改嵌套dict
            # 改写Runner内部已经保存的历史。
            planner_context = (
                context.model_copy(
                    deep=True
                )
            )

            try:
                # asyncio.timeout限制的是当前这一轮
                # Planner调用时间，不是整个Agent总时间。
                #
                # 每次进入下一轮规划都会重新建立超时边界。
                async with asyncio.timeout(
                    self._planner_timeout_seconds
                ):
                    decision = await (
                        self._planner.plan(
                            context=planner_context,

                            # 每轮提供独立深复制，
                            # 防止Provider修改下一轮的
                            # 工具描述。
                            tool_schemas=deepcopy(
                                round_tool_schemas
                            ),
                        )
                    )

            except asyncio.CancelledError:
                # 外层FastAPI请求、服务器关闭或上层Task
                # 主动取消时，不能伪装成Planner错误。
                #
                # 继续抛出取消异常，
                # 让真正的调用方完成取消传播。
                raise

            except (
                TimeoutError,
                LLMTimeoutError,
            ):
                # asyncio.timeout到期后，
                # 会在上下文管理器外转换成TimeoutError。
                logger.warning(
                    "agent_planner_timeout "
                    "completed_steps=%d",
                    len(interactions),
                )

                return _build_aborted_result(
                    termination_reason=(
                        "planner_timeout"
                    ),
                    termination_message=(
                        "Agent规划服务响应超时"
                    ),
                    missing_information=(
                        "规划服务未返回下一步决定，"
                        "当前任务尚未形成可信最终草稿"
                    ),
                    interactions=interactions,
                )

            except InvalidLLMResponseError:
                # Provider已经收到上游响应，
                # 但响应没有通过Tool Calling或结束JSON契约。
                #
                # 这里只记录稳定事件和已完成步骤，
                # 不记录原始模型响应或异常消息，
                # 避免日志泄露不可信文本。
                logger.warning(
                    "agent_planner_invalid_response "
                    "type=InvalidLLMResponseError "
                    "completed_steps=%d",
                    len(interactions),
                )

                return _build_aborted_result(
                    termination_reason=(
                        "invalid_planner_response"
                    ),
                    termination_message=(
                        "Agent规划服务返回了"
                        "无效结构"
                    ),
                    missing_information=(
                        "规划结果没有通过内部数据校验，"
                        "当前任务尚未形成可信最终草稿"
                    ),
                    interactions=interactions,
                )

            except Exception as exc:
                # Planner普通异常不能使Agent主链路
                # 暴露原始上游错误。
                #
                # 这里只记录异常类型，
                # 不记录str(exc)或repr(exc)。
                logger.warning(
                    "agent_planner_error "
                    "type=%s completed_steps=%d",
                    type(exc).__name__,
                    len(interactions),
                )

                return _build_aborted_result(
                    termination_reason=(
                        "planner_error"
                    ),
                    termination_message=(
                        "Agent规划服务暂时不可用"
                    ),
                    missing_information=(
                        "规划服务未返回下一步决定，"
                        "当前任务尚未形成可信最终草稿"
                    ),
                    interactions=interactions,
                )

            # Python不会根据Protocol返回注解
            # 自动验证实际返回对象。
            if not isinstance(
                decision,
                AgentPlannerDecision,
            ):
                logger.warning(
                    "agent_planner_invalid_response "
                    "type=%s completed_steps=%d",
                    type(decision).__name__,
                    len(interactions),
                )

                return _build_aborted_result(
                    termination_reason=(
                        "invalid_planner_response"
                    ),
                    termination_message=(
                        "Agent规划服务返回了"
                        "无效结构"
                    ),
                    missing_information=(
                        "规划结果没有通过内部数据校验，"
                        "当前任务尚未形成可信最终草稿"
                    ),
                    interactions=interactions,
                )

            if decision.decision == "finish":
                # AgentPlannerDecision已经保证：
                #
                # finish_reason和final_message不为空，
                # tool_call为空。
                return AgentRunResult(
                    state="completed",
                    termination_reason=(
                        "planner_finished"
                    ),
                    termination_message=(
                        "Agent规划正常结束"
                    ),

                    # 正常结束时同样复制历史，
                    # 避免结果对象与内部运行容器共享dict。
                    interactions=tuple(
                        interaction.model_copy(
                            deep=True
                        )
                        for interaction
                        in interactions
                    ),

                    # 当前PlannerDecision还没有单独的
                    # missing_information字段。
                    #
                    # insufficient_information的具体说明
                    # 暂时仍保存在final_message中。
                    missing_information=(),

                    finish_reason=(
                        decision.finish_reason
                    ),
                    final_message=(
                        decision.final_message
                    ),
                )

            # AgentPlannerDecision已经保证call_tool
            # 决定必须包含ToolCall。
            tool_call = decision.tool_call

            if tool_call is None:
                # 正常Pydantic模型无法进入该分支。
                #
                # 保留防御性检查，避免未来Provider
                # 或模型构造方式变化后抛出普通AttributeError。
                logger.warning(
                    "agent_planner_invalid_response "
                    "reason=missing_tool_call "
                    "completed_steps=%d",
                    len(interactions),
                )

                return _build_aborted_result(
                    termination_reason=(
                        "invalid_planner_response"
                    ),
                    termination_message=(
                        "Agent规划服务返回了"
                        "无效工具调用"
                    ),
                    missing_information=(
                        "规划结果缺少合法工具请求，"
                        "当前任务尚未形成可信最终草稿"
                    ),
                    interactions=interactions,
                )

            # 深复制模型及其arguments字典，
            # 不直接执行Provider持有的原始对象。
            executable_call = (
                tool_call.model_copy(
                    deep=True
                )
            )

            # max_steps表示最多实际执行多少次工具。
            #
            # 达到上限后仍允许Planner再进行一次规划：
            #
            # 如果返回finish，可以正常结束；
            # 如果继续请求工具，则在执行前中止。
            if (
                len(interactions)
                >= self._max_steps
            ):
                return _build_aborted_result(
                    termination_reason=(
                        "max_steps_reached"
                    ),
                    termination_message=(
                        "Agent达到最大工具步骤限制"
                    ),
                    missing_information=(
                        "当前任务仍需要额外工具步骤，"
                        "但已经达到安全执行上限"
                    ),
                    interactions=interactions,
                )

            try:
                signature = (
                    _canonical_tool_call_signature(
                        executable_call
                    )
                )

            except TypeError:
                # ToolCall只保证arguments是dict，
                # 其中的Any值仍可能不是合法JSON数据。
                #
                # 不把具体参数内容写入日志。
                logger.warning(
                    "agent_planner_invalid_response "
                    "reason=non_json_arguments "
                    "completed_steps=%d",
                    len(interactions),
                )

                return _build_aborted_result(
                    termination_reason=(
                        "invalid_planner_response"
                    ),
                    termination_message=(
                        "Agent规划服务返回了"
                        "无效工具参数"
                    ),
                    missing_information=(
                        "工具参数不是合法JSON数据，"
                        "当前任务尚未形成可信最终草稿"
                    ),
                    interactions=interactions,
                )

            if (
                executable_call.call_id
                in seen_call_ids
                or signature
                in seen_call_signatures
            ):
                # 重复请求不会交给Executor，
                # 因此不会产生新的Interaction。
                return _build_aborted_result(
                    termination_reason=(
                        "duplicate_tool_call"
                    ),
                    termination_message=(
                        "Agent请求了重复工具调用"
                    ),
                    missing_information=(
                        "Planner未能提出新的工具调用，"
                        "当前任务尚未形成可信最终草稿"
                    ),
                    interactions=interactions,
                )

            result = await (
                self._executor.execute(
                    executable_call,

                    # 与Planner看到的工具范围使用同一个
                    # 本轮不可变集合，形成真正的执行期边界。
                    # 即使Provider返回了本轮已经隐藏的工具名，
                    # Executor也不会执行它。
                    allowed_tool_names=(
                        executor_tool_names
                    ),
                )
            )

            # 无论Executor返回success、empty、
            # rejected、timeout还是error，
            # 都先记录真实结果供下一轮Planner观察。
            interaction = AgentToolInteraction(
                step_number=(
                    len(interactions) + 1
                ),

                # 深复制可以避免后续对象持有者
                # 修改Runner保存的历史。
                tool_call=(
                    executable_call.model_copy(
                        deep=True
                    )
                ),
                result=result.model_copy(
                    deep=True
                ),
            )

            interactions.append(interaction)
            seen_call_ids.add(
                executable_call.call_id
            )
            seen_call_signatures.add(
                signature
            )

            # 失败策略必须在Interaction保存之后执行。
            #
            # 这样即使本次结果导致Agent立即中止，
            # 审计历史中仍然保留导致中止的工具调用和结果。
            disposition = (
                classify_tool_execution_result(
                    result
                )
            )

            if disposition == "success":
                # 成功结果说明Agent已经找到一条有效路径，
                # 因此此前的连续失败链被打断。
                consecutive_tool_failures = 0

                # 进入下一轮Planner，
                # 让它观察本次成功结果并决定继续或结束。
                continue

            if (
                disposition
                == "fatal_policy_violation"
            ):
                # 工具名和error_code来自经过校验的结构字段，
                # 日志不记录arguments、output或用户任务正文。
                logger.warning(
                    "agent_tool_policy_violation "
                    "tool=%s error_code=%s "
                    "completed_steps=%d",
                    result.tool_name,
                    result.error_code,
                    len(interactions),
                )

                return _build_aborted_result(
                    termination_reason=(
                        "tool_policy_violation"
                    ),
                    termination_message=(
                        "Agent工具请求违反安全策略"
                    ),
                    missing_information=(
                        "当前工具请求不符合只读或"
                        "风险权限边界，任务不能自动继续"
                    ),
                    interactions=interactions,
                )

            # 除success和fatal_policy_violation外，
            # classify_tool_execution_result当前只可能返回
            # recoverable_failure。
            consecutive_tool_failures += 1

            if (
                consecutive_tool_failures
                >= self._max_consecutive_tool_failures
            ):
                logger.warning(
                    "agent_tool_failure_limit_reached "
                    "status=%s error_code=%s "
                    "consecutive_failures=%d "
                    "completed_steps=%d",
                    result.status,
                    result.error_code,
                    consecutive_tool_failures,
                    len(interactions),
                )

                return _build_aborted_result(
                    termination_reason=(
                        "tool_failure_limit_reached"
                    ),
                    termination_message=(
                        "Agent连续工具调用失败"
                    ),
                    missing_information=(
                        "连续工具调用未能产生可用结果，"
                        "需要补充输入或人工检查工具状态"
                    ),
                    interactions=interactions,
                )

            # 第一次可恢复失败尚未达到默认限制。
            #
            # 返回循环开头，让Planner观察失败状态、
            # error_code和public_message后调整方案。
            continue
