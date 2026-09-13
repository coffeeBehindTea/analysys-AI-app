"""Agent单次工具调用的受控执行器。

本模块负责：

1. 从ToolRegistry精确查找工具；
2. 使用input_model校验LLM提供的参数；
3. 再次检查只读权限和风险边界；
4. 应用单工具异步超时；
5. 使用output_model校验工具返回值；
6. 把所有结果转换成稳定的ToolExecutionResult。

本模块不选择工具、不管理Agent循环，
也不把底层异常内容直接返回给用户。
"""

import asyncio
import logging

from typing import (
    Any,
)

from time import (
    perf_counter,
)

from pydantic import (
    BaseModel,
    ValidationError,
)

from app.errors import (
    InvalidVisionResponseError,
    VisionTimeoutError,
    VisionUpstreamError,
)
from app.agent.registry import (
    ToolRegistry,
)
from app.schemas.agent import (
    ToolCall,
    ToolExecutionResult,
)


# logger记录服务端可审计事件。
#
# __name__会得到app.agent.executor，
# 日志配置可以据此单独调整该模块的级别。
logger = logging.getLogger(__name__)


def _elapsed_milliseconds(
    started_at: float,
) -> float:
    """根据单调高精度计时器计算非负毫秒耗时。"""

    elapsed = (
        perf_counter()
        - started_at
    ) * 1_000

    # 正常情况下perf_counter不会倒退。
    # max()作为防御性处理，确保数据契约永远收到非负数。
    return max(
        0.0,
        elapsed,
    )


class ToolExecutor:
    """执行ToolRegistry中的一项工具调用。

    ToolExecutor本身不保存Agent步骤历史，
    所以重复工具调用检测仍由后续AgentRunner负责。
    """

    def __init__(
        self,
        *,
        registry: ToolRegistry,
    ) -> None:
        """保存经过类型检查的工具注册表。"""

        # Python不会根据类型注解自动拒绝None或dict，
        # 所以在任何工具调用前进行运行时检查。
        if not isinstance(
            registry,
            ToolRegistry,
        ):
            raise TypeError(
                "registry必须是ToolRegistry"
            )

        self._registry = registry

    def build_openai_tool_schemas(
        self,
    ) -> tuple[
        dict[str, Any],
        ...,
    ]:
        """返回Executor实际使用白名单的Tool Schema快照。

        AgentRunner通过本方法取得规划模型可见的工具，
        确保模型看到的工具列表和Executor实际执行的
        ToolRegistry来自同一个对象。
        """

        return tuple(
            self._registry
            .build_openai_tool_schemas()
        )

    async def execute(
        self,
        tool_call: ToolCall,
        *,
        allowed_tool_names: (
            frozenset[str] | None
        ) = None,
    ) -> ToolExecutionResult:
        """在注册表和请求级边界内执行一次工具调用。"""

        # ToolCall必须已经完成通用结构校验。
        #
        # ToolExecutor不接受LLM SDK原始对象或普通dict，
        # 避免执行阶段绕过ToolCall数据契约。
        if not isinstance(
            tool_call,
            ToolCall,
        ):
            raise TypeError(
                "tool_call必须是ToolCall"
            )

        # None表示调用方没有进一步缩小注册表，
        # 用于兼容不需要请求级隔离的内部调用和既有测试。
        # 一旦提供范围，就只接受不可变frozenset，避免执行途中
        # 被其他代码原地add或remove而改变权限。
        if (
            allowed_tool_names is not None
            and not isinstance(
                allowed_tool_names,
                frozenset,
            )
        ):
            raise TypeError(
                "allowed_tool_names必须是"
                "frozenset或None"
            )

        if (
            allowed_tool_names is not None
            and not all(
                isinstance(name, str)
                and bool(name.strip())
                for name in allowed_tool_names
            )
        ):
            raise ValueError(
                "allowed_tool_names只能包含"
                "非空工具名"
            )

        started_at = perf_counter()

        # Planner通常只能看到过滤后的Schema，
        # 但不能把“模型看不到”当成真正的执行权限控制。
        # 如果模型、Provider或未来代码仍构造出范围外调用，
        # Executor会在注册表查找和Handler执行之前稳定拒绝。
        if (
            allowed_tool_names is not None
            and tool_call.tool_name
            not in allowed_tool_names
        ):
            return ToolExecutionResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                status="rejected",
                error_code="tool_blocked",
                public_message=(
                    "工具不在当前请求允许范围内"
                ),
                duration_ms=(
                    _elapsed_milliseconds(
                        started_at
                    )
                ),
            )

        registered_tool = (
            self._registry.get(
                tool_call.tool_name
            )
        )

        if registered_tool is None:
            # 未知工具不会执行，也不会尝试动态导入。
            return ToolExecutionResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                status="rejected",
                error_code="unknown_tool",
                public_message=(
                    "请求的工具未在允许列表中"
                ),
                duration_ms=(
                    _elapsed_milliseconds(
                        started_at
                    )
                ),
            )

        definition = (
            registered_tool.definition
        )

        # 注册表已经执行过相同检查。
        #
        # Executor再次检查属于纵深防御：
        # 即使未来注册表实现变化或内部数据被错误替换，
        # 执行阶段仍不会调用可写或高风险工具。
        if (
            not definition.read_only
            or definition.risk_level == "high"
        ):
            return ToolExecutionResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                status="rejected",
                error_code="tool_blocked",
                public_message=(
                    "工具权限不符合Agent安全策略"
                ),
                duration_ms=(
                    _elapsed_milliseconds(
                        started_at
                    )
                ),
            )

        try:
            # input_model是该工具自己的Pydantic模型类。
            #
            # 例如search_knowledge会在这里验证：
            # query、top_k和可选检索范围。
            validated_input = (
                definition.input_model
                .model_validate(
                    tool_call.arguments
                )
            )

        except ValidationError:
            # 不把完整参数或Pydantic错误详情放进公开消息，
            # 防止日志、用户数据或内部字段被意外泄露。
            return ToolExecutionResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                status="rejected",
                error_code=(
                    "invalid_tool_arguments"
                ),
                public_message=(
                    "工具调用参数未通过校验"
                ),
                duration_ms=(
                    _elapsed_milliseconds(
                        started_at
                    )
                ),
            )

        try:
            # asyncio.timeout()是Python 3.11提供的
            # 异步上下文管理器。
            #
            # 如果代码块没有在规定时间内完成，
            # 退出上下文时会抛出TimeoutError。
            async with asyncio.timeout(
                definition.timeout_seconds
            ):
                raw_output = await (
                    registered_tool.handler(
                        validated_input
                    )
                )

        except VisionTimeoutError:
            # Vision Provider已经把SDK超时转换成稳定的
            # 应用异常。这里保留“Vision请求超时”语义，
            # 避免它落入通用tool_execution_error后失去原因。
            logger.warning(
                "agent_vision_tool_timeout "
                "call_id=%s tool=%s",
                tool_call.call_id,
                tool_call.tool_name,
            )

            return ToolExecutionResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                status="timeout",
                error_code="vision_timeout",
                public_message=(
                    "Vision模型响应超时"
                ),
                duration_ms=(
                    _elapsed_milliseconds(
                        started_at
                    )
                ),
            )

        except VisionUpstreamError:
            # 上游错误可能来自连接失败、限流或5xx状态。
            # 不公开原始异常正文，只返回可审计的稳定错误码。
            logger.warning(
                "agent_vision_tool_upstream_error "
                "call_id=%s tool=%s",
                tool_call.call_id,
                tool_call.tool_name,
            )

            return ToolExecutionResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                status="error",
                error_code=(
                    "vision_upstream_error"
                ),
                public_message=(
                    "Vision上游服务暂时不可用"
                ),
                duration_ms=(
                    _elapsed_milliseconds(
                        started_at
                    )
                ),
            )

        except InvalidVisionResponseError:
            # Vision服务已经返回，但内容为空、JSON非法，
            # 或字段不满足VisionObservation内部契约。
            logger.warning(
                "agent_vision_tool_invalid_response "
                "call_id=%s tool=%s",
                tool_call.call_id,
                tool_call.tool_name,
            )

            return ToolExecutionResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                status="error",
                error_code=(
                    "invalid_vision_response"
                ),
                public_message=(
                    "Vision模型返回无效结果"
                ),
                duration_ms=(
                    _elapsed_milliseconds(
                        started_at
                    )
                ),
            )

        except TimeoutError:
            logger.warning(
                "agent_tool_timeout "
                "call_id=%s tool=%s",
                tool_call.call_id,
                tool_call.tool_name,
            )

            return ToolExecutionResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                status="timeout",
                error_code="tool_timeout",
                public_message=(
                    "工具执行超过允许时间"
                ),
                duration_ms=(
                    _elapsed_milliseconds(
                        started_at
                    )
                ),
            )

        except Exception as exc:
            # 只记录异常类型，不记录str(exc)或上游响应体。
            #
            # 异常内容可能包含API地址、请求参数、
            # 敏感日志或第三方响应。
            logger.warning(
                "agent_tool_execution_error "
                "call_id=%s tool=%s "
                "exception_type=%s",
                tool_call.call_id,
                tool_call.tool_name,
                type(exc).__name__,
            )

            return ToolExecutionResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                status="error",
                error_code=(
                    "tool_execution_error"
                ),
                public_message="工具执行失败",
                duration_ms=(
                    _elapsed_milliseconds(
                        started_at
                    )
                ),
            )

        # None具有明确语义：
        # 工具正常完成，但没有可用结果。
        if raw_output is None:
            return ToolExecutionResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                status="empty",
                error_code="empty_result",
                public_message=(
                    "工具没有返回可用结果"
                ),
                duration_ms=(
                    _elapsed_milliseconds(
                        started_at
                    )
                ),
            )

        try:
            # 即使Handler类型注解声明返回BaseModel，
            # Python运行时仍可能返回dict、错误模型或其他对象。
            #
            # model_validate()将原始返回值转换并校验为
            # ToolDefinition中规定的输出模型。
            validated_output: BaseModel = (
                definition.output_model
                .model_validate(
                    raw_output
                )
            )

        except ValidationError:
            logger.warning(
                "agent_tool_invalid_output "
                "call_id=%s tool=%s",
                tool_call.call_id,
                tool_call.tool_name,
            )

            return ToolExecutionResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                status="error",
                error_code=(
                    "invalid_tool_output"
                ),
                public_message=(
                    "工具返回结果未通过校验"
                ),
                duration_ms=(
                    _elapsed_milliseconds(
                        started_at
                    )
                ),
            )

        # mode="json"把datetime、UUID等对象转换成
        # 可以安全进入JSON消息和审计轨迹的值。
        serialized_output = (
            validated_output.model_dump(
                mode="json"
            )
        )

        return ToolExecutionResult(
            call_id=tool_call.call_id,
            tool_name=tool_call.tool_name,
            status="success",
            output=serialized_output,
            duration_ms=(
                _elapsed_milliseconds(
                    started_at
                )
            ),
        )
