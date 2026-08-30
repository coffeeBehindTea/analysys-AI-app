"""Agent工具协议和只读工具注册表。

本模块负责：

1. 定义异步工具处理器需要满足的结构协议；
2. 将ToolDefinition与真实Python处理器绑定；
3. 拒绝重复、同步、可写或高风险工具；
4. 为AgentRunner提供工具定义和OpenAI Tool Schema。

本模块不执行工具、不调用LLM、不处理HTTP请求，
也不根据字符串动态导入Python函数。
"""

from collections.abc import (
    Awaitable,
)
from dataclasses import (
    dataclass,
)
from inspect import (
    iscoroutinefunction,
)
from typing import (
    Any,
    Protocol,
)

from pydantic import (
    BaseModel,
)

from app.schemas.agent import (
    ToolDefinition,
)


class ToolHandler(Protocol):
    """所有Agent工具处理器都应满足的异步调用协议。

    Protocol采用结构化类型检查：
    处理器不需要继承ToolHandler，
    只要具有兼容的__call__签名即可。
    """

    def __call__(
        self,
        tool_input: BaseModel,
        /,
    ) -> Awaitable[
        BaseModel | None
    ]:
        """接收校验后的输入模型，并异步返回输出模型或None。

        返回None表示工具正常完成，
        但没有找到可供Agent使用的结果。
        """

        ...


class ToolRegistryError(ValueError):
    """工具注册表配置错误的父类。"""


class DuplicateToolNameError(
    ToolRegistryError
):
    """注册表中已经存在同名工具。"""


class UnsafeToolRegistrationError(
    ToolRegistryError
):
    """工具权限或风险不符合Week 4安全边界。"""


@dataclass(
    frozen=True,
    slots=True,
)
class RegisteredTool:
    """注册表内部保存的一项可信工具。

    definition保存工具的静态数据契约；
    handler保存应用启动时明确注入的异步处理器。
    """

    definition: ToolDefinition
    handler: ToolHandler


def _is_async_callable(
    handler: object,
) -> bool:
    """判断对象是否是async def函数或异步可调用对象。"""

    # 普通async def函数会在这里被识别。
    if iscoroutinefunction(handler):
        return True

    # 某些工具会使用带有async __call__()的类实例。
    #
    # 这种对象本身不一定被iscoroutinefunction识别，
    # 因此还要检查它的__call__方法。
    call_method = getattr(
        handler,
        "__call__",
        None,
    )

    return iscoroutinefunction(
        call_method
    )


class ToolRegistry:
    """保存Agent允许调用的只读工具白名单。

    注册表只接受应用代码显式提供的ToolDefinition和处理器。
    LLM只能在这些名字中选择，不能通过工具名寻找任意函数。
    """

    def __init__(
        self,
    ) -> None:
        """创建空工具注册表。"""

        # Python 3.7及以后普通dict保持插入顺序。
        #
        # 因此list_definitions()和生成给LLM的工具列表
        # 会保持注册顺序，便于测试和审计。
        self._tools: dict[
            str,
            RegisteredTool,
        ] = {}

    def register(
        self,
        *,
        definition: ToolDefinition,
        handler: ToolHandler,
    ) -> RegisteredTool:
        """校验并注册一项只读、非高风险异步工具。"""

        # 类型注解不会在Python运行时自动拒绝dict，
        # 所以需要显式检查。
        if not isinstance(
            definition,
            ToolDefinition,
        ):
            raise TypeError(
                "definition必须是ToolDefinition"
            )

        # Week 4要求所有工具使用异步调用，
        # 防止同步I/O长期阻塞FastAPI事件循环。
        if not _is_async_callable(handler):
            raise TypeError(
                "handler必须是异步可调用对象"
            )

        # Week 4 Agent没有任何真实设备控制或
        # 外部状态写入权限。
        if not definition.read_only:
            raise UnsafeToolRegistrationError(
                "Week 4 Agent只允许注册只读工具"
            )

        # high表示可能影响设备、系统或生产状态。
        #
        # 即使定义错误地标记read_only=True，
        # high工具仍然不能进入当前注册表。
        if definition.risk_level == "high":
            raise UnsafeToolRegistrationError(
                "Week 4 Agent禁止注册高风险工具"
            )

        if definition.name in self._tools:
            raise DuplicateToolNameError(
                f"工具已经注册：{definition.name}"
            )

        registered_tool = RegisteredTool(
            definition=definition,
            handler=handler,
        )

        self._tools[
            definition.name
        ] = registered_tool

        return registered_tool

    def get(
        self,
        name: str,
    ) -> RegisteredTool | None:
        """按精确名称取得工具，未知名称返回None。"""

        # 这里不进行模糊匹配、大小写转换或动态查找。
        #
        # ToolCall已经使用ToolName校验名称格式，
        # 注册表只负责精确白名单匹配。
        return self._tools.get(name)

    def list_definitions(
        self,
    ) -> tuple[ToolDefinition, ...]:
        """按注册顺序返回不可变的工具定义快照。"""

        return tuple(
            registered.definition
            for registered
            in self._tools.values()
        )

    def build_openai_tool_schemas(
        self,
    ) -> list[dict[str, Any]]:
        """生成提供给LLM的OpenAI兼容工具描述列表。"""

        return [
            definition.to_openai_tool_schema()
            for definition
            in self.list_definitions()
        ]

    def __contains__(
        self,
        name: object,
    ) -> bool:
        """支持使用'name in registry'检查工具是否存在。"""

        return name in self._tools

    def __len__(
        self,
    ) -> int:
        """返回当前已注册工具数量。"""

        return len(self._tools)