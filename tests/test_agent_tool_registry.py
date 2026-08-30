"""Agent只读工具注册表的离线单元测试。

本模块验证工具定义与异步Handler的安全绑定，
不会真正调用Handler，也不会访问任何外部服务。
"""

from dataclasses import (
    FrozenInstanceError,
)

import pytest

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from app.agent.registry import (
    DuplicateToolNameError,
    RegisteredTool,
    ToolRegistry,
    UnsafeToolRegistrationError,
)
from app.schemas.agent import (
    ToolDefinition,
)


class ExampleRegistryInput(BaseModel):
    """注册表测试使用的工具输入契约。"""

    model_config = ConfigDict(
        extra="forbid",
    )

    query: str = Field(
        min_length=1,
    )


class ExampleRegistryOutput(BaseModel):
    """注册表测试使用的工具输出契约。"""

    result: str = Field(
        min_length=1,
    )


async def fake_async_handler(
    tool_input: BaseModel,
) -> BaseModel:
    """模拟普通async def工具处理器。"""

    # 注册表测试不会真正调用此函数。
    # 保留可执行函数体可以证明它是完整异步处理器，
    # 而不只是为了类型注解创建的占位对象。
    return ExampleRegistryOutput(
        result=str(tool_input),
    )


async def second_async_handler(
    tool_input: BaseModel,
) -> BaseModel:
    """模拟另一项工具的异步处理器。"""

    return ExampleRegistryOutput(
        result=f"second:{tool_input}",
    )


def fake_sync_handler(
    tool_input: BaseModel,
) -> BaseModel:
    """模拟会被注册表拒绝的同步处理器。"""

    return ExampleRegistryOutput(
        result=str(tool_input),
    )


class AsyncCallableHandler:
    """使用async __call__实现的可调用工具对象。"""

    async def __call__(
        self,
        tool_input: BaseModel,
    ) -> BaseModel:
        """异步返回符合输出契约的对象。"""

        return ExampleRegistryOutput(
            result=str(tool_input),
        )


def make_definition(
    *,
    name: str = "search_knowledge",
    risk_level: str = "low",
    read_only: bool = True,
) -> ToolDefinition:
    """创建一项注册表测试使用的工具定义。"""

    return ToolDefinition.model_validate({
        "name": name,
        "description": (
            f"用于离线测试{name}注册行为，"
            "不会访问真实外部服务。"
        ),
        "input_model": ExampleRegistryInput,
        "output_model": ExampleRegistryOutput,
        "risk_level": risk_level,
        "read_only": read_only,
        "timeout_seconds": 5.0,
    })


def test_empty_registry_exposes_empty_snapshots(
) -> None:
    """新注册表应为空，并返回空定义和Schema快照。"""

    registry = ToolRegistry()

    assert len(registry) == 0
    assert registry.list_definitions() == ()
    assert registry.build_openai_tool_schemas() == []
    assert registry.get("search_knowledge") is None
    assert "search_knowledge" not in registry


def test_registry_registers_async_function(
) -> None:
    """合法只读定义应与async def处理器绑定。"""

    registry = ToolRegistry()
    definition = make_definition()

    registered = registry.register(
        definition=definition,
        handler=fake_async_handler,
    )

    assert isinstance(
        registered,
        RegisteredTool,
    )
    assert registered.definition is definition
    assert registered.handler is (
        fake_async_handler
    )
    assert len(registry) == 1
    assert "search_knowledge" in registry
    assert registry.get(
        "search_knowledge"
    ) is registered
    assert registry.list_definitions() == (
        definition,
    )


def test_registry_accepts_async_callable_object(
) -> None:
    """带async __call__的实例也属于合法异步Handler。"""

    registry = ToolRegistry()
    handler = AsyncCallableHandler()

    registered = registry.register(
        definition=make_definition(),
        handler=handler,
    )

    assert registered.handler is handler


def test_registry_preserves_registration_order(
) -> None:
    """定义快照和LLM Schema应保持显式注册顺序。"""

    registry = ToolRegistry()

    search_definition = make_definition(
        name="search_knowledge",
    )
    telemetry_definition = make_definition(
        name="get_robot_telemetry",
    )

    registry.register(
        definition=search_definition,
        handler=fake_async_handler,
    )
    registry.register(
        definition=telemetry_definition,
        handler=second_async_handler,
    )

    definitions = (
        registry.list_definitions()
    )
    schemas = (
        registry.build_openai_tool_schemas()
    )

    assert definitions == (
        search_definition,
        telemetry_definition,
    )
    assert [
        schema["function"]["name"]
        for schema in schemas
    ] == [
        "search_knowledge",
        "get_robot_telemetry",
    ]


def test_registry_schema_list_is_a_fresh_snapshot(
) -> None:
    """修改返回的Schema列表不能改变注册表内部状态。"""

    registry = ToolRegistry()
    registry.register(
        definition=make_definition(),
        handler=fake_async_handler,
    )

    first_snapshot = (
        registry.build_openai_tool_schemas()
    )

    # clear()只清空调用方得到的临时列表。
    first_snapshot.clear()

    second_snapshot = (
        registry.build_openai_tool_schemas()
    )

    assert first_snapshot == []
    assert len(second_snapshot) == 1
    assert len(registry) == 1


def test_registry_rejects_duplicate_name_without_replacement(
) -> None:
    """重复注册不能覆盖最初授权的Handler。"""

    registry = ToolRegistry()
    definition = make_definition()

    original = registry.register(
        definition=definition,
        handler=fake_async_handler,
    )

    with pytest.raises(
        DuplicateToolNameError,
        match="search_knowledge",
    ):
        registry.register(
            definition=definition,
            handler=second_async_handler,
        )

    assert len(registry) == 1
    assert registry.get(
        "search_knowledge"
    ) is original
    assert original.handler is fake_async_handler


def test_registry_rejects_non_definition_object(
) -> None:
    """普通dict不能冒充已经校验的ToolDefinition。"""

    registry = ToolRegistry()

    with pytest.raises(
        TypeError,
        match="definition必须是ToolDefinition",
    ):
        registry.register(
            definition={
                "name": "search_knowledge",
            },
            handler=fake_async_handler,
        )

    assert len(registry) == 0


@pytest.mark.parametrize(
    "invalid_handler",
    [
        fake_sync_handler,
        object(),
    ],
)
def test_registry_rejects_non_async_handler(
    invalid_handler: object,
) -> None:
    """同步函数和不可调用对象都不能进入异步注册表。"""

    registry = ToolRegistry()

    with pytest.raises(
        TypeError,
        match="handler必须是异步可调用对象",
    ):
        registry.register(
            definition=make_definition(),
            handler=invalid_handler,
        )

    assert len(registry) == 0


def test_registry_rejects_writable_tool(
) -> None:
    """可能修改外部状态的工具不能进入Week 4白名单。"""

    registry = ToolRegistry()

    with pytest.raises(
        UnsafeToolRegistrationError,
        match="只允许注册只读工具",
    ):
        registry.register(
            definition=make_definition(
                name="change_robot_state",
                read_only=False,
            ),
            handler=fake_async_handler,
        )

    assert len(registry) == 0


def test_registry_rejects_high_risk_tool_even_if_read_only(
) -> None:
    """high工具不能通过错误的read_only标记绕过风险边界。"""

    registry = ToolRegistry()

    with pytest.raises(
        UnsafeToolRegistrationError,
        match="禁止注册高风险工具",
    ):
        registry.register(
            definition=make_definition(
                name="approve_robot_motion",
                risk_level="high",
                read_only=True,
            ),
            handler=fake_async_handler,
        )

    assert len(registry) == 0


def test_registry_accepts_read_only_medium_risk_draft_tool(
) -> None:
    """不产生外部副作用的medium草案工具可以注册。"""

    registry = ToolRegistry()

    registered = registry.register(
        definition=make_definition(
            name="draft_test_case",
            risk_level="medium",
            read_only=True,
        ),
        handler=fake_async_handler,
    )

    assert registered.definition.risk_level == (
        "medium"
    )
    assert registered.definition.read_only is True


def test_registry_lookup_is_exact(
) -> None:
    """注册表不能通过大小写或首尾空格模糊匹配工具。"""

    registry = ToolRegistry()
    registry.register(
        definition=make_definition(),
        handler=fake_async_handler,
    )

    assert registry.get(
        "search_knowledge"
    ) is not None
    assert registry.get(
        "Search_Knowledge"
    ) is None
    assert registry.get(
        " search_knowledge "
    ) is None


def test_registered_tool_is_frozen(
) -> None:
    """注册绑定建立后不能替换其Handler。"""

    registry = ToolRegistry()
    registered = registry.register(
        definition=make_definition(),
        handler=fake_async_handler,
    )

    with pytest.raises(FrozenInstanceError):
        registered.handler = second_async_handler

