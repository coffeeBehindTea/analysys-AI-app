"""Agent单次工具执行器的异步离线测试。

测试使用内存ToolRegistry和Fake Handler，覆盖参数校验、
超时、空结果、异常脱敏、输出校验和取消传播。
"""

import asyncio
import logging

from datetime import (
    datetime,
    timezone,
)
from uuid import (
    UUID,
)

import pytest

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from app.agent.executor import (
    ToolExecutor,
)
from app.agent.registry import (
    RegisteredTool,
    ToolRegistry,
)
from app.schemas.agent import (
    ToolCall,
    ToolDefinition,
)


# Fake Handler会抛出包含该文本的异常。
# 测试使用它确认公开结果和服务端日志都没有泄漏原文。
SECRET_ERROR_TEXT = (
    "secret-api-key-and-private-upstream-body"
)


class SearchInput(BaseModel):
    """模拟search_knowledge工具的输入契约。"""

    model_config = ConfigDict(
        extra="forbid",
    )

    query: str = Field(
        min_length=1,
    )

    top_k: int = Field(
        default=3,
        ge=1,
        le=20,
    )


class SearchOutput(BaseModel):
    """模拟知识库工具的成功输出契约。"""

    evidence_ids: list[str] = Field(
        min_length=1,
    )


class WrongOutput(BaseModel):
    """故意不符合SearchOutput的数据模型。"""

    unexpected_value: str


class JsonSerializableOutput(BaseModel):
    """用于验证mode='json'序列化的数据模型。"""

    observed_at: datetime
    record_id: UUID


class RecordingSuccessHandler:
    """记录调用次数和输入对象的成功Fake Handler。"""

    def __init__(
        self,
    ) -> None:
        self.call_count = 0
        self.received_input: (
            BaseModel | None
        ) = None

    async def __call__(
        self,
        tool_input: BaseModel,
    ) -> BaseModel:
        self.call_count += 1
        self.received_input = tool_input

        return SearchOutput(
            evidence_ids=[
                "document-id:000001",
            ],
        )


class EmptyHandler:
    """正常执行但没有找到可用结果。"""

    async def __call__(
        self,
        tool_input: BaseModel,
    ) -> None:
        del tool_input
        return None


class TimeoutHandler:
    """持续等待直到被asyncio.timeout取消。"""

    def __init__(
        self,
    ) -> None:
        self.was_cancelled = False

    async def __call__(
        self,
        tool_input: BaseModel,
    ) -> BaseModel:
        del tool_input

        try:
            # 实际不会等待60秒，Executor会在更短的
            # ToolDefinition超时到达时取消当前等待。
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            self.was_cancelled = True
            raise

        return SearchOutput(
            evidence_ids=["unreachable"],
        )


class RaisingHandler:
    """抛出包含敏感文本的普通工具异常。"""

    async def __call__(
        self,
        tool_input: BaseModel,
    ) -> BaseModel:
        del tool_input
        raise RuntimeError(
            SECRET_ERROR_TEXT
        )


class InvalidOutputHandler:
    """返回错误Pydantic模型的Fake Handler。"""

    async def __call__(
        self,
        tool_input: BaseModel,
    ) -> BaseModel:
        del tool_input
        return WrongOutput(
            unexpected_value="wrong",
        )


class DictionaryOutputHandler:
    """返回可由output_model校验的普通字典。"""

    async def __call__(
        self,
        tool_input: BaseModel,
    ) -> BaseModel:
        del tool_input

        # 该返回值故意违背静态Protocol注解。
        # Executor仍应通过output_model执行运行时校验，
        # 而不是信任Python类型注解。
        return {
            "evidence_ids": [
                "dict-result:000001",
            ],
        }


class JsonOutputHandler:
    """返回包含datetime和UUID的合法输出。"""

    async def __call__(
        self,
        tool_input: BaseModel,
    ) -> BaseModel:
        del tool_input
        return JsonSerializableOutput(
            observed_at=datetime(
                2026,
                8,
                24,
                12,
                30,
                tzinfo=timezone.utc,
            ),
            record_id=UUID(
                "12345678-1234-5678-1234-567812345678"
            ),
        )


class BlockingHandler:
    """用于验证外部任务取消不会被Executor吞掉。"""

    def __init__(
        self,
    ) -> None:
        self.started = asyncio.Event()

    async def __call__(
        self,
        tool_input: BaseModel,
    ) -> BaseModel:
        del tool_input
        self.started.set()

        # 新Event永远不会自行变为已设置状态，
        # 测试会从外部取消运行该Handler的Task。
        await asyncio.Event().wait()

        return SearchOutput(
            evidence_ids=["unreachable"],
        )


def make_definition(
    *,
    name: str = "search_knowledge",
    output_model: type[BaseModel] = SearchOutput,
    timeout_seconds: float = 1.0,
    risk_level: str = "low",
    read_only: bool = True,
) -> ToolDefinition:
    """创建Executor测试使用的工具定义。"""

    return ToolDefinition.model_validate({
        "name": name,
        "description": (
            f"离线执行{name}并验证ToolExecutor安全边界。"
        ),
        "input_model": SearchInput,
        "output_model": output_model,
        "risk_level": risk_level,
        "read_only": read_only,
        "timeout_seconds": timeout_seconds,
    })


def make_call(
    *,
    tool_name: str = "search_knowledge",
    arguments: dict[str, object] | None = None,
) -> ToolCall:
    """创建一项已经通过通用结构校验的工具调用。"""

    if arguments is None:
        arguments = {
            "query": "ERR-NET-4001",
            "top_k": 3,
        }

    return ToolCall(
        call_id="call_001",
        tool_name=tool_name,
        arguments=arguments,
    )


def make_executor(
    *,
    handler: object,
    definition: ToolDefinition | None = None,
) -> ToolExecutor:
    """注册一个Fake Handler并返回对应ToolExecutor。"""

    registry = ToolRegistry()
    registry.register(
        definition=(
            definition
            if definition is not None
            else make_definition()
        ),
        handler=handler,
    )

    return ToolExecutor(
        registry=registry
    )


def test_executor_requires_registry_instance(
) -> None:
    """构造函数不能接受None或普通字典冒充注册表。"""

    with pytest.raises(
        TypeError,
        match="registry必须是ToolRegistry",
    ):
        ToolExecutor(registry=None)


def test_executor_exposes_registered_tool_schemas(
) -> None:
    """Executor应向Runner公开同一注册表中的工具白名单。"""

    executor = make_executor(
        handler=RecordingSuccessHandler()
    )

    schemas = (
        executor
        .build_openai_tool_schemas()
    )

    # tuple固定工具列表的外层顺序和数量。
    assert isinstance(schemas, tuple)
    assert len(schemas) == 1

    function_schema = schemas[0][
        "function"
    ]

    # 名称和输入JSON Schema必须来自真实ToolDefinition，
    # 不能由Runner维护另一份可能漂移的手写列表。
    assert function_schema["name"] == (
        "search_knowledge"
    )
    assert (
        function_schema
        ["parameters"]
        ["properties"]
        ["query"]
        ["type"]
        == "string"
    )


def test_executor_returns_fresh_tool_schema_snapshot(
) -> None:
    """调用方修改一次Schema结果不能污染下一次白名单快照。"""

    executor = make_executor(
        handler=RecordingSuccessHandler()
    )

    first_snapshot = (
        executor
        .build_openai_tool_schemas()
    )

    # 模拟一个有缺陷的Planner修改嵌套字典。
    first_snapshot[0][
        "function"
    ]["name"] = "mutated_tool"

    second_snapshot = (
        executor
        .build_openai_tool_schemas()
    )

    assert (
        second_snapshot[0]
        ["function"]
        ["name"]
        == "search_knowledge"
    )


@pytest.mark.asyncio
async def test_execute_requires_tool_call_instance(
) -> None:
    """执行器不能直接接收未经ToolCall校验的字典。"""

    executor = ToolExecutor(
        registry=ToolRegistry()
    )

    with pytest.raises(
        TypeError,
        match="tool_call必须是ToolCall",
    ):
        await executor.execute({
            "call_id": "call_001",
            "tool_name": "search_knowledge",
            "arguments": {},
        })


@pytest.mark.asyncio
async def test_unknown_tool_returns_rejected_result(
) -> None:
    """未知工具应稳定拒绝，不能尝试动态查找函数。"""

    executor = ToolExecutor(
        registry=ToolRegistry()
    )

    result = await executor.execute(
        make_call(
            tool_name="unknown_tool"
        )
    )

    assert result.status == "rejected"
    assert result.error_code == "unknown_tool"
    assert result.output is None
    assert result.duration_ms >= 0


@pytest.mark.asyncio
async def test_success_validates_input_and_output(
) -> None:
    """成功路径应把字典参数转换成工具输入模型。"""

    handler = RecordingSuccessHandler()
    executor = make_executor(
        handler=handler
    )

    result = await executor.execute(
        make_call()
    )

    assert handler.call_count == 1
    assert isinstance(
        handler.received_input,
        SearchInput,
    )
    assert handler.received_input.query == (
        "ERR-NET-4001"
    )
    assert handler.received_input.top_k == 3

    assert result.status == "success"
    assert result.output == {
        "evidence_ids": [
            "document-id:000001",
        ],
    }
    assert result.error_code is None


@pytest.mark.asyncio
async def test_invalid_arguments_do_not_call_handler(
) -> None:
    """具体工具参数校验失败时Handler不能执行。"""

    handler = RecordingSuccessHandler()
    executor = make_executor(
        handler=handler
    )

    result = await executor.execute(
        make_call(
            arguments={
                "query": "ERR-NET-4001",
                "top_k": 0,
            }
        )
    )

    assert result.status == "rejected"
    assert result.error_code == (
        "invalid_tool_arguments"
    )
    assert result.output is None
    assert handler.call_count == 0


@pytest.mark.asyncio
async def test_none_output_becomes_empty_result(
) -> None:
    """Handler返回None应表示正常完成但没有结果。"""

    executor = make_executor(
        handler=EmptyHandler()
    )

    result = await executor.execute(
        make_call()
    )

    assert result.status == "empty"
    assert result.error_code == "empty_result"
    assert result.output is None


@pytest.mark.asyncio
async def test_handler_timeout_returns_stable_result(
) -> None:
    """超过单工具超时后应取消Handler并返回tool_timeout。"""

    handler = TimeoutHandler()
    executor = make_executor(
        handler=handler,
        definition=make_definition(
            timeout_seconds=0.01
        ),
    )

    result = await executor.execute(
        make_call()
    )

    assert result.status == "timeout"
    assert result.error_code == "tool_timeout"
    assert result.output is None
    assert handler.was_cancelled is True


@pytest.mark.asyncio
async def test_handler_exception_is_sanitized(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """底层异常原文不能进入公开结果或服务端日志。"""

    executor = make_executor(
        handler=RaisingHandler()
    )

    with caplog.at_level(
        logging.WARNING,
        logger="app.agent.executor",
    ):
        result = await executor.execute(
            make_call()
        )

    assert result.status == "error"
    assert result.error_code == (
        "tool_execution_error"
    )
    assert result.public_message == (
        "工具执行失败"
    )
    assert SECRET_ERROR_TEXT not in (
        result.model_dump_json()
    )
    assert SECRET_ERROR_TEXT not in caplog.text
    assert "RuntimeError" in caplog.text


@pytest.mark.asyncio
async def test_invalid_handler_output_returns_error(
) -> None:
    """错误输出模型不能被当成成功工具结果。"""

    executor = make_executor(
        handler=InvalidOutputHandler()
    )

    result = await executor.execute(
        make_call()
    )

    assert result.status == "error"
    assert result.error_code == (
        "invalid_tool_output"
    )
    assert result.output is None


@pytest.mark.asyncio
async def test_runtime_validation_accepts_valid_dictionary_output(
) -> None:
    """运行时output_model应校验而不是盲信Handler注解。"""

    executor = make_executor(
        handler=DictionaryOutputHandler()
    )

    result = await executor.execute(
        make_call()
    )

    assert result.status == "success"
    assert result.output == {
        "evidence_ids": [
            "dict-result:000001",
        ],
    }


@pytest.mark.asyncio
async def test_output_is_serialized_in_json_mode(
) -> None:
    """datetime和UUID应转换成JSON字符串而不是Python对象。"""

    executor = make_executor(
        handler=JsonOutputHandler(),
        definition=make_definition(
            output_model=(
                JsonSerializableOutput
            ),
        ),
    )

    result = await executor.execute(
        make_call()
    )

    assert result.status == "success"
    assert isinstance(
        result.output["observed_at"],
        str,
    )
    assert result.output["observed_at"] == (
        "2026-08-24T12:30:00Z"
    )
    assert result.output["record_id"] == (
        "12345678-1234-5678-1234-567812345678"
    )


@pytest.mark.asyncio
async def test_executor_blocks_unsafe_definition_defensively(
) -> None:
    """即使内部注册表被错误污染，Executor仍拒绝high工具。"""

    registry = ToolRegistry()
    unsafe_definition = make_definition(
        name="approve_robot_motion",
        risk_level="high",
        read_only=True,
    )
    handler = RecordingSuccessHandler()

    # 正常register()会拒绝该定义。
    # 这里故意模拟未来代码缺陷或内部状态损坏，
    # 直接污染私有字典以验证Executor的纵深防御。
    registry._tools[
        unsafe_definition.name
    ] = RegisteredTool(
        definition=unsafe_definition,
        handler=handler,
    )

    executor = ToolExecutor(
        registry=registry
    )

    result = await executor.execute(
        make_call(
            tool_name="approve_robot_motion"
        )
    )

    assert result.status == "rejected"
    assert result.error_code == "tool_blocked"
    assert handler.call_count == 0


@pytest.mark.asyncio
async def test_external_cancellation_propagates(
) -> None:
    """上层取消请求时Executor不能转换成普通工具错误。"""

    handler = BlockingHandler()
    executor = make_executor(
        handler=handler,
        definition=make_definition(
            timeout_seconds=10.0
        ),
    )

    task = asyncio.create_task(
        executor.execute(
            make_call()
        )
    )

    # 等待Handler真正开始后再取消Task，
    # 避免取消发生在测试协程尚未进入Handler之前。
    await handler.started.wait()
    task.cancel()

    with pytest.raises(
        asyncio.CancelledError
    ):
        await task
