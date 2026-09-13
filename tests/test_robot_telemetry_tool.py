"""get_robot_telemetry工具的结构、Store、Handler和执行链测试。

所有遥测记录都在测试内存中创建，测试不会连接真实机器人、
PLC、RCS、RMS、数据库、网络服务或本地数据文件。
"""

from datetime import (
    datetime,
    timezone,
)
from typing import (
    Any,
)

import pytest

from pydantic import (
    BaseModel,
    ValidationError,
)

from app.agent.executor import (
    ToolExecutor,
)
from app.agent.registry import (
    ToolRegistry,
)
from app.agent.tools.robot_telemetry import (
    GET_ROBOT_TELEMETRY_TOOL_DEFINITION,
    GetRobotTelemetryToolHandler,
    InMemoryRobotTelemetryStore,
)
from app.schemas.agent import (
    ToolCall,
)
from app.schemas.agent_tools import (
    GetRobotTelemetryToolInput,
    RobotTelemetryToolOutput,
)


# 使用固定UTC时间，使JSON序列化和字段断言始终可重复。
TEST_OBSERVED_AT = datetime(
    2026,
    8,
    24,
    6,
    24,
    44,
    tzinfo=timezone.utc,
)


class UnrelatedInput(BaseModel):
    """用于模拟绕过Executor传入的错误Pydantic模型。"""

    value: str


class RecordingTelemetryStore:
    """记录精确查询编号并返回预设对象的Fake Store。"""

    def __init__(
        self,
        *,
        response: object,
    ) -> None:
        self.response = response
        self.robot_ids: list[str] = []

    def get(
        self,
        robot_id: str,
        /,
    ) -> RobotTelemetryToolOutput | None:
        """记录查询并返回测试预设值。"""

        self.robot_ids.append(robot_id)

        # object允许错误分支故意返回dict，
        # 验证Handler是否执行运行时返回类型检查。
        return self.response  # type: ignore[return-value]


class StoreWithoutGet:
    """故意缺少get()方法的错误依赖。"""


def make_snapshot(
    **overrides: Any,
) -> RobotTelemetryToolOutput:
    """创建一条固定、脱敏、带时区的模拟遥测快照。"""

    values: dict[str, Any] = {
        "robot_id": "robot-001",
        "observed_at": TEST_OBSERVED_AT,
        "location": (
            "warehouse-demo/aisle-07/node-14"
        ),
        "battery_percent": 42.5,
        "operational_state": "paused",
        "current_task_id": "task-demo-1001",
        "active_fault_codes": (
            "ERR-NET-4001",
        ),
        "speed_mps": 0.0,
        "network_connected": True,
    }
    values.update(overrides)

    return RobotTelemetryToolOutput.model_validate(
        values
    )


def make_handler_and_store(
    *,
    response: object,
) -> tuple[
    GetRobotTelemetryToolHandler,
    RecordingTelemetryStore,
]:
    """创建共享同一个Fake Store的Handler测试组合。"""

    store = RecordingTelemetryStore(
        response=response
    )
    handler = GetRobotTelemetryToolHandler(
        store=store
    )

    return handler, store


def make_registered_executor(
    *,
    records: list[
        RobotTelemetryToolOutput
    ],
) -> ToolExecutor:
    """将真实内存Store和Handler接入注册表及执行器。"""

    store = InMemoryRobotTelemetryStore(
        records=records
    )
    handler = GetRobotTelemetryToolHandler(
        store=store
    )
    registry = ToolRegistry()
    registry.register(
        definition=(
            GET_ROBOT_TELEMETRY_TOOL_DEFINITION
        ),
        handler=handler,
    )

    return ToolExecutor(
        registry=registry
    )


def test_input_strips_robot_id(
) -> None:
    """输入契约应清理机器人编号首尾空白。"""

    tool_input = GetRobotTelemetryToolInput(
        robot_id="  robot-001  "
    )

    assert tool_input.robot_id == "robot-001"


def test_input_rejects_unexpected_control_field(
) -> None:
    """输入契约不能接受LLM添加的设备控制字段。"""

    with pytest.raises(
        ValidationError
    ) as exc_info:
        GetRobotTelemetryToolInput.model_validate({
            "robot_id": "robot-001",
            "resume_task": True,
        })

    assert (
        exc_info.value.errors()[0]["type"]
        == "extra_forbidden"
    )


@pytest.mark.parametrize(
    ("overrides", "expected_error_type"),
    [
        (
            {"observed_at": datetime(2026, 8, 24)},
            "value_error",
        ),
        (
            {"battery_percent": -0.1},
            "greater_than_equal",
        ),
        (
            {"battery_percent": 100.1},
            "less_than_equal",
        ),
        (
            {
                "active_fault_codes": (
                    "ERR_NET_4001",
                )
            },
            "string_pattern_mismatch",
        ),
        (
            {
                "active_fault_codes": (
                    "ERR-NET-4001",
                    "ERR-NET-4001",
                )
            },
            "value_error",
        ),
        (
            {
                "operational_state": (
                    "executing_task"
                ),
                "current_task_id": None,
            },
            "value_error",
        ),
        (
            {"source": "real_robot"},
            "literal_error",
        ),
    ],
)
def test_snapshot_rejects_invalid_or_ambiguous_values(
    overrides: dict[str, Any],
    expected_error_type: str,
) -> None:
    """快照契约应拒绝无时区、越界值和伪造来源。"""

    with pytest.raises(
        ValidationError
    ) as exc_info:
        make_snapshot(**overrides)

    error_types = {
        error["type"]
        for error in exc_info.value.errors()
    }

    assert expected_error_type in error_types


def test_snapshot_json_serialization_preserves_simulated_source(
) -> None:
    """datetime和tuple应转换成JSON值并保留模拟来源。"""

    serialized = make_snapshot().model_dump(
        mode="json"
    )

    assert serialized["observed_at"] == (
        "2026-08-24T06:24:44Z"
    )
    assert serialized[
        "active_fault_codes"
    ] == ["ERR-NET-4001"]
    assert serialized["source"] == (
        "simulated_memory"
    )


def test_store_indexes_records_and_detaches_source_iterable(
) -> None:
    """Store应建立自己的字典，不继续依赖外部列表。"""

    source_records = [make_snapshot()]
    store = InMemoryRobotTelemetryStore(
        records=source_records
    )

    # 修改原始列表不应改变Store已经建立的索引。
    source_records.append(
        make_snapshot(robot_id="robot-002")
    )

    assert len(store) == 1
    assert store.get("robot-001") == (
        make_snapshot()
    )
    assert store.get("robot-002") is None


def test_store_uses_exact_robot_id_matching(
) -> None:
    """Store不能通过大小写或相似编号返回其他机器人。"""

    store = InMemoryRobotTelemetryStore(
        records=[make_snapshot()]
    )

    assert store.get("robot-001") is not None
    assert store.get("ROBOT-001") is None
    assert store.get("robot-01") is None


def test_store_rejects_unvalidated_record(
) -> None:
    """普通字典不能绕过RobotTelemetryToolOutput进入Store。"""

    with pytest.raises(
        TypeError,
        match=(
            "records中的第 0 项必须是"
            "RobotTelemetryToolOutput"
        ),
    ):
        InMemoryRobotTelemetryStore(
            records=[
                {
                    "robot_id": "robot-001"
                }
            ]
        )


def test_store_rejects_duplicate_robot_id(
) -> None:
    """同一Store不能包含两个相同robot_id的当前快照。"""

    with pytest.raises(
        ValueError,
        match=(
            "模拟遥测包含重复robot_id："
            "robot-001"
        ),
    ):
        InMemoryRobotTelemetryStore(
            records=[
                make_snapshot(),
                make_snapshot(
                    location=(
                        "warehouse-demo/aisle-08"
                    )
                ),
            ]
        )


def test_store_requires_string_robot_id(
) -> None:
    """直接调用Store时也不能使用非字符串编号。"""

    store = InMemoryRobotTelemetryStore(
        records=[make_snapshot()]
    )

    with pytest.raises(
        TypeError,
        match="robot_id必须是字符串",
    ):
        store.get(1)


def test_definition_exposes_read_only_telemetry_schema(
) -> None:
    """工具定义应只公开robot_id并明确低风险只读属性。"""

    definition = (
        GET_ROBOT_TELEMETRY_TOOL_DEFINITION
    )
    openai_schema = (
        definition.to_openai_tool_schema()
    )

    assert definition.name == (
        "get_robot_telemetry"
    )
    assert definition.risk_level == "low"
    assert definition.read_only is True
    assert definition.timeout_seconds == 2.0
    assert definition.input_model is (
        GetRobotTelemetryToolInput
    )
    assert definition.output_model is (
        RobotTelemetryToolOutput
    )

    parameters = (
        openai_schema["function"]["parameters"]
    )
    assert set(parameters["properties"]) == {
        "robot_id"
    }
    assert parameters[
        "additionalProperties"
    ] is False


def test_handler_requires_store_get_method(
) -> None:
    """Handler构造时应拒绝没有get()方法的对象。"""

    with pytest.raises(
        TypeError,
        match=(
            "store必须提供可调用的get方法"
        ),
    ):
        GetRobotTelemetryToolHandler(
            store=StoreWithoutGet()
        )


@pytest.mark.asyncio
async def test_handler_returns_requested_snapshot(
) -> None:
    """Handler应把经过校验的精确robot_id交给Store。"""

    snapshot = make_snapshot()
    handler, store = make_handler_and_store(
        response=snapshot
    )

    result = await handler(
        GetRobotTelemetryToolInput(
            robot_id="robot-001"
        )
    )

    assert store.robot_ids == ["robot-001"]
    assert result is snapshot


@pytest.mark.asyncio
async def test_handler_maps_unknown_robot_to_none(
) -> None:
    """Store没有记录时Handler应返回None而不是伪造快照。"""

    handler, store = make_handler_and_store(
        response=None
    )

    result = await handler(
        GetRobotTelemetryToolInput(
            robot_id="robot-999"
        )
    )

    assert result is None
    assert store.robot_ids == ["robot-999"]


@pytest.mark.asyncio
async def test_handler_rejects_unrelated_input_before_store_call(
) -> None:
    """错误输入模型不能到达Store。"""

    handler, store = make_handler_and_store(
        response=make_snapshot()
    )

    with pytest.raises(
        TypeError,
        match=(
            "tool_input必须是"
            "GetRobotTelemetryToolInput"
        ),
    ):
        await handler(
            UnrelatedInput(value="wrong")
        )

    assert store.robot_ids == []


@pytest.mark.asyncio
async def test_handler_rejects_invalid_store_output(
) -> None:
    """Store返回未经遥测契约校验的dict时Handler必须拒绝。"""

    handler, store = make_handler_and_store(
        response={
            "robot_id": "robot-001",
            "source": "simulated_memory",
        }
    )

    with pytest.raises(
        TypeError,
        match=(
            "store.get必须返回"
            "RobotTelemetryToolOutput或None"
        ),
    ):
        await handler(
            GetRobotTelemetryToolInput(
                robot_id="robot-001"
            )
        )

    assert store.robot_ids == ["robot-001"]


@pytest.mark.asyncio
async def test_handler_rejects_cross_robot_snapshot(
) -> None:
    """查询robot-001时不能接受Store返回的robot-002快照。"""

    handler, store = make_handler_and_store(
        response=make_snapshot(
            robot_id="robot-002"
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "遥测快照robot_id与查询不一致"
        ),
    ):
        await handler(
            GetRobotTelemetryToolInput(
                robot_id="robot-001"
            )
        )

    assert store.robot_ids == ["robot-001"]


@pytest.mark.asyncio
async def test_executor_returns_serialized_telemetry_snapshot(
) -> None:
    """完整工具链应返回带模拟来源的JSON安全遥测。"""

    executor = make_registered_executor(
        records=[make_snapshot()]
    )

    result = await executor.execute(
        ToolCall(
            call_id="call_telemetry_001",
            tool_name="get_robot_telemetry",
            arguments={
                "robot_id": "robot-001"
            },
        )
    )

    assert result.status == "success"
    assert result.error_code is None
    assert result.output is not None
    assert result.output["robot_id"] == (
        "robot-001"
    )
    assert result.output["observed_at"] == (
        "2026-08-24T06:24:44Z"
    )
    assert result.output[
        "active_fault_codes"
    ] == ["ERR-NET-4001"]
    assert result.output["source"] == (
        "simulated_memory"
    )


@pytest.mark.asyncio
async def test_executor_maps_unknown_robot_to_empty_result(
) -> None:
    """未知robot_id应成为稳定empty结果而不是工具错误。"""

    executor = make_registered_executor(
        records=[make_snapshot()]
    )

    result = await executor.execute(
        ToolCall(
            call_id="call_telemetry_002",
            tool_name="get_robot_telemetry",
            arguments={
                "robot_id": "robot-999"
            },
        )
    )

    assert result.status == "empty"
    assert result.error_code == "empty_result"
    assert result.output is None


@pytest.mark.asyncio
async def test_executor_rejects_extra_control_argument_early(
) -> None:
    """LLM添加控制参数时Store不能被调用。"""

    snapshot = make_snapshot()
    handler, store = make_handler_and_store(
        response=snapshot
    )
    registry = ToolRegistry()
    registry.register(
        definition=(
            GET_ROBOT_TELEMETRY_TOOL_DEFINITION
        ),
        handler=handler,
    )
    executor = ToolExecutor(
        registry=registry
    )

    result = await executor.execute(
        ToolCall(
            call_id="call_telemetry_003",
            tool_name="get_robot_telemetry",
            arguments={
                "robot_id": "robot-001",
                "resume_task": True,
            },
        )
    )

    assert result.status == "rejected"
    assert result.error_code == (
        "invalid_tool_arguments"
    )
    assert result.output is None
    assert store.robot_ids == []
