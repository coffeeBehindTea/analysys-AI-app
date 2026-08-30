"""get_current_time工具的数据契约、Clock和执行链测试。

除SystemUtcClock的最小冒烟测试外，其余测试使用固定FakeClock，
避免断言依赖测试实际运行时刻。测试不会访问网络或设备。
"""

from datetime import (
    datetime,
    timedelta,
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
from app.agent.tools.current_time import (
    GET_CURRENT_TIME_TOOL_DEFINITION,
    GetCurrentTimeToolHandler,
    SystemUtcClock,
)
from app.schemas.agent import (
    ToolCall,
)
from app.schemas.agent_tools import (
    GetCurrentTimeToolInput,
    GetCurrentTimeToolOutput,
)


# Fake时钟固定返回该时间，使精确断言不会随测试时间变化。
FIXED_UTC_TIME = datetime(
    2026,
    8,
    24,
    8,
    30,
    0,
    tzinfo=timezone.utc,
)


class FakeUtcClock:
    """记录调用次数并返回预设值的同步Fake时钟。"""

    def __init__(
        self,
        *,
        value: object = FIXED_UTC_TIME,
    ) -> None:
        self.value = value
        self.call_count = 0

    def now_utc(
        self,
    ) -> datetime:
        """记录调用并返回预设值，错误测试可预设非datetime。"""

        self.call_count += 1

        # value故意声明为object，以便测试错误Clock实现。
        return self.value  # type: ignore[return-value]


class NonCallableClock:
    """具有同名属性但该属性不能调用的错误Clock。"""

    now_utc = "not-callable"


class UnrelatedInput(BaseModel):
    """用于验证Handler拒绝其他Pydantic输入模型。"""

    value: str


def make_registered_executor(
    clock: object,
) -> ToolExecutor:
    """使用真实定义和Handler构造完整工具执行链。"""

    registry = ToolRegistry()
    registry.register(
        definition=(
            GET_CURRENT_TIME_TOOL_DEFINITION
        ),
        handler=GetCurrentTimeToolHandler(
            clock=clock  # type: ignore[arg-type]
        ),
    )

    return ToolExecutor(
        registry=registry
    )


def test_definition_describes_low_risk_read_only_clock(
) -> None:
    """工具静态定义必须使用空输入和UTC输出契约。"""

    definition = (
        GET_CURRENT_TIME_TOOL_DEFINITION
    )

    assert definition.name == (
        "get_current_time"
    )
    assert definition.risk_level == "low"
    assert definition.read_only is True
    assert definition.timeout_seconds == 1.0
    assert definition.input_model is (
        GetCurrentTimeToolInput
    )
    assert definition.output_model is (
        GetCurrentTimeToolOutput
    )


def test_definition_exposes_empty_argument_schema(
) -> None:
    """LLM应看到空参数对象，不能提交时区或命令字段。"""

    schema = (
        GET_CURRENT_TIME_TOOL_DEFINITION
        .to_openai_tool_schema()
    )
    parameters = schema["function"][
        "parameters"
    ]

    assert schema["function"]["name"] == (
        "get_current_time"
    )
    assert parameters["properties"] == {}
    assert parameters[
        "additionalProperties"
    ] is False


def test_empty_input_accepts_only_empty_object(
) -> None:
    """空字典应通过，但任何未声明参数都必须失败。"""

    tool_input = (
        GetCurrentTimeToolInput
        .model_validate({})
    )

    assert tool_input.model_dump() == {}

    with pytest.raises(
        ValidationError
    ) as exc_info:
        GetCurrentTimeToolInput.model_validate(
            {
                "timezone": "Asia/Shanghai",
            }
        )

    assert {
        error["type"]
        for error in exc_info.value.errors()
    } == {"extra_forbidden"}


def test_output_accepts_aware_utc_time(
) -> None:
    """合法UTC时间应保留来源并可序列化为JSON。"""

    output = GetCurrentTimeToolOutput(
        current_time=FIXED_UTC_TIME
    )

    assert output.current_time == (
        FIXED_UTC_TIME
    )
    assert output.timezone == "UTC"
    assert output.source == "system_clock"
    assert output.model_dump(
        mode="json"
    ) == {
        "current_time": (
            "2026-08-24T08:30:00Z"
        ),
        "timezone": "UTC",
        "source": "system_clock",
    }


@pytest.mark.parametrize(
    ("invalid_time", "expected_message"),
    [
        (
            datetime(
                2026,
                8,
                24,
                8,
                30,
            ),
            "current_time必须包含时区",
        ),
        (
            datetime(
                2026,
                8,
                24,
                16,
                30,
                tzinfo=timezone(
                    timedelta(hours=8)
                ),
            ),
            "current_time必须是UTC时间",
        ),
    ],
)
def test_output_rejects_naive_or_non_utc_time(
    invalid_time: datetime,
    expected_message: str,
) -> None:
    """无时区时间和非零偏移时间都不能冒充UTC。"""

    with pytest.raises(
        ValidationError,
        match=expected_message,
    ):
        GetCurrentTimeToolOutput(
            current_time=invalid_time
        )


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("timezone", "Asia/Shanghai"),
        ("source", "robot_telemetry"),
    ],
)
def test_output_rejects_false_timezone_or_source_labels(
    field_name: str,
    invalid_value: str,
) -> None:
    """输出不能把系统UTC时间标记成其他来源或时区。"""

    output_data: dict[str, Any] = {
        "current_time": FIXED_UTC_TIME,
        "timezone": "UTC",
        "source": "system_clock",
    }
    output_data[field_name] = invalid_value

    with pytest.raises(ValidationError):
        GetCurrentTimeToolOutput.model_validate(
            output_data
        )


def test_system_clock_returns_time_between_two_observations(
) -> None:
    """真实Clock应返回调用时刻附近的带时区UTC时间。"""

    clock = SystemUtcClock()
    before = datetime.now(timezone.utc)
    observed = clock.now_utc()
    after = datetime.now(timezone.utc)

    assert before <= observed <= after
    assert observed.tzinfo is not None
    assert observed.utcoffset() == (
        timedelta(0)
    )


@pytest.mark.parametrize(
    "invalid_clock",
    [
        object(),
        NonCallableClock(),
    ],
)
def test_handler_constructor_requires_callable_now_utc(
    invalid_clock: object,
) -> None:
    """没有可调用now_utc方法的对象不能成为时钟依赖。"""

    with pytest.raises(
        TypeError,
        match=(
            "clock必须提供可调用的now_utc方法"
        ),
    ):
        GetCurrentTimeToolHandler(
            clock=invalid_clock  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_handler_reads_injected_clock_exactly_once(
) -> None:
    """Handler应读取Fake时钟一次并返回相同时间。"""

    clock = FakeUtcClock()
    handler = GetCurrentTimeToolHandler(
        clock=clock
    )

    result = await handler(
        GetCurrentTimeToolInput()
    )

    assert isinstance(
        result,
        GetCurrentTimeToolOutput,
    )
    assert result.current_time == (
        FIXED_UTC_TIME
    )
    assert clock.call_count == 1


@pytest.mark.asyncio
async def test_handler_rejects_unrelated_input_model(
) -> None:
    """绕过Executor直接调用时仍应进行输入类型检查。"""

    handler = GetCurrentTimeToolHandler(
        clock=FakeUtcClock()
    )

    with pytest.raises(
        TypeError,
        match=(
            "tool_input必须是"
            "GetCurrentTimeToolInput"
        ),
    ):
        await handler(
            UnrelatedInput(value="wrong")
        )


@pytest.mark.asyncio
async def test_handler_rejects_non_datetime_clock_result(
) -> None:
    """Clock返回字符串时不能交给输出模型冒充时间。"""

    handler = GetCurrentTimeToolHandler(
        clock=FakeUtcClock(
            value=(
                "2026-08-24T08:30:00Z"
            )
        )
    )

    with pytest.raises(
        TypeError,
        match=(
            "clock.now_utc必须返回datetime"
        ),
    ):
        await handler(
            GetCurrentTimeToolInput()
        )


@pytest.mark.asyncio
async def test_executor_returns_serialized_fixed_time(
) -> None:
    """完整执行链应返回成功和JSON格式UTC时间。"""

    clock = FakeUtcClock()
    executor = make_registered_executor(clock)

    result = await executor.execute(
        ToolCall(
            call_id="call_time_001",
            tool_name="get_current_time",
            arguments={},
        )
    )

    assert result.status == "success"
    assert result.error_code is None
    assert result.output == {
        "current_time": (
            "2026-08-24T08:30:00Z"
        ),
        "timezone": "UTC",
        "source": "system_clock",
    }
    assert clock.call_count == 1


@pytest.mark.asyncio
async def test_executor_rejects_undeclared_time_argument(
) -> None:
    """模型提交timezone字段时应在调用Clock前被拒绝。"""

    clock = FakeUtcClock()
    executor = make_registered_executor(clock)

    result = await executor.execute(
        ToolCall(
            call_id="call_time_002",
            tool_name="get_current_time",
            arguments={
                "timezone": "Asia/Shanghai"
            },
        )
    )

    assert result.status == "rejected"
    assert result.output is None
    assert result.error_code == (
        "invalid_tool_arguments"
    )
    assert clock.call_count == 0
