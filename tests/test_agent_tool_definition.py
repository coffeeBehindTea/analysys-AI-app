"""Agent生命周期状态和静态工具定义的数据契约测试。

本模块只测试app.schemas.agent中的Pydantic契约，
不会调用LLM、Embedding、Chroma、HTTP或真实工具。
"""

import pytest

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
)

from app.schemas.agent import (
    AgentLifecycleState,
    ToolCall,
    ToolDefinition,
    ToolExecutionResult,
)


# 这些常量表示一项合法的教学测试工具定义。
# 多个测试共用同一组值，可以避免各测试之间出现
# 无意义的名称和说明差异。
TOOL_NAME = "search_knowledge"
TOOL_DESCRIPTION = (
    "检索机器人知识库并返回真实证据，"
    "不得执行设备控制或伪造引用。"
)


class ExampleToolInput(BaseModel):
    """测试工具接收的输入模型。"""

    model_config = ConfigDict(
        # 验证生成的JSON Schema会声明
        # 不允许工具调用提供额外参数。
        extra="forbid",
    )

    query: str = Field(
        min_length=1,
        description="需要检索的问题",
    )

    top_k: int = Field(
        default=3,
        ge=1,
        le=20,
        description="最多返回的候选数量",
    )


class ExampleToolOutput(BaseModel):
    """测试工具成功时返回的输出模型。"""

    evidence_ids: list[str] = Field(
        default_factory=list,
        description="工具实际返回的证据标识",
    )


def make_tool_definition(
    **overrides: object,
) -> ToolDefinition:
    """创建合法工具定义，并允许单项测试覆盖指定字段。"""

    # payload使用字典保存构造参数，便于失败测试只替换
    # 一个字段，其余字段仍保持完全合法。
    payload: dict[str, object] = {
        "name": TOOL_NAME,
        "description": TOOL_DESCRIPTION,
        "input_model": ExampleToolInput,
        "output_model": ExampleToolOutput,
        "risk_level": "low",
        "read_only": True,
        "timeout_seconds": 10.0,
    }

    # update()使用调用方提供的字段覆盖默认值。
    # 例如timeout_seconds=0只改变超时字段，
    # 可以确认ValidationError确实由该字段触发。
    payload.update(overrides)

    return ToolDefinition.model_validate(
        payload
    )


@pytest.mark.parametrize(
    "state",
    [
        "received",
        "planning",
        "tool_running",
        "observing",
        "completed",
        "aborted",
    ],
)
def test_agent_lifecycle_state_accepts_documented_values(
    state: str,
) -> None:
    """Agent生命周期契约应接受六种已登记状态。"""

    # AgentLifecycleState是Literal类型别名，
    # 不能像BaseModel一样直接调用model_validate()。
    # TypeAdapter为普通类型提供Pydantic运行时校验。
    adapter = TypeAdapter(
        AgentLifecycleState
    )

    assert adapter.validate_python(state) == state


def test_agent_lifecycle_state_rejects_unknown_value(
) -> None:
    """未登记的thinking状态不能进入Agent状态机。"""

    adapter = TypeAdapter(
        AgentLifecycleState
    )

    with pytest.raises(
        ValidationError,
        match="thinking",
    ):
        adapter.validate_python("thinking")


def test_tool_definition_accepts_valid_contract(
) -> None:
    """合法工具定义应保存模型、权限、风险和超时。"""

    definition = make_tool_definition(
        # 验证str_strip_whitespace会清理首尾空格。
        name=f"  {TOOL_NAME}  ",
    )

    assert definition.name == TOOL_NAME
    assert definition.description == (
        TOOL_DESCRIPTION
    )
    assert definition.input_model is (
        ExampleToolInput
    )
    assert definition.output_model is (
        ExampleToolOutput
    )
    assert definition.risk_level == "low"
    assert definition.read_only is True
    assert definition.timeout_seconds == 10.0


def test_tool_definition_builds_openai_schema(
) -> None:
    """工具定义应生成只暴露输入契约的OpenAI Tool Schema。"""

    definition = make_tool_definition()

    schema = (
        definition.to_openai_tool_schema()
    )

    assert schema["type"] == "function"

    function_schema = schema["function"]

    assert function_schema["name"] == (
        TOOL_NAME
    )
    assert function_schema["description"] == (
        TOOL_DESCRIPTION
    )

    parameters = function_schema[
        "parameters"
    ]

    # query没有默认值，因此进入required；
    # top_k有默认值，因此不是必填字段。
    assert parameters["required"] == [
        "query"
    ]
    assert parameters["properties"][
        "query"
    ]["type"] == "string"
    assert parameters["properties"][
        "top_k"
    ]["default"] == 3

    # ExampleToolInput设置了extra="forbid"，
    # 所以生成的JSON Schema禁止额外参数。
    assert parameters[
        "additionalProperties"
    ] is False

    # output_model、risk_level、read_only和timeout
    # 只供Python执行器使用，不应暴露为LLM调用参数。
    assert "output_model" not in parameters
    assert "risk_level" not in parameters
    assert "read_only" not in parameters
    assert "timeout_seconds" not in parameters


@pytest.mark.parametrize(
    "invalid_name",
    [
        "ab",
        "Search_Knowledge",
        "search-knowledge",
        "1search_knowledge",
        "search knowledge",
    ],
)
def test_tool_definition_rejects_invalid_name(
    invalid_name: str,
) -> None:
    """工具名必须使用受约束的小写snake_case格式。"""

    with pytest.raises(
        ValidationError,
        match="name",
    ):
        make_tool_definition(
            name=invalid_name
        )


@pytest.mark.parametrize(
    "invalid_timeout",
    [
        -1,
        0,
        120.1,
    ],
)
def test_tool_definition_rejects_invalid_timeout(
    invalid_timeout: float,
) -> None:
    """工具超时必须大于0秒且不能超过120秒。"""

    with pytest.raises(
        ValidationError,
        match="timeout_seconds",
    ):
        make_tool_definition(
            timeout_seconds=invalid_timeout
        )


def test_tool_definition_rejects_unknown_risk_level(
) -> None:
    """工具风险只能是low、medium或high。"""

    with pytest.raises(
        ValidationError,
        match="risk_level",
    ):
        make_tool_definition(
            risk_level="critical"
        )


@pytest.mark.parametrize(
    "model_field",
    [
        "input_model",
        "output_model",
    ],
)
def test_tool_definition_requires_pydantic_model_classes(
    model_field: str,
) -> None:
    """输入和输出契约必须是BaseModel子类而不是普通类型。"""

    with pytest.raises(
        ValidationError,
        match=model_field,
    ):
        make_tool_definition(
            **{
                model_field: dict,
            }
        )


def test_tool_definition_rejects_extra_field(
) -> None:
    """未声明字段不能偷偷进入工具定义。"""

    with pytest.raises(
        ValidationError,
        match="shell_command",
    ):
        make_tool_definition(
            shell_command="do not execute"
        )


def test_tool_definition_is_frozen_after_creation(
) -> None:
    """工具注册后不能在运行期间修改权限属性。"""

    definition = make_tool_definition()

    with pytest.raises(
        ValidationError,
        match="frozen",
    ):
        # frozen=True会阻止Pydantic模型字段赋值。
        definition.risk_level = "high"


def test_tool_call_accepts_structured_request(
) -> None:
    """合法ToolCall应清理标识并保留未校验参数字典。"""

    call = ToolCall(
        # str_strip_whitespace会清理两个标识字段。
        call_id="  call_001  ",
        tool_name="  search_knowledge  ",
        arguments={
            "query": "ERR-NET-4001",
            "top_k": 3,
        },
    )

    assert call.call_id == "call_001"
    assert call.tool_name == (
        "search_knowledge"
    )
    assert call.arguments == {
        "query": "ERR-NET-4001",
        "top_k": 3,
    }


def test_tool_call_rejects_non_object_arguments(
) -> None:
    """ToolCall的arguments必须是JSON对象而不是数组。"""

    with pytest.raises(
        ValidationError,
        match="arguments",
    ):
        ToolCall.model_validate({
            "call_id": "call_001",
            "tool_name": "search_knowledge",

            # LLM SDK解析后如果得到列表，
            # 不能把它当成工具关键字参数。
            "arguments": [
                "ERR-NET-4001",
            ],
        })


def test_tool_call_rejects_extra_top_level_field(
) -> None:
    """LLM不能在ToolCall顶层增加未声明执行字段。"""

    with pytest.raises(
        ValidationError,
        match="execute_directly",
    ):
        ToolCall.model_validate({
            "call_id": "call_001",
            "tool_name": "search_knowledge",
            "arguments": {},
            "execute_directly": True,
        })


def test_tool_call_is_frozen_after_creation(
) -> None:
    """ToolCall建立后不能替换工具名等顶层字段。"""

    call = ToolCall(
        call_id="call_001",
        tool_name="search_knowledge",
        arguments={},
    )

    with pytest.raises(
        ValidationError,
        match="frozen",
    ):
        call.tool_name = "get_robot_telemetry"


def test_tool_execution_result_accepts_success(
) -> None:
    """成功结果必须携带输出且不能携带错误信息。"""

    result = ToolExecutionResult(
        call_id="call_001",
        tool_name="search_knowledge",
        status="success",
        output={
            "evidence_ids": [
                "demo:000001",
            ],
        },
        duration_ms=12.5,
    )

    assert result.status == "success"
    assert result.output == {
        "evidence_ids": [
            "demo:000001",
        ],
    }
    assert result.error_code is None
    assert result.public_message is None
    assert result.duration_ms == 12.5


@pytest.mark.parametrize(
    (
        "status",
        "error_code",
    ),
    [
        (
            "empty",
            "empty_result",
        ),
        (
            "rejected",
            "unknown_tool",
        ),
        (
            "rejected",
            "invalid_tool_arguments",
        ),
        (
            "rejected",
            "tool_blocked",
        ),
        (
            "rejected",
            "duplicate_tool_call",
        ),
        (
            "timeout",
            "tool_timeout",
        ),
        (
            "error",
            "tool_execution_error",
        ),
        (
            "error",
            "invalid_tool_output",
        ),
        (
            "timeout",
            "vision_timeout",
        ),
        (
            "error",
            "vision_upstream_error",
        ),
        (
            "error",
            "invalid_vision_response",
        ),
    ],
)
def test_tool_execution_result_accepts_documented_failure_shape(
    status: str,
    error_code: str,
) -> None:
    """每种非成功状态应接受与之对应的稳定错误码。"""

    result = ToolExecutionResult.model_validate({
        "call_id": "call_002",
        "tool_name": "search_knowledge",
        "status": status,
        "error_code": error_code,
        "public_message": "工具没有产生可用结果",
        "duration_ms": 5.0,
    })

    assert result.status == status
    assert result.error_code == error_code
    assert result.output is None


def test_success_result_requires_output(
) -> None:
    """success不能在没有经过校验的output时成立。"""

    with pytest.raises(
        ValidationError,
        match="必须包含output",
    ):
        ToolExecutionResult(
            call_id="call_001",
            tool_name="search_knowledge",
            status="success",
            duration_ms=1.0,
        )


@pytest.mark.parametrize(
    "error_fields",
    [
        {
            "error_code": "tool_execution_error",
        },
        {
            "public_message": "不应出现的错误消息",
        },
    ],
)
def test_success_result_rejects_error_information(
    error_fields: dict[str, str],
) -> None:
    """success结果不能同时伪装成失败结果。"""

    with pytest.raises(
        ValidationError,
        match="不能包含错误信息",
    ):
        ToolExecutionResult(
            call_id="call_001",
            tool_name="search_knowledge",
            status="success",
            output={
                "evidence_ids": [],
            },
            duration_ms=1.0,
            **error_fields,
        )


def test_non_success_result_rejects_output(
) -> None:
    """失败状态不能携带可能被Agent误用的output。"""

    with pytest.raises(
        ValidationError,
        match="不能包含output",
    ):
        ToolExecutionResult(
            call_id="call_002",
            tool_name="search_knowledge",
            status="empty",
            output={
                "evidence_ids": [
                    "untrusted:000001",
                ],
            },
            error_code="empty_result",
            public_message="没有可用证据",
            duration_ms=2.0,
        )


def test_non_success_result_requires_error_code(
) -> None:
    """非成功结果必须提供机器可读错误码。"""

    with pytest.raises(
        ValidationError,
        match="必须包含error_code",
    ):
        ToolExecutionResult(
            call_id="call_002",
            tool_name="search_knowledge",
            status="empty",
            public_message="没有可用证据",
            duration_ms=2.0,
        )


def test_non_success_result_requires_public_message(
) -> None:
    """非成功结果必须提供经过清理的公开说明。"""

    with pytest.raises(
        ValidationError,
        match="必须包含public_message",
    ):
        ToolExecutionResult(
            call_id="call_002",
            tool_name="search_knowledge",
            status="empty",
            error_code="empty_result",
            duration_ms=2.0,
        )


@pytest.mark.parametrize(
    (
        "status",
        "error_code",
    ),
    [
        (
            "empty",
            "tool_timeout",
        ),
        (
            "timeout",
            "empty_result",
        ),
        (
            "rejected",
            "tool_execution_error",
        ),
        (
            "error",
            "unknown_tool",
        ),
    ],
)
def test_tool_execution_result_rejects_mismatched_error_code(
    status: str,
    error_code: str,
) -> None:
    """状态与错误码不匹配时不能建立执行结果。"""

    with pytest.raises(
        ValidationError,
        match="状态与error_code不匹配",
    ):
        ToolExecutionResult.model_validate({
            "call_id": "call_002",
            "tool_name": "search_knowledge",
            "status": status,
            "error_code": error_code,
            "public_message": "错误码与状态不匹配",
            "duration_ms": 2.0,
        })


@pytest.mark.parametrize(
    "invalid_duration",
    [
        -0.1,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_tool_execution_result_rejects_invalid_duration(
    invalid_duration: float,
) -> None:
    """耗时不能为负数、NaN或正负无穷大。"""

    with pytest.raises(
        ValidationError,
        match="duration_ms",
    ):
        ToolExecutionResult(
            call_id="call_001",
            tool_name="search_knowledge",
            status="success",
            output={
                "evidence_ids": [],
            },
            duration_ms=invalid_duration,
        )


def test_tool_execution_result_rejects_extra_field(
) -> None:
    """执行结果不能携带未声明的底层异常对象。"""

    with pytest.raises(
        ValidationError,
        match="raw_exception",
    ):
        ToolExecutionResult.model_validate({
            "call_id": "call_001",
            "tool_name": "search_knowledge",
            "status": "success",
            "output": {
                "evidence_ids": [],
            },
            "duration_ms": 1.0,
            "raw_exception": "secret upstream body",
        })


def test_tool_execution_result_is_frozen(
) -> None:
    """工具结果建立后不能把失败状态改写成成功。"""

    result = ToolExecutionResult(
        call_id="call_002",
        tool_name="search_knowledge",
        status="empty",
        error_code="empty_result",
        public_message="没有可用证据",
        duration_ms=2.0,
    )

    with pytest.raises(
        ValidationError,
        match="frozen",
    ):
        result.status = "success"
