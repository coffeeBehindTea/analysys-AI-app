"""Agent工具调用公开轨迹的统一构造模块。

本模块负责把AgentRunner产生的内部工具调用记录转换成
可以公开给客户端的脱敏轨迹。

本模块负责：

1. 根据工具类型重新校验成功输出；
2. 为工具输入生成脱敏摘要；
3. 为工具结果生成脱敏摘要；
4. 构造统一的AgentToolTraceEvent；
5. 同时返回经过校验的具体工具输出模型；
6. 让普通JSON响应和SSE实时事件复用同一套轨迹规则。

本模块不负责：

1. 执行Agent工具；
2. 调用Planner或LLM；
3. 发布SSE事件；
4. 编码text/event-stream；
5. 构造最终诊断结论；
6. 保存诊断会话。
"""

from dataclasses import (
    dataclass,
)
from typing import (
    TypeAlias,
    cast,
)

from pydantic import (
    BaseModel,
    ValidationError,
)

from app.schemas.agent import (
    ToolCall,
)
from app.schemas.agent_api import (
    AgentToolTraceEvent,
)
from app.schemas.agent_planning import (
    AgentToolInteraction,
)
from app.schemas.agent_tools import (
    DraftTestCaseToolOutput,
    GetCurrentTimeToolOutput,
    RobotTelemetryToolOutput,
    SearchKnowledgeToolOutput,
)
from app.schemas.vision import (
    VisionObservation,
)


# 所有已经有正式输出契约的只读Agent工具输出。
#
# 该联合类型用于说明typed_output可能是哪一种
# 经过Pydantic重新校验的具体工具输出。
AgentValidatedToolOutput: TypeAlias = (
    SearchKnowledgeToolOutput
    | RobotTelemetryToolOutput
    | DraftTestCaseToolOutput
    | GetCurrentTimeToolOutput
    | VisionObservation
)


# 工具名称与正式输出数据契约的唯一映射。
#
# 当ToolExecutor返回success时，本模块根据工具名称
# 找到对应Pydantic模型，并对result.output再次校验。
_TOOL_OUTPUT_MODELS: dict[
    str,
    type[BaseModel],
] = {
    "search_knowledge": (
        SearchKnowledgeToolOutput
    ),
    "get_robot_telemetry": (
        RobotTelemetryToolOutput
    ),
    "draft_test_case": (
        DraftTestCaseToolOutput
    ),
    "get_current_time": (
        GetCurrentTimeToolOutput
    ),
    "analyze_robot_image": (
        VisionObservation
    ),
}


@dataclass(
    frozen=True,
    slots=True,
)
class AgentToolTraceBuildResult:
    """一次工具轨迹构造产生的内部结果。

    trace：
        可以进入普通JSON响应或SSE事件的
        脱敏公开轨迹。

    typed_output：
        经过正式Pydantic契约重新校验的工具输出。

        非success结果和尚未配置专门输出模型的新工具，
        该字段为None。
    """

    trace: AgentToolTraceEvent

    typed_output: (
        AgentValidatedToolOutput
        | None
    )


def _compact_summary(
    value: str,
    *,
    max_length: int,
) -> str:
    """把内部说明转换成短小的公开摘要。

    split()会按照任意空白字符切分，
    再用单个空格连接，因此会清除：

    1. 换行；
    2. 制表符；
    3. 连续空格。

    如果摘要超过限制，只保留前部并添加省略号。
    """

    compact = " ".join(
        value.split()
    )

    if not compact:
        compact = "无公开摘要"

    if len(compact) > max_length:
        return (
            compact[: max_length - 1]
            + "…"
        )

    return compact


def build_agent_tool_input_summary(
    tool_call: ToolCall,
) -> str:
    """根据工具调用构造脱敏输入摘要。

    本函数只接收ToolCall，不要求工具已经完成。

    因此它既可以用于：

    1. tool_started实时SSE事件；
    2. 最终AgentToolTraceEvent。

    这里不能直接序列化tool_call.arguments，
    因为其中可能包含完整日志、检索问题、
    测试目标或其他业务数据。
    """

    if not isinstance(
        tool_call,
        ToolCall,
    ):
        raise TypeError(
            "tool_call必须是ToolCall"
        )

    tool_name = tool_call.tool_name
    arguments = tool_call.arguments

    if tool_name == "search_knowledge":
        query = arguments.get("query")

        query_length = (
            len(query)
            if isinstance(query, str)
            else 0
        )

        return _compact_summary(
            (
                "知识库检索；"
                f"query_chars={query_length}；"
                "top_k="
                f"{arguments.get('top_k', 'default')}"
            ),
            max_length=500,
        )

    if tool_name == "get_robot_telemetry":
        robot_id = arguments.get(
            "robot_id"
        )

        public_robot_id = (
            robot_id
            if isinstance(robot_id, str)
            else "invalid"
        )

        return _compact_summary(
            (
                "读取脱敏模拟遥测；"
                f"robot_id={public_robot_id}"
            ),
            max_length=500,
        )

    if tool_name == "draft_test_case":
        objective = arguments.get(
            "objective"
        )
        evidence_ids = arguments.get(
            "evidence_chunk_ids"
        )

        objective_length = (
            len(objective)
            if isinstance(objective, str)
            else 0
        )

        evidence_count = (
            len(evidence_ids)
            if isinstance(
                evidence_ids,
                (list, tuple),
            )
            else 0
        )

        return _compact_summary(
            (
                "生成测试草案；"
                "objective_chars="
                f"{objective_length}；"
                "evidence_count="
                f"{evidence_count}"
            ),
            max_length=500,
        )

    if tool_name == "get_current_time":
        return (
            "读取应用服务器UTC时间；"
            "无输入参数"
        )

    if tool_name == "analyze_robot_image":
        image_ref = arguments.get(
            "image_ref"
        )
        analysis_goal = arguments.get(
            "analysis_goal"
        )

        public_image_ref = (
            image_ref
            if isinstance(image_ref, str)
            else "invalid"
        )

        analysis_goal_length = (
            len(analysis_goal)
            if isinstance(
                analysis_goal,
                str,
            )
            else 0
        )

        return _compact_summary(
            (
                "分析请求级图片；"
                f"image_ref={public_image_ref}；"
                "analysis_goal_chars="
                f"{analysis_goal_length}"
            ),
            max_length=500,
        )

    # 尚未专门适配的新工具只公开参数数量，
    # 不公开参数名称和值。
    return _compact_summary(
        (
            "调用注册工具；"
            "argument_fields="
            f"{len(arguments)}"
        ),
        max_length=500,
    )


def _validate_success_output(
    interaction: AgentToolInteraction,
) -> AgentValidatedToolOutput | None:
    """重新校验工具成功输出。

    ToolExecutor正常情况下已经使用工具的
    output_model校验过结果。

    这里再次校验，是为了防御：

    1. 单元测试中的FakeRunner；
    2. 将来新增的Runner实现；
    3. 反序列化后的历史数据；
    4. 绕过ToolExecutor直接构造的运行结果。

    非success结果没有可信业务输出，因此返回None。
    """

    result = interaction.result

    if result.status != "success":
        return None

    output_model = _TOOL_OUTPUT_MODELS.get(
        interaction.tool_call.tool_name
    )

    # 新工具如果还没有加入输出模型映射，
    # 仍然可以生成通用脱敏轨迹，
    # 但不能把未经识别的字典当成可信观察。
    if output_model is None:
        return None

    try:
        validated_output = (
            output_model.model_validate(
                result.output
            )
        )
    except ValidationError as exc:
        # success结果却不符合正式输出契约，
        # 表示内部执行边界出现错误。
        #
        # 异常信息只包含工具名称，
        # 不包含可能敏感的原始输出。
        raise TypeError(
            interaction.tool_call.tool_name
            + "成功结果不符合输出契约"
        ) from exc

    return cast(
        AgentValidatedToolOutput,
        validated_output,
    )


def _build_result_summary(
    interaction: AgentToolInteraction,
    typed_output: (
        AgentValidatedToolOutput
        | None
    ),
) -> str:
    """生成工具结果的脱敏公开摘要。

    成功结果只公开状态、计数和必要标识，
    不复制知识库正文、完整遥测对象或图片内容。
    """

    result = interaction.result

    if result.status != "success":
        public_message = (
            result.public_message
            or "工具未返回公开说明"
        )

        return _compact_summary(
            (
                f"status={result.status}；"
                "error_code="
                f"{result.error_code}；"
                f"message={public_message}"
            ),
            max_length=1_000,
        )

    if isinstance(
        typed_output,
        SearchKnowledgeToolOutput,
    ):
        # 不公开answer和citation.excerpt，
        # 只说明取得了多少条已确认引用。
        return _compact_summary(
            (
                "知识库返回"
                f"{len(typed_output.citations)}条"
                "已确认引用；"
                "retrieval_ms="
                f"{typed_output.retrieval_ms:.3f}"
            ),
            max_length=1_000,
        )

    if isinstance(
        typed_output,
        RobotTelemetryToolOutput,
    ):
        return _compact_summary(
            (
                "取得脱敏模拟遥测；"
                f"robot_id={typed_output.robot_id}；"
                "state="
                f"{typed_output.operational_state}；"
                "fault_count="
                f"{len(typed_output.active_fault_codes)}"
            ),
            max_length=1_000,
        )

    if isinstance(
        typed_output,
        DraftTestCaseToolOutput,
    ):
        return _compact_summary(
            (
                "生成待人工批准测试草案；"
                "step_count="
                f"{len(typed_output.steps)}；"
                "evidence_count="
                f"{len(typed_output.evidence)}；"
                "draft_only=true；"
                "requires_human_approval=true"
            ),
            max_length=1_000,
        )

    if isinstance(
        typed_output,
        GetCurrentTimeToolOutput,
    ):
        # 轨迹只说明工具已经取得时间，
        # 不重复公开完整时间值。
        return (
            "已读取应用服务器UTC系统时间"
        )

    if isinstance(
        typed_output,
        VisionObservation,
    ):
        return _compact_summary(
            (
                "取得结构化视觉观察；"
                f"status={typed_output.status}；"
                "image_quality="
                f"{typed_output.image_quality}；"
                "observation_count="
                f"{len(typed_output.observations)}；"
                "indicator_count="
                f"{len(typed_output.visible_indicators)}；"
                "requires_human_check="
                f"{str(typed_output.requires_human_check).lower()}"
            ),
            max_length=1_000,
        )

    # success工具尚未配置专门输出模型时，
    # 只公开输出字段数量。
    field_count = len(
        result.output or {}
    )

    return _compact_summary(
        (
            "注册工具执行成功；"
            f"output_fields={field_count}"
        ),
        max_length=1_000,
    )


def build_agent_tool_trace(
    interaction: AgentToolInteraction,
) -> AgentToolTraceBuildResult:
    """把一条内部工具交互转换成公开轨迹。

    处理流程：

    1. 检查输入必须是AgentToolInteraction；
    2. 对success输出执行第二次Pydantic校验；
    3. 根据ToolCall生成脱敏输入摘要；
    4. 根据ToolExecutionResult生成脱敏结果摘要；
    5. 构造AgentToolTraceEvent；
    6. 同时返回公开轨迹和具体输出模型。

    SSE观察者使用trace发布tool_finished事件。

    AgentDiagnosisService使用trace构造最终执行摘要，
    并使用typed_output提取遥测、视觉观察、引用和测试草案。
    """

    if not isinstance(
        interaction,
        AgentToolInteraction,
    ):
        raise TypeError(
            "interaction必须是"
            "AgentToolInteraction"
        )

    typed_output = (
        _validate_success_output(
            interaction
        )
    )

    trace = AgentToolTraceEvent(
        step_id=interaction.step_number,
        tool_name=(
            interaction.tool_call.tool_name
        ),
        input_summary=(
            build_agent_tool_input_summary(
                interaction.tool_call
            )
        ),
        result_summary=(
            _build_result_summary(
                interaction,
                typed_output,
            )
        ),
        status=interaction.result.status,
        duration_ms=(
            interaction.result.duration_ms
        ),
        error_code=(
            interaction.result.error_code
        ),
    )

    return AgentToolTraceBuildResult(
        trace=trace,
        typed_output=typed_output,
    )