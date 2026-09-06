"""Robot Diagnostic Agent公开API的数据契约。

本模块定义：

1. POST /api/v1/agent/diagnose请求体；
2. 一次工具执行的脱敏公开轨迹；
3. Agent循环终止状态的公开摘要；
4. 最终诊断、模拟遥测和测试草案的响应包装。

本模块不运行Agent、不执行工具、不调用LLM，
也不允许把Planner原始输出直接作为API响应。
"""

from typing import (
    Annotated,
    Literal,
    Self,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from app.schemas.agent import (
    ToolErrorCode,
    ToolExecutionStatus,
    ToolName,
)
from app.schemas.agent_planning import (
    AgentPlannerFinishReason,
)
from app.schemas.agent_runtime import (
    AgentMissingInformation,
    AgentRunTerminationReason,
)
from app.schemas.agent_tools import (
    AgentImageReference,
    DraftTestCaseToolOutput,
    RobotTelemetryToolOutput,
)
from app.schemas.diagnostics import (
    DiagnosisReport,
)
from app.schemas.vision import (
    VisionImagePayload,
    VisionObservation,
)


# 一次Agent诊断请求最多接受3张图片。
#
# 这个限制同时控制：
#
# 1. HTTP请求体大小；
# 2. 单次Agent可选择的视觉工具范围；
# 3. Vision调用次数、延迟和费用的上界。
#
# 它不是VisionInputAdapter的单张图片大小限制。
# 单张图片仍然必须经过适配器的字节、尺寸和格式校验。
MAX_AGENT_DIAGNOSIS_IMAGES = 3


# 公开轨迹中的摘要字段必须足够短。
#
# 完整用户日志、文档正文和工具输出不能直接进入轨迹，
# 防止API响应泄露敏感数据或产生过大的响应体。
AgentTraceInputSummary = Annotated[
    str,
    Field(
        min_length=1,
        max_length=500,
        description=(
            "经过脱敏和长度限制的工具输入摘要"
        ),
    ),
]


AgentTraceResultSummary = Annotated[
    str,
    Field(
        min_length=1,
        max_length=1_000,
        description=(
            "经过脱敏和长度限制的工具结果摘要"
        ),
    ),
]


class AgentDiagnosisRequest(BaseModel):
    """POST /api/v1/agent/diagnose的JSON请求体。

    用户只能提供任务所需的故障上下文，
    不能通过请求修改工具白名单、最大步数、
    Planner超时、证据门槛或风险权限。
    """

    model_config = ConfigDict(
        # 清除普通字符串字段首尾空白。
        str_strip_whitespace=True,

        # 拒绝run_shell、max_steps、tools等
        # 没有在API契约中声明的额外字段。
        extra="forbid",
    )

    robot_id: str = Field(
        min_length=1,
        max_length=100,
        description=(
            "需要诊断的机器人编号"
        ),
        examples=["robot-001"],
    )

    symptom: str = Field(
        min_length=1,
        max_length=2_000,
        description=(
            "用户观察到的故障现象"
        ),
        examples=[
            "机器人网络恢复后仍未继续任务"
        ],
    )

    log_excerpt: str = Field(
        min_length=1,
        max_length=10_000,
        description=(
            "已经脱敏的日志摘要，"
            "不能提交完整敏感日志"
        ),
        examples=[
            (
                "ERR-NET-4001 heartbeat timeout "
                "exceeded 1500 ms"
            )
        ],
    )

    # None表示使用默认诊断目标。
    #
    # 提供该字段时，它仍然属于不可信用户数据，
    # 不能扩大Agent工具权限。
    task_goal: str | None = Field(
        default=None,
        min_length=1,
        max_length=2_000,
        description=(
            "可选任务目标；"
            "例如要求核对故障原因和恢复条件"
        ),
        examples=[
            "核对故障原因、当前遥测和安全恢复条件"
        ],
    )

    # 保留空列表默认值，使原有纯文本Agent请求继续兼容。
    #
    # 客户端只提交图片载荷，不提交image_ref。
    # Service会在适配成功后按请求内顺序生成
    # image_001、image_002等不透明引用，防止客户端
    # 伪造引用去读取其他请求中的图片。
    images: list[
        VisionImagePayload
    ] = Field(
        default_factory=list,
        max_length=(
            MAX_AGENT_DIAGNOSIS_IMAGES
        ),
        description=(
            "本次诊断附带的图片；"
            "每张图片必须是Base64载荷，"
            "完整图片不会进入公开响应或执行轨迹"
        ),
    )


class AgentToolTraceEvent(BaseModel):
    """公开响应中的一次脱敏工具执行记录。

    一条事件对应一个已经实际完成的
    AgentToolInteraction。

    它不会包含完整arguments、完整output、
    API Key、模型私有思维链或完整敏感日志。
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    step_id: int = Field(
        ge=1,
        le=20,
        description=(
            "从1开始连续递增的工具执行步骤编号"
        ),
    )

    # 当前公开轨迹每一项都表示一次工具执行。
    #
    # received、planning等生命周期状态由
    # AgentExecutionSummary统一表达，
    # 不伪造没有实际执行结果的工具事件。
    event_type: Literal[
        "tool_execution"
    ] = Field(
        default="tool_execution",
        description=(
            "当前事件固定表示一次工具执行"
        ),
    )

    tool_name: ToolName = Field(
        description=(
            "本步骤实际请求的注册工具名称"
        ),
    )

    input_summary: (
        AgentTraceInputSummary
    ) = Field(
        description=(
            "工具参数的脱敏摘要，"
            "不是原始arguments"
        ),
    )

    result_summary: (
        AgentTraceResultSummary
    ) = Field(
        description=(
            "工具结果或公开错误的脱敏摘要，"
            "不是原始output"
        ),
    )

    status: ToolExecutionStatus = Field(
        description=(
            "ToolExecutor返回的结构化执行状态"
        ),
    )

    duration_ms: float = Field(
        ge=0.0,
        description=(
            "本次工具执行耗时，单位为毫秒"
        ),
    )

    error_code: ToolErrorCode | None = Field(
        default=None,
        description=(
            "非成功状态对应的稳定机器错误码"
        ),
    )

    @model_validator(mode="after")
    def status_and_error_code_must_match(
        self,
    ) -> Self:
        """公开轨迹状态必须与错误码一致。

        虽然内部ToolExecutionResult已经做过校验，
        公开轨迹仍要独立校验，防止Service转换错误。
        """

        if self.status == "success":
            if self.error_code is not None:
                raise ValueError(
                    "success轨迹不能包含error_code"
                )

            return self

        # 每种非成功状态只能使用对应错误码。
        expected_error_codes: dict[
            ToolExecutionStatus,
            set[ToolErrorCode],
        ] = {
            "success": set(),

            "empty": {
                "empty_result",
            },

            "rejected": {
                "unknown_tool",
                "invalid_tool_arguments",
                "tool_blocked",
                "duplicate_tool_call",
            },

            "timeout": {
                "tool_timeout",
                "vision_timeout",
            },

            "error": {
                "tool_execution_error",
                "invalid_tool_output",
                "vision_upstream_error",
                "invalid_vision_response",
            },
        }

        if self.error_code is None:
            raise ValueError(
                "非success轨迹必须包含error_code"
            )

        if (
            self.error_code
            not in expected_error_codes[
                self.status
            ]
        ):
            raise ValueError(
                "轨迹状态与error_code不匹配"
            )

        return self


class AgentExecutionSummary(BaseModel):
    """Agent循环的公开执行摘要。

    本模型保留可审计终止信息，
    但不返回Planner原始final_message，
    因为它仍属于未经最终校验的模型输出。
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        frozen=True,
    )

    planner_prompt_version: str = Field(
        min_length=1,
        max_length=100,
        description=(
            "本次运行使用的Planner Prompt版本"
        ),
    )

    state: Literal[
        "completed",
        "aborted",
    ] = Field(
        description=(
            "Agent循环正常结束还是安全中止"
        ),
    )

    termination_reason: (
        AgentRunTerminationReason
    ) = Field(
        description=(
            "由Python控制逻辑决定的终止原因"
        ),
    )

    termination_message: str = Field(
        min_length=1,
        max_length=500,
        description=(
            "由应用产生、可以公开的终止说明"
        ),
    )

    finish_reason: (
        AgentPlannerFinishReason | None
    ) = Field(
        default=None,
        description=(
            "正常结束时Planner提供的结构化原因"
        ),
    )

    step_count: int = Field(
        ge=0,
        le=20,
        description=(
            "实际完成的工具执行步骤数量"
        ),
    )

    steps: list[
        AgentToolTraceEvent
    ] = Field(
        default_factory=list,
        max_length=20,
        description=(
            "按执行顺序排列的脱敏工具轨迹"
        ),
    )

    missing_information: list[
        AgentMissingInformation
    ] = Field(
        default_factory=list,
        max_length=20,
        description=(
            "Agent结束时仍缺少或无法确认的信息"
        ),
    )

    @field_validator(
        "missing_information"
    )
    @classmethod
    def missing_information_must_be_unique(
        cls,
        values: list[str],
    ) -> list[str]:
        """公开缺失信息不能重复。"""

        if len(values) != len(set(values)):
            raise ValueError(
                "missing_information"
                "不能包含重复项"
            )

        return values

    @model_validator(mode="after")
    def validate_execution_summary(
        self,
    ) -> Self:
        """校验步骤编号和终止字段关系。"""

        if self.step_count != len(
            self.steps
        ):
            raise ValueError(
                "step_count必须等于steps数量"
            )

        actual_step_ids = [
            step.step_id
            for step in self.steps
        ]

        expected_step_ids = list(
            range(
                1,
                len(self.steps) + 1,
            )
        )

        if actual_step_ids != expected_step_ids:
            raise ValueError(
                "steps.step_id必须从1开始"
                "连续递增"
            )

        if self.state == "completed":
            if (
                self.termination_reason
                != "planner_finished"
            ):
                raise ValueError(
                    "completed执行必须由"
                    "planner_finished结束"
                )

            if self.finish_reason is None:
                raise ValueError(
                    "completed执行必须包含"
                    "finish_reason"
                )

            if (
                self.finish_reason
                == "task_completed"
                and self.missing_information
            ):
                raise ValueError(
                    "task_completed不能同时包含"
                    "missing_information"
                )

            if (
                self.finish_reason
                != "task_completed"
                and not self.missing_information
            ):
                raise ValueError(
                    "信息不足或需要人工复核时"
                    "必须说明missing_information"
                )

            return self

        # state只能是completed或aborted，
        # 因此前面没有返回时必然是aborted。
        if (
            self.termination_reason
            == "planner_finished"
        ):
            raise ValueError(
                "aborted执行不能使用"
                "planner_finished"
            )

        if self.finish_reason is not None:
            raise ValueError(
                "aborted执行不能包含"
                "finish_reason"
            )

        if not self.missing_information:
            raise ValueError(
                "aborted执行必须说明"
                "missing_information"
            )

        return self


class AgentVisionObservation(BaseModel):
    """一次视觉工具调用产生的公开观察。

    image_ref只在当前请求内有效，用于把响应中的观察
    与Planner实际选择的图片关联起来。

    observation保留source="vision_model"、图片SHA-256、
    Prompt版本和模型名，但不包含原始图片字节。
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        frozen=True,
    )

    step_id: int = Field(
        ge=1,
        le=20,
        description=(
            "产生这项视觉观察的Agent工具步骤编号"
        ),
    )

    image_ref: AgentImageReference = Field(
        description=(
            "当前请求内由服务端分配的不透明图片引用"
        ),
    )

    observation: VisionObservation = Field(
        description=(
            "Vision Provider返回的结构化可见观察；"
            "它不是知识库确认的工程事实"
        ),
    )


class AgentDiagnosisResponse(BaseModel):
    """POST /api/v1/agent/diagnose的成功响应。

    diagnosis沿用第三周DiagnosisReport，
    execution提供Week 4 Agent执行审计信息。

    视觉观察、模拟遥测和测试草案使用各自明确的
    Pydantic输出契约，不能返回任意工具字典。
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        frozen=True,
    )

    request_id: str = Field(
        min_length=1,
        max_length=200,
        description=(
            "用于关联响应头、日志和诊断报告的请求标识"
        ),
    )

    diagnosis: DiagnosisReport = Field(
        description=(
            "沿用第三周证据与引用白名单规则的"
            "结构化诊断结果"
        ),
    )

    execution: AgentExecutionSummary = Field(
        description=(
            "Agent循环的公开终止信息和脱敏轨迹"
        ),
    )

    vision_observations: list[
        AgentVisionObservation
    ] = Field(
        default_factory=list,
        max_length=20,
        description=(
            "Agent实际取得的结构化视觉观察；"
            "没有成功调用视觉工具时为空"
        ),
    )

    telemetry_observations: list[
        RobotTelemetryToolOutput
    ] = Field(
        default_factory=list,
        max_length=10,
        description=(
            "Agent实际读取到的脱敏模拟遥测；"
            "没有调用遥测工具时为空"
        ),
    )

    test_case_drafts: list[
        DraftTestCaseToolOutput
    ] = Field(
        default_factory=list,
        max_length=5,
        description=(
            "Agent实际生成的待人工批准测试草案；"
            "没有调用草案工具时为空"
        ),
    )

    @field_validator(
        "telemetry_observations"
    )
    @classmethod
    def telemetry_robot_ids_must_be_unique(
        cls,
        values: list[
            RobotTelemetryToolOutput
        ],
    ) -> list[RobotTelemetryToolOutput]:
        """同一响应不能重复返回同一机器人快照。"""

        robot_ids = [
            item.robot_id
            for item in values
        ]

        if len(robot_ids) != len(
            set(robot_ids)
        ):
            raise ValueError(
                "telemetry_observations"
                "不能包含重复robot_id"
            )

        return values

    @model_validator(mode="after")
    def validate_response_relationships(
        self,
    ) -> Self:
        """校验请求ID和测试草案证据来源。"""

        if (
            self.request_id
            != self.diagnosis.request_id
        ):
            raise ValueError(
                "响应request_id必须与"
                "diagnosis.request_id一致"
            )

        # 视觉观察必须与公开执行轨迹中的一次成功
        # analyze_robot_image调用一一对应。
        #
        # 这样，响应不能凭空添加模型观察，
        # 也不能隐藏一次已经成功执行的视觉分析。
        successful_vision_step_ids = {
            step.step_id
            for step in self.execution.steps
            if (
                step.tool_name
                == "analyze_robot_image"
                and step.status == "success"
            )
        }

        returned_vision_step_ids = [
            item.step_id
            for item in self.vision_observations
        ]

        if len(returned_vision_step_ids) != len(
            set(returned_vision_step_ids)
        ):
            raise ValueError(
                "vision_observations不能包含"
                "重复step_id"
            )

        if set(returned_vision_step_ids) != (
            successful_vision_step_ids
        ):
            raise ValueError(
                "vision_observations必须与成功的"
                "analyze_robot_image步骤一一对应"
            )

        # 测试草案中的引用必须同时出现在
        # 最终DiagnosisReport的真实证据列表中。
        #
        # 这样客户端可以直接定位并审核草案依据，
        # 不能只收到无法追溯的Chunk ID。
        allowed_chunk_ids = {
            evidence.chunk_id
            for evidence
            in self.diagnosis.evidence
        }

        draft_chunk_ids = {
            evidence.chunk_id
            for draft
            in self.test_case_drafts
            for evidence
            in draft.evidence
        }

        unknown_draft_chunk_ids = sorted(
            draft_chunk_ids
            - allowed_chunk_ids
        )

        if unknown_draft_chunk_ids:
            raise ValueError(
                "测试草案引用了诊断响应中"
                "不存在的Chunk ID："
                + ", ".join(
                    unknown_draft_chunk_ids
                )
            )

        return self
