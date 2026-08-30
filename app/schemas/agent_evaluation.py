"""Robot Diagnostic Agent评测场景的数据契约。

本模块负责定义：

1. Agent评测场景的稳定编号和类型；
2. 发送给Agent API的诊断请求；
3. 必须调用、允许调用和禁止调用的工具；
4. 可以接受的执行状态和终止原因；
5. 可以接受的诊断状态和Planner结束原因；
6. 任务完成、安全拒答和预期证据之间的关系。
7. 一条返回引用的脱敏评分记录；
8. 一次场景实际执行后的结构化评分结果。
9. 整批Agent评测的汇总指标；
10. 指标与逐场景结果一致的最终评测报告。

本模块不运行Agent、不发送HTTP请求、不计算评测指标，
也不读取data/eval目录中的文件。
"""

# isclose用于比较浮点比率。
#
# 浮点除法可能产生极小的二进制表示误差，
# 因而不直接使用==比较计算结果。
from math import (
    isclose,
)


from typing import (
    Annotated,
    Literal,
    Self,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from app.schemas.agent import (
    ToolName,
)
from app.schemas.agent_api import (
    AgentDiagnosisRequest,
)
from app.schemas.agent_planning import (
    AgentPlannerFinishReason,
)
from app.schemas.agent_runtime import (
    AgentRunTerminationReason,
)
from app.schemas.diagnostics import (
    DiagnosisStatus,
)
from app.schemas.evaluation import (
    ExpectedEvidence,
)


# 场景类型用于分类统计和阅读报告。
#
# 它不直接决定评分结果；
# 真正的评分依据仍然是场景中声明的
# required_tools、预期状态和预期证据。
AgentEvaluationScenarioType = Literal[
    "knowledge_only",
    "knowledge_and_telemetry",
    "knowledge_and_test_draft",
    "multi_tool_diagnosis",
    "insufficient_evidence",
    "tool_failure",
    "prompt_injection",
    "high_risk_request",
]


# Agent执行层当前只有正常完成和安全中止两种公开状态。
#
# 这里单独声明类型别名，是为了让评测场景与
# AgentExecutionSummary.state保持相同含义。
AgentExpectedExecutionState = Literal[
    "completed",
    "aborted",
]


# 标签只用于筛选和分类评测场景。
#
# 例如：
#
# multi_tool
# safety
# telemetry
# prompt_injection
AgentEvaluationTag = Annotated[
    str,
    Field(
        min_length=1,
        max_length=50,
        pattern=r"^[a-z0-9_]+$",
        description=(
            "只包含小写字母、数字和下划线的"
            "Agent评测标签"
        ),
    ),
]


class AgentEvaluationScenario(BaseModel):
    """一条机器可读取的Agent Gold评测场景。

    本模型描述的是“预期行为”，不是Agent实际结果。

    后续评测脚本会：

    1. 读取本模型中的request；
    2. 调用POST /api/v1/agent/diagnose；
    3. 从响应轨迹取得实际工具；
    4. 将实际状态、工具和证据与本模型比较。
    """

    model_config = ConfigDict(
        # 清理普通字符串字段首尾空白。
        str_strip_whitespace=True,

        # 拒绝拼错或未声明的字段。
        #
        # 如果把required_tools误写成required_tool，
        # 不能静默忽略并继续产生错误评测结果。
        extra="forbid",

        # Gold场景创建后不可重新赋值。
        #
        # 这样评测过程不能在运行到一半时
        # 修改预期答案。
        frozen=True,
    )

    scenario_id: str = Field(
        pattern=r"^agent-\d{3}$",
        description=(
            "Agent评测场景稳定编号，"
            "例如agent-001"
        ),
    )

    name: str = Field(
        min_length=1,
        max_length=200,
        description="供人工阅读的场景名称",
    )

    scenario_type: (
        AgentEvaluationScenarioType
    ) = Field(
        description=(
            "场景所属的主要能力或失败类型"
        ),
    )

    # 直接复用公开API请求契约。
    #
    # 这样评测集中的请求如果包含非法字段、
    # 空robot_id或过长日志，会在加载阶段失败，
    # 而不是运行到HTTP请求时才发现。
    request: AgentDiagnosisRequest = Field(
        description=(
            "发送给Agent诊断API的请求"
        ),
    )

    # required_tools表示要完成当前任务，
    # 至少必须实际成功选择过的工具。
    #
    # 评分时不比较调用次数；
    # 重复调用造成的低效率由average_steps反映。
    required_tools: tuple[
        ToolName,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=10,
        description=(
            "本场景至少必须选择的工具集合"
        ),
    )

    # allowed_tools表示该场景允许出现的全部工具。
    #
    # 实际轨迹出现允许列表以外的工具时，
    # 工具选择评分失败。
    allowed_tools: tuple[
        ToolName,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=10,
        description=(
            "本场景允许出现的全部工具集合"
        ),
    )

    # forbidden_tools专门记录安全边界。
    #
    # 例如run_shell虽然符合ToolName字符串格式，
    # 但不属于当前Agent注册表，也不允许执行。
    forbidden_tools: tuple[
        ToolName,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=10,
        description=(
            "本场景明确禁止选择的工具"
        ),
    )

    expected_execution_states: tuple[
        AgentExpectedExecutionState,
        ...,
    ] = Field(
        min_length=1,
        max_length=2,
        description=(
            "本场景可以接受的Agent执行状态"
        ),
    )

    expected_termination_reasons: tuple[
        AgentRunTerminationReason,
        ...,
    ] = Field(
        min_length=1,
        max_length=8,
        description=(
            "本场景可以接受的结构化终止原因"
        ),
    )

    # aborted执行没有Planner正常结束原因，
    # 因而纯aborted场景必须使用空tuple。
    expected_finish_reasons: tuple[
        AgentPlannerFinishReason,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=3,
        description=(
            "completed执行可以接受的"
            "Planner结束原因"
        ),
    )

    expected_diagnosis_statuses: tuple[
        DiagnosisStatus,
        ...,
    ] = Field(
        min_length=1,
        max_length=3,
        description=(
            "本场景可以接受的最终诊断状态"
        ),
    )

    # 继续复用Week 2/3的ExpectedEvidence。
    #
    # 每项证据使用：
    #
    # source_file + page_or_section
    #
    # 定位，不依赖重新摄取后可能变化的Chunk ID。
    expected_evidence: tuple[
        ExpectedEvidence,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=10,
        description=(
            "任务完成时最终诊断必须覆盖的"
            "知识库来源位置"
        ),
    )

    # 这个字段决定任务完成率的分母。
    #
    # 安全拒答和故障中止场景不能被错误地算作
    # “本应完成但没有完成”。
    expects_task_completion: bool = Field(
        description=(
            "当前场景是否要求Agent完成诊断任务"
        ),
    )

    # 这个字段决定安全拒答率的分母。
    #
    # 只有明确要求拒答的场景才参与该指标。
    expects_safe_refusal: bool = Field(
        description=(
            "当前场景是否要求返回安全拒答"
        ),
    )

    tags: tuple[
        AgentEvaluationTag,
        ...,
    ] = Field(
        min_length=1,
        max_length=10,
        description=(
            "用于筛选和分类场景的标签"
        ),
    )

    notes: str = Field(
        min_length=1,
        max_length=2_000,
        description=(
            "场景设计依据和人工审核说明"
        ),
    )

    @field_validator(
        "required_tools",
        "allowed_tools",
        "forbidden_tools",
        "expected_execution_states",
        "expected_termination_reasons",
        "expected_finish_reasons",
        "expected_diagnosis_statuses",
        "tags",
    )
    @classmethod
    def tuple_values_must_be_unique(
        cls,
        values: tuple[object, ...],
        info: ValidationInfo,
    ) -> tuple[object, ...]:
        """集合语义字段不能包含重复值。

        ValidationInfo.field_name表示当前正在校验的字段名，
        因而一个Validator可以复用到多个tuple字段。
        """

        if len(values) != len(set(values)):
            raise ValueError(
                f"{info.field_name}"
                "不能包含重复项"
            )

        return values

    @model_validator(mode="after")
    def validate_scenario_relationships(
        self,
    ) -> Self:
        """校验工具、状态、完成和拒答之间的关系。

        mode="after"表示所有字段已经先完成：

        1. 基础类型转换；
        2. Literal范围校验；
        3. 嵌套Pydantic模型校验。

        因此这里可以安全地进行跨字段比较。
        """

        required_tools = set(
            self.required_tools
        )
        allowed_tools = set(
            self.allowed_tools
        )
        forbidden_tools = set(
            self.forbidden_tools
        )

        # 必需工具必须同时属于允许工具。
        #
        # 否则同一个Gold场景会要求Agent调用一个
        # 又不允许调用的工具。
        if not required_tools.issubset(
            allowed_tools
        ):
            raise ValueError(
                "required_tools必须是"
                "allowed_tools的子集"
            )

        # 同一工具不能既允许又禁止。
        if allowed_tools & forbidden_tools:
            raise ValueError(
                "allowed_tools和"
                "forbidden_tools不能重叠"
            )

        # 预期证据使用来源文件和位置联合去重。
        evidence_locations = [
            (
                item.source_file,
                item.page_or_section,
            )
            for item in self.expected_evidence
        ]

        if len(evidence_locations) != len(
            set(evidence_locations)
        ):
            raise ValueError(
                "expected_evidence不能包含"
                "重复位置"
            )

        # 完成型场景必须同时具备：
        #
        # completed执行
        # completed诊断
        # task_completed结束原因
        # planner_finished终止原因
        # 至少一条预期知识证据
        #
        # 这些条件共同定义当前项目里的
        # “任务真正完成”。
        if self.expects_task_completion:
            if (
                "completed"
                not in self.expected_execution_states
            ):
                raise ValueError(
                    "完成型场景必须允许"
                    "completed执行状态"
                )

            if (
                "completed"
                not in self.expected_diagnosis_statuses
            ):
                raise ValueError(
                    "完成型场景必须允许"
                    "completed诊断状态"
                )

            if (
                "task_completed"
                not in self.expected_finish_reasons
            ):
                raise ValueError(
                    "完成型场景必须包含"
                    "task_completed"
                )

            if (
                "planner_finished"
                not in (
                    self
                    .expected_termination_reasons
                )
            ):
                raise ValueError(
                    "完成型场景必须包含"
                    "planner_finished"
                )

            if not self.expected_evidence:
                raise ValueError(
                    "完成型场景必须包含"
                    "预期证据"
                )

        # 安全拒答不是任务完成。
        #
        # 它必须允许abstained诊断，
        # 并且不能预期最终诊断仍暴露知识证据。
        if self.expects_safe_refusal:
            if self.expects_task_completion:
                raise ValueError(
                    "安全拒答场景不能同时"
                    "要求任务完成"
                )

            if (
                "abstained"
                not in self.expected_diagnosis_statuses
            ):
                raise ValueError(
                    "安全拒答场景必须允许"
                    "abstained诊断"
                )

            if self.expected_evidence:
                raise ValueError(
                    "安全拒答场景不能预期"
                    "最终诊断证据"
                )

        completed_is_allowed = (
            "completed"
            in self.expected_execution_states
        )

        aborted_is_allowed = (
            "aborted"
            in self.expected_execution_states
        )

        # Planner只有在执行状态为completed时
        # 才会提供finish_reason。
        if (
            completed_is_allowed
            and not self.expected_finish_reasons
        ):
            raise ValueError(
                "允许completed执行时必须设置"
                "expected_finish_reasons"
            )

        if (
            not completed_is_allowed
            and self.expected_finish_reasons
        ):
            raise ValueError(
                "不允许completed执行时不能设置"
                "expected_finish_reasons"
            )

        # planner_finished只能对应completed执行。
        if (
            completed_is_allowed
            and "planner_finished"
            not in self.expected_termination_reasons
        ):
            raise ValueError(
                "允许completed执行时必须包含"
                "planner_finished"
            )

        if (
            not completed_is_allowed
            and "planner_finished"
            in self.expected_termination_reasons
        ):
            raise ValueError(
                "不允许completed执行时不能包含"
                "planner_finished"
            )

        # aborted执行必须至少声明一种
        # 非planner_finished终止原因。
        non_finished_reasons = {
            reason
            for reason
            in self.expected_termination_reasons
            if reason != "planner_finished"
        }

        if (
            aborted_is_allowed
            and not non_finished_reasons
        ):
            raise ValueError(
                "允许aborted执行时必须包含"
                "中止终止原因"
            )

        # 如果场景不允许aborted，
        # 就不能把超时、失败等中止原因列为可接受。
        if (
            not aborted_is_allowed
            and non_finished_reasons
        ):
            raise ValueError(
                "不允许aborted执行时不能包含"
                "中止终止原因"
            )

        return self
    

# 一项公开的场景失败原因。
#
# 失败原因由评分代码生成，不保存原始异常、
# API响应正文或模型输出。
AgentEvaluationFailureReason = Annotated[
    str,
    Field(
        min_length=1,
        max_length=500,
        description=(
            "经过脱敏、可以写入评测报告的"
            "场景失败原因"
        ),
    ),
]


class AgentCitationEvaluation(BaseModel):
    """一条Agent返回引用的脱敏评分记录。

    本模型保留：

    1. Chunk和文档标识；
    2. 原始文件名；
    3. 页码或章节；
    4. 是否命中Gold预期位置。

    它不保存完整excerpt，避免批量评测报告
    再次复制原始语料。
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        frozen=True,
    )

    chunk_id: str = Field(
        min_length=1,
        max_length=200,
        description=(
            "Agent响应中真实引用的Chunk ID"
        ),
    )

    document_id: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description=(
            "引用所属文档的SHA-256标识"
        ),
    )

    source_file: str = Field(
        min_length=1,
        max_length=255,
        description="引用所属的原始文件名",
    )

    page_or_section: str = Field(
        min_length=1,
        max_length=500,
        description=(
            "引用在原始文件中的页码或章节"
        ),
    )

    correct: bool = Field(
        description=(
            "引用位置是否命中当前场景的"
            "Gold预期证据"
        ),
    )

    @model_validator(mode="after")
    def chunk_must_belong_to_document(
        self,
    ) -> Self:
        """Chunk ID必须以对应document_id开头。"""

        expected_prefix = (
            self.document_id + ":"
        )

        if not self.chunk_id.startswith(
            expected_prefix
        ):
            raise ValueError(
                "chunk_id必须属于document_id"
            )

        return self


class AgentScenarioEvaluation(BaseModel):
    """一次Agent评测场景的实际结果。

    本模型连接两类数据：

    1. API实际返回的执行状态、工具和引用；
    2. 评分函数根据Gold场景产生的布尔结果。

    它不负责调用API，也不自行判断哪些工具正确。
    这些操作由后续评分函数完成。
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        frozen=True,
    )

    scenario_id: str = Field(
        pattern=r"^agent-\d{3}$",
        description=(
            "对应AgentEvaluationScenario的编号"
        ),
    )

    scenario_type: (
        AgentEvaluationScenarioType
    ) = Field(
        description="被执行场景的分类",
    )

    # True表示：
    #
    # 1. HTTP请求成功；
    # 2. 响应可以解析成AgentDiagnosisResponse。
    #
    # HTTP 200但响应结构非法仍应记为False。
    request_succeeded: bool = Field(
        description=(
            "是否取得了可以进行Agent评分的"
            "合法API响应"
        ),
    )

    # 网络连接失败时没有HTTP状态码，
    # 因此允许使用None。
    http_status_code: int | None = Field(
        default=None,
        ge=100,
        le=599,
        description=(
            "API返回的HTTP状态码；"
            "传输失败时为空"
        ),
    )

    # API错误响应通常仍包含request_id，
    # 所以请求失败时该字段也允许保留。
    request_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
        description=(
            "用于关联API响应头和服务端日志的"
            "请求标识"
        ),
    )

    request_error: str | None = Field(
        default=None,
        min_length=1,
        max_length=500,
        description=(
            "请求失败时经过脱敏的公开错误说明"
        ),
    )

    # 这里保留实际工具顺序，并且允许重复。
    #
    # required_tools等Gold字段使用集合语义；
    # actual_tool_sequence必须保留重复调用，
    # 才能计算真实step_count和平均步骤数。
    actual_tool_sequence: tuple[
        ToolName,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=20,
        description=(
            "按照公开执行轨迹排列的实际工具顺序"
        ),
    )

    actual_execution_state: (
        AgentExpectedExecutionState | None
    ) = Field(
        default=None,
        description=(
            "API响应中的Agent执行状态"
        ),
    )

    actual_termination_reason: (
        AgentRunTerminationReason | None
    ) = Field(
        default=None,
        description=(
            "API响应中的结构化终止原因"
        ),
    )

    actual_finish_reason: (
        AgentPlannerFinishReason | None
    ) = Field(
        default=None,
        description=(
            "Planner正常结束时的finish_reason"
        ),
    )

    actual_diagnosis_status: (
        DiagnosisStatus | None
    ) = Field(
        default=None,
        description=(
            "最终DiagnosisReport的状态"
        ),
    )

    step_count: int = Field(
        ge=0,
        le=20,
        description=(
            "本次实际完成的工具执行步骤数"
        ),
    )

    citation_evaluations: tuple[
        AgentCitationEvaluation,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=10,
        description=(
            "Agent最终诊断返回的脱敏引用评分"
        ),
    )

    expected_evidence: tuple[
        ExpectedEvidence,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=10,
        description=(
            "从Gold场景复制的全部预期证据位置"
        ),
    )

    matched_expected_evidence: tuple[
        ExpectedEvidence,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=10,
        description=(
            "被正确引用实际覆盖的预期证据位置"
        ),
    )

    tool_selection_correct: bool = Field(
        description=(
            "必需工具是否出现且实际工具"
            "是否全部处于允许范围"
        ),
    )

    # None表示该场景不参与任务完成率分母。
    task_completion_correct: bool | None = Field(
        default=None,
        description=(
            "要求任务完成的场景是否实际完成；"
            "不参与该指标时为空"
        ),
    )

    # None表示该场景不参与安全拒答率分母。
    safe_refusal_correct: bool | None = Field(
        default=None,
        description=(
            "要求安全拒答的场景是否实际拒答；"
            "不参与该指标时为空"
        ),
    )

    passed: bool = Field(
        description=(
            "当前场景是否满足全部适用的"
            "评测要求"
        ),
    )

    failure_reasons: tuple[
        AgentEvaluationFailureReason,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=20,
        description=(
            "当前场景未通过的脱敏原因"
        ),
    )

    @field_validator(
        "failure_reasons"
    )
    @classmethod
    def failure_reasons_must_be_unique(
        cls,
        values: tuple[str, ...],
    ) -> tuple[str, ...]:
        """同一失败原因不能重复记录。"""

        if len(values) != len(set(values)):
            raise ValueError(
                "failure_reasons不能包含重复项"
            )

        return values

    @model_validator(mode="after")
    def validate_result_relationships(
        self,
    ) -> Self:
        """校验请求、响应、引用和评分之间的关系。"""

        # 每个公开轨迹事件代表一次实际工具执行，
        # 因而步骤数必须等于工具顺序长度。
        if self.step_count != len(
            self.actual_tool_sequence
        ):
            raise ValueError(
                "step_count必须等于"
                "actual_tool_sequence长度"
            )

        # 不能重复统计同一个Chunk。
        chunk_ids = [
            citation.chunk_id
            for citation
            in self.citation_evaluations
        ]

        if len(chunk_ids) != len(
            set(chunk_ids)
        ):
            raise ValueError(
                "citation_evaluations中的"
                "chunk_id不能重复"
            )

        expected_locations = [
            (
                item.source_file,
                item.page_or_section,
            )
            for item in self.expected_evidence
        ]

        matched_locations = [
            (
                item.source_file,
                item.page_or_section,
            )
            for item
            in self.matched_expected_evidence
        ]

        if len(expected_locations) != len(
            set(expected_locations)
        ):
            raise ValueError(
                "expected_evidence不能包含重复位置"
            )

        if len(matched_locations) != len(
            set(matched_locations)
        ):
            raise ValueError(
                "matched_expected_evidence"
                "不能包含重复位置"
            )

        expected_location_set = set(
            expected_locations
        )
        matched_location_set = set(
            matched_locations
        )

        if not matched_location_set.issubset(
            expected_location_set
        ):
            raise ValueError(
                "matched_expected_evidence必须是"
                "expected_evidence的子集"
            )

        # API或传输失败时，不存在可信的Agent响应。
        #
        # 此时不能构造工具轨迹、状态、引用或命中证据，
        # 否则报告会把没有发生的执行伪造成事实。
        if not self.request_succeeded:
            if self.request_error is None:
                raise ValueError(
                    "失败请求必须包含request_error"
                )

            if (
                self.actual_tool_sequence
                or self.step_count != 0
            ):
                raise ValueError(
                    "失败请求不能包含"
                    "Agent工具观察"
                )

            if (
                self.actual_execution_state
                is not None
                or self.actual_termination_reason
                is not None
                or self.actual_finish_reason
                is not None
                or self.actual_diagnosis_status
                is not None
            ):
                raise ValueError(
                    "失败请求不能包含"
                    "Agent响应状态"
                )

            if self.citation_evaluations:
                raise ValueError(
                    "失败请求不能包含引用评分"
                )

            if self.matched_expected_evidence:
                raise ValueError(
                    "失败请求不能包含"
                    "已命中预期证据"
                )

        else:
            # 取得合法Agent响应时HTTP必须为2xx。
            if (
                self.http_status_code is None
                or not (
                    200
                    <= self.http_status_code
                    < 300
                )
            ):
                raise ValueError(
                    "成功请求必须使用"
                    "2xx HTTP状态码"
                )

            if self.request_id is None:
                raise ValueError(
                    "成功请求必须包含request_id"
                )

            if self.request_error is not None:
                raise ValueError(
                    "成功请求不能包含request_error"
                )

            if (
                self.actual_execution_state
                is None
            ):
                raise ValueError(
                    "成功请求必须包含"
                    "actual_execution_state"
                )

            if (
                self.actual_termination_reason
                is None
            ):
                raise ValueError(
                    "成功请求必须包含"
                    "actual_termination_reason"
                )

            if (
                self.actual_diagnosis_status
                is None
            ):
                raise ValueError(
                    "成功请求必须包含"
                    "actual_diagnosis_status"
                )

            if (
                self.actual_execution_state
                == "completed"
            ):
                if (
                    self.actual_finish_reason
                    is None
                ):
                    raise ValueError(
                        "completed执行必须包含"
                        "actual_finish_reason"
                    )

                if (
                    self.actual_termination_reason
                    != "planner_finished"
                ):
                    raise ValueError(
                        "completed执行必须使用"
                        "planner_finished"
                    )

            else:
                # 公开执行状态只有completed和aborted，
                # 所以前面未进入时必然是aborted。
                if (
                    self.actual_finish_reason
                    is not None
                ):
                    raise ValueError(
                        "aborted执行不能包含"
                        "actual_finish_reason"
                    )

                if (
                    self.actual_termination_reason
                    == "planner_finished"
                ):
                    raise ValueError(
                        "aborted执行不能使用"
                        "planner_finished"
                    )

        # correct字段不能由评分函数任意填写。
        #
        # 一条引用的位置在Gold预期集合中时，
        # correct必须为True；否则必须为False。
        correct_citation_locations: set[
            tuple[str, str]
        ] = set()

        for citation in (
            self.citation_evaluations
        ):
            location = (
                citation.source_file,
                citation.page_or_section,
            )

            location_is_expected = (
                location
                in expected_location_set
            )

            if (
                citation.correct
                != location_is_expected
            ):
                raise ValueError(
                    "citation.correct必须与"
                    "预期证据位置一致"
                )

            if citation.correct:
                correct_citation_locations.add(
                    location
                )

        # matched_expected_evidence应当是
        # 正确引用命中位置的去重结果。
        if (
            matched_location_set
            != correct_citation_locations
        ):
            raise ValueError(
                "matched_expected_evidence必须等于"
                "正确引用命中的预期位置"
            )

        # 任务完成评分为True时，
        # 实际响应必须同时满足全部完成条件。
        if self.task_completion_correct is True:
            actual_task_completed = (
                self.request_succeeded
                and (
                    self.actual_execution_state
                    == "completed"
                )
                and (
                    self.actual_termination_reason
                    == "planner_finished"
                )
                and (
                    self.actual_finish_reason
                    == "task_completed"
                )
                and (
                    self.actual_diagnosis_status
                    == "completed"
                )
            )

            if not actual_task_completed:
                raise ValueError(
                    "task_completion_correct为True时"
                    "实际结果必须完整完成"
                )

        # 安全拒答评分为True时，
        # 必须确实取得合法的abstained诊断。
        if self.safe_refusal_correct is True:
            if (
                not self.request_succeeded
                or (
                    self.actual_diagnosis_status
                    != "abstained"
                )
            ):
                raise ValueError(
                    "safe_refusal_correct为True时"
                    "诊断必须abstained"
                )

        # 总通过标记与失败原因必须互相对应。
        if self.passed:
            if self.failure_reasons:
                raise ValueError(
                    "passed结果不能包含"
                    "failure_reasons"
                )
        elif not self.failure_reasons:
            raise ValueError(
                "未通过结果必须包含"
                "failure_reasons"
            )

        return self
        
        
def _expected_ratio(
    numerator: int,
    denominator: int,
) -> float:
    """计算Validator期望看到的比率。

    这个函数只用于校验已经生成的Metrics，
    不代替后续正式的指标计算函数。

    分母为0表示没有适用样本，
    当前项目统一使用0.0，避免NaN和除零。
    """

    if denominator == 0:
        return 0.0

    return numerator / denominator


class AgentEvaluationMetrics(BaseModel):
    """整批Agent场景的汇总指标。

    本模型包含任务6要求的五类核心指标：

    1. 工具选择正确率；
    2. 任务完成率；
    3. 引用正确率；
    4. 平均工具步骤数；
    5. 安全拒答率。

    另外保留请求成功率、场景总通过率和
    引用覆盖率，帮助定位核心指标变化的原因。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,

        # 拒绝NaN和正负无穷大。
        allow_inf_nan=False,
    )

    scenario_count: int = Field(
        ge=1,
        description="参与本次评测的场景总数",
    )

    request_success_count: int = Field(
        ge=0,
        description=(
            "成功取得合法Agent API响应的场景数"
        ),
    )

    request_success_rate: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "request_success_count / scenario_count"
        ),
    )

    passed_scenario_count: int = Field(
        ge=0,
        description=(
            "满足全部适用要求的场景数"
        ),
    )

    scenario_pass_rate: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "passed_scenario_count / scenario_count"
        ),
    )

    tool_selection_correct_count: int = Field(
        ge=0,
        description=(
            "工具选择满足required、allowed和"
            "forbidden约束的场景数"
        ),
    )

    tool_selection_accuracy: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "tool_selection_correct_count / "
            "scenario_count"
        ),
    )

    # 只有expects_task_completion=True的场景
    # 进入这个分母。
    expected_completion_count: int = Field(
        ge=0,
        description=(
            "要求Agent完整完成任务的场景数"
        ),
    )

    completed_task_count: int = Field(
        ge=0,
        description=(
            "本应完成并且实际完整完成的场景数"
        ),
    )

    task_completion_rate: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "completed_task_count / "
            "expected_completion_count"
        ),
    )

    returned_citation_count: int = Field(
        ge=0,
        description=(
            "所有场景最终诊断返回的引用总数"
        ),
    )

    correct_citation_count: int = Field(
        ge=0,
        description=(
            "来源位置命中Gold预期的引用总数"
        ),
    )

    citation_correctness: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "correct_citation_count / "
            "returned_citation_count"
        ),
    )

    expected_evidence_count: int = Field(
        ge=0,
        description=(
            "全部场景声明的Gold预期证据总数"
        ),
    )

    matched_expected_evidence_count: int = Field(
        ge=0,
        description=(
            "被正确引用覆盖的Gold证据总数"
        ),
    )

    citation_coverage: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "matched_expected_evidence_count / "
            "expected_evidence_count"
        ),
    )

    # 只有expects_safe_refusal=True的场景
    # 进入安全拒答分母。
    safe_refusal_case_count: int = Field(
        ge=0,
        description=(
            "明确要求Agent安全拒答的场景数"
        ),
    )

    correct_safe_refusal_count: int = Field(
        ge=0,
        description=(
            "实际返回abstained安全拒答的场景数"
        ),
    )

    safe_refusal_rate: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "correct_safe_refusal_count / "
            "safe_refusal_case_count"
        ),
    )

    total_step_count: int = Field(
        ge=0,
        description=(
            "全部合法Agent响应中的工具步骤总数"
        ),
    )

    average_steps: float = Field(
        ge=0.0,
        le=20.0,
        description=(
            "total_step_count / scenario_count"
        ),
    )

    @model_validator(mode="after")
    def counts_and_rates_must_match(
        self,
    ) -> Self:
        """校验所有计数关系和派生比率。"""

        if (
            self.request_success_count
            > self.scenario_count
        ):
            raise ValueError(
                "request_success_count不能超过"
                "scenario_count"
            )

        # 没有取得合法响应的场景不能通过。
        if (
            self.passed_scenario_count
            > self.request_success_count
        ):
            raise ValueError(
                "passed_scenario_count不能超过"
                "request_success_count"
            )

        if (
            self.tool_selection_correct_count
            > self.scenario_count
        ):
            raise ValueError(
                "tool_selection_correct_count"
                "不能超过scenario_count"
            )

        if (
            self.tool_selection_correct_count
            > self.request_success_count
        ):
            raise ValueError(
                "tool_selection_correct_count"
                "不能超过request_success_count"
            )

        if (
            self.completed_task_count
            > self.expected_completion_count
        ):
            raise ValueError(
                "completed_task_count不能超过"
                "expected_completion_count"
            )

        if (
            self.correct_citation_count
            > self.returned_citation_count
        ):
            raise ValueError(
                "correct_citation_count不能超过"
                "returned_citation_count"
            )

        if (
            self.matched_expected_evidence_count
            > self.expected_evidence_count
        ):
            raise ValueError(
                "matched_expected_evidence_count"
                "不能超过expected_evidence_count"
            )

        if (
            self.correct_safe_refusal_count
            > self.safe_refusal_case_count
        ):
            raise ValueError(
                "correct_safe_refusal_count"
                "不能超过safe_refusal_case_count"
            )

        # 每次合法响应最多包含20个公开轨迹步骤。
        if (
            self.total_step_count
            > self.request_success_count * 20
        ):
            raise ValueError(
                "total_step_count不能超过"
                "成功请求的最大步骤数"
            )

        # 字段名、实际值、根据计数计算的期望值。
        expected_rates = {
            "request_success_rate": (
                self.request_success_rate,
                _expected_ratio(
                    self.request_success_count,
                    self.scenario_count,
                ),
            ),
            "scenario_pass_rate": (
                self.scenario_pass_rate,
                _expected_ratio(
                    self.passed_scenario_count,
                    self.scenario_count,
                ),
            ),
            "tool_selection_accuracy": (
                self.tool_selection_accuracy,
                _expected_ratio(
                    (
                        self
                        .tool_selection_correct_count
                    ),
                    self.scenario_count,
                ),
            ),
            "task_completion_rate": (
                self.task_completion_rate,
                _expected_ratio(
                    self.completed_task_count,
                    self.expected_completion_count,
                ),
            ),
            "citation_correctness": (
                self.citation_correctness,
                _expected_ratio(
                    self.correct_citation_count,
                    self.returned_citation_count,
                ),
            ),
            "citation_coverage": (
                self.citation_coverage,
                _expected_ratio(
                    (
                        self
                        .matched_expected_evidence_count
                    ),
                    self.expected_evidence_count,
                ),
            ),
            "safe_refusal_rate": (
                self.safe_refusal_rate,
                _expected_ratio(
                    self.correct_safe_refusal_count,
                    self.safe_refusal_case_count,
                ),
            ),
            "average_steps": (
                self.average_steps,
                _expected_ratio(
                    self.total_step_count,
                    self.scenario_count,
                ),
            ),
        }

        for (
            field_name,
            (
                actual_value,
                expected_value,
            ),
        ) in expected_rates.items():
            # abs_tol处理接近0的结果；
            # rel_tol处理一般比例值。
            if not isclose(
                actual_value,
                expected_value,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    f"{field_name}与计数不一致"
                )

        return self


class AgentEvaluationReport(BaseModel):
    """一次完整Agent批量评测的最终报告。

    Report既保存运行配置，也保存：

    1. 汇总Metrics；
    2. 每个场景的脱敏结果。

    本模型故意不包含generated_at或其他生成时间，
    保证报告内容不会因为运行时刻变化而产生无意义差异。
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        frozen=True,
    )

    evaluation_version: str = Field(
        min_length=1,
        max_length=100,
        description=(
            "Agent评测逻辑和指标口径版本"
        ),
        examples=["agent-evaluation-v1"],
    )

    scenario_source: str = Field(
        min_length=1,
        max_length=500,
        description=(
            "本次评测使用的场景JSONL路径"
        ),
    )

    api_url: str = Field(
        min_length=1,
        max_length=500,
        pattern=r"^https?://",
        description=(
            "被评测的Agent诊断API地址"
        ),
    )

    # 记录Planner配置是为了保证不同报告可比较。
    planner_model: str = Field(
        min_length=1,
        max_length=200,
        description=(
            "本次评测使用的生成式Planner模型"
        ),
    )

    planner_prompt_version: str = Field(
        min_length=1,
        max_length=100,
        description=(
            "本次评测使用的Planner Prompt版本"
        ),
    )

    embedding_model: str = Field(
        min_length=1,
        max_length=200,
        description=(
            "search_knowledge使用的Embedding模型"
        ),
    )

    collection_name: str = Field(
        min_length=1,
        max_length=200,
        description=(
            "search_knowledge使用的Chroma Collection"
        ),
    )

    metrics: AgentEvaluationMetrics = Field(
        description=(
            "整批场景的汇总指标"
        ),
    )

    results: tuple[
        AgentScenarioEvaluation,
        ...,
    ] = Field(
        min_length=1,
        description=(
            "按评测输入顺序保存的逐场景结果"
        ),
    )

    @model_validator(mode="after")
    def metrics_must_match_results(
        self,
    ) -> Self:
        """从results重新计算所有汇总计数。

        AgentEvaluationMetrics已经校验计数与比率；
        Report再验证这些计数是否真的来自results。
        """

        scenario_ids = [
            result.scenario_id
            for result in self.results
        ]

        if len(scenario_ids) != len(
            set(scenario_ids)
        ):
            raise ValueError(
                "results中的scenario_id不能重复"
            )

        calculated_counts = {
            "scenario_count": len(
                self.results
            ),
            "request_success_count": sum(
                result.request_succeeded
                for result in self.results
            ),
            "passed_scenario_count": sum(
                result.passed
                for result in self.results
            ),
            "tool_selection_correct_count": sum(
                result.tool_selection_correct
                for result in self.results
            ),
            "expected_completion_count": sum(
                (
                    result.task_completion_correct
                    is not None
                )
                for result in self.results
            ),
            "completed_task_count": sum(
                (
                    result.task_completion_correct
                    is True
                )
                for result in self.results
            ),
            "returned_citation_count": sum(
                len(
                    result.citation_evaluations
                )
                for result in self.results
            ),
            "correct_citation_count": sum(
                citation.correct
                for result in self.results
                for citation
                in result.citation_evaluations
            ),
            "expected_evidence_count": sum(
                len(result.expected_evidence)
                for result in self.results
            ),
            (
                "matched_expected_evidence_count"
            ): sum(
                len(
                    result
                    .matched_expected_evidence
                )
                for result in self.results
            ),
            "safe_refusal_case_count": sum(
                (
                    result.safe_refusal_correct
                    is not None
                )
                for result in self.results
            ),
            "correct_safe_refusal_count": sum(
                (
                    result.safe_refusal_correct
                    is True
                )
                for result in self.results
            ),
            "total_step_count": sum(
                result.step_count
                for result in self.results
            ),
        }

        for (
            field_name,
            calculated_value,
        ) in calculated_counts.items():
            metric_value = getattr(
                self.metrics,
                field_name,
            )

            if metric_value != calculated_value:
                raise ValueError(
                    "metrics与results汇总不一致: "
                    f"{field_name}"
                )

        return self