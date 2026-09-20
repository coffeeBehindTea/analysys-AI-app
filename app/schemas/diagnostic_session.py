"""诊断会话、脱敏轨迹和查询结果的数据契约。

本模块只定义可以进入会话存储的数据形状，
不生成session_id、不执行Agent、不读写文件，
也不提供HTTP接口。

持久化记录明确排除：

1. 完整Base64图片；
2. 完整原始日志；
3. API Key或其他密钥；
4. Planner原始消息和模型私有思维链；
5. 未经过公开响应契约校验的工具参数或输出。
"""

from datetime import (
    datetime,
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
    field_validator,
    model_validator,
)

from app.schemas.agent import (
    ToolName,
)
from app.schemas.agent_api import (
    AgentToolTraceEvent,
    AgentVisionObservation,
)
from app.schemas.agent_planning import (
    AgentPlannerFinishReason,
)
from app.schemas.agent_progress import (
    AgentProgressEvidenceId,
    AgentProgressSourceId,
)
from app.schemas.agent_runtime import (
    AgentRunTerminationReason,
)
from app.schemas.agent_tools import (
    DraftTestCaseToolOutput,
    RobotTelemetryToolOutput,
)
from app.schemas.diagnostics import (
    DiagnosisCause,
    DiagnosisCheck,
    DiagnosisEvidence,
    DiagnosisRiskLevel,
    DiagnosisStatus,
    MissingInformationItem,
)


# session_id由服务端使用uuid.uuid4()生成。
#
# 正则同时限制：
#
# 1. 必须使用小写UUID文本；
# 2. 必须是UUID版本4；
# 3. variant位必须符合RFC 4122；
# 4. 客户端不能使用任意字符串冒充会话ID。
DiagnosticSessionId = Annotated[
    str,
    Field(
        pattern=(
            r"^[0-9a-f]{8}-"
            r"[0-9a-f]{4}-"
            r"4[0-9a-f]{3}-"
            r"[89ab][0-9a-f]{3}-"
            r"[0-9a-f]{12}$"
        ),
        description=(
            "由服务端生成的小写UUID4诊断会话标识"
        ),
    ),
]


# 日志正文不会进入会话记录。
#
# 会话只保存日志的SHA-256摘要，
# 用于判断两次请求是否使用了相同日志，
# 但不能通过这个字段直接还原日志正文。
DiagnosticSessionSha256 = Annotated[
    str,
    Field(
        pattern=r"^[0-9a-f]{64}$",
        description="内容的SHA-256十六进制摘要",
    ),
]


DiagnosticSessionStatus = Literal[
    "completed",
    "partial",
    "abstained",
    "human_review_required",
    "failed",
]


class DiagnosticSessionRequestSummary(BaseModel):
    """一次诊断请求可以持久化的脱敏摘要。

    本模型不保存完整log_excerpt，也不保存图片载荷。

    symptom_summary和task_goal_summary必须由后续Builder
    进行清理和长度限制，不能直接复制无限长度的原始输入。
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        frozen=True,
    )

    robot_id: str = Field(
        min_length=1,
        max_length=100,
        description="请求中声明的机器人编号",
    )

    symptom_summary: str = Field(
        min_length=1,
        max_length=500,
        description=(
            "经过长度限制的故障现象摘要"
        ),
    )

    task_goal_summary: str | None = Field(
        default=None,
        min_length=1,
        max_length=500,
        description=(
            "经过长度限制的可选任务目标摘要"
        ),
    )

    log_excerpt_sha256: (
        DiagnosticSessionSha256
    ) = Field(
        description=(
            "原始脱敏日志摘要的SHA-256，"
            "不保存日志正文"
        ),
    )

    log_excerpt_char_count: int = Field(
        ge=1,
        le=10_000,
        description=(
            "原始脱敏日志摘要的字符数量"
        ),
    )

    image_count: int = Field(
        ge=0,
        le=3,
        description=(
            "请求携带的图片数量；"
            "会话记录不保存图片Base64"
        ),
    )


class DiagnosticSessionDiagnosisSnapshot(
    BaseModel
):
    """从最终诊断报告提取的安全持久化快照。

    DiagnosisReport中的symptoms可能包含用户提交的
    log_excerpt，因此不能直接把整个DiagnosisReport
    写入长期会话存储。

    本模型只保留诊断结论、引用和缺失信息。
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    prompt_version: str = Field(
        min_length=1,
        max_length=100,
        description="诊断Prompt版本",
    )

    gate_version: str = Field(
        min_length=1,
        max_length=100,
        description="证据门控版本",
    )

    status: DiagnosisStatus = Field(
        description=(
            "诊断报告的completed、partial、"
            "abstained或human_review_required状态"
        ),
    )

    evidence: tuple[
        DiagnosisEvidence,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=10,
        description=(
            "最终诊断实际公开的知识库引用"
        ),
    )

    possible_causes: tuple[
        DiagnosisCause,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=10,
        description=(
            "经过证据白名单校验的可能原因"
        ),
    )

    next_checks: tuple[
        DiagnosisCheck,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=10,
        description=(
            "经过证据和安全规则校验的检查项"
        ),
    )

    risk_level: DiagnosisRiskLevel = Field(
        description="最终诊断风险等级",
    )

    missing_information: tuple[
        MissingInformationItem,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=20,
        description=(
            "仍然缺少或需要人工确认的信息"
        ),
    )

    abstained: bool = Field(
        description="诊断是否拒绝形成原因和检查项",
    )

    @model_validator(mode="after")
    def status_must_match_content(
        self,
    ) -> Self:
        """保持诊断状态与内容一致。"""

        if (
            self.status == "abstained"
        ) != self.abstained:
            raise ValueError(
                "status必须与abstained一致"
            )

        if self.abstained:
            if (
                self.possible_causes
                or self.next_checks
            ):
                raise ValueError(
                    "拒答快照不能包含原因或检查项"
                )

            if not self.missing_information:
                raise ValueError(
                    "拒答快照必须说明缺失信息"
                )

            return self

        # human_review_required是独立的安全业务终态，
        # 不是证据不足拒答，也不表示已经形成诊断结论。
        # 请求级安全策略可能在Planner和工具运行前直接转人工，
        # 因此不能强制它携带知识证据、原因或检查项。
        if self.status == "human_review_required":
            if not self.missing_information:
                raise ValueError(
                    "human_review_required快照必须说明"
                    "需要人工审核的原因"
                )

            return self

        # completed与partial都声称形成了至少一部分诊断结论，
        # 所以必须具有引用证据和被证据支持的原因或检查项。
        if not self.evidence:
            raise ValueError(
                "非拒答快照必须包含引用证据"
            )

        if not (
            self.possible_causes
            or self.next_checks
        ):
            raise ValueError(
                "非拒答快照至少需要原因或检查项"
            )

        if (
            self.status == "partial"
            and not self.missing_information
        ):
            raise ValueError(
                "partial快照必须说明缺失信息"
            )

        if (
            self.status == "completed"
            and self.missing_information
        ):
            raise ValueError(
                "completed快照不能包含缺失信息"
            )

        return self


class DiagnosticSessionEvidenceCoverage(
    BaseModel
):
    """会话结束时的证据覆盖快照。

    confirmed_source_ids和covered_evidence_ids来自
    AgentProgress，而不是由Planner自行填写。

    cited_chunk_ids来自最终DiagnosisEvidence。
    这些ID能够说明最终报告实际使用了哪些知识证据。
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        frozen=True,
    )

    confirmed_source_ids: tuple[
        AgentProgressSourceId,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=100,
        description=(
            "工具执行过程中确认的全部来源标识"
        ),
    )

    covered_evidence_ids: tuple[
        AgentProgressEvidenceId,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=100,
        description=(
            "工具执行过程中覆盖的全部证据标识"
        ),
    )

    cited_chunk_ids: tuple[
        str,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=10,
        description=(
            "最终诊断报告实际引用的Chunk ID"
        ),
    )

    @field_validator(
        "confirmed_source_ids",
        "covered_evidence_ids",
        "cited_chunk_ids",
    )
    @classmethod
    def identifiers_must_be_unique(
        cls,
        values: tuple[str, ...],
    ) -> tuple[str, ...]:
        """同一种标识不能重复保存。"""

        if len(values) != len(set(values)):
            raise ValueError(
                "会话证据标识不能重复"
            )

        return values


class DiagnosticSessionMetrics(BaseModel):
    """一次诊断会话的可观测性统计。

    这些指标描述运行过程，不代表模型准确率。

    failed_tool_count不包括empty：
    empty表示工具正常执行但没有数据；
    rejected、timeout和error才计入失败工具。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    total_duration_ms: float = Field(
        ge=0.0,
        description=(
            "从诊断Service开始到公开诊断响应构造完成的"
            "管线耗时；不包含会话JSON自身的写盘耗时"
        ),
    )

    tool_step_count: int = Field(
        ge=0,
        le=20,
        description="实际完成的工具步骤数",
    )

    failed_tool_count: int = Field(
        ge=0,
        le=20,
        description=(
            "状态为rejected、timeout或error的工具数量"
        ),
    )

    failed_tool_names: tuple[
        ToolName,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=5,
        description=(
            "至少失败一次的工具名称，"
            "按照首次失败顺序排列"
        ),
    )

    confirmed_source_count: int = Field(
        ge=0,
        le=100,
        description=(
            "AgentProgress确认的来源数量"
        ),
    )

    covered_evidence_count: int = Field(
        ge=0,
        le=100,
        description=(
            "AgentProgress覆盖的证据数量"
        ),
    )

    citation_count: int = Field(
        ge=0,
        le=10,
        description=(
            "最终诊断实际公开的知识引用数量"
        ),
    )

    @field_validator(
        "failed_tool_names"
    )
    @classmethod
    def failed_tool_names_must_be_unique(
        cls,
        values: tuple[
            ToolName,
            ...,
        ],
    ) -> tuple[ToolName, ...]:
        """工具名称只记录第一次出现的位置。"""

        if len(values) != len(set(values)):
            raise ValueError(
                "failed_tool_names不能重复"
            )

        return values

    @model_validator(mode="after")
    def counts_must_be_possible(
        self,
    ) -> Self:
        """失败工具次数不能超过全部工具步骤。"""

        if (
            self.failed_tool_count
            > self.tool_step_count
        ):
            raise ValueError(
                "failed_tool_count不能超过"
                "tool_step_count"
            )

        if (
            self.failed_tool_count == 0
            and self.failed_tool_names
        ):
            raise ValueError(
                "没有失败步骤时不能包含"
                "failed_tool_names"
            )

        if (
            self.failed_tool_count > 0
            and not self.failed_tool_names
        ):
            raise ValueError(
                "存在失败步骤时必须记录"
                "failed_tool_names"
            )

        return self


class DiagnosticSessionRecord(BaseModel):
    """可以写入会话存储的完整脱敏记录。"""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    schema_version: Literal[
        "diagnostic-session-v1"
    ] = Field(
        default="diagnostic-session-v1",
        description="持久化会话契约版本",
    )

    session_id: DiagnosticSessionId = Field(
        description=(
            "独立于HTTP request_id的诊断会话ID"
        ),
    )

    request_id: str = Field(
        min_length=1,
        max_length=200,
        description=(
            "用于关联HTTP响应头和服务日志的请求ID"
        ),
    )

    # 这里的created_at是业务会话创建时间，
    # 用于查询最近会话。
    #
    # 它不是评测报告的生成时间，
    # 因而不违反报告不得写入生成时间的约束。
    created_at: datetime = Field(
        description=(
            "包含明确时区的会话创建时间"
        ),
    )

    request_summary: (
        DiagnosticSessionRequestSummary
    ) = Field(
        description="经过脱敏的请求摘要",
    )

    status: DiagnosticSessionStatus = Field(
        description=(
            "会话最终业务状态，"
            "包含人工审核和运行失败"
        ),
    )

    execution_state: Literal[
        "completed",
        "aborted",
    ] = Field(
        description=(
            "AgentRunner正常结束还是安全中止"
        ),
    )

    termination_reason: (
        AgentRunTerminationReason
    ) = Field(
        description="Python控制层确定的终止原因",
    )

    finish_reason: (
        AgentPlannerFinishReason | None
    ) = Field(
        default=None,
        description=(
            "正常结束时的结构化业务原因"
        ),
    )

    diagnosis: (
        DiagnosticSessionDiagnosisSnapshot
    ) = Field(
        description=(
            "不包含用户原始日志的诊断快照"
        ),
    )

    tool_events: tuple[
        AgentToolTraceEvent,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=20,
        description=(
            "已经公开脱敏的工具执行轨迹"
        ),
    )

    vision_observations: tuple[
        AgentVisionObservation,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=20,
        description=(
            "不包含图片Base64的结构化视觉观察"
        ),
    )

    telemetry_observations: tuple[
        RobotTelemetryToolOutput,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=10,
        description=(
            "带有simulated_memory来源标记的遥测"
        ),
    )

    test_case_drafts: tuple[
        DraftTestCaseToolOutput,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=5,
        description=(
            "明确标记为draft_only的测试草案"
        ),
    )

    evidence_coverage: (
        DiagnosticSessionEvidenceCoverage
    ) = Field(
        description="会话结束时的证据覆盖情况",
    )

    metrics: DiagnosticSessionMetrics = Field(
        description="会话运行统计",
    )

    @field_validator("created_at")
    @classmethod
    def created_at_must_have_timezone(
        cls,
        value: datetime,
    ) -> datetime:
        """最近会话排序不能依赖无时区时间。"""

        if (
            value.tzinfo is None
            or value.utcoffset() is None
        ):
            raise ValueError(
                "created_at必须包含明确时区"
            )

        return value

    @model_validator(mode="after")
    def record_fields_must_match(
        self,
    ) -> Self:
        """防止会话摘要与真实轨迹互相矛盾。"""

        if (
            self.metrics.tool_step_count
            != len(self.tool_events)
        ):
            raise ValueError(
                "tool_step_count必须等于"
                "tool_events数量"
            )

        failed_events = tuple(
            event
            for event in self.tool_events
            if event.status in {
                "rejected",
                "timeout",
                "error",
            }
        )

        if (
            self.metrics.failed_tool_count
            != len(failed_events)
        ):
            raise ValueError(
                "failed_tool_count必须等于"
                "失败工具事件数量"
            )

        expected_failed_tool_names = tuple(
            dict.fromkeys(
                event.tool_name
                for event in failed_events
            )
        )

        if (
            self.metrics.failed_tool_names
            != expected_failed_tool_names
        ):
            raise ValueError(
                "failed_tool_names必须与"
                "失败工具事件一致"
            )

        coverage = self.evidence_coverage

        if (
            self.metrics.confirmed_source_count
            != len(
                coverage.confirmed_source_ids
            )
        ):
            raise ValueError(
                "confirmed_source_count必须与"
                "来源标识数量一致"
            )

        if (
            self.metrics.covered_evidence_count
            != len(
                coverage.covered_evidence_ids
            )
        ):
            raise ValueError(
                "covered_evidence_count必须与"
                "证据标识数量一致"
            )

        diagnosis_chunk_ids = tuple(
            evidence.chunk_id
            for evidence
            in self.diagnosis.evidence
        )

        if (
            coverage.cited_chunk_ids
            != diagnosis_chunk_ids
        ):
            raise ValueError(
                "cited_chunk_ids必须与"
                "诊断引用一致"
            )

        if (
            self.metrics.citation_count
            != len(diagnosis_chunk_ids)
        ):
            raise ValueError(
                "citation_count必须与"
                "诊断引用数量一致"
            )

        if self.execution_state == "aborted":
            if self.status != "failed":
                raise ValueError(
                    "aborted执行必须记录为failed会话"
                )

            if self.finish_reason is not None:
                raise ValueError(
                    "aborted执行不能包含finish_reason"
                )

            return self

        # 以下都是Runner正常结束的情况。
        if self.status == "failed":
            raise ValueError(
                "正常结束不能记录为failed会话"
            )

        if (
            self.finish_reason
            == "human_review_required"
        ):
            if (
                self.status
                != "human_review_required"
            ):
                raise ValueError(
                    "人工审核结束必须使用"
                    "human_review_required状态"
                )

            return self

        if (
            self.status
            != self.diagnosis.status
        ):
            raise ValueError(
                "普通会话状态必须与诊断状态一致"
            )

        return self


class DiagnosticSessionSummary(BaseModel):
    """最近会话列表使用的轻量摘要。

    列表接口不返回完整工具轨迹、Vision观察和引用正文，
    避免一次查询加载大量会话详情。
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    session_id: DiagnosticSessionId
    request_id: str = Field(
        min_length=1,
        max_length=200,
    )
    created_at: datetime
    robot_id: str = Field(
        min_length=1,
        max_length=100,
    )
    symptom_summary: str = Field(
        min_length=1,
        max_length=500,
    )
    status: DiagnosticSessionStatus
    termination_reason: (
        AgentRunTerminationReason
    )
    tool_step_count: int = Field(
        ge=0,
        le=20,
    )
    failed_tool_count: int = Field(
        ge=0,
        le=20,
    )
    citation_count: int = Field(
        ge=0,
        le=10,
    )
    total_duration_ms: float = Field(
        ge=0.0,
    )

    @field_validator("created_at")
    @classmethod
    def created_at_must_have_timezone(
        cls,
        value: datetime,
    ) -> datetime:
        """列表中的时间同样必须可以可靠排序。"""

        if (
            value.tzinfo is None
            or value.utcoffset() is None
        ):
            raise ValueError(
                "created_at必须包含明确时区"
            )

        return value


class DiagnosticSessionListResponse(BaseModel):
    """查询最近诊断会话的API响应契约。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    sessions: tuple[
        DiagnosticSessionSummary,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=100,
        description=(
            "按照created_at从新到旧排列的会话"
        ),
    )

    total: int = Field(
        ge=0,
        description="当前存储中的会话总数",
    )

    limit: int = Field(
        ge=1,
        le=100,
        description="本次查询使用的条数上限",
    )

    @model_validator(mode="after")
    def list_metadata_must_match(
        self,
    ) -> Self:
        """列表数量不能超过limit或total。"""

        if len(self.sessions) > self.limit:
            raise ValueError(
                "sessions数量不能超过limit"
            )

        if len(self.sessions) > self.total:
            raise ValueError(
                "sessions数量不能超过total"
            )

        return self
