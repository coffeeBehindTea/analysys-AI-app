"""Agent证据驱动进度状态的数据契约。

本模块定义：

1. Agent当前处于哪一种业务进度状态；
2. 每次工具调用是否产生了新的可信信息；
3. 已经完成了哪些请求能力；
4. 已经确认了哪些来源和证据；
5. 是否存在相互冲突的来源；
6. 下一轮仍然允许调用哪些工具；
7. Agent停止时使用的结构化原因。

本模块只定义数据结构和字段关系，不负责：

1. 调用Planner；
2. 执行工具；
3. 从工具输出中提取证据；
4. 计算下一份AgentProgress；
5. 生成最终HTTP响应。

后续AgentProgressReducer会读取工具策略、旧进度和新的
工具执行结果，构造一份新的AgentProgress。
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
)
from app.schemas.agent_tool_policy import (
    AGENT_READ_ONLY_TOOL_ORDER,
    CAPABILITY_TO_TOOL,
    AgentReadOnlyToolName,
    AgentRequiredCapability,
)


# 一条尚未解决的信息说明。
#
# 该基础类型放在AgentProgress所在的底层Schema中，
# 使AgentRunResult可以安全包含AgentProgress，避免两个
# Schema模块相互导入形成循环依赖。
AgentMissingInformation = Annotated[
    str,
    Field(
        min_length=1,
        max_length=500,
        description=(
            "Agent结束时仍然缺少或"
            "无法确认的一项信息"
        ),
    ),
]


# AgentProgressState表达业务任务的完成程度。
#
# 它与AgentLifecycleState不同：
#
# AgentLifecycleState表示程序正在planning还是tool_running；
# AgentProgressState表示现有证据能够支持什么级别的结果。
AgentProgressState = Literal[
    # Agent仍需执行允许范围内的工具。
    "collecting",

    # 所有必要能力都已有可信工具结果，
    # 下一轮只允许Planner生成finish草稿，不能再调用工具。
    # 该状态还不是completed，因为最终草稿尚未通过
    # 结构校验、引用白名单和报告构造校验。
    "ready_to_finish",

    # 所有必要能力都已经完成，并且没有缺失信息。
    "completed",

    # 已有证据可以支持部分结果，
    # 但仍缺少至少一项必要信息。
    "partial",

    # 现有信息不足以安全形成原因或检查建议。
    "abstained",

    # 发现证据冲突或风险边界，必须交给人工处理。
    "human_review_required",

    # Planner、工具或内部处理发生了不可恢复失败。
    "failed",
]


# stop_reason是稳定的机器可读停止原因。
#
# 业务代码和测试应该判断这些值，
# 不应该解析可能变化的中文说明。
AgentProgressStopReason = Literal[
    "sufficient_evidence",
    "partial_evidence",
    "insufficient_information",
    "conflicting_evidence",
    "human_review_required",
    "runtime_failure",
]


# 来源标识不等于知识库Chunk ID。
#
# 来源可以是：
#
# knowledge:<chunk_id>
# vision:<image_sha256>
# telemetry:<robot_id>:<observed_at>
# test_draft:<call_id>
AgentProgressSourceId = Annotated[
    str,
    Field(
        min_length=1,
        max_length=500,
        description=(
            "一项经过应用确认的数据来源标识"
        ),
    ),
]


# evidence_id表示一项能够支持结论的具体证据。
#
# 知识库证据通常直接使用Chunk ID。
# 视觉和遥测证据可以使用由应用构造的稳定标识。
AgentProgressEvidenceId = Annotated[
    str,
    Field(
        min_length=1,
        max_length=500,
        description=(
            "一项已经覆盖当前任务事实的证据标识"
        ),
    ),
]


# 工具参数不能原样保存在Progress中，
# 因为其中可能出现用户日志或其他敏感内容。
#
# 后续Reducer会对规范化后的工具名和参数计算SHA-256，
# 只在Progress中保存64位十六进制摘要。
AgentToolCallSignature = Annotated[
    str,
    Field(
        pattern=r"^[0-9a-f]{64}$",
        description=(
            "规范化工具名称与参数的SHA-256摘要"
        ),
    ),
]


class AgentToolProgressRecord(BaseModel):
    """一次工具尝试产生的脱敏进度记录。

    它不是完整ToolExecutionResult的副本。

    这里只保留状态机判断下一步需要的信息：

    1. 调用了哪个工具；
    2. 调用参数的不可逆摘要；
    3. 工具执行状态；
    4. 是否产生了新来源或新证据；
    5. 失败时的稳定错误码。
    """

    model_config = ConfigDict(
        # 清除普通字符串字段首尾空白。
        str_strip_whitespace=True,

        # 拒绝未声明字段，避免保存原始参数、
        # 完整工具输出或模型私有推理。
        extra="forbid",

        # 进度历史创建后不能被原地修改。
        frozen=True,
    )

    step_number: int = Field(
        ge=1,
        le=20,
        description=(
            "该工具尝试在Agent循环中的步骤编号"
        ),
    )

    call_id: str = Field(
        min_length=1,
        max_length=200,
        description=(
            "用于关联原始ToolCall和ToolExecutionResult"
        ),
    )

    tool_name: AgentReadOnlyToolName = Field(
        description=(
            "本次实际尝试调用的只读工具"
        ),
    )

    call_signature: AgentToolCallSignature = Field(
        description=(
            "用于检测相同工具和参数是否被重复请求"
        ),
    )

    status: ToolExecutionStatus = Field(
        description=(
            "工具执行成功、空结果、拒绝、超时或错误"
        ),
    )

    produced_new_information: bool = Field(
        description=(
            "本次调用是否增加了新的来源或证据"
        ),
    )

    new_source_ids: tuple[
        AgentProgressSourceId,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=20,
        description=(
            "本次工具调用首次确认的来源"
        ),
    )

    new_evidence_ids: tuple[
        AgentProgressEvidenceId,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=20,
        description=(
            "本次工具调用首次覆盖的证据"
        ),
    )

    error_code: ToolErrorCode | None = Field(
        default=None,
        description=(
            "非成功工具结果对应的稳定错误码"
        ),
    )

    @field_validator(
        "new_source_ids",
        "new_evidence_ids",
    )
    @classmethod
    def new_items_must_be_unique(
        cls,
        values: tuple[str, ...],
    ) -> tuple[str, ...]:
        """单次工具记录不能重复声明同一项新信息。"""

        if len(values) != len(set(values)):
            raise ValueError(
                "单次工具记录中的新信息不能重复"
            )

        return values

    @model_validator(mode="after")
    def fields_must_match_tool_status(
        self,
    ) -> Self:
        """校验执行状态、新信息和错误码之间的关系。"""

        has_new_information = bool(
            self.new_source_ids
            or self.new_evidence_ids
        )

        if (
            self.produced_new_information
            != has_new_information
        ):
            raise ValueError(
                "produced_new_information必须与"
                "new_source_ids和new_evidence_ids一致"
            )

        if self.status == "success":
            if self.error_code is not None:
                raise ValueError(
                    "成功工具记录不能包含error_code"
                )

            return self

        # 空结果、拒绝、超时和错误不能声称
        # 自己确认了新的可信来源或证据。
        if has_new_information:
            raise ValueError(
                "非成功工具记录不能产生新信息"
            )

        if self.error_code is None:
            raise ValueError(
                "非成功工具记录必须包含error_code"
            )

        return self


class AgentEvidenceConflict(BaseModel):
    """两项或更多已确认来源之间的证据冲突。

    冲突必须引用已经进入当前Progress的来源，
    不能由Planner直接编造一个没有出处的冲突。
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        frozen=True,
    )

    description: str = Field(
        min_length=1,
        max_length=500,
        description=(
            "经过清理、可以公开记录的冲突说明"
        ),
    )

    source_ids: tuple[
        AgentProgressSourceId,
        ...,
    ] = Field(
        min_length=2,
        max_length=10,
        description=(
            "参与冲突的至少两个已确认来源"
        ),
    )

    evidence_ids: tuple[
        AgentProgressEvidenceId,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=10,
        description=(
            "参与冲突的具体证据标识"
        ),
    )

    @field_validator(
        "source_ids",
        "evidence_ids",
    )
    @classmethod
    def conflict_items_must_be_unique(
        cls,
        values: tuple[str, ...],
    ) -> tuple[str, ...]:
        """一条冲突中不能重复引用同一标识。"""

        if len(values) != len(set(values)):
            raise ValueError(
                "冲突记录不能包含重复标识"
            )

        return values


class AgentProgress(BaseModel):
    """一次Agent请求在某个时间点的完整进度快照。

    每次状态发生变化时都创建一个新对象，
    而不是修改旧对象。

    这样可以使状态转换可测试、可审计、可复现。
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        frozen=True,
    )

    progress_version: str = Field(
        default="agent-progress-v1",
        min_length=1,
        max_length=100,
        pattern=r"^[a-z0-9][a-z0-9._-]*$",
        description=(
            "进度计算规则的版本"
        ),
    )

    state: AgentProgressState = Field(
        default="collecting",
        description=(
            "当前证据能够支持的业务进度状态"
        ),
    )

    required_capabilities: tuple[
        AgentRequiredCapability,
        ...,
    ] = Field(
        min_length=1,
        max_length=5,
        description=(
            "工具策略确定的本次请求必要能力"
        ),
    )

    completed_capabilities: tuple[
        AgentRequiredCapability,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=5,
        description=(
            "已经由真实工具结果完成的必要能力"
        ),
    )

    tool_records: tuple[
        AgentToolProgressRecord,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=20,
        description=(
            "按实际发生顺序保存的脱敏工具尝试记录"
        ),
    )

    confirmed_source_ids: tuple[
        AgentProgressSourceId,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=100,
        description=(
            "截至当前已经确认的全部来源"
        ),
    )

    covered_evidence_ids: tuple[
        AgentProgressEvidenceId,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=100,
        description=(
            "截至当前已经覆盖的全部证据"
        ),
    )

    evidence_conflicts: tuple[
        AgentEvidenceConflict,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=10,
        description=(
            "当前已确认来源之间仍未解决的冲突"
        ),
    )

    missing_information: tuple[
        AgentMissingInformation,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=20,
        description=(
            "当前仍缺少或无法确认的信息"
        ),
    )

    allowed_next_tool_names: tuple[
        AgentReadOnlyToolName,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=5,
        description=(
            "下一轮仍允许Planner请求的最小工具集合"
        ),
    )

    stop_reason: (
        AgentProgressStopReason | None
    ) = Field(
        default=None,
        description=(
            "终止状态使用的机器可读原因；"
            "collecting状态必须为空"
        ),
    )

    @field_validator(
        "required_capabilities",
        "completed_capabilities",
        "confirmed_source_ids",
        "covered_evidence_ids",
        "missing_information",
    )
    @classmethod
    def summary_items_must_be_unique(
        cls,
        values: tuple[str, ...],
    ) -> tuple[str, ...]:
        """进度摘要字段不能包含重复项。"""

        if len(values) != len(set(values)):
            raise ValueError(
                "AgentProgress摘要字段不能包含重复项"
            )

        return values

    @field_validator(
        "allowed_next_tool_names"
    )
    @classmethod
    def next_tools_must_use_stable_order(
        cls,
        values: tuple[
            AgentReadOnlyToolName,
            ...,
        ],
    ) -> tuple[
        AgentReadOnlyToolName,
        ...,
    ]:
        """下一步工具必须使用项目统一的稳定顺序。"""

        expected = tuple(
            tool_name
            for tool_name
            in AGENT_READ_ONLY_TOOL_ORDER
            if tool_name in values
        )

        if values != expected:
            raise ValueError(
                "allowed_next_tool_names必须按照"
                "AGENT_READ_ONLY_TOOL_ORDER排列"
            )

        return values

    @model_validator(mode="after")
    def progress_relationships_must_be_consistent(
        self,
    ) -> Self:
        """校验能力、工具历史、证据和状态之间的关系。"""

        required_set = set(
            self.required_capabilities
        )
        completed_set = set(
            self.completed_capabilities
        )

        # Agent不能完成一个本次任务原本不需要的能力。
        if not completed_set.issubset(
            required_set
        ):
            raise ValueError(
                "completed_capabilities必须是"
                "required_capabilities的子集"
            )

        # 工具记录的步骤必须从1开始连续。
        actual_steps = tuple(
            record.step_number
            for record in self.tool_records
        )
        expected_steps = tuple(
            range(
                1,
                len(self.tool_records) + 1,
            )
        )

        if actual_steps != expected_steps:
            raise ValueError(
                "tool_records步骤必须从1开始连续递增"
            )

        # 汇总来源必须恰好等于各工具记录首次增加来源的
        # 去重、首次出现顺序结果。
        expected_source_ids = tuple(
            dict.fromkeys(
                source_id
                for record in self.tool_records
                for source_id in record.new_source_ids
            )
        )

        if (
            self.confirmed_source_ids
            != expected_source_ids
        ):
            raise ValueError(
                "confirmed_source_ids必须与"
                "tool_records中的新来源一致"
            )

        # 汇总证据使用同样的确定性计算规则。
        expected_evidence_ids = tuple(
            dict.fromkeys(
                evidence_id
                for record in self.tool_records
                for evidence_id
                in record.new_evidence_ids
            )
        )

        if (
            self.covered_evidence_ids
            != expected_evidence_ids
        ):
            raise ValueError(
                "covered_evidence_ids必须与"
                "tool_records中的新证据一致"
            )

        confirmed_source_set = set(
            self.confirmed_source_ids
        )
        covered_evidence_set = set(
            self.covered_evidence_ids
        )

        # 冲突只能引用当前进度中已经确认的内容。
        for conflict in self.evidence_conflicts:
            if not set(
                conflict.source_ids
            ).issubset(
                confirmed_source_set
            ):
                raise ValueError(
                    "证据冲突引用了未确认来源"
                )

            if not set(
                conflict.evidence_ids
            ).issubset(
                covered_evidence_set
            ):
                raise ValueError(
                    "证据冲突引用了未覆盖证据"
                )

        possible_tools = {
            CAPABILITY_TO_TOOL[capability]
            for capability
            in self.required_capabilities
        }

        # 状态机不能在运行过程中扩大请求级权限。
        if not set(
            self.allowed_next_tool_names
        ).issubset(
            possible_tools
        ):
            raise ValueError(
                "下一步工具不能超出必要能力对应范围"
            )

        # 有冲突时不能声称任务已经完成，
        # 也不能继续静默自动处理。
        if (
            self.evidence_conflicts
            and self.state
            != "human_review_required"
        ):
            raise ValueError(
                "存在证据冲突时必须转人工审核"
            )

        if self.state == "collecting":
            if self.stop_reason is not None:
                raise ValueError(
                    "collecting状态不能包含stop_reason"
                )

            if completed_set == required_set:
                raise ValueError(
                    "全部必要能力完成后必须进入"
                    "ready_to_finish"
                )

            if not self.allowed_next_tool_names:
                raise ValueError(
                    "collecting状态必须允许至少一个下一步工具"
                )

            incomplete_capabilities = (
                required_set
                - completed_set
            )
            expected_next_tools = tuple(
                tool_name
                for tool_name
                in AGENT_READ_ONLY_TOOL_ORDER
                if tool_name
                in {
                    CAPABILITY_TO_TOOL[capability]
                    for capability
                    in incomplete_capabilities
                }
            )

            if (
                self.allowed_next_tool_names
                != expected_next_tools
            ):
                raise ValueError(
                    "collecting状态的下一步工具必须"
                    "与未完成能力完全对应"
                )

            return self

        if self.state == "ready_to_finish":
            if self.stop_reason is not None:
                raise ValueError(
                    "ready_to_finish状态不能包含"
                    "stop_reason"
                )

            if self.allowed_next_tool_names:
                raise ValueError(
                    "ready_to_finish状态不能继续允许工具调用"
                )

            if completed_set != required_set:
                raise ValueError(
                    "ready_to_finish必须完成全部必要能力"
                )

            if self.evidence_conflicts:
                raise ValueError(
                    "ready_to_finish不能包含证据冲突"
                )

            return self

        # collecting和ready_to_finish以外都是终止状态。
        if self.allowed_next_tool_names:
            raise ValueError(
                "终止状态不能继续允许工具调用"
            )

        if self.stop_reason is None:
            raise ValueError(
                "终止状态必须包含stop_reason"
            )

        if self.state == "completed":
            if (
                self.stop_reason
                != "sufficient_evidence"
            ):
                raise ValueError(
                    "completed必须使用"
                    "sufficient_evidence"
                )

            if completed_set != required_set:
                raise ValueError(
                    "completed必须完成全部必要能力"
                )

            if (
                self.missing_information
                or self.evidence_conflicts
            ):
                raise ValueError(
                    "completed不能包含缺失信息或证据冲突"
                )

            return self

        if self.state == "partial":
            if (
                self.stop_reason
                != "partial_evidence"
            ):
                raise ValueError(
                    "partial必须使用partial_evidence"
                )

            if not self.completed_capabilities:
                raise ValueError(
                    "partial必须至少完成一项能力"
                )

            if not self.missing_information:
                raise ValueError(
                    "partial必须说明缺失信息"
                )

            return self

        if self.state == "abstained":
            if (
                self.stop_reason
                != "insufficient_information"
            ):
                raise ValueError(
                    "abstained必须使用"
                    "insufficient_information"
                )

            if not self.missing_information:
                raise ValueError(
                    "abstained必须说明缺失信息"
                )

            return self

        if (
            self.state
            == "human_review_required"
        ):
            if self.stop_reason not in {
                "conflicting_evidence",
                "human_review_required",
            }:
                raise ValueError(
                    "human_review_required使用了"
                    "错误的stop_reason"
                )

            if (
                self.stop_reason
                == "conflicting_evidence"
                and not self.evidence_conflicts
            ):
                raise ValueError(
                    "conflicting_evidence必须包含"
                    "evidence_conflicts"
                )

            if not self.missing_information:
                raise ValueError(
                    "人工审核状态必须说明待人工处理内容"
                )

            return self

        # 只剩failed状态。
        if self.stop_reason != "runtime_failure":
            raise ValueError(
                "failed必须使用runtime_failure"
            )

        if not self.missing_information:
            raise ValueError(
                "failed必须说明未完成内容"
            )

        return self
