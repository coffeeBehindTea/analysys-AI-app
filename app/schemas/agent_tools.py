"""Robot Diagnostic Agent各个具体工具的数据契约。

本模块只定义工具输入和工具输出的结构，
不执行知识库检索，不调用LLM，也不管理Agent循环。

通用的ToolDefinition、ToolCall和ToolExecutionResult
仍然保存在app.schemas.agent中。
"""

from datetime import (
    datetime,
    timedelta,
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
    StringConstraints,
    field_validator,
    model_validator,
)

from app.schemas.knowledge_query import (
    KnowledgeCitation,
)


# Agent只能通过这种受限的不透明引用访问
# 当前请求已经登记的图片。
#
# StringConstraints把字符串清理、长度和正则限制
# 放进可复用类型中。工具输入和请求级图片Store
# 将共享同一份约束，避免两边规则逐渐不一致。
AgentImageReference = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
        pattern=(
            r"^image_[A-Za-z0-9]"
            r"[A-Za-z0-9_-]{0,121}$"
        ),
    ),
]


class AnalyzeRobotImageToolInput(BaseModel):
    """analyze_robot_image工具的输入契约。

    Planner只能引用当前请求已经登记的图片，
    并说明本次需要观察的目标。

    工具输入故意不接受Base64、远程URL、文件路径、
    设备控制参数或任意命令，防止Planner绕过请求级
    图片存储和VisionInputAdapter的安全检查。
    """

    model_config = ConfigDict(
        # 清理普通字符串两端的空白，
        # 但不会改写字符串内部的图片引用或分析目标。
        str_strip_whitespace=True,

        # 拒绝image_base64、image_url、command等
        # 所有未声明的字段。
        extra="forbid",

        # 输入通过校验后不允许重新赋值，
        # 避免Handler执行期间改变图片引用或分析目标。
        frozen=True,
    )

    image_ref: AgentImageReference = Field(
        description=(
            "当前Agent请求中已经验证并登记的"
            "不透明图片引用；不是文件路径或URL"
        ),
        examples=[
            "image_primary",
        ],
    )

    analysis_goal: str = Field(
        min_length=1,
        max_length=2_000,
        description=(
            "需要从图片中直接观察的目标；"
            "该文本属于不可信Planner输入，"
            "不能改变工具权限或系统约束"
        ),
        examples=[
            "检查设备面板上可见的指示灯状态",
        ],
    )


class SearchKnowledgeToolInput(BaseModel):
    """search_knowledge工具的输入契约。

    Agent规划模型只能提供query和top_k，
    不能直接指定文件名、Chunk ID、相似度或引用内容。
    """

    model_config = ConfigDict(
        # 自动清理查询文本首尾空白。
        str_strip_whitespace=True,

        # 拒绝没有声明的参数。
        #
        # 例如LLM错误地生成：
        #
        # {"query": "...", "shell_command": "..."}
        #
        # Pydantic会直接拒绝该调用。
        extra="forbid",

        # 工具输入创建后不允许重新赋值。
        #
        # 这可以避免Handler执行期间修改已经校验的参数。
        frozen=True,
    )

    query: str = Field(
        min_length=1,
        max_length=2_000,
        description=(
            "需要根据知识库证据回答的自然语言问题"
        ),
    )

    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description=(
            "知识库最多返回的候选Chunk数量"
        ),
    )


class SearchKnowledgeToolOutput(BaseModel):
    """search_knowledge成功时的纯证据输出契约。

    当前工具只返回经过确定性门控确认的真实知识库引用，
    不在Agent工具内部再次调用LLM生成回答。

    证据不足时Handler返回None，
    由ToolExecutor转换成empty工具结果。
    """

    model_config = ConfigDict(
        # 清理普通字符串字段的首尾空白。
        str_strip_whitespace=True,

        # 防止工具实现泄漏未声明的内部数据。
        #
        # 特别是旧实现中的answer字段已经不属于
        # 纯检索工具输出，再次出现时必须拒绝。
        extra="forbid",

        # 输出创建后不允许重新赋值，
        # 避免进入Planner历史后又被其他代码修改。
        frozen=True,

        # retrieval_ms不能是NaN或无穷大。
        allow_inf_nan=False,
    )

    citations: list[KnowledgeCitation] = Field(
        min_length=1,
        max_length=10,
        description=(
            "由Python根据门控确认的真实Chunk构造的引用"
        ),
    )

    retrieval_ms: float = Field(
        ge=0.0,
        description=(
            "查询改写、混合检索和确定性证据门控耗时，"
            "单位为毫秒"
        ),
    )


# 机器人当前运行状态。
#
# 这些状态只描述模拟遥测快照，
# 不代表Agent有权改变机器人状态。
RobotOperationalState = Literal[
    "idle",
    "executing_task",
    "paused",
    "faulted",
    "charging",
    "offline",
]


# 故障代码使用规范化后的大写连字符形式，
# 例如ERR-NET-4001。
RobotFaultCode = Annotated[
    str,
    Field(
        min_length=3,
        max_length=64,
        pattern=(
            r"^[A-Z0-9]+"
            r"(?:-[A-Z0-9]+)*$"
        ),
        description=(
            "模拟遥测中已经规范化的活动故障代码"
        ),
    ),
]


class GetRobotTelemetryToolInput(BaseModel):
    """get_robot_telemetry工具的输入契约。"""

    model_config = ConfigDict(
        # 清理机器人编号首尾空格。
        str_strip_whitespace=True,

        # 禁止LLM添加控制指令、目标位置等字段。
        extra="forbid",

        # 输入创建后不能重新赋值。
        frozen=True,
    )

    robot_id: str = Field(
        min_length=1,
        max_length=100,
        description=(
            "需要读取模拟遥测的机器人编号"
        ),
        examples=["robot-001"],
    )


class RobotTelemetryToolOutput(BaseModel):
    """一台机器人的脱敏模拟遥测快照。

    本模型明确标记数据来源为simulated_memory，
    防止Agent把教学模拟数据描述成真实生产遥测。
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    robot_id: str = Field(
        min_length=1,
        max_length=100,
        description="遥测所属机器人编号",
    )

    observed_at: datetime = Field(
        description=(
            "带时区的模拟快照观测时间"
        ),
    )

    location: str = Field(
        min_length=1,
        max_length=200,
        description=(
            "经过脱敏处理的模拟位置"
        ),
        examples=["warehouse-demo/aisle-07/node-14"],
    )

    battery_percent: float = Field(
        ge=0.0,
        le=100.0,
        description="模拟剩余电量百分比",
    )

    operational_state: (
        RobotOperationalState
    ) = Field(
        description="模拟机器人运行状态",
    )

    current_task_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
        description=(
            "当前模拟任务编号；没有任务时为空"
        ),
    )

    active_fault_codes: tuple[
        RobotFaultCode,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=20,
        description=(
            "当前模拟遥测中已经出现的故障代码"
        ),
    )

    speed_mps: float = Field(
        ge=0.0,
        le=20.0,
        description="模拟行驶速度，单位为米每秒",
    )

    network_connected: bool = Field(
        description=(
            "模拟机器人是否连接到调度网络"
        ),
    )

    source: Literal[
        "simulated_memory"
    ] = Field(
        default="simulated_memory",
        description=(
            "明确说明该快照来自脱敏内存模拟数据"
        ),
    )

    @field_validator("observed_at")
    @classmethod
    def observed_at_must_have_timezone(
        cls,
        observed_at: datetime,
    ) -> datetime:
        """观测时间必须包含明确时区。"""

        # tzinfo不为空仍不一定代表有效时区，
        # 因此还要检查utcoffset()。
        if (
            observed_at.tzinfo is None
            or observed_at.utcoffset() is None
        ):
            raise ValueError(
                "observed_at必须包含时区"
            )

        return observed_at

    @field_validator("active_fault_codes")
    @classmethod
    def fault_codes_must_be_unique(
        cls,
        fault_codes: tuple[str, ...],
    ) -> tuple[str, ...]:
        """同一故障代码不能重复出现。"""

        if len(fault_codes) != len(
            set(fault_codes)
        ):
            raise ValueError(
                "active_fault_codes不能重复"
            )

        return fault_codes

    @model_validator(mode="after")
    def executing_task_requires_task_id(
        self,
    ) -> Self:
        """执行任务状态必须提供模拟任务编号。"""

        if (
            self.operational_state
            == "executing_task"
            and self.current_task_id is None
        ):
            raise ValueError(
                "executing_task状态必须提供"
                "current_task_id"
            )

        return self


# draft_test_case只接受当前Agent请求中
# 已经由search_knowledge确认过的Chunk ID。
ConfirmedEvidenceChunkId = Annotated[
    str,
    Field(
        min_length=1,
        max_length=200,
        description=(
            "当前Agent请求中已经确认的知识库Chunk ID"
        ),
    ),
]


# 测试用例草案中的普通文本字段使用统一约束。
#
# 使用类型别名可以避免在多个tuple字段中
# 重复编写相同的字符串长度规则。
DraftTestCaseText = Annotated[
    str,
    Field(
        min_length=1,
        max_length=2_000,
    ),
]


class DraftTestCaseToolInput(BaseModel):
    """draft_test_case工具的输入契约。

    Agent只能提供测试目标和已确认的Chunk ID，
    不能自行提供文件名、页码、证据正文或设备命令。
    """

    model_config = ConfigDict(
        # 清除测试目标和Chunk ID首尾空格。
        str_strip_whitespace=True,

        # 拒绝command、shell、robot_action等
        # 没有在契约中声明的额外参数。
        extra="forbid",

        # 输入校验完成后不允许重新赋值。
        frozen=True,
    )

    objective: str = Field(
        min_length=1,
        max_length=1_000,
        description=(
            "需要形成测试用例草案的测试目标"
        ),
    )

    evidence_chunk_ids: tuple[
        ConfirmedEvidenceChunkId,
        ...,
    ] = Field(
        min_length=1,
        max_length=5,
        description=(
            "必须来自当前请求已确认证据白名单的"
            "Chunk ID，顺序表示期望引用顺序"
        ),
    )

    @field_validator("evidence_chunk_ids")
    @classmethod
    def evidence_chunk_ids_must_be_unique(
        cls,
        chunk_ids: tuple[str, ...],
    ) -> tuple[str, ...]:
        """同一个Chunk不能在一次草案中重复引用。"""

        if len(chunk_ids) != len(
            set(chunk_ids)
        ):
            raise ValueError(
                "evidence_chunk_ids不能重复"
            )

        return chunk_ids


class DraftTestCaseEvidence(BaseModel):
    """测试草案中一条经过白名单解析的证据引用。

    这些字段只能由Python根据ConfirmedEvidenceStore中的
    KnowledgeCitation构造，不能直接相信LLM提供的元数据。
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        frozen=True,
    )

    chunk_id: str = Field(
        min_length=1,
        max_length=200,
        description="真实知识库Chunk的唯一标识",
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
            "引用在原始文档中的页码或章节"
        ),
    )

    chunk_index: int = Field(
        ge=0,
        description=(
            "Chunk在原始文档中的零基顺序"
        ),
    )

    excerpt: str = Field(
        min_length=1,
        max_length=4_000,
        description=(
            "供人工审核草案使用的证据正文预览"
        ),
    )

    excerpt_truncated: bool = Field(
        description=(
            "证据正文是否因输出长度限制被截断"
        ),
    )

    source: Literal[
        "confirmed_knowledge"
    ] = Field(
        default="confirmed_knowledge",
        description=(
            "明确说明证据来自当前请求的"
            "已确认知识库引用"
        ),
    )


class DraftTestCaseStep(BaseModel):
    """测试用例草案中的一个有序步骤。"""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        frozen=True,
    )

    order: int = Field(
        ge=1,
        le=20,
        description=(
            "从1开始的测试步骤顺序"
        ),
    )

    action: str = Field(
        min_length=1,
        max_length=2_000,
        description=(
            "需要人工审核后执行的草案动作"
        ),
    )

    expected_observation: str = Field(
        min_length=1,
        max_length=2_000,
        description=(
            "执行该草案步骤后应记录的观察结果"
        ),
    )


class DraftTestCaseToolOutput(BaseModel):
    """draft_test_case成功时返回的测试用例草案。

    输出明确标记为draft_only并要求人工批准，
    因此不能被Agent解释成已经执行的测试结果。
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        frozen=True,
    )

    title: str = Field(
        min_length=1,
        max_length=200,
        description="测试用例草案标题",
    )

    objective: str = Field(
        min_length=1,
        max_length=1_000,
        description="经过输入契约校验的测试目标",
    )

    preconditions: tuple[
        DraftTestCaseText,
        ...,
    ] = Field(
        min_length=2,
        max_length=8,
        description=(
            "开始测试前必须人工确认的前置条件"
        ),
    )

    steps: tuple[
        DraftTestCaseStep,
        ...,
    ] = Field(
        min_length=3,
        max_length=10,
        description="有序测试步骤草案",
    )

    evidence: tuple[
        DraftTestCaseEvidence,
        ...,
    ] = Field(
        min_length=1,
        max_length=5,
        description=(
            "由已确认证据白名单还原的真实来源"
        ),
    )

    limitations: tuple[
        DraftTestCaseText,
        ...,
    ] = Field(
        min_length=2,
        max_length=8,
        description=(
            "草案的适用限制和人工审核要求"
        ),
    )

    draft_only: Literal[True] = Field(
        default=True,
        description=(
            "结果只能作为草案，不能表示测试已经执行"
        ),
    )

    requires_human_approval: Literal[
        True
    ] = Field(
        default=True,
        description=(
            "草案投入实际使用前必须经过人工批准"
        ),
    )

    @field_validator("evidence")
    @classmethod
    def evidence_must_be_unique(
        cls,
        evidence: tuple[
            DraftTestCaseEvidence,
            ...,
        ],
    ) -> tuple[DraftTestCaseEvidence, ...]:
        """草案输出中不能重复引用同一Chunk。"""

        chunk_ids = tuple(
            item.chunk_id
            for item in evidence
        )

        if len(chunk_ids) != len(
            set(chunk_ids)
        ):
            raise ValueError(
                "evidence中的chunk_id不能重复"
            )

        return evidence

    @model_validator(mode="after")
    def step_order_must_be_continuous(
        self,
    ) -> Self:
        """步骤编号必须从1开始并且连续递增。"""

        actual_orders = tuple(
            step.order
            for step in self.steps
        )

        expected_orders = tuple(
            range(
                1,
                len(self.steps) + 1,
            )
        )

        if actual_orders != expected_orders:
            raise ValueError(
                "steps.order必须从1开始连续递增"
            )

        return self


class GetCurrentTimeToolInput(BaseModel):
    """get_current_time工具的空输入契约。

    当前v1只返回统一的UTC时间，
    因此Agent不需要提供时区或日期参数。
    """

    model_config = ConfigDict(
        # 拒绝timezone、date、command等
        # 所有未声明参数。
        extra="forbid",

        # 空输入对象创建后也不允许增加属性。
        frozen=True,
    )


class GetCurrentTimeToolOutput(BaseModel):
    """应用服务器当前UTC时间的结构化观察结果。

    本模型描述的是系统时钟观察，
    不是机器人遥测时间或文档发布时间。
    """

    model_config = ConfigDict(
        # 禁止工具实现泄漏未声明字段。
        extra="forbid",

        # 输出创建后不可重新赋值。
        frozen=True,
    )

    current_time: datetime = Field(
        description=(
            "工具执行时从应用服务器系统时钟"
            "读取的带时区UTC时间"
        ),
    )

    timezone: Literal["UTC"] = Field(
        default="UTC",
        description=(
            "当前工具统一使用UTC时间"
        ),
    )

    source: Literal[
        "system_clock"
    ] = Field(
        default="system_clock",
        description=(
            "明确说明时间来自应用服务器系统时钟"
        ),
    )

    @field_validator("current_time")
    @classmethod
    def current_time_must_be_utc(
        cls,
        current_time: datetime,
    ) -> datetime:
        """当前时间必须包含时区并且UTC偏移为零。"""

        # tzinfo不为空仍不一定表示时区有效，
        # 所以还需要调用utcoffset()检查实际偏移。
        if (
            current_time.tzinfo is None
            or current_time.utcoffset() is None
        ):
            raise ValueError(
                "current_time必须包含时区"
            )

        # UTC的时区偏移量必须等于零。
        #
        # 例如Asia/Shanghai通常是+08:00，
        # 因此不会通过当前UTC输出契约。
        if (
            current_time.utcoffset()
            != timedelta(0)
        ):
            raise ValueError(
                "current_time必须是UTC时间"
            )

        return current_time
