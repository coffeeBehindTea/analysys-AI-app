"""Week 6 多模态 Agent 离线回归夹具的数据契约。

本模块定义三类可重复的 Fake 行为：

1. Fake Planner 每一轮返回决定还是抛出异常；
2. Fake Vision 针对指定图片返回观察还是模拟失败；
3. Fake 工具针对指定 call_id 返回结果、空结果或失败。

这些模型只描述测试输入，不运行 Agent、不读取图片、
不调用外部服务，也不计算评测指标。

后续离线回归执行器会读取这些夹具，并将它们注入真实的：

AgentDiagnosisService
→ AgentRunner
→ AgentProgressReducer
→ ToolExecutor

从而在没有真实 LLM、Vision、Embedding、Chroma 和 HTTP
服务的情况下复现 30 条多模态场景。
"""

from typing import (
    Annotated,
    Any,
    Literal,
    Self,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from app.schemas.agent import (
    ToolCallId,
    ToolName,
)
from app.schemas.agent_api import (
    AgentDiagnosisResponse,
)
from app.schemas.agent_planning import (
    AgentPlannerDecision,
)
from app.schemas.vision import (
    VisionModelDraft,
)


# 修改夹具字段或校验语义时必须升级版本。
#
# 报告记录这个版本后，可以确认两次离线回归
# 是否使用了相同的 Fake 行为定义。
MULTIMODAL_OFFLINE_FIXTURE_VERSION = (
    "multimodal-offline-fixture-v1"
)


# 离线夹具和执行审计共用同一图片摘要格式。
#
# 统一类型可以避免Fixture接受一种SHA格式，
# 但执行结果又接受另一种格式。
OfflineImageSha256 = Annotated[
    str,
    Field(
        pattern=r"^[0-9a-f]{64}$",
        description=(
            "图片原始字节的64位小写SHA-256摘要"
        ),
    ),
]


# Fake Planner 支持模拟的三类失败。
#
# timeout：
# Planner 在 Runner 规定时间内没有返回。
#
# error：
# Planner 或其上游服务发生普通异常。
#
# invalid_response：
# Planner 返回的结构不能转换成合法决定。
OfflinePlannerFailure = Literal[
    "timeout",
    "error",
    "invalid_response",
]


# Fake Vision 支持模拟的失败类型。
OfflineVisionFailure = Literal[
    "timeout",
    "upstream_error",
    "invalid_response",
]


# Fake 工具结果只模拟已经进入 Executor 后的结果。
#
# rejected 不放在这里，因为工具越权和参数错误
# 应由真实 Runner、ProgressReducer 和 Executor 校验，
# 不能由 Fake 结果直接伪造。
OfflineToolOutcomeStatus = Literal[
    "success",
    "empty",
    "timeout",
    "error",
]


class OfflinePlannerTurn(BaseModel):
    """Fake Planner 一轮调用的预设结果。

    一轮只能出现两种情况之一：

    1. decision：
       返回合法 AgentPlannerDecision；
    2. failure：
       模拟 Planner 超时、异常或无效响应。

    两者不能同时存在，也不能同时为空。
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        frozen=True,
    )

    decision: AgentPlannerDecision | None = Field(
        default=None,
        description=(
            "Fake Planner 本轮返回的结构化决定"
        ),
    )

    failure: OfflinePlannerFailure | None = Field(
        default=None,
        description=(
            "Fake Planner 本轮需要模拟的失败"
        ),
    )

    @model_validator(mode="after")
    def exactly_one_outcome_is_required(
        self,
    ) -> Self:
        """保证决定和失败严格二选一。"""

        outcome_count = sum(
            (
                self.decision is not None,
                self.failure is not None,
            )
        )

        if outcome_count != 1:
            raise ValueError(
                "decision和failure必须且只能设置一个"
            )

        return self


class OfflineVisionOutcome(BaseModel):
    """Fake Vision 针对一张图片的预设结果。

    FakeVisionProvider 使用图片 SHA-256 查找结果，
    因而这里不保存图片字节、Base64 或本地绝对路径。
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        frozen=True,
    )

    image_sha256: OfflineImageSha256 = Field(
        description=(
            "与场景图片对应的 SHA-256 十六进制摘要"
        ),
    )

    draft: VisionModelDraft | None = Field(
        default=None,
        description=(
            "Fake Vision 成功时返回的结构化视觉草稿"
        ),
    )

    failure: OfflineVisionFailure | None = Field(
        default=None,
        description=(
            "Fake Vision 需要模拟的失败"
        ),
    )

    @model_validator(mode="after")
    def exactly_one_outcome_is_required(
        self,
    ) -> Self:
        """保证视觉草稿和视觉失败严格二选一。"""

        outcome_count = sum(
            (
                self.draft is not None,
                self.failure is not None,
            )
        )

        if outcome_count != 1:
            raise ValueError(
                "draft和failure必须且只能设置一个"
            )

        return self


class OfflineToolOutcome(BaseModel):
    """一次非 Vision 工具调用的预设 Fake 结果。

    call_id 和 tool_name 必须与 Fake Planner 提出的
    ToolCall 对应。

    output 保存工具处理函数的原始结构化结果。
    后续仍要由真实 ToolExecutor 使用工具的
    output_model 重新校验，不能绕过生产数据契约。
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        frozen=True,
    )

    call_id: ToolCallId = Field(
        description=(
            "与 Planner 工具请求对应的调用 ID"
        ),
    )

    tool_name: ToolName = Field(
        description=(
            "与 Planner 工具请求对应的工具名称"
        ),
    )

    status: OfflineToolOutcomeStatus = Field(
        description=(
            "Fake 工具返回成功、空结果、超时还是异常"
        ),
    )

    output: dict[str, Any] | None = Field(
        default=None,
        description=(
            "status为success时交给真实输出模型校验的数据"
        ),
    )

    @model_validator(mode="after")
    def status_must_match_output(
        self,
    ) -> Self:
        """工具状态与是否携带业务输出必须一致。"""

        if self.status == "success":
            if self.output is None:
                raise ValueError(
                    "status与output不一致："
                    "success必须包含output"
                )

            return self

        # empty、timeout和error都不能携带业务结果。
        #
        # 它们会由离线执行器转换成None或稳定异常，
        # 再交给真实ToolExecutor生成统一执行结果。
        if self.output is not None:
            raise ValueError(
                "status与output不一致："
                "非success状态不能包含output"
            )

        return self


class MultimodalOfflineScenarioFixture(BaseModel):
    """一条多模态 Gold 场景对应的完整离线行为夹具。

    本模型不会复制 Gold 中的预期结果。

    Gold 回答“系统应该做什么”；
    Fixture 回答“外部依赖在本次测试中返回什么”。

    将两者分离可以避免把预期答案直接当成实际结果，
    从而降低评测自证和测试过拟合的风险。
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        frozen=True,
    )

    fixture_version: Literal[
        "multimodal-offline-fixture-v1"
    ] = Field(
        default=MULTIMODAL_OFFLINE_FIXTURE_VERSION,
        description=(
            "离线回归夹具的数据契约版本"
        ),
    )

    scenario_id: str = Field(
        pattern=r"^multimodal-agent-[0-9]{3}$",
        description=(
            "必须与多模态Gold场景编号一致"
        ),
    )

    planner_turns: tuple[
        OfflinePlannerTurn,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=10,
        description=(
            "Fake Planner 按顺序消费的预设轮次"
        ),
    )

    vision_outcomes: tuple[
        OfflineVisionOutcome,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=3,
        description=(
            "按图片SHA-256索引的Fake Vision结果"
        ),
    )

    tool_outcomes: tuple[
        OfflineToolOutcome,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=10,
        description=(
            "非Vision工具调用对应的Fake结果"
        ),
    )

    @model_validator(mode="after")
    def fixture_parts_must_be_consistent(
        self,
    ) -> Self:
        """交叉校验 Planner、Vision 和工具夹具。

        检查内容包括：

        1. Planner call_id 不能重复；
        2. Planner 的终止轮次只能位于最后；
        3. 非空 Planner 脚本必须有确定的终止轮次；
        4. 非 Vision 工具调用与工具结果逐项对应；
        5. Vision 调用必须配置 Vision 结果；
        6. 没有 Vision 调用时不能保留无用结果；
        7. 同一图片不能配置多个竞争结果。
        """

        planner_calls = tuple(
            turn.decision.tool_call
            for turn in self.planner_turns
            if (
                turn.decision is not None
                and turn.decision.decision
                == "call_tool"
            )
        )

        call_ids = tuple(
            tool_call.call_id
            for tool_call in planner_calls
            if tool_call is not None
        )

        if len(call_ids) != len(set(call_ids)):
            raise ValueError(
                "planner_turns不能包含重复call_id"
            )

        terminal_indexes = tuple(
            index
            for index, turn
            in enumerate(self.planner_turns)
            if (
                turn.failure is not None
                or (
                    turn.decision is not None
                    and turn.decision.decision
                    == "finish"
                )
            )
        )

        if any(
            index != len(self.planner_turns) - 1
            for index in terminal_indexes
        ):
            raise ValueError(
                "终止Planner轮次必须位于最后"
            )

        if (
            self.planner_turns
            and not terminal_indexes
        ):
            raise ValueError(
                "最后一轮必须结束规划或模拟Planner失败"
            )

        planned_non_vision_calls = tuple(
            (
                tool_call.call_id,
                tool_call.tool_name,
            )
            for tool_call in planner_calls
            if (
                tool_call is not None
                and tool_call.tool_name
                != "analyze_robot_image"
            )
        )

        supplied_tool_outcomes = tuple(
            (
                outcome.call_id,
                outcome.tool_name,
            )
            for outcome in self.tool_outcomes
        )

        # 同一个(call_id, tool_name)也不能配置两次。
        if len(supplied_tool_outcomes) != len(
            set(supplied_tool_outcomes)
        ):
            raise ValueError(
                "tool_outcomes不能包含重复工具结果"
            )

        # 不只比较集合，还必须比较顺序。
        #
        # ScriptedOfflineToolHandler按队列顺序消费结果。
        # 如果两个结果的call_id集合相同但顺序相反，
        # 第一次调用就会错误取得第二次调用的输出。
        if (
            planned_non_vision_calls
            != supplied_tool_outcomes
        ):
            raise ValueError(
                "非Vision工具调用必须与"
                "tool_outcomes按顺序逐项对应"
            )

        has_vision_call = any(
            tool_call is not None
            and tool_call.tool_name
            == "analyze_robot_image"
            for tool_call in planner_calls
        )

        if (
            has_vision_call
            and not self.vision_outcomes
        ):
            raise ValueError(
                "Vision工具调用必须配置vision_outcomes"
            )

        if (
            not has_vision_call
            and self.vision_outcomes
        ):
            raise ValueError(
                "没有Vision调用时不能配置vision_outcomes"
            )

        image_hashes = tuple(
            outcome.image_sha256
            for outcome in self.vision_outcomes
        )

        if len(image_hashes) != len(
            set(image_hashes)
        ):
            raise ValueError(
                "vision_outcomes不能包含"
                "重复image_sha256"
            )

        return self


class OfflineToolCallAudit(BaseModel):
    """一个非 Vision 工具的离线调用计数。

    planned_outcome_count来自夹具中预设的结果数量；
    actual_call_count来自Fake Handler真实记录的调用数量。

    两者分开保存，后续报告才能识别：

    - 预设结果没有被消费；
    - Planner额外调用了没有预设结果的工具；
    - 实际调用次数与夹具设计一致。
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        frozen=True,
    )

    tool_name: ToolName = Field(
        description="被审计的非Vision工具名称",
    )

    planned_outcome_count: int = Field(
        ge=0,
        le=10,
        description="夹具为该工具预设的结果数量",
    )

    actual_call_count: int = Field(
        ge=0,
        le=20,
        description="Agent主链实际调用该工具的次数",
    )

    @property
    def fully_consumed(self) -> bool:
        """判断该工具的预设结果是否被恰好消费。"""

        return (
            self.planned_outcome_count
            == self.actual_call_count
        )


class MultimodalOfflineScenarioExecution(BaseModel):
    """一条离线场景执行后的结构化审计结果。

    response是现有AgentDiagnosisService产生的真实公开响应；
    其他字段只记录Fake依赖的调用计数和工具暴露范围。

    本模型不计算场景是否通过。后续评分器会把response
    与Gold期望比较，避免执行器自己既运行又给自己打分。
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        frozen=True,
    )

    fixture_version: Literal[
        "multimodal-offline-fixture-v1"
    ] = Field(
        default=MULTIMODAL_OFFLINE_FIXTURE_VERSION,
        description="本次执行使用的离线夹具版本",
    )

    scenario_id: str = Field(
        pattern=r"^multimodal-agent-[0-9]{3}$",
        description="本次执行对应的Gold场景编号",
    )

    response: AgentDiagnosisResponse = Field(
        description="现有Agent业务主链生成的公开响应",
    )

    planner_turn_count: int = Field(
        ge=0,
        le=10,
        description="夹具预设的Planner轮次数量",
    )

    planner_call_count: int = Field(
        ge=0,
        le=20,
        description="AgentRunner实际调用Planner的次数",
    )

    planner_exposed_tool_names: tuple[
        tuple[ToolName, ...],
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=20,
        description=(
            "每次Planner调用实际看到的最小工具名称集合"
        ),
    )

    vision_outcome_count: int = Field(
        ge=0,
        le=3,
        description="夹具预设的Vision结果数量",
    )

    planned_vision_image_sha256s: tuple[
        OfflineImageSha256,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=3,
        description=(
            "夹具预设Vision结果对应的图片摘要"
        ),
    )

    vision_call_count: int = Field(
        ge=0,
        le=40,
        description="Fake Vision Provider实际调用次数",
    )

    vision_called_image_sha256s: tuple[
        OfflineImageSha256,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=40,
        description=(
            "Provider每次实际调用对应的图片摘要；"
            "同一图片重试时允许重复"
        ),
    )

    tool_call_audits: tuple[
        OfflineToolCallAudit,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=5,
        description="每个非Vision工具的预设与实际调用计数",
    )

    fixture_fully_consumed: bool = Field(
        description=(
            "Planner、Vision和普通工具预设结果"
            "是否全部被恰好消费"
        ),
    )

    @model_validator(mode="after")
    def audit_counts_must_be_consistent(
        self,
    ) -> Self:
        """保证审计数量和明细结构相互一致。"""

        if len(
            self.planner_exposed_tool_names
        ) != self.planner_call_count:
            raise ValueError(
                "planner_exposed_tool_names数量"
                "必须等于planner_call_count"
            )

        if self.vision_outcome_count != len(
            self.planned_vision_image_sha256s
        ):
            raise ValueError(
                "planned_vision_image_sha256s数量"
                "必须等于vision_outcome_count"
            )

        if self.vision_call_count != len(
            self.vision_called_image_sha256s
        ):
            raise ValueError(
                "vision_called_image_sha256s数量"
                "必须等于vision_call_count"
            )

        tool_names = tuple(
            audit.tool_name
            for audit in self.tool_call_audits
        )

        if len(tool_names) != len(set(tool_names)):
            raise ValueError(
                "tool_call_audits不能包含重复tool_name"
            )

        calculated_fully_consumed = (
            self.planner_turn_count
            == self.planner_call_count
            # Vision结果按图片摘要索引，而不是按调用次数
            # 顺序消费。同一图片因Provider重试被调用两次时，
            # 仍然只对应一个预设结果。
            and set(
                self.planned_vision_image_sha256s
            )
            == set(
                self.vision_called_image_sha256s
            )
            and all(
                audit.fully_consumed
                for audit in self.tool_call_audits
            )
        )

        if (
            self.fixture_fully_consumed
            != calculated_fully_consumed
        ):
            raise ValueError(
                "fixture_fully_consumed与审计计数不一致"
            )

        return self
