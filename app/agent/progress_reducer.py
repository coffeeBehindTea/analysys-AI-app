"""Agent证据驱动进度的确定性状态转换器。

本模块负责：

1. 根据请求级工具策略创建初始AgentProgress；
2. 在工具执行前检查当前状态是否允许该调用；
3. 计算不包含原始参数的工具调用摘要；
4. 从五种已校验工具输出中提取来源和证据标识；
5. 记录工具是否产生了新信息；
6. 更新已经完成的能力；
7. 所有必要能力完成后进入ready_to_finish。

本模块不负责：

1. 调用LLM或Planner；
2. 执行工具；
3. 判断自然语言答案是否正确；
4. 直接生成HTTP响应；
5. 保存完整工具参数、图片、日志或模型思维链。

AgentRunner后续会在每次工具执行前后调用本模块。
"""

from hashlib import (
    sha256,
)
import json
from typing import (
    Literal,
)

from app.schemas.agent import (
    ToolCall,
)
from app.schemas.agent_planning import (
    AgentPlannerFinishReason,
    AgentToolInteraction,
)
from app.schemas.agent_progress import (
    AgentEvidenceConflict,
    AgentProgress,
    AgentProgressEvidenceId,
    AgentProgressSourceId,
    AgentToolCallSignature,
    AgentToolProgressRecord,
)
from app.schemas.agent_runtime import (
    AgentMissingInformation,
)
from app.schemas.agent_tool_policy import (
    CAPABILITY_TO_TOOL,
    AgentRequiredCapability,
    AgentToolPolicyDecision,
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


# 状态转换被拒绝时使用稳定的机器可读原因。
#
# Runner应该判断reason，而不是解析异常中的中文文本。
AgentProgressTransitionReason = Literal[
    # 只有collecting状态允许请求工具。
    "progress_not_collecting",

    # Planner请求了当前进度没有允许的工具。
    "tool_not_allowed",

    # 工具名和参数与已经尝试过的调用相同。
    "duplicate_tool_call",

    # AgentToolInteraction的步骤编号与当前进度不连续。
    "step_number_mismatch",

    # 已经终止的Progress不能再次完成或失败。
    "progress_already_terminal",
]


class AgentProgressTransitionError(ValueError):
    """一次Agent进度转换被确定性规则拒绝。

    reason供Runner、测试和日志做稳定判断；
    异常消息只用于开发人员理解问题。
    """

    def __init__(
        self,
        *,
        reason: AgentProgressTransitionReason,
        message: str,
    ) -> None:
        """保存稳定原因并初始化ValueError。"""

        # 调用父类构造器后，
        # str(exc)才能返回message。
        super().__init__(message)

        self.reason = reason


# 由现有能力到工具的映射反向生成。
#
# CAPABILITY_TO_TOOL：
#     ability -> tool
#
# TOOL_TO_CAPABILITY：
#     tool -> ability
#
# 使用现有映射反向生成，可以避免维护两张可能不一致的表。
TOOL_TO_CAPABILITY = {
    tool_name: capability
    for capability, tool_name
    in CAPABILITY_TO_TOOL.items()
}


# 当Planner过早结束或明确声明信息不足时，
# Python根据尚未完成的能力生成稳定缺失信息。
# 不使用LLM自然语言推断哪些能力仍然缺失。
CAPABILITY_MISSING_INFORMATION: dict[
    AgentRequiredCapability,
    AgentMissingInformation,
] = {
    "vision_observation": (
        "尚未取得任务所需的图片观察"
    ),
    "knowledge_evidence": (
        "尚未取得任务所需的知识库证据"
    ),
    "robot_telemetry": (
        "尚未取得任务所需的模拟遥测快照"
    ),
    "test_case_draft": (
        "尚未形成任务要求的测试用例草案"
    ),
    "current_time": (
        "尚未取得任务所需的UTC时间观察"
    ),
}


def _merge_unique_items(
    *groups: tuple[str, ...],
) -> tuple[str, ...]:
    """合并多个元组，同时保留首次出现顺序。"""

    return tuple(
        dict.fromkeys(
            item
            for group in groups
            for item in group
        )
    )


def _incomplete_capability_messages(
    progress: AgentProgress,
    /,
) -> tuple[AgentMissingInformation, ...]:
    """按必要能力原始顺序生成缺失信息。"""

    completed_capabilities = set(
        progress.completed_capabilities
    )

    return tuple(
        CAPABILITY_MISSING_INFORMATION[
            capability
        ]
        for capability
        in progress.required_capabilities
        if capability
        not in completed_capabilities
    )


def _validate_active_progress(
    progress: AgentProgress,
    /,
) -> None:
    """终止转换只接受尚未终止的进度。"""

    if not isinstance(
        progress,
        AgentProgress,
    ):
        raise TypeError(
            "progress必须是AgentProgress"
        )

    if progress.state not in {
        "collecting",
        "ready_to_finish",
    }:
        raise AgentProgressTransitionError(
            reason="progress_already_terminal",
            message=(
                "已经终止的AgentProgress"
                "不能再次转换"
            ),
        )


def _build_updated_progress(
    progress: AgentProgress,
    /,
    **updates: object,
) -> AgentProgress:
    """应用更新并重新执行AgentProgress全部校验。"""

    progress_data = progress.model_dump(
        mode="python"
    )
    progress_data.update(updates)

    return AgentProgress.model_validate(
        progress_data
    )


def _canonical_tool_call_signature(
    tool_call: ToolCall,
    /,
) -> AgentToolCallSignature:
    """生成不包含原始参数的稳定工具调用摘要。

    相同工具和相同JSON参数即使键顺序不同，
    也会得到相同SHA-256。
    """

    if not isinstance(tool_call, ToolCall):
        raise TypeError(
            "tool_call必须是ToolCall"
        )

    try:
        # sort_keys=True固定字典键顺序。
        #
        # separators去掉不影响JSON含义的空格，
        # 避免格式差异产生不同摘要。
        canonical_arguments = json.dumps(
            tool_call.arguments,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    except TypeError as exc:
        # OpenAI Tool Calling参数应该是JSON数据。
        #
        # 不能使用repr()兜底，否则Python对象的
        # 临时表示可能造成不稳定或泄漏数据。
        raise TypeError(
            "tool_call.arguments必须是"
            "JSON可序列化数据"
        ) from exc

    signature_payload = (
        tool_call.tool_name
        + "\n"
        + canonical_arguments
    ).encode("utf-8")

    return sha256(
        signature_payload
    ).hexdigest()


def _append_new_items(
    existing_items: tuple[str, ...],
    candidate_items: tuple[str, ...],
    /,
) -> tuple[str, ...]:
    """返回候选项中尚未出现在旧状态里的内容。

    保留candidate_items首次出现的顺序，
    同时去除候选项内部的重复内容。
    """

    seen_items = set(existing_items)
    new_items: list[str] = []

    for item in candidate_items:
        if item in seen_items:
            continue

        seen_items.add(item)
        new_items.append(item)

    return tuple(new_items)


def _extract_success_information(
    interaction: AgentToolInteraction,
    /,
) -> tuple[
    tuple[AgentProgressSourceId, ...],
    tuple[AgentProgressEvidenceId, ...],
]:
    """从一条成功工具交互中提取来源和证据标识。

    ToolExecutor已经使用工具的output_model校验过output，
    这里再次还原成对应Pydantic模型，避免直接读取任意dict。

    返回值的第一项是来源标识，第二项是证据标识。
    """

    result = interaction.result

    if result.status != "success":
        return (), ()

    # ToolExecutionResult契约保证success一定包含output。
    #
    # 这里保留显式检查，防止未来调用链绕过数据契约。
    if result.output is None:
        raise ValueError(
            "成功工具结果必须包含output"
        )

    tool_name = interaction.tool_call.tool_name

    if tool_name == "search_knowledge":
        output = (
            SearchKnowledgeToolOutput
            .model_validate(result.output)
        )

        # document_id描述来源文档；
        # chunk_id描述文档中支持事实的具体证据。
        source_ids = tuple(
            f"knowledge:{citation.document_id}"
            for citation in output.citations
        )
        evidence_ids = tuple(
            citation.chunk_id
            for citation in output.citations
        )

        return source_ids, evidence_ids

    if tool_name == "analyze_robot_image":
        output = VisionObservation.model_validate(
            result.output
        )

        # 图片SHA-256标识来源图片。
        source_id = (
            "vision:"
            + output.source_image_sha256
        )

        # 同一张图片可以针对不同analysis_goal产生
        # 不同观察，因此证据标识还要包含call_id。
        evidence_id = (
            source_id
            + ":"
            + interaction.tool_call.call_id
        )

        return (
            (source_id,),
            (evidence_id,),
        )

    if tool_name == "get_robot_telemetry":
        output = (
            RobotTelemetryToolOutput
            .model_validate(result.output)
        )

        observed_at = (
            output.observed_at.isoformat()
        )

        # 来源表示哪一个模拟遥测Store中的机器人。
        source_id = (
            "telemetry:"
            + output.source
            + ":"
            + output.robot_id
        )

        # 证据加入观测时间，区分同一机器人
        # 在不同时间产生的多份快照。
        evidence_id = (
            "telemetry:"
            + output.robot_id
            + ":"
            + observed_at
        )

        return (
            (source_id,),
            (evidence_id,),
        )

    if tool_name == "draft_test_case":
        # model_validate确认草案确实满足：
        #
        # - draft_only=True；
        # - requires_human_approval=True；
        # - 证据引用唯一；
        # - 步骤编号连续。
        DraftTestCaseToolOutput.model_validate(
            result.output
        )

        # 测试草案是由已有证据生成的派生结果，
        # 不是新的事实证据。
        #
        # 因此只记录草案来源，不把它的引用再次
        # 计入covered_evidence_ids。
        source_id = (
            "test_draft:"
            + interaction.tool_call.call_id
        )

        return (source_id,), ()

    if tool_name == "get_current_time":
        output = (
            GetCurrentTimeToolOutput
            .model_validate(result.output)
        )

        current_time = (
            output.current_time.isoformat()
        )

        source_id = "system_clock"
        evidence_id = (
            "system_clock:"
            + current_time
        )

        return (
            (source_id,),
            (evidence_id,),
        )

    # AgentReadOnlyToolName当前只有上述五项。
    #
    # 如果未来增加工具但没有为Reducer实现来源提取，
    # 必须显式失败，不能把未知输出默认为可信证据。
    raise ValueError(
        "没有为当前工具实现进度信息提取"
    )


class AgentProgressReducer:
    """根据策略和工具交互生成不可变AgentProgress。"""

    def create_initial_progress(
        self,
        policy: AgentToolPolicyDecision,
        /,
    ) -> AgentProgress:
        """根据继续执行的工具策略创建初始进度。

        提前拒答和转人工审核的策略不会进入Runner，
        因而不应该创建collecting状态。
        """

        if not isinstance(
            policy,
            AgentToolPolicyDecision,
        ):
            raise TypeError(
                "policy必须是AgentToolPolicyDecision"
            )

        if (
            policy.disposition
            != "continue_to_planner"
        ):
            raise ValueError(
                "只有continue_to_planner策略"
                "可以创建初始AgentProgress"
            )

        return AgentProgress(
            state="collecting",
            required_capabilities=(
                policy.required_capabilities
            ),
            allowed_next_tool_names=(
                policy.allowed_tool_names
            ),
        )

    def validate_tool_call(
        self,
        progress: AgentProgress,
        tool_call: ToolCall,
        /,
    ) -> AgentToolCallSignature:
        """在执行工具前检查当前进度是否允许该调用。

        校验成功时返回工具调用摘要，
        供执行后的AgentToolProgressRecord复用。
        """

        if not isinstance(
            progress,
            AgentProgress,
        ):
            raise TypeError(
                "progress必须是AgentProgress"
            )

        if not isinstance(
            tool_call,
            ToolCall,
        ):
            raise TypeError(
                "tool_call必须是ToolCall"
            )

        if progress.state != "collecting":
            raise AgentProgressTransitionError(
                reason="progress_not_collecting",
                message=(
                    "当前Agent进度不允许继续调用工具"
                ),
            )

        if (
            tool_call.tool_name
            not in progress.allowed_next_tool_names
        ):
            raise AgentProgressTransitionError(
                reason="tool_not_allowed",
                message=(
                    "工具不在当前进度允许范围内"
                ),
            )

        signature = (
            _canonical_tool_call_signature(
                tool_call
            )
        )

        previous_call_ids = {
            record.call_id
            for record in progress.tool_records
        }
        previous_signatures = {
            record.call_signature
            for record in progress.tool_records
        }

        if (
            tool_call.call_id in previous_call_ids
            or signature in previous_signatures
        ):
            raise AgentProgressTransitionError(
                reason="duplicate_tool_call",
                message=(
                    "工具调用标识或相同工具参数"
                    "已经尝试过"
                ),
            )

        return signature

    def record_interaction(
        self,
        progress: AgentProgress,
        interaction: AgentToolInteraction,
        /,
    ) -> AgentProgress:
        """把一次已完成工具交互归纳成新的进度快照。

        旧progress不会被修改。
        """

        if not isinstance(
            progress,
            AgentProgress,
        ):
            raise TypeError(
                "progress必须是AgentProgress"
            )

        if not isinstance(
            interaction,
            AgentToolInteraction,
        ):
            raise TypeError(
                "interaction必须是AgentToolInteraction"
            )

        expected_step_number = (
            len(progress.tool_records) + 1
        )

        if (
            interaction.step_number
            != expected_step_number
        ):
            raise AgentProgressTransitionError(
                reason="step_number_mismatch",
                message=(
                    "工具交互步骤与当前进度不连续"
                ),
            )

        # 执行前和记录时都检查一次。
        #
        # Runner集成后，第一次检查发生在Executor之前；
        # 这里的第二次检查防止其他代码绕过正常调用链，
        # 直接把非法Interaction写入Progress。
        call_signature = (
            self.validate_tool_call(
                progress,
                interaction.tool_call,
            )
        )

        candidate_source_ids, (
            candidate_evidence_ids
        ) = _extract_success_information(
            interaction
        )

        new_source_ids = _append_new_items(
            progress.confirmed_source_ids,
            candidate_source_ids,
        )
        new_evidence_ids = _append_new_items(
            progress.covered_evidence_ids,
            candidate_evidence_ids,
        )

        produced_new_information = bool(
            new_source_ids
            or new_evidence_ids
        )

        record = AgentToolProgressRecord(
            step_number=(
                interaction.step_number
            ),
            call_id=(
                interaction.tool_call.call_id
            ),
            tool_name=(
                interaction.tool_call.tool_name
            ),
            call_signature=call_signature,
            status=interaction.result.status,
            produced_new_information=(
                produced_new_information
            ),
            new_source_ids=new_source_ids,
            new_evidence_ids=new_evidence_ids,
            error_code=(
                interaction.result.error_code
            ),
        )

        completed_capability_set = set(
            progress.completed_capabilities
        )

        # 只有通过输出契约校验的success结果
        # 才能完成对应能力。
        if interaction.result.status == "success":
            completed_capability_set.add(
                TOOL_TO_CAPABILITY[
                    interaction.tool_call.tool_name
                ]
            )

        # 按required_capabilities原始顺序输出，
        # 不让工具调用顺序改变报告字段顺序。
        completed_capabilities = tuple(
            capability
            for capability
            in progress.required_capabilities
            if capability
            in completed_capability_set
        )

        all_capabilities_completed = (
            set(completed_capabilities)
            == set(
                progress.required_capabilities
            )
        )

        if all_capabilities_completed:
            # 工具能力已经齐全。
            #
            # 下一轮Planner只能生成finish决定，
            # 不能继续调用其他工具。
            next_state = "ready_to_finish"
            allowed_next_tool_names = ()

        else:
            next_state = "collecting"

            if interaction.result.status == "success":
                # 当前工具已经成功完成对应能力，
                # 后续不再暴露该工具。
                allowed_next_tool_names = tuple(
                    tool_name
                    for tool_name
                    in progress.allowed_next_tool_names
                    if tool_name
                    != interaction.tool_call.tool_name
                )
            else:
                # 一次empty、timeout或普通error不直接移除工具。
                #
                # Planner仍可使用不同参数重试；
                # 完全相同的调用会由call_signature阻止。
                allowed_next_tool_names = (
                    progress.allowed_next_tool_names
                )

        # model_copy(update=...)默认不会重新验证更新字段。
        #
        # 因此不能在安全状态模型中使用它完成转换。
        # 这里先取得普通Python数据，再通过model_validate()
        # 重新执行AgentProgress的全部字段和跨字段校验。
        progress_data = progress.model_dump(
            mode="python"
        )

        progress_data.update({
            "state": next_state,
            "completed_capabilities": (
                completed_capabilities
            ),
            "tool_records": (
                progress.tool_records
                + (record,)
            ),
            "confirmed_source_ids": (
                progress.confirmed_source_ids
                + new_source_ids
            ),
            "covered_evidence_ids": (
                progress.covered_evidence_ids
                + new_evidence_ids
            ),
            "allowed_next_tool_names": (
                allowed_next_tool_names
            ),
        })

        return AgentProgress.model_validate(
            progress_data
        )

    def finalize_planner_decision(
        self,
        progress: AgentProgress,
        /,
        *,
        finish_reason: AgentPlannerFinishReason,
        missing_information: tuple[
            AgentMissingInformation,
            ...,
        ] = (),
    ) -> AgentProgress:
        """把Planner的finish决定转换成证据驱动终止状态。

        Planner的finish_reason只是请求，不是最终权限。
        Python会根据completed_capabilities决定是否真的完成。
        """

        _validate_active_progress(progress)

        if finish_reason not in {
            "task_completed",
            "insufficient_information",
            "human_review_required",
        }:
            raise ValueError(
                "finish_reason不是受支持的"
                "AgentPlannerFinishReason"
            )

        if not isinstance(
            missing_information,
            tuple,
        ):
            raise TypeError(
                "missing_information必须是tuple"
            )

        incomplete_messages = (
            _incomplete_capability_messages(
                progress
            )
        )

        if finish_reason == "task_completed":
            if progress.state == "ready_to_finish":
                return _build_updated_progress(
                    progress,
                    state="completed",
                    missing_information=(),
                    allowed_next_tool_names=(),
                    stop_reason=(
                        "sufficient_evidence"
                    ),
                )

            # Planner在能力尚未收齐时声称完成。
            # Python保留已有真实进度，但拒绝completed声明。
            effective_missing = (
                _merge_unique_items(
                    missing_information,
                    incomplete_messages,
                )
            )

            if progress.completed_capabilities:
                return _build_updated_progress(
                    progress,
                    state="partial",
                    missing_information=(
                        effective_missing
                    ),
                    allowed_next_tool_names=(),
                    stop_reason="partial_evidence",
                )

            return _build_updated_progress(
                progress,
                state="abstained",
                missing_information=(
                    effective_missing
                ),
                allowed_next_tool_names=(),
                stop_reason=(
                    "insufficient_information"
                ),
            )

        if (
            finish_reason
            == "insufficient_information"
        ):
            effective_missing = (
                _merge_unique_items(
                    missing_information,
                    incomplete_messages,
                )
            )

            # 所有能力都完成，但Planner仍声明不足时，
            # 必须保留一个明确的待复核事项。
            if not effective_missing:
                effective_missing = (
                    "Planner声明现有信息仍不足，"
                    "需要复核最终诊断草稿",
                )

            if progress.completed_capabilities:
                return _build_updated_progress(
                    progress,
                    state="partial",
                    missing_information=(
                        effective_missing
                    ),
                    allowed_next_tool_names=(),
                    stop_reason="partial_evidence",
                )

            return _build_updated_progress(
                progress,
                state="abstained",
                missing_information=(
                    effective_missing
                ),
                allowed_next_tool_names=(),
                stop_reason=(
                    "insufficient_information"
                ),
            )

        # 只剩human_review_required。
        effective_missing = _merge_unique_items(
            missing_information,
            incomplete_messages,
        )

        if not effective_missing:
            effective_missing = (
                "当前结果需要具备权限和资质的人员复核",
            )

        return _build_updated_progress(
            progress,
            state="human_review_required",
            missing_information=effective_missing,
            allowed_next_tool_names=(),
            stop_reason="human_review_required",
        )

    def record_conflicts(
        self,
        progress: AgentProgress,
        /,
        *,
        conflicts: tuple[
            AgentEvidenceConflict,
            ...,
        ],
        missing_information: tuple[
            AgentMissingInformation,
            ...,
        ] = (),
    ) -> AgentProgress:
        """将已确认来源冲突转换成人工审核状态。"""

        _validate_active_progress(progress)

        if not isinstance(conflicts, tuple):
            raise TypeError(
                "conflicts必须是tuple"
            )

        if not conflicts:
            raise ValueError(
                "record_conflicts必须包含至少一条冲突"
            )

        if not all(
            isinstance(
                conflict,
                AgentEvidenceConflict,
            )
            for conflict in conflicts
        ):
            raise TypeError(
                "conflicts只能包含"
                "AgentEvidenceConflict"
            )

        if not isinstance(
            missing_information,
            tuple,
        ):
            raise TypeError(
                "missing_information必须是tuple"
            )

        effective_missing = _merge_unique_items(
            missing_information,
            (
                "已确认来源之间存在冲突，"
                "需要人工核对原始证据",
            ),
        )

        return _build_updated_progress(
            progress,
            state="human_review_required",
            evidence_conflicts=conflicts,
            missing_information=effective_missing,
            allowed_next_tool_names=(),
            stop_reason="conflicting_evidence",
        )

    def fail_progress(
        self,
        progress: AgentProgress,
        /,
        *,
        missing_information: tuple[
            AgentMissingInformation,
            ...,
        ],
    ) -> AgentProgress:
        """将不可恢复的运行错误转换成failed状态。"""

        _validate_active_progress(progress)

        if not isinstance(
            missing_information,
            tuple,
        ):
            raise TypeError(
                "missing_information必须是tuple"
            )

        if not missing_information:
            raise ValueError(
                "failed状态必须说明未完成内容"
            )

        return _build_updated_progress(
            progress,
            state="failed",
            missing_information=(
                _merge_unique_items(
                    missing_information
                )
            ),
            allowed_next_tool_names=(),
            stop_reason="runtime_failure",
        )
