"""Agent证据驱动进度状态的数据契约测试。

本文件验证：

1. 工具进度记录只能声明与执行状态一致的新信息；
2. AgentProgress能够表达收集、完成、部分、拒答、人工审核和失败；
3. 完成能力不能超出请求最初要求的能力；
4. 汇总来源和证据必须来自真实工具记录；
5. 冲突只能引用已经确认的来源和证据；
6. 下一步工具不能扩大请求级最小权限；
7. 终止状态不能继续暴露工具。

测试不调用LLM、网络、Chroma、图片服务或任何真实工具。
"""

import pytest

from pydantic import (
    ValidationError,
)

from app.schemas.agent_progress import (
    AgentEvidenceConflict,
    AgentProgress,
    AgentToolProgressRecord,
)


# call_signature只保存工具名称和规范化参数的SHA-256摘要。
# 使用固定64位十六进制字符串，使测试结果完全可复现。
CALL_SIGNATURE_A = "a" * 64
CALL_SIGNATURE_B = "b" * 64


# 来源标识用于区分知识库文档来源和遥测来源。
# 它们是测试数据，不对应真实生产系统。
KNOWLEDGE_SOURCE_ID = (
    "knowledge:demo-document:000001"
)
TELEMETRY_SOURCE_ID = (
    "telemetry:robot-001:demo-observation"
)


# 证据标识表示能够支持任务事实的具体证据。
# 来源和证据分开，可以覆盖一个来源产生多条证据的情况。
KNOWLEDGE_EVIDENCE_ID = (
    "demo-document:000001"
)
TELEMETRY_EVIDENCE_ID = (
    "robot-001:demo-observation"
)


def make_knowledge_record(
    *,
    step_number: int = 1,
    produced_new_information: bool = True,
) -> AgentToolProgressRecord:
    """构造一次确定性的成功知识检索进度记录。

    produced_new_information=True时，记录包含一项新来源和
    一项新证据；False时模拟工具成功但只返回已见过内容。
    """

    if produced_new_information:
        source_ids = (
            KNOWLEDGE_SOURCE_ID,
        )
        evidence_ids = (
            KNOWLEDGE_EVIDENCE_ID,
        )
    else:
        source_ids = ()
        evidence_ids = ()

    return AgentToolProgressRecord(
        step_number=step_number,
        call_id=f"call_knowledge_{step_number}",
        tool_name="search_knowledge",
        call_signature=CALL_SIGNATURE_A,
        status="success",
        produced_new_information=(
            produced_new_information
        ),
        new_source_ids=source_ids,
        new_evidence_ids=evidence_ids,
    )


def make_telemetry_record(
    *,
    step_number: int = 2,
) -> AgentToolProgressRecord:
    """构造一次产生新遥测信息的成功工具记录。"""

    return AgentToolProgressRecord(
        step_number=step_number,
        call_id=f"call_telemetry_{step_number}",
        tool_name="get_robot_telemetry",
        call_signature=CALL_SIGNATURE_B,
        status="success",
        produced_new_information=True,
        new_source_ids=(
            TELEMETRY_SOURCE_ID,
        ),
        new_evidence_ids=(
            TELEMETRY_EVIDENCE_ID,
        ),
    )


def test_collecting_progress_accepts_minimum_next_tool() -> None:
    """收集中状态应接受必要能力对应的最小工具范围。

    被测试模块是AgentProgress。
    测试创建一份只需要知识证据、尚未执行工具的初始进度。
    预期模型保持collecting，并只允许search_knowledge。
    """

    progress = AgentProgress(
        required_capabilities=(
            "knowledge_evidence",
        ),
        allowed_next_tool_names=(
            "search_knowledge",
        ),
    )

    assert progress.state == "collecting"
    assert progress.tool_records == ()
    assert progress.allowed_next_tool_names == (
        "search_knowledge",
    )
    assert progress.stop_reason is None


def test_success_record_can_add_new_information() -> None:
    """成功工具记录可以声明首次确认的来源和证据。

    被测试模块是AgentToolProgressRecord。
    预期辅助函数产生success记录，并且新信息标记为True。
    """

    record = make_knowledge_record()

    assert record.status == "success"
    assert record.produced_new_information is True
    assert record.new_source_ids == (
        KNOWLEDGE_SOURCE_ID,
    )
    assert record.new_evidence_ids == (
        KNOWLEDGE_EVIDENCE_ID,
    )
    assert record.error_code is None


def test_success_record_can_report_no_new_information() -> None:
    """工具成功但只返回旧内容时应保留无新增信息状态。

    被测试模块是AgentToolProgressRecord。
    该状态供后续Reducer识别无新证据查询，而不是把它误报
    为工具异常。预期status仍为success，但新增字段为空。
    """

    record = make_knowledge_record(
        produced_new_information=False,
    )

    assert record.status == "success"
    assert record.produced_new_information is False
    assert record.new_source_ids == ()
    assert record.new_evidence_ids == ()


def test_new_information_flag_must_match_new_items() -> None:
    """新增信息布尔值不能与实际新增标识矛盾。

    被测试模块是AgentToolProgressRecord的模型级校验器。
    测试声明produced_new_information=False，却同时提供来源。
    预期Pydantic抛出ValidationError。
    """

    with pytest.raises(
        ValidationError,
        match="produced_new_information",
    ):
        AgentToolProgressRecord(
            step_number=1,
            call_id="call_mismatch",
            tool_name="search_knowledge",
            call_signature=CALL_SIGNATURE_A,
            status="success",
            produced_new_information=False,
            new_source_ids=(
                KNOWLEDGE_SOURCE_ID,
            ),
        )


def test_non_success_record_cannot_add_information() -> None:
    """空结果、拒绝、超时或错误不能产生可信新信息。

    被测试模块是AgentToolProgressRecord。
    测试模拟empty工具结果却携带新证据，预期契约拒绝它。
    """

    with pytest.raises(
        ValidationError,
        match="非成功工具记录不能产生新信息",
    ):
        AgentToolProgressRecord(
            step_number=1,
            call_id="call_empty_with_evidence",
            tool_name="search_knowledge",
            call_signature=CALL_SIGNATURE_A,
            status="empty",
            produced_new_information=True,
            new_evidence_ids=(
                KNOWLEDGE_EVIDENCE_ID,
            ),
            error_code="empty_result",
        )


def test_non_success_record_requires_error_code() -> None:
    """非成功工具记录必须提供稳定错误码。

    被测试模块是AgentToolProgressRecord。
    预期没有error_code的empty记录不能通过模型校验。
    """

    with pytest.raises(
        ValidationError,
        match="必须包含error_code",
    ):
        AgentToolProgressRecord(
            step_number=1,
            call_id="call_empty_without_code",
            tool_name="search_knowledge",
            call_signature=CALL_SIGNATURE_A,
            status="empty",
            produced_new_information=False,
        )


def test_completed_progress_requires_full_coverage() -> None:
    """全部必要能力完成后才能形成completed状态。

    被测试模块是AgentProgress。
    测试把成功检索记录、汇总来源和汇总证据对应起来。
    预期状态以sufficient_evidence正常完成。
    """

    record = make_knowledge_record()

    progress = AgentProgress(
        state="completed",
        required_capabilities=(
            "knowledge_evidence",
        ),
        completed_capabilities=(
            "knowledge_evidence",
        ),
        tool_records=(record,),
        confirmed_source_ids=(
            KNOWLEDGE_SOURCE_ID,
        ),
        covered_evidence_ids=(
            KNOWLEDGE_EVIDENCE_ID,
        ),
        stop_reason="sufficient_evidence",
    )

    assert progress.state == "completed"
    assert progress.completed_capabilities == (
        "knowledge_evidence",
    )
    assert progress.allowed_next_tool_names == ()


def test_ready_to_finish_requires_all_capabilities() -> None:
    """证据能力齐备后应先进入等待最终草稿状态。

    被测试模块是AgentProgress。
    测试提供完整知识能力和真实工具记录，但尚未生成最终报告。
    预期ready_to_finish合法、无stop_reason并且不再允许工具。
    """

    record = make_knowledge_record()

    progress = AgentProgress(
        state="ready_to_finish",
        required_capabilities=(
            "knowledge_evidence",
        ),
        completed_capabilities=(
            "knowledge_evidence",
        ),
        tool_records=(record,),
        confirmed_source_ids=(
            KNOWLEDGE_SOURCE_ID,
        ),
        covered_evidence_ids=(
            KNOWLEDGE_EVIDENCE_ID,
        ),
    )

    assert progress.state == "ready_to_finish"
    assert progress.allowed_next_tool_names == ()
    assert progress.stop_reason is None


def test_ready_to_finish_rejects_incomplete_capabilities() -> None:
    """尚未满足全部必要能力时不能进入最终草稿阶段。

    被测试模块是AgentProgress。
    测试要求知识和遥测，却只完成知识能力，预期校验失败。
    """

    record = make_knowledge_record()

    with pytest.raises(
        ValidationError,
        match="ready_to_finish必须完成全部必要能力",
    ):
        AgentProgress(
            state="ready_to_finish",
            required_capabilities=(
                "knowledge_evidence",
                "robot_telemetry",
            ),
            completed_capabilities=(
                "knowledge_evidence",
            ),
            tool_records=(record,),
            confirmed_source_ids=(
                KNOWLEDGE_SOURCE_ID,
            ),
            covered_evidence_ids=(
                KNOWLEDGE_EVIDENCE_ID,
            ),
        )


def test_partial_progress_requires_completed_capability_and_gap() -> None:
    """部分结果必须同时包含已完成能力和缺失信息。

    被测试模块是AgentProgress。
    知识证据已经完成，但遥测仍缺失，因此预期允许形成
    partial，而不能伪装成completed。
    """

    record = make_knowledge_record()

    progress = AgentProgress(
        state="partial",
        required_capabilities=(
            "knowledge_evidence",
            "robot_telemetry",
        ),
        completed_capabilities=(
            "knowledge_evidence",
        ),
        tool_records=(record,),
        confirmed_source_ids=(
            KNOWLEDGE_SOURCE_ID,
        ),
        covered_evidence_ids=(
            KNOWLEDGE_EVIDENCE_ID,
        ),
        missing_information=(
            "缺少当前机器人遥测快照",
        ),
        stop_reason="partial_evidence",
    )

    assert progress.state == "partial"
    assert progress.missing_information == (
        "缺少当前机器人遥测快照",
    )


def test_abstained_progress_requires_missing_information() -> None:
    """没有足够证据时应形成带解释的abstained状态。

    被测试模块是AgentProgress。
    测试没有成功工具记录，并明确说明知识库没有证据。
    预期状态合法，但不允许任何下一步工具。
    """

    progress = AgentProgress(
        state="abstained",
        required_capabilities=(
            "knowledge_evidence",
        ),
        missing_information=(
            "知识库没有足够证据",
        ),
        stop_reason="insufficient_information",
    )

    assert progress.state == "abstained"
    assert progress.allowed_next_tool_names == ()


def test_conflicting_sources_require_human_review() -> None:
    """已确认来源互相冲突时应转为人工审核。

    被测试模块是AgentProgress和AgentEvidenceConflict。
    测试构造知识库与遥测两条成功来源，并声明二者冲突。
    预期只能使用human_review_required和conflicting_evidence。
    """

    knowledge_record = make_knowledge_record()
    telemetry_record = make_telemetry_record()

    progress = AgentProgress(
        state="human_review_required",
        required_capabilities=(
            "knowledge_evidence",
            "robot_telemetry",
        ),
        completed_capabilities=(
            "knowledge_evidence",
            "robot_telemetry",
        ),
        tool_records=(
            knowledge_record,
            telemetry_record,
        ),
        confirmed_source_ids=(
            KNOWLEDGE_SOURCE_ID,
            TELEMETRY_SOURCE_ID,
        ),
        covered_evidence_ids=(
            KNOWLEDGE_EVIDENCE_ID,
            TELEMETRY_EVIDENCE_ID,
        ),
        evidence_conflicts=(
            AgentEvidenceConflict(
                description=(
                    "知识库规则与当前遥测状态不一致"
                ),
                source_ids=(
                    KNOWLEDGE_SOURCE_ID,
                    TELEMETRY_SOURCE_ID,
                ),
                evidence_ids=(
                    KNOWLEDGE_EVIDENCE_ID,
                    TELEMETRY_EVIDENCE_ID,
                ),
            ),
        ),
        missing_information=(
            "需要人工核对机器人真实状态",
        ),
        stop_reason="conflicting_evidence",
    )

    assert progress.state == (
        "human_review_required"
    )
    assert len(progress.evidence_conflicts) == 1


def test_failed_progress_requires_runtime_failure_reason() -> None:
    """不可恢复运行错误应形成明确failed状态。

    被测试模块是AgentProgress。
    预期failed必须使用runtime_failure，并说明未完成内容。
    """

    progress = AgentProgress(
        state="failed",
        required_capabilities=(
            "knowledge_evidence",
        ),
        missing_information=(
            "规划服务异常，任务尚未完成",
        ),
        stop_reason="runtime_failure",
    )

    assert progress.state == "failed"
    assert progress.stop_reason == (
        "runtime_failure"
    )


def test_completed_capabilities_must_be_required() -> None:
    """状态机不能凭空完成本次请求不需要的能力。

    被测试模块是AgentProgress。
    测试只要求知识证据，却声称遥测已经完成，预期被拒绝。
    """

    with pytest.raises(
        ValidationError,
        match="completed_capabilities",
    ):
        AgentProgress(
            required_capabilities=(
                "knowledge_evidence",
            ),
            completed_capabilities=(
                "robot_telemetry",
            ),
            allowed_next_tool_names=(
                "search_knowledge",
            ),
        )


def test_completed_state_rejects_incomplete_capabilities() -> None:
    """completed不能遗漏任何必要能力。

    被测试模块是AgentProgress。
    测试要求知识和遥测，却只完成知识能力，预期校验失败。
    """

    record = make_knowledge_record()

    with pytest.raises(
        ValidationError,
        match="完成全部必要能力",
    ):
        AgentProgress(
            state="completed",
            required_capabilities=(
                "knowledge_evidence",
                "robot_telemetry",
            ),
            completed_capabilities=(
                "knowledge_evidence",
            ),
            tool_records=(record,),
            confirmed_source_ids=(
                KNOWLEDGE_SOURCE_ID,
            ),
            covered_evidence_ids=(
                KNOWLEDGE_EVIDENCE_ID,
            ),
            stop_reason="sufficient_evidence",
        )


def test_completed_state_rejects_missing_information() -> None:
    """completed状态不能同时声称仍有信息缺口。

    被测试模块是AgentProgress。
    预期包含missing_information的completed模型被拒绝。
    """

    record = make_knowledge_record()

    with pytest.raises(
        ValidationError,
        match="缺失信息",
    ):
        AgentProgress(
            state="completed",
            required_capabilities=(
                "knowledge_evidence",
            ),
            completed_capabilities=(
                "knowledge_evidence",
            ),
            tool_records=(record,),
            confirmed_source_ids=(
                KNOWLEDGE_SOURCE_ID,
            ),
            covered_evidence_ids=(
                KNOWLEDGE_EVIDENCE_ID,
            ),
            missing_information=(
                "仍缺少必要信息",
            ),
            stop_reason="sufficient_evidence",
        )


def test_partial_state_requires_missing_information() -> None:
    """partial状态必须解释没有完成的内容。

    被测试模块是AgentProgress。
    测试提供已完成能力但不提供信息缺口，预期校验失败。
    """

    record = make_knowledge_record()

    with pytest.raises(
        ValidationError,
        match="partial必须说明缺失信息",
    ):
        AgentProgress(
            state="partial",
            required_capabilities=(
                "knowledge_evidence",
                "robot_telemetry",
            ),
            completed_capabilities=(
                "knowledge_evidence",
            ),
            tool_records=(record,),
            confirmed_source_ids=(
                KNOWLEDGE_SOURCE_ID,
            ),
            covered_evidence_ids=(
                KNOWLEDGE_EVIDENCE_ID,
            ),
            stop_reason="partial_evidence",
        )


def test_terminal_state_cannot_allow_more_tools() -> None:
    """任何终止状态都不能继续暴露工具。

    被测试模块是AgentProgress。
    测试构造abstained状态却仍允许检索，预期契约拒绝。
    """

    with pytest.raises(
        ValidationError,
        match="终止状态不能继续允许工具调用",
    ):
        AgentProgress(
            state="abstained",
            required_capabilities=(
                "knowledge_evidence",
            ),
            missing_information=(
                "知识库没有足够证据",
            ),
            allowed_next_tool_names=(
                "search_knowledge",
            ),
            stop_reason="insufficient_information",
        )


def test_collecting_state_requires_next_tool() -> None:
    """收集中状态不能在没有下一步能力时悬空。

    被测试模块是AgentProgress。
    collecting既未终止又没有工具会形成死状态，预期被拒绝。
    """

    with pytest.raises(
        ValidationError,
        match="至少一个下一步工具",
    ):
        AgentProgress(
            required_capabilities=(
                "knowledge_evidence",
            ),
        )


def test_collecting_state_rejects_fully_completed_capabilities() -> None:
    """全部必要能力完成后不能继续保持collecting。

    被测试模块是AgentProgress。
    测试把知识能力标记为完成但仍保留检索工具，预期模型要求
    调用方改用ready_to_finish，防止继续进行不必要检索。
    """

    with pytest.raises(
        ValidationError,
        match="ready_to_finish",
    ):
        AgentProgress(
            required_capabilities=(
                "knowledge_evidence",
            ),
            completed_capabilities=(
                "knowledge_evidence",
            ),
            allowed_next_tool_names=(
                "search_knowledge",
            ),
        )


def test_collecting_state_requires_every_incomplete_tool() -> None:
    """收集中状态不能遗漏尚未完成能力对应的工具。

    被测试模块是AgentProgress。
    请求需要知识和遥测却只允许知识检索，预期契约拒绝这个
    无法完成遥测目标的死状态。
    """

    with pytest.raises(
        ValidationError,
        match="与未完成能力完全对应",
    ):
        AgentProgress(
            required_capabilities=(
                "knowledge_evidence",
                "robot_telemetry",
            ),
            allowed_next_tool_names=(
                "search_knowledge",
            ),
        )


def test_tool_record_steps_must_be_continuous() -> None:
    """工具记录的步骤编号必须从1开始连续。

    被测试模块是AgentProgress。
    测试把首条工具记录编号设为2，预期模型拒绝该轨迹。
    """

    record = make_knowledge_record(
        step_number=2,
    )

    with pytest.raises(
        ValidationError,
        match="步骤必须从1开始连续",
    ):
        AgentProgress(
            required_capabilities=(
                "knowledge_evidence",
            ),
            tool_records=(record,),
            confirmed_source_ids=(
                KNOWLEDGE_SOURCE_ID,
            ),
            covered_evidence_ids=(
                KNOWLEDGE_EVIDENCE_ID,
            ),
            allowed_next_tool_names=(
                "search_knowledge",
            ),
        )


def test_source_summary_must_match_tool_records() -> None:
    """汇总来源不能包含工具记录没有产生的来源。

    被测试模块是AgentProgress。
    测试伪造额外来源，预期汇总一致性校验拒绝该对象。
    """

    record = make_knowledge_record()

    with pytest.raises(
        ValidationError,
        match="confirmed_source_ids",
    ):
        AgentProgress(
            required_capabilities=(
                "knowledge_evidence",
            ),
            tool_records=(record,),
            confirmed_source_ids=(
                KNOWLEDGE_SOURCE_ID,
                "knowledge:invented-source",
            ),
            covered_evidence_ids=(
                KNOWLEDGE_EVIDENCE_ID,
            ),
            allowed_next_tool_names=(
                "search_knowledge",
            ),
        )


def test_evidence_summary_must_match_tool_records() -> None:
    """汇总证据不能遗漏工具记录实际产生的证据。

    被测试模块是AgentProgress。
    测试保留来源却清空证据汇总，预期模型拒绝不完整摘要。
    """

    record = make_knowledge_record()

    with pytest.raises(
        ValidationError,
        match="covered_evidence_ids",
    ):
        AgentProgress(
            required_capabilities=(
                "knowledge_evidence",
            ),
            tool_records=(record,),
            confirmed_source_ids=(
                KNOWLEDGE_SOURCE_ID,
            ),
            allowed_next_tool_names=(
                "search_knowledge",
            ),
        )


def test_conflict_cannot_reference_unknown_source() -> None:
    """冲突记录不能引用未经工具确认的来源。

    被测试模块是AgentProgress和AgentEvidenceConflict。
    测试让冲突引用一个伪造来源，预期白名单检查失败。
    """

    record = make_knowledge_record()

    with pytest.raises(
        ValidationError,
        match="未确认来源",
    ):
        AgentProgress(
            state="human_review_required",
            required_capabilities=(
                "knowledge_evidence",
            ),
            completed_capabilities=(
                "knowledge_evidence",
            ),
            tool_records=(record,),
            confirmed_source_ids=(
                KNOWLEDGE_SOURCE_ID,
            ),
            covered_evidence_ids=(
                KNOWLEDGE_EVIDENCE_ID,
            ),
            evidence_conflicts=(
                AgentEvidenceConflict(
                    description="存在待确认冲突",
                    source_ids=(
                        KNOWLEDGE_SOURCE_ID,
                        "knowledge:invented-source",
                    ),
                ),
            ),
            missing_information=(
                "需要人工核对冲突来源",
            ),
            stop_reason="conflicting_evidence",
        )


def test_conflict_requires_human_review_state() -> None:
    """存在冲突时不能继续collecting或声称completed。

    被测试模块是AgentProgress。
    测试在collecting状态中加入真实来源冲突，预期被强制拒绝。
    """

    knowledge_record = make_knowledge_record()
    telemetry_record = make_telemetry_record()

    with pytest.raises(
        ValidationError,
        match="必须转人工审核",
    ):
        AgentProgress(
            required_capabilities=(
                "knowledge_evidence",
                "robot_telemetry",
            ),
            tool_records=(
                knowledge_record,
                telemetry_record,
            ),
            confirmed_source_ids=(
                KNOWLEDGE_SOURCE_ID,
                TELEMETRY_SOURCE_ID,
            ),
            covered_evidence_ids=(
                KNOWLEDGE_EVIDENCE_ID,
                TELEMETRY_EVIDENCE_ID,
            ),
            evidence_conflicts=(
                AgentEvidenceConflict(
                    description="两个来源结论不一致",
                    source_ids=(
                        KNOWLEDGE_SOURCE_ID,
                        TELEMETRY_SOURCE_ID,
                    ),
                ),
            ),
            allowed_next_tool_names=(
                "search_knowledge",
                "get_robot_telemetry",
            ),
        )


def test_conflict_stop_reason_requires_conflict_record() -> None:
    """不能只写冲突停止原因而不提供冲突来源。

    被测试模块是AgentProgress。
    预期没有evidence_conflicts的conflicting_evidence状态失败。
    """

    with pytest.raises(
        ValidationError,
        match="必须包含evidence_conflicts",
    ):
        AgentProgress(
            state="human_review_required",
            required_capabilities=(
                "knowledge_evidence",
            ),
            missing_information=(
                "需要人工核对证据",
            ),
            stop_reason="conflicting_evidence",
        )


def test_next_tools_cannot_exceed_required_capabilities() -> None:
    """状态机不能增加请求级工具策略未授予的能力。

    被测试模块是AgentProgress。
    请求只需要知识证据，却暴露遥测工具，预期权限校验失败。
    """

    with pytest.raises(
        ValidationError,
        match="不能超出必要能力",
    ):
        AgentProgress(
            required_capabilities=(
                "knowledge_evidence",
            ),
            allowed_next_tool_names=(
                "get_robot_telemetry",
            ),
        )


def test_next_tools_must_use_stable_order() -> None:
    """下一步工具排列必须稳定，保证轨迹可复现。

    被测试模块是AgentProgress的字段校验器。
    测试把遥测放在知识检索之前，预期因顺序错误被拒绝。
    """

    with pytest.raises(
        ValidationError,
        match="AGENT_READ_ONLY_TOOL_ORDER",
    ):
        AgentProgress(
            required_capabilities=(
                "knowledge_evidence",
                "robot_telemetry",
            ),
            allowed_next_tool_names=(
                "get_robot_telemetry",
                "search_knowledge",
            ),
        )


def test_progress_summary_rejects_duplicate_items() -> None:
    """进度汇总不能通过重复项虚增完成度或证据数量。

    被测试模块是AgentProgress的字段校验器。
    测试重复声明同一必要能力，预期Pydantic拒绝该输入。
    """

    with pytest.raises(
        ValidationError,
        match="不能包含重复项",
    ):
        AgentProgress(
            required_capabilities=(
                "knowledge_evidence",
                "knowledge_evidence",
            ),
            allowed_next_tool_names=(
                "search_knowledge",
            ),
        )
