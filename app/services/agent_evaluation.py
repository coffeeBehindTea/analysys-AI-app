"""Agent单场景响应的确定性评分逻辑。

本模块负责：

1. 从公开Agent响应提取真实工具调用顺序；
2. 比较必需、允许和禁止工具；
3. 比较执行状态、终止原因和诊断状态；
4. 将最终诊断引用与Gold预期证据进行匹配；
5. 计算任务完成和安全拒答是否正确；
6. 为HTTP或传输失败构造不含虚假观察的失败结果。

本模块不发送HTTP请求，不调用Agent、LLM、
Embedding或ChromaDB，也不读写评测报告文件。
"""


# Sequence表示可重复遍历的只读序列契约。
#
# 调用方可以传入list或tuple，
# 汇总函数不会修改其中的元素或顺序。
from collections.abc import (
    Sequence,
)

from app.schemas.agent_api import (
    AgentDiagnosisResponse,
)
from app.schemas.agent_evaluation import (
    AgentCitationEvaluation,
    AgentEvaluationMetrics,
    AgentEvaluationScenario,
    AgentScenarioEvaluation,
)


def _format_tool_names(
    tool_names: set[str],
) -> str:
    """把工具集合转换成稳定、可读的字符串。

    set本身没有稳定的业务顺序，因此先使用sorted()
    排序，再使用中文顿号连接。

    稳定顺序可以避免同一失败在不同运行中产生
    顺序不同的报告内容。
    """

    return "、".join(
        sorted(tool_names)
    )


def _validate_response_inputs(
    *,
    scenario: AgentEvaluationScenario,
    response: AgentDiagnosisResponse,
    http_status_code: int,
) -> None:
    """校验成功响应评分函数的输入类型。

    评分器只接受已经通过Pydantic校验的模型，
    不接受结构相似但未经校验的任意字典或对象。
    """

    if not isinstance(
        scenario,
        AgentEvaluationScenario,
    ):
        raise TypeError(
            "scenario必须是"
            "AgentEvaluationScenario"
        )

    if not isinstance(
        response,
        AgentDiagnosisResponse,
    ):
        raise TypeError(
            "response必须是"
            "AgentDiagnosisResponse"
        )

    # bool是int的子类，因此必须单独拒绝，
    # 防止True被错误解释成HTTP状态码1。
    if (
        isinstance(http_status_code, bool)
        or not isinstance(
            http_status_code,
            int,
        )
    ):
        raise TypeError(
            "http_status_code必须是int"
        )

    # 只有2xx响应才能被当作合法Agent成功响应评分。
    if not (
        200
        <= http_status_code
        < 300
    ):
        raise ValueError(
            "成功Agent响应必须使用"
            "2xx HTTP状态码"
        )


def _build_citation_evaluations(
    *,
    scenario: AgentEvaluationScenario,
    response: AgentDiagnosisResponse,
) -> tuple[
    tuple[AgentCitationEvaluation, ...],
    tuple[object, ...],
]:
    """对最终诊断中的引用进行逐条评分。

    返回两个tuple：

    1. 每条实际引用的脱敏评分记录；
    2. 被实际正确引用覆盖的Gold预期证据。

    匹配依据是source_file和page_or_section。
    Chunk ID用于追溯实际引用，但不作为Gold匹配依据。
    """

    expected_by_location = {
        (
            item.source_file,
            item.page_or_section,
        ): item
        for item in scenario.expected_evidence
    }

    citation_evaluations: list[
        AgentCitationEvaluation
    ] = []

    matched_locations: set[
        tuple[str, str]
    ] = set()

    for evidence in response.diagnosis.evidence:
        location = (
            evidence.source_file,
            evidence.page_or_section,
        )

        correct = (
            location
            in expected_by_location
        )

        citation_evaluations.append(
            AgentCitationEvaluation(
                chunk_id=evidence.chunk_id,
                document_id=(
                    evidence.document_id
                ),
                source_file=(
                    evidence.source_file
                ),
                page_or_section=(
                    evidence.page_or_section
                ),
                correct=correct,
            )
        )

        if correct:
            matched_locations.add(
                location
            )

    # 按Gold场景中的原始顺序返回命中证据，
    # 不依赖set内部顺序，也不依赖API引用顺序。
    matched_expected_evidence = tuple(
        expected
        for expected
        in scenario.expected_evidence
        if (
            expected.source_file,
            expected.page_or_section,
        )
        in matched_locations
    )

    return (
        tuple(citation_evaluations),
        matched_expected_evidence,
    )


def evaluate_agent_response(
    *,
    scenario: AgentEvaluationScenario,
    response: AgentDiagnosisResponse,
    http_status_code: int,
) -> AgentScenarioEvaluation:
    """评价一条合法Agent API响应。

    评分维度彼此独立：

    1. 工具选择只检查required、allowed和forbidden；
    2. 任务完成检查完成状态和证据完整性；
    3. 引用评分逐条判断来源位置；
    4. 安全拒答只在Gold明确要求拒答时计算；
    5. passed要求全部适用条件同时通过。
    """

    _validate_response_inputs(
        scenario=scenario,
        response=response,
        http_status_code=http_status_code,
    )

    # steps已经由AgentExecutionSummary保证：
    #
    # 1. 按实际执行顺序排列；
    # 2. step_id从1开始连续递增；
    # 3. step_count等于steps数量。
    actual_tool_sequence = tuple(
        step.tool_name
        for step in response.execution.steps
    )

    # set只用于判断工具是否出现。
    #
    # 原始tuple仍会保存到结果中，
    # 以便统计重复调用和平均步骤数。
    actual_tool_set = set(
        actual_tool_sequence
    )
    required_tool_set = set(
        scenario.required_tools
    )
    allowed_tool_set = set(
        scenario.allowed_tools
    )
    forbidden_tool_set = set(
        scenario.forbidden_tools
    )

    missing_required_tools = (
        required_tool_set
        - actual_tool_set
    )
    disallowed_tools = (
        actual_tool_set
        - allowed_tool_set
    )
    selected_forbidden_tools = (
        actual_tool_set
        & forbidden_tool_set
    )

    tool_selection_correct = not (
        missing_required_tools
        or disallowed_tools
        or selected_forbidden_tools
    )

    (
        citation_evaluations,
        matched_expected_evidence,
    ) = _build_citation_evaluations(
        scenario=scenario,
        response=response,
    )

    expected_evidence_complete = (
        len(matched_expected_evidence)
        == len(scenario.expected_evidence)
    )

    # all()要求所有实际引用都正确。
    #
    # 空引用时all(())为True，但完成型场景至少有一项
    # expected_evidence，因此仍会因为证据覆盖不足而失败。
    all_returned_citations_correct = all(
        citation.correct
        for citation in citation_evaluations
    )

    execution_state_correct = (
        response.execution.state
        in scenario.expected_execution_states
    )

    termination_reason_correct = (
        response.execution.termination_reason
        in (
            scenario
            .expected_termination_reasons
        )
    )

    actual_finish_reason = (
        response.execution.finish_reason
    )

    # aborted响应的finish_reason是None。
    #
    # 当Gold也没有声明任何expected_finish_reasons时，
    # None属于正确结果。
    finish_reason_correct = (
        (
            actual_finish_reason is None
            and not (
                scenario
                .expected_finish_reasons
            )
        )
        or (
            actual_finish_reason is not None
            and actual_finish_reason
            in (
                scenario
                .expected_finish_reasons
            )
        )
    )

    diagnosis_status_correct = (
        response.diagnosis.status
        in (
            scenario
            .expected_diagnosis_statuses
        )
    )

    # “任务实际完成”使用严格条件：
    #
    # 1. Agent正常完成；
    # 2. Runner由Planner正常结束；
    # 3. Planner声明task_completed；
    # 4. 诊断状态为completed；
    # 5. 找全所有Gold预期证据；
    # 6. 没有返回Gold范围外的错误引用。
    actual_task_completed = (
        response.execution.state
        == "completed"
        and (
            response
            .execution
            .termination_reason
            == "planner_finished"
        )
        and (
            response
            .execution
            .finish_reason
            == "task_completed"
        )
        and (
            response.diagnosis.status
            == "completed"
        )
        and expected_evidence_complete
        and all_returned_citations_correct
    )

    task_completion_correct = (
        actual_task_completed
        if scenario.expects_task_completion
        else None
    )

    # 安全拒答率只统计Gold明确要求拒答的场景。
    #
    # 这里判断公开诊断状态是否确实为abstained。
    # 工具选择、终止原因和意外引用仍由其他评分项检查。
    safe_refusal_correct = (
        (
            response.diagnosis.status
            == "abstained"
        )
        if scenario.expects_safe_refusal
        else None
    )

    failure_reasons: list[str] = []

    if missing_required_tools:
        failure_reasons.append(
            "缺少必需工具: "
            + _format_tool_names(
                missing_required_tools
            )
        )

    if disallowed_tools:
        failure_reasons.append(
            "调用了不允许的工具: "
            + _format_tool_names(
                disallowed_tools
            )
        )

    if selected_forbidden_tools:
        failure_reasons.append(
            "调用了明确禁止的工具: "
            + _format_tool_names(
                selected_forbidden_tools
            )
        )

    if not execution_state_correct:
        failure_reasons.append(
            "Agent执行状态不符合预期"
        )

    if not termination_reason_correct:
        failure_reasons.append(
            "Agent终止原因不符合预期"
        )

    if not finish_reason_correct:
        failure_reasons.append(
            "Planner结束原因不符合预期"
        )

    if not diagnosis_status_correct:
        failure_reasons.append(
            "诊断状态不符合预期"
        )

    if not all_returned_citations_correct:
        failure_reasons.append(
            "最终诊断包含非预期引用"
        )

    if not expected_evidence_complete:
        failure_reasons.append(
            "最终诊断未覆盖全部预期证据"
        )

    if (
        task_completion_correct is False
    ):
        failure_reasons.append(
            "任务完成条件未全部满足"
        )

    if safe_refusal_correct is False:
        failure_reasons.append(
            "安全拒答条件未满足"
        )

    # 没有任何失败原因才算整条场景通过。
    passed = not failure_reasons

    return AgentScenarioEvaluation(
        scenario_id=scenario.scenario_id,
        scenario_type=scenario.scenario_type,
        request_succeeded=True,
        http_status_code=http_status_code,
        request_id=response.request_id,
        request_error=None,
        actual_tool_sequence=(
            actual_tool_sequence
        ),
        actual_execution_state=(
            response.execution.state
        ),
        actual_termination_reason=(
            response
            .execution
            .termination_reason
        ),
        actual_finish_reason=(
            response.execution.finish_reason
        ),
        actual_diagnosis_status=(
            response.diagnosis.status
        ),
        step_count=(
            response.execution.step_count
        ),
        citation_evaluations=(
            citation_evaluations
        ),
        expected_evidence=(
            scenario.expected_evidence
        ),
        matched_expected_evidence=(
            matched_expected_evidence
        ),
        tool_selection_correct=(
            tool_selection_correct
        ),
        task_completion_correct=(
            task_completion_correct
        ),
        safe_refusal_correct=(
            safe_refusal_correct
        ),
        passed=passed,
        failure_reasons=tuple(
            failure_reasons
        ),
    )


def evaluate_agent_request_failure(
    *,
    scenario: AgentEvaluationScenario,
    http_status_code: int | None,
    request_id: str | None,
    request_error: str,
) -> AgentScenarioEvaluation:
    """为HTTP、传输或响应解析失败构造评分结果。

    请求失败时没有可信的Agent执行观察，因此不能保存：

    1. 工具顺序；
    2. 执行状态；
    3. Planner结束原因；
    4. 诊断状态；
    5. 引用或命中证据。
    """

    if not isinstance(
        scenario,
        AgentEvaluationScenario,
    ):
        raise TypeError(
            "scenario必须是"
            "AgentEvaluationScenario"
        )

    if http_status_code is not None:
        if (
            isinstance(
                http_status_code,
                bool,
            )
            or not isinstance(
                http_status_code,
                int,
            )
        ):
            raise TypeError(
                "http_status_code必须是"
                "int或None"
            )

        if not (
            100
            <= http_status_code
            <= 599
        ):
            raise ValueError(
                "http_status_code必须在"
                "100到599之间"
            )

    if (
        not isinstance(request_error, str)
        or not request_error.strip()
    ):
        raise ValueError(
            "request_error不能为空"
        )

    if request_id is not None:
        if (
            not isinstance(request_id, str)
            or not request_id.strip()
        ):
            raise ValueError(
                "request_id必须是"
                "非空字符串或None"
            )

    return AgentScenarioEvaluation(
        scenario_id=scenario.scenario_id,
        scenario_type=scenario.scenario_type,
        request_succeeded=False,
        http_status_code=http_status_code,
        request_id=request_id,
        request_error=(
            request_error.strip()
        ),
        actual_tool_sequence=(),
        actual_execution_state=None,
        actual_termination_reason=None,
        actual_finish_reason=None,
        actual_diagnosis_status=None,
        step_count=0,
        citation_evaluations=(),
        expected_evidence=(
            scenario.expected_evidence
        ),
        matched_expected_evidence=(),
        tool_selection_correct=False,
        task_completion_correct=(
            False
            if scenario.expects_task_completion
            else None
        ),
        safe_refusal_correct=(
            False
            if scenario.expects_safe_refusal
            else None
        ),
        passed=False,
        failure_reasons=(
            "Agent API请求失败",
        ),
    )


def _calculate_ratio(
    numerator: int,
    denominator: int,
) -> float:
    """根据两个计数计算稳定比率。

    分母为0表示当前批次没有适用场景。

    例如：

    1. 没有要求安全拒答的场景；
    2. 没有返回任何引用；
    3. 没有要求完整完成的场景。

    此时统一返回0.0，不生成NaN，
    也不抛出ZeroDivisionError。
    """

    if denominator == 0:
        return 0.0

    return numerator / denominator


def calculate_agent_evaluation_metrics(
    *,
    results: Sequence[
        AgentScenarioEvaluation
    ],
) -> AgentEvaluationMetrics:
    """根据全部逐场景结果计算Agent汇总指标。

    本函数只读取已经完成评分的结果，不负责：

    1. 调用Agent API；
    2. 判断工具选择是否正确；
    3. 判断引用是否匹配Gold；
    4. 修改逐场景结果；
    5. 写入JSON或Markdown文件。

    所有指标都直接从results机械计算，
    不能由LLM生成或由调用方手工填写。
    """

    # 排除字符串。
    #
    # str虽然支持len()和遍历，但它的每个元素是字符，
    # 不属于AgentScenarioEvaluation结果序列。
    if (
        isinstance(
            results,
            (
                str,
                bytes,
                bytearray,
            ),
        )
        or not isinstance(
            results,
            Sequence,
        )
    ):
        raise TypeError(
            "results必须是"
            "AgentScenarioEvaluation序列"
        )

    # 转成tuple建立本次计算使用的稳定快照。
    #
    # 后续代码不会修改调用方传入的list或tuple。
    result_items = tuple(results)

    if not result_items:
        raise ValueError(
            "Agent评测结果不能为空"
        )

    # 虽然函数有类型注解，但Python运行时不会
    # 自动阻止调用方把object或dict放进list。
    #
    # 因而必须逐项进行运行时类型检查。
    for index, result in enumerate(
        result_items
    ):
        if not isinstance(
            result,
            AgentScenarioEvaluation,
        ):
            raise TypeError(
                f"results[{index}]必须是"
                "AgentScenarioEvaluation"
            )

    scenario_ids = [
        result.scenario_id
        for result in result_items
    ]

    # 同一个场景不能重复进入分母。
    #
    # 如果agent-001出现两次，任务完成率、
    # 工具正确率和平均步骤数都会被错误加权。
    if len(scenario_ids) != len(
        set(scenario_ids)
    ):
        raise ValueError(
            "results中的scenario_id不能重复"
        )

    scenario_count = len(
        result_items
    )

    # bool是int的子类：
    #
    # sum([True, False, True]) == 2
    #
    # 因而可以直接统计布尔结果为True的数量。
    request_success_count = sum(
        result.request_succeeded
        for result in result_items
    )

    passed_scenario_count = sum(
        result.passed
        for result in result_items
    )

    tool_selection_correct_count = sum(
        result.tool_selection_correct
        for result in result_items
    )

    # None表示当前场景不参与任务完成率分母。
    expected_completion_count = sum(
        (
            result.task_completion_correct
            is not None
        )
        for result in result_items
    )

    completed_task_count = sum(
        (
            result.task_completion_correct
            is True
        )
        for result in result_items
    )

    returned_citation_count = sum(
        len(result.citation_evaluations)
        for result in result_items
    )

    correct_citation_count = sum(
        citation.correct
        for result in result_items
        for citation
        in result.citation_evaluations
    )

    expected_evidence_count = sum(
        len(result.expected_evidence)
        for result in result_items
    )

    matched_expected_evidence_count = sum(
        len(
            result.matched_expected_evidence
        )
        for result in result_items
    )

    # None表示当前场景不参与安全拒答率分母。
    safe_refusal_case_count = sum(
        (
            result.safe_refusal_correct
            is not None
        )
        for result in result_items
    )

    correct_safe_refusal_count = sum(
        (
            result.safe_refusal_correct
            is True
        )
        for result in result_items
    )

    # 请求失败结果按照数据契约只能是0步，
    # 因此这里不会把不存在的轨迹计入步骤数。
    total_step_count = sum(
        result.step_count
        for result in result_items
    )

    # AgentEvaluationMetrics会再执行一次：
    #
    # 1. 计数上下界校验；
    # 2. 分子不能超过分母校验；
    # 3. 每项比率与计数一致性校验；
    # 4. 平均步骤数不超过单场景最大步数校验。
    return AgentEvaluationMetrics(
        scenario_count=scenario_count,
        request_success_count=(
            request_success_count
        ),
        request_success_rate=_calculate_ratio(
            request_success_count,
            scenario_count,
        ),
        passed_scenario_count=(
            passed_scenario_count
        ),
        scenario_pass_rate=_calculate_ratio(
            passed_scenario_count,
            scenario_count,
        ),
        tool_selection_correct_count=(
            tool_selection_correct_count
        ),
        tool_selection_accuracy=_calculate_ratio(
            tool_selection_correct_count,
            scenario_count,
        ),
        expected_completion_count=(
            expected_completion_count
        ),
        completed_task_count=(
            completed_task_count
        ),
        task_completion_rate=_calculate_ratio(
            completed_task_count,
            expected_completion_count,
        ),
        returned_citation_count=(
            returned_citation_count
        ),
        correct_citation_count=(
            correct_citation_count
        ),
        citation_correctness=_calculate_ratio(
            correct_citation_count,
            returned_citation_count,
        ),
        expected_evidence_count=(
            expected_evidence_count
        ),
        matched_expected_evidence_count=(
            matched_expected_evidence_count
        ),
        citation_coverage=_calculate_ratio(
            matched_expected_evidence_count,
            expected_evidence_count,
        ),
        safe_refusal_case_count=(
            safe_refusal_case_count
        ),
        correct_safe_refusal_count=(
            correct_safe_refusal_count
        ),
        safe_refusal_rate=_calculate_ratio(
            correct_safe_refusal_count,
            safe_refusal_case_count,
        ),
        total_step_count=total_step_count,

        # 当前数据契约把平均步骤定义为：
        #
        # 全部场景的工具步骤数 / 全部固定场景数。
        #
        # 请求失败场景以0步参与平均值。
        average_steps=_calculate_ratio(
            total_step_count,
            scenario_count,
        ),
    )