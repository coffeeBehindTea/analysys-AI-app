"""请求级最小工具策略的离线单元测试。

本文件只调用AgentToolPolicy.decide()的公开接口：

1. 不启动FastAPI；
2. 不调用Planner或LLM；
3. 不执行任何Agent工具；
4. 不解码或识别图片像素；
5. 验证任务目标到最小只读工具集合的确定性映射；
6. 验证缺少必要图片和图片明显无关时的提前拒答。
"""

import pytest

from app.agent.request_tool_policy import (
    AgentToolPolicy,
)
from app.schemas.agent_api import (
    AgentDiagnosisRequest,
)
from app.schemas.vision import (
    VisionImagePayload,
)


# 该字符串满足VisionImagePayload的外部结构契约。
#
# 当前被测策略只读取图片的analysis_goal，
# 不解码image_base64，也不把图片发给Vision Provider。
# 因此测试无需创建真实图片或依赖Pillow。
TEST_IMAGE_BASE64 = "QUFB"


def make_image(
    analysis_goal: str,
    /,
) -> VisionImagePayload:
    """创建只用于策略判断的固定图片载荷。"""

    return VisionImagePayload(
        mime_type="image/png",
        encoding="base64",
        image_base64=TEST_IMAGE_BASE64,
        analysis_goal=analysis_goal,
        detail="auto",
    )


def make_request(
    *,
    task_goal: str | None,
    images: list[VisionImagePayload] | None = None,
) -> AgentDiagnosisRequest:
    """创建包含固定诊断上下文的Agent请求。"""

    return AgentDiagnosisRequest(
        robot_id="robot-001",
        symptom="网络恢复后机器人仍处于暂停状态",
        log_excerpt=(
            "ERR-NET-4001 heartbeat timeout"
        ),
        task_goal=task_goal,
        images=(
            []
            if images is None
            else images
        ),
    )


def test_decide_allows_only_knowledge_for_rule_lookup(
) -> None:
    """纯规则检索只能开放知识库检索工具。"""

    request = make_request(
        task_goal="检索知识库中的网络恢复规则",
    )

    decision = AgentToolPolicy().decide(
        request
    )

    assert decision.disposition == (
        "continue_to_planner"
    )
    assert decision.required_capabilities == (
        "knowledge_evidence",
    )
    assert decision.allowed_tool_names == (
        "search_knowledge",
    )
    assert decision.reason_codes == (
        "knowledge_evidence_required",
    )


def test_decide_recognizes_current_simulated_state_as_telemetry(
) -> None:
    """“当前模拟状态”必须映射到模拟遥测能力。

    被测试模块是AgentToolPolicy.decide()（当前修改模块）。
    测试输入同时要求核对故障原因、当前模拟状态和安全恢复
    条件；前后两项工程知识由knowledge_evidence覆盖，中间的
    实时模拟字段由robot_telemetry覆盖。预期策略按稳定顺序
    返回search_knowledge和get_robot_telemetry，证明策略不会
    再把宽泛但明确的当前状态请求漏成纯知识检索。
    """

    decision = AgentToolPolicy().decide(
        make_request(
            task_goal=(
                "核对故障原因、当前模拟状态"
                "和安全恢复条件"
            ),
        )
    )

    assert decision.required_capabilities == (
        "knowledge_evidence",
        "robot_telemetry",
    )
    assert decision.allowed_tool_names == (
        "search_knowledge",
        "get_robot_telemetry",
    )
    assert decision.reason_codes == (
        "knowledge_evidence_required",
        "robot_telemetry_required",
    )


def test_decide_maps_all_requested_capabilities_in_stable_order(
) -> None:
    """五项能力应映射成五个按注册顺序排列的工具。"""

    request = make_request(
        task_goal=(
            "读取图片中的网络状态，检索知识库证据，"
            "读取模拟遥测，生成测试草案，并查询当前时间"
        ),
        images=[
            make_image(
                "读取图片中的网络状态"
            ),
        ],
    )

    decision = AgentToolPolicy().decide(
        request
    )

    assert decision.required_capabilities == (
        "vision_observation",
        "knowledge_evidence",
        "robot_telemetry",
        "test_case_draft",
        "current_time",
    )
    assert decision.allowed_tool_names == (
        "analyze_robot_image",
        "search_knowledge",
        "get_robot_telemetry",
        "draft_test_case",
        "get_current_time",
    )
    assert decision.reason_codes == (
        "image_input_available",
        "vision_observation_required",
        "knowledge_evidence_required",
        "robot_telemetry_required",
        "test_case_draft_required",
        "current_time_required",
    )


def test_decide_combines_knowledge_and_test_case_draft(
) -> None:
    """证据驱动的测试草案需要检索和草案两个工具。"""

    decision = AgentToolPolicy().decide(
        make_request(
            task_goal=(
                "依据知识库证据生成恢复验证测试草案"
            ),
        )
    )

    assert decision.required_capabilities == (
        "knowledge_evidence",
        "test_case_draft",
    )
    assert decision.allowed_tool_names == (
        "search_knowledge",
        "draft_test_case",
    )


def test_decide_allows_only_current_time_when_requested(
) -> None:
    """只询问系统时间时不应开放诊断类工具。"""

    decision = AgentToolPolicy().decide(
        make_request(
            task_goal="查询当前系统时间",
        )
    )

    assert decision.required_capabilities == (
        "current_time",
    )
    assert decision.allowed_tool_names == (
        "get_current_time",
    )


def test_decide_explicit_exclusions_override_positive_words(
) -> None:
    """明确的否定表达必须覆盖同句中的正向关键词。"""

    decision = AgentToolPolicy().decide(
        make_request(
            task_goal=(
                "检索网络恢复规则，"
                "不读取遥测也不生成测试草案"
            ),
        )
    )

    assert decision.required_capabilities == (
        "knowledge_evidence",
    )
    assert decision.allowed_tool_names == (
        "search_knowledge",
    )


def test_decide_excludes_vision_when_request_says_not_to_read_image(
) -> None:
    """任务明确不读图片时，即使附图也不能开放Vision工具。"""

    decision = AgentToolPolicy().decide(
        make_request(
            task_goal=(
                "不读取图片，只检索知识库恢复规则"
            ),
            images=[
                make_image(
                    "读取面板上的故障码"
                ),
            ],
        )
    )

    assert decision.required_capabilities == (
        "knowledge_evidence",
    )
    assert decision.allowed_tool_names == (
        "search_knowledge",
    )
    assert "image_input_available" not in (
        decision.reason_codes
    )


def test_decide_defaults_to_knowledge_when_task_goal_is_missing(
) -> None:
    """未提供task_goal时仍执行证据约束的默认诊断。"""

    decision = AgentToolPolicy().decide(
        make_request(
            task_goal=None,
        )
    )

    assert decision.required_capabilities == (
        "knowledge_evidence",
    )
    assert decision.allowed_tool_names == (
        "search_knowledge",
    )


def test_decide_defaults_unknown_goal_to_knowledge(
) -> None:
    """无法识别专用能力时应回到知识证据诊断职责。"""

    decision = AgentToolPolicy().decide(
        make_request(
            task_goal="帮助分析当前现象",
        )
    )

    assert decision.disposition == (
        "continue_to_planner"
    )
    assert decision.allowed_tool_names == (
        "search_knowledge",
    )


def test_decide_abstains_when_required_image_is_missing(
) -> None:
    """依赖图片但没有附图时必须在Planner之前拒答。"""

    decision = AgentToolPolicy().decide(
        make_request(
            task_goal=(
                "读取图片中的故障码并检索知识库证据"
            ),
        )
    )

    assert decision.disposition == "abstained"
    assert decision.required_capabilities == (
        "vision_observation",
        "knowledge_evidence",
    )
    assert decision.allowed_tool_names == ()
    assert decision.reason_codes == (
        "required_image_missing",
        "no_safe_tool_required",
    )
    assert decision.public_message == (
        "当前任务需要读取现场图片，"
        "但请求没有提供可用图片"
    )
    assert decision.missing_information == (
        "缺少与当前任务对应的现场图片",
    )


def test_decide_abstains_when_image_facets_are_disjoint(
) -> None:
    """连接器任务配电量图片时应判定图片明显无关。"""

    decision = AgentToolPolicy().decide(
        make_request(
            task_goal=(
                "观察图片中的J3连接器锁紧环是否贴合"
            ),
            images=[
                make_image(
                    "读取图片里的电池电量和当前速度"
                ),
            ],
        )
    )

    assert decision.disposition == "abstained"
    assert decision.allowed_tool_names == ()
    assert decision.reason_codes == (
        "image_not_relevant",
        "no_safe_tool_required",
    )
    assert decision.missing_information == (
        "缺少与当前任务目标相关的现场图片",
    )


def test_decide_continues_when_one_of_multiple_images_is_relevant(
) -> None:
    """多张图片中至少一张分面相关时应允许Vision观察。"""

    decision = AgentToolPolicy().decide(
        make_request(
            task_goal=(
                "观察图片中的J3连接器锁紧环间隙"
            ),
            images=[
                make_image(
                    "读取电池电量和当前速度"
                ),
                make_image(
                    "检查J3连接器锁紧环是否贴合"
                ),
            ],
        )
    )

    assert decision.disposition == (
        "continue_to_planner"
    )
    assert decision.allowed_tool_names == (
        "analyze_robot_image",
    )


def test_decide_does_not_open_vision_merely_because_image_exists(
) -> None:
    """附带图片本身不能自动扩大任务的工具权限。"""

    decision = AgentToolPolicy().decide(
        make_request(
            task_goal="检索知识库中的恢复条件",
            images=[
                make_image(
                    "检查面板上的NET指示灯"
                ),
            ],
        )
    )

    assert decision.required_capabilities == (
        "knowledge_evidence",
    )
    assert decision.allowed_tool_names == (
        "search_knowledge",
    )


def test_decide_supports_english_vision_and_knowledge_intent(
) -> None:
    """常见英文任务表达也应映射到视觉与知识能力。"""

    decision = AgentToolPolicy().decide(
        make_request(
            task_goal=(
                "Inspect image connector and search manual evidence"
            ),
            images=[
                make_image(
                    "Check connector locking ring"
                ),
            ],
        )
    )

    assert decision.required_capabilities == (
        "vision_observation",
        "knowledge_evidence",
    )
    assert decision.allowed_tool_names == (
        "analyze_robot_image",
        "search_knowledge",
    )


def test_decide_uses_request_context_for_generic_connection_observation(
) -> None:
    """“观察连接状态”应结合J3上下文开放Vision。

    被测试模块是AgentToolPolicy.decide()中的视觉能力识别和
    图片分面相关性判断。任务目标只写宽泛的“观察连接状态”，
    但现象、日志和图片analysis_goal都明确指向J3连接器。

    预期流程是：策略先从观察动作识别潜在视觉需求，再组合
    task_goal、symptom和log_excerpt确认连接器分面，最后映射出
    analyze_robot_image、search_knowledge和draft_test_case。该测试
    防止008/029一类真实场景因为任务措辞不含“图片”而误拦Vision。
    """

    request = AgentDiagnosisRequest(
        robot_id="robot-003",
        symptom="需要为读码器连接间隙准备检查草案",
        log_excerpt=(
            "DM260 J3 connector visible gap; inspection only"
        ),
        task_goal=(
            "观察连接状态、检索手册证据并生成只读检查草案；"
            "不得带电插拔或控制设备"
        ),
        images=[
            make_image(
                "观察传感器插头、J3插座和锁紧环是否存在间隙"
            ),
        ],
    )

    decision = AgentToolPolicy().decide(
        request
    )

    assert decision.required_capabilities == (
        "vision_observation",
        "knowledge_evidence",
        "test_case_draft",
    )
    assert decision.allowed_tool_names == (
        "analyze_robot_image",
        "search_knowledge",
        "draft_test_case",
    )


def test_decide_treats_visibility_check_as_image_observation(
) -> None:
    """“检查许可是否可见”应调用Vision后再决定是否拒答。

    被测试模块是AgentToolPolicy.decide()（当前修改模块）。请求
    携带一张analysis_goal声明遮挡检查的图片，任务要求判断恢复
    许可是否可见。预期策略只开放analyze_robot_image，而不是在
    尚未观察像素前错误开放search_knowledge或提前拒答。
    """

    request = AgentDiagnosisRequest(
        robot_id="robot-001",
        symptom="用户要求读取被遮挡的恢复许可字段",
        log_excerpt="recovery permission value unavailable",
        task_goal="检查恢复许可是否可见；不可见时不得批准恢复",
        images=[
            make_image(
                "区分仍然可见的标签与已经被遮挡的具体数值"
            ),
        ],
    )

    decision = AgentToolPolicy().decide(
        request
    )

    assert decision.required_capabilities == (
        "vision_observation",
    )
    assert decision.allowed_tool_names == (
        "analyze_robot_image",
    )


def test_decide_keeps_generic_unreadable_numeric_image_for_battery_task(
) -> None:
    """通用数值遮挡目标不能被误判为与电量任务无关。

    被测试模块是AgentToolPolicy.decide()中的图片分面判断。任务
    明确读取精确电量，图片analysis_goal只声明“具体数值是否被
    遮挡”，没有重复写“电池”。预期流程仍允许Vision观察；后续
    Vision可以返回unusable并触发安全拒答，而策略层不能在读取
    图片前猜测该数值一定与电量无关。
    """

    request = AgentDiagnosisRequest(
        robot_id="robot-001",
        symptom="需要确认被遮挡的具体电量",
        log_excerpt="battery value absent from log",
        task_goal="读取图片中的精确电量；不可见时安全拒答",
        images=[
            make_image(
                "区分仍然可见的标签与已经被遮挡的具体数值"
            ),
        ],
    )

    decision = AgentToolPolicy().decide(
        request
    )

    assert decision.disposition == (
        "continue_to_planner"
    )
    assert decision.allowed_tool_names == (
        "analyze_robot_image",
    )


def test_decide_rejects_wrong_request_type(
) -> None:
    """策略边界必须拒绝绕过AgentDiagnosisRequest的数据。"""

    with pytest.raises(
        TypeError,
        match="request必须是AgentDiagnosisRequest",
    ):
        AgentToolPolicy().decide(  # type: ignore[arg-type]
            object()
        )
