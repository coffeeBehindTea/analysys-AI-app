"""请求级Agent工具策略数据契约的离线测试。

本文件只验证AgentToolPolicyDecision的字段和跨字段关系。

测试不会调用Planner、LLM、Vision、知识库、遥测或其他外部服务。
"""

from typing import (
    Any,
)

import pytest

from pydantic import (
    ValidationError,
)

from app.schemas.agent_tool_policy import (
    AGENT_READ_ONLY_TOOL_ORDER,
    CAPABILITY_TO_TOOL,
    AgentToolPolicyDecision,
)


# 正常继续进入Planner时使用的最小合法样本。
#
# 每个测试只覆盖自己关心的字段，避免重复构造整份数据。
BASE_CONTINUE_DATA: dict[str, Any] = {
    "disposition": "continue_to_planner",
    "required_capabilities": (
        "vision_observation",
        "knowledge_evidence",
    ),
    "allowed_tool_names": (
        "analyze_robot_image",
        "search_knowledge",
    ),
    "reason_codes": (
        "image_input_available",
        "vision_observation_required",
        "knowledge_evidence_required",
    ),
}


def make_continue_data(
    **overrides: Any,
) -> dict[str, Any]:
    """返回一份可以由单个测试安全修改的输入字典。"""

    data = dict(BASE_CONTINUE_DATA)
    data.update(overrides)
    return data


def test_continue_decision_accepts_minimal_matching_tools(
) -> None:
    """所需能力与最小工具完全对应时应允许进入Planner。"""

    decision = AgentToolPolicyDecision.model_validate(
        make_continue_data()
    )

    assert decision.policy_version == (
        "agent-tool-policy-v1"
    )
    assert decision.disposition == (
        "continue_to_planner"
    )
    assert decision.allowed_tool_names == (
        "analyze_robot_image",
        "search_knowledge",
    )
    assert decision.public_message is None
    assert decision.missing_information == ()


def test_abstained_decision_accepts_missing_image(
) -> None:
    """任务依赖图片但缺图时应允许无工具提前拒答。"""

    decision = AgentToolPolicyDecision(
        disposition="abstained",
        required_capabilities=(
            "vision_observation",
        ),
        allowed_tool_names=(),
        reason_codes=(
            "required_image_missing",
            "no_safe_tool_required",
        ),
        public_message=(
            "当前任务需要现场图片，但请求没有提供图片"
        ),
        missing_information=(
            "缺少能够支持视觉观察的现场图片",
        ),
    )

    assert decision.disposition == "abstained"
    assert decision.allowed_tool_names == ()
    assert decision.required_capabilities == (
        "vision_observation",
    )


def test_human_review_decision_accepts_high_risk_request(
) -> None:
    """高风险控制请求应允许无工具转人工审核。"""

    decision = AgentToolPolicyDecision(
        disposition="human_review_required",
        reason_codes=(
            "high_risk_control_detected",
            "no_safe_tool_required",
        ),
        public_message=(
            "该请求涉及设备安全控制，需要人工审核"
        ),
        missing_information=(
            "需要具备权限的人员现场确认安全状态",
        ),
    )

    assert decision.disposition == (
        "human_review_required"
    )
    assert decision.allowed_tool_names == ()


@pytest.mark.parametrize(
    "reason_code",
    (
        "external_link_action_detected",
        "untrusted_instruction_execution_detected",
    ),
)
def test_human_review_accepts_untrusted_action_reason_codes(
    reason_code: str,
) -> None:
    """外部链接和不可信命令应具有可审计的独立原因码。"""

    decision = AgentToolPolicyDecision(
        disposition="human_review_required",
        reason_codes=(
            reason_code,
            "no_safe_tool_required",
        ),
        public_message=(
            "请求要求执行不可信来源中的外部动作"
        ),
        missing_information=(
            "需要人工核验外部内容及其授权边界",
        ),
    )

    assert decision.reason_codes == (
        reason_code,
        "no_safe_tool_required",
    )
    assert decision.allowed_tool_names == ()


def test_unknown_tool_name_is_rejected(
) -> None:
    """未声明的工具不能进入请求级权限集合。"""

    with pytest.raises(ValidationError):
        AgentToolPolicyDecision.model_validate(
            make_continue_data(
                allowed_tool_names=(
                    "run_shell",
                ),
            )
        )


def test_unknown_capability_is_rejected(
) -> None:
    """未声明的任务能力不能绕过能力到工具的映射。"""

    with pytest.raises(ValidationError):
        AgentToolPolicyDecision.model_validate(
            make_continue_data(
                required_capabilities=(
                    "remote_robot_control",
                ),
            )
        )


@pytest.mark.parametrize(
    ("field_name", "duplicated_value"),
    [
        (
            "required_capabilities",
            "vision_observation",
        ),
        (
            "allowed_tool_names",
            "analyze_robot_image",
        ),
        (
            "reason_codes",
            "vision_observation_required",
        ),
        (
            "missing_information",
            "缺少现场图片",
        ),
    ],
)
def test_tuple_fields_reject_duplicate_items(
    field_name: str,
    duplicated_value: str,
) -> None:
    """所有策略元组都应拒绝重复项目。"""

    if field_name == "missing_information":
        data: dict[str, Any] = {
            "disposition": "abstained",
            "reason_codes": (
                "required_image_missing",
            ),
            "public_message": "请求缺少图片",
            field_name: (
                duplicated_value,
                duplicated_value,
            ),
        }
    else:
        data = make_continue_data(
            **{
                field_name: (
                    duplicated_value,
                    duplicated_value,
                ),
            }
        )

    with pytest.raises(
        ValidationError,
        match="不能包含重复项",
    ):
        AgentToolPolicyDecision.model_validate(
            data
        )


def test_allowed_tools_reject_unstable_order(
) -> None:
    """内容正确但顺序错误的工具集合也应被拒绝。"""

    with pytest.raises(
        ValidationError,
        match=(
            "AGENT_READ_ONLY_TOOL_ORDER"
        ),
    ):
        AgentToolPolicyDecision.model_validate(
            make_continue_data(
                allowed_tool_names=(
                    "search_knowledge",
                    "analyze_robot_image",
                ),
            )
        )


@pytest.mark.parametrize(
    "allowed_tool_names",
    [
        # 缺少知识库工具。
        (
            "analyze_robot_image",
        ),
        # 多暴露了遥测工具。
        (
            "analyze_robot_image",
            "search_knowledge",
            "get_robot_telemetry",
        ),
    ],
)
def test_continue_rejects_tools_that_do_not_match_capabilities(
    allowed_tool_names: tuple[str, ...],
) -> None:
    """继续执行时不能缺少必要工具或额外扩大权限。"""

    with pytest.raises(
        ValidationError,
        match=(
            "required_capabilities完全对应"
        ),
    ):
        AgentToolPolicyDecision.model_validate(
            make_continue_data(
                allowed_tool_names=(
                    allowed_tool_names
                ),
            )
        )


def test_continue_requires_at_least_one_capability(
) -> None:
    """没有任务能力时不能无目的地进入Planner。"""

    with pytest.raises(
        ValidationError,
        match=(
            "必须声明required_capabilities"
        ),
    ):
        AgentToolPolicyDecision(
            disposition=(
                "continue_to_planner"
            ),
            reason_codes=(
                "no_safe_tool_required",
            ),
        )


def test_continue_rejects_public_message(
) -> None:
    """策略层不能在进入Planner前提前生成用户结论。"""

    with pytest.raises(
        ValidationError,
        match="public_message必须为空",
    ):
        AgentToolPolicyDecision.model_validate(
            make_continue_data(
                public_message=(
                    "已经完成诊断"
                ),
            )
        )


def test_continue_rejects_missing_information(
) -> None:
    """继续执行和提前声明信息不足不能同时出现。"""

    with pytest.raises(
        ValidationError,
        match=(
            "missing_information必须为空"
        ),
    ):
        AgentToolPolicyDecision.model_validate(
            make_continue_data(
                missing_information=(
                    "仍然缺少现场图片",
                ),
            )
        )


def test_early_termination_rejects_allowed_tools(
) -> None:
    """提前结束时不能仍向Planner或Executor开放工具。"""

    with pytest.raises(
        ValidationError,
        match=(
            "allowed_tool_names必须为空"
        ),
    ):
        AgentToolPolicyDecision(
            disposition="abstained",
            required_capabilities=(
                "vision_observation",
            ),
            allowed_tool_names=(
                "analyze_robot_image",
            ),
            reason_codes=(
                "required_image_missing",
            ),
            public_message="请求缺少图片",
            missing_information=(
                "缺少现场图片",
            ),
        )


def test_early_termination_requires_public_message(
) -> None:
    """提前结束必须向调用方提供安全公开说明。"""

    with pytest.raises(
        ValidationError,
        match="必须提供public_message",
    ):
        AgentToolPolicyDecision(
            disposition="abstained",
            reason_codes=(
                "required_image_missing",
            ),
            missing_information=(
                "缺少现场图片",
            ),
        )


def test_early_termination_requires_missing_information(
) -> None:
    """提前结束必须说明信息缺口或人工处理事项。"""

    with pytest.raises(
        ValidationError,
        match=(
            "必须提供missing_information"
        ),
    ):
        AgentToolPolicyDecision(
            disposition=(
                "human_review_required"
            ),
            reason_codes=(
                "high_risk_control_detected",
            ),
            public_message=(
                "该请求需要人工审核"
            ),
        )


def test_extra_field_is_rejected(
) -> None:
    """策略结果不能携带未声明的调试或模型字段。"""

    with pytest.raises(
        ValidationError,
        match=(
            "Extra inputs are not permitted"
        ),
    ):
        AgentToolPolicyDecision.model_validate(
            make_continue_data(
                private_reasoning=(
                    "不应进入数据契约"
                ),
            )
        )


def test_capability_mapping_covers_every_tool_in_order(
) -> None:
    """能力映射应覆盖全部现有只读工具且顺序可复现。"""

    mapped_tools = tuple(
        CAPABILITY_TO_TOOL.values()
    )

    assert mapped_tools == (
        AGENT_READ_ONLY_TOOL_ORDER
    )
