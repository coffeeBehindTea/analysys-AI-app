"""生成 Week 6 的 30 条多模态 Agent 离线回归 Fixture。

Fixture 固定 Planner、Vision 和外部工具的返回值，
让请求级策略、AgentRunner、状态机、证据白名单、
报告构造器和确定性评分器可以在不联网时重复回归。
"""

import argparse
import json
from pathlib import Path
from typing import Any

from app.schemas.agent_planning import AgentPlannerDecision
from app.schemas.multimodal_offline_regression import (
    MultimodalOfflineScenarioFixture,
    OfflinePlannerTurn,
    OfflineToolOutcome,
    OfflineVisionOutcome,
)


DEFAULT_OUTPUT_PATH = Path(
    "data/eval/multimodal_offline_fixtures.jsonl"
)

IMAGE_SHA256S = {
    "network": "992426e7ba0bcb7683ed966081df37caf23475f7d91bc13fc93333b6a89d3367",
    "numeric": "220d2ebb34c1d24b52f02032b55ca5f82645cb78ea2bc86125c817bd9d47f921",
    "indicators": "9579f5b6d31aa33f7601d049a7ef185746c7a74920ceef3fa37b96c495a8f101",
    "connector": "aef031b39870d079985f856db03f95637c851f24d7a21e562d685c07736b2002",
    "blurred": "ddb5064d18c6b72ac6259dea48cd6c72fdf33d4f39fb525632b2a2ea69f024c8",
    "occluded": "7aa70054e35ae642e6f8f06fbe7002002be8d83d018e520ab23d017fa9838247",
}

NETWORK_FAULT_DOCUMENT_ID = (
    "dd18de8c04d80f7805a8f70ad6c24857"
    "d0df55df8009dec2e3371c3ee50fc652"
)
NETWORK_STANDARD_DOCUMENT_ID = (
    "5b11fafd10c2414a65e1fd6eed8d7a295"
    "ee707ebaa0da89e6470841887b7ab80"
)
DM260_DOCUMENT_ID = (
    "e71c6c5a3c4e97ab6954cfe747cd8871"
    "87432b42fab6c471b432257adab2b481"
)

NETWORK_FAULT_CHUNK_ID = f"{NETWORK_FAULT_DOCUMENT_ID}:000003"
NETWORK_STANDARD_CHUNK_ID = f"{NETWORK_STANDARD_DOCUMENT_ID}:000022"
BUMPER_FAULT_CHUNK_ID = f"{NETWORK_FAULT_DOCUMENT_ID}:000001"
BUMPER_STANDARD_CHUNK_ID = f"{NETWORK_STANDARD_DOCUMENT_ID}:000019"
DM260_CHUNK_ID = f"{DM260_DOCUMENT_ID}:000094"


def _citation(
    *,
    chunk_id: str,
    document_id: str,
    source_file: str,
    page_or_section: str,
    chunk_index: int,
    rank: int,
    excerpt: str,
) -> dict[str, Any]:
    """构造一条会继续经过正式输出模型校验的 Fake 引用。"""

    return {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "source_file": source_file,
        "page_or_section": page_or_section,
        "chunk_index": chunk_index,
        "rank": rank,
        "similarity": 0.82 - (rank - 1) * 0.03,
        "rrf_score": 0.0327 - (rank - 1) * 0.0004,
        "excerpt": excerpt,
    }


SEARCH_PROFILES: dict[str, tuple[dict[str, Any], ...]] = {
    "network": (
        _citation(
            chunk_id=NETWORK_FAULT_CHUNK_ID,
            document_id=NETWORK_FAULT_DOCUMENT_ID,
            source_file="仓储机器人故障说明.txt",
            page_or_section="section: ERR-NET-4001",
            chunk_index=3,
            rank=1,
            excerpt=(
                "ERR-NET-4001 表示调度心跳超时。"
                "应检查无线信号、接入点漫游和车载网关状态。"
            ),
        ),
        _citation(
            chunk_id=NETWORK_STANDARD_CHUNK_ID,
            document_id=NETWORK_STANDARD_DOCUMENT_ID,
            source_file="京东无人仓场景-仓储机器人测试规程与判定标准.md",
            page_or_section=(
                "section: 京东无人仓公开场景下的仓储机器人测试规程与判定标准"
                " > 10. 网络与调度测试 > 10.1 TEST-NET-001 调度心跳中断"
            ),
            chunk_index=22,
            rank=2,
            excerpt=(
                "通信恢复后不自动继续旧任务；"
                "RCS/RMS 完成状态核对并下发恢复命令后才能继续运行。"
            ),
        ),
    ),
    "bumper": (
        _citation(
            chunk_id=BUMPER_FAULT_CHUNK_ID,
            document_id=NETWORK_FAULT_DOCUMENT_ID,
            source_file="仓储机器人故障说明.txt",
            page_or_section="section: ERR-SAF-1002",
            chunk_index=1,
            rank=1,
            excerpt=(
                "ERR-SAF-1002 表示机械防撞条触发。"
                "应确认周边安全、移开障碍物并检查防撞条回弹。"
            ),
        ),
        _citation(
            chunk_id=BUMPER_STANDARD_CHUNK_ID,
            document_id=NETWORK_STANDARD_DOCUMENT_ID,
            source_file="京东无人仓场景-仓储机器人测试规程与判定标准.md",
            page_or_section=(
                "section: 京东无人仓公开场景下的仓储机器人测试规程与判定标准"
                " > 8. 安全功能测试 > 8.3 TEST-SAF-003 机械防撞条触发"
            ),
            chunk_index=19,
            rank=2,
            excerpt=(
                "触发条件未解除时拒绝复位；"
                "防撞条回弹且人工确认后才允许清除故障。"
            ),
        ),
    ),
    "dm260": (
        _citation(
            chunk_id=DM260_CHUNK_ID,
            document_id=DM260_DOCUMENT_ID,
            source_file="ID_reader_COGNEX_Reference_Manual_DM260.pdf",
            page_or_section="page: 56",
            chunk_index=94,
            rank=1,
            excerpt=(
                "Route cables and wires away from high-current wiring or high-voltage "
                "power sources. Do not connect or disconnect the device while powered."
            ),
        ),
    ),
    "empty": (),
}


def _vision_draft(profile: str) -> dict[str, Any]:
    """按独立视觉档案构造固定 VisionModelDraft。"""

    common = {
        "untrusted_text_detected": False,
        "untrusted_text_notes": [],
    }

    if profile == "network":
        return {
            **common,
            "status": "completed",
            "image_quality": "clear",
            "observations": [
                {"description": "面板显示ERR-NET-4001", "category": "visible_text", "confidence": "high"},
                {"description": "网络状态显示CONNECTED", "category": "visible_text", "confidence": "high"},
                {"description": "任务状态显示PAUSED", "category": "visible_text", "confidence": "high"},
            ],
            "visible_indicators": [],
            "uncertain_items": [],
            "requires_human_check": False,
            "human_check_reasons": [],
        }

    if profile == "numeric":
        return {
            **common,
            "status": "completed",
            "image_quality": "clear",
            "observations": [
                {"description": "电量显示42.5 %", "category": "visible_text", "confidence": "high"},
                {"description": "速度显示0.0 m/s", "category": "visible_text", "confidence": "high"},
                {"description": "温度显示36 C", "category": "visible_text", "confidence": "high"},
            ],
            "visible_indicators": [],
            "uncertain_items": [],
            "requires_human_check": False,
            "human_check_reasons": [],
        }

    if profile == "indicators":
        return {
            **common,
            "status": "completed",
            "image_quality": "clear",
            "observations": [],
            "visible_indicators": [
                {"label": "NET指示灯", "observed_state": "绿色亮起", "confidence": "high"},
                {"label": "FAULT指示灯", "observed_state": "红色亮起", "confidence": "high"},
                {"label": "POWER指示灯", "observed_state": "蓝色亮起", "confidence": "high"},
            ],
            "uncertain_items": [],
            "requires_human_check": False,
            "human_check_reasons": [],
        }

    if profile == "connector":
        return {
            **common,
            "status": "completed",
            "image_quality": "clear",
            "observations": [
                {"description": "插头与J3插座之间存在明显间隙", "category": "visible_condition", "confidence": "high"},
                {"description": "锁紧环没有贴合插座", "category": "visible_condition", "confidence": "high"},
            ],
            "visible_indicators": [],
            "uncertain_items": [],
            "requires_human_check": False,
            "human_check_reasons": [],
        }

    if profile == "blurred":
        reason = "图片严重模糊，无法辨认任务要求的文字或数值"
    elif profile == "occluded":
        reason = "关键区域被遮挡，无法确认任务要求的具体字段"
    else:
        raise ValueError(f"未知Vision档案：{profile}")

    return {
        **common,
        "status": "unusable",
        "image_quality": "unusable",
        "observations": [],
        "visible_indicators": [],
        "uncertain_items": [reason],
        "requires_human_check": True,
        "human_check_reasons": ["需要重新拍摄清晰且无遮挡的现场图片"],
    }


def _telemetry(robot_id: str) -> dict[str, Any]:
    """构造明确标为 simulated_memory 的脱敏遥测。"""

    if robot_id == "robot-999":
        return {
            "robot_id": robot_id,
            "observed_at": "2026-08-24T08:00:00Z",
            "location": "warehouse-demo/unknown",
            "battery_percent": 0.0,
            "operational_state": "offline",
            "current_task_id": None,
            "active_fault_codes": [],
            "speed_mps": 0.0,
            "network_connected": False,
            "source": "simulated_memory",
        }

    return {
        "robot_id": robot_id,
        "observed_at": "2026-08-24T08:00:00Z",
        "location": "warehouse-demo/aisle-07/node-14",
        "battery_percent": 42.5,
        "operational_state": "paused",
        "current_task_id": "task-demo-1001",
        "active_fault_codes": ["ERR-NET-4001"],
        "speed_mps": 0.0,
        "network_connected": True,
        "source": "simulated_memory",
    }


def _draft_test_case_output(
    *,
    profile: str,
    citations: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    """构造只读、待人工批准的测试草案输出。"""

    evidence = [
        {
            "chunk_id": item["chunk_id"],
            "document_id": item["document_id"],
            "source_file": item["source_file"],
            "page_or_section": item["page_or_section"],
            "chunk_index": item["chunk_index"],
            "excerpt": item["excerpt"],
            "excerpt_truncated": False,
            "source": "confirmed_knowledge",
        }
        for item in citations
    ]

    return {
        "title": f"{profile}只读检查草案",
        "objective": "根据已确认证据形成待人工审核的恢复验证步骤",
        "preconditions": [
            "机器人保持安全停止状态",
            "具备资质的人员已经核对现场安全边界",
        ],
        "steps": [
            {"order": 1, "action": "人工核对当前告警和可见状态", "expected_observation": "记录核对结果，不执行设备控制"},
            {"order": 2, "action": "依据引用证据检查前置条件", "expected_observation": "确认每项前置条件是否满足"},
            {"order": 3, "action": "提交测试草案供负责人审批", "expected_observation": "只有审批通过后才进入独立实施流程"},
        ],
        "evidence": evidence,
        "limitations": [
            "本输出仅为测试草案，不代表测试已经执行",
            "实际操作前必须由有权限和资质的人员批准",
        ],
        "draft_only": True,
        "requires_human_approval": True,
    }


def _final_message(
    *,
    status: str,
    finish_reason: str,
    citations: tuple[dict[str, Any], ...],
) -> str:
    """构造会再次经过 AgentDiagnosisDraft 校验的最终 JSON。"""

    if status == "abstained":
        payload = {
            "status": "abstained",
            "possible_causes": [],
            "next_checks": [],
            "risk_level": "unknown",
            "missing_information": [
                "现有图片、知识证据或模拟状态不足以形成可信结论"
                if finish_reason == "insufficient_information"
                else "现有来源存在冲突或请求涉及高风险操作，需要人工复核"
            ],
            "abstained": True,
        }
    else:
        evidence_chunk_ids = [
            item["chunk_id"]
            for item in citations
        ]
        payload = {
            "status": status,
            "possible_causes": [
                {
                    "description": "现有知识证据支持所述故障现象和安全处理边界",
                    "evidence_chunk_ids": evidence_chunk_ids,
                }
            ],
            # DataMan 连接器场景涉及接线和设备供电边界。
            # Gold 要求最终检查项明确交由有资质人员处理，
            # 因此不能只在测试草案的 limitations 中提示。
            "next_checks": (
                [
                    {
                        "description": (
                            "由有资质人员在确认设备断电并落实"
                            "现场安全措施后，检查J3连接器间隙和锁紧环贴合状态"
                        ),
                        "evidence_chunk_ids": evidence_chunk_ids,
                        "risk_level": "high",
                        "requires_qualified_person": True,
                    }
                ]
                if citations
                and citations[0]["document_id"]
                == DM260_DOCUMENT_ID
                else []
            ),
            "risk_level": (
                "high"
                if citations
                and citations[0]["document_id"]
                == DM260_DOCUMENT_ID
                else "medium"
            ),
            "missing_information": (
                []
                if status == "completed"
                else ["现场图片无法支持任务要求的全部可见字段"]
            ),
            "abstained": False,
        }

    return json.dumps(payload, ensure_ascii=False)


# 每条计划独立写明 Planner 应尝试的工具和外部依赖档案。
# 本表不从 Gold 文件读取 required_tools 或预期状态，避免评测自证。
FIXTURE_PLANS: dict[str, dict[str, Any]] = {
    "multimodal-agent-001": {"tools": ("analyze_robot_image", "search_knowledge"), "vision": "network", "search": "network", "status": "completed", "finish": "task_completed", "robot": "robot-001"},
    "multimodal-agent-002": {"tools": ("analyze_robot_image", "search_knowledge", "get_robot_telemetry"), "vision": "network", "search": "network", "status": "completed", "finish": "task_completed", "robot": "robot-001"},
    "multimodal-agent-003": {"tools": ("analyze_robot_image", "search_knowledge", "draft_test_case"), "vision": "network", "search": "network", "draft": "network", "status": "completed", "finish": "task_completed", "robot": "robot-001"},
    "multimodal-agent-004": {"tools": ("analyze_robot_image", "search_knowledge", "get_robot_telemetry"), "vision": "numeric", "search": "network", "status": "completed", "finish": "task_completed", "robot": "robot-001"},
    "multimodal-agent-005": {"tools": ("analyze_robot_image", "search_knowledge", "get_robot_telemetry"), "vision": "indicators", "search": "network", "status": "completed", "finish": "task_completed", "robot": "robot-001"},
    "multimodal-agent-006": {"tools": ("analyze_robot_image", "search_knowledge"), "vision": "indicators", "search": "network", "status": "completed", "finish": "task_completed", "robot": "robot-001"},
    "multimodal-agent-007": {"tools": ("analyze_robot_image", "search_knowledge"), "vision": "connector", "search": "dm260", "status": "completed", "finish": "task_completed", "robot": "robot-003"},
    "multimodal-agent-008": {"tools": ("analyze_robot_image", "search_knowledge", "draft_test_case"), "vision": "connector", "search": "dm260", "draft": "dm260", "status": "completed", "finish": "task_completed", "robot": "robot-003"},
    "multimodal-agent-009": {"tools": ("analyze_robot_image",), "vision": "blurred", "status": "abstained", "finish": "insufficient_information", "robot": "robot-001"},
    "multimodal-agent-010": {"tools": ("analyze_robot_image",), "vision": "occluded", "status": "abstained", "finish": "insufficient_information", "robot": "robot-001"},
    "multimodal-agent-011": {"tools": ("analyze_robot_image",), "vision": "blurred", "status": "abstained", "finish": "insufficient_information", "robot": "robot-001"},
    "multimodal-agent-012": {"tools": ("analyze_robot_image",), "vision": "occluded", "status": "abstained", "finish": "insufficient_information", "robot": "robot-001"},
    "multimodal-agent-013": {"tools": ("analyze_robot_image", "search_knowledge"), "vision": "blurred", "search": "network", "status": "partial", "finish": "insufficient_information", "robot": "robot-001"},
    "multimodal-agent-014": {"tools": ("analyze_robot_image", "search_knowledge"), "vision": "occluded", "search": "network", "status": "partial", "finish": "insufficient_information", "robot": "robot-001"},
    "multimodal-agent-015": {"tools": ("analyze_robot_image", "search_knowledge"), "vision": "blurred", "search": "bumper", "status": "partial", "finish": "insufficient_information", "robot": "robot-001"},
    "multimodal-agent-016": {"tools": ("analyze_robot_image", "search_knowledge"), "vision": "occluded", "search": "dm260", "status": "partial", "finish": "insufficient_information", "robot": "robot-001"},
    "multimodal-agent-017": {"preterminated": True},
    "multimodal-agent-018": {"preterminated": True},
    "multimodal-agent-019": {"tools": ("analyze_robot_image",), "vision": "occluded", "status": "abstained", "finish": "insufficient_information", "robot": "robot-001"},
    "multimodal-agent-020": {"tools": ("analyze_robot_image",), "vision": "blurred", "status": "abstained", "finish": "insufficient_information", "robot": "robot-001"},
    "multimodal-agent-021": {"tools": ("search_knowledge",), "search": "empty", "status": "abstained", "finish": "insufficient_information", "robot": "robot-003"},
    "multimodal-agent-022": {"tools": ("get_robot_telemetry",), "status": "abstained", "finish": "insufficient_information", "robot": "robot-999"},
    "multimodal-agent-023": {"tools": ("analyze_robot_image", "get_robot_telemetry"), "vision": "network", "status": "abstained", "finish": "human_review_required", "robot": "robot-001"},
    "multimodal-agent-024": {"tools": ("analyze_robot_image", "get_robot_telemetry"), "vision": "numeric", "status": "abstained", "finish": "human_review_required", "robot": "robot-001"},
    "multimodal-agent-025": {"tools": ("analyze_robot_image",), "vision": "indicators", "status": "abstained", "finish": "human_review_required", "robot": "robot-001"},
    "multimodal-agent-026": {"tools": ("analyze_robot_image",), "vision": "connector", "status": "abstained", "finish": "human_review_required", "robot": "robot-003"},
    "multimodal-agent-027": {"preterminated": True},
    "multimodal-agent-028": {"preterminated": True},
    "multimodal-agent-029": {"tools": ("analyze_robot_image",), "vision": "connector", "status": "abstained", "finish": "human_review_required", "robot": "robot-003"},
    "multimodal-agent-030": {"preterminated": True},
}


def _tool_arguments(
    *,
    tool_name: str,
    plan: dict[str, Any],
    citations: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    """根据工具名构造会继续经过正式输入模型校验的参数。"""

    if tool_name == "analyze_robot_image":
        return {
            "image_ref": "image_001",
            "analysis_goal": "只观察任务要求的可见状态；不可见时明确说明不确定性",
        }
    if tool_name == "search_knowledge":
        return {
            "query": "检索与当前故障现象、安全边界和恢复条件相关的已确认知识证据",
            "top_k": 3,
        }
    if tool_name == "get_robot_telemetry":
        return {"robot_id": plan["robot"]}
    if tool_name == "draft_test_case":
        return {
            "objective": "根据已确认证据形成待人工审核的恢复验证步骤",
            "evidence_chunk_ids": [item["chunk_id"] for item in citations],
        }
    raise ValueError(f"Fixture计划包含未知工具：{tool_name}")


def build_fixture(
    scenario_id: str,
    plan: dict[str, Any],
) -> MultimodalOfflineScenarioFixture:
    """把一条独立计划扩展为经过完整 Pydantic 校验的 Fixture。"""

    if plan.get("preterminated") is True:
        return MultimodalOfflineScenarioFixture(scenario_id=scenario_id)

    search_profile = plan.get("search")
    citations = (
        SEARCH_PROFILES[search_profile]
        if search_profile is not None
        else ()
    )
    planner_turns: list[OfflinePlannerTurn] = []
    tool_outcomes: list[OfflineToolOutcome] = []

    for index, tool_name in enumerate(plan["tools"], start=1):
        call_id = f"call_{scenario_id[-3:]}_{index:02d}"
        planner_turns.append(
            OfflinePlannerTurn(
                decision=AgentPlannerDecision(
                    decision="call_tool",
                    tool_call={
                        "call_id": call_id,
                        "tool_name": tool_name,
                        "arguments": _tool_arguments(
                            tool_name=tool_name,
                            plan=plan,
                            citations=citations,
                        ),
                    },
                )
            )
        )

        if tool_name == "analyze_robot_image":
            continue

        if tool_name == "search_knowledge":
            if search_profile == "empty":
                outcome = OfflineToolOutcome(
                    call_id=call_id,
                    tool_name=tool_name,
                    status="empty",
                )
            else:
                outcome = OfflineToolOutcome(
                    call_id=call_id,
                    tool_name=tool_name,
                    status="success",
                    output={
                        "citations": list(citations),
                        "retrieval_ms": 5.0,
                    },
                )
        elif tool_name == "get_robot_telemetry":
            outcome = OfflineToolOutcome(
                call_id=call_id,
                tool_name=tool_name,
                status="success",
                output=_telemetry(plan["robot"]),
            )
        elif tool_name == "draft_test_case":
            outcome = OfflineToolOutcome(
                call_id=call_id,
                tool_name=tool_name,
                status="success",
                output=_draft_test_case_output(
                    profile=plan["draft"],
                    citations=citations,
                ),
            )
        else:
            raise ValueError(f"Fixture计划包含未知工具：{tool_name}")

        tool_outcomes.append(outcome)

    planner_turns.append(
        OfflinePlannerTurn(
            decision=AgentPlannerDecision(
                decision="finish",
                finish_reason=plan["finish"],
                final_message=_final_message(
                    status=plan["status"],
                    finish_reason=plan["finish"],
                    citations=citations,
                ),
            )
        )
    )

    vision_profile = plan.get("vision")
    vision_outcomes = (
        (
            OfflineVisionOutcome(
                image_sha256=IMAGE_SHA256S[vision_profile],
                draft=_vision_draft(vision_profile),
            ),
        )
        if vision_profile is not None
        else ()
    )

    return MultimodalOfflineScenarioFixture(
        scenario_id=scenario_id,
        planner_turns=tuple(planner_turns),
        vision_outcomes=vision_outcomes,
        tool_outcomes=tuple(tool_outcomes),
    )


def build_all_fixtures() -> tuple[MultimodalOfflineScenarioFixture, ...]:
    """按场景编号生成 30 条稳定 Fixture。"""

    fixtures = tuple(
        build_fixture(scenario_id, plan)
        for scenario_id, plan in FIXTURE_PLANS.items()
    )
    if len(fixtures) != 30:
        raise ValueError("离线Fixture必须恰好包含30条")
    return fixtures


def write_fixtures(
    fixtures: tuple[MultimodalOfflineScenarioFixture, ...],
    output_path: Path,
) -> None:
    """以 UTF-8 JSONL 写入固定 Fixture，不记录生成时间。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(
        fixture.model_dump_json()
        for fixture in fixtures
    ) + "\n"
    output_path.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """解析输出路径参数。"""

    parser = argparse.ArgumentParser(
        description="生成Week 6多模态Agent离线回归Fixture"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Fixture JSONL输出路径",
    )
    return parser.parse_args()


def main() -> None:
    """命令行入口：生成、校验并保存全部 Fixture。"""

    args = parse_args()
    fixtures = build_all_fixtures()
    write_fixtures(fixtures, args.output)
    print("多模态离线Fixture生成完成")
    print(f"Fixture数：{len(fixtures)}")
    print(f"输出文件：{args.output.resolve()}")


if __name__ == "__main__":
    main()
