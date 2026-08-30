"""Agent内部诊断草稿数据契约的离线测试。

本文件只创建Pydantic模型：

1. 不启动FastAPI；
2. 不调用Planner或外部LLM；
3. 不执行任何Agent工具；
4. 不读取真实知识库或机器人遥测；
5. 不把格式合法的Chunk ID误判为已经通过证据白名单。

测试重点是AgentDiagnosisDraft自身负责的字段和关系校验。
Chunk ID真实性将在后续AgentDiagnosisService测试中验证。
"""

import pytest

from pydantic import ValidationError

from app.schemas.agent_diagnosis import AgentDiagnosisDraft
from app.schemas.diagnostics import DiagnosisCause, DiagnosisCheck


# 使用符合项目真实Chunk ID外观的固定测试文档ID。
# 该值只用于离线数据契约测试，不对应真实知识库文档。
TEST_DOCUMENT_ID = "a" * 64


# 原因和检查项通过该ID建立证据引用关系。
# 当前Schema只检查它是合法字符串，不检查它是否存在于Store。
TEST_CHUNK_ID = f"{TEST_DOCUMENT_ID}:000001"


def make_cause(
    *,
    chunk_id: str = TEST_CHUNK_ID,
) -> DiagnosisCause:
    """创建一项带Chunk ID引用的可能原因。"""

    return DiagnosisCause(
        description="任务仍在等待调度系统完成状态核对",
        evidence_chunk_ids=[chunk_id],
    )


def make_check(
    *,
    chunk_id: str = TEST_CHUNK_ID,
    risk_level: str = "medium",
    requires_qualified_person: bool = False,
) -> DiagnosisCheck:
    """创建一项带证据引用的下一步安全检查。"""

    return DiagnosisCheck.model_validate({
        "description": "核对RCS/RMS是否已经下发恢复命令",
        "evidence_chunk_ids": [chunk_id],
        "risk_level": risk_level,
        "requires_qualified_person": requires_qualified_person,
    })


def make_completed_draft(
    *,
    chunk_id: str = TEST_CHUNK_ID,
) -> AgentDiagnosisDraft:
    """创建一份没有信息缺口的完整诊断草稿。"""

    return AgentDiagnosisDraft(
        status="completed",
        possible_causes=(make_cause(chunk_id=chunk_id),),
        next_checks=(make_check(chunk_id=chunk_id),),
        risk_level="medium",
        missing_information=(),
        abstained=False,
    )


def test_completed_draft_accepts_supported_conclusions() -> None:
    """完整草稿应接受带证据ID的原因和检查项。"""

    draft = make_completed_draft()

    assert draft.status == "completed"
    assert draft.abstained is False
    assert len(draft.possible_causes) == 1
    assert len(draft.next_checks) == 1
    assert draft.missing_information == ()


def test_partial_draft_requires_and_accepts_missing_information() -> None:
    """部分草稿应保留已支持结论并明确剩余信息缺口。"""

    draft = AgentDiagnosisDraft(
        status="partial",
        possible_causes=(make_cause(),),
        next_checks=(),
        risk_level="medium",
        missing_information=("尚未取得机器人当前模拟遥测快照",),
        abstained=False,
    )

    assert draft.status == "partial"
    assert draft.possible_causes
    assert draft.missing_information == (
        "尚未取得机器人当前模拟遥测快照",
    )


def test_abstained_draft_accepts_safe_empty_conclusions() -> None:
    """拒答草稿应只返回unknown风险和明确缺失信息。"""

    draft = AgentDiagnosisDraft(
        status="abstained",
        risk_level="unknown",
        missing_information=("当前请求尚未取得足够知识库证据",),
        abstained=True,
    )

    assert draft.possible_causes == ()
    assert draft.next_checks == ()
    assert draft.risk_level == "unknown"


def test_draft_strips_text_and_converts_json_arrays_to_tuples() -> None:
    """模型应清理文本空白并把JSON数组转换成内部tuple。"""

    draft = AgentDiagnosisDraft.model_validate({
        "status": "partial",
        "possible_causes": [
            {
                "description": "  任务仍在等待状态核对  ",
                "evidence_chunk_ids": [TEST_CHUNK_ID],
            }
        ],
        "next_checks": [],
        "risk_level": "medium",
        "missing_information": ["  尚未取得当前任务状态  "],
        "abstained": False,
    })

    assert isinstance(draft.possible_causes, tuple)
    assert isinstance(draft.missing_information, tuple)
    assert draft.possible_causes[0].description == "任务仍在等待状态核对"
    assert draft.missing_information == ("尚未取得当前任务状态",)


def test_draft_rejects_model_generated_chunk_metadata() -> None:
    """Planner不能在草稿中填写来源文件或证据正文。"""

    with pytest.raises(ValidationError, match="extra_forbidden"):
        AgentDiagnosisDraft.model_validate({
            "status": "abstained",
            "possible_causes": [],
            "next_checks": [],
            "risk_level": "unknown",
            "missing_information": ["现有证据不足"],
            "abstained": True,
            "source_file": "伪造文件.txt",
            "excerpt": "伪造证据正文",
        })


@pytest.mark.parametrize(
    ("status", "abstained"),
    [
        ("completed", True),
        ("partial", True),
        ("abstained", False),
    ],
)
def test_draft_rejects_mismatched_status_and_abstained(
    status: str,
    abstained: bool,
) -> None:
    """status与abstained不能表达互相矛盾的结果。"""

    with pytest.raises(ValidationError, match="status与abstained不一致"):
        AgentDiagnosisDraft.model_validate({
            "status": status,
            "possible_causes": [],
            "next_checks": [],
            "risk_level": "unknown",
            "missing_information": ["固定测试信息缺口"],
            "abstained": abstained,
        })


def test_abstained_draft_rejects_possible_causes() -> None:
    """拒答结果不能同时输出未经确认的可能原因。"""

    with pytest.raises(ValidationError, match="拒答草稿不能包含"):
        AgentDiagnosisDraft(
            status="abstained",
            possible_causes=(make_cause(),),
            risk_level="unknown",
            missing_information=("现有证据不足",),
            abstained=True,
        )


def test_abstained_draft_requires_missing_information() -> None:
    """拒答不能只返回状态而不解释还缺少什么。"""

    with pytest.raises(ValidationError, match="missing_information"):
        AgentDiagnosisDraft(
            status="abstained",
            risk_level="unknown",
            missing_information=(),
            abstained=True,
        )


def test_abstained_draft_requires_unknown_risk() -> None:
    """拒答时不能在没有诊断结论的情况下声明确定风险。"""

    with pytest.raises(ValidationError, match="risk_level"):
        AgentDiagnosisDraft(
            status="abstained",
            risk_level="high",
            missing_information=("现有证据不足",),
            abstained=True,
        )


def test_non_abstained_draft_requires_cause_or_check() -> None:
    """非拒答草稿不能只有状态而没有任何诊断内容。"""

    with pytest.raises(ValidationError, match="至少需要"):
        AgentDiagnosisDraft(
            status="completed",
            risk_level="medium",
            abstained=False,
        )


def test_partial_draft_rejects_empty_missing_information() -> None:
    """partial必须说明为什么当前诊断仍不完整。"""

    with pytest.raises(ValidationError, match="partial草稿"):
        AgentDiagnosisDraft(
            status="partial",
            possible_causes=(make_cause(),),
            risk_level="medium",
            missing_information=(),
            abstained=False,
        )


def test_completed_draft_rejects_missing_information() -> None:
    """completed不能同时声明仍有尚未解决的信息。"""

    with pytest.raises(ValidationError, match="completed草稿"):
        AgentDiagnosisDraft(
            status="completed",
            possible_causes=(make_cause(),),
            risk_level="medium",
            missing_information=("仍然缺少当前任务状态",),
            abstained=False,
        )


def test_draft_rejects_duplicate_missing_information() -> None:
    """同一信息缺口不能在草稿中重复出现。"""

    with pytest.raises(ValidationError, match="不能包含重复项"):
        AgentDiagnosisDraft(
            status="partial",
            possible_causes=(make_cause(),),
            risk_level="medium",
            missing_information=(
                "尚未取得当前任务状态",
                "尚未取得当前任务状态",
            ),
            abstained=False,
        )


def test_draft_reuses_high_risk_qualification_rule() -> None:
    """高风险检查必须继续沿用第三周的资质人员约束。"""

    with pytest.raises(ValidationError, match="高风险检查"):
        AgentDiagnosisDraft(
            status="completed",
            next_checks=(
                make_check(
                    risk_level="high",
                    requires_qualified_person=False,
                ),
            ),
            risk_level="high",
            abstained=False,
        )


def test_schema_accepts_syntactic_id_before_service_whitelist() -> None:
    """Schema层不应伪装成能够证明Chunk ID来自真实检索。"""

    unconfirmed_chunk_id = "not-confirmed-in-current-request"
    draft = make_completed_draft(chunk_id=unconfirmed_chunk_id)

    assert draft.possible_causes[0].evidence_chunk_ids == [
        unconfirmed_chunk_id,
    ]


def test_draft_top_level_fields_are_frozen() -> None:
    """已校验草稿的顶层状态不能在创建后被直接替换。"""

    draft = make_completed_draft()

    with pytest.raises(ValidationError, match="frozen_instance"):
        draft.status = "partial"
