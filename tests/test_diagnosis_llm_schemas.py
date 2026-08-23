"""结构化诊断LLM内部输出契约的离线测试。

本模块只构造Pydantic模型，不访问LLM、Embedding、
Chroma、文件系统或HTTP服务。

测试范围：
1. 合法完成报告草稿；
2. 临时证据编号格式与去重；
3. 高风险检查的人员资质约束；
4. completed、partial和abstained状态一致性；
5. 拒答与非拒答输出内容约束；
6. 缺失信息不能为空白；
7. 未声明字段必须被拒绝。
"""

import pytest
from pydantic import ValidationError

from app.schemas.diagnosis_llm import (
    DiagnosisDraftCause,
    DiagnosisDraftCheck,
    DiagnosisLLMDraft,
)


def make_completed_draft(
) -> DiagnosisLLMDraft:
    """构造引用E1的合法完整诊断草稿。"""

    return DiagnosisLLMDraft(
        status="completed",
        possible_causes=[
            DiagnosisDraftCause(
                description="调度通信曾经中断",
                evidence_ids=["E1"],
            )
        ],
        next_checks=[
            DiagnosisDraftCheck(
                description="核对机器人与调度任务状态",
                evidence_ids=["E1"],
                risk_level="low",
                requires_qualified_person=False,
            )
        ],
        risk_level="low",
        missing_information=[],
        abstained=False,
    )


def test_completed_draft_is_valid(
) -> None:
    """格式、状态和证据编号都合法时应生成草稿对象。"""

    draft = make_completed_draft()

    assert draft.status == "completed"
    assert draft.abstained is False
    assert draft.possible_causes[
        0
    ].evidence_ids == ["E1"]


@pytest.mark.parametrize(
    "invalid_evidence_id",
    [
        "E0",
        "e1",
        "real-chunk-id",
    ],
)
def test_evidence_id_requires_temporary_format(
    invalid_evidence_id: str,
) -> None:
    """LLM只能引用E1、E2等本次Prompt临时编号。"""

    with pytest.raises(
        ValidationError,
        match="string_pattern_mismatch",
    ):
        DiagnosisDraftCause(
            description="测试原因",
            evidence_ids=[
                invalid_evidence_id
            ],
        )


def test_cause_rejects_duplicate_evidence_ids(
) -> None:
    """同一原因不能重复引用同一个临时证据。"""

    with pytest.raises(
        ValidationError,
        match="evidence_ids不能包含重复项",
    ):
        DiagnosisDraftCause(
            description="测试原因",
            evidence_ids=["E1", "E1"],
        )


def test_high_risk_draft_check_requires_qualification(
) -> None:
    """LLM不能把高风险检查标成普通人员可执行。"""

    with pytest.raises(
        ValidationError,
        match=(
            "高风险检查必须标记为"
            "需要有资质人员确认"
        ),
    ):
        DiagnosisDraftCheck(
            description="检查高压电池连接",
            evidence_ids=["E1"],
            risk_level="high",
            requires_qualified_person=False,
        )


@pytest.mark.parametrize(
    ("status", "abstained"),
    [
        ("abstained", False),
        ("completed", True),
        ("partial", True),
    ],
)
def test_status_and_abstained_must_agree(
    status: str,
    abstained: bool,
) -> None:
    """草稿状态文字与拒答布尔值不能互相矛盾。"""

    with pytest.raises(
        ValidationError,
        match="status与abstained不一致",
    ):
        DiagnosisLLMDraft(
            status=status,
            possible_causes=[],
            next_checks=[],
            risk_level="unknown",
            missing_information=[
                "测试缺失项"
            ],
            abstained=abstained,
        )


def test_abstained_draft_requires_missing_information(
) -> None:
    """LLM拒答时必须解释仍缺少哪些信息。"""

    with pytest.raises(
        ValidationError,
        match="拒答草稿必须说明缺失信息",
    ):
        DiagnosisLLMDraft(
            status="abstained",
            possible_causes=[],
            next_checks=[],
            risk_level="unknown",
            missing_information=[],
            abstained=True,
        )


def test_abstained_draft_cannot_include_conclusions(
) -> None:
    """拒答草稿不能同时给出原因或检查动作。"""

    with pytest.raises(
        ValidationError,
        match="拒答草稿不能包含原因或排查项",
    ):
        DiagnosisLLMDraft(
            status="abstained",
            possible_causes=[
                DiagnosisDraftCause(
                    description="通信中断",
                    evidence_ids=["E1"],
                )
            ],
            next_checks=[],
            risk_level="unknown",
            missing_information=[
                "需要更多日志"
            ],
            abstained=True,
        )


def test_non_abstained_draft_requires_a_conclusion(
) -> None:
    """非拒答草稿至少需要一项原因或检查。"""

    with pytest.raises(
        ValidationError,
        match="非拒答草稿至少需要一项原因或排查项",
    ):
        DiagnosisLLMDraft(
            status="completed",
            possible_causes=[],
            next_checks=[],
            risk_level="unknown",
            missing_information=[],
            abstained=False,
        )


def test_partial_draft_requires_missing_information(
) -> None:
    """部分诊断必须明确剩余信息缺口。"""

    with pytest.raises(
        ValidationError,
        match="partial草稿必须说明缺失信息",
    ):
        DiagnosisLLMDraft(
            status="partial",
            possible_causes=[
                DiagnosisDraftCause(
                    description="可能存在通信中断",
                    evidence_ids=["E1"],
                )
            ],
            next_checks=[],
            risk_level="unknown",
            missing_information=[],
            abstained=False,
        )


def test_completed_draft_rejects_missing_information(
) -> None:
    """仍有信息缺口时不能宣称诊断已经完成。"""

    with pytest.raises(
        ValidationError,
        match="completed草稿不能包含缺失信息",
    ):
        DiagnosisLLMDraft(
            status="completed",
            possible_causes=[
                DiagnosisDraftCause(
                    description="通信中断",
                    evidence_ids=["E1"],
                )
            ],
            next_checks=[],
            risk_level="low",
            missing_information=[
                "仍缺少调度状态"
            ],
            abstained=False,
        )


def test_missing_information_rejects_blank_item(
) -> None:
    """纯空白字符串不能冒充拒答解释。"""

    with pytest.raises(
        ValidationError,
        match="string_too_short",
    ):
        DiagnosisLLMDraft(
            status="abstained",
            possible_causes=[],
            next_checks=[],
            risk_level="unknown",
            missing_information=["   "],
            abstained=True,
        )


def test_draft_rejects_unknown_fields(
) -> None:
    """LLM自行添加的文件名或结论字段必须被拒绝。"""

    draft_data = make_completed_draft().model_dump()
    draft_data["source_file"] = "invented.pdf"

    with pytest.raises(
        ValidationError,
        match="extra_forbidden",
    ):
        DiagnosisLLMDraft.model_validate(
            draft_data
        )

