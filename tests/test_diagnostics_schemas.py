"""结构化诊断API数据契约的离线测试。

本模块只构造Pydantic模型，不访问检索器、LLM、
Chroma、文件系统或HTTP服务。

测试范围：
1. 请求字段清理和可选检索范围；
2. 检索范围必须非空且不能重复；
3. 高风险检查必须标记需有资质人员确认；
4. 正常报告中的证据引用白名单；
5. 未知Chunk引用和重复证据拒绝；
6. completed、partial和abstained状态一致性；
7. 非拒答报告必须拥有真实证据；
8. 未声明字段必须被拒绝。
"""

import pytest
from pydantic import ValidationError

from app.schemas.diagnostics import (
    DiagnosisCause,
    DiagnosisCheck,
    DiagnosisEvidence,
    DiagnosisReport,
    DiagnosisRequest,
    DiagnosisRetrievalScope,
    DiagnosisSymptom,
)


# 使用稳定SHA-256格式值构造测试文档身份。
TEST_DOCUMENT_ID = "a" * 64
TEST_CHUNK_ID = (
    f"{TEST_DOCUMENT_ID}:000001"
)


def make_evidence(
    *,
    chunk_id: str = TEST_CHUNK_ID,
    rank: int = 1,
) -> DiagnosisEvidence:
    """构造一条来自真实检索结果形状的证据。"""

    return DiagnosisEvidence(
        chunk_id=chunk_id,
        document_id=TEST_DOCUMENT_ID,
        source_file="robot-faults.txt",
        page_or_section="section: ERR-NET-4001",
        rank=rank,
        rrf_score=0.032,
        vector_similarity=0.58,
        excerpt=(
            "通信恢复后不自动继续旧任务，"
            "需要调度系统核对状态。"
        ),
    )


def make_symptoms(
) -> list[DiagnosisSymptom]:
    """构造来自用户描述和日志的两项观察。"""

    return [
        DiagnosisSymptom(
            description="机器人停止执行任务",
            source="user_report",
        ),
        DiagnosisSymptom(
            description="日志出现ERR-NET-4001",
            source="log_excerpt",
        ),
    ]


def make_completed_report(
) -> DiagnosisReport:
    """构造证据、原因和检查项全部自洽的报告。"""

    return DiagnosisReport(
        request_id="request-001",
        prompt_version="diagnosis-evidence-v1",
        gate_version="hybrid-evidence-gate-v1",
        status="completed",
        symptoms=make_symptoms(),
        evidence=[make_evidence()],
        possible_causes=[
            DiagnosisCause(
                description="调度通信曾经中断",
                evidence_chunk_ids=[
                    TEST_CHUNK_ID
                ],
            )
        ],
        next_checks=[
            DiagnosisCheck(
                description=(
                    "核对调度系统与机器人状态"
                ),
                evidence_chunk_ids=[
                    TEST_CHUNK_ID
                ],
                risk_level="low",
                requires_qualified_person=False,
            )
        ],
        risk_level="low",
        missing_information=[],
        abstained=False,
    )


def test_request_strips_text_and_accepts_scope(
) -> None:
    """请求应清理空白并保存合法文档范围。"""

    request = DiagnosisRequest(
        robot_id="  robot-001  ",
        symptom="  机器人停止移动  ",
        log_excerpt=(
            "  ERR-NET-4001 heartbeat lost  "
        ),
        retrieval_scope=DiagnosisRetrievalScope(
            document_ids=[TEST_DOCUMENT_ID],
            source_files=["robot-faults.txt"],
        ),
    )

    assert request.robot_id == "robot-001"
    assert request.symptom == "机器人停止移动"
    assert request.log_excerpt == (
        "ERR-NET-4001 heartbeat lost"
    )
    assert request.retrieval_scope is not None
    assert request.retrieval_scope.document_ids == [
        TEST_DOCUMENT_ID
    ]


def test_retrieval_scope_requires_a_filter(
) -> None:
    """显式提供的检索范围不能是空对象。"""

    with pytest.raises(
        ValidationError,
        match="检索范围至少需要一个筛选条件",
    ):
        DiagnosisRetrievalScope()


def test_retrieval_scope_rejects_blank_source_file(
) -> None:
    """纯空白文件名不能冒充有效检索范围。"""

    # str_strip_whitespace会先把"   "清理成""；
    # SourceFileName的min_length=1随后必须拒绝它。
    with pytest.raises(
        ValidationError,
        match="string_too_short",
    ):
        DiagnosisRetrievalScope(
            source_files=["   "]
        )


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        (
            "document_ids",
            [TEST_DOCUMENT_ID, TEST_DOCUMENT_ID],
        ),
        (
            "source_files",
            ["manual.pdf", "manual.pdf"],
        ),
    ],
    ids=[
        "duplicate-document-id",
        "duplicate-source-file",
    ],
)
def test_retrieval_scope_rejects_duplicates(
    field_name: str,
    field_value: list[str],
) -> None:
    """重复过滤值不能造成重复检索或歧义。"""

    with pytest.raises(
        ValidationError,
        match="不能包含重复项",
    ):
        DiagnosisRetrievalScope(
            **{field_name: field_value}
        )


def test_high_risk_check_requires_qualified_person(
) -> None:
    """高风险检查不得被包装成普通用户执行指令。"""

    with pytest.raises(
        ValidationError,
        match=(
            "高风险检查必须标记为"
            "需要有资质人员确认"
        ),
    ):
        DiagnosisCheck(
            description="检查高压电池连接",
            evidence_chunk_ids=[TEST_CHUNK_ID],
            risk_level="high",
            requires_qualified_person=False,
        )


def test_completed_report_accepts_known_evidence(
) -> None:
    """原因和检查项只引用报告证据时应通过校验。"""

    report = make_completed_report()

    assert report.status == "completed"
    assert report.abstained is False
    assert len(report.evidence) == 1
    assert report.possible_causes[
        0
    ].evidence_chunk_ids == [TEST_CHUNK_ID]


def test_report_rejects_unknown_evidence_reference(
) -> None:
    """模型虚构或引用未返回Chunk时必须拒绝报告。"""

    with pytest.raises(
        ValidationError,
        match="引用了本次报告中不存在的Chunk ID",
    ):
        DiagnosisReport(
            request_id="request-002",
            prompt_version="diagnosis-evidence-v1",
            gate_version="hybrid-evidence-gate-v1",
            status="completed",
            symptoms=make_symptoms(),
            evidence=[make_evidence()],
            possible_causes=[
                DiagnosisCause(
                    description="未知原因",
                    evidence_chunk_ids=[
                        "invented-chunk:000001"
                    ],
                )
            ],
            next_checks=[],
            risk_level="unknown",
            missing_information=[],
            abstained=False,
        )


def test_report_rejects_duplicate_evidence_chunks(
) -> None:
    """同一个Chunk不能在证据列表中重复出现。"""

    with pytest.raises(
        ValidationError,
        match="evidence不能包含重复的chunk_id",
    ):
        DiagnosisReport(
            request_id="request-003",
            prompt_version="diagnosis-evidence-v1",
            gate_version="hybrid-evidence-gate-v1",
            status="completed",
            symptoms=make_symptoms(),
            evidence=[
                make_evidence(rank=1),
                make_evidence(rank=2),
            ],
            possible_causes=[],
            next_checks=[
                DiagnosisCheck(
                    description="核对状态",
                    evidence_chunk_ids=[
                        TEST_CHUNK_ID
                    ],
                    risk_level="low",
                    requires_qualified_person=False,
                )
            ],
            risk_level="low",
            missing_information=[],
            abstained=False,
        )


def test_abstained_report_requires_missing_information(
) -> None:
    """拒答必须说明还缺少什么，不能只返回空结果。"""

    with pytest.raises(
        ValidationError,
        match="拒答报告必须说明缺失信息",
    ):
        DiagnosisReport(
            request_id="request-004",
            prompt_version="diagnosis-evidence-v1",
            gate_version="hybrid-evidence-gate-v1",
            status="abstained",
            symptoms=make_symptoms(),
            evidence=[],
            possible_causes=[],
            next_checks=[],
            risk_level="unknown",
            missing_information=[],
            abstained=True,
        )


def test_missing_information_rejects_blank_item(
) -> None:
    """纯空白文本不能满足拒答报告的缺失信息要求。"""

    # 如果列表中的空白字符串能够通过，
    # model_validator会误以为系统已经解释了缺失信息。
    with pytest.raises(
        ValidationError,
        match="string_too_short",
    ):
        DiagnosisReport(
            request_id="request-blank-missing",
            prompt_version="diagnosis-evidence-v1",
            gate_version="hybrid-evidence-gate-v1",
            status="abstained",
            symptoms=make_symptoms(),
            evidence=[],
            possible_causes=[],
            next_checks=[],
            risk_level="unknown",
            missing_information=["   "],
            abstained=True,
        )


def test_abstained_report_cannot_include_conclusions(
) -> None:
    """拒答状态不能同时给出原因或排查动作。"""

    with pytest.raises(
        ValidationError,
        match="拒答报告不能包含原因或排查项",
    ):
        DiagnosisReport(
            request_id="request-005",
            prompt_version="diagnosis-evidence-v1",
            gate_version="hybrid-evidence-gate-v1",
            status="abstained",
            symptoms=make_symptoms(),
            evidence=[make_evidence()],
            possible_causes=[
                DiagnosisCause(
                    description="通信中断",
                    evidence_chunk_ids=[
                        TEST_CHUNK_ID
                    ],
                )
            ],
            next_checks=[],
            risk_level="unknown",
            missing_information=[
                "需要更多日志"
            ],
            abstained=True,
        )


def test_non_abstained_report_requires_evidence(
) -> None:
    """声称完成或部分完成的报告必须有真实证据。"""

    with pytest.raises(
        ValidationError,
        match="非拒答报告必须包含证据",
    ):
        DiagnosisReport(
            request_id="request-006",
            prompt_version="diagnosis-evidence-v1",
            gate_version="hybrid-evidence-gate-v1",
            status="completed",
            symptoms=make_symptoms(),
            evidence=[],
            possible_causes=[],
            next_checks=[],
            risk_level="unknown",
            missing_information=[],
            abstained=False,
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
    """状态文字和布尔拒答标志不能互相矛盾。"""

    with pytest.raises(
        ValidationError,
        match="status与abstained不一致",
    ):
        DiagnosisReport(
            request_id="request-007",
            prompt_version="diagnosis-evidence-v1",
            gate_version="hybrid-evidence-gate-v1",
            status=status,
            symptoms=make_symptoms(),
            evidence=[make_evidence()],
            possible_causes=[],
            next_checks=[],
            risk_level="unknown",
            missing_information=[
                "测试缺失项"
            ],
            abstained=abstained,
        )


def test_partial_report_requires_missing_information(
) -> None:
    """部分诊断必须明确哪些信息仍然不足。"""

    with pytest.raises(
        ValidationError,
        match="partial报告必须说明缺失信息",
    ):
        DiagnosisReport(
            request_id="request-008",
            prompt_version="diagnosis-evidence-v1",
            gate_version="hybrid-evidence-gate-v1",
            status="partial",
            symptoms=make_symptoms(),
            evidence=[make_evidence()],
            possible_causes=[],
            next_checks=[
                DiagnosisCheck(
                    description="核对日志",
                    evidence_chunk_ids=[
                        TEST_CHUNK_ID
                    ],
                    risk_level="low",
                    requires_qualified_person=False,
                )
            ],
            risk_level="low",
            missing_information=[],
            abstained=False,
        )


def test_completed_report_rejects_missing_information(
) -> None:
    """仍有缺失信息时不能把报告标成completed。"""

    report_data = make_completed_report().model_dump()
    report_data["missing_information"] = [
        "仍缺少控制器状态"
    ]

    with pytest.raises(
        ValidationError,
        match="completed报告不能包含缺失信息",
    ):
        DiagnosisReport.model_validate(
            report_data
        )


def test_request_rejects_unknown_fields(
) -> None:
    """请求字段拼写错误不能被静默忽略。"""

    with pytest.raises(
        ValidationError,
        match="extra_forbidden",
    ):
        DiagnosisRequest(
            robot_id="robot-001",
            symptom="机器人停止移动",
            log_excerpt="heartbeat lost",
            unknown_field="unexpected",
        )
