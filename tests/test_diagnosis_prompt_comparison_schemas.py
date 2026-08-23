"""诊断Prompt对照实验数据契约的离线测试。

本模块不调用LLM、Embedding、Chroma或HTTP服务。

测试范围：
1. 单次Prompt运行成功和失败状态是否自洽；
2. 同一案例是否绑定相同的请求、证据和两个不同Prompt版本；
3. 完整报告是否记录模型、temperature和唯一案例编号；
4. 未声明字段是否被拒绝。
5. 报告是否明确禁止保存生成时间。
"""

import pytest
from pydantic import ValidationError

from app.schemas.diagnosis_llm import (
    DiagnosisLLMDraft,
)
from app.schemas.diagnosis_prompt_comparison import (
    DiagnosisPromptComparisonCase,
    DiagnosisPromptComparisonReport,
    DiagnosisPromptRunResult,
    DiagnosisPromptVariantMetrics,
)
from app.schemas.diagnostics import DiagnosisRequest
from app.schemas.retrieval import (
    DocumentChunk,
    HybridRetrievedChunk,
)


# 测试数据遵守真实DocumentChunk的64位哈希契约。
TEST_DOCUMENT_ID = "a" * 64
TEST_CONTENT_HASH = "b" * 64


def make_draft() -> DiagnosisLLMDraft:
    """构造一个引用E1且通过Schema校验的诊断草稿。"""

    return DiagnosisLLMDraft(
        status="partial",
        possible_causes=[
            {
                "description": (
                    "调度状态可能尚未完成核对"
                ),
                "evidence_ids": ["E1"],
            }
        ],
        next_checks=[
            {
                "description": (
                    "核对调度系统中的当前任务状态"
                ),
                "evidence_ids": ["E1"],
                "risk_level": "low",
                "requires_qualified_person": False,
            }
        ],
        risk_level="low",
        missing_information=[
            "调度系统是否已经下发恢复命令"
        ],
        abstained=False,
    )


def make_request() -> DiagnosisRequest:
    """构造两个实验组共同使用的诊断请求。"""

    return DiagnosisRequest(
        robot_id="robot-001",
        symptom="网络恢复后机器人仍未继续任务",
        log_excerpt=(
            "ERR-NET-4001 heartbeat recovered"
        ),
    )


def make_evidence(
) -> list[HybridRetrievedChunk]:
    """构造两个实验组共同使用的一条已排序证据。"""

    return [
        HybridRetrievedChunk(
            chunk=DocumentChunk(
                chunk_id=(
                    f"{TEST_DOCUMENT_ID}:000001"
                ),
                document_id=TEST_DOCUMENT_ID,
                source_file="robot-faults.txt",
                page_or_section=(
                    "section: ERR-NET-4001"
                ),
                chunk_index=1,
                content_hash=TEST_CONTENT_HASH,
                content=(
                    "网络恢复后必须先核对调度状态，"
                    "再决定是否恢复任务。"
                ),
            ),
            rrf_score=0.032,
            rank=1,
            vector_rank=1,
            vector_similarity=0.68,
            keyword_rank=1,
            keyword_score=12.0,
            matched_identifiers=(
                "err-net-4001",
            ),
        )
    ]


def make_success_result(
    prompt_version: str,
) -> DiagnosisPromptRunResult:
    """构造一次成功且已通过Pydantic校验的Prompt结果。"""

    return DiagnosisPromptRunResult(
        prompt_version=prompt_version,
        outcome="success",
        generation_ms=125.5,
        draft=make_draft(),
        error_detail=None,
    )


def make_case(
    *,
    case_id: str = "dpc001",
) -> DiagnosisPromptComparisonCase:
    """构造基础Prompt与证据优先Prompt的同输入对照案例。"""

    return DiagnosisPromptComparisonCase(
        case_id=case_id,
        request=make_request(),
        evidence=make_evidence(),
        basic_result=make_success_result(
            "diagnosis-basic-v1"
        ),
        evidence_first_result=(
            make_success_result(
                "diagnosis-evidence-v1"
            )
        ),
    )


def make_metrics(
    prompt_version: str,
) -> DiagnosisPromptVariantMetrics:
    """构造与一个成功案例一致的Prompt汇总指标。"""

    return DiagnosisPromptVariantMetrics(
        prompt_version=prompt_version,
        case_count=1,
        success_count=1,
        invalid_response_count=0,
        timeout_count=0,
        upstream_error_count=0,
        abstained_count=0,
        whitelist_passed_count=1,
        unknown_evidence_id_count=0,
        schema_success_rate=1.0,
        evidence_whitelist_pass_rate=1.0,
        average_generation_ms=125.5,
    )


def test_success_result_requires_validated_draft(
) -> None:
    """成功结果必须保存经过Schema校验的诊断草稿。"""

    result = make_success_result(
        "diagnosis-evidence-v1"
    )

    assert result.outcome == "success"
    assert isinstance(
        result.draft,
        DiagnosisLLMDraft,
    )
    assert result.error_detail is None


def test_success_result_rejects_missing_draft(
) -> None:
    """不能把没有结构化草稿的运行标记为成功。"""

    with pytest.raises(
        ValidationError,
        match="success结果必须包含draft",
    ):
        DiagnosisPromptRunResult(
            prompt_version=(
                "diagnosis-evidence-v1"
            ),
            outcome="success",
            generation_ms=100.0,
            draft=None,
            error_detail=None,
        )


def test_success_result_rejects_error_detail(
) -> None:
    """成功和失败信息不能同时出现在同一运行结果中。"""

    with pytest.raises(
        ValidationError,
        match="success结果不能包含error_detail",
    ):
        DiagnosisPromptRunResult(
            prompt_version=(
                "diagnosis-evidence-v1"
            ),
            outcome="success",
            generation_ms=100.0,
            draft=make_draft(),
            error_detail="不应存在的错误",
        )


@pytest.mark.parametrize(
    "outcome",
    (
        "invalid_response",
        "timeout",
        "upstream_error",
    ),
)
def test_failure_result_requires_safe_error_detail(
    outcome: str,
) -> None:
    """三类失败都必须有说明且不能伪造成功草稿。"""

    result = DiagnosisPromptRunResult(
        prompt_version="diagnosis-basic-v1",
        outcome=outcome,
        generation_ms=50.0,
        draft=None,
        error_detail="安全的内部失败摘要",
    )

    assert result.draft is None
    assert result.error_detail == (
        "安全的内部失败摘要"
    )


def test_failure_result_rejects_draft(
) -> None:
    """失败结果不能同时携带一个看似有效的诊断草稿。"""

    with pytest.raises(
        ValidationError,
        match="失败结果不能包含draft",
    ):
        DiagnosisPromptRunResult(
            prompt_version="diagnosis-basic-v1",
            outcome="invalid_response",
            generation_ms=50.0,
            draft=make_draft(),
            error_detail="JSON不符合契约",
        )


def test_failure_result_requires_error_detail(
) -> None:
    """失败类型必须附带可审计但安全的错误摘要。"""

    with pytest.raises(
        ValidationError,
        match="失败结果必须包含error_detail",
    ):
        DiagnosisPromptRunResult(
            prompt_version="diagnosis-basic-v1",
            outcome="timeout",
            generation_ms=50.0,
            draft=None,
            error_detail=None,
        )


def test_comparison_case_binds_same_input_to_two_results(
) -> None:
    """一个案例应同时保存共享输入和两个Prompt运行结果。"""

    case = make_case()

    assert case.case_id == "dpc001"
    assert case.request.robot_id == "robot-001"
    assert len(case.evidence) == 1
    assert (
        case.basic_result.prompt_version
        == "diagnosis-basic-v1"
    )
    assert (
        case.evidence_first_result.prompt_version
        == "diagnosis-evidence-v1"
    )


def test_comparison_case_requires_distinct_versions(
) -> None:
    """同一Prompt版本不能冒充一次对照实验。"""

    same_result = make_success_result(
        "diagnosis-evidence-v1"
    )

    with pytest.raises(
        ValidationError,
        match="两个实验组必须使用不同Prompt版本",
    ):
        DiagnosisPromptComparisonCase(
            case_id="dpc001",
            request=make_request(),
            evidence=make_evidence(),
            basic_result=same_result,
            evidence_first_result=same_result,
        )


def test_variant_metrics_accept_consistent_counts(
) -> None:
    """汇总指标应保存自洽的结果计数、比率和耗时。"""

    metrics = make_metrics(
        "diagnosis-evidence-v1"
    )

    assert metrics.case_count == 1
    assert metrics.success_count == 1
    assert metrics.schema_success_rate == 1.0
    assert (
        metrics.evidence_whitelist_pass_rate
        == 1.0
    )


def test_variant_metrics_reject_outcome_count_mismatch(
) -> None:
    """成功和三类失败的总和必须等于案例总数。"""

    with pytest.raises(
        ValidationError,
        match="运行结果计数之和必须等于case_count",
    ):
        DiagnosisPromptVariantMetrics(
            prompt_version=(
                "diagnosis-evidence-v1"
            ),
            case_count=2,
            success_count=1,
            invalid_response_count=0,
            timeout_count=0,
            upstream_error_count=0,
            abstained_count=0,
            whitelist_passed_count=1,
            unknown_evidence_id_count=0,
            schema_success_rate=0.5,
            evidence_whitelist_pass_rate=0.5,
            average_generation_ms=125.5,
        )


def test_variant_metrics_reject_incorrect_rate(
) -> None:
    """调用方不能提交与整数计数不一致的成功率。"""

    with pytest.raises(
        ValidationError,
        match=(
            "schema_success_rate"
            "与success_count不一致"
        ),
    ):
        DiagnosisPromptVariantMetrics(
            prompt_version=(
                "diagnosis-evidence-v1"
            ),
            case_count=2,
            success_count=1,
            invalid_response_count=1,
            timeout_count=0,
            upstream_error_count=0,
            abstained_count=0,
            whitelist_passed_count=1,
            unknown_evidence_id_count=0,

            # 正确值应为1 / 2 = 0.5。
            schema_success_rate=1.0,
            evidence_whitelist_pass_rate=0.5,
            average_generation_ms=125.5,
        )


def test_report_rejects_duplicate_case_ids(
) -> None:
    """同一案例不能在汇总指标中被重复计数。"""

    with pytest.raises(
        ValidationError,
        match="Prompt对照报告包含重复case_id",
    ):
        DiagnosisPromptComparisonReport(
            llm_model="test-model",
            temperature=0.0,
            basic_metrics=make_metrics(
                "diagnosis-basic-v1"
            ),
            evidence_first_metrics=make_metrics(
                "diagnosis-evidence-v1"
            ),
            cases=[
                make_case(case_id="dpc001"),
                make_case(case_id="dpc001"),
            ],
        )


def test_report_serializes_complete_experiment(
) -> None:
    """报告必须保留复现实验所需的模型参数和完整案例。"""

    report = DiagnosisPromptComparisonReport(
        llm_model="test-model",
        temperature=0.0,
        basic_metrics=make_metrics(
            "diagnosis-basic-v1"
        ),
        evidence_first_metrics=make_metrics(
            "diagnosis-evidence-v1"
        ),
        cases=[make_case()],
    )

    data = report.model_dump(
        mode="json"
    )

    assert data["llm_model"] == "test-model"
    assert data["temperature"] == 0.0

    # 用户要求JSON报告不保存生成时间。
    assert "generated_at" not in data

    assert data["basic_metrics"][
        "schema_success_rate"
    ] == 1.0
    assert data["evidence_first_metrics"][
        "evidence_whitelist_pass_rate"
    ] == 1.0
    assert len(data["cases"]) == 1
    assert data["cases"][0][
        "basic_result"
    ]["outcome"] == "success"


def test_unknown_report_field_is_rejected(
) -> None:
    """拼错或未声明的报告字段不能被静默忽略。"""

    with pytest.raises(
        ValidationError,
        match="Extra inputs are not permitted",
    ):
        DiagnosisPromptComparisonReport(
            llm_model="test-model",
            temperature=0.0,
            basic_metrics=make_metrics(
                "diagnosis-basic-v1"
            ),
            evidence_first_metrics=make_metrics(
                "diagnosis-evidence-v1"
            ),
            cases=[make_case()],
            notes="未声明字段",  # type: ignore[call-arg]
        )


def test_report_rejects_generated_at_field(
) -> None:
    """生成时间不属于Prompt对照报告的数据契约。"""

    with pytest.raises(
        ValidationError,
        match="Extra inputs are not permitted",
    ):
        DiagnosisPromptComparisonReport(
            llm_model="test-model",
            temperature=0.0,
            basic_metrics=make_metrics(
                "diagnosis-basic-v1"
            ),
            evidence_first_metrics=make_metrics(
                "diagnosis-evidence-v1"
            ),
            cases=[make_case()],

            # extra="forbid"应拒绝重新加入该字段。
            generated_at=(
                "2026-08-21T08:30:00Z"
            ),  # type: ignore[call-arg]
        )
