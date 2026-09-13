"""诊断Prompt对照运行器的离线测试。

本模块使用Fake Provider和Fake Clock，
不访问真实LLM、Embedding、Chroma或HTTP服务。

测试范围：
1. 成功草稿是否转换成success运行结果；
2. 三类预期外部异常是否转换成可保存的失败结果；
3. 编程错误是否继续抛出；
4. 两个Prompt组是否收到同一个请求和同一份证据；
5. 两组版本相同时是否在调用Provider前失败。
"""

from collections.abc import Iterator
import pytest

from app.errors import (
    InvalidLLMResponseError,
    LLMTimeoutError,
    LLMUpstreamError,
)
from app.schemas.diagnosis_llm import (
    DiagnosisLLMDraft,
)
from app.schemas.diagnosis_prompt_comparison import (
    DiagnosisPromptComparisonCase,
    DiagnosisPromptRunResult,
)
from app.schemas.diagnostics import DiagnosisRequest
from app.schemas.retrieval import (
    DocumentChunk,
    HybridRetrievedChunk,
)
from app.services.diagnosis_prompt_comparison import (
    build_diagnosis_prompt_comparison_report,
    calculate_diagnosis_prompt_metrics,
    collect_diagnosis_evidence_ids,
    compare_diagnosis_prompt_case,
    run_diagnosis_prompt_variant,
)


# 测试Chunk继续使用真实Schema要求的64位哈希。
TEST_DOCUMENT_ID = "c" * 64
TEST_CONTENT_HASH = "d" * 64


def make_request() -> DiagnosisRequest:
    """构造两个Prompt组共同接收的请求对象。"""

    return DiagnosisRequest(
        robot_id="robot-001",
        symptom="网络恢复后机器人仍未继续任务",
        log_excerpt=(
            "ERR-NET-4001 heartbeat recovered"
        ),
    )


def make_evidence(
) -> tuple[HybridRetrievedChunk, ...]:
    """构造两个Prompt组共同接收的不可变证据快照。"""

    return (
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
                    "通信恢复后必须先核对调度状态。"
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
        ),
    )


def make_draft() -> DiagnosisLLMDraft:
    """构造一个已经通过内部响应契约的草稿。"""

    return DiagnosisLLMDraft(
        status="completed",
        possible_causes=[
            {
                "description": "调度状态尚未核对",
                "evidence_ids": ["E1"],
            }
        ],
        next_checks=[
            {
                "description": "核对调度任务状态",
                "evidence_ids": ["E1"],
                "risk_level": "low",
                "requires_qualified_person": False,
            }
        ],
        risk_level="low",
        missing_information=[],
        abstained=False,
    )


def make_draft_with_ids(
    *,
    cause_ids: list[str],
    check_ids: list[str],
) -> DiagnosisLLMDraft:
    """构造用于检查引用去重和未知编号的合法草稿。"""

    return DiagnosisLLMDraft(
        status="completed",
        possible_causes=[
            {
                "description": "可能原因",
                "evidence_ids": cause_ids,
            }
        ],
        next_checks=[
            {
                "description": "下一步检查",
                "evidence_ids": check_ids,
                "risk_level": "low",
                "requires_qualified_person": False,
            }
        ],
        risk_level="low",
        missing_information=[],
        abstained=False,
    )


def make_abstained_draft(
) -> DiagnosisLLMDraft:
    """构造不包含原因、检查或引用的合法拒答草稿。"""

    return DiagnosisLLMDraft(
        status="abstained",
        possible_causes=[],
        next_checks=[],
        risk_level="unknown",
        missing_information=[
            "当前证据不足"
        ],
        abstained=True,
    )


def make_success_run(
    *,
    version: str,
    draft: DiagnosisLLMDraft,
    generation_ms: float,
) -> DiagnosisPromptRunResult:
    """构造用于指标测试的成功运行结果。"""

    return DiagnosisPromptRunResult(
        prompt_version=version,
        outcome="success",
        generation_ms=generation_ms,
        draft=draft,
        error_detail=None,
    )


def make_failure_run(
    *,
    version: str,
    generation_ms: float,
) -> DiagnosisPromptRunResult:
    """构造用于指标测试的非法响应结果。"""

    return DiagnosisPromptRunResult(
        prompt_version=version,
        outcome="invalid_response",
        generation_ms=generation_ms,
        draft=None,
        error_detail="JSON不符合内部契约",
    )


def make_comparison_case(
    *,
    case_id: str,
    basic_result: DiagnosisPromptRunResult,
    evidence_first_result: (
        DiagnosisPromptRunResult
    ),
) -> DiagnosisPromptComparisonCase:
    """把指定的两组运行结果绑定到同一输入。"""

    return DiagnosisPromptComparisonCase(
        case_id=case_id,
        request=make_request(),
        evidence=list(make_evidence()),
        basic_result=basic_result,
        evidence_first_result=(
            evidence_first_result
        ),
    )


class FakeClock:
    """按预设顺序返回时间点，避免测试依赖真实执行速度。"""

    def __init__(
        self,
        *values: float,
    ) -> None:
        self._values: Iterator[float] = iter(
            values
        )

    def __call__(self) -> float:
        """模拟time.perf_counter()的可调用接口。"""

        return next(self._values)


class FakeVersionedProvider:
    """记录调用参数，并返回草稿或抛出预设异常。"""

    def __init__(
        self,
        *,
        prompt_version: str,
        draft: DiagnosisLLMDraft | None = None,
        error: Exception | None = None,
    ) -> None:
        self.prompt_version = prompt_version
        self._draft = draft
        self._error = error
        self.calls: list[
            tuple[
                DiagnosisRequest,
                tuple[HybridRetrievedChunk, ...],
            ]
        ] = []

    async def generate_draft(
        self,
        *,
        request: DiagnosisRequest,
        evidence: tuple[
            HybridRetrievedChunk,
            ...,
        ],
    ) -> DiagnosisLLMDraft:
        """模拟异步Provider，并记录收到的对象。"""

        self.calls.append(
            (request, evidence)
        )

        if self._error is not None:
            raise self._error

        if self._draft is None:
            raise AssertionError(
                "Fake Provider没有配置draft"
            )

        return self._draft


@pytest.mark.asyncio
async def test_run_variant_records_success(
) -> None:
    """合法草稿应形成带版本和耗时的success结果。"""

    provider = FakeVersionedProvider(
        prompt_version="diagnosis-evidence-v1",
        draft=make_draft(),
    )

    result = await run_diagnosis_prompt_variant(
        provider=provider,
        request=make_request(),
        evidence=make_evidence(),
        clock=FakeClock(10.0, 10.125),
    )

    assert result.outcome == "success"
    assert result.prompt_version == (
        "diagnosis-evidence-v1"
    )
    assert result.generation_ms == pytest.approx(
        125.0
    )
    assert result.draft == make_draft()
    assert result.error_detail is None
    assert len(provider.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected_outcome"),
    (
        (
            InvalidLLMResponseError(
                "JSON不符合内部契约"
            ),
            "invalid_response",
        ),
        (
            LLMTimeoutError("LLM响应超时"),
            "timeout",
        ),
        (
            LLMUpstreamError("LLM上游不可用"),
            "upstream_error",
        ),
    ),
)
async def test_run_variant_records_expected_failure(
    error: Exception,
    expected_outcome: str,
) -> None:
    """预期外部失败应转换成报告数据而不中断实验。"""

    provider = FakeVersionedProvider(
        prompt_version="diagnosis-basic-v1",
        error=error,
    )

    result = await run_diagnosis_prompt_variant(
        provider=provider,
        request=make_request(),
        evidence=make_evidence(),
        clock=FakeClock(20.0, 20.05),
    )

    assert result.outcome == expected_outcome
    assert result.draft is None
    assert result.error_detail == str(error)
    assert result.generation_ms == pytest.approx(
        50.0
    )


@pytest.mark.asyncio
async def test_run_variant_does_not_hide_programming_error(
) -> None:
    """未知异常不能被误报成一次正常的LLM失败。"""

    provider = FakeVersionedProvider(
        prompt_version="diagnosis-basic-v1",
        error=RuntimeError("测试中的程序错误"),
    )

    with pytest.raises(
        RuntimeError,
        match="测试中的程序错误",
    ):
        await run_diagnosis_prompt_variant(
            provider=provider,
            request=make_request(),
            evidence=make_evidence(),
            clock=FakeClock(30.0, 30.1),
        )


@pytest.mark.asyncio
async def test_compare_case_uses_identical_input(
) -> None:
    """基础组和证据优先组必须收到同一请求与证据对象。"""

    request = make_request()
    evidence = make_evidence()
    basic_provider = FakeVersionedProvider(
        prompt_version="diagnosis-basic-v1",
        draft=make_draft(),
    )
    evidence_provider = FakeVersionedProvider(
        prompt_version="diagnosis-evidence-v1",
        draft=make_draft(),
    )

    case = await compare_diagnosis_prompt_case(
        case_id="dpc001",
        request=request,
        evidence=evidence,
        basic_provider=basic_provider,
        evidence_first_provider=(
            evidence_provider
        ),
        clock=FakeClock(
            1.0,
            1.1,
            2.0,
            2.2,
        ),
    )

    assert case.case_id == "dpc001"
    assert case.basic_result.generation_ms == (
        pytest.approx(100.0)
    )
    assert (
        case.evidence_first_result.generation_ms
        == pytest.approx(200.0)
    )

    # 使用is检查对象身份，而不只是内容相等。
    # 这证明两组没有重新构造或偷偷改写输入。
    assert basic_provider.calls[0][0] is request
    assert evidence_provider.calls[0][0] is request
    assert basic_provider.calls[0][1] is evidence
    assert evidence_provider.calls[0][1] is evidence


@pytest.mark.asyncio
async def test_compare_case_rejects_same_prompt_version(
) -> None:
    """版本相同应在产生任何LLM调用前失败。"""

    basic_provider = FakeVersionedProvider(
        prompt_version="diagnosis-evidence-v1",
        draft=make_draft(),
    )
    evidence_provider = FakeVersionedProvider(
        prompt_version="diagnosis-evidence-v1",
        draft=make_draft(),
    )

    with pytest.raises(
        ValueError,
        match="两个Provider必须使用不同Prompt版本",
    ):
        await compare_diagnosis_prompt_case(
            case_id="dpc001",
            request=make_request(),
            evidence=make_evidence(),
            basic_provider=basic_provider,
            evidence_first_provider=(
                evidence_provider
            ),
            clock=FakeClock(),
        )

    assert basic_provider.calls == []
    assert evidence_provider.calls == []


def test_collect_evidence_ids_preserves_first_seen_order(
) -> None:
    """原因和检查中的重复证据编号应按首次出现顺序去重。"""

    draft = make_draft_with_ids(
        cause_ids=["E2", "E1"],
        check_ids=["E1", "E3"],
    )

    assert collect_diagnosis_evidence_ids(
        draft
    ) == (
        "E2",
        "E1",
        "E3",
    )


def test_calculate_metrics_counts_failures_and_whitelist(
) -> None:
    """指标应同时计入成功、未知引用、失败和合法拒答。"""

    cases = [
        make_comparison_case(
            case_id="dpc001",
            basic_result=make_success_run(
                version="diagnosis-basic-v1",
                draft=make_draft(),
                generation_ms=100.0,
            ),
            evidence_first_result=make_success_run(
                version="diagnosis-evidence-v1",
                draft=make_draft(),
                generation_ms=80.0,
            ),
        ),
        make_comparison_case(
            case_id="dpc002",
            basic_result=make_success_run(
                version="diagnosis-basic-v1",
                draft=make_draft_with_ids(
                    cause_ids=["E999"],
                    check_ids=["E999"],
                ),
                generation_ms=200.0,
            ),
            evidence_first_result=make_success_run(
                version="diagnosis-evidence-v1",
                draft=make_draft(),
                generation_ms=100.0,
            ),
        ),
        make_comparison_case(
            case_id="dpc003",
            basic_result=make_failure_run(
                version="diagnosis-basic-v1",
                generation_ms=300.0,
            ),
            evidence_first_result=make_success_run(
                version="diagnosis-evidence-v1",
                draft=make_abstained_draft(),
                generation_ms=120.0,
            ),
        ),
    ]

    basic_metrics = (
        calculate_diagnosis_prompt_metrics(
            cases=cases,
            group="basic",
        )
    )
    evidence_metrics = (
        calculate_diagnosis_prompt_metrics(
            cases=cases,
            group="evidence_first",
        )
    )

    assert basic_metrics.success_count == 2
    assert (
        basic_metrics.invalid_response_count
        == 1
    )
    assert basic_metrics.whitelist_passed_count == 1
    assert basic_metrics.unknown_evidence_id_count == 1
    assert basic_metrics.schema_success_rate == (
        pytest.approx(2 / 3)
    )
    assert (
        basic_metrics.evidence_whitelist_pass_rate
        == pytest.approx(1 / 3)
    )
    assert basic_metrics.average_generation_ms == (
        pytest.approx(200.0)
    )

    assert evidence_metrics.success_count == 3
    assert evidence_metrics.abstained_count == 1
    assert (
        evidence_metrics.whitelist_passed_count
        == 3
    )
    assert (
        evidence_metrics.unknown_evidence_id_count
        == 0
    )
    assert evidence_metrics.average_generation_ms == (
        pytest.approx(100.0)
    )


def test_calculate_metrics_rejects_mixed_versions(
) -> None:
    """同一实验组不能跨案例混用v1和v2。"""

    cases = [
        make_comparison_case(
            case_id="dpc001",
            basic_result=make_success_run(
                version="diagnosis-basic-v1",
                draft=make_draft(),
                generation_ms=100.0,
            ),
            evidence_first_result=make_success_run(
                version="diagnosis-evidence-v1",
                draft=make_draft(),
                generation_ms=100.0,
            ),
        ),
        make_comparison_case(
            case_id="dpc002",
            basic_result=make_success_run(
                version="diagnosis-basic-v2",
                draft=make_draft(),
                generation_ms=100.0,
            ),
            evidence_first_result=make_success_run(
                version="diagnosis-evidence-v1",
                draft=make_draft(),
                generation_ms=100.0,
            ),
        ),
    ]

    with pytest.raises(
        ValueError,
        match="同一实验组必须使用同一个Prompt版本",
    ):
        calculate_diagnosis_prompt_metrics(
            cases=cases,
            group="basic",
        )


def test_build_report_calculates_both_groups(
) -> None:
    """报告构造器应自动计算两组指标并绑定逐案例结果。"""

    case = make_comparison_case(
        case_id="dpc001",
        basic_result=make_success_run(
            version="diagnosis-basic-v1",
            draft=make_draft(),
            generation_ms=120.0,
        ),
        evidence_first_result=make_success_run(
            version="diagnosis-evidence-v1",
            draft=make_draft(),
            generation_ms=90.0,
        ),
    )

    report = (
        build_diagnosis_prompt_comparison_report(
            llm_model="test-model",
            temperature=0.0,
            cases=[case],
        )
    )

    assert report.basic_metrics.case_count == 1
    assert (
        report.evidence_first_metrics.prompt_version
        == "diagnosis-evidence-v1"
    )
    assert report.cases == [case]
