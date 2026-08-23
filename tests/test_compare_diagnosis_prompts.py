"""诊断Prompt真实对照脚本中纯函数部分的离线测试。

本模块不访问真实LLM、Embedding、Chroma或HTTP服务。

测试范围：
1. 命令行默认参数和显式参数是否被正确解析；
2. 非法top_k是否在进入真实检索前被拒绝；
3. Markdown报告是否同时呈现实验限制、两组指标和逐案例结果；
4. JSON与Markdown报告是否写入指定位置；
5. 终端摘要是否准确显示两组核心指标。
"""

import argparse
import json
from pathlib import Path

import pytest

import scripts.compare_diagnosis_prompts as comparison_script

from app.schemas.diagnosis_llm import DiagnosisLLMDraft
from app.schemas.diagnosis_prompt_comparison import (
    DiagnosisPromptComparisonCase,
    DiagnosisPromptRunResult,
)
from app.schemas.diagnostics import DiagnosisRequest
from app.schemas.retrieval import (
    DocumentChunk,
    HybridRetrievedChunk,
)
from app.services.gated_hybrid_retrieval import (
    GatedHybridRetrievalResult,
)
from app.services.hybrid_retrieval_gate import (
    HybridEvidenceGateDecision,
)
from app.services.query_rewriting import (
    RewrittenRetrievalQuery,
)
from app.services.diagnosis_prompt_comparison import (
    build_diagnosis_prompt_comparison_report,
)
from scripts.compare_diagnosis_prompts import (
    DEFAULT_CASE_ID,
    DEFAULT_JSON_OUTPUT,
    DEFAULT_MARKDOWN_OUTPUT,
    DEFAULT_ROBOT_ID,
    DEFAULT_TOP_K,
    parse_args,
    print_summary,
    render_markdown_report,
    run_prompt_comparison_experiment,
    write_reports,
)


# 真实DocumentChunk要求document_id和content_hash均为64位十六进制字符串。
TEST_DOCUMENT_ID = "a" * 64
TEST_CONTENT_HASH = "b" * 64


class FakeDiagnosisRetrievalProvider:
    """记录检索参数并返回预先构造的门控结果。"""

    def __init__(
        self,
        *,
        result: GatedHybridRetrievalResult,
    ) -> None:
        self._result = result

        # 每一项保存query、top_k和retrieval_scope。
        # 测试用它确认编排函数正确构造并传递参数。
        self.calls: list[dict[str, object]] = []

    async def retrieve(
        self,
        *,
        query: str,
        top_k: int = 3,
        retrieval_scope: object = None,
    ) -> GatedHybridRetrievalResult:
        """模拟异步检索，但不调用Embedding或Chroma。"""

        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "retrieval_scope": retrieval_scope,
            }
        )

        return self._result


class FakeVersionedDiagnosisDraftProvider:
    """模拟带版本号的诊断草稿Provider并记录调用。"""

    def __init__(
        self,
        *,
        prompt_version: str,
        draft: DiagnosisLLMDraft,
    ) -> None:
        self._prompt_version = prompt_version
        self._draft = draft

        # 每次调用保存收到的request和evidence对象。
        # 这样不仅能比较内容，还能使用is确认对象相同。
        self.calls: list[
            tuple[
                DiagnosisRequest,
                tuple[HybridRetrievedChunk, ...],
            ]
        ] = []

    @property
    def prompt_version(self) -> str:
        """返回该Fake代表的Prompt版本。"""

        return self._prompt_version

    async def generate_draft(
        self,
        *,
        request: DiagnosisRequest,
        evidence: tuple[
            HybridRetrievedChunk,
            ...,
        ],
    ) -> DiagnosisLLMDraft:
        """记录输入并返回预设的合法草稿。"""

        self.calls.append(
            (request, evidence)
        )

        return self._draft


class FakeAsyncClient:
    """只记录异步close()次数的外部客户端替身。"""

    def __init__(
        self,
        *,
        name: str,
    ) -> None:
        self.name = name
        self.close_count = 0

    async def close(self) -> None:
        """模拟AsyncOpenAI.close()并记录资源释放。"""

        self.close_count += 1


class FakeRuntimeSettings:
    """提供真实运行入口在当前测试中需要的最小配置。"""

    chroma_persist_directory = Path(
        "fake-chroma-data"
    )
    chroma_collection_name = (
        "fake_robot_knowledge"
    )


def make_evidence() -> list[HybridRetrievedChunk]:
    """构造一个真实Schema可接受的Top-1混合检索候选。"""

    return [
        HybridRetrievedChunk(
            chunk=DocumentChunk(
                chunk_id=f"{TEST_DOCUMENT_ID}:000001",
                document_id=TEST_DOCUMENT_ID,
                source_file="robot-faults.txt",
                page_or_section="section: ERR-NET-4001",
                chunk_index=1,
                content_hash=TEST_CONTENT_HASH,
                content=(
                    "通信恢复后不得自动继续旧任务，"
                    "应先核对调度状态。"
                ),
            ),
            rrf_score=0.032,
            rank=1,
            vector_rank=1,
            vector_similarity=0.68,
            keyword_rank=1,
            keyword_score=12.0,
            matched_identifiers=("err-net-4001",),
        )
    ]


def make_draft(
    *,
    evidence_id: str,
) -> DiagnosisLLMDraft:
    """构造一个结构合法、但引用编号可控的诊断草稿。"""

    return DiagnosisLLMDraft(
        status="partial",
        possible_causes=[
            {
                "description": "调度状态可能尚未完成核对",
                "evidence_ids": [evidence_id],
            }
        ],
        next_checks=[
            {
                "description": "核对调度系统中的任务状态",
                "evidence_ids": [evidence_id],
                "risk_level": "low",
                "requires_qualified_person": False,
            }
        ],
        risk_level="low",
        missing_information=["是否已下发恢复命令"],
        abstained=False,
    )


def make_report():
    """构造一份能显示基础组未知引用差异的完整报告。"""

    request = DiagnosisRequest(
        robot_id="diagnostic-robot",
        symptom="网络恢复后机器人仍未继续任务",
        log_excerpt=(
            "ERR-NET-4001 heartbeat timeout exceeded 1500 ms"
        ),
    )

    case = DiagnosisPromptComparisonCase(
        case_id="dpc001",
        request=request,
        evidence=make_evidence(),
        basic_result=DiagnosisPromptRunResult(
            prompt_version="diagnosis-basic-v1",
            outcome="success",
            generation_ms=120.0,

            # E999不在本案例E1白名单中。
            # Schema只检查草稿结构，实验指标负责识别未知引用。
            draft=make_draft(evidence_id="E999"),
            error_detail=None,
        ),
        evidence_first_result=DiagnosisPromptRunResult(
            prompt_version="diagnosis-evidence-v1",
            outcome="success",
            generation_ms=150.0,
            draft=make_draft(evidence_id="E1"),
            error_detail=None,
        ),
    )

    return build_diagnosis_prompt_comparison_report(
        llm_model="fake-diagnosis-model",
        temperature=0.0,
        cases=[case],
    )


def make_gated_result(
    *,
    accepted: bool,
) -> GatedHybridRetrievalResult:
    """构造门控放行或拒绝的完整检索结果。"""

    query_rewrite = RewrittenRetrievalQuery(
        original_query=(
            "网络恢复后机器人仍未继续任务\n"
            "ERR-NET-4001"
        ),
        normalized_query=(
            "网络恢复后机器人仍未继续任务\n"
            "err-net-4001"
        ),
        rewritten_query=(
            "网络恢复后机器人仍未继续任务 "
            "err-net-4001 network recovery"
        ),
        added_terms=("network recovery",),
        applied_rule_ids=(
            "identifier-err-net-4001",
        ),
        rewrite_version="deterministic-v1",
    )

    if accepted:
        evidence = tuple(make_evidence())
        top_candidate = evidence[0]

        decision = HybridEvidenceGateDecision(
            accepted=True,
            reason="exact_identifier_support",
            policy_version=(
                "hybrid-evidence-gate-v1"
            ),
            evaluated_candidate_count=1,
            top_rrf_score=(
                top_candidate.rrf_score
            ),
            top_vector_similarity=(
                top_candidate.vector_similarity
            ),
            max_vector_similarity=(
                top_candidate.vector_similarity
            ),
            dual_path_candidate_count=1,
            identifier_candidate_count=1,
            model_lexical_candidate_count=0,
            supporting_chunk_ids=(
                top_candidate.chunk.chunk_id,
            ),
        )
    else:
        evidence = ()

        decision = HybridEvidenceGateDecision(
            accepted=False,
            reason="no_candidates",
            policy_version=(
                "hybrid-evidence-gate-v1"
            ),
            evaluated_candidate_count=0,
            top_rrf_score=None,
            top_vector_similarity=None,
            max_vector_similarity=None,
            dual_path_candidate_count=0,
            identifier_candidate_count=0,
            model_lexical_candidate_count=0,
            supporting_chunk_ids=(),
        )

    return GatedHybridRetrievalResult(
        query_rewrite=query_rewrite,
        retrieved_chunks=evidence,
        decision=decision,
    )


def install_fake_runtime_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    """把真实入口使用的外部工厂替换成可审计Fake。"""

    settings = FakeRuntimeSettings()
    embedding_client = FakeAsyncClient(
        name="embedding-client",
    )
    llm_client = FakeAsyncClient(
        name="llm-client",
    )
    fake_vector_store = object()
    retrieval_provider = (
        FakeDiagnosisRetrievalProvider(
            result=make_gated_result(
                accepted=True,
            ),
        )
    )

    # 普通列表用于记录工厂收到的关键参数。
    vector_store_calls: list[
        dict[str, object]
    ] = []
    retrieval_factory_calls: list[
        dict[str, object]
    ] = []
    draft_provider_calls: list[
        dict[str, object]
    ] = []

    def fake_vector_store_factory(
        *,
        persist_directory: Path,
        collection_name: str,
        embedding_model: str,
    ) -> object:
        """记录Chroma构造参数但不打开真实数据库。"""

        vector_store_calls.append(
            {
                "persist_directory": (
                    persist_directory
                ),
                "collection_name": (
                    collection_name
                ),
                "embedding_model": (
                    embedding_model
                ),
            }
        )

        return fake_vector_store

    def fake_retrieval_factory(
        *,
        embedding_client: object,
        vector_store: object,
        settings: object,
    ) -> FakeDiagnosisRetrievalProvider:
        """记录共享检索工厂参数并返回Fake检索器。"""

        retrieval_factory_calls.append(
            {
                "embedding_client": (
                    embedding_client
                ),
                "vector_store": vector_store,
                "settings": settings,
            }
        )

        return retrieval_provider

    def fake_draft_provider_factory(
        *,
        client: object,
        model: str,
        prompt_variant: object,
    ) -> FakeVersionedDiagnosisDraftProvider:
        """记录LLM、模型和Prompt变体并返回Fake Provider。"""

        prompt_version = (
            prompt_variant.version
        )
        draft_provider_calls.append(
            {
                "client": client,
                "model": model,
                "prompt_version": (
                    prompt_version
                ),
            }
        )

        return FakeVersionedDiagnosisDraftProvider(
            prompt_version=prompt_version,
            draft=make_draft(
                evidence_id="E1",
            ),
        )

    # monkeypatch.setattr()只在当前测试期间替换属性，
    # 测试结束后pytest会自动恢复真实对象。
    monkeypatch.setattr(
        comparison_script,
        "get_settings",
        lambda: settings,
    )
    monkeypatch.setattr(
        comparison_script,
        "get_embedding_model",
        lambda received: "fake-embedding-model",
    )
    monkeypatch.setattr(
        comparison_script,
        "get_llm_model",
        lambda received: "fake-llm-model",
    )
    monkeypatch.setattr(
        comparison_script,
        "create_embedding_client",
        lambda received: embedding_client,
    )
    monkeypatch.setattr(
        comparison_script,
        "create_llm_client",
        lambda received: llm_client,
    )
    monkeypatch.setattr(
        comparison_script,
        "ChromaVectorStore",
        fake_vector_store_factory,
    )
    monkeypatch.setattr(
        comparison_script,
        "build_diagnosis_retrieval_provider",
        fake_retrieval_factory,
    )
    monkeypatch.setattr(
        comparison_script,
        "OpenAIDiagnosisDraftProvider",
        fake_draft_provider_factory,
    )

    return {
        "settings": settings,
        "embedding_client": embedding_client,
        "llm_client": llm_client,
        "vector_store": fake_vector_store,
        "retrieval_provider": (
            retrieval_provider
        ),
        "vector_store_calls": (
            vector_store_calls
        ),
        "retrieval_factory_calls": (
            retrieval_factory_calls
        ),
        "draft_provider_calls": (
            draft_provider_calls
        ),
    }


def test_parse_args_uses_documented_defaults() -> None:
    """空参数应生成可直接运行的单案例冒烟实验配置。"""

    args = parse_args([])

    assert args.case_id == DEFAULT_CASE_ID
    assert args.robot_id == DEFAULT_ROBOT_ID
    assert args.top_k == DEFAULT_TOP_K
    assert args.json_output == DEFAULT_JSON_OUTPUT
    assert args.markdown_output == DEFAULT_MARKDOWN_OUTPUT


def test_parse_args_preserves_explicit_values() -> None:
    """显式命令行值应原样进入真实检索和报告路径配置。"""

    args = parse_args(
        [
            "--case-id",
            "dpc009",
            "--robot-id",
            "robot-009",
            "--symptom",
            "机器人无法继续任务",
            "--log-excerpt",
            "ERR-NET-4001",
            "--top-k",
            "5",
            "--json-output",
            "docs/custom.json",
            "--markdown-output",
            "docs/custom.md",
        ]
    )

    assert args.case_id == "dpc009"
    assert args.robot_id == "robot-009"
    assert args.symptom == "机器人无法继续任务"
    assert args.log_excerpt == "ERR-NET-4001"
    assert args.top_k == 5
    assert args.json_output == Path("docs/custom.json")
    assert args.markdown_output == Path("docs/custom.md")


def test_parse_args_rejects_non_positive_top_k() -> None:
    """top_k=0必须由argparse拒绝，不能进入真实Chroma检索。"""

    with pytest.raises(SystemExit):
        parse_args(["--top-k", "0"])


def test_render_markdown_report_contains_auditable_comparison() -> None:
    """Markdown应呈现实验限制、两组指标、证据和未知引用。"""

    markdown = render_markdown_report(make_report())

    assert "# 诊断 Prompt A/B 对照实验" in markdown
    assert "单案例" in markdown
    assert "不能代表统计显著性" in markdown
    assert "diagnosis-basic-v1" in markdown
    assert "diagnosis-evidence-v1" in markdown
    assert "dpc001" in markdown
    assert "ERR-NET-4001" in markdown
    assert "E999" in markdown
    assert "robot-faults.txt" in markdown

    # 用户要求Markdown报告不显示生成时间。
    assert "生成时间" not in markdown


def test_write_reports_creates_parent_directories(
    tmp_path: Path,
) -> None:
    """写入函数应创建父目录并生成可解析JSON与UTF-8 Markdown。"""

    report = make_report()
    json_path = tmp_path / "nested" / "comparison.json"
    markdown_path = tmp_path / "nested" / "comparison.md"

    write_reports(
        report=report,
        json_path=json_path,
        markdown_path=markdown_path,
    )

    json_data = json.loads(
        json_path.read_text(encoding="utf-8")
    )
    markdown = markdown_path.read_text(encoding="utf-8")

    assert json_data["llm_model"] == "fake-diagnosis-model"
    assert json_data["cases"][0]["case_id"] == "dpc001"
    assert "generated_at" not in json_data
    assert "诊断 Prompt A/B 对照实验" in markdown


def test_print_summary_reports_both_groups(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """终端摘要应显示两组结构成功率和引用白名单通过率。"""

    print_summary(
        report=make_report(),
        json_path=Path("docs/comparison.json"),
        markdown_path=Path("docs/comparison.md"),
    )

    output = capsys.readouterr().out

    assert "诊断Prompt对照实验完成" in output
    assert "基础Prompt" in output
    assert "证据优先Prompt" in output
    assert "结构成功率" in output
    assert "引用白名单通过率" in output
    # print_summary()会调用Path.resolve()输出绝对路径。
    # 不能在测试中硬编码“/”分隔符：
    # Windows使用“\\”，Linux和macOS使用“/”。
    expected_json_path = str(
        Path("docs/comparison.json").resolve()
    )
    expected_markdown_path = str(
        Path("docs/comparison.md").resolve()
    )

    assert expected_json_path in output
    assert expected_markdown_path in output


@pytest.mark.asyncio
async def test_run_prompt_comparison_experiment_uses_same_gated_evidence(
) -> None:
    """门控放行后两个Prompt必须收到同一请求和证据快照。"""

    request = DiagnosisRequest(
        robot_id="diagnostic-robot",
        symptom="网络恢复后机器人仍未继续任务",
        log_excerpt="ERR-NET-4001 heartbeat recovered",
    )
    gated_result = make_gated_result(
        accepted=True,
    )
    retrieval_provider = (
        FakeDiagnosisRetrievalProvider(
            result=gated_result,
        )
    )
    basic_provider = (
        FakeVersionedDiagnosisDraftProvider(
            prompt_version="diagnosis-basic-v1",
            draft=make_draft(
                evidence_id="E999",
            ),
        )
    )
    evidence_first_provider = (
        FakeVersionedDiagnosisDraftProvider(
            prompt_version=(
                "diagnosis-evidence-v1"
            ),
            draft=make_draft(
                evidence_id="E1",
            ),
        )
    )

    report = await run_prompt_comparison_experiment(
        case_id="dpc001",
        request=request,
        top_k=3,
        retrieval_provider=retrieval_provider,
        basic_provider=basic_provider,
        evidence_first_provider=(
            evidence_first_provider
        ),
        llm_model="fake-diagnosis-model",
        temperature=0.0,
    )

    # 编排器应只执行一次检索。
    assert retrieval_provider.calls == [
        {
            "query": (
                "网络恢复后机器人仍未继续任务\n"
                "ERR-NET-4001 heartbeat recovered"
            ),
            "top_k": 3,
            "retrieval_scope": None,
        }
    ]

    assert len(basic_provider.calls) == 1
    assert len(
        evidence_first_provider.calls
    ) == 1

    basic_request, basic_evidence = (
        basic_provider.calls[0]
    )
    evidence_request, evidence_evidence = (
        evidence_first_provider.calls[0]
    )

    # is比较对象身份，而不只是字段值相等。
    # 这证明脚本没有为两个组重新检索或改写证据。
    assert basic_request is request
    assert evidence_request is request
    assert (
        basic_evidence
        is gated_result.retrieved_chunks
    )
    assert (
        evidence_evidence
        is gated_result.retrieved_chunks
    )

    # 基础组的E999结构合法但不在E1白名单；
    # 证据优先组的E1通过白名单检查。
    assert (
        report.basic_metrics
        .evidence_whitelist_pass_rate
        == 0.0
    )
    assert (
        report.evidence_first_metrics
        .evidence_whitelist_pass_rate
        == 1.0
    )


@pytest.mark.asyncio
async def test_run_prompt_comparison_experiment_skips_llm_when_gate_rejects(
) -> None:
    """门控拒绝时实验必须停止，两个Prompt都不能调用LLM。"""

    request = DiagnosisRequest(
        robot_id="diagnostic-robot",
        symptom="完全无关的未知问题",
        log_excerpt="no matching evidence",
    )
    retrieval_provider = (
        FakeDiagnosisRetrievalProvider(
            result=make_gated_result(
                accepted=False,
            ),
        )
    )
    basic_provider = (
        FakeVersionedDiagnosisDraftProvider(
            prompt_version="diagnosis-basic-v1",
            draft=make_draft(
                evidence_id="E1",
            ),
        )
    )
    evidence_first_provider = (
        FakeVersionedDiagnosisDraftProvider(
            prompt_version=(
                "diagnosis-evidence-v1"
            ),
            draft=make_draft(
                evidence_id="E1",
            ),
        )
    )

    with pytest.raises(
        RuntimeError,
        match="no_candidates",
    ):
        await run_prompt_comparison_experiment(
            case_id="dpc002",
            request=request,
            top_k=3,
            retrieval_provider=(
                retrieval_provider
            ),
            basic_provider=basic_provider,
            evidence_first_provider=(
                evidence_first_provider
            ),
            llm_model="fake-diagnosis-model",
            temperature=0.0,
        )

    # 检索发生一次，但两个Provider均保持零调用。
    assert len(retrieval_provider.calls) == 1
    assert basic_provider.calls == []
    assert evidence_first_provider.calls == []


@pytest.mark.asyncio
async def test_run_real_prompt_comparison_assembles_and_closes_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """真实入口应装配共享资源、运行实验并关闭两个客户端。"""

    runtime = install_fake_runtime_dependencies(
        monkeypatch
    )
    args = parse_args(
        [
            "--case-id",
            "dpc003",
            "--robot-id",
            "robot-runtime",
            "--symptom",
            "网络恢复后仍未继续任务",
            "--log-excerpt",
            "ERR-NET-4001 recovered",
            "--top-k",
            "3",
        ]
    )

    report = (
        await comparison_script
        .run_real_prompt_comparison(args)
    )

    assert runtime["vector_store_calls"] == [
        {
            "persist_directory": Path(
                "fake-chroma-data"
            ),
            "collection_name": (
                "fake_robot_knowledge"
            ),
            "embedding_model": (
                "fake-embedding-model"
            ),
        }
    ]
    assert runtime[
        "retrieval_factory_calls"
    ] == [
        {
            "embedding_client": runtime[
                "embedding_client"
            ],
            "vector_store": runtime[
                "vector_store"
            ],
            "settings": runtime["settings"],
        }
    ]

    draft_provider_calls = runtime[
        "draft_provider_calls"
    ]
    assert isinstance(
        draft_provider_calls,
        list,
    )
    assert [
        item["prompt_version"]
        for item in draft_provider_calls
    ] == [
        "diagnosis-basic-v1",
        "diagnosis-evidence-v1",
    ]
    assert all(
        item["client"]
        is runtime["llm_client"]
        for item in draft_provider_calls
    )
    assert all(
        item["model"] == "fake-llm-model"
        for item in draft_provider_calls
    )

    embedding_client = runtime[
        "embedding_client"
    ]
    llm_client = runtime["llm_client"]
    assert isinstance(
        embedding_client,
        FakeAsyncClient,
    )
    assert isinstance(
        llm_client,
        FakeAsyncClient,
    )
    assert embedding_client.close_count == 1
    assert llm_client.close_count == 1

    assert report.llm_model == "fake-llm-model"
    assert report.cases[0].case_id == "dpc003"
    assert (
        report.cases[0].request.robot_id
        == "robot-runtime"
    )


@pytest.mark.asyncio
async def test_run_real_prompt_comparison_closes_clients_after_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """实验阶段抛出异常时finally仍必须关闭两个外部客户端。"""

    runtime = install_fake_runtime_dependencies(
        monkeypatch
    )

    async def fail_experiment(
        **kwargs: object,
    ) -> None:
        """模拟检索或Prompt实验阶段的非预期失败。"""

        raise RuntimeError(
            "forced experiment failure"
        )

    monkeypatch.setattr(
        comparison_script,
        "run_prompt_comparison_experiment",
        fail_experiment,
    )

    with pytest.raises(
        RuntimeError,
        match="forced experiment failure",
    ):
        await comparison_script.run_real_prompt_comparison(
            parse_args([])
        )

    embedding_client = runtime[
        "embedding_client"
    ]
    llm_client = runtime["llm_client"]
    assert isinstance(
        embedding_client,
        FakeAsyncClient,
    )
    assert isinstance(
        llm_client,
        FakeAsyncClient,
    )
    assert embedding_client.close_count == 1
    assert llm_client.close_count == 1


def test_main_runs_experiment_then_writes_and_prints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同步CLI入口应依次运行异步实验、写报告并打印摘要。"""

    report = make_report()
    observed: dict[str, object] = {}

    async def fake_run_real(
        args: argparse.Namespace,
    ):
        """记录main()解析后的参数并返回固定报告。"""

        observed["args"] = args
        return report

    def fake_write_reports(
        *,
        report: object,
        json_path: Path,
        markdown_path: Path,
    ) -> None:
        """记录main()传给报告写入函数的对象。"""

        observed["write"] = (
            report,
            json_path,
            markdown_path,
        )

    def fake_print_summary(
        *,
        report: object,
        json_path: Path,
        markdown_path: Path,
    ) -> None:
        """记录main()传给终端摘要函数的对象。"""

        observed["print"] = (
            report,
            json_path,
            markdown_path,
        )

    monkeypatch.setattr(
        comparison_script,
        "run_real_prompt_comparison",
        fake_run_real,
    )
    monkeypatch.setattr(
        comparison_script,
        "write_reports",
        fake_write_reports,
    )
    monkeypatch.setattr(
        comparison_script,
        "print_summary",
        fake_print_summary,
    )

    comparison_script.main(
        [
            "--case-id",
            "dpc004",
            "--json-output",
            "docs/runtime.json",
            "--markdown-output",
            "docs/runtime.md",
        ]
    )

    parsed_args = observed["args"]
    assert isinstance(
        parsed_args,
        argparse.Namespace,
    )
    assert parsed_args.case_id == "dpc004"

    expected_arguments = (
        report,
        Path("docs/runtime.json"),
        Path("docs/runtime.md"),
    )
    assert observed["write"] == (
        expected_arguments
    )
    assert observed["print"] == (
        expected_arguments
    )
