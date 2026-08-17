"""RAG与裸LLM对比实验编排核心的离线测试。"""

# json.loads()用于检查HTTP请求体，
# 也用于验证写出的JSON报告可以重新解析。
import json

# datetime用于构造固定的报告生成时间。
#
# 测试使用固定时间，不使用当前时间，
# 可以让测试结果保持稳定。
from datetime import datetime

# Path用于标注pytest临时目录参数的类型。
from pathlib import Path

# AsyncMock用于模拟需要await的异步函数；
# MagicMock用于模拟普通同步函数和计时器。
from unittest.mock import (
    AsyncMock,
    MagicMock,
)

import httpx
import pytest

# Settings用于构造完全独立于真实.env的测试配置。
from app.config import Settings

from app.schemas.evaluation import (
    ExpectedEvidence,
    GoldQuestion,
)
from app.schemas.knowledge_query import (
    KnowledgeCitation,
    KnowledgeQueryResponse,
)
from app.schemas.rag_comparison import (
    BareLLMResult,
    RagVsBareLLMCase,
    RagVsBareLLMReport,
)

# 使用生产代码中的真实Prompt构造报告，
# 防止测试中的Prompt副本与实际实现发生漂移。
from app.services.bare_llm import (
    BARE_LLM_SYSTEM_PROMPT,
)

# 导入模块对象，用于替换其中的perf_counter。
import scripts.compare_rag_vs_bare_llm as comparison_module

from scripts.compare_rag_vs_bare_llm import (
    RagComparisonRequestError,
    compare_rag_and_bare,
    parse_args,
    print_summary,
    render_markdown_blockquote,
    render_markdown_report,
    run_real_comparison,
    select_gold_questions,
    write_reports,
)


API_URL = (
    "http://knowledge.test"
    "/api/v1/knowledge/query"
)


class FakeBareLLMProvider:
    """记录问题并返回预设结果的裸LLM Fake。"""

    def __init__(
        self,
        answers: dict[str, str],
    ) -> None:
        self._answers = answers

        # 保存所有调用过的问题及其顺序。
        self.calls: list[str] = []

    async def generate_answer(
        self,
        *,
        question: str,
    ) -> BareLLMResult:
        """返回预设裸LLM回答，不访问网络。"""

        self.calls.append(question)

        return BareLLMResult(
            answer=self._answers[question],
            generation_ms=250.0,
        )


class FakeClosableLLMClient:
    """记录异步close()调用次数的假LLM客户端。"""

    def __init__(
        self,
    ) -> None:
        self.close_calls = 0

    async def close(
        self,
    ) -> None:
        """模拟关闭SDK内部HTTP连接池。"""

        self.close_calls += 1


def make_q014_gold() -> GoldQuestion:
    """创建有答案的多跳故障问题。"""

    return GoldQuestion(
        question_id="q014",
        question=(
            "应怎样安全验证驱动电机过温故障，"
            "从触发到恢复需要满足哪些条件？"
        ),
        question_type="multi_hop",
        answerable=True,
        reference_answer=(
            "应注入模拟温度信号；高于85°C"
            "并持续10秒后触发ERR-DRV-3005。"
        ),
        expected_evidence=[
            ExpectedEvidence(
                source_file=(
                    "京东无人仓场景-仓储机器人"
                    "测试规程与判定标准.md"
                ),
                page_or_section=(
                    "section: 12.3 TEST-DRV-001 "
                    "驱动电机过温保护"
                ),
            ),
            ExpectedEvidence(
                source_file=(
                    "仓储机器人故障说明.txt"
                ),
                page_or_section=(
                    "section: ERR-DRV-3005"
                ),
            ),
        ],
        tags=[
            "motor",
            "temperature",
            "multi-hop",
        ],
        notes="RAG与裸LLM有答案对照问题。",
    )


def make_q020_gold() -> GoldQuestion:
    """创建知识库无答案的实时遥测问题。"""

    return GoldQuestion(
        question_id="q020",
        question=(
            "robot-001当前在哪里，"
            "剩余电量是多少？"
        ),
        question_type="unanswerable",
        answerable=False,
        reference_answer=None,
        expected_evidence=[],
        tags=[
            "abstention",
            "telemetry",
        ],
        notes="静态知识库不包含实时遥测。",
    )


def make_q014_rag_payload(
) -> dict[str, object]:
    """创建q014对应的合法RAG API响应。"""

    procedure_document_id = "a" * 64
    fault_document_id = "b" * 64

    response = KnowledgeQueryResponse(
        answer=(
            "应注入模拟温度信号；高于85°C"
            "并持续10秒后触发ERR-DRV-3005，"
            "完成人工检查前不得复位。"
        ),
        citations=[
            KnowledgeCitation(
                chunk_id=(
                    f"{procedure_document_id}:000028"
                ),
                document_id=procedure_document_id,
                source_file=(
                    "京东无人仓场景-仓储机器人"
                    "测试规程与判定标准.md"
                ),
                page_or_section=(
                    "section: 12.3 TEST-DRV-001 "
                    "驱动电机过温保护"
                ),
                chunk_index=28,
                rank=1,
                similarity=0.62,
                excerpt=(
                    "通过测试接口注入高于85°C的"
                    "模拟温度值并保持10秒。"
                ),
            ),
            KnowledgeCitation(
                chunk_id=(
                    f"{fault_document_id}:000002"
                ),
                document_id=fault_document_id,
                source_file=(
                    "仓储机器人故障说明.txt"
                ),
                page_or_section=(
                    "section: ERR-DRV-3005"
                ),
                chunk_index=2,
                rank=2,
                similarity=0.61,
                excerpt=(
                    "温度恢复并完成人工检查前"
                    "不得复位。"
                ),
            ),
        ],
        retrieval_ms=180.0,
        abstained=False,
    )

    return response.model_dump(
        mode="json"
    )


def make_q020_rag_payload(
) -> dict[str, object]:
    """创建q020对应的合法RAG拒答响应。"""

    response = KnowledgeQueryResponse(
        answer=(
            "知识库没有足够证据回答该问题。"
        ),
        citations=[],
        retrieval_ms=80.0,
        abstained=True,
    )

    return response.model_dump(
        mode="json"
    )


def make_fake_bare_provider(
) -> FakeBareLLMProvider:
    """创建能够回答q014和q020的Fake Provider。"""

    return FakeBareLLMProvider(
        answers={
            make_q014_gold().question: (
                "一般应监控电机温度并在过热时停机。"
            ),
            make_q020_gold().question: (
                "我无法访问robot-001的实时状态。"
            ),
        }
    )


def make_comparison_report(
) -> RagVsBareLLMReport:
    """创建包含可回答题和无答案题的固定对照报告。"""

    q014 = make_q014_gold()
    q020 = make_q020_gold()

    return RagVsBareLLMReport(
        # 使用固定且带时区的时间，
        # 避免测试依赖当前机器时间。
        generated_at=datetime.fromisoformat(
            "2026-08-15T10:00:00+08:00"
        ),
        rag_api_url=API_URL,
        llm_model="test-llm",
        embedding_model="embedding-3",
        collection_name=(
            "robot_knowledge_v4"
        ),
        top_k=3,
        similarity_threshold=0.60,

        # 保存生产代码实际使用的裸LLM Prompt。
        bare_system_prompt=(
            BARE_LLM_SYSTEM_PROMPT
        ),
        cases=[
            RagVsBareLLMCase(
                gold_question=q014,

                # model_validate()让预设字典重新经过
                # KnowledgeQueryResponse契约校验。
                rag_response=(
                    KnowledgeQueryResponse
                    .model_validate(
                        make_q014_rag_payload()
                    )
                ),
                rag_total_ms=500.0,
                bare_result=BareLLMResult(
                    answer=(
                        "一般应监控电机温度，"
                        "并在过热时停机。"
                    ),
                    generation_ms=250.0,
                ),
            ),
            RagVsBareLLMCase(
                gold_question=q020,
                rag_response=(
                    KnowledgeQueryResponse
                    .model_validate(
                        make_q020_rag_payload()
                    )
                ),
                rag_total_ms=100.0,
                bare_result=BareLLMResult(
                    answer=(
                        "我无法访问robot-001的"
                        "实时位置和电量。"
                    ),
                    generation_ms=200.0,
                ),
            ),
        ],
    )


def make_test_settings(
) -> Settings:
    """创建不读取真实.env的对照实验配置。"""

    return Settings(
        # _env_file=None禁止BaseSettings读取项目.env。
        #
        # 这样测试结果不会依赖开发者电脑上的
        # API Key、模型名或Collection配置。
        _env_file=None,
        llm_model="test-llm",
        embedding_model="embedding-3",
        chroma_collection_name=(
            "robot_knowledge_v4"
        ),
        rag_similarity_threshold=0.60,
    )


def test_select_gold_questions_preserves_requested_order(
) -> None:
    """选题结果应遵守question_ids指定的顺序。"""

    questions = [
        make_q014_gold(),
        make_q020_gold(),
    ]

    selected = select_gold_questions(
        questions=questions,
        question_ids=[
            "q020",
            "q014",
        ],
    )

    assert [
        question.question_id
        for question in selected
    ] == [
        "q020",
        "q014",
    ]


def test_select_gold_questions_rejects_missing_id(
) -> None:
    """请求不存在的题号时应在实验开始前失败。"""

    with pytest.raises(
        ValueError,
        match="q999",
    ):
        select_gold_questions(
            questions=[
                make_q014_gold(),
            ],
            question_ids=[
                "q999",
            ],
        )


def test_select_gold_questions_rejects_duplicate_id(
) -> None:
    """同一道题不能在一次实验中被重复选择。"""

    with pytest.raises(
        ValueError,
        match="不能重复",
    ):
        select_gold_questions(
            questions=[
                make_q014_gold(),
            ],
            question_ids=[
                "q014",
                "q014",
            ],
        )


@pytest.mark.asyncio
async def test_compare_builds_report_from_both_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """两道题应分别执行RAG和裸LLM并生成完整报告。"""

    q014 = make_q014_gold()
    q020 = make_q020_gold()

    received_requests: list[
        tuple[str, str, dict[str, object]]
    ] = []

    def handle_request(
        request: httpx.Request,
    ) -> httpx.Response:
        """根据问题返回预设RAG响应。"""

        request_body = json.loads(
            request.content.decode("utf-8")
        )

        received_requests.append(
            (
                request.method,
                request.url.path,
                request_body,
            )
        )

        if (
            request_body["question"]
            == q014.question
        ):
            return httpx.Response(
                status_code=200,
                headers={
                    "X-Request-ID": "request-q014",
                },
                json=make_q014_rag_payload(),
                request=request,
            )

        if (
            request_body["question"]
            == q020.question
        ):
            return httpx.Response(
                status_code=200,
                headers={
                    "X-Request-ID": "request-q020",
                },
                json=make_q020_rag_payload(),
                request=request,
            )

        return httpx.Response(
            status_code=400,
            json={
                "detail": "unexpected question",
            },
            request=request,
        )

    transport = httpx.MockTransport(
        handle_request
    )

    # 两次RAG请求分别读取开始、结束时间：
    #
    # q014：(10.5 - 10.0) × 1000 = 500ms
    # q020：(20.1 - 20.0) × 1000 = 100ms
    mock_clock = MagicMock(
        side_effect=[
            10.0,
            10.5,
            20.0,
            20.1,
        ]
    )

    monkeypatch.setattr(
        comparison_module,
        "perf_counter",
        mock_clock,
    )

    bare_provider = (
        make_fake_bare_provider()
    )

    async with httpx.AsyncClient(
        transport=transport,
    ) as client:
        report = await compare_rag_and_bare(
            questions=[
                q014,
                q020,
            ],
            rag_client=client,
            bare_provider=bare_provider,
            rag_api_url=API_URL,
            llm_model="test-llm",
            embedding_model="embedding-3",
            collection_name=(
                "robot_knowledge_v4"
            ),
            top_k=3,
            similarity_threshold=0.60,
        )

    assert [
        case.gold_question.question_id
        for case in report.cases
    ] == [
        "q014",
        "q020",
    ]

    q014_case = report.cases[0]
    q020_case = report.cases[1]

    # q014的RAG响应有两条真实结构引用。
    assert q014_case.rag_response.abstained is False
    assert len(
        q014_case.rag_response.citations
    ) == 2
    assert q014_case.rag_total_ms == pytest.approx(
        500.0
    )

    # q020的RAG响应必须正确拒答。
    assert q020_case.rag_response.abstained is True
    assert q020_case.rag_response.citations == []
    assert q020_case.rag_total_ms == pytest.approx(
        100.0
    )

    # 两道题都必须以原始问题文本调用裸LLM。
    assert bare_provider.calls == [
        q014.question,
        q020.question,
    ]

    assert (
        q014_case.bare_result.answer
        == "一般应监控电机温度并在过热时停机。"
    )

    # 验证真正发送给RAG API的HTTP请求。
    assert received_requests == [
        (
            "POST",
            "/api/v1/knowledge/query",
            {
                "question": q014.question,
                "top_k": 3,
            },
        ),
        (
            "POST",
            "/api/v1/knowledge/query",
            {
                "question": q020.question,
                "top_k": 3,
            },
        ),
    ]

    assert report.llm_model == "test-llm"
    assert (
        report.embedding_model
        == "embedding-3"
    )
    assert (
        report.collection_name
        == "robot_knowledge_v4"
    )
    assert report.similarity_threshold == 0.60
    assert mock_clock.call_count == 4


@pytest.mark.asyncio
async def test_compare_rejects_metadata_before_requests(
) -> None:
    """无效元数据必须在任何HTTP或裸LLM调用前失败。"""

    received_requests: list[str] = []

    def handle_request(
        request: httpx.Request,
    ) -> httpx.Response:
        received_requests.append(
            request.url.path
        )

        return httpx.Response(
            status_code=500,
            request=request,
        )

    bare_provider = (
        make_fake_bare_provider()
    )

    transport = httpx.MockTransport(
        handle_request
    )

    async with httpx.AsyncClient(
        transport=transport,
    ) as client:
        with pytest.raises(
            ValueError,
            match="llm_model",
        ):
            await compare_rag_and_bare(
                questions=[
                    make_q014_gold(),
                ],
                rag_client=client,
                bare_provider=bare_provider,
                rag_api_url=API_URL,
                llm_model="   ",
                embedding_model="embedding-3",
                collection_name=(
                    "robot_knowledge_v4"
                ),
                top_k=3,
                similarity_threshold=0.60,
            )

    assert received_requests == []
    assert bare_provider.calls == []


@pytest.mark.asyncio
async def test_compare_reports_http_error_with_request_id(
) -> None:
    """RAG API返回502时应保留题号和Request ID。"""

    q014 = make_q014_gold()

    def handle_request(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            status_code=502,
            headers={
                "X-Request-ID": "request-failed",
            },
            json={
                "error": {
                    "code": "llm_upstream_error",
                }
            },
            request=request,
        )

    transport = httpx.MockTransport(
        handle_request
    )

    bare_provider = (
        make_fake_bare_provider()
    )

    async with httpx.AsyncClient(
        transport=transport,
    ) as client:
        with pytest.raises(
            RagComparisonRequestError,
        ) as exc_info:
            await compare_rag_and_bare(
                questions=[
                    q014,
                ],
                rag_client=client,
                bare_provider=bare_provider,
                rag_api_url=API_URL,
                llm_model="test-llm",
                embedding_model="embedding-3",
                collection_name=(
                    "robot_knowledge_v4"
                ),
                top_k=3,
                similarity_threshold=0.60,
            )

    error_message = str(
        exc_info.value
    )

    assert "q014" in error_message
    assert "502" in error_message
    assert (
        "request-failed"
        in error_message
    )

    # RAG已经失败，不应继续调用裸LLM。
    assert bare_provider.calls == []


@pytest.mark.asyncio
async def test_compare_rejects_invalid_rag_contract(
) -> None:
    """HTTP 200但响应字段不完整时实验必须失败。"""

    def handle_request(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            headers={
                "X-Request-ID": "request-invalid",
            },
            json={
                # 缺少citations、retrieval_ms和abstained。
                "answer": "字段不完整",
            },
            request=request,
        )

    transport = httpx.MockTransport(
        handle_request
    )

    async with httpx.AsyncClient(
        transport=transport,
    ) as client:
        with pytest.raises(
            RagComparisonRequestError,
            match="KnowledgeQueryResponse",
        ):
            await compare_rag_and_bare(
                questions=[
                    make_q014_gold(),
                ],
                rag_client=client,
                bare_provider=(
                    make_fake_bare_provider()
                ),
                rag_api_url=API_URL,
                llm_model="test-llm",
                embedding_model="embedding-3",
                collection_name=(
                    "robot_knowledge_v4"
                ),
                top_k=3,
                similarity_threshold=0.60,
            )


@pytest.mark.asyncio
async def test_compare_converts_http_timeout(
) -> None:
    """HTTPX超时应转换成实验层异常并保留题号。"""

    q014 = make_q014_gold()

    def handle_request(
        request: httpx.Request,
    ) -> httpx.Response:
        raise httpx.ReadTimeout(
            "test timeout",
            request=request,
        )

    transport = httpx.MockTransport(
        handle_request
    )

    bare_provider = (
        make_fake_bare_provider()
    )

    async with httpx.AsyncClient(
        transport=transport,
    ) as client:
        with pytest.raises(
            RagComparisonRequestError,
            match="q014",
        ):
            await compare_rag_and_bare(
                questions=[
                    q014,
                ],
                rag_client=client,
                bare_provider=bare_provider,
                rag_api_url=API_URL,
                llm_model="test-llm",
                embedding_model="embedding-3",
                collection_name=(
                    "robot_knowledge_v4"
                ),
                top_k=3,
                similarity_threshold=0.60,
            )

    assert bare_provider.calls == []


def test_render_markdown_blockquote_preserves_lines(
) -> None:
    """多行文本和中间空行应正确转换成Markdown引用块。"""

    result = render_markdown_blockquote(
        "第一行\n\n第二行"
    )

    assert result == (
        "> 第一行\n"
        ">\n"
        "> 第二行"
    )


def test_render_markdown_blockquote_rejects_blank_text(
) -> None:
    """全空白文本不能生成没有内容的报告区块。"""

    with pytest.raises(
        ValueError,
        match="不能为空",
    ):
        render_markdown_blockquote(
            "   \n\t  "
        )


def test_render_markdown_report_contains_both_groups(
) -> None:
    """Markdown应完整展示Gold、RAG和裸LLM两组结果。"""

    report = make_comparison_report()

    markdown = render_markdown_report(
        report
    )

    # 验证报告标题与可复现配置。
    assert (
        "# RAG 与裸 LLM 对照实验"
        in markdown
    )
    assert "test-llm" in markdown
    assert "embedding-3" in markdown
    assert (
        "robot_knowledge_v4"
        in markdown
    )
    assert "0.600" in markdown

    # 两种问题都必须进入报告。
    assert (
        "### q014 — multi_hop"
        in markdown
    )
    assert (
        "### q020 — unanswerable"
        in markdown
    )

    # q014必须包含问题、参考信息、
    # RAG回答和两条可追溯引用。
    assert (
        "应怎样安全验证驱动电机过温故障"
        in markdown
    )
    assert "ERR-DRV-3005" in markdown
    assert "Top-1" in markdown
    assert "Top-2" in markdown
    assert (
        "仓储机器人故障说明.txt"
        in markdown
    )

    # 两种耗时口径都必须显示。
    assert (
        "API端到端耗时：500.000 ms"
        in markdown
    )
    assert (
        "响应记录的检索耗时："
        "180.000 ms"
        in markdown
    )
    assert (
        "生成耗时：250.000 ms"
        in markdown
    )

    # 裸LLM回答必须独立出现。
    assert (
        "一般应监控电机温度"
        in markdown
    )

    # q020是无答案问题：
    # 报告必须显示Gold无答案和预期拒答。
    assert (
        "该题被标注为知识库无答案"
        in markdown
    )
    assert (
        "无；该题预期系统拒答"
        in markdown
    )
    assert (
        "我无法访问robot-001"
        in markdown
    )

    # 报告必须保存真实控制组Prompt，
    # 使实验条件可以复查。
    assert (
        "本次没有向你提供检索文档"
        in markdown
    )

    # 每道题都应预留人工观察位置。
    assert (
        markdown.count(
            "人工观察（实验运行后填写）"
        )
        == 2
    )


def test_write_reports_creates_utf8_files(
    tmp_path: Path,
) -> None:
    """写入函数应创建目录并生成一致的JSON和Markdown。"""

    report = make_comparison_report()

    # tmp_path由pytest为本测试单独创建。
    #
    # nested/reports目前并不存在，
    # 可以同时验证write_reports()会创建父目录。
    output_directory = (
        tmp_path
        / "nested"
        / "reports"
    )

    json_output = (
        output_directory
        / "rag-vs-bare.json"
    )

    markdown_output = (
        output_directory
        / "rag-vs-bare.md"
    )

    write_reports(
        report=report,
        json_output=json_output,
        markdown_output=markdown_output,
    )

    # exists()检查目标路径是否已经存在。
    assert json_output.exists()
    assert markdown_output.exists()

    # read_text()按明确的UTF-8编码读回文件。
    json_text = json_output.read_text(
        encoding="utf-8"
    )

    markdown_text = (
        markdown_output.read_text(
            encoding="utf-8"
        )
    )

    # ensure_ascii=False应使中文原样保存在文件中。
    assert "机器人" in json_text

    # json.loads()重新解析输出文件，
    # 证明文件不仅存在，而且是合法JSON。
    json_data = json.loads(
        json_text
    )

    assert (
        json_data["llm_model"]
        == "test-llm"
    )
    assert (
        json_data["collection_name"]
        == "robot_knowledge_v4"
    )
    assert len(json_data["cases"]) == 2
    assert (
        json_data["cases"][0]
        ["gold_question"]
        ["question_id"]
        == "q014"
    )
    assert (
        json_data["cases"][1]
        ["gold_question"]
        ["question_id"]
        == "q020"
    )

    # 写入文件的Markdown必须与纯渲染函数
    # 对同一报告产生的结果完全一致。
    assert markdown_text == (
        render_markdown_report(
            report
        )
    )


def test_parse_args_uses_comparison_defaults(
) -> None:
    """没有命令行参数时应选择默认题目和输出路径。"""

    # 传入空列表表示模拟：
    #
    # python -m scripts.compare_rag_vs_bare_llm
    args = parse_args([])

    assert args.questions == Path(
        "data/eval/gold_questions.jsonl"
    )
    assert args.question_ids == [
        "q014",
        "q020",
    ]
    assert args.api_url == (
        "http://127.0.0.1:8000"
        "/api/v1/knowledge/query"
    )
    assert args.json_output == Path(
        "docs/rag-vs-bare-llm.json"
    )
    assert args.markdown_output == Path(
        "docs/rag-vs-bare-llm.md"
    )
    assert args.top_k == 3
    assert args.timeout_seconds == 120.0


def test_parse_args_accepts_repeated_question_ids(
) -> None:
    """重复使用--question-id应按输入顺序收集题号。"""

    args = parse_args(
        [
            "--questions",
            "custom/questions.jsonl",
            "--question-id",
            "q020",
            "--question-id",
            "q014",
            "--api-url",
            (
                "http://knowledge.test"
                "/api/v1/knowledge/query"
            ),
            "--json-output",
            "output/result.json",
            "--markdown-output",
            "output/result.md",
            "--top-k",
            "5",
            "--timeout-seconds",
            "45",
        ]
    )

    assert args.questions == Path(
        "custom/questions.jsonl"
    )
    assert args.question_ids == [
        "q020",
        "q014",
    ]
    assert args.api_url == API_URL
    assert args.json_output == Path(
        "output/result.json"
    )
    assert args.markdown_output == Path(
        "output/result.md"
    )

    # argparse根据type=int和type=float
    # 完成命令行字符串到Python数值的转换。
    assert args.top_k == 5
    assert args.timeout_seconds == 45.0


@pytest.mark.asyncio
async def test_run_real_comparison_orchestrates_and_closes_clients(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """真实入口应正确编排依赖并在成功后关闭客户端。"""

    settings = make_test_settings()
    expected_report = (
        make_comparison_report()
    )

    fake_llm_client = (
        FakeClosableLLMClient()
    )

    # 这些Mock分别代替：
    #
    # - Gold文件读取；
    # - 真实LLM客户端创建；
    # - 已单独测试过的实验核心；
    # - 已单独测试过的报告写入。
    load_mock = MagicMock(
        return_value=[
            make_q014_gold(),
            make_q020_gold(),
        ]
    )

    create_client_mock = MagicMock(
        return_value=fake_llm_client
    )

    compare_mock = AsyncMock(
        return_value=expected_report
    )

    write_mock = MagicMock()

    monkeypatch.setattr(
        comparison_module,
        "load_gold_questions",
        load_mock,
    )

    monkeypatch.setattr(
        comparison_module,
        "create_llm_client",
        create_client_mock,
    )

    monkeypatch.setattr(
        comparison_module,
        "compare_rag_and_bare",
        compare_mock,
    )

    monkeypatch.setattr(
        comparison_module,
        "write_reports",
        write_mock,
    )

    questions_path = (
        tmp_path
        / "gold_questions.jsonl"
    )
    json_output = (
        tmp_path
        / "result.json"
    )
    markdown_output = (
        tmp_path
        / "result.md"
    )

    result = await run_real_comparison(
        settings=settings,
        questions_path=questions_path,

        # 故意使用与Gold列表相反的顺序，
        # 验证入口会按照指定顺序选择问题。
        question_ids=[
            "q020",
            "q014",
        ],
        api_url=API_URL,
        json_output=json_output,
        markdown_output=markdown_output,
        top_k=3,
        timeout_seconds=30.0,
    )

    # run_real_comparison()应原样返回
    # compare_rag_and_bare()生成的报告。
    assert result is expected_report

    load_mock.assert_called_once_with(
        questions_path
    )

    create_client_mock.assert_called_once_with(
        settings
    )

    # 异步函数必须被await一次，
    # 不能只创建协程却忘记执行。
    compare_mock.assert_awaited_once()

    # await_args保存最近一次await时收到的参数。
    compare_arguments = (
        compare_mock.await_args.kwargs
    )

    assert [
        question.question_id
        for question
        in compare_arguments["questions"]
    ] == [
        "q020",
        "q014",
    ]

    assert (
        compare_arguments["rag_api_url"]
        == API_URL
    )
    assert (
        compare_arguments["llm_model"]
        == "test-llm"
    )
    assert (
        compare_arguments["embedding_model"]
        == "embedding-3"
    )
    assert (
        compare_arguments["collection_name"]
        == "robot_knowledge_v4"
    )
    assert compare_arguments["top_k"] == 3
    assert (
        compare_arguments[
            "similarity_threshold"
        ]
        == 0.60
    )

    # OpenAIBareLLMProvider应作为
    # BareLLMProvider传入实验核心。
    assert callable(
        compare_arguments[
            "bare_provider"
        ].generate_answer
    )

    # run_real_comparison()退出async with后，
    # httpx.AsyncClient应已经关闭。
    rag_client = compare_arguments[
        "rag_client"
    ]

    assert isinstance(
        rag_client,
        httpx.AsyncClient,
    )
    assert rag_client.is_closed is True

    # AsyncOpenAI替身也必须关闭一次。
    assert (
        fake_llm_client.close_calls
        == 1
    )

    # 完整实验成功后才允许写报告。
    write_mock.assert_called_once_with(
        report=expected_report,
        json_output=json_output,
        markdown_output=markdown_output,
    )


@pytest.mark.asyncio
async def test_run_real_comparison_closes_clients_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """实验核心失败时应关闭客户端且不写残缺报告。"""

    settings = make_test_settings()

    fake_llm_client = (
        FakeClosableLLMClient()
    )

    monkeypatch.setattr(
        comparison_module,
        "load_gold_questions",
        MagicMock(
            return_value=[
                make_q014_gold(),
            ]
        ),
    )

    monkeypatch.setattr(
        comparison_module,
        "create_llm_client",
        MagicMock(
            return_value=fake_llm_client
        ),
    )

    compare_mock = AsyncMock(
        side_effect=(
            RagComparisonRequestError(
                "模拟RAG请求失败"
            )
        )
    )

    write_mock = MagicMock()

    monkeypatch.setattr(
        comparison_module,
        "compare_rag_and_bare",
        compare_mock,
    )

    monkeypatch.setattr(
        comparison_module,
        "write_reports",
        write_mock,
    )

    with pytest.raises(
        RagComparisonRequestError,
        match="模拟RAG请求失败",
    ):
        await run_real_comparison(
            settings=settings,
            questions_path=(
                tmp_path
                / "gold_questions.jsonl"
            ),
            question_ids=[
                "q014",
            ],
            api_url=API_URL,
            json_output=(
                tmp_path
                / "result.json"
            ),
            markdown_output=(
                tmp_path
                / "result.md"
            ),
            top_k=3,
            timeout_seconds=30.0,
        )

    # finally必须在异常路径执行。
    assert (
        fake_llm_client.close_calls
        == 1
    )

    # 即使AsyncMock抛出异常，
    # 它仍然记录收到的调用参数。
    rag_client = (
        compare_mock
        .await_args
        .kwargs["rag_client"]
    )

    assert rag_client.is_closed is True

    # 实验未完成，不能写一份貌似正式的报告。
    write_mock.assert_not_called()


@pytest.mark.asyncio
async def test_run_real_comparison_validates_before_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """非法超时参数应在读取数据和创建客户端前失败。"""

    load_mock = MagicMock()
    create_client_mock = MagicMock()

    monkeypatch.setattr(
        comparison_module,
        "load_gold_questions",
        load_mock,
    )

    monkeypatch.setattr(
        comparison_module,
        "create_llm_client",
        create_client_mock,
    )

    with pytest.raises(
        ValueError,
        match="timeout_seconds",
    ):
        await run_real_comparison(
            settings=make_test_settings(),
            questions_path=(
                tmp_path
                / "gold_questions.jsonl"
            ),
            question_ids=[
                "q014",
            ],
            api_url=API_URL,
            json_output=(
                tmp_path
                / "result.json"
            ),
            markdown_output=(
                tmp_path
                / "result.md"
            ),
            top_k=3,

            # 非法值必须在发生任何外部动作前被拒绝。
            timeout_seconds=0.0,
        )

    load_mock.assert_not_called()
    create_client_mock.assert_not_called()


def test_print_summary_reports_each_case(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """终端摘要应显示每题状态、引用数和两组耗时。"""

    report = make_comparison_report()

    print_summary(report)

    # capsys捕获测试期间写入stdout和stderr的内容。
    captured = capsys.readouterr()

    assert captured.err == ""

    assert (
        "RAG与裸LLM对照实验完成"
        in captured.out
    )
    assert (
        "实验问题数：2"
        in captured.out
    )
    assert (
        "q014：RAG=已回答，引用数=2"
        in captured.out
    )
    assert (
        "RAG端到端耗时=500.0ms"
        in captured.out
    )
    assert (
        "裸LLM耗时=250.0ms"
        in captured.out
    )
    assert (
        "q020：RAG=拒答，引用数=0"
        in captured.out
    )