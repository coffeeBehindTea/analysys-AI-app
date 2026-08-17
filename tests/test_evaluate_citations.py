"""引用HTTP评测编排脚本的离线测试。"""

# json.loads()用于检查HTTPX实际发送的请求体。
import json

# datetime和timezone用于构造带UTC时区的测试报告。
from datetime import datetime, timezone

# Path用于标注pytest临时目录参数。
from pathlib import Path

# httpx.MockTransport在HTTP传输层拦截请求，
# 不建立真实TCP连接。
import httpx

import pytest

# CitationEvaluationReport是完整引用评测报告的数据契约。
#
# 测试使用它构造模拟报告，以验证Markdown和JSON输出功能。
from app.schemas.citation_evaluation import (
    CitationEvaluationReport,
)

# ExpectedEvidence描述一道Gold Question预期引用的证据位置；
# GoldQuestion描述评测集中的一道标准问题。
from app.schemas.evaluation import (
    ExpectedEvidence,
    GoldQuestion,
)

# KnowledgeCitation描述知识库API返回的一条引用；
# KnowledgeQueryResponse描述知识库查询API的完整响应。
from app.schemas.knowledge_query import (
    KnowledgeCitation,
    KnowledgeQueryResponse,
)

# 这两个函数是底层引用评分服务。
#
# evaluate_citation_question：
# 比较一道Gold Question与一次知识库响应。
#
# calculate_citation_metrics：
# 汇总多道题的单题评分结果。
from app.services.citation_evaluation import (
    calculate_citation_metrics,
    evaluate_citation_question,
)

# 这些是当前测试文件真正要测试的脚本层对象。
#
# CitationEvaluationRequestError：
# 引用评测访问知识库API失败时使用的统一异常。
#
# evaluate_citations_via_api：
# 编排HTTP请求、响应验证、单题评分和报告汇总。
#
# compact_preview：
# 将长文本压缩为报告中的简短预览。
#
# render_markdown_report：
# 把结构化报告转换成Markdown。
#
# write_reports：
# 将报告写成JSON和Markdown文件。
from scripts.evaluate_citations import (
    CitationEvaluationRequestError,
    compact_preview,
    evaluate_citations_via_api,

    # parse_args负责把命令行字符串转换成带类型的参数对象。
    parse_args,

    # print_summary负责把结构化指标输出到标准输出。
    print_summary,

    render_markdown_report,
    write_reports,
)


API_URL = (
    "http://knowledge.test"
    "/api/v1/knowledge/query"
)


def make_q005_gold(
) -> GoldQuestion:
    """创建一条可回答的q005 Gold Question。"""

    return GoldQuestion(
        question_id="q005",
        question=(
            "《工业移动机器人安全标准》"
            "征求意见稿适用于哪些机器人？"
        ),
        question_type="single_hop",
        answerable=True,
        reference_answer=(
            "该征求意见稿适用于工业移动机器人。"
        ),
        expected_evidence=[
            ExpectedEvidence(
                source_file=(
                    "工业移动机器人安全标准.pdf"
                ),
                page_or_section=(
                    "page: 5; section: 1 范围"
                ),
            )
        ],
        tags=[
            "safety-standard",
        ],
        notes="引用HTTP评测成功路径测试。",
    )


def make_q020_gold(
) -> GoldQuestion:
    """创建一条无答案的q020 Gold Question。"""

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
        notes="静态知识库没有实时遥测。",
    )


def make_q005_response_payload(
) -> dict[str, object]:
    """创建q005对应的合法API JSON响应数据。"""

    document_id = "6" * 64

    response = KnowledgeQueryResponse(
        answer=(
            "该征求意见稿适用于"
            "工业移动机器人。"
        ),
        citations=[
            KnowledgeCitation(
                chunk_id=(
                    f"{document_id}:000015"
                ),
                document_id=document_id,
                source_file=(
                    "工业移动机器人安全标准.pdf"
                ),
                page_or_section="page: 5",
                chunk_index=15,
                rank=2,
                similarity=0.66,
                excerpt=(
                    "本文件适用于"
                    "工业移动机器人。"
                ),
            )
        ],
        retrieval_ms=120.0,
        abstained=False,
    )

    # mode="json"把Pydantic模型转换成
    # 可以直接作为HTTP JSON响应的数据。
    return response.model_dump(
        mode="json"
    )


def make_q020_response_payload(
) -> dict[str, object]:
    """创建q020对应的合法拒答JSON响应数据。"""

    response = KnowledgeQueryResponse(
        answer="知识库没有足够证据回答该问题。",
        citations=[],
        retrieval_ms=80.0,
        abstained=True,
    )

    return response.model_dump(
        mode="json"
    )


@pytest.mark.asyncio
async def test_evaluate_citations_via_api_builds_report(
) -> None:
    """q005正常回答和q020拒答应生成完整报告。"""

    q005 = make_q005_gold()
    q020 = make_q020_gold()

    # 保存MockTransport收到的请求，
    # 用于验证HTTP方法、路径和JSON请求体。
    received_requests: list[
        tuple[str, str, dict[str, object]]
    ] = []

    def handle_request(
        request: httpx.Request,
    ) -> httpx.Response:
        """根据问题内容返回预设的API响应。"""

        # request.content是HTTP请求体原始字节。
        #
        # decode("utf-8")把字节转换成JSON字符串；
        # json.loads()再转换成Python字典。
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
            == q005.question
        ):
            return httpx.Response(
                status_code=200,
                headers={
                    "X-Request-ID": "request-q005",
                },
                json=make_q005_response_payload(),

                # 将当前Request附加到Response，
                # 使raise_for_status()拥有完整上下文。
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
                json=make_q020_response_payload(),
                request=request,
            )

        # 如果编排函数发送了意外问题，
        # 返回400使测试明确失败。
        return httpx.Response(
            status_code=400,
            json={
                "detail": "unexpected question",
            },
            request=request,
        )

    # MockTransport不会解析域名或建立网络连接。
    # 所有请求都会交给handle_request()。
    transport = httpx.MockTransport(
        handle_request
    )

    # async with会在代码块结束时自动执行：
    #
    # await client.aclose()
    #
    # 从而释放客户端资源。
    async with httpx.AsyncClient(
        transport=transport,
    ) as client:
        report = await evaluate_citations_via_api(
            questions=[
                q005,
                q020,
            ],
            client=client,
            api_url=API_URL,
            llm_model="test-llm",
            embedding_model="embedding-3",
            collection_name=(
                "robot_knowledge_v4"
            ),
            top_k=3,
            similarity_threshold=0.70,
        )

    # 验证实际发送了两个POST请求，
    # 顺序与Gold Question列表一致。
    assert received_requests == [
        (
            "POST",
            "/api/v1/knowledge/query",
            {
                "question": q005.question,
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

    # 验证实验条件被写入报告。
    assert report.api_url == API_URL
    assert report.llm_model == "test-llm"

    assert (
        report.embedding_model
        == "embedding-3"
    )

    assert (
        report.collection_name
        == "robot_knowledge_v4"
    )

    assert report.top_k == 3

    assert (
        report.similarity_threshold
        == 0.70
    )

    # 报告必须保留两道题的原始顺序。
    assert len(report.results) == 2

    assert (
        report.results[0]
        .gold_question
        .question_id
        == "q005"
    )

    assert (
        report.results[1]
        .gold_question
        .question_id
        == "q020"
    )

    # q005的page 5引用匹配Gold，
    # 因此属于完整证据回答。
    q005_result = report.results[0]

    assert q005_result.fully_grounded is True
    assert q005_result.citation_correctness == 1.0
    assert q005_result.citation_coverage == 1.0

    # q020属于正确拒答。
    q020_result = report.results[1]

    assert q020_result.correctly_abstained is True
    assert q020_result.response.citations == []

    # 汇总数据集包含一条正确引用，
    # 一道完整回答和一道正确拒答。
    metrics = report.metrics

    assert metrics.total_question_count == 2
    assert metrics.answerable_question_count == 1
    assert metrics.unanswerable_question_count == 1

    assert (
        metrics.answered_answerable_count
        == 1
    )

    assert (
        metrics.fully_grounded_answer_count
        == 1
    )

    assert (
        metrics
        .correctly_abstained_unanswerable_count
        == 1
    )

    assert metrics.returned_citation_count == 1
    assert metrics.correct_citation_count == 1

    assert metrics.citation_correctness == 1.0
    assert metrics.citation_coverage == 1.0

    assert (
        metrics.answerable_response_rate
        == 1.0
    )

    assert (
        metrics.fully_grounded_answer_rate
        == 1.0
    )

    assert (
        metrics.unanswerable_abstention_rate
        == 1.0
    )


@pytest.mark.asyncio
async def test_evaluation_reports_http_error_with_request_id(
) -> None:
    """API返回502时应保留题号、状态码和Request ID。"""

    q005 = make_q005_gold()

    def handle_request(
        request: httpx.Request,
    ) -> httpx.Response:
        """模拟知识库API返回502。"""

        return httpx.Response(
            status_code=502,
            headers={
                "X-Request-ID": (
                    "request-upstream-failure"
                ),
            },
            json={
                "request_id": (
                    "request-upstream-failure"
                ),
                "error": {
                    "code": "llm_upstream_error",
                    "message": (
                        "LLM上游服务暂时不可用"
                    ),
                },
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
            CitationEvaluationRequestError,
            match="返回错误状态",
        ) as exc_info:
            await evaluate_citations_via_api(
                questions=[q005],
                client=client,
                api_url=API_URL,
                llm_model="test-llm",
                embedding_model="embedding-3",
                collection_name=(
                    "robot_knowledge_v4"
                ),
                top_k=3,
                similarity_threshold=0.70,
            )

    # exc_info.value是实际捕获的异常对象。
    error_message = str(
        exc_info.value
    )

    assert "question_id=q005" in error_message
    assert "status=502" in error_message

    # Request ID可以用于查询对应的服务端日志。
    assert (
        "request_id=request-upstream-failure"
        in error_message
    )


@pytest.mark.asyncio
async def test_evaluation_rejects_invalid_json_response(
) -> None:
    """API返回200但正文不是JSON时评测必须失败。"""

    q005 = make_q005_gold()

    def handle_request(
        request: httpx.Request,
    ) -> httpx.Response:
        """模拟200状态配普通文本正文。"""

        return httpx.Response(
            status_code=200,
            headers={
                "X-Request-ID": (
                    "request-invalid-json"
                ),
                "Content-Type": "text/plain",
            },

            # content接收原始响应字节。
            content=b"this is not json",
            request=request,
        )

    transport = httpx.MockTransport(
        handle_request
    )

    async with httpx.AsyncClient(
        transport=transport,
    ) as client:
        with pytest.raises(
            CitationEvaluationRequestError,
            match="非法JSON",
        ) as exc_info:
            await evaluate_citations_via_api(
                questions=[q005],
                client=client,
                api_url=API_URL,
                llm_model="test-llm",
                embedding_model="embedding-3",
                collection_name=(
                    "robot_knowledge_v4"
                ),
                top_k=3,
                similarity_threshold=0.70,
            )

    assert "question_id=q005" in str(
        exc_info.value
    )


@pytest.mark.asyncio
async def test_evaluation_rejects_invalid_response_contract(
) -> None:
    """合法JSON缺少接口字段时评测必须失败。"""

    q005 = make_q005_gold()

    def handle_request(
        request: httpx.Request,
    ) -> httpx.Response:
        """模拟结构不完整的200 JSON响应。"""

        return httpx.Response(
            status_code=200,
            headers={
                "X-Request-ID": (
                    "request-invalid-contract"
                ),
            },

            # 这是合法JSON，
            # 但缺少citations、retrieval_ms和abstained。
            json={
                "answer": "结构不完整的回答",
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
            CitationEvaluationRequestError,
            match="KnowledgeQueryResponse契约",
        ) as exc_info:
            await evaluate_citations_via_api(
                questions=[q005],
                client=client,
                api_url=API_URL,
                llm_model="test-llm",
                embedding_model="embedding-3",
                collection_name=(
                    "robot_knowledge_v4"
                ),
                top_k=3,
                similarity_threshold=0.70,
            )

    assert "question_id=q005" in str(
        exc_info.value
    )


@pytest.mark.asyncio
async def test_evaluation_converts_http_timeout(
) -> None:
    """HTTPX超时应转换成评测层异常并标明题号。"""

    q005 = make_q005_gold()

    def handle_request(
        request: httpx.Request,
    ) -> httpx.Response:
        """模拟HTTP传输层读取超时。"""

        # ReadTimeout继承自TimeoutException。
        #
        # 这里直接抛异常，
        # 模拟客户端等待服务端响应超过限制。
        raise httpx.ReadTimeout(
            "simulated timeout",
            request=request,
        )

    transport = httpx.MockTransport(
        handle_request
    )

    async with httpx.AsyncClient(
        transport=transport,
    ) as client:
        with pytest.raises(
            CitationEvaluationRequestError,
            match="请求超时",
        ) as exc_info:
            await evaluate_citations_via_api(
                questions=[q005],
                client=client,
                api_url=API_URL,
                llm_model="test-llm",
                embedding_model="embedding-3",
                collection_name=(
                    "robot_knowledge_v4"
                ),
                top_k=3,
                similarity_threshold=0.70,
            )

    assert "q005" in str(
        exc_info.value
    )


@pytest.mark.asyncio
async def test_duplicate_question_id_is_rejected_before_http(
) -> None:
    """重复Gold编号必须在发送请求前被拒绝。"""

    q005 = make_q005_gold()

    # 如果Handler被调用，
    # Request对象就会被添加到这个列表。
    received_requests: list[
        httpx.Request
    ] = []

    def handle_request(
        request: httpx.Request,
    ) -> httpx.Response:
        """记录所有意外发出的HTTP请求。"""

        received_requests.append(
            request
        )

        return httpx.Response(
            status_code=200,
            json=make_q005_response_payload(),
            request=request,
        )

    transport = httpx.MockTransport(
        handle_request
    )

    async with httpx.AsyncClient(
        transport=transport,
    ) as client:
        with pytest.raises(
            ValueError,
            match="重复question_id",
        ):
            await evaluate_citations_via_api(
                # 同一个q005被传入两次。
                questions=[
                    q005,
                    q005,
                ],
                client=client,
                api_url=API_URL,
                llm_model="test-llm",
                embedding_model="embedding-3",
                collection_name=(
                    "robot_knowledge_v4"
                ),
                top_k=3,
                similarity_threshold=0.70,
            )

    # 重复编号校验发生在for循环和client.post()之前，
    # 所以不能发出任何请求。
    assert received_requests == []


def make_success_report(
) -> CitationEvaluationReport:
    """创建q005通过、q020正确拒答的测试报告。"""

    q005_result = evaluate_citation_question(
        question=make_q005_gold(),
        response=(
            KnowledgeQueryResponse.model_validate(
                make_q005_response_payload()
            )
        ),
    )

    q020_result = evaluate_citation_question(
        question=make_q020_gold(),
        response=(
            KnowledgeQueryResponse.model_validate(
                make_q020_response_payload()
            )
        ),
    )

    results = [
        q005_result,
        q020_result,
    ]

    return CitationEvaluationReport(
        generated_at=datetime.now(
            timezone.utc
        ),
        api_url=API_URL,
        llm_model="test-llm",
        embedding_model="embedding-3",
        collection_name="robot_knowledge_v4",
        top_k=3,
        similarity_threshold=0.70,
        metrics=calculate_citation_metrics(
            results
        ),
        results=results,
    )


def make_answerable_abstention_report(
) -> CitationEvaluationReport:
    """创建Gold可回答但API拒答的失败报告。"""

    q005 = make_q005_gold()

    abstained_response = KnowledgeQueryResponse(
        answer="知识库没有足够证据回答该问题。",
        citations=[],
        retrieval_ms=90.0,
        abstained=True,
    )

    result = evaluate_citation_question(
        question=q005,
        response=abstained_response,
    )

    results = [
        result,
    ]

    return CitationEvaluationReport(
        generated_at=datetime.now(
            timezone.utc
        ),
        api_url=API_URL,
        llm_model="test-llm",
        embedding_model="embedding-3",
        collection_name="robot_knowledge_v4",
        top_k=3,
        similarity_threshold=0.70,
        metrics=calculate_citation_metrics(
            results
        ),
        results=results,
    )


def test_render_markdown_contains_summary_and_passed_questions(
) -> None:
    """成功报告应包含汇总指标和两道通过题目。"""

    markdown = render_markdown_report(
        make_success_report()
    )

    assert (
        "# Day 5–7 RAG回答引用评测"
        in markdown
    )

    assert (
        "robot_knowledge_v4"
        in markdown
    )

    # 成功数据集中唯一一条返回引用是正确的。
    assert (
        "| 引用正确率 | 1.000 |"
        in markdown
    )

    assert (
        "| 引用覆盖率 | 1.000 |"
        in markdown
    )

    assert "`q005`" in markdown
    assert "`q020`" in markdown

    # q005完整证据回答、q020正确拒答，
    # 所以没有失败案例。
    assert (
        "本次评测没有发现失败案例。"
        in markdown
    )


def test_render_markdown_explains_answerable_abstention(
) -> None:
    """可回答题拒答时报告应说明失败原因。"""

    markdown = render_markdown_report(
        make_answerable_abstention_report()
    )

    assert "### q005" in markdown

    assert (
        "- API状态：拒答"
        in markdown
    )

    assert (
        "Gold标注为可回答，"
        "但API返回了拒答"
        in markdown
    )

    assert (
        "没有引用覆盖全部Gold预期证据"
        in markdown
    )

    # 失败详情应保留参考答案，
    # 方便人工检查回答质量。
    assert "- 参考答案：" in markdown

    # API拒答，所以返回引用部分应显示“无”。
    assert "- API返回引用：" in markdown
    assert "  - 无" in markdown


def test_write_reports_creates_parseable_files(
    tmp_path: Path,
) -> None:
    """JSON和Markdown应写入临时目录且内容可读取。"""

    report = make_success_report()

    # tmp_path由pytest为当前测试创建，
    # 每次测试都会得到隔离的临时目录。
    json_output = (
        tmp_path
        / "nested"
        / "citation-evaluation.json"
    )

    markdown_output = (
        tmp_path
        / "nested"
        / "citation-evaluation.md"
    )

    write_reports(
        report=report,
        json_output=json_output,
        markdown_output=markdown_output,
    )

    assert json_output.is_file()
    assert markdown_output.is_file()

    # 重新读取并解析JSON，
    # 证明文件不是只写入了不可解析的普通文本。
    parsed_json = json.loads(
        json_output.read_text(
            encoding="utf-8"
        )
    )

    assert (
        parsed_json["metrics"]
        ["citation_correctness"]
        == 1.0
    )

    assert len(
        parsed_json["results"]
    ) == 2

    markdown = markdown_output.read_text(
        encoding="utf-8"
    )

    assert (
        "# Day 5–7 RAG回答引用评测"
        in markdown
    )

    assert (
        "| 无答案正确拒答率 | 1.000 |"
        in markdown
    )


def test_compact_preview_normalizes_and_truncates(
) -> None:
    """预览应压缩空白并在超长时添加省略号。"""

    # 换行和连续空格会被压缩成一个空格。
    assert compact_preview(
        "第一行\n\n第二行",
        max_length=20,
    ) == "第一行 第二行"

    # max_length表示省略号之前最多保留的字符数。
    assert compact_preview(
        "123456789",
        max_length=5,
    ) == "12345……"


def test_compact_preview_rejects_invalid_length(
) -> None:
    """预览最大长度必须是正整数。"""

    with pytest.raises(
        ValueError,
        match="max_length",
    ):
        compact_preview(
            "测试内容",
            max_length=0,
        )


def test_parse_args_uses_reproducible_defaults(
) -> None:
    """不传命令行参数时应使用可复现的默认配置。"""

    # 传入空列表表示“没有用户命令行参数”。
    #
    # 这样不会读取pytest进程自身的sys.argv，
    # 避免把pytest的-v等参数误认为评测脚本参数。
    args = parse_args([])

    # type=Path使字符串默认值最终成为Path对象。
    assert args.questions == Path(
        "data/eval/gold_questions.jsonl"
    )

    assert args.api_url == (
        "http://127.0.0.1:8000"
        "/api/v1/knowledge/query"
    )

    assert args.json_output == Path(
        "docs/citation-evaluation.json"
    )

    assert args.markdown_output == Path(
        "docs/citation-evaluation.md"
    )

    # type=int和type=float确保数值参数具有正确类型。
    assert args.top_k == 3
    assert isinstance(args.top_k, int)

    assert args.timeout_seconds == 120.0
    assert isinstance(
        args.timeout_seconds,
        float,
    )


def test_parse_args_converts_user_overrides(
) -> None:
    """用户参数应被解析并转换成声明的Python类型。"""

    args = parse_args(
        [
            "--questions",
            "data/eval/custom.jsonl",
            "--api-url",
            (
                "http://knowledge.test"
                "/api/v1/knowledge/query"
            ),
            "--json-output",
            "build/result.json",
            "--markdown-output",
            "build/result.md",
            "--top-k",
            "5",
            "--timeout-seconds",
            "45.5",
        ]
    )

    # argparse按照add_argument(type=Path)
    # 将路径字符串转换成Path对象。
    assert args.questions == Path(
        "data/eval/custom.jsonl"
    )

    assert args.api_url == (
        "http://knowledge.test"
        "/api/v1/knowledge/query"
    )

    assert args.json_output == Path(
        "build/result.json"
    )

    assert args.markdown_output == Path(
        "build/result.md"
    )

    # 命令行中的所有原始参数都是字符串，
    # argparse根据type=int和type=float执行类型转换。
    assert args.top_k == 5
    assert isinstance(args.top_k, int)

    assert args.timeout_seconds == 45.5
    assert isinstance(
        args.timeout_seconds,
        float,
    )


def test_print_summary_outputs_key_metrics(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """终端摘要应输出报告中的关键指标。"""

    # make_success_report()生成两道题的报告：
    #
    # q005：可回答、正常回答、引用正确且证据完整；
    # q020：无答案、正确拒答、引用为空。
    report = make_success_report()

    # print_summary()预期只打印报告，
    # 不修改报告，也不返回新的业务数据。
    print_summary(report)

    # capsys是pytest提供的标准输出捕获Fixture。
    #
    # readouterr()读取调用期间产生的：
    # - out：标准输出；
    # - err：标准错误。
    captured = capsys.readouterr()

    assert "引用评测完成" in captured.out
    assert "问题总数：2" in captured.out
    assert "可回答问题数：1" in captured.out
    assert "无答案问题数：1" in captured.out
    assert (
        "实际回答的可回答问题数：1"
        in captured.out
    )
    assert "引用正确率：1.000" in captured.out
    assert "引用覆盖率：1.000" in captured.out
    assert (
        "可回答问题响应率：1.000"
        in captured.out
    )
    assert (
        "完整证据回答率：1.000"
        in captured.out
    )
    assert (
        "无答案问题正确拒答率：1.000"
        in captured.out
    )

    # 正常摘要不应该向标准错误输出内容。
    assert captured.err == ""