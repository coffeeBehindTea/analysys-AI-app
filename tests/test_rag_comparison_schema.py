"""RAG与裸LLM对比实验数据契约的离线测试。"""

import pytest

# ValidationError表示输入没有通过Pydantic契约校验。
from pydantic import ValidationError

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


def make_q014_gold() -> GoldQuestion:
    """创建有答案的多跳故障处理问题。"""

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
        notes="RAG与裸LLM对比测试问题。",
    )


def make_q020_gold() -> GoldQuestion:
    """创建知识库无答案的实时状态问题。"""

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
        notes=(
            "静态知识库不包含机器人实时遥测。"
        ),
    )


def make_answered_rag_response(
) -> KnowledgeQueryResponse:
    """创建具有两条真实结构引用的RAG响应。"""

    procedure_document_id = "a" * 64
    fault_document_id = "b" * 64

    return KnowledgeQueryResponse(
        answer=(
            "应通过测试接口注入模拟温度信号，"
            "高于85°C并持续10秒后触发"
            "ERR-DRV-3005；完成人工检查前不得复位。"
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


def make_abstained_rag_response(
) -> KnowledgeQueryResponse:
    """创建知识库证据不足时的合法拒答响应。"""

    return KnowledgeQueryResponse(
        answer=(
            "知识库没有足够证据回答该问题。"
        ),
        citations=[],
        retrieval_ms=90.0,
        abstained=True,
    )


def make_answerable_case(
) -> RagVsBareLLMCase:
    """创建q014的RAG与裸LLM对比结果。"""

    return RagVsBareLLMCase(
        gold_question=make_q014_gold(),
        rag_response=(
            make_answered_rag_response()
        ),
        rag_total_ms=950.0,
        bare_result=BareLLMResult(
            answer=(
                "一般应监控电机温度并在过热时停机。"
            ),
            generation_ms=420.0,
        ),
    )


def make_unanswerable_case(
) -> RagVsBareLLMCase:
    """创建q020的RAG与裸LLM对比结果。"""

    return RagVsBareLLMCase(
        gold_question=make_q020_gold(),
        rag_response=(
            make_abstained_rag_response()
        ),
        rag_total_ms=150.0,
        bare_result=BareLLMResult(
            answer=(
                "我无法访问robot-001的实时位置"
                "和电量信息。"
            ),
            generation_ms=260.0,
        ),
    )


def make_report(
    *,
    cases: list[RagVsBareLLMCase],
) -> RagVsBareLLMReport:
    """使用指定实验Case创建完整报告。"""

    return RagVsBareLLMReport(
        rag_api_url=(
            "http://127.0.0.1:8000"
            "/api/v1/knowledge/query"
        ),
        llm_model="test-llm",
        embedding_model="embedding-3",
        collection_name="robot_knowledge_v4",
        top_k=3,
        similarity_threshold=0.60,
        bare_system_prompt=(
            "直接回答用户问题；不知道时明确说明不知道。"
        ),
        cases=cases,
    )


def test_valid_report_preserves_both_experiment_groups(
) -> None:
    """合法报告应同时保存RAG与裸LLM结果。"""

    report = make_report(
        cases=[
            make_answerable_case(),
            make_unanswerable_case(),
        ]
    )

    assert len(report.cases) == 2

    q014_case = report.cases[0]
    q020_case = report.cases[1]

    # q014的RAG实验组应有两条可追溯引用。
    assert (
        q014_case.gold_question.question_id
        == "q014"
    )
    assert q014_case.rag_response.abstained is False
    assert len(
        q014_case.rag_response.citations
    ) == 2

    # 裸LLM实验组只保存文本和耗时。
    assert (
        q014_case.bare_result.answer
        == "一般应监控电机温度并在过热时停机。"
    )
    assert (
        q014_case.bare_result.generation_ms
        == 420.0
    )

    # q020的RAG实验组应正确拒答。
    assert (
        q020_case.gold_question.question_id
        == "q020"
    )
    assert q020_case.rag_response.abstained is True
    assert q020_case.rag_response.citations == []

    # mode="json"递归序列化所有嵌套Pydantic模型。
    serialized = report.model_dump(
        mode="json"
    )

    assert "generated_at" not in serialized
    assert (
        serialized["cases"][0]
        ["rag_response"]["citations"][0]
        ["rank"]
        == 1
    )


def test_bare_result_rejects_negative_latency(
) -> None:
    """裸LLM生成耗时不能是负数。"""

    with pytest.raises(
        ValidationError,
        match="generation_ms",
    ):
        BareLLMResult(
            answer="测试回答",
            generation_ms=-0.01,
        )


def test_bare_result_rejects_unexpected_fields(
) -> None:
    """裸LLM不能伪装成具有知识库引用的响应。"""

    with pytest.raises(
        ValidationError,
        match="citations",
    ):
        BareLLMResult.model_validate(
            {
                "answer": "测试回答",
                "generation_ms": 10.0,

                # BareLLMResult没有声明citations字段；
                # extra="forbid"必须拒绝它。
                "citations": [],
            }
        )


def test_report_rejects_duplicate_question_ids(
) -> None:
    """同一道题不能重复计入同一份实验报告。"""

    q014_case = make_answerable_case()

    with pytest.raises(
        ValidationError,
        match="重复question_id",
    ):
        make_report(
            cases=[
                q014_case,
                q014_case,
            ]
        )


def test_report_requires_at_least_one_case(
) -> None:
    """实验报告至少要包含一道对比问题。"""

    with pytest.raises(
        ValidationError,
        match="cases",
    ):
        make_report(
            cases=[],
        )
