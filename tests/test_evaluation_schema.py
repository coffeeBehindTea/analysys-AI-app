"""任务四检索评测结果数据契约测试。"""

# datetime 用来构造报告生成时间；
from datetime import datetime, timezone

import pytest

# ValidationError 是 Pydantic 在模型数据
# 不符合 Field 约束时抛出的异常。
from pydantic import ValidationError

from app.schemas.evaluation import (
    ExpectedEvidence,
    RetrievalEvaluationReport,
    RetrievalMetrics,
    RetrievalQuestionEvaluation,
)
from app.schemas.retrieval import (
    DocumentChunk,
    RetrievedChunk,
)


def make_retrieved_chunk() -> RetrievedChunk:
    """创建供多个测试复用的合法检索结果。"""

    document_chunk = DocumentChunk(
        chunk_id="document-001:000000",
        document_id="document-001",
        source_file="manual.md",
        page_or_section="section: safety",
        chunk_index=0,

        # content_hash 要求是恰好 64 位的
        # 小写十六进制字符串。
        content_hash="a" * 64,

        content="机器人触发安全保护后应立即停车。",
    )

    return RetrievedChunk(
        chunk=document_chunk,
        similarity=0.91,
        rank=1,
    )


def make_valid_report() -> RetrievalEvaluationReport:
    """创建一份满足所有字段约束的评测报告。"""

    expected_evidence = ExpectedEvidence(
        source_file="manual.md",
        page_or_section="section: safety",
    )

    question_result = RetrievalQuestionEvaluation(
        question_id="q001",
        question="机器人在什么情况下应停车？",
        question_type="single_hop",
        answerable=True,
        expected_evidence=[expected_evidence],
        retrieved_chunks=[make_retrieved_chunk()],
        matched_evidence_at_1=1,
        matched_evidence_at_3=1,
        fully_recalled_at_1=True,
        fully_recalled_at_3=True,
        top_similarity=0.91,

        # 这是可回答问题，不参与无答案错误召回，
        # 因而设为 False。
        false_recall=False,
    )

    metrics = RetrievalMetrics(
        answerable_question_count=1,
        unanswerable_question_count=0,
        expected_evidence_count=1,
        matched_evidence_at_1=1,
        matched_evidence_at_3=1,
        recall_at_1=1.0,
        recall_at_3=1.0,
        fully_recalled_at_1_count=1,
        fully_recalled_at_3_count=1,
        false_recall_count=0,
        false_recall_rate=0.0,
    )

    return RetrievalEvaluationReport(
        # datetime.now(timezone.utc) 返回带时区的当前时间。
        generated_at=datetime.now(timezone.utc),
        embedding_model="embedding-3",
        collection_name="robot_knowledge",
        top_k=3,
        similarity_threshold=0.70,
        metrics=metrics,
        results=[question_result],
    )


def test_valid_retrieval_evaluation_report() -> None:
    """合法数据应该能够构造完整评测报告。"""

    report = make_valid_report()

    assert report.embedding_model == "embedding-3"
    assert report.collection_name == "robot_knowledge"
    assert report.top_k == 3
    assert report.metrics.recall_at_3 == 1.0
    assert report.results[0].fully_recalled_at_3 is True

    # results[0] 是第一道题的结果；
    # retrieved_chunks[0] 是该题排名第一的 Chunk。
    assert (
        report.results[0]
        .retrieved_chunks[0]
        .chunk
        .source_file
        == "manual.md"
    )


def test_report_top_k_cannot_be_less_than_three() -> None:
    """报告需要计算 Recall@3，因此 top_k 不能小于 3。"""

    valid_report = make_valid_report()

    with pytest.raises(
        ValidationError,
        match="top_k",
    ):
        # model_copy() 根据现有 Pydantic 模型创建副本。
        #
        # update 指定副本中需要替换的字段。
        # 但 model_copy 默认不会重新验证更新值，
        # 所以这里先转换为字典，再重新构造模型。
        report_data = valid_report.model_dump()
        report_data["top_k"] = 2

        RetrievalEvaluationReport.model_validate(
            report_data
        )


def test_similarity_cannot_exceed_one() -> None:
    """余弦相似度不能超过模型声明的最大值 1。"""

    with pytest.raises(
        ValidationError,
        match="similarity",
    ):
        RetrievedChunk(
            chunk=make_retrieved_chunk().chunk,
            similarity=1.1,
            rank=1,
        )


def test_report_rejects_unknown_fields() -> None:
    """extra='forbid' 应拒绝数据契约之外的字段。"""

    report_data = make_valid_report().model_dump()

    # unknown_field 没有在
    # RetrievalEvaluationReport 中声明。
    report_data["unknown_field"] = "unexpected"

    with pytest.raises(
        ValidationError,
        match="unknown_field",
    ):
        RetrievalEvaluationReport.model_validate(
            report_data
        )


def test_report_can_be_serialized_for_json() -> None:
    """报告应该能转换成可写入 JSON 文件的数据。"""

    report = make_valid_report()

    # mode="json" 会把 datetime 等 Python 对象
    # 转换为 JSON 可以表示的数据类型。
    serialized = report.model_dump(mode="json")

    assert isinstance(serialized["generated_at"], str)
    assert serialized["metrics"]["recall_at_3"] == 1.0
    assert (
        serialized["results"][0]["question_id"]
        == "q001"
    )