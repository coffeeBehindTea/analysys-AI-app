"""候选策略真实评测脚本中纯函数部分的离线测试。

本模块不调用run_real_strategy()，因此不会访问
Embedding API、Chroma数据库或真实Gold文件。

测试范围：
1. 命令行整数参数解析；
2. 四种策略参数契约构造；
3. 默认和自定义报告路径；
4. 纯向量及混合报告的Markdown渲染；
5. JSON和Markdown文件写入。
"""

import argparse
import json
from pathlib import Path
import sys

import pytest

from app.schemas.evaluation import (
    ExpectedEvidence,
    GoldQuestion,
)
from app.schemas.retrieval import (
    DocumentChunk,
    HybridRetrievedChunk,
    RetrievedChunk,
)
from app.schemas.retrieval_strategy import (
    CandidateStrategyParameters,
    CandidateStrategyReport,
    HeadPreservingRerankParameters,
)
from app.services.head_preserving_reranking import (
    HEAD_PRESERVING_RERANK_VERSION,
    HeadPreservationPolicy,
)
from app.services.query_rewriting import (
    DETERMINISTIC_QUERY_REWRITE_VERSION,
)
from app.services.retrieval_strategy_evaluation import (
    build_candidate_strategy_report,
    evaluate_candidate_question,
)
from scripts.evaluate_retrieval_strategies import (
    build_runtime_rerank_policy,
    build_strategy_parameters,
    parse_args,
    parse_positive_integer,
    parse_top_k,
    render_markdown_report,
    resolve_output_paths,
    write_reports,
)


TEST_CONTENT_HASH = "d" * 64


def make_chunk() -> DocumentChunk:
    """构造报告渲染测试使用的Chunk。"""

    return DocumentChunk(
        chunk_id="document-001:000000",
        document_id="document-001",
        source_file="manual|test.pdf",
        page_or_section="page: 56",
        chunk_index=0,
        content_hash=TEST_CONTENT_HASH,
        content=(
            "第一行候选正文。\n"
            "第二行包含更多可审计内容。"
        ),
    )


def make_question() -> GoldQuestion:
    """构造一条能够由测试Chunk回答的问题。"""

    return GoldQuestion(
        question_id="q001",
        question="测试证据位于哪里？",
        question_type="single_hop",
        answerable=True,
        reference_answer="证据位于第56页。",
        expected_evidence=[
            ExpectedEvidence(
                source_file="manual|test.pdf",
                page_or_section="page: 56",
            )
        ],
        tags=["test"],
        notes="报告渲染测试",
    )


def make_vector_report(
) -> CandidateStrategyReport:
    """构造一份纯向量候选策略报告。"""

    evaluation = evaluate_candidate_question(
        question=make_question(),
        retrieved_chunks=[
            RetrievedChunk(
                chunk=make_chunk(),
                similarity=0.8,
                rank=1,
            )
        ],
    )

    return build_candidate_strategy_report(
        embedding_model="embedding-3",
        collection_name="robot_knowledge_v4",
        parameters=build_strategy_parameters(
            strategy="vector_baseline",
            top_k=3,
            candidate_k=20,
            rank_constant=60,
        ),
        evaluations=[evaluation],
    )


def make_hybrid_report(
    *,
    strategy: str = "hybrid_rrf",
) -> CandidateStrategyReport:
    """构造一份带关键词审计信息的混合策略报告。"""

    evaluation = evaluate_candidate_question(
        question=make_question(),
        retrieved_chunks=[
            HybridRetrievedChunk(
                chunk=make_chunk(),
                rrf_score=0.032,
                rank=1,
                vector_rank=2,
                vector_similarity=0.72,
                keyword_rank=1,
                keyword_score=12.0,
                matched_identifiers=(
                    "err-test-1001",
                ),
                matched_lexical_terms=(
                    "测试",
                ),
            )
        ],
    )

    return build_candidate_strategy_report(
        embedding_model="embedding-3",
        collection_name="robot_knowledge_v4",
        parameters=build_strategy_parameters(
            strategy=strategy,
            top_k=3,
            candidate_k=20,
            rank_constant=60,
        ),
        evaluations=[evaluation],
    )


def test_positive_integer_parser_accepts_integer(
) -> None:
    """合法正整数文本应转换成int。"""

    assert parse_positive_integer("20") == 20


@pytest.mark.parametrize(
    "raw_value",
    [
        "not-an-integer",
        "0",
        "-1",
    ],
    ids=[
        "invalid-text",
        "zero",
        "negative",
    ],
)
def test_positive_integer_parser_rejects_invalid_values(
    raw_value: str,
) -> None:
    """非法文本、零和负数应成为argparse错误。"""

    with pytest.raises(
        argparse.ArgumentTypeError,
    ):
        parse_positive_integer(
            raw_value
        )


@pytest.mark.parametrize(
    "raw_value",
    [
        "1",
        "2",
    ],
    ids=[
        "top-one",
        "top-two",
    ],
)
def test_top_k_parser_requires_at_least_three(
    raw_value: str,
) -> None:
    """候选报告需要计算Recall@3。"""

    with pytest.raises(
        argparse.ArgumentTypeError,
        match="必须大于或等于3",
    ):
        parse_top_k(raw_value)


def test_parse_args_uses_expected_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """只给策略名时应取得稳定默认参数。"""

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_retrieval_strategies.py",
            "--strategy",
            "hybrid_rrf",
        ],
    )

    args = parse_args()

    assert args.strategy == "hybrid_rrf"
    assert args.questions == Path(
        "data/eval/gold_questions.jsonl"
    )
    assert args.top_k == 3
    assert args.candidate_k == 20
    assert args.rank_constant == 60
    assert args.json_output is None
    assert args.markdown_output is None


def test_parse_args_accepts_rewrite_rerank_strategy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """命令行choices必须接受第四种候选策略。"""

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_retrieval_strategies.py",
            "--strategy",
            "hybrid_rrf_rewrite_rerank",
        ],
    )

    args = parse_args()

    assert args.strategy == (
        "hybrid_rrf_rewrite_rerank"
    )


@pytest.mark.parametrize(
    (
        "strategy",
        "expected_candidate_k",
        "expected_rank_constant",
        "expected_rewrite_version",
        "expected_rerank_version",
    ),
    [
        (
            "vector_baseline",
            None,
            None,
            None,
            None,
        ),
        (
            "hybrid_rrf",
            20,
            60,
            None,
            None,
        ),
        (
            "hybrid_rrf_rewrite",
            20,
            60,
            DETERMINISTIC_QUERY_REWRITE_VERSION,
            None,
        ),
        (
            "hybrid_rrf_rewrite_rerank",
            20,
            60,
            DETERMINISTIC_QUERY_REWRITE_VERSION,
            HEAD_PRESERVING_RERANK_VERSION,
        ),
    ],
    ids=[
        "vector",
        "hybrid",
        "hybrid-rewrite",
        "hybrid-rewrite-rerank",
    ],
)
def test_strategy_parameters_match_selected_strategy(
    strategy: str,
    expected_candidate_k: int | None,
    expected_rank_constant: int | None,
    expected_rewrite_version: str | None,
    expected_rerank_version: str | None,
) -> None:
    """每种策略只能保留自己真正使用的参数。"""

    parameters = build_strategy_parameters(
        strategy=strategy,
        top_k=3,
        candidate_k=20,
        rank_constant=60,
    )

    assert parameters.strategy == strategy
    assert parameters.top_k == 3
    assert (
        parameters.candidate_k
        == expected_candidate_k
    )
    assert (
        parameters.rank_constant
        == expected_rank_constant
    )
    assert (
        parameters.rewrite_version
        == expected_rewrite_version
    )

    actual_rerank_version = (
        parameters.rerank_parameters.version
        if parameters.rerank_parameters is not None
        else None
    )

    assert (
        actual_rerank_version
        == expected_rerank_version
    )


def test_runtime_rerank_policy_is_built_from_audited_parameters(
) -> None:
    """报告中的版本和深度应转换成真实运行时策略。"""

    parameters = build_strategy_parameters(
        strategy="hybrid_rrf_rewrite_rerank",
        top_k=3,
        candidate_k=20,
        rank_constant=60,
    )

    policy = build_runtime_rerank_policy(
        parameters=parameters
    )

    assert isinstance(
        policy,
        HeadPreservationPolicy,
    )
    assert policy == HeadPreservationPolicy(
        fused_head_k=1,
        vector_head_k=1,
        keyword_head_k=2,
    )


def test_runtime_rerank_policy_is_none_for_plain_strategy(
) -> None:
    """没有重排参数的旧策略必须保持普通RRF行为。"""

    parameters = build_strategy_parameters(
        strategy="hybrid_rrf_rewrite",
        top_k=3,
        candidate_k=20,
        rank_constant=60,
    )

    assert build_runtime_rerank_policy(
        parameters=parameters
    ) is None


def test_runtime_rerank_policy_rejects_version_mismatch(
) -> None:
    """历史报告版本与当前代码不一致时不能静默运行。"""

    parameters = CandidateStrategyParameters(
        strategy="hybrid_rrf_rewrite_rerank",
        top_k=3,
        candidate_k=20,
        rank_constant=60,
        rewrite_version=(
            DETERMINISTIC_QUERY_REWRITE_VERSION
        ),
        rerank_parameters=(
            HeadPreservingRerankParameters(
                version="head-preserving-v999",
                fused_head_k=1,
                vector_head_k=1,
                keyword_head_k=2,
            )
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            "实际头部保留重排版本与"
            "策略参数声明不一致"
        ),
    ):
        build_runtime_rerank_policy(
            parameters=parameters
        )


def test_unsupported_strategy_is_rejected(
) -> None:
    """直接调用参数函数时也不能接受未知策略。"""

    with pytest.raises(
        ValueError,
        match="不支持的候选策略",
    ):
        build_strategy_parameters(
            strategy="unknown",
            top_k=3,
            candidate_k=20,
            rank_constant=60,
        )


def test_default_output_paths_include_strategy_name(
) -> None:
    """省略路径时应为每种策略生成独立文件。"""

    json_output, markdown_output = (
        resolve_output_paths(
            strategy="hybrid_rrf_rewrite",
            json_output=None,
            markdown_output=None,
        )
    )

    assert json_output == Path(
        "docs/"
        "retrieval-strategy-"
        "hybrid-rrf-rewrite.json"
    )
    assert markdown_output == Path(
        "docs/"
        "retrieval-strategy-"
        "hybrid-rrf-rewrite.md"
    )


def test_rerank_default_output_paths_include_full_strategy_name(
) -> None:
    """第四种策略必须使用独立文件，不能覆盖第三种报告。"""

    json_output, markdown_output = (
        resolve_output_paths(
            strategy=(
                "hybrid_rrf_rewrite_rerank"
            ),
            json_output=None,
            markdown_output=None,
        )
    )

    assert json_output == Path(
        "docs/"
        "retrieval-strategy-"
        "hybrid-rrf-rewrite-rerank.json"
    )
    assert markdown_output == Path(
        "docs/"
        "retrieval-strategy-"
        "hybrid-rrf-rewrite-rerank.md"
    )


def test_explicit_output_paths_are_preserved(
    tmp_path: Path,
) -> None:
    """调用方指定的路径不能被默认规则覆盖。"""

    expected_json = (
        tmp_path / "custom.json"
    )
    expected_markdown = (
        tmp_path / "custom.md"
    )

    actual_json, actual_markdown = (
        resolve_output_paths(
            strategy="vector_baseline",
            json_output=expected_json,
            markdown_output=expected_markdown,
        )
    )

    assert actual_json == expected_json
    assert actual_markdown == expected_markdown


def test_vector_markdown_contains_similarity_and_metrics(
) -> None:
    """纯向量报告应显示余弦相似度而不是RRF分数。"""

    markdown = render_markdown_report(
        make_vector_report()
    )

    assert "纯向量基线" in markdown
    assert "生成时间" not in markdown
    assert "Recall@1 | 1.000" in markdown
    assert "Recall@3 | 1.000" in markdown
    assert "余弦相似度：0.800000" in markdown
    assert "RRF分数：" not in markdown

    # 文件名中的竖线进入Markdown表格时必须转义。
    assert "manual\\|test.pdf" in markdown

    # 正文中的换行应在预览中压缩为空格。
    assert (
        "第一行候选正文。 第二行包含更多可审计内容。"
        in markdown
    )


def test_hybrid_markdown_contains_rrf_audit_fields(
) -> None:
    """混合报告应保留两条路径的原始排名和命中词。"""

    markdown = render_markdown_report(
        make_hybrid_report()
    )

    assert "关键词 + 向量 + RRF" in markdown
    assert "生成时间" not in markdown
    assert "RRF分数：0.032000" in markdown
    assert "向量路径：Top-2，相似度 0.720000" in markdown
    assert "关键词路径：Top-1，关键词分数 12.000" in markdown
    assert "关键词命中：err-test-1001、测试" in markdown


def test_rerank_markdown_contains_version_and_head_depths(
) -> None:
    """第四种策略报告必须显示真实重排版本和三个深度。"""

    markdown = render_markdown_report(
        make_hybrid_report(
            strategy=(
                "hybrid_rrf_rewrite_rerank"
            )
        )
    )

    assert (
        "关键词 + 向量 + RRF + 查询改写"
        " + 头部保留重排"
        in markdown
    )
    assert (
        "| 头部保留重排版本 | "
        "head-preserving-v1 |"
        in markdown
    )
    assert "| RRF头部保护深度 | 1 |" in markdown
    assert "| 向量头部保护深度 | 1 |" in markdown
    assert "| 关键词头部保护深度 | 2 |" in markdown
    assert "生成时间" not in markdown


def test_write_reports_creates_json_and_markdown(
    tmp_path: Path,
) -> None:
    """写入函数应创建父目录并保存同一份报告。"""

    report = make_vector_report()
    json_output = (
        tmp_path
        / "nested"
        / "strategy.json"
    )
    markdown_output = (
        tmp_path
        / "nested"
        / "strategy.md"
    )

    write_reports(
        report=report,
        json_output=json_output,
        markdown_output=markdown_output,
    )

    assert json_output.is_file()
    assert markdown_output.is_file()

    json_data = json.loads(
        json_output.read_text(
            encoding="utf-8"
        )
    )
    markdown = markdown_output.read_text(
        encoding="utf-8"
    )

    assert (
        json_data["parameters"]["strategy"]
        == "vector_baseline"
    )
    assert (
        json_data["metrics"]["recall_at_1"]
        == 1.0
    )
    assert "# Week 3 候选检索策略评测" in markdown
