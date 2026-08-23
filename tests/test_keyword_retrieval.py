"""关键词特征提取与Top-K检索层的离线测试。

本文件测试：
1. 查询或文档文本如何被拆成不同类别的关键词；
2. 普通英文词和中文二字、三字词如何生成；
3. 停用词、重复词和结构化词如何被过滤；
4. 下层归一化异常是否通过公共入口继续向上传递；
5. 关键词权重、候选过滤、Top-K和稳定排序；
6. 索引输入校验以及Chunk快照隔离。

这里不测试RRF融合；RRF属于下一层独立模块。
"""

from dataclasses import FrozenInstanceError
from hashlib import sha256

import pytest

from app.schemas.retrieval import (
    DocumentChunk,
    KeywordRetrievedChunk,
)
from app.services.keyword_retrieval import (
    KeywordFeatures,
    KeywordRetriever,
    extract_keyword_features,
)
from app.services.retrieval_scope import (
    RetrievalScopeFilter,
)


def make_chunk(
    *,
    chunk_id: str,
    content: str,
    document_id: str | None = None,
    source_file: str = "knowledge.txt",
    page_or_section: str = "section: test",
    chunk_index: int = 0,
) -> DocumentChunk:
    """创建满足真实Pydantic契约的测试Chunk。"""

    # hexdigest()返回64位小写十六进制SHA-256摘要，
    # 正好满足DocumentChunk.content_hash的字段约束。
    content_hash = sha256(
        content.encode("utf-8")
    ).hexdigest()

    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=(
            document_id
            if document_id is not None
            else f"document-{chunk_id}"
        ),
        source_file=source_file,
        page_or_section=page_or_section,
        chunk_index=chunk_index,
        content_hash=content_hash,
        content=content,
    )


def test_structured_terms_are_classified_separately(
) -> None:
    """错误码、型号和数值应进入各自的精确特征类别。"""

    # 通过公共函数输入一段同时包含四类特征的真实查询形态。
    result = extract_keyword_features(
        "ERR_NET_4001 网络恢复任务 DM260 1500 ms"
    )

    # 公共入口应返回声明的数据契约，而不是普通字典。
    assert isinstance(
        result,
        KeywordFeatures,
    )

    # 下层归一化结果应被完整保留，方便调试和后续建索引。
    assert result.normalized_text == (
        "err-net-4001 网络恢复任务 "
        "dataman-260 1500ms"
    )

    # 三类具有精确意义的词分别进入独立字段。
    assert result.identifiers == (
        "err-net-4001",
    )
    assert result.model_aliases == (
        "dataman-260",
    )
    assert result.numeric_terms == (
        "1500ms",
    )

    # 结构化词已经拥有独立类别，不能再次进入普通词，
    # 否则后续评分时同一条信息会被重复加分。
    assert "err-net-4001" not in (
        result.lexical_terms
    )
    assert "dataman-260" not in (
        result.lexical_terms
    )
    assert "1500ms" not in (
        result.lexical_terms
    )


def test_chinese_text_generates_ordered_bigrams_and_trigrams(
) -> None:
    """连续中文应生成按原文位置排列的二字词和三字词。"""

    result = extract_keyword_features(
        "电机过温"
    )

    # “电机过温”共有三个二字窗口和两个三字窗口。
    # 排序规则是：先按起始位置，再按词长排序。
    assert result.lexical_terms == (
        "电机",
        "电机过",
        "机过",
        "机过温",
        "过温",
    )


def test_ascii_terms_are_normalized_filtered_and_deduplicated(
) -> None:
    """英文词应统一大小写、删除停用词并按首次出现去重。"""

    result = extract_keyword_features(
        "Motor controller AND motor feedback"
    )

    # Motor先经casefold()变成motor；
    # and属于停用词；第二个motor属于重复词。
    assert result.lexical_terms == (
        "motor",
        "controller",
        "feedback",
    )


def test_single_character_ascii_terms_are_filtered(
) -> None:
    """单字符英文词不应成为低区分度的检索关键词。"""

    result = extract_keyword_features(
        "A motor x controller"
    )

    # a虽然也是英文停用词，x则不是；
    # 两者都因为只有一个字符而不会进入结果。
    assert result.lexical_terms == (
        "motor",
        "controller",
    )


def test_exact_chinese_stop_term_is_removed(
) -> None:
    """完全等于停用词的中文N-gram应被过滤。"""

    result = extract_keyword_features(
        "如何"
    )

    # “如何”只表示提问方式，不能区分知识库文档。
    assert result.lexical_terms == ()


def test_longer_context_around_stop_term_is_preserved(
) -> None:
    """停用词过滤不能误删包含上下文的更长三字词。"""

    result = extract_keyword_features(
        "后如何"
    )

    # 二字词“如何”被删除；
    # “后如”和“后如何”仍携带上下文，应继续保留。
    assert result.lexical_terms == (
        "后如",
        "后如何",
    )


def test_duplicate_chinese_ngrams_keep_first_order(
) -> None:
    """重复中文N-gram应去重，并保留第一次出现的位置顺序。"""

    result = extract_keyword_features(
        "电机过温电机"
    )

    # “电机”在开头和结尾各出现一次，
    # 输出中只能保留第一次出现时建立的那一项。
    assert result.lexical_terms == (
        "电机",
        "电机过",
        "机过",
        "机过温",
        "过温",
        "过温电",
        "温电",
        "温电机",
    )


@pytest.mark.parametrize(
    "blank_text",
    [
        "",
        "   ",
    ],
    ids=[
        "empty",
        "spaces-only",
    ],
)
def test_blank_input_error_is_propagated(
    blank_text: str,
) -> None:
    """公共特征入口应保留下层对空文本的明确拒绝。"""

    # extract_keyword_features()先调用
    # normalize_retrieval_text()；后者发现空文本后
    # 抛出的ValueError不应被吞掉或替换。
    with pytest.raises(
        ValueError,
        match="不能为空",
    ):
        extract_keyword_features(
            blank_text
        )


def test_non_string_input_error_is_propagated(
) -> None:
    """普通Python调用传入非字符串时应得到清楚的类型错误。"""

    with pytest.raises(
        TypeError,
        match="必须是字符串",
    ):
        extract_keyword_features(
            123  # type: ignore[arg-type]
        )


def test_keyword_features_are_immutable(
) -> None:
    """已生成的特征快照不能在建索引或评分途中被改写。"""

    result = extract_keyword_features(
        "motor feedback"
    )

    # KeywordFeatures使用@dataclass(frozen=True)，
    # 给字段重新赋值时应抛出FrozenInstanceError。
    with pytest.raises(
        FrozenInstanceError,
    ):
        result.lexical_terms = (  # type: ignore[misc]
            "changed",
        )


def test_retrieve_scores_all_match_categories(
) -> None:
    """检索器应按固定权重累计四类命中分数。"""

    retriever = KeywordRetriever(
        chunks=[
            make_chunk(
                chunk_id="chunk-all-features",
                content=(
                    "ERR_NET_4001 DM260 "
                    "1500 ms 网络恢复"
                ),
            )
        ]
    )

    results = retriever.retrieve(
        "err-net-4001 DataMan 260 "
        "1500ms 网络恢复",
        top_k=3,
    )

    assert len(results) == 1

    result = results[0]

    # 公共方法必须返回关键词检索专用契约。
    assert isinstance(
        result,
        KeywordRetrievedChunk,
    )

    assert result.rank == 1
    assert result.matched_identifiers == (
        "err-net-4001",
    )
    assert result.matched_model_aliases == (
        "dataman-260",
    )
    assert result.matched_numeric_terms == (
        "1500ms",
    )
    assert result.matched_lexical_terms == (
        "网络",
        "网络恢",
        "络恢",
        "络恢复",
        "恢复",
    )

    # 12分错误码 + 8分型号 + 6分数值
    # + 5个普通词 × 0.5分 = 28.5分。
    #
    # approx()允许极小的浮点计算误差。
    assert result.keyword_score == (
        pytest.approx(28.5)
    )


def test_lexical_score_is_capped(
) -> None:
    """很长的普通词查询最多只能获得4分普通词分数。"""

    lexical_text = (
        "alpha beta gamma delta epsilon "
        "zeta theta lambda omega"
    )

    retriever = KeywordRetriever(
        chunks=[
            make_chunk(
                chunk_id="chunk-many-words",
                content=lexical_text,
            )
        ]
    )

    result = retriever.retrieve(
        lexical_text,
        top_k=1,
    )[0]

    # 9 × 0.5本来是4.5，
    # 但MAX_LEXICAL_MATCH_SCORE把它限制为4.0。
    assert len(
        result.matched_lexical_terms
    ) == 9
    assert result.keyword_score == (
        pytest.approx(4.0)
    )


def test_structured_weights_determine_top_k_order(
) -> None:
    """错误码、型号和数值应按12、8、6分顺序排序。"""

    chunks = [
        make_chunk(
            chunk_id="chunk-numeric",
            content="控制器等待1500 ms",
        ),
        make_chunk(
            chunk_id="chunk-model",
            content="DataMan 260参考资料",
        ),
        make_chunk(
            chunk_id="chunk-identifier",
            content="故障代码ERR_NET_4001",
        ),
    ]

    retriever = KeywordRetriever(
        chunks=chunks
    )

    results = retriever.retrieve(
        "ERR-NET-4001 DM260 1500ms",
        top_k=2,
    )

    # top_k=2只保留分数最高的错误码和型号结果。
    assert [
        item.chunk.chunk_id
        for item in results
    ] == [
        "chunk-identifier",
        "chunk-model",
    ]

    assert [
        item.keyword_score
        for item in results
    ] == [
        pytest.approx(12.0),
        pytest.approx(8.0),
    ]

    assert [
        item.rank
        for item in results
    ] == [1, 2]


def test_lexical_only_candidate_requires_two_matches(
) -> None:
    """没有结构化命中时，单个普通词不能形成候选。"""

    retriever = KeywordRetriever(
        chunks=[
            make_chunk(
                chunk_id="chunk-one-word",
                content="motor",
            ),
            make_chunk(
                chunk_id="chunk-two-words",
                content="motor feedback",
            ),
        ]
    )

    results = retriever.retrieve(
        "motor feedback",
        top_k=3,
    )

    assert [
        item.chunk.chunk_id
        for item in results
    ] == [
        "chunk-two-words",
    ]

    assert results[0].matched_lexical_terms == (
        "motor",
        "feedback",
    )
    assert results[0].keyword_score == (
        pytest.approx(1.0)
    )


def test_equal_scores_use_chunk_id_as_stable_tie_breaker(
) -> None:
    """完全同分时结果顺序应与输入顺序无关。"""

    # 故意把chunk-b放在chunk-a之前，
    # 验证排序不是简单沿用构造时的列表顺序。
    retriever = KeywordRetriever(
        chunks=[
            make_chunk(
                chunk_id="chunk-b",
                content="motor feedback",
            ),
            make_chunk(
                chunk_id="chunk-a",
                content="motor feedback",
            ),
        ]
    )

    results = retriever.retrieve(
        "motor feedback",
        top_k=3,
    )

    assert [
        item.chunk.chunk_id
        for item in results
    ] == [
        "chunk-a",
        "chunk-b",
    ]

    assert [
        item.rank
        for item in results
    ] == [1, 2]


def test_empty_index_returns_empty_results(
) -> None:
    """没有Chunk的关键词索引应正常返回空列表。"""

    retriever = KeywordRetriever(
        chunks=[]
    )

    assert retriever.retrieve(
        "motor feedback"
    ) == []


def test_query_without_searchable_features_returns_empty_results(
) -> None:
    """只含停用词的查询不应产生伪关键词候选。"""

    retriever = KeywordRetriever(
        chunks=[
            make_chunk(
                chunk_id="chunk-normal",
                content="motor feedback",
            )
        ]
    )

    # “如何”是精确中文停用词，
    # 特征提取后四类特征都为空。
    assert retriever.retrieve(
        "如何"
    ) == []


def test_source_file_participates_in_keyword_index(
) -> None:
    """型号出现在文件名中时也应支持精确召回。"""

    retriever = KeywordRetriever(
        chunks=[
            make_chunk(
                chunk_id="chunk-model-file",
                source_file=(
                    "DM260-reference.pdf"
                ),
                content="optical reader manual",
            )
        ]
    )

    result = retriever.retrieve(
        "DataMan 260",
        top_k=1,
    )[0]

    assert result.chunk.chunk_id == (
        "chunk-model-file"
    )
    assert result.matched_model_aliases == (
        "dataman-260",
    )
    assert result.keyword_score == (
        pytest.approx(8.0)
    )


def test_page_number_does_not_become_numeric_evidence(
) -> None:
    """page_or_section中的页码不能污染工程数值检索。"""

    retriever = KeywordRetriever(
        chunks=[
            make_chunk(
                chunk_id="chunk-page-1500",
                page_or_section="page: 1500",
                content="unrelated material",
            )
        ]
    )

    # 1500只存在于来源页码中，
    # 没有出现在文件名或正文中，所以不能被召回。
    assert retriever.retrieve(
        "1500",
        top_k=3,
    ) == []


def test_duplicate_chunk_id_is_rejected(
) -> None:
    """索引不能接受两个身份相同的Chunk。"""

    first_chunk = make_chunk(
        chunk_id="duplicate-id",
        content="first content",
    )
    second_chunk = make_chunk(
        chunk_id="duplicate-id",
        content="second content",
    )

    with pytest.raises(
        ValueError,
        match="重复.*chunk_id",
    ):
        KeywordRetriever(
            chunks=[
                first_chunk,
                second_chunk,
            ]
        )


def test_non_chunk_index_item_is_rejected(
) -> None:
    """类型注解之外还必须执行运行时Chunk类型检查。"""

    with pytest.raises(
        TypeError,
        match="第 0 项.*DocumentChunk",
    ):
        KeywordRetriever(
            chunks=[
                object(),  # type: ignore[list-item]
            ]
        )


@pytest.mark.parametrize(
    "invalid_top_k",
    [
        True,
        1.5,
        "3",
    ],
    ids=[
        "boolean",
        "float",
        "string",
    ],
)
def test_retrieve_rejects_non_integer_top_k(
    invalid_top_k: object,
) -> None:
    """bool、浮点数和字符串不能作为Top-K数量。"""

    retriever = KeywordRetriever(
        chunks=[]
    )

    with pytest.raises(
        TypeError,
        match="top_k.*整数",
    ):
        retriever.retrieve(
            "motor feedback",
            top_k=invalid_top_k,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "invalid_top_k",
    [
        0,
        -1,
    ],
    ids=[
        "zero",
        "negative",
    ],
)
def test_retrieve_rejects_non_positive_top_k(
    invalid_top_k: int,
) -> None:
    """Top-K必须至少请求一个结果。"""

    retriever = KeywordRetriever(
        chunks=[]
    )

    with pytest.raises(
        ValueError,
        match="top_k.*大于或等于1",
    ):
        retriever.retrieve(
            "motor feedback",
            top_k=invalid_top_k,
        )


@pytest.mark.parametrize(
    "blank_query",
    [
        "",
        "   ",
    ],
    ids=[
        "empty",
        "spaces-only",
    ],
)
def test_retrieve_propagates_blank_query_error(
    blank_query: str,
) -> None:
    """检索入口应保留归一化层的空查询错误。"""

    retriever = KeywordRetriever(
        chunks=[]
    )

    with pytest.raises(
        ValueError,
        match="不能为空",
    ):
        retriever.retrieve(
            blank_query
        )


def test_retrieve_propagates_non_string_query_error(
) -> None:
    """检索入口应保留归一化层的查询类型错误。"""

    retriever = KeywordRetriever(
        chunks=[]
    )

    # Python类型注解不会在运行时自动拒绝整数，
    # 因而必须验证公共retrieve()调用链最终会抛出TypeError。
    with pytest.raises(
        TypeError,
        match="必须是字符串",
    ):
        retriever.retrieve(
            123  # type: ignore[arg-type]
        )


def test_index_and_returned_chunks_are_isolated_copies(
) -> None:
    """外部修改输入或返回Chunk都不能污染内部索引快照。"""

    original_chunk = make_chunk(
        chunk_id="chunk-isolated",
        content="motor feedback",
    )

    retriever = KeywordRetriever(
        chunks=[original_chunk]
    )

    # 索引已经建立后修改调用方原始对象。
    original_chunk.content = (
        "caller changed original"
    )

    first_result = retriever.retrieve(
        "motor feedback",
        top_k=1,
    )[0]

    assert first_result.chunk.content == (
        "motor feedback"
    )

    # 再修改第一次返回给调用方的对象。
    first_result.chunk.content = (
        "caller changed result"
    )

    second_result = retriever.retrieve(
        "motor feedback",
        top_k=1,
    )[0]

    # 第二次结果仍来自未受污染的内部快照。
    assert second_result.chunk.content == (
        "motor feedback"
    )


def test_source_scope_filters_before_top_k(
) -> None:
    """范围外高分候选不能挤掉范围内低分候选。"""

    retriever = KeywordRetriever(
        chunks=[
            # 精确错误码命中12分，
            # 但该Chunk位于范围外。
            make_chunk(
                chunk_id="outside-high-score",
                source_file="outside.txt",
                content="ERR-NET-4001",
            ),
            # 两个普通词只获得1分，
            # 但它位于本次允许范围内。
            make_chunk(
                chunk_id="inside-lower-score",
                source_file="allowed.txt",
                content="motor feedback",
            ),
        ]
    )
    scope = RetrievalScopeFilter(
        source_files=("allowed.txt",),
    )

    results = retriever.retrieve(
        "ERR-NET-4001 motor feedback",
        top_k=1,
        retrieval_scope=scope,
    )

    # 如果实现错误地先取全库Top-1再过滤，
    # outside-high-score会先占据唯一名额，
    # 最后得到空结果。
    #
    # 当前预期证明过滤发生在评分和Top-K之前。
    assert [
        result.chunk.chunk_id
        for result in results
    ] == ["inside-lower-score"]
    assert results[0].rank == 1


def test_document_id_scope_filters_keyword_candidates(
) -> None:
    """document_ids应限制可参与关键词打分的文档。"""

    allowed_document_id = "4" * 64
    other_document_id = "5" * 64
    retriever = KeywordRetriever(
        chunks=[
            make_chunk(
                chunk_id="allowed-document-chunk",
                document_id=allowed_document_id,
                content="motor feedback",
            ),
            make_chunk(
                chunk_id="other-document-chunk",
                document_id=other_document_id,
                content="motor feedback",
            ),
        ]
    )

    results = retriever.retrieve(
        "motor feedback",
        top_k=3,
        retrieval_scope=(
            RetrievalScopeFilter(
                document_ids=(
                    allowed_document_id,
                ),
            )
        ),
    )

    assert [
        result.chunk.chunk_id
        for result in results
    ] == ["allowed-document-chunk"]


def test_combined_scope_requires_document_and_file(
) -> None:
    """两类范围同时存在时Chunk必须同时满足二者。"""

    allowed_document_id = "6" * 64
    other_document_id = "7" * 64
    retriever = KeywordRetriever(
        chunks=[
            # 同时满足文档ID和文件名。
            make_chunk(
                chunk_id="matches-both",
                document_id=allowed_document_id,
                source_file="allowed.txt",
                content="motor feedback",
            ),
            # 只满足文档ID。
            make_chunk(
                chunk_id="matches-document-only",
                document_id=allowed_document_id,
                source_file="other.txt",
                content="motor feedback",
            ),
            # 只满足文件名。
            make_chunk(
                chunk_id="matches-file-only",
                document_id=other_document_id,
                source_file="allowed.txt",
                content="motor feedback",
            ),
        ]
    )
    scope = RetrievalScopeFilter(
        document_ids=(
            allowed_document_id,
        ),
        source_files=("allowed.txt",),
    )

    results = retriever.retrieve(
        "motor feedback",
        retrieval_scope=scope,
    )

    assert [
        result.chunk.chunk_id
        for result in results
    ] == ["matches-both"]


def test_scope_without_matching_chunks_returns_empty(
) -> None:
    """索引有命中但范围内无Chunk时应返回空列表。"""

    retriever = KeywordRetriever(
        chunks=[
            make_chunk(
                chunk_id="outside-only",
                source_file="outside.txt",
                content="ERR-NET-4001",
            )
        ]
    )

    results = retriever.retrieve(
        "ERR-NET-4001",
        retrieval_scope=(
            RetrievalScopeFilter(
                source_files=(
                    "allowed.txt",
                ),
            )
        ),
    )

    assert results == []


def test_retrieve_rejects_invalid_scope_type(
) -> None:
    """普通dict不能冒充经过校验的范围过滤器。"""

    retriever = KeywordRetriever(
        chunks=[]
    )

    with pytest.raises(
        TypeError,
        match=(
            "retrieval_scope必须是"
            "RetrievalScopeFilter或None"
        ),
    ):
        retriever.retrieve(
            "motor feedback",
            retrieval_scope={  # type: ignore[arg-type]
                "source_files": [
                    "allowed.txt"
                ]
            },
        )
