"""RRF纯算法和HybridRetriever编排层的离线测试。

这些测试使用手工构造的DocumentChunk、RetrievedChunk和
KeywordRetrievedChunk，不调用Embedding、Chroma、关键词索引、
LLM或网络服务。

测试目标包括：
1. 确认RRF只使用排名计算、正确合并同一Chunk；
2. 确认融合结果保留两侧审计字段；
3. 确认会破坏排名语义的输入被拒绝；
4. 确认HybridRetriever把请求参数正确传给两条召回路径；
5. 确认同一个检索范围同时约束向量与关键词路径；
6. 确认任一路径失败时不会返回不完整的融合结果；
7. 确认可选头部保留重排发生在完整RRF候选池上。
"""

from hashlib import sha256

import pytest

from app.schemas.retrieval import (
    DocumentChunk,
    HybridRetrievedChunk,
    KeywordRetrievedChunk,
    RetrievedChunk,
)
from app.services.hybrid_retrieval import (
    HybridRetriever,
    reciprocal_rank_fusion,
)
from app.services.head_preserving_reranking import (
    HeadPreservationPolicy,
)
from app.services.retrieval_scope import (
    RetrievalScopeFilter,
)


def make_chunk(
    chunk_id: str,
    *,
    content: str | None = None,
) -> DocumentChunk:
    """根据测试ID创建稳定、合法且可区分的Chunk。"""

    resolved_content = (
        content
        if content is not None
        else f"{chunk_id}测试正文"
    )

    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=sha256(
            chunk_id.encode("utf-8")
        ).hexdigest(),
        source_file=f"{chunk_id}.txt",
        page_or_section=(
            f"section: {chunk_id}"
        ),
        chunk_index=0,
        content_hash=sha256(
            resolved_content.encode("utf-8")
        ).hexdigest(),
        content=resolved_content,
    )


def make_vector_result(
    chunk: DocumentChunk,
    *,
    rank: int,
    similarity: float,
) -> RetrievedChunk:
    """创建一项已经排好名的向量候选。"""

    return RetrievedChunk(
        chunk=chunk,
        similarity=similarity,
        rank=rank,
    )


def make_keyword_result(
    chunk: DocumentChunk,
    *,
    rank: int,
    keyword_score: float,
    matched_identifier: str = (
        "err-test-0001"
    ),
) -> KeywordRetrievedChunk:
    """创建一项带可审计命中依据的关键词候选。"""

    return KeywordRetrievedChunk(
        chunk=chunk,
        keyword_score=keyword_score,
        rank=rank,
        matched_identifiers=(
            matched_identifier,
        ),
    )


class FakeVectorRetriever:
    """记录调用参数并返回预设向量候选的测试替身。"""

    def __init__(
        self,
        results: list[RetrievedChunk],
        *,
        error: Exception | None = None,
    ) -> None:
        """保存预设结果、可选异常和调用记录。"""

        self._results = results
        self._error = error

        # 每次retrieve()调用都会追加一份参数记录。
        # 测试据此确认HybridRetriever如何调用依赖。
        self.calls: list[
            dict[str, object]
        ] = []

    def retrieve(
        self,
        query_embedding: list[float],
        *,
        query_embedding_model: str,
        top_k: int = 3,
        retrieval_scope: (
            RetrievalScopeFilter | None
        ) = None,
    ) -> list[RetrievedChunk]:
        """模拟向量召回，不访问真实Chroma。"""

        self.calls.append(
            {
                # 保存原对象引用，以便测试是否原样传递。
                "query_embedding": query_embedding,
                "query_embedding_model": (
                    query_embedding_model
                ),
                "top_k": top_k,
                # 记录范围对象，验证编排器没有丢失、
                # 重建或只传给其中一条检索路径。
                "retrieval_scope": retrieval_scope,
            }
        )

        if self._error is not None:
            raise self._error

        # 返回新列表，避免测试替身自己的容器
        # 被被测代码意外修改。
        return list(self._results)


class FakeKeywordRetriever:
    """记录调用参数并返回预设关键词候选的测试替身。"""

    def __init__(
        self,
        results: list[KeywordRetrievedChunk],
        *,
        error: Exception | None = None,
    ) -> None:
        """保存预设结果、可选异常和调用记录。"""

        self._results = results
        self._error = error
        self.calls: list[
            dict[str, object]
        ] = []

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 3,
        retrieval_scope: (
            RetrievalScopeFilter | None
        ) = None,
    ) -> list[KeywordRetrievedChunk]:
        """模拟关键词召回，不建立真实关键词索引。"""

        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "retrieval_scope": retrieval_scope,
            }
        )

        if self._error is not None:
            raise self._error

        return list(self._results)


def test_consensus_candidate_accumulates_both_rrf_contributions(
) -> None:
    """双路命中的Chunk应累加两项倒数排名贡献并升到第一。"""

    shared_chunk = make_chunk(
        "chunk-shared"
    )
    keyword_only_chunk = make_chunk(
        "chunk-keyword-only"
    )

    vector_results = [
        make_vector_result(
            shared_chunk,
            rank=1,
            similarity=0.61,
        )
    ]

    keyword_results = [
        make_keyword_result(
            keyword_only_chunk,
            rank=1,
            keyword_score=20.0,
        ),
        make_keyword_result(
            shared_chunk,
            rank=2,
            keyword_score=12.0,
            matched_identifier=(
                "err-net-4001"
            ),
        ),
    ]

    results = reciprocal_rank_fusion(
        vector_results=vector_results,
        keyword_results=keyword_results,
        top_k=3,
    )

    assert isinstance(
        results[0],
        HybridRetrievedChunk,
    )
    assert results[0].chunk.chunk_id == (
        "chunk-shared"
    )
    assert results[0].rank == 1

    # shared在向量列表排1、关键词列表排2。
    assert results[0].rrf_score == (
        pytest.approx(
            (1.0 / 61.0)
            + (1.0 / 62.0)
        )
    )

    # 两条原始路径的信息都不能丢失。
    assert results[0].vector_rank == 1
    assert results[0].vector_similarity == (
        pytest.approx(0.61)
    )
    assert results[0].keyword_rank == 2
    assert results[0].keyword_score == (
        pytest.approx(12.0)
    )
    assert results[0].matched_identifiers == (
        "err-net-4001",
    )


def test_single_source_candidates_keep_only_their_own_fields(
) -> None:
    """向量独占和关键词独占候选都应保留且字段不能串线。"""

    vector_chunk = make_chunk(
        "chunk-b"
    )
    keyword_chunk = make_chunk(
        "chunk-a"
    )

    results = reciprocal_rank_fusion(
        vector_results=[
            make_vector_result(
                vector_chunk,
                rank=1,
                similarity=0.70,
            )
        ],
        keyword_results=[
            make_keyword_result(
                keyword_chunk,
                rank=1,
                keyword_score=12.0,
            )
        ],
        top_k=3,
    )

    # 两项RRF分数、来源数和最好排名均相同，
    # 最终由chunk_id稳定排序。
    assert [
        item.chunk.chunk_id
        for item in results
    ] == [
        "chunk-a",
        "chunk-b",
    ]

    keyword_only = results[0]
    vector_only = results[1]

    assert keyword_only.vector_rank is None
    assert keyword_only.keyword_rank == 1
    assert vector_only.vector_rank == 1
    assert vector_only.keyword_rank is None
    assert vector_only.matched_identifiers == ()


def test_rrf_ignores_vector_similarity_magnitude(
) -> None:
    """向量相似度大小不能代替已经确定的向量排名。"""

    rank_one = make_chunk("chunk-rank-one")
    rank_two = make_chunk("chunk-rank-two")

    results = reciprocal_rank_fusion(
        vector_results=[
            make_vector_result(
                rank_one,
                rank=1,
                similarity=0.10,
            ),
            make_vector_result(
                rank_two,
                rank=2,
                similarity=0.99,
            ),
        ],
        keyword_results=[],
        top_k=2,
    )

    # 测试数据故意让rank=2拥有更高相似度。
    # RRF仍必须以输入rank为准。
    assert [
        item.chunk.chunk_id
        for item in results
    ] == [
        "chunk-rank-one",
        "chunk-rank-two",
    ]


def test_rrf_ignores_keyword_score_magnitude(
) -> None:
    """关键词原始分数大小不能直接进入RRF加法。"""

    rank_one = make_chunk("chunk-keyword-1")
    rank_two = make_chunk("chunk-keyword-2")

    results = reciprocal_rank_fusion(
        vector_results=[],
        keyword_results=[
            make_keyword_result(
                rank_one,
                rank=1,
                keyword_score=1.0,
            ),
            make_keyword_result(
                rank_two,
                rank=2,
                keyword_score=100.0,
            ),
        ],
        top_k=2,
    )

    assert [
        item.chunk.chunk_id
        for item in results
    ] == [
        "chunk-keyword-1",
        "chunk-keyword-2",
    ]


def test_custom_rank_constant_changes_formula(
) -> None:
    """调用方提供的rank_constant必须真正参与RRF公式。"""

    chunk = make_chunk("chunk-custom-k")

    result = reciprocal_rank_fusion(
        vector_results=[
            make_vector_result(
                chunk,
                rank=1,
                similarity=0.50,
            )
        ],
        keyword_results=[],
        top_k=1,
        rank_constant=1,
    )[0]

    assert result.rrf_score == (
        pytest.approx(1.0 / 2.0)
    )


def test_top_k_limits_union_and_reassigns_final_ranks(
) -> None:
    """融合后的候选并集应截断到Top-K并生成连续最终排名。"""

    chunks = [
        make_chunk(f"chunk-{index}")
        for index in range(1, 4)
    ]

    vector_results = [
        make_vector_result(
            chunk,
            rank=index,
            similarity=(0.9 - index * 0.1),
        )
        for index, chunk in enumerate(
            chunks,
            start=1,
        )
    ]

    results = reciprocal_rank_fusion(
        vector_results=vector_results,
        keyword_results=[],
        top_k=2,
    )

    assert len(results) == 2
    assert [
        item.rank
        for item in results
    ] == [1, 2]
    assert [
        item.chunk.chunk_id
        for item in results
    ] == [
        "chunk-1",
        "chunk-2",
    ]


def test_unsorted_inputs_are_sorted_without_mutating_input_lists(
) -> None:
    """函数应尊重rank字段，但不能原地修改调用方列表顺序。"""

    rank_one = make_vector_result(
        make_chunk("chunk-one"),
        rank=1,
        similarity=0.8,
    )
    rank_two = make_vector_result(
        make_chunk("chunk-two"),
        rank=2,
        similarity=0.7,
    )

    vector_results = [
        rank_two,
        rank_one,
    ]

    results = reciprocal_rank_fusion(
        vector_results=vector_results,
        keyword_results=[],
        top_k=2,
    )

    assert [
        item.chunk.chunk_id
        for item in results
    ] == [
        "chunk-one",
        "chunk-two",
    ]

    # sorted()返回新列表，所以原始输入仍保持2、1顺序。
    assert [
        item.rank
        for item in vector_results
    ] == [2, 1]


def test_empty_rankings_return_empty_results(
) -> None:
    """两条检索路径都没有候选时应返回空列表。"""

    assert reciprocal_rank_fusion(
        vector_results=[],
        keyword_results=[],
        top_k=3,
    ) == []


def test_duplicate_vector_chunk_id_is_rejected(
) -> None:
    """同一向量列表不能让一个Chunk获得两次贡献。"""

    chunk = make_chunk("duplicate-vector")

    with pytest.raises(
        ValueError,
        match="vector_results.*重复.*chunk_id",
    ):
        reciprocal_rank_fusion(
            vector_results=[
                make_vector_result(
                    chunk,
                    rank=1,
                    similarity=0.8,
                ),
                make_vector_result(
                    chunk,
                    rank=2,
                    similarity=0.7,
                ),
            ],
            keyword_results=[],
        )


def test_duplicate_keyword_chunk_id_is_rejected(
) -> None:
    """同一关键词列表不能让一个Chunk获得两次贡献。"""

    chunk = make_chunk("duplicate-keyword")

    with pytest.raises(
        ValueError,
        match="keyword_results.*重复.*chunk_id",
    ):
        reciprocal_rank_fusion(
            vector_results=[],
            keyword_results=[
                make_keyword_result(
                    chunk,
                    rank=1,
                    keyword_score=12.0,
                ),
                make_keyword_result(
                    chunk,
                    rank=2,
                    keyword_score=8.0,
                ),
            ],
        )


@pytest.mark.parametrize(
    "broken_source",
    [
        "vector",
        "keyword",
    ],
    ids=[
        "vector-rank-gap",
        "keyword-rank-gap",
    ],
)
def test_rank_sequence_must_be_continuous(
    broken_source: str,
) -> None:
    """候选排名出现1、3缺口时必须拒绝融合。"""

    first_chunk = make_chunk("gap-one")
    third_chunk = make_chunk("gap-three")

    vector_results: list[
        RetrievedChunk
    ] = []
    keyword_results: list[
        KeywordRetrievedChunk
    ] = []

    if broken_source == "vector":
        vector_results = [
            make_vector_result(
                first_chunk,
                rank=1,
                similarity=0.8,
            ),
            make_vector_result(
                third_chunk,
                rank=3,
                similarity=0.6,
            ),
        ]
    else:
        keyword_results = [
            make_keyword_result(
                first_chunk,
                rank=1,
                keyword_score=12.0,
            ),
            make_keyword_result(
                third_chunk,
                rank=3,
                keyword_score=6.0,
            ),
        ]

    with pytest.raises(
        ValueError,
        match="rank.*从1开始连续",
    ):
        reciprocal_rank_fusion(
            vector_results=vector_results,
            keyword_results=keyword_results,
        )


@pytest.mark.parametrize(
    "invalid_source",
    [
        "vector",
        "keyword",
    ],
    ids=[
        "invalid-vector-item",
        "invalid-keyword-item",
    ],
)
def test_result_items_must_match_their_contract(
    invalid_source: str,
) -> None:
    """两份列表中的元素必须分别使用正确Pydantic契约。"""

    vector_results: list[RetrievedChunk] = []
    keyword_results: list[
        KeywordRetrievedChunk
    ] = []

    if invalid_source == "vector":
        vector_results = [
            object(),  # type: ignore[list-item]
        ]
    else:
        keyword_results = [
            object(),  # type: ignore[list-item]
        ]

    with pytest.raises(
        TypeError,
        match="第 0 项必须是",
    ):
        reciprocal_rank_fusion(
            vector_results=vector_results,
            keyword_results=keyword_results,
        )


@pytest.mark.parametrize(
    "invalid_source",
    [
        "vector",
        "keyword",
    ],
    ids=[
        "vector-tuple",
        "keyword-tuple",
    ],
)
def test_ranked_sources_must_be_lists(
    invalid_source: str,
) -> None:
    """公共融合契约不接受元组、字符串或单个结果对象。"""

    vector_results: object = []
    keyword_results: object = []

    if invalid_source == "vector":
        vector_results = ()
    else:
        keyword_results = ()

    with pytest.raises(
        TypeError,
        match="必须是列表",
    ):
        reciprocal_rank_fusion(
            vector_results=(
                vector_results  # type: ignore[arg-type]
            ),
            keyword_results=(
                keyword_results  # type: ignore[arg-type]
            ),
        )


@pytest.mark.parametrize(
    (
        "invalid_arguments",
        "expected_exception",
    ),
    [
        (
            {
                "top_k": True,
            },
            TypeError,
        ),
        (
            {
                "top_k": 0,
            },
            ValueError,
        ),
        (
            {
                "rank_constant": False,
            },
            TypeError,
        ),
        (
            {
                "rank_constant": 0,
            },
            ValueError,
        ),
    ],
    ids=[
        "boolean-top-k",
        "zero-top-k",
        "boolean-rank-constant",
        "zero-rank-constant",
    ],
)
def test_invalid_configuration_is_rejected(
    invalid_arguments: dict[
        str,
        int | bool,
    ],
    expected_exception: type[Exception],
) -> None:
    """Top-K和RRF常数必须是大于等于1的真正整数。"""

    with pytest.raises(
        expected_exception,
    ):
        reciprocal_rank_fusion(
            vector_results=[],
            keyword_results=[],
            **invalid_arguments,
        )


def test_same_chunk_id_with_different_snapshot_is_rejected(
) -> None:
    """两条路径使用不同正文快照时不能按相同ID静默合并。"""

    vector_chunk = make_chunk(
        "shared-id",
        content="旧版本正文",
    )
    keyword_chunk = make_chunk(
        "shared-id",
        content="新版本正文",
    )

    with pytest.raises(
        ValueError,
        match="相同chunk_id.*对应不同Chunk",
    ):
        reciprocal_rank_fusion(
            vector_results=[
                make_vector_result(
                    vector_chunk,
                    rank=1,
                    similarity=0.8,
                )
            ],
            keyword_results=[
                make_keyword_result(
                    keyword_chunk,
                    rank=1,
                    keyword_score=12.0,
                )
            ],
        )


def test_fused_chunk_is_deeply_isolated_from_input(
) -> None:
    """修改输入或输出Chunk都不能跨越融合边界互相污染。"""

    source_result = make_vector_result(
        make_chunk(
            "isolated-chunk",
            content="原始正文",
        ),
        rank=1,
        similarity=0.8,
    )

    fused_result = reciprocal_rank_fusion(
        vector_results=[source_result],
        keyword_results=[],
        top_k=1,
    )[0]

    # 融合完成后修改输入对象。
    source_result.chunk.content = (
        "调用方修改了输入"
    )

    assert fused_result.chunk.content == (
        "原始正文"
    )

    # 再修改返回对象，输入对象也不能随之变化。
    fused_result.chunk.content = (
        "调用方修改了输出"
    )

    assert source_result.chunk.content == (
        "调用方修改了输入"
    )


def test_hybrid_retriever_orchestrates_both_candidate_paths(
) -> None:
    """编排器应以候选深度调用两路检索并返回融合Top-K。"""

    shared_chunk = make_chunk(
        "hybrid-shared"
    )
    vector_only_chunk = make_chunk(
        "hybrid-vector-only"
    )

    vector_retriever = FakeVectorRetriever(
        [
            make_vector_result(
                shared_chunk,
                rank=1,
                similarity=0.72,
            ),
            make_vector_result(
                vector_only_chunk,
                rank=2,
                similarity=0.68,
            ),
        ]
    )
    keyword_retriever = FakeKeywordRetriever(
        [
            make_keyword_result(
                shared_chunk,
                rank=1,
                keyword_score=18.0,
                matched_identifier=(
                    "err-hybrid-1001"
                ),
            )
        ]
    )

    retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        keyword_retriever=(
            keyword_retriever
        ),
        # 首尾空白应在构造时被清理。
        embedding_model=(
            "  embedding-test-model  "
        ),
        candidate_k=5,
        # 使用较小常数，方便精确验证公式。
        rank_constant=1,
    )

    query_embedding = [0.1, 0.2, 0.3]

    results = retriever.retrieve(
        query=(
            "  ERR-HYBRID-1001如何恢复？  "
        ),
        query_embedding=query_embedding,
        top_k=2,
    )

    # 两条路径都按candidate_k=5取候选，
    # 而不是只按最终top_k=2取候选。
    assert len(vector_retriever.calls) == 1
    assert (
        vector_retriever.calls[0][
            "query_embedding"
        ]
        is query_embedding
    )
    assert vector_retriever.calls[0] == {
        "query_embedding": query_embedding,
        "query_embedding_model": (
            "embedding-test-model"
        ),
        "top_k": 5,
        # 不提供范围时必须维持原有全库检索语义。
        "retrieval_scope": None,
    }

    assert keyword_retriever.calls == [
        {
            "query": (
                "ERR-HYBRID-1001如何恢复？"
            ),
            "top_k": 5,
            "retrieval_scope": None,
        }
    ]

    # shared同时位于两条路径第1名，
    # 所以RRF分数为1/(1+1)+1/(1+1)。
    assert len(results) == 2
    assert results[0].chunk.chunk_id == (
        "hybrid-shared"
    )
    assert results[0].rrf_score == (
        pytest.approx(1.0)
    )
    assert results[0].vector_rank == 1
    assert results[0].keyword_rank == 1
    assert results[0].matched_identifiers == (
        "err-hybrid-1001",
    )

    # 第二项只来自向量路径，最终排名仍连续。
    assert results[1].chunk.chunk_id == (
        "hybrid-vector-only"
    )
    assert results[1].rank == 2
    assert results[1].keyword_rank is None


def test_hybrid_retriever_forwards_same_scope_to_both_paths(
) -> None:
    """同一范围对象必须同时传给向量和关键词候选检索。"""

    vector_retriever = FakeVectorRetriever(
        []
    )
    keyword_retriever = FakeKeywordRetriever(
        []
    )

    retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        keyword_retriever=(
            keyword_retriever
        ),
        embedding_model="embedding-test-model",
    )

    # 同时包含文档ID和文件名，代表诊断请求要求
    # 两种条件共同约束本次检索。
    retrieval_scope = RetrievalScopeFilter(
        document_ids=("a" * 64,),
        source_files=("allowed-manual.pdf",),
    )

    results = retriever.retrieve(
        query="范围内故障怎么检查？",
        query_embedding=[0.1, 0.2],
        top_k=3,
        retrieval_scope=retrieval_scope,
    )

    # 两个Fake都返回空候选，所以RRF的预期结果为空。
    assert results == []

    # 使用is验证传递的是同一个不可变范围对象，
    # 而不是两条路径各自重建出可能不一致的条件。
    assert (
        vector_retriever.calls[0][
            "retrieval_scope"
        ]
        is retrieval_scope
    )
    assert (
        keyword_retriever.calls[0][
            "retrieval_scope"
        ]
        is retrieval_scope
    )


def test_hybrid_retriever_rejects_invalid_scope_before_dependencies(
) -> None:
    """无效范围必须在任何候选检索开始前被拒绝。"""

    vector_retriever = FakeVectorRetriever(
        []
    )
    keyword_retriever = FakeKeywordRetriever(
        []
    )

    retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        keyword_retriever=(
            keyword_retriever
        ),
        embedding_model="embedding-test-model",
    )

    with pytest.raises(
        TypeError,
        match=(
            "retrieval_scope必须是"
            "RetrievalScopeFilter或None"
        ),
    ):
        retriever.retrieve(
            query="测试非法范围",
            query_embedding=[0.1],
            # 普通dict没有经过范围契约的清理与校验，
            # 不能直接进入两条底层检索路径。
            retrieval_scope={  # type: ignore[arg-type]
                "source_files": ["manual.pdf"]
            },
        )

    # 必须先拒绝非法公共参数，不能先执行一半检索。
    assert vector_retriever.calls == []
    assert keyword_retriever.calls == []


def test_hybrid_retriever_returns_empty_when_both_paths_are_empty(
) -> None:
    """两路都无候选时，完整编排流程应返回空列表。"""

    vector_retriever = FakeVectorRetriever(
        []
    )
    keyword_retriever = FakeKeywordRetriever(
        []
    )

    retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        keyword_retriever=(
            keyword_retriever
        ),
        embedding_model="embedding-test-model",
    )

    results = retriever.retrieve(
        query="没有命中的测试问题",
        query_embedding=[0.1, 0.2],
        top_k=3,
    )

    assert results == []
    assert len(vector_retriever.calls) == 1
    assert len(keyword_retriever.calls) == 1


@pytest.mark.parametrize(
    (
        "constructor_arguments",
        "expected_exception",
        "expected_message",
    ),
    [
        (
            {"embedding_model": 123},
            TypeError,
            "embedding_model必须是字符串",
        ),
        (
            {"embedding_model": "   "},
            ValueError,
            "embedding_model不能为空",
        ),
        (
            {"candidate_k": True},
            TypeError,
            "candidate_k必须是整数",
        ),
        (
            {"candidate_k": 0},
            ValueError,
            "candidate_k必须大于或等于1",
        ),
        (
            {"rank_constant": False},
            TypeError,
            "rank_constant必须是整数",
        ),
        (
            {"rank_constant": 0},
            ValueError,
            "rank_constant必须大于或等于1",
        ),
        (
            {"rerank_policy": object()},
            TypeError,
            "rerank_policy必须是HeadPreservationPolicy或None",
        ),
    ],
    ids=[
        "non-string-embedding-model",
        "blank-embedding-model",
        "boolean-candidate-k",
        "zero-candidate-k",
        "boolean-rank-constant",
        "zero-rank-constant",
        "invalid-rerank-policy",
    ],
)
def test_hybrid_retriever_rejects_invalid_constructor_configuration(
    constructor_arguments: dict[
        str,
        object,
    ],
    expected_exception: type[Exception],
    expected_message: str,
) -> None:
    """构造器必须在执行检索前拒绝无效固定配置。"""

    arguments: dict[str, object] = {
        "vector_retriever": (
            FakeVectorRetriever([])
        ),
        "keyword_retriever": (
            FakeKeywordRetriever([])
        ),
        "embedding_model": (
            "embedding-test-model"
        ),
    }
    arguments.update(
        constructor_arguments
    )

    with pytest.raises(
        expected_exception,
        match=expected_message,
    ):
        HybridRetriever(
            **arguments,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    (
        "invalid_query",
        "expected_exception",
        "expected_message",
    ),
    [
        (
            123,
            TypeError,
            "query必须是字符串",
        ),
        (
            "   ",
            ValueError,
            "query不能为空",
        ),
    ],
    ids=[
        "non-string-query",
        "blank-query",
    ],
)
def test_hybrid_retriever_rejects_invalid_query_before_dependencies(
    invalid_query: object,
    expected_exception: type[Exception],
    expected_message: str,
) -> None:
    """无效问题应在调用任何候选检索依赖前被拒绝。"""

    vector_retriever = FakeVectorRetriever(
        []
    )
    keyword_retriever = FakeKeywordRetriever(
        []
    )

    retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        keyword_retriever=(
            keyword_retriever
        ),
        embedding_model="embedding-test-model",
    )

    with pytest.raises(
        expected_exception,
        match=expected_message,
    ):
        retriever.retrieve(
            query=(
                invalid_query  # type: ignore[arg-type]
            ),
            query_embedding=[0.1],
        )

    assert vector_retriever.calls == []
    assert keyword_retriever.calls == []


@pytest.mark.parametrize(
    (
        "invalid_top_k",
        "expected_exception",
        "expected_message",
    ),
    [
        (
            True,
            TypeError,
            "top_k必须是整数",
        ),
        (
            0,
            ValueError,
            "top_k必须大于或等于1",
        ),
        (
            4,
            ValueError,
            "top_k不能大于candidate_k",
        ),
    ],
    ids=[
        "boolean-top-k",
        "zero-top-k",
        "top-k-larger-than-candidate-k",
    ],
)
def test_hybrid_retriever_rejects_invalid_top_k_before_dependencies(
    invalid_top_k: int | bool,
    expected_exception: type[Exception],
    expected_message: str,
) -> None:
    """无效最终深度应在两路候选召回开始前被拒绝。"""

    vector_retriever = FakeVectorRetriever(
        []
    )
    keyword_retriever = FakeKeywordRetriever(
        []
    )

    retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        keyword_retriever=(
            keyword_retriever
        ),
        embedding_model="embedding-test-model",
        candidate_k=3,
    )

    with pytest.raises(
        expected_exception,
        match=expected_message,
    ):
        retriever.retrieve(
            query="合法问题",
            query_embedding=[0.1],
            top_k=invalid_top_k,
        )

    assert vector_retriever.calls == []
    assert keyword_retriever.calls == []


def test_hybrid_retriever_stops_when_vector_path_fails(
) -> None:
    """先执行的向量路径失败后，不应继续关键词召回。"""

    vector_retriever = FakeVectorRetriever(
        [],
        error=RuntimeError(
            "vector retrieval failed"
        ),
    )
    keyword_retriever = FakeKeywordRetriever(
        []
    )

    retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        keyword_retriever=(
            keyword_retriever
        ),
        embedding_model="embedding-test-model",
    )

    with pytest.raises(
        RuntimeError,
        match="vector retrieval failed",
    ):
        retriever.retrieve(
            query="测试向量路径错误",
            query_embedding=[0.1],
        )

    assert len(vector_retriever.calls) == 1
    assert keyword_retriever.calls == []


def test_hybrid_retriever_propagates_keyword_path_failure(
) -> None:
    """关键词路径失败时，不应伪造只有向量证据的融合结果。"""

    vector_retriever = FakeVectorRetriever(
        []
    )
    keyword_retriever = FakeKeywordRetriever(
        [],
        error=RuntimeError(
            "keyword retrieval failed"
        ),
    )

    retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        keyword_retriever=(
            keyword_retriever
        ),
        embedding_model="embedding-test-model",
    )

    with pytest.raises(
        RuntimeError,
        match="keyword retrieval failed",
    ):
        retriever.retrieve(
            query="测试关键词路径错误",
            query_embedding=[0.1],
        )

    # 调用顺序是先向量、后关键词，
    # 因而两份调用记录都应存在。
    assert len(vector_retriever.calls) == 1
    assert len(keyword_retriever.calls) == 1


@pytest.mark.parametrize(
    (
        "rerank_policy",
        "expected_chunk_ids",
    ),
    [
        (
            None,
            [
                "shared-head",
                "ordinary-dual-path",
            ],
        ),
        (
            HeadPreservationPolicy(),
            [
                "shared-head",
                "keyword-head",
            ],
        ),
    ],
    ids=[
        "plain-rrf-keeps-fused-top-k",
        "rerank-rescues-keyword-head",
    ],
)
def test_hybrid_retriever_applies_optional_reranking_to_full_fused_pool(
    rerank_policy: HeadPreservationPolicy | None,
    expected_chunk_ids: list[str],
) -> None:
    """可选重排必须看见RRF截断线外的单路头部候选。"""

    shared_head = make_chunk(
        "shared-head"
    )
    ordinary_dual_path = make_chunk(
        "ordinary-dual-path"
    )
    keyword_head = make_chunk(
        "keyword-head"
    )

    # shared-head和ordinary-dual-path都被两条路径召回，
    # 因而各自获得两次RRF贡献，原融合排名为第一和第二。
    vector_retriever = FakeVectorRetriever(
        [
            make_vector_result(
                shared_head,
                rank=1,
                similarity=0.90,
            ),
            make_vector_result(
                ordinary_dual_path,
                rank=2,
                similarity=0.80,
            ),
        ]
    )

    keyword_retriever = FakeKeywordRetriever(
        [
            # keyword-head是关键词Top-1，
            # 但只来自关键词路径，所以原RRF排名第三。
            make_keyword_result(
                keyword_head,
                rank=1,
                keyword_score=12.0,
            ),
            make_keyword_result(
                shared_head,
                rank=2,
                keyword_score=10.0,
            ),
            make_keyword_result(
                ordinary_dual_path,
                rank=3,
                keyword_score=8.0,
            ),
        ]
    )

    retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        keyword_retriever=keyword_retriever,
        embedding_model="embedding-test-model",
        candidate_k=3,
        rank_constant=60,
        # None表示保留原RRF行为；
        # 策略对象表示先融合完整候选池再执行头部保护。
        rerank_policy=rerank_policy,
    )

    results = retriever.retrieve(
        query="关键词头部是否能进入最终结果？",
        query_embedding=[0.1, 0.2],
        top_k=2,
    )

    assert [
        item.chunk.chunk_id
        for item in results
    ] == expected_chunk_ids

    # 无论是否启用重排，对外排名都必须重新连续编号。
    assert [
        item.rank
        for item in results
    ] == [1, 2]
