"""候选检索策略批量执行器的异步离线测试。

本模块使用Fake Embedding、Fake向量检索器和
Fake混合检索器，不访问网络、Chroma或LLM。

测试目标：
1. 四种策略进入各自正确的执行分支；
2. 多道问题只执行一次批量Embedding；
3. 查询改写版本和报告参数保持一致；
4. 全部策略使用同一个Gold判卷规则；
5. 无效输入不会产生不必要的外部调用。
"""

import pytest

from app.schemas.evaluation import (
    ExpectedEvidence,
    GoldQuestion,
)
from app.schemas.retrieval import (
    DocumentChunk,
    EmbeddingVector,
    HybridRetrievedChunk,
    RetrievedChunk,
)
from app.schemas.retrieval_strategy import (
    CandidateStrategyParameters,
    HeadPreservingRerankParameters,
)
from app.services.query_rewriting import (
    DETERMINISTIC_QUERY_REWRITE_VERSION,
)
from app.services.retrieval_strategy_runner import (
    run_candidate_strategy,
)


TEST_CONTENT_HASH = "c" * 64


def make_chunk(
    *,
    chunk_id: str,
    source_file: str,
    page_or_section: str,
) -> DocumentChunk:
    """构造策略测试使用的可追溯Chunk。"""

    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=f"document-{chunk_id}",
        source_file=source_file,
        page_or_section=page_or_section,
        chunk_index=0,
        content_hash=TEST_CONTENT_HASH,
        content=f"{chunk_id}对应的测试证据",
    )


def make_vector_result(
    *,
    chunk_id: str,
    source_file: str,
    page_or_section: str,
) -> RetrievedChunk:
    """构造排名第一的纯向量候选。"""

    return RetrievedChunk(
        chunk=make_chunk(
            chunk_id=chunk_id,
            source_file=source_file,
            page_or_section=page_or_section,
        ),
        similarity=0.80,
        rank=1,
    )


def make_hybrid_result(
    *,
    chunk_id: str,
    source_file: str,
    page_or_section: str,
) -> HybridRetrievedChunk:
    """构造排名第一且来自两条路径的融合候选。"""

    return HybridRetrievedChunk(
        chunk=make_chunk(
            chunk_id=chunk_id,
            source_file=source_file,
            page_or_section=page_or_section,
        ),
        rrf_score=0.032,
        rank=1,
        vector_rank=1,
        vector_similarity=0.75,
        keyword_rank=1,
        keyword_score=4.0,
        matched_lexical_terms=(
            "测试",
        ),
    )


def make_question(
    *,
    question_id: str,
    question: str,
    source_file: str,
    page_or_section: str,
) -> GoldQuestion:
    """构造一条单跳可回答Gold问题。"""

    return GoldQuestion(
        question_id=question_id,
        question=question,
        question_type="single_hop",
        answerable=True,
        reference_answer="测试参考答案",
        expected_evidence=[
            ExpectedEvidence(
                source_file=source_file,
                page_or_section=page_or_section,
            )
        ],
        tags=["test"],
        notes="策略执行器离线测试",
    )


class FakeBatchEmbeddingProvider:
    """记录批量文本并返回预设向量对象。"""

    def __init__(
        self,
        *,
        result: object,
    ) -> None:
        self.result = result
        self.received_batches: list[
            list[str]
        ] = []

    async def embed_texts(
        self,
        texts: list[str],
    ) -> list[EmbeddingVector]:
        """模拟一次批量Embedding调用。"""

        self.received_batches.append(
            list(texts)
        )

        # 错误返回值测试会故意违反Protocol。
        return self.result  # type: ignore[return-value]


class FakeVectorRetriever:
    """逐次返回预设向量候选并记录调用参数。"""

    def __init__(
        self,
        *,
        results_by_call: list[
            list[RetrievedChunk]
        ],
    ) -> None:
        self.results_by_call = results_by_call
        self.received_embeddings: list[
            EmbeddingVector
        ] = []
        self.received_models: list[str] = []
        self.received_top_k_values: list[int] = []

    def retrieve(
        self,
        query_embedding: EmbeddingVector,
        *,
        query_embedding_model: str,
        top_k: int = 3,
    ) -> list[RetrievedChunk]:
        """返回与当前调用位置对应的候选列表。"""

        call_index = len(
            self.received_embeddings
        )

        self.received_embeddings.append(
            list(query_embedding)
        )
        self.received_models.append(
            query_embedding_model
        )
        self.received_top_k_values.append(
            top_k
        )

        return self.results_by_call[
            call_index
        ]


class FakeHybridRetriever:
    """逐次返回预设融合候选并记录查询参数。"""

    def __init__(
        self,
        *,
        results_by_call: object,
    ) -> None:
        self.results_by_call = results_by_call
        self.received_queries: list[str] = []
        self.received_embeddings: list[
            EmbeddingVector
        ] = []
        self.received_top_k_values: list[int] = []

    def retrieve(
        self,
        *,
        query: str,
        query_embedding: EmbeddingVector,
        top_k: int = 3,
    ) -> list[HybridRetrievedChunk]:
        """返回与当前调用位置对应的融合候选。"""

        call_index = len(
            self.received_queries
        )

        self.received_queries.append(query)
        self.received_embeddings.append(
            list(query_embedding)
        )
        self.received_top_k_values.append(
            top_k
        )

        # 某些测试会传入含错误类型的候选。
        return self.results_by_call[
            call_index
        ]  # type: ignore[index,return-value]


def make_two_questions(
) -> list[GoldQuestion]:
    """构造批量向量化测试使用的两道问题。"""

    return [
        make_question(
            question_id="q101",
            question="第一道原始问题",
            source_file="first.md",
            page_or_section="section: First",
        ),
        make_question(
            question_id="q102",
            question="第二道原始问题",
            source_file="second.md",
            page_or_section="section: Second",
        ),
    ]


@pytest.mark.asyncio
async def test_vector_baseline_uses_one_embedding_batch(
) -> None:
    """纯向量策略应批量向量化并且不调用混合检索。"""

    questions = make_two_questions()
    embedding_provider = FakeBatchEmbeddingProvider(
        result=[
            [1.0, 0.0],
            [0.0, 1.0],
        ],
    )
    vector_retriever = FakeVectorRetriever(
        results_by_call=[
            [
                make_vector_result(
                    chunk_id="first-correct",
                    source_file="first.md",
                    page_or_section="section: First",
                )
            ],
            [
                make_vector_result(
                    chunk_id="second-correct",
                    source_file="second.md",
                    page_or_section="section: Second",
                )
            ],
        ],
    )
    hybrid_retriever = FakeHybridRetriever(
        results_by_call=[],
    )
    parameters = CandidateStrategyParameters(
        strategy="vector_baseline",
        top_k=3,
    )

    report = await run_candidate_strategy(
        questions=questions,
        embedding_provider=embedding_provider,
        vector_retriever=vector_retriever,
        hybrid_retriever=hybrid_retriever,
        embedding_model=" embedding-3 ",
        collection_name=" robot_knowledge_v4 ",
        parameters=parameters,
    )

    # 两道题必须合并成一次embed_texts()调用。
    assert embedding_provider.received_batches == [
        [
            "第一道原始问题",
            "第二道原始问题",
        ]
    ]

    assert vector_retriever.received_embeddings == [
        [1.0, 0.0],
        [0.0, 1.0],
    ]
    assert vector_retriever.received_models == [
        "embedding-3",
        "embedding-3",
    ]
    assert vector_retriever.received_top_k_values == [
        3,
        3,
    ]

    # 纯向量分支不能调用混合检索器。
    assert hybrid_retriever.received_queries == []

    assert report.parameters.strategy == (
        "vector_baseline"
    )
    assert report.embedding_model == "embedding-3"
    assert report.collection_name == (
        "robot_knowledge_v4"
    )
    assert report.metrics.recall_at_1 == 1.0
    assert report.metrics.recall_at_3 == 1.0
    assert len(report.results) == 2
    assert isinstance(
        report.results[0].retrieved_chunks[0],
        RetrievedChunk,
    )


@pytest.mark.asyncio
async def test_hybrid_rrf_uses_original_query(
) -> None:
    """普通混合策略必须把原问题交给两条召回路径。"""

    question = make_question(
        question_id="q103",
        question="ERR-NET-4001如何恢复任务？",
        source_file="faults.txt",
        page_or_section="section: ERR-NET-4001",
    )
    embedding_provider = FakeBatchEmbeddingProvider(
        result=[[0.4, 0.6]],
    )
    vector_retriever = FakeVectorRetriever(
        results_by_call=[],
    )
    hybrid_retriever = FakeHybridRetriever(
        results_by_call=[
            [
                make_hybrid_result(
                    chunk_id="network-correct",
                    source_file="faults.txt",
                    page_or_section=(
                        "section: ERR-NET-4001"
                    ),
                )
            ]
        ],
    )
    parameters = CandidateStrategyParameters(
        strategy="hybrid_rrf",
        top_k=3,
        candidate_k=20,
        rank_constant=60,
    )

    report = await run_candidate_strategy(
        questions=[question],
        embedding_provider=embedding_provider,
        vector_retriever=vector_retriever,
        hybrid_retriever=hybrid_retriever,
        embedding_model="embedding-3",
        collection_name="robot_knowledge_v4",
        parameters=parameters,
    )

    assert embedding_provider.received_batches == [
        [question.question]
    ]
    assert hybrid_retriever.received_queries == [
        question.question
    ]
    assert hybrid_retriever.received_embeddings == [
        [0.4, 0.6]
    ]
    assert vector_retriever.received_embeddings == []
    assert report.metrics.recall_at_1 == 1.0
    assert isinstance(
        report.results[0].retrieved_chunks[0],
        HybridRetrievedChunk,
    )


@pytest.mark.asyncio
async def test_rewrite_strategy_uses_rewritten_query(
) -> None:
    """第三种策略应批量向量化并检索改写后的问题。"""

    question = make_question(
        question_id="q104",
        question=(
            "DM260的线缆靠近高压电源时"
            "有什么风险？"
        ),
        source_file="reader-manual.pdf",
        page_or_section="page: 56",
    )
    embedding_provider = FakeBatchEmbeddingProvider(
        result=[[0.3, 0.7]],
    )
    hybrid_retriever = FakeHybridRetriever(
        results_by_call=[
            [
                make_hybrid_result(
                    chunk_id="reader-correct",
                    source_file="reader-manual.pdf",
                    page_or_section="page: 56",
                )
            ]
        ],
    )
    parameters = CandidateStrategyParameters(
        strategy="hybrid_rrf_rewrite",
        top_k=3,
        candidate_k=20,
        rank_constant=60,
        rewrite_version=(
            DETERMINISTIC_QUERY_REWRITE_VERSION
        ),
    )

    report = await run_candidate_strategy(
        questions=[question],
        embedding_provider=(
            embedding_provider
        ),
        vector_retriever=FakeVectorRetriever(
            results_by_call=[],
        ),
        hybrid_retriever=hybrid_retriever,
        embedding_model="embedding-3",
        collection_name="robot_knowledge_v4",
        parameters=parameters,
    )

    rewritten_query = (
        embedding_provider.received_batches[0][0]
    )

    assert rewritten_query != question.question
    assert "DataMan 260" in rewritten_query
    assert "cable" in rewritten_query
    assert (
        "high-voltage power sources"
        in rewritten_query
    )

    # 向量和关键词路径必须使用完全相同的改写文本。
    assert hybrid_retriever.received_queries == [
        rewritten_query
    ]
    assert report.parameters.rewrite_version == (
        DETERMINISTIC_QUERY_REWRITE_VERSION
    )
    assert report.metrics.recall_at_1 == 1.0


@pytest.mark.asyncio
async def test_rewrite_rerank_strategy_uses_rewritten_query_and_reports_policy(
) -> None:
    """第四种策略应改写问题并完整保留重排审计参数。"""

    question = make_question(
        question_id="q105",
        question=(
            "DM260的线缆靠近高压电源时"
            "有什么风险？"
        ),
        source_file="reader-manual.pdf",
        page_or_section="page: 56",
    )

    embedding_provider = FakeBatchEmbeddingProvider(
        result=[[0.25, 0.75]],
    )

    hybrid_retriever = FakeHybridRetriever(
        results_by_call=[
            [
                make_hybrid_result(
                    chunk_id="reranked-correct",
                    source_file="reader-manual.pdf",
                    page_or_section="page: 56",
                )
            ]
        ],
    )

    rerank_parameters = (
        HeadPreservingRerankParameters(
            version="head-preserving-v1",
            fused_head_k=1,
            vector_head_k=1,
            keyword_head_k=2,
        )
    )

    parameters = CandidateStrategyParameters(
        strategy=(
            "hybrid_rrf_rewrite_rerank"
        ),
        top_k=3,
        candidate_k=20,
        rank_constant=60,
        rewrite_version=(
            DETERMINISTIC_QUERY_REWRITE_VERSION
        ),
        rerank_parameters=rerank_parameters,
    )

    report = await run_candidate_strategy(
        questions=[question],
        embedding_provider=embedding_provider,
        vector_retriever=FakeVectorRetriever(
            results_by_call=[],
        ),
        hybrid_retriever=hybrid_retriever,
        embedding_model="embedding-3",
        collection_name="robot_knowledge_v4",
        parameters=parameters,
    )

    rewritten_query = (
        embedding_provider.received_batches[0][0]
    )

    # 第四种策略与第三种策略一样，
    # 必须先使用确定性规则改写查询。
    assert rewritten_query != question.question
    assert "DataMan 260" in rewritten_query
    assert "high-voltage power sources" in (
        rewritten_query
    )

    # FakeHybridRetriever代表已经配置重排策略的
    # 真实HybridRetriever。执行器必须把同一份改写文本
    # 交给混合检索，并使用相同文本生成查询向量。
    assert hybrid_retriever.received_queries == [
        rewritten_query
    ]
    assert (
        hybrid_retriever.received_embeddings
        == [[0.25, 0.75]]
    )

    # 报告必须保留完整策略参数，不能在执行后退化成
    # 普通hybrid_rrf_rewrite报告。
    assert report.parameters.strategy == (
        "hybrid_rrf_rewrite_rerank"
    )
    assert (
        report.parameters.rerank_parameters
        == rerank_parameters
    )
    assert report.metrics.recall_at_1 == 1.0


@pytest.mark.asyncio
async def test_rewrite_version_mismatch_stops_before_embedding(
) -> None:
    """报告声明版本与实际规则不一致时不能运行评测。"""

    embedding_provider = FakeBatchEmbeddingProvider(
        result=[[1.0, 0.0]],
    )
    parameters = CandidateStrategyParameters(
        strategy="hybrid_rrf_rewrite",
        top_k=3,
        candidate_k=20,
        rank_constant=60,
        rewrite_version="deterministic-v999",
    )

    with pytest.raises(
        ValueError,
        match="实际查询改写版本与策略参数声明不一致",
    ):
        await run_candidate_strategy(
            questions=make_two_questions(),
            embedding_provider=embedding_provider,
            vector_retriever=FakeVectorRetriever(
                results_by_call=[],
            ),
            hybrid_retriever=FakeHybridRetriever(
                results_by_call=[],
            ),
            embedding_model="embedding-3",
            collection_name="robot_knowledge_v4",
            parameters=parameters,
        )

    assert embedding_provider.received_batches == []


@pytest.mark.asyncio
async def test_empty_questions_stop_before_embedding(
) -> None:
    """空Gold列表不能产生Embedding请求。"""

    embedding_provider = FakeBatchEmbeddingProvider(
        result=[],
    )

    with pytest.raises(
        ValueError,
        match="questions不能为空",
    ):
        await run_candidate_strategy(
            questions=[],
            embedding_provider=embedding_provider,
            vector_retriever=FakeVectorRetriever(
                results_by_call=[],
            ),
            hybrid_retriever=FakeHybridRetriever(
                results_by_call=[],
            ),
            embedding_model="embedding-3",
            collection_name="robot_knowledge_v4",
            parameters=CandidateStrategyParameters(
                strategy="vector_baseline",
                top_k=3,
            ),
        )

    assert embedding_provider.received_batches == []


@pytest.mark.asyncio
async def test_duplicate_question_ids_stop_before_embedding(
) -> None:
    """重复题号必须在产生API费用前被拒绝。"""

    question = make_two_questions()[0]
    embedding_provider = FakeBatchEmbeddingProvider(
        result=[],
    )

    with pytest.raises(
        ValueError,
        match="不能包含重复的question_id",
    ):
        await run_candidate_strategy(
            questions=[
                question,
                question.model_copy(deep=True),
            ],
            embedding_provider=embedding_provider,
            vector_retriever=FakeVectorRetriever(
                results_by_call=[],
            ),
            hybrid_retriever=FakeHybridRetriever(
                results_by_call=[],
            ),
            embedding_model="embedding-3",
            collection_name="robot_knowledge_v4",
            parameters=CandidateStrategyParameters(
                strategy="vector_baseline",
                top_k=3,
            ),
        )

    assert embedding_provider.received_batches == []


@pytest.mark.asyncio
async def test_non_list_embedding_result_is_rejected(
) -> None:
    """Embedding Provider违反列表契约时应报错。"""

    with pytest.raises(
        TypeError,
        match="embedding_provider必须返回列表",
    ):
        await run_candidate_strategy(
            questions=make_two_questions(),
            embedding_provider=(
                FakeBatchEmbeddingProvider(
                    result=(),
                )
            ),
            vector_retriever=FakeVectorRetriever(
                results_by_call=[],
            ),
            hybrid_retriever=FakeHybridRetriever(
                results_by_call=[],
            ),
            embedding_model="embedding-3",
            collection_name="robot_knowledge_v4",
            parameters=CandidateStrategyParameters(
                strategy="vector_baseline",
                top_k=3,
            ),
        )


@pytest.mark.asyncio
async def test_embedding_count_mismatch_is_rejected(
) -> None:
    """查询和向量数量不一致时不能漏评Gold问题。"""

    vector_retriever = FakeVectorRetriever(
        results_by_call=[],
    )

    with pytest.raises(
        ValueError,
        match="查询向量数量与Gold问题数量不一致",
    ):
        await run_candidate_strategy(
            questions=make_two_questions(),
            embedding_provider=(
                FakeBatchEmbeddingProvider(
                    # 两道题却只返回一个向量。
                    result=[[1.0, 0.0]],
                )
            ),
            vector_retriever=vector_retriever,
            hybrid_retriever=FakeHybridRetriever(
                results_by_call=[],
            ),
            embedding_model="embedding-3",
            collection_name="robot_knowledge_v4",
            parameters=CandidateStrategyParameters(
                strategy="vector_baseline",
                top_k=3,
            ),
        )

    assert vector_retriever.received_embeddings == []


@pytest.mark.parametrize(
    (
        "embedding_model",
        "expected_exception",
        "expected_message",
    ),
    [
        (
            None,
            TypeError,
            "embedding_model必须是字符串",
        ),
        (
            "   ",
            ValueError,
            "embedding_model不能为空",
        ),
    ],
    ids=[
        "non-string",
        "blank",
    ],
)
@pytest.mark.asyncio
async def test_invalid_embedding_model_stops_before_embedding(
    embedding_model: object,
    expected_exception: type[Exception],
    expected_message: str,
) -> None:
    """无效模型名不能触发Embedding调用。"""

    embedding_provider = FakeBatchEmbeddingProvider(
        result=[],
    )

    with pytest.raises(
        expected_exception,
        match=expected_message,
    ):
        await run_candidate_strategy(
            questions=make_two_questions(),
            embedding_provider=embedding_provider,
            vector_retriever=FakeVectorRetriever(
                results_by_call=[],
            ),
            hybrid_retriever=FakeHybridRetriever(
                results_by_call=[],
            ),
            embedding_model=(
                embedding_model  # type: ignore[arg-type]
            ),
            collection_name="robot_knowledge_v4",
            parameters=CandidateStrategyParameters(
                strategy="vector_baseline",
                top_k=3,
            ),
        )

    assert embedding_provider.received_batches == []


@pytest.mark.parametrize(
    (
        "collection_name",
        "expected_exception",
        "expected_message",
    ),
    [
        (
            None,
            TypeError,
            "collection_name必须是字符串",
        ),
        (
            "   ",
            ValueError,
            "collection_name不能为空",
        ),
    ],
    ids=[
        "non-string",
        "blank",
    ],
)
@pytest.mark.asyncio
async def test_invalid_collection_name_stops_before_embedding(
    collection_name: object,
    expected_exception: type[Exception],
    expected_message: str,
) -> None:
    """无效Collection名称不能触发Embedding调用。"""

    embedding_provider = FakeBatchEmbeddingProvider(
        result=[],
    )

    with pytest.raises(
        expected_exception,
        match=expected_message,
    ):
        await run_candidate_strategy(
            questions=make_two_questions(),
            embedding_provider=embedding_provider,
            vector_retriever=FakeVectorRetriever(
                results_by_call=[],
            ),
            hybrid_retriever=FakeHybridRetriever(
                results_by_call=[],
            ),
            embedding_model="embedding-3",
            collection_name=(
                collection_name  # type: ignore[arg-type]
            ),
            parameters=CandidateStrategyParameters(
                strategy="vector_baseline",
                top_k=3,
            ),
        )

    assert embedding_provider.received_batches == []
