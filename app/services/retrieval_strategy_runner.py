"""候选检索策略的批量执行编排。

本模块负责：
1. 接收一组经过校验的Gold问题；
2. 根据策略决定是否执行查询改写；
3. 批量生成全部查询向量；
4. 调用纯向量或混合检索；
5. 使用统一判卷器评价每道题；
6. 构造经过Pydantic校验的策略报告。

本模块不创建Embedding客户端、不读取JSONL、
不直接连接Chroma，也不输出JSON或Markdown文件。
这些外部工作由后续命令行脚本负责。
"""

from typing import Protocol

from app.schemas.evaluation import (
    GoldQuestion,
)
from app.schemas.retrieval import (
    EmbeddingVector,
    HybridRetrievedChunk,
    RetrievedChunk,
)
from app.schemas.retrieval_strategy import (
    CandidateStrategyParameters,
    CandidateStrategyReport,
)
from app.services.query_rewriting import (
    RewrittenRetrievalQuery,
    rewrite_retrieval_query,
)
from app.services.retrieval_strategy_evaluation import (
    build_candidate_strategy_report,
    evaluate_candidate_question,
)


class BatchEmbeddingProvider(
    Protocol
):
    """批量策略评测需要的最小Embedding能力。"""

    async def embed_texts(
        self,
        texts: list[str],
    ) -> list[EmbeddingVector]:
        """按照输入顺序返回每个查询的向量。"""

        ...


class VectorCandidateRetriever(
    Protocol
):
    """纯向量策略需要的最小检索能力。"""

    def retrieve(
        self,
        query_embedding: EmbeddingVector,
        *,
        query_embedding_model: str,
        top_k: int = 3,
    ) -> list[RetrievedChunk]:
        """返回按余弦相似度排序的向量候选。"""

        ...


class HybridCandidateRetriever(
    Protocol
):
    """两种混合策略需要的最小检索能力。"""

    def retrieve(
        self,
        *,
        query: str,
        query_embedding: EmbeddingVector,
        top_k: int = 3,
    ) -> list[HybridRetrievedChunk]:
        """返回关键词与向量经过RRF融合的候选。"""

        ...


def _validate_questions(
    questions: object,
) -> list[GoldQuestion]:
    """在调用Embedding前校验Gold问题列表。"""

    if not isinstance(questions, list):
        raise TypeError(
            "questions必须是列表"
        )

    if not questions:
        raise ValueError(
            "questions不能为空"
        )

    seen_question_ids: set[str] = set()

    for position, question in enumerate(
        questions
    ):
        if not isinstance(
            question,
            GoldQuestion,
        ):
            raise TypeError(
                "questions中的第 "
                f"{position} 项必须是GoldQuestion"
            )

        # 重复题号会导致同一问题被重复计入指标。
        #
        # 必须在调用Embedding前拒绝，
        # 避免为无效数据产生外部API费用。
        if (
            question.question_id
            in seen_question_ids
        ):
            raise ValueError(
                "questions不能包含重复的"
                f"question_id："
                f"{question.question_id}"
            )

        seen_question_ids.add(
            question.question_id
        )

    return questions


def _validate_embedding_model(
    embedding_model: object,
) -> str:
    """清理并校验Embedding模型名称。"""

    if not isinstance(
        embedding_model,
        str,
    ):
        raise TypeError(
            "embedding_model必须是字符串"
        )

    cleaned_model = embedding_model.strip()

    if not cleaned_model:
        raise ValueError(
            "embedding_model不能为空"
        )

    return cleaned_model


def _prepare_retrieval_queries(
    *,
    questions: list[GoldQuestion],
    parameters: CandidateStrategyParameters,
) -> tuple[
    list[str],
    tuple[RewrittenRetrievalQuery, ...],
]:
    """根据策略准备真正用于检索的问题文本。"""

    # 只有下面两种策略需要执行查询改写：
    #
    # 1. 混合RRF + 查询改写；
    # 2. 混合RRF + 查询改写 + 头部保留重排。
    #
    # 头部保留重排发生在检索之后，
    # 不会改变两种策略都必须先改写查询这一事实。
    rewrite_enabled_strategies = (
        "hybrid_rrf_rewrite",
        "hybrid_rrf_rewrite_rerank",
    )

    if (
        parameters.strategy
        not in rewrite_enabled_strategies
    ):
        # 纯向量基线和普通混合RRF
        # 都必须使用未经改写的原问题。
        #
        # 返回空改写元组，明确表示本次
        # 没有产生RewrittenRetrievalQuery。
        return (
            [
                question.question
                for question in questions
            ],
            (),
        )

    query_rewrites = tuple(
        rewrite_retrieval_query(
            question.question
        )
        for question in questions
    )

    # 参数契约已经要求改写策略必须提供版本，
    # 这里再确认实际执行版本和报告声明版本一致。
    #
    # 否则可能出现：
    #
    # 报告声明一个旧版本，
    # 实际代码却已经运行另一套新规则。
    for query_rewrite in query_rewrites:
        if (
            query_rewrite.rewrite_version
            != parameters.rewrite_version
        ):
            raise ValueError(
                "实际查询改写版本与"
                "策略参数声明不一致"
            )

    return (
        [
            query_rewrite.rewritten_query
            for query_rewrite in query_rewrites
        ],
        query_rewrites,
    )


async def run_candidate_strategy(
    *,
    questions: list[GoldQuestion],
    embedding_provider: (
        BatchEmbeddingProvider
    ),
    vector_retriever: (
        VectorCandidateRetriever
    ),
    hybrid_retriever: (
        HybridCandidateRetriever
    ),
    embedding_model: str,
    collection_name: str,
    parameters: CandidateStrategyParameters,
) -> CandidateStrategyReport:
    """运行一种候选检索策略并返回完整报告。"""

    validated_questions = (
        _validate_questions(
            questions
        )
    )

    if not isinstance(
        parameters,
        CandidateStrategyParameters,
    ):
        raise TypeError(
            "parameters必须是"
            "CandidateStrategyParameters"
        )

    cleaned_embedding_model = (
        _validate_embedding_model(
            embedding_model
        )
    )

    if not isinstance(
        collection_name,
        str,
    ):
        raise TypeError(
            "collection_name必须是字符串"
        )

    cleaned_collection_name = (
        collection_name.strip()
    )

    if not cleaned_collection_name:
        raise ValueError(
            "collection_name不能为空"
        )

    (
        retrieval_queries,
        _query_rewrites,
    ) = _prepare_retrieval_queries(
        questions=validated_questions,
        parameters=parameters,
    )

    # 一次性批量生成全部查询向量。
    #
    # 28道Gold题通常只需一次Embedding请求，
    # 而不是循环发送28次网络请求。
    query_embeddings = (
        await embedding_provider.embed_texts(
            retrieval_queries
        )
    )

    # Protocol只描述预期接口，
    # Python运行时仍要检查真实返回值。
    if not isinstance(
        query_embeddings,
        list,
    ):
        raise TypeError(
            "embedding_provider必须返回列表"
        )

    if (
        len(query_embeddings)
        != len(validated_questions)
    ):
        raise ValueError(
            "查询向量数量与Gold问题数量不一致"
        )

    evaluations = []

    # strict=True要求问题、检索文本和向量数量一致。
    #
    # 即使前面的数量检查以后被修改，
    # 这里也不会静默漏掉任何一道题。
    for (
        question,
        retrieval_query,
        query_embedding,
    ) in zip(
        validated_questions,
        retrieval_queries,
        query_embeddings,
        strict=True,
    ):
        if (
            parameters.strategy
            == "vector_baseline"
        ):
            retrieved_chunks = (
                vector_retriever.retrieve(
                    query_embedding,
                    query_embedding_model=(
                        cleaned_embedding_model
                    ),
                    top_k=parameters.top_k,
                )
            )
        else:
            # 三种混合策略都通过注入的
            # HybridRetriever执行候选检索。
            #
            # hybrid_rrf
            # → 使用原始问题，不启用重排；
            #
            # hybrid_rrf_rewrite
            # → 使用确定性改写问题，不启用重排；
            #
            # hybrid_rrf_rewrite_rerank
            # → 使用确定性改写问题，并由已经配置好的
            #   HybridRetriever执行头部保留重排。
            #
            # 本执行器不直接创建或修改重排策略，
            # 具体HybridRetriever由外层装配代码提供。
            retrieved_chunks = (
                hybrid_retriever.retrieve(
                    query=retrieval_query,
                    query_embedding=query_embedding,
                    top_k=parameters.top_k,
                )
            )

        # 统一使用同一个判卷器。
        #
        # 它会校验候选类型、重复Chunk、排名连续性，
        # 并计算Top-1和Top-3证据命中。
        evaluation = (
            evaluate_candidate_question(
                question=question,
                retrieved_chunks=retrieved_chunks,
            )
        )

        evaluations.append(
            evaluation
        )

    return build_candidate_strategy_report(
        embedding_model=(
            cleaned_embedding_model
        ),
        collection_name=(
            cleaned_collection_name
        ),
        parameters=parameters,
        evaluations=evaluations,
    )
