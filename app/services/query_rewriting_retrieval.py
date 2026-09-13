"""查询改写增强的混合检索编排服务。

本模块负责：
1. 对原始问题执行确定性查询改写；
2. 为改写后的问题生成查询向量；
3. 将同一个改写查询交给关键词检索；
4. 将可选检索范围传给已有HybridRetriever；
5. 调用已有HybridRetriever完成RRF融合；
6. 返回改写审计信息和最终候选快照。

本模块不实现具体的Embedding算法，
不直接访问Chroma，也不重新实现关键词检索、
文档范围过滤或RRF。
这些能力都由注入的依赖对象提供。
"""

from dataclasses import dataclass
from typing import Protocol

from app.schemas.retrieval import (
    EmbeddingVector,
    HybridRetrievedChunk,
)
from app.services.query_rewriting import (
    RewrittenRetrievalQuery,
    rewrite_retrieval_query,
)
# RetrievalScopeFilter是一份已经完成清理和校验的
# 不可变检索范围。
#
# 当前编排器不执行具体过滤，
# 只负责把范围传给下游HybridRetriever。
from app.services.retrieval_scope import (
    RetrievalScopeFilter,
)


class QueryEmbeddingProvider(
    Protocol
):
    """查询改写策略所需的最小Embedding能力。"""

    async def embed_query(
        self,
        query: str,
    ) -> EmbeddingVector:
        """将一条查询转换成向量。"""

        ...


class HybridCandidateRetriever(
    Protocol
):
    """查询改写策略所需的最小混合检索能力。"""

    def retrieve(
        self,
        *,
        query: str,
        query_embedding: EmbeddingVector,
        top_k: int = 3,

        # None表示全库检索；
        # 提供过滤器时，下游两条候选路径
        # 必须共同使用该范围。
        retrieval_scope: (
            RetrievalScopeFilter | None
        ) = None,
    ) -> list[HybridRetrievedChunk]:
        """根据查询、向量和可选范围返回RRF候选。"""

        ...


@dataclass(
    frozen=True,
    slots=True,
)
class QueryRewritingRetrievalResult:
    """一次查询改写混合检索的完整内部结果。"""

    # 保存原始问题、归一化问题、最终改写问题、
    # 新增词、命中规则和改写版本。
    query_rewrite: RewrittenRetrievalQuery

    # 使用tuple保存最终候选快照，
    # 防止调用方增删列表元素或改变候选顺序。
    retrieved_chunks: tuple[
        HybridRetrievedChunk,
        ...,
    ]


def _validate_top_k(
    top_k: object,
) -> int:
    """在调用Embedding前校验最终候选数量。"""

    # bool是int的子类：
    #
    # isinstance(True, int) == True
    #
    # 但True不能作为有意义的Top-K，
    # 所以必须单独排除。
    if (
        isinstance(top_k, bool)
        or not isinstance(top_k, int)
    ):
        raise TypeError(
            "top_k必须是整数"
        )

    if top_k < 1:
        raise ValueError(
            "top_k必须大于或等于1"
        )

    return top_k


class QueryRewritingHybridRetriever:
    """编排查询改写、Embedding和已有混合检索。"""

    def __init__(
        self,
        *,
        embedding_provider: (
            QueryEmbeddingProvider
        ),
        hybrid_retriever: (
            HybridCandidateRetriever
        ),
    ) -> None:
        """保存由外部注入的两个依赖对象。"""

        # 真实运行时：
        #
        # embedding_provider通常是EmbeddingService；
        # hybrid_retriever通常是HybridRetriever。
        #
        # 离线测试时可以替换成Fake对象，
        # 因此当前类不依赖具体实现类型。
        self._embedding_provider = (
            embedding_provider
        )
        self._hybrid_retriever = (
            hybrid_retriever
        )

    async def retrieve(
        self,
        *,
        query: str,
        top_k: int = 3,

        # None保持原有全库检索行为。
        #
        # 该对象只沿调用链传递；
        # 当前模块不会自行解释或改变过滤语义。
        retrieval_scope: (
            RetrievalScopeFilter | None
        ) = None,
    ) -> QueryRewritingRetrievalResult:
        """执行带可选文档范围的查询改写混合检索。"""

        # 在可能调用外部Embedding服务之前校验参数，
        # 避免top_k非法时产生无意义的网络请求。
        validated_top_k = _validate_top_k(
            top_k
        )

        # 类型注解不会在Python运行时自动拒绝dict。
        #
        # 必须在调用外部Embedding服务之前完成校验，
        # 避免为一个最终必然失败的请求产生网络调用和费用。
        if (
            retrieval_scope is not None
            and not isinstance(
                retrieval_scope,
                RetrievalScopeFilter,
            )
        ):
            raise TypeError(
                "retrieval_scope必须是"
                "RetrievalScopeFilter或None"
            )

        # rewrite_retrieval_query()还会负责：
        #
        # 1. 检查query是否为字符串；
        # 2. 拒绝空白问题；
        # 3. 统一错误码、型号和数值；
        # 4. 根据规则补充双语检索词。
        query_rewrite = (
            rewrite_retrieval_query(
                query
            )
        )

        # 向量路径必须使用改写后的问题。
        #
        # await表示暂停当前协程，
        # 等待异步Embedding结果返回；
        # 等待期间事件循环可以处理其他任务。
        query_embedding = (
            await self._embedding_provider.embed_query(
                query_rewrite.rewritten_query
            )
        )

        # 关键词路径也必须使用同一个改写查询。
        #
        # HybridRetriever内部会：
        #
        # 1. 使用query_embedding执行向量召回；
        # 2. 使用query执行关键词召回；
        # 3. 使用RRF融合两份排名。
        retrieved_chunks = (
            self._hybrid_retriever.retrieve(
                # 向量与关键词路径使用同一份改写查询。
                query=(
                    query_rewrite.rewritten_query
                ),

                # 这是改写查询对应的Embedding，
                # 必须与传给关键词路径的文本保持一致。
                query_embedding=query_embedding,

                # 控制RRF融合后最终返回的候选数量。
                top_k=validated_top_k,

                # HybridRetriever会继续把同一个范围
                # 传给向量检索和关键词检索。
                retrieval_scope=retrieval_scope,
            )
        )

        # Protocol只用于静态检查。
        #
        # Python运行时仍可能注入一个行为错误的对象，
        # 所以公共编排边界需要检查真实返回值。
        if not isinstance(
            retrieved_chunks,
            list,
        ):
            raise TypeError(
                "hybrid_retriever必须返回列表"
            )

        for position, retrieved in enumerate(
            retrieved_chunks
        ):
            if not isinstance(
                retrieved,
                HybridRetrievedChunk,
            ):
                raise TypeError(
                    "混合检索结果中的第 "
                    f"{position} 项必须是"
                    "HybridRetrievedChunk"
                )

        # model_copy(deep=True)是Pydantic提供的方法。
        #
        # deep=True不仅复制HybridRetrievedChunk，
        # 也复制其内部嵌套的DocumentChunk。
        #
        # 这样调用方之后修改原检索结果，
        # 不会改变已经返回的策略评测快照。
        result_snapshots = tuple(
            retrieved.model_copy(
                deep=True
            )
            for retrieved in retrieved_chunks
        )

        return QueryRewritingRetrievalResult(
            query_rewrite=query_rewrite,
            retrieved_chunks=result_snapshots,
        )