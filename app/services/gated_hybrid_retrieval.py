"""编排查询改写混合检索和组合证据门控。

本模块负责：

1. 接收并向下传递可选文档检索范围；
2. 调用查询改写混合检索器；
3. 校验下游返回的数据契约；
4. 对最终Top-K执行组合证据门控；
5. 返回查询改写、候选证据和门控决定的快照。

本模块不创建Embedding客户端、不直接访问Chroma、
不自行过滤文档、不调用LLM，也不生成HTTP响应。
"""

from dataclasses import dataclass
from typing import Protocol

from app.schemas.retrieval import (
    HybridRetrievedChunk,
)
from app.services.hybrid_retrieval_gate import (
    HybridEvidenceGateDecision,
    HybridEvidenceGatePolicy,
    evaluate_hybrid_evidence_gate,
)
# RetrievalScopeFilter定义一次检索允许访问的文档范围。
#
# 当前门控编排器不执行具体过滤，
# 只保证范围能够完整传给下游检索链。
from app.services.retrieval_scope import (
    RetrievalScopeFilter,
)
from app.services.query_rewriting import (
    RewrittenRetrievalQuery,
)
from app.services.query_rewriting_retrieval import (
    QueryRewritingRetrievalResult,
)


class QueryRewritingRetrievalProvider(
    Protocol
):
    """当前编排器需要的最小下游检索能力。"""

    async def retrieve(
        self,
        *,
        query: str,
        top_k: int = 3,

        # None表示全库检索；
        # 提供过滤器时只允许下游搜索指定文档范围。
        retrieval_scope: (
            RetrievalScopeFilter | None
        ) = None,
    ) -> QueryRewritingRetrievalResult:
        """执行范围受限的查询改写混合检索。"""

        ...


@dataclass(
    frozen=True,
    slots=True,
)
class GatedHybridRetrievalResult:
    """一次混合检索和门控的完整业务结果。"""

    # 保存原始问题、归一化问题、
    # 最终改写文本、规则版本和命中规则。
    query_rewrite: RewrittenRetrievalQuery

    # 使用tuple保存候选快照，
    # 防止调用方增删候选或改变候选顺序。
    retrieved_chunks: tuple[
        HybridRetrievedChunk,
        ...,
    ]

    # 保存是否放行、放行原因、策略版本
    # 和真正支持放行的Chunk ID。
    decision: HybridEvidenceGateDecision


class GatedHybridRetriever:
    """把查询改写混合检索与证据门控组合起来。"""

    def __init__(
        self,
        *,
        retrieval_provider: (
            QueryRewritingRetrievalProvider
        ),
        policy: HybridEvidenceGatePolicy,
    ) -> None:
        """保存下游检索器和固定门控策略。"""

        # Protocol主要用于静态类型检查，
        # 不会在普通Python运行时自动验证对象。
        #
        # retrieval_provider最终会在调用retrieve()
        # 时接受Python自身的方法与参数检查。
        self._retrieval_provider = (
            retrieval_provider
        )

        # 门控策略必须是经过校验且带版本号的对象，
        # 不能让普通字典绕过__post_init__校验。
        if not isinstance(
            policy,
            HybridEvidenceGatePolicy,
        ):
            raise TypeError(
                "policy必须是HybridEvidenceGatePolicy"
            )

        # HybridEvidenceGatePolicy使用：
        #
        # frozen=True
        #
        # 因此保存后不能被调用方修改阈值。
        self._policy = policy

    async def retrieve(
        self,
        *,
        query: str,
        top_k: int = 3,

        # None保持原来的全库检索行为。
        #
        # 门控层只传递范围，不自行读取文档ID、
        # 文件名或构造数据库where条件。
        retrieval_scope: (
            RetrievalScopeFilter | None
        ) = None,
    ) -> GatedHybridRetrievalResult:
        """执行范围受限的混合检索和证据门控。"""

        # Python类型注解不会自动拒绝普通dict。
        #
        # 在调用异步下游前快速失败，可以避免：
        #
        # 1. 产生无意义的Embedding请求；
        # 2. 执行一部分检索后才发现范围非法；
        # 3. 不同下游实现对dict产生不同解释。
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
        
        # 下游真实实现通常是：
        #
        # QueryRewritingHybridRetriever
        #
        # 它会依次执行：
        #
        # 1. 确定性查询改写；
        # 2. 查询Embedding；
        # 3. 向量与关键词候选召回；
        # 4. RRF融合。
        #
        # 它也会在产生外部Embedding请求前
        # 校验query和top_k。
        retrieval_result = (
            await self._retrieval_provider.retrieve(
                # 原始问题由下游执行确定性改写。
                query=query,

                # 控制最终参与门控的候选数量。
                top_k=top_k,

                # 下游会继续把同一个不可变范围传给：
                #
                # QueryRewritingHybridRetriever
                # → HybridRetriever
                # → VectorStore和KeywordRetriever
                retrieval_scope=retrieval_scope,
            )
        )

        # Protocol不会自动检查真实返回值。
        #
        # 如果错误实现返回dict、None或普通列表，
        # 应在这里停止，而不是稍后产生难以定位的
        # AttributeError。
        if not isinstance(
            retrieval_result,
            QueryRewritingRetrievalResult,
        ):
            raise TypeError(
                "retrieval_provider必须返回"
                "QueryRewritingRetrievalResult"
            )

        # QueryRewritingRetrievalResult中的tuple
        # 只能防止增删候选，不能阻止其中的
        # Pydantic对象被修改。
        #
        # model_copy(deep=True)会递归复制：
        #
        # HybridRetrievedChunk
        #     └── DocumentChunk
        #
        # 因此当前Service拥有独立的候选快照。
        candidate_snapshots = [
            candidate.model_copy(deep=True)
            for candidate
            in retrieval_result.retrieved_chunks
        ]

        # 门控函数的公共契约要求list，
        # 所以这里先使用列表执行门控。
        #
        # evaluate_hybrid_evidence_gate()会检查：
        #
        # 1. 候选类型；
        # 2. 重复Chunk ID；
        # 3. rank是否从1连续排列；
        # 4. 精确编号支持；
        # 5. 型号加主题词支持；
        # 6. 通用双路支持。
        decision = evaluate_hybrid_evidence_gate(
            retrieved_chunks=candidate_snapshots,
            policy=self._policy,
        )

        return GatedHybridRetrievalResult(
            # RewrittenRetrievalQuery本身是
            # frozen=True的不可变dataclass，
            # 可以安全共享。
            query_rewrite=(
                retrieval_result.query_rewrite
            ),

            # 对外转换成tuple，
            # 表示这是一份已经完成排序的快照。
            retrieved_chunks=tuple(
                candidate_snapshots
            ),
            decision=decision,
        )