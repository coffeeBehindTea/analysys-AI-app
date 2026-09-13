"""编排混合检索门控、知识回答和真实引用构造。

本模块负责：

1. 接收已经通过Pydantic校验的知识问答请求；
2. 调用查询改写、混合检索和证据门控链；
3. 门控拒绝时跳过LLM并返回稳定拒答；
4. 门控放行时调用知识回答Provider；
5. 校验LLM返回的临时证据编号；
6. 将E1、E2等编号映射回真实融合候选；
7. 由Python构造带向量分数和RRF分数的引用。

本模块不实现查询改写、Embedding、关键词检索、
向量检索、RRF算法、门控规则或LLM SDK调用。
这些能力都由注入的依赖对象负责。
"""

# logging是Python标准库的日志模块。
#
# 这里只记录拒答发生在哪个阶段、
# 门控版本、门控原因和候选数量；
# 不记录用户问题或Chunk正文。
import logging

# Sequence表示一组有顺序、可读取的元素。
#
# 当前模块只遍历和排序候选，
# 不会修改调用方传入的证据集合。
from collections.abc import Sequence

# perf_counter()是高精度单调计时器。
#
# 它适合测量一段程序经过的时间，
# 不受系统时间被人工修改的影响。
from time import perf_counter

# Protocol用于声明依赖对象需要具备的最小能力。
#
# 它主要服务于静态类型检查，
# 不会在普通Python运行时自动校验对象。
from typing import Protocol

from app.errors import (
    InvalidLLMResponseError,
)
from app.schemas.knowledge_query import (
    KnowledgeAnswerDraft,
    KnowledgeCitation,
    KnowledgeQueryRequest,
    KnowledgeQueryResponse,
)
from app.schemas.retrieval import (
    HybridRetrievedChunk,
)
from app.services.gated_hybrid_retrieval import (
    GatedHybridRetrievalResult,
)
from app.services.knowledge_query import (
    INSUFFICIENT_EVIDENCE_ANSWER,
    KnowledgeAnswerProvider,
)
from app.services.retrieval_scope import (
    RetrievalScopeFilter,
)


# __name__是当前模块的完整导入名称。
# logging使用它标识日志来源，
# 便于从多个Service的输出中定位本模块。
logger = logging.getLogger(__name__)


class GatedHybridRetrievalProvider(
    Protocol
):
    """混合知识问答Service需要的门控检索能力。"""

    async def retrieve(
        self,
        *,
        query: str,
        top_k: int = 3,
        retrieval_scope: (
            RetrievalScopeFilter | None
        ) = None,
    ) -> GatedHybridRetrievalResult:
        """返回查询改写、融合候选和门控决定。"""

        ...


def build_hybrid_citations(
    retrieved_chunks: Sequence[
        HybridRetrievedChunk
    ],
) -> list[KnowledgeCitation]:
    """根据真实融合候选构造结构化引用。"""

    validated_chunks: list[
        HybridRetrievedChunk
    ] = []

    # Python的类型标注不会自动阻止错误对象进入函数。
    #
    # 因此在读取chunk、rank等属性前，
    # 先执行明确的运行时类型检查。
    for position, retrieved_chunk in enumerate(
        retrieved_chunks
    ):
        if not isinstance(
            retrieved_chunk,
            HybridRetrievedChunk,
        ):
            raise TypeError(
                "retrieved_chunks中的第 "
                f"{position} 项必须是"
                "HybridRetrievedChunk"
            )

        validated_chunks.append(
            retrieved_chunk
        )

    # 最终引用根据RRF融合后的rank排序。
    #
    # 即使调用方传入顺序为Top-2、Top-1，
    # API仍会稳定返回Top-1、Top-2。
    ordered_chunks = sorted(
        validated_chunks,
        key=lambda item: item.rank,
    )

    citations: list[
        KnowledgeCitation
    ] = []

    for retrieved_chunk in ordered_chunks:
        chunk = retrieved_chunk.chunk

        # 以下所有来源字段都直接复制自
        # 本次真实检索返回的DocumentChunk。
        #
        # 不允许LLM生成文件名、页码、
        # Chunk ID或检索分数。
        citations.append(
            KnowledgeCitation(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                source_file=chunk.source_file,
                page_or_section=(
                    chunk.page_or_section
                ),
                chunk_index=chunk.chunk_index,
                rank=retrieved_chunk.rank,

                # 双路候选保留真实余弦相似度。
                #
                # 关键词单路候选没有向量测量值，
                # 因而这里允许为None。
                similarity=(
                    retrieved_chunk
                    .vector_similarity
                ),

                # 所有混合候选都有真实RRF分数。
                #
                # 该数值只解释融合排序，
                # 不是概率或回答置信度。
                rrf_score=(
                    retrieved_chunk.rrf_score
                ),

                # excerpt必须是原始Chunk正文，
                # 不能使用LLM改写后的文字。
                excerpt=chunk.content,
            )
        )

    return citations


class HybridKnowledgeQueryService:
    """编排混合检索门控和证据约束知识问答。"""

    def __init__(
        self,
        *,
        retrieval_provider: (
            GatedHybridRetrievalProvider
        ),
        answer_provider: (
            KnowledgeAnswerProvider
        ),
    ) -> None:
        """保存由依赖注入提供的检索和回答组件。"""

        # 正式运行时通常注入：
        #
        # retrieval_provider：
        #     GatedHybridRetriever
        #
        # answer_provider：
        #     OpenAIKnowledgeAnswerProvider
        #
        # 离线测试时则注入Fake对象，
        # 所以当前Service不绑定具体实现类。
        self._retrieval_provider = (
            retrieval_provider
        )
        self._answer_provider = (
            answer_provider
        )

    async def answer_query(
        self,
        request: KnowledgeQueryRequest,
    ) -> KnowledgeQueryResponse:
        """执行门控混合检索并生成证据约束回答。"""

        # 类型标注不会在Python运行时
        # 自动拒绝普通dict。
        #
        # 在调用Embedding或检索前检查请求类型，
        # 可以避免非法请求产生外部调用和费用。
        if not isinstance(
            request,
            KnowledgeQueryRequest,
        ):
            raise TypeError(
                "request必须是"
                "KnowledgeQueryRequest"
            )

        # 从开始执行查询改写和检索时开始计时。
        retrieval_started_at = perf_counter()

        # GatedHybridRetriever内部会依次执行：
        #
        # 1. 确定性查询改写；
        # 2. 查询Embedding；
        # 3. 向量候选召回；
        # 4. 关键词候选召回；
        # 5. RRF融合；
        # 6. 组合证据门控。
        retrieval_result = (
            await self._retrieval_provider.retrieve(
                query=request.question,
                top_k=request.top_k,

                # 当前KnowledgeQueryRequest
                # 暂时没有文档范围字段，
                # 所以明确执行全库检索。
                retrieval_scope=None,
            )
        )

        # retrieval_ms只统计检索和门控阶段。
        #
        # 后面的LLM生成耗时不包含在内，
        # 以便继续和原来的检索评测指标比较。
        retrieval_ms = (
            perf_counter()
            - retrieval_started_at
        ) * 1_000

        # Protocol不会自动验证真实返回值。
        #
        # 如果错误依赖返回dict或None，
        # 必须在调用LLM前立即停止。
        if not isinstance(
            retrieval_result,
            GatedHybridRetrievalResult,
        ):
            raise TypeError(
                "retrieval_provider必须返回"
                "GatedHybridRetrievalResult"
            )

        # 门控拒绝表示当前Top-K证据组合
        # 不足以支撑安全回答。
        #
        # 该分支不得调用LLM：
        #
        # 1. 防止模型在弱证据上猜测；
        # 2. 节省外部LLM调用成本；
        # 3. 保持无答案问题的正确拒答率。
        if not retrieval_result.decision.accepted:
            # 该日志证明拒答发生在LLM调用之前。
            #
            # logger.info()使用%s占位符延迟格式化：
            # 只有该日志级别真正需要输出时，
            # logging才会把后面的参数填入模板。
            logger.info(
                "knowledge_query_abstained "
                "stage=gate "
                "gate_version=%s "
                "gate_reason=%s "
                "candidate_count=%s",
                (
                    retrieval_result
                    .decision
                    .policy_version
                ),
                retrieval_result.decision.reason,
                len(
                    retrieval_result
                    .retrieved_chunks
                ),
            )

            return KnowledgeQueryResponse(
                answer=(
                    INSUFFICIENT_EVIDENCE_ANSWER
                ),
                citations=[],
                retrieval_ms=retrieval_ms,
                abstained=True,
            )

        validated_candidates: list[
            HybridRetrievedChunk
        ] = []

        # GatedHybridRetrievalResult是dataclass，
        # Python不会自动验证它内部tuple中的元素类型。
        #
        # 在证据进入LLM前再次保护编排边界。
        for position, candidate in enumerate(
            retrieval_result.retrieved_chunks
        ):
            if not isinstance(
                candidate,
                HybridRetrievedChunk,
            ):
                raise TypeError(
                    "retrieved_chunks中的第 "
                    f"{position} 项必须是"
                    "HybridRetrievedChunk"
                )

            validated_candidates.append(
                candidate
            )

        # 使用最终融合rank稳定排序。
        ordered_candidates = tuple(
            sorted(
                validated_candidates,
                key=lambda item: item.rank,
            )
        )

        # 门控已经放行，但如果错误依赖返回空候选，
        # 仍然不能调用LLM生成无证据回答。
        if not ordered_candidates:
            raise TypeError(
                "门控放行结果必须包含候选证据"
            )

        # 只有门控放行后才调用LLM。
        #
        # Provider会：
        #
        # 1. 构造System和User消息；
        # 2. 将候选标记为E1、E2等临时编号；
        # 3. 调用OpenAI兼容Chat API；
        # 4. 解析并校验KnowledgeAnswerDraft。
        answer_draft = (
            await self._answer_provider.generate_answer(
                question=request.question,
                evidence=ordered_candidates,
            )
        )

        # 返回类型注解不会执行运行时校验。
        #
        # dict不能冒充经过Pydantic校验的回答草稿。
        if not isinstance(
            answer_draft,
            KnowledgeAnswerDraft,
        ):
            raise InvalidLLMResponseError(
                "知识库回答服务没有返回"
                "KnowledgeAnswerDraft"
            )

        # 这是第二道拒答出口。
        #
        # 第一道人为确定性门控已经放行，
        # 但是LLM逐条阅读候选正文后仍可能发现：
        #
        # 1. 候选只命中了型号和通用关键词；
        # 2. 候选没有覆盖问题要求的必要事实；
        # 3. 当前证据无法形成受支持的答案。
        if answer_draft.abstained:
            # 该日志证明确定性门控已经放行，
            # 拒答来自LLM阅读候选正文后的二次判断。
            #
            # 不输出answer_draft.answer或候选正文，
            # 避免把用户数据复制到服务端日志。
            logger.info(
                "knowledge_query_abstained "
                "stage=llm_secondary "
                "gate_version=%s "
                "gate_reason=%s "
                "candidate_count=%s",
                (
                    retrieval_result
                    .decision
                    .policy_version
                ),
                retrieval_result.decision.reason,
                len(ordered_candidates),
            )

            # 不直接返回模型生成的拒答文字。
            #
            # 使用应用定义的固定公开文案，可以保证：
            #
            # 1. API响应稳定；
            # 2. 不暴露模型内部解释；
            # 3. 评测可以统一识别拒答；
            # 4. 拒答结果不会携带无关引用。
            return KnowledgeQueryResponse(
                answer=(
                    INSUFFICIENT_EVIDENCE_ANSWER
                ),
                citations=[],
                retrieval_ms=retrieval_ms,
                abstained=True,
            )

        # 建立本次查询的临时证据白名单：
        #
        # E1 → 最终融合rank=1的候选
        # E2 → 最终融合rank=2的候选
        #
        # 这个映射完全由Python构造，
        # LLM没有权力创建新的真实Chunk。
        evidence_by_id: dict[
            str,
            HybridRetrievedChunk,
        ] = {}

        for candidate in ordered_candidates:
            evidence_id = (
                f"E{candidate.rank}"
            )

            # 正常GatedHybridRetriever已经保证
            # rank从1开始且没有重复。
            #
            # 这里继续保护Service边界，
            # 防止错误Fake或替代实现静默覆盖证据。
            if evidence_id in evidence_by_id:
                raise TypeError(
                    "混合检索结果不能包含"
                    "重复rank："
                    f"{candidate.rank}"
                )

            evidence_by_id[evidence_id] = (
                candidate
            )

        # 找出LLM声称使用、但本次并未提供的编号。
        unknown_evidence_ids = [
            evidence_id
            for evidence_id
            in answer_draft.used_evidence_ids
            if evidence_id not in evidence_by_id
        ]

        if unknown_evidence_ids:
            raise InvalidLLMResponseError(
                "知识库回答引用了未提供的证据编号："
                + ", ".join(
                    unknown_evidence_ids
                )
            )

        # 只选择LLM声明实际用于回答的证据。
        #
        # KnowledgeAnswerDraft的数据契约保证：
        #
        # 1. 至少选择一条证据；
        # 2. 证据编号格式为E1、E2等；
        # 3. 编号不能重复。
        selected_candidates = [
            evidence_by_id[evidence_id]
            for evidence_id
            in answer_draft.used_evidence_ids
        ]

        # 文件名、页码、Chunk ID、正文和检索分数
        # 都由Python从真实融合候选中复制。
        citations = build_hybrid_citations(
            selected_candidates
        )

        return KnowledgeQueryResponse(
            answer=answer_draft.answer,
            citations=citations,
            retrieval_ms=retrieval_ms,
            abstained=False,
        )
