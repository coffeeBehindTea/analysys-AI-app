"""编排知识库检索、阈值拒答、LLM回答和代码引用。"""

# asyncio.to_thread() 可以把同步阻塞函数
# 放到工作线程执行，避免阻塞异步事件循环。
import asyncio

# perf_counter() 是高精度单调计时器，
# 适合测量一段程序经过的时间。
from time import perf_counter

from typing import Protocol

from app.errors import InvalidLLMResponseError
from app.schemas.knowledge_query import (
    KnowledgeAnswerDraft,
    KnowledgeCitation,
    KnowledgeQueryRequest,
    KnowledgeQueryResponse,
)
from app.schemas.retrieval import (
    EmbeddingVector,
    HybridRetrievedChunk,
    RetrievedChunk,
)
# Sequence用于声明Provider只读取有序证据，
# 不要求调用方必须使用list。
from collections.abc import Sequence


# 低相似度或空知识库时使用固定回答。
#
# 这段文字由代码决定，不交给LLM生成，
# 因为拒答分支根本不会调用LLM。
INSUFFICIENT_EVIDENCE_ANSWER = (
    "知识库没有足够证据回答该问题。"
)



class QueryEmbeddingProvider(Protocol):
    """查询向量服务需要提供的最小能力。"""

    async def embed_query(
        self,
        query: str,
    ) -> EmbeddingVector:
        """将一个自然语言问题转换成向量。"""

        ...


class RetrievalStore(Protocol):
    """RAG查询所需的最小向量检索能力。"""

    def retrieve(
        self,
        query_embedding: EmbeddingVector,
        *,
        query_embedding_model: str,
        top_k: int = 3,
    ) -> list[RetrievedChunk]:
        """返回与问题最相近的Top-K Chunk。"""

        ...


class KnowledgeAnswerProvider(Protocol):
    """根据证据生成回答和所用证据编号。"""

    async def generate_answer(
        self,
        *,
        question: str,

        # 协议允许纯向量和混合检索编排器
        # 复用同一个回答Provider。
        evidence: Sequence[
            RetrievedChunk
            | HybridRetrievedChunk
        ],
    ) -> KnowledgeAnswerDraft:
        """返回回答正文和所用临时证据编号。"""

        ...


def build_citations(
    retrieved_chunks: list[RetrievedChunk],
) -> list[KnowledgeCitation]:
    """根据真实召回Chunk构造结构化引用。"""

    # 根据rank排序，使返回引用顺序稳定。
    ordered_chunks = sorted(
        retrieved_chunks,
        key=lambda item: item.rank,
    )

    citations: list[KnowledgeCitation] = []

    for retrieved_chunk in ordered_chunks:
        chunk = retrieved_chunk.chunk

        # 这里没有使用LLM返回值。
        #
        # 引用的文件名、位置、相似度和正文
        # 全部直接复制自VectorStore的真实检索结果。
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
                similarity=(
                    retrieved_chunk.similarity
                ),
                excerpt=chunk.content,
            )
        )

    return citations


class KnowledgeQueryService:
    """知识库证据约束问答的业务编排服务。"""

    def __init__(
        self,
        *,
        embedding_provider: QueryEmbeddingProvider,
        retrieval_store: RetrievalStore,
        answer_provider: KnowledgeAnswerProvider,
        embedding_model: str,
        similarity_threshold: float,
    ) -> None:
        """保存依赖并校验RAG策略配置。"""

        cleaned_embedding_model = (
            embedding_model.strip()
        )

        if not cleaned_embedding_model:
            raise ValueError(
                "Embedding模型名称不能为空"
            )

        # 当前API策略只允许0到1之间的阈值。
        #
        # 虽然理论余弦相似度可以到-1，
        # 但使用负阈值会接受几乎完全不相关的证据。
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError(
                "similarity_threshold必须在0到1之间"
            )

        self._embedding_provider = (
            embedding_provider
        )
        self._retrieval_store = retrieval_store
        self._answer_provider = answer_provider
        self._embedding_model = (
            cleaned_embedding_model
        )
        self._similarity_threshold = (
            similarity_threshold
        )

    async def answer_query(
        self,
        request: KnowledgeQueryRequest,
    ) -> KnowledgeQueryResponse:
        """检索证据，必要时拒答，否则调用LLM回答。"""

        # 开始统计检索耗时。
        #
        # perf_counter()返回的单位是秒，
        # 但它的起始值没有业务意义，只能用于做差。
        retrieval_started_at = perf_counter()

        # EmbeddingService.embed_query()内部会调用
        # OpenAI兼容的POST /embeddings。
        query_embedding = (
            await self._embedding_provider.embed_query(
                request.question
            )
        )

        # ChromaVectorStore.retrieve()是同步方法，
        # 其中的本地数据库访问可能短暂阻塞。
        #
        # asyncio.to_thread()让它在工作线程中运行，
        # 当前事件循环仍能处理其他HTTP请求。
        retrieved_chunks = await asyncio.to_thread(
            self._retrieval_store.retrieve,
            query_embedding,
            query_embedding_model=(
                self._embedding_model
            ),
            top_k=request.top_k,
        )

        # VectorStore契约规定结果按排名返回，
        # 这里再次排序，使Service也保护自己的边界。
        ordered_chunks = sorted(
            retrieved_chunks,
            key=lambda item: item.rank,
        )

        # 这里结束计时，所以retrieval_ms只包含：
        #
        # 查询Embedding + Chroma检索 + 结果排序。
        #
        # 不包含后面的LLM生成时间。
        retrieval_ms = (
            perf_counter() - retrieval_started_at
        ) * 1_000

        # 空知识库或没有检索结果时直接拒答。
        if not ordered_chunks:
            return KnowledgeQueryResponse(
                answer=INSUFFICIENT_EVIDENCE_ANSWER,
                citations=[],
                retrieval_ms=retrieval_ms,
                abstained=True,
            )

        # rank=1是最高相似度结果。
        top_similarity = (
            ordered_chunks[0].similarity
        )

        # “低于阈值”使用严格小于号。
        #
        # 相似度刚好等于阈值时可以继续回答。
        if (
            top_similarity
            < self._similarity_threshold
        ):
            return KnowledgeQueryResponse(
                answer=INSUFFICIENT_EVIDENCE_ANSWER,
                citations=[],
                retrieval_ms=retrieval_ms,
                abstained=True,
            )

        # 只有通过证据阈值后才调用LLM。
        answer_draft = (
            await self._answer_provider.generate_answer(
                question=request.question,
                evidence=ordered_chunks,
            )
        )

        # Protocol是静态能力契约，
        # 运行时仍要防御错误实现。
        if not isinstance(
            answer_draft,
            KnowledgeAnswerDraft,
        ):
            raise InvalidLLMResponseError(
                "知识库回答服务没有返回"
                "KnowledgeAnswerDraft"
            )

        # 本次查询允许使用的证据编号白名单。
        #
        # 例如：
        # E1 → rank=1的RetrievedChunk
        # E2 → rank=2的RetrievedChunk
        evidence_by_id = {
            f"E{item.rank}": item
            for item in ordered_chunks
        }

        # 模型只能选择白名单中真实存在的编号。
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

        # 只选择模型声明实际使用的真实Chunk。
        selected_chunks = [
            evidence_by_id[evidence_id]
            for evidence_id
            in answer_draft.used_evidence_ids
        ]

        # 文件名、页码、Chunk ID、相似度和正文
        # 仍然全部由Python从真实Chunk复制。
        citations = build_citations(
            selected_chunks
        )

        return KnowledgeQueryResponse(
            answer=answer_draft.answer,
            citations=citations,
            retrieval_ms=retrieval_ms,
            abstained=False,
        )