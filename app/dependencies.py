"""FastAPI 依赖装配函数。

这里决定 Settings、AsyncOpenAI 和 TriageService 如何创建与连接，
但不实现具体业务。
"""

# AsyncIterator[T] 表示可以通过 async for 消费、逐步产生 T 的异步迭代器。
# FastAPI 也用这种返回注解识别含 yield 的异步资源依赖。
from collections.abc import AsyncIterator

# Annotated[真实类型, 元数据] 用于在类型上附加 Depends 等框架信息。
from typing import Annotated

# Depends 声明“此参数由 FastAPI 调用指定函数提供”，而不是由客户端传入。
from fastapi import Depends

# AsyncOpenAI 是 OpenAI Python SDK 的异步客户端类型。
from openai import AsyncOpenAI

from app.config import Settings, get_settings
from app.services.llm_client import create_llm_client, get_llm_model
from app.services.triage import TriageService

from app.services.document_catalog import (
    DocumentCatalogService,
)
from app.services.vector_store import (
    ChromaVectorStore,
)

from app.services.chunking import TextChunker
from app.services.document_preparation import (
    TextDocumentPreparationService,
)
from app.services.document_service import (
    DocumentService,
)
from app.services.embedding import EmbeddingService
from app.services.embedding_client import (
    create_embedding_client,
    get_embedding_model,
)

from app.services.knowledge_answer import (
    OpenAIKnowledgeAnswerProvider,
)
# HybridKnowledgeQueryService负责：
#
# 查询改写 → 混合检索 → RRF → 证据门控
# → LLM回答 → 混合引用构造。
from app.services.hybrid_knowledge_query import (
    HybridKnowledgeQueryService,
)
from app.services.diagnosis_generation import (
    OpenAIDiagnosisDraftProvider,
)
from app.services.diagnosis_service import (
    DiagnosisService,
)
from app.services.gated_hybrid_retrieval import (
    GatedHybridRetriever,
)
from app.services.head_preserving_reranking import (
    HeadPreservationPolicy,
)
from app.services.hybrid_retrieval import (
    HybridRetriever,
)
from app.services.hybrid_retrieval_gate import (
    HybridEvidenceGatePolicy,
)
from app.services.keyword_retrieval import (
    KeywordRetriever,
)
from app.services.query_rewriting_retrieval import (
    QueryRewritingHybridRetriever,
)


async def get_llm_client(
    settings: Annotated[Settings, Depends(get_settings)],
) -> AsyncIterator[AsyncOpenAI]:
    """为一次 HTTP 请求创建并最终关闭异步 LLM 客户端。

    settings 参数由 FastAPI 调用 get_settings() 注入。
    yield 产生的 AsyncOpenAI 会继续注入下游依赖。
    """

    # 工厂函数负责检查配置，并构造真正的 SDK 客户端。
    client = create_llm_client(settings)

    try:
        # yield 暂停当前依赖，把 client 提供给 get_triage_service()。
        yield client
    finally:
        # close() 是 AsyncOpenAI 的异步资源清理方法，因此必须 await。
        # finally 保证正常响应或异常时都尽量关闭其 HTTP 连接资源。
        await client.close()


def get_triage_service(
    client: Annotated[AsyncOpenAI, Depends(get_llm_client)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> TriageService:
    """使用已注入的客户端和 Settings 构造故障分析 Service。"""

    # get_llm_model() 负责检查模型名；TriageService 只接收可用依赖。
    return TriageService(
        client=client,
        model=get_llm_model(settings),
    )


async def get_embedding_client(
    settings: Annotated[
        Settings,
        Depends(get_settings),
    ],
) -> AsyncIterator[AsyncOpenAI]:
    """为一次请求创建并最终关闭 Embedding 客户端。"""

    # create_embedding_client() 使用独立的：
    # EMBEDDING_API_KEY、EMBEDDING_BASE_URL。
    #
    # 创建客户端本身不会发送网络请求。
    client = create_embedding_client(settings)

    try:
        # FastAPI 会把 client 注入下游的
        # get_document_service()。
        yield client
    finally:
        # 无论上传成功、校验失败还是上游异常，
        # 都关闭 AsyncOpenAI 持有的 HTTP 连接池。
        await client.close()


def get_vector_store(
    settings: Annotated[
        Settings,
        Depends(get_settings),
    ],
) -> ChromaVectorStore:
    """打开并校验当前配置指定的 Chroma Collection。"""

    return ChromaVectorStore(
        persist_directory=(
            settings.chroma_persist_directory
        ),
        collection_name=(
            settings.chroma_collection_name
        ),
        embedding_model=get_embedding_model(settings),
    )


def get_document_catalog_service(
    vector_store: Annotated[
        ChromaVectorStore,
        Depends(get_vector_store),
    ],
) -> DocumentCatalogService:
    """使用共享的 VectorStore 构造文档目录服务。"""

    return DocumentCatalogService(
        vector_store=vector_store,
    )


def get_document_service(
    client: Annotated[
        AsyncOpenAI,
        Depends(get_embedding_client),
    ],
    vector_store: Annotated[
        ChromaVectorStore,
        Depends(get_vector_store),
    ],
    settings: Annotated[
        Settings,
        Depends(get_settings),
    ],
) -> DocumentService:
    """装配真实的文档摄取服务。"""

    # 取得经过非空校验的模型名称。
    embedding_model = get_embedding_model(
        settings
    )

    # EmbeddingService 负责把一批文本发送给
    # OpenAI 兼容 Embedding API，并校验返回向量。
    embedding_service = EmbeddingService(
        client=client,
        model=embedding_model,
    )

    # PreparationService 负责：
    # 文件类型路由 → Loader → SourceTextSegment → Chunk。
    preparation_service = (
        TextDocumentPreparationService(
            chunker=TextChunker(
                chunk_size=settings.rag_chunk_size,
                overlap=settings.rag_chunk_overlap,
            )
        )
    )

    return DocumentService(
        preparation_service=preparation_service,
        embedding_provider=embedding_service,
        vector_store=vector_store,
        embedding_model=embedding_model,
        batch_size=settings.embedding_batch_size,
        max_file_size_bytes=(
            settings.max_document_size_bytes
        ),
    )


def get_knowledge_query_service(
    # 查询改写阶段需要调用Embedding接口。
    embedding_client: Annotated[
        AsyncOpenAI,
        Depends(get_embedding_client),
    ],

    # 门控放行后使用LLM生成证据约束回答。
    llm_client: Annotated[
        AsyncOpenAI,
        Depends(get_llm_client),
    ],

    # 同一个VectorStore同时用于：
    #
    # 1. Chroma向量候选召回；
    # 2. 读取Chunk并建立关键词索引。
    vector_store: Annotated[
        ChromaVectorStore,
        Depends(get_vector_store),
    ],

    # Settings提供Embedding和LLM模型名称。
    settings: Annotated[
        Settings,
        Depends(get_settings),
    ],
) -> HybridKnowledgeQueryService:
    """装配在线知识库混合RAG问答服务。"""

    # 复用诊断API也使用的通用混合检索工厂。
    #
    # 该工厂会装配：
    #
    # EmbeddingService
    # → KeywordRetriever
    # → HybridRetriever
    # → QueryRewritingHybridRetriever
    # → GatedHybridRetriever
    gated_retriever = (
        build_gated_hybrid_retrieval_provider(
            embedding_client=embedding_client,
            vector_store=vector_store,
            settings=settings,
        )
    )

    # Answer Provider只负责：
    #
    # 1. 把问题和已经通过门控的证据构造成Prompt；
    # 2. 调用OpenAI兼容Chat API；
    # 3. 解析并校验KnowledgeAnswerDraft。
    answer_provider = (
        OpenAIKnowledgeAnswerProvider(
            client=llm_client,

            # get_llm_model()负责检查模型名
            # 已配置且不是空字符串。
            model=get_llm_model(settings),
        )
    )

    # HybridKnowledgeQueryService负责高层业务编排：
    #
    # 1. 调用门控混合检索；
    # 2. 拒绝弱证据；
    # 3. 放行后调用回答Provider；
    # 4. 校验E1、E2证据白名单；
    # 5. 构造带RRF信息的真实引用。
    return HybridKnowledgeQueryService(
        retrieval_provider=gated_retriever,
        answer_provider=answer_provider,
    )


def build_gated_hybrid_retrieval_provider(
    *,
    embedding_client: AsyncOpenAI,
    vector_store: ChromaVectorStore,
    settings: Settings,
) -> GatedHybridRetriever:
    """装配可供问答、诊断和评测复用的门控混合检索链。

    这是一个普通Python工厂，不依赖FastAPI的Depends。

    因此它可以同时被：

    1. get_knowledge_query_service()；
    2. build_diagnosis_retrieval_provider()；
    3. Prompt对照实验脚本；
    4. 离线装配测试

    直接或间接调用。
    """

    # 查询向量必须使用与Collection相同的
    # Embedding模型，否则查询和文档不在同一向量空间。
    embedding_model = get_embedding_model(
        settings
    )

    # EmbeddingService只保存客户端和模型配置。
    #
    # 此处不会联网；真正的Embedding请求
    # 发生在retrieve()调用链中。
    embedding_provider = EmbeddingService(
        client=embedding_client,
        model=embedding_model,
    )

    # 从当前Collection取得全部Chunk正文和元数据，
    # 为确定性的关键词检索建立内存索引。
    #
    # list_chunks()不会加载全部Embedding向量。
    keyword_retriever = KeywordRetriever(
        chunks=vector_store.list_chunks(),
    )

    # HybridRetriever组合：
    #
    # 1. Chroma向量召回；
    # 2. 内存关键词召回；
    # 3. RRF排名融合；
    # 4. 头部保留重排。
    #
    # 未显式传参时继续使用已经评测的：
    #
    # candidate_k=20
    # rank_constant=60
    # fused_head_k=1
    # vector_head_k=1
    # keyword_head_k=2
    #
    # HeadPreservationPolicy()的默认值
    # 对应已通过候选召回和门控安全评测的
    # head-preserving-v1。
    hybrid_retriever = HybridRetriever(
        vector_retriever=vector_store,
        keyword_retriever=keyword_retriever,
        embedding_model=embedding_model,
        rerank_policy=(
            HeadPreservationPolicy()
        ),
    )

    # 查询改写检索器负责：
    #
    # 原始查询
    #   → 确定性术语改写
    #   → 查询Embedding
    #   → 混合检索
    #   → 头部保留重排。
    rewriting_retriever = (
        QueryRewritingHybridRetriever(
            embedding_provider=(
                embedding_provider
            ),
            hybrid_retriever=(
                hybrid_retriever
            ),
        )
    )

    # 最外层门控器检查融合候选是否满足
    # 进入生成式LLM的证据条件。
    #
    # HybridEvidenceGatePolicy()继续使用
    # hybrid-evidence-gate-v1默认配置。
    return GatedHybridRetriever(
        retrieval_provider=(
            rewriting_retriever
        ),
        policy=HybridEvidenceGatePolicy(),
    )


def build_diagnosis_retrieval_provider(
    *,
    embedding_client: AsyncOpenAI,
    vector_store: ChromaVectorStore,
    settings: Settings,
) -> GatedHybridRetriever:
    """兼容原有诊断代码使用的检索工厂名称。

    真正的组件装配已经移动到通用工厂。
    保留这个包装函数是为了避免诊断API、
    Prompt对照实验和已有导入路径同时失效。
    """

    return (
        build_gated_hybrid_retrieval_provider(
            embedding_client=embedding_client,
            vector_store=vector_store,
            settings=settings,
        )
    )


def get_diagnosis_service(
    # 查询改写阶段需要调用Embedding接口。
    #
    # FastAPI会先执行get_embedding_client()，
    # 再把客户端注入当前参数。
    embedding_client: Annotated[
        AsyncOpenAI,
        Depends(get_embedding_client),
    ],

    # 门控放行后，诊断草稿Provider
    # 使用该客户端调用生成式LLM。
    llm_client: Annotated[
        AsyncOpenAI,
        Depends(get_llm_client),
    ],

    # 同一个VectorStore同时负责：
    #
    # 1. 提供全部Chunk建立关键词索引；
    # 2. 根据查询向量执行向量召回。
    vector_store: Annotated[
        ChromaVectorStore,
        Depends(get_vector_store),
    ],

    # Settings提供Embedding和LLM模型名称。
    settings: Annotated[
        Settings,
        Depends(get_settings),
    ],
) -> DiagnosisService:
    """装配完整的证据约束结构化诊断服务。"""

    # 在线API和离线Prompt实验共享同一个纯检索工厂。
    #
    # 这样查询改写、混合检索参数或门控策略发生变化时，
    # 两条入口会同时获得更新，不会形成两套实现。
    gated_retriever = (
        build_diagnosis_retrieval_provider(
            embedding_client=embedding_client,
            vector_store=vector_store,
            settings=settings,
        )
    )

    # 诊断草稿Provider使用生成式LLM，
    # 而不是Embedding模型。
    draft_provider = (
        OpenAIDiagnosisDraftProvider(
            client=llm_client,
            model=get_llm_model(
                settings
            ),
        )
    )

    # DiagnosisService只依赖两个高层能力：
    #
    # 1. 返回候选和门控决定；
    # 2. 根据已放行证据生成诊断草稿。
    #
    # 内部具体由多少层组件组成，
    # DiagnosisService不需要知道。
    return DiagnosisService(
        retrieval_provider=gated_retriever,
        draft_provider=draft_provider,
    )
