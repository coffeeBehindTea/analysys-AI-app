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
from app.services.knowledge_query import (
    KnowledgeQueryService,
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
    # 查询问题需要独立的Embedding客户端。
    embedding_client: Annotated[
        AsyncOpenAI,
        Depends(get_embedding_client),
    ],

    # 高相似度证据通过后，
    # 使用LLM客户端生成回答。
    llm_client: Annotated[
        AsyncOpenAI,
        Depends(get_llm_client),
    ],

    # 查询使用与文档摄取相同的Collection和模型。
    vector_store: Annotated[
        ChromaVectorStore,
        Depends(get_vector_store),
    ],

    # Settings提供模型名称和阈值。
    settings: Annotated[
        Settings,
        Depends(get_settings),
    ],
) -> KnowledgeQueryService:
    """装配完整的知识库RAG问答服务。"""

    # get_embedding_model()会检查：
    # EMBEDDING_MODEL存在且不是空字符串。
    embedding_model = get_embedding_model(
        settings
    )

    # 查询Embedding和文档Embedding必须使用同一模型，
    # 否则二者不在同一个向量空间中。
    embedding_provider = EmbeddingService(
        client=embedding_client,
        model=embedding_model,
    )

    # Answer Provider只负责把问题和真实证据
    # 交给LLM并返回回答正文。
    answer_provider = (
        OpenAIKnowledgeAnswerProvider(
            client=llm_client,

            # get_llm_model()检查LLM_MODEL配置。
            model=get_llm_model(settings),
        )
    )

    return KnowledgeQueryService(
        embedding_provider=embedding_provider,
        retrieval_store=vector_store,
        answer_provider=answer_provider,
        embedding_model=embedding_model,
        similarity_threshold=(
            settings.rag_similarity_threshold
        ),
    )