"""知识库 FastAPI 依赖装配的离线测试。"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

import app.dependencies as dependencies
from app.config import Settings
from app.services.document_service import (
    DocumentService,
)

from app.services.embedding import (
    EmbeddingService,
)
from app.services.gated_hybrid_retrieval import (
    GatedHybridRetriever,
)
from app.services.head_preserving_reranking import (
    HeadPreservationPolicy,
)
from app.services.hybrid_knowledge_query import (
    HybridKnowledgeQueryService,
)
from app.services.hybrid_retrieval import (
    DEFAULT_HYBRID_CANDIDATE_K,
    DEFAULT_RRF_RANK_CONSTANT,
    HybridRetriever,
)
from app.services.hybrid_retrieval_gate import (
    HYBRID_EVIDENCE_GATE_VERSION,
)
from app.services.keyword_retrieval import (
    KeywordRetriever,
)
from app.services.knowledge_answer import (
    OpenAIKnowledgeAnswerProvider,
)
from app.services.query_rewriting_retrieval import (
    QueryRewritingHybridRetriever,
)

class FakeAsyncOpenAIClient:
    """用于测试客户端资源清理的假异步客户端。"""

    def __init__(self) -> None:
        self.closed = False
        self.network_calls = 0

    async def close(self) -> None:
        """记录客户端连接池已被关闭。"""

        self.closed = True


def make_settings(
    tmp_path: Path,
) -> Settings:
    """创建不读取 .env 的测试配置。"""

    return Settings(
        # _env_file=None 保证测试不读取真实 .env。
        _env_file=None,

        # 本测试只需要合法模型名，不需要真实 API Key。
        llm_model="test-llm-model",
        embedding_model="test-embedding-model",

        # 使用 pytest 临时目录，不接触正式 V4 数据。
        chroma_persist_directory=(
            tmp_path / "chroma"
        ),
        chroma_collection_name=(
            "test_knowledge_dependencies"
        ),
        rag_chunk_size=800,
        rag_chunk_overlap=120,
        rag_similarity_threshold=0.70,
        embedding_batch_size=16,
        max_document_size_bytes=1024 * 1024,
    )


@pytest.mark.asyncio
async def test_embedding_client_is_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Embedding 依赖结束时必须关闭异步客户端。"""

    settings = make_settings(tmp_path)
    fake_client = FakeAsyncOpenAIClient()

    def fake_create_embedding_client(
        received_settings: Settings,
    ) -> FakeAsyncOpenAIClient:
        """替换真实客户端工厂，避免读取密钥和联网。"""

        assert received_settings is settings
        return fake_client

    # 替换 dependencies 模块中实际使用的工厂名称。
    monkeypatch.setattr(
        dependencies,
        "create_embedding_client",
        fake_create_embedding_client,
    )

    # get_embedding_client() 是异步生成器。
    dependency_generator = (
        dependencies.get_embedding_client(
            settings
        )
    )

    # anext() 执行到 yield，并取得依赖提供的客户端。
    provided_client = await anext(
        dependency_generator
    )

    assert provided_client is fake_client
    assert fake_client.closed is False

    # aclose() 结束异步生成器，
    # 从而执行 finally 中的 await client.close()。
    await dependency_generator.aclose()

    assert fake_client.closed is True
    assert fake_client.network_calls == 0


def test_vector_store_uses_temporary_collection(
    tmp_path: Path,
) -> None:
    """VectorStore 依赖应打开测试配置指定的空 Collection。"""

    settings = make_settings(tmp_path)

    store = dependencies.get_vector_store(
        settings
    )

    # 新 Collection 应为空，并且没有接触正式 V4。
    assert store.list_documents() == []


def test_document_service_can_be_assembled_without_network(
    tmp_path: Path,
) -> None:
    """完整摄取 Service 的构造过程不应发送网络请求。"""

    settings = make_settings(tmp_path)
    fake_client = FakeAsyncOpenAIClient()
    store = dependencies.get_vector_store(
        settings
    )

    service = dependencies.get_document_service(
        client=fake_client,  # type: ignore[arg-type]
        vector_store=store,
        settings=settings,
    )

    assert isinstance(service, DocumentService)

    # Embedding 请求只会在 ingest_document() 中发生，
    # 构造 Service 时不应联网。
    assert fake_client.network_calls == 0


def test_knowledge_query_service_can_be_assembled_without_network(
    tmp_path: Path,
) -> None:
    """混合RAG问答依赖图应完整装配且不联网。"""

    settings = make_settings(tmp_path)

    embedding_client = FakeAsyncOpenAIClient()
    llm_client = FakeAsyncOpenAIClient()

    # 使用pytest临时目录中的空Collection，
    # 不读取或修改正式V4。
    store = dependencies.get_vector_store(
        settings
    )

    service = (
        dependencies.get_knowledge_query_service(
            # Fake没有显式继承AsyncOpenAI，
            # 但构造阶段只会保存它，不会调用SDK方法。
            embedding_client=embedding_client,  # type: ignore[arg-type]
            llm_client=llm_client,  # type: ignore[arg-type]
            vector_store=store,
            settings=settings,
        )
    )

    assert isinstance(
        service,
        HybridKnowledgeQueryService,
    )

    # 最外层Service首先依赖组合证据门控检索器。
    gated_retriever = (
        service._retrieval_provider
    )
    assert isinstance(
        gated_retriever,
        GatedHybridRetriever,
    )
    assert gated_retriever._policy.version == (
        HYBRID_EVIDENCE_GATE_VERSION
    )

    # 门控下方是查询改写编排器。
    rewriting_retriever = (
        gated_retriever
        ._retrieval_provider
    )
    assert isinstance(
        rewriting_retriever,
        QueryRewritingHybridRetriever,
    )

    # 查询改写后的文本由EmbeddingService
    # 使用请求级Embedding客户端转成查询向量。
    embedding_provider = (
        rewriting_retriever
        ._embedding_provider
    )
    assert isinstance(
        embedding_provider,
        EmbeddingService,
    )
    assert embedding_provider._client is (
        embedding_client
    )
    assert embedding_provider._model == (
        "test-embedding-model"
    )

    # 同一份改写查询同时进入向量和关键词路径，
    # 再由HybridRetriever执行RRF融合。
    hybrid_retriever = (
        rewriting_retriever
        ._hybrid_retriever
    )
    assert isinstance(
        hybrid_retriever,
        HybridRetriever,
    )
    assert hybrid_retriever._vector_retriever is (
        store
    )
    assert isinstance(
        hybrid_retriever._keyword_retriever,
        KeywordRetriever,
    )
    assert hybrid_retriever._candidate_k == (
        DEFAULT_HYBRID_CANDIDATE_K
    )
    assert hybrid_retriever._rank_constant == (
        DEFAULT_RRF_RANK_CONSTANT
    )

    # 在线知识问答必须使用已通过离线评测的
    # head-preserving-v1头部保留参数。
    rerank_policy = (
        hybrid_retriever._rerank_policy
    )
    assert isinstance(
        rerank_policy,
        HeadPreservationPolicy,
    )
    assert rerank_policy.fused_head_k == 1
    assert rerank_policy.vector_head_k == 1
    assert rerank_policy.keyword_head_k == 2

    # 门控放行后使用生成式LLM回答Provider，
    # 不能误用Embedding客户端。
    answer_provider = (
        service._answer_provider
    )
    assert isinstance(
        answer_provider,
        OpenAIKnowledgeAnswerProvider,
    )
    assert answer_provider._client is llm_client
    assert answer_provider._model == (
        "test-llm-model"
    )

    # 构造Service只创建并连接对象；
    # 真正的Embedding和LLM调用发生在answer_query()。
    assert embedding_client.network_calls == 0
    assert llm_client.network_calls == 0


def test_knowledge_query_service_reuses_generic_hybrid_factory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """知识问答依赖必须复用通用混合检索装配工厂。"""

    settings = make_settings(tmp_path)
    embedding_client = object()
    llm_client = object()
    vector_store = object()

    # sentinel_retriever只作为身份标记，
    # 当前测试不会调用它的retrieve()方法。
    sentinel_retriever = object()

    # MagicMock会记录工厂收到的关键字参数，
    # 并返回固定的身份标记。
    retrieval_factory = MagicMock(
        return_value=sentinel_retriever
    )

    monkeypatch.setattr(
        dependencies,
        "build_gated_hybrid_retrieval_provider",
        retrieval_factory,
    )

    service = (
        dependencies.get_knowledge_query_service(
            embedding_client=(
                embedding_client  # type: ignore[arg-type]
            ),
            llm_client=(
                llm_client  # type: ignore[arg-type]
            ),
            vector_store=(
                vector_store  # type: ignore[arg-type]
            ),
            settings=settings,
        )
    )

    retrieval_factory.assert_called_once_with(
        embedding_client=embedding_client,
        vector_store=vector_store,
        settings=settings,
    )
    assert isinstance(
        service,
        HybridKnowledgeQueryService,
    )
    assert service._retrieval_provider is (
        sentinel_retriever
    )

    # 通用工厂只负责检索链；
    # 回答Provider仍由知识问答依赖函数创建。
    assert isinstance(
        service._answer_provider,
        OpenAIKnowledgeAnswerProvider,
    )
