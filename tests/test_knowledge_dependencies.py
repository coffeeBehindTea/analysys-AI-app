"""知识库 FastAPI 依赖装配的离线测试。"""

from pathlib import Path

import pytest

import app.dependencies as dependencies
from app.config import Settings
from app.services.document_service import (
    DocumentService,
)

from app.services.knowledge_query import (
    KnowledgeQueryService,
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
    """RAG问答Service的构造过程不应联网。"""

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
        KnowledgeQueryService,
    )

    # 构造Service只是在连接对象，
    # 真正的Embedding和LLM调用发生在answer_query()。
    assert embedding_client.network_calls == 0
    assert llm_client.network_calls == 0