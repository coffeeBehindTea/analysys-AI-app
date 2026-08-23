"""结构化诊断真实依赖装配的离线测试。

本文件不启动FastAPI、不访问真实Chroma，
也不调用Embedding或LLM网络接口。

测试目标：

1. get_diagnosis_service()返回DiagnosisService；
2. 真实组件按照既定检索与生成调用链连接；
3. 关键词索引使用当前VectorStore提供的Chunk；
4. Embedding客户端和LLM客户端进入各自正确的Provider；
5. 混合检索、门控和诊断使用已经评测过的默认参数；
6. FastAPI Service依赖复用可供评测脚本调用的纯检索工厂。
"""

from unittest.mock import MagicMock

import pytest

from app.config import Settings
from app.dependencies import (
    build_diagnosis_retrieval_provider,
    get_diagnosis_service,
)
from app.schemas.retrieval import (
    DocumentChunk,
)
from app.services.diagnosis_generation import (
    OpenAIDiagnosisDraftProvider,
)
from app.services.diagnosis_service import (
    DEFAULT_DIAGNOSIS_TOP_K,
    DiagnosisService,
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
from app.services.query_rewriting_retrieval import (
    QueryRewritingHybridRetriever,
)


# 使用固定哈希构造一条完全脱敏的测试Chunk。
# 这些值只用于满足DocumentChunk的数据契约。
TEST_DOCUMENT_ID = "a" * 64
TEST_CONTENT_HASH = "b" * 64

# 测试显式指定两个模型名称，
# 用于检查装配函数没有把LLM模型和Embedding模型接反。
TEST_EMBEDDING_MODEL = "test-embedding-model"
TEST_LLM_MODEL = "test-chat-model"


def make_test_chunk() -> DocumentChunk:
    """创建可被关键词检索命中的脱敏模拟Chunk。"""

    return DocumentChunk(
        chunk_id=(
            TEST_DOCUMENT_ID
            + ":000000"
        ),
        document_id=TEST_DOCUMENT_ID,
        source_file="robot-fault-demo.txt",
        page_or_section=(
            "section: ERR-DEMO-1001"
        ),
        chunk_index=0,
        content_hash=TEST_CONTENT_HASH,
        content=(
            "模拟故障代码ERR-DEMO-1001，"
            "定位质量下降后进入受控停止状态。"
        ),
    )


class FakeChunkStore:
    """只实现依赖装配阶段需要的list_chunks()能力。"""

    def __init__(
        self,
        chunks: list[DocumentChunk],
    ) -> None:
        """保存Chunk快照并初始化调用计数。"""

        self._chunks = list(chunks)
        self.list_chunks_calls = 0

    def list_chunks(
        self,
    ) -> list[DocumentChunk]:
        """返回列表副本并记录调用次数。"""

        self.list_chunks_calls += 1

        # 返回新列表，避免装配代码修改Fake内部容器。
        return list(self._chunks)


def test_get_diagnosis_service_wires_complete_chain(
) -> None:
    """依赖工厂应连接真实组件且不在构造阶段联网。"""

    chunk = make_test_chunk()
    vector_store = FakeChunkStore(
        [chunk]
    )

    # object()只是两个不同的身份标记。
    # 如果构造阶段尝试调用其SDK方法，测试会立即失败。
    embedding_client = object()
    llm_client = object()

    settings = Settings(
        _env_file=None,
        embedding_model=(
            TEST_EMBEDDING_MODEL
        ),
        llm_model=TEST_LLM_MODEL,
    )

    service = get_diagnosis_service(
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

    assert isinstance(
        service,
        DiagnosisService,
    )
    assert service._retrieval_top_k == (
        DEFAULT_DIAGNOSIS_TOP_K
    )

    # DiagnosisService的检索依赖应首先是门控包装器。
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

    # 门控包装器下方应是查询改写编排器。
    rewriting_retriever = (
        gated_retriever
        ._retrieval_provider
    )
    assert isinstance(
        rewriting_retriever,
        QueryRewritingHybridRetriever,
    )

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
        TEST_EMBEDDING_MODEL
    )

    # 查询改写之后使用已实现的两路混合检索器。
    hybrid_retriever = (
        rewriting_retriever
        ._hybrid_retriever
    )
    assert isinstance(
        hybrid_retriever,
        HybridRetriever,
    )
    assert hybrid_retriever._vector_retriever is (
        vector_store
    )
    assert isinstance(
        hybrid_retriever._keyword_retriever,
        KeywordRetriever,
    )
    assert hybrid_retriever._embedding_model == (
        TEST_EMBEDDING_MODEL
    )
    assert hybrid_retriever._candidate_k == (
        DEFAULT_HYBRID_CANDIDATE_K
    )
    assert hybrid_retriever._rank_constant == (
        DEFAULT_RRF_RANK_CONSTANT
    )

    # 诊断API复用同一个通用检索工厂，
    # 所以也必须装配同一套头部保留策略。
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

    # 装配函数应读取一次当前Collection中的Chunk，
    # 并用它们建立内存关键词索引。
    assert vector_store.list_chunks_calls == 1

    keyword_results = (
        hybrid_retriever
        ._keyword_retriever
        .retrieve(
            "ERR-DEMO-1001",
            top_k=3,
        )
    )
    assert len(keyword_results) == 1
    assert keyword_results[0].chunk.chunk_id == (
        chunk.chunk_id
    )

    # 生成依赖必须使用LLM客户端和LLM模型，
    # 不能误用Embedding客户端或Embedding模型。
    draft_provider = (
        service._draft_provider
    )
    assert isinstance(
        draft_provider,
        OpenAIDiagnosisDraftProvider,
    )
    assert draft_provider._client is llm_client
    assert draft_provider._model == TEST_LLM_MODEL


def test_get_diagnosis_service_reuses_retrieval_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HTTP依赖装配必须调用脚本也能复用的纯检索工厂。"""

    embedding_client = object()
    llm_client = object()
    vector_store = FakeChunkStore(
        [make_test_chunk()]
    )
    settings = Settings(
        _env_file=None,
        embedding_model=TEST_EMBEDDING_MODEL,
        llm_model=TEST_LLM_MODEL,
    )

    # sentinel_retriever只是身份标记。
    # 当前测试不调用retrieve()，只检查装配关系。
    sentinel_retriever = object()

    # MagicMock记录调用参数并返回固定对象。
    retrieval_factory = MagicMock(
        return_value=sentinel_retriever
    )

    # get_diagnosis_service()运行时会从
    # app.dependencies模块全局名称中查找工厂，
    # 所以必须替换该模块中的属性。
    monkeypatch.setattr(
        "app.dependencies."
        "build_diagnosis_retrieval_provider",
        retrieval_factory,
    )

    service = get_diagnosis_service(
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

    retrieval_factory.assert_called_once_with(
        embedding_client=embedding_client,
        vector_store=vector_store,
        settings=settings,
    )
    assert service._retrieval_provider is (
        sentinel_retriever
    )
