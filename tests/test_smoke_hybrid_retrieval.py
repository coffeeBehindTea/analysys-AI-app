"""真实混合检索冒烟脚本的离线测试。

本文件不访问真实.env、Embedding API或Chroma目录。

测试通过Fake对象替换外部边界，但保留真实的：

1. KeywordRetriever关键词特征提取与排序；
2. HybridRetriever双路召回编排；
3. reciprocal_rank_fusion()融合算法；
4. 冒烟脚本的参数校验、输出和资源关闭逻辑。
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.schemas.retrieval import (
    DocumentChunk,
    RetrievedChunk,
)
from app.services.retrieval_scope import (
    RetrievalScopeFilter,
)
import scripts.smoke_hybrid_retrieval as module


# 测试使用的固定Embedding模型名。
#
# 它只用于检查参数传递，
# 不对应任何真实供应商或线上模型。
TEST_EMBEDDING_MODEL = (
    "embedding-test-model"
)

# Fake Embedding Service返回的固定查询向量。
#
# 测试不关心向量的语义，只关心它是否沿调用链
# 原样传给向量检索器。
TEST_QUERY_VECTOR = [0.1, 0.2, 0.3]

# 该Chunk同时能够被Fake向量路径和真实关键词路径召回。
#
# 正文包含ERR-DEMO-1001，保证真实KeywordRetriever
# 能够提取并命中结构化错误码。
TEST_CHUNK = DocumentChunk(
    chunk_id=(
        "a" * 64
        + ":000000"
    ),
    document_id="a" * 64,
    source_file="robot-fault-demo.txt",
    page_or_section=(
        "section: ERR-DEMO-1001"
    ),
    chunk_index=0,
    content_hash="b" * 64,
    content=(
        "故障代码 ERR-DEMO-1001。"
        "恢复条件是模拟定位质量恢复正常。"
    ),
)


class FakeAsyncClient:
    """记录异步close()调用的Embedding客户端替身。"""

    def __init__(self) -> None:
        """创建尚未关闭的Fake客户端。"""

        self.close_call_count = 0

    async def close(self) -> None:
        """模拟AsyncOpenAI.close()并记录调用次数。"""

        self.close_call_count += 1


class FakeEmbeddingService:
    """返回固定查询向量或预设异常的Embedding替身。"""

    def __init__(
        self,
        *,
        query_vector: list[float],
        error: Exception | None = None,
    ) -> None:
        """保存预设向量、可选异常和查询记录。"""

        self._query_vector = query_vector
        self._error = error
        self.queries: list[str] = []

    async def embed_query(
        self,
        query: str,
    ) -> list[float]:
        """记录问题，然后返回向量或抛出预设异常。"""

        self.queries.append(query)

        if self._error is not None:
            raise self._error

        # 返回新列表，防止调用方修改Fake内部的预设值。
        return list(self._query_vector)


class FakeVectorStore:
    """模拟Chroma的Chunk读取和向量候选召回。"""

    def __init__(
        self,
        *,
        chunks: list[DocumentChunk],
        vector_results: list[RetrievedChunk],
    ) -> None:
        """保存预设Chunk、候选结果和调用记录。"""

        self._chunks = chunks
        self._vector_results = vector_results
        self.list_chunks_call_count = 0
        self.retrieve_calls: list[
            dict[str, object]
        ] = []

    def list_chunks(
        self,
    ) -> list[DocumentChunk]:
        """模拟从Chroma恢复全部Chunk。"""

        self.list_chunks_call_count += 1

        return [
            chunk.model_copy(deep=True)
            for chunk in self._chunks
        ]

    def retrieve(
        self,
        query_embedding: list[float],
        *,
        query_embedding_model: str,
        top_k: int = 3,
        retrieval_scope: (
            RetrievalScopeFilter | None
        ) = None,
    ) -> list[RetrievedChunk]:
        """记录向量检索参数、范围并返回预设排名。"""

        self.retrieve_calls.append(
            {
                "query_embedding": query_embedding,
                "query_embedding_model": (
                    query_embedding_model
                ),
                "top_k": top_k,
                # 冒烟脚本当前执行全库检索，
                # 因而预期收到None；记录该字段可确保
                # Fake与真实ChromaVectorStore契约同步。
                "retrieval_scope": retrieval_scope,
            }
        )

        return [
            result.model_copy(deep=True)
            for result in self._vector_results
        ]


def install_fake_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    chunks: list[DocumentChunk],
    vector_results: list[RetrievedChunk],
    embedding_error: Exception | None = None,
) -> tuple[
    FakeAsyncClient,
    FakeEmbeddingService,
    FakeVectorStore,
    dict[str, object],
]:
    """把脚本的外部依赖替换成可检查的Fake对象。"""

    fake_settings = SimpleNamespace(
        chroma_persist_directory=Path(
            "fake-chroma-directory"
        ),
        chroma_collection_name=(
            "fake_collection"
        ),
    )

    fake_client = FakeAsyncClient()
    fake_embedding_service = (
        FakeEmbeddingService(
            query_vector=TEST_QUERY_VECTOR,
            error=embedding_error,
        )
    )
    fake_vector_store = FakeVectorStore(
        chunks=chunks,
        vector_results=vector_results,
    )

    # 保存各个工厂函数收到的构造参数，
    # 用于检查脚本是否正确装配依赖。
    constructor_calls: dict[
        str,
        object,
    ] = {}

    def fake_vector_store_factory(
        *,
        persist_directory: Path,
        collection_name: str,
        embedding_model: str,
    ) -> FakeVectorStore:
        """记录Chroma构造参数并返回同一个Fake存储。"""

        constructor_calls[
            "vector_store"
        ] = {
            "persist_directory": persist_directory,
            "collection_name": collection_name,
            "embedding_model": embedding_model,
        }

        return fake_vector_store

    def fake_embedding_service_factory(
        *,
        client: object,
        model: str,
    ) -> FakeEmbeddingService:
        """记录Embedding Service依赖并返回Fake服务。"""

        constructor_calls[
            "embedding_service"
        ] = {
            "client": client,
            "model": model,
        }

        return fake_embedding_service

    def fake_client_factory(
        settings: object,
    ) -> FakeAsyncClient:
        """记录客户端工厂参数并返回Fake异步客户端。"""

        constructor_calls[
            "embedding_client_settings"
        ] = settings

        return fake_client

    # monkeypatch.setattr()只在当前测试期间替换属性；
    # 测试结束后pytest会自动恢复原对象。
    monkeypatch.setattr(
        module,
        "get_settings",
        lambda: fake_settings,
    )
    monkeypatch.setattr(
        module,
        "get_embedding_model",
        lambda settings: TEST_EMBEDDING_MODEL,
    )
    monkeypatch.setattr(
        module,
        "ChromaVectorStore",
        fake_vector_store_factory,
    )
    monkeypatch.setattr(
        module,
        "create_embedding_client",
        fake_client_factory,
    )
    monkeypatch.setattr(
        module,
        "EmbeddingService",
        fake_embedding_service_factory,
    )

    return (
        fake_client,
        fake_embedding_service,
        fake_vector_store,
        constructor_calls,
    )


def test_parse_args_uses_reproducible_defaults(
) -> None:
    """无命令行覆盖时应使用公开语料和固定RRF参数。"""

    args = module.parse_args([])

    assert args.query == module.DEFAULT_QUERY
    assert "ERR-DEMO-1001" in args.query
    assert args.top_k == 3
    assert args.candidate_k == 20
    assert args.rank_constant == 60


def test_parse_args_accepts_and_converts_overrides(
) -> None:
    """命令行覆盖值应被清理并转换成整数。"""

    args = module.parse_args(
        [
            "--query",
            "  ERR-DEMO-1001如何恢复？  ",
            "--top-k",
            "2",
            "--candidate-k",
            "8",
            "--rank-constant",
            "30",
        ]
    )

    assert args.query == (
        "ERR-DEMO-1001如何恢复？"
    )
    assert args.top_k == 2
    assert args.candidate_k == 8
    assert args.rank_constant == 30


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param(
            ["--query", "   "],
            id="blank-query",
        ),
        pytest.param(
            ["--top-k", "0"],
            id="zero-top-k",
        ),
        pytest.param(
            ["--candidate-k", "not-an-int"],
            id="non-integer-candidate-k",
        ),
        pytest.param(
            ["--rank-constant", "-1"],
            id="negative-rank-constant",
        ),
        pytest.param(
            [
                "--top-k",
                "4",
                "--candidate-k",
                "3",
            ],
            id="top-k-larger-than-candidate-k",
        ),
    ],
)
def test_parse_args_rejects_invalid_cli_values(
    argv: list[str],
) -> None:
    """非法命令行参数应由argparse使用退出码2拒绝。"""

    with pytest.raises(
        SystemExit,
    ) as exc_info:
        module.parse_args(argv)

    assert exc_info.value.code == 2


@pytest.mark.parametrize(
    (
        "arguments",
        "expected_exception",
        "expected_message",
    ),
    [
        (
            {"query": 123},
            TypeError,
            "query必须是字符串",
        ),
        (
            {"query": "   "},
            ValueError,
            "query不能为空",
        ),
        (
            {"top_k": True},
            TypeError,
            "top_k必须是整数",
        ),
        (
            {"top_k": 0},
            ValueError,
            "top_k必须大于或等于1",
        ),
        (
            {"candidate_k": False},
            TypeError,
            "candidate_k必须是整数",
        ),
        (
            {"candidate_k": 0},
            ValueError,
            "candidate_k必须大于或等于1",
        ),
        (
            {"rank_constant": True},
            TypeError,
            "rank_constant必须是整数",
        ),
        (
            {"rank_constant": 0},
            ValueError,
            "rank_constant必须大于或等于1",
        ),
        (
            {
                "top_k": 4,
                "candidate_k": 3,
            },
            ValueError,
            "top_k不能大于candidate_k",
        ),
    ],
    ids=[
        "non-string-query",
        "blank-query",
        "boolean-top-k",
        "zero-top-k",
        "boolean-candidate-k",
        "zero-candidate-k",
        "boolean-rank-constant",
        "zero-rank-constant",
        "top-k-larger-than-candidate-k",
    ],
)
def test_runtime_validation_rejects_invalid_direct_calls(
    arguments: dict[str, object],
    expected_exception: type[Exception],
    expected_message: str,
) -> None:
    """绕过argparse直接调用时仍必须执行完整边界校验。"""

    valid_arguments: dict[str, object] = {
        "query": "合法问题",
        "top_k": 3,
        "candidate_k": 20,
        "rank_constant": 60,
    }
    valid_arguments.update(arguments)

    with pytest.raises(
        expected_exception,
        match=expected_message,
    ):
        module._validate_runtime_parameters(
            **valid_arguments,
        )


@pytest.mark.asyncio
async def test_run_smoke_executes_real_keyword_and_rrf_flow(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Fake外部边界下应走完真实关键词检索和RRF流程。"""

    vector_result = RetrievedChunk(
        chunk=TEST_CHUNK.model_copy(
            deep=True
        ),
        similarity=0.72,
        rank=1,
    )

    (
        fake_client,
        fake_embedding_service,
        fake_vector_store,
        constructor_calls,
    ) = install_fake_runtime(
        monkeypatch,
        chunks=[TEST_CHUNK],
        vector_results=[vector_result],
    )

    await module.run_smoke_hybrid_retrieval(
        query=(
            "  ERR-DEMO-1001如何恢复？  "
        ),
        top_k=1,
        candidate_k=5,
        rank_constant=60,
    )

    output = capsys.readouterr().out

    # 脚本应使用Settings中的三个值构造向量库。
    assert constructor_calls[
        "vector_store"
    ] == {
        "persist_directory": Path(
            "fake-chroma-directory"
        ),
        "collection_name": (
            "fake_collection"
        ),
        "embedding_model": (
            TEST_EMBEDDING_MODEL
        ),
    }

    # list_chunks()只应执行一次，
    # 取得的Chunk随后交给真实KeywordRetriever。
    assert (
        fake_vector_store.list_chunks_call_count
        == 1
    )

    # Embedding使用清理首尾空白后的问题。
    assert fake_embedding_service.queries == [
        "ERR-DEMO-1001如何恢复？"
    ]

    # 向量路径应收到Fake生成的向量、
    # 同一模型名称和candidate_k=5。
    assert fake_vector_store.retrieve_calls == [
        {
            "query_embedding": (
                TEST_QUERY_VECTOR
            ),
            "query_embedding_model": (
                TEST_EMBEDDING_MODEL
            ),
            "top_k": 5,
            "retrieval_scope": None,
        }
    ]

    # 正常结束也必须关闭异步客户端。
    assert fake_client.close_call_count == 1

    # 同一Chunk被向量和关键词路径同时召回，
    # 所以输出应包含两条路径的Top-1审计信息。
    assert "Chroma Chunk数：1" in output
    assert "查询向量维度：3" in output
    assert "单路候选深度：5" in output
    assert "融合结果数：1" in output
    assert "融合Top-1" in output
    assert "向量路径：Top-1" in output
    assert "关键词路径：Top-1" in output
    assert "err-demo-1001" in output
    assert "robot-fault-demo.txt" in output


@pytest.mark.asyncio
async def test_run_smoke_closes_client_when_embedding_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Embedding异常向上传播时finally仍必须关闭客户端。"""

    (
        fake_client,
        fake_embedding_service,
        fake_vector_store,
        constructor_calls,
    ) = install_fake_runtime(
        monkeypatch,
        chunks=[TEST_CHUNK],
        vector_results=[],
        embedding_error=RuntimeError(
            "embedding failed"
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="embedding failed",
    ):
        await module.run_smoke_hybrid_retrieval(
            query="ERR-DEMO-1001",
            top_k=1,
            candidate_k=5,
            rank_constant=60,
        )

    assert fake_embedding_service.queries == [
        "ERR-DEMO-1001"
    ]
    assert fake_vector_store.retrieve_calls == []
    assert fake_client.close_call_count == 1

    # 读取变量避免测试替身的装配结果被静默忽略。
    assert "embedding_service" in constructor_calls


@pytest.mark.asyncio
async def test_run_smoke_rejects_empty_collection_before_client_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """空知识库应在创建Embedding客户端前明确失败。"""

    (
        fake_client,
        fake_embedding_service,
        fake_vector_store,
        constructor_calls,
    ) = install_fake_runtime(
        monkeypatch,
        chunks=[],
        vector_results=[],
    )

    with pytest.raises(
        RuntimeError,
        match="Chroma知识库中没有",
    ):
        await module.run_smoke_hybrid_retrieval(
            query="合法问题",
            top_k=1,
            candidate_k=5,
            rank_constant=60,
        )

    assert (
        fake_vector_store.list_chunks_call_count
        == 1
    )

    # 客户端创建发生在空Collection检查之后，
    # 因此这里不应出现客户端工厂调用记录。
    assert (
        "embedding_client_settings"
        not in constructor_calls
    )
    assert fake_client.close_call_count == 0
    assert fake_embedding_service.queries == []
