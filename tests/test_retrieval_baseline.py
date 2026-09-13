"""检索基线批量向量化逻辑的离线测试。"""

from hashlib import sha256

import pytest

from app.errors import InvalidEmbeddingResponseError
from app.schemas.retrieval import (
    DocumentChunk,
    EmbeddingVector,
)
from app.services.retrieval_baseline import (
    EmbeddingCache,
    embed_document_chunks,
)


class FakeEmbeddingProvider:
    """不联网的确定性测试向量提供者。"""

    def __init__(self) -> None:
        """初始化调用记录。"""

        # 每个内部 list[str] 表示一次 embed_texts() 调用。
        #
        # 例如：
        # [
        #     ["文本一", "文本二"],
        #     ["文本三"],
        # ]
        #
        # 表示 Provider 被调用两次：
        # 第一批有两项，第二批有一项。
        self.calls: list[list[str]] = []

    async def embed_texts(
        self,
        texts: list[str],
    ) -> list[EmbeddingVector]:
        """根据文本长度生成简单、稳定的假向量。"""

        # list(texts) 创建一个列表副本。
        #
        # 如果调用方之后修改原始 texts，
        # 不会影响这里已经记录的调用参数。
        self.calls.append(
            list(texts)
        )

        # 每段文本得到一个二维向量：
        #
        # 第一维：文本字符数；
        # 第二维：固定为 1.0。
        #
        # 这些向量没有真实语义，只用于验证批处理和对应顺序。
        return [
            [
                float(len(text)),
                1.0,
            ]
            for text in texts
        ]


class BrokenEmbeddingProvider:
    """故意少返回向量的错误 Provider。"""

    async def embed_texts(
        self,
        texts: list[str],
    ) -> list[EmbeddingVector]:
        """无论输入多少文本，都只返回一个向量。"""

        return [[1.0, 0.0]]


def make_chunk(
    chunk_id: str,
    content: str,
    *,
    chunk_index: int = 0,
) -> DocumentChunk:
    """根据正文创建测试使用的 DocumentChunk。"""

    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="document-001",
        source_file="test.txt",
        page_or_section="section: test",
        chunk_index=chunk_index,

        # 测试也使用真实的内容哈希，
        # 这样才能正确验证正文去重和缓存键。
        content_hash=sha256(
            content.encode("utf-8")
        ).hexdigest(),

        content=content,
    )


@pytest.mark.asyncio
async def test_embeddings_are_requested_in_batches() -> None:
    """缺失向量应按照 batch_size 分批生成。"""

    provider = FakeEmbeddingProvider()

    chunks = [
        make_chunk(
            "chunk-1",
            "第一个文本",
            chunk_index=0,
        ),
        make_chunk(
            "chunk-2",
            "第二个更长的文本",
            chunk_index=1,
        ),
        make_chunk(
            "chunk-3",
            "第三个最长的测试文本",
            chunk_index=2,
        ),
    ]

    embedded = await embed_document_chunks(
        chunks,
        provider=provider,
        embedding_model="test-model",
        batch_size=2,
    )

    # 三项按照 batch_size=2 分成两次调用：
    # 第一批两项，第二批一项。
    assert provider.calls == [
        [
            "第一个文本",
            "第二个更长的文本",
        ],
        [
            "第三个最长的测试文本",
        ],
    ]

    assert len(embedded) == 3

    # 虽然内部经历了分批和缓存，
    # 最终输出必须保持原始 Chunk 顺序。
    assert [
        item.chunk.chunk_id
        for item in embedded
    ] == [
        "chunk-1",
        "chunk-2",
        "chunk-3",
    ]

    # 模型名称应写入每个 EmbeddedChunk。
    assert all(
        item.embedding_model == "test-model"
        for item in embedded
    )


@pytest.mark.asyncio
async def test_duplicate_content_is_embedded_once() -> None:
    """正文相同的两个 Chunk 应只生成一次向量。"""

    provider = FakeEmbeddingProvider()

    chunks = [
        make_chunk(
            "chunk-1",
            "完全相同的正文",
            chunk_index=0,
        ),
        make_chunk(
            "chunk-2",
            "完全相同的正文",
            chunk_index=1,
        ),
    ]

    embedded = await embed_document_chunks(
        chunks,
        provider=provider,
        embedding_model="test-model",
    )

    # content_hash 相同，因此只发送一次正文。
    assert provider.calls == [
        ["完全相同的正文"]
    ]

    # 两个 Chunk 仍然分别存在，
    # 只是复用了同一个正文对应的向量。
    assert len(embedded) == 2

    assert (
        embedded[0].chunk.chunk_id
        != embedded[1].chunk.chunk_id
    )

    assert (
        embedded[0].embedding
        == embedded[1].embedding
    )


@pytest.mark.asyncio
async def test_external_cache_is_reused() -> None:
    """多次调用应复用调用方提供的缓存。"""

    provider = FakeEmbeddingProvider()

    # 这个字典由测试代码创建，
    # 并在两次函数调用中使用同一个对象。
    cache: EmbeddingCache = {}

    chunks = [
        make_chunk(
            "chunk-1",
            "可以缓存的正文",
        )
    ]

    await embed_document_chunks(
        chunks,
        provider=provider,
        embedding_model="test-model",
        cache=cache,
    )

    await embed_document_chunks(
        chunks,
        provider=provider,
        embedding_model="test-model",
        cache=cache,
    )

    # 第一次调用生成向量；
    # 第二次调用应完全命中缓存。
    assert provider.calls == [
        ["可以缓存的正文"]
    ]

    # 证明函数修改的是调用方传入的缓存，
    # 而不是另行创建了一个无法访问的新字典。
    assert len(cache) == 1


@pytest.mark.asyncio
async def test_cache_is_isolated_by_model() -> None:
    """同一正文使用不同模型时不能复用向量。"""

    provider = FakeEmbeddingProvider()
    cache: EmbeddingCache = {}

    chunks = [
        make_chunk(
            "chunk-1",
            "相同的正文",
        )
    ]

    await embed_document_chunks(
        chunks,
        provider=provider,
        embedding_model="model-a",
        cache=cache,
    )

    await embed_document_chunks(
        chunks,
        provider=provider,
        embedding_model="model-b",
        cache=cache,
    )

    # 因为模型不同，所以必须分别生成一次。
    assert provider.calls == [
        ["相同的正文"],
        ["相同的正文"],
    ]

    # 缓存中应分别存在：
    #
    # ("model-a", content_hash)
    # ("model-b", content_hash)
    assert len(cache) == 2


@pytest.mark.asyncio
async def test_invalid_provider_response_is_rejected() -> None:
    """Provider 返回数量不正确时必须停止处理。"""

    chunks = [
        make_chunk(
            "chunk-1",
            "第一个文本",
            chunk_index=0,
        ),
        make_chunk(
            "chunk-2",
            "第二个文本",
            chunk_index=1,
        ),
    ]

    with pytest.raises(
        InvalidEmbeddingResponseError,
        match="返回数量与输入不一致",
    ):
        await embed_document_chunks(
            chunks,
            provider=BrokenEmbeddingProvider(),
            embedding_model="test-model",
            batch_size=2,
        )


@pytest.mark.asyncio
async def test_invalid_batch_size_is_rejected() -> None:
    """batch_size 必须至少为 1。"""

    with pytest.raises(
        ValueError,
        match="batch_size",
    ):
        await embed_document_chunks(
            [],
            provider=FakeEmbeddingProvider(),
            embedding_model="test-model",
            batch_size=0,
        )


@pytest.mark.asyncio
async def test_blank_model_name_is_rejected() -> None:
    """只包含空格的模型名称不能进入缓存键。"""

    with pytest.raises(
        ValueError,
        match="模型名称不能为空",
    ):
        await embed_document_chunks(
            [],
            provider=FakeEmbeddingProvider(),
            embedding_model="   ",
        )