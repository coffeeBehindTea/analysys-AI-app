"""检索基线使用的批量向量化和缓存逻辑。"""

# Protocol 用来描述对象必须提供哪些方法。
#
# 实际 EmbeddingService 和测试 Fake
# 只要实现相同方法，就可以传给下面的函数。
from typing import Protocol

from app.errors import InvalidEmbeddingResponseError
from app.schemas.retrieval import (
    DocumentChunk,
    EmbeddedChunk,
    EmbeddingVector,
)


class EmbeddingProvider(Protocol):
    """文本向量提供者需要满足的最小接口。"""

    async def embed_texts(
        self,
        texts: list[str],
    ) -> list[EmbeddingVector]:
        """把一批文本转换成同顺序向量。"""

        # Protocol 方法只描述接口，不提供实现。
        ...


# 缓存键由两部分组成：
#
# 1. Embedding 模型名；
# 2. Chunk 正文哈希。
#
# 相同文本使用不同模型时，向量不能复用。
EmbeddingCache = dict[
    tuple[str, str],
    EmbeddingVector,
]


async def embed_document_chunks(
    chunks: list[DocumentChunk],
    *,
    provider: EmbeddingProvider,
    embedding_model: str,
    cache: EmbeddingCache | None = None,
    batch_size: int = 64,
) -> list[EmbeddedChunk]:
    """批量生成文档向量，并复用相同模型下的内容缓存。"""

    cleaned_model = embedding_model.strip()

    if not cleaned_model:
        raise ValueError(
            "Embedding 模型名称不能为空"
        )

    if batch_size < 1:
        raise ValueError(
            "Embedding batch_size 必须大于或等于 1"
        )

    # 不能写成：
    #
    # active_cache = cache or {}
    #
    # 因为空字典会被判断为 False，从而被替换成新字典，
    # 调用方传入的空缓存就无法收到新生成的向量。
    active_cache: EmbeddingCache = (
        cache
        if cache is not None
        else {}
    )

    # 保存尚未存在于缓存中的唯一 Chunk。
    #
    # dict 会保持插入顺序，使批量请求和测试结果稳定。
    missing_chunks: dict[
        tuple[str, str],
        DocumentChunk,
    ] = {}

    for chunk in chunks:
        cache_key = (
            cleaned_model,
            chunk.content_hash,
        )

        # 第一项检查缓存中是否已有向量；
        # 第二项检查本轮是否已经登记了相同正文。
        if (
            cache_key not in active_cache
            and cache_key not in missing_chunks
        ):
            missing_chunks[cache_key] = chunk

    # items() 产生：
    # (缓存键, DocumentChunk)
    #
    # list(...) 保存后才能按照 batch_size 使用切片。
    missing_items = list(
        missing_chunks.items()
    )

    # range(start, stop, step) 按 batch_size 递增。
    #
    # 例如有 130 项、batch_size=64：
    # start 依次是 0、64、128。
    for start in range(
        0,
        len(missing_items),
        batch_size,
    ):
        batch_items = missing_items[
            start:start + batch_size
        ]

        batch_texts = [
            chunk.content
            for _, chunk in batch_items
        ]

        # provider 在真实运行时是 EmbeddingService；
        # 单元测试中则可以是不会联网的 Fake。
        batch_vectors = await provider.embed_texts(
            batch_texts
        )

        # EmbeddingService 本身已经做过此项检查，
        # 这里仍然保护 Protocol 的其他实现，
        # 特别是测试 Fake 或未来替换的本地模型。
        if len(batch_vectors) != len(batch_items):
            raise InvalidEmbeddingResponseError(
                "批量 Embedding 返回数量与输入不一致"
            )

        # zip(..., strict=True) 将缓存项和向量逐项配对。
        #
        # strict=True 表示长度不一致时立即报错，
        # 防止 zip 默认静默丢弃较长一侧的剩余数据。
        for (
            cache_item,
            vector,
        ) in zip(
            batch_items,
            batch_vectors,
            strict=True,
        ):
            cache_key, _ = cache_item
            active_cache[cache_key] = vector

    # 按原始 chunks 的顺序组装 EmbeddedChunk。
    #
    # 缓存只负责避免重复请求，不应该改变文档顺序。
    embedded_chunks: list[EmbeddedChunk] = []

    for chunk in chunks:
        cache_key = (
            cleaned_model,
            chunk.content_hash,
        )

        embedded_chunks.append(
            EmbeddedChunk(
                chunk=chunk,
                embedding_model=cleaned_model,
                embedding=active_cache[cache_key],
            )
        )

    return embedded_chunks