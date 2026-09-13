"""内存向量检索及其数学计算。"""

# fsum() 比普通 sum() 更适合累加大量浮点数，
# 可以减少浮点舍入误差。
#
# isfinite() 检查数值是否为普通有限值，
# 拒绝 NaN、正无穷和负无穷。
#
# sqrt() 计算平方根。
from math import fsum, isfinite, sqrt

from app.schemas.retrieval import (
    EmbeddedChunk,
    EmbeddingVector,
    RetrievedChunk,
)


def cosine_similarity(
    left: EmbeddingVector,
    right: EmbeddingVector,
) -> float:
    """计算两个同维向量之间的余弦相似度。"""

    # Python 类型注解不会自动校验函数调用参数。
    #
    # 即使 EmbeddingVector 声明了 min_length=1，
    # 调用者仍然可能直接把 [] 传进普通 Python 函数，
    # 因此这里需要执行运行时检查。
    if not left or not right:
        raise ValueError(
            "计算余弦相似度时向量不能为空"
        )

    # 余弦相似度只能比较相同维度的向量。
    #
    # 例如 1536 维向量不能和 3072 维向量逐项相乘。
    if len(left) != len(right):
        raise ValueError(
            "计算余弦相似度时向量维度必须一致"
        )

    # 遍历两个向量中的所有数值。
    #
    # 两层生成器相当于：
    # 先遍历 left，再遍历 right。
    if any(
        not isfinite(value)
        for vector in (left, right)
        for value in vector
    ):
        raise ValueError(
            "计算余弦相似度时向量不能包含 NaN 或无穷值"
        )

    # zip(left, right) 将两个向量的同位置元素配对。
    #
    # fsum() 再把逐项乘积精确累加，得到点积。
    dot_product = fsum(
        left_value * right_value
        for left_value, right_value in zip(left, right)
    )

    # 向量长度等于：
    # 每个元素平方之和，再开平方。
    left_magnitude = sqrt(
        fsum(
            value * value
            for value in left
        )
    )
    right_magnitude = sqrt(
        fsum(
            value * value
            for value in right
        )
    )

    # 零向量的所有元素都是 0，它的长度也是 0。
    #
    # 余弦公式需要除以两个向量长度；
    # 如果其中一个长度为 0，就会发生除零，
    # 而且零向量本身也不存在可比较的方向。
    if left_magnitude == 0.0 or right_magnitude == 0.0:
        raise ValueError(
            "零向量没有可计算的余弦相似度"
        )

    similarity = (
        dot_product
        / (left_magnitude * right_magnitude)
    )

    # 正常输入应该得到有限结果。
    #
    # 该检查可以防止极端浮点数计算产生无穷或 NaN。
    if not isfinite(similarity):
        raise ValueError(
            "余弦相似度计算结果不是有限值"
        )

    # 从数学上说，余弦相似度范围是 [-1, 1]。
    #
    # 但浮点运算可能产生：
    # 1.0000000000000002
    #
    # min() 和 max() 将这种微小误差收回理论范围：
    # min(1.0, similarity) 限制最大值；
    # max(-1.0, ...)      限制最小值。
    return max(
        -1.0,
        min(1.0, similarity),
    )

class InMemoryRetriever:
    """在内存中对文档 Chunk 执行余弦相似度检索。"""

    def __init__(
        self,
        chunks: list[EmbeddedChunk],
    ) -> None:
        """复制并检查待检索的文档向量。"""

        # tuple(...) 创建一个元组副本。
        #
        # 如果调用方之后对原来的 chunks 列表执行 append()，
        # 不会在不知情的情况下改变当前检索器中的文档集合。
        self._chunks: tuple[EmbeddedChunk, ...] = tuple(
            chunks
        )

        # 空知识库是允许存在的。
        # 查询空知识库时会返回空结果，之后由 RAG 层决定拒答。
        if not self._chunks:
            return

        # 第一个 Chunk 用来确定当前内存索引的：
        # 1. Embedding 模型；
        # 2. 向量维度。
        first_chunk = self._chunks[0]
        expected_model = first_chunk.embedding_model
        expected_dimension = len(first_chunk.embedding)

        # self._chunks[1:] 表示从第二项开始检查，
        # 因为第一项已经用作参考标准。
        for embedded_chunk in self._chunks[1:]:
            # 不同 Embedding 模型构建的是不同向量空间，
            # 即使维度相同，也不能混合计算相似度。
            if embedded_chunk.embedding_model != expected_model:
                raise ValueError(
                    "内存检索器不能混合不同的 Embedding 模型"
                )

            # 同一个索引中的所有向量必须具有相同维度。
            if (
                len(embedded_chunk.embedding)
                != expected_dimension
            ):
                raise ValueError(
                    "内存检索器不能混合不同维度的向量"
                )

    def retrieve(
        self,
        query_embedding: EmbeddingVector,
        *,
        query_embedding_model: str,
        top_k: int = 3,
    ) -> list[RetrievedChunk]:
        """返回与查询向量最相似的 Top-K 文档 Chunk。"""

        # top_k 表示最多返回多少项。
        # Top-0 和负数在检索语义上没有意义。
        if top_k < 1:
            raise ValueError(
                "top_k 必须大于或等于 1"
            )

        # 模型名称也是检索契约的一部分。
        # strip() 后为空说明调用方没有提供有效模型名。
        cleaned_query_model = query_embedding_model.strip()

        if not cleaned_query_model:
            raise ValueError(
                "查询向量的 Embedding 模型不能为空"
            )

        # 空知识库没有可以参与比较的文档，
        # 返回空结果而不是抛出异常。
        if not self._chunks:
            return []

        # 构造器已经保证所有 Chunk 的模型和维度一致，
        # 因此可以使用第一项代表当前内存索引的配置。
        stored_model = self._chunks[0].embedding_model

        if cleaned_query_model != stored_model:
            raise ValueError(
                "查询向量和文档向量必须使用同一个 Embedding 模型"
            )

        # 每一项保存：
        # (EmbeddedChunk, 余弦相似度)
        scored_chunks: list[
            tuple[EmbeddedChunk, float]
        ] = []

        for embedded_chunk in self._chunks:
            # cosine_similarity() 会继续检查：
            # 空向量、维度、零向量和非法浮点数。
            similarity = cosine_similarity(
                query_embedding,
                embedded_chunk.embedding,
            )

            scored_chunks.append(
                (embedded_chunk, similarity)
            )

        # sorted(...) 返回排序后的新列表，
        # 不修改 scored_chunks 本身。
        #
        # 排序键是一个二元组：
        #
        # 1. -item[1]
        #    item[1] 是相似度。
        #    添加负号后，升序排序等价于相似度降序。
        #
        # 2. item[0].chunk.chunk_id
        #    如果两个结果的相似度完全相同，
        #    使用 chunk_id 升序作为稳定的第二排序条件。
        #
        # 这样相同输入永远产生相同顺序，
        ordered_chunks = sorted(
            scored_chunks,
            key=lambda item: (
                -item[1],
                item[0].chunk.chunk_id,
            ),
        )

        # [:top_k] 截取最多前 top_k 项。
        # 当 top_k 大于文档数量时，Python 会安全返回全部文档。
        top_chunks = ordered_chunks[:top_k]

        results: list[RetrievedChunk] = []

        # enumerate(..., start=1) 在遍历时同时产生排名。
        #
        # start=1 表示排名从 1 开始：
        # 第一项 rank=1，而不是 Python 默认的 0。
        for rank, item in enumerate(
            top_chunks,
            start=1,
        ):
            embedded_chunk, similarity = item

            # RetrievedChunk 只返回原始 Chunk、相似度和排名。
            #
            # 不返回 embedding，可以避免把几千个浮点数
            # 传给后续 LLM 或最终 API 客户端。
            results.append(
                RetrievedChunk(
                    chunk=embedded_chunk.chunk,
                    similarity=similarity,
                    rank=rank,
                )
            )

        return results