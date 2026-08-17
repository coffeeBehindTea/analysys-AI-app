"""调用 Embedding 接口并把文本安全地转换成浮点向量。"""

# isfinite() 用于检查浮点数是否为普通有限值，
# 它会拒绝 NaN、正无穷和负无穷。
from math import isfinite

# 这些类型来自 OpenAI Python SDK。
#
# AsyncOpenAI：
#   异步客户端类型。
#
# APITimeoutError：
#   请求等待时间超过客户端 timeout。
#
# APIConnectionError：
#   DNS、连接建立、连接中断等网络层错误。
#
# APIStatusError：
#   上游已经返回 HTTP 响应，但状态码是 4xx 或 5xx。
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
)

from app.errors import (
    EmbeddingTimeoutError,
    EmbeddingUpstreamError,
    InvalidEmbeddingResponseError,
)
from app.schemas.retrieval import EmbeddingVector


class EmbeddingService:
    """负责把查询文本或文档文本批量转换成 Embedding 向量。"""

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
    ) -> None:
        """保存由外部注入的 SDK 客户端和模型名称。"""

        # 下划线前缀表示这些属性只供类的内部实现使用。
        self._client = client
        self._model = model

    async def embed_texts(
        self,
        texts: list[str],
    ) -> list[EmbeddingVector]:
        """批量生成向量，并保证返回顺序与输入文本顺序一致。"""

        # 空批次无法生成向量，通常意味着调用方代码存在逻辑错误。
        #
        # 这里使用 ValueError，而不是上游异常，因为此时还没有发出请求。
        if not texts:
            raise ValueError("Embedding 文本列表不能为空")

        # strip() 删除每段文本首尾空白。
        #
        # 使用新的列表，不直接修改调用方传入的 texts。
        cleaned_texts = [
            text.strip()
            for text in texts
        ]

        # any(...) 只要发现一段清理后为空的文本，就返回 True。
        #
        # Embedding 接口不接受空字符串；同时，空文本的向量
        # 对检索也没有实际语义价值。
        if any(not text for text in cleaned_texts):
            raise ValueError(
                "Embedding 文本不能是空字符串或只包含空白"
            )

        try:
            # embeddings 选择 SDK 中的 Embedding API 资源。
            #
            # create(...) 对应 POST /embeddings。
            #
            # input 传入字符串列表，要求上游为每个字符串生成一个向量。
            # 而不是经过 Base64 编码的二进制文本。
            #
            # await 会在等待网络响应时让出事件循环，
            # 使服务器可以继续处理其他异步任务。
            response = await self._client.embeddings.create(
                # 指定本次生成向量使用的模型。
                model=self._model,
                # 字符串数组表示在一次请求中批量生成多个向量。
                input=cleaned_texts,
            )
        except APITimeoutError as exc:
            # APITimeoutError 也是一种连接类错误，因此必须先捕获它。
            raise EmbeddingTimeoutError(
                "Embedding 服务响应超时"
            ) from exc
        except APIConnectionError as exc:
            raise EmbeddingUpstreamError(
                "无法连接 Embedding 服务"
            ) from exc
        except APIStatusError as exc:
            # status_code 是 APIStatusError 对象提供的 HTTP 状态码属性。
            raise EmbeddingUpstreamError(
                "Embedding 服务返回错误状态："
                f"{exc.status_code}"
            ) from exc

        # response.data 是 SDK 返回的 Embedding 对象列表。
        #
        # 每个对象的 index 表示它属于 input 中的第几个文本。
        # 不直接假设服务器一定按输入顺序返回，而是根据 index 排序。
        ordered_items = sorted(
            response.data,
            key=lambda item: item.index,
        )

        # 正确响应的 index 应该是：
        # 输入 2 段文本 -> [0, 1]
        # 输入 3 段文本 -> [0, 1, 2]
        actual_indexes = [
            item.index
            for item in ordered_items
        ]
        expected_indexes = list(
            range(len(cleaned_texts))
        )

        # 该检查同时可以发现：
        # 1. 少返回了向量；
        # 2. 多返回了向量；
        # 3. index 重复；
        # 4. index 越界或不连续。
        if actual_indexes != expected_indexes:
            raise InvalidEmbeddingResponseError(
                "Embedding 响应数量或索引与输入不一致"
            )

        # item.embedding 是 SDK 返回的 list[float]。
        #
        # list(...) 创建独立列表，避免后续代码意外修改
        # SDK 响应对象内部保存的原始列表。
        vectors: list[EmbeddingVector] = [
            list(item.embedding)
            for item in ordered_items
        ]

        # 正常情况下前面的索引检查已经保证 vectors 非空，
        # 这里仍显式检查，使响应验证逻辑更加独立和清楚。
        if not vectors or any(
            not vector
            for vector in vectors
        ):
            raise InvalidEmbeddingResponseError(
                "Embedding 响应中包含空向量"
            )

        # 同一个模型在同一次请求中返回的向量维度必须完全相同。
        #
        # 例如第一个向量有 1536 项，后续也都必须有 1536 项。
        expected_dimension = len(vectors[0])

        if any(
            len(vector) != expected_dimension
            for vector in vectors
        ):
            raise InvalidEmbeddingResponseError(
                "Embedding 响应中的向量维度不一致"
            )

        # 两层 for 表示遍历每个向量中的每个浮点数。
        #
        # isfinite(value) 只有在 value 不是 NaN 和无穷值时才返回 True。
        if any(
            not isfinite(value)
            for vector in vectors
            for value in vector
        ):
            raise InvalidEmbeddingResponseError(
                "Embedding 响应中包含非有限浮点数"
            )

        return vectors

    async def embed_query(
        self,
        query: str,
    ) -> EmbeddingVector:
        """为单个查询生成向量。"""

        # 复用批量方法，保证文档和查询经过完全相同的：
        # 空值检查、异常转换和响应校验。
        vectors = await self.embed_texts([query])

        # 单元素输入验证成功后，必然只对应一个向量。
        return vectors[0]