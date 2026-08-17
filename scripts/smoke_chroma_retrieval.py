"""对持久化 ChromaDB 执行一次真实检索冒烟验证。"""

# asyncio.run()负责从同步命令行入口
# 启动异步 Embedding 查询。
import asyncio

from app.config import get_settings
from app.services.embedding import EmbeddingService
from app.services.embedding_client import (
    create_embedding_client,
    get_embedding_model,
)
from app.services.vector_store import (
    ChromaVectorStore,
)


async def main() -> None:
    """生成真实查询向量并检索持久化知识库。"""

    # get_settings()读取并缓存 .env 和默认配置。
    settings = get_settings()

    # 取得经过非空校验的 Embedding 模型名。
    embedding_model = get_embedding_model(
        settings
    )

    # 创建 OpenAI 兼容异步客户端。
    #
    # 创建对象时不会联网；
    # 真正请求发生在 embed_query()。
    client = create_embedding_client(
        settings
    )

    try:
        # EmbeddingService 将问题文本转成向量。
        embedding_service = EmbeddingService(
            client=client,
            model=embedding_model,
        )

        # 使用与摄取时相同的：
        # 1. 持久化目录；
        # 2. Collection；
        # 3. Embedding 模型。
        vector_store = ChromaVectorStore(
            persist_directory=(
                settings.chroma_persist_directory
            ),
            collection_name=(
                settings.chroma_collection_name
            ),
            embedding_model=embedding_model,
        )

        query = (
            "仓储机器人出现 ERR-DRV-3005 时，"
            "触发条件是什么，应检查哪些问题？"
        )

        # embed_query()内部会复用 embed_texts([query])，
        # 因而使用与文档完全相同的响应校验规则。
        query_embedding = (
            await embedding_service.embed_query(
                query
            )
        )

        # retrieve()不会再次调用 Embedding 服务。
        #
        # 它把已经生成的查询向量传给 Chroma，
        # 返回相似度最高的三个 Chunk。
        results = vector_store.retrieve(
            query_embedding,
            query_embedding_model=embedding_model,
            top_k=3,
        )

        print(f"问题：{query}")
        print(f"查询向量维度：{len(query_embedding)}")
        print(f"召回结果数：{len(results)}")

        for result in results:
            # result.chunk 是 RetrievedChunk 中嵌套的
            # DocumentChunk Pydantic 对象。
            chunk = result.chunk

            # split()按连续空白切分；
            # join()把正文整理成单行，方便终端预览。
            preview = " ".join(
                chunk.content.split()
            )[:240]

            print()
            print(f"Top-{result.rank}")
            print(
                f"相似度：{result.similarity:.6f}"
            )
            print(
                f"来源文件：{chunk.source_file}"
            )
            print(
                f"位置：{chunk.page_or_section}"
            )
            print(f"Chunk ID：{chunk.chunk_id}")
            print(f"正文预览：{preview}")

    finally:
        # 无论查询成功还是发生异常，
        # 都关闭 AsyncOpenAI 的底层 HTTP 连接池。
        await client.close()


if __name__ == "__main__":
    # asyncio.run()创建事件循环，
    # 执行 main()，完成后关闭事件循环。
    asyncio.run(main())