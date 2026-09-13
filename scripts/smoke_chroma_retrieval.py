"""对持久化 ChromaDB 执行一次真实检索冒烟验证。"""

# argparse负责把命令行参数转换成Python对象。
import argparse

# asyncio.run()负责从同步命令行入口
# 启动异步Embedding查询。
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


# 默认问题必须与公开仓库中的脱敏样例一致。
#
# 旧问题ERR-DRV-3005只存在于本地完整语料，
# 无法用于验证公开仓库的最小复现流程。
DEFAULT_QUERY = (
    "模拟故障代码 ERR-DEMO-1001 的观察现象、"
    "安全排查步骤和恢复条件是什么？"
)

# 默认查看前三个候选Chunk。
DEFAULT_TOP_K = 3


def parse_non_blank_query(
    value: str,
) -> str:
    """清理并校验命令行中的查询文本。"""

    # strip()删除查询两端的空格和换行。
    cleaned_value = value.strip()

    if not cleaned_value:
        # ArgumentTypeError会让argparse输出标准的
        # 命令用法和参数错误，并以退出码2结束。
        raise argparse.ArgumentTypeError(
            "--query 不能为空或只包含空白"
        )

    return cleaned_value


def parse_positive_int(
    value: str,
) -> int:
    """把命令行文本转换成大于零的整数。"""

    try:
        parsed_value = int(value)
    except ValueError as exc:
        # from exc保留原始int转换错误作为异常原因。
        raise argparse.ArgumentTypeError(
            "--top-k 必须是整数"
        ) from exc

    if parsed_value < 1:
        raise argparse.ArgumentTypeError(
            "--top-k 必须大于或等于1"
        )

    return parsed_value


def parse_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """解析真实命令行或测试传入的参数列表。"""

    parser = argparse.ArgumentParser(
        description=(
            "对持久化Chroma知识库执行一次"
            "真实Embedding检索冒烟验证"
        )
    )

    parser.add_argument(
        "--query",
        type=parse_non_blank_query,
        default=DEFAULT_QUERY,
        help="需要进行向量检索的问题",
    )

    parser.add_argument(
        "--top-k",
        type=parse_positive_int,
        default=DEFAULT_TOP_K,
        help="最多返回多少个检索候选",
    )

    # argv=None时，argparse自动读取终端参数；
    # 测试传入列表时，不读取pytest自己的参数。
    return parser.parse_args(argv)


async def run_smoke_retrieval(
    *,
    query: str,
    top_k: int,
) -> None:
    """生成查询向量并检索持久化知识库。"""

    # 即使调用方绕过parse_args()直接调用本函数，
    # 业务边界仍然拒绝非法参数。
    cleaned_query = query.strip()

    if not cleaned_query:
        raise ValueError(
            "query不能为空或只包含空白"
        )

    if top_k < 1:
        raise ValueError(
            "top_k必须大于或等于1"
        )

    # get_settings()从.env读取并缓存配置。
    settings = get_settings()

    # 取得经过非空校验的Embedding模型名。
    embedding_model = get_embedding_model(
        settings
    )

    # 创建OpenAI兼容异步客户端。
    #
    # 创建对象时不会联网；
    # 真正请求发生在embed_query()。
    client = create_embedding_client(
        settings
    )

    try:
        # EmbeddingService负责：
        # 输入清理、网络调用、异常转换和向量校验。
        embedding_service = EmbeddingService(
            client=client,
            model=embedding_model,
        )

        # 使用与摄取时完全相同的持久化目录、
        # Collection和Embedding模型。
        vector_store = ChromaVectorStore(
            persist_directory=(
                settings.chroma_persist_directory
            ),
            collection_name=(
                settings.chroma_collection_name
            ),
            embedding_model=embedding_model,
        )

        # embed_query()内部复用embed_texts([query])，
        # 确保文档和问题使用相同的Embedding规则。
        query_embedding = (
            await embedding_service.embed_query(
                cleaned_query
            )
        )

        # retrieve()不会再次调用Embedding服务。
        #
        # 它将已经生成的查询向量交给Chroma，
        # 按余弦相似度返回Top-K Chunk。
        results = vector_store.retrieve(
            query_embedding,
            query_embedding_model=embedding_model,
            top_k=top_k,
        )

        # 冒烟验证必须至少取回一个结果。
        #
        # 如果只打印“结果数0”然后正常退出，
        # 自动化环境可能误以为验证成功。
        if not results:
            raise RuntimeError(
                "Chroma知识库没有返回任何检索结果"
            )

        print(f"问题：{cleaned_query}")
        print(
            f"查询向量维度：{len(query_embedding)}"
        )
        print(f"召回结果数：{len(results)}")

        for result in results:
            # result.chunk是RetrievedChunk中嵌套的
            # DocumentChunk Pydantic对象。
            chunk = result.chunk

            # split()按连续空白切分；
            # join()将正文整理成单行；
            # [:240]限制终端预览长度。
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
        # 无论查询成功、无结果还是发生异常，
        # 都关闭AsyncOpenAI底层HTTP连接池。
        await client.close()


def main() -> None:
    """同步命令行入口。"""

    args = parse_args()

    # asyncio.run()创建事件循环，
    # 执行异步检索，结束后关闭事件循环。
    asyncio.run(
        run_smoke_retrieval(
            query=args.query,
            top_k=args.top_k,
        )
    )


if __name__ == "__main__":
    # 只有使用python -m执行脚本时才调用main()。
    #
    # pytest导入本模块时不会发起真实网络请求。
    main()