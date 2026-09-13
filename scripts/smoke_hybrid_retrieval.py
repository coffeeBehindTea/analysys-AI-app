"""对真实Chroma知识库执行一次混合检索冒烟验证。

本脚本会执行：

1. 从配置中打开真实Chroma Collection；
2. 从Chroma恢复全部DocumentChunk；
3. 为这些Chunk建立内存关键词索引；
4. 调用真实Embedding服务生成查询向量；
5. 分别执行向量检索和关键词检索；
6. 使用RRF融合两份候选排名；
7. 输出融合结果及两条路径的审计信息。

本脚本不会调用LLM，也不会生成最终问答内容。
"""

# argparse把终端传入的字符串参数
# 转换为带属性的Namespace对象。
import argparse

# asyncio.run()负责从同步命令行入口
# 启动异步Embedding调用。
import asyncio

from app.config import get_settings
from app.services.embedding import (
    EmbeddingService,
)
from app.services.embedding_client import (
    create_embedding_client,
    get_embedding_model,
)
from app.services.hybrid_retrieval import (
    DEFAULT_HYBRID_CANDIDATE_K,
    DEFAULT_RRF_RANK_CONSTANT,
    HybridRetriever,
)
from app.services.keyword_retrieval import (
    KeywordRetriever,
)
from app.services.vector_store import (
    ChromaVectorStore,
)


# 默认问题来自公开仓库中的脱敏模拟语料，
# 不依赖不能提交到GitHub的真实资料。
DEFAULT_QUERY = (
    "模拟故障代码 ERR-DEMO-1001 的观察现象、"
    "安全排查步骤和恢复条件是什么？"
)

# RRF融合后默认只展示前三项结果。
DEFAULT_TOP_K = 3


def parse_non_blank_query(
    value: str,
) -> str:
    """清理并校验命令行中的查询文本。"""

    # strip()返回删除首尾空白后的新字符串，
    # 不会修改原字符串。
    cleaned_value = value.strip()

    if not cleaned_value:
        # argparse遇到ArgumentTypeError后会：
        #
        # 1. 打印参数错误；
        # 2. 显示命令用法；
        # 3. 使用退出码2结束。
        raise argparse.ArgumentTypeError(
            "查询问题不能为空或只包含空白"
        )

    return cleaned_value


def parse_positive_int(
    value: str,
) -> int:
    """把命令行文本转换成大于等于1的整数。"""

    try:
        # 命令行参数最初都是字符串，
        # int()负责把"20"转换成整数20。
        parsed_value = int(value)
    except ValueError as exc:
        # from exc保留原始转换异常，
        # 方便调试时查看完整异常链。
        raise argparse.ArgumentTypeError(
            "参数必须是整数"
        ) from exc

    if parsed_value < 1:
        raise argparse.ArgumentTypeError(
            "参数必须大于或等于1"
        )

    return parsed_value


def parse_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """解析真实终端参数或测试提供的参数列表。"""

    parser = argparse.ArgumentParser(
        description=(
            "使用真实Embedding、Chroma、关键词检索"
            "和RRF执行一次混合检索冒烟验证"
        )
    )

    parser.add_argument(
        "--query",
        type=parse_non_blank_query,
        default=DEFAULT_QUERY,
        help="需要检索的问题",
    )

    parser.add_argument(
        "--top-k",
        type=parse_positive_int,
        default=DEFAULT_TOP_K,
        help="RRF融合后最终返回的结果数量",
    )

    parser.add_argument(
        "--candidate-k",
        type=parse_positive_int,
        default=DEFAULT_HYBRID_CANDIDATE_K,
        help="向量和关键词路径各自召回的候选数量",
    )

    parser.add_argument(
        "--rank-constant",
        type=parse_positive_int,
        default=DEFAULT_RRF_RANK_CONSTANT,
        help="RRF公式中的排名平滑常数",
    )

    # argv=None时，parse_args()读取真实终端参数。
    #
    # 测试可以传入字符串列表，避免误读pytest参数。
    args = parser.parse_args(argv)

    # 最终返回数量不能超过每条路径的候选深度。
    #
    # parser.error()会打印错误消息，
    # 然后以退出码2结束。
    if args.top_k > args.candidate_k:
        parser.error(
            "--top-k不能大于--candidate-k"
        )

    return args


def _validate_positive_integer(
    *,
    value: object,
    name: str,
) -> int:
    """校验直接调用脚本函数时传入的整数参数。"""

    # bool是int的子类：
    #
    # isinstance(True, int)
    # → True
    #
    # 但True不能作为有意义的候选深度，
    # 所以必须先排除bool。
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
    ):
        raise TypeError(
            f"{name}必须是整数"
        )

    if value < 1:
        raise ValueError(
            f"{name}必须大于或等于1"
        )

    return value


def _validate_runtime_parameters(
    *,
    query: object,
    top_k: object,
    candidate_k: object,
    rank_constant: object,
) -> tuple[str, int, int, int]:
    """校验绕过命令行解析器直接调用时的参数。"""

    # Python类型注解不会在运行时自动检查输入。
    if not isinstance(query, str):
        raise TypeError(
            "query必须是字符串"
        )

    cleaned_query = query.strip()

    if not cleaned_query:
        raise ValueError(
            "query不能为空或只包含空白"
        )

    validated_top_k = (
        _validate_positive_integer(
            value=top_k,
            name="top_k",
        )
    )

    validated_candidate_k = (
        _validate_positive_integer(
            value=candidate_k,
            name="candidate_k",
        )
    )

    validated_rank_constant = (
        _validate_positive_integer(
            value=rank_constant,
            name="rank_constant",
        )
    )

    if validated_top_k > validated_candidate_k:
        raise ValueError(
            "top_k不能大于candidate_k"
        )

    return (
        cleaned_query,
        validated_top_k,
        validated_candidate_k,
        validated_rank_constant,
    )


async def run_smoke_hybrid_retrieval(
    *,
    query: str,
    top_k: int,
    candidate_k: int,
    rank_constant: int,
) -> None:
    """执行一次真实混合检索并打印可审计结果。"""

    (
        cleaned_query,
        validated_top_k,
        validated_candidate_k,
        validated_rank_constant,
    ) = _validate_runtime_parameters(
        query=query,
        top_k=top_k,
        candidate_k=candidate_k,
        rank_constant=rank_constant,
    )

    # get_settings()读取环境变量和.env，
    # 并返回经过Pydantic校验的Settings对象。
    settings = get_settings()

    # 取得非空的Embedding模型名称。
    embedding_model = get_embedding_model(
        settings
    )

    # 使用与文档摄取时相同的：
    #
    # 1. Chroma持久化目录；
    # 2. Collection名称；
    # 3. Embedding模型。
    vector_store = ChromaVectorStore(
        persist_directory=(
            settings.chroma_persist_directory
        ),
        collection_name=(
            settings.chroma_collection_name
        ),
        embedding_model=embedding_model,
    )

    # list_chunks()只读取正文和元数据，
    # 不会把全部Embedding向量加载进内存。
    all_chunks = vector_store.list_chunks()

    if not all_chunks:
        raise RuntimeError(
            "Chroma知识库中没有可用于混合检索的Chunk"
        )

    # KeywordRetriever在内存中为全部Chunk建立
    # 确定性的关键词特征索引。
    #
    # 它不调用LLM，也不访问网络。
    keyword_retriever = KeywordRetriever(
        chunks=all_chunks,
    )

    # HybridRetriever只负责编排：
    #
    # 向量召回 → 关键词召回 → RRF融合。
    #
    # 它本身不会生成查询Embedding。
    hybrid_retriever = HybridRetriever(
        vector_retriever=vector_store,
        keyword_retriever=keyword_retriever,
        embedding_model=embedding_model,
        candidate_k=validated_candidate_k,
        rank_constant=validated_rank_constant,
    )

    # 创建OpenAI兼容Embedding客户端。
    #
    # 创建客户端对象本身不会联网，
    # 真正的网络请求发生在embed_query()中。
    client = create_embedding_client(
        settings
    )

    try:
        embedding_service = EmbeddingService(
            client=client,
            model=embedding_model,
        )

        # 将自然语言问题转换成查询向量。
        query_embedding = (
            await embedding_service.embed_query(
                cleaned_query
            )
        )

        # HybridRetriever接收已经生成好的查询向量，
        # 不会重复调用Embedding服务。
        results = hybrid_retriever.retrieve(
            query=cleaned_query,
            query_embedding=query_embedding,
            top_k=validated_top_k,
        )

        # 非空Chroma中，向量路径通常至少会返回一项。
        #
        # 如果最终融合结果为空，说明真实调用链
        # 没有按预期工作，应当让冒烟验证失败。
        if not results:
            raise RuntimeError(
                "混合检索没有返回任何候选结果"
            )

        print(f"问题：{cleaned_query}")
        print(
            f"Chroma Chunk数：{len(all_chunks)}"
        )
        print(
            f"查询向量维度：{len(query_embedding)}"
        )
        print(
            "单路候选深度："
            f"{validated_candidate_k}"
        )
        print(
            "RRF排名常数："
            f"{validated_rank_constant}"
        )
        print(
            f"融合结果数：{len(results)}"
        )

        for result in results:
            chunk = result.chunk

            # 把正文中的换行和连续空白压缩成单个空格，
            # 并限制终端预览长度。
            preview = " ".join(
                chunk.content.split()
            )[:240]

            if (
                result.vector_rank is not None
                and result.vector_similarity
                is not None
            ):
                vector_summary = (
                    f"Top-{result.vector_rank}；"
                    "余弦相似度="
                    f"{result.vector_similarity:.6f}"
                )
            else:
                vector_summary = (
                    "未进入向量候选列表"
                )

            if (
                result.keyword_rank is not None
                and result.keyword_score is not None
            ):
                keyword_summary = (
                    f"Top-{result.keyword_rank}；"
                    "关键词分数="
                    f"{result.keyword_score:.3f}"
                )
            else:
                keyword_summary = (
                    "未进入关键词候选列表"
                )

            # 四类命中词合并后只用于终端审计，
            # 不会改变RRF排名。
            matched_terms = (
                *result.matched_identifiers,
                *result.matched_model_aliases,
                *result.matched_numeric_terms,
                *result.matched_lexical_terms,
            )

            matched_terms_summary = (
                "、".join(matched_terms)
                if matched_terms
                else "无"
            )

            print()
            print(f"融合Top-{result.rank}")
            print(
                f"RRF分数：{result.rrf_score:.6f}"
            )
            print(
                f"向量路径：{vector_summary}"
            )
            print(
                f"关键词路径：{keyword_summary}"
            )
            print(
                "关键词命中："
                f"{matched_terms_summary}"
            )
            print(
                f"来源文件：{chunk.source_file}"
            )
            print(
                f"位置：{chunk.page_or_section}"
            )
            print(
                f"Chunk ID：{chunk.chunk_id}"
            )
            print(f"正文预览：{preview}")

    finally:
        # close()是异步方法，负责关闭
        # AsyncOpenAI内部HTTP连接池。
        #
        # finally保证正常结束或异常时都会执行。
        await client.close()


def main() -> None:
    """同步命令行入口。"""

    args = parse_args()

    # asyncio.run()创建事件循环，
    # 运行异步函数，结束后关闭事件循环。
    asyncio.run(
        run_smoke_hybrid_retrieval(
            query=args.query,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            rank_constant=args.rank_constant,
        )
    )


if __name__ == "__main__":
    # 只有使用python -m执行脚本时才运行。
    #
    # pytest导入本模块时不会访问真实数据库，
    # 也不会发送Embedding请求。
    main()