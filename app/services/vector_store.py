"""知识库向量存储接口与 ChromaDB 持久化实现。"""

# Path 表示文件系统路径。
from pathlib import Path

# Protocol 用于定义对象必须提供哪些方法，
# 不要求实现类继承这个 Protocol。
from typing import Protocol

# chromadb 是向量数据库客户端库。
import chromadb

from app.errors import (
    DuplicateDocumentError,
    VectorStoreError,
)
from app.schemas.knowledge import DocumentRecord
from app.schemas.retrieval import (
    DocumentChunk,
    EmbeddedChunk,
    EmbeddingVector,
    RetrievedChunk,
)
# 可以识别并拒绝 NaN、正无穷和负无穷。
from math import isfinite


class VectorStore(Protocol):
    """知识库服务所依赖的向量存储能力。"""

    def has_document(
        self,
        document_id: str,
    ) -> bool:
        """判断指定文档是否已经存在。"""

        ...

    def add_document(
        self,
        *,
        record: DocumentRecord,
        chunks: list[EmbeddedChunk],
    ) -> None:
        """保存一份文档产生的全部 Chunk。"""

        ...

    def list_documents(
        self,
    ) -> list[DocumentRecord]:
        """返回已经入库的文档记录。"""

        ...

    def retrieve(
    self,
    query_embedding: EmbeddingVector,
    *,
    query_embedding_model: str,
    top_k: int = 3,
    ) -> list[RetrievedChunk]:
        """返回与查询向量最接近的 Top-K Chunk。"""

        ...

    def delete_document(
        self,
        document_id: str,
    ) -> bool:
        """删除文档的全部 Chunk，并报告是否找到文档。"""

        ...


class ChromaVectorStore:
    """使用本地持久化 ChromaDB 保存知识库 Chunk。"""

    def __init__(
        self,
        *,
        persist_directory: Path,
        collection_name: str,
        embedding_model: str,
    ) -> None:
        """打开持久化数据库并取得知识库 Collection。"""

        cleaned_collection_name = (
            collection_name.strip()
        )
        cleaned_embedding_model = (
            embedding_model.strip()
        )

        if not cleaned_collection_name:
            raise ValueError(
                "Chroma Collection 名称不能为空"
            )

        if not cleaned_embedding_model:
            raise ValueError(
                "Embedding 模型名称不能为空"
            )

        # PersistentClient 把数据库保存在指定目录。
        #
        # str(...) 把 Path 转成 Chroma 客户端接受的字符串路径。
        self._client = chromadb.PersistentClient(
            path=str(persist_directory)
        )

        self._embedding_model = (
            cleaned_embedding_model
        )

        # Collection 类似关系数据库中的表，
        # 用于保存一组使用相同向量空间的数据。
        #
        # get_or_create_collection()：
        # - 已存在时取得原 Collection；
        # - 不存在时创建新 Collection。
        self._collection = (
            self._client.get_or_create_collection(
                name=cleaned_collection_name,

                # 已经通过 EmbeddingService 生成向量，
                # 不让 Chroma 再调用自己的 Embedding 模型。
                embedding_function=None,

                # HNSW 是近似最近邻向量索引。
                #
                # space="cosine" 表示使用余弦距离，
                # 与 Day 1–2 的余弦相似度基线保持一致。
                configuration={
                    "hnsw": {
                        "space": "cosine",
                    }
                },

                # Collection 级元数据记录其向量模型。
                #
                # Collection 中不能混合不同模型产生的向量。
                metadata={
                    "embedding_model": (
                        cleaned_embedding_model
                    ),
                },
            )
        )

        # get_or_create_collection() 在 Collection 已存在时，
        # 不会用新 metadata 覆盖旧 metadata。
        #
        # 所以必须读取实际保存的模型并进行检查。
        collection_metadata = (
            self._collection.metadata or {}
        )
        stored_model = collection_metadata.get(
            "embedding_model"
        )

        if stored_model != cleaned_embedding_model:
            raise ValueError(
                "现有 Chroma Collection 使用的 "
                "Embedding 模型与当前配置不一致"
            )

    def has_document(
        self,
        document_id: str,
    ) -> bool:
        """根据文档哈希检查是否已经存在对应 Chunk。"""

        cleaned_document_id = document_id.strip()

        if not cleaned_document_id:
            raise ValueError(
                "document_id 不能为空"
            )

        try:
            # get() 是按照 ID 或元数据读取记录，
            # 不执行向量相似度搜索。
            #
            # where 表示元数据过滤条件。
            # limit=1 表示只需找到一项即可证明文档存在。
            result = self._collection.get(
                where={
                    "document_id": cleaned_document_id,
                },
                limit=1,

                # 重复检查只需要 ID，
                # 无需取回正文、元数据或向量。
                include=[],
            )
        except Exception as exc:
            raise VectorStoreError(
                "检查重复文档时访问向量库失败"
            ) from exc

        # result["ids"] 是符合过滤条件的 Chunk ID 列表。
        #
        # bool([]) 是 False；
        # bool(["某个 ID"]) 是 True。
        return bool(result["ids"])

    def add_document(
        self,
        *,
        record: DocumentRecord,
        chunks: list[EmbeddedChunk],
    ) -> None:
        """保存文档 Chunk、向量、正文和来源元数据。"""

        if not chunks:
            raise ValueError(
                "写入向量库的 Chunk 列表不能为空"
            )

        if len(chunks) != record.chunk_count:
            raise ValueError(
                "DocumentRecord 的 chunk_count "
                "与实际 Chunk 数量不一致"
            )

        # 在真正写入前检查每个 Chunk 是否属于当前文档。
        for embedded_chunk in chunks:
            chunk = embedded_chunk.chunk

            if chunk.document_id != record.document_id:
                raise ValueError(
                    "不能把其他文档的 Chunk "
                    "写入当前 DocumentRecord"
                )

            if chunk.source_file != record.source_file:
                raise ValueError(
                    "Chunk 文件名与 DocumentRecord 不一致"
                )

            if (
                embedded_chunk.embedding_model
                != record.embedding_model
            ):
                raise ValueError(
                    "Chunk 的 Embedding 模型 "
                    "与 DocumentRecord 不一致"
                )

            if (
                embedded_chunk.embedding_model
                != self._embedding_model
            ):
                raise ValueError(
                    "Chunk 的 Embedding 模型 "
                    "与 Chroma Collection 不一致"
                )

        # 使用整份文件哈希检查重复。
        #
        # 相同内容即使换了文件名，
        # document_id 仍相同，因而会被拒绝。
        if self.has_document(record.document_id):
            raise DuplicateDocumentError(
                f"文档已经存在：{record.source_file}"
            )

        # Chroma 使用“列式参数”：
        #
        # ids[0]、embeddings[0]、documents[0]、
        # metadatas[0] 共同描述第一条记录。
        ids: list[str] = []
        embeddings: list[list[float]] = []
        documents: list[str] = []
        metadatas: list[dict[str, str | int]] = []

        for embedded_chunk in chunks:
            chunk = embedded_chunk.chunk

            ids.append(chunk.chunk_id)
            embeddings.append(
                list(embedded_chunk.embedding)
            )
            documents.append(chunk.content)

            # Chroma metadata 只保存简单标量值，
            # 不在其中嵌套 Pydantic 模型或字典。
            #
            # 文档级元数据会在每个 Chunk 中重复保存，
            # 这样可以按 document_id 过滤、列出和删除。
            metadatas.append(
                {
                    "document_id": record.document_id,
                    "source_file": record.source_file,
                    "file_type": record.file_type,
                    "file_size_bytes": (
                        record.file_size_bytes
                    ),
                    "page_or_section_count": (
                        record.page_or_section_count
                    ),
                    "chunk_count": record.chunk_count,
                    "embedding_model": (
                        record.embedding_model
                    ),
                    "status": record.status,

                    # 以下是当前 Chunk 特有的元数据。
                    "page_or_section": (
                        chunk.page_or_section
                    ),
                    "chunk_index": chunk.chunk_index,
                    "content_hash": chunk.content_hash,
                }
            )

        try:
            # add() 新增记录。
            #
            # 我们不用 upsert()，因为 upsert 会在 ID 已存在时
            # 更新旧数据，不符合“重复文档必须拒绝”的要求。
            self._collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
        except Exception as exc:
            raise VectorStoreError(
                f"写入向量库失败：{record.source_file}"
            ) from exc

    def list_documents(
        self,
    ) -> list[DocumentRecord]:
        """从 Chunk 元数据中还原去重后的文档列表。"""

        try:
            # 当前只需要文档记录，所以不读取：
            # - Chunk 正文；
            # - Embedding 向量。
            result = self._collection.get(
                include=["metadatas"]
            )
        except Exception as exc:
            raise VectorStoreError(
                "读取知识库文档列表失败"
            ) from exc

        metadata_items = result["metadatas"] or []

        # 使用 document_id 作为字典键执行去重。
        records_by_id: dict[str, DocumentRecord] = {}

        try:
            for metadata in metadata_items:
                if metadata is None:
                    raise ValueError(
                        "Chunk 缺少元数据"
                    )

                document_id = str(
                    metadata["document_id"]
                )

                # 同一文档的 53 个 Chunk 都带有相同的
                # 文档级元数据，只需要创建一次记录。
                if document_id in records_by_id:
                    continue

                records_by_id[document_id] = (
                    DocumentRecord(
                        document_id=document_id,
                        source_file=str(
                            metadata["source_file"]
                        ),
                        file_type=str(
                            metadata["file_type"]
                        ),
                        file_size_bytes=int(
                            metadata[
                                "file_size_bytes"
                            ]
                        ),
                        page_or_section_count=int(
                            metadata[
                                "page_or_section_count"
                            ]
                        ),
                        chunk_count=int(
                            metadata["chunk_count"]
                        ),
                        embedding_model=str(
                            metadata[
                                "embedding_model"
                            ]
                        ),
                        status=str(
                            metadata["status"]
                        ),
                    )
                )
        except (
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            raise VectorStoreError(
                "向量库中的文档元数据不完整或无效"
            ) from exc

        # 排序使相同数据始终返回相同顺序，
        # 避免接口结果随数据库内部顺序变化。
        return sorted(
            records_by_id.values(),
            key=lambda record: (
                record.source_file.casefold(),
                record.document_id,
            ),
        )

    def retrieve(
        self,
        query_embedding: EmbeddingVector,
        *,
        query_embedding_model: str,
        top_k: int = 3,
    ) -> list[RetrievedChunk]:
        """使用查询向量召回最接近的知识库 Chunk。"""

        # Top-0 或负数没有检索意义。
        if top_k < 1:
            raise ValueError(
                "top_k 必须大于或等于 1"
            )

        cleaned_model = query_embedding_model.strip()

        if not cleaned_model:
            raise ValueError(
                "查询向量的 Embedding 模型不能为空"
            )

        # 普通 Python 函数不会因为 EmbeddingVector
        # 类型注解而自动执行 Pydantic 校验，
        # 因此仍需检查运行时输入。
        if not query_embedding:
            raise ValueError(
                "查询向量不能为空"
            )

        # Chroma 不能安全处理 NaN 或无穷值，
        # 这些值也没有合法的向量检索含义。
        if any(
            not isfinite(value)
            for value in query_embedding
        ):
            raise ValueError(
                "查询向量不能包含 NaN 或无穷值"
            )

        # 查询向量和数据库中的文档向量
        # 必须来自同一个 Embedding 模型。
        if cleaned_model != self._embedding_model:
            raise ValueError(
                "查询向量和 Chroma Collection "
                "必须使用同一个 Embedding 模型"
            )

        try:
            # count() 返回 Collection 中的记录数量。
            #
            # 当前一条 Chroma 记录对应一个 Chunk，
            # 因此这里得到的是 Chunk 数，不是文档数。
            stored_chunk_count = self._collection.count()

            # 空知识库没有候选证据，
            # 返回空列表，后续 RAG 层据此执行拒答。
            if stored_chunk_count == 0:
                return []

            # query() 执行向量近邻搜索。
            #
            # query_embeddings 必须是二维列表：
            # 外层表示一批查询；
            # 内层的每一项是一条查询向量。
            #
            # 当前只有一个问题，所以使用：
            # [query_embedding]
            result = self._collection.query(
                query_embeddings=[
                    list(query_embedding)
                ],

                # 当知识库只有两个 Chunk、top_k=3 时，
                # 实际最多只能取回两个结果。
                n_results=min(
                    top_k,
                    stored_chunk_count,
                ),

                # 还原 RetrievedChunk 需要：
                # 1. documents：Chunk 正文；
                # 2. metadatas：来源与 Chunk 元数据；
                # 3. distances：与查询向量的距离。
                #
                # 不需要取回原始 Embedding，
                # 避免数据库返回大量无用浮点数。
                include=[
                    "documents",
                    "metadatas",
                    "distances",
                ],
            )
        except Exception as exc:
            raise VectorStoreError(
                "执行 Chroma 向量检索失败"
            ) from exc

        try:
            # query() 支持一次传入多个查询向量，
            # 所以返回值是两层列表。
            #
            # 例如两个查询各取 Top-2：
            #
            # ids = [
            #     ["query-1 的结果 1", "结果 2"],
            #     ["query-2 的结果 1", "结果 2"],
            # ]
            id_batches = result["ids"]
            document_batches = result["documents"]
            metadata_batches = result["metadatas"]
            distance_batches = result["distances"]

            # 当前只传入一个查询向量，
            # 因此应该得到一个结果批次。
            if len(id_batches) != 1:
                raise ValueError(
                    "Chroma 返回的查询批次数量无效"
                )

            if (
                document_batches is None
                or metadata_batches is None
                or distance_batches is None
                or len(document_batches) != 1
                or len(metadata_batches) != 1
                or len(distance_batches) != 1
            ):
                raise ValueError(
                    "Chroma 检索结果缺少必要字段"
                )

            # [0] 取得当前唯一查询对应的结果列表。
            ids = id_batches[0]
            documents = document_batches[0]
            metadatas = metadata_batches[0]
            distances = distance_batches[0]

            # 四列必须一一对应。
            result_count = len(ids)

            if not (
                len(documents) == result_count
                and len(metadatas) == result_count
                and len(distances) == result_count
            ):
                raise ValueError(
                    "Chroma 检索结果各字段数量不一致"
                )

            retrieved_chunks: list[
                RetrievedChunk
            ] = []

            # zip() 将相同位置的 ID、正文、元数据和距离
            # 组合成同一条 Chroma 记录。
            #
            # enumerate(..., start=1) 同时产生从 1 开始的排名。
            for rank, (
                chunk_id,
                document,
                metadata,
                distance,
            ) in enumerate(
                zip(
                    ids,
                    documents,
                    metadatas,
                    distances,
                    strict=True,
                ),
                start=1,
            ):
                if (
                    document is None
                    or metadata is None
                    or distance is None
                ):
                    raise ValueError(
                        "Chroma 检索结果包含空字段"
                    )

                # 确认召回结果仍属于当前模型的向量空间。
                stored_model = str(
                    metadata["embedding_model"]
                )

                if stored_model != self._embedding_model:
                    raise ValueError(
                        "召回 Chunk 的 Embedding "
                        "模型与 Collection 不一致"
                    )

                distance_value = float(distance)

                if not isfinite(distance_value):
                    raise ValueError(
                        "Chroma 返回了非法距离"
                    )

                # 当前 Collection 使用 cosine 距离：
                #
                # cosine_distance = 1 - cosine_similarity
                #
                # 因此：
                #
                # cosine_similarity = 1 - cosine_distance
                similarity = 1.0 - distance_value

                # 理论余弦相似度范围是 [-1, 1]。
                #
                # 这里允许极小的浮点误差，
                # 但拒绝明显超出范围的数据。
                tolerance = 1e-6

                if (
                    similarity < -1.0 - tolerance
                    or similarity > 1.0 + tolerance
                ):
                    raise ValueError(
                        "由 Chroma 距离转换出的 "
                        "相似度超出合法范围"
                    )

                # 消除类似 1.0000000001 的浮点误差，
                # 使其满足 RetrievedChunk 的字段约束。
                similarity = max(
                    -1.0,
                    min(1.0, similarity),
                )

                # Chroma 中正文和元数据是分开保存的，
                # 这里重新构造项目内部统一的 DocumentChunk。
                chunk = DocumentChunk(
                    chunk_id=str(chunk_id),
                    document_id=str(
                        metadata["document_id"]
                    ),
                    source_file=str(
                        metadata["source_file"]
                    ),
                    page_or_section=str(
                        metadata["page_or_section"]
                    ),
                    chunk_index=int(
                        metadata["chunk_index"]
                    ),
                    content_hash=str(
                        metadata["content_hash"]
                    ),
                    content=str(document),
                )

                retrieved_chunks.append(
                    RetrievedChunk(
                        chunk=chunk,
                        similarity=similarity,
                        rank=rank,
                    )
                )

        except (
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            # KeyError：缺少必要元数据字段；
            # TypeError：返回结构或字段类型错误；
            # ValueError：数值转换或 Pydantic 校验失败。
            raise VectorStoreError(
                "Chroma 检索结果结构无效"
            ) from exc

        return retrieved_chunks

    def delete_document(
        self,
        document_id: str,
    ) -> bool:
        """删除指定文档产生的所有向量和 Chunk。"""

        cleaned_document_id = document_id.strip()

        if not cleaned_document_id:
            raise ValueError(
                "document_id 不能为空"
            )

        # 不存在时返回 False，
        # 让上层决定转换成 HTTP 404。
        if not self.has_document(cleaned_document_id):
            return False

        try:
            # delete(where=...) 会删除所有符合元数据条件的记录。
            #
            # 一份文档可能有几十个 Chunk，
            # 所以不能只删除某一个 chunk_id。
            self._collection.delete(
                where={
                    "document_id": cleaned_document_id,
                }
            )
        except Exception as exc:
            raise VectorStoreError(
                "从向量库删除文档失败"
            ) from exc

        return True