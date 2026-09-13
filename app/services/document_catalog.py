"""知识库文档目录的查询与删除服务。"""

from typing import Protocol

from app.errors import DocumentNotFoundError
from app.schemas.knowledge import DocumentRecord


class DocumentCatalogStore(Protocol):
    """文档目录服务依赖的最小存储接口。"""

    def list_documents(
        self,
    ) -> list[DocumentRecord]:
        """返回已经成功摄取的文档记录。"""

        ...

    def delete_document(
        self,
        document_id: str,
    ) -> bool:
        """删除文档并返回是否找到该文档。"""

        ...


class DocumentCatalogService:
    """负责列出和删除知识库中的已有文档。"""

    def __init__(
        self,
        *,
        vector_store: DocumentCatalogStore,
    ) -> None:
        """保存实现文档目录操作的存储对象。"""

        # 这里只要求 list_documents() 和 delete_document()。
        #
        # 真实运行时传入 ChromaVectorStore；
        # 单元测试中可以传入 Fake Store。
        self._vector_store = vector_store

    def list_documents(
        self,
    ) -> list[DocumentRecord]:
        """列出知识库中已经完成摄取的文档。"""

        # 文档元数据由 VectorStore 从每个 Chunk 中还原并去重。
        # Catalog Service 负责把该业务能力暴露给上层 Router。
        return self._vector_store.list_documents()

    def delete_document(
        self,
        document_id: str,
    ) -> None:
        """删除指定文档产生的全部 Chunk。"""

        # VectorStore 返回 bool：
        # True 表示找到并删除；
        # False 表示没有这个 document_id。
        deleted = self._vector_store.delete_document(
            document_id
        )

        if not deleted:
            # Service 把底层的 False 转换成有业务意义的异常。
            #
            # 后续 FastAPI 异常处理器会将其转换为 404 响应。
            raise DocumentNotFoundError(
                f"知识库文档不存在：{document_id}"
            )