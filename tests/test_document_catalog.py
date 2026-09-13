"""DocumentCatalogService 的离线单元测试。"""

import pytest

from app.errors import DocumentNotFoundError
from app.schemas.knowledge import DocumentRecord
from app.services.document_catalog import (
    DocumentCatalogService,
)


def make_record() -> DocumentRecord:
    """创建供目录服务测试使用的文档记录。"""

    return DocumentRecord(
        document_id="a" * 64,
        source_file="manual.pdf",
        file_type="pdf",
        file_size_bytes=1024,
        page_or_section_count=10,
        chunk_count=20,
        embedding_model="embedding-3",
    )


class FakeDocumentCatalogStore:
    """只在内存中保存 DocumentRecord 的假目录存储。"""

    def __init__(
        self,
        records: list[DocumentRecord] | None = None,
    ) -> None:
        # 使用 document_id 作为键，模拟数据库中的唯一文档。
        self.records = {
            record.document_id: record
            for record in (records or [])
        }

    def list_documents(
        self,
    ) -> list[DocumentRecord]:
        """返回当前保存的所有文档记录。"""

        # list(...) 创建新列表，
        # 防止测试调用方直接修改内部字典。
        return list(self.records.values())

    def delete_document(
        self,
        document_id: str,
    ) -> bool:
        """删除文档并返回它是否原本存在。"""

        if document_id not in self.records:
            return False

        del self.records[document_id]
        return True


def test_list_documents_returns_stored_records() -> None:
    """Catalog Service 应返回存储层中的文档记录。"""

    record = make_record()
    store = FakeDocumentCatalogStore([record])
    service = DocumentCatalogService(
        vector_store=store,
    )

    result = service.list_documents()

    assert result == [record]


def test_delete_document_removes_existing_record() -> None:
    """删除已有文档后，目录列表应为空。"""

    record = make_record()
    store = FakeDocumentCatalogStore([record])
    service = DocumentCatalogService(
        vector_store=store,
    )

    service.delete_document(record.document_id)

    assert service.list_documents() == []


def test_delete_missing_document_raises_error() -> None:
    """删除不存在的文档应产生明确的业务异常。"""

    store = FakeDocumentCatalogStore()
    service = DocumentCatalogService(
        vector_store=store,
    )

    with pytest.raises(
        DocumentNotFoundError,
        match="不存在",
    ):
        service.delete_document("b" * 64)