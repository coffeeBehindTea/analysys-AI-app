"""知识库文档级 Schema 的单元测试。"""

import pytest

# ValidationError 表示传入的数据不满足 Pydantic 模型约束。
from pydantic import ValidationError

from app.schemas.knowledge import (
    DocumentListResponse,
    DocumentRecord,
)


def test_document_record_accepts_valid_data() -> None:
    """成功摄取的文档记录应能通过校验。"""

    document_id = "a" * 64

    record = DocumentRecord(
        document_id=document_id,
        source_file="工业移动机器人安全标准.pdf",
        file_type="pdf",
        file_size_bytes=1_111_510,
        page_or_section_count=34,
        chunk_count=53,
        embedding_model="embedding-3",
    )

    assert record.document_id == document_id
    assert record.file_type == "pdf"
    assert record.page_or_section_count == 34
    assert record.chunk_count == 53

    # status 没有显式传入时，
    # 应使用模型中声明的默认值。
    assert record.status == "ready"

    # model_dump() 把 Pydantic 模型转换成普通字典，
    # 后续 API 序列化和日志记录都可以使用它。
    dumped = record.model_dump()

    assert dumped["source_file"] == (
        "工业移动机器人安全标准.pdf"
    )
    assert dumped["status"] == "ready"


def test_document_record_rejects_invalid_hash() -> None:
    """document_id 必须是 64 位小写 SHA-256。"""

    with pytest.raises(
        ValidationError,
        match="document_id",
    ):
        DocumentRecord(
            document_id="not-a-sha256",
            source_file="manual.pdf",
            file_type="pdf",
            file_size_bytes=100,
            page_or_section_count=1,
            chunk_count=1,
            embedding_model="embedding-3",
        )


def test_document_record_rejects_unsupported_type() -> None:
    """没有实现 Loader 的文件类型不能进入记录。"""

    with pytest.raises(
        ValidationError,
        match="file_type",
    ):
        DocumentRecord(
            document_id="b" * 64,
            source_file="manual.docx",
            file_type="docx",
            file_size_bytes=100,
            page_or_section_count=1,
            chunk_count=1,
            embedding_model="embedding-3",
        )


def test_document_record_rejects_extra_fields() -> None:
    """模型未声明的字段不能被静默保存。"""

    with pytest.raises(
        ValidationError,
        match="unexpected_field",
    ):
        DocumentRecord(
            document_id="c" * 64,
            source_file="manual.txt",
            file_type="txt",
            file_size_bytes=100,
            page_or_section_count=1,
            chunk_count=1,
            embedding_model="embedding-3",

            # 这个字段不属于数据契约，
            # extra="forbid" 应拒绝它。
            unexpected_field="unexpected",
        )


def test_document_list_response_serializes_records() -> None:
    """文档列表响应应保存并递归序列化 DocumentRecord。"""

    record = DocumentRecord(
        document_id="d" * 64,
        source_file="manual.pdf",
        file_type="pdf",
        file_size_bytes=1024,
        page_or_section_count=10,
        chunk_count=20,
        embedding_model="embedding-3",
    )

    # 将已经校验的 DocumentRecord 放入列表响应。
    response = DocumentListResponse(
        documents=[record],
    )

    assert len(response.documents) == 1
    assert response.documents[0].document_id == (
        "d" * 64
    )

    # model_dump() 会递归把外层响应和内部
    # DocumentRecord 都转换成普通 Python 字典。
    dumped = response.model_dump()

    assert dumped["documents"][0]["source_file"] == (
        "manual.pdf"
    )
    assert dumped["documents"][0]["chunk_count"] == 20


def test_document_list_response_has_independent_empty_list() -> None:
    """不同响应实例不能共享同一个默认 documents 列表。"""

    first_response = DocumentListResponse()
    second_response = DocumentListResponse()

    # 修改第一个响应的列表。
    first_response.documents.append(
        DocumentRecord(
            document_id="e" * 64,
            source_file="faults.txt",
            file_type="txt",
            file_size_bytes=100,
            page_or_section_count=1,
            chunk_count=1,
            embedding_model="embedding-3",
        )
    )

    assert len(first_response.documents) == 1

    # default_factory=list 应保证第二个响应仍为空。
    assert second_response.documents == []