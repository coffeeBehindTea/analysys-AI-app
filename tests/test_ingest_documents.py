"""真实摄取脚本中纯编排部分的离线测试。"""

from pathlib import Path

import pytest

from app.config import Settings
from app.errors import (
    DocumentValidationError,
    DuplicateDocumentError,
)
from app.schemas.knowledge import DocumentRecord
from scripts.ingest_documents import (
    IngestionLogEntry,
    build_ingestion_report,
    find_source_files,
    ingest_files,
)

class FakeDocumentIngester:
    """根据文件名模拟成功、重复和失败。"""

    async def ingest_document(
        self,
        path: Path,
    ) -> DocumentRecord:
        """返回假记录或抛出指定业务异常。"""

        if path.name == "duplicate.txt":
            raise DuplicateDocumentError(
                "文档已经存在：duplicate.txt"
            )

        if path.name == "invalid.pdf":
            raise DocumentValidationError(
                "PDF 没有可提取文本"
            )

        return DocumentRecord(
            document_id="a" * 64,
            source_file=path.name,
            file_type="txt",
            file_size_bytes=100,
            page_or_section_count=2,
            chunk_count=3,
            embedding_model="fake-embedding",
        )


def test_find_source_files_filters_and_sorts(
    tmp_path: Path,
) -> None:
    """只选择受支持文件，并按文件名稳定排序。"""

    (tmp_path / "b.txt").write_text(
        "文本",
        encoding="utf-8",
    )
    (tmp_path / "A.md").write_text(
        "# 文档",
        encoding="utf-8",
    )
    (tmp_path / "c.pdf").write_bytes(
        b"fake pdf"
    )
    (tmp_path / "ignored.docx").write_bytes(
        b"fake docx"
    )
    (tmp_path / ".gitkeep").write_bytes(b"")

    files = find_source_files(tmp_path)

    assert [
        path.name
        for path in files
    ] == [
        "A.md",
        "b.txt",
        "c.pdf",
    ]


@pytest.mark.asyncio
async def test_ingest_files_classifies_results(
    tmp_path: Path,
) -> None:
    """批量摄取应区分成功、重复跳过和失败。"""

    paths = [
        tmp_path / "success.txt",
        tmp_path / "duplicate.txt",
        tmp_path / "invalid.pdf",
    ]

    entries = await ingest_files(
        service=FakeDocumentIngester(),
        source_files=paths,
    )

    assert [
        entry.status
        for entry in entries
    ] == [
        "ingested",
        "skipped",
        "failed",
    ]

    assert entries[0].record is not None
    assert entries[0].record.chunk_count == 3

    assert entries[1].record is None
    assert "已经存在" in entries[1].message

    assert entries[2].record is None
    assert (
        "没有可提取文本"
        in entries[2].message
    )


def test_build_ingestion_report_contains_summary() -> None:
    """Markdown 日志应包含统计和成功文档信息。"""

    record = DocumentRecord(
        document_id="b" * 64,
        source_file="manual.txt",
        file_type="txt",
        file_size_bytes=100,
        page_or_section_count=2,
        chunk_count=3,
        embedding_model="fake-embedding",
    )

    # 使用 ingest_files()产生的对象类型，
    # 避免在测试中重复实现日志结构。
    from scripts.ingest_documents import (
        IngestionLogEntry,
    )

    entries = [
        IngestionLogEntry(
            source_file="manual.txt",
            status="ingested",
            record=record,
            message="摄取成功",
        ),
        IngestionLogEntry(
            source_file="duplicate.txt",
            status="skipped",
            record=None,
            message="文档已经存在",
        ),
    ]

    settings = Settings(
        _env_file=None,
        chroma_persist_directory=Path(
            "test-chroma"
        ),
        chroma_collection_name="test_knowledge",
        rag_chunk_size=800,
        rag_chunk_overlap=120,
    )

    report = build_ingestion_report(
        entries=entries,
        settings=settings,
        embedding_model="fake-embedding",
    )

    assert "成功 1；跳过 1；失败 0" in report
    assert "manual.txt" in report
    assert "`" + ("b" * 64) + "`" in report
    assert "test_knowledge" in report
    assert "生成时间" not in report

    # 报告不应包含 Chunk 正文或向量。
    assert "机器人急停正文" not in report
    assert "[1.0, 0.0]" not in report
