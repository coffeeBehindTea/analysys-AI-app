"""文本文件预处理编排服务的单元测试。"""

from hashlib import sha256
from pathlib import Path

import pytest

from app.services.chunking import TextChunker
from app.services.document_preparation import (
    TextDocumentPreparationService,
    calculate_file_sha256,
)

from app.schemas.retrieval import SourceTextSegment

# patch 用于临时替换指定方法，
# 让测试不依赖真实 PDF 解析。
from unittest.mock import patch


def test_calculate_file_sha256_uses_original_bytes(
    tmp_path: Path,
) -> None:
    """文件哈希必须对应原始字节。"""

    path = tmp_path / "example.txt"
    raw_bytes = "机器人故障说明".encode("utf-8")

    # write_bytes() 直接写入 bytes，
    # 不执行文本编码和换行处理。
    path.write_bytes(raw_bytes)

    expected = sha256(raw_bytes).hexdigest()

    assert calculate_file_sha256(path) == expected


def test_prepare_txt_creates_traceable_chunks(
    tmp_path: Path,
) -> None:
    """TXT 应经过加载和切分得到完整 Chunk 元数据。"""

    path = tmp_path / "faults.txt"

    path.write_text(
        """故障代码 (ErrorCode)
ERR-NAV-2001
故障名称
定位丢失
排查步骤
检查二维码和激光雷达。
""",
        encoding="utf-8",
    )

    service = TextDocumentPreparationService(
        chunker=TextChunker(
            chunk_size=1000,
            overlap=100,
        )
    )

    chunks = service.prepare(path)

    assert len(chunks) == 1

    chunk = chunks[0]

    # document_id 来自整份原始文件。
    assert (
        chunk.document_id
        == calculate_file_sha256(path)
    )

    # 不应保存 tmp_path 的绝对目录。
    assert chunk.source_file == "faults.txt"

    assert (
        chunk.page_or_section
        == "section: ERR-NAV-2001"
    )

    assert chunk.chunk_id == (
        f"{chunk.document_id}:000000"
    )

    assert "检查二维码" in chunk.content


def test_prepare_markdown_preserves_heading_path(
    tmp_path: Path,
) -> None:
    """Markdown 章节路径应继续保留在最终 Chunk 中。"""

    path = tmp_path / "manual.md"

    path.write_text(
        """# 测试规程

总体说明。

## 安全检查

测试前必须确认急停按钮可用。
""",
        encoding="utf-8",
    )

    service = TextDocumentPreparationService(
        chunker=TextChunker(
            chunk_size=1000,
            overlap=100,
        )
    )

    chunks = service.prepare(path)

    assert [
        chunk.page_or_section
        for chunk in chunks
    ] == [
        "section: 测试规程",
        "section: 测试规程 > 安全检查",
    ]


def test_same_file_content_has_same_document_id(
    tmp_path: Path,
) -> None:
    """文件名不同但内容相同时，document_id 应相同。"""

    first_path = tmp_path / "first.txt"
    second_path = tmp_path / "second.txt"

    content = "相同的文档正文"

    first_path.write_text(
        content,
        encoding="utf-8",
    )
    second_path.write_text(
        content,
        encoding="utf-8",
    )

    service = TextDocumentPreparationService(
        chunker=TextChunker(
            chunk_size=100,
            overlap=10,
        )
    )

    first_chunks = service.prepare(first_path)
    second_chunks = service.prepare(second_path)

    assert (
        first_chunks[0].document_id
        == second_chunks[0].document_id
    )


def test_changed_file_content_changes_document_id(
    tmp_path: Path,
) -> None:
    """文件内容改变后应得到不同 document_id。"""

    first_path = tmp_path / "first.txt"
    second_path = tmp_path / "second.txt"

    first_path.write_text(
        "修改前的正文",
        encoding="utf-8",
    )
    second_path.write_text(
        "修改后的正文",
        encoding="utf-8",
    )

    assert (
        calculate_file_sha256(first_path)
        != calculate_file_sha256(second_path)
    )


def test_unsupported_file_type_is_rejected(
    tmp_path: Path,
) -> None:
    """DOCX 尚未实现 Loader，应被明确拒绝。"""

    path = tmp_path / "manual.docx"
    path.write_bytes(b"fake-docx")

    service = TextDocumentPreparationService(
        chunker=TextChunker(
            chunk_size=450,
            overlap=80,
        )
    )

    with pytest.raises(
        ValueError,
        match="只支持",
    ):
        service.prepare(path)


def test_invalid_hash_block_size_is_rejected(
    tmp_path: Path,
) -> None:
    """哈希读取块大小必须是正数。"""

    path = tmp_path / "example.txt"
    path.write_text(
        "测试正文",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="块大小",
    ):
        calculate_file_sha256(
            path,
            block_size=0,
        )



def test_prepare_pdf_preserves_page_metadata(
    tmp_path: Path,
) -> None:
    """PDF 页面位置应继续保留在最终 Chunk 中。"""

    path = tmp_path / "manual.pdf"

    # 预处理服务需要根据原始文件字节计算 document_id，
    # 所以即使 PDF Loader 被 Mock，也要创建一个文件。
    path.write_bytes(b"fake pdf bytes")

    service = TextDocumentPreparationService(
        chunker=TextChunker(
            chunk_size=1000,
            overlap=100,
        )
    )

    # 这里模拟 PdfDocumentLoader 已经完成逐页解析。
    parsed_segments = [
        SourceTextSegment(
            page_or_section="page: 1",
            content="第一页安全要求。",
        ),
        SourceTextSegment(
            page_or_section="page: 3",
            content="第三页故障处理步骤。",
        ),
    ]

    # patch 的目标应该是被测模块实际使用的名字。
    #
    # prepare() 使用的是 document_preparation 模块中
    # 导入的 PdfDocumentLoader，因此替换这个位置。
    with patch(
        (
            "app.services.document_preparation"
            ".PdfDocumentLoader.load"
        ),
        return_value=parsed_segments,
    ) as mocked_load:
        chunks = service.prepare(path)

    # 确认 PDF 文件确实被分配给 PDF Loader。
    mocked_load.assert_called_once_with(path)

    # 两个页面都短于 chunk_size，
    # 所以每个页面产生一个 Chunk。
    assert len(chunks) == 2

    # Loader 创建的物理页码应传递到最终 Chunk，
    # 不能在切分阶段丢失。
    assert [
        chunk.page_or_section
        for chunk in chunks
    ] == [
        "page: 1",
        "page: 3",
    ]

    # 两个 Chunk 都属于同一份原始 PDF。
    expected_document_id = calculate_file_sha256(path)

    assert all(
        chunk.document_id == expected_document_id
        for chunk in chunks
    )

    # chunk_index 是整份文档中的全局顺序，
    # 而不是每页重新从 0 开始。
    assert [
        chunk.chunk_index
        for chunk in chunks
    ] == [0, 1]

    assert chunks[0].source_file == "manual.pdf"