"""固定窗口文本切分器的单元测试。"""

from hashlib import sha256

import pytest

from app.schemas.retrieval import SourceTextSegment
from app.services.chunking import TextChunker


def test_split_without_overlap() -> None:
    """不重叠时，每个字符只属于一个 Chunk。"""

    chunker = TextChunker(
        chunk_size=4,
        overlap=0,
    )

    chunks = chunker.split_document(
        document_id="doc-001",
        source_file="test.txt",
        segments=[
            SourceTextSegment(
                page_or_section="section:test",
                content="ABCDEFGHIJ",
            )
        ],
    )

    assert [
        chunk.content
        for chunk in chunks
    ] == [
        "ABCD",
        "EFGH",
        "IJ",
    ]


def test_split_with_overlap() -> None:
    """相邻 Chunk 应保留指定数量的重复字符。"""

    chunker = TextChunker(
        chunk_size=4,
        overlap=2,
    )

    chunks = chunker.split_document(
        document_id="doc-001",
        source_file="test.txt",
        segments=[
            SourceTextSegment(
                page_or_section="section:test",
                content="ABCDEFGHIJ",
            )
        ],
    )

    assert [
        chunk.content
        for chunk in chunks
    ] == [
        "ABCD",
        "CDEF",
        "EFGH",
        "GHIJ",
    ]


def test_chunk_indexes_are_global_across_segments() -> None:
    """Chunk 索引不能在新页面或章节重新从 0 开始。"""

    chunker = TextChunker(
        chunk_size=4,
        overlap=0,
    )

    chunks = chunker.split_document(
        document_id="doc-001",
        source_file="test.txt",
        segments=[
            SourceTextSegment(
                page_or_section="page: 1",
                content="ABCDE",
            ),
            SourceTextSegment(
                page_or_section="page: 2",
                content="XYZ",
            ),
        ],
    )

    assert [
        chunk.chunk_index
        for chunk in chunks
    ] == [0, 1, 2]

    assert [
        chunk.chunk_id
        for chunk in chunks
    ] == [
        "doc-001:000000",
        "doc-001:000001",
        "doc-001:000002",
    ]

    assert [
        chunk.page_or_section
        for chunk in chunks
    ] == [
        "page: 1",
        "page: 1",
        "page: 2",
    ]


def test_chunk_contains_sha256_of_its_content() -> None:
    """content_hash 必须对应当前 Chunk 的真实正文。"""

    chunker = TextChunker(
        chunk_size=4,
        overlap=0,
    )

    chunks = chunker.split_document(
        document_id="doc-001",
        source_file="test.txt",
        segments=[
            SourceTextSegment(
                page_or_section="section:test",
                content="ABCDEFGH",
            )
        ],
    )

    expected_hash = sha256(
        "ABCD".encode("utf-8")
    ).hexdigest()

    assert chunks[0].content_hash == expected_hash


def test_empty_segment_list_is_rejected() -> None:
    """没有文本段的文档不能被切分入库。"""

    chunker = TextChunker(
        chunk_size=450,
        overlap=80,
    )

    with pytest.raises(
        ValueError,
        match="至少需要一个文本段",
    ):
        chunker.split_document(
            document_id="doc-001",
            source_file="empty.txt",
            segments=[],
        )


def test_invalid_chunk_size_is_rejected() -> None:
    """Chunk 大小必须符合 DocumentChunk 数据契约。"""

    with pytest.raises(
        ValueError,
        match="chunk_size",
    ):
        TextChunker(
            chunk_size=0,
            overlap=0,
        )


def test_negative_overlap_is_rejected() -> None:
    """重叠字符数不能为负数。"""

    with pytest.raises(
        ValueError,
        match="overlap 不能",
    ):
        TextChunker(
            chunk_size=450,
            overlap=-1,
        )


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    """重叠量不能阻止切分窗口继续前进。"""

    with pytest.raises(
        ValueError,
        match="overlap 必须小于",
    ):
        TextChunker(
            chunk_size=100,
            overlap=100,
        )