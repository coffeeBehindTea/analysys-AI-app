"""知识库与文档摄取配置的单元测试。"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import Settings


def test_knowledge_configuration_is_parsed(
    tmp_path: Path,
) -> None:
    """知识库配置应转换成正确的 Python 类型。"""

    persist_directory = (
        tmp_path / "test-chroma"
    )

    settings = Settings(
        # 不读取项目中的真实 .env，
        # 避免测试依赖开发者本地配置。
        _env_file=None,

        chroma_persist_directory=(
            persist_directory
        ),
        chroma_collection_name=(
            "test_knowledge"
        ),
        rag_chunk_size=800,
        rag_chunk_overlap=120,
        embedding_batch_size=32,
        max_document_size_bytes=1_000_000,
        rag_similarity_threshold=0.70,
    )

    # Path 字段应保存为 Path 对象，
    # 而不是普通字符串。
    assert (
        settings.chroma_persist_directory
        == persist_directory
    )
    assert isinstance(
        settings.chroma_persist_directory,
        Path,
    )

    assert (
        settings.chroma_collection_name
        == "test_knowledge"
    )
    assert settings.rag_chunk_size == 800
    assert settings.rag_chunk_overlap == 120
    assert settings.embedding_batch_size == 32
    assert (
        settings.max_document_size_bytes
        == 1_000_000
    )

    # 创建 Settings 不应该产生数据库目录。
    # 数据库目录由 ChromaVectorStore 创建。
    assert persist_directory.exists() is False

    assert (
    settings.rag_similarity_threshold
    == 0.70
    )


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    """重叠长度不能等于或超过 Chunk 大小。"""

    with pytest.raises(
        ValidationError,
        match="必须小于",
    ):
        Settings(
            _env_file=None,
            rag_chunk_size=800,
            rag_chunk_overlap=800,
        )


def test_embedding_batch_size_must_be_positive() -> None:
    """Embedding 批次大小不能为零。"""

    with pytest.raises(
        ValidationError,
        match="embedding_batch_size",
    ):
        Settings(
            _env_file=None,
            embedding_batch_size=0,
        )


def test_max_document_size_must_be_positive() -> None:
    """允许的最大文件大小不能为零。"""

    with pytest.raises(
        ValidationError,
        match="max_document_size_bytes",
    ):
        Settings(
            _env_file=None,
            max_document_size_bytes=0,
        )


def test_similarity_threshold_must_be_in_range(
) -> None:
    """在线问答相似度阈值必须在0到1之间。"""

    with pytest.raises(
        ValidationError,
        match="rag_similarity_threshold",
    ):
        Settings(
            _env_file=None,
            rag_similarity_threshold=1.01,
        )


def test_default_similarity_threshold_uses_calibrated_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """未提供外部配置时应使用校准后的0.60阈值。"""

    # 删除测试进程中可能存在的同名系统环境变量。
    #
    # raising=False表示变量原本不存在时也不报错。
    # pytest会在测试结束后自动恢复环境。
    monkeypatch.delenv(
        "RAG_SIMILARITY_THRESHOLD",
        raising=False,
    )

    # _env_file=None禁止读取项目中的真实.env，
    # 从而验证Settings类本身声明的默认值。
    settings = Settings(
        _env_file=None,
    )

    assert (
        settings.rag_similarity_threshold
        == 0.60
    )