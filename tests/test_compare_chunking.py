"""切分参数对比脚本的离线测试。"""

from pathlib import Path

import pytest

from scripts.compare_chunking import (
    build_report,
)


def test_build_report_compares_both_configurations(
    tmp_path: Path,
) -> None:
    """报告应包含两套参数及其不同 Chunk 数量。"""

    source_dir = tmp_path / "source"
    source_dir.mkdir()

    # 1000 个字符在两套参数下分别产生：
    #
    # 450/80：
    # 0-450、370-820、740-1000，共 3 个。
    #
    # 800/120：
    # 0-800、680-1000，共 2 个。
    source_file = source_dir / "sample.txt"
    source_file.write_text(
        "A" * 1000,
        encoding="utf-8",
    )

    report = build_report(source_dir)

    assert "| 450/80 | sample.txt | 3 |" in report
    assert "| 800/120 | sample.txt | 2 |" in report

    # build_report() 只返回字符串，
    # 不应该自行创建报告文件。
    assert list(
        tmp_path.glob("*.md")
    ) == []


def test_build_report_ignores_unsupported_files(
    tmp_path: Path,
) -> None:
    """PDF 在 Loader 完成前不应进入文本对比。"""

    source_dir = tmp_path / "source"
    source_dir.mkdir()

    (source_dir / "sample.txt").write_text(
        "测试文本",
        encoding="utf-8",
    )

    (source_dir / "manual.pdf").write_bytes(
        b"fake-pdf"
    )

    report = build_report(source_dir)

    assert "sample.txt" in report
    assert "manual.pdf" not in report


def test_build_report_rejects_empty_source_directory(
    tmp_path: Path,
) -> None:
    """没有可处理文本时应明确失败。"""

    source_dir = tmp_path / "source"
    source_dir.mkdir()

    with pytest.raises(
        ValueError,
        match="没有可处理",
    ):
        build_report(source_dir)


from scripts.compare_chunking import (
    build_report,
    compact_preview,
)


def test_compact_preview_marks_complete_content() -> None:
    """短内容应明确标记为完整内容。"""

    preview = compact_preview(
        "完整的短文本",
        max_length=100,
    )

    assert preview.startswith(
        "**完整内容：**"
    )
    assert "开头" not in preview
    assert "结尾" not in preview


def test_compact_preview_separates_head_and_tail() -> None:
    """长内容应明确区分开头、省略部分和结尾。"""

    preview = compact_preview(
        "START-" + "A" * 100 + "-END",
        max_length=40,
    )

    assert "**开头：**" in preview
    assert "**结尾：**" in preview
    assert "中间省略" in preview
    assert "START-" in preview
    assert "-END" in preview