"""TXT 文档加载器的单元测试。"""

from pathlib import Path

import pytest

from app.services.document_loader import (
    MarkdownDocumentLoader,
    PdfDocumentLoader,
    TxtDocumentLoader,
)

# MagicMock 创建行为可配置的模拟对象；
# patch 临时替换代码实际使用的对象。
from unittest.mock import MagicMock, patch

def test_txt_loader_splits_fault_sections(
    tmp_path: Path,
) -> None:
    """故障 TXT 应根据真正的故障编号切成多个 Segment。"""

    # tmp_path 是 pytest 内置 Fixture。
    #
    # pytest 会为当前测试创建一个独立临时目录，
    # 测试结束后负责清理。
    path = tmp_path / "faults.txt"

    # write_text() 创建临时测试文件。
    # 测试不依赖 data/source 中的真实语料。
    path.write_text(
        """故障代码 (ErrorCode)
ERR-NAV-2001
故障名称
定位丢失

故障代码 (ErrorCode)
ERR-SAF-1002
故障名称
触边碰撞
""",
        encoding="utf-8",
    )

    loader = TxtDocumentLoader()
    segments = loader.load(path)

    assert [
        segment.page_or_section
        for segment in segments
    ] == [
        "section: ERR-NAV-2001",
        "section: ERR-SAF-1002",
    ]

    assert "定位丢失" in segments[0].content
    assert "触边碰撞" in segments[1].content


def test_malformed_marker_does_not_create_fake_section(
    tmp_path: Path,
) -> None:
    """没有紧跟 ERR 编号的粘连文字不能被当成新故障。"""

    path = tmp_path / "faults.txt"

    path.write_text(
        """故障代码 (ErrorCode)
ERR-DRV-3005
故障名称
驱动电机过温
严重等级故障代码 (ErrorCode)
重度 (Critical)

故障代码 (ErrorCode)
ERR-NET-4001
故障名称
心跳包超时
""",
        encoding="utf-8",
    )

    loader = TxtDocumentLoader()
    segments = loader.load(path)

    # 应该只有两个真正包含 ERR 编号的章节，
    # 不能产生一个没有编号的虚假章节。
    assert len(segments) == 2

    assert [
        segment.page_or_section
        for segment in segments
    ] == [
        "section: ERR-DRV-3005",
        "section: ERR-NET-4001",
    ]

    # 粘连内容仍然保留在 ERR-DRV-3005 章节中，
    # 不会因为解析而丢失原文。
    assert (
        "严重等级故障代码"
        in segments[0].content
    )


def test_generic_txt_becomes_one_segment(
    tmp_path: Path,
) -> None:
    """没有故障编号的普通 TXT 仍然可以被读取。"""

    path = tmp_path / "manual.txt"

    path.write_text(
        "这是一个没有故障编号的普通说明文档。",
        encoding="utf-8",
    )

    loader = TxtDocumentLoader()
    segments = loader.load(path)

    assert len(segments) == 1
    assert (
        segments[0].page_or_section
        == "section: text-1"
    )
    assert (
        segments[0].content
        == "这是一个没有故障编号的普通说明文档。"
    )


def test_empty_txt_is_rejected(
    tmp_path: Path,
) -> None:
    """只包含空白的 TXT 文件不能进入知识库。"""

    path = tmp_path / "empty.txt"

    path.write_text(
        "   \n\n   ",
        encoding="utf-8",
    )

    loader = TxtDocumentLoader()

    with pytest.raises(
        ValueError,
        match="没有可用文本",
    ):
        loader.load(path)


def test_wrong_file_extension_is_rejected(
    tmp_path: Path,
) -> None:
    """TXT Loader 不负责解析 Markdown。"""

    path = tmp_path / "manual.md"

    path.write_text(
        "# 测试文档",
        encoding="utf-8",
    )

    loader = TxtDocumentLoader()

    with pytest.raises(
        ValueError,
        match="只支持 .txt",
    ):
        loader.load(path)


def test_markdown_loader_builds_heading_paths(
    tmp_path: Path,
) -> None:
    """Markdown 标题应转换成完整章节路径。"""

    path = tmp_path / "manual.md"

    path.write_text(
        """# 机器人测试规程

文档总体说明。

## 3. 系统边界

系统边界说明。

### 3.1 测试对象

测试对象正文。

### 3.2 系统组成

系统组成正文。

## 4. 角色职责

角色职责正文。
""",
        encoding="utf-8",
    )

    loader = MarkdownDocumentLoader()
    segments = loader.load(path)

    assert [
        segment.page_or_section
        for segment in segments
    ] == [
        "section: 机器人测试规程",
        "section: 机器人测试规程 > 3. 系统边界",
        (
            "section: 机器人测试规程"
            " > 3. 系统边界"
            " > 3.1 测试对象"
        ),
        (
            "section: 机器人测试规程"
            " > 3. 系统边界"
            " > 3.2 系统组成"
        ),
        "section: 机器人测试规程 > 4. 角色职责",
    ]

    assert (
        "系统组成正文"
        in segments[3].content
    )

    # 3.2 章节不应该包含后面的 4. 角色职责。
    assert (
        "角色职责正文"
        not in segments[3].content
    )


def test_markdown_section_keeps_heading_in_content(
    tmp_path: Path,
) -> None:
    """标题本身应保留在用于 Embedding 的正文中。"""

    path = tmp_path / "manual.md"

    path.write_text(
        """## 安全检查

测试前必须确认急停按钮可用。
""",
        encoding="utf-8",
    )

    segments = MarkdownDocumentLoader().load(path)

    assert segments[0].content.startswith(
        "## 安全检查"
    )


def test_markdown_without_headings_becomes_one_segment(
    tmp_path: Path,
) -> None:
    """没有标题的 Markdown 应作为完整文档处理。"""

    path = tmp_path / "notes.md"

    path.write_text(
        "这是一篇没有 Markdown 标题的说明。",
        encoding="utf-8",
    )

    segments = MarkdownDocumentLoader().load(path)

    assert len(segments) == 1
    assert (
        segments[0].page_or_section
        == "section: document"
    )


def test_markdown_preamble_is_preserved(
    tmp_path: Path,
) -> None:
    """第一条标题前的文字不能被丢弃。"""

    path = tmp_path / "notes.md"

    path.write_text(
        """这是一段标题前的说明。

# 正式章节

正式正文。
""",
        encoding="utf-8",
    )

    segments = MarkdownDocumentLoader().load(path)

    assert (
        segments[0].page_or_section
        == "section: preamble"
    )
    assert (
        segments[0].content
        == "这是一段标题前的说明。"
    )


def test_markdown_loader_rejects_txt(
    tmp_path: Path,
) -> None:
    """Markdown Loader 不负责处理 TXT。"""

    path = tmp_path / "notes.txt"

    path.write_text(
        "普通文本",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="只支持 .md",
    ):
        MarkdownDocumentLoader().load(path)


def test_pdf_loader_extracts_non_empty_pages(
    tmp_path: Path,
) -> None:
    """PDF Loader 应保留真实物理页码并跳过空页。"""

    path = tmp_path / "manual.pdf"

    # Loader 会先使用 is_file() 检查文件是否存在，
    # 所以这里创建一个占位文件。
    #
    # PDF 的实际解析将由 Mock 接管，
    # 文件内容不需要是真实 PDF。
    path.write_bytes(b"fake pdf")

    # 模拟三个 pdfplumber Page 对象。
    first_page = MagicMock()
    first_page.extract_text.return_value = (
        "  第一页安全说明。  "
    )

    blank_page = MagicMock()
    blank_page.extract_text.return_value = None

    third_page = MagicMock()
    third_page.extract_text.return_value = (
        "\n第三页故障处理。\n"
    )

    # 模拟 pdfplumber 打开后返回的整份 PDF。
    fake_pdf = MagicMock()
    fake_pdf.pages = [
        first_page,
        blank_page,
        third_page,
    ]

    # with pdfplumber.open(...) as pdf 会调用
    # 返回对象的 __enter__() 方法。
    #
    # 这里规定进入 with 后得到 fake_pdf 本身。
    fake_pdf.__enter__.return_value = fake_pdf

    # __exit__() 返回 False，表示如果 with 内出现异常，
    # 不要吞掉异常。
    fake_pdf.__exit__.return_value = False

    # patch() 临时替换 document_loader 模块实际使用的
    # pdfplumber.open。
    #
    # 离开 with 后，原来的函数会自动恢复。
    with patch(
        "app.services.document_loader.pdfplumber.open",
        return_value=fake_pdf,
    ) as mocked_open:
        segments = PdfDocumentLoader().load(path)

    # 确认 Loader 确实使用当前路径打开了一次 PDF。
    mocked_open.assert_called_once_with(path)

    # 第二页虽然被跳过，但第三页仍必须标记为 page: 3，
    # 不能错误地变成 page: 2。
    assert [
        segment.page_or_section
        for segment in segments
    ] == [
        "page: 1",
        "page: 3",
    ]

    # strip() 应清理文字首尾的空白。
    assert segments[0].content == "第一页安全说明。"
    assert segments[1].content == "第三页故障处理。"


def test_pdf_loader_rejects_document_without_text(
    tmp_path: Path,
) -> None:
    """全部页面都无文字层时不能假装摄取成功。"""

    path = tmp_path / "scanned.pdf"
    path.write_bytes(b"fake pdf")

    blank_page = MagicMock()
    blank_page.extract_text.return_value = None

    whitespace_page = MagicMock()
    whitespace_page.extract_text.return_value = (
        "  \n  "
    )

    fake_pdf = MagicMock()
    fake_pdf.pages = [
        blank_page,
        whitespace_page,
    ]
    fake_pdf.__enter__.return_value = fake_pdf
    fake_pdf.__exit__.return_value = False

    with patch(
        "app.services.document_loader.pdfplumber.open",
        return_value=fake_pdf,
    ):
        with pytest.raises(
            ValueError,
            match="没有可提取文本",
        ):
            PdfDocumentLoader().load(path)


def test_pdf_loader_wraps_parsing_failure(
    tmp_path: Path,
) -> None:
    """损坏 PDF 的底层异常应转换成清晰的解析错误。"""

    path = tmp_path / "broken.pdf"
    path.write_bytes(b"not a valid pdf")

    # side_effect 表示调用被替换的函数时，
    # 不返回结果，而是抛出指定异常。
    with patch(
        "app.services.document_loader.pdfplumber.open",
        side_effect=OSError("damaged pdf"),
    ):
        with pytest.raises(
            ValueError,
            match="PDF 解析失败",
        ) as exc_info:
            PdfDocumentLoader().load(path)

    # raise ... from exc 会把原始异常保存在 __cause__。
    # 这个断言确认底层原因没有丢失。
    assert isinstance(
        exc_info.value.__cause__,
        OSError,
    )


def test_markdown_groups_subsections_into_level_three(
    tmp_path: Path,
) -> None:
    """三级测试用例及其四级子章节应组成一个 Segment。"""

    path = tmp_path / "standard.md"

    path.write_text(
        """# 机器人测试规程

## 网络与调度测试

### TEST-RCS-001 交叉路口冲突调度

#### 测试目的

验证调度系统能够解决交叉路口冲突。

#### 测试步骤

连续执行 20 轮冲突调度测试。

#### 通过标准

每轮只能有一台机器人进入冲突区域。
""",
        encoding="utf-8",
    )

    segments = MarkdownDocumentLoader().load(path)

    # 一级和二级标题没有自己的正文；
    # 四级标题全部归入三级测试用例，
    # 因此最终只有一个语义 Segment。
    assert len(segments) == 1

    assert segments[0].page_or_section == (
        "section: 机器人测试规程"
        " > 网络与调度测试"
        " > TEST-RCS-001 交叉路口冲突调度"
    )

    # 用于 Embedding 的内容同时保留：
    # 三级测试编号和全部四级子章节。
    assert segments[0].content.startswith(
        "### TEST-RCS-001 交叉路口冲突调度"
    )
    assert "#### 测试目的" in segments[0].content
    assert "#### 测试步骤" in segments[0].content
    assert "#### 通过标准" in segments[0].content
    assert "连续执行 20 轮" in segments[0].content


def test_grouped_markdown_sections_do_not_overlap(
    tmp_path: Path,
) -> None:
    """相邻三级测试用例不能互相包含正文。"""

    path = tmp_path / "standard.md"

    path.write_text(
        """# 测试规程

## 网络测试

### TEST-A

#### 通过标准

这是测试 A 的标准。

### TEST-B

#### 通过标准

这是测试 B 的标准。
""",
        encoding="utf-8",
    )

    segments = MarkdownDocumentLoader().load(path)

    assert len(segments) == 2

    assert "这是测试 A 的标准" in segments[0].content
    assert "这是测试 B 的标准" not in segments[0].content

    assert "这是测试 B 的标准" in segments[1].content
    assert "这是测试 A 的标准" not in segments[1].content