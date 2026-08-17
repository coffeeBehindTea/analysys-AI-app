"""把 PDF、Markdown 和 TXT 解析成带来源位置的文本段。"""

# re 是 Python 标准库中的正则表达式模块，
# 当前用于识别 TXT 故障编号和 Markdown 标题。
import re

# Path 是面向对象的文件路径类型。
from pathlib import Path

# pdfplumber 是第三方 PDF 解析库。
# 本模块使用它打开 PDF，并逐页提取文字层。
import pdfplumber

from app.schemas.retrieval import SourceTextSegment


# 识别一个真正的故障章节起点。
#
# 只有“故障代码 (ErrorCode)”后面紧跟 ERR-... 编号时，
# 才把它判断成新章节。
#
# (?=...) 是正向先行断言：
# 它检查当前位置后面的内容是否符合规则，
# 但分割时不会删除被匹配的“故障代码”文本。
FAULT_SECTION_PATTERN = re.compile(
    r"(?="
    r"故障代码\s*\(ErrorCode\)"
    r"\s*\r?\n\s*"
    r"ERR-[A-Z]+-\d+"
    r")"
)

def _read_utf8_text(
    path: Path,
    *,
    expected_suffix: str,
    loader_name: str,
) -> str:
    """检查文件并返回清理后的 UTF-8 文本。"""

    if path.suffix.lower() != expected_suffix:
        raise ValueError(
            f"{loader_name} 只支持 {expected_suffix} 文件"
        )

    if not path.is_file():
        raise FileNotFoundError(
            f"文件不存在：{path}"
        )

    try:
        text = path.read_text(
            encoding="utf-8"
        )
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"文件不是有效的 UTF-8 文本：{path.name}"
        ) from exc

    cleaned_text = text.strip()

    if not cleaned_text:
        raise ValueError(
            f"文件没有可用文本：{path.name}"
        )

    return cleaned_text


class PdfDocumentLoader:
    """读取 PDF 的文字层，并按物理页生成文本段。"""

    def load(
        self,
        path: Path,
    ) -> list[SourceTextSegment]:
        """把 PDF 中有可提取文字的页面转换成文本段。"""

        # lower() 将扩展名转成小写，
        # 因而 ".PDF" 和 ".pdf" 都能够被接受。
        if path.suffix.lower() != ".pdf":
            raise ValueError(
                "PdfDocumentLoader 只支持 .pdf 文件"
            )

        # is_file() 同时检查：
        # 1. 路径是否存在；
        # 2. 路径是否确实指向一个普通文件。
        if not path.is_file():
            raise FileNotFoundError(
                f"文件不存在：{path}"
            )

        segments: list[SourceTextSegment] = []

        try:
            # pdfplumber.open() 打开 PDF，
            # 返回一个 pdfplumber PDF 对象。
            #
            # 离开 with 代码块时，无论成功还是发生异常，
            # PDF 文件都会被正确关闭。
            with pdfplumber.open(path) as pdf:
                # pdf.pages 保存这份 PDF 的所有 Page 对象。
                #
                # enumerate(..., start=1) 在遍历页面的同时，
                # 生成从 1 开始的物理页码。
                for page_number, page in enumerate(
                    pdf.pages,
                    start=1,
                ):
                    # extract_text() 尝试从当前页的文字层
                    # 按阅读顺序提取字符串。
                    #
                    # 没有文字层或页面为空时，
                    # 它可能返回 None 或空字符串。
                    # PDF 保存的通常是带坐标的字符，而不是完整的单词和段落。
                    # pdfplumber 会根据相邻字符的水平距离判断是否插入空格。
                    #
                    # x_tolerance=2：
                    # 相邻字符水平间距大于 2 个 PDF 坐标单位时插入空格。
                    # 当前 DataMan 手册使用默认容差时会发生英文单词粘连，
                    # 降为 2 后可以恢复正常的英文单词边界。
                    #
                    # y_tolerance=3：
                    # 控制字符是否属于同一行。这里显式保留 pdfplumber
                    # 当前使用的常规纵向容差，避免轻微的字体基线差异拆散行。
                    extracted_text = page.extract_text(
                        x_tolerance=2,
                        y_tolerance=3,
                    )

                    if extracted_text is None:
                        # continue 跳过当前页面，
                        # 继续处理下一页。
                        continue

                    # strip() 删除正文首尾的空格、换行符
                    # 和制表符。
                    cleaned_text = extracted_text.strip()

                    if not cleaned_text:
                        # 只包含空白的页面也不创建 Segment。
                        continue

                    # 每一个有文字的 PDF 页面，
                    # 对应一个 SourceTextSegment。
                    #
                    # page_number 是 PDF 中的物理页面顺序，
                    # 不一定等于页面正文中印刷的页码。
                    segments.append(
                        SourceTextSegment(
                            page_or_section=(
                                f"page: {page_number}"
                            ),
                            content=cleaned_text,
                        )
                    )

        except Exception as exc:
            # pdfplumber 或其底层 pdfminer 可能因为
            # PDF 损坏、加密或格式异常而抛出不同异常。
            #
            # Loader 位于第三方库与业务系统的边界，
            # 因而在这里将它们统一转换成清晰的解析错误。
            #
            # “from exc”保留原始异常作为 __cause__，
            # 方便日志和调试查看真正原因。
            raise ValueError(
                f"PDF 解析失败：{path.name}"
            ) from exc

        # 有些 PDF 可以正常打开，但每一页都是扫描图片。
        # 这种文档不能产生 Embedding 文本，
        # 因此不能假装摄取成功。
        if not segments:
            raise ValueError(
                f"PDF 没有可提取文本：{path.name}"
            )

        return segments


# 在每个章节中提取故障编号。
#
# \b                 表示单词边界；
# ERR-               表示固定前缀；
# [A-Z]+             表示一个或多个大写英文字母；
# -                   表示中间的连字符；
# \d+                 表示一个或多个数字。
ERROR_CODE_PATTERN = re.compile(
    r"\bERR-[A-Z]+-\d+\b"
)

# 识别 Markdown 的 ATX 标题：
#
# # 一级标题
# ## 二级标题
# ### 三级标题
#
# ^ 和 $ 在 re.MULTILINE 模式下匹配每一行的开头和结尾。
#
# group(1) 捕获一个到六个 #，用于计算标题层级。
# group(2) 捕获标题正文。
MARKDOWN_HEADING_PATTERN = re.compile(
    r"^(#{1,6})[ \t]+"
    r"(.+?)"
    r"(?:[ \t]+#+)?"
    r"[ \t]*$",
    re.MULTILINE,
)

# 当前知识库把 Markdown 的三级标题
# 作为主要语义检索单元。
#
# 例如：
#
# ### 10.2 TEST-RCS-001 交叉路口冲突调度
#
# 它下面的四级标题“测试目的、测试步骤、
# 通过标准、失败条件”会组合成同一个 Segment。
MARKDOWN_GROUPED_HEADING_LEVEL = 3

class TxtDocumentLoader:
    """读取 UTF-8 TXT 文件并识别其中的故障章节。"""

    def load(
        self,
        path: Path,
    ) -> list[SourceTextSegment]:
        """把一个 TXT 文件解析成有来源位置的文本段。"""

        # 公共辅助函数负责文件类型、存在性、
        # UTF-8 编码和空文本检查。
        cleaned_text = _read_utf8_text(
            path,
            expected_suffix=".txt",
            loader_name="TxtDocumentLoader",
        )

        if not cleaned_text:
            raise ValueError(
                f"TXT 文件没有可用文本：{path.name}"
            )

        # Pattern.split() 按照故障章节起点切分正文。
        #
        # 因为正则使用了 (?=...)，
        # “故障代码 (ErrorCode)”仍会保留在对应章节中。
        raw_sections = FAULT_SECTION_PATTERN.split(
            cleaned_text
        )

        # 清理每段首尾空白，并过滤空字符串。
        sections = [
            section.strip()
            for section in raw_sections
            if section.strip()
        ]

        segments: list[SourceTextSegment] = []

        # enumerate(..., start=1) 同时提供章节内容和从 1 开始的序号。
        #
        # 序号用于没有故障编号的普通 TXT 文档兜底。
        for section_number, section in enumerate(
            sections,
            start=1,
        ):
            # Pattern.search() 在整段文字中搜索第一个故障编号。
            #
            # 找到时返回 re.Match 对象；
            # 找不到时返回 None。
            code_match = ERROR_CODE_PATTERN.search(
                section
            )

            if code_match is not None:
                # group(0) 取得整个正则匹配结果，
                # 例如 "ERR-NAV-2001"。
                section_name = (
                    f"section: {code_match.group(0)}"
                )
            else:
                # 普通 TXT 可能没有故障编号。
                # 此时仍然可以作为一个通用文本段处理。
                section_name = (
                    f"section: text-{section_number}"
                )

            segments.append(
                SourceTextSegment(
                    page_or_section=section_name,
                    content=section,
                )
            )

        return segments


class MarkdownDocumentLoader:
    """读取 UTF-8 Markdown 并按标题层级产生文本段。"""

    def load(
        self,
        path: Path,
    ) -> list[SourceTextSegment]:
        """把 Markdown 文件解析成带章节路径的文本段。"""

        text = _read_utf8_text(
            path,
            expected_suffix=".md",
            loader_name="MarkdownDocumentLoader",
        )

        # finditer() 查找全部标题，
        # 返回一个可以依次产生 re.Match 的迭代器。
        #
        # list(...) 将迭代器结果保存下来，
        # 因为后面需要同时查看当前标题和下一个标题。
        heading_matches = list(
            MARKDOWN_HEADING_PATTERN.finditer(text)
        )

        # 没有任何标题的 Markdown 仍然是合法文本。
        # 此时把整篇文件作为一个 Segment。
        if not heading_matches:
            return [
                SourceTextSegment(
                    page_or_section="section: document",
                    content=text,
                )
            ]

        segments: list[SourceTextSegment] = []

        # 第一条标题前可能存在前言。
        #
        # match.start() 返回标题匹配在完整字符串中的
        # 起始字符位置。
        preamble = text[
            :heading_matches[0].start()
        ].strip()

        if preamble:
            segments.append(
                SourceTextSegment(
                    page_or_section="section: preamble",
                    content=preamble,
                )
            )

        # heading_stack 保存当前标题层级路径。
        #
        # 例如处理到 ### 3.2 时，可能保存：
        # [
        #     "测试规程",
        #     "3. 系统边界",
        #     "3.2 系统组成",
        # ]
        heading_stack: list[str] = []

        for index, heading_match in enumerate(
            heading_matches
        ):
            # group(1) 是井号，例如 "###"。
            # 井号数量就是标题级别。
            level = len(
                heading_match.group(1)
            )

            # group(2) 是不包含井号的标题正文。
            title = heading_match.group(2).strip()

            # 当前标题为 level 级时，只保留它的父级标题。
            #
            # 例如原来是：
            # [一级, 二级, 三级]
            #
            # 遇到新的二级标题时：
            # heading_stack[:1] -> [一级]
            heading_stack = heading_stack[
                :level - 1
            ]

            # 将当前标题加入层级路径。
            heading_stack.append(title)

            # join() 使用 " > " 连接标题路径。
            section_path = " > ".join(
                heading_stack
            )

            # 当前语义单元从当前标题行开始。
            section_start = heading_match.start()

            # 检查当前四级或更深标题，
            # 是否已经包含在前面的三级语义单元中。
            #
            # reversed(...) 从当前位置向前查找最近的
            # 一级、二级或三级标题。
            nearest_group_boundary_level: int | None = None

            for previous_match in reversed(
                heading_matches[:index]
            ):
                previous_level = len(
                    previous_match.group(1)
                )

                if (
                    previous_level
                    <= MARKDOWN_GROUPED_HEADING_LEVEL
                ):
                    nearest_group_boundary_level = (
                        previous_level
                    )

                    # 已经找到距离当前标题最近的
                    # 一级、二级或三级边界，无需继续向前搜索。
                    break

            if (
                level > MARKDOWN_GROUPED_HEADING_LEVEL
                and nearest_group_boundary_level
                == MARKDOWN_GROUPED_HEADING_LEVEL
            ):
                # 当前四级或更深标题已经包含在
                # 最近的三级语义单元中。
                #
                # continue 跳过当前标题，
                # 继续处理下一个 heading_match。
                continue

            if level == MARKDOWN_GROUPED_HEADING_LEVEL:
                # 三级标题需要包含其所有四级及更深子章节。
                #
                # 默认一直延伸到文档末尾。
                section_end = len(text)

                # 找到下一个同级或更高级标题，
                # 作为当前三级语义单元的结束位置。
                for following_match in heading_matches[
                    index + 1:
                ]:
                    following_level = len(
                        following_match.group(1)
                    )

                    if following_level <= level:
                        section_end = (
                            following_match.start()
                        )
                        break
            else:
                # 一级、二级以及没有归入三级单元的标题，
                # 只读取它们自己的直接正文。
                if index + 1 < len(heading_matches):
                    section_end = heading_matches[
                        index + 1
                    ].start()
                else:
                    section_end = len(text)

            # 取得当前标题之后的正文。
            #
            # 对三级标题而言，这里也包含它下面的
            # 四级标题和四级标题正文。
            section_body = text[
                heading_match.end():section_end
            ].strip()

            # 纯结构标题仍保留在 heading_stack 中，
            # 但不单独创建可向量化 Segment。
            if not section_body:
                continue

            # 保留当前标题行以及完整正文。
            section_content = text[
                section_start:section_end
            ].strip()

            segments.append(
                SourceTextSegment(
                    page_or_section=(
                        f"section: {section_path}"
                    ),
                    content=section_content,
                )
            )

        return segments