"""比较两套 Chunk 切分参数在真实文本语料上的效果。"""

# argparse 是 Python 标准库的命令行参数解析模块。
import argparse

# Path 负责目录遍历、目录创建和报告文件写入。
from pathlib import Path

from app.schemas.retrieval import DocumentChunk
from app.services.chunking import TextChunker
from app.services.document_preparation import (
    TextDocumentPreparationService,
)


# Day 1-2 要比较的两套参数。
#
# 每个元组依次保存：
# 显示名称、chunk_size、overlap。
CHUNKING_CONFIGURATIONS = [
    ("450/80", 450, 80),
    ("800/120", 800, 120),
]


# 当前只处理已经实现 Loader 的文件类型。
SUPPORTED_SUFFIXES = {
    ".txt",
    ".md",
}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    # ArgumentParser 负责读取命令行并生成帮助信息。
    parser = argparse.ArgumentParser(
        description=(
            "比较 450/80 与 800/120 两套文本切分参数"
        )
    )

    # type=Path 表示把命令行字符串自动转换成 Path。
    #
    # default 是用户没有传入参数时使用的默认值。
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/source"),
        help="原始语料目录",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/chunking-comparison.md"
        ),
        help="Markdown 对比报告输出位置",
    )

    # parse_args() 读取当前进程收到的命令行参数，
    # 返回 argparse.Namespace 对象。
    return parser.parse_args()


def find_supported_files(
    source_dir: Path,
) -> list[Path]:
    """按文件名排序取得目录中的 TXT 和 Markdown。"""

    if not source_dir.is_dir():
        raise NotADirectoryError(
            f"语料目录不存在：{source_dir}"
        )

    # iterdir() 遍历当前目录的直接子项，
    # 不会递归进入子目录。
    #
    # 生成器中的两个条件分别确保：
    # 1. 当前路径是文件；
    # 2. 扩展名已经有对应 Loader。
    source_files = (
        path
        for path in source_dir.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_SUFFIXES
        )
    )

    # sorted() 创建排序后的列表。
    # 避免大小写影响报告顺序。
    return sorted(
        source_files,
        key=lambda path: path.name.lower(),
    )


def compact_preview(
    content: str,
    *,
    max_length: int = 480,
) -> str:
    """生成明确区分开头、结尾和省略部分的 Chunk 预览。"""

    # 把换行、制表符和连续空格压缩成单个空格。
    compacted = " ".join(
        content.split()
    )

    if max_length < 20:
        raise ValueError(
            "预览最大长度不能小于 20"
        )

    # 内容没有超过限制时，明确说明展示的是完整内容。
    if len(compacted) <= max_length:
        escaped_content = compacted.replace(
            "|",
            r"\|",
        )

        return (
            "**完整内容：** "
            f"{escaped_content}"
        )

    # 内容过长时，将可展示的正文长度平均分给开头和结尾。
    head_length = max_length // 2
    tail_length = max_length - head_length

    head = compacted[:head_length]
    tail = compacted[-tail_length:]

    # 计算中间有多少字符没有展示。
    omitted_count = (
        len(compacted)
        - head_length
        - tail_length
    )

    # 分别转义开头和结尾中的 Markdown 表格分隔符。
    escaped_head = head.replace(
        "|",
        r"\|",
    )
    escaped_tail = tail.replace(
        "|",
        r"\|",
    )

    # 使用明确标签，防止读者误认为两段内容原本连续。
    return (
        f"**开头：** {escaped_head} "
        f"**——中间省略 {omitted_count} 个字符——** "
        f"**结尾：** {escaped_tail}"
    )


def build_report(
    source_dir: Path,
) -> str:
    """运行两套切分参数并返回 Markdown 报告文本。"""

    source_files = find_supported_files(
        source_dir
    )

    if not source_files:
        raise ValueError(
            "语料目录中没有可处理的 TXT 或 Markdown"
        )

    # 保存 Markdown 的每一行，
    # 最后使用换行符一次性连接。
    lines: list[str] = [
        "# Chunk 切分参数对比",
        "",
        "本报告由 `scripts.compare_chunking` 自动生成。",
        "",
        "当前只统计已经实现 Loader 的 TXT 和 Markdown；"
        "PDF 将在 PDF Loader 完成后加入。",
        "",
        "## 汇总",
        "",
        (
            "| 参数（chunk_size/overlap）"
            " | 文件 | Chunk 数 | 最短字符数"
            " | 最长字符数 | 平均字符数 |"
        ),
        "|---|---|---:|---:|---:|---:|",
    ]

    # 保存每套配置产生的 Chunk，
    # 后面用于输出具体边界预览。
    prepared_chunks: dict[
        tuple[str, str],
        list[DocumentChunk],
    ] = {}

    total_counts: dict[str, int] = {}

    for (
        configuration_name,
        chunk_size,
        overlap,
    ) in CHUNKING_CONFIGURATIONS:
        chunker = TextChunker(
            chunk_size=chunk_size,
            overlap=overlap,
        )

        preparation_service = (
            TextDocumentPreparationService(
                chunker=chunker
            )
        )

        total_count = 0

        for source_file in source_files:
            chunks = preparation_service.prepare(
                source_file
            )

            prepared_chunks[
                (
                    configuration_name,
                    source_file.name,
                )
            ] = chunks

            total_count += len(chunks)

            # 每个 Chunk 的正文字符数。
            content_lengths = [
                len(chunk.content)
                for chunk in chunks
            ]

            shortest = min(content_lengths)
            longest = max(content_lengths)

            # 当前不需要 statistics 库，
            # 总长度除以数量即可得到平均值。
            average = (
                sum(content_lengths)
                / len(content_lengths)
            )

            lines.append(
                "| "
                f"{configuration_name} | "
                f"{source_file.name} | "
                f"{len(chunks)} | "
                f"{shortest} | "
                f"{longest} | "
                f"{average:.1f} |"
            )

        total_counts[
            configuration_name
        ] = total_count

    lines.extend(
        [
            "",
            "## 总 Chunk 数",
            "",
            "| 参数 | 总 Chunk 数 |",
            "|---|---:|",
        ]
    )

    for (
        configuration_name,
        _,
        _,
    ) in CHUNKING_CONFIGURATIONS:
        lines.append(
            f"| {configuration_name} | "
            f"{total_counts[configuration_name]} |"
        )

    lines.extend(
        [
            "",
            "## Chunk 边界预览",
            "",
            "每份文档展示前三个 Chunk。",
            "",
        ]
    )

    for (
        configuration_name,
        _,
        _,
    ) in CHUNKING_CONFIGURATIONS:
        lines.append(
            f"### 参数 {configuration_name}"
        )
        lines.append("")

        for source_file in source_files:
            lines.append(
                f"#### {source_file.name}"
            )
            lines.append("")

            chunks = prepared_chunks[
                (
                    configuration_name,
                    source_file.name,
                )
            ]

            # 优先展示最长的 Chunk，
            # 更容易观察 chunk_size 对边界产生的实际影响。
            preview_chunks = sorted(
                chunks,
                key=lambda chunk: (
                    -len(chunk.content),
                    chunk.chunk_index,
                ),
            )[:3]

            for chunk in preview_chunks:
                preview = compact_preview(
                    chunk.content
                )

                lines.append(
                    "- "
                    f"`{chunk.chunk_id}`；"
                    f"位置：`{chunk.page_or_section}`；"
                    f"字符数：{len(chunk.content)}；"
                    f"预览：{preview}"
                )

            lines.append("")

    lines.extend(
        [
            "## 人工观察",
            "",
            "- 450/80 的边界是否更容易切断完整操作步骤：待填写。",
            "- 800/120 是否混入多个不同主题：待填写。",
            "- 哪套参数更适合当前语料：需要结合 Top-3 召回结果判断。",
            "",
        ]
    )

    # join() 把所有行组合成一个 Markdown 字符串。
    return "\n".join(lines)


def main() -> None:
    """执行对比并把报告写入指定位置。"""

    args = parse_args()

    report = build_report(
        args.source_dir
    )

    # output.parent 是输出文件所在的目录。
    #
    # parents=True：
    # 必要时创建多层父目录。
    #
    # exist_ok=True：
    # 目录已经存在时不报错。
    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # write_text() 会创建或覆盖报告文件。
    args.output.write_text(
        report,
        encoding="utf-8",
    )

    print(
        f"切分对比报告已生成：{args.output}"
    )


# 只有直接运行当前模块时才执行 main()。
#
# 当测试代码 import build_report 时，
# 不会自动读取真实语料或写报告。
if __name__ == "__main__":
    main()