"""把 PDF、Markdown、TXT 真实摄取到持久化 ChromaDB。"""

# argparse 用于解析命令行参数。
import argparse

# asyncio.run() 从同步命令行入口运行异步摄取流程。
import asyncio

# dataclass 用于定义脚本内部使用的轻量结果对象。
from dataclasses import dataclass

from openai import AsyncOpenAI

# datetime 记录日志生成时间；
# timezone.utc 表示 UTC 时区。
from datetime import datetime, timezone

from pathlib import Path

from typing import Literal, Protocol

from app.config import Settings, get_settings
from app.errors import (
    ApplicationError,
    DuplicateDocumentError,
)
from app.schemas.knowledge import DocumentRecord
from app.services.chunking import TextChunker
from app.services.document_preparation import (
    TextDocumentPreparationService,
)
from app.services.document_service import (
    DocumentService,
    SUPPORTED_FILE_TYPES,
)
from app.services.embedding import EmbeddingService
from app.services.embedding_client import (
    create_embedding_client,
    get_embedding_model,
)
from app.services.vector_store import (
    ChromaVectorStore,
)


# 一份文件在本次批量摄取中的三种结果。
IngestionStatus = Literal[
    "ingested",
    "skipped",
    "failed",
]


class DocumentIngester(Protocol):
    """摄取脚本依赖的最小文档服务接口。"""

    async def ingest_document(
        self,
        path: Path,
    ) -> DocumentRecord:
        """摄取一份文件并返回文档记录。"""

        ...


@dataclass(
    frozen=True,
    slots=True,
)
class IngestionLogEntry:
    """一份文件的摄取日志记录。"""

    # 输入文件名。
    source_file: str

    # ingested：本次成功写入；
    # skipped：已经存在，跳过；
    # failed：校验、Embedding 或存储失败。
    status: IngestionStatus

    # 成功时保存完整 DocumentRecord；
    # 跳过或失败时为 None。
    record: DocumentRecord | None

    # 面向开发者的简短结果说明。
    message: str


def parse_args() -> argparse.Namespace:
    """解析真实摄取脚本的命令行参数。"""

    parser = argparse.ArgumentParser(
        description=(
            "将 PDF、Markdown 和 TXT 文档"
            "摄取到本地 ChromaDB"
        )
    )

    # nargs="*" 表示可以传入零个或多个文件路径。
    #
    # 没有显式文件时，扫描 --source-dir；
    # 提供文件时，只摄取指定文件。
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help=(
            "可选的文件列表；省略时扫描整个 source-dir"
        ),
    )

    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/source"),
        help="没有指定 files 时扫描的语料目录",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/ingestion-log.md"),
        help="摄取日志输出位置",
    )

    return parser.parse_args()


def find_source_files(
    source_dir: Path,
) -> list[Path]:
    """找出目录中支持摄取的文件并稳定排序。"""

    if not source_dir.is_dir():
        raise NotADirectoryError(
            f"语料目录不存在：{source_dir}"
        )

    source_files = [
        path
        for path in source_dir.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_FILE_TYPES
        )
    ]

    # casefold()用于忽略英文大小写进行排序。
    #
    # 稳定排序有助于让每次日志顺序一致，
    # 也使失败后重新执行更容易观察。
    return sorted(
        source_files,
        key=lambda path: path.name.casefold(),
    )


def select_source_files(
    *,
    explicit_files: list[Path],
    source_dir: Path,
) -> list[Path]:
    """选择显式文件，或者扫描默认语料目录。"""

    if explicit_files:
        # list(...) 创建副本，
        # 避免后续代码修改 argparse 保存的原列表。
        return list(explicit_files)

    return find_source_files(source_dir)


def create_document_service(
    settings: Settings,
) -> tuple[
    DocumentService,
    AsyncOpenAI,
]:
    """根据统一配置创建真实文档服务和 Embedding 客户端。"""

    # 取得经过非空校验的模型名称。
    embedding_model = get_embedding_model(
        settings
    )

    # 创建 AsyncOpenAI 兼容客户端。
    #
    # 此时只创建客户端对象；
    # 真正网络请求发生在 embed_texts()。
    client = create_embedding_client(settings)

    embedding_service = EmbeddingService(
        client=client,
        model=embedding_model,
    )

    preparation_service = (
        TextDocumentPreparationService(
            chunker=TextChunker(
                chunk_size=settings.rag_chunk_size,
                overlap=settings.rag_chunk_overlap,
            )
        )
    )

    vector_store = ChromaVectorStore(
        persist_directory=(
            settings.chroma_persist_directory
        ),
        collection_name=(
            settings.chroma_collection_name
        ),
        embedding_model=embedding_model,
    )

    document_service = DocumentService(
        preparation_service=preparation_service,
        embedding_provider=embedding_service,
        vector_store=vector_store,
        embedding_model=embedding_model,
        batch_size=settings.embedding_batch_size,
        max_file_size_bytes=(
            settings.max_document_size_bytes
        ),
    )

    # 返回 client 是为了让调用方最终关闭其 HTTP 连接池。
    return document_service, client


async def ingest_files(
    *,
    service: DocumentIngester,
    source_files: list[Path],
) -> list[IngestionLogEntry]:
    """依次摄取文件，并区分成功、重复和失败。"""

    entries: list[IngestionLogEntry] = []

    for path in source_files:
        try:
            record = await service.ingest_document(
                path
            )
        except DuplicateDocumentError as exc:
            # 重复文件是预期业务结果，
            # 不应让整批摄取停止。
            entries.append(
                IngestionLogEntry(
                    source_file=path.name,
                    status="skipped",
                    record=None,
                    message=str(exc),
                )
            )
        except ApplicationError as exc:
            # 解析、Embedding 和向量库的预期异常
            # 会记录为失败，然后继续下一份文件。
            entries.append(
                IngestionLogEntry(
                    source_file=path.name,
                    status="failed",
                    record=None,
                    message=str(exc),
                )
            )
        else:
            entries.append(
                IngestionLogEntry(
                    source_file=path.name,
                    status="ingested",
                    record=record,
                    message="摄取成功",
                )
            )

    return entries


def escape_markdown_cell(
    value: str,
) -> str:
    """清理会破坏 Markdown 表格结构的字符。"""

    # 换行会破坏当前表格行，所以替换为空格。
    compacted = " ".join(value.split())

    # 竖线是 Markdown 表格的列分隔符，
    # 必须使用反斜杠转义。
    return compacted.replace(
        "|",
        r"\|",
    )


def build_ingestion_report(
    *,
    entries: list[IngestionLogEntry],
    settings: Settings,
    embedding_model: str,
    generated_at: str,
) -> str:
    """把摄取结果转换成不含正文和密钥的 Markdown 日志。"""

    ingested_count = sum(
        entry.status == "ingested"
        for entry in entries
    )
    skipped_count = sum(
        entry.status == "skipped"
        for entry in entries
    )
    failed_count = sum(
        entry.status == "failed"
        for entry in entries
    )

    lines: list[str] = [
        "# Day 3–4 文档摄取日志",
        "",
        f"- 生成时间：`{generated_at}`",
        f"- Embedding 模型：`{embedding_model}`",
        (
            "- Chroma Collection："
            f"`{settings.chroma_collection_name}`"
        ),
        (
            "- Chroma 持久化目录："
            f"`{settings.chroma_persist_directory}`"
        ),
        (
            "- Chunk 参数："
            f"`{settings.rag_chunk_size}/"
            f"{settings.rag_chunk_overlap}`"
        ),
        (
            "- 文件总数："
            f"{len(entries)}；"
            f"成功 {ingested_count}；"
            f"跳过 {skipped_count}；"
            f"失败 {failed_count}"
        ),
        "",
        "## 逐文件结果",
        "",
        (
            "| 文件 | 结果 | Document ID |"
            " 位置数 | Chunk 数 | 说明 |"
        ),
        "|---|---|---|---:|---:|---|",
    ]

    for entry in entries:
        record = entry.record

        if record is None:
            document_id = "-"
            location_count = "-"
            chunk_count = "-"
        else:
            document_id = record.document_id
            location_count = str(
                record.page_or_section_count
            )
            chunk_count = str(
                record.chunk_count
            )

        lines.append(
            "| "
            f"{escape_markdown_cell(entry.source_file)}"
            " | "
            f"{entry.status}"
            " | "
            f"`{document_id}`"
            " | "
            f"{location_count}"
            " | "
            f"{chunk_count}"
            " | "
            f"{escape_markdown_cell(entry.message)}"
            " |"
        )

    lines.extend(
        [
            "",
            "## 说明",
            "",
            (
                "- `ingested`：本次完成解析、切分、"
                "Embedding 和 Chroma 写入。"
            ),
            (
                "- `skipped`：相同文件内容已经存在，"
                "未再次调用 Embedding。"
            ),
            (
                "- `failed`：该文件未完成摄取，"
                "具体原因见表格。"
            ),
            (
                "- 日志只记录文件级统计和标识，"
                "不记录 Chunk 正文、向量或 API Key。"
            ),
            "",
        ]
    )

    return "\n".join(lines)


async def run_ingestion(
    args: argparse.Namespace,
) -> str:
    """创建真实依赖并执行整批摄取。"""

    source_files = select_source_files(
        explicit_files=args.files,
        source_dir=args.source_dir,
    )

    if not source_files:
        raise ValueError(
            "没有找到可摄取的 PDF、Markdown 或 TXT"
        )

    settings = get_settings()
    embedding_model = get_embedding_model(
        settings
    )

    service, client = create_document_service(
        settings
    )

    try:
        entries = await ingest_files(
            service=service,
            source_files=source_files,
        )
    finally:
        # create_embedding_client()返回 AsyncOpenAI。
        #
        # close()是异步方法，用来关闭其底层 HTTP 连接池。
        await client.close()

    generated_at = datetime.now(
        timezone.utc
    ).astimezone().isoformat(
        timespec="seconds"
    )

    return build_ingestion_report(
        entries=entries,
        settings=settings,
        embedding_model=embedding_model,
        generated_at=generated_at,
    )


def main() -> None:
    """同步命令行入口。"""

    args = parse_args()

    # asyncio.run()创建事件循环，
    # 执行异步摄取，结束后关闭事件循环。
    report = asyncio.run(
        run_ingestion(args)
    )

    # parents=True：
    # 如果上级 docs 目录不存在则一起创建。
    #
    # exist_ok=True：
    # 目录已经存在时不报错。
    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        report,
        encoding="utf-8",
    )

    print(
        f"摄取日志已生成：{args.output}"
    )


if __name__ == "__main__":
    main()