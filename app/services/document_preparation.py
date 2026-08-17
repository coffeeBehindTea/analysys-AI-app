"""编排文档加载、文件标识计算和文本切分。"""

# sha256 用于计算整份原始文件的内容指纹。
from hashlib import sha256

# Path 表示跨平台文件路径。
from pathlib import Path

from app.schemas.retrieval import DocumentChunk
from app.services.chunking import TextChunker
from app.services.document_loader import (
    MarkdownDocumentLoader,
    PdfDocumentLoader,
    TxtDocumentLoader,
)

def calculate_file_sha256(
    path: Path,
    *,
    block_size: int = 1_048_576,
) -> str:
    """分块读取文件并计算整份文件的 SHA-256。"""

    # block_size 是每次从磁盘读取的字节数。
    #
    # 1_048_576 字节等于 1 MiB。
    # Python 允许使用下划线提高大整数的可读性，
    # 实际数值仍然是 1048576。
    if block_size < 1:
        raise ValueError(
            "文件哈希读取块大小必须大于 0"
        )

    if not path.is_file():
        raise FileNotFoundError(
            f"待计算哈希的文件不存在：{path}"
        )

    # sha256() 不传入初始数据时，
    # 返回一个可以持续接收文件内容的哈希计算器。
    digest = sha256()

    # open("rb") 以二进制只读模式打开文件：
    #
    # r = read，只读；
    # b = binary，读取原始字节。
    #
    # 哈希必须根据原始文件字节计算，
    # 不能根据经过换行转换或字符解码后的字符串计算。
    #
    # with 是上下文管理器。
    # 离开代码块时会自动关闭文件，即使中途发生异常。
    with path.open("rb") as file:
        while True:
            # read(block_size) 最多读取指定数量的字节。
            block = file.read(block_size)

            # 读到文件末尾时，read() 返回 b""。
            # 空 bytes 在 if 中会被判断为 False。
            if not block:
                break

            # update() 把当前字节块加入哈希计算。
            #
            # 多次 update() 的最终效果，
            # 等同于一次传入整份文件内容。
            digest.update(block)

    # hexdigest() 返回 64 位小写十六进制摘要。
    return digest.hexdigest()


class TextDocumentPreparationService:
    """把 PDF、TXT、Markdown 准备成可执行 Embedding 的 Chunk。"""

    def __init__(
        self,
        chunker: TextChunker,
    ) -> None:
        """保存切分器并创建无状态文本 Loader。"""

        # chunker 由外部传入，
        # 因为不同基线实验需要使用不同切分参数。
        self._chunker = chunker

        # 每个 Loader 只负责一种文件格式。
        self._pdf_loader = PdfDocumentLoader()
        self._txt_loader = TxtDocumentLoader()
        self._markdown_loader = MarkdownDocumentLoader()

    def prepare(
        self,
        path: Path,
    ) -> list[DocumentChunk]:
        """加载一份文档并生成带完整元数据的 Chunk。"""

        # 大写扩展名也能被识别。
        suffix = path.suffix.lower()

        # 根据文件扩展名，把文件分配给相应 Loader。
        #
        # 三种 Loader 最终都返回：
        # list[SourceTextSegment]
        #
        # 因而下游 Chunker 不需要了解原始文件格式。
        if suffix == ".pdf":
            segments = self._pdf_loader.load(path)
        elif suffix == ".txt":
            segments = self._txt_loader.load(path)
        elif suffix == ".md":
            segments = self._markdown_loader.load(path)
        else:
            raise ValueError(
                "文档预处理只支持 .pdf、.txt 和 .md 文件"
            )

        # 整份原始文件的 SHA-256 作为 document_id。
        #
        # 内容完全相同的文件即使名称不同，
        # 也会得到相同 document_id。
        document_id = calculate_file_sha256(path)

        # path.name 只取得文件名，不包含服务器绝对路径。
        #
        # 例如：
        # D:\data\manual.txt
        # 只保存为：
        # manual.txt
        return self._chunker.split_document(
            document_id=document_id,
            source_file=path.name,
            segments=segments,
        )