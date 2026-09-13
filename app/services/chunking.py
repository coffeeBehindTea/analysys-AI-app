"""把带来源位置的长文本切分成可追溯 DocumentChunk。"""

# sha256() 创建 SHA-256 哈希计算器。
from hashlib import sha256

from app.schemas.retrieval import (
    DocumentChunk,
    SourceTextSegment,
)


class TextChunker:
    """使用固定字符窗口和重叠区域切分文本。"""

    def __init__(
        self,
        chunk_size: int,
        overlap: int,
    ) -> None:
        """检查并保存切分参数。"""

        # DocumentChunk.content 当前最多允许 20_000 个字符，
        # 因此切分器不能配置出超过数据契约的 Chunk。
        if chunk_size < 1 or chunk_size > 20_000:
            raise ValueError(
                "chunk_size 必须在 1 到 20000 之间"
            )

        # overlap 表示相邻 Chunk 重复保留多少个字符。
        # 它可以为 0，但不能是负数。
        if overlap < 0:
            raise ValueError(
                "overlap 不能小于 0"
            )
        # overlap 必须小于 chunk_size。
        #
        # 如果二者相等，例如：
        # chunk_size=100，overlap=100
        #
        # 那么下一轮的起点不会向前移动，
        # while 循环将永远无法结束。
        if overlap >= chunk_size:
            raise ValueError(
                "overlap 必须小于 chunk_size"
            )

        self._chunk_size = chunk_size
        self._overlap = overlap

    def split_document(
        self,
        *,
        document_id: str,
        source_file: str,
        segments: list[SourceTextSegment],
    ) -> list[DocumentChunk]:
        """把一份文档的所有页面或章节切成全局有序 Chunk。"""

        # 一份没有任何有效页面或章节的文档不能入库。
        #
        # 后续完整摄取管道会将这个错误转换成明确的上传错误。
        if not segments:
            raise ValueError(
                "待切分文档至少需要一个文本段"
            )

        chunks: list[DocumentChunk] = []

        # chunk_index 是整份文档范围内的全局顺序，
        # 不能在每一页或每个章节重新从 0 开始。
        chunk_index = 0

        for segment in segments:
            # SourceTextSegment 已由 Pydantic 清理首尾空白。
            # 此处再次 strip() 可以使切分器不依赖调用方
            # 是否一定通过 Pydantic 创建了规范对象。
            text = segment.content.strip()

            # 当前窗口在这个页面或章节中的起始字符位置。
            start = 0

            while start < len(text):
                # 当前窗口的理想结束位置是：
                # start + chunk_size
                #
                # min() 防止最后一个窗口超出文本末尾。
                end = min(
                    start + self._chunk_size,
                    len(text),
                )

                # 使用字符串切片取得当前窗口。
                content = text[start:end].strip()

                # 如果窗口只有空白，strip() 后可能为空。
                # 空 Chunk 不应该进入 Embedding 和向量数据库。
                if content:
                    # str.encode("utf-8") 将 Python 字符串转换成字节。
                    #
                    # sha256(...) 计算这些字节的 SHA-256。
                    #
                    # hexdigest() 将摘要转换成长度为 64 的
                    # 小写十六进制字符串。
                    content_hash = sha256(
                        content.encode("utf-8")
                    ).hexdigest()

                    # :06d 表示把整数格式化成至少六位十进制，
                    # 不足的部分在左侧补 0。
                    #
                    # 例如：
                    # 0  -> 000000
                    # 12 -> 000012
                    #
                    # 这样 Chunk ID 按字符串排序时，
                    # 仍然与数字顺序保持一致。
                    chunk_id = (
                        f"{document_id}:{chunk_index:06d}"
                    )

                    chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_id,
                            document_id=document_id,
                            source_file=source_file,
                            page_or_section=(
                                segment.page_or_section
                            ),
                            chunk_index=chunk_index,
                            content_hash=content_hash,
                            content=content,
                        )
                    )

                    # 只有实际产生 Chunk 时才增加索引。
                    chunk_index += 1

                # 已经到达文本结尾，不再计算下一窗口。
                if end == len(text):
                    break

                # 下一窗口向前移动：
                #
                # chunk_size - overlap
                #
                # 例如 chunk_size=450、overlap=80：
                # 第一段范围：0～450
                # 第二段起点：450-80=370
                #
                # 第 370～450 的 80 个字符会同时出现在两段中。
                start = end - self._overlap

        return chunks