"""将 HTTP 上传文件安全地暂存为本地临时文件。"""

# AsyncIterator[T] 表示异步迭代过程中会产生 T。
#
# stage_uploaded_file() 会通过 yield 暂时产生一个 Path，
# 因此它的异步生成器类型是 AsyncIterator[Path]。
from collections.abc import AsyncIterator

# asynccontextmanager 可以把“包含一次 yield 的异步生成器”
# 转换成可以通过 async with 使用的异步上下文管理器。
from contextlib import asynccontextmanager

from pathlib import Path

# TemporaryDirectory 创建临时目录，
# 离开 with 代码块时会自动删除目录及其内容。
from tempfile import TemporaryDirectory

from typing import Protocol

from app.errors import DocumentValidationError


# 每次从上传流读取 1 MiB。
#
# 分块读取可以避免一次性把整个 PDF 放进内存。
DEFAULT_UPLOAD_READ_SIZE_BYTES = 1024 * 1024


class AsyncUploadedFile(Protocol):
    """暂存服务要求上传对象满足的数据契约。

    FastAPI 的 UploadFile 已经具有这些属性和方法，
    因而不需要显式继承本 Protocol。
    """

    # filename 来自 multipart 请求中的原始文件名。
    #
    # 客户端可能不提供文件名，因此类型包含 None。
    filename: str | None

    async def read(
        self,
        size: int = -1,
    ) -> bytes:
        """异步读取最多 size 字节。"""

        ...

    async def close(self) -> None:
        """关闭上传文件持有的底层资源。"""

        ...


def sanitize_upload_filename(
    filename: str | None,
) -> str:
    """提取安全文件名，防止客户端指定服务器路径。"""

    # filename 是不可信的客户端输入。
    #
    # 将 Windows 的反斜杠统一成正斜杠后，
    # 无论客户端提交哪种路径分隔符，都只保留最后一段。
    normalized_name = (
        filename or ""
    ).replace("\\", "/")

    # rsplit("/", 1) 最多从右侧分割一次。
    #
    # [-1] 取得路径最后一部分：
    # "../../manual.txt" -> "manual.txt"
    safe_name = normalized_name.rsplit(
        "/",
        1,
    )[-1].strip()

    if (
        not safe_name
        or safe_name in {".", ".."}
    ):
        raise DocumentValidationError(
            "上传文件缺少有效文件名"
        )

    # 这些字符不能安全地用于 Windows 文件名。
    #
    # 路径分隔符已经在上面处理，
    # 这里继续拒绝其他特殊字符。
    if any(
        character in safe_name
        for character in '<>:"|?*\x00'
    ):
        raise DocumentValidationError(
            "上传文件名包含不允许的字符"
        )

    return safe_name


@asynccontextmanager
async def stage_uploaded_file(
    upload: AsyncUploadedFile,
    *,
    max_file_size_bytes: int,
    read_size_bytes: int = (
        DEFAULT_UPLOAD_READ_SIZE_BYTES
    ),
) -> AsyncIterator[Path]:
    """将上传内容暂存为文件，并在使用结束后自动清理。"""

    try:
        # 这些参数来自应用配置，但这个函数也要保护自己的边界，
        # 避免它被其他调用方传入无效参数。
        if max_file_size_bytes < 1:
            raise ValueError(
                "max_file_size_bytes 必须大于 0"
            )

        if read_size_bytes < 1:
            raise ValueError(
                "read_size_bytes 必须大于 0"
            )

        safe_name = sanitize_upload_filename(
            upload.filename
        )

        # TemporaryDirectory() 返回临时目录的字符串路径。
        #
        # 离开此 with 后，目录及暂存文件都会自动删除。
        with TemporaryDirectory(
            prefix="knowledge-upload-",
        ) as temporary_directory:
            staged_path = (
                Path(temporary_directory)
                / safe_name
            )

            total_size_bytes = 0

            # w 表示创建并写入文件；
            # b 表示按原始字节写入，而不是按文本编码写入。
            with staged_path.open("wb") as output:
                while True:
                    # UploadFile.read() 是异步方法，
                    # 所以必须使用 await。
                    data = await upload.read(
                        read_size_bytes
                    )

                    # read() 返回 b"" 表示已经到达文件末尾。
                    if not data:
                        break

                    total_size_bytes += len(data)

                    # 在继续写入之前检查总大小，
                    # 避免恶意客户端无限上传并占满磁盘。
                    if (
                        total_size_bytes
                        > max_file_size_bytes
                    ):
                        raise DocumentValidationError(
                            "上传文档超过大小限制；"
                            f"最大允许 "
                            f"{max_file_size_bytes} 字节"
                        )

                    # write() 返回写入的字节数，
                    # 当前不需要使用这个返回值。
                    output.write(data)

            # 暂存文件已经关闭，可以安全交给
            # DocumentService、pdfplumber 等组件读取。
            #
            # 调用方离开 async with 后，
            # 本函数会从这里继续运行并清理临时目录。
            yield staged_path

    finally:
        # 无论文件名校验失败、超出大小限制、
        # DocumentService 失败还是正常完成，
        # 都关闭 FastAPI UploadFile 的底层资源。
        await upload.close()