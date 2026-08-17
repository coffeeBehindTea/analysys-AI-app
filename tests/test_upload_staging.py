"""上传文件暂存服务的离线测试。"""

from pathlib import Path

import pytest

from app.errors import DocumentValidationError
from app.services.upload_staging import (
    stage_uploaded_file,
)


class FakeUploadedFile:
    """模拟 FastAPI UploadFile 的最小行为。"""

    def __init__(
        self,
        *,
        filename: str | None,
        content: bytes,
    ) -> None:
        """保存文件名、内容和当前读取位置。"""

        self.filename = filename
        self._content = content
        self._position = 0
        self.closed = False

    async def read(
        self,
        size: int = -1,
    ) -> bytes:
        """从当前位置读取最多 size 字节。"""

        if self.closed:
            raise RuntimeError(
                "不能读取已经关闭的上传文件"
            )

        # 已经读到末尾时返回空字节串。
        if self._position >= len(self._content):
            return b""

        if size < 0:
            end_position = len(self._content)
        else:
            end_position = min(
                self._position + size,
                len(self._content),
            )

        data = self._content[
            self._position:end_position
        ]

        self._position = end_position
        return data

    async def close(self) -> None:
        """记录上传资源已经被关闭。"""

        self.closed = True


@pytest.mark.asyncio
async def test_stage_uploaded_file_preserves_content(
) -> None:
    """暂存服务应保存内容、清理路径并关闭上传资源。"""

    upload = FakeUploadedFile(
        # 故意使用带路径的客户端文件名，
        # 验证服务器只保留最后的文件名。
        filename=r"..\..\manual.txt",
        content=b"robot safety manual",
    )

    staged_path: Path | None = None

    async with stage_uploaded_file(
        upload,
        max_file_size_bytes=1024,
        # 使用较小的读取块，验证循环分块读取。
        read_size_bytes=5,
    ) as current_path:
        staged_path = current_path

        assert current_path.exists()
        assert current_path.name == "manual.txt"
        assert current_path.read_bytes() == (
            b"robot safety manual"
        )

        # 在 async with 内，UploadFile 尚未关闭。
        assert upload.closed is False

    # 离开上下文后，TemporaryDirectory
    # 应已经删除暂存文件。
    assert staged_path is not None
    assert staged_path.exists() is False

    # finally 应关闭原始上传资源。
    assert upload.closed is True


@pytest.mark.asyncio
async def test_stage_uploaded_file_rejects_oversize(
) -> None:
    """上传总字节数超过限制时应抛出文档校验异常。"""

    upload = FakeUploadedFile(
        filename="large.txt",
        content=b"123456",
    )

    with pytest.raises(
        DocumentValidationError,
        match="超过大小限制",
    ):
        async with stage_uploaded_file(
            upload,
            max_file_size_bytes=5,
            read_size_bytes=2,
        ):
            # 超限异常发生在 yield 之前，
            # 因此正常情况下不会进入这里。
            pytest.fail(
                "超限文件不应进入上下文代码块"
            )

    assert upload.closed is True


@pytest.mark.asyncio
async def test_stage_uploaded_file_rejects_missing_name(
) -> None:
    """缺少文件名时应在写入临时文件前失败。"""

    upload = FakeUploadedFile(
        filename=None,
        content=b"content",
    )

    with pytest.raises(
        DocumentValidationError,
        match="缺少有效文件名",
    ):
        async with stage_uploaded_file(
            upload,
            max_file_size_bytes=1024,
        ):
            pytest.fail(
                "无文件名上传不应进入上下文代码块"
            )

    # 即使文件名校验失败，finally 仍应执行。
    assert upload.closed is True