"""编排文档校验、解析、切分、向量化和持久化。"""

from pathlib import Path

from app.errors import (
    DocumentValidationError,
    DuplicateDocumentError,
)
from app.schemas.knowledge import (
    DocumentRecord,
    SupportedDocumentType,
)
from app.services.document_preparation import (
    TextDocumentPreparationService,
    calculate_file_sha256,
)
from app.services.retrieval_baseline import (
    EmbeddingProvider,
    embed_document_chunks,
)
from app.services.vector_store import VectorStore


# 文件扩展名到知识库文件类型的映射。
SUPPORTED_FILE_TYPES: dict[
    str,
    SupportedDocumentType,
] = {
    ".pdf": "pdf",
    ".md": "md",
    ".txt": "txt",
}


# 默认限制为 20 MiB。
#
# 1 MiB = 1024 × 1024 字节。
DEFAULT_MAX_FILE_SIZE_BYTES = (
    20 * 1024 * 1024
)


class DocumentService:
    """负责一份文档从本地文件到知识库的完整摄取流程。"""

    def __init__(
        self,
        *,
        preparation_service: TextDocumentPreparationService,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        embedding_model: str,
        batch_size: int = 64,
        max_file_size_bytes: int = (
            DEFAULT_MAX_FILE_SIZE_BYTES
        ),
    ) -> None:
        """保存依赖并检查摄取配置。"""

        cleaned_embedding_model = (
            embedding_model.strip()
        )

        if not cleaned_embedding_model:
            raise ValueError(
                "Embedding 模型名称不能为空"
            )

        if batch_size < 1:
            raise ValueError(
                "Embedding batch_size 必须大于或等于 1"
            )

        if max_file_size_bytes < 1:
            raise ValueError(
                "最大文件大小必须大于 0"
            )

        # PreparationService 负责：
        # Loader 选择、文档解析和 Chunk 切分。
        self._preparation_service = (
            preparation_service
        )

        # EmbeddingProvider 可以是真实 EmbeddingService，
        # 也可以是测试中的 Fake。
        self._embedding_provider = (
            embedding_provider
        )

        # VectorStore 可以是真实 ChromaVectorStore，
        # 也可以是测试中的 Fake。
        self._vector_store = vector_store

        self._embedding_model = (
            cleaned_embedding_model
        )
        self._batch_size = batch_size
        self._max_file_size_bytes = (
            max_file_size_bytes
        )

    async def ingest_document(
        self,
        path: Path,
    ) -> DocumentRecord:
        """校验并将一份文档完整摄取到知识库。"""

        # is_file() 同时检查：
        # 1. 路径存在；
        # 2. 路径指向普通文件，而不是目录。
        if not path.is_file():
            raise DocumentValidationError(
                f"待摄取文件不存在：{path}"
            )

        # suffix 返回带点的扩展名，例如 ".PDF"。
        #
        # lower() 使大写扩展名也能识别。
        suffix = path.suffix.lower()

        # get() 找不到键时返回 None，
        # 不会像字典的 [] 访问一样抛出 KeyError。
        file_type = SUPPORTED_FILE_TYPES.get(
            suffix
        )

        if file_type is None:
            raise DocumentValidationError(
                "知识库只支持 .pdf、.md 和 .txt 文件"
            )

        # stat() 返回文件系统状态信息。
        #
        # st_size 是原始文件的字节数。
        file_size_bytes = path.stat().st_size

        if file_size_bytes == 0:
            raise DocumentValidationError(
                f"文档不能为空：{path.name}"
            )

        if (
            file_size_bytes
            > self._max_file_size_bytes
        ):
            raise DocumentValidationError(
                f"文档超过大小限制：{path.name}；"
                f"最大允许 "
                f"{self._max_file_size_bytes} 字节"
            )

        # 根据原始文件字节计算 SHA-256。
        #
        # 这一步发生在解析和 Embedding 之前，
        # 可以让重复文档尽早失败，避免调用收费接口。
        document_id = calculate_file_sha256(
            path
        )

        if self._vector_store.has_document(
            document_id
        ):
            raise DuplicateDocumentError(
                f"文档已经存在：{path.name}"
            )

        try:
            # prepare() 内部依次完成：
            #
            # 文件格式路由
            # → Loader 解析
            # → SourceTextSegment
            # → TextChunker
            # → DocumentChunk
            document_chunks = (
                self._preparation_service.prepare(
                    path
                )
            )
        except (
            FileNotFoundError,
            ValueError,
        ) as exc:
            # 将 Loader 和 Chunker 抛出的底层输入错误，
            # 转换为统一的文档校验错误。
            #
            # from exc 会把原始异常保存在 __cause__ 中。
            raise DocumentValidationError(
                f"文档解析或切分失败：{path.name}；"
                f"{exc}"
            ) from exc

        if not document_chunks:
            # 正常 Loader 和 Chunker 不应该返回空列表，
            # 但在业务边界继续防御空结果。
            raise DocumentValidationError(
                f"文档没有产生可用 Chunk：{path.name}"
            )

        # DocumentPreparationService 当前也会计算一次文件哈希。
        #
        # 如果文件在第一次计算哈希之后被其他程序修改，
        # 两次 document_id 会不一致。
        if any(
            chunk.document_id != document_id
            for chunk in document_chunks
        ):
            raise DocumentValidationError(
                "文档在摄取过程中发生变化，请重新上传"
            )

        # embed_document_chunks() 会：
        #
        # 1. 按 batch_size 分批；
        # 2. 在当前调用内复用相同正文的向量；
        # 3. 调用 provider.embed_texts()；
        # 4. 按原 Chunk 顺序构造 EmbeddedChunk。
        embedded_chunks = await embed_document_chunks(
            document_chunks,
            provider=self._embedding_provider,
            embedding_model=self._embedding_model,
            batch_size=self._batch_size,
        )

        # 一个页面或章节可能被切成多个 Chunk。
        #
        # 所以这里统计的是有效页面
        # 章节数量，而不是 Chunk 数量。
        locations = {
            chunk.page_or_section
            for chunk in document_chunks
        }

        record = DocumentRecord(
            document_id=document_id,
            source_file=path.name,
            file_type=file_type,
            file_size_bytes=file_size_bytes,
            page_or_section_count=len(locations),
            chunk_count=len(embedded_chunks),
            embedding_model=self._embedding_model,

            # status 使用 DocumentRecord 的默认值 "ready"。
        )

        # 只有解析、切分和所有向量生成成功后，
        # 才将数据写入向量库。
        #
        # add_document() 成功返回后，
        # record 才真正代表一份 ready 文档。
        self._vector_store.add_document(
            record=record,
            chunks=embedded_chunks,
        )

        return record