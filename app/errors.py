"""应用可以预期并安全转换为 HTTP 响应的异常类型。"""


class ApplicationError(RuntimeError):
    """应用内部可预期错误的基类。

    RuntimeError 是 Python 内置运行时异常。建立 ApplicationError 后，
    异常处理器可以只捕获这类已知错误，而不掩盖 TypeError 等程序 Bug。
    """


class LLMConfigurationError(ApplicationError):
    """LLM 配置缺失或无效。"""


class LLMTimeoutError(ApplicationError):
    """LLM 请求超时。"""


class LLMUpstreamError(ApplicationError):
    """LLM 上游服务不可用或返回错误状态。"""


class InvalidLLMResponseError(ApplicationError):
    """LLM 返回的内容为空或不符合约定格式。"""

class EmbeddingConfigurationError(ApplicationError):
    """Embedding 模型配置缺失或无效。"""


class EmbeddingTimeoutError(ApplicationError):
    """Embedding 请求超时。"""


class EmbeddingUpstreamError(ApplicationError):
    """Embedding 上游服务不可用或返回错误状态。"""


class InvalidEmbeddingResponseError(ApplicationError):
    """Embedding 上游响应不完整或向量不符合约定。"""


class DuplicateDocumentError(ApplicationError):
    """相同文件内容已经存在于知识库中。"""


class VectorStoreError(ApplicationError):
    """向量数据库读写失败或返回了无效数据。"""


class DocumentValidationError(ApplicationError):
    """上传文档的类型、大小或内容不符合摄取要求。"""


class DocumentNotFoundError(ApplicationError):
    """指定的知识库文档不存在。"""