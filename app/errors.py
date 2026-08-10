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
