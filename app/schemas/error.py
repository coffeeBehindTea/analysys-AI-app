"""API 统一错误响应的数据契约。"""

# BaseModel 提供运行时校验和序列化；ConfigDict 配置模型行为。
from pydantic import BaseModel, ConfigDict


class ErrorDetail(BaseModel):
    """描述一次具体的应用错误。"""

    # 禁止返回模型中没有声明的额外字段，保持错误契约稳定。
    model_config = ConfigDict(extra="forbid")

    # 程序可识别的稳定错误码，例如 llm_timeout。
    code: str

    # 展示给 API 调用方的安全错误信息。
    message: str


class ErrorResponse(BaseModel):
    """所有可预期应用错误使用的统一响应结构。"""

    model_config = ConfigDict(extra="forbid")

    # 用于在服务端日志中追踪本次失败请求。
    request_id: str

    # 错误的具体代码和公开信息。
    # 嵌套 BaseModel 后，Pydantic 会递归校验内部错误结构。
    error: ErrorDetail
