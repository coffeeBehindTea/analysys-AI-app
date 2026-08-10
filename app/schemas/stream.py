"""SSE 流式接口中各类事件的数据契约。"""

# Literal 表示字段只能使用列出的固定字符串。
from typing import Literal

# BaseModel 提供运行时校验和 JSON 序列化；ConfigDict 配置模型行为。
from pydantic import BaseModel, ConfigDict, Field

from app.schemas.error import ErrorDetail


class StreamMetaData(BaseModel):
    """meta 事件的数据：在流开始时告诉客户端本次请求的追踪 ID。"""

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(
        min_length=1,
        description="本次流式请求的唯一追踪标识",
    )


class StreamDeltaData(BaseModel):
    """delta 事件的数据：只携带本次新增的文本片段。"""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(
        min_length=1,
        description="相对于前一个事件新增的 LLM 文本",
    )


class StreamDoneData(BaseModel):
    """done 事件的数据：明确表示流已经正常完成。"""

    model_config = ConfigDict(extra="forbid")

    # Literal["completed"] 既是类型限制，也是协议约定。
    status: Literal["completed"] = "completed"


class StreamErrorData(BaseModel):
    """error 事件的数据：表示流开始后发生了可预期错误。"""

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(
        min_length=1,
        description="发生错误的流式请求追踪标识",
    )

    # 复用普通 JSON 错误响应的 code/message 结构，
    # 让非流式接口与流式接口使用相同的公开错误语义。
    error: ErrorDetail
