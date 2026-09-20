"""Agent公开事件的SSE文本编码器。

本模块位于AgentEventPublisher和FastAPI StreamingResponse之间。

本模块负责：

1. 接收已经通过Pydantic校验的Agent SSE事件；
2. 把事件序号编码成SSE id字段；
3. 把event_type编码成SSE event字段；
4. 把完整公开事件编码成紧凑JSON data字段；
5. 使用空行结束单个SSE事件帧；
6. 保留中文内容，不把中文转换成Unicode转义序列。

本模块不负责：

1. 分配事件序号；
2. 检查事件先后顺序；
3. 执行Agent、Planner或工具；
4. 创建FastAPI响应；
5. 处理客户端断开；
6. 保存诊断会话。
"""

import json

from app.schemas.agent_events import (
    AgentSseEvent,
    AgentSseEventBase,
)


# FastAPI StreamingResponse使用的响应媒体类型。
#
# charset=utf-8保证浏览器按照UTF-8解释中文事件内容。
AGENT_SSE_CONTENT_TYPE = (
    "text/event-stream; charset=utf-8"
)


# SSE响应不应被浏览器或反向代理缓存，
# 否则事件可能无法及时显示。
AGENT_SSE_CACHE_CONTROL = "no-cache"


class AgentSseEncodingError(
    RuntimeError
):
    """已经校验的公开事件无法编码成SSE文本。

    这是服务端内部错误。

    上层流式服务捕获该异常后，应尝试发布安全的
    stream_error事件，不能把原始异常直接发送给客户端。
    """


def encode_agent_sse_event(
    event: AgentSseEvent,
) -> str:
    """把一个结构化Agent事件编码成完整SSE文本帧。

    返回格式为：

    id: <事件序号>
    event: <事件类型>
    data: <JSON对象>
    <空行>

    最后的两个换行符非常重要。SSE客户端只有读取到空行，
    才会认为当前事件已经结束并触发对应的事件处理器。
    """

    # AgentSseEvent是类型联合，不能直接用于isinstance()。
    #
    # 九种具体事件都继承AgentSseEventBase，
    # 因此运行时使用共同基类验证输入对象。
    if not isinstance(
        event,
        AgentSseEventBase,
    ):
        raise TypeError(
            "event必须是"
            "AgentSseEventBase的具体事件实例"
        )

    try:
        # mode="json"把Pydantic对象中的嵌套模型、
        # tuple和其他字段转换成可以交给JSON编码器的值。
        event_data = event.model_dump(
            mode="json",
        )

        # separators去除JSON中不必要的空格，
        # 减少每个流式事件的传输体积。
        #
        # ensure_ascii=False保留中文，
        # 使浏览器、日志和测试输出更容易阅读。
        #
        # allow_nan=False拒绝NaN和Infinity，
        # 因为它们不是标准JSON值。
        json_data = json.dumps(
            event_data,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise AgentSseEncodingError(
            "Agent公开事件无法编码成标准JSON"
        ) from exc

    # JSON字符串内部的换行符会被编码成\\n，
    # 因此不会意外产生额外的SSE data行。
    #
    # sequence和event_type已经经过Pydantic校验，
    # 调用方不能在这里伪造id或事件名称。
    return (
        f"id: {event.sequence}\n"
        f"event: {event.event_type}\n"
        f"data: {json_data}\n"
        "\n"
    )