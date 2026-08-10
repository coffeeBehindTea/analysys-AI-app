"""FastAPI 依赖装配函数。

这里决定 Settings、AsyncOpenAI 和 TriageService 如何创建与连接，
但不实现具体业务。
"""

# AsyncIterator[T] 表示可以通过 async for 消费、逐步产生 T 的异步迭代器。
# FastAPI 也用这种返回注解识别含 yield 的异步资源依赖。
from collections.abc import AsyncIterator

# Annotated[真实类型, 元数据] 用于在类型上附加 Depends 等框架信息。
from typing import Annotated

# Depends 声明“此参数由 FastAPI 调用指定函数提供”，而不是由客户端传入。
from fastapi import Depends

# AsyncOpenAI 是 OpenAI Python SDK 的异步客户端类型。
from openai import AsyncOpenAI

from app.config import Settings, get_settings
from app.services.llm_client import create_llm_client, get_llm_model
from app.services.triage import TriageService


async def get_llm_client(
    settings: Annotated[Settings, Depends(get_settings)],
) -> AsyncIterator[AsyncOpenAI]:
    """为一次 HTTP 请求创建并最终关闭异步 LLM 客户端。

    settings 参数由 FastAPI 调用 get_settings() 注入。
    yield 产生的 AsyncOpenAI 会继续注入下游依赖。
    """

    # 工厂函数负责检查配置，并构造真正的 SDK 客户端。
    client = create_llm_client(settings)

    try:
        # yield 暂停当前依赖，把 client 提供给 get_triage_service()。
        yield client
    finally:
        # close() 是 AsyncOpenAI 的异步资源清理方法，因此必须 await。
        # finally 保证正常响应或异常时都尽量关闭其 HTTP 连接资源。
        await client.close()


def get_triage_service(
    client: Annotated[AsyncOpenAI, Depends(get_llm_client)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> TriageService:
    """使用已注入的客户端和 Settings 构造故障分析 Service。"""

    # get_llm_model() 负责检查模型名；TriageService 只接收可用依赖。
    return TriageService(
        client=client,
        model=get_llm_model(settings),
    )
