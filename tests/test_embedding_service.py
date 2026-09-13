"""EmbeddingService 的不联网单元测试。"""

# SimpleNamespace 可以快速创建带属性的简单对象，
# 用来模拟 SDK 返回的 response.data、item.index 和 item.embedding。
from types import SimpleNamespace

# AsyncMock 模拟需要 await 的异步方法；
# MagicMock 模拟普通对象及其嵌套属性。
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.errors import InvalidEmbeddingResponseError
from app.services.embedding import EmbeddingService


@pytest.mark.asyncio
async def test_embed_texts_restores_input_order() -> None:
    """服务应根据 index 恢复向量与输入文本的对应顺序。"""

    client = MagicMock()

    # 故意让 index=1 出现在 index=0 前面，
    # 验证服务不是盲目信任服务器的返回顺序。
    client.embeddings.create = AsyncMock(
        return_value=SimpleNamespace(
            data=[
                SimpleNamespace(
                    index=1,
                    embedding=[0.0, 1.0],
                ),
                SimpleNamespace(
                    index=0,
                    embedding=[1.0, 0.0],
                ),
            ]
        )
    )

    service = EmbeddingService(
        client=client,
        model="test-embedding-model",
    )

    vectors = await service.embed_texts(
        [
            "  电机超时  ",
            "安全激光雷达触发",
        ]
    )

    # 输出必须恢复为输入文本的原始顺序。
    assert vectors == [
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    # assert_awaited_once_with() 是 AsyncMock 提供的方法，
    # 它验证异步方法只被等待一次，而且参数完全符合预期。
    client.embeddings.create.assert_awaited_once_with(
        model="test-embedding-model",
        input=[
            "电机超时",
            "安全激光雷达触发",
        ]
    )


@pytest.mark.asyncio
async def test_embed_texts_rejects_empty_batch() -> None:
    """空批次应在联网前被拒绝。"""

    client = MagicMock()
    client.embeddings.create = AsyncMock()

    service = EmbeddingService(
        client=client,
        model="test-embedding-model",
    )

    with pytest.raises(
        ValueError,
        match="文本列表不能为空",
    ):
        await service.embed_texts([])

    # 因为输入在本地校验阶段就失败了，
    # 所以不能发生任何上游调用。
    client.embeddings.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_embed_texts_rejects_incomplete_response() -> None:
    """上游少返回一个向量时应拒绝该响应。"""

    client = MagicMock()

    # 输入将有两个文本，但这里只返回 index=0。
    client.embeddings.create = AsyncMock(
        return_value=SimpleNamespace(
            data=[
                SimpleNamespace(
                    index=0,
                    embedding=[1.0, 0.0],
                ),
            ]
        )
    )

    service = EmbeddingService(
        client=client,
        model="test-embedding-model",
    )

    with pytest.raises(
        InvalidEmbeddingResponseError,
        match="数量或索引",
    ):
        await service.embed_texts(
            ["第一个文本", "第二个文本"]
        )