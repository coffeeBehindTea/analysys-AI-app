"""请求级已验证图片Store的离线单元测试。

本模块只验证内存映射、输入约束和请求隔离。
它不会调用Vision模型，不访问网络，也不保存图片。
"""

from base64 import (
    b64decode,
)
from hashlib import (
    sha256,
)
from typing import (
    Any,
)

import pytest

from app.agent.vision_input_store import (
    RequestVisionInputStore,
)
from app.schemas.vision import (
    VisionImageMetadata,
    VisionInput,
)


# 一个仅用于离线测试的1×1 PNG。
# 测试不会把它发送给任何外部服务。
TEST_PNG_BYTES = b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"
    "CAQAAAC1HAwCAAAAC0lEQVR42mNk+A8A"
    "AQUBAScY42YAAAAASUVORK5CYII="
)


# 两个合法且互不相同的请求级图片引用。
PRIMARY_IMAGE_REF = "image_primary"
SECONDARY_IMAGE_REF = "image_secondary"


def make_vision_input(
    *,
    analysis_goal: str = "检查设备面板状态",
) -> VisionInput:
    """构造一项字段相互一致的内部VisionInput。"""

    metadata = VisionImageMetadata(
        mime_type="image/png",
        size_bytes=len(TEST_PNG_BYTES),
        width_px=1,
        height_px=1,
        pixel_count=1,
        sha256_hex=(
            sha256(TEST_PNG_BYTES)
            .hexdigest()
        ),
    )

    return VisionInput(
        metadata=metadata,
        image_bytes=TEST_PNG_BYTES,
        analysis_goal=analysis_goal,
        detail="auto",
    )


def test_new_store_is_empty(
) -> None:
    """新Store应没有图片，合法未知引用应返回None。"""

    store = RequestVisionInputStore()

    assert len(store) == 0
    assert store.get(PRIMARY_IMAGE_REF) is None


def test_register_returns_same_validated_input(
) -> None:
    """登记后应按精确引用返回同一个VisionInput。"""

    store = RequestVisionInputStore()
    vision_input = make_vision_input()

    store.register(
        image_ref=PRIMARY_IMAGE_REF,
        vision_input=vision_input,
    )

    assert len(store) == 1
    assert store.get(PRIMARY_IMAGE_REF) is (
        vision_input
    )


def test_reference_is_normalized_on_register_and_get(
) -> None:
    """登记和查询都应复用共享的首尾空白清理规则。"""

    store = RequestVisionInputStore()
    vision_input = make_vision_input()

    store.register(
        image_ref=(
            f"  {PRIMARY_IMAGE_REF}  "
        ),
        vision_input=vision_input,
    )

    assert store.get(PRIMARY_IMAGE_REF) is (
        vision_input
    )
    assert store.get(
        f"  {PRIMARY_IMAGE_REF}  "
    ) is vision_input


def test_multiple_references_remain_independent(
) -> None:
    """同一请求中的不同引用应分别指向各自图片输入。"""

    store = RequestVisionInputStore()
    primary_input = make_vision_input(
        analysis_goal="检查主图片",
    )
    secondary_input = make_vision_input(
        analysis_goal="检查辅助图片",
    )

    store.register(
        image_ref=PRIMARY_IMAGE_REF,
        vision_input=primary_input,
    )
    store.register(
        image_ref=SECONDARY_IMAGE_REF,
        vision_input=secondary_input,
    )

    assert len(store) == 2
    assert store.get(PRIMARY_IMAGE_REF) is (
        primary_input
    )
    assert store.get(SECONDARY_IMAGE_REF) is (
        secondary_input
    )


def test_duplicate_reference_cannot_replace_original(
) -> None:
    """重复引用应被拒绝，并保留首次登记的对象。"""

    store = RequestVisionInputStore()
    original_input = make_vision_input(
        analysis_goal="检查原始图片",
    )
    replacement_input = make_vision_input(
        analysis_goal="不应覆盖原始图片",
    )

    store.register(
        image_ref=PRIMARY_IMAGE_REF,
        vision_input=original_input,
    )

    with pytest.raises(
        ValueError,
        match="已经在当前请求中登记",
    ):
        store.register(
            image_ref=PRIMARY_IMAGE_REF,
            vision_input=replacement_input,
        )

    assert len(store) == 1
    assert store.get(PRIMARY_IMAGE_REF) is (
        original_input
    )


@pytest.mark.parametrize(
    "invalid_reference",
    [
        "../robot.png",
        "https://example.com/robot.png",
        "robot-without-prefix",
    ],
)
def test_register_rejects_invalid_reference_format(
    invalid_reference: str,
) -> None:
    """路径、URL和错误前缀不能进入请求图片白名单。"""

    store = RequestVisionInputStore()

    with pytest.raises(
        ValueError,
        match="不符合请求级图片引用格式",
    ):
        store.register(
            image_ref=invalid_reference,
            vision_input=make_vision_input(),
        )

    assert len(store) == 0


def test_store_rejects_non_string_reference(
) -> None:
    """绕过类型注解传入非字符串引用时应明确失败。"""

    store = RequestVisionInputStore()

    with pytest.raises(
        TypeError,
        match="image_ref必须是字符串",
    ):
        store.get(123)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "invalid_input",
    [
        {"image_bytes": b"fake"},
        b"raw-image-bytes",
        "base64-image-content",
    ],
)
def test_register_rejects_non_vision_input(
    invalid_input: Any,
) -> None:
    """Store不能接收dict、原始字节或Base64字符串。"""

    store = RequestVisionInputStore()

    with pytest.raises(
        TypeError,
        match="vision_input必须是VisionInput",
    ):
        store.register(
            image_ref=PRIMARY_IMAGE_REF,
            vision_input=invalid_input,
        )

    assert len(store) == 0


def test_two_store_instances_do_not_share_images(
) -> None:
    """两个请求级Store实例之间不能共享已登记图片。"""

    first_request_store = (
        RequestVisionInputStore()
    )
    second_request_store = (
        RequestVisionInputStore()
    )
    vision_input = make_vision_input()

    first_request_store.register(
        image_ref=PRIMARY_IMAGE_REF,
        vision_input=vision_input,
    )

    assert first_request_store.get(
        PRIMARY_IMAGE_REF
    ) is vision_input
    assert second_request_store.get(
        PRIMARY_IMAGE_REF
    ) is None
    assert len(first_request_store) == 1
    assert len(second_request_store) == 0
