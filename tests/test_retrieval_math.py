"""余弦相似度数学函数的单元测试。"""

from math import sqrt

import pytest

from app.services.retrieval import cosine_similarity


def test_identical_vectors_have_similarity_one() -> None:
    """方向完全相同的向量，相似度应为 1。"""

    similarity = cosine_similarity(
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
    )

    # 浮点计算不适合始终使用严格相等判断。
    # pytest.approx() 会允许非常小的浮点误差。
    assert similarity == pytest.approx(1.0)


def test_orthogonal_vectors_have_similarity_zero() -> None:
    """互相垂直的向量，相似度应为 0。"""

    similarity = cosine_similarity(
        [1.0, 0.0],
        [0.0, 1.0],
    )

    assert similarity == pytest.approx(0.0)


def test_opposite_vectors_have_similarity_negative_one() -> None:
    """方向完全相反的向量，相似度应为 -1。"""

    similarity = cosine_similarity(
        [1.0, 0.0],
        [-1.0, 0.0],
    )

    assert similarity == pytest.approx(-1.0)


def test_known_angle_has_expected_similarity() -> None:
    """验证一个可以手工计算的 45 度示例。"""

    similarity = cosine_similarity(
        [1.0, 1.0],
        [1.0, 0.0],
    )

    # 两个向量夹角为 45 度：
    # cos(45°) = 1 / sqrt(2)
    assert similarity == pytest.approx(
        1.0 / sqrt(2.0)
    )


def test_different_dimensions_are_rejected() -> None:
    """维度不同的向量不能进行比较。"""

    with pytest.raises(
        ValueError,
        match="维度必须一致",
    ):
        cosine_similarity(
            [1.0, 2.0],
            [1.0, 2.0, 3.0],
        )


def test_zero_vector_is_rejected() -> None:
    """零向量不存在可比较的方向。"""

    with pytest.raises(
        ValueError,
        match="零向量",
    ):
        cosine_similarity(
            [0.0, 0.0],
            [1.0, 0.0],
        )


def test_non_finite_value_is_rejected() -> None:
    """NaN 和无穷值不能进入相似度计算。"""

    with pytest.raises(
        ValueError,
        match="NaN 或无穷值",
    ):
        cosine_similarity(
            [float("nan"), 1.0],
            [1.0, 0.0],
        )