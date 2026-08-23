"""检索文本确定性归一化模块的离线测试。

这些测试不访问Embedding、LLM、ChromaDB或网络。

测试目标是保证错误码、型号、数值和Unicode字符
始终按同一组确定性规则转换，避免后续关键词检索
因为输入格式变化而产生不可预测的召回结果。
"""

import pytest

from app.services.lexical_normalization import (
    NormalizedRetrievalText,
    normalize_retrieval_text,
)


@pytest.mark.parametrize(
    (
        "raw_text",
        "expected_identifier",
    ),
    [
        (
            "ERR-SAF-1002",
            "err-saf-1002",
        ),
        (
            "err saf 1002",
            "err-saf-1002",
        ),
        (
            "ERR_SAF_1002",
            "err-saf-1002",
        ),
        (
            "ERRSAF1002",
            "err-saf-1002",
        ),
    ],
    ids=[
        "hyphen",
        "spaces",
        "underscores",
        "no-separators",
    ],
)
def test_identifier_variants_are_normalized(
    raw_text: str,
    expected_identifier: str,
) -> None:
    """同一故障码的不同格式应得到相同规范值。"""

    # 调用真正的生产归一化函数，
    # 而不是在测试中复制一份归一化逻辑。
    result = normalize_retrieval_text(
        raw_text
    )

    # 函数应返回声明的内部数据对象。
    assert isinstance(
        result,
        NormalizedRetrievalText,
    )

    # 当前输入只有一个故障码，
    # 因此结果应为只含一个元素的元组。
    assert result.identifiers == (
        expected_identifier,
    )

    # 完整归一化文本中也必须使用规范形式，
    # 供后续普通关键词切分器使用。
    assert (
        expected_identifier
        in result.normalized_text
    )


def test_identifiers_are_unique_and_ordered(
) -> None:
    """重复编号应去重，并保留首次出现顺序。"""

    result = normalize_retrieval_text(
        "ERR-SAF-1002、"
        "err saf 1002、"
        "TEST_NET_001"
    )

    # 前两个字符串表示同一个故障码，
    # 所以err-saf-1002只能出现一次。
    #
    # test-net-001在它之后首次出现，
    # 因此应排在元组第二位。
    assert result.identifiers == (
        "err-saf-1002",
        "test-net-001",
    )


@pytest.mark.parametrize(
    "raw_text",
    [
        "DataMan 260",
        "DM260",
        "dm-260",
        "ＤＭ２６０",
    ],
    ids=[
        "full-name",
        "short-name",
        "hyphenated-short-name",
        "full-width-short-name",
    ],
)
def test_model_aliases_are_normalized(
    raw_text: str,
) -> None:
    """产品全称、简称和全角字符应映射到同一型号。"""

    result = normalize_retrieval_text(
        raw_text
    )

    assert result.model_aliases == (
        "dataman-260",
    )

    assert (
        "dataman-260"
        in result.normalized_text
    )

    # 型号中的260属于型号结构，
    # 不应再作为独立数值重复计分。
    assert result.numeric_terms == ()


@pytest.mark.parametrize(
    (
        "raw_text",
        "expected_numeric_term",
    ),
    [
        (
            "-10°C",
            "-10°c",
        ),
        (
            "−10 ℃",
            "-10°c",
        ),
        (
            "1500 ms",
            "1500ms",
        ),
        (
            "1.2 m",
            "1.2m",
        ),
        (
            "+5 V",
            "+5v",
        ),
    ],
    ids=[
        "ascii-minus-celsius",
        "unicode-minus-celsius",
        "milliseconds",
        "decimal-meters",
        "positive-voltage",
    ],
)
def test_numeric_terms_preserve_sign_and_unit(
    raw_text: str,
    expected_numeric_term: str,
) -> None:
    """数值归一化不能丢失正负号、小数或单位。"""

    result = normalize_retrieval_text(
        raw_text
    )

    assert result.numeric_terms == (
        expected_numeric_term,
    )

    assert (
        expected_numeric_term
        in result.normalized_text
    )


def test_structured_numbers_are_not_double_counted(
) -> None:
    """错误码和型号中的数字不应变成普通数值词。"""

    result = normalize_retrieval_text(
        "ERR_NET_4001、DM260、TEST-ID-001"
    )

    assert result.identifiers == (
        "err-net-4001",
        "test-id-001",
    )

    assert result.model_aliases == (
        "dataman-260",
    )

    # 4001、260和001都属于更精确的结构化词，
    # 所以不能再次进入numeric_terms。
    assert result.numeric_terms == ()


def test_duplicate_numeric_terms_keep_first_order(
) -> None:
    """重复数值应去重，但不同数值仍保留原顺序。"""

    result = normalize_retrieval_text(
        "1500 ms、1500ms、-10°C、1500 ms"
    )

    assert result.numeric_terms == (
        "1500ms",
        "-10°c",
    )


def test_unicode_case_whitespace_and_dash_are_normalized(
) -> None:
    """全角字符、大小写、空白和Unicode横线应统一。"""

    result = normalize_retrieval_text(
        "  ＥＲＲ＿ＳＡＦ＿１００２"
        "\tDataMan—260  "
    )

    # 全角字符先经过NFKC转换；
    # em dash再被转换成ASCII连字符；
    # Tab和多余空格最后被压缩。
    assert result.normalized_text == (
        "err-saf-1002 dataman-260"
    )

    assert result.identifiers == (
        "err-saf-1002",
    )

    assert result.model_aliases == (
        "dataman-260",
    )


@pytest.mark.parametrize(
    "blank_text",
    [
        "",
        "   ",
        "\n\t",
    ],
    ids=[
        "empty",
        "spaces-only",
        "control-whitespace-only",
    ],
)
def test_blank_text_is_rejected(
    blank_text: str,
) -> None:
    """空字符串和纯空白不能进入关键词检索。"""

    # pytest.raises()要求with代码块内
    # 必须抛出指定类型的异常。
    #
    # match使用正则表达式检查异常消息，
    # 防止测试因为无关ValueError而错误通过。
    with pytest.raises(
        ValueError,
        match="不能为空",
    ):
        normalize_retrieval_text(
            blank_text
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        123,
        None,
    ],
    ids=[
        "integer",
        "none",
    ],
)
def test_non_string_input_is_rejected(
    invalid_value: object,
) -> None:
    """普通Python调用传入非字符串时应明确失败。"""

    with pytest.raises(
        TypeError,
        match="必须是字符串",
    ):
        # 测试故意传入与类型注解不一致的数据，
        # 用于验证函数的运行时防御。
        normalize_retrieval_text(
            invalid_value  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    (
        "raw_text",
        "expected_normalized_text",
    ),
    [
        (
            "ERR_NET_4001 网络恢复",
            "err-net-4001 网络恢复",
        ),
        (
            "第 20 章",
            "第 20 章",
        ),
        (
            "1500 ms 后继续任务",
            "1500ms 后继续任务",
        ),
    ],
    ids=[
        "identifier-before-chinese",
        "unitless-number-between-chinese",
        "number-with-unit-before-chinese",
    ],
)
def test_numeric_normalization_preserves_word_boundaries(
    raw_text: str,
    expected_normalized_text: str,
) -> None:
    """数值归一化不能吞掉数值外部的分隔空格。"""

    result = normalize_retrieval_text(
        raw_text
    )

    # 有单位时，只删除数字和单位之间的空格；
    # 没有单位时，保留数字与后续文本之间的空格。
    assert (
        result.normalized_text
        == expected_normalized_text
    )