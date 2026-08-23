"""确定性查询改写模块的离线测试。

这些测试不访问LLM、Embedding、Chroma或网络。

测试目标：
1. 型号别名和中英文术语可以正确扩展；
2. 错误码、数值等归一化结果不会丢失；
3. 已经存在的英文词不会被重复追加；
4. 没有规则命中的问题保持原义；
5. 相同输入始终产生相同结果；
6. 组合语义规则只在全部必要条件都存在时触发；
7. 空值、错误类型和非法规则配置能够被发现。
"""

from dataclasses import FrozenInstanceError

import pytest

from app.services.query_rewriting import (
    DETERMINISTIC_QUERY_REWRITE_VERSION,
    QUERY_REWRITE_RULES,
    RewrittenRetrievalQuery,
    rewrite_retrieval_query,
)


def test_model_and_electrical_terms_are_expanded(
) -> None:
    """型号、线缆和电源术语应得到对应英文扩展。"""

    result = rewrite_retrieval_query(
        "DM260读码器的线缆靠近"
        "大电流线路或高压电源时，"
        "可能造成哪些风险？"
    )

    assert isinstance(
        result,
        RewrittenRetrievalQuery,
    )

    assert result.rewrite_version == (
        DETERMINISTIC_QUERY_REWRITE_VERSION
    )

    # DM260先被已有归一化模块转换成规范型号。
    assert (
        "dataman-260"
        in result.normalized_query
    )

    # 最终检索文本必须保留原中文问题。
    assert (
        "线缆靠近大电流线路或高压电源"
        in result.rewritten_query
    )

    assert result.added_terms == (
        "DataMan 260",
        "DM260",
        "cable",
        "cables",
        "high-current wiring",
        "high-voltage power sources",
        "route cables and wires away",
        "precautions",
    )

    assert result.applied_rule_ids == (
        "model-dataman-260",
        "zh-cable",
        "zh-high-current-wiring",
        "zh-high-voltage-power",
        "zh-electrical-routing-context",
    )


def test_charge_discharge_temperature_is_expanded(
) -> None:
    """充放电温度问题应补充英文手册常用词。"""

    result = rewrite_retrieval_query(
        "电池的充电温度范围和"
        "放电温度范围分别是多少？"
    )

    assert result.added_terms == (
        "charge",
        "charging",
        "discharge",
        "discharging",
        "temperature",
        "range",
        "charge temperature range",
        "discharge temperature range",
        "charge and discharge warnings",
    )

    assert result.applied_rule_ids == (
        "zh-charge",
        "zh-discharge",
        "zh-temperature",
        "zh-range",
        "zh-charge-discharge-temperature-context",
    )

    # 原始中文条件仍然存在，
    # 不能被英文扩展词替换掉。
    assert (
        "充电温度范围"
        in result.rewritten_query
    )
    assert (
        "放电温度范围"
        in result.rewritten_query
    )


def test_context_rule_requires_all_required_terms(
) -> None:
    """组合规则缺少任一必要概念时不得触发。"""

    # 这个问题包含“充电”和“温度”，但没有询问“放电”。
    # 基础的充电、温度和范围翻译仍应执行，
    # 但不能把它扩展成同时询问充放电范围的问题。
    result = rewrite_retrieval_query(
        "电池充电温度范围是多少？"
    )

    assert result.added_terms == (
        "charge",
        "charging",
        "temperature",
        "range",
    )

    assert (
        "zh-charge-discharge-temperature-context"
        not in result.applied_rule_ids
    )

    assert (
        "charge and discharge warnings"
        not in result.rewritten_query
    )


def test_electrical_context_requires_cable_term(
) -> None:
    """电气布线组合规则不能由普通高压问题单独触发。"""

    # “高压电源”会执行自己的直接翻译规则，
    # 但问题没有“线缆”，不能追加线缆布置章节词。
    result = rewrite_retrieval_query(
        "高压电源可能造成什么风险？"
    )

    assert result.added_terms == (
        "high-voltage power sources",
    )

    assert (
        "zh-electrical-routing-context"
        not in result.applied_rule_ids
    )

    assert "precautions" not in (
        result.rewritten_query
    )


def test_cleaning_query_is_expanded(
) -> None:
    """镜头积尘和清洁问题应补充英文维护术语。"""

    result = rewrite_retrieval_query(
        "DM260的镜头保护盖积尘后"
        "应该怎样清洁？"
    )

    assert result.added_terms == (
        "DataMan 260",
        "DM260",
        "reader lens cover",
        "dust",
        "clean",
        "cleaning",
    )

    assert result.applied_rule_ids == (
        "model-dataman-260",
        "zh-reader-lens-cover",
        "zh-dust",
        "zh-cleaning",
    )


def test_network_recovery_query_is_expanded(
) -> None:
    """网络恢复和继续任务应补充英文流程表达。"""

    result = rewrite_retrieval_query(
        "网络恢复后如何继续任务？"
    )

    assert result.added_terms == (
        "network recovery",
        "resume task",
    )

    assert result.applied_rule_ids == (
        "zh-network-recovery",
        "zh-resume-task",
    )


def test_structured_terms_remain_normalized(
) -> None:
    """查询改写不能破坏错误码和精确数值。"""

    result = rewrite_retrieval_query(
        "ERR_NET_4001在1500 ms后出现告警"
    )

    # 查询改写器复用了已有的归一化服务。
    assert (
        "err-net-4001"
        in result.normalized_query
    )
    assert (
        "1500ms"
        in result.normalized_query
    )

    assert (
        "err-net-4001"
        in result.rewritten_query
    )
    assert (
        "1500ms"
        in result.rewritten_query
    )

    # 当前问题没有命中双语扩展规则。
    assert result.added_terms == ()
    assert result.applied_rule_ids == ()


def test_query_without_matching_rule_is_unchanged(
) -> None:
    """没有命中规则时只进行基础归一化。"""

    result = rewrite_retrieval_query(
        "机器人当前位于哪里？"
    )

    assert result.added_terms == ()
    assert result.applied_rule_ids == ()

    # 没有扩展词时，最终查询应等于归一化查询。
    assert (
        result.rewritten_query
        == result.normalized_query
    )


def test_existing_expansion_terms_are_not_duplicated(
) -> None:
    """查询已经包含扩展词时不能再次追加相同内容。"""

    result = rewrite_retrieval_query(
        "充电 charge charging"
    )

    # “充电”会命中规则，但两个英文扩展词
    # 已经存在，因此查询没有真正发生改变。
    assert result.added_terms == ()
    assert result.applied_rule_ids == ()

    assert (
        result.rewritten_query
        == result.normalized_query
    )


def test_same_query_produces_same_result(
) -> None:
    """确定性改写对相同输入必须返回完全相同的结果。"""

    query = (
        "DM260镜头保护盖积尘后怎样清洁？"
    )

    first_result = rewrite_retrieval_query(
        query
    )
    second_result = rewrite_retrieval_query(
        query
    )

    # dataclass自动生成的相等比较会逐字段比较。
    assert first_result == second_result


def test_rewrite_rule_configuration_is_valid(
) -> None:
    """规则ID、触发词和扩展词必须完整且不重复。"""

    assert QUERY_REWRITE_RULES

    rule_ids = [
        rule.rule_id
        for rule in QUERY_REWRITE_RULES
    ]

    # 同一个规则ID不能表示两套不同规则。
    assert len(rule_ids) == len(
        set(rule_ids)
    )

    for rule in QUERY_REWRITE_RULES:
        assert rule.rule_id.strip()
        assert rule.trigger_terms
        assert rule.expansion_terms

        # 同一条规则内部不能重复声明触发词。
        assert len(rule.trigger_terms) == len(
            set(rule.trigger_terms)
        )

        # 同一条规则内部不能重复声明扩展词。
        assert len(rule.expansion_terms) == len(
            set(rule.expansion_terms)
        )

        # required_terms表示必须同时存在的附加条件。
        # 普通单条件规则使用空元组；组合规则不能重复声明条件。
        assert len(rule.required_terms) == len(
            set(rule.required_terms)
        )

        # 不允许空字符串规则。
        assert all(
            term.strip()
            for term in rule.trigger_terms
        )
        assert all(
            term.strip()
            for term in rule.expansion_terms
        )
        assert all(
            term.strip()
            for term in rule.required_terms
        )


@pytest.mark.parametrize(
    "invalid_query",
    [
        None,
        123,
        [],
    ],
    ids=[
        "none",
        "integer",
        "list",
    ],
)
def test_non_string_query_is_rejected(
    invalid_query: object,
) -> None:
    """公共函数必须拒绝非字符串输入。"""

    with pytest.raises(
        TypeError,
        match="query必须是字符串",
    ):
        rewrite_retrieval_query(
            invalid_query,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "blank_query",
    [
        "",
        "   ",
        "\n\t",
    ],
    ids=[
        "empty",
        "spaces",
        "newline-and-tab",
    ],
)
def test_blank_query_is_rejected(
    blank_query: str,
) -> None:
    """空字符串和只包含空白的查询必须被拒绝。"""

    with pytest.raises(
        ValueError,
        match="query不能为空",
    ):
        rewrite_retrieval_query(
            blank_query
        )


def test_rewrite_result_is_immutable(
) -> None:
    """改写结果生成后不能在检索途中被修改。"""

    result = rewrite_retrieval_query(
        "网络恢复后如何继续任务？"
    )

    # frozen=True使字段赋值抛出FrozenInstanceError。
    with pytest.raises(
        FrozenInstanceError,
    ):
        result.rewrite_version = (  # type: ignore[misc]
            "unexpected-version"
        )
