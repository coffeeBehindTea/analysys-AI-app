"""面向检索的确定性查询改写。

本模块负责：
1. 调用已有归一化模块统一错误码、型号和数值格式；
2. 根据人工维护的词汇表补充型号别名；
3. 为中文工程术语补充可审计的英文检索词；
4. 返回改写版本、命中规则和新增词。

本模块不调用LLM、Embedding、Chroma或关键词检索。

它只负责把一个原始问题转换成更适合后续检索的文本，
不能生成故障原因、技术参数、操作步骤或其他答案内容。
"""

from dataclasses import dataclass

from app.services.lexical_normalization import (
    normalize_retrieval_text,
)


# v3增加了DataMan连接器与安全检查的
# 中英文检索词对齐。
#
# v4把Planner可能生成的“间隙、锁紧环、检查要求”等
# 连接状态查询也归入DataMan连接器安全检查语境。
#
# 版本号进入评测报告，用来区分：
# - deterministic-v1：只执行单个术语扩展；
# - deterministic-v2：在此基础上增加组合条件扩展。
# - deterministic-v3：增加受型号条件限制的
#   DataMan连接器和检查安全术语扩展；
# - deterministic-v4：扩展Planner常用的连接状态
#   检查表达，但仍不加入页码或答案事实。
DETERMINISTIC_QUERY_REWRITE_VERSION = (
    "deterministic-v4"
)


@dataclass(
    frozen=True,
    slots=True,
)
class QueryRewriteRule:
    """一条确定性查询改写规则。"""

    # 规则的稳定标识，用于测试和报告审计。
    rule_id: str

    # 只要归一化查询包含其中任意一个触发词，
    # 当前规则就可以执行。
    trigger_terms: tuple[str, ...]

    # 追加到查询中的检索词。
    #
    # 这些词只能是触发词的别名或直接翻译，
    # 不能包含问题中没有出现的答案事实。
    expansion_terms: tuple[str, ...]

    # 组合规则要求这些概念全部出现在归一化查询中。
    #
    # 空元组表示当前规则没有附加条件，
    # 因此原来的普通规则不受影响。
    #
    # 例如充放电温度组合规则会设置：
    #
    # trigger_terms=("充电",)
    # required_terms=("放电", "温度")
    #
    # 只有三个概念都存在时才触发。
    required_terms: tuple[str, ...] = ()


@dataclass(
    frozen=True,
    slots=True,
)
class RewrittenRetrievalQuery:
    """一次确定性查询改写的完整审计结果。"""

    # 清理首尾空白后的用户原始问题。
    original_query: str

    # 经过错误码、型号、Unicode和数值归一化的查询。
    normalized_query: str

    # 最终交给Embedding和关键词检索的完整查询。
    rewritten_query: str

    # 本次实际追加的型号别名或双语检索词。
    added_terms: tuple[str, ...]

    # 本次真正改变查询的规则ID。
    applied_rule_ids: tuple[str, ...]

    # 当前使用的确定性规则版本。
    rewrite_version: str


# 经过人工确认的查询扩展规则。
#
# 这些规则只扩展原问题中已经出现的概念。
#
# 例如：
#
# “充电”可以扩展成charge和charging，
# 因为它们是同一个概念的中英文表达。
#
# 但是不能把“充电”直接扩展成某个温度范围，
# 因为温度范围属于需要从知识库检索的答案。
QUERY_REWRITE_RULES: tuple[
    QueryRewriteRule,
    ...,
] = (
    QueryRewriteRule(
        rule_id="model-dataman-260",
        trigger_terms=(
            "dataman-260",
        ),
        expansion_terms=(
            "DataMan 260",
            "DM260",
        ),
    ),
    QueryRewriteRule(
        rule_id=(
            "dataman-connector-context"
        ),

        # 这些词都表示连接件或接口。
        #
        # 中文、英文和J3接口代号
        # 出现任意一项都可以触发。
        trigger_terms=(
            "连接器",
            "插头",
            "插座",
            "connector",
            "j3",
        ),

        # 只有查询中同时存在DataMan 260型号
        # 才执行这组扩展。
        #
        # 这可以避免普通的“机器人连接器”问题
        # 被错误引导到DataMan设备手册。
        required_terms=(
            "dataman-260",
        ),

        # 扩展词只是连接器概念的英文同义表达，
        # 不包含连接是否正确或应如何操作的答案。
        expansion_terms=(
            "connector",
            "cable connector",
        ),
    ),
    QueryRewriteRule(
        rule_id=(
            "dataman-connector-precautions-context"
        ),

        # 只有问题本身正在询问检查、连接状态或插拔边界，
        # 才补充英文安全注意事项检索词。
        #
        # Planner在多轮执行中可能把原始任务中的
        # “不得带电插拔”压缩成“锁紧环检查要求”。
        # 这些表达仍然属于连接器安全检查语境，
        # 不能因为自然语言改写而失去手册注意事项检索词。
        trigger_terms=(
            "带电插拔",
            "插拔",
            "只读检查",
            "检查草案",
            "检查要求",
            "检查步骤",
            "可见间隙",
            "锁紧环",
            "未贴合",
            "inspection only",
            "visible gap",
            "locking ring",
            "not seated",
            "do not connect or disconnect",
        ),
        required_terms=(
            "dataman-260",
        ),

        # 这里只执行双语概念对齐。
        #
        # 具体的供电条件、线缆种类和操作限制
        # 仍必须由后续检索出的真实手册Chunk提供。
        expansion_terms=(
            "connect or disconnect",
            "precautions",
        ),
    ),
    QueryRewriteRule(
        rule_id="zh-precautions",
        trigger_terms=(
            "注意事项",
        ),
        expansion_terms=(
            "precautions",
        ),
    ),
    QueryRewriteRule(
        rule_id="zh-electrical-safety",
        trigger_terms=(
            "电气安全",
        ),
        expansion_terms=(
            "electrical safety",
        ),
    ),
    QueryRewriteRule(
        rule_id="zh-cable",
        trigger_terms=(
            "线缆",
            "电缆",
        ),
        expansion_terms=(
            "cable",
            "cables",
        ),
    ),
    QueryRewriteRule(
        rule_id="zh-high-current-wiring",
        trigger_terms=(
            "大电流线路",
        ),
        expansion_terms=(
            "high-current wiring",
        ),
    ),
    QueryRewriteRule(
        rule_id="zh-high-voltage-power",
        trigger_terms=(
            "高压电源",
        ),
        expansion_terms=(
            "high-voltage power sources",
        ),
    ),
        QueryRewriteRule(
        rule_id=(
            "zh-electrical-routing-context"
        ),

        # “线缆”和“电缆”是同一概念的不同表达，
        # 出现任意一个即可满足主触发条件。
        trigger_terms=(
            "线缆",
            "电缆",
        ),

        # 同时出现大电流线路和高压电源时，
        # 才把问题识别成布线安全场景。
        required_terms=(
            "大电流线路",
            "高压电源",
        ),

        # 这些内容是原问题的英文同义表达和章节词，
        # 不包含过电压、线路噪声等答案事实。
        expansion_terms=(
            "route cables and wires away",
            "precautions",
        ),
    ),
    QueryRewriteRule(
        rule_id="zh-charge",
        trigger_terms=(
            "充电",
        ),
        expansion_terms=(
            "charge",
            "charging",
        ),
    ),
    QueryRewriteRule(
        rule_id="zh-discharge",
        trigger_terms=(
            "放电",
        ),
        expansion_terms=(
            "discharge",
            "discharging",
        ),
    ),
    QueryRewriteRule(
        rule_id="zh-temperature",
        trigger_terms=(
            "温度",
        ),
        expansion_terms=(
            "temperature",
        ),
    ),
    QueryRewriteRule(
        rule_id="zh-range",
        trigger_terms=(
            "范围",
        ),
        expansion_terms=(
            "range",
        ),
    ),
        QueryRewriteRule(
        rule_id=(
            "zh-charge-discharge-temperature-context"
        ),

        # “充电”作为组合规则的主触发概念。
        trigger_terms=(
            "充电",
        ),

        # 必须同时询问放电和温度，
        # 才能执行充放电温度组合扩展。
        required_terms=(
            "放电",
            "温度",
        ),

        # 前两项保留“充电范围”和“放电范围”的
        # 短语结构，避免只追加彼此分散的单词。
        #
        # 最后一项对应英文手册中的章节语义，
        # 但不包含任何具体温度数值。
        expansion_terms=(
            "charge temperature range",
            "discharge temperature range",
            "charge and discharge warnings",
        ),
    ),
    QueryRewriteRule(
        rule_id="zh-reader-lens-cover",
        trigger_terms=(
            "镜头保护盖",
        ),
        expansion_terms=(
            "reader lens cover",
            "lens cover",
        ),
    ),
    QueryRewriteRule(
        rule_id="zh-dust",
        trigger_terms=(
            "积尘",
            "灰尘",
        ),
        expansion_terms=(
            "dust",
        ),
    ),
    QueryRewriteRule(
        rule_id="zh-cleaning",
        trigger_terms=(
            "清洁",
        ),
        expansion_terms=(
            "clean",
            "cleaning",
        ),
    ),
    QueryRewriteRule(
        rule_id="zh-network-recovery",
        trigger_terms=(
            "网络恢复",
        ),
        expansion_terms=(
            "network recovery",
        ),
    ),
    QueryRewriteRule(
        rule_id="zh-resume-task",
        trigger_terms=(
            "继续任务",
            "恢复任务",
        ),
        expansion_terms=(
            "resume task",
        ),
    ),
)


def _rule_matches(
    *,
    normalized_query: str,
    rule: QueryRewriteRule,
) -> bool:
    """判断归一化查询是否满足一条改写规则。"""

    # casefold()生成适合不区分大小写比较的文本。
    #
    # 它比lower()覆盖的Unicode大小写情况更多，
    # 适合处理中英文混合查询中的英文术语。
    comparison_query = (
        normalized_query.casefold()
    )

    # trigger_terms采用“任意一个命中”的关系。
    #
    # 例如：
    # trigger_terms=("线缆", "电缆")
    #
    # 出现其中任意一个都可以。
    trigger_matches = any(
        trigger.casefold()
        in comparison_query
        for trigger in rule.trigger_terms
    )

    # required_terms采用“全部命中”的关系。
    #
    # all(())的结果是True，
    # 所以没有附加条件的旧规则不会受到影响。
    required_terms_match = all(
        required.casefold()
        in comparison_query
        for required in rule.required_terms
    )

    # 主触发条件和所有必要条件必须同时满足。
    return (
        trigger_matches
        and required_terms_match
    )


def rewrite_retrieval_query(
    query: str,
) -> RewrittenRetrievalQuery:
    """归一化并扩展一个检索问题。"""

    # Python类型注解不会在运行时自动拒绝None或整数，
    # 因此公共函数仍要显式校验。
    if not isinstance(query, str):
        raise TypeError(
            "query必须是字符串"
        )

    # strip()删除用户问题首尾的空格、换行和Tab。
    cleaned_query = query.strip()

    if not cleaned_query:
        raise ValueError(
            "query不能为空"
        )

    # 先复用已经测试过的归一化模块。
    #
    # 例如：
    #
    # ERR_NET_4001 -> err-net-4001
    # DM260         -> dataman-260
    # −10 ℃         -> -10°c
    normalized = normalize_retrieval_text(
        cleaned_query
    )

    normalized_query = (
        normalized.normalized_text
    )

    added_terms: list[str] = []
    applied_rule_ids: list[str] = []

    # comparison_text用于检查某个扩展词是否已经存在。
    #
    # 后续每增加一个词，也会同步更新该字符串，
    # 从而避免不同规则重复追加相同内容。
    comparison_text = (
        normalized_query.casefold()
    )

    for rule in QUERY_REWRITE_RULES:
        if not _rule_matches(
            normalized_query=normalized_query,
            rule=rule,
        ):
            continue

        # 只有当前规则真正增加了词，
        # 才将其记录到applied_rule_ids。
        rule_added_term = False

        for expansion in rule.expansion_terms:
            cleaned_expansion = (
                expansion.strip()
            )

            expansion_key = (
                cleaned_expansion.casefold()
            )

            # 原问题或前面的规则已经包含这个词时，
            # 不再重复追加。
            if expansion_key in comparison_text:
                continue

            added_terms.append(
                cleaned_expansion
            )

            # 更新用于后续去重比较的文本。
            comparison_text = (
                f"{comparison_text} "
                f"{expansion_key}"
            )

            rule_added_term = True

        if rule_added_term:
            applied_rule_ids.append(
                rule.rule_id
            )

    # 保留完整归一化问题，然后在末尾追加扩展词。
    #
    # 不能只保留英文扩展词，否则可能丢失：
    #
    # 1. 原问题的具体条件；
    # 2. 错误码和数值；
    # 3. 中文知识库需要的原始表达。
    rewritten_query = " ".join(
        (
            normalized_query,
            *added_terms,
        )
    )

    return RewrittenRetrievalQuery(
        original_query=cleaned_query,
        normalized_query=normalized_query,
        rewritten_query=rewritten_query,
        added_terms=tuple(
            added_terms
        ),
        applied_rule_ids=tuple(
            applied_rule_ids
        ),
        rewrite_version=(
            DETERMINISTIC_QUERY_REWRITE_VERSION
        ),
    )
