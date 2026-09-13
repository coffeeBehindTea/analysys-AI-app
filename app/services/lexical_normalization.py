"""为关键词检索提供确定性的文本归一化能力。

本模块不调用LLM、Embedding或数据库。

它只负责把错误码、产品型号和数值单位等
具有精确含义的文本转换成稳定形式，使查询和
文档使用相同规则后可以执行可靠的关键词匹配。
"""

from collections.abc import Iterable
from dataclasses import dataclass
import re
import unicodedata


# 将不同的Unicode横线统一成ASCII连字符"-"。
#
# 例如：
# ERR–SAF–1002  使用en dash；
# −10°C         使用数学负号。
#
# 如果不统一，这些字符虽然看起来相似，
# 但在字符串比较时并不相等。
DASH_TRANSLATION = str.maketrans(
    {
        "‐": "-",
        "‑": "-",
        "‒": "-",
        "–": "-",
        "—": "-",
        "−": "-",
    }
)


# \s+表示一个或多个连续空白字符，
# 包括普通空格、Tab和换行。
WHITESPACE_PATTERN = re.compile(r"\s+")


# 匹配项目中的结构化错误码和测试编号。
#
# 可以识别：
# ERR-SAF-1002
# err saf 1002
# ERR_SAF_1002
# ERRSAF1002
# TEST-NET-001
STRUCTURED_IDENTIFIER_PATTERN = re.compile(
    # 前面不能紧邻英文字母或数字，
    # 防止从另一个更长的单词中间开始匹配。
    r"(?<![a-z0-9])"

    # prefix只允许err或test。
    r"(?P<prefix>err|test)"

    # 各部分之间允许没有分隔符，
    # 或使用空格、下划线、连字符。
    r"[\s_-]*"

    # family对应saf、net、drv、nav等类别。
    r"(?P<family>[a-z]{2,12})"

    r"[\s_-]*"

    # 当前项目编号使用3到6位数字。
    r"(?P<number>\d{3,6})"

    # 后面不能继续紧邻英文字母或数字。
    r"(?![a-z0-9])",

    # IGNORECASE表示匹配时不区分大小写。
    re.IGNORECASE,
)


# 产品型号别名表。
#
# 每一项包含：
# 1. 统一后的规范名称；
# 2. 可以匹配原始文本不同写法的正则表达式。
#
# 后续如果加入更多产品，可以继续向该元组追加规则。
MODEL_ALIAS_PATTERNS: tuple[
    tuple[str, re.Pattern[str]],
    ...
] = (
    (
        "dataman-260",
        re.compile(
            r"(?<![a-z0-9])"
            r"(?:data[\s_-]*man|dm)"
            r"[\s_-]*260"
            r"(?![a-z0-9])",
            re.IGNORECASE,
        ),
    ),
)


# 匹配需要保留精确形式的数值和单位。
#
# 关键边界：
#
# 数字后的空白只有在后面确实存在单位时
# 才属于当前匹配。
#
# 例如：
#
# 1500 ms
# → 匹配完整的"1500 ms"
#
# 4001 网络恢复
# → 只匹配"4001"
# → 保留数字后的空格
NUMERIC_TERM_PATTERN = re.compile(
    r"(?<![a-z0-9])"
    r"(?P<number>[+-]?\d+(?:\.\d+)?)"

    # 整个非捕获组都是可选的。
    #
    # 组内的unit必须存在，
    # 因而\s*不会单独消费数字后的空白。
    r"(?:"
    r"\s*"
    r"(?P<unit>"
    r"°\s*c"
    r"|ms"
    r"|mm"
    r"|cm"
    r"|km"
    r"|kg"
    r"|kw"
    r"|hz"
    r"|s"
    r"|m"
    r"|v"
    r"|a"
    r"|w"
    r"|%"
    r")"
    r")?"

    r"(?![a-z0-9])",
    re.IGNORECASE,
)


@dataclass(
    frozen=True,
    slots=True,
)
class NormalizedRetrievalText:
    """一段文本经过确定性归一化后的结果。"""

    # 用于后续普通关键词切分和匹配的完整文本。
    normalized_text: str

    # 已经统一格式的错误码或测试编号。
    #
    # 例如：
    # ERR_NET_4001
    # → err-net-4001
    identifiers: tuple[str, ...]

    # 已经统一的产品型号别名。
    #
    # 例如：
    # DM260
    # → dataman-260
    model_aliases: tuple[str, ...]

    # 保留正负号、数值和单位的精确数值词。
    #
    # 例如：
    # −10 ℃
    # → -10°c
    numeric_terms: tuple[str, ...]


def _unique_in_order(
    values: Iterable[str],
) -> tuple[str, ...]:
    """去除重复值，同时保留第一次出现的顺序。"""

    # Python字典会保留键的插入顺序。
    #
    # dict.fromkeys()把每个字符串放入字典键，
    # 相同键只保留一次。
    return tuple(
        dict.fromkeys(values)
    )


def _canonical_identifier(
    match: re.Match[str],
) -> str:
    """把一个结构化编号转换成统一的连字符格式。"""

    # group("prefix")读取正则中的命名捕获组。
    prefix = match.group("prefix").casefold()
    family = match.group("family").casefold()
    number = match.group("number")

    return f"{prefix}-{family}-{number}"


def _canonical_numeric_term(
    match: re.Match[str],
) -> str:
    """统一数值与单位之间的空白和大小写。"""

    number = match.group("number")
    unit = match.group("unit")

    # 没有单位时只返回数字本身。
    if unit is None:
        return number

    # 删除单位内部的空白。
    #
    # 例如：
    # -10° c
    # → -10°c
    normalized_unit = WHITESPACE_PATTERN.sub(
        "",
        unit.casefold(),
    )

    return number + normalized_unit


def _span_overlaps(
    span: tuple[int, int],
    protected_spans: list[tuple[int, int]],
) -> bool:
    """判断字符范围是否与已有结构化词范围重叠。"""

    start, end = span

    return any(
        start < protected_end
        and protected_start < end
        for protected_start, protected_end
        in protected_spans
    )


def normalize_retrieval_text(
    text: str,
) -> NormalizedRetrievalText:
    """归一化查询或文档正文中的精确检索词。"""

    # 类型注解不会阻止普通Python代码传入非字符串，
    # 因此这里执行运行时防御。
    if not isinstance(text, str):
        raise TypeError(
            "待归一化内容必须是字符串"
        )

    cleaned_text = text.strip()

    if not cleaned_text:
        raise ValueError(
            "待归一化内容不能为空"
        )

    # NFKC是Unicode兼容规范化。
    #
    # 例如：
    # 全角字符ＥＲＲ
    # → ERR
    #
    # 摄氏度符号℃也会被统一成° C形式。
    base_text = unicodedata.normalize(
        "NFKC",
        cleaned_text,
    )

    # 统一各种横线后，再执行不区分大小写的转换。
    #
    # casefold()比lower()更适合Unicode文本的
    # 无大小写比较。
    base_text = (
        base_text
        .translate(DASH_TRANSLATION)
        .casefold()
    )

    # 将连续空格、换行和Tab压缩成一个普通空格。
    base_text = WHITESPACE_PATTERN.sub(
        " ",
        base_text,
    ).strip()

    identifier_matches = list(
        STRUCTURED_IDENTIFIER_PATTERN.finditer(
            base_text
        )
    )

    identifiers = _unique_in_order(
        _canonical_identifier(match)
        for match in identifier_matches
    )

    # 保存产品型号的规范名称及其匹配对象。
    #
    # 后面需要match.span()，以防把型号中的260
    # 再次当成独立数值关键词。
    model_matches = [
        (canonical_name, match)
        for canonical_name, pattern
        in MODEL_ALIAS_PATTERNS
        for match in pattern.finditer(base_text)
    ]

    # 如果以后存在多组型号规则，
    # 按型号在原文中的位置排序。
    model_matches.sort(
        key=lambda item: item[1].start()
    )

    model_aliases = _unique_in_order(
        canonical_name
        for canonical_name, _ in model_matches
    )

    # 错误码中的4001和型号中的260
    # 已经属于更精确的结构化词，
    # 不应再次作为普通数值重复计分。
    protected_spans = [
        match.span()
        for match in identifier_matches
    ] + [
        match.span()
        for _, match in model_matches
    ]

    numeric_matches = [
        match
        for match
        in NUMERIC_TERM_PATTERN.finditer(
            base_text
        )
        if not _span_overlaps(
            match.span(),
            protected_spans,
        )
    ]

    numeric_terms = _unique_in_order(
        _canonical_numeric_term(match)
        for match in numeric_matches
    )

    # 在完整文本中也执行相同替换，
    # 供后续普通关键词切分器使用。
    normalized_text = (
        STRUCTURED_IDENTIFIER_PATTERN.sub(
            _canonical_identifier,
            base_text,
        )
    )

    for canonical_name, pattern in (
        MODEL_ALIAS_PATTERNS
    ):
        normalized_text = pattern.sub(
            canonical_name,
            normalized_text,
        )

    normalized_text = NUMERIC_TERM_PATTERN.sub(
        _canonical_numeric_term,
        normalized_text,
    )

    normalized_text = WHITESPACE_PATTERN.sub(
        " ",
        normalized_text,
    ).strip()

    return NormalizedRetrievalText(
        normalized_text=normalized_text,
        identifiers=identifiers,
        model_aliases=model_aliases,
        numeric_terms=numeric_terms,
    )