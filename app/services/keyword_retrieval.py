"""轻量、确定性、可解释的关键词检索。

本模块负责：
1. 归一化查询和文档中的结构化检索词；
2. 提取错误码、型号、数值和普通关键词；
3. 为DocumentChunk建立内存关键词索引；
4. 计算可解释的关键词分数；
5. 返回按分数排序的Top-K结果。

本模块不调用LLM、Embedding、Chroma或网络服务。
它只处理调用方传入的DocumentChunk。
"""

# Iterable表示可以逐项遍历的数据来源。
#
# list、tuple和生成器都可以作为Iterable使用。
from collections.abc import Iterable

from dataclasses import dataclass
import re

from app.schemas.retrieval import (
    DocumentChunk,
    KeywordRetrievedChunk,
)
from app.services.lexical_normalization import (
    normalize_retrieval_text,
)

# RetrievalScopeFilter规定本次查询允许搜索的
# 文档ID和来源文件名范围。
from app.services.retrieval_scope import (
    RetrievalScopeFilter,
)


# 匹配普通ASCII英文词。
#
# 可以匹配：
# motor
# controller
# high-voltage
# dataman-260
#
# 结构化错误码和型号也可能被该正则匹配，
# 但后续会通过protected_terms排除，
# 避免同一个词被重复计入不同类别。
ASCII_TERM_PATTERN = re.compile(
    r"(?<![a-z0-9])"
    r"[a-z][a-z0-9]*"
    r"(?:-[a-z0-9]+)*"
    r"(?![a-z0-9])"
)


# 匹配连续的中文字符。
#
# \u3400-\u4dbf是CJK扩展A区；
# \u4e00-\u9fff是常用中日韩统一表意文字区。
CJK_SEQUENCE_PATTERN = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff]+"
)


# 对每段连续中文同时生成：
#
# 2个字符的bigram；
# 3个字符的trigram。
#
# 元组顺序也决定同一位置上先生成二字词，
# 再生成三字词。
CJK_NGRAM_SIZES = (
    2,
    3,
)


# 普通关键词停用词。
#
# 停用词是大量问题中频繁出现、
# 但对区分具体文档帮助较小的词。
#
# 这里只设置一个小型、明确的列表，
# 不使用过大的通用词库，避免误删
# “安全”“复位”“过温”等工程关键词。
LEXICAL_STOP_TERMS = frozenset(
    {
        # 英文常见功能词。
        "the",
        "and",
        "for",
        "with",
        "from",
        "into",
        "that",
        "this",
        "what",
        "which",
        "when",
        "where",
        "how",
        "should",
        "could",
        "would",
        "does",
        "are",
        "is",
        "to",
        "of",
        "in",
        "on",
        "or",
        "an",

        # 中文问题中常见、区分能力较弱的词。
        "什么",
        "哪些",
        "怎么",
        "如何",
        "是否",
        "应该",
        "需要",
        "可以",
        "根据",
        "进行",
        "一个",
        "这个",
    }
)


# 各类关键词的排序权重。
#
# 这些值不是概率，也不是最终回答置信度，
# 只用于关键词候选内部的相对排序。
IDENTIFIER_MATCH_WEIGHT = 12.0
MODEL_ALIAS_MATCH_WEIGHT = 8.0
NUMERIC_TERM_MATCH_WEIGHT = 6.0
LEXICAL_TERM_MATCH_WEIGHT = 0.5


# 普通词来自中文二字、三字词和英文词。
#
# 长查询会产生较多普通词，因此限制普通词
# 能够贡献的总分，防止它们压过精确错误码。
MAX_LEXICAL_MATCH_SCORE = 4.0


# 如果没有错误码、型号或数值命中，
# 至少需要两个普通词命中才接纳该Chunk。
#
# 关键词检索是向量检索的补充，
# 不需要依靠单个常见词召回所有文档。
MIN_LEXICAL_MATCH_COUNT = 2


@dataclass(
    frozen=True,
    slots=True,
)
class KeywordFeatures:
    """查询或文档经过处理后的关键词特征。"""

    # 完整的规范化文本，
    # 主要用于调试和测试。
    normalized_text: str

    # 结构化错误码或测试编号。
    identifiers: tuple[str, ...]

    # 产品型号规范名称。
    model_aliases: tuple[str, ...]

    # 保留符号和单位的数值词。
    numeric_terms: tuple[str, ...]

    # 普通英文词和中文N-gram。
    lexical_terms: tuple[str, ...]


@dataclass(
    frozen=True,
    slots=True,
)
class _IndexedKeywordChunk:
    """已经完成特征提取的内部Chunk索引项。"""

    # 保存完整Chunk快照。
    chunk: DocumentChunk

    # 使用frozenset保存各类特征，
    # 便于检索时执行高频的成员判断。
    identifiers: frozenset[str]
    model_aliases: frozenset[str]
    numeric_terms: frozenset[str]
    lexical_terms: frozenset[str]


@dataclass(
    frozen=True,
    slots=True,
)
class _KeywordCandidate:
    """尚未分配最终排名的关键词候选。"""

    chunk: DocumentChunk
    keyword_score: float

    matched_identifiers: tuple[str, ...]
    matched_model_aliases: tuple[str, ...]
    matched_numeric_terms: tuple[str, ...]
    matched_lexical_terms: tuple[str, ...]


def _unique_in_order(
    values: list[str],
) -> tuple[str, ...]:
    """去除重复关键词，同时保留首次出现顺序。"""

    # dict.fromkeys()把字符串作为字典键。
    #
    # 相同键只保留一次；
    # Python字典保留第一次插入顺序。
    return tuple(
        dict.fromkeys(values)
    )


def _extract_lexical_terms(
    *,
    normalized_text: str,
    protected_terms: frozenset[str],
) -> tuple[str, ...]:
    """提取普通英文词及中文二字、三字词。"""

    # 每个候选保存：
    #
    # 1. 在原文本中的起始位置；
    # 2. 词项长度；
    # 3. 词项正文。
    #
    # 保存位置后可以在英文和中文候选合并时
    # 恢复稳定的文本顺序。
    candidates: list[
        tuple[int, int, str]
    ] = []

    for match in ASCII_TERM_PATTERN.finditer(
        normalized_text
    ):
        term = match.group(0)

        # 单字符英文通常区分能力很弱。
        #
        # 最大长度与RetrievalTerm契约保持一致。
        if not 2 <= len(term) <= 200:
            continue

        # 错误码、型号和数值拥有独立高权重类别，
        # 不能再作为普通词重复计分。
        if term in protected_terms:
            continue

        if term in LEXICAL_STOP_TERMS:
            continue

        candidates.append(
            (
                match.start(),
                len(term),
                term,
            )
        )

    # 对每段连续中文生成二字和三字N-gram。
    for sequence_match in (
        CJK_SEQUENCE_PATTERN.finditer(
            normalized_text
        )
    ):
        sequence = sequence_match.group(0)
        sequence_start = sequence_match.start()

        for ngram_size in CJK_NGRAM_SIZES:
            # 当前中文片段短于ngram_size时，
            # 不会产生任何有效切片。
            if len(sequence) < ngram_size:
                continue

            # 例如：
            #
            # sequence="电机过温"
            # ngram_size=2
            #
            # start依次为0、1、2，
            # 得到电机、机过、过温。
            for start in range(
                len(sequence)
                - ngram_size
                + 1
            ):
                term = sequence[
                    start:
                    start + ngram_size
                ]

                if term in LEXICAL_STOP_TERMS:
                    continue

                candidates.append(
                    (
                        sequence_start + start,
                        ngram_size,
                        term,
                    )
                )

    # 先按原文位置排序。
    #
    # 同一位置存在二字和三字词时，
    # 长度较短的二字词排在前面。
    #
    # term作为最后的稳定排序条件。
    ordered_candidates = sorted(
        candidates,
        key=lambda item: (
            item[0],
            item[1],
            item[2],
        ),
    )

    return _unique_in_order(
        [
            term
            for _, _, term
            in ordered_candidates
        ]
    )


def extract_keyword_features(
    text: str,
) -> KeywordFeatures:
    """把查询或文档文本转换成关键词特征。"""

    # 先复用已经通过独立离线测试的
    # 确定性归一化函数。
    normalized = normalize_retrieval_text(
        text
    )

    # 这些词已经进入高权重结构化类别。
    #
    # frozenset是不可变集合，
    # 适合执行高频的"in"成员判断。
    protected_terms = frozenset(
        normalized.identifiers
        + normalized.model_aliases
        + normalized.numeric_terms
    )

    lexical_terms = _extract_lexical_terms(
        normalized_text=(
            normalized.normalized_text
        ),
        protected_terms=protected_terms,
    )

    return KeywordFeatures(
        normalized_text=(
            normalized.normalized_text
        ),
        identifiers=normalized.identifiers,
        model_aliases=(
            normalized.model_aliases
        ),
        numeric_terms=(
            normalized.numeric_terms
        ),
        lexical_terms=lexical_terms,
    )


def _match_terms(
    *,
    query_terms: tuple[str, ...],
    indexed_terms: frozenset[str],
) -> tuple[str, ...]:
    """返回查询和Chunk共同拥有的关键词。"""

    # 按查询中的顺序返回命中词，
    # 使日志和评测报告更容易阅读。
    #
    # query_terms本身已经去重，
    # 因此返回结果也不会包含重复项。
    return tuple(
        term
        for term in query_terms
        if term in indexed_terms
    )


def _calculate_keyword_score(
    *,
    matched_identifiers: tuple[str, ...],
    matched_model_aliases: tuple[str, ...],
    matched_numeric_terms: tuple[str, ...],
    matched_lexical_terms: tuple[str, ...],
) -> float:
    """根据四类命中词计算确定性关键词分数。"""

    # 普通词按数量加分，但设置总分上限。
    #
    # 例如命中12个普通词：
    #
    # 12 × 0.5 = 6
    #
    # 由于上限为4，最终只贡献4分。
    lexical_score = min(
        (
            len(matched_lexical_terms)
            * LEXICAL_TERM_MATCH_WEIGHT
        ),
        MAX_LEXICAL_MATCH_SCORE,
    )

    return (
        len(matched_identifiers)
        * IDENTIFIER_MATCH_WEIGHT
        + len(matched_model_aliases)
        * MODEL_ALIAS_MATCH_WEIGHT
        + len(matched_numeric_terms)
        * NUMERIC_TERM_MATCH_WEIGHT
        + lexical_score
    )


class KeywordRetriever:
    """对DocumentChunk执行轻量关键词Top-K检索。"""

    def __init__(
        self,
        *,
        chunks: Iterable[DocumentChunk],
    ) -> None:
        """为一批Chunk建立不可变的内存关键词索引。"""

        indexed_chunks: list[
            _IndexedKeywordChunk
        ] = []

        # Chunk ID是Chunk的唯一身份。
        #
        # 如果同一个ID在索引中出现两次，
        # 后续RRF会无法判断它是一个候选还是两个候选。
        seen_chunk_ids: set[str] = set()

        for position, chunk in enumerate(
            chunks
        ):
            # 类型注解不会自动执行运行时校验，
            # 因此这里拒绝非DocumentChunk对象。
            if not isinstance(
                chunk,
                DocumentChunk,
            ):
                raise TypeError(
                    "关键词索引中的第 "
                    f"{position} 项必须是DocumentChunk"
                )

            if chunk.chunk_id in seen_chunk_ids:
                raise ValueError(
                    "关键词索引不能包含重复的"
                    f"chunk_id：{chunk.chunk_id}"
                )

            seen_chunk_ids.add(
                chunk.chunk_id
            )

            # Pydantic的model_copy(deep=True)
            # 创建一个深复制快照。
            #
            # 调用方以后修改原始Chunk时，
            # 不会造成索引特征和Chunk正文不一致。
            chunk_snapshot = chunk.model_copy(
                deep=True
            )

            # 文件名可以提供产品型号或文档主题；
            # content是真正的知识正文。
            #
            # 不加入document_id和chunk_id，
            # 因为它们是哈希或内部编号，没有语义。
            #
            # 不加入page_or_section，
            # 避免"page: 5"中的页码5被误判成
            # 文档正文里的工程数值。
            searchable_text = "\n".join(
                (
                    chunk_snapshot.source_file,
                    chunk_snapshot.content,
                )
            )

            features = extract_keyword_features(
                searchable_text
            )

            indexed_chunks.append(
                _IndexedKeywordChunk(
                    chunk=chunk_snapshot,
                    identifiers=frozenset(
                        features.identifiers
                    ),
                    model_aliases=frozenset(
                        features.model_aliases
                    ),
                    numeric_terms=frozenset(
                        features.numeric_terms
                    ),
                    lexical_terms=frozenset(
                        features.lexical_terms
                    ),
                )
            )

        # 使用tuple保存索引，表示创建完成后
        # 不再增删内部索引项。
        self._indexed_chunks = tuple(
            indexed_chunks
        )

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 3,

        # None表示本次查询允许搜索整个关键词索引。
        #
        # 提供RetrievalScopeFilter时，
        # 只有满足该范围的Chunk才能参与打分。
        retrieval_scope: (
            RetrievalScopeFilter | None
        ) = None,
    ) -> list[KeywordRetrievedChunk]:
        """返回与查询关键词最匹配的Top-K Chunk。"""

        # bool是int的子类：
        #
        # isinstance(True, int)
        # → True
        #
        # 但top_k=True没有清晰的业务含义，
        # 因此要先单独排除bool。
        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
        ):
            raise TypeError(
                "top_k必须是整数"
            )

        if top_k < 1:
            raise ValueError(
                "top_k必须大于或等于1"
            )
        # 类型注解不会在Python运行时自动阻止
        # 普通dict、list或其他对象进入函数。
        #
        # None是合法值，表示不限制检索范围。
        if (
            retrieval_scope is not None
            and not isinstance(
                retrieval_scope,
                RetrievalScopeFilter,
            )
        ):
            raise TypeError(
                "retrieval_scope必须是"
                "RetrievalScopeFilter或None"
            )


        # 查询和文档必须使用完全相同的
        # 归一化与特征提取规则。
        query_features = (
            extract_keyword_features(
                query
            )
        )

        # 如果查询只包含被过滤的停用词或
        # 单字符英文词，就没有可比较特征。
        if not any(
            (
                query_features.identifiers,
                query_features.model_aliases,
                query_features.numeric_terms,
                query_features.lexical_terms,
            )
        ):
            return []

        candidates: list[
            _KeywordCandidate
        ] = []

        for indexed_chunk in (
            self._indexed_chunks
        ):
            # 范围过滤必须发生在：
            #
            # 1. 关键词命中计算之前；
            # 2. 关键词分数计算之前；
            # 3. 候选排序之前；
            # 4. Top-K截取之前。
            #
            # 这样范围外高分Chunk不会占据
            # 范围内候选的Top-K名额。
            if (
                retrieval_scope is not None
                and not retrieval_scope.matches(
                    indexed_chunk.chunk
                )
            ):
                continue

            matched_identifiers = (
                _match_terms(
                    query_terms=(
                        query_features.identifiers
                    ),
                    indexed_terms=(
                        indexed_chunk.identifiers
                    ),
                )
            )

            matched_model_aliases = (
                _match_terms(
                    query_terms=(
                        query_features.model_aliases
                    ),
                    indexed_terms=(
                        indexed_chunk.model_aliases
                    ),
                )
            )

            matched_numeric_terms = (
                _match_terms(
                    query_terms=(
                        query_features.numeric_terms
                    ),
                    indexed_terms=(
                        indexed_chunk.numeric_terms
                    ),
                )
            )

            matched_lexical_terms = (
                _match_terms(
                    query_terms=(
                        query_features.lexical_terms
                    ),
                    indexed_terms=(
                        indexed_chunk.lexical_terms
                    ),
                )
            )

            # 错误码、型号和数值都属于结构化命中。
            has_structured_match = any(
                (
                    matched_identifiers,
                    matched_model_aliases,
                    matched_numeric_terms,
                )
            )

            # 没有结构化命中时，
            # 至少需要两个普通关键词重合。
            if (
                not has_structured_match
                and len(matched_lexical_terms)
                < MIN_LEXICAL_MATCH_COUNT
            ):
                continue

            keyword_score = (
                _calculate_keyword_score(
                    matched_identifiers=(
                        matched_identifiers
                    ),
                    matched_model_aliases=(
                        matched_model_aliases
                    ),
                    matched_numeric_terms=(
                        matched_numeric_terms
                    ),
                    matched_lexical_terms=(
                        matched_lexical_terms
                    ),
                )
            )

            candidates.append(
                _KeywordCandidate(
                    chunk=indexed_chunk.chunk,
                    keyword_score=keyword_score,
                    matched_identifiers=(
                        matched_identifiers
                    ),
                    matched_model_aliases=(
                        matched_model_aliases
                    ),
                    matched_numeric_terms=(
                        matched_numeric_terms
                    ),
                    matched_lexical_terms=(
                        matched_lexical_terms
                    ),
                )
            )

        # Python的sorted()返回一个新列表，
        # 不会修改原始candidates。
        #
        # 负号表示数值越大越靠前。
        candidates = sorted(
            candidates,
            key=lambda item: (
                -item.keyword_score,

                # 总分相同时优先精确结构化命中。
                -len(
                    item.matched_identifiers
                ),
                -len(
                    item.matched_model_aliases
                ),
                -len(
                    item.matched_numeric_terms
                ),

                # 然后比较普通词命中数量。
                -len(
                    item.matched_lexical_terms
                ),

                # 所有业务分数相同时，
                # 使用chunk_id保证排序可复现。
                item.chunk.chunk_id,
            ),
        )

        results: list[
            KeywordRetrievedChunk
        ] = []

        # candidates[:top_k]最多取得top_k项；
        # 如果实际候选更少，就返回实际数量。
        for rank, candidate in enumerate(
            candidates[:top_k],
            start=1,
        ):
            results.append(
                KeywordRetrievedChunk(
                    # 返回深复制，避免调用方修改
                    # KeywordRetriever内部保存的Chunk快照。
                    chunk=(
                        candidate.chunk.model_copy(
                            deep=True
                        )
                    ),
                    keyword_score=(
                        candidate.keyword_score
                    ),
                    rank=rank,
                    matched_identifiers=(
                        candidate.matched_identifiers
                    ),
                    matched_model_aliases=(
                        candidate.matched_model_aliases
                    ),
                    matched_numeric_terms=(
                        candidate.matched_numeric_terms
                    ),
                    matched_lexical_terms=(
                        candidate.matched_lexical_terms
                    ),
                )
            )

        return results