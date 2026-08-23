"""对混合检索Top-K执行可解释的证据门控。

本模块负责：

1. 保存带版本号的门控策略参数；
2. 校验混合候选列表的运行时契约；
3. 综合向量、错误码、型号和关键词信号；
4. 返回可审计的放行或拒答决定。

本模块不生成Embedding、不执行检索、
不调用LLM，也不直接生成HTTP响应。
"""

from dataclasses import dataclass

# isfinite()用于拒绝NaN、正无穷和负无穷。
from math import isfinite

# Literal把门控原因限制在明确集合中。
from typing import Literal

from app.schemas.retrieval import (
    HybridRetrievedChunk,
)


# 门控策略版本必须进入后续评测报告。
#
# 只要阈值、规则顺序或信号定义改变，
# 就应该创建新版本，而不是静默覆盖旧结果。
HYBRID_EVIDENCE_GATE_VERSION = (
    "hybrid-evidence-gate-v1"
)


HybridEvidenceGateReason = Literal[
    "no_candidates",
    "exact_identifier_support",
    "model_and_lexical_support",
    "general_dual_path_support",
    "insufficient_combined_support",
]


def _validate_similarity_threshold(
    *,
    value: object,
    name: str,
) -> float:
    """校验并返回0到1之间的有限浮点阈值。"""

    # bool是int的子类：
    #
    # isinstance(True, int) == True
    #
    # 但True不应被当作相似度1.0使用，
    # 因此必须单独拒绝。
    if (
        isinstance(value, bool)
        or not isinstance(
            value,
            (int, float),
        )
    ):
        raise TypeError(
            f"{name}必须是数值"
        )

    normalized_value = float(value)

    if not isfinite(normalized_value):
        raise ValueError(
            f"{name}必须是有限数值"
        )

    if not 0.0 <= normalized_value <= 1.0:
        raise ValueError(
            f"{name}必须在0到1之间"
        )

    return normalized_value


@dataclass(
    frozen=True,
    slots=True,
)
class HybridEvidenceGatePolicy:
    """一次混合证据门控使用的固定参数。"""

    version: str = (
        HYBRID_EVIDENCE_GATE_VERSION
    )

    # 普通查询需要达到的向量相似度。
    #
    # 同时还必须存在具体内容词命中，
    # 不能只凭向量分数放行。
    general_min_vector_similarity: float = 0.50

    # 错误码是高精度结构化标识符，
    # 所以可以使用略低的向量阈值。
    identifier_min_vector_similarity: float = 0.45

    # 产品型号本身不足以回答问题。
    #
    # 型号必须和至少若干具体主题词共同出现，
    # 才能使用这个较低阈值。
    model_min_vector_similarity: float = 0.48

    # 例如：
    #
    # dataman-260 + cable + high-voltage
    #
    # 型号之外至少需要两个具体主题词。
    min_model_lexical_matches: int = 2

    def __post_init__(self) -> None:
        """在不可变对象创建完成后校验字段。"""

        if not isinstance(
            self.version,
            str,
        ):
            raise TypeError(
                "version必须是字符串"
            )

        cleaned_version = self.version.strip()

        if not cleaned_version:
            raise ValueError(
                "version不能为空"
            )

        general_threshold = (
            _validate_similarity_threshold(
                value=(
                    self.general_min_vector_similarity
                ),
                name=(
                    "general_min_vector_similarity"
                ),
            )
        )

        identifier_threshold = (
            _validate_similarity_threshold(
                value=(
                    self.identifier_min_vector_similarity
                ),
                name=(
                    "identifier_min_vector_similarity"
                ),
            )
        )

        model_threshold = (
            _validate_similarity_threshold(
                value=(
                    self.model_min_vector_similarity
                ),
                name=(
                    "model_min_vector_similarity"
                ),
            )
        )

        if (
            isinstance(
                self.min_model_lexical_matches,
                bool,
            )
            or not isinstance(
                self.min_model_lexical_matches,
                int,
            )
        ):
            raise TypeError(
                "min_model_lexical_matches"
                "必须是整数"
            )

        if self.min_model_lexical_matches < 1:
            raise ValueError(
                "min_model_lexical_matches"
                "必须大于或等于1"
            )

        # 两种结构化信号是通用规则的补充路径，
        # 因而它们的阈值不应高于通用阈值。
        if identifier_threshold > general_threshold:
            raise ValueError(
                "identifier_min_vector_similarity"
                "不能大于"
                "general_min_vector_similarity"
            )

        if model_threshold > general_threshold:
            raise ValueError(
                "model_min_vector_similarity"
                "不能大于"
                "general_min_vector_similarity"
            )

        # frozen=True禁止普通字段赋值。
        #
        # __post_init__中使用object.__setattr__，
        # 将字符串清理结果和int转float结果
        # 写回这个正在初始化的不可变对象。
        object.__setattr__(
            self,
            "version",
            cleaned_version,
        )
        object.__setattr__(
            self,
            "general_min_vector_similarity",
            general_threshold,
        )
        object.__setattr__(
            self,
            "identifier_min_vector_similarity",
            identifier_threshold,
        )
        object.__setattr__(
            self,
            "model_min_vector_similarity",
            model_threshold,
        )


@dataclass(
    frozen=True,
    slots=True,
)
class HybridEvidenceGateDecision:
    """一次门控判断的可审计结果。"""

    accepted: bool
    reason: HybridEvidenceGateReason
    policy_version: str

    evaluated_candidate_count: int

    # RRF分数只用于记录融合排名，
    # 不把它解释成回答置信度。
    top_rrf_score: float | None

    # Top-1候选真正的向量相似度。
    top_vector_similarity: float | None

    # Top-K中最大的向量相似度，
    # 可能来自非Top-1融合候选。
    max_vector_similarity: float | None

    # 同时进入向量和关键词路径的候选数量。
    dual_path_candidate_count: int

    # 含有精确错误码命中的候选数量。
    identifier_candidate_count: int

    # 同时含型号和足够主题词的候选数量。
    model_lexical_candidate_count: int

    # 真正触发放行规则的Chunk ID。
    #
    # 拒绝时必须为空元组。
    supporting_chunk_ids: tuple[str, ...]


def _validate_and_order_candidates(
    retrieved_chunks: object,
) -> list[HybridRetrievedChunk]:
    """校验候选容器、类型、ID和连续排名。"""

    if not isinstance(
        retrieved_chunks,
        list,
    ):
        raise TypeError(
            "retrieved_chunks必须是列表"
        )

    seen_chunk_ids: set[str] = set()

    for position, candidate in enumerate(
        retrieved_chunks
    ):
        if not isinstance(
            candidate,
            HybridRetrievedChunk,
        ):
            raise TypeError(
                "retrieved_chunks中的第 "
                f"{position} 项必须是"
                "HybridRetrievedChunk"
            )

        chunk_id = candidate.chunk.chunk_id

        if chunk_id in seen_chunk_ids:
            raise ValueError(
                "retrieved_chunks"
                "不能包含重复的chunk_id："
                f"{chunk_id}"
            )

        seen_chunk_ids.add(chunk_id)

    # 不要求调用方必须提前排好顺序，
    # 但必须拥有合法且无缺口的rank。
    ordered_candidates = sorted(
        retrieved_chunks,
        key=lambda candidate: candidate.rank,
    )

    actual_ranks = [
        candidate.rank
        for candidate in ordered_candidates
    ]
    expected_ranks = list(
        range(
            1,
            len(ordered_candidates) + 1,
        )
    )

    if actual_ranks != expected_ranks:
        raise ValueError(
            "retrieved_chunks的rank"
            "必须从1开始连续"
        )

    return ordered_candidates


def _candidate_has_general_keyword_support(
    candidate: HybridRetrievedChunk,
) -> bool:
    """判断候选是否命中能描述问题内容的关键词。"""

    # 型号名称单独存在时不能证明：
    #
    # - 具体子型号；
    # - 设备序列号；
    # - 真实部署情况；
    # - 当前运行状态。
    #
    # 因此这里不把matched_model_aliases
    # 单独算作通用内容支持。
    return any(
        (
            candidate.matched_identifiers,
            candidate.matched_numeric_terms,
            candidate.matched_lexical_terms,
        )
    )


def _candidate_has_model_lexical_support(
    *,
    candidate: HybridRetrievedChunk,
    policy: HybridEvidenceGatePolicy,
) -> bool:
    """判断型号是否与足够数量的主题词共同命中。"""

    return bool(
        candidate.matched_model_aliases
    ) and (
        len(candidate.matched_lexical_terms)
        >= policy.min_model_lexical_matches
    )


def _make_decision(
    *,
    accepted: bool,
    reason: HybridEvidenceGateReason,
    policy: HybridEvidenceGatePolicy,
    ordered_candidates: list[
        HybridRetrievedChunk
    ],
    supporting_candidates: tuple[
        HybridRetrievedChunk,
        ...,
    ] = (),
) -> HybridEvidenceGateDecision:
    """根据候选和放行来源构造审计决定。"""

    if ordered_candidates:
        top_candidate = ordered_candidates[0]
        top_rrf_score: float | None = (
            top_candidate.rrf_score
        )
        top_vector_similarity = (
            top_candidate.vector_similarity
        )
    else:
        top_rrf_score = None
        top_vector_similarity = None

    vector_similarities = [
        candidate.vector_similarity
        for candidate in ordered_candidates
        if candidate.vector_similarity is not None
    ]

    max_vector_similarity = (
        max(vector_similarities)
        if vector_similarities
        else None
    )

    dual_path_candidate_count = sum(
        1
        for candidate in ordered_candidates
        if (
            candidate.vector_rank is not None
            and candidate.keyword_rank is not None
        )
    )

    identifier_candidate_count = sum(
        1
        for candidate in ordered_candidates
        if candidate.matched_identifiers
    )

    model_lexical_candidate_count = sum(
        1
        for candidate in ordered_candidates
        if _candidate_has_model_lexical_support(
            candidate=candidate,
            policy=policy,
        )
    )

    return HybridEvidenceGateDecision(
        accepted=accepted,
        reason=reason,
        policy_version=policy.version,
        evaluated_candidate_count=len(
            ordered_candidates
        ),
        top_rrf_score=top_rrf_score,
        top_vector_similarity=(
            top_vector_similarity
        ),
        max_vector_similarity=(
            max_vector_similarity
        ),
        dual_path_candidate_count=(
            dual_path_candidate_count
        ),
        identifier_candidate_count=(
            identifier_candidate_count
        ),
        model_lexical_candidate_count=(
            model_lexical_candidate_count
        ),
        supporting_chunk_ids=tuple(
            candidate.chunk.chunk_id
            for candidate in supporting_candidates
        ),
    )


def evaluate_hybrid_evidence_gate(
    *,
    retrieved_chunks: object,
    policy: HybridEvidenceGatePolicy,
) -> HybridEvidenceGateDecision:
    """综合Top-K信号决定是否允许进入回答阶段。"""

    if not isinstance(
        policy,
        HybridEvidenceGatePolicy,
    ):
        raise TypeError(
            "policy必须是"
            "HybridEvidenceGatePolicy"
        )

    ordered_candidates = (
        _validate_and_order_candidates(
            retrieved_chunks
        )
    )

    if not ordered_candidates:
        return _make_decision(
            accepted=False,
            reason="no_candidates",
            policy=policy,
            ordered_candidates=[],
        )

    # 第一条放行路径：精确错误码。
    #
    # 它扫描全部Top-K，而不是只检查Top-1。
    identifier_supporting_candidates = tuple(
        candidate
        for candidate in ordered_candidates
        if (
            candidate.matched_identifiers
            and candidate.vector_similarity
            is not None
            and candidate.vector_similarity
            >= (
                policy
                .identifier_min_vector_similarity
            )
        )
    )

    if identifier_supporting_candidates:
        return _make_decision(
            accepted=True,
            reason="exact_identifier_support",
            policy=policy,
            ordered_candidates=(
                ordered_candidates
            ),
            supporting_candidates=(
                identifier_supporting_candidates
            ),
        )

    # 第二条放行路径：型号 + 具体主题词。
    #
    # 型号不能单独放行，避免把产品手册系列名称
    # 误认为真实部署子型号或序列号。
    model_supporting_candidates = tuple(
        candidate
        for candidate in ordered_candidates
        if (
            _candidate_has_model_lexical_support(
                candidate=candidate,
                policy=policy,
            )
            and candidate.vector_similarity
            is not None
            and candidate.vector_similarity
            >= (
                policy
                .model_min_vector_similarity
            )
        )
    )

    if model_supporting_candidates:
        return _make_decision(
            accepted=True,
            reason="model_and_lexical_support",
            policy=policy,
            ordered_candidates=(
                ordered_candidates
            ),
            supporting_candidates=(
                model_supporting_candidates
            ),
        )

    # 第三条放行路径：普通双路支持。
    #
    # 这里使用融合排名第一的候选，
    # 同时要求：
    #
    # 1. 它进入向量路径；
    # 2. 它达到通用向量阈值；
    # 3. 它拥有错误码、数值或普通主题词命中。
    #
    # 第3项也意味着它进入了关键词路径，
    # 因为HybridRetrievedChunk契约禁止
    # 没有keyword_rank却携带命中解释。
    top_candidate = ordered_candidates[0]

    if (
        top_candidate.vector_similarity
        is not None
        and top_candidate.vector_similarity
        >= (
            policy
            .general_min_vector_similarity
        )
        and _candidate_has_general_keyword_support(
            top_candidate
        )
    ):
        return _make_decision(
            accepted=True,
            reason="general_dual_path_support",
            policy=policy,
            ordered_candidates=(
                ordered_candidates
            ),
            supporting_candidates=(
                top_candidate,
            ),
        )

    return _make_decision(
        accepted=False,
        reason="insufficient_combined_support",
        policy=policy,
        ordered_candidates=(
            ordered_candidates
        ),
    )