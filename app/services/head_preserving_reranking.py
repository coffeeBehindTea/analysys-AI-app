"""对RRF候选执行确定性的头部保留重排。

本模块负责：
1. 保留原RRF排名最靠前的候选；
2. 防止向量头部候选被双路候选完全挤出；
3. 防止关键词头部候选被双路候选完全挤出；
4. 在最终Top-K容量有限时按原RRF顺序确定取舍；
5. 返回重新连续编号的候选深复制。

本模块不执行Embedding、Chroma检索、关键词评分、
RRF计算、证据门控或LLM回答生成。
"""

from dataclasses import dataclass

from app.schemas.retrieval import (
    HybridRetrievedChunk,
)


# 重排策略的稳定版本。
#
# 修改默认头部深度、候选保护规则或替换规则时，
# 必须升级为head-preserving-v2等新版本。
HEAD_PRESERVING_RERANK_VERSION = (
    "head-preserving-v1"
)


def _validate_non_negative_integer(
    *,
    value: object,
    name: str,
) -> int:
    """校验必须大于或等于0的整数参数。"""

    # bool是int的子类：
    #
    # isinstance(True, int) == True
    #
    # 但True不能清晰表示候选深度，
    # 因此必须在int检查之前单独拒绝。
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
    ):
        raise TypeError(
            f"{name}必须是整数"
        )

    if value < 0:
        raise ValueError(
            f"{name}必须大于或等于0"
        )

    return value


def _validate_positive_integer(
    *,
    value: object,
    name: str,
) -> int:
    """校验必须大于或等于1的整数参数。"""

    validated = (
        _validate_non_negative_integer(
            value=value,
            name=name,
        )
    )

    if validated < 1:
        raise ValueError(
            f"{name}必须大于或等于1"
        )

    return validated


@dataclass(
    frozen=True,
    slots=True,
)
class HeadPreservationPolicy:
    """头部保留重排使用的不可变策略参数。"""

    # 至少保护原RRF结果的第一名。
    #
    # 这样重排不会轻易推翻两条检索路径
    # 共同支持的最强融合候选。
    fused_head_k: int = 1

    # 保护向量路径的第一名，
    # 用于保留语义路径独立找到的证据。
    vector_head_k: int = 1

    # 保护关键词路径的前两名。
    #
    # q008的正确证据位于关键词第一名；
    # q024的正确证据位于关键词第二名。
    keyword_head_k: int = 2

    def __post_init__(
        self,
    ) -> None:
        """在不可变策略创建完成后校验三个深度。"""

        _validate_non_negative_integer(
            value=self.fused_head_k,
            name="fused_head_k",
        )
        _validate_non_negative_integer(
            value=self.vector_head_k,
            name="vector_head_k",
        )
        _validate_non_negative_integer(
            value=self.keyword_head_k,
            name="keyword_head_k",
        )


def _validate_candidates(
    candidates: object,
) -> list[HybridRetrievedChunk]:
    """校验并按原RRF排名整理候选。"""

    if not isinstance(candidates, list):
        raise TypeError(
            "candidates必须是列表"
        )

    seen_chunk_ids: set[str] = set()

    for position, candidate in enumerate(
        candidates
    ):
        if not isinstance(
            candidate,
            HybridRetrievedChunk,
        ):
            raise TypeError(
                "candidates中的第 "
                f"{position} 项必须是"
                "HybridRetrievedChunk"
            )

        chunk_id = candidate.chunk.chunk_id

        if chunk_id in seen_chunk_ids:
            raise ValueError(
                "candidates不能包含重复的"
                f"chunk_id：{chunk_id}"
            )

        seen_chunk_ids.add(chunk_id)

    # 调用方即使传入了乱序列表，
    # 仍以候选对象保存的rank字段为准。
    ordered_candidates = sorted(
        candidates,
        key=lambda candidate: (
            candidate.rank
        ),
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
            "candidates的rank必须从1开始"
            "连续且不能重复"
        )

    return ordered_candidates


def _is_head_candidate(
    *,
    candidate: HybridRetrievedChunk,
    policy: HeadPreservationPolicy,
) -> bool:
    """判断候选是否属于任意需要保护的头部。"""

    is_fused_head = (
        candidate.rank
        <= policy.fused_head_k
    )

    is_vector_head = (
        candidate.vector_rank is not None
        and candidate.vector_rank
        <= policy.vector_head_k
    )

    is_keyword_head = (
        candidate.keyword_rank is not None
        and candidate.keyword_rank
        <= policy.keyword_head_k
    )

    return any(
        (
            is_fused_head,
            is_vector_head,
            is_keyword_head,
        )
    )


def rerank_hybrid_candidates(
    *,
    candidates: list[HybridRetrievedChunk],
    top_k: int,
    policy: HeadPreservationPolicy,
) -> list[HybridRetrievedChunk]:
    """保留各路径头部并返回最终Top-K候选。"""

    # 必须先校验Top-K。
    #
    # 这样top_k非法时不会继续读取或复制
    # 可能很大的候选列表。
    validated_top_k = (
        _validate_positive_integer(
            value=top_k,
            name="top_k",
        )
    )

    if not isinstance(
        policy,
        HeadPreservationPolicy,
    ):
        raise TypeError(
            "policy必须是"
            "HeadPreservationPolicy"
        )

    ordered_candidates = (
        _validate_candidates(
            candidates
        )
    )

    if not ordered_candidates:
        return []

    # 候选不足top_k时返回实际数量。
    effective_top_k = min(
        validated_top_k,
        len(ordered_candidates),
    )

    # 先得到所有具有头部保护资格的候选。
    #
    # ordered_candidates已经按照原RRF排名排序，
    # 所以这里的顺序也是稳定的原融合顺序。
    head_candidates = [
        candidate
        for candidate in ordered_candidates
        if _is_head_candidate(
            candidate=candidate,
            policy=policy,
        )
    ]

    # 保护候选数量可能超过最终Top-K。
    #
    # 此时按照原RRF顺序保留最靠前的候选，
    # 而不是使用集合的无序结果随机选择。
    required_candidates = (
        head_candidates[
            :effective_top_k
        ]
    )

    required_chunk_ids = {
        candidate.chunk.chunk_id
        for candidate in required_candidates
    }

    # 先取得原RRF的Top-K作为基础结果。
    selected_candidates = list(
        ordered_candidates[
            :effective_top_k
        ]
    )

    selected_chunk_ids = {
        candidate.chunk.chunk_id
        for candidate in selected_candidates
    }

    for required in required_candidates:
        required_id = (
            required.chunk.chunk_id
        )

        # 该头部候选已经位于原Top-K时，
        # 不需要进行任何替换。
        if required_id in selected_chunk_ids:
            continue

        # 从当前结果尾部向前寻找
        # 不属于保护集合的最低排名候选。
        replacement_index: int | None = None

        for index in range(
            len(selected_candidates) - 1,
            -1,
            -1,
        ):
            current_id = (
                selected_candidates[
                    index
                ].chunk.chunk_id
            )

            if (
                current_id
                not in required_chunk_ids
            ):
                replacement_index = index
                break

        # required_candidates最多只有effective_top_k项。
        #
        # 如果某个必选候选尚未进入结果，
        # 理论上一定存在一个非必选位置可替换。
        if replacement_index is None:
            raise RuntimeError(
                "无法为头部候选分配"
                "最终Top-K位置"
            )

        removed_id = (
            selected_candidates[
                replacement_index
            ].chunk.chunk_id
        )

        selected_chunk_ids.remove(
            removed_id
        )

        selected_candidates[
            replacement_index
        ] = required

        selected_chunk_ids.add(
            required_id
        )

    # 替换完成后恢复原RRF相对顺序。
    #
    # 重排器只决定哪些候选能够留下，
    # 不重新计算或伪造RRF分数。
    selected_candidates.sort(
        key=lambda candidate: (
            candidate.rank
        )
    )

    reranked_results: list[
        HybridRetrievedChunk
    ] = []

    for new_rank, candidate in enumerate(
        selected_candidates,
        start=1,
    ):
        # model_copy(deep=True)由Pydantic提供。
        #
        # deep=True会同时复制嵌套的DocumentChunk，
        # 防止调用方修改返回正文后污染原候选池。
        #
        # update只更新新的公开排名，
        # 原rrf_score、vector_rank和keyword_rank
        # 全部保留用于审计。
        reranked_results.append(
            candidate.model_copy(
                deep=True,
                update={
                    "rank": new_rank,
                },
            )
        )

    return reranked_results