"""关键词与向量候选的混合检索和倒数排名融合。

本模块负责：
1. 定义向量与关键词候选检索的最小能力契约；
2. 编排两条候选召回路径；
3. 校验两份候选排名；
4. 根据chunk_id识别同一证据；
5. 计算Reciprocal Rank Fusion分数；
6. 返回可审计的融合Top-K结果。

本模块不生成Embedding、不调用LLM，也不负责证据门控。
具体的向量检索和关键词检索由注入的依赖对象完成。
"""

from dataclasses import dataclass

# Protocol定义编排器依赖的最小能力，
# 不要求具体实现类显式继承。
from typing import Protocol

from app.schemas.retrieval import (
    DocumentChunk,
    EmbeddingVector,
    HybridRetrievedChunk,
    KeywordRetrievedChunk,
    RetrievedChunk,
)

# HeadPreservationPolicy保存是否需要保护
# RRF、向量和关键词路径头部的固定策略。
#
# rerank_hybrid_candidates接收完整RRF候选池，
# 在最终Top-K容量内执行确定性头部保留。
from app.services.head_preserving_reranking import (
    HeadPreservationPolicy,
    rerank_hybrid_candidates,
)

# RetrievalScopeFilter表示一次检索允许访问的文档范围。
#
# HybridRetriever不负责解释HTTP请求中的范围字段，
# 只负责把已经校验完成的统一范围传给两条检索路径。
from app.services.retrieval_scope import (
    RetrievalScopeFilter,
)

# RRF公式中的排名平滑常数。
#
# rank=1时，单条检索路径贡献：
#
# 1 / (60 + 1)
#
# 较大的常数会缩小相邻排名之间的分数差距，
# 让“两条路径共同召回”比单条路径中微小的
# 排名差异更重要。
DEFAULT_RRF_RANK_CONSTANT = 60

# 每条检索路径默认先取Top-20候选，
# 再由RRF融合成最终Top-K。
#
# 候选深度大于最终返回数量，
# 才能给RRF提供足够的交叉排名空间。
DEFAULT_HYBRID_CANDIDATE_K = 20


class VectorCandidateRetriever(
    Protocol
):
    """混合检索需要的最小向量召回能力。"""

    def retrieve(
        self,
        query_embedding: EmbeddingVector,
        *,
        query_embedding_model: str,
        top_k: int = 3,

        # None表示搜索整个向量Collection；
        # 提供过滤器时只搜索允许的文档范围。
        retrieval_scope: (
            RetrievalScopeFilter | None
        ) = None,
    ) -> list[RetrievedChunk]:
        """在可选范围内返回按相似度排列的向量候选。"""

        ...


class KeywordCandidateRetriever(
    Protocol
):
    """混合检索需要的最小关键词召回能力。"""

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 3,

        # 关键词检索必须使用与向量检索完全相同的范围。
        retrieval_scope: (
            RetrievalScopeFilter | None
        ) = None,
    ) -> list[KeywordRetrievedChunk]:
        """在可选范围内返回按关键词分数排列的候选。"""

        ...


@dataclass(
    slots=True,
)
class _FusionCandidate:
    """RRF计算过程中可逐步补全的内部候选。"""

    # 保存完整Chunk快照。
    chunk: DocumentChunk

    # 构造后会分别累加向量和关键词排名贡献。
    rrf_score: float = 0.0

    # 向量路径的原始审计数据。
    vector_rank: int | None = None
    vector_similarity: float | None = None

    # 关键词路径的原始审计数据。
    keyword_rank: int | None = None
    keyword_score: float | None = None

    # 关键词路径提供的具体命中解释。
    matched_identifiers: tuple[str, ...] = ()
    matched_model_aliases: tuple[str, ...] = ()
    matched_numeric_terms: tuple[str, ...] = ()
    matched_lexical_terms: tuple[str, ...] = ()


def _validate_positive_integer(
    *,
    value: object,
    name: str,
) -> int:
    """校验RRF配置中的正整数参数。"""

    # bool是int的子类，但True和False
    # 不应被当作合法排名配置。
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
    ):
        raise TypeError(
            f"{name}必须是整数"
        )

    if value < 1:
        raise ValueError(
            f"{name}必须大于或等于1"
        )

    return value


def _validate_vector_results(
    results: object,
) -> list[RetrievedChunk]:
    """校验并按rank整理向量候选。"""

    # 当前公共契约明确要求列表，
    # 防止None、字符串或单个结果对象进入融合。
    if not isinstance(results, list):
        raise TypeError(
            "vector_results必须是列表"
        )

    seen_chunk_ids: set[str] = set()

    for position, result in enumerate(
        results
    ):
        if not isinstance(
            result,
            RetrievedChunk,
        ):
            raise TypeError(
                "vector_results中的第 "
                f"{position} 项必须是RetrievedChunk"
            )

        chunk_id = result.chunk.chunk_id

        # 同一排名列表中不能重复出现同一个Chunk。
        #
        # 否则该Chunk会从同一检索路径获得两次RRF分数。
        if chunk_id in seen_chunk_ids:
            raise ValueError(
                "vector_results不能包含重复的"
                f"chunk_id：{chunk_id}"
            )

        seen_chunk_ids.add(
            chunk_id
        )

    # 输入列表本身可以不是rank顺序，
    # 融合逻辑以结果对象的rank字段为准。
    ordered_results = sorted(
        results,
        key=lambda result: result.rank,
    )

    actual_ranks = [
        result.rank
        for result in ordered_results
    ]

    expected_ranks = list(
        range(
            1,
            len(ordered_results) + 1,
        )
    )

    # 一份完整候选排名应当是：
    #
    # 1、2、3……
    #
    # 不能出现重复排名或1、3、5之类的缺口。
    if actual_ranks != expected_ranks:
        raise ValueError(
            "vector_results的rank"
            "必须从1开始连续且不能重复"
        )

    return ordered_results


def _validate_keyword_results(
    results: object,
) -> list[KeywordRetrievedChunk]:
    """校验并按rank整理关键词候选。"""

    if not isinstance(results, list):
        raise TypeError(
            "keyword_results必须是列表"
        )

    seen_chunk_ids: set[str] = set()

    for position, result in enumerate(
        results
    ):
        if not isinstance(
            result,
            KeywordRetrievedChunk,
        ):
            raise TypeError(
                "keyword_results中的第 "
                f"{position} 项必须是"
                "KeywordRetrievedChunk"
            )

        chunk_id = result.chunk.chunk_id

        if chunk_id in seen_chunk_ids:
            raise ValueError(
                "keyword_results不能包含重复的"
                f"chunk_id：{chunk_id}"
            )

        seen_chunk_ids.add(
            chunk_id
        )

    ordered_results = sorted(
        results,
        key=lambda result: result.rank,
    )

    actual_ranks = [
        result.rank
        for result in ordered_results
    ]

    expected_ranks = list(
        range(
            1,
            len(ordered_results) + 1,
        )
    )

    if actual_ranks != expected_ranks:
        raise ValueError(
            "keyword_results的rank"
            "必须从1开始连续且不能重复"
        )

    return ordered_results


def _count_sources(
    candidate: _FusionCandidate,
) -> int:
    """返回候选进入了几条检索路径。"""

    # Python中：
    #
    # True可以作为1参与求和；
    # False可以作为0参与求和。
    #
    # 所以返回值只能是1或2。
    return sum(
        (
            candidate.vector_rank
            is not None,
            candidate.keyword_rank
            is not None,
        )
    )


def _best_source_rank(
    candidate: _FusionCandidate,
) -> int:
    """返回候选在两条检索路径中的最好原始排名。"""

    ranks = [
        rank
        for rank in (
            candidate.vector_rank,
            candidate.keyword_rank,
        )
        if rank is not None
    ]

    # 每个_FusionCandidate至少来自一条路径，
    # 因此ranks不可能为空。
    return min(ranks)


def reciprocal_rank_fusion(
    *,
    vector_results: list[RetrievedChunk],
    keyword_results: list[
        KeywordRetrievedChunk
    ],
    top_k: int = 3,
    rank_constant: int = (
        DEFAULT_RRF_RANK_CONSTANT
    ),
) -> list[HybridRetrievedChunk]:
    """融合向量和关键词排名并返回最终Top-K。"""

    validated_top_k = (
        _validate_positive_integer(
            value=top_k,
            name="top_k",
        )
    )

    validated_rank_constant = (
        _validate_positive_integer(
            value=rank_constant,
            name="rank_constant",
        )
    )

    ordered_vector_results = (
        _validate_vector_results(
            vector_results
        )
    )

    ordered_keyword_results = (
        _validate_keyword_results(
            keyword_results
        )
    )

    # chunk_id是跨两条检索路径识别
    # “同一个知识证据”的稳定主键。
    candidates_by_id: dict[
        str,
        _FusionCandidate,
    ] = {}

    for result in ordered_vector_results:
        chunk = result.chunk
        chunk_id = chunk.chunk_id

        # 向量列表内部已经检查过重复ID，
        # 所以此时每个chunk_id只会创建一次。
        candidate = _FusionCandidate(
            # 深复制防止调用方随后修改原始结果，
            # 造成融合结果内容变化。
            chunk=chunk.model_copy(
                deep=True
            ),
            vector_rank=result.rank,
            vector_similarity=(
                result.similarity
            ),
        )

        # RRF不使用原始余弦相似度，
        # 只使用候选在当前列表中的排名。
        candidate.rrf_score += (
            1.0
            / (
                validated_rank_constant
                + result.rank
            )
        )

        candidates_by_id[
            chunk_id
        ] = candidate

    for result in ordered_keyword_results:
        chunk = result.chunk
        chunk_id = chunk.chunk_id

        candidate = candidates_by_id.get(
            chunk_id
        )

        if candidate is None:
            # 该Chunk只进入了关键词候选，
            # 因而现在创建新的融合候选。
            candidate = _FusionCandidate(
                chunk=chunk.model_copy(
                    deep=True
                )
            )

            candidates_by_id[
                chunk_id
            ] = candidate

        else:
            # 相同chunk_id必须描述完全相同的Chunk。
            #
            # 如果正文、来源、哈希或索引不同，
            # 说明两条检索路径使用了不一致的
            # 语料快照，不能静默合并。
            if candidate.chunk != chunk:
                raise ValueError(
                    "相同chunk_id在向量结果和"
                    "关键词结果中对应不同Chunk"
                )

        candidate.keyword_rank = (
            result.rank
        )
        candidate.keyword_score = (
            result.keyword_score
        )

        candidate.matched_identifiers = (
            result.matched_identifiers
        )
        candidate.matched_model_aliases = (
            result.matched_model_aliases
        )
        candidate.matched_numeric_terms = (
            result.matched_numeric_terms
        )
        candidate.matched_lexical_terms = (
            result.matched_lexical_terms
        )

        # 如果该Chunk也出现在向量列表中，
        # 这里会在已有向量贡献上继续累加。
        candidate.rrf_score += (
            1.0
            / (
                validated_rank_constant
                + result.rank
            )
        )

    ordered_candidates = sorted(
        candidates_by_id.values(),
        key=lambda candidate: (
            # 第一排序条件：RRF分数越高越靠前。
            -candidate.rrf_score,

            # RRF分数完全相同时，
            # 优先两条路径共同召回的候选。
            -_count_sources(
                candidate
            ),

            # 然后比较它在任意一条路径中的
            # 最好原始排名。
            _best_source_rank(
                candidate
            ),

            # 所有业务条件相同时，
            # 使用chunk_id保证结果可复现。
            candidate.chunk.chunk_id,
        ),
    )

    fused_results: list[
        HybridRetrievedChunk
    ] = []

    for rank, candidate in enumerate(
        ordered_candidates[
            :validated_top_k
        ],
        start=1,
    ):
        fused_results.append(
            HybridRetrievedChunk(
                # 再次深复制，使返回对象与
                # 内部候选快照相互隔离。
                chunk=(
                    candidate.chunk.model_copy(
                        deep=True
                    )
                ),
                rrf_score=(
                    candidate.rrf_score
                ),
                rank=rank,
                vector_rank=(
                    candidate.vector_rank
                ),
                vector_similarity=(
                    candidate.vector_similarity
                ),
                keyword_rank=(
                    candidate.keyword_rank
                ),
                keyword_score=(
                    candidate.keyword_score
                ),
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

    return fused_results


class HybridRetriever:
    """编排向量候选、关键词候选和RRF融合。"""

    def __init__(
        self,
        *,
        vector_retriever: (
            VectorCandidateRetriever
        ),
        keyword_retriever: (
            KeywordCandidateRetriever
        ),
        embedding_model: str,
        candidate_k: int = (
            DEFAULT_HYBRID_CANDIDATE_K
        ),
        rank_constant: int = (
            DEFAULT_RRF_RANK_CONSTANT
        ),

        # None表示保持原来的纯RRF截断行为。
        #
        # 提供HeadPreservationPolicy时，
        # HybridRetriever会先保留完整RRF候选池，
        # 再执行头部保留并截取最终Top-K。
        rerank_policy: (
            HeadPreservationPolicy | None
        ) = None,
    ) -> None:
        """保存混合检索依赖和固定策略参数。"""

        # Embedding模型名称会传给向量存储层，
        # 用于确认查询向量和Collection属于
        # 同一个向量空间。
        if not isinstance(
            embedding_model,
            str,
        ):
            raise TypeError(
                "embedding_model必须是字符串"
            )

        cleaned_embedding_model = (
            embedding_model.strip()
        )

        if not cleaned_embedding_model:
            raise ValueError(
                "embedding_model不能为空"
            )

        validated_candidate_k = (
            _validate_positive_integer(
                value=candidate_k,
                name="candidate_k",
            )
        )

        validated_rank_constant = (
            _validate_positive_integer(
                value=rank_constant,
                name="rank_constant",
            )
        )

        # 类型注解不会在Python运行时自动拒绝
        # 普通dict、字符串或任意object。
        #
        # 构造时提前验证，可以避免执行完两路检索后
        # 才因为重排策略类型错误而失败。
        if (
            rerank_policy is not None
            and not isinstance(
                rerank_policy,
                HeadPreservationPolicy,
            )
        ):
            raise TypeError(
                "rerank_policy必须是"
                "HeadPreservationPolicy或None"
            )

        # 保存依赖对象。
        #
        # Protocol用于静态检查；
        # 真正调用时，Python会执行这些对象的
        # retrieve()方法。
        self._vector_retriever = (
            vector_retriever
        )
        self._keyword_retriever = (
            keyword_retriever
        )

        self._embedding_model = (
            cleaned_embedding_model
        )
        self._candidate_k = (
            validated_candidate_k
        )
        self._rank_constant = (
            validated_rank_constant
        )

        # HeadPreservationPolicy使用frozen=True，
        # 创建后不能修改，因此可以安全地保存引用。
        self._rerank_policy = rerank_policy

    def retrieve(
        self,
        *,
        query: str,
        query_embedding: EmbeddingVector,
        top_k: int = 3,

        # None保持原有全库检索行为；
        # 提供过滤器时，两条候选路径必须共同使用它。
        retrieval_scope: (
            RetrievalScopeFilter | None
        ) = None,
    ) -> list[HybridRetrievedChunk]:
        """在可选文档范围内执行两路召回和RRF融合。"""

        # 类型注解不会自动拒绝整数或None，
        # 因而公共业务边界仍要执行运行时校验。
        if not isinstance(
            query,
            str,
        ):
            raise TypeError(
                "query必须是字符串"
            )

        cleaned_query = query.strip()

        if not cleaned_query:
            raise ValueError(
                "query不能为空"
            )

        validated_top_k = (
            _validate_positive_integer(
                value=top_k,
                name="top_k",
            )
        )

        # 两条路径各自只提供candidate_k个候选。
        #
        # 如果要求最终返回数量比单路候选深度还大，
        # 融合结果的覆盖范围将不再符合当前策略定义。
        if (
            validated_top_k
            > self._candidate_k
        ):
            raise ValueError(
                "top_k不能大于candidate_k"
            )

        # 类型注解不会在Python运行时自动拦截普通dict。
        #
        # 在编排层提前校验，可以保证不会出现：
        #
        # 1. 向量路径已经执行；
        # 2. 关键词路径才发现范围非法；
        # 3. 两条路径只完成一半。
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

        # 向量检索已经拥有查询向量，
        # 此处不会再次调用Embedding API。
        vector_results = (
            self._vector_retriever.retrieve(
                query_embedding,
                query_embedding_model=(
                    self._embedding_model
                ),
                top_k=self._candidate_k,

                # ChromaVectorStore会把它转换成where条件，
                # 然后在过滤后的向量集合中执行Top-K。
                retrieval_scope=retrieval_scope,
            )
        )

        # 关键词路径使用清理首尾空白后的原始问题。
        keyword_results = (
            self._keyword_retriever.retrieve(
                cleaned_query,
                top_k=self._candidate_k,

                # KeywordRetriever会先通过matches()
                # 排除范围外Chunk，再计算关键词排名。
                retrieval_scope=retrieval_scope,
            )
        )

        # 保存到局部变量，后面的分支都使用同一个策略快照。
        #
        # HeadPreservationPolicy是不可变对象，
        # 因而一次retrieve()过程中策略不会发生变化。
        rerank_policy = self._rerank_policy

        # 未启用重排时维持原行为：
        #
        # RRF只需要直接返回最终Top-K。
        #
        # 启用重排时，RRF必须先返回完整候选池。
        # 两条检索路径最多各返回candidate_k项，
        # 所以去重前的最大候选数量是candidate_k * 2。
        fusion_top_k = (
            validated_top_k
            if rerank_policy is None
            else self._candidate_k * 2
        )

        # RRF负责：
        #
        # 1. 校验两份候选排名；
        # 2. 按chunk_id合并同一证据；
        # 3. 累加两条路径的排名贡献；
        # 4. 产生连续的融合排名。
        fused_candidates = reciprocal_rank_fusion(
            vector_results=vector_results,
            keyword_results=keyword_results,
            top_k=fusion_top_k,
            rank_constant=(
                self._rank_constant
            ),
        )

        # 没有提供重排策略时，fused_candidates
        # 已经是原始RRF最终Top-K，可以直接返回。
        if rerank_policy is None:
            return fused_candidates

        # 启用重排时，fused_candidates包含完整融合候选池。
        #
        # 重排器可以看到原RRF Top-K之外的
        # 向量Top-1和关键词Top-2候选，
        # 然后在最终Top-K容量内执行最小替换。
        return rerank_hybrid_candidates(
            candidates=fused_candidates,
            top_k=validated_top_k,
            policy=rerank_policy,
        )