"""Agent当前请求已经确认的知识库证据Store。

本Store只记录search_knowledge在当前Agent请求中
真实返回的KnowledgeCitation。

它不是知识库，不执行检索，也不持久化数据。
"""

from collections.abc import (
    Iterable,
)

from app.schemas.knowledge_query import (
    KnowledgeCitation,
)


def _stable_evidence_identity(
    citation: KnowledgeCitation,
) -> tuple[
    str,
    str,
    str,
    str,
    int,
    str,
]:
    """返回不随查询排名变化的证据身份字段。

    rank、similarity和rrf_score属于某一次查询的排序信息，
    同一Chunk在不同查询中出现不同数值是正常现象，不能据此
    判定正文冲突。
    """

    return (
        citation.chunk_id,
        citation.document_id,
        citation.source_file,
        citation.page_or_section,
        citation.chunk_index,
        citation.excerpt,
    )


class ConfirmedEvidenceStoreError(
    ValueError
):
    """已确认证据Store错误的父类。"""


class ConflictingConfirmedEvidenceError(
    ConfirmedEvidenceStoreError
):
    """同一Chunk ID出现互相冲突的引用内容。"""


class ConfirmedEvidenceStore:
    """保存当前Agent请求已经确认的引用白名单。

    Store实例必须属于单个Agent请求，
    不能跨用户请求全局缓存。
    """

    def __init__(
        self,
    ) -> None:
        """创建空的已确认证据Store。"""

        self._citations: dict[
            str,
            KnowledgeCitation,
        ] = {}

    def record(
        self,
        citations: Iterable[
            KnowledgeCitation
        ],
        /,
    ) -> tuple[str, ...]:
        """原子记录一组真实知识库引用。

        成功时返回本次处理的Chunk ID；
        任意一项不合法时不会写入部分结果。
        """

        pending: dict[
            str,
            KnowledgeCitation,
        ] = {}

        for position, citation in enumerate(
            citations
        ):
            if not isinstance(
                citation,
                KnowledgeCitation,
            ):
                raise TypeError(
                    "citations中的第 "
                    f"{position} 项必须是"
                    "KnowledgeCitation"
                )

            # KnowledgeCitation当前不是frozen模型。
            #
            # 使用deep=True复制嵌套内容，
            # 防止调用方在记录后修改原对象。
            trusted_copy = citation.model_copy(
                deep=True
            )

            existing = pending.get(
                trusted_copy.chunk_id
            )

            if existing is None:
                existing = self._citations.get(
                    trusted_copy.chunk_id
                )

            if existing is not None:
                # 只比较稳定来源和正文。
                # 检索rank与分数可能随查询变化，不属于证据身份。
                if (
                    _stable_evidence_identity(existing)
                    != _stable_evidence_identity(
                        trusted_copy
                    )
                ):
                    raise (
                        ConflictingConfirmedEvidenceError(
                            "同一chunk_id出现冲突引用："
                            f"{trusted_copy.chunk_id}"
                        )
                    )

                # 相同证据再次出现时保留第一次确认的快照。
                # 这样后续查询不会静默改写已有审计信息。
                pending.setdefault(
                    trusted_copy.chunk_id,
                    existing.model_copy(deep=True),
                )
                continue

            pending[
                trusted_copy.chunk_id
            ] = trusted_copy

        # 只有全部输入都通过检查后才更新正式字典。
        #
        # 这叫原子式更新：不会出现前两条已经写入，
        # 第三条失败后留下半组证据的情况。
        self._citations.update(pending)

        return tuple(pending)

    def get(
        self,
        chunk_id: str,
        /,
    ) -> KnowledgeCitation | None:
        """按精确Chunk ID取得已确认的证据。"""

        if not isinstance(chunk_id, str):
            raise TypeError(
                "chunk_id必须是字符串"
            )

        citation = self._citations.get(
            chunk_id
        )

        if citation is None:
            return None

        # 返回深复制，避免调用方修改Store内部白名单。
        return citation.model_copy(
            deep=True
        )

    def resolve_many(
        self,
        chunk_ids: Iterable[str],
        /,
    ) -> tuple[
        KnowledgeCitation,
        ...,
    ] | None:
        """按输入顺序解析多条证据。

        任意Chunk ID不存在时返回None，
        不允许只使用其中一部分证据继续生成草案。
        """

        validated_ids: list[str] = []

        for position, chunk_id in enumerate(
            chunk_ids
        ):
            if not isinstance(chunk_id, str):
                raise TypeError(
                    "chunk_ids中的第 "
                    f"{position} 项必须是字符串"
                )

            if chunk_id in validated_ids:
                raise ValueError(
                    "chunk_ids不能包含重复项"
                )

            validated_ids.append(chunk_id)

        resolved: list[
            KnowledgeCitation
        ] = []

        for chunk_id in validated_ids:
            citation = self.get(chunk_id)

            if citation is None:
                # all-or-nothing：
                # 一条未知证据会使整组解析失败。
                return None

            resolved.append(citation)

        return tuple(resolved)

    def __contains__(
        self,
        chunk_id: object,
    ) -> bool:
        """支持使用'chunk_id in store'检查白名单。"""

        return chunk_id in self._citations

    def __len__(
        self,
    ) -> int:
        """返回当前已确认证据数量。"""

        return len(self._citations)
