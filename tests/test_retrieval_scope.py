"""通用检索范围过滤器的纯离线测试。

本模块不访问Chroma、Embedding、LLM或HTTP服务。

测试范围：
1. 范围条件的数据类型、非空、格式和去重；
2. 过滤对象的不可变性；
3. DocumentChunk内存匹配语义；
4. document_ids和source_files组合时的交集语义；
5. 与相同语义对应的Chroma where表达式。
"""

from dataclasses import FrozenInstanceError

import pytest

from app.schemas.retrieval import (
    DocumentChunk,
)
from app.services.retrieval_scope import (
    RetrievalScopeFilter,
)


# 两个合法的SHA-256文档标识，
# 用于区分允许文档和其他文档。
ALLOWED_DOCUMENT_ID = "1" * 64
OTHER_DOCUMENT_ID = "2" * 64
TEST_CONTENT_HASH = "3" * 64


def make_chunk(
    *,
    document_id: str = ALLOWED_DOCUMENT_ID,
    source_file: str = "allowed-manual.pdf",
) -> DocumentChunk:
    """构造可自定义文档ID和来源文件的Chunk。"""

    return DocumentChunk(
        chunk_id=f"{document_id}:000001",
        document_id=document_id,
        source_file=source_file,
        page_or_section="page: 1",
        chunk_index=1,
        content_hash=TEST_CONTENT_HASH,
        content="检索范围测试正文",
    )


def test_scope_requires_at_least_one_constraint(
) -> None:
    """没有任何条件时不应创建无意义的范围对象。"""

    with pytest.raises(
        ValueError,
        match="检索范围至少需要一个筛选条件",
    ):
        RetrievalScopeFilter()


def test_scope_normalizes_string_whitespace(
) -> None:
    """来源文件名首尾空白应在创建时统一清理。"""

    scope = RetrievalScopeFilter(
        source_files=(
            "  allowed-manual.pdf  ",
        ),
    )

    assert scope.source_files == (
        "allowed-manual.pdf",
    )


def test_scope_is_immutable(
) -> None:
    """检索开始后不能静默更换允许文档范围。"""

    scope = RetrievalScopeFilter(
        document_ids=(
            ALLOWED_DOCUMENT_ID,
        ),
    )

    with pytest.raises(FrozenInstanceError):
        scope.document_ids = ()  # type: ignore[misc]


@pytest.mark.parametrize(
    "field_name",
    [
        "document_ids",
        "source_files",
    ],
)
def test_scope_fields_require_tuples(
    field_name: str,
) -> None:
    """可变list不能冒充不可变检索范围快照。"""

    with pytest.raises(
        TypeError,
        match=f"{field_name}必须是tuple",
    ):
        RetrievalScopeFilter(
            **{
                field_name: [
                    ALLOWED_DOCUMENT_ID
                ],
            }
        )


def test_scope_rejects_invalid_document_id(
) -> None:
    """文档过滤ID必须保持SHA-256格式。"""

    with pytest.raises(
        ValueError,
        match="document_ids中的值必须是64位",
    ):
        RetrievalScopeFilter(
            document_ids=(
                "not-a-document-hash",
            ),
        )


def test_scope_rejects_duplicates_after_cleaning(
) -> None:
    """清理空白后相同的文件名仍属于重复条件。"""

    with pytest.raises(
        ValueError,
        match="source_files不能包含重复项",
    ):
        RetrievalScopeFilter(
            source_files=(
                "manual.pdf",
                "  manual.pdf  ",
            ),
        )


def test_document_id_scope_matches_only_allowed_document(
) -> None:
    """只设置文档ID时应忽略来源文件名差异。"""

    scope = RetrievalScopeFilter(
        document_ids=(
            ALLOWED_DOCUMENT_ID,
        ),
    )

    assert scope.matches(
        make_chunk(
            document_id=ALLOWED_DOCUMENT_ID,
            source_file="any-name.pdf",
        )
    ) is True
    assert scope.matches(
        make_chunk(
            document_id=OTHER_DOCUMENT_ID,
            source_file="allowed-manual.pdf",
        )
    ) is False


def test_source_file_scope_matches_only_allowed_file(
) -> None:
    """只设置文件名时应按完整文件名精确匹配。"""

    scope = RetrievalScopeFilter(
        source_files=(
            "allowed-manual.pdf",
        ),
    )

    assert scope.matches(
        make_chunk(
            document_id=OTHER_DOCUMENT_ID,
            source_file="allowed-manual.pdf",
        )
    ) is True
    assert scope.matches(
        make_chunk(
            source_file="other-manual.pdf",
        )
    ) is False


def test_document_and_file_scopes_use_intersection(
) -> None:
    """同时提供两类条件时Chunk必须同时满足两者。"""

    scope = RetrievalScopeFilter(
        document_ids=(
            ALLOWED_DOCUMENT_ID,
        ),
        source_files=(
            "allowed-manual.pdf",
        ),
    )

    assert scope.matches(
        make_chunk()
    ) is True

    # 文档ID正确但文件名错误，仍然拒绝。
    assert scope.matches(
        make_chunk(
            source_file="other-manual.pdf",
        )
    ) is False

    # 文件名正确但文档ID错误，也必须拒绝。
    assert scope.matches(
        make_chunk(
            document_id=OTHER_DOCUMENT_ID,
        )
    ) is False


def test_matches_rejects_non_chunk_object(
) -> None:
    """普通dict不能绕过DocumentChunk的数据校验。"""

    scope = RetrievalScopeFilter(
        source_files=(
            "allowed-manual.pdf",
        ),
    )

    with pytest.raises(
        TypeError,
        match="chunk必须是DocumentChunk",
    ):
        scope.matches(
            {  # type: ignore[arg-type]
                "source_file": (
                    "allowed-manual.pdf"
                )
            }
        )


def test_document_scope_builds_chroma_where(
) -> None:
    """文档ID白名单应转换成Chroma的$in条件。"""

    scope = RetrievalScopeFilter(
        document_ids=(
            ALLOWED_DOCUMENT_ID,
            OTHER_DOCUMENT_ID,
        ),
    )

    assert scope.to_chroma_where() == {
        "document_id": {
            "$in": [
                ALLOWED_DOCUMENT_ID,
                OTHER_DOCUMENT_ID,
            ]
        }
    }


def test_source_scope_builds_chroma_where(
) -> None:
    """文件名白名单应转换成Chroma的$in条件。"""

    scope = RetrievalScopeFilter(
        source_files=(
            "a.pdf",
            "b.txt",
        ),
    )

    assert scope.to_chroma_where() == {
        "source_file": {
            "$in": [
                "a.pdf",
                "b.txt",
            ]
        }
    }


def test_combined_scope_builds_chroma_and_where(
) -> None:
    """两类条件应使用$and保持与内存匹配相同语义。"""

    scope = RetrievalScopeFilter(
        document_ids=(
            ALLOWED_DOCUMENT_ID,
        ),
        source_files=(
            "allowed-manual.pdf",
        ),
    )

    assert scope.to_chroma_where() == {
        "$and": [
            {
                "document_id": {
                    "$in": [
                        ALLOWED_DOCUMENT_ID
                    ]
                }
            },
            {
                "source_file": {
                    "$in": [
                        "allowed-manual.pdf"
                    ]
                }
            },
        ]
    }
