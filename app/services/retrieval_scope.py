"""定义向量检索和关键词检索共用的范围过滤规则。

本模块负责：

1. 保存不可变的文档ID和来源文件名白名单；
2. 校验范围值的类型、格式、数量和重复项；
3. 对DocumentChunk执行内存范围判断；
4. 生成具有相同语义的Chroma where表达式。

本模块不执行向量检索、关键词打分或HTTP处理。
"""

from dataclasses import dataclass

# fullmatch()要求整个字符串都符合正则，
# 而不是只有其中一部分符合。
from re import fullmatch

from app.schemas.retrieval import (
    DocumentChunk,
)


# 与DiagnosisRetrievalScope保持相同的最大条件数量。
MAX_SCOPE_VALUES = 20

# 来源文件名的最大长度与API Schema保持一致。
MAX_SOURCE_FILE_LENGTH = 255


def _clean_document_ids(
    values: tuple[str, ...],
) -> tuple[str, ...]:
    """清理并校验文档ID白名单。"""

    if len(values) > MAX_SCOPE_VALUES:
        raise ValueError(
            "document_ids最多包含20项"
        )

    cleaned_values: list[str] = []

    for position, value in enumerate(
        values,
        start=1,
    ):
        # 类型注解不会自动阻止整数进入tuple。
        if not isinstance(value, str):
            raise TypeError(
                "document_ids中的第"
                f"{position}项必须是字符串"
            )

        cleaned_value = value.strip()

        # ^和$在这里不是必需的，
        # 因为fullmatch()本身要求整个字符串匹配。
        #
        # [0-9a-f]{64}表示：
        #
        # 1. 只允许数字0-9和小写a-f；
        # 2. 必须正好有64个字符。
        if (
            fullmatch(
                r"[0-9a-f]{64}",
                cleaned_value,
            )
            is None
        ):
            raise ValueError(
                "document_ids中的值必须是"
                "64位小写十六进制文本"
            )

        cleaned_values.append(
            cleaned_value
        )

    if (
        len(cleaned_values)
        != len(set(cleaned_values))
    ):
        raise ValueError(
            "document_ids不能包含重复项"
        )

    return tuple(cleaned_values)


def _clean_source_files(
    values: tuple[str, ...],
) -> tuple[str, ...]:
    """清理并校验来源文件名白名单。"""

    if len(values) > MAX_SCOPE_VALUES:
        raise ValueError(
            "source_files最多包含20项"
        )

    cleaned_values: list[str] = []

    for position, value in enumerate(
        values,
        start=1,
    ):
        if not isinstance(value, str):
            raise TypeError(
                "source_files中的第"
                f"{position}项必须是字符串"
            )

        cleaned_value = value.strip()

        if not cleaned_value:
            raise ValueError(
                "source_files中的值不能为空"
            )

        if (
            len(cleaned_value)
            > MAX_SOURCE_FILE_LENGTH
        ):
            raise ValueError(
                "source_files中的值"
                "不能超过255个字符"
            )

        cleaned_values.append(
            cleaned_value
        )

    # 必须在strip()之后去重。
    #
    # 否则：
    #
    # "manual.pdf"
    # "  manual.pdf  "
    #
    # 会被错误地视为两个不同条件。
    if (
        len(cleaned_values)
        != len(set(cleaned_values))
    ):
        raise ValueError(
            "source_files不能包含重复项"
        )

    return tuple(cleaned_values)


@dataclass(
    frozen=True,
    slots=True,
)
class RetrievalScopeFilter:
    """一次检索使用的不可变文档范围。"""

    # tuple表示条件创建完成后不能append或remove。
    document_ids: tuple[str, ...] = ()
    source_files: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """在dataclass初始化后校验并清理字段。"""

        # dataclass不会根据类型注解自动拒绝list，
        # 因此仍然需要运行时检查。
        if not isinstance(
            self.document_ids,
            tuple,
        ):
            raise TypeError(
                "document_ids必须是tuple"
            )

        if not isinstance(
            self.source_files,
            tuple,
        ):
            raise TypeError(
                "source_files必须是tuple"
            )

        cleaned_document_ids = (
            _clean_document_ids(
                self.document_ids
            )
        )
        cleaned_source_files = (
            _clean_source_files(
                self.source_files
            )
        )

        # 创建范围对象时必须有实际限制。
        #
        # “不限制范围”应该直接使用None，
        # 而不是创建两个字段都为空的对象。
        if not (
            cleaned_document_ids
            or cleaned_source_files
        ):
            raise ValueError(
                "检索范围至少需要一个筛选条件"
            )

        # frozen=True禁止普通字段赋值。
        #
        # 但__post_init__仍需要把strip()后的结果
        # 保存回当前对象，因此使用底层的
        # object.__setattr__()完成初始化阶段赋值。
        object.__setattr__(
            self,
            "document_ids",
            cleaned_document_ids,
        )
        object.__setattr__(
            self,
            "source_files",
            cleaned_source_files,
        )

    def matches(
        self,
        chunk: DocumentChunk,
    ) -> bool:
        """判断一个Chunk是否位于允许检索的范围内。"""

        if not isinstance(
            chunk,
            DocumentChunk,
        ):
            raise TypeError(
                "chunk必须是DocumentChunk"
            )

        # 未提供document_ids时，
        # 文档ID条件不限制当前Chunk。
        document_matches = (
            not self.document_ids
            or (
                chunk.document_id
                in self.document_ids
            )
        )

        # 未提供source_files时，
        # 文件名条件不限制当前Chunk。
        source_matches = (
            not self.source_files
            or (
                chunk.source_file
                in self.source_files
            )
        )

        # 不同字段之间使用AND。
        return (
            document_matches
            and source_matches
        )

    def to_chroma_where(
        self,
    ) -> dict[str, object]:
        """生成与matches()语义一致的Chroma过滤条件。"""

        conditions: list[
            dict[str, object]
        ] = []

        if self.document_ids:
            conditions.append(
                {
                    "document_id": {
                        # $in表示元数据字段必须存在于
                        # 给定值列表中。
                        "$in": list(
                            self.document_ids
                        )
                    }
                }
            )

        if self.source_files:
            conditions.append(
                {
                    "source_file": {
                        "$in": list(
                            self.source_files
                        )
                    }
                }
            )

        # __post_init__已经保证至少存在一个条件。
        if len(conditions) == 1:
            return conditions[0]

        # 同时存在两类限制时，
        # 使用$and要求二者同时成立。
        return {
            "$and": conditions
        }
