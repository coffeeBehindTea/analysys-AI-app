"""证据约束机器人诊断API的数据契约。

本模块定义：

1. 诊断请求和可选检索范围；
2. 用户报告症状的来源；
3. 从真实Chunk构造的证据；
4. 绑定证据的可能原因和下一步检查；
5. 完成、部分完成和拒答报告之间的关系。

本模块不执行检索、不调用LLM，
也不直接处理HTTP请求。
"""

from typing import (
    Annotated,
    Literal,
    Self,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


# 文档ID来自原始文件内容的SHA-256。
DocumentId = Annotated[
    str,
    Field(
        pattern=r"^[0-9a-f]{64}$",
        description="原始文档的SHA-256标识",
    ),
]

# Chunk ID作为原因、检查项与证据之间的引用键。
EvidenceChunkId = Annotated[
    str,
    Field(
        min_length=1,
        max_length=200,
        description="本次检索返回的真实Chunk ID",
    ),
]

# 来源文件名必须是真正的非空文本，
# 不能使用空字符串或纯空白字符串冒充筛选条件。
SourceFileName = Annotated[
    str,
    Field(
        min_length=1,
        max_length=255,
        description="不包含服务器绝对路径的来源文件名",
    ),
]


# 每项缺失信息必须包含实际说明，
# 不能使用空白文本让拒答报告通过校验。
MissingInformationItem = Annotated[
    str,
    Field(
        min_length=1,
        max_length=2_000,
        description="形成更完整诊断仍需补充的一项信息",
    ),
]

# 三种状态分别表示：
#
# completed：
# 当前证据足以生成完整诊断。
#
# partial：
# 可以给出部分证据支持的结论，
# 但仍然缺少必要信息。
#
# abstained：
# 证据不足、冲突或置信度太低，
# 系统拒绝给出原因和排查动作。
DiagnosisStatus = Literal[
    "completed",
    "partial",
    "abstained",
]

DiagnosisRiskLevel = Literal[
    "low",
    "medium",
    "high",
    "unknown",
]

# 症状不是知识库产生的事实，
# 而是来自用户描述或日志。
DiagnosisSymptomSource = Literal[
    "user_report",
    "log_excerpt",
]


class DiagnosisRetrievalScope(BaseModel):
    """限制本次诊断允许检索的文档范围。"""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    # 可以按照文档内容哈希限定范围。
    document_ids: list[DocumentId] = Field(
        default_factory=list,
        max_length=20,
        description="允许检索的文档ID",
    )

    # 也可以按照不包含服务器路径的文件名限定范围。
    source_files: list[
        SourceFileName
    ] = Field(
        default_factory=list,
        max_length=20,
        description="允许检索的来源文件名",
    )

    @field_validator(
        "document_ids",
        "source_files",
    )
    @classmethod
    def filter_values_must_be_unique(
        cls,
        values: list[str],
    ) -> list[str]:
        """每一种范围过滤值都不能重复。"""

        if len(values) != len(set(values)):
            raise ValueError(
                "检索范围不能包含重复项"
            )

        return values

    @model_validator(mode="after")
    def require_at_least_one_filter(
        self,
    ) -> Self:
        """显式提供检索范围时必须有实际条件。"""

        if not (
            self.document_ids
            or self.source_files
        ):
            raise ValueError(
                "检索范围至少需要一个筛选条件"
            )

        return self


class DiagnosisRequest(BaseModel):
    """POST /api/v1/diagnostics的请求体。"""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    robot_id: str = Field(
        min_length=1,
        max_length=100,
        description="发生异常的机器人编号",
        examples=["robot-001"],
    )

    symptom: str = Field(
        min_length=1,
        max_length=2_000,
        description="用户观察到的故障现象",
        examples=["机器人通信恢复后仍未继续任务"],
    )

    log_excerpt: str = Field(
        min_length=1,
        max_length=10_000,
        description="经过脱敏处理的日志摘要",
        examples=[
            "ERR-NET-4001 heartbeat timeout"
        ],
    )

    # None表示允许检索当前知识库中的全部文档。
    #
    # 提供该对象时，后续检索Service必须真正应用范围，
    # 不能只接收字段但仍搜索整个知识库。
    retrieval_scope: (
        DiagnosisRetrievalScope | None
    ) = Field(
        default=None,
        description="本次诊断可选的文档检索范围",
    )


class DiagnosisSymptom(BaseModel):
    """由请求信息提取的一项观察现象。"""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    description: str = Field(
        min_length=1,
        max_length=2_000,
        description="用户描述或日志明确出现的现象",
    )

    # 症状引用请求来源，而不是伪造知识库Chunk引用。
    source: DiagnosisSymptomSource = Field(
        description="该现象来自用户描述还是日志",
    )


class DiagnosisEvidence(BaseModel):
    """从本次真实混合检索结果构造的一条证据。"""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        allow_inf_nan=False,
    )

    chunk_id: EvidenceChunkId

    document_id: DocumentId

    source_file: SourceFileName

    page_or_section: str = Field(
        min_length=1,
        max_length=500,
        description="证据在原始文件中的页码或章节",
    )

    # Week 3单次检索中，该值与RRF结果排名一致。
    #
    # Week 4 Agent可能执行多次独立检索，
    # 多个工具结果中可能同时存在局部rank=1。
    # Agent报告构造器会按照草稿首次引用顺序，
    # 重新生成从1开始的唯一报告级顺序。
    rank: int = Field(
        ge=1,
        description=(
            "证据在最终诊断报告中的唯一顺序；"
            "单次检索时与RRF排名一致"
        ),
    )

    # RRF分数只用于说明融合排序，
    # 不能解释为概率或回答置信度。
    rrf_score: float = Field(
        gt=0.0,
        description="该证据的RRF融合分数",
    )

    # 关键词独占候选可能没有进入向量候选集合，
    # 因而向量相似度允许为None。
    vector_similarity: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description="向量余弦相似度；未进入向量候选时为空",
    )

    # excerpt必须由Python从真实Chunk复制，
    # 不能使用LLM自己生成或改写的内容。
    excerpt: str = Field(
        min_length=1,
        max_length=20_000,
        description="真实Chunk原文",
    )


class DiagnosisCause(BaseModel):
    """一项由真实证据支持的可能原因。"""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    description: str = Field(
        min_length=1,
        max_length=2_000,
        description="谨慎表述的可能原因",
    )

    # 每个原因至少引用一个证据Chunk。
    evidence_chunk_ids: list[
        EvidenceChunkId
    ] = Field(
        min_length=1,
        max_length=10,
        description="支持该原因的真实Chunk ID",
    )

    @field_validator(
        "evidence_chunk_ids"
    )
    @classmethod
    def evidence_ids_must_be_unique(
        cls,
        values: list[str],
    ) -> list[str]:
        """同一原因不能重复引用同一个Chunk。"""

        if len(values) != len(set(values)):
            raise ValueError(
                "evidence_chunk_ids不能包含重复项"
            )

        return values


class DiagnosisCheck(BaseModel):
    """一项由证据支持的下一步安全检查。"""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    description: str = Field(
        min_length=1,
        max_length=2_000,
        description="根据证据提出的下一步检查",
    )

    evidence_chunk_ids: list[
        EvidenceChunkId
    ] = Field(
        min_length=1,
        max_length=10,
        description="支持该检查项的真实Chunk ID",
    )

    risk_level: DiagnosisRiskLevel = Field(
        description="执行该检查涉及的风险级别",
    )

    # 对高压、电池拆装、安全回路修改等操作，
    # 必须明确由有资质人员确认。
    requires_qualified_person: bool = Field(
        description="是否需要有资质人员确认或执行",
    )

    @field_validator(
        "evidence_chunk_ids"
    )
    @classmethod
    def evidence_ids_must_be_unique(
        cls,
        values: list[str],
    ) -> list[str]:
        """同一检查项不能重复引用同一个Chunk。"""

        if len(values) != len(set(values)):
            raise ValueError(
                "evidence_chunk_ids不能包含重复项"
            )

        return values

    @model_validator(mode="after")
    def high_risk_requires_qualification(
        self,
    ) -> Self:
        """高风险检查必须交给有资质人员确认。"""

        if (
            self.risk_level == "high"
            and not self.requires_qualified_person
        ):
            raise ValueError(
                "高风险检查必须标记为"
                "需要有资质人员确认"
            )

        return self


class DiagnosisReport(BaseModel):
    """诊断API成功时返回的完整结构化报告。"""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        allow_inf_nan=False,
    )

    # 用于关联日志、响应头和错误记录。
    request_id: str = Field(
        min_length=1,
        max_length=200,
        description="当前HTTP请求的追踪标识",
    )

    # Prompt和门控规则变化都会影响结果，
    # 因此必须进入报告用于复现和审计。
    prompt_version: str = Field(
        min_length=1,
        max_length=100,
        description="结构化诊断Prompt版本",
    )

    gate_version: str = Field(
        min_length=1,
        max_length=100,
        description="证据门控策略版本",
    )

    status: DiagnosisStatus

    # 症状来自用户报告或日志，不冒充知识库事实。
    symptoms: list[DiagnosisSymptom] = Field(
        min_length=1,
        max_length=20,
    )

    # evidence由Python从本次真实检索结果构造。
    evidence: list[DiagnosisEvidence] = Field(
        default_factory=list,
        max_length=10,
    )

    possible_causes: list[
        DiagnosisCause
    ] = Field(
        default_factory=list,
        max_length=10,
    )

    next_checks: list[
        DiagnosisCheck
    ] = Field(
        default_factory=list,
        max_length=10,
    )

    risk_level: DiagnosisRiskLevel

    missing_information: list[
        MissingInformationItem
    ] = Field(
        default_factory=list,
        max_length=20,
        description="形成更完整诊断仍需补充的信息",
    )

    abstained: bool = Field(
        description="系统是否因证据不足或冲突而拒答",
    )

    @model_validator(mode="after")
    def validate_report_contract(
        self,
    ) -> Self:
        """校验状态、证据和所有Chunk引用。"""

        status_says_abstained = (
            self.status == "abstained"
        )

        # status和布尔字段面向不同使用方式：
        #
        # status便于表达completed/partial；
        # abstained便于客户端快速判断是否拒答。
        #
        # 二者不能表达互相矛盾的状态。
        if (
            status_says_abstained
            != self.abstained
        ):
            raise ValueError(
                "status与abstained不一致"
            )

        evidence_chunk_ids = [
            item.chunk_id
            for item in self.evidence
        ]

        if (
            len(evidence_chunk_ids)
            != len(set(evidence_chunk_ids))
        ):
            raise ValueError(
                "evidence不能包含重复的chunk_id"
            )

        evidence_ranks = [
            item.rank
            for item in self.evidence
        ]

        if (
            len(evidence_ranks)
            != len(set(evidence_ranks))
        ):
            raise ValueError(
                "evidence不能包含重复的rank"
            )

        # 拒答时可以保留已经检索到的证据，
        # 以解释为何发生冲突或证据不足；
        # 但不能同时输出原因和操作建议。
        if self.abstained:
            if (
                self.possible_causes
                or self.next_checks
            ):
                raise ValueError(
                    "拒答报告不能包含原因或排查项"
                )

            if not self.missing_information:
                raise ValueError(
                    "拒答报告必须说明缺失信息"
                )

            return self

        # completed和partial都声称给出了某种诊断，
        # 所以必须存在真实检索证据。
        if not self.evidence:
            raise ValueError(
                "非拒答报告必须包含证据"
            )

        # 非拒答报告不能只有证据列表，
        # 必须至少形成一个原因或检查项。
        if not (
            self.possible_causes
            or self.next_checks
        ):
            raise ValueError(
                "非拒答报告至少需要一项原因或排查项"
            )

        if (
            self.status == "partial"
            and not self.missing_information
        ):
            raise ValueError(
                "partial报告必须说明缺失信息"
            )

        if (
            self.status == "completed"
            and self.missing_information
        ):
            raise ValueError(
                "completed报告不能包含缺失信息"
            )

        allowed_chunk_ids = set(
            evidence_chunk_ids
        )

        referenced_chunk_ids = {
            chunk_id
            for cause in self.possible_causes
            for chunk_id
            in cause.evidence_chunk_ids
        }

        referenced_chunk_ids.update(
            chunk_id
            for check in self.next_checks
            for chunk_id
            in check.evidence_chunk_ids
        )

        unknown_chunk_ids = sorted(
            referenced_chunk_ids
            - allowed_chunk_ids
        )

        # 这是引用白名单检查。
        #
        # LLM即使生成了格式正确的Chunk ID，
        # 只要该ID不在本次evidence中也必须拒绝。
        if unknown_chunk_ids:
            raise ValueError(
                "原因或排查项引用了本次报告中"
                "不存在的Chunk ID："
                + ", ".join(unknown_chunk_ids)
            )

        return self