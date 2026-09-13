"""结构化诊断LLM内部输出的数据契约。

本模块只定义模型允许返回的字段：

1. 可能原因及其临时证据编号；
2. 下一步检查及其临时证据编号；
3. 风险、缺失信息和拒答状态；
4. 各字段之间必须满足的业务关系。

LLM不允许返回真实Chunk ID、文件名、页码、
request_id、Prompt版本或门控版本。
这些字段必须由Python从可信对象中组装。
"""

from typing import (
    Annotated,
    Self,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from app.schemas.diagnostics import (
    DiagnosisRiskLevel,
    DiagnosisStatus,
    MissingInformationItem,
)


# E1、E2、E3是一次Prompt内部使用的临时证据编号。
#
# E0无效，因为检索排名从1开始；
# 小写e1也无效，避免编号格式产生歧义。
DiagnosisEvidenceId = Annotated[
    str,
    Field(
        pattern=r"^E[1-9][0-9]*$",
        description="本次Prompt中的临时证据编号",
    ),
]


class DiagnosisDraftCause(BaseModel):
    """LLM返回的一项可能原因。"""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    description: str = Field(
        min_length=1,
        max_length=2_000,
        description="只根据证据形成的谨慎原因描述",
    )

    # LLM只能引用E1、E2等临时编号，
    # 不能直接填写真实Chunk ID。
    evidence_ids: list[
        DiagnosisEvidenceId
    ] = Field(
        min_length=1,
        max_length=10,
        description="支持该原因的临时证据编号",
    )

    @field_validator("evidence_ids")
    @classmethod
    def evidence_ids_must_be_unique(
        cls,
        values: list[str],
    ) -> list[str]:
        """同一原因不能重复引用同一证据。"""

        if len(values) != len(set(values)):
            raise ValueError(
                "evidence_ids不能包含重复项"
            )

        return values


class DiagnosisDraftCheck(BaseModel):
    """LLM返回的一项下一步检查。"""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    description: str = Field(
        min_length=1,
        max_length=2_000,
        description="只根据证据提出的下一步检查",
    )

    evidence_ids: list[
        DiagnosisEvidenceId
    ] = Field(
        min_length=1,
        max_length=10,
        description="支持该检查的临时证据编号",
    )

    risk_level: DiagnosisRiskLevel = Field(
        description="执行该检查涉及的风险等级",
    )

    requires_qualified_person: bool = Field(
        description="是否需要有资质人员确认或执行",
    )

    @field_validator("evidence_ids")
    @classmethod
    def evidence_ids_must_be_unique(
        cls,
        values: list[str],
    ) -> list[str]:
        """同一检查不能重复引用同一证据。"""

        if len(values) != len(set(values)):
            raise ValueError(
                "evidence_ids不能包含重复项"
            )

        return values

    @model_validator(mode="after")
    def high_risk_requires_qualification(
        self,
    ) -> Self:
        """高风险检查必须要求有资质人员确认。"""

        if (
            self.risk_level == "high"
            and not self.requires_qualified_person
        ):
            raise ValueError(
                "高风险检查必须标记为"
                "需要有资质人员确认"
            )

        return self


class DiagnosisLLMDraft(BaseModel):
    """LLM结构化诊断JSON的完整内部契约。"""

    model_config = ConfigDict(
        str_strip_whitespace=True,

        # 禁止LLM添加source_file、chunk_id、
        # confidence等未经声明的字段。
        extra="forbid",
    )

    status: DiagnosisStatus

    possible_causes: list[
        DiagnosisDraftCause
    ] = Field(
        default_factory=list,
        max_length=10,
    )

    next_checks: list[
        DiagnosisDraftCheck
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
    )

    abstained: bool

    @model_validator(mode="after")
    def validate_draft_contract(
        self,
    ) -> Self:
        """校验状态、拒答和内容之间的关系。"""

        status_says_abstained = (
            self.status == "abstained"
        )

        if (
            status_says_abstained
            != self.abstained
        ):
            raise ValueError(
                "status与abstained不一致"
            )

        if self.abstained:
            # 如果模型声称拒答，
            # 就不能同时给出原因或排查动作。
            if (
                self.possible_causes
                or self.next_checks
            ):
                raise ValueError(
                    "拒答草稿不能包含原因或排查项"
                )

            if not self.missing_information:
                raise ValueError(
                    "拒答草稿必须说明缺失信息"
                )

            return self

        # completed和partial都声称形成了某种诊断，
        # 因而至少需要一项原因或检查。
        if not (
            self.possible_causes
            or self.next_checks
        ):
            raise ValueError(
                "非拒答草稿至少需要一项原因或排查项"
            )

        if (
            self.status == "partial"
            and not self.missing_information
        ):
            raise ValueError(
                "partial草稿必须说明缺失信息"
            )

        if (
            self.status == "completed"
            and self.missing_information
        ):
            raise ValueError(
                "completed草稿不能包含缺失信息"
            )

        return self