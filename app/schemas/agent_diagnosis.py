"""Robot Diagnostic Agent内部诊断草稿的数据契约。

本模块定义Planner结束Agent循环时允许生成的诊断内容。

Planner可以填写：

1. 诊断状态；
2. 可能原因；
3. 下一步安全检查；
4. 已确认知识库证据的Chunk ID；
5. 风险等级；
6. 缺失信息和拒答状态。

Planner不能填写：

1. request_id；
2. Prompt版本或门控版本；
3. 来源文件名、页码或章节；
4. 文档ID、证据正文或检索分数；
5. Agent执行轨迹；
6. 模拟遥测的来源标记。

上述可信字段必须由AgentDiagnosisService
从真实工具结果和当前请求证据白名单中构造。
"""

from typing import (
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
    DiagnosisCause,
    DiagnosisCheck,
    DiagnosisRiskLevel,
    DiagnosisStatus,
    MissingInformationItem,
)


class AgentDiagnosisDraft(BaseModel):
    """Planner完成工具调用后生成的内部诊断草稿。

    本模型只保证字段格式和字段关系正确。

    evidence_chunk_ids是否真实存在，
    仍然必须由后续AgentDiagnosisService
    使用当前请求的ConfirmedEvidenceStore校验。
    """

    model_config = ConfigDict(
        # 自动清除普通字符串字段首尾空白。
        str_strip_whitespace=True,

        # 拒绝模型添加source_file、excerpt、
        # command等没有声明的字段。
        extra="forbid",

        # 草稿创建后不能直接替换顶层字段。
        #
        # tuple字段也避免调用方直接append()。
        frozen=True,
    )

    status: DiagnosisStatus = Field(
        description=(
            "当前诊断是完整、部分完成还是拒答"
        ),
    )

    # 这里复用第三周的DiagnosisCause。
    #
    # 它已经规定：
    #
    # 1. 原因描述不能为空；
    # 2. 每个原因至少引用一个Chunk ID；
    # 3. 同一原因内不能重复引用同一个Chunk。
    possible_causes: tuple[
        DiagnosisCause,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=10,
        description=(
            "由已确认知识库证据支持的可能原因"
        ),
    )

    # 这里复用第三周的DiagnosisCheck。
    #
    # 它已经规定高风险检查必须设置
    # requires_qualified_person=True。
    next_checks: tuple[
        DiagnosisCheck,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=10,
        description=(
            "由已确认知识库证据支持的"
            "下一步安全检查"
        ),
    )

    risk_level: DiagnosisRiskLevel = Field(
        description=(
            "当前诊断草稿的总体风险等级"
        ),
    )

    missing_information: tuple[
        MissingInformationItem,
        ...,
    ] = Field(
        default_factory=tuple,
        max_length=20,
        description=(
            "形成更完整诊断仍然缺少的信息"
        ),
    )

    abstained: bool = Field(
        description=(
            "是否因证据不足或冲突而拒绝"
            "生成原因和检查项"
        ),
    )

    @field_validator(
        "missing_information"
    )
    @classmethod
    def missing_information_must_be_unique(
        cls,
        values: tuple[str, ...],
    ) -> tuple[str, ...]:
        """缺失信息不能包含重复项。

        field_validator只校验一个字段。

        values是Pydantic已经转换完成的tuple；
        set(values)会删除重复字符串。
        """

        if len(values) != len(set(values)):
            raise ValueError(
                "missing_information"
                "不能包含重复项"
            )

        return values

    @model_validator(mode="after")
    def validate_draft_contract(
        self,
    ) -> Self:
        """校验状态、拒答和诊断内容之间的关系。

        model_validator用于同时检查多个字段。

        mode="after"表示先完成每个字段的类型转换，
        再检查它们组合在一起是否符合业务规则。
        """

        status_says_abstained = (
            self.status == "abstained"
        )

        # status和abstained不能表达相反含义。
        if (
            status_says_abstained
            != self.abstained
        ):
            raise ValueError(
                "status与abstained不一致"
            )

        if self.abstained:
            # 拒答表示当前证据不足以支持诊断结论，
            # 因此不能同时输出原因或操作建议。
            if (
                self.possible_causes
                or self.next_checks
            ):
                raise ValueError(
                    "拒答草稿不能包含"
                    "原因或检查项"
                )

            # 拒答不能只有“无法回答”，
            # 必须告诉调用方还缺什么。
            if not self.missing_information:
                raise ValueError(
                    "拒答草稿必须说明"
                    "missing_information"
                )

            # 没有形成证据支持的结论时，
            # 不能让模型自行声明确定的风险等级。
            if self.risk_level != "unknown":
                raise ValueError(
                    "拒答草稿的risk_level"
                    "必须是unknown"
                )

            return self

        # completed和partial都声称形成了某些结论，
        # 因此至少需要一个原因或一个检查项。
        if not (
            self.possible_causes
            or self.next_checks
        ):
            raise ValueError(
                "非拒答草稿至少需要"
                "一项原因或检查项"
            )

        # partial表示只完成了一部分，
        # 所以必须明确说明尚缺少的信息。
        if (
            self.status == "partial"
            and not self.missing_information
        ):
            raise ValueError(
                "partial草稿必须说明"
                "missing_information"
            )

        # completed表示现有信息已经足以完成诊断，
        # 不能同时声明仍有信息缺口。
        if (
            self.status == "completed"
            and self.missing_information
        ):
            raise ValueError(
                "completed草稿不能包含"
                "missing_information"
            )

        return self