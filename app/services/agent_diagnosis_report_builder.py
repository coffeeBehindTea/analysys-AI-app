"""把Agent诊断草稿和请求级证据Store组装成可信报告。

本模块负责：

1. 从AgentDiagnosisRequest构造带来源的症状；
2. 收集草稿真正引用的Chunk ID；
3. 使用ConfirmedEvidenceStore执行证据白名单检查；
4. 从真实KnowledgeCitation复制Chunk元数据；
5. 为多次检索的证据建立唯一报告顺序；
6. 构造正常、部分完成或拒答DiagnosisReport；
7. 为AgentRunner中止生成稳定的拒答报告。

本模块不运行Agent、不调用LLM、不执行工具、
不处理HTTP，也不信任模型生成的Chunk元数据。
"""

from app.agent.evidence_store import (
    ConfirmedEvidenceStore,
)
from app.errors import (
    InvalidLLMResponseError,
)
from app.schemas.agent_api import (
    AgentDiagnosisRequest,
)
from app.schemas.agent_diagnosis import (
    AgentDiagnosisDraft,
)
from app.schemas.diagnostics import (
    DiagnosisCause,
    DiagnosisCheck,
    DiagnosisEvidence,
    DiagnosisReport,
    DiagnosisSymptom,
)
from app.schemas.knowledge_query import (
    KnowledgeCitation,
)


# 该版本表示Agent最终报告使用的证据规则：
#
# 1. Chunk ID必须存在于当前请求的Store；
# 2. 元数据只能从KnowledgeCitation复制；
# 3. 引用采用all-or-nothing检查；
# 4. 最终报告最多包含10条唯一证据；
# 5. 不允许为缺失的检索分数编造替代值。
AGENT_EVIDENCE_GATE_VERSION = (
    "agent-confirmed-evidence-v1"
)


# DiagnosisReport.evidence的数据契约最多允许10项。
#
# Builder使用同一个常量提前拒绝超限草稿，
# 不能静默截断模型选择的证据。
MAX_AGENT_REPORT_EVIDENCE = 10


def _clean_tracking_text(
    value: object,
    *,
    field_name: str,
    max_length: int,
) -> str:
    """清理请求ID或版本字段并执行长度检查。

    在访问Store和构造报告前提前检查，
    避免完成业务处理后才发现追踪字段非法。
    """

    if not isinstance(value, str):
        raise TypeError(
            f"{field_name}必须是字符串"
        )

    cleaned_value = value.strip()

    if not cleaned_value:
        raise ValueError(
            f"{field_name}不能为空"
        )

    if len(cleaned_value) > max_length:
        raise ValueError(
            f"{field_name}不能超过"
            f"{max_length}个字符"
        )

    return cleaned_value


def build_agent_diagnosis_symptoms(
    request: AgentDiagnosisRequest,
    /,
) -> list[DiagnosisSymptom]:
    """从Agent API请求构造带来源的观察现象。

    symptom和log_excerpt属于用户输入，
    不能因为其中包含错误码就冒充知识库证据。
    """

    if not isinstance(
        request,
        AgentDiagnosisRequest,
    ):
        raise TypeError(
            "request必须是"
            "AgentDiagnosisRequest"
        )

    return [
        DiagnosisSymptom(
            description=request.symptom,
            source="user_report",
        ),
        DiagnosisSymptom(
            description=request.log_excerpt,
            source="log_excerpt",
        ),
    ]


def _collect_referenced_chunk_ids(
    draft: AgentDiagnosisDraft,
    /,
) -> tuple[str, ...]:
    """按首次引用顺序收集草稿使用的Chunk ID。

    同一个Chunk可以同时支持原因和检查项，
    但最终DiagnosisReport.evidence只需要包含一次。
    """

    if not isinstance(
        draft,
        AgentDiagnosisDraft,
    ):
        raise TypeError(
            "draft必须是AgentDiagnosisDraft"
        )

    collected_ids: list[str] = []
    seen_ids: set[str] = set()

    # 原因排列在检查项前面。
    #
    # 这使最终证据顺序保持稳定：
    #
    # 1. 先收集支持原因的证据；
    # 2. 再收集只支持检查项的证据；
    # 3. 已经出现的Chunk不重复添加。
    for item in [
        *draft.possible_causes,
        *draft.next_checks,
    ]:
        for chunk_id in (
            item.evidence_chunk_ids
        ):
            if chunk_id in seen_ids:
                continue

            seen_ids.add(chunk_id)
            collected_ids.append(chunk_id)

    return tuple(collected_ids)


def _build_agent_diagnosis_evidence(
    *,
    citation: KnowledgeCitation,
    report_rank: int,
) -> DiagnosisEvidence:
    """把白名单中的真实引用转换成诊断证据。

    除report_rank外，所有来源字段都直接复制自
    ConfirmedEvidenceStore返回的KnowledgeCitation。
    """

    if not isinstance(
        citation,
        KnowledgeCitation,
    ):
        raise TypeError(
            "citation必须是KnowledgeCitation"
        )

    # Week 4的search_knowledge使用混合检索。
    #
    # 如果引用没有rrf_score，不能用：
    #
    # 1. similarity冒充RRF分数；
    # 2. 0.0表示缺失值；
    # 3. 任意常量使DiagnosisEvidence通过校验。
    if citation.rrf_score is None:
        raise InvalidLLMResponseError(
            "Agent已确认证据缺少RRF分数"
        )

    return DiagnosisEvidence(
        # 以下字段全部来自请求级证据Store，
        # 不从AgentDiagnosisDraft读取。
        chunk_id=citation.chunk_id,
        document_id=citation.document_id,
        source_file=citation.source_file,
        page_or_section=(
            citation.page_or_section
        ),

        # KnowledgeCitation.rank是某一次检索中的局部排名。
        #
        # Agent可能执行多次检索，因此多条引用都可能rank=1。
        # 最终报告要求rank唯一，所以这里按草稿首次引用顺序
        # 建立跨检索的报告级顺序。
        rank=report_rank,

        rrf_score=citation.rrf_score,
        vector_similarity=(
            citation.similarity
        ),
        excerpt=citation.excerpt,
    )


def build_agent_diagnosis_report_from_draft(
    *,
    request_id: str,
    request: AgentDiagnosisRequest,
    planner_prompt_version: str,
    draft: AgentDiagnosisDraft,
    evidence_store: ConfirmedEvidenceStore,

    # 该字段只能由Service根据已经成功执行的
    # DraftTestCaseToolOutput收集。
    #
    # 默认空tuple保持原有调用方兼容。
    additional_evidence_chunk_ids: tuple[
        str,
        ...,
    ] = (),
) -> DiagnosisReport:
    """把Agent草稿转换成经过证据白名单校验的报告。

    最终证据来源包括：

    1. 原因和检查项引用的Chunk ID；
    2. 实际生成的测试草案引用的附加Chunk ID。

    两类ID都会经过同一个请求级
    ConfirmedEvidenceStore执行all-or-nothing解析。

    只要一个Chunk ID未确认，
    整份报告就不会生成部分可信结果。
    """

    if not isinstance(
        request,
        AgentDiagnosisRequest,
    ):
        raise TypeError(
            "request必须是"
            "AgentDiagnosisRequest"
        )

    if not isinstance(
        draft,
        AgentDiagnosisDraft,
    ):
        raise TypeError(
            "draft必须是AgentDiagnosisDraft"
        )

    if not isinstance(
        evidence_store,
        ConfirmedEvidenceStore,
    ):
        raise TypeError(
            "evidence_store必须是"
            "ConfirmedEvidenceStore"
        )

    if not isinstance(
        additional_evidence_chunk_ids,
        tuple,
    ):
        raise TypeError(
            "additional_evidence_chunk_ids"
            "必须是tuple"
        )

    cleaned_request_id = (
        _clean_tracking_text(
            request_id,
            field_name="request_id",
            max_length=200,
        )
    )

    cleaned_prompt_version = (
        _clean_tracking_text(
            planner_prompt_version,
            field_name=(
                "planner_prompt_version"
            ),
            max_length=100,
        )
    )

    # 先取得原因和检查项使用的证据。
    draft_chunk_ids = (
        _collect_referenced_chunk_ids(
            draft
        )
    )

    # 使用list保存稳定顺序：
    #
    # 1. 原因和检查证据在前；
    # 2. 只被测试草案使用的证据在后；
    # 3. 同一个Chunk只保留第一次出现的位置。
    merged_chunk_ids = list(
        draft_chunk_ids
    )

    seen_chunk_ids = set(
        merged_chunk_ids
    )

    for position, chunk_id in enumerate(
        additional_evidence_chunk_ids
    ):
        if not isinstance(chunk_id, str):
            raise TypeError(
                "additional_evidence_chunk_ids"
                "中的第 "
                f"{position} 项必须是字符串"
            )

        if chunk_id in seen_chunk_ids:
            continue

        seen_chunk_ids.add(chunk_id)
        merged_chunk_ids.append(chunk_id)

    referenced_chunk_ids = tuple(
        merged_chunk_ids
    )

    if (
        len(referenced_chunk_ids)
        > MAX_AGENT_REPORT_EVIDENCE
    ):
        # 不能只保留前10条。
        #
        # 静默截断可能导致：
        #
        # 1. 原因引用不在evidence中；
        # 2. 检查项引用不在evidence中；
        # 3. 测试草案引用不在evidence中。
        raise InvalidLLMResponseError(
            "Agent最终报告"
            "最多只能引用10条唯一证据"
        )

    report_evidence: list[
        DiagnosisEvidence
    ] = []

    # 非拒答草稿一定会通过原因或检查项产生证据ID。
    #
    # 拒答草稿本身没有原因和检查，但可能存在
    # 已经实际生成的测试草案附加证据。
    if referenced_chunk_ids:
        # resolve_many()采用all-or-nothing规则：
        #
        # 1. 按输入顺序解析；
        # 2. 任意ID不存在时返回None；
        # 3. 不返回部分成功结果；
        # 4. 返回的是Store内部可信引用的深复制。
        resolved_citations = (
            evidence_store.resolve_many(
                referenced_chunk_ids
            )
        )

        if resolved_citations is None:
            raise InvalidLLMResponseError(
                "Agent诊断草稿或测试草案"
                "引用了当前请求未确认的Chunk ID"
            )

        report_evidence = [
            _build_agent_diagnosis_evidence(
                citation=citation,

                # 多次检索中可能重复出现局部rank=1。
                #
                # 最终报告按照合并后的首次引用顺序，
                # 重新建立1、2、3……的唯一顺序。
                report_rank=report_rank,
            )
            for report_rank, citation
            in enumerate(
                resolved_citations,
                start=1,
            )
        ]

    # 创建新的嵌套模型，
    # 不与模型草稿共享可变的证据ID列表。
    report_causes = [
        DiagnosisCause(
            description=cause.description,
            evidence_chunk_ids=list(
                cause.evidence_chunk_ids
            ),
        )
        for cause in draft.possible_causes
    ]

    report_checks = [
        DiagnosisCheck(
            description=check.description,
            evidence_chunk_ids=list(
                check.evidence_chunk_ids
            ),
            risk_level=check.risk_level,
            requires_qualified_person=(
                check
                .requires_qualified_person
            ),
        )
        for check in draft.next_checks
    ]

    # DiagnosisReport继续执行最终关系校验。
    #
    # abstained报告允许保留证据用于审计，
    # 但不能包含原因或检查项。
    return DiagnosisReport(
        request_id=cleaned_request_id,
        prompt_version=(
            cleaned_prompt_version
        ),
        gate_version=(
            AGENT_EVIDENCE_GATE_VERSION
        ),
        status=draft.status,
        symptoms=(
            build_agent_diagnosis_symptoms(
                request
            )
        ),
        evidence=report_evidence,
        possible_causes=report_causes,
        next_checks=report_checks,
        risk_level=draft.risk_level,
        missing_information=list(
            draft.missing_information
        ),
        abstained=draft.abstained,
    )


def build_agent_abstained_report(
    *,
    request_id: str,
    request: AgentDiagnosisRequest,
    planner_prompt_version: str,
    missing_information: tuple[
        str,
        ...,
    ],
) -> DiagnosisReport:
    """为Agent中止或最终草稿失败构造稳定拒答报告。

    本函数不会根据已有工具结果猜测原因，
    也不会为了返回partial而自动降低证据要求。
    """

    if not isinstance(
        request,
        AgentDiagnosisRequest,
    ):
        raise TypeError(
            "request必须是"
            "AgentDiagnosisRequest"
        )

    if not isinstance(
        missing_information,
        tuple,
    ):
        raise TypeError(
            "missing_information必须是tuple"
        )

    if not missing_information:
        raise ValueError(
            "missing_information"
            "缺失信息不能为空"
        )

    if (
        len(missing_information)
        != len(set(missing_information))
    ):
        raise ValueError(
            "missing_information"
            "不能包含重复项"
        )

    cleaned_request_id = (
        _clean_tracking_text(
            request_id,
            field_name="request_id",
            max_length=200,
        )
    )

    cleaned_prompt_version = (
        _clean_tracking_text(
            planner_prompt_version,
            field_name=(
                "planner_prompt_version"
            ),
            max_length=100,
        )
    )

    return DiagnosisReport(
        request_id=cleaned_request_id,
        prompt_version=(
            cleaned_prompt_version
        ),
        gate_version=(
            AGENT_EVIDENCE_GATE_VERSION
        ),
        status="abstained",
        symptoms=(
            build_agent_diagnosis_symptoms(
                request
            )
        ),
        evidence=[],
        possible_causes=[],
        next_checks=[],
        risk_level="unknown",
        missing_information=list(
            missing_information
        ),
        abstained=True,
    )