"""把门控结果和LLM草稿组装成可信DiagnosisReport。

本模块负责：

1. 从DiagnosisRequest构造带来源的症状；
2. 从HybridRetrievedChunk构造真实证据；
3. 检查LLM临时证据编号白名单；
4. 把E1、E2转换成真实Chunk ID；
5. 区分门控拒答和模型拒答；
6. 生成最终DiagnosisReport。

本模块不执行检索、不调用LLM，
也不生成HTTP响应。
"""

from typing import Literal

from app.errors import (
    InvalidLLMResponseError,
)
from app.schemas.diagnosis_llm import (
    DiagnosisLLMDraft,
)
from app.schemas.diagnostics import (
    DiagnosisCause,
    DiagnosisCheck,
    DiagnosisEvidence,
    DiagnosisReport,
    DiagnosisRequest,
    DiagnosisSymptom,
)
from app.schemas.retrieval import (
    HybridRetrievedChunk,
)
from app.services.diagnosis_prompts import (
    DIAGNOSIS_PROMPT_VERSION,
    build_diagnosis_evidence_map,
)
from app.services.gated_hybrid_retrieval import (
    GatedHybridRetrievalResult,
)


# 门控拒绝时根本没有调用LLM，
# 因此不能把diagnosis-evidence-v1
# 记录成已经实际使用的Prompt。
NO_LLM_PROMPT_VERSION = "not-invoked"


# accepted=True时只允许出现这三种原因。
#
# frozenset是不可变集合：
#
# 1. 不能add()或remove()；
# 2. 适合保存不会在运行时变化的白名单；
# 3. 使用in检查成员通常为常数时间。
ACCEPTED_GATE_REASONS = frozenset(
    {
        "exact_identifier_support",
        "model_and_lexical_support",
        "general_dual_path_support",
    }
)


# accepted=False时只允许出现这两种原因。
REJECTED_GATE_REASONS = frozenset(
    {
        "no_candidates",
        "insufficient_combined_support",
    }
)

# 诊断生成阶段允许出现的稳定失败原因。
#
# Literal主要服务于编辑器、静态检查器和调用方：
# 调用函数时只能从这三个原因代码中选择。
DiagnosisGenerationFailureReason = Literal[
    "invalid_llm_response",
    "llm_timeout",
    "llm_upstream_error",
]


# 把内部失败原因转换成稳定的公开消息。
#
# 键是供程序判断的原因代码；
# 值是可以安全返回给客户端的固定文本。
#
# 这里不能使用str(exc)，因为原始异常可能包含：
#
# 1. 上游地址；
# 2. 模型名称；
# 3. SDK内部信息；
# 4. 请求内容；
# 5. 其他不应该暴露给客户端的实现细节。
GENERATION_FAILURE_MESSAGES: dict[
    DiagnosisGenerationFailureReason,
    str,
] = {
    "invalid_llm_response": (
        "结构化诊断结果未通过数据和引用校验，"
        "需要重试或由人工复核"
    ),
    "llm_timeout": (
        "结构化诊断生成服务响应超时，"
        "需要稍后重试或由人工复核"
    ),
    "llm_upstream_error": (
        "结构化诊断生成服务暂时不可用，"
        "需要稍后重试或由人工复核"
    ),
}

def build_diagnosis_symptoms(
    request: DiagnosisRequest,
) -> list[DiagnosisSymptom]:
    """从请求构造带真实来源的观察现象。"""

    # 普通dict不能绕过DiagnosisRequest中的
    # 字符串长度、空白清理和未知字段校验。
    if not isinstance(
        request,
        DiagnosisRequest,
    ):
        raise TypeError(
            "request必须是DiagnosisRequest"
        )

    return [
        DiagnosisSymptom(
            # 这项内容来自用户对故障现象的描述。
            description=request.symptom,
            source="user_report",
        ),
        DiagnosisSymptom(
            # 这项内容来自用户提交的脱敏日志摘要。
            #
            # 即使日志包含错误码，
            # 它仍然是请求输入，不冒充知识库证据。
            description=request.log_excerpt,
            source="log_excerpt",
        ),
    ]


def build_diagnosis_evidence(
    candidate: HybridRetrievedChunk,
) -> DiagnosisEvidence:
    """把真实混合检索候选转换成报告证据。"""

    if not isinstance(
        candidate,
        HybridRetrievedChunk,
    ):
        raise TypeError(
            "candidate必须是"
            "HybridRetrievedChunk"
        )

    chunk = candidate.chunk

    # 下列可信字段全部来自检索系统，
    # 没有任何一个字段来自LLM草稿。
    return DiagnosisEvidence(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        source_file=chunk.source_file,
        page_or_section=(
            chunk.page_or_section
        ),
        rank=candidate.rank,
        rrf_score=candidate.rrf_score,
        vector_similarity=(
            candidate.vector_similarity
        ),
        excerpt=chunk.content,
    )


def _collect_draft_evidence_ids(
    draft: DiagnosisLLMDraft,
) -> tuple[str, ...]:
    """收集草稿中实际使用的临时证据编号。"""

    collected_ids: list[str] = []
    seen_ids: set[str] = set()

    # 原因和检查项使用相同的证据编号空间。
    #
    # 星号解包把两个列表组合成一个新列表：
    #
    # [
    #     *draft.possible_causes,
    #     *draft.next_checks,
    # ]
    for item in [
        *draft.possible_causes,
        *draft.next_checks,
    ]:
        for evidence_id in item.evidence_ids:
            # 同一证据可能同时支持一个原因和一个检查项。
            #
            # 最终evidence列表只需要包含一次，
            # 因而这里进行去重。
            if evidence_id in seen_ids:
                continue

            seen_ids.add(evidence_id)
            collected_ids.append(
                evidence_id
            )

    # 返回tuple表示收集结果不会再被调用方增删。
    return tuple(collected_ids)


def build_diagnosis_report_from_draft(
    *,
    request_id: str,
    request: DiagnosisRequest,
    retrieval_result: (
        GatedHybridRetrievalResult
    ),
    draft: DiagnosisLLMDraft,
) -> DiagnosisReport:
    """把门控放行结果和LLM草稿组装成最终报告。"""

    # 以下检查建立运行时模块边界。
    #
    # 类型注解只帮助编辑器和静态检查工具，
    # 不会自动阻止普通dict进入函数。
    if not isinstance(
        request,
        DiagnosisRequest,
    ):
        raise TypeError(
            "request必须是DiagnosisRequest"
        )

    if not isinstance(
        retrieval_result,
        GatedHybridRetrievalResult,
    ):
        raise TypeError(
            "retrieval_result必须是"
            "GatedHybridRetrievalResult"
        )

    if not isinstance(
        draft,
        DiagnosisLLMDraft,
    ):
        raise TypeError(
            "draft必须是DiagnosisLLMDraft"
        )

    decision = retrieval_result.decision

    # 只有门控放行后才允许存在LLM草稿。
    #
    # 如果门控已经拒绝，
    # 正确路径应该是build_gate_abstained_report()。
    if not decision.accepted:
        raise ValueError(
            "只有门控放行结果"
            "才能组装LLM诊断报告"
        )

    # 防御accepted和reason互相矛盾的对象。
    if (
        decision.reason
        not in ACCEPTED_GATE_REASONS
    ):
        raise ValueError(
            "门控放行结果包含不合法的reason"
        )

    # 建立：
    #
    # E1 -> rank=1真实候选
    # E2 -> rank=2真实候选
    #
    # 该函数还会检查空候选、重复Chunk和rank断裂。
    evidence_map = (
        build_diagnosis_evidence_map(
            evidence=(
                retrieval_result
                .retrieved_chunks
            ),
        )
    )

    referenced_evidence_ids = (
        _collect_draft_evidence_ids(
            draft
        )
    )

    allowed_evidence_ids = set(
        evidence_map
    )

    unknown_evidence_ids = sorted(
        set(referenced_evidence_ids)
        - allowed_evidence_ids
    )

    # E999可能符合E编号正则，
    # 但如果不在当前evidence_map中仍属于虚构引用。
    if unknown_evidence_ids:
        raise InvalidLLMResponseError(
            "结构化诊断引用了未提供的"
            "证据编号："
            + ", ".join(
                unknown_evidence_ids
            )
        )

    referenced_evidence_id_set = set(
        referenced_evidence_ids
    )

    if draft.abstained:
        # 门控可能认为候选足以进入分析，
        # 但模型阅读后发现证据互相冲突。
        #
        # 此时保留全部检索候选供审计，
        # 但DiagnosisLLMDraft契约已经保证
        # 原因和检查项为空。
        selected_candidates = list(
            evidence_map.values()
        )
    else:
        # 非拒答报告只返回模型实际引用的候选。
        #
        # evidence_map按照E1、E2顺序保存，
        # 因此筛选后的结果仍按RRF rank排列。
        selected_candidates = [
            candidate
            for (
                evidence_id,
                candidate,
            ) in evidence_map.items()
            if (
                evidence_id
                in referenced_evidence_id_set
            )
        ]

    report_evidence = [
        build_diagnosis_evidence(
            candidate
        )
        for candidate in selected_candidates
    ]

    report_causes = [
        DiagnosisCause(
            description=cause.description,

            # 真实Chunk ID只能从evidence_map复制。
            evidence_chunk_ids=[
                evidence_map[
                    evidence_id
                ].chunk.chunk_id
                for evidence_id
                in cause.evidence_ids
            ],
        )
        for cause in draft.possible_causes
    ]

    report_checks = [
        DiagnosisCheck(
            description=check.description,
            evidence_chunk_ids=[
                evidence_map[
                    evidence_id
                ].chunk.chunk_id
                for evidence_id
                in check.evidence_ids
            ],
            risk_level=check.risk_level,
            requires_qualified_person=(
                check
                .requires_qualified_person
            ),
        )
        for check in draft.next_checks
    ]

    # DiagnosisReport会再次执行最终契约校验：
    #
    # 1. status与abstained必须一致；
    # 2. 最终Chunk引用必须存在于evidence；
    # 3. 拒答不能包含原因或检查；
    # 4. partial必须说明信息缺口；
    # 5. completed不能仍有缺失信息；
    # 6. 高风险检查必须要求资质人员。
    return DiagnosisReport(
        request_id=request_id,
        prompt_version=(
            DIAGNOSIS_PROMPT_VERSION
        ),
        gate_version=(
            decision.policy_version
        ),
        status=draft.status,
        symptoms=(
            build_diagnosis_symptoms(
                request
            )
        ),
        evidence=report_evidence,
        possible_causes=report_causes,
        next_checks=report_checks,
        risk_level=draft.risk_level,

        # 创建新列表，避免最终报告和草稿
        # 共享同一个可变list对象。
        missing_information=list(
            draft.missing_information
        ),
        abstained=draft.abstained,
    )


def build_gate_abstained_report(
    *,
    request_id: str,
    request: DiagnosisRequest,
    retrieval_result: (
        GatedHybridRetrievalResult
    ),
) -> DiagnosisReport:
    """为门控拒绝生成不调用LLM的稳定报告。"""

    if not isinstance(
        request,
        DiagnosisRequest,
    ):
        raise TypeError(
            "request必须是DiagnosisRequest"
        )

    if not isinstance(
        retrieval_result,
        GatedHybridRetrievalResult,
    ):
        raise TypeError(
            "retrieval_result必须是"
            "GatedHybridRetrievalResult"
        )

    decision = retrieval_result.decision

    if decision.accepted:
        raise ValueError(
            "门控已经放行，"
            "不能生成门控拒答报告"
        )

    if (
        decision.reason
        not in REJECTED_GATE_REASONS
    ):
        raise ValueError(
            "门控拒绝结果包含不合法的reason"
        )

    if decision.reason == "no_candidates":
        missing_information = [
            "知识库未检索到可用于诊断的"
            "候选证据"
        ]
    else:
        missing_information = [
            "候选证据未达到组合门控要求，"
            "需要补充更具体的故障代码、"
            "脱敏日志或现场状态"
        ]

    # 未通过门控的候选不作为正式证据返回。
    #
    # 否则客户端可能把“被系统判定为不足的候选”
    # 误读为已经支持诊断结论的证据。
    return DiagnosisReport(
        request_id=request_id,

        # 明确表示本次根本没有调用Prompt和LLM。
        prompt_version=(
            NO_LLM_PROMPT_VERSION
        ),
        gate_version=(
            decision.policy_version
        ),
        status="abstained",
        symptoms=(
            build_diagnosis_symptoms(
                request
            )
        ),
        evidence=[],
        possible_causes=[],
        next_checks=[],
        risk_level="unknown",
        missing_information=(
            missing_information
        ),
        abstained=True,
    )


def build_generation_abstained_report(
    *,
    request_id: str,
    request: DiagnosisRequest,
    retrieval_result: (
        GatedHybridRetrievalResult
    ),
    failure_reason: (
        DiagnosisGenerationFailureReason
    ),
) -> DiagnosisReport:
    """为门控放行后的已知生成失败构造稳定拒答报告。"""

    # 类型注解不会自动拦截运行时传入的普通dict，
    # 因此继续在模块边界进行显式类型检查。
    if not isinstance(
        request,
        DiagnosisRequest,
    ):
        raise TypeError(
            "request必须是DiagnosisRequest"
        )

    if not isinstance(
        retrieval_result,
        GatedHybridRetrievalResult,
    ):
        raise TypeError(
            "retrieval_result必须是"
            "GatedHybridRetrievalResult"
        )

    decision = retrieval_result.decision

    # 该函数只处理：
    #
    # 门控已放行
    #   → 已经准备或尝试调用LLM
    #   → 生成阶段失败
    #
    # 如果门控已经拒绝，
    # 应调用build_gate_abstained_report()。
    if not decision.accepted:
        raise ValueError(
            "只有门控放行后"
            "才能生成诊断生成降级报告"
        )

    # 防御accepted=True却搭配了拒绝原因的矛盾对象。
    if (
        decision.reason
        not in ACCEPTED_GATE_REASONS
    ):
        raise ValueError(
            "门控放行结果包含不合法的reason"
        )

    # Literal不能提供运行时保护，
    # 所以这里仍然执行真正的白名单检查。
    if (
        failure_reason
        not in GENERATION_FAILURE_MESSAGES
    ):
        raise ValueError(
            "不支持的诊断生成降级原因"
        )

    # 复用现有证据映射函数。
    #
    # 它不仅建立E1、E2等映射，还会检查：
    #
    # 1. 候选是否为空；
    # 2. Chunk是否重复；
    # 3. rank是否从1开始且连续；
    # 4. 候选是否符合HybridRetrievedChunk契约。
    evidence_map = (
        build_diagnosis_evidence_map(
            evidence=(
                retrieval_result
                .retrieved_chunks
            ),
        )
    )

    # 门控已经认为这些候选具备足够支持，
    # 因此生成失败时仍保留它们用于审计。
    #
    # values()取得字典中的候选对象；
    # build_diagnosis_evidence()再把它们转换成
    # 对外的DiagnosisEvidence数据契约。
    report_evidence = [
        build_diagnosis_evidence(
            candidate
        )
        for candidate in evidence_map.values()
    ]

    return DiagnosisReport(
        request_id=request_id,

        # 该分支发生在门控放行并进入生成阶段之后，
        # 所以不能记录成not-invoked。
        prompt_version=(
            DIAGNOSIS_PROMPT_VERSION
        ),
        gate_version=(
            decision.policy_version
        ),
        status="abstained",
        symptoms=(
            build_diagnosis_symptoms(
                request
            )
        ),
        evidence=report_evidence,

        # LLM没有产出经过校验的诊断草稿，
        # 因而不能返回任何原因或操作建议。
        possible_causes=[],
        next_checks=[],
        risk_level="unknown",

        # 只使用固定公开消息，
        # 不使用底层异常的原始文本。
        missing_information=[
            GENERATION_FAILURE_MESSAGES[
                failure_reason
            ]
        ],
        abstained=True,
    )