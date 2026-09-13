"""证据约束结构化诊断的应用服务。

本模块负责：

1. 从DiagnosisRequest构造检索查询；
2. 把API检索范围转换成内部不可变范围；
3. 调用查询改写、混合检索和证据门控链；
4. 门控拒绝时跳过LLM并生成稳定拒答报告；
5. 门控放行时调用结构化诊断草稿提供者；
6. 将LLM临时证据编号映射为真实Chunk ID。

本模块不实现Embedding、关键词检索、RRF、
证据门控、Prompt、LLM客户端或HTTP处理。
"""


import logging

from typing import Protocol

from app.errors import (
    InvalidLLMResponseError,
    LLMTimeoutError,
    LLMUpstreamError,
)

from app.schemas.diagnosis_llm import (
    DiagnosisLLMDraft,
)
from app.schemas.diagnostics import (
    DiagnosisReport,
    DiagnosisRequest,
)
from app.schemas.retrieval import (
    HybridRetrievedChunk,
)
from app.services.diagnosis_report_builder import (
    DiagnosisGenerationFailureReason,
    build_diagnosis_report_from_draft,
    build_gate_abstained_report,
    build_generation_abstained_report,
)
from app.services.gated_hybrid_retrieval import (
    GatedHybridRetrievalResult,
)
from app.services.retrieval_scope import (
    RetrievalScopeFilter,
)

# __name__在当前文件中等于：
#
# app.services.diagnosis_service
#
# 使用模块名创建Logger后，
# 日志可以精确定位到产生它的模块。
logger = logging.getLogger(__name__)

# 诊断请求不允许由客户端任意扩大候选深度。
#
# 服务端使用固定默认值控制：
#
# 1. Prompt证据数量；
# 2. LLM输入长度；
# 3. 引用审计复杂度；
# 4. 检索与生成成本。
DEFAULT_DIAGNOSIS_TOP_K = 3


class DiagnosisRetrievalProvider(
    Protocol
):
    """诊断Service所需的最小门控检索能力。"""

    async def retrieve(
        self,
        *,
        query: str,
        top_k: int = 3,
        retrieval_scope: (
            RetrievalScopeFilter | None
        ) = None,
    ) -> GatedHybridRetrievalResult:
        """返回查询改写、混合候选和门控决定。"""

        ...


class DiagnosisDraftProvider(
    Protocol
):
    """诊断Service所需的最小LLM草稿能力。"""

    async def generate_draft(
        self,
        *,
        request: DiagnosisRequest,
        evidence: tuple[
            HybridRetrievedChunk,
            ...,
        ],
    ) -> DiagnosisLLMDraft:
        """根据请求和真实候选生成结构化诊断草稿。"""

        ...


def _validate_retrieval_top_k(
    value: object,
) -> int:
    """校验服务端固定的诊断候选深度。"""

    # bool是int的子类：
    #
    # isinstance(True, int) == True
    #
    # 但True不能表示有意义的候选数量，
    # 因而需要在整数检查之前单独排除。
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
    ):
        raise TypeError(
            "retrieval_top_k必须是整数"
        )

    if value < 1:
        raise ValueError(
            "retrieval_top_k必须大于或等于1"
        )

    return value


def _clean_request_id(
    request_id: object,
) -> str:
    """在调用检索和LLM前校验请求追踪标识。"""

    if not isinstance(
        request_id,
        str,
    ):
        raise TypeError(
            "request_id必须是字符串"
        )

    cleaned_request_id = (
        request_id.strip()
    )

    if not cleaned_request_id:
        raise ValueError(
            "request_id不能为空"
        )

    # 与DiagnosisReport.request_id的数据契约保持一致。
    #
    # 提前检查可以避免完成检索和LLM调用后，
    # 才在最终报告构造阶段发现请求标识非法。
    if len(cleaned_request_id) > 200:
        raise ValueError(
            "request_id不能超过200个字符"
        )

    return cleaned_request_id


def build_diagnosis_retrieval_query(
    request: DiagnosisRequest,
) -> str:
    """用现象和日志构造知识库检索查询。"""

    if not isinstance(
        request,
        DiagnosisRequest,
    ):
        raise TypeError(
            "request必须是DiagnosisRequest"
        )

    # symptom包含用户观察到的现象；
    # log_excerpt可能包含错误码、型号、阈值和状态。
    #
    # 使用换行保留两个输入字段的边界，
    # 后续查询归一化会继续处理空白和专有词。
    #
    # robot_id通常只是具体机器人实例编号，
    # 不是知识库中的故障语义。
    # 将它加入检索文本可能引入无关词并降低召回质量。
    return (
        request.symptom
        + "\n"
        + request.log_excerpt
    )


def build_diagnosis_retrieval_scope(
    request: DiagnosisRequest,
) -> RetrievalScopeFilter | None:
    """把API范围转换成内部不可变检索范围。"""

    if not isinstance(
        request,
        DiagnosisRequest,
    ):
        raise TypeError(
            "request必须是DiagnosisRequest"
        )

    api_scope = request.retrieval_scope

    # None明确表示不限制文档范围。
    #
    # 不应创建两个字段都为空的RetrievalScopeFilter，
    # 因为该内部契约要求至少存在一个真实条件。
    if api_scope is None:
        return None

    # DiagnosisRetrievalScope使用list，
    # 因为JSON数组会被Pydantic解析为Python列表。
    #
    # 内部RetrievalScopeFilter使用tuple和frozen=True，
    # 避免请求对象后续被修改时改变正在执行的检索。
    return RetrievalScopeFilter(
        document_ids=tuple(
            api_scope.document_ids
        ),
        source_files=tuple(
            api_scope.source_files
        ),
    )


class DiagnosisService:
    """编排门控检索、诊断生成和最终报告构造。"""

    def __init__(
        self,
        *,
        retrieval_provider: (
            DiagnosisRetrievalProvider
        ),
        draft_provider: (
            DiagnosisDraftProvider
        ),
        retrieval_top_k: int = (
            DEFAULT_DIAGNOSIS_TOP_K
        ),
    ) -> None:
        """保存依赖并校验固定检索策略。"""

        # 正式运行时通常注入：
        #
        # retrieval_provider：
        #     GatedHybridRetriever
        #
        # draft_provider：
        #     OpenAIDiagnosisDraftProvider
        #
        # 离线测试时可以注入Fake，
        # 所以当前类不绑定具体实现类型。
        self._retrieval_provider = (
            retrieval_provider
        )
        self._draft_provider = (
            draft_provider
        )

        self._retrieval_top_k = (
            _validate_retrieval_top_k(
                retrieval_top_k
            )
        )

    async def diagnose(
        self,
        request: DiagnosisRequest,
        request_id: str,
    ) -> DiagnosisReport:
        """根据请求和真实知识库证据生成结构化诊断。"""

        # 类型注解不会在Python运行时拒绝普通dict。
        #
        # 必须在任何外部调用前确认请求已经通过
        # DiagnosisRequest的Pydantic数据契约。
        if not isinstance(
            request,
            DiagnosisRequest,
        ):
            raise TypeError(
                "request必须是DiagnosisRequest"
            )

        cleaned_request_id = (
            _clean_request_id(
                request_id
            )
        )

        retrieval_query = (
            build_diagnosis_retrieval_query(
                request
            )
        )

        retrieval_scope = (
            build_diagnosis_retrieval_scope(
                request
            )
        )

        # 下游会依次完成：
        #
        # 1. 确定性查询改写；
        # 2. 查询Embedding；
        # 3. 范围内向量与关键词召回；
        # 4. RRF融合；
        # 5. 组合证据门控。
        retrieval_result = (
            await self._retrieval_provider.retrieve(
                query=retrieval_query,
                top_k=self._retrieval_top_k,
                retrieval_scope=retrieval_scope,
            )
        )

        # Protocol不会自动检查真实运行时返回值。
        #
        # 如果错误依赖返回dict或None，
        # 应在调用LLM之前立即停止。
        if not isinstance(
            retrieval_result,
            GatedHybridRetrievalResult,
        ):
            raise TypeError(
                "retrieval_provider必须返回"
                "GatedHybridRetrievalResult"
            )

        # 门控未放行说明当前证据不够可靠。
        #
        # 此分支绝不能调用LLM：
        #
        # 1. 防止模型在弱证据上猜测；
        # 2. 节省外部调用成本；
        # 3. 保证拒答原因由确定性Python规则产生。
        if not retrieval_result.decision.accepted:
            return build_gate_abstained_report(
                request_id=cleaned_request_id,
                request=request,
                retrieval_result=(
                    retrieval_result
                ),
            )

        # 只有门控放行后才进入生成阶段。
        #
        # 该变量只能保存Report Builder认识的
        # 三种稳定失败原因代码。
        failure_reason: (
            DiagnosisGenerationFailureReason
        )

        # 保存项目异常已经清理过的诊断信息。
        # 该内容只写入服务端日志，
        # 不会进入返回给客户端的DiagnosisReport。
        failure_detail: str

        try:
            # 将已经通过门控的真实候选交给
            # 结构化诊断草稿提供者。
            draft = (
                await self._draft_provider
                .generate_draft(
                    request=request,
                    evidence=(
                        retrieval_result
                        .retrieved_chunks
                    ),
                )
            )

            # Protocol和返回类型注解不会自动执行
            # Python运行时类型校验。
            #
            # 返回dict或None属于依赖实现错误，
            # 不应该伪装成正常的LLM降级。
            if not isinstance(
                draft,
                DiagnosisLLMDraft,
            ):
                raise TypeError(
                    "draft_provider必须返回"
                    "DiagnosisLLMDraft"
                )

            # 必须把报告组装也放在try中。
            #
            # LLM可能返回符合Schema的E999，
            # 但E999并不在本次真实证据白名单中。
            # Builder会为此抛出
            # InvalidLLMResponseError。
            return (
                build_diagnosis_report_from_draft(
                    request_id=(
                        cleaned_request_id
                    ),
                    request=request,
                    retrieval_result=(
                        retrieval_result
                    ),
                    draft=draft,
                )
            )

        except InvalidLLMResponseError as exc:
            # 覆盖非法JSON、缺失字段、
            # Schema冲突和虚构证据引用。
            failure_reason = (
                "invalid_llm_response"
            )

            # str(exc)取得项目异常已经整理过的消息。
            # 原始SDK异常和响应体不会进入公开报告。
            failure_detail = str(exc)

        except LLMTimeoutError as exc:
            # LLM没有在规定时间内返回结果。
            failure_reason = "llm_timeout"
            failure_detail = str(exc)

        except LLMUpstreamError as exc:
            # LLM连接失败或上游返回错误状态。
            failure_reason = (
                "llm_upstream_error"
            )
            failure_detail = str(exc)

        # 只有上面三种已知错误会运行到这里。
        #
        # TypeError、RuntimeError等程序错误
        # 没有被捕获，仍然会向外抛出。
        #
        # 参数化日志使用%s占位符；
        # logging只在该级别真正需要输出时
        # 才完成最终字符串格式化。
        logger.warning(
            "diagnosis_generation_degraded "
            "request_id=%s reason=%s detail=%s",
            cleaned_request_id,
            failure_reason,
            failure_detail,
        )

        return build_generation_abstained_report(
            request_id=cleaned_request_id,
            request=request,
            retrieval_result=retrieval_result,
            failure_reason=failure_reason,
        )
