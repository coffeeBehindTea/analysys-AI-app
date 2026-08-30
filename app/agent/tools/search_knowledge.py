"""Agent知识库纯证据检索工具。

本模块把第三周的门控混合检索能力
适配成Week 4统一ToolHandler协议。

它负责：
1. 调用查询改写、混合检索和确定性证据门控；
2. 只选择真正触发门控放行的Chunk；
3. 由Python构造真实引用；
4. 将真实引用记录到当前请求的证据白名单。

它不调用知识回答LLM，不生成最终诊断，
也不重新实现Embedding、关键词检索、RRF或查询改写。
"""

from inspect import (
    iscoroutinefunction,
)
from time import (
    perf_counter,
)
from typing import (
    Protocol,
)

from pydantic import (
    BaseModel,
)

from app.agent.evidence_store import (
    ConfirmedEvidenceStore,
)
from app.schemas.agent import (
    ToolDefinition,
)
from app.schemas.agent_tools import (
    SearchKnowledgeToolInput,
    SearchKnowledgeToolOutput,
)
from app.schemas.retrieval import (
    HybridRetrievedChunk,
)
from app.services.gated_hybrid_retrieval import (
    GatedHybridRetrievalResult,
)
from app.services.hybrid_knowledge_query import (
    build_hybrid_citations,
)
from app.services.hybrid_retrieval_gate import (
    HybridEvidenceGateDecision,
)
from app.services.retrieval_scope import (
    RetrievalScopeFilter,
)


class KnowledgeRetrievalProvider(
    Protocol
):
    """search_knowledge依赖的最小检索能力。

    正式运行时由GatedHybridRetriever满足；
    离线测试时可以注入具有相同异步retrieve()
    方法的Fake对象。

    Protocol只描述所需方法，
    不要求依赖继承某个具体父类。
    """

    async def retrieve(
        self,
        *,
        query: str,
        top_k: int = 3,
        retrieval_scope: (
            RetrievalScopeFilter | None
        ) = None,
    ) -> GatedHybridRetrievalResult:
        """执行查询改写、混合检索和证据门控。"""

        ...


class SearchKnowledgeToolHandler:
    """把门控混合检索适配成Agent工具Handler。

    本类只处理search_knowledge特有的业务逻辑。

    工具名检查、输入Schema校验、超时控制、
    异常转换和最终输出模型校验仍由
    ToolExecutor统一负责。
    """

    def __init__(
        self,
        *,
        retrieval_provider: (
            KnowledgeRetrievalProvider
        ),
        evidence_store: (
            ConfirmedEvidenceStore
        ),
    ) -> None:
        """保存门控检索器和当前请求的证据Store。"""

        # getattr()读取对象属性。
        #
        # 如果对象没有retrieve属性，
        # 第三个参数None会作为默认返回值。
        retrieve_method = getattr(
            retrieval_provider,
            "retrieve",
            None,
        )

        # iscoroutinefunction()来自Python标准库inspect。
        #
        # 它检查retrieve是否使用async def定义。
        # 检索过程包含异步Embedding请求，
        # 因此不能接受同步方法冒充正式依赖。
        if not iscoroutinefunction(
            retrieve_method
        ):
            raise TypeError(
                "retrieval_provider.retrieve"
                "必须是异步方法"
            )

        if not isinstance(
            evidence_store,
            ConfirmedEvidenceStore,
        ):
            raise TypeError(
                "evidence_store必须是"
                "ConfirmedEvidenceStore"
            )

        self._retrieval_provider = (
            retrieval_provider
        )
        self._evidence_store = (
            evidence_store
        )

    async def __call__(
        self,
        tool_input: BaseModel,
        /,
    ) -> BaseModel | None:
        """执行一次门控知识证据检索。

        门控拒绝时返回None，
        由ToolExecutor转换成empty_result。

        门控放行时，只返回并记录真正支持
        放行决定的Chunk，不把其他背景候选
        自动提升为已确认证据。
        """

        # ToolExecutor正常情况下已经使用
        # SearchKnowledgeToolInput.model_validate()
        # 完成参数校验。
        #
        # 这里继续保留防御性类型检查，
        # 防止其他代码绕过Executor直接调用Handler。
        if not isinstance(
            tool_input,
            SearchKnowledgeToolInput,
        ):
            raise TypeError(
                "tool_input必须是"
                "SearchKnowledgeToolInput"
            )

        # perf_counter()是高精度单调计时器，
        # 适合测量经过时间。
        #
        # 它不会因为系统时间被人工调整而倒退。
        retrieval_started_at = (
            perf_counter()
        )

        # 直接调用应用内部检索对象，
        # 不通过HTTP请求自己的知识库Router。
        #
        # retrieval_scope=None明确表示在当前
        # 已配置Collection中执行全库检索。
        retrieval_result = (
            await self
            ._retrieval_provider
            .retrieve(
                query=tool_input.query,
                top_k=tool_input.top_k,
                retrieval_scope=None,
            )
        )

        retrieval_ms = (
            perf_counter()
            - retrieval_started_at
        ) * 1_000

        # Protocol和返回类型标注不会在运行时
        # 自动阻止dict、None等错误对象。
        #
        # 因此必须在访问decision属性前检查类型。
        if not isinstance(
            retrieval_result,
            GatedHybridRetrievalResult,
        ):
            raise TypeError(
                "retrieval_provider.retrieve"
                "必须返回"
                "GatedHybridRetrievalResult"
            )

        decision = retrieval_result.decision

        # GatedHybridRetrievalResult是dataclass，
        # 不会像Pydantic模型一样自动递归校验
        # 调用方传入的每个内部对象。
        if not isinstance(
            decision,
            HybridEvidenceGateDecision,
        ):
            raise TypeError(
                "retrieval_result.decision"
                "必须是"
                "HybridEvidenceGateDecision"
            )

        # 门控拒绝时，supporting_chunk_ids
        # 按契约必须为空。
        #
        # 如果错误Fake或替代实现同时声称
        # “拒绝”与“存在支持证据”，应立即报错，
        # 而不是静默选择其中一种语义。
        if not decision.accepted:
            if decision.supporting_chunk_ids:
                raise ValueError(
                    "门控拒绝结果不能包含"
                    "supporting_chunk_ids"
                )

            return None

        # 门控放行必须明确说明：
        # 到底是哪些Chunk触发了放行规则。
        #
        # 不能因为Top-K中存在若干候选，
        # 就把全部候选都写入证据白名单。
        if not decision.supporting_chunk_ids:
            raise ValueError(
                "门控放行结果必须包含"
                "supporting_chunk_ids"
            )

        candidates_by_id: dict[
            str,
            HybridRetrievedChunk,
        ] = {}

        for position, candidate in enumerate(
            retrieval_result.retrieved_chunks
        ):
            # dataclass外层的类型注解不会自动
            # 检查tuple中的每一个成员。
            if not isinstance(
                candidate,
                HybridRetrievedChunk,
            ):
                raise TypeError(
                    "retrieved_chunks中的第 "
                    f"{position} 项必须是"
                    "HybridRetrievedChunk"
                )

            chunk_id = candidate.chunk.chunk_id

            # 一个检索结果中不能用同一个Chunk ID
            # 表示两个不同候选。
            if chunk_id in candidates_by_id:
                raise ValueError(
                    "检索候选不能包含重复"
                    f"Chunk ID：{chunk_id}"
                )

            candidates_by_id[chunk_id] = (
                candidate
            )

        # 检查门控声称支持的每个Chunk，
        # 是否真的存在于本次检索候选中。
        unknown_supporting_ids = [
            chunk_id
            for chunk_id
            in decision.supporting_chunk_ids
            if chunk_id not in candidates_by_id
        ]

        if unknown_supporting_ids:
            raise ValueError(
                "门控支持的Chunk ID"
                "不存在于检索候选："
                + ", ".join(
                    unknown_supporting_ids
                )
            )

        # 按门控给出的supporting_chunk_ids
        # 选择真正支持放行的候选。
        #
        # 其他Top-K候选可能只是背景信息，
        # 不能自动成为“已确认证据”。
        supporting_candidates = [
            candidates_by_id[chunk_id]
            for chunk_id
            in decision.supporting_chunk_ids
        ]

        # build_hybrid_citations()复用第三周
        # 已经过测试的Python引用构造逻辑。
        #
        # 文件名、页码、Chunk ID、正文、向量分数
        # 和RRF分数全部来自真实检索候选，
        # 不由Planner或其他LLM生成。
        citations = build_hybrid_citations(
            supporting_candidates
        )

        # 先完成输出模型校验，再写入证据Store。
        #
        # 如果输出违反Schema，
        # 当前请求的证据白名单不会留下半成品。
        tool_output = (
            SearchKnowledgeToolOutput(
                citations=citations,
                retrieval_ms=retrieval_ms,
            )
        )

        # record()执行原子式记录：
        #
        # 任意引用不合法时，整组引用都不会
        # 部分写入当前请求的证据白名单。
        self._evidence_store.record(
            tool_output.citations
        )

        return tool_output


# 这是search_knowledge工具的静态定义。
#
# 应用装配阶段会把该定义与
# SearchKnowledgeToolHandler实例一起
# 注册到ToolRegistry白名单。
SEARCH_KNOWLEDGE_TOOL_DEFINITION = (
    ToolDefinition(
        name="search_knowledge",
        description=(
            "从已经摄取的机器人知识库中检索"
            "经过确定性门控确认的真实Chunk和引用。"
            "工具不生成最终回答；调用者应阅读citations"
            "中的原始证据。"
            "query应聚焦一个需要从文档确认的知识问题，"
            "不要混入实时位置、电量、当前任务或"
            "测试草案生成要求。"
            "适用于故障代码、设备手册、安全标准和"
            "测试规程。"
            "知识库证据不足时返回空结果。"
        ),
        input_model=(
            SearchKnowledgeToolInput
        ),
        output_model=(
            SearchKnowledgeToolOutput
        ),

        # 工具只读取已经摄取的知识库，
        # 不修改文档、机器人或外部系统状态。
        risk_level="low",
        read_only=True,

        # 超时覆盖查询Embedding、向量检索、
        # 关键词检索、RRF、重排和确定性门控。
        #
        # 当前调用链已经不包含知识回答LLM。
        timeout_seconds=120.0,
    )
)