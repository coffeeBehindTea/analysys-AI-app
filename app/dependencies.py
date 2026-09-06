"""FastAPI依赖装配函数。

这里集中决定配置、SDK客户端、Provider、Service和Agent组件
如何创建、连接与释放，但不实现具体业务。
"""

# AsyncIterator[T] 表示可以通过 async for 消费、逐步产生 T 的异步迭代器。
# FastAPI 也用这种返回注解识别含 yield 的异步资源依赖。
from collections.abc import AsyncIterator

# Annotated[真实类型, 元数据] 用于在类型上附加 Depends 等框架信息。
from typing import Annotated

# Depends 声明“此参数由 FastAPI 调用指定函数提供”，而不是由客户端传入。
from fastapi import Depends

# AsyncOpenAI 是 OpenAI Python SDK 的异步客户端类型。
from openai import AsyncOpenAI

from datetime import (
    datetime,
    timezone,
)
from functools import (
    lru_cache,
)

from app.config import Settings, get_settings
from app.services.llm_client import create_llm_client, get_llm_model
from app.services.triage import TriageService

from app.services.document_catalog import (
    DocumentCatalogService,
)
from app.services.vector_store import (
    ChromaVectorStore,
)

from app.services.chunking import TextChunker
from app.services.document_preparation import (
    TextDocumentPreparationService,
)
from app.services.document_service import (
    DocumentService,
)
from app.services.embedding import EmbeddingService
from app.services.embedding_client import (
    create_embedding_client,
    get_embedding_model,
)
from app.services.vision_client import (
    create_vision_client,
    get_vision_model,
)
from app.services.vision_input import (
    VisionInputAdapter,
)
from app.services.vision_provider import (
    OpenAICompatibleVisionProvider,
    VisionProvider,
)

from app.agent.demo_telemetry import (
    build_demo_robot_telemetry_store,
)
from app.agent.evidence_store import (
    ConfirmedEvidenceStore,
)
from app.agent.executor import (
    ToolExecutor,
)
from app.agent.vision_input_store import (
    RequestVisionInputStore,
)
# OpenAICompatibleAgentPlanner把OpenAI兼容SDK响应
# 转换成Runner认识的AgentPlannerDecision。
from app.agent.openai_planner import (
    OpenAICompatibleAgentPlanner,
)

# AgentRunner负责规划、工具执行和观察的受控循环。
from app.agent.runner import (
    AgentRunner,
)
from app.agent.registry import (
    ToolRegistry,
)
from app.agent.tools import (
    ANALYZE_ROBOT_IMAGE_TOOL_DEFINITION,
    DRAFT_TEST_CASE_TOOL_DEFINITION,
    GET_CURRENT_TIME_TOOL_DEFINITION,
    GET_ROBOT_TELEMETRY_TOOL_DEFINITION,
    SEARCH_KNOWLEDGE_TOOL_DEFINITION,
    AnalyzeRobotImageToolHandler,
    DraftTestCaseToolHandler,
    GetCurrentTimeToolHandler,
    GetRobotTelemetryToolHandler,
    InMemoryRobotTelemetryStore,
    SearchKnowledgeToolHandler,
    SystemUtcClock,
)


from app.services.knowledge_answer import (
    OpenAIKnowledgeAnswerProvider,
)
# HybridKnowledgeQueryService负责：
#
# 查询改写 → 混合检索 → RRF → 证据门控
# → LLM回答 → 混合引用构造。
from app.services.hybrid_knowledge_query import (
    HybridKnowledgeQueryService,
)
from app.services.diagnosis_generation import (
    OpenAIDiagnosisDraftProvider,
)
from app.services.diagnosis_service import (
    DiagnosisService,
)
# AgentDiagnosisService负责把AgentRunner结果、
# 工具观察、证据白名单和公开响应组合起来。
from app.services.agent_diagnosis_service import (
    AgentDiagnosisService,
)
from app.services.gated_hybrid_retrieval import (
    GatedHybridRetriever,
)
from app.services.head_preserving_reranking import (
    HeadPreservationPolicy,
)
from app.services.hybrid_retrieval import (
    HybridRetriever,
)
from app.services.hybrid_retrieval_gate import (
    HybridEvidenceGatePolicy,
)
from app.services.keyword_retrieval import (
    KeywordRetriever,
)
from app.services.query_rewriting_retrieval import (
    QueryRewritingHybridRetriever,
)


async def get_llm_client(
    settings: Annotated[Settings, Depends(get_settings)],
) -> AsyncIterator[AsyncOpenAI]:
    """为一次 HTTP 请求创建并最终关闭异步 LLM 客户端。

    settings 参数由 FastAPI 调用 get_settings() 注入。
    yield 产生的 AsyncOpenAI 会继续注入下游依赖。
    """

    # 工厂函数负责检查配置，并构造真正的 SDK 客户端。
    client = create_llm_client(settings)

    try:
        # yield 暂停当前依赖，把 client 提供给 get_triage_service()。
        yield client
    finally:
        # close() 是 AsyncOpenAI 的异步资源清理方法，因此必须 await。
        # finally 保证正常响应或异常时都尽量关闭其 HTTP 连接资源。
        await client.close()


def get_triage_service(
    client: Annotated[AsyncOpenAI, Depends(get_llm_client)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> TriageService:
    """使用已注入的客户端和 Settings 构造故障分析 Service。"""

    # get_llm_model() 负责检查模型名；TriageService 只接收可用依赖。
    return TriageService(
        client=client,
        model=get_llm_model(settings),
    )


async def get_embedding_client(
    settings: Annotated[
        Settings,
        Depends(get_settings),
    ],
) -> AsyncIterator[AsyncOpenAI]:
    """为一次请求创建并最终关闭 Embedding 客户端。"""

    # create_embedding_client() 使用独立的：
    # EMBEDDING_API_KEY、EMBEDDING_BASE_URL。
    #
    # 创建客户端本身不会发送网络请求。
    client = create_embedding_client(settings)

    try:
        # FastAPI 会把 client 注入下游的
        # get_document_service()。
        yield client
    finally:
        # 无论上传成功、校验失败还是上游异常，
        # 都关闭 AsyncOpenAI 持有的 HTTP 连接池。
        await client.close()


async def get_vision_client(
    settings: Annotated[
        Settings,
        Depends(get_settings),
    ],
) -> AsyncIterator[AsyncOpenAI]:
    """为一次请求创建并最终关闭Vision异步客户端。

    FastAPI会执行到yield，把客户端注入下游依赖；
    请求结束后继续执行finally并关闭HTTP连接资源。
    """

    # 工厂函数负责校验独立的Vision API Key、
    # 处理可选Base URL，并传入SDK层超时。
    #
    # 创建客户端本身不会发送图片或访问网络。
    client = create_vision_client(
        settings
    )

    try:
        # yield把当前客户端提供给
        # get_vision_provider()。
        yield client
    finally:
        # 无论下游成功、拒答还是抛出异常，
        # 都关闭AsyncOpenAI持有的异步HTTP连接池。
        await client.close()


def get_vision_input_adapter(
    settings: Annotated[
        Settings,
        Depends(get_settings),
    ],
) -> VisionInputAdapter:
    """使用当前图片资源限制创建Vision输入适配器。"""

    # Adapter不读取.env，也不知道Settings的存在。
    # 依赖层把配置值转换成它明确需要的构造参数。
    return VisionInputAdapter(
        max_image_size_bytes=(
            settings.max_vision_image_size_bytes
        ),
        max_image_dimension_px=(
            settings.max_vision_image_dimension_px
        ),
        max_image_pixels=(
            settings.max_vision_image_pixels
        ),
    )


def get_vision_provider(
    client: Annotated[
        AsyncOpenAI,
        Depends(get_vision_client),
    ],
    settings: Annotated[
        Settings,
        Depends(get_settings),
    ],
) -> OpenAICompatibleVisionProvider:
    """组装真实OpenAI兼容Vision Provider。"""

    # get_vision_model()在Provider创建前
    # 检查VISION_MODEL是否存在且不是空字符串。
    #
    # Provider只接收已经可用的客户端、模型和超时，
    # 不直接读取环境变量，也不管理客户端生命周期。
    return OpenAICompatibleVisionProvider(
        client=client,
        model=get_vision_model(
            settings
        ),
        timeout_seconds=(
            settings.vision_timeout_seconds
        ),
    )


def get_vector_store(
    settings: Annotated[
        Settings,
        Depends(get_settings),
    ],
) -> ChromaVectorStore:
    """打开并校验当前配置指定的 Chroma Collection。"""

    return ChromaVectorStore(
        persist_directory=(
            settings.chroma_persist_directory
        ),
        collection_name=(
            settings.chroma_collection_name
        ),
        embedding_model=get_embedding_model(settings),
    )


def get_document_catalog_service(
    vector_store: Annotated[
        ChromaVectorStore,
        Depends(get_vector_store),
    ],
) -> DocumentCatalogService:
    """使用共享的 VectorStore 构造文档目录服务。"""

    return DocumentCatalogService(
        vector_store=vector_store,
    )


def get_document_service(
    client: Annotated[
        AsyncOpenAI,
        Depends(get_embedding_client),
    ],
    vector_store: Annotated[
        ChromaVectorStore,
        Depends(get_vector_store),
    ],
    settings: Annotated[
        Settings,
        Depends(get_settings),
    ],
) -> DocumentService:
    """装配真实的文档摄取服务。"""

    # 取得经过非空校验的模型名称。
    embedding_model = get_embedding_model(
        settings
    )

    # EmbeddingService 负责把一批文本发送给
    # OpenAI 兼容 Embedding API，并校验返回向量。
    embedding_service = EmbeddingService(
        client=client,
        model=embedding_model,
    )

    # PreparationService 负责：
    # 文件类型路由 → Loader → SourceTextSegment → Chunk。
    preparation_service = (
        TextDocumentPreparationService(
            chunker=TextChunker(
                chunk_size=settings.rag_chunk_size,
                overlap=settings.rag_chunk_overlap,
            )
        )
    )

    return DocumentService(
        preparation_service=preparation_service,
        embedding_provider=embedding_service,
        vector_store=vector_store,
        embedding_model=embedding_model,
        batch_size=settings.embedding_batch_size,
        max_file_size_bytes=(
            settings.max_document_size_bytes
        ),
    )


def get_knowledge_query_service(
    # 查询改写阶段需要调用Embedding接口。
    embedding_client: Annotated[
        AsyncOpenAI,
        Depends(get_embedding_client),
    ],

    # 门控放行后使用LLM生成证据约束回答。
    llm_client: Annotated[
        AsyncOpenAI,
        Depends(get_llm_client),
    ],

    # 同一个VectorStore同时用于：
    #
    # 1. Chroma向量候选召回；
    # 2. 读取Chunk并建立关键词索引。
    vector_store: Annotated[
        ChromaVectorStore,
        Depends(get_vector_store),
    ],

    # Settings提供Embedding和LLM模型名称。
    settings: Annotated[
        Settings,
        Depends(get_settings),
    ],
) -> HybridKnowledgeQueryService:
    """装配在线知识库混合RAG问答服务。"""

    # 复用诊断API也使用的通用混合检索工厂。
    #
    # 该工厂会装配：
    #
    # EmbeddingService
    # → KeywordRetriever
    # → HybridRetriever
    # → QueryRewritingHybridRetriever
    # → GatedHybridRetriever
    gated_retriever = (
        build_gated_hybrid_retrieval_provider(
            embedding_client=embedding_client,
            vector_store=vector_store,
            settings=settings,
        )
    )

    # Answer Provider只负责：
    #
    # 1. 把问题和已经通过门控的证据构造成Prompt；
    # 2. 调用OpenAI兼容Chat API；
    # 3. 解析并校验KnowledgeAnswerDraft。
    answer_provider = (
        OpenAIKnowledgeAnswerProvider(
            client=llm_client,

            # get_llm_model()负责检查模型名
            # 已配置且不是空字符串。
            model=get_llm_model(settings),
        )
    )

    # HybridKnowledgeQueryService负责高层业务编排：
    #
    # 1. 调用门控混合检索；
    # 2. 拒绝弱证据；
    # 3. 放行后调用回答Provider；
    # 4. 校验E1、E2证据白名单；
    # 5. 构造带RRF信息的真实引用。
    return HybridKnowledgeQueryService(
        retrieval_provider=gated_retriever,
        answer_provider=answer_provider,
    )


def build_gated_hybrid_retrieval_provider(
    *,
    embedding_client: AsyncOpenAI,
    vector_store: ChromaVectorStore,
    settings: Settings,
) -> GatedHybridRetriever:
    """装配可供问答、诊断和评测复用的门控混合检索链。

    这是一个普通Python工厂，不依赖FastAPI的Depends。

    因此它可以同时被：

    1. get_knowledge_query_service()；
    2. build_diagnosis_retrieval_provider()；
    3. Prompt对照实验脚本；
    4. 离线装配测试

    直接或间接调用。
    """

    # 查询向量必须使用与Collection相同的
    # Embedding模型，否则查询和文档不在同一向量空间。
    embedding_model = get_embedding_model(
        settings
    )

    # EmbeddingService只保存客户端和模型配置。
    #
    # 此处不会联网；真正的Embedding请求
    # 发生在retrieve()调用链中。
    embedding_provider = EmbeddingService(
        client=embedding_client,
        model=embedding_model,
    )

    # 从当前Collection取得全部Chunk正文和元数据，
    # 为确定性的关键词检索建立内存索引。
    #
    # list_chunks()不会加载全部Embedding向量。
    keyword_retriever = KeywordRetriever(
        chunks=vector_store.list_chunks(),
    )

    # HybridRetriever组合：
    #
    # 1. Chroma向量召回；
    # 2. 内存关键词召回；
    # 3. RRF排名融合；
    # 4. 头部保留重排。
    #
    # 未显式传参时继续使用已经评测的：
    #
    # candidate_k=20
    # rank_constant=60
    # fused_head_k=1
    # vector_head_k=1
    # keyword_head_k=2
    #
    # HeadPreservationPolicy()的默认值
    # 对应已通过候选召回和门控安全评测的
    # head-preserving-v1。
    hybrid_retriever = HybridRetriever(
        vector_retriever=vector_store,
        keyword_retriever=keyword_retriever,
        embedding_model=embedding_model,
        rerank_policy=(
            HeadPreservationPolicy()
        ),
    )

    # 查询改写检索器负责：
    #
    # 原始查询
    #   → 确定性术语改写
    #   → 查询Embedding
    #   → 混合检索
    #   → 头部保留重排。
    rewriting_retriever = (
        QueryRewritingHybridRetriever(
            embedding_provider=(
                embedding_provider
            ),
            hybrid_retriever=(
                hybrid_retriever
            ),
        )
    )

    # 最外层门控器检查融合候选是否满足
    # 进入生成式LLM的证据条件。
    #
    # HybridEvidenceGatePolicy()继续使用
    # hybrid-evidence-gate-v1默认配置。
    return GatedHybridRetriever(
        retrieval_provider=(
            rewriting_retriever
        ),
        policy=HybridEvidenceGatePolicy(),
    )


def get_agent_search_retrieval_provider(
    # search_knowledge仍然需要Embedding，
    # 因为查询文本必须转换成与文档相同向量空间中的向量。
    embedding_client: Annotated[
        AsyncOpenAI,
        Depends(get_embedding_client),
    ],

    # 同一个VectorStore同时提供：
    #
    # 1. Chroma向量召回；
    # 2. 全部Chunk正文，用于建立内存关键词索引。
    vector_store: Annotated[
        ChromaVectorStore,
        Depends(get_vector_store),
    ],

    # Settings提供Embedding模型名称、
    # Collection路径和检索相关配置。
    settings: Annotated[
        Settings,
        Depends(get_settings),
    ],
) -> GatedHybridRetriever:
    """装配Agent search_knowledge使用的纯检索链。

    当前依赖只创建：

    EmbeddingService
    → KeywordRetriever
    → HybridRetriever
    → QueryRewritingHybridRetriever
    → GatedHybridRetriever

    它不会创建OpenAIKnowledgeAnswerProvider，
    因而search_knowledge内部不会再次调用生成式LLM。
    """

    # 复用第三周已经过召回评测和安全门控评测的
    # 通用检索工厂，避免为Agent复制第二套检索参数。
    return build_gated_hybrid_retrieval_provider(
        embedding_client=embedding_client,
        vector_store=vector_store,
        settings=settings,
    )


def build_diagnosis_retrieval_provider(
    *,
    embedding_client: AsyncOpenAI,
    vector_store: ChromaVectorStore,
    settings: Settings,
) -> GatedHybridRetriever:
    """兼容原有诊断代码使用的检索工厂名称。

    真正的组件装配已经移动到通用工厂。
    保留这个包装函数是为了避免诊断API、
    Prompt对照实验和已有导入路径同时失效。
    """

    return (
        build_gated_hybrid_retrieval_provider(
            embedding_client=embedding_client,
            vector_store=vector_store,
            settings=settings,
        )
    )


def get_diagnosis_service(
    # 查询改写阶段需要调用Embedding接口。
    #
    # FastAPI会先执行get_embedding_client()，
    # 再把客户端注入当前参数。
    embedding_client: Annotated[
        AsyncOpenAI,
        Depends(get_embedding_client),
    ],

    # 门控放行后，诊断草稿Provider
    # 使用该客户端调用生成式LLM。
    llm_client: Annotated[
        AsyncOpenAI,
        Depends(get_llm_client),
    ],

    # 同一个VectorStore同时负责：
    #
    # 1. 提供全部Chunk建立关键词索引；
    # 2. 根据查询向量执行向量召回。
    vector_store: Annotated[
        ChromaVectorStore,
        Depends(get_vector_store),
    ],

    # Settings提供Embedding和LLM模型名称。
    settings: Annotated[
        Settings,
        Depends(get_settings),
    ],
) -> DiagnosisService:
    """装配完整的证据约束结构化诊断服务。"""

    # 在线API和离线Prompt实验共享同一个纯检索工厂。
    #
    # 这样查询改写、混合检索参数或门控策略发生变化时，
    # 两条入口会同时获得更新，不会形成两套实现。
    gated_retriever = (
        build_diagnosis_retrieval_provider(
            embedding_client=embedding_client,
            vector_store=vector_store,
            settings=settings,
        )
    )

    # 诊断草稿Provider使用生成式LLM，
    # 而不是Embedding模型。
    draft_provider = (
        OpenAIDiagnosisDraftProvider(
            client=llm_client,
            model=get_llm_model(
                settings
            ),
        )
    )

    # DiagnosisService只依赖两个高层能力：
    #
    # 1. 返回候选和门控决定；
    # 2. 根据已放行证据生成诊断草稿。
    #
    # 内部具体由多少层组件组成，
    # DiagnosisService不需要知道。
    return DiagnosisService(
        retrieval_provider=gated_retriever,
        draft_provider=draft_provider,
    )


@lru_cache(maxsize=1)
def get_robot_telemetry_store(
) -> InMemoryRobotTelemetryStore:
    """创建并缓存当前进程的教学模拟遥测Store。

    第一次调用时创建三条快照；
    后续调用返回同一个Store实例，
    不会把观测时间伪装成每次查询时的实时数据。
    """

    # datetime.now(timezone.utc)取得带UTC时区的当前时间。
    #
    # 该时间表示模拟快照集合的创建时间，
    # 不是从真实机器人采集到的设备时间。
    observed_at = datetime.now(
        timezone.utc
    )

    return build_demo_robot_telemetry_store(
        observed_at=observed_at
    )


def get_system_utc_clock(
) -> SystemUtcClock:
    """创建读取应用服务器UTC时间的系统时钟。

    SystemUtcClock本身不保存时间值；
    每次调用now_utc()都会重新读取当前系统时钟。
    """

    return SystemUtcClock()


def get_confirmed_evidence_store(
) -> ConfirmedEvidenceStore:
    """为一次Agent请求创建独立的空证据白名单。

    本依赖不能使用lru_cache。FastAPI会在同一次请求内复用
    这个对象，但下一次HTTP请求必须获得新的空Store，避免
    不同用户或不同Agent任务之间共享已确认引用。
    """

    return ConfirmedEvidenceStore()


def get_request_vision_input_store(
) -> RequestVisionInputStore:
    """为一次Agent请求创建独立的空图片输入Store。

    本依赖不能使用lru_cache。FastAPI会在同一次请求内
    复用这个对象，但下一次HTTP请求必须取得新的空Store，
    防止不同请求通过相同image_ref读取到彼此的图片。

    当前任务3只完成工具注册，因此Store初始为空。
    任务4的多模态API会在Agent运行前把已经通过
    VisionInputAdapter检查的图片登记到同一个Store中。
    """

    return RequestVisionInputStore()


def get_agent_tool_registry(
    # Agent搜索工具只依赖门控混合检索器，
    # 不再依赖完整的HybridKnowledgeQueryService。
    knowledge_retrieval_provider: Annotated[
        GatedHybridRetriever,
        Depends(
            get_agent_search_retrieval_provider
        ),
    ],

    # 模拟遥测Store由进程级缓存依赖提供。
    telemetry_store: Annotated[
        InMemoryRobotTelemetryStore,
        Depends(get_robot_telemetry_store),
    ],

    # search_knowledge把真实引用写入这个请求级Store；
    # draft_test_case只能使用这里已经存在的Chunk ID。
    confirmed_evidence_store: Annotated[
        ConfirmedEvidenceStore,
        Depends(get_confirmed_evidence_store),
    ],

    # 时间工具使用可被测试Fake替换的系统时钟。
    utc_clock: Annotated[
        SystemUtcClock,
        Depends(get_system_utc_clock),
    ],

    # Vision Provider负责真正的异步结构化图片观察。
    # Handler依赖VisionProvider协议，不绑定具体供应商SDK。
    vision_provider: Annotated[
        VisionProvider,
        Depends(get_vision_provider),
    ],

    # 当前请求图片Store只保存已经验证的VisionInput。
    # 任务4的请求处理层和当前视觉工具必须共享此实例。
    vision_input_store: Annotated[
        RequestVisionInputStore,
        Depends(get_request_vision_input_store),
    ],
) -> ToolRegistry:
    """注册当前Agent允许调用的只读工具。

    本函数只连接已经创建好的依赖和明确允许的工具。

    它不会根据LLM返回的任意字符串：
    1. 导入Python模块；
    2. 查找Python函数；
    3. 创建新工具；
    4. 绕过工具白名单。
    """

    registry = ToolRegistry()

    # search_knowledge直接使用门控混合检索器。
    #
    # 工具只返回supporting_chunk_ids对应的真实引用，
    # 不再在工具内部调用知识回答LLM。
    registry.register(
        definition=(
            SEARCH_KNOWLEDGE_TOOL_DEFINITION
        ),
        handler=SearchKnowledgeToolHandler(
            retrieval_provider=(
                knowledge_retrieval_provider
            ),
            evidence_store=(
                confirmed_evidence_store
            ),
        ),
    )

    # get_robot_telemetry只读取脱敏的
    # 进程内模拟遥测快照。
    registry.register(
        definition=(
            GET_ROBOT_TELEMETRY_TOOL_DEFINITION
        ),
        handler=GetRobotTelemetryToolHandler(
            store=telemetry_store
        ),
    )

    # draft_test_case与search_knowledge共享
    # 同一个当前请求证据Store。
    #
    # 因此Planner只能使用此前检索并确认过的
    # Chunk ID创建测试草案。
    registry.register(
        definition=(
            DRAFT_TEST_CASE_TOOL_DEFINITION
        ),
        handler=DraftTestCaseToolHandler(
            evidence_store=(
                confirmed_evidence_store
            ),
        ),
    )

    # get_current_time读取应用服务器UTC时间，
    # 不读取机器人时钟，也不修改外部状态。
    registry.register(
        definition=(
            GET_CURRENT_TIME_TOOL_DEFINITION
        ),
        handler=GetCurrentTimeToolHandler(
            clock=utc_clock
        ),
    )

    # analyze_robot_image只能读取当前请求Store中
    # 已经登记的图片，并通过Vision Provider生成观察。
    #
    # 当前Store没有对应image_ref时，Handler返回None，
    # ToolExecutor会将其转换成empty_result。
    registry.register(
        definition=(
            ANALYZE_ROBOT_IMAGE_TOOL_DEFINITION
        ),
        handler=AnalyzeRobotImageToolHandler(
            provider=vision_provider,
            input_store=vision_input_store,
        ),
    )

    return registry


def get_agent_tool_executor(
    registry: Annotated[
        ToolRegistry,
        Depends(get_agent_tool_registry),
    ],
) -> ToolExecutor:
    """使用当前请求的工具注册表创建执行器。"""

    return ToolExecutor(
        registry=registry
    )


def get_agent_planner(
    # get_llm_client是带yield的请求级资源依赖。
    #
    # FastAPI会先创建AsyncOpenAI客户端，
    # 再把同一个客户端注入当前Planner。
    llm_client: Annotated[
        AsyncOpenAI,
        Depends(get_llm_client),
    ],

    # Settings由get_settings读取并缓存。
    #
    # 当前Planner与已有知识问答、诊断服务
    # 复用同一套生成式LLM配置。
    settings: Annotated[
        Settings,
        Depends(get_settings),
    ],
) -> OpenAICompatibleAgentPlanner:
    """创建当前请求使用的真实Agent Planner。

    本函数只组装对象，不会立即请求LLM。
    真正的网络调用发生在：
    AgentRunner.run()
    → planner.plan()
    → chat.completions.create()
    """

    return OpenAICompatibleAgentPlanner(
        client=llm_client,

        # get_llm_model()统一检查：
        #
        # 1. LLM_MODEL不是None；
        # 2. 不是空字符串；
        # 3. 不只包含空白字符。
        model=get_llm_model(
            settings
        ),
    )


def get_agent_runner(
    # FastAPI调用get_agent_planner()，
    # 注入使用真实OpenAI兼容接口的Planner。
    planner: Annotated[
        OpenAICompatibleAgentPlanner,
        Depends(get_agent_planner),
    ],

    # FastAPI调用get_agent_tool_executor()，
    # 注入只允许执行当前注册表工具的Executor。
    executor: Annotated[
        ToolExecutor,
        Depends(get_agent_tool_executor),
    ],
) -> AgentRunner:
    """组装当前请求使用的受控Agent循环。

    Runner只依赖两个高层能力：

    1. Planner：返回下一步结构化决定；
    2. Executor：校验并执行白名单工具。

    Runner不知道知识库、遥测Store、OpenAI客户端
    和各工具Handler是怎样创建的。
    """

    return AgentRunner(
        planner=planner,
        executor=executor,

        # 当前继续使用AgentRunner中经过测试的安全默认值：
        #
        # max_steps=5
        # planner_timeout_seconds=30.0
        # max_consecutive_tool_failures=2
        #
        # 这些值只由服务端代码控制，
        # 未来的API请求不能自行放宽安全边界。
    )


def get_agent_diagnosis_service(
    # Runner执行Planner与工具之间的受控循环。
    #
    # FastAPI会先调用get_agent_runner()，
    # 再把结果注入当前参数。
    runner: Annotated[
        AgentRunner,
        Depends(get_agent_runner),
    ],

    # Service需要记录本次实际使用的Planner Prompt版本。
    #
    # 这里显式注入Planner，而不是读取
    # runner._planner这样的私有属性。
    #
    # FastAPI会在同一次请求中复用Runner已经使用的
    # 同一个get_agent_planner()依赖结果。
    planner: Annotated[
        OpenAICompatibleAgentPlanner,
        Depends(get_agent_planner),
    ],

    # 这是当前HTTP请求独有的证据白名单。
    #
    # get_agent_tool_registry()中的search_knowledge
    # 会把真实引用写入这个Store；
    # AgentDiagnosisService构造最终报告时，
    # 会使用同一个Store验证Chunk ID。
    confirmed_evidence_store: Annotated[
        ConfirmedEvidenceStore,
        Depends(get_confirmed_evidence_store),
    ],

    # Service在Runner启动前使用Adapter验证外部图片。
    # Adapter只做本地解码、格式和资源上限检查，
    # 不调用Vision模型。
    vision_input_adapter: Annotated[
        VisionInputAdapter,
        Depends(get_vision_input_adapter),
    ],

    # 这个Store与get_agent_tool_registry()注入给
    # analyze_robot_image Handler的Store是同一实例。
    # FastAPI会在单次请求中缓存同一个依赖结果。
    vision_input_store: Annotated[
        RequestVisionInputStore,
        Depends(get_request_vision_input_store),
    ],
) -> AgentDiagnosisService:
    """装配当前请求使用的Robot Diagnostic Agent Service。

    本函数只连接依赖，不执行Agent循环。

    真正的Planner、Embedding、知识检索和工具调用，
    只有Router之后调用await service.diagnose()时才会发生。
    """

    return AgentDiagnosisService(
        runner=runner,
        evidence_store=(
            confirmed_evidence_store
        ),
        vision_input_adapter=(
            vision_input_adapter
        ),
        vision_input_store=(
            vision_input_store
        ),

        # prompt_version是Planner提供的只读@property。
        #
        # Service把它写入DiagnosisReport和
        # AgentExecutionSummary，便于追踪一次响应
        # 究竟使用了哪个Prompt版本。
        planner_prompt_version=(
            planner.prompt_version
        ),
    )
