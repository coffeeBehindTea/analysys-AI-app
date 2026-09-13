"""知识库RAG编排服务的离线测试。"""

import pytest

# 测试非法LLM证据编号时需要检查这个异常。
# 当前这个正常回答测试暂时还不会使用它，
# 但后续测试会使用，所以现在保留。
from app.errors import InvalidLLMResponseError

# KnowledgeAnswerDraft是LLM回答服务返回给
# KnowledgeQueryService的内部结构化数据。
#
# KnowledgeQueryRequest是进入业务服务的查询请求。
from app.schemas.knowledge_query import (
    KnowledgeAnswerDraft,
    KnowledgeQueryRequest,
)

# 这些模型用于构造假的查询向量和检索结果。
from app.schemas.retrieval import (
    DocumentChunk,
    EmbeddingVector,
    RetrievedChunk,
)

# KnowledgeQueryService是本文件真正测试的模块。
from app.services.knowledge_query import (
    INSUFFICIENT_EVIDENCE_ANSWER,
    KnowledgeQueryService,
)


class FakeQueryEmbeddingProvider:
    """返回固定查询向量，不访问Embedding API。"""

    def __init__(
        self,
        vector: EmbeddingVector,
    ) -> None:
        self._vector = list(vector)

        # 记录所有收到的问题。
        self.queries: list[str] = []

    async def embed_query(
        self,
        query: str,
    ) -> EmbeddingVector:
        """记录问题并返回测试向量。"""

        self.queries.append(query)
        return list(self._vector)


class FakeRetrievalStore:
    """返回预设检索结果，不访问Chroma。"""

    def __init__(
        self,
        results: list[RetrievedChunk],
    ) -> None:
        self._results = list(results)

        # 每次调用记录：
        # 查询向量、模型名称和top_k。
        self.calls: list[
            tuple[EmbeddingVector, str, int]
        ] = []

    def retrieve(
        self,
        query_embedding: EmbeddingVector,
        *,
        query_embedding_model: str,
        top_k: int = 3,
    ) -> list[RetrievedChunk]:
        """记录查询参数并返回结果副本。"""

        self.calls.append(
            (
                list(query_embedding),
                query_embedding_model,
                top_k,
            )
        )

        return list(self._results)


class FakeAnswerProvider:
    """返回固定Draft，并记录是否被调用。"""

    def __init__(
        self,
        draft: KnowledgeAnswerDraft,
    ) -> None:
        self._draft = draft

        self.calls: list[
            tuple[str, list[RetrievedChunk]]
        ] = []

    async def generate_answer(
        self,
        *,
        question: str,
        evidence: list[RetrievedChunk],
    ) -> KnowledgeAnswerDraft:
        """记录问题和证据并返回固定Draft。"""

        self.calls.append(
            (
                question,
                list(evidence),
            )
        )

        return self._draft


def make_retrieved_chunk(
    *,
    similarity: float,
    rank: int = 1,
    page_or_section: str = "page: 12",
    content: str = (
        "急停装置复位后，应检查"
        "安全控制系统状态。"
    ),
) -> RetrievedChunk:
    """创建一条可追溯的假检索结果。"""

    document_id = "b" * 64
    chunk_index = rank - 1

    return RetrievedChunk(
        chunk=DocumentChunk(
            chunk_id=(
                f"{document_id}:"
                f"{chunk_index:06d}"
            ),
            document_id=document_id,
            source_file="safety-manual.pdf",
            chunk_index=chunk_index,
            content_hash="c" * 64,
            page_or_section=page_or_section,
            content=content,
        ),
        similarity=similarity,
        rank=rank,
    )


@pytest.mark.asyncio
async def test_answer_query_uses_retrieved_evidence(
) -> None:
    """正常回答时只引用LLM声明实际使用的证据。"""

    # E1是向量检索排名第一的结果。
    #
    # 它的相似度最高，但正文只有文档标题，
    # 不能真正支持问题的答案。
    irrelevant_result = make_retrieved_chunk(
        similarity=0.82,
        rank=1,
        page_or_section="page: 1",
        content="工业移动机器人安全标准",
    )

    # E2虽然排名第二，但包含能够支持回答的正文。
    supporting_result = make_retrieved_chunk(
        similarity=0.76,
        rank=2,
        page_or_section="page: 5",
        content=(
            "本标准适用于工业移动机器人"
            "设计、制造、安装、运行、维护"
            "和报废等生命周期阶段。"
        ),
    )

    # FakeQueryEmbeddingProvider代替真实Embedding服务。
    #
    # 无论收到什么问题，它都返回固定向量，
    # 因而测试不会访问外部Embedding API。
    embedding_provider = (
        FakeQueryEmbeddingProvider(
            vector=[1.0, 0.0],
        )
    )

    # FakeRetrievalStore代替真实Chroma。
    #
    # retrieve()被调用时会直接返回上面准备的
    # E1和E2，不访问本地向量数据库。
    retrieval_store = FakeRetrievalStore(
        results=[
            irrelevant_result,
            supporting_result,
        ],
    )

    # FakeAnswerProvider代替真实LLM。
    #
    # 它返回结构化的KnowledgeAnswerDraft：
    #
    # answer是回答正文；
    # used_evidence_ids=["E2"]表示回答只使用了E2。
    answer_provider = FakeAnswerProvider(
        draft=KnowledgeAnswerDraft(
            answer=(
                "该标准适用于工业移动机器人"
                "设计、制造、安装、运行、维护"
                "和报废等生命周期阶段。"
            ),
            used_evidence_ids=["E2"],
        ),
    )

    # 创建被测试的RAG业务编排服务。
    #
    # 这里使用依赖注入，将三个Fake对象交给Service，
    # 所以测试过程完全离线。
    service = KnowledgeQueryService(
        embedding_provider=embedding_provider,
        retrieval_store=retrieval_store,
        answer_provider=answer_provider,
        embedding_model="fake-embedding",
        similarity_threshold=0.70,
    )

    # 构造进入Service的业务请求。
    #
    # top_k=2表示最多召回两个Chunk。
    request = KnowledgeQueryRequest(
        question="该标准适用于哪些生命周期阶段？",
        top_k=2,
    )

    # 调用本测试真正要验证的方法。
    response = await service.answer_query(
        request
    )

    # 证据通过阈值并且LLM成功生成回答，
    # 因此不应该进入拒答状态。
    assert response.abstained is False

    # 最终回答正文应来自KnowledgeAnswerDraft.answer。
    assert response.answer == (
        "该标准适用于工业移动机器人"
        "设计、制造、安装、运行、维护"
        "和报废等生命周期阶段。"
    )

    # retrieval_ms记录查询向量化和检索耗时，
    # 所以结果必须是非负数。
    assert response.retrieval_ms >= 0.0

    # 验证Service确实将原始问题送入了
    # QueryEmbeddingProvider.embed_query()。
    assert embedding_provider.queries == [
        "该标准适用于哪些生命周期阶段？"
    ]

    # 验证Service调用RetrievalStore.retrieve()时，
    # 正确传入了：
    #
    # 1. 查询向量；
    # 2. Embedding模型名称；
    # 3. 请求中的top_k。
    assert retrieval_store.calls == [
        (
            [1.0, 0.0],
            "fake-embedding",
            2,
        )
    ]

    # Top-1相似度0.82高于阈值0.70，
    # 所以Service应该调用一次回答Provider。
    assert len(answer_provider.calls) == 1

    # FakeAnswerProvider记录的第一个参数是问题。
    assert answer_provider.calls[0][0] == (
        "该标准适用于哪些生命周期阶段？"
    )

    # FakeAnswerProvider收到的第二个参数是
    # 完整的候选证据列表，不是只传E2。
    #
    # LLM需要看到E1和E2，才能判断使用哪一条。
    assert answer_provider.calls[0][1] == [
        irrelevant_result,
        supporting_result,
    ]

    # LLM声明只使用了E2，
    # 因此最终响应中只能出现一条引用。
    assert len(response.citations) == 1

    citation = response.citations[0]

    # 引用必须对应E2，也就是检索排名第二的Chunk。
    #
    # rank仍然保留为2，不会因为只返回一条引用
    # 就被重新编号成1。
    assert citation.rank == 2

    # chunk_id必须来自真实的supporting_result，
    # 不能由LLM生成。
    assert (
        citation.chunk_id
        == supporting_result.chunk.chunk_id
    )

    # 页码必须来自E2，而不是E1。
    assert citation.page_or_section == "page: 5"

    # 引用正文必须是E2的原始Chunk正文。
    assert (
        citation.excerpt
        == supporting_result.chunk.content
    )

    # 最终引用不能错误地指向E1。
    assert (
        citation.chunk_id
        != irrelevant_result.chunk.chunk_id
    )


@pytest.mark.asyncio
async def test_low_similarity_abstains_without_llm(
) -> None:
    """Top-1低于阈值时必须拒答且不调用LLM。"""

    # 构造一条相似度为0.69的检索结果。
    #
    # Service配置的阈值是0.70，
    # 因此这条证据不足以支持系统回答。
    result = make_retrieved_chunk(
        similarity=0.69,
        rank=1,
        page_or_section="page: 12",
        content="这段内容与用户问题没有足够关系。",
    )

    # 使用固定向量代替真实Embedding API。
    embedding_provider = (
        FakeQueryEmbeddingProvider(
            vector=[1.0, 0.0],
        )
    )

    # 使用固定检索结果代替真实Chroma查询。
    retrieval_store = FakeRetrievalStore(
        results=[result],
    )

    # FakeAnswerProvider的构造函数现在要求接收
    # KnowledgeAnswerDraft，而不再接收answer字符串。
    #
    # 这里虽然必须给它一个合法Draft，
    # 但这个Draft不应该被业务代码读取或返回。
    # 后面的calls == []会验证Provider从未被调用。
    answer_provider = FakeAnswerProvider(
        draft=KnowledgeAnswerDraft(
            answer="这段回答不应该被使用。",
            used_evidence_ids=["E1"],
        ),
    )

    # 创建被测试的RAG编排服务。
    #
    # similarity_threshold=0.70表示：
    # 只有Top-1相似度大于或等于0.70，
    # 才允许继续调用LLM。
    service = KnowledgeQueryService(
        embedding_provider=embedding_provider,
        retrieval_store=retrieval_store,
        answer_provider=answer_provider,
        embedding_model="fake-embedding",
        similarity_threshold=0.70,
    )

    # 没有填写top_k，所以会使用
    # KnowledgeQueryRequest定义的默认值3。
    request = KnowledgeQueryRequest(
        question="知识库之外的问题",
    )

    # 调用被测试方法。
    response = await service.answer_query(
        request
    )

    # 相似度0.69低于阈值0.70，
    # 所以响应必须标记为拒答。
    assert response.abstained is True

    # 拒答内容必须是代码中定义的固定文本，
    # 而不是由LLM自由生成的内容。
    assert response.answer == (
        INSUFFICIENT_EVIDENCE_ANSWER
    )

    # 拒答不能携带引用。
    #
    # 因为系统已经判断现有证据不足，
    # 返回这些证据可能使用户误认为它们支持答案。
    assert response.citations == []

    # 即使最后拒答，问题仍然应该先被向量化。
    #
    # 不进行向量化和检索，就无法知道
    # 当前问题是否拥有足够证据。
    assert embedding_provider.queries == [
        "知识库之外的问题"
    ]

    # 验证检索确实执行了一次，并收到：
    #
    # 查询向量：[1.0, 0.0]
    # 模型名称：fake-embedding
    # top_k默认值：3
    assert retrieval_store.calls == [
        (
            [1.0, 0.0],
            "fake-embedding",
            3,
        )
    ]

    # 这是该测试最重要的安全断言。
    #
    # calls为空表示：
    # FakeAnswerProvider.generate_answer()
    # 从未被调用。
    #
    # 也就是说，低相似度分支在进入LLM之前
    # 就已经结束了请求。
    assert answer_provider.calls == []

    # 即使拒答，查询向量化和向量检索仍然发生了，
    # 所以应当存在一个非负的检索耗时。
    assert response.retrieval_ms >= 0.0


@pytest.mark.asyncio
async def test_empty_store_abstains_without_llm(
) -> None:
    """空知识库应拒答且不调用LLM。"""

    # 使用固定查询向量，避免访问真实Embedding API。
    embedding_provider = (
        FakeQueryEmbeddingProvider(
            vector=[1.0, 0.0],
        )
    )

    # results=[]模拟向量数据库没有返回任何Chunk。
    #
    # 可能的真实原因包括：
    # 1. Collection中还没有摄取文档；
    # 2. 数据被删除；
    # 3. 检索过滤条件没有匹配项。
    retrieval_store = FakeRetrievalStore(
        results=[],
    )

    # 构造一个合法但不应该被使用的Draft。
    #
    # 即使当前不存在E1也没有关系，
    # 因为空结果分支必须在调用AnswerProvider之前返回。
    answer_provider = FakeAnswerProvider(
        draft=KnowledgeAnswerDraft(
            answer="这段回答不应该被使用。",
            used_evidence_ids=["E1"],
        ),
    )

    service = KnowledgeQueryService(
        embedding_provider=embedding_provider,
        retrieval_store=retrieval_store,
        answer_provider=answer_provider,
        embedding_model="fake-embedding",
        similarity_threshold=0.70,
    )

    request = KnowledgeQueryRequest(
        question="空知识库问题",
    )

    response = await service.answer_query(
        request
    )

    # 没有任何检索结果时必须拒答。
    assert response.abstained is True

    # 回答必须使用统一的固定拒答文本。
    assert response.answer == (
        INSUFFICIENT_EVIDENCE_ANSWER
    )

    # 没有检索结果，自然也不能生成引用。
    assert response.citations == []

    # 问题仍然需要先进行向量化。
    assert embedding_provider.queries == [
        "空知识库问题"
    ]

    # 检索也必须实际发生一次。
    #
    # 返回空列表是检索结果，
    # 不代表检索过程没有运行。
    assert retrieval_store.calls == [
        (
            [1.0, 0.0],
            "fake-embedding",
            3,
        )
    ]

    # 没有证据时绝对不能调用LLM。
    assert answer_provider.calls == []

    # 检索阶段已经运行，所以仍然会产生耗时。
    assert response.retrieval_ms >= 0.0


def test_query_service_rejects_invalid_threshold(
) -> None:
    """相似度阈值超出0到1时应在构造阶段失败。"""

    # pytest.raises()声明：
    #
    # with代码块内必须抛出ValueError。
    #
    # match会使用正则表达式搜索异常消息，
    # 确认异常确实与similarity_threshold有关，
    # 而不是其他地方偶然抛出的ValueError。
    with pytest.raises(
        ValueError,
        match="similarity_threshold",
    ):
        # Python会先创建下面三个依赖对象，
        # 然后才调用KnowledgeQueryService.__init__()。
        KnowledgeQueryService(
            embedding_provider=(
                FakeQueryEmbeddingProvider(
                    vector=[1.0, 0.0],
                )
            ),
            retrieval_store=(
                FakeRetrievalStore(
                    results=[],
                )
            ),

            # FakeAnswerProvider现在需要接收
            # KnowledgeAnswerDraft，而不再接收answer字符串。
            #
            # 这个Draft只用于满足构造参数的数据契约，
            # 不会真正被调用。
            answer_provider=(
                FakeAnswerProvider(
                    draft=KnowledgeAnswerDraft(
                        answer="不会被使用的回答",
                        used_evidence_ids=["E1"],
                    ),
                )
            ),

            embedding_model="fake-embedding",

            # 1.1超出了Service允许的[0, 1]范围，
            # 因此构造函数必须立即抛出ValueError。
            similarity_threshold=1.1,
        )


@pytest.mark.asyncio
async def test_answer_query_rejects_unknown_evidence_id(
) -> None:
    """LLM选择本次检索中不存在的证据时必须报错。"""

    # 本次检索只返回一条结果。
    #
    # 因为rank=1，所以这条证据的临时编号是E1。
    result = make_retrieved_chunk(
        similarity=0.82,
        rank=1,
        page_or_section="page: 12",
        content=(
            "急停装置复位后，"
            "应检查安全控制系统状态。"
        ),
    )

    embedding_provider = (
        FakeQueryEmbeddingProvider(
            vector=[1.0, 0.0],
        )
    )

    # 检索结果中只有E1，没有E2。
    retrieval_store = FakeRetrievalStore(
        results=[result],
    )

    # 模拟LLM返回结构正确、但业务内容非法的Draft。
    #
    # "E2"符合EvidenceId的格式要求，
    # 但本次检索结果里并不存在rank=2的Chunk。
    answer_provider = FakeAnswerProvider(
        draft=KnowledgeAnswerDraft(
            answer="这是一个引用了错误证据的回答。",
            used_evidence_ids=["E2"],
        ),
    )

    service = KnowledgeQueryService(
        embedding_provider=embedding_provider,
        retrieval_store=retrieval_store,
        answer_provider=answer_provider,
        embedding_model="fake-embedding",
        similarity_threshold=0.70,
    )

    request = KnowledgeQueryRequest(
        question="急停复位后应检查什么？",
        top_k=1,
    )

    # 本测试预期的结果不是正常响应，
    # 而是Service抛出InvalidLLMResponseError。
    #
    # match验证异常消息确实指出：
    # LLM使用了未提供的证据编号。
    with pytest.raises(
        InvalidLLMResponseError,
        match="未提供的证据编号",
    ) as exc_info:
        await service.answer_query(
            request
        )

    # exc_info.value是实际捕获到的异常对象。
    #
    # str()会取得异常消息，
    # 这里进一步确认消息中明确包含非法编号E2。
    assert "E2" in str(exc_info.value)

    # 在发现非法编号之前，
    # 问题向量化应该已经正常执行。
    assert embedding_provider.queries == [
        "急停复位后应检查什么？"
    ]

    # 检索也应该正常执行，并使用请求中的top_k=1。
    assert retrieval_store.calls == [
        (
            [1.0, 0.0],
            "fake-embedding",
            1,
        )
    ]

    # Top-1相似度0.82通过阈值，
    # 所以AnswerProvider确实应该被调用一次。
    assert len(answer_provider.calls) == 1

    # AnswerProvider收到的候选证据只有E1对应的result。
    assert answer_provider.calls[0][1] == [
        result
    ]