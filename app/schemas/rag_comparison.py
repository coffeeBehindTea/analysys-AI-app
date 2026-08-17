"""RAG与裸LLM可复现实验使用的数据契约。"""

# datetime记录实验报告的生成时间。
from datetime import datetime

# Self表示当前Pydantic模型实例自身的类型，
# 用于model_validator的返回类型标注。
from typing import Self

# BaseModel提供数据校验与序列化；
# ConfigDict配置整个模型的行为；
# Field配置单个字段的长度、数值范围和说明；
# model_validator校验多个字段之间的关系。
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

# GoldQuestion保存问题、参考答案、是否可回答
# 以及人工标注的预期证据。
from app.schemas.evaluation import (
    GoldQuestion,
)

# KnowledgeQueryResponse是正式RAG查询API
# 返回的answer、citations、retrieval_ms和abstained。
from app.schemas.knowledge_query import (
    KnowledgeQueryResponse,
)


class BareLLMResult(BaseModel):
    """裸LLM在没有检索证据时的回答结果。"""

    model_config = ConfigDict(
        # 清理字符串首尾空白。
        str_strip_whitespace=True,

        # 拒绝Schema未声明的额外字段，
        # 避免报告字段拼写错误被静默忽略。
        extra="forbid",

        # 禁止NaN和正负无穷等非法浮点数。
        allow_inf_nan=False,
    )

    # 裸LLM只返回文本，没有可验证的知识库引用。
    answer: str = Field(
        min_length=1,
        max_length=20_000,
        description=(
            "没有提供检索证据时的LLM回答"
        ),
    )

    # 从发出裸LLM请求到取得完整回答的墙钟耗时。
    generation_ms: float = Field(
        ge=0.0,
        description=(
            "裸LLM生成回答的总耗时，单位毫秒"
        ),
    )


class RagVsBareLLMCase(BaseModel):
    """同一道Gold Question上的RAG与裸LLM结果。"""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        allow_inf_nan=False,
    )

    # 保存完整Gold Question，
    # 便于在报告中同时查看参考答案和预期证据。
    gold_question: GoldQuestion

    # 正式知识库API返回的RAG响应。
    #
    # 该对象自身会继续检查：
    # - 拒答时引用必须为空；
    # - 非拒答时至少有一条引用。
    rag_response: KnowledgeQueryResponse

    # 从实验脚本发出HTTP请求，
    # 到收到完整知识库API响应的总耗时。
    #
    # 它包含网络、序列化、检索和LLM回答时间，
    # 不等同于rag_response.retrieval_ms。
    rag_total_ms: float = Field(
        ge=0.0,
        description=(
            "RAG知识库API的端到端总耗时，单位毫秒"
        ),
    )

    # 同一道问题在没有知识库证据时的裸LLM结果。
    bare_result: BareLLMResult


class RagVsBareLLMReport(BaseModel):
    """一次RAG与裸LLM对照实验的完整报告。"""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        allow_inf_nan=False,
    )

    generated_at: datetime

    # 记录实际调用的RAG API地址，
    # 使实验请求可以由其他开发者复现。
    rag_api_url: str = Field(
        min_length=1,
        max_length=2_000,
    )

    # 两组实验使用同一个生成模型。
    llm_model: str = Field(
        min_length=1,
        max_length=500,
    )

    # Embedding和Collection只属于RAG实验组，
    # 裸LLM不会使用它们。
    embedding_model: str = Field(
        min_length=1,
        max_length=500,
    )

    collection_name: str = Field(
        min_length=1,
        max_length=512,
    )

    top_k: int = Field(
        ge=1,
        le=10,
    )

    similarity_threshold: float = Field(
        ge=0.0,
        le=1.0,
    )

    # 把裸LLM实际使用的System Prompt写入报告。
    #
    # 这样读者可以明确知道裸LLM得到了什么规则，
    # 而不是只看到一个无法复现的答案。
    bare_system_prompt: str = Field(
        min_length=1,
        max_length=10_000,
    )

    # 一次实验至少比较一道问题。
    cases: list[RagVsBareLLMCase] = Field(
        min_length=1,
    )

    @model_validator(mode="after")
    def question_ids_must_be_unique(
        self,
    ) -> Self:
        """同一道Gold Question不能重复计入报告。"""

        question_ids = [
            case.gold_question.question_id
            for case in self.cases
        ]

        if len(set(question_ids)) != len(
            question_ids
        ):
            raise ValueError(
                "RAG与裸LLM对比报告包含重复question_id"
            )

        # mode="after"校验器必须返回校验后的模型实例。
        return self