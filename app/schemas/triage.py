"""机器人故障分诊的请求、API 响应和 LLM 响应契约。"""

# BaseModel 提供运行时数据校验；ConfigDict 配置模型；
# Field 为字段增加长度限制、说明和 Swagger 示例。
from pydantic import BaseModel, ConfigDict, Field


class TriageRequest(BaseModel):
    """客户端提交给 POST /api/v1/triage 的 JSON 请求体。"""

    # 自动去除所有字符串首尾空格，并拒绝未声明的额外字段。
    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    # Field() 的 min_length/max_length 会在业务函数执行前校验输入。
    robot_id: str = Field(
        min_length=1,
        max_length=100,
        description="发生故障的机器人标识",
        examples=["robot-001"],
    )
    symptom: str = Field(
        min_length=1,
        max_length=1000,
        description="观察到的故障现象",
        examples=["机器人移动时持续向左偏移"],
    )
    log_excerpt: str = Field(
        min_length=1,
        max_length=10_000,
        description="经过脱敏处理的日志片段",
        examples=["WARN motor_controller: left wheel speed mismatch"],
    )


class TriageResponse(BaseModel):
    """API 成功时向客户端承诺返回的 JSON 结构。"""

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(
        min_length=1,
        description="本次请求的唯一追踪标识",
        examples=["f90fd399-1464-4944-a75b-d76b075ecc45"],
    )
    summary: str = Field(
        min_length=1,
        description="根据现有信息生成的谨慎故障摘要",
    )
    recommended_actions: list[str] = Field(
        min_length=1,
        description="建议执行的排查操作",
    )


class LLMAnalysis(BaseModel):
    """LLM 返回 JSON 的内部校验结构，不直接代表 HTTP 响应。"""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    summary: str = Field(
        min_length=1,
        max_length=2000,
        description="根据已有证据生成的谨慎故障摘要",
    )
    # 对 list 使用 min_length/max_length 会限制列表项目数量。
    recommended_actions: list[str] = Field(
        min_length=1,
        max_length=8,
        description="建议执行的排查操作",
    )
