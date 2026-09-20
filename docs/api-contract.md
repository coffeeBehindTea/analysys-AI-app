# RobotOps Copilot MVP API 契约

## 1. 文档目的

本文档冻结 RobotOps Copilot MVP 的外部 API 和公开数据契约，作为以下模块共同遵守的边界：

- Web 控制台；
- FastAPI Router；
- Agent 诊断服务；
- Agent SSE 事件发布器；
- 诊断会话存储；
- Web 控制台中的 Markdown 和 JSON 确定性导出；
- 端到端测试与验收脚本。

本文档只规定可以公开交换的数据，不描述 Planner 私有推理过程，也不允许客户端通过请求扩大 Agent 的工具权限。

## 2. 契约状态

本文档中的能力分为两类：

| 标记 | 含义 |
|---|---|
| 已实现 | 当前 Week 7 代码中已经存在，可以直接调用或在控制台使用 |
| 暂不支持 | 不属于当前 MVP 的已实现能力 |

当前状态如下：

| 接口 | 状态 |
|---|---|
| `POST /api/v1/agent/diagnose` | 已实现 |
| `GET /api/v1/diagnostic-sessions` | 已实现 |
| `GET /api/v1/diagnostic-sessions/{session_id}` | 已实现 |
| `POST /api/v1/agent/diagnose/stream` | 已实现 |
| Web 控制台 JSON/Markdown 导出 | 已实现 |

## 3. 全局规则

### 3.1 内容类型

普通 JSON 接口使用：

`application/json; charset=utf-8`

SSE 接口使用：

`text/event-stream; charset=utf-8`

Markdown 导出使用：

`text/markdown; charset=utf-8`

### 3.2 请求追踪

每次 HTTP 请求都必须具有 `request_id`。

服务端通过 `X-Request-ID` 响应头公开该标识。最终 JSON 响应、SSE 事件和诊断会话中的 `request_id` 必须与响应头一致。

`request_id` 用于关联：

- HTTP 响应；
- 服务端脱敏日志；
- SSE 事件；
- 诊断报告；
- 诊断会话。

### 3.3 会话标识

成功创建诊断会话后，服务端返回 UUID4 格式的 `session_id`。

`session_id` 用于后续读取和导出诊断记录。客户端不能自行指定 `session_id`。

### 3.4 不可信输入

以下内容都必须被视为不可信输入：

- `robot_id`；
- `symptom`；
- `log_excerpt`；
- `task_goal`；
- 图片中的文字；
- Planner 生成的工具参数；
- 知识库文档正文。

这些内容不能修改系统 Prompt、工具白名单、安全策略、最大步骤数、超时配置或证据门槛。

### 3.5 禁止公开的内容

任何普通响应、SSE 事件、会话记录和导出文件都不得包含：

- API Key；
- Authorization 请求头；
- 完整图片 Base64；
- 本地图片路径；
- 未脱敏的完整日志；
- Planner 私有思维链；
- 工具的完整内部参数；
- 工具的完整内部返回对象；
- 可直接执行的机器人控制命令。

## 4. Agent 诊断请求

### 4.1 接口

`POST /api/v1/agent/diagnose`

状态：已实现。

该接口执行一次完整诊断，并在全部处理完成后返回一个 JSON 响应。

### 4.2 请求模型

Pydantic 模型：`AgentDiagnosisRequest`

实现位置：`app/schemas/agent_api.py`

| 字段 | 类型 | 必填 | 约束 | 含义 |
|---|---|---:|---|---|
| `robot_id` | `str` | 是 | 1～100 个字符 | 需要诊断的机器人编号 |
| `symptom` | `str` | 是 | 1～2000 个字符 | 用户直接观察到的故障现象 |
| `log_excerpt` | `str` | 是 | 1～10000 个字符 | 已经脱敏的日志摘要 |
| `task_goal` | `str \| null` | 否 | 非空时最多 2000 个字符 | 本次诊断希望完成的目标 |
| `images` | `list[VisionImagePayload]` | 否 | 最多 3 张 | 本次请求附带的图片 |

请求模型必须使用 `extra="forbid"`，拒绝客户端提交未声明字段，例如：

- `tools`；
- `max_steps`；
- `run_shell`；
- `allow_control`；
- `disable_safety`；
- `retrieval_threshold`。

客户端无权直接指定 Agent 可以使用哪些工具。

### 4.3 图片模型

每个 `VisionImagePayload` 包含：

| 字段 | 类型 | 约束 |
|---|---|---|
| `mime_type` | `str` | 只能是 `image/jpeg`、`image/png` 或 `image/webp` |
| `encoding` | `str` | 当前固定为 `base64` |
| `image_base64` | `str` | 图片的 Base64 载荷 |
| `analysis_goal` | `str` | 1～2000 个字符 |
| `detail` | `str` | `auto`、`low` 或 `high` |

Pydantic 请求模型只校验字段结构。图片能否解码、真实格式是否匹配、尺寸和像素数量是否合法，仍由 `VisionInputAdapter` 继续校验。

### 4.4 请求示例

```json
{
  "robot_id": "robot-001",
  "symptom": "网络恢复后机器人仍处于暂停状态",
  "log_excerpt": "ERR-NET-4001 heartbeat timeout exceeded 1500 ms",
  "task_goal": "核对故障原因、当前状态和安全恢复条件",
  "images": []
}
```

## 5. Agent 诊断响应

### 5.1 响应模型

Pydantic 模型：`AgentDiagnosisResponse`

实现位置：`app/schemas/agent_api.py`

| 字段 | 类型 | 含义 |
|---|---|---|
| `request_id` | `str` | 当前 HTTP 请求的追踪标识 |
| `session_id` | `UUID4 \| null` | 已持久化诊断会话的标识 |
| `diagnosis` | `DiagnosisReport` | 最终结构化诊断报告 |
| `execution` | `AgentExecutionSummary` | Agent 终止状态和脱敏工具轨迹 |
| `vision_observations` | `list[AgentVisionObservation]` | 成功取得的图片观察 |
| `telemetry_observations` | `list[RobotTelemetryToolOutput]` | 成功读取的模拟遥测 |
| `test_case_drafts` | `list[DraftTestCaseToolOutput]` | 等待人工审核的测试草案 |

响应中的 `request_id` 必须与 `diagnosis.request_id` 一致。

真实 HTTP 服务启用会话存储时，`session_id` 必须是有效 UUID4。离线单元测试可以使用 `null`。

### 5.2 响应示例

```json
{
  "request_id": "request-demo-001",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "diagnosis": {
    "request_id": "request-demo-001",
    "prompt_version": "agent-tool-calling-v2",
    "gate_version": "agent-confirmed-evidence-v1",
    "status": "completed",
    "symptoms": [
      {
        "description": "网络恢复后机器人仍处于暂停状态",
        "source": "user_report"
      },
      {
        "description": "ERR-NET-4001 heartbeat timeout exceeded 1500 ms",
        "source": "log_excerpt"
      }
    ],
    "evidence": [
      {
        "chunk_id": "example-document-id:000022",
        "document_id": "0000000000000000000000000000000000000000000000000000000000000000",
        "source_file": "公开测试规程.md",
        "page_or_section": "section: 网络与调度测试",
        "rank": 1,
        "rrf_score": 0.0327,
        "vector_similarity": 0.58,
        "excerpt": "通信恢复后不自动继续旧任务。"
      }
    ],
    "possible_causes": [],
    "next_checks": [
      {
        "description": "由调度系统核对旧任务状态和机器人当前状态",
        "evidence_chunk_ids": [
          "example-document-id:000022"
        ],
        "risk_level": "low",
        "requires_qualified_person": false
      }
    ],
    "risk_level": "low",
    "missing_information": [],
    "abstained": false
  },
  "execution": {
    "planner_prompt_version": "agent-tool-calling-v2",
    "state": "completed",
    "termination_reason": "planner_finished",
    "termination_message": "Agent规划正常结束",
    "finish_reason": "task_completed",
    "step_count": 1,
    "steps": [
      {
        "step_id": 1,
        "event_type": "tool_execution",
        "tool_name": "search_knowledge",
        "input_summary": "知识库检索；query_chars=32；top_k=5",
        "result_summary": "知识库返回1条已确认引用",
        "status": "success",
        "duration_ms": 210.5,
        "error_code": null
      }
    ],
    "progress": null,
    "missing_information": []
  },
  "vision_observations": [],
  "telemetry_observations": [],
  "test_case_drafts": []
}
```

示例中的文档 ID 和内容仅用于说明 JSON 结构，不是正式验收数据。

## 6. 四类公开诊断状态

第七周必须把 `DiagnosisStatus` 扩展为：

- `completed`；
- `partial`；
- `abstained`；
- `human_review_required`。

### 6.1 `completed`

任务所需能力已经完成，必要结论具有可信来源，最终报告通过结构校验和引用白名单校验。

约束：

- `abstained` 必须为 `false`；
- 必须具有支持最终结论的证据；
- `missing_information` 必须为空；
- 不表示现场操作已经执行。

### 6.2 `partial`

系统取得了部分可信结果，但仍缺少完成全部任务所需的信息。

约束：

- `abstained` 必须为 `false`；
- 必须至少具有一条可信证据；
- 只能返回已有证据支持的部分结论；
- `missing_information` 不能为空。

### 6.3 `abstained`

系统没有足够证据形成安全结论，或者有效工具没有返回可用结果。

约束：

- `abstained` 必须为 `true`；
- `possible_causes` 和 `next_checks` 必须为空；
- `missing_information` 不能为空；
- 可以保留用于解释证据冲突或证据不足的已确认来源。

### 6.4 `human_review_required`

请求涉及真实设备控制、安全保护绕过、高风险现场操作、证据冲突，或者必须由有权限和资质的人员确认。

约束：

- `abstained` 必须为 `false`；
- 可以保留已经完成的只读 Vision、知识库或模拟遥测观察；
- `missing_information` 必须说明为什么需要人工审核；
- 不得输出可直接执行的设备控制指令；
- 不得把“需要人工审核”降级成普通 `completed`；
- 即使只读观察成功，最终状态仍可以是 `human_review_required`。

`human_review_required` 是业务终态，不等于 Agent 运行失败。

## 7. 诊断报告契约

Pydantic 模型：`DiagnosisReport`

实现位置：`app/schemas/diagnostics.py`

| 字段 | 类型 | 含义 |
|---|---|---|
| `request_id` | `str` | 请求追踪标识 |
| `prompt_version` | `str` | 最终诊断或 Planner Prompt 版本 |
| `gate_version` | `str` | 证据门控或报告白名单规则版本 |
| `status` | `DiagnosisStatus` | 四类公开状态之一 |
| `symptoms` | `list[DiagnosisSymptom]` | 来自用户描述或日志的明确现象 |
| `evidence` | `list[DiagnosisEvidence]` | Python 从真实检索结果构造的证据 |
| `possible_causes` | `list[DiagnosisCause]` | 有证据支持的可能原因 |
| `next_checks` | `list[DiagnosisCheck]` | 有证据支持的下一步只读检查 |
| `risk_level` | `str` | `low`、`medium`、`high` 或 `unknown` |
| `missing_information` | `list[str]` | 仍缺少或需要人工确认的信息 |
| `abstained` | `bool` | 兼容客户端快速判断拒答状态 |

每个原因和检查项中的 `evidence_chunk_ids` 必须属于当前报告的 `evidence` 白名单。

高风险检查必须设置：

`requires_qualified_person=true`

## 8. 引用契约

最终诊断中的引用使用 `DiagnosisEvidence`。

知识库查询工具内部使用 `KnowledgeCitation`。服务层需要把它转换成最终报告引用，不能让 Planner 自己填写文件名、页码、相似度或正文。

### 8.1 `DiagnosisEvidence`

| 字段 | 类型 | 约束 |
|---|---|---|
| `chunk_id` | `str` | 真实知识库 Chunk 标识 |
| `document_id` | `str` | 64 位小写 SHA-256 |
| `source_file` | `str` | 不包含服务器绝对路径的来源文件名 |
| `page_or_section` | `str` | 原文页码或章节 |
| `rank` | `int` | 从 1 开始的报告级唯一顺序 |
| `rrf_score` | `float` | 必须大于 0 |
| `vector_similarity` | `float \| null` | 范围为 -1 到 1 |
| `excerpt` | `str` | 从真实 Chunk 复制的原文 |

`rrf_score` 是融合排序分数，不是答案正确概率。

`vector_similarity` 为 `null` 表示该证据只出现在关键词候选中，不表示证据无效。

### 8.2 引用白名单

最终报告允许使用的证据 ID 必须来自本次请求中成功执行的 `search_knowledge` 工具结果。

以下行为必须被拒绝：

- Planner 虚构 Chunk ID；
- 使用其他请求曾经检索到的 Chunk；
- 只提供文件名但没有本次请求内证据；
- 把 Vision 观察伪装成知识库引用；
- 把模拟遥测伪装成文档证据。

## 9. 工具轨迹契约

Pydantic 模型：`AgentToolTraceEvent`

实现位置：`app/schemas/agent_api.py`

| 字段 | 类型 | 含义 |
|---|---|---|
| `step_id` | `int` | 从 1 开始连续递增的步骤编号 |
| `event_type` | `tool_execution` | 当前固定表示工具执行记录 |
| `tool_name` | `str` | 实际调用的注册工具名 |
| `input_summary` | `str` | 经过脱敏和长度限制的输入摘要 |
| `result_summary` | `str` | 经过脱敏和长度限制的结果摘要 |
| `status` | `str` | `success`、`empty`、`rejected`、`timeout` 或 `error` |
| `duration_ms` | `float` | 工具执行耗时 |
| `error_code` | `str \| null` | 非成功状态对应的稳定错误码 |

状态与错误码必须匹配：

| 状态 | 允许的错误码 |
|---|---|
| `success` | 必须为 `null` |
| `empty` | `empty_result` |
| `rejected` | `unknown_tool`、`invalid_tool_arguments`、`tool_blocked`、`duplicate_tool_call` |
| `timeout` | `tool_timeout`、`vision_timeout` |
| `error` | `tool_execution_error`、`invalid_tool_output`、`vision_upstream_error`、`invalid_vision_response` |

公开轨迹只能保存摘要，不能保存完整工具参数和完整工具输出。

## 10. Agent 执行摘要

Pydantic 模型：`AgentExecutionSummary`

| 字段 | 类型 | 含义 |
|---|---|---|
| `planner_prompt_version` | `str` | Planner Prompt 版本 |
| `state` | `completed \| aborted` | Runner 是否正常结束 |
| `termination_reason` | `str` | Python 控制层决定的终止原因 |
| `termination_message` | `str` | 可以公开的终止说明 |
| `finish_reason` | `str \| null` | Planner 正常结束时的业务原因 |
| `step_count` | `int` | 已完成工具步骤数量 |
| `steps` | `list[AgentToolTraceEvent]` | 脱敏工具轨迹 |
| `progress` | `AgentProgress \| null` | 最终进度状态快照 |
| `missing_information` | `list[str]` | 执行结束时仍存在的缺口 |

`step_count` 必须等于 `steps` 的长度，`step_id` 必须从 1 开始连续递增。

需要区分两个层次：

- `execution.state` 表示 Runner 是否正常结束；
- `diagnosis.status` 表示诊断业务结果。

例如，安全策略正常识别出高风险请求时：

- `execution.state` 可以是 `completed`；
- `finish_reason` 为 `human_review_required`；
- `diagnosis.status` 为 `human_review_required`。

这不是运行失败。

## 11. SSE 诊断接口

### 11.1 接口

`POST /api/v1/agent/diagnose/stream`

状态：已实现。

请求体与 `AgentDiagnosisRequest` 相同。

响应类型为 `text/event-stream`。

该接口必须复用 `AgentDiagnosisService`，不能再实现一套独立的诊断业务流程。

### 11.2 SSE 事件结构

每个 SSE 事件包含：

- SSE `id`：从 1 开始单调递增；
- SSE `event`：公开事件类型；
- SSE `data`：经过 Pydantic 校验后序列化的 JSON。

所有事件的公共 JSON 字段为：

| 字段 | 类型 | 含义 |
|---|---|---|
| `schema_version` | `str` | 当前固定为 `agent-sse-event-v1` |
| `sequence` | `int` | 与 SSE `id` 一致 |
| `event_type` | `str` | 事件类型判别字段 |
| `request_id` | `str` | 当前请求追踪标识 |
| `state` | `str` | 当前公开生命周期状态 |
| `message` | `str` | 简短、安全、可展示的信息 |

事件模型应使用 Pydantic 判别联合：

`event_type` 是 discriminator。

这样不同事件只能携带自己允许的字段，不能通过大量可选字段拼出互相矛盾的事件。

### 11.3 事件类型

| `event_type` | 附加数据 | 作用 |
|---|---|---|
| `request_received` | 图片数量、是否包含任务目标 | 请求已通过基础结构校验 |
| `safety_classified` | 策略处置、原因码 | 确定是否继续、拒答或转人工 |
| `tool_scope_decided` | 必要能力、允许工具名 | 公开最小工具范围 |
| `planning_started` | 下一步骤号 | Planner 开始决定下一步 |
| `tool_started` | 步骤号、工具名、输入摘要 | 工具开始执行 |
| `tool_finished` | `AgentToolTraceEvent` | 工具成功、空结果、拒绝、超时或失败 |
| `progress_updated` | `AgentProgress` | 确定性状态机完成一次归约 |
| `diagnosis_finished` | `AgentDiagnosisResponse` | 最终结果已经校验并持久化 |
| `stream_error` | `ErrorDetail`、是否可重试 | 流已经开始后发生公开错误 |

### 11.4 事件顺序

正常流程至少满足：

```text
request_received
→ safety_classified
→ tool_scope_decided
→ planning_started
→ tool_started
→ tool_finished
→ progress_updated
→ diagnosis_finished
```

工具可以执行多次，因此中间四类事件可以重复。

每个 `tool_started` 必须有同一 `step_id` 的 `tool_finished`，除非连接中断或进程异常终止。

一次流只能有一个终端事件：

- `diagnosis_finished`；或者
- `stream_error`。

发送终端事件后，服务端必须关闭 SSE 连接。

### 11.5 SSE 示例

```text
id: 1
event: request_received
data: {"schema_version":"agent-sse-event-v1","sequence":1,"event_type":"request_received","request_id":"request-demo-001","state":"received","message":"诊断请求已经接收","image_count":1}

id: 2
event: tool_started
data: {"schema_version":"agent-sse-event-v1","sequence":2,"event_type":"tool_started","request_id":"request-demo-001","state":"tool_running","message":"开始执行图片观察工具","step_id":1,"tool_name":"analyze_robot_image","input_summary":"图片观察；image_ref=image_001"}

id: 3
event: tool_finished
data: {"schema_version":"agent-sse-event-v1","sequence":3,"event_type":"tool_finished","request_id":"request-demo-001","state":"observing","message":"图片观察工具执行完成","trace":{"step_id":1,"event_type":"tool_execution","tool_name":"analyze_robot_image","input_summary":"图片观察；image_ref=image_001","result_summary":"返回结构化视觉观察","status":"success","duration_ms":860.2,"error_code":null}}
```

示例只展示事件编码方式，不代表完整诊断流。

## 12. SSE 错误处理

SSE 响应开始之前发生错误时，使用普通 HTTP JSON 错误响应。

SSE 响应已经开始后发生错误时，不能再修改 HTTP 状态码，必须发送 `stream_error` 事件并关闭连接。

客户端主动断开时：

- 服务端应停止继续向该连接发送事件；
- 应取消不再需要的下游等待；
- 已经执行的工具结果仍可按照会话策略保存；
- 不得伪造 `diagnosis_finished`；
- 控制台必须显示连接已经中断。

## 13. 统一错误响应

Pydantic 模型：`ErrorResponse`

```json
{
  "request_id": "request-demo-001",
  "error": {
    "code": "invalid_request",
    "message": "请求内容不符合接口契约"
  }
}
```

常见 HTTP 状态：

| 状态码 | 含义 |
|---:|---|
| `200` | 请求被业务层正常处理，包括完成、部分结果、拒答和人工审核 |
| `404` | 合法格式的会话 ID 不存在 |
| `422` | 请求字段、路径参数或查询参数未通过 Pydantic/FastAPI 校验 |
| `502` | LLM、Vision、Embedding 或知识库上游返回错误或无效响应 |
| `503` | 必要配置缺失或基础服务不可用 |
| `504` | 上游服务调用超时 |

业务上的 `abstained` 和 `human_review_required` 不是 HTTP 错误，因此通常仍返回 `200`。

## 14. 诊断会话接口

### 14.1 最近会话

`GET /api/v1/diagnostic-sessions?limit=20`

状态：已实现。

约束：

- `limit` 最小为 1；
- `limit` 最大为 100；
- 按创建时间从新到旧返回；
- 只返回摘要，不返回完整工具轨迹和引用正文。

响应模型：`DiagnosticSessionListResponse`

字段：

- `sessions`；
- `total`；
- `limit`。

### 14.2 会话详情

`GET /api/v1/diagnostic-sessions/{session_id}`

状态：已实现。

`session_id` 必须是 UUID4。

返回 `DiagnosticSessionRecord`，其中包括：

- 脱敏请求摘要；
- 四类业务状态或运行失败状态；
- 终止原因；
- 诊断快照；
- 工具轨迹；
- Vision 观察；
- 模拟遥测；
- 测试草案；
- 证据覆盖摘要；
- 耗时指标。

记录不得包含完整图片和完整原始日志。

## 15. 控制台诊断结果导出

### 15.1 实现位置与边界

当前 MVP 不提供独立的服务端会话导出路由。

Web 控制台在收到已经通过服务端契约校验的 `AgentDiagnosisResponse` 后，将该对象保存在浏览器内存中的 `latestAgentResponse`。用户点击导出按钮时，`app/web/app.js` 中的 `exportLatestResponseAsJson()` 或 `exportLatestResponseAsMarkdown()` 确定性生成下载文件。

支持格式：

- `json`；
- `markdown`。

### 15.2 JSON 导出

JSON 导出使用当前控制台已经收到的脱敏诊断响应，不得重新调用 Planner、LLM、Vision 或知识库。

Content-Type：

`application/json; charset=utf-8`

### 15.3 Markdown 导出

Markdown 导出由控制台已经收到并通过服务端校验的诊断响应确定性生成。

Content-Type：

`text/markdown; charset=utf-8`

Markdown 至少包含：

- 会话 ID；
- 请求摘要；
- 最终状态；
- 症状及来源；
- Vision 观察及图片引用；
- 模拟遥测；
- 可能原因；
- 下一步检查；
- 知识库引用；
- 工具轨迹；
- 终止原因；
- 缺失信息；
- Planner、策略和 Prompt 版本。

导出操作不得重新生成工程结论，也不得加入报告生成时间。

## 16. 来源分层

控制台、SSE 和导出报告必须区分以下来源：

| 来源 | 可信含义 |
|---|---|
| `user_report` | 用户描述的现象，不代表已经验证 |
| `log_excerpt` | 用户提交的脱敏日志内容 |
| `vision_model` | 模型对图片可见内容的观察，不是知识库事实 |
| `simulated_memory` | 教学用模拟遥测，不是真实生产遥测 |
| `confirmed_knowledge` | 通过检索和证据门控确认的知识库内容 |
| `system_clock` | 应用服务器的 UTC 时间 |

不同来源不能互相冒充。

Vision 观察可以支持“图片上看到了什么”，但不能单独证明设备规范、故障根因或安全操作要求。

## 17. Pydantic 模型归属

当前已经存在的模型：

| 模型 | 文件 |
|---|---|
| `AgentDiagnosisRequest` | `app/schemas/agent_api.py` |
| `AgentDiagnosisResponse` | `app/schemas/agent_api.py` |
| `AgentToolTraceEvent` | `app/schemas/agent_api.py` |
| `AgentExecutionSummary` | `app/schemas/agent_api.py` |
| `DiagnosisReport` | `app/schemas/diagnostics.py` |
| `DiagnosisEvidence` | `app/schemas/diagnostics.py` |
| `KnowledgeCitation` | `app/schemas/knowledge_query.py` |
| `DiagnosticSessionRecord` | `app/schemas/diagnostic_session.py` |
| `ErrorResponse` | `app/schemas/error.py` |

第七周已经新增或扩展的模型：

| 模型 | 目标文件 | 作用 |
|---|---|---|
| 扩展后的 `DiagnosisStatus` | `app/schemas/diagnostics.py` | 增加 `human_review_required` |
| `AgentSseEventBase` | `app/schemas/agent_events.py` | 定义所有 SSE 事件的公共字段 |
| 各类具体 SSE 事件模型 | `app/schemas/agent_events.py` | 约束每种事件允许携带的数据 |
| `AgentSseEvent` | `app/schemas/agent_events.py` | 使用 `event_type` 组成判别联合 |
| 控制台导出函数 | `app/web/app.js` | 将已校验诊断响应确定性导出为 JSON 或 Markdown |

## 18. 端到端调用链

普通 JSON 诊断：

```text
Web控制台（收集脱敏文字和可选图片）
→ FastAPI Agent Router（校验AgentDiagnosisRequest）
→ AgentDiagnosisService（协调请求策略、Runner和会话保存）
→ AgentToolPolicy与安全分类器（确定业务处置和最小工具范围）
→ AgentRunner（执行受控规划循环）
→ ToolExecutor（校验并执行只读工具）
→ 报告构造器（执行引用白名单和最终数据校验）
→ DiagnosticSessionStore（持久化脱敏会话）
→ AgentDiagnosisResponse（返回最终JSON）
```

SSE 诊断实现链路：

```text
Web控制台（发起流式诊断）
→ FastAPI SSE Router（校验请求并建立SSE连接）
→ AgentEventPublisher（接收内部进度并生成公开事件）
→ AgentDiagnosisService（复用现有诊断主流程）
→ AgentRunner与工具层（持续产生内部状态变化）
→ AgentEventPublisher（转换成经过Pydantic校验的脱敏事件）
→ SSE Router（编码并推送事件）
→ DiagnosticSessionStore（保存最终脱敏会话）
→ diagnosis_finished或stream_error（结束事件流）
```

## 19. 契约级验收条件

第七周实现必须至少验证：

1. 普通 JSON 请求仍兼容现有文本诊断和可选图片。
2. 额外字段无法扩大工具权限。
3. 四类公开状态都能通过 Pydantic 校验。
4. `human_review_required` 不会被错误转换成 `abstained` 或 `completed`。
5. SSE `sequence` 从 1 开始连续递增。
6. 每次成功工具执行都产生对应的公开完成事件。
7. 工具失败只公开安全错误码和脱敏摘要。
8. SSE 只产生一个终端事件。
9. 最终 SSE 结果与保存的诊断会话一致。
10. 引用只能使用本次请求已确认的 Chunk。
11. Vision、遥测和知识库来源被明确区分。
12. 会话导出不重新调用概率型服务。
13. 响应、事件、会话和导出中不包含图片 Base64、密钥、完整日志或私有思维链。
14. 高风险请求可以完成允许的只读观察，但最终必须进入 `human_review_required`。
15. 现有 JSON 接口、会话列表和会话详情接口保持兼容。
