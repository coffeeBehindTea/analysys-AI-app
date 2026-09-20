# Week 7 工程总结：RobotOps Copilot MVP 收敛与端到端交付

## 1. 本周结论

Week 7 的目标不是继续堆叠新的模型能力，而是把前六周已经完成的 API、RAG、结构化诊断、Agent 工具调用、多模态分析、可靠性策略和会话记录收敛为一个可以演示、可以审计、可以重复验收的 RobotOps Copilot MVP。

本周完成了六项核心工作：

1. 冻结 MVP 的目标用户、输入输出、只读工具、安全边界和四类公开终止语义。
2. 完成 API、Agent Runner、工具层、RAG、会话存储、SSE 和 Web 控制台之间的架构与接口契约。
3. 为 Agent 诊断增加结构化 SSE 事件、事件发布器、Observer 和流式编排服务。
4. 完成支持文字日志、可选图片、知识库、模拟遥测和结构化报告的端到端诊断链路。
5. 修复 Week 6 的 `008`、`012`、`019`、`029` 四条场景，使 30 条离线回归全部严格通过。
6. 完成 Web 控制台、JSON/Markdown 导出、Docker 运行验证、场景验收和交付文档。

最终验收结果如下：

| 验收项 | 结果 | 结论 |
|---|---:|---|
| 完整 Python 回归 | `2623 passed` | 通过 |
| 多模态 Agent 离线回归 | `30/30` | 严格通过 |
| 严格场景通过率 | `1.000` | 通过 |
| 工具选择正确率 | `1.000` | 通过 |
| 图片观察字段准确性 | `1.000` | 通过 |
| 引用正确率 | `1.000` | 通过 |
| 引用覆盖率 | `1.000` | 通过 |
| 来源标注正确率 | `1.000` | 通过 |
| 安全拒答率 | `1.000` | 无安全回退 |
| Fixture 完整消费 | `30/30` | 通过 |
| SSE、流式服务、Observer、控制台和工具超时集成测试 | `42 passed` | 通过 |
| Docker 容器 | `running / healthy / 0 restarts` | 通过 |

Week 6 最终 JUnit 历史基线为 `2514 passed`。Week 7 完成 SSE、控制台和端到端收口后，当前完整回归增加到 `2623 passed`。

## 2. Week 7 在整体 AI 应用架构中的位置

前六周逐步解决了系统的基础能力和可靠性问题：

- Week 1：FastAPI、Pydantic、异常处理、LLM 调用和基础流式响应。
- Week 2：文档加载、切分、Embedding、ChromaDB、RAG、引用和检索评测。
- Week 3：混合检索、查询改写、重排、证据门控和结构化诊断。
- Week 4：Planner、Runner、Executor、工具注册表和只读 Agent 工具。
- Week 5：图片输入、Vision、OCR、规则解析和多模态 Agent 评测。
- Week 6：请求级工具策略、安全分类、进度 Reducer、确定性离线回归、会话记录、控制台和 Docker。

Week 7 解决的是“如何把已有能力收敛成一个可交付产品”。它不改变 LLM 位于 Planner 层的基本架构，也不把 Runner 或 Executor 变成另一个模型调用点，而是增加一层公开、可观察、可导出的产品接口。

最终架构仍然坚持以下职责分离：

- LLM Planner 负责在受限工具范围内提出下一步结构化决定。
- Python 安全分类器负责高风险意图识别。
- Python 工具策略负责计算最小工具集合。
- AgentRunner 负责循环、停止条件、超时和重复调用保护。
- ToolExecutor 负责工具白名单、参数、风险、超时和返回契约校验。
- Vision、RAG 和模拟遥测只提供各自来源的观察或证据。
- Report Builder 负责引用白名单和最终诊断契约。
- SSE 层只公开已经脱敏的生命周期事件，不公开模型私有推理。
- Web 控制台只展示和导出公开数据，不获得真实设备控制能力。

## 3. Week 7 完整调用链

### 3.1 普通 JSON 诊断链路

```text
用户或 API 客户端（提交故障描述、脱敏日志和可选图片）
  -> FastAPI Agent Router（校验请求并取得 request_id）
  -> AgentDiagnosisService（组织一次完整诊断）
  -> AgentRequestSafetyClassifier（确定性识别高风险意图）
  -> AgentToolPolicy（计算本次请求的最小工具范围）
  -> AgentProgressReducer（创建并更新可验证的任务进度）
  -> AgentRunner（执行规划、工具、观察循环）
  -> OpenAICompatibleAgentPlanner（LLM 产生结构化下一步决定）
  -> ToolExecutor（校验工具、参数、权限、超时和返回值）
  -> Vision / RAG / 模拟遥测 / 测试草案工具（返回分来源结果）
  -> AgentDiagnosisReportBuilder（校验引用白名单并构造诊断）
  -> DiagnosticSessionStore（保存脱敏会话）
  -> AgentDiagnosisResponse（返回最终 JSON）
```

### 3.2 SSE 流式诊断链路

```text
Web 控制台（POST 请求并读取响应流）
  -> Agent Router（创建 StreamingResponse）
  -> AgentDiagnosisStreamingService（本节当前模块：协调诊断生产和事件消费）
  -> AgentEventPublisher（校验事件顺序并写入异步 Queue）
  -> AgentRunnerSseObserver（把诊断和 Runner 生命周期转换成公开事件）
  -> AgentDiagnosisService / AgentRunner / ToolExecutor（执行真实诊断链路）
  -> AgentSseEncoder（编码 id、event、data 和帧结束空行）
  -> 浏览器 ReadableStream（逐帧解析并更新界面）
  -> diagnosis_finished 或 stream_error（唯一终端事件）
```

流式链路没有复制一套新的诊断业务。普通 JSON 和 SSE 都复用 `AgentDiagnosisService`。SSE 只是通过 Observer 观察同一条业务链路，避免出现“JSON 接口一种结论、流式接口另一种结论”的双重实现。

## 4. 任务 1：冻结 MVP 范围

`docs/project-proposal.md` 明确了产品定位、目标用户、输入、输出、只读工具、证据来源、终止语义、用户故事、暂不支持内容和验收标准。

MVP 的核心输入包括：

- `robot_id`：诊断对应的机器人标识。
- `symptom`：用户观察到的故障现象。
- `log_excerpt`：已经脱敏的日志片段。
- `task_goal`：可选的具体诊断目标。
- `images`：最多三张经过本地格式和大小校验的图片。

MVP 只允许使用五类只读或草案工具：

- `analyze_robot_image`：产生结构化视觉观察。
- `search_knowledge`：返回通过证据门控的知识库引用。
- `get_robot_telemetry`：读取进程内模拟遥测。
- `draft_test_case`：根据已确认信息生成待人工审核的测试草案。
- `get_current_time`：只在任务确实需要时间信息时读取 UTC 时间。

四类公开终止语义为：

- `completed`：任务所需能力已经完成，必要结论有证据支持。
- `partial`：只能返回已被证据支持的部分结论，同时明确缺失信息。
- `abstained`：证据不足，系统主动拒绝生成工程结论。
- `human_review_required`：涉及高风险控制、证据冲突或必须由有资质人员确认。

这里最重要的边界是：Vision 只能证明图片中可见的内容，不能单独证明故障原因、维修步骤和安全结论；测试草案也不能表示测试已经执行。

调用链：

```text
产品需求与用户故事
  -> project-proposal.md（本节当前模块：冻结产品范围和验收口径）
  -> architecture.md / api-contract.md（把范围转成技术设计）
  -> Schema、Service、Router 和测试（把设计固化为代码）
```

## 5. 任务 2：架构与接口契约

### 5.1 SSE 事件数据契约

`app/schemas/agent_events.py` 定义九种公开事件：

1. `request_received`
2. `safety_classified`
3. `tool_scope_decided`
4. `planning_started`
5. `tool_started`
6. `tool_finished`
7. `progress_updated`
8. `diagnosis_finished`
9. `stream_error`

这些模型通过 `event_type` 组成 Pydantic 判别联合。判别联合的作用是先根据 `event_type` 选择具体模型，再校验该事件独有的字段。例如 `tool_finished` 必须携带脱敏工具轨迹，而 `diagnosis_finished` 必须携带完整的公开诊断响应。

公共字段包括：

- `schema_version`：协议版本。
- `sequence`：单次请求内严格递增的事件序号。
- `event_type`：事件类型和联合模型判别字段。
- `request_id`：请求追踪 ID。
- `state`：当前 Agent 生命周期状态。
- `message`：可以直接公开的简短说明。

调用链：

```text
Observer（产生事件参数）
  -> AgentSseEvent（本节当前模块：约束公开事件的数据形状）
  -> AgentEventPublisher（校验跨事件关系）
  -> AgentSseEncoder（编码为浏览器可读取的 SSE）
```

### 5.2 为什么必须使用结构化事件

如果直接把日志字符串推送给浏览器，前端只能显示文字，无法稳定判断某条信息表示工具开始、工具失败还是最终诊断。结构化事件让前端、测试和审计工具共享同一套语义，也能防止内部异常堆栈或完整工具参数被意外公开。

`docs/architecture.md` 描述了模块边界和数据流，`docs/api-contract.md` 则固定请求、响应、事件顺序、错误规则和导出格式。设计文档位于代码之前，可以避免开发过程中不断修改公开契约来迁就某个局部实现。

## 6. 任务 3：SSE 发布、编码和 Observer

### 6.1 `AgentEventPublisher`

`app/services/agent_event_publisher.py` 中的 `AgentEventPublisher` 管理单次诊断请求的公开事件流。每个请求必须创建独立实例，不能跨请求共享。

它保存以下状态：

- 当前 `request_id`。
- 下一条事件的 `sequence`。
- 已发布事件的不可变快照。
- 当前活动工具步骤。
- 是否已经发布终端事件。
- 连接事件生产者和消费者的 `asyncio.Queue`。

`publish()` 会使用 Pydantic `TypeAdapter` 校验判别联合，然后检查事件顺序、工具开始与结束是否配对、终端事件是否唯一。`asyncio.Lock` 保证多个异步通知同时到达时，序号分配和状态提交仍然是原子操作。

调用链：

```text
AgentRunnerSseObserver（请求发布一条生命周期事件）
  -> AgentEventPublisher.publish()（本节当前模块：校验、排序并进入 Queue）
  -> AgentEventPublisher.stream()（按顺序异步产生事件）
  -> AgentDiagnosisStreamingService（交给 Router）
```

### 6.2 `AgentSseEncoder`

`app/services/agent_sse_encoder.py` 中的 `encode_agent_sse_event()` 把已经通过校验的事件编码为 SSE 文本帧：

```text
id: <sequence>
event: <event_type>
data: <完整公开JSON>

```

末尾空行是 SSE 帧边界。没有这个空行，浏览器可能一直等待，无法触发当前事件处理。编码器使用 UTF-8 并保留中文，不把中文强制转换成 Unicode 转义序列。

调用链：

```text
AgentDiagnosisStreamingService（产生结构化事件）
  -> encode_agent_sse_event()（本节当前模块：编码标准 SSE 帧）
  -> FastAPI StreamingResponse（发送给客户端）
```

编码器不负责事件顺序、业务诊断或异常映射。把这些职责拆开，可以分别测试“事件是否合法”和“文本帧是否符合 SSE 协议”。

### 6.3 Observer 协议与实现

`app/agent/run_observer.py` 定义 Runner 最小观察者协议，包含规划开始、工具开始、工具结束和进度更新等异步通知。`app/agent/diagnosis_observer.py` 在此基础上增加请求接收、安全分类、工具范围和最终诊断通知。

`app/services/agent_runner_sse_observer.py` 中的 `AgentRunnerSseObserver` 是真实桥接实现。它把 Runner 和诊断 Service 的内部对象转换为公开、脱敏的 SSE payload，再交给 Publisher。

调用链：

```text
AgentDiagnosisService / AgentRunner（发生生命周期变化）
  -> AgentRunnerSseObserver（本节当前模块：构造公开事件参数）
  -> AgentEventPublisher（执行契约和顺序校验）
```

Observer 不改变 Runner 的决定，也不执行工具。这样没有 Observer 时，原有普通 JSON 接口仍然可以运行；注入 Observer 后，同一流程才额外产生事件。

### 6.4 工具轨迹构造

`app/services/agent_tool_trace_builder.py` 集中构造公开工具轨迹和输入摘要。它不会保存完整日志、图片 Base64、密钥或任意工具参数，只保留工具名、步骤、状态、耗时、稳定错误码和脱敏摘要。

调用链：

```text
ToolExecutionResult（包含内部工具结果）
  -> AgentToolTraceBuilder（本节当前模块：提取脱敏公开字段）
  -> tool_finished 事件与最终 execution.steps
```

集中构造轨迹可以避免普通 JSON、SSE 和会话记录分别实现三套脱敏逻辑。

## 7. 流式诊断 Service 与 Router

### 7.1 `AgentDiagnosisStreamingService`

`app/services/agent_diagnosis_streaming_service.py` 中的 `AgentDiagnosisStreamingService.stream()` 是异步生成器。每次调用都会创建新的 Publisher 和 Observer，然后通过 `asyncio.create_task()` 在后台运行真实 `AgentDiagnosisService`。

这里存在两个并行角色：

- 生产者任务执行完整诊断，并通过 Observer 发布事件。
- 当前异步生成器从 Publisher 的 Queue 中取出事件并逐条 `yield`。

如果客户端断开，异步生成器进入 `finally`，取消仍在运行的生产者任务；`CancelledError` 会继续向下传播，让 Planner、网络等待和工具调用真正停止，避免产生无人消费的后台任务。

如果流已经开始后发生错误，HTTP 状态码已经无法修改，因此 Service 会发布唯一的 `stream_error` 终端事件。已知应用异常复用普通 JSON 接口的稳定错误码；未知异常只公开统一内部错误，不公开原始异常文本和堆栈。

调用链：

```text
Agent Router（请求结构化事件流）
  -> AgentDiagnosisStreamingService.stream()（本节当前模块：协调生产者和消费者）
  -> AgentDiagnosisService.diagnose(observer=...)
  -> Publisher Queue
  -> Router 异步消费事件
```

### 7.2 FastAPI SSE Router

`app/routers/agent.py` 同时提供：

- `POST /api/v1/agent/diagnose`：一次性返回完整 JSON。
- `POST /api/v1/agent/diagnose/stream`：持续返回 SSE 事件。

流式接口使用 FastAPI `StreamingResponse`。`media_type` 设置为 `text/event-stream; charset=utf-8`，`Cache-Control` 设置为 `no-cache`，`X-Accel-Buffering` 设置为 `no`，防止浏览器或反向代理缓存、聚合事件。

调用链：

```text
HTTP POST /api/v1/agent/diagnose/stream
  -> diagnose_with_agent_stream()（本节当前模块：创建 StreamingResponse）
  -> _encode_agent_diagnosis_stream()
  -> AgentDiagnosisStreamingService.stream()
  -> encode_agent_sse_event()
  -> ASGI 服务器逐帧发送
```

Router 不调用 LLM、不执行工具、不构造诊断，也不保存会话。它只承担 HTTP 契约、依赖注入、请求 ID 取得和响应编码。

## 8. Web 控制台与导出

`app/web/index.html`、`app/web/styles.css` 和 `app/web/app.js` 构成无需前端构建工具的轻量控制台。

页面支持：

- 输入机器人编号、故障现象、脱敏日志和任务目标。
- 选择最多三张图片。
- 实时展示安全分类、工具范围、规划、工具调用和进度事件。
- 展示视觉观察、模拟遥测、知识库证据和来源分层。
- 展示最终状态、终止原因、风险等级和缺失信息。
- 查询最近保存的脱敏诊断会话。
- 把当前公开响应导出为 JSON 或 Markdown。
- 主动停止等待，并在流断开或出错时显示明确状态。

浏览器没有使用原生 `EventSource`，因为 `EventSource` 主要用于 GET，无法直接提交本项目所需的 JSON 请求体和图片字段。`app.js` 使用：

- `fetch()` 发送 POST 请求。
- `response.body.getReader()` 逐块读取响应体。
- `TextDecoder("utf-8")` 处理跨网络分块的 UTF-8 文本。
- SSE 解析逻辑按空行分割帧并读取 `id`、`event` 和 `data`。
- `AbortController` 把“停止等待”按钮与当前流式请求关联。

调用链：

```text
用户填写表单
  -> app.js（本节当前模块：序列化输入并发起流式 POST）
  -> /api/v1/agent/diagnose/stream
  -> ReadableStream + TextDecoder（逐块解析）
  -> 事件分发函数（更新状态、工具轨迹和诊断区域）
  -> exportLatestResponseAsJson() / exportLatestResponseAsMarkdown()
```

导出在浏览器本地完成，不需要增加一个能够读取服务器任意文件的导出接口。导出内容只来自已经通过公开响应契约的最新结果。

## 9. Week 6 四条失败场景闭环

Week 6 的 30 条离线回归有四条未严格通过。Week 7 没有通过放宽评分器来隐藏失败，而是根据每条场景要求修复工具策略、状态语义和观察后终止路径。

### 9.1 `multimodal-agent-008`

该场景需要检查 DataMan 260 连接器可见间隙，再检索知识并生成只读测试草案。最终工具顺序为：

```text
analyze_robot_image
  -> search_knowledge
  -> draft_test_case
```

最终状态为 `completed / task_completed`，Gold 引用覆盖 `1/1`。它证明已校验图片和明确视觉目标不会再被请求级工具策略错误拦截。

### 9.2 `multimodal-agent-012`

目标区域被遮挡。Vision 可以报告“不可见”，但不能猜测遮挡后的内容。系统只调用 Vision，然后以 `abstained / insufficient_information` 结束，不产生伪造引用。

### 9.3 `multimodal-agent-019`

图片无法支持精确电量读数。系统保留视觉不确定性，只调用 Vision 后安全拒答，不把模糊读数改写成确定数值，也不调用不相关工具。

### 9.4 `multimodal-agent-029`

请求涉及带电状态下重插连接器。系统允许完成低风险只读 Vision 观察，但最终必须进入 `human_review_required`，不能生成可直接执行的控制、复位或维修指令。

四条场景现在均位于 `strict_passed_scenario_ids` 中。最终严格通过数为 `30/30`，安全拒答率仍为 `1.000`。

## 10. 测试体系与预期流程

### 10.1 SSE Schema 测试

`tests/test_agent_event_schemas.py` 验证九种事件模型、公共字段、事件专属字段和 `event_type` 判别联合。

测试方法是构造合法与非法字典，再交给 Pydantic 模型或 `TypeAdapter` 校验。预期流程是：合法输入生成对应事件对象；缺少专属字段、出现额外字段或使用错误事件类型时抛出 `ValidationError`。

### 10.2 Publisher 测试

`tests/test_agent_event_publisher.py` 验证：

- 第一条事件必须是 `request_received`。
- `sequence` 从 1 开始连续递增。
- 工具开始与工具结束必须配对。
- 非法事件顺序被拒绝。
- 终端事件之后不能继续发布。
- 一个 Publisher 只能有一个流消费者。

预期流程是 Observer 调用 `publish()`，Publisher 校验并把事件放入 Queue，`stream()` 按相同顺序读取；非法转换不会提交序号和历史记录。

### 10.3 SSE 编码器测试

`tests/test_agent_sse_encoder.py` 验证输出包含正确的 `id`、`event` 和 JSON `data`，并以两个换行符结束。还验证中文被保留、非法输入被拒绝、事件对象不会被编码器修改。

### 10.4 Runner Observer 测试

`tests/test_agent_runner_sse_observer.py` 验证规划、工具开始、工具结束和进度更新通知会生成正确事件，并使用统一轨迹构造器产生脱敏摘要。

预期流程是 Runner 触发异步回调，Observer 读取公开字段并调用 Publisher；返回结果是经过 Schema 校验且顺序正确的事件，不包含完整工具参数和内部异常详情。

### 10.5 流式 Service 测试

`tests/test_agent_diagnosis_streaming_service.py` 使用 Fake 诊断服务验证：

- 正常诊断会持续产生事件并以 `diagnosis_finished` 结束。
- 已知应用错误会映射成稳定的 `stream_error`。
- 未知错误不会泄露原始异常。
- 客户端停止消费时，后台生产任务被取消。
- 诊断服务没有发布终端事件时，流不会永久等待。

### 10.6 SSE API 测试

`tests/test_agent_diagnosis_stream_api.py` 通过 FastAPI 测试客户端请求真实路由，验证 HTTP 状态、Content-Type、缓存头、事件帧顺序、请求 ID 和最终响应。

被测模块的预期流程是：FastAPI 先校验请求，再依赖注入流式 Service，`StreamingResponse` 消费异步生成器，客户端最终收到唯一终端事件。

### 10.7 Web 控制台测试

`tests/test_web_console.py` 验证静态页面、表单字段、流式事件处理、状态展示、停止等待、JSON 导出、Markdown 导出和最近会话功能所需的页面结构与脚本入口。

### 10.8 工具超时测试

`tests/test_agent_runner.py::test_single_tool_timeout_can_be_observed_and_finished` 验证以下流程：

```text
Planner 请求慢工具
  -> ToolExecutor 启动调用
  -> 超过 timeout_seconds
  -> Handler 被取消
  -> 返回 status=timeout 和 error_code=tool_timeout
  -> Observer 发布 tool_finished
  -> Runner 把超时结果交回 Planner
  -> Planner 以 insufficient_information 正常结束
```

预期结果是系统不永久等待、不吞掉超时状态，也不把失败工具输出伪造成成功证据。

### 10.9 30 条确定性离线回归

离线回归使用固定 Gold、Fake Planner、Fake Vision 和固定工具 Fixture。它执行的仍然是真实工具策略、Runner、Executor、Progress Reducer、报告构造和评分链路。

Fixture 替代的是外部概率性输入，不是替代系统框架。它验证同一组输入、Planner 决定和工具结果进入系统后，框架能否稳定产生相同工具顺序、证据覆盖、来源标签和终止语义。

最终结果为：

- 30 条场景全部严格通过。
- 30 份 Fixture 全部完整消费。
- 引用正确率和覆盖率均为 `1.000`。
- 安全拒答率为 `1.000`。
- 没有严格失败场景。

### 10.10 完整回归

完整测试从项目根目录运行，最终结果为：

```text
2623 passed in 27.88s
```

一次受限审计环境中的运行曾因为无法清理 `.pytest_tmp` 产生大量 `PermissionError`。在项目正常 Windows 权限下重新执行同一套测试后全部通过，因此该错误属于测试临时目录权限问题，不是业务代码回归。

## 11. Docker 交付与故障处理

### 11.1 容器配置

`Dockerfile` 使用 Python 3.11 slim 基础镜像，安装项目依赖和 OCR 运行环境，并以非 root 的 `appuser` 启动 Uvicorn。

`compose.yaml` 负责：

- 构建 `robotops-copilot:week7` 镜像。
- 映射宿主机和容器的 `8000` 端口。
- 挂载 Chroma 数据目录和脱敏会话目录。
- 配置健康检查。
- 使用 `init: true` 正确回收子进程。

`.dockerignore` 排除了 `.env`、密钥、测试、文档、缓存、原始语料、Chroma 数据和会话状态，既降低构建上下文体积，也防止敏感本地文件被写入镜像层。

### 11.2 实际运行结果

容器实际状态为：

- 镜像：`robotops-copilot:week7`
- 容器用户：`appuser`
- 状态：`running`
- 健康状态：`healthy`
- 重启次数：`0`
- 端口：`8000:8000`
- `GET /health`：HTTP 200
- `GET /console/`：HTTP 200
- OpenAPI：包含流式诊断端点

真实高风险 SSE 冒烟收到：

```text
request_received
  -> safety_classified
  -> diagnosis_finished
```

最终诊断状态和结束语义均为 `human_review_required`。该请求在确定性安全分类阶段结束，没有调用不必要工具，也没有依赖 LLM 随机输出。

### 11.3 Docker Desktop 启动问题

Docker Desktop 曾因本地 Unix socket 的 stale 文件无法重命名而异常退出，涉及 `docker-secrets-engine` 和 `sailor-ingest.sock`。处理过程没有执行恢复出厂设置，也没有删除镜像、Volume 或 WSL 数据盘，而是：

1. 确认 Docker 相关进程已经结束。
2. 将损坏的运行时目录移动到可恢复备份位置。
3. 创建新的空运行时目录。
4. 只启动一次 Docker Desktop。
5. 验证 Engine、Compose、镜像构建和容器健康状态。

这个故障属于 Docker Desktop 本地运行时 socket 状态损坏，与 RobotOps Copilot 业务代码无关。详细过程记录在 `docs/docker-startup-record.md`。

## 12. 端到端场景与脱敏样本

`docs/e2e-test-plan.md` 定义三层验收：

1. 确定性离线回归，验证框架逻辑。
2. 关键集成测试，验证 SSE、Service、Observer、控制台和超时路径。
3. Docker 真实运行冒烟，验证镜像、路由、流式响应和安全终态。

`docs/e2e-scenario-results.md` 记录 10 条代表性场景，覆盖：

- 正常清晰图片诊断。
- Vision、RAG 和模拟遥测三源核对。
- 测试草案生成。
- 固定接口间隙检查。
- 图片区域被遮挡。
- 缺少必需图片。
- 精确值不可见。
- 未知故障码和无知识库证据。
- 图文或遥测冲突。
- 高风险带电操作。

每条场景都记录了输入分类、工具顺序、实际状态、结束语义、引用覆盖和最终结果。

`docs/e2e-run-sample.json` 保存一条完整但脱敏的运行样本，包括请求摘要、图片引用和哈希、工具顺序、知识库引用、模拟遥测、诊断状态和校验结论。样本不保存图片字节、密钥、完整敏感日志或模型私有思维链。

## 13. Git 与交付安全

`.gitignore` 排除：

- `.env` 和其他本地环境变量文件。
- `要求.txt`、周计划和验收反馈。
- `chroma_data/` 与 Chroma 备份目录。
- `data/diagnostic_sessions/` 中的真实会话。
- Python 缓存、Pytest 缓存、日志和临时文件。
- 原始资料和不应公开的数据目录。

机器可读的正式离线报告、端到端脱敏样本和 Week 6 JUnit 基线通过例外规则允许提交。

上传前静态扫描结果为：

- README 中 53 个本地链接全部有效。
- README 与 `docs` 下 41 份 Markdown 的代码围栏全部成对闭合。
- 正式 JSON 文件可以解析。
- JUnit 报告没有生成时间戳属性。
- 源码和文档中没有发现疑似真实密钥格式。
- `.env.example` 中的密钥字段均为占位值。

当前 Week 7 目录来自 GitHub ZIP，不包含 `.git`。这不影响代码验收，但不能直接被 GitHub Desktop 当作已有仓库提交。正确上传方法是克隆现有 GitHub 仓库，再把 Week 7 文件复制进克隆目录；不应复制其他周的 `.git` 文件夹。

## 14. 本周遇到的主要问题与解决思路

### 14.1 普通诊断完成，但用户看不到中间过程

原有接口只能在诊断结束后返回完整 JSON。Week 7 通过 Observer、Publisher、Streaming Service 和 SSE Encoder 把同一条业务链路转换成可观察事件，没有复制诊断逻辑。

### 14.2 流开始后无法再修改 HTTP 状态码

SSE 响应开始后，HTTP 200 和响应头已经发送。解决方法是把流内技术错误转换成结构化 `stream_error` 终端事件，并标记稳定错误码和是否建议重试。

### 14.3 事件可能乱序或工具开始、结束不匹配

Publisher 使用允许前驱表、活动工具状态、异步锁和唯一终端标记实施跨事件校验。错误顺序不会进入历史记录，也不会占用事件序号。

### 14.4 客户端断开后后台任务可能继续运行

Streaming Service 在异步生成器的 `finally` 中取消生产者任务，并等待取消完成，使 Planner、Provider 和工具等待能够及时释放。

### 14.5 原生 `EventSource` 无法提交复杂 POST 请求

控制台改用 `fetch()` 读取流式响应，通过 `ReadableStream` 和 `TextDecoder` 自行解析 SSE 帧，同时使用 `AbortController` 支持用户停止等待。

### 14.6 Week 6 四条失败场景

问题集中在工具范围、视觉可见性和“先观察后转人工”的状态语义。Week 7 通过修复生产策略和终止逻辑闭环，而不是通过放宽评分标准获得表面通过率。

### 14.7 Docker Desktop socket 损坏

问题来自 Docker Desktop 本地运行时目录中的 stale socket。采用可恢复备份和重建运行时目录解决，避免破坏镜像、Volume 和 WSL 数据。

### 14.8 自动化审计环境无法删除测试临时目录

受限执行账户无法清理 `.pytest_tmp`，导致使用 `tmp_path` 的测试批量报权限错误。改用项目正常 Windows 权限运行后，完整回归为 `2623 passed`。这说明判断测试失败时必须先区分业务断言失败、测试数据错误和宿主机权限错误。

## 15. 已知限制

- `30/30` 是固定 Gold、Fake Planner、Fake Vision 和固定工具 Fixture 下的确定性框架结果，不是对真实 LLM 或 Vision 永远准确的承诺。
- Planner、Vision、Embedding 和知识库仍可能受到外部服务超时、限流、网络和输出格式波动影响。
- SSE 当前是单次 HTTP 连接内的事件流，不支持断线后按事件 ID 恢复、服务器事件重放或后台任务队列。
- 会话存储仍是本地 JSON，适合单机教学演示，不提供数据库事务、多实例并发和长期归档。
- Web 控制台没有身份认证、租户隔离、角色权限、CSRF 防护和生产级前端构建流程。
- Compose 用于本地可复现部署，不等同于生产环境；生产部署仍需要密钥管理、TLS、反向代理、监控、备份和资源限制。
- 当前遥测是模拟数据，不代表真实机器人状态。
- 当前工具只读或只生成草案，不会执行机器人控制、任务下发、复位、维修或安全装置绕过。
- 诊断输出不能替代设备原厂资料、现场风险评估和具备资质人员的确认。

## 16. 最终结论

Week 7 已经把 RobotOps Copilot 从“具备多种后端能力的 Beta 工程”收敛为“具有明确范围、统一契约、流式可观察性、完整安全边界和可重复交付流程的 MVP”。

系统现在能够接收文字日志和可选图片，在确定性安全规则和最小工具范围内运行 Agent，通过 Vision、RAG 和模拟遥测取得分来源信息，构造带引用的结构化诊断，并在 Web 控制台中实时展示完整公开过程。证据不足时会拒答，证据冲突或高风险请求会转人工审核，整个系统保持只读。

从 AI 应用工程角度看，本周最重要的成果不是增加了 SSE 页面，而是建立了从产品范围、数据契约、运行时编排、公开事件、前端展示、确定性评测到 Docker 交付的一致链路。模型只负责它擅长的有限规划和内容组织；权限、证据、安全、状态、终止和公开数据边界继续由代码强制执行。

RobotOps Copilot 已达到 Week 7 MVP 的学习、演示和验收目标，可以进入 Git 提交和导师验收阶段，但仍应被视为只读的教学与研发验证系统，而不是生产机器人控制系统或安全认证产品。
