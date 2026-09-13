# Week 6 工程总结：Agent 可靠性收口与产品化部署

## 1. 本周结论

Week 6 没有继续增加新的生成式 AI 能力，而是把 Week 5 已经能运行的多模态 Agent 收口为一个更可控、可解释、可复现的 RobotOps Copilot Beta。

本周完成了六项核心工作：

1. 在 Planner 之前增加请求级最小工具策略，限制本次请求可以看到和调用的工具。
2. 在 Agent Loop 之前增加确定性安全分类，使高风险控制意图不依赖 LLM 自由判断。
3. 用 `AgentProgress` 和 `AgentProgressReducer` 记录能力完成情况、工具结果、证据覆盖、冲突、缺失信息和下一步工具范围。
4. 使用 Fake Planner、Fake Vision 和固定 Fixture 建立 30 条场景的确定性离线回归。
5. 保存脱敏诊断会话，并提供最近会话和按 ID 查询详情的 API。
6. 提供轻量 Web 控制台、Dockerfile、Compose 配置和带图片的容器端到端演示。

正式离线回归结果为：

| 指标 | Week 5 基线 | Week 6 结果 | 结论 |
|---|---:|---:|---|
| 严格通过数 | 7/30 | 26/30 | 增加 19 条 |
| 严格通过率 | 0.233 | 0.867 | 超过 24/30 门槛 |
| 安全拒答率 | 1.000 | 1.000 | 无安全回退 |
| 工具选择正确率 | — | 0.867 | 工具范围基本稳定 |
| 任务完成率 | — | 0.875 | 7/8 个预期完成场景完成 |
| 引用正确率 | — | 1.000 | 返回的引用均在 Gold 范围内 |
| 引用覆盖率 | — | 0.952 | 20/21 条预期证据被覆盖 |
| 图片观察字段准确性 | — | 0.886 | 31/35 个视觉字段匹配 |
| 来源标注正确率 | — | 0.867 | 视觉、用户、知识库等来源大部分正确 |

完整 Python 回归为 `2514 passed`，失败数和收集错误数均为 0。

## 2. Week 6 在整体 AI 应用架构中的位置

前五周解决的是“系统拥有哪些能力”：

- Week 1：FastAPI、数据契约、异常处理、LLM 调用和流式接口。
- Week 2：文档加载、切分、Embedding、ChromaDB、RAG 与引用评测。
- Week 3：混合检索、查询改写、重排、证据门控和结构化诊断。
- Week 4：Planner、Runner、Executor、工具注册表和只读 Agent 工具。
- Week 5：图片输入、Vision、OCR、规则解析和多模态 Agent 评测。

Week 6 解决的是“怎样让这些能力可靠地协同工作并被交付”：

```text
用户请求
  -> 请求级安全与工具范围
  -> 结构化进度状态
  -> Planner 选择下一步
  -> Runner 校验并推进循环
  -> Executor 执行白名单工具
  -> 证据约束的诊断结果
  -> 脱敏会话记录
  -> API / Web 控制台展示
  -> Docker 可复现运行
```

这种设计把生成式模型擅长的“根据上下文选择步骤和组织内容”与 Python 擅长的“权限、状态、数据校验、停止条件和持久化”分开。LLM 仍然位于 Planner 层，Runner、Executor、工具策略、进度 Reducer 和会话存储都不依赖模型自由发挥。

## 3. 完整在线调用链

```text
客户端或 Web 控制台（提交文本和可选图片）
  -> FastAPI Router（校验 HTTP 请求契约并取得 request_id）
  -> AgentDiagnosisService（组织一次诊断请求）
  -> AgentRequestSafetyClassifier（确定性识别高风险控制或提示注入意图）
  -> AgentToolPolicy（根据任务目标计算最小工具范围）
  -> AgentProgressReducer（创建初始进度并给出首轮允许工具）
  -> AgentRunner（执行受限的规划、工具、观察循环）
  -> OpenAICompatibleAgentPlanner（LLM 只产生结构化下一步决定）
  -> ToolExecutor（校验工具名、参数、权限、超时和返回契约）
  -> 只读工具（Vision、知识库、模拟遥测、测试草案或当前时间）
  -> AgentProgressReducer（合并工具结果并缩小下一步工具范围）
  -> AgentDiagnosisReportBuilder（校验最终草稿和引用白名单）
  -> DiagnosticSessionBuilder（从公开结果生成脱敏会话记录）
  -> DiagnosticSessionStore（以 JSON 原子写入本地会话目录）
  -> AgentDiagnosisResponse（返回诊断、来源、引用和脱敏轨迹）
```

安全分类和工具策略在 Planner 之前执行，所以模型不能先看到全部工具再自行决定是否遵守限制。进度 Reducer 在每次工具返回后执行，所以模型也不能通过重复规划绕过“已完成能力”“无新证据”或“当前不允许调用”的约束。

## 4. 任务 1：请求级最小工具策略

### 4.1 数据契约

`app/schemas/agent_tool_policy.py` 定义 `AgentToolPolicyDecision`。它不是工具执行结果，而是一次请求进入 Agent Loop 之前的权限决定，主要字段包括：

- `policy_version`：策略版本，当前为 `agent-tool-policy-v1`。
- `disposition`：继续交给 Planner，还是直接拒答或要求人工审核。
- `required_capabilities`：本次任务真正需要完成的能力，例如视觉观察、知识库证据、遥测读取或测试草案。
- `allowed_tool_names`：本次请求允许暴露给 Planner 和 Executor 的最小工具名集合。
- `reason_codes`：机器可读取的决策依据，便于测试和审计。
- `public_message`：策略提前终止时可公开给用户的说明。
- `missing_information`：当前缺少哪些输入、证据或人工条件。

调用链：

```text
AgentDiagnosisService（准备诊断请求）
  -> AgentToolPolicyDecision（本节当前模块：约束策略输出的数据形状）
  -> AgentRunner（读取 allowed_tool_names 和 required_capabilities）
```

### 4.2 策略实现

`app/agent/request_tool_policy.py` 中的 `AgentToolPolicy.decide()` 将请求中的故障现象、脱敏日志、任务目标和图片情况做确定性归一化，然后识别本次任务需要哪些能力，再把能力映射到工具：

- `vision_observation` -> `analyze_robot_image`
- `knowledge_evidence` -> `search_knowledge`
- `telemetry_observation` -> `get_robot_telemetry`
- `test_case_draft` -> `draft_test_case`
- `current_time` -> `get_current_time`

调用链：

```text
AgentDiagnosisService（取得已校验请求）
  -> AgentToolPolicy.decide()（本节当前模块：计算最小能力和工具集合）
  -> AgentRunner.run()（只接收本次允许的工具名）
  -> ToolExecutor（执行时再次校验工具权限）
```

这里同时在 Planner 和 Executor 两层实施约束。Planner 看不到不相关工具可以减少误选；Executor 再次校验可以防止无效 Planner 输出或程序错误绕过权限边界。

## 5. 任务 2：确定性安全分类

`app/agent/request_safety_classifier.py` 中的 `AgentRequestSafetyClassifier` 在调用 Planner 前识别以下类型的请求：

- 要求执行或绕过急停、安全联锁和保护装置。
- 要求远程控制真实设备、下发任务或修改设备状态。
- 要求访问图片、二维码或日志中出现的外部链接。
- 要求执行图片或不可信文本中的命令、Prompt 或脚本。
- 其他必须由具备权限和资质人员确认的高风险控制意图。

普通诊断请求返回 `None`，表示安全分类器不提前拦截，后续仍要经过工具策略和 Agent Loop。高风险请求返回 `human_review_required` 决策，不向 Planner 暴露工具，并提供稳定的公开说明和缺失人工条件。

调用链：

```text
AgentDiagnosisService（读取用户意图）
  -> AgentRequestSafetyClassifier.classify()（本节当前模块：确定性识别高风险意图）
  -> 高风险：直接构造 human_review_required 响应
  -> 普通请求：继续交给 AgentToolPolicy
```

安全分类使用 Python 规则而不是只写 Prompt，是因为 Prompt 属于软约束，可能被模型误解或被不可信图片文字干扰；代码分支才能稳定决定“不调用哪些工具”和“在哪一层结束”。

## 6. 任务 3：证据驱动的 Agent 进度状态

### 6.1 `AgentProgress` 记录什么

`app/schemas/agent_progress.py` 定义三类核心对象：

- `AgentToolProgressRecord`：一次工具调用为哪个能力服务、状态如何、产生了哪些新来源或证据。
- `AgentEvidenceConflict`：记录相互冲突的来源和冲突说明，避免把矛盾证据合并成确定结论。
- `AgentProgress`：保存当前状态、必需能力、已完成能力、工具记录、已确认来源、已覆盖证据、冲突、缺失信息、下一步允许工具和停止原因。

`AgentProgress` 是 Agent Loop 的业务状态，不是给 LLM 的自由文本记忆。它回答三个问题：已经完成了什么、还缺什么、下一步最多允许做什么。

调用链：

```text
AgentToolPolicy（产生 required_capabilities 和 allowed_tool_names）
  -> AgentProgress（本节当前模块：保存可验证的任务进度）
  -> AgentRunner（每轮把当前进度用于限制工具和判断结束）
  -> AgentDiagnosisService（把最终状态转换成公开诊断语义）
```

### 6.2 Reducer 为什么存在

`app/agent/progress_reducer.py` 中的 `AgentProgressReducer` 接收“旧进度 + 新事件”，返回一份新的不可变进度对象。Reducer 负责：

- 验证 Planner 请求的工具当前是否允许。
- 生成规范化工具调用签名，阻止完全重复调用。
- 从成功工具结果中提取能力完成情况、来源 ID 和证据 ID。
- 记录空结果、失败、冲突和缺失信息。
- 在能力完成后移除不再需要的工具。
- 判断已经满足完成条件、只能返回部分结果，还是必须拒答或人工审核。

调用链：

```text
AgentRunner（收到 Planner 决定或工具结果）
  -> AgentProgressReducer（本节当前模块：验证转换并生成下一状态）
  -> AgentRunner（使用 allowed_next_tool_names 继续或停止）
```

Reducer 的价值是让相同输入和工具结果得到相同状态转换。Planner 可以提出建议，但不能自己声明某项证据已经取得、某项安全要求已经满足或某个失败可以忽略。

## 7. 任务 4：30 条场景离线回归

### 7.1 Fixture 的作用

`data/eval/multimodal_offline_fixtures.jsonl` 为 30 条 Gold 场景提供固定的 Planner 决定、Vision 观察和工具结果。Fixture 不是伪造一次线上准确率，而是把外部概率性服务替换成可重复输入，从而单独验证系统框架。

离线回归不会访问真实 LLM、Vision、Embedding、ChromaDB、遥测平台或设备。它验证的是：

- 请求级工具范围是否正确。
- Planner 决定是否会被 Runner 和 Executor 正确执行或拦截。
- 进度状态是否按工具结果推进。
- 终止原因、诊断状态和缺失信息是否一致。
- 视觉观察、引用和来源标签是否正确传递。
- 安全场景是否保持拒答或人工审核。

### 7.2 加载、配对和执行

`app/services/multimodal_offline_fixture_loader.py` 加载 Fixture，并按 `scenario_id` 与 Gold 场景一一配对；重复、缺失或多余 ID 都会导致校验失败。

调用链：

```text
评测脚本（读取 Gold 与 Fixture 路径）
  -> pair_multimodal_offline_regression_cases()（本节当前模块：一一配对并校验完整性）
  -> MultimodalOfflineScenarioExecutor（执行单条确定性场景）
```

`app/services/multimodal_offline_scenario_executor.py` 为单个场景创建 Fake Planner、Fake Vision 和受控 Fake 工具，然后调用真实的策略、Runner、Executor、报告构造与评分链路。

调用链：

```text
MultimodalOfflineRegressionService（遍历已配对场景）
  -> MultimodalOfflineScenarioExecutor.execute()（本节当前模块：注入 Fixture 并运行真实框架）
  -> AgentDiagnosisService / AgentRunner / ToolExecutor（真实编排代码）
  -> 单场景响应、轨迹和 Fixture 消费审计
```

`app/services/multimodal_offline_regression_service.py` 批量执行 30 条场景，交给既有评分器计算指标，再生成阈值摘要、失败场景和 30 份脱敏轨迹。

调用链：

```text
scripts/evaluate_multimodal_offline_regression.py（命令行入口）
  -> MultimodalOfflineRegressionService.run()（本节当前模块：批量执行、评分和汇总）
  -> JSON / Markdown 报告与脱敏轨迹
```

### 7.3 正式结果与剩余失败

正式报告为 `docs/multimodal-offline-regression.md`。26 条场景严格通过，失败场景为：

- `multimodal-agent-008`
- `multimodal-agent-012`
- `multimodal-agent-019`
- `multimodal-agent-029`

四条失败都涉及预期 Vision 工具没有被允许或执行，进而造成视觉状态、来源标注、工具顺序或最终状态不符合 Gold；其中 008 和 029 还涉及有资质人员审核要求。它们是明确保留的已知缺口，不影响本周 24/30 门槛已经达成的结论。

离线 0.867 不能等同于真实模型线上准确率。它证明同一组外部决定和工具结果进入系统后，框架行为稳定；真实 Planner、Vision 和上游 API 的概率性、延迟、限流与输出波动仍需单独抽样验证。

## 8. 任务 5：诊断会话与运行轨迹

### 8.1 会话契约

`app/schemas/diagnostic_session.py` 定义：

- `DiagnosticSessionRequestSummary`：脱敏请求摘要。
- `DiagnosticSessionDiagnosisSnapshot`：诊断状态、风险和缺失信息快照。
- `DiagnosticSessionEvidenceCoverage`：引用数量、来源和证据覆盖。
- `DiagnosticSessionMetrics`：总耗时、工具步骤数、失败工具等运行指标。
- `DiagnosticSessionRecord`：一条可持久化的完整脱敏会话。
- `DiagnosticSessionSummary`：最近会话列表中的紧凑摘要。
- `DiagnosticSessionListResponse`：分页或限量列表响应。

完整图片、API Key、完整敏感日志、完整工具参数和模型私有思维链不属于这些契约，因此不会因对象序列化而被意外写入会话文件。

### 8.2 构造与存储

`app/services/diagnostic_session_builder.py` 中的 `DiagnosticSessionBuilder` 从已经通过公开响应契约校验的诊断结果构造会话记录。它压缩用户输入、计算证据覆盖和工具指标，并把业务状态映射为稳定的会话状态。

调用链：

```text
AgentDiagnosisService（取得最终公开响应和执行轨迹）
  -> DiagnosticSessionBuilder.build()（本节当前模块：生成脱敏会话记录）
  -> DiagnosticSessionStore.save()（持久化）
```

`app/services/diagnostic_session_store.py` 中的 `DiagnosticSessionStore` 负责创建存储目录、校验 `session_id`、原子写入 JSON、按 ID 读取和列出最近会话。原子写入先写临时文件再替换目标文件，避免进程中断留下半份 JSON。

调用链：

```text
DiagnosticSessionBuilder（产生已校验记录）
  -> DiagnosticSessionStore（本节当前模块：保存、按 ID 读取、列出最近会话）
  -> Diagnostic Sessions Router / Web 控制台（只读取公开字段）
```

### 8.3 查询 API

`app/routers/diagnostic_sessions.py` 提供：

- `GET /api/v1/diagnostic-sessions`：查询最近会话摘要。
- `GET /api/v1/diagnostic-sessions/{session_id}`：查询指定会话详情。

调用链：

```text
客户端或 Web 控制台
  -> Diagnostic Sessions Router（本节当前模块：校验参数并调用存储层）
  -> DiagnosticSessionStore
  -> DiagnosticSessionListResponse 或 DiagnosticSessionRecord
```

不存在的会话返回稳定的 404 业务错误，不会泄露宿主机路径。

## 9. 任务 6：Web 控制台与 Docker

### 9.1 轻量控制台

`app/web/index.html`、`app/web/styles.css` 和 `app/web/app.js` 构成不依赖前端构建工具的轻量控制台。页面支持：

- 输入机器人编号、故障现象、脱敏日志和可选任务目标。
- 选择并提交最多三张图片。
- 展示诊断状态、可能原因、下一步检查、风险和缺失信息。
- 展示知识库引用、视觉观察、模拟遥测、测试草案和工具轨迹。
- 查询并打开最近保存的脱敏诊断会话。

调用链：

```text
浏览器中的 app.js（本节当前模块：收集输入并调用公开 API）
  -> POST /api/v1/agent/diagnose
  -> GET /api/v1/diagnostic-sessions
  -> 页面组件（按来源展示结果与轨迹）
```

控制台是教学演示与排错入口，不包含身份认证、租户隔离或设备控制按钮。

### 9.2 容器配置

- `Dockerfile` 使用 Python 3.11 slim 基础镜像，安装应用依赖、Tesseract 与中英文语言数据，并以非 root 的 `appuser` 运行 Uvicorn。
- `compose.yaml` 暴露 `8000` 端口，配置健康检查，并将宿主机 Chroma 数据和脱敏会话目录绑定挂载到容器。
- `.dockerignore` 排除 `.env`、测试、文档、原始语料、缓存、Chroma 数据和本地会话，减少构建上下文并避免敏感文件被复制进镜像。

调用链：

```text
docker compose up --detach --build
  -> Dockerfile（本节当前模块：构建可运行应用镜像）
  -> compose.yaml（启动服务、端口、健康检查和持久化挂载）
  -> Uvicorn / FastAPI
  -> /health、/console/ 和 /api/v1/*
```

本地容器验证结果：

- 镜像：`robotops-copilot:week6`。
- Compose 服务健康，`8000:8000` 端口映射正常。
- `/health` 和 `/console/` 返回 200。
- 容器能够读取 `robot_knowledge_week3_vector_baseline` Collection 的 355 个 Chunk。
- 带图片诊断请求能够完成 Vision 工具调用并生成可追踪会话。
- 会话通过绑定挂载保存到宿主机目录，并能在控制台最近会话中查看。

演示截图见 `docs/robotops-copilot-docker-demo.png`。

## 10. 测试设计与最终结果

Week 6 测试不是只验证“对象可以导入”，而是覆盖以下预期流程：

| 测试范围 | 测试方法 | 被测试模块的预期流程 | 预期结果 |
|---|---|---|---|
| 工具策略契约 | 构造合法和非法 Pydantic 对象 | 输入策略字段 -> 交叉字段校验 -> 冻结模型 | 合法对象稳定序列化，矛盾字段被拒绝 |
| 请求级工具策略 | 构造图片、知识、遥测、草案和否定表达请求 | 归一化请求 -> 识别能力 -> 映射最小工具 | 只暴露完成任务所需工具 |
| 安全分类器 | 参数化高风险、提示注入和普通诊断文本 | 识别未被否定的危险意图 -> 生成终止决定 | 高风险进入人工审核，普通诊断继续 |
| Agent 进度契约 | 构造不同状态和证据集合 | 校验状态、记录、来源、冲突和工具范围 | 不一致进度不能进入 Runner |
| Progress Reducer | 输入初始进度和连续工具结果 | 校验工具 -> 合并新信息 -> 重算能力和下一步 | 重复或越权工具被阻止，进度确定性推进 |
| 离线 Fixture | 加载 30 条 JSONL | Schema 校验 -> ID 去重 -> 与 Gold 一一配对 | 缺失、重复或多余 Fixture 被拒绝 |
| 离线单场景执行 | 注入 Fake Planner、Fake Vision 和 Fake 工具 | 运行真实策略、Runner、Executor 和报告链 | 不访问外部服务也能复现完整轨迹 |
| 离线批量回归 | 执行 30 条配对场景 | 单场景执行 -> 评分 -> 阈值 -> 报告 | 至少 24/30 且安全拒答率不下降 |
| 会话 Builder | 输入完成、部分、拒答和失败响应 | 压缩公开字段 -> 计算指标和覆盖 -> 构造记录 | 不写入图片、密钥、完整日志或思维链 |
| 会话 Store | 使用 pytest 临时目录 | 原子保存 -> 按 ID 获取 -> 最近列表 | 数据可恢复，非法 ID 和缺失会话稳定失败 |
| 会话 API | FastAPI 依赖覆盖和异步客户端 | Router -> Store -> Schema 响应 | 列表与详情返回正确契约，缺失返回 404 |
| Web 控制台 | TestClient 请求页面和静态资源 | FastAPI 静态挂载 -> HTML/CSS/JS | `/console/` 可访问，未知资源返回 404 |
| Docker 配置 | 构建、Compose 启动、健康与端到端请求 | 镜像 -> 容器 -> API -> 会话挂载 | 干净运行环境可复现主链路 |

最终 JUnit 报告为 `docs/pytest-results-week6-final.xml`：

```text
tests    : 2514
failures : 0
errors   : 0
skipped  : 0
```

测试运行期间发现并修正了一个 Week 6 依赖装配测试：生产函数 `get_agent_diagnosis_service()` 已新增 `session_builder` 和 `session_store`，旧测试仍按 Week 5 参数调用。修改位置是 `tests/test_agent_tool_dependencies.py`，测试现在用 `tmp_path` 创建隔离的 `DiagnosticSessionStore`，把两个新依赖传入工厂，并断言服务保留同一对象。该修正确保测试覆盖当前真实装配流程，而不是降低生产函数要求。

## 11. 上传 GitHub 的交付边界

应提交：

- `app/`、`scripts/`、`tests/` 中的源码与测试。
- `data/eval/` 中的 Gold 和离线 Fixture。
- `data/multimodal/` 中的公开教学图片。
- `samples/corpus/` 中的公开模拟语料。
- `README.md`、`.env.example`、`.gitignore`、`.dockerignore`、`Dockerfile` 和 `compose.yaml`。
- `docs/工程总结_week6.md`、离线回归 Markdown、脱敏会话样例、Docker 演示截图和 Week 6 最终 JUnit 报告。

不得提交：

- `.env` 和任何真实密钥、证书或凭据。
- `要求.txt`、`week6.txt` 和验收反馈。
- `chroma_data/`、`data/diagnostic_sessions/` 和其他本地运行状态。
- `data/source/` 中的原始 PDF 或可能受授权限制的资料。
- `__pycache__/`、`.pytest_cache/`、临时目录、日志和编辑器配置。
- 未脱敏的 JSON 轨迹或包含完整用户日志、图片 Base64、宿主机隐私信息的报告。

## 12. 已知限制

- 0.867 是固定 Fixture 下的系统框架回归率，不是对真实 LLM 或 Vision 准确率的承诺。
- 四条离线失败场景仍存在 Vision 工具范围、状态映射或人工审核语义不一致。
- Planner 和 Vision Provider 仍依赖外部服务，可能受到输出波动、超时、限流和兼容性影响。
- 当前会话存储是本地 JSON 文件，适合单机教学演示，不提供数据库事务、多实例并发或长期归档能力。
- Web 控制台没有身份认证、权限控制、租户隔离、CSRF 防护或生产级前端构建流程。
- Compose 是本地可复现配置，不等同于生产部署；生产环境仍需要密钥管理、TLS、反向代理、监控、备份和资源限制。
- 当前工具均为只读或草案生成工具，系统不会控制机器人、绕过安全装置或自动执行维修与复位。
- 诊断输出不能替代设备原厂资料、现场风险评估和具备资质人员的确认。

## 13. 最终结论

Week 6 已完成计划中的请求级最小工具策略、确定性安全分类、证据驱动进度状态、30 场景离线回归、脱敏诊断会话、轻量控制台与 Docker 启动。系统从“依赖 Planner 临场决定所有步骤”演进为“Planner 在 Python 权限和状态机内做有限决策”，并且具备了可重复回归、失败定位、会话追踪和干净环境演示能力。

它已经达到本周 RobotOps Copilot Beta 的学习与演示目标，但仍是只读的教学工程，不是生产机器人控制系统或安全认证产品。
