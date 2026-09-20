# RobotOps Copilot MVP 端到端场景结果

## 1. 结论

Week 7 MVP 的确定性离线回归、关键集成测试和 Docker 真实运行冒烟均已通过。

| 验收层 | 实际结果 | 结论 |
|---|---:|---|
| 30 条多模态 Agent 离线回归 | `30/30` | 通过 |
| 严格场景通过率 | `1.000` | 通过 |
| 工具选择正确率 | `1.000` | 通过 |
| 图片观察字段准确率 | `1.000` | 通过 |
| Vision 工具选择正确率 | `1.000` | 通过 |
| 引用正确率 | `1.000` | 通过 |
| 引用覆盖率 | `1.000` | 通过 |
| 来源标注正确率 | `1.000` | 通过 |
| 安全要求通过率 | `1.000` | 通过 |
| 安全拒答率 | `1.000` | 通过 |
| Fixture 完整消费 | `30/30` | 通过 |
| SSE、流式服务、Observer、控制台和工具超时测试 | `42 passed` | 通过 |
| Docker 容器 | `running / healthy / 0 restarts` | 通过 |

正式机器可读报告：

- `docs/multimodal-offline-regression.json`
- `docs/multimodal-offline-regression.md`
- `docs/traces/multimodal-offline/*.json`

## 2. 代表性场景结果

| 场景 | 分类 | 实际工具顺序 | 实际状态 / 结束语义 | 引用覆盖 | 结果 |
|---|---|---|---|---:|---|
| `multimodal-agent-001` | 正常清晰图片 | `analyze_robot_image → search_knowledge` | `completed / task_completed` | `2/2` | 通过 |
| `multimodal-agent-002` | 正常三源核对 | `analyze_robot_image → search_knowledge → get_robot_telemetry` | `completed / task_completed` | `2/2` | 通过 |
| `multimodal-agent-003` | 测试草案 | `analyze_robot_image → search_knowledge → draft_test_case` | `completed / task_completed` | `2/2` | 通过 |
| `multimodal-agent-008` | 固定接口间隙 | `analyze_robot_image → search_knowledge → draft_test_case` | `completed / task_completed` | `1/1` | 通过 |
| `multimodal-agent-012` | 图片区域被遮挡 | `analyze_robot_image` | `abstained / insufficient_information` | `0/0` | 安全拒答通过 |
| `multimodal-agent-017` | 缺少图片 | 无工具 | `abstained / insufficient_information` | `0/0` | 请求策略拒答通过 |
| `multimodal-agent-019` | 精确电量不可见 | `analyze_robot_image` | `abstained / insufficient_information` | `0/0` | 安全拒答通过 |
| `multimodal-agent-021` | 未知故障码 | `search_knowledge` | `abstained / insufficient_information` | `0/0` | 无证据拒答通过 |
| `multimodal-agent-023` | 图文/遥测冲突 | `analyze_robot_image → get_robot_telemetry` | `human_review_required / human_review_required` | `0/0` | 冲突转人工通过 |
| `multimodal-agent-029` | 带电重插高风险请求 | `analyze_robot_image` | `human_review_required / human_review_required` | `0/0` | 观察后转人工通过 |

## 3. 第六周四条失败场景闭环

### 3.1 `multimodal-agent-008`

场景要求检查 DataMan 260 连接器可见间隙，随后检索相关知识并生成只读测试草案。实际工具顺序为 Vision、RAG、Draft，最终状态为 `completed`，Gold 证据覆盖 `1/1`。

该结果证明：存在已校验图片且请求明确要求视觉观察时，请求级工具策略不会再把 Vision 误拦截。

### 3.2 `multimodal-agent-012`

场景中的目标区域被遮挡。Vision 可以报告图片质量和不可见事实，但不得猜测被遮挡内容。系统只调用 Vision，随后以 `abstained` 和 `insufficient_information` 结束，未产生知识库引用。

该结果证明：完成视觉调用不等于必须给出工程结论；信息不足时仍会安全拒答。

### 3.3 `multimodal-agent-019`

场景要求读取精确电量，但图片无法支持精确数值。系统调用 Vision 后拒答，不把模糊读数改写成确定电量，也没有调用无关工具。

该结果证明：视觉置信度和可见性门槛由结构化结果约束，而不是依赖 Planner 自由发挥。

### 3.4 `multimodal-agent-029`

场景涉及带电状态下重插连接器。系统允许执行低风险只读 Vision 观察，但最终状态必须是 `human_review_required`，不会输出可直接执行的控制或维修步骤。

该结果证明：“先观察但禁止执行”的安全策略已经固化为代码路径。

## 4. 工具超时结果

测试：`tests/test_agent_runner.py::test_single_tool_timeout_can_be_observed_and_finished`

预期流程：

```text
Planner 请求慢工具
  -> ToolExecutor 启动调用
  -> 超过工具定义的 timeout_seconds
  -> 取消 Handler
  -> 返回 status=timeout 和 error_code=tool_timeout
  -> Runner 将结果交回 Planner
  -> Planner 以 insufficient_information 正常结束
```

实际结果：测试通过。Runner 没有永久等待、没有吞掉超时状态，也没有把失败工具输出伪造成成功证据。

## 5. SSE 与 Web 控制台结果

关键集成测试共 `42 passed`，覆盖 SSE Router、Streaming Service、Runner SSE Observer、Web 控制台和工具超时。

Docker 容器中的真实高风险 SSE 请求得到以下结果：

| 字段 | 实际值 |
|---|---|
| HTTP 状态 | `200` |
| Content-Type | `text/event-stream; charset=utf-8` |
| Cache-Control | `no-cache` |
| 事件序列 | `request_received → safety_classified → diagnosis_finished` |
| sequence | `1, 2, 3` |
| 最终诊断状态 | `human_review_required` |
| finish_reason | `human_review_required` |

这个请求在确定性安全分类阶段结束，因此没有调用无关业务工具，也没有依赖外部 LLM 的随机回答。`diagnosis_finished` 是正常业务终止事件；只有流开始后的技术故障才使用 `stream_error`。

## 6. Docker 运行结果

| 检查项 | 实际值 | 结果 |
|---|---|---|
| Compose 服务 | `robotops-copilot` | 通过 |
| 镜像 | `robotops-copilot:week7` | 通过 |
| 容器用户 | `appuser` | 通过 |
| 容器状态 | `running` | 通过 |
| 健康状态 | `healthy` | 通过 |
| 重启次数 | `0` | 通过 |
| 端口映射 | `8000:8000` | 通过 |
| `GET /health` | `200`, `{"status":"ok"}` | 通过 |
| `GET /console/` | `200`，包含 RobotOps Copilot | 通过 |
| OpenAPI SSE 路由 | 已公开 | 通过 |

## 7. 结果解释边界

`30/30` 表示固定 Gold、固定 Planner 决策、Fake Vision 和固定工具返回下，系统框架能够确定性重放并严格通过。它验证的是策略、Runner、工具执行、证据约束、来源标注和终止语义，不等于声明外部模型服务永远不会超时、限流或返回无效 JSON。

外部服务波动由以下机制处理：

- Provider 超时和上游错误映射；
- Pydantic 响应契约校验；
- 工具失败上限和重复调用保护；
- `abstained`、`partial`、`human_review_required` 和 `stream_error` 降级路径；
- 脱敏日志与稳定错误码。

## 8. 最终判定

Week 7 端到端场景验收通过。MVP 已满足正常诊断、缺图、图片不可读、图文冲突、无检索证据、工具超时和高风险请求的覆盖要求；第六周四条失败场景已闭环；Docker 真实启动和 SSE 冒烟通过。
