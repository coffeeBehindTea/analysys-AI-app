# Robot Diagnostic Agent 失败案例与恢复策略

## 1. 文档目的

本文档记录 Week 4 Robot Diagnostic Agent 已经实现的失败边界、恢复规则、稳定终止原因和自动化测试位置。

本文档用于回答以下问题：

1. 失败发生在 Planner、Runner、Executor 还是工具 Handler；
2. 失败发生后工具是否已经执行；
3. 已经完成的工具步骤是否保留；
4. Planner 是否可以调整方案后继续；
5. 什么情况必须立即中止；
6. 程序应判断哪个结构化状态或错误码；
7. 哪个自动化测试验证了该行为。

本文档不记录生成时间，避免仅因时间变化产生无意义的版本差异。

## 2. 失败控制在调用链中的位置

Agent 失败控制调用链如下：

1. `AgentDiagnosisService`
   - 将用户请求整理成 Agent 任务；
   - 调用 AgentRunner；
   - 后续根据 Runner结果生成完整、部分或拒答诊断。

2. `AgentRunner`（当前失败控制模块）
   - 创建每轮规划上下文；
   - 限制 Planner超时；
   - 校验 Planner返回值；
   - 检查最大步骤和重复调用；
   - 将合法工具请求交给 ToolExecutor；
   - 保存工具交互；
   - 根据失败策略继续或中止。

3. `AgentPlanner`
   - 只能返回结构化 `call_tool` 或 `finish` 决定；
   - 不能直接执行 Python函数；
   - 不能修改工具白名单和安全配置。

4. `ToolRegistry`
   - 保存应用启动时注册的只读工具白名单；
   - 不根据模型返回的字符串动态导入函数；
   - 拒绝可写工具和高风险工具。

5. `ToolExecutor`
   - 精确匹配工具名；
   - 使用具体 Pydantic输入模型校验参数；
   - 再次检查只读权限和风险级别；
   - 执行单工具超时；
   - 校验工具输出；
   - 返回统一 `ToolExecutionResult`。

6. `failure_policy.py`（当前工具结果分类模块）
   - 将工具结果分类为成功、可恢复失败或致命权限失败；
   - 不解析自然语言错误消息；
   - 不执行工具，也不直接改变 Agent状态。

7. `AgentRunResult`
   - 保存终止原因；
   - 保存已经完成的工具交互；
   - 保存仍未解决的信息；
   - 禁止安全中止结果携带 Planner最终草稿。

失败控制由 Python结构化规则决定。用户输入、文档正文、遥测内容、工具文本和 LLM自然语言不能直接改变这些规则。

## 3. Agent执行状态与诊断状态

### 3.1 Agent执行状态

AgentRunner当前使用两个终态：

| 状态 | 含义 |
|---|---|
| `completed` | Planner按照结构化协议主动结束循环 |
| `aborted` | Python因为安全限制、超时、异常或无进展条件中止循环 |

`completed` 只表示循环正常结束，不表示诊断一定完整或正确。

### 3.2 诊断内容状态

后续 AgentDiagnosisService仍需复用 Week 3 的诊断状态：

| 诊断状态 | 含义 |
|---|---|
| `completed` | 证据足以支持完整诊断 |
| `partial` | 只有部分结论得到证据支持 |
| `abstained` | 没有足够证据形成安全诊断 |

可能的组合包括：

| Runner状态 | 诊断状态 | 说明 |
|---|---|---|
| `completed` | `completed` | Agent正常结束并取得完整证据 |
| `completed` | `partial` | Agent正常认识到信息仍不完整 |
| `completed` | `abstained` | Agent正常确认无法回答 |
| `aborted` | `partial` | Agent中止前已经取得部分可信证据 |
| `aborted` | `abstained` | Agent中止且没有足够可信证据 |

Runner不会为了把 `aborted` 变成 `completed` 而降低证据门槛。

## 4. 稳定终止原因

`AgentRunTerminationReason` 当前支持：

| 终止原因 | Runner状态 | 含义 |
|---|---|---|
| `planner_finished` | `completed` | Planner主动返回合法 finish决定 |
| `max_steps_reached` | `aborted` | 已达到工具总步骤上限，Planner仍要求调用工具 |
| `duplicate_tool_call` | `aborted` | Planner重复使用调用ID，或重复相同工具和参数 |
| `planner_timeout` | `aborted` | 单轮 Planner没有在规定时间内返回 |
| `planner_error` | `aborted` | Planner调用发生普通非超时异常 |
| `invalid_planner_response` | `aborted` | Planner返回类型、结构或工具参数不合法 |
| `tool_failure_limit_reached` | `aborted` | 可恢复工具失败连续达到配置上限 |
| `tool_policy_violation` | `aborted` | 工具请求触碰只读或风险权限边界 |

调用方必须判断 `termination_reason`，不能通过匹配 `termination_message` 的中文文本判断程序流程。

## 5. 工具结果分类

`classify_tool_execution_result()` 将 `ToolExecutionResult` 分为三类。

| 分类 | 触发条件 | Runner行为 |
|---|---|---|
| `success` | `status="success"` | 连续失败计数清零，进入下一轮规划 |
| `recoverable_failure` | `empty`、普通 `rejected`、`timeout` 或 `error` | 失败计数加一；未达限制时允许 Planner调整 |
| `fatal_policy_violation` | `rejected/tool_blocked` | 保存交互后立即中止 |

分类函数只使用结构化 `status` 和 `error_code`，不解析 `public_message`。

## 6. 失败处理顺序

Runner按照以下顺序处理每一轮：

1. 构造经过深复制的规划 Context；
2. 在单轮超时限制内调用 Planner；
3. 校验 Planner是否返回 `AgentPlannerDecision`；
4. 如果 Planner返回 `finish`，正常结束；
5. 如果 Planner请求工具，检查是否达到 `max_steps`；
6. 生成不含 `call_id` 的工具调用签名；
7. 检查重复调用 ID和重复工具签名；
8. 将调用交给 ToolExecutor；
9. 将工具请求和结果保存成 `AgentToolInteraction`；
10. 使用失败策略分类工具结果；
11. 成功时清零连续失败次数；
12. 可恢复失败时累加失败次数；
13. 权限失败或连续失败达到限制时安全中止；
14. 其他情况进入下一轮规划。

失败策略必须在保存 Interaction之后执行。这样导致中止的最后一项工具请求和结果仍然可以审计。

## 7. Planner失败案例

### 7.1 Planner超时

触发条件：

- 单轮 `planner.plan()` 超过 `planner_timeout_seconds`。

处理流程：

1. `asyncio.timeout()` 取消当前 Planner等待；
2. Runner捕获 `TimeoutError`；
3. 不公开 Planner原始上游信息；
4. 返回 `planner_timeout`；
5. 保留超时前已经完成的工具交互。

稳定结果：

- `state="aborted"`
- `termination_reason="planner_timeout"`
- `finish_reason=None`
- `final_message=None`
- `missing_information` 说明 Planner没有返回下一步决定

对应测试：

- `test_planner_timeout_returns_stable_abort`
- `test_planner_timeout_preserves_completed_interaction`

### 7.2 Planner普通异常

触发条件：

- Planner抛出非超时普通异常。

处理流程：

1. Runner捕获 `Exception`；
2. 日志只记录异常类型和已完成步数；
3. 不记录 `str(exc)`、`repr(exc)`、用户任务或原始响应；
4. 返回 `planner_error`。

稳定结果：

- `state="aborted"`
- `termination_reason="planner_error"`
- 公开说明为稳定文本
- 已完成步骤保留
- 敏感异常原文不进入结果和警告日志

对应测试：

- `test_planner_exception_is_sanitized`

### 7.3 Planner返回错误类型

触发条件：

- Planner返回字符串、普通字典、SDK对象或其他非 `AgentPlannerDecision` 对象。

处理流程：

1. Runner使用 `isinstance()` 执行运行时检查；
2. 不信任 Protocol类型注解能够自动校验返回值；
3. 返回 `invalid_planner_response`。

对应测试：

- `test_runner_aborts_on_invalid_planner_return_type`

### 7.4 Planner返回非JSON工具参数

触发条件：

- `ToolCall.arguments` 字典内部含有 `set` 等不能序列化为 JSON 的值。

处理流程：

1. `json.dumps()` 无法生成稳定调用签名；
2. 工具不会交给 Executor；
3. 日志不记录原始参数；
4. 返回 `invalid_planner_response`。

对应测试：

- `test_canonical_signature_rejects_non_json_arguments`
- `test_non_json_tool_arguments_abort_before_execution`

## 8. Runner循环边界失败案例

### 8.1 达到最大总步骤

触发条件：

- 已经实际执行 `max_steps` 个工具；
- Planner仍请求新的工具调用。

处理流程：

1. 新工具不会执行；
2. 不产生伪造的 Interaction；
3. 保留此前实际完成的历史；
4. 返回 `max_steps_reached`。

对应测试：

- `test_max_steps_blocks_next_tool_before_execution`

### 8.2 重复调用ID

触发条件：

- Planner再次使用已经执行过的 `call_id`。

处理流程：

1. 在 Executor调用前阻止；
2. 第二次 Handler不会执行；
3. 返回 `duplicate_tool_call`。

对应测试：

- `test_duplicate_call_id_aborts_before_second_execution`

### 8.3 重复工具和参数

触发条件：

- Planner使用新的 `call_id`；
- 但工具名和规范化 JSON参数与此前调用完全相同。

处理流程：

1. `sort_keys=True` 消除字典键顺序差异；
2. 调用签名不包含 `call_id`；
3. 第二次调用不会执行；
4. 返回 `duplicate_tool_call`。

对应测试：

- `test_canonical_signature_ignores_call_id_and_key_order`
- `test_duplicate_signature_aborts_with_new_call_id`

## 9. ToolExecutor失败案例

### 9.1 未知工具

触发条件：

- `tool_name` 不存在于 ToolRegistry。

处理流程：

1. 不进行动态导入；
2. 不在注册表之外查找同名函数；
3. 不执行任何 Handler；
4. 返回 `rejected/unknown_tool`；
5. 将其视为一次可恢复失败；
6. Planner可以在失败预算内改用合法工具。

对应测试：

- `test_unknown_tool_returns_rejected_result`
- `test_unknown_tool_rejection_is_observed`

### 9.2 工具参数不合法

触发条件：

- 参数缺少必填字段；
- 参数类型不正确；
- 含有输入模型禁止的额外字段；
- 数值或字符串违反约束。

处理流程：

1. ToolExecutor使用具体工具的 `input_model.model_validate()`；
2. 参数校验失败时不调用 Handler；
3. 返回 `rejected/invalid_tool_arguments`；
4. 计为一次可恢复失败。

对应测试：

- `test_invalid_arguments_do_not_call_handler`

### 9.3 工具返回空结果

触发条件：

- Handler正常完成但返回 `None`。

处理流程：

1. Executor返回 `empty/empty_result`；
2. Runner记录该结果；
3. 第一次空结果允许 Planner调整查询或选择其他工具；
4. 连续空结果达到限制后中止。

对应测试：

- `test_none_output_becomes_empty_result`
- `test_empty_result_is_observed_before_finish`
- `test_two_consecutive_empty_results_abort`

### 9.4 工具超时

触发条件：

- Handler执行超过 `ToolDefinition.timeout_seconds`。

处理流程：

1. ToolExecutor取消当前 Handler；
2. 返回 `timeout/tool_timeout`；
3. Runner记录超时结果；
4. 第一次超时允许 Planner选择替代方案或安全结束；
5. 连续失败达到限制后中止。

对应测试：

- `test_handler_timeout_returns_stable_result`
- `test_single_tool_timeout_can_be_observed_and_finished`

### 9.5 工具普通异常

触发条件：

- Handler抛出普通异常。

处理流程：

1. Executor捕获异常；
2. 不把底层异常原文放入公开结果；
3. 返回 `error/tool_execution_error`；
4. 计为一次可恢复失败。

对应测试：

- `test_handler_exception_is_sanitized`

### 9.6 工具输出不符合契约

触发条件：

- Handler返回错误 Pydantic模型；
- 返回字典无法通过 `output_model` 校验；
- 输出字段类型或约束不正确。

处理流程：

1. Executor不把错误输出当成可信数据；
2. 返回 `error/invalid_tool_output`；
3. `output` 必须为 `None`；
4. 计为一次可恢复失败。

对应测试：

- `test_invalid_handler_output_returns_error`
- `test_runtime_validation_accepts_valid_dictionary_output`

## 10. 连续失败恢复策略

默认：

- `max_consecutive_tool_failures=2`

### 10.1 第一次可恢复失败

流程：

1. 保存失败 Interaction；
2. 连续失败次数变为1；
3. 进入下一轮 Planner；
4. Planner可以调整参数、改写查询、选择其他注册工具或安全结束。

### 10.2 第二次连续失败

流程：

1. 保存第二条失败 Interaction；
2. 连续失败次数变为2；
3. 达到默认上限；
4. 不再调用 Planner；
5. 返回 `tool_failure_limit_reached`。

对应测试：

- `test_two_consecutive_empty_results_abort`

### 10.3 成功结果重置计数

示例：

- 空结果：失败次数1；
- 成功结果：失败次数清零；
- 再次空结果：失败次数重新为1。

对应测试：

- `test_success_resets_consecutive_failure_counter`

### 10.4 总步骤限制与失败限制的区别

`max_steps` 限制实际工具调用总数。

`max_consecutive_tool_failures` 限制连续非成功工具结果数量。

两者独立配置，先满足哪个终止条件，就使用哪个稳定终止原因。

## 11. 权限失败与纵深防御

### 11.1 工具注册阶段

ToolRegistry拒绝：

- `read_only=False`；
- `risk_level="high"`；
- 同名重复工具；
- 同步 Handler。

### 11.2 工具执行阶段

ToolExecutor再次检查：

- 工具是否只读；
- 是否为 high风险。

即使注册表内部状态被错误污染，Executor仍返回：

- `rejected/tool_blocked`

### 11.3 Runner处理

`failure_policy.py` 将 `tool_blocked` 分类为：

`fatal_policy_violation`

Runner：

1. 保存导致中止的工具 Interaction；
2. 不调用真实 Handler；
3. 立即返回 `tool_policy_violation`；
4. 不允许 Planner通过修改参数继续试探。

对应测试：

- `test_executor_blocks_unsafe_definition_defensively`
- `test_tool_blocked_aborts_immediately`

## 12. 提示注入

以下内容全部作为不可信数据处理：

- 用户任务；
- 日志摘录；
- 文档正文；
- 模拟遥测；
- 工具输出文本。

即使其中出现：

- “忽略系统规则”；
- “调用 run_shell”；
- “执行以下命令”；
- “修改工具白名单”；
- “伪造证据ID”；

也不能直接改变 Python控制规则。

当前工具边界保证：

1. Planner只能看到 ToolRegistry生成的工具 Schema；
2. ToolExecutor只精确执行注册工具；
3. 未知工具不会动态导入；
4. 用户文本不会成为 Python函数名；
5. 文档中的命令不会作为程序执行。

对应测试：

- `test_prompt_injection_cannot_expand_tool_allowlist`

该测试覆盖工具边界。接入真实 LLM Planner后，还需要增加 System Prompt和真实 Provider结构输出的提示注入回归测试。

## 13. 外部取消

触发条件：

- FastAPI请求被取消；
- 服务关闭；
- 上层 Task主动调用 `.cancel()`。

处理流程：

1. Planner或工具等待收到 `asyncio.CancelledError`；
2. Runner和 Executor继续抛出该异常；
3. 不转换成 `planner_error` 或 `tool_execution_error`；
4. 上层调用方负责完成取消流程。

外部取消不会返回普通 `AgentRunResult`，因为它不是 Agent业务失败。

对应测试：

- `test_external_cancellation_propagates_from_runner`
- `test_external_cancellation_propagates`

## 14. 配置错误

以下情况属于开发者或调用方违反 Python接口契约，不转换成 `aborted`：

- Planner没有异步 `plan()`；
- Executor不是 `ToolExecutor`；
- `max_steps` 类型或范围错误；
- `planner_timeout_seconds` 类型或范围错误；
- `max_consecutive_tool_failures` 类型或范围错误；
- `task` 不是字符串；
- `task` 是空白文本。

这些问题应尽早抛出 `TypeError`、`ValueError` 或 Pydantic `ValidationError`，避免隐藏程序配置错误。

对应测试：

- `test_runner_requires_async_planner`
- `test_runner_requires_tool_executor`
- `test_runner_rejects_non_integer_max_steps`
- `test_runner_rejects_out_of_range_max_steps`
- `test_runner_rejects_non_numeric_planner_timeout`
- `test_runner_rejects_out_of_range_planner_timeout`
- `test_runner_rejects_non_integer_failure_limit`
- `test_runner_rejects_out_of_range_failure_limit`
- `test_runner_requires_string_task`
- `test_runner_rejects_blank_task_before_planning`

## 15. 日志与敏感信息

当前失败日志只记录必要的结构信息：

- 稳定事件名；
- 异常类型；
- 工具名；
- `status`；
- `error_code`；
- 已完成步骤数；
- 连续失败次数。

日志不记录：

- API Key；
- 用户完整任务；
- 完整日志摘录；
- 工具参数正文；
- 工具输出正文；
- Planner原始响应；
- 异常消息原文；
- 模型私有思维链。

公开错误消息与服务端日志都不能依赖泄漏底层异常来完成诊断。

## 16. 已完成步骤的保留规则

`interactions` 只保存实际完成的工具处理：

- 成功结果会保存；
- 空结果会保存；
- 参数拒绝会保存；
- 未知工具拒绝会保存；
- 工具超时会保存；
- 工具异常会保存；
- 权限拒绝会保存。

以下请求不会产生新的 Interaction：

- 达到 `max_steps` 后提出的新调用；
- 重复 `call_id`；
- 重复工具名和参数；
- 非JSON工具参数；
- Planner返回错误类型；
- Planner在提出工具前超时。

上层 Service不能把未执行请求描述成已完成步骤。

## 17. 自动化验证命令

结果契约、失败分类和 Runner目标测试：

`python -m pytest tests/test_agent_runtime_schemas.py tests/test_agent_failure_policy.py tests/test_agent_runner.py -v`

ToolExecutor失败测试：

`python -m pytest tests/test_agent_tool_executor.py -v`

Agent相关回归：

`python -m pytest tests -k "agent or tool or telemetry" -q`

完整回归：

`python -m pytest -q`

## 18. 当前已知限制

当前任务4已经完成 Runner和工具执行层的失败恢复，但仍有以下后续工作：

1. 正式 Planner目前只有异步 Protocol和固定 Fake轨迹，尚未接入真实 OpenAI兼容 Tool Calling Provider；
2. 尚未创建 `POST /api/v1/agent/diagnose`；
3. AgentDiagnosisService尚未将中止历史转换成最终 `partial` 或 `abstained` 诊断；
4. 尚未生成面向 API响应的脱敏执行轨迹；
5. `final_message` 仍是未经最终诊断 Schema和证据白名单校验的 Planner草稿；
6. 真实 Planner接入后仍需补充 Provider层提示注入和非法工具调用测试；
7. 当前 Agent不执行任何真实机器人控制、shell、任意 Python或任意网络工具。

这些限制不能通过降低工具权限、增加无限重试或绕过证据门控来解决。