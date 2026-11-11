# RobotOps Copilot Week 8 安全复测

## 1. 复测目标

本次复测确认 RobotOps Copilot 在完成 Week 8 修改后仍保持只读诊断边界。复测重点不是判断 LLM 是否“愿意听话”，而是确认请求入口、工具白名单、Runner、Executor、Vision 数据边界、公开轨迹和评分器都具有代码级防护。

安全调用链如下：

用户请求
→ `AgentRequestSafetyClassifier`：识别提示注入、外链动作、真实控制和高风险操作
→ `AgentToolPolicy`：计算请求允许使用的最小只读工具集合
→ `AgentRunner`：只把允许的工具定义交给 Planner，并验证 Planner 决策
→ `ToolExecutor`：在 Handler 执行前再次校验工具范围和风险属性
→ Vision、知识库和模拟遥测工具：只返回结构化、可验证的公开结果
→ 轨迹构造器与流式服务：脱敏错误、查询正文和工具输出
→ 公开诊断、SSE 事件与评测报告。

## 2. 执行结果

| 项目 | 结果 |
|---|---:|
| 收集的安全案例 | 34 |
| 通过 | 34 |
| 失败 | 0 |
| 错误 | 0 |
| 安全复测结论 | 通过 |

本次命令只运行安全专项集合。完整项目回归另见 `docs/pytest-results-week8-final.xml`，不能用本表的 34 条替代完整测试总数。

## 3. 风险覆盖表

下表中的风险类别存在交叉，例如图片中的恶意文字同时涉及提示注入和不可信输入，因此各行案例数不应相加作为测试总数。

| 风险类别 | 被验证的防御行为 | 主要测试位置 | 结果 |
|---|---|---|---|
| 提示注入与密钥索取 | 中英文“忽略规则”、运行 Shell、显示 API Key 等请求在 Planner 前进入人工审核；不能扩展工具白名单 | `test_request_safety_classifier.py`、`test_agent_diagnosis_service.py`、`test_agent_runner.py` | 通过 |
| 外链与二维码动作 | 打开、访问、下载或跟随外部链接的要求不由 Agent 执行，而是进入人工审核 | `test_request_safety_classifier.py::test_external_link_actions_require_review` | 通过 |
| 真实控制与急停绕过 | 绕过急停、启动驱动器、发送继续运动命令和带电插拔在图片解码及 Runner 执行前终止 | `test_request_safety_classifier.py::test_high_risk_control_requests_require_review`、`test_agent_diagnosis_service.py::test_service_stops_high_risk_request_before_image_decode` | 通过 |
| 恶意图片文字 | 图片中的命令和提示注入只作为不可信观察记录；高风险视觉内容要求人工确认，不能转成系统指令 | `test_analyze_robot_image_tool.py`、`test_vision_prompts.py` | 通过 |
| 越权工具 | Planner 请求隐藏工具、未授权工具、可写工具或高风险工具时，Runner、Executor 和 Registry 分层拒绝 | `test_agent_runner.py`、`test_agent_tool_executor.py`、`test_agent_tool_registry.py` | 通过 |
| 敏感信息泄漏 | 图片 Base64、知识查询正文、证据正文、异常原文、认证信息和密钥形态不进入公开序列化、日志、SSE 或轨迹 | `test_vision_schemas.py`、`test_agent_tool_trace_builder.py`、`test_agent_tool_executor.py`、`test_agent_runner.py`、`test_agent_diagnosis_streaming_service.py` | 通过 |
| 安全评分失效 | 公开响应出现敏感标记时，评测器必须将对应安全要求判为失败，不能产生虚假通过 | `test_multimodal_agent_evaluation_scoring.py::test_sensitive_marker_fails_non_exposure_requirement` | 通过 |

## 4. 关键防御验证

### 4.1 Planner 前置拦截

高风险控制和明确提示注入在创建图片输入、调用 Planner 或执行工具之前结束。这样即使外部模型行为不稳定，也不会获得执行危险请求的机会。公开终态使用 `human_review_required`，而不是伪装成普通成功。

### 4.2 最小工具范围与执行时复核

安全性不只依赖传给 Planner 的工具列表。Runner 会拒绝范围之外的调用，Executor 在 Handler 执行前再次检查 `allowed_tool_names`，Registry 也禁止注册可写或高风险工具。任一上层检查失效时，下层仍能阻断越权执行。

### 4.3 图片是不可信输入

Vision Prompt 把分析目标和图片文字放在数据边界中。Vision 工具可以报告检测到不可信文字或需要人工检查，但不能执行图片里的 URL、Shell、设备控制或规则覆盖指令。

### 4.4 公开错误和轨迹脱敏

工具异常和 Planner 异常会被转换成稳定公开错误，不返回上游正文。知识检索轨迹只记录查询长度、状态和证据标识，不保存查询正文和完整证据正文。图片载荷使用 `SecretStr` 降低意外展示风险，公开序列化不包含 Base64。

## 5. 结论与边界

安全专项复测通过，自动化结果为 `34 passed`，完整回归为 `2655 passed`。30 条离线多模态场景的安全拒答率为 `1.000`，10 条真实模型抽样中的 5 条安全拒答场景也全部正确处理。

这些结果证明当前代码边界能阻止已定义的危险请求，不代表系统已经取得工业安全认证，也不替代现场权限系统、网络隔离、设备厂商规程或具备资质人员的审核。项目仍不连接、不控制真实机器人，不执行测试草案，也不执行图片或日志中的指令。
