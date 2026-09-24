# RobotOps Copilot 简历与面试描述

## 1. 一行描述

设计并实现 RobotOps Copilot，只读多模态机器人诊断 Agent，融合工程文档 RAG、图片观察、脱敏模拟遥测、代码级安全策略、SSE 可观测性与 Docker 可复现交付。

## 2. 三条成果

- 构建 FastAPI + ChromaDB 的机器人知识与诊断系统，实现 PDF/Markdown/TXT 摄取、混合检索、RRF 融合、重排、证据门控和真实 Chunk 引用，支持知识问答、结构化诊断与多步 Agent。
- 设计 Planner、Runner、Executor、Tool Registry 和 Progress Reducer 分层架构，将安全分类、请求级最小工具范围、参数校验、超时、重复调用检测和引用白名单固化在 Python 代码中；支持 Vision、知识检索、脱敏模拟遥测、测试草案和时间五类只读工具。
- 建立离线 Fixture、真实模型抽样、人工复核、安全专项和完整回归五层评测：30/30 固定多模态场景通过，34/34 安全专项通过，2655/2655 自动化测试通过；真实模型固定抽样严格 4/10，并保留失败场景和质量边界。

## 3. 详细项目描述

RobotOps Copilot 面向机器人研发和运维场景，接收脱敏故障描述、日志及最多三张图片。系统先通过确定性安全分类和请求级工具策略限制 Agent 能力，再由 LLM Planner 在允许范围内选择只读工具。Runner 负责循环与终止，Executor 负责名称、参数、风险、超时和输出 Schema 校验。

知识侧使用确定性文本归一化、关键词与向量双路召回、RRF 融合、头部保留重排和组合证据门控。模型只引用本次真实检索产生的 Chunk ID，引用元数据由 Python 组装并再次执行白名单校验。

图片侧执行 Base64、真实格式、大小、边长、像素数和解压缩风险检查，为每张图片生成请求内唯一引用，并通过 OpenAI 兼容 Vision Provider 返回结构化可见观察。视觉、知识库、模拟遥测和用户输入使用不同来源标记，冲突或高风险请求进入 `partial`、`abstained` 或 `human_review_required`。

系统通过 SSE 展示生命周期，持久化脱敏会话，并提供轻量 Web 控制台、最近会话查询、JSON/Markdown 导出和 Docker Compose 本地运行。

## 4. STAR 表达

### Situation

机器人诊断信息分散在工程文档、日志、状态数据和现场图片中，直接使用 LLM 容易产生无来源结论、高风险操作建议和不可审计过程。

### Task

构建一个只读、可追溯、可拒答、可审计的多模态诊断候选系统，并提供真实评测和可复现本地交付。

### Action

- 构建混合检索、重排、证据门控和引用白名单。
- 实现 Planner、Runner、Executor、Registry 和进度 Reducer。
- 在 Planner 前增加确定性安全分类和最小工具策略。
- 接入 Vision、图片资源校验和来源分层。
- 建立 SSE、会话、控制台、导出、Docker 和分层评测。

### Result

完成 30/30 固定场景回归、34/34 安全专项、2655/2655 自动化测试和完整候选演示闭环；同时通过 10 条真实模型抽样识别出特定页检索、Vision 无效响应和 Planner JSON 稳定性问题，为后续迭代提供可复现基线。
