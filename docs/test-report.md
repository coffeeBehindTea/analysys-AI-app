# Robot Incident Triage API 测试报告

## 1. 测试目标

本轮测试用于验证 Robot Incident Triage API 的基础可靠性，包括：

- FastAPI 健康检查；
- 请求字段和长度校验；
- 非流式接口成功响应；
- 应用异常到 HTTP 错误响应的转换；
- SSE 成功事件顺序；
- SSE 失败事件顺序；
- LLM 配置缺失处理；
- OpenAI SDK 超时异常转换；
- LLM 合法 JSON 解析；
- LLM 空内容、非法 JSON 和错误结构处理；
- `request_id` 在响应头、响应体和 SSE 事件中的一致性；
- 内部异常信息不向客户端泄漏。

## 2. 测试环境

```text
操作系统：Windows
Python：3.11.15
pytest：9.1.1
pytest-asyncio：1.4.0
FastAPI：0.141.1
HTTPX：0.28.1
OpenAI Python SDK：2.53.0
sse-starlette：3.4.6
```

测试使用 Python 3.11 Conda 环境执行。

## 3. 执行命令

```powershell
python -m pytest tests -v
```

## 4. 测试结果

```text
collected 21 items
21 passed
```

所有自动化测试通过。

自动化测试不要求：

- 真实 API Key；
- 真实 LLM 服务；
- 外部网络连接；
- 正在运行的 Uvicorn 服务。

## 5. 测试文件

### `tests/test_api_contract.py`

覆盖：

- `GET /health` 返回 `200 OK`；
- 健康检查响应体为 `{"status": "ok"}`；
- 健康检查响应包含 `X-Request-ID`；
- 缺少 `robot_id` 返回 `422`；
- 纯空白 `symptom` 返回 `422`；
- 超过 10,000 字符的 `log_excerpt` 返回 `422`；
- 未声明的额外字段返回 `422`；
- 无效请求不会调用 `TriageService.analyze()`。

### `tests/test_error_responses.py`

覆盖：

- 非流式接口成功响应；
- 响应头和响应体使用同一个 `request_id`；
- `LLMConfigurationError` 转换为 `503`；
- `LLMTimeoutError` 转换为 `504`；
- `LLMUpstreamError` 转换为 `502`；
- `InvalidLLMResponseError` 转换为 `502`；
- 错误响应只包含公开错误码和安全提示；
- 内部异常详情不会直接返回给客户端。

### `tests/test_stream.py`

覆盖：

- SSE 成功响应的媒体类型为 `text/event-stream`；
- 成功事件顺序为 `meta → delta* → done`；
- `meta.request_id` 与 `X-Request-ID` 一致；
- 多个 `delta.text` 可以按顺序拼接；
- 拼接后的 LLM 文本是合法 JSON；
- 流开始前发生超时时，事件顺序为 `meta → error`；
- 已输出部分内容后发生超时时，事件顺序为 `meta → delta → error`；
- `error` 后不会发送 `done`；
- SSE 错误事件不会泄漏内部异常详情。

### `tests/test_service_errors.py`

覆盖：

- 不读取真实 `.env` 时，缺少 API Key 会抛出 `LLMConfigurationError`；
- OpenAI SDK 的 `APITimeoutError` 会被转换成 `LLMTimeoutError`；
- LLM SDK 的异步 `create()` 方法只被等待一次；
- Service 正确传递模型名称；
- Service 正确构造 system 和 user 消息。

### `tests/test_service_responses.py`

覆盖：

- 合法 LLM JSON 被解析成 `TriageResponse`；
- `request_id` 被正确保留；
- `message.content` 为 `None` 时返回 `InvalidLLMResponseError`；
- `message.content` 为空字符串时返回 `InvalidLLMResponseError`；
- LLM 返回非 JSON 文本时返回 `InvalidLLMResponseError`；
- JSON 缺少必填业务字段时返回 `InvalidLLMResponseError`；
- `completion.choices` 为空时返回 `InvalidLLMResponseError`。

## 6. 测试分层

### API 契约测试

测试范围：

```text
HTTPX
  → FastAPI
  → Middleware
  → Router
  → Pydantic
  → Fake Service
```

使用 `ASGITransport` 直接调用内存中的 FastAPI 应用，不启动真实 HTTP 服务器。

主要验证：

- HTTP 状态码；
- 响应头；
- JSON 响应结构；
- 请求字段校验；
- 依赖注入；
- SSE 事件协议。

### Service 单元测试

测试范围：

```text
TriageService
  → Prompt Builder
  → Mock OpenAI Client
  → 响应解析
  → 应用异常
```

使用 `MagicMock` 和 `AsyncMock` 替代真实 LLM 客户端。

主要验证：

- SDK 请求参数；
- SDK 异常转换；
- 模型文本解析；
- Pydantic 业务结构校验。

### 手动冒烟测试

另外使用真实 Uvicorn 服务和已配置的 LLM 服务完成了手动验证：

- `GET /health`；
- `POST /api/v1/triage`；
- `POST /api/v1/triage/stream`；
- `X-Request-ID` 响应头；
- SSE `meta`、`delta` 和 `done` 事件；
- curl 关闭输出缓冲后的逐步显示。

手动冒烟测试用于确认：

```text
真实配置
+ 真实网络
+ 真实 LLM
+ 真实 HTTP 服务
```

能够共同工作。

## 7. 为什么自动化测试不调用真实 LLM

真实 LLM 具有以下不确定因素：

- 网络延迟；
- 上游服务可用性；
- 自动重试；
- 限流；
- 调用费用；
- 模型输出措辞变化；
- SSE 分块边界变化。

因此自动化测试使用 Fake 和 Mock 固定外部行为，只验证本项目自己的工程逻辑。

真实 LLM 只用于少量手动冒烟验证。

## 8. 已覆盖的主要风险

| 风险 | 验证方式 |
|---|---|
| 服务无法响应 | `/health` 测试 |
| 请求字段缺失 | `422` 参数化测试 |
| 日志过长 | 最大长度测试 |
| 非法额外字段 | `extra="forbid"` 测试 |
| API Key 缺失 | 客户端工厂测试 |
| LLM 超时 | `APITimeoutError` Mock 测试 |
| LLM 返回空内容 | Service 响应测试 |
| LLM 返回非法 JSON | JSON 解析测试 |
| LLM 返回错误字段结构 | Pydantic 结构测试 |
| SSE 事件乱序 | 成功和失败顺序测试 |
| 错误后错误发送 `done` | SSE 失败测试 |
| 请求无法追踪 | `request_id` 一致性测试 |
| 内部异常信息泄漏 | 公开错误响应测试 |

## 9. 当前未覆盖范围

本周自动化测试尚未覆盖：

- 高并发请求；
- 负载和性能测试；
- 客户端断开后的完整资源回收；
- OpenAI SDK 自动重试次数；
- 不同 LLM 供应商的兼容性差异；
- 真实网络故障注入；
- 长时间运行的 SSE 连接；
- 安全渗透测试；
- RAG 检索质量；
- 正式前端行为。

这些内容不属于当前第一周的最小交付范围，应在后续阶段逐步补充。

## 10. 结论

当前自动化测试覆盖了本周要求的核心场景：

```text
健康检查
+ 请求字段校验
+ 非流式成功
+ SSE 成功事件顺序
+ SSE 失败事件顺序
+ 上游超时
+ 密钥缺失
+ 无效 LLM 响应
```

当前测试结果为：

```text
21 passed
```

