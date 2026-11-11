# SSE 流式接口协议

## 1. 接口用途

`POST /api/v1/triage/stream` 接收机器人故障描述，并以 Server-Sent Events（SSE）持续返回 LLM 生成的文本增量。

与非流式接口不同，客户端不需要等待模型生成完整结果，而是可以在每个文本分块到达后立即处理。

当前流式接口的调用链为：

```text
客户端
  → FastAPI Router
  → TriageService
  → OpenAI 兼容 LLM
  → TriageService 提取文本增量
  → Router 包装成 SSE 事件
  → 客户端
```

## 2. 请求

### 请求方法与路径

```http
POST /api/v1/triage/stream
```

### 请求头

```http
Content-Type: application/json
```

### 请求体

```json
{
  "robot_id": "robot-001",
  "symptom": "机器人启动后无法向前移动",
  "log_excerpt": "motor controller accepted move command; wheel feedback unavailable"
}
```

字段说明：

| 字段 | 类型 | 说明 |
|---|---|---|
| `robot_id` | string | 发生故障的机器人标识 |
| `symptom` | string | 已观察到的故障现象 |
| `log_excerpt` | string | 已脱敏的相关日志片段 |

## 3. 响应

流成功建立后，HTTP 响应状态通常为：

```http
HTTP/1.1 200 OK
Content-Type: text/event-stream; charset=utf-8
X-Request-ID: <request-id>
```

`X-Request-ID` 是本次请求的唯一追踪标识。

客户端不能只根据 HTTP `200` 判断业务成功。流式处理的最终状态由 `done` 或 `error` 事件决定。

## 4. 事件格式

每个 SSE 事件包含：

```text
event: <事件名称>
data: <JSON 字符串>

```

事件之间使用空行分隔。

当前协议定义四种业务事件：

- `meta`
- `delta`
- `done`
- `error`

## 5. meta 事件

`meta` 是流成功建立后的第一个业务事件。

示例：

```text
event: meta
data: {"request_id":"f90fd399-1464-4944-a75b-d76b075ecc45"}

```

数据结构：

```json
{
  "request_id": "f90fd399-1464-4944-a75b-d76b075ecc45"
}
```

用途：

- 告诉客户端本次流的追踪 ID；
- 将客户端错误与服务端日志关联；
- 验证它与 `X-Request-ID` 响应头一致。

## 6. delta 事件

`delta` 只携带相对于前一个事件新增的文本。

示例：

```text
event: delta
data: {"text":"{\"summary\":\"根据输入，"}

```

数据结构：

```json
{
  "text": "本次新增的文本"
}
```

客户端必须按照接收顺序追加 `text`：

```text
完整文本 = 第一个 delta.text
         + 第二个 delta.text
         + 第三个 delta.text
         + ...
```

不能使用新的 `delta.text` 覆盖之前的内容。

一个 `delta` 不保证对应完整汉字、单词、JSON 字段或句子。分块边界由模型服务、SDK 和网络传输共同决定。

当前 LLM 被要求生成 JSON，因此每个 `delta.text` 可能只是最终 JSON 文本的一部分。客户端应在收到 `done` 后，再解析拼接完成的 JSON。

## 7. done 事件

`done` 表示：

- 上游 LLM 流正常结束；
- 所有文本增量已经发送；
- 拼接后的完整内容通过了服务端结构校验。

示例：

```text
event: done
data: {"status":"completed"}

```

数据结构：

```json
{
  "status": "completed"
}
```

客户端只有收到 `done`，才能把本次流视为业务成功。

收到 `done` 后不会再有 `delta` 或 `error` 事件。

## 8. error 事件

`error` 表示流开始后发生了可预期的应用错误。

示例：

```text
event: error
data: {"request_id":"f90fd399-1464-4944-a75b-d76b075ecc45","error":{"code":"llm_timeout","message":"LLM 服务响应超时"}}

```

数据结构：

```json
{
  "request_id": "f90fd399-1464-4944-a75b-d76b075ecc45",
  "error": {
    "code": "llm_timeout",
    "message": "LLM 服务响应超时"
  }
}
```

当前公开错误码包括：

| 错误码 | 含义 |
|---|---|
| `llm_configuration_error` | LLM 配置缺失或无效 |
| `llm_timeout` | 等待 LLM 响应超时 |
| `llm_upstream_error` | 无法连接上游或上游返回错误状态 |
| `invalid_llm_response` | LLM 返回空内容、非法 JSON 或错误结构 |
| `application_error` | 未单独登记的应用错误 |

`message` 是可以安全展示给客户端的公开信息。

服务端内部异常详情不会直接放入事件数据，避免泄漏上游地址、供应商响应或其他诊断信息。

收到 `error` 后，客户端应：

1. 停止等待后续增量；
2. 丢弃或标记已经收到但尚未完成的文本；
3. 向用户展示公开错误信息；
4. 保存 `request_id`，用于问题追踪。

收到 `error` 后不会再发送 `done`。

## 9. 合法事件顺序

### 正常完成

```text
meta → delta* → done
```

`delta*` 表示可以有零个或多个 `delta` 事件。

不过当前业务要求 LLM 返回非空内容，因此正常情况下至少会出现一个 `delta`。

### 在任何文本产生前失败

```text
meta → error
```

### 已产生部分文本后失败

```text
meta → delta* → error
```

无论哪种失败情况，`error` 后都不能再出现 `done`。

## 10. 成功事件日志样例

以下为脱敏后的成功事件日志：

```text
event: meta
data: {"request_id":"example-success-request-id"}

event: delta
data: {"text":"{\"summary\":\"根据输入，只能确认机器人"}

event: delta
data: {"text":"无法向前移动。\",\"recommended_actions\":[\"检查急停状态\"]}"}

event: done
data: {"status":"completed"}

```

两个 `delta.text` 拼接后得到：

```json
{
  "summary": "根据输入，只能确认机器人无法向前移动。",
  "recommended_actions": [
    "检查急停状态"
  ]
}
```

## 11. 失败事件日志样例

以下为脱敏后的超时事件日志：

```text
event: meta
data: {"request_id":"example-timeout-request-id"}

event: error
data: {"request_id":"example-timeout-request-id","error":{"code":"llm_timeout","message":"LLM 服务响应超时"}}

```

虽然 SSE 连接的 HTTP 状态可能已经是 `200`，客户端仍应根据 `error` 事件把本次业务处理判断为失败。

## 12. curl 验证

建议把请求数据保存在：

```text
samples/triage-request.json
```

执行：

```powershell
curl.exe -N -i -X POST "http://127.0.0.1:8000/api/v1/triage/stream" `
  -H "Content-Type: application/json" `
  --data-binary "@samples/triage-request.json"
```

参数说明：

- `-N`：关闭 curl 输出缓冲，事件到达后立即显示；
- `-i`：同时显示 HTTP 响应头；
- `-X POST`：使用 POST 请求；
- `-H`：设置请求头；
- `--data-binary`：从文件读取并原样发送请求体。

## 13. 测试边界

自动化测试使用 HTTPX `ASGITransport` 直接调用 FastAPI 应用，不启动 Uvicorn，也不访问真实 LLM。

自动化测试负责验证：

- 事件名称；
- 事件顺序；
- 事件数据；
- 公开错误码；
- `request_id` 一致性；
- 内部异常详情不会泄漏。

`ASGITransport` 在测试中会收集完整响应体，因此不能证明事件在真实网络中出现的时间间隔。

真实的逐步输出效果由 `curl.exe -N` 手动冒烟测试验证。

## 14. 当前限制

- 当前接口只提供文本增量；
- 本周不实现正式前端；
- 本周不接入 RAG 或 Agent；
- 客户端断开后的完整资源回收测试将在可靠性阶段继续补充；
- 浏览器原生 `EventSource` 主要用于 GET 请求，而当前接口需要 POST JSON，因此浏览器客户端需要使用支持流式读取的 `fetch` 或其他 HTTP 客户端。