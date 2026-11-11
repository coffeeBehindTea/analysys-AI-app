# SSE Streaming Protocol

## 1. Interface Use

`POST /api/v1/triage/stream` Receives robotic failure descriptions and returns LLM-generated text increments on a continuous basis with Server-Sent Events (SSE).

Unlike non-flow interfaces, the client does not have to wait for the full results of the model to be generated, but can process it as soon as each text segment arrives.

The call chain for the current stream interface is:

```text
Client
  → FastAPI Router
  → TriageService
  → OpenAI Compatible LLM
  → TriageService Ripping text increment
  → Router package as SSE event
  → Client
```

## 2. Request

### Method and path of request

```http
POST /api/v1/triage/stream
```

### Request Header

```http
Content-Type: application/json
```

### Request

```json
{
  "robot_id": "robot-001",
  "symptom": "机器人启动后无法向前移动",
  "log_excerpt": "motor controller accepted move command; wheel feedback unavailable"
}
```

The fields state:

| Fields | Type | Annotations |
|---|---|---|
| `robot_id` | string | Faulted robot markers. |
| `symptom` | string | Observed failures |
| `log_excerpt` | string | Redacted related log clips |

## 3. Response

When the stream is successfully created, HTTP response status is usually:

```http
HTTP/1.1 200 OK
Content-Type: text/event-stream; charset=utf-8
X-Request-ID: <request-id>
```

`X-Request-ID` is the only trace mark for this request.

Clients cannot judge business success solely on the basis of HTTP `200`. The final state of flow processing is determined by the `done` or `error` events.

## 4. Event Format

Each SSE event consists of:

```text
event: <事件名称>
data: <JSON 字符串>

```

Use an empty line separation between events.

The current agreement defines four business events:

- `meta`
- `delta`
- `done`
- `error`

## 5. Meta Events

`meta` is the first business event since the successful establishment of the stream.

Example:

```text
event: meta
data: {"request_id":"f90fd399-1464-4944-a75b-d76b075ecc45"}

```

Data structure:

```json
{
  "request_id": "f90fd399-1464-4944-a75b-d76b075ecc45"
}
```

Purpose:

- Tell the client to track the current ID;
- Link client error to service log;
- Verify it's consistent with the `X-Request-ID` response head.

## 6. delta events

`delta` carries only new text relative to the previous event.

Example:

```text
event: delta
Data: {"text":("summery": 'by input,'}

```

Data structure:

```json
{
  "text": "本次新增的文本"
}
```

The client must add `text` in the order of receipt:

```text
Full text = first delta.text
         + Second delta.text
         + Third delta.text
         + ...
```

Cannot use new `delta.text` to overwrite pre-contents.

An `delta` does not guarantee the complete Chinese word, word, JSON field or sentence. The boundary segments are determined jointly by model services, SDK and network transmissions.

The current LLM is required to generate JSON, so each `delta.text` may be only part of the final JSON text. The client should decipher the json after receiving `done`.

## 7. Doe Events

`done` says:

- Normal end of upstream LLM flow;
- All text increments have been sent;
- The complete contents of the cipher were verified through the service-end structure.

Example:

```text
event: done
data: {"status":"completed"}

```

Data structure:

```json
{
  "status": "completed"
}
```

The current flow can only be considered a business success if the client receives `done`.

There will be no further incidents of `delta` or `error` after `done` received.

## 8. Error Event

`error` indicates that there was a foreseeable application error after the flow began.

Example:

```text
event: error
Data: {"request_id:"f90fd399-1464-4944-a75b-d76b075ecc45", "error": < < llm_timeout" >, "message": "LLM service response timeout"}

```

Data structure:

```json
{
  "request_id": "f90fd399-1464-4944-a75b-d76b075ecc45",
  "error": {
    "code": "llm_timeout",
    "message": "LLM 服务响应超时"
  }
}
```

Current public error codes include:

| Error code | Meaning |
|---|---|
| `llm_configuration_error` | LLM configuration is missing or invalid |
| `llm_timeout` | Waiting for LLM response timeout |
| `llm_upstream_error` | Unable to connect upstream or upstream to return error state |
| `invalid_llm_response` | LLM returns empty content, illegal JSON or error structure |
| `application_error` | Application error not registered separately |

`message` is a publicly available information that can be displayed safely to clients.

Intra-service anomalies details will not be entered directly into incident data to avoid leakage of upstream addresses, vendor responses or other diagnostic information.

Upon receipt of `error`, the client should:

1. Stop waiting for subsequent increments;
2. Discard or mark text that has been received but is not completed;
3. (a) Showing the user of the publicly available error information;
4. Keep `request_id` for problem tracking.

`error` will not be sent after receipt `done`.

## 9. Sequence of lawful events

### Normal completion

```text
meta → delta* → done
```

`delta*` indicates that there can be zero or more `delta` incidents.

However, the current operation requires LLM to return non-empty content, so that there will normally be at least one `delta`.

### Could not close temporary folder:%s

```text
meta → error
```

### Failed after selected text has been produced

```text
meta → delta* → error
```

In either case, `error` can no longer happen after `done`.

## 10. Successful Event Log Sample

The following are the successful post-sensitization event logs:

```text
event: meta
data: {"request_id":"example-success-request-id"}

event: delta
Data: {"text":("summary":) by input, only robots can be identified

event: delta
Data: {"text": 'can't move forward. ", "recommended_actions: ["Checking Stavron"]

event: done
data: {"status":"completed"}

```

Two `delta.text` adhesived and received:

```json
{
  "summary": "根据输入，只能确认机器人无法向前移动。",
  "recommended_actions": [
    "检查急停状态"
  ]
}
```

## 11. Failed Event Log Sample

The following is the post-sensitization timeout log:

```text
event: meta
data: {"request_id":"example-timeout-request-id"}

event: error
Data: {"request_id:"example-timeout-request-id", "error": < < llm_timeout" >, "message": "LLM service response timeout"}

```

Although the HTTP status of the SSE connection may already be `200`, the client should still judge this business processing as a failure based on the `error` incident.

## 12. Curl Authentication

It is recommended that the requested data be kept at:

```text
samples/triage-request.json
```

Implementation:

```powershell
curl.exe -N -i -X POST "http://127.0.0.1:8000/api/v1/triage/stream" `
  -H "Content-Type: application/json" `
  --data-binary "@samples/triage-request.json"
```

Parameter description:

- `-N`: Close the curl output buffer and display it immediately upon arrival;
- `-i`: also display HTTP response headers;
- `-X POST`: Use POST request;
- `-H`: set the header;
- `--data-binary`: Read from the file and send the request in its original form.

## 13. Test Borders

Automation tests use HTTPX `ASGITransport` to call directly the FastAPI application, not start Uvicorn, nor access real LLM.

Automation tests verify:

- (a) The name of the event;
- (a) the sequence of events;
- Event data;
- Public error codes;
- `request_id` Coherence;
- The details of the internal anomalies are not leaked.

`ASGITransport` The complete response body is collected during the test and therefore the time interval between events in the real network cannot be established.

The real step-out effects are verified by the `curl.exe -N` manual smoke test.

## 14. Current Limitations

- The current interface only provides text increments;
- No formal front end is achieved this week;
- No access to RAG or Agent this week;
- The complete resource recovery test after the client has been disconnected will continue to be supplemented during the reliability phase;
- The browser original `EventSource` is used mainly for the GET request, while the current interface requires POST JSON, so the browser client needs `fetch` or other HTTP clients that are read in a support stream.
