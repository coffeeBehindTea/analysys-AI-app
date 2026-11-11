# Robot Incident Triage API Test Report

## 1. Test Objectives

This round of tests is used to verify the basic reliability of the Robot Incident Triage API, including:

- FastAPI Health check-ups;
- Request field and length verification;
- Successful response from non-stream interfaces;
- Apply an abnormal to an HTTP error response;
- SSE sequence of successful events;
- SSE failure event sequence;
- LLM configuration missing processing;
- OpenAI SDK overtime abnormality conversion;
- LLM legal JSON resolution;
- LLM empty content, illegal JSON and error structure processing;
- `request_id` Consistency in response head, responder and SSE incidents;
- Internal abnormal information is not leaked to the client.

## 2. Test Environment

```text
Operating system: Windows
Python：3.11.15
pytest：9.1.1
pytest-asyncio：1.4.0
FastAPI：0.141.1
HTTPX：0.28.1
OpenAI Python SDK：2.53.0
sse-starlette：3.4.6
```

Test using Python 3.11 Conda environment execution.

## 3. Execute Command

```powershell
python -m pytest tests -v
```

## 4. Test Results

```text
collected 21 items
21 passed
```

All automated tests passed.

Automation tests do not require:

- Real API Key;
- Real LLM service;
- External network connectivity;
- Running Uvicorn service.

## 5. Test File

### `tests/test_api_contract.py`

Coverage:

- `GET /health` returns `200 OK`;
- The health check-up response is `{"status": "ok"}`;
- The health check response consists of `X-Request-ID`;
- Missing `robot_id` returns `422`;
- Pure blank `symptom` returns `422`;
- `log_excerpt` returns `422` in excess of 10, 000 characters;
- Undeclared additional field returns `422`;
- Invalid request will not be called `TriageService.analyze()`.

### `tests/test_error_responses.py`

Coverage:

- Successful response from non-stream interfaces;
- The response head and the responder use the same `request_id`;
- `LLMConfigurationError` converted to `503`;
- `LLMTimeoutError` converted to `504`;
- `LLMUpstreamError` converted to `502`;
- `InvalidLLMResponseError` converted to `502`;
- Error response only contains open error codes and security tips;
- Internal anomaly details do not return directly to the client.

### `tests/test_stream.py`

Coverage:

- The type of media that SSE has successfully responded to is `text/event-stream`;
- The sequence of successful events is `meta → delta* → done`;
- `meta.request_id` consistent with `X-Request-ID`;
- Multiple `delta.text` can be spelled sequentially;
- The spelled LLM text is valid JSON;
- The sequence of events before the flow begins is `meta → error`;
- The event sequence is `meta → delta → error` when the timeout occurs after the output of part of the content;
- `done` shall not be sent after `error`;
- The SSE error does not reveal internal anomalies.

### `tests/test_service_errors.py`

Coverage:

- The absence of API Key throws `LLMConfigurationError` without reading the real `.env`;
- `APITimeoutError` of OpenAI SDK is converted to `LLMTimeoutError`;
- The LLM SDK step `create()` method is waiting only once;
- Service correctly transmits model names;
- Service correctly constructs system and user messages.

### `tests/test_service_responses.py`

Coverage:

- The legal LLM JSON has been deconstructed to `TriageResponse`;
- `request_id` correctly retained;
- `message.content` returns `InvalidLLMResponseError` at `None`;
- returns `InvalidLLMResponseError` when `message.content` is an empty string;
- LLM returns non-JSON text back to `InvalidLLMResponseError`;
- JSON returns `InvalidLLMResponseError` if it lacks the required business fields;
- `completion.choices` returns `InvalidLLMResponseError` when it is empty.

## 6. Test Layer

### API Contract Test

Test range:

```text
HTTPX
  → FastAPI
  → Middleware
  → Router
  → Pydantic
  → Fake Service
```

Use `ASGITransport` to directly call the FastAPI application in the memory and not start the real HTTP server.

Main authentication:

- HTTP status code;
- Response header;
- JSON Response Structure;
- Request field validation;
- Dependence on injection;
- SSE event protocol.

### Service module testing

Test range:

```text
TriageService
  → Prompt Builder
  → Mock OpenAI Client
  → Reply Parsing
  → Apply Abnormal
```

Use `MagicMock` and `AsyncMock` to replace the real LLM client.

Main authentication:

- SDK request parameters;
- SDK abnormal conversion;
- Model text resolution;
- Pydantic Business Structure Validation.

### Manual smoke testing

Manual validation was also completed using the real Uvicorn service and the configured LLM service:

- `GET /health`；
- `POST /api/v1/triage`；
- `POST /api/v1/triage/stream`；
- `X-Request-ID` Response header;
- SSE `meta`, `delta` and `done` incidents;
- Curl closes the output buffer and shows step by step.

Manual smoke testing to confirm:

```text
Real Configuration
+ Real network.
+ Real LLM
+ Real HTTP service
```

Can work together.

## 7. Why doesn't automated testing call real LLM?

Real LLM has the following uncertainties:

- Network delays;
- Upstream service availability;
- (a) Automatically retrying;
- The flow limitation;
- (c) Call costs;
- Changes in model output wording;
- SSE segment boundary changes.

The automated testing therefore uses Fake and Mock to fix external behaviour, only to validate the project's own engineering logic.

The real LLM is only for a small amount of manual smoke verification.

## 8. Main risks covered

| Risk | Authentication Method |
|---|---|
| Services are not responsive | `/health` Test |
| Request field missing | `422` Parameterisation Test |
| The log is too long. | Maximum length test |
| Illegal Extra Fields | `extra="forbid"` Test |
| API Key Missing | Client-end plant testing |
| LLM timeout | `APITimeoutError` Mark Test |
| LLM returns empty content | Service Response Test |
| LLM Return Illegal JSON | JSON Parsing Test |
| LLM returns the wrong field structure | Pydantic Structure Test |
| SSE SSE Incident Sequence | Success and failure sequence test |
| Error sent after error `done` | SSE Failed Test |
| Request untraceable. | `request_id` Consistency test |
| The leak of internal abnormal information. | Open Error Response Test |

## 9. Not covered at present

Automation tests this week have not yet covered:

- (b) High-level requests;
- Loads and performance tests;
- Full resource recovery after client break-up;
- OpenAI SDK automatic retry times;
- Differences in compatibility between different LLM suppliers;
- Real network failure injection;
- SSE connections run for long periods;
- (b) Security infiltration tests;
- RAG Retrieving quality;
- Formal front-end behaviour.

These elements, which are not within the minimum delivery of the current first week, should be progressively supplemented in the follow-up phase.

## 10. Conclusions

The current automated tests cover the core scenes requested this week:

```text
Health screening
+ Request field validation
+ It's a non-flux success.
+ SSE Successful Event Order
+ SSE Failed Event Order
+ Upstream timeout
+ Key Missing
+ Invalid LLM response
```

The current test results are:

```text
21 passed
```
