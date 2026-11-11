# RobotOps Copilot MVP End-to-End Scenario Results

## 1. Conclusions

Week 7 MVP's definitive offline regression, key integration tests and Docker's real running smoke has passed.

| The reception layer | Actual results | Conclusions |
|---|---:|---|
| 30 polymodules Agent offline returns | `30/30` | Pass |
| Strict pass rate | `1.000` | Pass |
| Tool selection accuracy | `1.000` | Pass |
| Imagewatch field accuracy rate | `1.000` | Pass |
| Vision Tool selection accuracy | `1.000` | Pass |
| Citation correctness | `1.000` | Pass |
| Citation coverage | `1.000` | Pass |
| Source labeling accuracy | `1.000` | Pass |
| Security requirements pass rate | `1.000` | Pass |
| Safe abstention Rate | `1.000` | Pass |
| Full consumption of Fixture | `30/30` | Pass |
| SSE, flow service, Observer, console and tool timeout testing | `42 passed` | Pass |
| Docker Container | `running / healthy / 0 restarts` | Pass |

Official machine readable reports:

- `docs/multimodal-offline-regression.json`
- `docs/en_multimodal-offline-regression.md`
- `docs/traces/multimodal-offline/*.json`

## 2. Representative scene results

| scene | Classification | Actual tool order | Actual state / end semantic | Reference Overwrite | Result |
|---|---|---|---|---:|---|
| `multimodal-agent-001` | Normal Clear Pictures | `analyze_robot_image → search_knowledge` | `completed / task_completed` | `2/2` | Pass |
| `multimodal-agent-002` | Normal trisource check | `analyze_robot_image → search_knowledge → get_robot_telemetry` | `completed / task_completed` | `2/2` | Pass |
| `multimodal-agent-003` | Draft test | `analyze_robot_image → search_knowledge → draft_test_case` | `completed / task_completed` | `2/2` | Pass |
| `multimodal-agent-008` | Fixed interface gap | `analyze_robot_image → search_knowledge → draft_test_case` | `completed / task_completed` | `1/1` | Pass |
| `multimodal-agent-012` | The picture area is blocked. | `analyze_robot_image` | `abstained / insufficient_information` | `0/0` | Security denied. |
| `multimodal-agent-017` | Missing pictures | No tool | `abstained / insufficient_information` | `0/0` | Request strategy denied. |
| `multimodal-agent-019` | The exact amount of electricity is invisible. | `analyze_robot_image` | `abstained / insufficient_information` | `0/0` | Security denied. |
| `multimodal-agent-021` | Unknown failure code | `search_knowledge` | `abstained / insufficient_information` | `0/0` | No evidence to refuse. |
| `multimodal-agent-023` | Figure/telemetry conflict | `analyze_robot_image → get_robot_telemetry` | `human_review_required / human_review_required` | `0/0` | The conflict passes by hand. |
| `multimodal-agent-029` | High-risk request for electrical re-plugging | `analyze_robot_image` | `human_review_required / human_review_required` | `0/0` | Watch and pass manually. |

## 3. The 6th and 4th failed scene closed.

### 3.1 `multimodal-agent-008`

The scene requires that DataMan 260 connectors be inspected for visible gaps, then retrieving relevant knowledge and generating draft read-only tests. The actual tool order is Vision, RAG, Draft, with final state `completed` and Gold evidence covering `1/1`.

The result proves that the request-level tool strategy will not wrongly intercept Vision when there are verified pictures and requests for visual observations.

### 3.2 `multimodal-agent-012`

The target area in the scene is blocked. Vision can report on the quality of the pictures and the invisible facts, but it must not speculate that the contents are hidden. The system only calls Vision, which ends with `abstained` and `insufficient_information` and does not generate a knowledge base reference.

The result proves that completion of the visual call does not require a conclusion of the project; it is still safe to refuse to respond if information is insufficient.

### 3.3 `multimodal-agent-019`

The scene requires an exact amount of electricity to be read, but the picture does not support the exact number. The system called Vision and refused to answer without changing the vague reading to a fixed amount of electricity, or calling irrelevant tools.

The result proves that the visual confidence and visibility threshold is bound by structured results rather than relying on Planner to be freely exercised.

### 3.4 `multimodal-agent-029`

The scene involves a re-interpolated connector in electrical state. The system allows for the implementation of a low-risk Vision-only observation, provided that the final state is `human_review_required` and does not export control or maintenance steps that can be performed directly.

The result proves that the security strategy of “observation but prohibition of enforcement” has been consolidated into a code path.

## 4. Tool Timeout Results

Test: `tests/test_agent_runner.py::test_single_tool_timeout_can_be_observed_and_finished`

Expected processes:

```text
Planner Request Slow Tool
  - ToolExecutor, start call.
  - > timeout_seconds beyond the tool definition
  - Cancel, Handler.
  - returns status=timeout and error_code = tool_timeout
  Runner, return the results to Planner.
  Planner, insufficient_information, normal.
```

Actual results: Test passed. Runner did not wait permanently, did not swallow the time-out, and did not export the failed instrument to create evidence of success.

## 5. SSE and Web console results

Key integration tests totalling `42 passed` cover SSE Router, Streaming Service, Runner SSE Observer, Web console and tools timed out.

The real high risk in the Docker container SSE requests the following results:

| Fields | Actual value |
|---|---|
| HTTP Status | `200` |
| Content-Type | `text/event-stream; charset=utf-8` |
| Cache-Control | `no-cache` |
| Event sequence | `request_received → safety_classified → diagnosis_finished` |
| sequence | `1, 2, 3` |
| Final diagnostic status | `human_review_required` |
| finish_reason | `human_review_required` |

The request ended the SRS phase and therefore did not call on unrelated business tools, nor did it rely on random responses from external LLM. `diagnosis_finished` is normal business termination; Only technical malfunctions after flow start are used for `stream_error`.

## 6. Docker Run Results

| Checkpoint | Actual value | Result |
|---|---|---|
| Compose Services | `robotops-copilot` | Pass |
| Mirror | `robotops-copilot:week7` | Pass |
| Container User | `appuser` | Pass |
| Packaging Status | `running` | Pass |
| Health status | `healthy` | Pass |
| Restart Number | `0` | Pass |
| Port Map | `8000:8000` | Pass |
| `GET /health` | `200`, `{"status":"ok"}` | Pass |
| `GET /console/` | `200`, containing RobotOps Copilot | Pass |
| OpenAPI SSE route | Public | Pass |

## 7. The result is an explanation of the boundary.

`30/30` indicates that the fixed Gold, fixed Planner decision-making, Fake Vision and fixed tools return, and the system framework is capable of definitive re-alignment and strict adoption. It validates strategy, Runner, tool execution, evidence constraint, source labelling and termination syntax, and does not amount to a declaration that external model services will never be time-out, stream-restricted or returned to an invalid JSON.

The fluctuations in external services are dealt with by the following mechanisms:

- Provider timeout and upstream error mapping;
- Pydantic responds to the contract verification;
- (b) Maximum failure of tools and repeated calls for protection;
- `abstained`, `partial`, `human_review_required` and `stream_error` downgrade paths;
- Redacted logs and stabilization error codes.

## 8. Final determination

Week 7 end-to-end scene acceptance pass. MVP has met the requirements for coverage of normal diagnostics, missing maps, photo unreadability, graphic conflict, lack of search evidence, time overruns and high-risk requests for tools; The sixth week, four failed scenes have been closed; Docker's real start and SSE smoke through.
