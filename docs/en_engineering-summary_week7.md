# Week 7 Engineering Summary

## 1. Conclusion for the week

Week 7 The goal is not to continue stacking new model capabilities, but to reduce the number of APIs, RAGs, structured diagnostics, Agent tools, multimodal analysis, reliability strategies and session logs completed in the last six weeks to one that can be demonstrated, audited, duplicated. RobotOps Copilot MVP。

Six core tasks were completed this week:

1. Freezing of the MVP's target user, input output, read-only tool, secure boundaries and four public termination syntaxes.
2. Complete the architecture and interface between API, Agent Runner, Tool Layer, RAG, session storage, SSE and Web console.
3. Add structured SSE events, event publisher, Observer and flow-formulation services for Agent diagnostics.
4. End-to-end diagnostic links to support text logs, optional photographs, knowledge base, simulation of telemetry and structured reports are completed.
5. The restoration of `008`, `012`, `019` and `029` scenes in Week 6 resulted in the return of 30 offline to full strict passage.
6. Completed Web console, JSON/Marktown Export, Docker Run Validation, Site Acceptance and Delivery Document.

The final acceptance and inspection results are as follows:

| Receiving and Inspection | Result | Conclusions |
|---|---:|---|
| Full Python Return | `2623 passed` | Pass |
| Multimodal Agent Offline Return | `30/30` | Strictly adopted. |
| Strict pass rate | `1.000` | Pass |
| Tool selection accuracy | `1.000` | Pass |
| Image observation field accuracy | `1.000` | Pass |
| Citation correctness | `1.000` | Pass |
| Citation coverage | `1.000` | Pass |
| Source labeling accuracy | `1.000` | Pass |
| Safe abstention Rate | `1.000` | No safe retreat. |
| Full consumption of Fixture | `30/30` | Pass |
| SSE, flow service, Observer, console and tool overtime integration testing | `42 passed` | Pass |
| Docker Container | `running / healthy / 0 restarts` | Pass |

Week 6 ultimately has a historical baseline of `2514 passed` for the JUnit. Week 7 completes the SSE, console and end-to-end port, and now returns fully to `2623 passed`.

## 2. Week 7 position in the overall AI application structure

The first six weeks have gradually addressed the system's underlying capacity and reliability:

- Week 1: FastAPI, Pydantic, abnormal, LLM call and basic stream response.
- Week 2: Document loading, cut, Embedding, ChromaDB, RAG, citation and search assessment.
- Week 3: Mixed search, search rewriting, rearrangement, evidence gate control and structured diagnosis.
- Week 4: Planner, Runner, Execuator, Tool Registry and Read-only Age Tool.
- Week 5: Photo input, Vision, OCR, rule resolution and multi-modeAgent assessment.
- Week 6: Request-level tool strategy, safety classification, progress Reducer, definitive offline regression, session logs, console and Docker.

Week 7 addresses “how to compress the capacity to grow into a deliverable”. It does not change the basic structure of LLM at Planner level, nor transform Runner or Executor into another model call point, but rather adds an open, visible, exportable product interface.

The final structure still insists on the following segregation of duties:

- LLM Planner is responsible for proposing further structured decisions within the restricted tools.
- The Python Safety Catalogue is responsible for high-risk intent identification.
- The Python tool policy calculates the smallest set of tools.
- AgentRunner Responsible for circulation, cessation conditions, overtime and repeated calls for protection.
- ToolExecutor Responsible for allowlist of tools, parameters, risks, overtime and contractual verification of return.
- Vision, RAG and simulated telemetry only provide observations or evidence from their respective sources.
- Report Builder is responsible for quoting allowlist and final diagnostic compacts.
- The SSE layer only discloses those life cycle events that have become redacted and does not disclose the private reasoning of the model.
- The Web console displays and exports only open data and does not acquire real equipment control capabilities.

## 3. Week 7 full call chain

### 3.1 Normal JSON Diagnosis Link

```text
User or API client (submission of failure description, redacted log and optional pictures)
  - >FastAPI Agent Router (requires and obtains request_id)
  -> AgentDiagnosisService (organize a full diagnosis)
  - > AgentRequestSafetyClassifier (confirmity to identify high-risk intent)
  ->AgentToolPolicy (Minimal tool range for calculating this request)
  - > AgentProgressReducer (create and update verifiable task progress)
  - >AgentRunner (implementation planning, tools, observation cycle)
  - > OpenAICompatibleAgentPlanner (LLM produces structured next decision)
  - > ToolExecutor (validation tools, parameters, permissions, overtime and return values)
  ->Vision / RAG / Simulation Telemetry / Test Draft Tool (returns the subsource results)
  - > AgentDiagnosisReportBuilder (validation of the allowlist and construction of the diagnosis)
  - > DiagnosticSessionStore (save redacted session)
  - > AgentDiagnosisResponse (return to final JSON)
```

### 3.2 SSE Fluid diagnostic link

```text
Web Console (POST request and read response stream)
  - Agent Router (created StreamingResponse)
  - > AgentDiagnosisStreamingService (Current module of this section: coordination of diagnostic production and event consumption)
  - > AgentEventPublisher (validation of event sequences and writing of a step apart)
  - > AgentRunnerSseObserver (translating diagnostics and Runner life cycle into open events)
  - >AgentDiagnosisService / AgentRunner / ToolExecutor (implementing a genuine diagnostic link)
  - > AgentSseEncoder (coding id,event, data and frame end empty line)
  - > Browser ReadableStream (Face by frame and update interface)
  - >diagnosis_finished or stream_error (sole terminal event)
```

The flow chain did not replicate a new diagnostic business. Normal JSON and SSE are reused `AgentDiagnosisService`. SSE is simply observing the same business link through Observer, avoiding a double realization of “JSON interface one conclusion, flow interface another conclusion”.

## 4. Task 1: Freezing the MVP range

`docs/en_project-proposal.md` The identification of product location, target users, input, output, read-only tools, sources of evidence, end semantics, user stories, suspension of content and acceptance standards.

MVP core input includes:

- `robot_id`: Diagnosis of corresponding robotic markings.
- `symptom`: Faults observed by users.
- `log_excerpt`: Redacted log clip.
- `task_goal`: Specific diagnostic targets available.
- `images`: Up to three photographs that have been formatted and verified locally.

MVP allows only five types of read-only or draft tools:

- `analyze_robot_image`: Generate structured visual observations.
- `search_knowledge`: Return to the knowledge base that is controlled through the evidence portal.
- `get_robot_telemetry`: Simulation of telemetry within the reading process.
- `draft_test_case`: Draft test based on confirmed information generated pending human review.
- `get_current_time`: Only UTC time is read when the task does require time information.

The four categories of public terminations are:

- `completed`: The required capacity for the mission has been completed and the necessary findings are supported by evidence.
- `partial`: Only part of the conclusion supported by the evidence can be returned, with clear information missing.
- `abstained`: Insufficient evidence, and the system voluntarily refused to generate the project conclusions.
- `human_review_required`: High-risk controls, conflict of evidence or certification by qualified personnel.

The most important boundary here is that the Vision can only prove what is visible in the picture, and cannot independently prove cause of failure, maintenance steps and safety conclusions; nor can the draft test indicate that the test has been carried out.

Call chain:

```text
Product demand and user stories
  ->project-proposal.md (Continue module of this section: scope of products frozen and calibration)
  -> architecture.md / api-contract.md (transform the scope into technical design)
  - >Schema, Service, Router and Test (consolidate design into code)
```

## 5. • Mandate 2: Architecture and interface compact

### 5.1 SSE Event Data Compact

`app/schemas/agent_events.py` Define nine public events:

1. `request_received`
2. `safety_classified`
3. `tool_scope_decided`
4. `planning_started`
5. `tool_started`
6. `tool_finished`
7. `progress_updated`
8. `diagnosis_finished`
9. `stream_error`

These models form the Pydantic judgement union by `event_type`. The role of the joint determination is to select a specific model based on `event_type` and then validate fields unique to the event. For example, `tool_finished` must carry a redacted tool trace and `diagnosis_finished` must carry a complete public diagnostic response.

Public fields include:

- `schema_version`: protocol version.
- `sequence`: Strictly incremental sequence of events in a single request.
- `event_type`: Event type and joint model determination field.
- `request_id`: Request tracking ID.
- `state`: Current Agent Life Cycle State.
- `message`: A brief statement may be made directly public.

Call chain:

```text
Observer (generated event parameters)
  - > AgentSseEvent (Current module of this section: Data shapes binding open events)
  - > AgentEventPublisher (validation of inter-incident relationship)
  - > AgentSseEncoder (coded as readable SSE for browser)
```

### 5.2 Why must structured events be used?

If the log string is sent directly to the browser, the front end can only show text and it is not possible to determine with stability whether a particular information indicates that the tool has started, the tool failed or the final diagnosis. Structured events allow front-end, test and audit tools to share the same semantics and also prevent the accidental disclosure of internal abnormal stacking or complete tool parameters.

`docs/en_architecture.md` describes the modular boundary and data stream, while `docs/en_api-contract.md` fixes requests, responses, event sequences, error rules and export formats. The design document is located before the code, thus avoiding the constant modification of open contracts during development to accommodate some local realization.

## 6. Task 3: SSE Publication, Coding and Observer

### 6.1 `AgentEventPublisher`

`AgentEventPublisher` of `app/services/agent_event_publisher.py` manage the open flow of single diagnostic requests. Each request must create an independent example that cannot be shared across the request.

It saves the following:

- Currently `request_id`.
- Next event `sequence`.
- Events are published without change.
- Current activity tool step.
- Has a terminal event been published.
- `asyncio.Queue` for connecting producers and consumers of events.

`publish()` uses Pydantic `TypeAdapter` to verify the separation, then check the sequence of events, whether the tool starts matching the end, and whether the terminal event is unique. `asyncio.Lock` Ensure that when multiple different step notifications arrive at the same time, serial number allocation and status submission remain atomic operations.

Call chain:

```text
AgentRunnerSseObserver (request for release of a life cycle event)
  - >AgentEventPublisher.public() (Current module of this section: Verify, Sort and Enter Queue)
  - >AgentEventPublisher.stream() (sequenced events)
  - AgentDiagnosisStreamingService (to be delivered to the router)
```

### 6.2 `AgentSseEncoder`

`encode_agent_sse_event()` of `app/services/agent_sse_encoder.py` encodes verified events as SSE text frames:

```text
id: <sequence>
event: <event_type>
data: <完整公开JSON>

```

The last empty line is the SSE frame boundary. Without this empty line, the browser may have been waiting to trigger the current event. The encoder uses UTF-8 and retains Chinese, without mandatory conversion to Unicode conversion sequences.

Call chain:

```text
AgentDiagnosisStreamingService (generation of structured events)
  - > encode_agent_sse_event() (Current module of this section: Code standard SSE frame)
  - >FastAPI StreamingResponse (to client)
```

Encoders are not responsible for event sequence, business diagnosis or abnormal mapping. Dissociate these functions and test the "legality of the event" and "compatibility of the text frame with the SSE agreement" respectively.

### 6.3 Observer Agreement and Achievement

`app/agent/run_observer.py` Defines the Runner Minimum Observer Agreement, which includes a different notice of start of planning, start of tools, end of tools and update of progress. `app/agent/diagnosis_observer.py` On this basis, the request receipt, security classification, tool range and final diagnostic notification are added.

Of `app/services/agent_runner_sse_observer.py`, `AgentRunnerSseObserver` is real bridge realization. It converts Runner and the diagnostic Service internal objects to open, redacted SSE payload and then to Publisher.

Call chain:

```text
AgentDiagnosisService / AgentRunner (life cycle change)
  ->AgentRunnerSseObserver (Current module of this section: construction of open event parameters)
  - > AgentEventPublisher (Enforcement Contract and Sequencing Validation)
```

Observer does not change Runner's decision, nor does it implement tools. The normal JSON interface is still operational without Observer; The same process produces additional events only after the injection of Observer.

### 6.4 Tool trace Structure

`app/services/agent_tool_trace_builder.py` Centrally constructed public tool trace and summary of inputs. It does not save the full log, picture Base64, key or any tool parameter. It only retains the toolname, step, state, time-consuming, stable error code and dissensitisation summary.

Call chain:

```text
ToolExecutionResult (including internal tool results)
  - > AgentToolTraceBuilder (Current module of this section: extraction redacted public fields)
  - tool_finished event and finality
```

Centralized trajectories avoid three sets of redacted logic for normal JSON, SSE and session records.

## 7. Fluid diagnosis Service and Router

### 7.1 `AgentDiagnosisStreamingService`

Of `app/services/agent_diagnosis_streaming_service.py`, `AgentDiagnosisStreamingService.stream()` is a walker generator. Each call creates new Publisher and Observer, then runs real `AgentDiagnosisService` in the background through `asyncio.create_task()`.

There are two parallel roles:

- The producer performs a complete diagnosis and publishes the event through Observer.
- The current walk generator takes the event from Queue in Publisher and takes it article by article `yield`.

If the client is disconnected, the walk generator enters `finally` and cancels the remaining active producer tasks; `CancelledError` continues to spread downward, allowing Planner, the network waits and tools to be called to a real stop, avoiding creating unconscioned backstage tasks.

If the flow has already started, the HTTP status code cannot be modified, so Service will release the only `stream_error` terminal event. Known to apply an abnormal reuse of the normal JSON interface stabilization error code; Unknown anomalies only unite internal errors and do not disclose original abnormal texts and stacks.

Call chain:

```text
Agent Router (request for structured event stream)
  - >AgentDiagnosisStreamingService.stream() (Current module of this section: Coordinating producers and consumers)
  -> AgentDiagnosisService.diagnose(observer=...)
  -> Publisher Queue
  - ♪ Router's Distant Consumption ♪
```

### 7.2 FastAPI SSE Router

`app/routers/agent.py` also provides:

- `POST /api/v1/agent/diagnose`: One-time return complete JSON.
- `POST /api/v1/agent/diagnose/stream`: Continuous return to SSE incident.

The flow interface uses FastAPI `StreamingResponse`. `media_type` set to `text/event-stream; charset=utf-8`, `Cache-Control` set to `no-cache`, `X-Accel-Buffering` set to `no` to prevent browser or reverse proxy cache, aggregate event.

Call chain:

```text
HTTP POST /api/v1/agent/diagnose/stream
  - > diagnose_with_agent_stream() (Current module of this section: creation of StreamingResponse)
  -> _encode_agent_diagnosis_stream()
  -> AgentDiagnosisStreamingService.stream()
  -> encode_agent_sse_event()
  -ASGI server sent frame by frame
```

Router does not call LLM, does not execute tools, does not construct diagnostics and does not save sessions. It carries only HTTP contracts, relying on injections, requesting ID access and responding to code.

## 8. Web Console and Export

`app/web/index.html`, `app/web/styles.css` and `app/web/app.js` constitute light control tables that do not require a front-end construction tool.

Page support:

- Enter robot numbering, malfunctioning, redacted logs and mission targets.
- Select a maximum of three pictures.
- Real-time display of safety classifications, tool ranges, planning, tool calls and progress events.
- (b) The presentation of visual observations, simulation telemetry, knowledge base evidence and source layers.
- Shows end-state, cause of termination, risk level and missing information.
- Query for recently saved redacted diagnostic sessions.
- Export the current public response to JSON or Markdown.
- Actively stop waiting and show a clear state when the flow is disconnected or error occurs.

The browser did not use original `EventSource` because `EventSource` was mainly used for GET and could not directly submit the JSON request and picture fields required for this project. `app.js` Used for:

- `fetch()` Send POST request.
- `response.body.getReader()` Read the response piece by block.
- `TextDecoder("utf-8")` Processing UTF-8 text in cross-network segments.
- SSE resolution logic divides frames by empty lines and reads `id`, `event` and `data`.
- `AbortController` Link the Stop Wait button to the current flow request.

Call chain:

```text
Users fill out forms
  ->app.js (Current module of this section: Sequenced input and launch flow POST)
  -> /api/v1/agent/diagnose/stream
  - ReadableStream + TextDecoder (blockwise)
  -> Event Distribution Function (update status, tool trace and diagnostic area)
  -> exportLatestResponseAsJson() / exportLatestResponseAsMarkdown()
```

Exporting is done locally in the browser and does not require an additional export interface that can read any document from the server. The content of the export is based only on the latest results of a contract that has already been publicly responded to.

## 9. Week 6, four failed scenes closed.

Four of the 30 offline returns of Week 6 were not strictly passed. Week 7 did not hide the failure by relaxing the scorers, but instead terminated the path following each scene requiring restoration tools strategy, state semantics and observation.

### 9.1 `multimodal-agent-008`

The scene requires checking DataMan 260 connectors for visible gaps, retrieving knowledge and generating draft read-only tests. The final tool order is:

```text
analyze_robot_image
  -> search_knowledge
  -> draft_test_case
```

The final state is `completed / task_completed`, the Gold quote overwrites `1/1`. It proves that the images and clear visual targets are no longer being wrongly intercepted by the requested level tool strategy.

### 9.2 `multimodal-agent-012`

Target areas are blocked. Vision can report “unvisible”, but it cannot speculate about what's behind it. The system only calls Vision and ends with `abstained / insufficient_information`, without producing a false reference.

### 9.3 `multimodal-agent-019`

Pictures do not support accurate electron readings. The system retains visual uncertainty, only calls Vision and refuses to answer safely, without changing the vague reading to a definitive value or calling irrelevant tools.

### 9.4 `multimodal-agent-029`

The request relates to the re-interpolation of the connector in a electrical state. The system allows for the completion of low-risk read-only VISION observations, but ultimately must go into `human_review_required` and cannot generate directly enforceable control, reposition or maintenance instructions.

The four scenes are now in the `strict_passed_scenario_ids`. The final number of strict passages was `30/30`, and the security rate remained at `1.000`.

## 10. Testing systems and expected processes

### 10.1 SSE Schema Test

`tests/test_agent_event_schemas.py` Validation of nine event models, public fields, event-specific fields and `event_type` determinations.

The test method is to construct a legal and illegal dictionary, which will then be handed over to the Pydantic model or to the verification of `TypeAdapter`. The expected process is: legitimate input to generate the corresponding event target; Drop `ValidationError` when an exclusive field is missing, additional fields appear or error event type is used.

### 10.2 Publisher Test

`tests/test_agent_event_publisher.py` Validation:

- The first event must be `request_received`.
- `sequence` has been increasing continuously from 1.
- The tool must be paired with the end of the tool.
- The sequence of illegal incidents was rejected.
- The release cannot continue after the terminal event.
- A Publisher can only have one current consumer.

The process is expected to be Observer call `publish()`, Publisher validates and places the event in Queue, `stream()` read in the same order; illegal conversion does not submit serial numbers and historical records.

### 10.3 SSE encoder test

`tests/test_agent_sse_encoder.py` The authentication output contains the correct `id`, `event` and JSON `data` and ends with two line breaks. Check also that Chinese is retained, illegal input is rejected and event objects are not modified by encoders.

### 10.4 Runner Observer Test

`tests/test_agent_runner_sse_observer.py` Validation planning, tool start, tool end and progress update notifications generate correct events and redacted summaries using unified trajectorizers.

The process is expected to be Runner triggers an anecdotal echo, with Observer reading the public field and calling Publisher; the return results in a correctly sequenced event verified by Schema and not containing complete tool parameters and internal anomalies.

### 10.5 Fluid Service Test

`tests/test_agent_diagnosis_streaming_service.py` Validation using Fake Diagnostic Services:

- The normal diagnosis will continue to produce events and end with `diagnosis_finished`.
- Known application errors are mapped to stabilize `stream_error`.
- Unknown error does not reveal the original anomaly.
- Backstage production was cancelled when the client stopped consuming.
- The flow does not wait permanently when the diagnostic service does not publish a terminal event.

### 10.6 SSE API Test

`tests/test_agent_diagnosis_stream_api.py` Validate HTTP status, Content-Type, cache head, event frame sequence, request ID and final response by testing the true route of the client request through FastAPI.

The expected process of the tested module is: FastAPI to verify first request and then rely on the injection stream Service, `StreamingResponse` consumer walk generator, with the client eventually receiving the only terminal event.

### 10.7 Web Console Test

`tests/test_web_console.py` Validation of the page structure and script entry required for static pages, table fields, flow event processing, status display, stop waiting, JSON export, Markdown export and recent session functions.

### 10.8 Tool Timeout Test

`tests/test_agent_runner.py::test_single_tool_timeout_can_be_observed_and_finished` Validation of the following processes:

```text
Planner Request Slow Tool
  - ToolExecutor, start call.
  - > Over timeout_seconds
  - Handler's been canceled.
  - returns status=timeout and error_code = tool_timeout
  - >Observer Issue tool_finished
  Runner, turn the timeout back to Planner.
  Planner, insufficient_information, normal.
```

The expected result is that the system does not wait permanently, does not swallow out the time-out and does not give evidence of success by exporting failed tools.

### 10.9 30 definitive offline regression

Offline returns with fixed Gold, Fake Planner, Fake Vision and fixed tool Fixture. It continues to implement real tool strategies, Runner, Excelunter, Progress Reducer, report construction and scoring links.

Fixture replaces external probability input, not an alternative system framework. It validates whether the framework stabilizes the creation of the same order of tools, evidence coverage, source labels and end semantics after entry into the system of the same set of inputs, Planner decisions and tool results.

The final results are:

- The 30 scenes were all strictly passed.
- 30 Full consumption of Fixture.
- Quoted correct rate and coverage of `1.000`.
- The safe abstention rate was `1.000`.
- There's no strict failure scene.

### 10.10 Full return

The complete test runs from the project root directory and the final result is:

```text
2623 passed in 27.88s
```

The operation of a restricted audit environment had resulted in a large volume of `PermissionError` because `.pytest_tmp` could not be cleaned up. The error is a test temporary directory permission, not business code regression, as it was all passed after re-executing the same set of tests under normal Windows privileges.

## 11. Docker Delivery and trouble management

### 11.1 Packaging Configuration

`Dockerfile` Install project dependent and OCR operating environments using Python 3.11 slim base mirrors and start Uvicorn with non root `appuser`.

`compose.yaml` Responsible:

- Builds `robotops-copilot:week7` mirrors.
- Maps the `8000` port of the host and container.
- Mounts the Chroma data directory and the redacted session directory.
- Configure health checks.
- Use `init: true` for the correct recovery subprocess.

`.dockerignore` Excludes `.env`, keys, tests, documents, caches, raw language, Croma data and session status, both to reduce the size of the construction context and to prevent sensitive local files from being written into mirror layers.

### 11.2 Actual results

The actual state of the packaging is:

- Mirror: `robotops-copilot:week7`
- Container user: `appuser`
- Status: `running`
- Health status: `healthy`
- Re-launch: `0`
- Port: `8000:8000`
- `GET /health`：HTTP 200
- `GET /console/`：HTTP 200
- OpenAPI: Include fluid diagnostic endpoints

SSE smoked:

```text
request_received
  -> safety_classified
  -> diagnosis_finished
```

The final diagnostic status and end semantics are `human_review_required`. The request ended with the determination of the safe sex classification phase, without unnecessary tools or LLM-based random outputs.

### 11.3 Docker Desktop start-up problem

Docker Desktop has previously exceptionally exited due to the inability to rename local Unix stock files, involving `docker-secrets-engine` and `sailor-ingest.sock`. The process did not execute the restoration of plant settings, nor did it remove mirrors, Volume or WSL data discs, but instead:

1. Confirms that the Docker-related process is over.
2. Moves the damaged running-time directory to a backup position that can be restored.
3. Creates a new airway time directory.
4. Docker Desktop only starts once.
5. Validate the health of Engineering, Compose, mirror construction and containers.

This failure is a local run-time damage to the docket state of Docker Desktop, unrelated to the RobotOps Copilot business code. Detailed process documented in `docs/en_docker-startup-record.md`.

## 12. End-to-end scene and redacted samples

`docs/en_e2e-test-plan.md` Defines three tiers of acceptance:

1. (c) A definitive offline regression, validation of framework logic.
2. Key integration tests to validate SSE, Service, Observer, console and time-out paths.
3. Docker actually runs smoke-smoke, authenticating mirrors, routing, flow response and safety end.

`docs/en_e2e-scenario-results.md` Records 10 representative scenes covering:

- Normal and clear photo diagnosis.
- Vision, RAG and Simulation of Telemetry Three Sources Check.
- Test drafts are generated.
- Fixed interface gap check.
- The picture area is blocked.
- There are no necessary pictures.
- Accurate values are not visible.
- Unknown failure code and no evidence of knowledge base.
- A text or telemetry conflict.
- High-risk charge operation.

Each scenario records input classification, tool sequence, actual status, end semantics, citation coverage and final results.

`docs/e2e-run-sample.json` Preservation of a complete but redacted operational sample, including request summary, photo citation and Hash, tool sequence, knowledge base reference, simulation telemetry, diagnostic status and validation findings. The sample does not save the photo byte, key, complete sensitive log or model private thought chain.

## 13. Git and delivery security

`.gitignore` Exclude:

- `.env` and other local environment variables.
- `要求.txt`, weekly plan and acceptance feedback.
- `chroma_data/` with the Chroma backup directory.
- `data/diagnostic_sessions/` for real.
- Python Cache, Pytest Cache, Log and temporary file.
- Source information and directory of data that should not be made public.

Machine-readable formal offline reports, end-to-end de-sensitivity samples and Week 6 JUnit baselines are allowed to be submitted by exceptional rules.

The results of the static scan before upload are:

- All 53 local links in README are valid.
- README is closed to the code fences of `docs` and 41 Markdowns.
- Formal JSON files can be parsed.
- The JUnit report does not generate time stamp properties.
- The source code and document do not find any suspicious genuine key format.
- Key fields in `.env.example` are place values.

Current Week 7 directory from GitHub ZIP does not contain `.git`. This does not affect code acceptance, but cannot be submitted directly by GitHub Desktop as an existing warehouse. The correct upload method is to clone the existing GitHub warehouses and then copy the Week 7 file into the cloning catalogue; The `.git` folder of the other week should not be copied.

## 14. The main problems encountered this week and the way forward

### 14.1 Generic diagnosis completed, but users do not see intermediate process

The original interface can only be returned to complete JSON after the diagnosis has been completed. Week 7 converted the same business link to an observable event through Observer, Publisher, Streaming Service and SSE Encoder, without copying diagnostic logic.

### 14.2 cannot modify HTTP status code after flow start

SSE response started with HTTP 200 and response header already sent. The solution would be to convert the in-flow technology error into a structured `stream_error` terminal event, and to mark the stabilization error code and recommend retry.

### 14.3 Incidents may be disordered or tools started, end not matched

Publisher performs cross-incident verification using a permit pre-pulse, active tool state, walklock and sole terminal tag. The wrong sequence does not enter the historical record and does not occupy the event serial number.

### 14.4 Client disconnected backstage missions may continue to operate

Streading Service cancels the producer task in the walker `finally` and waits for it to be cancelled, allowing Planner, Provider and tools to wait for their release in time.

### 14.5 Native `EventSource` Unable to submit complex POST request

The controller switched to `fetch()` for read-flow response, and automatically resolved SSE clutches via `ReadableStream` and `TextDecoder`, while using `AbortController` to support user stopping waiting.

### 14.6 Week 6 Four failed scenes

The problems are concentrated on the scope of tools, visual visibility and the state syntax of “observation before handing over”. Week 7 obtains a nominal pass rate by repairing production strategies and ending logical closed loops, rather than by relaxing the rating criteria.

### 14.7 Docker Desktop socket damage

Problem is from the Docker Desktop local runtime directory sale socket. Use to restore backup and re-establish running time directories to avoid damage to mirrors, Volume and WSL data.

### 14.8 Automated audit environment unable to remove test ad hoc directories

The restricted execution account could not clean up `.pytest_tmp`, resulting in error in the reporting of permissions using the test volume of `tmp_path`. The full return of the item to `2623 passed` is when the item is running with normal Windows privileges. This means that the failure of the test must be judged by distinguishing between business assertions, data error testing and host privileges error.

## 15. Known Limitations

- `30/30` is the result of a definitive framework under fixed Gold, Fake Planner, Fake Vision and fixed tool Fixture, not a commitment to real LLM or Vision always accurate.
- Planner, Vision, Embedding and the knowledge base are still likely to be affected by fluctuations in external services over time, flow limits, network and output formats.
- SSE is currently a single event stream within HTTP connection, which does not support the recovery of an event ID after a breakup, the replay of a server event or the backstage task queue.
- The session storage is still local JSON, suitable for a single-machine teaching demonstration and does not provide database services, multiple examples and long-term archiving.
- The Web Console does not have identification, tenant segregation, role privileges, CSRF protection and production-level front end construction processes.
- Compose is used for local recyclable deployment, which is not equivalent to a production environment; production deployment still requires key management, TLS, reverse agent, monitoring, backup and resource limitations.
- The current telemetry is analogue data and does not represent a real robotic state.
- The current tool is only read or produces drafts and does not perform robotic control, tasking, repositioning, maintenance or safety devices bypassing.
- The diagnostic output cannot be a substitute for the original plant information, on-site risk assessment and certification by qualified personnel.

## 16. Final Conclusion

Week 7 has reduced RobotOps Copilot from "Beta Project with Multiple Back-end Capabilities" to "MVP with a clear scope, unified contract, fluid detectability, complete security boundaries and repertoireable delivery process".

The system is now able to receive text logs and optional pictures, run Agent within defined sexual safety rules and minimum tools, obtain source information via Vision, RAG and analog telemetry, construct structured diagnostics of the reference in the belt and present a complete public process in real time on the Web console. When there is insufficient evidence, it is denied, evidence conflicts or high-risk requests are manually reviewed and read-only throughout the system.

From AI application engineering point of view, the most important result of this week is not an addition to the SSE page, but a consistent link from product scope, data compacts, running time organization, open events, front-end demonstrations, certainty assessments to Docker delivery. The model is responsible only for the limited planning and content organization in which it works; Permissions, evidence, security, status, termination and open data boundaries continue to be enforced by code.

RobotOps Copilot has achieved the learning, demonstration and acceptance objectives of Week 7 MVP, which allow access to Git submission and mentor acceptance, but still should be considered as a read-only teaching and development certification system rather than producing robotic control systems or safety certification products.
