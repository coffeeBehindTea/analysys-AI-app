# Week 6 Engineering Summary

## 1. Conclusion for the week

Week 6 does not continue to add new generating AI capabilities, but instead uses the multimodular Agent portal of Week 5 as a more manageable, interpretable and recapable RobotOps Copilot Beta.

Six core tasks were completed this week:

1. Add a minimum tool strategy at the request level before Planner to limit the tools that can be seen and called upon in this request.
2. Adding a defined safety classification before Agent Loop, leaving LLM free to judge high-risk control intent.
3. `AgentProgress` and `AgentProgressReducer` document capability, tool results, evidence coverage, conflict, missing information and range of next steps.
4. Use Fake Planner, Fake Vision and Fixed Fixture to create a definitive offline regression of 30 scenarios.
5. Saves redacted diagnostic sessions and provides an API for recent sessions and searching for details by id.
6. Provides a lightweight Web console, Dockerfile, Compose configuration and photo end-to-end demonstration.

The official offline return results are:

| Indicators | Week 5 baseline | Week 6, results. | Conclusions |
|---|---:|---:|---|
| Strict pass. | 7/30 | 26/30 | Add 19 |
| Strict pass rate | 0.233 | 0.867 | Over 24/30 threshold |
| Safe abstention Rate | 1.000 | 1.000 | No safe retreat. |
| Tool selection accuracy | — | 0.867 | The scope of the tool is basically stable |
| Task completion rate | — | 0.875 | 7/8 expected scenes completed |
| Citation correctness | — | 1.000 | Returning references are within Gold |
| Citation coverage | — | 0.952 | 20/21 anticipated evidence covered |
| Image observation field accuracy | — | 0.886 | 31/35 visual fields match |
| Source labeling accuracy | — | 0.867 | Most of the sources are correct, such as visual, user, knowledge base, etc. |

Full Python returns to `2514 passed`, with both failures and collection errors of 0.

## 2. Week 6 position in the overall AI application structure

The first five weeks have been devoted to the question of what capacity the system has:

- Week 1: FastAPI, data contracts, abnormal processes, LLM call and current interface.
- Week 2: Document loading, cut, Embedding, ChromaDB, RAG and quote assessment.
- Week 3: Mixed search, search rewriting, rearrangement, evidence gate control and structured diagnosis.
- Week 4: Planner, Runner, Execuator, Tool Registry and Read-only Age Tool.
- Week 5: Photo input, Vision, OCR, rule resolution and multi-modeAgent assessment.

Week 6 addresses “how these capabilities can be reliably worked together and delivered”:

```text
User Request
  - > Request level security and scope of tools
  - >Structural progress.
  - Planner, choose the next step.
  Runner validates and advances the cycle
  -Execuator, allowlist enforcement tool.
  - >Diagnosis of evidence constraints
  - Redacted sessions.
  - >API / Web Console Presentation
  - > Docker Re-enactable
```

This design separates the "Selection of steps and organizational content according to context" for which the generation model is good from the "authorities, status, data validation, cessation conditions and durability" for Python. LLM is still located on Planner level, and Runner, Execuator, Tool Strategy, Progress Reducer and Session Storage are not free to rely on models.

## 3. Full online call chain

```text
Client or Web Console (submission of text and optional pictures)
  - > FastAPI Router (validation of HTTP request contract and acquisition of request_id)
  - > AgentDiagnosisService (organized a diagnostic request)
  - > AgentRequestSafetyClassifier (identified identification of high-risk control or intent to induce injection)
  - > AgentToolPolicy (maximity of tool ranges based on mission objectives)
  - >AgentProgressReducer (create initial progress and give first-round permission tool)
  - > AgentRunner (implement restricted planning, tools, observation cycle)
  -> OpenAICompatibleAgentPlanner (LLM only generates a structured decision on the next step)
  - > ToolExecutor (validation of tool name, parameters, privileges, timeout and return contract)
  - > Read-only tools (Vision, Knowledge Bank, Simulation Telemetry, Draft Test or Current Time)
  ->AgentProgressReducer (consolidate the results of the tool and narrow down the scope of the next tool)
  - > AgentDiagnosisReportBuilder (validation of final draft and allowlist quoted)
  - > DiagnosticSessionBuilder (from open result to redacted session)
  - > DiagnosticSessionStore (Book to local session directory with JSON atoms)
  - > AgentDiagnosisResponse (return to diagnostic, source, reference and dissensitisation trajectory)
```

The safety classification and tool strategy is implemented before Planner, so the model cannot see all the tools before it decides whether to comply with the restrictions. Progress Reducer executes each tool after it has returned, so the model cannot circumvent the “no new evidence” or “no current callable” constraint of “completed capacity” through repeated planning.

## 4. Task 1: Minimal tool policy at request level

### 4.1 Data compacts

`app/schemas/agent_tool_policy.py` Definition `AgentToolPolicyDecision`. It is not the result of the tool implementation, but a decision on the authority to request access to Agent Loop. The main fields include:

- `policy_version`: Strategic version, currently `agent-tool-policy-v1`.
- `disposition`: Continue to be handed over to Planner, or refuse directly or request human review.
- `required_capabilities`: The real capacity of the mission to complete, such as visual observation, knowledge base evidence, telemetry reading or testing of drafts.
- `allowed_tool_names`: This request allows the minimum tool names exposed to Planner and Execuator.
- `reason_codes`: Machine-readable decision-making basis for testing and auditing.
- `public_message`: A statement to the user may be made publicly available upon the early termination of the strategy.
- `missing_information`: What input, evidence or manual conditions are currently lacking.

Call chain:

```text
AgentDiagnosisService (preparation of diagnostic request)
  ->AgentToolPolicyDecision (Current module of this section: Data shapes for binding policy output)
  - > AgentRunner (read allowed_tool_names and required_capabilities)
```

### 4.2 Strategy achieved

In `app/agent/request_tool_policy.py`, `AgentToolPolicy.decide()`, the requested malfunctions, redacted logs, mission objectives and picture profiles are standardized for certainty, and the capabilities required for this mission are identified and then mapped into tools:

- `vision_observation` -> `analyze_robot_image`
- `knowledge_evidence` -> `search_knowledge`
- `telemetry_observation` -> `get_robot_telemetry`
- `test_case_draft` -> `draft_test_case`
- `current_time` -> `get_current_time`

Call chain:

```text
AgentDiagnosisService (received verified request)
  - >AgentToolPolicy.decide() (Current module of this section: Compute minimum capabilities and set tools)
  - >AgentRunner.run() (only accept this allowed toolname)
  - > ToolExecutor (validation of tool privileges at execution)
```

Here, both levels of constraint are applied to Planner and Execuator. Planner can't see unrelated tools to reduce misselection; Executor again verifies to prevent invalid Planner output or error of program bypassing the permission boundary.

## 5. • Mandate 2: Defined safety classification

Of `app/agent/request_safety_classifier.py`, `AgentRequestSafetyClassifier` before calling Planner identifies the following types of requests:

- Require implementation or bypassing of stop-and-break, safety-lock and protective devices.
- Remote control of the real equipment, job release or modification of the equipment state is required.
- Ask for access to external links that appear in pictures, 2D codes or logs.
- Requires the execution of an order, Prompt or script in a picture or untrustworthy text.
- Other high-risk control intentions must be confirmed by competent and qualified personnel.

The generic diagnostic request returned to `None`, indicating that the safe classification was not stopped earlier and that subsequent use of tools and Agent Loop was still required. High-risk requests for return to `human_review_required` decision-making, with no exposure tool to Planner and stable public statements and missing labour.

Call chain:

```text
AgentDiagnosisService (read user intent)
  - >AgentRequestSafetyClassifier.classify() (Current module of this section: Identification of high-risk intent for certainty)
  - >High risk: Direct construction human_review_required response
  - > General request: continue with AgentToolPolicy
```

The Python rule is used for the security classification rather than simply for Prompt, because Prompt is softly restrained and may be misunderstood by models or disturbed by untrustworthy photographic text; the code branch is able to determine with stability which tools it does not call and at which level they end.

## 6. Task 3: Evidence-driven Agent progress state

### 6.1 `AgentProgress`. What's the record?

`app/schemas/agent_progress.py` Define three core categories of objects:

- `AgentToolProgressRecord`: What capacity is served by a single tool, what status, what new sources or evidence are generated.
- `AgentEvidenceConflict`: Document conflicting sources and conflict statements, avoiding consolidating contradictory evidence into conclusive conclusions.
- `AgentProgress`: Maintenance of current status, required capacity, completed capability, tool logs, confirmed sources, covered evidence, conflicts, missing information, next permitted tools and reasons for discontinuation.

`AgentProgress` is the business state of Agent Loop, not a free text memory for LLM. It answers three questions: what has been done, what is missing and what the next step is permitted to do at the most.

Call chain:

```text
AgentToolPolicy (generations required_capabilities and allowed_tool_names)
  -> AgentProgress (Current module of this section: Save verifiable task progress)
  - > AgentRunner (per round of current progress to limit tools and end of judgement)
  - > AgentDiagnosisService (translating the end state into a public diagnostic syntax)
```

### 6.2 Reducer. Why exists?

Of `app/agent/progress_reducer.py`, `AgentProgressReducer` receives " Old Progress + New Event" and returns a new non-changeable progress object. Reducer is responsible for:

- Could not close temporary folder:%s
- Generates a standardized tool to call a signature and to prevent a complete re-call.
- Extracts the capability completion, source ID and evidence ID from the results of the successful tool.
- Record empty results, failures, conflicts and missing information.
- Remove tools that are no longer required after the capacity has been completed.
- It is determined that the conditions for completion have been met, that only part of the result has to be returned, or that the answer must be refused or manually reviewed.

Call chain:

```text
AgentRunner (received Planner decision or tool result)
  -> AgentProgressReducer (Current module in this section: Verify conversion and generate the following state)
  - > AgentRunner (use allowed_next_tool_names to continue or stop)
```

Reducer values the same state conversion of the same input and tool results. Planner may make a recommendation but cannot declare himself that a particular evidence has been obtained, that a security requirement has been met or that a failure can be ignored.

## 7. Mission 4: 30 scenes returned offline

### The role of 7.1 Fixture

`data/eval/multimodal_offline_fixtures.jsonl` Provide fixed Planner decisions, Vision observations and tool results for 30 Gold scenes. Fixture does not forge a single online accuracy rate, but replaces an external probability service with a repertoireable input, thus independently authenticating the system framework.

Offline regression does not visit real LLM, Vision, Embedding, ChromaDB, telemetry platforms or equipment. It confirms that:

- Correct scope of request-level tool.
- Planner decides whether Runner and Execut will be executed or intercepted correctly.
- Whether the progress status progresses according to the results of the tool.
- Consistency of termination causes, diagnostic status and missing information.
- Visual observation, reference and correct transmission of source labels.
- Whether the security scene remains closed or manually audited.

### 7.2 Load, pair and execute

`app/services/multimodal_offline_fixture_loader.py` Load Fixture and match `scenario_id` with Gold; duplicate, missing or redundant IDs can lead to verification failure.

Call chain:

```text
Evaluate scripts (read Gold and Fixture paths)
  -> pair_multimodal_offline_regression_cases() (Current module of this section: one pair and verification of integrity)
  - > MultimodalOfflineScenarioExecutor (execution of a single certainty scenario)
```

`app/services/multimodal_offline_scenario_executor.py` Create Fake Planner, Fake Vision and controlled Fake tools for individual scenes, then call on real strategies, Runner, Excelctor, report construction and rating links.

Call chain:

```text
MultimodalOfflineRegressionService (through paired scenes)
  - >MultimodalOfflineScenarioExecutor.execute() (Current module of this section: Injecting Fixture and running the real framework)
  -> AgentDiagnosisService / AgentRunner / ToolExecutor (true code of the sorting)
  ->Single scene response, trajectory and Foxture consumption audit
```

`app/services/multimodal_offline_regression_service.py` batch execution of 30 scenarios, handed over to established scorers for calculation indicators, generating a summary of threshold values, failure scenes and 30 redacted tracks.

Call chain:

```text
scripts/evaluate_multimodal_offline_regression.py (command line entrance)
  - >MultimodalOfflineRegressionService.run() (Current module of this section: Batch execution, rating and aggregation)
  - >JSON / Markdown Report and Redacted Track
```

### 7.3 Formal results and remaining failures

The official report is `docs/en_multimodal-offline-regression.md`. The 26 scenes were strictly passed, and the failure scenes were:

- `multimodal-agent-008`
- `multimodal-agent-012`
- `multimodal-agent-019`
- `multimodal-agent-029`

All four failures relate to the expectation that the Vision tool is not allowed or implemented, resulting in visual status, source labeling, tool sequencing or end state not being consistent with Gold; Of these, 008 and 029 also relate to the requirement of qualified personnel. They are known gaps in the express reservation, without prejudice to the conclusions reached at the threshold of 24/30 this week.

Offline 0.867 cannot be equal to the accuracy rate on the real model line. It demonstrates that framework behaviour is stable once the same set of external decisions and tool results are in the system; Probability, delay, restricted flow and output fluctuations for real Planner, Vision and upstream API still require separate sampling.

## 8. Task 5: Diagnosis sessions and running trajectory

### 8.1 Session Contract

`app/schemas/diagnostic_session.py` Definition:

- `DiagnosticSessionRequestSummary`: Summary of redacted requests.
- `DiagnosticSessionDiagnosisSnapshot`: Diagnosis status, risk and missing information snapshot.
- `DiagnosticSessionEvidenceCoverage`: Number of references, sources and evidence coverage.
- `DiagnosticSessionMetrics`: Operational indicators such as total time-consuming, number of steps in the tool, failure tool, etc.
- `DiagnosticSessionRecord`: A complete redacted session that can be sustained.
- `DiagnosticSessionSummary`: A compact summary in the list of recent sessions.
- `DiagnosticSessionListResponse`: A page break or a limited list response.

Full pictures, API Key, complete sensitive logs, full tool parameters and model private thought chains do not belong to these contracts and therefore are not unexpectedly written into session files because of object sequence.

### 8.2 Construct and Storage

Of `app/services/diagnostic_session_builder.py`, `DiagnosticSessionBuilder` constructs the session record from diagnostic results that have been verified through a public response contract. It compresses user input, calculation of evidence coverage and tool indicators and maps business status to stable session status.

Call chain:

```text
AgentDiagnosisService (acquire final public response and implementation trajectory)
  - >DiagnosticSessionBuilder.build() (Current module of this section: Generating redacted sessions)
  - >DiagnosticSessionStore.save() (continuing)
```

Of `app/services/diagnostic_session_store.py`, `DiagnosticSessionStore` is responsible for creating storage directories, verifying `session_id`, writing atoms to JSON, reading and listing recent sessions by ID. Atoms write a temporary file before replacing the target file, avoiding the interruption of process leaving half a JSON.

Call chain:

```text
DiagnosticSessionBuilder (generation of verified records)
  ->DiagnosticSessionStore (current module of this section: save, read by id, list the most recent sessions)
  - > Diagnostic Sessions Router / Web Console (read public fields only)
```

### 8.3 Query API

`app/routers/diagnostic_sessions.py` Provide:

- `GET /api/v1/diagnostic-sessions`: Query for summary of recent sessions.
- `GET /api/v1/diagnostic-sessions/{session_id}`: Search for specified session details.

Call chain:

```text
Client or Web Console
  - > Diagnostic Statements Router (current module of this section: verify parameters and call memory)
  -> DiagnosticSessionStore
  - > DiagnosticSessionListResponse or DiagnosticSessionRecord
```

The non-existent session returns a stable 404 business error and does not disclose the host path.

## 9. Task 6: Web Console and Docker

### 9.1 Light control counter

`app/web/index.html`, `app/web/styles.css` and `app/web/app.js` constitute light control tables that do not rely on front-end construction tools. Page support:

- Enter robot numbering, failure phenomena, redacted logs and optional mission targets.
- Select and submit up to three pictures.
- Demonstrate the state of diagnosis, possible causes, next steps, risks and missing information.
- Demonstrating knowledge base references, visual observations, simulation telemetry, draft testing and tool trace.
- Query and open the recently saved redacted diagnostic session.

Call chain:

```text
App.js in browser (current module of this section: collect input and call public API)
  -> POST /api/v1/agent/diagnose
  -> GET /api/v1/diagnostic-sessions
  -> Page Component (presentation of results and tracks by source)
```

The console is a teaching demonstration and miscalculation portal, which does not contain identification, tenant isolation or equipment control buttons.

### 9.2 Packaging Configuration

- `Dockerfile` Install application dependent, Tesseract and Chinese English language data using Python 3.11 slim base mirrors and run Uvicorn in non root `appuser`.
- `compose.yaml` Exposure `8000` port, configuration of health check and binding of the host Shroma data and redacted session catalogue to the container.
- `.dockerignore` Excludes `.env`, testing, documentation, raw language, cache, croma data and local sessions, reduces context construction and avoids the reproduction of sensitive documents into mirrors.

Call chain:

```text
docker compose up --detach --build
  - > Dockerfile (Current module of this section: construction of a runable application mirror)
  - >compose.yaml (start service, port, health check and permanent mount)
  -> Uvicorn / FastAPI
  >/health,/console/ and /api/v1/*
```

Local container validation results:

- Mirror: `robotops-copilot:week6`.
- Compose Service Health, `8000:8000` port map normal.
- `/health` and `/console/` returned 200.
- The container can read 355 Chunk from the `robot_knowledge_week3_vector_baseline` Collection.
- A photo diagnostic request can complete the Vision tool call and generate traceable sessions.
- Sessions are stored by binding mount to the host directory and can be viewed in the console's recent session.

A demonstration screenshot can be found in `docs/robotops-copilot-docker-demo.png`.

## 10. Test design and final results

Week 6 tests do not only validate that objects can be imported, but cover the following anticipated processes:

| Test Range | Test Method | Expected process of tested module | Expected results |
|---|---|---|---|
| Tool policy compacts | Construct valid and illegal Pydantic objects | Enter a policy field - > cross-field verification - > freeze model | Stable sequence of valid objects, contradictory fields rejected |
| Request Level Tool Policy | Construct pictures, knowledge, telemetry, drafts and negative expression requests | Normalization Request - > Identification Capability - > Map Minimum Tool | Only the tools needed to accomplish the mission are exposed |
| Safety Catalogue | Parametered high-risk, alert injection and generic diagnostic text | Identification of unneglected dangerous intent - > Generating termination decision | High-risk access human review, general diagnosis continued. |
| Agent Progress Compact | Construct different states and collections of evidence | Verification of status, records, sources, conflicts and scope of tools | Inconsistencies are not allowed into Runner. |
| Progress Reducer | Enter initial progress and continuous tool results | Validation tool - > Merge new information - > Recostability and Next | Repeated or ultra vires tools prevented, progress certainty advanced |
| Offline Fixture | Load 30 JSONL | Schema Verify - ID to Heavy - pair with Gold | Missing, repeated or redundant Fixture rejected |
| Offline single scene execution | Fake Planner, Fake Vision and Fake Tool | Run real policy, Runner, Execuator and reporting chain | The complete trajectory can also be repeated without access to external services. |
| Offline batch returns | Implementation of 30 pairing scenarios | Single scene execution - > Rating - > Threshold - > Reporting | At least 24/30 and no drop in security. |
| Session Builda | Enter completed, partially rejected and failed responses | Compress the Open Field - > Calculate Indicators and Overwrite - > Construct Record | Do not write pictures, keys, complete logs or think chains |
| Session | Use pytest temporary directory | Atomic Save - > Press ID to get - > Recent List | Data can be restored, illegal ID and missing session stable failed |
| Session API | FastAPI Reliance over and across clients | Router-Store-Schema Response | List and details return correct contract, missing 404 |
| Web Console | TestClient Request pages and static resources | FastAPI Static Mount - > HTML/CSS/JS | `/console/` accessible, unknown resource returns 404 |
| Docker Configuration | Build, Compose Start, Health & End-to-end Request | Mirror - > Container - > API - > Session Mount | Clean operating environment restores the main link. |

Finally, the JUnit report is `docs/pytest-results-week6-final.xml`:

```text
tests    : 2514
failures : 0
errors   : 0
skipped  : 0
```

A Week 6 relying assembly test was found and amended during the testing run: Production function `get_agent_diagnosis_service()` has been added `session_builder` and `session_store`, while the old test is still called by Week 5. The modified position is `tests/test_agent_tool_dependencies.py`, and the test now uses `tmp_path` to create an isolated `DiagnosticSessionStore`, to bring two new dependencies into the plant and assert that the service remains the same. This amendment ensures that the test covers the current real assembly process, not lowers the production function requirements.

## 11. Uploaded GitHub delivery boundary

Shall submit:

- Source code and test in `app/`, `scripts/`, `tests/`.
- `data/eval/` of Gold and Offline Fixture.
- Public teaching pictures in `data/multimodal/`.
- Open simulator in `samples/corpus/`.
- `README.md`, `.env.example`, `.gitignore`, `.dockerignore`, `Dockerfile` and `compose.yaml`.
- `docs/en_engineering-summary_week6.md`, offline return Markdown, redacted session sample, Docker demonstration cut-off and Week 6 final JUnit report.

No submission shall be made:

- `.env` and any real keys, certificates or documents.
- `要求.txt`, `week6.txt` and acceptance feedback.
- `chroma_data/`, `data/diagnostic_sessions/` and other local operational status.
- Original PDF in `data/source/` or information that may be restricted by the authorization.
- `__pycache__/`, `.pytest_cache/`, temporary directory, log and editor configuration.
- Unsensitized JSON tracks or reports containing complete user logs, pictures Base64, host privacy information.

## 12. Known Limitations

- 0.867 is the rate of regression of the system under fixed Fixture and is not a commitment to the true LLM or Vision accuracy rate.
- Four offline failure scenes still have inconsistent Vision tool ranges, state mapping or human review syntax.
- Planner and Vision Project are still dependent on external services and may be affected by output fluctuations, time overruns, flow restrictions and compatibility.
- The current session storage is a local JSON file that is suitable for a single-machine teaching demonstration and does not provide database services, multiple examples, or long-term archiving capability.
- The Web Console does not have an identification, access control, tenant segregation, CSRF protection or a production front end construction process.
- Compose is a locally recapable configuration that is not equivalent to production deployment; the production environment still requires key management, TLS, reverse agent, monitoring, backup and resource limitations.
- The current tools are read-only or draft-generated, and the system does not control robots, bypass safety devices or auto-manufacturing maintenance and repositioning.
- The diagnostic output cannot be a substitute for the original plant information, on-site risk assessment and certification by qualified personnel.

## 13. Final Conclusion

Week 6 has completed the planned minimum-level tool strategy, the SDS, evidence-driven progress status, 30 scene offline regression, redacted sessions, lightweight console and Docker startup. The system evolved from “dependent Planner’s approach to deciding all steps” to “Planner makes limited decisions in Python privileges and status machines” and has the capability to repeat regression, fail location, session tracking and clean environmental demonstration.

It has reached the learning and demonstration target of RobotOps Copilot Beta this week, but it is still a read-only teaching project, not a production of robotic control systems or safety certification products.
