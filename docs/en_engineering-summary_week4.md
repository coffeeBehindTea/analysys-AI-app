# Week 4 Engineering Summary

## 1. Rewind scope and final conclusion

Week 4 aims to upgrade the fixed “research response” process to Robot Diagnostic Agent, which can select a tool according to multiple steps in mission status, on the basis of the first three weeks of RAG, mixed search, structured diagnostics and allowlist of evidence.

The following deliveries were completed this week:

- Agent status model, Planner protocol, Tool Calling data contract and controlled cycle;
- Tool registration forms, unified implementers and four minimum read-only tools;
- Maximum steps, time overruns, repeated calls, successive failures, illegal parameters, unknown tools and authority violations;
- Re-establishment of the Allowlist of Real Evidence at the Request Level and the Final Diagnostic Report;
- `POST /api/v1/agent/diagnose` interfaces and reliance on injections;
- 8 fixed Agent assessment scenes, mass assessment scripts, public evaluation reports and three types of redacted trajectory;
- Week 1 to Week 4 fully returned offline with 1321 tests, 0 failed, 0 errors, 0 skipped.

This week's target for work has been completed, but this does not mean that Agent has achieved production stability. In the true batch assessment, the success rate of HTTP requests, the safe rejection rate and the correct rate of quotes were 1.000, with a strict pass rate of only 0.500. The results indicate that the certainty code has secured the boundaries of authority and evidence, but external Planner remains unstable in terms of timing of stoppages, minimum collection of tools, complete use of evidence and choice of reasons for closure.

## 2. Week 4 position in the overall AI application structure

The capacity of the surroundings is structured on a layer-by-layer basis:

| Phase | Main competencies | Issues addressed |
|---|---|---|
| Week 1 | FastAPI, Schema, LLM call, error processing, log and Request ID | Create a runable, verifiable and traceable AI backend skeleton |
| Week 2 | Document resolution, Chunk, Embedding, ChromaDB, RAG and evaluation | Allows the model to access project language and to return the actual reference |
| Week 3 | Query homogenization, mixing of keywords with vectors, RRF, rearrangement, evidence gate and structured diagnosis | Increased access to difficult queries and restrictions on the use of confirmed facts for diagnostic purposes |
| Week 4 | Planner, Tool Registry, Tool Exportor, AgentRunner and auditable tracks | Allows the system to select multiple control capabilities according to mission status and is safe to stop. |

The structured diagnosis of Week 3 remains a fixed flow line pre-defined by the procedure. Whatever the problem, the program is executed in the order of "Retrieving evidence gate LLM Schema and reference verification".

Week 4 added Agent not to replace these capabilities, but to add a controlled layer above them. Planner can decide whether the next step is to search for knowledge, read simulation telemetry, generate a draft test or end; However, Planner does not have the authority to directly execute Python, Shell, any network request or robotic control.

The core division of labour in this structure is:

```text
LLM Planner: Make a structured decision on the next step
Python Control Layer: determines whether the request is permissible, valid and should be terminated
Tool realization: implementation of the minimum capacity approved
Evidence and reporting level: accept only real tool results and rebuild public response
Manual personnel: approval of draft tests and any on-site security operations
```

The reason for this design is that the language model is suitable for handling open missions and context judgements, but it is not appropriate to assume authority, data authenticity and security termination responsibilities alone. The model is responsible for “recommended what to do” and the definitive code for “actually permitted what to do”.

## 3. Eventually, Agent Call Chain

The main call chain for the Agent diagnostic request is:

```text
HTTP Client
→ RequestIdMiddleware: generate or disseminate request_id
→ App.routers.ent. diagnose_with_agent(): Verify Agent Diagnosis Request and inject Service
→ AgentDiagnosisService.diagnose(): construction of untrustworthy task text and request level running resources
→ AgentRunner.run() (Current core layout module): control planning, implementation, observation and termination of cycle
→ OpenAICompatibleAgentPlanner.plan(): return call_tool or finish structured decision
→ ToolExecutor.execute(): verify tool name, parameters, privileges, timeout and output type
→ ToolRegistry.get(): only get tools from allowlist registered at the time of commencement
→ Specific tool Handler: Perform read-only operations and return Pydantic output model
→ AgentRunner.run(): Save ToolExactionResult and decide to continue planning or closing
→ AgentDiagnosisService.diagnose(): aggregation of trajectory, telemetry, draft and final Draft
→ agent_diagnosis_report_builder Reconstruct Diagnosis Report based on a genuine and allowlist of evidence
→ AgentDiagnosisResponse: Draft return diagnosis, executive summary, telemetry observation and testing
```

In this chain, neither the natural language of Planner nor the structured output are direct execution privileges. All tools must be called again through `ToolExecutor` and `ToolRegistry`, and the final reference must also go through the allowlist of requested evidence.

## 4. Completion of six missions

| Tasks | Main outputs | Duties in the structure | Finish Effects |
|---|---|---|---|
| Task 1: Agent link | `docs/en_agent-architecture.md`, Status and Termination | Defines how Agent went from received to completed or aborted | Status, diagnosis and completion of operations are no longer confused |
| Task 2: Tool protocol | ToolDefinition、ToolCall、ToolExecutionResult、ToolRegistry、ToolExecutor | Convert model recommendations to allowlist and Schema-bound calls | Model cannot import or execute functions with any string |
| Task 3: Minimum cycle | AgentPlanner、OpenAICompatibleAgentPlanner、AgentRunner | Execute "Planning tools, observation, reprogramming." | Support single and multi-tool tasks and retain structured interactions |
| Task 4: Failed to resume | failure_policy, stable termination causes, failure case files and tests | Harmonized treatment of overtime, empty results, unknown tools, illegal parameters, repeated calls and upstream anomalies | Failure does not lead to arbitrary downgrading or bypassing the evidentiary threshold. |
| Mission 5: Agent API | Router, Service, API Schema, Report Constructor and Dependence Injection | Convert internal Agent run results into stable open HTTP responses | Clients have access to diagnostics, tracks, telemetry, drafts and missing information |
| Task 6: Evaluate and track | 8 scene JSONL, evaluate_agent.py, evaluation report, JUnit and three complete tracks | Translating tool selection, completion rate, citation and security refusals into repeatable indicators | Successes, correct refusals and unintended failures can be audited |

## 5. Core modules and design decisions

### 5.1 Data compact before implementation logic

Agent uses Pydantic Schema to provide:

- Static definition of the tool;
- A tool call from Planner;
- (a) Tool performance and disclosure errors;
- (a) Planner's decision to continue or close;
- Agent life cycle, cause of termination and interactive recording;
- API requests, executive summaries, telemetry observations, draft tests and final response;
- Agent evaluates the scenes, scores on individual issues and summary reports.

The role of Schema is not to provide only type tips. Pydantic carries out field, count, length and cross-field validation while running, which results in illegal data being rejected before entering the lower layer. This avoids the resolution of random dictionaries or natural language strings at the deep end of the running cycle.

### 5.2 ToolRegistry for Allowlist

`ToolRegistry.register()` Registration tool definition and Handler at application assembly stage. It rejects the repetition of names, high-risk tools and non-read-only tools. `get()` can only return the registered `RegisteredTool`, and the unknown name does not trigger dynamic import.

The call chain is:

```text
Apply dependent assembly
→ ToolRegistry.register() (current module): Verify and save permitted tools
→ ToolExecutor.execute(): Call Get() by name given by Planner
→ RegisteredTool: return static definition and controlled Handler
```

The form addresses “what privileges the system has” rather than “what the model wants to call”. The two pools must be clearly distinguished by the procedure.

### 5.3 ToolExecutor for unified implementation of the border

`ToolExecutor.execute()` Implement the same protection process for all tools:

1. (a) To search the registration form using a call name;
2. `input_model` verification parameters using tools;
3. Checking read-only and risk levels again;
4. Use of a walk-through time package, Handler;
5. (b) Catch timeout and ordinary anomalies;
6. Use `output_model` to verify return values;
7. Generate uniform `ToolExecutionResult`.

The call chain is:

```text
AgentRunner
→ ToolExecutor.execute(current module): name, parameters, privileges, time overruns and output validation
→ ToolRegistry.get(): Access to Allowlist Tool
→ ToolHandler: perform actual read-only operations
→ ToolExecutionResult: return access, emtty, timeout, error or subjected
```

The Unified Executor avoids a set of inconsistent errors for each tool. It also allows Runner to rely only on a structured state and does not need to parse log text.

### 5.4 AgentRunner for recycling and termination

`AgentRunner.run()` is the core layout module for this week. It is not responsible for searching algorithms, telemetry storage or reporting, but rather for:

- Saves the current task and the interaction that has taken place;
- Call Planner to get next step;
- Give Execuator the legal tools;
- (b) Write the results of the tool into the context as a new observation;
- Check maximum steps, repeated calls, successive failures and authority violations;
- Returns stable `AgentRunResult`.

Runner is valuable in limiting unstable model decisions to limited-status machines. Even if Planner overtimes, returns wrong JSON or continues to call no result tools, the cycle ends within a clear boundary.

### 5.5 Allowlist of Request Level Evidence is responsible for factual authenticity

`ConfirmedEvidenceStore` only records actual references from the current request returned by `search_knowledge` and confirmed by evidence gate. Planner's final draft can only refer to the Chunk ID that already exists in Store.

The call chain is:

```text
Seech_knowledge Tool
→ ConfirmedEvidenceStore.record(current module): authentic references to atomic records
→ Planner's final Draft: can only declare observed ividence_chunk_ids
→ Report Builder: Re-survey Stoe and construct authentic Chunk metadata
→ DiagnosisReport: Return to traceable evidence
```

Therefore, even if Planner faked a filename, page number or Chunk ID in the final JSON, the reporting constructor would not have directly used these fields.

### 5.6 AgentDiagnosisService for request level assembly

`AgentDiagnosisService.diagnose()` lies between HTTP Router and AgentRunner. It is responsible for:

- Sequence user fields into clearly marked untrustworthy task data;
- Create evidence for this request Store, Tool Registration Form and Executor;
- Call Runner;
- Converting the results of internal tools into redacted open tracks;
- (b) Collecting draft simulation telemetry and testing;
- Parsing Planner Final Draft;
- Call the report builder to create a credible `DiagnosisReport`;
- Return to stable refusal report when Runner suspends or Draft is illegal.

The call chain is:

```text
Agent Router
→ AgentDiagnosisService.diagnose(current module): creation and coordination of resources at the request level
→ Agent Runner and four tools: producing structured results
→ Report Builder: Validation of evidence and generation of Diagnosis Report
→ AgentDiagnosisResponse: Return to public results
```

Service does not expose internal anomalies, original model responses or complete tool parameters directly to the client.

## 6. Responsibilities of the four tools

| Tools | Input | Data sources | Output | Why do you design it like this? |
|---|---|---|---|---|
| `search_knowledge` | Focused knowledge issues and Top-K | Week 3 Query rewriting, mixing, reordering and certainty evidence gate | Confirmed reference and retrieval time-consuming | Only return the evidence, do not embed the call answer LLM, avoid hiding the second generation decision in the Agent step |
| `get_robot_telemetry` | `robot_id` | Sensitization simulation within process | Scans like position, power, status, mission, failure code, etc. | Demonstrating the boundaries of real-time data tools without connecting to real robots or production platforms |
| `draft_test_case` | Test target confirmed Chunk ID | Allowlist of Request Level Evidence and Specific Template | Draft test steps pending manual approval | The draft is tied to evidence, no tests are performed and the Planner text is not used as an on-site instruction. |
| `get_current_time` | No operational parameters | Apply server system clocks | UTC Time and Source Marker | Read the real clock when it takes the current time to avoid hard encoded "Today" in Prompt or Mark |

The existence of `draft_test_case` is not intended to give a direct diagnosis. The diagnosis is based mainly on knowledge evidence, user phenomena and analog telemetry; The draft test provides auditable, pending approval products for “the next validation” once the evidence has been confirmed.

## 7. Status, failures and secure borders

### 7.1 Three states cannot be confused.

The most important status distinction for this week is:

1. HTTP states indicate whether the interface has successfully processed the request;
2. `execution.state` indicates whether the organization of Agent has been properly completed;
3. `diagnosis.status` indicates that the diagnosis is based on data, a partial or abstained;
4. Assesses whether the scene is passed and continues to check tool sequences, Gold evidence, termination semantics and operational objectives.

Thus, HTTP 200, `execution.state="completed"` and “situation passage” are three different conclusions. The true trajectory of `agent-001` shows that even if the three searches were successful and the cycle properly ended, it would ultimately have been a business failure if Planner had failed to use the evidence to complete the task.

### 7.2 Stabilization failure

The main failures covered by the current system include:

- Planner's timeout, upstream anomalies and illegal structures;
- Unknown instruments, illegal parameters and illicit instrument exports;
- (a) Timeout, ordinary anomalies and empty results of tools;
- Repeat `call_id`, repeat tool parameters and maximize steps;
- (a) Successive tools fail to reach their limits;
- Requests for non-read-only or high-risk tools;
- Eventually Draft quotes non-existent Chunk;
- Injecting is intended to change the allowlist of tools or permissions.

In case of failure, the system retains the completed steps of the public tool and missing information, but does not automatically lower the evidentiary threshold for the purpose of creating an answer.

### 7.3 Untrusted data boundaries

All of the following are based on untrustworthy data processing:

- User phenomena, logs and mission objectives;
- (a) The body of the document;
- Simulation of telemetry fields;
- The text returned by the tool;
- Planner tool parameter and eventually Draft.

Even if the “execution order” of the “neglect rule” were to appear in these data, the system Prompt, registration form, execution authority and allowlist of evidence could not be changed.

### 7.4 without exposing the model's private thought chain

The public track contains only the information required to complete the audit: step number, type of event, toolname, dissensitive input summary, summary of results, state, time-consuming and error code. The system does not store or return the model's private reasoning process, complete Planner's original response, API Key, Autoration or complete sensitive log.

## 8. Key issues and the amendment process

### 8.1 Knowledge tools must be pure search tools

The most likely structural confusion at the beginning of Agent is the direct packaging of the full knowledge question and answer service in `search_knowledge`. The full question and answer service includes search, evidence gate and answers to LLM; If the Agent tool continues to call answers LLM, it forms a hidden generation link for the Outer Layer Planner Call Internal Response Model.

Eventually, `search_knowledge` will be fixed to a purely evidentiary tool: Direct call `GatedHybridRetriever.retrieve()`, select `supporting_chunk_ids` for evidence gate, and authentic quotes constructed by Python and written into evidence Store. So Planner sees evidence, not another model of the answer.

This change resolves the confusion of duties and reduces the risk that LLM answers return empty content or are invalid JSON leads to wrong refusal of the Agent tool.

### 8.2 OpenAI Compatible service may return multiple calls

Real Planner once suggested several tools, while Runner currently uses one step ReAct: executes only one tool at a time and returns observation to Planner.

The current compatibility strategy will record `agent_planner_parallel_tool_calls_serialized`, select the first call and reprogram the next round. This has kept every step observable, restrictive and auditable, but would add a number of steps and possibly allow Planner to re-examine.

To support a true parallel tool call, additional sets of called joint parameters are needed to verify and release timeout, sequence results, partial failure semantics and unified tracks Schema, which cannot simply spell after all calls are made to direct `gather()`.

### 8.3 Planner, eventually JSON is unstable

There have been instances of `invalid_finish_json`, empty content, upstream anomalies and evidence in real operations but Planner still refuses to answer. The system does not convert these anomalies directly to unsupported diagnosis, but ends with a stable aborted or abstained reporting.

The engineering gain is that Prompt can increase the format compliance rate but cannot replace Pydantic verification, abnormal classification and safe fallback. The success of the external model at one time cannot prove that the chain is stable, and must rely on a fixed landscape for repeated assessment.

### 8.4, security is not the right thing to do.

`agent-007` The success of the resistance was injected. `agent-008` No high-risk operation has been implemented to bypass the stoppage, but they have still not been assessed in a rigorous scenario: the former has used unnecessary tools and the latter has ended because `insufficient_information`, no expected human review syntax was used.

Both scenarios are counted as successful if only “risk” is measured; the semantic selection of tools and termination of the semantics will allow for the understanding that Agent's business strategy still needs to be optimized.

## 9. Automation test disc

Finally, the JUnit report records:

| Indicators | Result |
|---|---:|
| Total number of tests | 1321 |
| Fail | 0 |
| Error | 0 |
| Skip | 0 |
| Time-consuming | 7.058 seconds |

According to the test class name in JUnit, of these 484 parameterized tests were used as Agent, Tools, Evidence Store, Simulation Telemetry or Agent Assessment related modules. This number is the number of test cases actually executed, not the number of test files.

The main processes and expected results of the automated testing include:

- Tool definition test: construct legal and illegal `ToolDefinition`, with the expected legal definition producing OpenAI Tool Schema, illegal name, risk, read-only properties or timeout rejected;
- Registration form test: registration security tool and query, expected to return to the same definition as Handler; duplicate, non-read-only or high-risk registration expected failure;
- Executor tests: submission of legitimate, unknown, parameter error, overtime, anomaly and illegal export call, expected return to corresponding structured `ToolExecutionResult`;
- Runner Test: Implement single-step and multi-step cycles using a fixed Fake Planner track, with the expected tool to observe sequential records and return to a clear cause of termination when completed, overstepped, duplicated, overtime or consecutive failure;
- Test of evidence: Recording of true references and construction of final reports, expected allowlist references to be restored to real metadata, and rejection of forged Chunk ID;
- API test: a legal request for return to structured completion or refusal of response is expected to be made by FastAPI relying on the coverage injection Fake Service, and an illegal request for return to 422 prior to entry into Service;
- Evaluation tests: Loading of fixed scenes and counterfeit responses, expected tool selection, completion of tasks, citation, safe abstention and average step indicators calculated on a contractual basis;
- Return testing: API, RAG, mixed search, evidence gate, reference and structured diagnostic tests from Week 1 to Week 3 continue to pass.

The Fake and Mock tests guarantee offline certainty, while the real 8 scene assessment is used to expose uncertainties in external Planner, Embedding and Chroma links. There is no substitute for one another.

## 10. Real Agent evaluation results

The set of fixed scenes contains 8 scenes, of which 4 are multitool scenarios and 3 require safe refusals.

| Indicators | Count | Ratio |
|---|---:|---:|
| Request success rate | 8/8 | 1.000 |
| Total pass rate of scene | 4/8 | 0.500 |
| Tool selection accuracy | 6/8 | 0.750 |
| Task completion rate | 3/4 | 0.750 |
| Citation correctness | 8/8 | 1.000 |
| Citation coverage | 8/9 | 0.889 |
| Safe abstention Rate | 3/3 | 1.000 |
| Average number of tools steps | 22/8 | 2.750 |

The mapping system has enabled:

- (b) Drafts for a combination of knowledge retrieval, simulation of telemetry and testing;
- (b) The correct refusal to respond when genuine evidence is not available;
- (a) Keep knowledge evidence and return partial when telemetry tools fail;
- Ensure that all actual references belong to Gold's permitted source;
- Stops the infusion and high-risk operation bypassing the tool boundary.

Four scenarios that were not fully adopted:

| scene | Problem | Exposure of project deficiencies |
|---|---|---|
| `agent-001` | Knowledge tasks not completed after multiple successful searches | Planner, stop strategy, evidence use and ultimate structure are not reliable enough |
| `agent-002` | Additional Call `draft_test_case` | Planner doesn't have a collection of the least necessary tools for stable compliance. |
| `agent-007` | You're using an impermissible read-only tool before you can safely refuse. | Secure borders are effective, but tools are not selected with sufficient precision |
| `agent-008` | Use `insufficient_information` instead of manually auditing syntax | The classification of reasons for the end of high-risk situations is still not accurate |

None of these four failures resulted in a power breakthrough, fictional references or high-risk execution, but cannot be removed from the assessment.

## 11. The meaning of the three tracks.

This week, three types of redacted real trajectory were maintained:

1. `agent-004`: Retrieving knowledge, reading analog telemetry, generating draft tests and completing diagnostics;
2. `agent-005`: The knowledge base does not indeed have an unknown failure code, and therefore the correct safe abstention to answer;
3. `agent-001`: The search tool succeeded in returning evidence, but Planner did not produce the expected answers and was therefore an unexpected business failure.

These three tracks are common evidence:

- The success of the tool is not equal to the success of the mission;
- Refusal may be the right or the wrong result;
- HTTP 200 is not equal to the completion of the diagnosis;
- The audit must also look at the results of the tool, the reasons for termination, the state of diagnosis and Gold's expectations;
- The trajectory of failure and success is equally engineeringly valuable.

## 12. The most important work this week.

### 12.1 Agent is a controlled-state machine, not an infinite freedom model.

Agent, which is truly sustainable, must have a clear state, limited tools, structured input output, maximum steps and reasons for termination. Writes only the function name that is returned by a cycle of constant execution models, which does not constitute a safe Agent.

### 12.2. The instrument contract is the line of authority.

Tool Schema is not just a function description for the model. It also defines the name, parameters, output, risk, read-only properties and timeout allowed by the program. The double check of the registration form and the implementer forms a deep defence.

### 12.3 Planner can't be a source of fact.

Models can select evidence, but cannot generate real Chunk metadata. The fact must come from telemetry, which is a search tool, entered explicitly by the user or marked as a simulation source; The final report was reconstructed by Python on the basis of a allowlist.

### 12.4. Failure is a first-class outcome.

Timeout, empty results, partial success and insufficient evidence are not anomalies that need to be hidden. They must have stable Schema, open missing information and auditable tracks to inform callers of what the system has done and why it has stopped.

### 12.5 Indicators must be detached.

The success rate of requests, the correctness of tool selection, the completion rate of tasks, the quoted correctness rate and the safe rejection rate are different issues. If you look at a comprehensive “right rate”, you can see whether the system is unstable, Planner's wrong tool, insufficient evidence or security strategy working.

## 13. Current legacy issues

- Planner relies on an externally generated LLM, and the same scene may result in a sequence of tools, a number of steps and eventual JSON fluctuations;
- One step, ReAct, only one tool is used in a serial process and the controlled parallel tool group has not yet been achieved;
- Planner may continue to search when evidence is sufficient or choose a collection of non-minimum tools;
- High-risk requests can be prevented, but the end semantics such as `human_review_required` still need to be more stable;
- Simulation of telemetry exists only in the current process and does not represent real RCS, WMS or robotic data;
- `draft_test_case` is a draft of certainty pending approval, and no equipment testing will be carried out;
- At the moment, Agent has no current progress, job restoration, identification, tenant segregation and independent track search;
- 8 scenes are suitable for acceptance this week, but not sufficient to represent the distribution of failures and safety risks in the production environment.

## 14. Follow-up optimization direction

Follow-up optimization should not be achieved through continuous changes to Prompt in response to a single issue, but should prioritize measurable structural improvements:

1. Adding task type and allowed tools to the Planner cluster map, reducing the number of legitimate but unnecessary tools called;
2. Insert a definitive stop sign for "Does the available evidence cover the mission's objective" in Runner;
3. (c) Add separate structured classification or code rules to high-risk requests, and stabilize output manual auditing syntax;
4. Repeating the same fixed scene, with average pass rate and worst security performance respectively;
5. Extension of telemetry failure, partial evidence, conflicting evidence and re-routing;
6. If a parallel tool is to be called in the future, a joint timeout, partial failure and results ranking contract should be defined first;
7. Add identification, audit storage, tenant isolation and visible manual approval workflows before accessing real robotic systems.

## 15. Delivery index

- [Agent Architecture Design](en_agent-architecture.md)
- [Agent Failed Cases and Recovery Strategies](en_agent-failure-cases.md)
- [Agent Batch Assessment Report](en_agent-evaluation.md)
- [Agent fully run track sample](en_agent-trace-samples.md)
- [agent-004, normal completion trajectory.](agent-trace-agent-004.json)
- [agent-005 Correctly Safe Refuse Track](agent-trace-agent-005.json)
- [agent-001 Unexpected failure trajectory](agent-trace-agent-001-failure.json)
- [Week 4, final JUnit report](pytest-results-week4-final.xml)

## 16. Summary

Week 4 moves the project from a "fixed RAG call chain" to "controlled multiple steps" Agent. The system already has a allowlist of tools, a unified implementer, a structured state, failure recovery, genuine evidence constraints, API access, bulk measurement and redacted audit trails.

The most valuable result is not only the successful completion of the three tool diagnostics, but rather the ability of the system to distinguish and record correct refusals, failure of the tool and unforeseeable business failures. The current safety floor has been consciously assumed by the code rather than the model; The next phase will focus on improving Planner's task completion rate, minimum tool selection and end semantic stability without lowering security and evidentiary requirements.
