# Week 5 Engineering Summary

## 1. Scope of the summary and final conclusions

Week 5 aims to enable Agent to process live photographs, meter intercepts and equipment light photographs based on Week 4 controlled Robot Diagnostic Agent, while continuing to comply with the evidence, references, risks and refusals established during the first three weeks.

The following deliveries were completed this week:

- Photo requests, post-validation photographs, draft visual models and data compacts for open visual observations;
- Base64 Decoding, Real Photo Format Recognition, Size, Size, Pixels and Empty Content Verification;
- OpenAI Compatible Vision Project, Independent Vision Configuration and Fake Vision Project;
- Local Tesseract OCR, text lines and boundary frame contracts, certainty panel rules
- OCR contrasts the experiment with the rules, the Vision model, and the 6 sample of the three direct refusal routes;
- Request-level pictures Store and read-only `analyze_robot_image` Agent tools;
- Joint diagnostic links for pictures, phenomena, redacted logs, knowledge evidence and simulation of telemetry;
- 30 multimodal Agent scenes, mass measurements of scripts, final reports and complete redacted tracks;
- Week 1 to Week 5 returns completely offline, with 2214 tests, 0 failed, 0 errors, 0 skips.

The learning target for this week has been completed, but the system is still not a production-level robot diagnostic product. In the final 30 scene assessment, the success rate of requests was 1.000, the security rejection rate was 1.000 and the correct selection rate for the Vision tool was 0.933; The total pass rate for the strict scene is only 0.233. The results show that photographic input, safe abstention and tool boundary formation have become stable, but the real Planner remains unstable in terms of minimum tool collection, end semantics, source utilization, complete evidence coverage and mission completion.

## 2. Week 5 position in the overall AI application structure

Five weeks of capacity has been increased in order of service skeletons, external knowledge, reliable retrieval, controlled operations, multimodal observations:

| Phase | Core competencies | Issues addressed |
|---|---|---|
| Week 1 | FastAPI, Pydantic, LLM, error processing, SSE, Request ID | Create a runable, verifiable and traceable AI service skeleton |
| Week 2 | Document resolution, Chunk, Embedding, ChromaDB, RAG, Quote Assessment | Allow the model to access the project language and return to a real reference that can be located |
| Week 3 | Query integration, combination of keywords and vectors, RRF, reordering, evidence gate, structured diagnosis | Increase access to difficult queries and limit diagnosis to the use of confirmed knowledge evidence |
| Week 4 | Planner, Tool Registry, Unified Executor, AgentRunner, failed downgrade, audit trail | Let the model select multiple read-only capabilities in the controlled cycle |
| Week 5 | Photo appliance, Vision, OCR, visual tools, source layer, multimodal assessment | Allows Agent to observe pictures without infer visualizing them as engineering facts. |

Week 5 does not replace RAG, diagnosis Schema or AgentRunner. It has added a new source of information to the Agent tool layer: visual observation.

Visual models are suitable for answering “what is seen in the picture”, knowledge banks are appropriate for answering “what these phenomena mean in equipment”, and simulation telemetry is appropriate for answering “what is the current demonstration of robotic state”. These three types of information cannot be replaced by one another:

```text
Vision model: observation of colour, text, brightening, shielding and widget relations in pictures
Knowledge base RG: provision of equipment information, failure statement and retroactive basis in safety protocols
Simulation of telemetry: provide the location, amount of electricity, task and malfunction of the current demonstration object
Planner: Select a read-only tool to call according to the task
Python Control Level: Validation Permissions, Data Contracts, Sources, Evidence and Final Response
Manual personnel: drafts for identification of high-risk operations, on-site status and testing
```

The reason for this design is that both visual and linguistic models are probabilistic models. They can assist in observing and organizing information, but cannot decide alone on the authenticity of the picture, project conclusions, security clearances or on-site operations.

## 3. Full Call Chain for Multi-Model Agent

A photo-based diagnostic request passed through the following links:

```text
HTTP Client
→ RequestIdMiddleware: generate or disseminate request_id
→ App.routers.ent. diagnose_with_agent (validation of AgentDiagnosisRequest)
→ AgentDiagnosisService.diagnose(): Coordinate the current request for Agent resources
→ VisionInputAdapter: Decoding Base64 and verifying the true picture format and resource ceiling
→ RequestVisionInputStore: Save certified images with request-level references such as image_001
→ AgentRunner.run(): Start planning, tool implementation, observation and end cycle
→ OpenAICompatibleAgentPlanner.plan(): return call_tool or finish decision
→ ToolExecutor.execute(): verify toolname, parameters, read-only privileges, risks, timeout and output
→ analyze_robot_image: read pictures of the current request by image_ref
→ OpenAICompatibleVisionProvider. analyze_image(): Send original multimodular messages
→ VisionObservation: Return structured visual observations from belt source, model and Prompt version
→ AgentRunner.run(): Record observation and decide whether to continue retrieving knowledge, read telemetry or close
→ AgentDiagnosisService.diagnose(): Constructing open trajectory and collecting visual, telemetry and draft results
→ agent_diagnosis_report_builder: Validation of final draft and allowlist of knowledge evidence
→ AgentDiagnosisResponse: Draft return diagnosis, executive summary, visual observation, telemetry and testing
```

The link divides the "photo access" into three separate questions: "What does the model see from the picture?"

1. `VisionInputAdapter` to determine whether or not the input is an acceptable real picture;
2. VisionProvider generates structured and visible observations;
3. Agent, Knowledge Retrieval and Reporting Constructors determine whether there is sufficient basis to form a diagnosis.

The fact that the picture is verified by format does not mean that the contents of the picture are true; either the Vision output is verified by Schema or the project conclusion is correct.

## 4. Completion of six missions

| Tasks | Main outputs | Duties in the structure | Finish Effects |
|---|---|---|---|
| Task 1: Picture input adapter | `vision.py`、`vision_input.py`、Vision Provider、Fake Provider | Convert untrustworthy Base64 request into internal image input verified by resources and format | Illegal formats, blank images, mega-grams, abnormal sizes, overtime and upstream errors all have clear boundaries. |
| Task 2: Tool selection contrast experiment | OCRProvider, rule solver, 6 samples, evaluation scripts and reports | Comparison of the OCR plus rules, the Vision model and the scope of application of direct refusals | Prove that different photo missions require different processing routes and cannot be uniformly handed over to the most expensive models |
| Task 3: Visual tools | `RequestVisionInputStore`, `analyze_robot_image` Definitions and Handler | Registering VisionProvider as a request level, read-only, auditable Agent tool | Planner can only read pictures of current requests registered, and cannot access any file or web site |
| Task 4: Multimodal diagnostic organization | Agent API image field, Service assembly, visual response and cross-field verification | Week 4 Agent Cycle and Week 3 Diagnostic Report | Pictures, logs, knowledge evidence and telemetry can be used jointly, with sources still distinct from each other |
| Task 5: 30 assessment collection | `multimodal_scenarios.jsonl`, photo index, scene description and production script | Fixed normality, noise, loss, conflict, injection and high-risk behaviour Gold | Real model performance can be measured in bulk and no longer rely on individual successful presentations |
| Task 6: Assessment and reporting | `evaluate_multimodal.py`, definitive rating, ancillary Judge, cost and trajectory report | Statistical vision, tools, missions, references, sources, security, delays and cost coverage | You can distinguish between HTTP success, visual correctness, mission completion and strict passage. |

## 5. Picture Input Adaptation Layer

### 5.1 External payload separated from internal input

`VisionImagePayload` indicates the external payload submitted by the client, consisting mainly of:

- `mime_type`: the type of mime declared by the user;
- `encoding`: currently fixed at Base64;
- `image_base64`: does not contain the prefixed image content of Data URL;
- `analysis_goal`: Objectives to be observed;
- `detail`: Level of detail for visual models.

This Schema can only prove that the field structure is valid, that Base64 can successfully decode, and that the document content corresponds to the type of MIME stated. Therefore Provider does not accept `VisionImagePayload` directly.

The call chain for `VisionInputAdapter` is:

```text
AgentDiagnosisService
→ VisionInputAdapter.adapt(current module)
→ Strict Base64 decoded.
→ Pillow reads real picture formats and dimensions
→ Bytes, edges and total pixel checks
→ SHA- 256 metadata binding
→ VisionInput
```

After authentication, `VisionInput` can enter theProvider. It contains real graphic metadata, protected photo bytes, analytical objectives and detail levels. The byte of the picture is used `SecretBytes` and is excluded from the normal `model_dump()`, log expression and public response.

### 5.2 Why limit both size, length and pixels

Limiting the number of Base64 characters is not enough to cover all picture risks:

- Document bytes limit the cost of controlling the request, memory and network;
- Unilateral size limits reject abnormal broad or long maps;
- Total pixels limits to prevent large memory after decryption of small compressed files;
- The true format of Pillow recognizes to prevent any inconsistency between the extension or MIME statements;
- Empty content and damaged pictures were denied locally before calling external Vision services.

The default single-shot image limit is 5 MiB, with no more than 4096 pixels on any side and no more than 1600 million total pixels; One time, Agent requested to contain up to 3 pictures. These are the resource boundaries of current teaching and learning projects and are not common values for all production systems.

## 6. VISION PROVIDER AND STRUCTURE OBSERVATIONS

### 6.1 Provider Agreement

`VisionProvider` is dependent on the reverse border. The Agent tool relies only on the ability to "receive post-validated pictures and return `VisionObservation`" and not on a specific object of SDK, a particular supplier.

The call chain is:

```text
AnalyzeRobotImageToolHandler
→ VisionProvider. analyze_image (Current agreement)
→ OpenAIcompatibleVisionProvider or FakeVisionProvider
→ VisionObservation
```

Realization of the OpenAI compatible interface using `AsyncOpenAI`. Fake achieves the preservation of redacted call records and returns to fixed observation or preset anomalies so that automated testing does not require a real key, network or billing call.

### 6.2 Primary Multiple Modular Input

RealProvider places the analysis target and picture Data URL together in the message center of the model. Pictures go straight into multimodal models, without two-stage links:

```text
Picture → The first model generates an unsourced summary → The text model treats the summary as the only fact.
```

The reason for avoiding this link is that the intermediate summary loses the source, visible area, uncertainty and detail of the picture, and the second model cannot recheck the original map.

### 6.3 Data boundary for visual observation

`VisionObservation` can express:

- Visible observations in pictures;
- (b) the identifiable indicator and its surface state;
- Projects that cannot be identified or require human review;
- (a) The overall state and whether manual checks are required;
- `source="vision_model"`；
- Source photo abstract, model name and Prompt version.

Visual observation allows only a description of what “is seen” and does not directly state the cause of it by simply using the field structure. For example, the "NET indicator light red is constant" may be visual observation, and "network modules have been damaged" requires knowledge information, telemetry or manual inspection support.

## 7. OCR, rule resolution and tool selection experiment

### 7.1 OCR Route

`TesseractOcrProvider` Use `pytesseract` to call the local Tesseract engine and return text lines, confidence and boundary frames. OCR is suitable for clear failure codes, numbers and state text, but is not very good at judging colours, bright spots, plugs or widget spatial relationships.

The OCR call chain is:

```text
Verified Image
→ TesseractOcrProvider. analyze_image (Current module)
→ Pytesteract calling local Tesseract subprocess
→ Text, Confidence and Boundary Frame
→ OcrObservation
```

`OcrRuleParser` Retrieve supported panel fields in accordance with fixed rules. The rule resolution is definitive: the same OCR text will receive the same field and no LLM reasoning is required; Returns `unsupported` or `partial` when no rule is matched to support. No non-existent status will be added.

The rule solves the call chain as follows:

```text
OcrObservation
→ OcrRuleParser.parse() (current module)
→ Fixed field and value-taking rules
→ OcrRuleParseResult
```

### 7.2, three route results.

6 redacted samples passed three separate routes, resulting in 18 single-line results:

| Route | Pass rate | Average delay | P95 Delay | External Model Call | Main scope of application |
|---|---:|---:|---:|---:|---|
| Local OCR Add Rules Parsing | 0.333 | 651.523 ms | 875.409 ms | 0 | Clear text panel, malfunction code and fixed field |
| Vision Model | 0.333 | 12587.276 ms | 27044.173 ms | 6 | Colours, brightening, blocking, connecting and widget relations |
| Direct refusal | 0.333 | 10.950 ms | 19.529 ms | 0 | Images missing, seriously blurred or unverifiable |

The same pass rate for the three routes does not mean the same capacity. Each route passes only on two types of samples suitable for itself: OCR is not a substitute for visual relationship judgement, Vision should not be called unconditionally, and direct refusal to answer cannot process clearly reciprocated pictures.

## 8. `analyze_robot_image` Agent Tool

### 8.1 Request-level picture

The client cannot provide `image_ref` on its own. Service, after a successful photo verification, generated opacity references to `image_001`, `image_002` etc. in the sequence of requests and wrote `RequestVisionInputStore`.

The call chain is:

```text
AgentDiagnosisRequest.images
→ VisionInputAdapter
→ Agent Diagnosis Service
→ RequestVisionInputStore.register() (current module)
→ Image_ref bound to VisionInput
→ Analize_robot_image by reference
```

Store is the requested level object and cannot use the global cache. This would avoid a picture requesting to read another request and would not allow the picture to accumulate indefinitely within the process.

### 8.2 Tool definition and implementation

`analyze_robot_image` is a low-risk, read-only tool. Inputs are only:

- `image_ref`: Photo references in current request;
- `analysis_goal`: Objectives to be observed for this time.

The implementation chain is as follows:

```text
AgentRunner
→ ToolExecutor.execute()
→ ToolRegistry.get("analyze_robot_image")
→ AnalyzeRobotImageToolHandler._call__() (current module)
→ RequestVisionInputStore.get(image_ref)
→ VisionProvider.analyze_image()
→ VisionObservation
→ ToolExecutionResult
```

Handler does not perform device control, does not modify pictures, does not dynamically import tools, does not access a two-dimensional code or a web site in a picture. Unknown image reference returns empty result; Vision Timeout, Upstream Error and Invalid Response were converted into auditable dissensitisation error codes, respectively.

## 9. Multimodal diagnostic organization and source hierarchy

### 9.1 API Extension

`POST /api/v1/agent/diagnose` Add `images` to the table while maintaining compatibility of pure text requests. When no picture is requested, the list is by default empty; When you have a picture, each one of the images is adapted locally and then enters the Agent cycle.

Increase in public response by `vision_observations`. This list must correspond to the successful `analyze_robot_image` steps in the implementation trajectory: it cannot increase visual observations in a vacuum, nor can it hide a video call that has been successfully implemented.

### 9.2 Four categories of information sources

| Source | Contents indicated | What can you prove? | I can't prove anything automatically. |
|---|---|---|---|
| `user_report` / `log_excerpt` | User description and redacted logs | Which phenomena and log text were submitted by the user | The contents are true, the log is complete or the root causes are confirmed |
| `vision_model` | Structurally visible in pictures | What text, colour, state and uncertainty items are observed in the model | In-house equipment malfunctions, operations implemented or knowledge rules established |
| `confirmed_knowledge` | RG Door Confirmed Real CHunk | The information clearly indicates which engineering rules and safety requirements | The current site must meet the trigger conditions in the data. |
| `simulated_memory` | Simulated telemetry of redacted within process | Current Presentation Store Returned What Status | Live status of real robots, RCS or WMS |

The final diagnosis must retain the difference of origin. When pictures conflict with the log, the system cannot choose one of them silently; Conflicts should be preserved, conclusions reduced in intensity and manual checks required.

## 10. Multi-mode security boundaries

### 10.1, the image is not credible.

None of the following are implemented as systematic instructions:

- The first step is to avoid the use of a simple injection, such as a quick injection.
- Shell, Python, robotic control or repositioning commands;
- Web sites in two-dimensional code or screen;
- Examples of API Key, Southorization or Keys;
- Words that require an expanded allowlist of tools or changes in risk levels.

The QR code, even if read, can only be used as normal text data and does not trigger links to browsers, HTTP clients or other tools.

### 10.2. Track dissensitisation.

Open multimodal trajectory is saved only:

- (a) Scene numbering and redacted input summaries;
- Toolname, state, time-consuming and error codes;
- Open visual observation and diagnostic response;
- • Determines the score and the cause of the failure.

Tracks do not save the full Base64, photo body, local image path, request-level photo metadata, Authorization, API Key, complete sensitive log or model private think chain. Open visual observation allows `source_image_sha256` to be preserved as a source complete fingerprint; The summary cannot be restored to the picture, nor is it a local path.

### 10.3 High-risk operations remain manual

Current five Agent tools are all read-only. The system does not have robotic motion control, long-range repositioning, mission commissioning, charge control, configuration modification or real test execution capability. `draft_test_case` generates only draft certainty pending manual approval, which does not mean that the test has been implemented.

## 11. 30 multimodal assessments

The assessment set up the following categories:

| Category | Number | Main validation target |
|---|---:|---|
| `normal_image` | 8 | Joint visual, intellectual and telemetry diagnostics completed in clear images |
| `noisy_or_low_quality` | 8 | Fuzzy, masked and low-quality images without speculation and retain the available partial results |
| `missing_or_unanswerable` | 6 | Correct refusal to respond if missing, unrelated or knowledge is insufficient |
| `image_log_conflict` | 4 | Keeps the source and requests for review in case of conflict between pictures and logs |
| `prompt_injection_or_high_risk` | 4 | Reject photo commands, link access and device writing operations |
| **Grand total** | **30** | Coverage of normal, degraded, absent, conflict and confrontation situations |

Each scene records not only problems and pictures, but also whether it is expected to call Vision, necessary tools, permissible tools, prohibited tools, anticipated diagnostic status, anticipated evidence, expected sources, safety requirements and mission fulfilment conditions.

Evaluates the path, SHA-256, format, dimensions, pixels and bytes of the whole picture before sending the first real HTTP request. This does not run until half of the time for bad images to end, nor does it consume external models when the data set is invalid.

## 12. The final real evaluation.

### 12.1 Core indicators

| Indicators | Count | Ratio |
|---|---:|---:|
| Request success rate | 30/30 | 1.000 |
| Total pass rate of scene | 7/30 | 0.233 |
| Image observation field accuracy | 29/35 | 0.829 |
| Tool selection accuracy | 18/30 | 0.600 |
| Vision Tool selection accuracy | 28/30 | 0.933 |
| Task completion rate | 4/8 | 0.500 |
| Citation correctness | 13/16 | 0.812 |
| Citation coverage | 13/21 | 0.619 |
| Source labeling accuracy | 21/30 | 0.700 |
| Security requirements pass rate | 79/91 | 0.868 |
| Safe abstention Rate | 14/14 | 1.000 |

### 12.2 Efficiency and cost

| Indicators | Result |
|---|---:|
| Average number of tools steps | 2.400 |
| Average-to-end delay | 26586.408 ms |
| P50 End-to-end delay | 21778.774 ms |
| P95 End-to-end delay | 62020.046 ms |
| Costed estimated coverage | 0.000 |
| Total estimated costs known | Unable to estimate |
| Estimated average cost of request | Unable to estimate |
| Auxiliary Judge Assessments | 2 |

The current Agent API response does not provide Planner, Vision and the Judge Token usage that can be used for pricing, and therefore cannot be reliably calculated. “Unable to estimate” means that the measurement is missing and does not mean that the cost is zero. The request is time-consuming, the tool steps and the length of the return text cannot replace Token usage.

### 12.3. Why is the pass rate so low?

The overall pass rate is a joint indicator. A scene must satisfy both:

- HTTP request successful;
- Agent execution and termination is correct;
- Vision calls for correct judgment;
- Photowatch coverage Gold;
- (a) The tool collection and key sequences are correct;
- The conditions for the completion of the mission are correct;
- Quote correctly and overwhelm fully;
- (a) The information sources are correctly marked;
- Security rules and refusals are correct.

None of these failed, and the whole scene went through. Thus, 0.233 cannot be simply interpreted as “only 23.3% images are identified” nor can the HTTP success rate of 1.000 be used in lieu of business correctness.

The main failures focused on: Vision's inconsistent status with Gold; Planner's reasons for termination not meeting expectations; inconsistent sources; Agent's undesired state; incomplete knowledge evidence; and `search_knowledge` or `get_current_time`, which was not required to transfer the scene.

## 13. Key issues, amendments and design trade-offs

### 13.1 Determination rating cannot be based solely on a fixed text match

Real Vision and Planner may use price expressions such as "unobserved state" "no clear display" "interface does not fully fit". Matching a fixed phrase would misjudge the result in the correct semantic language as a failure.

This week has expanded the determinations of regularization and equivalence, and allowed LLM-as-judge to provide support for a small number of boundary expressions. But the auxiliary Judge cannot rewrite a true response, nor can he substitute for a definitive rating of tools, references, sources and safety rules.

### 13.2 Failure of a single request cannot terminate the entire instalment

Batch assessments convert timeout, connection failure, non-2 xx, illegal JSON and response to Schema errors into redacted failure results, then continue the rest of the scene. The generation of a misleading official report will be refused only if the data set itself cannot form an effective and complete trajectory.

The call chain is:

```text
30 scenes
→ evaluate_multimodal.py pre-screen all pictures
→ evaluation service (current module) calls Agent API on an article-by-article basis
→ Single failed to structure failed rating
→ Continue the rest of the scene.
→ Summary of indicators, tracks and reports
```

### 13.3 Vision, mistakes must be distinguishable.

Initially, all visual failures were recorded as common tool errors, and it was not possible to locate whether they were time-out, upstream unusable or models returned to invalid structures. Eventually, redacted errors such as `vision_timeout`, `vision_upstream_error` and `invalid_vision_response` were added, which neither disclose the sensitive upstream text nor support failure statistics.

### 13.4, it's not enough to keep three successful tracks.

The final report retains three representative tracks, while preserving the complete redacted trajectory of failure to respond to a contractual but definitive rating. This can be traced back from the failure scene to the true tool sequence, status and error code instead of seeing only one aggregated percentage.

### 13.5 Undo keyword tool scope and fixed-number hide policy

The project tried to narrow the range of visible tools according to the mission word keyword and to hide `search_knowledge` after two successful knowledge retrievals. Both rules misperceived natural language expressions and fixed call times as reliable signals of whether a mission has been completed, make it easier to match the current Gold and may also lose the evidence required for subsequent complex missions.

The strategy for eventual retention is structural boundaries:

- (b) When no picture is requested, no visual tools that cannot be implemented are exposed;
- When a picture is requested, the visual tool can only read the current request Store;
- The remaining registration-only tools remain visible;
- Runner does not hide knowledge from a fixed number of success stories;
- Whether to continue or terminate is still proposed by Planner, and is subject to maximum steps, repeated calls, successive failures and authority strategies.

This retreat avoids sacrificing the general structure for local evaluation of figures.

## 14. Automation test disc

Final complete offline regression result:

| Indicators | Result |
|---|---:|
| Total number of tests | 2214 |
| Fail | 0 |
| Error | 0 |
| Skip | 0 |

The main processes and expected results to be covered by the additional tests include:

- Photo Schema test: valid MIME, Base64, analytical target and detail level can be constructed; additional fields, empty values and illegal count rejected by Pydantic;
- Input fit testing: conversion of legitimate images to `VisionInput` with real sizes and abstracts; damage, empty content, format mismatches, hyper-sized and pixels crossing the border failed before external calls were made;
- (a) Vision Projecter Test: Confirms that the pictures are sent in a native multimodular condition; fixed JSON is deconstructed as structured observation; timeout, upstream error, empty content and illegal JSON conversion to project anomaly;
- (a) Fake Projecter tests: fixed response, abnormal and redacted call records can be duplicated without access to the network;
- OCR tests: text lines, boundary frames, confidence, empty results, low confidence and sub-processe timed out of contract;
- Rules resolution test: supported fields are structured in a fixed structure; unsupported, partial and empty texts are not guessed to be complete;
- Visual tool testing: legal request-level reference returns to observation; unknown references, illegal parameters, tips into text, aerial observations and high-risk content enter the controlled path;
- Agent API testing: Compatibility of pure text requests; successful matching and registration of pictures; visual observation must correspond to the steps of a successful tool; complete images will not appear in response;
- Site data testing: 30 categories, tool constraints, sources, safety requirements, photo summaries and serialization loops remain stable;
- Evaluation tests: graphic fields, tools, tasks, references, sources, safe refusals, P50/P95, cost coverage and support Judge is based on a fixed sample;
- Track testing: Private thinking chains that represent tracks and failure tracks that do not include Base64, local paths, authentication information or models;
- Regression tests: Week 1 to Week 4, RAG, search, evidence gate, reference, structured diagnosis and text Agent behavior continue to pass.

These tests mainly use Fake Provider, Mark HTTP, FastAPI relying on coverage, `tmp_path` and temporary pictures. They demonstrate that code contracts meet expectations under controlled input, but are not a substitute for the real batch assessments of Vision, Planner, Embedding, Croma and the network links.

## 15. Major dependencys involved in this week

| Dependency | Use this week | Key borders |
|---|---|---|
| `Pillow` | Decoding pictures, identifying real formats, reading sizes and pixels | It's for input security, not for image semantic diagnosis. |
| `pytesseract` | Call Local Tesseract OCR | System executable files and language data required; OCR confidence is not project authenticity |
| `openai` | Creates a walk OpenAI compatible Vision client and sends multimodular messages | Provider protocol quarantine SDK; Vision configuration independent of text LLM |
| `pydantic` | Pictures, observations, tools, APIs, scenes and reporting on the running time compacts | Schema's legitimacy does not mean that it's true. It still requires rules of origin and evidence. |
| `fastapi` | Receive multimodal Agent JSON requests and injects request level resources | Client cannot specify allowlist of tools, maximum steps or risk privileges |
| `httpx` | Batch evaluation call Agent API and simulation HTTP error | Mock Test does not access real networks |
| `pytest` / `pytest-asyncio` | Sync, Step, Parameters and Unusual Path Return | Offline testing differs from real model assessment responsibilities. |

## 16. GitHub Open delivery boundary

Retention of open warehouse:

- Production codes, tests, scripts and fixed data contracts;
- 30 redacted scenes and samples of teaching pictures;
- Finally, `en_multimodal-evaluation.md`;
- `en_multimodal-tool-selection.md` and `en_multimodal-scenarios.md`;
- Full redacted polymodular trajectory;
- Placeholder configuration in `.env.example`.

`.gitignore` Exclude:

- `.env` and other real configurations;
- `要求.txt` weekly plan and acceptance feedback;
- Original language, processed body text and Chroma vector database;
- Chroma Backup Directory;
- Original JSON evaluation and multi-cycle versioning intermediate reports;
- Python, pytest, editor and temporary file cache.

Open tracks are already over-reliable key mode, local absolute path, photo body and sensitive field checks. This check reduces the risk of missubmission, but is not a substitute for an item-by-item check on the Git client before submission.

## 17. Current Limitations

- The Vision model output is indecisive, and the same picture may produce different observations in different models, Prompt or operating;
- Fuzzy, reflective, masking, small words, complex background and similar colours may still result in missing or misreading;
- OCR relies on local Tesseract installation, language data and operating system environment;
- The current Base64 JSON request increases the volume of the request, and production systems usually require storage and short-term reference by the controlled object;
- The current picture, SHA-256, can only be used for completeness and cannot prove timing, equipment identity or physical authenticity;
- Current analogue telemetry does not connect real robots, RCS, WMS or production platforms;
- All Agent tools are read-only and the system cannot perform maintenance, repositioning, motor control or real testing;
- There is currently no identification, segregation of tenants, document virus scanning, production audit platform or official front end;
- Planner and Vision rely on external models, which may still overtime, return ineffective structures or select redundant tools;
- (a) The current Token usage does not enter the public response and cannot give a reliable single request cost;
- The total pass rate for the strict scene is only 0.233, which does not allow for a declaration of production availability or safe certification;
- LLM-as-judge covers only a small number of boundary expressions and can only be used as supporting indicators;
- The current 30 scenes are fixed teaching data sets and still require more extensive equipment, environment, filming conditions and manual labelling to validate the generalization.

## 18. Follow-up improvements

Follow-up improvements should prioritize systemic issues rather than continuing to add keyword rules to individual scenarios:

1. Collecting token usage from Planner, Vision and Judge within the service end and maintaining auditable model price versions;
2. Disassembly the "Voice-observation Project Interpretation Current Status Check" into a clear mission phase, reducing Planner's double search and source confusion;
3. (b) Establish independent certainty checks or dedicated models for photo quality, text readability, colour and connectivity;
4. (b) Continue to refine the conditions for the cessation of Planner, the selection of the minimum necessary tools and the semantics of the end of the high risk;
5. (b) Expansion of manually reviewed multi-equipment, multi-light, multi-angle and multi-noise assessment collections;
6. Create real, read-only telemetry adapters and maintain the same Provider or Protocol boundary as simulator Store;
7. Increased identification before entering the production environment, segregation of tenants, storage of objects, detection of malicious documents, speed limits and independent audit logs;
8. Maintaining layers of sources of visual observation, knowledge evidence, real-time data and manual confirmation does not allow for the removal of security denials in order to increase pass rates.

## 19. Final Conclusion

Week 5 completes the project extension from Text Agent to Multi-Model Agent:

```text
Security Picture Input
+ Vision original mode call
+ OCR and the rules baseline
+ analize_robot_image read-only tool
+ Request Level Picture Segregation
+ Visual, intellectual, log and telemetry source layers
+ 30 Site Batch Assessment
+ Identified indicators and supporting Judge
+ Redacted trajectory and cost boundary
+ 2214 offline returns
```

The most important result of this week is not “models are already visible”, but rather the creation of input security, source tracking, permission control, structured output, failure downgrade and batch assessment of boundaries required for images to enter AI applications.

The current system is suitable for use as a multimodal AI application engineering learning project and controlled prototype, directly for robotic control, on-site security decision-making or automated disposal of production failures. If the next phase continues to evolve, priority should be given to improving the stability of mission completion and source utilization, and to supplementing real costs, identity privileges and access to production data, rather than simply optimizing text matching results for individual assessment scenarios.
