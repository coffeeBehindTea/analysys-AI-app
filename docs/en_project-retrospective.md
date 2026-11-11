# RobotOps Copilot Project Retrospective

## 1. Project objectives

The goal of RobotOps Copilot is not to allow large models to control robotics directly, but rather to construct a read-only diagnostic aids system: to receive redacted failure descriptions, logs and optional images, to read engineering knowledge, simulate telemetry or visual observations as required by the mission, and to structure diagnostics of the origin of the final output belt, references, risks, causes of termination and redacted tool trace.

The core contradiction to be addressed by the project is that robotic diagnostics require language understanding and multi-step planning capabilities in LLM, but engineering safety does not allow for the authority, evidence and control of boundaries to be left to the probabilistic model. The final architecture therefore clearly separates “what models are good at” from “what codes have to answer”.

## 2. Eight Weeks of Evolution

| Phase | Main competencies | Engineering focus |
|---|---|---|
| Week 1 | General breakdown and SSE | FastAPI, structured response, abnormal mapping and current output |
| Week 2 | Document Knowledge base and RAG | Document ingestion, Chunk, Embedding, Chroma, citation and refusal |
| Week 3 | Mixed search and evidence diagnosis | Normalization, keyword retrieval, RRF, reordering, evidence gate and evaluation |
| Week 4 | Text Diagnosis | Planner, Runner, Execuator, Registry, read-only and failed borders |
| Week 5 | Multimodal Agent | Photo input, Vision, OCR, source layer and multimodular assessment |
| Week 6 | Reliability and local delivery | Request level policy, security classification, progress Reducer, session, console and Docker |
| Week 7 | MVP Consistency | SSE Life Cycle, Four-Category Finals, Session Export and Full End to End Presentation |
| Week 8 | Candidature delivery | Rejection of intent to repair, clean environment, real model sampling, safe retrospection and compilation of works |

## 3. Final architecture trade-offs

### 3.1 LLM participation only in Planner and generation chain

Planner selects the next tool according to the mission and already available observations; Runner control cycle; Executor verifys the tool name, parameters, privileges, timeout and output. Runner does not contain LLM with Executor and does not bypass the registration form because of model requirements.

This division avoids placing the wrong local equivalent of "models say callable" as "system allowed callable". The model determines intent, the code determines permission.

### 3.2 Safety classification and scope of tools determined before Planner

`AgentRequestSafetyClassifier` Identify high-risk intentions such as real control, severe detour, electrical handling and the execution of untrustworthy photographic orders. `AgentToolPolicy` The minimum tool range is then calculated on the basis of actual user requirements and addresses negative expressions such as “no, no, no, no, no”, etc.

High-risk requests can be completed with low-risk read-only observations when the strategy permits, but ultimately they must go into `human_review_required` and cannot be upgraded to operating licences because the picture appears normal.

### 3.3 RAG citation assembled by Python

LLM selects only the evidence that exists for this request. File name, page number, chapter, Chunk ID, ranking and similarity are constructed by Python based on the true search result. The final report re-executes the request-level allowlist reference and avoids the modelling of document sources.

### 3.4 Vision Observation & Engineering Factual Layer

The picture is first verified by Base64, real format, file size, edge, pixel number and compression risk. The Vision output only describes visible text, indicators, widget status, uncertainty items and human review requirements, and marks `source="vision_model"`.

The knowledge base is de facto marked at `knowledge_base` and simulated telemetry at `simulated_memory`. Different sources can be grouped together and can also form a conflict.

### 3.5 Offline regression separated from real model assessment

The 30 offline scenes were verified with framework certainty using scripted Planner, Fake Vision and fixed Fixture. The 10 real model samples were called from the real external Planner, Vision, Embedding and knowledge base, which validated current external chain performance.

Offline `30/30` and true rigour `4/10` cannot be added, averaged or combined into a “overall accuracy rate”. The human review of articles 6, 2 in part and 2 failed could not recapitulate the definitive machine rating.

## 4. Critical difficulties and failures

### 4.1 Modular Boundary Confusion

Earlier, the route statement was misplaced into the Schema module, resulting in an import phase of `router is not defined`. This issue point indicates that the responsibilities of Schema, Router, Service and Dependency must remain one-way dependent and HTTP logic cannot be inserted into the data compact file for quick access.

### 4.2. Retrieval is not the final answer.

Week 3's candidate search once reached Top-3 full evidence recall rate of `0.957`, but the final reference to the evaluation would still fail due to over-control of the door, instability of Prompt output or failure of the model to use all necessary evidence. The statistics will be removed from subsequent scoring, evidence gate, generation and reference, avoiding the replacement of end-to-end quality with candidate tiers of indicators.

### 4.3 Local rules tend to result in the “dismantling of the East Wall to the West”

The increasing number of keyword rules for individual failure topics may allow the current sample to pass, while reducing the tool selection, citation or task completion rate of other scenarios. The project ultimately retains a common, interpretable strategy: the identification identifier is unified, model aliases, RRF, head retention reordering, evidence gate control and minimum size tool range; Sample drivers such as "Hide the tool after two successful searches" were withdrawn.

### 4.4 Planner output is unstable

The real model has returned to the parallel tool call, repeated call, empty content or invalid final JSON. The treatment is not for Runner to speculate, but rather to add serialization, Schema verification, repeated call detection, successive failure caps, Planner overtime and stable downgrade results. Even if HTTP returns 200, the execution status and the cause of termination will clearly reveal the semantic of failure.

### 4.5 Vision reads maps, but does not mean that you understand engineering syntax

True Vision can read the panel text correctly, and it may misjudge the diagram, blur the picture or lock the ring. The system thus retains `partial`, `unusable`, uncertainty and `requires_human_check` and prohibits the direct conversion of visual observations into equipment maintenance licences.

### 4.6 Docker malfunction from host state

Docker Desktop was unable to start at the final acceptance due to residuals of `sailor-ingest.sock.stale` and `engine.sock.stale`. The problem is not with the project mirror, but with two running directories running alternately after the Docker Desktop abnormal exit. Neither the mirror, volume or project file has been deleted by exiting Docker in its entirety, back-uping the two running directories and starting recovery with a single instance.

## 5. How to avoid overworking the test sample

The project is subject to the following:

1. Priority is given to repairing the common contract and the call chain rather than the final answer for a single issue coded hard.
2. Saves the Gold scene, Fixture, real model output and manual judgement separately.
3. Each local change runs the relevant unit tests, 30 offline regressions and final integrity pytest.
4. At the same time, observing tool selection, mission completion, citation, source and security is not only a single pass rate.
5. Keeps the wrong code, downgrades and manual conclusions for real model failure, without Fake results pretending to be real success.
6. When optimization causes fluctuations in overall indicators, priority is given to an interpretable and more balanced global strategy.

## 6. Final quality results

| Level | Result | Annotations |
|---|---:|---|
| Fixed multimodular offline regression | `30/30` | Framework certainty adopted |
| Targeted Safety Tests | `34/34` | A secure border has been defined to pass |
| Full automated regression | `2655/2655` | Current submission no failure and collection error |
| Fixed Sample with Live Models | Strict `4/10`. | Tool selection and security denials, `1.000`. |
| Human review | 6 Passed, 2 Partial, 2 Failed | Do not overwrite machine rating |
| Release Candidate Demo Acceptance | Regular diagnosis and human review | Docker, SSE, session, export end-to-end workflow to demonstrate |

## 7. Current Limitations

- The system does not connect real robots, RCS, WMS or production telemetry platforms.
- Telemetry is derived from redacted simulation data within the process, which is recreated once the process has been restarted.
- All tools are read-only; the draft test will not be implemented.
- The strict adoption rate of the true model remains at `0.400` and cannot be claimed to have reached production accuracy.
- There is no identification, tenant segregation, TLS, production key management, task queues and centralized logs.
- No reliable Token usage, no real cost.
- There is no industrial safety certification to replace the original plant information and qualified personnel.

## 8. Follow-up directions

1. Add auditable sub-missions to the “sub-mission-evidence” coverage matrix for complex issues.
2. Introduction of a stand-alone reordering model or cross-codifier and re-establishment of the Gold baseline, subject to the maintenance of the allowlist.
3. Improves the binding decodement, retry and recoverability of Planner's final JSON.
4. Increase equipment area detection, image quality pre-screening and multi-perspective consistency verification for Vision.
5. Increases the storage of controlled objects to avoid transmitting bulk Base64 in JSON.
6. Add authentication, privileges, audits, task queues, centralized logs and multi-case session memory.
7. (b) The completion of threat modelling, competency assessment, simulation certification and industrial safety reviews before any real equipment is accessed.

## 9. Important engineering recognition

The most important result of this project is not “to automatically do more”, but rather to establish boundaries: the model sets the next step, the code decides whether or not to do it; the picture provides visible observations, the knowledge base provides the basis for the work; the system can refuse when there is insufficient evidence and the system has to be artificially transferred when there is a high risk.

Retroactive, reversible and auditable are more important for robotic scenarios than purely automatic.
