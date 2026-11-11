# RobotOps Copilot

[简体中文](ch_README.md)

RobotOps Copilot is a read-only multimodal diagnostic agent for robotics development and operations. It combines redacted fault descriptions and logs, field images, simulated telemetry, and engineering documents into structured diagnoses with traceable evidence. It can abstain when the evidence is insufficient and records a tool trace for review.

The LLM plans steps and drafts the response. Python enforces tool permissions, safety classification, evidence thresholds, citation allowlists, termination rules, and redaction.

## Project Overview

| Problem | How RobotOps Copilot handles it |
|---|---|
| Relevant engineering documents take time to find | Chunks and embeds PDF, Markdown, and TXT files, then combines keyword and vector retrieval with reciprocal rank fusion (RRF) and reranking. Citations point back to retrieved chunks. |
| The LLM can fill gaps with unsupported claims | Applies a composite evidence gate, validates structured output, and limits citations to an allowlist of retrieved chunks. |
| Diagnosing a fault requires several sources | Uses Vision, knowledge-base retrieval, and redacted simulated telemetry through the minimum set of tools permitted for the request. |
| Image observations can be mistaken for engineering facts | Labels user input and each source separately, including `vision_model`, `knowledge_base`, and `simulated_memory`. |
| High-risk requests require human review | Runs a deterministic safety classifier before the planner. Requests involving emergency-stop bypass, live device control, or energized work require human review. |
| An agent run can be difficult to audit | Publishes SSE events and records redacted tool traces and sessions. Sessions can be exported as JSON or Markdown. |

### Final Release Candidate at a Glance

| Check | Week 8 result | Scope |
|---|---:|---|
| Offline regression with fixed fixtures | `30/30` | Repeatability of the deterministic framework on fixed inputs. |
| Targeted safety tests | `34/34` | Defined cases for prompt injection, external links, unauthorized tool use, control requests, and information leakage. |
| Full automated test suite | `2655/2655` | The recorded run had zero failures and collection errors. |
| Fixed sample with live models | `4/10` strict passes | Tool selection accuracy and safe abstention rate were both `1.000`. Planner, Vision, and retrieval failures still occurred. |
| Human review of the live-model sample | 6 pass, 2 partial pass, 2 fail | Reviewers assessed acceptable fallback behavior and equivalent wording separately from automated scoring. |
| Release candidate demo acceptance | Normal diagnosis and human review passed | Verified the local Docker, SSE, session, and export workflow. |

### Quick Links

- [Final Evaluation](docs/en_final-evaluation.md)
- [Release Candidate Acceptance](docs/en_candidate-acceptance.md)
- [Project Retrospective](docs/en_project-retrospective.md)
- [Demo Script](docs/en_demo-script.md)
- [Full Demo Video](https://github.com/coffeeBehindTea/analysys-AI-app/releases/latest/download/robotops-copilot-demo.mp4)
- [Normal Diagnosis Screenshot](docs/robotops-copilot-docker-demo.png) and [Human Review Screenshot](docs/robotops-copilot-sse-human-review.png)

The knowledge base ingests public or redacted PDF, Markdown, and TXT documents. It chunks the documents, embeds them, and stores the vectors and metadata in ChromaDB. Users can ask questions in natural language and receive evidence-based answers with citations to a source file, page, or section.

- **Week 1** introduced fault triage and an SSE endpoint.
- **Week 2** added document management and knowledge-base Q&A.
- **Week 3** added deterministic query normalization, hybrid retrieval, RRF, reranking that preserves the highest-ranked results, composite evidence gating, and the structured diagnosis API.
- **Week 4** added the controlled Robot Diagnostic Agent, which uses registered read-only tools for multi-step diagnosis.
- **Week 5** integrated field images into the agent.
- **Week 6** added per-request tool policies, deterministic safety classification, evidence-driven progress tracking, a 30-scenario offline regression suite, redacted sessions, a lightweight console, and reproducible Docker startup.
- **Week 7** added agent lifecycle events over SSE, four public terminal statuses, live console updates, session export, and end-to-end delivery records.
- **Week 8** corrected tool scoping for negated intent, validated startup with an empty knowledge base, sampled live models, and retested the release candidate.

The earlier APIs are available for comparing direct LLM triage, RAG Q&A, evidence-constrained diagnosis, the text agent, and the multimodal agent.

> RobotOps Copilot is an AI application engineering learning project. Qualified personnel must confirm diagnoses, repairs, and safety procedures against the original equipment manufacturer's documentation.

Week 8 completed the teaching demo and portfolio release candidate. Before planning, the agent classifies safety and computes a request-specific tool allowlist. After each tool result, `AgentProgressReducer` updates completed capabilities, evidence coverage, missing information, and the allowed next tools. The application validates image format and resource limits locally, then sends the image to the Vision provider as native multimodal message content. Public responses, stored sessions, and evaluation traces exclude complete image payloads.

The offline fixture suite passed all 30 scenarios under strict scoring. Image observations, tool selection, task completion, citations, source labeling, and safe abstention each scored `1.000`. The live-model sample passed 4 of 10 scenarios under strict scoring. The separate human review recorded 6 passes, 2 partial passes, and 2 failures.

The full Python regression recorded `2655 passed`. See the [Final Evaluation](docs/en_final-evaluation.md), [Live-Model Evaluation](docs/en_real-model-evaluation.md), [Safety Retest](docs/en_security-retest.md), [JUnit Report](docs/pytest-results-week8-final.xml), [Docker Startup Record](docs/en_docker-startup-record.md), and [Release Candidate Acceptance](docs/en_candidate-acceptance.md).

## 1. Core Capabilities

### Knowledge-Base Document Management

- Upload PDF, Markdown, and TXT documents.
- Validate file type and size and reject documents with empty text.
- Derive a stable `document_id` from the SHA-256 hash of the file and reject duplicates.
- Store metadata for documents, chunks, and embeddings.
- List ingested documents and their chunk counts.
- Delete all vectors for a document by `document_id`.

### RAG Q&A

- Normalize error codes, model names, numeric values, and units. Apply the `deterministic-v2` query rewrite.
- Run Chroma vector retrieval and in-memory keyword retrieval.
- Fuse both rankings with RRF and rerank while preserving the highest-ranked results to build the final top K.
- Apply a composite evidence gate before calling the generative LLM.
- Return the file name, page or section, chunk ID, rank, similarity, and excerpt.
- Instruct the model to answer using the supplied evidence. Python assembles citations from retrieved results.
- Retain source qualifiers and safety constraints relevant to the question.

### Structured Diagnosis

- `POST /api/v1/diagnostics` takes a robot ID, symptoms, redacted logs, and an optional document search scope.
- The response includes the status, reported symptoms, evidence, possible causes, follow-up checks, risks, missing information, and abstention flag.
- Every cause and follow-up check references an actual chunk returned for the request.
- The report flags high-risk checks for confirmation or execution by qualified personnel.
- Gate rejection, invalid model output, unknown citations, timeouts, and upstream errors each have defined fallback behavior.

### Robot Diagnostic Agent

- `POST /api/v1/agent/diagnose` accepts a robot ID, symptoms, redacted logs, an optional task goal, and up to three Base64 images.
- `AgentRunner` runs a bounded loop that plans a step, calls a tool, observes the result, and decides whether to continue.
- The planner can request `analyze_robot_image`, `search_knowledge`, `get_robot_telemetry`, `draft_test_case`, and `get_current_time` from the tool registry. It has no shell, Python execution, arbitrary URL access, or device-control tools.
- `search_knowledge` returns evidence from hybrid retrieval after deterministic gating. It makes no additional call to an answer-generation LLM.
- `get_robot_telemetry` reads redacted simulated snapshots from memory.
- `draft_test_case` builds a deterministic draft from chunks confirmed for the current request. Execution requires human approval.
- `analyze_robot_image` reads validated images registered for the request. The Vision provider returns structured observations. The tool leaves images unchanged and treats embedded links and QR codes as untrusted input.
- Each run enforces step limits, planner and tool timeouts, and policies for repeated calls, consecutive failures, and permission violations.
- The final `DiagnosisReport` uses a request-specific evidence allowlist. Python validates chunk metadata against that evidence.
- Public responses include redacted tool traces, visual observations, simulated telemetry, test drafts, termination reasons, and missing information. They exclude complete images, raw tool arguments, full logs, and private model reasoning.

### Agent Reliability and Observability

- `AgentRequestSafetyClassifier` classifies high-risk control requests, safety bypass requests, and attempts to execute untrusted image instructions before planning.
- `AgentToolPolicy` computes the minimum `allowed_tool_names` needed for each request.
- `AgentProgressReducer` tracks completed capabilities, tool results, confirmed sources, evidence coverage, conflicts, missing information, and allowed next steps. It prevents unauthorized and repeated calls.
- Thirty deterministic offline scenarios use a fake planner, fake Vision provider, and fixed fixtures. Live-model evaluations measure model behavior separately.
- Each diagnosis records a redacted session. Look up recent sessions or query by `session_id` to inspect traces, termination reasons, visual observations, citations, and run metrics.
- Public diagnosis statuses are `completed`, `partial`, `abstained`, and `human_review_required`. High-risk requests may include permitted read-only observations and end in human review.
- `POST /api/v1/agent/diagnose/stream` publishes request receipt, safety classification, tool scope, planning, tool start and end, progress, and the final diagnosis. Technical errors after the stream starts end with `stream_error`.
- `/console/` displays live SSE status, tool traces, visual observations, sources, citations, and termination reasons. It exports sessions in JSON or Markdown.
- Docker Compose starts the application and health check locally and bind mounts Chroma data and diagnostic sessions.

### Vision, OCR, and Visual Evidence

- `VisionImagePayload` defines the external image payload. `VisionInputAdapter` decodes Base64, checks the actual format and resource limits, rejects empty content, and guards against decompression bombs.
- Validated images receive opaque, request-scoped IDs such as `image_001` and `image_002` and are stored in `RequestVisionInputStore`.
- `OpenAICompatibleVisionProvider` sends each image as a native `image_url` data URL alongside the analysis prompt.
- `VisionObservation` records visible observations and indicators, uncertainty, and human-review requirements. It also records `source="vision_model"`, the model name, prompt version, and source image digest.
- The local `pytesseract` OCR baseline and deterministic rule parser handle text panels. Vision handles color, component and connection states, and more complex visual relationships. The system can abstain when an image is missing or unreadable.
- Text, QR codes, URLs, commands, prompts, and apparent credentials in images are untrusted data. They cannot change tool permissions, system constraints, or logging behavior. Embedded links are never fetched automatically.

### Evaluations and Experiments

- The machine-readable gold set contains 28 questions, including 23 answerable and 5 unanswerable questions. It covers single and multiple evidence items, error code variants, model names, numeric boundaries, and cross-document procedures.
- Week 3 added eight difficult questions to test keyword retrieval, rank fusion, and safe abstention.
- Retrieval evaluation measures Recall@1, Recall@3, and false retrieval on unanswerable questions. Citation evaluation measures precision, coverage, complete-evidence answer rate, and correct abstention.
- The repository records a reproducible comparison of RAG and a standalone LLM.
- Eight fixed agent scenarios include four multi-tool cases and three safe-abstention cases. The batch evaluation tracks request success, tool selection, task completion, evidence citations, average steps, and safe abstention.
- Redacted trace examples cover successful multi-tool diagnosis, expected safe abstention, and unexpected task failure.
- Thirty multimodal scenarios include eight normal images, eight noisy or low-quality images, six cases with missing information or unreadable images, four image-text conflicts, and four prompt-injection or high-risk cases.
- Multimodal metrics cover observation fields, tool and Vision selection, completion, citations, source labeling, safe abstention, tool steps, P50/P95 latency, and cost estimation coverage.
- LLM-as-judge supplements equivalent-wording checks. Deterministic fields, evidence citations, safety rules, and manual spot checks provide the primary assessment.
- Fake embedding and LLM providers and mocked HTTP allow tests to run offline.

## 2. RAG and the Agent in the Application Architecture
```mermaid
flowchart TD
    subgraph Ingestion["Offline/management links: document ingestion"]
        Source["PDF / Markdown / TXT"]
        Loader["Document Loader<br/>Parsing text and location"]
        Chunker["Text Chunker<br/>Divisions and Overlap"]
        EmbedDocument["Embedding Service<br/>Document Quantification"]
        Chroma["ChromaDB<br/>Vector, body, metadata"]

        Source --> Loader
        Loader --> Chunker
        Chunker --> EmbedDocument
        EmbedDocument --> Chroma
    end

    subgraph Query["Online links: question and answer and diagnostic shared evidence layers"]
        User["Problem / trouble description"]
        Router["FastAPI Router"]
        Rewrite["Normalize + Query Rewrite"]
        Dual["Vector Top-20 + Keyword Top-20"]
        Fusion["RRF Integration + Head Retrieval"]
        Gate{"Combining evidence gate control"}
        Abstain["Python's steady refusal.<br/>Do not call LLM when evidence gate refuses"]
        AnswerLLM["Questions and answers or diagnosis of LLM"]
        Validate["Pydantic + Reference Allowlist"]
        Response["Questions and answers / DiagnosisReport"]

        User --> Router
        Router --> Rewrite
        Rewrite --> Dual
        Dual --> Chroma
        Chroma --> Fusion
        Dual --> Fusion
        Fusion --> Gate
        Gate -- 拒绝 --> Abstain
        Gate -- 放行 --> AnswerLLM
        AnswerLLM --> Validate
        Abstain --> Response
        Validate --> Response
        Response --> Router
    end
```

RAG does not run training models and does not put all documents directly into Prompt. It has broken down knowledge access into two phases:

```text
Retrieval: First, we find the most relevant little evidence from the knowledge base.
Generate: and let LLM organize answers based on this evidence only
```

The main reasons for this design are:

- LLM training may be outdated or not include internal rules of the organization.
- The full document usually exceeds the context window of the model.
- Direct reliance on model memory does not provide a verifiable source.
- Industrial safety and trouble management cannot allow models to fill in parameters at will.
- Where there is no evidence, the business code should reject the answer and not require model speculation.

### Agent Request Flow

Fixed RAG interfaces usually perform a prewritten "Retrieving evidence gate generation" process. The Robot Diagnostic Agent, located between HTTP API and established knowledge, telemetry and draft capabilities, is responsible for selecting the next tool according to mission status, but the tool privileges, parameter validation, time overrun, allowlist of evidence and final response are still controlled by Python code.

```mermaid
flowchart TD
    Client["Client diagnostic request"]
    Image["Base64 Photo Load<br/>Untrustworthy user data"]
    Middleware["RequestIdMiddleware<br/>Generate request tracking ID"]
    AgentRouter["Agent Router<br/>Verify HTTP Request"]
    Service["AgentDiagnosisService<br/>Organize requests and construct public responses"]
    Safety["RequestSafetyClassifier<br/>Identification of high-risk intent"]
    Policy["AgentToolPolicy<br/>Calculate the minimum utility range for the requested level"]
    Progress["AgentProgressReducer<br/>Tracking capacity, evidence and next steps"]
    VisionAdapter["VisionInputAdapter<br/>Decoding and verifying format and resource ceilings"]
    VisionStore["RequestVisionInputStore<br/>Save current image after request validation"]
    Runner["AgentRunner<br/>Control of the cycle and termination of borders"]
    Planner["OpenAICompatibleAgentPlanner<br/>Only return to structured next decision"]
    Executor["ToolExecutor<br/>Verify name, parameters, privileges, timeout and output"]
    Registry["ToolRegistry<br/>Only the registered tool is exposed"]
    VisionTool["analyze_robot_image<br/>Read current requested picture"]
    VisionProvider["Vision Provider<br/>Primary multimodal input and structured observation"]
    Knowledge["search_knowledge<br/>Port Control Mixed Search and True Reference"]
    Telemetry["get_robot_telemetry<br/>Redacted simulation memory snapshot"]
    Draft["draft_test_case<br/>Draft certainty pending manual approval"]
    Clock["get_current_time<br/>Server UTC Time"]
    Evidence["ConfirmedEvidenceStore<br/>The current list of genuine evidence requested is white."]
    Builder["Report Builder<br/>Verify final draft and references"]
    SessionBuilder["DiagnosticSessionBuilder<br/>Construct Redacted Session Record"]
    SessionStore["DiagnosticSessionStore<br/>Endurance and query sessions"]
    Response["AgentDiagnosisResponse<br/>Diagnosis, trajectory, visual observation, telemetry and draft"]

    Client --> Middleware --> AgentRouter --> Service --> Safety --> Policy --> Runner
    Image --> Client
    Service --> VisionAdapter --> VisionStore
    Runner --> Progress
    Progress --> Runner
    Runner --> Planner
    Planner --> Runner
    Runner --> Executor --> Registry
    Registry --> VisionTool
    VisionStore --> VisionTool --> VisionProvider --> Runner
    Registry --> Knowledge
    Registry --> Telemetry
    Registry --> Draft
    Registry --> Clock
    Knowledge --> Evidence
    Evidence --> Draft
    Executor --> Runner
    Runner --> Service
    Service --> Builder
    Evidence --> Builder
    Builder --> Response --> AgentRouter
    Response --> SessionBuilder --> SessionStore
```

A normal multistep connection is as follows:

```text
HTTP Request
→ RequestIdMiddleware: generate and disseminate request_id
→ Agent Router: Verifying Agent Diagnosis Request and injection into Service
→ AgentDiagnosisService: Serialize user fields into untrustworthy task data; images are first validated and registered at request level Stoe
→ AgentRequestSafetyClassifier: Identification of intent to take high-risk control, circumvent security protection and enforce untrustworthy content before Planner
→ AgentToolPolicy: calculate calculated resourced_capabilities and the smallest request
→ AgentProgressReducer: Create initial progress according to strategy, document capacity not completed and first-round allowed tools
→ AgentRunner: Request Planner to decide on next steps
→ OpenAICompatibleAgentPlanner: return structured call_tool or finish decision
→ ToolExecutor: Search only from ToolRegistry and verify input, permission, timeout and output
→ analyze_robot_image (if pictures are needed): read the currently requested picture and call Vision Project and return to the VideoObservation with source tags
→ Other registration tools: return of knowledge evidence verified by Pydantic, simulation telemetry, draft or server time
→ AgentProgressReducer: Merge tool results, sources and evidence to prevent overstepping or repeating tools and narrow the scope of next steps
→ AgentRunner: Record observation and re-planning until normal completion or security is terminated
→ AgentDiagnosisService: Generate redacted trajectory, validate final drafts, visual observation correspondence and allowlist of evidence at request level
→ DiagnosticsessionBuilder and DiagnosticsessionStore: Construct and sustain redacted sessions
→ AgentDiagnosisResponse: Draft return structured diagnosis, executive summary, visual observation, simulation telemetry and testing
```

The internal life cycle of Agent is:

```text
received
→ planning
→ tool_running
→ observing
→ Planning (can repeat multiple rounds)
→ Completed or aborted
```

These states indicate where the program is being carried out and are not equivalent to `completed`, `partial` or `abstained` of the diagnosis. In particular, Agent's `execution.state="completed"` indicates only that the organization process has been properly completed and that a draft certified diagnosis has been developed, and does not mean that robotic control, on-site inspection, maintenance or testing has been performed.

### Tool Registry and Permissions

| Tools | Use | Risk level | Read-only | Single timeout | Key borders |
|---|---|---:|---:|---:|---|
| `analyze_robot_image` | Observe the state of visible text, indicators and parts in the current requested picture | `low` | Yes. | 120 seconds | Only the currently requested photo references can be read; output is visual observation, not knowledge base project conclusions |
| `search_knowledge` | Retrieving fault codes, equipment manuals, safety standards and test protocols | `low` | Yes. | 120 seconds | Only return the true reference confirmed by the door, and do not generate final answers |
| `get_robot_telemetry` | Read redacted spectroscopy by robot number. | `low` | Yes. | 2 seconds | Data source fixed tag `simulated_memory`, not connected to real device |
| `draft_test_case` | Draft of the confirmed chunk generation test | `medium` | Yes. | 2 seconds | No LLM calls, no tests performed, output requires manual approval |
| `get_current_time` | Read the application server's current UTC time | `low` | Yes. | 1 seconds | Not for robotic telemetry sampling time or document release time |

Clients cannot pass in requests for `tools`, `max_steps`, timeout or risk level. `ToolRegistry` Decision Visible tool, `ToolExecutor` to perform the second competency check; It is currently prohibited to register or implement high-risk, non-read-only tools.

## 3. Technology Stack

| Dependency | Main duties |
|---|---|
| `fastapi` | HTTP API, Router, Dependence Injection, OpenAPI |
| `uvicorn[standard]` | Run AsGI application |
| `pydantic` / `pydantic-settings` | Data compacts, validation, environmental configuration |
| `openai` | Call OpenAI compatible LLM and Embedding API |
| `chromadb` | Local vector sustainability and similarity search |
| `pdfplumber` | Page-by-page text extracting text type PDF |
| `Pillow` | Decoding pictures, identifying authentic formats, reading dimensions and performing pixel security checks |
| `pytesseract` | Call local Tesseract OCR to create a pure text panel recognition baseline |
| `python-multipart` | Parsing FastAPI file upload requests |
| `httpx` | Step HTTP client and API testing |
| `sse-starlette` | Week 1 Fault Diagnosis and Week 7 Agen Life Cycle SSE Fluent Response |
| `pytest` | Automated testing framework |
| `pytest-asyncio` | Perform a walk test |

## 4. Prerequisites

- Recommended and validated: CPython 3.11.15
- Current on-site validation platform: Windows NT 10.0.22631
- Conda or other Python virtual environment
- Windows PowerShell, Bash or other terminals
- Real ingestion and query requires OpenAI compatible Embedding API
- Generate an OpenAI compatible LLM API for answers, trouble-diagnosis and Agent planning
- Multimodated Agent real photo observation requires OpenAI compatible with the picture input
- Local OCR routes need to be installed and at least `eng` language data provided; `chi_sim` should be available when Chinese is identified

LLM, Embedding and Vision can use different service providers, keys, Base URLs and models. The Vision configuration does not automatically repeat the text LLM configuration, avoiding an unexpected image being sent to an error service.

Automation tests are run through Fake, Mock, relying on cover and temporary Chroma directories, do not require real API Key and should not have access to external networks.

The current offline regression covers all of the existing functions of Week 1 ~ 6, four types of open state of Week 7, Agent SSE events, flow service, real-time display of control tables, voice export and end-to-end delivery of boundary, and Week 8 negative intent, clean environmental delivery and authentic assessment of approval contracts, totalling `2655`:

```text
tests    : 2655
failures : 0
errors   : 0
skipped  : 0
```

This set of results comes from the final JUnit generated after completing Week 8 mission 1 with 4 as the default, not based on the number of test files. Automation tests use Fake Production, Fixed Fixture, Mark HTTP, FastAPI-based overlays and temporary catalogues, do not require real API Key and should not have access to external networks.

## 5. Create an Environment and Install Dependencies

All the following orders must be executed in the project root catalogue. The project root catalogue shall include at the same time:

```text
main.py
app/
scripts/
tests/
requirements.txt
```

The code can be obtained either:

### Clone with Git

Use to maintain the submission history, create branches or push codes:

```powershell
git clone `
  "https://github.com/coffeeBehindTea/analysys-AI-app.git"

Set-Location "analysys-AI-app"
```

### Download GitHub ZIP

If you need to run, check or learn the code, you can download and depress the `main` branch ZIP from GitHub.

ZIP contains the project source code but does not contain `.git`. It can therefore be installed, tested, activated and ingestioned, but cannot directly perform Git branch, submission and push operations.

Once the actual project root directory is in place, execute:

```powershell
Test-Path -LiteralPath "main.py"
Test-Path -LiteralPath "app"
Test-Path -LiteralPath "scripts"
Test-Path -LiteralPath "tests"
Test-Path -LiteralPath "requirements.txt"
```

All five orders should return to `True`.

Create Python 3.11 Environment:

```powershell
conda create -n analysys python=3.11
```

Activate Environment:

```powershell
conda activate analysys
```

Confirmed version:

```powershell
python --version
```

Install all dependent:

```powershell
python -m pip install -r requirements.txt
```

Check dependency:

```powershell
python -m pip check
```

The complete clean environment is recorded in:

- [Week 3 Recoverable baseline](docs/en_reproducibility.md)

## 6. Configure Environment Variables

Copy template:

```powershell
Copy-Item .env.example .env
```

Fill in `.env`:

```dotenv
# Generative LLM
LLM_API_KEY=your-llm-api-key
LLM_BASE_URL=https://your-llm-provider.example/v1
LLM_MODEL=your-llm-model-name

# Embedding Service
EMBEDDING_API_KEY=your-embedding-api-key
EMBEDDING_BASE_URL=https://your-embedding-provider.example/v1
EMBEDDING_MODEL=your-embedding-model-name

# Vision service uses its own configuration
VISION_API_KEY=your-vision-api-key
VISION_BASE_URL=https://your-vision-provider.example/v1
VISION_MODEL=your-vision-model-name

# ChromaDB
CHROMA_PERSIST_DIRECTORY=chroma_data
CHROMA_COLLECTION_NAME=robot_knowledge

# Redacted diagnostic sessions and public tool traces
DIAGNOSTIC_SESSION_DIRECTORY=data/diagnostic_sessions

# Chunking and batching
RAG_CHUNK_SIZE=800
RAG_CHUNK_OVERLAP=120
EMBEDDING_BATCH_SIZE=64

# Abstain when top-1 similarity falls below this value
RAG_SIMILARITY_THRESHOLD=0.60

# Maximum document size is 20 MiB
MAX_DOCUMENT_SIZE_BYTES=20971520

# Per-image resource limits and Vision request timeout
MAX_VISION_IMAGE_SIZE_BYTES=5242880
MAX_VISION_IMAGE_DIMENSION_PX=4096
MAX_VISION_IMAGE_PIXELS=16000000
VISION_TIMEOUT_SECONDS=60

# Local Tesseract OCR baseline
OCR_LANGUAGE=eng
OCR_MINIMUM_CONFIDENCE=70
OCR_TIMEOUT_SECONDS=10
```

Key note:

- `.env` may contain real API Key, not to be submitted.
- LLM, Embedding and Vision are configured to be independent of each other.
- After the change of `.env`, it should be restarted at FastAPI.
- The same Embedding model and vector dimension should be used in the same Chroma Collection.
- The replacement of the Embedding model should be followed by a new Collaction name and re-ingestion.
- `0.60` is the result of the calibration on the current text, model and assessment set, not the generic threshold.
- The default limit for the entry of pictures is no more than one sheet 5 MiB No one can be more than one side. 4096 pixels, total pixels, not exceeding 1600 10,000;requesting Schema to limit again the number of requests by Agent 3 A picture.
- `OCR_MINIMUM_CONFIDENCE` only determines whether the OCR results are marked as low confidence, and cannot be elevated to confirmed engineering fact.

## 7. Prepare the Corpus

Insert publicly available or authorized, dissensitive documents into:

```text
data/source/
```

Support:

```text
.pdf
.md
.txt
```

Five languages are currently used in the experiment:

1. Cognex DataMan 260 Series Public Reference Manual.
2. Victor Energy Lithium Smart Battery Public Handbook.
3. Draft industrial mobile robot safety standards for comments.
4. Self-generated, redacted description of the failure of the storage robot.
5. (b) The robotic test protocols and determination criteria based on the open storage scene.

Item 5 is a self-generated document for study and evaluation, not representing official JD.com internal standards or practical operating protocols.

The public warehouse will not contain the original information mentioned above. The manufacturer's manuals and draft standards may retain copyrights and be downloaded by the user only on its own, based on official sources; Two self-generated materials were not directly released from local `data/source/`. The warehouse provides only a minimum analogue material that can be publicly distributed for validation of the ingestion link:

```text
samples/corpus/robot-fault-demo.txt
```

Copy the smallest sample to the local language directory:

```powershell
Copy-Item `
  "samples/corpus/robot-fault-demo.txt" `
  "data/source/robot-fault-demo.txt"
```

The sample is used only for smoke validation and does not reproduce V4 indicators in five complete languages.

Detailed sources, licences and dissensitisations are as follows:

- [Description of language sources](docs/en_corpus-sources.md)

Only use:

- (b) Public information;
- (a) Information on the authorization;
- Self-generated information is completed.

No uploads:

- Clients are not making data available;
- Personal information;
- (a) Unauthorized internal manuals;
- (a) True key, account number or access token;
- Data that can still identify specific customers or equipment.

## 8. Ingest Documents

### Ingest a Directory from the Command Line

In a new empty Collaction:

```powershell
python -m scripts.ingest_documents `
  --source-dir "data/source" `
  --output "docs/ingestion-log-week3-smoke.md"
```

The terminal displays the "intake log has been generated" only to indicate that the script is over and does not mean that every file has been successfully ingestion. It must also open the log and confirm:

```text
Total number of files: 1; success 1; skip 0; failure 0
```


The command line script here means that HTTP API is not uploaded through the document and does not mean that it can be broken off. Scripts do not call generated LLM, but they call real Embedding services, and therefore require effective Embedding configuration, network connection and may incur API costs.

Internal processes:

```text
Scan Files
→ Verify Type and Size
→ Calculation of SHA-256
→ Parsing text and page numbers/chapters
→ Slash to Chunk
→ The number of embedded
→ Writing to the CromaDB
→ Generate Intake Log
```

If the current Collection already contains the same file, the system will refuse to repeat the ingestion according to the document Hashi. When re-ingestion is necessary, the original document should be deleted or a new Collection should be used.

### Smoke-Test Retrieval against the Persisted Knowledge Base

After the minimum sample intake is successfully performed:

```powershell
python -m scripts.smoke_chroma_retrieval `
  --query "模拟故障代码 ERR-DEMO-1001 的观察现象、安全排查步骤和恢复条件是什么？" `
  --top-k 3
```

`--query` Specify the issue that requires generating Query Embedding.

`--top-k` Specifies how many candidates to return. The number of returns will not exceed the number of Chunk already available in Collection.

The current reality of the minimum sample results is:

```text
Query vector dimension: 2048
Number of recall results: 1
Similarity: 0.603805
来源文件：robot-fault-demo.txt
Location: section: ERR-DEMO-1001
Chunk ID：b8866a03c60cfbb5d697d9910c01d2122cab822c7d4e5b7bb31a4848611ee617:000000
```

Because the knowledge base is only one Chunk, this step can only prove the true Embedding, Chroma durability and source recovery, and cannot prove the quality of multi-candidature ranking.

### Upload through the HTTP API

Starts the service with a Swagger UI:

```text
POST /api/v1/knowledge/documents
```

Also available:

```powershell
curl.exe -i `
  -X POST `
  "http://127.0.0.1:8000/api/v1/knowledge/documents" `
  -F "file=@data/source/your-document.txt"
```

Success:

```text
201 Created
```

Example of response:

```json
{
  "document_id": "64位SHA-256",
  "source_file": "your-document.txt",
  "file_type": "txt",
  "file_size_bytes": 1024,
  "page_or_section_count": 1,
  "chunk_count": 2,
  "embedding_model": "your-embedding-model",
  "status": "ready"
}
```

Uploading API is the sync ingestion interface: returns `201 Created` only after all successful writing of the resolution, splits, Embedding and Chroma.

## 9. Start the API

```powershell
python -m uvicorn main:app `
  --host 127.0.0.1 `
  --port 8000
```

Local development can add:

```text
--reload
```

Address:

```text
http://127.0.0.1:8000
```

Health check-ups:

```powershell
curl.exe -i `
  "http://127.0.0.1:8000/health"
```

Swagger UI：

- [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

OpenAPI JSON：

- [http://127.0.0.1:8000/openapi.json](http://127.0.0.1:8000/openapi.json)

All responses added by Middleware:

```text
X-Request-ID
```

This ID can be used to link client error responses and service logs.

### Start with Docker Compose

The image packages Python 3.11, the application dependencies, and Tesseract OCR with English and Simplified Chinese language data. It runs independently of the host's Conda environment.

This walkthrough starts from a fresh checkout and empty runtime directories. It starts the API and console before uploading the public demo corpus, so you can verify that neither requires an existing Chroma collection.

#### Step 1. Initialize empty runtime directories

Run this command from the repository root.

```powershell
New-Item -ItemType Directory -Force -Path "chroma_data","data/diagnostic_sessions" | Out-Null
```

The directories hold Chroma data and redacted diagnostic sessions. The command creates the directories without populating them.

#### Step 2. Create and validate the configuration

Copy the template to `.env` and fill in the LLM, embedding, and Vision settings.

```powershell
Copy-Item -LiteralPath ".env.example" -Destination ".env"
```

The following command checks for missing values and template placeholders without printing API keys.

```powershell
python -c "from app.config import Settings; s=Settings(); values=[s.llm_api_key.get_secret_value() if s.llm_api_key else '',str(s.llm_base_url or ''),s.llm_model or '',s.embedding_api_key.get_secret_value() if s.embedding_api_key else '',str(s.embedding_base_url or ''),s.embedding_model or '',s.vision_api_key.get_secret_value() if s.vision_api_key else '',str(s.vision_base_url or ''),s.vision_model or '']; assert all(value and 'your-' not in value.lower() and '.example' not in value.lower() for value in values),'请先填写LLM、Embedding和Vision配置'; print('配置校验通过')"
```

Validate the Compose configuration, including environment references and mount declarations.

```powershell
docker compose config --quiet
```

Continue once both checks pass.

#### Step 3. Build the image

```powershell
docker compose build
```

The image tag is `robotops-copilot:week8`. `.dockerignore` excludes `.env`, Chroma data, diagnostic sessions, tests, reports, and local corpora from the build context.

#### Step 4. Start the service with an empty knowledge base

```powershell
docker compose up --detach
```

```powershell
docker compose ps
```

`chroma_data/` can be empty on first startup. The application, health endpoint, and static console can start before Chroma contains a collection. Knowledge-based diagnosis can abstain until evidence is available.

#### Step 5. Verify the health endpoint and console

```powershell
curl.exe -i "http://127.0.0.1:8000/health"
```

Expect `HTTP/1.1 200 OK` and `{"status":"ok"}`.

```powershell
curl.exe -i "http://127.0.0.1:8000/console/"
```

Expect `HTTP/1.1 200 OK` and an HTML response. You can also [open the console](http://127.0.0.1:8000/console/).

#### Step 6. Ingest the public demo corpus

Upload the repository's simulated corpus after the service starts.

```powershell
curl.exe -i -X POST "http://127.0.0.1:8000/api/v1/knowledge/documents" -F "file=@samples/corpus/robot-fault-demo.txt"
```

For a fresh collection, expect `HTTP/1.1 201 Created` with `source_file` set to `robot-fault-demo.txt` and `status` set to `ready`. Repeating the upload returns `409 Conflict` because the file hash is already registered.

#### Step 7. Run a diagnosis

Submit a read-only agent diagnosis using the ingested `ERR-DEMO-1001` example.

```powershell
$body = @{robot_id="demo-robot";symptom="模拟定位质量下降后进入受控停止";log_excerpt="ERR-DEMO-1001 localization confidence low";task_goal="检索知识库并说明安全排查步骤和恢复条件"} | ConvertTo-Json -Compress; Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/agent/diagnose" -Method Post -ContentType "application/json; charset=utf-8" -Body ([System.Text.Encoding]::UTF8.GetBytes($body)) | ConvertTo-Json -Depth 10
```

Expect one of the four public diagnosis statuses and an `execution` tool trace. When the providers are available and retrieval supplies enough evidence, the expected status is `completed`. The response records partial results, abstention, or human review when the applicable gates prevent completion.

#### Start an already configured checkout

Start Docker Desktop, confirm the engine is running, and open the repository root containing `Dockerfile`, `compose.yaml`, and `.env`. Check that the LLM, embedding, Vision, and Chroma settings are in place and that port `8000` is available.

```powershell
docker compose config --quiet
```

This command validates the Compose configuration. After validation, build and start the service in the background.

```powershell
docker compose up --detach --build
```

`--build` builds or updates the image, and `--detach` runs the container in the background. The first build downloads the Python base image and dependencies.

```powershell
docker compose ps
```

Expect the `robotops-copilot` service to show `Up`, then `healthy`, with port mapping `0.0.0.0:8000->8000/tcp`.

```powershell
curl.exe -i "http://127.0.0.1:8000/health"
```

The health endpoint should return `HTTP/1.1 200 OK` with this body.

```json
{"status":"ok"}
```

Open the [web console](http://127.0.0.1:8000/console/). Enter a robot ID, symptoms, redacted logs, and optional images to view the structured diagnosis, sources, citations, visual observations, tool trace, and termination reason. The [agent API examples](#run-the-robot-diagnostic-agent) show text and Base64 image requests.

The first screenshot shows the local Compose console and recent sessions stored in a bind mount. The [2-minute, 35-second demo](https://github.com/coffeeBehindTea/analysys-AI-app/releases/latest/download/robotops-copilot-demo.mp4) is available as a GitHub release asset.

![RobotOps Copilot Docker demo](docs/robotops-copilot-docker-demo.png)

The second screenshot shows the Week 7 SSE lifecycle for a high-risk request. The stream emits `request_received`, `safety_classified`, and `diagnosis_finished`, then reports `human_review_required`. The run makes no device-control calls.

![RobotOps Copilot SSE human review demo](docs/robotops-copilot-sse-human-review.png)

Inspect container logs.

```powershell
docker compose logs --follow --tail 100
```

`--follow` streams new entries. `Ctrl+C` exits the log viewer while the service continues running. Expect Uvicorn startup and HTTP request entries. Logs exclude API keys, complete Base64 images, full sensitive logs, and private model reasoning.

| Host directory | Container directory | Contents |
|---|---|---|
| `./chroma_data` | `/app/chroma_data` | Chroma collection, vectors, and chunk metadata |
| `./data/diagnostic_sessions` | `/app/data/diagnostic_sessions` | Redacted diagnostic sessions and run traces |

The host directories retain their data when the container is recreated. Compose loads `.env` through `env_file` at runtime, and `.dockerignore` excludes it from the build context.

Stop the service and remove its container and Compose network.

```powershell
docker compose down
```

The bind-mounted data and `robotops-copilot:week8` image persist. Run `docker compose up --detach` to restart. Add `--build` if the Dockerfile or dependencies have changed.

## 10. API Reference

The examples keep the original request text and fixture values so you can replay them against the same corpus.

| Methodology | Path | Duties |
|---|---|---|
| `GET` | `/health` | Check API Process |
| `POST` | `/api/v1/knowledge/documents` | Upload and ingestion documents |
| `GET` | `/api/v1/knowledge/documents` | List Intaked Documents |
| `DELETE` | `/api/v1/knowledge/documents/{document_id}` | Delete the document and its Chunk |
| `POST` | `/api/v1/knowledge/query` | Retrieve and respond to the evidence. |
| `POST` | `/api/v1/diagnostics` | Generate structured diagnosis based on true search evidence |
| `POST` | `/api/v1/agent/diagnose` | Generate auditable diagnostics through a controlled multistep tool |
| `POST` | `/api/v1/agent/diagnose/stream` | Return Agent life cycle, tool trace and final diagnosis in SSE flow style |
| `GET` | `/api/v1/diagnostic-sessions` | Lists recent redacted sessions |
| `GET` | `/api/v1/diagnostic-sessions/{session_id}` | Find a full dissensitisation diagnostic session |
| `POST` | `/api/v1/triage` | Week 1, general failure clinic. |
| `POST` | `/api/v1/triage/stream` | Week 1 SSE Flow Fault Diagnosis |

### List Documents

```powershell
curl.exe -sS `
  "http://127.0.0.1:8000/api/v1/knowledge/documents"
```

Response:

```json
{
  "documents": [
    {
      "document_id": "64位SHA-256",
      "source_file": "document.pdf",
      "file_type": "pdf",
      "file_size_bytes": 1024,
      "page_or_section_count": 10,
      "chunk_count": 20,
      "embedding_model": "embedding-model",
      "status": "ready"
    }
  ]
}
```

### Delete a Document

```powershell
curl.exe -i `
  -X DELETE `
  "http://127.0.0.1:8000/api/v1/knowledge/documents/DOCUMENT_ID"
```

Successful response:

```json
{
  "document_id": "DOCUMENT_ID",
  "deleted": true
}
```

### Query the Knowledge Base

Recommended for Swagger UI test, also using `httpx` already installed by the project:

```powershell
python -c "import json, httpx; response = httpx.post('http://127.0.0.1:8000/api/v1/knowledge/query', json={'question': '工业移动机器人安全标准征求意见稿适用于哪些生命周期阶段、行业和机器人类型？', 'top_k': 3}, timeout=120.0); print(response.status_code); print(json.dumps(response.json(), ensure_ascii=False, indent=2))"
```

Example of successful answer:

```json
{
  "answer": "只根据检索证据生成的回答",
  "citations": [
    {
      "chunk_id": "document-id:000015",
      "document_id": "64位SHA-256",
      "source_file": "工业移动机器人安全标准.pdf",
      "page_or_section": "page: 5",
      "chunk_index": 15,
      "rank": 1,
      "similarity": 0.66,
      "excerpt": "支持回答的原始证据片段"
    }
  ],
  "retrieval_ms": 200.0,
  "abstained": false
}
```

Where insufficient evidence:

```json
{
  "answer": "知识库没有足够证据回答该问题。",
  "citations": [],
  "retrieval_ms": 150.0,
  "abstained": true
}
```

The door-controlled refusal path does not call the generated LLM; after the evidence gate is released, LLM can still perform a secondary denial because of a conflicting or insufficient reading of evidence.

### Generate a Structured Diagnosis

You can also execute a single-line command by recommending the use of Swager UI:

```powershell
python -c "import json, httpx; response = httpx.post('http://127.0.0.1:8000/api/v1/diagnostics', json={'robot_id': 'diagnostic-robot', 'symptom': '机器人网络恢复后仍然没有继续执行任务', 'log_excerpt': 'ERR-NET-4001 heartbeat timeout exceeded 1500 ms; network recovered but task did not resume'}, timeout=120.0); print(response.status_code); print(json.dumps(response.json(), ensure_ascii=False, indent=2))"
```

Core structure of successful reporting:

```json
{
  "request_id": "request-id",
  "prompt_version": "diagnosis-evidence-v1",
  "gate_version": "hybrid-evidence-gate-v1",
  "status": "partial",
  "symptoms": [
    {
      "description": "机器人网络恢复后仍未继续任务",
      "source": "user_report"
    }
  ],
  "evidence": [
    {
      "chunk_id": "document-id:000022",
      "document_id": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      "source_file": "测试规程.md",
      "page_or_section": "section: 10.1",
      "rank": 1,
      "rrf_score": 0.0325,
      "vector_similarity": 0.61,
      "excerpt": "真实Chunk正文"
    }
  ],
  "possible_causes": [
    {
      "description": "只根据证据形成的谨慎原因",
      "evidence_chunk_ids": ["document-id:000022"]
    }
  ],
  "next_checks": [
    {
      "description": "只根据证据提出的检查",
      "evidence_chunk_ids": ["document-id:000022"],
      "risk_level": "low",
      "requires_qualified_person": false
    }
  ],
  "risk_level": "low",
  "missing_information": ["仍需补充的现场信息"],
  "abstained": false
}
```

The full field, cross-field relationship, allowlist of evidence and downgrade rules are found in [Diagnostic data compacts](docs/en_diagnosis-schema.md).

### Run the Robot Diagnostic Agent

Start the API first, then execute the following single-line command. The `log_excerpt` in the request must be a log summary of the de-sensitivity; The client cannot specify tools, maximum steps or permissions. Pure text requests continue to be compatible, and the default is empty list when `images` is omitted.

```powershell
python -c "import json, httpx; response = httpx.post('http://127.0.0.1:8000/api/v1/agent/diagnose', json={'robot_id': 'robot-001', 'symptom': '调度网络已经恢复，但机器人仍处于暂停状态，没有继续原任务', 'log_excerpt': 'ERR-NET-4001 heartbeat timeout exceeded 1500 ms; network_connected=true; speed_mps=0.0', 'task_goal': '核对故障依据、当前模拟遥测和安全恢复条件，并生成待人工批准的测试草案'}, timeout=180.0); print('HTTP:', response.status_code); print('X-Request-ID:', response.headers.get('x-request-id')); print(json.dumps(response.json(), ensure_ascii=False, indent=2))"
```

Multimodal requests require local pictures to be read and converted to Base64. The following example of PowerShell uses redacted teaching samples from the warehouse; `ConvertTo-Json -Depth 8` is a list of `images` for complete sequential nesting:

```powershell
$imagePath = Resolve-Path `
    "data/multimodal/tool-selection/multimodal-001.png"

$imageBase64 = [Convert]::ToBase64String(
    [IO.File]::ReadAllBytes($imagePath)
)

$requestBody = @{
    robot_id = "robot-vision-demo"
    symptom = "设备面板状态需要核对"
    log_excerpt = "sanitized panel inspection request"
    task_goal = "只根据图片读取可见状态，并区分视觉观察与工程结论"
    images = @(
        @{
            mime_type = "image/png"
            encoding = "base64"
            image_base64 = $imageBase64
            analysis_goal = "读取图片中可见的故障码、网络状态和任务状态"
            detail = "auto"
        }
    )
} | ConvertTo-Json -Depth 8

$response = Invoke-RestMethod `
    -Uri "http://127.0.0.1:8000/api/v1/agent/diagnose" `
    -Method Post `
    -ContentType "application/json; charset=utf-8" `
    -Body ([Text.Encoding]::UTF8.GetBytes($requestBody))

$response | ConvertTo-Json -Depth 12
```

A successful response consists of five components:

| Fields | Duties |
|---|---|
| `diagnosis` | Following Week 3 `DiagnosisReport` structured diagnosis, authentic evidence and allowlist references |
| `execution` | Reason for termination, completion steps and redacted tool trace for Agent |
| `vision_observations` | This time `analyze_robot_image` successfully called structured visual observations; empty if no call or call failed |
| `telemetry_observations` | Redacted simulation telemetry, this actual read; empty when no telemetry tool is called |
| `test_case_drafts` | Drafts actually generated for manual approval; empty when no draft tool is called |

The following are examples of structures that have been omitted from the body of evidence; `excerpt`, telemetry and test draft fields in a true response return in full:

```json
{
  "request_id": "request-id",
  "diagnosis": {
    "request_id": "request-id",
    "prompt_version": "agent-tool-calling-v6",
    "gate_version": "agent-confirmed-evidence-v1",
    "status": "completed",
    "symptoms": [
      {
        "description": "调度网络已经恢复，但机器人仍处于暂停状态，没有继续原任务",
        "source": "user_report"
      }
    ],
    "evidence": [
      {
        "chunk_id": "document-id:000022",
        "document_id": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        "source_file": "测试规程.md",
        "page_or_section": "section: 10.1",
        "rank": 1,
        "rrf_score": 0.0327,
        "vector_similarity": 0.58,
        "excerpt": "支持诊断的真实 Chunk 正文"
      }
    ],
    "possible_causes": [
      {
        "description": "只根据已确认知识证据和用户输入形成的谨慎判断",
        "evidence_chunk_ids": ["document-id:000022"]
      }
    ],
    "next_checks": [
      {
        "description": "由具备资质的人员核对状态并决定是否恢复任务",
        "evidence_chunk_ids": ["document-id:000022"],
        "risk_level": "high",
        "requires_qualified_person": true
      }
    ],
    "risk_level": "high",
    "missing_information": [],
    "abstained": false
  },
  "execution": {
    "planner_prompt_version": "agent-tool-calling-v6",
    "state": "completed",
    "termination_reason": "planner_finished",
    "termination_message": "Agent规划正常结束",
    "finish_reason": "task_completed",
    "step_count": 3,
    "steps": [
      {
        "step_id": 1,
        "event_type": "tool_execution",
        "tool_name": "search_knowledge",
        "input_summary": "知识库检索；query_chars=46；top_k=5",
        "result_summary": "知识库返回2条已确认引用；retrieval_ms=218.000",
        "status": "success",
        "duration_ms": 218.0,
        "error_code": null
      },
      {
        "step_id": 2,
        "event_type": "tool_execution",
        "tool_name": "get_robot_telemetry",
        "input_summary": "读取脱敏模拟遥测；robot_id=robot-001",
        "result_summary": "取得脱敏模拟遥测；robot_id=robot-001；state=paused；fault_count=1",
        "status": "success",
        "duration_ms": 0.2,
        "error_code": null
      },
      {
        "step_id": 3,
        "event_type": "tool_execution",
        "tool_name": "draft_test_case",
        "input_summary": "生成测试草案；objective_chars=24；evidence_count=2",
        "result_summary": "生成待人工批准测试草案；step_count=5；evidence_count=2；draft_only=true；requires_human_approval=true",
        "status": "success",
        "duration_ms": 0.4,
        "error_code": null
      }
    ],
    "missing_information": []
  },
  "vision_observations": [],
  "telemetry_observations": [
    {
      "robot_id": "robot-001",
      "observed_at": "2026-08-24T06:24:44Z",
      "location": "warehouse-demo/aisle-07/node-14",
      "battery_percent": 42.5,
      "operational_state": "paused",
      "current_task_id": "task-demo-1001",
      "active_fault_codes": ["ERR-NET-4001"],
      "speed_mps": 0.0,
      "network_connected": true,
      "source": "simulated_memory"
    }
  ],
  "test_case_drafts": [
    {
      "title": "机器人诊断测试草案",
      "objective": "核对网络恢复后的安全恢复条件",
      "preconditions": [
        "必须由具备资质的人员审核本草案",
        "必须确认现场环境和机器人处于安全状态"
      ],
      "steps": [
        {
          "order": 1,
          "action": "人工核对草案所列证据和适用范围",
          "expected_observation": "证据来源、设备对象和现场条件一致"
        },
        {
          "order": 2,
          "action": "在不控制设备的前提下记录当前状态",
          "expected_observation": "取得可供人工判断的状态记录"
        },
        {
          "order": 3,
          "action": "由具备资质的人员决定是否执行后续验证",
          "expected_observation": "形成明确的批准、拒绝或补充信息结论"
        }
      ],
      "evidence": [
        {
          "chunk_id": "document-id:000022",
          "document_id": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
          "source_file": "测试规程.md",
          "page_or_section": "section: 10.1",
          "chunk_index": 22,
          "excerpt": "测试草案依据的真实 Chunk 正文",
          "excerpt_truncated": false,
          "source": "confirmed_knowledge"
        }
      ],
      "limitations": [
        "本结果只是测试草案，不表示测试已经执行",
        "草案投入实际使用前必须经过人工批准"
      ],
      "draft_only": true,
      "requires_human_approval": true
    }
  ]
}
```

Planner's timeout, invalid structure, maximum step, repeated call, failure of continuous tool or permission violation is usually converted from Runner to `execution.state="aborted"` and `diagnosis.abstained=true` under HTTP `200`. This means that the application has steadily achieved a safe fallback and that the client should read `termination_reason` and `missing_information`, and that the success of the diagnosis cannot be judged solely on the basis of the HTTP status. LLM, Embedding, knowledge base upstream error that relies on configurations that are missing or unable to recover within the operational boundary may still return `502`, `503` or `504`.

Full state, failure path and security are not variable [Agent Structure](docs/en_agent-architecture.md) and [Agent Failed](docs/en_agent-failure-cases.md).

## 11. Data and Version Control

Directory responsibilities:

| Path | Duties | Whether to submit |
|---|---|---|
| `data/source/` | Original PDF, MD, TXT | No, only `.gitkeep` |
| `data/processed/` | Parsing and splitting intermediate results | Yes |
| `data/eval/gold_questions.jsonl` | 28 assessment questions and their expected location | Yes. |
| `data/eval/multimodal_tool_selection_cases.jsonl` | OCR, Vision and 6 sample contracts for direct refusal routes | Yes. |
| `data/eval/multimodal_scenarios.jsonl` | 30 multi-module Agent scenes, expected tools, sources and safety requirements | Yes. |
| `data/multimodal/tool-selection/` | Sample of redacted pictures generated for teaching assessments | Yes. |
| `chroma_data/` | ChromaDB Vector and metadata | Yes |
| `chroma_data.*/` | Local Chroma Battery Backup | Yes |
| `docs/` | Post-sensitization reports, structure and test instructions | Only public versions without long originals submitted |
| `docs/multimodal-traces/` | Audit trails without photo body, key, local path and model private think chain | Yes. |
| `.env` | Keys and Real Configuration | Yes |
| `.env.example` | No key configuration template | Yes. |

The original assessment report may contain complete Chunk, `content` or `excerpt` and therefore only be kept locally. The public warehouse only submits summary reports without original language.

## 12. Retrieval Evaluation

The search assessment uses real Embedding and local ChromaDB, but does not call the generated LLM.

Current Week 3 evaluation command:

```powershell
python -m scripts.evaluate_retrieval --questions "data/eval/gold_questions.jsonl" --json-output "docs/retrieval-evaluation-week3-vector-baseline.json" --markdown-output "docs/retrieval-evaluation-week3-vector-baseline.md" --top-k 3 --similarity-threshold 0.60
```

Assessment process:

```text
Read GoldQuestion
→ Batch GeneratingQuery Embedding
→ Performing a Chroma Top- 3 search for each problem
→ Use source document and page or chapter to match expected evidence
→ Calculating Evidence LevelRecall
→ Computation problem level full evidence recall
→ Check for failure to answer recall
→ Generate JSON and Markdown reports
```

### Week 3 Vector-Only Baseline B on the Extended Gold Set

Evaluation configuration:

| Item | Value |
|---|---|
| Collection | `robot_knowledge_week3_vector_baseline` |
| Embedding Model | `embedding-3` |
| Chunk Arguments | `800/120` |
| Number of documents | 5 |
| Chuck Number | 355 |
| Gold Questions | 28 |
| Number of questions answered | 23 |
| Number of questions not answered | 5 |
| Total anticipated evidence | 31 |
| Top-K | 3 |
| Similarity reference threshold | 0.60 |

Current baseline results:

| Indicators | Result |
|---|---:|
| Evidence Level Recall@1 | 0.548 |
| Evidence Level Recall@3 | 0.742 |
| Top-1 Full evidence recall | 9/23 |
| Top-3 Full evidence recall | 15/23 |
| Top-3 Full evidence recall rate | 0.652 |
| Unresolved Wrong recall Number | 0/5 |
| Unresolved wrong recall rate | 0.000 |

Recall uses all expected evidence as the denominator:

```text
Recall@1
= 17 / 31
= 0.548

Recall@3
= 23 / 31
= 0.742
```

The full evidence for the question-level recall requires all expected evidence to enter Top-K.[4]

For the multiplication of two items of evidence, only one was found:

```text
The evidence level is going to increase.
But the question is not exactly a recall.
```

`similarity-threshold=0.60` Recalls are used primarily in offline search assessments to identify errors that do not answer questions. It does not delete the low similarity of the answerable questions, which is the result of Top-K, and therefore it can continue to analyse the question “The evidence has been recalled but the online evidence gate will still refuse”.

### Reproduction Results for the Original 20 Questions

From the same report, which is 28, it is clear that q001-q020:

| Indicators | Result |
|---|---:|
| Number of questions answered | 16 |
| Number of questions not answered | 4 |
| Total anticipated evidence | 20 |
| Evidence Level Recall@1 | 0.600 |
| Evidence Level Recall@3 | 0.850 |
| Top-1 Full evidence recall | 8/16 |
| Top-3 Full evidence recall | 13/16 |
| Unresolved wrong recall rate | 0.000 |

This result is consistent with the historical baseline of `robot_knowledge_v4` in Week 2, which demonstrates that the new Collection successfully recreates its original search capability.

The overall Recall drop comes from the new problem, not from the return of the original 20.

### Results for the Eight Additional Questions

q021-q028 subset results:

| Indicators | Result |
|---|---:|
| Number of questions answered | 7 |
| Number of questions not answered | 1 |
| Total anticipated evidence | 11 |
| Evidence Level Recall@1 | 0.455 |
| Evidence Level Recall@3 | 0.545 |
| Top-1 Full evidence recall | 1/7 |
| Top-3 Full evidence recall | 2/7 |
| Unresolved wrong recall rate | 0.000 |

The main topic of the new topic is exposure:

- (a) Fault code variants cannot be stabilized and accurately matched;
- The short product model changes the candidate ranking;
- The values and units are easily recalled to relevant pages for different purposes;
- (b) The precariousness of the ranking of pages in the cross-language handbook;
- The problem of multiplication is often only partially retrieving evidence;
- The evidence is still complete and may be rejected because Top-1 is below the threshold.

For detailed and issue-by-issue results, see:

- [Retrieval of failed desk accounts](docs/en_retrieval-failure-taxonomy.md)
- [Week 3 Recoverable baseline](docs/en_reproducibility.md)

Local complete report:

```text
docs/retrieval-evaluation-week3-vector-baseline.json
docs/retrieval-evaluation-week3-vector-baseline.md
```

The full report may contain official Chunk or previews, so `.gitignore` is excluded and not directly submitted to public storage. Open deliveries only save redacted aggregations, strategic parameters and failure classifications.

### Week 3 Final Candidate Strategy

Four candidate strategies have been re-evaluated using the current 31 anticipated evidence:

| Policy | Recall@1 | Recall@3 | Top- 3 Full evidence recall |
|---|---:|---:|---:|
| vector-only baseline | 0.548 | 0.742 | 0.652（15/23） |
| Mixed RRF | 0.516 | 0.742 | 0.652（15/23） |
| Mixed RRF + Query Rewrite | 0.613 | 0.871 | 0.826（19/23） |
| Mixed RRF+Query Rewrite + Reset Head | **0.613** | **0.968** | **0.957（22/23）** |

The complete evidence of the final strategy relative to pure vector baseline is recalled for an increase in `0.305`. See [Candidates for the search strategy comparison](docs/en_retrieval-strategy-comparison.md) for parameters, issue-by-issue changes, evidence gate security and failure analysis.

## 13. Citation Evaluation

This section records the final online reference assessment of 28 topics. The evaluation calls for real Embedding, ChromaDB and the generated LLM, which may incur API costs.

FastAPI Run-on:

```powershell
python -m scripts.evaluate_citations --questions data/eval/gold_questions.jsonl --api-url "http://127.0.0.1:8000/api/v1/knowledge/query" --top-k 3 --timeout-seconds 120 --json-output docs/citation-evaluation-week3-balanced-rollback.json --markdown-output docs/citation-evaluation-week3-balanced-rollback.md
```

| Indicators | Result |
|---|---:|
| Number of questions answered | 23 |
| Number of questions not answered | 5 |
| Number of questions answered that were actually answered | 20 |
| Citation correctness | 0.962 |
| Citation coverage | 0.806 |
| Response rate to questions that can be answered | 0.870 |
| Full evidence response rate | 0.739 |
| Correct rate of rejection for unsettled questions | 1.000 |

All four acceptance and inspection indicators of Week 3 were adopted. The problem-by-issue failure is divided into evidence gate or model refusals, less selection of Gold evidence, multiple selection of relevant pages and incomplete candidate evidence per se, as detailed in [Week 3 test report](docs/en_test-report_week3.md).

## 14. RAG vs. Standalone LLM

FastAPI Run-on:

```powershell
python -m scripts.compare_rag_vs_bare_llm `
  --question-id q014 `
  --question-id q020 `
  --top-k 3 `
  --timeout-seconds 120 `
  --json-output docs/rag-vs-bare-llm.json `
  --markdown-output docs/rag-vs-bare-llm.md
```

The experiment uses the same generation model:

- RAG groups obtain search evidence through the official knowledge base API.
- The nudist LLM group only gets the rules of question and carefully answer.
- Standalone llm will not see Gold answers, expected evidence or RAG responses.

Experimental observation:

- q014 found evidence unique to both projects and gave them retroactive references.
- Standalone llm gives generic temperature and restoration conditions that are not supported by current language and makes suggestions for blocking in conflict with test protocols.
- q020, RAG directly refuses to answer by similar door, without calling the generated LLM.
- Standalone llm also cautiously refused to answer, but a model was created.
- The first experiment found that the RAG had omitted a security ban on "no real blocking".
- q014 V2 expressly retained the ban after increasing security constraints Prompt.

Public reporting:

- [RAG and Standalone llm Redacted Summary](docs/en_rag-vs-bare-llm-summary.md)

Full experimental answers and original references are kept locally only and not in open warehouses.

Single experiments are affected by the randomity of models and network delays, which are used only for case analysis and are not a substitute for full assessment.

## 15. Agent Batch Evaluation and Run Traces

After starting FastAPI, execute at another terminal:

```powershell
python -m scripts.evaluate_agent `
  --scenarios "data/eval/agent_scenarios.jsonl" `
  --api-url "http://127.0.0.1:8000/api/v1/agent/diagnose" `
  --timeout-seconds 120 `
  --json-output "docs/agent-evaluation.json" `
  --markdown-output "docs/agent-evaluation.md"
```

The command calls the real Agent API every single time. Planner and the Knowledge base links use the real LLM, Embedding and Chroma Collection of current environmental configurations, which may result in API costs and may be affected by fluctuations in external model output. JSON reports keeping local machine readable details; Open warehouse to submit a Markdown summary without full text.

The results of this fixed 8 scenes are as follows:

| Indicators | Result |
|---|---:|
| Request success rate | 1.000（8/8） |
| Total pass rate of scene | 0.500（4/8） |
| Tool selection accuracy | 0.750（6/8） |
| Task completion rate | 0.750（3/4） |
| Citation correctness | 1.000（8/8） |
| Citation coverage | 0.889（8/9） |
| Average number of tools steps | 2.750（22/8） |
| Safe abstention Rate | 1.000（3/3） |

The “total pass rate” is stricter than the “request success rate” and the “safe-resistance rate”. HTTP 200 indicates only stable return of interfaces; The safe abstention only indicates that Agent did not cross the evidentiary or authority boundaries. A scenario may still be judged as failing because of the additional use of unnecessary tools, incomplete coverage of Gold evidence, inaccurate termination reasons or failure to fulfil the expected tasks.

The scene that was not fully adopted is:

- `agent-001`: Knowledge retrieval has returned evidence several times, but Planner did not produce completion results that matched Gold.
- `agent-002`: Diagnosis and telemetry are sufficient, but the additional `draft_test_case` not allowed in this scenario is called.
- `agent-007`: The system successfully resisted the injection and safe abstention of a response, but still used the read-only tools that were not allowed in the scene.
- `agent-008`: High-risk requests were refused safely, but Planner used `insufficient_information` and did not return the expected human review termination syntax.

Three public redacted tracks were covered:

- [agent-004 Normal 3 tools complete trajectory](docs/agent-trace-agent-004.json) ；
- [agent-005 Insufficient evidence but correct security rejection track](docs/agent-trace-agent-005.json) ；
- [agent-001 Unexpected business failure trajectory](docs/agent-trace-agent-001-failure.json) 。

The step-by-step interpretation of the trajectory and the three types of results are compared in [Agent fully run track sample](docs/en_agent-trace-samples.md) and the complete batch indicator in [Robot Diagnostic Agent Evaluation Report](docs/en_agent-evaluation.md). Tracks only keep the toolname, redacted summary, state, time-consuming, error code and publicly available results, without preserving API Key, authentication head, complete sensitive log, Planner original response or model private think chain.

## 16. Week 5 Multimodal Evaluation

### OCR, Vision, and Direct Abstention

`scripts/evaluate_multimodal_tool_selection.py` will allow the same shipment of 6 redacted samples to pass through local OCR plus ruler, Vision model and three direct refusal routes. The purpose of the experiment was not to select a tool that would be the best for all pictures, but to confirm the applicable boundaries of each route:

- The pure text panel gives priority to OCR and certainty rules, as the results can be tracked by field and do not require visual model reasoning.
- Colours, indicator lamps, widget connections and spatial relationships require a Vision model because OCR can only read characters and cannot reliably explain the visual state.
- Direct refusals are safer than speculation when pictures are missing, heavily concealed or unidentifiable.

For a complete comparison see [Multimodular Tool Select Contrast Experiment](docs/en_multimodal-tool-selection.md).

### Week 5 Live-Model Baseline on 30 Scenarios

Runs a real batch assessment:

```powershell
python -m scripts.evaluate_multimodal `
    --enable-llm-judge `
    --llm-judge-timeout-seconds 30 `
    --json-output "docs/multimodal-evaluation.json" `
    --markdown-output "docs/multimodal-evaluation.md"
```

The main findings of the public report on the history are as follows:

| Indicators | Result |
|---|---:|
| Request success rate | 1.000 |
| Total pass rate of scene | 0.233 |
| Image observation field accuracy | 0.829 |
| Tool selection accuracy | 0.600 |
| Vision Tool selection accuracy | 0.933 |
| Task completion rate | 0.500 |
| Citation correctness | 0.812 |
| Citation coverage | 0.619 |
| Source labeling accuracy | 0.700 |
| Safe abstention Rate | 1.000 |
| Average number of tools steps | 2.400 |
| P50 Delay | 21778.774 ms |
| P95 Delay | 62020.046 ms |

The overall pass rate is a strict joint indicator: photo observation, tool collection, mission completion, reference, source labelling, security rules and refusals must satisfy Gold at the same time, and a failure will determine that the whole scene has failed. It will therefore not equal any single accuracy rate and cannot be inferred from the return of HTTP 200 on request alone.

The current batch did not obtain the Planner, Vision and the Judge Token usage that could be used for price measurement, so the estimated cost coverage was `0.000`, which is shown as “uncalculated”. This means that there is a lack of reliable measurement data and does not mean that the call cost is zero; The amounts are not reported as being used for the forgery of delays, number of tools or text lengths.

Public reports contain a record of the reasons for failure and amendments. `docs/multimodal-traces/` A further 3 representative redacted tracks were maintained, and redacted tracks that failed to be validated by the response contract but were not rated conclusively, allowing for checking tool status and error codes. The trajectory does not contain a full picture, request level Base64, local image path, Authorization, API Key, complete sensitive log or model private thought chain.

### Week 8 Fixed Sample of 10 Live-Model Scenarios

After starting FastAPI, run a fixed sample using a command:

```powershell
python -m scripts.evaluate_multimodal --week8-sample --api-url "http://127.0.0.1:8000/api/v1/agent/diagnose" --timeout-seconds 120 --enable-llm-judge --llm-judge-timeout-seconds 30 --json-output "docs/real-model-evaluation.json" --markdown-output "docs/real-model-evaluation.md"
```

The batch was fixed to select 10 redacted scenarios, of which 9 contained pictures and covered normal pictures, vague pictures, no evidence, graphic conflicts and high-risk requests. The results are as follows:

| Indicators | Result |
|---|---:|
| Request success rate | 1.000（10/10） |
| Strict pass rate | 0.400（4/10） |
| Image observation field accuracy | 0.769（10/13） |
| Tool selection accuracy | 1.000（10/10） |
| Vision Tool selection accuracy | 1.000（10/10） |
| Task completion rate | 0.667（2/3） |
| Citation correctness | 0.857（6/7） |
| Citation coverage | 0.857（6/7） |
| Source labeling accuracy | 0.800（8/10） |
| Safe abstention Rate | 1.000（5/5） |
| Average number of tools steps | 1.400 |
| P50 Delay | 9916.566 ms |
| P95 Delay | 20794.776 ms |

The human review of the independent records was made by 6 articles, partially by 2 articles, and failed by 2 articles. The manual result does not write back `passed` in JSON, nor is it merged with `30/30` in offline Fixture. See [Live-Model Evaluation](docs/en_real-model-evaluation.md) and [Final Evaluation](docs/en_final-evaluation.md) for details of failed borders.

### Week 8 Offline Regression with Fixed Fixtures

Offline returns using Fake Planner, Fake Vision and fixed tool results in the same group 30 Gold scene, validate production Runner, tool strategy, progress return, end state and reporting framework:

```powershell
python -m scripts.evaluate_multimodal_offline_regression --scenarios "data/eval/multimodal_scenarios.jsonl" --fixtures "data/eval/multimodal_offline_fixtures.jsonl" --json-output "docs/multimodal-offline-regression.json" --markdown-output "docs/multimodal-offline-regression.md"
```

The current results are strictly passed by `30/30`, the security rejection rate is `1.000`, and the total consumption of `30/30` is fixed. The result only proves that the certainty framework can be repeated and does not demonstrate that the true external model has the same accuracy.

## 17. Automated Tests

Runs all tests and produces final machine-readable reports. `pytest.ini` automatically fixes the test temporary directory to the project `.pytest_tmp`:

```powershell
# Runs all tests and generates the current Week 8 JUnit.
python -m pytest -q --junitxml="docs/pytest-results-week8-final.xml"
```

Pytest automatically writes to the JUnit at the time of generation. Before being publicly submitted, the following command running Windows PowerShell deleted all `timestamp` properties of `testsuite`; It does not modify the name, number, result or time of the test:

```powershell
$path=(Resolve-Path -LiteralPath "docs\pytest-results-week8-final.xml").Path; [xml]$document=Get-Content -LiteralPath $path -Raw -Encoding UTF8; @($document.SelectNodes('//testsuite')) | ForEach-Object { $_.RemoveAttribute('timestamp') }; $document.Save($path)
```

The current source code generated Week 8 with a final return of `2655 passed`:

```text
tests    : 2655
failures : 0
errors   : 0
skipped  : 0
```

The above figures are from `docs/pytest-results-week8-final.xml`, not derived from `.pytest_cache` or the number of test files. Return to existing functions covering Week 1 to Week 6, four types of open diagnostic status in Week 7, the Agent SSE contract, the event publisher, Runner Observer, flow-based diagnosis Service, SSE Router, consumption and end-to-end delivery boundary for control station events, and Week 8's negative intent, clean environmental delivery, 10 authentic sample and report replicating contracts.

Core test uses:

- Fake Embedding；
- Fake LLM；
- `httpx.MockTransport`；
- FastAPI Dependence on coverage;
- pytest `monkeypatch`；
- Temporary directory `tmp_path`.

Automation tests should not:

- Access to real LLM;
- Access to real Embedding;
- Use real API Key;
- Change the official Chroma Collection;
- Reliance on external networks.

Main coverage:

- PDF, Markdown, TXT resolution;
- c. Chunk boundaries, overlaps and metadata;
- Embedding batch and cache;
- Cosine similarity retrieval;
- ChromaDB for writing, searching and deleting;
- Repeat ingestion and upload size limits;
- Gold Question Data Compact;
- (a) Retrieving and quoting assessment calculations;
- Top-K rankings and sources;
- (a) The unitization of the error codes, models, values and units;
- (b) Keyword retrieval, RRF integration and head retention reordering;
- (a) Combining evidence gate release and refusal of answer;
- Reference to allowlist mapping;
- LLM invalid response and timeout;
- Knowledge base API;
- Structured diagnostic requests, reports, search scope and cross-field compacts;
- Diagnosis Prompt, JSON analysis, conflict evidence and stabilization downgrading;
- (b) Boundaries of qualified personnel for high-risk inspections;
- RAG with standalone LLM experiments and reports generated.
- Extend Gold's number, type, label, control, expected evidence location and no answer boundaries;
- Agent tool static definition, Pydantic input output contract and OpenAI Tool Schema;
- ToolRegistry Allowlist, duplicate registration and refusal of high risk or non-read-only tools;
- ToolExecutor unknown tools, illegal parameters, empty results, overtime, abnormal and illegal output;
- AgentRunner single-step and multi-step tracks, maximum number of steps, double calls, Planner timeout and continuous failure;
- (b) Injecting into data processing as untrustworthy, without changing the allowlist or permission to use the tool;
- (a) The request level `ConfirmedEvidenceStore`, the forgery of Chunk ID, and the retroactiveness of evidence for refusing and testing drafts;
- Agent API requests verification, reliance on injection, completion of response, security suspension response and redacted trajectory;
- `search_knowledge` pure search boundaries to ensure that no encorded call answers are available within the tool;
- Agent fixed scene contract, batch HTTP execution, tool sequence rating, task completion rating, quote rating, security rejection rating and report generation.
- Image Base64, MIME, real format, size, edge, pixel numbers, empty content and time-out verification;
- (a) VisionProvider's original multimodular message, structured output, empty response, illegal JSON and upstream abnormal conversion;
- (b) The interpretation of Tesseract OCR, confidence, text boundary and certainty rules;
- `analyze_robot_image` Request-level photo references, read-only privileges, tips into borders, aerial observation and high-risk content processing;
- (a) Multimodular Agent visual observations, knowledge evidence, user logs and simulation of telemetry source layers;
- 30 scene pre-screening, batch execution, definitive rating, auxiliary Judge, delayed digit number, cost coverage and redacted trajectory generation.
- Minimum-level tool strategy, defined safety classification and negative expression and high-risk boundary.
- `AgentProgress` Capacity completion, tool recording, evidence coverage, conflict, missing information and range of next steps tools.
- 30 Fixture loading, Gold matching, Fake implementation, consumption auditing, strict rating and safety back-threshold threshold.
- The structure of the redacted diagnostic session, the storage of atoms, the recent list, the id-pressed query and the absence of a session error.
- Four public diagnostic states and `human_review_required` cross-layer consistency.
- Agent SSE event sequence, separation joint, public field, only termination event and UTF-8 code.
- Streading Service, Runner SSE Observer, `stream_error` after flow and client-end break the border.
- The real-time event consumption, failure state, JSON/Markdown export and static resource mount of the Web console.
- Docker relies on assembly, non-root users, health check and eventual complete return.

## 18. Repository Structure

```text
analysys-AI-app/
├── app/
│   ├── agent/
│   │   ├── tools/
│   │   │   ├── analyze_robot_image.py
│   │   │   ├── current_time.py
│   │   │   ├── draft_test_case.py
│   │   │   ├── robot_telemetry.py
│   │   │   └── search_knowledge.py
│   │   ├── demo_telemetry.py
│   │   ├── evidence_store.py
│   │   ├── executor.py
│   │   ├── failure_policy.py
│   │   ├── openai_planner.py
│   │   ├── planner.py
│   │   ├── progress_reducer.py
│   │   ├── registry.py
│   │   ├── request_safety_classifier.py
│   │   ├── request_tool_policy.py
│   │   ├── runner.py
│   │   └── vision_input_store.py
│   ├── middleware/
│   │   └── request_id.py
│   ├── routers/
│   │   ├── agent.py
│   │   ├── diagnostics.py
│   │   ├── diagnostic_sessions.py
│   │   ├── health.py
│   │   ├── knowledge.py
│   │   └── triage.py
│   ├── schemas/
│   │   ├── agent.py
│   │   ├── agent_api.py
│   │   ├── agent_diagnosis.py
│   │   ├── agent_planning.py
│   │   ├── agent_runtime.py
│   │   ├── agent_progress.py
│   │   ├── agent_tool_policy.py
│   │   ├── agent_tools.py
│   │   ├── diagnostic_session.py
│   │   ├── diagnosis_llm.py
│   │   ├── diagnostics.py
│   │   ├── evaluation.py
│   │   ├── knowledge.py
│   │   ├── knowledge_query.py
│   │   ├── multimodal_agent_evaluation.py
│   │   ├── multimodal_offline_regression.py
│   │   ├── multimodal_offline_regression_report.py
│   │   ├── multimodal_tool_selection.py
│   │   ├── multimodal_tool_selection_evaluation.py
│   │   ├── ocr.py
│   │   ├── ocr_rule.py
│   │   ├── retrieval.py
│   │   ├── retrieval_strategy.py
│   │   ├── triage.py
│   │   └── vision.py
│   ├── services/
│   │   ├── agent_diagnosis_report_builder.py
│   │   ├── agent_diagnosis_service.py
│   │   ├── diagnosis_generation.py
│   │   ├── diagnosis_prompts.py
│   │   ├── diagnosis_report_builder.py
│   │   ├── diagnosis_service.py
│   │   ├── diagnostic_session_builder.py
│   │   ├── diagnostic_session_store.py
│   │   ├── document_catalog.py
│   │   ├── document_loader.py
│   │   ├── document_service.py
│   │   ├── embedding.py
│   │   ├── gated_hybrid_retrieval.py
│   │   ├── head_preserving_reranking.py
│   │   ├── hybrid_retrieval.py
│   │   ├── hybrid_retrieval_gate.py
│   │   ├── keyword_retrieval.py
│   │   ├── knowledge_answer.py
│   │   ├── lexical_normalization.py
│   │   ├── multimodal_tool_selection_executor.py
│   │   ├── multimodal_tool_selection_experiment.py
│   │   ├── multimodal_offline_fixture_loader.py
│   │   ├── multimodal_offline_regression_service.py
│   │   ├── multimodal_offline_scenario_executor.py
│   │   ├── ocr_provider.py
│   │   ├── ocr_rule_parser.py
│   │   ├── query_rewriting.py
│   │   ├── query_rewriting_retrieval.py
│   │   ├── retrieval_scope.py
│   │   ├── vector_store.py
│   │   ├── vision_client.py
│   │   ├── vision_input.py
│   │   ├── vision_prompts.py
│   │   └── vision_provider.py
│   ├── web/
│   │   ├── app.js
│   │   ├── index.html
│   │   └── styles.css
│   ├── config.py
│   ├── dependencies.py
│   ├── errors.py
│   └── exception_handlers.py
├── data/
│   ├── eval/
│   │   ├── agent_scenarios.jsonl
│   │   ├── gold_questions.jsonl
│   │   ├── multimodal_offline_fixtures.jsonl
│   │   ├── multimodal_scenarios.jsonl
│   │   └── multimodal_tool_selection_cases.jsonl
│   ├── multimodal/
│   │   └── tool-selection/
│   ├── processed/
│   └── source/
│       └── .gitkeep
├── samples/
│   └── corpus/
│       └── robot-fault-demo.txt
├── docs/
│   ├── en_agent-architecture.md
│   ├── en_agent-evaluation.md
│   ├── en_agent-failure-cases.md
│   ├── agent-trace-agent-001-failure.json
│   ├── agent-trace-agent-004.json
│   ├── agent-trace-agent-005.json
│   ├── en_agent-trace-samples.md
│   ├── en_corpus-sources.md
│   ├── en_architecture.md
│   ├── en_api-contract.md
│   ├── en_project-proposal.md
│   ├── diagnostic-session-sample.json
│   ├── en_diagnosis-schema.md
│   ├── en_docker-startup-record.md
│   ├── e2e-run-sample.json
│   ├── en_e2e-scenario-results.md
│   ├── en_e2e-test-plan.md
│   ├── en_multimodal-evaluation.md
│   ├── en_multimodal-scenarios.md
│   ├── en_multimodal-tool-selection.md
│   ├── en_multimodal-offline-regression.md
│   ├── real-model-evaluation.json
│   ├── en_real-model-evaluation.md
│   ├── en_security-retest.md
│   ├── en_final-evaluation.md
│   ├── multimodal-traces/
│   ├── robotops-copilot-docker-demo.png
│   ├── en_retrieval-strategy-comparison.md
│   ├── en_test-report_week3.md
│   ├── en_engineering-retrospective_week3.md
│   ├── en_engineering-retrospective_week4.md
│   ├── en_engineering-summary_week4.md
│   ├── en_engineering-summary_week5.md
│   ├── en_engineering-summary_week6.md
│   ├── en_reproducibility.md
│   ├── pytest-results-week3-final.xml
│   ├── pytest-results-week4-final.xml
│   ├── pytest-results-week6-final.xml
│   └── pytest-results-week8-final.xml
├── scripts/
│   ├── compare_diagnosis_prompts.py
│   ├── compare_retrieval_strategies.py
│   ├── evaluate_agent.py
│   ├── evaluate_citations.py
│   ├── evaluate_hybrid_gate.py
│   ├── evaluate_multimodal.py
│   ├── evaluate_multimodal_offline_regression.py
│   ├── evaluate_multimodal_tool_selection.py
│   ├── evaluate_retrieval_strategies.py
│   ├── generate_multimodal_agent_scenarios.py
│   ├── generate_multimodal_tool_selection_samples.py
│   ├── build_multimodal_offline_fixtures.py
│   ├── ingest_documents.py
│   └── smoke_hybrid_retrieval.py
├── tests/
├── .env.example
├── .dockerignore
├── .gitignore
├── compose.yaml
├── Dockerfile
├── main.py
├── en_README.md
└── requirements.txt
```

## 19. Known Limitations

- `pdfplumber` extracts text from PDFs with a text layer. Scanned PDFs, complex tables, multi-column layouts, and text embedded in images may be incomplete. Heuristic restoration of English word spacing can also fail on some layouts.
- ChromaDB persists data locally on a single machine. The current API lacks authentication, authorization, rate limiting, and tenant isolation.
- Document uploads ingest synchronously. Large files take longer because the request waits for embedding.
- Similarity thresholds apply to the current corpus, embedding model, and gold evaluation set.
- The final retrieval strategy still misses the second piece of evidence for q027, a cross-document question about equipment cleaning and workstation acceptance.
- For q007, retrieval finds the correct candidate and the composite gate still abstains. For q025, the generic prompt tends to abstain on strict equality boundaries.
- The knowledge base works with a static corpus. It cannot answer questions about live location, battery levels, or orders.
- The agent reads redacted simulated telemetry in memory. It recreates the simulated store when the process restarts. All five agent tools are read-only. `draft_test_case` generates a draft for human approval using a deterministic template.
- The planner requires an external generative LLM. A compatible provider may suggest multiple tools in a single response. The single-step ReAct loop executes the first tool and replans using the observation. This can produce an `agent_planner_parallel_tool_calls_serialized` warning.
- Week 8 offline regression recorded `1.000` for both strict pass rate and safe abstention on fixed planner decisions, fake Vision responses, and fixed tool outputs. The separate live-model sample recorded a `0.400` strict pass rate. Retrieval of a specific DataMan page, mapping of invalid Vision responses, and planner final JSON remain failure points.
- Vision observations are probabilistic. Low image quality, occlusion, reflections, and small text can cause omissions or misreads. Local OCR requires Tesseract and language data. OCR confidence reflects recognition confidence.
- Requests embed Base64 images in JSON, increasing payload size. A production system would generally use controlled object storage, short-lived signed references, and a separate upload flow.
- The live models did not provide token usage, so costs are unavailable.
- `execution.state="completed"` means the agent finished its orchestration and diagnosis draft. It does not record execution of any on-site inspection, repair, reset, test, or device control.
- SSE and the console show the agent lifecycle live. The current implementation has no resume, event replay, cross-process recovery, or background queue. A disconnected client cannot resume the same run from an event ID.
- Model outputs require schema validation, application rules, and human review. Qualified personnel must verify safety-critical operations.
- The console and Docker Compose setup support local teaching demos. Production capabilities such as authentication, multi-user isolation, TLS, secret management, centralized logs, backups, and multi-instance deployment are outside this version.
- SSE is available for Week 1 triage and Week 7 agent diagnosis. Knowledge-base queries return ordinary JSON.

## 20. Further Reading

- [Week 8 candidate acceptance](docs/en_candidate-acceptance.md)
- [Week 8, normal diagnostic machine, read and receive evidence.](docs/candidate-acceptance-normal.json)
- [Week 8, manual machine-readable evidence.](docs/candidate-acceptance-human-review.json)
- [Rewind project Week 8](docs/en_project-retrospective.md)
- [Week 8 Presentation script](docs/en_demo-script.md)
- [Week 8 Final Evaluation](docs/en_final-evaluation.md)
- [Week 8 Live-Model Evaluation](docs/en_real-model-evaluation.md)
- [Week 8 Safety Retest](docs/en_security-retest.md)
- [Week 8, final JUnit returns report](docs/pytest-results-week8-final.xml) （ `2655 passed` ）
- [Week 8 fixed Fixture offline return](docs/en_multimodal-offline-regression.md)
- [Week 7 Summary of the Project](docs/en_engineering-summary_week7.md)
- [Week 7 MVP Program](docs/en_project-proposal.md)
- [Week 7 API compact](docs/en_api-contract.md)
- [Week 7 End-to-end test plan](docs/en_e2e-test-plan.md)
- [Week 7 End to End scene result](docs/en_e2e-scenario-results.md)
- [Week 7 Docker starter log](docs/en_docker-startup-record.md)
- [Week 7 complete redacted sample](docs/e2e-run-sample.json)
- [Week 6 Summary of the Project](docs/en_engineering-summary_week6.md)
- [Week 6, final JUnit returns report](docs/pytest-results-week6-final.xml) (History baseline: `2514 passed`; Week 7 current complete return: `2623 passed`)
- [RobotOps Copilot Docker demoshot](docs/robotops-copilot-docker-demo.png)
- [RobotOps Copilot SSE human review demonstration screenshot](docs/robotops-copilot-sse-human-review.png)
- [RobotOps Copilot full presentation video](https://github.com/coffeeBehindTea/analysys-AI-app/releases/latest/download/robotops-copilot-demo.mp4)
- [Week 5 Summary of the Project](docs/en_engineering-summary_week5.md)
- [Multimodular Agent Reliability Assessment](docs/en_multimodal-evaluation.md)
- [Multimodular Tool Select Contrast Experiment](docs/en_multimodal-tool-selection.md)
- [Multistate scene and field description](docs/en_multimodal-scenarios.md)
- [Week 3 Recoverable baseline](docs/en_reproducibility.md)
- [Robot Diagnostic Agent Architecture](docs/en_agent-architecture.md)
- [Robot Diagnostic Agent failure and recovery strategy](docs/en_agent-failure-cases.md)
- [Robot Diagnostic Agent Batch Evaluation](docs/en_agent-evaluation.md)
- [Robot Diagnostic Agent fully run track samples](docs/en_agent-trace-samples.md)
- [Week 4, final JUnit returns report](docs/pytest-results-week4-final.xml)
- [Week 4 Summary of the Project](docs/en_engineering-summary_week4.md)
- [Week 4 Study Rediscovery](docs/en_engineering-retrospective_week4.md)
- [Description of language sources](docs/en_corpus-sources.md)
- [System architecture](docs/en_architecture.md)
- [Candidates for the search strategy comparison](docs/en_retrieval-strategy-comparison.md)
- [Diagnostic data compacts](docs/en_diagnosis-schema.md)
- [Week 3 Automated testing and acceptance reports](docs/en_test-report_week3.md)
- [Week 3 Project Rediscretion](docs/en_engineering-retrospective_week3.md)
