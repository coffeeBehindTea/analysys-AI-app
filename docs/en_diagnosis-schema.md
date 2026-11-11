# Robot Diagnostic Evidence API Data Contract

## 1. Purpose

This document describes the request, response, evidence reference, downgrading rules and risk boundary for `POST /api/v1/diagnostics`.

The interface is not a robotic controller or an automated maintenance system. It generates retrospective structured diagnostic recommendations based only on open or redacted engineering files; Any security critical operations will still need to be identified by qualified personnel on the basis of original plant information and on-site processes.

## 2. Position in Call Chain

```text
HTTP Client
  → Diagnostics Router: Parsing and Validating Diagnosis
  → DiagnosisService: Construct search search and organize the entire diagnostic process
  → Query Normalization and Deterministic-v2 Rewrite
  → Vector Retrieval + Keyword Retrieval + RRF Integration
  → head-preserving- v1 reordered
  → hybrid-evidence-gate- v1 gate control
      → Rejected: Python directly construct abstaned report without calling LLM
      → Release: OpenAIDIGNOsisDraftProvider from DiagnosisLLMDraft
  → DiagnosisReportBuilder: Validation of provisional evidence numbers and mapping of real Chunk ID
  → DiagnosisReport: perform final cross-field and reference allowlist validation
  → FastAPI Sequencing JSON Response
```

The functional boundaries of the modules are as follows:

| Modules | Input | Main Operations | Output |
|---|---|---|---|
| `app/routers/diagnostics.py` | HTTP JSON、`Request.state.request_id` | Declares route, triggers FastAPI verification, calls Service | `DiagnosisReport` |
| `app/schemas/diagnostics.py` | Python field value | Pydantic field verification and cross-field verification | Credible request or final reporting target |
| `app/services/diagnosis_service.py` | `DiagnosisRequest`、Request ID | Organize search, evidence gate, LLM and downgrade paths | `DiagnosisReport` |
| `app/services/diagnosis_prompts.py` | The request and the real candidate behind the door. | Create `E1`, `E2` temporary numbers and construct Prompt | LLM message |
| `app/services/diagnosis_generation.py` | LLM message | Call OpenAI compatibility interface, clear code fence, decipher JSON | `DiagnosisLLMDraft` |
| `app/services/diagnosis_report_builder.py` | Drafts, evidence gate results, real candidates. | Execute allowlist checks and map temporary numbers to real Chunk | `DiagnosisReport` |

## 3. Request for contract

### 3.1 `DiagnosisRequest`

| Fields | Type | shall fill | Constraints | Semantic |
|---|---|---|---|---|
| `robot_id` | `str` | Yes. | 1 ~100 characters | Unusual robot numbering;retrieved text not added |
| `symptom` | `str` | Yes. | 1 ~2000 characters | Faults observed by users |
| `log_excerpt` | `str` | Yes. | 1 ~10000 characters | Redacted log summary, with error codes, model numbers and thresholds |
| `retrieval_scope` | `DiagnosisRetrievalScope \| null` | Yes | See section below | Limit the document to which this search allows access |

All strings automatically remove the initial and end blanks, and undeclared fields are rejected. `robot_id` for reporting tracking purposes, but not for retrieval, as specific case numbers usually do not contain synonyms for project failure.

### 3.2 `DiagnosisRetrievalScope`

| Fields | Type | Constraints | Semantic |
|---|---|---|---|
| `document_ids` | `list[str]` | Up to 20; each must be 64 lowercase SHA-256; not repeatable | Limit scope by document contents |
| `source_files` | `list[str]` | Up to 20 characters; 1 ~ 255 characters each; not renewable | Limit the scope by file name without server path |

When visible `retrieval_scope` is provided, the two lists cannot be empty at the same time. When both fields exist, the internal range filter limits the search to a combination of conditions instead of filtering Top-K after a full library search.

Example requested:

```json
{
  "robot_id": "diagnostic-robot",
  "symptom": "机器人网络恢复后仍然没有继续执行任务",
  "log_excerpt": "ERR-NET-4001 heartbeat timeout exceeded 1500 ms; network recovered but task did not resume",
  "retrieval_scope": {
    "source_files": [
      "仓储机器人故障说明.txt",
      "京东无人仓场景-仓储机器人测试规程与判定标准.md"
    ]
  }
}
```

## 4. Responding to compacts

### 4.1 `DiagnosisReport`

| Fields | Type | Constraints and sources |
|---|---|---|
| `request_id` | `str` | From Request ID Middleware for associated response header and log |
| `prompt_version` | `str` | Practical use of diagnostic Prompt version; `not-invoked` when evidence gate is rejected earlier |
| `gate_version` | `str` | Practically used version of the door to evidence |
| `status` | `completed \| partial \| abstained` | Indicate complete diagnosis, partial diagnosis or refusal of response |
| `symptoms` | `list[DiagnosisSymptom]` | 1 ~ 20 items from user descriptions or logs only |
| `evidence` | `list[DiagnosisEvidence]` | Up to 10 entries, which can only be constructed by Python from a genuine search candidate |
| `possible_causes` | `list[DiagnosisCause]` | Up to 10, bind at least one Chunk ID in the current report |
| `next_checks` | `list[DiagnosisCheck]` | Up to 10 pieces of evidence bound each and declaration of the risk boundary |
| `risk_level` | `low \| medium \| high \| unknown` | Highest level of risk for the entire report |
| `missing_information` | `list[str]` | Up to 20 cases, indicating information still lacking to form a more complete diagnosis |
| `abstained` | `bool` | Quick-judgement system for client refusals |

### 4.2 Symptoms contract

`DiagnosisSymptom` includes:

- `description`: a summary of the original phenomenon or log in the request;
- `source`: only `user_report` or `log_excerpt`.

Symptoms are user-inputed, not disguised as knowledge-base facts, nor do the Chunk quotes.

### 4.3 Evidentiary contract

`DiagnosisEvidence` includes:

| Fields | Meaning |
|---|---|
| `chunk_id` | True identification of document ID and Chunk serial numbers |
| `document_id` | SHA-256 on content of original document |
| `source_file` | Source filename without absolute server path |
| `page_or_section` | Original page number or chapter location |
| `rank` | This final RRF results ranking, starting with 1 |
| `rrf_score` | RRF integration score, used only for ranking audit, not probability |
| `vector_similarity` | Cosine symmetry; single candidates for keywords can be `null` |
| `excerpt` | Python copied from the real Chunk body, not generated by LLM |

### 4.4 Reasons and Inspection Items

`DiagnosisCause` contains a careful presentation of `description` and at least one `evidence_chunk_ids`. For the same reason, the same Chunk cannot be cited again.

`DiagnosisCheck`. Also includes:

- `risk_level`；
- `requires_qualified_person`。

When `risk_level="high"`, `requires_qualified_person` must be `true`, otherwise Pydantic rejects the object.

## 5. Cross-field rules

Ultimately, `DiagnosisReport` must satisfy the following relationships:

1. `status="abstained"` and `abstained=true` must be established at the same time; other states must be responsive to `abstained=false`.
2. The refusal report cannot contain `possible_causes` or `next_checks`, but must be filled in `missing_information`.
3. The non-rejecting report must contain the true `evidence` and include at least one reason or check item.
4. `partial` must indicate the missing information.
5. `completed` cannot at the same time claim that missing information remains.
6. Chunk ID and rankings in `evidence` cannot be repeated.
7. The reason and the reference to each Chunk ID must be in this report `evidence`.
8. High-risk inspections must be marked as requiring the identification or performance of qualified personnel.

These rules are implemented by Pydantic's `field_validator` and `model_validator(mode="after")`. Field checker checks duplicate entries in individual lists; The model verifier checks cross-field relationships after the object field is fully parsed.

## 6. Internal LLM draft and final report boundary of confidence

LLM is allowed to return to the interior only `DiagnosisLLMDraft`:

```text
status
possible_causes[].description
possible_causes[].evidence_ids
next_checks[].description
next_checks[].evidence_ids
next_checks[].risk_level
next_checks[].requires_qualified_person
risk_level
missing_information
abstained
```

LLM can only select temporary numbers such as `E1`, `E2` and cannot fill in the following credible fields:

- Real Chuck ID;
- Document ID;
- (a) File name;
- Page number or chapter;
- RRF fractions;
- Vector similarities;
- Request ID；
- Prompt and evidence gate version.

`DiagnosisReportBuilder` Create the following map:

```text
E1 → rank = 1 The real Hybrid Retrieved Chunk
E2 → rank = 2 The real Hybrid Retrieved Chunk
E3 → rank = 3 The real Hybrid Retrieved Chunk
```

If LLM returns the `E999` in a valid format but does not exist this time, Builder will throw `InvalidLLMResponseError` and then the service returns a stable downgrade report instead of sending the false number to the client.

## 7. Rule of demotion

| Trigger condition | Whether to call LLM | HTTP successfully responded | Evidence processing | Public Results |
|---|---:|---|---|---|
| No candidate. | Yes | `200` | Do not return weak candidates | `abstained=true`, indicating that no candidate has been retrieved |
| Combining evidence gate inadequate | Yes | `200` | Do not return unapproved candidate | `abstained=true`, request additional error code, log or live status |
| LLM takes the initiative to judge conflicting or insufficient evidence | Yes. | `200` | Retain candidatures for audit | Do not give reasons and operations, fill in missing information |
| LLM JSON is not valid, field is missing or unknown | Yes. | `200` | Keep the evidence gate free of evidence | Fixed structural verification failure indicates that no internal response is revealed |
| LLM timeout | Yes. | `200` | Keep the evidence gate free of evidence | Fixed timeout description |
| LLM upstream error | Yes. | `200` | Keep the evidence gate free of evidence | No description available for fixed services |
| Embedding configuration, overtime or upstream error | Yes | `503`, `504` or `502` | No Retrievable Results | Revert error response from the Unified Aberrant Processor |

The known failures of the generation phase were converted to `200 + abstained` because the client still had access to structurally stable and auditable diagnostic reports. Errors, such as relying on the return of a normal dictionary, are not disguised as business refusals.

## 8. Structured diagnostic sample

The following is an extract of the successful results that have been validated through genuine search and real LLM links. (a) To control the length of the document, only the summary is reserved for `excerpt`; The actual API returns the real Chunk body.

```json
{
  "request_id": "04fd7664-9246-4b82-be96-1b84573bcaa9",
  "prompt_version": "diagnosis-evidence-v1",
  "gate_version": "hybrid-evidence-gate-v1",
  "status": "partial",
  "symptoms": [
    {
      "description": "机器人网络恢复后仍然没有继续执行任务",
      "source": "user_report"
    },
    {
      "description": "ERR-NET-4001 heartbeat timeout exceeded 1500 ms; network recovered but task did not resume",
      "source": "log_excerpt"
    }
  ],
  "evidence": [
    {
      "chunk_id": "5b11fafd10c2414a65e1fd6eed8d7a295ee707ebaa0da89e6470841887b7ab80:000022",
      "document_id": "5b11fafd10c2414a65e1fd6eed8d7a295ee707ebaa0da89e6470841887b7ab80",
      "source_file": "京东无人仓场景-仓储机器人测试规程与判定标准.md",
      "page_or_section": "section: 10.1 TEST-NET-001 调度心跳中断",
      "rank": 1,
      "rrf_score": 0.032522,
      "vector_similarity": 0.610440,
      "excerpt": "通信恢复后不自动继续旧任务；RCS/RMS完成状态核对并下发恢复命令后才能继续运行。"
    },
    {
      "chunk_id": "dd18de8c04d80f7805a8f70ad6c24857d0df55df8009dec2e3371c3ee50fc652:000003",
      "document_id": "dd18de8c04d80f7805a8f70ad6c24857d0df55df8009dec2e3371c3ee50fc652",
      "source_file": "仓储机器人故障说明.txt",
      "page_or_section": "section: ERR-NET-4001",
      "rank": 2,
      "rrf_score": 0.032522,
      "vector_similarity": 0.634444,
      "excerpt": "可能原因包括AP漫游失败、信号盲区或干扰、车载网关断电。"
    }
  ],
  "possible_causes": [
    {
      "description": "任务未继续可能是RCS/RMS尚未完成状态核对或尚未下发恢复命令。",
      "evidence_chunk_ids": [
        "5b11fafd10c2414a65e1fd6eed8d7a295ee707ebaa0da89e6470841887b7ab80:000022"
      ]
    }
  ],
  "next_checks": [
    {
      "description": "确认RCS/RMS是否完成状态核对并下发恢复命令。",
      "evidence_chunk_ids": [
        "5b11fafd10c2414a65e1fd6eed8d7a295ee707ebaa0da89e6470841887b7ab80:000022"
      ],
      "risk_level": "low",
      "requires_qualified_person": false
    },
    {
      "description": "车载网关供电和重启操作需由有资质人员确认。",
      "evidence_chunk_ids": [
        "dd18de8c04d80f7805a8f70ad6c24857d0df55df8009dec2e3371c3ee50fc652:000003"
      ],
      "risk_level": "high",
      "requires_qualified_person": true
    }
  ],
  "risk_level": "high",
  "missing_information": [
    "RCS/RMS是否已完成状态核对并下发恢复命令",
    "现场Wi-Fi信噪比实测值"
  ],
  "abstained": false
}
```

## 9. Related achievement and testing

- Data compact: `app/schemas/diagnostics.py`
- LLM internal draft: `app/schemas/diagnosis_llm.py`
- HTTP route: `app/routers/diagnostics.py`
- Organization of business: `app/services/diagnosis_service.py`
- Prompt：`app/services/diagnosis_prompts.py`
- Generate and resolve with JSON: `app/services/diagnosis_generation.py`
- Reference map and downgrade: `app/services/diagnosis_report_builder.py`
- Schema Test: `tests/test_diagnostics_schemas.py`
- Generating test: `tests/test_diagnosis_generation.py`
- Service Test: `tests/test_diagnosis_service.py`
- API Test: `tests/test_diagnostics_api.py`
