# Week 3 Testing and Acceptance Report

## 1. Test Objectives

Two core results of Week 3 are validated in this report:

1. Candidates to retrieve from a pure vector baseline to " Keyword + Vector + RRF + Query Rewrite + Head " without reducing the lack of answer security.
2. `POST /api/v1/diagnostics` is capable of exporting structured diagnostic reports that are bound by Pydantic, citing true Chunk, with stable downgrade paths.

The report also confirms that the failure breakdown of Week 1 and Week 2 have not returned to their ability to document management, ingestion, vector storage and knowledge base questions and answers.

## 2. Test Environment

| Item | Value |
|---|---|
| Operating systems | Windows |
| Python | CPython 3.11.15 |
| pytest | 9.1.1 |
| pytest-asyncio | 1.4.0 |
| FastAPI | 0.141.1 |
| Pydantic | 2.13.4 |
| OpenAI SDK | 2.53.0 |
| ChromaDB | 1.5.9 |
| Gold's problem. | 28 |
| Answerable questions | 23 |
| No answer to question | 5 |
| Current anticipated evidence | 31 |

Reliance on all versions fixed to `requirements.txt`. Tests the temporary root directory from `pytest.ini` to `.pytest_tmp` within the project, avoiding the Windows system temporary directory privileges and locking problems.

## 3. Final offline regression.

Execute orders:

```powershell
python -m pytest --junitxml=docs/pytest-results-week3-final.xml
```

Real results:

```text
collected : 798
passed    : 798
failures  : 0
errors    : 0
skipped   : 0
duration  : 18.74s
```

Machine-readable evidence:

```text
docs/pytest-results-week3-final.xml
```

The `time` in the JUnit indicates that the test is time-consuming and not a report-generation time. The report does not save the date or time of generation.

## 4. Offline isolation

The complete pytest returns without access to real LLM or Embedding services, nor does it use official Croma Collection.

| Testing a double or mechanism | Alternative Object | Purpose |
|---|---|---|
| Fake Embedding | Real Embedding API | Return predictable vectors, validate batch processing, cache and organization |
| Fake LLM | Real Generation Model | Returning positive results such as success, refusal, illegal JSON, missing field and overtime |
| `httpx.MockTransport` | Real HTTP network. | Validation of SDK boundaries and abnormal transformations |
| FastAPI `dependency_overrides` | Formal reliance on assembly | Injecting Fake Service for API and validating HTTP contracts |
| `tmp_path` with temporary Chroma catalogue | Formal Vector | Separating writing, retrieval, deletion and durability tests |
| `monkeypatch` | Environmental variables or modular functions | Control configuration and anomaly branch |

Fake is not intended to demonstrate upstream model quality, but only to validate the responsibility of the current module. For example, the denial of evidence gate requires confirmation that “LLM is not called” and that Fake, with a call count, is more accurate and stable than a real network request.

## 5. Mission 7 must be measured.

### 5.1 Exact recall of error code

Representative Test:

- `tests/test_hybrid_retrieval_gate.py::test_exact_identifier_in_top_k_can_use_lower_threshold`
- `tests/test_gated_hybrid_retrieval.py::test_exact_identifier_candidate_is_accepted`
- `tests/test_lexical_normalization.py`
- `tests/test_query_rewriting.py`

Tested process:

```text
ERR_NET_4001 or err saf 1002
→ Resolute certainty
→ ERR-NET- 4001 or ERR-SAF-1002
→ Keyword Candidates and Vector Candidates
→ Door IDexact_identifier_support
```

The result is expected to be an error code variant that matches the normative form and precise identifier evidence can use a specially calibrated evidence gate path.

### 5.2 Integration ranking and head retention

Representative Test:

- `tests/test_hybrid_retrieval.py::test_consensus_candidate_accumulates_both_rrf_contributions`
- `tests/test_hybrid_retrieval.py::test_rrf_ignores_vector_similarity_magnitude`
- `tests/test_head_preserving_reranking.py::test_vector_and_keyword_heads_preserve_complementary_evidence`
- `tests/test_head_preserving_reranking.py::test_existing_heads_keep_original_rrf_order`

Tested process:

```text
Vector ranking + Keyword ranking
→ RRF contributions by ranking
→ Same Chunk combined contribution
→ Sort by RRF
→ Protecting the head, vector head and keyword head of RRF
→ Go heavy and return to final Top-K
```

The expected result is that RRF relies only on the number of paths and does not add the cosine similarity or the error in the original score of the keyword to the same schematic; only once each Chunk has been reordered.

### 5.3 Gate release and denial

Representative Test:

- `tests/test_gated_hybrid_retrieval.py::test_exact_identifier_candidate_is_accepted`
- `tests/test_gated_hybrid_retrieval.py::test_weak_candidate_is_rejected`
- `tests/test_diagnosis_service.py::test_gate_rejection_skips_llm_and_returns_stable_report`
- `tests/test_diagnosis_service.py::test_accepted_evidence_calls_llm_and_builds_traceable_report`

Expected processes:

- (a) Gate control release: call the LLM draftProvider and assemble the report;
- (a) Evidence gate refusal: LLM is not called and returns directly to `abstained=true`;
- Vacancies: use `no_candidates`;
- Weak combination support: use `insufficient_combined_support`.

### 5.4 Quoted Allowlist

Representative Test:

- `tests/test_diagnosis_prompts.py::test_system_prompt_requires_evidence_whitelist`
- `tests/test_diagnosis_report_builder.py::test_unknown_evidence_id_is_rejected`
- `tests/test_diagnostics_schemas.py::test_report_rejects_unknown_evidence_reference`
- `tests/test_hybrid_knowledge_query_service.py::test_unknown_evidence_id_is_rejected`

The process is expected to be LLM only able to return temporary numbers such as `E1`, `E2`; Builder remixes the real Chunk. Even if the format of `E999` is valid, it must be rejected as long as it is not in this candidature.

### 5.5 Illegal Structures and Missing Fields

Representative Test:

- `tests/test_diagnosis_generation.py::test_parser_rejects_invalid_json`
- `tests/test_diagnosis_generation.py::test_parser_rejects_missing_required_field`
- `tests/test_diagnosis_generation.py::test_parser_rejects_unsafe_high_risk_check`
- `tests/test_diagnosis_service.py` stabilization generation downgrade test

The expected result is that illegal JSON, missing fields and non-fulfilment of cross-field relationships do not pass through Schema; known creation errors are converted into stable refusal reports without exposure to upstream internal information.

### 5.6 Evidence of conflict

Representative Test:

- `tests/test_diagnosis_prompts.py::test_system_prompt_requires_abstention_on_conflict`
- `tests/test_diagnosis_llm_schemas.py` test for refusal of contract

The expected result is that the evidence conflict, which cannot be safely resolved, must be constructed to accommodate `abstained=true`, the cause and the inspection items are empty, and the elements still to be verified are described in `missing_information`.

### 5.7 No answer

Representative Test:

- `tests/test_diagnosis_report_builder.py::test_no_candidate_gate_rejection_builds_stable_report`
- `tests/test_diagnostics_api.py::test_diagnostics_returns_abstained_report_with_200`
- `tests/test_hybrid_knowledge_query_service.py::test_rejected_gate_abstains_without_calling_llm`

The expected result is no speculation when there are no answers or insufficient evidence; the knowledge base returns to blank references, structured diagnosis of the reasons for return and reports of examination items that are empty.

### 5.8 High-risk recommendations

Representative Test:

- `tests/test_diagnosis_llm_schemas.py::test_high_risk_draft_check_requires_qualification`
- `tests/test_diagnostics_schemas.py::test_high_risk_check_requires_qualified_person`
- `tests/test_diagnosis_report_builder.py::test_high_risk_check_preserves_qualification_boundary`

The expected result is that `risk_level="high"` must be met simultaneously in `requires_qualified_person=true`. The model does not describe high pressure, battery dismantling, safe circuit modification or the operation of power supply on a vehicle-mounted device as directly enforceable instructions for ordinary users.

## 6. Real candidate search assessment

The evaluation calls real Embedding and reads local Croma, but does not call the generated LLM.

In order to eliminate the denominator differences resulting from the `q021` Gold amendment, four stored candidate reports have been re-rated using the current 31 anticipated evidence.

| Policy | Recall@1 | Recall@3 | Top- 3 Full evidence recall |
|---|---:|---:|---:|
| vector-only baseline | 0.548（17/31） | 0.742（23/31） | 0.652（15/23） |
| Mixed RRF | 0.516（16/31） | 0.742（23/31） | 0.652（15/23） |
| Mixed RRF + Query Rewrite | 0.613（19/31） | 0.871（27/31） | 0.826（19/23） |
| Mixed RRF+Query Rewrite + Reset | **0.613（19/31）** | **0.968（30/31）** | **0.957（22/23）** |

The complete evidence of the final candidate programme's relative purity baseline is a recall of the increase in `0.305`, with no return.

## 7. Real evidence gate evaluation

| Indicators | Result |
|---|---:|
| Questionable release rate | 0.957（22/23） |
| Full evidence availability | 0.913（21/23） |
| Error release rate without answer | 0.000（0/5） |
| No answer correct response rate | 1.000（5/5） |

The security check at the door was passed and the final candidate programme was therefore granted access to the online API.

## 8. Real Knowledge base Answers and Quote Evaluations

Implementation portal:

```powershell
python -m scripts.evaluate_citations --questions data/eval/gold_questions.jsonl --api-url "http://127.0.0.1:8000/api/v1/knowledge/query" --top-k 3 --timeout-seconds 120 --json-output docs/citation-evaluation-week3-balanced-rollback.json --markdown-output docs/citation-evaluation-week3-balanced-rollback.md
```

Results:

| Indicators | Result | Receiving and inspection target | Conclusions |
|---|---:|---:|---|
| Response rate to questions that can be answered | 0.870（20/23） | ≥ 0.70 | Pass |
| Full evidence response rate | 0.739（17/23） | ≥ 0.65 | Pass |
| Citation correctness | 0.962（25/26） | ≥ 0.95 | Pass |
| Citation coverage | 0.806（25/31） | Observation indicators | Recorded |
| No answer correct response rate | 1.000（5/5） | ≥ 0.95 | Pass |

All Week 3 online acceptance indicators passed.

## 9. Diagnostic Prompt Control Experiment

The experiment fixes the same request, the same search evidence, the same model and `temperature=0.0`, only changes Prompt.

| Experimental team | Prompt Version | Structural success rate | Quoting allowlist pass rate |
|---|---|---:|---:|
| BasicPrompt | `diagnosis-basic-v1` | 0.000（0/1） | 0.000（0/1） |
| Prompts for the Evidence Priority | `diagnosis-evidence-v1` | 1.000（1/1） | 1.000（1/1） |

The basic Prompt returned the draft contract that violated the `completed` status; Prompt's priority was to return the legal `partial` report and correctly cited two authentic evidence. The experiment was used only in one case to validate the call chain and failure patterns, without claiming statistical prominence.

## 10. Structured diagnostic API measurements

True request verified:

- HTTP status is `200`;
- `prompt_version=diagnosis-evidence-v1`；
- `gate_version=hybrid-evidence-gate-v1`；
- `status=partial`；
- Both of the evidence was taken from this search;
- The reason and the reference to Chunk ID in the check item are in the returned `evidence`;
- High-risk gateway operator tag `requires_qualified_person=true`;
- (a) Missing information is clearly identified;
- The candidate evidence was not translated as a counterfeit source.

See `docs/en_diagnosis-schema.md` for full field synonyms and abbreviations.

## 11. Receipt and inspection list

| Receiving and Inspection | Evidence | Conclusions |
|---|---|---|
| The full evidence of the candidate's return is no less than the pure vector. | 0.957 to 0.652 | Pass |
| Answerable response rate 0.70 | 0.870 | Pass |
| Full evidence response rate 0.65. | 0.739 | Pass |
| Quote correct rate 0.95 | 0.962 | Pass |
| No answer is correct. 0.95. | 1.000 | Pass |
| Every non-empty diagnosis is traceable | Schema, Allowlist and Builder Tests | Pass |
| High-risk recommendations with qualified borders | Pydantic cross-field rules and tests | Pass |
| Not less than 25 automated tests | 798 items | Pass |
| Three or more candidate strategies | Actual comparison: 4 | Pass |
| Structured diagnostic sample | `docs/en_diagnosis-schema.md` | Pass |
| Finally, JUnit | `docs/pytest-results-week3-final.xml` | Pass |

## 12. Legacy issues

- The second evidence for the cross-document `q027` “Equipment Cleaning + Workstation Acceptance” still does not enter Top-3.
- The correct candidate for the search results of `q007` was refused.
- `q025` The strictly negative boundary question is kept under General Prompt.
- The answers for `q013`, `q026` are essentially complete, but no documents specified by Gold have been selected.
- Over `q024` was selected as a page of a manual that is relevant but not included in the Gold Allowlist.
- (b) `temperature=0` is not equal to full service-level certainty.
- The current system is not a robotic controller and no diagnostic recommendations can be automatically implemented.
