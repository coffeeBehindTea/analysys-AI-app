# Robot Knowledge Base API Test Report

## 1. Test Conclusion

The final automated regression test for Robot Knowledge Base API is all passed.

```text
Test nodes: 244
Adopted: 244
Failed: 0
Error: 0
Skip: 0
```

Tests covered Week 1 fail-diagnosis and SSE baselines, as well as Week 2 document resolution, cutting, Embedding, ChromaDB, RAG queries, evidence references, evaluation of scripts and RAG against standalone LLM experiments.

Core findings:

- Application modules can be imported normally;
- FastAPI route assembly successful;
- Document ingestion chain passage;
- RAG queries and low-symmetry denials;
- Quoting allowlist and code assembly through;
- (a) The calculation of the assessment indicators is adopted;
- Automation testing does not depend on real LLM, Embedding or external networks;
- The real API and model links have been validated by a separate manual integration.

## 2. Test time and evidence

Final validation time:

```text
2026-08-17T11:26:09+08:00
```

Machine-readable evidence:

```text
docs/pytest-results-week2.xml
tests：244
failures：0
errors：0
skipped：0
Pytest log time: 3.766 seconds
```

Note:

- (a) API compacts, business services, data contracts, ingestion, retrieval, citation, evaluation and unusual paths that were tested for `tests/` coverage;
- The execution process is a pytest collection test, followed by offline testing using Fake, Mock, temporary files and temporary Chroma database, which ends with the writing of each test case and time-consuming JUnit XML;
- The expected results are 244 passes, 0 failers, 0 errors, 0 skips and the testing process does not access real LLM, Embedding services or official vector banks;
- (a) `.pytest_cache` is only a local acceleration cache of pytest, which may be retained for deleted test nodes and therefore no longer supports the number of tests;
- `docs/pytest-results-week2.xml` is auditable evidence from this real execution.

Regenerated the report using:

```powershell
python -m pytest `
  tests/ `
  -q `
  -p no:cacheprovider `
  --junitxml=docs/pytest-results-week2.xml
```

## 3. Test Environment

```text
Operating system: Windows
Python：3.11.15
Virtual environment: Conda anallysys
```

Core dependent version:

| Dependency | Version |
|---|---:|
| FastAPI | 0.141.1 |
| Pydantic | 2.13.4 |
| HTTPX | 0.28.1 |
| pytest | 9.1.1 |
| OpenAI Python SDK | 2.53.0 |
| ChromaDB | 1.5.9 |
| pdfplumber | 0.11.10 |

Dependence on inspection:

```powershell
python -m pip check
```

Results:

```text
No broken requirements found.
```

## 4. Final execution of the order

### Python Version

```powershell
python --version
```

Results:

```text
Python 3.11.15
```

### Reliance on integrity

```powershell
python -m pip check
```

Results:

```text
No broken requirements found.
```

### Production syntax compilation

```powershell
python -m compileall `
  -q `
  app `
  scripts `
  main.py
```

Result: Pass, ungrammatically wrong output.

### Apply import and route assembly

```powershell
python -c "from main import app; print('应用导入成功')"
```

Results:

```text
Apply import success
```

Main proven routes:

```text
GET    /health
POST   /api/v1/triage
POST   /api/v1/triage/stream
POST   /api/v1/knowledge/documents
GET    /api/v1/knowledge/documents
DELETE /api/v1/knowledge/documents/{document_id}
POST   /api/v1/knowledge/query
```

### Full Test

```powershell
python -m pytest `
  tests/ `
  -q `
  -p no:cacheprovider `
  --junitxml=docs/pytest-results-week2.xml
```

Results:

```text
244 passed in 3.78s
```

JUnit XML records timed: 3.766 seconds. The timing of terminal displays differs slightly from the time calibre recorded in XML and is normal.

## 5. Test quarantine policy

Automation tests do not call for real external services.

```text
Test Input
  ↓
Fake / Mock / Temporary Directory
  ↓
Operating modules tested
  ↓
Fixed and acclaimable results
```

Main means of isolation:

### Fake Embedding

Fake Embedding returns the fixed vector according to the test input.

For validation of:

- Batch processing;
- Text and vector order;
- Vector cache;
- Document ingestion;
- Query Embedding；
- Search and organize.

### Fake LLM

Fake LLM returns a fixed structured answer or preset anomalies.

For validation of:

- Prompt input;
- JSON Parsing;
- Evidence number;
- Timeout and upstream errors;
- (a) Invalid response;
- RAG answers and rejects.

### `MagicMock`

Simulates the sync object, the SDK attribute chain and the timer.

### `AsyncMock`

Methods used to simulate that must be `await`, e.g.:

```text
client.embeddings.create()
client.chat.completions.create()
compare_rag_and_bare()
```

### `httpx.MockTransport`

Simulates the HTTP response in memory, without accessing the real network.

For validation of:

- (a) Evaluating scripts;
- (a) RAG control experiments;
- HTTP status error;
- Request ID；
- Timeout conversion;
- Responding to contractual error.

### FastAPI Dependence on Coverage

Use:

```text
application.dependency_overrides
```

Replace real Service with Fake Service.

This way the API test can cover:

```text
HTTPX
→ FastAPI
→ Middleware
→ Router
→ Pydantic
→ Fake Service
→ HTTP Response
```

### `tmp_path`

Pytest creates an independent temporary directory for each relevant test.

For:

- Temporary PDF, Markdown and TXT;
- Provisional uploading of documents;
- Temporary Chroma data;
- Interim JSON/Markdown report.

The test will not modify the formal:

```text
chroma_data/
robot_knowledge_v4
data/source/
```

## 6. Test files and quantity

| Test File | Number | Main Test Object |
|---|---:|---|
| `test_api_contract.py` | 5 | Health check-ups, requests for verification, Request ID |
| `test_bare_llm.py` | 8 | standalone LLM Prompt, SDK call, anomaly and time-consuming |
| `test_chunking.py` | 8 | Fixed window, overlap, Chunk ID, metadata |
| `test_citation_evaluation.py` | 12 | Individual Quoting Matching and Summarizing Indicators |
| `test_citation_evaluation_schemas.py` | 10 | Quote the Pydantic compact |
| `test_compare_chunking.py` | 5 | Two sets of split parameters compared report |
| `test_compare_rag_vs_bare_llm.py` | 18 | CLI and resources are closed against the core of the experiment. |
| `test_document_catalog.py` | 3 | Document List and Delete Business |
| `test_document_loader.py` | 15 | PDF, TXT, Markdown Parsing and Error |
| `test_document_preparation.py` | 8 | Document Hash, Loader route and Chuck ready. |
| `test_document_service.py` | 4 | Full ingestion and failure path |
| `test_embedding_client.py` | 3 | Embedding Configuration and Client Plant |
| `test_embedding_service.py` | 3 | Embedding Request and Response Verification |
| `test_error_responses.py` | 5 | Apply anomaly to safe HTTP response |
| `test_evaluate_citations.py` | 14 | Quote HTTP scripts and reports |
| `test_evaluation_data.py` | 4 | 20 Gold Question data quality |
| `test_evaluation_schema.py` | 5 | Gold Question and Retrieval Report Compact |
| `test_in_memory_retriever.py` | 8 | MemoryTop-K Retrieving and Sorting |
| `test_ingest_documents.py` | 3 | Batch Ingestion Script |
| `test_knowledge_answer.py` | 9 | RAG Prompt, answer resolution and security constraints |
| `test_knowledge_api.py` | 12 | Document Management and Query HTTP API |
| `test_knowledge_config.py` | 6 | RAG Configuration Range and Default Threshold |
| `test_knowledge_dependencies.py` | 4 | FastAPI relies on assembly and client close |
| `test_knowledge_query_schemas.py` | 8 | Question and answer requests, responses, quotations and Draft compacts |
| `test_knowledge_query_service.py` | 5 | Retrieving, rejecting, LLM and referencing allowlist |
| `test_knowledge_schemas.py` | 6 | Document records, lists and deletion responses |
| `test_rag_comparison_schema.py` | 5 | RAG Compact against Experimental Reporting |
| `test_retrieval_baseline.py` | 7 | Embedding batch and cache |
| `test_retrieval_evaluation.py` | 6 | Evidence location matching and retrieval indicators |
| `test_retrieval_math.py` | 7 | Cosine Similarity Boundary and Illegal Vector |
| `test_run_retrieval_baseline.py` | 5 | Pure function part of a real baseline script |
| `test_service_errors.py` | 2 | Week 1 LLM configuration and overtime conversion |
| `test_service_responses.py` | 6 | Week 1 LLM response |
| `test_stream.py` | 3 | SSE Success and Failed Event Series |
| `test_upload_staging.py` | 3 | Upload suspense, size limits and cleanup |
| `test_vector_store.py` | 9 | Chroma writes, retrieves, repeats and deletes |
| **Grand total** | **244** | |

One test may contain multiple assertions, so the number of tests is not equal to the amount asserted.

## 7. Document Ingestion Test

Tested modules:

```text
DocumentLoader
TextDocumentPreparationService
TextChunker
EmbeddingService
DocumentService
ChromaVectorStore
```

Expected processes:

```text
Documentation
→ Type and Size Validation
→ SHA-256
→ Repeat check
→ Loader Parsing
→ SourceTextSegment
→ TextChunker
→ DocumentChunk
→ Embedding
→ EmbeddedChunk
→ ChromaDB
→ DocumentRecord
```

Main coverage:

- (a) Documents that do not exist;
- (a) Empty files;
- Unsupported extensions;
- (a) Super-large documents;
- UTF-8 text;
- PDF textless page
- Markdown chapter path;
- TXT chapters;
- (b) PDF enzyme restoration;
- Chunk Size and Overlap;
- Global `chunk_index`;
- `content_hash`；
- Repeat the text vector reuse;
- Embedding batch processing;
- Duplicate documents;
- Changes in the original document during ingestion;
- All Embedding is written after success;
- DocumentRecord and the actual number of Chunk.

Expected successful results:

```text
DocumentRecord.status = ready
DocumentRecord. chunk_count = actual number of Chunk
Every chunk can be traced back to the file and location.
```

## 8. ChromaDB Test

Tested modules:

```text
ChromaVectorStore
```

Tests use a temporary Durable Directory and a real Chroma client, but do not connect to an external database.

Main coverage:

- Create and open Collection;
- (b) Collection Embedding Model Metadata;
- (b) The mixing of different Embedding models is not permitted;
- Document presence check;
- Document consistency with the Chunk metadata;
- Use `add()` for writing;
- Repeating document rejection;
- Top-K Retrieval;
- Cosine Distinction, Similarity;
- The ranking starts with 1;
- Refusal of unlawful distance;
- Missing metadata rejected;
- Delete all Chunk at `document_id`;
- Deletes a document that does not exist.

Anticipated search process:

```text
query_embedding
→ Chroma query
→ ids/documents/metadatas/distances
→ Number of fields verified
→ similarity = 1 - distance
→ DocumentChunk
→ RetrievedChunk
```

## 9. RAG Query Test

Tested modules:

```text
KnowledgeQueryService
OpenAIKnowledgeAnswerProvider
Knowledge Prompt Builder
```

Expected success processes:

```text
KnowledgeQueryRequest
→ Query Embedding
→ Top- K Retrieval
→ Top- 1 Reaching threshold
→ LLM generation of Knowledge
→ Validation of allowlist
→ Python assemblyCitiation
→ KnowledgeQueryResponse
```

Main coverage:

- (a) Direct rejection of empty search results;
- Top-1 Direct refusal below threshold;
- Do not call LLM when refusing to answer;
- Call LLM when Top-1 is equal to or above the threshold;
- LLM must return `KnowledgeAnswerDraft`;
- Unknown Evidence ID rejected;
- Quoting metadata from real Chunk;
- (b) `citations=[]` when the answer is denied;
- (a) At least one reference exists in the response;
- `retrieval_ms` not negative;
- (a) Retain the rules for qualification of status such as “the exposure draft”;
- (b) To retain rules on security prohibitions, coercion and preconditions;
- (b) The rule of minimum sufficient evidence pool;
- Illegal JSON, empty content and empty candidate results.

## 10. Knowledge base API testing

Tested route:

```text
POST   /api/v1/knowledge/documents
GET    /api/v1/knowledge/documents
DELETE /api/v1/knowledge/documents/{document_id}
POST   /api/v1/knowledge/query
```

Main coverage:

- Upload successfully returned `201`;
- Repeat upload returns `409`;
- Super file returns `422`;
- (b) Uploading temporary files and cleaning them after success and failure;
- Document lists have data and empty lists;
- Successfully deleted;
- Deletes that no document returns `404`;
- Illegal Document ID returns `422`;
- Regular query returns Answer and Citation;
- (a) Return structured refusals to answer without an answer;
- The blank question returns `422`;
- LLM returns `504` overtime;
- Request ID appears in the unified error response.

Test does not start real Uvicorn, but calls FastAPI applications in memory using the ASGI Transport.

## 11. Evaluation system testing

### Retrieval Evaluation

Tested modules:

```text
retrieval_evaluation.py
evaluate_retrieval.py
```

Main authentication:

- (a) Filename matching;
- Standardization of page numbers and chapters;
- Top-1 and Top-3 hits;
- (a) Multiple leaps in the integrity of evidence;
- (b) Unresolved wrong recall;
- Recall indicator;
- JSON and Markdown report.

### Citation Evaluation

Tested modules:

```text
citation_evaluation.py
evaluate_citations.py
```

Main authentication:

- Returns whether the reference matches the Gold file and location;
- Number of correct references;
- (b) Different expected evidence is heavy;
- Reference correctness;
- Citation coverage;
- Response rate to answerable questions;
- Complete evidence response rate;
- No answer to the correct rejection rate;
- HTTP failed and ineffective responses;
- Sequencing of reports;
- command line parameters.

## 12. RAG and Standalone llm Experiment Test

Tested modules:

```text
bare_llm.py
rag_comparison.py
compare_rag_vs_bare_llm.py
```

Expected processes:

```text
Select Gold Question
→ Call RAG HTTP API
→ Call Naked LLMProvider
→ Recording two time-consuming
→ Construct RagVsBareLLMReport
→ It's JSON and Marktown.
```

Main coverage:

- Default q014 and q020;
- (a) Specify the order of the title;
- Repeated and non-existent titles;
- standalone LLM does not receive Gold and RAG evidence;
- RAG HTTP error and Request ID;
- HTTP timeout;
- RAG Responds to Contract Validation;
- Close when the LLM client is successful or unsuccessful;
- RG HTTP client closed;
- (b) Do not write a missing report when the experiment fails;
- UTF-8 JSON；
- Markdown contains two sets of answers and references;
- Terminal summary.

## 13. Week 1 regression test

Week 2, following Week 1 code, continues to cover:

- `/health`；
- Triage requests validation;
- Non-flow LLM analysis;
- LLM configuration is missing;
- LLM timeout;
- Invalid LLM JSON;
- Request ID；
- SSE `meta → delta* → done`；
- SSE `error` termination;
- Internal information is not leaked.

This proves that the inclusion of the RAG did not destroy the original fault-sharing capability.

## 14. Authentication for real integration

The real integration validation is carried out separately from the automated testing, as it uses the network, API Key and the real local knowledge base.

### 14.1 Ingestion

Collection：

```text
robot_knowledge_v4
```

Results:

| Documentation | Chuck Number |
|---|---:|
| Cognex DataMan 260 Reference Manual | 98 |
| Victron Lithium Smart Battery Manual | 165 |
| Warehousing robot test protocol and determination criteria | 35 |
| Description of storage robot failure | 4 |
| Input draft of industrial mobile robotic safety standards | 53 |
| **Grand total** | **355** |

### 14.2 Document API

Authenticated:

- First upload returns `201 Created`;
- Second upload of the same file back to `409 Conflict`;
- List interface returns the ingestion document;
- Delete interfaces to delete all Chunk of the specified document;
- The document does not appear again in the list after it is deleted.

### 14.3 RAG Query

Authenticated:

- The question of the standard scope of application returns `200`;
- Answers with real files, page numbers, Chunk IDs and similarities;
- Real-time telemetry returns `abstained=true` and empty references;
- q014 returns two multiple jumps of evidence.

### 14.4 Retrieval assessment

Finally, V4:

| Indicators | Result |
|---|---:|
| Recall@1 | 0.600 |
| Recall@3 | 0.850 |
| Unresolved wrong recall rate | 0.000 |

### 14.5 Quoting Assessment

Threshold `0.60` V2:

| Indicators | Result |
|---|---:|
| Citation correctness | 1.000 |
| Citation coverage | 0.400 |
| Response rate to questions that can be answered | 0.438 |
| Full evidence response rate | 0.438 |
| Correct rate of rejection for unsettled questions | 1.000 |

### 14.6 RAG with standalone LLM

Authenticated:

- q014 RAG returns two pieces of evidence specific to the project;
- q014 Standalone llm creates generic blocking suggestions that conflict with current security protocols;
- q020 RAG directly rejected the answer through the threshold;
- q020 Standalone llm, although refused to answer, had a model call;
- Prompt V2 expressly retains the phrase “no genuine blocking of a circuit for test purposes”.

## 15. Risk coverage

| Risk | Authentication Method |
|---|---|
| Original file type is not valid | Document Service Test |
| Empty or super documents | Document Service and Upload Tests |
| Additional costs for duplicate documents | Hash and repeated testing. |
| Chunk lost his source position. | Chunk Schema and Loader testing |
| Embedding Number Wrong | Embedding Service Test |
| Vector contains Nan or dimension anomalies | Embedding and Retrieving Math |
| Combining Embedding Model | Chroma Collection Test |
| Chroma returns to damaged structure. | Victor Store error test |
| No evidence to call LLM | Low Similarity Refusal Test |
| LLM forged evidence number | Evidence ID allowlist test |
| LLM forged file name or page number | Python Reference assembly test |
| Unsettled Question Return Error Reference | Query Schema and Service tests |
| LLM omitted security binding rules | Prompt Security Code Test and q014 V2 |
| Upstream timeout | SDK Abnormal Conversion Test |
| Configure Missing | Clint Factory Test |
| temporary upload file leak | Upload Staging Test |
| HTTP error leaking internal details | Exception Handler Test |
| Client cannot associate logs | Request ID testing |
| Error in evaluation formula | Manually expected indicator testing |
| We still have a missing report on the failed experiment. | Comparison CLI test |

## 16. Not covered at present

The current automated tests do not yet cover:

- and carry loads and throughputs;
- (a) Long-term operating stability;
- Multi-processes are also written into the same Chroma Collection;
- Large-scale knowledge base performance;
- Scan PDF OCR;
- Complex tables and graphic syntax;
- Consistency between different Embedding suppliers;
- Full compatibility of different LLM suppliers;
- (b) Real re-testing of network vibrations and restricted flow;
- Prompt Injection Red Team Test;
- (a) Document virus scanning;
- User identity and document privileges;
- (a) Multiple tenant segregation;
- Real RCS, WMS and telemetry integration;
- (a) Auto-execution for safe operations;
- (a) The official frontend;
- Production deployment and high availability.

## 17. Test Result Boundaries

`244 passed` indicates that the acts declared in the current test are in line with expectations and do not imply:

- All PDFs can be correctly analyzed;
- All natural language questions can be answered correctly;
- LLM always adheres to Prompt;
- The current threshold applies to other languages;
- The system meets the requirements for production safety certification;
- The system directly controls robots.

Automation test certification of engineering contracts, Gold assessment of current language quality and authentic integration validation to confirm that external systems can be connected. There is no substitute for one another.

## 18. Final Conclusion

Week 2 final test results:

```text
244 passed
0 failed
```

Overwrite:

```text
Document Parsing
+ Text splits
+ Embedding
+ Chroma Enduring
+ Top- K Retrieval
+ Refusal of threshold
+ Evidentiary constraints
+ Code Reference
+ Document Management API
+ Retrieval Evaluation
+ Citation Evaluation
+ RAG and Standalone llm Experiment
+ Week 1 return
```

Week 2 learning goals are currently being achieved, leading to re-engineering and subsequent RAG optimization.
