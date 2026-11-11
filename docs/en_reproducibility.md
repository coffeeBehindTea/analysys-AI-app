# Week 3 Reproducibility Baseline

Date of record: 2026-08-17  
Project: Robot Knowledge Base API  
Status: Week 3 Day 1 – 2 Mission I baseline validated

## 1. Purpose

This document records that the Week 3 project, starting with a clean code snapshot, has completed the following actual processes and results:

1. Create a new Python virtual environment;
2. (a) Dependence on lock-on version installation;
3. Run offline regression without the old `.env`, the Chroma database and testing caches;
4. (b) Implementation of real Embedding intake using the minimum publicly available, redacted language;
5. Retrieving the correct Chunk from the newly created EnduranceCholaction;
6. Saves machine-readable JUnit test evidence.

This baseline is used to distinguish between the question of follow-up from:

- (b) Environment or dependency;
- Document resolution and cut;
- Embedding Services;
- Vector storage;
- (a) Retrieving, sorting and evidence gate;
- Week 3 new code.

Until this baseline is established, no adjustments to the hybrid search, rearrangement or evidence gate parameters will be made.

## 2. Baseline conclusion

| Checkpoint | Result |
|---|---|
| Clean code snapshots can be installed. | Pass |
| Python relies on the integrity of relationships | Pass |
| Week 2 Succession Test | 244/244 Passed |
| Minimal redacted language intake | 1 successful files |
| Number of document parsing locations | 1 |
| Generate Chunk | 1 |
| Query vector dimensions | 2048 |
| Endurance Chroma Retrieval | Correctly recall target, Chunk. |
| Add smoke script test | 4/4 Passed |
| Smoke foot phase full test. | 248/248 Passed |
| Add Gold Data Quality Test | 11/11 Passed |
| Current Full Test | 259/259 Passed |
| Full Text Re-ingestion | 5 documents, 355 Chunk |
| Current 20 pure vector baseline recurrence | Recall@1 0.600；Recall@3 0.850 |
| Extension 28 pure vector baseline | Recall@1 0.567；Recall@3 0.767 |
| Extension baseline no answer error recall rate | 0.000 |
| Test failure, error, skip | Both, 0. |

## 3. Code source and boundary

Local Week 3 code from the following open warehouse `main` branch in Week 2 after consolidation Download ZIP:

[https://github.com/coffeeBehindTea/analysys-AI-app](https://github.com/coffeeBehindTea/analysys-AI-app)

Download date: 2026-08-17.

ZIP does not contain `.git`, so the local directory has no branch and submission history. This record uses a baseline that relies on documents such as Hashi, sample language such as Hashi, JUnit and the results of the actual operation.

This method allows for a clean code snapshot, but does not permanently identify a Git submission like a committee SHA. When week 3 is subsequently officially published, the corresponding submitting SHA or creating version labels should be recorded.

The initial directory does not explicitly contain:

- `.env`；
- `chroma_data/`；
- `.pytest_cache/`；
- `__pycache__/`；
- Week 2 produced local vector databases;
- The full source of the authorization was not circulated.

The initial `data/source/` contains only `.gitkeep`.

## 4. Operating environment

| Item | Actual value |
|---|---|
| Operating System Platform String | `Windows-10-10.0.22631-SP0` |
| Python Achieved | `CPython` |
| Python Version | `3.11.15` |
| Environmental management tools | Conda |
| Conda Environment Name | `analysys-week3` |
| Reliance on installation | `python -m pip install -r requirements.txt` |

Python returns a Windows Platform string that retains as a true execution value. `10.0.22631` is the current Windows NT build number.

## 5. Dependence on baselines

The project relies directly on the `==` lock version used in the root directory `requirements.txt`.

`requirements.txt` for SHA-256:

```text
361917cfab9b3e1c20d809a3e812002db3a8594325b4afe12c7e4b88d4528e39
```

Create environment and installation dependency:

```powershell
# Create a separate Week3 Python environment.
conda create `
    --name analysys-week3 `
    python=3.11 `
    pip `
    -y

# Activate the environment.
conda activate analysys-week3

# Installs all dependencies in requirements.txt.
python -m pip install `
    -r requirements.txt

# Checks if the package relies on missing or conflicting versions.
python -m pip check
```

Actual `pip check` Results:

```text
No broken requirements found.
```

`pip check` proves that the dependency between currently installed packages is complete, but cannot be a substitute for the project operations test.

## 6. Modify pre-delineal regression baseline

No `.env` was created at the time of testing, nor was there an old Chroma database. Tests are isolated from real external services through Fake, Mock, relying on coverage and temporary catalogues.

The Windows default pytest temporary directory was created by a separate identity and is not accessible to current users. To avoid relying on global temporary directory privileges, the use of a dedicated directory within the project is tested.

Create parent directory first:

```powershell
# Pytest will not revert to the creation of the --basetemp parent directory.
# It is therefore necessary to create a tmp directory that is excluded by...gitignore.
New-Item `
    -ItemType Directory `
    -Path "tmp" `
    -Force |
    Out-Null
```

Test command before modification:

```powershell
python -m pytest `
    tests `
    -q `
    -p no:cacheprovider `
    --basetemp="tmp/pytest-week3" `
    --junitxml="docs/pytest-results-week3-baseline.xml"
```

Parameter description:

- `-q`: Reduce article-by-article output, retain results and false information;
- `-p no:cacheprovider`: disabled `.pytest_cache`;
- `--basetemp`: Specify a temporary directory for pytestFixture;
- `--junitxml`: Generate machine-readable test reports.

Actual JUnit results:

| Fields | Value |
|---|---:|
| tests | 244 |
| failures | 0 |
| errors | 0 |
| skipped | 0 |
| time | 3.504 seconds |

Evidentiary documents:

```text
docs/pytest-results-week3-baseline.xml
```

The 244 tests covered the main capabilities inherited from Week 2, including:

- FastAPI Compact of Request and Response;
- Pydantic Schema；
- (b) Intermediate and unified abnormal responses;
- (a) Document resolution, cut-off and Hash;
- Embedding Client and Response Validation;
- Chroma temporary storage and retrieval;
- Document upload, query and deletion interfaces;
- RAG Gate Control and Reference Allowlist;
- (a) Retrieving and referencing the assessment script;
- RAG versus nud LLM against script.

## 7. Minimum language for public disclosure

The complete Week 2 language material contains copyright-reserved manufacturers ' manuals, draft standards and self-generated materials for local learning only, and therefore is not re-distributed from open warehouses.

Open warehouses provide only:

```text
samples/corpus/robot-fault-demo.txt
```

This document is a teaching simulation that does not correspond to real customers, warehouses, robotic models or production parameters.

Sample file SHA-256:

```text
b8866a03c60cfbb5d697d9910c01d2122cab822c7d4e5b7bb31a4848611ee617
```

Copy to local ingestion directory:

```powershell
Copy-Item `
    -LiteralPath "samples/corpus/robot-fault-demo.txt" `
    -Destination "data/source/robot-fault-demo.txt"
```

Before and after copying SHA-256 comparison results:

```text
True
```

The Hash also became `document_id` after the document was ingestion.

The source of the full text, authorized boundaries and documents are found in Hashi:

```text
docs/corpus-sources.md
```

## 8. Local Configuration

Create local configuration from an open template:

```powershell
Copy-Item `
    -LiteralPath ".env.example" `
    -Destination ".env"
```

`.env` is kept on this machine only, does not submit, and does not record any API Key in this document.

Unsensitive configuration used for this minimum ingestion:

| Configure | Value |
|---|---|
| Embedding Model | `embedding-3` |
| Chroma Endurance Directory | `chroma_data` |
| Chroma Collection | `robot_knowledge_week3_smoke` |
| Chunk Size | 800 |
| Chunk Overlap | 120 |
| Embedding Batch Size | 64 |

Ingestion script does not call the generated LLM, but calls the real Embedding service and therefore needs to be effective:

- `EMBEDDING_API_KEY`
- `EMBEDDING_BASE_URL`
- `EMBEDDING_MODEL`

## 9. Minimal Ingestion Validation

Execute orders:

```powershell
python -m scripts.ingest_documents `
    --source-dir "data/source" `
    --output "docs/ingestion-log-week3-smoke.md"
```

Ingestion process:

```text
Scan supported files
→ Verify Extensions and File Size
→ Calculate original file SHA-256
→ Parsing TXT Chapters
→ Generate SourceTextSegment
→ 800/120 split by DocumentChunk
→ Call for real Embedding Service
→ Verify the number, index, dimension and value of vectors
→ Writing to Croma Collection
→ Generate document-level intake logs
```

Actual results:

| Item | Result |
|---|---|
| Total number of documents | 1 |
| Success | 1 |
| Skip | 0 |
| Fail | 0 |
| source_file | `robot-fault-demo.txt` |
| document_id | `b8866a03c60cfbb5d697d9910c01d2122cab822c7d4e5b7bb31a4848611ee617` |
| Number of places | 1 |
| Chuck Number | 1 |
| Status | `ingested` |

The local intake log does not contain API Key, vector or Chunk text, but is still excluded as a running process file by `.gitignore`. A summary of the publicly available results is kept here.

## 10. Lasting Search Authentication

The smoke retrieval question for Week 2 was hard-coded as `ERR-DRV-3005` in the full local language. Week 3 Replace the script with the receiving:

- `--query`
- `--top-k`

The default question is changed to `ERR-DEMO-1001` in the public sample.

Execute orders:

```powershell
python -m scripts.smoke_chroma_retrieval `
    --query "模拟故障代码 ERR-DEMO-1001 的观察现象、安全排查步骤和恢复条件是什么？" `
    --top-k 3
```

Actual results:

| Item | Result |
|---|---|
| Query vector dimensions | 2048 |
| Request Top-K | 3 |
| Actual recalls | 1 |
| Top-1 Similarity | 0.603805 |
| Source File | `robot-fault-demo.txt` |
| Location | `section: ERR-DEMO-1001` |
| Chunk ID | `b8866a03c60cfbb5d697d9910c01d2122cab822c7d4e5b7bb31a4848611ee617:000000` |

Only one result is returned because only one of the Collecations is chunk. `ChromaVectorStore` Practical requests:

```text
min(top_k, stored_chunk_count)
= min(3, 1)
= 1
```

Similarity is converted from cosine distance:

```text
similarity = 1 - distance
```

`0.603805` does not answer the correct rate or factual confidence, but only the degree of proximity in the current model vector space.

This smoke test proves that:

- `.env` configurations are readable;
- Real queries Embedding can be generated;
- Query is compatible with the document vector dimension;
- Sustainableization can be reopened;
- Correct Chunk, text and source metadata can be retrieved.

Because there is only one Chunk in the knowledge base, this result does not prove the quality of multi-candidature rankings or replace a complete Gold assessment.

## 11. Week 3, first modified test

New document:

```text
tests/test_smoke_chroma_retrieval.py
```

Add 4 smoke search script offline tests covering:

- Use the open `ERR-DEMO-1001` for default questions;
- `--query` and `--top-k` parameters can override default values;
- Empty queries are rejected;
- The integer number of `top_k` was rejected.

Add the following new text:

```text
tests/test_week3_gold_questions.py
```

The document contains 11 extended Gold data quality tests for official assessment data:

```text
data/eval/gold_questions.jsonl
```

The test process is:

```text
Read real Gold JSONL files
→ Parse by line-by-line
→ Execute Pydantic data validation using GoldQuestion
→ Index by Question_id
→ Check the number, type distribution, labels, control relationships and evidence locations
```

These tests specifically examine:

- The total number of Gold questions is at least 28;
- `q001` ~ `q028` All present and numbered consecutively;
- The number of single evidence, multiple evidence and no answer questions meets the design requirements;
- (a) The total number of questions available and the total number of anticipated evidence meets the requirements;
- New subjects cover fault-code variants, product models, numerical boundaries and multiple jump processes;
- The text of the six sets of old and new questions is different, but the evidence is expected to intersect;
- (a) The expected source file name and page number or chapter format of the evidence is valid;
- `q028` is an answerless question of “entity exists but relations do not exist”.

These tests do not perform real searches, nor do they call Embedding, ChromaDB or LLM. They test the “correctness of the ruler itself” rather than the ability of the retrieval system to answer these questions.

Current Full Volume Test Command:

```powershell
python -m pytest `
    tests `
    -q `
    -p no:cacheprovider `
    --basetemp="tmp/pytest-week3-full" `
    --junitxml="docs/pytest-results-week3.xml"
```

Actual JUnit results:

| Fields | Value |
|---|---:|
| tests | 259 |
| failures | 0 |
| errors | 0 |
| skipped | 0 |
| time | 3.683 seconds |

Evidentiary documents:

```text
docs/pytest-results-week3.xml
```

The number of tested changes are:

```text
244 Week 2 Succession tests
→ Add 4 smoke script tests
→ 248 adopted
→ Add 11 new Gold data quality tests
→ 259 adopted
```

Therefore, the expansion of Gold data did not disrupt the original document processing, quantification, retrieval, RAG, knowledge base API, reference assessment and troubleshooting.

## 12. Week 2 full text history baseline

The data below are from the stored redacted summaries of Week 2 and are not recalculated by a single Chunk public sample.

Full text history configuration:

| Item | Value |
|---|---|
| Collection | `robot_knowledge_v4` |
| Embedding Model | `embedding-3` |
| Gold Questions | 20 |
| Answerable questions | 16 |
| No answer to question | 4 |
| Top-K | 3 |
| Online gate control threshold | 0.60 |

Search for historical indicators:

| Indicators | Result |
|---|---:|
| Recall@1 | 0.600 |
| Recall@3 | 0.850 |
| Top-1 Full recall | 8/16 |
| Top-3 Full recall | 13/16 |
| Unresolved wrong recall rate | 0.000 |

Online reference to historical indicators:

| Indicators | Result |
|---|---:|
| Answerable questions actually answered | 7/16 |
| Response rate to questions that can be answered | 0.438 |
| Full evidence response rate | 0.438 |
| Citation correctness | 1.000 |
| No answer correct response rate | 1.000 |

Sources of evidence:

- `docs/en_retrieval-evaluation-summary.md`
- `docs/en_citation-evaluation-summary.md`
- `docs/en_rag-vs-bare-llm-summary.md`

The minimum public sample can only be duplicated by modern code, ingestion and lasting retrieval links and cannot reproduce the indicators on the five full text above.

## 13. Delivery at the border

The following may be submitted publicly:

- Source code;
- Automation testing;
- `requirements.txt`；
- `.env.example`；
- `.gitignore`；
- Open redacted samples;
- This statement could have been reproduced;
- (a) The results of the JUnit test, which does not contain the text and key;
- Summary of redacted assessments.

Keep only locally:

- `.env`；
- API Key, token and account number;
- `chroma_data/`；
- `tmp/`；
- `__pycache__/`；
- `.pytest_cache/`；
- A copy of the actual language in `data/source/`;
- Original assessment report with full Chunk text or long quotations;
- Manuals and standard information for manufacturers were not re-issued.

`.gitignore` is responsible for preventing these local products from entering public delivery.

## 14. Current Limitations

1. Code from `main` branch ZIP, no local committee SHA; formal subsequent delivery should be recorded for submission to SHA.
2. The smallest example is only one Chunk, which does not measure the quality of the candidate's sorting.
3. Real Embedding requests to rely on networks, service provider availability and account balances.
4. Vector results and similarities may change when the Embedding service upgrade model is followed.
5. The replacement of the Embedding model will require the use of a new Collaction and re-ingestion.
6. The complete Week 2 indicator relies on five local languages and open warehouses do not redistribut these documents.
7. JUnit Time is the single result of this machine and is only recorded and not used as a stability indicator.

## 15. Order of assessment and current checkpoint

Before modifying the search algorithm, the assessment is carried out in the following order:

```text
Week 2 old 20 historical baseline
    ↓
Create a search failed account on a question-by-issue basis
    ↓
Add 8 difficult Gold questions
    ↓
Run pure vector B on the extension of Gold
    ↓
Achieve a combination of keywords and vectors
    ↓
Compare candidate policies with the same input conditions
```

Current completion state:

- [x] Preservation of Week 2 old 20 historical baseline;
- [x] Completion of q001-q020 as the cause of the failure;
- [x] Add a total of 8 difficult questions from q021 to q028;
- [x] Completion of the extended Gold data quality test;
- [x] Re-ingestion of five complete local languages;
- [x] Operation of pure vector baseline B on 28 lanes Gold;
- [] Achieve keyword retrieval;
- [] Integrate vectors with keyword candidates;
- [] To achieve a rewrite or reordering of candidate options for queries;
- [] Comparing three search strategies under the same conditions;
- [] Calibrate online access controls based on search results.

Pure vector baseline B has been frozen. The following shall be used for subsequent mixed search, rewrite or rearrange:

- Five copies of language material;
- 28 Gold Question;
- Embedding model;
- chunk parameters;
- Top-K；
- Evidence location matching rule.

Otherwise, changes in indicators may result from input differences and cannot be shown to be improvements resulting from search strategies.

## 16. Expand Gold pure vector baseline B

### 16.1 Purpose

Baseline B Achieved using pure vector search that has not been completed to retrieve, integrate or rewrite a keyword.

Its duty is to answer:

```text
In the face of the expanded 28 Gold Roads problem,
How many correct evidence can you find in the current pure vector search?
```

All subsequent candidate strategies will be compared with the results.

### 16.2 Full Language Re-ingestion

The `chroma_data` of Week 2 was not copied, but was re-enacted from five locally reserved languages:

```text
Document Parsing
→ SourceTextSegment
→ Chunk cut.
→ Embedding
→ Writing to the new Croma Collaction
```

Ingestion configuration:

| Item | Value |
|---|---|
| Collection | `robot_knowledge_week3_vector_baseline` |
| Embedding Model | `embedding-3` |
| Chunk Size | 800 |
| Chunk Overlap | 120 |
| Number of documents | 5 |
| Success | 5 |
| Skip | 0 |
| Fail | 0 |
| Chunk total | 355 |

Intake log:

```text
docs/ingestion-log-week3-vector-baseline.md
```

The ingestion log saves only file-level statistics, documents IDs and Chunk numbers, without saving vectors, API Key or Chunk text.

### 16.3 Evaluation configuration

| Item | Value |
|---|---|
| Gold File | `data/eval/gold_questions.jsonl` |
| Gold Total Problem | 28 |
| Number of questions answered | 23 |
| Number of questions not answered | 5 |
| Total anticipated evidence | 30 |
| Top-K | 3 |
| Similarity reference threshold | 0.60 |
| Retrieval Policy | Pure vector search |

Execute orders:

```powershell
python -m scripts.evaluate_retrieval --questions "data/eval/gold_questions.jsonl" --json-output "docs/retrieval-evaluation-week3-vector-baseline.json" --markdown-output "docs/retrieval-evaluation-week3-vector-baseline.md" --top-k 3 --similarity-threshold 0.60
```

The order:

- Call the real Embedding service to generate 28 search vectors;
- Perform local Chroma Top-3 search for each query vector;
- Do not call the generated LLM;
- Match evidence based on the source document and page number or chapter in the Gold file;
- Generates machine-readable JSON and manual readable Markdown reports.

### 16.4 Overall results

| Indicators | Result |
|---|---:|
| Number of questions answered | 23 |
| Number of questions not answered | 5 |
| Total anticipated evidence | 30 |
| Top-1 Hits of evidence | 17 |
| Top-3 Hits of evidence | 23 |
| Recall@1 | 0.567 |
| Recall@3 | 0.767 |
| Top-1 Full recall questions | 10/23 |
| Top-3 Full recall questions | 16/23 |
| Unresolved Wrong recall Number | 0/5 |
| Unresolved wrong recall rate | 0.000 |

The evidence level Recall calculates that:

```text
Recall@1
= 17 / 30
= 0.5667
≈ 0.567

Recall@3
= 23 / 30
= 0.7667
≈ 0.767
```

The “full recall question” is different from the evidence level Recall.

For the question of multiplication, which contains two anticipated evidence:

```text
Only one piece of evidence is recalled.
→ Recall can add to the evidence.
→ But the question isn't exactly recall.
```

It is therefore important to report both evidence- and problem-level indicators.

### 16.5 Old 20 Recurring results

From the 28 report alone q001-q020:

| Indicators | Result |
|---|---:|
| Number of questions answered | 16 |
| Number of questions not answered | 4 |
| Total anticipated evidence | 20 |
| Recall@1 | 0.600 |
| Recall@3 | 0.850 |
| Top-1 Full recall | 8/16 |
| Top-3 Full recall | 13/16 |
| Unresolved wrong recall rate | 0.000 |

The result is consistent with the historical baseline saved by Week 2.

This proves that:

- Five copies of re-ingestion succeeded;
- The document resolution and the Chunk split results are consistent;
- Current Embedding and Chroma configurations can reproduce historical search indicators;
- The general recall drop is not a return to the old problem.

### 16.6 Add 8 result

q021-q028 subsets consist of:

- 7 answers;
- 1 questions without answers;
- 10 anticipated evidence.

Results:

| Indicators | Result |
|---|---:|
| Recall@1 | 0.500 |
| Recall@3 | 0.600 |
| Top-1 Full recall | 2/7 |
| Top-3 Full recall | 3/7 |
| Unresolved wrong recall rate | 0.000 |

The new problem has been revealed:

- (b) Fault code variants are not stable and accurate recall;
- (b) The precariousness of the ranking of pages in the cross-language handbook;
- The values are pages that are easy to recall but have different uses;
- The problem of multiplication is often only partially proven;
- The issue of having found complete evidence may still be rejected by the Top-1 threshold.

Thus, the overall reduction of the 28 Recall was due to new difficulties rather than to the degradation of the former 20.

### 16.7 Threshold of likeness

This offline search assessment, `0.60`, is used to determine whether there has been an error in recall of the unsettled questions:

```text
No answer question Top-1 Similarity 0.60 or more
→ false_recall=true
```

For questions that can be answered, the results of the assessment script will remain below `0.60` and Top-K will be retained for the purposes of identifying:

```text
The correct evidence has entered Top-K.
But the online portal still refuses to answer.
```

Therefore, `Recall@3` cannot be directly understood as an online response rate.

The online answer is:

```text
Search
→ Evidence gate.
→ LLM Generation
→ Quote Allowlist
→ Respond to Schema Verify
```

### 16.8 Reporting document and delivery boundary

Local complete report:

```text
docs/retrieval-evaluation-week3-vector-baseline.json
docs/retrieval-evaluation-week3-vector-baseline.md
```

The full report contains the results of the subject-by-issue search, the Chunk metadata and the text preview, so it is kept locally and is not submitted directly to the open warehouse.

The publicly available conclusions should be used:

- (a) Summarized indicators, excluding the Chunk text;
- Failed classification;
- Post-sensitization problem number;
- Policy parameters;
- Description of changes in indicators.

The reason for the failure is:

```text
docs/retrieval-failure-taxonomy.md
```

### 16.9 Baseline B Conclusion

The key results for pure vector baseline B are:

```text
Recall@1：0.567
Recall@3：0.767
Top- 3 Full evidence recall: 16/23
Unresolved wrong recall rate: 0.000
```

Day 3 – 4 must meet at least:

1. (a) The full evidence recall rate is not less than that baseline;
2. Explain all changes in Recall@1 and Recall@3;
3. Improving failure codes, models, values and multiple jumps;
4. q017-q020 and q028 are not allowed to be wrongly released;
5. No selective samples are used instead of complete 28 questions.
