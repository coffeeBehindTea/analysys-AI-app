# Week 2 Engineering Summary

## 1. Targets and final results for the week

The goal of the second week is not simply to add a few files to the large language model, but to create a complete, traceable and measurable RAG link:

```text
Open or redacted documents
→ Document Parsing
→ Text splits
→ Embedding Quantification
→ ChromaDB Endurance
→ Top- K Retrieval
→ Similarity evidence gate
→ Evidentiary constraints
→ Code assembly references
→ Automation assessment
```

The following deliveries were finalized this week:

- Supported PDF, Markdown, TXT document ingestion;
- Verify file type, size, empty text and duplicate document;
- SHA-256 to generate stable `document_id`;
- (a) To save complete source metadata for each Chunk;
- Using a stand-alone Embedding configuration to call to quantitative services;
- Use ChromaDB durable document vectors, body text and metadata;
- (a) To enable documents to be uploaded, listed and deleted API;
- (a) RAG Query API with Evidence Quoting and Refusal;
- Creation of 20 Gold Question;
- Complete the search assessment, quote evaluation and the RAG vs. nud LLM experiment;
- (b) The completion of non-network automation tests using Fake, Mock and interim databases;
- Final full automated regression test `244 passed`.

The current knowledge base consists of:

```text
Number of documents: 5
Chunk: 355
Collection：robot_knowledge_v4
Embedding Model: embedding-3
```

Targets for final search:

| Indicators | Result |
|---|---:|
| Number of questions answered | 16 |
| Number of questions not answered | 4 |
| Recall@1 | 0.600 |
| Recall@3 | 0.850 |
| Unresolved wrong recall rate | 0.000 |

Final reference indicator:

| Indicators | Result |
|---|---:|
| Number of questions answered that were actually answered | 7/16 |
| Citation correctness | 1.000 |
| Citation coverage | 0.400 |
| Response rate to questions that can be answered | 0.438 |
| Full evidence response rate | 0.438 |
| Correct rate of rejection for unsettled questions | 1.000 |

This week's request for `Recall@3 >= 0.80` has been met.

---

## 2. Position of this week's work in the overall AI application architecture

The first week was completed with an ordinary LLM application:

```text
User Input
→ FastAPI
→ Operations Service
→ LLM
→ Structured Answers
```

This structure is suitable for summary, rewrite and general analysis, but models rely mainly on the knowledge gained during training. There are obvious problems with robotic instructions, failure codes, safety standards and internal operating procedures:

- The model may not have seen project-specific information;
- Modelling training knowledge may be outdated;
- Models may be mixed with parameters of different models;
- The model does not provide verifiable documents and page numbers;
- The model may complete a reasonable answer when there is insufficient evidence.

The second week added RAG Level of Evidence:

```text
User problems
→ Query Embedding
→ Vector Retrieval
→ Get a small amount of relevant chunk
→ Determining whether evidence is sufficient
→ LLM answers based on evidence.
→ PythonMap Real Reference
```

RAG did not train or modify LLM itself. It changes the information that can be accessed before the model answers, and the system's way of binding the answers.

The RAG layer achieved this week is responsible for:

1. Convert unstructured engineering documents into searchable knowledge;
2. (b) To find the evidence most relevant to the current problem;
3. Control of which issues allow access to the generation model;
4. (b) Providing retroactive sources for the response;
5. Provides a controlled knowledge tool for future RobotOps Copilot or Agent.

Even if the future were to be joined by Agent, Agent should not rely directly on model memory to answer engineering questions, but should use the knowledge base search capability that has been achieved this week.

---

## 3. Why build an evaluation and then develop RAG API?

One of the most important engineering decisions this week is to define what is correctly retrieved and then develop a question-and-answer interface.

If there is no Gold Question, the developers can usually judge it subjectively by a few ad hoc questions:

```text
That looks good.
```

But "looks good" does not answer the question:

- Did the correct evidence enter Top-3?
- Do you recall all the pieces of evidence?
- Are the questions left wrongly?
- Is the parameter adjusted to be really better or is it just an example of improvement?
- Does the reference really come from Gold's expected document?

So this week 20 machine-readable questions were created:

```text
16 answerable questions
4 unanswered questions
It contains 4 multiple jumps.
```

Each of the answers to the question indicates:

- Question ID;
- Question text;
- Type of problem;
- Reference answers;
- Expected `source_file`;
- Expected `page_or_section`.

In this way, the effects of the search can be calculated through repeated proceedings rather than relying on subjective impressions.

---

## 4. Design of language and evidence compacts

### 4.1 Language Selection

This week, five public or redacted materials were used:

1. Cognex DataMan 260 Public reference manual;
2. Victor Lithium Smart Battery Public Handbook;
3. (b) A draft of industrial mobile robotic safety standards for comments;
4. Self-generated, redacted statement of failure of storage robots;
5. Test protocols and determination criteria based on open storage scenes.

This information covers:

- Sensors;
- Battery and implementer-related parameters;
- Safety standards;
- (a) Failure records;
- Test protocol.

This brings the assessment to a real knowledge base of robotic development, which includes both English product manuals and Chinese default codes, normative provisions and cross-documentary issues.

### 4.2 Why multiple data compacts?

The project does not use a common dictionary to carry all phases, but rather to define different data compacts for different processing stages.

The main conversion relationships are:

```text
SourceTextSegment
→ DocumentChunk
→ EmbeddedChunk
→ Chroma Records
→ RetrievedChunk
→ Citation
```

#### `SourceTextSegment`

is a location-based text section that Loader deciphers from the original file.

It saves:

- Text content;
- Page number or chapter;
- Position in the original document.

Its function is to retain the original document structure, not to specify the final Chunk size.

#### `DocumentChunk`

is a block of text that is to be quantified through the Chunker split.

It adds:

- `document_id`；
- `chunk_id`；
- `chunk_index`；
- `content_hash`；
- `source_file`；
- `page_or_section`。

#### `EmbeddedChunk`

Means you've got Embedding vectors for Chunk.

It also includes:

- Chunk text and source;
- Embedding Float Vector.

#### `RetrievedChunk`

is evidence of a return from the vector library in a query.

It adds:

- Retrieving ranking `rank`;
- Similarity `similarity`.

#### `Citation`

indicates that the reference to the API caller is finally returned.

It consists of:

- (a) File name;
- Page number or chapter;
- Chunk ID；
- (a) Ranking;
- Similarity;
- Original Snippets.

The definition of a different contract for each stage would prevent downstream codes from relying on fields that do not exist upstream, and allow testing to accurately validate the input and output of each step.

---

## 5. Project experience with document resolution and cut

### 5.1 Loader and Chunker separated

Loader answers:

```text
What's the text in the original file?
Which page or section did these words come from?
```

Chunker answers:

```text
How many search units should these words be cut?
How many overlaps do neighbouring Chunk need to be retained?
```

After the separation:

- You can replace the split parameters without modifying the PDF resolution;
- New document formats could be added without modifying search logic;
- (b) Is capable of independently testing the resolution and the cut;
- The original page number or chapter is not lost by cutting.

### 5.2 Comparison of cut parameters

This week, two sets of parameters were compared:

```text
chunk_size=450，overlap=80
chunk_size=800，overlap=120
```

In the early Markdown preview, the two sets of parameters look exactly the same. The problem is not that the cut is really the same, but the preview function shows only the same beginning, covering up the difference between the end of the Chunk and the boundary.

Follow-up improvement preview:

- Increase in visible length;
- (b) Mark the beginning and end of each;
- Avoids the appearance of discontinuity at the end of the line as a continuous text.

This suggests that the assessment tool itself needs to be validated. An unclear visualized report may give developers an erroneous judgement about the actual data.

Final intake by:

```text
chunk_size=800
overlap=120
```

Fixed character windows are easy to understand and reproduce, but they do not recognize tables, lists and semantic paragraph boundaries.

### 5.3 PDF Rehabilitation of English spaces

Some English PDF extracts with characters or word adhesion, effects:

- Manual reading;
- Keyword matching;
- Embedding semantics;
- _Other Organiser

The project added inspirational English spaces to be restored and smoke proof was performed using the Cognex manual.

After repair, `robot_knowledge_v4` was created and re-ingestioned. V4 is the same as the Recall indicator for V3, but the body readability and quality of references have improved.

The quality of the resolution affects, in addition to the rate of recall, the readability of the evidence that eventually appears to the user.

---

## 6. Embedding and Protocol design experience

### 6.1 Why Embedding and LLM are separately configured

Embedding Model is responsible for:

```text
Text → Numerical vector
```

Generating LLM is responsible for:

```text
Problem + Evidence → Natural Language Answer
```

The two have different responsibilities and may also come from different service providers, and are therefore assigned independent:

- API Key；
- Base URL；
- Model；
- Timeout and error processing.

Embedding model determines vector space. The document vector and the query vector must be from the same Embedding model in the same Chroma Collection.

Even if the two models output the same dimensions, their vectors cannot be considered comparable.

### Value of 6.2 `Protocol`

Project use `EmbeddingProvider` 、 `VectorStore` 、 `RetrievalStore` 、 `KnowledgeAnswerProvider` Wait. `Protocol` Defines the capacity compact.

Business, for example, only needs to know that an object has:

```text
embed_texts()
embed_query()
retrieve()
generate_answer()
```

It does not need to rely on specific categories OpenAI SDK or ChromaDB.

So, when actually running, you can inject:

```text
EmbeddingService
ChromaVectorStore
OpenAIKnowledgeAnswerProvider
```

The test can be filled with:

```text
FakeEmbeddingProvider
FakeRetrievalStore
FakeKnowledgeAnswerProvider
```

This reflects reliance on inversion:

```text
Business organization relies on abstract capabilities
Instead of relying on specific external libraries.
```

`Protocol` is not intended to be responsible for the implementation of quantitative targets. It only provides “what attributes and methods must be present for the object to be imported”.

### 6.3 Embedding Service Error

Embedding API returns `404` when running the real baseline.

Such errors indicate that:

- The Python call chain has successfully reached upstream;
- Local batch processing and helipading are not necessarily problematic;
- More likely, the Base URL, interface path or service provider compatibility is incorrect.

This check helps to distinguish between:

```text
Business logic error
and
External service configuration error
```

---

## 7. Document heavy, delayed writing with ChromaDB

### 7.1 Stable Document ID

System bytes for original file SHA-256:

```text
Document contents
→ SHA-256
→ document_id
```

The same file will receive the same `document_id` content, not related to the file's storage path.

This would allow:

- (b) Detection of duplicate uploads;
- Stabilizing associated documents and chunk;
- (a) Delete all vectors by document;
- Duplicate calls for Embedding services.

### 7.2 Chunk ID

Chunk ID usage:

```text
{document_id}:{chunk_index}
```

For example:

```text
6115fd...32f8:000015
```

At the same time, it states:

- Which document does the Chung belong to;
- Orders of Chunk in this document;
- The ID remains stable when the same document is retaken.

### 7.3 Why late writing?

Ingestion process completed:

```text
Verify
→ Parsing
→ Severation
→ Embedding All
```

Last written in ChromaDB.

This way, if a batch of Embedding failed, a document marked as `ready` will not be left in advance, but is actually missing part of the Chunk.

This is an application level delayed writing policy, but it is not equivalent to database services. The current ChromaDB operation still lacks complete cross-step services and automatic rollback.

### 7.4 Why ChromaDB?

ChromaDB is also saved:

```text
chunk_id
Embedding Vector
Chunk Body
Source metadata
```

The current project does not require a separate deployment relationship database to enable:

- Retrieval Top-K;
- Can not open message
- Filter and remove on `document_id`;
- Lists document statistics;
- (b) Retain retroactivity metadata.

It is appropriate for the current single machine learning project, but does not yet have the required competence segregation, high-availability, complex services and large-scale expansion capabilities for the production system.

---

## 8. The iterative process of retrieval assessment

The search indicators have undergone four main versions:

| Version | Recall@1 | Recall@3 | Unresolved wrong recall rate | Phase conclusions |
|---|---:|---:|---:|---|
| V1 | 0.300 | 0.600 | 0.000 | Initial sustainable search is below target |
| V2 | 0.450 | 0.750 | 0.000 | Improved but still outstanding 0.80 |
| V3 | 0.600 | 0.850 | 0.000 | We're on target for the week. |
| V4 | 0.600 | 0.850 | 0.000 | Maintenance of improved indicators in English |

This process proves that the RAG should not be declared completed after its first operation, but should be continuously checked through a failure example:

- (a) Whether correct evidence exists in the language;
- Loader corrects the analysis;
- Whether `page_or_section` is correct;
- Whether Chunk overwrites the full answer;
- (a) Whether the language of Query and the evidence is consistent;
- The correct evidence is in the third place;
- Failures come from searching, or from evaluation matching.

For example, the second expected file for q013 was once in fifth place. The Top-10 report shows that it was not completely unretributed, but did not enter Top-3.

Finally, q013 recalled two anticipated evidence in Top-3.

---

## 9. Current Three Main Search Failed Cases

V4 still has q007, q008, q009 and three possible answers to questions that are not expected in life at Top-3.

### 9.1 q007: Cognex cable away from high-voltage lines

The correct evidence is:

```text
ID_reader_COGNEX_Reference_Manual_DM260.pdf
page: 56
section: Precautions
```

Actual Top-3 returned other pages in the same manual.

This means that vector search identifies product themes, but does not accurately target specific security concerns.

Possible improvements:

- Query Rewrite;
- Increase the search weight for titles and chapter names;
- (b) A combination of keywords and vectors;
- Use Reranker to sort the candidate chunk in second place.

### 9.2 q008: Victron charged temperature range

The correct evidence is on page 4 of the manual, but the search results are more favourable to the battery content on page 22.

Both contain synonyms such as “temperature” and “charging”, resulting in vector retrieval returning to Chunk, whose subject matter is similar but which does not fully answer the question.

This indicates that:

```text
Semantic Related
Not equal to
Complete evidence.
```

Possible improvements:

- (a) Include numerical ranges and units in keyword retrieval;
- (a) Extending the query on “charge and discharge” in the question;
- Use Reranker to determine which Chunk can fully cover two temperature ranges.

### 9.3 q009: ERR-SAF-1002 Security screening steps

The correct evidence is in the case of a failure statement, but the actual results are more oriented towards stop-test protocols and general safety standards.

Yeah. `ERR-SAF-1002` Such precise identifiers may be less reliable than keyword searches in pure vectors.

Possible improvements:

- (a) A precise match of the failure code;
- Structuralized identifiers for identifying problems;
- BM25 integration with vector search;
- Raise the Chunk ranking for failure code.

Together, these three failures show that the next step is not to continue to modify Prompt, but to add mixed search and reranker.

---

## 10. Similar evidence gate and refusal strategy

Online question-and-answer session:

```text
Top- 1 Similarity < 0.60
→ Direct refusal
→ citations=[]
→ abstained=true
→ Do not call LLM
```

The benefits of such design are:

- (b) To prevent the free completion of models without evidence;
- Reduce LLM call costs;
- Reducing delays in the absence of answers;
- The results of the refusal are stable and can be tested.

When the higher threshold was initially used, only 1 out of 16 answers were released.

Align the online threshold to `0.60`:

```text
Number of questions answered: 1 → 7
No answer correctly declined: 4/4
```

The threshold is not as low as possible. The continued lowering of the threshold, while possibly allowing more answers to questions, will also gradually move into a similar distribution of questions without answers and increase the risk of erroneous answers.

The current threshold is:

```text
Based on current text
+ Current Embedding Model
+ Current Gold Question
```

The engineering discounts obtained were not common constants that could be reused for all projects.

---

## 11. Why must the reference be coded?

Each piece of evidence received by the model is assigned a temporary number:

```text
E1
E2
E3
```

The model only allows returns:

```json
{
  "answer": "回答正文",
  "used_evidence_ids": ["E1", "E2"]
}
```

Python executes:

```text
E1
→ Retrieved Chunk for the current search for real rank = 1
→ source_file
→ page_or_section
→ chunk_id
→ similarity
→ excerpt
```

This division of responsibilities:

```text
The model determines which of the evidence provided was used
Python determines the true identity of these evidences.
```

If the model returns the `E99` that does not exist, the system will reject the output instead of generating a false reference.

This is more reliable than requiring the model to directly output file names and page numbers, since both file names, page numbers and Chunk IDs are derived from the results of a search that the program has verified.

---

## 12. Reference Assessment and Prompt fix

### 12.1 First round of reference assessments

Results of the first round under the higher threshold:

```text
Practical answers to answerable questions: 1/16
Quoting correct rate: 1.000
Reference coverage: 0.050
No answer correct rejection rate: 1.000
```

The system is very cautious, but over-rejected.

### 12.2 Threshold adjusted to 0.60

Adjusted to:

```text
Practical answers to answerable questions: 7/16
Quoting correct rate: 0.889
Reference coverage: 0.400
Full evidence response rate: 0.375
No answer correct rejection rate: 1.000
```

The decline in the correct rate of reference is not a sudden deterioration of the search, but rather a return of a repeat or unnecessary reference to one of the answers.

### 12.3 Minimum Adequate Evidence Rule

Prompt then adds:

```text
Used_evidence_ids must be the minimum sufficient body of evidence necessary to support the answer.
If a single piece of evidence is complete to support the answer, no longer can it be chosen to provide evidence that only repeats the background.
```

After repair:

```text
Quoting correct rate: 0.889 → 1.000
Full evidence response rate: 0.375 → 0.438
```

This suggests that Prompt optimization must be directed at specific failed behaviour observed, rather than an undefended stacking rule.

---

## 13. Status qualifier and security constraints

### 13.1 Missing from the “opinion draft”

The test found that the model might answer the scope of application on the basis of correct evidence, leaving out the document's status of “excerpt for comment”.

This is not a minor drafting issue in terms of engineering. The omission of the status limit may allow users to interpret documents that have not yet been officially issued as official standards in force.

Prompt therefore adds the status qualification rule, requiring that:

- (b) A draft of comments;
- Draft;
- Recommendations;
- Possible;
- Not yet confirmed.

At the same time, automated testing is added to ensure that Systems Prompt explicitly requires `answer` to retain these qualifications.

### 13.2 q014 Security prohibitions omitted

In contrast to the standalone LLM, RAG's initial reply, while using two correct pieces of evidence, omitted:

```text
No real blockages for testing.
```

This indicates that:

```text
Quote Correct
It doesn't necessarily mean that the answer is complete and secure.
```

Prompt then adds security binding rules requiring that reservations be directly related to the issue:

- Prohibition;
- Not;
- Not permitted;
- It must be;
- It's just that.
- Operating preconditions;
- value threshold.

V2 Answers explicitly include:

```text
No real blockages for testing purposes.
```

This restoration suggests that for robotic safety scenarios, the model cannot simply retain “what to do” but must also retain:

```text
What can't do?
Under what conditions?
What threshold cannot be exceeded?
```

---

## 14. RAG versus standalone LLM against the conclusions of the experiment

Use the same generation model and the same problem against the experiment, but enter different:

### RAG Group

```text
Problem
+ Search for evidence
+ Quote constraints
```

### Standalone llm Group

```text
Only the problem.
+ Answer the rules carefully.
```

q014 for:

- The RAG found evidence specific to two projects;
- Answers can be traced to real documents and chapters;
- Prompt maintains a security ban that cannot be physically blocked after repair;
- standalone LLM gives generic temperature and restoration conditions that are not supported by the current knowledge base;
- Standalone llm has also made a real blockage proposal that conflicts with the test protocol.

q020 for:

- RAG refuses to answer directly at the same level of evidence gate;
- No generation LLM called;
- returns empty references;
- Standalone llm, while also cautiously refusing to answer, produced a model call.

The experiment shows that the main value of RAG is not to make sure that the model answers forever, but to:

- Provide project evidence to answer the question;
- Limiting the scope of knowledge;
- Providing traceability sources;
- (a) A definitive refusal to respond when insufficient evidence is available;
- Avoid unnecessary model calls.

A single control experiment may be affected by the randomity of the model and network delay, and therefore can only serve as a case analysis and cannot substitute for a volume assessment.

---

## 15. API Layer and One Cycle Import Problem

The knowledge base interface continues:

```text
Router
→ Schema
→ Service
→ ExternalProvider or Stoe
```

Levels of responsibility:

### Router

Responsible for:

- HTTP path;
- Parameters for the request;
- Status code;
- FastAPI Reliance on injection;
- Give the request to Service.

### Schema

Responsible for:

- Data structure for requests and responses;
- Pydantic Validation
- OpenAPI Schema；
- Prevent illegal data from entering the business hierarchy.

### Service

Responsible for:

- (a) Document ingestion;
- Query to quantify;
- Top-K Retrieval;
- Similarity evidence gate;
- LLM call;
- The Citation Map.

The Router code was erroneously placed in the Schema file during development, resulting in:

```text
dependencies
→ document_catalog
→ schemas.knowledge
→ dependencies
```

Form a circular import and appear:

```text
partially initialized module
```

This was followed by the presence of `@router.post(...)` in the Schema file:

```text
NameError: name 'router' is not defined
```

This question note indicates that the hierarchy is not only for the purpose of completing the catalogue, but also for determining the direction of the module dependence.

The right direction should maintain:

```text
Router can rely on Schema and Service.
Service can rely on Schema or a field contract.
Schema should not reverse import Router or Despentencies
```

---

## 16. PowerShell debugging experience with HTTP

Two types of terminal problems were encountered this week.

### 16.1 JSON Question of quotation marks

When PowerShell is used `curl.exe`, the JSON string is analyzed through the PowerShell and Curl layers, resulting in invalid receipt of JSON by the service, returning:

```text
422 Unprocessable Entity
JSON decode error
```

Follow-up:

```text
PowerShell Object
→ ConvertTo-Json
→ UTF- 8 bytes
→ Invoke-WebRequest
```

The problem of manual conversion has been reduced.

### 16.2 UTF-8, response disorder

The service actually returned correct UTF-8 JSON, but the old Windows PowerShell's `Invoke-WebRequest` miscoded the response body, resulting in Chinese being shown as a bad code.

It's not about FastAPI or LLM creating a hyphen, but it's about how the client decodes.

The solution is:

```text
RawContentStream gets original bytes
→ Encoding.UTF8.GetString()
→ ConvertFrom-Json
```

This question helps to distinguish between three layers of code:

- Request to send the code;
- HTTP response bytes;
- Terminal display and client decoding.

---

## 17. Automation testing strategy

Final Full Volume Test Results:

```text
Test nodes: 244
Adopted: 244
Failed: 0
```

Automation test does not access:

- Real LLM;
- Real Embedding;
- External networks;
- Official, Croma Collection;
- Real API Key.

Main test double:

### Fake

Fake is a manageable simplification.

For example, Fake Embedding can return a fixed vector by input to verify:

- Enter order;
- Batch processing;
- Cache;
- Query Embedding；
- Document ingestion.

### Mock

Mock is used to record and control the interaction of objects, such as authentication:

- A method has been transferred several times;
- What are the call parameters;
- Whether `await` has been implemented;
- Whether or not the upstream timeout is converted to business anomalies.

### Temporary resources

`tmp_path` and temporary Chroma Collaction are used to avoid testing official data on pollution.

### Reliance Overwrite

`dependency_overrides` of FastAPI is used to replace real Service with Fake Service, thus certifying only:

```text
HTTP Request
→ Router
→ Schema
→ Service Contract
→ HTTP Response
```

This design allows the test results to be repeated, stable at a steady rate and avoids testing for API costs.

---

## 18. The most important meeting this week.

### 18.1 RAG is not a single LLM call

A reliable RAG actually consists of several subsystems:

```text
Language governance
+ Document Parsing
+ Chunk Design
+ Embedding
+ Vector Storage
+ Search
+ Threshold gate control
+ Prompt
+ Output Verify
+ Reference Map
+ Evaluation
```

Any one of these steps is wrong and the final answer may be unreliable.

### 18.2 Retrieving, quoting and answering must be assessed separately

Three layers of quality cannot be mixed into one indicator:

#### Retrieving quality

Whether the correct evidence entered Top-K.

#### Quote quality

Whether the return reference is correct or does it cover the evidence required.

#### Answer Quality

The answer is whether the facts, qualifications and security rules in the evidence are preserved in their entirety.

This week, q014 says:

```text
Correctly retrieved
+ Quote Correct
≠
The answer must be complete.
```

### 18.3 Prompt cannot repair missing evidence

If the correct evidence does not enter Top-K, Prompt cannot restore it in vain.

The order of optimization should therefore be:

```text
Check the text first.
→ Check the resolution again.
→ Check the cut again.
→ Optimise the search
→ Final Adjustment
```

### 18.4. Refusal is the right result.

The most reliable behaviour of the system is not to produce a reasonable answer to the lack of answers to the knowledge base, but to:

```text
The knowledge base does not have sufficient evidence to answer this question.
```

In engineering questions and answers and security scenes, reliable refusals are often more valuable than unconstrained answers.

### 18.5 Quoting does not mean automatic execution.

Even if the answer is correctly quoted, it cannot be directly driven by:

- (b) Urgent halts;
- (b) Electrical blockage;
- (b) Electric plug-ins;
- Parameters are written;
- (b) Solidware upgrades;
- Robotic motion.

The current system is a knowledge support layer, not a security controller.

---

## 19. Current technical debt and constraints

### Document Processing

- No OCR;
- Scanning version PDF cannot be reliably extracted;
- Limited support for complex tables and multi-bar layouts;
- The English space recovery is an inspirational rule;
- Fixed character window does not recognize the full semantic structure.

### Search

- Vector search only;
- No BM25 or keyword search;
- No Query Rewrite;
- No Reranker;
- Insufficient weights for failure codes and precise models;
- Whether or not to let go is based on the similarity of Top-1.

### Generate and evaluate

- Prompt is still soft bound;
- (a) The current main evaluation reference, not a full semantic rating of the answer;
- No security qualifiers are automatically detected;
- A single LLM experiment will be influenced by randomity.

### Data and services

- ChromaDB is a local single memory;
- No complete services and rollbacks;
- Document ingestion is a synchronized request;
- No back-office task queues;
- No document version management;
- (b) Lack of certification, authorization and segregation of tenants;
- There are no speed limits and no audit system.

### Product range

In accordance with the mandate boundaries of this week, at present it is not achieved:

- (a) The official frontend;
- Agent；
- Multi-user systems;
- Production deployment;
- Real-time telemetry access;
- Knowledge base query SSE.

These are not omissions this week, but are intentional controls. Make sure the evidence layer is correct and expands to more complex Agent and tools.

## 20. Summary of the week

The second week was not just a Demo that could upload documents and chat, but a RAG infrastructure with the following project characteristics:

- (a) The input language has its origin and redacted description;
- Documents, Chunk and references can be traced retroactively;
- Embedding and LLM segregation of duties;
- Vector data are sustainable;
- Repeating documents can be refused;
- (a) Refusal of a code for certainty in the absence of evidence;
- Quoting cannot be forged by a model;
- Quantified indicators for retrieval and citation;
- Testing does not depend on networks and real keys;
- Known failures are clearly documented;
- Structural constraints and follow-up direction are clear.

The most important engineering findings are:

```text
A reliable AI application is not just about whether the model can generate answers.
It's about controlling the answer to what.
In what circumstances?
How to prove the source,
And how the system is measured and discovered still doesn't know what.
```
