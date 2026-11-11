# Week 3 Retrieval Failure Log

Current progress: Original q001-q020 fully analysed
Status: Original 20 attribution completed; extension Gold to be created

## 1. Purpose

This paper attributes the results of Gold Question's search and online responses to a problem-by-issue error:

```text
Source
→ Document Parsing
→ Chunk cut.
→ Candidature for recall
→ Candidate Sorting
→ Similarity evidence gate
→ Generating Answers
```

This account is used to guide Week 3 keyword retrieval, integration sorting, query rewrite and evidence gate calibration.

The classification must be based on the search report, the location of Gold evidence and the results of an online response, not on subjective impressions.

## 2. Baseline source

This account currently uses the full text history of Week 2:

| Configure | Value |
|---|---|
| Collection | `robot_knowledge_v4` |
| Embedding Model | `embedding-3` |
| Gold Questions | 20 |
| Answerable questions | 16 |
| No answer to question | 4 |
| Top-K | 3 |
| Online gate control threshold | 0.60 |

Historical access indicators:

| Indicators | Result |
|---|---:|
| Recall@1 | 0.600 |
| Recall@3 | 0.850 |
| Top-1 Full recall | 8/16 |
| Top-3 Full recall | 13/16 |
| Unresolved wrong recall rate | 0.000 |

Historical online indicators:

| Indicators | Result |
|---|---:|
| Response rate to questions that can be answered | 7/16 |
| Full evidence response rate | 7/16 |
| Citation correctness | 1.000 |
| No answer correct response rate | 1.000 |

The complete indicator above is not recalculated from the public menu of the Chunk sample, but is derived from:

- `docs/en_retrieval-evaluation-summary.md`
- `docs/en_citation-evaluation-summary.md`

## 3. Definition of classification

### `baseline_success`

The correct evidence enters the required Top-K, and the online API answers correctly.

Success issues must remain on the table as a non-destructible return contrast for subsequent mixed searches.

### `parsing_missing`

There are answers in the original document, but there is no corresponding content in Loader's `SourceTextSegment` generation.

The determination must be accompanied by proof that:

1. The original document does contain the target content;
2. There is no target content in the analysis.

Scan PDF without layers of text, complex tables, multi-column error or OCR missing may cause such problems.

### `chunk_boundary_break`

The target content has been deciphered, but after the split:

- Questions and answers were removed to unreasonable numbers of Chunk;
- The title is not linked to the body;
- (b) The values and conditions of application are separated;
- A single Chunk cannot produce complete evidence.

It's not just because the problem is, `multi_hop` can be judged to be a cut failure. The cross-document problem would have required multiple evidence.

### `identifier_or_error_code`

Problems include failure codes, product models or other structured identifiers, but pure vector retrieval does not give priority to the return of exact matches.

Typical forms include:

```text
ERR-SAF-1002
ERR-NET-4001
DataMan 260
TEST-NAV-001
```

Such identifiers have a strong literally significant meaning and are suitable for keywords or precise matching, and should not rely entirely on semantic vectors.

### `semantic_ambiguity`

The system finds the subject matter relevant, but it does not fully support the answer.

Common situations include:

- The error chapter is located when the English document is retrieved for Chinese questions;
- The charging temperature is confused with the operating temperature;
- The relevant product page presses the real parameter page;
- The general security statement overwhelms specific trouble management steps.

### `gate_rejection_with_evidence`

The correct evidence is complete into Top-K, but only Top-1 similarity is checked on the online portal, which ultimately refuses to call LLM.

The classification requires:

1. `fully_recalled_at_3=true`；
2. Top-1 Similarity below the online threshold;
3. The online result is a negative answer.

### `truly_unanswerable`

The information requested by the question is not in the current static knowledge base.

For example:

- (b) The real warehouse equipment model;
- Real-time robotic position;
- (a) Current power levels;
- (a) Current orders;
- Unproven manufacturer association.

The correct act is to refuse to answer and return to the empty quote.

### `pending_verification`

The current report is not sufficient to rule out the problem of decomposition or splitting, and the classification is used provisionally.

One reason must not be imposed in order to fill the bill as soon as possible.

## 4. Attribution

1. From upstream to downstream, select the first layer of proven failure.
2. The success of the resolution or cut cannot be judged solely on the basis of similarity.
3. The correct evidence entered in Top-3 to prove that the evidence had at least been deconstructed, cut, quantified and stored.
4. Top-3 There is no correct evidence to directly prove the missing and it is necessary to continue to check the existence of the target Chunk.
5. The refusal to answer the questions is the right result and is not a systemic failure.
6. Success continues to be used as a model for return and cannot be degraded after optimization.
7. One is recorded for one major category only, while other phenomena are included in “secondary observations”.

## 5. q001-q006 On-the-board accounts

| ID | Top-1 Similarity | Top-3 Complete evidence | Online results | Main categories | Secondary observation |
|---|---:|---|---|---|---|
| q001 | 0.569 | Yes. | Refusal | `gate_rejection_with_evidence` | Clear 1500 ms numeric evidence has entered Top-3, but Top-1 is below 0.60 |
| q002 | 0.640 | Yes. | Replyed | `baseline_success` | Percent threshold and charge-back correct recall |
| q003 | 0.619 | Yes. | Replyed | `baseline_success` | The English clean-up note was successfully recalled in Chinese |
| q004 | 0.612 | Yes. | Replyed | `baseline_success` | The English security requirement was successfully recalled in Chinese. |
| q005 | 0.748 | Yes. | Replyed | `baseline_success` | The correct evidence is not in Top-1 but enters Top-3; there is a ranking risk |
| q006 | 0.564 | Yes. | Refusal | `gate_rejection_with_evidence` | The fail code evidence is in Top-3, and the main failure now occurs at the evidence gate level. |

## 6. q001 Attribution

The problem requires that the communication interruption threshold and safety state in the test protocol be tested.

Historical results show that:

- Gold evidence entered Top-3;
- Top-1 was hit with an expected piece of evidence;
- Top-1 Similarity is 0.569;
- The threshold is 0.60 online;
- Online API finally refused to answer.

Thus, the correction, cut-off and candidate recall has been successful, with the earliest proven failure being evidence gate.

Main categories:

```text
gate_rejection_with_evidence
```

The next direction is not to directly lower the threshold, but to study:

- Whether the evidence gate should combine Top-K evidence;
- (a) Whether the value and keyword are precise enough to be used as a release signal;
- Whether or not to release q008 or other error search by relaxing the evidence gate.

## 7. q002 Attribution

The problem requires low power thresholds and behaviour to enter recharge queues.

The results are as follows:

- (a) The correct evidence entered Top-1;
- Top-1 Similarity is 0.640;
- above the 0.60 evidence gate threshold;
- Online API answers correctly.

Main categories:

```text
baseline_success
```

The topic should be retained as a retrogression sample for the issue of numerical thresholds. Follow-up hybrid searches cannot reduce their correct ranking.

## 8. q003 Attribution

Question in Chinese for clean-up methods in the DataMan 260 manual.

The results are as follows:

- The correct English evidence entered Top-1;
- Top-1 Similarity is 0.619;
- Online API answers correctly.

Main categories:

```text
baseline_success
```

The question proves that the current Embedding model already has some Chinese and English cross-linguistic search capability, but it does not infer that all English detail issues can be correctly located. q007 still requires a separate analysis.

## 9. q004 Attribution

Question in Chinese about the qualification requirements of the English Battery Manual.

The results are as follows:

- (a) The correct evidence entered Top-1;
- Top-1 Similarity is 0.612;
- Online API answers correctly.

Main categories:

```text
baseline_success
```

It will be used as a back-to-back check for English-language security requirements.

## 10. q005 Attribution

The question was asked about the scope of application of the draft industrial mobile robotic safety standard.

The results are as follows:

- The correct evidence is not ranked in Top-1;
- (a) The correct evidence entered Top-3;
- Top-1 Similarity is 0.748;
- Online API finally answered correctly;
- The reference to the “opinion draft” was retained.

Main categories:

```text
baseline_success
```

Secondary risks:

```text
top1_not_gold
```

This suggests a high Top-1 similarity does not guarantee that the first one is Gold evidence. The success of the current response depends on evidence that the answer level is able to select from Top-3 the real support for the answer.

## 11. q006 Attribution

The problem consists directly of the failure code `ERR-DRV-3005`, requiring trigger conditions and routing matters.

The results are as follows:

- The correct malfunction section enters Top-1 and Top-3;
- Top-3 Complete evidence;
- Top-1 Similarity is 0.564;
- The online portal refused to answer.

Main categories:

```text
gate_rejection_with_evidence
```

Although the problem contains a wrong code, the right chapter has been recalled, so the main reason for the current online failure is not the wrong code, but the rules of evidence gate.

Subsequent hybrid searches could still be used to provide a more stable sorting or evidence gate signal for the subject in precise failure codes.

## 12. q001-q006 Summary

Q001 and Q006 are:

| Classification | Number |
|---|---:|
| `baseline_success` | 4 |
| `gate_rejection_with_evidence` | 2 |
| Other types of failure | 0 |

The current batch has not yet been confirmed:

- `parsing_missing`
- `chunk_boundary_break`
- `identifier_or_error_code`
- `semantic_ambiguity`
- `truly_unanswerable`

These types will continue to be determined in subsequent topics.

## 13. q007-q012 On-the-board accounts

| ID | Top-1 Similarity | Top-3 Complete evidence | Online results | Main categories | Secondary observation |
|---|---:|---|---|---|---|
| q007 | 0.537 | Yes | Refusal | `semantic_ambiguity` | Found the correct product manual, but did not put the correct Precautions page into Top-3 |
| q008 | 0.598 | Yes | Refusal | `semantic_ambiguity` | The relevant settings page in the same battery manual overwhelms the pages that actually contain the charger temperature range |
| q009 | 0.556 | Yes | Refusal | `identifier_or_error_code` | Generic security content overwhelms the malfunction chapter that matches the exact `ERR-SAF-1002`. |
| q010 | 0.530 | Yes. | Refusal | `gate_rejection_with_evidence` | The correct network failure section has entered Top-1, but similarity is below 0.60. |
| q011 | 0.681 | Yes. | Replyed | `baseline_success` | Repeats, cross-crossings and death-locks are all correctly recalled |
| q012 | 0.625 | Yes. | Replyed | `baseline_success` | 0 error workstations and at least 98 automatic completions of 100 |

## 14. q007 Attribution

Question in Chinese about electrical safety concerns in the English handbook DataMan 260.

The correct evidence is:

```text
ID_reader_COGNEX_Reference_Manual_DM260.pdf
page: 56
section: Precautions
```

Historical results show that:

- The search results were obtained from the correct manual DataMan 260;
- Top-3 returned other pages of the same manual;
- The correct `Precautions` page did not enter Top-3;
- Top-1 Similarity is 0.537;
- Online API refused because there was no acceptable evidence.

Week 2 has verified that the English text of this page can be extracted by PDF Loader, so there is no evidence that this is `parsing_missing`.

Main categories:

```text
semantic_ambiguity
```

Secondary causes:

```text
cross_language_detail_ranking
```

This is an important contrast with q003:

- q003: Successful recall of the English Cleaning Section for Chinese cleaning;
- q007: The Chinese electrical security problem is located only in the correct manual and not in the correct chapter.

This suggests that the current Embedding model has some cross-linguistic skills, but finding the right product theme does not mean " finding the right detail chapter."

Follow-up candidates:

- Renumber the Chinese query as a keyword;
- Retain `DataMan 260`;
- (b) Extensions of `high-current wiring`, `high-voltage power sources`, `precautions`;
- (a) Add chapter headings to the keyword index;
- The larger pool of candidates recalled in vectors is re-ordered.

## 15. q008 Attribution

Question requested to be answered at the same time:

- (b) The charge temperature range;
- The discharge temperature range.

The correct evidence is:

```text
victron energy_Manual_Lithium_Smart_Battery_en.pdf
page: 4
section: 1.2 Charge and discharge warnings
```

The historical results tend to favour the battery settings on page 22 of the manual.

The candidate for errors also includes the relevant synonyms of " cell" "temperature" "charging", but cannot answer two different temperature ranges in full.

Main categories:

```text
semantic_ambiguity
```

Secondary causes:

```text
numeric_range_confusion
```

This is a question note:

```text
Subject-related
Not equal to
Complete evidence.
```

Particular attention is required to:

```text
q008 Top- 1 Similarity = 0.598
Current evidence gate threshold = 0.600
```

The difference between the two is only 0.002. If the threshold is lowered directly to allow q001, q006, q010, the error evidence of q008 may also be released.

Therefore, it is not possible to attribute all of the refusals to excessive door-to-evidence gate.

Follow-up candidates:

- Renumber the question of Chinese as `charge temperature range` and `discharge temperature range`;
- Match keywords with `charge`, `discharge`, `temperature` and `range`;
- (b) Relayers check if candidates are covering both charge and discharge conditions;
- Increased integrity judgement for candidates with units, floors and floors and scope structures.

## 16. q009 Attribution

Question includes precise failure code:

```text
ERR-SAF-1002
```

The correct evidence is:

```text
仓储机器人故障说明.txt
section: ERR-SAF-1002
```

The historical results are more biased:

- (b) The urgent suspension of testing protocols;
- General safety standards;
- Other semantics relate to security content.

These elements are related to the subject of security, but cannot be a substitute for the routing and repositioning process in the specific malfunction code chapter.

Main categories:

```text
identifier_or_error_code
```

The problem is not " the model does not understand the security semantics at all ", but pure vector retrieval does not give sufficient sorting advantages to precise strings.

The error code has the following characteristics:

- Short string;
- Structure fixed;
- Each character has a different meaning;
- Semantic similarity is not a substitute for precise equivalence.

For example:

```text
ERR-SAF-1002
```

The following cannot be replaced by security-related:

```text
ERR-NAV-2001
ERR-DRV-3005
General Emergency Rules
```

Follow-up candidates:

- (a) The extraction of the failure code using a regular expression;
- (b) Performing accurate matching of standardized failure codes;
- Add keyword or BM25 search;
- Combining the exact target of the Chunk with the vector candidate;
- Sets clear, testable ranking rules for precise failure code hits.

## 17. q010 Attribution

The problem requires trigger conditions and common causes for `ERR-NET-4001`.

The results are as follows:

- The correct malfunction section enters Top-1;
- Top-3 Full recall;
- Top-1 Similarity is 0.530;
- The threshold is 0.60 online;
- Online API finally refused to answer.

Main categories:

```text
gate_rejection_with_evidence
```

The difference with q009 is:

| Problem | Whether or not the correct evidence has entered Top-3 | The first failed layer |
|---|---|---|
| q009 | Yes | Can not get folder:%s:%s |
| q010 | Yes, and at Top-1. | Evidence gate. |

Both issues contain the wrong code, but cannot be classified as such using the same failure.

Follow-up studies could examine:

- (a) Whether the precise failure code can be a door-controlled signal;
- Whether the evidence gate should examine both sources, keyword hits and Top-K evidence;
- The rule would not be wrongly released without an answer.

## 18. q011 attribution

The question calls for simultaneous recall:

- 20 consecutive rounds;
- Only one robot can enter first on each wheel;
- Another station stopped in the waiting area.
- No collisions, rubs or dead locks;
- After the robot leaves, wait for the robot to continue.

The results are as follows:

- (a) The correct evidence entered Top-1;
- Top-1 Similarity is 0.681;
- Online API answers correctly.

Main categories:

```text
baseline_success
```

This issue should serve as a model for re-entry of the “multi-conditional acceptance rule”.

Subsequent rearrangements cannot retain only the value of “20 rounds” and lose the conditions for cross-cutting and death-locks.

## 19. q012 Attribution

The problem requires a distinction between two conditions for acceptance and acceptance:

```text
The number of error stations must be 0
At least 98 of the 100 consecutive missions were automated.
```

The results are as follows:

- (a) The correct evidence entered Top-1;
- Top-1 Similarity is 0.625;
- Online API answers correctly.

Main categories:

```text
baseline_success
```

This topic should serve as a model for regression where “multiple numerical bindings are in the same evidence”.

It contrasts with q008:

- q012 is correctly recalled;
- q008 does not correctly recall the charge temperature range in Chinese and English.

Therefore, q008 cannot simply be attributed to the " vector model does not support values" but rather to ranking problems that are caused by a combination of language, similar themes and evidentiary integrity.

## 20. q001-q012 Cumulative result

| Classification | Number | Response issues |
|---|---:|---|
| `baseline_success` | 6 | q002、q003、q004、q005、q011、q012 |
| `gate_rejection_with_evidence` | 3 | q001、q006、q010 |
| `semantic_ambiguity` | 2 | q007、q008 |
| `identifier_or_error_code` | 1 | q009 |
| `parsing_missing` | 0 | - |
| `chunk_boundary_break` | 0 | - |
| `truly_unanswerable` | 0 | - |
| `pending_verification` | 0 | - |

The most important conclusions of the current project are:

1. q007-q009 cannot be resolved by simply lowering the threshold for evidence gate because the correct evidence does not enter Top-3.
2. The correct evidence for q001, q006, q010 has entered Top-3 and failed at the evidence gate level.
3. q008 is close to 0.60, and a precipitous lowering of the threshold may add to the wrong answer.
4. Follow-up must be accompanied by improvements in candidate recall, sequencing and evidence gate, rather than the optimization of only one of them.

## 21. q013-q020 On-the-board accounts

| ID | Top-1 Similarity | Top-3 Complete evidence | Online results | Main categories | Secondary observation |
|---|---:|---|---|---|---|
| q013 | 0.581 | Yes. | Refusal | `gate_rejection_with_evidence` | Both cross-documentary evidences have entered Top-3, but evidence gate only checks Top-1 |
| q014 | 0.623 | Yes. | Replyed | `baseline_success` | Both evidences are fully recalled; negative security constraints were omitted from the generation layer |
| q015 | 0.589 | Yes. | Refusal | `gate_rejection_with_evidence` | Network failure causes and recovery processes are in progress Top-3 |
| q016 | 0.589 | Yes. | Refusal | `gate_rejection_with_evidence` | The English cleaning instructions and the Chinese regression test protocols are in place Top-3. |
| q017 | 0.482 | Not applicable | Correctly declined. | `truly_unanswerable` | The knowledge base does not have a real Kyoto robot laser radar brand or model. |
| q018 | 0.480 | Not applicable | Correctly declined. | `truly_unanswerable` | The two types of entities exist, but there is no evidence that the Kyoto robots use the Victron battery |
| q019 | 0.527 | Not applicable | Correctly declined. | `truly_unanswerable` | The knowledge base is only simulating distance and cannot be impersonating real production parameters. |
| q020 | 0.519 | Not applicable | Correctly declined. | `truly_unanswerable` | The static knowledge base has no real-time location, electricity and order status |

## 22. q013 attributions

q013 is a cross-document multiple jump problem that requires simultaneous recall:

1. (a) Continuous recognition of failure conditions and state of protection in the 2-D code in the test protocol;
2. `ERR-NAV-2001` recovery of positioning process in the malfunction statement.

Finally, V4 shows that:

- Both Gold evidence entered Top-3;
- Top-3 Full recall;
- Top-1 Similarity is 0.581;
- Online evidence gate threshold of 0.60;
- Online API finally refused to answer.

Main categories:

```text
gate_rejection_with_evidence
```

In the early assessment, the second expected document had been placed fifth. Follow-up optimization allows it to enter Top-3, but the sequencing remains fragile.

Secondary risks:

```text
multi_hop_ranking_fragility
```

The current ultimate failure does not occur at the level of decomposition, cut-off or candidate recall, as two complete pieces of evidence have entered Top-3.

The failure occurred:

```text
Top- 1 Similarity 0.581
    ↓
Below the threshold for evidence gate 0.60
    ↓
LLM didn't have a chance to see two evidences that had been recalled.
```

Follow-up directions:

- Evidence gate cannot only look at Top-1;
- A combination of Top-K evidence could be considered;
- The failure code and the test number may be considered for a key hit;
- Multiple irrelevant and weak evidences must be prevented from being wrongly grouped as grounds for release.

## 23. q014 Attribution

q014 needs a combination of:

1. Testing safety failure injection methods in protocols;
2. Trigger conditions, protective behaviour and recovery inspections in the statement of failure.

The results are as follows:

- Top-1 expected evidence of impact;
- Top-3 to find both anticipated evidence;
- Top-1 Similarity is 0.623;
- above the online threshold of 0.60;
- Online API answers correctly and returns two references.

Main categories:

```text
baseline_success
```

The issue also exposed a generation layer risk: the first version of the answer, while correctly quoted, omitted the negative security constraint “not allowing high temperatures to be created through a genuine blackout mechanism”.

It is therefore important to distinguish between:

```text
Retrieval and citation correct
Not equal to
Answer the semantics must be complete.
```

The answer to this question is not to retain the full security boundary until the rules on reservations, prohibitions, mandatoryity, preconditions and numerical thresholds have been added.

Secondary observation:

```text
generation_safety_qualifier_risk
```

q014 shall be simultaneously:

- The question of multiple and complete recall;
- The issue of the return of negative security constraints;
- The question of return is quoted at the minimum full.

## 24. q015 Attribution

q015 requests grouping:

1. `ERR-NET-4001` trigger thresholds and common causes;
2. Safe parking and network recovery processing requirements in test protocols.

The results are as follows:

- Both anticipated evidence entered Top-3;
- Top-3 Full recall;
- Top-1 Similarity is 0.589;
- Online API refused to answer because it was below 0.60.

Main categories:

```text
gate_rejection_with_evidence
```

The subject proves that the current candidate level has been able to complete the cross-document grouping, but the evidence gate did not take advantage of this result.

The follow-up evidence gate programme requires judgement:

- Whether the two candidates will cover the different parts of the issue, respectively;
- Whether there is an accurate error code or test number;
- Whether or not candidates really complement each other, rather than repeat or contradict each other.

Cannot be used simply:

```text
Top- 3 Similarity & Excess
→ Automatically Release
```

The combination of weak and relevant evidence can also give rise to false high marks.

## 25. q016 attributions

q016 is a cross-language, cross-documentation problem that needs to be grouped:

1. DataMan 260 Portal Protection Cleaning Requirements in English Manual;
2. 100 read re-entry tests and test conditions in the Chinese test protocol.

The results are as follows:

- Both anticipated evidence entered Top-3;
- Top-3 Full recall;
- Top-1 Similarity is 0.589;
- Online API finally refused to answer.

Main categories:

```text
gate_rejection_with_evidence
```

The difference between it and q007 is:

| Problem | Whether or not the correct evidence has entered Top-3 | Main Failed Layer |
|---|---|---|
| q007 | Yes | Can not get folder:%s:%s |
| q016 | Yes, both are in. | Evidence gate. |

Both issues were related to Chinese and English manuals, but could not be attributed to the failure of the various languages.

q016 also indicates that the single Top-1 similarity of the multiple problem does not represent the full quality of the evidence.

## 26. q017 Attribution

q017 Ask about laser radar brands and models used by robots in real warehouses in Kyoto.

The current knowledge base does not:

- A list of real Kyoto robotic hardware;
- Laser radar manufacturers;
- Laser radar model;
- Procurement or integration of information that can support the equipment relationship.

Main categories:

```text
truly_unanswerable
```

The results are as follows:

- Top-1 Similarity is 0.482;
- Below the online threshold;
- API returns `abstained=true`;
- Quoted list is empty.

That's the right thing to do.

The existence of other sensor manufacturers ' manuals in the knowledge base does not allow the system to assume that real robots use counterpart products.

## 27. q018 Attribution

q018 Asks whether the Tokyo Real Warehouse robot uses Victor Lithium Smart Battery, as well as specific batteries and BMS models.

The knowledge base is divided into the following:

- Storage robotic simulation scenario information;
- Victron Battery Open Manual.

However, there is no evidence of any of the following:

```text
Kyodong Real Warehouse Robot.
Use
Victron Lithium Smart Battery
```

Main categories:

```text
truly_unanswerable
```

This is an answerless question of “entity exists, but relationship does not exist”.

It is more dangerous than completely unrelated issues, as future mixed searches may be recalled separately:

- A Chunk with "Kyoto storage scene";
- A ‘Victron Battery’ of Chunk.

LLM cannot establish a user relationship on its own because of the simultaneous appearance of two pieces of evidence.

Subsequent structured responses must require direct evidentiary support for relationship conclusions.

## 28. q019 Attribution

q019 Ask about the speed reduction distance and protection distance of the barrier at the real production warehouse in Kyoto.

The self-compiled simulation protocol may include:

```text
1.2 m
0.5 m
```

These values, however, are clearly a learning project simulation and are not official production parameters in Kyoto.

Main categories:

```text
truly_unanswerable
```

This is not a question of “no relevant figures at all in the database”, but of:

```text
Numbers in database
But the scope of application of numbers does not support user questions.
```

The system must therefore understand at the same time:

- Values;
- Sources;
- (a) The nature of the document;
- (a) The objects to be applied;
- “Simulation” and the qualification of “real production”.

Top-1 Similarity is 0.527 and the current system correctly rejects the answer.

Future keyword search may be due to the fact that the word “trip” in Kyoto increases the ranking of the relevant Chunk, so that the issue must be confirmed before the commingled search is online.

## 29. q020 Attribution

q020 Ask for current running `robot-001`:

- Where are they located;
- (b) Residual power;
- What orders are in progress.

The current knowledge base contains only static documents and is not connected to:

- (b) Robotic telemetry;
- RCS；
- RMS；
- WMS；
- Order systems;
- Real-time database.

Main categories:

```text
truly_unanswerable
```

The continued addition of static statement documents does not provide a reliable answer to the " current state ".

The right structure should be:

```text
Static rules and manuals
→ RAG Knowledge Base

Real-time position, power and tasks
→ Telemetry or Operations System Tool Call
```

The long-term inclusion of a real-time state in the static vector bank would generate out-of-date data and would not be suitable as a reliable source for the current state.

Current results:

- Top-1 Similarity is 0.519;
- API correct refusal;
- `abstained=true`；
- Quote as empty.

## 30. Original 20 final classification

| Classification | Number | Response issues |
|---|---:|---|
| `baseline_success` | 7 | q002、q003、q004、q005、q011、q012、q014 |
| `gate_rejection_with_evidence` | 6 | q001、q006、q010、q013、q015、q016 |
| `semantic_ambiguity` | 2 | q007、q008 |
| `identifier_or_error_code` | 1 | q009 |
| `truly_unanswerable` | 4 | q017、q018、q019、q020 |
| `parsing_missing` | 0 | - |
| `chunk_boundary_break` | 0 | - |
| `pending_verification` | 0 | - |
| Total | 20 | q001–q020 |

## 31. Categorize corresponding project directions

| Classification | Current Title | Priority Project Directions |
|---|---|---|
| `baseline_success` | q002–q005、q011、q012、q014 | Maintaining a regression contrast to prevent optimal degradation |
| `gate_rejection_with_evidence` | q001、q006、q010、q013、q015、q016 | Design based on evidence gate of Top-K, precise hit and evidence coverage |
| `semantic_ambiguity` | q007、q008 | Query rewriting, chapter titles, expanded candidate lists, reordering |
| `identifier_or_error_code` | q009 | Standardization of identifiers, precise matching, keyword retrieval, integration sorting |
| `truly_unanswerable` | q017–q020 | Keep rejecting answers to prevent mixed search from introducing error release |

## 32. Original conclusion on question 20

The original Gold collection revealed two main problems:

### Failed to submit or sort

```text
q007
q008
q009
```

The correct evidence for these three questions did not enter Top-3 and cannot be repaired by adjusting Prompt.

### Evidence gate denied evidence.

```text
q001
q006
q010
q013
q015
q016
```

The complete evidence of the six issues has entered Top-3, but LLM has not obtained it.

At the same time, it is necessary to protect:

```text
q017
q018
q019
q020
```

These four questions are rightly rejected.

Therefore, the goal of Week 3 is not simply to increase the number of answers, but to:

```text
Increase in the rate of recall and release of available evidence
At the same time
No reduction in the correct rate of non-response to questions
```

The next phase of the expansion of the Gold set requires an increased focus:

- Error code and its case, space and hyphene variants;
- (a) Product types and their format variants;
- Positive or negative values and units;
- (b) The issue of the Chinese and English languages;
- Cross-document process;
- (a) Simulation values to the actual value boundary;
- Problems of physical existence but non-existence of relationship;
- Real-time status issues.

## 33. Expand Gold's pure vector baseline B

Before a hybrid search is achieved, the extended 28 track Gold questions are re-evaluated using the unmodified pure vector search.

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
| Total anticipated evidence | 30 |
| Top-K | 3 |
| Evidence gate reference threshold | 0.60 |

Overall indicators:

| Indicators | Result |
|---|---:|
| Recall@1 | 0.567 |
| Recall@3 | 0.767 |
| Top-1 Full recall questions | 10/23 |
| Top-3 Full recall questions | 16/23 |
| Unresolved Wrong recall Number | 0/5 |
| Unresolved wrong recall rate | 0.000 |

The original 20 sub-set still gets:

| Indicators | Result |
|---|---:|
| Recall@1 | 0.600 |
| Recall@3 | 0.850 |
| Top-1 Full recall | 8/16 |
| Top-3 Full recall | 13/16 |
| Unresolved wrong recall rate | 0.000 |

This is consistent with the historical results of Week 2, which proves that the new Collection successfully recreates the original pure vector baseline.

Add q021 to q028 subsets with the following results:

| Indicators | Result |
|---|---:|
| Number of questions answered | 7 |
| Number of questions not answered | 1 |
| Total anticipated evidence | 10 |
| Recall@1 | 0.500 |
| Recall@3 | 0.600 |
| Top-1 Full recall | 2/7 |
| Top-3 Full recall | 3/7 |
| Unresolved wrong recall rate | 0.000 |

Thus, the overall reduction in Recall is due to new difficulties rather than to the degradation of original search capabilities.

## 34. q021-q028 Issue-by-issue result

| ID | Top-1 Similarity | Top-3 Complete evidence | Main categories | Critical observations |
|---|---:|---|---|---|
| q021 | 0.539 | Yes. | `gate_rejection_with_evidence` | Lowercase space failure code still recalls the right chapter, but below the threshold for evidence gate |
| q022 | 0.598 | Yes | `semantic_ambiguity` | We found the failure note, we didn't find the network test protocol, but we called back the generic recovery section. |
| q023 | 0.496 | Yes. | `gate_rejection_with_evidence` | Correct Precautions page in Top-2, but Top-1 and evidence gate is not ideal |
| q024 | 0.581 | Yes | `semantic_ambiguity` | Still recall configuration temperature page, no charge alert page |
| q025 | 0.586 | Yes. | `gate_rejection_with_evidence` | The correct network test chapter is located at Top-1 but below the evidence gate threshold |
| q026 | 0.533 | Yes | `identifier_or_error_code` | We found the temperature test protocol, no `ERR-DRV-3005`, and erroneously recalled `ERR-SAF-1002`. |
| q027 | 0.506 | Yes | `semantic_ambiguity` | Two-dimensional regression tests were found, but the manual cleaning page was not entered Top-3. |
| q028 | 0.546 | Not applicable | `truly_unanswerable` | Found the product manual, but no proof of true workstation subtypes and serial numbers. |

## 35. q021 Attribution

q021 will put the standard failure code:

```text
ERR-SAF-1002
```

Replace the text with:

```text
err saf 1002
```

The results are as follows:

- (a) Correct failure statement to enter Top-1;
- Top-3 Full recall;
- Top-1 Similarity is 0.539;
- Below the current online evidence gate threshold of 0.60.

Main categories:

```text
gate_rejection_with_evidence
```

This indicates that the vector model is able to understand the variant in this query, but similarity is not sufficient to be controlled through the current door.

It contrasts with q009:

| Problem | Fault Code Format | Correct evidence position |
|---|---|---|
| q009 | `ERR-SAF-1002` | Not entered Top-3 |
| q021 | `err saf 1002` | Top-1 |

This result cannot be interpreted as “a better format than a standard format”, but rather as an indication that pure vector ranking is sensitive to the wording of the entire sentence and does not provide certainty identifier matching assurance.

This should be followed by the harmonization of the two formats into a single standard identifier, followed by the implementation of keyword retrieval and integration sequencing.

## 36. q022 Attribution

q022 Use underlined form:

```text
ERR_NET_4001
```

And ask for a simultaneous answer:

1. (a) Causes of failure;
2. How to continue the operation after the network has been restored.

The results are as follows:

- `ERR-NET-4001` failure statement entered Top-1;
- `TEST-NET-001` test protocol not entered Top-3;
- Top-3 The synonymally related but not equivalent general restoration section `TEST-REC-001` has emerged;
- Only one of the two anticipated evidences was found;
- The actual similarity of Top-1 is 0.598177.

Main categories:

```text
semantic_ambiguity
```

Secondary observation:

```text
identifier_variant
multi_hop_incomplete
```

The threshold cannot be lowered alone, as the necessary evidence of the problem is still incomplete.

The keyword search will need to be followed by the identification of:

```text
ERR_NET_4001
→ ERR-NET-4001
→ Network failure chapter
→ TEST-NET-001
```

The second evidence is also ranked higher by splitting or rearranging through queries.

## 37. q023 Attribution

q023 Using product abbreviations `DM260`, query the risk of cables approaching large current lines or high voltage power sources.

The results are as follows:

- Top-1 as manual `page: 48`;
- Correct `Precautions` evidence in Top-2;
- Top-3 fully recalled;
- Top-1 Similarity is 0.496;
- The current online portal will refuse to answer.

Main categories:

```text
gate_rejection_with_evidence
```

Secondary observation:

```text
cross_language_ranking_fragility
```

It contrasts with q007:

| Problem | Correct. `page: 56`, enter Top-3. |
|---|---|
| q007 | Yes |
| q023 | Yes, at Top-2. |

The slight change in the wording and type of aliases changed the ranking, suggesting that pure vector retrieval was unstable with respect to the cross-language evidence.

Follow-up needs

- Consider `DM260` and `DataMan 260` as aliases;
- Candidates for recall of English keywords `high-current`, `high-voltage`, `over-voltage`, `line noise`;
- Increase `page: 56` by sorting, rather than relying only on vector distance.

## 38. q024 Attribution

q024 Using specific temperature `-10°C`, it is required to determine whether chargeable or discharged.

The results are as follows:

- Correct `page: 4` not entered Top-3;
- Top-1 and Top-2 from `page: 22`;
- Top-3 from `page: 27`;
- Top-1 Similarity is 0.581.

Main categories:

```text
semantic_ambiguity
```

Like q008, it did not recall the correct recharge warning page, suggesting that the temperature range problem was a failure of a stable pure vector, rather than an accidental single formulation.

Subsequent keyword retrieval must be retained:

- Negative sign;
- Values;
- `°C` units;
- `charge`；
- `discharge`；
- `warning`。

At the same time, it is important to avoid miscalculating the configuration temperature to allow for chargeability simply because other pages also contain temperature figures.

## 39. q025 Attribution

q025 Inspection:

```text
The interruption of communications coincided with 1500 ms.
```

The failure trigger in the protocol has been met.

The results are as follows:

- Correct `TEST-NET-001` chapters entered Top-1;
- Top-3 Full recall;
- Top-1 Similarity is 0.586;
- Below the evidence gate threshold of 0.60.

Main categories:

```text
gate_rejection_with_evidence
```

The correct evidence has been found, but subsequent generation must be retained:

```text
It's just right there.
```

And:

```text
More than
```

Boundary differences between.

This is also the retrogression of numerically comparable semantics. It cannot only check whether a string `1500 ms` appears.

## 40. q026 Attribution

q026 will:

```text
ERR-DRV-3005
```

Replace the text with:

```text
err drv 3005
```

A combination of malfunctions is also required to inject protocols and failure statements.

The results are as follows:

- `TEST-DRV-001` into Top-1;
- `ERR-DRV-3005` did not enter Top-3;
- Top-3 appears instead of `ERR-SAF-1002`;
- Only one of the two anticipated evidences was found;
- Top-1 Similarity is 0.533.

Main categories:

```text
identifier_or_error_code
```

Secondary observation:

```text
multi_hop_incomplete
```

This means that pure vector search does not guarantee that errors in the structure of a character are correctly distinguished.

The subsequent use of certainty is required to integrate and accurately match, not to allow LLM or Embedding to guess the wrong code.

## 41. q027 Attribution

q027 requests grouping:

1. DM260 camera protection cap cleaning requirements;
2. Workstation 2D to read receiving and inspection.

The results are as follows:

- `TEST-ID-001` into Top-1;
- Manual `page: 55` not entered Top-3;
- Top-2 and Top-3 are manuals `page: 50` and `page: 51`;
- Only one of the two anticipated evidences was found;
- Top-1 Similarity is 0.506.

Main categories:

```text
semantic_ambiguity
```

Secondary observation:

```text
cross_language_ranking_fragility
multi_hop_incomplete
```

It contrasts with q016:

| Problem | Top-3 Complete evidence |
|---|---|
| q016 | Yes. |
| q027 | Yes |

The synonyms of the two issues are similar, but the different rankings resulting from changes in product abbreviations and wording suggest that the multi-linguistic recall is still unstable.

## 42. q028 Attribution

q028 Ask about the use of real production stations in Kyoto:

- DataMan 260 specific sub-models;
- Device serial number.

The knowledge base has only a series of DataMan 260 public handbooks, none of which:

- List of real work station equipment in Kyoto;
- The relationship of the workstation to the specific sub-model;
- Device serial number.

The results are as follows:

- Top-1 is DataMan 260 Manual `page: 1`;
- Top-1 Similarity is 0.546;
- (a) Below the threshold for evidence gate;
- `false_recall=false`。

Main categories:

```text
truly_unanswerable
```

This is an important security return issue for mixed retrieval. The future keyword search is likely to push the product manual to a higher ranking, but the existence of the “product manual” does not prove that “the real workstation uses the product”, let alone provide an absence of serial numbers.

## 43. Extended final classification

| Classification | Number | Response issues |
|---|---:|---|
| `baseline_success` | 7 | q002、q003、q004、q005、q011、q012、q014 |
| `gate_rejection_with_evidence` | 9 | q001、q006、q010、q013、q015、q016、q021、q023、q025 |
| `semantic_ambiguity` | 5 | q007、q008、q022、q024、q027 |
| `identifier_or_error_code` | 2 | q009、q026 |
| `truly_unanswerable` | 5 | q017、q018、q019、q020、q028 |
| `parsing_missing` | 0 | - |
| `chunk_boundary_break` | 0 | - |
| `pending_verification` | 0 | - |
| Total | 28 | q001–q028 |

The most important improvements after the expansion of Gold are:

```text
Fix 5 candidate or semantic sorting issues
+ Restoration of 2 error code definitive recall problem
+ Calibration of 9 fully substantiated questions denied by door
+ Protect 5 questions and continue to refuse to answer correctly.
```
