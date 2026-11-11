# Week 3 Engineering Summary

## 1. Objectives and results for the week

Week 2 has completed document ingestion, vector storage, knowledge base questions and quotations, but online assessments answered only 7 out of 16 answers. The goal of Week 3 is not to continue stacking interfaces, but to create a reliability loop that can be located “in a library that cannot be searched, searched without being released, released without being correctly quoted”, and to convert evidence into structured diagnostic reports that robotic research can audit.

Final result:

- Gold expanded from 20 to 28, of which 23 were answered, 5 were unanswered;
- Candidate Top-3 Full evidence recall from `15/23=0.652` in pure vector to `22/23=0.957`;
- The response rate to questions that can be answered is `20/23=0.870`;
- (a) Complete evidence response rate of `17/23=0.739`;
- (a) The citation correctness is `25/26=0.962`;
- Five questions with no answers are all rejected correctly.
- Add structured diagnostic API and complete downgrade paths;
- 798 off-line automation tests completed, all passed.

## 2. Location in the AI application frame

The full AI application is not just a "one call LLM" but consists of multiple layers of credibility:

```text
External input
→ HTTP and Pydantic enter boundary
→ Normalization of certainty queries
→ Retrieval and candidate sorting
→ Evidence Gating
→ LLM Generation
→ Structured Schema Verify
→ Allowlist Map of Evidence
→ Degraded and detectable
→ External response
→ Offline and online assessments
```

Week 3 has been supplemented by the intermediate “research reliability, evidence gate, structured output, reference binding and evaluation of closed rings”. These layers exist because LLM is good at language organization, but are not fit to assume the duties that must be performed, such as the regularization of identifiers, filtering of privileges, reference to identity, mandatory rules of risk and stabilization downgrading.

## 3. From pure vector to mixed search

### 3.1 Deficiencies in pure vectors

Pure vector search can handle semantics but is unstable for the following inputs:

- `ERR-NET-4001`、`ERR_NET_4001`、`err net 4001`；
- `DM260` and `DataMan 260`;
- `-10°C`, `1500 ms` and in units;
- The terminological correspondence in the manual in Chinese and English;
- Process problems need to be found for both documents.

These elements contain a large amount of symbolic and word-level information, which cannot be based solely on the overall semantic similarity of the vector.

### 3.2 Keywords Retrieving Why Can't It Be Worked Out?

After the first keyword search and RRF were added, Top-,3 complete evidence was recalled to `15/23`, which is not above the pure vector. The keyword path did save the wrong code, but it also squeezed out part of the vector candidate because of a lack of consistency between the model aliases and the Chinese/English expression.

This results indicate that “an additional search path” is not the same as “an indicator necessarily increases”. Both paths must first share stable query indications, and the final candidate window needs to address path competition.

### 3.3 Duty to rewrite definitive queries

`deterministic-v2` Not calling LLM, but using testable rules:

- (a) The error code separator is uniform;
- (b) Model aliases extension;
- (a) Harmonization of numbers and units;
- Extension of specific Chinese and English terms.

This design is because the error code and model belong to the Identifier. LLM guesses that the identifier increases delay, cost and non-recurrence, and may write non-existent models to queries.

Query rewrite to recall the full evidence to `19/23=0.826`.

### 3.4 Rearrangement of head

RRF has converted the rankings of different scales into uniform contributions, but eventually Top-3 may still be filled by similar candidates on a certain path. `head-preserving-v1` Retain the complementary candidates in RRF Top-1, Vector Top-1 and Keyword Top-2 and replace them with the original RRF.

It is not a learning semantic reorder, nor does it re-judge the quality of the body. It addresses the issue of “path representation” in a limited number of candidate windows and remains fully identifiable, interpretable and testable.

The final complete evidence was recalled to `22/23=0.957`.

## 4. Why can't the evidence gate just look at Top-1 similarity?

The single threshold for Week 2 can easily create two types of problems:

1. The error codes are fully matched, but the vectors are somewhat similar, leading to the rejection of questions that can be answered;
2. No answer questions are close to the subject of the text, Top- 1 is not too similar and may be wrongly released.

`hybrid-evidence-gate-v1` of Week 3 and checking:

- Accuracy error code support;
- Joint support of models and subject words;
- Two-way support for vectors and keywords;
- The calibration threshold for different problem types.

As a result, 22/23 answers were released and all five were rejected. The goal of evidence gate is not to maximize recall rates, but to control “whether there are enough grounds to be generated” before entering LLM.

## 5. Why do structured diagnostics need two layers of Schema?

The diagnostic output uses two different layers of Pydantic contracts.

### 5.1 Internal `DiagnosisLLMDraft`

LLM returns only:

- Reason and the text of the check item;
- `E1`, `E2` Interim evidence number;
- Risks, missing information and state of refusal.

### 5.2 External `DiagnosisReport`

Python is in charge of:

- Request ID；
- Prompt and evidence gate versions;
- Real Chunk ID, filename and location;
- RRF fractions and vector similarities;
- Source of user symptoms.

This layer prevents LLM from filling in its own file name, page number, Chunk ID or internal version. Even if the model returns the correct format `E999`, Builder will refuse it as long as it is not on this candidate allowlist.

## 6. Degrading is not an auxiliary function of abnormal treatment.

This week, the downgrading was designed as a formal business path:

- (a) Evidence gate refusal: no LLM call, return stable refusal;
- LLM refuses to respond at its own initiative: preserves evidence, clears cause and operation;
- Illegal JSON, missing field, unknown reference: conversion to structural verification failure report;
- LLM overtime or upstream error: returns a fixed public statement;
- Program Bug: Not disguised as a normal refusal to respond, continues to be exposed to the error-processing system.

This distinction is very important. (a) Operational “unsubstantiation” should be expected results; The code returns the wrong type or violates Protocol is a development defect and cannot be swallowed.

## 7. The conclusion of the Prompt control experiment

The basic diagnosis, Prompt, returned to JSON in the real case, with a structural success rate of 0. Prompt gives priority to state relationships, quote numbers, risk boundaries and missing information, and the structural success rate and allowlist adoption rate are 1/1.

But the experiment was only one case, and it proved that:

- The call chain can be implemented;
- The structured rules have indeed changed the pattern of failure;
- Pydantic was able to intercept illegal drafts.

It does not prove that Prompt is superior to other versions on all issues.

## 8. The lessons of Prompt's convergence

In order to repair the individual reference questions, the following points have been gradually added:

- (a) Forced compression to the minimum;
- (b) Force the choice of only one case when a single piece of evidence is capable of answering;
- (b) Discrepancies of findings;
- Strictly equivalent boundary ad hoc.

These rules can create a gap between accuracy and coverage: repairing the redundant quotation of one subject may result in the omission of the second necessary evidence from another multiple jump; reducing the number of refusals; or allowing the model to interpret the lack of evidence as a valid negative conclusion.

Eventually back to General Prompt, only to keep:

- (a) Based only on evidence;
- (a) Status and security qualifiers;
- (b) Sufficient evidence and a rejection structure;
- Quoting allowlist;
- The minimum of sufficient evidence.

All acceptance targets for the back-up were still adopted, and therefore continued addition of global Prompt rules to the monographs was discontinued.

## 9. Engineering awareness of the evaluation system

### 9.1. The recall is not the right answer.

The full evidence of the candidate was recalled to `0.957`, with a final complete evidence response rate of `0.739`. The gap between the two stems from:

- (a) Evidence gate refusal;
- LLM second refusal;
- LLM does not select the necessary evidence;
- LLM chooses evidence other than Gold.

The order of the examination must therefore be:

```text
Let's see if the right evidence is on the list.
→ And watch if the door's free.
→ Let's see if the LLM answers.
→ See if the references are complete and there are no redundant entries.
```

### 9.2 Gold himself is a data contract.

`q021` Initially, only the description of the failure was marked and later it was discovered that the problem also required a safe repositioning condition, which was amended to include two cross-documentary evidence. The denominator of the earlier report after the amendment was changed from 30 to 31, which cannot be compared directly with the new report.

The final report uses the stored candidate rankings, which are re-rated according to the current Gold, to ensure that the strategy is of the same calibre. This suggests that changes in the assessment of data must also be translated and impact recorded.

### 9.3 “Complex evidence” is a strict mechanical indicator

The single test protocols `q013` and `q026`, Chunk, already contain most or even all of the answers to the facts, but Gold requires two documents and is thus still marked as incomplete. The second manual page of `q024` is relevant and supports BMS thresholds but is not on the Gold Allowlist and is therefore counted as redundant.

Mechanical indicators are suitable for return and comparison, but cannot be a complete substitute for artificial semantic review.

## 10. The most effective engineering practice

The most effective practices this week include:

1. First create failed billboards, then change the search code.
2. Writes the error code and model in a defined pure function.
3. Using Protocol to make Service dependent on capacity rather than on specific SDK.
4. Use Fake to verify the branches of “shall call” and “should not call”.
5. Give the reference to Python instead of LLM.
6. Writes door controls, Prompt and rearranges to version fields or reporting parameters.
7. Report separately on candidate, evidence gate, answer and quote indicators.
8. Retain independent reports for each round of optimization without covering the old results.
9. It was found that Prompt started to retreat in time for a single issue expansion.

## 11. Legacy issues

### 11.1 `q027` Cross-documentation recall

The current query can find DM260 clean pages, but it is not stable to think about the warehouse workstation acceptance protocol. Follow-up may consider:

- (a) A composite intention template for certainty;
- (a) Inter-document relationship metadata;
- Establishing a searchable link between "maintenance action " and "receipt action ";
- Reordering of the Learning Cross-Encoder;
- Controlled query decomposition.

### 11.2 Gate control and model refusal

`q007` and `q025` indicate that having the correct candidate does not amount to a final answer. The evidence gate should then be calibrated separately and the adequacy of the response should not be adjusted for a single global similarity.

### 11.3 Uncertainty about real services

Even when external LLM uses `temperature=0`, it may result in different results due to service-end model updates, reasoning or load changes. The long-term assessment should run multiple rounds and report stability of averages, the most margin and failures.

### 11.4 Productive capacity gap

The system is still lacking:

- OCR and complex PDF layouts;
- (b) User certification, privileges and tenant segregation;
- Document version and deletion audit;
- (b) Backstage ingestion tasks;
- Distribution vector database;
- Real-time RCS, WMS and telemetry tools;
- Production level monitoring, speed limits and capacity planning.

## 12. Final Conclusion

Week 3 has completed the evolution from "run RAG" to "capable of locating failure, comparing strategy, controlling evidence, stabilizing downgrade and exporting structured diagnosis".

The final system is still not a robotic controller, nor does it guarantee the correct answer in all natural languages, but it has established critical engineering boundaries:

- Retrieving offline baselines;
- Comparable indicators of changes in strategies;
- No answer is certain.
- LLM output has Schema;
- Quoting a real Chunk allowlist;
- (b) The compulsory qualification boundaries for high-risk movements;
- (b) Stable downgrading of failures;
- Automation tests and audit reports are available on the full link.
