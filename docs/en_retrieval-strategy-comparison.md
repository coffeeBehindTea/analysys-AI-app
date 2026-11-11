# Week 3 Retrieval Strategy Comparison

## 1. The experimental problem.

This report compares four candidate search strategies and answers the following questions:

1. Can keyword retrieval complement the recall of pure vectors expressed in error codes, models and values?
2. Can Query Rewrite Deal with Changes of Form `ERR_NET_4001`, `err saf 1002` and `DM260`?
3. Can retention of head reorders preserve complementary evidence of vector and keyword paths in the final Top-3?
4. Will the improvement of the candidate level undermine the safe abstention to answer questions without answers?

The candidate's assessment only determines whether the correct evidence is ranked, does not call the generated LLM or is not equivalent to a final answer correctness rate.

## 2. Common calibration

| Item | Value |
|---|---|
| Gold total number of questions | 28 |
| Number of questions answered | 23 |
| Number of questions not answered | 5 |
| Total evidence currently expected | 31 |
| Final Candidate Depth | Top-3 |
| Embedding Model | `embedding-3` |
| Chroma Collection | `robot_knowledge_week3_vector_baseline` |
| Number of documents | 5 |
| Chunk | 355 |

When the early strategy report was produced, `q021` indicated only one expected piece of evidence; Gold eventually corrected it to cross-documentary issues, with two anticipated evidence. In order to ensure that the four strategies are comparable, this report reads the rankings of candidates that have been saved by each strategy and re-counts using the current `data/eval/gold_questions.jsonl` and official rating functions.

This re-scoring process:

- Do not recreate Embedding;
- Do not visit ChromaDB;
- No change in the order of candidatures;
- Do not call LLM;
- Only current Gold recalculates hits and targets.

## 3. Policy definition and parameters

| Policy | Vector | Keyword | RRF | Query Rewrite | Reorder |
|---|---:|---:|---:|---|---|
| vector-only baseline | Yes. | Yes | Yes | Yes | Yes |
| Mixed RRF | Yes. | Yes. | `candidate_k=20`、`k=60` | Yes | Yes |
| Mixed RRF + Rewrite | Yes. | Yes. | `candidate_k=20`、`k=60` | `deterministic-v2` | Yes |
| Mixed RRF+ Rewrite + Reset | Yes. | Yes. | `candidate_k=20`、`k=60` | `deterministic-v2` | `head-preserving-v1` |

`head-preserving-v1` Protection of complementary head in the final Top-3 in the following order:

1. Retain RRF Integration Top-1;
2. Retain vector path Top-1;
3. Retains the candidate not yet present in the keyword path Top-2;
4. (b) The remaining positions are to be completed in the order of the original RRF;
5. The same chunk appeared only once.

The default policy parameters are:

| Parameters | Value |
|---|---:|
| `top_k` | 3 |
| `candidate_k` | 20 |
| `rank_constant` | 60 |
| `fused_head_k` | 1 |
| `vector_head_k` | 1 |
| `keyword_head_k` | 2 |

## 4. Final comparable indicators

| Policy | Recall@1 | Recall@3 | Top- 1 Full evidence recall | Top- 3 Full evidence recall |
|---|---:|---:|---:|---:|
| vector-only baseline | 0.548（17/31） | 0.742（23/31） | 0.391（9/23） | 0.652（15/23） |
| Mixed RRF | 0.516（16/31） | 0.742（23/31） | 0.391（9/23） | 0.652（15/23） |
| Mixed RRF + Rewrite | 0.613（19/31） | 0.871（27/31） | 0.478（11/23） | 0.826（19/23） |
| Mixed RRF+ Rewrite + Reset | **0.613（19/31）** | **0.968（30/31）** | **0.478（11/23）** | **0.957（22/23）** |

Changes in the final strategy relative to pure vector baselines:

| Indicators | Baseline | Final strategy | Change |
|---|---:|---:|---:|
| Recall@1 | 0.548 | 0.613 | +0.065 |
| Recall@3 | 0.742 | 0.968 | +0.226 |
| Top- 3 Full evidence recall rate | 0.652 | 0.957 | +0.305 |

The acceptance requirement is that the rate of full recall of the candidate's relative pure vector baseline does not decrease. The final plan was upgraded from `15/23` to `22/23`, meeting the requirements.

## 5. Reasons for change in indicators

### 5.1 Only RRF without rewriting has no net improvement

The mix of RRFs is the same as Recall@3 from a pure vector baseline and the full evidence recall. The reason for this is not that keyword retrieval is not working, but rather that the model aliases, error code separator and Chinese-English expression in the original query are still not uniform. The keyword path adds the correct candidate to some of the topics and crowds out the original candidates for some of the topics, with a net result of zero.

### 5.2 Rewrite queries to address differences in expression

`deterministic-v2` Completion using the certainty rule:

- (a) Unanimously underlined, spaced and hyphenated error codes;
- `DM260` extended to `DataMan 260` aliases;
- Extensions of terms such as Chinese and English cables, high voltage, large currents;
- Numbers are harmonized with unit formats.

Retrieval of complete evidence from `15/23` to `19/23`.

### 5.3 Head reordering to resolve multi-road evidence competition

RRF has integrated the rankings in both lists, but eventually Top-3 may still be filled with similar candidates of the same category. Head reordering does not recalculate semantic relevance, but ensures that both the head of vector and the head of keywords have access to the final window, thereby stabilizing complementary evidence on cross-documentary issues.

Reorder Recall@3 from `27/31` to `30/31`, and full evidence recall from `19/23` to `22/23` without lowering Recall@1.

## 6. Subject-by-issue differences

The table below shows only four strategies, Top-3, for changing the number of hits.

| Node. | Expected evidence | vector-only | Mixed RRF | Add Rewrite | Heavy platoons | Main explanation |
|---|---:|---:|---:|---:|---:|---|
| q003 | 1 | 1 | 0 | 1 | 1 | Original keyword competition has led to regression, model and term rewriting |
| q007 | 1 | 0 | 0 | 1 | 1 | DataMan model and English risk word extension recall correct manual page |
| q008 | 1 | 0 | 0 | 0 | 1 | Reset the correct temperature range page in the vector path |
| q009 | 1 | 0 | 1 | 1 | 1 | Accurate error code to fill the keyword path with failure description |
| q016 | 2 | 2 | 1 | 2 | 2 | Original mixed search drops and rewriting cross-document evidence |
| q021 | 2 | 1 | 1 | 1 | 2 | Reorder while maintaining the failure statement and security repositioning protocol |
| q022 | 2 | 1 | 2 | 2 | 2 | `ERR_NET_4001` Normalization and Keyword Path to Second Jump |
| q023 | 1 | 1 | 0 | 1 | 1 | `DM260` and rewording of the cable risk terminology to repair mixed regressions |
| q024 | 1 | 0 | 0 | 0 | 1 | Reorder to keep the correct charger page at Top-3 |
| q026 | 2 | 1 | 2 | 2 | 2 | The error code is standardized to complete the failure statement and test protocol. |

The final strategy has not lost the complete recall of the pure vector baseline.

## 7. Outstanding candidate issues

The `q027` still only recalls one of the two expected evidence:

```text
Question: How do you complete the acceptance survey after DM260 camera protection cap cleaning?
Recalled: DataMan 260 Clean evidence
Not in Top-3: workstation 2D to read acceptance protocol
```

This problem requires a cross-document " Device Maintenance + Business Receiving and Inspection" relationship. The current definitive rewording allows for the identification of model and cleaning semantics, but there are no mechanical rules that can stabilize the inference that “a workstation acceptance protocol is also required after cleaning”. This is the main candidate for subsequent search decomposition, relationship indexing or learning-type reordering.

## 8. Evidence gate security assessment

Before the winner had access to online API, `hybrid-evidence-gate-v1` was used for independent door-to-door assessments.

| Indicators | Result |
|---|---:|
| Questionable release rate | 0.957（22/23） |
| Release and complete evidence. | 21/23 = 0.913 |
| Released but incomplete evidence | 1/23 |
| Error release rate without answer | 0/5 = 0.000 |
| No answer correct response rate | 5/5 = 1.000 |

The evidence gate does not use only Top-1 similarity, but rather uses it according to the characteristics of the problem:

- `exact_identifier_support`: Accurate identifiers such as error codes are supported;
- `model_and_lexical_support`: Joint support of models and thematic keywords;
- `general_dual_path_support`: Vector and keyword two-track support;
- `insufficient_combined_support`: rejected when not substantiated.

The correct evidence of `q007` was put forward, but the portfolio support was still below the threshold requirements and was therefore rejected. `q027` was released but the evidence was incomplete. All five questions without answers were rightly rejected, and candidate improvements did not reduce the security of the answers.

## 9. Final online response and reference assessment

The balance of “general evidence constraint + minimum full reference” is ultimately used, and no equivalent or mandatory single evidence rules are retained for additions to the monograph.

| Indicators | Result | Receiving and inspection target | Conclusions |
|---|---:|---:|---|
| Response rate to questions that can be answered | 0.870（20/23） | ≥ 0.70 | Pass |
| Full evidence response rate | 0.739（17/23） | ≥ 0.65 | Pass |
| Citation correctness | 0.962（25/26） | ≥ 0.95 | Pass |
| Citation coverage | 0.806（25/31） | Observation indicators | Recorded |
| No answer correct response rate | 1.000（5/5） | ≥ 0.95 | Pass |

The fact that the full evidence of the candidacy was recalled by `0.957` more than the final full evidence response rate of `0.739` indicates that the remaining gaps occurred mainly in evidence gate and LLM evidence selection rather than in candidate recall.

The topics that were not fully adopted were divided into four categories:

- `q007`, `q025`: Candidate exists but is refused by evidence gate or LLM;
- `q013`, `q026`: the answer is basically complete, but only one of two Gold evidences is selected;
- `q024`: The correct evidence was cited, while a more relevant page was selected but not on the Gold Allowlist;
- `q027`: Candidate Top-3 lacks the second necessary evidence.

The different issues are not being repaired through global Prompt patches, and the increase in individual indicators does not undermine the response rate or the completeness of references to other issues.

## 10. Final Selection

The final production candidate strategy is:

```text
deterministic- v2 Query redraft
→ Vector candidate Top-20
→ Keyword candidate Top-20
→ RRF(k=60)
→ head-preserving-v1(1/1/2)
→ Final Top-3
→ hybrid-evidence-gate-v1
```

_Other Organiser

1. Top- 3 with the highest recall of complete evidence at `0.957`;
2. (b) A relative pure vector baseline increase of `0.305`;
3. There is no complete recall of the lost baseline;
4. The error release rate of five questions without answers is maintained at `0.000`;
5. The final online answers and quotes are all through the Week 3 acceptance line.

## 11. Basis of the report

- `data/eval/gold_questions.jsonl`
- `docs/retrieval-strategy-vector-baseline.json`
- `docs/retrieval-strategy-hybrid-rrf.json`
- `docs/retrieval-strategy-hybrid-rrf-rewrite-v2.json`
- `docs/retrieval-strategy-hybrid-rrf-rewrite-rerank-v2.json`
- `docs/hybrid-gate-evaluation-rerank-v2.json`
- `docs/citation-evaluation-week3-balanced-rollback.json`

The JSON report contains candidate body or reference clips, which are retained locally only; this Markdown report only discloses parameters, indicators, question numbers and failed classification.
