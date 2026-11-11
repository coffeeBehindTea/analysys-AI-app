# Week 2 Citation Evaluation Summary

## 1. Purpose

This document is a summary of the public redacted of the threshold `0.60` and V2 quoted assessments.

The complete local report contains model answers, reference answers and quotation details and is therefore not submitted to the open warehouse. This article does not contain original Chunk, quote `excerpt` or complete model answers.

## 2. Evaluation Configuration

| Configure | Value |
|---|---|
| Total number of questions | 20 |
| Answerable questions | 16 |
| No answer to question | 4 |
| Top-K | 3 |
| Online similarity evidence gate threshold | 0.60 |
| LLM Model | `deepseek-v4-flash` |
| Embedding Model | `embedding-3` |
| Chroma Collection | `robot_knowledge_v4` |

## 3. Aggregate Metrics

| Indicators | Result |
|---|---:|
| Number of questions answered that were actually answered | 7/16 |
| Full evidence answers | 7/16 |
| Number of unanswered questions correctly rejected | 4/4 |
| Returns total references | 8 |
| Total correct references | 8 |
| Citation correctness | 1.000 |
| Citation coverage | 0.400 |
| Response rate to questions that can be answered | 0.438 |
| Full evidence response rate | 0.438 |
| Correct rate of rejection for unsettled questions | 1.000 |

## 4. Issue-by-issue status

### Answered and quoted in full

```text
q002 q003 q004 q005 q011 q012 q014
```

All references to these questions returned match the Gold file and location, and no model self-made filenames, page numbers or Chunk IDs are accepted.

### Gold can answer but online API refuses

```text
q001 q006 q007 q008 q009 q010 q013 q015 q016
```

There are two types of reasons:

1. q007, q008 and q009 did not enter Top-3;
2. Other questions, Top-3, contain the correct evidence, but Top-1 is below `0.60` and the answer is denied by code.

### No answer is correct.

```text
q017 q018 q019 q020
```

These questions are returned to `abstained=true` and empty references.

## 5. V2 Improvements

The threshold `0.60` was reported in the first round of 9 returns with only 8 correct references and a citation correctness of `0.889`.

Prompt adds the rule of “minimum adequate collection of evidence”, so V2 returns only 8 pieces of evidence that are actually required, and 8 is all correct:

```text
Quoting correct rate: 0.889 → 1.000
Full evidence response rate: 0.375 → 0.438
```

## 6. Acceptance Conclusion

Quoting authenticity and non-resolved refusals is stable, but online response rate is only `7/16`. Therefore, the RAG on the `Recall@3 = 0.850` decision line cannot be used for its available quality. The next phase will require improvements in retrieval and evidence gate, while maintaining the correct rate of reference and the correct rate of non-response.
