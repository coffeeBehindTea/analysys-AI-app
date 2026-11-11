# Week 2 Retrieval Evaluation Summary

## 1. Purpose

This document is an open, redacted summary of the `robot_knowledge_v4` search assessments.

The complete local report contains a preview of the Chunk text and original language and is therefore excluded by `.gitignore`. This paper retains only indicators, title numbers, ranking results and failure classifications, which do not contain `content`, `excerpt` or large paragraphs.

## 2. Evaluation Configuration

| Configure | Value |
|---|---|
| Gold Question Number | 20 |
| Answerable questions | 16 |
| No answer to question | 4 |
| Embedding Model | `embedding-3` |
| Chroma Collection | `robot_knowledge_v4` |
| Top-K | 3 |
| Unresolved recall threshold for this report | 0.70 |

Note: The threshold for response evidence gate for the online knowledge base API is `0.60`. The search report's `0.70` is used only for the unresolved recall determination of the offline report, which cannot be confused with the same configuration.

## 3. Aggregate Metrics

| Indicators | Result |
|---|---:|
| Total anticipated evidence | 20 |
| Recall@1 | 0.600 |
| Recall@3 | 0.850 |
| Top-1 Full recall questions | 8/16 |
| Top-3 Full recall questions | 13/16 |
| Unresolved Wrong recall Number | 0/4 |
| Unresolved wrong recall rate | 0.000 |

The offline target of `Recall@3 >= 0.80` for Week 2 has been achieved, but this does not mean that the availability of online questions and answers has been achieved.

## 4. Redacted results, issue by issue.

| ID | Type | Top- 1 Hit evidence | Top- 3 Hit evidence | Top- 3 Full recall | Top- 1 Similarity |
|---|---|---:|---:|---|---:|
| q001 | single_hop | 1 | 1 | Yes. | 0.569 |
| q002 | single_hop | 1 | 1 | Yes. | 0.640 |
| q003 | single_hop | 1 | 1 | Yes. | 0.619 |
| q004 | single_hop | 1 | 1 | Yes. | 0.612 |
| q005 | single_hop | 0 | 1 | Yes. | 0.748 |
| q006 | single_hop | 1 | 1 | Yes. | 0.564 |
| q007 | single_hop | 0 | 0 | Yes | 0.537 |
| q008 | single_hop | 0 | 0 | Yes | 0.598 |
| q009 | single_hop | 0 | 0 | Yes | 0.556 |
| q010 | single_hop | 1 | 1 | Yes. | 0.530 |
| q011 | single_hop | 1 | 1 | Yes. | 0.681 |
| q012 | single_hop | 1 | 1 | Yes. | 0.625 |
| q013 | multi_hop | 1 | 2 | Yes. | 0.581 |
| q014 | multi_hop | 1 | 2 | Yes. | 0.623 |
| q015 | multi_hop | 1 | 2 | Yes. | 0.589 |
| q016 | multi_hop | 1 | 2 | Yes. | 0.589 |
| q017 | unanswerable | 0 | 0 | Not applicable | 0.482 |
| q018 | unanswerable | 0 | 0 | Not applicable | 0.480 |
| q019 | unanswerable | 0 | 0 | Not applicable | 0.527 |
| q020 | unanswerable | 0 | 0 | Not applicable | 0.519 |

## 5. Failed to classify

| ID | Type of failure | Redacted | Follow-up directions |
|---|---|---|---|
| q007 | Cross-linguistic Details Positioning | The correct product manual was found, but the correct security precautions were not assigned to Top-3 | Rewrite, chapter context, reordering in Chinese English |
| q008 | The syntax is similar, but the evidence is incomplete. | Returns the relevant configuration page in the same product manual, but does not cover the full range of parameters | Keyword retrieval, scope enhancement, reordering |
| q009 | The structured identifier was not prioritized | Pandaming security content overwhelms the exact failure code chapter. | Fault code exact match, mixed search, metadata filter |

## 6. Door risk management

Top-3 of q001, q006, q010, q013, q015 and q016 already contain complete Gold evidence, but Top-1 similarity is below the online evidence gate threshold of `0.60`, so online API will refuse to answer directly.

The Top-1 similarity of q008 is close to `0.60`, but Top-3 has no correct evidence. Directly lowering the threshold may allow error search, so a simple lower threshold cannot be used as an alternative to retrieval optimization.

## 7. Conclusions

Pure vector search has reached the Week 2 offline Recall@3, but the next stage must simultaneously optimize candidate recall, integration sequencing and evidence gate strategies, with a common acceptance with an answerable response rate, full evidence response rate and no answer security.
