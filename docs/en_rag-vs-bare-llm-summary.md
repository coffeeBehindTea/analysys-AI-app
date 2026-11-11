# Week 2 RAG vs. Standalone LLM Evaluation Summary

## 1. Purpose of the experiment

The experiment compares the behaviour of the same generation model under two input conditions:

```text
RAG Group: Problem + Current Knowledge Bank Retrieving Evidence
standalone LLM group: Question + Responsive answer to rule
```

Standalone llm will not see Gold reference answers, expected evidence or RAG responses. Full answers and original references are only stored locally, and only the classification of results, time-consuming and project conclusions are disclosed.

## 2. Experiment Configuration

| Configure | Value |
|---|---|
| LLM Model | `deepseek-v4-flash` |
| Embedding Model | `embedding-3` |
| Chroma Collection | `robot_knowledge_v4` |
| Top-K | 3 |
| RAG evidence gate threshold | 0.60 |
| Experimental topics | q014、q020 |

## 3. Initial results

| ID | Gold type | RAG Status | RAG references | RAGend to End Time | RAG Retrieving Time | Standalone llm time |
|---|---|---|---:|---:|---:|---:|
| q014 | How many jumps can answer? | Replyed | 2 | 11400.6 ms | 754.3 ms | 8595.8 ms |
| q020 | Unanswerable | Rejected | 0 | 276.1 ms | 218.1 ms | 2238.4 ms |

Time-consuming results are the result of a single real network experiment, which is influenced by the randomity of service providers ' loads, networks and models and cannot be used as a stability benchmark.

## 4. q014 Observation

RAG found two project-specific evidence and returned references that can be located to the true Chunk. standalone LLM has no project evidence and is mixed with generic temperature, restoration conditions and test recommendations that are not supported by current language.

The initial RAG response, while correctly quoted, omitted negative security constraints from the evidence. This indicates that:

```text
Retrieving correct + Quoting correct
Not equal to
Answer the semantics must be complete.
```

Prompt subsequently added rules on retention prohibitions, mandatory, preconditions and numerical thresholds. q014 V2 expressly retains the safe boundary “not permitted to block a genuine transmission machine for test purposes”.

V2 Single time-consuming records:

| Item | Result |
|---|---:|
| RAGend to End Time | 13322.1 ms |
| RAG Retrieving Time | 874.7 ms |
| Time-consuming generation of standalone LLM | 21755.0 ms |

## 5. q020 Observation

q020 requires a real-time robotic state, while the current knowledge base contains only static documents.

RAG returns a structured rejection and empty reference after a similar level of evidence gate, without calling the generated answer LLM. Standalone llm has also given a cautious conclusion, but has executed one request for generation.

The case demonstrates that the code layer controls can provide:

- More definitive refusals;
- (a) Empty reference assurance;
- Lower cost of an answerless path;
- It does not rely on the model to determine whether or not to know the real-time state.

## 6. Conclusions

The main value of the RAG is not to allow the model to answer all questions, but to provide project evidence, retroactivity and code-level refusal. Experiments also exposed key safety constraints that could still be missing from the gestation layer, and it was therefore necessary to use Prompt, Schema, allowlist of evidence, automated tests and manual assessments simultaneously.
