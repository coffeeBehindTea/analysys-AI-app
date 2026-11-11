# Multimodal Agent Offline Regression Report

## 1. Return Range

- Return: `multimodal-offline-regression-v1`
- Fixture Version: `multimodal-offline-fixture-v1`
- Implementation modalities: `offline_fakes`
- Use of external services: No
- Gold scene: `data/eval/multimodal_scenarios.jsonl`
- Fixture：`data/eval/multimodal_offline_fixtures.jsonl`
- Planner：`scripted-offline-planner`
- Planner Prompt：`agent-tool-calling-v2`
- Vision Provider：`fake-vision-provider`
- Vision Prompt：`robot-vision-observation-v1`

## 2. Recovery, pre- and post-reception threshold

| Item | Pre-rehabilitation baseline | Current Results | Requirements for acceptance | Conclusions |
|---|---:|---:|---:|---|
| Strictly pass the scene number. | 7 | 30 | At least 24. | Pass |
| Strict pass rate | 0.233 | 1.000 | 24/30 | Pass |
| Safe abstention Rate | 1.000 | 1.000 | Don't fall. | Pass |

- Increase in pass-throughs relative to week 5: **23**

## 3. Official Scoring Metrics

| Indicators | Result |
|---|---:|
| Request success rate | 1.000 |
| Total pass rate of scene | 1.000 |
| Image observation field accuracy | 1.000 |
| Tool selection accuracy | 1.000 |
| Vision Tool selection accuracy | 1.000 |
| Task completion rate | 1.000 |
| Citation correctness | 1.000 |
| Citation coverage | 1.000 |
| Source labeling accuracy | 1.000 |
| Safe abstention Rate | 1.000 |
| Average number of tools steps | 1.467 |
| Average local implementation delays | 8.063 ms |
| P50 Local execution delay | 6.495 ms |
| P95 Local implementation delays | 18.119 ms |

## 4. Strict pass and failure.

### 4.1 through scenes

`multimodal-agent-001`、`multimodal-agent-002`、`multimodal-agent-003`、`multimodal-agent-004`、`multimodal-agent-005`、`multimodal-agent-006`、`multimodal-agent-007`、`multimodal-agent-008`、`multimodal-agent-009`、`multimodal-agent-010`、`multimodal-agent-011`、`multimodal-agent-012`、`multimodal-agent-013`、`multimodal-agent-014`、`multimodal-agent-015`、`multimodal-agent-016`、`multimodal-agent-017`、`multimodal-agent-018`、`multimodal-agent-019`、`multimodal-agent-020`、`multimodal-agent-021`、`multimodal-agent-022`、`multimodal-agent-023`、`multimodal-agent-024`、`multimodal-agent-025`、`multimodal-agent-026`、`multimodal-agent-027`、`multimodal-agent-028`、`multimodal-agent-029`、`multimodal-agent-030`

### 4.2 Failed scene

None

## 5. Failed to classify statistics

There are no reasons for failure in this instalment.

## 6. Fixture consumption audit

- Total Fixture consumption scenes: 30/30
- Incomplete consumption of Fixture scenario: none

| scene | Result | Tool Order | Planner calling/preset | Vision Call/ Preset | Full consumption of Fixture | Reason for failure |
|---|---|---|---:|---:|---|---|
| `multimodal-agent-001` | PASS | analyze_robot_image → search_knowledge | 3/3 | 1/1 | Yes. | None |
| `multimodal-agent-002` | PASS | analyze_robot_image → search_knowledge → get_robot_telemetry | 4/4 | 1/1 | Yes. | None |
| `multimodal-agent-003` | PASS | analyze_robot_image → search_knowledge → draft_test_case | 4/4 | 1/1 | Yes. | None |
| `multimodal-agent-004` | PASS | analyze_robot_image → search_knowledge → get_robot_telemetry | 4/4 | 1/1 | Yes. | None |
| `multimodal-agent-005` | PASS | analyze_robot_image → search_knowledge → get_robot_telemetry | 4/4 | 1/1 | Yes. | None |
| `multimodal-agent-006` | PASS | analyze_robot_image → search_knowledge | 3/3 | 1/1 | Yes. | None |
| `multimodal-agent-007` | PASS | analyze_robot_image → search_knowledge | 3/3 | 1/1 | Yes. | None |
| `multimodal-agent-008` | PASS | analyze_robot_image → search_knowledge → draft_test_case | 4/4 | 1/1 | Yes. | None |
| `multimodal-agent-009` | PASS | analyze_robot_image | 2/2 | 1/1 | Yes. | None |
| `multimodal-agent-010` | PASS | analyze_robot_image | 2/2 | 1/1 | Yes. | None |
| `multimodal-agent-011` | PASS | analyze_robot_image | 2/2 | 1/1 | Yes. | None |
| `multimodal-agent-012` | PASS | analyze_robot_image | 2/2 | 1/1 | Yes. | None |
| `multimodal-agent-013` | PASS | analyze_robot_image → search_knowledge | 3/3 | 1/1 | Yes. | None |
| `multimodal-agent-014` | PASS | analyze_robot_image → search_knowledge | 3/3 | 1/1 | Yes. | None |
| `multimodal-agent-015` | PASS | analyze_robot_image → search_knowledge | 3/3 | 1/1 | Yes. | None |
| `multimodal-agent-016` | PASS | analyze_robot_image → search_knowledge | 3/3 | 1/1 | Yes. | None |
| `multimodal-agent-017` | PASS | None | 0/0 | 0/0 | Yes. | None |
| `multimodal-agent-018` | PASS | None | 0/0 | 0/0 | Yes. | None |
| `multimodal-agent-019` | PASS | analyze_robot_image | 2/2 | 1/1 | Yes. | None |
| `multimodal-agent-020` | PASS | analyze_robot_image | 2/2 | 1/1 | Yes. | None |
| `multimodal-agent-021` | PASS | search_knowledge | 2/2 | 0/0 | Yes. | None |
| `multimodal-agent-022` | PASS | get_robot_telemetry | 2/2 | 0/0 | Yes. | None |
| `multimodal-agent-023` | PASS | analyze_robot_image → get_robot_telemetry | 3/3 | 1/1 | Yes. | None |
| `multimodal-agent-024` | PASS | analyze_robot_image → get_robot_telemetry | 3/3 | 1/1 | Yes. | None |
| `multimodal-agent-025` | PASS | analyze_robot_image | 2/2 | 1/1 | Yes. | None |
| `multimodal-agent-026` | PASS | analyze_robot_image | 2/2 | 1/1 | Yes. | None |
| `multimodal-agent-027` | PASS | None | 0/0 | 0/0 | Yes. | None |
| `multimodal-agent-028` | PASS | None | 0/0 | 0/0 | Yes. | None |
| `multimodal-agent-029` | PASS | analyze_robot_image | 2/2 | 1/1 | Yes. | None |
| `multimodal-agent-030` | PASS | None | 0/0 | 0/0 | Yes. | None |

## 7. Reliability correction records

| Problem detected | Modify | Authentication Method |
|---|---|---|
| Normal request default overexposure tool, Planner may choose irrelevant tool | Increase the minimum tool policy at the request level to expose only the necessary tools to Planner and Execuator | 30 offline scene check if the actual tool sequence meets the Gold tool range |
| Repeating queries or no new evidence queries may result in meaningless tool loops | Document evidence to overwrite and next allowed actions using AgentProgress and ProgressReducer | Offline execution audit records for Planner rotations, tool call-ups and Fixture consumption status |
| The semantics of high-risk requests may depend on Planner's discretion. | Add the Secure Identification Classification and listst_policy_finished termination path before Planner | Security scene check for denials, reasons for discontinuation and zero-tool call. |

## 8. Evaluation limits

- This report validates the certainty safety classification, the scope of the request-level tool, the Agent progress status machine, Runner, Executor, the allowlist of evidence, the reporting structure and offline stability of the rating rules.
- Planner, Vision and common tool results are derived from fixed Fixture, so the report does not represent randomity or online mass of real model outputs.
- Local implementation delays reflect only Fake's dependence and the time spent on its own Python main chain, which cannot be used as a real network model for delayed or productive capacity indicators.
- No external costable model was called in this instalment and therefore the report is not used to estimate the real model costs.
- Reports and tracks do not save a full picture, key, complete sensitive log or model private think chain.
