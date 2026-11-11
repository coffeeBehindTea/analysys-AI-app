# Robot Diagnostic Agent Failure Cases and Recovery Strategies

## 1. Purpose

This document records the failed boundary, restoration rules, stable termination causes and automated testing positions that Week 4 ROBOT Diagnostic Agent has achieved.

This document is used to answer questions:

1. The failure occurred in Planner, Runner, Execuator or tool Handler;
2. Has the tool been implemented after failure;
3. Retention of the completed tool steps;
4. (a) Whether Planner could adjust the programme to continue;
5. What situations must be discontinued immediately;
6. The procedure should determine which structural state or wrong code;
7. Which automated test validated the act.

This document does not record the time of generation and avoids meaningless differences in versions due to time changes only.

## 2. Failed to control position in call chain

Agent failed to control the call chain as follows:

1. `AgentDiagnosisService`
   - Collapse user requests into Agent jobs;
   - Call AgentRunner;
   - Follow-up to a complete, partial or negative diagnosis based on Runner results.

2. `AgentRunner` (Current failure control module)
   - Create the context for each round of planning;
   - Limit Planner timeout;
   - Verifying Planner return value;
   - Check maximum steps and repeat calls;
   - Hand over the legal tool request to ToolExecutor;
   - (b) Preservation tool interaction;
   - Continue or abort according to the failure strategy.

3. `AgentPlanner`
   - Only the structured `call_tool` or `finish` decision can be returned;
   - Could not directly execute the Python function;
   - Could not close temporary folder:%s

4. `ToolRegistry`
   - (b) To maintain a allowlist of read-only tools registered at start-up;
   - A dynamic import function that is not returned by the model;
   - Refusal to write and high-risk tools.

5. `ToolExecutor`
   - Accurate matching tool name;
   - Validation of parameters using specific Pydantic input models;
   - Re-check read-only permissions and risk levels;
   - Timeout of the execution order tool;
   - (a) Verify the tool output;
   - returns unified `ToolExecutionResult`.

6. `failure_policy.py` (Classification module for current tool results)
   - Categorizing the results of the tool as successful, failing to recover or failure in lethal privileges;
   - Can not parse the false message of natural language;
   - No implementation tool, and no direct change in the Agent status.

7. `AgentRunResult`
   - (b) To preserve the cause of termination;
   - (a) Preservation of the tools completed for interaction;
   - (a) Keep information that remains unresolved;
   - It is forbidden to carry Planner's final draft with the security suspension.

The failure control is determined by the Python structured rules. User input, document body, telemetry content, tool text and LLM natural language cannot directly change these rules.

## 3. Agent's execution and diagnosis.

### 3.1 Agent Implementation Status

Agentrunner currently uses two finals:

| Status | Meaning |
|---|---|
| `completed` | Planner's on his own to end the cycle with a structured agreement. |
| `aborted` | Python suspended the cycle due to security restrictions, overtime, abnormal or no progress conditions |

`completed` indicates only the normal end of the cycle and does not mean that the diagnosis must be complete or correct.

### 3.2 Status of diagnostic content

After that, AgentDiagnosisService still needs to re-use Week 3's diagnosis:

| Diagnosis status | Meaning |
|---|---|
| `completed` | The evidence is sufficient to support a full diagnosis. |
| `partial` | Only part of the conclusion is supported by evidence. |
| `abstained` | There's not enough evidence to make a safe diagnosis. |

Possible combinations include:

| Runner Status | Diagnosis status | Annotations |
|---|---|---|
| `completed` | `completed` | Agent's normal end and complete evidence. |
| `completed` | `partial` | Agent is normally aware that the information is still incomplete. |
| `completed` | `abstained` | Agent's normal confirmation is not an answer. |
| `aborted` | `partial` | Agent had some credible evidence before the suspension. |
| `aborted` | `abstained` | Agent's suspended and not sufficiently credible. |

Runner will not lower the evidentiary threshold to turn `aborted` into `completed`.

## 4. Reasons for stabilization and termination

`AgentRunTerminationReason` Currently supported:

| Quantity | Runner Status | Meaning |
|---|---|---|
| `planner_finished` | `completed` | Planner voluntarily returned to the lawful Finnish decision |
| `max_steps_reached` | `aborted` | The overall tool step ceiling has been reached and Planner continues to request the tool |
| `duplicate_tool_call` | `aborted` | Planner repeats the call ID, or repeats the same tools and parameters |
| `planner_timeout` | `aborted` | Single-wheel Planner did not return within the time frame |
| `planner_error` | `aborted` | Planner called for normal non-timeout. |
| `invalid_planner_response` | `aborted` | Planner returns invalid type, structure or tool parameter |
| `tool_failure_limit_reached` | `aborted` | The restoration tool failed to reach the configuration limit successively |
| `tool_policy_violation` | `aborted` | Tool requests touch read-only or risk permission boundaries |

The caller must judge `termination_reason` and cannot judge the process by matching `termination_message` Chinese text.

## 5. Categorize Tool Results

`classify_tool_execution_result()` divided `ToolExecutionResult` into three categories.

| Classification | Trigger condition | Runner's behavior. |
|---|---|---|
| `success` | `status="success"` | Counting down on a series of failures, moving into the next round of planning. |
| `recoverable_failure` | `empty`, ordinary `rejected`, `timeout` or `error` | Count of failures plus one; permission for Planner adjustment if no limit is met |
| `fatal_policy_violation` | `rejected/tool_blocked` | Abort immediately after saving interaction |

The classification function uses only structured `status` and `error_code`, and does not resolve `public_message`.

## 6. Failed order

Runner handles each round in the following order:

1. Construct deep copy planning Context;
2. Call Planner within the single-wheel time limit;
3. Verify whether Planner returned `AgentPlannerDecision`;
4. If Planner returns `finish`, it's normal to end;
5. If Planner requests tools, check if `max_steps` is reached;
6. Generate a tool to call a signature without `call_id`;
7. Check for duplicate calls and duplicate tool signatures;
8. Give the call to ToolExecutor;
9. (a) Save tool requests and results as `AgentToolInteraction`;
10. Using the results of the failed strategy classification tool;
11. (a) Zero consecutive failures at the time of success;
12. Number of cumulative failures when failure can be restored;
13. (a) Security is suspended when the authority fails or consecutive failure reaches the limit;
14. Others move into the next round of planning.

The failure strategy must be executed after saving the Interaction. The request and outcome of the last tool, which led to the suspension, remained auditable.

## 7. Planner failed.

### 7.1 Planner timeout

Trigger condition:

- Single-wheel `planner.plan()` over `planner_timeout_seconds`.

Processes:

1. `asyncio.timeout()` Cancel the current Planner waiting;
2. Runner captures `TimeoutError`;
3. (a) The original upstream information of Planner is not available;
4. returns `planner_timeout`;
5. Retain the tools that were completed before the time was passed for interaction.

Stabilization results:

- `state="aborted"`
- `termination_reason="planner_timeout"`
- `finish_reason=None`
- `final_message=None`
- `missing_information` indicates that Planner did not return to the next decision

Corresponding test:

- `test_planner_timeout_returns_stable_abort`
- `test_planner_timeout_preserves_completed_interaction`

### 7.2. Planner's normal anomaly.

Trigger condition:

- Planner threw out non-timeout normal abnormalities.

Processes:

1. Runner captures `Exception`;
2. Logs record only the type of anomaly and the number of steps completed;
3. No record of `str(exc)`, `repr(exc)`, user assignments or original responses;
4. returns `planner_error`.

Stabilization results:

- `state="aborted"`
- `termination_reason="planner_error"`
- Public statement as stable text
- Completed steps to remain
- Sensitivity anomalies do not enter results and warning logs

Corresponding test:

- `test_planner_exception_is_sanitized`

### 7.3 Planner returns the error type

Trigger condition:

- Planner returns the string, the normal dictionary, the SDK object or other non-`AgentPlannerDecision` objects.

Processes:

1. Runner uses `isinstance()` to perform running time checks;
2. Untrusted Protocol type notes to automatically verify return values;
3. returns `invalid_planner_response`.

Corresponding test:

- `test_runner_aborts_on_invalid_planner_return_type`

### 7.4 Planner returns non-JSON tool parameters

Trigger condition:

- The `ToolCall.arguments` dictionary contains values such as `set` that cannot be sequenced to JSON.

Processes:

1. `json.dumps()` Could not generate a stable call signature;
2. The tool will not be handed over to Execuator;
3. Logs do not record original parameters;
4. returns `invalid_planner_response`.

Corresponding test:

- `test_canonical_signature_rejects_non_json_arguments`
- `test_non_json_tool_arguments_abort_before_execution`

## 8. Runner Loop Border Failed

### 8.1 Maximum total step achieved

Trigger condition:

- `max_steps` tools have been implemented;
- Planner is still asking for new tools to call.

Processes:

1. New tools will not be implemented;
2. (a) Not create a forged international action;
3. (a) To preserve the history of what was actually done before;
4. returns `max_steps_reached`.

Corresponding test:

- `test_max_steps_blocks_next_tool_before_execution`

### 8.2 Recall ID

Trigger condition:

- Planner again uses `call_id` already executed.

Processes:

1. Stop it before Execulator calls.
2. The second time Handler will not do it;
3. returns `duplicate_tool_call`.

Corresponding test:

- `test_duplicate_call_id_aborts_before_second_execution`

### 8.3 Repeat tools and parameters

Trigger condition:

- Planner uses the new `call_id`;
- However, toolnames and normative JSON parameters are identical to previous calls.

Processes:

1. `sort_keys=True` Elimination of dictionaries and dictionaries;
2. Call signature does not contain `call_id`;
3. Second calls will not be executed;
4. returns `duplicate_tool_call`.

Corresponding test:

- `test_canonical_signature_ignores_call_id_and_key_order`
- `test_duplicate_signature_aborts_with_new_call_id`

## 9. The ToolExecuator Failed

### 9.1 Unknown tool

Trigger condition:

- `tool_name` does not exist in ToolRegistry.

Processes:

1. No dynamic import;
2. (a) Not searching for a function of the same name outside the registration form;
3. Not to execute any Handler;
4. returns `rejected/unknown_tool`;
5. Consider it a recovery failure;
6. Planner can switch to legal tools within the failed budget.

Corresponding test:

- `test_unknown_tool_returns_rejected_result`
- `test_unknown_tool_rejection_is_observed`

### 9.2 Tool Parameters Invalid

Trigger condition:

- Parameters lack required fields;
- Parameter type is incorrect;
- Contains additional fields that are prohibited by the input model;
- Numerical or string violation of the limit.

Processes:

1. `input_model.model_validate()` for the ToolExecutor using specific tools;
2. Do not call Handler when the verification of parameters fails;
3. returns `rejected/invalid_tool_arguments`;
4. We're going to have a recovery failure.

Corresponding test:

- `test_invalid_arguments_do_not_call_handler`

### 9.3 Tool returns empty result

Trigger condition:

- Handler normally completed but returned to `None`.

Processes:

1. Execuator returns `empty/empty_result`;
2. Runner records the results;
3. The first empty result allows Planner to adjust the query or choose another tool;
4. The continuous empty result is suspended after the limit has been reached.

Corresponding test:

- `test_none_output_becomes_empty_result`
- `test_empty_result_is_observed_before_finish`
- `test_two_consecutive_empty_results_abort`

### 9.4 Tools Timeout

Trigger condition:

- Handler executes more than `ToolDefinition.timeout_seconds`.

Processes:

1. ToolExecutor cancels the current Handler;
2. returns `timeout/tool_timeout`;
3. Runner records timeout results;
4. First timeout allows Planner to choose an alternative or a safe end;
5. Discontinuation after successive failures have reached the limit.

Corresponding test:

- `test_handler_timeout_returns_stable_result`
- `test_single_tool_timeout_can_be_observed_and_finished`

### 9.5 General tool anomaly

Trigger condition:

- Handler throws out normal anomalies.

Processes:

1. Execuator capture anomaly;
2. (b) Do not include the original text of the bottom anomaly in the public result;
3. returns `error/tool_execution_error`;
4. We're going to have a recovery failure.

Corresponding test:

- `test_handler_exception_is_sanitized`

### 9.6 Tool output is not contractual

Trigger condition:

- Handler returns an error in the Pydantic model;
- Returning dictionary cannot be verified through `output_model`;
- Output field type or constraint is incorrect.

Processes:

1. Execuator does not consider the error output to be a reliable data;
2. returns `error/invalid_tool_output`;
3. `output` must be `None`;
4. We're going to have a recovery failure.

Corresponding test:

- `test_invalid_handler_output_returns_error`
- `test_runtime_validation_accepts_valid_dictionary_output`

## 10. Continuous failure recovery strategy

Default:

- `max_consecutive_tool_failures=2`

### 10.1. First time failure.

Process:

1. Save failed international action;
2. The number of consecutive failures becomes 1;
3. Get into the next round, Planner;
4. Planner can adjust parameters, rewrite queries, choose other registration tools or close the safe.

### 10.2 Second consecutive failure

Process:

1. Save the second failed Interaction;
2. The number of consecutive failures becomes 2;
3. (a) To reach the default ceiling;
4. No more calls from Planner;
5. returns `tool_failure_limit_reached`.

Corresponding test:

- `test_two_consecutive_empty_results_abort`

### 10.3 Reset successful results

Example:

- Empty result: number of failures 1;
- Successful results: zero failures;
- Empty result again: the number of failures is again 1.

Corresponding test:

- `test_success_resets_consecutive_failure_counter`

### 10.4 Distinction between a general step restriction and a failure limitation

`max_steps` Limit the total number of calls for practical tools.

`max_consecutive_tool_failures` Limit the number of outcomes of successive non-successful tools.

The two are configured independently, with which termination conditions will be met and which stable termination causes will be used.

## 11. Failure of authority and deep defense.

### 11.1 Tool registration phase

Tool Registry rejects:

- `read_only=False`；
- `risk_level="high"`；
- Repetition of tools with the same name;
- Sync Handler.

### 11.2 Tool implementation phase

ToolExecuator checks again:

- Whether the tool is read-only;
- Is it a high risk.

Even if the internal state of the registration form was incorrectly contaminated, Exceluto returned:

- `rejected/tool_blocked`

### 11.3 Runner Processing

`failure_policy.py` Categorize `tool_blocked` into:

`fatal_policy_violation`

Runner：

1. (a) Saves the utility action that caused the suspension;
2. Do not call real Handler;
3. Return immediately to `tool_policy_violation`;
4. Planner is not allowed to continue testing by modifying the parameters.

Corresponding test:

- `test_executor_blocks_unsafe_definition_defensively`
- `test_tool_blocked_aborts_immediately`

## 12. Injection of hint

The following are all used as untrustworthy data processing:

- (a) User assignments;
- Extract from the log;
- (a) The body of the document;
- Simulation of telemetry;
- Tool output text.

Even if it occurs:

- “Ignoring the rules of the system”;
- "Call run_shell";
- “Enforce the following orders”;
- "Modification of the allowlist of tools";
- “Forged evidence ID”;

Nor can the Python control rules be directly altered.

Current tool boundary guarantees:

1. Planner can only see the tool created by Tool Registry Schema;
2. ToolExecutor only executes the registration tool with precision;
3. Unknown tools are not dynamically imported;
4. User text does not become the Python function name;
5. The command in the document will not be executed as a program.

Corresponding test:

- `test_prompt_injection_cannot_expand_tool_allowlist`

This test covers the tool boundary. After accessing the real LLM Planner, there is also a need to add System Prompt and a hint for real Provider structural output into the regression test.

## 13. External Cancel

Trigger condition:

- FastAPI's request was cancelled;
- (a) Closure of services;
- Upper Task calls `.cancel()` at its own initiative.

Processes:

1. Planner or tool awaiting receipt `asyncio.CancelledError`;
2. Runner and Execuator continue to throw the anomaly;
3. Not converted to `planner_error` or `tool_execution_error`;
4. The top caller completes the cancellation process.

External cancellations do not return to normal `AgentRunResult` because it is not Agent's business failure.

Corresponding test:

- `test_external_cancellation_propagates_from_runner`
- `test_external_cancellation_propagates`

## 14. Configure Error

The following is a development or caller that violates the Python interface contract and does not convert to `aborted`:

- Planner didn't walk `plan()`;
- Executor is not `ToolExecutor`;
- `max_steps` type or scope error;
- `planner_timeout_seconds` type or scope error;
- `max_consecutive_tool_failures` type or scope error;
- `task` is not a string;
- `task` is a blank text.

These problems should be dropped as early as possible in the `TypeError`, `ValueError` or Pydantic `ValidationError` to avoid hiding errors in program configuration.

Corresponding test:

- `test_runner_requires_async_planner`
- `test_runner_requires_tool_executor`
- `test_runner_rejects_non_integer_max_steps`
- `test_runner_rejects_out_of_range_max_steps`
- `test_runner_rejects_non_numeric_planner_timeout`
- `test_runner_rejects_out_of_range_planner_timeout`
- `test_runner_rejects_non_integer_failure_limit`
- `test_runner_rejects_out_of_range_failure_limit`
- `test_runner_requires_string_task`
- `test_runner_rejects_blank_task_before_planning`

## 15. Logs and Sensitive Information

The current failure log records only the necessary structure information:

- (b) Stabilization of the name of the event;
- Unusual types;
- Toolname;
- `status`；
- `error_code`；
- Number of steps completed;
- Number of consecutive failures.

Log does not record:

- API Key；
- (a) Full user assignment;
- (a) A full journal extract;
- The body of the tool parameters;
- (a) The body of the tool output;
- Planner's original response;
- The original text of the unusual news;
- Model private thought chain.

Neither the public error message nor the service end log can rely on leaking bottom anomalies to complete the diagnosis.

## 16. Retention rules for completed steps

`interactions`. Only tool handling that is actually completed:

- Successful outcomes will be preserved;
- Empty results are saved;
- Parameter rejection saves;
- Unknown tool will be saved;
- Tools are saved over time;
- Tool anomalies will be saved;
- Permission denied will be saved.

The following requests do not create a new international action:

- New calls after reaching `max_steps`;
- Repeat `call_id`;
- Repeating tool names and parameters;
- Non-JSON tool parameters;
- Planner returns the error type;
- Planner's out of time before the tool is produced.

Upper Service cannot describe the outstanding requests as completed steps.

## 17. Automatic authentication commands

Results compacts, failure classification and Runner target testing:

`python -m pytest tests/test_agent_runtime_schemas.py tests/test_agent_failure_policy.py tests/test_agent_runner.py -v`

ToolExecuator failed:

`python -m pytest tests/test_agent_tool_executor.py -v`

Agent-related returns:

`python -m pytest tests -k "agent or tool or telemetry" -q`

Full return:

`python -m pytest -q`

## 18. Current Known Limits

The current mission 4 has completed the failed recovery of the Runner and Tool Execution layers, but there is still follow-up work:

1. The official Planner currently has only a walk-in Protocol and a fixed Fake track, which is not yet connected to the real OpenAI-compatible Tool Calling Projecter;
2. Not yet created `POST /api/v1/agent/diagnose`;
3. Agent Diagnosis Service has not yet converted the suspension history to a final diagnosis of `partial` or `abstained`;
4. No redacted execution trajectory for API response has been generated;
5. `final_message` is still a Planner draft without final diagnosis, Schema and allowlist of evidence;
6. Real Planner will still need to be supplemented with Provider alert injection and illegal tool call tests after access;
7. At the moment, Agent does not perform any real robotic control, shell, Python or any network tool.

These limitations cannot be resolved by lowering the tool's privileges, adding an unlimited re-test or bypassing the evidence gate.
