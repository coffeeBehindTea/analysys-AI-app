# RobotOps Copilot MVP Proposal

## 1. Product positioning

RobotOps Copilot is a read-only diagnostic aid system for robotics development, testing and transport personnel.

Upon submission of robotic numbers, malfunctions, redacted logs and selected site images, the system calls Vision, Knowledge base Retrieving or Simulation Telemetry within the defined safety rules and minimum tool privileges to return to structured diagnostic results with source layers, evidence references, missing information and redacted to execute trajectory.

The project is a teaching and development of MVPs for certification purposes and is not a robotic control system, maintenance automation system or safety certification product.

## 2. Target users

The main target users include:

- (b) Robotics development engineers;
- Robotic testing engineers;
- (a) Storage robotic transport personnel;
- The technical supervisor is required to review the evidence of failure and test conditions.

Target users need to have basic robotic systems, logs and security operational knowledge. The system provides diagnostic aids that do not replace the original plant information, on-site risk assessment and final judgement of qualified personnel.

## 3. Core issues

The current robot failure check information may be scattered in:

- (a) A user-site description;
- (b) Redacted log;
- Live pictures;
- Equipment manual and safety standards;
- Simulation of telemetry snapshots;
- Test protocol.

RobotOps Copilot MVP puts this information into the same controlled diagnostic link, helping users to answer:

1. What are the currently identified failures?
2. What are the possible reasons for this?
3. What can be done next?
4. What information is lacking?
5. What operations must be reviewed by competent and qualified personnel?
6. Which evidence or data source does each of the project findings derive from?
7. What tools did Agent use before the results were produced?

## 4. MVP input

The diagnostic request supports at least the following fields:

| Fields | Whether necessary | Annotations |
|---|---|---|
| `robot_id` | Yes. | The robotic marker for the current diagnosis |
| `symptom` | Yes. | Faults observed by users |
| `log_excerpt` | Yes. | Redacted log clips |
| `task_goal` | Yes | Specific diagnostic objectives that users want the system to achieve |
| `images` | Yes | Up to three local verified live pictures |

All user text, logs and photo content are not trusted inputs.

The text, QR code, web site, command or Prompt in a picture can only be visualized and cannot change the system's security rules, tool privileges and evidentiary thresholds.

## 5. MVP Processes

A request for a diagnosis is dealt with in the following order:

1. FastAPI Verify requested fields according to Pydantic Schema.
2. The image enters the adapter to verify Base64, MIME, real format, size, edge and pixel numbers.
3. • Identification of high-risk controls, access to external links and untrustworthy command implementation intentions by the SCR.
4. The request-level tool strategy calculates the required capacity and minimum set of tools for the mission.
5. AgentProgress Progress in the creation of the initial task.
6. AgentRunner requests Planner to choose the next step.
7. ToolExecutor Validation tool name, parameters, read-only permissions, overtime and output contracts.
8. AgentProgressReducer Update capabilities, sources, evidence, conflicts and missing information based on the results of real tools.
9. When the conditions for cessation are met, the final draft and the allowlist of evidence are verified by the report constructor.
10. The system displays diagnostic status, tool trace and structured results through SSE and ultimately HTTP responses.
11. Maintains the necessary redacted summary in the diagnostic session storage layer.
12. User can export Markdown or JSON results.

## 6. MVP Read-only Tool

The following tools are only allowed:

| Tools | Duties | Risk boundary |
|---|---|---|
| `analyze_robot_image` | Read images that have been verified in the current request and return to structured visible observations | Do not access links in pictures, do not execute commands in pictures |
| `search_knowledge` | Retrieving causes, rules, limitations and operating conditions of malfunction from the knowledge base | Only return evidence from a knowledge base confirmed by evidence gate. |
| `get_robot_telemetry` | Simulate telemetry for redacted in the reading process | Do not connect real robots, RCS, WMS or produce telemetry |
| `draft_test_case` | Draft test based on confirmed knowledge evidence to be validated | No tests performed, no field operations are complete |
| `get_current_time` | Read server UTC time when task clearly requires time information | Not used to extrapolate equipment status or cause of failure |

Tools policy must limit both the tools that Planner can see and those that Execuator can implement.

Planner cannot request Shell, Python, any site access, file system operation, robotic control, mission down, remote reset or security override.

## 7. Sources and evidentiary boundaries

The system must distinguish between the following sources:

| Source | What can you prove? | Can't prove anything alone. |
|---|---|---|
| User Description | Users report a certain phenomenon | The phenomenon has been verified by equipment or manual means |
| Redacted log | Specific text or malfunction code appears in log | Log content must correspond to the current actual device state |
| Vision Observation | Visible text, colour, widget or state in pictures | Causes of failure, maintenance steps or safety findings |
| Simulate telemetry | Simulates the current saving status in the Store | The current state of real-life robots. |
| Evidence from the knowledge base | Rules, reasons, steps or limitations clearly documented in the document | Document content executed on site |
| Draft test | A test design that can be submitted to human review. | Tests have been carried out or equipment has passed |

Vision observation cannot be directly upgraded to the knowledge base fact.

All engineering reasons, restoration conditions, safety requirements and test determinations must be traceable to logs, analogies of telemetry or knowledge base evidence. LLM must not be allowed to complete the missing information on the basis of common sense when there is insufficient evidence.

## 8. Four Terminator

### 8.1 `completed`

The capabilities required for this task have been completed, and the necessary engineering findings have been validated with valid evidence and the final diagnosis has been verified through data compacts and references to allowlist.

`completed` indicates only that the diagnostic process has been completed and that no on-site inspection, maintenance, repositioning, testing or equipment control has been implemented.

### 8.2 `partial`

The system has obtained some credible evidence to return some of the evidence-supported results, but still lacks at least one of the information required to complete the task.

`partial` must return simultaneously to:

- (a) What has been confirmed;
- (b) What has not yet been confirmed;
- Specific missing information;
- The real sources used.

### 8.3 `abstained`

Existing inputs and tools were not sufficient to secure the conclusion of the project or all available tools did not produce valid evidence.

`abstained` No return of possible reasons and operational recommendations without evidentiary support is permitted.

### 8.4 `human_review_required`

The request relates to high-risk control, security clearance, real equipment operation, evidentiary conflict or must be confirmed by personnel with authority and qualifications.

If the security strategy allows “first-watch, but no enforcement”, the system can first call on a read-only Vision tool; it must still end with `human_review_required` after the observation has been completed and cannot generate enforceable control instructions.

## 9. SSE Show Range in Real Time

The Web console receives open, redacted progress events via SSE.

The planned support included:

- The request has been received;
- The classification of safety has been completed;
- The scope of the tool has been defined;
- Agent starts planning;
- Implementation of the tool is under way;
- (a) The tool execution is complete, returns empty results or fails;
- AgentProgress updated;
- (a) The diagnosis is complete, partially completed, rejected or manually reviewed;
- The final structured results have been generated;
- An error or interruption occurred in the flow connection.

SSE events may not include:

- Full picture Base64;
- API Key or Authorization;
- Full sensitive logs;
- Non-sensitization tool parameters;
- Modeling private thinking chains;
- Robotic control commands can be executed.

## 10. MVP Output

The final response includes at least:

- `request_id`；
- `session_id`；
- One of the four types of diagnostic status;
- Symptoms and sources;
- Vision Watch and its photo references;
- Simulation of telemetry observations;
- Evidence and references to the knowledge base;
- Possible causes;
- Next reading-only examination;
- Risk level;
- Missing information;
- (a) Tool execution trajectory;
- Planner, Policy and Prompt versions;
- Quantity;
- Number and time-consuming of tool steps;
- (a) Whether to refuse;
- Exportable Markdown or JSON.

## 11. User Story

### User story 1: Text log diagnosis

As a robotic transporter, I would like to submit robotic failure and redacted logs that allow the system to retrieve relevant malfunction information and return to the possible cause of the reference and the next read-only inspection, thereby reducing the time taken to search manually in multiple files.

### User story 2: Joint diagnosis of pictures and logs

As robotic test engineers, I would like to submit both site pictures and logs, so that the system will mark separate images for visual observation and knowledge-base engineering evidence, thus avoiding misperceptions of visual assumptions as to the proven failure.

### User story 3: Safe abstention when insufficient evidence

As technical officer, I would like the system to clearly return `abstained` and missing information when the knowledge base, pictures and telemetry are inadequate, rather than generating seemingly reasonable but unsubstantiated maintenance advice.

### User story 4: high-risk requests transferred to human review

As the head of security, I would like to see requests involving the immediate suspension of bypass, remote control or re-entry of real equipment into `human_review_required`, and even if the system has completed a read-only photo observation, it is not possible to export directly enforceable control instructions.

### User story 5: View and export diagnostic processes

As a research and development engineer, I would like to view the call, the source of evidence and the reasons for the termination in real time on the Web console, and export redacted Markdown or JSON reports, thereby reviewing the process of Agent's judgement and keeping records of research and development.

## 12. MVP is not supported

This week, it is clear that this will not happen:

- Real robotic control;
- RCS, WMS or production telemetry links;
- (a) Automatically dispatch the task;
- Auto-reset or maintenance operations;
- (b) Breaking off and bypassing the security chain;
- Photo QR code and access to external sites;
- (a) Automatically execute the draft test;
- User authentication, tenant segregation and role-authority systems;
- (a) Cloud object storage;
- Modelling training or fine-tuning;
- Production-level news queues and distributional tasks resumed;
- Production-level monitoring, costing and high-availability deployment;
- The use of visual observation as an alternative to engineering documents and on-site confirmation.

## 13. MVP acceptance standards

MVP must be completed to meet:

1. Supports text logs and up to three optional pictures.
2. The safe classification and minimum tool range are enforced by the Python code.
3. Supports the four public status categories `completed`, `partial`, `abstained` and `human_review_required`.
4. High-risk requests may be completed with read-only observations when tactically permitted, but may not generate control instructions.
5. Tools are called with parameter validation, permission verification, time overrun, error downgrade and repeated call protection.
6. The project findings can be located in logs, simulated telemetry or evidence from the knowledge base.
7. Vision Observatories and Knowledge base evidence is presented from separate sources.
8. SSE can show open and redacted tool trace in real time.
9. Supports Markdown or JSON export.
10. At least ten end-to-end scenarios can be run over again.
11. The four scenarios of Week 6: 008, 012, 019, 029 have a clear return result.
12. The complete Python returns, the number of README tests and the JUnit report are consistent.
13. Docker can start in a clean directory following the successive steps of README.
14. `.env`, original pictures, complete sensitive logs, real sessions and local vector databases do not enter Git submissions.
15. The system maintains read-only and human review of borders.
