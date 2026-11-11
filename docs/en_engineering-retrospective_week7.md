# Week 7 Engineering Retrospective

#### Event Definition

The acts of angent are defined as 9 events, namely:

1. `request_received`
2. `safety_classified`
3. `tool_scope_decided`
4. `planning_started`
5. `tool_started`
6. `tool_finished`
7. `progress_updated`
8. `diagnosis_finished`
9. `stream_error`

This allows the log to be sent to a browser to show each event the information it carries, such as a tool result.

#### Event stream

The single flow of events was observed by the observer and converted to payload, which was handed over to the publicsher for management of the diagnosis.

In the case of Agent Diagnosis SteamingService, new publicsher and observer are created every time they are called.



#### Harder.

It's a relatively smooth week, no obvious difficulties.
