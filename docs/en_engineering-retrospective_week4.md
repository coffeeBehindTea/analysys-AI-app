# Week 4 Engineering Retrospective

#### angent structure

There are three core parts: Runner, Planner, execuator.

Runner controls the angent loop. Starts from initialization, saves the tool results, calls the tool execuator, checks the termination conditions and returns the final result.

Planner is part of the LLM, which reads questions and available information, makes decisions about how to use tools or close the cycle.

Execuator is responsible for calling specific tools and verifying the legality of calls and results.

Other modules include ToolRegistry for storing tool allowlist, business input routers, and format conversion modules that serve as a translation hub.

#### Full Process

Runner's in the first round of initialization.

Planner reads user questions, and the tool calls the results history (currently empty) to judge. (e.g., here to choose the call for Get_robot_telemetry)

LLM gives structured call requests to runner

Runner's going to request execuator.

Execuator validation execution tool and return result

Runner records tool results, without trigger termination conditions entering a new round

Planner Reader (View) get_robot_telemetry Results and a new round of decision-making.

Runner stops the cycle after the termination condition has been triggered (if maximum step is reached, Planner thinks it can be terminated, if service is interrupted, if the user is disconnected).



#### Harder.

The process was moving well, but the uncertainty of LLM output led to a lack of stability in the test results.
