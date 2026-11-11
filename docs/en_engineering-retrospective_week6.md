# Week 6 Engineering Retrospective

#### Tool Limit

The minimum tool measurement has been added this week to avoid over-altering LLM tools or calling irrelevant ones. A tool to judge the need by matching a problem, and then to match it with a map to the real tool



#### Injection/advance issues

Security risk inputs are identified prior to the Planner call and high risk is submitted directly to human review.



#### Evidence-based progress status

Document what has been done, what has to be done and what is allowed to be done next. Optimizing the behaviour and unstable output of Planner through dynamic restraint tools and judgement.



#### Joined by fixture

fixture provides a fixed planner decision and tool result. Its role is to test the system framework itself separately from the uncertainties of real tools.



#### Session Record

Keep the complete execution track as a session record and write the json file after checking.



#### Add to Frontend Page

The page is divided into three parts, from the left to the right, and the left at the end is the diagnostic submission part, which fills in the task requirements, selects the picture, etc. The intermediate is the diagnosis, which includes the tool trace and results, the end state, etc. To the right is the session record that allows you to choose a history session.



#### Docker container sealed

The project can be replicated and distributed with the deployment of the docker container to enable certainty.



#### Harder.

It is difficult to judge the minimum range of tools and safety tests because it is not only a key word match but also a negative one; The most troublesome place was in the docker, where the installation was damaged and then encountered socket being taken over by residual processes. The whole thing was fixed up again.
