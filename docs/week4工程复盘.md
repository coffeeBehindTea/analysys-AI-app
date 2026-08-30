### week4工程复盘

#### agent 架构

核心部分有三个，分别是runner，planner，executor。

runner负责控制agent循环。从初始化，保存工具结果，调用tool executor，检查终止条件，返回最后结果。

planner则是LLM所在的部分，它读取问题和现有的信息，做出决策选择调用工具或结束循环。

executor负责调用具体的工具并校验调用和结果的合法性。

其他的模块还有保存工具白名单的 ToolRegistry，业务入口（router），以及充当“翻译”的格式转换模块等。

#### 完整流程

runner进行初始化，进入第一轮

planner读取用户问题，工具调用结果历史（当前为空）做出判断。（比如这里选择调用get_robot_telemetry）

LLM将结构化的调用请求给到runner

runner将请求给到executor

executor校验后执行工具并返回结果

runner记录工具结果，未触发终止条件则进入新一轮

planner读取工具（get_robot_telemetry）结果并进行新一轮决策。

runner在触发终止条件（如达到最大步数，planner认为可以结束，服务中断如用户断联）后停止循环。



#### 困难点

流程的推进还是挺顺利的，但是LLM输出的不确定性导致测试结果没法稳定。