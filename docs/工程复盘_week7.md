### week7 工程复盘

#### 事件定义

agent的行为被定义为9种事件，分别是：

1. `request_received`
2. `safety_classified`
3. `tool_scope_decided`
4. `planning_started`
5. `tool_started`
6. `tool_finished`
7. `progress_updated`
8. `diagnosis_finished`
9. `stream_error`

这样在把日志推送到浏览器的时候就可以对每个事件显示其所携带的信息如工具结果。

#### 事件流

单次事件流由observer观察，转换为payload之后交给publisher管理这次诊断。

而AgentDiagnosisStreamingService则会在每次被调用时创建新的publisher和observer。



#### 困难点

本周相对顺利无明显困难点