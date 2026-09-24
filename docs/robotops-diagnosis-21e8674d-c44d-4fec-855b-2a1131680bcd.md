# RobotOps Copilot 诊断报告

> 本报告来自只读诊断系统。视觉观察不是知识库事实，测试草案投入使用前必须经过人工批准。

## 执行摘要

| 字段 | 值 |
|---|---|
| 请求 ID | 7deac7cb-c61f-4ab7-8c2d-06c386e67d80 |
| 会话 ID | 21e8674d-c44d-4fec-855b-2a1131680bcd |
| 诊断状态 | completed |
| 风险等级 | high |
| Agent状态 | completed |
| 终止原因 | planner_finished |
| 结束语义 | task_completed |
| Planner Prompt版本 | — |
| 证据门控版本 | agent-confirmed-evidence-v1 |

## 已报告症状

没有公开症状记录。

## 可能原因

- ERR-NET-4001 为心跳包超时（Heartbeat Loss）：AGV 与 5G/Wi-Fi 调度服务器失去通信连通性超过 1500ms 后触发。可能根因包括仓库 AP 漫游切换失败、信号盲区或高频电磁干扰、车载网关模块断电。
  - 证据：dd18de8c04d80f7805a8f70ad6c24857d0df55df8009dec2e3371c3ee50fc652:000003
- 通信中断超过 1500ms 后机器人进入原地制动或预先批准的安全停车状态，且通信恢复后不会自动继续旧任务，必须由 RCS/RMS 完成状态核对并下发恢复命令后才能继续运行——这解释了“网络已连接但机器人保持暂停”的现象。
  - 证据：5b11fafd10c2414a65e1fd6eed8d7a295ee707ebaa0da89e6470841887b7ab80:000022

## 后续检查

1. 观察机器人是否处于原地制动/安全停车状态，并确认其未自动恢复旧任务（当前模拟遥测显示 operational_state=paused、speed_mps=0.0、current_task_id=task-demo-1001，与安全停车后不自动续跑一致）。
   - 风险：medium
   - 需要有资质人员：否
   - 证据：5b11fafd10c2414a65e1fd6eed8d7a295ee707ebaa0da89e6470841887b7ab80:000022、dd18de8c04d80f7805a8f70ad6c24857d0df55df8009dec2e3371c3ee50fc652:000003
2. 检查现场 Wi-Fi 信号强度（信噪比 SNR），排查 AP 漫游切换失败、信号盲区或高频电磁干扰。
   - 风险：medium
   - 需要有资质人员：否
   - 证据：dd18de8c04d80f7805a8f70ad6c24857d0df55df8009dec2e3371c3ee50fc652:000003
3. 由 RCS/RMS 完成状态核对并下发恢复命令后，方可让机器人继续运行；如需通过手持终端局域网直连或重启车载无线网关，须由有资质人员执行。
   - 风险：high
   - 需要有资质人员：是
   - 证据：5b11fafd10c2414a65e1fd6eed8d7a295ee707ebaa0da89e6470841887b7ab80:000022、dd18de8c04d80f7805a8f70ad6c24857d0df55df8009dec2e3371c3ee50fc652:000003

## 缺失信息

当前结果没有声明缺失信息。

## 知识库引用

### 引用 1：仓储机器人故障说明.txt

- 位置：section: ERR-NET-4001
- Chunk ID：dd18de8c04d80f7805a8f70ad6c24857d0df55df8009dec2e3371c3ee50fc652:000003
- RRF分数：0.03252247488101534
- 向量相似度：0.6398411393165588

证据原文：

> 故障代码 (ErrorCode)
> ERR-NET-4001
> 故障名称
> 心跳包超时 (Heartbeat Loss)
> 触发条件 / 现象
> AGV 与 5G/Wi-Fi 调度服务器失去通信通信连通性超 1500ms。
> 可能原因 (Root Cause)
> 1. 仓库 AP 漫游切换失败；2. 信号盲区或高频电磁干扰；3. 车载网关模块断电。
> 排查与修复步骤 (Action Steps)
> 1. 观察机器人是否自动进入原地制动状态；2. 检查现场 Wi-Fi 信号强度（信噪比 SNR）；3. 手持终端通过局域网直连或重启车载无线网关。
> 严重等级
> 紧急 (Fatal)

### 引用 2：京东无人仓场景-仓储机器人测试规程与判定标准.md

- 位置：section: 京东无人仓公开场景下的仓储机器人测试规程与判定标准 > 10. 网络与调度测试 > 10.1 TEST-NET-001 调度心跳中断
- Chunk ID：5b11fafd10c2414a65e1fd6eed8d7a295ee707ebaa0da89e6470841887b7ab80:000022
- RRF分数：0.03252247488101534
- 向量相似度：0.5841712355613708

证据原文：

> ### 10.1 TEST-NET-001 调度心跳中断
> 
> #### 测试目的
> 
> 验证机器人与 RCS/RMS 的通信中断超过 1500 ms 后进入安全状态。
> 
> #### 测试步骤
> 
> 1. 下发一项无载移动任务。
> 2. 使用故障注入工具中断机器人与调度服务器的通信。
> 3. 保持中断超过 1500 ms。
> 4. 记录车体动作、本地告警和服务器状态。
> 5. 恢复网络，但不下发人工恢复命令。
> 
> #### 通过标准
> 
> - 超过 1500 ms 后上报或本地记录 `ERR-NET-4001`；
> - 机器人进入原地制动或预先批准的安全停车状态；
> - 通信恢复后不自动继续旧任务；
> - RCS/RMS 完成状态核对并下发恢复命令后才能继续运行。

## 视觉观察

### 图片 image_001

- 工具步骤：1
- 分析状态：completed
- 图片质量：clear
- 来源：vision_model
- 模型：deepseek-v4-flash-vision-exp
- Prompt版本：robot-vision-observation-v2
- 图片SHA-256：992426e7ba0bcb7683ed966081df37caf23475f7d91bc13fc93333b6a89d3367

可见观察：

- 画面为深色圆角面板，浅色背景，面板标题为“ROBOT STATUS PANEL”，文字清晰可辨。（visible_text，high）
- 面板内以左列标签、右列取值的形式排列三行状态信息：FAULT CODE、NETWORK、TASK。（visible_condition，high）
- 面板底部有一行较小的灰蓝色文字“SYNTHETIC TRAINING SAMPLE”。（visible_text，high）
- FAULT CODE：ERR-NET-4001（high）
- NETWORK：CONNECTED（high）
- TASK：PAUSED（high）

需要人工复核：是
无法确认：面板底部标注“SYNTHETIC TRAINING SAMPLE”，因此无法确认这些取值是否来自真实设备的实时状态画面。；图中未显示时间戳、设备编号或数据来源信息。

## 模拟遥测

| 机器人 | 来源 | 位置 | 状态 | 电量 | 速度 | 网络 | 当前任务 | 故障码 |
|---|---|---|---|---:|---:|---|---|---|
| robot-001 | simulated_memory | warehouse-demo/zone-a/aisle-07/node-14 | paused | 42.5% | 0 m/s | 已连接 | task-demo-1001 | ERR-NET-4001 |

## 测试草案

本次运行没有生成测试草案。

## 工具执行轨迹

| 步骤 | 工具 | 状态 | 输入摘要 | 结果摘要 | 耗时 | 错误码 |
|---:|---|---|---|---|---:|---|
| 1 | analyze_robot_image | success | 分析请求级图片；image_ref=image_001；analysis_goal_chars=21 | 取得结构化视觉观察；status=completed；image_quality=clear；observation_count=3；indicator_count=3；requires_human_check=true | 4.55 s | — |
| 2 | search_knowledge | success | 知识库检索；query_chars=33；top_k=3 | 知识库返回2条已确认引用；retrieval_ms=1360.506 | 1.36 s | — |
| 3 | get_robot_telemetry | success | 读取脱敏模拟遥测；robot_id=robot-001 | 取得脱敏模拟遥测；robot_id=robot-001；state=paused；fault_count=1 | 0.1 ms | — |
