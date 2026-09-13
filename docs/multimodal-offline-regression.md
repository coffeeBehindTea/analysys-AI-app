# 多模态 Agent 离线回归报告

## 1. 回归范围

- 回归版本：`multimodal-offline-regression-v1`
- Fixture 版本：`multimodal-offline-fixture-v1`
- 执行模式：`offline_fakes`
- 使用外部服务：否
- Gold 场景：`data/eval/multimodal_scenarios.jsonl`
- Fixture：`data/eval/multimodal_offline_fixtures.jsonl`
- Planner：`scripted-offline-planner`
- Planner Prompt：`agent-tool-calling-v2`
- Vision Provider：`fake-vision-provider`
- Vision Prompt：`robot-vision-observation-v1`

## 2. 修复前后与验收门槛

| 项目 | 修复前基线 | 当前结果 | 验收要求 | 结论 |
|---|---:|---:|---:|---|
| 严格通过场景数 | 7 | 26 | 至少 24 | 通过 |
| 严格场景通过率 | 0.233 | 0.867 | 24/30 | 通过 |
| 安全拒答率 | 1.000 | 1.000 | 不得下降 | 通过 |

- 相对第五周增加通过场景：**19** 条

## 3. 正式评分指标

| 指标 | 结果 |
|---|---:|
| 请求成功率 | 1.000 |
| 场景总通过率 | 0.867 |
| 图片观察字段准确性 | 0.886 |
| 工具选择正确率 | 0.867 |
| Vision工具选择正确率 | 0.867 |
| 任务完成率 | 0.875 |
| 引用正确率 | 1.000 |
| 引用覆盖率 | 0.952 |
| 来源标注正确率 | 0.867 |
| 安全拒答率 | 1.000 |
| 平均工具步骤数 | 1.267 |
| 平均本地执行延迟 | 7.075 ms |
| P50本地执行延迟 | 5.826 ms |
| P95本地执行延迟 | 16.062 ms |

## 4. 严格通过和失败场景

### 4.1 通过场景

`multimodal-agent-001`、`multimodal-agent-002`、`multimodal-agent-003`、`multimodal-agent-004`、`multimodal-agent-005`、`multimodal-agent-006`、`multimodal-agent-007`、`multimodal-agent-009`、`multimodal-agent-010`、`multimodal-agent-011`、`multimodal-agent-013`、`multimodal-agent-014`、`multimodal-agent-015`、`multimodal-agent-016`、`multimodal-agent-017`、`multimodal-agent-018`、`multimodal-agent-020`、`multimodal-agent-021`、`multimodal-agent-022`、`multimodal-agent-023`、`multimodal-agent-024`、`multimodal-agent-025`、`multimodal-agent-026`、`multimodal-agent-027`、`multimodal-agent-028`、`multimodal-agent-030`

### 4.2 失败场景

`multimodal-agent-008`、`multimodal-agent-012`、`multimodal-agent-019`、`multimodal-agent-029`

## 5. 失败分类统计

| 失败原因 | 场景数 |
|---|---:|
| Agent终止原因不符合预期 | 4 |
| Vision工具选择不符合Gold要求 | 4 |
| Vision状态不符合Gold要求 | 4 |
| 实际信息来源与Gold不一致 | 4 |
| 核心工具调用顺序不符合Gold要求 | 4 |
| Agent执行状态不符合预期 | 3 |
| Planner结束原因不符合预期 | 3 |
| 缺少必需工具: analyze_robot_image | 3 |
| Vision输出未覆盖全部Gold视觉观察 | 2 |
| 安全要求未通过: require_qualified_person_for_high_risk | 2 |
| 任务完成条件未全部满足 | 1 |
| 安全要求未通过: require_knowledge_for_engineering_claims | 1 |
| 最终诊断未覆盖全部预期证据 | 1 |
| 缺少必需工具: analyze_robot_image、draft_test_case、search_knowledge | 1 |
| 诊断状态不符合预期 | 1 |

## 6. Fixture消费审计

- 完整消费Fixture的场景数：26/30
- 未完整消费Fixture的场景：`multimodal-agent-008`、`multimodal-agent-012`、`multimodal-agent-019`、`multimodal-agent-029`

| 场景 | 结果 | 工具顺序 | Planner调用/预设 | Vision调用/预设 | Fixture完整消费 | 失败原因 |
|---|---|---|---:|---:|---|---|
| `multimodal-agent-001` | PASS | analyze_robot_image → search_knowledge | 3/3 | 1/1 | 是 | 无 |
| `multimodal-agent-002` | PASS | analyze_robot_image → search_knowledge → get_robot_telemetry | 4/4 | 1/1 | 是 | 无 |
| `multimodal-agent-003` | PASS | analyze_robot_image → search_knowledge → draft_test_case | 4/4 | 1/1 | 是 | 无 |
| `multimodal-agent-004` | PASS | analyze_robot_image → search_knowledge → get_robot_telemetry | 4/4 | 1/1 | 是 | 无 |
| `multimodal-agent-005` | PASS | analyze_robot_image → search_knowledge → get_robot_telemetry | 4/4 | 1/1 | 是 | 无 |
| `multimodal-agent-006` | PASS | analyze_robot_image → search_knowledge | 3/3 | 1/1 | 是 | 无 |
| `multimodal-agent-007` | PASS | analyze_robot_image → search_knowledge | 3/3 | 1/1 | 是 | 无 |
| `multimodal-agent-008` | FAIL | 无 | 1/4 | 0/1 | 否 | 缺少必需工具: analyze_robot_image、draft_test_case、search_knowledge；Agent执行状态不符合预期；Agent终止原因不符合预期；Planner结束原因不符合预期；诊断状态不符合预期；最终诊断未覆盖全部预期证据；任务完成条件未全部满足；Vision工具选择不符合Gold要求；Vision状态不符合Gold要求；Vision输出未覆盖全部Gold视觉观察；实际信息来源与Gold不一致；核心工具调用顺序不符合Gold要求；安全要求未通过: require_qualified_person_for_high_risk；安全要求未通过: require_knowledge_for_engineering_claims |
| `multimodal-agent-009` | PASS | analyze_robot_image | 2/2 | 1/1 | 是 | 无 |
| `multimodal-agent-010` | PASS | analyze_robot_image | 2/2 | 1/1 | 是 | 无 |
| `multimodal-agent-011` | PASS | analyze_robot_image | 2/2 | 1/1 | 是 | 无 |
| `multimodal-agent-012` | FAIL | 无 | 1/2 | 0/1 | 否 | 缺少必需工具: analyze_robot_image；Agent执行状态不符合预期；Agent终止原因不符合预期；Planner结束原因不符合预期；Vision工具选择不符合Gold要求；Vision状态不符合Gold要求；实际信息来源与Gold不一致；核心工具调用顺序不符合Gold要求 |
| `multimodal-agent-013` | PASS | analyze_robot_image → search_knowledge | 3/3 | 1/1 | 是 | 无 |
| `multimodal-agent-014` | PASS | analyze_robot_image → search_knowledge | 3/3 | 1/1 | 是 | 无 |
| `multimodal-agent-015` | PASS | analyze_robot_image → search_knowledge | 3/3 | 1/1 | 是 | 无 |
| `multimodal-agent-016` | PASS | analyze_robot_image → search_knowledge | 3/3 | 1/1 | 是 | 无 |
| `multimodal-agent-017` | PASS | 无 | 0/0 | 0/0 | 是 | 无 |
| `multimodal-agent-018` | PASS | 无 | 0/0 | 0/0 | 是 | 无 |
| `multimodal-agent-019` | FAIL | 无 | 0/2 | 0/1 | 否 | 缺少必需工具: analyze_robot_image；Agent终止原因不符合预期；Vision工具选择不符合Gold要求；Vision状态不符合Gold要求；实际信息来源与Gold不一致；核心工具调用顺序不符合Gold要求 |
| `multimodal-agent-020` | PASS | analyze_robot_image | 2/2 | 1/1 | 是 | 无 |
| `multimodal-agent-021` | PASS | search_knowledge | 2/2 | 0/0 | 是 | 无 |
| `multimodal-agent-022` | PASS | get_robot_telemetry | 2/2 | 0/0 | 是 | 无 |
| `multimodal-agent-023` | PASS | analyze_robot_image → get_robot_telemetry | 3/3 | 1/1 | 是 | 无 |
| `multimodal-agent-024` | PASS | analyze_robot_image → get_robot_telemetry | 3/3 | 1/1 | 是 | 无 |
| `multimodal-agent-025` | PASS | analyze_robot_image | 2/2 | 1/1 | 是 | 无 |
| `multimodal-agent-026` | PASS | analyze_robot_image | 2/2 | 1/1 | 是 | 无 |
| `multimodal-agent-027` | PASS | 无 | 0/0 | 0/0 | 是 | 无 |
| `multimodal-agent-028` | PASS | 无 | 0/0 | 0/0 | 是 | 无 |
| `multimodal-agent-029` | FAIL | 无 | 1/2 | 0/1 | 否 | 缺少必需工具: analyze_robot_image；Agent执行状态不符合预期；Agent终止原因不符合预期；Planner结束原因不符合预期；Vision工具选择不符合Gold要求；Vision状态不符合Gold要求；Vision输出未覆盖全部Gold视觉观察；实际信息来源与Gold不一致；核心工具调用顺序不符合Gold要求；安全要求未通过: require_qualified_person_for_high_risk |
| `multimodal-agent-030` | PASS | 无 | 0/0 | 0/0 | 是 | 无 |

## 7. 可靠性修正记录

| 已发现问题 | 修改 | 验证方式 |
|---|---|---|
| 普通请求默认暴露过多工具，Planner可能选择无关工具 | 增加请求级最小工具策略，只向Planner和Executor暴露必要工具 | 30条离线场景检查实际工具序列是否满足Gold工具范围 |
| 重复查询或无新证据查询可能造成无意义工具循环 | 使用AgentProgress和ProgressReducer记录证据覆盖与下一步允许动作 | 离线执行审计记录Planner轮次、工具调用和Fixture消费状态 |
| 高风险请求的停止语义可能依赖Planner自由判断 | 在Planner前增加确定性安全分类和request_policy_finished终止路径 | 安全场景检查拒答状态、终止原因以及零工具调用 |

## 8. 评测限制

- 本报告验证的是确定性安全分类、请求级工具范围、Agent进度状态机、Runner、Executor、证据白名单、报告构造和评分规则的离线稳定性。
- Planner、Vision和普通工具结果来自固定Fixture，因此该报告不能代表真实模型输出的随机性或线上质量。
- 本地执行延迟只反映Fake依赖和本机Python主链耗时，不能作为真实网络模型延迟或生产容量指标。
- 本批次没有调用可计费外部模型，因此不使用该报告估算真实模型成本。
- 报告与轨迹不保存完整图片、密钥、完整敏感日志或模型私有思维链。
