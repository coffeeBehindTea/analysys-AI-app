# Week 3 测试与验收报告

## 1. 测试目标

本报告验证 Week 3 的两个核心结果：

1. 候选检索从纯向量基线演进为“关键词 + 向量 + RRF + 查询改写 + 头部保留重排”，并且不降低无答案安全性。
2. `POST /api/v1/diagnostics` 能输出受Pydantic约束、引用真实Chunk、具备稳定降级路径的结构化诊断报告。

报告同时确认 Week 1 的故障分诊和 Week 2 的文档管理、摄取、向量存储及知识库问答能力没有回归。

## 2. 测试环境

| 项目 | 值 |
|---|---|
| 操作系统 | Windows |
| Python | CPython 3.11.15 |
| pytest | 9.1.1 |
| pytest-asyncio | 1.4.0 |
| FastAPI | 0.141.1 |
| Pydantic | 2.13.4 |
| OpenAI SDK | 2.53.0 |
| ChromaDB | 1.5.9 |
| Gold问题 | 28 |
| 可回答问题 | 23 |
| 无答案问题 | 5 |
| 当前预期证据 | 31 |

依赖版本全部固定在 `requirements.txt`。测试临时根目录由 `pytest.ini` 固定为项目内的 `.pytest_tmp`，避免Windows系统临时目录权限和锁定问题。

## 3. 最终离线回归

执行命令：

```powershell
python -m pytest --junitxml=docs/pytest-results-week3-final.xml
```

真实结果：

```text
collected : 798
passed    : 798
failures  : 0
errors    : 0
skipped   : 0
duration  : 18.74s
```

机器可读证据：

```text
docs/pytest-results-week3-final.xml
```

JUnit中的 `time` 表示测试执行耗时，不是报告生成时间。报告不保存生成日期或生成时间。

## 4. 离线隔离方式

完整pytest回归不访问真实LLM或Embedding服务，也不使用正式Chroma Collection。

| 测试替身或机制 | 替代对象 | 目的 |
|---|---|---|
| Fake Embedding | 真实Embedding API | 返回可预测向量，验证批处理、缓存和编排 |
| Fake LLM | 真实生成模型 | 返回成功、拒答、非法JSON、缺字段和超时等确定性结果 |
| `httpx.MockTransport` | 真实HTTP网络 | 验证SDK边界和异常转换 |
| FastAPI `dependency_overrides` | 正式依赖装配 | 为API注入Fake Service并验证HTTP契约 |
| `tmp_path`与临时Chroma目录 | 正式向量库 | 隔离写入、检索、删除和持久化测试 |
| `monkeypatch` | 环境变量或模块函数 | 控制配置和异常分支 |

Fake不是为了证明上游模型质量，而是为了只验证当前模块的责任。例如门控拒绝测试需要确认“LLM没有被调用”，使用带调用计数的Fake比真实网络请求更准确、更稳定。

## 5. 任务 7 必测场景

### 5.1 错误码精确召回

代表测试：

- `tests/test_hybrid_retrieval_gate.py::test_exact_identifier_in_top_k_can_use_lower_threshold`
- `tests/test_gated_hybrid_retrieval.py::test_exact_identifier_candidate_is_accepted`
- `tests/test_lexical_normalization.py`
- `tests/test_query_rewriting.py`

被测试流程：

```text
ERR_NET_4001或err saf 1002
→ 确定性归一化
→ ERR-NET-4001或ERR-SAF-1002
→ 关键词候选与向量候选
→ 门控识别exact_identifier_support
```

预期结果是错误码变体能够匹配规范形式，且精确标识符证据可以使用专门校准的门控路径。

### 5.2 融合排名与头部保留

代表测试：

- `tests/test_hybrid_retrieval.py::test_consensus_candidate_accumulates_both_rrf_contributions`
- `tests/test_hybrid_retrieval.py::test_rrf_ignores_vector_similarity_magnitude`
- `tests/test_head_preserving_reranking.py::test_vector_and_keyword_heads_preserve_complementary_evidence`
- `tests/test_head_preserving_reranking.py::test_existing_heads_keep_original_rrf_order`

被测试流程：

```text
向量排名 + 关键词排名
→ 按排名计算RRF贡献
→ 相同Chunk合并贡献
→ 按RRF排序
→ 保护RRF头部、向量头部和关键词头部
→ 去重并返回最终Top-K
```

预期结果是RRF只依赖各路径名次，不把余弦相似度或关键词原始分数错误当作同一量纲相加；重排后每个Chunk只出现一次。

### 5.3 门控放行与拒答

代表测试：

- `tests/test_gated_hybrid_retrieval.py::test_exact_identifier_candidate_is_accepted`
- `tests/test_gated_hybrid_retrieval.py::test_weak_candidate_is_rejected`
- `tests/test_diagnosis_service.py::test_gate_rejection_skips_llm_and_returns_stable_report`
- `tests/test_diagnosis_service.py::test_accepted_evidence_calls_llm_and_builds_traceable_report`

预期流程：

- 门控放行：调用LLM草稿Provider并组装报告；
- 门控拒绝：不调用LLM，直接返回 `abstained=true`；
- 空候选：使用 `no_candidates`；
- 弱组合支持：使用 `insufficient_combined_support`。

### 5.4 引用白名单

代表测试：

- `tests/test_diagnosis_prompts.py::test_system_prompt_requires_evidence_whitelist`
- `tests/test_diagnosis_report_builder.py::test_unknown_evidence_id_is_rejected`
- `tests/test_diagnostics_schemas.py::test_report_rejects_unknown_evidence_reference`
- `tests/test_hybrid_knowledge_query_service.py::test_unknown_evidence_id_is_rejected`

预期流程是LLM只能返回 `E1`、`E2` 等临时编号；Builder再映射真实Chunk。即使 `E999` 格式合法，只要不在本次候选中也必须拒绝。

### 5.5 非法结构与缺失字段

代表测试：

- `tests/test_diagnosis_generation.py::test_parser_rejects_invalid_json`
- `tests/test_diagnosis_generation.py::test_parser_rejects_missing_required_field`
- `tests/test_diagnosis_generation.py::test_parser_rejects_unsafe_high_risk_check`
- `tests/test_diagnosis_service.py` 中的稳定生成降级测试

预期结果是非法JSON、缺字段和不满足跨字段关系的结果不会穿过Schema；已知生成错误被转换成不暴露上游内部信息的稳定拒答报告。

### 5.6 冲突证据

代表测试：

- `tests/test_diagnosis_prompts.py::test_system_prompt_requires_abstention_on_conflict`
- `tests/test_diagnosis_llm_schemas.py` 中的拒答契约测试

预期结果是无法安全消解的证据冲突必须设置 `abstained=true`，原因和检查项为空，并在 `missing_information` 中说明仍需核实的内容。

### 5.7 无答案

代表测试：

- `tests/test_diagnosis_report_builder.py::test_no_candidate_gate_rejection_builds_stable_report`
- `tests/test_diagnostics_api.py::test_diagnostics_returns_abstained_report_with_200`
- `tests/test_hybrid_knowledge_query_service.py::test_rejected_gate_abstains_without_calling_llm`

预期结果是无答案或证据不足时不猜测；知识库问答返回空引用，结构化诊断返回原因和检查项为空的报告。

### 5.8 高风险建议

代表测试：

- `tests/test_diagnosis_llm_schemas.py::test_high_risk_draft_check_requires_qualification`
- `tests/test_diagnostics_schemas.py::test_high_risk_check_requires_qualified_person`
- `tests/test_diagnosis_report_builder.py::test_high_risk_check_preserves_qualification_boundary`

预期结果是 `risk_level="high"` 必须同时满足 `requires_qualified_person=true`。模型不能把高压、电池拆装、安全回路修改或车载设备供电操作描述成普通用户可以直接执行的指令。

## 6. 真实候选检索评测

该评测调用真实Embedding并读取本地Chroma，但不调用生成式LLM。

为了消除 `q021` Gold修订带来的分母差异，四份已保存候选报告已使用当前31份预期证据统一重评分。

| 策略 | Recall@1 | Recall@3 | Top-3完整证据召回 |
|---|---:|---:|---:|
| 纯向量基线 | 0.548（17/31） | 0.742（23/31） | 0.652（15/23） |
| 混合RRF | 0.516（16/31） | 0.742（23/31） | 0.652（15/23） |
| 混合RRF + 查询改写 | 0.613（19/31） | 0.871（27/31） | 0.826（19/23） |
| 混合RRF + 查询改写 + 重排 | **0.613（19/31）** | **0.968（30/31）** | **0.957（22/23）** |

最终候选方案相对纯向量基线完整证据召回提升 `0.305`，没有发生回归。

## 7. 真实门控评测

| 指标 | 结果 |
|---|---:|
| 可回答题门控放行率 | 0.957（22/23） |
| 完整证据可用率 | 0.913（21/23） |
| 无答案错误放行率 | 0.000（0/5） |
| 无答案正确拒答率 | 1.000（5/5） |

门控安全检查通过，因此最终候选方案获准接入在线API。

## 8. 真实知识库回答与引用评测

执行入口：

```powershell
python -m scripts.evaluate_citations --questions data/eval/gold_questions.jsonl --api-url "http://127.0.0.1:8000/api/v1/knowledge/query" --top-k 3 --timeout-seconds 120 --json-output docs/citation-evaluation-week3-balanced-rollback.json --markdown-output docs/citation-evaluation-week3-balanced-rollback.md
```

结果：

| 指标 | 结果 | 验收目标 | 结论 |
|---|---:|---:|---|
| 可回答问题响应率 | 0.870（20/23） | ≥ 0.70 | 通过 |
| 完整证据回答率 | 0.739（17/23） | ≥ 0.65 | 通过 |
| 引用正确率 | 0.962（25/26） | ≥ 0.95 | 通过 |
| 引用覆盖率 | 0.806（25/31） | 观察指标 | 已记录 |
| 无答案正确拒答率 | 1.000（5/5） | ≥ 0.95 | 通过 |

全部Week 3在线验收指标通过。

## 9. 诊断Prompt对照实验

实验固定相同请求、相同检索证据、相同模型和 `temperature=0.0`，只改变Prompt。

| 实验组 | Prompt版本 | 结构成功率 | 引用白名单通过率 |
|---|---|---:|---:|
| 基础Prompt | `diagnosis-basic-v1` | 0.000（0/1） | 0.000（0/1） |
| 证据优先Prompt | `diagnosis-evidence-v1` | 1.000（1/1） | 1.000（1/1） |

基础Prompt返回了违反 `completed` 状态契约的草稿；证据优先Prompt返回合法的 `partial` 报告，并正确引用两份真实证据。该实验只有一个案例，用于验证调用链和失败模式，不声称具有统计显著性。

## 10. 结构化诊断API实测

真实请求已验证：

- HTTP状态为 `200`；
- `prompt_version=diagnosis-evidence-v1`；
- `gate_version=hybrid-evidence-gate-v1`；
- `status=partial`；
- 两份证据均来自本次检索；
- 原因和检查项引用的Chunk ID均在返回的 `evidence` 中；
- 高风险网关操作标记 `requires_qualified_person=true`；
- 缺失信息被明确列出；
- 没有把候选证据改写成伪造来源。

完整字段语义和缩略样例见 `docs/diagnosis-schema.md`。

## 11. 验收清单

| 验收项 | 证据 | 结论 |
|---|---|---|
| 候选方案完整证据召回不低于纯向量 | 0.957 对 0.652 | 通过 |
| 可回答响应率≥0.70 | 0.870 | 通过 |
| 完整证据回答率≥0.65 | 0.739 | 通过 |
| 引用正确率≥0.95 | 0.962 | 通过 |
| 无答案正确拒答率≥0.95 | 1.000 | 通过 |
| 每个非空诊断结论可追溯 | Schema、白名单和Builder测试 | 通过 |
| 高风险建议有资质边界 | Pydantic跨字段规则和测试 | 通过 |
| 不少于25项自动化测试 | 798项 | 通过 |
| 三种以上候选策略对比 | 实际比较4种 | 通过 |
| 结构化诊断样例 | `docs/diagnosis-schema.md` | 通过 |
| 最终JUnit | `docs/pytest-results-week3-final.xml` | 通过 |

## 12. 遗留问题

- `q027` 的跨文档“设备清洁 + 工作站验收”第二份证据仍未进入Top-3。
- `q007` 的正确候选进入检索结果，但组合门控仍然拒绝。
- `q025` 在通用Prompt下会保守拒答严格等号边界问题。
- `q013`、`q026` 的答案内容基本完整，但没有选择Gold规定的全部文档。
- `q024` 多选了一份相关但不在Gold白名单中的手册页面。
- 单次真实LLM评测仍受外部服务变化影响；`temperature=0` 不等于服务端完全确定。
- 当前系统不是机器人控制器，不能自动执行任何诊断建议。

