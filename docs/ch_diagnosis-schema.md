# Robot Diagnostic Evidence API 数据契约

## 1. 文档目的

本文档说明 `POST /api/v1/diagnostics` 的请求、响应、证据引用、降级规则和风险边界。

该接口不是机器人控制器，也不是自动维修系统。它只根据公开或脱敏工程文档生成可追溯的结构化诊断建议；任何安全关键操作仍需由具备资质的人员依据原厂资料和现场流程确认。

## 2. 在调用链中的位置

```text
HTTP客户端
  → diagnostics Router：解析并校验DiagnosisRequest
  → DiagnosisService：构造检索查询并编排整个诊断流程
  → 查询归一化与deterministic-v2改写
  → 向量检索 + 关键词检索 + RRF融合
  → head-preserving-v1重排
  → hybrid-evidence-gate-v1门控
      → 拒绝：Python直接构造abstained报告，不调用LLM
      → 放行：OpenAIDiagnosisDraftProvider生成DiagnosisLLMDraft
  → DiagnosisReportBuilder：校验临时证据编号并映射真实Chunk ID
  → DiagnosisReport：执行最终跨字段与引用白名单校验
  → FastAPI序列化JSON响应
```

各模块的职责边界如下：

| 模块 | 输入 | 主要操作 | 输出 |
|---|---|---|---|
| `app/routers/diagnostics.py` | HTTP JSON、`Request.state.request_id` | 声明路由、触发FastAPI校验、调用Service | `DiagnosisReport` |
| `app/schemas/diagnostics.py` | Python字段值 | Pydantic字段校验和跨字段校验 | 可信的请求或最终报告对象 |
| `app/services/diagnosis_service.py` | `DiagnosisRequest`、Request ID | 编排检索、门控、LLM和降级路径 | `DiagnosisReport` |
| `app/services/diagnosis_prompts.py` | 请求与门控后的真实候选 | 为候选建立`E1`、`E2`临时编号并构造Prompt | LLM消息 |
| `app/services/diagnosis_generation.py` | LLM消息 | 调用OpenAI兼容接口、清理代码围栏、解析JSON | `DiagnosisLLMDraft` |
| `app/services/diagnosis_report_builder.py` | 草稿、门控结果、真实候选 | 执行证据白名单检查，将临时编号映射为真实Chunk | `DiagnosisReport` |

## 3. 请求契约

### 3.1 `DiagnosisRequest`

| 字段 | 类型 | 必填 | 约束 | 语义 |
|---|---|---|---|---|
| `robot_id` | `str` | 是 | 1～100字符 | 发生异常的机器人编号；不加入检索文本 |
| `symptom` | `str` | 是 | 1～2000字符 | 用户观察到的故障现象 |
| `log_excerpt` | `str` | 是 | 1～10000字符 | 已脱敏日志摘要，可包含错误码、型号和阈值 |
| `retrieval_scope` | `DiagnosisRetrievalScope \| null` | 否 | 见下节 | 限制本次检索允许访问的文档 |

所有字符串自动去除首尾空白，未声明字段会被拒绝。`robot_id`用于报告追踪，但不参与检索，因为具体实例编号通常不包含工程故障语义。

### 3.2 `DiagnosisRetrievalScope`

| 字段 | 类型 | 约束 | 语义 |
|---|---|---|---|
| `document_ids` | `list[str]` | 最多20项；每项必须是64位小写SHA-256；不可重复 | 按文档内容标识限定范围 |
| `source_files` | `list[str]` | 最多20项；每项1～255字符；不可重复 | 按不含服务器路径的文件名限定范围 |

显式提供 `retrieval_scope` 时，两个列表不能同时为空。两个字段同时存在时，内部范围过滤器使用组合条件限制检索，而不是在全库检索后再过滤Top-K。

请求示例：

```json
{
  "robot_id": "diagnostic-robot",
  "symptom": "机器人网络恢复后仍然没有继续执行任务",
  "log_excerpt": "ERR-NET-4001 heartbeat timeout exceeded 1500 ms; network recovered but task did not resume",
  "retrieval_scope": {
    "source_files": [
      "仓储机器人故障说明.txt",
      "京东无人仓场景-仓储机器人测试规程与判定标准.md"
    ]
  }
}
```

## 4. 响应契约

### 4.1 `DiagnosisReport`

| 字段 | 类型 | 约束与来源 |
|---|---|---|
| `request_id` | `str` | 来自Request ID Middleware，用于关联响应头和日志 |
| `prompt_version` | `str` | 实际使用的诊断Prompt版本；门控提前拒绝时为`not-invoked` |
| `gate_version` | `str` | 实际使用的证据门控版本 |
| `status` | `completed \| partial \| abstained` | 表示完整诊断、部分诊断或拒答 |
| `symptoms` | `list[DiagnosisSymptom]` | 1～20项，只来自用户描述或日志 |
| `evidence` | `list[DiagnosisEvidence]` | 最多10项，只能由Python从真实检索候选构造 |
| `possible_causes` | `list[DiagnosisCause]` | 最多10项，每项至少绑定一个当前报告中的Chunk ID |
| `next_checks` | `list[DiagnosisCheck]` | 最多10项，每项绑定证据并声明风险边界 |
| `risk_level` | `low \| medium \| high \| unknown` | 整份报告的最高风险等级 |
| `missing_information` | `list[str]` | 最多20项，说明形成更完整诊断仍缺少的信息 |
| `abstained` | `bool` | 供客户端快速判断系统是否拒答 |

### 4.2 症状契约

`DiagnosisSymptom` 包含：

- `description`：请求中的原始现象或日志摘要；
- `source`：只能是 `user_report` 或 `log_excerpt`。

症状属于用户输入，不被伪装成知识库事实，也不使用Chunk引用。

### 4.3 证据契约

`DiagnosisEvidence` 包含：

| 字段 | 含义 |
|---|---|
| `chunk_id` | 文档ID与Chunk序号组成的真实标识 |
| `document_id` | 原文件内容的SHA-256 |
| `source_file` | 不含服务器绝对路径的来源文件名 |
| `page_or_section` | 原始页码或章节位置 |
| `rank` | 本次最终RRF结果中的排名，从1开始 |
| `rrf_score` | RRF融合分数，只用于排序审计，不是概率 |
| `vector_similarity` | 余弦相似度；关键词独占候选可以为`null` |
| `excerpt` | Python从真实Chunk复制的正文，不由LLM生成 |

### 4.4 原因与检查项

`DiagnosisCause` 包含谨慎表述的 `description` 和至少一个 `evidence_chunk_ids`。同一个原因不能重复引用同一Chunk。

`DiagnosisCheck` 另外包含：

- `risk_level`；
- `requires_qualified_person`。

当 `risk_level="high"` 时，`requires_qualified_person` 必须为 `true`，否则Pydantic拒绝该对象。

## 5. 跨字段规则

最终 `DiagnosisReport` 必须满足以下关系：

1. `status="abstained"` 与 `abstained=true` 必须同时成立；其他状态必须对应 `abstained=false`。
2. 拒答报告不能包含 `possible_causes` 或 `next_checks`，但必须填写 `missing_information`。
3. 非拒答报告必须包含真实 `evidence`，并至少包含一项原因或检查项。
4. `partial` 必须说明缺失信息。
5. `completed` 不能同时声称仍有缺失信息。
6. `evidence` 中的Chunk ID和排名不能重复。
7. 原因和检查项引用的每个Chunk ID都必须存在于本次报告的 `evidence` 中。
8. 高风险检查必须标记为需要有资质人员确认或执行。

这些规则由Pydantic的 `field_validator` 和 `model_validator(mode="after")` 执行。字段校验器检查单个列表中的重复项；模型校验器在对象字段全部解析后检查跨字段关系。

## 6. LLM内部草稿与最终报告的信任边界

LLM只允许返回内部 `DiagnosisLLMDraft`：

```text
status
possible_causes[].description
possible_causes[].evidence_ids
next_checks[].description
next_checks[].evidence_ids
next_checks[].risk_level
next_checks[].requires_qualified_person
risk_level
missing_information
abstained
```

LLM只能选择 `E1`、`E2` 等临时编号，不能填写以下可信字段：

- 真实Chunk ID；
- 文档ID；
- 文件名；
- 页码或章节；
- RRF分数；
- 向量相似度；
- Request ID；
- Prompt和门控版本。

`DiagnosisReportBuilder` 建立如下映射：

```text
E1 → rank=1的真实HybridRetrievedChunk
E2 → rank=2的真实HybridRetrievedChunk
E3 → rank=3的真实HybridRetrievedChunk
```

如果LLM返回格式合法但本次不存在的 `E999`，Builder会抛出 `InvalidLLMResponseError`，随后服务返回稳定的降级报告，而不是把虚构编号发送给客户端。

## 7. 降级规则

| 触发条件 | 是否调用LLM | HTTP成功响应 | 证据处理 | 公开结果 |
|---|---:|---|---|---|
| 没有候选 | 否 | `200` | 不返回弱候选 | `abstained=true`，说明未检索到候选 |
| 组合门控不足 | 否 | `200` | 不返回未获准候选 | `abstained=true`，要求补充错误码、日志或现场状态 |
| LLM主动判断证据冲突或不足 | 是 | `200` | 保留候选供审计 | 不给出原因和操作，填写缺失信息 |
| LLM JSON非法、字段缺失或未知引用 | 是 | `200` | 保留门控已放行证据 | 固定的结构校验失败说明，不暴露内部响应 |
| LLM超时 | 是 | `200` | 保留门控已放行证据 | 固定的超时说明 |
| LLM上游错误 | 是 | `200` | 保留门控已放行证据 | 固定的服务不可用说明 |
| Embedding配置、超时或上游错误 | 否 | `503`、`504`或`502` | 无可用检索结果 | 由统一异常处理器返回错误响应 |

生成阶段的已知失败被转换成 `200 + abstained`，原因是客户端仍能获得结构稳定、可审计的诊断报告。程序错误，例如错误依赖返回普通字典，则不会被伪装成业务拒答。

## 8. 结构化诊断样例

下面是已通过真实检索和真实LLM链路验证的成功结果节选。为控制文档长度，`excerpt` 仅保留摘要；实际API返回真实Chunk正文。

```json
{
  "request_id": "04fd7664-9246-4b82-be96-1b84573bcaa9",
  "prompt_version": "diagnosis-evidence-v1",
  "gate_version": "hybrid-evidence-gate-v1",
  "status": "partial",
  "symptoms": [
    {
      "description": "机器人网络恢复后仍然没有继续执行任务",
      "source": "user_report"
    },
    {
      "description": "ERR-NET-4001 heartbeat timeout exceeded 1500 ms; network recovered but task did not resume",
      "source": "log_excerpt"
    }
  ],
  "evidence": [
    {
      "chunk_id": "5b11fafd10c2414a65e1fd6eed8d7a295ee707ebaa0da89e6470841887b7ab80:000022",
      "document_id": "5b11fafd10c2414a65e1fd6eed8d7a295ee707ebaa0da89e6470841887b7ab80",
      "source_file": "京东无人仓场景-仓储机器人测试规程与判定标准.md",
      "page_or_section": "section: 10.1 TEST-NET-001 调度心跳中断",
      "rank": 1,
      "rrf_score": 0.032522,
      "vector_similarity": 0.610440,
      "excerpt": "通信恢复后不自动继续旧任务；RCS/RMS完成状态核对并下发恢复命令后才能继续运行。"
    },
    {
      "chunk_id": "dd18de8c04d80f7805a8f70ad6c24857d0df55df8009dec2e3371c3ee50fc652:000003",
      "document_id": "dd18de8c04d80f7805a8f70ad6c24857d0df55df8009dec2e3371c3ee50fc652",
      "source_file": "仓储机器人故障说明.txt",
      "page_or_section": "section: ERR-NET-4001",
      "rank": 2,
      "rrf_score": 0.032522,
      "vector_similarity": 0.634444,
      "excerpt": "可能原因包括AP漫游失败、信号盲区或干扰、车载网关断电。"
    }
  ],
  "possible_causes": [
    {
      "description": "任务未继续可能是RCS/RMS尚未完成状态核对或尚未下发恢复命令。",
      "evidence_chunk_ids": [
        "5b11fafd10c2414a65e1fd6eed8d7a295ee707ebaa0da89e6470841887b7ab80:000022"
      ]
    }
  ],
  "next_checks": [
    {
      "description": "确认RCS/RMS是否完成状态核对并下发恢复命令。",
      "evidence_chunk_ids": [
        "5b11fafd10c2414a65e1fd6eed8d7a295ee707ebaa0da89e6470841887b7ab80:000022"
      ],
      "risk_level": "low",
      "requires_qualified_person": false
    },
    {
      "description": "车载网关供电和重启操作需由有资质人员确认。",
      "evidence_chunk_ids": [
        "dd18de8c04d80f7805a8f70ad6c24857d0df55df8009dec2e3371c3ee50fc652:000003"
      ],
      "risk_level": "high",
      "requires_qualified_person": true
    }
  ],
  "risk_level": "high",
  "missing_information": [
    "RCS/RMS是否已完成状态核对并下发恢复命令",
    "现场Wi-Fi信噪比实测值"
  ],
  "abstained": false
}
```

## 9. 相关实现与测试

- 数据契约：`app/schemas/diagnostics.py`
- LLM内部草稿：`app/schemas/diagnosis_llm.py`
- HTTP路由：`app/routers/diagnostics.py`
- 业务编排：`app/services/diagnosis_service.py`
- Prompt：`app/services/diagnosis_prompts.py`
- 生成与JSON解析：`app/services/diagnosis_generation.py`
- 引用映射与降级：`app/services/diagnosis_report_builder.py`
- Schema测试：`tests/test_diagnostics_schemas.py`
- 生成测试：`tests/test_diagnosis_generation.py`
- Service测试：`tests/test_diagnosis_service.py`
- API测试：`tests/test_diagnostics_api.py`

