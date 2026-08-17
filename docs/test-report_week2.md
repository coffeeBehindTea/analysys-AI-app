# Robot Knowledge Base API 测试报告

## 1. 测试结论

Robot Knowledge Base API 的最终自动化回归测试全部通过。

```text
测试节点数：244
通过：244
失败：0
错误：0
跳过：0
```

测试覆盖 Week 1 的故障分诊和 SSE 基线，以及 Week 2 的文档解析、切分、Embedding、ChromaDB、RAG 查询、证据引用、评测脚本和 RAG 与裸 LLM 对照实验。

核心结论：

- 应用模块可以正常导入；
- FastAPI 路由装配成功；
- 文档摄取链路通过；
- RAG 查询和低相似度拒答通过；
- 引用白名单与代码组装通过；
- 评测指标计算通过；
- 自动化测试不依赖真实 LLM、Embedding 服务或外部网络；
- 真实 API 和模型链路已通过单独的人工集成验证。

## 2. 测试时间与证据

最终验证时间：

```text
2026-08-17T11:26:09+08:00
```

机器可读证据：

```text
docs/pytest-results-week2.xml
tests：244
failures：0
errors：0
skipped：0
pytest记录耗时：3.766秒
```

说明：

- 被测试对象是 `tests/` 覆盖的 API 契约、业务服务、数据契约、摄取、检索、引用、评测和异常路径；
- 执行流程是 pytest 收集测试，然后用 Fake、Mock、临时文件和临时 Chroma 数据库运行离线测试，最后把每个测试用例和耗时写入 JUnit XML；
- 预期结果是 244 项通过、0 项失败、0 项错误、0 项跳过，并且测试过程不访问真实 LLM、Embedding 服务或正式向量库；
- `.pytest_cache` 只是 pytest 的本地加速缓存，可能保留已经删除的测试节点，因此不再用它证明测试数量；
- `docs/pytest-results-week2.xml` 是本次真实执行产生的可审计证据。

重新生成该报告时使用：

```powershell
python -m pytest `
  tests/ `
  -q `
  -p no:cacheprovider `
  --junitxml=docs/pytest-results-week2.xml
```

## 3. 测试环境

```text
操作系统：Windows
Python：3.11.15
虚拟环境：Conda analysys
```

核心依赖版本：

| 依赖 | 版本 |
|---|---:|
| FastAPI | 0.141.1 |
| Pydantic | 2.13.4 |
| HTTPX | 0.28.1 |
| pytest | 9.1.1 |
| OpenAI Python SDK | 2.53.0 |
| ChromaDB | 1.5.9 |
| pdfplumber | 0.11.10 |

依赖检查：

```powershell
python -m pip check
```

结果：

```text
No broken requirements found.
```

## 4. 最终执行命令

### Python 版本

```powershell
python --version
```

结果：

```text
Python 3.11.15
```

### 依赖完整性

```powershell
python -m pip check
```

结果：

```text
No broken requirements found.
```

### 生产代码语法编译

```powershell
python -m compileall `
  -q `
  app `
  scripts `
  main.py
```

结果：通过，无语法错误输出。

### 应用导入与路由装配

```powershell
python -c "from main import app; print('应用导入成功')"
```

结果：

```text
应用导入成功
```

已验证的主要路由：

```text
GET    /health
POST   /api/v1/triage
POST   /api/v1/triage/stream
POST   /api/v1/knowledge/documents
GET    /api/v1/knowledge/documents
DELETE /api/v1/knowledge/documents/{document_id}
POST   /api/v1/knowledge/query
```

### 全量测试

```powershell
python -m pytest `
  tests/ `
  -q `
  -p no:cacheprovider `
  --junitxml=docs/pytest-results-week2.xml
```

结果：

```text
244 passed in 3.78s
```

JUnit XML 记录耗时：3.766 秒。终端显示时间与 XML 记录时间口径略有不同，属于正常现象。

## 5. 测试隔离策略

自动化测试不调用真实外部服务。

```text
测试输入
  ↓
Fake / Mock / 临时目录
  ↓
被测试业务模块
  ↓
固定且可断言的结果
```

主要隔离手段：

### Fake Embedding

Fake Embedding 根据测试输入返回固定向量。

用于验证：

- 批处理；
- 文本与向量顺序；
- 向量缓存；
- 文档摄取；
- Query Embedding；
- 检索编排。

### Fake LLM

Fake LLM 返回固定的结构化回答或预设异常。

用于验证：

- Prompt 输入；
- JSON 解析；
- 证据编号；
- 超时和上游错误；
- 无效响应；
- RAG 回答和拒答。

### `MagicMock`

用于模拟同步对象、SDK 属性链和计时器。

### `AsyncMock`

用于模拟必须被 `await` 的方法，例如：

```text
client.embeddings.create()
client.chat.completions.create()
compare_rag_and_bare()
```

### `httpx.MockTransport`

在内存中模拟 HTTP 响应，不访问真实网络。

用于验证：

- 评测脚本；
- RAG 对照实验；
- HTTP 状态错误；
- Request ID；
- 超时转换；
- 响应契约错误。

### FastAPI 依赖覆盖

使用：

```text
application.dependency_overrides
```

将真实 Service 替换为 Fake Service。

这样 API 测试可以覆盖：

```text
HTTPX
→ FastAPI
→ Middleware
→ Router
→ Pydantic
→ Fake Service
→ HTTP响应
```

### `tmp_path`

pytest 为每条相关测试创建独立临时目录。

用于：

- 临时 PDF、Markdown 和 TXT；
- 临时上传文件；
- 临时 Chroma 数据；
- 临时 JSON/Markdown 报告。

测试不会修改正式的：

```text
chroma_data/
robot_knowledge_v4
data/source/
```

## 6. 测试文件与数量

| 测试文件 | 数量 | 主要测试对象 |
|---|---:|---|
| `test_api_contract.py` | 5 | 健康检查、请求校验、Request ID |
| `test_bare_llm.py` | 8 | 裸 LLM Prompt、SDK调用、异常与耗时 |
| `test_chunking.py` | 8 | 固定窗口、重叠、Chunk ID、元数据 |
| `test_citation_evaluation.py` | 12 | 单题引用匹配与汇总指标 |
| `test_citation_evaluation_schemas.py` | 10 | 引用评测 Pydantic 契约 |
| `test_compare_chunking.py` | 5 | 两套切分参数比较报告 |
| `test_compare_rag_vs_bare_llm.py` | 18 | 对照实验核心、报告、CLI和资源关闭 |
| `test_document_catalog.py` | 3 | 文档列表和删除业务 |
| `test_document_loader.py` | 15 | PDF、TXT、Markdown解析与错误 |
| `test_document_preparation.py` | 8 | 文件哈希、Loader路由和Chunk准备 |
| `test_document_service.py` | 4 | 完整摄取编排和失败路径 |
| `test_embedding_client.py` | 3 | Embedding配置和客户端工厂 |
| `test_embedding_service.py` | 3 | Embedding请求与响应校验 |
| `test_error_responses.py` | 5 | 应用异常到安全HTTP响应 |
| `test_evaluate_citations.py` | 14 | 引用评测HTTP脚本和报告 |
| `test_evaluation_data.py` | 4 | 20条Gold Question数据质量 |
| `test_evaluation_schema.py` | 5 | Gold Question与检索报告契约 |
| `test_in_memory_retriever.py` | 8 | 内存Top-K检索和排序 |
| `test_ingest_documents.py` | 3 | 批量摄取脚本 |
| `test_knowledge_answer.py` | 9 | RAG Prompt、答案解析和安全约束 |
| `test_knowledge_api.py` | 12 | 文档管理和查询HTTP API |
| `test_knowledge_config.py` | 6 | RAG配置范围与默认阈值 |
| `test_knowledge_dependencies.py` | 4 | FastAPI依赖装配与客户端关闭 |
| `test_knowledge_query_schemas.py` | 8 | 问答请求、响应、引用和Draft契约 |
| `test_knowledge_query_service.py` | 5 | 检索、拒答、LLM和引用白名单 |
| `test_knowledge_schemas.py` | 6 | 文档记录、列表和删除响应 |
| `test_rag_comparison_schema.py` | 5 | RAG对照实验报告契约 |
| `test_retrieval_baseline.py` | 7 | Embedding批处理和缓存 |
| `test_retrieval_evaluation.py` | 6 | 证据位置匹配和检索指标 |
| `test_retrieval_math.py` | 7 | 余弦相似度边界和非法向量 |
| `test_run_retrieval_baseline.py` | 5 | 真实基线脚本的纯函数部分 |
| `test_service_errors.py` | 2 | Week 1 LLM配置和超时转换 |
| `test_service_responses.py` | 6 | Week 1 LLM响应解析 |
| `test_stream.py` | 3 | SSE成功与失败事件序列 |
| `test_upload_staging.py` | 3 | 上传暂存、大小限制和清理 |
| `test_vector_store.py` | 9 | Chroma写入、检索、重复和删除 |
| **总计** | **244** | |

一条测试可能包含多项断言，因此测试数量不等同于断言数量。

## 7. 文档摄取测试

被测试模块：

```text
DocumentLoader
TextDocumentPreparationService
TextChunker
EmbeddingService
DocumentService
ChromaVectorStore
```

预期流程：

```text
文件
→ 类型和大小校验
→ SHA-256
→ 重复检查
→ Loader解析
→ SourceTextSegment
→ TextChunker
→ DocumentChunk
→ Embedding
→ EmbeddedChunk
→ ChromaDB
→ DocumentRecord
```

主要覆盖：

- 不存在的文件；
- 空文件；
- 不支持的扩展名；
- 超大文件；
- UTF-8 文本；
- PDF 无文本页面；
- Markdown 章节路径；
- TXT 章节；
- PDF 英文字符粘连恢复；
- Chunk 大小和 overlap；
- 全局 `chunk_index`；
- `content_hash`；
- 重复正文向量复用；
- Embedding 批处理；
- 重复文档；
- 摄取过程中原文件发生变化；
- 全部 Embedding 成功后才写入；
- DocumentRecord 和实际 Chunk 数一致。

预期成功结果：

```text
DocumentRecord.status = ready
DocumentRecord.chunk_count = 实际写入Chunk数
每个Chunk可以追溯到文件和位置
```

## 8. ChromaDB 测试

被测试模块：

```text
ChromaVectorStore
```

测试使用临时持久化目录和真实 Chroma 客户端，但不连接外部数据库。

主要覆盖：

- 创建和打开 Collection；
- Collection Embedding 模型元数据；
- 不允许混合不同 Embedding 模型；
- 文档存在检查；
- 文档与 Chunk 元数据一致性；
- 使用 `add()` 写入；
- 重复文档拒绝；
- Top-K 检索；
- Cosine Distance 转 Similarity；
- 排名从 1 开始；
- 非法距离拒绝；
- 缺失元数据拒绝；
- 按 `document_id` 删除全部 Chunk；
- 删除不存在文档。

预期检索流程：

```text
query_embedding
→ Chroma query
→ ids/documents/metadatas/distances
→ 字段数量校验
→ similarity = 1 - distance
→ DocumentChunk
→ RetrievedChunk
```

## 9. RAG 查询测试

被测试模块：

```text
KnowledgeQueryService
OpenAIKnowledgeAnswerProvider
Knowledge Prompt Builder
```

预期成功流程：

```text
KnowledgeQueryRequest
→ Query Embedding
→ Top-K检索
→ Top-1达到阈值
→ LLM生成KnowledgeAnswerDraft
→ evidence_id白名单校验
→ Python组装Citation
→ KnowledgeQueryResponse
```

主要覆盖：

- 空检索结果直接拒答；
- Top-1 低于阈值直接拒答；
- 拒答时不调用 LLM；
- Top-1 等于或高于阈值时调用 LLM；
- LLM 必须返回 `KnowledgeAnswerDraft`；
- 未知 Evidence ID 被拒绝；
- 引用元数据来自真实 Chunk；
- 拒答时 `citations=[]`；
- 回答时至少存在一条引用；
- `retrieval_ms` 非负；
- 保留“征求意见稿”等状态限定词规则；
- 保留安全禁止、强制和前置条件规则；
- 最小充分证据集合规则；
- 非法 JSON、空内容和空候选结果。

## 10. 知识库 API 测试

被测试路由：

```text
POST   /api/v1/knowledge/documents
GET    /api/v1/knowledge/documents
DELETE /api/v1/knowledge/documents/{document_id}
POST   /api/v1/knowledge/query
```

主要覆盖：

- 上传成功返回 `201`；
- 重复上传返回 `409`；
- 超大文件返回 `422`；
- 上传临时文件在成功和失败后清理；
- 文档列表有数据和空列表；
- 删除成功；
- 删除不存在文档返回 `404`；
- 非法 Document ID 返回 `422`；
- 正常查询返回 Answer 和 Citation；
- 无答案查询返回结构化拒答；
- 空白问题返回 `422`；
- LLM 超时返回 `504`；
- Request ID 出现在统一错误响应中。

测试不会启动真实 Uvicorn，而是使用 ASGI Transport 调用内存中的 FastAPI 应用。

## 11. 评测系统测试

### 检索评测

被测试模块：

```text
retrieval_evaluation.py
evaluate_retrieval.py
```

主要验证：

- 文件名匹配；
- 页码和章节标准化；
- Top-1 和 Top-3 命中；
- 多跳证据完整性；
- 无答案错误召回；
- Recall 指标；
- JSON 和 Markdown 报告。

### 引用评测

被测试模块：

```text
citation_evaluation.py
evaluate_citations.py
```

主要验证：

- 返回引用是否匹配 Gold 文件和位置；
- 正确引用数量；
- 不同预期证据去重；
- 引用正确率；
- 引用覆盖率；
- 可回答问题响应率；
- 完整证据回答率；
- 无答案正确拒答率；
- HTTP 失败和无效响应；
- 报告序列化；
- 命令行参数。

## 12. RAG 与裸 LLM 实验测试

被测试模块：

```text
bare_llm.py
rag_comparison.py
compare_rag_vs_bare_llm.py
```

预期流程：

```text
选择Gold Question
→ 调用RAG HTTP API
→ 调用裸LLM Provider
→ 记录两种耗时
→ 构造RagVsBareLLMReport
→ 写JSON和Markdown
```

主要覆盖：

- 默认 q014 和 q020；
- 指定题号顺序；
- 重复和不存在题号；
- 裸 LLM 不接收 Gold 和 RAG 证据；
- RAG HTTP 错误和 Request ID；
- HTTP 超时；
- RAG 响应契约校验；
- LLM客户端成功和失败时关闭；
- RAG HTTP客户端关闭；
- 实验失败时不写残缺报告；
- UTF-8 JSON；
- Markdown 中包含两组回答和引用；
- 终端摘要。

## 13. Week 1 回归测试

Week 2 沿用 Week 1 代码，因此继续覆盖：

- `/health`；
- Triage 请求校验；
- 非流式 LLM 分析；
- LLM 配置缺失；
- LLM 超时；
- 无效 LLM JSON；
- Request ID；
- SSE `meta → delta* → done`；
- SSE `error` 终止事件；
- 内部异常信息不泄漏。

这证明加入 RAG 后没有破坏原有故障分诊能力。

## 14. 真实集成验证

真实集成验证与自动化测试分开执行，因为它会使用网络、API Key 和真实本地知识库。

### 14.1 摄取

Collection：

```text
robot_knowledge_v4
```

结果：

| 文档 | Chunk 数 |
|---|---:|
| Cognex DataMan 260 Reference Manual | 98 |
| Victron Lithium Smart Battery Manual | 165 |
| 仓储机器人测试规程与判定标准 | 35 |
| 仓储机器人故障说明 | 4 |
| 工业移动机器人安全标准征求意见稿 | 53 |
| **总计** | **355** |

### 14.2 文档 API

已验证：

- 第一次上传返回 `201 Created`；
- 相同文件第二次上传返回 `409 Conflict`；
- 列表接口返回已摄取文档；
- 删除接口删除指定文档全部 Chunk；
- 删除后列表中不再出现该文档。

### 14.3 RAG 查询

已验证：

- 标准适用范围问题返回 `200`；
- 回答附带真实文件、页码、Chunk ID 和相似度；
- 实时遥测问题返回 `abstained=true` 和空引用；
- q014 返回两条多跳证据。

### 14.4 检索评测

最终 V4：

| 指标 | 结果 |
|---|---:|
| Recall@1 | 0.600 |
| Recall@3 | 0.850 |
| 无答案错误召回率 | 0.000 |

### 14.5 引用评测

阈值 `0.60` V2：

| 指标 | 结果 |
|---|---:|
| 引用正确率 | 1.000 |
| 引用覆盖率 | 0.400 |
| 可回答问题响应率 | 0.438 |
| 完整证据回答率 | 0.438 |
| 无答案问题正确拒答率 | 1.000 |

### 14.6 RAG 与裸 LLM

已验证：

- q014 RAG 返回项目专有的两条证据；
- q014 裸 LLM 产生与当前安全规程冲突的通用堵转建议；
- q020 RAG 通过阈值直接拒答；
- q020 裸 LLM 虽然拒答，但仍发生一次模型调用；
- Prompt V2 明确保留“不允许为了测试而真实堵转电机”。

## 15. 风险覆盖

| 风险 | 验证方式 |
|---|---|
| 原文件类型非法 | DocumentService测试 |
| 空文档或超大文件 | DocumentService和上传测试 |
| 重复文档产生额外费用 | 哈希和重复检测测试 |
| Chunk失去来源位置 | Chunk Schema与Loader测试 |
| Embedding数量错位 | EmbeddingService测试 |
| 向量包含NaN或维度异常 | Embedding和检索数学测试 |
| 混用Embedding模型 | Chroma Collection测试 |
| Chroma返回损坏结构 | VectorStore错误测试 |
| 无证据仍调用LLM | 低相似度拒答测试 |
| LLM伪造证据编号 | Evidence ID白名单测试 |
| LLM伪造文件名或页码 | Python引用组装测试 |
| 无答案问题错误返回引用 | Query Schema和Service测试 |
| LLM遗漏安全约束规则 | Prompt安全规则测试和q014 V2 |
| 上游超时 | SDK异常转换测试 |
| 配置缺失 | Client Factory测试 |
| 临时上传文件泄漏 | Upload Staging测试 |
| HTTP错误泄漏内部详情 | Exception Handler测试 |
| 客户端无法关联日志 | Request ID测试 |
| 评测公式实现错误 | 手工预期指标测试 |
| 实验失败仍写残缺报告 | Comparison CLI测试 |

## 16. 当前未覆盖范围

当前自动化测试尚未覆盖：

- 并发负载和吞吐量；
- 长时间运行稳定性；
- 多进程同时写入同一 Chroma Collection；
- 大规模知识库性能；
- 扫描 PDF OCR；
- 复杂表格和图片语义；
- 不同 Embedding 供应商的一致性；
- 不同 LLM 供应商完整兼容性；
- 网络抖动和限流下的真实重试；
- Prompt Injection 红队测试；
- 文件病毒扫描；
- 用户身份和文档权限；
- 多租户隔离；
- 真实 RCS、WMS 和遥测集成；
- 安全操作的自动执行；
- 正式前端；
- 生产部署和高可用。

## 17. 测试结果边界

`244 passed` 表示当前测试中声明的行为符合预期，不表示：

- 所有 PDF 都能正确解析；
- 所有自然语言问题都能正确回答；
- LLM 永远遵守 Prompt；
- 当前阈值适用于其他语料；
- 系统达到生产安全认证要求；
- 系统可以直接控制机器人。

自动化测试证明工程契约，Gold 评测衡量当前语料质量，真实集成验证确认外部系统能够连接。三者不能互相替代。

## 18. 最终结论

Week 2 最终测试结果：

```text
244 passed
0 failed
```

已覆盖：

```text
文档解析
+ 文本切分
+ Embedding
+ Chroma持久化
+ Top-K检索
+ 阈值拒答
+ 证据约束回答
+ 代码引用
+ 文档管理API
+ 检索评测
+ 引用评测
+ RAG与裸LLM实验
+ Week 1回归
```

当前实现达到 Week 2 学习目标，可以进入工程复盘和后续 RAG 优化阶段。
