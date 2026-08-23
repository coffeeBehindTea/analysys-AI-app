# Robot Knowledge Base API

一个面向机器人研发场景的工程文档 RAG 与证据约束结构化诊断服务。

项目支持摄取公开或脱敏的 PDF、Markdown 和 TXT 文档，将文档切分、向量化并持久化到 ChromaDB；用户可以通过自然语言查询知识库，获得只基于检索证据生成的回答和可定位到原始文件、页码或章节的引用。Week 3 在此基础上增加确定性查询归一化、关键词与向量混合检索、RRF融合、头部保留重排、组合证据门控和结构化诊断API。

项目同时保留 Week 1 的普通故障分诊与SSE接口，以及 Week 2 的文档管理和知识库问答，用于比较普通LLM、RAG问答和证据约束诊断之间的职责差异。

> 本项目是 AI 应用工程学习项目，不是生产级机器人控制或安全认证系统。所有诊断、维修和安全操作仍需由具备资质的人员依据设备原厂资料确认。

> 当前开发阶段：Week 3 已完成。最终候选策略为 `deterministic-v2` 查询改写、向量与关键词双路Top-20、`RRF(k=60)`、`head-preserving-v1` 重排和 `hybrid-evidence-gate-v1` 门控。
>
> 在当前31份Gold证据口径下，纯向量Top-3完整证据召回为 `15/23=0.652`，最终候选为 `22/23=0.957`。最终在线评测达到：可回答响应率 `0.870`、完整证据回答率 `0.739`、引用正确率 `0.962`、无答案正确拒答率 `1.000`。完整离线回归为 `798 passed`。
>
> 最终依据见 [策略对比](docs/retrieval-strategy-comparison.md)、[诊断数据契约](docs/diagnosis-schema.md)、[Week 3测试报告](docs/test-report_week3.md) 和 [Week 3工程复盘](docs/工程复盘_week3.md)。

## 1. 核心能力

### 知识库文档管理

- 上传 PDF、Markdown、TXT 文档。
- 校验文件类型、大小和空文本。
- 使用文件 SHA-256 生成稳定的 `document_id`。
- 拒绝重复文档。
- 保存文档、Chunk 和 Embedding 元数据。
- 列出已摄取文档及 Chunk 数量。
- 按 `document_id` 删除文档产生的全部向量。

### RAG 问答

- 确定性归一化错误码、型号、数值和单位，并执行 `deterministic-v2` 查询改写。
- 同时执行Chroma向量检索和内存关键词检索。
- 使用RRF融合两路排名，并通过头部保留重排构造最终Top-K。
- 使用组合证据门控决定是否进入生成式LLM。
- 返回文件名、页码或章节、Chunk ID、排名、相似度和原文片段。
- Prompt 要求模型只根据当前证据回答。
- 引用由 Python 根据真实检索结果组装，不接受模型自造来源。
- 保留文档状态限定词和与问题直接相关的安全约束。

### 结构化诊断

- 提供 `POST /api/v1/diagnostics`。
- 输入机器人编号、现象、脱敏日志和可选文档检索范围。
- 输出状态、症状、真实证据、可能原因、下一步检查、风险、缺失信息和拒答标记。
- 每项原因和检查都绑定本次返回的真实Chunk ID。
- 高风险检查必须标记为需要有资质人员确认或执行。
- 门控拒绝、LLM非法输出、未知引用、超时和上游错误都有稳定降级结果。

### 评测与实验

- 28 条机器可读取的 Gold Question，其中包含 23 道可回答题和 5 道无答案题。
- 包含单证据、多证据、故障码变体、产品型号、数值边界、跨文档流程和知识库无答案问题。
- Week 3 新增 8 道困难题，用于检验关键词检索、融合排序和安全拒答策略。
- 检索评测：Recall@1、Recall@3、无答案错误召回率。
- 引用评测：引用正确率、覆盖率、完整证据回答率和正确拒答率。
- RAG 与裸 LLM 可复现实验。
- 使用 Fake Embedding、Fake LLM 和 Mock HTTP 完成无网络测试。

## 2. RAG 在整体 AI 应用架构中的位置

```mermaid
flowchart TD
    subgraph Ingestion["离线/管理链路：文档摄取"]
        Source["PDF / Markdown / TXT"]
        Loader["Document Loader<br/>解析正文与位置"]
        Chunker["Text Chunker<br/>分块与重叠"]
        EmbedDocument["Embedding Service<br/>文档向量化"]
        Chroma["ChromaDB<br/>向量、正文、元数据"]

        Source --> Loader
        Loader --> Chunker
        Chunker --> EmbedDocument
        EmbedDocument --> Chroma
    end

    subgraph Query["在线链路：问答与诊断共用证据层"]
        User["问题 / 故障描述"]
        Router["FastAPI Router"]
        Rewrite["归一化 + 查询改写"]
        Dual["向量Top-20 + 关键词Top-20"]
        Fusion["RRF融合 + 头部保留重排"]
        Gate{"组合证据门控"}
        Abstain["Python稳定拒答<br/>门控拒绝时不调用LLM"]
        AnswerLLM["问答或诊断LLM"]
        Validate["Pydantic + 引用白名单"]
        Response["问答响应 / DiagnosisReport"]

        User --> Router
        Router --> Rewrite
        Rewrite --> Dual
        Dual --> Chroma
        Chroma --> Fusion
        Dual --> Fusion
        Fusion --> Gate
        Gate -- 拒绝 --> Abstain
        Gate -- 放行 --> AnswerLLM
        AnswerLLM --> Validate
        Abstain --> Response
        Validate --> Response
        Response --> Router
    end
```

RAG 不负责训练模型，也不会把全部文档直接放进 Prompt。它把知识访问拆成两个阶段：

```text
检索：先从知识库找到最相关的少量证据
生成：再让LLM仅依据这些证据组织回答
```

这样设计的主要原因是：

- LLM 的训练知识可能过时或不包含组织内部规则。
- 全量文档通常超过模型上下文窗口。
- 直接依赖模型记忆无法提供可验证来源。
- 工业安全和故障处理不能允许模型随意补全参数。
- 无证据时应由业务代码拒答，而不是要求模型猜测。

## 3. 技术栈

| 依赖 | 主要职责 |
|---|---|
| `fastapi` | HTTP API、Router、依赖注入、OpenAPI |
| `uvicorn[standard]` | 运行 ASGI 应用 |
| `pydantic` / `pydantic-settings` | 数据契约、校验、环境配置 |
| `openai` | 调用 OpenAI 兼容的 LLM 和 Embedding API |
| `chromadb` | 本地向量持久化与相似度检索 |
| `pdfplumber` | 提取文本型 PDF 的逐页正文 |
| `python-multipart` | 解析 FastAPI 文件上传请求 |
| `httpx` | 异步 HTTP 客户端和 API 测试 |
| `sse-starlette` | Week 1 故障分诊 SSE 流式响应 |
| `pytest` | 自动化测试框架 |
| `pytest-asyncio` | 执行异步测试 |

## 4. 环境要求

- 推荐并已验证：CPython 3.11.15
- 当前本机验证平台：Windows NT 10.0.22631
- Conda 或其他 Python 虚拟环境
- Windows PowerShell、Bash 或其他终端
- 真实摄取和查询需要 OpenAI 兼容的 Embedding API
- 生成回答和故障分诊需要 OpenAI 兼容的生成式 LLM API

LLM 和 Embedding 可以使用不同的服务商、密钥、Base URL 和模型。

自动化测试通过 Fake、Mock、依赖覆盖和临时 Chroma 目录运行，不需要真实 API Key，也不应访问外部网络。

最终离线回归覆盖 Week 1～3 全部功能，共 `798` 项：

```text
tests    : 798
failures : 0
errors   : 0
skipped  : 0
JUnit    : docs/pytest-results-week3-final.xml
```

## 5. 创建环境并安装依赖

以下所有命令都必须在项目根目录执行。项目根目录应同时包含：

```text
main.py
app/
scripts/
tests/
requirements.txt
```

可以使用以下任一方式取得代码：

### 使用 Git 克隆

需要保留提交历史、创建分支或推送代码时使用：

```powershell
git clone `
  "https://github.com/coffeeBehindTea/analysys-AI-app.git"

Set-Location "analysys-AI-app"
```

### 下载 GitHub ZIP

只需要运行、检查或学习代码时，可以从 GitHub 下载 `main` 分支 ZIP 并解压。

ZIP 包含项目源码，但不包含 `.git`。因此它可以正常安装、测试、启动和摄取，只是不能直接执行 Git 分支、提交和推送操作。

进入实际项目根目录后执行：

```powershell
Test-Path -LiteralPath "main.py"
Test-Path -LiteralPath "app"
Test-Path -LiteralPath "scripts"
Test-Path -LiteralPath "tests"
Test-Path -LiteralPath "requirements.txt"
```

五条命令都应返回 `True`。

创建 Python 3.11 环境：

```powershell
conda create -n analysys python=3.11
```

激活环境：

```powershell
conda activate analysys
```

确认版本：

```powershell
python --version
```

安装全部依赖：

```powershell
python -m pip install -r requirements.txt
```

检查依赖关系：

```powershell
python -m pip check
```

完整的干净环境复现记录见：

- [Week 3 可复现基线](docs/reproducibility.md)

## 6. 配置环境变量

复制模板：

```powershell
Copy-Item .env.example .env
```

填写 `.env`：

```dotenv
# 生成式LLM
LLM_API_KEY=your-llm-api-key
LLM_BASE_URL=https://your-llm-provider.example/v1
LLM_MODEL=your-llm-model-name

# Embedding服务
EMBEDDING_API_KEY=your-embedding-api-key
EMBEDDING_BASE_URL=https://your-embedding-provider.example/v1
EMBEDDING_MODEL=your-embedding-model-name

# ChromaDB
CHROMA_PERSIST_DIRECTORY=chroma_data
CHROMA_COLLECTION_NAME=robot_knowledge

# 切分与批处理
RAG_CHUNK_SIZE=800
RAG_CHUNK_OVERLAP=120
EMBEDDING_BATCH_SIZE=64

# Top-1低于该值时直接拒答
RAG_SIMILARITY_THRESHOLD=0.60

# 单文件上限：20 MiB
MAX_DOCUMENT_SIZE_BYTES=20971520
```

重要说明：

- `.env` 可能包含真实 API Key，不得提交。
- LLM 和 Embedding 配置彼此独立。
- 修改 `.env` 后应重启 FastAPI。
- 同一个 Chroma Collection 中应使用相同的 Embedding 模型和向量维度。
- 更换 Embedding 模型后应使用新的 Collection 名称并重新摄取。
- `0.60` 是当前语料、模型和评测集上的校准结果，不是通用阈值。

## 7. 准备语料

将公开或经过授权、脱敏的文档放入：

```text
data/source/
```

支持：

```text
.pdf
.md
.txt
```

当前实验使用了五份语料：

1. Cognex DataMan 260 Series 公开参考手册。
2. Victron Energy Lithium Smart Battery 公开手册。
3. 工业移动机器人安全标准征求意见稿。
4. 自编、脱敏的仓储机器人故障说明。
5. 根据公开仓储场景编写的机器人测试规程与判定标准。

第 5 项是用于学习和评测的自编文档，不代表京东官方内部标准或实际作业规程。

公开仓库不会包含上述原始资料。厂商手册和标准草案可能保留版权，只能由使用者根据官方来源自行下载；两份自编资料也不从本地 `data/source/` 直接发布。仓库仅提供一个可公开分发的最小模拟语料，用于验证摄取链路：

```text
samples/corpus/robot-fault-demo.txt
```

将最小样例复制到本地语料目录：

```powershell
Copy-Item `
  "samples/corpus/robot-fault-demo.txt" `
  "data/source/robot-fault-demo.txt"
```

该样例只用于冒烟验证，不能复现五份完整语料上的 V4 评测指标。

详细来源、许可证和脱敏说明见：

- [语料来源说明](docs/corpus-sources.md)

只允许使用：

- 公开资料；
- 获得授权的资料；
- 完成脱敏的自编资料。

不得上传：

- 客户未公开数据；
- 个人信息；
- 未获授权的内部手册；
- 真实密钥、账号或访问令牌；
- 仍能识别具体客户或设备的数据。

## 8. 摄取文档

### 通过命令行脚本摄取目录

在一个新的空 Collection 中执行：

```powershell
python -m scripts.ingest_documents `
  --source-dir "data/source" `
  --output "docs/ingestion-log-week3-smoke.md"
```

终端显示“摄取日志已生成”只表示脚本已经结束，不代表每个文件都摄取成功。还必须打开日志并确认：

```text
文件总数：1；成功 1；跳过 0；失败 0
```


这里的“命令行脚本”表示不经过文档上传 HTTP API，并不表示可以断网运行。脚本不会调用生成式 LLM，但会调用真实 Embedding 服务，因此需要有效的 Embedding 配置、网络连接，并可能产生 API 费用。

内部流程：

```text
扫描文件
→ 校验类型和大小
→ 计算SHA-256
→ 解析正文和页码/章节
→ 切分Chunk
→ 批量Embedding
→ 写入ChromaDB
→ 生成摄取日志
```

如果当前 Collection 已包含相同文件，系统会根据文件哈希拒绝重复摄取。需要重新摄取时，应删除原文档或使用新的 Collection。

### 对持久化知识库执行检索冒烟验证

最小样例摄取成功后执行：

```powershell
python -m scripts.smoke_chroma_retrieval `
  --query "模拟故障代码 ERR-DEMO-1001 的观察现象、安全排查步骤和恢复条件是什么？" `
  --top-k 3
```

`--query` 指定需要生成 Query Embedding 的问题。

`--top-k` 指定最多返回多少个候选 Chunk。实际返回数量不会超过 Collection 中已有的 Chunk 数量。

当前最小样例的真实验证结果为：

```text
查询向量维度：2048
召回结果数：1
相似度：0.603805
来源文件：robot-fault-demo.txt
位置：section: ERR-DEMO-1001
Chunk ID：b8866a03c60cfbb5d697d9910c01d2122cab822c7d4e5b7bb31a4848611ee617:000000
```

因为该知识库只有一个 Chunk，这一步只能证明真实 Embedding、Chroma 持久化和来源还原正常，不能证明多候选排序质量。

### 通过 HTTP API 上传

启动服务后可在 Swagger UI 使用：

```text
POST /api/v1/knowledge/documents
```

也可以使用：

```powershell
curl.exe -i `
  -X POST `
  "http://127.0.0.1:8000/api/v1/knowledge/documents" `
  -F "file=@data/source/your-document.txt"
```

成功状态：

```text
201 Created
```

响应示例：

```json
{
  "document_id": "64位SHA-256",
  "source_file": "your-document.txt",
  "file_type": "txt",
  "file_size_bytes": 1024,
  "page_or_section_count": 1,
  "chunk_count": 2,
  "embedding_model": "your-embedding-model",
  "status": "ready"
}
```

上传 API 是同步摄取接口：只有解析、切分、Embedding 和 Chroma 写入全部成功后才返回 `201 Created`。

## 9. 启动 API

```powershell
python -m uvicorn main:app `
  --host 127.0.0.1 `
  --port 8000
```

本地开发时可以增加：

```text
--reload
```

服务地址：

```text
http://127.0.0.1:8000
```

健康检查：

```powershell
curl.exe -i `
  "http://127.0.0.1:8000/health"
```

Swagger UI：

- [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

OpenAPI JSON：

- [http://127.0.0.1:8000/openapi.json](http://127.0.0.1:8000/openapi.json)

所有响应都由 Middleware 添加：

```text
X-Request-ID
```

该 ID 可用于关联客户端错误响应和服务端日志。

## 10. API 接口

| 方法 | 路径 | 职责 |
|---|---|---|
| `GET` | `/health` | 检查 API 进程 |
| `POST` | `/api/v1/knowledge/documents` | 上传并摄取文档 |
| `GET` | `/api/v1/knowledge/documents` | 列出已摄取文档 |
| `DELETE` | `/api/v1/knowledge/documents/{document_id}` | 删除文档及其 Chunk |
| `POST` | `/api/v1/knowledge/query` | 检索并根据证据回答 |
| `POST` | `/api/v1/diagnostics` | 根据真实检索证据生成结构化诊断 |
| `POST` | `/api/v1/triage` | Week 1 普通故障分诊 |
| `POST` | `/api/v1/triage/stream` | Week 1 SSE 流式故障分诊 |

### 列出文档

```powershell
curl.exe -sS `
  "http://127.0.0.1:8000/api/v1/knowledge/documents"
```

响应：

```json
{
  "documents": [
    {
      "document_id": "64位SHA-256",
      "source_file": "document.pdf",
      "file_type": "pdf",
      "file_size_bytes": 1024,
      "page_or_section_count": 10,
      "chunk_count": 20,
      "embedding_model": "embedding-model",
      "status": "ready"
    }
  ]
}
```

### 删除文档

```powershell
curl.exe -i `
  -X DELETE `
  "http://127.0.0.1:8000/api/v1/knowledge/documents/DOCUMENT_ID"
```

成功响应：

```json
{
  "document_id": "DOCUMENT_ID",
  "deleted": true
}
```

### 查询知识库

推荐通过 Swagger UI 测试，也可以使用项目已经安装的 `httpx`：

```powershell
python -c "import json, httpx; response = httpx.post('http://127.0.0.1:8000/api/v1/knowledge/query', json={'question': '工业移动机器人安全标准征求意见稿适用于哪些生命周期阶段、行业和机器人类型？', 'top_k': 3}, timeout=120.0); print(response.status_code); print(json.dumps(response.json(), ensure_ascii=False, indent=2))"
```

成功回答示例：

```json
{
  "answer": "只根据检索证据生成的回答",
  "citations": [
    {
      "chunk_id": "document-id:000015",
      "document_id": "64位SHA-256",
      "source_file": "工业移动机器人安全标准.pdf",
      "page_or_section": "page: 5",
      "chunk_index": 15,
      "rank": 1,
      "similarity": 0.66,
      "excerpt": "支持回答的原始证据片段"
    }
  ],
  "retrieval_ms": 200.0,
  "abstained": false
}
```

证据不足时：

```json
{
  "answer": "知识库没有足够证据回答该问题。",
  "citations": [],
  "retrieval_ms": 150.0,
  "abstained": true
}
```

门控拒答路径不会调用生成式 LLM；门控放行后，LLM仍可以因为阅读到证据冲突或不足而执行二次拒答。

### 生成结构化诊断

推荐使用Swagger UI，也可以执行单行命令：

```powershell
python -c "import json, httpx; response = httpx.post('http://127.0.0.1:8000/api/v1/diagnostics', json={'robot_id': 'diagnostic-robot', 'symptom': '机器人网络恢复后仍然没有继续执行任务', 'log_excerpt': 'ERR-NET-4001 heartbeat timeout exceeded 1500 ms; network recovered but task did not resume'}, timeout=120.0); print(response.status_code); print(json.dumps(response.json(), ensure_ascii=False, indent=2))"
```

成功报告的核心结构：

```json
{
  "request_id": "request-id",
  "prompt_version": "diagnosis-evidence-v1",
  "gate_version": "hybrid-evidence-gate-v1",
  "status": "partial",
  "symptoms": [
    {
      "description": "机器人网络恢复后仍未继续任务",
      "source": "user_report"
    }
  ],
  "evidence": [
    {
      "chunk_id": "document-id:000022",
      "document_id": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      "source_file": "测试规程.md",
      "page_or_section": "section: 10.1",
      "rank": 1,
      "rrf_score": 0.0325,
      "vector_similarity": 0.61,
      "excerpt": "真实Chunk正文"
    }
  ],
  "possible_causes": [
    {
      "description": "只根据证据形成的谨慎原因",
      "evidence_chunk_ids": ["document-id:000022"]
    }
  ],
  "next_checks": [
    {
      "description": "只根据证据提出的检查",
      "evidence_chunk_ids": ["document-id:000022"],
      "risk_level": "low",
      "requires_qualified_person": false
    }
  ],
  "risk_level": "low",
  "missing_information": ["仍需补充的现场信息"],
  "abstained": false
}
```

完整字段、跨字段关系、证据白名单和降级规则见 [诊断数据契约](docs/diagnosis-schema.md)。

## 11. 数据与版本控制规则

以下内容已通过 `.gitignore` 排除：

```text
.env
data/source/*
data/processed/
chroma_data/
__pycache__/
.pytest_cache/
要求.txt
week*验收.txt
week1.txt
week2.txt
week3.txt
docs中的原始JSON和含长原文评测报告
```

目录职责：

| 路径 | 职责 | 是否提交 |
|---|---|---|
| `data/source/` | 原始 PDF、MD、TXT | 否，只保留 `.gitkeep` |
| `data/processed/` | 解析和切分中间结果 | 否 |
| `data/eval/gold_questions.jsonl` | 28 条评测问题及其预期证据位置 | 是 |
| `chroma_data/` | ChromaDB 向量和元数据 | 否 |
| `docs/` | 脱敏后的报告、架构和测试说明 | 只提交不含长原文的公开版本 |
| `.env` | 密钥和真实配置 | 否 |
| `.env.example` | 无密钥配置模板 | 是 |

原始评测报告可能包含完整 Chunk、`content` 或 `excerpt`，因此只保存在本地。公开仓库只提交不含原文的汇总报告。

## 12. 检索评测

检索评测使用真实 Embedding 和本地 ChromaDB，但不会调用生成式 LLM。

当前 Week 3 评测命令：

```powershell
python -m scripts.evaluate_retrieval --questions "data/eval/gold_questions.jsonl" --json-output "docs/retrieval-evaluation-week3-vector-baseline.json" --markdown-output "docs/retrieval-evaluation-week3-vector-baseline.md" --top-k 3 --similarity-threshold 0.60
```

评测流程：

```text
读取Gold Question
→ 批量生成Query Embedding
→ 对每个问题执行Chroma Top-3检索
→ 使用来源文件和页码或章节匹配预期证据
→ 计算证据级Recall
→ 计算问题级完整证据召回
→ 检查无答案错误召回
→ 生成JSON和Markdown报告
```

### Week 3 扩展 Gold 纯向量基线 B

评测配置：

| 项目 | 值 |
|---|---|
| Collection | `robot_knowledge_week3_vector_baseline` |
| Embedding 模型 | `embedding-3` |
| Chunk 参数 | `800/120` |
| 文档数 | 5 |
| Chunk 数 | 355 |
| Gold 问题数 | 28 |
| 可回答问题数 | 23 |
| 无答案问题数 | 5 |
| 预期证据总数 | 31 |
| Top-K | 3 |
| 相似度参考阈值 | 0.60 |

当前基线结果：

| 指标 | 结果 |
|---|---:|
| 证据级 Recall@1 | 0.548 |
| 证据级 Recall@3 | 0.742 |
| Top-1 完整证据召回 | 9/23 |
| Top-3 完整证据召回 | 15/23 |
| Top-3 完整证据召回率 | 0.652 |
| 无答案错误召回数 | 0/5 |
| 无答案错误召回率 | 0.000 |

证据级 Recall 使用全部预期证据作为分母：

```text
Recall@1
= 17 / 31
= 0.548

Recall@3
= 23 / 31
= 0.742
```

问题级完整证据召回要求一道题的所有预期证据都进入 Top-K。

对于两项证据的多跳问题，只找到其中一项时：

```text
证据级Recall会增加
但该问题不算完整证据召回
```

`similarity-threshold=0.60` 在离线检索评测中主要用于识别无答案问题的错误召回。它不会删除可回答题的低相似度 Top-K 结果，因此可以继续分析“证据已经召回，但在线门控仍会拒答”的问题。

### 原 20 题复现结果

从同一份 28 题报告中单独统计 q001～q020：

| 指标 | 结果 |
|---|---:|
| 可回答问题数 | 16 |
| 无答案问题数 | 4 |
| 预期证据总数 | 20 |
| 证据级 Recall@1 | 0.600 |
| 证据级 Recall@3 | 0.850 |
| Top-1 完整证据召回 | 8/16 |
| Top-3 完整证据召回 | 13/16 |
| 无答案错误召回率 | 0.000 |

该结果与 Week 2 的 `robot_knowledge_v4` 历史基线一致，说明新 Collection 成功复现了原有检索能力。

总体 Recall 下降来自新增困难题，不是原 20 题发生回归。

### 新增 8 题结果

q021～q028 子集结果：

| 指标 | 结果 |
|---|---:|
| 可回答问题数 | 7 |
| 无答案问题数 | 1 |
| 预期证据总数 | 11 |
| 证据级 Recall@1 | 0.455 |
| 证据级 Recall@3 | 0.545 |
| Top-1 完整证据召回 | 1/7 |
| Top-3 完整证据召回 | 2/7 |
| 无答案错误召回率 | 0.000 |

新增题主要暴露：

- 故障码格式变体不能稳定精确匹配；
- 产品型号简称会改变候选排名；
- 数值和单位容易召回用途不同的相关页面；
- 跨语言手册页面排名不稳定；
- 多跳问题经常只能召回一部分证据；
- 已有完整证据仍可能因 Top-1 低于阈值而被拒答。

详细逐题结果见：

- [检索失败台账](docs/retrieval-failure-taxonomy.md)
- [Week 3 可复现基线](docs/reproducibility.md)

本地完整报告：

```text
docs/retrieval-evaluation-week3-vector-baseline.json
docs/retrieval-evaluation-week3-vector-baseline.md
```

完整报告可能包含 Chunk 正文或预览，因此受 `.gitignore` 排除，不直接提交公开仓库。公开交付只保存脱敏汇总、策略参数和失败分类。

### Week 3 最终候选策略

四种候选策略已使用当前31份预期证据统一重评分：

| 策略 | Recall@1 | Recall@3 | Top-3完整证据召回 |
|---|---:|---:|---:|
| 纯向量基线 | 0.548 | 0.742 | 0.652（15/23） |
| 混合RRF | 0.516 | 0.742 | 0.652（15/23） |
| 混合RRF + 查询改写 | 0.613 | 0.871 | 0.826（19/23） |
| 混合RRF + 查询改写 + 头部保留重排 | **0.613** | **0.968** | **0.957（22/23）** |

最终策略相对纯向量基线完整证据召回提升 `0.305`。参数、逐题变化、门控安全和失败分析见 [候选检索策略对比](docs/retrieval-strategy-comparison.md)。

## 13. 引用评测

本节记录最终28题在线引用评测。该评测会调用真实Embedding、ChromaDB和生成式LLM，可能产生API费用。

FastAPI 运行时执行：

```powershell
python -m scripts.evaluate_citations --questions data/eval/gold_questions.jsonl --api-url "http://127.0.0.1:8000/api/v1/knowledge/query" --top-k 3 --timeout-seconds 120 --json-output docs/citation-evaluation-week3-balanced-rollback.json --markdown-output docs/citation-evaluation-week3-balanced-rollback.md
```

| 指标 | 结果 |
|---|---:|
| 可回答问题数 | 23 |
| 无答案问题数 | 5 |
| 实际回答的可回答问题数 | 20 |
| 引用正确率 | 0.962 |
| 引用覆盖率 | 0.806 |
| 可回答问题响应率 | 0.870 |
| 完整证据回答率 | 0.739 |
| 无答案问题正确拒答率 | 1.000 |

Week 3 的四项验收指标全部通过。逐题失败分为门控或模型拒答、少选Gold证据、多选相关页面和候选证据本身不完整四类，详见 [Week 3测试报告](docs/test-report_week3.md)。

## 14. RAG 与裸 LLM 对照实验

FastAPI 运行时执行：

```powershell
python -m scripts.compare_rag_vs_bare_llm `
  --question-id q014 `
  --question-id q020 `
  --top-k 3 `
  --timeout-seconds 120 `
  --json-output docs/rag-vs-bare-llm.json `
  --markdown-output docs/rag-vs-bare-llm.md
```

该实验使用同一个生成模型：

- RAG 组通过正式知识库 API 获得检索证据。
- 裸 LLM 组只获得问题和谨慎回答规则。
- 裸 LLM 不会看到 Gold 答案、预期证据或 RAG 响应。

实验观察：

- q014 中，RAG 找到两项项目专有证据并给出可追溯引用。
- 裸 LLM 给出了无当前语料支持的通用温度和恢复条件，并提出与测试规程冲突的堵转建议。
- q020 中，RAG 通过相似度门控直接拒答，不调用生成式 LLM。
- 裸 LLM 也谨慎拒答，但仍产生了一次模型调用。
- 初次实验发现 RAG 遗漏了“不得真实堵转”的安全禁令。
- 增强安全约束 Prompt 后，q014 V2 明确保留了该禁令。

公开报告：

- [RAG 与裸 LLM 脱敏摘要](docs/rag-vs-bare-llm-summary.md)

完整实验回答和原始引用只保存在本地，不进入公开仓库。

单次实验受模型随机性和网络延迟影响，只用于案例分析，不替代完整评测。

## 15. 自动化测试

运行全部测试并禁用 pytest 缓存：

```powershell
# pytest不会自动递归创建--basetemp的父目录。
New-Item `
  -ItemType Directory `
  -Path "tmp" `
  -Force |
  Out-Null

# 运行全部离线测试并生成最终Week3 JUnit报告。
python -m pytest --junitxml="docs/pytest-results-week3-final.xml"
```

当前源码的实际结果为 `798 passed`：

```text
tests    : 798
failures : 0
errors   : 0
skipped  : 0
time     : 18.74s
```
JUnit XML 是真实测试执行产生的机器可读证据，不能使用 `.pytest_cache` 中可能过期的节点列表代替。

核心测试使用：

- Fake Embedding；
- Fake LLM；
- `httpx.MockTransport`；
- FastAPI 依赖覆盖；
- pytest `monkeypatch`；
- 临时目录 `tmp_path`。

自动化测试不应：

- 访问真实 LLM；
- 访问真实 Embedding；
- 使用真实 API Key；
- 修改正式 Chroma Collection；
- 依赖外部网络。

主要覆盖：

- PDF、Markdown、TXT 解析；
- Chunk 边界、重叠和元数据；
- Embedding 批处理与缓存；
- 余弦相似度检索；
- ChromaDB 写入、检索和删除；
- 重复摄取和上传大小限制；
- Gold Question 数据契约；
- 检索与引用评测计算；
- Top-K 排名和来源；
- 错误码、型号、数值和单位归一化；
- 关键词检索、RRF融合和头部保留重排；
- 组合证据门控放行和拒答；
- 引用白名单映射；
- LLM 无效响应与超时；
- 知识库 API；
- 结构化诊断请求、报告、检索范围和跨字段契约；
- 诊断Prompt、JSON解析、冲突证据和稳定降级；
- 高风险检查的资质人员边界；
- RAG 与裸 LLM 实验和报告生成。
- 扩展 Gold 的题量、题型、标签、对照关系、预期证据位置和无答案边界；

## 16. 项目结构

```text
analysys-AI-app/
├── app/
│   ├── middleware/
│   │   └── request_id.py
│   ├── routers/
│   │   ├── diagnostics.py
│   │   ├── health.py
│   │   ├── knowledge.py
│   │   └── triage.py
│   ├── schemas/
│   │   ├── diagnosis_llm.py
│   │   ├── diagnostics.py
│   │   ├── evaluation.py
│   │   ├── knowledge.py
│   │   ├── knowledge_query.py
│   │   ├── retrieval.py
│   │   ├── retrieval_strategy.py
│   │   └── triage.py
│   ├── services/
│   │   ├── diagnosis_generation.py
│   │   ├── diagnosis_prompts.py
│   │   ├── diagnosis_report_builder.py
│   │   ├── diagnosis_service.py
│   │   ├── document_catalog.py
│   │   ├── document_loader.py
│   │   ├── document_service.py
│   │   ├── embedding.py
│   │   ├── gated_hybrid_retrieval.py
│   │   ├── head_preserving_reranking.py
│   │   ├── hybrid_retrieval.py
│   │   ├── hybrid_retrieval_gate.py
│   │   ├── keyword_retrieval.py
│   │   ├── knowledge_answer.py
│   │   ├── lexical_normalization.py
│   │   ├── query_rewriting.py
│   │   ├── query_rewriting_retrieval.py
│   │   ├── retrieval_scope.py
│   │   └── vector_store.py
│   ├── config.py
│   ├── dependencies.py
│   ├── errors.py
│   └── exception_handlers.py
├── data/
│   ├── eval/
│   │   └── gold_questions.jsonl
│   ├── processed/
│   └── source/
│       └── .gitkeep
├── samples/
│   └── corpus/
│       └── robot-fault-demo.txt
├── docs/
│   ├── corpus-sources.md
│   ├── architecture.md
│   ├── diagnosis-schema.md
│   ├── retrieval-strategy-comparison.md
│   ├── test-report_week3.md
│   ├── 工程复盘_week3.md
│   ├── reproducibility.md
│   ├── pytest-results-week3-baseline.xml
│   └── pytest-results-week3-final.xml
├── scripts/
│   ├── compare_diagnosis_prompts.py
│   ├── compare_retrieval_strategies.py
│   ├── evaluate_citations.py
│   ├── evaluate_hybrid_gate.py
│   ├── evaluate_retrieval_strategies.py
│   ├── ingest_documents.py
│   └── smoke_hybrid_retrieval.py
├── tests/
├── .env.example
├── .gitignore
├── main.py
├── README.md
└── requirements.txt
```

## 17. 已知限制

- `pdfplumber` 适用于含文本层的 PDF，不提供完整 OCR。
- 扫描版 PDF、复杂表格、多栏排版和图片文字可能解析不完整。
- PDF 英文空格恢复使用启发式规则，不能保证所有版式完全正确。
- 当前 ChromaDB 是本地单机持久化方案，不适合直接作为多用户生产数据库。
- 当前 API 没有用户认证、权限控制、速率限制和租户隔离。
- 文档上传采用同步摄取，大文件需要等待 Embedding 完成。
- 相似度阈值只对当前语料、Embedding 模型和 Gold 集有效。
- 最终候选仍未完整召回 q027 的跨文档“设备清洁 + 工作站验收”第二份证据。
- q007 的正确候选已经召回，但组合门控仍然保守拒绝。
- q025 在通用Prompt下会保守拒答严格等号边界问题。
- RAG 只适合当前静态知识库，不能回答实时位置、电量和订单。
- 实时机器人状态应通过 RCS、WMS、遥测服务或工具调用获取。
- Prompt 是软约束，模型输出仍需经过 Schema、代码规则和人工审核。
- 安全关键操作不能只依赖 LLM 回答执行。
- 当前没有正式前端、Agent、多用户系统或生产部署配置。
- SSE 只用于 Week 1 故障分诊，知识库查询当前使用普通 JSON 响应。

## 18. 进一步阅读

- [Week 3 可复现基线](docs/reproducibility.md)
- [语料来源说明](docs/corpus-sources.md)
- [系统架构](docs/architecture.md)
- [候选检索策略对比](docs/retrieval-strategy-comparison.md)
- [诊断数据契约](docs/diagnosis-schema.md)
- [Week 3自动化测试与验收报告](docs/test-report_week3.md)
- [Week 3工程复盘](docs/工程复盘_week3.md)
