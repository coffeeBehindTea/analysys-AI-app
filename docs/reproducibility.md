# Week 3 可复现基线

记录日期：2026-08-17  
项目：Robot Knowledge Base API  
状态：Week 3 Day 1–2 任务一基线已验证

## 1. 文档目的

本文记录 Week 3 项目从干净代码快照开始，完成以下操作的真实过程和结果：

1. 创建新的 Python 虚拟环境；
2. 按锁定版本安装依赖；
3. 在没有旧 `.env`、Chroma 数据库和测试缓存的条件下运行离线回归；
4. 使用公开、脱敏的最小语料执行真实 Embedding 摄取；
5. 从新建的持久化 Chroma Collection 中检索回正确 Chunk；
6. 保存机器可读的 JUnit 测试证据。

这份基线用于区分后续问题究竟来自：

- 环境或依赖；
- 文档解析与切分；
- Embedding 服务；
- 向量存储；
- 检索、排序与门控；
- Week 3 新增代码。

在建立此基线前，不调整混合检索、重排或门控参数。

## 2. 基线结论

| 检查项 | 结果 |
|---|---|
| 干净代码快照可安装 | 通过 |
| Python 依赖关系完整 | 通过 |
| Week 2 继承测试 | 244/244 通过 |
| 最小脱敏语料摄取 | 1 个文件成功 |
| 文档解析位置数 | 1 |
| 生成 Chunk 数 | 1 |
| 查询向量维度 | 2048 |
| 持久化 Chroma 检索 | 正确召回目标 Chunk |
| 新增冒烟脚本测试 | 4/4 通过 |
| 冒烟脚本阶段全量测试 | 248/248 通过 |
| 新增 Gold 数据质量测试 | 11/11 通过 |
| 当前全量测试 | 259/259 通过 |
| 完整语料重新摄取 | 5 个文件、355 个 Chunk |
| 原 20 题纯向量基线复现 | Recall@1 0.600；Recall@3 0.850 |
| 扩展 28 题纯向量基线 | Recall@1 0.567；Recall@3 0.767 |
| 扩展基线无答案错误召回率 | 0.000 |
| 测试失败、错误、跳过 | 均为 0 |

## 3. 代码来源与边界

本地 Week 3 代码来自以下公开仓库 `main` 分支在 Week 2 合并完成后的 Download ZIP：

[https://github.com/coffeeBehindTea/analysys-AI-app](https://github.com/coffeeBehindTea/analysys-AI-app)

下载日期：2026-08-17。

ZIP 不包含 `.git`，因此本地目录没有分支和提交历史。本次记录使用依赖文件哈希、样例语料哈希、JUnit 和实际运行结果标识基线。

该方式可以得到干净代码快照，但不能像 commit SHA 一样永久标识某个 Git 提交。后续正式发布 Week 3 时，应记录对应提交 SHA 或创建版本标签。

初始目录明确不包含：

- `.env`；
- `chroma_data/`；
- `.pytest_cache/`；
- `__pycache__/`；
- Week 2 已生成的本地向量数据库；
- 未获再分发授权的完整原始资料。

初始 `data/source/` 只包含 `.gitkeep`。

## 4. 运行环境

| 项目 | 实际值 |
|---|---|
| 操作系统平台字符串 | `Windows-10-10.0.22631-SP0` |
| Python 实现 | `CPython` |
| Python 版本 | `3.11.15` |
| 环境管理工具 | Conda |
| Conda 环境名称 | `analysys-week3` |
| 依赖安装方式 | `python -m pip install -r requirements.txt` |

Python 返回的 Windows 平台字符串保留为真实执行值。`10.0.22631` 是当前 Windows NT 构建号。

## 5. 依赖基线

项目直接依赖在根目录 `requirements.txt` 中使用 `==` 锁定版本。

`requirements.txt` 的 SHA-256：

```text
361917cfab9b3e1c20d809a3e812002db3a8594325b4afe12c7e4b88d4528e39
```

创建环境和安装依赖：

```powershell
# 创建独立的Week3 Python环境。
conda create `
    --name analysys-week3 `
    python=3.11 `
    pip `
    -y

# 激活环境。
conda activate analysys-week3

# 安装requirements.txt中的全部依赖。
python -m pip install `
    -r requirements.txt

# 检查包依赖是否缺失或版本冲突。
python -m pip check
```

实际 `pip check` 结果：

```text
No broken requirements found.
```

`pip check` 证明当前已安装包之间的依赖约束完整，但不能替代项目业务测试。

## 6. 修改前离线回归基线

测试时没有创建 `.env`，也没有旧 Chroma 数据库。测试通过 Fake、Mock、依赖覆盖和临时目录与真实外部服务隔离。

Windows 默认 pytest 临时目录曾由另一个隔离身份创建，当前用户无法访问。为避免依赖全局临时目录权限，测试使用项目内专用目录。

先创建父目录：

```powershell
# pytest不会递归创建--basetemp的父目录，
# 因此必须先创建被.gitignore排除的tmp目录。
New-Item `
    -ItemType Directory `
    -Path "tmp" `
    -Force |
    Out-Null
```

修改前测试命令：

```powershell
python -m pytest `
    tests `
    -q `
    -p no:cacheprovider `
    --basetemp="tmp/pytest-week3" `
    --junitxml="docs/pytest-results-week3-baseline.xml"
```

参数说明：

- `-q`：减少逐条输出，保留结果和错误信息；
- `-p no:cacheprovider`：禁用 `.pytest_cache`；
- `--basetemp`：指定 pytest Fixture 使用的临时目录；
- `--junitxml`：生成机器可读的测试报告。

实际 JUnit 结果：

| 字段 | 值 |
|---|---:|
| tests | 244 |
| failures | 0 |
| errors | 0 |
| skipped | 0 |
| time | 3.504 秒 |

证据文件：

```text
docs/pytest-results-week3-baseline.xml
```

这 244 项测试覆盖继承自 Week 2 的主要能力，包括：

- FastAPI 请求与响应契约；
- Pydantic Schema；
- 中间件与统一异常响应；
- 文档解析、切分和哈希；
- Embedding 客户端和响应校验；
- Chroma 临时存储与检索；
- 文档上传、查询与删除接口；
- RAG 门控与引用白名单；
- 检索和引用评测脚本；
- RAG 与裸 LLM 对照脚本。

## 7. 公开最小语料

完整 Week 2 语料包含保留版权的厂商手册、标准草案和仅供本地学习的自编资料，因此不从公开仓库重新分发。

公开仓库只提供：

```text
samples/corpus/robot-fault-demo.txt
```

该文件是教学模拟语料，不对应真实客户、仓库、机器人型号或生产参数。

样例文件 SHA-256：

```text
b8866a03c60cfbb5d697d9910c01d2122cab822c7d4e5b7bb31a4848611ee617
```

复制到本地摄取目录：

```powershell
Copy-Item `
    -LiteralPath "samples/corpus/robot-fault-demo.txt" `
    -Destination "data/source/robot-fault-demo.txt"
```

复制前后 SHA-256 比较结果：

```text
True
```

该哈希同时成为文档摄取后的 `document_id`。

完整语料的来源、授权边界和文件哈希见：

```text
docs/corpus-sources.md
```

## 8. 本地配置

从公开模板创建本地配置：

```powershell
Copy-Item `
    -LiteralPath ".env.example" `
    -Destination ".env"
```

`.env` 只保存在本机，不提交，也不在本文记录任何 API Key。

本次最小摄取使用的非敏感配置：

| 配置 | 值 |
|---|---|
| Embedding 模型 | `embedding-3` |
| Chroma 持久化目录 | `chroma_data` |
| Chroma Collection | `robot_knowledge_week3_smoke` |
| Chunk 大小 | 800 |
| Chunk 重叠 | 120 |
| Embedding 批大小 | 64 |

摄取脚本不调用生成式 LLM，但会调用真实 Embedding 服务，因此需要有效的：

- `EMBEDDING_API_KEY`
- `EMBEDDING_BASE_URL`
- `EMBEDDING_MODEL`

## 9. 最小摄取验证

执行命令：

```powershell
python -m scripts.ingest_documents `
    --source-dir "data/source" `
    --output "docs/ingestion-log-week3-smoke.md"
```

摄取流程：

```text
扫描支持的文件
→ 校验扩展名和文件大小
→ 计算原文件SHA-256
→ 解析TXT章节
→ 生成SourceTextSegment
→ 按800/120切分DocumentChunk
→ 调用真实Embedding服务
→ 校验向量数量、索引、维度和数值
→ 写入Chroma Collection
→ 生成文档级摄取日志
```

实际结果：

| 项目 | 结果 |
|---|---|
| 文件总数 | 1 |
| 成功 | 1 |
| 跳过 | 0 |
| 失败 | 0 |
| source_file | `robot-fault-demo.txt` |
| document_id | `b8866a03c60cfbb5d697d9910c01d2122cab822c7d4e5b7bb31a4848611ee617` |
| 位置数 | 1 |
| Chunk 数 | 1 |
| 状态 | `ingested` |

本地摄取日志不包含 API Key、向量或 Chunk 正文，但仍作为运行过程文件由 `.gitignore` 排除。可公开结果汇总保存在本文。

## 10. 持久化检索验证

Week 2 的冒烟检索问题曾硬编码为完整本地语料中的 `ERR-DRV-3005`。Week 3 将脚本改为接收：

- `--query`
- `--top-k`

默认问题改为公开样例中的 `ERR-DEMO-1001`。

执行命令：

```powershell
python -m scripts.smoke_chroma_retrieval `
    --query "模拟故障代码 ERR-DEMO-1001 的观察现象、安全排查步骤和恢复条件是什么？" `
    --top-k 3
```

实际结果：

| 项目 | 结果 |
|---|---|
| 查询向量维度 | 2048 |
| 请求 Top-K | 3 |
| 实际召回数 | 1 |
| Top-1 相似度 | 0.603805 |
| 来源文件 | `robot-fault-demo.txt` |
| 位置 | `section: ERR-DEMO-1001` |
| Chunk ID | `b8866a03c60cfbb5d697d9910c01d2122cab822c7d4e5b7bb31a4848611ee617:000000` |

只返回一个结果是因为 Collection 中只有一个 Chunk。`ChromaVectorStore` 实际请求：

```text
min(top_k, stored_chunk_count)
= min(3, 1)
= 1
```

相似度由余弦距离转换：

```text
similarity = 1 - distance
```

`0.603805` 不是回答正确率或事实置信度，只表示当前模型向量空间中的接近程度。

这次冒烟验证证明了：

- `.env` 配置可以被读取；
- 真实查询 Embedding 可以生成；
- 查询与文档向量维度兼容；
- 持久化 Collection 可以重新打开；
- 正确 Chunk、正文和来源元数据可以取回。

因为知识库中只有一个 Chunk，本结果不能证明多候选排序质量，也不能代替完整 Gold 评测。

## 11. Week 3 第一处修改后的测试

新增文件：

```text
tests/test_smoke_chroma_retrieval.py
```

新增 4 项冒烟检索脚本离线测试，覆盖：

- 默认问题使用公开的 `ERR-DEMO-1001`；
- `--query` 和 `--top-k` 参数能够覆盖默认值；
- 空查询被拒绝；
- 非正整数 `top_k` 被拒绝。

随后新增：

```text
tests/test_week3_gold_questions.py
```

该文件包含 11 项扩展 Gold 数据质量测试，测试对象是正式评测数据：

```text
data/eval/gold_questions.jsonl
```

测试流程为：

```text
读取真实Gold JSONL文件
→ 通过load_gold_questions()逐行解析
→ 使用GoldQuestion执行Pydantic数据校验
→ 按question_id建立索引
→ 检查题量、类型分布、标签、对照关系和证据位置
```

这些测试具体检查：

- Gold 问题总数至少为 28；
- `q001`～`q028` 全部存在且编号连续；
- 单证据、多证据和无答案题的数量满足设计要求；
- 可回答题总数和预期证据总数满足要求；
- 新题覆盖故障码变体、产品型号、数值边界和多跳流程；
- 六组新旧对照题的问题文本不同，但预期证据存在交集；
- 预期证据中的来源文件名和页码或章节格式合法；
- `q028` 是“实体存在但关系不存在”的无答案题。

这些测试不会执行真实检索，也不会调用 Embedding、ChromaDB 或 LLM。它们验证的是“评测尺子本身是否正确”，而不是检索系统能否答对这些题。

当前全量测试命令：

```powershell
python -m pytest `
    tests `
    -q `
    -p no:cacheprovider `
    --basetemp="tmp/pytest-week3-full" `
    --junitxml="docs/pytest-results-week3.xml"
```

实际 JUnit 结果：

| 字段 | 值 |
|---|---:|
| tests | 259 |
| failures | 0 |
| errors | 0 |
| skipped | 0 |
| time | 3.683 秒 |

证据文件：

```text
docs/pytest-results-week3.xml
```

测试数量变化过程为：

```text
244项Week 2继承测试
→ 新增4项冒烟脚本测试
→ 248项全部通过
→ 新增11项Gold数据质量测试
→ 259项全部通过
```

因此，扩展 Gold 数据没有破坏原有文档处理、向量化、检索、RAG、知识库 API、引用评测及故障分诊功能。

## 12. Week 2 完整语料历史基线

下面的数据来自 Week 2 已保存的脱敏摘要，不是由单 Chunk 公开样例重新计算。

完整语料历史配置：

| 项目 | 值 |
|---|---|
| Collection | `robot_knowledge_v4` |
| Embedding 模型 | `embedding-3` |
| Gold 问题数 | 20 |
| 可回答问题 | 16 |
| 无答案问题 | 4 |
| Top-K | 3 |
| 在线门控阈值 | 0.60 |

检索历史指标：

| 指标 | 结果 |
|---|---:|
| Recall@1 | 0.600 |
| Recall@3 | 0.850 |
| Top-1 完整召回 | 8/16 |
| Top-3 完整召回 | 13/16 |
| 无答案错误召回率 | 0.000 |

在线引用评测历史指标：

| 指标 | 结果 |
|---|---:|
| 实际回答的可回答问题 | 7/16 |
| 可回答问题响应率 | 0.438 |
| 完整证据回答率 | 0.438 |
| 引用正确率 | 1.000 |
| 无答案正确拒答率 | 1.000 |

证据来源：

- `docs/retrieval-evaluation-summary.md`
- `docs/citation-evaluation-summary.md`
- `docs/rag-vs-bare-llm-summary.md`

公开最小样例只能复现代码、摄取和持久化检索链路，不能复现上述五份完整语料上的指标。

## 13. 交付边界

可以公开提交：

- 源码；
- 自动化测试；
- `requirements.txt`；
- `.env.example`；
- `.gitignore`；
- 公开脱敏样例；
- 本可复现说明；
- 不含正文和密钥的 JUnit 测试结果；
- 脱敏评测摘要。

只保留在本地：

- `.env`；
- API Key、令牌和账号；
- `chroma_data/`；
- `tmp/`；
- `__pycache__/`；
- `.pytest_cache/`；
- `data/source/` 中的实际语料副本；
- 含完整 Chunk 正文或长引用的原始评测报告；
- 未获再分发授权的厂商手册和标准资料。

`.gitignore` 负责阻止这些本地产物进入公开交付。

## 14. 当前限制

1. 代码来自 `main` 分支 ZIP，没有本地 commit SHA；后续正式交付应记录提交 SHA。
2. 最小样例只有一个 Chunk，不能测量候选排序质量。
3. 真实 Embedding 请求依赖网络、服务商可用性和账户余额。
4. Embedding 服务后续升级模型时，向量结果和相似度可能变化。
5. 更换 Embedding 模型后必须使用新的 Collection 并重新摄取。
6. 完整 Week 2 指标依赖五份本地语料，公开仓库不会重新分发这些文件。
7. JUnit 时间是本机单次结果，只用于记录，不作为稳定性能指标。

## 15. 评测顺序与当前检查点

在修改检索算法前，评测工作按照以下顺序执行：

```text
Week 2原20题历史基线
    ↓
逐题建立检索失败台账
    ↓
新增8条困难Gold问题
    ↓
在扩展Gold上运行纯向量基线B
    ↓
实现关键词与向量混合检索
    ↓
在相同输入条件下比较候选策略
```

当前完成状态：

- [x] 保存 Week 2 原 20 题历史基线；
- [x] 完成 q001～q020 逐题失败归因；
- [x] 新增 q021～q028 共 8 道困难题；
- [x] 完成扩展 Gold 数据质量测试；
- [x] 重新摄取五份完整本地语料；
- [x] 在 28 道 Gold 问题上运行纯向量基线 B；
- [ ] 实现关键词检索；
- [ ] 实现向量与关键词候选融合；
- [ ] 实现查询改写或重排候选方案；
- [ ] 在相同条件下比较三种检索策略；
- [ ] 根据检索结果校准在线门控。

纯向量基线 B 已经冻结。后续混合检索、查询改写或重排必须使用相同的：

- 五份语料；
- 28 道 Gold 问题；
- Embedding 模型；
- Chunk 参数；
- Top-K；
- 证据位置匹配规则。

否则指标变化可能来自输入差异，无法证明是检索策略产生的改进。

## 16. 扩展 Gold 纯向量基线 B

### 16.1 目的

基线 B 使用尚未加入关键词检索、融合排序或查询改写的纯向量检索实现。

它的职责是回答：

```text
面对扩展后的28道Gold问题，
当前纯向量检索究竟能找到多少正确证据？
```

后续所有候选策略都要与该结果比较。

### 16.2 完整语料重新摄取

没有复制 Week 2 的 `chroma_data`，而是从五份本地保留的语料重新执行：

```text
文档解析
→ SourceTextSegment
→ Chunk切分
→ Embedding
→ 写入新的Chroma Collection
```

摄取配置：

| 项目 | 值 |
|---|---|
| Collection | `robot_knowledge_week3_vector_baseline` |
| Embedding 模型 | `embedding-3` |
| Chunk 大小 | 800 |
| Chunk 重叠 | 120 |
| 文件数 | 5 |
| 成功 | 5 |
| 跳过 | 0 |
| 失败 | 0 |
| Chunk 总数 | 355 |

摄取日志：

```text
docs/ingestion-log-week3-vector-baseline.md
```

摄取日志只保存文件级统计、Document ID 和 Chunk 数，不保存向量、API Key 或 Chunk 正文。

### 16.3 评测配置

| 项目 | 值 |
|---|---|
| Gold 文件 | `data/eval/gold_questions.jsonl` |
| Gold 问题总数 | 28 |
| 可回答问题数 | 23 |
| 无答案问题数 | 5 |
| 预期证据总数 | 30 |
| Top-K | 3 |
| 相似度参考阈值 | 0.60 |
| 检索策略 | 纯向量检索 |

执行命令：

```powershell
python -m scripts.evaluate_retrieval --questions "data/eval/gold_questions.jsonl" --json-output "docs/retrieval-evaluation-week3-vector-baseline.json" --markdown-output "docs/retrieval-evaluation-week3-vector-baseline.md" --top-k 3 --similarity-threshold 0.60
```

该命令：

- 调用真实 Embedding 服务生成 28 个查询向量；
- 对每个查询向量执行本地 Chroma Top-3 检索；
- 不调用生成式 LLM；
- 根据 Gold 文件中的来源文件和页码或章节匹配证据；
- 生成机器可读 JSON 和人工可读 Markdown 报告。

### 16.4 总体结果

| 指标 | 结果 |
|---|---:|
| 可回答问题数 | 23 |
| 无答案问题数 | 5 |
| 预期证据总数 | 30 |
| Top-1 命中证据数 | 17 |
| Top-3 命中证据数 | 23 |
| Recall@1 | 0.567 |
| Recall@3 | 0.767 |
| Top-1 完整召回问题数 | 10/23 |
| Top-3 完整召回问题数 | 16/23 |
| 无答案错误召回数 | 0/5 |
| 无答案错误召回率 | 0.000 |

证据级 Recall 的计算为：

```text
Recall@1
= 17 / 30
= 0.5667
≈ 0.567

Recall@3
= 23 / 30
= 0.7667
≈ 0.767
```

“完整召回问题数”与证据级 Recall 不同。

对于包含两项预期证据的多跳问题：

```text
只召回一项证据
→ 证据级Recall可以增加
→ 但该问题不算完整召回
```

因此必须同时报告证据级和问题级指标。

### 16.5 原 20 题复现结果

从 28 题报告中单独统计 q001～q020：

| 指标 | 结果 |
|---|---:|
| 可回答问题数 | 16 |
| 无答案问题数 | 4 |
| 预期证据总数 | 20 |
| Recall@1 | 0.600 |
| Recall@3 | 0.850 |
| Top-1 完整召回 | 8/16 |
| Top-3 完整召回 | 13/16 |
| 无答案错误召回率 | 0.000 |

该结果与 Week 2 保存的历史基线一致。

这证明：

- 五份语料重新摄取成功；
- 文档解析和 Chunk 切分结果保持一致；
- 当前 Embedding 和 Chroma 配置能够复现历史检索指标；
- 总体 Recall 下降不是旧问题发生回归。

### 16.6 新增 8 题结果

q021～q028 子集包含：

- 7 道可回答题；
- 1 道无答案题；
- 10 项预期证据。

结果：

| 指标 | 结果 |
|---|---:|
| Recall@1 | 0.500 |
| Recall@3 | 0.600 |
| Top-1 完整召回 | 2/7 |
| Top-3 完整召回 | 3/7 |
| 无答案错误召回率 | 0.000 |

新增困难题暴露了：

- 故障码格式变体不能稳定精确召回；
- 跨语言手册页面排名不稳定；
- 数值与单位容易召回语义相关但用途不同的页面；
- 多跳问题经常只能找到一部分证据；
- 已找到完整证据的问题仍可能被 Top-1 阈值拒绝。

因此，28 题总体 Recall 下降来自新增困难题，而不是原 20 题退化。

### 16.7 相似度阈值的作用边界

本次离线检索评测中的 `0.60` 主要用于判断无答案问题是否发生错误召回：

```text
无答案题Top-1相似度达到或超过0.60
→ false_recall=true
```

对于可回答问题，评测脚本仍会保留低于 `0.60` 的 Top-K 结果，以便识别：

```text
正确证据已经进入Top-K
但在线门控仍会拒答
```

因此，不能把 `Recall@3` 直接理解为在线回答率。

在线回答还要经过：

```text
检索
→ 门控
→ LLM生成
→ 引用白名单
→ 响应Schema校验
```

### 16.8 报告文件与交付边界

本地完整报告：

```text
docs/retrieval-evaluation-week3-vector-baseline.json
docs/retrieval-evaluation-week3-vector-baseline.md
```

完整报告包含逐题检索结果、Chunk 元数据和正文预览，因此只保存在本地，不直接提交公开仓库。

可公开提交的结论应使用：

- 不含 Chunk 正文的汇总指标；
- 失败分类；
- 脱敏后的问题编号；
- 策略参数；
- 指标变化说明。

逐题失败归因见：

```text
docs/retrieval-failure-taxonomy.md
```

### 16.9 基线 B 结论

纯向量基线 B 的关键结果为：

```text
Recall@1：0.567
Recall@3：0.767
Top-3完整证据召回：16/23
无答案错误召回率：0.000
```

Day 3–4 的候选策略必须至少满足：

1. 完整证据召回率不低于该基线；
2. 解释 Recall@1 和 Recall@3 的所有变化；
3. 改进故障码、型号、数值和多跳问题；
4. 不让 q017～q020、q028 等无答案题被错误放行；
5. 不使用选择性样本代替完整 28 题评测。