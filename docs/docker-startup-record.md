# RobotOps Copilot Week 8 干净环境 Docker 复现记录

## 1. 复现目标

本记录验证 RobotOps Copilot Week 8 候选版本可以在不复用宿主机现有知识库和诊断会话的情况下完成以下闭环：

1. 校验 Docker Compose 配置；
2. 从受 `.dockerignore` 约束的源码上下文构建生产镜像；
3. 使用空 Chroma 数据卷和空会话数据卷启动 FastAPI；
4. 在空知识库状态下访问健康检查和 Web 控制台；
5. 通过文档上传 API 摄取公开演示语料；
6. 使用真实 Planner、Embedding 和检索链路完成一次只读 Agent 诊断；
7. 验证引用和脱敏诊断会话已经持久化。

本次验证没有把本地 355 个 Chunk 的历史知识库复制进隔离环境，也没有用 Fake Planner、Fake Embedding 或测试 Fixture 替代外部服务。

## 2. 验证对象

| 项目 | 实际值 |
|---|---|
| Compose project | `robotops-copilot` |
| Service | `robotops-copilot` |
| Image | `robotops-copilot:week8` |
| Image ID | `sha256:7082e12b48d6f9941fd37d4638bbf5d251b7e979ffa18639c672874bfec97187` |
| Image size | `236438650` bytes |
| Base image | `python:3.11-slim-bookworm` |
| Docker Engine | `29.7.2` |
| Container user | `appuser` |
| Container command | `python -m uvicorn main:app --host 0.0.0.0 --port 8000` |
| Compose default port | `8000:8000` |
| Isolated validation port | `127.0.0.1:18080:8000` |
| Chroma validation volume | `robotops-week8-clean-chroma` |
| Session validation volume | `robotops-week8-clean-sessions` |

宿主机 `8000` 端口在验证时已被 WSL 转发进程占用。为避免停止用户正在运行的服务，本次隔离容器只把宿主机端口改为 `18080`；容器内部端口、Compose 默认配置和 README 正式命令均未改变。

## 3. 构建与密钥边界

### 3.1 实际构建结果

执行：

```powershell
docker compose config --quiet
```

Compose 配置校验退出码为 `0`。

执行：

```powershell
docker compose build
```

构建成功生成 `robotops-copilot:week8`。构建日志显示：

- 已加载 `.dockerignore`；
- 构建上下文为 `4.35 MB`；
- 复制的生产内容为 `app/`、`scripts/`、`main.py` 和 `requirements.txt`；
- 最终进程使用非 root 用户 `appuser`。

### 3.2 排除规则

`.dockerignore` 在文件进入 Docker 构建上下文之前排除以下内容：

- `.env` 和 `.env.*`；
- Git、IDE 和本地 AI 工具目录；
- `chroma_data/`；
- `data/diagnostic_sessions/`；
- 原始语料、处理中间文件、评测数据和教学图片；
- `tests/`、`docs/`、`要求.txt` 和每周任务文件；
- Python 缓存、覆盖率文件、日志、压缩包和临时文件；
- 私钥、证书和常见凭据文件。

真实 `.env` 只在容器启动时通过 `--env-file` 注入。它没有被复制到镜像，也没有写入本记录、HTTP 响应或容器日志。

## 4. 空知识库启动验证

### 4.1 隔离方式

本次使用两个新建的 Docker 命名卷代替项目目录中的历史数据：

```text
robotops-week8-clean-chroma   -> /app/chroma_data
robotops-week8-clean-sessions -> /app/data/diagnostic_sessions
```

启动隔离容器时，两个目录的递归条目数均为 `0`。因此后续结果不能来自宿主机原有的 Chroma Collection 或历史会话。

### 4.2 启动结果

| 检查项 | 实际结果 |
|---|---|
| Container state | `running` |
| Health state | `healthy` |
| Container user | `appuser` |
| Restart count | `0` |
| Chroma entries before ingestion | `0` |
| Session entries before diagnosis | `0` |
| `GET /health` | HTTP `200`，`{"status":"ok"}` |
| `GET /console/` | HTTP `200`，页面包含 `RobotOps Copilot` |

该结果证明健康检查和控制台路由不依赖预先存在的 Chroma Collection。空知识库只会限制检索和诊断能力，不会阻止应用启动。

## 5. 演示语料摄取验证

宿主机把仓库内公开、脱敏的最小演示语料上传给隔离容器：

```powershell
curl.exe -i -X POST "http://127.0.0.1:18080/api/v1/knowledge/documents" -F "file=@samples/corpus/robot-fault-demo.txt"
```

实际结果：

| 字段 | 实际值 |
|---|---|
| HTTP status | `201 Created` |
| Source file | `robot-fault-demo.txt` |
| File type | `txt` |
| File size | `999` bytes |
| Section count | `1` |
| Chunk count | `1` |
| Embedding model | `embedding-3` |
| Document status | `ready` |

摄取完成后，隔离 Chroma Volume 中出现 Collection `robot_knowledge_week3_vector_baseline`，其 Chunk 数为 `1`。

调用链如下：

```text
公开演示文件
  -> Knowledge Router（接收 multipart 上传）
  -> 文档摄取服务（校验、解析和切块）
  -> Embedding Provider（生成向量）
  -> Chroma Vector Store（写入 1 个 Chunk）
  -> DocumentRecord（返回 201 与 ready 状态）
```

## 6. 真实 Agent 诊断验证

使用刚摄取的 `ERR-DEMO-1001` 演示知识执行只读诊断。请求包含脱敏机器人编号、模拟症状、模拟日志和“检索知识库并说明安全排查步骤和恢复条件”的任务目标。

实际结果：

| 检查项 | 实际结果 |
|---|---|
| HTTP status | `200` |
| Response/header request ID | 一致 |
| Diagnosis status | `completed` |
| Abstained | `false` |
| Runner termination reason | `planner_finished` |
| Finish reason | `task_completed` |
| Tool sequence | `search_knowledge` |
| Tool step count | `1` |
| Citation count | `1` |
| Persisted session count | `1` |
| Latest session status | `completed` |

调用链如下：

```text
Agent Diagnosis Router
  -> AgentDiagnosisService（协调请求生命周期）
  -> 安全分类器（确认只读诊断边界）
  -> 请求级工具策略（只开放必要工具）
  -> Planner（选择 search_knowledge）
  -> ToolExecutor（执行受控工具调用）
  -> 混合检索与证据门控（返回已确认 Chunk）
  -> AgentRunner（形成正常结束结果）
  -> 诊断报告构造器（恢复真实引用元数据）
  -> DiagnosticSessionStore（持久化脱敏会话）
  -> AgentDiagnosisResponse
```

该结果证明容器不仅能够启动，还能够在空知识库上完成“先初始化演示知识，再执行带引用诊断”的交付闭环。

## 7. 隔离资源收尾

验证结束后，一次性容器已经正常停止并移除，宿主机 `18080` 端口已经释放。两个命名卷继续保留，以便复查本次生成的 1 个知识 Chunk 和 1 份脱敏诊断会话。

本次未执行以下破坏性操作：

- 未删除项目原有知识库；
- 未删除项目诊断会话；
- 未停止占用宿主机 `8000` 端口的现有进程；
- 未删除 Docker 镜像或其他 Volume；
- 未使用 `Reset to factory defaults`；
- 未修改或输出真实 `.env` 内容。

## 8. Docker Desktop 启动故障与恢复

### 8.1 故障现象

Docker Desktop 启动时，Ingest server 无法把旧的 `sailor-ingest.sock` 重命名为 `.stale`，并报告文件无法访问。此前 Secrets Engine 也出现过相同类型的 `engine.sock` 错误。

该故障发生在 `%LOCALAPPDATA%\Docker\run` 和 `%LOCALAPPDATA%\docker-secrets-engine`，不在 RobotOps Copilot、Dockerfile、Compose、知识库或 `.env` 中。

### 8.2 原因判断

这些 socket 在 Windows 上表现为重解析点。Docker Desktop 异常退出或启动竞争后，旧 socket 可能继续存在；新后端尝试将旧 socket 改名时被 Windows 拒绝，导致 Engine 尚未启动就退出。

### 8.3 本次恢复方式

1. 确认 Docker Desktop、backend、dockerd 和 vpnkit 相关进程数量为 `0`；
2. 把 `%LOCALAPPDATA%\Docker\run` 整体改名为带 `stale-week8` 标记的备份目录；
3. 把 `%LOCALAPPDATA%\docker-secrets-engine` 整体改名为带 `stale-week8` 标记的备份目录；
4. 在原位置创建新的空目录；
5. 只启动一次 Docker Desktop；
6. 使用 Docker API 验证 Engine `29.7.2` 已经可以响应；
7. 再开始项目镜像构建和隔离验证。

旧目录通过改名保留，未被删除。该方式只重建 Docker Desktop 的临时运行时 socket 目录，不会清除镜像、Volume 或 WSL 虚拟磁盘。

## 9. 标准复现顺序

新使用者应以 README 中的七步流程为准：

1. 初始化空的 `chroma_data` 和 `data/diagnostic_sessions`；
2. 从 `.env.example` 创建本地 `.env` 并校验必需配置；
3. 执行 `docker compose build`；
4. 执行 `docker compose up --detach`；
5. 验证 `/health` 和 `/console/`；
6. 上传 `samples/corpus/robot-fault-demo.txt`；
7. 对 `ERR-DEMO-1001` 执行一次只读 Agent 诊断。

README 的正式命令使用 Compose 默认宿主机端口 `8000`。只有端口已被其他程序占用时，才需要为临时验收选择其他宿主机端口。

## 10. 限制

- 健康检查只证明 API 进程和基础路由可用，不等于外部模型质量达标；
- 演示语料只有 1 个 Chunk，只能验证初始化与端到端链路，不能替代完整知识库检索评测；
- LLM、Embedding 和 Vision 能力仍依赖各自上游服务、密钥、网络和账户额度；
- 命名卷验证可以证明不复用宿主机历史目录，但不能替代在另一台机器上的最终交付复现；
- Docker Desktop 的 socket 故障属于宿主机运行时问题，项目代码无法在 Engine 启动前修复它。

## 11. 最终判定

Week 8 镜像已在受控构建上下文中成功生成。隔离容器在空知识库和空会话目录下达到 `healthy`，健康检查与控制台均返回 HTTP `200`。公开演示语料成功生成 1 个知识 Chunk，真实 Agent 随后仅调用 `search_knowledge`，返回 `completed`、1 条引用，并持久化 1 份脱敏会话。

因此，Week 8 任务 2 所要求的无密钥模板、Docker 排除边界、空知识库启动、演示语料初始化方式和从空目录开始的复现记录已经形成完整闭环。
