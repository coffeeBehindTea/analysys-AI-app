# RobotOps Copilot Week 7 Docker 启动记录

## 1. 复现目标

本记录验证从 GitHub 下载并解压的 Week 7 分发目录可以通过 Docker Compose 构建和启动 RobotOps Copilot MVP。分发目录不包含 `.git`，生产镜像不包含 `.env`、知识库数据、诊断会话、测试、报告或内部任务文档。

本次验证使用宿主机运行时挂载恢复知识库和诊断会话目录，因此“干净目录”指源码来自 GitHub 分发包且不依赖本地 Git 历史，不表示没有运行所需的 `.env` 与授权数据卷。

## 2. Docker 配置

| 项目 | 配置 |
|---|---|
| Compose project | `robotops-copilot` |
| Service | `robotops-copilot` |
| Image | `robotops-copilot:week7` |
| Base image | `python:3.11-slim-bookworm` |
| Application user | `appuser`，UID/GID `10001` |
| Application command | `python -m uvicorn main:app --host 0.0.0.0 --port 8000` |
| Host/container port | `8000:8000` |
| Chroma mount | `./chroma_data:/app/chroma_data` |
| Session mount | `./data/diagnostic_sessions:/app/data/diagnostic_sessions` |
| Restart policy | `unless-stopped` |
| Init process | enabled |
| Health endpoint | `http://127.0.0.1:8000/health` |
| OCR | Tesseract English and Simplified Chinese language data |

## 3. 安全构建边界

`.dockerignore` 在 Dockerfile 执行前过滤构建上下文。关键排除项包括：

- `.env` 和 `.env.*`；
- Git、IDE 和本地 AI 工具目录；
- `chroma_data/`；
- `data/diagnostic_sessions/`；
- 原始语料、处理中间文件、评测数据和教学图片；
- `tests/`、`docs/`、`要求.txt` 和每周任务文件；
- Python 缓存、覆盖率文件、日志、压缩包和临时文件；
- 私钥、证书和常见凭据文件。

生产代码 `app/`、`scripts/`、`main.py` 和 `requirements.txt` 未被排除。SSE Router、事件发布器、Streaming Service 和 Web 控制台静态资源都位于 `app/`，会正常进入镜像。

## 4. 标准复现步骤

以下命令均在项目根目录运行，并保持单行形式。

### 4.1 校验 Compose

```powershell
docker compose config --quiet
```

预期结果：退出码为 `0`，不输出配置错误。

### 4.2 构建镜像

```powershell
docker compose build
```

预期结果：生成 `robotops-copilot:week7`，构建过程中加载 `.dockerignore`。

### 4.3 启动或重建容器

```powershell
docker compose up --detach --force-recreate
```

预期结果：服务被创建并进入运行状态。

### 4.4 查看健康状态

```powershell
docker compose ps
```

预期结果：容器显示 `Up` 和 `healthy`，端口显示 `0.0.0.0:8000->8000/tcp`。

### 4.5 停止服务

```powershell
docker compose stop
```

`stop` 保留容器和挂载数据。需要再次启动时运行 `docker compose up --detach`。

## 5. 本次实际构建结果

| 检查项 | 实际结果 |
|---|---|
| Docker Desktop | `running` |
| Docker Engine | `29.7.2` |
| Docker Desktop Engine OS | `Docker Desktop` |
| 构建上下文 | `4.35 MB` |
| 镜像 | `robotops-copilot:week7` |
| 镜像摘要 | `sha256:d4d3cbdf8131b897027d5bb6b5dca673c1f91100420a3eb8dfc0cfa54ff6268c` |
| 容器 | `robotops-copilot-robotops-copilot-1` |
| 容器用户 | `appuser` |
| 容器状态 | `running` |
| 健康状态 | `healthy` |
| 重启次数 | `0` |
| 端口 | `0.0.0.0:8000->8000/tcp` 和 IPv6 对应映射 |

构建上下文只有 `4.35 MB`，且构建日志明确出现 `load .dockerignore`，符合本地数据库、报告、测试和密钥未进入构建上下文的预期。

## 6. HTTP 冒烟结果

| 请求 | 实际结果 |
|---|---|
| `GET /health` | HTTP `200`，正文 `{"status":"ok"}` |
| `GET /console/` | HTTP `200`，页面包含 `RobotOps Copilot` |
| `GET /openapi.json` | 已包含 `/api/v1/agent/diagnose/stream` |
| `POST /api/v1/agent/diagnose/stream` | HTTP `200`，`text/event-stream; charset=utf-8` |

高风险 SSE 冒烟事件顺序为：

```text
request_received -> safety_classified -> diagnosis_finished
```

事件序号为 `1, 2, 3`，最终 `diagnosis.status` 和 `execution.finish_reason` 都是 `human_review_required`。请求没有进入真实控制路径。

## 7. 本次 Docker Desktop 启动故障与恢复

### 7.1 故障现象

Docker Desktop 启动时报告 Ingest server 或 Secrets Engine 无法处理 AF_UNIX socket。核心错误是：

```text
rename .../sailor-ingest.sock .../sailor-ingest.sock.stale:
The file cannot be accessed by the system.
```

问题发生在 Docker Desktop 的宿主机运行时目录，不在 RobotOps Copilot 项目、Dockerfile 或 Compose 配置中。

### 7.2 原因判断

`sailor-ingest.sock` 和 `engine.sock` 在 Windows 上表现为重解析点。Docker Desktop 异常退出或并发启动后，旧 socket 仍存在或被残留后端占用；新后端尝试把旧 socket 重命名为 `.stale` 时被 Windows 拒绝，导致后端在 Engine 启动前崩溃。

### 7.3 恢复操作

1. 等待 Docker Desktop 崩溃窗口自行退出；
2. 确认 Docker Desktop、Docker backend 和 Docker CLI 残留进程均已结束；
3. 将损坏的 `C:\Users\dada2\AppData\Local\Docker\run` 整体移动到明确命名的可恢复备份目录；
4. 创建新的空 `run` 目录；
5. 对 Secrets Engine 运行时目录采用相同的可恢复备份方式；
6. 只启动一次 Docker Desktop；
7. 通过 Docker API 验证 Desktop 和 Engine 均为 `running`；
8. 再执行 Compose 构建和启动。

已保留的运行时备份包括：

- `C:\Users\dada2\AppData\Local\Docker\run.stale-week7-20260915`；
- `C:\Users\dada2\AppData\Local\Docker\run.stale-week7-20260915-race2`；
- `C:\Users\dada2\AppData\Local\docker-secrets-engine.stale-week7-20260915`。

这些目录只保存故障时的运行时 socket 状态，不是 Docker 镜像、卷或 WSL 虚拟磁盘。

### 7.4 未执行的破坏性操作

- 未点击 `Reset to factory defaults`；
- 未删除 Docker Desktop 数据目录；
- 未删除 WSL 发行版或虚拟磁盘；
- 未删除 Docker 镜像或卷；
- 未清空 Chroma 知识库；
- 未修改项目 `.env`。

## 8. 前置条件与限制

- Windows 上必须先启动 Docker Desktop，并确认 Engine 为 `running`；
- 项目根目录必须存在有效 `.env`，但该文件不得提交 Git 或进入镜像；
- `chroma_data/` 必须包含项目期望的 Collection，Compose 只负责挂载，不自动摄取语料；
- 宿主机 `8000` 端口必须可用；
- 外部 LLM、Embedding 和 Vision 功能还依赖相应服务可用、密钥有效和账户额度充足；
- 健康检查只证明 API 进程可以响应，不替代外部模型端到端质量评测；
- 遇到相同 socket 错误时，应先完全退出 Docker Desktop并重启 Windows，不能直接执行 factory reset。

## 9. 最终判定

Week 7 镜像已在 GitHub 分发目录中成功构建，容器以非 root 用户运行并达到健康状态，HTTP、控制台和 SSE 冒烟均通过。Docker 启动故障已被定位为宿主机运行时 socket 问题，并采用可恢复方式处理。
