# Day 1 工程基线

## 1. 基线目的

本文档记录 Robot Incident Triage API 在 Day 1 阶段经过验证的工程状态。

工程基线用于确认：

- Python 环境可以正常使用。
- 项目依赖已经正确安装。
- FastAPI 应用可以启动。
- 健康检查接口可以正常响应。
- FastAPI 自动生成的 API 文档可以访问。

后续开发中出现问题时，可以使用这份基线判断问题属于环境、服务启动还是新增业务功能。

## 2. 开发环境

| 项目 | 当前配置 |
|---|---|
| Python | 3.11.15 |
| 环境管理工具 | Conda |
| Conda 环境名称 | `analysys` |
| Web 框架 | FastAPI |
| ASGI 服务器 | Uvicorn |
| 操作系统终端 | PowerShell |

## 3. 项目直接依赖

当前 `requirements.txt` 包含：

```txt
fastapi
uvicorn[standard]
openai
pydantic-settings
pytest
pytest-asyncio
httpx
sse-starlette
```

各依赖的主要职责如下：

| 依赖 | 职责 |
|---|---|
| `fastapi` | 定义 HTTP API、路由和请求处理逻辑 |
| `uvicorn` | 监听网络端口并运行 FastAPI 应用 |
| `openai` | 调用 OpenAI 兼容的 LLM 接口 |
| `pydantic-settings` | 从环境变量和 `.env` 文件读取配置 |
| `pytest` | 编写和运行自动化测试 |
| `pytest-asyncio` | 测试异步 Python 函数 |
| `httpx` | 发送 HTTP 请求并测试异步 API |
| `sse-starlette` | 实现 SSE 流式响应 |

## 4. 环境验证

激活 Conda 环境：

```powershell
conda activate analysys
```

检查 Python 版本：

```powershell
python --version
```

验证结果：

```text
Python 3.11.15
```

检查已安装依赖：

```powershell
python -m pip check
```

验证结果：

```text
No broken requirements found.
```

## 5. 服务启动

在项目根目录执行：

```powershell
cd D:\analysys\week-01code
conda activate analysys
python -m uvicorn main:app --reload
```

启动后，Uvicorn 默认监听：

```text
http://127.0.0.1:8000
```

`--reload` 用于本地开发。代码文件发生变化后，Uvicorn 会自动重新加载应用。

## 6. 健康检查验证

请求：

```http
GET /health
```

PowerShell 验证命令：

```powershell
curl.exe http://127.0.0.1:8000/health
```

预期 HTTP 状态：

```text
200 OK
```

预期响应：

```json
{
  "status": "ok"
}
```

实际服务器日志：

```text
INFO: 127.0.0.1:53255 - "GET /health HTTP/1.1" 200 OK
```

验证结论：健康检查接口可以正常访问。

## 7. API 文档验证

Swagger UI 地址：

```text
http://127.0.0.1:8000/docs
```

OpenAPI JSON 地址：

```text
http://127.0.0.1:8000/openapi.json
```

验证结论：两个地址均可以正常访问。

## 8. 当前应用代码

当前 `main.py`：

```python
from fastapi import FastAPI


app = FastAPI(
    title="Robot Incident Triage API",
    version="0.1.0",
)


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}
```

## 9. 当前项目结构

```text
week-01code/
├── docs/
│   └── baseline.md
├── .env.example
├── .gitignore
├── main.py
├── README.md
├── requirements.txt
└── week1.txt
```

运行服务后，Python 可能生成 `__pycache__/`。该目录是 Python 字节码缓存，已经通过 `.gitignore` 排除，不需要提交。

## 10. 基线结论

Day 1 工程基线验证通过：

- Python 3.11 环境正常。
- 项目依赖安装完整。
- FastAPI 应用可以启动。
- `GET /health` 返回 `200 OK`。
- Swagger UI 可以正常访问。
- OpenAPI JSON 可以正常访问。
- `.env` 已通过 `.gitignore` 排除。
- `.env.example` 可以作为环境变量配置模板提交。

下一阶段将建立 `router-schema-service-config` 分层结构，并将 `/health` 路由从 `main.py` 移入独立路由模块。