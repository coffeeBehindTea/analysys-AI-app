# Robot Incident Triage API

一个使用 FastAPI 和大语言模型实现的机器人故障摘要服务。

本项目接收脱敏后的机器人故障描述和日志摘要，通过 LLM 生成谨慎、可追踪的故障分析与排查建议，并提供普通响应和 SSE 流式响应接口。

## 环境要求

- Python 3.11+
- Conda
- Windows PowerShell 或其他命令行终端

## 创建并激活环境

如果还没有创建 Conda 环境，可以执行：

```powershell
conda create -n analysys python=3.11
```

激活环境：

```powershell
conda activate analysys
```

确认 Python 版本：

```powershell
python --version
```

## 安装依赖

在项目根目录执行：

```powershell
python -m pip install -r requirements.txt
```

检查依赖是否完整：

```powershell
python -m pip check
```

## 配置环境变量

复制环境变量模板：

```powershell
Copy-Item .env.example .env
```

然后打开 `.env`，填写实际配置：

```dotenv
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://your-llm-provider.example/v1
LLM_MODEL=your-model-name
```

`.env` 中可能包含真实 API Key，因此不能提交到 Git。项目只提交不包含真实密钥的 `.env.example`。

当前 `/health` 接口不需要 API Key，可以在尚未配置 LLM 的情况下正常使用。

## 启动服务

在项目根目录执行：

```powershell
conda activate analysys
python -m uvicorn main:app --reload
```

命令中的：

- `main` 表示 `main.py` 模块。
- `app` 表示 `main.py` 中创建的 FastAPI 应用对象。
- `--reload` 表示代码修改后自动重启服务，只用于本地开发。

服务默认运行在：

```text
http://127.0.0.1:8000
```

## 健康检查

使用浏览器访问：

[http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

也可以在另一个终端中执行：

```powershell
curl.exe http://127.0.0.1:8000/health
```

预期响应状态：

```text
200 OK
```

预期响应内容：

```json
{
  "status": "ok"
}
```

## API 文档

服务启动后，可以访问 FastAPI 自动生成的接口文档：

- Swagger UI：[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- OpenAPI JSON：[http://127.0.0.1:8000/openapi.json](http://127.0.0.1:8000/openapi.json)

## 当前接口

### `GET /health`

检查 API 服务是否正在运行。

响应示例：

```json
{
  "status": "ok"
}
```

## 运行测试

项目添加测试后，可以执行：

```powershell
python -m pytest tests/ -v
```

测试不应依赖真实 API Key 或外部网络。

## 当前项目结构

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

后续将逐步增加以下分层目录：

```text
app/
├── routers/
├── schemas/
├── services/
└── utils/
```
