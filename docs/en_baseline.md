# Day 1 Engineering Baseline

## 1. Baseline purpose

This document records the validated state of the project at Day 1.

The scope baseline is used to confirm:

- The Python environment can be used properly.
- The project relies on having been correctly installed.
- FastAPI applications can be activated.
- The health check interface is able to respond normally.
- FastAPI Automatically generated API documents are accessible.

This baseline can be used to determine whether the issue is one of environment, service start-up or new business functions in subsequent developments.

## 2. Development environment

| Item | Current Configuration |
|---|---|
| Python | 3.11.15 |
| Environmental management tools | Conda |
| Conda Environment Name | `analysys` |
| Web Frame | FastAPI |
| ASGI Server | Uvicorn |
| Operating System Terminal | PowerShell |

## 3. Directly dependent on the project

Current `requirements.txt` contains:

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

The main responsibilities relied upon are as follows:

| Dependency | Duties |
|---|---|
| `fastapi` | Definition of HTTP API, route and request processing logic |
| `uvicorn` | Listen to network ports and run FastAPI applications |
| `openai` | Call for OpenAI compatible LLM interface |
| `pydantic-settings` | Read configurations from environment variables and `.env` files |
| `pytest` | Development and operation of automated testing |
| `pytest-asyncio` | Testing the Python function |
| `httpx` | Send HTTP request and test for apogee API |
| `sse-starlette` | Achieve SSE Fluent Response |

## 4. Environmental validation

Activate Conda Environment:

```powershell
conda activate analysys
```

Check Python version:

```powershell
python --version
```

Validation results:

```text
Python 3.11.15
```

Check installed dependency:

```powershell
python -m pip check
```

Validation results:

```text
No broken requirements found.
```

## 5. Service start

In the project root directory:

```powershell
# Enter the root of the project after cloning; do not write the absolute path of the developer.
Set-Location <repository-directory>
conda activate analysys
python -m uvicorn main:app --reload
```

After startup, Uvicorn default listening:

```text
http://127.0.0.1:8000
```

`--reload` for local development. When the code file changes, Uvicorn automatically reloads the application.

## 6. Health check verification

Requests:

```http
GET /health
```

PowerShell Validation order:

```powershell
curl.exe http://127.0.0.1:8000/health
```

Expected HTTP status:

```text
200 OK
```

Expected response:

```json
{
  "status": "ok"
}
```

Actual server log:

```text
INFO: 127.0.0.1:53255 - "GET /health HTTP/1.1" 200 OK
```

Validation of conclusions: Health check interfaces are accessible regularly.

## 7. API Document Validation

Swagger UI Address:

```text
http://127.0.0.1:8000/docs
```

OpenAPI JSON Address:

```text
http://127.0.0.1:8000/openapi.json
```

Validation conclusion: Both addresses are regularly accessible.

## 8. Current application code

Current `main.py`

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

## 9. Current project structure

```text
week-01code/
├── docs/
│   └── en_baseline.md
├── .env.example
├── .gitignore
├── main.py
├── en_README.md
├── requirements.txt
└── week1.txt
```

After running the service, Python may generate `__pycache__/`. The directory is a Python byte cache that has been excluded through `.gitignore` and does not need to be submitted.

## 10. Baseline conclusion

Day 1 Project baseline validation is through:

- Python 3.11 is safe.
- The project relies on installation integrity.
- FastAPI applications can be activated.
- `GET /health` returns `200 OK`.
- Swagger UI can be accessed normally.
- OpenAPI JSON can normally access.
- `.env` has been excluded by `.gitignore`.
- `.env.example` can be submitted as a template for the configuration of environmental variables.

The next phase will build `router-schema-service-config` stratification structures and move `/health` route from `main.py` to the stand-alone route module.
