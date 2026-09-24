"""Week 8干净环境交付材料与空知识库启动测试。

本文件验证的是交付契约，而不是LLM、Embedding或Vision质量：

1. .env.example覆盖Settings公开的全部配置字段；
2. .dockerignore阻止密钥、本地数据库和会话进入镜像；
3. compose.yaml使用Week 8镜像并挂载运行时目录；
4. README提供按固定顺序执行的干净环境闭环；
5. 不存在Chroma目录时，健康检查和Web控制台仍可启动。

测试不会访问真实网络，不会创建Chroma Collection，
也不会读取本机.env中的密钥值。
"""

from pathlib import Path

import httpx
import pytest

from app.config import Settings, get_settings
from main import create_app


# tests目录的上一级是项目根目录。
# 所有交付文件路径都从这里构造，避免依赖pytest启动位置。
PROJECT_ROOT = Path(__file__).resolve().parents[1]

ENV_TEMPLATE_PATH = PROJECT_ROOT / ".env.example"
DOCKERIGNORE_PATH = PROJECT_ROOT / ".dockerignore"
COMPOSE_PATH = PROJECT_ROOT / "compose.yaml"
README_PATH = PROJECT_ROOT / "README.md"


# 这些规则共同保证密钥和本地运行数据不会进入Docker构建上下文。
REQUIRED_DOCKERIGNORE_RULES = frozenset({
    ".env",
    ".env.*",
    "chroma_data/",
    "data/diagnostic_sessions/",
    "data/source/*",
    "tests/",
    "docs/",
    "要求.txt",
})


# README中的七个标记就是新使用者应执行的实际顺序。
# 测试不仅检查文字存在，还检查位置递增，防止后续编辑打乱闭环。
CLEAN_START_SEQUENCE = (
    "#### 第一步：初始化空运行目录",
    "#### 第二步：创建并校验配置",
    "#### 第三步：构建镜像",
    "#### 第四步：启动空知识库服务",
    "#### 第五步：验证健康检查和控制台",
    "#### 第六步：摄取公开演示语料",
    "#### 第七步：执行一次诊断",
)


def _read_env_template() -> dict[str, str]:
    """读取无密钥模板中的键值，不读取本机.env。"""

    entries: dict[str, str] = {}

    for raw_line in ENV_TEMPLATE_PATH.read_text(
        encoding="utf-8"
    ).splitlines():
        line = raw_line.strip()

        if (
            not line
            or line.startswith("#")
            or "=" not in line
        ):
            continue

        key, value = line.split("=", 1)
        entries[key.strip().lower()] = (
            value.strip()
        )

    return entries


def test_env_example_covers_every_settings_field_without_real_keys(
) -> None:
    """配置模板应完整映射Settings并只保存占位密钥。

    被测试对象是.env.example。测试读取模板键名，并与
    Settings.model_fields公开的数据模型字段逐项比较。

    预期模板不缺字段、没有未知字段；三种Provider的API Key
    必须使用your-前缀占位值，不能把本机.env密钥复制进模板。
    """

    entries = _read_env_template()
    expected_fields = set(
        Settings.model_fields
    )

    assert set(entries) == expected_fields

    for key in (
        "llm_api_key",
        "embedding_api_key",
        "vision_api_key",
    ):
        assert entries[key].startswith(
            "your-"
        )


def test_dockerignore_excludes_secrets_and_local_runtime_data(
) -> None:
    """Docker构建上下文必须排除密钥和本地持久化数据。

    被测试对象是.dockerignore。测试忽略空行和注释后建立规则
    集合，再检查安全边界要求的规则全部存在。预期任何缺少项
    都会直接显示在断言差集中，而不是等镜像构建后才发现泄露。
    """

    rules = {
        line.strip()
        for line in DOCKERIGNORE_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
        and not line.lstrip().startswith("#")
    }

    assert (
        REQUIRED_DOCKERIGNORE_RULES
        - rules
    ) == set()


def test_compose_uses_week8_image_and_runtime_bind_mounts(
) -> None:
    """Compose应使用Week 8镜像和两个运行时绑定挂载。

    被测试对象是compose.yaml。这里不启动Docker，而是检查
    交付文件中的稳定声明。预期镜像标签已经从week7更新为
    week8，并继续通过.env注入配置、挂载Chroma与会话目录。
    """

    compose_text = COMPOSE_PATH.read_text(
        encoding="utf-8"
    )

    assert "image: robotops-copilot:week8" in (
        compose_text
    )
    assert "- .env" in compose_text
    assert "source: ./chroma_data" in compose_text
    assert "target: /app/chroma_data" in compose_text
    assert (
        "source: ./data/diagnostic_sessions"
        in compose_text
    )
    assert (
        "target: /app/data/diagnostic_sessions"
        in compose_text
    )


def test_readme_preserves_clean_start_sequence_and_demo_corpus(
) -> None:
    """README必须给出可按顺序执行的干净环境闭环。

    被测试对象是README的Docker干净启动小节。测试逐项查找
    七个步骤并验证行文顺序，同时检查公开演示语料和上传接口。

    预期新使用者不会在目录尚未初始化时挂载，也不会在服务
    启动前执行HTTP上传；摄取使用仓库中的脱敏模拟语料。
    """

    readme = README_PATH.read_text(
        encoding="utf-8"
    )

    positions = [
        readme.index(marker)
        for marker in CLEAN_START_SEQUENCE
    ]

    assert positions == sorted(positions)
    assert (
        "samples/corpus/robot-fault-demo.txt"
        in readme
    )
    assert (
        "/api/v1/knowledge/documents"
        in readme
    )
    assert "ERR-DEMO-1001" in readme


@pytest.mark.asyncio
async def test_health_and_console_start_with_empty_runtime_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """空知识库不能阻止健康检查或控制台启动。

    被测试模块是main.create_app()装配出的FastAPI应用。
    测试把Chroma和会话路径指向尚不存在的临时目录，然后使用
    HTTPX ASGITransport在内存中请求/health与/console/。

    预期两个接口都返回200，而且请求过程不会为了展示健康状态
    或静态页面而创建Chroma目录、Collection或会话目录。这证明
    空知识库是合法启动状态，真实外部Provider也不会被调用。
    """

    empty_chroma = tmp_path / "chroma_data"
    empty_sessions = (
        tmp_path
        / "data"
        / "diagnostic_sessions"
    )

    monkeypatch.setenv(
        "CHROMA_PERSIST_DIRECTORY",
        str(empty_chroma),
    )
    monkeypatch.setenv(
        "DIAGNOSTIC_SESSION_DIRECTORY",
        str(empty_sessions),
    )
    get_settings.cache_clear()

    assert not empty_chroma.exists()
    assert not empty_sessions.exists()

    application = create_app()
    transport = httpx.ASGITransport(
        app=application
    )

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://example.test",
    ) as client:
        health_response = await client.get(
            "/health"
        )
        console_response = await client.get(
            "/console/"
        )

    assert health_response.status_code == 200
    assert health_response.json() == {
        "status": "ok",
    }
    assert console_response.status_code == 200
    assert "RobotOps Copilot" in (
        console_response.text
    )

    assert not empty_chroma.exists()
    assert not empty_sessions.exists()

    get_settings.cache_clear()
