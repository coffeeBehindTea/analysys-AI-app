"""Week 7轻量Web控制台与SSE接入的离线集成测试。

这些测试只通过ASGI接口读取本地静态文件，
不会启动真实端口，也不会调用LLM、Vision、Embedding、
Chroma、Agent工具或诊断会话存储。
"""

from contextlib import (
    asynccontextmanager,
)
from typing import (
    AsyncIterator,
)

import httpx
import pytest

from main import create_app


# 控制台公开路径使用尾部斜杠。
#
# /console是StaticFiles挂载点，Starlette会把它重定向到
# /console/，使相对静态资源和index.html目录语义保持一致。
CONSOLE_PATH = "/console/"

STYLES_PATH = "/console/styles.css"

SCRIPT_PATH = "/console/app.js"


@asynccontextmanager
async def create_test_client(
) -> AsyncIterator[httpx.AsyncClient]:
    """创建不监听真实网络端口的异步ASGI客户端。

    ASGITransport把httpx请求直接交给FastAPI应用，
    base_url只用于补全相对URL，不会访问example.test。
    """

    application = create_app()

    transport = httpx.ASGITransport(
        app=application,
    )

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://example.test",
        follow_redirects=False,
    ) as client:
        yield client


@pytest.mark.asyncio
async def test_console_root_serves_html(
) -> None:
    """控制台根路径应返回包含主要工作区的HTML。

    预期流程：

    1. FastAPI匹配/console静态挂载；
    2. StaticFiles因html=True读取index.html；
    3. 响应使用text/html媒体类型；
    4. 页面包含诊断表单、结果区和会话列表。
    """

    async with create_test_client() as client:
        response = await client.get(
            CONSOLE_PATH,
        )

    assert response.status_code == 200
    assert response.headers[
        "content-type"
    ].startswith("text/html")

    html = response.text

    assert "RobotOps Copilot Beta" in html
    assert 'id="diagnosis-form"' in html
    assert 'id="diagnosis-result"' in html
    assert 'id="session-list"' in html

    # Week 7新增区域必须允许用户看到实时事件，
    # 并在不执行任何机器人控制的前提下停止等待连接。
    assert 'id="stream-progress"' in html
    assert 'id="stream-event-list"' in html
    assert 'id="abort-stream-button"' in html

    # 图标使用内嵌SVG，浏览器不应再请求一个不存在的
    # /favicon.ico文件，也不会访问外部图标服务。
    assert 'rel="icon"' in html
    assert 'href="data:image/svg+xml,' in html


@pytest.mark.asyncio
async def test_console_mount_redirects_to_slash(
) -> None:
    """不带斜杠的挂载路径应重定向到/console/。

    这个测试确认用户输入/console时不会得到404，
    同时保留静态目录的标准URL语义。
    """

    async with create_test_client() as client:
        response = await client.get(
            "/console",
        )

    assert response.status_code == 307
    assert response.headers[
        "location"
    ] == "http://example.test/console/"


@pytest.mark.asyncio
async def test_console_stylesheet_is_served(
) -> None:
    """CSS资源应由同一个FastAPI应用返回。

    除了检查200状态，还检查CSS变量和移动端规则，
    防止误把空文件或其他文本文件当成样式表。
    """

    async with create_test_client() as client:
        response = await client.get(
            STYLES_PATH,
        )

    assert response.status_code == 200
    assert response.headers[
        "content-type"
    ].startswith("text/css")
    assert "--color-primary" in response.text
    assert "@media (max-width: 760px)" in (
        response.text
    )


@pytest.mark.asyncio
async def test_console_script_is_served(
) -> None:
    """JavaScript资源应包含真实API路径和安全渲染方式。

    测试不在浏览器中执行JavaScript，只校验FastAPI返回的
    是预期入口脚本，而不是404页面或错误文件。
    """

    async with create_test_client() as client:
        response = await client.get(
            SCRIPT_PATH,
        )

    assert response.status_code == 200

    content_type = response.headers[
        "content-type"
    ]

    assert (
        "javascript" in content_type
        or "text/plain" in content_type
    )

    script = response.text

    assert (
        "/api/v1/agent/diagnose/stream"
        in script
    )
    assert "/api/v1/diagnostic-sessions" in script
    assert ".textContent" in script
    assert ".innerHTML" not in script


@pytest.mark.asyncio
async def test_console_script_consumes_and_validates_sse(
) -> None:
    """控制台必须流式读取POST响应并校验终端事件。

    预期流程：

    1. fetch向流式POST接口发送JSON；
    2. ReadableStream.getReader逐块读取字节；
    3. TextDecoder按UTF-8恢复SSE文本；
    4. 解析器核对id、event、sequence和event_type；
    5. diagnosis_finished返回最终响应；
    6. stream_error或提前断开显示明确失败状态。
    """

    async with create_test_client() as client:
        response = await client.get(
            SCRIPT_PATH,
        )

    assert response.status_code == 200
    script = response.text

    assert "response.body.getReader()" in script
    assert 'new TextDecoder("utf-8")' in script
    assert "parseAgentSseFrame" in script
    assert "eventData.sequence" in script
    assert "diagnosis_finished" in script
    assert "stream_error" in script
    assert "AbortController" in script
    assert "最终结果返回前中断" in script


@pytest.mark.asyncio
async def test_console_exports_final_response_as_json_and_markdown(
) -> None:
    """两个导出按钮必须复用最终公开响应并生成本地文件。

    预期流程：

    1. diagnosis_finished或会话详情进入renderAgentResponse；
    2. setExportResponse保存同一份公开响应并启用按钮；
    3. JSON导出使用JSON.stringify保留机器可读结构；
    4. Markdown导出整理状态、证据、观察和工具轨迹；
    5. Blob和临时对象URL触发浏览器本地下载；
    6. 导出内容不重新调用LLM，也不加入生成时间。
    """

    async with create_test_client() as client:
        html_response = await client.get(
            CONSOLE_PATH,
        )
        script_response = await client.get(
            SCRIPT_PATH,
        )

    assert html_response.status_code == 200
    assert script_response.status_code == 200

    html = html_response.text
    script = script_response.text

    assert 'id="export-json-button"' in html
    assert 'id="export-markdown-button"' in html
    assert "latestAgentResponse" in script
    assert "buildAgentDiagnosisMarkdown" in script
    assert "JSON.stringify" in script
    assert '"application/json"' in script
    assert '"text/markdown"' in script
    assert "new Blob(" in script
    assert "URL.createObjectURL" in script
    assert "URL.revokeObjectURL" in script

    # 生成的Markdown应包含完整的公开来源分层，
    # 而不是只导出最终状态或一段摘要。
    for heading in (
        "## 已报告症状",
        "## 可能原因",
        "## 后续检查",
        "## 知识库引用",
        "## 视觉观察",
        "## 模拟遥测",
        "## 测试草案",
        "## 工具执行轨迹",
    ):
        assert heading in script


@pytest.mark.asyncio
async def test_console_unknown_asset_returns_404(
) -> None:
    """不存在的静态资源必须返回404。

    StaticFiles不能把index.html错误地当作所有未知文件的响应，
    否则浏览器会把HTML当作CSS或JavaScript解析。
    """

    async with create_test_client() as client:
        response = await client.get(
            "/console/missing-resource.js",
        )

    assert response.status_code == 404


def test_openapi_keeps_api_routes_separate_from_console(
) -> None:
    """静态控制台不能替代或覆盖已有Agent API。

    StaticFiles挂载不会生成普通OpenAPI操作；
    Agent和会话API仍然必须保留在OpenAPI paths中。
    """

    application = create_app()
    openapi_paths = application.openapi()[
        "paths"
    ]

    assert "/api/v1/agent/diagnose" in (
        openapi_paths
    )
    assert "/api/v1/agent/diagnose/stream" in (
        openapi_paths
    )
    assert "/api/v1/diagnostic-sessions" in (
        openapi_paths
    )
    assert CONSOLE_PATH not in openapi_paths
