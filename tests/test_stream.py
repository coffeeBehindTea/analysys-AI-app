"""验证 SSE 流式接口的成功事件协议。"""

# json 是 Python 标准库，用于解析 SSE data 中的 JSON 字符串。
import json

# AsyncIterator[str] 描述 Fake Service 逐个产生字符串。
from collections.abc import AsyncIterator

# pytest 提供测试发现、异步标记和断言结果报告。
import pytest

# ASGITransport 让 HTTPX 直接调用 FastAPI 应用；
# AsyncClient 提供异步 HTTP 请求方法。
from httpx import ASGITransport, AsyncClient

# FastAPI 的 dependency_overrides 必须使用原始依赖函数作为字典键。
from app.dependencies import get_triage_service

# Fake Service 仍然接收真实请求 Schema，
# 保证它的方法调用方式与真实 Service 一致。
from app.schemas.triage import TriageRequest

# LLMTimeoutError 表示已经被 Service 转换过的上游超时。
# Router 应该把它转换成安全的 SSE error 事件。
from app.errors import LLMTimeoutError

# 导入已经装配好 Router 和 Middleware 的 FastAPI 应用。
from main import app


# 测试使用的合法请求体。
VALID_PAYLOAD = {
    "robot_id": "test-robot-001",
    "symptom": "机器人无法移动",
    "log_excerpt": "WARN wheel feedback unavailable",
}


class FakeSuccessStreamService:
    """不访问真实 LLM，按固定顺序产生两个文本增量。"""

    async def stream_analysis(
        self,
        request: TriageRequest,
    ) -> AsyncIterator[str]:
        """模拟 LLM 将一个合法 JSON 拆成两个分块返回。"""

        # 读取 request.robot_id，证明 Router 传入了经过校验的请求模型。
        assert request.robot_id == "test-robot-001"

        # 两个字符串拼接后是符合 LLMAnalysis 契约的完整 JSON。
        yield '{"summary":"测试摘要",'
        yield '"recommended_actions":["检查连接"]}'

class FakeTimeoutStreamService:
    """模拟 LLM 在流式输出过程中发生超时。"""

    def __init__(
        self,
        emit_delta_before_error: bool,
    ) -> None:
        """保存是否要在抛出异常前先产生一个文本增量。"""

        self._emit_delta_before_error = (
            emit_delta_before_error
        )

    async def stream_analysis(
        self,
        request: TriageRequest,
    ) -> AsyncIterator[str]:
        """根据测试参数立即超时，或先输出一个分块再超时。"""

        # 确认 Router 传入了经过 Pydantic 校验的请求模型。
        assert request.robot_id == "test-robot-001"

        if self._emit_delta_before_error:
            # 模拟上游已经返回部分内容，随后网络超时。
            yield '{"summary":"尚未完成'

        # 这是服务端内部诊断信息。
        # Router 应记录它，但不能原样发送给客户端。
        raise LLMTimeoutError(
            "测试使用的内部超时详情"
        )

def parse_sse_events(
    response_body: str,
) -> list[tuple[str, dict[str, object]]]:
    """把 SSE 响应文本解析成“事件名和事件数据”列表。"""

    # SSE 标准通常使用 \r\n；先统一替换为 \n，
    # 方便后续根据空行拆分事件。
    normalized_body = response_body.replace("\r\n", "\n")

    # 两个连续换行表示一个 SSE 事件结束。
    raw_blocks = normalized_body.split("\n\n")

    events: list[tuple[str, dict[str, object]]] = []

    for raw_block in raw_blocks:
        # strip() 去掉事件块首尾空白。
        block = raw_block.strip()

        # 响应末尾可能产生空块，直接跳过。
        if not block:
            continue

        event_name: str | None = None
        data_lines: list[str] = []

        # splitlines() 将一个事件块按行拆开。
        for line in block.splitlines():
            if line.startswith("event:"):
                # removeprefix() 删除字段名称；
                # strip() 再删除字段值前后的空格。
                event_name = line.removeprefix("event:").strip()

            elif line.startswith("data:"):
                data_line = line.removeprefix("data:").strip()
                data_lines.append(data_line)

        # 忽略没有 event 或 data 字段的内容，
        # 例如 SSE 库可能发送的注释心跳。
        if event_name is None or not data_lines:
            continue

        # 一个 SSE 事件可以包含多行 data；
        # 使用换行符重新组合后，再解析其中的 JSON。
        data_text = "\n".join(data_lines)
        event_data = json.loads(data_text)

        events.append((event_name, event_data))

    return events


@pytest.mark.asyncio
async def test_stream_success_event_order() -> None:
    """成功流应该按 meta、delta、delta、done 顺序输出。"""

    fake_service = FakeSuccessStreamService()

    def override_get_triage_service() -> FakeSuccessStreamService:
        """让 FastAPI 注入 Fake Service，不创建真实 LLM 客户端。"""

        return fake_service

    # 将原来的 get_triage_service 依赖替换成测试函数。
    app.dependency_overrides[
        get_triage_service
    ] = override_get_triage_service

    try:
        # ASGITransport 将 HTTP 请求直接送入 FastAPI 应用，
        # 不启动 Uvicorn，也不访问网络。
        transport = ASGITransport(app=app)

        # AsyncClient 实现异步上下文管理器；
        # async with 结束时会释放客户端资源。
        async with AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            # json= 自动将字典序列化为 JSON 请求体，
            # 同时设置 Content-Type: application/json。
            response = await client.post(
                "/api/v1/triage/stream",
                json=VALID_PAYLOAD,
            )

    finally:
        # dependency_overrides 属于全局 app；
        # 即使测试请求抛出异常，也必须恢复应用状态。
        app.dependency_overrides.clear()

    # SSE 连接成功建立时，HTTP 状态应该是 200。
    assert response.status_code == 200

    # Content-Type 还可能包含 charset=utf-8，
    # 因此使用 startswith()，不要求整个字符串完全相等。
    assert response.headers["content-type"].startswith(
        "text/event-stream"
    )

    # HTTPX 在这个测试中收集完整响应体；
    # parse_sse_events() 再把文本解析成事件列表。
    events = parse_sse_events(response.text)

    # 列表推导式取出每个二元组中的事件名称。
    event_names = [
        event_name
        for event_name, _event_data in events
    ]

    assert event_names == [
        "meta",
        "delta",
        "delta",
        "done",
    ]

    # meta 事件中的 request_id 应与 Middleware 添加的响应头一致。
    assert (
        events[0][1]["request_id"]
        == response.headers["x-request-id"]
    )

    # 取出所有 delta 的 text，再按顺序拼接。
    delta_parts = [
        event_data["text"]
        for event_name, event_data in events
        if event_name == "delta"
    ]
    full_llm_content = "".join(delta_parts)

    # 拼接后的文本应该仍然是合法 JSON。
    parsed_llm_content = json.loads(full_llm_content)

    assert parsed_llm_content == {
        "summary": "测试摘要",
        "recommended_actions": ["检查连接"],
    }

    # 最后一个事件必须明确表示正常完成。
    assert events[-1] == (
        "done",
        {"status": "completed"},
    )

@pytest.mark.parametrize(
    (
        "emit_delta_before_error",
        "expected_event_names",
    ),
    [
        # 上游在产生任何文本之前失败。
        (
            False,
            ["meta", "error"],
        ),

        # 上游先产生一个文本增量，随后失败。
        (
            True,
            ["meta", "delta", "error"],
        ),
    ],
)
@pytest.mark.asyncio
async def test_stream_timeout_event(
    emit_delta_before_error: bool,
    expected_event_names: list[str],
) -> None:
    """流式超时应该以 error 结束，并且不能发送 done。"""

    fake_service = FakeTimeoutStreamService(
        emit_delta_before_error=emit_delta_before_error,
    )

    def override_get_triage_service() -> FakeTimeoutStreamService:
        """让 FastAPI 为本次测试注入超时 Fake Service。"""

        return fake_service

    app.dependency_overrides[
        get_triage_service
    ] = override_get_triage_service

    try:
        transport = ASGITransport(app=app)

        async with AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/api/v1/triage/stream",
                json=VALID_PAYLOAD,
            )

    finally:
        # 即使请求执行失败，也要移除全局依赖覆盖。
        app.dependency_overrides.clear()

    # SSE 已经发送了 meta，所以 HTTP 响应状态保持为 200。
    # 流内失败通过 error 事件表达，而不是修改成 HTTP 504。
    assert response.status_code == 200

    assert response.headers["content-type"].startswith(
        "text/event-stream"
    )

    events = parse_sse_events(response.text)

    event_names = [
        event_name
        for event_name, _event_data in events
    ]

    # 两组参数分别验证：
    # meta → error
    # meta → delta → error
    assert event_names == expected_event_names

    # error 必须是流的最后一个事件。
    assert events[-1][0] == "error"

    # 不论错误发生在什么阶段，都不能再发送 done。
    assert "done" not in event_names

    error_data = events[-1][1]

    # error 事件中的 request_id 应与响应头一致。
    assert (
        error_data["request_id"]
        == response.headers["x-request-id"]
    )

    # 客户端只看到稳定错误码和安全公开信息。
    assert error_data["error"] == {
        "code": "llm_timeout",
        "message": "LLM 服务响应超时",
    }

    # Service 中的内部诊断内容不能泄漏给客户端。
    assert (
        "测试使用的内部超时详情"
        not in response.text
    )