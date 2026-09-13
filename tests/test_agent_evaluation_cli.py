"""Agent真实评测编排入口的离线测试。

被测试模块：

scripts.evaluate_agent

被测试函数：

1. run_real_evaluation；
2. main。

预期调用链：

命令行参数和Settings
→ 加载固定场景JSONL
→ 创建带超时的httpx.AsyncClient
→ 调用批量Agent评测函数
→ 写出JSON和Markdown
→ main打印终端摘要。

本测试会替换场景加载、批量HTTP执行和报告写出，
因此不会连接真实Agent API、LLM、Embedding或ChromaDB。
"""

import argparse

from pathlib import Path

import pytest

from app.agent.openai_planner import (
    AGENT_PLANNER_PROMPT_VERSION,
)
from app.config import (
    Settings,
)
from scripts import evaluate_agent as module


# 这些常量代表报告必须记录的运行元数据。
#
# 测试通过固定值确认run_real_evaluation没有：
#
# 1. 丢失配置；
# 2. 把LLM模型和Embedding模型写反；
# 3. 静默改写Collection名称。
TEST_PLANNER_MODEL = "planner-test-model"
TEST_EMBEDDING_MODEL = "embedding-test-model"
TEST_COLLECTION_NAME = "agent_test_collection"
TEST_API_URL = (
    "http://127.0.0.1:8000"
    "/api/v1/agent/diagnose"
)


def make_settings() -> Settings:
    """创建只供报告元数据使用的Settings。"""

    return Settings(
        llm_model=TEST_PLANNER_MODEL,
        embedding_model=TEST_EMBEDDING_MODEL,
        chroma_collection_name=(
            TEST_COLLECTION_NAME
        ),
    )


@pytest.mark.asyncio
async def test_run_real_evaluation_connects_complete_flow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """真实编排函数应加载场景、批量执行并写出同一报告。"""

    scenarios_path = (
        tmp_path / "agent_scenarios.jsonl"
    )
    json_output = (
        tmp_path / "agent-evaluation.json"
    )
    markdown_output = (
        tmp_path / "agent-evaluation.md"
    )

    # 使用8个哨兵对象模拟已经加载的8条场景。
    # 它们不会进入真实评分器，因为批量函数也会被替换。
    scenario_sentinels = tuple(
        object()
        for _ in range(8)
    )
    report_sentinel = object()
    captured: dict[str, object] = {}

    def fake_load_scenarios(
        path: Path,
    ) -> list[object]:
        """记录加载路径并返回固定场景集合。"""

        captured["loaded_path"] = path
        return list(scenario_sentinels)

    async def fake_batch_evaluation(
        **kwargs: object,
    ) -> object:
        """记录传给批量执行器的全部编排参数。"""

        captured["batch_kwargs"] = kwargs

        # AsyncClient已经在run_real_evaluation中创建，
        # 此处只检查超时配置，不发送HTTP请求。
        client = kwargs["client"]
        captured["client_timeout"] = (
            client.timeout.read
        )

        return report_sentinel

    def fake_write_reports(
        **kwargs: object,
    ) -> None:
        """记录报告对象和两个输出路径。"""

        captured["write_kwargs"] = kwargs

    monkeypatch.setattr(
        module,
        "load_agent_evaluation_scenarios",
        fake_load_scenarios,
    )
    monkeypatch.setattr(
        module,
        "evaluate_agent_scenarios_via_api",
        fake_batch_evaluation,
    )
    monkeypatch.setattr(
        module,
        "write_reports",
        fake_write_reports,
    )

    result = await module.run_real_evaluation(
        settings=make_settings(),
        scenarios_path=scenarios_path,
        api_url=TEST_API_URL,
        json_output=json_output,
        markdown_output=markdown_output,
        timeout_seconds=45.0,
    )

    assert result is report_sentinel
    assert captured["loaded_path"] == (
        scenarios_path
    )
    assert captured["client_timeout"] == 45.0

    batch_kwargs = captured["batch_kwargs"]

    assert batch_kwargs["scenarios"] == (
        list(scenario_sentinels)
    )
    assert batch_kwargs["api_url"] == TEST_API_URL
    assert batch_kwargs["evaluation_version"] == (
        "agent-evaluation-v1"
    )
    assert batch_kwargs["scenario_source"] == str(
        scenarios_path
    )
    assert batch_kwargs["planner_model"] == (
        TEST_PLANNER_MODEL
    )
    assert batch_kwargs[
        "planner_prompt_version"
    ] == AGENT_PLANNER_PROMPT_VERSION
    assert batch_kwargs["embedding_model"] == (
        TEST_EMBEDDING_MODEL
    )
    assert batch_kwargs["collection_name"] == (
        TEST_COLLECTION_NAME
    )

    assert captured["write_kwargs"] == {
        "report": report_sentinel,
        "json_output": json_output,
        "markdown_output": markdown_output,
    }


@pytest.mark.asyncio
async def test_run_real_evaluation_rejects_timeout_before_loading(
    tmp_path: Path,
) -> None:
    """程序化调用使用非法超时时也必须在读文件前失败。"""

    with pytest.raises(
        ValueError,
        match="timeout_seconds必须大于0",
    ):
        await module.run_real_evaluation(
            settings=make_settings(),
            scenarios_path=(
                tmp_path / "missing.jsonl"
            ),
            api_url=TEST_API_URL,
            json_output=(
                tmp_path / "report.json"
            ),
            markdown_output=(
                tmp_path / "report.md"
            ),
            timeout_seconds=0.0,
        )


def test_main_connects_args_settings_runner_and_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同步入口应只负责编排，不重复实现异步评测逻辑。"""

    args = argparse.Namespace(
        scenarios=Path(
            "data/eval/agent_scenarios.jsonl"
        ),
        api_url=TEST_API_URL,
        timeout_seconds=120.0,
        json_output=Path(
            "docs/agent-evaluation.json"
        ),
        markdown_output=Path(
            "docs/agent-evaluation.md"
        ),
    )
    settings = make_settings()
    report_sentinel = object()
    captured: dict[str, object] = {}

    async def fake_run_real_evaluation(
        **kwargs: object,
    ) -> object:
        """记录main传给异步入口的参数。"""

        captured["run_kwargs"] = kwargs
        return report_sentinel

    def fake_print_summary(
        **kwargs: object,
    ) -> None:
        """记录main传给终端摘要函数的数据。"""

        captured["summary_kwargs"] = kwargs

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: args,
    )
    monkeypatch.setattr(
        module,
        "get_settings",
        lambda: settings,
    )
    monkeypatch.setattr(
        module,
        "run_real_evaluation",
        fake_run_real_evaluation,
    )
    monkeypatch.setattr(
        module,
        "print_summary",
        fake_print_summary,
    )

    module.main()

    assert captured["run_kwargs"] == {
        "settings": settings,
        "scenarios_path": args.scenarios,
        "api_url": args.api_url,
        "json_output": args.json_output,
        "markdown_output": (
            args.markdown_output
        ),
        "timeout_seconds": (
            args.timeout_seconds
        ),
    }
    assert captured["summary_kwargs"] == {
        "report": report_sentinel,
        "json_path": args.json_output,
        "markdown_path": (
            args.markdown_output
        ),
    }
