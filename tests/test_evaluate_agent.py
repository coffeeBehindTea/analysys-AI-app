"""Agent评测脚本单场景HTTP执行器的离线测试。

被测试模块：

scripts.evaluate_agent.evaluate_agent_scenario_via_api

预期调用链：

AgentEvaluationScenario
→ 将scenario.request序列化为公开API请求体
→ httpx.AsyncClient.post()
→ 成功响应转换成AgentDiagnosisResponse
→ evaluate_agent_response()评分
→ 返回AgentScenarioEvaluation。

超时、连接失败、非2xx、非法JSON和无效响应契约
都必须转换成可审计的失败结果，而不是终止整批评测。
"""

import json

import httpx
import pytest

from app.schemas.agent_evaluation import (
    AgentEvaluationScenario,
)
from scripts.evaluate_agent import (
    evaluate_agent_scenarios_via_api,
    evaluate_agent_scenario_via_api,
)


# 固定请求和证据标识用于验证HTTP响应到评分结果的转换。
TEST_REQUEST_ID = "request-agent-http-001"
TEST_DOCUMENT_ID = "a" * 64
TEST_CHUNK_ID = f"{TEST_DOCUMENT_ID}:000003"
TEST_SOURCE_FILE = "仓储机器人故障说明.txt"
TEST_LOCATION = "section: ERR-NET-4001"


def make_scenario(
    *,
    scenario_id: str = "agent-001",
    robot_id: str = "robot-001",
) -> AgentEvaluationScenario:
    """创建要求知识检索和遥测的完成型场景。"""

    return AgentEvaluationScenario(
        scenario_id=scenario_id,
        name="网络恢复后的多工具诊断",
        scenario_type=(
            "knowledge_and_telemetry"
        ),
        request={
            "robot_id": robot_id,
            "symptom": (
                "网络恢复后机器人仍处于暂停状态"
            ),
            "log_excerpt": (
                "ERR-NET-4001 heartbeat timeout"
            ),
            "task_goal": (
                "核对知识证据和当前遥测"
            ),
        },
        required_tools=(
            "search_knowledge",
            "get_robot_telemetry",
        ),
        allowed_tools=(
            "search_knowledge",
            "get_robot_telemetry",
        ),
        forbidden_tools=(
            "run_shell",
        ),
        expected_execution_states=(
            "completed",
        ),
        expected_termination_reasons=(
            "planner_finished",
        ),
        expected_finish_reasons=(
            "task_completed",
        ),
        expected_diagnosis_statuses=(
            "completed",
        ),
        expected_evidence=(
            {
                "source_file": TEST_SOURCE_FILE,
                "page_or_section": TEST_LOCATION,
            },
        ),
        expects_task_completion=True,
        expects_safe_refusal=False,
        tags=(
            "multi_tool",
            "knowledge",
            "telemetry",
        ),
        notes=(
            "验证HTTP执行器成功和失败转换。"
        ),
    )


def make_success_payload(
    *,
    request_id: str = TEST_REQUEST_ID,
) -> dict[str, object]:
    """创建符合AgentDiagnosisResponse契约的JSON响应。"""

    steps = [
        {
            "step_id": 1,
            "event_type": "tool_execution",
            "tool_name": "search_knowledge",
            "input_summary": (
                "知识库检索；query_chars=25；top_k=5"
            ),
            "result_summary": (
                "status=success；evidence_count=1"
            ),
            "status": "success",
            "duration_ms": 24.5,
            "error_code": None,
        },
        {
            "step_id": 2,
            "event_type": "tool_execution",
            "tool_name": "get_robot_telemetry",
            "input_summary": (
                "模拟遥测读取；robot_id=robot-001"
            ),
            "result_summary": (
                "status=success；snapshot_count=1"
            ),
            "status": "success",
            "duration_ms": 1.5,
            "error_code": None,
        },
    ]

    return {
        "request_id": request_id,
        "diagnosis": {
            "request_id": request_id,
            "prompt_version": (
                "agent-tool-calling-v2"
            ),
            "gate_version": (
                "agent-confirmed-evidence-v1"
            ),
            "status": "completed",
            "symptoms": [
                {
                    "description": (
                        "网络恢复后仍未继续任务"
                    ),
                    "source": "user_report",
                },
                {
                    "description": (
                        "ERR-NET-4001 heartbeat timeout"
                    ),
                    "source": "log_excerpt",
                },
            ],
            "evidence": [
                {
                    "chunk_id": TEST_CHUNK_ID,
                    "document_id": TEST_DOCUMENT_ID,
                    "source_file": TEST_SOURCE_FILE,
                    "page_or_section": TEST_LOCATION,
                    "rank": 1,
                    "rrf_score": 0.0327,
                    "vector_similarity": 0.71,
                    "excerpt": (
                        "通信恢复后不自动继续旧任务。"
                    ),
                }
            ],
            "possible_causes": [
                {
                    "description": (
                        "旧任务仍在等待调度状态核对"
                    ),
                    "evidence_chunk_ids": [
                        TEST_CHUNK_ID
                    ],
                }
            ],
            "next_checks": [],
            "risk_level": "medium",
            "missing_information": [],
            "abstained": False,
        },
        "execution": {
            "planner_prompt_version": (
                "agent-tool-calling-v2"
            ),
            "state": "completed",
            "termination_reason": (
                "planner_finished"
            ),
            "termination_message": (
                "Agent规划正常结束"
            ),
            "finish_reason": "task_completed",
            "step_count": 2,
            "steps": steps,
            "missing_information": [],
        },
        "telemetry_observations": [],
        "test_case_drafts": [],
    }


def make_eight_scenarios(
) -> list[AgentEvaluationScenario]:
    """创建任务6要求的最小八场景有序批次。"""

    return [
        make_scenario(
            scenario_id=f"agent-{index:03d}",
            robot_id=f"robot-{index:03d}",
        )
        for index in range(1, 9)
    ]


@pytest.mark.asyncio
async def test_single_scenario_success_posts_public_request_and_scores(
) -> None:
    """合法200响应应被验证并交给确定性评分器。"""

    captured_request_body: dict[
        str,
        object,
    ] = {}

    async def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        nonlocal captured_request_body

        captured_request_body = json.loads(
            request.content.decode("utf-8")
        )

        return httpx.Response(
            status_code=200,
            json=make_success_payload(),
            headers={
                "X-Request-ID": TEST_REQUEST_ID,
            },
            request=request,
        )

    transport = httpx.MockTransport(
        handler
    )

    async with httpx.AsyncClient(
        transport=transport,
    ) as client:
        result = (
            await evaluate_agent_scenario_via_api(
                scenario=make_scenario(),
                client=client,
                api_url=(
                    "http://testserver/"
                    "api/v1/agent/diagnose"
                ),
            )
        )

    # 评测脚本只能发送公开请求契约字段，
    # 不能夹带tools、max_steps等权限配置。
    assert captured_request_body == {
        "robot_id": "robot-001",
        "symptom": (
            "网络恢复后机器人仍处于暂停状态"
        ),
        "log_excerpt": (
            "ERR-NET-4001 heartbeat timeout"
        ),
        "task_goal": (
            "核对知识证据和当前遥测"
        ),
    }

    assert result.request_succeeded is True
    assert result.http_status_code == 200
    assert result.request_id == TEST_REQUEST_ID
    assert result.tool_selection_correct is True
    assert result.task_completion_correct is True
    assert result.passed is True


@pytest.mark.asyncio
async def test_http_error_becomes_auditable_failure(
) -> None:
    """非2xx响应应保留状态码和追踪ID并继续评测。"""

    async def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            status_code=502,
            json={
                "error": {
                    "code": "llm_upstream_error",
                    "message": "上游服务不可用",
                }
            },
            headers={
                "X-Request-ID": "request-http-502",
            },
            request=request,
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            handler
        ),
    ) as client:
        result = (
            await evaluate_agent_scenario_via_api(
                scenario=make_scenario(),
                client=client,
                api_url=(
                    "http://testserver/"
                    "api/v1/agent/diagnose"
                ),
            )
        )

    assert result.request_succeeded is False
    assert result.http_status_code == 502
    assert result.request_id == "request-http-502"
    assert result.request_error == (
        "Agent API返回错误状态：502"
    )
    assert result.actual_tool_sequence == ()


@pytest.mark.asyncio
async def test_timeout_becomes_failure_without_fake_status(
) -> None:
    """请求超时没有HTTP响应，状态码和请求ID都应为空。"""

    async def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        raise httpx.ReadTimeout(
            "simulated timeout",
            request=request,
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            handler
        ),
    ) as client:
        result = (
            await evaluate_agent_scenario_via_api(
                scenario=make_scenario(),
                client=client,
                api_url=(
                    "http://testserver/"
                    "api/v1/agent/diagnose"
                ),
            )
        )

    assert result.request_succeeded is False
    assert result.http_status_code is None
    assert result.request_id is None
    assert result.request_error == (
        "Agent API请求超时"
    )


@pytest.mark.asyncio
async def test_connection_error_becomes_sanitized_failure(
) -> None:
    """连接异常不能把底层地址或异常正文写进评测报告。"""

    async def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        raise httpx.ConnectError(
            "secret internal host detail",
            request=request,
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            handler
        ),
    ) as client:
        result = (
            await evaluate_agent_scenario_via_api(
                scenario=make_scenario(),
                client=client,
                api_url=(
                    "http://testserver/"
                    "api/v1/agent/diagnose"
                ),
            )
        )

    assert result.request_error == (
        "无法连接Agent API"
    )
    assert "secret" not in result.request_error


@pytest.mark.asyncio
async def test_invalid_json_becomes_parse_failure(
) -> None:
    """HTTP 200但正文不是JSON时仍属于请求评分失败。"""

    async def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            content=b"not-json",
            headers={
                "X-Request-ID": "request-invalid-json",
            },
            request=request,
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            handler
        ),
    ) as client:
        result = (
            await evaluate_agent_scenario_via_api(
                scenario=make_scenario(),
                client=client,
                api_url=(
                    "http://testserver/"
                    "api/v1/agent/diagnose"
                ),
            )
        )

    assert result.http_status_code == 200
    assert result.request_id == (
        "request-invalid-json"
    )
    assert result.request_error == (
        "Agent API返回非法JSON"
    )


@pytest.mark.asyncio
async def test_invalid_response_contract_becomes_failure(
) -> None:
    """合法JSON缺少响应字段时不能进入成功评分路径。"""

    async def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={
                "request_id": "request-invalid-contract"
            },
            headers={
                "X-Request-ID": (
                    "request-invalid-contract"
                ),
            },
            request=request,
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            handler
        ),
    ) as client:
        result = (
            await evaluate_agent_scenario_via_api(
                scenario=make_scenario(),
                client=client,
                api_url=(
                    "http://testserver/"
                    "api/v1/agent/diagnose"
                ),
            )
        )

    assert result.request_succeeded is False
    assert result.request_error == (
        "Agent API响应不符合数据契约"
    )
    assert result.step_count == 0


@pytest.mark.asyncio
async def test_batch_runs_eight_scenarios_in_order_and_builds_report(
) -> None:
    """最小合格批次应顺序调用八次并生成完整报告。"""

    observed_robot_ids: list[str] = []

    async def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        request_body = json.loads(
            request.content.decode("utf-8")
        )
        observed_robot_ids.append(
            request_body["robot_id"]
        )

        request_id = (
            "request-batch-"
            f"{len(observed_robot_ids):03d}"
        )

        return httpx.Response(
            status_code=200,
            json=make_success_payload(
                request_id=request_id,
            ),
            headers={
                "X-Request-ID": request_id,
            },
            request=request,
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            handler
        ),
    ) as client:
        report = (
            await evaluate_agent_scenarios_via_api(
                scenarios=make_eight_scenarios(),
                client=client,
                api_url=(
                    "http://testserver/"
                    "api/v1/agent/diagnose"
                ),
                evaluation_version=(
                    "agent-evaluation-v1"
                ),
                scenario_source=(
                    "data/eval/agent_scenarios.jsonl"
                ),
                planner_model="deepseek-chat",
                planner_prompt_version=(
                    "agent-tool-calling-v2"
                ),
                embedding_model="embedding-3",
                collection_name=(
                    "robot_knowledge_week3_"
                    "vector_baseline"
                ),
            )
        )

    assert observed_robot_ids == [
        f"robot-{index:03d}"
        for index in range(1, 9)
    ]
    assert [
        result.scenario_id
        for result in report.results
    ] == [
        f"agent-{index:03d}"
        for index in range(1, 9)
    ]
    assert report.metrics.scenario_count == 8
    assert report.metrics.request_success_rate == 1.0
    assert report.metrics.scenario_pass_rate == 1.0
    assert report.metrics.task_completion_rate == 1.0
    assert report.metrics.citation_correctness == 1.0
    assert report.metrics.citation_coverage == 1.0
    assert report.metrics.average_steps == 2.0

    # 评测报告不得包含每次运行都会变化的生成时间。
    assert "generated_at" not in report.model_dump()


@pytest.mark.asyncio
async def test_batch_continues_after_one_http_failure(
) -> None:
    """第三条失败后仍应运行第四至第八条并如实降低指标。"""

    call_count = 0

    async def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        nonlocal call_count
        call_count += 1

        request_id = (
            f"request-mixed-{call_count:03d}"
        )

        if call_count == 3:
            return httpx.Response(
                status_code=502,
                json={
                    "error": {
                        "code": "llm_upstream_error",
                        "message": "上游不可用",
                    }
                },
                headers={
                    "X-Request-ID": request_id,
                },
                request=request,
            )

        return httpx.Response(
            status_code=200,
            json=make_success_payload(
                request_id=request_id,
            ),
            headers={
                "X-Request-ID": request_id,
            },
            request=request,
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            handler
        ),
    ) as client:
        report = (
            await evaluate_agent_scenarios_via_api(
                scenarios=make_eight_scenarios(),
                client=client,
                api_url=(
                    "http://testserver/"
                    "api/v1/agent/diagnose"
                ),
                evaluation_version=(
                    "agent-evaluation-v1"
                ),
                scenario_source=(
                    "data/eval/agent_scenarios.jsonl"
                ),
                planner_model="deepseek-chat",
                planner_prompt_version=(
                    "agent-tool-calling-v2"
                ),
                embedding_model="embedding-3",
                collection_name=(
                    "robot_knowledge_week3_"
                    "vector_baseline"
                ),
            )
        )

    assert call_count == 8
    assert report.results[2].scenario_id == (
        "agent-003"
    )
    assert report.results[2].request_succeeded is False
    assert report.results[2].request_error == (
        "Agent API返回错误状态：502"
    )
    assert report.results[3].request_succeeded is True

    assert report.metrics.request_success_count == 7
    assert report.metrics.request_success_rate == 0.875
    assert report.metrics.passed_scenario_count == 7
    assert report.metrics.scenario_pass_rate == 0.875
    assert report.metrics.completed_task_count == 7
    assert report.metrics.task_completion_rate == 0.875
    assert report.metrics.citation_correctness == 1.0
    assert report.metrics.citation_coverage == 0.875
    assert report.metrics.average_steps == 1.75


@pytest.mark.asyncio
async def test_batch_rejects_fewer_than_eight_before_http(
) -> None:
    """不足八条不能冒充任务6固定批量评测。"""

    call_count = 0

    async def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(
            status_code=500,
            request=request,
        )

    scenarios = make_eight_scenarios()[:7]

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            handler
        ),
    ) as client:
        with pytest.raises(
            ValueError,
            match=(
                "Agent批量评测至少需要8条场景"
            ),
        ):
            await evaluate_agent_scenarios_via_api(
                scenarios=scenarios,
                client=client,
                api_url=(
                    "http://testserver/"
                    "api/v1/agent/diagnose"
                ),
                evaluation_version=(
                    "agent-evaluation-v1"
                ),
                scenario_source="scenarios.jsonl",
                planner_model="deepseek-chat",
                planner_prompt_version=(
                    "agent-tool-calling-v2"
                ),
                embedding_model="embedding-3",
                collection_name="test_collection",
            )

    assert call_count == 0


@pytest.mark.asyncio
async def test_batch_rejects_duplicate_ids_before_http(
) -> None:
    """重复场景编号应在发出第一条请求前被拒绝。"""

    call_count = 0

    async def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(
            status_code=500,
            request=request,
        )

    scenarios = make_eight_scenarios()
    scenarios[-1] = scenarios[-2]

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            handler
        ),
    ) as client:
        with pytest.raises(
            ValueError,
            match=(
                "Agent批量评测场景包含"
                "重复scenario_id"
            ),
        ):
            await evaluate_agent_scenarios_via_api(
                scenarios=scenarios,
                client=client,
                api_url=(
                    "http://testserver/"
                    "api/v1/agent/diagnose"
                ),
                evaluation_version=(
                    "agent-evaluation-v1"
                ),
                scenario_source="scenarios.jsonl",
                planner_model="deepseek-chat",
                planner_prompt_version=(
                    "agent-tool-calling-v2"
                ),
                embedding_model="embedding-3",
                collection_name="test_collection",
            )

    assert call_count == 0
