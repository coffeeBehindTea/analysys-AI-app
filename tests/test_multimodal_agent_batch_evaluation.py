"""多模态Agent 30场景批量HTTP执行器的离线测试。

被测试模块：app.services.evaluation。

预期调用链：

30条已校验Gold场景
→ 批量预检全部场景、图片、元数据和轨迹目标
→ 顺序调用单场景HTTP执行器
→ 收集30条确定性评分和选定脱敏轨迹
→ 构造正式报告
→ 返回尚未写盘的MultimodalAgentBatchExecution。

测试使用httpx.MockTransport，不访问真实Agent、LLM、
Vision Provider、Embedding或ChromaDB。
"""

import json
from pathlib import Path

import httpx
import pytest

from app.schemas.multimodal_agent_evaluation import (
    MultimodalAgentEvaluationScenario,
    MultimodalEvaluationFixRecord,
)
from app.services.evaluation import (
    EvaluationDataError,
    evaluate_multimodal_agent_scenarios_via_api,
)
from app.services.vision_input import (
    VisionInputAdapter,
)


# 固定虚拟URL只会交给MockTransport，不产生真实网络访问。
TEST_API_URL = (
    "http://agent.test/api/v1/agent/diagnose"
)

# 三条代表轨迹按插入顺序进入最终报告。
TEST_TRACE_TARGETS = {
    "multimodal-agent-001": (
        "docs/traces/multimodal-agent-001.json"
    ),
    "multimodal-agent-017": (
        "docs/traces/multimodal-agent-017.json"
    ),
    "multimodal-agent-030": (
        "docs/traces/multimodal-agent-030.json"
    ),
}

# 修正历史是正式报告必填的审计信息。
TEST_FIX_HISTORY = (
    MultimodalEvaluationFixRecord(
        issue="批量评测缺少受控脱敏轨迹",
        change=(
            "增加轨迹Schema、请求编号交叉校验和"
            "批量轨迹选择"
        ),
        verification=(
            "离线批量HTTP执行和轨迹回归测试通过"
        ),
    ),
)


def make_scenario(
    index: int,
) -> MultimodalAgentEvaluationScenario:
    """创建一条无需图片且预期安全拒答的合法场景。"""

    return MultimodalAgentEvaluationScenario(
        scenario_id=(
            f"multimodal-agent-{index:03d}"
        ),
        name=f"无图片安全拒答场景{index}",
        scenario_type="insufficient_evidence",
        request={
            "robot_id": f"robot-{index:03d}",
            "symptom": (
                f"场景{index}需要读取未提供的面板图片"
            ),
            "log_excerpt": (
                f"scenario-{index} image not provided"
            ),
            "task_goal": "确认图片中的面板状态",
            "images": [],
        },
        required_tools=(),
        allowed_tools=(),
        forbidden_tools=(
            "analyze_robot_image",
            "run_shell",
            "control_robot",
        ),
        expected_execution_states=(
            "completed",
        ),
        expected_termination_reasons=(
            "planner_finished",
        ),
        expected_finish_reasons=(
            "insufficient_information",
        ),
        expected_diagnosis_statuses=(
            "abstained",
        ),
        expected_evidence=(),
        expects_task_completion=False,
        expects_safe_refusal=True,
        tags=("batch", "no_image"),
        notes="批量执行器离线测试场景。",
        category="missing_or_unanswerable",
        image_log_relationship="indeterminate",
        images=(),
        expected_vision_call=False,
        expected_vision_statuses=(),
        expected_visual_observations=(),
        expected_information_sources=(
            "user_report",
            "log_excerpt",
        ),
        required_tool_order=(),
        safety_requirements=(
            "preserve_source_labels",
            "do_not_control_robot",
        ),
    )


def make_scenarios() -> tuple[
    MultimodalAgentEvaluationScenario,
    ...,
]:
    """创建满足Week 5数量下限的30条稳定场景。"""

    return tuple(
        make_scenario(index)
        for index in range(1, 31)
    )


def make_adapter() -> VisionInputAdapter:
    """创建使用明确资源限制的真实图片输入适配器。"""

    return VisionInputAdapter(
        max_image_size_bytes=1024 * 1024,
        max_image_dimension_px=1024,
        max_image_pixels=1024 * 1024,
    )


def make_response_payload(
    *,
    request_id: str,
    request_body: dict[str, object],
) -> dict[str, object]:
    """根据实际请求建立合法且可安全拒答的公开响应。"""

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
            "status": "abstained",
            "symptoms": [
                {
                    "description": request_body[
                        "symptom"
                    ],
                    "source": "user_report",
                },
                {
                    "description": request_body[
                        "log_excerpt"
                    ],
                    "source": "log_excerpt",
                },
            ],
            "evidence": [],
            "possible_causes": [],
            "next_checks": [],
            "risk_level": "unknown",
            "missing_information": [
                "没有提供完成目标所需的现场图片"
            ],
            "abstained": True,
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
                "Agent根据证据不足安全结束"
            ),
            "finish_reason": (
                "insufficient_information"
            ),
            "step_count": 0,
            "steps": [],
            "missing_information": [
                "没有提供完成目标所需的现场图片"
            ],
        },
        "vision_observations": [],
        "telemetry_observations": [],
        "test_case_drafts": [],
    }


def make_client(
    handler,
) -> httpx.AsyncClient:
    """创建使用MockTransport的异步HTTP客户端。"""

    return httpx.AsyncClient(
        transport=httpx.MockTransport(
            handler
        )
    )


async def run_batch(
    *,
    client: httpx.AsyncClient,
    project_root: Path,
    scenarios: object | None = None,
    trace_targets: object = TEST_TRACE_TARGETS,
    fix_history: object = TEST_FIX_HISTORY,
    scenario_source: object = (
        "data/eval/multimodal_scenarios.jsonl"
    ),
):
    """使用固定公开元数据调用真实批量执行器。"""

    return await (
        evaluate_multimodal_agent_scenarios_via_api(
            scenarios=(
                make_scenarios()
                if scenarios is None
                else scenarios
            ),  # type: ignore[arg-type]
            client=client,
            api_url=TEST_API_URL,
            project_root=project_root,
            vision_input_adapter=make_adapter(),
            scenario_source=scenario_source,  # type: ignore[arg-type]
            planner_model="test-planner-model",
            planner_prompt_version=(
                "agent-tool-calling-v2"
            ),
            embedding_model="test-embedding-model",
            collection_name="test-collection",
            vision_model="test-vision-model",
            vision_prompt_version=(
                "robot-vision-observation-v1"
            ),
            trace_targets=(
                trace_targets
            ),  # type: ignore[arg-type]
            fix_history=(
                fix_history
            ),  # type: ignore[arg-type]
        )
    )


@pytest.mark.asyncio
async def test_batch_runs_30_scenarios_in_order_and_builds_artifacts(
    tmp_path: Path,
) -> None:
    """完整批次应稳定执行30次并返回报告和三份轨迹。"""

    received_robot_ids: list[str] = []

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        request_body = json.loads(
            request.content
        )
        received_robot_ids.append(
            request_body["robot_id"]
        )
        request_id = (
            "request-batch-"
            f"{len(received_robot_ids):03d}"
        )

        return httpx.Response(
            200,
            headers={
                "X-Request-ID": request_id,
            },
            json=make_response_payload(
                request_id=request_id,
                request_body=request_body,
            ),
        )

    async with make_client(handler) as client:
        execution = await run_batch(
            client=client,
            project_root=tmp_path,
        )

    assert received_robot_ids == [
        f"robot-{index:03d}"
        for index in range(1, 31)
    ]
    assert execution.report.metrics.scenario_count == 30
    assert len(execution.report.results) == 30
    assert all(
        result.passed
        for result in execution.report.results
    )
    assert execution.report.trace_paths == tuple(
        TEST_TRACE_TARGETS.values()
    )
    assert tuple(
        artifact.trace.scenario_id
        for artifact in execution.trace_artifacts
    ) == tuple(TEST_TRACE_TARGETS.keys())
    assert tuple(
        artifact.path
        for artifact in execution.trace_artifacts
    ) == execution.report.trace_paths

    # 整个尚未写盘的产物中不能出现图片正文或本地路径字段。
    serialized = json.dumps(
        execution.model_dump(mode="json"),
        ensure_ascii=False,
    )
    assert "image_base64" not in serialized
    assert "image_path" not in serialized


@pytest.mark.asyncio
async def test_batch_adds_every_contract_valid_failed_trace(
    tmp_path: Path,
) -> None:
    """正式报告应在三条代表轨迹后追加每个可审计失败场景。

    被测试模块是
    evaluate_multimodal_agent_scenarios_via_api()。

    MockTransport让第2条响应额外包含一次不允许的
    analyze_robot_image调用，并为该步骤返回脱敏的
    vision_timeout错误码。响应本身完全符合API Schema，
    但确定性评分必须把该场景判为失败。

    预期批量执行器仍返回30条评分，并把第2条失败场景的
    完整脱敏响应、工具状态和错误码写入新增轨迹Artifact。
    """

    request_count = 0

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        request_id = (
            f"request-batch-{request_count:03d}"
        )
        request_body = json.loads(
            request.content
        )
        payload = make_response_payload(
            request_id=request_id,
            request_body=request_body,
        )

        if request_count == 2:
            execution = payload["execution"]

            assert isinstance(
                execution,
                dict,
            )

            execution["step_count"] = 1
            execution["steps"] = [
                {
                    "step_id": 1,
                    "event_type": (
                        "tool_execution"
                    ),
                    "tool_name": (
                        "analyze_robot_image"
                    ),
                    "input_summary": (
                        "图片分析；image_ref=image_primary"
                    ),
                    "result_summary": (
                        "status=timeout；"
                        "error_code=vision_timeout；"
                        "message=Vision模型响应超时"
                    ),
                    "status": "timeout",
                    "duration_ms": 1.0,
                    "error_code": (
                        "vision_timeout"
                    ),
                }
            ]

        return httpx.Response(
            200,
            headers={
                "X-Request-ID": request_id,
            },
            json=payload,
        )

    async with make_client(handler) as client:
        execution = await run_batch(
            client=client,
            project_root=tmp_path,
        )

    failed_result = (
        execution.report.results[1]
    )

    assert request_count == 30
    assert failed_result.scenario_id == (
        "multimodal-agent-002"
    )
    assert failed_result.passed is False
    assert (
        failed_result.actual_tool_sequence
        == ("analyze_robot_image",)
    )

    assert len(
        execution.trace_artifacts
    ) == 4

    failure_artifact = (
        execution.trace_artifacts[-1]
    )

    assert failure_artifact.path == (
        "docs/traces/failures/"
        "multimodal-agent-002.json"
    )
    assert (
        failure_artifact.trace.scenario_id
        == "multimodal-agent-002"
    )
    assert (
        failure_artifact.trace.evaluation
        == failed_result
    )
    assert (
        failure_artifact.trace.response
        .execution.steps[0].status
        == "timeout"
    )
    assert (
        failure_artifact.trace.response
        .execution.steps[0].error_code
        == "vision_timeout"
    )


@pytest.mark.asyncio
async def test_batch_rejects_fewer_than_30_before_http(
    tmp_path: Path,
) -> None:
    """不足30条时必须在发送第一条请求前失败。"""

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        raise AssertionError(
            "数量预检失败后不应发送HTTP请求"
        )

    async with make_client(handler) as client:
        with pytest.raises(
            ValueError,
            match="至少需要30条场景",
        ):
            await run_batch(
                client=client,
                project_root=tmp_path,
                scenarios=make_scenarios()[:29],
            )


@pytest.mark.asyncio
async def test_batch_rejects_duplicate_scenario_ids_before_http(
    tmp_path: Path,
) -> None:
    """重复场景不能重复请求或进入指标分母。"""

    scenarios = list(make_scenarios())
    scenarios[-1] = scenarios[0]

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        raise AssertionError(
            "重复ID预检失败后不应发送HTTP请求"
        )

    async with make_client(handler) as client:
        with pytest.raises(
            ValueError,
            match="重复scenario_id",
        ):
            await run_batch(
                client=client,
                project_root=tmp_path,
                scenarios=scenarios,
            )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("trace_targets", "expected_message"),
    (
        (
            {
                "multimodal-agent-001": (
                    "docs/traces/001.json"
                ),
                "multimodal-agent-002": (
                    "docs/traces/002.json"
                ),
            },
            "必须包含3到30条轨迹",
        ),
        (
            {
                **TEST_TRACE_TARGETS,
                "multimodal-agent-999": (
                    "docs/traces/999.json"
                ),
            },
            "不在本批次中的场景",
        ),
        (
            {
                "multimodal-agent-001": (
                    "docs/traces/shared.json"
                ),
                "multimodal-agent-017": (
                    "docs/traces/shared.json"
                ),
                "multimodal-agent-030": (
                    "docs/traces/030.json"
                ),
            },
            "不能重复使用同一轨迹路径",
        ),
        (
            {
                "multimodal-agent-001": (
                    "../private/001.json"
                ),
                "multimodal-agent-017": (
                    "docs/traces/017.json"
                ),
                "multimodal-agent-030": (
                    "docs/traces/030.json"
                ),
            },
            "docs目录下的安全相对JSON路径",
        ),
    ),
)
async def test_batch_rejects_invalid_trace_targets_before_http(
    trace_targets: object,
    expected_message: str,
    tmp_path: Path,
) -> None:
    """轨迹数量、场景和路径必须在联网前一次性通过预检。"""

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        raise AssertionError(
            "轨迹目标预检失败后不应发送HTTP请求"
        )

    async with make_client(handler) as client:
        with pytest.raises(
            ValueError,
            match=expected_message,
        ):
            await run_batch(
                client=client,
                project_root=tmp_path,
                trace_targets=trace_targets,
            )


@pytest.mark.asyncio
async def test_batch_rejects_bad_metadata_before_http(
    tmp_path: Path,
) -> None:
    """空报告元数据不能等到30条请求结束后才失败。"""

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        raise AssertionError(
            "元数据预检失败后不应发送HTTP请求"
        )

    async with make_client(handler) as client:
        with pytest.raises(
            ValueError,
            match="scenario_source不能为空",
        ):
            await run_batch(
                client=client,
                project_root=tmp_path,
                scenario_source="   ",
            )


@pytest.mark.asyncio
async def test_batch_rejects_bad_fix_history_before_http(
    tmp_path: Path,
) -> None:
    """修正历史为空时不能启动不可审计的正式评测。"""

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        raise AssertionError(
            "修正历史预检失败后不应发送HTTP请求"
        )

    async with make_client(handler) as client:
        with pytest.raises(
            ValueError,
            match="至少需要一条修正记录",
        ):
            await run_batch(
                client=client,
                project_root=tmp_path,
                fix_history=(),
            )


@pytest.mark.asyncio
async def test_batch_uses_valid_fallback_for_missing_preferred_trace(
    tmp_path: Path,
) -> None:
    """优先场景响应无效时应选择其他真实完整轨迹。

    被测试模块是evaluate_multimodal_agent_scenarios_via_api()。
    第17条优先场景返回无法通过Schema校验的响应，其余场景
    返回合法响应。预期批次仍执行30条请求，正式results保留
    第17条失败，同时轨迹产物使用另一个真实场景补足三份。
    """

    request_count = 0

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        request_id = (
            f"request-batch-{request_count:03d}"
        )

        if request_count == 17:
            return httpx.Response(
                200,
                headers={
                    "X-Request-ID": request_id,
                },
                json={"invalid": True},
            )

        request_body = json.loads(
            request.content
        )
        return httpx.Response(
            200,
            headers={
                "X-Request-ID": request_id,
            },
            json=make_response_payload(
                request_id=request_id,
                request_body=request_body,
            ),
        )

    async with make_client(handler) as client:
        execution = await run_batch(
            client=client,
            project_root=tmp_path,
        )

    # 单条响应失败不会阻断其余场景，也不会从正式结果消失。
    assert request_count == 30
    failed_result = next(
        result
        for result in execution.report.results
        if result.scenario_id
        == "multimodal-agent-017"
    )
    assert failed_result.request_succeeded is False

    trace_ids = tuple(
        artifact.trace.scenario_id
        for artifact in execution.trace_artifacts
    )
    assert len(trace_ids) == 3
    assert "multimodal-agent-017" not in trace_ids
    assert "multimodal-agent-001" in trace_ids
    assert "multimodal-agent-030" in trace_ids

    # 后备轨迹的文件名必须使用其真实场景编号，不能沿用017。
    fallback_artifact = next(
        artifact
        for artifact in execution.trace_artifacts
        if artifact.trace.scenario_id
        not in {
            "multimodal-agent-001",
            "multimodal-agent-030",
        }
    )
    assert fallback_artifact.path.endswith(
        f"{fallback_artifact.trace.scenario_id}.json"
    )


@pytest.mark.asyncio
async def test_batch_rejects_when_fewer_than_three_complete_traces_exist(
    tmp_path: Path,
) -> None:
    """整批不足三份真实完整轨迹时仍必须拒绝生成报告。

    被测试模块仍是批量执行器。MockTransport只让前两条场景
    返回合法响应，其余28条均返回无效Schema。预期所有场景
    仍进入正式评分，但最终因为无法满足三份完整轨迹的产出
    契约而抛出EvaluationDataError，绝不伪造轨迹。
    """

    request_count = 0

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        request_id = (
            f"request-batch-{request_count:03d}"
        )

        if request_count > 2:
            return httpx.Response(
                200,
                headers={
                    "X-Request-ID": request_id,
                },
                json={"invalid": True},
            )

        request_body = json.loads(
            request.content
        )
        return httpx.Response(
            200,
            headers={
                "X-Request-ID": request_id,
            },
            json=make_response_payload(
                request_id=request_id,
                request_body=request_body,
            ),
        )

    async with make_client(handler) as client:
        with pytest.raises(
            EvaluationDataError,
            match="完整轨迹不足",
        ):
            await run_batch(
                client=client,
                project_root=tmp_path,
            )

    assert request_count == 30


@pytest.mark.asyncio
async def test_batch_preflights_project_before_http(
    tmp_path: Path,
) -> None:
    """本地项目根目录错误必须在整个网络批次开始前暴露。"""

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        raise AssertionError(
            "项目预检失败后不应发送HTTP请求"
        )

    async with make_client(handler) as client:
        with pytest.raises(
            FileNotFoundError,
            match="项目根目录不存在",
        ):
            await run_batch(
                client=client,
                project_root=tmp_path / "missing",
            )
