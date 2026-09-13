"""多模态Agent评测请求准备器的离线测试。

被测试模块：app.services.evaluation。

预期流程：

已校验多模态场景
→ 安全解析项目相对图片路径
→ 读取图片并核对SHA-256
→ VisionImagePayload
→ VisionInputAdapter验证真实格式和尺寸
→ AgentDiagnosisRequest
→ 受控HTTP请求字典。

测试图片由Pillow写入pytest临时目录；本文件不联网。
"""

from base64 import (
    b64encode,
)
from hashlib import (
    sha256,
)
from pathlib import Path

from PIL import Image
import pytest

from app.schemas.multimodal_agent_evaluation import (
    MultimodalAgentEvaluationScenario,
)
from app.services.evaluation import (
    EvaluationDataError,
    prepare_multimodal_agent_request,
    serialize_multimodal_agent_request_for_http,
)
from app.services.vision_input import (
    VisionInputAdapter,
)


# 所有测试使用同一个受支持的小尺寸PNG相对位置。
TEST_IMAGE_RELATIVE_PATH = (
    "data/images/panel.png"
)
TEST_MISMATCH_RELATIVE_PATH = (
    "data/images/panel.jpg"
)


def make_adapter() -> VisionInputAdapter:
    """创建使用小而明确资源上限的真实输入适配器。"""

    return VisionInputAdapter(
        max_image_size_bytes=1024 * 1024,
        max_image_dimension_px=1024,
        max_image_pixels=1024 * 1024,
    )


def write_test_png(
    project_root: Path,
) -> tuple[Path, bytes]:
    """在临时项目的data目录生成一张8×6 PNG。"""

    image_path = (
        project_root
        / "data"
        / "images"
        / "panel.png"
    )
    image_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Image.new(
        "RGB",
        (8, 6),
        color=(20, 150, 80),
    ).save(
        image_path,
        format="PNG",
    )

    return image_path, image_path.read_bytes()


def make_scenario(
    *,
    image_sha256: str,
    mime_type: str = "image/png",
    include_image: bool = True,
) -> MultimodalAgentEvaluationScenario:
    """构造包含一张图片或不需要图片的合法场景。"""

    if include_image:
        scenario_type = "multi_tool_diagnosis"
        required_tools = (
            "analyze_robot_image",
            "search_knowledge",
        )
        images = (
            {
                "image_path": (
                    TEST_MISMATCH_RELATIVE_PATH
                    if mime_type == "image/jpeg"
                    else TEST_IMAGE_RELATIVE_PATH
                ),
                "image_sha256": image_sha256,
                "mime_type": mime_type,
                "analysis_goal": (
                    "只读取图片中可见的故障码"
                ),
                "detail": "auto",
            },
        )
        expected_vision_statuses = (
            "completed",
        )
        expected_visual_observations = (
            "面板显示ERR-DEMO-1001",
        )
        expected_sources = (
            "user_report",
            "log_excerpt",
            "vision_model",
            "knowledge_base",
        )
        required_tool_order = required_tools
        expected_finish_reasons = (
            "task_completed",
        )
        expected_diagnosis_statuses = (
            "completed",
        )
        expected_evidence = (
            {
                "source_file": (
                    "仓储机器人故障说明.txt"
                ),
                "page_or_section": (
                    "section: ERR-DEMO-1001"
                ),
            },
        )
        expects_task_completion = True
        expects_safe_refusal = False
        forbidden_tools = (
            "run_shell",
            "control_robot",
        )
    else:
        scenario_type = "insufficient_evidence"
        required_tools = ()
        images = ()
        expected_vision_statuses = ()
        expected_visual_observations = ()
        expected_sources = (
            "user_report",
            "log_excerpt",
        )
        required_tool_order = required_tools
        expected_finish_reasons = (
            "insufficient_information",
        )
        expected_diagnosis_statuses = (
            "abstained",
        )
        expected_evidence = ()
        expects_task_completion = False
        expects_safe_refusal = True
        # 无图片场景明确禁止Vision工具，避免Agent
        # 在没有受控图片输入时自行发起视觉调用。
        forbidden_tools = (
            "run_shell",
            "control_robot",
            "analyze_robot_image",
        )

    return MultimodalAgentEvaluationScenario(
        scenario_id="multimodal-agent-001",
        name="评测请求准备场景",
        scenario_type=scenario_type,
        request={
            "robot_id": "robot-001",
            "symptom": "机器人面板显示故障",
            "log_excerpt": "ERR-DEMO-1001",
            "task_goal": "读取图片并核对知识证据",
            "images": [],
        },
        required_tools=required_tools,
        allowed_tools=required_tools,
        forbidden_tools=forbidden_tools,
        expected_execution_states=(
            "completed",
        ),
        expected_termination_reasons=(
            "planner_finished",
        ),
        expected_finish_reasons=(
            expected_finish_reasons
        ),
        expected_diagnosis_statuses=(
            expected_diagnosis_statuses
        ),
        expected_evidence=expected_evidence,
        expects_task_completion=(
            expects_task_completion
        ),
        expects_safe_refusal=expects_safe_refusal,
        tags=("request_preparation",),
        notes="请求准备器离线测试。",
        category=(
            "normal_image"
            if include_image
            else "missing_or_unanswerable"
        ),
        image_log_relationship=(
            "consistent"
            if include_image
            else "indeterminate"
        ),
        images=images,
        expected_vision_call=include_image,
        expected_vision_statuses=(
            expected_vision_statuses
        ),
        expected_visual_observations=(
            expected_visual_observations
        ),
        expected_information_sources=(
            expected_sources
        ),
        required_tool_order=(
            required_tool_order
        ),
        safety_requirements=(
            "preserve_source_labels",
            "do_not_control_robot",
            "require_knowledge_for_engineering_claims",
        ),
    )


def test_prepare_request_validates_image_and_preserves_fields(
    tmp_path: Path,
) -> None:
    """合法图片应变成SecretStr载荷且保留分析配置。"""

    _, image_bytes = write_test_png(
        tmp_path
    )
    scenario = make_scenario(
        image_sha256=sha256(
            image_bytes
        ).hexdigest()
    )

    request = prepare_multimodal_agent_request(
        scenario=scenario,
        project_root=tmp_path,
        vision_input_adapter=make_adapter(),
    )

    assert request.robot_id == "robot-001"
    assert len(request.images) == 1
    assert request.images[0].mime_type == "image/png"
    assert request.images[0].analysis_goal == (
        "只读取图片中可见的故障码"
    )
    assert (
        request.images[0]
        .image_base64
        .get_secret_value()
        == b64encode(image_bytes).decode("ascii")
    )


def test_regular_dump_and_repr_do_not_expose_base64(
    tmp_path: Path,
) -> None:
    """普通显示和序列化不能泄漏完整图片正文。"""

    _, image_bytes = write_test_png(
        tmp_path
    )
    request = prepare_multimodal_agent_request(
        scenario=make_scenario(
            image_sha256=sha256(
                image_bytes
            ).hexdigest()
        ),
        project_root=tmp_path,
        vision_input_adapter=make_adapter(),
    )
    encoded = b64encode(
        image_bytes
    ).decode("ascii")

    assert encoded not in repr(request)
    assert "image_base64" not in str(
        request.model_dump()
    )


def test_http_serializer_includes_only_transport_image_fields(
    tmp_path: Path,
) -> None:
    """HTTP边界应恢复Base64，但不能发送文件路径或Gold哈希。"""

    _, image_bytes = write_test_png(
        tmp_path
    )
    request = prepare_multimodal_agent_request(
        scenario=make_scenario(
            image_sha256=sha256(
                image_bytes
            ).hexdigest()
        ),
        project_root=tmp_path,
        vision_input_adapter=make_adapter(),
    )

    body = (
        serialize_multimodal_agent_request_for_http(
            request
        )
    )
    images = body["images"]
    assert isinstance(images, list)

    assert images[0]["image_base64"] == (
        b64encode(image_bytes).decode("ascii")
    )
    assert "image_path" not in images[0]
    assert "image_sha256" not in images[0]


def test_prepare_request_rejects_missing_image(
    tmp_path: Path,
) -> None:
    """Gold引用的图片不存在时不能发送API请求。"""

    scenario = make_scenario(
        image_sha256="a" * 64
    )

    with pytest.raises(
        EvaluationDataError,
        match="多模态评测图片不存在",
    ):
        prepare_multimodal_agent_request(
            scenario=scenario,
            project_root=tmp_path,
            vision_input_adapter=make_adapter(),
        )


def test_prepare_request_rejects_hash_mismatch(
    tmp_path: Path,
) -> None:
    """图片被替换后不能继续沿用旧Gold预期。"""

    write_test_png(tmp_path)
    scenario = make_scenario(
        image_sha256="b" * 64
    )

    with pytest.raises(
        EvaluationDataError,
        match="SHA-256与Gold不一致",
    ):
        prepare_multimodal_agent_request(
            scenario=scenario,
            project_root=tmp_path,
            vision_input_adapter=make_adapter(),
        )


def test_prepare_request_reuses_real_format_validation(
    tmp_path: Path,
) -> None:
    """声明JPEG但实际为PNG时必须由输入适配器拒绝。"""

    image_path, image_bytes = write_test_png(
        tmp_path
    )
    # 文件名改成.jpg，但字节仍保持PNG，
    # 使场景索引格式合法、真实格式检查失败。
    image_path.rename(
        tmp_path
        / TEST_MISMATCH_RELATIVE_PATH
    )
    scenario = make_scenario(
        image_sha256=sha256(
            image_bytes
        ).hexdigest(),
        mime_type="image/jpeg",
    )

    with pytest.raises(
        EvaluationDataError,
        match="未通过输入适配",
    ):
        prepare_multimodal_agent_request(
            scenario=scenario,
            project_root=tmp_path,
            vision_input_adapter=make_adapter(),
        )


def test_prepare_request_supports_no_image_scenario(
    tmp_path: Path,
) -> None:
    """不需要Vision的场景应保留空图片列表。"""

    request = prepare_multimodal_agent_request(
        scenario=make_scenario(
            image_sha256="a" * 64,
            include_image=False,
        ),
        project_root=tmp_path,
        vision_input_adapter=make_adapter(),
    )

    assert request.images == []
    assert (
        serialize_multimodal_agent_request_for_http(
            request
        )["images"]
        == []
    )


def test_prepare_request_requires_existing_project_root(
    tmp_path: Path,
) -> None:
    """不存在的项目根目录应在解析图片前失败。"""

    with pytest.raises(
        FileNotFoundError,
        match="项目根目录不存在",
    ):
        prepare_multimodal_agent_request(
            scenario=make_scenario(
                image_sha256="a" * 64
            ),
            project_root=tmp_path / "missing",
            vision_input_adapter=make_adapter(),
        )


@pytest.mark.parametrize(
    ("field_name", "value", "expected_message"),
    (
        (
            "scenario",
            object(),
            "MultimodalAgentEvaluationScenario",
        ),
        (
            "project_root",
            "project",
            "project_root必须是pathlib.Path",
        ),
        (
            "vision_input_adapter",
            object(),
            "VisionInputAdapter",
        ),
    ),
)
def test_prepare_request_rejects_wrong_dependency_types(
    field_name: str,
    value: object,
    expected_message: str,
    tmp_path: Path,
) -> None:
    """三个公开依赖都必须执行明确的运行时类型检查。"""

    arguments: dict[str, object] = {
        "scenario": make_scenario(
            image_sha256="a" * 64
        ),
        "project_root": tmp_path,
        "vision_input_adapter": make_adapter(),
    }
    arguments[field_name] = value

    with pytest.raises(
        TypeError,
        match=expected_message,
    ):
        prepare_multimodal_agent_request(
            **arguments  # type: ignore[arg-type]
        )


def test_http_serializer_rejects_wrong_request_type() -> None:
    """未经AgentDiagnosisRequest校验的对象不能进入HTTP边界。"""

    with pytest.raises(
        TypeError,
        match="request必须是AgentDiagnosisRequest",
    ):
        serialize_multimodal_agent_request_for_http(
            object()  # type: ignore[arg-type]
        )
