"""多模态工具选择路线执行器的离线单元测试。

本文件只测试MultimodalToolSelectionRouteExecutor的编排责任：

1. 三条路线是否被正确分派；
2. OCR和Vision输出是否被转换成统一评分项；
3. 技术异常是否被转换成结构化失败记录；
4. 图片摘要、路线白名单和不可信文字安全检查是否生效。

测试不运行真实Tesseract，也不请求真实Vision API。
上游能力通过可编程Fake Provider提供，避免把外部环境和模型随机性
混入执行器自身的单元测试。
"""

from collections.abc import (
    Callable,
)
from hashlib import (
    sha256,
)
from pathlib import (
    Path,
)

from PIL import (
    Image,
)
import pytest

from app.errors import (
    InvalidOcrResponseError,
    InvalidVisionResponseError,
    OcrConfigurationError,
    OcrExecutionError,
    OcrTimeoutError,
    VisionConfigurationError,
    VisionTimeoutError,
    VisionUpstreamError,
)
from app.schemas.multimodal_tool_selection import (
    MultimodalToolSelectionCase,
    ToolSelectionRoute,
)
from app.schemas.ocr import (
    OcrBoundingBox,
    OcrObservation,
    OcrTextLine,
)
from app.schemas.vision import (
    VisionInput,
    VisionObservation,
    VisionObservationItem,
    VisionVisibleIndicator,
)
from app.services.multimodal_tool_selection_executor import (
    MultimodalToolSelectionRouteExecutor,
)
from app.services.multimodal_tool_selection_scoring import (
    MultimodalToolSelectionScorer,
)
from app.services.ocr_rule_parser import (
    OcrRuleParser,
)
from app.services.vision_input import (
    VisionInputAdapter,
)


# 测试图片采用固定小尺寸，既能被Pillow完整验证，
# 又不会让单元测试依赖仓库中的真实样本图片。
TEST_IMAGE_WIDTH_PX = 16
TEST_IMAGE_HEIGHT_PX = 12

# Adapter上限明显高于测试图片，但低于应用绝对上限。
TEST_MAX_IMAGE_SIZE_BYTES = 1_000_000
TEST_MAX_IMAGE_DIMENSION_PX = 1_024
TEST_MAX_IMAGE_PIXELS = 1_000_000

# Fake结果中的版本和模型名只用于验证追踪字段能够通过契约。
TEST_OCR_ENGINE_VERSION = "test-tesseract-1"
TEST_VISION_PROMPT_VERSION = "test-vision-prompt-v1"
TEST_VISION_MODEL_NAME = "fake-vision-model"


OcrOutcome = (
    OcrObservation
    | BaseException
    | Callable[[VisionInput], OcrObservation]
)

VisionOutcome = (
    VisionObservation
    | BaseException
    | Callable[[VisionInput], VisionObservation]
)


class ScriptedOcrProvider:
    """按测试预设返回OCR结果或异常的Fake Provider。

    calls保存执行器实际传入的VisionInput，测试据此确认：
    执行器确实调用了OCR一次，而且传入的是完成图片校验后的内部对象。
    """

    def __init__(
        self,
        outcome: OcrOutcome,
    ) -> None:
        self._outcome = outcome
        self.calls: list[VisionInput] = []

    async def analyze_image(
        self,
        *,
        vision_input: VisionInput,
    ) -> OcrObservation:
        """记录调用，并执行测试预设行为。"""

        self.calls.append(
            vision_input
        )

        if isinstance(
            self._outcome,
            BaseException,
        ):
            raise self._outcome

        if callable(self._outcome):
            return self._outcome(
                vision_input
            )

        return self._outcome


class ScriptedVisionProvider:
    """按测试预设返回Vision结果或异常的Fake Provider。"""

    def __init__(
        self,
        outcome: VisionOutcome,
    ) -> None:
        self._outcome = outcome
        self.calls: list[VisionInput] = []

    async def analyze_image(
        self,
        *,
        vision_input: VisionInput,
    ) -> VisionObservation:
        """记录调用，并执行测试预设行为。"""

        self.calls.append(
            vision_input
        )

        if isinstance(
            self._outcome,
            BaseException,
        ):
            raise self._outcome

        if callable(self._outcome):
            return self._outcome(
                vision_input
            )

        return self._outcome


def make_test_image(
    project_root: Path,
    *,
    filename: str = "case.png",
) -> tuple[str, str]:
    """创建一张合法PNG并返回仓库相对路径和SHA-256。"""

    image_directory = (
        project_root
        / "data"
        / "multimodal"
        / "tool-selection"
    )
    image_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_path = (
        image_directory
        / filename
    )

    Image.new(
        mode="RGB",
        size=(
            TEST_IMAGE_WIDTH_PX,
            TEST_IMAGE_HEIGHT_PX,
        ),
        color="white",
    ).save(
        image_path,
        format="PNG",
    )

    image_bytes = image_path.read_bytes()

    return (
        image_path.relative_to(
            project_root
        ).as_posix(),
        sha256(
            image_bytes
        ).hexdigest(),
    )


def make_text_case(
    project_root: Path,
    *,
    image_sha256: str | None = None,
    contains_untrusted_text: bool = False,
) -> MultimodalToolSelectionCase:
    """创建首选OCR路线的清晰文字面板案例。"""

    image_path, actual_sha256 = (
        make_test_image(
            project_root
        )
    )

    return MultimodalToolSelectionCase(
        case_id="multimodal-001",
        name="清晰机器人状态文字面板",
        sample_kind="text_panel",
        image_path=image_path,
        image_sha256=(
            image_sha256
            or actual_sha256
        ),
        question="读取故障码、网络状态和任务状态",
        preferred_route="ocr_rule",
        acceptable_routes=(
            "ocr_rule",
        ),
        routes_to_compare=(
            "ocr_rule",
            "vision_model",
            "direct_abstention",
        ),
        expected_text_terms=(
            "ERR-NET-4001",
            "CONNECTED",
            "PAUSED",
        ),
        expected_visual_observations=(),
        expects_abstention=False,
        contains_untrusted_text=(
            contains_untrusted_text
        ),
        rationale=(
            "固定标签和字符适合本地OCR加确定性规则"
        ),
        tags=(
            "text-panel",
        ),
    )


def make_visual_case(
    project_root: Path,
    *,
    contains_untrusted_text: bool = False,
) -> MultimodalToolSelectionCase:
    """创建首选Vision路线的指示灯视觉案例。"""

    image_path, image_sha256 = (
        make_test_image(
            project_root,
            filename="visual.png",
        )
    )

    return MultimodalToolSelectionCase(
        case_id="multimodal-003",
        name="设备指示灯颜色状态",
        sample_kind="visual_state",
        image_path=image_path,
        image_sha256=image_sha256,
        question="观察NET和FAULT指示灯的颜色",
        preferred_route="vision_model",
        acceptable_routes=(
            "vision_model",
        ),
        routes_to_compare=(
            "vision_model",
            "ocr_rule",
            "direct_abstention",
        ),
        expected_text_terms=(),
        expected_visual_observations=(
            "NET指示灯呈绿色常亮",
            "FAULT指示灯呈红色常亮",
        ),
        expects_abstention=False,
        contains_untrusted_text=(
            contains_untrusted_text
        ),
        rationale=(
            "颜色和亮灭属于普通OCR不能可靠表达的视觉状态"
        ),
        tags=(
            "visual-state",
        ),
    )


def make_degraded_case(
    project_root: Path,
) -> MultimodalToolSelectionCase:
    """创建标准答案要求直接拒答的退化图片案例。"""

    image_path, image_sha256 = (
        make_test_image(
            project_root,
            filename="degraded.png",
        )
    )

    return MultimodalToolSelectionCase(
        case_id="multimodal-005",
        name="严重模糊图片",
        sample_kind="degraded_image",
        image_path=image_path,
        image_sha256=image_sha256,
        question="判断图片中的设备状态",
        preferred_route=(
            "direct_abstention"
        ),
        acceptable_routes=(
            "direct_abstention",
        ),
        routes_to_compare=(
            "direct_abstention",
            "ocr_rule",
            "vision_model",
        ),
        expected_text_terms=(),
        expected_visual_observations=(),
        expects_abstention=True,
        rationale=(
            "图片质量不足时不应猜测设备状态"
        ),
        tags=(
            "degraded-image",
        ),
    )


def make_ocr_observation(
    vision_input: VisionInput,
    *,
    status: str = "completed",
    text: str = (
        "FAULT CODE: ERR-NET-4001\n"
        "NETWORK: CONNECTED\n"
        "TASK: PAUSED"
    ),
) -> OcrObservation:
    """创建与输入图片摘要绑定的OCR观察。"""

    if status == "empty":
        recognized_text = ""
        lines: tuple[OcrTextLine, ...] = ()
        mean_confidence = None
        requires_human_check = True
        uncertain_items = (
            "没有识别到可用文字",
        )
    else:
        text_lines = text.splitlines()
        recognized_text = "\n".join(
            text_lines
        )
        confidence = (
            40.0
            if status == "low_confidence"
            else 95.0
        )
        lines = tuple(
            OcrTextLine(
                line_index=line_index,
                text=line_text,
                confidence=confidence,
                token_count=max(
                    1,
                    len(
                        line_text.split()
                    ),
                ),
                bounding_box=(
                    OcrBoundingBox(
                        left_px=0,
                        top_px=line_index,
                        width_px=10,
                        height_px=1,
                    )
                ),
            )
            for line_index, line_text
            in enumerate(text_lines)
        )
        mean_confidence = confidence
        requires_human_check = (
            status == "low_confidence"
        )
        uncertain_items = (
            (
                "OCR平均置信度低于门槛",
            )
            if requires_human_check
            else ()
        )

    return OcrObservation(
        status=status,
        lines=lines,
        mean_confidence=mean_confidence,
        minimum_confidence=80.0,
        language="eng",
        source_image_sha256=(
            vision_input
            .metadata
            .sha256_hex
        ),
        engine_version=(
            TEST_OCR_ENGINE_VERSION
        ),
        duration_ms=1.0,
        text_character_count=len(
            recognized_text
        ),
        recognized_text_sha256=(
            sha256(
                recognized_text.encode(
                    "utf-8"
                )
            ).hexdigest()
        ),
        requires_human_check=(
            requires_human_check
        ),
        uncertain_items=uncertain_items,
    )


def make_indicator_vision_observation(
    vision_input: VisionInput,
    *,
    untrusted_text_detected: bool = False,
) -> VisionObservation:
    """创建完整的指示灯视觉观察。"""

    if untrusted_text_detected:
        untrusted_text_notes = (
            "图片包含不应执行的指令式文字",
        )
        requires_human_check = True
        human_check_reasons = (
            "需要人工确认不可信图片文字",
        )
    else:
        untrusted_text_notes = ()
        requires_human_check = False
        human_check_reasons = ()

    return VisionObservation(
        status="completed",
        image_quality="clear",
        observations=(),
        visible_indicators=(
            VisionVisibleIndicator(
                label="NET",
                observed_state="绿色常亮",
                confidence="high",
                region="面板左侧",
            ),
            VisionVisibleIndicator(
                label="FAULT指示灯",
                observed_state="红色常亮",
                confidence="high",
                region="面板右侧",
            ),
        ),
        uncertain_items=(),
        untrusted_text_detected=(
            untrusted_text_detected
        ),
        untrusted_text_notes=(
            untrusted_text_notes
        ),
        requires_human_check=(
            requires_human_check
        ),
        human_check_reasons=(
            human_check_reasons
        ),
        source_image_sha256=(
            vision_input
            .metadata
            .sha256_hex
        ),
        prompt_version=(
            TEST_VISION_PROMPT_VERSION
        ),
        model_name=TEST_VISION_MODEL_NAME,
    )


def make_unusable_vision_observation(
    vision_input: VisionInput,
) -> VisionObservation:
    """创建因图片质量不足而主动拒答的Vision观察。"""

    return VisionObservation(
        status="unusable",
        image_quality="unusable",
        observations=(),
        visible_indicators=(),
        uncertain_items=(
            "图片严重模糊，无法确认设备状态",
        ),
        requires_human_check=True,
        human_check_reasons=(
            "需要补充清晰图片",
        ),
        source_image_sha256=(
            vision_input
            .metadata
            .sha256_hex
        ),
        prompt_version=(
            TEST_VISION_PROMPT_VERSION
        ),
        model_name=TEST_VISION_MODEL_NAME,
    )


def make_partial_vision_observation(
    vision_input: VisionInput,
) -> VisionObservation:
    """创建有一项可用观察但仍需人工复核的Vision结果。"""

    return VisionObservation(
        status="partial",
        image_quality="limited",
        observations=(
            VisionObservationItem(
                description=(
                    "插头与J3接口之间存在可见间隙"
                ),
                category="visible_condition",
                confidence="medium",
                region="接口区域",
            ),
        ),
        visible_indicators=(),
        uncertain_items=(
            "无法确认锁紧环是否完全就位",
        ),
        requires_human_check=True,
        human_check_reasons=(
            "连接器局部被遮挡",
        ),
        source_image_sha256=(
            vision_input
            .metadata
            .sha256_hex
        ),
        prompt_version=(
            TEST_VISION_PROMPT_VERSION
        ),
        model_name=TEST_VISION_MODEL_NAME,
    )


def make_adapter() -> VisionInputAdapter:
    """创建每个测试共享配置但不共享状态的输入适配器。"""

    return VisionInputAdapter(
        max_image_size_bytes=(
            TEST_MAX_IMAGE_SIZE_BYTES
        ),
        max_image_dimension_px=(
            TEST_MAX_IMAGE_DIMENSION_PX
        ),
        max_image_pixels=(
            TEST_MAX_IMAGE_PIXELS
        ),
    )


def make_executor(
    *,
    project_root: Path,
    ocr_outcome: OcrOutcome,
    vision_outcome: VisionOutcome,
) -> tuple[
    MultimodalToolSelectionRouteExecutor,
    ScriptedOcrProvider,
    ScriptedVisionProvider,
]:
    """组装执行器以及可供断言调用次数的两个Fake Provider。"""

    ocr_provider = ScriptedOcrProvider(
        ocr_outcome
    )
    vision_provider = (
        ScriptedVisionProvider(
            vision_outcome
        )
    )

    executor = (
        MultimodalToolSelectionRouteExecutor(
            project_root=project_root,
            input_adapter=make_adapter(),
            ocr_provider=ocr_provider,
            vision_provider=(
                vision_provider
            ),
            rule_parser=OcrRuleParser(),
            scorer=(
                MultimodalToolSelectionScorer()
            ),
        )
    )

    return (
        executor,
        ocr_provider,
        vision_provider,
    )


def unused_ocr_outcome(
    vision_input: VisionInput,
) -> OcrObservation:
    """为不应调用OCR的测试提供合法但不会被执行的工厂。"""

    return make_ocr_observation(
        vision_input
    )


def unused_vision_outcome(
    vision_input: VisionInput,
) -> VisionObservation:
    """为不应调用Vision的测试提供合法但不会被执行的工厂。"""

    return make_indicator_vision_observation(
        vision_input
    )


def test_constructor_rejects_provider_without_analyze_image(
    tmp_path: Path,
) -> None:
    """构造器应在启动阶段拒绝没有Provider方法的对象。

    预期流程：构造器检查ocr_provider.analyze_image是否可调用。
    预期结果：立即抛出TypeError，不等到路线执行时才失败。
    """

    with pytest.raises(
        TypeError,
        match="ocr_provider",
    ):
        MultimodalToolSelectionRouteExecutor(
            project_root=tmp_path,
            input_adapter=make_adapter(),
            ocr_provider=object(),
            vision_provider=(
                ScriptedVisionProvider(
                    unused_vision_outcome
                )
            ),
            rule_parser=OcrRuleParser(),
            scorer=(
                MultimodalToolSelectionScorer()
            ),
        )


@pytest.mark.asyncio
async def test_direct_abstention_uses_no_provider_and_passes_degraded_case(
    tmp_path: Path,
) -> None:
    """直接拒答路线应只校验图片，不调用OCR或Vision。

    预期流程：图片适配与摘要校验 → 固定拒答 → 评分。
    预期结果：退化案例通过，外部模型调用数为0，两个Provider均未调用。
    """

    case = make_degraded_case(
        tmp_path
    )
    (
        executor,
        ocr_provider,
        vision_provider,
    ) = make_executor(
        project_root=tmp_path,
        ocr_outcome=unused_ocr_outcome,
        vision_outcome=(
            unused_vision_outcome
        ),
    )

    result = await executor.execute(
        case=case,
        route="direct_abstention",
    )

    assert result.execution_status == (
        "abstained"
    )
    assert result.outcome_correct is True
    assert result.route_appropriate is True
    assert result.passed is True
    assert result.external_model_call_count == 0
    assert result.cost_status == "not_applicable"
    assert result.estimated_cost_usd == 0.0
    assert result.latency_ms >= 0.0
    assert ocr_provider.calls == []
    assert vision_provider.calls == []


@pytest.mark.asyncio
async def test_direct_abstention_does_not_pass_answerable_case(
    tmp_path: Path,
) -> None:
    """安全拒答不能冒充清晰可回答图片的正确路线。

    预期流程：执行器仍固定拒答，评分器再结合Gold判断。
    预期结果：状态安全但内容结果不正确，最终passed=False。
    """

    case = make_text_case(tmp_path)
    executor, _, _ = make_executor(
        project_root=tmp_path,
        ocr_outcome=unused_ocr_outcome,
        vision_outcome=(
            unused_vision_outcome
        ),
    )

    result = await executor.execute(
        case=case,
        route="direct_abstention",
    )

    assert result.abstained is True
    assert result.outcome_correct is False
    assert result.route_appropriate is False
    assert result.passed is False


@pytest.mark.asyncio
async def test_ocr_route_parses_complete_status_panel(
    tmp_path: Path,
) -> None:
    """OCR路线应把三项状态字段转换成统一评分项。

    预期流程：输入适配 → Fake OCR → OcrRuleParser → 评分器。
    预期结果：三项Gold全部匹配，OCR只调用一次，Vision不调用。
    """

    case = make_text_case(tmp_path)
    (
        executor,
        ocr_provider,
        vision_provider,
    ) = make_executor(
        project_root=tmp_path,
        ocr_outcome=make_ocr_observation,
        vision_outcome=(
            unused_vision_outcome
        ),
    )

    result = await executor.execute(
        case=case,
        route="ocr_rule",
    )

    assert result.execution_status == (
        "completed"
    )
    assert result.actual_items == (
        "ERR-NET-4001",
        "CONNECTED",
        "PAUSED",
    )
    assert result.content_accuracy == 1.0
    assert result.passed is True
    assert result.requires_human_check is False
    assert result.external_model_call_count == 0
    assert len(ocr_provider.calls) == 1
    assert vision_provider.calls == []
    assert (
        ocr_provider.calls[0]
        .metadata
        .sha256_hex
        == case.image_sha256
    )


@pytest.mark.asyncio
async def test_low_confidence_ocr_becomes_partial_human_check(
    tmp_path: Path,
) -> None:
    """低置信度OCR即使提取完整字段也只能形成部分结果。

    预期流程：OCR返回low_confidence，规则解析器保留字段并降级为partial。
    预期结果：三项内容仍匹配，但必须人工复核且带脱敏说明。
    """

    case = make_text_case(tmp_path)

    def low_confidence_outcome(
        vision_input: VisionInput,
    ) -> OcrObservation:
        return make_ocr_observation(
            vision_input,
            status="low_confidence",
        )

    executor, _, _ = make_executor(
        project_root=tmp_path,
        ocr_outcome=(
            low_confidence_outcome
        ),
        vision_outcome=(
            unused_vision_outcome
        ),
    )

    result = await executor.execute(
        case=case,
        route="ocr_rule",
    )

    assert result.execution_status == "partial"
    assert result.content_accuracy == 1.0
    assert result.requires_human_check is True
    assert result.public_message is not None
    assert result.passed is True


@pytest.mark.asyncio
async def test_empty_ocr_becomes_active_abstention(
    tmp_path: Path,
) -> None:
    """OCR空结果应成为主动拒答，而不是技术失败或伪造字段。

    预期流程：OCR返回empty → 规则解析返回empty → 执行器清空观察项。
    预期结果：execution_status=abstained，failure_kind保持None。
    """

    case = make_text_case(tmp_path)

    def empty_outcome(
        vision_input: VisionInput,
    ) -> OcrObservation:
        return make_ocr_observation(
            vision_input,
            status="empty",
        )

    executor, _, _ = make_executor(
        project_root=tmp_path,
        ocr_outcome=empty_outcome,
        vision_outcome=(
            unused_vision_outcome
        ),
    )

    result = await executor.execute(
        case=case,
        route="ocr_rule",
    )

    assert result.execution_status == (
        "abstained"
    )
    assert result.actual_items == ()
    assert result.failure_kind is None
    assert result.requires_human_check is True
    assert result.passed is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "ocr_error",
        "expected_failure_kind",
        "expected_message",
    ),
    (
        (
            OcrTimeoutError("private timeout"),
            "timeout",
            "本地OCR处理超时",
        ),
        (
            InvalidOcrResponseError(
                "private invalid data"
            ),
            "invalid_response",
            "OCR结果不符合内部数据契约",
        ),
        (
            OcrConfigurationError(
                "private configuration"
            ),
            "unexpected_error",
            "本地OCR暂时不可用",
        ),
        (
            OcrExecutionError(
                "private execution"
            ),
            "unexpected_error",
            "本地OCR暂时不可用",
        ),
        (
            RuntimeError("private bug detail"),
            "unexpected_error",
            "本地OCR路线执行失败",
        ),
    ),
)
async def test_ocr_errors_become_sanitized_failure_records(
    tmp_path: Path,
    ocr_error: BaseException,
    expected_failure_kind: str,
    expected_message: str,
) -> None:
    """不同OCR异常应分类成稳定、脱敏、可汇总的失败结果。

    预期流程：Fake OCR抛异常 → 执行器捕获并调用_score_failure()。
    预期结果：failed结果不含实际观察，不泄露异常原文，也不记录模型成本。
    """

    case = make_text_case(tmp_path)
    executor, ocr_provider, _ = make_executor(
        project_root=tmp_path,
        ocr_outcome=ocr_error,
        vision_outcome=(
            unused_vision_outcome
        ),
    )

    result = await executor.execute(
        case=case,
        route="ocr_rule",
    )

    assert result.execution_status == "failed"
    assert result.failure_kind == (
        expected_failure_kind
    )
    assert result.public_message == (
        expected_message
    )
    assert "private" not in result.public_message
    assert result.actual_items == ()
    assert result.external_model_call_count == 0
    assert result.cost_status == "not_applicable"
    assert result.estimated_cost_usd == 0.0
    assert len(ocr_provider.calls) == 1


@pytest.mark.asyncio
async def test_vision_route_normalizes_indicator_labels_and_scores_result(
    tmp_path: Path,
) -> None:
    """Vision路线应把指示器结构转换成统一中文观察项。

    预期流程：Fake Vision → 标签规范化 → 去重 → 评分器。
    预期结果：NET自动补“指示灯”，两项Gold全部匹配并记录一次模型调用。
    """

    case = make_visual_case(tmp_path)
    (
        executor,
        ocr_provider,
        vision_provider,
    ) = make_executor(
        project_root=tmp_path,
        ocr_outcome=unused_ocr_outcome,
        vision_outcome=(
            make_indicator_vision_observation
        ),
    )

    result = await executor.execute(
        case=case,
        route="vision_model",
    )

    assert result.execution_status == (
        "completed"
    )
    assert result.actual_items == (
        "NET指示灯呈绿色常亮",
        "FAULT指示灯呈红色常亮",
    )
    assert result.content_accuracy == 1.0
    assert result.passed is True
    assert result.external_model_call_count == 1
    assert result.cost_status == "unavailable"
    assert result.estimated_cost_usd is None
    assert result.cost_note is not None
    assert ocr_provider.calls == []
    assert len(vision_provider.calls) == 1


@pytest.mark.asyncio
async def test_partial_vision_preserves_observation_and_requests_review(
    tmp_path: Path,
) -> None:
    """Vision部分结果应保留已确认项，同时明确要求人工复核。

    预期流程：Provider返回partial → 执行器保留observations并合并不确定说明。
    预期结果：不是拒答，实际项存在，requires_human_check=True。
    """

    image_path, image_sha256 = (
        make_test_image(
            tmp_path,
            filename="connector.png",
        )
    )
    case = MultimodalToolSelectionCase(
        case_id="multimodal-004",
        name="连接器可见间隙",
        sample_kind="visual_state",
        image_path=image_path,
        image_sha256=image_sha256,
        question="检查插头与J3接口连接状态",
        preferred_route="vision_model",
        acceptable_routes=(
            "vision_model",
        ),
        routes_to_compare=(
            "vision_model",
            "direct_abstention",
        ),
        expected_visual_observations=(
            "插头与J3接口之间存在可见间隙",
        ),
        expects_abstention=False,
        rationale="部件位置关系需要视觉观察",
    )
    executor, _, _ = make_executor(
        project_root=tmp_path,
        ocr_outcome=unused_ocr_outcome,
        vision_outcome=(
            make_partial_vision_observation
        ),
    )

    result = await executor.execute(
        case=case,
        route="vision_model",
    )

    assert result.execution_status == "partial"
    assert result.abstained is False
    assert result.requires_human_check is True
    assert result.actual_items == (
        "插头与J3接口之间存在可见间隙",
    )
    assert result.public_message == (
        "无法确认锁紧环是否完全就位"
    )
    assert result.passed is True


@pytest.mark.asyncio
async def test_unusable_vision_is_abstention_but_not_appropriate_route(
    tmp_path: Path,
) -> None:
    """Vision识别到图片不可用时应拒答，但不能冒充首选拒答路线。

    预期流程：Vision返回unusable → 执行器清空观察项 → 评分器。
    预期结果：拒答行为正确，但路线不在acceptable_routes中，所以不通过。
    """

    case = make_degraded_case(tmp_path)
    executor, _, _ = make_executor(
        project_root=tmp_path,
        ocr_outcome=unused_ocr_outcome,
        vision_outcome=(
            make_unusable_vision_observation
        ),
    )

    result = await executor.execute(
        case=case,
        route="vision_model",
    )

    assert result.execution_status == (
        "abstained"
    )
    assert result.outcome_correct is True
    assert result.route_appropriate is False
    assert result.passed is False
    assert result.actual_items == ()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "vision_error",
        "expected_failure_kind",
        "expected_message",
    ),
    (
        (
            VisionTimeoutError("private timeout"),
            "timeout",
            "Vision模型响应超时",
        ),
        (
            VisionUpstreamError(
                "private upstream"
            ),
            "upstream_error",
            "Vision上游服务暂时不可用",
        ),
        (
            InvalidVisionResponseError(
                "private invalid response"
            ),
            "invalid_response",
            "Vision结果不符合内部数据契约",
        ),
        (
            VisionConfigurationError(
                "private configuration"
            ),
            "unexpected_error",
            "Vision配置不可用",
        ),
        (
            RuntimeError("private bug detail"),
            "unexpected_error",
            "Vision路线执行失败",
        ),
    ),
)
async def test_vision_errors_become_sanitized_failure_records(
    tmp_path: Path,
    vision_error: BaseException,
    expected_failure_kind: str,
    expected_message: str,
) -> None:
    """Vision异常应转换成脱敏失败结果并记录一次外部调用。

    预期流程：Fake Vision抛异常 → 执行器分类 → _score_failure()。
    预期结果：失败类型稳定，私有异常正文不进入报告，成本诚实标为无法估算。
    """

    case = make_visual_case(tmp_path)
    executor, _, vision_provider = (
        make_executor(
            project_root=tmp_path,
            ocr_outcome=(
                unused_ocr_outcome
            ),
            vision_outcome=vision_error,
        )
    )

    result = await executor.execute(
        case=case,
        route="vision_model",
    )

    assert result.execution_status == "failed"
    assert result.failure_kind == (
        expected_failure_kind
    )
    assert result.public_message == (
        expected_message
    )
    assert "private" not in result.public_message
    assert result.external_model_call_count == 1
    assert result.cost_status == "unavailable"
    assert result.estimated_cost_usd is None
    assert result.cost_note is not None
    assert len(vision_provider.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "provider_detects_untrusted_text",
        "expected_safety_passed",
        "expected_passed",
    ),
    (
        (True, True, True),
        (False, False, False),
    ),
)
async def test_vision_untrusted_text_safety_requires_detection_and_review(
    tmp_path: Path,
    provider_detects_untrusted_text: bool,
    expected_safety_passed: bool,
    expected_passed: bool,
) -> None:
    """包含不可信文字的图片必须被检测并进入人工复核路径。

    预期流程：案例安全标签与Vision观察中的检测/复核字段共同计算安全结果。
    预期结果：只有“已检测且要求人工检查”时，safety_passed和passed才为True。
    """

    case = make_visual_case(
        tmp_path,
        contains_untrusted_text=True,
    )

    def vision_outcome(
        vision_input: VisionInput,
    ) -> VisionObservation:
        return make_indicator_vision_observation(
            vision_input,
            untrusted_text_detected=(
                provider_detects_untrusted_text
            ),
        )

    executor, _, _ = make_executor(
        project_root=tmp_path,
        ocr_outcome=unused_ocr_outcome,
        vision_outcome=vision_outcome,
    )

    result = await executor.execute(
        case=case,
        route="vision_model",
    )

    assert result.safety_passed is (
        expected_safety_passed
    )
    assert result.passed is expected_passed


@pytest.mark.asyncio
async def test_image_sha_mismatch_stops_before_any_route_provider(
    tmp_path: Path,
) -> None:
    """Gold图片摘要与真实文件不一致时应停止整个实验路线。

    预期流程：Adapter计算真实SHA → 执行器与Gold摘要比较。
    预期结果：抛出ValueError，OCR和Vision均不调用，避免比较不同图片。
    """

    case = make_text_case(
        tmp_path,
        image_sha256=(
            "0" * 64
        ),
    )
    (
        executor,
        ocr_provider,
        vision_provider,
    ) = make_executor(
        project_root=tmp_path,
        ocr_outcome=unused_ocr_outcome,
        vision_outcome=(
            unused_vision_outcome
        ),
    )

    with pytest.raises(
        ValueError,
        match="SHA-256",
    ):
        await executor.execute(
            case=case,
            route="ocr_rule",
        )

    assert ocr_provider.calls == []
    assert vision_provider.calls == []


@pytest.mark.asyncio
async def test_route_must_be_declared_by_case(
    tmp_path: Path,
) -> None:
    """执行器不能运行Gold案例未声明的实验路线。

    预期流程：execute()先检查routes_to_compare，再读取图片。
    预期结果：非法路线抛出ValueError，不产生不可审计的额外实验记录。
    """

    image_path, image_sha256 = (
        make_test_image(tmp_path)
    )
    case = MultimodalToolSelectionCase(
        case_id="multimodal-001",
        name="只比较两条路线",
        sample_kind="text_panel",
        image_path=image_path,
        image_sha256=image_sha256,
        question="读取状态",
        preferred_route="ocr_rule",
        acceptable_routes=(
            "ocr_rule",
        ),
        routes_to_compare=(
            "ocr_rule",
            "vision_model",
        ),
        expected_text_terms=(
            "ERR-NET-4001",
        ),
        expects_abstention=False,
        rationale="测试路线白名单",
    )
    executor, _, _ = make_executor(
        project_root=tmp_path,
        ocr_outcome=unused_ocr_outcome,
        vision_outcome=(
            unused_vision_outcome
        ),
    )

    with pytest.raises(
        ValueError,
        match="routes_to_compare",
    ):
        await executor.execute(
            case=case,
            route=(
                "direct_abstention"
            ),
        )


@pytest.mark.asyncio
async def test_provider_result_sha_mismatch_is_not_scored(
    tmp_path: Path,
) -> None:
    """Provider返回其他图片的摘要时，评分器应拒绝结果。

    预期流程：Vision调用成功 → 结果适配 → 评分器比较Provider摘要与Gold。
    预期结果：抛出ValueError，防止把另一张图片的观察算到当前案例。
    """

    case = make_visual_case(tmp_path)

    def wrong_hash_observation(
        vision_input: VisionInput,
    ) -> VisionObservation:
        valid_observation = (
            make_indicator_vision_observation(
                vision_input
            )
        )

        return valid_observation.model_copy(
            update={
                "source_image_sha256": (
                    "f" * 64
                ),
            },
        )

    executor, _, vision_provider = (
        make_executor(
            project_root=tmp_path,
            ocr_outcome=(
                unused_ocr_outcome
            ),
            vision_outcome=(
                wrong_hash_observation
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match="Gold案例图片摘要",
    ):
        await executor.execute(
            case=case,
            route="vision_model",
        )

    assert len(vision_provider.calls) == 1
