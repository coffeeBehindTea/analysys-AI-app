"""六张脱敏多模态工具选择样本生成器的测试。"""

from __future__ import annotations

from collections.abc import Iterator
from hashlib import sha256
from pathlib import Path

from PIL import Image
import pytest

from app.schemas.multimodal_tool_selection import (
    MultimodalToolSelectionCase,
)
import scripts.generate_multimodal_tool_selection_samples as generator


# 生成器必须固定产生六个案例。
EXPECTED_CASE_IDS = (
    "multimodal-001",
    "multimodal-002",
    "multimodal-003",
    "multimodal-004",
    "multimodal-005",
    "multimodal-006",
)


@pytest.fixture
def isolated_output_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Path]:
    """把生成器输出重定向到pytest临时目录。"""

    image_directory = (
        tmp_path
        / "data"
        / "multimodal"
        / "tool-selection"
    )
    manifest_path = (
        tmp_path
        / "data"
        / "eval"
        / "multimodal_tool_selection_cases.jsonl"
    )

    monkeypatch.setattr(
        generator,
        "PROJECT_ROOT",
        tmp_path,
    )
    monkeypatch.setattr(
        generator,
        "IMAGE_OUTPUT_DIRECTORY",
        image_directory,
    )
    monkeypatch.setattr(
        generator,
        "MANIFEST_OUTPUT_PATH",
        manifest_path,
    )

    image_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    yield tmp_path


def load_manifest(
    manifest_path: Path,
) -> tuple[
    MultimodalToolSelectionCase,
    ...,
]:
    """逐行读取并校验生成的JSONL清单。"""

    lines = tuple(
        line
        for line in manifest_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    )

    return tuple(
        MultimodalToolSelectionCase.model_validate_json(
            line
        )
        for line in lines
    )


def test_build_all_cases_returns_expected_case_set(
    isolated_output_directory: Path,
) -> None:
    """生成器应返回编号稳定的六个案例。"""

    cases = generator.build_all_cases()

    assert tuple(
        case.case_id
        for case in cases
    ) == EXPECTED_CASE_IDS


def test_build_all_cases_has_balanced_sample_kinds(
    isolated_output_directory: Path,
) -> None:
    """三种图片类别应各有两个案例。"""

    cases = generator.build_all_cases()

    assert sum(
        case.sample_kind == "text_panel"
        for case in cases
    ) == 2
    assert sum(
        case.sample_kind == "visual_state"
        for case in cases
    ) == 2
    assert sum(
        case.sample_kind == "degraded_image"
        for case in cases
    ) == 2


def test_every_case_compares_all_three_routes(
    isolated_output_directory: Path,
) -> None:
    """每个案例都应比较OCR、Vision和直接拒答。"""

    cases = generator.build_all_cases()

    assert all(
        case.routes_to_compare
        == generator.ALL_COMPARISON_ROUTES
        for case in cases
    )


def test_generated_images_are_valid_fixed_size_pngs(
    isolated_output_directory: Path,
) -> None:
    """六个文件都应是静态RGB PNG且尺寸相同。"""

    cases = generator.build_all_cases()

    for case in cases:
        image_path = (
            isolated_output_directory
            / case.image_path
        )

        assert image_path.is_file()

        with Image.open(image_path) as image:
            image.load()

            assert image.format == "PNG"
            assert image.mode == "RGB"
            assert image.size == (
                generator.IMAGE_WIDTH_PX,
                generator.IMAGE_HEIGHT_PX,
            )
            assert getattr(
                image,
                "is_animated",
                False,
            ) is False


def test_case_hash_matches_saved_png_bytes(
    isolated_output_directory: Path,
) -> None:
    """清单SHA-256必须与最终保存的PNG字节一致。"""

    cases = generator.build_all_cases()

    for case in cases:
        image_path = (
            isolated_output_directory
            / case.image_path
        )

        actual_sha256 = sha256(
            image_path.read_bytes()
        ).hexdigest()

        assert actual_sha256 == case.image_sha256


def test_generated_paths_and_hashes_are_unique(
    isolated_output_directory: Path,
) -> None:
    """六个案例不能误用同一路径或同一图片内容。"""

    cases = generator.build_all_cases()

    image_paths = tuple(
        case.image_path
        for case in cases
    )
    image_hashes = tuple(
        case.image_sha256
        for case in cases
    )

    assert len(image_paths) == len(
        set(image_paths)
    )
    assert len(image_hashes) == len(
        set(image_hashes)
    )


def test_text_cases_have_ocr_expectations(
    isolated_output_directory: Path,
) -> None:
    """文字样本应首选OCR并包含标准文字。"""

    cases = generator.build_all_cases()
    text_cases = tuple(
        case
        for case in cases
        if case.sample_kind == "text_panel"
    )

    assert all(
        case.preferred_route == "ocr_rule"
        for case in text_cases
    )
    assert all(
        case.expected_text_terms
        for case in text_cases
    )
    assert all(
        "vision_model"
        in case.acceptable_routes
        for case in text_cases
    )


def test_visual_cases_have_visual_expectations(
    isolated_output_directory: Path,
) -> None:
    """视觉样本应首选Vision并包含标准视觉观察。"""

    cases = generator.build_all_cases()
    visual_cases = tuple(
        case
        for case in cases
        if case.sample_kind == "visual_state"
    )

    assert all(
        case.preferred_route == "vision_model"
        for case in visual_cases
    )
    assert all(
        case.expected_visual_observations
        for case in visual_cases
    )
    assert all(
        case.acceptable_routes
        == ("vision_model",)
        for case in visual_cases
    )


def test_degraded_cases_only_accept_abstention(
    isolated_output_directory: Path,
) -> None:
    """模糊和遮挡样本只能把拒答判为正确。"""

    cases = generator.build_all_cases()
    degraded_cases = tuple(
        case
        for case in cases
        if case.sample_kind == "degraded_image"
    )

    assert all(
        case.expects_abstention
        for case in degraded_cases
    )
    assert all(
        case.preferred_route
        == "direct_abstention"
        for case in degraded_cases
    )
    assert all(
        case.acceptable_routes
        == ("direct_abstention",)
        for case in degraded_cases
    )


def test_indicator_sample_contains_expected_pixel_colors(
    isolated_output_directory: Path,
) -> None:
    """指示灯中心像素应分别为绿、红、蓝。"""

    generator.build_all_cases()

    image_path = (
        isolated_output_directory
        / "data"
        / "multimodal"
        / "tool-selection"
        / "multimodal-003.png"
    )

    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")

        try:
            assert rgb_image.getpixel(
                (300, 260)
            ) == (34, 197, 94)
            assert rgb_image.getpixel(
                (600, 260)
            ) == (239, 68, 68)
            assert rgb_image.getpixel(
                (900, 260)
            ) == (59, 130, 246)
        finally:
            rgb_image.close()


def test_occluded_sample_has_opaque_cover_over_values(
    isolated_output_directory: Path,
) -> None:
    """遮挡案例的值区域中心必须是固定不透明深色。"""

    generator.build_all_cases()

    image_path = (
        isolated_output_directory
        / "data"
        / "multimodal"
        / "tool-selection"
        / "multimodal-006.png"
    )

    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")

        try:
            assert rgb_image.getpixel(
                (750, 250)
            ) == (3, 7, 18)
        finally:
            rgb_image.close()


def test_write_manifest_round_trips_all_cases(
    isolated_output_directory: Path,
) -> None:
    """JSONL中的六行都应通过Schema并还原原案例。"""

    cases = generator.build_all_cases()
    generator.write_manifest(cases)

    restored_cases = load_manifest(
        generator.MANIFEST_OUTPUT_PATH
    )

    assert restored_cases == cases


def test_manifest_contains_no_generation_time_field(
    isolated_output_directory: Path,
) -> None:
    """实验清单不得加入生成时间等不稳定字段。"""

    cases = generator.build_all_cases()
    generator.write_manifest(cases)

    manifest_text = (
        generator.MANIFEST_OUTPUT_PATH
        .read_text(encoding="utf-8")
    )

    assert "generated_at" not in manifest_text
    assert "generation_time" not in manifest_text
    assert "timestamp" not in manifest_text


def test_repeated_generation_is_byte_deterministic(
    isolated_output_directory: Path,
) -> None:
    """相同环境重复生成时，图片字节和摘要不应变化。"""

    first_cases = generator.build_all_cases()

    first_bytes = {
        case.case_id: (
            isolated_output_directory
            / case.image_path
        ).read_bytes()
        for case in first_cases
    }

    second_cases = generator.build_all_cases()

    second_bytes = {
        case.case_id: (
            isolated_output_directory
            / case.image_path
        ).read_bytes()
        for case in second_cases
    }

    assert second_cases == first_cases
    assert second_bytes == first_bytes


def test_main_creates_outputs_and_prints_summary(
    isolated_output_directory: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """命令行入口应创建目录、图片、清单和稳定摘要。"""

    generator.main()

    output = capsys.readouterr().out

    assert "多模态工具选择样本生成完成" in output
    assert "案例数：6" in output
    assert "文字面板数：2" in output
    assert "视觉状态数：2" in output
    assert "退化图片数：2" in output

    assert generator.MANIFEST_OUTPUT_PATH.is_file()
    assert len(
        tuple(
            generator.IMAGE_OUTPUT_DIRECTORY.glob(
                "multimodal-*.png"
            )
        )
    ) == 6
