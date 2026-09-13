"""Week 6 多模态离线 Fixture 加载与配对服务测试。

被测试模块：
app.services.multimodal_offline_fixture_loader。

预期调用链：

Fixture JSONL
→ load_multimodal_offline_fixtures()（当前模块：逐行解析与校验）
→ MultimodalOfflineScenarioFixture
→ pair_multimodal_offline_regression_cases()（当前模块：按ID配对）
→ MultimodalOfflineRegressionCase
→ 后续批量离线执行服务。

本测试不读取图片，不运行 Agent，也不调用任何外部服务。
"""

import json
from pathlib import Path

import pytest

from app.schemas.multimodal_agent_evaluation import (
    MultimodalAgentEvaluationScenario,
)
from app.schemas.multimodal_offline_regression import (
    MultimodalOfflineScenarioFixture,
)
from app.services.evaluation import (
    EvaluationDataError,
    load_multimodal_agent_evaluation_scenarios,
)
from app.services.multimodal_offline_fixture_loader import (
    MultimodalOfflineRegressionCase,
    load_multimodal_offline_fixtures,
    pair_multimodal_offline_regression_cases,
)


# 真实 Gold 文件只用于取得已经通过完整场景契约校验的对象。
# 配对测试不会读取这些场景引用的图片，也不会运行 Agent。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENARIO_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "eval"
    / "multimodal_scenarios.jsonl"
)


def make_fixture(
    scenario_id: str,
) -> MultimodalOfflineScenarioFixture:
    """创建一条最小合法 Fixture。

    空脚本适合测试加载和配对；它是否足以驱动某条 Gold 场景，
    属于后续执行器和评分器的职责，不应由文件加载器猜测。
    """

    return MultimodalOfflineScenarioFixture(
        scenario_id=scenario_id,
    )


def write_fixture_jsonl(
    path: Path,
    fixtures: tuple[
        MultimodalOfflineScenarioFixture,
        ...,
    ],
) -> None:
    """把 Fixture 模型写成 UTF-8 JSONL 测试文件。"""

    path.write_text(
        "\n".join(
            json.dumps(
                fixture.model_dump(
                    mode="json"
                ),
                ensure_ascii=False,
            )
            for fixture in fixtures
        )
        + "\n",
        encoding="utf-8",
    )


def load_two_gold_scenarios() -> tuple[
    MultimodalAgentEvaluationScenario,
    MultimodalAgentEvaluationScenario,
]:
    """读取前两条真实 Gold 场景用于配对边界测试。"""

    scenarios = (
        load_multimodal_agent_evaluation_scenarios(
            SCENARIO_DATA_PATH
        )
    )

    return scenarios[0], scenarios[1]


def test_loader_returns_models_and_preserves_jsonl_order(
    tmp_path: Path,
) -> None:
    """合法 Fixture 应按 JSONL 原始顺序返回模型元组。

    测试方法：写入两条顺序确定的 Fixture，再调用加载器。
    预期结果：返回值是 tuple，元素是专用 Pydantic 模型，
    scenario_id 顺序和文件行序一致。
    """

    path = tmp_path / "fixtures.jsonl"
    write_fixture_jsonl(
        path,
        (
            make_fixture(
                "multimodal-agent-002"
            ),
            make_fixture(
                "multimodal-agent-001"
            ),
        ),
    )

    fixtures = (
        load_multimodal_offline_fixtures(
            path
        )
    )

    assert isinstance(fixtures, tuple)
    assert all(
        isinstance(
            fixture,
            MultimodalOfflineScenarioFixture,
        )
        for fixture in fixtures
    )
    assert tuple(
        fixture.scenario_id
        for fixture in fixtures
    ) == (
        "multimodal-agent-002",
        "multimodal-agent-001",
    )


def test_loader_requires_path_object() -> None:
    """普通字符串不能绕过统一 pathlib.Path 路径边界。"""

    with pytest.raises(
        TypeError,
        match="path必须是pathlib.Path",
    ):
        load_multimodal_offline_fixtures(
            "data/eval/fixtures.jsonl"  # type: ignore[arg-type]
        )


def test_loader_requires_jsonl_suffix(
    tmp_path: Path,
) -> None:
    """普通 JSON 文件不能冒充逐行 Fixture 数据集。"""

    path = tmp_path / "fixtures.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(
        EvaluationDataError,
        match="必须使用 .jsonl 扩展名",
    ):
        load_multimodal_offline_fixtures(
            path
        )


def test_loader_requires_existing_file(
    tmp_path: Path,
) -> None:
    """不存在的 Fixture 文件应保留具体路径错误。"""

    with pytest.raises(
        FileNotFoundError,
        match="Fixture文件不存在",
    ):
        load_multimodal_offline_fixtures(
            tmp_path / "missing.jsonl"
        )


def test_loader_rejects_blank_line_with_line_number(
    tmp_path: Path,
) -> None:
    """JSONL 中的空行必须报告准确行号，不能静默跳过。"""

    path = tmp_path / "fixtures.jsonl"
    first_fixture = make_fixture(
        "multimodal-agent-001"
    )

    path.write_text(
        json.dumps(
            first_fixture.model_dump(
                mode="json"
            )
        )
        + "\n\n",
        encoding="utf-8",
    )

    with pytest.raises(
        EvaluationDataError,
        match="第 2 行为空",
    ):
        load_multimodal_offline_fixtures(
            path
        )


def test_loader_rejects_invalid_json_with_line_number(
    tmp_path: Path,
) -> None:
    """JSON 语法错误必须指出发生错误的行。"""

    path = tmp_path / "fixtures.jsonl"
    path.write_text(
        '{"scenario_id":\n',
        encoding="utf-8",
    )

    with pytest.raises(
        EvaluationDataError,
        match="第 1 行不是合法 JSON",
    ):
        load_multimodal_offline_fixtures(
            path
        )


def test_loader_rejects_fixture_schema_error_with_line_number(
    tmp_path: Path,
) -> None:
    """字段合法性错误应转换为带行号的评测数据错误。"""

    path = tmp_path / "fixtures.jsonl"
    path.write_text(
        json.dumps(
            {
                "scenario_id": "invalid-id",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        EvaluationDataError,
        match=(
            "第 1 行不符合离线Fixture数据契约"
        ),
    ):
        load_multimodal_offline_fixtures(
            path
        )


def test_loader_rejects_duplicate_scenario_id(
    tmp_path: Path,
) -> None:
    """同一场景不能在 Fixture 文件中配置两份竞争脚本。"""

    path = tmp_path / "fixtures.jsonl"
    fixture = make_fixture(
        "multimodal-agent-001"
    )
    write_fixture_jsonl(
        path,
        (fixture, fixture),
    )

    with pytest.raises(
        EvaluationDataError,
        match="场景编号重复",
    ):
        load_multimodal_offline_fixtures(
            path
        )


def test_loader_rejects_empty_file(
    tmp_path: Path,
) -> None:
    """空文件不能作为没有任何回归分母的 Fixture 集合。"""

    path = tmp_path / "fixtures.jsonl"
    path.write_text("", encoding="utf-8")

    with pytest.raises(
        EvaluationDataError,
        match="Fixture文件不能为空",
    ):
        load_multimodal_offline_fixtures(
            path
        )


def test_regression_case_exposes_matching_scenario_id(
) -> None:
    """配对容器应提供统一的只读 scenario_id 属性。"""

    first_scenario, _ = (
        load_two_gold_scenarios()
    )

    case = MultimodalOfflineRegressionCase(
        scenario=first_scenario,
        fixture=make_fixture(
            first_scenario.scenario_id
        ),
    )

    assert case.scenario_id == (
        first_scenario.scenario_id
    )


def test_regression_case_rejects_mismatched_ids(
) -> None:
    """单个配对对象也必须拒绝 Gold 与 Fixture 编号不一致。"""

    first_scenario, second_scenario = (
        load_two_gold_scenarios()
    )

    with pytest.raises(
        ValueError,
        match="scenario_id必须一致",
    ):
        MultimodalOfflineRegressionCase(
            scenario=first_scenario,
            fixture=make_fixture(
                second_scenario.scenario_id
            ),
        )


def test_pairing_uses_ids_and_preserves_gold_order(
) -> None:
    """Fixture 行序相反时，配对结果仍应遵循 Gold 顺序。

    这验证配对依据是 scenario_id，而不是两个文件的行号。
    """

    first_scenario, second_scenario = (
        load_two_gold_scenarios()
    )

    cases = (
        pair_multimodal_offline_regression_cases(
            scenarios=[
                first_scenario,
                second_scenario,
            ],
            fixtures=[
                make_fixture(
                    second_scenario.scenario_id
                ),
                make_fixture(
                    first_scenario.scenario_id
                ),
            ],
        )
    )

    assert tuple(
        case.scenario_id
        for case in cases
    ) == (
        first_scenario.scenario_id,
        second_scenario.scenario_id,
    )
    assert cases[0].scenario is first_scenario
    assert cases[1].scenario is second_scenario


def test_pairing_reports_missing_and_orphan_ids_together(
) -> None:
    """集合不一致时应同时报告缺失 Fixture 和孤立 Fixture。"""

    first_scenario, second_scenario = (
        load_two_gold_scenarios()
    )

    with pytest.raises(
        EvaluationDataError,
    ) as exc_info:
        pair_multimodal_offline_regression_cases(
            scenarios=[
                first_scenario,
                second_scenario,
            ],
            fixtures=[
                make_fixture(
                    first_scenario.scenario_id
                ),
                make_fixture(
                    "multimodal-agent-030"
                ),
            ],
        )

    error_message = str(exc_info.value)

    assert "缺少Fixture的场景" in error_message
    assert second_scenario.scenario_id in error_message
    assert "没有对应Gold的Fixture" in error_message
    assert "multimodal-agent-030" in error_message


def test_pairing_rejects_duplicate_gold_ids(
) -> None:
    """直接调用配对函数时也不能重复计算同一 Gold 场景。"""

    first_scenario, _ = (
        load_two_gold_scenarios()
    )

    with pytest.raises(
        EvaluationDataError,
        match="Gold场景集合包含重复scenario_id",
    ):
        pair_multimodal_offline_regression_cases(
            scenarios=[
                first_scenario,
                first_scenario,
            ],
            fixtures=[
                make_fixture(
                    first_scenario.scenario_id
                )
            ],
        )


def test_pairing_rejects_duplicate_fixture_ids(
) -> None:
    """直接传入 Fixture 时也不能为一个场景提供两份脚本。"""

    first_scenario, _ = (
        load_two_gold_scenarios()
    )
    fixture = make_fixture(
        first_scenario.scenario_id
    )

    with pytest.raises(
        EvaluationDataError,
        match="Fixture集合包含重复scenario_id",
    ):
        pair_multimodal_offline_regression_cases(
            scenarios=[first_scenario],
            fixtures=[fixture, fixture],
        )


@pytest.mark.parametrize(
    ("scenarios", "fixtures", "message"),
    [
        ([], [make_fixture("multimodal-agent-001")], "Gold场景集合不能为空"),
        ([object()], [make_fixture("multimodal-agent-001")], "scenarios只能包含"),
        ([load_two_gold_scenarios()[0]], [], "离线Fixture集合不能为空"),
        ([load_two_gold_scenarios()[0]], [object()], "fixtures只能包含"),
    ],
)
def test_pairing_rejects_empty_or_wrong_element_collections(
    scenarios: list[object],
    fixtures: list[object],
    message: str,
) -> None:
    """配对函数必须保护自己对集合内容的公开调用边界。"""

    with pytest.raises(
        (TypeError, EvaluationDataError),
        match=message,
    ):
        pair_multimodal_offline_regression_cases(
            scenarios=scenarios,  # type: ignore[arg-type]
            fixtures=fixtures,  # type: ignore[arg-type]
        )
