"""Week 6 多模态 Agent 离线 Fixture 加载与配对服务。

本模块负责：

1. 从 JSONL 文件逐行读取离线 Fixture；
2. 使用 Pydantic 校验每条 Fixture；
3. 拒绝空行、无效 JSON、重复编号和空文件；
4. 将 Gold 场景与 Fixture 按 scenario_id 一一配对；
5. 拒绝缺少 Fixture 或出现孤立 Fixture 的数据集。

本模块不执行 Agent、不读取图片、不调用外部服务，
也不计算场景是否通过。
"""

from dataclasses import (
    dataclass,
)
import json
from pathlib import (
    Path,
)

from pydantic import (
    ValidationError,
)

from app.schemas.multimodal_agent_evaluation import (
    MultimodalAgentEvaluationScenario,
)
from app.schemas.multimodal_offline_regression import (
    MultimodalOfflineScenarioFixture,
)
from app.services.evaluation import (
    EvaluationDataError,
)


@dataclass(
    frozen=True,
    slots=True,
)
class MultimodalOfflineRegressionCase:
    """一条已经完成 Gold 与 Fixture 配对的离线回归案例。

    scenario 描述系统应该怎样表现；
    fixture 描述本次离线运行中外部依赖返回什么。

    本类只是内部配对容器。
    scenario 和 fixture 自身已经分别通过 Pydantic 校验，
    因此这里使用轻量 dataclass，而不重复定义字段约束。
    """

    scenario: MultimodalAgentEvaluationScenario
    fixture: MultimodalOfflineScenarioFixture

    def __post_init__(self) -> None:
        """校验两个对象的类型和场景编号。

        __post_init__ 会在 dataclass 自动生成的
        __init__ 执行结束后自动调用。
        """

        if not isinstance(
            self.scenario,
            MultimodalAgentEvaluationScenario,
        ):
            raise TypeError(
                "scenario必须是"
                "MultimodalAgentEvaluationScenario"
            )

        if not isinstance(
            self.fixture,
            MultimodalOfflineScenarioFixture,
        ):
            raise TypeError(
                "fixture必须是"
                "MultimodalOfflineScenarioFixture"
            )

        if (
            self.scenario.scenario_id
            != self.fixture.scenario_id
        ):
            raise ValueError(
                "scenario与fixture的"
                "scenario_id必须一致"
            )

    @property
    def scenario_id(self) -> str:
        """返回当前配对案例的稳定场景编号。

        调用方不需要知道编号存放在 scenario 中，
        可以直接读取 case.scenario_id。
        """

        return self.scenario.scenario_id


def load_multimodal_offline_fixtures(
    path: Path,
) -> tuple[
    MultimodalOfflineScenarioFixture,
    ...,
]:
    """从 JSONL 文件读取并校验离线 Fixture。

    返回顺序与 JSONL 中的行顺序完全一致，
    方便生成稳定、可复现的审计报告。

    本函数只加载 Fixture，不加载 Gold，也不运行 Agent。
    """

    if not isinstance(path, Path):
        raise TypeError(
            "path必须是pathlib.Path"
        )

    if path.suffix.lower() != ".jsonl":
        raise EvaluationDataError(
            "多模态离线Fixture文件必须使用"
            " .jsonl 扩展名"
        )

    if not path.is_file():
        raise FileNotFoundError(
            "多模态离线Fixture文件不存在："
            f"{path}"
        )

    fixtures: list[
        MultimodalOfflineScenarioFixture
    ] = []

    seen_scenario_ids: set[str] = set()

    # JSONL 每一行都是一个独立 JSON 对象。
    #
    # 逐行解析可以在发生错误时报告准确行号，
    # 而不必把整个文件作为一个大型 JSON 数组读取。
    with path.open(
        "r",
        encoding="utf-8",
    ) as fixture_file:
        for line_number, line in enumerate(
            fixture_file,
            start=1,
        ):
            stripped_line = line.strip()

            if not stripped_line:
                raise EvaluationDataError(
                    "多模态离线Fixture文件"
                    f"第 {line_number} 行为空"
                )

            try:
                raw_fixture = json.loads(
                    stripped_line
                )
            except json.JSONDecodeError as exc:
                raise EvaluationDataError(
                    "多模态离线Fixture文件"
                    f"第 {line_number} 行"
                    "不是合法 JSON"
                ) from exc

            try:
                fixture = (
                    MultimodalOfflineScenarioFixture
                    .model_validate(raw_fixture)
                )
            except ValidationError as exc:
                raise EvaluationDataError(
                    "多模态离线Fixture文件"
                    f"第 {line_number} 行不符合"
                    "离线Fixture数据契约"
                ) from exc

            if (
                fixture.scenario_id
                in seen_scenario_ids
            ):
                raise EvaluationDataError(
                    "多模态离线Fixture"
                    "场景编号重复："
                    f"{fixture.scenario_id}"
                )

            seen_scenario_ids.add(
                fixture.scenario_id
            )
            fixtures.append(fixture)

    if not fixtures:
        raise EvaluationDataError(
            "多模态离线Fixture文件不能为空"
        )

    return tuple(fixtures)


def pair_multimodal_offline_regression_cases(
    *,
    scenarios: (
        list[
            MultimodalAgentEvaluationScenario
        ]
        | tuple[
            MultimodalAgentEvaluationScenario,
            ...,
        ]
    ),
    fixtures: (
        list[
            MultimodalOfflineScenarioFixture
        ]
        | tuple[
            MultimodalOfflineScenarioFixture,
            ...,
        ]
    ),
) -> tuple[
    MultimodalOfflineRegressionCase,
    ...,
]:
    """将 Gold 场景与离线 Fixture 一一配对。

    配对使用 scenario_id，而不是依赖两个文件的行号。

    返回顺序以 Gold 场景顺序为准，使后续执行顺序、
    报告顺序和原始 Gold 文件保持一致。
    """

    if not isinstance(
        scenarios,
        (list, tuple),
    ):
        raise TypeError(
            "scenarios必须是list或tuple"
        )

    if not isinstance(
        fixtures,
        (list, tuple),
    ):
        raise TypeError(
            "fixtures必须是list或tuple"
        )

    if not scenarios:
        raise EvaluationDataError(
            "Gold场景集合不能为空"
        )

    if not fixtures:
        raise EvaluationDataError(
            "离线Fixture集合不能为空"
        )

    if not all(
        isinstance(
            scenario,
            MultimodalAgentEvaluationScenario,
        )
        for scenario in scenarios
    ):
        raise TypeError(
            "scenarios只能包含"
            "MultimodalAgentEvaluationScenario"
        )

    if not all(
        isinstance(
            fixture,
            MultimodalOfflineScenarioFixture,
        )
        for fixture in fixtures
    ):
        raise TypeError(
            "fixtures只能包含"
            "MultimodalOfflineScenarioFixture"
        )

    scenario_ids = tuple(
        scenario.scenario_id
        for scenario in scenarios
    )

    fixture_ids = tuple(
        fixture.scenario_id
        for fixture in fixtures
    )

    # 加载器已经检查过重复值，
    # 但本函数也是公开函数，调用方可能直接传入列表，
    # 因此这里仍要保护自己的调用边界。
    if len(scenario_ids) != len(
        set(scenario_ids)
    ):
        raise EvaluationDataError(
            "Gold场景集合包含重复scenario_id"
        )

    if len(fixture_ids) != len(
        set(fixture_ids)
    ):
        raise EvaluationDataError(
            "离线Fixture集合包含重复scenario_id"
        )

    scenario_id_set = set(scenario_ids)
    fixture_id_set = set(fixture_ids)

    # missing_fixture_ids 表示有 Gold，
    # 但没有定义对应的外部依赖行为。
    missing_fixture_ids = sorted(
        scenario_id_set - fixture_id_set
    )

    # orphan_fixture_ids 表示有 Fixture，
    # 但没有对应 Gold，无法判断执行结果是否正确。
    orphan_fixture_ids = sorted(
        fixture_id_set - scenario_id_set
    )

    data_errors: list[str] = []

    if missing_fixture_ids:
        data_errors.append(
            "缺少Fixture的场景："
            + "、".join(missing_fixture_ids)
        )

    if orphan_fixture_ids:
        data_errors.append(
            "没有对应Gold的Fixture："
            + "、".join(orphan_fixture_ids)
        )

    if data_errors:
        raise EvaluationDataError(
            "；".join(data_errors)
        )

    fixture_by_scenario_id = {
        fixture.scenario_id: fixture
        for fixture in fixtures
    }

    # 这里按 scenarios 的顺序构造结果，
    # 不使用 Fixture 文件自身的行顺序。
    #
    # 因而即使 Fixture 文件被重新排序，
    # 最终回归报告仍然保持 Gold 的稳定顺序。
    return tuple(
        MultimodalOfflineRegressionCase(
            scenario=scenario,
            fixture=(
                fixture_by_scenario_id[
                    scenario.scenario_id
                ]
            ),
        )
        for scenario in scenarios
    )