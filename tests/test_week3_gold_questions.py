"""Week 3扩展Gold Question数据集的质量测试。"""

from collections import Counter

from pathlib import Path

# pytest提供Fixture、参数化测试和测试发现能力。
import pytest

# GoldQuestion是每道Gold问题通过校验后的Pydantic模型。
from app.schemas.evaluation import GoldQuestion

# load_gold_questions负责逐行读取JSONL，
# 执行JSON解析、Pydantic校验和重复ID检查。
from app.services.evaluation import (
    load_gold_questions,
)


# __file__是当前测试文件的位置。
#
# resolve()将其转换成规范化绝对路径；
# parents[0]是tests目录；
# parents[1]是项目根目录。
PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

# 使用项目根目录拼接Gold文件路径，
# 避免依赖执行pytest时的当前工作目录。
GOLD_QUESTION_PATH = (
    PROJECT_ROOT
    / "data"
    / "eval"
    / "gold_questions.jsonl"
)

# Week 3本轮新增的8道困难问题。
NEW_DIFFICULT_QUESTION_IDS = {
    "q021",
    "q022",
    "q023",
    "q024",
    "q025",
    "q026",
    "q027",
    "q028",
}


@pytest.fixture(scope="module")
def gold_questions() -> list[GoldQuestion]:
    """加载一次真实Gold数据并提供给本模块测试。"""

    # scope="module"表示本测试文件中的全部测试
    # 共享同一次加载结果，不重复读取JSONL。
    return load_gold_questions(
        GOLD_QUESTION_PATH
    )


@pytest.fixture(scope="module")
def questions_by_id(
    gold_questions: list[GoldQuestion],
) -> dict[str, GoldQuestion]:
    """建立question_id到GoldQuestion的索引。"""

    # 字典让后续测试能够按q021等ID直接取题，
    # 不需要反复遍历完整列表。
    return {
        question.question_id: question
        for question in gold_questions
    }


def test_gold_dataset_contains_required_questions(
    gold_questions: list[GoldQuestion],
) -> None:
    """数据集应至少包含q001至q028且顺序稳定。"""

    question_ids = [
        question.question_id
        for question in gold_questions
    ]

    # Week 3要求在原20题基础上至少新增8题，
    # 因此使用>=28，而不是阻止未来继续增加问题。
    assert len(question_ids) >= 28

    required_ids = {
        f"q{index:03d}"
        for index in range(1, 29)
    }

    # issubset()检查q001至q028是否都存在，
    # 同时允许未来增加q029、q030等问题。
    assert required_ids.issubset(
        set(question_ids)
    )

    # 固定三位数字的ID可以直接按字符串排序。
    #
    # 该断言保证报告和Git diff中的问题顺序稳定。
    assert question_ids == sorted(question_ids)


def test_gold_dataset_has_required_distribution(
    gold_questions: list[GoldQuestion],
) -> None:
    """题型、答案类型和证据数应满足扩展基线。"""

    # Counter会得到类似：
    #
    # {
    #     "single_hop": 15,
    #     "multi_hop": 8,
    #     "unanswerable": 5,
    # }
    type_counts = Counter(
        question.question_type
        for question in gold_questions
    )

    assert type_counts["single_hop"] >= 15
    assert type_counts["multi_hop"] >= 8
    assert type_counts["unanswerable"] >= 5

    answerable_count = sum(
        question.answerable
        for question in gold_questions
    )

    unanswerable_count = sum(
        not question.answerable
        for question in gold_questions
    )

    expected_evidence_count = sum(
        len(question.expected_evidence)
        for question in gold_questions
    )

    assert answerable_count >= 23
    assert unanswerable_count >= 5
    assert expected_evidence_count >= 30


def test_q021_requires_fault_steps_and_reset_safety_evidence(
    questions_by_id: dict[str, GoldQuestion],
) -> None:
    """q021必须保留排查步骤和安全复位条件。"""

    question = questions_by_id["q021"]

    # 该问题同时需要故障说明和测试规程，
    # 因此必须使用multi_hop数据契约。
    assert question.question_type == "multi_hop"
    assert question.answerable is True

    evidence_locations = {
        (
            evidence.source_file,
            evidence.page_or_section,
        )
        for evidence in question.expected_evidence
    }

    # 第一份证据提供现场排查和手动复位步骤。
    assert (
        "仓储机器人故障说明.txt",
        "section: ERR-SAF-1002",
    ) in evidence_locations

    # 第二份证据提供不得跳过的
    # 触发解除、回弹和人工确认条件。
    assert (
        (
            "京东无人仓场景-"
            "仓储机器人测试规程与判定标准.md"
        ),
        (
            "section: 8.3 TEST-SAF-003 "
            "机械防撞条触发"
        ),
    ) in evidence_locations

    # 参考答案也必须保留这些安全语义，
    # 不能只在expected_evidence中增加位置。
    assert question.reference_answer is not None
    assert "触发条件未解除" in (
        question.reference_answer
    )
    assert "人工确认" in (
        question.reference_answer
    )


def test_new_questions_cover_required_challenges(
    questions_by_id: dict[str, GoldQuestion],
) -> None:
    """新增题应覆盖错误码、型号、数值和多跳流程。"""

    new_questions = [
        questions_by_id[question_id]
        for question_id in sorted(
            NEW_DIFFICULT_QUESTION_IDS
        )
    ]

    # 将全部新增题的tag展开后统计。
    tag_counts = Counter(
        tag
        for question in new_questions
        for tag in question.tags
    )

    # 至少3题覆盖错误码格式变体：
    # q021、q022、q026。
    assert (
        tag_counts["fault-code-variant"]
        >= 3
    )

    # 至少2题覆盖DM260型号别名：
    # q023、q027。
    assert tag_counts["model-alias"] >= 2

    # 至少3题覆盖数值或边界：
    # q024、q025、q026。
    assert tag_counts["numeric"] >= 3

    # 至少3题覆盖跨文档流程：
    # q022、q026、q027。
    assert tag_counts["multi-hop"] >= 3


@pytest.mark.parametrize(
    (
        "base_question_id",
        "difficult_question_id",
    ),
    [
        pytest.param(
            "q009",
            "q021",
            id="fault-code-space-variant",
        ),
        pytest.param(
            "q010",
            "q022",
            id="fault-code-underscore-variant",
        ),
        pytest.param(
            "q007",
            "q023",
            id="model-alias-variant",
        ),
        pytest.param(
            "q008",
            "q024",
            id="negative-temperature-variant",
        ),
        pytest.param(
            "q014",
            "q026",
            id="fault-injection-variant",
        ),
        pytest.param(
            "q016",
            "q027",
            id="cross-document-model-variant",
        ),
    ],
)
def test_difficult_variant_reuses_known_evidence(
    base_question_id: str,
    difficult_question_id: str,
    questions_by_id: dict[str, GoldQuestion],
) -> None:
    """困难改写题应复用已知证据而不是编造新来源。"""

    base_question = questions_by_id[
        base_question_id
    ]

    difficult_question = questions_by_id[
        difficult_question_id
    ]

    base_evidence = {
        (
            evidence.source_file,
            evidence.page_or_section,
        )
        for evidence
        in base_question.expected_evidence
    }

    difficult_evidence = {
        (
            evidence.source_file,
            evidence.page_or_section,
        )
        for evidence
        in difficult_question.expected_evidence
    }

    # 两道题的文字必须不同，
    # 否则只是重复复制原题。
    assert (
        base_question.question
        != difficult_question.question
    )

    # &是集合交集运算。
    #
    # 至少共享一项已验证证据，说明困难题是在
    # 控制其他条件的情况下改变错误码、型号、
    # 数值表达或组合方式。
    assert base_evidence & difficult_evidence


def test_evidence_locations_are_stable_and_safe(
    gold_questions: list[GoldQuestion],
) -> None:
    """证据只保存文件名和稳定页码或章节。"""

    for question in gold_questions:
        for evidence in question.expected_evidence:
            # source_file不能包含服务器绝对路径
            # 或任意目录结构。
            assert "/" not in evidence.source_file
            assert "\\" not in evidence.source_file

            # 当前支持的稳定定位形式是：
            # page: 物理页码
            # section: 标题或故障码
            assert (
                evidence.page_or_section.startswith(
                    (
                        "page: ",
                        "section: ",
                    )
                )
            )


def test_new_dataset_contains_relationship_abstention(
    questions_by_id: dict[str, GoldQuestion],
) -> None:
    """扩展集应包含实体存在但关系不存在的问题。"""

    question = questions_by_id["q028"]

    assert question.question_type == "unanswerable"
    assert question.answerable is False
    assert question.reference_answer is None
    assert question.expected_evidence == []

    # 该标签明确说明它测试的是
    # “两个实体都出现但没有关系证据”。
    assert "vendor-association" in question.tags
