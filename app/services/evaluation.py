"""读取并校验检索评测使用的 JSONL 文件。"""

import json

from pathlib import Path

# ValidationError 表示数据没有通过 Pydantic 模型校验。
from pydantic import ValidationError

from app.schemas.evaluation import GoldQuestion


# AgentEvaluationScenario定义Week 4固定Agent评测场景。
#
# 加载器会把每一行JSON转换成该Pydantic模型，
# 在发出任何真实Agent请求之前完成数据校验。
from app.schemas.agent_evaluation import (
    AgentEvaluationScenario,
)


class EvaluationDataError(ValueError):
    """评测数据文件格式或内容不符合要求。"""


def load_gold_questions(
    path: Path,
) -> list[GoldQuestion]:
    """逐行读取 JSONL 并返回经过校验的评测问题。"""

    if path.suffix.lower() != ".jsonl":
        raise EvaluationDataError(
            "评测问题文件必须使用 .jsonl 扩展名"
        )

    if not path.is_file():
        raise FileNotFoundError(
            f"评测问题文件不存在：{path}"
        )

    questions: list[GoldQuestion] = []

    # 用于检查 question_id 是否重复。
    seen_question_ids: set[str] = set()

    # 按文本模式逐行读取，
    # 不需要一次把整个评测文件加载到内存。
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        # enumerate(..., start=1) 使行号与编辑器一致。
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            stripped_line = line.strip()

            # JSONL 中间出现空行可能意味着数据被意外编辑，
            # 因此这里选择明确报错。
            if not stripped_line:
                raise EvaluationDataError(
                    f"评测文件第 {line_number} 行为空"
                )

            try:
                # loads() 只解析当前一行。
                raw_question = json.loads(
                    stripped_line
                )
            except json.JSONDecodeError as exc:
                raise EvaluationDataError(
                    f"评测文件第 {line_number} 行不是合法 JSON"
                ) from exc

            try:
                # model_validate() 将 dict 转换成 GoldQuestion，
                # 并执行字段约束和 model_validator。
                question = GoldQuestion.model_validate(
                    raw_question
                )
            except ValidationError as exc:
                raise EvaluationDataError(
                    f"评测文件第 {line_number} 行不符合问题契约"
                ) from exc

            if question.question_id in seen_question_ids:
                raise EvaluationDataError(
                    "评测问题编号重复："
                    f"{question.question_id}"
                )

            seen_question_ids.add(
                question.question_id
            )
            questions.append(question)

    if not questions:
        raise EvaluationDataError(
            "评测问题文件不能为空"
        )

    return questions


def load_agent_evaluation_scenarios(
    path: Path,
) -> list[AgentEvaluationScenario]:
    """逐行读取并校验Agent固定评测场景JSONL。

    本函数只负责加载和验证数据，不负责：

    1. 调用Agent API；
    2. 执行工具；
    3. 计算评分；
    4. 生成报告。

    只有整个文件全部通过校验后，
    调用方才会取得完整的场景列表。
    """

    # suffix取得路径的最后一个扩展名。
    #
    # lower()允许.JSONL等大小写写法，
    # 但仍然拒绝普通.json文件。
    if path.suffix.lower() != ".jsonl":
        raise EvaluationDataError(
            "Agent评测场景文件必须使用"
            " .jsonl 扩展名"
        )

    # is_file()同时确认路径存在并且是普通文件。
    #
    # 一个同名目录不能被当成评测文件读取。
    if not path.is_file():
        raise FileNotFoundError(
            "Agent评测场景文件不存在："
            f"{path}"
        )

    scenarios: list[
        AgentEvaluationScenario
    ] = []

    # set用于记录已经成功加载的场景编号。
    #
    # set查找通常是常数时间，适合逐行检查重复ID。
    seen_scenario_ids: set[str] = set()

    # with语句保证正常结束或发生异常时，
    # 文件句柄都会被自动关闭。
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        # start=1使这里的行号与编辑器显示一致。
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            stripped_line = line.strip()

            # JSONL规定每一行是一条完整JSON记录。
            #
            # 中间空行可能意味着场景被误删，
            # 因此不能静默跳过。
            if not stripped_line:
                raise EvaluationDataError(
                    "Agent评测场景文件"
                    f"第 {line_number} 行为空"
                )

            try:
                # json.loads()只解析当前这一行，
                # 不会把整个JSONL当成一个JSON数组。
                raw_scenario = json.loads(
                    stripped_line
                )
            except json.JSONDecodeError as exc:
                # from exc保留底层JSON解析异常作为原因，
                # 同时对外提供包含行号的稳定错误。
                raise EvaluationDataError(
                    "Agent评测场景文件"
                    f"第 {line_number} 行"
                    "不是合法 JSON"
                ) from exc

            try:
                # model_validate()会执行：
                #
                # 1. 顶层字段类型和长度校验；
                # 2. request嵌套请求契约校验；
                # 3. 工具集合关系校验；
                # 4. 执行状态和终止原因关系校验；
                # 5. 完成、拒答和证据关系校验。
                scenario = (
                    AgentEvaluationScenario
                    .model_validate(
                        raw_scenario
                    )
                )
            except ValidationError as exc:
                raise EvaluationDataError(
                    "Agent评测场景文件"
                    f"第 {line_number} 行不符合"
                    "Agent评测场景契约"
                ) from exc

            if (
                scenario.scenario_id
                in seen_scenario_ids
            ):
                raise EvaluationDataError(
                    "Agent评测场景编号重复: "
                    f"{scenario.scenario_id}"
                )

            seen_scenario_ids.add(
                scenario.scenario_id
            )
            scenarios.append(
                scenario
            )

    # 一个空文件虽然没有语法错误，
    # 但无法产生任何评测分母和指标。
    if not scenarios:
        raise EvaluationDataError(
            "Agent评测场景文件不能为空"
        )

    # list顺序与JSONL中的行顺序完全一致，
    # 后续报告也会保持该稳定顺序。
    return scenarios