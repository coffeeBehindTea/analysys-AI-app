"""读取并校验检索评测使用的 JSONL 文件。"""

import json

from pathlib import Path

# ValidationError 表示数据没有通过 Pydantic 模型校验。
from pydantic import ValidationError

from app.schemas.evaluation import GoldQuestion


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