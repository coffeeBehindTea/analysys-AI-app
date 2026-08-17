"""评测问题模型与 JSONL 加载器测试。"""

from pathlib import Path

import pytest

from app.services.evaluation import (
    EvaluationDataError,
    load_gold_questions,
)


def test_load_gold_questions(
    tmp_path: Path,
) -> None:
    """加载器应解析可回答和不可回答问题。"""

    path = tmp_path / "questions.jsonl"

    path.write_text(
        "\n".join(
            [
                (
                    '{"question_id":"q001",'
                    '"question":"机器人何时停车？",'
                    '"question_type":"single_hop",'
                    '"answerable":true,'
                    '"reference_answer":"触发安全保护时停车。",'
                    '"expected_evidence":['
                    '{"source_file":"manual.md",'
                    '"page_or_section":"section: safety"}],'
                    '"tags":["safety"],'
                    '"notes":"测试安全规则召回。"}'
                ),
                (
                    '{"question_id":"q002",'
                    '"question":"当前机器人在哪里？",'
                    '"question_type":"unanswerable",'
                    '"answerable":false,'
                    '"reference_answer":null,'
                    '"expected_evidence":[],'
                    '"tags":["realtime"],'
                    '"notes":"知识库没有实时状态。"}'
                ),
            ]
        ),
        encoding="utf-8",
    )

    questions = load_gold_questions(path)

    assert len(questions) == 2

    assert questions[0].question_id == "q001"
    assert questions[0].answerable is True
    assert (
        questions[0]
        .expected_evidence[0]
        .source_file
        == "manual.md"
    )

    assert questions[1].question_type == "unanswerable"
    assert questions[1].reference_answer is None


def test_invalid_json_reports_line_number(
    tmp_path: Path,
) -> None:
    """JSON 语法错误应指出所在行。"""

    path = tmp_path / "questions.jsonl"

    path.write_text(
        '{"question_id":"q001"}\n'
        '{"broken json"}',
        encoding="utf-8",
    )

    with pytest.raises(
        EvaluationDataError,
        match="第 1 行不符合问题契约",
    ):
        load_gold_questions(path)


def test_duplicate_question_id_is_rejected(
    tmp_path: Path,
) -> None:
    """评测问题编号不能重复。"""

    valid_line = (
        '{"question_id":"q001",'
        '"question":"测试问题",'
        '"question_type":"single_hop",'
        '"answerable":true,'
        '"reference_answer":"测试答案",'
        '"expected_evidence":['
        '{"source_file":"manual.md",'
        '"page_or_section":"section: test"}],'
        '"tags":["test"],'
        '"notes":"测试说明"}'
    )

    path = tmp_path / "questions.jsonl"
    path.write_text(
        valid_line + "\n" + valid_line,
        encoding="utf-8",
    )

    with pytest.raises(
        EvaluationDataError,
        match="问题编号重复",
    ):
        load_gold_questions(path)


def test_unanswerable_question_cannot_have_evidence(
    tmp_path: Path,
) -> None:
    """无答案问题不能错误地标注预期来源。"""

    path = tmp_path / "questions.jsonl"

    path.write_text(
        (
            '{"question_id":"q001",'
            '"question":"未知问题",'
            '"question_type":"unanswerable",'
            '"answerable":false,'
            '"reference_answer":null,'
            '"expected_evidence":['
            '{"source_file":"manual.md",'
            '"page_or_section":"section: test"}],'
            '"tags":["test"],'
            '"notes":"错误示例"}'
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        EvaluationDataError,
        match="不符合问题契约",
    ):
        load_gold_questions(path)