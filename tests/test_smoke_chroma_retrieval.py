"""持久化Chroma冒烟脚本的离线参数测试。"""

import pytest

from scripts.smoke_chroma_retrieval import (
    DEFAULT_QUERY,
    DEFAULT_TOP_K,
    parse_args,
)


def test_parse_args_uses_public_demo_defaults(
) -> None:
    """默认参数必须适用于公开脱敏语料。"""

    # 传入空列表表示没有用户覆盖参数，
    # 同时避免读取pytest进程自己的命令行。
    args = parse_args([])

    # 默认问题必须指向公开样例中的故障码，
    # 不能继续依赖未公开的Week2完整语料。
    assert args.query == DEFAULT_QUERY
    assert "ERR-DEMO-1001" in args.query

    # 默认检索前三个候选。
    assert args.top_k == DEFAULT_TOP_K
    assert args.top_k == 3


def test_parse_args_accepts_user_overrides(
) -> None:
    """用户参数应被清理并转换成正确类型。"""

    args = parse_args([
        "--query",
        "  自定义检索问题  ",
        "--top-k",
        "1",
    ])

    # parse_non_blank_query()删除了两端空格。
    assert args.query == "自定义检索问题"

    # argparse调用parse_positive_int()，
    # 将字符串"1"转换成整数1。
    assert args.top_k == 1


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param(
            [
                "--query",
                "   ",
            ],
            id="blank-query",
        ),
        pytest.param(
            [
                "--top-k",
                "0",
            ],
            id="non-positive-top-k",
        ),
    ],
)
def test_parse_args_rejects_invalid_values(
    argv: list[str],
) -> None:
    """空问题和非正Top-K必须被命令行层拒绝。"""

    # argparse遇到参数错误时会抛出SystemExit，
    # 退出码2表示命令行使用方式错误。
    with pytest.raises(SystemExit) as exc_info:
        parse_args(argv)

    assert exc_info.value.code == 2