"""诊断会话本地JSON存储的离线单元测试。

测试只访问pytest提供的临时目录，
不会写入正式的data/diagnostic_sessions目录，
也不会调用Agent、LLM、Vision、知识库或网络。
"""

import asyncio
from datetime import (
    datetime,
    timedelta,
    timezone,
)
from pathlib import Path

import pytest

from app.schemas.diagnostic_session import (
    DiagnosticSessionDiagnosisSnapshot,
    DiagnosticSessionEvidenceCoverage,
    DiagnosticSessionMetrics,
    DiagnosticSessionRecord,
    DiagnosticSessionRequestSummary,
)
from app.services.diagnostic_session_store import (
    DiagnosticSessionAlreadyExistsError,
    DiagnosticSessionCorruptedError,
    DiagnosticSessionStore,
)


# 三个固定值都符合DiagnosticSessionId要求的
# 小写UUID4格式，用于构造可重复的多会话测试数据。
TEST_SESSION_ID = (
    "123e4567-e89b-42d3-a456-426614174000"
)
TEST_SECOND_SESSION_ID = (
    "223e4567-e89b-42d3-a456-426614174001"
)
TEST_THIRD_SESSION_ID = (
    "323e4567-e89b-42d3-a456-426614174002"
)

# 固定UTC时间用于验证最近会话的排序。
TEST_CREATED_AT = datetime(
    2026,
    9,
    9,
    8,
    0,
    tzinfo=timezone.utc,
)


def make_record(
    *,
    session_id: str = TEST_SESSION_ID,
    created_at: datetime = TEST_CREATED_AT,
    robot_id: str = "robot-001",
) -> DiagnosticSessionRecord:
    """创建一条无工具调用、证据不足的合法会话。

    使用abstained会话可以把测试关注点限定在存储行为，
    不需要为每个测试重复构造知识引用和工具轨迹。
    """

    return DiagnosticSessionRecord(
        session_id=session_id,
        request_id=f"request-{session_id}",
        created_at=created_at,
        request_summary=(
            DiagnosticSessionRequestSummary(
                robot_id=robot_id,
                symptom_summary=(
                    "知识库没有足够证据确认故障原因"
                ),
                task_goal_summary=(
                    "确认还需要补充哪些信息"
                ),
                log_excerpt_sha256="a" * 64,
                log_excerpt_char_count=32,
                image_count=0,
            )
        ),
        status="abstained",
        execution_state="completed",
        termination_reason="planner_finished",
        finish_reason="insufficient_information",
        diagnosis=(
            DiagnosticSessionDiagnosisSnapshot(
                prompt_version=(
                    "agent-tool-calling-v2"
                ),
                gate_version=(
                    "agent-confirmed-evidence-v1"
                ),
                status="abstained",
                evidence=(),
                possible_causes=(),
                next_checks=(),
                risk_level="unknown",
                missing_information=(
                    "当前请求没有取得足够证据",
                ),
                abstained=True,
            )
        ),
        tool_events=(),
        vision_observations=(),
        telemetry_observations=(),
        test_case_drafts=(),
        evidence_coverage=(
            DiagnosticSessionEvidenceCoverage()
        ),
        metrics=DiagnosticSessionMetrics(
            total_duration_ms=8.5,
            tool_step_count=0,
            failed_tool_count=0,
            failed_tool_names=(),
            confirmed_source_count=0,
            covered_evidence_count=0,
            citation_count=0,
        ),
    )


def make_store(
    tmp_path: Path,
) -> DiagnosticSessionStore:
    """创建指向pytest独立临时目录的存储对象。"""

    return DiagnosticSessionStore(
        root_directory=(
            tmp_path / "diagnostic_sessions"
        )
    )


def test_constructor_does_not_create_directory(
    tmp_path: Path,
) -> None:
    """仅构造Store时不应产生文件系统写入。"""

    store = make_store(tmp_path)

    assert store.root_directory.is_absolute()
    assert not store.root_directory.exists()


@pytest.mark.asyncio
async def test_save_creates_utf8_json_file(
    tmp_path: Path,
) -> None:
    """首次保存应创建目录和以session_id命名的JSON。"""

    store = make_store(tmp_path)
    record = make_record()

    saved_record = await store.save(record)

    output_path = (
        store.root_directory
        / f"{record.session_id}.json"
    )

    assert saved_record == record
    assert output_path.is_file()
    assert record.session_id in output_path.read_text(
        encoding="utf-8"
    )
    assert not tuple(
        store.root_directory.glob("*.tmp")
    )


@pytest.mark.asyncio
async def test_get_round_trips_saved_record(
    tmp_path: Path,
) -> None:
    """保存后按ID读取应恢复同一份Pydantic记录。"""

    store = make_store(tmp_path)
    record = make_record()

    await store.save(record)
    loaded_record = await store.get(
        record.session_id
    )

    assert loaded_record == record


@pytest.mark.asyncio
async def test_get_missing_session_returns_none(
    tmp_path: Path,
) -> None:
    """合法但不存在的ID应返回None供路由转换成404。"""

    store = make_store(tmp_path)

    loaded_record = await store.get(
        TEST_SESSION_ID
    )

    assert loaded_record is None


@pytest.mark.asyncio
async def test_list_recent_returns_empty_response(
    tmp_path: Path,
) -> None:
    """存储目录尚不存在时应返回稳定空列表。"""

    store = make_store(tmp_path)

    response = await store.list_recent(
        limit=10
    )

    assert response.sessions == ()
    assert response.total == 0
    assert response.limit == 10


@pytest.mark.asyncio
async def test_list_recent_sorts_newest_first_and_applies_limit(
    tmp_path: Path,
) -> None:
    """列表应按业务时间倒序，并保留截取前的总数。"""

    store = make_store(tmp_path)

    oldest = make_record(
        session_id=TEST_SESSION_ID,
        created_at=TEST_CREATED_AT,
        robot_id="robot-oldest",
    )
    newest = make_record(
        session_id=TEST_SECOND_SESSION_ID,
        created_at=(
            TEST_CREATED_AT
            + timedelta(minutes=2)
        ),
        robot_id="robot-newest",
    )
    middle = make_record(
        session_id=TEST_THIRD_SESSION_ID,
        created_at=(
            TEST_CREATED_AT
            + timedelta(minutes=1)
        ),
        robot_id="robot-middle",
    )

    # 故意不按时间顺序保存，证明排序依据是created_at，
    # 而不是文件创建顺序或glob()返回顺序。
    await store.save(middle)
    await store.save(oldest)
    await store.save(newest)

    response = await store.list_recent(
        limit=2
    )

    assert response.total == 3
    assert response.limit == 2
    assert tuple(
        item.session_id
        for item in response.sessions
    ) == (
        TEST_SECOND_SESSION_ID,
        TEST_THIRD_SESSION_ID,
    )
    assert all(
        not hasattr(item, "tool_events")
        for item in response.sessions
    )


@pytest.mark.asyncio
async def test_save_rejects_duplicate_session_id(
    tmp_path: Path,
) -> None:
    """相同ID不能覆盖已经存在的审计记录。"""

    store = make_store(tmp_path)
    record = make_record()

    await store.save(record)

    with pytest.raises(
        DiagnosticSessionAlreadyExistsError,
        match="已经存在",
    ):
        await store.save(record)

    # 重复保存失败后，原始记录仍然可以正常读取。
    assert await store.get(
        record.session_id
    ) == record


@pytest.mark.asyncio
async def test_get_rejects_path_traversal_session_id(
    tmp_path: Path,
) -> None:
    """任意路径文本不能越过会话存储目录。"""

    store = make_store(tmp_path)

    with pytest.raises(
        ValueError,
        match="小写UUID4",
    ):
        await store.get("../../.env")


@pytest.mark.asyncio
async def test_save_requires_validated_record(
    tmp_path: Path,
) -> None:
    """普通字典不能绕过DiagnosticSessionRecord契约。"""

    store = make_store(tmp_path)

    with pytest.raises(
        TypeError,
        match="DiagnosticSessionRecord",
    ):
        await store.save(  # type: ignore[arg-type]
            {"session_id": TEST_SESSION_ID}
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_limit",
    [True, "10", 1.5],
)
async def test_list_recent_rejects_non_integer_limit(
    tmp_path: Path,
    invalid_limit: object,
) -> None:
    """bool、字符串和小数都不能作为列表数量。"""

    store = make_store(tmp_path)

    with pytest.raises(
        TypeError,
        match="limit必须是int",
    ):
        await store.list_recent(
            limit=invalid_limit,  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_limit",
    [0, 101],
)
async def test_list_recent_rejects_out_of_range_limit(
    tmp_path: Path,
    invalid_limit: int,
) -> None:
    """最近会话数量必须限制在1到100之间。"""

    store = make_store(tmp_path)

    with pytest.raises(
        ValueError,
        match="1到100",
    ):
        await store.list_recent(
            limit=invalid_limit
        )


@pytest.mark.asyncio
async def test_get_rejects_corrupted_json(
    tmp_path: Path,
) -> None:
    """被截断的JSON不能作为合法历史会话返回。"""

    store = make_store(tmp_path)
    store.root_directory.mkdir(
        parents=True
    )
    path = (
        store.root_directory
        / f"{TEST_SESSION_ID}.json"
    )
    path.write_text(
        '{"session_id":',
        encoding="utf-8",
    )

    with pytest.raises(
        DiagnosticSessionCorruptedError,
        match="内容无效",
    ):
        await store.get(TEST_SESSION_ID)


@pytest.mark.asyncio
async def test_get_rejects_filename_record_id_mismatch(
    tmp_path: Path,
) -> None:
    """文件名和JSON中的session_id不一致时应拒绝读取。"""

    store = make_store(tmp_path)
    store.root_directory.mkdir(
        parents=True
    )
    record = make_record(
        session_id=TEST_SECOND_SESSION_ID
    )
    path = (
        store.root_directory
        / f"{TEST_SESSION_ID}.json"
    )
    path.write_text(
        record.model_dump_json(indent=2),
        encoding="utf-8",
    )

    with pytest.raises(
        DiagnosticSessionCorruptedError,
        match="session_id不一致",
    ):
        await store.get(TEST_SESSION_ID)


@pytest.mark.asyncio
async def test_list_recent_rejects_invalid_json_filename(
    tmp_path: Path,
) -> None:
    """专用存储目录中的非法JSON文件应被视为完整性问题。"""

    store = make_store(tmp_path)
    store.root_directory.mkdir(
        parents=True
    )
    invalid_path = (
        store.root_directory
        / "manually-created.json"
    )
    invalid_path.write_text(
        make_record().model_dump_json(
            indent=2
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        DiagnosticSessionCorruptedError,
        match="文件名不是合法",
    ):
        await store.list_recent()


@pytest.mark.asyncio
async def test_concurrent_distinct_saves_preserve_all_records(
    tmp_path: Path,
) -> None:
    """同一Store的并发保存应经过锁串行化且不丢记录。"""

    store = make_store(tmp_path)
    records = (
        make_record(
            session_id=TEST_SESSION_ID
        ),
        make_record(
            session_id=TEST_SECOND_SESSION_ID
        ),
        make_record(
            session_id=TEST_THIRD_SESSION_ID
        ),
    )

    saved_records = await asyncio.gather(
        *(
            store.save(record)
            for record in records
        )
    )
    response = await store.list_recent(
        limit=10
    )

    assert tuple(saved_records) == records
    assert response.total == 3


def test_constructor_rejects_existing_file_as_root(
    tmp_path: Path,
) -> None:
    """存储根路径已经是普通文件时应立即报告配置错误。"""

    file_path = tmp_path / "not-a-directory"
    file_path.write_text(
        "occupied",
        encoding="utf-8",
    )

    with pytest.raises(
        NotADirectoryError,
        match="必须是目录",
    ):
        DiagnosticSessionStore(
            root_directory=file_path
        )
