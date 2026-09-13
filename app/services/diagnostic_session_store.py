"""诊断会话的本地JSON持久化存储。

本模块负责：

1. 把经过DiagnosticSessionRecord校验的会话写入JSON文件；
2. 根据session_id读取一条完整会话；
3. 按created_at从新到旧查询最近会话；
4. 重新使用Pydantic校验从磁盘读取的数据；
5. 拒绝非法session_id和损坏的存储文件；
6. 通过asyncio.to_thread避免阻塞异步事件循环。

本模块不负责：

1. 执行Agent；
2. 构造或脱敏会话；
3. 生成HTTP响应；
4. 保存完整图片、完整日志、密钥或模型私有思维链。
"""

import asyncio
from contextlib import suppress
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from pydantic import (
    TypeAdapter,
    ValidationError,
)

from app.errors import (
    DiagnosticSessionAlreadyExistsError,
    DiagnosticSessionCorruptedError,
    DiagnosticSessionStoreError,
)
from app.schemas.diagnostic_session import (
    DiagnosticSessionId,
    DiagnosticSessionListResponse,
    DiagnosticSessionRecord,
)
from app.services.diagnostic_session_builder import (
    build_diagnostic_session_summary,
)


# 最近会话列表一次最多返回100条。
#
# 这个限制与DiagnosticSessionListResponse中的
# sessions和limit字段上限保持一致。
MAX_RECENT_SESSION_LIMIT = 100


# DiagnosticSessionId是Annotated类型别名，
# 不能像BaseModel那样直接调用model_validate()。
#
# TypeAdapter可以为普通类型、Annotated类型、
# tuple和联合类型提供Pydantic校验能力。
_SESSION_ID_ADAPTER = TypeAdapter(
    DiagnosticSessionId
)


class DiagnosticSessionStoreProtocol(
    Protocol
):
    """会话存储层对上层公开的行为契约。

    Agent业务服务或会话查询路由只需要依赖
    这三个方法，不必依赖本地JSON存储的实现细节。

    测试中可以提供内存Fake Store；
    未来也可以替换成SQLite或PostgreSQL实现。
    """

    async def save(
        self,
        record: DiagnosticSessionRecord,
        /,
    ) -> DiagnosticSessionRecord:
        """保存并返回会话记录。"""

        ...

    async def get(
        self,
        session_id: str,
        /,
    ) -> DiagnosticSessionRecord | None:
        """根据ID查询完整会话。"""

        ...

    async def list_recent(
        self,
        *,
        limit: int = 20,
    ) -> DiagnosticSessionListResponse:
        """查询最近的会话摘要。"""

        ...


def _validate_session_id(
    session_id: str,
    /,
) -> str:
    """验证外部传入的session_id。

    验证完成后才能把session_id用于文件名。

    DiagnosticSessionId只接受小写UUID4，
    因此诸如“../secret”或任意文件路径都无法通过。
    """

    if not isinstance(session_id, str):
        raise TypeError(
            "session_id必须是str"
        )

    try:
        return _SESSION_ID_ADAPTER.validate_python(
            session_id
        )
    except ValidationError as exc:
        raise ValueError(
            "session_id必须是小写UUID4"
        ) from exc


class DiagnosticSessionStore:
    """以单会话单JSON文件方式保存诊断记录。

    文件布局如下：

    data/diagnostic_sessions/
    ├── 11111111-1111-4111-8111-111111111111.json
    └── 22222222-2222-4222-8222-222222222222.json

    当前实现适合本地Beta、单进程服务和人工审计。

    如果未来使用多个Uvicorn Worker或多个服务实例，
    应将该实现替换成SQLite、PostgreSQL等具备
    跨进程事务能力的存储。
    """

    def __init__(
        self,
        *,
        root_directory: Path,
    ) -> None:
        """保存会话目录并创建进程内异步锁。

        构造函数不创建目录。

        这样仅仅导入FastAPI应用或建立依赖对象时，
        不会产生文件系统写入副作用。
        第一次save()时才真正创建目录。
        """

        if not isinstance(
            root_directory,
            Path,
        ):
            raise TypeError(
                "root_directory必须是Path"
            )

        # resolve(strict=False)生成规范化绝对路径，
        # 即使目录尚未创建也不会报错。
        resolved_directory = (
            root_directory.resolve(
                strict=False
            )
        )

        if (
            resolved_directory.exists()
            and not resolved_directory.is_dir()
        ):
            raise NotADirectoryError(
                "会话存储路径必须是目录"
            )

        self._root_directory = (
            resolved_directory
        )

        # asyncio.Lock保护当前Python进程中的
        # 并发保存、读取和列表操作。
        #
        # 它不能代替数据库的跨进程事务，
        # 但适合当前单进程Beta服务。
        self._lock = asyncio.Lock()

    @property
    def root_directory(self) -> Path:
        """返回规范化后的会话存储目录。

        property让调用方使用：

            store.root_directory

        而不是：

            store.root_directory()

        返回Path对象，不返回可变内部状态。
        """

        return self._root_directory

    def _path_for_session_id(
        self,
        session_id: str,
        /,
    ) -> Path:
        """把合法session_id转换成确定文件路径。

        该方法必须先调用_validate_session_id()，
        不能直接把外部字符串拼接到目录后面。
        """

        validated_session_id = (
            _validate_session_id(session_id)
        )

        return self._root_directory / (
            f"{validated_session_id}.json"
        )

    def _load_record_from_path(
        self,
        path: Path,
        /,
    ) -> DiagnosticSessionRecord:
        """读取并验证一个会话JSON文件。

        校验分为三层：

        1. 文件名必须是合法UUID4；
        2. JSON内容必须符合DiagnosticSessionRecord；
        3. JSON中的session_id必须等于文件名。
        """

        try:
            filename_session_id = (
                _validate_session_id(
                    path.stem
                )
            )
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise (
                DiagnosticSessionCorruptedError(
                    "会话文件名不是合法的"
                    "小写UUID4："
                    f"{path.name}"
                )
            ) from exc

        try:
            serialized_record = path.read_text(
                encoding="utf-8"
            )
        except OSError as exc:
            raise DiagnosticSessionStoreError(
                "读取诊断会话文件失败："
                f"{path.name}"
            ) from exc

        try:
            # model_validate_json()会同时：
            #
            # 1. 解析JSON文本；
            # 2. 建立DiagnosticSessionRecord；
            # 3. 执行字段校验器和跨字段校验器。
            record = (
                DiagnosticSessionRecord
                .model_validate_json(
                    serialized_record
                )
            )
        except (
            ValidationError,
            ValueError,
        ) as exc:
            raise (
                DiagnosticSessionCorruptedError(
                    "诊断会话文件内容无效："
                    f"{path.name}"
                )
            ) from exc

        if (
            record.session_id
            != filename_session_id
        ):
            raise (
                DiagnosticSessionCorruptedError(
                    "会话文件名与记录中的"
                    "session_id不一致："
                    f"{path.name}"
                )
            )

        return record

    def _save_sync(
        self,
        record: DiagnosticSessionRecord,
        /,
    ) -> DiagnosticSessionRecord:
        """在工作线程中执行同步文件保存。

        使用“临时文件 → replace”的流程，
        避免直接写目标文件时因进程中断而留下
        半个JSON文件。
        """

        try:
            self._root_directory.mkdir(
                parents=True,
                exist_ok=True,
            )
        except OSError as exc:
            raise DiagnosticSessionStoreError(
                "无法创建诊断会话存储目录"
            ) from exc

        destination_path = (
            self._path_for_session_id(
                record.session_id
            )
        )

        if destination_path.exists():
            raise (
                DiagnosticSessionAlreadyExistsError(
                    "诊断会话已经存在："
                    f"{record.session_id}"
                )
            )

        # 临时文件使用随机后缀，
        # 避免同一目录中的临时文件互相冲突。
        temporary_path = (
            self._root_directory
            / (
                f".{record.session_id}."
                f"{uuid4().hex}.tmp"
            )
        )

        try:
            # model_dump_json()使用Pydantic自己的
            # 序列化规则处理datetime、tuple和嵌套模型。
            serialized_record = (
                record.model_dump_json(
                    indent=2
                )
                + "\n"
            )

            temporary_path.write_text(
                serialized_record,
                encoding="utf-8",
                newline="\n",
            )

            # 在当前进程的asyncio.Lock保护下，
            # 再检查一次，避免同一进程内覆盖已有会话。
            if destination_path.exists():
                raise (
                    DiagnosticSessionAlreadyExistsError(
                        "诊断会话已经存在："
                        f"{record.session_id}"
                    )
                )

            # replace()在同一文件系统内把已经完整写好的
            # 临时文件发布为最终JSON文件。
            temporary_path.replace(
                destination_path
            )
        except (
            DiagnosticSessionAlreadyExistsError
        ):
            raise
        except OSError as exc:
            raise DiagnosticSessionStoreError(
                "保存诊断会话文件失败："
                f"{record.session_id}"
            ) from exc
        finally:
            # 如果写入或替换失败，尽量删除临时文件。
            #
            # suppress(OSError)表示清理失败不能覆盖
            # 原本更重要的存储异常。
            with suppress(OSError):
                temporary_path.unlink(
                    missing_ok=True
                )

        return record

    async def save(
        self,
        record: DiagnosticSessionRecord,
        /,
    ) -> DiagnosticSessionRecord:
        """异步保存一条会话记录。

        文件系统API本身是同步阻塞操作。
        asyncio.to_thread()把_sync方法放到工作线程，
        避免阻塞FastAPI所在的事件循环。
        """

        if not isinstance(
            record,
            DiagnosticSessionRecord,
        ):
            raise TypeError(
                "record必须是"
                "DiagnosticSessionRecord"
            )

        async with self._lock:
            return await asyncio.to_thread(
                self._save_sync,
                record,
            )

    def _get_sync(
        self,
        session_id: str,
        /,
    ) -> DiagnosticSessionRecord | None:
        """在工作线程中读取一条会话。"""

        path = self._path_for_session_id(
            session_id
        )

        if not path.exists():
            return None

        if not path.is_file():
            raise DiagnosticSessionStoreError(
                "诊断会话路径不是普通文件："
                f"{path.name}"
            )

        return self._load_record_from_path(
            path
        )

    async def get(
        self,
        session_id: str,
        /,
    ) -> DiagnosticSessionRecord | None:
        """根据session_id异步读取完整会话。

        返回值语义：

        - 找到：返回DiagnosticSessionRecord；
        - 不存在：返回None；
        - 文件损坏：抛出DiagnosticSessionCorruptedError；
        - 文件系统失败：抛出DiagnosticSessionStoreError。
        """

        validated_session_id = (
            _validate_session_id(
                session_id
            )
        )

        async with self._lock:
            return await asyncio.to_thread(
                self._get_sync,
                validated_session_id,
            )

    def _list_recent_sync(
        self,
        *,
        limit: int,
    ) -> DiagnosticSessionListResponse:
        """在工作线程中读取并排序最近会话。"""

        if not self._root_directory.exists():
            return DiagnosticSessionListResponse(
                sessions=(),
                total=0,
                limit=limit,
            )

        if not self._root_directory.is_dir():
            raise DiagnosticSessionStoreError(
                "诊断会话存储路径不是目录"
            )

        records: list[
            DiagnosticSessionRecord
        ] = []

        # glob("*.json")只读取当前目录下的JSON，
        # 不会递归进入其他目录，也不会读取.tmp文件。
        for path in (
            self._root_directory.glob(
                "*.json"
            )
        ):
            records.append(
                self._load_record_from_path(
                    path
                )
            )

        # 先按created_at从新到旧排序。
        #
        # session_id作为第二排序键，
        # 可保证两个记录时间完全相同时结果仍然稳定。
        records.sort(
            key=lambda record: (
                record.created_at,
                record.session_id,
            ),
            reverse=True,
        )

        total = len(records)

        # 列表接口只返回轻量摘要，
        # 不把全部工具轨迹和引用正文一次性发送出去。
        summaries = tuple(
            build_diagnostic_session_summary(
                record
            )
            for record in records[:limit]
        )

        return DiagnosticSessionListResponse(
            sessions=summaries,
            total=total,
            limit=limit,
        )

    async def list_recent(
        self,
        *,
        limit: int = 20,
    ) -> DiagnosticSessionListResponse:
        """异步查询最近的会话摘要。

        limit只允许1到100之间的整数。
        bool虽然是int的子类，但不能作为数量使用。
        """

        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
        ):
            raise TypeError(
                "limit必须是int"
            )

        if not (
            1
            <= limit
            <= MAX_RECENT_SESSION_LIMIT
        ):
            raise ValueError(
                "limit必须在1到100之间"
            )

        async with self._lock:
            return await asyncio.to_thread(
                self._list_recent_sync,
                limit=limit,
            )
