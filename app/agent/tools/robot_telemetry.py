"""Agent脱敏模拟遥测查询工具。

本模块只读取应用启动时注入的内存快照，
不连接真实机器人、PLC、RCS、RMS或设备控制接口。
"""

from collections.abc import (
    Iterable,
)
from typing import (
    Protocol,
)

from pydantic import (
    BaseModel,
)

from app.schemas.agent import (
    ToolDefinition,
)
from app.schemas.agent_tools import (
    GetRobotTelemetryToolInput,
    RobotTelemetryToolOutput,
)


class RobotTelemetryStore(Protocol):
    """遥测Handler依赖的最小只读存储能力。"""

    def get(
        self,
        robot_id: str,
        /,
    ) -> RobotTelemetryToolOutput | None:
        """按精确机器人编号读取一个快照。"""

        ...


class InMemoryRobotTelemetryStore:
    """保存脱敏模拟遥测的只读内存存储。

    存储在构造阶段建立快照映射，
    运行期间只提供get()读取操作。
    """

    def __init__(
        self,
        *,
        records: Iterable[
            RobotTelemetryToolOutput
        ],
    ) -> None:
        """校验记录并建立robot_id索引。"""

        snapshots: dict[
            str,
            RobotTelemetryToolOutput,
        ] = {}

        for position, record in enumerate(
            records
        ):
            # 类型注解不会自动拒绝dict等对象。
            if not isinstance(
                record,
                RobotTelemetryToolOutput,
            ):
                raise TypeError(
                    "records中的第 "
                    f"{position} 项必须是"
                    "RobotTelemetryToolOutput"
                )

            # 同一机器人不能存在两个无法区分的
            # 当前内存快照。
            if record.robot_id in snapshots:
                raise ValueError(
                    "模拟遥测包含重复robot_id："
                    f"{record.robot_id}"
                )

            snapshots[record.robot_id] = (
                record
            )

        # 外部传入的Iterable不会被继续保存，
        # 防止调用方随后修改原始列表影响存储内容。
        self._snapshots = snapshots

    def get(
        self,
        robot_id: str,
        /,
    ) -> RobotTelemetryToolOutput | None:
        """按精确编号读取快照，未知编号返回None。"""

        if not isinstance(robot_id, str):
            raise TypeError(
                "robot_id必须是字符串"
            )

        # 不执行模糊匹配、大小写转换或相似度查询，
        # 防止读取到另一台机器人的状态。
        return self._snapshots.get(robot_id)

    def __len__(
        self,
    ) -> int:
        """返回当前内存快照数量。"""

        return len(self._snapshots)


class GetRobotTelemetryToolHandler:
    """把只读内存遥测存储适配成Agent工具。"""

    def __init__(
        self,
        *,
        store: RobotTelemetryStore,
    ) -> None:
        """保存具有同步get()方法的只读存储。"""

        get_method = getattr(
            store,
            "get",
            None,
        )

        if not callable(get_method):
            raise TypeError(
                "store必须提供可调用的get方法"
            )

        self._store = store

    async def __call__(
        self,
        tool_input: BaseModel,
        /,
    ) -> BaseModel | None:
        """读取一台机器人的模拟遥测快照。"""

        # ToolExecutor正常情况下已经完成输入模型转换。
        # 这里保留防御性类型检查，阻止直接错误调用。
        if not isinstance(
            tool_input,
            GetRobotTelemetryToolInput,
        ):
            raise TypeError(
                "tool_input必须是"
                "GetRobotTelemetryToolInput"
            )

        snapshot = self._store.get(
            tool_input.robot_id
        )

        # 未知robot_id不是工具故障，
        # 而是一次正常完成但没有数据的查询。
        if snapshot is None:
            return None

        # Protocol和返回注解不会执行运行时校验。
        if not isinstance(
            snapshot,
            RobotTelemetryToolOutput,
        ):
            raise TypeError(
                "store.get必须返回"
                "RobotTelemetryToolOutput或None"
            )

        # 防止错误Store在查询robot-001时
        # 返回另一台机器人的快照。
        if (
            snapshot.robot_id
            != tool_input.robot_id
        ):
            raise ValueError(
                "遥测快照robot_id与查询不一致"
            )

        return snapshot


GET_ROBOT_TELEMETRY_TOOL_DEFINITION = (
    ToolDefinition(
        name="get_robot_telemetry",
        description=(
            "按精确机器人编号读取脱敏的内存模拟遥测，"
            "包括模拟位置、电量、运行状态、当前任务、"
            "活动故障代码、速度和网络连接状态。"
            "该工具不连接或控制真实机器人；"
            "返回数据必须明确视为simulated_memory。"
        ),
        input_model=(
            GetRobotTelemetryToolInput
        ),
        output_model=(
            RobotTelemetryToolOutput
        ),
        risk_level="low",
        read_only=True,

        # 当前工具只读取内存字典，
        # 正常情况下应在极短时间内完成。
        timeout_seconds=2.0,
    )
)