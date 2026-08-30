"""Week 4 Agent使用的脱敏教学模拟遥测。

本模块中的编号、位置、任务和数值全部是教学模拟数据，
不对应任何真实客户、仓库、机器人或生产系统。

本模块只负责创建内存快照，
不连接机器人、RCS、RMS、PLC或外部数据库。
"""

from datetime import (
    datetime,
)

from app.agent.tools.robot_telemetry import (
    InMemoryRobotTelemetryStore,
)
from app.schemas.agent_tools import (
    RobotTelemetryToolOutput,
)


def build_demo_robot_telemetry_records(
    *,
    observed_at: datetime,
) -> tuple[
    RobotTelemetryToolOutput,
    ...,
]:
    """创建一组具有相同观测时间的教学模拟快照。

    observed_at由应用装配层提供，
    使测试可以注入固定时间，
    正式运行时也可以记录快照创建时刻。
    """

    if not isinstance(observed_at, datetime):
        raise TypeError(
            "observed_at必须是datetime"
        )

    # robot-001模拟：
    #
    # 网络已经恢复，但旧任务仍处于暂停状态。
    # Agent可以先读取该快照，再检索ERR-NET-4001规程，
    # 构成两步诊断链。
    robot_001 = RobotTelemetryToolOutput(
        robot_id="robot-001",
        observed_at=observed_at,
        location=(
            "warehouse-demo/"
            "zone-a/aisle-07/node-14"
        ),
        battery_percent=42.5,
        operational_state="paused",
        current_task_id="task-demo-1001",
        active_fault_codes=(
            "ERR-NET-4001",
        ),
        speed_mps=0.0,

        # 这里表示当前模拟网络已经恢复，
        # 不代表旧任务可以自动继续。
        network_connected=True,
    )

    # robot-002模拟：
    #
    # 机器人正在充电，没有活动故障和当前运输任务。
    robot_002 = RobotTelemetryToolOutput(
        robot_id="robot-002",
        observed_at=observed_at,
        location=(
            "warehouse-demo/"
            "charging-zone/station-02"
        ),
        battery_percent=18.0,
        operational_state="charging",
        current_task_id=None,
        active_fault_codes=(),
        speed_mps=0.0,
        network_connected=True,
    )

    # robot-003模拟：
    #
    # 正常执行脱敏教学任务，
    # 可用于验证Agent不会在无故障时虚构异常。
    robot_003 = RobotTelemetryToolOutput(
        robot_id="robot-003",
        observed_at=observed_at,
        location=(
            "warehouse-demo/"
            "zone-b/aisle-03/node-08"
        ),
        battery_percent=78.0,
        operational_state="executing_task",
        current_task_id="task-demo-3001",
        active_fault_codes=(),
        speed_mps=0.8,
        network_connected=True,
    )

    return (
        robot_001,
        robot_002,
        robot_003,
    )


def build_demo_robot_telemetry_store(
    *,
    observed_at: datetime,
) -> InMemoryRobotTelemetryStore:
    """创建装有教学模拟快照的只读内存Store。"""

    records = (
        build_demo_robot_telemetry_records(
            observed_at=observed_at
        )
    )

    return InMemoryRobotTelemetryStore(
        records=records
    )