"""Agent读取应用服务器当前UTC时间的只读工具。

本模块只读取系统时钟：

1. 不读取机器人遥测；
2. 不访问网络；
3. 不修改文件或数据库；
4. 不执行Shell命令；
5. 不把LLM提供的日期当成当前时间。
"""

from datetime import (
    datetime,
    timezone,
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
    GetCurrentTimeToolInput,
    GetCurrentTimeToolOutput,
)


class UtcClock(Protocol):
    """时间工具依赖的最小系统时钟协议。

    正式运行时由SystemUtcClock满足；
    自动化测试时可以使用返回固定时间的FakeClock。
    """

    def now_utc(
        self,
    ) -> datetime:
        """返回当前带时区的UTC时间。"""

        ...


class SystemUtcClock:
    """使用Python标准库读取真实系统UTC时间。"""

    def now_utc(
        self,
    ) -> datetime:
        """读取调用发生这一刻的UTC系统时间。"""

        # datetime.now(timezone.utc)返回带UTC时区的datetime。
        #
        # 不使用datetime.utcnow()，因为utcnow()返回的是
        # 没有tzinfo的naive datetime，容易被误当成本地时间。
        return datetime.now(
            timezone.utc
        )


class GetCurrentTimeToolHandler:
    """把UTC系统时钟适配成Agent异步工具Handler。"""

    def __init__(
        self,
        *,
        clock: UtcClock,
    ) -> None:
        """保存可以被Fake替换的系统时钟依赖。"""

        now_utc_method = getattr(
            clock,
            "now_utc",
            None,
        )

        # Protocol类型注解不会在Python运行时
        # 自动拒绝错误对象，所以需要显式检查方法。
        if not callable(now_utc_method):
            raise TypeError(
                "clock必须提供可调用的"
                "now_utc方法"
            )

        self._clock = clock

    async def __call__(
        self,
        tool_input: BaseModel,
        /,
    ) -> BaseModel:
        """读取并返回当前UTC系统时间。"""

        # ToolExecutor通常已经通过input_model
        # 将空arguments校验成GetCurrentTimeToolInput。
        #
        # 这里保留防御性检查，防止其他模块
        # 绕过Executor直接传入错误模型。
        if not isinstance(
            tool_input,
            GetCurrentTimeToolInput,
        ):
            raise TypeError(
                "tool_input必须是"
                "GetCurrentTimeToolInput"
            )

        current_time = (
            self._clock.now_utc()
        )

        # Python类型注解不会自动检查方法真实返回值。
        # Fake或错误实现仍可能返回字符串。
        if not isinstance(
            current_time,
            datetime,
        ):
            raise TypeError(
                "clock.now_utc必须返回datetime"
            )

        # GetCurrentTimeToolOutput会继续检查：
        #
        # 1. 时间是否具有时区；
        # 2. UTC偏移是否为零；
        # 3. source和timezone是否为固定值。
        return GetCurrentTimeToolOutput(
            current_time=current_time
        )


# get_current_time工具的静态定义。
GET_CURRENT_TIME_TOOL_DEFINITION = (
    ToolDefinition(
        name="get_current_time",
        description=(
            "读取应用服务器系统时钟并返回当前UTC时间。"
            "适用于用户问题中出现现在、今天、当前日期"
            "或需要建立明确时间参照的场景；"
            "不代表机器人遥测采样时间，"
            "也不用于查询知识库文档的发布日期。"
        ),
        input_model=GetCurrentTimeToolInput,
        output_model=GetCurrentTimeToolOutput,

        # 读取系统时钟不会修改任何外部状态。
        risk_level="low",
        read_only=True,

        # 当前操作只读取内存中的系统时钟，
        # 正常情况下应立即完成。
        timeout_seconds=1.0,
    )
)