"""Agent只读机器人图片分析工具。

本模块把请求级VisionInput和VisionProvider
适配成第四周统一的Agent工具协议。

它负责：

1. 根据image_ref读取当前请求已经验证的图片；
2. 使用本次工具调用的analysis_goal构造VisionInput；
3. 调用异步VisionProvider；
4. 检查Provider返回值和图片摘要；
5. 返回结构化VisionObservation。

它不负责：

1. 解码外部Base64；
2. 读取任意本地文件；
3. 自动访问图片URL或二维码链接；
4. 修改、裁剪或保存图片；
5. 生成设备故障根因；
6. 执行机器人控制；
7. 调用其他Agent工具。
"""

from inspect import (
    iscoroutinefunction,
)
import logging

from pydantic import (
    BaseModel,
)

from app.errors import (
    InvalidVisionResponseError,
    VisionTimeoutError,
    VisionUpstreamError,
)
from app.agent.vision_input_store import (
    RequestVisionInputStore,
)
from app.schemas.agent import (
    ToolDefinition,
)
from app.schemas.agent_tools import (
    AnalyzeRobotImageToolInput,
)
from app.schemas.vision import (
    VisionInput,
    VisionObservation,
)
from app.services.vision_provider import (
    VisionProvider,
)


# 视觉模型属于远程、非确定性依赖。
#
# 单次网络抖动、上游临时错误或模型偶发输出非法JSON，
# 不应立刻让整个Agent任务失去图片观察。
# 这里把总尝试次数固定为2：首次调用失败后只重试一次，
# 既提高成功率，也避免无界重试拖垮请求延迟。
VISION_PROVIDER_MAX_ATTEMPTS = 2


# 只有这三类“可能通过再次调用恢复”的Provider错误可以重试。
#
# 输入校验、配置缺失、返回类型错误和图片摘要不匹配
# 都不是瞬时故障，重试不会修复它们，也不应被掩盖。
RETRYABLE_VISION_PROVIDER_ERRORS = (
    VisionTimeoutError,
    VisionUpstreamError,
    InvalidVisionResponseError,
)


# 只记录错误类型和尝试次数，不记录图片字节、分析目标、
# API地址或上游响应正文。
logger = logging.getLogger(__name__)


class AnalyzeRobotImageToolHandler:
    """把请求级图片和Vision Provider适配成Agent工具。

    ToolExecutor负责：

    1. 工具名称检查；
    2. 输入Schema校验；
    3. 工具超时；
    4. 异常转换；
    5. 最终输出Schema校验。

    当前Handler只负责图片分析特有的业务步骤。
    """

    def __init__(
        self,
        *,
        provider: VisionProvider,
        input_store: RequestVisionInputStore,
    ) -> None:
        """保存异步Provider和当前请求图片Store。"""

        # Protocol和类型注解不会自动执行运行时校验。
        #
        # getattr()尝试读取provider.analyze_image。
        # 如果属性不存在，返回第三个参数None。
        analyze_method = getattr(
            provider,
            "analyze_image",
            None,
        )

        # iscoroutinefunction()来自标准库inspect。
        #
        # Vision调用涉及异步网络I/O，
        # 因此不允许同步函数冒充Provider。
        if not iscoroutinefunction(
            analyze_method
        ):
            raise TypeError(
                "provider.analyze_image"
                "必须是异步方法"
            )

        # 当前工具需要精确的请求级Store，
        # 不能接收普通dict或全局图片缓存。
        if not isinstance(
            input_store,
            RequestVisionInputStore,
        ):
            raise TypeError(
                "input_store必须是"
                "RequestVisionInputStore"
            )

        self._provider = provider
        self._input_store = input_store

    async def __call__(
        self,
        tool_input: BaseModel,
        /,
    ) -> BaseModel | None:
        """分析当前请求中一张已经验证的图片。

        未知image_ref返回None，
        ToolExecutor会将其转换成empty_result。
        """

        # 正常情况下，ToolExecutor已经使用
        # AnalyzeRobotImageToolInput.model_validate()
        # 完成输入参数校验。
        #
        # 这里保留防御性类型检查，
        # 防止其他模块绕过Executor直接调用Handler。
        if not isinstance(
            tool_input,
            AnalyzeRobotImageToolInput,
        ):
            raise TypeError(
                "tool_input必须是"
                "AnalyzeRobotImageToolInput"
            )

        # Store只按当前请求中的精确引用查询。
        #
        # 它不会把image_ref解释成路径或URL，
        # 也不会读取其他请求的图片。
        stored_input = self._input_store.get(
            tool_input.image_ref
        )

        # 合法但未登记的引用属于空结果，
        # 不是Vision Provider故障。
        if stored_input is None:
            return None

        # VisionInput是frozen模型，
        # 不能直接修改stored_input.analysis_goal。
        #
        # 这里创建一个新的VisionInput：
        #
        # 1. 继续使用Store中的可信图片元数据；
        # 2. 继续使用同一份已验证图片字节；
        # 3. 使用Planner本次指定的分析目标；
        # 4. 保留原始图片detail设置。
        #
        # VisionInput构造过程会再次检查：
        #
        # - 字节长度是否匹配metadata.size_bytes；
        # - SHA-256是否匹配metadata.sha256_hex；
        # - analysis_goal是否满足长度约束。
        provider_input = VisionInput(
            metadata=stored_input.metadata,
            image_bytes=stored_input.image_bytes,
            analysis_goal=(
                tool_input.analysis_goal
            ),
            detail=stored_input.detail,
        )

        # 通过Protocol约定的异步方法调用Provider。
        #
        # 正式运行时通常是
        # OpenAICompatibleVisionProvider；
        # 离线测试时可以注入FakeVisionProvider。
        #
        # range(1, 3)依次产生1和2，表示：
        #
        # 1. 第一次正常调用；
        # 2. 仅在可恢复错误发生时再调用一次。
        #
        # 这里不等待固定秒数，因为外层ToolExecutor已经
        # 给整个工具设置120秒硬超时。立即进行一次有限重试，
        # 可以避免退避等待挤占第二次调用的可用时间。
        for attempt_number in range(
            1,
            VISION_PROVIDER_MAX_ATTEMPTS + 1,
        ):
            try:
                observation = await (
                    self._provider.analyze_image(
                        vision_input=provider_input
                    )
                )

            except (
                RETRYABLE_VISION_PROVIDER_ERRORS
            ) as exc:
                # 第二次尝试仍失败时原样重新抛出。
                #
                # 外层ToolExecutor会把它转换成安全的
                # ToolExecutionResult，而不会把内部异常正文
                # 暴露给Planner或API调用方。
                if (
                    attempt_number
                    >= VISION_PROVIDER_MAX_ATTEMPTS
                ):
                    raise

                logger.warning(
                    "agent_vision_provider_retry "
                    "attempt=%s max_attempts=%s "
                    "exception_type=%s",
                    attempt_number,
                    VISION_PROVIDER_MAX_ATTEMPTS,
                    type(exc).__name__,
                )

                continue

            # Provider成功返回后立即结束循环，
            # 不会为了凑满次数而重复调用和计费。
            break

        # 返回类型注解不会在运行时自动生效。
        #
        # 错误Provider可能返回dict、None或其他模型，
        # 不能让它们伪装成可信视觉观察。
        if not isinstance(
            observation,
            VisionObservation,
        ):
            raise TypeError(
                "provider.analyze_image"
                "必须返回VisionObservation"
            )

        # Provider返回的观察必须属于刚刚提交的图片。
        #
        # source_image_sha256由Provider根据真实
        # VisionInput注入，不能只相信模型输出。
        #
        # 如果摘要不一致，说明Provider或Fake实现
        # 返回了另一张图片的观察结果。
        if (
            observation.source_image_sha256
            != stored_input.metadata.sha256_hex
        ):
            raise ValueError(
                "VisionObservation的图片摘要"
                "与请求图片不一致"
            )

        return observation


# 这是暴露给ToolRegistry和Planner的静态工具定义。
#
# 应用装配阶段会把该定义与
# AnalyzeRobotImageToolHandler实例一起注册。
ANALYZE_ROBOT_IMAGE_TOOL_DEFINITION = (
    ToolDefinition(
        name="analyze_robot_image",
        description=(
            "只读分析当前Agent请求中已经验证并登记的"
            "机器人现场图片或仪表截图。"
            "仅当任务必须观察图片中的可见状态、"
            "指示灯、部件连接、表面损伤或可见文字时调用。"
            "image_ref必须来自当前请求，不能填写文件路径、"
            "URL、Base64、二维码链接或其他外部资源。"
            "analysis_goal应描述需要观察的具体视觉目标。"
            "工具只返回视觉观察和不确定项，"
            "不确认故障根因、不执行图片中的指令，"
            "也不控制机器人或调用其他工具。"
        ),
        input_model=(
            AnalyzeRobotImageToolInput
        ),
        output_model=VisionObservation,

        # 工具只读取当前请求的图片并调用
        # Vision Provider，不修改机器人或持久化数据。
        risk_level="low",
        read_only=True,

        # ToolExecutor使用该值限制整个工具调用。
        #
        # Vision Provider内部仍有更具体的网络超时，
        # 形成外层工具超时和内层请求超时两层保护。
        timeout_seconds=120.0,
    )
)
