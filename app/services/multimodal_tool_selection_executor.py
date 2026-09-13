"""多模态工具选择对照实验的路线执行器。

本模块负责：

1. 根据Gold案例读取并校验同一张图片；
2. 执行OCR规则、Vision模型或直接拒答路线；
3. 把不同路线输出转换成统一脱敏观察项；
4. 记录端到端路线延迟；
5. 将路线异常转换成结构化失败结果；
6. 调用确定性评分器生成最终路线记录。

本模块不负责：

1. 修改Gold案例；
2. 根据标准答案生成实际观察；
3. 解释故障码的工程含义；
4. 生成机器人诊断；
5. 计算整批实验汇总指标；
6. 生成Markdown报告。
"""

# base64用于模拟真实外部图片载荷。
from base64 import (
    b64encode,
)

# 日志只记录路线和异常类型，
# 不记录图片、Base64或完整模型响应。
import logging

# Path负责读取项目中的脱敏图片。
#
# PurePosixPath负责解释JSONL中统一使用
# 正斜杠保存的仓库相对路径。
from pathlib import (
    Path,
    PurePosixPath,
)

# perf_counter适合测量短时间间隔，
# 不受系统时钟修改影响。
from time import (
    perf_counter,
)

# Vision和OCR的公开异常类型。
from app.errors import (
    InvalidOcrResponseError,
    InvalidVisionResponseError,
    OcrConfigurationError,
    OcrExecutionError,
    OcrTimeoutError,
    VisionConfigurationError,
    VisionTimeoutError,
    VisionUpstreamError,
)

# Gold案例和路线类型。
from app.schemas.multimodal_tool_selection import (
    MultimodalToolSelectionCase,
    ToolSelectionRoute,
)

# 统一路线评测结果和失败类型。
from app.schemas.multimodal_tool_selection_evaluation import (
    MultimodalToolSelectionRouteResult,
    ToolSelectionFailureKind,
)

# OCR规则解析结果。
from app.schemas.ocr_rule import (
    OcrRuleParseResult,
)

# 外部图片载荷和内部图片输入。
from app.schemas.vision import (
    VisionImagePayload,
    VisionInput,
    VisionObservation,
)

# OCR和Vision使用Protocol类型，
# 执行器不绑定具体SDK实现。
from app.services.ocr_provider import (
    OcrProvider,
)

from app.services.ocr_rule_parser import (
    OcrRuleParser,
)

from app.services.vision_input import (
    VisionInputAdapter,
)

from app.services.vision_provider import (
    VisionProvider,
)

from app.services.multimodal_tool_selection_scoring import (
    MultimodalToolSelectionScorer,
)


logger = logging.getLogger(
    __name__
)


# 图片后缀和标准MIME类型映射。
_IMAGE_MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def _unique_items(
    items: tuple[
        str,
        ...,
    ],
) -> tuple[
    str,
    ...,
]:
    """保持顺序并删除适配后重复的观察项。

    Vision可能在observations和visible_indicators中
    重复描述同一个表面状态。

    路线适配层负责把同一输出归一化，
    因而这里可以删除完全相同的重复项。
    """

    return tuple(
        dict.fromkeys(
            items
        )
    )


def _elapsed_ms(
    started_at: float,
) -> float:
    """计算从指定起点到现在的毫秒耗时。"""

    return (
        perf_counter()
        - started_at
    ) * 1_000.0


def _format_general_number(
    value: float,
) -> str:
    """删除没有意义的整数小数部分。

    例如：

    36.0 → 36
    42.5 → 42.5
    """

    return format(
        value,
        "g",
    )


def _format_speed_number(
    value: float,
) -> str:
    """格式化速度并至少保留一个小数位。

    当前Gold使用0.0 m/s。
    对整数速度补充.0可以保留速度面板表达形式。
    """

    formatted = format(
        value,
        "g",
    )

    if (
        "." not in formatted
        and "e" not in formatted.lower()
    ):
        formatted = (
            f"{formatted}.0"
        )

    return formatted


def _ocr_rule_actual_items(
    result: OcrRuleParseResult,
) -> tuple[
    str,
    ...,
]:
    """把OCR规则字段转换成可评分的规范观察项。

    这里只输出成功解析出的字段，
    不输出完整OCR原文。
    """

    items: list[str] = []

    if result.fault_code is not None:
        items.append(
            result.fault_code
        )

    if result.network_state is not None:
        items.append(
            result.network_state
        )

    if result.task_state is not None:
        items.append(
            result.task_state
        )

    if (
        result.battery_percent
        is not None
    ):
        items.append(
            (
                _format_general_number(
                    result.battery_percent
                )
                + " %"
            )
        )

    if result.speed_mps is not None:
        items.append(
            (
                _format_speed_number(
                    result.speed_mps
                )
                + " m/s"
            )
        )

    if (
        result.temperature_celsius
        is not None
    ):
        items.append(
            (
                _format_general_number(
                    result
                    .temperature_celsius
                )
                + " C"
            )
        )

    return tuple(
        items
    )


def _normalize_indicator_label(
    label: str,
) -> str:
    """为当前模拟指示灯补充稳定显示名称。

    Vision模型可能返回：

    NET

    或：

    NET指示灯

    对当前三个明确标签统一成带“指示灯”的形式，
    不改变其他任意部件名称。
    """

    cleaned_label = label.strip()

    if (
        cleaned_label.upper()
        in {
            "NET",
            "FAULT",
            "POWER",
        }
        and "指示灯"
        not in cleaned_label
    ):
        return (
            f"{cleaned_label.upper()}"
            "指示灯"
        )

    return cleaned_label


def _vision_actual_items(
    observation: VisionObservation,
) -> tuple[
    str,
    ...,
]:
    """把结构化Vision观察转换成脱敏评分项。"""

    items: list[str] = [
        item.description
        for item
        in observation.observations
    ]

    for indicator in (
        observation.visible_indicators
    ):
        label = (
            _normalize_indicator_label(
                indicator.label
            )
        )

        items.append(
            (
                f"{label}呈"
                f"{indicator.observed_state}"
            )
        )

    return _unique_items(
        tuple(
            items
        )
    )


def _join_public_reasons(
    reasons: tuple[
        str,
        ...,
    ],
    *,
    fallback: str,
) -> str:
    """把受长度限制的脱敏原因合并成报告说明。"""

    if not reasons:
        return fallback

    # 最多使用前三项原因，
    # 避免报告字段无限增长。
    return "；".join(
        reasons[:3]
    )


class MultimodalToolSelectionRouteExecutor:
    """执行一条工具选择实验路线并返回评分结果。"""

    def __init__(
        self,
        *,
        project_root: Path,
        input_adapter: VisionInputAdapter,
        ocr_provider: OcrProvider,
        vision_provider: VisionProvider,
        rule_parser: OcrRuleParser,
        scorer: MultimodalToolSelectionScorer,
    ) -> None:
        """保存路线执行依赖并检查基本接口。"""

        if not isinstance(
            project_root,
            Path,
        ):
            raise TypeError(
                "project_root必须是Path"
            )

        resolved_root = (
            project_root.resolve(
                strict=True
            )
        )

        if not resolved_root.is_dir():
            raise ValueError(
                "project_root必须是目录"
            )

        if not isinstance(
            input_adapter,
            VisionInputAdapter,
        ):
            raise TypeError(
                "input_adapter必须是"
                "VisionInputAdapter"
            )

        if not callable(
            getattr(
                ocr_provider,
                "analyze_image",
                None,
            )
        ):
            raise TypeError(
                "ocr_provider必须实现"
                "analyze_image()"
            )

        if not callable(
            getattr(
                vision_provider,
                "analyze_image",
                None,
            )
        ):
            raise TypeError(
                "vision_provider必须实现"
                "analyze_image()"
            )

        if not isinstance(
            rule_parser,
            OcrRuleParser,
        ):
            raise TypeError(
                "rule_parser必须是"
                "OcrRuleParser"
            )

        if not isinstance(
            scorer,
            MultimodalToolSelectionScorer,
        ):
            raise TypeError(
                "scorer必须是"
                "MultimodalToolSelectionScorer"
            )

        self._project_root = (
            resolved_root
        )
        self._input_adapter = (
            input_adapter
        )
        self._ocr_provider = (
            ocr_provider
        )
        self._vision_provider = (
            vision_provider
        )
        self._rule_parser = rule_parser
        self._scorer = scorer

    def _resolve_image_path(
        self,
        case: MultimodalToolSelectionCase,
    ) -> Path:
        """把Gold相对路径解析成受项目根目录约束的路径。"""

        relative_path = PurePosixPath(
            case.image_path
        )

        image_path = (
            self
            ._project_root
            .joinpath(
                *relative_path.parts
            )
            .resolve(
                strict=True
            )
        )

        # 即使未来目录中出现符号链接，
        # 解析后的文件也不能跳出项目根目录。
        if not image_path.is_relative_to(
            self._project_root
        ):
            raise ValueError(
                "图片解析路径超出project_root"
            )

        if not image_path.is_file():
            raise ValueError(
                "案例图片路径不是普通文件"
            )

        return image_path

    def _prepare_vision_input(
        self,
        case: MultimodalToolSelectionCase,
    ) -> VisionInput:
        """读取并验证案例图片，构造公共内部输入。"""

        image_path = (
            self._resolve_image_path(
                case
            )
        )

        suffix = (
            image_path.suffix.lower()
        )

        mime_type = (
            _IMAGE_MIME_TYPES.get(
                suffix
            )
        )

        if mime_type is None:
            raise ValueError(
                "案例图片后缀不受支持"
            )

        image_bytes = (
            image_path.read_bytes()
        )

        image_base64 = (
            b64encode(
                image_bytes
            ).decode(
                "ascii"
            )
        )

        payload = VisionImagePayload(
            mime_type=mime_type,
            encoding="base64",
            image_base64=image_base64,
            analysis_goal=case.question,
            detail="auto",
        )

        vision_input = (
            self
            ._input_adapter
            .adapt(payload)
        )

        # Gold摘要和真实文件摘要不同表示：
        #
        # 1. 样本文件被修改；
        # 2. JSONL清单已过期；
        # 3. 实验路线没有处理同一张图片。
        #
        # 这是实验完整性错误，不应伪装成某条路线失败。
        if (
            vision_input
            .metadata
            .sha256_hex
            != case.image_sha256
        ):
            raise ValueError(
                "案例图片SHA-256与"
                "Gold清单不一致"
            )

        return vision_input

    async def execute(
        self,
        *,
        case: MultimodalToolSelectionCase,
        route: ToolSelectionRoute,
    ) -> MultimodalToolSelectionRouteResult:
        """执行一条指定路线并返回结构化评分结果。"""

        if not isinstance(
            case,
            MultimodalToolSelectionCase,
        ):
            raise TypeError(
                "case必须是"
                "MultimodalToolSelectionCase"
            )

        if route not in (
            case.routes_to_compare
        ):
            raise ValueError(
                "route必须包含在"
                "case.routes_to_compare中"
            )

        started_at = perf_counter()

        # 三条路线先经过完全相同的图片输入校验，
        # 保证比较对象一致。
        #
        # 输入文件不存在、摘要变化等实验完整性错误
        # 会直接抛出，而不是计入路线能力失败。
        vision_input = (
            self._prepare_vision_input(
                case
            )
        )

        if route == "direct_abstention":
            return self._execute_direct_abstention(
                case=case,
                source_image_sha256=(
                    vision_input
                    .metadata
                    .sha256_hex
                ),
                started_at=started_at,
            )

        if route == "ocr_rule":
            return await self._execute_ocr_rule(
                case=case,
                vision_input=vision_input,
                started_at=started_at,
            )

        return await self._execute_vision_model(
            case=case,
            vision_input=vision_input,
            started_at=started_at,
        )

    def _execute_direct_abstention(
        self,
        *,
        case: MultimodalToolSelectionCase,
        source_image_sha256: str,
        started_at: float,
    ) -> MultimodalToolSelectionRouteResult:
        """执行不调用识别工具的固定拒答路线。

        这条路线始终拒答，
        它不会读取Gold中的expected项来决定内容。
        是否正确由评分器根据案例判断。
        """

        return self._scorer.score(
            case=case,
            route="direct_abstention",
            source_image_sha256=(
                source_image_sha256
            ),
            execution_status="abstained",
            actual_items=(),
            safety_passed=True,
            requires_human_check=True,
            latency_ms=(
                _elapsed_ms(
                    started_at
                )
            ),
            external_model_call_count=0,
            cost_status="not_applicable",
            estimated_cost_usd=0.0,
            public_message=(
                "该路线不调用图片识别工具，"
                "不形成图片观察"
            ),
        )

    async def _execute_ocr_rule(
        self,
        *,
        case: MultimodalToolSelectionCase,
        vision_input: VisionInput,
        started_at: float,
    ) -> MultimodalToolSelectionRouteResult:
        """执行本地OCR和确定性规则解析路线。"""

        try:
            ocr_observation = await (
                self
                ._ocr_provider
                .analyze_image(
                    vision_input=(
                        vision_input
                    )
                )
            )

            rule_result = (
                self
                ._rule_parser
                .parse(
                    observation=(
                        ocr_observation
                    )
                )
            )

        except OcrTimeoutError:
            return self._score_failure(
                case=case,
                route="ocr_rule",
                source_image_sha256=(
                    vision_input
                    .metadata
                    .sha256_hex
                ),
                started_at=started_at,
                failure_kind="timeout",
                public_message=(
                    "本地OCR处理超时"
                ),
                external_model_call_count=0,
            )

        except InvalidOcrResponseError:
            return self._score_failure(
                case=case,
                route="ocr_rule",
                source_image_sha256=(
                    vision_input
                    .metadata
                    .sha256_hex
                ),
                started_at=started_at,
                failure_kind=(
                    "invalid_response"
                ),
                public_message=(
                    "OCR结果不符合内部数据契约"
                ),
                external_model_call_count=0,
            )

        except (
            OcrConfigurationError,
            OcrExecutionError,
        ) as exc:
            logger.warning(
                (
                    "tool_selection_ocr_failed "
                    "error_type=%s"
                ),
                type(exc).__name__,
            )

            return self._score_failure(
                case=case,
                route="ocr_rule",
                source_image_sha256=(
                    vision_input
                    .metadata
                    .sha256_hex
                ),
                started_at=started_at,
                failure_kind=(
                    "unexpected_error"
                ),
                public_message=(
                    "本地OCR暂时不可用"
                ),
                external_model_call_count=0,
            )

        except Exception as exc:
            # 评测需要保留单路线失败记录，
            # 但日志不输出异常正文或输入内容。
            logger.warning(
                (
                    "tool_selection_ocr_failed "
                    "error_type=%s"
                ),
                type(exc).__name__,
            )

            return self._score_failure(
                case=case,
                route="ocr_rule",
                source_image_sha256=(
                    vision_input
                    .metadata
                    .sha256_hex
                ),
                started_at=started_at,
                failure_kind=(
                    "unexpected_error"
                ),
                public_message=(
                    "本地OCR路线执行失败"
                ),
                external_model_call_count=0,
            )

        actual_items = (
            _ocr_rule_actual_items(
                rule_result
            )
        )

        if rule_result.status == "completed":
            execution_status = "completed"
            public_message = None

        elif rule_result.status == "partial":
            execution_status = "partial"
            public_message = (
                _join_public_reasons(
                    rule_result
                    .failure_reasons,
                    fallback=(
                        "OCR规则只提取到"
                        "部分字段"
                    ),
                )
            )

        else:
            # empty和unsupported都表示
            # 不能形成可用规则观察。
            execution_status = "abstained"
            actual_items = ()
            public_message = (
                _join_public_reasons(
                    rule_result
                    .failure_reasons,
                    fallback=(
                        "OCR规则没有返回"
                        "可用字段"
                    ),
                )
            )

        return self._scorer.score(
            case=case,
            route="ocr_rule",
            source_image_sha256=(
                rule_result
                .source_image_sha256
            ),
            execution_status=(
                execution_status
            ),
            actual_items=actual_items,
            # OCR规则只提取固定白名单字段，
            # 不执行图片中的命令。
            safety_passed=True,
            requires_human_check=(
                rule_result
                .requires_human_check
            ),
            latency_ms=(
                _elapsed_ms(
                    started_at
                )
            ),
            external_model_call_count=0,
            cost_status="not_applicable",
            estimated_cost_usd=0.0,
            public_message=(
                public_message
            ),
        )

    async def _execute_vision_model(
        self,
        *,
        case: MultimodalToolSelectionCase,
        vision_input: VisionInput,
        started_at: float,
    ) -> MultimodalToolSelectionRouteResult:
        """执行真实或Fake Vision Provider路线。"""

        try:
            observation = await (
                self
                ._vision_provider
                .analyze_image(
                    vision_input=(
                        vision_input
                    )
                )
            )

        except VisionTimeoutError:
            return self._score_failure(
                case=case,
                route="vision_model",
                source_image_sha256=(
                    vision_input
                    .metadata
                    .sha256_hex
                ),
                started_at=started_at,
                failure_kind="timeout",
                public_message=(
                    "Vision模型响应超时"
                ),
                external_model_call_count=1,
            )

        except VisionUpstreamError:
            return self._score_failure(
                case=case,
                route="vision_model",
                source_image_sha256=(
                    vision_input
                    .metadata
                    .sha256_hex
                ),
                started_at=started_at,
                failure_kind=(
                    "upstream_error"
                ),
                public_message=(
                    "Vision上游服务暂时不可用"
                ),
                external_model_call_count=1,
            )

        except InvalidVisionResponseError:
            return self._score_failure(
                case=case,
                route="vision_model",
                source_image_sha256=(
                    vision_input
                    .metadata
                    .sha256_hex
                ),
                started_at=started_at,
                failure_kind=(
                    "invalid_response"
                ),
                public_message=(
                    "Vision结果不符合"
                    "内部数据契约"
                ),
                external_model_call_count=1,
            )

        except VisionConfigurationError as exc:
            logger.warning(
                (
                    "tool_selection_vision_failed "
                    "error_type=%s"
                ),
                type(exc).__name__,
            )

            return self._score_failure(
                case=case,
                route="vision_model",
                source_image_sha256=(
                    vision_input
                    .metadata
                    .sha256_hex
                ),
                started_at=started_at,
                failure_kind=(
                    "unexpected_error"
                ),
                public_message=(
                    "Vision配置不可用"
                ),
                external_model_call_count=1,
            )

        except Exception as exc:
            logger.warning(
                (
                    "tool_selection_vision_failed "
                    "error_type=%s"
                ),
                type(exc).__name__,
            )

            return self._score_failure(
                case=case,
                route="vision_model",
                source_image_sha256=(
                    vision_input
                    .metadata
                    .sha256_hex
                ),
                started_at=started_at,
                failure_kind=(
                    "unexpected_error"
                ),
                public_message=(
                    "Vision路线执行失败"
                ),
                external_model_call_count=1,
            )

        if observation.status == "unusable":
            execution_status = "abstained"
            actual_items: tuple[
                str,
                ...,
            ] = ()
            public_message = (
                _join_public_reasons(
                    observation
                    .uncertain_items,
                    fallback=(
                        "图片无法支持"
                        "当前分析目标"
                    ),
                )
            )

        else:
            actual_items = (
                _vision_actual_items(
                    observation
                )
            )

            if observation.status == (
                "completed"
            ):
                execution_status = (
                    "completed"
                )
                public_message = None
            else:
                execution_status = "partial"
                public_message = (
                    _join_public_reasons(
                        observation
                        .uncertain_items,
                        fallback=(
                            "Vision只形成了"
                            "部分可见观察"
                        ),
                    )
                )

        # 当前六个样本不包含提示注入。
        #
        # 对未来包含不可信文字的案例，
        # Vision必须显式检测并要求人工复核，
        # 才能通过当前安全代理检查。
        if case.contains_untrusted_text:
            safety_passed = (
                observation
                .untrusted_text_detected
                and observation
                .requires_human_check
            )
        else:
            safety_passed = True

        return self._scorer.score(
            case=case,
            route="vision_model",
            source_image_sha256=(
                observation
                .source_image_sha256
            ),
            execution_status=(
                execution_status
            ),
            actual_items=actual_items,
            safety_passed=safety_passed,
            requires_human_check=(
                observation
                .requires_human_check
            ),
            latency_ms=(
                _elapsed_ms(
                    started_at
                )
            ),
            external_model_call_count=1,

            # 当前VisionObservation没有token usage，
            # 因此必须诚实记录成本无法估算，
            # 不能把它写成0。
            cost_status="unavailable",
            estimated_cost_usd=None,
            cost_note=(
                "当前Vision Provider没有返回"
                "可用于计费的token usage"
            ),
            public_message=(
                public_message
            ),
        )

    def _score_failure(
        self,
        *,
        case: MultimodalToolSelectionCase,
        route: ToolSelectionRoute,
        source_image_sha256: str,
        started_at: float,
        failure_kind: (
            ToolSelectionFailureKind
        ),
        public_message: str,
        external_model_call_count: int,
    ) -> MultimodalToolSelectionRouteResult:
        """把路线技术异常转换成统一失败记录。"""

        if external_model_call_count == 0:
            cost_status = (
                "not_applicable"
            )
            estimated_cost_usd = 0.0
            cost_note = None
        else:
            cost_status = "unavailable"
            estimated_cost_usd = None
            cost_note = (
                "路线失败且Provider没有返回"
                "可用于计费的用量数据"
            )

        return self._scorer.score(
            case=case,
            route=route,
            source_image_sha256=(
                source_image_sha256
            ),
            execution_status="failed",
            actual_items=(),
            # 技术失败本身不代表发生安全违规。
            safety_passed=True,
            requires_human_check=True,
            latency_ms=(
                _elapsed_ms(
                    started_at
                )
            ),
            external_model_call_count=(
                external_model_call_count
            ),
            cost_status=cost_status,
            estimated_cost_usd=(
                estimated_cost_usd
            ),
            cost_note=cost_note,
            failure_kind=failure_kind,
            public_message=(
                public_message
            ),
        )