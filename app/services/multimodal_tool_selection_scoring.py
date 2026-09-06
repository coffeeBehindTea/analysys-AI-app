"""多模态工具选择单路线确定性评分服务。

本模块接收：

1. 一个MultimodalToolSelectionCase Gold案例；
2. 一条路线的实际执行状态；
3. 脱敏实际观察项；
4. 延迟、成本和安全检查结果。

然后自动计算：

1. 标准项命中和缺失；
2. 内容准确率；
3. 路线是否适合案例；
4. 实际结果是否正确；
5. 最终是否通过。

本模块不负责：

1. 读取图片；
2. 执行OCR；
3. 调用Vision模型；
4. 使用LLM判断语义相似性；
5. 计算整批实验指标；
6. 生成Markdown报告。
"""

# isclose用于安全比较浮点准确率。
from math import (
    isclose,
)

# re用于清理只影响表达形式的标点和空白。
import re

# unicodedata用于统一全角字符等Unicode形式。
import unicodedata

# Gold案例和路线类型。
from app.schemas.multimodal_tool_selection import (
    MultimodalToolSelectionCase,
    ToolSelectionRoute,
)

# 单路线评测结果及其公共类型。
from app.schemas.multimodal_tool_selection_evaluation import (
    MultimodalToolSelectionRouteResult,
    ToolSelectionCostStatus,
    ToolSelectionFailureKind,
    ToolSelectionRouteExecutionStatus,
)


def _unique_items(
    items: tuple[
        str,
        ...,
    ],
) -> tuple[
    str,
    ...,
]:
    """保持原顺序并删除重复项目。"""

    return tuple(
        dict.fromkeys(
            items
        )
    )


def _clean_actual_items(
    actual_items: tuple[
        str,
        ...,
    ],
) -> tuple[
    str,
    ...,
]:
    """检查并清理路线返回的脱敏观察项。

    本函数不会替路线补写观察，
    只清除普通字符串两端空白。
    """

    if not isinstance(
        actual_items,
        tuple,
    ):
        raise TypeError(
            "actual_items必须是tuple"
        )

    cleaned_items: list[str] = []

    for item in actual_items:
        if not isinstance(
            item,
            str,
        ):
            raise TypeError(
                "actual_items中的项目必须是str"
            )

        cleaned_item = item.strip()

        if not cleaned_item:
            raise ValueError(
                "actual_items不能包含空字符串"
            )

        cleaned_items.append(
            cleaned_item
        )

    # 这里不静默删除重复观察。
    #
    # 重复项会交给结果Schema拒绝，
    # 以便暴露路线输出或适配逻辑中的问题。
    return tuple(
        cleaned_items
    )


def _normalize_matching_text(
    value: str,
) -> str:
    """把文字转换成确定性匹配形式。

    这里只消除大小写、空白、全半角和少量
    不影响当前标准项含义的表达形式。

    本函数不会：

    1. 翻译中英文；
    2. 使用同义词模型；
    3. 猜测OCR错误；
    4. 把“可能”改成“确定”。
    """

    normalized = unicodedata.normalize(
        "NFKC",
        value,
    )

    # casefold()比lower()更适合通用Unicode文字，
    # 英文标签会统一成小写。
    normalized = normalized.casefold()

    # 统一可能出现在故障码中的横线。
    for dash_character in (
        "‐",
        "‑",
        "‒",
        "–",
        "—",
        "−",
    ):
        normalized = normalized.replace(
            dash_character,
            "-",
        )

    # 统一摄氏温度的常见写法：
    #
    # 36 C
    # 36°C
    # 36℃
    #
    # 都转换成包含普通c的形式。
    normalized = normalized.replace(
        "℃",
        "c",
    )

    normalized = normalized.replace(
        "°c",
        "c",
    )

    # “呈绿色”和“为绿色”中的呈、为
    # 在当前指示灯Gold语句中只承担连接作用。
    #
    # 删除后：
    #
    # NET指示灯呈绿色亮起
    # NET指示灯为绿色亮起
    #
    # 可以使用同一确定性形式比较。
    normalized = normalized.replace(
        "呈",
        "",
    )

    normalized = normalized.replace(
        "为",
        "",
    )

    # 删除空白和普通展示标点。
    #
    # 不删除：
    #
    # - 故障码中的短横线；
    # - 速度单位中的斜线；
    # - 电量单位中的百分号。
    normalized = re.sub(
        r"""[
            \s
            :：,
            ，;
            ；。
            .!
            ！?
            ？
            （）
            ()
            \[\]
            【】
            "'`
        ]+""",
        "",
        normalized,
        flags=re.VERBOSE,
    )

    return normalized


def _expected_item_is_matched(
    *,
    expected_item: str,
    actual_items: tuple[
        str,
        ...,
    ],
) -> bool:
    """判断一个Gold项是否被实际观察覆盖。

    使用“规范化Gold字符串包含在规范化实际观察中”
    作为保守、可重复的判定标准。

    它不会使用LLM做语义猜测。
    """

    normalized_expected = (
        _normalize_matching_text(
            expected_item
        )
    )

    return any(
        normalized_expected
        in _normalize_matching_text(
            actual_item
        )
        for actual_item in actual_items
    )


def _collect_expected_items(
    case: MultimodalToolSelectionCase,
) -> tuple[
    str,
    ...,
]:
    """按稳定顺序合并文字和视觉Gold项。"""

    return _unique_items(
        (
            case.expected_text_terms
            + case
            .expected_visual_observations
        )
    )


class MultimodalToolSelectionScorer:
    """把路线执行信息转换成可信评分记录。

    本类没有可变状态，
    同一个实例可以连续评分多个案例。
    """

    def score(
        self,
        *,
        case: MultimodalToolSelectionCase,
        route: ToolSelectionRoute,
        source_image_sha256: str,
        execution_status: (
            ToolSelectionRouteExecutionStatus
        ),
        actual_items: tuple[
            str,
            ...,
        ],
        safety_passed: bool,
        requires_human_check: bool,
        latency_ms: float,
        external_model_call_count: int,
        cost_status: ToolSelectionCostStatus,
        estimated_cost_usd: (
            float
            | None
        ),
        cost_note: str | None = None,
        failure_kind: (
            ToolSelectionFailureKind
            | None
        ) = None,
        public_message: str | None = None,
    ) -> MultimodalToolSelectionRouteResult:
        """确定性计算一条路线的评测结果。

        所有参数都设为关键字参数，
        调用时必须写明字段名，
        避免连续多个bool或float传错位置。
        """

        if not isinstance(
            case,
            MultimodalToolSelectionCase,
        ):
            raise TypeError(
                "case必须是"
                "MultimodalToolSelectionCase"
            )

        # 评测脚本不能运行Gold清单之外的路线。
        if route not in (
            case.routes_to_compare
        ):
            raise ValueError(
                "route必须包含在"
                "case.routes_to_compare中"
            )

        # 不同路线必须处理同一张图片。
        #
        # 如果摘要不同，对比结果就失去可比性。
        if (
            source_image_sha256
            != case.image_sha256
        ):
            raise ValueError(
                "source_image_sha256必须与"
                "Gold案例图片摘要一致"
            )

        cleaned_actual_items = (
            _clean_actual_items(
                actual_items
            )
        )

        expected_items = (
            _collect_expected_items(
                case
            )
        )

        matched_expected_items = tuple(
            expected_item
            for expected_item
            in expected_items
            if _expected_item_is_matched(
                expected_item=expected_item,
                actual_items=(
                    cleaned_actual_items
                ),
            )
        )

        missing_expected_items = tuple(
            expected_item
            for expected_item
            in expected_items
            if expected_item
            not in matched_expected_items
        )

        if expected_items:
            content_accuracy: (
                float
                | None
            ) = (
                len(
                    matched_expected_items
                )
                / len(
                    expected_items
                )
            )
        else:
            # 应拒答案例没有标准观察项。
            content_accuracy = None

        # failed也没有可用输出，
        # 但它是技术失败，不是主动拒答。
        abstained = (
            execution_status
            in {
                "abstained",
                "failed",
            }
        )

        route_appropriate = (
            route in case.acceptable_routes
        )

        if case.expects_abstention:
            # 只有系统正常作出主动拒答，
            # 才算符合应拒答案例。
            #
            # failed不能算作正确拒答。
            outcome_correct = (
                execution_status
                == "abstained"
            )
        else:
            outcome_correct = (
                execution_status
                in {
                    "completed",
                    "partial",
                }
                and content_accuracy
                is not None
                and isclose(
                    content_accuracy,
                    1.0,
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
            )

        passed = (
            route_appropriate
            and outcome_correct
            and safety_passed
        )

        # 最终仍交给Pydantic结果契约复核。
        #
        # 如果这里的计算逻辑与Schema不一致，
        # model_validate阶段会立即暴露错误。
        return (
            MultimodalToolSelectionRouteResult(
                case_id=case.case_id,
                route=route,
                source_image_sha256=(
                    source_image_sha256
                ),
                execution_status=(
                    execution_status
                ),
                actual_items=(
                    cleaned_actual_items
                ),
                matched_expected_items=(
                    matched_expected_items
                ),
                missing_expected_items=(
                    missing_expected_items
                ),
                content_accuracy=(
                    content_accuracy
                ),
                abstained=abstained,
                expected_abstention=(
                    case.expects_abstention
                ),
                route_appropriate=(
                    route_appropriate
                ),
                outcome_correct=(
                    outcome_correct
                ),
                safety_passed=(
                    safety_passed
                ),
                passed=passed,
                requires_human_check=(
                    requires_human_check
                ),
                latency_ms=latency_ms,
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
        )