"""OCR文字的确定性规则解析服务。

本模块接收OcrObservation，
把OCR识别出的普通文字转换成OcrRuleParseResult。

它负责：

1. 规范化OCR文字；
2. 判断文字属于状态面板还是遥测面板；
3. 用固定正则提取受支持字段；
4. 检查重复候选值和数值范围；
5. 根据OCR质量和字段完整性确定解析状态；
6. 对无法可靠解析的结果要求人工复核。

本模块不负责：

1. 读取或解码图片；
2. 调用Tesseract；
3. 调用Vision模型或LLM；
4. 解释故障码的工程含义；
5. 检索知识库；
6. 生成最终诊断结论。
"""

# dataclass用于定义模块内部的临时解析结果。
#
# 这个对象不会作为API响应，
# 只帮助解析器在多个函数之间传递字段。
from dataclasses import (
    dataclass,
)

# re是Python标准库中的正则表达式模块。
#
# compile()会预先编译固定规则，
# 避免每次解析图片时重复创建正则表达式。
import re

# unicodedata用于执行Unicode文本规范化。
#
# OCR可能返回全角字符、不同形式的破折号，
# NFKC规范化可以减少字符形式差异。
import unicodedata

# OcrObservation是上游OCR Provider的输出契约。
from app.schemas.ocr import (
    OcrObservation,
)

# 这里导入规则结果及其公共常量。
#
# 解析器与Schema共同使用相同的字段顺序、
# 必要字段列表和版本号，避免两边规则不一致。
from app.schemas.ocr_rule import (
    OCR_RULE_FIELD_ORDER,
    OCR_RULE_VERSION,
    STATUS_PANEL_REQUIRED_FIELDS,
    TELEMETRY_PANEL_REQUIRED_FIELDS,
    OcrRuleFieldName,
    OcrRulePanelType,
    OcrRuleParseResult,
)


# 正则中复用的十进制数格式。
#
# 支持：
#
# 42
# 42.5
# -10
# +3.2
# .5
#
# 不支持NaN、Infinity和科学计数法，
# 因为现场面板不应该使用这些形式。
_NUMBER_PATTERN = (
    r"[+-]?"
    r"(?:"
    r"\d+(?:\.\d+)?"
    r"|"
    r"\.\d+"
    r")"
)


# 状态面板字段规则。
#
# \b表示单词边界；
# \s*表示允许出现任意数量空白；
# [:=-]?表示标签和值之间可以有冒号、等号或短横线；
# 圆括号中的部分是需要提取的值。
_FAULT_CODE_PATTERN = re.compile(
    r"\bFAULT\s*CODE\s*[:=-]?\s*"
    r"(ERR-[A-Z0-9]+-[0-9]{4})\b"
)

_NETWORK_STATE_PATTERN = re.compile(
    r"\bNETWORK\s*[:=-]?\s*"
    r"(DISCONNECTED|CONNECTED)\b"
)

_TASK_STATE_PATTERN = re.compile(
    r"\bTASK\s*[:=-]?\s*"
    r"(PAUSED|RUNNING|STOPPED|IDLE|CHARGING)\b"
)


# 遥测面板字段规则。
_BATTERY_PATTERN = re.compile(
    rf"\bBATTERY\s*[:=-]?\s*"
    rf"({_NUMBER_PATTERN})\s*%"
)

_SPEED_PATTERN = re.compile(
    rf"\bSPEED\s*[:=-]?\s*"
    rf"({_NUMBER_PATTERN})\s*"
    rf"M\s*/\s*S\b"
)

_TEMPERATURE_PATTERN = re.compile(
    rf"\bTEMPERATURE\s*[:=-]?\s*"
    rf"({_NUMBER_PATTERN})\s*"
    rf"(?:°\s*)?C\b"
)


# 面板类型检测只检查标签，不直接提取字段值。
#
# 至少发现同一类面板的两个标签，
# 才认为面板类型具有基本可信度。
_STATUS_MARKER_PATTERNS = (
    re.compile(
        r"\bFAULT\s*CODE\b"
    ),
    re.compile(
        r"\bNETWORK\b"
    ),
    re.compile(
        r"\bTASK\b"
    ),
)

_TELEMETRY_MARKER_PATTERNS = (
    re.compile(
        r"\bBATTERY\b"
    ),
    re.compile(
        r"\bSPEED\b"
    ),
    re.compile(
        r"\bTEMPERATURE\b"
    ),
)


@dataclass(
    frozen=True,
    slots=True,
)
class _ParsedFields:
    """规则解析过程中的内部字段集合。

    下划线前缀表示这是模块私有实现，
    其他模块不应把它当成公共数据契约。

    frozen=True：
    创建以后不能修改字段，避免处理途中意外改值。

    slots=True：
    不给实例创建任意属性字典，
    防止拼错属性名后静默增加新属性。
    """

    fault_code: str | None = None
    network_state: str | None = None
    task_state: str | None = None
    battery_percent: float | None = None
    speed_mps: float | None = None
    temperature_celsius: float | None = None

    # 保存字段提取失败的原因。
    #
    # 这里只记录简短、脱敏的失败说明，
    # 不保存完整OCR文字。
    failure_reasons: tuple[
        str,
        ...,
    ] = ()


def _normalize_ocr_text(
    raw_text: str,
) -> str:
    """将OCR文字规范化成适合规则匹配的形式。

    规范化只改变文字形式，
    不会推断或补充OCR没有识别出的内容。
    """

    # NFKC会把很多全角字符转换为兼容的半角形式。
    normalized_text = unicodedata.normalize(
        "NFKC",
        raw_text,
    )

    # OCR和PDF中可能出现多种Unicode破折号。
    #
    # 故障码必须统一成ERR-NET-4001形式，
    # 所以将这些字符统一替换成ASCII短横线。
    for dash_character in (
        "‐",
        "‑",
        "‒",
        "–",
        "—",
        "−",
    ):
        normalized_text = (
            normalized_text.replace(
                dash_character,
                "-",
            )
        )

    # 规则只支持英文固定标签，
    # 因此统一转换成大写。
    normalized_text = (
        normalized_text.upper()
    )

    # 把换行、制表符和连续空格压缩成一个空格。
    #
    # 这样无论标签和值在同一行还是相邻行，
    # 固定正则都可以使用相同方式处理。
    normalized_text = re.sub(
        r"\s+",
        " ",
        normalized_text,
    )

    return normalized_text.strip()


def _count_matching_markers(
    *,
    normalized_text: str,
    marker_patterns: tuple[
        re.Pattern[str],
        ...,
    ],
) -> int:
    """统计某一类面板标签的命中数量。"""

    return sum(
        1
        for marker_pattern in marker_patterns
        if marker_pattern.search(
            normalized_text
        )
        is not None
    )


def _detect_panel_type(
    normalized_text: str,
) -> OcrRulePanelType:
    """根据标签数量判断面板类型。

    检测依据不是LLM语义推断，
    而是预先声明的固定标签。

    决策规则：

    1. 状态标签至少命中两个；
    2. 状态标签命中数大于遥测标签时，
       识别为robot_status；
    3. 遥测标签使用相同规则；
    4. 命中不足或两类得分相同时，
       返回unknown。
    """

    status_score = (
        _count_matching_markers(
            normalized_text=(
                normalized_text
            ),
            marker_patterns=(
                _STATUS_MARKER_PATTERNS
            ),
        )
    )

    telemetry_score = (
        _count_matching_markers(
            normalized_text=(
                normalized_text
            ),
            marker_patterns=(
                _TELEMETRY_MARKER_PATTERNS
            ),
        )
    )

    if (
        status_score >= 2
        and status_score
        > telemetry_score
    ):
        return "robot_status"

    if (
        telemetry_score >= 2
        and telemetry_score
        > status_score
    ):
        return "robot_telemetry"

    return "unknown"


def _unique_items(
    items: tuple[
        str,
        ...,
    ],
) -> tuple[
    str,
    ...,
]:
    """保持原始顺序并删除重复字符串。

    dict能够保持插入顺序，
    fromkeys()会把字符串作为键去重。
    """

    return tuple(
        dict.fromkeys(
            items
        )
    )


def _extract_text_value(
    *,
    normalized_text: str,
    pattern: re.Pattern[str],
    field_name: OcrRuleFieldName,
) -> tuple[
    str | None,
    str | None,
]:
    """从文字中提取一个唯一字符串值。

    返回值是二元tuple：

    第一个元素：
    成功提取的字段值，失败时为None。

    第二个元素：
    失败原因，成功时为None。
    """

    candidates = _unique_items(
        tuple(
            match.group(1)
            for match in pattern.finditer(
                normalized_text
            )
        )
    )

    if not candidates:
        return (
            None,
            (
                f"未找到{field_name}"
                "字段的有效值"
            ),
        )

    if len(candidates) > 1:
        return (
            None,
            (
                f"{field_name}字段存在"
                "多个不同候选值"
            ),
        )

    return (
        candidates[0],
        None,
    )


def _extract_number_value(
    *,
    normalized_text: str,
    pattern: re.Pattern[str],
    field_name: OcrRuleFieldName,
    minimum: float,
    maximum: float,
) -> tuple[
    float | None,
    str | None,
]:
    """提取唯一数值并执行允许范围检查。"""

    string_candidates = (
        _unique_items(
            tuple(
                match.group(1)
                for match
                in pattern.finditer(
                    normalized_text
                )
            )
        )
    )

    if not string_candidates:
        return (
            None,
            (
                f"未找到{field_name}"
                "字段的有效值"
            ),
        )

    # 不同文字可能表示相同浮点数，
    # 例如42和42.0。
    numeric_candidates = tuple(
        dict.fromkeys(
            float(candidate)
            for candidate
            in string_candidates
        )
    )

    if len(numeric_candidates) > 1:
        return (
            None,
            (
                f"{field_name}字段存在"
                "多个不同候选值"
            ),
        )

    value = numeric_candidates[0]

    if (
        value < minimum
        or value > maximum
    ):
        return (
            None,
            (
                f"{field_name}字段数值"
                "超出允许范围"
            ),
        )

    return (
        value,
        None,
    )


def _collect_failure_reasons(
    *reasons: str | None,
) -> tuple[
    str,
    ...,
]:
    """移除None并保持失败原因唯一。"""

    return _unique_items(
        tuple(
            reason
            for reason in reasons
            if reason is not None
        )
    )


def _parse_status_panel(
    normalized_text: str,
) -> _ParsedFields:
    """解析机器人状态面板的三个必要字段。"""

    (
        fault_code,
        fault_reason,
    ) = _extract_text_value(
        normalized_text=normalized_text,
        pattern=_FAULT_CODE_PATTERN,
        field_name="fault_code",
    )

    (
        network_state,
        network_reason,
    ) = _extract_text_value(
        normalized_text=normalized_text,
        pattern=_NETWORK_STATE_PATTERN,
        field_name="network_state",
    )

    (
        task_state,
        task_reason,
    ) = _extract_text_value(
        normalized_text=normalized_text,
        pattern=_TASK_STATE_PATTERN,
        field_name="task_state",
    )

    return _ParsedFields(
        fault_code=fault_code,
        network_state=network_state,
        task_state=task_state,
        failure_reasons=(
            _collect_failure_reasons(
                fault_reason,
                network_reason,
                task_reason,
            )
        ),
    )


def _parse_telemetry_panel(
    normalized_text: str,
) -> _ParsedFields:
    """解析机器人遥测面板的三个必要数值。"""

    (
        battery_percent,
        battery_reason,
    ) = _extract_number_value(
        normalized_text=normalized_text,
        pattern=_BATTERY_PATTERN,
        field_name="battery_percent",
        minimum=0.0,
        maximum=100.0,
    )

    (
        speed_mps,
        speed_reason,
    ) = _extract_number_value(
        normalized_text=normalized_text,
        pattern=_SPEED_PATTERN,
        field_name="speed_mps",
        minimum=-20.0,
        maximum=20.0,
    )

    (
        temperature_celsius,
        temperature_reason,
    ) = _extract_number_value(
        normalized_text=normalized_text,
        pattern=_TEMPERATURE_PATTERN,
        field_name=(
            "temperature_celsius"
        ),
        minimum=-100.0,
        maximum=200.0,
    )

    return _ParsedFields(
        battery_percent=battery_percent,
        speed_mps=speed_mps,
        temperature_celsius=(
            temperature_celsius
        ),
        failure_reasons=(
            _collect_failure_reasons(
                battery_reason,
                speed_reason,
                temperature_reason,
            )
        ),
    )


def _get_field_value(
    *,
    parsed_fields: _ParsedFields,
    field_name: OcrRuleFieldName,
) -> object | None:
    """按公共字段名读取内部解析结果。

    getattr(object, name)等价于object.name，
    但允许name在运行时由变量提供。
    """

    return getattr(
        parsed_fields,
        field_name,
    )


def _build_unknown_result(
    *,
    observation: OcrObservation,
    failure_reason: str,
) -> OcrRuleParseResult:
    """构造规则不支持的安全降级结果。"""

    return OcrRuleParseResult(
        status="unsupported",
        panel_type="unknown",
        ocr_status=observation.status,
        source_image_sha256=(
            observation.source_image_sha256
        ),
        source_text_sha256=(
            observation
            .recognized_text_sha256
        ),
        matched_fields=(),
        missing_fields=(),
        failure_reasons=(
            failure_reason,
        ),
        requires_human_check=True,
    )


def _build_known_panel_result(
    *,
    observation: OcrObservation,
    panel_type: OcrRulePanelType,
    parsed_fields: _ParsedFields,
) -> OcrRuleParseResult:
    """根据字段完整性和OCR质量构造最终结果。"""

    if panel_type == "robot_status":
        required_fields = (
            STATUS_PANEL_REQUIRED_FIELDS
        )
    else:
        required_fields = (
            TELEMETRY_PANEL_REQUIRED_FIELDS
        )

    # matched_fields必须按照Schema声明的固定顺序生成。
    matched_fields = tuple(
        field_name
        for field_name
        in OCR_RULE_FIELD_ORDER
        if _get_field_value(
            parsed_fields=parsed_fields,
            field_name=field_name,
        )
        is not None
    )

    missing_fields = tuple(
        field_name
        for field_name
        in required_fields
        if _get_field_value(
            parsed_fields=parsed_fields,
            field_name=field_name,
        )
        is None
    )

    # Schema规定partial至少包含一个成功字段。
    #
    # 如果标签看起来像某类面板，
    # 但一个有效值也没能提取出来，
    # 就不能假装这是有用的部分结果。
    if not matched_fields:
        failure_reasons = (
            parsed_fields.failure_reasons
            + (
                "识别到受支持面板标签，"
                "但没有解析出有效字段",
            )
        )

        return OcrRuleParseResult(
            status="unsupported",
            panel_type="unknown",
            ocr_status=observation.status,
            source_image_sha256=(
                observation
                .source_image_sha256
            ),
            source_text_sha256=(
                observation
                .recognized_text_sha256
            ),
            matched_fields=(),
            missing_fields=(),
            failure_reasons=(
                _unique_items(
                    failure_reasons
                )
            ),
            requires_human_check=True,
        )

    failure_reasons = (
        parsed_fields.failure_reasons
    )

    # 上游OCR低置信度时，
    # 即使规则碰巧解析出所有字段，
    # 也不能升级成completed。
    if observation.status == (
        "low_confidence"
    ):
        failure_reasons = _unique_items(
            failure_reasons
            + (
                "上游OCR平均置信度"
                "低于配置门槛",
            )
        )

    if (
        not missing_fields
        and observation.status
        == "completed"
    ):
        parse_status = "completed"
        requires_human_check = False

        # completed状态不允许携带失败原因。
        failure_reasons = ()
    else:
        parse_status = "partial"
        requires_human_check = True

    return OcrRuleParseResult(
        status=parse_status,
        panel_type=panel_type,
        ocr_status=observation.status,
        source_image_sha256=(
            observation.source_image_sha256
        ),
        source_text_sha256=(
            observation.recognized_text_sha256
        ),
        fault_code=(
            parsed_fields.fault_code
        ),
        network_state=(
            parsed_fields.network_state
        ),
        task_state=(
            parsed_fields.task_state
        ),
        battery_percent=(
            parsed_fields.battery_percent
        ),
        speed_mps=(
            parsed_fields.speed_mps
        ),
        temperature_celsius=(
            parsed_fields
            .temperature_celsius
        ),
        matched_fields=matched_fields,
        missing_fields=missing_fields,
        failure_reasons=failure_reasons,
        requires_human_check=(
            requires_human_check
        ),
    )


class OcrRuleParser:
    """把OcrObservation转换成确定性结构字段。

    该服务不保存可变状态，
    同一个实例可以连续解析多次OCR结果。
    """

    @property
    def rules_version(
        self,
    ) -> str:
        """返回当前实际使用的规则版本。"""

        return OCR_RULE_VERSION

    def parse(
        self,
        *,
        observation: OcrObservation,
    ) -> OcrRuleParseResult:
        """解析一次OCR观察结果。

        本方法是同步方法，因为它只执行字符串处理、
        正则匹配和Pydantic模型构造，
        不访问网络、文件或外部进程。
        """

        if not isinstance(
            observation,
            OcrObservation,
        ):
            raise TypeError(
                "observation必须是"
                "OcrObservation"
            )

        # 上游已经确定没有识别文字时，
        # 不再执行面板检测和字段正则。
        if observation.status == "empty":
            return OcrRuleParseResult(
                status="empty",
                panel_type="unknown",
                ocr_status="empty",
                source_image_sha256=(
                    observation
                    .source_image_sha256
                ),
                source_text_sha256=(
                    observation
                    .recognized_text_sha256
                ),
                matched_fields=(),
                missing_fields=(),
                failure_reasons=(
                    "上游OCR没有识别到"
                    "可供规则解析的文字",
                ),
                requires_human_check=True,
            )

        normalized_text = (
            _normalize_ocr_text(
                observation.recognized_text
            )
        )

        panel_type = _detect_panel_type(
            normalized_text
        )

        if panel_type == "unknown":
            return _build_unknown_result(
                observation=observation,
                failure_reason=(
                    "OCR文字不符合当前支持的"
                    "状态面板或遥测面板格式"
                ),
            )

        if panel_type == "robot_status":
            parsed_fields = (
                _parse_status_panel(
                    normalized_text
                )
            )
        else:
            parsed_fields = (
                _parse_telemetry_panel(
                    normalized_text
                )
            )

        return _build_known_panel_result(
            observation=observation,
            panel_type=panel_type,
            parsed_fields=parsed_fields,
        )