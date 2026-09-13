"""Agent请求的确定性安全分类器。

本模块在Agent Loop和请求级工具策略之前运行，负责识别：

1. 要求忽略系统规则、扩大权限或泄露敏感信息；
2. 要求访问图片、二维码或日志中的外部链接；
3. 要求执行图片、二维码或日志中的命令；
4. 要求绕过急停、安全联锁或控制真实设备。

分类器只处理请求文本中已经可见的安全信号。

它不会：

1. 调用LLM；
2. 访问任何URL；
3. 解码或识别图片像素；
4. 执行图片或日志中的命令；
5. 控制机器人；
6. 生成任何可执行控制步骤；
7. 判断正常请求需要哪些业务工具。

安全请求返回None，表示可以继续交给AgentToolPolicy。

不安全请求返回AgentToolPolicyDecision，
处理方向固定为human_review_required，
并且allowed_tool_names必须为空。
"""

import re
import unicodedata

from app.schemas.agent_api import (
    AgentDiagnosisRequest,
)
from app.schemas.agent_tool_policy import (
    AgentToolPolicyDecision,
    AgentToolPolicyReasonCode,
)


# 连续空白统一成一个普通空格。
#
# 这样换行、Tab或多个空格不会影响规则匹配。
_WHITESPACE_PATTERN = re.compile(
    r"\s+"
)


# 如果危险动作前面出现这些否定表达，
# 说明用户可能是在声明安全限制，
# 而不是要求系统执行该动作。
#
# 例如：
#
# “不得带电插拔”
# “不要访问日志中的链接”
# “禁止执行图片里的命令”
#
# 这些表达本身不应被当成危险请求。
_NEGATION_SUFFIX_PATTERN = re.compile(
    (
        r"(?:"
        r"不|不要|不得|禁止|严禁|切勿|"
        r"不可|不能|无需|避免|拒绝|"
        r"未要求|没有要求"
        r")"
        r".{0,8}$"
    )
)


# 通用提示注入和权限扩大信号。
#
# 这里匹配的是行为意图类别，
# 不是某一个评测案例的完整句子。
_PROMPT_INJECTION_PATTERNS: tuple[
    re.Pattern[str],
    ...,
] = (
    # 中文的忽略、覆盖系统规则表达。
    re.compile(
        (
            r"(?:忽略|无视|覆盖|删除|改写)"
            r".{0,20}"
            r"(?:系统规则|安全规则|先前指令|"
            r"之前指令|开发者指令|system prompt)"
        ),
        re.IGNORECASE,
    ),

    # 英文常见提示注入表达。
    re.compile(
        (
            r"(?:ignore|override|forget|discard)"
            r".{0,24}"
            r"(?:previous|prior|system|developer|"
            r"safety)"
            r".{0,16}"
            r"(?:instruction|rule|prompt)"
        ),
        re.IGNORECASE,
    ),

    # 要求调用不在白名单内的危险工具。
    re.compile(
        (
            r"(?:调用|运行|执行|使用|call|run|execute|use)"
            r".{0,16}"
            r"(?:run_shell|execute_python|"
            r"control_robot|reset_emergency_stop)"
        ),
        re.IGNORECASE,
    ),

    # 要求输出密钥、令牌或密码。
    re.compile(
        (
            r"(?:输出|显示|打印|泄露|返回|"
            r"print|show|reveal|expose)"
            r".{0,20}"
            r"(?:api[ _-]?key|密钥|token|令牌|"
            r"password|密码|secret)"
        ),
        re.IGNORECASE,
    ),
)


# 要求访问外部链接的安全信号。
#
# 系统当前只处理本地提交的脱敏输入，
# 不应根据图片、二维码或日志中的网址
# 自动打开网页或下载内容。
_EXTERNAL_LINK_ACTION_PATTERNS: tuple[
    re.Pattern[str],
    ...,
] = (
    re.compile(
        (
            r"(?:打开|访问|点击|请求|下载|跳转|扫描|跟随)"
            r".{0,30}"
            r"(?:https?://|www\.|网址|链接|url|二维码)"
        ),
        re.IGNORECASE,
    ),
    re.compile(
        (
            r"(?:open|visit|access|click|fetch|"
            r"download|follow|scan)"
            r".{0,30}"
            r"(?:https?://|www\.|url|link|qr[ _-]?code)"
        ),
        re.IGNORECASE,
    ),
)


# 要求执行不可信来源中命令的安全信号。
#
# 图片文字、二维码内容和日志正文只能作为数据观察，
# 不能改变Agent的工具权限或执行流程。
_UNTRUSTED_INSTRUCTION_PATTERNS: tuple[
    re.Pattern[str],
    ...,
] = (
    # “执行图片中的命令”一类表达。
    re.compile(
        (
            r"(?:执行|遵循|按照|运行|照做)"
            r".{0,20}"
            r"(?:图片|图像|照片|二维码|日志|链接|网址)"
            r".{0,16}"
            r"(?:命令|指令|步骤|操作)"
        ),
        re.IGNORECASE,
    ),

    # “图片中的命令需要执行”一类倒序表达。
    re.compile(
        (
            r"(?:图片|图像|照片|二维码|日志|链接|网址)"
            r".{0,16}"
            r"(?:命令|指令|步骤|操作)"
            r".{0,20}"
            r"(?:执行|遵循|按照|运行|照做)"
        ),
        re.IGNORECASE,
    ),

    # 对应的英文表达。
    re.compile(
        (
            r"(?:execute|follow|run|obey)"
            r".{0,20}"
            r"(?:image|photo|qr[ _-]?code|log|link)"
            r".{0,16}"
            r"(?:command|instruction|step|operation)"
        ),
        re.IGNORECASE,
    ),
)


# 真实设备高风险控制信号。
#
# 当前Agent是只读诊断系统，
# 因此不能解除保护装置、写入控制状态，
# 或向真实机器人下发运动命令。
_HIGH_RISK_CONTROL_PATTERNS: tuple[
    re.Pattern[str],
    ...,
] = (
    # 绕过、关闭或短接安全保护。
    re.compile(
        (
            r"(?:绕过|禁用|关闭|屏蔽|短接|解除)"
            r".{0,16}"
            r"(?:急停|安全联锁|安全回路|保护装置|"
            r"避障|制动|限位|"
            r"emergency[ _-]?stop|safety[ _-]?interlock)"
        ),
        re.IGNORECASE,
    ),

    # 英文的安全保护绕过表达。
    re.compile(
        (
            r"(?:bypass|disable|override|release|reset)"
            r".{0,20}"
            r"(?:emergency[ _-]?stop|e[ _-]?stop|"
            r"safety[ _-]?interlock|safety[ _-]?circuit|"
            r"brake|collision[ _-]?protection)"
        ),
        re.IGNORECASE,
    ),

    # 远程或直接控制机器人、驱动器和电机。
    re.compile(
        (
            r"(?:远程|直接)"
            r".{0,12}"
            r"(?:控制|启动|复位|解除|驱动)"
            r".{0,16}"
            r"(?:机器人|设备|电机|驱动器|制动器)"
        ),
        re.IGNORECASE,
    ),

    # 向设备发送运动、启动或复位命令。
    re.compile(
        (
            r"(?:向|给)"
            r".{0,16}"
            r"(?:机器人|设备|电机|驱动器)"
            r".{0,16}"
            r"(?:发送|下发|执行)"
            r".{0,16}"
            r"(?:运动|移动|启动|复位|继续运行|控制)"
            r".{0,8}"
            r"(?:命令|指令)"
        ),
        re.IGNORECASE,
    ),

    # 带电插拔和短接安全回路属于现场高风险动作。
    re.compile(
        (
            r"(?:带电插拔|短接安全回路|"
            r"拆除安全防护|强制释放制动)"
        ),
        re.IGNORECASE,
    ),
)


def _normalize_safety_text(
    value: str,
    /,
) -> str:
    """为安全规则匹配执行轻量文本归一化。"""

    if not isinstance(value, str):
        raise TypeError(
            "安全分类文本必须是字符串"
        )

    # NFKC把全角字符等兼容形式转换成稳定形式。
    normalized = unicodedata.normalize(
        "NFKC",
        value,
    )

    # casefold()是Unicode友好的大小写归一化。
    normalized = normalized.casefold()

    # 把多个空白字符统一成一个普通空格。
    return _WHITESPACE_PATTERN.sub(
        " ",
        normalized,
    ).strip()


def _request_texts(
    request: AgentDiagnosisRequest,
    /,
) -> tuple[str, ...]:
    """收集允许进入确定性安全检查的请求文本。

    image_base64不会被读取或加入结果，
    因为完整图片内容不能进入文本安全扫描、
    日志或错误报告。
    """

    texts: list[str] = [
        request.symptom,
        request.log_excerpt,
    ]

    if request.task_goal is not None:
        texts.append(
            request.task_goal
        )

    # analysis_goal也是外部传入的不可信文本，
    # 因此需要执行相同的安全扫描。
    texts.extend(
        image.analysis_goal
        for image in request.images
    )

    return tuple(
        _normalize_safety_text(text)
        for text in texts
    )


def _match_is_negated(
    *,
    text: str,
    match_start: int,
) -> bool:
    """判断一次危险动作匹配是否被明确否定。

    只检查匹配前的短窗口，
    防止句子前面很远处的“不”错误否定后续动作。
    """

    prefix_start = max(
        0,
        match_start - 16,
    )

    prefix = text[
        prefix_start:match_start
    ]

    return (
        _NEGATION_SUFFIX_PATTERN.search(
            prefix
        )
        is not None
    )


def _contains_unnegated_pattern(
    *,
    text: str,
    patterns: tuple[
        re.Pattern[str],
        ...,
    ],
) -> bool:
    """判断文本是否包含未被安全否定的规则匹配。"""

    for pattern in patterns:
        for match in pattern.finditer(text):
            if not _match_is_negated(
                text=text,
                match_start=match.start(),
            ):
                return True

    return False


def _detect_reason_codes(
    request: AgentDiagnosisRequest,
    /,
) -> tuple[
    AgentToolPolicyReasonCode,
    ...,
]:
    """返回本次请求命中的稳定安全原因码。"""

    matched_codes: list[
        AgentToolPolicyReasonCode
    ] = []

    for text in _request_texts(request):
        if _contains_unnegated_pattern(
            text=text,
            patterns=(
                _PROMPT_INJECTION_PATTERNS
            ),
        ):
            matched_codes.append(
                "prompt_injection_detected"
            )

        if _contains_unnegated_pattern(
            text=text,
            patterns=(
                _EXTERNAL_LINK_ACTION_PATTERNS
            ),
        ):
            matched_codes.append(
                "external_link_action_detected"
            )

        if _contains_unnegated_pattern(
            text=text,
            patterns=(
                _UNTRUSTED_INSTRUCTION_PATTERNS
            ),
        ):
            matched_codes.append(
                "untrusted_instruction_execution_detected"
            )

        if _contains_unnegated_pattern(
            text=text,
            patterns=(
                _HIGH_RISK_CONTROL_PATTERNS
            ),
        ):
            matched_codes.append(
                "high_risk_control_detected"
            )

    # dict.fromkeys()保留首次出现顺序并去重。
    #
    # 同一个信号即使同时出现在日志和task_goal中，
    # 最终原因码也只保留一次。
    return tuple(
        dict.fromkeys(
            matched_codes
        )
    )


def _build_public_message(
    reason_codes: tuple[
        AgentToolPolicyReasonCode,
        ...,
    ],
    /,
) -> str:
    """根据安全原因生成不可执行的公开说明。"""

    has_high_risk = (
        "high_risk_control_detected"
        in reason_codes
    )

    has_untrusted_action = any(
        reason_code
        in {
            "prompt_injection_detected",
            "external_link_action_detected",
            "untrusted_instruction_execution_detected",
        }
        for reason_code in reason_codes
    )

    if (
        has_high_risk
        and has_untrusted_action
    ):
        return (
            "请求同时涉及不可信外部指令和"
            "真实设备安全控制，"
            "系统不会执行工具调用，"
            "需要人工审核"
        )

    if has_high_risk:
        return (
            "请求涉及真实设备安全控制，"
            "当前系统仅提供只读诊断，"
            "需要由具备权限和资质的人员审核"
        )

    return (
        "请求包含要求改变系统规则或"
        "执行不可信外部内容的意图，"
        "系统不会访问链接或执行其中的命令，"
        "需要人工审核"
    )


def _build_missing_information(
    reason_codes: tuple[
        AgentToolPolicyReasonCode,
        ...,
    ],
    /,
) -> tuple[str, ...]:
    """说明自动处理不能继续的人工审核事项。"""

    missing_information: list[str] = []

    if any(
        reason_code
        in {
            "prompt_injection_detected",
            "external_link_action_detected",
            "untrusted_instruction_execution_detected",
        }
        for reason_code in reason_codes
    ):
        missing_information.append(
            "需要人工核验请求中的不可信指令、"
            "外部链接及其授权边界"
        )

    if (
        "high_risk_control_detected"
        in reason_codes
    ):
        missing_information.append(
            "需要具备相应权限和资质的人员"
            "在现场确认设备与安全保护状态"
        )

    return tuple(
        missing_information
    )


class AgentRequestSafetyClassifier:
    """在Planner之前识别必须转人工审核的请求。

    返回None表示本分类器没有发现阻断信号，
    不代表请求已经完成，也不代表可以开放全部工具。

    调用方随后仍必须调用AgentToolPolicy.decide()，
    计算真正需要的最小工具集合。
    """

    def classify(
        self,
        request: AgentDiagnosisRequest,
        /,
    ) -> AgentToolPolicyDecision | None:
        """返回安全终止决定，或允许进入工具策略阶段。"""

        if not isinstance(
            request,
            AgentDiagnosisRequest,
        ):
            raise TypeError(
                "request必须是"
                "AgentDiagnosisRequest"
            )

        reason_codes = (
            _detect_reason_codes(
                request
            )
        )

        # 没有发现安全阻断信号。
        #
        # 这里不能直接返回continue_to_planner，
        # 因为安全分类器不知道任务所需的工具能力。
        if not reason_codes:
            return None

        # no_safe_tool_required放在最后，
        # 表示前面的安全原因导致所有工具被关闭。
        final_reason_codes = (
            *reason_codes,
            "no_safe_tool_required",
        )

        return AgentToolPolicyDecision(
            policy_version=(
                "agent-tool-policy-v1"
            ),
            disposition=(
                "human_review_required"
            ),
            required_capabilities=(),
            allowed_tool_names=(),
            reason_codes=(
                final_reason_codes
            ),
            public_message=(
                _build_public_message(
                    reason_codes
                )
            ),
            missing_information=(
                _build_missing_information(
                    reason_codes
                )
            ),
        )