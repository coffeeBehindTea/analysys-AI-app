"""请求级最小工具策略。

本模块负责：

1. 从AgentDiagnosisRequest的任务目标中识别所需能力；
2. 识别任务对某项能力的明确排除要求；
3. 判断任务是否依赖图片；
4. 判断图片analysis_goal与任务目标是否明显无关；
5. 把所需能力映射成最小工具集合；
6. 在缺少必要图片时生成提前拒答策略；
7. 返回经过Pydantic校验的AgentToolPolicyDecision。

本模块暂时不负责：

1. 识别提示注入；
2. 识别高风险设备控制意图；
3. 调用Planner；
4. 执行工具；
5. 检查工具执行后的证据覆盖；
6. 生成最终DiagnosisReport。

提示注入和高风险请求将在独立安全分类器中处理。
工具执行后的状态管理将在AgentProgress中处理。
"""

import re
import unicodedata

from app.schemas.agent_api import (
    AgentDiagnosisRequest,
)
from app.schemas.agent_tool_policy import (
    AGENT_READ_ONLY_TOOL_ORDER,
    CAPABILITY_TO_TOOL,
    AgentRequiredCapability,
    AgentToolPolicyDecision,
    AgentToolPolicyReasonCode,
)


# 连续空白统一成一个普通空格。
#
# 这样换行、Tab和多个空格不会影响确定性规则匹配。
_WHITESPACE_PATTERN = re.compile(
    r"\s+"
)


# 每项能力对应一组通用任务意图模式。
#
# 这些规则描述的是能力类别，
# 不能包含multimodal-agent-001之类的案例编号，
# 也不能包含只为某一道Gold题设计的完整句子。
_CAPABILITY_PATTERNS: dict[
    AgentRequiredCapability,
    tuple[
        re.Pattern[str],
        ...,
    ],
] = {
    "vision_observation": (
        # “结合图片”“使用照片”同样明确要求视觉输入，
        # 不一定会同时出现“读取”或“观察”等动词。
        re.compile(
            (
                r"(?:结合|使用|根据)"
                r".{0,12}"
                r"(?:图片|图像|照片)"
            )
        ),
        # “核对可见状态”描述的是现场可见信息，
        # 即使句子没有再次写出“图片”也属于视觉能力。
        re.compile(
            (
                r"(?:核对|确认|判断|观察)"
                r".{0,12}"
                r"(?:可见状态|可见现象|外观状态)"
            )
        ),
        re.compile(
            (
                r"(?:读取|观察|查看|识别|分析|核对|判断|确认)"
                r".{0,16}"
                r"(?:图片|图像|照片|面板|指示灯|外观|连接器)"
            )
        ),
        re.compile(
            (
                r"(?:图片|图像|照片|面板|指示灯)"
                r".{0,16}"
                r"(?:文字|数值|颜色|亮灭|状态|外观|"
                r"故障码|错误码|可读|清晰|模糊|遮挡)"
            )
        ),
        re.compile(
            (
                r"(?:可见|不可见|遮挡|模糊|清晰)"
                r".{0,12}"
                r"(?:字段|数值|文字|故障码|错误码|部件|区域)"
            )
        ),
        re.compile(
            (
                r"(?:read|inspect|observe|check|analy[sz]e)"
                r".{0,24}"
                r"(?:image|photo|panel|indicator|connector)"
            ),
            re.IGNORECASE,
        ),
    ),
    "knowledge_evidence": (
        # 用户可以说“结合知识库”或“根据手册”，
        # 这种表达同样明确要求外部知识证据。
        re.compile(
            (
                r"(?:结合|使用|根据)"
                r".{0,16}"
                r"(?:知识库|文档|手册|标准|规范|证据)"
            )
        ),
        re.compile(
            (
                r"(?:检索|查询|查找|依据|引用)"
                r".{0,16}"
                r"(?:知识库|文档|手册|标准|规范|证据|规则)"
            )
        ),
        re.compile(
            (
                r"(?:故障原因|可能原因|为什么|恢复条件|"
                r"恢复规则|安全要求|限制条件|"
                r"排查步骤|检查步骤|预防措施)"
            )
        ),
        re.compile(
            (
                r"(?:search|retrieve|look up)"
                r".{0,24}"
                r"(?:knowledge|manual|document|evidence|rule)"
            ),
            re.IGNORECASE,
        ),
    ),
    "robot_telemetry": (
        re.compile(
            (
                r"(?:读取|查询|获取|核对|确认)"
                r".{0,16}"
                r"(?:模拟遥测|遥测|实时位置|当前位置|"
                r"当前电量|当前速度|当前任务|当前订单|"
                r"机器人状态|网络连接状态)"
            )
        ),
        re.compile(
            r"(?:模拟遥测|实时遥测)"
        ),
        re.compile(
            (
                r"(?:read|get|check)"
                r".{0,24}"
                r"(?:telemetry|current location|"
                r"battery|robot state|task state)"
            ),
            re.IGNORECASE,
        ),
    ),
    "test_case_draft": (
        re.compile(
            (
                r"(?:生成|准备|创建|起草)"
                r".{0,16}"
                r"(?:测试草案|验证草案|检查草案|"
                r"测试方案|验证方案|测试用例)"
            )
        ),
        re.compile(
            (
                r"(?:测试|验证|检查)"
                r"(?:草案|方案|用例)"
            )
        ),
        re.compile(
            (
                r"(?:draft|create|prepare)"
                r".{0,24}"
                r"(?:test case|test plan|validation plan)"
            ),
            re.IGNORECASE,
        ),
    ),
    "current_time": (
        re.compile(
            (
                r"(?:当前|现在|系统)"
                r".{0,8}"
                r"(?:时间|日期)"
            )
        ),
        re.compile(
            r"(?:现在几点|今天几号)"
        ),
        re.compile(
            (
                r"(?:current|system)"
                r".{0,12}"
                r"(?:time|date)"
            ),
            re.IGNORECASE,
        ),
    ),
}


# 明确排除某种能力的表达。
#
# 必须先检查排除规则，再检查正向意图。
#
# 例如：
#
# “不读取遥测”
#
# 同时含有“读取”和“遥测”，
# 如果只做普通关键词匹配会错误开放遥测工具。
_CAPABILITY_EXCLUSION_PATTERNS: dict[
    AgentRequiredCapability,
    tuple[
        re.Pattern[str],
        ...,
    ],
] = {
    "vision_observation": (
        re.compile(
            (
                r"(?:不|无需|不要|禁止)"
                r".{0,8}"
                r"(?:读取|观察|查看|分析|使用)"
                r".{0,8}"
                r"(?:图片|图像|照片|面板|指示灯)"
            )
        ),
    ),
    "knowledge_evidence": (
        re.compile(
            (
                r"(?:不|无需|不要|禁止)"
                r".{0,8}"
                r"(?:检索|查询|使用|引用)"
                r".{0,8}"
                r"(?:知识库|文档|手册|证据)"
            )
        ),
        re.compile(
            (
                r"(?:不|不得)"
                r".{0,8}"
                r"(?:推断|分析)"
                r".{0,8}"
                r"(?:工程原因|故障原因)"
            )
        ),
    ),
    "robot_telemetry": (
        re.compile(
            (
                r"(?:不|无需|不要|禁止)"
                r".{0,8}"
                r"(?:读取|查询|获取|使用)"
                r".{0,8}"
                r"(?:模拟遥测|遥测|实时状态)"
            )
        ),
    ),
    "test_case_draft": (
        re.compile(
            (
                r"(?:不|无需|不要|禁止)"
                r".{0,8}"
                r"(?:生成|准备|创建|起草)"
                r".{0,8}"
                r"(?:测试草案|验证草案|检查草案|"
                r"测试方案|测试用例)"
            )
        ),
    ),
    "current_time": (
        re.compile(
            (
                r"(?:不|无需|不要|禁止)"
                r".{0,8}"
                r"(?:读取|查询|使用)"
                r".{0,8}"
                r"(?:当前时间|系统时间|日期)"
            )
        ),
    ),
}


# 图片语义分面用于判断图片analysis_goal
# 与当前task_goal是否明显无关。
#
# 这里判断的是宽泛的信息类别，不执行图像识别。
_IMAGE_FACET_PATTERNS: dict[
    str,
    tuple[
        re.Pattern[str],
        ...,
    ],
] = {
    "fault_code": (
        re.compile(
            r"(?:故障码|错误码|报警码|error code|fault code)",
            re.IGNORECASE,
        ),
    ),
    "network": (
        re.compile(
            (
                r"(?:网络|联网|连接状态|network|connected|"
                r"\bnet\b|net(?=指示灯|灯|状态))"
            ),
            re.IGNORECASE,
        ),
    ),
    "task_state": (
        re.compile(
            r"(?:任务状态|暂停|继续任务|task state|paused)",
            re.IGNORECASE,
        ),
    ),
    "indicator": (
        re.compile(
            r"(?:指示灯|灯色|亮灭|led|indicator)",
            re.IGNORECASE,
        ),
    ),
    "connector": (
        re.compile(
            (
                r"(?:连接器|插头|插座|锁紧环|贴合|间隙|"
                r"\bj3\b|connector|locking ring)"
            ),
            re.IGNORECASE,
        ),
    ),
    "battery": (
        re.compile(
            r"(?:电量|电池|battery)",
            re.IGNORECASE,
        ),
    ),
    "speed": (
        re.compile(
            r"(?:速度|speed)",
            re.IGNORECASE,
        ),
    ),
    "temperature": (
        re.compile(
            r"(?:温度|temperature)",
            re.IGNORECASE,
        ),
    ),
    "numeric_panel": (
        re.compile(
            r"(?:数值|读数|百分比|numeric|value)",
            re.IGNORECASE,
        ),
    ),
    "readability": (
        re.compile(
            (
                r"(?:可读|不可读|清晰|模糊|遮挡|"
                r"readable|unreadable|blur|occlud)"
            ),
            re.IGNORECASE,
        ),
    ),
    "appearance": (
        re.compile(
            r"(?:外观|损伤|裂纹|变形|appearance|damage)",
            re.IGNORECASE,
        ),
    ),
}


# 所需能力对应的稳定原因码。
_CAPABILITY_REASON_CODES: dict[
    AgentRequiredCapability,
    AgentToolPolicyReasonCode,
] = {
    "vision_observation": (
        "vision_observation_required"
    ),
    "knowledge_evidence": (
        "knowledge_evidence_required"
    ),
    "robot_telemetry": (
        "robot_telemetry_required"
    ),
    "test_case_draft": (
        "test_case_draft_required"
    ),
    "current_time": (
        "current_time_required"
    ),
}


def _normalize_policy_text(
    value: str,
    /,
) -> str:
    """为确定性规则匹配执行轻量文本归一化。"""

    if not isinstance(value, str):
        raise TypeError(
            "工具策略文本必须是字符串"
        )

    # NFKC将全角字符等兼容形式转换成稳定形式。
    normalized = unicodedata.normalize(
        "NFKC",
        value,
    )

    # casefold()执行Unicode友好的不区分大小写转换。
    normalized = normalized.casefold()

    # 多个空白统一成一个普通空格。
    return _WHITESPACE_PATTERN.sub(
        " ",
        normalized,
    ).strip()


def _matches_any(
    text: str,
    patterns: tuple[
        re.Pattern[str],
        ...,
    ],
    /,
) -> bool:
    """判断文字是否命中任意一个已编译正则规则。"""

    return any(
        pattern.search(text) is not None
        for pattern in patterns
    )


def _capability_is_required(
    *,
    capability: AgentRequiredCapability,
    task_goal: str,
) -> bool:
    """判断任务目标是否明确要求一项能力。"""

    # 明确排除优先于正向关键词。
    if _matches_any(
        task_goal,
        _CAPABILITY_EXCLUSION_PATTERNS[
            capability
        ],
    ):
        return False

    return _matches_any(
        task_goal,
        _CAPABILITY_PATTERNS[
            capability
        ],
    )


def _extract_image_facets(
    text: str,
    /,
) -> frozenset[str]:
    """提取任务或图片分析目标涉及的宽泛视觉分面。"""

    return frozenset(
        facet
        for facet, patterns
        in _IMAGE_FACET_PATTERNS.items()
        if _matches_any(
            text,
            patterns,
        )
    )


def _has_relevant_image(
    *,
    request: AgentDiagnosisRequest,
    normalized_task_goal: str,
) -> bool:
    """判断至少一张图片与任务目标不存在明显分面冲突。

    该函数不会读取图片像素。

    它只比较：

    1. task_goal要求观察什么；
    2. 每张图片的analysis_goal声明图片准备观察什么。

    如果任意一侧没有可识别分面，
    系统采取保守策略，把图片视为可能相关，
    交给Vision确认，而不是提前误拒绝。
    """

    task_facets = _extract_image_facets(
        normalized_task_goal
    )

    for image in request.images:
        image_goal = _normalize_policy_text(
            image.analysis_goal
        )

        image_facets = _extract_image_facets(
            image_goal
        )

        # 缺少足够的分面信息时不能证明图片无关。
        if not task_facets or not image_facets:
            return True

        # 至少共享一个分面时，图片可能相关。
        if task_facets & image_facets:
            return True

    return False


def _detect_required_capabilities(
    request: AgentDiagnosisRequest,
    /,
) -> tuple[
    AgentRequiredCapability,
    ...,
]:
    """从任务目标识别按稳定顺序排列的能力集合。"""

    # task_goal缺省时，这是一个普通证据约束诊断，
    # 至少需要知识库证据。
    if request.task_goal is None:
        return (
            "knowledge_evidence",
        )

    task_goal = _normalize_policy_text(
        request.task_goal
    )

    capabilities = tuple(
        capability
        for capability
        in CAPABILITY_TO_TOOL
        if _capability_is_required(
            capability=capability,
            task_goal=task_goal,
        )
    )

    if capabilities:
        return capabilities

    # 显式task_goal没有命中专用能力时，
    # 仍按照诊断接口的默认职责使用知识库证据。
    #
    # 提示注入和高风险请求会在下一阶段先被
    # 安全分类器拦截，不会落入这个默认分支。
    return (
        "knowledge_evidence",
    )


def _tools_for_capabilities(
    capabilities: tuple[
        AgentRequiredCapability,
        ...,
    ],
    /,
) -> tuple[
    str,
    ...,
]:
    """把能力集合转换成稳定顺序的最小工具集合。"""

    required_tools = {
        CAPABILITY_TO_TOOL[capability]
        for capability in capabilities
    }

    return tuple(
        tool_name
        for tool_name
        in AGENT_READ_ONLY_TOOL_ORDER
        if tool_name in required_tools
    )


def _reason_codes_for_capabilities(
    capabilities: tuple[
        AgentRequiredCapability,
        ...,
    ],
    *,
    include_image_available: bool,
) -> tuple[
    AgentToolPolicyReasonCode,
    ...,
]:
    """构造稳定且不重复的策略原因码。"""

    reason_codes: list[
        AgentToolPolicyReasonCode
    ] = []

    if include_image_available:
        reason_codes.append(
            "image_input_available"
        )

    reason_codes.extend(
        _CAPABILITY_REASON_CODES[
            capability
        ]
        for capability in capabilities
    )

    return tuple(reason_codes)


class AgentToolPolicy:
    """为正常诊断请求计算最小只读工具集合。

    安全分类器将在后续步骤中位于本类之前。
    本类只处理已经允许进入普通能力判断的请求。
    """

    def decide(
        self,
        request: AgentDiagnosisRequest,
        /,
    ) -> AgentToolPolicyDecision:
        """分析任务能力并返回结构化工具策略决定。"""

        if not isinstance(
            request,
            AgentDiagnosisRequest,
        ):
            raise TypeError(
                "request必须是"
                "AgentDiagnosisRequest"
            )

        capabilities = (
            _detect_required_capabilities(
                request
            )
        )

        requires_vision = (
            "vision_observation"
            in capabilities
        )

        # 任务明确依赖图片，但请求没有提供图片。
        #
        # 此时不能调用Vision，也不能使用知识库
        # 猜测图片中本应读取的故障码或数值。
        if (
            requires_vision
            and not request.images
        ):
            return AgentToolPolicyDecision(
                disposition="abstained",
                required_capabilities=(
                    capabilities
                ),
                allowed_tool_names=(),
                reason_codes=(
                    "required_image_missing",
                    "no_safe_tool_required",
                ),
                public_message=(
                    "当前任务需要读取现场图片，"
                    "但请求没有提供可用图片"
                ),
                missing_information=(
                    "缺少与当前任务对应的现场图片",
                ),
            )

        normalized_task_goal = (
            _normalize_policy_text(
                request.task_goal
                or ""
            )
        )

        # 请求有图片不代表图片一定与任务相关。
        #
        # 只有任务确实需要视觉能力时才进行相关性检查。
        if (
            requires_vision
            and not _has_relevant_image(
                request=request,
                normalized_task_goal=(
                    normalized_task_goal
                ),
            )
        ):
            return AgentToolPolicyDecision(
                disposition="abstained",
                required_capabilities=(
                    capabilities
                ),
                allowed_tool_names=(),
                reason_codes=(
                    "image_not_relevant",
                    "no_safe_tool_required",
                ),
                public_message=(
                    "请求中的图片与当前观察目标"
                    "明显不相关"
                ),
                missing_information=(
                    "缺少与当前任务目标相关的现场图片",
                ),
            )

        allowed_tool_names = (
            _tools_for_capabilities(
                capabilities
            )
        )

        return AgentToolPolicyDecision(
            disposition=(
                "continue_to_planner"
            ),
            required_capabilities=(
                capabilities
            ),
            allowed_tool_names=(
                allowed_tool_names
            ),
            reason_codes=(
                _reason_codes_for_capabilities(
                    capabilities,
                    include_image_available=(
                        requires_vision
                    ),
                )
            ),
        )
