"""Robot Diagnostic Agent允许注册的具体工具。

本包中的工具必须满足以下安全要求：

1. 具有明确的Pydantic输入和输出契约；
2. 通过应用代码显式注册；
3. 不根据LLM返回的字符串动态导入函数；
4. 不执行任意Shell、Python或设备控制命令；
5. Week 4阶段只能注册只读、非高风险工具。
"""

from app.agent.tools.current_time import (
    GET_CURRENT_TIME_TOOL_DEFINITION,
    GetCurrentTimeToolHandler,
    SystemUtcClock,
    UtcClock,
)
from app.agent.tools.analyze_robot_image import (
    ANALYZE_ROBOT_IMAGE_TOOL_DEFINITION,
    AnalyzeRobotImageToolHandler,
)
from app.agent.tools.draft_test_case import (
    DRAFT_TEST_CASE_TOOL_DEFINITION,
    DraftTestCaseToolHandler,
)
from app.agent.tools.robot_telemetry import (
    GET_ROBOT_TELEMETRY_TOOL_DEFINITION,
    GetRobotTelemetryToolHandler,
    InMemoryRobotTelemetryStore,
)
from app.agent.tools.search_knowledge import (
    SEARCH_KNOWLEDGE_TOOL_DEFINITION,
    SearchKnowledgeToolHandler,
)


# __all__明确声明本包对外公开的名称。
#
# 其他模块可以从app.agent.tools统一导入，
# 不需要了解每个工具分别位于哪个子模块。
__all__ = [
    "ANALYZE_ROBOT_IMAGE_TOOL_DEFINITION",
    "AnalyzeRobotImageToolHandler",
    "GET_CURRENT_TIME_TOOL_DEFINITION",
    "GetCurrentTimeToolHandler",
    "SystemUtcClock",
    "UtcClock",
    "DRAFT_TEST_CASE_TOOL_DEFINITION",
    "DraftTestCaseToolHandler",
    "GET_ROBOT_TELEMETRY_TOOL_DEFINITION",
    "InMemoryRobotTelemetryStore",
    "GetRobotTelemetryToolHandler",
    "SEARCH_KNOWLEDGE_TOOL_DEFINITION",
    "SearchKnowledgeToolHandler",
]
