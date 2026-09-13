"""Week 6 多模态 Agent 单场景离线回归执行器。

本模块负责把一条多模态 Gold 场景和对应离线夹具，
装配到现有真实 Agent 主链中执行。

真实保留的生产逻辑包括：

1. 请求级确定性安全分类；
2. 请求级最小工具策略；
3. 图片输入适配与请求级图片 Store；
4. AgentRunner 工具循环；
5. AgentProgressReducer 进度状态机；
6. ToolRegistry 和 ToolExecutor 权限边界；
7. 最终诊断草稿校验；
8. 知识库证据白名单校验；
9. 公开响应和脱敏工具轨迹构造。

被 Fake 替换的外部或非确定性依赖包括：

1. Planner；
2. Vision Provider；
3. 知识库检索结果；
4. 模拟遥测、测试草案和系统时间工具结果。

本模块只执行一条场景并返回审计结果，
不与 Gold 比较，不计算通过率，也不生成报告。
"""

from pathlib import (
    Path,
)

from pydantic import (
    BaseModel,
)

from app.agent.evidence_store import (
    ConfirmedEvidenceStore,
)
from app.agent.executor import (
    ToolExecutor,
)
from app.agent.progress_reducer import (
    AgentProgressReducer,
)
from app.agent.registry import (
    ToolRegistry,
)
from app.agent.request_safety_classifier import (
    AgentRequestSafetyClassifier,
)
from app.agent.request_tool_policy import (
    AgentToolPolicy,
)
from app.agent.runner import (
    AgentRunner,
)
from app.agent.tools.analyze_robot_image import (
    ANALYZE_ROBOT_IMAGE_TOOL_DEFINITION,
    AnalyzeRobotImageToolHandler,
)
from app.agent.tools.current_time import (
    GET_CURRENT_TIME_TOOL_DEFINITION,
)
from app.agent.tools.draft_test_case import (
    DRAFT_TEST_CASE_TOOL_DEFINITION,
)
from app.agent.tools.robot_telemetry import (
    GET_ROBOT_TELEMETRY_TOOL_DEFINITION,
)
from app.agent.tools.search_knowledge import (
    SEARCH_KNOWLEDGE_TOOL_DEFINITION,
)
from app.agent.vision_input_store import (
    RequestVisionInputStore,
)
from app.schemas.agent import (
    ToolDefinition,
)
from app.schemas.agent_tools import (
    SearchKnowledgeToolOutput,
)
from app.schemas.multimodal_agent_evaluation import (
    MultimodalAgentEvaluationScenario,
)
from app.schemas.multimodal_offline_regression import (
    MultimodalOfflineScenarioExecution,
    MultimodalOfflineScenarioFixture,
    OfflineToolCallAudit,
    OfflineToolOutcome,
)
from app.services.agent_diagnosis_service import (
    AgentDiagnosisService,
)
from app.services.evaluation import (
    prepare_multimodal_agent_request,
)
from app.services.multimodal_offline_fakes import (
    ScriptedOfflinePlanner,
    ScriptedOfflineToolHandler,
    build_offline_fake_vision_provider,
)
from app.services.vision_input import (
    VisionInputAdapter,
)


# 离线回归固定使用独立的Planner版本。
#
# 它明确说明Planner决定来自离线Fixture，
# 不能把离线结果伪装成真实模型结果。
OFFLINE_AGENT_PLANNER_PROMPT_VERSION = (
    "offline-agent-tool-calling-v1"
)


# 保持和当前正式Agent装配相同的工具步骤上限。
#
# max_steps只计算已经完成的工具步骤，
# 不把最后一次Planner的finish决定算作工具步骤。
OFFLINE_AGENT_MAX_STEPS = 5


# Fake Planner不会访问网络，
# 但仍然经过Runner的真实超时边界。
#
# Fixture中的timeout会直接抛出TimeoutError，
# 因此不需要真的等待30秒才能测试超时分支。
OFFLINE_PLANNER_TIMEOUT_SECONDS = 30.0


# 保持正式Agent当前的连续工具失败限制。
#
# 连续两次工具失败后，Runner应安全中止，
# 防止Agent无意义地反复调用失败工具。
OFFLINE_MAX_CONSECUTIVE_TOOL_FAILURES = 2


# 非Vision工具继续使用已有正式ToolDefinition。
#
# 使用正式定义可以继续验证：
#
# 1. Planner可见的工具说明；
# 2. 输入Pydantic模型；
# 3. 输出Pydantic模型；
# 4. read_only和risk_level；
# 5. ToolExecutor超时边界。
_NON_VISION_TOOL_DEFINITIONS: tuple[
    ToolDefinition,
    ...,
] = (
    SEARCH_KNOWLEDGE_TOOL_DEFINITION,
    GET_ROBOT_TELEMETRY_TOOL_DEFINITION,
    DRAFT_TEST_CASE_TOOL_DEFINITION,
    GET_CURRENT_TIME_TOOL_DEFINITION,
)


class _EvidenceRecordingOfflineSearchHandler:
    """为离线知识工具补充真实证据白名单写入。

    ScriptedOfflineToolHandler只负责返回Fixture中的
    预设字典，不知道知识库工具的业务语义。

    但正式SearchKnowledgeToolHandler在返回引用前，
    还会把引用写入当前请求的ConfirmedEvidenceStore。

    最终DiagnosisReport只允许使用Store中已经确认的
    Chunk ID。因此离线执行也必须保留这一步，
    否则最终引用校验会错误拒绝所有Fake知识证据。
    """

    def __init__(
        self,
        *,
        scripted_handler: (
            ScriptedOfflineToolHandler
        ),
        evidence_store: ConfirmedEvidenceStore,
    ) -> None:
        """保存Fake结果处理器和当前请求的证据Store。"""

        if not isinstance(
            scripted_handler,
            ScriptedOfflineToolHandler,
        ):
            raise TypeError(
                "scripted_handler必须是"
                "ScriptedOfflineToolHandler"
            )

        if (
            scripted_handler.tool_name
            != "search_knowledge"
        ):
            raise ValueError(
                "scripted_handler必须负责"
                "search_knowledge"
            )

        if not isinstance(
            evidence_store,
            ConfirmedEvidenceStore,
        ):
            raise TypeError(
                "evidence_store必须是"
                "ConfirmedEvidenceStore"
            )

        self._scripted_handler = (
            scripted_handler
        )
        self._evidence_store = (
            evidence_store
        )

    async def __call__(
        self,
        validated_input: BaseModel,
    ) -> BaseModel | None:
        """取得Fake检索结果并登记其中的确认引用。

        调用顺序与正式知识工具一致：

        1. 取得原始结果；
        2. 如果为空则直接返回None；
        3. 使用SearchKnowledgeToolOutput校验；
        4. 将合法引用原子写入证据Store；
        5. 返回校验后的结构化输出。

        ToolExecutor之后还会依据正式ToolDefinition
        再进行一次输出校验，形成纵深防御。
        """

        raw_output = await (
            self._scripted_handler(
                validated_input
            )
        )

        # None表示工具正常执行，
        # 但没有取得足够证据。
        #
        # 外层ToolExecutor会将它转换成：
        #
        # status=empty
        # error_code=empty_result
        if raw_output is None:
            return None

        # Fixture中的output是普通dict。
        #
        # 先将它校验成正式工具输出模型，
        # 防止非法引用进入证据白名单。
        validated_output = (
            SearchKnowledgeToolOutput
            .model_validate(raw_output)
        )

        # record()会复制并原子记录所有引用。
        #
        # 任意引用不合法时，整组引用都不会留下
        # 部分写入结果。
        self._evidence_store.record(
            validated_output.citations
        )

        return validated_output


def _group_tool_outcomes(
    fixture: MultimodalOfflineScenarioFixture,
    /,
) -> dict[
    str,
    tuple[OfflineToolOutcome, ...],
]:
    """按照工具名拆分非Vision工具结果队列。

    Fixture中的tool_outcomes按照Planner全局调用顺序保存。

    每个ScriptedOfflineToolHandler只负责一个工具，
    因此执行器需要将全局序列拆成按工具名排列的队列。

    列表推导仍会保留原始相对顺序。例如第一次和第三次
    都调用search_knowledge时，它们在检索队列中的顺序
    仍然是第一次在前、第三次在后。
    """

    return {
        definition.name: tuple(
            outcome
            for outcome in fixture.tool_outcomes
            if (
                outcome.tool_name
                == definition.name
            )
        )
        for definition
        in _NON_VISION_TOOL_DEFINITIONS
    }


def _validate_fixture_image_sources(
    *,
    scenario: MultimodalAgentEvaluationScenario,
    fixture: MultimodalOfflineScenarioFixture,
) -> None:
    """禁止Fixture为场景以外的图片伪造Vision结果。

    Fixture中的Vision结果必须使用当前Gold场景已经声明的
    图片SHA-256。

    这里只检查Fixture结果是否属于场景图片，不要求每张场景
    图片都配置结果，因为：

    1. 场景可以携带与任务无关、不会调用Vision的图片；
    2. 高风险请求可能在读取图片前被安全策略终止；
    3. 测试可能有意让Planner使用错误image_ref，
       以验证工具输入或空结果路径。
    """

    scenario_image_hashes = {
        image.image_sha256
        for image in scenario.images
    }

    fixture_image_hashes = {
        outcome.image_sha256
        for outcome in fixture.vision_outcomes
    }

    unknown_image_hashes = sorted(
        fixture_image_hashes
        - scenario_image_hashes
    )

    if unknown_image_hashes:
        raise ValueError(
            "fixture.vision_outcomes包含"
            "不属于当前场景的图片SHA-256："
            + ", ".join(unknown_image_hashes)
        )


class MultimodalOfflineScenarioExecutor:
    """执行一条多模态Agent离线回归场景。

    一个Executor实例可以依次执行多条场景，
    但每次execute()都会重新创建：

    1. ConfirmedEvidenceStore；
    2. RequestVisionInputStore；
    3. Fake Planner；
    4. Fake Vision Provider；
    5. Fake工具Handler；
    6. ToolRegistry；
    7. ToolExecutor；
    8. AgentRunner；
    9. AgentDiagnosisService。

    因此不同场景之间不会共享证据、图片、工具调用历史
    或Planner轮次。
    """

    def __init__(
        self,
        *,
        project_root: Path,
        vision_input_adapter: VisionInputAdapter,
    ) -> None:
        """保存项目根目录和统一图片输入适配器。"""

        if not isinstance(project_root, Path):
            raise TypeError(
                "project_root必须是pathlib.Path"
            )

        try:
            resolved_project_root = (
                project_root.resolve(
                    strict=True
                )
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                "project_root不存在："
                f"{project_root}"
            ) from exc

        if not resolved_project_root.is_dir():
            raise NotADirectoryError(
                "project_root必须指向目录："
                f"{project_root}"
            )

        if not isinstance(
            vision_input_adapter,
            VisionInputAdapter,
        ):
            raise TypeError(
                "vision_input_adapter必须是"
                "VisionInputAdapter"
            )

        self._project_root = (
            resolved_project_root
        )
        self._vision_input_adapter = (
            vision_input_adapter
        )

    async def execute(
        self,
        *,
        scenario: MultimodalAgentEvaluationScenario,
        fixture: MultimodalOfflineScenarioFixture,
    ) -> MultimodalOfflineScenarioExecution:
        """执行一条场景并返回响应及Fake调用审计。

        本方法不会读取真实模型配置，
        也不会创建OpenAI、Embedding或Chroma客户端。
        """

        if not isinstance(
            scenario,
            MultimodalAgentEvaluationScenario,
        ):
            raise TypeError(
                "scenario必须是"
                "MultimodalAgentEvaluationScenario"
            )

        if not isinstance(
            fixture,
            MultimodalOfflineScenarioFixture,
        ):
            raise TypeError(
                "fixture必须是"
                "MultimodalOfflineScenarioFixture"
            )

        if (
            scenario.scenario_id
            != fixture.scenario_id
        ):
            raise ValueError(
                "scenario与fixture的"
                "scenario_id必须一致"
            )

        _validate_fixture_image_sources(
            scenario=scenario,
            fixture=fixture,
        )

        # 安全分类器只检查已经可见的请求文本，
        # 不需要图片像素。
        #
        # 如果请求在进入Planner前就应转人工审核，
        # 离线执行器不会读取或解码场景图片。
        # AgentDiagnosisService内部还会使用同一分类器
        # 再次执行正式的安全边界判断。
        safety_classifier = (
            AgentRequestSafetyClassifier()
        )
        safety_decision = (
            safety_classifier.classify(
                scenario.request
            )
        )

        if safety_decision is not None:
            # Gold中的request按契约不保存Base64图片。
            #
            # 安全前置已经决定停止，
            # 因而无需读取本地图片并构造图片载荷。
            prepared_request = (
                scenario.request
            )
        else:
            # 读取场景索引中的本地图片，
            # 校验文件路径、SHA-256、格式、尺寸和资源限制，
            # 再构造真实AgentDiagnosisRequest。
            prepared_request = (
                prepare_multimodal_agent_request(
                    scenario=scenario,
                    project_root=(
                        self._project_root
                    ),
                    vision_input_adapter=(
                        self
                        ._vision_input_adapter
                    ),
                )
            )

        # 以下Store必须每条场景单独创建。
        #
        # 证据或图片一旦跨场景复用，
        # 后一条场景就可能引用前一条场景的数据。
        evidence_store = (
            ConfirmedEvidenceStore()
        )
        vision_input_store = (
            RequestVisionInputStore()
        )

        planner = ScriptedOfflinePlanner(
            turns=fixture.planner_turns
        )

        vision_provider = (
            build_offline_fake_vision_provider(
                fixture
            )
        )

        grouped_tool_outcomes = (
            _group_tool_outcomes(fixture)
        )

        # 每个普通工具都创建独立Fake Handler。
        #
        # 即使某个工具没有预设结果也会创建空队列。
        # 如果Planner意外调用它，Handler会产生稳定的
        # “结果队列耗尽”异常，而不是静默成功。
        scripted_handlers: dict[
            str,
            ScriptedOfflineToolHandler,
        ] = {
            definition.name: (
                ScriptedOfflineToolHandler(
                    tool_name=definition.name,
                    outcomes=(
                        grouped_tool_outcomes[
                            definition.name
                        ]
                    ),
                )
            )
            for definition
            in _NON_VISION_TOOL_DEFINITIONS
        }

        registry = ToolRegistry()

        # Vision仍使用正式工具Handler。
        #
        # 只有最外层Provider被换成Fake，
        # 图片Store、analysis_goal替换、有限重试、
        # 摘要匹配和输出校验继续使用生产实现。
        registry.register(
            definition=(
                ANALYZE_ROBOT_IMAGE_TOOL_DEFINITION
            ),
            handler=(
                AnalyzeRobotImageToolHandler(
                    provider=vision_provider,
                    input_store=(
                        vision_input_store
                    ),
                )
            ),
        )

        for definition in (
            _NON_VISION_TOOL_DEFINITIONS
        ):
            scripted_handler = (
                scripted_handlers[
                    definition.name
                ]
            )

            if (
                definition.name
                == "search_knowledge"
            ):
                # 知识工具需要额外登记真实引用白名单。
                handler = (
                    _EvidenceRecordingOfflineSearchHandler(
                        scripted_handler=(
                            scripted_handler
                        ),
                        evidence_store=(
                            evidence_store
                        ),
                    )
                )
            else:
                handler = scripted_handler

            registry.register(
                definition=definition,
                handler=handler,
            )

        tool_executor = ToolExecutor(
            registry=registry
        )

        runner = AgentRunner(
            planner=planner,
            executor=tool_executor,
            max_steps=(
                OFFLINE_AGENT_MAX_STEPS
            ),
            planner_timeout_seconds=(
                OFFLINE_PLANNER_TIMEOUT_SECONDS
            ),
            max_consecutive_tool_failures=(
                OFFLINE_MAX_CONSECUTIVE_TOOL_FAILURES
            ),
            progress_reducer=(
                AgentProgressReducer()
            ),
        )

        diagnosis_service = (
            AgentDiagnosisService(
                runner=runner,
                evidence_store=evidence_store,
                vision_input_adapter=(
                    self._vision_input_adapter
                ),
                vision_input_store=(
                    vision_input_store
                ),
                safety_classifier=(
                    safety_classifier
                ),
                tool_policy=AgentToolPolicy(),
                planner_prompt_version=(
                    OFFLINE_AGENT_PLANNER_PROMPT_VERSION
                ),
            )
        )

        # 使用稳定request_id，方便相同场景的多次离线结果
        # 在测试和报告中进行确定性比较。
        response = await (
            diagnosis_service.diagnose(
                prepared_request,
                request_id=(
                    f"offline-"
                    f"{scenario.scenario_id}"
                ),
            )
        )

        planned_vision_image_sha256s = (
            tuple(
                outcome.image_sha256
                for outcome
                in fixture.vision_outcomes
            )
        )

        # FakeVisionCall不保存图片正文，
        # 只保存图片摘要、分析目标摘要和尺寸等审计数据。
        vision_called_image_sha256s = (
            tuple(
                call.source_image_sha256
                for call in vision_provider.calls
            )
        )

        tool_call_audits = tuple(
            OfflineToolCallAudit(
                tool_name=definition.name,
                planned_outcome_count=len(
                    grouped_tool_outcomes[
                        definition.name
                    ]
                ),
                actual_call_count=(
                    scripted_handlers[
                        definition.name
                    ].call_count
                ),
            )
            for definition
            in _NON_VISION_TOOL_DEFINITIONS
        )

        # Vision结果按图片SHA索引。
        #
        # 同一张图片因有限重试被调用两次时，
        # 调用次数是2，但仍然只消费同一项图片结果。
        fixture_fully_consumed = (
            len(fixture.planner_turns)
            == planner.call_count
            and set(
                planned_vision_image_sha256s
            )
            == set(
                vision_called_image_sha256s
            )
            and all(
                audit.fully_consumed
                for audit in tool_call_audits
            )
        )

        return (
            MultimodalOfflineScenarioExecution(
                fixture_version=(
                    fixture.fixture_version
                ),
                scenario_id=(
                    scenario.scenario_id
                ),
                response=response,
                planner_turn_count=len(
                    fixture.planner_turns
                ),
                planner_call_count=(
                    planner.call_count
                ),
                planner_exposed_tool_names=(
                    planner.exposed_tool_names
                ),
                vision_outcome_count=len(
                    fixture.vision_outcomes
                ),
                planned_vision_image_sha256s=(
                    planned_vision_image_sha256s
                ),
                vision_call_count=(
                    vision_provider.call_count
                ),
                vision_called_image_sha256s=(
                    vision_called_image_sha256s
                ),
                tool_call_audits=(
                    tool_call_audits
                ),
                fixture_fully_consumed=(
                    fixture_fully_consumed
                ),
            )
        )