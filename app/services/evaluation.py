"""读取、执行并汇总检索和Agent评测。"""

import json
import logging
from collections.abc import (
    Callable,
    Mapping,
    Sequence,
)
from base64 import (
    b64encode,
)
from hashlib import (
    sha256,
)
from time import (
    perf_counter,
)

import httpx

from pathlib import (
    Path,
    PurePosixPath,
)

# ValidationError 表示数据没有通过 Pydantic 模型校验。
from pydantic import ValidationError

from app.schemas.evaluation import GoldQuestion


# AgentEvaluationScenario定义Week 4固定Agent评测场景。
#
# 加载器会把每一行JSON转换成该Pydantic模型，
# 在发出任何真实Agent请求之前完成数据校验。
from app.schemas.agent_evaluation import (
    AgentEvaluationScenario,
)
from app.schemas.agent_api import (
    AgentDiagnosisRequest,
    AgentDiagnosisResponse,
)
from app.schemas.multimodal_agent_evaluation import (
    MultimodalAgentBatchExecution,
    MultimodalAgentEvaluationScenario,
    MultimodalAgentScenarioEvaluation,
    MultimodalAgentTrace,
    MultimodalAgentTraceArtifact,
    MultimodalEvaluationFixRecord,
)
from app.schemas.vision import (
    VisionImagePayload,
)
from app.services.vision_input import (
    VisionInputAdapter,
)
from app.errors import (
    ApplicationError,
    VisionInputValidationError,
)
from app.services.agent_evaluation import (
    build_multimodal_agent_trace,
    build_multimodal_agent_evaluation_report,
    evaluate_multimodal_agent_request_failure,
    evaluate_multimodal_agent_response,
)
from app.services.visual_equivalence_judge import (
    VisualEquivalenceJudge,
    VisualEquivalenceJudgeResult,
    build_visual_equivalence_judge_note,
)


class EvaluationDataError(ValueError):
    """评测数据文件格式或内容不符合要求。"""


# 只记录场景编号和后备选择，不记录请求正文、图片或模型输出。
logger = logging.getLogger(__name__)


# Week 5要求正式批量评测至少包含30条场景。
MIN_MULTIMODAL_AGENT_SCENARIO_COUNT = 30


def load_gold_questions(
    path: Path,
) -> list[GoldQuestion]:
    """逐行读取 JSONL 并返回经过校验的评测问题。"""

    if path.suffix.lower() != ".jsonl":
        raise EvaluationDataError(
            "评测问题文件必须使用 .jsonl 扩展名"
        )

    if not path.is_file():
        raise FileNotFoundError(
            f"评测问题文件不存在：{path}"
        )

    questions: list[GoldQuestion] = []

    # 用于检查 question_id 是否重复。
    seen_question_ids: set[str] = set()

    # 按文本模式逐行读取，
    # 不需要一次把整个评测文件加载到内存。
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        # enumerate(..., start=1) 使行号与编辑器一致。
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            stripped_line = line.strip()

            # JSONL 中间出现空行可能意味着数据被意外编辑，
            # 因此这里选择明确报错。
            if not stripped_line:
                raise EvaluationDataError(
                    f"评测文件第 {line_number} 行为空"
                )

            try:
                # loads() 只解析当前一行。
                raw_question = json.loads(
                    stripped_line
                )
            except json.JSONDecodeError as exc:
                raise EvaluationDataError(
                    f"评测文件第 {line_number} 行不是合法 JSON"
                ) from exc

            try:
                # model_validate() 将 dict 转换成 GoldQuestion，
                # 并执行字段约束和 model_validator。
                question = GoldQuestion.model_validate(
                    raw_question
                )
            except ValidationError as exc:
                raise EvaluationDataError(
                    f"评测文件第 {line_number} 行不符合问题契约"
                ) from exc

            if question.question_id in seen_question_ids:
                raise EvaluationDataError(
                    "评测问题编号重复："
                    f"{question.question_id}"
                )

            seen_question_ids.add(
                question.question_id
            )
            questions.append(question)

    if not questions:
        raise EvaluationDataError(
            "评测问题文件不能为空"
        )

    return questions


def load_agent_evaluation_scenarios(
    path: Path,
) -> list[AgentEvaluationScenario]:
    """逐行读取并校验Agent固定评测场景JSONL。

    本函数只负责加载和验证数据，不负责：

    1. 调用Agent API；
    2. 执行工具；
    3. 计算评分；
    4. 生成报告。

    只有整个文件全部通过校验后，
    调用方才会取得完整的场景列表。
    """

    # suffix取得路径的最后一个扩展名。
    #
    # lower()允许.JSONL等大小写写法，
    # 但仍然拒绝普通.json文件。
    if path.suffix.lower() != ".jsonl":
        raise EvaluationDataError(
            "Agent评测场景文件必须使用"
            " .jsonl 扩展名"
        )

    # is_file()同时确认路径存在并且是普通文件。
    #
    # 一个同名目录不能被当成评测文件读取。
    if not path.is_file():
        raise FileNotFoundError(
            "Agent评测场景文件不存在："
            f"{path}"
        )

    scenarios: list[
        AgentEvaluationScenario
    ] = []

    # set用于记录已经成功加载的场景编号。
    #
    # set查找通常是常数时间，适合逐行检查重复ID。
    seen_scenario_ids: set[str] = set()

    # with语句保证正常结束或发生异常时，
    # 文件句柄都会被自动关闭。
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        # start=1使这里的行号与编辑器显示一致。
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            stripped_line = line.strip()

            # JSONL规定每一行是一条完整JSON记录。
            #
            # 中间空行可能意味着场景被误删，
            # 因此不能静默跳过。
            if not stripped_line:
                raise EvaluationDataError(
                    "Agent评测场景文件"
                    f"第 {line_number} 行为空"
                )

            try:
                # json.loads()只解析当前这一行，
                # 不会把整个JSONL当成一个JSON数组。
                raw_scenario = json.loads(
                    stripped_line
                )
            except json.JSONDecodeError as exc:
                # from exc保留底层JSON解析异常作为原因，
                # 同时对外提供包含行号的稳定错误。
                raise EvaluationDataError(
                    "Agent评测场景文件"
                    f"第 {line_number} 行"
                    "不是合法 JSON"
                ) from exc

            try:
                # model_validate()会执行：
                #
                # 1. 顶层字段类型和长度校验；
                # 2. request嵌套请求契约校验；
                # 3. 工具集合关系校验；
                # 4. 执行状态和终止原因关系校验；
                # 5. 完成、拒答和证据关系校验。
                scenario = (
                    AgentEvaluationScenario
                    .model_validate(
                        raw_scenario
                    )
                )
            except ValidationError as exc:
                raise EvaluationDataError(
                    "Agent评测场景文件"
                    f"第 {line_number} 行不符合"
                    "Agent评测场景契约"
                ) from exc

            if (
                scenario.scenario_id
                in seen_scenario_ids
            ):
                raise EvaluationDataError(
                    "Agent评测场景编号重复: "
                    f"{scenario.scenario_id}"
                )

            seen_scenario_ids.add(
                scenario.scenario_id
            )
            scenarios.append(
                scenario
            )

    # 一个空文件虽然没有语法错误，
    # 但无法产生任何评测分母和指标。
    if not scenarios:
        raise EvaluationDataError(
            "Agent评测场景文件不能为空"
        )

    # list顺序与JSONL中的行顺序完全一致，
    # 后续报告也会保持该稳定顺序。
    return scenarios


def load_multimodal_agent_evaluation_scenarios(
    path: Path,
) -> list[
    MultimodalAgentEvaluationScenario
]:
    """逐行读取并校验多模态Agent评测场景JSONL。

    本函数验证场景索引数据，包括图片相对路径、
    SHA-256格式、视觉预期、信息来源和安全要求。
    它不会读取图片字节，也不会调用Agent API。
    """

    if not isinstance(path, Path):
        raise TypeError(
            "path必须是pathlib.Path"
        )

    if path.suffix.lower() != ".jsonl":
        raise EvaluationDataError(
            "多模态Agent评测场景文件必须使用"
            " .jsonl 扩展名"
        )

    if not path.is_file():
        raise FileNotFoundError(
            "多模态Agent评测场景文件不存在："
            f"{path}"
        )

    scenarios: list[
        MultimodalAgentEvaluationScenario
    ] = []
    seen_scenario_ids: set[str] = set()

    # JSONL逐行解析可以在错误信息中保留准确行号，
    # 也不会要求把整个文件先解析成一个JSON数组。
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            stripped_line = line.strip()

            if not stripped_line:
                raise EvaluationDataError(
                    "多模态Agent评测场景文件"
                    f"第 {line_number} 行为空"
                )

            try:
                raw_scenario = json.loads(
                    stripped_line
                )
            except json.JSONDecodeError as exc:
                raise EvaluationDataError(
                    "多模态Agent评测场景文件"
                    f"第 {line_number} 行"
                    "不是合法 JSON"
                ) from exc

            try:
                scenario = (
                    MultimodalAgentEvaluationScenario
                    .model_validate(
                        raw_scenario
                    )
                )
            except ValidationError as exc:
                raise EvaluationDataError(
                    "多模态Agent评测场景文件"
                    f"第 {line_number} 行不符合"
                    "多模态Agent评测场景契约"
                ) from exc

            if (
                scenario.scenario_id
                in seen_scenario_ids
            ):
                raise EvaluationDataError(
                    "多模态Agent评测场景编号重复: "
                    f"{scenario.scenario_id}"
                )

            seen_scenario_ids.add(
                scenario.scenario_id
            )
            scenarios.append(scenario)

    if not scenarios:
        raise EvaluationDataError(
            "多模态Agent评测场景文件不能为空"
        )

    # 返回顺序与JSONL行顺序一致，
    # 后续逐条API调用和报告结果也保持相同顺序。
    return scenarios


def prepare_multimodal_agent_request(
    *,
    scenario: MultimodalAgentEvaluationScenario,
    project_root: Path,
    vision_input_adapter: VisionInputAdapter,
) -> AgentDiagnosisRequest:
    """读取并验证场景图片，构造真实Agent API请求。

    返回对象中的Base64使用SecretStr保存，
    普通repr和model_dump()不会包含图片正文。
    """

    if not isinstance(
        scenario,
        MultimodalAgentEvaluationScenario,
    ):
        raise TypeError(
            "scenario必须是"
            "MultimodalAgentEvaluationScenario"
        )

    if not isinstance(project_root, Path):
        raise TypeError(
            "project_root必须是pathlib.Path"
        )

    if not isinstance(
        vision_input_adapter,
        VisionInputAdapter,
    ):
        raise TypeError(
            "vision_input_adapter必须是"
            "VisionInputAdapter"
        )

    try:
        resolved_root = project_root.resolve(
            strict=True
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "项目根目录不存在："
            f"{project_root}"
        ) from exc

    if not resolved_root.is_dir():
        raise NotADirectoryError(
            "project_root必须指向项目目录："
            f"{project_root}"
        )

    prepared_images: list[
        VisionImagePayload
    ] = []

    for image_index, image in enumerate(
        scenario.images,
        start=1,
    ):
        # 场景Schema已经要求正斜杠相对路径，
        # 这里再把PurePosixPath各部分交给本机Path。
        relative_path = PurePosixPath(
            image.image_path
        )
        candidate_path = resolved_root.joinpath(
            *relative_path.parts
        )

        try:
            resolved_image_path = (
                candidate_path.resolve(
                    strict=True
                )
            )
        except FileNotFoundError as exc:
            raise EvaluationDataError(
                "多模态评测图片不存在："
                f"scenario_id={scenario.scenario_id}，"
                f"image_index={image_index}，"
                f"path={image.image_path}"
            ) from exc

        # resolve()会展开符号链接。
        # 即使JSONL路径看似位于data目录，
        # 最终目标也不能逃出当前项目根目录。
        if not resolved_image_path.is_relative_to(
            resolved_root
        ):
            raise EvaluationDataError(
                "多模态评测图片解析后超出项目目录："
                f"scenario_id={scenario.scenario_id}，"
                f"image_index={image_index}"
            )

        if not resolved_image_path.is_file():
            raise EvaluationDataError(
                "多模态评测图片路径不是普通文件："
                f"scenario_id={scenario.scenario_id}，"
                f"image_index={image_index}"
            )

        image_bytes = (
            resolved_image_path.read_bytes()
        )
        actual_sha256 = sha256(
            image_bytes
        ).hexdigest()

        if actual_sha256 != image.image_sha256:
            raise EvaluationDataError(
                "多模态评测图片SHA-256与Gold不一致："
                f"scenario_id={scenario.scenario_id}，"
                f"image_index={image_index}"
            )

        image_base64 = b64encode(
            image_bytes
        ).decode("ascii")
        payload = VisionImagePayload(
            mime_type=image.mime_type,
            encoding="base64",
            image_base64=image_base64,
            analysis_goal=image.analysis_goal,
            detail=image.detail,
        )

        try:
            # 与服务端使用同一适配器，
            # 在联网前检查真实格式、尺寸和资源上限。
            vision_input_adapter.adapt(
                payload
            )
        except VisionInputValidationError as exc:
            raise EvaluationDataError(
                "多模态评测图片未通过输入适配："
                f"scenario_id={scenario.scenario_id}，"
                f"image_index={image_index}"
            ) from exc

        prepared_images.append(payload)

    # 重新走AgentDiagnosisRequest.model_validate()，
    # 不能用model_copy(update=...)绕过字段验证。
    request_data = scenario.request.model_dump(
        exclude={"images"},
        exclude_none=True,
    )
    request_data["images"] = prepared_images

    return AgentDiagnosisRequest.model_validate(
        request_data
    )


def serialize_multimodal_agent_request_for_http(
    request: AgentDiagnosisRequest,
) -> dict[str, object]:
    """把已验证请求转换成仅供HTTP发送的JSON字典。

    VisionImagePayload默认排除Base64字段，
    因此只有这个受控边界会显式读取SecretStr。
    调用方不得记录、缓存或写出返回字典。
    """

    if not isinstance(
        request,
        AgentDiagnosisRequest,
    ):
        raise TypeError(
            "request必须是AgentDiagnosisRequest"
        )

    request_body = request.model_dump(
        exclude={"images"},
        exclude_none=True,
    )
    request_body["images"] = [
        {
            "mime_type": image.mime_type,
            "encoding": image.encoding,
            "image_base64": (
                image.image_base64
                .get_secret_value()
            ),
            "analysis_goal": (
                image.analysis_goal
            ),
            "detail": image.detail,
        }
        for image in request.images
    ]

    return request_body


def _get_multimodal_evaluation_request_id(
    response: httpx.Response,
) -> str | None:
    """从响应头读取经过清理的请求追踪编号。

    HTTP Header名称不区分大小写。缺少请求编号或值只有
    空白时返回None，不生成可能被误认为真实编号的占位值。
    """

    request_id = response.headers.get(
        "X-Request-ID"
    )

    if request_id is None:
        return None

    cleaned_request_id = request_id.strip()

    return cleaned_request_id or None


def _calculate_multimodal_request_latency_ms(
    started_at: float,
) -> float:
    """根据单调时钟起点计算非负毫秒延迟。"""

    # perf_counter()是单调的高精度计时器，不受系统时间
    # 手工修改或时间同步影响，适合测量请求耗时。
    return max(
        (perf_counter() - started_at) * 1000.0,
        0.0,
    )


async def evaluate_multimodal_agent_scenario_via_api(
    *,
    scenario: MultimodalAgentEvaluationScenario,
    client: httpx.AsyncClient,
    api_url: str,
    project_root: Path,
    vision_input_adapter: VisionInputAdapter,
    trace_consumer: Callable[
        [MultimodalAgentTrace],
        None,
    ] | None = None,
    visual_equivalence_judge: (
        VisualEquivalenceJudge | None
    ) = None,
) -> MultimodalAgentScenarioEvaluation:
    """执行一条真实多模态Agent场景并返回确定性评分。

    成功路径：

    Gold场景
    → 准备并验证图片请求
    → POST Agent API
    → 校验HTTP、JSON和AgentDiagnosisResponse
    → 多模态确定性评分。

    传输、HTTP或响应校验失败不会中断整批评测，
    而是转换成不含虚假观察的失败评分结果。
    Gold数据或本地图片错误仍然直接抛出，避免把评测集
    配置错误误报成线上Agent失败。
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
        client,
        httpx.AsyncClient,
    ):
        raise TypeError(
            "client必须是httpx.AsyncClient"
        )

    if (
        not isinstance(api_url, str)
        or not api_url.strip()
    ):
        raise ValueError(
            "api_url不能为空"
        )

    if (
        trace_consumer is not None
        and not callable(trace_consumer)
    ):
        raise TypeError(
            "trace_consumer必须可调用或为None"
        )

    if (
        visual_equivalence_judge is not None
        and not callable(
            getattr(
                visual_equivalence_judge,
                "judge_equivalence",
                None,
            )
        )
    ):
        raise TypeError(
            "visual_equivalence_judge必须实现"
            "judge_equivalence()或为None"
        )

    cleaned_api_url = api_url.strip()

    # 图片路径、哈希、真实格式和资源限制必须在联网前
    # 验证。数据集错误应立即终止，而不是消耗API调用。
    request = prepare_multimodal_agent_request(
        scenario=scenario,
        project_root=project_root,
        vision_input_adapter=(
            vision_input_adapter
        ),
    )
    request_body = (
        serialize_multimodal_agent_request_for_http(
            request
        )
    )

    # 延迟口径从发出HTTP请求前开始，到完整响应被读取后
    # 结束，不包含本地图片读取、哈希和Base64准备时间。
    started_at = perf_counter()

    try:
        # json=让httpx完成JSON序列化、UTF-8编码和
        # Content-Type设置。返回前响应正文已被完整读取。
        http_response = await client.post(
            cleaned_api_url,
            json=request_body,
        )
    except httpx.TimeoutException:
        return evaluate_multimodal_agent_request_failure(
            scenario=scenario,
            http_status_code=None,
            request_id=None,
            request_error=(
                "多模态Agent API请求超时"
            ),
            latency_ms=(
                _calculate_multimodal_request_latency_ms(
                    started_at
                )
            ),
        )
    except httpx.RequestError:
        # 不保存原始网络异常文本，避免在报告中泄漏
        # 内部主机名、代理、端口或网络拓扑。
        return evaluate_multimodal_agent_request_failure(
            scenario=scenario,
            http_status_code=None,
            request_id=None,
            request_error=(
                "无法连接多模态Agent API"
            ),
            latency_ms=(
                _calculate_multimodal_request_latency_ms(
                    started_at
                )
            ),
        )

    latency_ms = (
        _calculate_multimodal_request_latency_ms(
            started_at
        )
    )
    header_request_id = (
        _get_multimodal_evaluation_request_id(
            http_response
        )
    )

    # 非2xx仍然是真实HTTP响应，所以保留状态码和
    # 响应头中的追踪编号，但绝不保存错误响应正文。
    if not (
        200 <= http_response.status_code < 300
    ):
        return evaluate_multimodal_agent_request_failure(
            scenario=scenario,
            http_status_code=(
                http_response.status_code
            ),
            request_id=header_request_id,
            request_error=(
                "多模态Agent API返回错误状态："
                f"{http_response.status_code}"
            ),
            latency_ms=latency_ms,
        )

    try:
        # 合法HTTP状态不代表响应正文一定是合法JSON。
        raw_response = http_response.json()
    except ValueError:
        return evaluate_multimodal_agent_request_failure(
            scenario=scenario,
            http_status_code=(
                http_response.status_code
            ),
            request_id=header_request_id,
            request_error=(
                "多模态Agent API返回非法JSON"
            ),
            latency_ms=latency_ms,
        )

    try:
        # model_validate()拒绝多余字段、错误嵌套结构、
        # 非法轨迹、无效引用和不一致的诊断状态。
        api_response = (
            AgentDiagnosisResponse.model_validate(
                raw_response
            )
        )
    except ValidationError:
        return evaluate_multimodal_agent_request_failure(
            scenario=scenario,
            http_status_code=(
                http_response.status_code
            ),
            request_id=header_request_id,
            request_error=(
                "多模态Agent API响应不符合数据契约"
            ),
            latency_ms=latency_ms,
        )

    # 响应头与正文都提供request_id时必须一致，
    # 否则报告无法可靠关联服务端日志与公开响应。
    if (
        header_request_id is not None
        and header_request_id
        != api_response.request_id
    ):
        return evaluate_multimodal_agent_request_failure(
            scenario=scenario,
            http_status_code=(
                http_response.status_code
            ),
            request_id=header_request_id,
            request_error=(
                "多模态Agent响应头与正文的"
                "request_id不一致"
            ),
            latency_ms=latency_ms,
        )

    # 当前公开Agent响应没有返回Planner与Vision模型的
    # Token usage。不能根据步骤数猜测成本，因此明确标为
    # unavailable，避免产生看似精确但不可审计的估算值。
    evaluation = evaluate_multimodal_agent_response(
        scenario=scenario,
        response=api_response,
        http_status_code=(
            http_response.status_code
        ),
        latency_ms=latency_ms,
        cost_status="unavailable",
        estimated_cost_usd=None,
        cost_note=(
            "Agent API公开响应未提供模型Token使用量，"
            "无法可靠估算单请求成本"
        ),
        llm_judge_used=False,
        llm_judge_passed=None,
        llm_judge_note=None,
    )

    # 正式确定性评分完成后，只把仍未匹配的Gold视觉字段交给
    # 可选Judge。已经由精确、包含或事实锚点匹配成功的字段
    # 不再产生模型成本；没有任何实际观察时也不存在“等价表达”
    # 可以判断，因此不会调用Judge。
    unmatched_expected_observations = (
        tuple(
            item.expected_observation
            for item in (
                evaluation
                .visual_observation_evaluations
            )
            if not item.matched
        )
        if visual_equivalence_judge is not None
        else ()
    )

    if (
        visual_equivalence_judge is not None
        and unmatched_expected_observations
        and evaluation.actual_visual_observations
    ):
        judge_passed: bool
        judge_note: str

        try:
            judge_result = await (
                visual_equivalence_judge
                .judge_equivalence(
                    expected_observations=(
                        unmatched_expected_observations
                    ),
                    actual_observations=(
                        evaluation
                        .actual_visual_observations
                    ),
                )
            )
        except ApplicationError:
            # Judge是辅助指标。已知上游、超时或模型响应错误
            # 不能覆盖真实Agent结果，也不能中断整批正式评测。
            judge_passed = False
            judge_note = (
                "辅助视觉等价Judge不可用；"
                "正式确定性评分保持不变"
            )
        else:
            if not isinstance(
                judge_result,
                VisualEquivalenceJudgeResult,
            ):
                raise TypeError(
                    "judge_equivalence()必须返回"
                    "VisualEquivalenceJudgeResult"
                )

            judge_passed = (
                judge_result.all_equivalent
            )
            judge_note = (
                build_visual_equivalence_judge_note(
                    judge_result
                )
            )

        # 重新调用纯评分函数只用于把辅助Judge结果写入经过
        # Pydantic交叉校验的正式结果。evaluate_multimodal_agent_response()
        # 的passed仍只由确定性失败原因计算，Judge不能删掉失败原因。
        evaluation = evaluate_multimodal_agent_response(
            scenario=scenario,
            response=api_response,
            http_status_code=(
                http_response.status_code
            ),
            latency_ms=latency_ms,
            cost_status="unavailable",
            estimated_cost_usd=None,
            cost_note=(
                "Agent API公开响应未提供模型Token使用量，"
                "无法可靠估算单请求成本"
            ),
            llm_judge_used=True,
            llm_judge_passed=judge_passed,
            llm_judge_note=judge_note,
        )

    # 只为已经通过HTTP、JSON和响应Schema校验的运行
    # 构造完整轨迹。失败结果没有可被信任的公开响应，
    # 因而不会调用trace_consumer。
    if trace_consumer is not None:
        trace_consumer(
            build_multimodal_agent_trace(
                scenario=scenario,
                response=api_response,
                evaluation=evaluation,
            )
        )

    return evaluation


async def evaluate_multimodal_agent_scenarios_via_api(
    *,
    scenarios: Sequence[
        MultimodalAgentEvaluationScenario
    ],
    client: httpx.AsyncClient,
    api_url: str,
    project_root: Path,
    vision_input_adapter: VisionInputAdapter,
    scenario_source: str,
    planner_model: str,
    planner_prompt_version: str,
    embedding_model: str,
    collection_name: str,
    vision_model: str,
    vision_prompt_version: str,
    trace_targets: Mapping[str, str],
    fix_history: Sequence[
        MultimodalEvaluationFixRecord
    ],
    visual_equivalence_judge: (
        VisualEquivalenceJudge | None
    ) = None,
) -> MultimodalAgentBatchExecution:
    """顺序执行完整多模态场景集并构造待写盘产物。

    本函数不写JSON或Markdown文件。它先完成数据集预检，
    再逐条调用单场景执行器，最后构造经过交叉校验的正式
    报告、至少三份代表轨迹，以及所有能够安全构造的
    失败场景脱敏轨迹。

    trace_targets表示优先保存的代表场景。若某个优先场景的
    HTTP或响应契约失败，本函数会从同一批次其他合法响应中
    优先选择不同类别的完整轨迹补足数量。失败场景本身仍保留
    在正式results中，后备轨迹只保证审计产物不会因一次模型
    波动而让整批报告丢失。除此之外，passed=False且响应
    契约有效的场景都会追加到失败轨迹目录中。
    """

    if (
        isinstance(
            scenarios,
            (str, bytes, bytearray),
        )
        or not isinstance(scenarios, Sequence)
    ):
        raise TypeError(
            "scenarios必须是"
            "MultimodalAgentEvaluationScenario序列"
        )

    # tuple建立本次批量运行的稳定只读快照，避免调用方
    # 在异步执行过程中修改原始list的内容或顺序。
    scenario_items = tuple(scenarios)

    if (
        len(scenario_items)
        < MIN_MULTIMODAL_AGENT_SCENARIO_COUNT
    ):
        raise ValueError(
            "多模态Agent批量评测至少需要"
            f"{MIN_MULTIMODAL_AGENT_SCENARIO_COUNT}"
            "条场景"
        )

    for index, scenario in enumerate(
        scenario_items
    ):
        if not isinstance(
            scenario,
            MultimodalAgentEvaluationScenario,
        ):
            raise TypeError(
                f"scenarios[{index}]必须是"
                "MultimodalAgentEvaluationScenario"
            )

    scenario_ids = tuple(
        scenario.scenario_id
        for scenario in scenario_items
    )

    if len(scenario_ids) != len(
        set(scenario_ids)
    ):
        raise ValueError(
            "多模态批量场景包含重复scenario_id"
        )

    if not isinstance(client, httpx.AsyncClient):
        raise TypeError(
            "client必须是httpx.AsyncClient"
        )

    # 报告元数据必须在第一条网络请求前完成类型、空白和
    # 两端空格清理，避免跑完30条后才发现报告无法构造。
    metadata_values: dict[str, object] = {
        "api_url": api_url,
        "scenario_source": scenario_source,
        "planner_model": planner_model,
        "planner_prompt_version": (
            planner_prompt_version
        ),
        "embedding_model": embedding_model,
        "collection_name": collection_name,
        "vision_model": vision_model,
        "vision_prompt_version": (
            vision_prompt_version
        ),
    }
    cleaned_metadata: dict[str, str] = {}

    for field_name, value in (
        metadata_values.items()
    ):
        if not isinstance(value, str):
            raise TypeError(
                f"{field_name}必须是字符串"
            )

        cleaned_value = value.strip()

        if not cleaned_value:
            raise ValueError(
                f"{field_name}不能为空"
            )

        cleaned_metadata[field_name] = (
            cleaned_value
        )

    if not isinstance(trace_targets, Mapping):
        raise TypeError(
            "trace_targets必须是场景编号到路径的映射"
        )

    # dict保留插入顺序。这个顺序会同时用于轨迹产物和
    # report.trace_paths，保证最终报告内容可重复。
    trace_target_items = tuple(
        trace_targets.items()
    )

    if not (
        3 <= len(trace_target_items) <= 30
    ):
        raise ValueError(
            "trace_targets必须包含3到30条轨迹"
        )

    preferred_scenario_ids: list[str] = []
    preferred_trace_paths: list[str] = []

    for index, item in enumerate(
        trace_target_items
    ):
        scenario_id, trace_path = item

        if not isinstance(scenario_id, str):
            raise TypeError(
                f"trace_targets第{index}个"
                "场景编号必须是字符串"
            )

        if not isinstance(trace_path, str):
            raise TypeError(
                f"trace_targets第{index}个"
                "轨迹路径必须是字符串"
            )

        cleaned_scenario_id = scenario_id.strip()
        cleaned_trace_path = trace_path.strip()

        if cleaned_scenario_id not in scenario_ids:
            raise ValueError(
                "trace_targets引用了不在本批次中的场景："
                f"{cleaned_scenario_id}"
            )

        # 与正式报告和轨迹Artifact使用相同的路径边界。
        parsed_path = PurePosixPath(
            cleaned_trace_path
        )

        if (
            parsed_path.is_absolute()
            or ".." in parsed_path.parts
            or not parsed_path.parts
            or parsed_path.parts[0] != "docs"
            or parsed_path.suffix.lower() != ".json"
        ):
            raise ValueError(
                "trace_targets路径必须是docs目录下的"
                "安全相对JSON路径"
            )

        preferred_scenario_ids.append(
            cleaned_scenario_id
        )
        preferred_trace_paths.append(
            cleaned_trace_path
        )

    if len(preferred_trace_paths) != len(
        set(preferred_trace_paths)
    ):
        raise ValueError(
            "trace_targets不能重复使用同一轨迹路径"
        )

    if (
        isinstance(
            fix_history,
            (str, bytes, bytearray),
        )
        or not isinstance(fix_history, Sequence)
    ):
        raise TypeError(
            "fix_history必须是修正记录序列"
        )

    fix_items = tuple(fix_history)

    if not fix_items:
        raise ValueError(
            "fix_history至少需要一条修正记录"
        )

    for index, item in enumerate(fix_items):
        if not isinstance(
            item,
            MultimodalEvaluationFixRecord,
        ):
            raise TypeError(
                f"fix_history[{index}]必须是"
                "MultimodalEvaluationFixRecord"
            )

    if (
        visual_equivalence_judge is not None
        and not callable(
            getattr(
                visual_equivalence_judge,
                "judge_equivalence",
                None,
            )
        )
    ):
        raise TypeError(
            "visual_equivalence_judge必须实现"
            "judge_equivalence()或为None"
        )

    # 在发出第一条HTTP请求之前验证全部本地图片。
    # 这里会在单场景执行时再次准备请求，但能够保证坏掉的
    # 第20条图片不会在已经消耗19次真实API调用后才暴露。
    for scenario in scenario_items:
        prepare_multimodal_agent_request(
            scenario=scenario,
            project_root=project_root,
            vision_input_adapter=(
                vision_input_adapter
            ),
        )

    results: list[
        MultimodalAgentScenarioEvaluation
    ] = []
    collected_traces: dict[
        str,
        MultimodalAgentTrace,
    ] = {}

    # 有意顺序执行，不使用asyncio.gather()。顺序运行可以
    # 降低Planner和Vision服务限流风险，并让报告顺序、
    # 服务端日志和JSONL行顺序稳定对应。
    for scenario in scenario_items:
        expected_scenario_id = scenario.scenario_id

        def collect_trace(
            trace: MultimodalAgentTrace,
            expected_id: str = expected_scenario_id,
        ) -> None:
            """收集当前场景通过全部响应契约校验的轨迹。"""

            if trace.scenario_id != expected_id:
                raise ValueError(
                    "单场景执行器返回了错误场景的轨迹"
                )

            collected_traces[expected_id] = trace

        result = await (
            evaluate_multimodal_agent_scenario_via_api(
                scenario=scenario,
                client=client,
                api_url=cleaned_metadata[
                    "api_url"
                ],
                project_root=project_root,
                vision_input_adapter=(
                    vision_input_adapter
                ),
                trace_consumer=collect_trace,
                visual_equivalence_judge=(
                    visual_equivalence_judge
                ),
            )
        )
        results.append(result)

    missing_preferred_ids = tuple(
        scenario_id
        for scenario_id in preferred_scenario_ids
        if scenario_id not in collected_traces
    )

    required_trace_count = len(
        preferred_scenario_ids
    )

    if len(collected_traces) < required_trace_count:
        raise EvaluationDataError(
            "本批次通过响应契约校验的完整轨迹不足："
            f"需要{required_trace_count}份，"
            f"实际{len(collected_traces)}份；"
            "缺失的优先场景："
            + (
                "、".join(missing_preferred_ids)
                if missing_preferred_ids
                else "无"
            )
        )

    # 优先保留调用方指定的代表场景。只有指定场景没有合法
    # 响应时才选择后备，因此正常运行的轨迹文件名保持不变。
    selected_scenario_ids = [
        scenario_id
        for scenario_id in preferred_scenario_ids
        if scenario_id in collected_traces
    ]
    selected_trace_paths = [
        trace_path
        for scenario_id, trace_path in zip(
            preferred_scenario_ids,
            preferred_trace_paths,
            strict=True,
        )
        if scenario_id in collected_traces
    ]
    selected_id_set = set(
        selected_scenario_ids
    )
    selected_path_set = set(
        selected_trace_paths
    )

    if missing_preferred_ids:
        used_categories = {
            scenario.category
            for scenario in scenario_items
            if scenario.scenario_id
            in selected_id_set
        }

        # 后备文件继续放在调用方指定的轨迹目录中，但文件名
        # 使用真实场景编号，避免路径名称与轨迹内容不一致。
        fallback_directory = PurePosixPath(
            preferred_trace_paths[0]
        ).parent

        # 第一轮优先补充尚未覆盖的场景类别；如果类别数量不足，
        # 第二轮再按Gold场景原始顺序补齐。这样既稳定又尽量保留
        # 正常、冲突、缺失和对抗场景的审计多样性。
        for require_new_category in (
            True,
            False,
        ):
            for scenario in scenario_items:
                if (
                    len(selected_scenario_ids)
                    >= required_trace_count
                ):
                    break

                scenario_id = scenario.scenario_id

                if (
                    scenario_id not in collected_traces
                    or scenario_id in selected_id_set
                    or (
                        require_new_category
                        and scenario.category
                        in used_categories
                    )
                ):
                    continue

                fallback_path = (
                    fallback_directory
                    / f"{scenario_id}.json"
                ).as_posix()

                # 正常配置下场景编号能够保证唯一。该循环额外
                # 防御调用方把优先路径命名成其他场景编号的情况。
                suffix = 1
                while fallback_path in selected_path_set:
                    fallback_path = (
                        fallback_directory
                        / (
                            f"{scenario_id}-"
                            f"fallback-{suffix}.json"
                        )
                    ).as_posix()
                    suffix += 1

                selected_scenario_ids.append(
                    scenario_id
                )
                selected_trace_paths.append(
                    fallback_path
                )
                selected_id_set.add(scenario_id)
                selected_path_set.add(fallback_path)
                used_categories.add(
                    scenario.category
                )

        logger.warning(
            "multimodal_trace_targets_fallback "
            "missing_preferred=%s selected=%s",
            ",".join(missing_preferred_ids),
            ",".join(selected_scenario_ids),
        )

    # 三条代表轨迹满足Week 5最小审计要求，但不足以定位
    # 批量评测中的每个失败。这里继续追加所有passed=False、
    # 且API响应已经通过公开契约校验的场景轨迹。
    #
    # 失败轨迹统一写入代表轨迹目录下的failures子目录，
    # 文件名使用真实scenario_id，便于从报告失败项直接定位。
    failure_trace_directory = (
        PurePosixPath(
            preferred_trace_paths[0]
        ).parent
        / "failures"
    )
    unavailable_failure_trace_ids: list[
        str
    ] = []

    for result in results:
        # passed=True的场景只在它属于三条代表轨迹时保存，
        # 避免把30条成功响应全部重复写入docs目录。
        if result.passed:
            continue

        scenario_id = result.scenario_id

        # 同一个失败场景可能已经是指定代表或后备代表。
        # 每个场景只保存一份轨迹，防止报告出现重复证据。
        if scenario_id in selected_id_set:
            continue

        trace = collected_traces.get(
            scenario_id
        )

        if trace is None:
            # HTTP失败或响应没有通过AgentDiagnosisResponse
            # 契约时，不能伪造“完整响应轨迹”。这类失败仍会
            # 保留在report.results的request_error字段中。
            unavailable_failure_trace_ids.append(
                scenario_id
            )
            continue

        failure_trace_path = (
            failure_trace_directory
            / f"{scenario_id}.json"
        ).as_posix()

        # scenario_id本身唯一，正常情况下不会冲突。
        # 保留防御性后缀，避免调用方把代表轨迹路径配置到
        # failures目录并占用同名文件。
        suffix = 1
        while (
            failure_trace_path
            in selected_path_set
        ):
            failure_trace_path = (
                failure_trace_directory
                / (
                    f"{scenario_id}-"
                    f"failure-{suffix}.json"
                )
            ).as_posix()
            suffix += 1

        selected_scenario_ids.append(
            scenario_id
        )
        selected_trace_paths.append(
            failure_trace_path
        )
        selected_id_set.add(
            scenario_id
        )
        selected_path_set.add(
            failure_trace_path
        )

    if unavailable_failure_trace_ids:
        logger.warning(
            "multimodal_failed_trace_unavailable "
            "scenario_ids=%s",
            ",".join(
                unavailable_failure_trace_ids
            ),
        )

    report = build_multimodal_agent_evaluation_report(
        results=results,
        scenario_source=cleaned_metadata[
            "scenario_source"
        ],
        api_url=cleaned_metadata["api_url"],
        planner_model=cleaned_metadata[
            "planner_model"
        ],
        planner_prompt_version=(
            cleaned_metadata[
                "planner_prompt_version"
            ]
        ),
        embedding_model=cleaned_metadata[
            "embedding_model"
        ],
        collection_name=cleaned_metadata[
            "collection_name"
        ],
        vision_model=cleaned_metadata[
            "vision_model"
        ],
        vision_prompt_version=(
            cleaned_metadata[
                "vision_prompt_version"
            ]
        ),
        trace_paths=selected_trace_paths,
        fix_history=fix_items,
    )

    artifacts = tuple(
        MultimodalAgentTraceArtifact(
            path=trace_path,
            trace=collected_traces[
                scenario_id
            ],
        )
        for scenario_id, trace_path in zip(
            selected_scenario_ids,
            selected_trace_paths,
            strict=True,
        )
    )

    # 最终Bundle会再次确认报告路径、轨迹场景和报告中的
    # 同场景评分逐一对应，然后才允许命令行脚本写文件。
    return MultimodalAgentBatchExecution(
        report=report,
        trace_artifacts=artifacts,
    )
