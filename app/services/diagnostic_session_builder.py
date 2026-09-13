"""由Agent请求和公开响应构造脱敏诊断会话。

本模块位于AgentDiagnosisService与会话存储层之间。

它只执行确定性数据转换：

1. 生成服务端session_id；
2. 创建带时区的会话时间；
3. 生成不包含完整日志和图片的请求摘要；
4. 复制已经通过公开响应契约校验的轨迹和观察；
5. 汇总失败工具、证据覆盖和运行耗时；
6. 返回经过DiagnosticSessionRecord校验的记录。

本模块不执行Agent、不调用模型、不执行工具，
也不负责把记录写入文件。
"""

from collections.abc import (
    Callable,
)
from datetime import (
    datetime,
    timezone,
)
from hashlib import (
    sha256,
)
import re
from uuid import (
    UUID,
    uuid4,
)

from app.schemas.agent_api import (
    AgentDiagnosisRequest,
    AgentDiagnosisResponse,
)
from app.schemas.diagnostic_session import (
    DiagnosticSessionDiagnosisSnapshot,
    DiagnosticSessionEvidenceCoverage,
    DiagnosticSessionMetrics,
    DiagnosticSessionRecord,
    DiagnosticSessionRequestSummary,
    DiagnosticSessionStatus,
    DiagnosticSessionSummary,
)


# symptom和task_goal写入会话前最多保留500个字符。
#
# 这与DiagnosticSessionRequestSummary中的字段上限一致。
_SESSION_SUMMARY_MAX_LENGTH = 500


# 匹配常见的“字段名=密钥值”形式。
#
# 这里只做会话摘要的纵深防御，
# 不能代替API入口对日志进行正式脱敏。
_SECRET_ASSIGNMENT_PATTERN = re.compile(
    (
        r"\b("
        r"api[_-]?key"
        r"|access[_-]?token"
        r"|token"
        r"|password"
        r"|secret"
        r")\b"
        r"\s*[:=]\s*"
        r"[^\s,;]+"
    ),
    re.IGNORECASE,
)


# 匹配HTTP Authorization常见的Bearer令牌形式。
_BEARER_TOKEN_PATTERN = re.compile(
    r"\bbearer\s+[A-Za-z0-9._~+/=-]+",
    re.IGNORECASE,
)


# 下面两个类型别名描述构造器的可注入依赖。
#
# 生产环境分别使用uuid4和UTC系统时间；
# 测试可以传入固定函数，从而得到可重复结果。
SessionIdFactory = Callable[[], UUID]
SessionClock = Callable[[], datetime]


def _default_utc_clock() -> datetime:
    """返回包含UTC时区的当前时间。"""

    return datetime.now(timezone.utc)


def _compact_public_summary(
    text: str,
    *,
    max_length: int,
) -> str:
    """压缩空白、遮盖常见密钥并限制摘要长度。

    处理顺序是：

    1. 将换行、Tab和连续空格压缩成单个空格；
    2. 遮盖常见“key=value”形式；
    3. 遮盖Bearer令牌；
    4. 超过限制时截断并添加省略号。

    函数不会把处理后的文本写日志。
    """

    if not isinstance(text, str):
        raise TypeError("text必须是str")

    if (
        not isinstance(max_length, int)
        or isinstance(max_length, bool)
    ):
        raise TypeError(
            "max_length必须是int"
        )

    if max_length < 2:
        raise ValueError(
            "max_length必须至少为2"
        )

    # split()不传参数时会把任意连续空白视为分隔符。
    # 再使用单个空格连接，得到紧凑的单行摘要。
    compacted = " ".join(text.split())

    compacted = (
        _SECRET_ASSIGNMENT_PATTERN.sub(
            lambda match: (
                f"{match.group(1)}=[REDACTED]"
            ),
            compacted,
        )
    )

    compacted = _BEARER_TOKEN_PATTERN.sub(
        "Bearer [REDACTED]",
        compacted,
    )

    if not compacted:
        raise ValueError(
            "摘要文本清理后不能为空"
        )

    if len(compacted) <= max_length:
        return compacted

    # 保留max_length - 1个字符，
    # 最后一个位置用于添加省略号。
    return (
        compacted[: max_length - 1]
        + "…"
    )


def _derive_session_status(
    response: AgentDiagnosisResponse,
    /,
) -> DiagnosticSessionStatus:
    """从执行结果推导统一会话状态。

    判断优先级不能颠倒：

    1. Runner异常中止统一记为failed；
    2. 人工审核需要独立于普通abstained；
    3. 其余正常结束沿用DiagnosisReport状态。
    """

    execution = response.execution

    if execution.state == "aborted":
        return "failed"

    # 使用进度状态机的终态优先识别人工审核。
    #
    # 该值由Python控制逻辑产生，
    # 不是Planner可以任意填写的自由文本。
    if (
        execution.progress is not None
        and execution.progress.state
        == "human_review_required"
    ):
        return "human_review_required"

    if (
        execution.finish_reason
        == "human_review_required"
    ):
        return "human_review_required"

    return response.diagnosis.status


def _build_request_summary(
    request: AgentDiagnosisRequest,
    /,
) -> DiagnosticSessionRequestSummary:
    """构造不保存完整日志和图片的请求摘要。"""

    task_goal_summary = (
        _compact_public_summary(
            request.task_goal,
            max_length=(
                _SESSION_SUMMARY_MAX_LENGTH
            ),
        )
        if request.task_goal is not None
        else None
    )

    # encode("utf-8")将Python字符串转换成确定字节。
    #
    # sha256(...).hexdigest()返回64位小写十六进制摘要。
    # 会话记录只保存摘要，不保存log_excerpt正文。
    log_excerpt_sha256 = sha256(
        request.log_excerpt.encode("utf-8")
    ).hexdigest()

    return DiagnosticSessionRequestSummary(
        robot_id=request.robot_id,
        symptom_summary=(
            _compact_public_summary(
                request.symptom,
                max_length=(
                    _SESSION_SUMMARY_MAX_LENGTH
                ),
            )
        ),
        task_goal_summary=task_goal_summary,
        log_excerpt_sha256=(
            log_excerpt_sha256
        ),
        log_excerpt_char_count=len(
            request.log_excerpt
        ),
        image_count=len(request.images),
    )


def _build_diagnosis_snapshot(
    response: AgentDiagnosisResponse,
    /,
) -> DiagnosticSessionDiagnosisSnapshot:
    """复制诊断结论，但排除用户症状和日志正文。"""

    diagnosis = response.diagnosis

    return DiagnosticSessionDiagnosisSnapshot(
        prompt_version=(
            diagnosis.prompt_version
        ),
        gate_version=(
            diagnosis.gate_version
        ),
        status=diagnosis.status,
        evidence=tuple(diagnosis.evidence),
        possible_causes=tuple(
            diagnosis.possible_causes
        ),
        next_checks=tuple(
            diagnosis.next_checks
        ),
        risk_level=diagnosis.risk_level,
        missing_information=tuple(
            diagnosis.missing_information
        ),
        abstained=diagnosis.abstained,
    )


def _build_evidence_coverage(
    response: AgentDiagnosisResponse,
    /,
) -> DiagnosticSessionEvidenceCoverage:
    """从最终Progress和诊断引用提取证据覆盖情况。"""

    progress = response.execution.progress

    if progress is None:
        confirmed_source_ids: tuple[
            str,
            ...,
        ] = ()
        covered_evidence_ids: tuple[
            str,
            ...,
        ] = ()
    else:
        confirmed_source_ids = tuple(
            progress.confirmed_source_ids
        )
        covered_evidence_ids = tuple(
            progress.covered_evidence_ids
        )

    cited_chunk_ids = tuple(
        evidence.chunk_id
        for evidence
        in response.diagnosis.evidence
    )

    return DiagnosticSessionEvidenceCoverage(
        confirmed_source_ids=(
            confirmed_source_ids
        ),
        covered_evidence_ids=(
            covered_evidence_ids
        ),
        cited_chunk_ids=cited_chunk_ids,
    )


def _build_metrics(
    *,
    response: AgentDiagnosisResponse,
    total_duration_ms: float,
    evidence_coverage: (
        DiagnosticSessionEvidenceCoverage
    ),
) -> DiagnosticSessionMetrics:
    """根据真实公开轨迹生成会话统计。"""

    tool_events = tuple(
        response.execution.steps
    )

    # empty表示工具正常执行但没有得到数据，
    # 不等同于Executor或上游服务执行失败。
    failed_events = tuple(
        event
        for event in tool_events
        if event.status in {
            "rejected",
            "timeout",
            "error",
        }
    )

    # dict.fromkeys()在Python 3.7及以后保持插入顺序。
    #
    # 这样同一工具失败多次时：
    #
    # failed_tool_count记录失败事件数；
    # failed_tool_names只保留一次工具名。
    failed_tool_names = tuple(
        dict.fromkeys(
            event.tool_name
            for event in failed_events
        )
    )

    return DiagnosticSessionMetrics(
        total_duration_ms=total_duration_ms,
        tool_step_count=len(tool_events),
        failed_tool_count=len(
            failed_events
        ),
        failed_tool_names=(
            failed_tool_names
        ),
        confirmed_source_count=len(
            evidence_coverage
            .confirmed_source_ids
        ),
        covered_evidence_count=len(
            evidence_coverage
            .covered_evidence_ids
        ),
        citation_count=len(
            evidence_coverage.cited_chunk_ids
        ),
    )


class DiagnosticSessionBuilder:
    """把一次Agent诊断转换成持久化会话记录。

    session_id_factory和clock使用依赖注入，
    使生产环境可以生成真实ID和时间，
    测试环境可以注入固定值。
    """

    def __init__(
        self,
        *,
        session_id_factory: (
            SessionIdFactory
        ) = uuid4,
        clock: SessionClock = (
            _default_utc_clock
        ),
    ) -> None:
        """保存会话ID生成器和业务时钟。"""

        if not callable(
            session_id_factory
        ):
            raise TypeError(
                "session_id_factory必须可调用"
            )

        if not callable(clock):
            raise TypeError(
                "clock必须可调用"
            )

        self._session_id_factory = (
            session_id_factory
        )
        self._clock = clock

    def build(
        self,
        *,
        request: AgentDiagnosisRequest,
        response: AgentDiagnosisResponse,
        total_duration_ms: float,
    ) -> DiagnosticSessionRecord:
        """构造一条经过完整Schema校验的会话记录。

        total_duration_ms由调用方在Agent执行前后使用
        time.perf_counter()测量后传入。

        Builder不自行启动计时，否则无法覆盖发生在
        AgentDiagnosisService内部的全部执行时间。
        """

        if not isinstance(
            request,
            AgentDiagnosisRequest,
        ):
            raise TypeError(
                "request必须是"
                "AgentDiagnosisRequest"
            )

        if not isinstance(
            response,
            AgentDiagnosisResponse,
        ):
            raise TypeError(
                "response必须是"
                "AgentDiagnosisResponse"
            )

        if (
            isinstance(total_duration_ms, bool)
            or not isinstance(
                total_duration_ms,
                (int, float),
            )
        ):
            raise TypeError(
                "total_duration_ms必须是数值"
            )

        request_summary = (
            _build_request_summary(request)
        )

        diagnosis_snapshot = (
            _build_diagnosis_snapshot(
                response
            )
        )

        evidence_coverage = (
            _build_evidence_coverage(
                response
            )
        )

        metrics = _build_metrics(
            response=response,
            total_duration_ms=float(
                total_duration_ms
            ),
            evidence_coverage=(
                evidence_coverage
            ),
        )

        # UUID.__str__()返回标准带连字符文本。
        #
        # lower()保证满足DiagnosticSessionId的
        # 小写UUID4契约。
        session_id = str(
            self._session_id_factory()
        ).lower()

        created_at = self._clock()

        return DiagnosticSessionRecord(
            session_id=session_id,
            request_id=response.request_id,
            created_at=created_at,
            request_summary=request_summary,
            status=_derive_session_status(
                response
            ),
            execution_state=(
                response.execution.state
            ),
            termination_reason=(
                response
                .execution
                .termination_reason
            ),
            finish_reason=(
                response
                .execution
                .finish_reason
            ),
            diagnosis=diagnosis_snapshot,
            tool_events=tuple(
                response.execution.steps
            ),
            vision_observations=tuple(
                response.vision_observations
            ),
            telemetry_observations=tuple(
                response
                .telemetry_observations
            ),
            test_case_drafts=tuple(
                response.test_case_drafts
            ),
            evidence_coverage=(
                evidence_coverage
            ),
            metrics=metrics,
        )


def build_diagnostic_session_summary(
    record: DiagnosticSessionRecord,
    /,
) -> DiagnosticSessionSummary:
    """从完整会话记录创建轻量列表摘要。

    列表摘要不包含完整工具轨迹、Vision观察、
    证据原文、遥测详情或测试草案。
    """

    if not isinstance(
        record,
        DiagnosticSessionRecord,
    ):
        raise TypeError(
            "record必须是"
            "DiagnosticSessionRecord"
        )

    return DiagnosticSessionSummary(
        session_id=record.session_id,
        request_id=record.request_id,
        created_at=record.created_at,
        robot_id=(
            record.request_summary.robot_id
        ),
        symptom_summary=(
            record
            .request_summary
            .symptom_summary
        ),
        status=record.status,
        termination_reason=(
            record.termination_reason
        ),
        tool_step_count=(
            record.metrics.tool_step_count
        ),
        failed_tool_count=(
            record
            .metrics
            .failed_tool_count
        ),
        citation_count=(
            record.metrics.citation_count
        ),
        total_duration_ms=(
            record.metrics.total_duration_ms
        ),
    )