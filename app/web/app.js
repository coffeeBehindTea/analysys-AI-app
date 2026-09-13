"use strict";

/*
RobotOps Copilot Beta浏览器端控制器。

本文件负责：

1. 检查FastAPI服务是否可用；
2. 校验用户填写的表单；
3. 把浏览器File对象转换成VisionImagePayload；
4. 调用POST /api/v1/agent/diagnose；
5. 展示结构化诊断、引用、视觉观察和工具轨迹；
6. 查询并展示最近保存的诊断会话。

本文件不决定Agent工具权限，不执行安全分类，
也不允许绕过后端Pydantic和Python安全策略。
*/


/* =========================================================
   API地址和前端限制
   ========================================================= */

const HEALTH_API_URL = "/health";

const AGENT_DIAGNOSIS_API_URL =
    "/api/v1/agent/diagnose";

const DIAGNOSTIC_SESSIONS_API_URL =
    "/api/v1/diagnostic-sessions";

/*
这个值必须与后端MAX_AGENT_DIAGNOSIS_IMAGES保持一致。

前端限制用于尽早提示用户；
后端限制仍然是最终可信边界。
*/
const MAX_IMAGE_COUNT = 3;

const SUPPORTED_IMAGE_TYPES = new Set([
    "image/jpeg",
    "image/png",
    "image/webp",
]);


/* =========================================================
   获取页面元素
   ========================================================= */

/*
document.getElementById()根据HTML中的唯一id取得DOM元素。

把所有元素集中保存在elements中，可以避免后续函数
不断重复查询DOM，也便于发现HTML和JavaScript字段不一致。
*/
const elements = {
    appStatus:
        document.getElementById("app-status"),

    form:
        document.getElementById("diagnosis-form"),

    robotId:
        document.getElementById("robot-id"),

    symptom:
        document.getElementById("symptom"),

    logExcerpt:
        document.getElementById("log-excerpt"),

    taskGoal:
        document.getElementById("task-goal"),

    imageFiles:
        document.getElementById("image-files"),

    analysisGoal:
        document.getElementById("analysis-goal"),

    imageDetail:
        document.getElementById("image-detail"),

    imageSummary:
        document.getElementById("image-summary"),

    submitButton:
        document.getElementById("submit-button"),

    resetButton:
        document.getElementById("reset-button"),

    resultPanel:
        document.querySelector(".result-panel"),

    resultStatus:
        document.getElementById("result-status"),

    emptyState:
        document.getElementById("empty-state"),

    errorPanel:
        document.getElementById("error-panel"),

    errorMessage:
        document.getElementById("error-message"),

    errorRequestId:
        document.getElementById("error-request-id"),

    diagnosisResult:
        document.getElementById("diagnosis-result"),

    responseRequestId:
        document.getElementById("response-request-id"),

    responseSessionId:
        document.getElementById("response-session-id"),

    diagnosisStatus:
        document.getElementById("diagnosis-status"),

    terminationReason:
        document.getElementById("termination-reason"),

    finishReason:
        document.getElementById("finish-reason"),

    toolStepCount:
        document.getElementById("tool-step-count"),

    possibleCauses:
        document.getElementById("possible-causes"),

    nextChecks:
        document.getElementById("next-checks"),

    missingInformation:
        document.getElementById("missing-information"),

    citationList:
        document.getElementById("citation-list"),

    visionObservationList:
        document.getElementById(
            "vision-observation-list"
        ),

    telemetryList:
        document.getElementById("telemetry-list"),

    testDraftList:
        document.getElementById("test-draft-list"),

    toolTraceBody:
        document.getElementById("tool-trace-body"),

    refreshSessionsButton:
        document.getElementById(
            "refresh-sessions-button"
        ),

    sessionListStatus:
        document.getElementById(
            "session-list-status"
        ),

    sessionList:
        document.getElementById("session-list"),
};


/* =========================================================
   统一错误类型
   ========================================================= */

/*
继承JavaScript内置Error，保存HTTP状态码、
请求ID和服务器返回的结构化数据。

这样调用层不需要把错误压缩成一段无法分析的字符串。
*/
class ApiRequestError extends Error {
    constructor(
        message,
        {
            statusCode = null,
            requestId = null,
            responseData = null,
        } = {},
    ) {
        super(message);

        this.name = "ApiRequestError";
        this.statusCode = statusCode;
        this.requestId = requestId;
        this.responseData = responseData;
    }
}


/* =========================================================
   通用DOM辅助函数
   ========================================================= */

/*
所有服务端文本都通过textContent写入页面。

不使用innerHTML的原因是：
模型输出、日志摘要和知识库正文都属于不可信数据。
textContent只把内容当作文字，不会执行其中的HTML或脚本。
*/
function createTextElement(
    tagName,
    text,
    className = null,
) {
    const element =
        document.createElement(tagName);

    element.textContent =
        text === null || text === undefined
            ? "—"
            : String(text);

    if (className !== null) {
        element.className = className;
    }

    return element;
}


function clearElement(element) {
    /*
    replaceChildren()删除元素的所有子节点。
    它比反复调用removeChild()更简洁。
    */
    element.replaceChildren();
}


function valueOrDash(value) {
    if (
        value === null
        || value === undefined
        || value === ""
    ) {
        return "—";
    }

    return String(value);
}


function formatNumber(
    value,
    fractionDigits = 3,
) {
    const numericValue = Number(value);

    if (!Number.isFinite(numericValue)) {
        return "—";
    }

    return numericValue.toFixed(
        fractionDigits
    );
}


function formatDuration(durationMs) {
    const numericValue = Number(durationMs);

    if (!Number.isFinite(numericValue)) {
        return "—";
    }

    if (numericValue < 1000) {
        return `${numericValue.toFixed(1)} ms`;
    }

    return `${(
        numericValue / 1000
    ).toFixed(2)} s`;
}


function formatDateTime(value) {
    if (!value) {
        return "—";
    }

    const date = new Date(value);

    if (Number.isNaN(date.getTime())) {
        return String(value);
    }

    return date.toLocaleString(
        "zh-CN",
        {
            hour12: false,
        },
    );
}


/*
只允许有限状态转换成CSS类。

不能直接把服务端返回值拼接成任意类名，
否则异常文本可能破坏页面样式。
*/
function getBadgeClass(status) {
    const classByStatus = {
        completed: "badge-completed",
        success: "badge-success",
        partial: "badge-partial",
        warning: "badge-warning",
        empty: "badge-empty",
        abstained: "badge-abstained",
        failed: "badge-failed",
        aborted: "badge-aborted",
        error: "badge-error",
        timeout: "badge-timeout",
        rejected: "badge-rejected",

        human_review_required:
            "badge-human-review",
    };

    return (
        classByStatus[status]
        ?? "badge-neutral"
    );
}


function createStatusBadge(status) {
    return createTextElement(
        "span",
        valueOrDash(status),
        `result-badge ${getBadgeClass(status)}`,
    );
}


function replaceWithStatusBadge(
    target,
    status,
) {
    clearElement(target);

    target.append(
        createStatusBadge(status)
    );
}


function appendEmptyMessage(
    container,
    message,
) {
    container.append(
        createTextElement(
            container.matches("ul, ol")
                ? "li"
                : "p",
            message,
            "empty-list-message",
        ),
    );
}


function appendDefinitionList(
    card,
    entries,
) {
    const definitionList =
        document.createElement("dl");

    for (const [label, value] of entries) {
        if (
            value === null
            || value === undefined
            || value === ""
        ) {
            continue;
        }

        definitionList.append(
            createTextElement("dt", label),
            createTextElement(
                "dd",
                value,
            ),
        );
    }

    card.append(definitionList);
}


function appendStringList(
    container,
    values,
    emptyMessage,
) {
    clearElement(container);

    if (
        !Array.isArray(values)
        || values.length === 0
    ) {
        appendEmptyMessage(
            container,
            emptyMessage,
        );
        return;
    }

    for (const value of values) {
        container.append(
            createTextElement(
                "li",
                value,
            ),
        );
    }
}


/* =========================================================
   服务状态
   ========================================================= */

function setServiceStatus(
    statusClass,
    message,
) {
    elements.appStatus.className =
        `status-pill ${statusClass}`;

    elements.appStatus.textContent =
        message;
}


async function checkServiceHealth() {
    setServiceStatus(
        "status-checking",
        "正在检查服务状态",
    );

    try {
        const response = await fetch(
            HEALTH_API_URL,
            {
                method: "GET",
                headers: {
                    Accept: "application/json",
                },
            },
        );

        if (!response.ok) {
            throw new Error(
                `健康检查返回HTTP ${response.status}`
            );
        }

        setServiceStatus(
            "status-ready",
            "服务可用",
        );
    } catch (error) {
        console.error(
            "Health check failed:",
            error,
        );

        setServiceStatus(
            "status-unavailable",
            "服务不可用",
        );
    }
}


/* =========================================================
   图片读取和请求构造
   ========================================================= */

function updateImageSummary() {
    clearElement(elements.imageSummary);

    const files = Array.from(
        elements.imageFiles.files ?? []
    );

    if (files.length === 0) {
        return;
    }

    for (const file of files) {
        const sizeKilobytes =
            file.size / 1024;

        elements.imageSummary.append(
            createTextElement(
                "li",
                `${file.name} · `
                + `${file.type || "未知格式"} · `
                + `${sizeKilobytes.toFixed(1)} KB`,
            ),
        );
    }
}


/*
FileReader是浏览器提供的异步文件读取API。

readAsDataURL()返回类似：

data:image/png;base64,iVBORw0KGgo...

后端只接受逗号后面的纯Base64主体，
所以后续需要移除Data URL前缀。
*/
function readFileAsDataUrl(file) {
    return new Promise(
        (resolve, reject) => {
            const reader = new FileReader();

            reader.addEventListener(
                "load",
                () => {
                    if (
                        typeof reader.result
                        !== "string"
                    ) {
                        reject(
                            new Error(
                                `无法读取图片：${file.name}`
                            ),
                        );
                        return;
                    }

                    resolve(reader.result);
                },
            );

            reader.addEventListener(
                "error",
                () => {
                    reject(
                        new Error(
                            `读取图片失败：${file.name}`
                        ),
                    );
                },
            );

            reader.readAsDataURL(file);
        },
    );
}


async function convertFileToVisionPayload(
    file,
    analysisGoal,
    detail,
) {
    if (
        !SUPPORTED_IMAGE_TYPES.has(
            file.type
        )
    ) {
        throw new Error(
            `不支持图片格式：${file.name}`
        );
    }

    const dataUrl =
        await readFileAsDataUrl(file);

    const separatorIndex =
        dataUrl.indexOf(",");

    if (separatorIndex < 0) {
        throw new Error(
            `图片没有合法Data URL：${file.name}`
        );
    }

    const imageBase64 =
        dataUrl.slice(separatorIndex + 1);

    if (!imageBase64) {
        throw new Error(
            `图片内容为空：${file.name}`
        );
    }

    /*
    返回值与后端VisionImagePayload完全对应。

    encoding固定为base64，
    用户不能在页面中选择其他传输协议。
    */
    return {
        mime_type: file.type,
        encoding: "base64",
        image_base64: imageBase64,
        analysis_goal: analysisGoal,
        detail,
    };
}


async function buildDiagnosisRequest() {
    const robotId =
        elements.robotId.value.trim();

    const symptom =
        elements.symptom.value.trim();

    const logExcerpt =
        elements.logExcerpt.value.trim();

    const taskGoal =
        elements.taskGoal.value.trim();

    const analysisGoal =
        elements.analysisGoal.value.trim();

    const imageDetail =
        elements.imageDetail.value;

    const files = Array.from(
        elements.imageFiles.files ?? []
    );

    if (!robotId) {
        throw new Error(
            "请填写机器人编号"
        );
    }

    if (!symptom) {
        throw new Error(
            "请填写故障现象"
        );
    }

    if (!logExcerpt) {
        throw new Error(
            "请填写脱敏日志摘要"
        );
    }

    if (
        files.length
        > MAX_IMAGE_COUNT
    ) {
        throw new Error(
            `一次最多选择${MAX_IMAGE_COUNT}张图片`
        );
    }

    if (
        files.length > 0
        && !analysisGoal
    ) {
        throw new Error(
            "选择图片后必须填写图片分析目标"
        );
    }

    /*
    Promise.all()并发读取本次请求中的图片。

    返回数组的顺序与files顺序一致，
    因而后端生成的image_001、image_002
    仍然对应用户的选择顺序。
    */
    const images = await Promise.all(
        files.map(
            (file) =>
                convertFileToVisionPayload(
                    file,
                    analysisGoal,
                    imageDetail,
                ),
        ),
    );

    const payload = {
        robot_id: robotId,
        symptom,
        log_excerpt: logExcerpt,
        images,
    };

    /*
    task_goal是可选字段。

    空字符串不发送，避免把空字符串交给后端后
    触发min_length校验错误。
    */
    if (taskGoal) {
        payload.task_goal = taskGoal;
    }

    return payload;
}


/* =========================================================
   HTTP响应解析
   ========================================================= */

async function parseJsonResponse(response) {
    const responseText =
        await response.text();

    if (!responseText) {
        return {};
    }

    try {
        return JSON.parse(responseText);
    } catch {
        throw new ApiRequestError(
            "服务返回了无法解析的JSON",
            {
                statusCode: response.status,
                requestId:
                    response.headers.get(
                        "x-request-id"
                    ),
            },
        );
    }
}


function extractPublicErrorMessage(
    responseData,
    statusCode,
) {
    /*
    应用错误通常使用：

    {
      "error": {
        "code": "...",
        "message": "..."
      }
    }
    */
    if (
        responseData
        && responseData.error
        && responseData.error.message
    ) {
        return responseData.error.message;
    }

    /*
    FastAPI请求校验错误通常使用detail数组。

    loc说明错误字段位置，
    msg是Pydantic提供的公开错误说明。
    */
    if (
        responseData
        && Array.isArray(responseData.detail)
    ) {
        const messages =
            responseData.detail.map(
                (item) => {
                    const location =
                        Array.isArray(item.loc)
                            ? item.loc.join(".")
                            : "request";

                    return (
                        `${location}: `
                        + `${item.msg ?? "输入无效"}`
                    );
                },
            );

        return messages.join("；");
    }

    return `请求失败，HTTP状态码：${statusCode}`;
}


async function requestJson(
    url,
    options = {},
) {
    const response = await fetch(
        url,
        options,
    );

    const responseData =
        await parseJsonResponse(response);

    const requestId =
        response.headers.get(
            "x-request-id"
        )
        ?? responseData.request_id
        ?? null;

    if (!response.ok) {
        throw new ApiRequestError(
            extractPublicErrorMessage(
                responseData,
                response.status,
            ),
            {
                statusCode: response.status,
                requestId,
                responseData,
            },
        );
    }

    return responseData;
}


/* =========================================================
   结果区域状态切换
   ========================================================= */

function setDiagnosisLoading(isLoading) {
    elements.submitButton.disabled =
        isLoading;

    elements.resetButton.disabled =
        isLoading;

    elements.refreshSessionsButton.disabled =
        isLoading;

    elements.resultPanel.classList.toggle(
        "is-loading",
        isLoading,
    );

    elements.submitButton.textContent =
        isLoading
            ? "诊断运行中…"
            : "开始诊断";
}


function showResultError(error) {
    elements.emptyState.hidden = true;
    elements.diagnosisResult.hidden = true;
    elements.errorPanel.hidden = false;

    elements.resultStatus.textContent =
        "请求失败";

    elements.errorMessage.textContent =
        error instanceof Error
            ? error.message
            : String(error);

    if (
        error instanceof ApiRequestError
        && error.requestId
    ) {
        elements.errorRequestId.textContent =
            `请求 ID：${error.requestId}`;
    } else {
        elements.errorRequestId.textContent =
            "";
    }
}


function showDiagnosisResult() {
    elements.emptyState.hidden = true;
    elements.errorPanel.hidden = true;
    elements.diagnosisResult.hidden = false;
}


/* =========================================================
   诊断原因和检查项
   ========================================================= */

function renderPossibleCauses(causes) {
    clearElement(elements.possibleCauses);

    if (
        !Array.isArray(causes)
        || causes.length === 0
    ) {
        appendEmptyMessage(
            elements.possibleCauses,
            "没有经过证据支持的可能原因",
        );
        return;
    }

    for (const cause of causes) {
        const item =
            document.createElement("li");

        item.append(
            document.createTextNode(
                valueOrDash(
                    cause.description
                )
            ),
        );

        if (
            Array.isArray(
                cause.evidence_chunk_ids
            )
            && cause.evidence_chunk_ids.length > 0
        ) {
            item.append(
                createTextElement(
                    "small",
                    `证据：${
                        cause.evidence_chunk_ids.join(
                            "、"
                        )
                    }`,
                    "field-help",
                ),
            );
        }

        elements.possibleCauses.append(item);
    }
}


function renderNextChecks(checks) {
    clearElement(elements.nextChecks);

    if (
        !Array.isArray(checks)
        || checks.length === 0
    ) {
        appendEmptyMessage(
            elements.nextChecks,
            "没有经过证据支持的后续检查",
        );
        return;
    }

    for (const check of checks) {
        const item =
            document.createElement("li");

        item.append(
            document.createTextNode(
                valueOrDash(
                    check.description
                )
            ),
        );

        const metadata = [];

        if (check.risk_level) {
            metadata.push(
                `风险：${check.risk_level}`
            );
        }

        if (
            check.requires_qualified_person
        ) {
            metadata.push(
                "需要有资质人员确认"
            );
        }

        if (
            Array.isArray(
                check.evidence_chunk_ids
            )
            && check.evidence_chunk_ids.length > 0
        ) {
            metadata.push(
                `证据：${
                    check.evidence_chunk_ids.join(
                        "、"
                    )
                }`
            );
        }

        if (metadata.length > 0) {
            item.append(
                createTextElement(
                    "small",
                    metadata.join("；"),
                    "field-help",
                ),
            );
        }

        elements.nextChecks.append(item);
    }
}


/* =========================================================
   知识库引用
   ========================================================= */

function renderCitations(citations) {
    clearElement(elements.citationList);

    if (
        !Array.isArray(citations)
        || citations.length === 0
    ) {
        appendEmptyMessage(
            elements.citationList,
            "本次结果没有知识库引用",
        );
        return;
    }

    for (const citation of citations) {
        const card =
            document.createElement("article");

        card.className = "data-card";

        card.append(
            createTextElement(
                "h4",
                `引用 ${valueOrDash(
                    citation.rank
                )} · ${valueOrDash(
                    citation.source_file
                )}`,
            ),
        );

        appendDefinitionList(
            card,
            [
                [
                    "位置",
                    citation.page_or_section,
                ],
                [
                    "Chunk ID",
                    citation.chunk_id,
                ],
                [
                    "RRF分数",
                    formatNumber(
                        citation.rrf_score,
                        6,
                    ),
                ],
                [
                    "向量相似度",
                    citation.vector_similarity
                        === null
                        || citation.vector_similarity
                        === undefined
                            ? "未进入向量候选"
                            : formatNumber(
                                citation.vector_similarity,
                                6,
                            ),
                ],
            ],
        );

        if (citation.excerpt) {
            card.append(
                createTextElement(
                    "pre",
                    citation.excerpt,
                    "card-excerpt",
                ),
            );
        }

        elements.citationList.append(card);
    }
}


/* =========================================================
   视觉观察
   ========================================================= */

function renderVisionObservations(
    visionObservations,
) {
    clearElement(
        elements.visionObservationList
    );

    if (
        !Array.isArray(visionObservations)
        || visionObservations.length === 0
    ) {
        appendEmptyMessage(
            elements.visionObservationList,
            "本次运行没有视觉观察",
        );
        return;
    }

    for (
        const agentObservation
        of visionObservations
    ) {
        const observation =
            agentObservation.observation ?? {};

        const card =
            document.createElement("article");

        card.className = "data-card";

        card.append(
            createTextElement(
                "h4",
                `图片 ${
                    valueOrDash(
                        agentObservation.image_ref
                    )
                }`,
            ),
        );

        appendDefinitionList(
            card,
            [
                [
                    "工具步骤",
                    agentObservation.step_id,
                ],
                [
                    "分析状态",
                    observation.status,
                ],
                [
                    "图片质量",
                    observation.image_quality,
                ],
                [
                    "来源",
                    observation.source,
                ],
                [
                    "模型",
                    observation.model_name,
                ],
                [
                    "Prompt版本",
                    observation.prompt_version,
                ],
                [
                    "图片SHA-256",
                    observation.source_image_sha256,
                ],
            ],
        );

        const visibleItems = [];

        for (
            const item
            of observation.observations ?? []
        ) {
            visibleItems.push(
                `${item.description}`
                + `（${item.category}，`
                + `${item.confidence}）`
            );
        }

        for (
            const indicator
            of observation.visible_indicators ?? []
        ) {
            visibleItems.push(
                `${indicator.label}：`
                + `${indicator.observed_state}`
                + `（${indicator.confidence}）`
            );
        }

        if (visibleItems.length > 0) {
            const list =
                document.createElement("ul");

            for (const item of visibleItems) {
                list.append(
                    createTextElement(
                        "li",
                        item,
                    ),
                );
            }

            card.append(list);
        }

        if (
            Array.isArray(
                observation.uncertain_items
            )
            && observation.uncertain_items.length > 0
        ) {
            card.append(
                createTextElement(
                    "p",
                    `无法确认：${
                        observation.uncertain_items.join(
                            "；"
                        )
                    }`,
                ),
            );
        }

        if (
            observation.requires_human_check
        ) {
            card.append(
                createTextElement(
                    "p",
                    "该视觉观察需要人工复核",
                    "field-help",
                ),
            );
        }

        elements.visionObservationList.append(
            card
        );
    }
}


/* =========================================================
   模拟遥测
   ========================================================= */

function renderTelemetry(
    telemetryObservations,
) {
    clearElement(elements.telemetryList);

    if (
        !Array.isArray(
            telemetryObservations
        )
        || telemetryObservations.length === 0
    ) {
        appendEmptyMessage(
            elements.telemetryList,
            "本次运行没有读取模拟遥测",
        );
        return;
    }

    for (
        const telemetry
        of telemetryObservations
    ) {
        const card =
            document.createElement("article");

        card.className = "data-card";

        card.append(
            createTextElement(
                "h4",
                valueOrDash(
                    telemetry.robot_id
                ),
            ),
        );

        appendDefinitionList(
            card,
            [
                [
                    "来源",
                    telemetry.source,
                ],
                [
                    "观测时间",
                    formatDateTime(
                        telemetry.observed_at
                    ),
                ],
                [
                    "模拟位置",
                    telemetry.location,
                ],
                [
                    "运行状态",
                    telemetry.operational_state,
                ],
                [
                    "剩余电量",
                    telemetry.battery_percent
                        === undefined
                            ? null
                            : `${
                                telemetry.battery_percent
                            }%`,
                ],
                [
                    "速度",
                    telemetry.speed_mps
                        === undefined
                            ? null
                            : `${
                                telemetry.speed_mps
                            } m/s`,
                ],
                [
                    "网络连接",
                    telemetry.network_connected
                        ? "已连接"
                        : "未连接",
                ],
                [
                    "当前任务",
                    telemetry.current_task_id,
                ],
                [
                    "活动故障码",
                    Array.isArray(
                        telemetry.active_fault_codes
                    )
                        ? telemetry.active_fault_codes.join(
                            "、"
                        )
                        : null,
                ],
            ],
        );

        elements.telemetryList.append(card);
    }
}


/* =========================================================
   测试草案
   ========================================================= */

function renderTestDrafts(testDrafts) {
    clearElement(elements.testDraftList);

    if (
        !Array.isArray(testDrafts)
        || testDrafts.length === 0
    ) {
        appendEmptyMessage(
            elements.testDraftList,
            "本次运行没有生成测试草案",
        );
        return;
    }

    for (const draft of testDrafts) {
        const card =
            document.createElement("article");

        card.className = "data-card";

        card.append(
            createTextElement(
                "h4",
                valueOrDash(draft.title),
            ),
            createTextElement(
                "p",
                valueOrDash(draft.objective),
            ),
        );

        appendDefinitionList(
            card,
            [
                [
                    "草案状态",
                    draft.draft_only
                        ? "仅为草案"
                        : "未标记",
                ],
                [
                    "人工批准",
                    draft.requires_human_approval
                        ? "必须"
                        : "未要求",
                ],
            ],
        );

        if (
            Array.isArray(draft.preconditions)
            && draft.preconditions.length > 0
        ) {
            card.append(
                createTextElement(
                    "h4",
                    "前置条件",
                ),
            );

            const list =
                document.createElement("ul");

            for (
                const precondition
                of draft.preconditions
            ) {
                list.append(
                    createTextElement(
                        "li",
                        precondition,
                    ),
                );
            }

            card.append(list);
        }

        if (
            Array.isArray(draft.steps)
            && draft.steps.length > 0
        ) {
            card.append(
                createTextElement(
                    "h4",
                    "草案步骤",
                ),
            );

            const list =
                document.createElement("ol");

            for (const step of draft.steps) {
                list.append(
                    createTextElement(
                        "li",
                        `${step.action}；`
                        + `预期观察：`
                        + `${step.expected_observation}`,
                    ),
                );
            }

            card.append(list);
        }

        if (
            Array.isArray(draft.limitations)
            && draft.limitations.length > 0
        ) {
            card.append(
                createTextElement(
                    "p",
                    `限制：${
                        draft.limitations.join("；")
                    }`,
                    "field-help",
                ),
            );
        }

        elements.testDraftList.append(card);
    }
}


/* =========================================================
   工具执行轨迹
   ========================================================= */

function renderToolTrace(steps) {
    clearElement(elements.toolTraceBody);

    if (
        !Array.isArray(steps)
        || steps.length === 0
    ) {
        const row =
            document.createElement("tr");

        const cell =
            createTextElement(
                "td",
                "本次运行没有执行工具",
                "empty-list-message",
            );

        cell.colSpan = 6;

        row.append(cell);
        elements.toolTraceBody.append(row);
        return;
    }

    for (const step of steps) {
        const row =
            document.createElement("tr");

        row.append(
            createTextElement(
                "td",
                step.step_id,
            ),
            createTextElement(
                "td",
                step.tool_name,
            ),
        );

        const statusCell =
            document.createElement("td");

        statusCell.append(
            createStatusBadge(step.status)
        );

        row.append(
            statusCell,
            createTextElement(
                "td",
                step.input_summary,
            ),
            createTextElement(
                "td",
                step.result_summary,
            ),
            createTextElement(
                "td",
                formatDuration(
                    step.duration_ms
                ),
            ),
        );

        elements.toolTraceBody.append(row);
    }
}


/* =========================================================
   完整响应渲染
   ========================================================= */

function renderAgentResponse(responseData) {
    const diagnosis =
        responseData.diagnosis ?? {};

    const execution =
        responseData.execution ?? {};

    showDiagnosisResult();

    elements.resultStatus.textContent =
        `状态：${
            valueOrDash(diagnosis.status)
        }`;

    elements.responseRequestId.textContent =
        valueOrDash(responseData.request_id);

    elements.responseSessionId.textContent =
        valueOrDash(responseData.session_id);

    replaceWithStatusBadge(
        elements.diagnosisStatus,
        diagnosis.status,
    );

    elements.terminationReason.textContent =
        valueOrDash(
            execution.termination_reason
        );

    elements.finishReason.textContent =
        valueOrDash(
            execution.finish_reason
        );

    elements.toolStepCount.textContent =
        String(
            execution.step_count
            ?? execution.steps?.length
            ?? 0
        );

    renderPossibleCauses(
        diagnosis.possible_causes
    );

    renderNextChecks(
        diagnosis.next_checks
    );

    appendStringList(
        elements.missingInformation,
        diagnosis.missing_information,
        "当前结果没有声明缺失信息",
    );

    renderCitations(
        diagnosis.evidence
    );

    renderVisionObservations(
        responseData.vision_observations
    );

    renderTelemetry(
        responseData.telemetry_observations
    );

    renderTestDrafts(
        responseData.test_case_drafts
    );

    renderToolTrace(
        execution.steps
    );
}


/* =========================================================
   最近会话
   ========================================================= */

function renderSessionList(sessionData) {
    clearElement(elements.sessionList);

    const sessions =
        Array.isArray(sessionData.sessions)
            ? sessionData.sessions
            : [];

    elements.sessionListStatus.textContent =
        `显示 ${sessions.length} 条，`
        + `存储中共 ${sessionData.total ?? 0} 条`;

    if (sessions.length === 0) {
        appendEmptyMessage(
            elements.sessionList,
            "目前没有已保存会话",
        );
        return;
    }

    for (const session of sessions) {
        const item =
            document.createElement("li");

        const button =
            document.createElement("button");

        button.type = "button";
        button.className = "session-button";

        button.append(
            createTextElement(
                "strong",
                `${session.robot_id} · `
                + `${session.status}`,
            ),
            createTextElement(
                "span",
                session.symptom_summary,
            ),
            createTextElement(
                "span",
                formatDateTime(
                    session.created_at
                ),
            ),
            createTextElement(
                "span",
                `步骤 ${session.tool_step_count}`
                + ` · 失败工具 ${
                    session.failed_tool_count
                }`
                + ` · 引用 ${
                    session.citation_count
                }`,
            ),
        );

        button.addEventListener(
            "click",
            () => {
                loadSessionDetail(
                    session.session_id,
                    button,
                );
            },
        );

        item.append(button);
        elements.sessionList.append(item);
    }
}


async function loadRecentSessions() {
    elements.sessionListStatus.textContent =
        "正在加载最近会话…";

    elements.refreshSessionsButton.disabled =
        true;

    try {
        const sessionData =
            await requestJson(
                `${DIAGNOSTIC_SESSIONS_API_URL}`
                + "?limit=20",
                {
                    method: "GET",
                    headers: {
                        Accept: "application/json",
                    },
                },
            );

        renderSessionList(sessionData);
    } catch (error) {
        console.error(
            "Loading sessions failed:",
            error,
        );

        elements.sessionListStatus.textContent =
            error instanceof Error
                ? `会话加载失败：${error.message}`
                : "会话加载失败";
    } finally {
        elements.refreshSessionsButton.disabled =
            false;
    }
}


function convertSessionRecordToResponse(
    record,
) {
    /*
    会话详情和实时Agent响应使用不同Schema。

    这里将会话详情中需要展示的字段映射成
    renderAgentResponse()能够处理的页面视图对象。
    不修改服务器返回的原始record。
    */
    return {
        request_id: record.request_id,
        session_id: record.session_id,
        diagnosis: record.diagnosis ?? {},
        execution: {
            state: record.execution_state,
            termination_reason:
                record.termination_reason,

            finish_reason:
                record.finish_reason,

            step_count:
                record.metrics?.tool_step_count
                ?? record.tool_events?.length
                ?? 0,

            steps:
                record.tool_events ?? [],

            missing_information:
                record.diagnosis
                    ?.missing_information
                ?? [],
        },
        vision_observations:
            record.vision_observations ?? [],

        telemetry_observations:
            record.telemetry_observations ?? [],

        test_case_drafts:
            record.test_case_drafts ?? [],
    };
}


async function loadSessionDetail(
    sessionId,
    selectedButton,
) {
    const sessionButtons =
        elements.sessionList.querySelectorAll(
            ".session-button"
        );

    for (const button of sessionButtons) {
        button.classList.remove(
            "is-selected"
        );
    }

    selectedButton.classList.add(
        "is-selected"
    );

    elements.resultStatus.textContent =
        "正在加载会话详情…";

    try {
        const record =
            await requestJson(
                `${DIAGNOSTIC_SESSIONS_API_URL}`
                + `/${encodeURIComponent(sessionId)}`,
                {
                    method: "GET",
                    headers: {
                        Accept: "application/json",
                    },
                },
            );

        renderAgentResponse(
            convertSessionRecordToResponse(
                record
            )
        );

        elements.resultStatus.textContent =
            "已加载保存的会话";
    } catch (error) {
        showResultError(error);
    }
}


/* =========================================================
   表单提交
   ========================================================= */

async function handleDiagnosisSubmit(event) {
    /*
    preventDefault()阻止浏览器执行传统表单跳转，
    让JavaScript使用fetch发送JSON。
    */
    event.preventDefault();

    setDiagnosisLoading(true);

    elements.errorPanel.hidden = true;
    elements.emptyState.hidden = false;
    elements.diagnosisResult.hidden = true;

    elements.resultStatus.textContent =
        "正在执行Agent诊断…";

    try {
        const payload =
            await buildDiagnosisRequest();

        const responseData =
            await requestJson(
                AGENT_DIAGNOSIS_API_URL,
                {
                    method: "POST",
                    headers: {
                        "Content-Type":
                            "application/json",

                        Accept:
                            "application/json",
                    },

                    body:
                        JSON.stringify(payload),
                },
            );

        renderAgentResponse(responseData);

        /*
        新诊断已生成会话记录，
        所以成功后刷新最近会话列表。
        */
        await loadRecentSessions();
    } catch (error) {
        console.error(
            "Diagnosis request failed:",
            error,
        );

        showResultError(error);
    } finally {
        setDiagnosisLoading(false);
    }
}


/* =========================================================
   页面事件注册
   ========================================================= */

elements.form.addEventListener(
    "submit",
    handleDiagnosisSubmit,
);

elements.imageFiles.addEventListener(
    "change",
    updateImageSummary,
);

elements.form.addEventListener(
    "reset",
    () => {
        /*
        reset事件触发时，浏览器尚未完成字段复位。
        queueMicrotask()把页面清理安排到当前同步代码之后。
        */
        queueMicrotask(
            () => {
                updateImageSummary();

                elements.errorPanel.hidden =
                    true;

                elements.diagnosisResult.hidden =
                    true;

                elements.emptyState.hidden =
                    false;

                elements.resultStatus.textContent =
                    "尚未执行诊断";
            },
        );
    },
);

elements.refreshSessionsButton.addEventListener(
    "click",
    loadRecentSessions,
);


/* =========================================================
   页面初始化
   ========================================================= */

/*
脚本使用defer加载，因此执行到这里时HTML已经解析完成。

健康检查和会话列表互不依赖，
可以同时开始，不必串行等待。
*/
void Promise.all([
    checkServiceHealth(),
    loadRecentSessions(),
]);