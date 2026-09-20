"""Robot Diagnostic Agent的HTTP路由。

本模块负责：

1. 声明普通JSON与SSE流式诊断接口；
2. 让FastAPI校验AgentDiagnosisRequest请求体；
3. 通过依赖注入取得普通或流式诊断Service；
4. 从Request.state读取Middleware生成的request_id；
5. 普通接口返回AgentDiagnosisResponse；
6. 流式接口逐条编码并发送公开Agent事件。

本模块不调用LLM、不执行工具、不访问知识库，
也不构造最终诊断报告。
"""

from collections.abc import (
    AsyncIterator,
)
from typing import (
    Annotated,
)

from fastapi import (
    APIRouter,
    Depends,
    Request,
)
from fastapi.responses import (
    StreamingResponse,
)

from app.dependencies import (
    get_agent_diagnosis_service,
    get_agent_diagnosis_streaming_service,
)
from app.schemas.agent_api import (
    AgentDiagnosisRequest,
    AgentDiagnosisResponse,
)
from app.schemas.error import (
    ErrorResponse,
)
from app.services.agent_diagnosis_service import (
    AgentDiagnosisService,
)
from app.services.agent_diagnosis_streaming_service import (
    AgentDiagnosisStreamingService,
)
from app.services.agent_sse_encoder import (
    AGENT_SSE_CACHE_CONTROL,
    AGENT_SSE_CONTENT_TYPE,
    encode_agent_sse_event,
)


# prefix会添加到本文件所有路由路径前。
#
# 当前接口装饰器使用"/diagnose"，
# 因此最终路径是：
#
# POST /api/v1/agent/diagnose
router = APIRouter(
    prefix="/api/v1/agent",
    tags=["agent"],
)


@router.post(
    "/diagnose",

    # response_model的作用包括：
    #
    # 1. 校验Router成功返回值；
    # 2. 把Pydantic对象序列化成JSON；
    # 3. 过滤Schema中不存在的内部字段；
    # 4. 为Swagger/OpenAPI生成响应文档。
    response_model=AgentDiagnosisResponse,

    # responses只描述可能出现的HTTP错误，
    # 不负责捕获异常。
    #
    # Planner运行期间的超时、无效输出和工具失败，
    # 通常会由Runner或Service转换成200状态下的
    # aborted/abstained业务响应。
    #
    # 这里主要描述依赖创建或底层基础服务
    # 在请求进入业务方法前后产生的应用错误。
    responses={
        502: {
            "model": ErrorResponse,
            "description": (
                "LLM、Embedding或知识库上游"
                "返回了错误或无效响应"
            ),
        },
        503: {
            "model": ErrorResponse,
            "description": (
                "LLM、Embedding或知识库配置缺失，"
                "或者基础服务暂时不可用"
            ),
        },
        504: {
            "model": ErrorResponse,
            "description": (
                "LLM或Embedding上游请求超时"
            ),
        },
    },
)
async def diagnose_with_agent(
    # 没有Query、Path或Header来源标记的
    # Pydantic模型参数会被FastAPI解析为JSON请求体。
    #
    # 在函数执行前，FastAPI会校验：
    #
    # 1. robot_id是否存在；
    # 2. symptom和log_excerpt是否非空；
    # 3. task_goal是否合法；
    # 4. 是否出现tools、max_steps等额外字段。
    payload: AgentDiagnosisRequest,

    # Request由FastAPI直接提供，
    # 不属于客户端JSON请求体。
    #
    # 它包含请求头、URL、state等当前HTTP请求信息。
    http_request: Request,

    # Depends告诉FastAPI：
    #
    # 此参数不是客户端提供的，而是通过
    # get_agent_diagnosis_service()装配得到。
    service: Annotated[
        AgentDiagnosisService,
        Depends(
            get_agent_diagnosis_service
        ),
    ],
) -> AgentDiagnosisResponse:
    """执行一次受控Robot Diagnostic Agent诊断。"""

    # RequestIdMiddleware在Router运行前已经执行：
    #
    # request.state.request_id = request_id
    #
    # state是当前请求专属的临时状态容器，
    # 不会跨请求共享。
    request_id = (
        http_request.state.request_id
    )

    # diagnose()是异步方法。
    #
    # await等待Agent循环完成，同时允许事件循环
    # 继续处理其他HTTP请求。
    return await service.diagnose(
        payload,
        request_id=request_id,
    )


async def _encode_agent_diagnosis_stream(
    *,
    payload: AgentDiagnosisRequest,
    request_id: str,
    service: AgentDiagnosisStreamingService,
) -> AsyncIterator[str]:
    """把结构化Agent事件逐条转换成SSE文本帧。

    StreamingResponse会使用async for消费本异步生成器。
    客户端断开连接时，Starlette会关闭这个生成器；
    关闭动作继续传递给流式Service，由它取消后台诊断任务。
    """

    async for event in service.stream(
        payload,
        request_id,
    ):
        # 编码器统一生成id、event和data字段，
        # 并在每个事件末尾添加一个空行作为SSE帧边界。
        yield encode_agent_sse_event(event)


@router.post(
    "/diagnose/stream",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": (
                "按执行顺序返回Robot Diagnostic "
                "Agent公开SSE事件"
            ),
            "content": {
                "text/event-stream": {
                    "schema": {
                        "type": "string",
                    },
                },
            },
        },
        502: {
            "model": ErrorResponse,
            "description": (
                "事件流开始前，上游服务返回错误"
            ),
        },
        503: {
            "model": ErrorResponse,
            "description": (
                "事件流开始前，必要配置或服务不可用"
            ),
        },
        504: {
            "model": ErrorResponse,
            "description": (
                "事件流开始前，上游服务响应超时"
            ),
        },
    },
)
async def diagnose_with_agent_stream(
    payload: AgentDiagnosisRequest,
    http_request: Request,
    service: Annotated[
        AgentDiagnosisStreamingService,
        Depends(
            get_agent_diagnosis_streaming_service
        ),
    ],
) -> StreamingResponse:
    """启动一次诊断并以SSE持续返回公开执行事件。"""

    request_id = (
        http_request.state.request_id
    )

    # StreamingResponse不会在这里一次性等待完整诊断。
    # ASGI服务器开始响应后，会逐条消费body_iterator，
    # 每产生一个字符串就立即向客户端发送一个SSE事件。
    return StreamingResponse(
        _encode_agent_diagnosis_stream(
            payload=payload,
            request_id=request_id,
            service=service,
        ),
        media_type=AGENT_SSE_CONTENT_TYPE,
        headers={
            # 禁止浏览器或中间代理缓存旧事件。
            "Cache-Control": (
                AGENT_SSE_CACHE_CONTROL
            ),

            # Nginx识别此响应头后不会聚合多个事件，
            # 能保持工具开始/结束等事件及时到达客户端。
            "X-Accel-Buffering": "no",
        },
    )
