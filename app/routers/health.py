"""服务健康检查路由。"""

# APIRouter 是一组相关 route 的容器，稍后由 main.py 安装到主应用。
from fastapi import APIRouter


# tags 只影响 OpenAPI/Swagger 中的接口分组，不改变 URL。
router = APIRouter(tags=["health"])


# @router.get() 装饰器登记 GET 方法和 /health 路径之间的映射。
@router.get("/health")
async def health_check() -> dict[str, str]:
    """确认当前 API 进程能够接收并响应 HTTP 请求。"""

    # FastAPI 会把这个 Python dict 自动序列化成 JSON 响应。
    return {"status": "ok"}
