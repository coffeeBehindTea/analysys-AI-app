"""Vision图片输入异常到HTTP公开错误的离线测试。

本文件只调用异常元数据映射函数，
不启动FastAPI服务器，也不处理真实HTTP请求。
"""

from starlette import status

from app.errors import (
    VisionInputValidationError,
)
from app.exception_handlers import (
    get_error_metadata,
)


def test_vision_input_error_maps_to_safe_422_response(
) -> None:
    """非法图片应映射成稳定422，而不是暴露内部细节。"""

    exception = VisionInputValidationError(
        # 这是只应写入服务端日志的内部原因。
        "声明image/png但真实格式为JPEG"
    )

    status_code, error_code, public_message = (
        get_error_metadata(exception)
    )

    assert status_code == (
        status.HTTP_422_UNPROCESSABLE_CONTENT
    )
    assert error_code == (
        "vision_input_validation_error"
    )
    assert public_message == (
        "图片输入不符合视觉分析要求"
    )

    # 公开信息不得包含攻击者提交的格式细节，
    # 具体原因只通过服务端日志关联request_id排查。
    assert str(exception) not in public_message
