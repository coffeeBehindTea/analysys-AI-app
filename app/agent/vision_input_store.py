"""当前Agent请求使用的已验证图片输入存储。

本模块负责建立：

image_ref → VisionInput

之间的请求级映射。

它不负责：

1. 接收HTTP请求；
2. 解码Base64；
3. 使用Pillow验证图片；
4. 调用Vision Provider；
5. 保存图片到磁盘；
6. 把图片暴露给Planner；
7. 在不同HTTP请求之间共享图片。

进入本Store的VisionInput必须已经通过
VisionInputAdapter的格式、大小、尺寸和摘要检查。
"""

from pydantic import (
    TypeAdapter,
    ValidationError,
)

from app.schemas.agent_tools import (
    AgentImageReference,
)
from app.schemas.vision import (
    VisionInput,
)


# TypeAdapter允许直接校验Annotated类型，
# 不必为了校验单独一个image_ref再创建BaseModel。
#
# 这里会复用AgentImageReference中声明的：
#
# 1. 首尾空白清理；
# 2. 长度限制；
# 3. image_前缀；
# 4. 字符范围限制。
_IMAGE_REFERENCE_ADAPTER = TypeAdapter(
    AgentImageReference
)


def _normalize_image_reference(
    image_ref: object,
) -> str:
    """校验并规范化一个请求级图片引用。

    本函数是Store内部使用的统一入口，
    register()和get()不会各自实现不同的校验逻辑。
    """

    # Python类型注解不会在运行时自动阻止：
    #
    # image_ref=123
    #
    # 因此先做明确的类型检查。
    if not isinstance(image_ref, str):
        raise TypeError(
            "image_ref必须是字符串"
        )

    try:
        # validate_python()使用
        # AgentImageReference中的Pydantic约束
        # 校验普通Python对象。
        #
        # 返回值已经完成首尾空白清理。
        normalized_ref = (
            _IMAGE_REFERENCE_ADAPTER
            .validate_python(image_ref)
        )
    except ValidationError as exc:
        # 对Store调用者提供稳定、明确的错误。
        #
        # 原始Pydantic异常仍然通过异常链保留，
        # 调试时可以检查具体违反了哪项约束。
        raise ValueError(
            "image_ref不符合请求级图片引用格式"
        ) from exc

    return normalized_ref


class RequestVisionInputStore:
    """保存一次Agent请求中已经验证的图片输入。

    Store只保存当前请求生命周期内的内存引用，
    不会把图片写入文件、数据库或全局缓存。

    请求隔离最终由FastAPI依赖生命周期保证：
    每个HTTP请求必须创建一个新的Store实例，
    不能对依赖使用lru_cache。
    """

    def __init__(
        self,
    ) -> None:
        """创建一个空的请求级图片映射。"""

        # 键是经过规范化的image_ref。
        #
        # 值是已经通过VisionInputAdapter检查的
        # VisionInput，而不是外部原始Base64字符串。
        self._vision_inputs: dict[
            str,
            VisionInput,
        ] = {}

    def register(
        self,
        *,
        image_ref: str,
        vision_input: VisionInput,
    ) -> None:
        """登记一张当前请求已经验证的图片。

        相同image_ref不能被重复登记，
        防止Planner已经看到引用后，
        该引用又被替换成另一张图片。
        """

        # 先完成引用规范化。
        #
        # 例如"  image_primary  "会被保存成
        # "image_primary"。
        normalized_ref = (
            _normalize_image_reference(
                image_ref
            )
        )

        # 类型注解不会自动检查运行时对象。
        #
        # Store只接受内部VisionInput，
        # 不接受VisionImagePayload、dict、
        # Base64字符串或任意图片字节。
        if not isinstance(
            vision_input,
            VisionInput,
        ):
            raise TypeError(
                "vision_input必须是VisionInput"
            )

        # 禁止覆盖已经存在的引用。
        #
        # 即使新对象表示同一张图片也不允许覆盖，
        # 因为重复登记通常意味着装配流程存在错误。
        if normalized_ref in self._vision_inputs:
            raise ValueError(
                "image_ref已经在当前请求中登记："
                f"{normalized_ref}"
            )

        # 所有校验完成后才写入字典。
        #
        # 如果前面的任意检查失败，
        # Store不会留下部分写入的状态。
        self._vision_inputs[
            normalized_ref
        ] = vision_input

    def get(
        self,
        image_ref: str,
        /,
    ) -> VisionInput | None:
        """按精确图片引用读取已验证输入。

        未知但格式合法的引用返回None。

        None表示当前请求没有登记这张图片，
        不表示Vision Provider发生了错误。
        """

        normalized_ref = (
            _normalize_image_reference(
                image_ref
            )
        )

        # dict.get()在键不存在时返回None。
        #
        # 这里不进行模糊匹配、大小写转换，
        # 也不会尝试把引用解释成文件路径。
        return self._vision_inputs.get(
            normalized_ref
        )

    def __len__(
        self,
    ) -> int:
        """返回当前请求已经登记的图片数量。"""

        return len(self._vision_inputs)