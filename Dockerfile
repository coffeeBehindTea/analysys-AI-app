# 使用官方Python 3.11精简版Linux镜像作为运行基础。
#
# bookworm明确固定Debian发行版，避免slim标签以后切换
# 到其他Debian版本而导致系统包名称或路径发生变化。
FROM python:3.11-slim-bookworm


# 设置容器内Python和系统工具的运行环境。
#
# PYTHONDONTWRITEBYTECODE：
# 不在容器中生成__pycache__和.pyc文件。
#
# PYTHONUNBUFFERED：
# 让日志立即写到标准输出，便于docker logs实时查看。
#
# PIP_DISABLE_PIP_VERSION_CHECK：
# 安装依赖时不执行无关的pip版本检查。
#
# PIP_NO_CACHE_DIR：
# 不在最终镜像中保存pip下载缓存。
#
# TESSDATA_PREFIX：
# 指向Debian安装Tesseract语言模型的位置，
# OCR Provider会优先读取这个环境变量。
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata


# 后续COPY、RUN和CMD默认以/app作为工作目录。
#
# 应用中的相对路径：
#
# chroma_data
# data/diagnostic_sessions
#
# 会分别解析为：
#
# /app/chroma_data
# /app/data/diagnostic_sessions
WORKDIR /app


# 安装Python库之外的Linux系统依赖。
#
# ca-certificates：
# 让Python能够验证LLM、Embedding和Vision服务的HTTPS证书。
#
# libgomp1：
# 为部分科学计算和向量相关二进制库提供OpenMP运行时。
#
# tesseract-ocr：
# 安装Tesseract OCR可执行程序。
#
# tesseract-ocr-eng：
# 安装英文OCR语言模型。
#
# tesseract-ocr-chi-sim：
# 安装简体中文OCR语言模型。
#
# --no-install-recommends避免安装无关推荐包。
# 最后删除apt索引，减小镜像体积。
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        libgomp1 \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-chi-sim \
    && rm -rf /var/lib/apt/lists/*


# 先只复制依赖清单。
#
# Docker会按指令生成可缓存的镜像层。
# 只要requirements.txt没有变化，修改普通Python代码后
# 再次构建时通常可以复用Python依赖安装层。
COPY requirements.txt ./requirements.txt


# 在镜像内使用当前Python解释器对应的pip安装固定版本依赖。
#
# python -m pip比直接执行pip更明确：
# 它保证使用的pip属于当前python:3.11基础镜像。
RUN python -m pip install \
    --no-cache-dir \
    --requirement ./requirements.txt


# 创建专用的非root用户。
#
# UID和GID固定为10001，使镜像中的文件身份更稳定。
# /usr/sbin/nologin表示该用户不能作为交互式登录账号使用。
RUN groupadd \
        --gid 10001 \
        appuser \
    && useradd \
        --uid 10001 \
        --gid 10001 \
        --create-home \
        --home-dir /home/appuser \
        --shell /usr/sbin/nologin \
        appuser


# 只复制生产运行需要的源码。
#
# --chown在复制时把文件所有者设为appuser，
# 使非root进程能够正常读取应用代码。
#
# app目录包括：
# - Router
# - Service
# - Agent
# - Schema
# - Web控制台静态文件
COPY --chown=appuser:appuser app ./app

# main.py是Uvicorn加载FastAPI应用的入口。
COPY --chown=appuser:appuser main.py ./main.py

# 保留运维和文档摄取脚本，使获得授权语料后
# 可以在容器中运行摄取或冒烟验证。
COPY --chown=appuser:appuser scripts ./scripts


# 创建需要写入的运行时目录。
#
# Chroma会在/app/chroma_data中读写向量数据库。
# 会话Store会在/app/data/diagnostic_sessions中保存脱敏记录。
#
# Compose稍后会把宿主机目录挂载到这两个位置。
RUN mkdir --parents \
        /app/chroma_data \
        /app/data/diagnostic_sessions \
    && chown --recursive \
        appuser:appuser \
        /app/chroma_data \
        /app/data


# 从这一行开始，后续进程都以非root用户运行。
#
# 即使应用或第三方库出现安全问题，
# 进程也不会默认拥有容器内root权限。
USER appuser


# 声明应用预计监听8000端口。
#
# EXPOSE只描述镜像使用的端口，
# 它本身不会把端口映射到Windows宿主机。
# 真正的端口映射会由Compose完成。
EXPOSE 8000


# Docker定期请求FastAPI健康检查接口。
#
# interval：
# 每30秒检查一次。
#
# timeout：
# 一次检查最多等待5秒。
#
# start-period：
# 容器启动后的前20秒属于启动缓冲期。
#
# retries：
# 连续失败3次后把容器标记为unhealthy。
#
# 使用Python标准库urllib.request，不额外安装curl。
# urlopen()遇到非2xx响应或连接失败时会抛出异常，
# Docker会据此把本次健康检查判定为失败。
HEALTHCHECK \
    --interval=30s \
    --timeout=5s \
    --start-period=20s \
    --retries=3 \
    CMD ["python", "-c", "import urllib.request; response = urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5); response.close()"]


# 定义容器默认启动命令。
#
# python -m uvicorn：
# 使用当前Python环境中的Uvicorn模块。
#
# main:app：
# 导入main.py中的app对象。
#
# --host 0.0.0.0：
# 监听容器的所有网络接口。
# 如果使用默认127.0.0.1，宿主机端口映射将无法访问服务。
#
# 不使用--reload：
# 容器运行采用稳定服务模式，不启动开发用自动重载子进程。
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
