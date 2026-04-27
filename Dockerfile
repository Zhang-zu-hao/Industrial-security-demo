# Industrial Security Demo - Optimized Docker Image
# Features:
#   - No NGC login required (uses Ubuntu official image)
#   - Smaller image size (~500MB vs ~5GB)
#   - Better compatibility across JetPack versions
#   - Uses host TensorRT/CUDA via NVIDIA runtime (bind-mounted from host)
#
# Build: docker build --network=host -t industrial-security-demo:latest .
# Run:   docker compose up -d

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Shanghai \
    # CUDA/TensorRT 环境变量（通过 volume 挂载宿主机库）
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:/usr/lib/tensorrt:${LD_LIBRARY_PATH:-} \
    PATH=/usr/local/cuda/bin:${PATH}

# 安装基础依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-numpy \
    python3-opencv \
    # GStreamer 硬件解码支持
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    libgstrtspserver-1.0-0 \
    # TensorRT 运行时依赖（库文件从宿主机挂载）
    libnvinfer-plugin10 \
    libnvinfer10 \
    libnvonnxparsers10 \
    libcudnn8 \
    # 清理
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# 安装 Python 依赖
COPY requirements-docker.txt .
RUN pip3 install --no-cache-dir --break-system-packages -r requirements-docker.txt \
    || pip3 install --no-cache-dir -r requirements-docker.txt

# 复制应用代码
COPY . .

# 创建输出目录
RUN mkdir -p /app/output

ENTRYPOINT ["python3", "-u", "app/behavior_demo.py"]
CMD ["--config", "config/demo_config.json", "--no-window"]
