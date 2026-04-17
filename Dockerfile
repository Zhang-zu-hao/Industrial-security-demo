# Industrial Security Demo - Optimized Docker Image
# Features:
#   - No NGC login required (uses Ubuntu official image)
#   - Smaller image size (~500MB vs ~5GB)
#   - Better compatibility across JetPack versions
#   - Uses host TensorRT/CUDA via NVIDIA runtime
#
# Build: docker build -t industrial-security-demo:latest .
# Run:   docker-compose up -d

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Shanghai

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-numpy \
    python3-opencv \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    libgstrtspserver-1.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

COPY requirements-docker.txt .
RUN pip3 install --no-cache-dir --break-system-packages -r requirements-docker.txt \
    || pip3 install --no-cache-dir -r requirements-docker.txt

COPY . .

RUN mkdir -p /app/output

ENTRYPOINT ["python3", "-u", "app/behavior_demo.py"]
CMD ["--config", "config/demo_config.json", "--no-window"]
