#!/usr/bin/env bash
# 将已构建的镜像导出为 tar.gz，便于离线拷贝到其它 Jetson
# 用法：bash scripts/docker-export.sh [镜像名:tag] [输出文件]
set -euo pipefail

IMAGE="${1:-industrial-security-demo:latest}"
OUT="${2:-industrial-security-demo_image.tar.gz}"

echo "Exporting ${IMAGE} -> ${OUT}"
docker save "${IMAGE}" | gzip -1 > "${OUT}"
ls -lh "${OUT}"
echo "Done. Copy to target Jetson and run: gunzip -c ${OUT} | docker load"
