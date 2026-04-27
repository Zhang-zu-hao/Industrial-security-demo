# Docker 部署指南

本文档详细说明如何在 NVIDIA Jetson 设备上使用 Docker 部署 Industrial Security Demo。

## 前置条件

- NVIDIA Jetson 设备（Orin NX / Orin Nano）
- JetPack 6.x（Ubuntu 22.04）
- Docker Engine 已安装（可使用 `install-docker-jetson.sh` 脚本）
- NVIDIA Container Runtime 已配置

## 快速开始

### 1. 构建镜像

```bash
cd Industrial-security-demo
docker build --network=host -t industrial-security-demo:latest .
```

> **注意**：使用 `--network=host` 确保构建过程中能下载依赖。

### 2. 运行容器

```bash
docker compose up -d
```

### 3. 访问 Web 面板

浏览器打开：`http://<Jetson-IP>:8080`

### 4. 查看日志

```bash
docker compose logs -f
```

## 配置说明

### 摄像头设备

默认情况下，容器不挂载摄像头设备。如需使用 USB 摄像头，编辑 `docker-compose.yml`：

```yaml
devices:
  # 取消注释以下行
  - /dev/video0:/dev/video0
```

### RTSP 摄像头

如果使用 RTSP 网络摄像头，无需修改设备映射，只需在 `config/demo_config.json` 中配置 RTSP URL：

```json
{
  "camera": {
    "source": "rtsp://admin:@192.168.3.10/Streaming/Channels/101",
    "use_gstreamer": true
  }
}
```

### 自定义配置

配置文件通过 volume 挂载到容器内，修改宿主机上的 `config/demo_config.json` 后重启容器即可生效：

```bash
docker compose restart
```

## TensorRT GPU 加速

容器通过 bind-mount 方式使用宿主机的 TensorRT/CUDA 库，无需在镜像内安装完整 CUDA Toolkit。

### 挂载的库

- `/usr/local/cuda` - CUDA Toolkit
- `/usr/lib/tensorrt` - TensorRT 库
- `/usr/lib/python3/dist-packages/tensorrt*` - TensorRT Python 绑定
- `/dev/nvidia*` - NVIDIA GPU 设备

### 验证 GPU 加速

查看容器日志，应显示：

```
[TRT] Loading engine from /app/models/yolo26n_fp16.engine...
✅ YOLO TensorRT ready (e2e head, 640px)
```

如果显示 `Using OpenCV DNN CPU fallback`，说明 TensorRT 未正确加载。

## 离线部署

### 导出镜像

```bash
bash scripts/docker-export.sh industrial-security-demo:latest ./industrial-security-demo.tar.gz
```

### 导入镜像

```bash
gunzip -c industrial-security-demo.tar.gz | docker load
```

## 故障排查

### TensorRT 初始化失败

**症状**：日志显示 `TensorRT init failed, falling back to OpenCV DNN CPU`

**解决方案**：
1. 确认 JetPack 6.x 已正确安装
2. 检查 TensorRT Python 绑定路径是否正确：
   ```bash
   ls -la /usr/lib/python3/dist-packages/tensorrt*
   ```
3. 如果路径不同，修改 `docker-compose.yml` 中的 volume 映射

### 设备文件不存在

**症状**：容器启动失败，提示 `/dev/nvidia0: no such file or directory`

**解决方案**：
1. 确认 NVIDIA 驱动已加载：
   ```bash
   ls -la /dev/nvidia*
   ```
2. 如果没有设备文件，加载内核模块：
   ```bash
   sudo modprobe nvidia
   ```

### 摄像头无法访问

**症状**：无法打开视频源

**解决方案**：
1. 确认摄像头设备存在：
   ```bash
   ls -la /dev/video*
   ```
2. 在 `docker-compose.yml` 中取消注释对应的设备映射
3. 如果是 RTSP 摄像头，检查网络连接

### 端口冲突

**症状**：`Address already in use`

**解决方案**：
1. 检查端口占用：
   ```bash
   ss -ltnp | grep 8080
   ```
2. 修改 `docker-compose.yml` 中的端口映射：
   ```yaml
   ports:
     - "8081:8080"  # 将容器 8080 映射到宿主机 8081
   ```

## 性能优化

### 使用硬件解码

确保 `config/demo_config.json` 中：

```json
{
  "camera": {
    "use_gstreamer": true
  }
}
```

### 调整推理间隔

如果帧率过高，可以增加推理间隔：

```json
{
  "detector": {
    "infer_interval": 2  # 每 2 帧推理一次
  }
}
```

## 安全注意事项

1. **不要在生产环境暴露 Web 端口到公网**
2. 如需外部访问，配置防火墙：
   ```bash
   sudo ufw allow from 192.168.1.0/24 to any port 8080
   ```
3. 考虑添加反向代理和 HTTPS

## 与 SenseCraft Solution 集成

此项目可以作为 SenseCraft Solution 平台的一个方案集成。参考 `app_collaboration` 项目中的 `integrate-jetson-solution` skill 自动生成方案配置。

### 集成步骤

1. 在 `solutions/` 目录下创建新方案
2. 使用 `docker_remote` 部署器类型
3. 配置 `docker-compose.yml` 作为部署文件
4. 编写方案描述和部署指南

详细参考：`app_collaboration/.claude/skills/integrate-jetson-solution/SKILL.md`
