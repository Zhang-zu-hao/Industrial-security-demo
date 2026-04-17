# Industrial Security Demo

面向 **NVIDIA Jetson**（reComputer Industrial 等）的工业安防演示：RTSP 摄像头接入、**TensorRT FP16** 人员检测、质心跟踪、区域入侵 / 越线 / 徘徊规则，以及浏览器监控面板（HTTP + 可选 WebSocket 低延迟流）。

---

## 目录

- [功能概览](#功能概览)
- [系统架构](#系统架构)
- [运行环境与依赖](#运行环境与依赖)
- [仓库结构](#仓库结构)
- [快速开始](#快速开始)
- [部署教程](#部署教程)
- [配置说明](#配置说明)
- [Web 与优化模式](#web-与优化模式)
- [模型与 TensorRT 引擎](#模型与-tensorrt-引擎)
- [API 与事件输出](#api-与事件输出)
- [命令行参数](#命令行参数)
- [故障排查](#故障排查)
- [开发与扩展](#开发与扩展)

---

## 功能概览

| 能力 | 说明 |
|------|------|
| RTSP / USB 视频源 | 支持 `rtsp://...` 与本地摄像头索引（如 `0`） |
| Jetson NVDEC 硬解 | GStreamer 管线解码，降低 CPU 占用（可关闭回退软解） |
| 人员检测 | 默认 **YOLOv5n ONNX → TensorRT FP16**；检测器支持 YOLOv5 / YOLOv8 等 Ultralytics 导出 ONNX（见 `app/yolo_trt_detector.py`） |
| 跟踪 | 质心跟踪（`CentroidTracker`），可配置距离与超时 |
| 行为规则 | 区域入侵、越线、徘徊（依赖跟踪与功能开关） |
| 本地窗口 | OpenCV 窗口预览（需 `DISPLAY`） |
| Web 面板 | 静态页面 + MJPEG；若安装 `websockets` 则启用 **优化服务**（双 WebSocket + 动态 JPEG 质量等） |
| 事件 | `events.jsonl` + `output/events/` 截图 |

---

## 系统架构

```
RTSP ──► GStreamer (可选) ──► AsyncCapture 线程读帧
                                    │
                                    ▼
                          检测 (TensorRT / HOG)
                                    │
                                    ▼
                          跟踪 + 规则引擎
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
            OpenCV 本地显示                    Web：HTTP / WS
```

- **主进程**：`app/behavior_demo.py` 中的 `BehaviorDemo` 循环读帧、推理、画框、写事件。
- **Web**：
  - 优先加载 `app/web_server_optimized.py`（需 `pip install websockets`）：HTTP 页面 + 视频 WebSocket + 配置 WebSocket，低延迟 JPEG 推送。
  - 若 `import web_server_optimized` 失败，回退到 `app/web_server.py`（仅标准库 + MJPEG `/api/stream`）。

---

## 运行环境与依赖

### 硬件

- **测试设备**: NVIDIA Jetson Orin Nano 8G (Seeed Industrial J3011)
- NVIDIA Jetson（文档与测试目标：**Orin 系列** + reComputer Industrial）
- 可选：PoE 网口连接 IP 摄像头、HDMI 本地调试

### 软件（典型 JetPack 镜像）

- **JetPack 6.x**（Ubuntu 22.04）
- **Python 3.10+**
- **OpenCV** 带 **GStreamer** 后端（`cv2.getBuildInformation()` 中 GStreamer: YES）
- **TensorRT 10.x**（与 JetPack 匹配的 Python 包 `tensorrt`）

### 支持的摄像头设备

#### 当前测试使用的摄像头

- **品牌/类型**: 明创达网络摄像机
- **型号**: MCD-300W
- **分辨率**: 300 万像素（3MP），IP 高清彩色摄像机
- **支持协议**: ONVIF 协议（可兼容大部分 NVR 和第三方监控平台）

> ⚠️ **重要提示**: 不推荐使用明创达 MCD-300W 型号
> 
> 该摄像头**没有提供 ARM 架构的 SDK**，仅支持标准 ONVIF/RTSP 协议。虽然可以通过 RTSP 流进行基本监控，但无法使用摄像头的高级功能（如 PTZ 控制、特定事件检测等）。
> 
> **推荐方案**: 建议开发者选择支持 ARM 架构 SDK 的摄像头品牌，例如：
> - 海康威视（Hikvision）部分型号
> - 大华（Dahua）部分型号
> - 宇视（Uniview）部分型号
> 
> 这些品牌通常提供更完善的 ARM SDK，可以充分发挥项目中的传输协议和高级功能。

#### 项目支持的传输协议

本项目支持以下摄像头接入方式：

1. **RTSP 协议**（推荐）
   - 标准 RTSP 流的 IP 摄像头
   - 通过 ONVIF 协议自动发现摄像头
   - 适用于所有支持 RTSP 的网络摄像机

2. **USB 摄像头**
   - 标准 UVC 协议的 USB 摄像头
   - Jetson 兼容的 USB 视觉模块

3. **MIPI CSI 摄像头**
   - Jetson 原生的 MIPI CSI 接口摄像头模块
   - 需要特定的设备树配置和驱动支持

### Python 额外包

- **默认路径（推荐）**：不安装额外 pip 包即可运行检测 + 基础 Web（`web_server.py`）。
- **优化 Web**：需要安装 `websockets`，否则自动使用简易版 HTTP 服务：

```bash
pip3 install --user websockets
```

---

## 仓库结构

```
industrial-security-demo/
├── app/
│   ├── behavior_demo.py        # 主程序：采集、检测、跟踪、规则、显示
│   ├── yolo_trt_detector.py   # TensorRT / OpenCV DNN 检测器（多 ONNX 头格式）
│   ├── web_server.py          # 轻量 HTTP + MJPEG（stdlib）
│   └── web_server_optimized.py # 优化：WebSocket 视频 + 配置（需 websockets）
├── config/
│   └── demo_config.json       # 摄像头、检测器、规则、显示、Web
├── models/
│   ├── yolov5n.onnx           # 默认 YOLOv5n ONNX
│   └── yolov5n_fp16.engine    # 首次 TRT 推理生成或 trtexec 编译
├── web/
│   ├── index.html             # 监控面板（优化路径下由前端使用）
│   └── index_optimized.html   # 可选备用页面
├── output/
│   ├── events.jsonl           # 事件日志
│   └── events/                # 事件截图（可选上限与清理逻辑见代码）
├── scripts/
│   └── probe_camera.py        # RTSP 路径探测
├── build_yolov8_engine.py    # 使用 trtexec 为 YOLOv8n 构建 engine（可选）
├── run_demo.sh                # 一键启动（设置 DISPLAY / 字体等）
└── README.md
```

---

## 快速开始

```bash
cd industrial-security-demo

# 确认模型存在（默认 yolov5n）
ls -la models/

# 若缺少 ONNX，可下载（示例为官方资源链接，以实际可用地址为准）
wget -O models/yolov5n.onnx \
  https://github.com/ultralytics/assets/releases/download/v7.0/yolov5n.onnx
```

启动（使用 `config/demo_config.json`）：

```bash
bash run_demo.sh
```

仅后台 + Web（无本地窗口）：

```bash
python3 app/behavior_demo.py --no-window
```

指定端口（覆盖配置中的 `web.port`）：

```bash
python3 app/behavior_demo.py --no-window --web-port 8080
```

短时间自检（跑固定帧数后退出）：

```bash
python3 app/behavior_demo.py --no-window --no-web --max-frames 100
```

---

## 部署教程

### 1. 网络与摄像头

- 保证 Jetson 与摄像头 **IP 互通**（同一网段或正确路由）。
- 常见做法：为连接摄像头的网口配置静态 IP，例如：

```bash
# 示例：网口名以实际为准（如 enP8p1s0）
sudo nmcli connection add type ethernet ifname <网口名> \
  con-name camera-lan ipv4.method manual \
  ipv4.addresses 192.168.1.2/24 ipv6.method disabled
sudo nmcli connection up camera-lan
ping -c 3 <摄像头IP>
```

### 2. 探测 RTSP 地址

```bash
python3 scripts/probe_camera.py --ip <摄像头IP> --user admin --password ""
```

将输出的可用 `rtsp://...` 写入 `config/demo_config.json` 的 `camera.source`。

### 3. 防火墙与端口

默认 HTTP：**8080**。若使用优化 Web 服务，启动日志会打印类似：

```text
[Web] HTTP:8080 WS:8081 ConfigWS:8082
```

需在主机防火墙放行对应端口，例如：

```bash
sudo ufw allow 8080/tcp
sudo ufw allow 8081/tcp
sudo ufw allow 8082/tcp
```

### 4. 远程访问面板

浏览器打开：

```text
http://<Jetson的IP>:8080
```

### 5. 无显示器 / SSH 启动

使用 `--no-window`；若需本地 HDMI 窗口，`run_demo.sh` 会尝试设置 `DISPLAY` 与 `XAUTHORITY`。

### 6. 长期运行（可选）

可使用 `systemd` 将 `python3 app/behavior_demo.py --no-window` 注册为服务，注意工作目录设为项目根、`User` 与 `WorkingDirectory` 一致，并处理日志轮转。

---

## 配置说明

配置文件：`config/demo_config.json`。

### 摄像头 `camera`

| 字段 | 含义 |
|------|------|
| `source` | RTSP URL 或摄像头索引字符串 `"0"` |
| `use_gstreamer` | `true` 时对 `rtsp://` 使用 GStreamer 硬解管线 |

### 检测器 `detector`

| 字段 | 含义 |
|------|------|
| `backend` | `yolov5_trt`：TensorRT 引擎；`opencv_hog`：不依赖模型，CPU HOG 行人 |
| `onnx_file` | `models/` 下 ONNX 文件名 |
| `conf_threshold` / `iou_threshold` | 置信度与 NMS |
| `fp16` | TensorRT FP16 推理 |
| `infer_interval` | 每 N 帧推理一次，大于 1 可降低算力占用 |

### 跟踪 `tracker`

- `max_distance`：匹配同一目标的最大像素距离  
- `max_age_seconds`：丢失后保留轨迹的时间  

### 规则 `rules`

- `zones`：多边形顶点为 **归一化坐标** `[0,1]`  
- `lines`：`start` / `end` 同样为归一化坐标  
- `event_cooldown_seconds`：同类事件冷却时间  

### 功能开关 `features`（可选）

未配置时，代码内默认全部为 `true`。可按需关闭以减负：

```json
"features": {
  "human_detect": true,
  "tracking": true,
  "zone_detection": true,
  "line_crossing": true,
  "loitering": false
}
```

### 显示 `display`

- `resize_width`：按宽度缩放，0 表示不缩放  
- `jpeg_quality`：优化 Web 路径下 JPEG 质量（1–100），见前端与 `web_server_optimized`  

### Web `web`

- `port`：HTTP 端口（默认 8080）  
- `ws_port` / `config_ws_port`：优化服务使用（若未设置，代码中会用 `port+1`、`port+2` 作为默认）  

---

## Web 与优化模式

### 简易模式（`web_server.py`）

- 依赖少：标准库 HTTP + OpenCV JPEG  
- 视频：`GET /api/stream`（MJPEG）  
- 事件：`GET /api/events`，统计：`GET /api/stats`，配置：`GET/POST /api/config`、`POST /api/rules`  

### 优化模式（`web_server_optimized.py`）

需 `websockets`：

- **双 WebSocket**：视频流与配置分离，减少互相阻塞  
- **动态 JPEG 质量**、帧缓冲策略、功能开关与统计由前端与后端协同（详见 `web/index.html` 与优化服务器实现）  
- 若优化模块不可用，自动回退简易模式，不中断主程序  

---

## 模型与 TensorRT 引擎

1. 将 ONNX 放在 `models/` 下，在 `demo_config.json` 中设置 `detector.onnx_file`。  
2. 检测器会查找同名 `*_fp16.engine`（见 `YOLOv5TRTDetector` 内逻辑）；若不存在会尝试构建或回退 DNN（见代码与日志）。  
3. **YOLOv8 专用构建脚本**（可选）：`build_yolov8_engine.py` 调用系统 `trtexec` 生成 `yolov8n_fp16.engine`（路径与文件名以脚本内为准）。  

引擎与 **不同 Jetson 设备**混用可能触发 TensorRT 警告或错误；请在目标设备上生成 engine。

---

## API 与事件输出

- 事件以 **JSON Lines** 写入 `output/events.jsonl`（路径由 `output.root` 决定）。  
- 每条含时间戳、`event_type`、`track_id`、`bbox`、`centroid` 等。  
- 截图保存在 `output/events/`，数量过多时实现侧可能清理旧文件（见 `EventWriter`）。  

---

## 命令行参数

`python3 app/behavior_demo.py`：

| 参数 | 说明 |
|------|------|
| `--config` | 配置文件路径，默认 `config/demo_config.json` |
| `--source` | 覆盖配置中的视频源 |
| `--max-frames` | 运行若干帧后退出 |
| `--no-window` | 不显示 OpenCV 窗口 |
| `--no-web` | 不启动 Web |
| `--web-port` | 覆盖 HTTP 端口 |

`run_demo.sh` 会把参数原样传给 `behavior_demo.py`。

---

## 故障排查

| 现象 | 处理方向 |
|------|----------|
| `Cannot open video source` | 检查 RTSP URL、`ping` 摄像头、`probe_camera.py`；尝试 `use_gstreamer: false` |
| TensorRT 初始化失败 | 查看是否缺 engine、ONNX 是否匹配；必要时重新 trtexec 生成 |
| `Address already in use`（8080） | 更换 `--web-port` 或结束占用进程：`ss -ltnp \| grep 8080` |
| 优化 Web 不生效 | 安装 `pip3 install --user websockets`，查看启动日志是否只有 `HTTP:端口` |
| 无画面 / DISPLAY | SSH 无桌面时用 `--no-window`，仅用 Web |
| 检测框与规则不触发 | 确认 `features` 中 `human_detect`/`tracking` 与规则开关 |

---

## 开发与扩展

- **新检测后端**：在 `create_detector()` 中扩展分支，或替换 ONNX/TRT 流程。  
- **规则**：在 `BehaviorDemo._apply_rules` 中扩展事件类型（注意与 `features` 配合）。  
- **前端**：静态资源在 `web/`，修改后刷新浏览器即可（强缓存时可硬刷新）。  

---

## 许可证与声明

若仓库内另有 `LICENSE` 文件，以该文件为准。本文档描述的是演示项目行为，生产环境请补充安全（HTTPS、鉴权、审计）与运维规范。
