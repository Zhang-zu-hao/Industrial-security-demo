# Industrial Security Demo

<p align="center">
  <b>Edge AI-Powered Industrial Security Monitoring on NVIDIA Jetson</b>
</p>

面向 **Seeed reComputer Industrial** 系列 Jetson 边缘设备的工业安防演示：RTSP/USB 摄像头接入、**TensorRT FP16** 人员检测、质心跟踪、可交互绘制的区域入侵/越线/徘徊规则，以及浏览器实时监控面板。

***

## Why Edge AI? 项目亮点

| 优势 | 说明 |
|------|------|
| **数据安全，隐私合规** | 全链路本地推理，视频流与事件数据不出厂区/园区，满足工业安全与隐私合规要求。无需将敏感视频上传云端 |
| **TensorRT FP16 加速** | 利用 Jetson GPU + TensorRT 进行 FP16 量化推理，YOLO26n 延迟仅 ~3.7ms（268 QPS），实时性远超云端方案 |
| **NMS-Free 端到端推理** | 支持最新 Ultralytics YOLO26，原生无需 NMS 后处理，进一步降低延迟，专为边缘场景优化 |
| **GStreamer 硬件解码** | Jetson NVDEC 硬解 RTSP 视频流，CPU 几乎零开销 |
| **离线部署，低带宽** | 不依赖互联网，适合矿山、工厂、仓库、工地等无网/弱网环境 |
| **灵活二次开发** | 支持自训练模型 (YOLOv5/v8/v11/v26)、自定义规则、REST API 对接，开箱即用也能深度定制 |
| **交互式区域配置** | 浏览器中直接在视频画面上绘制检测区域，无需修改配置文件 |

***

## 当前测试设备

| 项目 | 详情 |
|------|------|
| **设备** | [Seeed reComputer Industrial J401](https://www.seeedstudio.com/reComputer-Industrial-J4012-p-5684.html) |
| **SoM** | NVIDIA Jetson Orin NX 16GB (p3767-0000-super) |
| **JetPack** | 6.2 (L4T R36.4.3, Ubuntu 22.04) |
| **GPU** | Ampere, 1024 CUDA cores, TensorRT 10.3.0 |

> **兼容性**：本项目适配 Seeed reComputer Industrial 全系列 Jetson 设备（Orin NX / Orin Nano 等），以及其他运行 JetPack 6.x 的 Jetson 平台。

***

## 目录

- [功能概览](#功能概览)
- [系统架构](#系统架构)
- [运行环境与依赖](#运行环境与依赖)
- [快速开始](#快速开始)
- [部署教程](#部署教程)
- [配置说明](#配置说明)
- [模型与 TensorRT 引擎](#模型与-tensorrt-引擎)
- [自训练模型与二次开发](#自训练模型与二次开发)
- [API 与事件输出](#api-与事件输出)
- [Web 与优化模式](#web-与优化模式)
- [命令行参数](#命令行参数)
- [故障排查](#故障排查)
- [Docker on Jetson](#docker-on-jetson)

***

## 功能概览

| 能力 | 说明 |
|------|------|
| RTSP / USB 视频源 | 支持 `rtsp://...` 与本地摄像头索引（如 `0`） |
| Jetson NVDEC 硬解 | GStreamer 管线解码，降低 CPU 占用（可关闭回退软解） |
| 人员检测 | 默认 **YOLO26n → TensorRT FP16**（NMS-free 端到端推理）；兼容 YOLOv5/v8/v11 ONNX |
| 目标跟踪 | 质心跟踪（`CentroidTracker`），可配置距离与超时 |
| 行为规则 | 区域入侵、越线、徘徊（支持浏览器交互绘制检测区域） |
| Web 面板 | 静态页面 + WebSocket 低延迟视频流 + 实时配置推送 |
| 事件记录 | `events.jsonl` + `output/events/` 截图，支持按日期筛选 |

***

## 系统架构

```
RTSP ──► GStreamer NVDEC ──► AsyncCapture 线程读帧
                                   │
                                   ▼
                      YOLO26 TensorRT FP16 推理
                            (NMS-free, ~3.7ms)
                                   │
                                   ▼
                         质心跟踪 + 规则引擎
                                   │
                   ┌───────────────┴───────────────┐
                   ▼                               ▼
           OpenCV 本地显示                    Web：HTTP + WS
                                            (视频流 + 配置)
```

***

## 运行环境与依赖

### 硬件

- **推荐设备**：[Seeed reComputer Industrial](https://www.seeedstudio.com/reComputer-Industrial-c-2013.html) 系列（Jetson Orin NX / Orin Nano）
- 支持任何运行 JetPack 6.x 的 NVIDIA Jetson 设备
- 可选：PoE 网口连接 IP 摄像头

### 软件

- **JetPack 6.x**（Ubuntu 22.04）
- **Python 3.10+**
- **OpenCV** 带 GStreamer 后端
- **TensorRT 10.x**（JetPack 自带）
- **CUDA 12.x**（JetPack 自带）

### Python 额外包

```bash
pip3 install --user numpy websockets
```

> TensorRT / CUDA / cuDNN 由 JetPack 系统提供，无需 pip 安装。

***

## 仓库结构

```
Industrial-security-demo/
├── app/
│   ├── behavior_demo.py         # 主程序：采集、检测、跟踪、规则、显示
│   ├── yolo_trt_detector.py     # TensorRT / DNN 检测器（v5/v8/v11/v26 自动识别）
│   ├── web_server.py            # 轻量 HTTP + MJPEG（stdlib）
│   └── web_server_optimized.py  # 优化 WebSocket 视频 + 配置（需 websockets）
├── config/
│   └── demo_config.json         # 摄像头、检测器、规则、显示、Web 配置
├── models/
│   ├── yolo26n.onnx             # 默认 YOLO26n ONNX（NMS-free）
│   ├── yolo26n_fp16.engine      # TensorRT FP16 引擎（设备上构建）
│   ├── yolov5n.onnx             # 可选 YOLOv5n
│   └── yolov8n.onnx             # 可选 YOLOv8n
├── web/
│   ├── index.html
│   └── index_optimized.html     # 优化版 Web 面板（区域绘制、WS 视频流）
├── output/
│   ├── events.jsonl             # 事件日志
│   └── events/                  # 事件截图
├── scripts/
│   └── probe_camera.py          # RTSP 路径探测
├── build_yolov8_engine.py       # 可选 trtexec 构建脚本
├── run_demo.sh                  # 一键启动
└── README.md
```

***

## 快速开始

```bash
cd Industrial-security-demo

# 1. 安装依赖
pip3 install --user numpy websockets

# 2. 确认模型存在
ls -la models/

# 3. 启动（使用配置文件）
bash run_demo.sh

# 或仅后台 + Web（无本地窗口，适合 SSH）
python3 app/behavior_demo.py --no-window
```

浏览器打开：`http://<Jetson的IP>:8080`

### 短时间自检

```bash
python3 app/behavior_demo.py --no-window --no-web --max-frames 100
```

***

## 部署教程

### 1. 网络与摄像头

确保 Jetson 与摄像头在同一网段：

```bash
ping -c 3 <摄像头IP>
```

### 2. 探测 RTSP 地址

```bash
python3 scripts/probe_camera.py --ip <摄像头IP> --user admin --password ""
```

将输出的 `rtsp://...` 写入 `config/demo_config.json` 的 `camera.source`。

### 3. 构建 TensorRT 引擎（首次/换设备时）

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=models/yolo26n.onnx \
  --saveEngine=models/yolo26n_fp16.engine \
  --fp16
```

> 引擎绑定硬件，不同 Jetson 设备之间不可混用，需在目标设备上构建。

### 4. 防火墙

```bash
sudo ufw allow 8080/tcp
sudo ufw allow 8081/tcp
sudo ufw allow 8082/tcp
```

### 5. 长期运行

可使用 `systemd` 注册为服务：

```bash
python3 app/behavior_demo.py --no-window
```

***

## 配置说明

配置文件：`config/demo_config.json`

### 摄像头 `camera`

| 字段 | 含义 |
|------|------|
| `source` | RTSP URL 或摄像头索引 `"0"` |
| `use_gstreamer` | `true` 使用 GStreamer NVDEC 硬解 |

### 检测器 `detector`

| 字段 | 含义 |
|------|------|
| `backend` | `yolov5_trt`：TensorRT 推理 |
| `onnx_file` | `models/` 下 ONNX 文件名，如 `yolo26n.onnx` |
| `conf_threshold` | 置信度阈值 |
| `iou_threshold` | NMS IoU 阈值（v5/v8 使用，v26 NMS-free 忽略） |
| `fp16` | TensorRT FP16 推理 |
| `infer_interval` | 每 N 帧推理一次 |

### 规则 `rules`

- `zones`：多边形顶点为归一化坐标 `[0,1]`，可在浏览器中交互绘制
- `lines`：`start` / `end` 为归一化坐标
- `event_cooldown_seconds`：同类事件冷却时间

### 功能开关 `features`

```json
"features": {
  "human_detect": true,
  "tracking": true,
  "zone_detection": true,
  "line_crossing": true,
  "loitering": false
}
```

***

## 模型与 TensorRT 引擎

### 支持的模型

| 模型 | 输出格式 | NMS | 说明 |
|------|----------|-----|------|
| **YOLO26n** (默认) | `(1, 300, 6)` | 内置 (NMS-free) | 最新架构，边缘最优 |
| YOLOv5n | `(1, 25200, 85)` | 后处理 | 经典轻量 |
| YOLOv8n | `(1, 84, 8400)` | 后处理 | 精度/速度平衡 |
| YOLO11n | 同 v8 | 后处理 | v8 架构升级 |

### 性能对比 (Jetson Orin NX 16G, FP16)

| 模型 | GPU 延迟 | 吞吐量 |
|------|----------|--------|
| YOLO26n | 3.72ms | 268 QPS |
| YOLOv5n | 2.96ms | 337 QPS |
| YOLOv8n | 3.89ms | 256 QPS |

### 引擎构建

检测器首次运行时自动构建引擎（调用 `trtexec`），也可手动：

```bash
/usr/src/tensorrt/bin/trtexec --onnx=models/<model>.onnx --saveEngine=models/<model>_fp16.engine --fp16
```

***

## 自训练模型与二次开发

### 使用自训练 YOLO 模型

1. 在任意机器上训练 YOLO 模型并导出 ONNX：

```python
from ultralytics import YOLO
model = YOLO("yolo26n.pt")
model.train(data="your_dataset.yaml", epochs=100)
model.export(format="onnx", imgsz=640)
```

2. 将导出的 `.onnx` 复制到 `models/` 目录
3. 在 `config/demo_config.json` 中修改 `detector.onnx_file`
4. 首次运行时自动构建 TensorRT 引擎

### 扩展开发

- **新检测后端**：在 `app/behavior_demo.py` 的 `create_detector()` 中扩展
- **新规则**：在 `BehaviorDemo._apply_rules` 中添加事件类型
- **前端定制**：修改 `web/` 下的 HTML/CSS/JS，刷新浏览器即可
- **API 对接**：使用 REST API 获取实时数据，对接上层平台

***

## API 与事件输出

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/stats` | GET | 实时统计（FPS、检测数、跟踪数、事件数） |
| `/api/events` | GET | 事件列表，支持 `?date=YYYYMMDD` 筛选 |
| `/api/events/images` | GET | 事件截图列表，支持 `?date=YYYYMMDD` 筛选 |
| `/api/events/img/<name>` | GET | 事件截图图片 |
| `/api/config` | GET/POST | 读取/更新运行时配置 |
| `/api/models` | GET | 可用模型列表 |
| `/api/model/switch` | POST | 运行时切换模型 |
| `/api/stream` | GET | MJPEG 视频流（HTTP 回退） |
| `/api/events/clear` | POST | 清空事件日志 |

WebSocket 端口：
- `:8081` — 视频流（二进制 JPEG 帧）
- `:8082` — 配置通道（JSON 双向）

***

## Web 与优化模式

安装 `websockets` 后自动启用优化模式：

- **双 WebSocket**：视频流与配置通道分离，互不阻塞
- **动态 JPEG 质量**：浏览器滑块实时调节
- **交互式区域绘制**：在视频上直接画多边形检测区域
- **功能开关**：浏览器中实时开关检测/跟踪/区域/越线/徘徊
- **实时事件流**：带缩略图的事件列表，按日期筛选

若 `websockets` 不可用，自动回退到 HTTP MJPEG 模式。

***

## 命令行参数

| 参数 | 说明 |
|------|------|
| `--config` | 配置文件路径，默认 `config/demo_config.json` |
| `--source` | 覆盖配置中的视频源 |
| `--max-frames` | 运行若干帧后退出 |
| `--no-window` | 不显示 OpenCV 窗口 |
| `--no-web` | 不启动 Web 服务 |
| `--web-port` | 覆盖 HTTP 端口 |

***

## 故障排查

| 现象 | 处理方向 |
|------|----------|
| `Cannot open video source` | 检查 RTSP URL、ping 摄像头、使用 `probe_camera.py` |
| TensorRT 初始化失败 | 检查 engine 文件是否在当前设备上构建 |
| `Address already in use` | `--web-port` 换端口或 `ss -ltnp \| grep 8080` |
| YOLO26 DNN 回退失败 | YOLO26 NMS-free 需要 TensorRT，不支持 OpenCV DNN 回退 |
| 无画面 / DISPLAY | SSH 时用 `--no-window`，仅用 Web |
| 区域/越线不触发 | 检查 `features` 中 `zone_detection`/`line_crossing` 开关 |

***

## Docker on Jetson

### 镜像特点

- **无需 NGC 登录**：基于 `ubuntu:22.04`
- **体积极小**：压缩后仅 ~333 MB
- **适配性强**：不绑定特定 L4T 版本

### 快速开始

```bash
# 构建
docker build --network=host -t industrial-security-demo:latest .

# 运行
docker compose up -d

# 访问
http://<Jetson-IP>:8080
```

### 离线部署

```bash
# 导出
bash scripts/docker-export.sh industrial-security-demo:latest ./industrial-security-demo.tar.gz

# 导入
gunzip -c industrial-security-demo.tar.gz | docker load
```

***

## 许可证与声明

若仓库内另有 `LICENSE` 文件，以该文件为准。本文档描述的是演示项目行为，生产环境请补充安全（HTTPS、鉴权、审计）与运维规范。
