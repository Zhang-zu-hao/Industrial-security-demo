# Industrial Security Demo

<p align="center">
  <b>Edge AI-Powered Industrial Security Monitoring on NVIDIA Jetson</b>
</p>

面向 **Seeed reComputer Industrial** 系列 Jetson 边缘设备的工业安防演示：**多摄像头** RTSP/USB 接入、**TensorRT FP16** 人员检测、质心跟踪、可交互绘制的区域入侵/越线/徘徊规则，**SQLite 事件持久化**，以及浏览器实时监控面板。

***

## Why Edge AI? 项目亮点

| 优势 | 说明 |
|------|------|
| **数据安全，隐私合规** | 全链路本地推理，视频流与事件数据不出厂区/园区，满足工业安全与隐私合规要求。无需将敏感视频上传云端 |
| **TensorRT FP16 加速** | 利用 Jetson GPU + TensorRT 进行 FP16 量化推理，YOLO26n 延迟仅 ~3.7ms（268 QPS），实时性远超云端方案 |
| **NMS-Free 端到端推理** | 支持最新 Ultralytics YOLO26，原生无需 NMS 后处理，进一步降低延迟，专为边缘场景优化 |
| **GStreamer 硬件解码** | Jetson NVDEC 硬解 RTSP 视频流，CPU 几乎零开销 |
| **多摄像头支持** | 同时接入多路 RTSP 摄像头，独立处理管线，共享检测模型，Web 端自适应网格布局 |
| **离线部署，低带宽** | 不依赖互联网，适合矿山、工厂、仓库、工地等无网/弱网环境 |
| **灵活二次开发** | 支持自训练模型 (YOLOv5/v8/v11/v26)、自定义规则、REST API 对接，开箱即用也能深度定制 |
| **交互式区域配置** | 浏览器中直接在视频画面上绘制检测区域，每个摄像头独立配置，无需修改配置文件 |
| **事件持久化** | SQLite 数据库存储事件，支持历史查询、按日期筛选，重启不丢失 |

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
- [多摄像头管理](#多摄像头管理)
- [HDMI 显示与全屏切换](#hdmi-显示与全屏切换)
- [Web 端区域绘制](#web-端区域绘制)
- [事件持久化与查询](#事件持久化与查询)
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
| 多摄像头接入 | 支持**多路 RTSP/USB** 摄像头同时接入，独立处理管线，共享检测模型 |
| 摄像头自动发现 | 自动扫描子网内的 RTSP 摄像头，支持 Web 端手动添加/移除 |
| Jetson NVDEC 硬解 | GStreamer 管线解码，降低 CPU 占用（可关闭回退软解） |
| 人员检测 | 默认 **YOLO26n → TensorRT FP16**（NMS-free 端到端推理）；兼容 YOLOv5/v8/v11 ONNX |
| 目标跟踪 | 质心跟踪（`CentroidTracker`），可配置距离与超时 |
| 行为规则 | 区域入侵、越线、徘徊（支持浏览器交互绘制检测区域，**每摄像头独立配置**） |
| Web 面板 | 静态页面 + WebSocket 低延迟视频流 + 实时配置推送，自适应网格布局 |
| 事件持久化 | **SQLite 数据库**存储事件，支持按日期查询，重启不丢失，自动清理过期数据 |
| 事件记录 | `events.jsonl` + `output/<cam-id>/events/` 截图，支持按日期筛选 |
| HDMI 全屏显示 | 启动后自动检测 HDMI 并全屏显示，按 **F** 键切换全屏/窗口 |
| 摄像头健康监控 | 自动检测离线摄像头并重连 |

***

## 系统架构

```
RTSP-1 ──► GStreamer NVDEC ──► AsyncCapture ──┐
RTSP-2 ──► GStreamer NVDEC ──► AsyncCapture ──┤
  ...                                          │
                                               ▼
                              YOLO26 TensorRT FP16 推理
                               (共享模型, 串行推理)
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
              Pipeline-1      Pipeline-2       Pipeline-N
              (跟踪+规则)     (跟踪+规则)      (跟踪+规则)
                    │               │               │
                    ▼               ▼               ▼
              EventStore       EventStore       EventStore
              (SQLite)         (SQLite)         (SQLite)
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
  OpenCV 本地显示          Web：HTTP + WS
 (HDMI 自动全屏,         (视频流 + 配置 +
  F键切换全屏)           摄像头管理面板)
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
│   ├── behavior_demo.py         # 主程序：采集、检测、跟踪、规则、HDMI显示
│   ├── multi_camera_manager.py  # 多摄像头管线管理、帧缓冲、事件存储
│   ├── camera_discovery.py      # 摄像头自动发现与手动添加
│   ├── event_store_db.py        # SQLite 事件持久化存储
│   ├── yolo_trt_detector.py     # TensorRT / DNN 检测器（v5/v8/v11/v26 自动识别）
│   ├── web_server.py            # 轻量 HTTP + MJPEG（stdlib）
│   └── web_server_optimized.py  # 优化 WebSocket 视频 + 配置 + 摄像头管理 API
├── config/
│   └── demo_config.json         # 摄像头、检测器、规则、显示、Web 配置
├── models/
│   ├── yolo26n.onnx             # 默认 YOLO26n ONNX（NMS-free）
│   ├── yolo26n_fp16.engine      # TensorRT FP16 引擎（设备上构建）
│   ├── yolov5n.onnx             # 可选 YOLOv5n
│   └── yolov8n.onnx             # 可选 YOLOv8n
├── web/
│   └── index.html               # Web 面板（多摄像头、区域绘制、WS 视频流）
├── output/
│   ├── cam-0/                   # 摄像头0的事件数据
│   │   ├── events.jsonl         # 事件日志
│   │   ├── events.db            # SQLite 事件数据库
│   │   └── events/              # 事件截图
│   └── cam-1/                   # 摄像头1的事件数据
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

### 摄像头 `cameras`

| 字段 | 含义 |
|------|------|
| `mode` | `manual`：手动配置；`auto`：自动扫描子网 |
| `manual` | 手动摄像头列表，每项包含 `id`、`name`、`source`、`use_gstreamer`、`enabled` |
| `auto_discover.subnet` | 自动扫描的子网，如 `192.168.3.0/24` |
| `auto_discover.username` | RTSP 用户名 |
| `auto_discover.password` | RTSP 密码 |
| `auto_discover.rtsp_paths` | 尝试的 RTSP 路径列表 |
| `auto_discover.scan_interval_seconds` | 扫描间隔（秒） |

示例配置：

```json
"cameras": {
  "mode": "manual",
  "manual": [
    {
      "id": "cam-0",
      "name": "poe-camera-1",
      "source": "rtsp://admin:@192.168.3.10/Streaming/Channels/101",
      "use_gstreamer": true,
      "enabled": true
    },
    {
      "id": "cam-1",
      "name": "poe-camera-2",
      "source": "rtsp://admin:@192.168.3.20/Streaming/Channels/101",
      "use_gstreamer": true,
      "enabled": true
    }
  ]
}
```

> 也可通过 Web 端"添加摄像头"功能动态添加，无需修改配置文件。

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

### 显示 `display`

| 字段 | 含义 |
|------|------|
| `show_window` | 是否显示 OpenCV 本地窗口 |
| `window_name` | 窗口标题 |
| `resize_width` | 视频缩放宽度（像素） |
| `web_jpeg_quality` | Web 端 JPEG 编码质量（1-100，默认 50，越低延迟越低） |

***

## 多摄像头管理

### 架构设计

每个摄像头运行独立的处理管线（`CameraPipeline`），包含：

- **AsyncCapture**：独立线程读取视频帧
- **CentroidTracker**：独立跟踪器
- **EventStore**：独立事件存储（SQLite）
- **FrameBuffer**：线程安全的帧缓冲

所有摄像头共享一个 **TensorRT 检测模型**（`SharedDetector`），串行推理避免 GPU 竞争。

### Web 端管理

- **摄像头列表**：顶部标签页切换不同摄像头
- **自适应布局**：1 摄像头全屏、2 摄像头左右分屏、3-4 摄像头 2×2 网格
- **添加摄像头**：在控制面板中输入 IP、用户名、密码、RTSP 路径，点击"探测并添加"
- **移除摄像头**：通过 API 移除指定摄像头
- **独立区域配置**：每个摄像头有独立的检测区域，互不影响

### 自动发现

配置 `cameras.mode = "auto"` 后，系统会定期扫描子网内的 RTSP 摄像头，自动添加新发现的摄像头。

### 健康监控

每个摄像头管线内置健康检查：
- 10 秒无帧 → 标记为离线
- 自动尝试重连
- Web 端显示摄像头在线/离线状态

### 统计数据平滑

系统对检测人数、追踪目标、FPS 等关键指标进行了双重平滑优化：

**后端滑动窗口**（`StatsCollector`）：
- 维护最近 5 次更新的历史记录队列
- 每次更新时加入新值，移除最旧值
- 返回队列平均值（FPS）或平均取整（检测数、追踪数）
- 有效过滤瞬时波动，数据更稳定

**前端动画过渡**（`pollStats`）：
- 每次更新只移动 30% 的差值（`SMOOTH_FACTOR = 0.3`）
- 数值变化小于 0.5 时直接显示目标值
- 切换摄像头时重置平滑状态，避免显示旧数据

**效果**：数值从突变跳动变为平滑渐变，Web 端和 HDMI 端均受益。

***

## HDMI 显示与全屏切换

### 自动全屏

启动时自动检测 HDMI 显示器连接状态（读取 `/sys/class/drm/card0-HDMI-A-1/status`），检测到 HDMI 时自动全屏显示。

### 全屏/窗口切换

在 HDMI 显示窗口中按 **F** 键可切换全屏和窗口模式：
- **全屏模式**：适合监控大屏部署
- **窗口模式**：适合开发调试

### 多摄像头布局

HDMI 显示自动适配摄像头数量：
- 1 个摄像头：全屏显示
- 2 个摄像头：左右分屏
- 3-4 个摄像头：2×2 网格
- 更多摄像头：3 列网格

### 实时 HUD 信息

每个摄像头画面上叠加显示实时状态信息：

| 位置 | 内容 | 说明 |
|------|------|------|
| **左上角** | 🟢/🔴 状态指示灯 | 绿色=在线，红色=离线 |
| **右上角** | 摄像头名称 | 如 `poe-camera-1`（绿色） |
| **左侧** | FPS | 实时帧率（滑动窗口平滑） |
| **左侧** | Tracks | 当前追踪目标数（平滑） |
| **左侧** | Detections | 当前检测人数（平滑） |
| **左侧** | Events | 累计事件数 |

> 所有统计数据经过后端滑动窗口（最近 5 次）和前端动画过渡（30% 渐变）双重平滑处理，数值变化稳定无跳动。

### 字体自适应

所有 HUD 文字大小根据画面宽度动态计算，全屏和窗口模式下均清晰可读，最小字体保护避免负值报错。

***

## Web 端区域绘制

### 绘制检测区域

1. 打开浏览器访问 `http://<Jetson的IP>:8080`
2. 在顶部标签页选择要配置的摄像头
3. 在右侧控制面板找到"检测区域"部分
4. 点击"启用区域绘制"开关
5. 在视频画面上点击鼠标左键绘制多边形顶点（至少 3 个点）
6. 绘制完成后自动保存，或点击"完成绘制"/"取消"按钮

> **重要**：每个摄像头的检测区域是独立的，切换摄像头时会自动加载该摄像头的区域配置。

### 管理检测区域

- **查看区域列表**：已绘制的区域会显示在控制面板中
- **删除单个区域**：点击区域旁边的 ✕ 按钮
- **清空所有区域**：点击"清空所有区域"按钮（仅清空当前摄像头的区域）

### 区域参数

- **徘徊时间**：在"检测规则"中设置徘徊检测的时间阈值（秒）
- **冷却时间**：同类事件的最小间隔时间（秒）

***

## 事件持久化与查询

### SQLite 存储

所有事件自动写入 SQLite 数据库（`output/<cam-id>/events.db`），支持：

- **持久化**：应用重启后事件不丢失
- **按日期查询**：通过 API 或 Web 端按日期筛选事件
- **自动清理**：超过 30 天的事件自动清理
- **双写机制**：同时写入内存环形缓冲区和 SQLite，查询优先使用 SQLite

### 事件数据结构

每个事件包含：

| 字段 | 说明 |
|------|------|
| `timestamp` | 事件时间戳（`YYYYMMDD-HHMMSS` 格式） |
| `camera_id` | 摄像头 ID（如 `cam-0`） |
| `camera_name` | 摄像头名称（如 `poe-camera-1`） |
| `event_type` | 事件类型：`zone_enter`、`loitering`、`line_cross` |
| `track_id` | 目标跟踪 ID |
| `zone_name` / `line_name` | 触发的区域/线名称 |
| `dwell_seconds` | 徘徊时长（仅徘徊事件） |
| `bbox` | 目标边界框 |
| `centroid` | 目标质心坐标 |

### 事件截图

每个事件自动保存截图到 `output/<cam-id>/events/` 目录，最多保留 200 张，超出后自动清理最旧的截图。

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
| `/api/cameras` | GET | 摄像头列表（ID、名称、状态） |
| `/api/cameras/<id>/stream` | GET | 指定摄像头的 MJPEG 视频流 |
| `/api/cameras/<id>/stats` | GET | 指定摄像头的实时统计 |
| `/api/cameras/<id>/events` | GET | 指定摄像头的事件列表 |
| `/api/cameras/add` | POST | 添加摄像头（需 IP、用户名、密码等） |
| `/api/cameras/remove` | POST | 移除摄像头（需 `camera_id`） |
| `/api/cameras/discover` | GET | 触发摄像头自动发现 |
| `/api/cameras/probe` | POST | 探测单个摄像头可达性 |
| `/api/stats` | GET | 全局统计（汇总所有摄像头） |
| `/api/events` | GET | 事件列表，支持 `?date=YYYYMMDD` 筛选 |
| `/api/events/images` | GET | 事件截图列表，支持 `?date=YYYYMMDD` 筛选 |
| `/api/events/img/<cam>/<name>` | GET | 事件截图图片 |
| `/api/events/clear` | POST | 清空事件日志 |
| `/api/config` | GET/POST | 读取/更新运行时配置 |
| `/api/rules` | POST | 更新检测规则（区域、线等），支持按摄像头配置 |
| `/api/models` | GET | 可用模型列表 |
| `/api/model/switch` | POST | 运行时切换模型 |

WebSocket 端口：
- `:8081` — 视频流（二进制 JPEG 帧）
- `:8082` — 配置通道（JSON 双向）

***

## Web 与优化模式

安装 `websockets` 后自动启用优化模式：

- **多摄像头网格**：自适应 1/2/3 列布局，标签页切换
- **双 WebSocket**：视频流与配置通道分离，互不阻塞
- **二进制帧协议**：视频帧使用二进制 WebSocket 传输，包含摄像头 ID 和时间戳
- **动态 JPEG 质量**：配置 `web_jpeg_quality` 调节 Web 端画质
- **交互式区域绘制**：在视频上直接画多边形检测区域，每摄像头独立配置
- **功能开关**：浏览器中实时开关检测/跟踪/区域/越线/徘徊
- **实时事件流**：带摄像头标签的事件列表，按日期筛选
- **摄像头管理**：Web 端添加/移除摄像头，探测可达性
- **自动重连**：WebSocket 断开后自动重连（指数退避）

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
| HDMI 窗口不全屏 | 检查 DISPLAY 环境变量，确保 X11 服务正常运行；按 F 键手动切换全屏 |
| Web 端卡顿 | 降低 `web_jpeg_quality`（默认 50），检查网络带宽 |
| 事件不显示 | 检查是否已绘制检测区域并启用 `zone_detection`/`loitering` 功能 |
| 摄像头离线 | 检查网络连接，系统会自动重连；查看 Web 端摄像头状态 |
| 区域绘制影响其他摄像头 | 每个摄像头区域独立存储，切换摄像头时自动加载对应配置 |

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
