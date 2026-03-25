# Industrial Security Demo

基于 **reComputer Industrial J3011** (Orin Nano 8GB / 40 TOPS) 的工业级安防行为检测系统。

## 📖 项目简介

本项目是一个完整的智能视频监控解决方案，利用 NVIDIA Jetson Orin Nano 的强大 AI 算力，实现对监控画面中人员行为的实时分析和告警。系统采用 TensorRT 加速的 YOLOv5 人员检测模型，结合高效的目标跟踪算法，能够准确识别并追踪画面中的人员，同时检测多种异常行为。

### 核心特性

- **高性能 AI 推理**：TensorRT FP16 加速，YOLOv5n 模型在 Orin Nano 上实现实时检测
- **硬件编解码**：GStreamer + NVDEC 硬件解码，大幅降低 CPU 占用
- **智能行为分析**：支持区域入侵、越线检测、徘徊检测等多种安防场景
- **实时 Web 监控**：浏览器即可访问的远程监控面板，支持中英文界面
- **事件记录追溯**：自动保存事件日志和截图，便于事后查证
- **零额外依赖**：完全使用 JetPack 自带库，无需 pip install

### 应用场景

- 工厂/仓库限制区域监控
- 办公楼安全通道管理
- 零售店铺客流分析
- 智慧园区安防巡逻
- 建筑工地安全监控

---

## 🎯 功能概览

| 功能 | 状态 | 说明 |
|---|---|---|
| RTSP PoE 摄像头接入 | ✅ | 支持标准 RTSP 协议的 IP 摄像头 |
| GStreamer NVDEC 硬件解码 | ✅ | GPU 硬件解码，降低 CPU 负载 |
| YOLOv5n 人员检测 | ✅ | TensorRT FP16 加速推理 |
| 异步采集 + 跳帧推理 | ✅ | 高性能流水线架构 |
| 目标跟踪 (质心追踪) | ✅ | 稳定的人员 ID 追踪 |
| 区域入侵检测 | ✅ | 检测人员进入限制区域 |
| 越线检测 | ✅ | 检测人员穿越警戒线 |
| 徘徊检测 | ✅ | 检测人员在某区域逗留过久 |
| 事件日志 + 截图 | ✅ | 自动保存事件记录 |
| Web 远程监控面板 | ✅ | 浏览器实时查看，支持中英文 |
| HDMI 本地显示 | ✅ | 支持本地显示器输出 |

---

## 💻 运行环境要求

### 硬件要求

- **开发板**：NVIDIA Jetson Orin Nano (reComputer Industrial J3011 或同系列)
- **摄像头**：支持 RTSP 协议的 IP 摄像头（通过 PoE 网口连接）
- **显示器**（可选）：HDMI 显示器用于本地调试
- **网络**：千兆以太网口连接摄像头

### 软件要求

- **系统**：JetPack 6.2.1+ (Ubuntu 22.04 + CUDA 12.x)
- **Python**：3.10+
- **OpenCV**：4.8.0+（含 GStreamer 后端）
- **NumPy**：1.21+
- **TensorRT**：10.3+

> **重要**：本项目**零额外依赖**，完全使用 JetPack 自带的 Python / OpenCV / NumPy / TensorRT，无需执行 `pip install`。

---

## 📁 项目结构

```
industrial-security-demo/
├── app/
│   ├── behavior_demo.py       # 主程序入口，行为检测逻辑
│   ├── yolo_trt_detector.py   # YOLOv5 TensorRT 检测器
│   └── web_server.py          # Web 仪表盘后端（MJPEG 流 + REST API）
├── config/
│   └── demo_config.json       # 配置文件（摄像头/检测/规则/显示）
├── models/
│   ├── yolov5n.onnx           # YOLOv5n ONNX 模型（~4MB）
│   └── yolov5n_fp16.engine    # TensorRT FP16 引擎文件（自动生成）
├── web/
│   ├── index.html             # Web 前端单页应用（中英文双语）
│   └── seeedIcon.png          # Seeed 公司图标
├── output/
│   ├── events.jsonl           # 事件日志（自动生成）
│   └── events/                # 事件截图（自动生成）
├── scripts/
│   └── probe_camera.py        # 摄像头 RTSP 地址探测工具
├── run_demo.sh                # 一键启动脚本
└── README.md                  # 项目文档
```

---

## 🚀 快速开始

### 第一步：确认项目文件

```bash
cd /home/seeed/industrial-security-demo
ls -la app/ config/ models/ web/
```

确认 `models/yolov5n.onnx` 存在。如果模型文件缺失，手动下载：

```bash
wget -O models/yolov5n.onnx \
  https://github.com/ultralytics/assets/releases/download/v7.0/yolov5n.onnx
```

### 第二步：配置摄像头网络

假设摄像头 IP 为 `192.168.1.10`，Jetson 办公网口为 `192.168.3.x`。需要给 PoE 网口配置同网段地址：

```bash
# 给 PoE 网口 enP8p1s0 配置 192.168.1.2
sudo nmcli connection add type ethernet ifname enP8p1s0 \
  con-name camera-lan ipv4.method manual \
  ipv4.addresses 192.168.1.2/24 ipv6.method disabled

sudo nmcli connection up camera-lan

# 验证连通性
ping -c 3 192.168.1.10
```

> **提示**：如果 `camera-lan` 连接已存在，改用 `sudo nmcli connection modify camera-lan ...` 即可。

### 第三步：探测摄像头 RTSP 地址

```bash
python3 scripts/probe_camera.py --ip 192.168.1.10 --user admin --password ""
```

脚本会自动尝试常见 RTSP 路径，输出可用的流地址。常见输出：

```
Found stream: rtsp://admin:@192.168.1.10/Streaming/Channels/101
Found stream: rtsp://admin:@192.168.1.10/Streaming/Channels/102
```

默认配置已写好 `rtsp://admin:@192.168.1.10/Streaming/Channels/101`，如果你的摄像头路径不同，编辑 `config/demo_config.json` 中的 `camera.source` 字段。

### 第四步：修改配置（可选）

编辑配置文件：

```bash
nano config/demo_config.json
```

关键配置项说明：

```jsonc
{
  "camera": {
    "name": "poe-camera-1",                    // 摄像头名称（显示在 Web 界面）
    "source": "rtsp://admin:@192.168.1.10/Streaming/Channels/101",
    "use_gstreamer": true                      // true=NVDEC 硬件解码，false=CPU 软解
  },
  "detector": {
    "backend": "yolov5_trt",                   // 检测后端：yolov5_trt 或 hog
    "onnx_file": "yolov5n.onnx",               // ONNX 模型文件
    "conf_threshold": 0.35,                    // 检测置信度阈值 (0.1~0.9)
    "iou_threshold": 0.45,                     // NMS IoU 阈值
    "fp16": true,                              // 使用 FP16 推理（更快）
    "infer_interval": 1                        // 每 N 帧推理一次 (1=实时，3=省资源)
  },
  "tracker": {
    "max_distance": 90,                        // 最大追踪距离（像素）
    "max_age_seconds": 1.5                     // 目标丢失后保留时间
  },
  "rules": {
    "event_cooldown_seconds": 8,               // 事件冷却时间（秒）
    "zones": [                                 // 监控区域配置
      {
        "name": "restricted_area",
        "dwell_seconds": 10,                   // 徘徊检测阈值（秒）
        "color_bgr": [0, 255, 255],            // 区域颜色（黄色）
        "points": [[0.60, 0.22], [0.94, 0.22], [0.94, 0.88], [0.60, 0.88]]
      }
    ],
    "lines": [                                 // 警戒线配置
      {
        "name": "gate_line",
        "color_bgr": [255, 0, 255],            // 线颜色（紫色）
        "start": [0.20, 0.55],                 // 起点（归一化坐标）
        "end": [0.78, 0.55]                    // 终点（归一化坐标）
      }
    ]
  },
  "output": {
    "root": "/home/seeed/industrial-security-demo/output",
    "save_snapshots": true                     // 是否保存事件截图
  },
  "display": {
    "show_window": true,                       // 是否显示本地窗口
    "window_name": "Industrial Security Demo",
    "resize_width": 640                        // 处理分辨率（越小越快）
  },
  "web": {
    "port": 8080                               // Web 面板端口
  }
}
```

### 第五步：运行程序

**方式 A — 一键启动（推荐，支持 HDMI 显示 + Web 面板）：**

```bash
bash run_demo.sh
```

> `run_demo.sh` 会自动检测 HDMI 桌面环境并设置 `DISPLAY`，让画面同时出现在本地显示器上。

**方式 B — 仅 Web 远程监控（无需 HDMI，后台运行）：**

```bash
python3 app/behavior_demo.py --no-window
```

**方式 C — 自定义参数：**

```bash
python3 app/behavior_demo.py \
  --source "rtsp://admin:@192.168.1.10/Streaming/Channels/101" \
  --web-port 8080 \
  --no-window
```

**完整命令行参数：**

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--config PATH` | 配置文件路径 | `config/demo_config.json` |
| `--source URL` | 覆盖摄像头源（RTSP 地址或 `0` 代表 USB 摄像头） | 取配置文件 |
| `--no-window` | 不显示本地窗口（纯后台运行） | 关闭 |
| `--no-web` | 不启动 Web 面板 | 关闭 |
| `--web-port PORT` | Web 面板端口 | `8080` |
| `--max-frames N` | 跑 N 帧后自动退出（测试用） | 无限 |

---

## 🌐 访问 Web 监控面板

程序启动后，在**同一网络内的任意设备**浏览器打开：

```
http://<Jetson 的 IP>:8080
```

例如：`http://192.168.3.145:8080`

### Web 界面功能

#### 实时指标（顶部）
- **帧率 / FPS**：当前视频处理帧率
- **检测人数 / Detections**：当前帧检测到的人数
- **跟踪目标 / Tracking**：正在追踪的人员数量
- **事件数量 / Events**：累计触发的告警事件数
- **运行时间 / Uptime**：程序运行时长
- **推理方式 / Inference**：显示 "TRT" 表示 TensorRT 加速

#### 实时监控画面（左侧）
- 显示摄像头的实时视频流
- 叠加显示检测框、追踪 ID、监控区域、警戒线
- 顶部 HUD 显示 FPS、检测数、追踪数

#### 事件日志（右侧上部）
- 实时显示所有告警事件
- 事件类型：区域进入、徘徊告警、越线告警
- 显示事件时间、追踪 ID、详细信息

#### 检测规则设置（右侧下部）
- **事件冷却时间 (秒)**：同一事件的最小间隔时间
- **徘徊检测时间 (秒)**：触发徘徊告警的逗留时间
- **置信度阈值**：人员检测的置信度阈值
- **应用规则**：保存配置并立即生效
- **清空日志**：清除所有事件记录

#### 语言切换
- 点击右上角 "中文" 或 "EN" 按钮切换界面语言
- 语言偏好会自动保存，下次访问时保持选择

---

## 📊 查看事件记录

### 查看事件日志

```bash
# 查看最新 10 条事件
tail -n 10 output/events.jsonl | python3 -m json.tool --no-ensure-ascii

# 查看所有事件（格式化输出）
cat output/events.jsonl | python3 -m json.tool --no-ensure-ascii

# 统计事件类型
cat output/events.jsonl | python3 -c "
import sys, json
from collections import Counter
events = [json.loads(line) for line in sys.stdin]
types = Counter(e['event_type'] for e in events)
for t, c in types.most_common():
    print(f'{t}: {c}')
"
```

### 查看事件截图

```bash
# 列出所有事件截图
ls -lh output/events/

# 使用图片查看器打开
sxiv output/events/*.jpg  # 需要安装 sxiv

# 或在 Web 界面查看
```

### 事件日志格式

每条事件包含以下字段：

```json
{
  "timestamp": "20260324-054129",     // 时间戳（格式化）
  "epoch_seconds": 1774336895.11,     // 时间戳（Unix 时间）
  "camera_name": "poe-camera-1",      // 摄像头名称
  "event_type": "zone_enter",         // 事件类型
  "track_id": 5,                      // 追踪 ID
  "bbox": [701, 188, 108, 195],       // 检测框 [x, y, w, h]
  "centroid": [755, 285],             // 质心坐标 [x, y]
  "zone_name": "restricted_area"      // 区域名称（如果是区域事件）
}
```

---

## 🔧 高级配置

### 调整检测性能

**提高检测速度（降低延迟）：**

```json
{
  "detector": {
    "infer_interval": 1,        // 每帧都检测（最快）
    "conf_threshold": 0.35      // 降低阈值，检测更灵敏
  },
  "display": {
    "resize_width": 480         // 降低处理分辨率
  }
}
```

**降低 CPU/GPU 占用：**

```json
{
  "detector": {
    "infer_interval": 3,        // 每 3 帧检测一次
    "conf_threshold": 0.5       // 提高阈值，减少误检
  },
  "display": {
    "resize_width": 320         // 更低的分辨率
  }
}
```

### 自定义监控区域

编辑 `config/demo_config.json` 中的 `rules.zones`：

```json
{
  "zones": [
    {
      "name": "entrance",
      "dwell_seconds": 5,
      "color_bgr": [0, 255, 0],
      "points": [
        [0.1, 0.1],   // 左上
        [0.4, 0.1],   // 右上
        [0.4, 0.6],   // 右下
        [0.1, 0.6]    // 左下
      ]
    }
  ]
}
```

> **提示**：坐标使用归一化值（0.0~1.0），会自动适配不同分辨率。

### 自定义警戒线

编辑 `config/demo_config.json` 中的 `rules.lines`：

```json
{
  "lines": [
    {
      "name": "door_line",
      "color_bgr": [0, 0, 255],
      "start": [0.3, 0.5],
      "end": [0.7, 0.5]
    }
  ]
}
```

### 使用 USB 摄像头

修改 `config/demo_config.json`：

```json
{
  "camera": {
    "source": "0",              // USB 摄像头设备号
    "use_gstreamer": false
  }
}
```

或命令行指定：

```bash
python3 app/behavior_demo.py --source 0
```

---

## 🛠️ 故障排查

### 问题 1：摄像头无法连接

**症状**：程序启动后提示 "Cannot open video source"

**解决方法**：

```bash
# 1. 检查网络连通性
ping 192.168.1.10

# 2. 检查 PoE 网口 IP 配置
ip addr show enP8p1s0

# 3. 测试 RTSP 流
python3 scripts/probe_camera.py --ip 192.168.1.10

# 4. 尝试使用 VLC 播放
vlc rtsp://admin:@192.168.1.10/Streaming/Channels/101
```

### 问题 2：FPS 过低

**症状**：FPS < 10，画面卡顿

**解决方法**：

```bash
# 1. 降低处理分辨率
nano config/demo_config.json
# 修改 "resize_width": 320

# 2. 增加推理间隔
"detector": {
  "infer_interval": 3
}

# 3. 检查 GPU 占用
tegrastats
```

### 问题 3：Web 面板无法访问

**症状**：浏览器显示 "无法连接"

**解决方法**：

```bash
# 1. 检查程序是否运行
ps aux | grep behavior_demo

# 2. 检查端口是否监听
netstat -tlnp | grep 8080

# 3. 检查防火墙
sudo ufw status
sudo ufw allow 8080/tcp

# 4. 检查 Jetson IP 地址
hostname -I
```

### 问题 4：检测效果不佳

**症状**：漏检、误检多

**解决方法**：

```bash
# 1. 调整置信度阈值
"detector": {
  "conf_threshold": 0.45  // 提高阈值减少误检
}

# 2. 检查摄像头角度和光线
# 确保监控区域光线充足，无逆光

# 3. 调整监控区域大小
# 确保区域覆盖目标范围
```

---

## 📝 技术架构

### 核心组件

1. **视频采集模块** (`AsyncCapture`)
   - GStreamer NVDEC 硬件解码
   - 异步采集线程，避免阻塞主循环
   - 自动重连机制

2. **目标检测模块** (`YOLOv5TRTDetector`)
   - TensorRT FP16 推理引擎
   - CUDA 内存管理（ pinned memory + device memory）
   - 仅检测 "person" 类（COCO class 0）

3. **目标跟踪模块** (`CentroidTracker`)
   - 质心距离匹配算法
   - 最大距离阈值过滤
   - 目标丢失超时机制

4. **行为分析模块** (`_apply_rules`)
   - 区域入侵检测（点在多边形内算法）
   - 越线检测（线段相交算法）
   - 徘徊检测（逗留时间统计）
   - 事件冷却机制（避免重复告警）

5. **Web 服务模块** (`web_server.py`)
   - MJPEG 流媒体服务
   - REST API（stats, events, config, rules）
   - 静态文件服务
   - 零依赖（stdlib only）

### 性能优化

- **跳帧推理**：`infer_interval` 参数控制检测频率
- **异步流水线**：采集、推理、绘制并行执行
- **硬件加速**：GPU 解码 + GPU 推理
- **内存优化**：使用 CUDA 统一内存，减少拷贝

---

## 📦 模型转换（可选）

如果 `models/yolov5n_fp16.engine` 不存在，需要手动构建：

```bash
# 从 ONNX 构建 TensorRT FP16 引擎
trtexec --onnx=models/yolov5n.onnx \
  --saveEngine=models/yolov5n_fp16.engine \
  --fp16 \
  --workspace=1024
```

> **注意**：首次运行程序时会自动检测并提示构建引擎文件。

---

## 🔒 安全注意事项

1. **摄像头密码**：默认配置中密码为空，请修改为实际密码
2. **网络安全**：确保摄像头网络与办公网络隔离
3. **Web 访问控制**：如需限制访问，可在路由器设置 ACL
4. **数据存储**：定期清理 `output/events/` 目录，避免磁盘占满

---

## 📄 许可证

本项目仅供学习和参考使用。

---

## 🤝 技术支持

- **硬件平台**：reComputer Industrial J3011 (Jetson Orin Nano)
- **系统版本**：JetPack 6.2.1+
- **项目位置**：`/home/seeed/industrial-security-demo`

---

## 📸 界面预览

### Web 监控面板

访问 `http://<jetson-ip>:8080` 即可看到：

- **顶部栏**：公司 Logo、标题、语言切换、连接状态、时钟
- **左侧**：实时监控画面（显示检测框、追踪 ID、监控区域、警戒线）
- **右侧上部**：事件日志（实时滚动显示）
- **右侧下部**：检测规则设置（可在线调整参数）

### 本地显示（HDMI）

连接 HDMI 显示器后，会自动显示：

- 实时视频画面
- 顶部 HUD（FPS、检测数、追踪数、跳帧数）
- 按 `q` 或 `Esc` 退出

---

**祝您使用愉快！** 🎉
