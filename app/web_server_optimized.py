#!/usr/bin/env python3
"""Optimized web server with stable WebSocket and low-latency streaming.

Key features:
  - Single event loop for all async operations
  - Robust WebSocket with auto-reconnect friendly behavior
  - Frame buffering with latest-frame-only strategy (lowest latency)
  - Configurable JPEG quality
  - Feature toggles
  - Auto port fallback

Architecture reference: VideoPlayTool uses native SDK → H.264 → WebSocket.
We use: RTSP → decode → JPEG → WebSocket (simplified but optimized).
"""
import asyncio
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from http.server import HTTPServer, SimpleHTTPRequestHandler, ThreadingHTTPServer
import websockets
import cv2
import numpy as np


def _server_clock_meta() -> Dict[str, Any]:
    """Wall-clock fields for dashboard (device time, not browser time)."""
    now = datetime.now().astimezone()
    tz = getattr(now.tzinfo, "key", None) or "UTC"
    return {
        "server_time_unix": time.time(),
        "server_tz_iana": tz,
    }


class FrameBuffer:
    """Thread-safe frame buffer optimized for low-latency streaming."""

    def __init__(self, max_quality: int = 75):
        self._lock = threading.RLock()
        self._frame: Optional[bytes] = None
        self._raw_frame: Optional[np.ndarray] = None
        self._quality = max_quality
        self._timestamp = 0.0
        self._encoding = False
        self._encode_thread: Optional[threading.Thread] = None
        self._running = True
        self._encode_event = threading.Event()
        self._last_update_time = 0.0
        self._start_encoder()

    def _start_encoder(self) -> None:
        def encoder_loop():
            while self._running:
                if self._encode_event.wait(timeout=0.05):
                    self._encode_event.clear()
                    with self._lock:
                        if self._raw_frame is not None:
                            try:
                                ok, jpeg = cv2.imencode(".jpg", self._raw_frame, [
                                    cv2.IMWRITE_JPEG_QUALITY,
                                    min(95, max(40, self._quality))
                                ])
                                if ok:
                                    self._frame = jpeg.tobytes()
                                    self._timestamp = time.time()
                            except Exception:
                                pass
        
        self._encode_thread = threading.Thread(target=encoder_loop, daemon=True)
        self._encode_thread.start()

    def update(self, frame: np.ndarray, quality: int = None) -> bool:
        if quality is not None:
            self._quality = max(1, min(100, quality))
        try:
            current_time = time.time()
            if current_time - self._last_update_time < 0.008:
                return False
            
            with self._lock:
                self._raw_frame = frame
                self._last_update_time = current_time
            self._encode_event.set()
            return True
        except Exception as e:
            return False

    def get(self) -> Optional[bytes]:
        with self._lock:
            return self._frame

    @property
    def timestamp(self) -> float:
        with self._lock:
            return self._timestamp

    def set_quality(self, quality: int) -> None:
        self._quality = max(1, min(100, quality))

    def stop(self) -> None:
        self._running = False
        self._encode_event.set()


class EventStore:
    """Thread-safe ring buffer for events."""

    def __init__(self, max_events: int = 500):
        self._lock = threading.Lock()
        self._events: List[Dict] = []
        self._max = max_events
        self._total = 0

    def add(self, event: Dict) -> None:
        with self._lock:
            self._events.append(event)
            self._total += 1
            if len(self._events) > self._max:
                self._events = self._events[-self._max:]

    def recent(self, n: int = 50) -> List[Dict]:
        with self._lock:
            return list(self._events[-n:])

    @property
    def total(self) -> int:
        with self._lock:
            return self._total

    def clear(self) -> None:
        with self._lock:
            self._events.clear()

    def get_events_by_date(self, date_str: str, limit: int = 100) -> List[Dict]:
        """从 events.jsonl 文件读取指定日期的事件"""
        if not date_str:
            # 没有日期筛选，返回最近的 events
            return self.recent(limit)
        
        # 先从内存缓存中查找
        with self._lock:
            cached_events = [e for e in self._events if e.get("timestamp", "").startswith(date_str)]
            if cached_events:
                return cached_events[:limit]
        
        # 如果内存中没有，再从文件读取
        log_path = Path(__file__).resolve().parent.parent / "output" / "events.jsonl"
        if not log_path.is_file():
            return []
        
        events = []
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        event = json.loads(line.strip())
                        # timestamp 格式：20260416-064358
                        if event.get("timestamp", "").startswith(date_str):
                            events.append(event)
                            if len(events) >= limit:
                                break
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass
        
        return events


class StatsCollector:
    """Real-time stats collector."""

    def __init__(self):
        self._lock = threading.Lock()
        self.fps: float = 0.0
        self.detections: int = 0
        self.tracks: int = 0
        self.uptime_start: float = time.time()
        self._last_update: float = 0.0
        self.infer_interval: int = 3
        self.feature_flags: Dict[str, bool] = {
            "human_detect": False,
            "tracking": False,
            "zone_detection": False,
            "line_crossing": False,
            "loitering": False,
        }

    def update(self, fps: float, detections: int, tracks: int, infer_interval: int = 3) -> None:
        with self._lock:
            self.fps = fps
            self.detections = detections
            self.tracks = tracks
            self.infer_interval = infer_interval
            self._last_update = time.time()

    def update_events(self, total_events: int) -> None:
        with self._lock:
            pass  # Events count is fetched directly from event_store.total

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "fps": round(self.fps, 1),
                "detections": self.detections,
                "tracks": self.tracks,
                "uptime_seconds": round(time.time() - self.uptime_start, 1),
                "total_events": event_store.total,
                "features": dict(self.feature_flags),
                "infer_interval": self.infer_interval,
                "current_model": _config_ref.get("detector", {}).get("onnx_file", ""),
                **_server_clock_meta(),
            }

    def update_features(self, features: Dict[str, bool]) -> None:
        with self._lock:
            self.feature_flags.update(features)


# Global singletons
frame_buffer = FrameBuffer(max_quality=75)
event_store = EventStore()
stats = StatsCollector()
_config_ref: Dict = {}
_config_lock = threading.Lock()


# WebSocket clients management with health tracking
class WebSocketManager:
    def __init__(self):
        self.clients = set()
        self.lock = threading.Lock()
        self.total_connections = 0
    
    def add_client(self, websocket):
        with self.lock:
            self.clients.add(websocket)
            self.total_connections += 1
            print(f"[WS] ✅ Client connected ({len(self.clients)} active)")
    
    def remove_client(self, websocket):
        with self.lock:
            if websocket in self.clients:
                self.clients.remove(websocket)
                print(f"[WS] ❌ Client disconnected ({len(self.clients)} active)")
    
    def get_clients(self):
        with self.lock:
            return list(self.clients)

ws_manager = WebSocketManager()


def set_config_ref(cfg: Dict) -> None:
    global _config_ref
    with _config_lock:
        _config_ref = cfg


def get_config_snapshot() -> Dict:
    with _config_lock:
        return json.loads(json.dumps(_config_ref))


def update_config(new_config: Dict) -> None:
    global _config_ref
    with _config_lock:
        if "rules" in new_config:
            _config_ref["rules"].update(new_config["rules"])
        if "detector" in new_config:
            _config_ref["detector"].update(new_config["detector"])
        if "features" in new_config:
            _config_ref["features"] = new_config["features"]
            stats.update_features(new_config["features"])


_demo_ref = None


def set_demo_ref(demo) -> None:
    global _demo_ref
    _demo_ref = demo


def list_models() -> list:
    models_dir = Path(__file__).resolve().parent.parent / "models"
    models = []
    for f in sorted(models_dir.glob("*.onnx")):
        engine = models_dir / (f.stem + "_fp16.engine")
        models.append({
            "name": f.stem,
            "onnx_file": f.name,
            "has_engine": engine.exists(),
            "size_mb": round(f.stat().st_size / 1024 / 1024, 1)
        })
    return models


# WebSocket server state
ws_server = None
ws_loop = None


async def websocket_handler(websocket):
    """Handle WebSocket video streaming with robust error handling."""
    ws_manager.add_client(websocket)
    last_ts = 0.0
    
    try:
        while True:
            try:
                ts = frame_buffer.timestamp
                if ts > last_ts:
                    frame = frame_buffer.get()
                    if frame is not None:
                        await websocket.send(frame)
                        last_ts = ts
                    await asyncio.sleep(0.005)
                else:
                    await asyncio.sleep(0.003)
            except websockets.exceptions.ConnectionClosed:
                break
            except Exception:
                break
    except Exception as e:
        if "Connection closed" not in str(e):
            print(f"[WS] Video handler error: {e}")
    finally:
        ws_manager.remove_client(websocket)


async def config_handler(websocket):
    """Handle WebSocket configuration channel with robust error handling."""
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                
                if isinstance(data, dict):
                    if "features" in data or "rules" in data or "detector" in data:
                        update_config(data)
                        await websocket.send(json.dumps({"status": "ok"}))
                        
                    elif "quality" in data:
                        quality = data.get("quality", 75)
                        frame_buffer.set_quality(int(quality))
                        await websocket.send(json.dumps({"status": "ok"}))
                    
            except json.JSONDecodeError:
                pass
            except Exception:
                pass
                
    except websockets.exceptions.ConnectionClosed:
        pass
    except Exception as e:
        if "Connection closed" not in str(e):
            print(f"[WS] Config handler error: {e}")


def start_websocket_servers(port: int = 8081, config_port: int = 8082):
    """Start WebSocket servers with auto port fallback."""
    global ws_loop
    
    def _try_port(host, try_port):
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, try_port))
            s.close()
            return try_port
        except OSError:
            s.close()
            return None
    
    # Find available ports
    ws_port = _try_port("0.0.0.0", port) or _try_port("0.0.0.0", port + 10)
    if not ws_port:
        for p in range(port + 11, port + 50):
            ws_port = _try_port("0.0.0.0", p)
            if ws_port:
                break
    
    cfg_port = _try_port("0.0.0.0", config_port) or _try_port("0.0.0.0", (ws_port or port) + 1)
    if not cfg_port:
        for p in range((ws_port or port) + 2, (ws_port or port) + 22):
            cfg_port = _try_port("0.0.0.0", p)
            if cfg_port:
                break
    
    if not ws_port or not cfg_port:
        print("[WS] ⚠️ Cannot find available ports")
        return
    
    def run_servers():
        global ws_server, ws_loop
        
        ws_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(ws_loop)
        
        async def serve():
            # Video streaming server with optimized settings for stability
            video_server = await websockets.serve(
                websocket_handler, "0.0.0.0", ws_port,
                ping_interval=30,
                ping_timeout=30,
                close_timeout=10,
                max_size=2*1024*1024,
                max_queue=1,
            )
            print(f"[WS] ✅ Video: ws://0.0.0.0:{ws_port}")

            config_server = await websockets.serve(
                config_handler, "0.0.0.0", cfg_port,
                ping_interval=30,
                ping_timeout=30,
                close_timeout=10,
            )
            print(f"[WS] ✅ Config: ws://0.0.0.0:{cfg_port}")

            video_closed = asyncio.ensure_future(video_server.wait_closed())
            config_closed = asyncio.ensure_future(config_server.wait_closed())
            done, pending = await asyncio.wait(
                [video_closed, config_closed],
                return_when=asyncio.FIRST_COMPLETED
            )
        
        try:
            ws_loop.run_until_complete(serve())
        except Exception as e:
            print(f"[WS] Error: {e}")
    
    thread = threading.Thread(target=run_servers, daemon=True)
    thread.start()
    time.sleep(0.8)


STATIC_DIR = Path(__file__).resolve().parent.parent / "web"


class DemoHandler(SimpleHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass

    def _serve_mjpeg(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        
        last_ts = 0
        try:
            while True:
                jpeg = frame_buffer.get()
                if jpeg is None:
                    time.sleep(0.033)
                    continue
                
                ts = frame_buffer.timestamp
                if ts == last_ts:
                    time.sleep(0.02)
                    continue
                last_ts = ts
                
                header = (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                )
                self.wfile.write(header + jpeg + b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json_response(self, data: Any, code: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length)

    def _get_events_by_date(self, date_filter: Optional[str], limit: int = 100) -> List[Dict]:
        """从 event_store 读取指定日期的事件"""
        return event_store.get_events_by_date(date_filter, limit)

    def do_GET(self):
        from urllib.parse import parse_qs, urlparse
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == "/api/stream":
            self._serve_mjpeg()
        elif path == "/api/events/images":
            # 支持按日期筛选：/api/events/images?date=20260416
            date_filter = params.get('date', [None])[0]
            self._serve_event_image_list(date_filter)
        elif path.startswith("/api/events/img/"):
            self._serve_event_image(path[len("/api/events/img/"):])
        elif path.startswith("/api/events"):
            # 支持按日期筛选：/api/events?date=20260416
            date_filter = params.get('date', [None])[0]
            self._json_response(self._get_events_by_date(date_filter, 100))
        elif path == "/api/stats":
            self._json_response(stats.snapshot())
        elif path == "/api/config":
            self._json_response(get_config_snapshot())
        elif path == "/api/models":
            self._json_response(list_models())
        else:
            self._serve_static()

    def _serve_event_image_list(self, date_filter: Optional[str] = None) -> None:
        """列出事件截图，支持按日期筛选"""
        events_dir = Path(__file__).resolve().parent.parent / "output" / "events"
        if not events_dir.is_dir():
            self._json_response([])
            return
        
        files = sorted(events_dir.glob("*.jpg"), key=lambda f: f.stat().st_mtime, reverse=True)
        
        # 如果指定了日期，过滤文件名
        if date_filter:
            # 文件名格式：20260416-064358_zone_enter_track2.jpg
            files = [f for f in files if f.name.startswith(date_filter)]
        
        # 限制返回数量
        files = files[:30]
        result = [{"name": f.name, "url": f"/api/events/img/{f.name}"} for f in files]
        self._json_response(result)

    def _serve_event_image(self, filename: str) -> None:
        import urllib.parse
        filename = urllib.parse.unquote(filename).split("?")[0].split("/")[0]
        fpath = Path(__file__).resolve().parent.parent / "output" / "events" / filename
        if not fpath.is_file() or not filename.endswith(".jpg"):
            self.send_error(404)
            return
        body = fpath.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "public, max-age=3600")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path == "/api/config":
            try:
                body = json.loads(self._read_body())
                update_config(body)
                self._json_response({"status": "ok"})
            except Exception as exc:
                self._json_response({"error": str(exc)}, 400)
        elif self.path == "/api/model/switch":
            try:
                body = json.loads(self._read_body())
                onnx_file = body.get("onnx_file")
                if onnx_file and _demo_ref:
                    result = _demo_ref.switch_model(onnx_file)
                    code = 200 if result.get("status") == "ok" else 500
                    self._json_response(result, code)
                else:
                    self._json_response({"error": "invalid request"}, 400)
            except Exception as exc:
                self._json_response({"error": str(exc)}, 500)
        elif self.path == "/api/events/clear":
            event_store.clear()
            self._json_response({"status": "ok"})
        elif self.path == "/api/rules":
            try:
                body = json.loads(self._read_body())
                update_config({"rules": body})
                self._json_response({"status": "ok"})
            except Exception as exc:
                self._json_response({"error": str(exc)}, 400)
        else:
            self._json_response({"error": "not found"}, 404)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _serve_static(self) -> None:
        rel = self.path.lstrip("/")
        if rel == "" or rel == "/":
            rel = "index.html"
        fpath = STATIC_DIR / rel
        if not fpath.is_file():
            fpath = STATIC_DIR / "index.html"
        if not fpath.is_file():
            self.send_error(404)
            return

        ext = fpath.suffix.lower()
        mime = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".json": "application/json; charset=utf-8",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".svg": "image/svg+xml",
            ".ico": "image/x-icon",
        }.get(ext, "application/octet-stream")

        body = fpath.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def start_http_server(port: int = 8080) -> threading.Thread:
    ThreadingHTTPServer.allow_reuse_address = True
    ThreadingHTTPServer.daemon_threads = True
    for attempt_port in [port] + list(range(port + 10, port + 30)):
        try:
            server = ThreadingHTTPServer(("0.0.0.0", attempt_port), DemoHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            print(f"[HTTP] Server: http://0.0.0.0:{attempt_port}")
            return thread
        except OSError as e:
            if "Address already in use" in str(e) or e.errno == 98:
                continue
            raise
    raise OSError(f"Cannot bind any port in range {port}-{port+30}")


def start_server(port: int = 8080, ws_port: int = 8081, config_ws_port: int = 8082):
    start_http_server(port)
    start_websocket_servers(ws_port, config_ws_port)
    print("[Server] All services started")
