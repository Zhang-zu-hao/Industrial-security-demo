#!/usr/bin/env python3
"""Optimized web server with multi-camera support.

Key features:
  - Per-camera MJPEG stream and WebSocket video
  - Per-camera stats, events, and snapshots
  - Camera discovery and hot-add API
  - Auto port fallback
"""
import asyncio
import json
import re
import socket
import struct
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from http.server import HTTPServer, SimpleHTTPRequestHandler, ThreadingHTTPServer
import websockets


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------
_CAM_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")
_FILENAME_RE = re.compile(r"^[a-zA-Z0-9_.-]+$")
_DATE_RE = re.compile(r"^\d{8}$")


def _validate_cam_id(cam_id: str) -> bool:
    return bool(cam_id and _CAM_ID_RE.match(cam_id))


def _validate_filename(filename: str) -> bool:
    return bool(filename and _FILENAME_RE.match(filename) and ".." not in filename)


def _validate_date(date_str: str) -> bool:
    return bool(date_str and _DATE_RE.match(date_str))


def _server_clock_meta() -> Dict[str, Any]:
    now = datetime.now().astimezone()
    tz = getattr(now.tzinfo, "key", None) or "UTC"
    return {
        "server_time_unix": time.time(),
        "server_tz_iana": tz,
    }


_config_ref: Dict = {}
_config_lock = threading.Lock()
_manager_ref = None
_manager_lock = threading.Lock()
_discovery_ref = None
_discovery_lock = threading.Lock()


def set_config_ref(cfg: Dict) -> None:
    global _config_ref
    with _config_lock:
        _config_ref = cfg


def get_config_snapshot() -> Dict:
    with _config_lock:
        return json.loads(json.dumps(_config_ref))


def set_manager_ref(manager) -> None:
    global _manager_ref
    with _manager_lock:
        _manager_ref = manager


def get_manager():
    with _manager_lock:
        return _manager_ref


def set_discovery_ref(discovery) -> None:
    global _discovery_ref
    with _discovery_lock:
        _discovery_ref = discovery


def get_discovery():
    with _discovery_lock:
        return _discovery_ref


def update_config(new_config: Dict) -> None:
    global _config_ref
    with _config_lock:
        if "rules" in new_config:
            rules = new_config["rules"]
            if "camera_zones" in rules:
                _config_ref["rules"].setdefault("camera_zones", {})
                _config_ref["rules"]["camera_zones"].update(rules["camera_zones"])
                del rules["camera_zones"]
            _config_ref["rules"].update(rules)
        if "detector" in new_config:
            _config_ref["detector"].update(new_config["detector"])
        if "features" in new_config:
            _config_ref["features"] = new_config["features"]


def list_models() -> list:
    models_dir = Path(__file__).resolve().parent.parent / "models"
    models = []
    for f in sorted(models_dir.glob("*.onnx")):
        engine = models_dir / (f.stem + "_fp16.engine")
        models.append({
            "name": f.stem,
            "onnx_file": f.name,
            "has_engine": engine.exists(),
            "size_mb": round(f.stat().st_size / 1024 / 1024, 1),
        })
    return models


class WebSocketManager:
    def __init__(self):
        self.clients = set()
        self.lock = threading.Lock()
        self.total_connections = 0

    def add_client(self, websocket):
        with self.lock:
            self.clients.add(websocket)
            self.total_connections += 1

    def remove_client(self, websocket):
        with self.lock:
            if websocket in self.clients:
                self.clients.remove(websocket)

    def get_clients(self):
        with self.lock:
            return list(self.clients)


ws_manager = WebSocketManager()
ws_server = None
ws_loop = None


async def websocket_handler(websocket):
    ws_manager.add_client(websocket)
    last_ts_map = {}

    try:
        await websocket.send(json.dumps({"proto": 2, "type": "handshake"}))
    except Exception:
        ws_manager.remove_client(websocket)
        return

    try:
        while True:
            try:
                manager = get_manager()
                if manager is None:
                    await asyncio.sleep(0.05)
                    continue

                pipelines = manager.get_all_pipelines()
                if not pipelines:
                    await asyncio.sleep(0.3)
                    continue

                for cam_id, pipeline in pipelines.items():
                    fb = pipeline.frame_buffer
                    ts = fb.timestamp
                    last_ts = last_ts_map.get(cam_id, 0.0)
                    if ts > last_ts:
                        frame = fb.get()
                        if frame is not None:
                            cam_id_bytes = cam_id.encode('utf-8')
                            header = (
                                len(cam_id_bytes).to_bytes(4, 'big') +
                                cam_id_bytes +
                                struct.pack('!d', ts)
                            )
                            await websocket.send(header + frame)
                            last_ts_map[cam_id] = ts
                await asyncio.sleep(0.033)
            except websockets.exceptions.ConnectionClosed:
                break
            except websockets.exceptions.ConnectionClosedError:
                break
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(0.1)
    except Exception:
        pass
    finally:
        ws_manager.remove_client(websocket)


async def config_handler(websocket):
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
                        manager = get_manager()
                        if manager:
                            for p in manager.get_all_pipelines().values():
                                p.frame_buffer.set_quality(int(quality))
                        await websocket.send(json.dumps({"status": "ok"}))
            except json.JSONDecodeError:
                pass
            except Exception:
                pass
    except websockets.exceptions.ConnectionClosed:
        pass
    except Exception:
        pass


def start_websocket_servers(port: int = 8081, config_port: int = 8082):
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
        print("[WS] Cannot find available ports")
        return

    def run_servers():
        global ws_server, ws_loop

        ws_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(ws_loop)

        async def serve():
            video_server = await websockets.serve(
                websocket_handler,
                "0.0.0.0", ws_port,
                ping_interval=30,
                ping_timeout=30,
                close_timeout=10,
                max_size=2 * 1024 * 1024,
                max_queue=1,
            )
            print(f"[WS] Video: ws://0.0.0.0:{ws_port}")

            config_server = await websockets.serve(
                config_handler, "0.0.0.0", cfg_port,
                ping_interval=30,
                ping_timeout=30,
                close_timeout=10,
            )
            print(f"[WS] Config: ws://0.0.0.0:{cfg_port}")

            video_closed = asyncio.ensure_future(video_server.wait_closed())
            config_closed = asyncio.ensure_future(config_server.wait_closed())
            done, pending = await asyncio.wait(
                [video_closed, config_closed],
                return_when=asyncio.FIRST_COMPLETED,
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

    def _get_pipeline(self, cam_id: str):
        manager = get_manager()
        if manager is None:
            return None
        return manager.get_pipeline(cam_id)

    def _serve_mjpeg(self, cam_id: str = None) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        try:
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception:
            pass

        last_ts = 0
        try:
            while True:
                manager = get_manager()
                if manager is None:
                    time.sleep(0.05)
                    continue

                if cam_id:
                    pipeline = manager.get_pipeline(cam_id)
                    if pipeline is None:
                        time.sleep(0.3)
                        continue
                    fb = pipeline.frame_buffer
                else:
                    pipelines = manager.get_all_pipelines()
                    if not pipelines:
                        time.sleep(0.3)
                        continue
                    first_id = next(iter(pipelines))
                    fb = pipelines[first_id].frame_buffer

                jpeg = fb.get()
                if jpeg is None:
                    time.sleep(0.005)
                    continue

                ts = fb.timestamp
                if ts == last_ts:
                    time.sleep(0.002)
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

    def _get_events_by_date(self, date_filter: Optional[str], limit: int = 100, cam_id: str = None) -> List[Dict]:
        manager = get_manager()
        if manager is None:
            return []

        if cam_id:
            pipeline = manager.get_pipeline(cam_id)
            if pipeline is None:
                return []
            return pipeline.event_store.get_events_by_date(date_filter, limit)

        all_events = []
        for pipeline in manager.get_all_pipelines().values():
            all_events.extend(pipeline.event_store.get_events_by_date(date_filter, limit))
        all_events.sort(key=lambda e: e.get("epoch_seconds", 0), reverse=True)
        return all_events[:limit]

    def _get_event_image_list(self, date_filter: Optional[str] = None, cam_id: str = None) -> List[Dict]:
        output_root = Path(__file__).resolve().parent.parent / "output"
        result = []

        try:
            if cam_id:
                dirs = [output_root / cam_id / "events"]
            else:
                dirs = sorted(output_root.glob("cam-*/events"))

            for events_dir in dirs:
                if not events_dir.is_dir():
                    continue
                cam = events_dir.parent.name
                try:
                    files = sorted(events_dir.glob("*.jpg"), key=lambda f: f.stat().st_mtime, reverse=True)
                except Exception:
                    continue
                if date_filter:
                    files = [f for f in files if f.name.startswith(date_filter)]
                for f in files[:30]:
                    result.append({
                        "name": f.name,
                        "url": f"/api/events/img/{cam}/{f.name}",
                        "camera_id": cam,
                    })
        except Exception:
            pass

        result.sort(key=lambda x: x["name"], reverse=True)
        return result[:30]

    def _serve_event_image(self, cam_id: str, filename: str) -> None:
        import urllib.parse
        if not _validate_cam_id(cam_id):
            self.send_error(400, "Invalid camera ID")
            return
        filename = urllib.parse.unquote(filename).split("?")[0].split("/")[0]
        if not filename.endswith(".jpg") or not _validate_filename(filename):
            self.send_error(400, "Invalid filename")
            return
        output_root = Path(__file__).resolve().parent.parent / "output"
        try:
            fpath = (output_root / cam_id / "events" / filename).resolve()
            if not fpath.is_relative_to(output_root.resolve()):
                self.send_error(403, "Access denied")
                return
        except Exception:
            self.send_error(400, "Invalid path")
            return
        if not fpath.is_file():
            self.send_error(404)
            return
        try:
            body = fpath.read_bytes()
        except Exception:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "public, max-age=3600")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        from urllib.parse import parse_qs, urlparse
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == "/api/cameras":
            manager = get_manager()
            if manager:
                self._json_response(manager.list_cameras())
            else:
                self._json_response([])

        elif path.startswith("/api/cameras/") and "/stream" in path:
            cam_id = path.split("/")[3]
            if not _validate_cam_id(cam_id):
                self._json_response({"error": "Invalid camera ID"}, 400)
                return
            self._serve_mjpeg(cam_id)

        elif path.startswith("/api/cameras/") and "/stats" in path:
            cam_id = path.split("/")[3]
            if not _validate_cam_id(cam_id):
                self._json_response({"error": "Invalid camera ID"}, 400)
                return
            pipeline = self._get_pipeline(cam_id)
            if pipeline:
                s = pipeline.stats.snapshot(
                    total_events=pipeline.event_store.total,
                    current_model=get_manager().current_model if get_manager() else "",
                )
                s.update(_server_clock_meta())
                self._json_response(s)
            else:
                self._json_response({"error": "camera not found"}, 404)

        elif path.startswith("/api/cameras/") and "/events" in path:
            cam_id = path.split("/")[3]
            if not _validate_cam_id(cam_id):
                self._json_response({"error": "Invalid camera ID"}, 400)
                return
            date_filter = params.get('date', [None])[0]
            if date_filter and not _validate_date(date_filter):
                self._json_response({"error": "Invalid date format"}, 400)
                return
            self._json_response(self._get_events_by_date(date_filter, 100, cam_id))

        elif path == "/api/stream":
            self._serve_mjpeg()

        elif path == "/api/events/images":
            date_filter = params.get('date', [None])[0]
            cam_id = params.get('camera_id', [None])[0]
            if date_filter and not _validate_date(date_filter):
                self._json_response({"error": "Invalid date format"}, 400)
                return
            if cam_id and not _validate_cam_id(cam_id):
                self._json_response({"error": "Invalid camera ID"}, 400)
                return
            self._json_response(self._get_event_image_list(date_filter, cam_id))

        elif path.startswith("/api/events/img/"):
            parts = path[len("/api/events/img/"):].split("/")
            if len(parts) >= 2:
                self._serve_event_image(parts[0], parts[1])
            else:
                self._serve_event_image("cam-0", parts[0])

        elif path == "/api/events":
            date_filter = params.get('date', [None])[0]
            cam_id = params.get('camera_id', [None])[0]
            if date_filter and not _validate_date(date_filter):
                self._json_response({"error": "Invalid date format"}, 400)
                return
            if cam_id and not _validate_cam_id(cam_id):
                self._json_response({"error": "Invalid camera ID"}, 400)
                return
            self._json_response(self._get_events_by_date(date_filter, 100, cam_id))

        elif path == "/api/stats":
            manager = get_manager()
            if manager:
                pipelines = manager.get_all_pipelines()
                if pipelines:
                    total_det = 0
                    total_trk = 0
                    total_evt = 0
                    fps_max = 0.0
                    uptime_min = float("inf")
                    for p in pipelines.values():
                        s = p.stats.snapshot(
                            total_events=p.event_store.total,
                            current_model=manager.current_model,
                        )
                        total_det += s.get("detections", 0)
                        total_trk += s.get("tracks", 0)
                        total_evt += s.get("total_events", 0)
                        fps_max = max(fps_max, s.get("fps", 0))
                        uptime_min = min(uptime_min, s.get("uptime_seconds", 0))
                    result = {
                        "fps": round(fps_max, 1),
                        "detections": total_det,
                        "tracks": total_trk,
                        "uptime_seconds": round(uptime_min, 1),
                        "total_events": total_evt,
                        "camera_count": len(pipelines),
                        "current_model": manager.current_model,
                    }
                    result.update(_server_clock_meta())
                    self._json_response(result)
                else:
                    self._json_response({"error": "no cameras"}, 404)
            else:
                self._json_response({"error": "not started"}, 404)

        elif path == "/api/config":
            self._json_response(get_config_snapshot())

        elif path == "/api/models":
            self._json_response(list_models())

        else:
            self._serve_static()

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
                manager = get_manager()
                if onnx_file and manager:
                    result = manager.switch_model(onnx_file)
                    code = 200 if result.get("status") == "ok" else 500
                    self._json_response(result, code)
                else:
                    self._json_response({"error": "invalid request"}, 400)
            except Exception as exc:
                self._json_response({"error": str(exc)}, 500)

        elif self.path == "/api/events/clear":
            cam_id = None
            try:
                body = json.loads(self._read_body())
                cam_id = body.get("camera_id")
            except Exception:
                pass
            manager = get_manager()
            if manager:
                if cam_id:
                    pipeline = manager.get_pipeline(cam_id)
                    if pipeline:
                        pipeline.event_store.clear()
                else:
                    for p in manager.get_all_pipelines().values():
                        p.event_store.clear()
            self._json_response({"status": "ok"})

        elif self.path == "/api/rules":
            try:
                body = json.loads(self._read_body())
                cam_id = body.pop("camera_id", None)
                if cam_id:
                    zones = body.get("zones", [])
                    lines = body.get("lines", [])
                    existing = get_config_snapshot().get("rules", {}).get("camera_zones", {})
                    existing[cam_id] = {"zones": zones, "lines": lines}
                    update_config({"rules": {"camera_zones": existing}})
                else:
                    update_config({"rules": body})
                self._json_response({"status": "ok"})
            except Exception as exc:
                self._json_response({"error": str(exc)}, 400)

        elif self.path == "/api/cameras/add":
            try:
                body = json.loads(self._read_body())
                discovery = get_discovery()
                manager = get_manager()
                if discovery and manager:
                    result = discovery.add_camera(body)
                    if result:
                        manager.add_camera(result)
                        self._json_response({"status": "ok", "camera": result})
                    else:
                        self._json_response({"status": "error", "error": "camera not reachable"}, 400)
                else:
                    self._json_response({"status": "error", "error": "discovery not available"}, 400)
            except Exception as exc:
                self._json_response({"error": str(exc)}, 500)

        elif self.path == "/api/cameras/remove":
            try:
                body = json.loads(self._read_body())
                cam_id = body.get("camera_id")
                manager = get_manager()
                discovery = get_discovery()
                if cam_id and manager:
                    ok = manager.remove_camera(cam_id)
                    if discovery:
                        discovery.remove_camera(cam_id)
                    self._json_response({"status": "ok" if ok else "not found"})
                else:
                    self._json_response({"error": "invalid request"}, 400)
            except Exception as exc:
                self._json_response({"error": str(exc)}, 500)

        elif self.path == "/api/cameras/discover":
            try:
                discovery = get_discovery()
                if discovery:
                    cameras = discovery.get_cameras()
                    self._json_response({"status": "ok", "cameras": cameras})
                else:
                    from camera_discovery import discover_cameras
                    cameras = discover_cameras(get_config_snapshot())
                    self._json_response({"status": "ok", "cameras": cameras})
            except Exception as exc:
                self._json_response({"error": str(exc)}, 500)

        elif self.path == "/api/cameras/probe":
            try:
                body = json.loads(self._read_body())
                from camera_discovery import probe_single_camera
                result = probe_single_camera(
                    ip=body.get("ip", ""),
                    username=body.get("username", "admin"),
                    password=body.get("password", ""),
                    rtsp_paths=body.get("rtsp_paths"),
                )
                if result:
                    self._json_response({"status": "ok", "reachable": True, "source": result["source"]})
                else:
                    self._json_response({"status": "ok", "reachable": False})
            except Exception as exc:
                self._json_response({"error": str(exc)}, 500)

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