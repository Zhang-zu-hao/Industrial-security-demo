#!/usr/bin/env python3
"""Lightweight web server for the Industrial Security Demo.

Features:
  - MJPEG live stream (no browser plugin needed)
  - REST API for events, stats, and rule configuration
  - Static file serving for the dashboard SPA
  - Zero external dependencies (stdlib only)
"""
import io
import json
import os
import threading
import time
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    """Thread-safe single-frame ring buffer for MJPEG streaming."""

    def __init__(self):
        self._lock = threading.Lock()
        self._frame: Optional[bytes] = None
        self._event = threading.Event()

    def update(self, frame: np.ndarray) -> None:
        # Use maximum JPEG quality (95) for best image quality
        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            return
        with self._lock:
            self._frame = jpeg.tobytes()
        self._event.set()

    def get(self, timeout: float = 2.0) -> Optional[bytes]:
        self._event.wait(timeout=timeout)
        self._event.clear()
        with self._lock:
            return self._frame


class EventStore:
    """In-memory ring buffer of the most recent events."""

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

    def get_events_by_date(self, date_str: str, limit: int = 100) -> List[Dict]:
        """从 events.jsonl 文件读取指定日期的事件"""
        if not date_str:
            # 没有日期筛选，返回最近的 events
            return self.recent(limit)
        
        events = []
        log_path = Path(__file__).resolve().parent.parent / "output" / "events.jsonl"
        if not log_path.is_file():
            return []
        
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
    """Collects per-frame stats for the dashboard."""

    def __init__(self):
        self._lock = threading.Lock()
        self.fps: float = 0.0
        self.detections: int = 0
        self.tracks: int = 0
        self.uptime_start: float = time.time()
        self._last_update: float = 0.0

    def update(self, fps: float, detections: int, tracks: int) -> None:
        with self._lock:
            self.fps = fps
            self.detections = detections
            self.tracks = tracks
            self._last_update = time.time()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            # If no update in last 5 seconds, reset stats to show disconnected
            if time.time() - self._last_update > 5.0 and self._last_update > 0:
                return {
                    "fps": 0.0,
                    "detections": 0,
                    "tracks": 0,
                    "uptime_seconds": round(time.time() - self.uptime_start, 1),
                    "total_events": event_store.total,
                    **_server_clock_meta(),
                }
            return {
                "fps": round(self.fps, 2),
                "detections": self.detections,
                "tracks": self.tracks,
                "uptime_seconds": round(time.time() - self.uptime_start, 1),
                "total_events": event_store.total,
                **_server_clock_meta(),
            }


# Module-level singletons shared between demo loop and HTTP server
frame_buffer = FrameBuffer()
event_store = EventStore()
stats = StatsCollector()
_config_ref: Dict = {}  # mutable reference to the live config
_config_lock = threading.Lock()


def set_config_ref(cfg: Dict) -> None:
    global _config_ref
    with _config_lock:
        _config_ref = cfg


def get_config_snapshot() -> Dict:
    with _config_lock:
        return json.loads(json.dumps(_config_ref))


def update_rules(new_rules: Dict) -> None:
    global _config_ref
    with _config_lock:
        # Update rules section
        if "rules" in _config_ref:
            _config_ref["rules"].update(new_rules)
        else:
            _config_ref["rules"] = new_rules
        
        # Also update detector config if confidence threshold is provided
        if "conf_threshold" in new_rules and "detector" in _config_ref:
            _config_ref["detector"]["conf_threshold"] = new_rules["conf_threshold"]


STATIC_DIR = Path(__file__).resolve().parent.parent / "web"


class DemoHandler(SimpleHTTPRequestHandler):
    """Handle MJPEG stream, REST API, and static files."""

    def log_message(self, fmt, *args):
        pass  # suppress per-request logs

    # ---- MJPEG stream ----
    def _serve_mjpeg(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        try:
            while True:
                jpeg = frame_buffer.get(timeout=2.0)
                if jpeg is None:
                    continue
                header = (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                )
                self.wfile.write(header + jpeg + b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    # ---- JSON helpers ----
    def _json_response(self, data: Any, code: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length)

    # ---- Routing ----
    def do_GET(self):
        if self.path == "/api/stream":
            self._serve_mjpeg()
        elif self.path.startswith("/api/events"):
            # 支持按日期筛选：/api/events?date=20260416
            from urllib.parse import parse_qs, urlparse
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            date_filter = params.get('date', [None])[0]
            self._json_response(self._get_events_by_date(date_filter, 100))
        elif self.path == "/api/stats":
            self._json_response(stats.snapshot())
        elif self.path == "/api/config":
            self._json_response(get_config_snapshot())
        else:
            self._serve_static()

    def _get_events_by_date(self, date_filter: Optional[str], limit: int = 100) -> List[Dict]:
        """从 event_store 读取指定日期的事件"""
        return event_store.get_events_by_date(date_filter, limit)

    def do_POST(self):
        if self.path == "/api/rules":
            try:
                body = json.loads(self._read_body())
                update_rules(body)
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


def start_server(port: int = 8080) -> threading.Thread:
    """Start the web server in a daemon thread and return the thread."""
    server = HTTPServer(("0.0.0.0", port), DemoHandler)
    server.timeout = 1
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"Web dashboard: http://0.0.0.0:{port}")
    return thread
