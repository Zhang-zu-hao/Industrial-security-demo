#!/usr/bin/env python3
"""Industrial Security Behavior Detection Demo.

Performance architecture:
  - GStreamer NVDEC hardware RTSP decode (offloads CPU)
  - Async capture thread (always has latest frame ready)
  - Configurable inference skip (detect every Nth frame, reuse results between)

Optionally starts a Web dashboard on port 8080 for remote browser access.
"""
import argparse
import json
import math
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu:/usr/share/fonts/truetype")
os.environ.setdefault("OPENCV_LOG_LEVEL", "WARNING")
os.environ.setdefault("OPENCV_VIDEOIO_DEBUG", "0")
os.environ.setdefault("QT_LOGGING_RULES", "*.debug=false;qt.qpa.fonts=false")

import cv2
import numpy as np

Point = Tuple[int, int]
BBox = Tuple[int, int, int, int]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def now_ts() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def center_of(box: BBox) -> Point:
    x, y, w, h = box
    return (x + w // 2, y + h // 2)


def point_in_polygon(pt: Point, poly: List[Point]) -> bool:
    x, y = pt
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / max((yj - yi), 1e-6) + xi):
            inside = not inside
        j = i
    return inside


def ccw(a: Point, b: Point, c: Point) -> bool:
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)


def norm_px(pts: List[List[float]], w: int, h: int) -> List[Point]:
    return [(int(clamp(x, 0, 1) * w), int(clamp(y, 0, 1) * h)) for x, y in pts]


# ---------------------------------------------------------------------------
# Async video capture thread
# ---------------------------------------------------------------------------
class AsyncCapture:
    """Reads frames in a background thread so the main loop never waits on I/O."""

    def __init__(self, source, use_gstreamer: bool = True):
        self._cap = self._open(source, use_gstreamer)
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    @staticmethod
    def _build_gst_pipeline(rtsp_url: str) -> str:
        return (
            f"rtspsrc location=\"{rtsp_url}\" latency=0 protocols=tcp ! "
            "rtph264depay ! h264parse ! nvv4l2decoder ! "
            "nvvidconv ! video/x-raw,format=BGRx ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=1 sync=0 max-buffers=1"
        )

    def _open(self, source, use_gst: bool) -> cv2.VideoCapture:
        if use_gst and isinstance(source, str) and source.startswith("rtsp://"):
            gst = self._build_gst_pipeline(source)
            cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                print(f"GStreamer HW decode pipeline opened")
                return cap
            print("GStreamer pipeline failed, falling back to default backend")
        src = int(source) if isinstance(source, str) and source.isdigit() else source
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")
        return cap

    def _reader(self) -> None:
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame = frame

    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def release(self) -> None:
        self._running = False
        self._thread.join(timeout=3)
        self._cap.release()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class Track:
    track_id: int
    bbox: BBox
    centroid: Point
    last_seen: float
    first_seen: float
    history: List[Point] = field(default_factory=list)
    zone_entered_at: Dict[str, float] = field(default_factory=dict)
    fired_events: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------
class HOGPersonDetector:
    def __init__(self, stride, scale, hit_threshold):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.stride, self.scale, self.hit_threshold = stride, scale, hit_threshold

    def detect(self, frame: np.ndarray) -> List[BBox]:
        boxes, weights = self.hog.detectMultiScale(
            frame, winStride=self.stride, padding=(8, 8),
            scale=self.scale, hitThreshold=self.hit_threshold)
        return [(int(x), int(y), int(w), int(h * 0.9))
                for (x, y, w, h), wt in zip(boxes, weights) if float(wt) >= 0.2]


def create_detector(config: Dict):
    backend = config["detector"].get("backend", "opencv_hog")
    if backend == "yolov5_trt":
        from yolo_trt_detector import YOLOv5TRTDetector
        dc = config["detector"]
        onnx = str(Path(__file__).resolve().parent.parent / "models" / dc.get("onnx_file", "yolov5n.onnx"))
        return YOLOv5TRTDetector(
            onnx_path=onnx, conf_threshold=float(dc.get("conf_threshold", 0.35)),
            iou_threshold=float(dc.get("iou_threshold", 0.45)), fp16=bool(dc.get("fp16", True)))
    dc = config["detector"]
    return HOGPersonDetector(
        tuple(dc.get("win_stride", [8, 8])), float(dc.get("scale", 1.05)),
        float(dc.get("hit_threshold", 0.0)))


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------
class CentroidTracker:
    def __init__(self, max_distance: float, max_age: float):
        self.max_distance, self.max_age = max_distance, max_age
        self._next_id = 1
        self._tracks: Dict[int, Track] = {}

    def update(self, dets: List[BBox], ts: float) -> List[Track]:
        dets = sorted(dets, key=lambda b: b[2] * b[3], reverse=True)
        unmatched = set(self._tracks.keys())
        assigned: Dict[int, BBox] = {}
        for det in dets:
            dc = center_of(det)
            best_id, best_d = None, float("inf")
            for tid in list(unmatched):
                d = math.dist(dc, self._tracks[tid].centroid)
                if d < best_d and d <= self.max_distance:
                    best_d, best_id = d, tid
            if best_id is not None:
                assigned[best_id] = det
                unmatched.remove(best_id)
        for tid, det in assigned.items():
            c = center_of(det)
            t = self._tracks[tid]
            t.bbox, t.centroid, t.last_seen = det, c, ts
            t.history.append(c)
            if len(t.history) > 32:
                t.history = t.history[-32:]
        used = set(assigned.values())
        for det in dets:
            if det in used:
                continue
            c = center_of(det)
            self._tracks[self._next_id] = Track(
                self._next_id, det, c, ts, ts, [c])
            self._next_id += 1
        for tid in [t for t, tr in self._tracks.items() if ts - tr.last_seen > self.max_age]:
            del self._tracks[tid]
        return list(self._tracks.values())


# ---------------------------------------------------------------------------
# Event writer
# ---------------------------------------------------------------------------
class EventWriter:
    MAX_SNAPSHOTS = 200

    def __init__(self, output_dir: Path, save_snapshots: bool):
        self.events_dir = output_dir / "events"
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = output_dir / "events.jsonl"
        self.save_snapshots = save_snapshots
        self._lock = threading.Lock()
        self._snap_count = len(list(self.events_dir.glob("*.jpg")))

    def write(self, event: Dict, frame: np.ndarray) -> None:
        with self._lock:
            with self.log_path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(event, ensure_ascii=True) + "\n")
            if self.save_snapshots:
                if self._snap_count >= self.MAX_SNAPSHOTS:
                    self._cleanup_old()
                name = f"{event['timestamp']}_{event['event_type']}_track{event['track_id']}.jpg"
                cv2.imwrite(str(self.events_dir / name), frame)
                self._snap_count += 1

    def _cleanup_old(self) -> None:
        files = sorted(self.events_dir.glob("*.jpg"), key=lambda f: f.stat().st_mtime)
        to_delete = len(files) - self.MAX_SNAPSHOTS // 2
        for f in files[:max(0, to_delete)]:
            try:
                f.unlink()
                self._snap_count -= 1
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Main demo (multi-camera)
# ---------------------------------------------------------------------------
class BehaviorDemo:
    def __init__(self, config: Dict):
        self.config = config
        self.show_window = bool(config["display"].get("show_window", True))
        self.window_name = config["display"].get("window_name", "BehaviorDemo")
        self._manager = None
        self._discovery_service = None

    def _get_camera_configs(self) -> List[Dict[str, Any]]:
        cameras_cfg = self.config.get("cameras")
        if cameras_cfg:
            from camera_discovery import discover_cameras
            return discover_cameras(self.config)
        if "camera" in self.config:
            cam = self.config["camera"]
            return [{
                "id": "cam-0",
                "name": cam.get("name", "camera-1"),
                "source": cam["source"],
                "use_gstreamer": cam.get("use_gstreamer", True),
                "enabled": True,
            }]
        return []

    def _start_web(self) -> None:
        try:
            from web_server_optimized import (
                set_config_ref,
                set_manager_ref,
                set_discovery_ref,
                start_server,
            )
            set_config_ref(self.config)
            set_manager_ref(self._manager)
            if self._discovery_service:
                set_discovery_ref(self._discovery_service)
            port = int(self.config.get("web", {}).get("port", 8080))
            ws_port = int(self.config.get("web", {}).get("ws_port", port + 1))
            config_ws_port = ws_port + 1
            start_server(port, ws_port, config_ws_port)
            print(f"[Web] HTTP:{port} WS:{ws_port} ConfigWS:{config_ws_port}")
        except ImportError:
            from web_server import set_config_ref, start_server
            set_config_ref(self.config)
            port = int(self.config.get("web", {}).get("port", 8080))
            start_server(port)
            print(f"[Web] HTTP:{port}")

    def run(self, override_source=None, max_frames=None, no_window=False, enable_web=False):
        from multi_camera_manager import MultiCameraManager

        camera_configs = self._get_camera_configs()
        if override_source:
            camera_configs = [{
                "id": "cam-0",
                "name": "override",
                "source": override_source,
                "use_gstreamer": True,
                "enabled": True,
            }]

        if not camera_configs:
            print("No cameras configured. Exiting.")
            return

        print(f"Starting {len(camera_configs)} camera(s):")
        for c in camera_configs:
            print(f"  {c['id']}: {c['name']} -> {c['source']}")

        self._manager = MultiCameraManager(self.config)
        self._manager.start_all(camera_configs)

        if enable_web:
            self._start_web()

        cameras_cfg = self.config.get("cameras", {})
        if cameras_cfg.get("mode") == "auto":
            from camera_discovery import CameraDiscoveryService
            self._discovery_service = CameraDiscoveryService(self.config)
            self._discovery_service.on_change(self._on_cameras_changed)
            self._discovery_service.start()

        show = self.show_window and not no_window
        if show and not os.environ.get("DISPLAY"):
            print("DISPLAY not set; headless mode")
            show = False

        fullscreen = False
        if show:
            try:
                hdmi_status = Path("/sys/class/drm/card0-HDMI-A-1/status").read_text().strip()
                if hdmi_status == "connected":
                    fullscreen = True
                    print("[Display] HDMI detected, starting in fullscreen")
            except Exception:
                pass
            try:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                if fullscreen:
                    cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            except cv2.error:
                show = False

        resize_w = int(self.config["display"].get("resize_width", 640))
        frame_n = 0

        try:
            while True:
                if show:
                    pipelines = self._manager.get_all_pipelines()
                    frames = []
                    for cam_id, pipeline in pipelines.items():
                        fb = pipeline.frame_buffer
                        raw = fb.raw_frame
                        if raw is not None:
                            frames.append((cam_id, pipeline, raw.copy()))

                    if frames:
                        display_frame = self._build_display_frame(frames, resize_w)
                        cv2.imshow(self.window_name, display_frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (27, ord("q")):
                            break
                        elif key == ord("f") or key == ord("F"):
                            fullscreen = not fullscreen
                            try:
                                if fullscreen:
                                    cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                                else:
                                    cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                            except cv2.error:
                                pass
                    else:
                        time.sleep(0.033)

                frame_n += 1
                if max_frames and frame_n >= max_frames:
                    break

        except KeyboardInterrupt:
            pass
        finally:
            self._manager.stop_all()
            if self._discovery_service:
                self._discovery_service.stop()
            if show:
                cv2.destroyAllWindows()

    def _build_display_frame(self, frames: List[tuple], resize_w: int) -> np.ndarray:
        n = len(frames)
        if n == 0:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        cols = 1 if n == 1 else 2 if n <= 4 else 3
        target_w = max(resize_w // cols, 320)

        resized = []
        for cam_id, pipeline, frame in frames:
            h, w = frame.shape[:2]
            r = target_w / max(w, 1)
            new_h = max(1, int(h * r))
            f = cv2.resize(frame, (target_w, new_h))
            f = self._draw_hud(f, cam_id, pipeline)
            resized.append(f)

        if n == 1:
            return resized[0]

        cell_h = max(f.shape[0] for f in resized)
        cell_w = max(f.shape[1] for f in resized)
        rows = (n + cols - 1) // cols

        canvas = np.zeros((cell_h * rows, cell_w * cols, 3), dtype=np.uint8)
        for i, f in enumerate(resized):
            r_idx = i // cols
            c_idx = i % cols
            y1 = r_idx * cell_h
            x1 = c_idx * cell_w
            fh, fw = f.shape[:2]
            canvas[y1:y1 + fh, x1:x1 + fw] = f

        return canvas

    @staticmethod
    def _draw_hud(frame: np.ndarray, cam_id: str, pipeline) -> np.ndarray:
        fh, fw = frame.shape[:2]
        fs = max(0.35, 0.5 * fw / 960)
        thick = max(1, int(fs * 2))
        pad = max(6, int(fs * 14))
        line_h = max(14, int(fs * 28))
        indent = pad + max(10, int(fs * 20))

        cam_name = pipeline.cam_name
        stats = pipeline.stats.snapshot(
            total_events=pipeline.event_store.total,
            current_model="",
        )
        online = pipeline.is_healthy()
        fps = stats.get("fps", 0.0)
        tracks = stats.get("tracks", 0)
        detections = stats.get("detections", 0)
        total_events = stats.get("total_events", 0)

        status_color = (0, 220, 100) if online else (0, 0, 220)
        cv2.circle(frame, (pad, pad + line_h // 2), max(4, int(fs * 8)), status_color, -1, cv2.LINE_AA)

        (tw, _), _ = cv2.getTextSize(cam_name, cv2.FONT_HERSHEY_SIMPLEX, fs, thick)
        cv2.putText(frame, cam_name, (fw - tw - pad, pad + line_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 255, 0), thick, cv2.LINE_AA)

        y = pad + line_h * 2
        cv2.putText(frame, f"FPS: {fps:.1f}", (indent, y),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (200, 200, 200), thick, cv2.LINE_AA)
        y += line_h
        cv2.putText(frame, f"Tracks: {tracks}", (indent, y),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (180, 130, 255), thick, cv2.LINE_AA)
        y += line_h
        cv2.putText(frame, f"Detections: {detections}", (indent, y),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (200, 180, 100), thick, cv2.LINE_AA)
        y += line_h
        cv2.putText(frame, f"Events: {total_events}", (indent, y),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (100, 200, 255), thick, cv2.LINE_AA)

        return frame

    def _on_cameras_changed(self, new_cameras: List[Dict]) -> None:
        current_ids = set(self._manager.get_all_pipelines().keys())
        new_ids = {c["id"] for c in new_cameras}

        for cam_id in current_ids - new_ids:
            self._manager.remove_camera(cam_id)
            print(f"[Demo] Removed camera: {cam_id}")

        for cam in new_cameras:
            if cam["id"] not in current_ids:
                self._manager.add_camera(cam)
                print(f"[Demo] Added camera: {cam['id']} ({cam['name']})")


def parse_args():
    p = argparse.ArgumentParser(description="Industrial Security Behavior Demo")
    p.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config" / "demo_config.json"))
    p.add_argument("--source", default=None)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--no-window", action="store_true")
    p.add_argument("--no-web", action="store_true")
    p.add_argument("--web-port", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as fp:
        config = json.load(fp)
    if args.web_port is not None:
        config.setdefault("web", {})["port"] = args.web_port
    demo = BehaviorDemo(config)
    demo.run(args.source, max_frames=args.max_frames, no_window=args.no_window,
             enable_web=not args.no_web)


if __name__ == "__main__":
    main()
