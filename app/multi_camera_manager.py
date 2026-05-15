#!/usr/bin/env python3
"""Multi-camera pipeline manager.

Each camera runs an independent pipeline: capture -> detect -> track -> rules.
The detector model is shared across cameras with serialized inference.
"""
import json
import math
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from behavior_demo import (
    AsyncCapture,
    CentroidTracker,
    EventWriter,
    Track,
    center_of,
    clamp,
    create_detector,
    norm_px,
    now_ts,
    point_in_polygon,
    segments_intersect,
)

BBox = Tuple[int, int, int, int]


class FrameBuffer:
    """Thread-safe frame buffer with synchronous low-latency encoding."""

    def __init__(self, web_quality: int = 50):
        self._lock = threading.RLock()
        self._frame: Optional[bytes] = None
        self._raw_frame: Optional[np.ndarray] = None
        self._web_quality = web_quality
        self._timestamp = 0.0
        self._last_update_time = 0.0

    def update(self, frame: np.ndarray, quality: int = None) -> bool:
        if quality is not None:
            self._web_quality = max(1, min(100, quality))
        try:
            current_time = time.time()
            if current_time - self._last_update_time < 0.003:
                return False
            ok, jpeg = cv2.imencode(".jpg", frame, [
                cv2.IMWRITE_JPEG_QUALITY,
                min(95, max(20, self._web_quality))
            ])
            if not ok:
                return False
            with self._lock:
                self._frame = jpeg.tobytes()
                self._raw_frame = frame
                self._timestamp = current_time
                self._last_update_time = current_time
            return True
        except Exception:
            return False

    def get(self) -> Optional[bytes]:
        with self._lock:
            return self._frame

    @property
    def raw_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._raw_frame

    @property
    def timestamp(self) -> float:
        with self._lock:
            return self._timestamp

    def set_quality(self, quality: int) -> None:
        self._web_quality = max(1, min(100, quality))

    def stop(self) -> None:
        pass


class EventStore:
    """Thread-safe ring buffer for events with SQLite persistence."""

    _shared_persistent = None
    _shared_persistent_lock = threading.Lock()

    def __init__(self, cam_id: str = "", db_path: str = "output/events.db", max_events: int = 500):
        self._lock = threading.Lock()
        self._events: List[Dict] = []
        self._max = max_events
        self._total = 0
        self._cam_id = cam_id
        self._db_path = db_path

    @classmethod
    def _get_shared_persistent(cls, db_path: str):
        with cls._shared_persistent_lock:
            if cls._shared_persistent is None:
                from event_store_db import PersistentEventStore
                cls._shared_persistent = PersistentEventStore(db_path)
            return cls._shared_persistent

    def _get_persistent(self):
        return self._get_shared_persistent(self._db_path)

    def add(self, event: Dict) -> None:
        with self._lock:
            self._events.append(event)
            self._total += 1
            if len(self._events) > self._max:
                self._events = self._events[-self._max:]
        # Write to SQLite asynchronously (non-blocking)
        try:
            self._get_persistent().add(event)
        except Exception:
            pass

    def recent(self, n: int = 50) -> List[Dict]:
        with self._lock:
            items = self._events[-n:]
            items.reverse()
            return items

    @property
    def total(self) -> int:
        with self._lock:
            total_mem = self._total
        # Include persistent count
        try:
            total_db = self._get_persistent().count(self._cam_id)
            return max(total_mem, total_db)
        except Exception:
            return total_mem

    def clear(self) -> None:
        with self._lock:
            self._events.clear()
        try:
            self._get_persistent().clear(self._cam_id)
        except Exception:
            pass

    def get_events_by_date(self, date_str: str, limit: int = 100) -> List[Dict]:
        try:
            return self._get_persistent().get_events_by_date(date_str, limit, self._cam_id)
        except Exception:
            pass
        if not date_str:
            return self.recent(limit)
        with self._lock:
            cached = [e for e in self._events if e.get("timestamp", "").startswith(date_str)]
            if cached:
                cached.sort(key=lambda e: e.get("epoch_seconds", 0), reverse=True)
                return cached[:limit]
        return []


class StatsCollector:
    """Real-time stats collector with sliding window smoothing."""

    def __init__(self, window_size: int = 5):
        self._lock = threading.Lock()
        self.uptime_start: float = time.time()
        self._last_update: float = 0.0
        self.infer_interval: int = 3
        self._window_size = window_size
        self._fps_history: List[float] = []
        self._detections_history: List[int] = []
        self._tracks_history: List[int] = []
        self._smoothed_fps: float = 0.0
        self._smoothed_detections: int = 0
        self._smoothed_tracks: int = 0
        self.feature_flags: Dict[str, bool] = {
            "human_detect": False,
            "tracking": False,
            "zone_detection": False,
            "line_crossing": False,
            "loitering": False,
        }

    def update(self, fps: float, detections: int, tracks: int, infer_interval: int = 3) -> None:
        with self._lock:
            self._fps_history.append(fps)
            self._detections_history.append(detections)
            self._tracks_history.append(tracks)

            if len(self._fps_history) > self._window_size:
                self._fps_history.pop(0)
            if len(self._detections_history) > self._window_size:
                self._detections_history.pop(0)
            if len(self._tracks_history) > self._window_size:
                self._tracks_history.pop(0)

            if self._fps_history:
                self._smoothed_fps = sum(self._fps_history) / len(self._fps_history)
            if self._detections_history:
                self._smoothed_detections = round(sum(self._detections_history) / len(self._detections_history))
            if self._tracks_history:
                self._smoothed_tracks = round(sum(self._tracks_history) / len(self._tracks_history))

            self.infer_interval = infer_interval
            self._last_update = time.time()

    def snapshot(self, total_events: int = 0, current_model: str = "") -> Dict[str, Any]:
        with self._lock:
            return {
                "fps": round(self._smoothed_fps, 1),
                "detections": self._smoothed_detections,
                "tracks": self._smoothed_tracks,
                "uptime_seconds": round(time.time() - self.uptime_start, 1),
                "total_events": total_events,
                "features": dict(self.feature_flags),
                "infer_interval": self.infer_interval,
                "current_model": current_model,
            }

    def update_features(self, features: Dict[str, bool]) -> None:
        with self._lock:
            self.feature_flags.update(features)


class SharedDetector:
    """Shared detector model with serialized inference for thread safety."""

    def __init__(self, config: Dict):
        self._config = config
        self._detector = create_detector(config)
        self._lock = threading.Lock()
        self._current_onnx = config["detector"].get("onnx_file", "")

    def detect(self, frame: np.ndarray) -> List[BBox]:
        with self._lock:
            return self._detector.detect(frame)

    def switch_model(self, onnx_file: str) -> Dict:
        with self._lock:
            try:
                self._config["detector"]["onnx_file"] = onnx_file
                self._detector = create_detector(self._config)
                self._current_onnx = onnx_file
                print(f"[Model] Switched to {onnx_file}")
                return {"status": "ok", "model": onnx_file}
            except Exception as e:
                print(f"[Model] Switch failed: {e}")
                return {"status": "error", "error": str(e)}

    def update_conf_threshold(self, threshold: float) -> None:
        with self._lock:
            if hasattr(self._detector, 'conf_threshold'):
                self._detector.conf_threshold = threshold

    @property
    def current_model(self) -> str:
        return self._current_onnx


class CameraPipeline:
    """Independent pipeline for a single camera."""

    def __init__(
        self,
        cam_id: str,
        cam_config: Dict[str, Any],
        global_config: Dict[str, Any],
        shared_detector: SharedDetector,
    ):
        self.cam_id = cam_id
        self.cam_config = cam_config
        self.global_config = global_config
        self.shared_detector = shared_detector

        self.cam_name = cam_config.get("name", cam_id)
        self.source = cam_config["source"]
        self.use_gst = bool(cam_config.get("use_gstreamer", True))

        tc = global_config["tracker"]
        self.tracker = CentroidTracker(
            float(tc.get("max_distance", 80)),
            float(tc.get("max_age_seconds", 1.5)),
        )

        output_root = Path(global_config["output"]["root"]).resolve()
        self.output_dir = output_root / cam_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.event_writer = EventWriter(
            self.output_dir,
            bool(global_config["output"].get("save_snapshots", True)),
        )

        self.frame_buffer = FrameBuffer(web_quality=50)
        db_path = str(output_root / "events.db")
        self.event_store = EventStore(cam_id=cam_id, db_path=db_path)
        self.stats = StatsCollector()

        self.cooldown = float(global_config["rules"].get("event_cooldown_seconds", 8))
        self.features = dict(global_config.get("features", {
            "human_detect": True,
            "tracking": True,
            "zone_detection": True,
            "line_crossing": True,
            "loitering": True,
        }))
        self.stats.update_features(self.features)

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._capture: Optional[AsyncCapture] = None
        self._health = {"online": False, "last_frame": 0.0, "frame_count": 0, "retry_count": 0}
        self._health_lock = threading.Lock()

    def start(self) -> None:
        self._running = True
        self._capture = AsyncCapture(self.source, use_gstreamer=self.use_gst)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[Pipeline] {self.cam_id} ({self.cam_name}) started")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        if self._capture:
            self._capture.release()
        self.frame_buffer.stop()
        print(f"[Pipeline] {self.cam_id} stopped")

    def is_healthy(self) -> bool:
        with self._health_lock:
            return self._health["online"]

    def _check_health(self) -> bool:
        now = time.time()
        with self._health_lock:
            last = self._health["last_frame"]
            # No frame for 10s = offline
            if now - last > 10.0:
                self._health["online"] = False
                return False
            self._health["online"] = True
            return True

    def _reconnect(self) -> bool:
        with self._health_lock:
            self._health["retry_count"] += 1
            retry = self._health["retry_count"]
        print(f"[Pipeline] {self.cam_id} reconnecting (attempt {retry})...")
        try:
            if self._capture:
                self._capture.release()
            self._capture = AsyncCapture(self.source, use_gstreamer=self.use_gst)
            with self._health_lock:
                self._health["retry_count"] = 0
                self._health["last_frame"] = time.time()
            print(f"[Pipeline] {self.cam_id} reconnected")
            return True
        except Exception as e:
            print(f"[Pipeline] {self.cam_id} reconnect failed: {e}")
            return False

    def _loop(self) -> None:
        resize_w = int(self.global_config["display"].get("resize_width", 640))
        infer_interval = int(self.global_config["detector"].get("infer_interval", 1))

        frame_n = 0
        t0 = time.time()
        last_boxes: List[BBox] = []
        fps_smooth = 0.0
        consecutive_none = 0

        while self._running:
            frame = self._capture.read()
            if frame is None:
                consecutive_none += 1
                time.sleep(0.005)
                # Health check: if no frame for 5s, try reconnect
                if consecutive_none >= 1000:  # ~5s at 0.005s sleep
                    consecutive_none = 0
                    if not self._check_health():
                        if not self._reconnect():
                            time.sleep(5.0)  # Wait before next retry
                continue

            consecutive_none = 0
            with self._health_lock:
                self._health["last_frame"] = time.time()
                self._health["frame_count"] += 1
                self._health["online"] = True

            if resize_w > 0 and frame.shape[1] > resize_w:
                r = resize_w / frame.shape[1]
                frame = cv2.resize(frame, (resize_w, int(frame.shape[0] * r)))

            zones = self.global_config["rules"].get("camera_zones", {}).get(self.cam_id, {}).get("zones", None)
            if zones is None:
                zones = self.global_config["rules"].get("zones", [])
            lines_cfg = self.global_config["rules"].get("camera_zones", {}).get(self.cam_id, {}).get("lines", None)
            if lines_cfg is None:
                lines_cfg = self.global_config["rules"].get("lines", [])
            self.cooldown = float(self.global_config["rules"].get("event_cooldown_seconds", 8))
            self.features = self.global_config.get("features", self.features)

            ts = time.time()
            detect_enabled = self.features.get("human_detect", True)

            if detect_enabled:
                if frame_n % infer_interval == 0:
                    last_boxes = self.shared_detector.detect(frame)
            else:
                last_boxes = []

            if detect_enabled and self.features.get("tracking", True):
                tracks = self.tracker.update(last_boxes, ts)
            else:
                tracks = []

            frame_n += 1
            elapsed = max(time.time() - t0, 1e-6)
            fps_instant = frame_n / elapsed
            fps_smooth = 0.9 * fps_smooth + 0.1 * fps_instant if fps_smooth > 0 else fps_instant

            if detect_enabled:
                self._draw_zones(frame, zones)
                self._draw_lines(frame, lines_cfg)

                tracking_on = self.features.get("tracking", True)

                if tracking_on and tracks:
                    for tr in tracks:
                        x, y, w, h = tr.bbox
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 0), 2)
                        lbl = f"#{tr.track_id}"
                        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                        cv2.rectangle(frame, (x, max(0, y - th - 8)), (x + tw + 6, y), (0, 220, 0), -1)
                        cv2.putText(frame, lbl, (x + 3, max(th + 4, y - 4)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
                else:
                    for box in last_boxes:
                        x, y, w, h = box
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 180, 255), 2)

                if tracking_on and tracks:
                    check_rules = (self.features.get("zone_detection", False) or
                                   self.features.get("line_crossing", False) or
                                   self.features.get("loitering", False))
                    if check_rules:
                        for tr in tracks:
                            self._apply_rules(frame, tr, zones, lines_cfg)

            quality = self.global_config.get("display", {}).get("web_jpeg_quality", 50)
            self.frame_buffer.update(frame, quality=quality)
            self.stats.update(fps_smooth, len(last_boxes), len(tracks), infer_interval)

    def _emit(self, frame, track, etype, meta=None):
        now = time.time()
        if now - track.fired_events.get(etype, 0.0) < self.cooldown:
            return
        track.fired_events[etype] = now
        ev = {
            "timestamp": now_ts(),
            "epoch_seconds": now,
            "camera_id": self.cam_id,
            "camera_name": self.cam_name,
            "event_type": etype,
            "track_id": track.track_id,
            "bbox": list(track.bbox),
            "centroid": list(track.centroid),
        }
        if meta:
            ev.update(meta)
        self.event_writer.write(ev, frame)
        self.event_store.add(ev)

    def _draw_zones(self, frame, zones):
        h, w = frame.shape[:2]
        for z in zones:
            try:
                pts_data = z.get("points", [])
                if not pts_data or len(pts_data) < 3:
                    continue
                pts = norm_px(pts_data, w, h)
                col = tuple(z.get("color_bgr", [0, 255, 255]))
                ov = frame.copy()
                cv2.fillPoly(ov, [np.array(pts, np.int32)], col)
                cv2.addWeighted(ov, 0.12, frame, 0.88, 0, frame)
                cv2.polylines(frame, [np.array(pts, np.int32)], True, col, 2)
                name = z.get("name", "zone")
                cv2.putText(frame, name, pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
            except Exception:
                continue

    def _draw_lines(self, frame, lines):
        h, w = frame.shape[:2]
        for lc in lines:
            try:
                start_data = lc.get("start")
                end_data = lc.get("end")
                if not start_data or not end_data:
                    continue
                p1, p2 = norm_px([start_data], w, h)[0], norm_px([end_data], w, h)[0]
                col = tuple(lc.get("color_bgr", [255, 0, 255]))
                cv2.arrowedLine(frame, p1, p2, col, 2, tipLength=0.03)
                name = lc.get("name", "line")
                cv2.putText(frame, name, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
            except Exception:
                continue

    def _apply_rules(self, frame, track, zones, lines):
        h, w = frame.shape[:2]

        if self.features.get("zone_detection", False):
            for z in zones:
                try:
                    zn = z.get("name", "zone")
                    pts_data = z.get("points", [])
                    if not pts_data or len(pts_data) < 3:
                        continue
                    pts = norm_px(pts_data, w, h)
                    inside = point_in_polygon(track.centroid, pts)
                    if inside and zn not in track.zone_entered_at:
                        track.zone_entered_at[zn] = time.time()
                        self._emit(frame, track, "zone_enter", {"zone_name": zn})
                    if not inside and zn in track.zone_entered_at:
                        del track.zone_entered_at[zn]
                    if inside and self.features.get("loitering", False) and zn in track.zone_entered_at:
                        dw = time.time() - track.zone_entered_at[zn]
                        if dw >= float(z.get("dwell_seconds", 10)):
                            self._emit(frame, track, "loitering", {"zone_name": zn, "dwell_seconds": round(dw, 2)})
                except Exception:
                    continue

        if self.features.get("line_crossing", False) and len(track.history) >= 2:
            prev, curr = track.history[-2], track.history[-1]
            for lc in lines:
                try:
                    start_data = lc.get("start")
                    end_data = lc.get("end")
                    if not start_data or not end_data:
                        continue
                    p1, p2 = norm_px([start_data], w, h)[0], norm_px([end_data], w, h)[0]
                    if segments_intersect(prev, curr, p1, p2):
                        self._emit(frame, track, "line_cross", {"line_name": lc.get("name", "line")})
                except Exception:
                    continue


class MultiCameraManager:
    """Manages all camera pipelines with hot-add/remove support."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.shared_detector = SharedDetector(config)
        self._pipelines: Dict[str, CameraPipeline] = {}
        self._lock = threading.Lock()
        self._running = False

    def start_all(self, camera_configs: List[Dict[str, Any]]) -> None:
        self._running = True
        for cam_cfg in camera_configs:
            self._add_pipeline(cam_cfg)

    def stop_all(self) -> None:
        self._running = False
        with self._lock:
            for cam_id, pipeline in list(self._pipelines.items()):
                pipeline.stop()
            self._pipelines.clear()
        if EventStore._shared_persistent is not None:
            try:
                EventStore._shared_persistent.close()
            except Exception:
                pass
            EventStore._shared_persistent = None

    def _add_pipeline(self, cam_cfg: Dict[str, Any]) -> Optional[str]:
        cam_id = cam_cfg["id"]
        with self._lock:
            if cam_id in self._pipelines:
                return None
            pipeline = CameraPipeline(cam_id, cam_cfg, self.config, self.shared_detector)
            self._pipelines[cam_id] = pipeline
        pipeline.start()
        return cam_id

    def add_camera(self, cam_cfg: Dict[str, Any]) -> Optional[str]:
        return self._add_pipeline(cam_cfg)

    def remove_camera(self, cam_id: str) -> bool:
        with self._lock:
            if cam_id not in self._pipelines:
                return False
            pipeline = self._pipelines.pop(cam_id)
        pipeline.stop()
        return True

    def get_pipeline(self, cam_id: str) -> Optional[CameraPipeline]:
        with self._lock:
            return self._pipelines.get(cam_id)

    def list_cameras(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                {
                    "id": p.cam_id,
                    "name": p.cam_name,
                    "source": p.source,
                    "use_gstreamer": p.use_gst,
                    "enabled": p._running,
                }
                for p in self._pipelines.values()
            ]

    def get_all_pipelines(self) -> Dict[str, CameraPipeline]:
        with self._lock:
            return dict(self._pipelines)

    def switch_model(self, onnx_file: str) -> Dict:
        return self.shared_detector.switch_model(onnx_file)

    @property
    def current_model(self) -> str:
        return self.shared_detector.current_model

    @staticmethod
    def list_models():
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