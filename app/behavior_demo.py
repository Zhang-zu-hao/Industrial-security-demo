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
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
        # Optimized GStreamer pipeline for maximum quality and performance
        return (
            f"uridecodebin uri={rtsp_url} ! nvvidconv ! "
            "video/x-raw,format=BGRx ! videoconvert ! "
            "video/x-raw,format=BGR ! appsink drop=0 sync=0 max-buffers=1"
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
        self._snap_count = len(list(self.events_dir.glob("*.jpg")))

    def write(self, event: Dict, frame: np.ndarray) -> None:
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
# Main demo
# ---------------------------------------------------------------------------
class BehaviorDemo:
    def __init__(self, config: Dict):
        self.config = config
        self.detector = create_detector(config)
        tc = config["tracker"]
        self.tracker = CentroidTracker(float(tc.get("max_distance", 80)), float(tc.get("max_age_seconds", 1.5)))
        self.output_dir = Path(config["output"]["root"]).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.event_writer = EventWriter(self.output_dir, bool(config["output"].get("save_snapshots", True)))
        self.cooldown = float(config["rules"].get("event_cooldown_seconds", 8))
        self.show_window = bool(config["display"].get("show_window", True))
        self.window_name = config["display"].get("window_name", "BehaviorDemo")
        self._web_fb = self._web_es = self._web_st = None
        self._model_switch_request = None
        self._model_switch_result = None
        self._model_switch_event = threading.Event()
        self._last_global_emit = 0.0
        self.features = config.get("features", {
            "human_detect": True,
            "tracking": True,
            "zone_detection": True,
            "line_crossing": True,
            "loitering": True
        })
        self.config.setdefault("features", dict(self.features))

    def enable_web(self) -> None:
        try:
            from web_server_optimized import frame_buffer, event_store, stats, set_config_ref, set_demo_ref, start_server
            self._web_fb, self._web_es, self._web_st = frame_buffer, event_store, stats
            set_config_ref(self.config)
            set_demo_ref(self)
            stats.update_features(self.features)
            port = int(self.config.get("web", {}).get("port", 8080))
            ws_port = int(self.config.get("web", {}).get("ws_port", port + 1))
            config_ws_port = ws_port + 1
            start_server(port, ws_port, config_ws_port)
            print(f"[Web] HTTP:{port} WS:{ws_port} ConfigWS:{config_ws_port}")
        except ImportError:
            from web_server import frame_buffer, event_store, stats, set_config_ref, start_server
            self._web_fb, self._web_es, self._web_st = frame_buffer, event_store, stats
            set_config_ref(self.config)
            port = int(self.config.get("web", {}).get("port", 8080))
            start_server(port)
            print(f"[Web] HTTP:{port}")

    def switch_model(self, onnx_file: str) -> dict:
        """Request model switch and block until complete (thread-safe)."""
        self._model_switch_result = None
        self._model_switch_event.clear()
        self._model_switch_request = onnx_file
        if self._model_switch_event.wait(timeout=600):
            return self._model_switch_result or {"status": "error", "error": "unknown"}
        return {"status": "error", "error": "timeout"}

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
                "size_mb": round(f.stat().st_size / 1024 / 1024, 1)
            })
        return models

    def _emit(self, frame, track, etype, meta=None):
        now = time.time()
        if now - track.fired_events.get(etype, 0.0) < self.cooldown:
            return
        if now - self._last_global_emit < 2.0:
            return
        track.fired_events[etype] = now
        self._last_global_emit = now
        ev = {"timestamp": now_ts(), "epoch_seconds": now,
              "camera_name": self.config["camera"].get("name", "cam-1"),
              "event_type": etype, "track_id": track.track_id,
              "bbox": list(track.bbox), "centroid": list(track.centroid)}
        if meta:
            ev.update(meta)
        self.event_writer.write(ev, frame)
        if self._web_es:
            self._web_es.add(ev)
        # Update stats with event count
        if self._web_st:
            self._web_st.update_events(self._web_es.total)

    def _draw_zones(self, frame, zones):
        h, w = frame.shape[:2]
        for z in zones:
            pts = norm_px(z["points"], w, h)
            col = tuple(z.get("color_bgr", [0, 255, 255]))
            ov = frame.copy()
            cv2.fillPoly(ov, [np.array(pts, np.int32)], col)
            cv2.addWeighted(ov, 0.12, frame, 0.88, 0, frame)
            cv2.polylines(frame, [np.array(pts, np.int32)], True, col, 2)
            cv2.putText(frame, z["name"], pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)

    def _draw_lines(self, frame, lines):
        h, w = frame.shape[:2]
        for lc in lines:
            p1, p2 = norm_px([lc["start"]], w, h)[0], norm_px([lc["end"]], w, h)[0]
            col = tuple(lc.get("color_bgr", [255, 0, 255]))
            cv2.arrowedLine(frame, p1, p2, col, 2, tipLength=0.03)
            cv2.putText(frame, lc["name"], p1, cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)

    def _apply_rules(self, frame, track, zones, lines):
        h, w = frame.shape[:2]
        
        # Zone detection (only if enabled)
        if self.features.get("zone_detection", True):
            for z in zones:
                zn = z["name"]
                pts = norm_px(z["points"], w, h)
                inside = point_in_polygon(track.centroid, pts)
                if inside and zn not in track.zone_entered_at:
                    track.zone_entered_at[zn] = time.time()
                    self._emit(frame, track, "zone_enter", {"zone_name": zn})
                if not inside and zn in track.zone_entered_at:
                    del track.zone_entered_at[zn]
                if inside and self.features.get("loitering", True):
                    dw = time.time() - track.zone_entered_at[zn]
                    if dw >= float(z.get("dwell_seconds", 10)):
                        self._emit(frame, track, "loitering", {"zone_name": zn, "dwell_seconds": round(dw, 2)})
        
        # Line crossing detection (only if enabled)
        if self.features.get("line_crossing", True) and len(track.history) >= 2:
            prev, curr = track.history[-2], track.history[-1]
            for lc in lines:
                p1, p2 = norm_px([lc["start"]], w, h)[0], norm_px([lc["end"]], w, h)[0]
                if segments_intersect(prev, curr, p1, p2):
                    self._emit(frame, track, "line_cross", {"line_name": lc["name"]})

    def run(self, override_source=None, max_frames=None, no_window=False):
        source = override_source or self.config["camera"]["source"]
        use_gst = bool(self.config["camera"].get("use_gstreamer", True))
        cap = AsyncCapture(source, use_gstreamer=use_gst)
        resize_w = int(self.config["display"].get("resize_width", 640))
        infer_interval = int(self.config["detector"].get("infer_interval", 1))
        show = self.show_window and not no_window

        if show and not os.environ.get("DISPLAY"):
            print("DISPLAY not set; headless mode")
            show = False
        if show:
            try:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            except cv2.error:
                show = False

        frame_n = 0
        t0 = time.time()
        last_boxes: List[BBox] = []
        fps_smooth = 0.0
        last_conf_threshold = self.config["detector"].get("conf_threshold", 0.35)

        while True:
            frame = cap.read()
            if frame is None:
                time.sleep(0.005)
                continue

            if resize_w > 0 and frame.shape[1] > resize_w:
                r = resize_w / frame.shape[1]
                frame = cv2.resize(frame, (resize_w, int(frame.shape[0] * r)))

            zones = self.config["rules"].get("zones", [])
            lines_cfg = self.config["rules"].get("lines", [])
            self.cooldown = float(self.config["rules"].get("event_cooldown_seconds", 8))
            self.features = self.config.get("features", self.features)

            if self._model_switch_request:
                onnx_file = self._model_switch_request
                self._model_switch_request = None
                try:
                    self.config["detector"]["onnx_file"] = onnx_file
                    self.detector = create_detector(self.config)
                    last_boxes = []
                    self._model_switch_result = {"status": "ok", "model": onnx_file}
                    print(f"[Model] Switched to {onnx_file}")
                except Exception as e:
                    self._model_switch_result = {"status": "error", "error": str(e)}
                    print(f"[Model] Switch failed: {e}")
                finally:
                    self._model_switch_event.set()

            current_conf = self.config["detector"].get("conf_threshold", 0.35)
            if abs(current_conf - last_conf_threshold) > 0.001:
                if hasattr(self.detector, 'conf_threshold'):
                    self.detector.conf_threshold = current_conf
                    last_conf_threshold = current_conf

            ts = time.time()
            detect_enabled = self.features.get("human_detect", True)

            if detect_enabled:
                if frame_n % infer_interval == 0:
                    last_boxes = self.detector.detect(frame)
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

                # Draw all bboxes FIRST so event snapshots include them
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

                # Apply rules AFTER drawing so snapshots contain annotations
                if tracking_on and tracks:
                    check_rules = (self.features.get("zone_detection", False) or
                                   self.features.get("line_crossing", False) or
                                   self.features.get("loitering", False))
                    if check_rules:
                        for tr in tracks:
                            self._apply_rules(frame, tr, zones, lines_cfg)

            if self._web_fb:
                quality = self.config.get("display", {}).get("jpeg_quality", 85)
                self._web_fb.update(frame, quality=quality)
            if self._web_st:
                self._web_st.update(fps_smooth, len(last_boxes), len(tracks), infer_interval)

            if show:
                cv2.imshow(self.window_name, frame)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break

            if max_frames and frame_n >= max_frames:
                print(f"frames={frame_n} fps={fps_smooth:.1f} source={source}")
                break

        cap.release()
        if show:
            cv2.destroyAllWindows()


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
    if not args.no_web:
        demo.enable_web()
    demo.run(args.source, max_frames=args.max_frames, no_window=args.no_window)


if __name__ == "__main__":
    main()
