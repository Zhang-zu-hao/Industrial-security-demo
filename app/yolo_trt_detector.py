#!/usr/bin/env python3
"""YOLO detector with native TensorRT FP16 GPU inference.

Supports Ultralytics ONNX exports for **YOLOv5** (output ``[1, N, 85]``) and
**YOLOv8 / YOLO11** (output ``[1, 4+nc, N]``, e.g. ``[1, 84, 8400]`` for COCO).

Uses TensorRT 10 Python API + ctypes CUDA memory management.
Falls back to OpenCV DNN (CPU) if TensorRT is unavailable or crashes.
Auto-rebuilds engine if built on different device.

Only uses COCO class0 (person) when ``person_only`` is True.
"""
import ctypes
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

BBox = Tuple[int, int, int, int]

_cudart = None


def _load_cudart():
    global _cudart
    if _cudart is None:
        _cudart = ctypes.CDLL("libcudart.so.12")
    return _cudart


class _CudaBuffer:
    """Paired host (pinned) + device buffer managed via ctypes."""

    def __init__(self, shape: tuple, dtype=np.float16):
        self.host = np.zeros(shape, dtype=dtype)
        lib = _load_cudart()
        self._dev = ctypes.c_void_p()
        ret = lib.cudaMalloc(ctypes.byref(self._dev), ctypes.c_size_t(self.host.nbytes))
        assert ret == 0, f"cudaMalloc failed ({ret})"

    @property
    def device_ptr(self) -> int:
        return self._dev.value

    def h2d(self, stream):
        lib = _load_cudart()
        lib.cudaMemcpyAsync(
            self._dev, self.host.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_size_t(self.host.nbytes), ctypes.c_int(1), stream)

    def d2h(self, stream):
        lib = _load_cudart()
        lib.cudaMemcpyAsync(
            self.host.ctypes.data_as(ctypes.c_void_p), self._dev,
            ctypes.c_size_t(self.host.nbytes), ctypes.c_int(2), stream)

    def free(self):
        if self._dev.value:
            try:
                _load_cudart().cudaFree(self._dev)
            except Exception:
                pass
            self._dev = ctypes.c_void_p()

    def __del__(self):
        self.free()


def _io_tensor_names(engine, trt):
    inp = out = None
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            inp = name
        elif mode == trt.TensorIOMode.OUTPUT:
            out = name
    return inp, out


def _static_shape(shape: tuple) -> tuple:
    return tuple(int(d) if d != -1 and d > 0 else 1 for d in shape)


def _head_kind_from_shape(raw_shape: tuple) -> str:
    dims = [int(d) for d in raw_shape if d > 0]
    if len(dims) == 2:
        r0, r1 = dims[0], dims[1]
        lo, hi = min(r0, r1), max(r0, r1)
        if 5 <= lo <= 128 and hi >= 1000 and hi > lo * 8:
            return "v8" if r0 == lo else "v8_t"
        return "v5"
    if len(dims) == 3:
        _, d1, d2 = dims[0], dims[1], dims[2]
        if d2 == 85:
            return "v5"
        if 5 <= d1 <= 128 and d2 > d1 and d2 >= 100:
            return "v8"
        if 5 <= d2 <= 128 and d1 > d2 and d1 >= 100:
            return "v8_t"
    return "v5"


def _is_yolov8_model(onnx_path: str) -> bool:
    p = Path(onnx_path).name.lower()
    return any(k in p for k in ["yolov8", "yolo8", "yolov11", "yolo11"])


def _rebuild_engine(onnx_path: str, engine_path: str) -> bool:
    """Rebuild TRT engine on current device. Returns True if successful."""
    trtexec = "/usr/src/tensorrt/bin/trtexec"
    if not Path(trtexec).exists():
        print(f"[TRT] {trtexec} not found, cannot rebuild")
        return False
    
    print(f"[TRT] Rebuilding engine on this device...")
    
    # Backup old engine
    if Path(engine_path).exists():
        import shutil
        shutil.copy(engine_path, engine_path + ".old")
    
    cmd = [trtexec, f"--onnx={onnx_path}", f"--saveEngine={engine_path}", "--fp16"]
    print(f"[TRT] Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0 and Path(engine_path).exists():
            size_mb = Path(engine_path).stat().st_size / 1024 / 1024
            print(f"[TRT] ✅ Engine rebuilt successfully! Size: {size_mb:.1f} MB")
            # Remove backup
            old_bak = engine_path + ".old"
            if Path(old_bak).exists():
                Path(old_bak).unlink()
            return True
        else:
            print(f"[TRT] ❌ Rebuild failed (exit {result.returncode})")
            # Restore backup
            old_bak = engine_path + ".old"
            if Path(old_bak).exists():
                import shutil
                shutil.copy(old_bak, engine_path)
                Path(old_bak).unlink()
            return False
            
    except subprocess.TimeoutExpired:
        print(f"[TRT] ❌ Rebuild timed out")
        return False
    except Exception as e:
        print(f"[TRT] ❌ Rebuild error: {e}")
        return False


class YOLOTRTDetector:
    """YOLO with TensorRT FP16 on Jetson; auto-detects v5 vs v8/v11 head layout."""

    def __init__(self, onnx_path: str, conf_threshold: float = 0.35,
                 iou_threshold: float = 0.45, person_only: bool = True,
                 fp16: bool = True, yolo_head: str = "auto", **_kw):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.person_only = person_only
        self.yolo_head = (yolo_head or "auto").lower().strip()
        self._trt_ok = False
        self._head_kind = "v5"
        self.INPUT_SIZE = 640
        self._inp_name = "images"
        self._out_name = "output0"
        self._onnx_path = onnx_path
        self._trt_fail_count = 0
        self._trt_max_fails = 3
        self._is_yolov8 = _is_yolov8_model(onnx_path)

        engine_path = onnx_path.replace(".onnx", "_fp16.engine")

        if self._cuda_available():
            try:
                self._init_trt(engine_path, onnx_path)
            except Exception as exc:
                err_str = str(exc)
                # If error is about cross-device engine, try rebuilding
                if "different models of devices" in err_str or "illegal memory access" in err_str:
                    print(f"[TRT] Engine incompatible, attempting rebuild...")
                    if _rebuild_engine(onnx_path, engine_path):
                        try:
                            self._init_trt(engine_path, onnx_path)
                        except Exception as exc2:
                            print(f"TensorRT init failed after rebuild ({exc2}), falling back to DNN")
                            self._init_dnn(onnx_path)
                    else:
                        print(f"TensorRT init failed ({exc}), falling back to OpenCV DNN CPU")
                        self._init_dnn(onnx_path)
                else:
                    print(f"TensorRT init failed ({exc}), falling back to OpenCV DNN CPU")
                    self._init_dnn(onnx_path)
        else:
            print("CUDA not available, using OpenCV DNN CPU")
            self._init_dnn(onnx_path)

    @staticmethod
    def _cuda_available() -> bool:
        try:
            lib = ctypes.CDLL("libcudart.so.12")
            device_count = ctypes.c_int()
            ret = lib.cudaGetDeviceCount(ctypes.byref(device_count))
            if ret != 0 or device_count.value <= 0:
                return False
            return True
        except Exception:
            return False

    def _init_trt(self, engine_path: str, onnx_path: str):
        import tensorrt as trt

        if not Path(engine_path).exists():
            print(f"[TRT] Engine not found at {engine_path}, attempting build...")
            if not _rebuild_engine(onnx_path, engine_path):
                raise RuntimeError("Engine file not found and build failed")

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        try:
            print(f"[TRT] Loading engine from {engine_path}...")
            with open(engine_path, "rb") as f:
                self._engine = runtime.deserialize_cuda_engine(f.read())
            if self._engine is None:
                raise RuntimeError("deserialize returned None")
            self._context = self._engine.create_execution_context()
        except Exception as e:
            raise RuntimeError(f"Failed to load engine: {e}")

        try:
            inp_name, out_name = _io_tensor_names(self._engine, trt)
            if not inp_name or not out_name:
                raise RuntimeError("Could not find input/output tensor names on engine")
            self._inp_name, self._out_name = inp_name, out_name

            in_shape_raw = tuple(self._engine.get_tensor_shape(inp_name))
            in_shape = _static_shape(in_shape_raw)
            
            if len(in_shape) == 4 and in_shape[2] == in_shape[3] and in_shape[2] > 0:
                self.INPUT_SIZE = int(in_shape[2])

            self._context.set_input_shape(inp_name, in_shape)
            out_shape_raw = tuple(self._context.get_tensor_shape(out_name))
            out_shape = _static_shape(out_shape_raw)

            if self.yolo_head == "auto":
                self._head_kind = _head_kind_from_shape(out_shape)
            elif self.yolo_head in ("v5", "yolov5"):
                self._head_kind = "v5"
            elif self.yolo_head in ("v8", "yolov8", "v11", "yolo11", "yolo12"):
                self._head_kind = "v8" if out_shape[1] <= out_shape[2] else "v8_t"
            else:
                self._head_kind = _head_kind_from_shape(out_shape)

            _dtype_map = {trt.float32: np.float32, trt.float16: np.float16, trt.int8: np.int8, trt.int32: np.int32}
            inp_dtype = _dtype_map.get(self._engine.get_tensor_dtype(inp_name), np.float16)
            out_dtype = _dtype_map.get(self._engine.get_tensor_dtype(out_name), np.float32)
            self._inp_buf = _CudaBuffer(in_shape, inp_dtype)
            self._out_buf = _CudaBuffer(out_shape, out_dtype)
            print(f"[TRT] Input: {in_shape} {inp_dtype.__name__}, Output: {out_shape} {out_dtype.__name__}")

            lib = _load_cudart()
            self._stream = ctypes.c_void_p()
            assert lib.cudaStreamCreate(ctypes.byref(self._stream)) == 0

            self._context.set_tensor_address(self._inp_name, self._inp_buf.device_ptr)
            self._context.set_tensor_address(self._out_name, self._out_buf.device_ptr)

            test_blob = np.random.randn(*in_shape).astype(inp_dtype) * 0.5 + 0.5
            test_blob = np.clip(test_blob, 0, 1)
            for i in range(3):
                np.copyto(self._inp_buf.host, test_blob)
                self._inp_buf.h2d(self._stream)
                ret = self._context.execute_async_v3(self._stream.value)
                if not ret:
                    raise RuntimeError(f"Warmup round {i+1}: execute_async_v3 returned False")
                self._out_buf.d2h(self._stream)
                sync_ret = lib.cudaStreamSynchronize(self._stream)
                if sync_ret != 0:
                    raise RuntimeError(f"Warmup round {i+1}: cudaSync returned {sync_ret}")
                out_data = self._out_buf.host.astype(np.float32)
                if np.any(np.isnan(out_data)) or np.any(np.isinf(out_data)):
                    raise RuntimeError(f"Warmup round {i+1}: output contains NaN/Inf")

            self._trt_ok = True
            self._trt_fail_count = 0
            print(f"✅ YOLO TensorRT ready ({self._head_kind} head, {self.INPUT_SIZE}px)")
        except Exception as e:
            raise RuntimeError(str(e))

    def _init_dnn(self, onnx_path: str):
        self._net = cv2.dnn.readNetFromONNX(onnx_path)
        self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self._dnn_head_guess: Optional[str] = None
        print("✅ Using OpenCV DNN CPU fallback")

    def _fallback_to_dnn(self):
        if self._trt_ok:
            print("⚠️  TRT unstable, switching to OpenCV DNN CPU")
            self._trt_ok = False
            try:
                self._inp_buf.free()
                self._out_buf.free()
            except Exception:
                pass
            self._init_dnn(self._onnx_path)

    def detect(self, frame: np.ndarray) -> List[BBox]:
        h0, w0 = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0,
                                     (self.INPUT_SIZE, self.INPUT_SIZE), swapRB=True)

        if self._trt_ok:
            try:
                if blob.shape != self._inp_buf.host.shape:
                    raise ValueError(f"Input shape mismatch")
                np.copyto(self._inp_buf.host, blob.astype(self._inp_buf.host.dtype))
                self._inp_buf.h2d(self._stream)
                ret = self._context.execute_async_v3(self._stream.value)
                if not ret:
                    self._trt_fail_count += 1
                    if self._trt_fail_count >= self._trt_max_fails:
                        self._fallback_to_dnn()
                    return []
                self._out_buf.d2h(self._stream)
                sync_ret = _load_cudart().cudaStreamSynchronize(self._stream)
                if sync_ret != 0:
                    raise RuntimeError(f"cudaSync returned {sync_ret}")
                raw = self._out_buf.host.astype(np.float32)
                if np.any(np.isnan(raw)) or np.any(np.isinf(raw)):
                    raise RuntimeError("Output has NaN/Inf")
                self._trt_fail_count = 0
            except Exception as e:
                self._trt_fail_count += 1
                if self._trt_fail_count >= self._trt_max_fails:
                    self._fallback_to_dnn()
                return []

        if not self._trt_ok:
            self._net.setInput(blob)
            out = self._net.forward()
            raw = out[0] if isinstance(out, (list, tuple)) else out
            raw3 = np.expand_dims(raw, 0) if raw.ndim == 2 else raw
            if self._dnn_head_guess is None:
                self._dnn_head_guess = _head_kind_from_shape(tuple(raw3.shape))
            self._head_kind = (
                self.yolo_head if self.yolo_head != "auto" else self._dnn_head_guess
            )
            if self.yolo_head in ("v5", "yolov5"):
                self._head_kind = "v5"
            elif self.yolo_head in ("v8", "yolov8", "v11", "yolo11", "yolo12"):
                self._head_kind = _head_kind_from_shape(tuple(raw3.shape))
            raw = raw3

        return self._postprocess(raw, w0, h0)

    def _postprocess(self, raw: np.ndarray, w0: int, h0: int) -> List[BBox]:
        if raw.ndim == 2:
            raw = np.expand_dims(raw, 0)
        kind = self._head_kind
        if kind == "v8_t":
            preds = raw[0].astype(np.float32)
        elif kind == "v8":
            preds = raw[0].T.astype(np.float32)
        else:
            preds = raw[0].astype(np.float32)

        if kind in ("v8", "v8_t"):
            return self._postprocess_v8(preds, w0, h0)
        return self._postprocess_v5(preds, w0, h0)

    def _postprocess_v5(self, preds: np.ndarray, w0: int, h0: int) -> List[BBox]:
        obj_mask = preds[:, 4] > self.conf_threshold
        preds = preds[obj_mask]
        if len(preds) == 0:
            return []

        class_scores = preds[:, 5:]
        class_ids = class_scores.argmax(axis=1)
        scores = preds[:, 4] * class_scores[np.arange(len(class_ids)), class_ids]

        if self.person_only:
            pmask = class_ids == 0
            preds, scores = preds[pmask], scores[pmask]
            if len(preds) == 0:
                return []

        sx, sy = w0 / self.INPUT_SIZE, h0 / self.INPUT_SIZE
        cx, cy = preds[:, 0] * sx, preds[:, 1] * sy
        bw, bh = preds[:, 2] * sx, preds[:, 3] * sy
        x1 = (cx - bw / 2).clip(0, w0)
        y1 = (cy - bh / 2).clip(0, h0)

        boxes = np.stack([x1, y1, bw, bh], axis=1).astype(np.float32).tolist()
        scores_list = scores.astype(np.float32).tolist()

        indices = cv2.dnn.NMSBoxes(boxes, scores_list, self.conf_threshold, self.iou_threshold)
        if indices is None or len(indices) == 0:
            return []

        flat = np.array(indices).flatten()
        return [(int(boxes[i][0]), int(boxes[i][1]), int(boxes[i][2]), int(boxes[i][3]))
                for i in flat]

    def _postprocess_v8(self, preds: np.ndarray, w0: int, h0: int) -> List[BBox]:
        cls = preds[:, 4:]
        if cls.max() > 1.0 or cls.min() < 0.0:
            cls = 1.0 / (1.0 + np.exp(-np.clip(cls, -50, 50)))
        class_ids = cls.argmax(axis=1)
        scores = cls[np.arange(cls.shape[0]), class_ids]

        mask = scores > self.conf_threshold
        if self.person_only:
            mask &= class_ids == 0
        preds = preds[mask]
        scores = scores[mask]
        if len(preds) == 0:
            return []

        sx, sy = w0 / self.INPUT_SIZE, h0 / self.INPUT_SIZE
        cx, cy = preds[:, 0] * sx, preds[:, 1] * sy
        bw, bh = preds[:, 2] * sx, preds[:, 3] * sy
        x1 = (cx - bw / 2).clip(0, w0)
        y1 = (cy - bh / 2).clip(0, h0)

        boxes = np.stack([x1, y1, bw, bh], axis=1).astype(np.float32).tolist()
        scores_list = scores.astype(np.float32).tolist()

        indices = cv2.dnn.NMSBoxes(boxes, scores_list, self.conf_threshold, self.iou_threshold)
        if indices is None or len(indices) == 0:
            return []

        flat = np.array(indices).flatten()
        return [(int(boxes[i][0]), int(boxes[i][1]), int(boxes[i][2]), int(boxes[i][3]))
                for i in flat]


YOLOv5TRTDetector = YOLOTRTDetector
