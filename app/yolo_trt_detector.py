#!/usr/bin/env python3
"""YOLOv5 detector with native TensorRT FP16 GPU inference.

Uses TensorRT 10 Python API + ctypes CUDA memory management.
Falls back to OpenCV DNN (CPU) if TensorRT is unavailable.

Only detects 'person' class (COCO class 0) for the security demo.
"""
import ctypes
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
            _load_cudart().cudaFree(self._dev)
            self._dev = ctypes.c_void_p()

    def __del__(self):
        self.free()


class YOLOv5TRTDetector:
    """YOLOv5n with native TensorRT FP16 inference on Jetson GPU."""

    INPUT_SIZE = 640

    def __init__(self, onnx_path: str, conf_threshold: float = 0.35,
                 iou_threshold: float = 0.45, person_only: bool = True,
                 fp16: bool = True, **_kw):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.person_only = person_only
        self._trt_ok = False

        engine_path = onnx_path.replace(".onnx", "_fp16.engine")

        try:
            self._init_trt(engine_path, onnx_path)
        except Exception as exc:
            print(f"TensorRT init failed ({exc}), falling back to OpenCV DNN CPU")
            self._init_dnn(onnx_path)

    def _init_trt(self, engine_path: str, onnx_path: str):
        import tensorrt as trt

        if not Path(engine_path).exists():
            raise FileNotFoundError(f"Engine not found: {engine_path}. "
                                    f"Build with: trtexec --onnx={onnx_path} "
                                    f"--saveEngine={engine_path} --fp16")

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())
        self._context = self._engine.create_execution_context()

        self._inp_buf = _CudaBuffer((1, 3, self.INPUT_SIZE, self.INPUT_SIZE), np.float16)
        self._out_buf = _CudaBuffer((1, 25200, 85), np.float16)

        lib = _load_cudart()
        self._stream = ctypes.c_void_p()
        assert lib.cudaStreamCreate(ctypes.byref(self._stream)) == 0

        self._context.set_tensor_address("images", self._inp_buf.device_ptr)
        self._context.set_tensor_address("output0", self._out_buf.device_ptr)

        # Warmup
        self._inp_buf.h2d(self._stream)
        self._context.execute_async_v3(self._stream.value)
        self._out_buf.d2h(self._stream)
        lib.cudaStreamSynchronize(self._stream)

        self._trt_ok = True
        print(f"YOLOv5 TensorRT FP16 GPU inference ready (engine: {Path(engine_path).name})")

    def _init_dnn(self, onnx_path: str):
        self._net = cv2.dnn.readNetFromONNX(onnx_path)
        self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print("YOLOv5 fallback: OpenCV DNN CPU")

    def detect(self, frame: np.ndarray) -> List[BBox]:
        h0, w0 = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0,
                                     (self.INPUT_SIZE, self.INPUT_SIZE), swapRB=True)

        if self._trt_ok:
            np.copyto(self._inp_buf.host, blob.astype(np.float16))
            self._inp_buf.h2d(self._stream)
            self._context.execute_async_v3(self._stream.value)
            self._out_buf.d2h(self._stream)
            _load_cudart().cudaStreamSynchronize(self._stream)
            raw = self._out_buf.host.astype(np.float32)
        else:
            self._net.setInput(blob)
            raw = self._net.forward()

        return self._postprocess(raw, w0, h0)

    def _postprocess(self, raw: np.ndarray, w0: int, h0: int) -> List[BBox]:
        preds = raw[0]  # (25200, 85)

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
