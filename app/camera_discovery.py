#!/usr/bin/env python3
"""Camera auto-discovery module.

Scans local subnet for RTSP cameras and returns available camera configurations.
Supports both auto-discovery and manual configuration modes.
"""
import ipaddress
import socket
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional

import cv2

COMMON_RTSP_PATHS = [
    "/Streaming/Channels/101",
    "/Streaming/Channels/1",
    "/cam/realmonitor?channel=1&subtype=0",
    "/h264/ch1/main/av_stream",
    "/live/ch00_0",
    "/live",
    "/stream1",
]


def _ping_host(ip: str, timeout: float = 0.5) -> bool:
    try:
        ret = subprocess.run(
            ["ping", "-c", "1", "-W", str(int(timeout)), ip],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout + 0.5,
        )
        return ret.returncode == 0
    except Exception:
        return False


def _try_rtsp(url: str, timeout: float = 3.0) -> bool:
    try:
        cap = cv2.VideoCapture(url)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(timeout * 1000))
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None:
            return True
        return False
    except Exception:
        return False


def _scan_single_ip(
    ip: str,
    username: str,
    password: str,
    rtsp_paths: List[str],
    timeout: float = 3.0,
) -> Optional[Dict[str, Any]]:
    if not _ping_host(ip, timeout=0.5):
        return None

    userinfo = username
    if password or password == "":
        userinfo = f"{username}:{password}"

    for path in rtsp_paths:
        url = f"rtsp://{userinfo}@{ip}{path}"
        if _try_rtsp(url, timeout=timeout):
            return {
                "source": url,
                "ip": ip,
                "rtsp_path": path,
                "use_gstreamer": True,
                "enabled": True,
            }
    return None


def discover_cameras(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    cameras_cfg = config.get("cameras", {})
    mode = cameras_cfg.get("mode", "manual")

    if mode == "auto":
        return _auto_discover(cameras_cfg)
    else:
        return _manual_load(cameras_cfg)


def _manual_load(cameras_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    manual_list = cameras_cfg.get("manual", [])
    result = []
    for cam in manual_list:
        if cam.get("enabled", True):
            result.append(dict(cam))
    return result


def _auto_discover(cameras_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    discover_cfg = cameras_cfg.get("auto_discover", {})
    subnet_str = discover_cfg.get("subnet", "192.168.3.0/24")
    username = discover_cfg.get("username", "admin")
    password = discover_cfg.get("password", "")
    rtsp_paths = discover_cfg.get("rtsp_paths", COMMON_RTSP_PATHS)

    try:
        network = ipaddress.ip_network(subnet_str, strict=False)
    except ValueError:
        print(f"[Discovery] Invalid subnet: {subnet_str}")
        return []

    hosts = list(network.hosts())
    hosts.sort(key=lambda h: h.packed)
    print(f"[Discovery] Scanning {len(hosts)} hosts in {subnet_str}...")

    found = []
    for i, host in enumerate(hosts):
        ip = str(host)
        result = _scan_single_ip(ip, username, password, rtsp_paths)
        if result:
            cam_id = f"cam-{len(found)}"
            result["id"] = cam_id
            result["name"] = f"camera-{ip}"
            found.append(result)
            print(f"[Discovery] Found camera at {ip} -> {cam_id}")

    print(f"[Discovery] Scan complete: {len(found)} cameras found")
    return found


def probe_single_camera(
    ip: str,
    username: str = "admin",
    password: str = "",
    rtsp_paths: List[str] = None,
) -> Optional[Dict[str, Any]]:
    if rtsp_paths is None:
        rtsp_paths = COMMON_RTSP_PATHS
    return _scan_single_ip(ip, username, password, rtsp_paths)


class CameraDiscoveryService:
    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._cameras: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._on_change_callbacks: List = []

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._scan_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _scan_loop(self) -> None:
        cameras_cfg = self._config.get("cameras", {})
        interval = cameras_cfg.get("auto_discover", {}).get("scan_interval_seconds", 30)

        while self._running:
            new_cameras = discover_cameras(self._config)
            with self._lock:
                old_ids = {c["id"] for c in self._cameras}
                new_ids = {c["id"] for c in new_cameras}
                if old_ids != new_ids:
                    self._cameras = new_cameras
                    for cb in self._on_change_callbacks:
                        try:
                            cb(new_cameras)
                        except Exception:
                            pass
            time.sleep(interval)

    def get_cameras(self) -> List[Dict[str, Any]]:
        with self._lock:
            if not self._cameras:
                self._cameras = discover_cameras(self._config)
            return list(self._cameras)

    def add_camera(self, cam_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        ip = cam_config.get("ip", "")
        username = cam_config.get("username", "admin")
        password = cam_config.get("password", "")
        rtsp_path = cam_config.get("rtsp_path", "/Streaming/Channels/101")

        userinfo = username
        if password or password == "":
            userinfo = f"{username}:{password}"
        url = f"rtsp://{userinfo}@{ip}{rtsp_path}"

        if not _try_rtsp(url):
            return None

        with self._lock:
            existing_ids = {c["id"] for c in self._cameras}
            idx = 0
            while f"cam-{idx}" in existing_ids:
                idx += 1
            cam_id = f"cam-{idx}"

            new_cam = {
                "id": cam_id,
                "name": cam_config.get("name", f"camera-{ip}"),
                "source": url,
                "ip": ip,
                "rtsp_path": rtsp_path,
                "use_gstreamer": cam_config.get("use_gstreamer", True),
                "enabled": True,
            }
            self._cameras.append(new_cam)

            for cb in self._on_change_callbacks:
                try:
                    cb(self._cameras)
                except Exception:
                    pass

            return new_cam

    def remove_camera(self, cam_id: str) -> bool:
        with self._lock:
            for i, cam in enumerate(self._cameras):
                if cam["id"] == cam_id:
                    self._cameras.pop(i)
                    for cb in self._on_change_callbacks:
                        try:
                            cb(self._cameras)
                        except Exception:
                            pass
                    return True
        return False

    def on_change(self, callback) -> None:
        self._on_change_callbacks.append(callback)