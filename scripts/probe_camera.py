#!/usr/bin/env python3。
import argparse
import base64
import urllib.error
import urllib.request
from typing import List

import cv2


COMMON_RTSP_PATHS: List[str] = [
    "/Streaming/Channels/101",
    "/Streaming/Channels/1",
    "/cam/realmonitor?channel=1&subtype=0",
    "/h264/ch1/main/av_stream",
    "/live/ch00_0",
    "/live",
    "/stream1",
]


def fetch_http(url: str, username: str, password: str) -> None:
    req = urllib.request.Request(url)
    if username or password:
        token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
        req.add_header("Authorization", f"Basic {token}")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            print(f"[HTTP] {url} -> {resp.status} {resp.reason}")
            server = resp.headers.get("Server", "")
            if server:
                print(f"[HTTP] Server: {server}")
    except urllib.error.HTTPError as exc:
        print(f"[HTTP] {url} -> HTTP {exc.code} {exc.reason}")
    except Exception as exc:
        print(f"[HTTP] {url} -> FAIL {exc}")


def try_rtsp(url: str) -> bool:
    cap = cv2.VideoCapture(url)
    ok, frame = cap.read()
    cap.release()
    if ok and frame is not None:
        print(f"[RTSP] OK {url} frame={frame.shape[1]}x{frame.shape[0]}")
        return True
    print(f"[RTSP] FAIL {url}")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe an IP camera via HTTP and RTSP")
    parser.add_argument("--ip", required=True, help="camera ip, e.g. 192.168.1.10")
    parser.add_argument("--user", default="admin", help="camera username")
    parser.add_argument("--password", default="", help="camera password, empty allowed")
    parser.add_argument("--http-only", action="store_true", help="only probe the web endpoint")
    args = parser.parse_args()

    fetch_http(f"http://{args.ip}", args.user, args.password)
    if args.http_only:
        return

    for path in COMMON_RTSP_PATHS:
        userinfo = args.user
        if args.password or args.password == "":
            userinfo = f"{args.user}:{args.password}"
        url = f"rtsp://{userinfo}@{args.ip}{path}"
        if try_rtsp(url):
            break


if __name__ == "__main__":
    main()
