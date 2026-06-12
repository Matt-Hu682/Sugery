#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""診斷目前 main_parallel.py 使用的設定。"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import (  # noqa: E402
    CAMERA_SETTING,
    CSV_OUTPUT,
    MODEL_PATH,
    OUTPUT_BASE_DIR,
    ROOM,
    STRIDE_SEC,
    TARGET_CAMERAS,
    TARGET_DATASETS,
    TEST_VIDEO_BASE,
    VIDEO_DIRS,
)

print("\n" + "=" * 70)
print("main_parallel 設定診斷")
print("=" * 70)

print("\n1. 基本設定")
print(f"   ROOM: {ROOM}")
print(f"   TARGET_CAMERAS: {TARGET_CAMERAS}")
print(f"   CAMERA_SETTING: {CAMERA_SETTING}")
print(f"   STRIDE_SEC: {STRIDE_SEC}")
print(f"   TARGET_DATASETS: {TARGET_DATASETS}")

print("\n2. 路徑")
for label, path in [
    ("TEST_VIDEO_BASE", TEST_VIDEO_BASE),
    ("OUTPUT_BASE_DIR", OUTPUT_BASE_DIR),
    ("CSV_OUTPUT", CSV_OUTPUT),
    ("MODEL_PATH", MODEL_PATH),
]:
    print(f"   {label}: {path}")
    print(f"      exists: {'yes' if os.path.exists(path) else 'no'}")

print("\n3. VIDEO_DIRS")
print(f"   count: {len(VIDEO_DIRS)}")
for path in VIDEO_DIRS[:10]:
    exists = os.path.isdir(path)
    print(f"   {'OK' if exists else 'NO'} {path}")
    if exists:
        videos = [
            name for name in os.listdir(path)
            if name.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
        ]
        print(f"      videos: {len(videos)}")
if len(VIDEO_DIRS) > 10:
    print(f"   ... 還有 {len(VIDEO_DIRS) - 10} 個")

print("\n" + "=" * 70)
print("診斷完成")
print("=" * 70)
