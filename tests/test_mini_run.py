#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
目前主流程的迷你設定檢查。

舊版 batch runner 已移除；這個腳本只檢查 main_parallel.py
會使用到的資料集、攝影機與輸出設定。
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import (  # noqa: E402
    CAMERA_SETTING,
    CSV_OUTPUT,
    OUTPUT_BASE_DIR,
    ROOM,
    TARGET_CAMERAS,
    TARGET_DATASETS,
    TEST_VIDEO_BASE,
    VIDEO_DIRS,
)


def test_run():
    """列出目前 main_parallel.py 會使用的基本設定。"""
    print("\n" + "=" * 70)
    print("main_parallel 設定檢查")
    print("=" * 70)
    print(f"ROOM: {ROOM}")
    print(f"TARGET_CAMERAS: {TARGET_CAMERAS}")
    print(f"CAMERA_SETTING: {CAMERA_SETTING}")
    print(f"TARGET_DATASETS: {TARGET_DATASETS}")
    print(f"TEST_VIDEO_BASE: {TEST_VIDEO_BASE}")
    print(f"OUTPUT_BASE_DIR: {OUTPUT_BASE_DIR}")
    print(f"CSV_OUTPUT: {CSV_OUTPUT}")

    print("\nVIDEO_DIRS:")
    for path in VIDEO_DIRS:
        exists = os.path.isdir(path)
        print(f"  {'OK' if exists else 'NO'} {path}")
        if exists:
            videos = [
                name for name in os.listdir(path)
                if name.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
            ]
            print(f"     videos: {len(videos)}")


if __name__ == "__main__":
    test_run()
