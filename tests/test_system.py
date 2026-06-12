# test_system.py
# 測試目前 main_parallel 主流程的基本環境完整性

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"


def test_ffmpeg():
    """測試 ffmpeg 是否安裝。"""
    print("\n測試 ffmpeg...")
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5,
        )
        if result.returncode == 0:
            version_line = result.stdout.decode(errors="ignore").split("\n")[0]
            print(f"   OK {version_line}")
            return True
        print("   NO ffmpeg 無法執行")
        return False
    except FileNotFoundError:
        print("   NO ffmpeg 未安裝")
        return False
    except Exception as exc:
        print(f"   NO 錯誤: {exc}")
        return False


def test_config():
    """測試目前主流程設定載入。"""
    print("\n測試 config.py...")
    try:
        sys.path.insert(0, str(SRC_DIR))
        from config import (  # noqa: E402
            CSV_OUTPUT,
            OUTPUT_BASE_DIR,
            ROOM,
            STRIDE_SEC,
            TARGET_CAMERAS,
            TEST_VIDEO_BASE,
            VIDEO_DIRS,
        )
        print("   OK config 載入成功")
        print(f"      ROOM: {ROOM}")
        print(f"      TARGET_CAMERAS: {TARGET_CAMERAS}")
        print(f"      STRIDE_SEC: {STRIDE_SEC}")
        print(f"      TEST_VIDEO_BASE: {TEST_VIDEO_BASE}")
        print(f"      VIDEO_DIRS: {len(VIDEO_DIRS)}")
        print(f"      OUTPUT_BASE_DIR: {OUTPUT_BASE_DIR}")
        print(f"      CSV_OUTPUT: {CSV_OUTPUT}")
        return True
    except Exception as exc:
        print(f"   NO config 載入失敗: {exc}")
        return False


def test_modules():
    """測試核心模組可載入。"""
    print("\n測試核心模組...")
    modules = [
        ("cv2", "OpenCV"),
        ("PIL", "Pillow"),
    ]

    all_ok = True
    for module_name, display_name in modules:
        try:
            __import__(module_name)
            print(f"   OK {display_name}")
        except ImportError:
            print(f"   NO {display_name} 未安裝")
            all_ok = False

    try:
        sys.path.insert(0, str(SRC_DIR))
        from realtime_pipeline import RealtimePipeline  # noqa: F401,E402
        from door_stage1 import DoorStage1  # noqa: F401,E402
        print("   OK RealtimePipeline")
        print("   OK DoorStage1")
    except Exception as exc:
        print(f"   NO 專案模組載入失敗: {exc}")
        all_ok = False

    return all_ok


def test_main_entrypoint():
    """確認目前主入口存在，避免 import 時載入大型模型。"""
    print("\n測試主入口...")
    entrypoint = SRC_DIR / "main_parallel.py"
    if not entrypoint.exists():
        print(f"   NO 找不到 {entrypoint}")
        return False
    text = entrypoint.read_text(encoding="utf-8")
    ok = "def main():" in text and "if __name__ ==" in text
    print(f"   {'OK' if ok else 'NO'} {entrypoint}")
    return ok


def test_video_sample():
    """測試影片寫入與 ffmpeg 剪輯。"""
    print("\n測試影片操作...")
    try:
        import cv2
        import numpy as np

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        test_video = "/tmp/test_video.mp4"
        test_output = "/tmp/test_clip.mp4"

        out = cv2.VideoWriter(test_video, cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 480))
        for _ in range(30):
            out.write(frame)
        out.release()

        cmd = ["ffmpeg", "-y", "-ss", "0", "-i", test_video, "-t", "1", "-c", "copy", test_output]
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)

        if result.returncode == 0:
            print("   OK 影片剪輯測試成功")
            for path in (test_video, test_output):
                if os.path.exists(path):
                    os.remove(path)
            return True
        print("   NO 影片剪輯測試失敗")
        return False
    except Exception as exc:
        print(f"   NO 影片操作測試失敗: {exc}")
        return False


def main():
    print("=" * 70)
    print("系統完整性測試 - main_parallel 主流程")
    print("=" * 70)

    tests = [
        ("ffmpeg", test_ffmpeg),
        ("配置", test_config),
        ("模組", test_modules),
        ("主入口", test_main_entrypoint),
        ("影片操作", test_video_sample),
    ]

    results = []
    for name, test_func in tests:
        try:
            results.append((name, test_func()))
        except Exception as exc:
            print(f"\nNO 測試 {name} 發生異常: {exc}")
            results.append((name, False))

    print("\n" + "=" * 70)
    print("測試結果匯總")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)
    for name, result in results:
        print(f"{'OK' if result else 'NO'} {name}")

    print(f"\n總計: {passed}/{total} 通過")
    if passed == total:
        print("\n所有基本檢查通過。主流程指令: PYTHONPATH=src python3 src/main_parallel.py")
        return 0
    print(f"\n有 {total - passed} 項測試失敗，請檢查上述問題")
    return 1


if __name__ == "__main__":
    sys.exit(main())
