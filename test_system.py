# test_system.py
# 測試系統完整性和功能

import subprocess
import sys
import os

def test_ffmpeg():
    """測試 ffmpeg 是否安裝"""
    print("\n📺 測試 ffmpeg...")
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE, 
                              timeout=5)
        if result.returncode == 0:
            version_line = result.stdout.decode().split('\n')[0]
            print(f"   ✅ {version_line}")
            return True
        else:
            print("   ❌ ffmpeg 無法執行")
            return False
    except FileNotFoundError:
        print("   ❌ ffmpeg 未安裝")
        return False
    except Exception as e:
        print(f"   ❌ 錯誤: {e}")
        return False


def test_config():
    """測試配置載入"""
    print("\n⚙️  測試配置載入...")
    try:
        sys.path.insert(0, 'src')
        from multi_gpu.config import (
            _detected_dates, PROCESS_DATES, TEST_DATE,
            CURRENT_TEST, ROOM, TARGET_CAMERAS,
            DATA_BASE_DIR, OUTPUTS_DIR,
            CROP_REGION, MOTION_DIFF_THRESH, STABLE_FRAME,
            get_csv_output_for_date, get_gpu_for_date
        )
        print(f"   ✅ 配置載入成功")
        print(f"      日期數: {len(_detected_dates)}")
        print(f"      任務: {CURRENT_TEST} | 手術室: {ROOM}")
        print(f"      資料源: {DATA_BASE_DIR}")
        return True
    except Exception as e:
        print(f"   ❌ 配置載入失敗: {e}")
        return False


def test_modules():
    """測試核心模組"""
    print("\n📦 測試核心模組...")
    modules = [
        ('torch', 'PyTorch'),
        ('cv2', 'OpenCV'),
        ('PIL', 'Pillow'),
    ]
    
    all_ok = True
    for module_name, display_name in modules:
        try:
            __import__(module_name)
            print(f"   ✅ {display_name}")
        except ImportError:
            print(f"   ❌ {display_name} 未安裝")
            all_ok = False
    
    try:
        sys.path.insert(0, 'src')
        from realtime_pipeline import RealtimePipeline
        print(f"   ✅ RealtimePipeline")
    except Exception as e:
        print(f"   ❌ RealtimePipeline 載入失敗: {e}")
        all_ok = False
    
    return all_ok


def test_processor():
    """測試處理器載入"""
    print("\n🔧 測試處理器...")
    try:
        sys.path.insert(0, 'src')
        from multi_gpu.processor import process_dates_on_gpus, process_events_and_clip_videos
        print(f"   ✅ 處理器載入成功")
        return True
    except Exception as e:
        print(f"   ❌ 處理器載入失敗: {e}")
        return False


def test_gpu():
    """測試 GPU 可用性"""
    print("\n🎮 測試 GPU...")
    try:
        import torch
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            print(f"   ✅ {count} 張 GPU 可用")
            for i in range(count):
                print(f"      GPU{i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            print("   ⚠️  CUDA 不可用（可能使用 CPU）")
            return False
    except Exception as e:
        print(f"   ❌ GPU 檢測失敗: {e}")
        return False


def test_video_sample():
    """測試影片緩衝操作"""
    print("\n🎬 測試影片操作...")
    try:
        import cv2
        import numpy as np
        
        # 建立測試影片（虛擬）
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        test_video = "/tmp/test_video.mp4"
        
        out = cv2.VideoWriter(test_video, cv2.VideoWriter_fourcc(*'mp4v'), 
                            30, (640, 480))
        for _ in range(30):
            out.write(frame)
        out.release()
        
        # 測試 ffmpeg 剪輯
        test_output = "/tmp/test_clip.mp4"
        cmd = [
            'ffmpeg', '-y',
            '-ss', '0',
            '-i', test_video,
            '-t', '1',
            '-c', 'copy',
            test_output
        ]
        
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, 
                              stderr=subprocess.DEVNULL, timeout=10)
        
        if result.returncode == 0:
            print(f"   ✅ 影片剪輯測試成功")
            os.remove(test_video)
            os.remove(test_output)
            return True
        else:
            print(f"   ❌ 影片剪輯測試失敗")
            return False
    except Exception as e:
        print(f"   ❌ 影片操作測試失敗: {e}")
        return False


def main():
    print("=" * 70)
    print("🔍 系統完整性測試 - 方案 B（CSV + 事件檢測 + 影片剪輯）")
    print("=" * 70)
    
    tests = [
        ("ffmpeg", test_ffmpeg),
        ("配置", test_config),
        ("模組", test_modules),
        ("處理器", test_processor),
        ("GPU", test_gpu),
        ("影片操作", test_video_sample),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ 測試 {name} 發生異常: {e}")
            results.append((name, False))
    
    # 匯總
    print("\n" + "=" * 70)
    print("📊 測試結果匯總")
    print("=" * 70)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅" if result else "❌"
        print(f"{status} {name}")
    
    print(f"\n總計: {passed}/{total} 通過")
    
    if passed == total:
        print("\n🎉 所有測試通過！系統已準備好執行方案 B")
        print("執行: python3 -m multi_gpu.processor")
        return 0
    else:
        print(f"\n⚠️  有 {total - passed} 項測試失敗，請檢查上述問題")
        return 1


if __name__ == "__main__":
    sys.exit(main())
