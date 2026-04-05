#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
迷你测试脚本：只处理前 2 個日期來驗證系統
"""

import os
import sys
from multiprocessing import Process, Queue
from datetime import datetime

# 新增 multi_gpu 和 src 路徑
sys.path.insert(0, "/home/ai/Sugery_AI")
sys.path.insert(0, "/home/ai/Sugery_AI/src")
sys.path.insert(0, "/home/ai/Sugery_AI/VLM-1/lib/python3.10/site-packages")

from multi_gpu.config import (
    PROCESS_DATES,
    GPU_ALLOCATION,
    get_video_dir_for_date,
    get_csv_output_for_date,
    CURRENT_TEST,
    TARGET_CAMERAS,
    print_config,
    ROOM
)

def test_run():
    """測試前 2 個日期"""
    print("\n" + "=" * 70)
    print("🧪 迷你測試模式：處理前 2 個日期")
    print("=" * 70)
    
    print_config()
    
    # 只取前 2 個日期
    test_dates = PROCESS_DATES[:2] if len(PROCESS_DATES) >= 2 else PROCESS_DATES
    
    print(f"\n📋 測試日期: {test_dates}")
    print(f"📷 目標攝像機: {TARGET_CAMERAS}")
    print(f"🏥 手術室: {ROOM}")
    print(f"🎯 任務: {CURRENT_TEST}")
    
    # 檢查日期相關路徑
    for date_str in test_dates:
        video_dir = get_video_dir_for_date(date_str)
        csv_output = get_csv_output_for_date(date_str)
        gpu_id = GPU_ALLOCATION.get(date_str, -1)
        
        print(f"\n📅 {date_str}:")
        print(f"   - GPU id: {gpu_id}")
        print(f"   - 視訊目錄: {video_dir}")
        print(f"   - 目錄存在: {'✓' if os.path.exists(video_dir) else '✗'}")
        print(f"   - CSV 輸出: {csv_output}")
        
        # 列出視訊檔案
        if os.path.exists(video_dir):
            videos = [f for f in os.listdir(video_dir) 
                     if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
            print(f"   - 視訊數量: {len(videos)}")
            if videos:
                for i, v in enumerate(videos[:3]):
                    print(f"     [{i+1}] {v}")
                if len(videos) > 3:
                    print(f"     ... 還有 {len(videos)-3} 個")

if __name__ == "__main__":
    test_run()
