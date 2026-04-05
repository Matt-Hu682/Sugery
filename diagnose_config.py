#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
診斷腳本：檢查配置和日期檢測
"""
import sys
import os
sys.path.insert(0, "/home/ai/Sugery_AI")

print("\n" + "=" * 70)
print("🔍 診斷多 GPU 配置")
print("=" * 70)

from multi_gpu.config import (
    DATA_BASE_DIR,
    PROCESS_DATES,
    GPU_ALLOCATION,
    GPU_IDS,
    OUTPUTS_DIR,
    auto_detect_dates
)

# 1. 檢查數據目錄
print(f"\n📁 1. 檢查數據目錄")
print(f"   DATA_BASE_DIR: {DATA_BASE_DIR}")
print(f"   存在: {'✓' if os.path.exists(DATA_BASE_DIR) else '✗'}")
print(f"   可讀: {'✓' if os.access(DATA_BASE_DIR, os.R_OK) else '✗'}")

if os.path.exists(DATA_BASE_DIR):
    contents = os.listdir(DATA_BASE_DIR)
    print(f"   目錄內容數: {len(contents)}")
    for item in sorted(contents)[:5]:
        print(f"     - {item}")
    if len(contents) > 5:
        print(f"     ... 還有 {len(contents)-5} 個")

# 2. 檢查自動檢測
print(f"\n📊 2. 日期自動檢測")
detected = auto_detect_dates()
print(f"   檢測到日期數: {len(detected)}")
if detected:
    for date in detected[:5]:
        print(f"     - {date}")
    if len(detected) > 5:
        print(f"     ... 還有 {len(detected)-5} 個")
else:
    print(f"   ⚠️  未檢測到任何日期！")

# 3. 檢查 PROCESS_DATES
print(f"\n📋 3. PROCESS_DATES 配置")
print(f"   日期數: {len(PROCESS_DATES)}")
if PROCESS_DATES:
    for date in PROCESS_DATES[:5]:
        print(f"     - {date}")
    if len(PROCESS_DATES) > 5:
        print(f"     ... 還有 {len(PROCESS_DATES)-5} 個")

# 4. 檢查 GPU 配置
print(f"\n🖥️  4. GPU 配置")
print(f"   GPU_IDS: {GPU_IDS}")
print(f"   類型: {type(GPU_IDS[0])}")

# 5. 檢查 GPU_ALLOCATION
print(f"\n⚙️  5. GPU 分配方案")
print(f"   總分配數: {len(GPU_ALLOCATION)}")
if GPU_ALLOCATION:
    allocation_summary = {}
    for date, gpu_id in GPU_ALLOCATION.items():
        if gpu_id not in allocation_summary:
            allocation_summary[gpu_id] = []
        allocation_summary[gpu_id].append(date)
    
    for gpu_id, dates in allocation_summary.items():
        print(f"\n   {gpu_id}:")
        print(f"     日期數: {len(dates)}")
        for date in dates[:3]:
            print(f"       - {date}")
        if len(dates) > 3:
            print(f"       ... 還有 {len(dates)-3} 個")
else:
    print(f"   ⚠️  GPU_ALLOCATION 為空！")

# 6. 檢查輸出目錄
print(f"\n💾 6. 輸出目錄")
print(f"   OUTPUTS_DIR: {OUTPUTS_DIR}")
print(f"   存在: {'✓' if os.path.exists(OUTPUTS_DIR) else '✗'}")

print("\n" + "=" * 70)
print("✅ 診斷完成")
print("=" * 70)
