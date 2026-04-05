#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU1 專用處理腳本（用於 tmux 下窗格）
"""
import sys
sys.path.insert(0, "/home/ai/Sugery_AI")
sys.path.insert(0, "/home/ai/Sugery_AI/src")
sys.path.insert(0, "/home/ai/Sugery_AI/VLM-1/lib/python3.10/site-packages")

from batch_runner.processor import process_dates_for_gpu
from batch_runner.config import GPU_ALLOCATION, PROCESS_DATES, GPU_IDS
from multiprocessing import Queue

# 使用第二個 GPU
gpu1_id = GPU_IDS[1]
gpu1_dates = [d for d in PROCESS_DATES if GPU_ALLOCATION.get(d) == gpu1_id]
queue = Queue()

print("\n" + "=" * 70)
print("🚀 GPU1 獨立行程啟動")
print(f"   GPU ID: {gpu1_id}")
print(f"   分配日期: {len(gpu1_dates)} 個")
print(f"   日期範圍: {gpu1_dates[0]} ~ {gpu1_dates[-1] if gpu1_dates else 'N/A'}")
print("=" * 70 + "\n")

if gpu1_dates:
    process_dates_for_gpu(gpu1_dates, gpu1_id, queue)
else:
    print("⚠️  沒有分配到任何日期！")

