#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU0 專用處理腳本（用於 tmux 上窗格）
"""
import sys
sys.path.insert(0, "/home/ai/Sugery_AI")
sys.path.insert(0, "/home/ai/Sugery_AI/src")
sys.path.insert(0, "/home/ai/Sugery_AI/VLM-1/lib/python3.10/site-packages")

from batch_runner.processor import process_dates_for_gpu
from batch_runner.config import GPU_ALLOCATION, PROCESS_DATES, GPU_IDS
from multiprocessing import Queue

# 使用第一個 GPU
gpu0_id = GPU_IDS[0]
gpu0_dates = [d for d in PROCESS_DATES if GPU_ALLOCATION.get(d) == gpu0_id]
queue = Queue()

print("\n" + "=" * 70)
print("🚀 GPU0 獨立行程啟動")
print(f"   GPU ID: {gpu0_id}")
print(f"   分配日期: {len(gpu0_dates)} 個")
print(f"   日期範圍: {gpu0_dates[0]} ~ {gpu0_dates[-1] if gpu0_dates else 'N/A'}")
print("=" * 70 + "\n")

if gpu0_dates:
    process_dates_for_gpu(gpu0_dates, gpu0_id, queue)
else:
    print("⚠️  沒有分配到任何日期！")

