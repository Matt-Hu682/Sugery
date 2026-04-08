# multi_gpu 設定檔案 (batch_config)
# 負責批次多日期的專用路徑、硬體與分配設定，並繼承通用設定
import os
import sys
from pathlib import Path

# 將 src 加入 Python Path 以便引入共用的 config
batch_runner_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(batch_runner_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# 匯入所有演算法與任務設定 (ROOM, CURRENT_TEST, STRIDE_SEC, HALF_WINDOW 等等)
from config import *

# ============ 批次專用路徑 ============
DATA_BASE_DIR = "/home/ai/nas/113-Student/F113151105/手術室/data_video/mask_video_202312"
# 確保 NAS 輸出目錄
NAS_OUTPUT_BASE = "/home/ai/nas/113-Student/F113151105/手術室/Sugery"
OUTPUTS_DIR = os.path.join(NAS_OUTPUT_BASE, "outputs")  # CSV 結果

# ============ GPU 分配設定 ============
# 手動指定使用哪 2 個 GPU (MIG UUID 設備 ID)
# 查詢 UUID: nvidia-smi -L
GPU_IDS = [
    "MIG-ab8e204e-b0aa-56e2-b0cf-6438e473bf86",  # Device 0
    "MIG-def3ca53-d2a7-51a8-890e-7bededa36e64"   # Device 1
]

def auto_detect_dates():
    """自動檢測 data 目錄中的所有日期資料夾"""
    dates = []
    if os.path.exists(DATA_BASE_DIR):
        for item in sorted(os.listdir(DATA_BASE_DIR)):
            item_path = os.path.join(DATA_BASE_DIR, item)
            if os.path.isdir(item_path):
                dates.append(item)
    return dates

# 自動檢測日期
_detected_dates = auto_detect_dates()

if _detected_dates:
    TEST_DATE = None
    PROCESS_DATES = _detected_dates
    
    # 生成 GPU 分配方案（前一半給第1個GPU，後一半給第2個GPU）
    GPU_ALLOCATION = {}
    mid_point = len(PROCESS_DATES) // 2
    for idx, date in enumerate(PROCESS_DATES):
        if idx < mid_point:
            GPU_ALLOCATION[date] = GPU_IDS[0]  # 前一半給第1個GPU
        else:
            GPU_ALLOCATION[date] = GPU_IDS[1]  # 後一半給第2個GPU
else:
    TEST_DATE = None
    PROCESS_DATES = []
    GPU_ALLOCATION = {}


def get_video_dir_for_date(date_str):
    """獲取指定日期的視訊目錄"""
    return os.path.join(DATA_BASE_DIR, date_str)

def get_csv_output_for_date(date_str):
    """生成指定日期的輸出CSV路徑 - 按日期分開"""
    # 最終結果: surgery_report_20231201_Surgery_20260405_HHMMSS.csv
    return os.path.join(OUTPUTS_DIR, f"surgery_report_{date_str}.csv")

def get_gpu_for_date(date_str):
    """獲取指定日期對應的GPU ID"""
    return GPU_ALLOCATION.get(date_str, 0)

def print_config():
    """列印目前設定"""
    print("\n" + "=" * 70)
    print("🔧 Multi-GPU 多日期處理設定 (繼承自 src/config.py)")
    print("=" * 70)
    print(f"\n資料目錄: {DATA_BASE_DIR}")
    print(f"輸出目錄: {OUTPUTS_DIR}")
    print(f"模型路徑: {MODEL_PATH}")
    print(f"\n任務設定: {CURRENT_TEST}")
    print(f"手術室: {ROOM}")
    print(f"目標攝像機: {TARGET_CAMERAS}")
    print(f"步長: {STRIDE_SEC}s")
    print(f"即時視窗: {HALF_WINDOW * 2 + 1} 幀")
    
    if TEST_DATE:
        print(f"\n📌 測試集: {TEST_DATE}")
    
    if GPU_ALLOCATION:
        print(f"\n📊 GPU分配方案:")
        for date in PROCESS_DATES:
            gpu_id = GPU_ALLOCATION[date]
            print(f"  GPU{gpu_id}: {date}")
    else:
        print("\n⚠️  未檢測到任何日期資料夾")
    
    print("\n" + "=" * 70)

