# config.py
import os
from pathlib import Path 

BASE_DIR = "/home/cvlabgodzilla/Desktop/Sugery_AI"
#data/20240909_S01-S02
VIDEO_DIR = os.path.join(BASE_DIR, "data", "20240805輪椅")
CSV_OUTPUT = os.path.join(BASE_DIR, "outputs", "surgery_report.csv")
MODEL_PATH = "/home/cvlabgodzilla/Desktop/Sugery_AI/models/Qwen3-VL-8B-Instruct-FP8"

ROOM = "A8"

OR_SETTING = {
    "A8": ["S01", "S02"],
    "A9": ["S03", "S04"]
}


CAMERA_SETTING = {
    "S01": "Door",
    "S02": "Room",
    "S03": "Door",
    "S04": "Room"
}

TARGET_CAMERAS = OR_SETTING[ROOM]

# Door
# Surgery
# Patient
CURRENT_TEST = "Door"
if CURRENT_TEST == "Door":
    required_cam_type = "Door"  # 門口機
elif CURRENT_TEST in ["Surgery", "Patient"]:
    required_cam_type = "Room"  # 房內機
else:
    required_cam_type = None

# 2. 自動從該手術室 (ROOM) 的名單中，挑選出符合視角的攝影機
TARGET_CAMERAS = [
    cam for cam in OR_SETTING[ROOM] 
    if CAMERA_SETTING.get(cam) == required_cam_type
]


# --- 參數設定 ---
STRIDE_SEC = 0.2 # 每隔多少秒抽1幀分析

# --- 顯示設定 ---
SHOW_WINDOW = False
VIS_HEIGHT = 480     
VIS_WIDTH = 640