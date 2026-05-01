# config.py
import os

# 基礎路徑
BASE_DIR = "/home/cvlabgodzilla/Desktop/Sugery"

VIDEO_DIR = os.path.join(BASE_DIR, "data", "20231211")
DEBUG_DIR_NAME = os.path.basename(VIDEO_DIR)  # 自動依照測試資料夾(VIDEO_DIR)的名稱修改

VIDEO_DIRS = [
    VIDEO_DIR,
]
CSV_OUTPUT = os.path.join(BASE_DIR, "outputs", "surgery_report.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "Qwen3-VL-8B-Instruct-FP8")

# 手術室與攝影機
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

# Door: 門口推床進出
# Surgery: 手術台與病人
CURRENT_TEST = "Door"  

if CURRENT_TEST == "Door":
    required_cam_type = "Door"  # 門口機
elif CURRENT_TEST == "Surgery":
    required_cam_type = "Room"  # 房內機
else:
    required_cam_type = None

TARGET_CAMERAS = [
    cam for cam in OR_SETTING[ROOM] 
    if CAMERA_SETTING.get(cam) == required_cam_type
]

# 處理參數
STRIDE_SEC = 0.2  # 每隔多少秒抽1幀分析

# 裁切設定
CROP_SETTING = {
    "A8": (400, 0, 680, 260),    # door Single 模式裁切：右上角門口 (640x480 解析度)
    "A9": (300, 200, 640, 480),  # surgery右下區域
}
CROP_REGION = CROP_SETTING.get(ROOM, None)

# Door Video 模式用裁切範圍 (需要更實的畫面來判斷推入/推出方向)
DOOR_VIDEO_CROP_SETTING = {
    "A8": (250, 0, 680, 340) ,    # Video 模式裁切：包含門口 (放大)
    "A9": None,
}
DOOR_VIDEO_CROP = DOOR_VIDEO_CROP_SETTING.get(ROOM, None)

# Door Stage 3 Video 視窗設定
USE_DOOR_VIDEO_CROSS = True
DOOR_VIDEO_MIN_FRAMES = 40       # 8 秒 (@5fps)
DOOR_VIDEO_MAX_FRAMES = 55
DOOR_VIDEO_TEMPORAL_STRIDE = 5

# 即時 Pipeline 參數
# Surgery 模式: 需要 900 幀 (3分鐘 @5fps) 穩定才確認手術開始
# Door 模式: 需要 25 幀 (5秒 @5fps) 穩定才確認推床過門檻
HALF_WINDOW = 10 if CURRENT_TEST == "Door" else 25  # 投票視窗
STABLE_FRAME = 900  # ENT 穩定期需要的幀數
MAX_GAP_FRAME = 50  # 穩定期允許的最大遮擋幀數
SEND_CONFIRM_THRESHOLD = 900  # SEND 確認觀察窗大小

EVENT_COOLDOWN_FRAMES = { # 事件冷卻時間 (幀數)
    # ENT(推入,2) 之後 → 下一個 SEND(推出) 至少等 15 分鐘
    "Door_ENT":  4500,   # 15 分鐘 @5fps
    # SEND(推出,3) 之後 → 下一個 ENT(推入) 至少等 5 分鐘
    "Door_SEND": 1500,   # 5 分鐘 @5fps
    "Surgery":   4500,   # 15 分鐘 @5fps
}

# 顯示設定
SHOW_WINDOW = False  # 是否開啟視窗顯示 (終端/SSH 環境請設 False)
VIS_HEIGHT = 480     
VIS_WIDTH = 640
