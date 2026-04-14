# config.py
# 全域基礎設定檔：定義模型、手術室配置、算法閾值等
import os

# ============ 基礎路徑 ============
BASE_DIR = "/home/cvlabgodzilla/Desktop/Sugery"

# 單一影片測試用的路徑 (供 scripts/run_single_test.py 類使用)
VIDEO_DIR = os.path.join(BASE_DIR, "data", "20240103")
CSV_OUTPUT = os.path.join(BASE_DIR, "outputs", "surgery_report.csv")

MODEL_PATH = os.path.join(BASE_DIR, "models", "Qwen3-VL-8B-Instruct-FP8")

# ============ 手術室與攝影機配置 ============
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

# ============ 任務設定 ============
# Door: 門口推床進出
# Surgery: 手術台是否進行手術
# Patient: 病患狀態
CURRENT_TEST = "Door"  

if CURRENT_TEST == "Door":
    required_cam_type = "Door"  # 門口機
elif CURRENT_TEST in ["Surgery", "Patient"]:
    required_cam_type = "Room"  # 房內機
else:
    required_cam_type = None

# 自動從該手術室 (ROOM) 的名單中，挑選出符合視角的攝影機
TARGET_CAMERAS = [
    cam for cam in OR_SETTING[ROOM] 
    if CAMERA_SETTING.get(cam) == required_cam_type
]

# ============ 處理參數 ============
STRIDE_SEC = 0.2  # 每隔多少秒抽1幀分析

# 裁切設定 (A9 需要裁切，A8 不需要) 
# 格式: (x1, y1, x2, y2) 或 None
CROP_SETTING = {
    "A8": (400, 0, 680, 260),    # Single 模式裁切：右上角門口 (640x480 解析度)
    "A9": (300, 200, 640, 480),  # 右下區域
}
CROP_REGION = CROP_SETTING.get(ROOM, None)

# Door Video 模式用裁切範圍 (需要更實的畫面來判斷推入/推出方向)
DOOR_VIDEO_CROP_SETTING = {
    "A8": (180, 0, 680, 340),    # Video 模式裁切：包含門口 + 左邊交化區
    "A9": None,
}
DOOR_VIDEO_CROP = DOOR_VIDEO_CROP_SETTING.get(ROOM, None)



# ============ 即時 Pipeline 參數 (狀態機參數) ============
# Surgery 模式: 需要 900 幀 (3分鐘 @5fps) 穩定才確認手術開始
# Door 模式: 需要 25 幀 (5秒 @5fps) 穩定才確認推床過門檻
HALF_WINDOW = 10 if CURRENT_TEST == "Door" else 25  # 投票視窗
STABLE_FRAME = 900  # ENT 穩定期需要的幀數
MAX_GAP_FRAME = 50  # 穩定期允許的最大遮擋幀數
SEND_CONFIRM_THRESHOLD = 900  # SEND 確認觀察窗大小

# 每次確認事件後，下一次 SEND 的最短間隔 (幀數, 5fps)
# 4500 幀 = 15 分鐘 @5fps
EVENT_COOLDOWN_FRAMES = {
    "Door":    4500,   # 15 分鐘 @5fps
    "Surgery": 4500,   # 15 分鐘 @5fps
}

# ============ 顯示設定 ============
SHOW_WINDOW = False  # 是否開啟視窗顯示 (終端/SSH 環境請設 False)
VIS_HEIGHT = 480     
VIS_WIDTH = 640