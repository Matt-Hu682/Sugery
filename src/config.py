# config.py
import os

# 路徑設定
BASE_DIR = os.environ.get("SURGERY_BASE_DIR", "/home/cvlabgodzilla/Desktop/Sugery")
OUTPUT_BASE_DIR = os.environ.get(
    "SURGERY_OUTPUT_BASE_DIR",
    "/home/cvlabgodzilla/Desktop/908_nas_2/113-Student/F113151105/手術室/Sugery/test-result",
)
# 即時監控頻繁覆寫圖片/JSON，預設放本機避免 NAS I/O 尖峰。
LIVE_MONITOR_BASE_DIR = os.environ.get(
    "SURGERY_LIVE_MONITOR_BASE_DIR",
    os.path.join(BASE_DIR, "live"),
)
TEST_VIDEO_BASE = os.environ.get(
    "SURGERY_TEST_VIDEO_BASE",
    #"/home/cvlabgodzilla/Desktop/908_nas_2/113-Student/F113151105/手術室/data_video/mask_video_202311",
    "/home/cvlabgodzilla/Desktop/Sugery/data/mask_video_202408"
)
# 指定資料集資料夾名稱；設為 None 時跑 TEST_VIDEO_BASE 下全部資料集
# 可填單一字串: "20231228"，或多個: ["20231228", "20231226"]
TARGET_DATASETS = ["20240803", "20240804", "20240805", "20240806", "20240807", "20240808", "20240813", "20240812"]

if TARGET_DATASETS is None:
    VIDEO_DIRS = sorted([
        os.path.join(TEST_VIDEO_BASE, d)
        for d in os.listdir(TEST_VIDEO_BASE)
        if os.path.isdir(os.path.join(TEST_VIDEO_BASE, d))
    ])
elif isinstance(TARGET_DATASETS, str):
    VIDEO_DIRS = [os.path.join(TEST_VIDEO_BASE, TARGET_DATASETS)]
else:
    VIDEO_DIRS = [os.path.join(TEST_VIDEO_BASE, d) for d in TARGET_DATASETS]
VIDEO_DIR = VIDEO_DIRS[0] if VIDEO_DIRS else TEST_VIDEO_BASE
CSV_OUTPUT = os.path.join(OUTPUT_BASE_DIR, "surgery_report.csv") # 
MODEL_PATH = os.path.join(BASE_DIR, "models", "Qwen3-VL-8B-Instruct-FP8")

# 任務與攝影機設定
# Door: 門口推床進出
# Surgery: 手術台與病人
CURRENT_TEST = "Door"
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

# Door Stage 1 OpenCV 開門判斷設定
# 在 CROP_REGION 裁切後的門口畫面上做 ROI 加權差異。
DOOR_STAGE1_USE_OPENCV = True
DOOR_STAGE1_CLOSED_TEMPLATE = os.path.join(BASE_DIR, "templates", f"door_closed_{ROOM}.jpg")
DOOR_STAGE1_DIFF_THRESHOLD = 25.0       # 加權平均 absdiff 分數門檻
DOOR_STAGE1_CHANGED_RATIO = 0.04    # 變動像素比例門檻（略提高，降低局部遮擋誤判）
DOOR_STAGE1_PIXEL_DIFF_THRESHOLD = 25   # 單像素視為變動的 absdiff 門檻
DOOR_STAGE1_BG_ALPHA = 0.03             # 門未開時背景 EMA 更新速度
DOOR_STAGE1_ROI_WEIGHTS = {
    # ROI 座標為 CROP_REGION 裁切後畫面的相對比例: (x1, y1, x2, y2, weight)
    # A8：三段 ROI（上/中/下），其中中間固定為 x=0~170, y=50~100（裁切圖座標）。
    # 裁切後畫面尺寸約 280x260px (x:400~680, y:0~260)
    "A8": [
        (0.183, 0.096, 0.567, 0.346, 4.0),  # x=44~136, y=25~90
        (0.204, 0.365, 0.646, 0.588, 2.0),  # x=49~155, y=95~153
        (0.188, 0.642, 0.662, 0.873, 0.5),  # x=45~159, y=167~227
    ],
    "A9": [
        (0.00, 0.00, 1.00, 1.00, 1.0),
    ],
}
DOOR_STAGE1_ACTIVE_ROIS = DOOR_STAGE1_ROI_WEIGHTS.get(ROOM, [(0.0, 0.0, 1.0, 1.0, 1.0)])

# Door Video 模式用裁切範圍 (需要更真實的畫面來判斷推入/推出方向)
DOOR_VIDEO_CROP_SETTING = {
    "A8": (250, 0, 680, 340),    # Video 模式裁切：包含門口 (放大)
    "A9": None,
}
DOOR_VIDEO_CROP = DOOR_VIDEO_CROP_SETTING.get(ROOM, None)

# Door Stage 3 Video 視窗設定
DOOR_VIDEO_MIN_FRAMES = 40       # 8 秒 (@5fps)
DOOR_VIDEO_MAX_FRAMES = 55
DOOR_VIDEO_TEMPORAL_STRIDE = 5


# 即時 Pipeline 參數
# Surgery 模式: 需要 900 幀 (3分鐘 @5fps) 穩定才確認手術開始
# Door 模式: 需要 25 幀 (5秒 @5fps) 穩定才確認推床過門檻
HALF_WINDOW = 10 if CURRENT_TEST == "Door" else 25  # 投票視窗
STABLE_FRAME = 900  # ENT 穩定期需要的幀數

EVENT_COOLDOWN_FRAMES = { # 事件冷卻時間 (幀數)
    # ENT(推入,2) 之後 → 下一個 SEND(推出) 至少等 15 分鐘
    "Door_ENT":  4500,   # 15 分鐘 @5fps
    # SEND(推出,3) 之後 → 下一個 ENT(推入) 至少等 5 分鐘
    "Door_SEND": 300,   # 1 分鐘 @5fps
    "Surgery":   4500,   # 15 分鐘 @5fps
}

# 門口 Debounce 確認幀數（Stage 1 純 OpenCV 判斷）
# 連續 N 幀都偄測到開門，才確認狀態轉换（防抖動誤判）
DOOR_OPEN_CONFIRM_FRAMES  = 20  # 門開確認需經過的幀數
DOOR_CLOSE_CONFIRM_FRAMES = 15  # 門關確認需經過的幀數
DOOR_EVENT_MATCH_BEFORE_FRAMES = 300  # ENT/SEND 事件前 N 幀內有 Door OPEN 即成立
DOOR_EVENT_MATCH_AFTER_FRAMES = 300   # ENT/SEND 事件後 N 幀內有 Door OPEN 即成立
DOOR_LOOKBACK_FRAMES = DOOR_EVENT_MATCH_BEFORE_FRAMES + 1000  # Door 狀態保留幀數（含事件確認延遲）

# Surgery SEND 穩定性判斷參數
# 在觀察窗內（send_confirm_threshold 幀）中：
SEND_MAX_CONSEC_ONE_FRAMES  = 50   # 最長連續 1 超過此值表示仍在手術中，SEND 失敗
SEND_MIN_CONSEC_ZERO_FRAMES = 150  # 最長連續 0 達到此值才確認 SEND 成功

# 顯示設定
SHOW_WINDOW = False  # 是否開啟視窗顯示 (終端/SSH 環境請設 False)
VIS_HEIGHT = 480     
VIS_WIDTH = 640

# VLM 推論與採樣設定
VLM_ENABLE_THINKING = False
VLM_DO_SAMPLE = False
VLM_TEMPERATURE = 0.7
VLM_TOP_P = 0.8
VLM_TOP_K = 20
VLM_REPETITION_PENALTY = 1.0
