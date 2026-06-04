# main_parallel.py
"""
逐對同步分析：Door 視角 + Surgery 視角

行為:
    1. 依影片索引配對 Door 與 Surgery。
    2. 同一對影片共用同步時間軸，結束後才進下一對。
    3. Door 狀態即時寫入 Surgery raw CSV 的 door_open 欄位。
"""

import csv
import os
import sys
import threading
import time
import warnings
from collections import deque
from datetime import datetime, timedelta
from core import PatientStatusAnalyzer
from realtime_pipeline import RealtimePipeline

import cv2

warnings.filterwarnings("ignore", category=UserWarning)

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config import (
    BASE_DIR,
    CAMERA_SETTING,
    CROP_REGION,
    CSV_OUTPUT,
    DOOR_CLOSE_CONFIRM_FRAMES,
    DOOR_EVENT_MATCH_AFTER_FRAMES,
    DOOR_EVENT_MATCH_BEFORE_FRAMES,
    DOOR_OPEN_CONFIRM_FRAMES,
    DOOR_LOOKBACK_FRAMES,
    OR_SETTING,
    OUTPUT_BASE_DIR,
    ROOM,
    STRIDE_SEC,
    VIDEO_DIR,
    VIDEO_DIRS,
)
from door_stage1 import DoorStage1
from live_monitor import LiveMonitorWriter
from utils import video_start_time
import datetime as _dt


# gpu設定
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 執行緒鎖
term_lock = threading.Lock()
term_state = {"door": "等待影片...", "surgery": "等待模型載入..."}
BUFFER_SECONDS = 10
print("\n")


def _redraw_ui():
    sys.stdout.write(f"\r\033[2K  [Door 視角]    {term_state['door']}\n")
    sys.stdout.write(f"\r\033[2K  [Surgery 視角] {term_state['surgery']}\033[1A")
    sys.stdout.flush() #用flush即時更新(因為標準輸出是buufer)。

# 更新 Door / Surgery 的狀態列
def update_ui(worker, text):
    with term_lock:
        term_state[worker] = text
        _redraw_ui()


# 寫入手術事件(log_event)在終端機，並更新ui。
def log_event(text):
    with term_lock:
        sys.stdout.write("\r\033[2K\n\r\033[2K\033[1A")
        sys.stdout.write(f"{text}\n\n\033[1A")
        _redraw_ui()

    
def collect_videos(cam_type: str) -> list[str]:
    """
    根據攝影機類型，door / surgery，去影片資料夾找出符合的影片。
    回傳形態:['.../s01/s01-....mp4',...
    A8:
        door:s01
        surgery:s02
    """
    
    cams = [cam for cam in OR_SETTING[ROOM] if CAMERA_SETTING.get(cam) == cam_type]
    dirs = [os.path.abspath(p) for p in (VIDEO_DIRS if VIDEO_DIRS else [VIDEO_DIR])] #如果有設定多個資料夾，就有，否則單一資料夾[]
    videos = [] # 找到的影片放這裡
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for fname in sorted(os.listdir(d)):
            if fname.lower().endswith((".mp4", ".avi")) and any(cam in fname for cam in cams):
                videos.append(os.path.join(d, fname))
    return sorted(videos)


def _get_dataset_name(video_path: str) -> str:
    """
    從「資料夾名稱」取得資料集 ID。
    來源：路徑的上一層資料夾名稱（非檔名）。
    例如: .../20231211/S01-20231211-072847-xxx.mp4 -> "20231211"

    注意：與 _get_surgery_date 的差異
        - 本函式讀的是「資料夾」，適合用來分辨不同天的資料集。
        - _get_surgery_date 讀的是「檔名第 2 段」，兩者通常相同，
        但若影片放在命名不同的資料夾下就會不一致。
    """
    return os.path.basename(os.path.dirname(video_path))


def _get_surgery_date(video_path: str) -> str:
    """
    從「檔名第 2 段」取得日期標籤，作為 CSV 的 Surgery_Date 欄位值。
    來源：檔名（非資料夾），格式 S01-YYYYMMDD-HHMMSS-xxx.mp4。
    例如: S01-20231211-072847-xxx.mp4 -> "20231211"

    注意：與 _get_dataset_name 的差異
        - 本函式讀的是「檔名」，適合作為輸出 CSV 的日期標籤。
        - _get_dataset_name 讀的是「資料夾名稱」，兩者通常相同。
    失敗時 fallback 至 _get_dataset_name（資料夾名稱）。
    """
    try:
        return os.path.basename(video_path).split("-")[1]
    except Exception:
        return _get_dataset_name(video_path)

    
def _prepare_surgery_outputs(run_date: str, room_video_path: str):
    """
    # 整個手術時間點的輸出的 CSV 檔案路徑
    """
    dataset_name = _get_dataset_name(room_video_path) #得到資料集名稱 (20231211)
    surgery_date = _get_surgery_date(room_video_path) #得到手術日期 (s01)
    room_cams = [cam for cam in OR_SETTING[ROOM] if CAMERA_SETTING.get(cam) == "Room"] #得到攝影機標籤
    cam_label = room_cams[0] if room_cams else "unknown"
    
    # 這個csv是產生過程中的原始手術資料
    raw_csv_path = os.path.join(
        os.path.dirname(CSV_OUTPUT), 
        dataset_name,
        f"surgery_report_Surgery_{dataset_name}_{run_date}_parallel.csv",
    )
    os.makedirs(os.path.dirname(raw_csv_path), exist_ok=True)

    
    report_dir = os.path.join(os.path.dirname(raw_csv_path), "result", ROOM) # 結果報告 分A8 與 A9
    os.makedirs(report_dir, exist_ok=True)
    pair_report_path = os.path.join(report_dir, f"Realtime_Events_Surgery_{dataset_name}_{run_date}.csv") # 最終報告

    #整個對於所有事件的偵測(包含有錯誤的情況)
    unified_events_dir = os.path.join(os.path.dirname(raw_csv_path), "all_events")
    os.makedirs(unified_events_dir, exist_ok=True)
    unified_events_path = os.path.join(unified_events_dir, f"Unified_Events_{dataset_name}_{run_date}.csv")

    return raw_csv_path, pair_report_path, unified_events_path #回傳原始手術資料，最終報告，所有事件


def _hms_to_seconds(value: str) -> int | None:
    """
    把影片時間從 "HH:MM:SS" 轉換成秒數，若失敗回傳 None。
    """

    try:
        h, m, s = (int(part) for part in value.split(":"))
        return h * 3600 + m * 60 + s
    except Exception:
        return None
    


def _event_sort_key(row: dict) -> tuple[int, int, int]:
    """
    根據事件的video做排序
    先從row拿出video_time(轉成秒數)與video_name。
    """
    video_time_sec = _hms_to_seconds(row.get("video_time", ""))
    video_name = row.get("video_name", "")

    # 檔名格式通常為 S01-YYYYMMDD-HHMMSS-1702250927.mp4。
    # 優先使用可讀的日期 + 起始時間；最後一段 timestamp 只作備援。
    try:
        base = os.path.splitext(os.path.basename(video_name))[0]
        parts = base.split("-")
        start_dt = datetime.strptime(f"{parts[1]}{parts[2]}", "%Y%m%d%H%M%S")
        if video_time_sec is not None:
            source_order = 0 if row.get("source") == "door" else 1
            return (int(start_dt.timestamp()) + video_time_sec, source_order, 0)
    except Exception:
        pass

    return (0, 0, 3)


def _sort_csv(path: str, sort_key):
    """
    讀取 CSV、依指定規則排序，再寫回同一個檔案。
    """
    if not os.path.exists(path):
        return
    
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f) #讀取csv檔案，並將每一行轉換成字典形式
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if not fieldnames: #沒有欄位名稱 end
        return

    #排序
    rows.sort(key=sort_key)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)



def _sort_unified_csv(path: str):
    """
    實際排序 unified csv 裡的事件，確保門口事件與手術事件按照真實時間順序排列。
    """
    _sort_csv(path, _event_sort_key)


#開門偵測步驟
def _door_detect_step(detector, frame_bgr, state):
    raw_open = detector.detect(frame_bgr)
    if not state["door_open"]: #門沒開
        if raw_open: #偵測到開門
            state["open_confirm_count"] += 1 
            state["close_confirm_count"] = 0
            if state["open_confirm_count"] >= DOOR_OPEN_CONFIRM_FRAMES: # 開門數到20幀
                state["door_open"] = True
                state["open_confirm_count"] = 0
        else:
            state["open_confirm_count"] = 0
    else:
        if not raw_open:#偵測到沒開門
            state["close_confirm_count"] += 1
            if state["close_confirm_count"] >= DOOR_CLOSE_CONFIRM_FRAMES: # 關門屬到15幀
                state["door_open"] = False
                state["close_confirm_count"] = 0
        else:
            state["close_confirm_count"] = 0
    return raw_open


def main():
    print("=" * 70)
    print("  真實即時系統模擬測試 (Door & Surgery 每對影片同步)")
    print("=" * 70)

    door_videos = collect_videos("Door") # door視角影片
    room_videos = collect_videos("Room") # 手術視角影片
    if not door_videos or not room_videos:
        print("Door 或 Surgery 影片不足，無法配對同步。")
        return

    
    pair_count = min(len(door_videos), len(room_videos)) #取較小的數量，因為要兩兩配對
    if len(door_videos) != len(room_videos):
        print(f"警告: Door={len(door_videos)} 支, Surgery={len(room_videos)} 支，僅同步前 {pair_count} 對。")

    run_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    unified_event_paths = set()  # 記錄已創建過 header 的路徑
    
    analyzer = PatientStatusAnalyzer() #手術狀態分析，內部會載入模型，可能需要一些時間，之後每幀呼叫 analyze_frame 就會得到狀態與推論時間。
    
    # RealtimePipeline 負責根據 analyze_frame 的結果，結合時間窗口等邏輯，推斷手術事件（ENT、SEND 等）並管理狀態轉變。
    pipeline = RealtimePipeline(
        half_window=25,
        stable_frame=900, # ENT的穩定時間
        max_gap_frame=50,
        send_confirm_threshold=900, # SEND的穩定時間
        task_type="Surgery",
    )
    last_stored_all_count = 0 # 上一次寫入 unified CSV 的事件數，用來確保每對影片結束後才 flush pipeline 並寫入事件，避免跨影片汙染。
    last_stored_pair_count = 0 # 上一次寫入 result CSV 的影片對數，
    report_surgery_count = 0  # 實際寫入 result CSV 的刀數計數（不含被濾除的）
    active_report_surgery_no: int | None = None # 當前正在進行中的刀號（從 1 開始），用於確保 SEND 事件必須在 ENT 之後才寫入 result CSV；當 SEND 發生時重置為 None，直到下一個 ENT 出現。
    active_report_ent_date_tag: str | None = None
    active_report_ent_time_sec: int | None = None

    # ── Door 偵測與狀態在所有影片之間共用（跨影片保留開門狀態）──────────
    detector = DoorStage1() # 門口偵測
    live_monitor = LiveMonitorWriter(OUTPUT_BASE_DIR, update_every_frames=5)
    door_state = {"door_open": False, "open_confirm_count": 0, "close_confirm_count": 0} # 初始狀態

    # 從當下幀往前保留門口狀態歷史（回溯查詢）
    # 每筆格式：(real_time_str "HH:MM:SS", is_open: bool)
    door_open_window: deque = deque(maxlen=DOOR_LOOKBACK_FRAMES)  # (real_time_str, bool)
    pending_surgery_events: list[dict] = []

    def _seconds_delta(ts_sec: int, target_sec: int) -> int:
        delta = ts_sec - target_sec
        if delta > 12 * 3600:
            delta -= 24 * 3600
        elif delta < -12 * 3600:
            delta += 24 * 3600
        return delta

    def _door_open_for_event(event_type: str, real_time_str: str) -> bool:
        """
        依事件方向查詢 Door OPEN：
            ENT  看事件前 1 分鐘內是否有 Door OPEN
            SEND 看事件後 1 分鐘內是否有 Door OPEN
        """
        target_sec = _hms_to_seconds(real_time_str)
        if target_sec is None:
            return door_state["door_open"]
        before_sec = DOOR_EVENT_MATCH_BEFORE_FRAMES * STRIDE_SEC
        after_sec = DOOR_EVENT_MATCH_AFTER_FRAMES * STRIDE_SEC
        event_type = (event_type or "").upper()

        for ts_str, is_open in door_open_window:
            if not is_open:
                continue
            ts_sec = _hms_to_seconds(ts_str)
            if ts_sec is None:
                continue
            delta = _seconds_delta(ts_sec, target_sec)
            if event_type == "ENT" and -before_sec <= delta <= 0:
                return True
            if event_type == "SEND" and 0 <= delta <= after_sec:
                return True
        return False

    def _door_status_at(real_time_str: str) -> str:
        """根據事件的 real_time，
            從最近 3 分鐘的 door_open_window 裡，
            找出事件發生當下的門狀態。。
        ex : _door_status_at("15:20:10") 回傳 open 或 close
        """
        if not door_open_window:
            return "OPEN" if door_state["door_open"] else "CLOSE"
        try:
            target = tuple(int(x) for x in real_time_str.split(":")) # 時間字串轉成 (h, m, s) tuple，方便比較大小 (15, 20, 10) <= (15, 21, 00)
        except Exception:
            return "OPEN" if door_open_window[-1][1] else "CLOSE" 
        # 找最接近且 <= target 的那一筆
        best = door_open_window[0] 
        for ts_str, is_open in door_open_window:
            try:
                ts = tuple(int(x) for x in ts_str.split(":"))
                if ts <= target: 
                    best = (ts_str, is_open)
                else:
                    break  # deque 按時間遞增，超過就停
            except Exception:
                continue
        return "OPEN" if best[1] else "CLOSE"

    def _door_timeline_ready(event_type: str, real_time_str: str) -> bool:
        """
        即時處理用 buffer 判斷。
        ENT 只需要 Door 時間軸至少追到事件時間；SEND 需要等待事件後 1 分鐘。
        """
        if not door_open_window:
            return False
        target_sec = _hms_to_seconds(real_time_str)
        latest_door_sec = _hms_to_seconds(door_open_window[-1][0])
        if target_sec is None or latest_door_sec is None:
            return True
        event_type = (event_type or "").upper()
        if event_type == "ENT":
            wait_sec = 0
        elif event_type == "SEND":
            wait_sec = DOOR_EVENT_MATCH_AFTER_FRAMES * STRIDE_SEC
        else:
            wait_sec = BUFFER_SECONDS
        return _seconds_delta(latest_door_sec, target_sec) >= wait_sec

    def _next_date_tag(date_tag: str) -> str:
        try:
            dt = datetime.strptime(date_tag, "%Y%m%d") + timedelta(days=1)
            return dt.strftime("%Y%m%d")
        except Exception:
            return date_tag

    def _result_row_from_event(evt: dict, surgery_date_tag: str) -> dict | None:
        """
        把一個手術事件 evt 轉成 result CSV 要寫入的一列資料
        """
                # 目前第幾刀                 是目前正在進行中的刀號。
        nonlocal report_surgery_count, active_report_surgery_no, active_report_ent_date_tag, active_report_ent_time_sec

        event_type = evt.get("event_type", "?")
        event_real_time = evt.get("real_time", "?")
        event_time_sec = _hms_to_seconds(event_real_time)
        row_date_tag = surgery_date_tag
        
        if event_type == "ENT":
            report_surgery_count += 1 # 每次 ENT 都代表新刀開始，刀數加 1
            active_report_surgery_no = report_surgery_count
            active_report_ent_date_tag = surgery_date_tag
            active_report_ent_time_sec = event_time_sec
        
        elif event_type == "SEND":
            # SEND 必須先有 ENT，否則跳過（不寫入 result CSV）
            if active_report_surgery_no is None:
                return None
            row_date_tag = active_report_ent_date_tag or surgery_date_tag
            if (
                active_report_ent_time_sec is not None
                and event_time_sec is not None
                and event_time_sec < active_report_ent_time_sec
            ):
                row_date_tag = _next_date_tag(row_date_tag)
        else:
            return None  # 其他事件類型不寫入 result CSV

        surgery_no = active_report_surgery_no or report_surgery_count
        row = {
            "Surgery_Date": row_date_tag,
            "Surgery_No":   f"第 {surgery_no} 刀",
            "Type":         event_type,
            "Real_Time":    evt.get("real_time", "?"),
        }
    
        if event_type == "SEND":
            active_report_surgery_no = None
            active_report_ent_date_tag = None
            active_report_ent_time_sec = None

        return row

    # ── 即時剪輯用：事先建立 Room 影片絕對時間戳記列表 ────────────────────
    
    PRE_POST_SEC = 90  # 前後各 1.5 分鐘（共 3 分鐘）

    def _get_video_abs_ts(path):
        """
        從檔名解析影片的絕對開始時間 (Unix timestamp)。
        格式：S02-20231222-153800-xxx.mp4
            parts[1]=日期, parts[2]=時間 → 2023-12-22 15:38:00
        """
        try:
            parts = os.path.basename(path).split("-")
            dt = _dt.datetime.strptime(f"{parts[1]}{parts[2]}", "%Y%m%d%H%M%S")
            return int(dt.timestamp())
        except Exception:
            return None

    room_videos_by_dataset = {}
    for vpath in sorted(collect_videos("Room")):
        vts = _get_video_abs_ts(vpath)
        if vts is not None:
            # 預先計算 fps 與 duration，避免 clip_event 每次都重開影片
            _cap_pre = cv2.VideoCapture(vpath)
            _fps_pre = _cap_pre.get(cv2.CAP_PROP_FPS) or 5.0
            _dur_pre = int(_cap_pre.get(cv2.CAP_PROP_FRAME_COUNT)) / _fps_pre
            _cap_pre.release()
            dataset_key = _get_dataset_name(vpath)
            room_videos_by_dataset.setdefault(dataset_key, []).append({
                "path": vpath, "ts": vts, "fps": _fps_pre, "dur": _dur_pre
            })
    for _videos in room_videos_by_dataset.values():
        _videos.sort(key=lambda x: x["ts"])

    def clip_event(row: dict, video_output_dir: str, dataset_name: str):
        """依 Real_Time 對同一 dataset 的 room 影片裁剪前後 PRE_POST_SEC 秒。"""
        dataset_room_videos = room_videos_by_dataset.get(dataset_name, [])
        if not dataset_room_videos:
            log_event(f"  [剪輯略過] dataset={dataset_name} 找不到 Room 影片")
            return
        surgery_date_tag = row.get("Surgery_Date", "unknown")
        sno    = str(row.get("Surgery_No", "?")).replace("/", "").replace("\\", "")
        etype  = row.get("Type", "?")
        rt_raw = row.get("Real_Time", "000000")
        rt_str = rt_raw.replace(":", "")
        dst_name = f"{surgery_date_tag}_{etype}_{sno}_{rt_str}.mp4"
        dst_path = os.path.join(video_output_dir, dst_name)
        if os.path.exists(dst_path):
            return
        try:
            rh, rm, rs = map(int, rt_raw.split(":"))
        except Exception:
            return

        
        event_abs_ts = None
        for v in dataset_room_videos:
            vts = v["ts"]
            dur_tmp = v["dur"]  # 使用預先計算的 duration，不需重開影片
            try:
                base_dt = _dt.datetime.fromtimestamp(vts)
                cand_dt = base_dt.replace(hour=rh, minute=rm, second=rs)
                if cand_dt < base_dt:  # 時間往回跳 -> 跨日
                    cand_dt += _dt.timedelta(days=1)  # 加一天
                cand = int(cand_dt.timestamp())
                if vts <= cand <= vts + dur_tmp:
                    event_abs_ts = cand
                    break
            except Exception:
                continue
        if event_abs_ts is None and dataset_room_videos:
            try:
                _fb_vts = dataset_room_videos[0]["ts"]
                _fb_base = _dt.datetime.fromtimestamp(_fb_vts)
                _fb_cand = _fb_base.replace(hour=rh, minute=rm, second=rs)
                if _fb_cand < _fb_base:  # 跨日修正
                    _fb_cand += _dt.timedelta(days=1)
                event_abs_ts = int(_fb_cand.timestamp())
            except Exception:
                return
        if event_abs_ts is None:
            return
        t_start = event_abs_ts - PRE_POST_SEC
        t_end   = event_abs_ts + PRE_POST_SEC
        try:
            writer = None
            for v in dataset_room_videos:
                cap_c = cv2.VideoCapture(v["path"])
                fps_c = v["fps"]  # 使用預先計算的 fps
                dur_c = v["dur"]  # 使用預先計算的 duration
                vs = v["ts"]; ve = vs + dur_c
                if ve < t_start or vs > t_end:
                    cap_c.release(); continue
                if writer is None:
                    os.makedirs(video_output_dir, exist_ok=True)
                    writer = cv2.VideoWriter(
                        dst_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_c,
                        (int(cap_c.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(cap_c.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                    )
                cap_c.set(cv2.CAP_PROP_POS_FRAMES, int(max(0.0, t_start - vs) * fps_c))
                limit = int(min(dur_c, t_end - vs) * fps_c)
                while cap_c.isOpened():
                    ret_c, frm_c = cap_c.read()
                    if not ret_c or cap_c.get(cv2.CAP_PROP_POS_FRAMES) > limit:
                        break
                    writer.write(frm_c)
                cap_c.release()
            if writer:
                writer.release()
                log_event(f"  [剪輯完成] {dst_name}")
        except Exception as e:
            log_event(f"  [剪輯失敗] {dst_name} ({e})")

    def _process_pending_surgery_events(
        path_unified: str,
        path_report: str,
        surgery_date_tag: str,
        dataset_name: str,
        force: bool = False,
    ):
        """
        將已等待足夠 Door 時間軸的 Surgery 事件寫出。
        force=True 用於影片/資料集結尾，把剩餘事件全部依目前已知 Door 狀態寫出。
        """
        nonlocal pending_surgery_events
        if not pending_surgery_events:
            return

        pending_surgery_events.sort(key=_event_sort_key)
        ready_evts = []
        remain_evts = []
        for evt in pending_surgery_events:
            evt_real_time = evt.get("real_time", "")
            if force or _door_timeline_ready(evt.get("event_type", ""), evt_real_time):
                ready_evts.append(evt)
            else:
                remain_evts.append(evt)

        if not ready_evts:
            pending_surgery_events = remain_evts
            return

        result_rows = []
        rejected_evts = []
        with open(path_unified, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "source", "event_type", "video_time", "real_time",
                "video_name", "door_status",
            ])
            for evt in ready_evts:
                evt_real_time = evt.get("real_time", "")
                door_status = "OPEN" if _door_open_for_event(evt.get("event_type", ""), evt_real_time) else "CLOSE"
                writer.writerow({
                    "source": "surgery",
                    "event_type": evt.get("event_type", ""),
                    "video_time": evt.get("video_time", ""),
                    "real_time":  evt_real_time,
                    "video_name": evt.get("video_name", ""),
                    "door_status": door_status,
                })
                if evt.get("event_type") in ("ENT", "SEND"):
                    if door_status == "OPEN":
                        row = _result_row_from_event(evt, surgery_date_tag)
                        if row is not None:
                            result_rows.append(row)
                        else:
                            log_event(
                                f"[Surgery 順序濾除] {evt.get('event_type','?')} "
                                f"@ {evt.get('real_time','?')} — SEND 沒有先行 ENT，跳過"
                            )
                    else:
                        rejected_evts.append(evt)

        _sort_unified_csv(path_unified)
        if result_rows:
            with open(path_report, "a", newline="", encoding="utf-8-sig") as f:
                csv.DictWriter(f, fieldnames=[
                    "Surgery_Date", "Surgery_No", "Type", "Real_Time"
                ]).writerows(result_rows)
            video_output_dir = os.path.join(os.path.dirname(path_report), "videos")
            for r in result_rows:
                log_event(
                    f"[Surgery 事件✓] {surgery_date_tag} {r['Surgery_No']} "
                    f"{r['Type']} 真實時間:{r['Real_Time']}"
                )
                clip_event(r, video_output_dir, dataset_name)
        for evt in rejected_evts:
            log_event(
                f"[Surgery 誤判濾除] {evt.get('event_type','?')} "
                f"影片時間:{evt.get('video_time','?')} — 事件指定方向時間窗內 Door 未開啟，彈出不計入"
            )

        pending_surgery_events = remain_evts

    # ── 依 dataset_name 分群，同一資料集跑完才 flush + reset ──────────────
    # 好處：同一天的影片（包含跨夜手術）在同一個 pipeline 內連續處理；
    #       不同天之間才 flush + reset，確保事件不跨資料集汙染。
    from itertools import groupby

    pairs = list(zip(door_videos[:pair_count], room_videos[:pair_count]))
    grouped = groupby(pairs, key=lambda p: _get_dataset_name(p[1]))
    dataset_groups = [(ds, list(ps)) for ds, ps in grouped]

    global_pair_idx = 0
    _last_pair_ctx = None

    def _flush_and_write_tail(path_unified, path_report, date_tag, dataset_name):
        """flush pipeline 並把尾端事件補寫到指定路徑的 CSV。"""
        nonlocal last_stored_all_count, last_stored_pair_count
        _process_pending_surgery_events(path_unified, path_report, date_tag, dataset_name, force=True)
        pipeline.flush()
        pipeline.force_close_pending_send()  # 若影片結尾手術未結束，強制補 SEND
        _all = pipeline.get_all_events()
        if len(_all) > last_stored_all_count:
            _new = _all[last_stored_all_count:]
            # 每個事件個別查詢指定方向時間窗內是否有 Door OPEN，與主迴圈 _door_open_for_event 邏輯一致
            _result_rows_d = []
            with open(path_unified, "a", newline="", encoding="utf-8-sig") as f:
                _w = csv.DictWriter(f, fieldnames=[
                    "source", "event_type", "video_time", "real_time",
                    "video_name", "door_status",
                ])
                for _evt in _new:
                    _rt = _evt.get("real_time", "")
                    _door_status = "OPEN" if _door_open_for_event(_evt.get("event_type", ""), _rt) else "CLOSE"
                    _w.writerow({
                        "source": "surgery",
                        "event_type": _evt.get("event_type", ""),
                        "video_time": _evt.get("video_time", ""),
                        "real_time": _rt,
                        "video_name": _evt.get("video_name", ""),
                        "door_status": _door_status,
                    })
                    if _evt.get("event_type") in ("ENT", "SEND") and _door_status == "OPEN":
                        _row = _result_row_from_event(_evt, date_tag)
                        if _row is not None:
                            _result_rows_d.append(_row)
            _sort_unified_csv(path_unified)
            if _result_rows_d:
                with open(path_report, "a", newline="", encoding="utf-8-sig") as f:
                    csv.DictWriter(f, fieldnames=[
                        "Surgery_Date", "Surgery_No", "Type", "Real_Time"
                    ]).writerows(_result_rows_d)
                video_output_dir = os.path.join(os.path.dirname(path_report), "videos")
                for _r in _result_rows_d:
                    log_event(
                        f"[Surgery 事件✓][資料集flush] {date_tag} "
                        f"{_r['Surgery_No']} {_r['Type']} 真實時間:{_r['Real_Time']}"
                    )
                    clip_event(_r, video_output_dir, dataset_name)
            last_stored_all_count = len(_all)

    for ds_idx, (dataset_name, ds_pairs) in enumerate(dataset_groups):
        log_event(
            f"[資料集 {ds_idx+1}/{len(dataset_groups)}] {dataset_name} "
            f"（共 {len(ds_pairs)} 對影片）"
        )
        last_stored_all_count = 0
        last_stored_pair_count = 0
        report_surgery_count = 0        # 每個資料集重置刀數，從第 1 刀重新計算
        active_report_surgery_no = None  # 每個資料集重置當前刀號
        active_report_ent_date_tag = None
        active_report_ent_time_sec = None

        for door_video, room_video in ds_pairs:
            global_pair_idx += 1
            log_event(
                f"[Pair {global_pair_idx}/{pair_count}] "
                f"Door={os.path.basename(door_video)} | Room={os.path.basename(room_video)}"
            )

            raw_csv_path, pair_report_path, unified_events_path = _prepare_surgery_outputs(run_date, room_video)

            # 統一事件 CSV （對每個 dataset 只建一次 header）
            if unified_events_path not in unified_event_paths:
                with open(unified_events_path, "w", newline="", encoding="utf-8-sig") as f:
                    csv.DictWriter(f, fieldnames=[
                        "source", "event_type", "video_time", "real_time",
                        "video_name", "door_status",
                    ]).writeheader()
                unified_event_paths.add(unified_events_path)
            if not os.path.exists(raw_csv_path):
                with open(raw_csv_path, "w", newline="", encoding="utf-8-sig") as f:
                    csv.writer(f).writerow([
                        "Video_name", "frame_index", "video_time", "real_time",
                        "status", "voted_status", "infer_time", "door_open",
                        "door_score", "door_ratio",
                    ])
            if not os.path.exists(pair_report_path):
                with open(pair_report_path, "w", newline="", encoding="utf-8-sig") as f:
                    csv.DictWriter(f, fieldnames=[
                        "Surgery_Date", "Surgery_No", "Type", "Real_Time"
                    ]).writeheader()

            door_cap = cv2.VideoCapture(door_video)
            room_cap = cv2.VideoCapture(room_video)
            if not door_cap.isOpened() or not room_cap.isOpened():
                if door_cap.isOpened():
                    door_cap.release()
                if room_cap.isOpened():
                    room_cap.release()
                continue

            # raw CSV 在整支影片開始前開啟一次，避免逐幀 open/close
            raw_csv_f = open(raw_csv_path, "a", newline="", encoding="utf-8-sig")
            raw_csv_writer = csv.writer(raw_csv_f)

            door_fps = door_cap.get(cv2.CAP_PROP_FPS) or 5.0
            room_fps = room_cap.get(cv2.CAP_PROP_FPS) or 5.0
            door_stride = max(1, int(door_fps * STRIDE_SEC))
            room_stride = max(1, int(room_fps * STRIDE_SEC))
            door_total = int(door_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            room_total = int(room_cap.get(cv2.CAP_PROP_FRAME_COUNT))

            door_frame_idx = 0
            room_frame_idx = 0
            total_frames_analyzed = 0
            # detector 與 door_state 跨影片共用，不重新建立
            # 每支新影片：只重設背景模型（讓 EMA 從新影片第一幀開始學），
            # 但保留 door_state["door_open"] 等狀態，避免重複觸發開門事件。
            detector.reset()  # 清除背景 EMA，但 door_state 繼續沿用
            door_start_dt = video_start_time(door_video) or datetime.now()
            room_start_dt = video_start_time(room_video) or datetime.now()

            # ── 開門期間逐幀存圖設定 ──────────────────────────────────────────────
            # 儲存路徑: outputs/{dataset_name}/door_open_frames/{door_video_stem}/
            door_video_stem = os.path.splitext(os.path.basename(door_video))[0]
            door_frames_dir: str | None = None  # 動態設定，每次開門時以該時間點命名
            door_open_start_tag: str | None = None  # 記錄「這次開門」的真實時間標籤

            while door_frame_idx < door_total and room_frame_idx < room_total:
                door_cap.set(cv2.CAP_PROP_POS_FRAMES, door_frame_idx)
                room_cap.set(cv2.CAP_PROP_POS_FRAMES, room_frame_idx)
                ret_door, door_frame = door_cap.read()
                ret_room, room_frame = room_cap.read()
                if not ret_door or not ret_room:
                    break

                current_sec = room_frame_idx / room_fps
                video_time_str = time.strftime("%H:%M:%S", time.gmtime(current_sec))
                real_time_str = (room_start_dt + timedelta(seconds=current_sec)).strftime("%H:%M:%S")
                door_sec = door_frame_idx / door_fps
                door_video_time_str = time.strftime("%H:%M:%S", time.gmtime(door_sec))
                door_real_time_str = (door_start_dt + timedelta(seconds=door_sec)).strftime("%H:%M:%S")

                if CROP_REGION is not None:
                    x1, y1, x2, y2 = CROP_REGION
                    door_analysis = door_frame[y1:y2, x1:x2]
                else:
                    door_analysis = door_frame
                prev_door_open = door_state["door_open"]
                raw_open = _door_detect_step(detector, door_analysis, door_state)  #偵測開門
                now_door_open = door_state["door_open"]

                if now_door_open != prev_door_open:
                    event_type = "door_open" if now_door_open else "door_close"
                    with open(unified_events_path, "a", newline="", encoding="utf-8-sig") as f:
                        csv.DictWriter(f, fieldnames=[
                            "source", "event_type", "video_time", "real_time",
                            "video_name", "door_status",
                        ]).writerow({
                            "source": "door",
                            "event_type": event_type,
                            "video_time": door_video_time_str,   # door 影片自己的時間
                            "real_time": door_real_time_str,     # door 影片對應的真實時間
                            "video_name": os.path.basename(door_video),
                            "door_status": "OPEN" if now_door_open else "CLOSE",
                        })
                    _sort_unified_csv(unified_events_path)
                    log_event(f"  [Door] {'OPEN' if now_door_open else 'CLOSE'} @ {door_video_time_str}")
                    # 開門轉態時記錄開門時間標籤；關門時清除
                    if now_door_open:
                        door_open_start_tag = door_real_time_str.replace(":", "")
                        # 以「開門時間點」為資料夾名稱，每次開門都建立獨立資料夾
                        # 例如：door_open_frames/OPEN_153859/
                        door_frames_dir = os.path.join(
                            OUTPUT_BASE_DIR, dataset_name, "door_open_frames",
                            f"OPEN_{door_open_start_tag}"
                        )
                    else:
                        door_open_start_tag = None
                        door_frames_dir = None
                update_ui("door", f"{video_time_str} | {'OPEN' if door_state['door_open'] else 'CLOSE'} | score={detector.last_score:.1f}")

                # ── 開門期間：存下 S01 每一個 frame（完整原始畫面）──────────────
                if door_state["door_open"] and door_open_start_tag and door_frames_dir:
                    os.makedirs(door_frames_dir, exist_ok=True)
                    # 檔名格式：OPEN_{開門時間}_{目前幀real_time}_f{frame_idx}.jpg
                    rt_tag = door_real_time_str.replace(":", "")
                    frame_filename = f"OPEN_{door_open_start_tag}_{rt_tag}_f{door_frame_idx:06d}.jpg"
                    cv2.imwrite(os.path.join(door_frames_dir, frame_filename), door_frame)

                status, _, infer_time = analyzer.analyze_frame(
                    room_frame,
                    "Surgery",
                    full_frame=room_frame,
                    current_sec=current_sec,
                    current_frame=total_frames_analyzed,
                    video_name=os.path.basename(room_video),
                    real_time=real_time_str,
                )
                if getattr(analyzer, "push_to_pipeline", True):
                    pipeline.push_frame_result(
                        status=status,
                        frame_idx=total_frames_analyzed,
                        video_time=video_time_str,
                        real_time=real_time_str,
                        video_name=os.path.basename(room_video),
                    )
                else:
                    pipeline.push_frame_result(
                        status=0,
                        frame_idx=total_frames_analyzed,
                        video_time=video_time_str,
                        real_time=real_time_str,
                        video_name=os.path.basename(room_video),
                    )

                voted = pipeline.voted_statuses[-1] if pipeline.voted_statuses else "-"
                state = pipeline.get_current_state()
                update_ui("surgery", f"{video_time_str} | Door:{'OPEN' if door_state['door_open'] else 'CLOSE'} | AI:{voted} ({state['confirmed_state_text']})")

                live_monitor.update(
                    frame_idx=total_frames_analyzed,
                    door_frame=door_frame,
                    door_analysis=door_analysis,
                    room_frame=room_frame,
                    status={
                        "dataset": dataset_name,
                        "surgery_date": _get_surgery_date(room_video),
                        "door_video": os.path.basename(door_video),
                        "room_video": os.path.basename(room_video),
                        "door_video_time": door_video_time_str,
                        "door_real_time": door_real_time_str,
                        "video_time": video_time_str,
                        "real_time": real_time_str,
                        "door_open": bool(door_state["door_open"]),
                        "raw_open": bool(raw_open),
                        "door_score": round(float(detector.last_score), 3),
                        "door_ratio": round(float(detector.last_ratio), 5),
                        "open_confirm_count": door_state["open_confirm_count"],
                        "close_confirm_count": door_state["close_confirm_count"],
                        "status": status,
                        "voted_status": voted,
                        "confirmed_state_text": state["confirmed_state_text"],
                        "infer_time": round(float(infer_time), 3),
                    },
                )

                raw_csv_writer.writerow([
                        os.path.basename(room_video),
                        room_frame_idx,
                        video_time_str,
                        real_time_str,
                        status,
                        voted,
                        f"{infer_time:.3f}",
                        1 if door_state["door_open"] else 0,
                        f"{detector.last_score:.3f}",
                        f"{detector.last_ratio:.5f}",
                ])

                # ── 更新門口狀態蒐集窗口（每幀必存，帶 Door 自己的 real_time 時間戳）──────
                door_open_window.append((door_real_time_str, door_state["door_open"]))

                all_detected = pipeline.get_all_events()
                if len(all_detected) > last_stored_all_count:
                    new_evts = all_detected[last_stored_all_count:]
                    surgery_date_tag = _get_surgery_date(room_video)
                    pending_surgery_events.extend(new_evts)
                    _process_pending_surgery_events(
                        unified_events_path,
                        pair_report_path,
                        surgery_date_tag,
                        dataset_name,
                    )
                    last_stored_all_count = len(all_detected)

                _process_pending_surgery_events(
                    unified_events_path,
                    pair_report_path,
                    _get_surgery_date(room_video),
                    dataset_name,
                )

                summary = pipeline.get_event_summary()
                if len(summary) > last_stored_pair_count:
                    last_stored_pair_count = len(summary)

                door_frame_idx += door_stride
                room_frame_idx += room_stride
                total_frames_analyzed += 1

            raw_csv_f.close()  # 影片跑完才關閉 raw CSV
            _process_pending_surgery_events(
                unified_events_path,
                pair_report_path,
                _get_surgery_date(room_video),
                dataset_name,
                force=True,
            )
            door_cap.release()
            room_cap.release()
            log_event(f"[Pair {global_pair_idx}] 同步完成")
            _last_pair_ctx = {
                "unified_events_path": unified_events_path,
                "pair_report_path":    pair_report_path,
                "surgery_date_tag":    _get_surgery_date(room_video),
                "dataset_name":        dataset_name,
            }
            # ↑ end of inner pair for-loop

        # ── 資料集所有影片跑完：flush + reset pipeline ────────────────
        log_event(f"[資料集 {dataset_name} 完成] flush + reset pipeline...")
        if _last_pair_ctx is not None:
            _flush_and_write_tail(
                _last_pair_ctx["unified_events_path"],
                _last_pair_ctx["pair_report_path"],
                _last_pair_ctx["surgery_date_tag"],
                _last_pair_ctx["dataset_name"],
            )
        pipeline = RealtimePipeline(
            half_window=25,
            stable_frame=900,
            max_gap_frame=50,
            send_confirm_threshold=900,
            task_type="Surgery",
        )
        last_stored_all_count = 0
        last_stored_pair_count = 0
    # ↑ end of outer dataset for-loop

    # ── 重新排序所有 Unified Events CSV ──────────────────────────────────
    for path in unified_event_paths:
        _sort_unified_csv(path)
        print(f"  [排序完成] {os.path.basename(path)}")

    print("\n\n" + "=" * 70)
    print("  同步測試完成")
    for path in sorted(unified_event_paths):
        print(f"  統一事件 CSV: {path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
