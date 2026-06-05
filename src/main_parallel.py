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
import time
import warnings
from collections import deque
from datetime import datetime, timedelta
from itertools import groupby

from core import PatientStatusAnalyzer
from realtime_pipeline import RealtimePipeline

import cv2

warnings.filterwarnings("ignore", category=UserWarning)

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config import CROP_REGION, DOOR_LOOKBACK_FRAMES, OUTPUT_BASE_DIR, STRIDE_SEC
from door_stage1 import DoorStage1
from pipeline.door_state import door_detect_step, door_open_for_event, door_timeline_ready
from pipeline.event_clipper import RoomEventClipper
from pipeline.event_utils import event_sort_key, hms_to_seconds, next_date_tag, sort_unified_csv
from live_monitor import LiveMonitorWriter
from monitoring.terminal_ui import log_event, update_ui
from utils import video_start_time
from pipeline.video_outputs import (
    collect_videos,
    get_dataset_name,
    get_surgery_date,
    prepare_surgery_outputs,
)


# gpu設定
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

BUFFER_SECONDS = 10
print("\n")

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

    def _door_open_for_event(event_type: str, real_time_str: str) -> bool:
        return door_open_for_event(
            event_type,
            real_time_str,
            door_open_window,
            door_state["door_open"],
        )

    def _door_timeline_ready(event_type: str, real_time_str: str) -> bool:
        return door_timeline_ready(
            event_type,
            real_time_str,
            door_open_window,
            BUFFER_SECONDS,
        )

    def _result_row_from_event(evt: dict, surgery_date_tag: str) -> dict | None:
        """
        把一個手術事件 evt 轉成 result CSV 要寫入的一列資料
        """
                # 目前第幾刀                 是目前正在進行中的刀號。
        nonlocal report_surgery_count, active_report_surgery_no, active_report_ent_date_tag, active_report_ent_time_sec

        event_type = evt.get("event_type", "?")
        event_real_time = evt.get("real_time", "?")
        event_time_sec = hms_to_seconds(event_real_time)
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
                row_date_tag = next_date_tag(row_date_tag)
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
    clipper = RoomEventClipper(log_event, pre_post_sec=90)

    def clip_event(row: dict, video_output_dir: str, dataset_name: str):
        clipper.clip_event(row, video_output_dir, dataset_name)

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

        pending_surgery_events.sort(key=event_sort_key)
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

        sort_unified_csv(path_unified)
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
    pairs = list(zip(door_videos[:pair_count], room_videos[:pair_count]))
    grouped = groupby(pairs, key=lambda p: get_dataset_name(p[1]))
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
            sort_unified_csv(path_unified)
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

            raw_csv_path, pair_report_path, unified_events_path = prepare_surgery_outputs(run_date, room_video)

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
                raw_open = door_detect_step(detector, door_analysis, door_state)  #偵測開門
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
                    sort_unified_csv(unified_events_path)
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
                        "surgery_date": get_surgery_date(room_video),
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
                    surgery_date_tag = get_surgery_date(room_video)
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
                    get_surgery_date(room_video),
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
                get_surgery_date(room_video),
                dataset_name,
                force=True,
            )
            door_cap.release()
            room_cap.release()
            log_event(f"[Pair {global_pair_idx}] 同步完成")
            _last_pair_ctx = {
                "unified_events_path": unified_events_path,
                "pair_report_path":    pair_report_path,
                "surgery_date_tag":    get_surgery_date(room_video),
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
        sort_unified_csv(path)
        print(f"  [排序完成] {os.path.basename(path)}")

    print("\n\n" + "=" * 70)
    print("  同步測試完成")
    for path in sorted(unified_event_paths):
        print(f"  統一事件 CSV: {path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
