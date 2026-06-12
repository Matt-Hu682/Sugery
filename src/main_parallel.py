# main_parallel.py
"""
Door + Surgery polling pipeline.

行為:
    1. 定期掃描新的 Door / Room 影片，不要求一開始兩路已配對完成。
    2. Door 與 Room 各自形成 stream，依檔名起始時間 + frame offset 的 real_time 排序處理。
    3. Room 產生 ENT/SEND 後先放入 pending，等 Door 時間軸追到事件驗證窗後再寫 result。
"""

import csv
import os
import sys
import time
import warnings
from collections import deque
from datetime import datetime, timedelta

import cv2
from core import PatientStatusAnalyzer
from realtime_pipeline import RealtimePipeline

warnings.filterwarnings("ignore", category=UserWarning)

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config import CROP_REGION, DOOR_LOOKBACK_FRAMES, LIVE_MONITOR_BASE_DIR, OUTPUT_BASE_DIR, STRIDE_SEC
from door_stage1 import DoorStage1
from live_monitor import LiveMonitorWriter
from monitoring.terminal_ui import log_event, update_ui
from pipeline.door_state import door_detect_step, door_open_for_event, door_timeline_ready
from pipeline.event_clipper import RoomEventClipper
from pipeline.event_utils import event_sort_key, hms_to_seconds, next_date_tag, sort_unified_csv
from pipeline.video_outputs import collect_videos, get_dataset_name, get_surgery_date, prepare_surgery_outputs
from utils import video_start_time


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

BUFFER_SECONDS = 0
POLL_INTERVAL_SEC = float(os.environ.get("SURGERY_POLL_INTERVAL_SEC", "2"))
POLL_FOREVER = os.environ.get("SURGERY_POLL_FOREVER", "0") == "1"
POLL_IDLE_LIMIT = int(os.environ.get("SURGERY_POLL_IDLE_LIMIT", "3"))
STREAM_EOF_RETRIES = int(os.environ.get("SURGERY_STREAM_EOF_RETRIES", "3"))
SLOW_LOOP_LOG_THRESHOLD_SEC = float(os.environ.get("SURGERY_SLOW_LOOP_LOG_THRESHOLD_SEC", "1.0"))
print("\n")

UNIFIED_FIELDS = ["source", "event_type", "video_time", "real_time", "video_name", "door_status"]
RAW_FIELDS = [
    "Video_name", "frame_index", "video_time", "real_time", "status",
    "voted_status", "infer_time", "door_open", "door_score", "door_ratio",
]
RESULT_FIELDS = ["Surgery_Date", "Surgery_No", "Type", "Real_Time"]


def _video_start_key(path: str) -> tuple[int, str]:
    start_dt = video_start_time(path)
    if start_dt is None:
        return (0, os.path.basename(path))
    return (int(start_dt.timestamp()), os.path.basename(path))


def _append_csv(path: str, fieldnames: list[str], row: dict):
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writerow(row)


def _read_monotonic_frame(stream: dict):
    target_idx = stream["next_frame_idx"]
    cap = stream["cap"]

    while stream["next_read_idx"] < target_idx:
        if not cap.grab():
            return None
        stream["next_read_idx"] += 1

    ret, frame = cap.read()
    if not ret:
        return None
    stream["next_read_idx"] += 1
    return frame


def _open_stream(path: str, kind: str) -> dict | None:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 5.0
    stride = max(1, int(fps * STRIDE_SEC))
    start_dt = video_start_time(path) or datetime.now()
    return {
        "kind": kind,
        "path": path,
        "cap": cap,
        "fps": fps,
        "stride": stride,
        "start_dt": start_dt,
        "next_frame_idx": 0,
        "next_read_idx": 0,
        "retry_after": 0.0,
        "eof_retries": 0,
        "done": False,
    }


def _stream_next_abs_ts(stream: dict) -> float:
    return stream["start_dt"].timestamp() + (stream["next_frame_idx"] / stream["fps"])


def main():
    print("=" * 70)
    print("  真實即時系統模擬測試 (Door & Surgery polling + time-window match)")
    print("=" * 70)

    run_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    unified_event_paths = set()
    output_cache: dict[str, tuple[str, str, str]] = {}
    raw_csv_files = {}
    raw_csv_writers = {}

    analyzer = PatientStatusAnalyzer()
    pipeline = RealtimePipeline(
        half_window=25,
        stable_frame=900,
        max_gap_frame=50,
        send_confirm_threshold=900,
        task_type="Surgery",
    )
    last_stored_all_count = 0
    report_surgery_count = 0
    active_report_surgery_no: int | None = None
    active_report_ent_date_tag: str | None = None
    active_report_ent_time_sec: int | None = None
    total_room_frames_analyzed = 0

    detector = DoorStage1()
    live_monitor = LiveMonitorWriter(LIVE_MONITOR_BASE_DIR, update_every_frames=5)
    door_state = {"door_open": False, "open_confirm_count": 0, "close_confirm_count": 0}
    door_open_window: deque = deque(maxlen=DOOR_LOOKBACK_FRAMES)
    pending_surgery_events: list[dict] = []
    clipper = RoomEventClipper(log_event, pre_post_sec=90)

    streams: list[dict] = []
    pending_stream_paths: list[dict] = []
    seen_paths: set[str] = set()
    last_door_frame = None
    last_door_analysis = None
    last_room_frame = None

    def _ensure_outputs(video_path: str) -> tuple[str, str, str]:
        dataset_name = get_dataset_name(video_path)
        if dataset_name in output_cache:
            return output_cache[dataset_name]

        raw_csv_path, pair_report_path, unified_events_path = prepare_surgery_outputs(run_date, video_path)
        if not os.path.exists(raw_csv_path):
            with open(raw_csv_path, "w", newline="", encoding="utf-8-sig") as f:
                csv.writer(f).writerow(RAW_FIELDS)
        if not os.path.exists(pair_report_path):
            with open(pair_report_path, "w", newline="", encoding="utf-8-sig") as f:
                csv.DictWriter(f, fieldnames=RESULT_FIELDS).writeheader()
        if unified_events_path not in unified_event_paths:
            with open(unified_events_path, "w", newline="", encoding="utf-8-sig") as f:
                csv.DictWriter(f, fieldnames=UNIFIED_FIELDS).writeheader()
            unified_event_paths.add(unified_events_path)

        output_cache[dataset_name] = (raw_csv_path, pair_report_path, unified_events_path)
        return output_cache[dataset_name]

    def _get_raw_csv_writer(raw_csv_path: str):
        writer = raw_csv_writers.get(raw_csv_path)
        if writer is not None:
            return writer

        f = open(raw_csv_path, "a", newline="", encoding="utf-8-sig")
        raw_csv_files[raw_csv_path] = f
        writer = csv.writer(f)
        raw_csv_writers[raw_csv_path] = writer
        return writer

    def _door_open_for_event(event_type: str, real_time_str: str) -> bool:
        return door_open_for_event(event_type, real_time_str, door_open_window, door_state["door_open"])

    def _door_timeline_ready(event_type: str, real_time_str: str) -> bool:
        return door_timeline_ready(event_type, real_time_str, door_open_window, BUFFER_SECONDS)

    def _result_row_from_event(evt: dict, surgery_date_tag: str) -> dict | None:
        nonlocal report_surgery_count, active_report_surgery_no, active_report_ent_date_tag, active_report_ent_time_sec

        event_type = evt.get("event_type", "?")
        event_real_time = evt.get("real_time", "?")
        event_time_sec = hms_to_seconds(event_real_time)
        row_date_tag = surgery_date_tag

        if event_type == "ENT":
            report_surgery_count += 1
            active_report_surgery_no = report_surgery_count
            active_report_ent_date_tag = surgery_date_tag
            active_report_ent_time_sec = event_time_sec
        elif event_type == "SEND":
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
            return None

        surgery_no = active_report_surgery_no or report_surgery_count
        row = {
            "Surgery_Date": row_date_tag,
            "Surgery_No": f"第 {surgery_no} 刀",
            "Type": event_type,
            "Real_Time": evt.get("real_time", "?"),
        }

        if event_type == "SEND":
            active_report_surgery_no = None
            active_report_ent_date_tag = None
            active_report_ent_time_sec = None

        return row

    def _process_pending_surgery_events(force: bool = False):
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

        for evt in ready_evts:
            path_unified = evt["unified_events_path"]
            path_report = evt["pair_report_path"]
            surgery_date_tag = evt["surgery_date_tag"]
            dataset_name = evt["dataset_name"]
            evt_real_time = evt.get("real_time", "")
            door_status = "OPEN" if _door_open_for_event(evt.get("event_type", ""), evt_real_time) else "CLOSE"

            _append_csv(path_unified, UNIFIED_FIELDS, {
                "source": "surgery",
                "event_type": evt.get("event_type", ""),
                "video_time": evt.get("video_time", ""),
                "real_time": evt_real_time,
                "video_name": evt.get("video_name", ""),
                "door_status": door_status,
            })
            sort_unified_csv(path_unified)

            if evt.get("event_type") not in ("ENT", "SEND"):
                continue
            if door_status != "OPEN":
                log_event(
                    f"[Surgery 誤判濾除] {evt.get('event_type','?')} "
                    f"影片時間:{evt.get('video_time','?')} — 事件指定方向時間窗內 Door 未開啟，彈出不計入"
                )
                continue

            row = _result_row_from_event(evt, surgery_date_tag)
            if row is None:
                log_event(
                    f"[Surgery 順序濾除] {evt.get('event_type','?')} "
                    f"@ {evt.get('real_time','?')} — SEND 沒有先行 ENT，跳過"
                )
                continue

            with open(path_report, "a", newline="", encoding="utf-8-sig") as f:
                csv.DictWriter(f, fieldnames=RESULT_FIELDS).writerow(row)
            log_event(
                f"[Surgery 事件✓] {surgery_date_tag} {row['Surgery_No']} "
                f"{row['Type']} 真實時間:{row['Real_Time']}"
            )
            video_output_dir = os.path.join(os.path.dirname(path_report), "videos")
            clipper.clip_event(row, video_output_dir, dataset_name)

        pending_surgery_events = remain_evts

    def _discover_new_streams() -> int:
        new_count = 0
        for kind, videos in (("Door", collect_videos("Door")), ("Room", collect_videos("Room"))):
            for path in sorted(videos, key=_video_start_key):
                if path in seen_paths:
                    continue
                start_dt = video_start_time(path) or datetime.now()
                seen_paths.add(path)
                pending_stream_paths.append({
                    "kind": kind,
                    "path": path,
                    "start_ts": start_dt.timestamp(),
                })
                _ensure_outputs(path)
                new_count += 1
                log_event(f"[Polling] 排隊{kind}影片: {os.path.basename(path)}")
        pending_stream_paths.sort(key=lambda item: (item["start_ts"], item["kind"], os.path.basename(item["path"])))
        return new_count

    def _activate_next_streams() -> int:
        opened = 0
        active_kinds = {s["kind"] for s in streams if not s["done"]}
        remaining = []
        for item in pending_stream_paths:
            if item["kind"] in active_kinds:
                remaining.append(item)
                continue
            stream = _open_stream(item["path"], item["kind"])
            if stream is None:
                log_event(f"[Polling] 開啟失敗，稍後重試: {os.path.basename(item['path'])}")
                remaining.append(item)
                continue
            streams.append(stream)
            active_kinds.add(item["kind"])
            opened += 1
            log_event(f"[Polling] 開始{item['kind']}影片: {os.path.basename(item['path'])}")
        pending_stream_paths[:] = remaining
        return opened

    def _handle_door_frame(stream: dict, frame, video_time_str: str, real_time_str: str):
        nonlocal last_door_frame, last_door_analysis
        dataset_name = get_dataset_name(stream["path"])
        _, _, unified_events_path = _ensure_outputs(stream["path"])

        if CROP_REGION is not None:
            x1, y1, x2, y2 = CROP_REGION
            door_analysis = frame[y1:y2, x1:x2]
        else:
            door_analysis = frame

        prev_door_open = door_state["door_open"]
        raw_open = door_detect_step(detector, door_analysis, door_state)
        now_door_open = door_state["door_open"]
        door_open_window.append((real_time_str, now_door_open))

        if now_door_open != prev_door_open:
            event_type = "door_open" if now_door_open else "door_close"
            _append_csv(unified_events_path, UNIFIED_FIELDS, {
                "source": "door",
                "event_type": event_type,
                "video_time": video_time_str,
                "real_time": real_time_str,
                "video_name": os.path.basename(stream["path"]),
                "door_status": "OPEN" if now_door_open else "CLOSE",
            })
            sort_unified_csv(unified_events_path)
            log_event(f"  [Door] {'OPEN' if now_door_open else 'CLOSE'} @ {real_time_str}")

        last_door_frame = frame
        last_door_analysis = door_analysis
        update_ui("door", f"{real_time_str} | {'OPEN' if now_door_open else 'CLOSE'} | score={detector.last_score:.1f}")
        _process_pending_surgery_events()
        return raw_open, dataset_name

    def _handle_room_frame(stream: dict, frame, frame_idx: int, video_time_str: str, real_time_str: str):
        nonlocal last_room_frame, last_stored_all_count, total_room_frames_analyzed
        loop_start = time.time()
        stage_start = loop_start
        stage_times = []

        def _mark_stage(name: str):
            nonlocal stage_start
            now = time.time()
            stage_times.append((name, now - stage_start))
            stage_start = now

        dataset_name = get_dataset_name(stream["path"])
        surgery_date_tag = get_surgery_date(stream["path"])
        raw_csv_path, pair_report_path, unified_events_path = _ensure_outputs(stream["path"])
        _mark_stage("setup")

        status, _, infer_time = analyzer.analyze_frame(
            frame,
            "Surgery",
            full_frame=frame,
            current_sec=frame_idx / stream["fps"],
            current_frame=total_room_frames_analyzed,
            video_name=os.path.basename(stream["path"]),
            real_time=real_time_str,
        )
        _mark_stage("infer")
        pipeline.push_frame_result(
            status=status if getattr(analyzer, "push_to_pipeline", True) else 0,
            frame_idx=total_room_frames_analyzed,
            video_time=video_time_str,
            real_time=real_time_str,
            video_name=os.path.basename(stream["path"]),
        )

        voted = pipeline.voted_statuses[-1] if pipeline.voted_statuses else "-"
        state = pipeline.get_current_state()
        _mark_stage("pipeline")
        _get_raw_csv_writer(raw_csv_path).writerow([
            os.path.basename(stream["path"]),
            frame_idx,
            video_time_str,
            real_time_str,
            status,
            voted,
            f"{infer_time:.3f}",
            1 if door_state["door_open"] else 0,
            f"{detector.last_score:.3f}",
            f"{detector.last_ratio:.5f}",
        ])
        _mark_stage("csv")

        all_detected = pipeline.get_all_events()
        if len(all_detected) > last_stored_all_count:
            for evt in all_detected[last_stored_all_count:]:
                evt = dict(evt)
                evt["dataset_name"] = dataset_name
                evt["surgery_date_tag"] = surgery_date_tag
                evt["pair_report_path"] = pair_report_path
                evt["unified_events_path"] = unified_events_path
                pending_surgery_events.append(evt)
            last_stored_all_count = len(all_detected)
        _mark_stage("event_collect")

        last_room_frame = frame
        live_monitor.update(
            frame_idx=total_room_frames_analyzed,
            door_frame=last_door_frame,
            door_analysis=last_door_analysis,
            room_frame=last_room_frame,
            status={
                "dataset": dataset_name,
                "surgery_date": surgery_date_tag,
                "door_open": bool(door_state["door_open"]),
                "door_score": round(float(detector.last_score), 3),
                "door_ratio": round(float(detector.last_ratio), 5),
                "room_video": os.path.basename(stream["path"]),
                "video_time": video_time_str,
                "real_time": real_time_str,
                "status": status,
                "voted_status": voted,
                "confirmed_state_text": state["confirmed_state_text"],
                "infer_time": round(float(infer_time), 3),
            },
        )
        _mark_stage("live_monitor")
        total_room_frames_analyzed += 1
        _process_pending_surgery_events()
        _mark_stage("pending_events")
        loop_time = time.time() - loop_start
        sample_sec = stream["stride"] / stream["fps"] if stream["fps"] else STRIDE_SEC
        speed = sample_sec / loop_time if loop_time > 0 else 0.0
        if loop_time >= SLOW_LOOP_LOG_THRESHOLD_SEC:
            breakdown = ", ".join(f"{name}={elapsed:.3f}s" for name, elapsed in stage_times)
            log_event(
                f"[Slow Room Frame] total={loop_time:.3f}s frame={frame_idx} "
                f"real={real_time_str} video={os.path.basename(stream['path'])} | {breakdown}"
            )
        update_ui(
            "surgery",
            f"{real_time_str} | Door:{'OPEN' if door_state['door_open'] else 'CLOSE'} | "
            f"AI:{voted} ({state['confirmed_state_text']}) | "
            f"infer={infer_time:.2f}s loop={loop_time:.2f}s speed={speed:.2f}x",
        )

    def _process_stream_frame(stream: dict) -> bool:
        frame_idx = stream["next_frame_idx"]
        frame = _read_monotonic_frame(stream)
        if frame is None:
            stream["eof_retries"] += 1
            if not POLL_FOREVER:
                stream["done"] = True
                stream["cap"].release()
                log_event(f"[Polling] {stream['kind']}影片完成: {os.path.basename(stream['path'])}")
                _activate_next_streams()
                return False

            stream["retry_after"] = time.time() + POLL_INTERVAL_SEC
            update_ui(
                "door" if stream["kind"] == "Door" else "surgery",
                f"WAIT {stream['kind']} read failed/EOF | retry={stream['eof_retries']}/{STREAM_EOF_RETRIES} "
                f"sleep={POLL_INTERVAL_SEC:.1f}s | {os.path.basename(stream['path'])}",
            )
            return False

        stream["eof_retries"] = 0
        current_sec = frame_idx / stream["fps"]
        video_time_str = time.strftime("%H:%M:%S", time.gmtime(current_sec))
        real_time_str = (stream["start_dt"] + timedelta(seconds=current_sec)).strftime("%H:%M:%S")

        if stream["kind"] == "Door":
            _handle_door_frame(stream, frame, video_time_str, real_time_str)
        else:
            _handle_room_frame(stream, frame, frame_idx, video_time_str, real_time_str)

        stream["next_frame_idx"] += stream["stride"]
        return True

    _discover_new_streams()
    _activate_next_streams()
    last_discover_ts = time.time()
    idle_polls = 0

    try:
        while True:
            now = time.time()
            if now - last_discover_ts >= POLL_INTERVAL_SEC:
                _discover_new_streams()
                _activate_next_streams()
                last_discover_ts = now
            active_streams = [s for s in streams if not s["done"] and s["retry_after"] <= now]

            if not active_streams:
                new_count = _discover_new_streams()
                opened_count = _activate_next_streams()
                active_remaining = any(not s["done"] for s in streams) or bool(pending_stream_paths)
                if new_count == 0 and opened_count == 0 and not active_remaining:
                    idle_polls += 1
                else:
                    idle_polls = 0

                if not POLL_FOREVER and idle_polls >= POLL_IDLE_LIMIT:
                    break
                update_ui(
                    "surgery",
                    f"WAIT no active stream | queued={len(pending_stream_paths)} active={sum(not s['done'] for s in streams)} | "
                    f"idle={idle_polls}/{POLL_IDLE_LIMIT} sleep={POLL_INTERVAL_SEC:.1f}s",
                )
                time.sleep(POLL_INTERVAL_SEC)
                continue

            room_streams = [s for s in active_streams if s["kind"] == "Room"]
            if room_streams:
                room_stream = min(room_streams, key=_stream_next_abs_ts)
                door_target_ts = _stream_next_abs_ts(room_stream) + BUFFER_SECONDS

                while True:
                    now = time.time()
                    door_streams = [
                        s for s in streams
                        if s["kind"] == "Door" and not s["done"] and s["retry_after"] <= now
                    ]
                    if not door_streams:
                        break
                    door_stream = min(door_streams, key=_stream_next_abs_ts)
                    if _stream_next_abs_ts(door_stream) > door_target_ts:
                        break
                    if not _process_stream_frame(door_stream):
                        break

                _process_stream_frame(room_stream)
            else:
                stream = min(active_streams, key=_stream_next_abs_ts)
                _process_stream_frame(stream)
    finally:
        for stream in streams:
            if not stream["done"]:
                stream["cap"].release()
                stream["done"] = True
        for f in raw_csv_files.values():
            f.close()

    log_event("[Polling] 收尾 flush pipeline...")
    _process_pending_surgery_events(force=True)
    pipeline.flush()
    pipeline.force_close_pending_send()
    all_detected = pipeline.get_all_events()
    if len(all_detected) > last_stored_all_count:
        fallback_room = next((s for s in streams if s["kind"] == "Room"), None)
        if fallback_room is not None:
            dataset_name = get_dataset_name(fallback_room["path"])
            surgery_date_tag = get_surgery_date(fallback_room["path"])
            _, pair_report_path, unified_events_path = _ensure_outputs(fallback_room["path"])
            for evt in all_detected[last_stored_all_count:]:
                evt = dict(evt)
                evt["dataset_name"] = dataset_name
                evt["surgery_date_tag"] = surgery_date_tag
                evt["pair_report_path"] = pair_report_path
                evt["unified_events_path"] = unified_events_path
                pending_surgery_events.append(evt)
        _process_pending_surgery_events(force=True)

    for path in unified_event_paths:
        sort_unified_csv(path)
        print(f"  [排序完成] {os.path.basename(path)}")

    print("\n\n" + "=" * 70)
    print("  Polling 同步測試完成")
    for path in sorted(unified_event_paths):
        print(f"  統一事件 CSV: {path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
