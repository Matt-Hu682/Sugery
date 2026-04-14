# main_realtime.py
"""
即時模式入口 — 邊播本地影片邊做手術事件偵測

用法:
    python src/main_realtime.py

流程:
    1. 載入 AI 模型
    2. 開啟影片，逐幀分析
    3. 每幀分析結果即時推入 RealtimePipeline
    4. 畫面疊加即時狀態 (OSD)
    5. 影片結束時輸出最終事件報告 CSV
"""

import os
import sys
import shutil
import subprocess
import cv2
import csv
import time
import numpy as np
from datetime import datetime, timedelta

from config import (
    VIDEO_DIR, CSV_OUTPUT, MODEL_PATH,
    STRIDE_SEC, VIS_HEIGHT, VIS_WIDTH,
    CURRENT_TEST, TARGET_CAMERAS, SHOW_WINDOW, CROP_REGION,
    ROOM
)
from core import PatientStatusAnalyzer
from utils import video_start_time
from realtime_pipeline import RealtimePipeline

os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 使用哪張GPU


def draw_osd(frame, state, infer_time_ms): # 顯示在畫面上 預設FALSE
    """
    在影片畫面上疊加即時狀態資訊 (On-Screen Display)
    """
    h, w = frame.shape[:2]

    # --- 半透明背景 ---
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # --- 狀態顏色 ---
    if state['confirmed_state'] == 1:
        color = (0, 200, 255)  # 橘色 (BGR) — 手術中
        status_text = "手術中 (Surgery)"
    else:
        color = (0, 255, 0)    # 綠色 — 非手術
        status_text = "非手術 (Idle)"

    # --- 主狀態顯示 ---
    cv2.putText(frame, status_text, (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

    # --- 待確認候選 ---
    if state['pending']:
        cv2.putText(frame, f"[{state['pending']}]", (15, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    # --- 最新事件 ---
    if state['latest_event']:
        evt = state['latest_event']
        evt_text = f"Latest: {evt['event_type']} @ {evt['video_time']}"
        cv2.putText(frame, evt_text, (15, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # --- 右上角: 推論時間 & 投票進度 ---
    info_text = f"Infer: {infer_time_ms:.0f}ms | Voted: {state['voted_count']}/{state['raw_count']}"
    text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    cv2.putText(frame, info_text, (w - text_size[0] - 15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    # --- 事件計數 ---
    n_events = len(state['confirmed_events'])
    if n_events > 0:
        count_text = f"Events: {n_events}"
        cv2.putText(frame, count_text, (w - 150, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    return frame


def main():
    print("=" * 60)
    print("  即時手術偵測系統 (Real-Time Surgery Detection)")
    print("=" * 60)

    # --- 1. 初始化 AI 模型 ---
    analyzer = PatientStatusAnalyzer()

    # --- 2. 初始化即時 Pipeline ---
    # 根據任務類型調整投票視窗
    if CURRENT_TEST == "Door": 
        half_window = 10
    else:
        half_window = 25  # Surgery 用更大的視窗

    pipeline = RealtimePipeline( 
        half_window=half_window,
        stable_frame=900,
        max_gap_frame=50,
        send_confirm_threshold=900,
        task_type=CURRENT_TEST
    )



    # --- 3. 搜尋影片 ---
    if not os.path.exists(VIDEO_DIR):
        print(f"Path not found: {VIDEO_DIR}")
        return

    video_files = []
    for f in os.listdir(VIDEO_DIR):
        if f.lower().endswith(('.mp4', '.avi')):
            if any(cam in f for cam in TARGET_CAMERAS):
                video_files.append(os.path.join(VIDEO_DIR, f))

    video_files.sort() # 排序
    print(f"\n共找到 {len(video_files)} 支影片")
    print(f"任務模式: {CURRENT_TEST} | 投票視窗: {half_window * 2 + 1} 幀")
    print(f"按 'q' 隨時退出\n")

    # --- 4. 準備 CSV 輸出 ---
    run_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(os.path.dirname(CSV_OUTPUT), exist_ok=True)
    raw_csv_path = CSV_OUTPUT.replace('.csv', f'_{CURRENT_TEST}_{run_date}_realtime.csv')

    headers = ["Video_name", "frame_index", "video_time", "real_time",
            "status", "voted_status", "infer_time"]
    with open(raw_csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        csv.writer(f).writerow(headers)

    # --- 5. 逐支影片、逐幀即時分析 ---
    total_frames_analyzed = 0
    user_quit = False
    video_path_map = {}  # video_name -> full_path 對照表

    for vid_idx, video_path in enumerate(video_files):
        if user_quit:
            break

        video_name = os.path.basename(video_path)
        video_path_map[video_name] = video_path  # 記錄影片路徑
        print(f"\n [{vid_idx + 1}/{len(video_files)}] {video_name}")



        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        stride_frames = int(fps * STRIDE_SEC)

        start_dt = video_start_time(video_path)
        if start_dt is None:
            start_dt = datetime.now()

        frame_idx = 0
        while cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            # 計算時間
            current_sec = frame_idx / fps
            video_time_str = time.strftime('%H:%M:%S', time.gmtime(current_sec))
            real_time_dt = start_dt + timedelta(seconds=current_sec)
            real_time_str = real_time_dt.strftime("%H:%M:%S")  # 只要時間，不要日期

            # A9 裁切 (A8 不裁切，CROP_REGION=None)
            analysis_frame = frame
            if CROP_REGION is not None:
                x1, y1, x2, y2 = CROP_REGION
                analysis_frame = frame[y1:y2, x1:x2]

            # === AI 分析 (每幀直接送 VLM) ===
            status, vlm_vote, infer_time = analyzer.analyze_frame(analysis_frame, CURRENT_TEST, full_frame=frame, current_sec=current_sec, current_frame=total_frames_analyzed)

            # 推入即時 Pipeline
            if getattr(analyzer, 'push_to_pipeline', True):
                pipeline.push_frame_result(
                    status=status,
                    frame_idx=total_frames_analyzed,
                    video_time=video_time_str,
                    real_time=real_time_str,
                    video_name=video_name
                )
            else:
                # 不送進 Pipeline 的狀態（例如 Single 模式的門開了），對 Pipeline 來說一律維持 0
                pipeline.push_frame_result(
                    status=0,
                    frame_idx=total_frames_analyzed,
                    video_time=video_time_str,
                    real_time=real_time_str,
                    video_name=video_name
                )

            # 取得目前狀態
            state = pipeline.get_current_state()

            # 統一從 pipeline.voted_statuses 讀取 voted（Door/Surgery 皆同）
            voted = pipeline.voted_statuses[-1] if pipeline.voted_statuses else '-'

            # 寫入 CSV
            row = [video_name, frame_idx, video_time_str, real_time_str,
                status, voted, f"{infer_time:.3f}"]
            with open(raw_csv_path, 'a', newline='', encoding='utf-8-sig') as f:
                csv.writer(f).writerow(row)

            # OSD 疊加顯示 (需要有螢幕才開啟，遠端 SSH 請保持 SHOW_WINDOW=False)
            if SHOW_WINDOW:
                vis = cv2.resize(frame, (VIS_WIDTH, VIS_HEIGHT))
                vis = draw_osd(vis, state, infer_time * 1000)
                cv2.imshow("Real-Time Surgery Detection", vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n\n使用者手動退出")
                    user_quit = True
                    break

            # terminal 簡易顯示
            vlm_info = f" vlm={vlm_vote}" if CURRENT_TEST == "Door" and vlm_vote != '' else ""
            print(f"\r  {video_time_str} | raw={status} voted={voted}{vlm_info} | "
                f"state={state['confirmed_state_text']} | "
                f"events={len(state['confirmed_events'])}   ", end="")

            frame_idx += stride_frames
            total_frames_analyzed += 1

            if frame_idx >= total_frames:
                break

        cap.release()

    # --- 6. 刷出剩餘未投票的幀 ---
    pipeline.flush()

    if SHOW_WINDOW:
        cv2.destroyAllWindows()

    # --- 7. 輸出事件報告 ---
    print("\n" + "=" * 60)
    print("  分析完成！")
    print("=" * 60)

    summary = pipeline.get_event_summary()
    if summary:
        # result/手術日期_手術室_攝影機/ 資料夾結構
        # 從第一支影片檔名取手術日期 (例如 S01-20240103-... → 20240103)
        first_video = video_files[0] if video_files else ""
        try:
            surgery_date = os.path.basename(first_video).split('-')[1]
        except (IndexError, AttributeError):
            surgery_date = datetime.now().strftime("%Y%m%d")

        # 取目前使用的攝影機編號 (例如 ["S01"] → "S01")
        cam_label = TARGET_CAMERAS[0] if TARGET_CAMERAS else "unknown"
        folder_name = f"{surgery_date}_{ROOM}_{cam_label}"
        report_dir = os.path.join(os.path.dirname(os.path.dirname(CSV_OUTPUT)), "result", folder_name)
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, f"Realtime_Events_{CURRENT_TEST}_{run_date}.csv")

        with open(report_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=['Surgery_No', 'Type', 'Video_Time',
                                                'Real_Time', 'Video_Name'])
            writer.writeheader()
            writer.writerows(summary)

        print(f"\n 事件報告: {report_path}")
        print(f"   共偵測到 {len(summary) // 2} 組手術事件")

        # --- 8. 額外輸出：所有單獨偵測事件 (不論成對) ---
        all_events = pipeline.get_all_events()
        if all_events:
            all_events_dir = os.path.join(os.path.dirname(CSV_OUTPUT), "all_events")
            os.makedirs(all_events_dir, exist_ok=True)
            all_events_path = os.path.join(all_events_dir, f"All_Recognized_Events_{CURRENT_TEST}_{run_date}.csv")
            with open(all_events_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=['event_type', 'video_time', 'real_time', 'video_name'])
                writer.writeheader()
                writer.writerows(all_events)
            print(f" 所有偵測事件紀錄於: {all_events_path} (共 {len(all_events)} 筆)")

        # --- 9. 跨影片剪輯事件前後 3 分鐘的片段 ---
        video_output_dir = os.path.join(report_dir, "videos")
        os.makedirs(video_output_dir, exist_ok=True)

        # 建立所有影片的「絕對時間軸」清單 (從檔名解析 unix timestamp)
        def get_video_abs_ts(path):
            try:
                return int(os.path.basename(path).split('-')[3].split('.')[0])
            except:
                return None

        all_videos = []
        for vpath in sorted(video_path_map.values()):
            ts = get_video_abs_ts(vpath)
            if ts:
                all_videos.append({"path": vpath, "ts": ts})
        all_videos.sort(key=lambda x: x["ts"])

        PRE_POST_SEC = 180  # 事件前後各 3 分鐘

        for row in summary:
            vname = row['Video_Name']
            event_type = row['Type']
            surg_num = ''.join(filter(str.isdigit, row['Surgery_No']))
            event_real_time = row['Real_Time'].replace(':', '')
            video_time_str = row['Video_Time']

            # 組出輸出檔名
            parts_name = vname.split('-')
            surg_date_str = parts_name[1] if len(parts_name) >= 2 else surgery_date
            vid_time_str  = parts_name[2] if len(parts_name) >= 3 else "000000"
            dst_name = f"{surg_date_str}-{event_type}_{surg_num}_{event_real_time}_{vid_time_str}.mp4"
            dst_path = os.path.join(video_output_dir, dst_name)

            if os.path.exists(dst_path):
                print(f"   影片已存在: {dst_name}")
                continue

            # 計算事件的絕對時間 (unix)
            src_path = video_path_map.get(vname)
            if not src_path:
                print(f"   找不到影片: {vname}")
                continue

            try:
                vt_parts = video_time_str.split(':')
                event_offset = int(vt_parts[0])*3600 + int(vt_parts[1])*60 + int(vt_parts[2])
                src_abs_ts = get_video_abs_ts(src_path)
                event_abs_ts = src_abs_ts + event_offset
                target_start_ts = event_abs_ts - PRE_POST_SEC
                target_end_ts   = event_abs_ts + PRE_POST_SEC

                writer = None
                written_frames = 0
                opened_vids = 0

                # 掃描所有影片，找時間軸有交集的
                for v in all_videos:
                    cap_c = cv2.VideoCapture(v["path"])
                    fps_c = cap_c.get(cv2.CAP_PROP_FPS) or 5.0
                    total_f = int(cap_c.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration_c = total_f / fps_c
                    vid_s = v["ts"]
                    vid_e = vid_s + duration_c

                    if vid_e < target_start_ts or vid_s > target_end_ts:
                        cap_c.release()
                        continue

                    opened_vids += 1
                    if writer is None:
                        cw = int(cap_c.get(cv2.CAP_PROP_FRAME_WIDTH))
                        ch = int(cap_c.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        writer = cv2.VideoWriter(dst_path, fourcc, fps_c, (cw, ch))

                    read_s = max(0, target_start_ts - vid_s)
                    read_e = min(duration_c, target_end_ts - vid_s)
                    cap_c.set(cv2.CAP_PROP_POS_FRAMES, int(read_s * fps_c))

                    while cap_c.isOpened():
                        ret_c, frm_c = cap_c.read()
                        if not ret_c: break
                        if cap_c.get(cv2.CAP_PROP_POS_FRAMES) > int(read_e * fps_c): break
                        writer.write(frm_c)
                        written_frames += 1

                    cap_c.release()

                if writer:
                    writer.release()
                    print(f"   影片已剪輯: {dst_name} (橫跨 {opened_vids} 支影片, {written_frames} 幀)")
                else:
                    print(f"   找不到對應時間範圍的影片: {dst_name}")

            except Exception as e:
                print(f"   剪輯失敗: {dst_name} ({e})")
    else:
        print("\n未偵測到成對的手術事件。")

    print(f"\n 原始資料 CSV: {raw_csv_path}")
    print(f"   共分析 {total_frames_analyzed} 幀")


if __name__ == "__main__":
    main()
