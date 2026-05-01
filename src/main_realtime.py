# main_realtime.py
import os
import csv
import time
from datetime import datetime, timedelta

import cv2

from config import (
    CROP_REGION,
    CSV_OUTPUT,
    CURRENT_TEST,
    ROOM,
    SHOW_WINDOW,
    STRIDE_SEC,
    TARGET_CAMERAS,
    VIDEO_DIR,
    VIDEO_DIRS,
    VIS_HEIGHT,
    VIS_WIDTH,
)
from core import PatientStatusAnalyzer
from realtime_pipeline import RealtimePipeline
from utils import video_start_time

# 設定gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def draw_osd(frame, state, infer_time_ms):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    if state['confirmed_state'] == 1:
        color = (0, 200, 255)
        status_text = "手術中 (Surgery)"
    else:
        color = (0, 255, 0)
        status_text = "非手術 (Idle)"

    cv2.putText(frame, status_text, (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

    if state['pending']:
        cv2.putText(frame, f"[{state['pending']}]", (15, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    if state['latest_event']:
        evt = state['latest_event']
        evt_text = f"Latest: {evt['event_type']} @ {evt['video_time']}"
        cv2.putText(frame, evt_text, (15, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    info_text = f"Infer: {infer_time_ms:.0f}ms | Voted: {state['voted_count']}/{state['raw_count']}"
    text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    cv2.putText(frame, info_text, (w - text_size[0] - 15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

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

    half_window = 10 if CURRENT_TEST == "Door" else 25

    pipeline = RealtimePipeline( 
        half_window=half_window,
        stable_frame=900,
        max_gap_frame=50,
        send_confirm_threshold=900,
        task_type=CURRENT_TEST
    )

    dataset_dirs = VIDEO_DIRS if VIDEO_DIRS else [VIDEO_DIR]
    dataset_dirs = [os.path.abspath(path) for path in dataset_dirs]

    missing_dirs = [path for path in dataset_dirs if not os.path.exists(path)]
    if missing_dirs:
        for path in missing_dirs:
            print(f"Path not found: {path}")
        return

    video_groups = []
    for dataset_dir in dataset_dirs:
        dataset_videos = []
        for f in os.listdir(dataset_dir):
            if f.lower().endswith(('.mp4', '.avi')) and any(cam in f for cam in TARGET_CAMERAS):
                dataset_videos.append(os.path.join(dataset_dir, f))
        dataset_videos.sort()
        if dataset_videos:
            video_groups.append((dataset_dir, dataset_videos))

    video_files = [video_path for _, videos in video_groups for video_path in videos]
    print(f"\n共找到 {len(dataset_dirs)} 個資料集資料夾")
    for dataset_dir in dataset_dirs:
        print(f"  - {dataset_dir}")
    print(f"\n共找到 {len(video_files)} 支影片")
    print(f"任務模式: {CURRENT_TEST} | 投票視窗: {half_window * 2 + 1} 幀")
    print(f"按 'q' 隨時退出\n")

    run_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(os.path.dirname(CSV_OUTPUT), exist_ok=True)
    headers = ["Video_name", "frame_index", "video_time", "real_time", "status", "voted_status", "infer_time"]
    user_quit = False
    analyzer = PatientStatusAnalyzer()

    for group_idx, (dataset_dir, dataset_video_files) in enumerate(video_groups, start=1):
        if user_quit:
            break

        dataset_name = os.path.basename(dataset_dir)
        print(f"\n{'=' * 60}")
        print(f"資料集 [{group_idx}/{len(video_groups)}]: {dataset_name}")
        print(f"影片數量: {len(dataset_video_files)}")
        print(f"{'=' * 60}")

        analyzer.reset_runtime_state()
        pipeline = RealtimePipeline(
            half_window=half_window,
            stable_frame=900,
            max_gap_frame=50,
            send_confirm_threshold=900,
            task_type=CURRENT_TEST
        )

        raw_csv_path = os.path.join(
            os.path.dirname(CSV_OUTPUT),
            dataset_name,
            f"surgery_report_{CURRENT_TEST}_{run_date}_realtime.csv",
        )
        os.makedirs(os.path.dirname(raw_csv_path), exist_ok=True)
        with open(raw_csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            csv.writer(f).writerow(headers)

        first_video = dataset_video_files[0] if dataset_video_files else ""
        try:
            surgery_date = os.path.basename(first_video).split('-')[1]
        except:
            surgery_date = dataset_name

        cam_label = TARGET_CAMERAS[0] if TARGET_CAMERAS else "unknown"
        folder_name = f"{surgery_date}_{ROOM}_{cam_label}"
        report_dir = os.path.join(os.path.dirname(os.path.dirname(CSV_OUTPUT)), "result", folder_name, dataset_name)
        os.makedirs(report_dir, exist_ok=True)

        pair_report_path = os.path.join(report_dir, f"Realtime_Events_{CURRENT_TEST}_{run_date}.csv")
        all_events_dir = os.path.join(os.path.dirname(raw_csv_path), "all_events")
        os.makedirs(all_events_dir, exist_ok=True)
        all_events_path = os.path.join(all_events_dir, f"All_Recognized_Events_{CURRENT_TEST}_{run_date}.csv")

        with open(pair_report_path, 'w', newline='', encoding='utf-8-sig') as f:
            csv.DictWriter(f, fieldnames=['Surgery_No', 'Type', 'Video_Time', 'Real_Time', 'Video_Name']).writeheader()
        with open(all_events_path, 'w', newline='', encoding='utf-8-sig') as f:
            csv.DictWriter(f, fieldnames=['event_type', 'video_time', 'real_time', 'video_name']).writeheader()

        total_frames_analyzed = 0
        video_path_map = {}
        last_stored_all_count = 0
        last_stored_pair_count = 0

        for vid_idx, video_path in enumerate(dataset_video_files, start=1):
            if user_quit:
                break

            video_name = os.path.basename(video_path)
            video_path_map[video_name] = video_path
            print(f"\n [{vid_idx}/{len(dataset_video_files)}] [{dataset_name}] {video_name}")

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

                current_sec = frame_idx / fps
                video_time_str = time.strftime('%H:%M:%S', time.gmtime(current_sec))
                real_time_dt = start_dt + timedelta(seconds=current_sec)
                real_time_str = real_time_dt.strftime("%H:%M:%S")

                analysis_frame = frame
                if CROP_REGION is not None:
                    x1, y1, x2, y2 = CROP_REGION
                    analysis_frame = frame[y1:y2, x1:x2]

                status, vlm_vote, infer_time = analyzer.analyze_frame(
                    analysis_frame,
                    CURRENT_TEST,
                    full_frame=frame,
                    current_sec=current_sec,
                    current_frame=total_frames_analyzed,
                    video_name=video_name,
                    real_time=real_time_str,
                )

                push_frame_idx = total_frames_analyzed
                push_video_time = video_time_str
                push_real_time = real_time_str
                push_video_name = video_name

                if CURRENT_TEST == "Door" and status in (2, 3):
                    override_meta = analyzer.pop_event_metadata_override()
                    if override_meta is not None:
                        push_frame_idx = override_meta.get("frame_idx", push_frame_idx)
                        push_video_time = override_meta.get("video_time", push_video_time)
                        push_real_time = override_meta.get("real_time", push_real_time)
                        push_video_name = override_meta.get("video_name", push_video_name)

                if getattr(analyzer, 'push_to_pipeline', True):
                    pipeline.push_frame_result(
                        status=status,
                        frame_idx=push_frame_idx,
                        video_time=push_video_time,
                        real_time=push_real_time,
                        video_name=push_video_name
                    )
                else:
                    pipeline.push_frame_result(
                        status=0,
                        frame_idx=total_frames_analyzed,
                        video_time=video_time_str,
                        real_time=real_time_str,
                        video_name=video_name
                    )

                state = pipeline.get_current_state()

                if CURRENT_TEST == "Door":
                    voted = vlm_vote
                else:
                    voted = pipeline.voted_statuses[-1] if pipeline.voted_statuses else '-'

                row = [video_name, frame_idx, video_time_str, real_time_str, status, voted, f"{infer_time:.3f}"]
                with open(raw_csv_path, 'a', newline='', encoding='utf-8-sig') as f:
                    csv.writer(f).writerow(row)

                all_detected = pipeline.get_all_events()
                if len(all_detected) > last_stored_all_count:
                    new_evts = all_detected[last_stored_all_count:]
                    with open(all_events_path, 'a', newline='', encoding='utf-8-sig') as af:
                        writer = csv.DictWriter(af, fieldnames=['event_type', 'video_time', 'real_time', 'video_name'])
                        writer.writerows(new_evts)
                    last_stored_all_count = len(all_detected)

                summary = pipeline.get_event_summary()
                if len(summary) > last_stored_pair_count:
                    new_pairs = summary[last_stored_pair_count:]
                    with open(pair_report_path, 'a', newline='', encoding='utf-8-sig') as pf:
                        writer = csv.DictWriter(pf, fieldnames=['Surgery_No', 'Type', 'Video_Time', 'Real_Time', 'Video_Name'])
                        writer.writerows(new_pairs)
                    last_stored_pair_count = len(summary)

                if SHOW_WINDOW:
                    vis = cv2.resize(frame, (VIS_WIDTH, VIS_HEIGHT))
                    vis = draw_osd(vis, state, infer_time * 1000)
                    cv2.imshow("Real-Time Surgery Detection", vis)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\n\n使用者手動退出")
                        user_quit = True
                        break

                vlm_info = f" vlm={vlm_vote}" if CURRENT_TEST == "Door" and vlm_vote != '' else ""
                print(f"\r  {video_time_str} | raw={status} voted={voted}{vlm_info} | "
                    f"state={state['confirmed_state_text']} | "
                    f"events={len(state['confirmed_events'])}   ", end="")

                frame_idx += stride_frames
                total_frames_analyzed += 1

                if frame_idx >= total_frames:
                    break

            cap.release()

        pipeline.flush()

        if SHOW_WINDOW:
            cv2.destroyAllWindows()

        print("\n" + "=" * 60)
        print(f"  資料集 {dataset_name} 分析完成！")
        print(f"  原始資料: {raw_csv_path}")
        print(f"  所有偵測紀錄: {all_events_path}")
        print(f"  成對事件報告: {pair_report_path}")
        print("=" * 60)

        summary = pipeline.get_event_summary()
        if summary:
            print(f"   共偵測到 {len(summary) // 2} 組手術事件。正在進行影片剪輯...")
            video_output_dir = os.path.join(report_dir, "videos")
            os.makedirs(video_output_dir, exist_ok=True)

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

            PRE_POST_SEC = 180 

            for row in summary:
                vname = row['Video_Name']
                event_type = row['Type']
                surg_num = ''.join(filter(str.isdigit, row['Surgery_No']))
                event_real_time = row['Real_Time'].replace(':', '')
                video_time_str = row['Video_Time']

                dst_name = f"{surgery_date}-{event_type}_{surg_num}_{event_real_time}.mp4"
                dst_path = os.path.join(video_output_dir, dst_name)

                if os.path.exists(dst_path):
                    continue

                src_path = video_path_map.get(vname)
                if not src_path:
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
                    for v in all_videos:
                        cap_c = cv2.VideoCapture(v["path"])
                        fps_c = cap_c.get(cv2.CAP_PROP_FPS) or 5.0
                        total_f = int(cap_c.get(cv2.CAP_PROP_FRAME_COUNT))
                        duration_c = total_f / fps_c
                        vid_s = v["ts"]; vid_e = vid_s + duration_c
                        if vid_e < target_start_ts or vid_s > target_end_ts:
                            cap_c.release()
                            continue

                        if writer is None:
                            cw = int(cap_c.get(cv2.CAP_PROP_FRAME_WIDTH))
                            ch = int(cap_c.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            writer = cv2.VideoWriter(dst_path, cv2.VideoWriter_fourcc(*'mp4v'), fps_c, (cw, ch))

                        read_s = max(0, target_start_ts - vid_s)
                        read_e = min(duration_c, target_end_ts - vid_s)
                        cap_c.set(cv2.CAP_PROP_POS_FRAMES, int(read_s * fps_c))
                        while cap_c.isOpened():
                            ret_c, frm_c = cap_c.read()
                            if not ret_c or cap_c.get(cv2.CAP_PROP_POS_FRAMES) > int(read_e * fps_c):
                                break
                            writer.write(frm_c)
                            written_frames += 1
                        cap_c.release()
                    if writer:
                        writer.release()
                except Exception as e:
                    print(f"   剪輯失敗: {dst_name} ({e})")
        else:
            print(f"\n資料集 {dataset_name} 未偵測到成對的手術事件。")

if __name__ == "__main__":
    main()
