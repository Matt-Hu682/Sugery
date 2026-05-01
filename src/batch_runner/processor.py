# multi_gpu 處理指令碼 (即時模式 — 與 main_realtime.py 一致)
import os
import sys
import time
import subprocess
import csv
import cv2
from multiprocessing import Process, Queue
from datetime import datetime, timedelta
from pathlib import Path

# 新增 batch_runner 和 src 路徑
batch_runner_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(batch_runner_dir)
project_root = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# 絕對導入
from batch_runner.config import (
    PROCESS_DATES,
    TEST_DATE,
    GPU_ALLOCATION,
    get_video_dir_for_date,
    get_gpu_for_date,
    get_csv_output_for_date,
    CURRENT_TEST,
    TARGET_CAMERAS,
    STRIDE_SEC,
    SHOW_WINDOW,
    ROOM,
    print_config,
    OUTPUTS_DIR,
    DATA_BASE_DIR,
    NAS_OUTPUT_BASE,
    GPU_IDS,
    CROP_REGION
)


def process_dates_for_gpu(date_list, gpu_id, queue):
    """
    為指定的 GPU 順序處理日期列表（即時模式）
    每個日期在同一進程中順序進行。
    分析每一幀後立即推入 RealtimePipeline 做即時投票和事件偵測，
    與 src/main_realtime.py 的處理方式完全一致。
    
    Args:
        date_list: 這個 GPU 要處理的日期列表
        gpu_id: GPU ID (MIG UUID 或數字)
        queue: 用於傳回結果的佇列
    """
    print(f"\n🔧 GPU 行程啟動 (ID: {gpu_id}), 分配 {len(date_list)} 個日期")
    print(f"   日期: {', '.join(date_list)}")
    
    # 新增虛擬環境 site-packages 到 sys.path
    venv_site_packages = "/home/ai/Sugery_AI/VLM-1/lib/python3.10/site-packages"
    if venv_site_packages not in sys.path:
        sys.path.insert(0, venv_site_packages)
    
    # 設定此進程的 GPU（MIG 設備 UUID 或數字 ID）
    if isinstance(gpu_id, str) and gpu_id.startswith("MIG-"):
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # 重要：在子進程中設置正確的 config 模組，讓 core.py 能找到
    from batch_runner import config as batch_runner_config
    sys.modules['config'] = batch_runner_config
    
    # 動態匯入，避免在主進程中載入 CUDA
    from core import PatientStatusAnalyzer
    from realtime_pipeline import RealtimePipeline
    from utils import video_start_time
    
    # 載入模型（移到迴圈外，每個 GPU 獨立行程只載入一次）
    try:
        print(f"⏳ 載入模型...")
        analyzer = PatientStatusAnalyzer()
        print(f"✓ 模型載入成功")
    except Exception as e:
        print(f"❌ 模型載入失敗: {e}")
        queue.put({
            "date": "init",
            "status": "failed",
            "message": f"Model load failed: {e}"
        })
        return
    
    for date_idx, date_str in enumerate(date_list):
        print(f"\n[GPU{gpu_id} - {date_idx+1}/{len(date_list)}] 開始處理: {date_str}")
        print("-" * 70)
        
        video_dir = get_video_dir_for_date(date_str)
        csv_output = get_csv_output_for_date(date_str)
        
        # 檢查目錄
        if not os.path.exists(video_dir):
            print(f"❌ 目錄不存在: {video_dir}")
            queue.put({
                "date": date_str,
                "status": "failed",
                "message": "Directory not found"
            })
            continue
        
        # 掃描視訊檔案
        video_files = []
        for f in os.listdir(video_dir):
            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                if any(cam in f for cam in TARGET_CAMERAS):
                    video_files.append(os.path.join(video_dir, f))
        
        video_files.sort()
        
        if not video_files:
            print(f"⚠️  未找到符合條件的視訊")
            queue.put({
                "date": date_str,
                "status": "no_videos",
                "count": 0
            })
            continue
        
        print(f"📹 找到 {len(video_files)} 個視訊檔案")
        
        # === 初始化即時 Pipeline (與 main_realtime.py 相同) ===
        half_window = 10 if CURRENT_TEST == "Door" else 25
        pipeline = RealtimePipeline(
            half_window=half_window,
            stable_frame=900,
            max_gap_frame=50,
            send_confirm_threshold=900,
            task_type=CURRENT_TEST
        )
        
        # === 準備 CSV 輸出 (含 voted_status，與 main_realtime.py 格式一致) ===
        run_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(os.path.dirname(csv_output), exist_ok=True)
        dir_name = os.path.dirname(csv_output)
        file_name, file_ext = os.path.splitext(os.path.basename(csv_output))
        actual_csv_path = os.path.join(dir_name, f"{file_name}_{CURRENT_TEST}_{run_date}{file_ext}")
        
        headers = ["Video_name", "frame_index", "video_time", "real_time",
                    "status", "voted_status", "infer_time"]
        with open(actual_csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            csv.writer(f).writerow(headers)
        
        print(f"📊 即時模式: 投票視窗 {half_window * 2 + 1} 幀")
        print(f"📄 CSV 輸出: {actual_csv_path}")
        
        # === 準備事件報告路徑與 Headers (供 Streaming Write 使用) ===
        cam_label = TARGET_CAMERAS[0] if TARGET_CAMERAS else "unknown"
        # 從 DATA_BASE_DIR 提取資料夾名稱末尾的月份，例如 mask_video_202312 -> 202312
        data_month = os.path.basename(DATA_BASE_DIR).split('_')[-1]
        folder_name = f"{date_str}_{ROOM}_{cam_label}"
        report_dir = os.path.join(os.path.dirname(OUTPUTS_DIR), "result", data_month, folder_name)
        os.makedirs(report_dir, exist_ok=True)
        
        pair_report_path = os.path.join(report_dir, f"Realtime_Events_{CURRENT_TEST}_{run_date}.csv")
        
        all_events_dir = os.path.join(os.path.dirname(OUTPUTS_DIR), "all_events", data_month, folder_name)
        os.makedirs(all_events_dir, exist_ok=True)
        all_events_path = os.path.join(all_events_dir, f"All_Recognized_Events_{CURRENT_TEST}_{run_date}.csv")
        
        with open(pair_report_path, 'w', newline='', encoding='utf-8-sig') as f:
            csv.DictWriter(f, fieldnames=['Surgery_No', 'Type', 'Video_Time', 'Real_Time', 'Video_Name']).writeheader()
        with open(all_events_path, 'w', newline='', encoding='utf-8-sig') as f:
            csv.DictWriter(f, fieldnames=['event_type', 'video_time', 'real_time', 'video_name']).writeheader()
            
        last_stored_all_count = 0
        last_stored_pair_count = 0
        
        # === 逐支影片、逐幀即時分析 (與 main_realtime.py 邏輯一致) ===
        start_time = time.time()
        total_frames_analyzed = 0
        processed_videos = 0
        failed_videos = 0
        video_path_map = {}  # video_name -> full_path 對照表
        
        for vid_idx, video_path in enumerate(video_files):
            video_name = os.path.basename(video_path)
            video_path_map[video_name] = video_path
            print(f"\n   [{vid_idx+1}/{len(video_files)}] ▶️  {video_name}")
            
            try:
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
                    real_time_str = real_time_dt.strftime("%H:%M:%S")
                    
                    # 裁切 (A9 需要裁切，A8 不需要)
                    analysis_frame = frame
                    if CROP_REGION is not None:
                        x1, y1, x2, y2 = CROP_REGION
                        analysis_frame = frame[y1:y2, x1:x2]
                    
                    # === AI 分析 (每幀直接送 VLM) ===
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
                    
                    # === 即時推入 Pipeline ===
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
                    
                    # 將實際階段資訊寫入 CSV 的 voted_status 欄位
                    if CURRENT_TEST == "Door":
                        voted = vlm_vote
                    else:
                        voted = pipeline.voted_statuses[-1] if pipeline.voted_statuses else '-'
                    
                    # 寫入每一幀 CSV
                    row = [video_name, frame_idx, video_time_str, real_time_str,
                        status, voted, f"{infer_time:.3f}"]
                    with open(actual_csv_path, 'a', newline='', encoding='utf-8-sig') as f_csv:
                        csv.writer(f_csv).writerow(row)
                        
                    # === 即時即寫 (Streaming Write) ===
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
                    
                    # terminal 即時顯示 (與 main_realtime.py 格式一致)
                    state = pipeline.get_current_state()
                    
                    # 取得當前使用的模式 (如果是 Surgery 則固定為 Single)
                    mode_str = getattr(analyzer, 'current_mode', 'Single')
                    
                    print(f"\r     {video_time_str} | [{mode_str[:3]}] raw={status} voted={voted} | "
                        f"state={state['confirmed_state_text']} | "
                        f"events={len(state['confirmed_events'])}   ", end="")
                    
                    frame_idx += stride_frames
                    total_frames_analyzed += 1
                    
                    if frame_idx >= total_frames:
                        break
                
                cap.release()
                print(f" ✓")
                processed_videos += 1
                
            except Exception as e:
                print(f" ✗ ({str(e)[:50]})")
                failed_videos += 1
                continue
        
        # === 刷出剩餘未投票的幀 (與 main_realtime.py 相同) ===
        pipeline.flush()
        
        elapsed = time.time() - start_time
        print(f"\n✅ {date_str} 處理完成 - {processed_videos}個影片成功, {failed_videos}個失敗, "
            f"共 {total_frames_analyzed} 幀, 耗時 {elapsed/60:.1f}分鐘")
        
        queue.put({
            "date": date_str,
            "status": "completed",
            "processed": processed_videos,
            "failed": failed_videos,
            "elapsed": elapsed,
            "csv_output": actual_csv_path
        })
        
        # === 剪輯影片 ===
        print(f"\n🎬 開始處理影片剪輯...")
        generate_report_and_clip_videos(pipeline, date_str, video_path_map)


def generate_report_and_clip_videos(pipeline, date_str, video_path_map):
    """
    從即時 Pipeline 直接取得事件摘要，產生報告並剪輯影片。
    不再從 CSV 重新讀取，與 main_realtime.py 的結果產出方式完全一致。
    
    Args:
        pipeline: 已完成分析的 RealtimePipeline 實例
        date_str: 日期字串
        video_path_map: {video_name: full_path} 對照表
    """
    # 取得事件摘要 (直接從 Pipeline 記憶體中取得)
    summary = pipeline.get_event_summary()
    if not summary:
        print(f"[影片剪輯] 未偵測到成對的事件，無需剪輯")
        return
    
    # 建立事件報告目錄 (影片存在這裡，依據月份分類)
    cam_label = TARGET_CAMERAS[0] if TARGET_CAMERAS else "unknown"
    data_month = os.path.basename(DATA_BASE_DIR).split('_')[-1]
    folder_name = f"{date_str}_{ROOM}_{cam_label}"
    report_dir = os.path.join(os.path.dirname(OUTPUTS_DIR), "result", data_month, folder_name)
    os.makedirs(report_dir, exist_ok=True)

    # --- 跨影片剪輯事件前後 3 分鐘的片段 ---
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
    clipped_count = 0

    for row in summary:
        vname = row['Video_Name']
        event_type = row['Type']
        surg_num = ''.join(filter(str.isdigit, row['Surgery_No']))
        event_real_time = row['Real_Time'].replace(':', '')
        video_time_str = row['Video_Time']

        dst_name = f"{date_str}-{event_type}_{surg_num}_{event_real_time}.mp4"
        dst_path = os.path.join(video_output_dir, dst_name)

        if os.path.exists(dst_path):
            print(f"[影片剪輯] ⊘ 影片已存在: {dst_name}")
            continue

        src_path = video_path_map.get(vname)
        if not src_path: 
            print(f"[影片剪輯] 找不到來源影片: {vname}")
            continue

        try:
            vt_parts = video_time_str.split(':')
            event_offset = int(vt_parts[0])*3600 + int(vt_parts[1])*60 + int(vt_parts[2])
            src_abs_ts = get_video_abs_ts(src_path)
            
            if not src_abs_ts:
                print(f"[影片剪輯] 無法解析時間戳: {vname} (改用備用邏輯...)")
                continue
                
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
                vid_s = v["ts"]
                vid_e = vid_s + duration_c
                
                if vid_e <= target_start_ts or vid_s >= target_end_ts:
                    cap_c.release()
                    continue

                if writer is None:
                    cw = int(cap_c.get(cv2.CAP_PROP_FRAME_WIDTH))
                    ch = int(cap_c.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    writer = cv2.VideoWriter(dst_path, cv2.VideoWriter_fourcc(*'mp4v'), fps_c, (cw, ch))

                # 計算這支影片要讀取的範圍
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
                print(f"[影片剪輯] ✓ 影片已跨檔合併: {dst_name}")
                clipped_count += 1
            else:
                print(f"[影片剪輯] ✗ 合併失敗 (範圍內無畫格): {dst_name}")
                
        except Exception as e:
            print(f"[影片剪輯] ✗ 剪輯異常: {dst_name} ({e})")
            
    print(f"[影片剪輯] 完成！共精準剪出 {clipped_count} 個跨影音片段")


def process_dates_on_gpus():
    """
    2 個 GPU 分工：GPU0 處理前一半日期，GPU1 處理後一半日期
    每個 GPU 內部順序處理分配給它的日期
    """
    if not PROCESS_DATES:
        print("\n⚠️  沒有需要處理的日期")
        return
    
    # 顯示設定
    print_config()
    
    # 確保 NAS 輸出目錄存在
    print("\n🔧 檢查 NAS 輸出目錄...")
    try:
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        print(f"✓ outputs 目錄: {OUTPUTS_DIR}")
    except Exception as e:
        print(f"❌ 無法建立 outputs 目錄: {e}")
        print(f"💡 請檢查 NAS 權限或手動建立目錄")
        return
    
    try:
        result_dir = os.path.join(NAS_OUTPUT_BASE, "result")
        os.makedirs(result_dir, exist_ok=True)
        print(f"✓ result 目錄: {result_dir}")
    except Exception as e:
        print(f"❌ 無法建立 result 目錄: {e}")
        print(f"💡 請檢查 NAS 權限或手動建立目錄")
        return
    
    # 按 GPU 分組日期（根據實際 GPU_ALLOCATION）
    gpu_dates_map = {}
    for date in PROCESS_DATES:
        gpu_id = GPU_ALLOCATION.get(date)
        if gpu_id not in gpu_dates_map:
            gpu_dates_map[gpu_id] = []
        gpu_dates_map[gpu_id].append(date)
    
    print(f"\n🔄 啟動多 GPU 分工處理 (即時模式)...")
    print(f"   GPU 數量: {len(gpu_dates_map)}")
    for gpu_id, dates in gpu_dates_map.items():
        gpu_label = gpu_id if isinstance(gpu_id, str) and gpu_id.startswith("MIG-") else f"GPU{gpu_id}"
        print(f"   {gpu_label}: {len(dates)} 個日期")
    print(f"   模式: 即時投票 + 各 GPU 順序處理自己的日期")
    print("-" * 70)
    
    # 建立進程（動態支援多個 GPU）
    processes = []
    queue = Queue()
    
    for gpu_id, dates in gpu_dates_map.items():
        if dates:
            gpu_label = gpu_id if isinstance(gpu_id, str) and gpu_id.startswith("MIG-") else f"GPU{gpu_id}"
            p = Process(
                target=process_dates_for_gpu,
                args=(dates, gpu_id, queue),
                name=f"Process-{gpu_label}"
            )
            processes.append((p, gpu_label))
            p.start()
    
    print(f"✓ 已啟動 {len(processes)} 個行程\n")
    
    # 等待所有行程完成
    results = {}
    for p, gpu_name in processes:
        p.join()
    
    # 收集佇列中的結果
    while not queue.empty():
        result = queue.get()
        results[result['date']] = result
    
    # 匯總結果
    print("\n" + "=" * 70)
    print("📋 所有日期處理完成")
    print("=" * 70)
    
    for date_str in PROCESS_DATES:
        result = results.get(date_str, {})
        status = result.get('status', 'unknown')
        
        if status == "completed":
            print(f"✅ {date_str}: {result['processed']}個成功, {result['failed']}個失敗")
        elif status == "no_videos":
            print(f"⚠️  {date_str}: 未找到視訊")
        else:
            print(f"❌ {date_str}: {result.get('message', '失敗')}")
    
    print("=" * 70)
    print("✅ 所有處理完成！")
    print("=" * 70)


def main():
    """主函數"""
    process_dates_on_gpus()


if __name__ == "__main__":
    main()
