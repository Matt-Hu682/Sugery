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
    
    for date_idx, date_str in enumerate(date_list):
        print(f"\n[GPU{gpu_id} - {date_idx+1}/{len(date_list)}] 開始處理: {date_str}")
        print("-" * 70)
        
        # 動態匯入，避免在主進程中載入 CUDA
        from core import PatientStatusAnalyzer
        from realtime_pipeline import RealtimePipeline
        from utils import video_start_time
        
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
        
        # 載入模型
        try:
            print(f"⏳ 載入模型...")
            analyzer = PatientStatusAnalyzer()
            print(f"✓ 模型載入成功")
        except Exception as e:
            print(f"❌ 模型載入失敗: {e}")
            queue.put({
                "date": date_str,
                "status": "failed",
                "message": f"Model load failed: {e}"
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
                    status, infer_time = analyzer.analyze_frame(analysis_frame, CURRENT_TEST)
                    
                    # === 即時推入 Pipeline (與 main_realtime.py 完全相同) ===
                    pipeline.push_frame_result(
                        status=status,
                        frame_idx=total_frames_analyzed,
                        video_time=video_time_str,
                        real_time=real_time_str,
                        video_name=video_name
                    )
                    
                    # 取得最新的 voted_status
                    voted = pipeline.voted_statuses[-1] if pipeline.voted_statuses else '-'
                    
                    # 寫入 CSV (含 voted_status)
                    row = [video_name, frame_idx, video_time_str, real_time_str,
                        status, voted, f"{infer_time:.3f}"]
                    with open(actual_csv_path, 'a', newline='', encoding='utf-8-sig') as f_csv:
                        csv.writer(f_csv).writerow(row)
                    
                    # terminal 即時顯示 (與 main_realtime.py 格式一致)
                    state = pipeline.get_current_state()
                    print(f"\r     {video_time_str} | raw={status} voted={voted} | "
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
        
        # === 直接從 Pipeline 產生事件報告 + 剪輯影片 ===
        print(f"\n🎬 開始處理事件報告和剪輯影片...")
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
        print(f"[事件處理] 未偵測到成對的事件")
        return
    
    print(f"[事件處理] 偵測到 {len(summary) // 2} 組事件")
    
    # 建立事件報告目錄
    cam_label = TARGET_CAMERAS[0] if TARGET_CAMERAS else "unknown"
    folder_name = f"{date_str}_{ROOM}_{cam_label}"
    report_dir = os.path.join(os.path.dirname(OUTPUTS_DIR), "result", folder_name)
    os.makedirs(report_dir, exist_ok=True)
    
    # 輸出事件報告 CSV (檔名格式與 main_realtime.py 一致: Realtime_Events_...)
    event_csv_path = os.path.join(
        report_dir,
        f"Realtime_Events_{CURRENT_TEST}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    try:
        with open(event_csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(
                f, fieldnames=['Surgery_No', 'Type', 'Video_Time', 'Real_Time', 'Video_Name']
            )
            writer.writeheader()
            writer.writerows(summary)
        print(f"[事件處理] ✓ 事件報告已儲存: {event_csv_path}")
    except Exception as e:
        print(f"[事件處理] ✗ 事件報告寫入失敗: {e}")
        return
    
    # 列印事件明細
    for row in summary:
        print(f"   {row['Surgery_No']} | {row['Type']} | {row['Video_Time']} | {row['Real_Time']}")
    
    # === 剪輯影片片段 (事件前後 3 分鐘) ===
    video_output_dir = os.path.join(report_dir, "videos")
    os.makedirs(video_output_dir, exist_ok=True)
    
    clipped_count = 0
    for row in summary:
        vname = row['Video_Name']
        event_type = row['Type']
        video_time_str = row['Video_Time']
        
        src_path = video_path_map.get(vname)
        if not src_path or not os.path.exists(src_path):
            print(f"[事件處理] 找不到影片: {vname}")
            continue
        
        # 生成輸出檔名 (與 main_realtime.py 格式一致，含影片起始時間)
        surg_num = ''.join(filter(str.isdigit, row['Surgery_No']))
        event_real_time = row['Real_Time'].replace(':', '')
        try:
            vid_start_time = vname.split('-')[2]
        except IndexError:
            vid_start_time = "000000"
        dst_name = f"{date_str}-{event_type}_{surg_num}_{event_real_time}_{vid_start_time}.mp4"
        dst_path = os.path.join(video_output_dir, dst_name)
        
        if os.path.exists(dst_path):
            print(f"[事件處理] ⊘ 影片已存在: {dst_name}")
            continue
        
        # 計算剪輯時間（事件時間前後 3 分鐘）
        try:
            parts = video_time_str.split(':')
            event_sec = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            clip_start = max(0, event_sec - 180)
            clip_duration = 360
            
            cmd = [
                'ffmpeg', '-y',
                '-ss', str(clip_start),
                '-i', src_path,
                '-t', str(clip_duration),
                '-c', 'copy',
                '-avoid_negative_ts', '1',
                dst_path
            ]
            
            result = subprocess.run(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60
            )
            if result.returncode == 0:
                print(f"[事件處理] ✓ 影片已剪輯: {dst_name}")
                clipped_count += 1
            else:
                print(f"[事件處理] ✗ 剪輯失敗: {dst_name}")
        except subprocess.TimeoutExpired:
            print(f"[事件處理] ✗ 剪輯超時: {dst_name}")
        except Exception as e:
            print(f"[事件處理] ✗ 剪輯異常: {dst_name} ({e})")
    
    print(f"[事件處理] 完成！共剪輯 {clipped_count} 個影片片段")


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
