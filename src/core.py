import os
import time
from datetime import datetime, timedelta

import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from config import (
    DOOR_VIDEO_CROP,
    DOOR_VIDEO_MAX_FRAMES,
    DOOR_VIDEO_MIN_FRAMES,
    DOOR_VIDEO_TEMPORAL_STRIDE,
    EVENT_COOLDOWN_FRAMES,
    MODEL_PATH,
    USE_DOOR_VIDEO_CROSS,
    VIS_HEIGHT,
    VIS_WIDTH,
)
from infra.vlm_runner import VLMRunner
from prompts.prompts import get_prompt
from utils import video_start_time


class PatientStatusAnalyzer:
    def __init__(self):
        print(f"Loading model from {MODEL_PATH}...")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_PATH, device_map="auto", dtype=torch.float16, trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
        self.vlm_runner = VLMRunner(self.model, self.processor)
        self.run_date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.door_min_frames = DOOR_VIDEO_MIN_FRAMES # 門口機最少幀數
        self.door_max_frames = DOOR_VIDEO_MAX_FRAMES # 門口最大幀數
        self.door_temporal_stride = DOOR_VIDEO_TEMPORAL_STRIDE # 門口stride時間
        self.use_door_video_cross = USE_DOOR_VIDEO_CROSS
        self.door_video_zero_timeout = 3
        self.door_open_threshold = 20 #觀察開門的幀數
        self.debug_save_frames = True  # 設為 True 可將每次 Stage 3A 的畫面存到 outputs/debug_frames/
        self._debug_frame_base = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "outputs"
        )
        self.reset_runtime_state()
        print("Model loaded successfully.")

    def reset_runtime_state(self):
        self.door_buffer = []     # door video buffer (DOOR_VIDEO_CROP 裁切，供 Stage 3A 穿門確認用)
        self.door_buffer_raw = [] # door video buffer 整張原圖（供 Stage 3B 方向判斷用）
        self.door_open = False
        self.bed_detected = False
        self.current_mode = "Single"
        self.door_video_zero_run = 0
        self.push_to_pipeline = False
        self.door_open_count = 0
        self.last_event_video_sec = None
        self.last_event_direction = None  # 2=ENT, 3=SEND，用於決定下次冷卻時間
        self.send_cooldown_sec = 4500
        self.last_stage3_cross_result = None
        self.last_stage3_direction_result = None
        self.pending_event_direction = None  
        self.pending_event_frame = None      
        self.pending_event_video_time = None
        self.pending_event_real_time = None
        self.pending_event_video_name = None
        self._frames_since_pending = 0       
        self.pending_confirm_threshold = 150 # 30秒 @5fps
        self.event_metadata_override = None

    def _build_messages(self, content): # 建立訊息格式
        return [{"role": "user", "content": content}]

    def _build_video_content(self, prompt, frames): # 建立video模式
        return [
            {
                "type": "video",
                "video": list(frames),
                "fps": 5.0,
            },
            {"type": "text", "text": prompt},
        ]

    def _sample_key_frames(self, frames, n=12):
        """從 buffer 均勻採樣 n 張關鍵幀（保證包含頭、尾），供 VLM 比較頭尾場景差異"""
        if len(frames) <= n:
            return list(frames)
        # 強制包含第 0 幀(起點)和最後一幀(終點)，中間均勻插入
        indices = [0]
        step = (len(frames) - 1) / (n - 1)
        for i in range(1, n - 1):
            indices.append(round(i * step))
        indices.append(len(frames) - 1)
        # 去重並排序
        indices = sorted(set(indices))
        return [frames[i] for i in indices]

    def _trim_door_buffer(self): # 保留最新的那一段畫面
        if len(self.door_buffer) > self.door_max_frames:
            self.door_buffer = self.door_buffer[-self.door_max_frames:]
        if len(self.door_buffer_raw) > self.door_max_frames:
            self.door_buffer_raw = self.door_buffer_raw[-self.door_max_frames:]

    def _save_debug_frames(self): # debug 看圖片用
        """將當前 door_buffer 中所有幀儲存至 outputs/<日期>/debug_frames/<日期-影片時間>/ """
        if not self.debug_save_frames or not self.door_buffer:
            return
        
        video_name, _ = getattr(self, "_debug_context", ("unknown", "000000"))
        
        import re
        match_date = re.search(r'(20\d{6})', video_name)
        if match_date:
            v_date = match_date.group(1)
        else:
            v_date = "UnknownDate"
            
        # 從檔名抓影片起始時間（若無則補 000000）
        match_time = re.search(r'(20\d{6})[^\d]*(\d{6})', video_name)
        if match_time:
            video_file_time = match_time.group(2)
        else:
            video_file_time = "000000"

        # 使用第一幀的影片時間作為資料夾時間標籤
        first_time = self.door_buffer[0]["v_time"] if self.door_buffer else "000000"
        folder_name = f"{v_date}-{video_file_time}-{first_time}"
        
        # 將截圖儲存到影片日期的資料夾下
        save_dir = os.path.join(self._debug_frame_base, v_date, "debug_frames", folder_name)
        os.makedirs(save_dir, exist_ok=True)
        collection_start_time = first_time
        for i, item in enumerate(self.door_buffer):
            pil_img = item["img"]
            # 檔案名稱使用「收集幀數開始時間」作為前綴
            pil_img.save(os.path.join(save_dir, f"{collection_start_time}_{i:02d}.jpg"))
        print(f"  [Debug] {len(self.door_buffer)} 幀已儲存至 {save_dir}")

    
    
    def _set_door_stage3_pending(self): # 到stage3，正在收集幀數
        self.current_mode = "Video"
        self.push_to_pipeline = False
        buf_count = len(self.door_buffer)
        print(
            f"  [Door] 看到推床！集幀中... ({buf_count}/{self.door_min_frames}, max={self.door_max_frames})",
            end="\r",
        )
        return -1, f"Stg3(Buf={buf_count})", 0.0

    def _build_door_messages(self, pil_image, pil_raw, pil_full=None): # 建立door模式的訊息
        if not self.door_open:
            self.current_mode = "Single"
            return self._build_messages( # 建立單一畫面的訊息
                [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": get_prompt("Door", mode="single")},
                ]
            )

        if not self.bed_detected:
            self.current_mode = "Single_Bed"
            return self._build_messages( # stage2 觀察圖片中是否有推床(裁切後的門口視野)
                [
                    {"type": "image", "image": pil_full},
                    {"type": "text", "text": get_prompt("Door", mode="single_bed")},
                ]
            )

        if len(self.door_buffer) < self.door_min_frames:
            return None

        self.current_mode = "Video"
        self._save_debug_frames()  # ← 每次觸發 Stage 3A 前，把畫面存下來
        
        # 均勻抽出 12 幀（強制包含頭尾幀）送 VLM
        n_samples = 12
        all_imgs = [item["img"] for item in self.door_buffer]
        key_frames = self._sample_key_frames(all_imgs, n=n_samples) 

        # 計算抽出的 index 以便印出對應檔名
        if len(self.door_buffer) <= n_samples:
            indices = list(range(len(self.door_buffer)))
        else:
            import numpy as np
            indices_f = [0]
            step = (len(self.door_buffer) - 1) / (n_samples - 1)
            for i in range(1, n_samples - 1):
                indices_f.append(round(i * step))
            indices_f.append(len(self.door_buffer) - 1)
            indices = sorted(set(indices_f))

        key_items = [self.door_buffer[i] for i in indices]
        # 顯示明確的抽樣檔名 (例如 142500_00)
        sampled_names = [f"{item['v_time']}_{idx:02d}" for idx, item in zip(indices, key_items)]
        sampled_count = len(key_frames)
        
        msg = f"Stage 3A：從 {len(self.door_buffer)} 幀中採樣 {sampled_count} 幀送 VLM。抽取圖片：{', '.join(sampled_names)}"
        print(f"  [Door] {msg}")
        
        if hasattr(self, 'vlm_runner') and hasattr(self.vlm_runner, '_write_log'):
            self.vlm_runner._write_log(f"\n📸 [{msg}]")
            
        return self._build_messages(
            self._build_video_content(
                get_prompt("Door", mode="video_cross"),
                key_frames,
            )
        )

    def _handle_door_single_result(self, vlm_result): # 處理門口
        self.push_to_pipeline = False
        pipeline_status = 1 if vlm_result == 1 else 0

        if pipeline_status == 1:
            self.door_open_count += 1
            print(f"  [Door] Stage 1: 門開中 ({self.door_open_count}/{self.door_open_threshold})", end="\r")
            if self.door_open_count >= self.door_open_threshold:
                self.door_open = True
                self.bed_detected = False
                self.door_open_count = 0
                self.door_buffer.clear()
                self.door_buffer_raw.clear()
                print("\n  [Door] Stage 1 OK! 門開了，進入 Stage 2 (尋找推床)")
        else:
            self.door_open_count = 0
            self.door_buffer.clear()
            self.door_buffer_raw.clear()

        return pipeline_status

    def _handle_door_bed_result(self, vlm_result): # 看是否有推床
        self.push_to_pipeline = False
        pipeline_status = 1

        if vlm_result == 1:
            self.bed_detected = True # 有偵測到床
            self.door_open_count = 0
            print("\n  [Door] Stage 2 OK! 抓到推床，接續 Stage 3 集幀...")
        else:
            self.door_open_count += 1 
            print(f"  [Door] Stage 2: 沒看到推床 ({self.door_open_count}/50)", end="\r")
            if self.door_open_count >= 50:
                self.door_open = False
                self.bed_detected = False
                self.door_open_count = 0
                print("\n  [Door] Stage 2 Timeout! 沒看到推床，退回 Stage 1")

        return pipeline_status



    def _is_stage3_in_cooldown(self, current_frame, vlm_result):
        if self.last_event_video_sec is None or current_frame is None:
            return False

        # 依上次已確認的方向動態選冷卻時間
        # ENT(2) 後 → SEND 要等 15 分鐘，SEND(3) 後 → ENT 只要等 5 分鐘
        if self.last_event_direction == 2:  # 上次是 ENT
            cooldown = EVENT_COOLDOWN_FRAMES.get("Door_ENT", 4500)
        elif self.last_event_direction == 3:  # 上次是 SEND
            cooldown = EVENT_COOLDOWN_FRAMES.get("Door_SEND", 1500)
        else:
            cooldown = EVENT_COOLDOWN_FRAMES.get("Door_ENT", 4500)  # 預設

        elapsed_frames = current_frame - self.last_event_video_sec
        remaining_frames = cooldown - elapsed_frames
        if remaining_frames <= 0:
            return False

        elapsed_min = elapsed_frames / 5 / 60
        remaining_min = remaining_frames / 5 / 60
        last_evt = "ENT" if self.last_event_direction == 2 else "SEND"
        print(
            f"  [Door] 冷卻中(上次={last_evt})，距上次事件 {elapsed_min:.1f} 分鐘 ({elapsed_frames} 幀)，"
            f"還需 {remaining_min:.1f} 分鐘，此次忽略"
        )
        return True

    def _handle_door_video_result(
        self,
        vlm_result,
        infer_time,
        current_frame,
        current_sec,
        video_name,
        real_time,
    ):
        self.push_to_pipeline = True
        self.last_stage3_cross_result = vlm_result
        self.last_stage3_direction_result = None

        print(f"  [Door] Stage 3A cross-check = {vlm_result}")

        # 穿門確認（1）：直接進入穩定期，不再當下判斷手術台狀態（因為 ENT 時病人還沒上台，SEND 時病人可能提早離台）
        crossing_confirmed = (vlm_result == 1)

        if crossing_confirmed:
            self.door_buffer.clear()
            self.door_buffer_raw.clear()
            self.door_open = False
            self.bed_detected = False
            self.door_video_zero_run = 0

            if self._is_stage3_in_cooldown(current_frame, 1):
                return 0, infer_time, 0

            # 進入單筆穩定期，阻塞新事件直至確認
            self.pending_event_direction = 1  
            self.pending_event_frame = current_frame  
            self.pending_event_video_time = time.strftime("%H:%M:%S", time.gmtime(current_sec or 0))
            self.pending_event_real_time = real_time
            self.pending_event_video_name = video_name
            self._frames_since_pending = 0
            print(f"\n  [Door] Stage 3A 確認穿門！開始穩定期等待 {self.pending_confirm_threshold} 幀...")
            return 0, infer_time, 0

        self.door_video_zero_run += 1
        if self.door_video_zero_run >= self.door_video_zero_timeout:
            self.bed_detected = False
            self.current_mode = "Single_Bed"
            self.door_buffer.clear()
            self.door_buffer_raw.clear()
            self.door_video_zero_run = 0
            self.door_open_count = 0
            print(f"  [Door] 連續 {self.door_video_zero_timeout} 次視窗未見穿越，退回 Stage 2 (重新尋找推床)")
        else:
            # 尾端續接：只保留上一窗最後幾幀，與後續新幀拼接成下一窗
            # 例：stride=5 時保留 4 幀（上一窗 36~39），避免時間窗跳太大
            tail_keep = max(1, self.door_temporal_stride - 1)
            self.door_buffer = self.door_buffer[-tail_keep:]
            self.door_buffer_raw = self.door_buffer_raw[-tail_keep:]
            print(
                f"  [Door] 未見完整通過，尾端續接追蹤..."
                f" stride={self.door_temporal_stride}，保留尾端 {len(self.door_buffer)} 幀"
            )

        return 0, infer_time, vlm_result

    def _build_door_vlm_vote(self, pipeline_status):
        if self.current_mode == "Single":
            return f"Stg1(Door={pipeline_status})"
        if self.current_mode == "Single_Bed":
            return f"Stg2(Bed={pipeline_status})"
        if self.current_mode == "Video":
            return (
                f"Stg3(Cross={self.last_stage3_cross_result},"
                f"Patient={self.last_stage3_direction_result},Out={pipeline_status})"
            )
        return ""



    def _handle_stage3b_pending(self, pil_raw):
        """處理 Stage 3B：穩定期等待與手術台最終狀態確認"""
        if self.pending_event_direction is None:
            return None

        self._frames_since_pending += 1
        remaining = self.pending_confirm_threshold - self._frames_since_pending
        
        if self._frames_since_pending >= self.pending_confirm_threshold: # 穩定期 30 秒 (150)
            print(f"\n  [Door] 穩定期結束，執行 Stage 3B 檢查手術台狀態...")
            stg3b_messages = self._build_messages([
                {"type": "image", "image": pil_raw},
                {"type": "text", "text": get_prompt("Door", mode="patient_check")},
            ])
            
            stg3b_res, stg3b_time = self.vlm_runner.run(stg3b_messages)
            table_result = int(stg3b_res) if str(stg3b_res).isdigit() else 0 # 0 或 1
            self.last_stage3_direction_result = table_result
            
            direction = 2 if table_result == 1 else 3 # 2 是進，3 是出
            
            confirmed_frame_idx = self.pending_event_frame 
            confirmed_video_time = self.pending_event_video_time 
            confirmed_real_time = self.pending_event_real_time 
            confirmed_video_name = self.pending_event_video_name 

            self.last_event_video_sec = confirmed_frame_idx
            self.last_event_direction = direction  # 記下此次方向，供下次冷卻判斷使用
            self.push_to_pipeline = True  
            
            self.pending_event_direction = None
            self.pending_event_frame = None
            self.pending_event_video_time = None
            self.pending_event_real_time = None
            self.pending_event_video_name = None
            self._frames_since_pending = 0
            
            self.event_metadata_override = {
                "frame_idx": confirmed_frame_idx,
                "video_time": confirmed_video_time,
                "real_time": confirmed_real_time,
                "video_name": confirmed_video_name,
            }
            
            event_str = "ENT(推入)" if direction == 2 else "SEND(推出)"
            print(f"  [Door] ✓ {event_str} 正式確認！（VLM 判斷手術台 {'有病人(1)' if table_result==1 else '沒病人(0)'}，已校正發布時間至穿門當下）")
            return direction, f"Stg3(Patient={table_result},Event={event_str})", 0.0
        
        # 還在穩定期內，不回傳任何結果，讓主流程繼續收集背景資訊
        return None

    def pop_event_metadata_override(self):
        meta = self.event_metadata_override
        self.event_metadata_override = None
        return meta

    def analyze_frame(  
        self,
        frame_bgr,
        task_type,
        full_frame=None,
        current_sec=None,
        current_frame=None,
        video_name=None,
        real_time=None,
    ):  # 逐幀分析
        if full_frame is None:
            full_frame = frame_bgr

        self.send_cooldown_sec = EVENT_COOLDOWN_FRAMES.get(task_type, 4500) #事件的冷卻時間

        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        if DOOR_VIDEO_CROP is not None: #有裁切的話就丟進去
            vx1, vy1, vx2, vy2 = DOOR_VIDEO_CROP
            video_frame = full_frame[vy1:vy2, vx1:vx2]
        
        else:
            video_frame = full_frame
        video_rgb = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
        pil_full = Image.fromarray(video_rgb) # 有裁切的

        raw_rgb = cv2.cvtColor(full_frame, cv2.COLOR_BGR2RGB)
        pil_raw = Image.fromarray(raw_rgb) # 沒裁切的圖，3b用到

        if task_type == "Door": 
            # 更新 debug 用的影片情境（影片名稱 + 影片內時間）
            self._debug_context = (video_name or "unknown", real_time or "000000")

            # 檢查是否處於「穩定期」(Stage 3B 準備)，如果是，交由專屬函式處理
            pending_result = self._handle_stage3b_pending(pil_raw)
            if pending_result is not None:
                return pending_result

            # 僅在「已看到推床」後才開始收集 Stage 3 影片視窗
            # 也就是 Stage1/Stage2（僅開門或找床）期間不集幀，避免混入無關畫面。
            if self.bed_detected:
                # 統一使用影片時間點（從影片起始累計），避免使用實際時鐘時間
                video_time_str_clean = (
                    time.strftime("%H%M%S", time.gmtime(current_sec))
                    if current_sec is not None
                    else f"f{current_frame}"
                )
                
                self.door_buffer.append({"img": pil_full, "v_time": video_time_str_clean})       # 裁切圖與時間
                self.door_buffer_raw.append(pil_raw)    # 整張原圖（供 Stage 3B 是否有人判斷）
                self._trim_door_buffer()
            messages = self._build_door_messages(pil_image, pil_raw, pil_full)
            if messages is None:
                return self._set_door_stage3_pending()
        
        # 不是door模式(surgery)
        else:
            messages = self._build_messages(
                [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": get_prompt(task_type, mode="single")},
                ]
            )

        # 確保 Runner 看到的是「本幀實際模式」，避免沿用上一幀模式造成顯示落差
        self.vlm_runner.current_mode = self.current_mode
        vlm_result, infer_time = self.vlm_runner.run(messages) 

        if task_type == "Door":
            if self.current_mode == "Single":
                pipeline_status = self._handle_door_single_result(vlm_result) 
            elif self.current_mode == "Single_Bed":
                pipeline_status = self._handle_door_bed_result(vlm_result)
            else:
                pipeline_status, infer_time, vlm_result = self._handle_door_video_result(
                    vlm_result, infer_time, current_frame, current_sec, video_name, real_time
                )
            return pipeline_status, self._build_door_vlm_vote(pipeline_status), infer_time

        self.push_to_pipeline = True
        return vlm_result, "", infer_time

    def run_analysis(self, video_path, csv_path, stride_sec=1.0, current_task=None, show_window=True):
        video_name = os.path.basename(video_path)
        print(f"\n分析: {video_name} | 任務: {current_task}")

        dir_name = os.path.dirname(csv_path)
        file_name, file_ext = os.path.splitext(os.path.basename(csv_path))
        date_str = self.run_date_str
        actual_csv_path = os.path.join(dir_name, f"{file_name}_{current_task}_{date_str}{file_ext}")

        #這邊是要開啟CSV
        headers = ["Video_name", "frame_index", "video_time", "real_time", "status", "infer_time"]
        file_exists = os.path.isfile(actual_csv_path)
        with open(actual_csv_path, mode="a", newline="", encoding="utf-8-sig") as csv_file:
            if not file_exists:
                import csv

                csv.writer(csv_file).writerow(headers)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        stride_frames = int(fps * stride_sec)

        start_dt = video_start_time(video_path) or datetime.now()
        frame_idx = 0

        while cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            current_sec = frame_idx / fps
            video_time_str = time.strftime("%H:%M:%S", time.gmtime(current_sec))
            real_time_str = (start_dt + timedelta(seconds=current_sec)).strftime("%Y-%m-%d %H:%M:%S")

            status, _, infer_time = self.analyze_frame(
                frame,
                current_task,
                full_frame=frame,
                current_sec=current_sec,
                current_frame=frame_idx,
                video_name=video_name,
                real_time=real_time_str,
            )
            if status == -1:
                frame_idx += stride_frames # FRAME_IDX = frame_idx + stride_frame 40 + 15 = 55
                continue
            
            print(f"\r Analyzing {real_time_str} ... | {status}", end="")
            with open(actual_csv_path, mode="a", newline="", encoding="utf-8-sig") as csv_file:
                import csv

                csv.writer(csv_file).writerow(
                    [video_name, frame_idx, video_time_str, real_time_str, status, f"{infer_time:.3f}"]
                )

            if show_window:
                vis = cv2.resize(frame, (VIS_WIDTH, VIS_HEIGHT))
                cv2.putText(vis, f"{current_task}: {status}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Analysis", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += stride_frames
            if frame_idx >= total_frames:
                break

        cap.release()
        if show_window:
            cv2.destroyAllWindows()
