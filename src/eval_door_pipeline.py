import os
import cv2
import re
import time
import argparse
import datetime
import copy
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

import config
from core import PatientStatusAnalyzer

# 設定 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class ExplainableVLMRunner:
    def __init__(self, model, processor, log_file):
        self.model = model
        self.processor = processor
        self.log_file = log_file

    def _write_log(self, text):
        # 同時輸出到終端機並寫入檔案
        print(text)
        if self.log_file:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(text + "\n")

    def run(self, messages, max_new_tokens=512):
        v_mode = getattr(self, 'current_mode', 'Unknown')
        can_do_detailed_reasoning = (v_mode in ["Single_Bed", "Video"])

        # 1) 先做快速判斷（短回覆）
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        actual_max_tokens = 10
        start_time = time.time()
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=actual_max_tokens)
        infer_time_fast = time.time() - start_time

        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        output_text = output_text.split("assistant\n")[-1].strip()

        # 4. 印出思考脈絡並記錄
        # 使用由外部傳入的時間戳記 (如果有的話)
        v_time = getattr(self, 'current_video_time', 'Unknown')
        
        # 先抓取結果數字
        numbers = re.findall(r'\b[01]\b', output_text)
        vlm_result = int(numbers[-1]) if numbers else 0

        # 先抓取結果數字
        numbers = re.findall(r'\b[01]\b', output_text)
        vlm_result = int(numbers[-1]) if numbers else 0

        # 2) Stage 2: 僅結果=1才做詳細推論；Stage 3: 每次都做詳細推論
        do_detailed_reasoning = (
            (v_mode == "Video")
            or (v_mode == "Single_Bed" and vlm_result == 1)
        )
        if do_detailed_reasoning:
            detailed_messages = copy.deepcopy(messages)
            for msg in detailed_messages:
                if isinstance(msg, dict) and "content" in msg:
                    for c in msg["content"]:
                        if isinstance(c, dict) and c.get("type") == "text":
                            c["text"] = c["text"].replace("只回答一個數字：0 或 1", "")
                            c["text"] = c["text"].replace("最後只能回答：0 或 1", "")
                            c["text"] = c["text"].replace("請只回答數字。：0 或 1", "")
                            c["text"] = c["text"].replace("請只回答數字：0 或 1", "")
                            c["text"] += (
                                "\n\n【特別要求】\n"
                                "請不要立刻給出數字。請你先用一段話詳細描述你在畫面中看到了什麼，"
                                "例如物體的特徵、相對位置、是否移動、門是開是關等。\n"
                                "解釋完原因後，最後請獨立換行，寫出結論數字：0 或是 1。"
                            )

            detailed_text = self.processor.apply_chat_template(
                detailed_messages, tokenize=False, add_generation_prompt=True
            )
            detailed_image_inputs, detailed_video_inputs = process_vision_info(detailed_messages)
            detailed_inputs = self.processor(
                text=[detailed_text],
                images=detailed_image_inputs,
                videos=detailed_video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)

            start_time = time.time()
            with torch.no_grad():
                detailed_generated_ids = self.model.generate(**detailed_inputs, max_new_tokens=max_new_tokens)
            infer_time = time.time() - start_time

            detailed_output_text = self.processor.batch_decode(
                detailed_generated_ids, skip_special_tokens=True
            )[0]
            detailed_output_text = detailed_output_text.split("assistant\n")[-1].strip()

            self._write_log(f"\n⏳ 分析時間: {v_time} | 階段: {v_mode}")
            self._write_log(f"[VLM 視覺推理解釋] (思考時間: {infer_time:.1f}s)")
            self._write_log("-" * 50)
            self._write_log(detailed_output_text)
            self._write_log("-" * 50)
        else:
            # 使用 \r 在終端機覆蓋同一行，避免每幀往下跳，並且不寫入 Log 文字檔以保持乾淨
            import sys
            infer_time = infer_time_fast
            sys.stdout.write(f"\r⏳ 分析時間: {v_time} | 階段: {v_mode} | 思考: {infer_time:.1f}s | 回應: {output_text}    ")
            sys.stdout.flush()

        return vlm_result, infer_time

def main():
    parser = argparse.ArgumentParser(description="Door Pipeline - Event Vision Reasoning Pipeline")
    parser.add_argument("--input", type=str, required=True, help="影片(.mp4)或包含影片的資料夾")
    args = parser.parse_args()

    # 不再這裡預先建立單一 Log 檔，而是改到迴圈中動態依據影片建立
    print(f"========================================================")
    print(f"📝 所有的推理記錄都會自動存到各個影片日期的 outputs/<日期>/debug_reasoning/ 下")
    print(f"========================================================\n")

    def write_main_log(text, path):
        print(text)
        if path:
            with open(path, "a", encoding="utf-8") as f:
                f.write(text + "\n")

    # 1. 取得要測試的影片陣列
    input_path = args.input
    videos_to_process = []
    if os.path.isdir(input_path):
        for root, _, files in os.walk(input_path):
            for f in files:
                if f.lower().endswith(('.mp4', '.avi', '.mov')):
                    videos_to_process.append(os.path.join(root, f))
    else:
        videos_to_process = [input_path]
    videos_to_process.sort()

    if not videos_to_process:
        print("❌ 找不到可以執行的影片檔。")
        return

    # 2. 建立 Analyzer
    print("正在初始化 Door 管線系統 (包含推理解釋引擎)...")
    analyzer = PatientStatusAnalyzer()
    
    # 抽換掉原本的 Runner，初始 Log 先設為 None
    analyzer.vlm_runner = ExplainableVLMRunner(analyzer.model, analyzer.processor, None)

    # 3. 逐幀跑流水線
    for v_idx, video_path in enumerate(videos_to_process, 1):
        video_name = os.path.basename(video_path)
        
        # 提取影片日期與時間 (支援 20231211142500 或 Door_20260427_163624.mp4 等格式)
        # 優先從完整的 video_path 裡面抓 8 碼日期，這樣就算資料夾名稱是日期也能抓到
        match_date = re.search(r'(20\d{6})', video_path)
        if match_date:
            v_date = match_date.group(1)
        else:
            v_date = datetime.datetime.now().strftime("%Y%m%d")

        # 嘗試從檔名抓取 6 碼時間 (通常接在日期後面)
        match_time = re.search(r'(20\d{6})[^\d]*(\d{6})', video_name)
        if match_time:
            v_time_file = match_time.group(2)
        else:
            v_time_file = "000000" # 如果沒有時間就補零

        try:
            start_dt = datetime.datetime.strptime(f"{v_date}{v_time_file}", "%Y%m%d%H%M%S")
        except ValueError:
            start_dt = datetime.datetime.now()

        # 每個影片獨立建立對應的 Log 檔案
        output_dir = os.path.join(config.BASE_DIR, "outputs", v_date, "debug_reasoning")
        os.makedirs(output_dir, exist_ok=True)
        current_log_path = os.path.join(output_dir, f"{v_time_file}.txt")
        analyzer.vlm_runner.log_file = current_log_path

        write_main_log(f"\n\n========================================================", current_log_path)
        write_main_log(f"🎬 開始處理影片 [{v_idx}/{len(videos_to_process)}]: {video_name}", current_log_path)
        write_main_log(f"========================================================", current_log_path)
        
        analyzer.reset_runtime_state()  # 重置每一部影片的階段
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 5.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_idx = 0
        while cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            current_sec = frame_idx / fps
            real_time_str = (start_dt + datetime.timedelta(seconds=current_sec)).strftime("%Y-%m-%d %H:%M:%S")
            video_time_str = time.strftime('%H:%M:%S', time.gmtime(current_sec))

            # 將目前時間餵給 Runner，當它真正被觸發時才會印出來
            analyzer.vlm_runner.current_video_time = video_time_str
            analyzer.vlm_runner.current_mode = analyzer.current_mode

            # 記錄進入此幀前的狀態
            was_door_open = analyzer.door_open

            status, vote, infer_time = analyzer.analyze_frame(
                frame,
                task_type="Door",
                full_frame=frame,
                current_sec=current_sec,
                current_frame=frame_idx,
                video_name=os.path.basename(video_path),
                real_time=real_time_str
            )

            # 檢查開門狀態是否在這一幀正式觸發 (從 False 變 True)
            if not was_door_open and analyzer.door_open:
                analyzer.vlm_runner._write_log(f"\n✅ 【正式開門】影片時間: {video_time_str} | 已累積到達設定門檻，準備進入 Stage 2 尋找推床")

            # --- 動態步長設計 (提升效率) ---
            # 為了能看到開門的每一幀，將這裡的步長改為 1 (也就是不跳幀，逐幀檢查)
            if analyzer.current_mode == "Single":
                stride_frames = 1
            else:
                stride_frames = 1
                
            # pipeline 回傳如果是 -1 表示正在集幀不動作
            if status == -1:
                frame_idx += 1  # 集幀時逐幀採樣
                continue

            frame_idx += stride_frames
            if frame_idx >= total_frames:
                break

        cap.release()
    
    print(f"\n✅ 所有處理均已完成！你可以到各影片的 outputs/<日期>/debug_reasoning/ 中觀看詳細的文字紀錄")

if __name__ == "__main__":
    main()
