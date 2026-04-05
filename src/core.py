# core.py
import os
import torch
import cv2
import csv
import numpy as np
import time
from PIL import Image
from datetime import datetime, timedelta
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from config import MODEL_PATH, VIS_HEIGHT, VIS_WIDTH, CAMERA_SETTING, CURRENT_TEST
from utils import video_start_time, parse_response

class PatientStatusAnalyzer:
    def __init__(self):
        print(f"Loading model from {MODEL_PATH}...")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_PATH, device_map="auto", dtype=torch.float16, trust_remote_code=True
        )
        
        self.processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
        self.run_date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.door_buffer = []  # 存放 Door 時序連續畫面
        self.door_max_frames = 15  # 因為一秒有 5 幀，所以設定為 15 幀 (即看過去 3 秒的連續動作)
        self.last_door_status = 0  # 追蹤上一次 Door 的結果，決定單圖/多圖模式
        print("Model loaded successfully.")
    
    def get_prompt(self, task_type):
        if task_type == "Door":
            return """
        請「專注觀察畫面大門區域」，忽略畫面中央的手術台與其他干擾。
            請判斷：大門口是否正有「醫護人員推著推床」進出？

        【輸出 1 的條件：推床跨越門檻】(必須滿足)
        - 大門口出現一台「有金屬欄杆或輪子的大型推床/車架」，且有穿著綠色手術服的醫護人員，正在將它【推越/跨過】這道大門的門檻。(不管推床上是空的、還是蓋滿了綠色布單，只要有「推床」被推著跨越門檻，就是 1)。

        【輸出 0 的條件：無推床進出】(只要符合任一項，立即輸出 0)
        - 【徒手走路 (極重要)】：大門口有穿著綠衣服的醫護人員進出，但他們是「徒手走路」，雙手【沒有】推動任何大型推床。
        - 【走廊路過】：外面走廊上的人事物只是背景路過，沒有轉彎跨越門檻進入。
        - 【大門口淨空】：大門關閉，或是門雖然開著但完全沒人進出。
        - 【動作不在門口】：畫面中央或左側發生的任何動作請直接忽略，只要大門沒有「推床被推過」，一律輸出 0。

        請只回答數字: 0 或 1
        """
        elif task_type == "Surgery":  # 測試2 手術中(s02)
            return """
                請觀察圖片中間的床。

                請判斷：【是否有一群「站著的人」圍繞著一個「手術台」?】

                判斷規則：
                1. 如果看到手術台床上躺著一個人，有綠色的棉被蓋住，輸出 1。
                2. 動作條件：必須有 2 位 或 2位以上的「站著的人」 緊密地圍在手術台旁邊，且他們的方向是朝向床上的 輸出1。

                【排除條件 (輸出 0)】：
                    - 空床：床上沒人，無論旁邊有多少人，都輸出 0。
                    - 黑色mask: 如果畫面中手術台有「黑色色塊」部分區域遮罩，並沒看到人躺在床上的(需有粉紅色帽子)，請一律輸出 0。
                    - 單人/散開：只有 1 個人在床邊，或是多人但分散在房間各處，輸出 0。

                若符合「圍繞」狀態，輸出: 1
                有人躺在手術台上，輸出: 1
                若不符合，輸出: 0

                請只回答數字。
                """

            
        return ""
    
    # 每個frame會分析，並回傳結果
    def analyze_frame(self, frame_bgr, task_type):
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        prompt = self.get_prompt(task_type)

        if task_type == "Door":
            # 永遠緩衝畫面（維持滑動窗口）
            self.door_buffer.append(pil_image)
            if len(self.door_buffer) > self.door_max_frames:
                self.door_buffer.pop(0)

            if self.last_door_status == 1:
                # === Clip 模式：上一幀偵測到病人，用多圖時序確認 ===
                content = []
                for img in self.door_buffer:
                    content.append({"type": "image", "image": img})
                content.append({"type": "text", "text": prompt})
            else:
                # === 單圖模式：平時只送當前幀，省推論成本 ===
                content = [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt},
                ]

            messages = [
                {
                    "role": "user",
                    "content": content,
                }
            ]
        else:
            # Surgery 等模式維持單解析單圖片
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        # 推論(計算推論時間)
        start_time = time.time()
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=10)
        infer_time = time.time() - start_time

        # 從模型輸出中解析結果
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].split("assistant\n")[-1].strip()
        result = parse_response(output_text)

        # Door: 更新模式追蹤
        if task_type == "Door":
            self.last_door_status = result

        return result, infer_time

    @staticmethod
    def detect_motion(gray_curr, gray_prev, diff_thresh=25, area_thresh=0.005):
        """
        Stage 0: 幀差法動態偵測
        比較兩幀灰度圖的差異，判斷是否有顯著動態
        Returns True if motion area ratio > area_thresh
        """
        diff = cv2.absdiff(gray_curr, gray_prev)
        _, binary = cv2.threshold(diff, diff_thresh, 255, cv2.THRESH_BINARY)
        ratio = cv2.countNonZero(binary) / binary.size
        return ratio > area_thresh

    def analyze_clip(self, clip_frames_bgr, task_type):
        """
        Stage 1: 對一組 clip 畫面做 VLM 推論
        只在 Stage 0 偵測到動態時才呼叫，大幅減少 VLM 調用次數

        Args:
            clip_frames_bgr: list of BGR numpy arrays (rolling buffer)
            task_type: 任務類型
        """
        prompt = self.get_prompt(task_type)

        content = []
        for frame_bgr in clip_frames_bgr:
            image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            content.append({"type": "image", "image": pil_image})
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)

        start_time = time.time()
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=10)
        infer_time = time.time() - start_time

        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].split("assistant\n")[-1].strip()
        result = parse_response(output_text)
        return result, infer_time

    def run_analysis(self, video_path, csv_path, stride_sec=1.0, current_task=None, show_window=True):
        """
        影片的分析主要流程，會根據config.py裡設定的current_task決定要跑哪一個prompt
        並把狀態寫入csv檔案
        """
        video_name = os.path.basename(video_path)
        
        print(f"\n🎥 分析: {video_name} | 任務: {current_task}")

        # 自動修改 CSV 檔名 (例如: report_Patient.csv)
        dir_name = os.path.dirname(csv_path)
        file_name, file_ext = os.path.splitext(os.path.basename(csv_path))
        date_str = self.run_date_str
        actual_csv_path = os.path.join(dir_name, f"{file_name}_{current_task}_{date_str}{file_ext}")

        # 統一標頭
        headers = ["Video_name", "frame_index", "video_time", "real_time", "status", "infer_time"]

        file_exists = os.path.isfile(actual_csv_path)
        with open(actual_csv_path, mode='a', newline='', encoding='utf-8-sig') as csv_file:
            writer = csv.writer(csv_file)
            if not file_exists:
                writer.writerow(headers)
        
        # 影片讀取
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        stride_frames = int(fps * stride_sec)
        
        start_dt = video_start_time(video_path)
        if start_dt is None: 
            start_dt = datetime.now()

        # 開始分析
        frame_idx = 0
        while cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret: 
                break

            current_sec = frame_idx / fps
            video_time_str = time.strftime('%H:%M:%S', time.gmtime(current_sec))
            real_time_str = (start_dt + timedelta(seconds=current_sec)).strftime("%Y-%m-%d %H:%M:%S")
            

            # 直接執行 main.py 傳進來的 current_task
            status, infer_time = self.analyze_frame(frame, current_task)

            print(f"\r Analyzing {real_time_str} ... | {status}", end="")
            # 寫入 CSV
            row_data = [video_name, frame_idx, video_time_str, real_time_str, status, f"{infer_time:.3f}"]
            with open(actual_csv_path, mode='a', newline='', encoding='utf-8-sig') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(row_data)

            # 顯示畫面
            if show_window:
                vis = cv2.resize(frame, (VIS_WIDTH, VIS_HEIGHT))
                txt = f"{current_task}: {status}"
                cv2.putText(vis, txt, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Analysis", vis)
                if cv2.waitKey(1) & 0xFF == ord('q'): 
                    break

            frame_idx += stride_frames
            if frame_idx >= total_frames: 
                break

        cap.release()
        if show_window: 
            cv2.destroyAllWindows()