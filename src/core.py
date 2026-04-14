# core.py
import os
import torch
import cv2
import csv
import time
from PIL import Image
from datetime import datetime, timedelta
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from config import MODEL_PATH, VIS_HEIGHT, VIS_WIDTH, DOOR_VIDEO_CROP, EVENT_COOLDOWN_FRAMES
from utils import video_start_time, parse_response

class PatientStatusAnalyzer:
    def __init__(self):
        # 模型載入
        print(f"Loading model from {MODEL_PATH}...")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_PATH, device_map="auto", dtype=torch.float16, trust_remote_code=True
        )
        
        self.processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
        self.run_date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.door_buffer = []     # 存放 Door 時序連續畫面
        self.door_max_frames = 50  # 以 5fps 計算，25 幀 = 5 秒的連續動作
        self.door_open = False     # 追蹤上一次 Single，決定是否要切 Video 模式
        self.current_mode = "Single"
        self.door_video_zero_run = 0   # Video 模式下連續 0 的計數
        self.door_video_zero_timeout = 3
        self.push_to_pipeline = False
        self.door_open_count = 0       # 連續偵測到「門打開」的幀數計數器
        self.door_open_threshold = 20  # 集滿 20 幀門開切到 Video 模式
        self.last_event_video_sec = None  # 上次確認事件的幀索引
        self.send_cooldown_sec = 4500   # ENT後 SEND 冷卻 15 分鐘 (4500 幀 @5fps)
        print("Model loaded successfully.")
    
    def get_prompt(self, task_type, mode="single"):
        if task_type == "Door":
            if mode == "single":
                # ── 第一階段：門的開關偵測（最簡單，成功率最高）──
                return """
        請觀察大門。

        請判斷：大門目前是「打開的」還是「關著的」？

        【輸出 1 的條件：門是打開的】
        - 打開的門看的到外面的景象。

        【輸出 0 的條件：門是關著的】
        - 大門是完全緊閉。


        請只回答數字: 0 或 1
        """
            else:
                # ── 第二階段：門打開後，用 Video 確認是否有病床在移動 ──
                return """
                請仔細觀察這段影像序列。攝影機位於手術室內部，鏡頭對準大門口。

                【空間定義】
                - 畫面上方：門口（外）
                - 畫面下方：室內（內）

                【最重要規則】

                本任務是判斷推床或輪椅是否「經過門」。
                只有當載人的推床或有坐著的人的輪椅從門的另一側出現，
                並完整通過門的位置，
                才允許輸出 2 或 3。

                如果沒有完整經過門，
                一律輸出 0。

                【判定邏輯】

                (ENT - 推入)：

                推床或輪椅：
                - 最初位於門外（上方）向室內移動
                - 過程完整穿過門並進入室內
                - 推床上載有病人或是輪椅上有坐著的人

                → 輸出：2
                
                
                (SEND - 推出)：

                推床：
                - 最初位於室內（下方)向門方向移動
                - 過程完整穿過門並離開室內，最終進入門外，或在門的位置消失
                - 上面載有病人（蓋綠色棉被）

                → 輸出：3


                符合以下任一情況視為0：

                1. 只有大門打開或只是醫護人員在走動。
                2. 畫面中沒有推床或輪椅穿過門。
                3. 推床或輪椅只在門口上方的走廊區域左右移動（例如從左到右或右到左）。
                4. 推床或輪椅停在門附近或正在準備進出，但沒有完整穿過門。
                5. 在整段影像中，推床沒有從門外進入室內，也沒有從室內離開。
                6. 醫護人員推的是金屬框架、垃圾桶、小型儀器、推車
                符合上述條件者，請輸出：0

                請只回答數字：0、2 或 3
        """
        elif task_type == "Surgery":  # 測試2 手術中(s02)
            return """
                請觀察圖片中間的床。

                請判斷：【是否有一群「站著的人」圍繞著一個「手術台」?】

                判斷規則：
                1. 如果看到手術台床上躺著一個人，有綠色的棉被蓋住，輸出 1。
                2. 動作條件：必須有 2 位 或 2位以上的「站著的人」 圍在手術台旁邊，且他們的方向是朝向床上的 輸出1。

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
    
    # 每個幀會分析，並回傳結果
    def analyze_frame(self, frame_bgr, task_type, full_frame=None, current_sec=None, current_frame=None):
        """
        frame_bgr     : 已裁切後的幀 (Single 模式使用)
        full_frame    : 完整幀 (Video 模式 buffer 使用)；若為 None 則跟 frame_bgr 相同
        current_sec   : 目前影片秒數 (備用)
        current_frame : 目前幀索引，用於計算事件冷卻幀數
        """
        if full_frame is None:
            full_frame = frame_bgr

        # 依照任務類型套用對應的 SEND 冷卻幀數
        self.send_cooldown_sec = EVENT_COOLDOWN_FRAMES.get(task_type, 4500)

        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # Video buffer 存裁切後的幀 (套用 DOOR_VIDEO_CROP)
        if DOOR_VIDEO_CROP is not None:
            vx1, vy1, vx2, vy2 = DOOR_VIDEO_CROP
            video_frame = full_frame[vy1:vy2, vx1:vx2]
        else:
            video_frame = full_frame
        video_rgb = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
        pil_full = Image.fromarray(video_rgb)

        if task_type == "Door":
            self.door_buffer.append(pil_full)  # Video buffer 存 Video 裁切後的幀

            if not self.door_open:
                # ── 第一階段：Single 模式，用裁切後小圖判斷門有沒有打開 ──
                self.current_mode = "Single"
                prompt = self.get_prompt(task_type, mode="single")
                content = [
                    {"type": "image", "image": pil_image},  # 裁切後小圖
                    {"type": "text", "text": prompt},
                ]
            elif len(self.door_buffer) < self.door_max_frames:
                # ── 第二階段：Video 模式，buffer 尚未蓄滿，繼續等待，跳過本幀 ──
                self.current_mode = "Video"
                self.push_to_pipeline = False
                buf_count = len(self.door_buffer)
                print(f"  [Door] 蓄集中... ({buf_count}/{self.door_max_frames} 幀)", end="\r")
                return -1, buf_count, 0.0  # vlm_vote = 目前已蓄集幀數
            else:
                # ── 第二階段：Video 模式，buffer 蓄滿 → 執行一次推論 ──
                self.current_mode = "Video"
                prompt = self.get_prompt(task_type, mode="video")
                content = [
                    {
                        "type": "video",
                        "video": list(self.door_buffer),
                        "fps": 5.0  # 每隔 0.2 秒一幀 = 5fps
                    },
                    {"type": "text", "text": prompt}
                ]

            messages = [
                {
                    "role": "user",
                    "content": content,
                }
            ]
        else:
            # Surgery 等模式維持單解析單圖片
            prompt = self.get_prompt(task_type, mode="single")
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
        vlm_result_str = parse_response(output_text)
        try:
            vlm_result = int(vlm_result_str)
        except ValueError:
            vlm_result = 0

        pipeline_status = vlm_result  # 預設把模型結果丟給外部

        # Door: 分階段更新狀態
        if task_type == "Door":
            if self.current_mode == "Single":
                # ── Single 的結果僅作內部開關，不進 Pipeline ──
                self.push_to_pipeline = False
                is_open = (vlm_result == 1)

                # Single 模式只是看門，對外界來說 1=門開, 0=門關
                pipeline_status = 1 if is_open else 0

                if is_open:
                    self.door_open_count += 1
                    print(f"  [Door]  門持續打開 ({self.door_open_count}/{self.door_open_threshold} 幀)", end="\r")
                    if self.door_open_count >= self.door_open_threshold:
                        # 集滿 20 幀連續門開，切換到 Video 模式
                        # buffer 裡此時就是純粹的 20 幀開門過程
                        self.door_open = True
                        self.door_video_zero_run = 0
                        self.door_open_count = 0
                        print(f"\n  [Door] 滿 {self.door_open_threshold} 幀連續門開！切換到 Video 模式（buffer={len(self.door_buffer)} 幀）...")
                else:
                    # 門關了，重置計數器且清空 buffer
                    if self.door_open_count > 0:
                        print(f"\n  [Door] 門關閉，計數器重置 ({self.door_open_count})")
                    self.door_open_count = 0
                    self.door_buffer.clear()  # 門關了就清空，下次開門從第1幀開始串集純正車
            else:
                # ── Video 的結果才進 Pipeline ──
                self.push_to_pipeline = True
                pipeline_status = vlm_result

                if vlm_result in [2, 3]:
                    # 若偵測到推床，清空 buffer 並準備進入冷卻/退回 Single
                    self.door_buffer.clear()

                    # 事件 (ENT 或 SEND) 冷卻期檢查 (以幀數計算)
                    if self.last_event_video_sec is not None and current_frame is not None:
                        elapsed_frames = current_frame - self.last_event_video_sec
                        remaining_frames = self.send_cooldown_sec - elapsed_frames
                        if remaining_frames > 0:
                            elapsed_min = elapsed_frames / 5 / 60
                            remaining_min = remaining_frames / 5 / 60
                            event_name = "ENT" if vlm_result == 2 else "SEND"
                            print(f"  [Door] {event_name} 冷卻中，距上次事件 {elapsed_min:.1f} 分鐘 ({elapsed_frames} 幀)，還需 {remaining_min:.1f} 分鐘，此次忽略")
                            pipeline_status = 0
                            self.door_open = False
                            return pipeline_status, 0, infer_time  # vlm_vote=0 (冷卻拒絕)

                    # ✅ 有病床通過大門，區分推入或推出！
                    self.door_video_zero_run = 0
                    self.door_open = False  # 退回 Single 模式，門還開着就繼續等候下一次
                    self.last_event_video_sec = current_frame  # 記錄事件幀索引

                    pipeline_status = vlm_result
                    event_str = "推入(ENT)" if vlm_result == 2 else "推出(SEND)"
                    print(f"  [Door] 病床確認為 {event_str}！(15 分鐘冷卻起算)")
                else:
                    # 門開著但病床未完整通過，或可能已經關門
                    pipeline_status = 0
                    self.door_video_zero_run += 1
                    
                    if self.door_video_zero_run >= self.door_video_zero_timeout:
                        # 連續 N 次未見推床完整通過，判定活動結束，退回 Single 模式
                        self.door_open = False
                        self.door_buffer.clear()
                        self.door_video_zero_run = 0
                        print(f"  [Door] 連續 {self.door_video_zero_timeout} 次視窗未見推床，退回單幀警戒模式")
                    else:
                        # === 加入滑動視窗 (Sliding Window) 邏輯 ===
                        # 丟掉最舊的 15 幀，保留最新的 10 幀，維持在 Video 模式繼續蓄集
                        temporal_stride = 15
                        self.door_buffer = self.door_buffer[temporal_stride:]
                        print(f"  [Door] 未見完整通過，滑動視窗追蹤...保留最新 {len(self.door_buffer)} 幀")
        else:
            # Surgery 等其他模式永遠進 Pipeline
            self.push_to_pipeline = True

        # Door 模式：vlm_vote = 實際 VLM 判定結果(0/2/3)或 Single 空字串
        if task_type == "Door":
            vlm_vote = pipeline_status if self.current_mode == "Video" else ''
            return pipeline_status, vlm_vote, infer_time

        return pipeline_status, '', infer_time



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

            # status == -1 代表 Door/Video 模式正在蓄集 buffer，尚未推論，跳過本幀
            if status == -1:
                frame_idx += stride_frames
                continue

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