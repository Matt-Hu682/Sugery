import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from config import MODEL_PATH
from infra.vlm_runner import VLMRunner
from prompts.prompts import get_prompt


class PatientStatusAnalyzer:
    """Surgery 視角的單幀狀態分析器。"""

    def __init__(self):
        print(f"Loading model from {MODEL_PATH}...")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_PATH, device_map="auto", dtype=torch.float16, trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
        self.vlm_runner = VLMRunner(self.model, self.processor)
        self.reset_runtime_state()
        print("Model loaded successfully.")

    def reset_runtime_state(self):
        self.current_mode = "Single"
        self.push_to_pipeline = True

    def analyze_frame(
        self,
        frame_bgr,
        task_type="Surgery",
        full_frame=None,
        current_sec=None,
        current_frame=None,
        video_name=None,
        real_time=None,
    ):
        """逐幀分析 Surgery 畫面，回傳 pipeline 狀態、VLM 標記與推論時間。"""
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        messages = self._build_messages(
            [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": get_prompt(task_type, mode="single")},
            ]
        )

        self.current_mode = "Single"
        self.vlm_runner.current_mode = self.current_mode
        vlm_result, infer_time = self.vlm_runner.run(messages)

        self.push_to_pipeline = True
        return vlm_result, "", infer_time

    def _build_messages(self, content):
        return [{"role": "user", "content": content}]
