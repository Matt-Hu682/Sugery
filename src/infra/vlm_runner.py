import time

import torch
from qwen_vl_utils import process_vision_info

from utils import parse_response


class VLMRunner:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    def run(self, messages, max_new_tokens=10):
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        start_time = time.time()
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        infer_time = time.time() - start_time

        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        output_text = output_text.split("assistant\n")[-1].strip()

        vlm_result_str = parse_response(output_text)
        try:
            vlm_result = int(vlm_result_str)
        except ValueError:
            vlm_result = 0

        return vlm_result, infer_time
