import time
import warnings

import torch
from transformers.utils import logging as transformers_logging
from qwen_vl_utils import process_vision_info

from utils import parse_response
from config import (
    VLM_ENABLE_THINKING,
    VLM_DO_SAMPLE,
    VLM_TEMPERATURE,
    VLM_TOP_P,
    VLM_TOP_K,
    VLM_REPETITION_PENALTY,
)


warnings.filterwarnings(
    "ignore",
    message=r".*Kwargs passed to `processor\.__call__` have to be in `processor_kwargs` dict.*",
)
transformers_logging.set_verbosity_error()


class VLMRunner:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    def run(self, messages, max_new_tokens=10):
        # ── Thinking Mode ─────────────────────────────────────────────────────
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=VLM_ENABLE_THINKING,
        )

        # ── 影像/影片輸入（含解析度控制）─────────────────────────────────────
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        # ── 推論參數 ──────────────────────────────────────────────────────────
        generate_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=VLM_DO_SAMPLE,
            repetition_penalty=VLM_REPETITION_PENALTY,
        )
        if VLM_DO_SAMPLE:
            generate_kwargs.update(
                temperature=VLM_TEMPERATURE,
                top_p=VLM_TOP_P,
                top_k=VLM_TOP_K,
            )

        start_time = time.time()
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **generate_kwargs)
        infer_time = time.time() - start_time

        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        output_text = output_text.split("assistant\n")[-1].strip()

        vlm_result_str = parse_response(output_text)
        try:
            vlm_result = int(vlm_result_str)
        except ValueError:
            vlm_result = 0

        return vlm_result, infer_time
