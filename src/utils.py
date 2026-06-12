# utils.py

"""
分析工具
- video_start_time: 從影片檔名解析開始時間
- parse_response:  解析 VLM 模型回傳的文字，擷取判斷結果
- extract_dual_frames:  同時讀取兩個視角的影片並左右拼接成單一畫面序列

"""

import cv2
import os
import re
import numpy as np
from datetime import datetime
from PIL import Image

def video_start_time(video_path):
    """
    從檔名解析影片開始時間
    檔名範例: S01-20240909-074509-....mp4
    """
    base = os.path.basename(video_path)
    name, _ = os.path.splitext(base)
    parts = name.split('-')
    if len(parts) < 3: return None
    try:
        return datetime.strptime(parts[1] + parts[2], "%Y%m%d%H%M%S")
    except: return None


def parse_response(response: str):
    """
    解析模型回傳的文字 (針對 0/1/2/3 分類)
    
    預期輸入: "1", "0", "2", "3"
    輸出: "3" (代表推床推出 SEND), "2" (代表推床推入 ENT), "1" (代表門被打開), "0" (排除/無人/門關)
    """
    if not response: return "0"
    
    # 轉成字串並去除頭尾空白與換行
    response = str(response).strip()
    
    # 因應 VLM 有時候廢話很多
    # 不要用單純的 in 來判斷，以免抓到錯誤的數字。我們抓第一個出現的 0, 1, 2, 3。
    import re
    match = re.search(r'[0123]', response)
    if match:
        return match.group(0)
    
    return "0" # 預設回傳 0 (保守估計)
