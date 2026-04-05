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
    解析模型回傳的文字 (針對 0/1/2 分類)
    
    預期輸入: "1", "0", "2", "Status: 2"
    輸出: "2" (步行進出), "1" (推床/輪椅進出), "0" (排除/無人)
    """
    if not response: return "0"
    
    # 轉成字串並去除頭尾空白與換行，確保比對準確
    response = str(response).strip()
    
    # 優先檢查 2 和 1，捕捉所有有病患進出的事件
    if "2" in response:
        return "2"
    elif "1" in response:
        return "1"
    elif "0" in response:
        return "0"
    
    return "0" # 預設回傳 0 (保守估計)



'''
def parse_response(response: str):
    """解析模型回傳的文字

    解析 VLM 模型回傳的文字，擷取「判斷」欄位的值。
    
    預期 VLM 回應格式：
        判斷: 是
        信心度: 85
        理由: ...
    
    
    """
    m = re.search(r"判斷[:：]\s*([^\n\r]+)", response)
    if not m: return None
    val = m.group(1).strip().replace(" ", "")
    if val.startswith("是"): return "是"
    if val.startswith("否"): return "否"
    return "不確定"

def extract_dual_frames(path_s01, path_s02, start_frame, num_frames=8):
    """
    同時讀取兩個視角（S01 門口 + S02 內部）的影片，
    將對應幀縮放後左右拼接成單一畫面序列。
    
    流程：
    1. 開啟兩個影片檔案
    2. 定位到 start_frame
    3. 逐幀讀取 num_frames 張
    4. 統一縮放到 (VIS_WIDTH_SINGLE, VIS_HEIGHT)
    5. 左右拼接（S01 在左，S02 在右）
    6. 轉成 PIL.Image 並收集到 list
    """
    
    #開啟s01、s02影片
    cap1 = cv2.VideoCapture(path_s01)
    cap2 = cv2.VideoCapture(path_s02) if path_s02 else None

    #取
    total_frames = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    if start_frame >= total_frames:
        cap1.release()
        if cap2: cap2.release()
        return []
    
    #定位到指定的起始幀
    cap1.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    if cap2: cap2.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = [] # 用來存放拼接後的畫面序列
    # 逐幀擷取
    for _ in range(num_frames):
        ret1, frame1 = cap1.read()
        if not ret1: break
        
        f2 = None
        if cap2:
            ret2, temp = cap2.read()
            if ret2: f2 = temp
        
        # 統一縮放
        frame1 = cv2.resize(frame1, (VIS_WIDTH_SINGLE, VIS_HEIGHT))
        
        if f2 is not None:
            f2 = cv2.resize(f2, (VIS_WIDTH_SINGLE, VIS_HEIGHT))
            combined = cv2.hconcat([frame1, f2])
        else:
            combined = frame1
            
        frames.append(Image.fromarray(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)))

    cap1.release()
    if cap2: cap2.release()
    return frames
'''