# door_stage1.py
"""
Door Stage 1：純 OpenCV 背景差分偵測門是否開啟。
不依賴 VLM，可單獨被 main_parallel.py 或 core.py 呼叫。
"""

import cv2

from config import (
    DOOR_STAGE1_ACTIVE_ROIS, # 觀察那些ROI區域
    DOOR_STAGE1_BG_ALPHA, #背景更新速度
    DOOR_STAGE1_CHANGED_RATIO, # 1變動像素比例
    DOOR_STAGE1_CLOSED_TEMPLATE, # 固定關門模板
    DOOR_STAGE1_DIFF_THRESHOLD, # 平均差異分數
    DOOR_STAGE1_PIXEL_DIFF_THRESHOLD, #像素差異多少才算有變
)


class DoorStage1:
    """
    對單張 BGR 幀做加權 ROI 背景差分，判斷門是否開著。

    用法：
        detector = DoorStage1()
        result = detector.detect(frame_bgr)   # True / False
        print(detector.last_score, detector.last_ratio)

    可以在每支影片開始時呼叫 reset() 清除背景模型。
    """

    def __init__(self):
        self._bg = None          # float32 灰階背景
        self._template_bg = None # 固定關門模板灰階背景
        self.last_score = 0.0    # 最後一幀的加權平均 diff 分數
        self.last_ratio = 0.0    # 最後一幀的加權變動像素比例
        self._load_template()

    def _load_template(self):
        """載入固定關門模板；模板不存在時保留 None，改用舊的第一幀背景。"""
        template = cv2.imread(DOOR_STAGE1_CLOSED_TEMPLATE) # 讀取模板圖片
        if template is None:
            self._template_bg = None
            return

        gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)#把BGR轉成灰階
        gray = cv2.GaussianBlur(gray, (5, 5), 0) # 做模糊處理
        self._template_bg = gray.astype("float32")# 背景轉gray

    def reset(self):
        """清除背景模型（切換影片時呼叫）。"""
        self._bg = None if self._template_bg is None else self._template_bg.copy()
        self.last_score = 0.0
        self.last_ratio = 0.0

    def detect(self, frame_bgr) -> bool:
        """
        判斷這一幀門是否開著。
        - 第一幀：初始化背景，回傳 False。
        - 門沒開：更新背景 EMA，回傳 False。
        - 門開：不更新背景（保留關門時的樣子），回傳 True。
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) # 把BGR轉成灰階
        gray = cv2.GaussianBlur(gray, (5, 5), 0) # 做模糊處理

        if self._bg is None:
            if self._template_bg is not None and self._template_bg.shape == gray.shape:
                self._bg = self._template_bg.copy()
            else:
                self._bg = gray.astype("float32")
                self.last_score = 0.0
                self.last_ratio = 0.0
                return False

        if self._bg.shape != gray.shape:
            if self._template_bg is not None:
                resized = cv2.resize(self._template_bg, (gray.shape[1], gray.shape[0]))
                self._bg = resized.astype("float32")
            else:
                self._bg = gray.astype("float32")
                self.last_score = 0.0
                self.last_ratio = 0.0
                return False


        bg = cv2.convertScaleAbs(self._bg)
        diff = cv2.absdiff(gray, bg)
        h, w = diff.shape[:2]

        weighted_score = 0.0
        weighted_ratio = 0.0
        total_weight = 0.0

        for x1r, y1r, x2r, y2r, weight in DOOR_STAGE1_ACTIVE_ROIS:
            x1 = max(0, min(w - 1, int(w * x1r)))
            y1 = max(0, min(h - 1, int(h * y1r)))
            x2 = max(x1 + 1, min(w, int(w * x2r)))
            y2 = max(y1 + 1, min(h, int(h * y2r)))
            roi = diff[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            roi_weight = max(float(weight), 0.0)
            changed_ratio = cv2.countNonZero(
                cv2.threshold(roi, DOOR_STAGE1_PIXEL_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
            ) / float(roi.size)
            weighted_score += float(roi.mean()) * roi_weight
            weighted_ratio += changed_ratio * roi_weight
            total_weight += roi_weight

        if total_weight <= 0:
            return False

        score = weighted_score / total_weight
        ratio = weighted_ratio / total_weight
        self.last_score = score
        self.last_ratio = ratio

        is_open = score >= DOOR_STAGE1_DIFF_THRESHOLD and ratio >= DOOR_STAGE1_CHANGED_RATIO
        if not is_open:
            # 門未開時持續更新背景（EMA）
            cv2.accumulateWeighted(gray, self._bg, DOOR_STAGE1_BG_ALPHA)

        return is_open

    def detect_int(self, frame_bgr) -> int:
        """與 detect() 相同，但回傳 1 / 0（供舊版 core.py 使用）。"""
        return 1 if self.detect(frame_bgr) else 0
