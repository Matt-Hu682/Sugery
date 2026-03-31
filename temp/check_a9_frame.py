"""
擷取 A9 影片的第一幀，加上格線和座標，存成圖片供檢視裁切區域。
"""
import cv2
import os
import sys

# S04 是 A9 的房內攝影機 (Room)
video_path = "/home/cvlabgodzilla/Desktop/Sugery_AI/data/20231116/S04-20231116-100003-1700100003.mp4"
output_dir = "/home/cvlabgodzilla/Desktop/Sugery_AI/outputs"

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    print("無法讀取影片")
    sys.exit(1)

h, w = frame.shape[:2]
print(f"影片解析度: {w} x {h}")

# 畫格線 (每 100 像素)
for x in range(0, w, 100):
    cv2.line(frame, (x, 0), (x, h), (0, 255, 0), 1)
    cv2.putText(frame, str(x), (x + 2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

for y in range(0, h, 100):
    cv2.line(frame, (0, y), (w, y), (0, 255, 0), 1)
    cv2.putText(frame, str(y), (2, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

# 中心十字
cv2.line(frame, (w//2, 0), (w//2, h), (0, 0, 255), 2)
cv2.line(frame, (0, h//2), (w, h//2), (0, 0, 255), 2)

# 標註四角座標
cv2.putText(frame, f"(0,0)", (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
cv2.putText(frame, f"({w},{h})", (w-120, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
cv2.putText(frame, f"Center({w//2},{h//2})", (w//2+5, h//2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

# 存圖
out_path = os.path.join(output_dir, "A9_S04_frame_with_grid.jpg")
cv2.imwrite(out_path, frame)
print(f"已存至: {out_path}")

# 也存一份不帶格線的原圖
cap2 = cv2.VideoCapture(video_path)
ret2, frame2 = cap2.read()
cap2.release()
out_path2 = os.path.join(output_dir, "A9_S04_frame_original.jpg")
cv2.imwrite(out_path2, frame2)
print(f"原圖已存至: {out_path2}")
