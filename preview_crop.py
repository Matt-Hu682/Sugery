# preview_crop.py
# 用法: python preview_crop.py
# 功能: 從影片中抽一幀，顯示裁切前後的對比圖，並儲存成 PNG

import cv2
import os
import sys

# ============ 設定區 ============
VIDEO_DIR = "data/20240812test"   # 影片資料夾
CROP = (400, 0, 680, 260)        # (x1, y1, x2, y2) 裁切範圍
OUTPUT_DIR = "outputs"            # 輸出資料夾
# ================================

def find_video(video_dir, camera="S01"):
    """自動找第一支符合 camera 的影片"""
    for root, _, files in os.walk(video_dir):
        for f in sorted(files):
            if camera in f and f.lower().endswith(('.mp4', '.avi')):
                return os.path.join(root, f)
    return None

def preview(video_path, crop, output_dir, n_frames=5):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"影片: {os.path.basename(video_path)}")
    print(f"解析度: {w}x{h} | 總幀數: {total} | FPS: {fps:.1f}")
    print(f"裁切範圍: x1={crop[0]}, y1={crop[1]}, x2={crop[2]}, y2={crop[3]}")

    os.makedirs(output_dir, exist_ok=True)
    x1, y1, x2, y2 = crop

    # 從影片均勻抽 n_frames 幀
    step = max(1, total // n_frames)
    saved = []
    for i in range(n_frames):
        frame_no = i * step
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        if not ret:
            continue

        # 原圖加紅框標示裁切區域
        vis_orig = frame.copy()
        cv2.rectangle(vis_orig, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(vis_orig, f"CROP: ({x1},{y1})-({x2},{y2})",
                    (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)

        # 裁切後的圖
        cropped = frame[y1:y2, x1:x2]

        # 左右合併輸出
        # 先把兩張圖都縮到同樣高度
        target_h = 480
        scale_orig   = target_h / frame.shape[0]
        scale_crop   = target_h / cropped.shape[0] if cropped.shape[0] > 0 else 1.0
        orig_resized = cv2.resize(vis_orig, (int(frame.shape[1] * scale_orig), target_h))
        crop_resized = cv2.resize(cropped,  (int(cropped.shape[1] * scale_crop), target_h))

        # 加標題文字
        pad = 40
        orig_pad = cv2.copyMakeBorder(orig_resized, pad, 0, 0, 0, cv2.BORDER_CONSTANT, value=(30,30,30))
        crop_pad = cv2.copyMakeBorder(crop_resized, pad, 0, 0, 0, cv2.BORDER_CONSTANT, value=(30,30,30))
        cv2.putText(orig_pad, "Original (red box = crop area)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(crop_pad, f"Cropped ({x2-x1}x{y2-y1})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,128), 2)

        combined = cv2.hconcat([orig_pad, crop_pad])
        out_path = os.path.join(output_dir, f"preview_crop_frame{i+1:02d}.png")
        cv2.imwrite(out_path, combined)
        saved.append(out_path)
        print(f"  儲存: {out_path}")

    cap.release()
    print(f"\n完成！共輸出 {len(saved)} 張預覽圖到 {output_dir}/")
    return saved

if __name__ == "__main__":
    cam = "S01"
    if len(sys.argv) > 1:
        cam = sys.argv[1]

    video = find_video(VIDEO_DIR, cam)
    if video is None:
        print(f"[Error] 找不到 {cam} 的影片，請確認 VIDEO_DIR={VIDEO_DIR}")
        sys.exit(1)

    preview(video, CROP, OUTPUT_DIR)
