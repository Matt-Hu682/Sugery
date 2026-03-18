# main.py
import os
from config import VIDEO_DIR, CSV_OUTPUT, STRIDE_SEC, SHOW_WINDOW, CURRENT_TEST, TARGET_CAMERAS
from core import PatientStatusAnalyzer

# 指定使用哪一張顯示卡 (如果只有一張 4090，設 0 沒問題)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main():
    #  初始化 AI 推論
    analyzer = PatientStatusAnalyzer()

    #  檢查資料夾是否存在
    if not os.path.exists(VIDEO_DIR):
        print(f"Path not found: {VIDEO_DIR}")
        return

    print(f" 目前允許分析的攝影機視角: {TARGET_CAMERAS}")

    #  搜尋影片 (動態過濾)
    video_files = []
    for f in os.listdir(VIDEO_DIR):
        if f.lower().endswith(('.mp4', '.avi')):
            # 只加入符合指定攝影機的影片
            if any(cam in f for cam in TARGET_CAMERAS):
                video_files.append(os.path.join(VIDEO_DIR, f))
    
    video_files.sort()
    print(f" 共找到 {len(video_files)} 支符合條件的影片。")

    # 4. 逐一分析
    for idx, video_path in enumerate(video_files):
        print(f"\nProcessing [{idx + 1}/{len(video_files)}]: {video_path}")
        
        # 呼叫 core.py 的核心分析邏輯
        analyzer.run_analysis(
            video_path=video_path,
            csv_path=CSV_OUTPUT, # 傳入基底路徑，core.py 會自動幫它加上 _patient 或 _transfer 後綴
            stride_sec=STRIDE_SEC,
            current_task=CURRENT_TEST, # 傳入當前測試模式，會根據測試config的設定決定要執行哪個任務
            show_window=SHOW_WINDOW
        )

if __name__ == "__main__":
    main()