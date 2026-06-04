# Sugery AI - Door 與 Surgery 同步分析系統

本專案目前的主程式是 `src/main_parallel.py`。舊版 README 中提到的 `batch_runner.processor`、雙 GPU 批次流程與 `tmux` 腳本不是目前主要使用路線；若要理解或執行現在的系統，請以本文件與 `src/main_parallel.py` 為準。

## 目前主入口

```bash
cd /home/cvlabgodzilla/Desktop/Sugery
PYTHONPATH=src python3 src/main_parallel.py
```

程式會同步讀取同一資料集中的 Door 視角與 Surgery/Room 視角影片，逐幀分析兩個視角，最後輸出正式手術事件報告、完整事件紀錄、逐幀 raw CSV、live monitor 狀態檔，以及事件前後剪輯影片。

## 系統目標

系統要解決的核心問題是：判斷手術開始與結束事件是否真的成立。

單靠 Surgery/Room 視角的 VLM 可能會誤判，因此目前流程加入 Door 視角作為驗證條件：

1. Door 視角用 OpenCV 判斷門是否開啟。
2. Surgery/Room 視角用 Qwen3-VL 判斷手術台是否進入手術狀態。
3. Surgery pipeline 產生 `ENT` 或 `SEND` 事件。
4. 只有當該 Surgery 事件符合方向性的 Door OPEN 時間窗，事件才會寫進正式報告。
5. Door 沒開的 Surgery 事件會保留在 unified events CSV，但不計入正式 result CSV。

## 執行環境

主要 Python 套件列在 `requirements.txt`：

```text
torch
torchvision
transformers
accelerate
qwen-vl-utils
opencv-python
numpy
Pillow
tiktoken
fastapi
uvicorn
```

模型預設放在：

```text
models/Qwen3-VL-8B-Instruct-FP8
```

主程式目前在 `src/main_parallel.py` 內固定指定：

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

因此目前主流程使用 GPU 0。若要換 GPU，需要修改這一行或改成外部環境變數控制。

## 主要設定

設定集中在 `src/config.py`。

### 路徑設定

```python
BASE_DIR = "/home/cvlabgodzilla/Desktop/Sugery"
OUTPUT_BASE_DIR = ".../Sugery/test-result"
TEST_VIDEO_BASE = ".../data_video/mask_video_202401"
MODEL_PATH = os.path.join(BASE_DIR, "models", "Qwen3-VL-8B-Instruct-FP8")
```

可用環境變數覆蓋：

```bash
SURGERY_BASE_DIR=/path/to/project
SURGERY_OUTPUT_BASE_DIR=/path/to/output
SURGERY_TEST_VIDEO_BASE=/path/to/video/root
```

### 資料集選擇

```python
TARGET_DATASETS = None
```

- `None`: 跑 `TEST_VIDEO_BASE` 底下所有資料夾。
- 字串，例如 `"20231228"`: 只跑單一資料集。
- list，例如 `["20231228", "20231229"]`: 跑多個指定資料集。

### 房間與攝影機

```python
ROOM = "A8"

OR_SETTING = {
    "A8": ["S01", "S02"],
    "A9": ["S03", "S04"],
}

CAMERA_SETTING = {
    "S01": "Door",
    "S02": "Room",
    "S03": "Door",
    "S04": "Room",
}
```

`main_parallel.py` 會依 `ROOM` 找出 Door 攝影機與 Room 攝影機，並從資料夾中收集檔名包含該攝影機代號的影片。

### 抽幀頻率

```python
STRIDE_SEC = 0.2
```

代表每 0.2 秒抽一幀分析，約等於 5 fps。

### Door 裁切與判斷門檻

```python
CROP_REGION = CROP_SETTING.get(ROOM, None)
DOOR_OPEN_CONFIRM_FRAMES = 20
DOOR_CLOSE_CONFIRM_FRAMES = 15
```

Door 視角會先裁切門口區域，再做 OpenCV 背景差分。連續達到指定幀數後才正式切換 OPEN/CLOSE，避免單幀雜訊造成誤判。

## 影片命名與配對方式

主程式透過 `collect_videos(cam_type)` 收集影片：

- `collect_videos("Door")`: 收集 Door 攝影機影片。
- `collect_videos("Room")`: 收集 Room 攝影機影片。

影片需放在 `VIDEO_DIRS` 或 `VIDEO_DIR` 對應的資料夾中，檔名需包含攝影機代號，例如 `S01`、`S02`。

檔名通常格式：

```text
S01-YYYYMMDD-HHMMSS-xxxx.mp4
S02-YYYYMMDD-HHMMSS-xxxx.mp4
```

配對方式是排序後依序 zip：

```python
pairs = list(zip(door_videos[:pair_count], room_videos[:pair_count]))
```

如果 Door 與 Room 數量不同，只會處理數量較少的一方。配對是否正確高度依賴檔名排序與資料夾內容。

## 主程式執行流程

`src/main_parallel.py` 的主流程如下：

```text
啟動 main_parallel.py
  -> 收集 Door 影片
  -> 收集 Room/Surgery 影片
  -> 依順序配對 Door 與 Room
  -> 依 dataset 分組
  -> 載入 Qwen3-VL 模型
  -> 建立 RealtimePipeline
  -> 建立 DoorStage1 偵測器
  -> 每個 dataset 逐對影片同步處理
      -> 讀 Door frame
      -> 讀 Room frame
      -> Door frame 做 OpenCV 開門判斷
      -> Room frame 做 VLM Surgery 判斷
      -> Surgery status 推入 RealtimePipeline
      -> Pipeline 產生 ENT/SEND 候選事件
      -> 等 Door 時間軸 buffer 足夠
      -> 依事件方向檢查 Door 是否 OPEN
      -> OPEN 則寫正式 result CSV 並剪輯影片
      -> CLOSE 則只寫 unified events，不計入正式報告
  -> dataset 結束時 flush pipeline
  -> 排序 unified events CSV
  -> 結束
```

## 相關檔案與職責

### `src/main_parallel.py`

目前實際使用的主程式。

負責：

- 設定 GPU。
- 收集 Door 與 Room 影片。
- 依順序配對影片。
- 依 dataset 分組處理。
- 同步讀取兩個視角的 frame。
- 呼叫 Door 開門偵測。
- 呼叫 Surgery VLM 分析。
- 呼叫 `RealtimePipeline` 產生手術事件。
- 用 Door OPEN 狀態過濾 Surgery 事件。
- 寫出 raw CSV、unified events CSV、正式 result CSV。
- 觸發事件剪輯。
- 更新 live monitor 檔案。

### `src/config.py`

全域設定檔。

負責：

- 專案路徑。
- 輸出路徑。
- 影片資料根目錄。
- 模型路徑。
- 要跑的資料集。
- 房間與攝影機對應。
- Door/Room 攝影機選擇。
- Door 裁切範圍。
- Door OpenCV 門檻。
- Surgery pipeline 穩定幀數。
- VLM 生成參數。

### `src/door_stage1.py`

Door 視角的第一階段開門偵測。

做法：

1. 將裁切後的 Door frame 轉灰階。
2. Gaussian blur 降低雜訊。
3. 與關門模板或背景模型做 `absdiff`。
4. 依 ROI 權重計算平均差異分數與變動像素比例。
5. 分數與比例都超過門檻時，回傳 raw open。
6. 門未開時用 EMA 更新背景；門開時不更新，避免把開門畫面學成背景。

`main_parallel.py` 會再做 debounce：

- raw open 連續 `DOOR_OPEN_CONFIRM_FRAMES` 幀才變成 OPEN。
- raw close 連續 `DOOR_CLOSE_CONFIRM_FRAMES` 幀才變成 CLOSE。

### `src/core.py`

Surgery 視角的 VLM 分析器。

核心 class：

```python
PatientStatusAnalyzer
```

初始化時會載入：

- `Qwen3VLForConditionalGeneration`
- `AutoProcessor`
- `VLMRunner`

每幀分析流程：

1. BGR frame 轉 RGB。
2. 轉為 PIL image。
3. 從 `prompts.get_prompt("Surgery")` 取得 prompt。
4. 呼叫 `VLMRunner.run()`。
5. 回傳模型判斷結果與推論時間。

目前 Surgery 模式主要輸出：

- `1`: 判斷為手術中或手術台上有人。
- `0`: 判斷為非手術狀態。

### `src/infra/vlm_runner.py`

包裝 Qwen3-VL 實際推論。

負責：

- 套用 chat template。
- 處理 image/video input。
- 呼叫 `model.generate()`。
- decode 模型輸出。
- 使用 `utils.parse_response()` 解析第一個 `0/1/2/3`。
- 回傳整數結果與推論時間。

### `src/prompts/prompts.py`

集中管理 VLM prompt。

目前 `main_parallel.py` 使用的是 Surgery prompt：

- 看圖片中間的手術台。
- 判斷是否有病人躺在手術台上。
- 判斷是否有兩位以上站著的人圍繞手術台且朝向床上。
- 符合輸出 `1`，否則輸出 `0`。

檔案中也保留 Door prompt 與其他模式，但目前 `main_parallel.py` 的 Door 判斷走 OpenCV，不走 Door VLM prompt。

### `src/realtime_pipeline.py`

把逐幀 VLM status 轉換成事件的狀態機。

在 `main_parallel.py` 中以 Surgery 模式建立：

```python
RealtimePipeline(
    half_window=25,
    stable_frame=900,
    max_gap_frame=50,
    send_confirm_threshold=900,
    task_type="Surgery",
)
```

功能：

- 保存 raw status。
- 做延遲投票，降低單幀誤判。
- 偵測手術開始 `ENT`。
- 偵測手術結束 `SEND`。
- 保存完整事件列表。
- dataset 結束時 flush 尚未投票的尾端幀。
- 必要時用 `force_close_pending_send()` 補發尾端 SEND。

### `src/live_monitor.py`

輸出即時監控檔案。

`LiveMonitorWriter` 會定期寫出：

- `door_frame.jpg`
- `door_crop.jpg`
- `door_crop_roi.jpg`
- `room_frame.jpg`
- `status.json`

輸出位置：

```text
OUTPUT_BASE_DIR/live/run_{timestamp}/
```

這些檔案可供 FastAPI 或其他前端即時查看目前分析狀態。

### `src/utils.py`

共用工具。

主要功能：

- `video_start_time(video_path)`: 從影片檔名解析開始時間。
- `parse_response(response)`: 從 VLM 回覆文字中擷取第一個 `0/1/2/3`。

`main_parallel.py` 依賴影片開始時間計算 `real_time`，因此檔名格式很重要。

### `scripts/live_monitor_api.py`

提供 live monitor API 的輔助程式。可搭配 `OUTPUT_BASE_DIR/live/...` 下的即時輸出檔查看目前畫面與狀態。

### `scripts/preview_crop.py`、`scripts/select_door_roi.py`

Door 裁切與 ROI 調整輔助工具。

可用於確認 `CROP_REGION` 與 Door ROI 是否切到正確位置。

### `scripts/diagnose_config.py`

設定診斷工具，用來確認目前路徑、資料集與設定是否符合預期。

### `src/batch_runner/*` 與 `src/scripts/run_gpu*.py`

舊版多日期、多 GPU 批次流程相關檔案。目前主程式不是這條路線。

若要跑目前同步 Door/Room 的流程，請使用：

```bash
PYTHONPATH=src python3 src/main_parallel.py
```

## Door 與 Surgery 的時間同步

Door 與 Room 影片各自有開始時間，程式透過檔名解析：

```text
S01-YYYYMMDD-HHMMSS-xxxx.mp4
```

每一幀會計算：

- `video_time`: 該幀在影片內的時間，例如 `00:03:12`
- `real_time`: 影片開始時間加上目前秒數，例如 `15:41:23`

Surgery 事件不會立刻被判斷 Door 狀態，而是先進入 `pending_surgery_events`。

程式會依事件類型等待 Door 時間軸：`ENT` 至少等 Door 追到事件時間，`SEND` 會等事件後 60 秒，避免 Door 還沒跑到可驗證區間就過早過濾事件。其他未知事件類型則使用 `BUFFER_SECONDS = 10` 作為保守 buffer。

## Door 過濾 Surgery 事件的規則

正式 result CSV 只接受 Door OPEN 附近的 Surgery 事件。

相關設定：

```python
DOOR_EVENT_MATCH_BEFORE_FRAMES = 300
DOOR_EVENT_MATCH_AFTER_FRAMES = 300
STRIDE_SEC = 0.2
```

換算時間約為：

```text
300 frames * 0.2 sec = 60 sec
```

目前採方向性判斷：`ENT` 只看事件前 60 秒內是否 Door OPEN；`SEND` 只看事件後 60 秒內是否 Door OPEN。

如果 Door 沒開：

- 事件仍寫入 unified events CSV。
- `door_status` 會是 `CLOSE`。
- 不寫入正式 result CSV。

## 輸出檔案

### Raw CSV

位置格式：

```text
{CSV_OUTPUT 目錄}/{dataset_name}/surgery_report_Surgery_{dataset_name}_{run_date}_parallel.csv
```

欄位：

```text
Video_name, frame_index, video_time, real_time,
status, voted_status, infer_time,
door_open, door_score, door_ratio
```

用途：逐幀記錄 Surgery VLM 結果與 Door 狀態。

### Unified Events CSV

位置格式：

```text
{CSV_OUTPUT 目錄}/{dataset_name}/all_events/Unified_Events_{dataset_name}_{run_date}.csv
```

欄位：

```text
source, event_type, video_time, real_time, video_name, door_status
```

用途：保留所有 Door 與 Surgery 事件，包含被 Door CLOSE 濾掉的 Surgery 誤判。

### Result CSV

位置格式：

```text
{CSV_OUTPUT 目錄}/{dataset_name}/result/{ROOM}/Realtime_Events_Surgery_{dataset_name}_{run_date}.csv
```

欄位：

```text
Surgery_Date, Surgery_No, Type, Real_Time
```

用途：正式輸出，只保留通過方向性 Door OPEN 條件的 `ENT` / `SEND`。

### 事件剪輯影片

位置：

```text
{result CSV 目錄}/videos/
```

每個正式事件會用 Room 視角影片裁剪事件前後各 90 秒，總長約 3 分鐘。

### Door 開門期間 frame

位置：

```text
OUTPUT_BASE_DIR/{dataset_name}/door_open_frames/OPEN_{開門時間}/
```

用途：保存 Door 開門期間的原始畫面，方便之後人工檢查。

### Live Monitor

位置：

```text
OUTPUT_BASE_DIR/live/run_{timestamp}/
```

內容：

```text
door_frame.jpg
door_crop.jpg
door_crop_roi.jpg
room_frame.jpg
status.json
```

## 常用操作

### 執行主程式

```bash
cd /home/cvlabgodzilla/Desktop/Sugery
PYTHONPATH=src python3 src/main_parallel.py
```

### 只跑特定資料集

修改 `src/config.py`：

```python
TARGET_DATASETS = "20231228"
```

或：

```python
TARGET_DATASETS = ["20231228", "20231229"]
```

再執行：

```bash
PYTHONPATH=src python3 src/main_parallel.py
```

### 查看目前 Git 狀態

```bash
git status --short
```

### 查看目前輸出

依 `src/config.py` 的 `OUTPUT_BASE_DIR` 和 `CSV_OUTPUT` 位置查看。

若正式 result CSV 事件少於預期，先看 unified events CSV，確認事件是否被 Door CLOSE 過濾。

## Debug 建議

### 事件不見了

先檢查：

1. `Unified_Events_*.csv` 是否有 Surgery 事件。
2. 該事件的 `door_status` 是否為 `CLOSE`。
3. Door open frame 是否有存圖。
4. `door_score` 與 `door_ratio` 是否接近門檻。
5. `CROP_REGION` 是否切到正確門口區域。

### Door 一直誤判

檢查：

1. `door_crop.jpg`
2. `door_crop_roi.jpg`
3. `templates/door_closed_{ROOM}.jpg`
4. `DOOR_STAGE1_DIFF_THRESHOLD`
5. `DOOR_STAGE1_CHANGED_RATIO`
6. `DOOR_STAGE1_ACTIVE_ROIS`

### Surgery 一直誤判

檢查：

1. Room 影片是否正確配對。
2. prompt 是否符合目前畫面。
3. `status` 與 `voted_status` 差異。
4. `RealtimePipeline` 的 `stable_frame` 與 `send_confirm_threshold` 是否過長或過短。

### Door/Room 不同步

檢查：

1. 檔名中的日期與時間是否正確。
2. Door 與 Room 影片排序後是否真的是同一時間段。
3. `video_start_time()` 是否能解析檔名。
4. unified events CSV 中 Door 與 Surgery 的 `real_time` 是否合理。

## 目前不作為主流程的檔案

以下檔案仍在專案中，但目前不是主執行流程：

- `src/batch_runner/processor.py`
- `src/batch_runner/config.py`
- `src/main_realtime.py`
- `src/scripts/run_gpu0.py`
- `src/scripts/run_gpu1.py`
- `src/scripts/run_with_tmux.sh`

這些檔案偏向舊版單視角、批次、多 GPU 或 tmux 流程。除非要維護舊流程，否則目前請優先閱讀與使用 `src/main_parallel.py`。

## 一句話總結

目前系統是以 `src/main_parallel.py` 為入口的雙視角同步分析流程：Door 視角負責驗證門是否開啟，Surgery/Room 視角負責用 VLM 找手術開始與結束，最後只把 Door OPEN 附近的 Surgery 事件輸出成正式報告。
