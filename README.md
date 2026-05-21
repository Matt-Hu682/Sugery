# Sugery AI 多日期處理系統 (即時模式)

## 快速開始

```bash
# 1. 查看目前設定
cd /home/cvlabgodzilla/Desktop/Sugery
PYTHONPATH=src python3 -c "from batch_runner.config import print_config; print_config()"

# 2. 啟動多GPU處理
PYTHONPATH=src python3 -m batch_runner.processor
```
或者，使用 `tmux` 啟動：
```bash
bash src/scripts/run_with_tmux.sh
```

## 運行原理

系統會自動：
1. 🔍 掃描 `data/` 目錄中的所有日期資料夾
2. ⚙️  將日期輪轉分配給 2 個GPU
3. 🚀 啟動多行程，各 GPU 並行處理 (採用與 realtime 相同的即時 Pipeline投票機制)

## 輸出結構

```
outputs/
├── surgery_report_日期1.csv
├── surgery_report_日期2.csv
└── ...

result/
├── 日期1_房間_攝影機/
│   ├── Realtime_Events_...csv
│   └── videos/
│       └── 事件片段...mp4
└── ...
```

## 設定修改

**1. 修改全域演算法/閾值設定** (`src/config.py`)：
```python
# 修改手術室與任務
ROOM = "A9"
CURRENT_TEST = "Surgery"  

# 修改分析頻率與 Pipeline 投票視窗
STRIDE_SEC = 0.2
HALF_WINDOW = 25
```

**2. 修改多批次/跨日期排程相關設定** (`src/batch_runner/config.py`)：
```python
# 手動指定使用哪 2 個 GPU
GPU_IDS = ["...", "..."]
```

## 檔案結構

```
Sugery_AI/
├── scripts/                  # 專案工具腳本
│   ├── diagnose_config.py    # 設定/日期/GPU 診斷
│   ├── preview_crop.py       # 裁切預覽圖產生工具
│   └── setup_mig_uuid.sh     # MIG UUID 設定輔助
│
├── src/                      # 核心演算法庫
│   ├── config.py             # 核心/全域共通設定 (模型、攝像頭對應、算法閾值)
│   ├── core.py               # VLM 推論引擎 (PatientStatusAnalyzer)
│   ├── main_realtime.py      # 單機/單日期即時模式入口
│   └── realtime_pipeline.py  # 即時狀態機處理
│
├── src/batch_runner/         # (原 multi_gpu) 負責排程、多日期處理
│   ├── __init__.py
│   ├── config.py             # 繼承 src.config，負責 NAS路徑、UUID設定、自動掃描
│   └── processor.py          # 多進程即時處理引擎
│
├── src/scripts/              # GPU/tmux 輔助腳本
│   ├── run_gpu0.py           # 單跑 GPU0 分配的批次
│   ├── run_gpu1.py           # 單跑 GPU1 分配的批次
│   └── run_with_tmux.sh      # tmux 雙窗格批次輔助
│
├── tests/                    # 測試與系統檢查
│   ├── test_mini_run.py
│   └── test_system.py
│
├── outputs/preview/          # 裁切預覽圖片
└── models/                   # VLM 模型存放
```

## 常用用法

### 查看所有檢測到的日期

```bash
PYTHONPATH=src python3 -c "
from batch_runner.config import _detected_dates, PROCESS_DATES
print('所有日期:', _detected_dates)
print('處理日期:', PROCESS_DATES)
"
```

### 單獨測試某個日期/環境效果

```bash
cd /home/cvlabgodzilla/Desktop/Sugery
PYTHONPATH=src python3 src/main_realtime.py
```

### 產生裁切預覽圖

```bash
python3 scripts/preview_crop.py
```

### 監控GPU使用情況

```bash
nvidia-smi -l 1  # 實時重新整理
```

## 日期組織規則

- 自動按字母/數字順序排序
- 日期輪轉分配：日期1→GPU0, 日期2→GPU1, 日期3→GPU0, ...

## 注意事項

- ⚠️  每個 GPU 需要約 10-15GB VRAM
- ⚠️  模型路徑預設為 `models/Qwen3-VL-8B-Instruct-FP8`
- ⚠️  影片需要放在 `data/[日期]/` 子目錄中
