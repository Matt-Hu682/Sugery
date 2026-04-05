#!/bin/bash
# tmux 分割終端腳本：同時監控 GPU0 和 GPU1 的處理進度

SESSION_NAME="gpu_proc"
WINDOW_NAME="2gpu"
VENV="/home/ai/Sugery_AI/VLM-1/bin/python3"

cd /home/ai/Sugery_AI

# 檢查是否已存在 session
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "❌ session 已存在，先刪除："
    echo "   tmux kill-session -t $SESSION_NAME"
    exit 1
fi

echo "🚀 建立 tmux session..."
echo "   上: GPU0 (10 個日期)"  
echo "   下: GPU1 (11 個日期)"
echo ""
echo "✨ 快捷鍵:"
echo "   Ctrl+B ↑↓: 切換窗格"
echo "   Ctrl+B D: 離開並保持執行"
echo "   Ctrl+B X: 殺死窗格"
echo ""

# 建立新 session
tmux new-session -d -s $SESSION_NAME -x 200 -y 50 -c /home/ai/Sugery_AI

# 分割為上下兩個窗格（比例 25:25）
tmux split-window -t $SESSION_NAME -v -l 25 -c /home/ai/Sugery_AI

# 上面窗格：GPU0
tmux send-keys -t $SESSION_NAME:0.0 "$VENV src/scripts/run_gpu0.py" Enter

# 下面窗格：GPU1  
tmux send-keys -t $SESSION_NAME:0.1 "$VENV src/scripts/run_gpu1.py" Enter

# 進入 tmux
echo "✅ 進入 tmux 會話..."
tmux attach-session -t $SESSION_NAME

