# realtime_pipeline.py
"""Realtime voting pipeline for Surgery and Door events.

This module keeps the public RealtimePipeline API stable while the voting,
Surgery state machine, and Door state machine live in smaller helper modules.
"""

from pipeline.door_state_machine import DoorStateMachineMixin
from pipeline.surgery_state_machine import SurgeryStateMachineMixin
from pipeline.voting import VotingMixin


class RealtimePipeline(VotingMixin, SurgeryStateMachineMixin, DoorStateMachineMixin):
    def __init__(self, half_window=5, stable_frame=900, max_gap_frame=50,
                send_confirm_threshold=900, task_type="Surgery"):
        """
        Args:
            half_window: 投票視窗大小 (前後各看幾幀)
            stable_frame: ENT 確認需要觀察的穩定幀數 (預設 900 = 3分鐘 @5fps)
            max_gap_frame: ENT 穩定期允許的最大遮擋幀數
            send_confirm_threshold: SEND 確認觀察窗大小
            task_type: "Surgery" 或 "Door"，決定使用哪種偵測邏輯
        """
        self.task_type = task_type

        # === 投票參數 ===
        self.half_window = half_window
        self.window_size = half_window * 2 + 1

        # === 原始結果緩衝區 ===
        self.raw_statuses: list[int] = []       # 原始 status (int)
        self.frame_metadata: list[dict] = []     # [{frame_idx, video_time, real_time, video_name}, ...]

        # === 投票結果 ===
        self.voted_statuses: list[int] = []     # 投票後的 status (int)

        # === 事件偵測狀態機 (通用) ===
        self.current_confirmed_state = 0   # 目前確認狀態: 0=非手術, 1=手術中
        self.confirmed_events = []         # 已確認的事件列表
        self.latest_event = None           # 最新一筆確認的事件 dict
        self._last_surgery_end_idx = None  # 上次手術結束的 index

        # === Surgery 模式參數 ===
        self.stable_frame = stable_frame
        self.max_gap_frame = max_gap_frame
        self.send_confirm_threshold = send_confirm_threshold
        self.min_interval_frames = 900

        # Surgery 狀態機變數
        self._ent_candidate_idx = None
        self._ent_check_idx = None
        self._ent_gap_start = None
        self._send_candidate_idx = None

        # === Door 模式參數 ===
        self.door_ent_check_window = 25        # ENT 穩定期 25f = 5 秒 (推床過門檻很快，5秒即可確認)
        self.door_send_check_window = 25       # SEND 穩定期 25f = 5 秒
        self.door_max_zero_tolerance = 10      # 每段連續 0 最多容許 10f (過門中途稍微沒抓到)
        self.door_min_zero_hold = 300          # 連續 0 達 300f = 1 分鐘才算活動結束 (確保徹底離開)
        self.door_ent_to_send_min_gap = 4500   # ENT 開始到 SEND 最少間隔 4500f = 15分鐘
        self.door_cooldown = 900               # SEND 結束後冷卻 900f = 3 分鐘

        # Door 狀態機變數
        self._door_state = 'IDLE'              # IDLE / ENT_CHECKING / ENT_ACTIVE
        # WAITING_SEND / SEND_CHECKING / SEND_ACTIVE
        self._door_candidate_idx = None        # 目前候選的起始 index
        self._door_zero_run = 0                # 目前連續 0 的計數
        self._door_last_one_idx = None         # 最後一個 voted=1 的 index
        self._door_ent_start_idx = None        # 確認的 ENT 開始 index (用於計算 gap)

    # 入口
    def push_frame_result(self, status, frame_idx, video_time, real_time, video_name):
        """
        AI 分析完一幀後呼叫此方法把狀態丟進來這裡。
        內部會自動做延遲投票 + 增量事件偵測。
        """
        status = int(status) if str(status).isdigit() else 0
        # 對 Door 任務，把 2 也視為 1 (有人進出)
        binary_status = 1 if status >= 1 else 0

        self.raw_statuses.append(binary_status) # 存原始狀態
        self.frame_metadata.append({  #存時間資訊
            'frame_idx': frame_idx,
            'video_time': video_time,
            'real_time': real_time,
            'video_name': video_name,
        })

        # 新邏輯：如果 status == 2 (推入) 或 3 (推出)，直接由 VLM 回報確認跳過投票！
        if self.task_type == "Door":
            if status == 2:
                self._door_direct_confirm(event_type="ENT")
            elif status == 3:
                self._door_direct_confirm(event_type="SEND")

        # 延遲投票: 只要累積夠 half_window 幀，就可以對較早的幀做置中投票
        self._try_delayed_vote()

    def get_current_state(self):
        """
        回傳目前系統狀態，供 OSD 疊加或 terminal 顯示。

        Returns:
            dict: {
                'confirmed_state': 0 或 1,
                'confirmed_state_text': '非手術' 或 '手術中',
                'confirmed_events': [...],
                'latest_event': {...} 或 None,
                'voted_count': int,
                'raw_count': int,
                'pending': str 或 None,  # 'ENT候選中' / 'SEND候選中'
            }
        """
        pending = None
        if self.task_type == "Door":
            if self._door_state in ('ENT_CHECKING',):
                pending = 'ENT候選中'
            elif self._door_state in ('SEND_CHECKING',):
                pending = 'SEND候選中'
            elif self._door_state == 'ENT_ACTIVE':
                pending = 'ENT活動中'
            elif self._door_state == 'WAITING_SEND':
                pending = '等待SEND'
            elif self._door_state == 'SEND_ACTIVE':
                pending = 'SEND活動中'
        else:
            if self._ent_candidate_idx is not None:
                pending = 'ENT候選中'
            elif self._send_candidate_idx is not None:
                pending = 'SEND候選中'

        if self.task_type == "Door":
            state_text = '出入中' if self.current_confirmed_state == 1 else '非出入'
        else:
            state_text = '手術中' if self.current_confirmed_state == 1 else '非手術'

        return {
            'confirmed_state': self.current_confirmed_state,
            'confirmed_state_text': state_text,
            'confirmed_events': list(self.confirmed_events),
            'latest_event': self.latest_event,
            'voted_count': len(self.voted_statuses),
            'raw_count': len(self.raw_statuses),
            'pending': pending,
        }

    def get_event_summary(self):
        """
        回傳成對的 ENT/SEND 事件摘要 (與 analyze_csv.py 格式一致)。
        """
        summary = []
        last_ent = None
        surg_idx = 1

        for evt in self.confirmed_events:
            if evt['event_type'] == 'ENT':
                last_ent = evt
            elif evt['event_type'] == 'SEND' and last_ent:
                summary.append({
                    'Surgery_No': f"第 {surg_idx} 刀",
                    'Type': 'ENT',
                    'Video_Time': last_ent['video_time'],
                    'Real_Time': last_ent['real_time'],
                    'Video_Name': last_ent['video_name'],
                })
                summary.append({
                    'Surgery_No': f"第 {surg_idx} 刀",
                    'Type': 'SEND',
                    'Video_Time': evt['video_time'],
                    'Real_Time': evt['real_time'],
                    'Video_Name': evt['video_name'],
                })
                last_ent = None
                surg_idx += 1

        return summary

    def get_all_events(self):
        """
        回傳所有已確認的單獨事件列表 (不管是否成對)。
        """
        return list(self.confirmed_events)
