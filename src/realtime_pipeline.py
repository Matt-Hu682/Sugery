# realtime_pipeline.py
"""
即時手術偵測 Pipeline
- 延遲置中投票清洗 (Delayed Centered Vote)
- 增量式事件偵測 (Incremental Event Detection)

用法:
    pipeline = RealtimePipeline(half_window=5, stable_frame=900, max_gap_frame=50)
    
    # 每次 AI 分析完一幀後呼叫:
    pipeline.push_frame_result(status, frame_idx, video_time, real_time, video_name)
    
    # 取得目前狀態:
    state = pipeline.get_current_state()
    
    # 影片結束時刷出剩餘的幀:
    pipeline.flush()
"""

from datetime import datetime


class RealtimePipeline:
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
        self.door_ent_check_window = 300       # ENT 穩定期 300f = 1 分鐘
        self.door_send_check_window = 300      # SEND 穩定期 300f = 1 分鐘
        self.door_max_zero_tolerance = 50      # 每段連續 0 最多容許 50f
        self.door_min_zero_hold = 300          # 連續 0 達 300f 才算活動結束
        self.door_ent_to_send_min_gap = 4500   # ENT 開始到 SEND 最少間隔 4500f = 15分鐘
        self.door_cooldown = 900               # SEND 結束後冷卻 900f

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

        # 延遲投票: 只要累積夠 half_window 幀，就可以對較早的幀做置中投票
        self._try_delayed_vote()

    def flush(self):
        """
        影片結束時呼叫，把尾端剩餘未投票的幀全部刷出。
        尾端幀因為沒有足夠的「未來」幀，改用可用範圍內的 mean 投票。
        """
        total = len(self.raw_statuses)
        while len(self.voted_statuses) < total:
            idx = len(self.voted_statuses)
            start = max(0, idx - self.half_window)
            end = min(total, idx + self.half_window + 1)
            window = self.raw_statuses[start:end]
            voted = 1 if (sum(window) / len(window)) >= 0.5 else 0
            self.voted_statuses.append(voted)
            self._incremental_event_detect()

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



    def _try_delayed_vote(self):
        """
        對傳進來的frame做投票。
        當第 N 幀的前後各有 half_window 幀時，才對第 N 幀投票。
        ex: half_window = 25 -> 傳進來第25幀，才對第0幀做投票
        """
        # 目前有幾幀
        total = len(self.raw_statuses)

        while True:
            # 下一個要投票的 index
            vote_idx = len(self.voted_statuses)
            # 需要的最後一幀 index
            needed = vote_idx + self.half_window
            if needed >= total:
                break  # 還不夠，等更多幀進來

            # 做置中投票
            start = max(0, vote_idx - self.half_window)
            end = vote_idx + self.half_window + 1  
            window = self.raw_statuses[start:end] # 包含前後的 half_window 幀
            voted = 1 if (sum(window) / len(window)) >= 0.5 else 0
            self.voted_statuses.append(voted)

            # 新產生一個 voted_status，嘗試增量事件偵測
            self._incremental_event_detect()


    def _incremental_event_detect(self):
        """
        每次新增一個 voted_status 時呼叫。
        根據 task_type 分流到對應的狀態機。
        """
        if self.task_type == "Door":
            self._door_incremental_detect()
        else:
            self._surgery_incremental_detect()

    def _surgery_incremental_detect(self):
        """
        Surgery 模式的增量式事件偵測。
        """
        idx = len(self.voted_statuses) - 1  # 最新的 voted index
        v_status = self.voted_statuses[idx] # 投票後的狀態
        meta = self.frame_metadata[idx] #時間資訊

        # --- 間隔檢查 ---
        if self._last_surgery_end_idx is not None:
            if (idx - self._last_surgery_end_idx) < self.min_interval_frames:
                return

        # === 狀態 0 → 1: ENT 候選 ===
        if v_status == 1 and self.current_confirmed_state == 0:
            if self._ent_candidate_idx is None:
                # 新的 ENT 候選
                self._ent_candidate_idx = idx
                self._ent_check_idx = idx
                self._ent_gap_start = None
                print(f"\n偵測到狀態改變: ENT 於 {meta['video_time']}")
                print(f"  > [進入候補] 開始掃描 {self.stable_frame} frame 穩定性...")

            # 繼續穩定性檢查
            self._check_ent_stability(idx)

        # === 狀態 1 → 0: SEND 候選 ===
        elif v_status == 0 and self.current_confirmed_state == 1:
            if self._send_candidate_idx is None:
                self._send_candidate_idx = idx
                print(f"\nSEND候選起始 frame {idx} ({meta['video_time']})")

            self._check_send_stability(idx) #作穩定

        # === ENT 候選中但遇到 0 ===
        elif v_status == 0 and self._ent_candidate_idx is not None and self.current_confirmed_state == 0:
            self._check_ent_stability(idx)

        # === SEND 候選中但遇到 1 ===
        elif v_status == 1 and self._send_candidate_idx is not None and self.current_confirmed_state == 1:
            self._check_send_stability(idx)

    def _check_ent_stability(self, current_idx):
        """
        檢查 ENT 候選的穩定性。
        """
        if self._ent_candidate_idx is None:
            return

        cand_idx = self._ent_candidate_idx
        elapsed = current_idx - cand_idx

        # 穩定期已達標 → 確認 ENT
        if elapsed >= self.stable_frame:
            meta = self.frame_metadata[cand_idx]
            event = {
                'event_type': 'ENT',
                'video_name': meta['video_name'],
                'video_time': meta['video_time'],
                'real_time': meta['real_time'],
            }
            self.confirmed_events.append(event)
            self.latest_event = event
            self.current_confirmed_state = 1
            print(f"  > [確認] {meta['video_name'][:20]}... | {meta['video_time']} | ENT 紀錄成功")

            # 重置 ENT 候選
            self._ent_candidate_idx = None
            self._ent_check_idx = None
            self._ent_gap_start = None
            # 重置 SEND 候選
            self._send_candidate_idx = None
            return

        v_status = self.voted_statuses[current_idx]

        if v_status == 1:
            # 狀態正常(1)，繼續觀察
            self._ent_gap_start = None
            return

        # 遇到 0 → 遮擋處理
        if self._ent_gap_start is None:
            self._ent_gap_start = current_idx

        gap_length = current_idx - self._ent_gap_start + 1

        if gap_length >= self.max_gap_frame:
            # 遮擋超過門檻 → ENT 失敗
            meta = self.frame_metadata[self._ent_gap_start]
            print(f"    [遮擋失敗] 狀態於 {meta['video_time']} 改變超過 {self.max_gap_frame} f。")
            self._ent_candidate_idx = None
            self._ent_check_idx = None
            self._ent_gap_start = None

    def _check_send_stability(self, current_idx):
        """
        檢查 SEND 候選的穩定性。
        在觀察窗累積到 send_confirm_threshold 後，進行判定。
        """
        if self._send_candidate_idx is None:
            return

        cand_idx = self._send_candidate_idx
        window_end = cand_idx + self.send_confirm_threshold

        # 觀察窗還沒累積夠
        if current_idx < window_end:
            return

        # 觀察窗已滿，只在剛好到達時做一次判定
        if current_idx != window_end:
            return

        window = self.voted_statuses[cand_idx:window_end]

        # 計算最長連續 1
        max_one_run = 0
        current_one_run = 0
        for s in window:
            if s == 1:
                current_one_run += 1
                max_one_run = max(max_one_run, current_one_run)
            else:
                current_one_run = 0

        # 計算最長連續 0
        max_zero_run = 0
        current_zero_run = 0
        for s in window:
            if s == 0:
                current_zero_run += 1
                max_zero_run = max(max_zero_run, current_zero_run)
            else:
                current_zero_run = 0

        zero_ratio = sum(1 for s in window if s == 0) / len(window)
        print(f" zero_ratio={zero_ratio:.3f}, max_one_run={max_one_run}, max_zero_run={max_zero_run}")

        # 判定
        if max_one_run >= 50:
            print(f"    [SEND失敗] 出現連續 {max_one_run} f 的 1 (仍在手術中)")
            self._send_candidate_idx = None  # 重設，等下次再重新開始
        elif max_zero_run >= 150:
            # SEND 確認
            meta = self.frame_metadata[cand_idx]
            event = {
                'event_type': 'SEND',
                'video_name': meta['video_name'],
                'video_time': meta['video_time'],
                'real_time': meta['real_time'],
            }
            self.confirmed_events.append(event)
            self.latest_event = event
            self.current_confirmed_state = 0
            self._last_surgery_end_idx = cand_idx
            print(f"  > [確認] {meta['video_name'][:20]}... | {meta['video_time']} | SEND 紀錄成功")
            self._send_candidate_idx = None
        else:
            print(f"    [SEND失敗] 條件不足(zero_ratio={zero_ratio:.3f}, max_zero_run={max_zero_run})，仍在手術中")
            self._send_candidate_idx = None

    # ==================================================================
    #  Door 模式：增量式事件偵測
    #
    #  狀態流程：
    #  IDLE → ENT_CHECKING → ENT_ACTIVE → WAITING_SEND
    #       → SEND_CHECKING → SEND_ACTIVE → IDLE (冷卻後)
    # ==================================================================

    def _door_incremental_detect(self):
        """
        Door 模式的增量式事件偵測。
        對應 Door_analyze.py 的邏輯，改為逐幀處理。
        """
        idx = len(self.voted_statuses) - 1 # 最新的 voted index 
        v_status = self.voted_statuses[idx]
        meta = self.frame_metadata[idx] #時間資訊
        # 前一幀的 voted_status (沒有前一幀就當 0)
        prev_status = self.voted_statuses[idx - 1] if idx > 0 else 0

        # ============================
        # 狀態：IDLE — 等待 0→1 (ENT)
        # ============================
        if self._door_state == 'IDLE':
            if v_status == 1 and prev_status == 0:
                # 冷卻檢查
                if self._last_surgery_end_idx is not None:
                    if (idx - self._last_surgery_end_idx) < self.door_cooldown:
                        return
                self._door_candidate_idx = idx
                self._door_zero_run = 0
                self._door_state = 'ENT_CHECKING'
                print(f"\n[Door] 偵測到 0→1: ENT 候選 frame {idx} ({meta['video_time']})")

        # ============================
        # 狀態：ENT_CHECKING — 穩定期檢查 (300 幀)
        # ============================
        elif self._door_state == 'ENT_CHECKING':
            elapsed = idx - self._door_candidate_idx

            if v_status == 0:
                self._door_zero_run += 1
                if self._door_zero_run >= self.door_max_zero_tolerance:
                    # 連續 0 超過 50 幀 → ENT 失敗
                    print(f"  [ENT 失敗] 連續 0 達 {self._door_zero_run} 幀")
                    self._door_state = 'IDLE'
                    self._door_candidate_idx = None
                    return
            else:
                self._door_zero_run = 0

            if elapsed >= self.door_ent_check_window:
                # 通過穩定期 → 確認 ENT
                cand_meta = self.frame_metadata[self._door_candidate_idx]
                event = {
                    'event_type': 'ENT',
                    'video_name': cand_meta['video_name'],
                    'video_time': cand_meta['video_time'],
                    'real_time': cand_meta['real_time'],
                }
                self.confirmed_events.append(event)
                self.latest_event = event
                self.current_confirmed_state = 1
                self._door_ent_start_idx = self._door_candidate_idx
                self._door_last_one_idx = idx
                self._door_zero_run = 0
                self._door_state = 'ENT_ACTIVE'
                print(f"  > [確認] ENT @ {cand_meta['video_time']} | {cand_meta['video_name'][:20]}...")

        # ============================
        # 狀態：ENT_ACTIVE — 追蹤 ENT 活動的真正結束
        # ============================
        elif self._door_state == 'ENT_ACTIVE':
            if v_status == 1:
                self._door_last_one_idx = idx
                self._door_zero_run = 0
            else:
                self._door_zero_run += 1
                if self._door_zero_run >= self.door_min_zero_hold:
                    # 連續 0 達 300 幀 → ENT 活動真正結束
                    print(f"  [ENT 活動結束] 最後有人 @ frame {self._door_last_one_idx}")
                    self._door_zero_run = 0
                    self._door_state = 'WAITING_SEND'

        # ============================
        # 狀態：WAITING_SEND — 等待 ENT 後夠久，再找 0→1 (SEND)
        # ============================
        elif self._door_state == 'WAITING_SEND':
            if v_status == 1 and prev_status == 0:
                # 檢查間隔：從 ENT 開始到現在 >= 4500 幀 (15 分鐘)
                gap = idx - self._door_ent_start_idx
                if gap >= self.door_ent_to_send_min_gap:
                    self._door_candidate_idx = idx
                    self._door_zero_run = 0
                    self._door_state = 'SEND_CHECKING'
                    print(f"\n[Door] 偵測到 0→1: SEND 候選 frame {idx} ({meta['video_time']})")

        # ============================
        # 狀態：SEND_CHECKING — SEND 穩定期檢查 (300 幀)
        # ============================
        elif self._door_state == 'SEND_CHECKING':
            elapsed = idx - self._door_candidate_idx

            if v_status == 0:
                self._door_zero_run += 1
                if self._door_zero_run >= self.door_max_zero_tolerance:
                    # SEND 穩定期失敗 → 回去繼續等
                    print(f"  [SEND 失敗] 連續 0 達 {self._door_zero_run} 幀，繼續等")
                    self._door_state = 'WAITING_SEND'
                    self._door_candidate_idx = None
                    self._door_zero_run = 0
                    return
            else:
                self._door_zero_run = 0

            if elapsed >= self.door_send_check_window:
                # 通過穩定期 → 確認 SEND
                cand_meta = self.frame_metadata[self._door_candidate_idx]
                event = {
                    'event_type': 'SEND',
                    'video_name': cand_meta['video_name'],
                    'video_time': cand_meta['video_time'],
                    'real_time': cand_meta['real_time'],
                }
                self.confirmed_events.append(event)
                self.latest_event = event
                self._door_last_one_idx = idx
                self._door_zero_run = 0
                self._door_state = 'SEND_ACTIVE'
                print(f"  > [確認] SEND @ {cand_meta['video_time']} | {cand_meta['video_name'][:20]}...")

        # ============================
        # 狀態：SEND_ACTIVE — 追蹤 SEND 活動的真正結束
        # ============================
        elif self._door_state == 'SEND_ACTIVE':
            if v_status == 1:
                self._door_last_one_idx = idx
                self._door_zero_run = 0
            else:
                self._door_zero_run += 1
                if self._door_zero_run >= self.door_min_zero_hold:
                    # SEND 活動真正結束 → 回到 IDLE，進入冷卻
                    self._last_surgery_end_idx = self._door_last_one_idx
                    self.current_confirmed_state = 0
                    self._door_state = 'IDLE'
                    self._door_candidate_idx = None
                    self._door_ent_start_idx = None
                    self._door_zero_run = 0
                    print(f"  [SEND 活動結束] 最後有人 @ frame {self._door_last_one_idx}")
