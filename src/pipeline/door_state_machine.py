"""Door ENT/SEND state machine for RealtimePipeline."""


class DoorStateMachineMixin:
    def _door_direct_confirm(self, event_type="ENT"):
        """Door 模式收到絕對確認時，繞過投票與穩定期，直接切換狀態。"""
        idx = len(self.raw_statuses) - 1
        meta = self.frame_metadata[idx]
        
        if self._last_surgery_end_idx is not None:
            if (idx - self._last_surgery_end_idx) < self.door_cooldown:
                return # 冷卻中
                
        if event_type == "ENT":
            event = {
                'event_type': 'ENT',
                'video_name': meta['video_name'],
                'video_time': meta['video_time'],
                'real_time': meta['real_time'],
            }
            self.confirmed_events.append(event)
            self.latest_event = event
            self.current_confirmed_state = 1
            self._door_ent_start_idx = idx
            self._door_state = 'WAITING_SEND'
            print(f"\n  > [光速確認] ENT (推入) @ {meta['video_time']} | {meta['video_name'][:20]}...")
            
        elif event_type == "SEND":
            if self._door_ent_start_idx is None:
                self._door_ent_start_idx = 0
            # 既然是模型直接看到的 SEND，不嚴格卡間隔限制，只要記錄即可
            event = {
                'event_type': 'SEND',
                'video_name': meta['video_name'],
                'video_time': meta['video_time'],
                'real_time': meta['real_time'],
            }
            self.confirmed_events.append(event)
            self.latest_event = event
            self.current_confirmed_state = 0
            self._last_surgery_end_idx = idx
            self._door_state = 'IDLE' # SEND 後回歸 IDLE
            self._door_ent_start_idx = None
            print(f"\n  > [光速確認] SEND (推出) @ {meta['video_time']} | {meta['video_name'][:20]}...")
    def _door_incremental_detect(self):
        """
        Door 模式的事件偵測。
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

