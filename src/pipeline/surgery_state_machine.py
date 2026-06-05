"""Surgery ENT/SEND state machine for RealtimePipeline."""

from config import SEND_MAX_CONSEC_ONE_FRAMES, SEND_MIN_CONSEC_ZERO_FRAMES


class SurgeryStateMachineMixin:
    def force_close_pending_send(self):
        """
        在 flush() 之後呼叫。
        只處理 dataset 結尾 SEND 觀察窗未滿的情境：
            - 若 _send_candidate_idx 存在（SEND 候選已啟動，只是窗口未滿），
            用候選起始時間補發 SEND（那才是手術真正結束的時刻）。
            - 若沒有 SEND 候選，即使仍在手術中，也不硬補最後一幀為 SEND。
        只在 Surgery 模式下有效。
        """
        if self.task_type != "Surgery":
            return
        if self.current_confirmed_state != 1:
            return  # 沒有進行中的手術，不需要強制 SEND
        if not self.frame_metadata:
            return  # 沒有任何幀資料
        if self._send_candidate_idx is None:
            return  # 沒有觀察到 SEND 候選，不把資料集最後一幀硬當成 SEND

        meta = self.frame_metadata[self._send_candidate_idx]
        reason = "SEND 候選起始時間（觀察窗未滿，資料集截止）"

        event = {
            'event_type': 'SEND',
            'video_name': meta['video_name'],
            'video_time': meta['video_time'],
            'real_time':  meta['real_time'],
            'forced':     True,  # 標記為強制補發
        }
        self.confirmed_events.append(event)
        self.latest_event = event
        self.current_confirmed_state = 0
        self._send_candidate_idx = None
        print(
            f"\n  > [強制SEND] 補發 @ {meta['video_time']} "
            f"（{reason}） | {meta['video_name'][:20]}..."
        )
    def _surgery_incremental_detect(self):
        """
        Surgery 模式的事件偵測。
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

        cand_idx = self._ent_candidate_idx #0->1的候選
        elapsed = current_idx - cand_idx # 時間差

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

        # 以下是遇到遮擋
        v_status = self.voted_statuses[current_idx] # 投票的狀態

        if v_status == 1:
            # 狀態正常(1)，繼續觀察
            self._ent_gap_start = None
            return

        # 遇到 0 → 遮擋處理
        if self._ent_gap_start is None:
            self._ent_gap_start = current_idx

        # 遮擋的長度
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
        window_end = cand_idx + self.send_confirm_threshold # 1->0的候選

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
        if max_one_run >= SEND_MAX_CONSEC_ONE_FRAMES: # 連續1大於閾值 frame，表示還在手術
            print(f"    [SEND失敗] 出現連續 {max_one_run} f 的 1 (仍在手術中)")
            self._send_candidate_idx = None  # 重設，等下次再重新開始
        elif max_zero_run >= SEND_MIN_CONSEC_ZERO_FRAMES: # 連續0 大於閾値 frame，
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
    #  Door 模式：增事件偵測
    #
    #  狀態流程：
    #  IDLE → ENT_CHECKING → ENT_ACTIVE → WAITING_SEND
    #       → SEND_CHECKING → SEND_ACTIVE → IDLE (冷卻後)
    # ==================================================================

