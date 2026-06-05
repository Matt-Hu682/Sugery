"""Voting helpers for RealtimePipeline."""


class VotingMixin:
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

            # 新產生一個 voted_status，嘗試事件偵測
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

    # surrgery測試用

