"""Room-video clipping for confirmed surgery events."""

import datetime as _dt
import os

import cv2

from pipeline.video_outputs import collect_videos, get_dataset_name


class RoomEventClipper:
    def __init__(self, log_event, pre_post_sec: int = 90):
        self.log_event = log_event
        self.pre_post_sec = pre_post_sec
        self.room_videos_by_dataset = {}

    def _get_video_abs_ts(self, path):
        try:
            parts = os.path.basename(path).split("-")
            dt = _dt.datetime.strptime(f"{parts[1]}{parts[2]}", "%Y%m%d%H%M%S")
            return int(dt.timestamp())
        except Exception:
            return None

    def _index_room_videos(self, dataset_name: str | None = None):
        by_dataset = {}
        for vpath in sorted(collect_videos("Room")):
            if dataset_name is not None and get_dataset_name(vpath) != dataset_name:
                continue
            vts = self._get_video_abs_ts(vpath)
            if vts is None:
                continue
            cap = cv2.VideoCapture(vpath)
            fps = cap.get(cv2.CAP_PROP_FPS) or 5.0
            dur = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps
            cap.release()
            dataset_key = get_dataset_name(vpath)
            by_dataset.setdefault(dataset_key, []).append({
                "path": vpath,
                "ts": vts,
                "fps": fps,
                "dur": dur,
            })
        for videos in by_dataset.values():
            videos.sort(key=lambda x: x["ts"])
        return by_dataset

    def _ensure_dataset_index(self, dataset_name: str):
        if dataset_name in self.room_videos_by_dataset:
            return
        self.room_videos_by_dataset.update(self._index_room_videos(dataset_name))

    def clip_event(self, row: dict, video_output_dir: str, dataset_name: str):
        self._ensure_dataset_index(dataset_name)
        dataset_room_videos = self.room_videos_by_dataset.get(dataset_name, [])
        if not dataset_room_videos:
            self.log_event(f"  [剪輯略過] dataset={dataset_name} 找不到 Room 影片")
            return

        surgery_date_tag = row.get("Surgery_Date", "unknown")
        sno = str(row.get("Surgery_No", "?")).replace("/", "").replace("\\", "")
        etype = row.get("Type", "?")
        rt_raw = row.get("Real_Time", "000000")
        rt_str = rt_raw.replace(":", "")
        dst_name = f"{surgery_date_tag}_{etype}_{sno}_{rt_str}.mp4"
        dst_path = os.path.join(video_output_dir, dst_name)
        if os.path.exists(dst_path):
            return
        try:
            rh, rm, rs = map(int, rt_raw.split(":"))
        except Exception:
            return

        event_abs_ts = None
        for video in dataset_room_videos:
            vts = video["ts"]
            try:
                base_dt = _dt.datetime.fromtimestamp(vts)
                cand_dt = base_dt.replace(hour=rh, minute=rm, second=rs)
                if cand_dt < base_dt:
                    cand_dt += _dt.timedelta(days=1)
                cand = int(cand_dt.timestamp())
                if vts <= cand <= vts + video["dur"]:
                    event_abs_ts = cand
                    break
            except Exception:
                continue

        if event_abs_ts is None and dataset_room_videos:
            try:
                fb_vts = dataset_room_videos[0]["ts"]
                fb_base = _dt.datetime.fromtimestamp(fb_vts)
                fb_cand = fb_base.replace(hour=rh, minute=rm, second=rs)
                if fb_cand < fb_base:
                    fb_cand += _dt.timedelta(days=1)
                event_abs_ts = int(fb_cand.timestamp())
            except Exception:
                return
        if event_abs_ts is None:
            return

        t_start = event_abs_ts - self.pre_post_sec
        t_end = event_abs_ts + self.pre_post_sec
        try:
            writer = None
            for video in dataset_room_videos:
                cap = cv2.VideoCapture(video["path"])
                fps = video["fps"]
                dur = video["dur"]
                vs = video["ts"]
                ve = vs + dur
                if ve < t_start or vs > t_end:
                    cap.release()
                    continue
                if writer is None:
                    os.makedirs(video_output_dir, exist_ok=True)
                    writer = cv2.VideoWriter(
                        dst_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (
                            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        ),
                    )
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(max(0.0, t_start - vs) * fps))
                limit = int(min(dur, t_end - vs) * fps)
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret or cap.get(cv2.CAP_PROP_POS_FRAMES) > limit:
                        break
                    writer.write(frame)
                cap.release()
            if writer:
                writer.release()
                self.log_event(f"  [剪輯完成] {dst_name}")
        except Exception as exc:
            self.log_event(f"  [剪輯失敗] {dst_name} ({exc})")
