import json
import os
import time
from datetime import datetime
from pathlib import Path

import cv2

from config import DOOR_STAGE1_ACTIVE_ROIS


class LiveMonitorWriter:
    """Write latest frames and status for the FastAPI live monitor."""

    def __init__(self, output_base_dir: str, update_every_frames: int = 5):
        run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
        self.live_dir = Path(output_base_dir) / "live" / run_id
        self.live_dir.mkdir(parents=True, exist_ok=True)
        self.update_every_frames = max(1, int(update_every_frames))
        self._last_frame_idx = -1

    def update(self, *, frame_idx: int, door_frame, door_analysis, room_frame, status: dict):
        if frame_idx % self.update_every_frames != 0 and frame_idx == self._last_frame_idx:
            return
        if frame_idx % self.update_every_frames != 0:
            return

        self._last_frame_idx = frame_idx
        status_payload = dict(status)
        status_payload["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        status_payload["live_dir"] = str(self.live_dir)

        self._write_image("door_frame.jpg", door_frame)
        self._write_image("door_crop.jpg", door_analysis)
        self._write_image("door_crop_roi.jpg", self._draw_door_rois(door_analysis))
        self._write_image("room_frame.jpg", room_frame)
        self._write_json("status.json", status_payload)

    def _write_image(self, filename: str, image):
        if image is None:
            return
        path = self.live_dir / filename
        tmp_path = self.live_dir / f".{filename}.tmp.jpg"
        cv2.imwrite(str(tmp_path), image)
        os.replace(tmp_path, path)

    def _write_json(self, filename: str, payload: dict):
        path = self.live_dir / filename
        tmp_path = self.live_dir / f".{filename}.tmp"
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_path, path)

    def _draw_door_rois(self, image):
        if image is None:
            return None
        overlay = image.copy()
        h, w = overlay.shape[:2]
        colors = [
            (0, 0, 255),
            (255, 140, 0),
            (0, 200, 255),
            (0, 220, 0),
            (255, 0, 180),
        ]
        for idx, (x1r, y1r, x2r, y2r, weight) in enumerate(DOOR_STAGE1_ACTIVE_ROIS, 1):
            x1 = max(0, min(w - 1, int(w * x1r)))
            y1 = max(0, min(h - 1, int(h * y1r)))
            x2 = max(x1 + 1, min(w, int(w * x2r)))
            y2 = max(y1 + 1, min(h, int(h * y2r)))
            color = colors[(idx - 1) % len(colors)]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                overlay,
                f"ROI {idx} w={float(weight):g}",
                (x1 + 4, max(16, y1 + 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
        return overlay
