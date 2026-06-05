"""Time and CSV helpers for unified realtime events."""

import csv
import os
from datetime import datetime, timedelta


def hms_to_seconds(value: str) -> int | None:
    """Convert HH:MM:SS into seconds. Return None on malformed input."""
    try:
        h, m, s = (int(part) for part in value.split(":"))
        return h * 3600 + m * 60 + s
    except Exception:
        return None


def seconds_delta(ts_sec: int, target_sec: int) -> int:
    """Return ts-target seconds, corrected for HH:MM:SS values crossing midnight."""
    delta = ts_sec - target_sec
    if delta > 12 * 3600:
        delta -= 24 * 3600
    elif delta < -12 * 3600:
        delta += 24 * 3600
    return delta


def next_date_tag(date_tag: str) -> str:
    try:
        dt = datetime.strptime(date_tag, "%Y%m%d") + timedelta(days=1)
        return dt.strftime("%Y%m%d")
    except Exception:
        return date_tag


def event_sort_key(row: dict) -> tuple[int, int, int]:
    video_time_sec = hms_to_seconds(row.get("video_time", ""))
    video_name = row.get("video_name", "")

    try:
        base = os.path.splitext(os.path.basename(video_name))[0]
        parts = base.split("-")
        start_dt = datetime.strptime(f"{parts[1]}{parts[2]}", "%Y%m%d%H%M%S")
        if video_time_sec is not None:
            source_order = 0 if row.get("source") == "door" else 1
            return (int(start_dt.timestamp()) + video_time_sec, source_order, 0)
    except Exception:
        pass

    return (0, 0, 3)


def sort_csv(path: str, sort_key):
    if not os.path.exists(path):
        return

    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if not fieldnames:
        return

    rows.sort(key=sort_key)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sort_unified_csv(path: str):
    sort_csv(path, event_sort_key)
