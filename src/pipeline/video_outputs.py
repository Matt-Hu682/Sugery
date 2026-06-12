"""Video discovery and output path helpers for the parallel pipeline."""

import os

from config import (
    CAMERA_SETTING,
    CSV_OUTPUT,
    OR_SETTING,
    ROOM,
    TARGET_DATASETS,
    TEST_VIDEO_BASE,
    VIDEO_DIR,
    VIDEO_DIRS,
)


def _current_video_dirs() -> list[str]:
    if TARGET_DATASETS is None and os.path.isdir(TEST_VIDEO_BASE):
        return sorted(
            os.path.join(TEST_VIDEO_BASE, name)
            for name in os.listdir(TEST_VIDEO_BASE)
            if os.path.isdir(os.path.join(TEST_VIDEO_BASE, name))
        )
    return list(VIDEO_DIRS if VIDEO_DIRS else [VIDEO_DIR])


def collect_videos(cam_type: str) -> list[str]:
    cams = [cam for cam in OR_SETTING[ROOM] if CAMERA_SETTING.get(cam) == cam_type]
    dirs = [os.path.abspath(p) for p in _current_video_dirs()]
    videos = []
    for directory in dirs:
        if not os.path.isdir(directory):
            continue
        for fname in sorted(os.listdir(directory)):
            if fname.lower().endswith((".mp4", ".avi")) and any(cam in fname for cam in cams):
                videos.append(os.path.join(directory, fname))
    return sorted(videos)


def get_dataset_name(video_path: str) -> str:
    return os.path.basename(os.path.dirname(video_path))


def get_surgery_date(video_path: str) -> str:
    try:
        return os.path.basename(video_path).split("-")[1]
    except Exception:
        return get_dataset_name(video_path)


def prepare_surgery_outputs(run_date: str, room_video_path: str):
    dataset_name = get_dataset_name(room_video_path)

    raw_csv_path = os.path.join(
        os.path.dirname(CSV_OUTPUT),
        dataset_name,
        f"surgery_report_Surgery_{dataset_name}_{run_date}_parallel.csv",
    )
    os.makedirs(os.path.dirname(raw_csv_path), exist_ok=True)

    report_dir = os.path.join(os.path.dirname(raw_csv_path), "result", ROOM)
    os.makedirs(report_dir, exist_ok=True)
    pair_report_path = os.path.join(
        report_dir,
        f"Realtime_Events_Surgery_{dataset_name}_{run_date}.csv",
    )

    unified_events_dir = os.path.join(os.path.dirname(raw_csv_path), "all_events")
    os.makedirs(unified_events_dir, exist_ok=True)
    unified_events_path = os.path.join(
        unified_events_dir,
        f"Unified_Events_{dataset_name}_{run_date}.csv",
    )

    return raw_csv_path, pair_report_path, unified_events_path
