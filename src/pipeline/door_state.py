"""Door debounce and event matching helpers."""

from config import (
    DOOR_CLOSE_CONFIRM_FRAMES,
    DOOR_EVENT_MATCH_AFTER_FRAMES,
    DOOR_EVENT_MATCH_BEFORE_FRAMES,
    DOOR_OPEN_CONFIRM_FRAMES,
    STRIDE_SEC,
)
from pipeline.event_utils import hms_to_seconds, seconds_delta


def door_detect_step(detector, frame_bgr, state):
    raw_open = detector.detect(frame_bgr)
    if not state["door_open"]:
        if raw_open:
            state["open_confirm_count"] += 1
            state["close_confirm_count"] = 0
            if state["open_confirm_count"] >= DOOR_OPEN_CONFIRM_FRAMES:
                state["door_open"] = True
                state["open_confirm_count"] = 0
        else:
            state["open_confirm_count"] = 0
    else:
        if not raw_open:
            state["close_confirm_count"] += 1
            if state["close_confirm_count"] >= DOOR_CLOSE_CONFIRM_FRAMES:
                state["door_open"] = False
                state["close_confirm_count"] = 0
        else:
            state["close_confirm_count"] = 0
    return raw_open


def door_open_for_event(event_type: str, real_time_str: str, door_open_window, current_open: bool) -> bool:
    """
    Directional Door OPEN lookup:
        ENT  checks the minute before the Surgery event.
        SEND checks the minute after the Surgery event.
    """
    target_sec = hms_to_seconds(real_time_str)
    if target_sec is None:
        return current_open
    before_sec = DOOR_EVENT_MATCH_BEFORE_FRAMES * STRIDE_SEC
    after_sec = DOOR_EVENT_MATCH_AFTER_FRAMES * STRIDE_SEC
    event_type = (event_type or "").upper()

    for ts_str, is_open in door_open_window:
        if not is_open:
            continue
        ts_sec = hms_to_seconds(ts_str)
        if ts_sec is None:
            continue
        delta = seconds_delta(ts_sec, target_sec)
        if event_type == "ENT" and -before_sec <= delta <= 0:
            return True
        if event_type == "SEND" and 0 <= delta <= after_sec:
            return True
    return False


def door_timeline_ready(event_type: str, real_time_str: str, door_open_window, fallback_buffer_sec: int) -> bool:
    if not door_open_window:
        return False
    target_sec = hms_to_seconds(real_time_str)
    latest_door_sec = hms_to_seconds(door_open_window[-1][0])
    if target_sec is None or latest_door_sec is None:
        return True

    event_type = (event_type or "").upper()
    if event_type == "ENT":
        wait_sec = 0
    elif event_type == "SEND":
        wait_sec = DOOR_EVENT_MATCH_AFTER_FRAMES * STRIDE_SEC
    else:
        wait_sec = fallback_buffer_sec
    return seconds_delta(latest_door_sec, target_sec) >= wait_sec
