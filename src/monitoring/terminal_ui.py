"""Terminal status helpers for the realtime parallel pipeline."""

import sys
import threading


_term_lock = threading.Lock()
_term_state = {"door": "等待影片...", "surgery": "等待模型載入..."}


def redraw_ui():
    sys.stdout.write(f"\r\033[2K  [Door 視角]    {_term_state['door']}\n")
    sys.stdout.write(f"\r\033[2K  [Surgery 視角] {_term_state['surgery']}\033[1A")
    sys.stdout.flush()


def update_ui(worker: str, text: str):
    with _term_lock:
        _term_state[worker] = text
        redraw_ui()


def log_event(text: str):
    with _term_lock:
        sys.stdout.write("\r\033[2K\n\r\033[2K\033[1A")
        sys.stdout.write(f"{text}\n\n\033[1A")
        redraw_ui()
