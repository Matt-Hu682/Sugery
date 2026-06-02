#!/usr/bin/env python3
"""
Select Door Stage1 ROI rectangles and print config.py-ready normalized values.

Usage:
  python scripts/select_door_roi.py
  python scripts/select_door_roi.py --image templates/door_closed_A8.jpg
  python scripts/select_door_roi.py --image path/to/full_frame.jpg --crop 400,0,640,260

Controls:
  Drag mouse: draw ROI
  1-9: set weight for the next ROI
  u: undo last ROI
  r: reset all ROIs
  Enter / q / Esc: finish
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = PROJECT_ROOT / "templates" / "door_closed_A8.jpg"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "door_roi_selection.jpg"


def parse_crop(value: str | None):
    if not value:
        return None
    parts = [int(p.strip()) for p in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("crop must be x1,y1,x2,y2")
    x1, y1, x2, y2 = parts
    if x2 <= x1 or y2 <= y1:
        raise argparse.ArgumentTypeError("crop must satisfy x2>x1 and y2>y1")
    return x1, y1, x2, y2


def draw_overlay(image, rois, draft=None, next_weight=1.0):
    vis = image.copy()
    colors = [(0, 0, 255), (255, 120, 0), (0, 180, 255), (0, 220, 0), (255, 0, 180)]
    for i, (x1, y1, x2, y2, weight) in enumerate(rois, 1):
        color = colors[(i - 1) % len(colors)]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            vis,
            f"ROI {i} w={weight:g} ({x1},{y1})-({x2},{y2})",
            (x1 + 4, max(18, y1 + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    if draft is not None:
        x1, y1, x2, y2 = draft
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 255), 1)
    cv2.putText(
        vis,
        f"drag=ROI | weight next={next_weight:g} | u=undo r=reset Enter/q=finish",
        (8, image.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return vis


def print_config(rois, width, height):
    print("\nconfig.py ROI values:")
    print('    "A8": [')
    for x1, y1, x2, y2, weight in rois:
        print(
            f"        ({x1 / width:.3f}, {y1 / height:.3f}, "
            f"{x2 / width:.3f}, {y2 / height:.3f}, {weight:.1f}),"
            f"  # x={x1}~{x2}, y={y1}~{y2}"
        )
    print("    ],")


def main():
    parser = argparse.ArgumentParser(description="Interactively select Door Stage1 ROIs.")
    parser.add_argument("--image", default=str(DEFAULT_IMAGE), help="Image to annotate. Default: templates/door_closed_A8.jpg")
    parser.add_argument("--crop", type=parse_crop, default=None, help="Optional crop for full frames: x1,y1,x2,y2")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Overlay image output path")
    args = parser.parse_args()

    image_path = Path(args.image)
    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"Cannot read image: {image_path}")

    if args.crop:
        x1, y1, x2, y2 = args.crop
        image = image[y1:y2, x1:x2]

    rois = []
    drawing = {"active": False, "start": None, "current": None}
    next_weight = {"value": 1.0}
    window = "select_door_roi"

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing["active"] = True
            drawing["start"] = (x, y)
            drawing["current"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drawing["active"]:
            drawing["current"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and drawing["active"]:
            sx, sy = drawing["start"]
            x1, x2 = sorted((sx, x))
            y1, y2 = sorted((sy, y))
            if x2 - x1 >= 3 and y2 - y1 >= 3:
                rois.append((x1, y1, x2, y2, next_weight["value"]))
            drawing["active"] = False
            drawing["start"] = None
            drawing["current"] = None

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, mouse_callback)

    while True:
        draft = None
        if drawing["active"] and drawing["start"] and drawing["current"]:
            sx, sy = drawing["start"]
            cx, cy = drawing["current"]
            draft = (*sorted((sx, cx)), *sorted((sy, cy)))
            draft = (min(sx, cx), min(sy, cy), max(sx, cx), max(sy, cy))
        cv2.imshow(window, draw_overlay(image, rois, draft=draft, next_weight=next_weight["value"]))
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 27, ord("q")):
            break
        if key == ord("u") and rois:
            rois.pop()
        elif key == ord("r"):
            rois.clear()
        elif ord("1") <= key <= ord("9"):
            next_weight["value"] = float(chr(key))

    cv2.destroyAllWindows()

    if not rois:
        print("No ROI selected.")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), draw_overlay(image, rois, next_weight=next_weight["value"]))
    print_config(rois, image.shape[1], image.shape[0])
    print(f"\nOverlay saved: {output_path.resolve()}")


if __name__ == "__main__":
    main()
