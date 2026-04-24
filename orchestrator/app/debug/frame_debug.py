from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np


DEBUG_DIR = Path("/data/debug")
DEBUG_DIR.mkdir(parents=True, exist_ok=True)


def save_input_frame(frame_bgr: np.ndarray) -> None:
    cv2.imwrite(str(DEBUG_DIR / "latest_input_frame.jpg"), frame_bgr)


def save_face_crop(frame_bgr: np.ndarray, bbox_xyxy: Sequence[int]) -> None:
    x1, y1, x2, y2 = map(int, bbox_xyxy)
    h, w = frame_bgr.shape[:2]

    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return

    crop = frame_bgr[y1:y2, x1:x2]
    cv2.imwrite(str(DEBUG_DIR / "latest_face_crop.jpg"), crop)


def save_annotated_frame(
    frame_bgr: np.ndarray,
    bbox_xyxy: Optional[Sequence[int]] = None,
    label: Optional[str] = None,
) -> None:
    vis = frame_bgr.copy()

    if bbox_xyxy is not None:
        x1, y1, x2, y2 = map(int, bbox_xyxy)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if label:
            cv2.putText(
                vis,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
    elif label:
        cv2.putText(
            vis,
            label,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(DEBUG_DIR / "latest_annotated_frame.jpg"), vis)