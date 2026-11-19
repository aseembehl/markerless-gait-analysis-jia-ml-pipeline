"""Video reading. Requires the ``video`` extra (opencv-python)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class VideoInfo:
    path: Path
    fps: float
    n_frames: int
    width: int
    height: int


def _cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install the 'video' extra: pip install -e '.[video]'") from exc
    return cv2


def probe(path: str | Path) -> VideoInfo:
    cv2 = _cv2()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    try:
        return VideoInfo(
            path=Path(path),
            fps=float(cap.get(cv2.CAP_PROP_FPS)),
            n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()


def iter_frames(path: str | Path) -> Iterator[np.ndarray]:
    """Yield frames as RGB uint8 arrays of shape (H, W, 3)."""
    cv2 = _cv2()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()
