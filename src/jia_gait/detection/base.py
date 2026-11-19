"""Stage 1 interface: one person bounding box per frame."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class BBox:
    """Axis-aligned box in pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 1.0

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def crop(self, frame: np.ndarray) -> np.ndarray:
        """Crop like ``PIL.Image.crop([int(b) for b in box])`` (coordinates truncated)."""
        x1, y1, x2, y2 = (int(v) for v in (self.x1, self.y1, self.x2, self.y2))
        return frame[max(y1, 0) : y2, max(x1, 0) : x2]


class PersonDetector(ABC):
    def reset(self) -> None:
        """Called at the start of each video (for detectors that keep state)."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> BBox | None:
        """Return the box of the walking child in an RGB frame, or None to skip the frame."""
