"""Stage 2 interface: anatomical landmarks on the cropped person image."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class LandmarkDetector(ABC):
    @abstractmethod
    def predict(self, crop: np.ndarray) -> np.ndarray | None:
        """Return ``(12, 3)`` (x, y, confidence) in *crop* pixel coordinates,
        ordered as ``skeleton.LANDMARKS``, or None if no person was found."""
