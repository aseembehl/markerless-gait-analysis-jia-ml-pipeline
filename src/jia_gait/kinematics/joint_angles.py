"""Stage 3: frame-by-frame sagittal joint angles from 2D landmarks.

Port of ``pointsToAngles.ipynb``. Each video contains walks in both
directions, so angles are computed for both body sides on every frame:

* "Angle 1" / right-to-left (``rtl``) uses the LEFT-side landmarks
  (the left side faces the camera when walking right to left);
* "Angle 2" / left-to-right (``ltr``) uses the RIGHT-side landmarks.

No filtering is applied to landmarks or angles (as in the original code).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..io.csv_formats import FEATURE_COLUMNS
from ..landmarks.skeleton import landmark_index

_DOWN = np.array([0.0, 1.0])  # +1 px in image y (pointing down)


def angle_between(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """Vectorised port of the original ``angle_between(p1, p2, p3)``.

    Angle at ``p2`` going from ``p1`` to ``p3`` in [0, 360), with directions
    measured as ``atan2(dx, dy)`` in image coordinates. Inputs are ``(..., 2)``.
    """
    d1 = (360.0 + np.degrees(np.arctan2(p1[..., 0] - p2[..., 0], p1[..., 1] - p2[..., 1]))) % 360.0
    d2 = (360.0 + np.degrees(np.arctan2(p3[..., 0] - p2[..., 0], p3[..., 1] - p2[..., 1]))) % 360.0
    return np.where(d1 <= d2, d2 - d1, 360.0 - (d1 - d2))


@dataclass
class JointAngles:
    """Per-frame angles in degrees, each ``(T,)``."""

    hip_ltr: np.ndarray
    hip_rtl: np.ndarray
    knee_ltr: np.ndarray
    knee_rtl: np.ndarray
    ankle_ltr: np.ndarray
    ankle_rtl: np.ndarray

    def features(self) -> np.ndarray:
        """``(T, 6)`` matrix in the original CSV column order (``FEATURE_COLUMNS``)."""
        return np.stack(
            [
                self.hip_ltr,
                self.hip_rtl,
                self.knee_ltr,
                self.knee_rtl,
                self.ankle_ltr,
                self.ankle_rtl,
            ],
            axis=1,
        )

    @classmethod
    def from_features(cls, features: np.ndarray) -> JointAngles:
        f = np.asarray(features, dtype=float)
        if f.ndim != 2 or f.shape[1] != len(FEATURE_COLUMNS):
            raise ValueError(f"Expected (T, {len(FEATURE_COLUMNS)}) features, got {f.shape}")
        return cls(*(f[:, i] for i in range(f.shape[1])))

    def direction(self, direction: str) -> dict[str, np.ndarray]:
        """Hip/knee/ankle series for ``"ltr"`` or ``"rtl"``."""
        if direction not in ("ltr", "rtl"):
            raise ValueError(f"direction must be 'ltr' or 'rtl', got {direction!r}")
        return {j: getattr(self, f"{j}_{direction}") for j in ("hip", "knee", "ankle")}


def compute_joint_angles(landmarks: np.ndarray) -> JointAngles:
    """Compute hip, knee and ankle angles for both walking directions.

    Args:
        landmarks: ``(T, 12, 2)`` or ``(T, 12, 3)`` image-coordinate landmarks
            ordered as ``skeleton.LANDMARKS``; a confidence channel is ignored.
    """
    xy = np.asarray(landmarks, dtype=float)[..., :2]

    def P(joint: str, side: str) -> np.ndarray:
        return xy[:, landmark_index(joint, side)]

    out = {}
    for direction, side in (("rtl", "left"), ("ltr", "right")):
        shoulder, hip, knee = P("shoulder", side), P("hip", side), P("knee", side)
        ankle, toe, heel = P("ankle", side), P("toe", side), P("heel", side)
        hip_knee = angle_between(shoulder, hip, knee)
        knee_ang = angle_between(hip, knee, ankle)
        foot = angle_between(heel, toe, toe + _DOWN)
        if direction == "rtl":
            # Original: ankle uses (knee, ankle, ankle + 1 px down).
            shank = angle_between(knee, ankle, ankle + _DOWN)
            out["hip_rtl"] = 180.0 - hip_knee
            out["knee_rtl"] = knee_ang - 180.0
            out["ankle_rtl"] = (foot - 90.0) - (shank - 90.0) - 90.0
        else:
            # Original has ``ankleVertical = ankle = (x, y + 1)``: the ankle used
            # for the shank direction is shifted 1 px down. Reproduced verbatim
            # to keep published numbers; its effect is a fraction of a degree.
            ankle_shifted = ankle + _DOWN
            shank = angle_between(knee, ankle_shifted, ankle_shifted)
            out["hip_ltr"] = hip_knee - 180.0
            out["knee_ltr"] = 180.0 - knee_ang
            out["ankle_ltr"] = -foot + (shank - 180.0) + 90.0
    return JointAngles(**out)
