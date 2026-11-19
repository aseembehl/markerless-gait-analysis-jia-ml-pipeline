"""Remove short landmark glitches from joint-angle series (Hampel filter).

ViTPose occasionally misplaces a joint for one or two frames (e.g. confusing
left and right foot). The angle then jumps by tens of degrees - sometimes to
physically impossible values - and back. Left in, a single such frame
distorts that cycle and the mean +- SD plot of the whole recording.

A sample is treated as a spike when it differs from the running median of
its neighbourhood by more than ``threshold_deg`` *and* by more than
``n_sigma`` robust standard deviations (1.4826 x the running median absolute
deviation); it is replaced by the running median.

The 20 deg default removes such glitches while leaving genuine fast
movement alone; a lower threshold can clip sharp ankle peaks after toe-off.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

from .joint_angles import JointAngles


def half_window_frames(half_window_s: float, fps: float) -> int:
    return max(1, round(half_window_s * fps))


def despike(
    values: np.ndarray,
    segment: np.ndarray | None = None,
    half_window: int = 3,
    threshold_deg: float = 20.0,
    n_sigma: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Hampel-filter each column of ``values`` (T, F), separately per segment.

    Returns ``(filtered, replaced)`` where ``replaced`` is a ``(T, F)`` bool mask.
    """
    x = np.asarray(values, dtype=float)
    one_d = x.ndim == 1
    if one_d:
        x = x[:, None]
    seg = np.zeros(len(x), dtype=int) if segment is None else np.asarray(segment)
    out = x.copy()
    replaced = np.zeros(x.shape, dtype=bool)
    size = 2 * half_window + 1
    for s in np.unique(seg):
        rows = np.flatnonzero(seg == s)
        if len(rows) < size:
            continue  # too short to tell a spike from movement
        block = x[rows]
        med = median_filter(block, size=(size, 1), mode="nearest")
        dev = np.abs(block - med)
        sigma = 1.4826 * median_filter(dev, size=(size, 1), mode="nearest")
        bad = dev > np.maximum(n_sigma * sigma, threshold_deg)
        block = np.where(bad, med, block)
        out[rows] = block
        replaced[rows] = bad
    if one_d:
        return out[:, 0], replaced[:, 0]
    return out, replaced


def despike_angles(
    angles: JointAngles,
    segment: np.ndarray | None,
    fps: float,
    half_window_s: float = 0.06,
    threshold_deg: float = 20.0,
    n_sigma: float = 3.0,
) -> tuple[JointAngles, np.ndarray]:
    """``despike`` on all six angle series; returns the angles and the ``(T, 6)`` mask."""
    f, mask = despike(
        angles.features(), segment, half_window_frames(half_window_s, fps), threshold_deg, n_sigma
    )
    return JointAngles.from_features(f), mask
