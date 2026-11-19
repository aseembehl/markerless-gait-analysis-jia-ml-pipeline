"""Synthetic data generators for tests."""

from __future__ import annotations

import numpy as np

from jia_gait.io.csv_formats import GaitEvents


def random_landmarks(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 1000, size=(n, 12, 2))


def periodic_events(n=600, period=40, seed=0, turn_at=None):
    """Heel-ons every ``period`` (+-2) frames: L->R before ``turn_at``, R->L after."""
    rng = np.random.default_rng(seed)
    turn_at = n // 2 if turn_at is None else turn_at
    frames, f = [], 10
    while f < n - 5:
        frames.append(f)
        f += period + int(rng.integers(-2, 3))
    frames = np.array(frames)
    ltr = frames[frames < turn_at]
    rtl = frames[frames >= turn_at + period // 2]
    toe = int(0.6 * period)
    return GaitEvents(
        heel_on_ltr=ltr,
        heel_on_rtl=rtl,
        toe_off_ltr=(ltr + toe)[ltr + toe < n],
        toe_off_rtl=(rtl + toe)[rtl + toe < n],
    )


def angles_with_events(events: GaitEvents, n, seed=0, noise=1.0):
    """6 angle features whose shape is phase-locked to the heel-ons."""
    rng = np.random.default_rng(seed)
    feats = rng.normal(0, noise, size=(n, 6))
    for col_offset, hs in ((0, events.heel_on_ltr), (1, events.heel_on_rtl)):
        hs = np.asarray(hs)
        for a, b in zip(hs[:-1], hs[1:]):
            phase = np.linspace(0, 2 * np.pi, b - a, endpoint=False)
            feats[a:b, 0 + col_offset] += 10 + 22 * np.cos(phase)  # hip
            feats[a:b, 2 + col_offset] += 30 - 28 * np.cos(phase + 0.9)  # knee
            feats[a:b, 4 + col_offset] += 8 * np.sin(phase)  # ankle
    return feats
