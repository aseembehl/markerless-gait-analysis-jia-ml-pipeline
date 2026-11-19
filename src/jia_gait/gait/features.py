"""Feature sets for the heel-on classifier (computed at the model frame rate).

``angles``  - the 6 per-frame joint angles.
``context`` - the 6 angles plus their change to neighbouring frames at the
              given offsets, giving the classifier temporal context. Offsets
              are clipped at walking-pass (segment) boundaries.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

DEFAULT_CONTEXT_OFFSETS = (-8, -4, -2, -1, 1, 2, 4, 8)


def angle_features(values: np.ndarray, segment: np.ndarray | None = None) -> np.ndarray:
    return np.asarray(values, dtype=float)


def context_features(
    values: np.ndarray,
    segment: np.ndarray | None = None,
    offsets: Sequence[int] = DEFAULT_CONTEXT_OFFSETS,
) -> np.ndarray:
    a = np.asarray(values, dtype=float)
    n = len(a)
    seg = np.zeros(n, dtype=int) if segment is None else np.asarray(segment)
    # first/last index of each row's segment, for clipping
    first = np.zeros(n, dtype=int)
    last = np.zeros(n, dtype=int)
    for s in np.unique(seg):
        idx = np.flatnonzero(seg == s)
        first[idx], last[idx] = idx[0], idx[-1]
    rows = np.arange(n)
    cols = [a]
    for o in offsets:
        j = np.clip(rows + o, first, last)
        cols.append(a[j] - a)
    return np.concatenate(cols, axis=1)


def make_feature_fn(name: str, offsets: Sequence[int] | None = None):
    if name == "angles":
        return angle_features
    if name == "context":
        offs = tuple(offsets) if offsets else DEFAULT_CONTEXT_OFFSETS
        return lambda v, s=None: context_features(v, s, offs)
    raise ValueError(f"Unknown feature set {name!r}")
