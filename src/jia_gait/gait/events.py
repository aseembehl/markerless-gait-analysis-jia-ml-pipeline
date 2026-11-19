"""Stage 4a: heel-on event detection with an XGBoost frame classifier.

Port of ``heal_on_event_prediction.ipynb``. Each frame is classified from its
six joint angles (``JointAngles.features()``) into

    0 = heel-on, walking left to right
    1 = heel-on, walking right to left
    2 = no event

and the per-frame predictions are turned into discrete events by
majority smoothing, taking the midpoint of each run, and discarding events
whose spacing is far from the median cycle length. Heel strikes cut off by
the start or end of a walking pass are recovered separately (``edge_min_run``,
not in the original notebook).

Toe-off events are not predicted (only expert-labelled toe-offs exist).

The model file is an XGBoost JSON model; it is evaluated with numpy
(``xgb_trees.py``), so XGBoost itself is not needed at inference time.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..io.csv_formats import GaitEvents
from ..sequence import FrameSequence, resample
from .features import make_feature_fn

HEEL_ON_LTR, HEEL_ON_RTL, NO_EVENT = 0, 1, 2


@dataclass
class PostprocessConfig:
    threshold: float = 0.5
    smooth_window: int = 5
    min_run: int = 4
    outlier_sd: float = 2.0
    # Heel strikes at the very start or end of a walking pass (tracking starts or
    # stops mid-run): accept a cut-off run of at least this many frames above the
    # threshold. 0 disables (original notebook behaviour).
    edge_min_run: int = 2


def _mode_smooth(y: np.ndarray, window: int) -> np.ndarray:
    half = window // 2
    out = y.astype(float).copy()
    for i in range(half, len(y) - half):
        out[i] = np.argmax(np.bincount(y[i - half : i + half + 1]))
    return out


def _run_midpoints(y: np.ndarray, cls: int, min_run: int) -> list[int]:
    """Midpoints of runs of ``cls`` with length >= ``min_run``.

    As in the original, a run only counts if it ends (the first index of a
    different class) at or before index ``n - 2``.
    """
    n = len(y)
    mids, i = [], 0
    while i < n:
        if y[i] != cls:
            i += 1
            continue
        start = i
        end = start + 1
        while end < n and y[end] == cls:
            end += 1
        if end <= n - 2 and end - start > min_run - 1:
            mids.append(int((start + end) / 2))
        i = end
    return mids


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """``(start, end)`` of the True runs of ``mask``, end exclusive."""
    d = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def _edge_events(
    proba: np.ndarray, passes: list[np.ndarray], peaks: np.ndarray, cfg: PostprocessConfig
) -> None:
    """Add heel strikes whose run of frames is cut off by a pass edge (in place).

    The normal rule needs a complete run of at least ``min_run`` frames, so a
    strike on the first or last frames of a pass - where the child walks into
    or out of the walking area - is lost, and with it a whole gait cycle. Here
    a run of frames with probability > ``threshold`` that touches a pass edge
    and is at least ``edge_min_run`` long gives a strike half a typical run
    (median complete run of that class in the recording) in from its cut-off
    end, clamped to the pass. Skipped if the normal rule already found a
    strike of that class within one typical run.
    """
    for cls in (HEEL_ON_LTR, HEEL_ON_RTL):
        complete = [
            e - s
            for rows in passes
            for s, e in _runs(proba[rows, cls] > cfg.threshold)
            if s > 0 and e < len(rows) and e - s >= cfg.min_run
        ]
        typical = float(np.median(complete)) if complete else 2.0 * cfg.min_run
        for rows in passes:
            n = len(rows)
            for s, e in _runs(proba[rows, cls] > cfg.threshold):
                if e - s < cfg.edge_min_run or (s > 0 and e < n) or (s == 0 and e == n):
                    continue
                pos = e - typical / 2 if s == 0 else s + typical / 2 - 1
                pos = int(min(max(round(pos), 0), n - 1))
                lo, hi = max(0, pos - int(typical)), pos + int(typical) + 1
                if (peaks[rows[lo:hi]] == cls).any():
                    continue
                peaks[rows[pos]] = cls


def _paired_intervals(peaks: np.ndarray, cls: int) -> list[tuple[int, int]]:
    """Intervals between same-class events, exactly as the original loop.

    An interval is only formed when no other-class event lies between the two
    events, and the second event of a pair is not reused as the start of the
    next pair (so intervals are (e0, e1), (e2, e3), ...).
    """
    n = len(peaks)
    pairs, end = [], -1
    for j in range(n):
        if peaks[j] == cls and j > end:
            for k in range(j + 1, n - 1):
                if peaks[k] == cls:
                    end = k
                    pairs.append((j, k))
                    break
                if peaks[k] in (HEEL_ON_LTR, HEEL_ON_RTL):
                    end = k
                    break
    return pairs


def postprocess(
    proba: np.ndarray, cfg: PostprocessConfig | None = None, segment: np.ndarray | None = None
) -> GaitEvents:
    """Turn per-frame class probabilities ``(T, 3)`` into heel-on events.

    With ``segment`` (walking-pass id per row), smoothing, run detection and the
    interval-based outlier check never cross a pass boundary; the median
    interval is still taken over the whole recording. Without it (one pass),
    the result is identical to the original notebook if ``cfg.edge_min_run``
    is 0.

    Note (original behaviour, kept for reproducibility): a frame where no class
    exceeds ``threshold`` becomes class 0 (heel-on L->R) before smoothing,
    because ``argmax`` of an all-zero row is 0.
    """
    cfg = cfg or PostprocessConfig()
    proba = np.asarray(proba, dtype=float)
    seg = np.zeros(len(proba), dtype=int) if segment is None else np.asarray(segment)
    passes = [np.flatnonzero(seg == s) for s in np.unique(seg)]

    peaks = np.full(len(proba), float(NO_EVENT))
    for rows in passes:
        hard = np.argmax((proba[rows] > cfg.threshold).astype(int), axis=1)
        smooth = _mode_smooth(hard, cfg.smooth_window)
        for cls in (HEEL_ON_LTR, HEEL_ON_RTL):
            peaks[rows[_run_midpoints(smooth, cls, cfg.min_run)]] = cls
    if cfg.edge_min_run > 0:
        _edge_events(proba, passes, peaks, cfg)

    pairs: dict[int, list[tuple[int, int]]] = {HEEL_ON_LTR: [], HEEL_ON_RTL: []}
    for rows in passes:
        for cls in pairs:
            pairs[cls] += [(rows[a], rows[b]) for a, b in _paired_intervals(peaks[rows], cls)]
    lengths = [b - a for p in pairs.values() for a, b in p]
    if lengths:
        med, tol = np.median(lengths), cfg.outlier_sd * np.ceil(np.std(lengths))
        for p in pairs.values():
            for a, b in p:
                if b - a > med + tol or b - a < med - tol:
                    peaks[a] = peaks[b] = NO_EVENT

    return GaitEvents(
        heel_on_ltr=np.flatnonzero(peaks == HEEL_ON_LTR),
        heel_on_rtl=np.flatnonzero(peaks == HEEL_ON_RTL),
    )


@dataclass
class ModelSettings:
    """Settings a model was trained and validated with.

    Stored next to the model as ``<model>.meta.json`` so inference always
    computes the same features at the same frame rate.
    """

    feature_set: str = "angles"
    context_offsets: list[int] | None = None
    model_fps: float = 50.0
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    info: dict = field(default_factory=dict)  # free-form provenance (training data, CV scores)

    @staticmethod
    def path_for(model_path: str | Path) -> Path:
        p = Path(model_path)
        return p.with_name(p.stem + ".meta.json")

    def save(self, model_path: str | Path) -> Path:
        path = self.path_for(model_path)
        path.write_text(json.dumps({"format": 1, **asdict(self)}, indent=2))
        return path

    @classmethod
    def load(cls, model_path: str | Path) -> ModelSettings | None:
        path = cls.path_for(model_path)
        if not path.exists():
            return None
        d = json.loads(path.read_text())
        d.pop("format", None)
        d["postprocess"] = PostprocessConfig(**d.get("postprocess", {}))
        return cls(**d)


class EventDetector:
    """A trained XGBoost heel-on classifier plus post-processing.

    Only inference lives in this package; the model is trained separately and
    loaded from an XGBoost model file (``events.model`` in the config) and its
    settings file ``<model>.meta.json``.
    """

    def __init__(self, model=None, settings: ModelSettings | None = None):
        self.model = model
        self.settings = settings or ModelSettings()
        self.feature_fn = make_feature_fn(self.settings.feature_set, self.settings.context_offsets)

    @property
    def cfg(self) -> PostprocessConfig:
        return self.settings.postprocess

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("EventDetector has no model; use EventDetector.load(path)")
        return self.model.predict_proba(np.asarray(features, dtype=float))

    def predict(self, angles: np.ndarray, segment: np.ndarray | None = None) -> GaitEvents:
        """Events on rows of ``angles`` (T, 6), assumed to be at the model's frame rate."""
        return postprocess(self.predict_proba(self.feature_fn(angles, segment)), self.cfg, segment)

    def detect(self, angles: np.ndarray, seq: FrameSequence, fps: float) -> GaitEvents:
        """Events for a recording at any frame rate.

        Angles are resampled (per walking pass) to the model's frame rate, the
        model's features computed, and detected events mapped back to the
        nearest row of ``seq``.
        """
        rs = resample(angles, seq, fps, self.settings.model_fps)
        ev = self.predict(rs.values, rs.segment)
        return GaitEvents(
            heel_on_ltr=np.unique(rs.src_rows[ev.heel_on_ltr]),
            heel_on_rtl=np.unique(rs.src_rows[ev.heel_on_rtl]),
        )

    @classmethod
    def load(cls, path: str | Path, fallback: ModelSettings | None = None) -> EventDetector:
        """Load a model; settings come from ``<model>.meta.json`` if present,
        else from ``fallback`` (e.g. built from the config)."""
        from .xgb_trees import XGBTreeModel

        return cls(XGBTreeModel(path), ModelSettings.load(path) or fallback)
