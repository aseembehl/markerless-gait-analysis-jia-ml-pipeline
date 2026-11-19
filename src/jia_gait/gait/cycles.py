"""Stage 4b: assemble joint-angle sequences into normalised gait cycles.

1. A cycle runs from one heel-on to the next heel-on *of the same walking
   direction*. Cycles interrupted by an event of the other direction (i.e. the
   last cycle before the child turns) are dropped, as is the open-ended last
   cycle of each direction.
2. Time within a cycle is normalised to 0..1 and linearly interpolated onto a
   grid of ``n_points`` (101 -> 0, 1, ..., 100 % of the cycle).
3. Optionally each cycle is smoothed with LOWESS using a short wrap-around
   pad at both ends (defaults: frac=0.08, it=1, wrap 4 %).

When frame ids / segments are given, cycles
that span a segment boundary (a long gap in the video) are dropped, and so are
cycles longer than ``max_duration_ratio`` times the recording's median cycle
duration - typically two strides merged because a heel strike was missed.

Left-to-right cycles use the "Angle 2" (right-side) series and right-to-left
cycles the "Angle 1" (left-side) series.
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass, field

import numpy as np

from ..io.csv_formats import GaitEvents
from ..kinematics.joint_angles import JointAngles

DIRECTIONS = ("ltr", "rtl")


@dataclass
class CycleConfig:
    n_points: int = 101
    smoothing: str | None = "lowess"  # "lowess" or None
    lowess_frac: float = 0.08
    lowess_it: int = 1
    wrap_frac: float = 0.04
    max_duration_ratio: float | None = 1.5  # None disables the duration check
    # Plausible stride duration in seconds (only applied when the frame rate is known)
    min_duration_s: float | None = 0.4
    max_duration_s: float | None = 3.0


@dataclass
class DirectionCycles:
    """Cycles for one walking direction; each angle array is ``(n_cycles, n_points)``."""

    angles: dict[str, np.ndarray]
    bounds: np.ndarray  # (n_cycles, 2) heel-on frame indices [start, end]
    toe_off: np.ndarray  # (n_cycles,) toe-off as fraction of cycle, NaN if unknown

    @property
    def n_cycles(self) -> int:
        return len(self.bounds)

    def mean_sd(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Mean and SD (ddof=1) across cycles per joint."""
        out = {}
        for joint, Y in self.angles.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                out[joint] = (np.nanmean(Y, axis=0), np.nanstd(Y, axis=0, ddof=1))
        return out


@dataclass
class GaitCycles:
    ltr: DirectionCycles
    rtl: DirectionCycles
    grid: np.ndarray = field(repr=False, default=None)  # type: ignore[assignment]

    def __getitem__(self, direction: str) -> DirectionCycles:
        return getattr(self, direction)

    @property
    def n_cycles(self) -> int:
        return self.ltr.n_cycles + self.rtl.n_cycles


def cycle_bounds(events: GaitEvents) -> dict[str, list[tuple[int, int]]]:
    """Heel-on pairs per direction, following the split/removal rules above."""
    ltr = np.asarray(events.heel_on_ltr, dtype=int)
    rtl = np.asarray(events.heel_on_rtl, dtype=int)
    clash = np.intersect1d(ltr, rtl)
    if clash.size:
        raise ValueError(
            f"Heel-on events in both directions on the same frame(s): {clash.tolist()}"
        )
    merged = sorted([(int(f), "ltr") for f in ltr] + [(int(f), "rtl") for f in rtl])
    out: dict[str, list[tuple[int, int]]] = {d: [] for d in DIRECTIONS}
    for (a, da), (b, db) in itertools.pairwise(merged):
        if da == db:
            out[da].append((a, b))
    return out


def _to_grid(y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    x = np.linspace(0.0, 1.0, len(y))
    ok = ~np.isnan(y)
    if ok.sum() < 2:
        return np.full(len(grid), np.nan)
    return np.interp(grid, x[ok], y[ok])


def _lowess_wrap(ygrid: np.ndarray, grid: np.ndarray, cfg: CycleConfig) -> np.ndarray:
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
    except ImportError as exc:  # pragma: no cover
        raise ImportError("LOWESS smoothing needs statsmodels") from exc
    w = max(2, round(cfg.wrap_frac * len(ygrid)))
    x_ext = np.concatenate([grid[-w:] - 1.0, grid, grid[:w] + 1.0])
    y_ext = np.concatenate([ygrid[-w:], ygrid, ygrid[:w]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sm = lowess(y_ext, x_ext, frac=cfg.lowess_frac, it=cfg.lowess_it, return_sorted=True)
    return _fill(np.interp(grid, sm[:, 0], sm[:, 1], left=np.nan, right=np.nan))


def _fill(y: np.ndarray) -> np.ndarray:
    ok = ~np.isnan(y)
    if ok.all() or not ok.any():
        return y
    idx = np.arange(len(y))
    return np.interp(idx, idx[ok], y[ok])


def filter_bounds(
    bounds: dict[str, list[tuple[int, int]]],
    imgids: np.ndarray | None,
    segment: np.ndarray | None,
    max_duration_ratio: float | None,
    fps: float | None = None,
    duration_range_s: tuple[float | None, float | None] = (None, None),
) -> dict[str, list[tuple[int, int]]]:
    """Drop cycles crossing a segment boundary, outside a plausible duration
    range (seconds, needs ``fps``), or much longer than the median."""
    out = {d: [(a, b) for a, b in v if b > a] for d, v in bounds.items()}
    if segment is not None:
        out = {d: [(a, b) for a, b in v if segment[a] == segment[b]] for d, v in out.items()}

    def duration(a: int, b: int) -> int:
        return int(b - a) if imgids is None else int(imgids[b] - imgids[a])

    lo, hi = duration_range_s
    if fps:
        out = {
            d: [
                (a, b)
                for a, b in v
                if (lo is None or duration(a, b) / fps >= lo)
                and (hi is None or duration(a, b) / fps <= hi)
            ]
            for d, v in out.items()
        }
    if max_duration_ratio is not None:
        dur = [duration(a, b) for v in out.values() for a, b in v]
        if dur:
            limit = max_duration_ratio * float(np.median(dur))
            out = {d: [(a, b) for a, b in v if duration(a, b) <= limit] for d, v in out.items()}
    return out


def extract_gait_cycles(
    angles: JointAngles,
    events: GaitEvents,
    cfg: CycleConfig | None = None,
    imgids: np.ndarray | None = None,
    segment: np.ndarray | None = None,
    fps: float | None = None,
) -> GaitCycles:
    cfg = cfg or CycleConfig()
    if cfg.smoothing not in (None, "lowess"):
        raise ValueError(f"Unknown smoothing {cfg.smoothing!r}")
    grid = np.linspace(0.0, 1.0, cfg.n_points)
    bounds = filter_bounds(
        cycle_bounds(events),
        imgids,
        segment,
        cfg.max_duration_ratio,
        fps,
        (cfg.min_duration_s, cfg.max_duration_s),
    )
    toe = {"ltr": np.asarray(events.toe_off_ltr, int), "rtl": np.asarray(events.toe_off_rtl, int)}

    result = {}
    for d in DIRECTIONS:
        series = angles.direction(d)
        curves: dict[str, list[np.ndarray]] = {j: [] for j in series}
        toe_frac = []
        for a, b in bounds[d]:
            for j, s in series.items():
                g = _to_grid(np.asarray(s[a : b + 1], dtype=float), grid)
                if cfg.smoothing == "lowess" and not np.isnan(g).all():
                    g = _lowess_wrap(g, grid, cfg)
                curves[j].append(g)
            t = toe[d][(toe[d] >= a) & (toe[d] <= b)]
            toe_frac.append((t[0] - a) / (b - a) if t.size and b > a else np.nan)
        result[d] = DirectionCycles(
            angles={
                j: np.asarray(v, dtype=float).reshape(-1, cfg.n_points) for j, v in curves.items()
            },
            bounds=np.asarray(bounds[d], dtype=int).reshape(-1, 2),
            toe_off=np.asarray(toe_frac, dtype=float),
        )
    return GaitCycles(ltr=result["ltr"], rtl=result["rtl"], grid=grid)
