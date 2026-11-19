"""Turn the kept frames of a recording into a gap-aware time series.

Stages 1-2 drop frames without an accepted person box or landmarks. Treating
the remaining rows as consecutive would compress time across gaps and lose
heel strikes that fall on dropped frames. Instead:

* short gaps (up to ``max_gap`` missing frames) inside a walking pass are filled
  by linear interpolation of the landmarks;
* longer gaps (child out of view, turning) split the recording into
  *segments*. Rows of different segments are still stored back to back, but
  gait cycles never span a segment boundary and frame-rate resampling is done
  per segment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FrameSequence:
    imgids: np.ndarray  # (T,) frame ids, strictly increasing
    landmarks: np.ndarray  # (T, K, C)
    segment: np.ndarray  # (T,) segment id (0, 1, ...)
    filled: np.ndarray  # (T,) True for interpolated rows

    def __len__(self) -> int:
        return len(self.imgids)

    def row_of(self) -> dict[int, int]:
        return {int(f): r for r, f in enumerate(self.imgids)}


def max_gap_frames(max_gap_s: float, fps: float) -> int:
    return max(0, round(max_gap_s * fps))


def fill_gaps(imgids: np.ndarray, landmarks: np.ndarray, max_gap: int) -> FrameSequence:
    """Interpolate gaps of up to ``max_gap`` missing frames; split at longer gaps."""
    ids = np.asarray(imgids, dtype=int)
    lm = np.asarray(landmarks, dtype=float)
    if len(ids) == 0:
        return FrameSequence(ids, lm, np.zeros(0, int), np.zeros(0, bool))
    order = np.argsort(ids, kind="stable")
    ids, lm = ids[order], lm[order]
    if np.any(np.diff(ids) == 0):
        raise ValueError("Duplicate frame ids")

    out_ids, out_lm, out_seg, out_fill = [ids[:1]], [lm[:1]], [0], [False]
    seg = 0
    for k in range(1, len(ids)):
        missing = ids[k] - ids[k - 1] - 1
        if 0 < missing <= max_gap:
            fill_ids = np.arange(ids[k - 1] + 1, ids[k])
            w = ((fill_ids - ids[k - 1]) / (ids[k] - ids[k - 1]))[:, None, None]
            out_ids.append(fill_ids)
            out_lm.append((1 - w) * lm[k - 1] + w * lm[k])
            out_seg += [seg] * len(fill_ids)
            out_fill += [True] * len(fill_ids)
        elif missing > max_gap:
            seg += 1
        out_ids.append(ids[k : k + 1])
        out_lm.append(lm[k : k + 1])
        out_seg.append(seg)
        out_fill.append(False)
    return FrameSequence(
        imgids=np.concatenate(out_ids),
        landmarks=np.concatenate(out_lm),
        segment=np.asarray(out_seg, dtype=int),
        filled=np.asarray(out_fill, dtype=bool),
    )


@dataclass
class Resampled:
    """A series resampled to a target rate, segment by segment."""

    values: np.ndarray  # (N, F)
    segment: np.ndarray  # (N,)
    time: np.ndarray  # (N,) seconds since the segment's first frame
    src_rows: np.ndarray  # (N,) nearest source row (for mapping results back)
    row_time: np.ndarray  # (T,) time of each source row within its segment
    row_segment: np.ndarray  # (T,) segment of each source row


def resample(values: np.ndarray, seq: FrameSequence, fps: float, target_fps: float) -> Resampled:
    """Linearly resample ``values`` (rows aligned with ``seq``) to ``target_fps``.

    When ``fps == target_fps`` this is the identity (same rows, same order).
    """
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    row_time = np.zeros(len(seq))
    for s in np.unique(seq.segment):
        m = seq.segment == s
        row_time[m] = (seq.imgids[m] - seq.imgids[m][0]) / fps
    if np.isclose(fps, target_fps):
        return Resampled(
            values, seq.segment.copy(), row_time, np.arange(len(seq)), row_time, seq.segment.copy()
        )

    vals, segs, times, rows = [], [], [], []
    for s in np.unique(seq.segment):
        r = np.flatnonzero(seq.segment == s)
        t_src = (seq.imgids[r] - seq.imgids[r[0]]) / fps
        t_new = np.arange(0.0, t_src[-1] + 1e-9, 1.0 / target_fps)
        vals.append(
            np.column_stack([np.interp(t_new, t_src, values[r, j]) for j in range(values.shape[1])])
        )
        segs.append(np.full(len(t_new), s))
        times.append(t_new)
        nearest = np.abs(t_src[None, :] - t_new[:, None]).argmin(axis=1)
        rows.append(r[nearest])
    return Resampled(
        np.concatenate(vals),
        np.concatenate(segs),
        np.concatenate(times),
        np.concatenate(rows),
        row_time,
        seq.segment.copy(),
    )


def to_resampled_index(rows: np.ndarray, rs: Resampled) -> np.ndarray:
    """Map source rows (e.g. labelled heel strikes) to the resampled index closest in time."""
    out = set()
    for r in np.asarray(rows, dtype=int):
        cand = np.flatnonzero(rs.segment == rs.row_segment[r])
        out.add(int(cand[np.abs(rs.time[cand] - rs.row_time[r]).argmin()]))
    return np.asarray(sorted(out), dtype=int)
