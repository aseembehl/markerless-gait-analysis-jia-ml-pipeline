"""Readers/writers for the CSV formats used by the original research code.

* ``point_predictions/<ID>.csv`` - per-frame landmarks (header, 1 + 24 columns)
* ``heel_on_events*/<ID>.csv`` - expert gait events, no header, one event per
  cell, values are image ids:
  ``<unused>, toe-off L->R, heel-on L->R, toe-off R->L, heel-on R->L``
* ``angle_preds*/<ID>.csv`` - per-frame angles plus event flags (``100`` or
  empty), see ``ANGLE_CSV_COLUMNS``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..landmarks.skeleton import LANDMARKS, csv_columns

FEATURE_COLUMNS: tuple[str, ...] = (
    "Hip Angle 2",
    "Hip Angle 1",
    "Knee Angle 2",
    "Knee Angle 1",
    "Ankle Angle 2",
    "Ankle Angle 1",
)
EVENT_COLUMNS: tuple[str, ...] = (
    "Toe off left to right",
    "Toe off right to left",
    "Heel on left to right",
    "Heel on right to left",
)
ANGLE_CSV_COLUMNS: tuple[str, ...] = ("img_id", *FEATURE_COLUMNS, *EVENT_COLUMNS)


@dataclass
class GaitEvents:
    """Event frame indices (row positions, not image ids)."""

    heel_on_ltr: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    heel_on_rtl: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    toe_off_ltr: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    toe_off_rtl: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))


def read_point_predictions(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(imgids (T,), landmarks (T, 12, 2))``."""
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = [h.strip() for h in next(reader)]
        rows = [[float(v) for v in row] for row in reader if row]
    expected = csv_columns()
    if header != expected:
        raise ValueError(f"{path}: unexpected header {header[:4]}..., expected {expected[:4]}...")
    data = np.asarray(rows, dtype=float).reshape(len(rows), len(expected))
    return data[:, 0].astype(int), data[:, 1:].reshape(len(rows), len(LANDMARKS), 2)


def write_point_predictions(path: str | Path, imgids: np.ndarray, landmarks: np.ndarray) -> None:
    flat = np.asarray(landmarks, dtype=float)[..., :2].reshape(len(imgids), 2 * len(LANDMARKS))
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(csv_columns())
        for i, row in zip(imgids, flat):
            w.writerow([int(i), *row.tolist()])


def read_event_frames(path: str | Path) -> GaitEvents:
    """Read a ``heel_on_events`` CSV; values stay *frame ids* (image ids)."""
    cols: list[list[int]] = [[], [], [], []]
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            for c in range(4):
                if len(row) > c + 1 and row[c + 1].strip() != "":
                    cols[c].append(int(float(row[c + 1])))
    toe_ltr, heel_ltr, toe_rtl, heel_rtl = (np.unique(np.array(c, dtype=int)) for c in cols)
    return GaitEvents(
        heel_on_ltr=heel_ltr, heel_on_rtl=heel_rtl, toe_off_ltr=toe_ltr, toe_off_rtl=toe_rtl
    )


def frames_to_rows(events: GaitEvents, imgids: np.ndarray) -> GaitEvents:
    """Map frame-id events to row indices of ``imgids``; frames not present are dropped."""
    row_of = {int(i): r for r, i in enumerate(imgids)}

    def rows(ids: np.ndarray) -> np.ndarray:
        return np.array(sorted({row_of[int(i)] for i in ids if int(i) in row_of}), dtype=int)

    return GaitEvents(
        heel_on_ltr=rows(events.heel_on_ltr),
        heel_on_rtl=rows(events.heel_on_rtl),
        toe_off_ltr=rows(events.toe_off_ltr),
        toe_off_rtl=rows(events.toe_off_rtl),
    )


def read_event_csv(path: str | Path, imgids: np.ndarray) -> GaitEvents:
    """Read a ``heel_on_events`` CSV and map image ids to row indices of ``imgids``."""
    return frames_to_rows(read_event_frames(path), imgids)


def write_angle_csv(
    path: str | Path, imgids: np.ndarray, features: np.ndarray, events: GaitEvents | None = None
) -> None:
    events = events or GaitEvents()
    flags = [
        set(events.toe_off_ltr.tolist()),
        set(events.toe_off_rtl.tolist()),
        set(events.heel_on_ltr.tolist()),
        set(events.heel_on_rtl.tolist()),
    ]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(ANGLE_CSV_COLUMNS)
        for r, (i, f) in enumerate(zip(imgids, features)):
            w.writerow([int(i), *f.tolist(), *(100 if r in s else "" for s in flags)])
