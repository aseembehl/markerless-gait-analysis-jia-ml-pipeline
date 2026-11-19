"""Anatomical landmark definition.

Twelve landmarks (six per body side), in the column order of the original
``point_predictions/<ID>.csv`` files::

    imgids, shoulder left_x, shoulder left_y, shoulder right_x, ..., heel right_y

The order below defines the landmark axis of every ``(T, K, 2|3)`` array in
the pipeline.
"""

from __future__ import annotations

JOINTS: tuple[str, ...] = ("shoulder", "hip", "knee", "ankle", "toe", "heel")
SIDES: tuple[str, ...] = ("left", "right")

# ("shoulder left", "shoulder right", "hip left", ...)
LANDMARKS: tuple[str, ...] = tuple(f"{j} {s}" for j in JOINTS for s in SIDES)


def landmark_index(joint: str, side: str) -> int:
    name = f"{joint} {side}"
    try:
        return LANDMARKS.index(name)
    except ValueError:
        raise KeyError(f"Unknown landmark '{name}'. Known: {LANDMARKS}") from None


def csv_columns() -> list[str]:
    """Header of a ``point_predictions`` CSV."""
    return ["imgids"] + [f"{name}_{axis}" for name in LANDMARKS for axis in ("x", "y")]
