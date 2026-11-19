"""Per-recording quality summary: how much of the recording was usable and why
cycles were dropped, with plain-language warnings for the user.

Written next to the plots as ``quality.json`` and ``quality.txt``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .gait.cycles import CycleConfig, cycle_bounds, filter_bounds

DIRECTION_LABEL = {"ltr": "walking left to right", "rtl": "walking right to left"}
JOINTS = ("hip", "knee", "ankle")

# Thresholds for warnings.
MIN_CYCLES = 3  # fewer cycles: mean +- SD is unreliable
MAX_PASSES = 10  # more passes: tracking was lost often
MAX_DESPIKED = 0.03  # larger share of replaced angle samples: noisy landmarks
MIN_FPS = 40.0  # lower frame rates: heel-strike timing less precise


def _cycle_drops(events, seq, fps: float, cfg: CycleConfig) -> dict[str, dict]:
    """Re-apply the cycle filters cumulatively to count what each one removed."""
    raw = cycle_bounds(events)
    durations = (cfg.min_duration_s, cfg.max_duration_s)
    # (reason, max_duration_ratio, fps, duration range) - each step adds one filter
    steps = [
        ("crosses a gap in tracking", None, None, (None, None)),
        (f"stride outside {cfg.min_duration_s}-{cfg.max_duration_s} s", None, fps, durations),
        (
            f"longer than {cfg.max_duration_ratio} x median stride (missed heel strike)",
            cfg.max_duration_ratio,
            fps,
            durations,
        ),
    ]
    out = {d: {"candidates": len(raw[d]), "dropped": {}} for d in raw}
    prev = {d: sum(b > a for a, b in raw[d]) for d in raw}
    for reason, ratio, step_fps, rng in steps:
        kept = filter_bounds(raw, seq.imgids, seq.segment, ratio, step_fps, rng)
        for d in raw:
            if prev[d] - len(kept[d]):
                out[d]["dropped"][reason] = prev[d] - len(kept[d])
            prev[d] = len(kept[d])
    return out


def summarise(result, cycle_cfg: CycleConfig) -> dict:
    """Quality summary of a ``PipelineResult``."""
    seq, fps = result.seq, float(result.fps)
    det = result.detected_imgids
    n_pass = int(seq.segment.max()) + 1 if len(seq) else 0
    pass_s = [
        float((seq.imgids[seq.segment == s][-1] - seq.imgids[seq.segment == s][0] + 1) / fps)
        for s in range(n_pass)
    ]
    q: dict = {
        "fps": fps,
        "frames_with_landmarks": len(det),
        "first_last_frame": [int(det[0]), int(det[-1])] if len(det) else None,
        "gap_filled_frames": int(seq.filled.sum()),
        "passes": n_pass,
        "pass_duration_s": [round(p, 2) for p in pass_s],
    }
    if result.despiked is not None and len(result.despiked):
        m = np.asarray(result.despiked)
        # features order: hip, hip, knee, knee, ankle, ankle (both directions)
        q["despiked_fraction"] = {
            j: round(float(m[:, 2 * i : 2 * i + 2].mean()), 4) for i, j in enumerate(JOINTS)
        }
    drops = _cycle_drops(result.events, seq, fps, cycle_cfg) if len(seq) else {}
    q["directions"] = {}
    for d in ("ltr", "rtl"):
        dc = result.cycles[d]
        b = dc.bounds
        stride = (seq.imgids[b[:, 1]] - seq.imgids[b[:, 0]]) / fps if len(b) else np.array([])
        q["directions"][d] = {
            "heel_strikes": len(getattr(result.events, f"heel_on_{d}")),
            "cycles": int(dc.n_cycles),
            "cycle_candidates": drops.get(d, {}).get("candidates", 0),
            "dropped": drops.get(d, {}).get("dropped", {}),
            "stride_s": (
                {
                    "median": round(float(np.median(stride)), 3),
                    "min": round(float(stride.min()), 3),
                    "max": round(float(stride.max()), 3),
                }
                if len(stride)
                else None
            ),
        }
    q["warnings"] = warnings_for(q)
    return q


def warnings_for(q: dict) -> list[str]:
    w = []
    for d, info in q["directions"].items():
        n = info["cycles"]
        if n == 0:
            w.append(f"No complete gait cycle {DIRECTION_LABEL[d]}; no plots for this direction.")
        elif n < MIN_CYCLES:
            w.append(
                f"Only {n} gait cycle(s) {DIRECTION_LABEL[d]}; "
                "the mean +- SD curves are not reliable."
            )
    if q["passes"] > MAX_PASSES:
        w.append(
            f"Tracking was interrupted often ({q['passes']} separate walking passes); "
            "check that the child stays in view and is not occluded."
        )
    spikes = q.get("despiked_fraction") or {}
    noisy = [j for j, f in spikes.items() if f > MAX_DESPIKED]
    if noisy:
        w.append(
            "Many landmark glitches were corrected for "
            + ", ".join(noisy)
            + "; check the video quality (lighting, clothing, resolution)."
        )
    if q["fps"] < MIN_FPS:
        w.append(
            f"Low frame rate ({q['fps']:g} fps); heel-strike timing is less precise than at 50 fps."
        )
    return w


def format_text(q: dict, name: str = "") -> str:
    lines = [f"Quality report{f' - {name}' if name else ''}", ""]
    lines.append(f"frame rate:            {q['fps']:g} fps")
    lines.append(f"frames with landmarks: {q['frames_with_landmarks']}")
    lines.append(f"gap-filled frames:     {q['gap_filled_frames']}")
    durs = ", ".join(f"{p:.1f}" for p in q["pass_duration_s"])
    lines.append(f"walking passes:        {q['passes']} ({durs} s)")
    if q.get("despiked_fraction"):
        spikes = "  ".join(f"{j} {100 * f:.1f}%" for j, f in q["despiked_fraction"].items())
        lines.append(f"glitches corrected:    {spikes} of angle samples")
    for d, info in q["directions"].items():
        lines.append("")
        lines.append(f"{DIRECTION_LABEL[d]}:")
        lines.append(f"  heel strikes: {info['heel_strikes']}")
        lines.append(f"  gait cycles:  {info['cycles']} of {info['cycle_candidates']} candidates")
        for reason, n in info["dropped"].items():
            lines.append(f"    dropped {n}: {reason}")
        if info["stride_s"]:
            s = info["stride_s"]
            lines.append(
                f"  stride time:  median {s['median']:.2f} s (range {s['min']:.2f}-{s['max']:.2f})"
            )
    lines.append("")
    if q["warnings"]:
        lines.append("Warnings:")
        lines += [f"  - {w}" for w in q["warnings"]]
    else:
        lines.append("No warnings.")
    return "\n".join(lines) + "\n"


def write(q: dict, out_dir: str | Path, name: str = "") -> tuple[Path, Path]:
    out = Path(out_dir)
    pj, pt = out / "quality.json", out / "quality.txt"
    pj.write_text(json.dumps(q, indent=2))
    pt.write_text(format_text(q, name))
    return pj, pt
