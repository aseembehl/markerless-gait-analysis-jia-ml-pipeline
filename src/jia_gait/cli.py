"""Command-line entry point: ``jia-gait``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import pipeline
from .config import PipelineConfig
from .io.csv_formats import read_event_frames, read_point_predictions, write_point_predictions


def _load_cfg(path: str | None) -> PipelineConfig:
    return PipelineConfig.from_yaml(path) if path else PipelineConfig()


def _summary(result: pipeline.PipelineResult) -> str:
    seq = result.seq
    n_pass = int(seq.segment.max()) + 1 if len(seq) else 0
    head = f"frames: {len(result.detected_imgids)} detected, {int(seq.filled.sum())} gap-filled"
    lines = [f"{head}, {n_pass} passes, {result.fps:g} fps"]
    for d, label in (("ltr", "left->right"), ("rtl", "right->left")):
        dc = result.cycles[d]
        line = f"  {label}: {dc.n_cycles} cycles"
        if dc.n_cycles:
            roms = {j: np.ptp(m) for j, (m, _) in dc.mean_sd().items()}
            line += "  ROM " + "  ".join(f"{j} {r:.1f}" for j, r in roms.items())
        lines.append(line)
    return "\n".join(lines)


def _finish(result, out_dir, name):
    paths = pipeline.write_outputs(result, out_dir, name=name)
    n_plots = sum(k.startswith("plot") for k in paths)
    print(_summary(result))
    if result.cycles.n_cycles == 0:
        n_hs = len(result.events.heel_on_ltr) + len(result.events.heel_on_rtl)
        print(
            f"WARNING: no complete gait cycle found ({n_hs} heel strikes detected). "
            "A cycle needs two consecutive heel strikes of the same foot within one "
            "uninterrupted walking pass; no plots were written.",
            file=sys.stderr,
        )
    print(f"saved -> {out_dir}  ({n_plots} plots in {Path(out_dir) / 'plots'})")


def _check_model(args, cfg) -> None:
    """Fail before any expensive work if heel strikes can't be detected."""
    if getattr(args, "model", None):
        cfg.events.model = str(Path(args.model).resolve())
    if getattr(args, "events", None):
        return
    model = cfg.events.model
    if not model:
        sys.exit(
            "error: no heel-strike model configured. Pass --model PATH, set events.model in a "
            "config file (-c), or give manually labelled heel strikes with --events."
        )
    if not Path(model).exists():
        sys.exit(f"error: heel-strike model not found: {model}")


def cmd_analyse(args, cfg):
    _check_model(args, cfg)
    imgids, lm = read_point_predictions(args.points)
    events = read_event_frames(args.events) if args.events else None
    result = pipeline.analyse_landmarks(lm, cfg, imgids=imgids, fps=args.fps, event_frames=events)
    _finish(result, args.output, args.name or Path(args.points).stem)


def cmd_landmarks(args, cfg):
    from .detection import FasterRCNNDetector
    from .io import video
    from .landmarks import ViTPoseLandmarkDetector

    imgids, lm, _ = pipeline.detect_landmarks(
        video.iter_frames(args.video),
        FasterRCNNDetector(cfg.detection),
        ViTPoseLandmarkDetector(cfg.landmarks),
        cfg,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_point_predictions(args.output, imgids, lm)
    print(f"{len(imgids)} frames with landmarks -> {args.output}")


def cmd_run(args, cfg):
    _check_model(args, cfg)
    result = pipeline.run(args.video, cfg)
    _finish(result, args.output, args.name or Path(args.video).stem)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="jia-gait", description="Markerless gait analysis for JIA"
    )
    parser.add_argument("-c", "--config", help="YAML config (default: built-in defaults)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="Video -> landmarks, angles, gait cycles and plots")
    p.add_argument("video")
    p.add_argument("-o", "--output", required=True, help="Output folder")
    p.add_argument("--name", help="Recording name used in plot titles (default: file name)")
    p.add_argument("--model", help="Heel-strike model file (overrides events.model)")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("landmarks", help="Stages 1-2: video -> point_predictions CSV")
    p.add_argument("video")
    p.add_argument("-o", "--output", required=True, help="Output landmarks .csv")
    p.set_defaults(func=cmd_landmarks)

    p = sub.add_parser("analyse", help="Landmarks CSV -> angles, gait cycles and plots")
    p.add_argument("points")
    p.add_argument("-o", "--output", required=True, help="Output folder")
    p.add_argument("--name", help="Recording name used in plot titles (default: file name)")
    p.add_argument("--model", help="Heel-strike model file (overrides events.model)")
    p.add_argument("--fps", type=float, help="Video frame rate (default: sequence.default_fps)")
    p.add_argument(
        "--events",
        help="Use manually labelled heel strikes (heel_on_events CSV) instead of the model",
    )
    p.set_defaults(func=cmd_analyse)

    args = parser.parse_args(argv)
    try:
        args.func(args, _load_cfg(args.config))
    except (pipeline.NoLandmarksError, ValueError, FileNotFoundError) as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
