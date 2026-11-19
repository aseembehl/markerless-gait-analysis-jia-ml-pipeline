"""End-to-end orchestration of the four stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import PipelineConfig
from .detection import BBox, FasterRCNNDetector, PersonDetector
from .gait import CycleConfig, EventDetector, GaitCycles, PostprocessConfig, extract_gait_cycles
from .gait.events import ModelSettings
from .io import video
from .io.csv_formats import GaitEvents, frames_to_rows
from .kinematics import JointAngles, compute_joint_angles, despike_angles
from .landmarks import LANDMARKS, LandmarkDetector, ViTPoseLandmarkDetector
from .sequence import FrameSequence, fill_gaps, max_gap_frames


@dataclass
class PipelineResult:
    """Rows of ``seq``/``angles``/``events`` follow the gap-filled frame sequence."""

    detected_imgids: np.ndarray  # (D,) frames with detected landmarks
    detected_landmarks: np.ndarray  # (D, 12, 2|3) x, y[, conf] in crop coordinates
    bboxes: np.ndarray  # (D, 4) crop box x1, y1, x2, y2 in the full frame
    fps: float
    seq: FrameSequence  # gap-filled sequence (frame ids, landmarks, segments)
    angles: JointAngles
    events: GaitEvents  # row indices into seq
    cycles: GaitCycles
    despiked: np.ndarray | None = None  # (T, 6) angle samples replaced by the spike filter
    quality: dict | None = None  # see jia_gait.quality

    @property
    def imgids(self) -> np.ndarray:
        return self.seq.imgids


def jpeg_roundtrip(img: np.ndarray, quality: int | None) -> np.ndarray:
    """Encode/decode an RGB image as JPEG, as writing and re-reading a file did."""
    if quality is None:
        return img
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"))


def detect_landmarks(
    frames,
    detector: PersonDetector,
    landmark_model: LandmarkDetector,
    cfg: PipelineConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stages 1-2 over an iterable of RGB frames.

    Frames without an accepted person box or without landmarks are dropped
    (not kept as NaN rows), as in the original; later stages work on the kept
    rows only. Returns ``(imgids, landmarks (T, 12, 3), bboxes (T, 4))``.
    """
    img_cfg = (cfg or PipelineConfig()).images
    detector.reset()
    ids, lms, boxes = [], [], []
    for idx, frame in enumerate(frames):
        frame = jpeg_roundtrip(frame, img_cfg.frame_jpeg_quality)
        box: BBox | None = detector.detect(frame)
        if box is None:
            continue
        crop = jpeg_roundtrip(box.crop(frame), img_cfg.crop_jpeg_quality)
        kp = landmark_model.predict(crop)
        if kp is None:
            continue
        ids.append(idx)
        lms.append(kp)
        boxes.append((box.x1, box.y1, box.x2, box.y2))
    return (
        np.asarray(ids, dtype=int),
        np.asarray(lms, dtype=float).reshape(-1, len(LANDMARKS), 3),
        np.asarray(boxes, dtype=float).reshape(-1, 4),
    )


def event_detector_from_config(cfg: PipelineConfig) -> EventDetector:
    e = cfg.events
    if not e.model:
        raise ValueError("No heel-on event model configured (events.model)")
    fallback = ModelSettings(
        model_fps=e.model_fps,
        postprocess=PostprocessConfig(
            e.threshold, e.smooth_window, e.min_run, e.outlier_sd, e.edge_min_run
        ),
    )
    return EventDetector.load(e.model, fallback)


def cycle_config(cfg: PipelineConfig) -> CycleConfig:
    c = cfg.cycles
    return CycleConfig(
        c.n_points,
        c.smoothing,
        c.lowess_frac,
        c.lowess_it,
        c.wrap_frac,
        c.max_duration_ratio,
        c.min_duration_s,
        c.max_duration_s,
    )


class NoLandmarksError(ValueError):
    """Raised when no frame of a recording has landmarks (no child detected)."""


def analyse_landmarks(
    landmarks: np.ndarray,
    cfg: PipelineConfig,
    imgids: np.ndarray | None = None,
    fps: float | None = None,
    event_frames: GaitEvents | None = None,
    event_detector: EventDetector | None = None,
    bboxes: np.ndarray | None = None,
) -> PipelineResult:
    """Stages 3-4 on detected landmarks.

    ``event_frames``: manually labelled events as *frame ids* (e.g. from
    ``read_event_frames``); if None, heel strikes are detected by the model.
    """
    imgids = np.arange(len(landmarks)) if imgids is None else np.asarray(imgids, dtype=int)
    if len(imgids) == 0:
        raise NoLandmarksError(
            "No frame has landmarks: no person was detected in the walking area. "
            "Check that the video is a sagittal recording with the child walking across "
            "the central part of the image."
        )
    fps = float(fps or cfg.sequence.default_fps)
    seq = fill_gaps(imgids, landmarks, max_gap_frames(cfg.sequence.max_gap_s, fps))
    angles = compute_joint_angles(seq.landmarks)
    despiked = None
    if cfg.angles.despike:
        a = cfg.angles
        angles, despiked = despike_angles(
            angles, seq.segment, fps, a.half_window_s, a.threshold_deg, a.n_sigma
        )
    if event_frames is not None:
        events = frames_to_rows(event_frames, seq.imgids)
    else:
        detector = event_detector or event_detector_from_config(cfg)
        events = detector.detect(angles.features(), seq, fps)
    cycles = extract_gait_cycles(
        angles, events, cycle_config(cfg), imgids=seq.imgids, segment=seq.segment, fps=fps
    )
    result = PipelineResult(
        detected_imgids=imgids,
        detected_landmarks=np.asarray(landmarks),
        bboxes=np.full((len(imgids), 4), np.nan) if bboxes is None else bboxes,
        fps=fps,
        seq=seq,
        angles=angles,
        events=events,
        cycles=cycles,
        despiked=despiked,
    )
    from . import quality

    result.quality = quality.summarise(result, cycle_config(cfg))
    return result


def run(
    video_path: str | Path,
    cfg: PipelineConfig,
    detector: PersonDetector | None = None,
    landmark_model: LandmarkDetector | None = None,
    event_detector: EventDetector | None = None,
) -> PipelineResult:
    detector = detector or FasterRCNNDetector(cfg.detection)
    landmark_model = landmark_model or ViTPoseLandmarkDetector(cfg.landmarks)
    fps = video.probe(video_path).fps
    imgids, landmarks, bboxes = detect_landmarks(
        video.iter_frames(video_path), detector, landmark_model, cfg
    )
    return analyse_landmarks(
        landmarks, cfg, imgids=imgids, fps=fps, event_detector=event_detector, bboxes=bboxes
    )


def write_outputs(result: PipelineResult, out_dir: str | Path, name: str = "") -> dict[str, Path]:
    """Write everything for one recording into ``out_dir``::

    landmarks.csv   detected landmarks (point_predictions format; dropped frames absent)
    angles.csv      angles on the gap-filled sequence + heel-on events (angle_preds format)
    cycles.npz      normalised cycles per direction and joint (+ events, bboxes,
                    and the mask of angle samples replaced by the spike filter)
    quality.json / quality.txt    usable frames, passes, dropped cycles, warnings
    plots/<joint>_<ltr|rtl>.png   mean +- SD gait-cycle plots
    """
    from .io.csv_formats import write_angle_csv, write_point_predictions
    from .plots import save_cycle_plots

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "landmarks": out / "landmarks.csv",
        "angles": out / "angles.csv",
        "cycles": out / "cycles.npz",
    }
    write_point_predictions(paths["landmarks"], result.detected_imgids, result.detected_landmarks)
    write_angle_csv(paths["angles"], result.imgids, result.angles.features(), result.events)

    arrays = {
        "imgids": result.imgids,
        "segment": result.seq.segment,
        "filled": result.seq.filled,
        "fps": result.fps,
        "detected_imgids": result.detected_imgids,
        "bboxes": result.bboxes,
        "heel_on_ltr": result.events.heel_on_ltr,
        "heel_on_rtl": result.events.heel_on_rtl,
        "cycle_grid": result.cycles.grid,
    }
    if result.despiked is not None:
        arrays["despiked"] = result.despiked
    for d in ("ltr", "rtl"):
        dc = result.cycles[d]
        arrays[f"{d}_bounds"] = dc.bounds
        arrays[f"{d}_toe_off"] = dc.toe_off
        for joint, Y in dc.angles.items():
            arrays[f"{d}_{joint}"] = Y
    np.savez(paths["cycles"], **arrays)

    if result.quality is not None:
        from . import quality

        paths["quality_json"], paths["quality_txt"] = quality.write(result.quality, out, name)

    for i, png in enumerate(save_cycle_plots(result.cycles, out / "plots", name=name)):
        paths[f"plot{i}"] = png
    return paths
