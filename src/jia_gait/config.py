"""Configuration loading."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

# Heel-strike model shipped with the repository (used when no other is configured).
BUNDLED_MODEL = Path(__file__).resolve().parents[2] / "weights" / "heel_on_xgb.json"


def _bundled_model() -> str | None:
    return str(BUNDLED_MODEL) if BUNDLED_MODEL.exists() else None


@dataclass
class DetectionConfig:
    """Faster R-CNN person crop."""

    score_threshold: float = 0.8
    edge_fraction: float = 1 / 6
    edge_rule: str = "overlaps"  # or "inside" (stricter, for tightly framed videos)
    max_edge_frames: int = 4
    min_height: float = 200  # px
    device: str = "auto"  # auto = cuda if available, else cpu (MPS is very slow for this model)


@dataclass
class LandmarkConfig:
    """ViTPose-H wholebody, run directly on the person crop."""

    model_size: str = "h"
    dataset: str = "wholebody"
    model_path: str | None = None  # local checkpoint; downloaded from the HF hub if unset
    device: str = "auto"  # auto = cuda, else mps (Apple GPU), else cpu


@dataclass
class ImageConfig:
    """JPEG compression applied to frames (quality 95) and person crops
    (quality 75) before landmark detection. Set to null to skip it.
    """

    frame_jpeg_quality: int | None = 95
    crop_jpeg_quality: int | None = 75


@dataclass
class SequenceConfig:
    """Handling of frames dropped in stages 1-2 (see jia_gait.sequence)."""

    max_gap_s: float = 0.1  # fill gaps up to this long by interpolation; longer gaps split passes
    default_fps: float = 50.0  # used when the frame rate is unknown (landmark CSV input)


@dataclass
class AngleConfig:
    """Spike removal on the joint-angle series (jia_gait.kinematics.despike)."""

    despike: bool = True
    half_window_s: float = 0.06  # running-median half window (3 frames at 50 fps)
    threshold_deg: float = 20.0  # minimum deviation from the running median to count as a spike
    n_sigma: float = 3.0  # ... and more than this many robust SDs


@dataclass
class EventConfig:
    # Path to a trained XGBoost model (.json); default: the bundled
    # weights/heel_on_xgb.json. Its settings file <model>.meta.json (feature set,
    # frame rate, post-processing) takes precedence over the values below, which
    # are only used for models without one.
    model: str | None = field(default_factory=_bundled_model)
    model_fps: float = 50.0
    threshold: float = 0.5
    smooth_window: int = 5
    min_run: int = 4
    outlier_sd: float = 2.0
    edge_min_run: int = 2


@dataclass
class CycleSettings:
    n_points: int = 101
    smoothing: str | None = "lowess"
    lowess_frac: float = 0.08
    lowess_it: int = 1
    wrap_frac: float = 0.04
    max_duration_ratio: float | None = 1.5  # drop cycles longer than this x median; null = keep
    min_duration_s: float | None = 0.4  # plausible stride duration range (seconds)
    max_duration_s: float | None = 3.0


@dataclass
class PipelineConfig:
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    landmarks: LandmarkConfig = field(default_factory=LandmarkConfig)
    images: ImageConfig = field(default_factory=ImageConfig)
    sequence: SequenceConfig = field(default_factory=SequenceConfig)
    angles: AngleConfig = field(default_factory=AngleConfig)
    events: EventConfig = field(default_factory=EventConfig)
    cycles: CycleSettings = field(default_factory=CycleSettings)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineConfig:
        kwargs = {}
        for f in fields(cls):
            section = data.get(f.name) or {}
            section_cls = f.default_factory  # type: ignore[misc]
            known = {sf.name for sf in fields(section_cls)}
            unknown = set(section) - known
            if unknown:
                raise ValueError(f"Unknown keys in '{f.name}' config: {sorted(unknown)}")
            kwargs[f.name] = section_cls(**section)
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        """Load a YAML config. Relative file paths in it (model files) are
        resolved against the config file's directory."""
        path = Path(path)
        with open(path) as fh:
            cfg = cls.from_dict(yaml.safe_load(fh) or {})
        base = path.resolve().parent

        def resolve(p: str | None) -> str | None:
            return p if p is None or Path(p).is_absolute() else str((base / p).resolve())

        cfg.events.model = resolve(cfg.events.model)
        cfg.landmarks.model_path = resolve(cfg.landmarks.model_path)
        return cfg
