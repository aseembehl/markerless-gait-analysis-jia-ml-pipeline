"""Stage 1: person detection and cropping with torchvision Faster R-CNN.

Port of ``vicon_crop.ipynb``. Per frame:

1. Run COCO-pretrained ``fasterrcnn_resnet50_fpn`` and keep ``person`` boxes
   with score > ``score_threshold``; take the highest-scoring one.
2. Edge rule: boxes near the left/right image border are only accepted for a
   few frames right after a box in the central region (the child walking out
   of the frame); otherwise the frame is skipped. This rejects bystanders at the
   edge of the room before the child enters.
3. Boxes shorter than ``min_height`` px are skipped (edge-accepted boxes bypass
   this check, as in the original).

Defaults are the cohort settings (``edge_rule="overlaps"``, ``min_height=200``).
The notebook's Vicon recordings used ``edge_rule="inside"`` and ``min_height=500``.
"""

from __future__ import annotations

import numpy as np

from ..config import DetectionConfig
from ..device import resolve_device
from .base import BBox, PersonDetector

COCO_PERSON = 1


class EdgeFilter:
    """The stateful accept/skip logic of the original crop loop (model-free, testable)."""

    def __init__(self, cfg: DetectionConfig):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        # Starts "exhausted" so edge boxes are skipped until a central box is seen.
        self.edge_count = self.cfg.max_edge_frames + 1

    def _at_edge(self, box: BBox, width: int) -> bool:
        margin = width * self.cfg.edge_fraction
        if self.cfg.edge_rule == "overlaps":  # cohort: box reaches into the margin
            return box.x1 < margin or box.x2 > width - margin
        if self.cfg.edge_rule == "inside":  # Vicon: box lies entirely within the margin
            return box.x2 < margin or box.x1 > width - margin
        raise ValueError(f"Unknown edge_rule {self.cfg.edge_rule!r}")

    def __call__(self, box: BBox | None, width: int) -> BBox | None:
        if box is None:
            return None
        if self._at_edge(box, width):
            if self.edge_count >= self.cfg.max_edge_frames:
                return None
            self.edge_count += 1
            return box
        if box.height < self.cfg.min_height:
            return None
        self.edge_count = 0
        return box


def best_person(boxes, labels, scores, threshold: float) -> BBox | None:
    """Highest-scoring COCO person box with score > ``threshold``."""
    best = None
    for b, lab, s in zip(boxes, labels, scores):
        if (
            int(lab) == COCO_PERSON
            and float(s) > threshold
            and (best is None or float(s) > best.score)
        ):
            best = BBox(*(float(v) for v in b), score=float(s))
    return best


class FasterRCNNDetector(PersonDetector):
    """Requires the ``models`` extra (torch, torchvision)."""

    def __init__(self, cfg: DetectionConfig):
        self.cfg = cfg
        self.filter = EdgeFilter(cfg)
        self._model = None

    def _load(self):
        import torch
        from torchvision.models.detection import (
            FasterRCNN_ResNet50_FPN_Weights,
            fasterrcnn_resnet50_fpn,
        )

        # COCO_V1 are the weights the original got with ``pretrained=True``.
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
        self._torch = torch
        self.device = resolve_device(self.cfg.device, allow_mps=False)
        self._model = model.to(self.device).eval()

    def reset(self) -> None:
        self.filter.reset()

    def detect(self, frame: np.ndarray) -> BBox | None:
        if self._model is None:
            self._load()
        torch = self._torch
        x = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).float().div(255)
        with torch.no_grad():
            pred = self._model([x.to(self.device)])[0]
        box = best_person(
            pred["boxes"].cpu().numpy(),
            pred["labels"].cpu().numpy(),
            pred["scores"].cpu().numpy(),
            self.cfg.score_threshold,
        )
        return self.filter(box, frame.shape[1])
