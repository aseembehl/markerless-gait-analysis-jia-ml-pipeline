"""Stage 2: ViTPose-H (COCO-WholeBody) landmarks on the person crop.

ViTPose runs directly on the Faster R-CNN crop, padded to the model's 3:4
aspect ratio. The original notebook (``vicon_yolo_pose.ipynb``, via
``easy_ViTPose``) first re-detected the person inside the crop with YOLOv8;
that step is redundant here and was removed to avoid the AGPL-licensed
ultralytics dependency; landmarks differ by about 1 px.

The model code (``vitpose_model/``) is a minimal copy of easy_ViTPose's
inference code (Apache-2.0); the pretrained weights are downloaded from the
``JunkyByte/easy_ViTPose`` Hugging Face repository unless a local file is set.

Landmark selection from the 133 COCO-WholeBody keypoints, as in the original:
keypoints 5..22, drop elbows and wrists, then ``np.delete([-3, -6])``. That
removes the *big* toes, so "toe" is the **small toe**:

    shoulder L/R (5, 6), hip L/R (11, 12), knee L/R (13, 14), ankle L/R (15, 16),
    toe L/R = small toe (18, 21), heel L/R (19, 22)

Coordinates stay in the crop frame (they were never mapped back to the full
image); joint angles do not depend on this.

Requires the ``models`` extra.
"""

from __future__ import annotations

import numpy as np

from ..config import LandmarkConfig
from ..device import resolve_device
from .base import LandmarkDetector

# COCO-WholeBody indices in skeleton.LANDMARKS order:
# shoulder L, shoulder R, hip L, hip R, knee L, knee R, ankle L, ankle R,
# toe L, toe R, heel L, heel R
WHOLEBODY_INDEX = np.array([5, 6, 11, 12, 13, 14, 15, 16, 18, 21, 19, 22])

HF_REPO = "JunkyByte/easy_ViTPose"
_MEAN = np.array([0.485, 0.456, 0.406])
_STD = np.array([0.229, 0.224, 0.225])
_ASPECT = 3 / 4  # width / height of the model input


def select_landmarks(kpts_yx: np.ndarray) -> np.ndarray:
    """``(133, 3)`` keypoints as (y, x, score) -> ``(12, 3)`` (x, y, score)."""
    k = np.asarray(kpts_yx, dtype=float)[WHOLEBODY_INDEX]
    return k[:, [1, 0, 2]]


def pad_to_aspect(image: np.ndarray, aspect_ratio: float = _ASPECT) -> tuple[np.ndarray, int, int]:
    """Zero-pad to ``aspect_ratio`` (width/height), centred, as easy_ViTPose's
    ``pad_image``. Returns ``(padded, left_pad, top_pad)``."""
    h, w = image.shape[:2]
    left = top = 0
    if w / h < aspect_ratio:
        pad = int(aspect_ratio * h) - w
        left = pad // 2
        image = np.pad(image, ((0, 0), (left, pad - left), (0, 0)), mode="constant")
    else:
        pad = int(w / aspect_ratio) - h
        top = pad // 2
        image = np.pad(image, ((top, pad - top), (0, 0), (0, 0)), mode="constant")
    return image, left, top


class ViTPoseLandmarkDetector(LandmarkDetector):
    def __init__(self, cfg: LandmarkConfig):
        self.cfg = cfg
        self._model = None

    def _weights_path(self) -> str:
        if self.cfg.model_path:
            return self.cfg.model_path
        from huggingface_hub import hf_hub_download

        size, ds = self.cfg.model_size, self.cfg.dataset
        return hf_hub_download(HF_REPO, f"torch/{ds}/vitpose-{size}-{ds}.pth")

    def _load(self) -> None:
        import torch

        from .vitpose_model.config import IMAGE_SIZE, wholebody_model_cfg
        from .vitpose_model.model import ViTPose

        if self.cfg.dataset != "wholebody":
            raise ValueError("Only the COCO-WholeBody ViTPose models are supported")
        model = ViTPose(wholebody_model_cfg(self.cfg.model_size))
        ckpt = torch.load(self._weights_path(), map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt.get("state_dict", ckpt))
        self._torch = torch
        self._input_size = IMAGE_SIZE  # (width, height)
        self.device = resolve_device(self.cfg.device, allow_mps=True)
        self._model = model.to(self.device).eval()

    def keypoints(self, image: np.ndarray) -> np.ndarray:
        """All 133 keypoints ``(y, x, score)`` for an RGB image of one person."""
        import cv2

        from .vitpose_model.decode import keypoints_from_heatmaps

        if self._model is None:
            self._load()
        h, w = image.shape[:2]
        x = cv2.resize(image, self._input_size, interpolation=cv2.INTER_LINEAR) / 255
        x = ((x - _MEAN) / _STD).transpose(2, 0, 1)[None].astype(np.float32)
        torch = self._torch
        with torch.no_grad():
            heatmaps = self._model(torch.from_numpy(x).to(self.device)).cpu().numpy()
        points, prob = keypoints_from_heatmaps(
            heatmaps=heatmaps,
            center=np.array([[w // 2, h // 2]]),
            scale=np.array([[w, h]]),
            unbiased=True,
            use_udp=True,
        )
        return np.concatenate([points[:, :, ::-1], prob], axis=2)[0]

    def predict(self, crop: np.ndarray) -> np.ndarray | None:
        if crop.size == 0:
            return None
        padded, left, top = pad_to_aspect(np.ascontiguousarray(crop, dtype=np.uint8))
        k = self.keypoints(padded)
        k[:, 0] -= top
        k[:, 1] -= left
        return select_landmarks(k)
