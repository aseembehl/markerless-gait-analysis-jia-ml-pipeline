# Vendored from easy_ViTPose (https://github.com/JunkyByte/easy_ViTPose), Apache-2.0,
# itself derived from ViTPose / MMPose (Apache-2.0). See LICENSE and NOTICE in this folder.
# Changes: import paths only; wholebody channel setup from configs/ViTPose_wholebody.py.

import copy

from . import config_common as _common

NUM_KEYPOINTS = 133  # COCO-WholeBody
IMAGE_SIZE = tuple(_common.data_cfg["image_size"])  # (width, height) = (192, 256)
_SIZES = {"s": "model_small", "b": "model_base", "l": "model_large", "h": "model_huge"}


def wholebody_model_cfg(size: str) -> dict:
    cfg = copy.deepcopy(getattr(_common, _SIZES[size]))
    cfg["keypoint_head"]["out_channels"] = NUM_KEYPOINTS
    return cfg
