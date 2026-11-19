"""Markerless sagittal-plane gait analysis pipeline for JIA.

Stages as implemented (ported from the original research notebooks):
    1. detection   - torchvision Faster R-CNN person box per frame -> crop
    2. landmarks   - ViTPose-H (COCO-WholeBody) landmarks on the crop
    3. kinematics  - frame-by-frame hip/knee/ankle angles for both walking directions
    4. gait        - XGBoost heel-on detection and normalised gait cycles

The README follows the paper's wording (Mask R-CNN + TCFormer); see
docs/implementation.md for how the code relates to it.
"""

__version__ = "0.1.0"
