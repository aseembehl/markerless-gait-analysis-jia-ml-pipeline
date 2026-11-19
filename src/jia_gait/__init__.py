"""Markerless sagittal-plane gait analysis pipeline for JIA.

Stages:
    1. detection   - torchvision Faster R-CNN person box per frame -> crop
    2. landmarks   - ViTPose-H (COCO-WholeBody) landmarks on the crop
    3. kinematics  - frame-by-frame hip/knee/ankle angles for both walking directions
    4. gait        - XGBoost heel-on detection and normalised gait cycles

See docs/usage.md for how to run it.
"""

__version__ = "0.1.0"
