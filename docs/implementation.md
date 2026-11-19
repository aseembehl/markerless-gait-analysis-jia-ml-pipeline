# Implementation notes

The README follows the paper's text. This page describes what the code in
this repository does and where each part comes from in the original research
notebooks.

Scope: the repository turns a sagittal gait video into gait-cycle plots
(hip, knee and ankle, mean +- SD per walking direction). The paper's
validation and statistical analyses (Vicon comparison, event-detection
accuracy, cohort statistics) are not part of it.

## Stages

| Stage | Code | Original notebook |
|---|---|---|
| 1. Person detection and crop | `detection/faster_rcnn.py` - torchvision `fasterrcnn_resnet50_fpn` (COCO), person score > 0.8, highest-scoring box, edge and minimum-height rules | `vicon_crop.ipynb` |
| 2. Landmarks | `landmarks/vitpose.py` - ViTPose-H COCO-WholeBody directly on the crop (model code vendored from `easy_ViTPose`, no YOLOv8 re-detection), 12 landmarks | `vicon_yolo_pose.ipynb` |
| 3. Joint angles | `kinematics/joint_angles.py` | `pointsToAngles.ipynb` |
| 4a. Heel-on events | `gait/events.py` - XGBoost frame classifier + post-processing | `heal_on_event_prediction.ipynb` |
| 4b. Gait cycles | `gait/cycles.py` - split at heel-on, 101-point normalisation, LOWESS | `plots_v7.ipynb` |
| Plots | `plots.py` - mean +- SD per joint and direction, Savitzky-Golay display smoothing | `plots_v7.ipynb` |

An earlier variant of stages 1-2 (MMDetection Faster R-CNN + TCFormer
COCO-WholeBody, MMPose 0.x) produced the 2023 `point_predictions` files. Those
CSVs can still be processed with `jia-gait analyse`.

## Landmarks

COCO-WholeBody keypoints used, in the order of `skeleton.LANDMARKS`:
shoulder L/R (5, 6), hip L/R (11, 12), knee L/R (13, 14), ankle L/R (15, 16),
toe L/R (18, 21), heel L/R (19, 22). "Toe" is the **small toe**: the original
selection `np.delete(..., [-3, -6])` removes the big toes.

Landmark coordinates are in the crop frame, as in the original; the crop box is
stored separately (`bboxes`). Joint angles do not depend on the offset.

Frames without an accepted person box or without landmarks are dropped, and
later stages work on the kept frames only.

## Known quirks kept for reproducibility

- Left-to-right ankle angle: the ankle point is shifted 1 px down
  (`ankleVertical = ankle = ...` in the original); negligible effect.
- Event post-processing: a frame where no class probability exceeds 0.5 is
  treated as a left-to-right heel-on before smoothing.
- Event outlier removal only checks every other inter-event interval.
- In the original notebooks the heel-on model was trained on angles from the
  TCFormer variant (2023 labelled sets) and applied to ViTPose-derived angles.

## Deliberate differences

- **No YOLOv8 inside the crop.** The notebook ran easy_ViTPose, which re-detects
  the person in the Faster R-CNN crop with YOLOv8 (ultralytics, AGPL-3.0)
  before ViTPose. The pipeline runs ViTPose directly on the crop padded to 3:4.
  On 100 K002 frames: landmarks differ by a median of 1.0 px, joint angles by a
  median of 0.4-1.4 deg (95th percentile 2-7 deg), similar to the port's own
  deviation from the original files. All dependencies are now under permissive
  licences (BSD/MIT/Apache-2.0); the vendored model code is Apache-2.0 (see
  `src/jia_gait/landmarks/vitpose_model/NOTICE`).

Improvements over the notebooks (they change results relative to the original
code, in favour of correct timing):

- **Dropped frames.** The notebooks treated the frames kept by stages 1-2 as
  consecutive, which compresses time across gaps and loses heel strikes that
  fall on dropped frames. The pipeline interpolates landmarks across gaps of up
  to 0.1 s and splits the recording into passes at longer gaps
  (`jia_gait/sequence.py`, config `sequence.max_gap_s`).
- **Frame rate.** The heel-on model and its post-processing work in frames at
  50 fps. Input angles are resampled to `events.model_fps` (50 Hz) per pass
  before detection, and events are mapped back to video frames, so videos at
  other frame rates are handled.
- **Cycle checks.** Cycles spanning a long gap are dropped, as are cycles
  longer than `cycles.max_duration_ratio` (1.5) times the recording's median
  cycle - usually two strides merged because a heel strike was missed.

Smaller differences:

- Frames and crops are JPEG-encoded in memory (quality 95 and 75) to mimic the
  original write-then-read workflow; encoders differ slightly from
  `cv2.imwrite`, so landmarks are close to, not bit-identical with, the
  original CSVs.

## Usage

```bash
uv venv -p 3.9 .venv
uv pip install -p .venv -e '.[dev,video,models]'

# Heel-on model: a trained XGBoost model file (JSON) at weights/heel_on_xgb.json
# (path set by `events.model` in the config). This repo contains inference only;
# the model is evaluated with numpy, so XGBoost itself is not installed.

# Video -> output folder with landmarks.csv, angles.csv, cycles.npz and plots/
jia-gait -c configs/default.yaml run video.mp4 -o outputs/K001
jia-gait run video.mp4 -o outputs/K001 --model path/to/heel_on_xgb.json

# Same from an existing landmarks CSV (point_predictions format);
# --events uses manually labelled heel strikes instead of the model
jia-gait -c configs/default.yaml analyse points/K001.csv -o outputs/K001 --fps 50
jia-gait analyse points/K001.csv -o outputs/K001 --events heel_on_events/K001.csv

# Stages 1-2 only
jia-gait landmarks video.mp4 -o points/K001.csv
```

Devices are chosen automatically (`device: auto`): ViTPose uses CUDA, else
Apple's GPU (MPS), else the CPU; Faster R-CNN uses CUDA, else the CPU. On an
M1 MacBook, ViTPose-H took 0.40 s per frame on MPS vs 1.47 s on the CPU with
identical landmarks, while Faster R-CNN was over 10x slower on MPS than on the
CPU (about 2-3 s per 1080p frame), so it stays on the CPU.
