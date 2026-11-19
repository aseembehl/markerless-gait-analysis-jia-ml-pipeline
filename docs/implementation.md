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

## Heel-on model

`weights/heel_on_xgb.json` is an XGBoost classifier (depth 4, 300 trees) on
the six joint angles at 50 Hz and their differences to the angles 1, 2, 4 and
8 frames before and after (`events.feature_set: context`, see
`gait/features.py`); its settings file `heel_on_xgb.meta.json` holds the
feature set, frame rate and post-processing. It was trained on expert-labelled
heel strikes of 52 recordings: ViTPose landmarks of 22 cohort recordings
(healthy, JIA, CP) and 12 Vicon-lab recordings, plus the 2023 TCFormer
landmarks of 18 cohort recordings. Angles were spike-filtered as in the
pipeline. Vicon recordings were trained only on the stretches where every
heel strike was labelled.

Leave-one-recording-out cross-validation on the 34 ViTPose recordings
(each recording held out with its TCFormer copy; a heel strike counts as found
within +-4 frames at 50 fps): 549 of 551 heel strikes found, no false
positives, mean timing error 0.47 frames (cohort 270/272, Vicon 279/279). The
notebook's model and post-processing (raw angles, XGBoost defaults, no
pass-edge rule) found 521/551 in the same test.

The model files contain no landmarks, angles or recording IDs.

## Known quirks kept for reproducibility

- Left-to-right ankle angle: the ankle point is shifted 1 px down
  (`ankleVertical = ankle = ...` in the original); negligible effect.
- Event post-processing: a frame where no class probability exceeds 0.5 is
  treated as a left-to-right heel-on before smoothing (the pass-edge rule
  below uses the probabilities directly, so it is not affected).
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
- **Landmark glitches.** ViTPose occasionally misplaces a joint for one or two
  frames, giving angle spikes of tens of degrees (sometimes impossible values
  such as a -200 deg ankle) that distort the mean +- SD plots. A Hampel filter
  replaces angle samples that differ from the 7-frame running median by more
  than 20 deg and 3 robust SDs (`jia_gait/kinematics/despike.py`, config
  `angles`). On 90 cohort recordings it repaired 17 of 19 cycles with
  impossible angles, while the range of motion of glitch-free mean curves was
  unchanged (ankle at most -1.2 deg; a 10 deg threshold clipped sharp ankle
  peaks after toe-off). Replaced samples are listed in `cycles.npz`
  (`despiked`). The heel-on model is trained on filtered angles.
- **Heel strikes at pass edges.** The notebook post-processing needs a
  complete run of at least 4 heel-on frames, so a heel strike on the last
  frames of a walking pass - typically as the child walks out of the walking
  area - was lost, and with it a whole gait cycle. A cut-off run of at least
  2 frames now gives a heel strike half a typical run in from where it was cut
  (config `events.edge_min_run`, 0 = off). In leave-one-recording-out
  cross-validation on 33 recordings it found 16 of 17 strikes within 4 frames
  of a pass edge (notebook rule: 9), with no false positives and unchanged
  timing; on 90 cohort recordings it added 44 gait cycles (866 -> 908).

Smaller differences:

- Frames and crops are JPEG-encoded in memory (quality 95 and 75) to mimic the
  original write-then-read workflow; encoders differ slightly from
  `cv2.imwrite`, so landmarks are close to, not bit-identical with, the
  original CSVs.

## Usage

```bash
uv venv -p 3.9 .venv
uv pip install -p .venv -e '.[dev,video,models]'

# Heel-on model: weights/heel_on_xgb.json (included; path set by `events.model`
# in the config). The model is evaluated with numpy, so XGBoost itself is not
# installed.

# Video -> output folder with landmarks.csv, angles.csv, cycles.npz, plots/ and
# quality.txt/.json (usable frames, walking passes, dropped cycles, warnings)
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
