# Running the pipeline

The pipeline turns a sagittal-plane video of a child walking into gait-cycle
plots: hip, knee and ankle angles over a normalised gait cycle (mean +- SD),
separately for walking left to right and right to left.

## Installation

Requires Python 3.10 or newer. With [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/aseembehl/markerless-gait-analysis-jia-ml-pipeline.git
cd markerless-gait-analysis-jia-ml-pipeline
uv venv -p 3.11 .venv
uv pip install -p .venv -e '.[video,models]'
source .venv/bin/activate
```

Or with pip: `python -m venv .venv`, `source .venv/bin/activate`, then
`pip install -e '.[video,models]'`.

The extras install PyTorch and OpenCV for processing videos. Without them, the
package can still analyse existing landmark files (`jia-gait analyse`).

On the first video, the ViTPose-H weights (about 2.4 GB) are downloaded from
the Hugging Face hub (`JunkyByte/easy_ViTPose`) and the Faster R-CNN weights
from torchvision. Both are cached for later runs.

## Quick start

```bash
jia-gait run path/to/video.mp4 -o results/child01
```

`results/child01/` then contains:

| File | Content |
|---|---|
| `plots/<joint>_<ltr\|rtl>.png` | gait-cycle plots for hip, knee and ankle, per walking direction |
| `quality.txt`, `quality.json` | how much of the video was usable, how many gait cycles were found and why others were dropped, and warnings |
| `angles.csv` | joint angles per frame, with the detected heel strikes |
| `landmarks.csv` | body landmarks per frame |
| `cycles.npz` | normalised gait cycles as arrays, for your own analysis |

Check `quality.txt` first. The command line also prints its warnings, for
example when fewer than three gait cycles were found in a direction or the
child was often lost from view.

## Recording the video

- Film from the side (sagittal plane), with the camera still and the child
  walking across the image, ideally several times in both directions.
- The whole body should be visible, and the child should be the only person
  walking through the central part of the image.
- 50 frames per second is ideal. Other frame rates work; below 40 fps the
  quality report adds a warning.

## Other commands

```bash
# Landmarks only (steps 1-2), e.g. to run the slow part on a GPU machine
jia-gait landmarks video.mp4 -o child01_landmarks.csv

# Steps 3-5 from a landmarks file (fast, no GPU needed)
jia-gait analyse child01_landmarks.csv -o results/child01 --fps 50

# Use your own heel-strike labels instead of the model
jia-gait analyse child01_landmarks.csv -o results/child01 --fps 50 --events heel_strikes.csv

# heel_strikes.csv: no header, frame numbers in five columns
#   <unused>, toe-off L->R, heel strike L->R, toe-off R->L, heel strike R->L
# (one event per cell; toe-offs are not used and may be empty)

# Name shown in the plot titles (default: file name)
jia-gait run video.mp4 -o results/child01 --name "Child 01, visit 2"
```

## What the steps do

1. **Person detection.** A Faster R-CNN detector finds the child in every
   frame, and the frame is cropped around them. Frames where the child is at
   the edge of the image or too small are skipped.
2. **Body landmarks.** ViTPose-H finds shoulders, hips, knees, ankles, heels
   and toes on both sides in each crop.
3. **Joint angles.** Hip, knee and ankle angles are computed per frame. Short
   gaps in tracking (up to 0.1 s) are filled; longer gaps split the video into
   separate walking passes. Single-frame landmark errors that make an angle
   jump by more than 20 degrees are replaced by the local median.
4. **Heel strikes.** A trained classifier (`weights/heel_on_xgb.json`) finds
   the frames where a heel touches the ground and the walking direction.
5. **Gait cycles and plots.** The angles between two heel strikes of the same
   foot form one gait cycle. Each cycle is resampled to 101 points (0-100 % of
   the cycle) and smoothed; cycles that span a tracking gap or are implausibly
   short or long are dropped. The plots show the mean +- SD over the remaining
   cycles.

## Configuration

All settings have defaults; to change them, copy `configs/default.yaml`, edit
it and pass it with `-c`:

```bash
jia-gait -c my_config.yaml run video.mp4 -o results/child01
```

The comments in `configs/default.yaml` explain each setting. Commonly
changed ones:

- `detection.min_height`: minimum height of the child in pixels (lower it for
  low-resolution videos or a distant camera).
- `detection.device`, `landmarks.device`: `auto`, `cuda`, `mps` or `cpu`.
- `cycles.min_duration_s`, `cycles.max_duration_s`: plausible stride duration.
- `events.model`: a different heel-strike model file.

## Speed

Landmark detection is the slow part. Devices are chosen automatically:
ViTPose runs on an NVIDIA GPU (CUDA) if available, else on Apple's GPU (MPS),
else on the CPU; Faster R-CNN runs on CUDA or the CPU (it is slower on MPS).
On an M1 MacBook a 400-frame video takes about 12 minutes; with an NVIDIA GPU
it is much faster.

## Licences

The code depends only on permissively licensed packages. The ViTPose model
code in `src/jia_gait/landmarks/vitpose_model/` is adapted from
[easy_ViTPose](https://github.com/JunkyByte/easy_ViTPose) (Apache-2.0, see its
`LICENSE` and `NOTICE`).

## Development

```bash
uv pip install -p .venv -e '.[dev,video,models]'
pytest
ruff check src tests
```
