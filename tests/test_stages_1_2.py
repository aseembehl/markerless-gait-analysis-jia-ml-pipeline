import numpy as np
import pytest

from jia_gait import pipeline
from jia_gait.config import DetectionConfig, PipelineConfig
from jia_gait.detection import BBox, EdgeFilter, PersonDetector, best_person
from jia_gait.landmarks import LANDMARKS, LandmarkDetector, select_landmarks

from .helpers import reference_outputs

W, H = 1920, 1080


def _random_track(n, seed):
    """Best-box sequence with misses, edge boxes and small boxes."""
    rng = np.random.default_rng(seed)
    boxes = []
    for _ in range(n):
        r = rng.random()
        if r < 0.15:
            boxes.append(None)
            continue
        x1 = rng.uniform(0, W - 100)
        w = rng.uniform(50, 500)
        y1 = rng.uniform(0, 500)
        h = rng.uniform(100, 700)
        boxes.append((x1, y1, min(W, x1 + w), y1 + h))
    return boxes


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("overlaps", [True, False])
def test_edge_filter_matches_reference(seed, overlaps):
    boxes = _random_track(400, seed)
    cfg = (
        DetectionConfig(edge_rule="overlaps", min_height=200)
        if overlaps
        else DetectionConfig(edge_rule="inside", min_height=500)
    )
    f = EdgeFilter(cfg)
    kept = [i for i, b in enumerate(boxes) if f(None if b is None else BBox(*b), W) is not None]
    assert kept == reference_outputs()[f"crop_{seed}_{int(overlaps)}"].tolist()


def test_edge_boxes_only_follow_a_central_box():
    f = EdgeFilter(DetectionConfig())
    edge, centre = BBox(10, 0, 300, 600), BBox(800, 0, 1100, 600)
    assert f(edge, W) is None  # bystander at the edge before the child appears
    assert f(centre, W) is centre
    assert [f(edge, W) is not None for _ in range(6)] == [True] * 4 + [False] * 2


def test_best_person_picks_highest_scoring_person():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 20, 20], [2, 2, 30, 30], [3, 3, 40, 40]], float)
    labels = np.array([1, 3, 1, 1])  # 3 = car
    scores = np.array([0.85, 0.99, 0.95, 0.7])
    b = best_person(boxes, labels, scores, 0.8)
    assert (b.x1, b.score) == (2.0, pytest.approx(0.95))
    assert best_person(boxes, labels, scores, 0.96) is None


def test_select_landmarks_matches_reference_csv_row():
    rng = np.random.default_rng(0)
    kpts = rng.uniform(0, 500, size=(133, 3))  # easy_ViTPose returns (y, x, score)
    row = reference_outputs()["select_row"]
    sel = select_landmarks(kpts)
    assert sel.shape == (len(LANDMARKS), 3)
    np.testing.assert_array_equal(sel[:, :2].reshape(-1), row)


def test_bbox_crop_truncates_like_pil():
    img = np.arange(100 * 100 * 3, dtype=np.uint8).reshape(100, 100, 3)
    crop = BBox(10.9, 20.2, 50.7, 60.99).crop(img)
    assert crop.shape == (40, 40, 3)
    np.testing.assert_array_equal(crop, img[20:60, 10:50])


class FakeDetector(PersonDetector):
    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1

    def detect(self, frame):
        v = int(frame[0, 0, 0])
        return None if v % 5 == 0 else BBox(100, 50, 300, 450)


class FakeLandmarks(LandmarkDetector):
    def predict(self, crop):
        assert crop.shape == (400, 200, 3)
        return None if crop[0, 0, 1] == 7 else np.ones((12, 3))


def test_detect_landmarks_drops_frames_without_detections():
    frames = []
    for i in range(20):
        f = np.zeros((H // 2, W // 2, 3), dtype=np.uint8)
        f[0, 0, 0] = i
        f[50, 100, 1] = 7 if i == 3 else 0  # landmark model fails on frame 3
        frames.append(f)
    cfg = PipelineConfig()
    cfg.images.frame_jpeg_quality = None
    cfg.images.crop_jpeg_quality = None
    det = FakeDetector()
    ids, lm, boxes = pipeline.detect_landmarks(frames, det, FakeLandmarks(), cfg)
    expected = [i for i in range(20) if i % 5 != 0 and i != 3]
    assert ids.tolist() == expected
    assert lm.shape == (len(expected), 12, 3) and boxes.shape == (len(expected), 4)
    assert det.resets == 1


def test_jpeg_roundtrip_changes_pixels_slightly():
    rng = np.random.default_rng(0)
    img = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
    out = pipeline.jpeg_roundtrip(img, 75)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert pipeline.jpeg_roundtrip(img, None) is img


def test_pad_to_aspect_centres_crop_like_easy_vitpose():
    from jia_gait.landmarks.vitpose import pad_to_aspect

    tall = np.ones((400, 100, 3), dtype=np.uint8)  # narrower than 3:4 -> pad width
    p, left, top = pad_to_aspect(tall)
    assert p.shape[:2] == (400, 300) and (left, top) == (100, 0)
    assert p[:, left : left + 100].all() and not p[:, :left].any()
    wide = np.ones((100, 300, 3), dtype=np.uint8)  # wider than 3:4 -> pad height
    p, left, top = pad_to_aspect(wide)
    assert p.shape[:2] == (400, 300) and (left, top) == (0, 150)


def test_vendored_model_builds_without_ultralytics():
    import sys

    pytest.importorskip("torch")
    from jia_gait.landmarks.vitpose_model.config import IMAGE_SIZE, wholebody_model_cfg
    from jia_gait.landmarks.vitpose_model.model import ViTPose

    cfg = wholebody_model_cfg("s")
    m = ViTPose(cfg)
    assert cfg["keypoint_head"]["out_channels"] == 133 and IMAGE_SIZE == (192, 256)
    assert sum(p.numel() for p in m.parameters()) > 1e6
    assert "ultralytics" not in sys.modules


def test_resolve_device_explicit_and_auto():
    pytest.importorskip("torch")
    from jia_gait.device import resolve_device

    assert resolve_device("cpu", allow_mps=True) == "cpu"
    assert resolve_device("auto", allow_mps=False) in ("cuda", "cpu")
    assert resolve_device("auto", allow_mps=True) in ("cuda", "mps", "cpu")
