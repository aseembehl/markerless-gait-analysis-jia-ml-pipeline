import numpy as np
import pytest

from jia_gait.gait import EventDetector, ModelSettings, PostprocessConfig
from jia_gait.gait.features import context_features, make_feature_fn


def test_context_features_clip_at_segment_boundaries():
    a = np.arange(6, dtype=float)[:, None]  # one feature: 0..5
    seg = np.array([0, 0, 0, 1, 1, 1])
    f = context_features(a, seg, offsets=(-1, 1))
    # columns: value, change to previous, change to next (within the segment)
    np.testing.assert_array_equal(f[:, 0], a[:, 0])
    np.testing.assert_array_equal(f[:, 1], [0, -1, -1, 0, -1, -1])
    np.testing.assert_array_equal(f[:, 2], [1, 1, 0, 1, 1, 0])


def test_make_feature_fn():
    a = np.ones((10, 6))
    assert make_feature_fn("angles")(a).shape == (10, 6)
    assert make_feature_fn("context", [-1, 1])(a).shape == (10, 18)
    with pytest.raises(ValueError):
        make_feature_fn("nope")


def test_settings_file_round_trip_and_precedence(tmp_path):
    import shutil
    from pathlib import Path

    model_path = tmp_path / "heel_on_xgb.json"
    shutil.copy(Path(__file__).parent / "data" / "tiny_xgb_18f.json", model_path)

    settings = ModelSettings(
        feature_set="context",
        context_offsets=[-1, 1],
        model_fps=50,
        postprocess=PostprocessConfig(threshold=0.3, min_run=3),
        info={"trained_on": "test"},
    )
    assert settings.save(model_path).name == "heel_on_xgb.meta.json"

    det = EventDetector.load(model_path, fallback=ModelSettings(model_fps=25))
    assert det.settings.feature_set == "context"
    assert det.settings.model_fps == 50  # settings file wins over the fallback
    assert det.cfg.threshold == 0.3 and det.cfg.min_run == 3
    assert det.predict_proba(det.feature_fn(np.zeros((7, 6)))).shape == (7, 3)

    (tmp_path / "heel_on_xgb.meta.json").unlink()
    assert (
        EventDetector.load(model_path, fallback=ModelSettings(model_fps=25)).settings.model_fps
        == 25
    )
