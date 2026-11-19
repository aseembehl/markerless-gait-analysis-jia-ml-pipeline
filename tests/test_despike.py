import numpy as np

from jia_gait.config import PipelineConfig
from jia_gait.kinematics import JointAngles, despike, despike_angles


def gait_like(n=500, fps=50.0, stride_s=1.0, amp=30.0):
    """Knee-like curve: 0-60 deg at a fast child's cadence, with a sharp swing peak."""
    t = np.arange(n) / fps
    phase = 2 * np.pi * t / stride_s
    return amp * (1 - np.cos(phase)) + 8 * np.sin(2 * phase)


def test_isolated_spikes_are_replaced():
    x = gait_like()
    y = x.copy()
    y[100] += 120  # one-frame glitch
    y[300:302] -= 60  # two-frame glitch
    out, mask = despike(y)
    assert set(np.flatnonzero(mask)) == {100, 300, 301}
    assert np.abs(out - x).max() < 6


def test_clean_gait_curve_is_untouched():
    for stride_s in (0.5, 0.8, 1.2):  # down to very fast strides
        x = gait_like(stride_s=stride_s)
        out, mask = despike(x)
        assert not mask.any(), stride_s
        np.testing.assert_array_equal(out, x)


def test_segments_are_filtered_separately():
    # a step between two passes must not be mistaken for a spike
    x = np.r_[np.zeros(20), np.full(20, 50.0)]
    seg = np.r_[np.zeros(20, int), np.ones(20, int)]
    _, mask = despike(x, seg)
    assert not mask.any()
    # and a too-short segment is left alone
    _, mask = despike(np.array([0, 90.0, 0]), np.zeros(3, int))
    assert not mask.any()


def test_despike_angles_window_scales_with_fps():
    f = np.column_stack([gait_like(fps=25.0)] * 6)
    f[50, 2] += 80
    a, mask = despike_angles(JointAngles.from_features(f), None, fps=25.0)
    assert mask.sum() == 1 and mask[50, 2]
    assert abs(a.knee_ltr[50] - f[49:52:2, 2].mean()) < 10


def test_config_section():
    cfg = PipelineConfig.from_dict({"angles": {"despike": False, "threshold_deg": 15}})
    assert cfg.angles.despike is False and cfg.angles.threshold_deg == 15
    assert PipelineConfig().angles.despike is True


def test_pipeline_records_and_can_disable_despike(tmp_path):
    from jia_gait import pipeline
    from jia_gait.io.csv_formats import GaitEvents

    from .helpers import random_landmarks

    lm = random_landmarks(n=120)
    ev = GaitEvents(heel_on_ltr=np.array([10, 60, 110]))
    cfg = PipelineConfig()
    res = pipeline.analyse_landmarks(lm, cfg, fps=50, event_frames=ev)
    assert res.despiked.shape == (120, 6)
    pipeline.write_outputs(res, tmp_path / "on")
    assert "despiked" in np.load(tmp_path / "on" / "cycles.npz")

    cfg.angles.despike = False
    res = pipeline.analyse_landmarks(lm, cfg, fps=50, event_frames=ev)
    assert res.despiked is None
    pipeline.write_outputs(res, tmp_path / "off")
    assert "despiked" not in np.load(tmp_path / "off" / "cycles.npz")
