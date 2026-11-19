import numpy as np

from jia_gait.gait import EventDetector, ModelSettings
from jia_gait.gait.cycles import filter_bounds
from jia_gait.sequence import fill_gaps, max_gap_frames, resample, to_resampled_index


def _lm(ids):
    ids = np.asarray(ids, float)
    lm = np.zeros((len(ids), 12, 2))
    lm[..., 0] = ids[:, None]  # x = frame id, so interpolation is easy to check
    lm[..., 1] = 2 * ids[:, None]
    return lm


def test_fill_short_gaps_split_long_ones():
    ids = [0, 1, 3, 4, 10, 11]
    seq = fill_gaps(ids, _lm(ids), max_gap=2)
    assert seq.imgids.tolist() == [0, 1, 2, 3, 4, 10, 11]
    assert seq.filled.tolist() == [False, False, True, False, False, False, False]
    assert seq.segment.tolist() == [0, 0, 0, 0, 0, 1, 1]
    np.testing.assert_allclose(seq.landmarks[2, :, 0], 2.0)
    np.testing.assert_allclose(seq.landmarks[2, :, 1], 4.0)


def test_max_gap_frames_scales_with_fps():
    assert max_gap_frames(0.1, 50) == 5
    assert max_gap_frames(0.1, 24) == 2


def test_resample_is_identity_at_model_rate():
    ids = np.arange(20)
    seq = fill_gaps(ids, _lm(ids), 0)
    vals = np.random.default_rng(0).normal(size=(20, 6))
    rs = resample(vals, seq, 50, 50)
    np.testing.assert_array_equal(rs.values, vals)
    np.testing.assert_array_equal(rs.src_rows, np.arange(20))


def test_resample_doubles_rate_per_segment():
    ids = np.r_[np.arange(0, 25), np.arange(100, 125)]
    seq = fill_gaps(ids, _lm(ids), 2)
    vals = ids[:, None].astype(float)
    rs = resample(vals, seq, 25, 50)
    assert len(rs.values) == 2 * 49  # 24 intervals per segment at twice the rate
    seg0 = rs.segment == 0
    np.testing.assert_allclose(rs.values[seg0, 0], np.arange(0, 24.01, 0.5))
    assert set(rs.src_rows[rs.segment == 1]) <= set(range(25, 50))
    np.testing.assert_array_equal(to_resampled_index([0, 10], rs), [0, 20])


class _PulseModel:
    """Class 0 where feature 0 > 0.5, class 1 where feature 1 > 0.5, else 2."""

    def predict_proba(self, X):
        p = np.full((len(X), 3), [0.05, 0.05, 0.9])
        p[X[:, 0] > 0.5] = [0.9, 0.05, 0.05]
        p[X[:, 1] > 0.5] = [0.05, 0.9, 0.05]
        return p


def test_detect_maps_events_back_to_source_frames_at_other_fps():
    fps, n = 25, 300
    truth = np.array([30, 55, 80, 105, 130])
    feats = np.zeros((n, 6))
    for t in truth:
        feats[t - 1 : t + 2, 0] = 1.0  # 3 frames at 25 fps = 6 frames at 50 fps
    ids = np.arange(n)
    seq = fill_gaps(ids, _lm(ids), 0)
    ev = EventDetector(_PulseModel(), ModelSettings(model_fps=50)).detect(feats, seq, fps)
    assert len(ev.heel_on_ltr) == len(truth)
    assert np.abs(seq.imgids[ev.heel_on_ltr] - truth).max() <= 1


def test_filter_bounds_drops_cycles_across_gaps_and_merged_strides():
    imgids = np.r_[np.arange(0, 100), np.arange(300, 400)]
    segment = np.r_[np.zeros(100, int), np.ones(100, int)]
    bounds = {"ltr": [(0, 25), (25, 50), (50, 100), (75, 120)], "rtl": [(110, 135), (135, 160)]}
    out = filter_bounds(bounds, imgids, segment, max_duration_ratio=1.5)
    # (75, 120) crosses the gap; (50, 100) is ~2 strides long
    assert out == {"ltr": [(0, 25), (25, 50)], "rtl": [(110, 135), (135, 160)]}
