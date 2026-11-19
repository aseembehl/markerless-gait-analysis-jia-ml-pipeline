"""Heel strikes cut off by the start or end of a walking pass (``edge_min_run``)."""

import json

import numpy as np

from jia_gait.gait.events import HEEL_ON_LTR, ModelSettings, PostprocessConfig, postprocess

STRIDE, RUN = 50, 7  # frames between strikes, frames above threshold around each strike


def _proba(n, strikes, cls=HEEL_ON_LTR):
    p = np.tile([0.0, 0.0, 1.0], (n, 1))
    for t in strikes:
        lo, hi = max(0, t - RUN // 2), min(n, t + RUN // 2 + 1)
        p[lo:hi] = 0.0
        p[lo:hi, cls] = 0.95
        p[lo:hi, 2] = 0.05
    return p


def test_strike_on_last_frames_of_a_pass_is_recovered():
    n = 160
    p = _proba(n, [20, 70, 120, 159])  # last strike: only 4 of its 7 frames visible
    off = postprocess(p, PostprocessConfig(edge_min_run=0)).heel_on_ltr
    on = postprocess(p).heel_on_ltr
    assert list(off) == [20, 70, 120]
    assert len(on) == 4 and abs(on[-1] - 159) <= 1


def test_strike_on_first_frames_of_a_pass_is_recovered():
    p = _proba(160, [1, 51, 101, 151])
    on = postprocess(p).heel_on_ltr
    assert len(on) == 4 and abs(on[0] - 1) <= 1


def test_each_pass_gets_its_own_edge_strike():
    p = np.r_[_proba(100, [20, 70, 99]), _proba(100, [0, 50])]
    seg = np.r_[np.zeros(100, int), np.ones(100, int)]
    ev = postprocess(p, segment=seg).heel_on_ltr
    assert len(ev) == 5
    assert np.abs(ev[seg[ev] == 0] - 99).min() <= 2  # cut-off strike at the end of pass 0
    assert np.abs(ev[seg[ev] == 1] - 100).min() <= 2  # and at the start of pass 1


def test_single_frame_blip_at_an_edge_is_ignored():
    p = _proba(160, [20, 70, 120])
    p[-1] = [0.9, 0.05, 0.05]
    assert list(postprocess(p).heel_on_ltr) == [20, 70, 120]


def test_complete_run_near_the_edge_is_not_duplicated():
    p = _proba(160, [20, 70, 120, 154])  # complete run ending 2 frames before the end
    ev = postprocess(p).heel_on_ltr
    assert len(ev) == len(np.unique(ev)) and np.sum(ev > 140) == 1


def test_old_settings_file_without_the_field_enables_the_rule(tmp_path):
    model = tmp_path / "m.json"
    meta = {"format": 1, "feature_set": "angles", "postprocess": {"threshold": 0.5}}
    ModelSettings.path_for(model).write_text(json.dumps(meta))
    assert ModelSettings.load(model).postprocess.edge_min_run == 2
