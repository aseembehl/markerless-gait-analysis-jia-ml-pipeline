import numpy as np
import pandas as pd
import pytest

from jia_gait.gait import CycleConfig, cycle_bounds, extract_gait_cycles
from jia_gait.io.csv_formats import FEATURE_COLUMNS, GaitEvents
from jia_gait.kinematics import JointAngles

from .helpers import angles_with_events, periodic_events
from .legacy_reference import legacy_cycle_to_grid, legacy_lowess_wrap, legacy_preprocess


def _legacy_df(feats, ev):
    df = pd.DataFrame(feats, columns=list(FEATURE_COLUMNS))
    for col, frames in (
        ("Toe off left to right", ev.toe_off_ltr),
        ("Toe off right to left", ev.toe_off_rtl),
        ("Heel on left to right", ev.heel_on_ltr),
        ("Heel on right to left", ev.heel_on_rtl),
    ):
        df[col] = np.nan
        df.loc[np.asarray(frames, dtype=int), col] = 100
    return df


EVENT_SETS = [
    periodic_events(n=600, period=40, seed=0),
    periodic_events(n=500, period=33, seed=1, turn_at=200),
    # interleaved directions and a lone event: exercises the removal rules
    GaitEvents(heel_on_ltr=np.array([5, 40, 80, 200, 240]), heel_on_rtl=np.array([120, 160, 300])),
]


@pytest.mark.parametrize("ev", EVENT_SETS)
def test_cycle_bounds_match_original(ev):
    n = 620
    feats = angles_with_events(ev, n)
    pre = legacy_preprocess(_legacy_df(feats, ev))
    expected = {"ltr": set(), "rtl": set()}
    for split, g in pre.groupby("f_split"):
        expected["ltr" if split.startswith("l") else "rtl"].add((g["time"].min(), g["time"].max()))
    got = cycle_bounds(ev)
    assert {d: set(v) for d, v in got.items()} == expected


@pytest.mark.parametrize("smoothing", [None, "lowess"])
def test_normalised_cycles_match_original(smoothing):
    ev = EVENT_SETS[0]
    n = 600
    feats = angles_with_events(ev, n, noise=2.0)
    pre = legacy_preprocess(_legacy_df(feats, ev))
    cycles = extract_gait_cycles(
        JointAngles.from_features(feats), ev, CycleConfig(smoothing=smoothing)
    )

    for d, prefix, col in (("ltr", "l", "Knee Angle 2"), ("rtl", "r", "Knee Angle 1")):
        dc = cycles[d]
        splits = sorted(
            (s for s in pre["f_split"].unique() if s.startswith(prefix)),
            key=lambda s: pre.loc[pre["f_split"] == s, "time"].min(),
        )
        assert len(splits) == dc.n_cycles > 0
        for k, s in enumerate(splits):
            g = pre[pre["f_split"] == s]
            ref = legacy_cycle_to_grid(g["relative_time"].to_numpy(float), g[col].to_numpy(float))
            if smoothing == "lowess":
                ref = legacy_lowess_wrap(ref)
            np.testing.assert_allclose(dc.angles["knee"][k], ref, atol=1e-8)


def test_toe_off_fraction_and_shapes():
    ev = periodic_events(n=600, period=40, seed=0)
    cycles = extract_gait_cycles(JointAngles.from_features(angles_with_events(ev, 600)), ev)
    for d in ("ltr", "rtl"):
        dc = cycles[d]
        assert dc.angles["hip"].shape == (dc.n_cycles, 101)
        np.testing.assert_allclose(dc.toe_off, 0.6, atol=0.06)


def test_simultaneous_opposite_events_rejected():
    with pytest.raises(ValueError, match="same frame"):
        cycle_bounds(GaitEvents(heel_on_ltr=np.array([10, 50]), heel_on_rtl=np.array([50, 90])))
