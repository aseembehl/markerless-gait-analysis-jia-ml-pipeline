import numpy as np
import pytest

from jia_gait.gait import CycleConfig, cycle_bounds, extract_gait_cycles
from jia_gait.io.csv_formats import GaitEvents
from jia_gait.kinematics import JointAngles

from .helpers import angles_with_events, periodic_events, reference_outputs

REF = reference_outputs()

EVENT_SETS = [
    periodic_events(n=600, period=40, seed=0),
    periodic_events(n=500, period=33, seed=1, turn_at=200),
    # interleaved directions and a lone event: exercises the removal rules
    GaitEvents(heel_on_ltr=np.array([5, 40, 80, 200, 240]), heel_on_rtl=np.array([120, 160, 300])),
]


@pytest.mark.parametrize("k", range(len(EVENT_SETS)))
def test_cycle_bounds_match_reference(k):
    got = cycle_bounds(EVENT_SETS[k])
    for d in ("ltr", "rtl"):
        assert sorted(got[d]) == [tuple(b) for b in REF[f"bounds_{k}_{d}"].tolist()]


@pytest.mark.parametrize("smoothing", [None, "lowess"])
def test_normalised_cycles_match_reference(smoothing):
    ev = EVENT_SETS[0]
    feats = angles_with_events(ev, 600, noise=2.0)
    cycles = extract_gait_cycles(
        JointAngles.from_features(feats), ev, CycleConfig(smoothing=smoothing)
    )
    for d in ("ltr", "rtl"):
        ref = REF[f"knee_{smoothing or 'none'}_{d}"]
        assert cycles[d].n_cycles == len(ref) > 0
        np.testing.assert_allclose(cycles[d].angles["knee"], ref, atol=1e-8)


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
