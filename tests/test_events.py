import sys
from pathlib import Path

import numpy as np
import pytest

from jia_gait.gait import EventDetector, postprocess
from jia_gait.gait.events import PostprocessConfig

DATA = Path(__file__).parent / "data"
_PRELOADED = set(sys.modules)

from .helpers import periodic_events, reference_outputs


def _noisy_proba(events, n, seed):
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 1.0, size=(n, 3))
    logits[:, 2] += 2.0
    for cls, frames in ((0, events.heel_on_ltr), (1, events.heel_on_rtl)):
        for f in frames:
            lo, hi = max(0, f - 3), min(n, f + 4)
            logits[lo:hi, cls] += 5.0
    # occasional spurious detections and a dropped event
    for f in rng.integers(0, n, size=6):
        logits[f, rng.integers(0, 2)] += 6.0
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


@pytest.mark.parametrize("seed", range(8))
def test_postprocess_matches_reference(seed):
    n = 700
    ev = periodic_events(n=n, period=38, seed=seed)
    got = postprocess(_noisy_proba(ev, n, seed), PostprocessConfig(edge_min_run=0))
    ref = reference_outputs()
    np.testing.assert_array_equal(got.heel_on_ltr, ref[f"events_{seed}_ltr"])
    np.testing.assert_array_equal(got.heel_on_rtl, ref[f"events_{seed}_rtl"])


def test_detector_loads_model_and_predicts():
    det = EventDetector.load(DATA / "tiny_xgb_6f.json")
    X = np.random.default_rng(0).normal(size=(300, 6))
    assert det.predict_proba(X).shape == (300, 3)
    ev = det.predict(X)
    assert ev.heel_on_ltr.ndim == 1 and ev.heel_on_rtl.ndim == 1


# XGBoost 2.1.4 (6 and 18 features), 3.1.3 and 3.2.0: the per-class intercept
# (base_score) is stored differently in each.
@pytest.mark.parametrize("name", ["6f", "18f", "v31", "v32"])
def test_numpy_tree_model_matches_xgboost(name):
    """Expected probabilities were produced by xgboost.XGBClassifier.predict_proba."""
    from jia_gait.gait.xgb_trees import XGBTreeModel

    ref = np.load(DATA / f"tiny_xgb_{name}_expected.npz")
    proba = XGBTreeModel(DATA / f"tiny_xgb_{name}.json").predict_proba(ref["X"])
    np.testing.assert_allclose(proba, ref["proba"], atol=1e-6)


def test_inference_does_not_import_xgboost():
    import sys

    EventDetector.load(DATA / "tiny_xgb_6f.json").predict(np.zeros((20, 6)))
    assert "xgboost" not in sys.modules or "xgboost" in _PRELOADED
