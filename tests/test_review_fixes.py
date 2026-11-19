"""Regression tests for issues found in code review."""

import numpy as np
import pytest

from jia_gait import cli, pipeline
from jia_gait.config import PipelineConfig
from jia_gait.gait import EventDetector, ModelSettings
from jia_gait.gait.cycles import cycle_bounds, filter_bounds
from jia_gait.gait.events import PostprocessConfig
from jia_gait.io.csv_formats import (
    GaitEvents,
    frames_to_rows,
    read_event_frames,
    write_point_predictions,
)
from jia_gait.sequence import fill_gaps


class _RowModel:
    """Predicts class 0 (heel-on L->R) on the rows given in ``hits``."""

    def __init__(self, hits):
        self.hits = set(hits)

    def predict_proba(self, X):
        p = np.tile([0.05, 0.05, 0.9], (len(X), 1))
        for r in self.hits:
            if r < len(X):
                p[r] = [0.9, 0.05, 0.05]
        return p


def _seq(ids):
    ids = np.asarray(ids)
    return fill_gaps(ids, np.zeros((len(ids), 12, 2)), max_gap=5)


def test_no_event_invented_from_runs_on_both_sides_of_a_gap():
    seq = _seq(np.r_[np.arange(0, 50), np.arange(200, 250)])
    # last 3 rows of pass 0 and first 3 rows of pass 1: neither is a run of 4 on its own
    hits = [47, 48, 49, 50, 51, 52]
    notebook = ModelSettings(model_fps=50, postprocess=PostprocessConfig(edge_min_run=0))
    ev = EventDetector(_RowModel(hits), notebook).detect(np.zeros((len(seq), 6)), seq, fps=50)
    assert len(ev.heel_on_ltr) == 0
    # the edge rule treats them as two cut-off runs, one strike per pass, none in the gap
    ev = EventDetector(_RowModel(hits), ModelSettings(model_fps=50)).detect(
        np.zeros((len(seq), 6)), seq, fps=50
    )
    assert set(seq.segment[ev.heel_on_ltr]) == {0, 1} and len(ev.heel_on_ltr) == 2


def test_outlier_check_does_not_measure_intervals_across_a_gap():
    ids = np.r_[np.arange(0, 130), np.arange(150, 700)]  # 0.4 s occlusion
    seq = _seq(ids)
    row_of = {int(f): r for r, f in enumerate(seq.imgids)}
    truth = np.arange(20, 700, 50)
    truth = truth[np.isin(truth, seq.imgids)]
    hits = [row_of[int(f)] + o for f in truth for o in (-3, -2, -1, 0, 1, 2) if int(f) in row_of]
    det = EventDetector(_RowModel(hits), ModelSettings(model_fps=50))
    ev = det.detect(np.zeros((len(seq), 6)), seq, fps=50)
    found = seq.imgids[ev.heel_on_ltr]
    for f in (120, 170):  # previously deleted as an "outlier" interval of 30 rows
        assert np.abs(found - f).min() <= 1


def test_cli_without_model_fails_early_with_message(tmp_path):
    pts = tmp_path / "p.csv"
    write_point_predictions(pts, np.arange(5), np.zeros((5, 12, 2)))
    no_model = tmp_path / "no_model.yaml"
    no_model.write_text("events:\n  model: null\n")
    with pytest.raises(SystemExit, match="no heel-strike model configured"):
        cli.main(["-c", str(no_model), "analyse", str(pts), "-o", str(tmp_path / "o")])
    with pytest.raises(SystemExit, match="model not found"):
        cli.main(
            ["analyse", str(pts), "-o", str(tmp_path / "o"), "--model", str(tmp_path / "x.json")]
        )


def test_relative_paths_in_config_resolve_against_config_file(tmp_path):
    (tmp_path / "cfgdir").mkdir()
    cfg_file = tmp_path / "cfgdir" / "c.yaml"
    cfg_file.write_text("events:\n  model: ../weights/m.json\n")
    cfg = PipelineConfig.from_yaml(cfg_file)
    assert cfg.events.model == str((tmp_path / "weights" / "m.json").resolve())


def test_duplicate_labels_do_not_create_zero_length_cycles(tmp_path):
    ev_csv = tmp_path / "ev.csv"
    ev_csv.write_text(",,10,,\n,,10,,\n,,60,,\n")
    ev = read_event_frames(ev_csv)
    assert ev.heel_on_ltr.tolist() == [10, 60]
    rows = frames_to_rows(GaitEvents(heel_on_ltr=np.array([10, 10, 60])), np.arange(100))
    assert rows.heel_on_ltr.tolist() == [10, 60]
    assert filter_bounds({"ltr": [(5, 5), (5, 30)], "rtl": []}, None, None, None) == {
        "ltr": [(5, 30)],
        "rtl": [],
    }


def test_conflicting_labels_give_clear_cli_error(tmp_path):
    pts = tmp_path / "p.csv"
    write_point_predictions(
        pts, np.arange(100), np.random.default_rng(0).uniform(0, 500, (100, 12, 2))
    )
    ev = tmp_path / "ev.csv"
    ev.write_text(",,10,,10\n,,60,,\n")
    with pytest.raises(SystemExit, match="same frame"):
        cli.main(["analyse", str(pts), "-o", str(tmp_path / "o"), "--events", str(ev)])
    with pytest.raises(ValueError):
        cycle_bounds(GaitEvents(heel_on_ltr=np.array([10]), heel_on_rtl=np.array([10])))


def test_rerun_removes_stale_plots(tmp_path):
    cfg = PipelineConfig()
    cfg.cycles.smoothing = None
    lm = np.random.default_rng(0).uniform(0, 500, (200, 12, 2))
    good = GaitEvents(heel_on_ltr=np.array([10, 60, 110]), heel_on_rtl=np.array([]))
    res = pipeline.analyse_landmarks(lm, cfg, fps=50, event_frames=good)
    pipeline.write_outputs(res, tmp_path / "o")
    assert len(list((tmp_path / "o" / "plots").glob("*.png"))) == 3
    res = pipeline.analyse_landmarks(lm, cfg, fps=50, event_frames=GaitEvents())
    pipeline.write_outputs(res, tmp_path / "o")
    assert list((tmp_path / "o" / "plots").glob("*.png")) == []


def test_bundled_model_is_used_without_a_config():
    from jia_gait.config import BUNDLED_MODEL

    assert BUNDLED_MODEL.exists()
    assert PipelineConfig().events.model == str(BUNDLED_MODEL)
    assert pipeline.event_detector_from_config(PipelineConfig()).settings.feature_set == "context"
