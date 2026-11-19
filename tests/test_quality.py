import json

import numpy as np

from jia_gait import pipeline
from jia_gait.config import PipelineConfig
from jia_gait.io.csv_formats import GaitEvents

from .helpers import random_landmarks


def _result():
    # frames 0-399, then a long gap (new pass) and frames 600-799
    ids = np.r_[np.arange(400), np.arange(600, 800)]
    lm = random_landmarks(n=len(ids))
    ev = GaitEvents(
        # 3 normal strides, one 2.6 s stride (> 1.5 x median), one across the gap
        heel_on_ltr=np.array([10, 60, 110, 160, 290, 650]),
        heel_on_rtl=np.array([700, 750]),  # a single cycle
    )
    cfg = PipelineConfig()
    cfg.cycles.smoothing = None
    return pipeline.analyse_landmarks(lm, cfg, imgids=ids, fps=50, event_frames=ev)


def test_counts_and_drop_reasons():
    q = _result().quality
    assert q["passes"] == 2 and q["fps"] == 50
    ltr = q["directions"]["ltr"]
    assert ltr["heel_strikes"] == 6 and ltr["cycle_candidates"] == 5
    assert ltr["cycles"] == 3
    reasons = ltr["dropped"]
    assert reasons["crosses a gap in tracking"] == 1
    assert sum(reasons.values()) == 2
    assert any("median stride" in r for r in reasons)
    assert ltr["stride_s"]["median"] == 1.0
    assert q["directions"]["rtl"]["cycles"] == 1


def test_warnings_and_files(tmp_path):
    res = _result()
    assert any("Only 1 gait cycle(s) walking right to left" in w for w in res.quality["warnings"])
    paths = pipeline.write_outputs(res, tmp_path)
    assert json.loads(paths["quality_json"].read_text())["passes"] == 2
    text = paths["quality_txt"].read_text()
    assert "walking passes:        2" in text and "Warnings:" in text


def test_cli_prints_quality_warnings(tmp_path, capsys):
    from jia_gait import cli
    from jia_gait.io.csv_formats import write_point_predictions

    pts = tmp_path / "p.csv"
    write_point_predictions(pts, np.arange(200), random_landmarks(n=200))
    ev = tmp_path / "ev.csv"
    ev.write_text(",,10,,\n,,60,,\n")
    cli.main(["analyse", str(pts), "-o", str(tmp_path / "out"), "--events", str(ev)])
    err = capsys.readouterr().err
    assert "Only 1 gait cycle(s) walking left to right" in err
    assert (tmp_path / "out" / "quality.txt").exists()
