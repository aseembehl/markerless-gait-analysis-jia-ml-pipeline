import numpy as np
import pytest

from jia_gait import cli, pipeline
from jia_gait.config import PipelineConfig
from jia_gait.gait.cycles import filter_bounds
from jia_gait.io.csv_formats import GaitEvents, write_point_predictions


def _cfg():
    cfg = PipelineConfig()
    cfg.cycles.smoothing = None
    return cfg


def test_no_landmarks_gives_clear_error():
    with pytest.raises(pipeline.NoLandmarksError, match="no person was detected"):
        pipeline.analyse_landmarks(np.zeros((0, 12, 2)), _cfg(), imgids=np.zeros(0, int))


def test_cli_exits_with_message_when_no_landmarks(tmp_path):
    pts = tmp_path / "empty.csv"
    write_point_predictions(pts, np.zeros(0, int), np.zeros((0, 12, 2)))
    ev = tmp_path / "ev.csv"
    ev.write_text(",,3,,\n")
    with pytest.raises(SystemExit, match="no person was detected"):
        cli.main(["analyse", str(pts), "-o", str(tmp_path / "out"), "--events", str(ev)])


@pytest.mark.parametrize("n", [1, 10])
def test_short_recordings_produce_no_cycles_and_no_plots(tmp_path, n):
    lm = np.random.default_rng(0).uniform(0, 500, (n, 12, 2))
    res = pipeline.analyse_landmarks(
        lm,
        _cfg(),
        imgids=np.arange(n),
        fps=50,
        event_frames=GaitEvents(heel_on_ltr=np.array([0, n - 1])),
    )
    assert res.cycles.n_cycles == 0  # < 0.4 s is not a stride
    out = pipeline.write_outputs(res, tmp_path / "out")
    assert not any(k.startswith("plot") for k in out)
    assert (tmp_path / "out" / "cycles.npz").exists()


def test_cli_warns_when_no_cycles(tmp_path, capsys):
    pts = tmp_path / "short.csv"
    write_point_predictions(
        pts, np.arange(20), np.random.default_rng(1).uniform(0, 500, (20, 12, 2))
    )
    ev = tmp_path / "ev.csv"
    ev.write_text(",,3,,\n,,15,,\n")
    cli.main(["analyse", str(pts), "-o", str(tmp_path / "out"), "--events", str(ev)])
    assert "no complete gait cycle" in capsys.readouterr().err


def test_duration_range_in_seconds():
    bounds = {"ltr": [(0, 10), (10, 60), (60, 300)], "rtl": []}
    out = filter_bounds(bounds, None, None, None, fps=50, duration_range_s=(0.4, 3.0))
    assert out["ltr"] == [(10, 60)]  # 0.2 s too short, 4.8 s too long
