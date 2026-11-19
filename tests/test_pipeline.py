from pathlib import Path

import numpy as np
import pytest

from jia_gait import cli
from jia_gait.config import PipelineConfig
from jia_gait.io.csv_formats import read_event_csv, write_point_predictions

from .helpers import periodic_events, random_landmarks

ROOT = Path(__file__).resolve().parents[1]


def test_default_config_loads():
    cfg = PipelineConfig.from_yaml(ROOT / "configs" / "default.yaml")
    assert cfg.cycles.n_points == 101
    assert cfg.events.min_run == 4
    assert cfg.detection.min_height == 200


def test_unknown_config_key_rejected():
    with pytest.raises(ValueError, match="Unknown keys"):
        PipelineConfig.from_dict({"cycles": {"n_point": 50}})


def _write_trial(tmp_path, n=500):
    imgids = np.arange(1000, 1000 + n)
    points = tmp_path / "K999.csv"
    write_point_predictions(points, imgids, random_landmarks(n))
    ev = periodic_events(n=n, period=40)
    events = tmp_path / "K999_events.csv"
    # layout: <unused>, toe-off L->R, heel-on L->R, toe-off R->L, heel-on R->L (image ids)
    cols = [ev.toe_off_ltr, ev.heel_on_ltr, ev.toe_off_rtl, ev.heel_on_rtl]
    with open(events, "w") as fh:
        fh.writelines(
            "," + ",".join(str(imgids[c[r]]) if r < len(c) else "" for c in cols) + "\n"
            for r in range(max(map(len, cols)))
        )
    return points, events, imgids, ev


def test_event_csv_maps_image_ids_to_rows(tmp_path):
    _, events, imgids, ev = _write_trial(tmp_path)
    got = read_event_csv(events, imgids)
    np.testing.assert_array_equal(got.heel_on_ltr, ev.heel_on_ltr)
    np.testing.assert_array_equal(got.toe_off_rtl, ev.toe_off_rtl)


def test_cli_analyse_writes_output_folder(tmp_path, capsys):
    points, events, _, _ = _write_trial(tmp_path)
    out = tmp_path / "K999_out"
    cli.main(["analyse", str(points), "-o", str(out), "--events", str(events)])
    res = np.load(out / "cycles.npz")
    assert res["ltr_knee"].shape[1] == 101
    assert res["ltr_knee"].shape[0] >= 3
    assert (out / "landmarks.csv").exists() and (out / "angles.csv").exists()
    plots = sorted(p.name for p in (out / "plots").glob("*.png"))
    assert plots == sorted(f"{j}_{d}.png" for j in ("hip", "knee", "ankle") for d in ("ltr", "rtl"))
    assert "left->right" in capsys.readouterr().out
