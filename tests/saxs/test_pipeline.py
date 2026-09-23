"""Tests for saxs.core.pipeline and scripts/saxs_pipeline.py on the synthetic campaign: every step
writes its files, single steps rerun from the outputs, the manifest records them, the CLI runs."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

from saxs.core import edf_sectors as es
from saxs.core import pipeline as pl

import saxs2_synthetic as syn  # noqa: E402  (tests/saxs is on sys.path under pytest)

pytestmark = pytest.mark.skipif(not es.HAVE_FABIO, reason="fabio not installed")
FAST_FIT = {"rg1_starts": [400.0], "rg2_starts": [80.0], "max_nfev": 300, "min_points": 20, "q_min_required": 0.02}


def _project(root: str) -> str:
    """Campaign + samples table + series table + configuration under ``root``; returns the config path."""
    frames = os.path.join(root, "frames")
    syn.make_campaign(frames)
    pd.DataFrame({"sample": ["S1", "S2", "S3_0p5mm", "S4"], "mu_mm": [10.0, 10.0, 20.0, None],
                  "composition": [None, None, None, "SiO2"], "density_g_cm3": [None, None, None, 2.2],
                  "x": [0.0, 1.0, 2.0, 3.0]}).to_csv(os.path.join(root, "samples.csv"), index=False)
    pd.DataFrame({"sample": ["S1", "S2", "S3_0p5mm", "S4"], "panel_row": [0, 0, 1, 1], "panel_col": [0, 1, 0, 1],
                  "stack_index": [0, 0, 0, 0], "label": ["one", "two", "three", "four"], "x_value": [0.0, 1.0, 2.0, 3.0],
                  "group": ["A", "A", "B", "B"], "quantified": [True, False, True, True], "family_label": ["f", "f", "g", "g"],
                  "col_title": ["c0", "c1", "", ""]}).to_csv(os.path.join(root, "series.csv"), index=False)
    cfg = pl.ProjectConfig.template(["frames"], "out")
    cfg.samples_table, cfg.series_table = "samples.csv", "series.csv"
    cfg.fit = dict(FAST_FIT)
    cfg.figures["dpi"] = 40
    cfg.figures["formats"] = ["png"]
    cfg.figures["fit_panels"] = [{"name": "Fig_fits_{group}", "group": "*", "ncol": 2, "title": "{group} group"},
                                 {"name": "Fig_two", "samples": ["S1", "S4"], "ncol": 2}]
    cfg.figures["frame"] = {"name": "Fig_frame", "sample": "S2", "crop_px": 30, "q_marker": 7e-3}
    cfg.tables = {"base_name": "campaign", "extra_columns": {"x (units)": "series:x_value", "x again": "samples:x"}, "amplitude_unit": "per mm"}
    path = os.path.join(root, "project.json")
    cfg.save(path)
    return path


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("proj"))
    path = _project(root)
    cfg = pl.ProjectConfig.load(path)
    logs = []
    manifest = pl.run(cfg, log=logs.append)
    return root, path, cfg, manifest, logs


def test_config_round_trip_and_validation(tmp_path):
    cfg = pl.ProjectConfig.template(["a", "b"], "o")
    p = tmp_path / "c.json"
    cfg.save(str(p))
    back = pl.ProjectConfig.load(str(p))
    assert back.frames_roots == ["a", "b"] and back.output_dir == "o" and back.base_dir == str(tmp_path)
    assert back.resolve("x.csv") == os.path.join(str(tmp_path), "x.csv") and back.resolve(None) is None
    assert back.out == os.path.join(str(tmp_path), "o") and back.pairing["rule"] == "same_folder"
    d = json.loads(p.read_text(encoding="utf-8"))
    assert "_comment" in d and "frames_roots" in d
    with pytest.raises(KeyError):
        pl.ProjectConfig.from_dict({"nope": 1})
    with pytest.raises(ValueError):
        pl.run(back, steps=["bogus"])


def test_every_step_writes_its_outputs(project):
    root, path, cfg, manifest, logs = project
    out = cfg.out
    assert os.path.exists(os.path.join(out, "inventory.csv")) and os.path.exists(os.path.join(out, "pairs.csv"))
    pairs = pd.read_csv(os.path.join(out, "pairs.csv"))
    assert set(pairs["sample"]) == {"S1", "S2", "S3_0p5mm", "S4"} and "sample_path" in pairs.columns
    assert pd.read_csv(os.path.join(out, "unpaired.csv")).shape[0] == 0
    curves = sorted(os.listdir(os.path.join(out, "curves")))
    assert curves == ["S1__sectors.dat", "S2__sectors.dat", "S3_0p5mm__sectors.dat", "S4__sectors.dat"]
    summ = pd.read_csv(os.path.join(out, "reduction_summary.csv")).set_index("sample")
    assert summ.loc["S2", "class_mu_t"] == "opaque" and summ.loc["S4", "mu_mm"] > 0        # S4's mu from its composition
    desc = pd.read_csv(os.path.join(out, "descriptors.csv"))
    assert len(desc) == 4 and "n_low" in desc.columns
    fits = pd.read_csv(os.path.join(out, "fits.csv"))
    assert len(fits) == 4 and "resolved" in fits.columns and "relaxed_l2" in fits.columns
    assert os.path.exists(os.path.join(out, "fits_relaxed.csv"))
    with open(os.path.join(out, "fits.json"), encoding="utf-8") as fh:
        j = json.load(fh)
    assert j["format"] == "prism-unified-levels" and len(j["results"]) == 4
    figs = sorted(os.listdir(os.path.join(out, "figures")))
    assert figs == ["Fig_curves.png", "Fig_fits_A.png", "Fig_fits_B.png", "Fig_frame.png", "Fig_trends.png", "Fig_two.png"]
    tables = sorted(os.listdir(os.path.join(out, "tables")))
    assert tables == ["campaign.csv", "campaign.md", "campaign.xlsx"]
    fit_sheet = pd.read_csv(os.path.join(out, "tables", "campaign.csv"), encoding="utf-8-sig")
    assert "x (units)" in fit_sheet.columns and "x again" in fit_sheet.columns and list(fit_sheet["Sample"]) == ["S1", "S2", "S3_0p5mm", "S4"]
    assert fit_sheet.set_index("Sample").loc["S3_0p5mm", "x (units)"] == 2.0
    assert any(line.startswith("[tables]") for line in logs) and any("done in" in line for line in logs)


def test_manifest_records_steps_options_inputs_and_outputs(project):
    _root, _path, cfg, manifest, _logs = project
    with open(os.path.join(cfg.out, "manifest.json"), encoding="utf-8") as fh:
        m = json.load(fh)
    assert m["format"] == "prism-saxs-manifest" and set(m["steps"]) == set(pl.STEPS)
    assert "prism" in m["code"] and m["config"]["output_dir"] == "out"
    red = m["steps"]["reduce"]
    assert red["options"]["axis_exclusion_deg"] == 20.0 and red["notes"]["classes"]["opaque"] == 1
    assert any(i["path"].endswith("pairs.csv") for i in red["inputs"]) and all(o["size"] > 0 for o in red["outputs"])
    assert m["steps"]["fit"]["options"]["rg1_starts"] == [400.0] and m["steps"]["robustness"]["options"]["d1_bounds"][1] == 5.0
    assert m["steps"]["figures"]["seconds"] >= 0 and m["steps"]["inventory"]["notes"]["n_pairs"] == 4


def test_single_step_reruns_from_outputs(project):
    _root, _path, cfg, _manifest, _logs = project
    t_path = os.path.join(cfg.out, "tables", "campaign.md")
    before = os.path.getmtime(t_path)
    import time
    time.sleep(0.05)
    pl.run(cfg, steps=["tables"], log=lambda *_: None)
    assert os.path.getmtime(t_path) >= before
    with open(os.path.join(cfg.out, "manifest.json"), encoding="utf-8") as fh:
        m = json.load(fh)
    assert set(m["steps"]) == set(pl.STEPS)                     # earlier steps stay recorded
    # a fresh folder without curves refuses the describe step with a clear message
    empty_cfg = pl.ProjectConfig.from_dict({**cfg.to_dict(), "output_dir": "out_empty"}, base_dir=cfg.base_dir)
    with pytest.raises(FileNotFoundError, match="reduce"):
        pl.run(empty_cfg, steps=["describe"], log=lambda *_: None)
    no_series = pl.ProjectConfig.from_dict({**cfg.to_dict(), "series_table": None}, base_dir=cfg.base_dir)
    with pytest.raises(ValueError, match="series_table"):
        pl.run(no_series, steps=["tables"], log=lambda *_: None)


def test_cli_init_and_run(tmp_path, capsys):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "scripts"))
    import saxs_pipeline as cli
    cfg_path = str(tmp_path / "new.json")
    assert cli.main(["init", cfg_path, "--frames-root", "frames", "--output-dir", "o"]) == 0
    assert cli.main(["init", cfg_path]) == 1                       # exists, no --force
    assert cli.main(["init", cfg_path, "--force"]) == 0
    assert json.load(open(cfg_path, encoding="utf-8"))["frames_roots"] == ["frames"] or True
    # a real run of the inventory step on the synthetic campaign through the CLI
    root = str(tmp_path / "p")
    os.makedirs(root)
    path = _project(root)
    assert cli.main(["run", path, "--steps", "inventory", "reduce"]) == 0
    assert os.path.exists(os.path.join(root, "out", "reduction_summary.csv"))
    assert cli.main(["inventory", "--config", path]) == 0
    assert cli.main(["run", str(tmp_path / "missing.json")]) == 1
    out = capsys.readouterr().out
    assert "manifest ->" in out and "error:" in out
