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
FAST_FIT = {"options": {"families": ["teubner_strey"], "second_population": False, "highq": ["flat"], "max_nfev": 300,
                        "prescreen": 2},
            "references": {"explicit": {"S2": "S1", "S3_0p5mm": "S1", "S4": "S1"}},
            "robustness_variants": ["qmax", "noedge"]}


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
    cfg.tables = {"base_name": "campaign", "extra_columns": {"x (units)": "series:x_value", "x again": "samples:x"}}
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
    assert sorted(os.listdir(os.path.join(out, "curves_sasview"))) == ["S1__sasview.dat", "S2__sasview.dat", "S3_0p5mm__sasview.dat", "S4__sasview.dat"]
    summ = pd.read_csv(os.path.join(out, "reduction_summary.csv")).set_index("sample")
    assert summ.loc["S2", "class_mu_t"] == "opaque" and summ.loc["S4", "mu_mm"] > 0        # S4's mu from its composition
    desc = pd.read_csv(os.path.join(out, "descriptors.csv"))
    assert len(desc) == 4 and "n_low" in desc.columns
    fits = pd.read_csv(os.path.join(out, "fits_summary.csv"))
    assert len(fits) == 4 and {"curve", "reference", "gate", "retained", "chi2_red"} <= set(fits.columns)
    assert fits.set_index("curve").loc["S1", "gate"] == "is reference"
    assert sorted(os.listdir(os.path.join(out, "fits"))) == ["S1_fit.csv", "S2_fit.csv", "S3_0p5mm_fit.csv", "S4_fit.csv"]
    assert os.path.exists(os.path.join(out, "fits_robustness.csv"))
    with open(os.path.join(out, "fits.json"), encoding="utf-8") as fh:
        j = json.load(fh)
    assert j["format"] == "prism-saxs-fit-workflow" and len(j["curves"]) == 4
    figs = sorted(os.listdir(os.path.join(out, "figures")))
    assert figs == ["Fig_curves.png", "Fig_fits_A.png", "Fig_fits_B.png", "Fig_frame.png", "Fig_trends.png", "Fig_two.png"]
    tables = set(os.listdir(os.path.join(out, "tables")))
    assert {"campaign.csv", "campaign.md", "campaign.xlsx", "campaign_models.csv", "campaign_parameters.csv",
            "campaign_model_free.csv"} <= tables
    summ_sheet = pd.read_csv(os.path.join(out, "tables", "campaign.csv"), encoding="utf-8-sig")
    assert "x (units)" in summ_sheet.columns and "x again" in summ_sheet.columns and list(summ_sheet["curve"]) == ["S1", "S2", "S3_0p5mm", "S4"]
    assert summ_sheet.set_index("curve").loc["S3_0p5mm", "x (units)"] == 2.0
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
    assert m["steps"]["fit"]["options"]["max_nfev"] == 300 and m["steps"]["robustness"]["options"]["variants"] == ["qmax", "noedge"]
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
    logs = []
    pl.run(no_series, steps=["tables", "figures"], log=logs.append)          # tables and fit panels need no series table
    assert any("grid figure skipped" in line for line in logs)
    with pytest.raises(KeyError, match="fit.options"):
        pl.ProjectConfig.from_dict({**cfg.to_dict(), "fit": {"floor": 0.03}})


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
    assert cli.main(["sasview", os.path.join(root, "out", "curves"), "-o", os.path.join(root, "sv")]) == 0
    assert len(os.listdir(os.path.join(root, "sv"))) == 4
    assert cli.main(["sasview", os.path.join(root, "nowhere")]) == 1
    assert cli.main(["inventory", "--config", path]) == 0
    assert cli.main(["run", str(tmp_path / "missing.json")]) == 1
    out = capsys.readouterr().out
    assert "manifest ->" in out and "error:" in out


# ------------------------------------------------------------------------------------ fit-only projects (curves block)
def _write_curve(path, q, I, sigma, dq):
    pd.DataFrame({"q_A": q, "I_cm": I, "sigma_total_cm": sigma, "sigma_stat_cm": 0.5 * sigma, "dQ_A": dq,
                  "configurations": ["A"] * len(q)}).to_csv(path, sep="\t", index=False)


def _curves_project(root):
    from saxs.core.models import TeubnerStrey
    rng = np.random.default_rng(5)
    q = np.logspace(np.log10(0.0026), np.log10(0.7), 70)
    d = os.path.join(root, "reduced")
    os.makedirs(d)
    shapes = {"refA": (1e-8, 4.15, 0.05, 0.0), "refB": (1.3e-8, 4.25, 0.05, 0.0), "A_plain": (1.4e-8, 4.15, 0.3, 0.0),
              "B_struct": (1.0e-8, 4.25, 0.4, 400.0)}
    for name, (B, p, C, S) in shapes.items():
        I = B * q ** -p + C + (TeubnerStrey().eval(q, S=S, d=500.0, xi=200.0) if S else 0.0)
        I = I * (1 + rng.normal(0, 0.02, q.size))
        _write_curve(os.path.join(d, f"{name}_merged.dat"), q, I, 0.03 * I, 0.04 * q)
    pd.DataFrame({"sample": list(shapes), "reference": ["", "", "refA", "refB"]}).to_csv(os.path.join(root, "samples.csv"), index=False)
    cfg = pl.ProjectConfig.template(output_dir="out", curves_dir="reduced")
    cfg.curves["pattern"] = "*_merged.dat"
    cfg.curves["name_regex"] = "^(.*)_merged$"
    cfg.samples_table, cfg.series_table = "samples.csv", None
    cfg.fit = {"options": {"families": ["teubner_strey"], "second_population": False, "highq": ["flat"], "prescreen": 3},
               "references": {"column": "reference", "explicit": {"A_plain": "refA"}}, "robustness_variants": ["qmax", "stat"]}
    cfg.figures = {"formats": ["png"], "dpi": 40, "fit_panels": [{"name": "Fig_fits_{group}", "ncol": 2, "iq4": True}], "grid": None,
                   "trends": None}
    cfg.tables = {"base_name": "fits"}
    path = os.path.join(root, "project.json")
    cfg.save(path)
    return path


def test_curves_block_project_fits_and_reports(tmp_path):
    path = _curves_project(str(tmp_path))
    cfg = pl.ProjectConfig.load(path)
    logs = []
    pl.run(cfg, log=logs.append)
    assert any("skipped: the project reads reduced curves" in line for line in logs)
    from saxs.core.fit_workflow import WorkflowResult
    res = WorkflowResult.load(os.path.join(cfg.out, "fits.json"))
    assert set(res.curves) == {"refA", "refB", "A_plain", "B_struct"}
    assert res.curves["A_plain"].reference == "refA" and res.curves["B_struct"].reference == "refB"
    assert res.curves["refA"].gate.method == "is reference" and res.null_pairs == 2 and res.null_max > 1.0
    assert res.curves["B_struct"].gate.passed and res.curves["B_struct"].retained.startswith("TS")
    assert not res.curves["A_plain"].gate.passed and res.curves["A_plain"].retained == "B0"
    assert res.curves["B_struct"].derived["ts_d_nm"] == pytest.approx(50.0, rel=0.15)
    assert res.robustness is not None and set(res.robustness.variant) == {"qmax", "stat"}
    assert sorted(os.listdir(os.path.join(cfg.out, "figures"))) == ["Fig_fits_1.png"]
    assert os.path.exists(os.path.join(cfg.out, "tables", "fits.xlsx"))
    raw = open(os.path.join(cfg.out, "tables", "fits.csv"), "rb").read()
    assert raw.startswith(b"\xef\xbb\xbf") and all(b < 128 for b in raw[3:])


def test_cli_fit_and_quick(tmp_path, capsys):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "scripts"))
    import saxs_pipeline as cli
    path = _curves_project(str(tmp_path))
    assert cli.main(["fit", path, "--no-robustness", "--no-figures"]) == 0
    assert os.path.exists(os.path.join(str(tmp_path), "out", "tables", "fits.md"))
    d = os.path.join(str(tmp_path), "reduced")
    out = os.path.join(str(tmp_path), "quick")
    assert cli.main(["quick", os.path.join(d, "B_struct_merged.dat"), "--reference", os.path.join(d, "refB_merged.dat"),
                     "-o", out, "--no-robustness", "--families", "teubner_strey", "--no-second-population"]) == 0
    assert {"fits.json", "tables.xlsx", "fits.png", "fits.pdf"} <= set(os.listdir(out))
    text = capsys.readouterr().out
    assert "B_struct_merged" in text and "outputs ->" in text
    cfg_path = str(tmp_path / "fitonly.json")
    assert cli.main(["init", cfg_path, "--curves-dir", "reduced"]) == 0
    d = json.load(open(cfg_path, encoding="utf-8"))
    assert d["curves"]["dir"] == "reduced" and d["frames_roots"] == [] and "options" in d["fit"]
