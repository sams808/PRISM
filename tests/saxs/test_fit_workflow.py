"""Tests for saxs/core/fit_workflow.py on the synthetic series of fit_synthetic.py: base fits, the reference-gate null
calibration, gate decisions (structure present / absent / reference curve / no reference), retention of the true
structure with its true parameters, recorded reasons, robustness, tables, JSON round trip, input building."""
from __future__ import annotations

import json

import numpy as np
import pytest

import fit_synthetic as fs  # noqa: E402  (tests/saxs is on sys.path under pytest)
from saxs.core import fit_workflow as fw
from saxs.core.loader import load_tabular_curve
from saxs.core.tables import write_workflow_tables


@pytest.fixture(scope="module")
def result():
    return fs.small_result()


def test_null_calibration_and_gates(result):
    assert result.null_pairs == 2 and 1.0 < result.null_max < 3.0      # two references, different grain exponents
    g = {n: c.gate for n, c in result.curves.items()}
    assert g["refA"].method == g["refB"].method == "is reference" and not g["refA"].passed
    assert g["B_struct"].passed and g["A_sph"].passed
    assert not g["A_plain"].passed and g["A_plain"].peak < result.null_max
    assert g["B_struct"].run.span >= 2 and g["B_struct"].peak > 10


def test_retains_the_true_structures_with_their_parameters(result):
    ts = result.curves["B_struct"]
    assert ts.retained == "TS" and ts.reasons["TS"] == ["ok"]
    assert ts.retained_fit.value("ts_d") == pytest.approx(fs.TRUE["B_struct"]["ts_d"], rel=0.05)
    assert ts.retained_fit.value("ts_xi") == pytest.approx(fs.TRUE["B_struct"]["ts_xi"], rel=0.10)
    assert ts.derived["ts_d_nm"] == pytest.approx(50.0, rel=0.05) and -1 < ts.derived["ts_fa"] < 0
    sph = result.curves["A_sph"]
    assert sph.retained == "SPH"
    f = sph.retained_fit
    assert f.value("sph_R") == pytest.approx(fs.TRUE["A_sph"]["sph_R"], rel=0.10)
    assert f.value("sph_phi") == pytest.approx(fs.TRUE["A_sph"]["sph_phi"], rel=0.25)        # absolute: contrast fixed
    assert f.params["sph_drho"].value == fs.CONTRAST and not f.params["sph_drho"].vary
    assert sph.derived["sph_phi"] == pytest.approx(f.value("sph_phi")) and sph.derived["sph_absolute"]
    assert result.curves["A_plain"].retained == "B0" and result.curves["refA"].retained == "B0"


def test_reasons_and_comparison_table(result):
    ts = result.curves["B_struct"]
    assert set(ts.reasons) == {"TS", "SPH"}
    cmp_ = fw.comparison_table(result)
    row = cmp_[(cmp_.curve == "B_struct") & (cmp_.model == "TS")].iloc[0]
    assert row.retained and row.verdict == "ok" and row.dBIC_vs_base > 10
    assert set(cmp_[cmp_.curve == "refA"].model) == {"B0"}           # references: base only


def test_selection_rules_on_a_handmade_result(result):
    import copy
    cr = copy.deepcopy(result.curves["B_struct"])
    o = result.options
    # a candidate no better than its base by more than dbic_min is refused
    f = cr.fits["TS"]
    f.bic = cr.fits["B0"].bic - 5.0
    fw.select(cr, o)
    assert cr.retained != "TS" and any("dBIC" in r for r in cr.reasons["TS"])
    # a parameter at a bound is refused
    cr = copy.deepcopy(result.curves["B_struct"])
    cr.fits["TS"].params["ts_d"].at_bound = True
    fw.select(cr, o)
    assert cr.retained != "TS" and any("at bound" in r for r in cr.reasons["TS"])
    # with every candidate refused the base is retained
    cr = copy.deepcopy(result.curves["B_struct"])
    for k in ("TS", "SPH"):
        cr.fits[k].params[next(iter(p for p in cr.fits[k].params if cr.fits[k].params[p].vary))].at_bound = True
    fw.select(cr, o)
    assert cr.retained == cr.base


def test_helpers(result):
    assert fw.parse_key("TS+UNI2@B1", "B0") == ("TS", True, "B1")
    assert fw.parse_key("SPH", "B0") == ("SPH", False, "B0")
    f = result.curves["B_struct"].fits["TS"]
    assert fw.component_visible(f, "ts", result.options)
    assert not fw.component_visible(f, "flat", fw.WorkflowOptions(visibility_qmax=0.01))
    assert fw.sphere_in_window(result.curves["A_sph"].fits["SPH"])


def test_json_round_trip(result, tmp_path):
    p = str(tmp_path / "fits.json")
    result.save(p)
    back = fw.WorkflowResult.load(p)
    assert set(back.curves) == set(result.curves) and back.null_pairs == result.null_pairs
    for n, cr in result.curves.items():
        b = back.curves[n]
        assert b.retained == cr.retained and b.gate.passed == cr.gate.passed and b.reasons == cr.reasons
        assert b.retained_fit.bic == pytest.approx(cr.retained_fit.bic)
    assert back.curves["B_struct"].derived["ts_d_nm"] == pytest.approx(result.curves["B_struct"].derived["ts_d_nm"])
    assert json.load(open(p, encoding="utf-8"))["format"] == "prism-saxs-fit-workflow"
    with pytest.raises(ValueError):
        fw.WorkflowResult.from_dict({"format": "other"})


def test_robustness_and_tables(tmp_path):
    res = fw.run_workflow(fs.inputs(), fs.fast_options(offset_variant={"factor": 1.1, "q_lo": 0.02, "q_hi": 0.035, "tag": "group", "value": "B"}))
    df = fw.robustness(res, fs.inputs())
    assert set(df.curve) == {"B_struct", "A_sph"}
    assert {"stat", "qmax", "noedge", "nosmear"} <= set(df.variant)
    assert set(df[df.curve == "B_struct"].variant) >= {"offset"} and "offset" not in set(df[df.curve == "A_sph"].variant)
    fl = fw.robust_flags(df, "B_struct")
    assert fl["robust_primary"] is True and "stat" in fl["robust_variants"]
    assert fw.param_robust(df, "B_struct", "ts_d")
    summ = fw.summary_table(res, extra={"B_struct": {"x": 1.0}})
    assert list(summ.columns[:1]) == ["x"] and summ.set_index("curve").loc["B_struct", "robust_primary"] is True
    paths = write_workflow_tables(res, str(tmp_path / "t"))
    assert {"xlsx", "csv", "md", "csv:models", "csv:parameters", "csv:robustness"} <= set(paths)
    raw = open(paths["csv"], "rb").read()
    assert raw.startswith(b"\xef\xbb\xbf") and all(b < 128 for b in raw[3:])
    assert "Gate" in open(paths["md"], encoding="utf-8").read()                     # method notes as bullets
    ct = fw.curve_table(res.curves["B_struct"])
    assert {"q_invA", "I", "model", "residual_norm", "comp_ts", "comp_grain", "comp_flat"} <= set(ct.columns)


def test_options_and_policies():
    with pytest.raises(KeyError):
        fw.WorkflowOptions.from_dict({"nope": 1})
    with pytest.raises(KeyError):
        fw.WorkflowOptions.from_dict({"families": ["gaussian_bump"]})
    o = fw.WorkflowOptions.from_dict({"grain_p_bounds": [3.5, 4.5], "families": ["debye"]})
    assert o.grain_p_bounds == (3.5, 4.5) and o.families == ("debye",)
    assert fw.WorkflowOptions.from_dict(o.to_dict()) == o
    # no reference: structures offered (flagged) or skipped
    ins = [c for c in fs.inputs() if c.name == "B_struct"]
    ins[0].reference = None
    r = fw.run_workflow(ins, fs.fast_options(families=("teubner_strey",)))
    assert r.curves["B_struct"].gate.method == "no reference" and r.curves["B_struct"].retained == "TS"
    r = fw.run_workflow(ins, fs.fast_options(families=("teubner_strey",), unreferenced="skip"))
    assert r.curves["B_struct"].retained == "B0"
    with pytest.raises(KeyError):
        fw.run_workflow([fw.CurveInput(name="x", q=fs.Q, I=fs.Q ** -4, sigma=None, reference="missing")], fs.fast_options())


def test_curve_input_from_tabular_file(tmp_path):
    paths = fs.write_tabular(str(tmp_path))
    c = load_tabular_curve(paths[0])
    ci = fw.CurveInput.from_curve(c, reference="refB")
    assert ci.name == c.name and ci.reference == "refB"
    np.testing.assert_allclose(ci.dq, 0.04 * fs.Q)
    np.testing.assert_allclose(ci.sigma_stat, 0.5 * ci.sigma)
    assert ci.smeared_flag.dtype == bool and ci.smeared_flag.sum() == int((fs.Q < 0.0035).sum())
