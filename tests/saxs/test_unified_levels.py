"""Tests for saxs.core.unified_levels: synthetic two-level curves with Poisson noise anchored to the
weakest feature the assertions depend on (the high-q plateau), forward-selection properties, bound
flags, robustness and the JSON round trip."""
from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from saxs.core import unified_levels as ul
from saxs.core.composite_models import BeaucageTied, BeaucageUnified, beaucage_polydisperse_prefactor
from saxs.core.curve import Curve

Q = np.geomspace(5e-3, 0.22, 640)          # a 1-pixel-like grid over the usable range
TRUTH = dict(logG1=3.0, Rg1=600.0, logB1=-7.0, d1=4.0, logG2=np.log10(20.0), Rg2=100.0, d2=3.5, logpf2=np.log10(0.5),
             logC=np.log10(0.02), logT=np.log10(0.5))


def true_curve(l2=True, tail=True, **over):
    v = dict(TRUTH, **over)
    return ul.model(v, Q, l2, tail)


def poisson_curve(I_true, counts_at_min=400.0, seed=0, name="syn", cls="marginal", usable=None):
    """Poisson noise anchored to the curve MINIMUM (the plateau), the weakest feature the selection of
    the tail / background depends on; the knee and the low-q grains then have far more counts."""
    rng = np.random.default_rng(seed)
    scale = counts_at_min / float(np.min(I_true))
    counts = rng.poisson(I_true * scale)
    I = counts / scale
    sig = np.sqrt(np.maximum(counts, 1.0)) / scale
    md = {"sample": name, "class_mu_t": cls, "per_mm": True,
          "columns": {"usable": (np.ones(len(Q), int) if usable is None else usable).tolist()}}
    return Curve(q=Q, intensity=I, sigma=sig, name=name, metadata=md)


@pytest.fixture(scope="module")
def two_level_fit():
    c = poisson_curve(true_curve(), name="two")
    return c, ul.fit_with_robustness(c, ul.UnifiedOptions())


def test_model_uses_the_shared_composite_components():
    v = dict(TRUTH)
    comps = ul.components(v, Q, True, True)
    np.testing.assert_allclose(comps["L1"], BeaucageUnified().eval(Q, G=10 ** v["logG1"], Rg=v["Rg1"], B=10 ** v["logB1"], p=v["d1"]), rtol=1e-14)
    np.testing.assert_allclose(comps["L2"], BeaucageTied().eval(Q, G=10 ** v["logG2"], Rg=v["Rg2"], d=v["d2"], pf=10 ** v["logpf2"]), rtol=1e-14)
    np.testing.assert_allclose(comps["background"], 10 ** v["logC"] + 10 ** v["logT"] * Q ** 2, rtol=1e-14)
    np.testing.assert_allclose(ul.model(v, Q, True, True), sum(comps.values()), rtol=1e-14)
    assert set(ul.components(v, Q, False, False)) == {"L1", "background"}


def test_two_level_curve_selects_l2_and_tail_and_recovers_the_parameters(two_level_fit):
    c, r = two_level_fit
    assert r.fitted and r.l2 and r.tail and r.n_params == 10 and r.name == "L1_L2_T"
    assert r.selection_path[0] == "L1" and r.selection_path[-1] == "L1_L2_T"
    assert r.dBIC_L2 > 10 and r.dBIC_tail > 10 and r.dBIC_vs_base > 20
    assert r.rms_log < 0.03 and r.rms_base > r.rms_log
    assert r.params["Rg2"].value == pytest.approx(100.0, rel=0.10)
    assert r.params["d2"].value == pytest.approx(3.5, abs=0.4)
    assert r.params["Rg1"].value == pytest.approx(600.0, rel=0.25)   # Guinier region of L1 below the window: loosely set
    assert 10 ** r.params["logG2"].value == pytest.approx(20.0, rel=0.3)
    assert 10 ** r.params["logC"].value == pytest.approx(0.02, rel=0.2)
    assert not r.params["Rg2"].at_bound and not r.params["d2"].at_bound
    assert r.derived["D2_eq_nm"] == pytest.approx(2 * r.params["Rg2"].value * np.sqrt(5 / 3) / 10, rel=1e-12)
    assert r.derived["B2"] == pytest.approx(10 ** r.params["logpf2"].value * beaucage_polydisperse_prefactor(
        10 ** r.params["logG2"].value, r.params["Rg2"].value, r.params["d2"].value), rel=1e-12)
    assert set(r.candidates) == {"L1", "L1_T", "L1_L2", "L1_L2_T"}
    assert all("bic" in v and "rms_log" in v for v in r.candidates.values())
    assert r.n_pts > 50 and r.q_min == pytest.approx(Q[0], rel=0.05) and r.q_max <= 0.20
    assert r.cls == "marginal" and r.per_mm is True
    assert len(r.model()) == r.n_pts and np.all(np.isfinite(r.residual_log()))
    assert abs(np.mean(r.residual_log())) < 0.01


def test_robustness_pass_marks_a_real_level_resolved(two_level_fit):
    _c, r = two_level_fit
    assert r.relaxed is not None and r.relaxed.fitted
    assert r.relaxed.params["d1"].max == 5.0 and r.relaxed.params["Rg1"].max == 3000.0
    assert r.resolved is True
    assert r.relaxed.params["Rg2"].value == pytest.approx(r.params["Rg2"].value, rel=0.2)
    # relaxed bounds contain the original ones: the RMS does not get worse (numerical slack allowed)
    assert r.relaxed.rms_log <= r.rms_log + 0.002
    # a relaxed variant that moved Rg2 by 30 % would not be resolved; a missing level gives None
    moved = dataclasses.replace(r.relaxed, params=dict(r.relaxed.params, Rg2=dataclasses.replace(r.params["Rg2"], value=r.params["Rg2"].value * 1.3)))
    assert ul.resolved_flag(r, moved) is False
    assert ul.resolved_flag(r, None) is False
    assert ul.resolved_flag(dataclasses.replace(r, l2=False), r.relaxed) is None


def test_single_level_curve_keeps_no_fake_level():
    c = poisson_curve(true_curve(l2=False, tail=False), name="one", seed=3)
    r = ul.fit_unified_levels(c)
    assert r.fitted and not r.l2
    assert r.n_params in (5, 6)
    assert r.rms_log < 0.03
    assert r.params["Rg1"].value == pytest.approx(600.0, rel=0.25)   # Guinier region of L1 below the window: loosely set
    assert 10 ** r.params["logC"].value == pytest.approx(0.02, rel=0.25)
    # every L2 candidate that was tried either lost on the BIC or was a fake level; none was kept
    assert "L1_L2" not in r.selection_path and "L1_L2_T" not in r.selection_path


def test_forward_selection_is_monotone_and_rejects_fake_levels():
    o = ul.UnifiedOptions()
    bics = {(False, False): 100.0, (False, True): 85.0, (True, False): 88.0, (True, True): 70.0}
    g2 = {(False, False): None, (False, True): None, (True, False): 1.0, (True, True): 1.0}
    chosen, path = ul.forward_select(bics, g2, 100.0, o)
    assert chosen == (True, True) and [k for k, _ in path] == [(False, False), (False, True), (True, True)]
    assert all(b1 - b2 > o.dbic_min for (_, b1), (_, b2) in zip(path, path[1:]))
    # a drop of 9 is not enough
    bics2 = {**bics, (False, True): 91.0, (True, False): 91.0}
    assert ul.forward_select(bics2, g2, 100.0, o)[0] == (False, False)
    # the L2 candidate wins by a lot but its G2 is 1e-9 of the maximum: it is never eligible
    bics3 = {(False, False): 100.0, (False, True): 95.0, (True, False): 20.0, (True, True): 5.0}
    g2_fake = {(False, False): None, (False, True): None, (True, False): 1e-7, (True, True): 1e-7}
    chosen3, _ = ul.forward_select(bics3, g2_fake, 100.0, o)
    assert chosen3 == (False, False)
    # with a real amplitude the same BICs select both terms
    assert ul.forward_select(bics3, g2, 100.0, o)[0] == (True, True)
    # missing candidates (a failed fit) are simply not options
    chosen4, _ = ul.forward_select({(False, False): 100.0, (True, False): 50.0}, {(False, False): None, (True, False): 1.0}, 100.0, o)
    assert chosen4 == (True, False)


def test_at_bound_flags_and_rule():
    assert ul.is_at_bound(4.5, 3.0, 4.5) and ul.is_at_bound(3.02, 3.0, 4.5) and not ul.is_at_bound(4.0, 3.0, 4.5)
    assert ul.is_at_bound(1490.0, 250.0, 1500.0) and not ul.is_at_bound(1400.0, 250.0, 1500.0)
    assert ul.is_at_bound(-0.995, -1.0, 0.18) and ul.is_at_bound(0.175, -1.0, 0.18) and not ul.is_at_bound(-0.3, -1.0, 0.18)
    assert not ul.is_at_bound(1.0, -np.inf, np.inf)
    # a grain level whose true exponent is the upper bound lands on it and is flagged
    c = poisson_curve(true_curve(l2=False, tail=True, d1=4.5, logB1=-8.5), name="bound", seed=5)
    r = ul.fit_unified_levels(c)
    assert r.params["d1"].at_bound and "d1" in r.at_bound_names()
    assert any("d1 at a bound" in w for w in r.warnings)
    assert r.params["d1"].value == pytest.approx(4.5, abs=0.05)


def test_prepare_curve_honours_usable_flag_qmax_and_minimum_points():
    I = true_curve()
    usable = np.ones(len(Q), int)
    usable[Q > 0.15] = 0
    c = poisson_curve(I, usable=usable)
    q, _I, _s, reason = ul.prepare_curve(c, ul.UnifiedOptions(q_max=0.20))
    assert reason is None and q.max() < 0.15 and len(q) > 40
    q2, *_r = ul.prepare_curve(c, ul.UnifiedOptions(use_usable=False))
    assert q2.max() > 0.15 and q2.max() <= 0.20
    short = poisson_curve(I, usable=np.r_[np.ones(20, int), np.zeros(len(Q) - 20, int)])
    assert "usable points" in ul.prepare_curve(short)[3]
    high = poisson_curve(I, usable=(Q > 0.05).astype(int))
    assert "usable only above" in ul.prepare_curve(high)[3]
    r = ul.fit_unified_levels(short)
    assert not r.fitted and r.skip_reason and r.n_params == 0


def test_json_round_trip_and_batch_table(two_level_fit, tmp_path):
    c, r = two_level_fit
    d = r.to_json()
    text = json.dumps(d)                                   # everything is JSON-serialisable
    back = ul.UnifiedFitResult.from_json(json.loads(text))
    assert back.sample == r.sample and back.l2 == r.l2 and back.tail == r.tail and back.resolved == r.resolved
    assert back.params["Rg2"].value == r.params["Rg2"].value and back.params["Rg2"].at_bound == r.params["Rg2"].at_bound
    assert back.relaxed is not None and back.relaxed.params["Rg1"].max == 3000.0
    np.testing.assert_allclose(back.model(), r.model(), rtol=1e-12)
    assert back.derived == r.derived and back.candidates == r.candidates
    path = tmp_path / "fits.json"
    ul.save_results_json([r], str(path))
    loaded = ul.load_results_json(str(path))
    assert len(loaded) == 1 and loaded[0].params["Rg2"].value == r.params["Rg2"].value
    with pytest.raises(ValueError):
        bad = tmp_path / "bad.json"
        bad.write_text('{"format": "other"}', encoding="utf-8")
        ul.load_results_json(str(bad))
    # batch: one fittable curve, one too short; the table keeps a row for both
    short = poisson_curve(true_curve(), name="short", usable=np.r_[np.ones(20, int), np.zeros(len(Q) - 20, int)])
    seen = []
    results, df = ul.batch_fit([c, short], robustness=False, progress=lambda i, n, m: seen.append(m))
    assert list(df["sample"]) == ["two", "short"] and seen[-1] == "done"
    assert bool(df.loc[0, "fitted"]) and not bool(df.loc[1, "fitted"]) and "usable points" in df.loc[1, "skip_reason"]
    assert bool(df.loc[0, "l2"]) and df.loc[0, "Rg2_A"] == pytest.approx(results[0].params["Rg2"].value)
    assert np.isnan(df.loc[1, "Rg2_A"]) and df.loc[0, "relaxed_l2"] is None
    for col in ("Rg1_at_bound", "d1_at_bound", "Rg2_at_bound", "d2_at_bound", "pf2_at_bound", "covariance_ok", "at_bounds",
                "dBIC_L2", "dBIC_tail", "D2_eq_nm", "resolved", "warnings"):
        assert col in df.columns, col
    row = ul.result_row(r)
    assert row["relaxed_l2"] is True and row["resolved"] is True and np.isfinite(row["relaxed_Rg2_A"])
    assert ul.results_table([]).shape[0] == 0


def test_options_round_trip_and_relaxed_variant():
    o = ul.UnifiedOptions(dbic_min=8.0, rg2_bounds=(30.0, 250.0))
    d = o.to_dict()
    back = ul.UnifiedOptions.from_dict(json.loads(json.dumps(d)))
    assert back.dbic_min == 8.0 and back.rg2_bounds == (30.0, 250.0) and back.rg1_starts == (400.0, 700.0)
    rel = back.relaxed()
    assert rel.d1_bounds == (3.0, 5.0) and rel.rg1_bounds == (250.0, 3000.0) and rel.rg2_bounds == (30.0, 250.0)
    with pytest.raises(KeyError):
        ul.UnifiedOptions.from_dict({"nope": 1})
    assert ul.candidate_name(True, False) == "L1_L2" and ul.candidate_name(False, True) == "L1_T"
