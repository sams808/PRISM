"""Tests for saxs_core/composite_staged.py — the staged fitting pipeline
(Phase 3: stages 0-4). Covers hygiene, sigma-model estimation, window
proposal, class guessing, determinism, never-raise behavior on both
peak-free and real profiles, and stage-by-stage result retention.
"""
from __future__ import annotations

import numpy as np
import pytest

from saxs_core.composite_fit import build_preset
from saxs_core.composite_staged import (
    apply_hygiene, estimate_sigma_model, fit_staged, guess_class,
    propose_windows,
)
from saxs_core.curve import Curve


def _ts_curve(name="synthetic_peaked", d=1200.0, xi=3000.0, S=5e6, seed=0, noise=True):
    """A realistic class-c curve: a low-q Guinier-Porod upturn (dominant
    at low q, the way a real measured profile actually looks — see the
    real physic_based/*__corr.dat header, I(q_min) ~ 1e8) with a
    Teubner-Strey peak riding on it and a decay to a small flat
    background — matches spec §7's own description, and matters in
    practice: a toy curve with a small flat background dominating
    everywhere (an earlier draft of this fixture) doesn't stress the
    Kratky-based peak/window detectors the same way real data does.
    Linearly-spaced q matches the real instrument's constant-Δq grid.
    """
    model = build_preset("BG_TS_GP")
    q = np.linspace(1e-3, 0.3, 900)
    true = {"bg_C": 500.0, "pl_B": 1e-9, "pl_p": 4.0,
            "ts_S": S, "ts_d": d, "ts_xi": xi,
            "gp_G": 4e8, "gp_Rg": 2000.0, "gp_p": 4.0}
    I = model.eval(q, true)
    if noise:
        rng = np.random.default_rng(seed)
        sigma = np.sqrt(np.abs(I)) * 0.005 + 0.5
        I = I + rng.normal(0, sigma)
    return Curve(q=q, intensity=np.clip(I, 1e-6, None), sigma=None, name=name)


def _flat_curve(name="synthetic_flat", seed=1):
    """A genuine class-a curve: pure background + mild power-law decay,
    no upturn/feature anywhere — q^2*I is monotonic, no interior peak."""
    q = np.linspace(1e-3, 0.3, 900)
    model = build_preset("BG")
    I = model.eval(q, {"bg_C": 500.0, "pl_B": 1e-9, "pl_p": 2.0})
    rng = np.random.default_rng(seed)
    I = I + rng.normal(0, 0.02 * np.sqrt(np.abs(I)) + 0.02, q.shape)
    return Curve(q=q, intensity=np.clip(I, 1e-6, None), sigma=None, name=name)


# ---------------------------------------------------------------------------
# Stage 0 building blocks
# ---------------------------------------------------------------------------

def test_estimate_sigma_model_is_positive_and_scales_with_intensity():
    q = np.linspace(1e-3, 0.3, 500)
    I = 100.0 * np.exp(-q * 10) + 5.0
    sigma = estimate_sigma_model(q, I)
    assert np.all(sigma > 0)
    assert sigma[np.argmax(I)] >= sigma[np.argmin(I)] * 0.5  # roughly tracks sqrt(I)


def test_apply_hygiene_trims_edges_and_drops_nonfinite():
    q = np.linspace(1e-3, 0.3, 100)
    I = np.full_like(q, 10.0)
    I[5] = np.nan
    I[50] = -1.0
    curve = Curve(q=q, intensity=I, sigma=None, name="dirty")
    result = apply_hygiene(curve, trim_n=3)
    assert result.n_trimmed_edge == 6
    assert result.n_dropped_nonfinite == 2
    assert np.all(np.isfinite(result.curve.intensity))
    assert np.all(result.curve.intensity >= 0)
    assert result.sigma_model == "poisson_like_estimated"
    assert result.curve.sigma is not None


def test_apply_hygiene_keeps_measured_sigma_when_present():
    q = np.linspace(1e-3, 0.3, 50)
    I = np.full_like(q, 10.0)
    sigma = np.full_like(q, 0.5)
    curve = Curve(q=q, intensity=I, sigma=sigma, name="withsigma")
    result = apply_hygiene(curve, trim_n=2)
    assert result.sigma_model == "measured"
    np.testing.assert_allclose(result.curve.sigma, 0.5)


def test_guess_class_distinguishes_peaked_from_featureless():
    peaked = _ts_curve(noise=False)
    flat = _flat_curve()
    cls_peak, prom_peak = guess_class(peaked.q, peaked.intensity)
    cls_flat, prom_flat = guess_class(flat.q, flat.intensity)
    assert cls_peak == "c"
    assert cls_flat == "a"
    assert prom_peak > prom_flat


def test_propose_windows_peak_window_brackets_true_peak():
    curve = _ts_curve(d=1200.0, noise=False)
    windows = propose_windows(curve.q, curve.intensity)
    q_true_peak = 2 * np.pi / 1200.0 * 0.99  # d and q_max nearly coincide for this xi
    lo, hi = windows["W_peak"]
    assert lo < q_true_peak < hi
    assert windows["W_loq"][1] <= windows["W_peak"][0] * 1.01
    assert windows["W_hiq"][0] >= windows["W_peak"][1] * 0.99


# ---------------------------------------------------------------------------
# fit_staged: never raises, stage retention, determinism
# ---------------------------------------------------------------------------

def test_fit_staged_never_raises_on_featureless_curve():
    curve = _flat_curve()
    result = fit_staged(curve, multistart_n=2)
    # TS must be rejected (the class-a guardrail, pulled forward from
    # spec Stage 6) — whether Stage 3 still provisionally adds a
    # guinier_porod term is for Phase 4's BIC ladder to properly settle,
    # not asserted here.
    assert result.no_peak is True
    assert "ts_rejected" in "".join(result.flags) or "ts_skipped" in "".join(result.flags)
    assert "gof" in result.__dict__ and result.gof["n_points"] > 0


def test_fit_staged_recovers_ts_peak_on_synthetic_curve():
    curve = _ts_curve(d=1200.0, xi=3000.0, seed=3)
    result = fit_staged(curve, multistart_n=4)
    assert result.no_peak is False
    assert "TS" in result.preset_chosen
    assert result.derived["d"] == pytest.approx(1200.0, rel=0.15)
    assert result.derived["xi"] == pytest.approx(3000.0, rel=0.3)
    assert -1.0 < result.derived["fa"] < 0.0


def test_fit_staged_retains_every_stage():
    curve = _ts_curve(seed=4)
    result = fit_staged(curve, multistart_n=2)
    assert set(result.stages) == {"stage0", "stage1", "stage2", "stage3", "stage4"}
    assert "class_guess" in result.stages["stage0"]
    assert "redchi" in result.stages["stage1"]


def test_fit_staged_is_deterministic_given_same_sample_id():
    curve = _ts_curve(seed=5)
    r1 = fit_staged(curve, sample_id="fixed_id", multistart_n=4)
    r2 = fit_staged(curve, sample_id="fixed_id", multistart_n=4)
    assert r1.derived["d"] == pytest.approx(r2.derived["d"], rel=1e-9)
    assert r1.gof["chi2red"] == pytest.approx(r2.gof["chi2red"], rel=1e-9)


def test_fit_staged_json_round_trip(tmp_path):
    """Round-trip fidelity of the JSON serialization itself — independent
    of whether this particular curve/seed happens to recover a TS peak
    (that recovery accuracy is covered separately)."""
    curve = _ts_curve(seed=6)
    result = fit_staged(curve, multistart_n=2)
    path = tmp_path / "fit_result.json"
    result.save_json(str(path))
    loaded = type(result).load_json(str(path))
    assert loaded.sample_id == result.sample_id
    assert loaded.preset_chosen == result.preset_chosen
    assert loaded.gof == pytest.approx(result.gof)
    assert loaded.windows["W_peak"] == tuple(result.windows["W_peak"])
    assert loaded.to_json() == result.to_json()


def test_fit_staged_runs_on_real_physic_based_profile_when_available():
    import os
    real_path = r"C:\Users\samso\Desktop\WSU_work\SAXS\PBi-sorted\physic_based\P5Bi8-12__corr.dat"
    if not os.path.isfile(real_path):
        pytest.skip("real SAXS data folder not present on this machine")
    from saxs_core.loader import load_curve
    curve = load_curve(real_path)
    result = fit_staged(curve, multistart_n=2)
    assert result.gof["n_points"] > 100
    if not result.no_peak:
        assert 700.0 <= result.derived["d"] <= 1700.0
        assert 2500.0 <= result.derived["xi"] <= 5000.0
