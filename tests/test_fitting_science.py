"""Tests for fitting_science.py — single-spectrum peak fitting (Gaussian /
pseudo-Voigt via lmfit + rampy), consolidated from three near-identical
copies that used to live inline in main.py's SingleFitWindow.
"""
from __future__ import annotations

import numpy as np
import pytest
import rampy as rp

import fitting_science as fs


def _gaussian_component(center=500.0, fwhm=30.0, amp=100.0, shape="G", eta=0.5):
    return dict(
        shape=shape,
        shift_val=center, shift_min=center - 100, shift_max=center + 100,
        fwhm_val=fwhm, fwhm_min=1.0, fwhm_max=200.0,
        amp_val=amp,
        eta_val=eta, eta_min=0.0, eta_max=1.0,
    )


# --------------------------------------------------------------------------
# Synthetic ground truth: build a spectrum from a KNOWN Gaussian, fit it,
# and check the recovered parameters land close to what was built in.
# --------------------------------------------------------------------------

def test_fit_spectrum_classic_recovers_known_gaussian_peak():
    x = np.linspace(400, 600, 400)
    true_center, true_fwhm, true_amp = 505.0, 25.0, 80.0
    y = rp.gaussian(x, true_amp, true_center, true_fwhm)
    rng = np.random.default_rng(0)
    y_noisy = y + rng.normal(0, 0.05, size=y.shape)

    comp = _gaussian_component(center=500.0, fwhm=30.0, amp=100.0, shape="G")
    result = fs.fit_spectrum(x, y_noisy, [comp], mode="classic")

    assert result.lmfit_result.params["f0"].value == pytest.approx(true_center, abs=1.0)
    assert result.lmfit_result.params["l0"].value == pytest.approx(true_fwhm, abs=3.0)
    assert result.lmfit_result.params["a0"].value == pytest.approx(true_amp, rel=0.1)
    assert result.chi2_red < 0.1


def test_fit_spectrum_classic_handles_gl_shape():
    x = np.linspace(400, 600, 400)
    y = rp.pseudovoigt(x, 90.0, 505.0, 28.0, 0.4)
    comp = _gaussian_component(center=500.0, fwhm=30.0, amp=100.0, shape="GL", eta=0.5)
    result = fs.fit_spectrum(x, y, [comp], mode="classic")
    assert result.lmfit_result.params["f0"].value == pytest.approx(505.0, abs=2.0)
    assert "eta0" in result.lmfit_result.params


# --------------------------------------------------------------------------
# Real bundled data: no independently-verified ground truth, so just assert
# the fit converges to something physically sane (finite, positive amplitude,
# center within the search bounds).
# --------------------------------------------------------------------------

def test_fit_spectrum_on_real_raman_example_is_physically_sane(raman_example_path):
    data = np.loadtxt(raman_example_path)
    x, y = data[:, 0], data[:, 1]
    peak_idx = int(np.argmax(y))
    center_guess = float(x[peak_idx])

    comp = _gaussian_component(center=center_guess, fwhm=20.0, amp=float(y[peak_idx]), shape="G")
    comp["shift_min"], comp["shift_max"] = center_guess - 50, center_guess + 50
    result = fs.fit_spectrum(x, y, [comp], mode="classic")

    assert np.isfinite(result.chi2_red)
    assert result.lmfit_result.params["a0"].value > 0
    assert comp["shift_min"] <= result.lmfit_result.params["f0"].value <= comp["shift_max"]


# --------------------------------------------------------------------------
# origin_step mode: one relaxation step, callable repeatedly (the GUI's own
# convergence-checking loop lives in main.py, not here).
# --------------------------------------------------------------------------

def test_fit_spectrum_origin_step_reduces_chi2_toward_truth():
    x = np.linspace(400, 600, 400)
    true_center, true_fwhm, true_amp = 505.0, 25.0, 80.0
    y = rp.gaussian(x, true_amp, true_center, true_fwhm)

    comp = _gaussian_component(center=480.0, fwhm=40.0, amp=60.0, shape="G")
    params_struct = [comp]
    lm_params = fs.build_lmfit_parameters(params_struct)

    y_fit_before, _ = fs.compute_model(x, lm_params, params_struct)
    chi2_before = fs.compute_chi2(y, y_fit_before, lm_params)

    result = fs.fit_spectrum(x, y, params_struct, mode="origin_step", lm_params=lm_params, alpha=0.5)
    assert result.chi2_red < chi2_before


def test_fit_spectrum_origin_step_requires_lm_params():
    x = np.linspace(400, 600, 10)
    y = np.ones_like(x)
    comp = _gaussian_component()
    with pytest.raises(ValueError):
        fs.fit_spectrum(x, y, [comp], mode="origin_step", lm_params=None)


# --------------------------------------------------------------------------
# Bug-guard: fit_spectrum must be safe to call directly with non-numeric
# input, not just from the one GUI path that happens to coerce upstream.
# --------------------------------------------------------------------------

def test_fit_spectrum_rejects_non_numeric_input_with_clear_error():
    x = np.array(["a", "b", "c"], dtype=object)
    y = np.array([1.0, 2.0, 3.0])
    comp = _gaussian_component()
    with pytest.raises(TypeError):
        fs.fit_spectrum(x, y, [comp], mode="classic")


def test_compute_model_matches_sum_of_components():
    x = np.linspace(0, 100, 50)
    comps = [_gaussian_component(center=30, fwhm=5, amp=10), _gaussian_component(center=70, fwhm=8, amp=20)]
    lm_params = fs.build_lmfit_parameters(comps)
    total, peaks = fs.compute_model(x, lm_params, comps)
    assert len(peaks) == 2
    assert np.allclose(total, peaks[0] + peaks[1])


# --------------------------------------------------------------------------
# M8 additions: R^2, peak centroid, 2nd-derivative auto-peak-finder.
# --------------------------------------------------------------------------

def test_compute_r_squared_perfect_fit_is_one():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert fs.compute_r_squared(y, y) == pytest.approx(1.0)


def test_compute_r_squared_worse_than_mean_is_negative():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    y_fit = np.array([4.0, 1.0, 4.0, 1.0])  # far from y, worse than predicting the mean
    assert fs.compute_r_squared(y, y_fit) < 0


def test_peak_centroid_symmetric_gaussian_matches_center():
    x = np.linspace(400, 600, 2000)
    y = rp.gaussian(x, 50.0, 505.0, 20.0)
    assert fs.peak_centroid(x, y) == pytest.approx(505.0, abs=0.5)


def test_peak_centroid_zero_area_returns_nan():
    x = np.linspace(0, 10, 50)
    y = np.zeros_like(x)
    assert np.isnan(fs.peak_centroid(x, y))


def test_find_peak_candidates_recovers_known_peak_centers():
    x = np.linspace(0, 1000, 2000)
    y = rp.gaussian(x, 100.0, 300.0, 15.0) + rp.gaussian(x, 80.0, 700.0, 15.0)
    centers = fs.find_peak_candidates(x, y, max_peaks=5)
    assert len(centers) >= 2
    assert any(abs(c - 300.0) < 10 for c in centers)
    assert any(abs(c - 700.0) < 10 for c in centers)


def test_find_peak_candidates_excludes_edge_artifact():
    """Regression guard, found via manual smoke-testing against real bundled
    data (EXAMPLES/Raman_example.txt): raw Raman spectra often have a sharp,
    strong feature right at the edge of the recorded window (Rayleigh-line
    tail / detector edge), which a pure curvature-based finder would
    otherwise report as the single strongest candidate. Build a synthetic
    analog: a sharp spike right at x=0 (stronger than the real peak) plus a
    genuine interior Gaussian — only the interior one should come back."""
    x = np.linspace(0, 1000, 2000)
    edge_spike = 300.0 * np.exp(-((x - 0.5) ** 2) / (2 * 3.0 ** 2))
    interior_peak = rp.gaussian(x, 80.0, 500.0, 15.0)
    y = edge_spike + interior_peak

    centers = fs.find_peak_candidates(x, y, max_peaks=5)
    assert len(centers) >= 1
    assert all(c > 20.0 for c in centers)  # nothing within the excluded edge margin
    assert any(abs(c - 500.0) < 10 for c in centers)


def test_find_peak_candidates_on_short_input_returns_empty():
    assert fs.find_peak_candidates(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == []
