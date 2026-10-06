"""Tests for the physical-fit building blocks: tabular loader, per-point Gaussian resolution smearing, the absolute
polydisperse-sphere component, the log-weighted fit engine, visibility runs, reference ratio, contrast and invariant
helpers. Every numerical check is against an independent closed form, not against the code's own output."""
from __future__ import annotations

import math

import numpy as np
import pytest

from saxs.core.composite_models import (
    COMPONENTS, Dab, SphereSchulz, percus_yevick_sq, schulz_moment, schulz_radii, sphere_amplitude,
)
from saxs.core.loader import load_tabular_curve
from saxs.core.physical_fit import (
    N_A, R_E_CM, TermSpec, electron_density_cm3, fit_physical, invariant, reference_ratio, residual_runs,
    two_phase_product_cm4, volume_fraction_from_invariant, xray_sld_1e10cm2,
)
from saxs.core.resolution import smear_points


# ---------------------------------------------------------------- loader
def test_tabular_loader_keeps_text_and_dq_columns_and_counts_drops(tmp_path):
    p = tmp_path / "c.dat"
    p.write_text("# header line\n# units: q in 1/A\n"
                 "q_A\tI_cm\tsigma_total_cm\tsigma_stat_cm\tdQ_A\tflag\tconfigurations\n"
                 "0.02\t10\t1\t0.5\t0.001\t0\tA+B\n"
                 "0.01\t20\t2\t1\t0.001\t1\tA\n"
                 "0.03\tnan\t1\t1\t0.001\t0\tB\n"
                 "0.04\t5\t0.5\t0.2\t0.002\t0\tB\n", encoding="utf-8")
    c = load_tabular_curve(str(p))
    assert np.allclose(c.q, [0.01, 0.02, 0.04])                       # sorted, NaN row dropped
    assert np.allclose(c.intensity, [20, 10, 5])
    assert np.allclose(c.sigma, [2, 1, 0.5])                          # sigma_total chosen by name
    assert c.metadata["n_dropped"] == 1
    assert np.allclose(c.metadata["dq"], [0.001, 0.001, 0.002])
    assert c.metadata["columns"]["configurations"] == ["A", "A+B", "B"]
    assert np.allclose(c.metadata["columns"]["flag"], [1, 0, 0])


def test_tabular_loader_explicit_columns(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text("x,y,e\n0.1,1,0.1\n0.2,2,0.2\n", encoding="utf-8")
    c = load_tabular_curve(str(p), q_col="x", i_col="y", sigma_col="e")
    assert np.allclose(c.intensity, [1, 2]) and np.allclose(c.sigma, [0.1, 0.2])


# ---------------------------------------------------------------- resolution
def test_smearing_preserves_constants_and_linear_functions():
    q = np.linspace(0.05, 0.2, 7); dq = np.full_like(q, 0.005)
    assert np.allclose(smear_points(lambda x: np.full_like(x, 3.0), q, dq), 3.0)
    assert np.allclose(smear_points(lambda x: 2 * x + 1, q, dq), 2 * q + 1, rtol=1e-10)


def test_smearing_of_a_gaussian_adds_variances():
    # Gaussian of width s convolved with Gaussian of width d -> Gaussian of width sqrt(s^2 + d^2), area conserved.
    s, d, q0 = 0.004, 0.003, 0.05
    g = lambda x, w: np.exp(-0.5 * ((x - q0) / w) ** 2) / (w * math.sqrt(2 * math.pi))
    q = np.linspace(0.035, 0.065, 31)
    out = smear_points(lambda x: g(x, s), q, np.full_like(q, d), n_nodes=40)
    assert np.allclose(out, g(q, math.hypot(s, d)), rtol=2e-3, atol=1e-6 * g(q0, s))


def test_smearing_inflates_a_steep_power_law_by_the_second_moment():
    # For I = q^-4 and small dq/q: <I> / I = 1 + 10 (dq/q)^2 + O((dq/q)^4) (second derivative term).
    q = np.array([0.01, 0.02]); dq = 0.02 * q
    ratio = smear_points(lambda x: x ** -4.0, q, dq, n_nodes=40) / q ** -4.0
    assert np.allclose(ratio, 1 + 10 * 0.02 ** 2 + 105 * 0.02 ** 4 * 1.0, rtol=2e-5)


def test_smearing_truncates_at_zero_q():
    q = np.array([0.001]); dq = np.array([0.002])                    # half the Gaussian mass below q = 0
    out = smear_points(lambda x: np.ones_like(x), q, dq)
    assert np.allclose(out, 1.0)


def test_zero_dq_returns_unsmeared():
    q = np.array([0.01, 0.02]); f = lambda x: x ** -3
    assert np.allclose(smear_points(f, q, np.zeros(2)), f(q))


# ---------------------------------------------------------------- spheres
def test_sphere_amplitude_limits_and_first_zero():
    assert sphere_amplitude(np.array([0.0]))[0] == pytest.approx(1.0)
    assert sphere_amplitude(np.array([4.493409457909064]))[0] == pytest.approx(0.0, abs=1e-12)  # tan x = x


def test_schulz_quadrature_integrates_moments_exactly():
    for sig in (0.05, 0.3, 0.8):
        radii, w = schulz_radii(100.0, sig)
        assert w.sum() == pytest.approx(1.0)
        for k in (1, 3, 6, 8):
            assert np.sum(w * radii ** k) == pytest.approx(schulz_moment(100.0, sig, k), rel=1e-8)
    assert schulz_moment(100.0, 0.3, 1) == pytest.approx(100.0)
    # variance = (sig R)^2
    assert schulz_moment(100.0, 0.3, 2) - 100.0 ** 2 == pytest.approx((30.0) ** 2, rel=1e-9)


def test_sphere_forward_scattering_is_phi_drho2_v2_over_v():
    comp = SphereSchulz()
    phi, R, sig, drho = 0.02, 120.0, 0.2, 8.0
    I0 = comp.eval(np.array([1e-6]), phi=phi, R=R, sig=sig, drho=drho)[0]
    v2_over_v = (4 / 3 * math.pi) * schulz_moment(R, sig, 6) / schulz_moment(R, sig, 3)
    assert I0 == pytest.approx(phi * drho ** 2 * v2_over_v * 1e-4, rel=1e-6)


def test_monodisperse_limit_of_sphere_component():
    comp = SphereSchulz(); q = np.array([0.005, 0.02, 0.03])
    R = 100.0; V = 4 / 3 * math.pi * R ** 3
    exact = 0.01 * 2.0 ** 2 * V * sphere_amplitude(q * R) ** 2 * 1e-4
    got = comp.eval(q, phi=0.01, R=R, sig=0.05, drho=2.0)
    assert got[0] == pytest.approx(exact[0], rel=0.05)


def test_sphere_matches_dense_numerical_average():
    # independent reference: 4000-point grid over the Schulz pdf (scipy.stats.gamma), q over the whole window
    from scipy.stats import gamma
    q = np.logspace(-2.6, -0.15, 120)
    for R, sig in [(150.0, 0.15), (150.0, 0.4), (40.0, 0.3), (20.0, 0.6), (300.0, 0.1), (300.0, 0.05), (80.0, 0.7)]:
        a = 1 / sig ** 2
        r = np.linspace(gamma.ppf(1e-8, a, scale=R / a), gamma.ppf(1 - 1e-8, a, scale=R / a), 8000)
        w = gamma.pdf(r, a, scale=R / a); V = 4 / 3 * math.pi * r ** 3
        F = sphere_amplitude(q[:, None] * r[None, :])
        ref = 0.01 * 64 * (np.sum(w * V ** 2 * F ** 2, axis=1) / np.sum(w * V)) * 1e-4
        got = SphereSchulz().eval(q, phi=0.01, R=R, sig=sig, drho=8.0)
        assert np.max(np.abs(got / ref - 1)) < 0.035, (R, sig)


def test_sphere_porod_limit():
    # High q: I -> 2 pi phi drho^2 (S/V) q^-4 with S/V = 3 <R^2>/<R^3>
    comp = SphereSchulz(); R, sig, phi, drho = 80.0, 0.4, 0.05, 3.0
    q = np.array([0.6])
    sv = 3 * schulz_moment(R, sig, 2) / schulz_moment(R, sig, 3)
    porod = 2 * math.pi * phi * (drho * 1e10) ** 2 * sv * 1e8 * (q * 1e8) ** -4.0     # all in cm units
    assert comp.eval(q, phi=phi, R=R, sig=sig, drho=drho)[0] == pytest.approx(porod[0], rel=0.08)


def test_percus_yevick_low_q_limit_is_the_compressibility():
    eta = 0.2
    s0 = percus_yevick_sq(np.array([1e-6]), 50.0, eta)[0]
    assert s0 == pytest.approx((1 - eta) ** 4 / (1 + 2 * eta) ** 2, rel=1e-3)
    assert np.allclose(percus_yevick_sq(np.array([0.1, 0.2]), 50.0, 0.0), 1.0)


def test_sphere_derived_sizes():
    d = SphereSchulz().derived(R=100.0, sig=0.0501)
    assert d["D_number"] == pytest.approx(200.0)
    assert d["D_vol_weighted"] == pytest.approx(200.0, rel=0.02)
    assert d["L_vol_scherrer"] == pytest.approx(150.0, rel=0.02)
    assert d["Rg_ensemble"] == pytest.approx(math.sqrt(0.6) * 100.0, rel=0.03)


def test_sphere_registered():
    assert COMPONENTS["sphere_schulz"] is SphereSchulz


# ---------------------------------------------------------------- fit engine
def _synthetic(seed=0):
    rng = np.random.default_rng(seed)
    q = np.logspace(np.log10(0.003), np.log10(0.6), 70)
    truth = dict(B=2e-8, p=4.2, C=0.3, A=50.0, xi=60.0)
    I = truth["B"] * q ** -truth["p"] + truth["C"] + truth["A"] / (1 + (q * truth["xi"]) ** 2) ** 2
    sig = 0.02 * I
    return q, I * (1 + rng.normal(0, 0.02, q.size)), sig, truth


def _terms(with_dab=True):
    t = [TermSpec("power_law", "grain_", values={"B": 1e-8, "p": 4.0}, bounds={"p": (3.0, 5.0), "B": (0, np.inf)}),
         TermSpec("flat_background", "flat_", values={"C": 0.1})]
    if with_dab:
        t.append(TermSpec("dab", "dab_", values={"A": 10.0, "xi": 40.0}, bounds={"xi": (5.0, 2000.0)}))
    return t


def test_fit_recovers_synthetic_parameters_and_bic_prefers_the_true_model():
    q, I, s, tr = _synthetic()
    full = fit_physical("full", q, I, s, _terms(True), floor=0.0)
    base = fit_physical("base", q, I, s, _terms(False), floor=0.0)
    assert full.value("dab_xi") == pytest.approx(tr["xi"], rel=0.1)
    assert full.value("grain_p") == pytest.approx(tr["p"], abs=0.05)
    assert full.value("flat_C") == pytest.approx(tr["C"], rel=0.1)
    assert full.bic < base.bic - 10
    assert full.redchi == pytest.approx(1.0, abs=0.5)
    assert not full.any_at_bound()
    assert full.params["dab_xi"].stderr is not None


def test_fit_flags_parameters_at_a_bound():
    q, I, s, _ = _synthetic()
    t = _terms(False); t[0].bounds["p"] = (3.0, 3.5)                   # true 4.2 is outside
    r = fit_physical("bound", q, I, s, t, floor=0.0)
    assert "grain_p" in r.any_at_bound()


def test_fit_with_resolution_and_fixed_parameter():
    q, I, s, tr = _synthetic(1)
    t = _terms(True); t[1].fixed["C"] = tr["C"]
    r = fit_physical("smear", q, I, s, t, dq=0.01 * q, floor=0.03)
    assert r.smeared and r.n_free == 4
    assert r.params["flat_C"].vary is False and r.value("flat_C") == tr["C"]


def test_prescreen_keeps_the_good_start():
    q, I, s, tr = _synthetic(2)
    bad = [{"dab_xi": 1500.0, "dab_A": 1e-3}] * 6
    good = [{"dab_xi": 55.0, "dab_A": 45.0}]
    r = fit_physical("pre", q, I, s, _terms(True), floor=0.0, starts=bad + good, prescreen=1)
    assert r.value("dab_xi") == pytest.approx(tr["xi"], rel=0.1)


def test_fit_excludes_and_counts_bad_points():
    q, I, s, _ = _synthetic()
    I = I.copy(); I[30] = -1.0
    r = fit_physical("x", q, I, s, _terms(False), q_range=(0.004, 0.5))
    assert r.n_excluded == 1 + int(np.sum((q < 0.004) | (q > 0.5)))
    assert "non-positive" in r.excluded_reason and "outside" in r.excluded_reason


def test_bound_rule_for_zero_lower_bounds():
    from saxs.core.physical_fit import _is_at_bound
    assert not _is_at_bound(0.0056, 0.0, 0.74)          # small volume fraction: a result
    assert _is_at_bound(1e-6, 0.0, 0.74)
    assert _is_at_bound(0.739, 0.0, 0.74)
    assert _is_at_bound(3.0, 3.0, 5.0) and not _is_at_bound(4.0, 3.0, 5.0)
    assert _is_at_bound(29.9, 1.0, 30.0)


def test_residual_runs():
    q = np.logspace(-2, 0, 11)
    r = np.array([0, 2, 2, 2, 0, -3, -3, 0, 0, 1.5, 0])
    runs = residual_runs(q, r, 1.0)
    assert runs[0].sign == 1 and runs[0].n == 3 and runs[0].span == pytest.approx(q[3] / q[1])
    assert {u.sign for u in runs} == {1, -1}


def test_reference_ratio_detects_an_excess_and_ignores_scale():
    q = np.logspace(-3.3, -0.5, 80)
    ref = 1e-8 * q ** -4 + 0.05
    s = 3.0 * (1e-8 * q ** -4) + 0.3 + 30.0 / (1 + (q * 50) ** 2) ** 2     # x3 grain + excess + higher flat
    rr = reference_ratio(q, s, 0.01 * s, q, ref, 0.01 * ref, flat=0.3, flat_ref=0.05, norm_window=(0.0005, 0.001),
                         rel_spread=0.05)
    assert rr.norm == pytest.approx(3.0, rel=0.05)
    best = rr.runs[0]
    assert best.sign == 1 and best.span > 2


def test_reference_ratio_accepts_q_dependent_flat_terms():
    q = np.logspace(-2.5, -0.5, 50)
    oz = 0.5 / (1 + (q * 3.0) ** 2)
    ref = 1e-8 * q ** -4 + 0.05
    s = 2 * 1e-8 * q ** -4 + oz
    rr = reference_ratio(q, s, 0.01 * s, q, ref, 0.01 * ref, flat=oz, flat_ref=0.05, norm_window=(0.003, 0.006))
    assert np.allclose(rr.ratio, 1.0, atol=1e-9) and rr.norm == pytest.approx(2.0)


# ---------------------------------------------------------------- contrast / invariant
def test_electron_density_of_silica_and_sld():
    rho_e = electron_density_cm3({"SiO2": 1.0}, 2.2)
    assert rho_e == pytest.approx(2.2 * N_A * 30 / 60.0843, rel=1e-4)
    assert xray_sld_1e10cm2(rho_e) == pytest.approx(R_E_CM * rho_e / 1e10)


def test_invariant_of_dab_matches_closed_form_and_round_trips_phi():
    A, xi = 20.0, 80.0
    Q = invariant(lambda x: Dab().eval(x, A=A, xi=xi))
    assert Q == pytest.approx(A * math.pi / (4 * xi ** 3), rel=1e-3)
    drho = 5.0
    phi = 0.03
    Q2 = 2 * math.pi ** 2 * phi * (1 - phi) * (drho * 1e10) ** 2 / 1e24
    assert two_phase_product_cm4(Q2) == pytest.approx(phi * (1 - phi) * (drho * 1e10) ** 2)
    assert volume_fraction_from_invariant(Q2, drho) == pytest.approx(phi, rel=1e-9)
    assert math.isnan(volume_fraction_from_invariant(Q2, 0.1))
