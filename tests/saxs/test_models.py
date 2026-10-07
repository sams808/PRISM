"""Tests for saxs/core/models.py - the SAXS model library. Every numerical check is against an independent closed
form (continuity of the Guinier-Porod construction, the Teubner-Strey argmax / classic form, the Beaucage 1996
prefactor, sphere forward scattering, Porod limit, a dense numerical size average, the Percus-Yevick compressibility)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from saxs.core.models import (
    COMPONENTS, BeaucageTied, BeaucageUnified, Dab, FlatBackground, Guinier, GuinierPorod, LorentzOZ, PowerLaw, SphereSchulz,
    TeubnerStrey, beaucage_polydisperse_prefactor, percus_yevick_sq, schulz_moment, schulz_radii, sphere_amplitude,
    ts_classic_from_physical, ts_physical_from_classic,
)

ALL_COMPONENTS = [cls() for cls in COMPONENTS.values()]


# ---------------------------------------------------------------- generic
@pytest.mark.parametrize("comp", ALL_COMPONENTS, ids=lambda c: c.name)
def test_component_eval_is_vectorized_and_finite(comp):
    q = np.linspace(1e-3, 0.5, 400)
    y = comp.eval(q, **{p.name: p.value for p in comp.params()})
    assert y.shape == q.shape and y.dtype == np.float64 and np.all(np.isfinite(y))


@pytest.mark.parametrize("comp", ALL_COMPONENTS, ids=lambda c: c.name)
def test_component_defaults_within_bounds_and_documented(comp):
    for p in comp.params():
        assert p.min <= p.value <= p.max
        assert p.doc
    assert comp.title and comp.reference


def test_registry_holds_only_the_physical_components():
    assert set(COMPONENTS) == {"flat_background", "power_law", "lorentz_oz", "guinier", "guinier_porod", "beaucage_unified",
                               "beaucage_tied", "dab", "teubner_strey", "sphere_schulz"}
    for name, cls in COMPONENTS.items():
        assert cls().name == name


def test_simple_forms():
    q = np.geomspace(1e-3, 0.5, 50)
    np.testing.assert_allclose(FlatBackground().eval(q, C=2.5), 2.5)
    np.testing.assert_allclose(PowerLaw().eval(q, B=2.0, p=3.5), 2.0 * q ** -3.5, rtol=1e-12)
    np.testing.assert_allclose(LorentzOZ().eval(q, A=3.0, xiL=4.0), 3.0 / (1 + (4 * q) ** 2), rtol=1e-12)
    np.testing.assert_allclose(Dab().eval(q, A=3.0, xi=40.0), 3.0 / (1 + (40 * q) ** 2) ** 2, rtol=1e-12)
    np.testing.assert_allclose(Guinier().eval(q, G=5.0, Rg=80.0), 5.0 * np.exp(-(q * 80) ** 2 / 3), rtol=1e-12)


# ---------------------------------------------------------------- Guinier-Porod
@pytest.mark.parametrize("G,Rg,p", [(100.0, 200.0, 4.0), (5.0, 800.0, 3.2), (1e4, 50.0, 4.3)])
def test_guinier_porod_value_continuity_at_q1(G, Rg, p):
    q1, D = GuinierPorod._q1_D(G, Rg, p)
    assert abs(D / q1 ** p - G * math.exp(-(q1 * Rg) ** 2 / 3)) / (G * math.exp(-(q1 * Rg) ** 2 / 3)) < 1e-10


@pytest.mark.parametrize("G,Rg,p", [(100.0, 200.0, 4.0), (5.0, 800.0, 3.2)])
def test_guinier_porod_slope_continuity_at_q1(G, Rg, p):
    comp = GuinierPorod()
    q1, _ = comp._q1_D(G, Rg, p)
    h = q1 * 1e-4
    f = lambda x: comp.eval(np.array([x]), G=G, Rg=Rg, p=p)[0]      # noqa: E731
    assert (f(q1) - f(q1 - h)) / h == pytest.approx((f(q1 + h) - f(q1)) / h, rel=1e-3)


# ---------------------------------------------------------------- Teubner-Strey
@pytest.mark.parametrize("d,xi", [(1200.0, 3000.0), (900.0, 4500.0), (1600.0, 2600.0)])
def test_teubner_strey_peak_is_at_the_analytic_q_max_and_equals_S(d, xi):
    comp = TeubnerStrey()
    k, kappa = comp._kkappa(d, xi)
    q_max = math.sqrt(k ** 2 - kappa ** 2)
    q = np.linspace(q_max * 0.5, q_max * 1.5, 20001)
    y = comp.eval(q, S=3.0, d=d, xi=xi)
    assert q[np.argmax(y)] == pytest.approx(q_max, rel=1e-4)
    assert comp.eval(np.array([q_max]), S=3.0, d=d, xi=xi)[0] == pytest.approx(3.0, rel=1e-12)
    assert comp.derived(S=3.0, d=d, xi=xi)["q_max"] == pytest.approx(q_max, rel=1e-12)


@pytest.mark.parametrize("d,xi", [(1200.0, 3000.0), (500.0, 10000.0), (2000.0, 5000.0)])
def test_teubner_strey_classic_form_round_trip(d, xi):
    a2, c1, c2 = ts_classic_from_physical(d, xi)
    d2, xi2 = ts_physical_from_classic(a2, c1, c2)
    assert (d2, xi2) == (pytest.approx(d, rel=1e-9), pytest.approx(xi, rel=1e-9))


@pytest.mark.parametrize("d,xi", [(1200.0, 3000.0), (900.0, 4500.0)])
def test_teubner_strey_is_the_classic_form_up_to_normalisation(d, xi):
    comp = TeubnerStrey()
    k, kappa = comp._kkappa(d, xi)
    a2, c1, c2 = ts_classic_from_physical(d, xi)
    q = np.linspace(1e-4, 0.05, 500)
    np.testing.assert_allclose(comp.eval(q, S=7.5, d=d, xi=xi), 7.5 * 4 * k ** 2 * kappa ** 2 / (a2 + c1 * q ** 2 + c2 * q ** 4),
                               rtol=1e-10)


def test_teubner_strey_amphiphilicity():
    d = TeubnerStrey().derived(S=1.0, d=1200.0, xi=3000.0)
    assert d["has_peak"] and -1.0 < d["fa"] < 0.0
    d = TeubnerStrey().derived(S=1.0, d=50000.0, xi=10.0)
    assert not d["has_peak"] and math.isnan(d["q_max"])


# ---------------------------------------------------------------- unified levels
def test_beaucage_tied_is_beaucage_unified_with_the_1996_prefactor():
    q = np.geomspace(1e-3, 0.3, 300)
    G, Rg, d, pf = 20.0, 100.0, 3.5, 0.5
    B = pf * beaucage_polydisperse_prefactor(G, Rg, d)
    np.testing.assert_allclose(BeaucageTied().eval(q, G=G, Rg=Rg, d=d, pf=pf), BeaucageUnified().eval(q, G=G, Rg=Rg, B=B, p=d), rtol=1e-12)
    # d = 4 spheres: B = G 4 / Rg^4 (96/60)^2 Gamma(2)
    assert beaucage_polydisperse_prefactor(1.0, 10.0, 4.0) == pytest.approx(4.0 / 1e4 * (96.0 / 60.0) ** 2, rel=1e-12)
    der = BeaucageTied().derived(G=G, Rg=Rg, d=d, pf=pf)
    assert der["B"] == pytest.approx(B, rel=1e-12) and der["D_eq"] == pytest.approx(2 * Rg * math.sqrt(5 / 3), rel=1e-12)


def test_unified_level_limits():
    q = np.array([1e-4, 1.0])
    y = BeaucageUnified().eval(q, G=10.0, Rg=100.0, B=1e-6, p=4.0)
    assert y[0] == pytest.approx(10.0, rel=1e-3)                 # Guinier plateau
    assert y[1] == pytest.approx(1e-6, rel=1e-3)                 # Porod tail B q^-p (erf -> 1)


# ---------------------------------------------------------------- spheres
def test_sphere_amplitude_limits_and_first_zero():
    assert sphere_amplitude(np.array([0.0]))[0] == pytest.approx(1.0)
    assert sphere_amplitude(np.array([4.493409457909064]))[0] == pytest.approx(0.0, abs=1e-12)


def test_schulz_quadrature_integrates_moments_exactly():
    for sig in (0.05, 0.3, 0.8):
        radii, w = schulz_radii(100.0, sig)
        assert w.sum() == pytest.approx(1.0)
        for k in (1, 3, 6, 8):
            assert np.sum(w * radii ** k) == pytest.approx(schulz_moment(100.0, sig, k), rel=1e-8)
    assert schulz_moment(100.0, 0.3, 2) - 100.0 ** 2 == pytest.approx(30.0 ** 2, rel=1e-9)


def test_sphere_forward_scattering_is_phi_drho2_v2_over_v():
    phi, R, sig, drho = 0.02, 120.0, 0.2, 8.0
    I0 = SphereSchulz().eval(np.array([1e-6]), phi=phi, R=R, sig=sig, drho=drho)[0]
    v2_over_v = (4 / 3 * math.pi) * schulz_moment(R, sig, 6) / schulz_moment(R, sig, 3)
    assert I0 == pytest.approx(phi * drho ** 2 * v2_over_v * 1e-4, rel=1e-6)


def test_sphere_porod_limit():
    R, sig, phi, drho = 80.0, 0.4, 0.05, 3.0
    q = np.array([0.6])
    sv = 3 * schulz_moment(R, sig, 2) / schulz_moment(R, sig, 3)
    porod = 2 * math.pi * phi * (drho * 1e10) ** 2 * sv * 1e8 * (q * 1e8) ** -4.0
    assert SphereSchulz().eval(q, phi=phi, R=R, sig=sig, drho=drho)[0] == pytest.approx(porod[0], rel=0.02)


def test_sphere_matches_dense_numerical_average():
    from scipy.stats import gamma
    q = np.logspace(-2.6, -0.15, 120)
    for R, sig in [(150.0, 0.15), (150.0, 0.4), (40.0, 0.3), (20.0, 0.6), (300.0, 0.1), (300.0, 0.05), (80.0, 0.7)]:
        a = 1 / sig ** 2
        r = np.linspace(gamma.ppf(1e-8, a, scale=R / a), gamma.ppf(1 - 1e-8, a, scale=R / a), 8000)
        w = gamma.pdf(r, a, scale=R / a)
        V = 4 / 3 * math.pi * r ** 3
        F = sphere_amplitude(q[:, None] * r[None, :])
        ref = 0.01 * 64 * (np.sum(w * V ** 2 * F ** 2, axis=1) / np.sum(w * V)) * 1e-4
        got = SphereSchulz().eval(q, phi=0.01, R=R, sig=sig, drho=8.0)
        assert np.max(np.abs(got / ref - 1)) < 0.035, (R, sig)


def test_percus_yevick_low_q_limit_is_the_compressibility():
    eta = 0.2
    assert percus_yevick_sq(np.array([1e-6]), 50.0, eta)[0] == pytest.approx((1 - eta) ** 4 / (1 + 2 * eta) ** 2, rel=1e-3)
    assert np.allclose(percus_yevick_sq(np.array([0.1, 0.2]), 50.0, 0.0), 1.0)


def test_sphere_derived_sizes():
    d = SphereSchulz().derived(R=100.0, sig=0.0501)
    assert d["D_number"] == pytest.approx(200.0)
    assert d["D_vol_weighted"] == pytest.approx(200.0, rel=0.02)
    assert d["L_vol_scherrer"] == pytest.approx(150.0, rel=0.02)
    assert d["Rg_ensemble"] == pytest.approx(math.sqrt(0.6) * 100.0, rel=0.03)
