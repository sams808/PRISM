"""Tests for saxs.core.descriptors on analytic curves with known answers. Noise is relative Gaussian
with a stated sigma (no Poisson anchoring pitfalls), so significance tests have an exact reference."""
from __future__ import annotations

import numpy as np
import pytest

from saxs.core import descriptors as ds
from saxs.core.curve import Curve

Q = np.geomspace(4e-3, 0.25, 700)


def _curve(I, rel_sigma=0.01, seed=0, name="syn", ratio=None, cls="measurable", usable=None):
    rng = np.random.default_rng(seed)
    sig = rel_sigma * I
    y = I + rng.normal(scale=sig)
    md = {"sample": name, "class_mu_t": cls, "transmission_T": 0.2, "t_eff_mm": 0.3, "mu_t_eff": 1.6,
          "q_min_use": float(Q[0]), "q_max_use": float(Q[-1]),
          "columns": {"ratio": (np.full_like(Q, 50.0) if ratio is None else ratio).tolist(),
                      "usable": (np.ones_like(Q, int) if usable is None else usable).tolist()}}
    return Curve(q=Q, intensity=y, sigma=sig, name=name, metadata=md)


def test_rebin_log_keeps_a_power_law_and_bins_per_decade():
    I = 1e-8 * Q ** -4
    qb, Ib, sb = ds.rebin_log(Q, I, 0.01 * I, per_decade=40)
    decades = np.log10(Q.max() / Q.min())
    assert abs(len(qb) - 40 * decades) < 6
    assert np.all(np.diff(qb) > 0)
    # the rebinned points still follow q^-4 within 0.5 % (geometric-mean q against arithmetic-mean I)
    assert np.allclose(Ib, 1e-8 * qb ** -4, rtol=5e-3)
    assert np.all(sb < 0.01 * Ib)                          # sigma shrinks with the bin population
    q0, I0, s0 = ds.rebin_log(np.array([]), np.array([]), None)
    assert q0.size == 0


def test_local_slopes_and_lowq_exponent_recover_the_exponent():
    I = 2e-9 * Q ** -3.6
    s = ds.local_slopes(Q, I)
    assert np.nanmax(np.abs(s + 3.6)) < 1e-6
    assert ds.slope_at(Q, s, 0.01) == pytest.approx(-3.6, abs=1e-6)
    assert np.isnan(ds.slope_at(Q, s, 5.0))              # far from any point
    n, n_err, K, npts = ds.lowq_exponent(Q, I, 0.01 * I, (4.5e-3, 1.2e-2))
    assert n == pytest.approx(3.6, abs=1e-6) and K == pytest.approx(2e-9, rel=1e-6) and npts > 6
    assert np.isnan(ds.lowq_exponent(Q, I, 0.01 * I, (1.0, 2.0))[0])


def test_plateau_porod_constant_and_invariant():
    K4, C = 3e-10, 5e-3
    I = K4 * Q ** -4 + C
    assert ds.porod_constant(Q, I - C, (4.5e-3, 1.2e-2)) == pytest.approx(K4, rel=1e-9)
    Cm, mad = ds.plateau(Q, I, (0.15, 0.20))
    assert abs(Cm - C) / C < 0.02 and mad < 0.01 * C
    assert np.isnan(ds.plateau(Q, I, (5.0, 6.0))[0])
    # invariant of a pure power law over a window equals the analytic integral of K4 q^-2
    lo, hi = 0.01, 0.1
    Qv = ds.invariant(Q, I, (lo, hi), C=C)
    assert Qv == pytest.approx(K4 * (1 / lo - 1 / hi), rel=2e-3)
    q_exc, q_signed, frac = ds.excess_over_grain(Q, I, K4, C, 0.1)
    assert q_exc < 1e-6 * Qv and abs(q_signed) < 1e-6 * Qv and abs(frac) < 1e-9
    assert np.isnan(ds.excess_over_grain(Q, I, np.nan, C, 0.1)[0])


def test_porod_shoulder_and_kratky_maximum_of_a_guinier_knee():
    Rg = 100.0
    I = 1e-9 * Q ** -4 + 50.0 * np.exp(-(Q * Rg) ** 2 / 3)
    C = 0.0
    sh = ds.porod_shoulders(Q, I, C, (7e-3, 8e-2))
    assert sh, "a Guinier knee makes an I q^4 maximum"
    q_sh, prom, size = sh[0]
    assert 0.02 < q_sh < 0.03                              # I q^4 of a Guinier term peaks at q = sqrt(6)/Rg = 0.0245
    assert size == pytest.approx(2 * np.pi / q_sh / 10, rel=1e-9)
    qk = ds.kratky_maximum(Q, I, C, (7e-3, 8e-2))
    assert 0.014 < qk < 0.021                              # I q^2 peaks at sqrt(3)/Rg = 0.0173
    assert ds.porod_shoulders(Q, 1e-9 * Q ** -4, C, (7e-3, 8e-2)) == []
    assert np.isnan(ds.kratky_maximum(Q, 1e-9 * Q ** -4, C, (7e-3, 8e-2)))


def test_maximum_test_distinguishes_monotonic_bump_and_knee():
    # a monotonic grain power law + plateau: no maximum in any representation
    I = 1e-9 * Q ** -4 + 5e-3
    c = _curve(I, rel_sigma=0.01, seed=1)
    m = ds.maximum_test(c.q, c.intensity, c.sigma)
    assert set(m) == {"I", "Iq2", "Iq4"}
    assert not ds.has_maximum(m) and not m["I"]["significant"] and not m["Iq2"]["significant"]
    # a 30 % Gaussian bump at q = 0.1, where the plateau dominates, is a real maximum of I at that q
    # (on the steep q^-4 flank the same relative bump does not reverse the slope: no maximum there)
    bump = I * (1 + 0.3 * np.exp(-((np.log(Q / 0.1)) ** 2) / (2 * 0.1 ** 2)))
    c2 = _curve(bump, rel_sigma=0.01, seed=2)
    m2 = ds.maximum_test(c2.q, c2.intensity, c2.sigma)
    assert m2["I"]["significant"] and abs(np.log(m2["I"]["q_max"] / 0.1)) < 0.1
    assert m2["I"]["prominence_sigma"] > 3 and ds.has_maximum(m2)
    flank = I * (1 + 0.3 * np.exp(-((np.log(Q / 0.02)) ** 2) / (2 * 0.1 ** 2)))
    assert not ds.maximum_test(Q, flank, 0.01 * flank)["I"]["significant"]
    # the same bump is not significant against a demanding threshold
    assert not ds.maximum_test(c2.q, c2.intensity, c2.sigma, min_prominence_sigma=1e6)["I"]["significant"]
    assert not ds.maximum_test(c2.q, c2.intensity, c2.sigma, min_relative_prominence=0.9)["I"]["significant"]
    # a Guinier knee: no maximum of I, but a significant one of I q^2 -> a Kratky maximum, not a spacing
    knee = 100.0 * np.exp(-(Q * 100.0) ** 2 / 3) + 1e-3
    c3 = _curve(knee, rel_sigma=0.01, seed=3)
    m3 = ds.maximum_test(c3.q, c3.intensity, c3.sigma)
    assert not m3["I"]["significant"] and m3["Iq2"]["significant"]
    assert not ds.has_maximum(m3) and ds.has_kratky_maximum(m3)
    assert abs(np.log(m3["Iq2"]["q_max"] / (np.sqrt(3) / 100.0))) < 0.15
    # an underestimated sigma cannot inflate the significance: the data-driven floor makes a curve with a
    # claimed sigma 2000x too small behave like the same curve with its true sigma
    rng = np.random.default_rng(9)
    noisy = I * (1 + rng.normal(scale=0.2, size=len(Q)))
    m6 = ds.maximum_test(Q, noisy, 1e-4 * I)
    m7 = ds.maximum_test(Q, noisy, 0.2 * I)
    assert m6["I"]["q_max"] == m7["I"]["q_max"] and m6["I"]["prominence_sigma"] < 15
    assert m6["I"]["prominence_sigma"] == pytest.approx(m7["I"]["prominence_sigma"], rel=0.5)
    # too few points: an empty answer, no crash
    m5 = ds.maximum_test(Q[:10], I[:10], None)
    assert m5["I"]["q_max"] is None and not ds.has_maximum(m5)


def test_describe_curve_and_batch_rows():
    I = 1e-9 * Q ** -4 + 1e-4              # plateau small enough not to bend the low-q window
    c = _curve(I, name="a")
    r = ds.describe_curve(c, rho_e=0.7)
    assert r.sample == "a" and r.cls == "measurable" and r.n_use == len(Q) and not r.has_maximum
    v = r.values
    assert v["n_low"] == pytest.approx(4.0, abs=0.05) and v["K4"] == pytest.approx(1e-9, rel=0.03)
    assert v["C_hi"] == pytest.approx(1e-4, rel=0.05) and v["ratio_hi"] == 50.0 and v["ratio_low"] == 50.0
    assert v["K4_over_rhoe2"] == pytest.approx(v["K4"] / 0.49, rel=1e-9) and v["rho_e"] == 0.7
    assert "n_at_0.02" in v and v["n_at_0.02"] == pytest.approx(-4.0, abs=0.1)
    assert np.isfinite(v["I_mm_at_0.02"]) and np.isnan(v["q_shoulder"])
    assert r.rebinned["q"].size > 50 and r.meta["T_used"] == 0.2 and r.meta["t_eff_mm"] == 0.3
    row = r.row()
    for col in ("sample", "cls", "n_use", "T_used", "n_low", "C_hi", "ratio_hi", "K4", "Q_exc", "has_maximum", "has_kratky_maximum",
                "max_q_I", "max_significant_Iq2", "rho_e", "K4_over_rhoe2"):
        assert col in row, col
    # batch: a second curve with a knee and a too-short curve
    knee = 1e-9 * Q ** -4 + 50.0 * np.exp(-(Q * 100.0) ** 2 / 3) + 1e-4
    short = _curve(I, name="short", usable=np.r_[np.ones(10, int), np.zeros(len(Q) - 10, int)])
    results, df = ds.describe_batch([c, _curve(knee, name="b"), short], rho_e={"a": 0.7, "b": 0.8})
    assert len(results) == 3 and list(df["sample"]) == ["a", "b", "short"]
    assert df.set_index("sample").loc["b", "shoulder_size_nm"] > 0 and df.set_index("sample").loc["b", "rho_e"] == 0.8
    assert df.set_index("sample").loc["short", "n_use"] == 10 and np.isnan(df.set_index("sample").loc["short", "n_low"])
    assert results[2].values["n_low"] != results[2].values["n_low"]   # NaN
    assert ds.descriptors_table([]).shape[0] == 0
    _r, df2 = ds.describe_batch([c], rho_e=lambda n: 0.5)
    assert df2.loc[0, "rho_e"] == 0.5


def test_options_round_trip():
    o = ds.DescriptorOptions(per_decade=30, lowq=(5e-3, 1e-2))
    d = o.to_dict()
    back = ds.DescriptorOptions.from_dict({"per_decade": 30, "lowq": [5e-3, 1e-2]})
    assert back.per_decade == 30 and back.lowq == (5e-3, 1e-2) and d["hiq"] == (0.15, 0.20)
    with pytest.raises(KeyError):
        ds.DescriptorOptions.from_dict({"bogus": 1})
