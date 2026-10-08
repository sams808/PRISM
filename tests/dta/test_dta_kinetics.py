"""Tests for dta_kinetics.py — the rate-dependence fits (Lasocka, Moynihan,
Kissinger, Flynn-Wall-Ozawa, Augis-Bennett).

Every test here runs with no data file and no fixture. That is the whole
point of dta_kinetics importing nothing from the rest of PRISM: the headline
check is a KNOWN-Ea round trip, peak temperatures generated from the
Kissinger relation with Ea = 250 kJ/mol and A = 1e15 /min and fed back
through the fit, and it would be worth much less if it depended on a file
whose "true" activation energy nobody knows.

Two themes recur, and both are about not lying to the user:
  * honest uncertainty at small n — np.polyfit raises at n = 2 and
    scipy.stats.linregress returns a stderr of exactly 0.0, so this module
    hand-rolls OLS and returns nan / None instead;
  * warn, never block — a negative activation energy comes back as the
    number PLUS a warning, with ok=True.
"""
from __future__ import annotations

import builtins
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import linregress
from scipy.stats import t as scipy_t

import dta.dta_kinetics as dk


# =============================================================================
# Synthetic data generators (no files, no fixtures)
# =============================================================================

RATES = (5.0, 10.0, 15.0, 20.0, 30.0)

#: Peak temperatures in K generated from the Kissinger relation with
#: Ea = 250.0 kJ/mol and A = 1.0e15 min^-1 at RATES. Pinned as literals so a
#: change in the generator is caught, and re-derived by bisection in
#: test_kissinger_fixture_matches_the_bisected_generator.
KISSINGER_TP_K = (833.611779355, 849.063490619, 858.363269788,
                  865.082666453, 874.728726793)

EA_TRUE_J = 250_000.0
A_TRUE_PER_MIN = 1.0e15
NU0_TRUE_PER_MIN = 1.0e13
AB_T0_C = 300.0


def _bisect(f, lo: float, hi: float, iters: int = 400) -> float:
    """Plain bisection on a bracketed root; no scipy, so the generator is
    independent of the library the module itself may call."""
    flo = f(lo)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if (fm > 0.0) == (flo > 0.0):
            lo, flo = mid, fm
        else:
            hi = mid
    return 0.5 * (lo + hi)


def kissinger_peaks_K(Ea_J: float = EA_TRUE_J, A_per_min: float = A_TRUE_PER_MIN,
                      rates=RATES):
    """Tp(q) solving ln(q/Tp²) = -Ea/(R·Tp) + ln(A·R/Ea) exactly."""
    out = []
    for q in rates:
        out.append(_bisect(
            lambda T, q=q: (np.log(q / T ** 2) + Ea_J / (dk.R_GAS * T)
                            - np.log(A_per_min * dk.R_GAS / Ea_J)),
            300.0, 2000.0))
    return np.array(out, dtype=float)


def augis_bennett_peaks_K(Ea_J: float = EA_TRUE_J,
                          nu0_per_min: float = NU0_TRUE_PER_MIN,
                          T0_C: float = AB_T0_C, rates=RATES):
    """Tp(q) solving ln(q/(Tp-T0)) = -Ea/(R·Tp) + ln(ν₀) exactly."""
    T0_K = dk.to_kelvin(T0_C)
    out = []
    for q in rates:
        out.append(_bisect(
            lambda T, q=q: (np.log(q / (T - T0_K)) + Ea_J / (dk.R_GAS * T)
                            - np.log(nu0_per_min)),
            T0_K + 1e-6, 3000.0))
    return np.array(out, dtype=float)


def peak_points(T_K, rates=RATES, with_T0: bool = False):
    """RatePoints for a peak-temperature series, temperatures given in K."""
    return [dk.RatePoint(q_C_per_min=float(q), T_C=float(T) - dk.K0,
                         quantity_kind="Tp", label=f"{q:g} K/min",
                         uid=f"row{i + 1}",
                         T0_C=AB_T0_C if with_T0 else float("nan"))
            for i, (q, T) in enumerate(zip(rates, T_K))]


def lasocka_points(A_C: float = 500.0, B: float = 8.0, rates=RATES,
                   time_divisor: float = 1.0, offset: float = 0.0):
    """RatePoints lying exactly on Tg = A + B·ln q.

    `time_divisor` rescales the time base of q (60 turns K/min into K/s) and
    `offset` shifts every temperature, which is how the offset-invariance of
    B and the time-base dependence of A are separated.
    """
    return [dk.RatePoint(q_C_per_min=float(q) / time_divisor,
                         T_C=A_C + B * float(np.log(q)) + offset,
                         quantity_kind="Tg", label=f"{q:g} K/min",
                         uid=f"tg{i + 1}")
            for i, q in enumerate(rates)]


# =============================================================================
# 1. Least-squares statistics
# =============================================================================

def test_ols_line_matches_polyfit_and_linregress_at_n3():
    """At n >= 3 the hand-rolled OLS must be the same fit both libraries give.

    This is what licenses hand-rolling it: the divergence at n = 2 is a
    deliberate honesty fix, not a different algorithm.
    """
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([2.0, 4.1, 5.9])
    fit = dk.ols_line(x, y)

    coeffs, cov = np.polyfit(x, y, 1, cov=True)
    lr = linregress(x, y)

    assert abs(fit.slope - coeffs[0]) <= 1e-12
    assert abs(fit.intercept - coeffs[1]) <= 1e-12
    assert abs(fit.slope_se - np.sqrt(cov[0, 0])) <= 1e-12
    assert abs(fit.intercept_se - np.sqrt(cov[1, 1])) <= 1e-12

    assert abs(fit.slope - lr.slope) <= 1e-12
    assert abs(fit.intercept - lr.intercept) <= 1e-12
    assert abs(fit.slope_se - lr.stderr) <= 1e-12
    assert abs(fit.intercept_se - lr.intercept_stderr) <= 1e-12
    assert abs(fit.r2 - lr.rvalue ** 2) <= 1e-12
    assert fit.n == 3 and fit.dof == 1


def test_ols_line_at_n2_reports_undefined_not_zero():
    """n = 2: nan and None here, while the libraries crash or lie.

    Documents both library behaviours next to ours so the reason this
    function is hand-rolled cannot be lost to a future "simplification".
    """
    x = np.array([1.0, 2.0])
    y = np.array([3.0, 5.0])
    fit = dk.ols_line(x, y)

    assert fit.slope == pytest.approx(2.0)
    assert np.isnan(fit.slope_se)
    assert np.isnan(fit.intercept_se)
    assert np.isnan(fit.cov_slope_intercept)
    assert fit.r2 is None
    assert np.isnan(fit.s_resid)
    assert fit.dof == 0

    with pytest.raises(ValueError):
        np.polyfit(x, y, 1, cov=True)          # RAISES — cannot scale the cov

    lr = linregress(x, y)
    assert lr.stderr == 0.0                    # LIES — a fake zero uncertainty
    assert lr.rvalue ** 2 == 1.0
    assert lr.intercept_stderr == 0.0


def test_ols_line_raises_on_one_point_and_on_zero_x_spread():
    """The two algebraic impossibilities, each naming what to change."""
    with pytest.raises(ValueError) as one:
        dk.ols_line([1.0], [2.0])
    assert "at least 2" in str(one.value)
    assert "heating rate" in str(one.value)

    with pytest.raises(ValueError) as flat:
        dk.ols_line([4.0, 4.0, 4.0], [1.0, 2.0, 3.0])
    assert "identical" in str(flat.value)
    assert "different heating rate" in str(flat.value)


def test_ols_line_drops_non_finite_pairs():
    """A NaN pair is dropped rather than poisoning the whole fit."""
    fit = dk.ols_line([1.0, 2.0, np.nan, 3.0], [2.0, 4.0, 10.0, 6.0])
    assert fit.n == 3
    assert fit.slope == pytest.approx(2.0)


# =============================================================================
# 2. The t quantile
# =============================================================================

def test_t_crit_95_table_matches_scipy_within_5e_4():
    """The hardcoded fallback table is worth what its comment claims."""
    for dof in range(1, 31):
        assert abs(dk._T95_TABLE[dof] - scipy_t.ppf(0.975, dof)) <= 5e-4


def test_t_crit_95_expansion_matches_scipy_within_1e_6():
    """The Cornish-Fisher tail is far more accurate than any DTA series needs."""
    for dof in (31, 40, 60, 100, 250, 1000, 2000):
        assert abs(dk._t95_expansion(dof) - scipy_t.ppf(0.975, dof)) <= 1e-6


def test_t_crit_95_matches_scipy_when_scipy_is_available():
    """With scipy present the exact quantile is used, not the table."""
    assert dk.t_crit_95(3) == pytest.approx(scipy_t.ppf(0.975, 3), abs=1e-12)


def test_t_crit_95_returns_none_at_dof_zero():
    """None, not a number: it is what forces 'no uncertainty' into the text."""
    assert dk.t_crit_95(0) is None
    assert dk.t_crit_95(-1) is None


def test_t_crit_95_without_scipy(monkeypatch):
    """Losing scipy degrades the last digits, never the interval itself."""
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        """Make any scipy import fail, exercising t_crit_95's fallback path."""
        if name.startswith("scipy"):
            raise ImportError("scipy blocked for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    assert dk.t_crit_95(1) == pytest.approx(12.706, abs=1e-3)
    assert dk.t_crit_95(30) == pytest.approx(2.042, abs=1e-3)
    assert dk.t_crit_95(100) == pytest.approx(1.9840, abs=1e-3)


# =============================================================================
# 3. The known-Ea round trip — the headline
# =============================================================================

def test_kissinger_fixture_matches_the_bisected_generator():
    """The pinned Tp literals really are the Kissinger solution at Ea = 250."""
    generated = kissinger_peaks_K()
    assert generated == pytest.approx(np.array(KISSINGER_TP_K), abs=1e-9)


def test_kissinger_recovers_known_ea_exactly():
    """THE headline test: Ea in, Ea out, to 1e-6 kJ/mol."""
    T_K = kissinger_peaks_K()
    res = dk.fit_kissinger(peak_points(T_K))

    assert res.ok is True
    assert res.Ea_kJ_mol == pytest.approx(250.0, abs=1e-6)
    assert res.intercept == pytest.approx(24.2275566806, abs=1e-8)
    assert res.extras["A_per_min"] == pytest.approx(A_TRUE_PER_MIN, rel=1e-9)
    assert res.r2 > 1 - 1e-12
    assert res.n == 5 and res.dof == 3
    assert res.Ea_ci95_kJ_mol is not None


def test_kissinger_predicts_back_the_generating_peak_at_the_reference_rate():
    """The reported T at q_ref inverts the same model it was fitted with."""
    T_K = kissinger_peaks_K()
    res = dk.fit_kissinger(peak_points(T_K), q_ref_C_per_min=10.0)
    assert res.T_at_q_ref_C == pytest.approx(T_K[1] - dk.K0, abs=1e-6)


def test_fwo_and_ozawa_bias_on_kissinger_exact_data():
    """FWO's Doyle correction is small; leaving it out costs 5.7 %.

    Pins WHY c = 1 is not offered as a peer method: on data that is EXACTLY
    Kissinger with Ea = 250, the uncorrected Ozawa slope returns 264.19, and
    the excess is 2·R·T̄p to within 0.05 kJ/mol.
    """
    T_K = kissinger_peaks_K()
    res = dk.fit_fwo(peak_points(T_K))

    assert res.Ea_kJ_mol == pytest.approx(251.1320, abs=1e-3)
    assert res.extras["Ea_c1_kJ_mol"] == pytest.approx(264.1909, abs=1e-3)
    assert res.extras["Ea_c1_kJ_mol"] - 250.0 == pytest.approx(
        2 * dk.R_GAS * float(T_K.mean()) / 1000.0, abs=0.05)
    assert res.extras["doyle_c"] == dk.DOYLE_C
    # +0.45 % with the correction, +5.68 % without it.
    assert (res.Ea_kJ_mol - 250.0) / 250.0 * 100 == pytest.approx(0.45, abs=0.02)
    assert (res.extras["Ea_c1_kJ_mol"] - 250.0) / 250.0 * 100 == pytest.approx(
        5.68, abs=0.02)


def test_fwo_uncorrected_option_still_reports_both_numbers():
    """Whichever form is asked for, extras carries both — no hidden default."""
    T_K = kissinger_peaks_K()
    res = dk.fit_fwo(peak_points(T_K), doyle_correction=False)
    assert res.Ea_kJ_mol == pytest.approx(264.1909, abs=1e-3)
    assert res.extras["Ea_c1_kJ_mol"] == pytest.approx(264.1909, abs=1e-3)
    assert res.extras["Ea_c1052_kJ_mol"] == pytest.approx(251.1320, abs=1e-3)


def test_augis_bennett_recovers_ea_and_nu0_from_its_own_relation():
    """Second known-value round trip: Ea AND the frequency factor nu0."""
    T_K = augis_bennett_peaks_K()
    res = dk.fit_augis_bennett(peak_points(T_K, with_T0=True))

    assert res.ok is True
    assert res.Ea_kJ_mol == pytest.approx(250.0, abs=1e-6)
    assert res.extras["nu0_per_min"] == pytest.approx(NU0_TRUE_PER_MIN, rel=1e-9)
    assert res.extras["ln_nu0"] == pytest.approx(float(np.log(NU0_TRUE_PER_MIN)),
                                                 abs=1e-9)


# =============================================================================
# 4. Lasocka, and the unit law
# =============================================================================

def test_lasocka_recovers_A_and_B_exactly():
    """Points lying exactly on Tg = A + B.ln q return A, B and zero scatter."""
    res = dk.fit_lasocka(lasocka_points(A_C=500.0, B=8.0))
    assert res.ok is True
    assert res.lasocka_A_C == pytest.approx(500.0, abs=1e-12)
    assert res.lasocka_B == pytest.approx(8.0, abs=1e-12)
    assert res.s_resid == 0.0
    assert res.extras["Ea_g_from_B_kJ_mol"] > 0.0


def test_lasocka_B_is_offset_invariant_and_A_is_not():
    """B survives the temperature unit; A moves by exactly the K0 offset.

    This is the half of the unit law that says a SLOPE in temperature is
    offset-free. The other half — that a heating RATE is offset-free too, so
    it is never converted — is pinned by
    test_kelvin_offset_appears_twice_in_the_module.
    """
    points = lasocka_points()
    in_C = dk.fit_lasocka(points)
    in_K = dk.fit_lasocka(points, temperature_in_K=True)

    # bit-identical, and ONLY because `lasocka_points` is exactly collinear:
    # with residuals, ybar is not shifted by exactly K0 in floating point and
    # the two Bs differ by ~1e-15 relative. See
    # test_lasocka_B_offset_invariance_is_machine_precision_not_bit_identity.
    assert in_K.lasocka_B == in_C.lasocka_B
    assert in_K.slope == in_C.slope
    assert in_K.intercept - in_C.intercept == pytest.approx(dk.K0, abs=1e-9)
    assert in_C.lasocka_A_K - in_C.lasocka_A_C == pytest.approx(dk.K0, abs=1e-9)
    assert in_K.lasocka_A_C == pytest.approx(in_C.lasocka_A_C, abs=1e-9)


def test_lasocka_A_shifts_with_time_base_not_with_temperature_unit():
    """q in K/s instead of K/min moves A by -B·ln 60 and leaves B alone."""
    per_second = dk.fit_lasocka(lasocka_points(time_divisor=60.0))
    assert per_second.lasocka_B == pytest.approx(8.0, abs=1e-12)
    assert per_second.lasocka_A_C == pytest.approx(532.754756498, abs=1e-6)
    assert per_second.lasocka_A_C - 500.0 == pytest.approx(
        8.0 * float(np.log(60.0)), abs=1e-9)


def test_kelvin_offset_appears_twice_in_the_module():
    """Exactly two mentions of the kelvin offset in the whole source file.

    A second conversion creeping in anywhere — a display-unit toggle reaching
    the fit, a `value_unit` field, a to_kelvin call inside a rate resolver —
    produces an offset that still looks like a plausible Tg while making Ea
    wrong by roughly a factor of three. Counting the literal is the cheapest
    structural guard against that, and it is why PLAUSIBLE_T_C is written
    with -K0 rather than a second copy of the number.
    """
    src = Path(dk.__file__).read_text(encoding="utf-8")
    assert src.count("273.15") == 2
    assert dk.PLAUSIBLE_T_C[0] == -dk.K0


def test_to_kelvin_handles_scalars_and_arrays():
    """The single conversion point must not turn a scalar into an array."""
    assert dk.to_kelvin(0.0) == pytest.approx(dk.K0)
    assert isinstance(dk.to_kelvin(25.0), float)
    out = dk.to_kelvin(np.array([0.0, 100.0]))
    assert isinstance(out, np.ndarray)
    assert out == pytest.approx(np.array([dk.K0, 100.0 + dk.K0]))


def test_temperatures_are_converted_and_rates_are_not():
    """Both halves of the unit law, read straight off a fitted result.

    x must be 1/(T_C + K0) — the temperature IS converted — while y must be
    built from the rate as given, with no offset anywhere near it.
    """
    T_K = kissinger_peaks_K()
    points = peak_points(T_K)
    res = dk.fit_kissinger(points)

    expected_x = 1.0 / np.array([p.T_C + dk.K0 for p in points])
    expected_y = np.log(np.array([p.q_C_per_min for p in points]) / T_K ** 2)
    assert res.x == pytest.approx(expected_x, rel=0, abs=0)
    assert res.y == pytest.approx(expected_y, rel=0, abs=0)
    assert res.temp_basis == "K (converted)"
    assert "kelvin_basis" in res.codes


# =============================================================================
# 5. Moynihan vs the Kissinger form on Tg
# =============================================================================

def test_moynihan_and_kissinger_tg_differ_by_two_R_Tg():
    """The Tg² term is worth exactly -2·R·T̄g, not noise."""
    points = lasocka_points()
    moy = dk.fit_moynihan(points)
    kis = dk.fit_kissinger_tg(points)

    difference = kis.Ea_kJ_mol - moy.Ea_kJ_mol
    Tg_mean_K = float(np.mean([p.T_K for p in points]))
    assert difference == pytest.approx(-13.19, abs=0.05)
    assert difference == pytest.approx(-2.0 * dk.R_GAS * Tg_mean_K / 1000.0,
                                       abs=0.05)


def test_fragility_m_reported_for_tg_fits():
    """m = Ea/(ln10·R·T̄g), reported wherever an Ea(Tg) is."""
    points = lasocka_points()
    moy = dk.fit_moynihan(points)
    assert moy.fragility_m == pytest.approx(43.0, abs=0.1)
    assert moy.fragility_m == pytest.approx(
        moy.Ea_kJ_mol * 1000.0 / (dk.LN10 * dk.R_GAS * moy.extras["Tg_mean_K"]),
        rel=1e-12)
    assert dk.fit_kissinger_tg(points).fragility_m is not None
    # A peak-temperature fit is not a glass transition; no m is claimed.
    assert dk.fit_kissinger(peak_points(kissinger_peaks_K())).fragility_m is None


def test_tg_ea_caveat_and_lasocka_note_are_substantive_text():
    """The two exported caveats are real paragraphs, not placeholders."""
    assert "not an activation energy in the Kissinger sense" in dk.TG_EA_CAVEAT
    assert "fragility" in dk.TG_EA_CAVEAT
    assert "B must be > 0" in dk.LASOCKA_B_NOTE


# =============================================================================
# 6. Warn, never block
# =============================================================================

def test_negative_ea_warns_and_still_returns_the_number():
    """The user asked to be able to process such data, and to be told.

    A peak temperature that FALLS with rate gives a negative Ea. That is
    physically impossible, and it is exactly the case the standing
    instruction covers: report the number, attach the warning, ok stays True.
    """
    falling = [dk.RatePoint(q_C_per_min=q, T_C=T, quantity_kind="Tp",
                            uid=f"n{i}")
               for i, (q, T) in enumerate([(5.0, 700.0), (10.0, 650.0),
                                           (20.0, 600.0), (30.0, 560.0)])]
    res = dk.fit_kissinger(falling)

    assert res.ok is True
    assert "negative_ea" in res.codes
    assert np.isfinite(res.Ea_kJ_mol)
    assert res.Ea_kJ_mol < 0
    assert "positive_slope_kissinger" in res.codes
    assert res.max_severity == "critical"
    negative = [w for w in res.warnings if w.code == "negative_ea"][0]
    assert negative.severity == "critical"
    assert "do not quote it" in negative.message.lower()
    assert negative.row_uids == ("n0", "n1", "n2", "n3")


def test_negative_lasocka_b_warns_and_still_returns_the_number():
    """A Tg that falls with rate: flagged as critical, fitted all the same."""
    falling = [dk.RatePoint(q_C_per_min=q, T_C=T, uid=f"b{i}")
               for i, (q, T) in enumerate([(5.0, 620.0), (10.0, 610.0),
                                           (20.0, 600.0)])]
    res = dk.fit_lasocka(falling)
    assert res.ok is True
    assert "negative_lasocka_b" in res.codes
    assert res.lasocka_B < 0
    assert np.isfinite(res.lasocka_A_C)
    # A cross-check Ea_g from a negative B would be nonsense, so none is made.
    assert "Ea_g_from_B_kJ_mol" not in res.extras


def test_two_points_fits_and_warns_two_points_only():
    """n = 2 fits, and says in every rendering that it cannot bound itself."""
    res = dk.fit_lasocka([dk.RatePoint(10.0, 500.0, uid="p1"),
                          dk.RatePoint(20.0, 510.0, uid="p2")])
    assert res.ok is True
    assert np.isfinite(res.slope)
    assert np.isnan(res.slope_se)
    assert res.r2 is None
    assert res.t_crit_95 is None
    assert res.Ea_ci95_kJ_mol is None
    assert "two_points_only" in res.codes
    assert "no uncertainty" in res.annotation()
    assert "no uncertainty" in res.report()


def test_dof_one_is_information_not_an_alarm():
    """dof = 1 is a fact about the design, so it is a note, not a warning."""
    res = dk.fit_lasocka(lasocka_points(rates=(5.0, 10.0, 20.0)))
    assert "dof_one" in res.codes
    note = [w for w in res.notes if w.code == "dof_one"][0]
    assert note.severity == "info"
    assert note.glyph == "ℹ"


def test_low_r2_warns_and_respects_an_overridden_threshold():
    """The layer above can pass its own r2_min through the fit."""
    scattered = [dk.RatePoint(q, T) for q, T in
                 [(5.0, 500.0), (10.0, 540.0), (20.0, 505.0), (30.0, 560.0)]]
    assert "low_r2" in dk.fit_lasocka(scattered).codes
    assert "low_r2" not in dk.fit_lasocka(scattered, r2_min=0.0).codes


def test_augis_bennett_drops_points_without_T0_and_says_so():
    """The drop is scoped to this fit and names Kissinger as the way out."""
    T_K = augis_bennett_peaks_K()
    points = peak_points(T_K, with_T0=True)
    points[2] = dk.RatePoint(q_C_per_min=points[2].q_C_per_min,
                             T_C=points[2].T_C, quantity_kind="Tp",
                             label=points[2].label, uid=points[2].uid)

    res = dk.fit_augis_bennett(points)
    assert res.ok is True
    assert res.n == 4
    assert res.n_total == 5
    assert "no_t0" in res.codes
    warning = [w for w in res.warnings if w.code == "no_t0"][0]
    assert "Kissinger" in warning.message
    assert warning.row_uids == ("row3",)
    # The same points still fit fine with Kissinger, which needs no T0.
    assert dk.fit_kissinger(points).n == 5


def test_augis_bennett_drops_a_T0_above_the_peak():
    """T0 must be the ramp START; a T0 above the peak makes ln() undefined."""
    T_K = augis_bennett_peaks_K()
    points = peak_points(T_K, with_T0=True)
    points[0].T0_C = float(T_K[0] - dk.K0) + 50.0
    res = dk.fit_augis_bennett(points)
    assert res.ok is True
    assert res.n == 4
    assert "no_t0" in res.codes


def test_no_fit_is_reported_not_raised():
    """Arithmetic impossibility comes back as ok=False with a reason."""
    for res in (dk.fit_kissinger([]),
                dk.fit_lasocka([dk.RatePoint(10.0, 500.0)]),
                dk.fit_lasocka([dk.RatePoint(10.0, 500.0),
                                dk.RatePoint(10.0, 520.0)]),
                dk.fit_augis_bennett(peak_points(kissinger_peaks_K()))):
        assert res.ok is False
        assert res.message
        assert res.n == 0
        assert res.x.size == 0
        assert "No fit" in res.report() or "No fit" in res.message \
            or "undefined" in res.message


def test_no_fit_message_says_what_to_change():
    """Each arithmetic dead end names the user action that clears it."""
    too_few = dk.fit_kissinger([dk.RatePoint(10.0, 500.0, quantity_kind="Tp")])
    assert "need 2" in too_few.message
    one_rate = dk.fit_lasocka([dk.RatePoint(10.0, 500.0),
                               dk.RatePoint(10.0, 520.0)])
    assert "different heating rate" in one_rate.message
    no_t0 = dk.fit_augis_bennett(peak_points(kissinger_peaks_K()))
    assert "Kissinger" in no_t0.message


def test_no_fit_never_raises_on_hostile_input():
    """Nothing a caller can put in a RatePoint escapes as an exception."""
    hostile = [dk.RatePoint(float("nan"), 500.0),
               dk.RatePoint(10.0, float("nan")),
               dk.RatePoint(0.0, 500.0),
               dk.RatePoint(10.0, -400.0),
               dk.RatePoint(float("inf"), float("inf"))]
    for name in dk.KINETIC_METHODS:
        res = dk.fit_points(hostile, name)
        assert res.ok is False
        assert res.message


def test_a_cooling_rate_is_fitted_on_its_magnitude():
    """Negative rates are a provenance warning upstairs, not a refusal here."""
    T_K = kissinger_peaks_K()
    cooling = [dk.RatePoint(q_C_per_min=-float(q), T_C=float(T) - dk.K0,
                            quantity_kind="Tp")
               for q, T in zip(RATES, T_K)]
    res = dk.fit_kissinger(cooling)
    assert res.ok is True
    assert res.Ea_kJ_mol == pytest.approx(250.0, abs=1e-6)


def test_no_numpy_warning_escapes_any_fit_at_n2_or_n3():
    """R5: a RankWarning or a divide-by-zero surfacing as a crash would be
    the second-worst failure after a fake zero uncertainty."""
    T_K = kissinger_peaks_K()
    for count in (2, 3):
        tp = peak_points(T_K[:count], rates=RATES[:count], with_T0=True)
        tg = lasocka_points(rates=RATES[:count])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for name in ("kissinger", "fwo", "augis_bennett"):
                res = dk.fit_points(tp, name)
                assert res.ok is True, (name, count, res.message)
                res.report()
                res.annotation()
                res.csv_fit_row()
                res.csv_point_rows()
            for name in ("lasocka", "moynihan", "kissinger_tg"):
                res = dk.fit_points(tg, name)
                assert res.ok is True, (name, count, res.message)
                res.report()
                res.annotation()


# =============================================================================
# 7. Warning plumbing
# =============================================================================

def test_series_warning_renders_four_ways():
    """glyph / line / short — the renderings a warning must survive intact."""
    w = dk.SeriesWarning(code="demo", severity="warn",
                         message="Something is wrong with the data " * 4,
                         row_uids=("a",), detail={"k": 1})
    assert w.glyph == "⚠"
    assert w.line().startswith("⚠ ")
    assert len(w.short()) <= 60
    assert w.short().endswith("…")
    info = dk.SeriesWarning(code="i", severity="info", message="Just so you know.")
    assert info.glyph == "ℹ"
    assert info.line() == "ℹ Just so you know."
    assert info.short() == "Just so you know."


def test_series_warning_line_does_not_double_the_glyph():
    """A caller whose message is already glyphed must not get a doubled one."""
    w = dk.SeriesWarning(code="demo", severity="warn", message="⚠ Already glyphed.")
    assert w.line() == "⚠ Already glyphed."
    assert w.short() == "Already glyphed."


def test_max_severity_and_sort_warnings_put_the_worst_first():
    """Truncation anywhere must keep the worst problem, not the first one."""
    ws = [dk.SeriesWarning("z_info", "info", "i"),
          dk.SeriesWarning("m_warn", "warn", "w"),
          dk.SeriesWarning("a_crit", "critical", "c"),
          dk.SeriesWarning("b_warn", "warn", "w2")]
    assert dk.max_severity(ws) == "critical"
    assert dk.max_severity([]) == ""
    assert dk.max_severity([ws[0]]) == "info"
    assert [w.code for w in dk.sort_warnings(ws)] == [
        "a_crit", "b_warn", "m_warn", "z_info"]


def test_warning_codes_always_include_the_series_wide_ones():
    """A single-row extract must never look innocent."""
    ws = [dk.SeriesWarning("series_wide", "warn", "m"),
          dk.SeriesWarning("row_only", "warn", "m", row_uids=("r1",)),
          dk.SeriesWarning("other_row", "warn", "m", row_uids=("r2",))]
    assert dk.warning_codes(ws) == ["other_row", "row_only", "series_wide"]
    assert dk.warning_codes(ws, "r1") == ["row_only", "series_wide"]
    assert dk.warning_codes(ws, "r9") == ["series_wide"]


def test_warnings_and_notes_split_by_severity():
    """One mechanical rule: critical/warn are warnings, info are notes."""
    res = dk.fit_lasocka(lasocka_points(rates=(5.0, 10.0, 20.0), B=-3.0))
    assert all(w.severity != "info" for w in res.warnings)
    assert all(w.severity == "info" for w in res.notes)
    assert {w.code for w in res.all_warnings} == set(res.codes)


def test_annotation_includes_the_worst_warnings_and_counts_the_rest():
    """The figure corner is finite, so it says how many it could not show."""
    res = dk.fit_lasocka([dk.RatePoint(20.0, 620.8), dk.RatePoint(15.0, 657.1),
                          dk.RatePoint(10.0, 634.1)])
    text = res.annotation(max_warnings=1)
    assert "negative_lasocka_b" not in text            # codes never leak to users
    assert "NEGATIVE" in text                          # the worst one is kept
    assert "more warning" in text
    assert res.annotation(max_warnings=99).count("\n") > text.count("\n")


def test_report_lists_every_warning_as_a_line():
    """The panel is the record: nothing is elided there."""
    res = dk.fit_lasocka([dk.RatePoint(20.0, 620.8), dk.RatePoint(15.0, 657.1),
                          dk.RatePoint(10.0, 634.1)])
    text = res.report()
    assert len(res.all_warnings) >= 3
    for w in res.all_warnings:
        assert w.line() in text


# =============================================================================
# 8. The real-data Lasocka fit, as literal values (no file needed)
# =============================================================================

def test_real_bm_triple_lasocka_exact_numbers():
    """The three per-heat Tg values measured on BM-0-cycles.txt.

    Frozen as literals so the numeric path is pinned without the 19k-row
    file: heats 1, 3 and 4 gave 620.8 / 657.1 / 634.1 °C at 20 / 15 / 10
    °C/min. The fit is a textbook example of what this module must NOT
    refuse and must NOT dress up: B is negative, R² is 0.07, and every one
    of those facts comes back as a warning riding on a real number.
    """
    res = dk.fit_lasocka([
        dk.RatePoint(q_C_per_min=20.0, T_C=620.8, label="heat 1", uid="h1"),
        dk.RatePoint(q_C_per_min=15.0, T_C=657.1, label="heat 3", uid="h3"),
        dk.RatePoint(q_C_per_min=10.0, T_C=634.1, label="heat 4", uid="h4"),
    ])

    assert res.ok is True
    assert res.lasocka_B == pytest.approx(-14.2053, abs=1e-4)
    assert res.lasocka_B_se == pytest.approx(50.7870, abs=1e-3)
    assert res.lasocka_A_C == pytest.approx(675.2443, abs=1e-3)
    assert res.lasocka_A_se == pytest.approx(136.3069, abs=1e-3)
    assert res.lasocka_A_K == pytest.approx(948.3943, abs=1e-3)
    assert res.r2 == pytest.approx(0.072558, abs=1e-6)
    assert res.s_resid == pytest.approx(25.0117, abs=1e-3)
    assert res.dof == 1
    assert res.t_crit_95 == pytest.approx(12.7062, abs=1e-3)
    assert set(res.codes) >= {"negative_lasocka_b", "low_r2", "dof_one"}
    assert res.rows_used == ["h1", "h3", "h4"]


# =============================================================================
# 9. Registry, dispatch, and what the Qt layer reads off a result
# =============================================================================

def test_every_registry_entry_is_complete_and_callable():
    """The Qt combo is built from this dict, so no entry may be half-filled."""
    for key, meta in dk.KINETIC_METHODS.items():
        assert callable(meta["fn"])
        assert meta["label"] and meta["equation"]
        assert meta["x_label"] and meta["y_label"]
        assert meta["temp_basis"] in ("°C", "K (converted)")
        assert set(meta["needs"]) <= set(dk.QUANTITY_KINDS)
        assert isinstance(meta["needs_T0"], bool)
    assert dk.KINETIC_METHODS["augis_bennett"]["needs_T0"] is True
    assert all(not m["needs_T0"] for k, m in dk.KINETIC_METHODS.items()
               if k != "augis_bennett")


def test_default_method_for_each_quantity():
    """Tg defaults to the plot the user asked for by name; Tp to Kissinger."""
    assert dk.default_method_for("Tg") == "lasocka"
    assert dk.default_method_for("Tp") == "kissinger"
    assert dk.default_method_for("tp") == "kissinger"
    assert dk.default_method_for("nonsense") == "lasocka"     # never raises
    assert dk.default_method_for("") == "lasocka"


def test_fit_points_dispatches_by_key_and_by_label():
    """A combo box may hand back either the key or the display label."""
    points = lasocka_points()
    by_key = dk.fit_points(points, "lasocka")
    by_label = dk.fit_points(points, dk.KINETIC_METHODS["lasocka"]["label"])
    assert by_key.slope == by_label.slope
    assert by_key.method == "lasocka"


def test_fit_points_reports_an_unknown_method_instead_of_raising():
    """A batch loop must not lose five good series to one bad method name."""
    res = dk.fit_points(lasocka_points(), "kissingre")
    assert res.ok is False
    assert "Unknown method" in res.message
    for key in dk.KINETIC_METHODS:
        assert key in res.message


def test_fit_points_reports_a_bad_keyword_instead_of_raising():
    """Same contract for an option the chosen model does not take."""
    res = dk.fit_points(lasocka_points(), "lasocka", doyle_correction=True)
    assert res.ok is False
    assert "does not accept those options" in res.message


def test_every_fit_carries_its_own_axis_labels_and_equation():
    """The Qt layer must never re-derive an axis label from the method name."""
    tp = peak_points(augis_bennett_peaks_K(), with_T0=True)
    tg = lasocka_points()
    for key, meta in dk.KINETIC_METHODS.items():
        points = tp if meta["needs"] == ("Tp",) else tg
        res = dk.fit_points(points, key)
        assert res.ok is True, (key, res.message)
        assert res.x_label == meta["x_label"]
        assert res.y_label == meta["y_label"]
        assert res.equation == meta["equation"]
        assert res.method_label == meta["label"]
        assert res.temp_basis == meta["temp_basis"]
        assert res.x.shape == res.y.shape == res.residuals.shape
        assert len(res.labels) == res.n


def test_counts_distinguish_handed_in_usable_and_fitted():
    """n_total / n_included / n_fitted each answer a different question."""
    points = peak_points(augis_bennett_peaks_K(), with_T0=True)
    points[0].T_C = float("nan")                    # unusable arithmetic
    points[1].T0_C = float("nan")                   # usable, but not for A-B
    res = dk.fit_augis_bennett(points)
    assert (res.n_total, res.n_included, res.n_fitted) == (5, 4, 3)
    assert "3 of 5 points fitted" in res.report()


# =============================================================================
# 10. CSV renderings
# =============================================================================

def test_csv_fit_row_leaves_irrelevant_columns_blank_not_zero():
    """A blank cell means 'not applicable'; a 0 would be a claim."""
    lasocka = dk.fit_lasocka(lasocka_points()).csv_fit_row()
    assert lasocka["Ea_kJ_mol"] == ""
    assert lasocka["lasocka_B"] == pytest.approx(8.0)
    assert lasocka["fit_ok"] is True

    kissinger = dk.fit_kissinger(peak_points(kissinger_peaks_K())).csv_fit_row()
    assert kissinger["lasocka_B"] == ""
    assert kissinger["Ea_kJ_mol"] == pytest.approx(250.0, abs=1e-6)
    for value in kissinger.values():
        assert value != "nan"


def test_csv_fit_row_of_a_two_point_fit_blanks_the_uncertainties():
    """The n = 2 honesty rule survives the trip into the export."""
    row = dk.fit_lasocka([dk.RatePoint(10.0, 500.0),
                          dk.RatePoint(20.0, 510.0)]).csv_fit_row()
    assert row["slope_se"] == ""
    assert row["r_squared"] == ""
    assert row["s_resid"] == ""
    assert row["t_crit_95"] == ""
    assert row["dof"] == 0
    assert "two_points_only" in row["warning_codes"]


def test_csv_point_rows_repeat_the_warnings_on_every_row():
    """A single-row extract must carry the whole fit's objections."""
    res = dk.fit_lasocka([
        dk.RatePoint(20.0, 620.8, uid="h1"),
        dk.RatePoint(15.0, 657.1, uid="h3"),
        dk.RatePoint(10.0, 634.1, uid="h4")])
    rows = res.csv_point_rows()
    assert len(rows) == 3
    for i, row in enumerate(rows):
        assert row["point_index"] == i + 1
        assert "negative_lasocka_b" in row["warning_codes"]
        assert row["max_severity"] == "critical"
        assert row["warning_messages"]
    assert [r["uid"] for r in rows] == ["h1", "h3", "h4"]


def test_csv_row_of_a_failed_fit_still_describes_the_series():
    """A batch export gets a row even when the arithmetic was impossible."""
    row = dk.fit_kissinger([]).csv_fit_row()
    assert row["fit_ok"] is False
    assert row["fit_message"]
    assert row["method"] == "kissinger"
    assert row["n_rows_fitted"] == 0


# ===========================================================================
# Adversarial-review regressions (2026-09-26)
# ===========================================================================

def test_lasocka_B_offset_invariance_is_machine_precision_not_bit_identity():
    """Adding K0 to every y does not shift ybar by EXACTLY K0 in floating
    point, so `y - ybar` differs in its last bits wherever the residuals are
    non-zero and B differs by ~1e-15 relative.

    The module docstring and `fit_lasocka` both claimed "bit-identical", and
    the `==` assertion in
    test_lasocka_B_is_offset_invariant_and_A_is_not survived only because
    `lasocka_points` is exactly collinear. Measured on a scattered fit:
    6.37839941770457841 in °C against 6.37839941770458907 in K.
    """
    rng = np.random.default_rng(3)
    noise = rng.normal(0.0, 0.7, len(RATES))
    scattered = [dk.RatePoint(q_C_per_min=float(q),
                              T_C=500.0 + 8.0 * float(np.log(q)) + float(n),
                              quantity_kind="Tg", uid=f"s{i}")
                 for i, (q, n) in enumerate(zip(RATES, noise))]
    in_C = dk.fit_lasocka(scattered)
    in_K = dk.fit_lasocka(scattered, temperature_in_K=True)

    assert in_K.lasocka_B == pytest.approx(in_C.lasocka_B, rel=1e-12)
    assert abs(in_K.lasocka_B - in_C.lasocka_B) < 1e-12 * abs(in_C.lasocka_B)
    assert in_K.lasocka_A_C == pytest.approx(in_C.lasocka_A_C, abs=1e-9)

    # the exactly-collinear fixture IS bit-identical, which is why the
    # stronger assertion elsewhere in this file is safe as long as it stays
    # on that fixture
    exact = lasocka_points()
    assert (dk.fit_lasocka(exact, temperature_in_K=True).lasocka_B
            == dk.fit_lasocka(exact).lasocka_B)

    src = Path(dk.__file__).read_text(encoding="utf-8")
    assert "bit-identical fitted" not in src


def test_a_point_the_arithmetic_cannot_take_is_named_not_only_counted():
    """`RateRow.fittable` and `_usable_points` disagree by exactly one case:
    a temperature at or below absolute zero.

    Lasocka is the one registry model whose `temp_basis` is °C — it fits T_C
    directly and never touches T_K — so for Lasocka that drop is a physical
    objection enforced by an arithmetic rule. Before the fix the point simply
    vanished: `n_fitted = 2` against `n_total = 3` with no warning naming it,
    which M0 forbids.
    """
    points = [dk.RatePoint(q_C_per_min=10.0, T_C=-500.0, quantity_kind="Tg",
                           label="sub-zero row", uid="p1"),
              dk.RatePoint(q_C_per_min=20.0, T_C=600.0, quantity_kind="Tg", uid="p2"),
              dk.RatePoint(q_C_per_min=40.0, T_C=610.0, quantity_kind="Tg", uid="p3")]
    res = dk.fit_lasocka(points)

    assert res.ok is True                       # warned, never blocked
    assert (res.n_fitted, res.n_total) == (2, 3)
    dropped = next(w for w in res.all_warnings if w.code == "dropped_points")
    assert dropped.severity == "warn"
    assert "sub-zero row" in dropped.message
    assert dropped.detail == {"n_dropped": 1, "n_total": 3, "method": "lasocka"}
    assert dropped.row_uids == ("p1",)

    # a zero rate is the same story on an Arrhenius model
    zero = [dk.RatePoint(q_C_per_min=0.0, T_C=600.0, quantity_kind="Tp", uid="z"),
            dk.RatePoint(q_C_per_min=10.0, T_C=610.0, quantity_kind="Tp", uid="a"),
            dk.RatePoint(q_C_per_min=20.0, T_C=620.0, quantity_kind="Tp", uid="b")]
    kis = dk.fit_kissinger(zero)
    assert kis.n_fitted == 2
    assert "dropped_points" in {w.code for w in kis.all_warnings}

    # a clean series says nothing
    assert "dropped_points" not in {w.code for w in
                                    dk.fit_lasocka(lasocka_points()).all_warnings}
