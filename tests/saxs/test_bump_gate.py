"""Tests for the level-2 visibility gate (saxs.core.unified_levels.bump_evidence and the
descriptors it rests on).

The gate exists because an information criterion is a SIGNIFICANCE test: on a very clean curve a
systematic mismatch of the grain level of a few percent is highly significant, and a second level is
then bought to describe the shape of the first one rather than a population. The gate asks instead
whether a level is VISIBLE, through the one thing a background cannot imitate - the local log-log
slope shallowing at a Guinier knee and steepening again in the level's own Porod region.

Synthetic curves carry Poisson noise anchored to the curve MINIMUM (the high-q plateau), the weakest
feature these assertions depend on; the knee and the grain decade then have far more counts.
"""
from __future__ import annotations

import numpy as np
import pytest

from saxs.core import descriptors as ds
from saxs.core import unified_levels as ul
from saxs.core.composite_models import BeaucageUnified
from saxs.core.curve import Curve

Q = np.geomspace(5e-3, 0.22, 640)
GRAIN = dict(logG1=3.0, Rg1=600.0, logB1=-7.0, d1=4.0, logC=np.log10(0.02), logT=np.log10(0.5))
LEVEL = dict(Rg2=100.0, d2=3.5, logpf2=np.log10(0.5))


def curve(G2: float, seed: int = 0, counts_at_min: float = 400.0, name: str = "syn", q=Q, extra=None) -> Curve:
    """A grain level + background, with a nanoscale level of Guinier amplitude ``G2`` (0 = none)."""
    v = dict(GRAIN, **LEVEL, logG2=np.log10(max(G2, 1e-12)))
    I_true = ul.model(v, q, G2 > 0, True) if extra is None else extra
    rng = np.random.default_rng(seed)
    scale = counts_at_min / float(np.min(I_true))
    counts = rng.poisson(I_true * scale)
    md = {"sample": name, "class_mu_t": "marginal", "per_mm": True,
          "columns": {"usable": np.ones(len(q), int).tolist()}}
    return Curve(q=q, intensity=counts / scale, sigma=np.sqrt(np.maximum(counts, 1.0)) / scale, name=name, metadata=md)


def evidence(c: Curve, options=None):
    """The gate's own view of a curve: prepare it exactly as the fit does, fit the L1-only null."""
    o = options or ul.UnifiedOptions()
    q, I, sig, reason = ul.prepare_curve(c, o)
    assert reason is None
    fits = {t: ul.fit_candidate(q, I, sig, False, t, o) for t in (False, True)}
    tail = min(fits, key=lambda t: fits[t].bic)
    return ul.bump_evidence(q, I, ul._values(fits[tail].params), tail, o)


# ------------------------------------------------------------------ the descriptors it rests on
def test_interior_peaks_ignores_a_maximum_at_the_edge_of_the_range():
    q = np.geomspace(1e-3, 1e-1, 60)
    falling = -np.log10(q)                       # monotone: its maximum is the last point
    assert ds.interior_peaks(q, falling, (1e-3, 1e-1), edge_pts=3, smooth_pts=1) == []
    hump = falling + 0.5 * np.exp(-((np.log10(q) + 2.0) ** 2) / 0.02)   # a bump at q = 1e-2, interior
    pk = ds.interior_peaks(q, hump, (1e-3, 1e-1), edge_pts=3, smooth_pts=1)
    assert pk and pk[0][0] == pytest.approx(1e-2, rel=0.25)


def test_slope_recovery_is_zero_for_a_power_law_with_a_background_and_positive_for_a_level():
    o = ul.UnifiedOptions()
    plain = ul.model(dict(GRAIN, Rg2=100.0, d2=3.5, logpf2=0.0, logG2=-12), Q, False, True)
    r0 = ds.slope_recovery(plain, Q, o.knee_range(), o.steep_limit()) if False else \
        ds.slope_recovery(Q, plain, o.knee_range(), o.steep_limit())
    assert r0["recovery"] == pytest.approx(0.0, abs=1e-6), "a grain power law + flat background + rising tail never steepens again"
    level = ul.model(dict(GRAIN, **LEVEL, logG2=np.log10(20.0)), Q, True, True)
    r1 = ds.slope_recovery(Q, level, o.knee_range(), o.steep_limit())
    assert r1["recovery"] > 1.0
    # the knee it finds is the level's own: q ~ 1/Rg2
    assert r1["q_knee"] == pytest.approx(1.0 / LEVEL["Rg2"], rel=0.6)
    assert r1["q_steep"] > r1["q_knee"]


def test_knee_signature_finds_the_kratky_maximum_of_a_dominant_level():
    o = ul.UnifiedOptions()
    v = dict(GRAIN, **LEVEL, logG2=np.log10(500.0))
    I = ul.model(v, Q, True, True)
    bg = ul.components(v, Q, True, True)["background"]
    sig = ds.knee_signature(Q, I, bg, o.knee_window())
    assert sig["kratky_prominence"] > 0.05 and sig["kratky_q"] == pytest.approx(1.0 / LEVEL["Rg2"], rel=0.8)


# ------------------------------------------------------------------ the gate
def test_no_level_means_no_bump_however_clean_the_curve():
    for counts in (400.0, 40_000.0):             # noise 5 % and 0.5 % at the plateau
        ev = evidence(curve(0.0, counts_at_min=counts))
        assert not ev.present
        assert ev.recovery == pytest.approx(0.0, abs=1e-6)
        assert "steepen" in ev.reason


def test_a_real_level_is_detected_and_the_recovery_grows_with_it():
    recs = [evidence(curve(G2)).recovery for G2 in (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)]
    assert recs == sorted(recs), f"the recovery must not decrease with the level amplitude: {recs}"
    assert recs[0] == pytest.approx(0.0, abs=1e-6) and recs[-1] > 2.0
    ev = evidence(curve(20.0))
    assert ev.present and ev.recovery_sigma > 10
    assert ev.Rg_from_knee_A == pytest.approx(LEVEL["Rg2"], rel=0.6)   # the knee gives the size before any fit


def test_the_gate_is_deterministic_and_blind_to_the_sample_name():
    a, b = curve(20.0, name="alpha"), curve(20.0, name="P8Bi8-12")
    ea, eb = evidence(a), evidence(b)
    assert ea.present and eb.present
    assert ea.recovery == eb.recovery and ea.prominence == eb.prominence


def test_a_maximum_of_the_intensity_is_not_a_guinier_knee():
    """A Guinier knee is a shoulder. A maximum of I is an interference peak or a reduction
    artefact, and the two-level model does not describe either."""
    v = dict(GRAIN, **LEVEL, logG2=np.log10(20.0))
    I = ul.model(v, Q, True, True)
    peaked = I * (1.0 + 3.0 * np.exp(-((np.log10(Q) - np.log10(0.012)) ** 2) / 0.004))
    ev = evidence(curve(0.0, extra=peaked))
    assert not ev.present and not ev.shoulder_ok
    assert ev.q_maximum == pytest.approx(0.012, rel=0.4)
    assert ev.category == "maximum in I", "a maximum also rises; the more specific reason must win"
    assert "maximum" in ev.reason


def test_a_curve_whose_intensity_rises_is_refused_however_big_its_feature():
    """A sum of positive Beaucage levels over a flat background decreases with q. A rise means the
    reduction failed there (a parasitic capillary lobe, an empty subtraction amplified by a
    transmission of 1e-5), and no size may be read from that curve however large its feature. The
    artefact usually sits at the low-q EDGE, where an interior-peak test cannot see it, so this one
    is measured from the first point."""
    v = dict(GRAIN, **LEVEL, logG2=np.log10(50.0))
    I = ul.model(v, Q, True, True)
    # a parasitic lobe just above the first point: the curve rises over three or four points and
    # falls again, so it never makes an interior MAXIMUM the shoulder test could catch
    lifted = I * (1.0 + 1.2 * np.exp(-((np.log10(Q) - np.log10(Q[0] * 1.25)) ** 2) / 0.0008))
    ev = evidence(curve(0.0, extra=lifted))
    assert not ev.present and not ev.monotone_ok, ev.category
    assert ev.rise > 0.10 and ev.rise_q[0] == pytest.approx(Q[0], rel=0.3)
    assert ev.category == "intensity rises" and "reduction failed" in ev.reason
    # the same curve without the lobe keeps its level
    assert evidence(curve(50.0)).present


def test_the_moving_average_keeps_its_normalisation_at_the_ends():
    """np.convolve(..., 'same') divides by the full window everywhere, which drags the first and
    last points down and fakes a rise at the start of a steeply decaying curve - the exact failure
    the monotonicity test would otherwise report on every profile."""
    y = np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0])
    np.testing.assert_allclose(ds.moving_average(y, 3), [9.5, 9.0, 8.0, 7.0, 6.0, 5.5])
    q = np.geomspace(5e-3, 0.1, 60)
    assert ds.largest_rise(q, q ** -4.0, 0.0, 0.1)["rise"] == 0.0
    assert ds.largest_rise(q, q ** -4.0 + 1e3, 1e3, 0.1)["rise"] == 0.0


def test_the_gate_vetoes_level_two_whatever_the_information_criterion_says():
    """The wiring: with l2_allowed False no L2 candidate is eligible, however far it lowers the BIC."""
    o = ul.UnifiedOptions()
    bics = {(False, False): 100.0, (False, True): 80.0, (True, False): 10.0, (True, True): -40.0}
    g2 = {(False, False): None, (False, True): None, (True, False): 5.0, (True, True): 5.0}
    assert ul.forward_select(bics, g2, 100.0, o, l2_allowed=True)[0] == (True, True)
    chosen, path = ul.forward_select(bics, g2, 100.0, o, l2_allowed=False)
    assert chosen == (False, True) and not any(k[0] for k, _ in path)


def test_fit_reports_the_gate_and_carries_it_through_json():
    c = curve(20.0)
    r = ul.fit_unified_levels(c)
    assert r.fitted and r.l2 and r.bump.present and not r.l2_gated
    assert r.candidates["L1_L2_T"]["gated"] is False
    back = ul.UnifiedFitResult.from_json(r.to_json())
    assert back.bump.present and back.bump.recovery == r.bump.recovery
    assert back.bump.window == r.bump.window
    row = ul.result_row(r)
    assert row["bump_present"] and row["bump_recovery"] == r.bump.recovery and row["l2_gated"] is False


def test_require_bump_can_be_switched_off():
    c = curve(20.0)
    o = ul.UnifiedOptions(require_bump=False)
    r = ul.fit_unified_levels(c, o)
    assert r.fitted and not r.l2_gated            # never gated when the gate is off
    assert r.bump.recovery > 1.0                  # but the evidence is still measured and reported
