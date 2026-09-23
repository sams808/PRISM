"""Tests for saxs.core.sector_batch: options, loading classes, mu sources, batch reduction of the
synthetic campaign, the .dat round trip and the summary table."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from saxs.core import edf_sectors as es
from saxs.core import frames as fr
from saxs.core import sector_batch as sb

import saxs2_synthetic as syn  # noqa: E402  (tests/saxs is on sys.path under pytest)

pytestmark = pytest.mark.skipif(not es.HAVE_FABIO, reason="fabio not installed")


@pytest.fixture(scope="module")
def reduced(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("batch"))
    syn.make_campaign(root)
    records = fr.scan_frames(root, diagnostics=False)
    pairing = fr.pair_frames(records)
    out = os.path.join(root, "reduced")
    mu = sb.MuSource.from_table({"S1": 10.0, "S2": 10.0, "S3_0p5mm": 20.0})       # S4 has no mu: no per-mm curve
    results = sb.reduce_pairs(pairing.pairs, sb.ReductionOptions(), mu, out_dir=out)
    return root, out, {r.name: r for r in results}


def test_options_round_trip_and_unknown_key():
    o = sb.ReductionOptions(axis_exclusion_deg=25.0, bar_rect_px=(30.0, 10.0))
    d = o.to_dict()
    assert d["axis_exclusion_deg"] == 25.0 and d["plateau_bands_px"] == ((6, 12), (12, 20))
    back = sb.ReductionOptions.from_dict({"axis_exclusion_deg": 25.0, "bar_rect_px": [30.0, 10.0], "plateau_bands_px": [[6, 12], [12, 20]]})
    assert back.bar_rect_px == (30.0, 10.0) and back.plateau_bands_px == ((6, 12), (12, 20))
    with pytest.raises(KeyError):
        sb.ReductionOptions.from_dict({"nope": 1})


def test_classify_loading_thresholds():
    o = sb.ReductionOptions()
    assert sb.classify_loading(1.0, 1.0, o) == "measurable"
    assert sb.classify_loading(2.999, 0.3, o) == "measurable"
    assert sb.classify_loading(3.0, 1.0, o) == "marginal"
    assert sb.classify_loading(5.99, 3.0, o) == "marginal"
    assert sb.classify_loading(6.0, 1.0, o) == "opaque"
    assert sb.classify_loading(np.inf, 1.0, o) == "opaque"
    assert sb.classify_loading(1.0, 0.06, o) == "unreliable"        # the uniformity rule comes first
    assert sb.classify_loading(1.0, 3.5, o) == "unreliable"
    assert sb.classify_loading(1.0, np.nan, o) == "unreliable"


def test_mu_source_from_table_and_from_compositions():
    m = sb.MuSource.from_table({"A": 7.0}, key_regex=r"^(.*?)(?:_rep\d+)?$")
    assert m.mu_for("A") == 7.0 and m.mu_for("A_rep2") == 7.0 and m.mu_for("B") is None
    df = pd.DataFrame({"sample": ["S", "Q", "M"], "composition": ["SiO2", "90(SiO2)10(Bi2O3)", None],
                       "density_g_cm3": [2.2, 3.0, None], "mu_mm": [None, None, 3.0]})
    ms = sb.MuSource.from_samples_table(df)
    mu_si = ms.mu_for("S", energy_ev=8041.0)
    assert mu_si is not None and 5 < mu_si < 12          # silica at Cu K-alpha: ~ 8 /mm
    from saxs.core.chemistry import calculate_absorption
    _, ab = calculate_absorption("SiO2", 2.2, energy_ev=8041.0)
    assert mu_si == pytest.approx(ab.mu_linear_mm_inv, rel=1e-12)
    assert ms.mu_for("Q", energy_ev=8041.0) > 3 * mu_si     # a bismuth silicate absorbs far more per mm
    assert ms.mu_for("M") == 3.0 and ms.rho_e_for("M") is None
    assert 0.5 < ms.rho_e_for("S", 8041.0) < 0.8             # electrons per A^3 of silica ~ 0.66
    assert ms.has("S") and not ms.has("nope")
    with pytest.raises(ValueError):
        sb.MuSource(compositions={"X": ("SiO2", 2.2)}).mu_for("X")     # no energy anywhere
    assert sb.MuSource(compositions={"X": ("SiO2", 2.2)}, energy_ev=8041.0).mu_for("X") == pytest.approx(mu_si, rel=1e-12)


def test_reduce_pairs_recovers_transmissions_and_classes(reduced):
    _, out, res = reduced
    truth = syn.campaign_truth()
    assert set(res) == set(truth)
    for name, (T, cls) in truth.items():
        r = res[name]
        assert r.transmission.T == pytest.approx(T, rel=0.03), name
        assert r.loading_class == cls, (name, r.mu_t, r.heterogeneity)
        assert os.path.exists(r.dat_path) and r.dat_path.endswith(f"{name}__sectors.dat")
    assert res["S1"].centre_source == "own direct beam" and abs(res["S1"].centre_offset_px[0] + 6.0) < 0.5
    assert res["S2"].centre_source.startswith("session empty")
    assert res["S1"].t_eff_mm == pytest.approx(-np.log(res["S1"].transmission.T) / 10.0, rel=1e-9)
    assert res["S4"].t_eff_mm is None and res["S4"].mu_mm is None and res["S4"].curve_per_mm() is None
    assert res["S1"].mu_mm == 10.0
    assert set(res["S1"].ratio_plateaus) == {"6-12", "12-20"}
    assert set(res["S1"].per_mm_probe) == {0.02, 0.05, 0.15}
    assert res["S1"].options["axis_exclusion_deg"] == 20.0


def test_per_mm_and_usable_curves(reduced):
    _, _, res = reduced
    r = res["S1"]
    pm = r.curve_per_mm()
    assert np.allclose(pm.intensity, r.curve.intensity / r.t_eff_mm) and pm.metadata["per_mm"] is True
    u = r.usable_curve()
    cols = u.metadata["columns"]
    assert u.npts == r.n_use and len(cols["ratio"]) == u.npts and all(v == 1 for v in cols["usable"])
    assert u.metadata["class_mu_t"] == "measurable" and u.metadata["per_mm"] is True
    assert r.q_min_use <= u.qmin and u.qmax <= r.q_max_use
    u4 = res["S4"].usable_curve()
    assert u4.metadata["per_mm"] is False and np.allclose(u4.intensity, res["S4"].curve.intensity[np.asarray(res["S4"].columns["usable"], bool)])


def test_dat_round_trip_and_loader(reduced):
    _, _, res = reduced
    r = res["S1"]
    back = sb.load_reduced_dat(r.dat_path)
    assert back.name == "S1" and back.loading_class == r.loading_class and back.centre_source == r.centre_source
    assert np.allclose(back.curve.q, r.curve.q) and np.allclose(back.curve.intensity, r.curve.intensity, rtol=1e-5)
    assert np.allclose(back.curve.sigma, r.curve.sigma, rtol=1e-5)
    assert back.transmission.T == pytest.approx(r.transmission.T, rel=1e-5)
    assert back.transmission.T_disc4 == pytest.approx(r.transmission.T_disc4, rel=1e-5)
    assert back.t_eff_mm == pytest.approx(r.t_eff_mm, rel=1e-6) and back.mu_mm == 10.0
    assert back.n_use == r.n_use and back.q_min_use == pytest.approx(r.q_min_use, rel=1e-6)
    assert back.centre_offset_px[0] == pytest.approx(r.centre_offset_px[0], abs=1e-3)
    assert back.options["axis_exclusion_deg"] == 20.0
    assert np.array_equal(back.columns["usable"], r.columns["usable"])
    assert np.allclose(back.columns["ratio"], r.columns["ratio"], rtol=1e-4)
    assert back.per_mm_probe.keys() == r.per_mm_probe.keys()
    # the generic loader reads the same file as an I_corr / sigma_I curve
    from saxs.core.loader import load_curve
    c = load_curve(r.dat_path)
    assert c.npts == r.curve.npts and np.allclose(c.intensity, r.curve.intensity, rtol=1e-5) and np.allclose(c.sigma, r.curve.sigma, rtol=1e-5)
    folder = sb.load_reduced_folder(os.path.dirname(r.dat_path), exclude_pattern="0p5")
    assert {x.name for x in folder} == {"S1", "S2", "S4"}


def test_summary_table_columns(reduced):
    _, _, res = reduced
    df = sb.summary_table(res.values())
    assert len(df) == 4
    for col in ("sample", "session", "sample_frame", "empty_frame", "T_used", "dT_rel", "T_disc4", "heterogeneity_T4_over_T",
                "ratio_plateau_6_12", "ratio_plateau_12_20", "mu_mm", "mu_t_eff", "t_eff_mm", "class_mu_t", "centre_source",
                "centre_dx_px", "q_min_use", "q_max_use", "n_use", "I_per_mm_at_0p02", "I_per_mm_at_0p15", "dat_path"):
        assert col in df.columns, col
    row = df.set_index("sample").loc["S2"]
    assert row["class_mu_t"] == "opaque" and row["mu_t_eff"] > 6
    assert sb.summary_table([]).shape[0] == 0


def test_reduce_arrays_with_options_in_memory():
    rng = np.random.default_rng(3)
    e = syn.empty_frame(noise=rng)
    s = syn.sample_frame(T=0.3, empty=e, ring_amp=400.0, noise=rng)
    geom = es.FrameGeometry(cx=syn.TRUE_CX, cy=syn.TRUE_CY, dist_m=syn.DIST_M, pixel_m=syn.PIXEL_M, wavelength_A=1.541891, exposure_s=100.0)
    o = sb.ReductionOptions(mu_t_measurable=1.0)          # stricter threshold: T = 0.3 -> mu t = 1.2 -> marginal
    r = sb.reduce_arrays_with_options(s, geom, e, geom, o, mu_mm=5.0, name="mem")
    assert r.loading_class == "marginal" and r.transmission.T == pytest.approx(0.3, rel=0.02)
    assert r.name == "mem" and r.reduction is not None and r.n_use > 0
    assert r.options["mu_t_measurable"] == 1.0
    assert r.header_dict()["class_mu_t"] == "marginal"
