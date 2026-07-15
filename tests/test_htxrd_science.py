"""Tests for htxrd_science.py (M20) — HTXRD series processing.

Pure-Python (no Qt), runs in the default suite. Uses the real bundled
EXAMPLES/HTXRD_example.rasx (one scan from the user's own P5Bi8-12
room-to-900C series) plus synthetic multi-pattern series built from known
Gaussians for the tracking/flagging logic.
"""
from __future__ import annotations

import numpy as np
import pytest
import rampy as rp

import htxrd_science as hs
from conftest import EXAMPLES_DIR


HTXRD_EXAMPLE = EXAMPLES_DIR / "HTXRD_example.rasx"


# --------------------------------------------------------------------------
# Filename-template ramp derivation (Jana-style ??? rewriting)
# --------------------------------------------------------------------------

def test_ramp_value_from_template_basic():
    assert hs.ramp_value_from_template("NB-LM01MO_100.XRDML", "NB-LM01MO_???.XRDML") == 100.0
    assert hs.ramp_value_from_template("scan_025.xy", "scan_???.xy") == 25.0


def test_ramp_value_from_template_case_insensitive_and_path_stripped():
    assert hs.ramp_value_from_template(r"C:\data\SCAN_300.XY", "scan_???.xy") == 300.0


def test_ramp_value_from_template_non_matching_returns_none():
    assert hs.ramp_value_from_template("other_100.xy", "scan_???.xy") is None
    assert hs.ramp_value_from_template("scan_abc.xy", "scan_???.xy") is None
    assert hs.ramp_value_from_template("scan_100.xy", "no_wildcards.xy") is None


# --------------------------------------------------------------------------
# Real .rasx loading
# --------------------------------------------------------------------------

def test_load_pattern_real_rasx_gets_temperature_from_metadata():
    pat = hs.load_pattern(str(HTXRD_EXAMPLE))
    assert pat.ramp_source == "metadata"
    assert pat.ramp_value is not None
    assert 0.0 < pat.ramp_value < 1000.0  # a physical temperature for a RT-900C series
    assert len(pat.x) > 100
    assert np.all(np.diff(pat.x) >= 0)  # sorted 2theta


def test_load_series_falls_back_to_index_without_metadata_or_template(tmp_path):
    # Two plain .xy files with no temperature info anywhere.
    x = np.linspace(10, 60, 200)
    for i, name in enumerate(["b.xy", "a.xy"]):
        y = rp.gaussian(x, 100.0, 30.0, 0.5)
        np.savetxt(tmp_path / name, np.column_stack([x, y]))
    series = hs.load_series([str(tmp_path / "b.xy"), str(tmp_path / "a.xy")])
    assert [p.ramp_source for p in series] == ["index", "index"]
    assert [p.ramp_value for p in series] == [0.0, 1.0]


def test_load_series_uses_filename_template(tmp_path):
    x = np.linspace(10, 60, 200)
    y = rp.gaussian(x, 100.0, 30.0, 0.5)
    for t in (300, 100, 200):
        np.savetxt(tmp_path / f"scan_{t}.xy", np.column_stack([x, y]))
    paths = [str(tmp_path / f"scan_{t}.xy") for t in (300, 100, 200)]
    series = hs.load_series(paths, filename_template="scan_???.xy")
    assert [p.ramp_value for p in series] == [100.0, 200.0, 300.0]  # sorted by ramp
    assert all(p.ramp_source == "filename" for p in series)


def test_find_series_files_filters_and_sorts(tmp_path):
    (tmp_path / "b.rasx").write_bytes(b"zip-ish")
    (tmp_path / "a.xy").write_text("1 2")
    (tmp_path / "notes.md").write_text("not a pattern")
    files = hs.find_series_files(str(tmp_path))
    names = [f.split("\\")[-1].split("/")[-1] for f in files]
    assert names == ["a.xy", "b.rasx"]


# --------------------------------------------------------------------------
# Peak tracking on a synthetic temperature series with KNOWN behavior:
# center shifts linearly with T (thermal expansion analog), constant FWHM.
# --------------------------------------------------------------------------

def _synthetic_series(centers_by_temp):
    x = np.linspace(20, 40, 800)
    rng = np.random.default_rng(0)
    series = []
    for temp, center in centers_by_temp:
        y = rp.gaussian(x, 500.0, center, 0.3) + 50.0 + rng.normal(0, 2.0, x.shape)
        series.append(hs.HtxrdPattern(path="", name=f"T{temp}", x=x, y=y, ramp_value=float(temp), ramp_source="metadata"))
    return series


def test_track_peak_recovers_linear_center_shift():
    temps_centers = [(100, 30.00), (200, 29.95), (300, 29.90), (400, 29.85)]
    series = _synthetic_series(temps_centers)
    results = hs.track_peak(series, window_lo=29.0, window_hi=31.0)

    assert len(results) == 4
    assert all(r.error is None for r in results)
    for (temp, true_center), r in zip(temps_centers, results):
        assert r.ramp_value == temp
        assert r.center == pytest.approx(true_center, abs=0.02)
        assert r.fwhm == pytest.approx(0.3, abs=0.05)


def test_track_peak_reports_error_row_for_empty_window():
    series = _synthetic_series([(100, 30.0)])
    results = hs.track_peak(series, window_lo=50.0, window_hi=55.0)  # no data there
    assert len(results) == 1
    assert results[0].error is not None
    assert np.isnan(results[0].center)


def test_track_peak_rejects_inverted_window():
    with pytest.raises(ValueError):
        hs.track_peak([], window_lo=31.0, window_hi=29.0)


# --------------------------------------------------------------------------
# Transition flagging
# --------------------------------------------------------------------------

def test_flag_transition_candidates_flags_anomalous_chi2():
    results = [
        hs.PeakTrackResult("a", 100.0, 30.0, 0.3, 500.0, 100.0, 1.0),
        hs.PeakTrackResult("b", 200.0, 30.0, 0.3, 500.0, 100.0, 1.1),
        hs.PeakTrackResult("c", 300.0, 30.0, 0.3, 500.0, 100.0, 0.9),
        hs.PeakTrackResult("d", 400.0, 30.0, 0.3, 500.0, 100.0, 50.0),  # transition!
        hs.PeakTrackResult("e", 500.0, 30.0, 0.3, 500.0, 100.0, 1.05),
    ]
    flags = hs.flag_transition_candidates(results, z=3.0)
    flagged_temps = [f[0] for f in flags]
    assert 400.0 in flagged_temps
    assert 100.0 not in flagged_temps


def test_flag_transition_candidates_includes_failed_fits():
    results = [
        hs.PeakTrackResult("a", 100.0, 30.0, 0.3, 500.0, 100.0, 1.0),
        hs.PeakTrackResult("b", 200.0, np.nan, np.nan, np.nan, np.nan, np.nan, error="peak vanished"),
    ]
    flags = hs.flag_transition_candidates(results)
    assert any("fit failed" in f[2] for f in flags)
    assert any(f[0] == 200.0 for f in flags)


def test_flag_transition_candidates_quiet_series_returns_no_chi2_flags():
    results = [hs.PeakTrackResult(f"p{i}", 100.0 * i, 30.0, 0.3, 500.0, 100.0, 1.0 + 0.01 * i) for i in range(5)]
    flags = hs.flag_transition_candidates(results, z=3.0)
    assert flags == []
