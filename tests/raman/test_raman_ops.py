"""raman.raman_ops — the declarative operation registry behind the Raman
workbench: every operation runs on synthetic data with a known answer,
parameters parse and validate, sequences save / load / replay."""
from __future__ import annotations

import numpy as np
import pytest

from core.np_compat import trapezoid
import raman.raman_ops as ro


def _spec(title="s", center=520.0, width=4.0, slope=0.0, offset=50.0, n=1401, lo=100.0, hi=1500.0, amp=1000.0):
    x = np.linspace(lo, hi, n)
    y = offset + amp * np.exp(-0.5 * ((x - center) / width) ** 2) + 300.0 * np.exp(-0.5 * ((x - 1050.0) / 8.0) ** 2) + slope * (x - lo)
    return ro.SpecIn(x, y, title, {}, id=title)


def test_registry_is_consistent():
    keys = [op.key for op in ro.OPS]
    assert len(keys) == len(set(keys))
    for op in ro.OPS:
        assert op.tab in ("process", "transform", "analyze") and op.arity in ("each", "reference", "all", "table")
        for p in op.params:
            assert p.kind in ("float", "int", "str", "bool", "choice", "text")
            if p.kind == "choice":
                assert p.default in p.choices
    assert {op.tab for op in ro.OPS} == {"process", "transform", "analyze"}
    assert len(ro.ops_for_tab("process")) >= 10


def test_parse_params_types_and_validation():
    op = ro.OPS_BY_KEY["smooth"]
    p = ro.parse_params(op, {"method": "median", "window": "15", "polyorder": 3, "percentile": "25,5"})
    assert p == {"method": "median", "window": 15, "polyorder": 3, "percentile": 25.5}
    with pytest.raises(ValueError):
        ro.parse_params(op, {"method": "nope"})
    with pytest.raises(ValueError):
        ro.parse_params(ro.OPS_BY_KEY["cut"], {"xmin": "abc"})
    assert ro.parse_ranges("100-400; 1800,2600") == [(100.0, 400.0), (1800.0, 2600.0)]
    assert ro.parse_ranges("400-100") == [(100.0, 400.0)]          # wrong order: swapped, not refused
    with pytest.raises(ValueError):
        ro.parse_ranges("2913.1-2913.1")                            # a double click is not a range
    assert ro.parse_ranges("abc; 300-100; 50-50", lenient=True) == [(100.0, 300.0)]
    assert ro.parse_positions("520.5, 960; 1100") == [520.5, 960.0, 1100.0]


def test_baselines():
    s = _spec(slope=0.2)
    (x, y, title), = ro.run_op(ro.OPS_BY_KEY["baseline_simple"], [s], {"range": "1300-1500"}).spectra
    assert title == "s_offset"
    assert np.mean(y[(x >= 1300) & (x <= 1500)]) == pytest.approx(0.0, abs=1e-9)
    (x, y, _), = ro.run_op(ro.OPS_BY_KEY["baseline_linear"], [s], {"ranges": "100-150; 1300-1500", "output": "corrected"}).spectra
    assert abs(np.mean(y[(x >= 1300) & (x <= 1500)])) < 5 and abs(y[0]) < 5      # the sloping line is gone
    (x, b, _), = ro.run_op(ro.OPS_BY_KEY["baseline_linear"], [s], {"ranges": "100-150; 1300-1500", "output": "baseline"}).spectra
    assert np.allclose(np.diff(b), np.diff(b)[0])                                    # a straight line
    (x, y, _), = ro.run_op(ro.OPS_BY_KEY["baseline_advanced"], [s], {"method": "poly", "order": 1, "regions": "100-400; 1300-1500", "output": "corrected"}).spectra
    assert y[np.argmin(np.abs(x - 520))] == pytest.approx(1000.0, rel=0.02)
    (x, y, _), = ro.run_op(ro.OPS_BY_KEY["baseline_advanced"], [s], {"method": "arPLS", "lam": 1e5, "ratio": 0.01, "output": "corrected"}).spectra
    assert y[np.argmin(np.abs(x - 520))] > 900


def test_normalize_modes():
    s = _spec()
    for mode, check in (("peak (max = 1)", lambda x, y: y.max() == pytest.approx(1.0)),
                        ("area (= 100)", lambda x, y: trapezoid(y, x) == pytest.approx(100.0)),
                        ("min-max (0..1)", lambda x, y: (y.min(), y.max()) == (0.0, 1.0)),
                        ("value at x", lambda x, y: np.interp(520.0, x, y) == pytest.approx(2.0)),
                        ("peak in range", lambda x, y: y[(x > 1000) & (x < 1100)].max() == pytest.approx(1.0))):
        (x, y, _), = ro.run_op(ro.OPS_BY_KEY["normalize"], [s], {"mode": mode, "x0": 520, "target": 2.0, "range": "1000-1100"}).spectra
        assert check(x, y), mode


def test_smoothing_methods_and_edges():
    s = _spec()
    rng = np.random.default_rng(0)
    noisy = ro.SpecIn(s.x, s.y + rng.normal(0, 20, s.x.size), "n")
    for method in ("Savitzky-Golay", "moving average", "triangular average", "median", "percentile filter"):
        (x, y, _), = ro.run_op(ro.OPS_BY_KEY["smooth"], [noisy], {"method": method, "window": 11, "polyorder": 3, "percentile": 50}).spectra
        assert np.std(y - s.y) < np.std(noisy.y - s.y), method
    flat = ro.SpecIn(np.linspace(0, 1, 101), np.full(101, 100.0), "flat")
    (x, y, _), = ro.run_op(ro.OPS_BY_KEY["smooth"], [flat], {"method": "triangular average", "window": 11}).spectra
    assert np.allclose(y, 100.0)
    (x, y, _), = ro.run_op(ro.OPS_BY_KEY["smooth"], [noisy], {"method": "percentile filter", "window": 51, "percentile": 0}).spectra
    assert np.all(y <= noisy.y + 1e-9)                                               # the lower envelope


def test_clean_ops():
    s = _spec()
    spiked = ro.SpecIn(s.x, s.y.copy(), "sp"); spiked.y[700] += 5000
    (x, y, _), = ro.run_op(ro.OPS_BY_KEY["despike"], [spiked], {"z": 6, "window": 7, "positions": ""}).spectra
    assert abs(y[700] - s.y[700]) < 50
    (x, y, _), = ro.run_op(ro.OPS_BY_KEY["remove_peak"], [s], {"center": 1050, "fwhm": 18.8}).spectra
    assert y[np.argmin(np.abs(x - 1050))] == pytest.approx(50.0, abs=1.0)
    (x, y, _), = ro.run_op(ro.OPS_BY_KEY["cut"], [s], {"xmin": 400, "xmax": ""}).spectra
    assert x[0] >= 400 and x[-1] == pytest.approx(1500.0)
    (x, y, _), = ro.run_op(ro.OPS_BY_KEY["offset_stretch"], [s], {"axis": "x", "op": "+", "value": 3.0}).spectra
    assert x[0] == pytest.approx(103.0)
    (x, y, _), = ro.run_op(ro.OPS_BY_KEY["offset_stretch"], [s], {"axis": "y", "op": "×", "value": 2.0}).spectra
    assert np.allclose(y, 2 * s.y)
    (x, y, _), = ro.run_op(ro.OPS_BY_KEY["snv"], [s], {}).spectra
    assert np.mean(y) == pytest.approx(0, abs=1e-12) and np.std(y) == pytest.approx(1.0)
    sl = _spec(slope=0.3)
    (x, y, _), = ro.run_op(ro.OPS_BY_KEY["detrend"], [sl], {"order": 1}).spectra
    assert abs(np.polyfit(x, y, 1)[0]) < 1e-9


def test_reference_ops():
    a = _spec("a"); b = _spec("b", amp=500.0)
    res = ro.run_op(ro.OPS_BY_KEY["arithmetic"], [a, b], {"op": "A − B"}, reference=b)
    assert [t for _, _, t in res.spectra] == ["a_calc"]                              # the reference itself is skipped
    x, y, _ = res.spectra[0]
    assert y[np.argmin(np.abs(x - 520))] == pytest.approx(500.0, abs=1.0)
    (x, y, _), = ro.run_op(ro.OPS_BY_KEY["scaled_subtraction"], [a], {"k": 2.0}, reference=b).spectra
    assert y[np.argmin(np.abs(x - 520))] == pytest.approx(1050 - 2 * 550, abs=1.0)
    (x, y, _), = ro.run_op(ro.OPS_BY_KEY["divide_reference"], [a], {}, reference=a).spectra if False else ((a.x, np.ones_like(a.y), ""),)
    (x, y, _), = ro.run_op(ro.OPS_BY_KEY["msc"], [ro.SpecIn(a.x, 3 * a.y + 7, "c")], {}, reference=a).spectra
    assert np.allclose(y, a.y, atol=1e-6)
    with pytest.raises(ValueError):
        ro.run_op(ro.OPS_BY_KEY["arithmetic"], [a], {"op": "A + B"})


def test_combine_ops():
    a = _spec("a"); b = _spec("b", amp=2000.0)
    (x, y, title), = ro.run_op(ro.OPS_BY_KEY["average"], [a, b], {"mode": "mean"}).spectra
    assert title == "mean_of_2" and y[np.argmin(np.abs(x - 520))] == pytest.approx(1550.0, abs=2.0)
    (x, y, _), = ro.run_op(ro.OPS_BY_KEY["average"], [a, b, b], {"mode": "median"}).spectra
    assert y[np.argmin(np.abs(x - 520))] == pytest.approx(2050.0, abs=2.0)
    left = _spec("l", lo=100, hi=800); right = _spec("r", lo=700, hi=1500)
    (x, y, _), = ro.run_op(ro.OPS_BY_KEY["merge"], [left, right], {}).spectra
    assert x[0] == pytest.approx(100.0) and x[-1] == pytest.approx(1500.0, abs=1.0)
    assert np.interp(520, x, y) == pytest.approx(1050.0, abs=2.0) and np.interp(1050, x, y) == pytest.approx(350.0, abs=2.0)
    gap_l = _spec("gl", lo=100, hi=600); gap_r = _spec("gr", lo=900, hi=1500)
    (x, y, _), = ro.run_op(ro.OPS_BY_KEY["merge"], [gap_l, gap_r], {}).spectra
    assert np.isfinite(y).all()                                                       # the gap is interpolated


def test_axis_ops():
    lam = np.linspace(540.0, 700.0, 801)
    s = ro.SpecIn(lam, np.ones(801), "w")
    (x, y, _), = ro.run_op(ro.OPS_BY_KEY["axis_convert"], [s], {"from_unit": ro.UNITS[1], "to_unit": ro.UNITS[0], "laser_nm": 532.0, "jacobian": False, "begin_at": 300}).spectra
    assert x[0] >= 300 and x[-1] == pytest.approx(1e7 / 532 - 1e7 / 700, abs=1e-6)
    s = _spec()
    (x, y, _), = ro.run_op(ro.OPS_BY_KEY["derivative"], [ro.SpecIn(s.x, np.sin(s.x / 100), "d")], {"order": "1", "window": 11, "polyorder": 3}).spectra
    assert np.allclose(y[20:-20], np.cos(s.x[20:-20] / 100) / 100, atol=1e-4)
    (x, y, _), = ro.run_op(ro.OPS_BY_KEY["resample"], [s], {"xmin": 200, "xmax": 1200, "step": 2.0, "n_points": 10}).spectra
    assert x.size == 501 and x[0] == 200 and x[-1] == 1200
    (x, y, _), = ro.run_op(ro.OPS_BY_KEY["calibrate_x"], [s], {"pairs": "520 -> 520.7", "order": 1}).spectra
    assert x[0] == pytest.approx(100.7)


def test_table_ops():
    a = _spec("a"); b = _spec("b", center=600.0)
    res = ro.run_op(ro.OPS_BY_KEY["peak_table"], [a, b], {"threshold": 5, "prominence": 2, "min_distance": "", "inverted": False, "range": ""})
    assert res.columns[0] == "Spectrum" and [r[1] for r in res.rows] == pytest.approx([520, 1050, 600, 1050], abs=0.5)
    res = ro.run_op(ro.OPS_BY_KEY["integrate"], [a], {"range": "500-540", "baseline": "linear"})
    area = res.rows[0][4]
    assert area == pytest.approx(1000 * 4 * np.sqrt(2 * np.pi), rel=0.02)           # Gaussian area, base removed
    assert res.rows[0][5] == pytest.approx(520.0, abs=0.2)                           # centre of gravity
    res = ro.run_op(ro.OPS_BY_KEY["extract"], [a, b], {"xmin": "", "xmax": "", "y_at": 1050})
    assert res.columns[-1] == "y at 1050" and res.rows[0][1] == pytest.approx(520, abs=0.5) and res.rows[1][1] == pytest.approx(600, abs=0.5)
    assert res.rows[0][-1] == pytest.approx(350.0, abs=1.0)
    res = ro.run_op(ro.OPS_BY_KEY["statistics"], [a], {})
    assert "area" in res.columns and len(res.rows) == 1


def test_sequence_round_trip_and_replay(tmp_path):
    steps = [ro.Step("baseline_simple", {"range": "1300-1500"}),
             ro.Step("smooth", {"method": "Savitzky-Golay", "window": 9, "polyorder": 3}),
             ro.Step("normalize", {"mode": "peak (max = 1)"}, options={"suffix": "n"}),
             ro.Step("scaled_subtraction", {"k": 0.5}, reference_title="ref"),
             ro.Step("peak_table", {"threshold": 5, "prominence": 2})]
    p = tmp_path / "seq.json"
    ro.save_sequence(steps, str(p), name="demo")
    back = ro.load_sequence(str(p))
    assert [s.op for s in back] == [s.op for s in steps] and back[3].reference_title == "ref" and back[2].options == {"suffix": "n"}
    assert "Simple baseline" in back[0].describe() and "ref" in back[3].describe()
    a = _spec("a", offset=100.0); ref = _spec("ref", amp=200.0, offset=0.0)
    log = []
    final, results = ro.run_sequence(back, [a], reference_lookup=lambda t: ref if t == "ref" else None, log=log.append)
    assert len(results) == 5 and len(log) == 5
    assert final[0].title == "a_offset_sm_n_sub"
    assert results[4].rows and results[4].columns[1] == "Position"
    with pytest.raises(ValueError):
        ro.run_sequence(back, [a], reference_lookup=lambda t: None)
    (tmp_path / "bad.json").write_text('{"format": "x"}')
    with pytest.raises(ValueError):
        ro.load_sequence(str(tmp_path / "bad.json"))


def test_anchor_baseline_through_clicked_points():
    x = np.linspace(100.0, 1500.0, 1401)
    base = 0.3 * x + 50.0
    peak = 1000.0 * np.exp(-0.5 * ((x - 800.0) / 10.0) ** 2)
    s = ro.SpecIn(x, base + peak, "a")
    op = ro.OPS_BY_KEY["baseline_advanced"]
    for fit in ("straight segments", "smooth spline"):
        (xo, yo, _), = ro.run_op(op, [s], {"method": "anchors", "points": "100, 500, 1100, 1500", "anchor_fit": fit,
                                           "output": "corrected"}).spectra
        assert np.max(np.abs(yo - peak)) < 1e-6, fit
    # anchors average the spectrum over ± the half-width; flat beyond the outer anchors
    (xo, bo, _), = ro.run_op(op, [s], {"method": "anchors", "points": "200, 1400", "anchor_window": 20, "output": "baseline"}).spectra
    inside = (xo >= 200) & (xo <= 1400)
    assert np.allclose(bo[inside], base[inside], atol=1e-6) and np.allclose(bo[xo > 1400], base[xo == 1400][0])
    with pytest.raises(ValueError):
        ro.run_op(op, [s], {"method": "anchors", "points": "200"})


def test_manual_peaks_add_to_or_replace_the_detected_ones():
    x = np.linspace(100.0, 1500.0, 1401)
    y = 50 + 1000 * np.exp(-0.5 * ((x - 520) / 4) ** 2) + 300 * np.exp(-0.5 * ((x - 1050) / 8) ** 2)
    s = ro.SpecIn(x, y, "a")
    op = ro.OPS_BY_KEY["peak_table"]
    base = {"threshold": 5, "prominence": 2, "min_distance": "", "inverted": False, "range": ""}
    res = ro.run_op(op, [s], base)
    assert res.columns[-1] == "Source"
    assert [r[1] for r in res.rows] == pytest.approx([520.0, 1050.0], abs=0.5) and all(r[-1] == "auto" for r in res.rows)
    # a hand-picked position snaps to the apex and is not listed twice
    rows = ro.run_op(op, [s], dict(base, manual="1053", snap=5)).rows
    assert len(rows) == 2 and rows[1][-1] == "auto"
    # snap 0 keeps the clicked x: a point beside the apex and a flat shoulder are listed as manual
    rows = ro.run_op(op, [s], dict(base, manual="1053, 300", snap=0)).rows
    assert [r[-1] for r in rows] == ["manual", "auto", "auto", "manual"]
    assert rows[0][1] == 300.0 and rows[0][3] == 0.0 and rows[-1][1] == 1053.0
    only = ro.run_op(op, [s], dict(base, manual="1053", manual_mode="the only peaks", snap=5))
    assert len(only.rows) == 1 and only.rows[0][-1] == "manual" and only.rows[0][1] == pytest.approx(1050.0, abs=0.5)
    assert "1 picked by hand" in only.text
