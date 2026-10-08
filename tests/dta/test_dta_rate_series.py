"""Tests for dta/dta_rate_series.py — multi-rate series assembly, provenance
warnings, fits and CSV.

Four kinds of test live here and they are deliberately not mixed:

* **Literal-row tests** build `manual_row`s and therefore always run, with no
  file at all. Every exact number (the real BM triple's Lasocka and Moynihan
  coefficients) and every warning-logic test lives here, because SPEC §0.C
  measured that a per-segment Tg drifts 0.1-0.5 K and a per-segment mass loss
  drifts by a FACTOR OF TWO with the segment boundaries — an exact number
  pinned against live segmentation would be pinning the segmenter.
* **Synthetic-file tests** write small TA-format exports into `tmp_path` and
  exercise `rows_from_file` / `rows_from_record` end to end, where the right
  answer is known by construction.
* **Fixture tests** run on the committed `EXAMPLES/DTA_cycles_example.txt`
  (every 4th row of the real 890 min, 4-cycle run). They always run.
* **Full-file tests** behind `@requires_dta_bm` pin the same quantities on the
  un-decimated 18996-row original and on the single-ramp BM-0.txt.

Every number here was measured on 2026-09-26 with `py -3.11`; where a number
differs from SPEC_rate_series the measured value is used and the difference is
called out in a comment next to it.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import core.io_universal as iou
from conftest import requires_dta_bm
from dta import dta_kinetics as dk
from dta import dta_program as dp
from dta import dta_rate_series as rs
from dta import dta_segments as ds

# ---------------------------------------------------------------------------
# Pinned measurements
# ---------------------------------------------------------------------------
#: The real BM-0-cycles triple: (heating rate, Tg at 350-700 °C). heat 2 is
#: absent because its baseline fails — that absence is the point of test 21.
BM_TRIPLE = ((20.0, 620.8), (15.0, 657.1), (10.0, 634.1))

#: The verbatim message dta_science raises on heat 2 at 350-700 °C.
HEAT2_ERROR = "Baseline range has too few points."

#: One physical specimen: Sample | Size mg | Run date, from the real header.
BM_SPECIMEN = "BM-0-cycles|97.8930|2026-09-25"

#: Cumulative mass loss at the end of each heating ramp, % of as-loaded mass.
#: Measured on the full file: 0.2585 / 0.6682 / 0.8473 / 1.0274.
BM_LOSSES = (0.31, 0.66, 0.84, 1.03)

#: Measured mode rates of the four heating ramps (full file / fixture).
FULL_HEAT_RATES = (19.949, 19.962, 14.991, 9.988)
FIXTURE_HEAT_RATES = (19.954, 19.965, 14.991, 9.986)

#: First ORIGINAL row of each heating segment (full file / fixture).
FULL_HEAT_ROW_STARTS = (0, 3610, 7565, 12844)
FIXTURE_HEAT_ROW_STARTS = (0, 903, 1892, 3212)

#: Per-segment Tg bands at 350-700 °C. Measured 620.710 / fail / 657.598 /
#: 633.612; the bands absorb the boundary drift SPEC §0.C measured.
TG_BANDS = {1: (615.0, 626.0), 3: (652.0, 662.0), 4: (628.0, 660.0)}

#: Window the crystallization-exotherm extras are searched in. SPEC §9.3
#: test 48 said "HF_max < 15 mW on heats 2-4" over the whole segment, but over
#: the WHOLE segment heat 2's maximum is 15.10 mW at 94.4 °C — the
#: start-of-ramp transient, not an exotherm. Restricted to 300-1300 °C the
#: four heats give 32.15 / 1.27 / 6.09 / 11.13 mW and the test means what it
#: says. Measured, and the spec followed the measurement.
HF_WINDOW = (300.0, 1300.0)

#: Codes whose message must end "Fitted anyway.", because a reader could
#: otherwise mistake them for a refusal. Exemptions are named (with their
#: reason) in test_every_refusable_warning_says_fitted_anyway.
MUST_SAY_FITTED_ANYWAY = frozenset({
    "few_rates", "duplicate_rate", "narrow_rate_span", "non_monotonic",
    "thermal_history", "virgin_mixed", "mixed_design", "mass_loss_spread",
    "mass_loss_high", "mixed_quantity", "mixed_method", "mixed_window",
    "mixed_signal", "model_quantity_mismatch", "rate_mismatch",
    "setpoint_not_reached", "partial_segment", "cooling_row", "unit_suspect",
    "value_outside_window",
})

#: Words that read as a refusal. The user asked to be able to process this
#: data; a warning may say the answer is wrong, never that the tool declined
#: to produce it.
BANNED_WORDS = ("cannot", "invalid", "refusing", "refuse", "refused")


# ---------------------------------------------------------------------------
# Builders — literal rows, no files
# ---------------------------------------------------------------------------
def _triple_series(name: str = "BM triple") -> rs.RateSeries:
    """The three measurable rows of the real run, as literal manual rows."""
    series = rs.RateSeries(name=name, quantity_kind="Tg")
    for q, tg in BM_TRIPLE:
        series.add(rs.manual_row(sample="BM-0-cycles", q_C_per_min=q, value_C=tg))
    return series


def _bm_like_series() -> rs.RateSeries:
    """The real four-row BM series reproduced literally: three measured rows
    plus the heat-2 row whose baseline failed, with the thermal history and
    mass loss that make the four rows four different materials.

    Reproduced by hand rather than segmented so it always runs and so the
    warning assertions test the warnings, not the segmenter.
    """
    series = rs.RateSeries(name="BM-0-cycles heats", quantity_kind="Tg")
    spec = (
        ("heat 1 (cycle 1)", 20.0, 620.8, 0, "virgin", BM_LOSSES[0], ""),
        ("heat 2 (cycle 2)", 20.0, float("nan"), 1, "after-cycle-1", BM_LOSSES[1],
         HEAT2_ERROR),
        ("heat 3 (cycle 3)", 15.0, 657.1, 2, "after-cycle-2", BM_LOSSES[2], ""),
        ("heat 4 (cycle 4)", 10.0, 634.1, 3, "after-cycle-3", BM_LOSSES[3], ""),
    )
    for label, q, value, prior, state, loss, error in spec:
        row = rs.manual_row(sample="BM-0-cycles", q_C_per_min=q, value_C=value,
                            specimen_key=BM_SPECIMEN, state=state,
                            prior_heats=prior, cum_mass_loss_pct=loss,
                            window_C=(350.0, 700.0), method="parallel",
                            error=error)
        row.source_kind = "segment"
        row.segment_label = label
        row.segment_kind = ds.KIND_HEAT
        row.role_in_cycle = ds.ROLE_HEAT
        row.cycle_index = prior + 1
        series.add(row)
    return series


def _clean_series(rates=(5.0, 10.0, 15.0, 20.0, 30.0), a=500.0, b=8.0) -> rs.RateSeries:
    """Five separate specimens, one per rate, on an exact Lasocka line.

    The textbook design, and the control for every "no alarm on a clean
    series" assertion: nothing about it may raise anything above `info`.
    """
    series = rs.RateSeries(name="five specimens", quantity_kind="Tg")
    for i, q in enumerate(rates):
        series.add(rs.manual_row(
            sample=f"G{q:g}", q_C_per_min=q, value_C=a + b * float(np.log(q)),
            specimen_key=f"G{q:g}|{10.0 + i:.4f}|2026-01-0{i + 1}"))
    return series


# ---------------------------------------------------------------------------
# Builders — synthetic TA files
# ---------------------------------------------------------------------------
_TA_HEADER = """CLOSED
Version\t2.0
Instrument\tSDT Q600 V20.9 Build 20
Operator\tSS
Sample\t{sample}
Size\t{size}\tmg
ProcName\tRamp
Method\tRamp
Nsig\t4
Sig1\tTime (min)
Sig2\tTemperature (\u00b0C)
Sig3\tWeight (mg)
Sig4\tTemperature Difference (\u00b0C)
Date\t{date}
Time\t09:00:00
OrgMethod\t1: Ramp {rate:.2f} \u00b0C/min to {top:.2f} \u00b0C
OrgMethod\t2: Isothermal for 5.00 min
OrgMethod\t3: End of method
StartOfData
"""


def _write_ta_ramp(path, *, rate, sample, size, date, tg=600.0, dt=0.05,
                   t_start=25.0, top=700.0, mass_loss=0.2):
    """One synthetic single-ramp TA export: ramp to `top`, then hold 5 min.

    UTF-8 with BOM like every real TA export, with a sigmoid in the
    temperature-difference signal centred on `tg` so a measurement has
    something to find. The right answer is known by construction, which is
    what makes the import tests a test of the importer.
    """
    n_ramp = int(round((top - t_start) / rate / dt)) + 1
    t_ramp = np.arange(n_ramp) * dt
    T_ramp = t_start + rate * t_ramp
    t_hold = t_ramp[-1] + dt * np.arange(1, int(5.0 / dt) + 1)
    T_hold = np.full(t_hold.size, top)
    t = np.concatenate([t_ramp, t_hold])
    T = np.concatenate([T_ramp, T_hold])
    w = size - mass_loss * (t / t[-1])
    y = 0.6 / (1.0 + np.exp(-(T - tg) / 8.0)) - 0.0004 * T
    lines = [_TA_HEADER.format(sample=sample, size=f"{size:.4f}", date=date,
                               rate=rate, top=top)]
    lines += [f"{a:.6f}\t{b:.4f}\t{c:.5f}\t{d:.6f}" for a, b, c, d in zip(t, T, w, y)]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    return Path(path)


def _five_files(tmp_path):
    """Five separate specimens at 5/10/15/20/30 °C/min, GL_q005.txt … q030."""
    out = []
    for i, q in enumerate((5.0, 10.0, 15.0, 20.0, 30.0)):
        out.append(_write_ta_ramp(tmp_path / f"GL_q{int(q):03d}.txt", rate=q,
                                  sample=f"GL-{int(q)}", size=12.0 + i,
                                  date=f"2026-02-0{i + 1}",
                                  tg=595.0 + 3.0 * float(np.log(q))))
    return out


def _midpoint_measure(x, y, window):
    """A deterministic stand-in for a Tg construction: the temperature where
    the signal crosses the midpoint of its range inside `window`.

    Used where the test is about the PLUMBING — does `measure` receive the
    right arrays, does a failure become a stored row — rather than about any
    particular Tg algorithm, so the assertion carries no dependence on
    dta_science's tangent constructions.
    """
    lo, hi = float(window[0]), float(window[1])
    mask = np.isfinite(x) & np.isfinite(y) & (x >= lo) & (x <= hi)
    if mask.sum() < 3:
        raise ValueError(HEAT2_ERROR)
    xw, yw = x[mask], y[mask]
    half = 0.5 * (float(np.nanmin(yw)) + float(np.nanmax(yw)))
    idx = int(np.argmin(np.abs(yw - half)))
    return float(xw[idx]), {"y_span": float(np.nanmax(yw) - np.nanmin(yw))}


# ---------------------------------------------------------------------------
# Loading helpers for the real / fixture files
# ---------------------------------------------------------------------------
def _load(path):
    """(header, dataframe) from a TA SDT export."""
    header, _signals, df = iou.parse_ta_sdt_txt(Path(path))
    return header, df


def _program(header):
    """The OrgMethod program, read through TAHeader's repeated lines."""
    return dp.program_from_header(header, header_lines=getattr(header, "lines", None))


def _tg_measure(window):
    """A real Tg measurer: parallel tangents over `window`, via dta_science.

    Raises exactly what dta_science raises, which is how the verbatim
    "Baseline range has too few points." reaches `RateRow.error`.
    """
    from dta import dta_science as sci

    def _measure(x, y, _window):
        """(value, extras) for one segment's prepared arrays."""
        with np.errstate(divide="ignore", invalid="ignore"):
            result = sci.compute_tg_parallel_improved(x, y, window[0], window[1])
        return result.tg, {}

    return _measure


@pytest.fixture(scope="module")
def cycles_fixture(request):
    """(header, df, SegmentedRun, path) for the committed decimated fixture."""
    path = request.config.rootpath / "EXAMPLES" / "DTA_cycles_example.txt"
    header, df = _load(path)
    run = rs._segment_dataframe(df, program=_program(header))
    return header, df, run, path


@pytest.fixture(scope="module")
def bm_cycles():
    """Same, for the un-decimated original. Only used behind @requires_dta_bm."""
    from conftest import DTA_BM_DIR
    path = DTA_BM_DIR / "BM-0-cycles.txt"
    if not path.exists():
        pytest.skip(f"{path} is not present")
    header, df = _load(path)
    run = rs._segment_dataframe(df, program=_program(header))
    return header, df, run, path


# ===========================================================================
# 1) The exact numbers, always run (SPEC §9.2 tests 19-20)
# ===========================================================================
def test_real_bm_triple_lasocka_exact_numbers():
    """Every Lasocka coefficient of the real triple, through the series layer."""
    res = rs.fit_series(_triple_series(), "lasocka")
    assert res.ok is True
    assert res.lasocka_B == pytest.approx(-14.2053, abs=1e-4)
    assert res.lasocka_B_se == pytest.approx(50.7870, abs=1e-3)
    assert res.lasocka_A_C == pytest.approx(675.2443, abs=1e-3)
    assert res.lasocka_A_se == pytest.approx(136.3069, abs=1e-3)
    assert res.lasocka_A_K == pytest.approx(948.3943, abs=1e-3)
    assert res.lasocka_A_K - res.lasocka_A_C == pytest.approx(273.15, abs=1e-9)
    assert res.r2 == pytest.approx(0.072558, abs=1e-6)
    assert res.s_resid == pytest.approx(25.0117, abs=1e-3)
    assert res.dof == 1
    assert res.T_at_q_ref_C == pytest.approx(642.5354, abs=1e-3)
    assert res.pairwise_slope == pytest.approx((-126.18, 56.72), abs=1e-2)
    # SPEC test 19 listed [-11.889, +20.324, -8.435] and called it
    # "rate-sorted plot order". Measured, that triple is the INSERTION order
    # (20, 15, 10); rate-sorted ascending (10, 15, 20) it is the list below.
    # fitted_rows() sorts ascending by |q| because that is plot order for
    # ln q, so this is what the series layer produces.
    assert res.residuals == pytest.approx([-8.435, 20.324, -11.889], abs=1e-2)
    assert res.labels[0].endswith("10.0 °C/min")
    assert res.labels[-1].endswith("20.0 °C/min")


def test_real_bm_triple_moynihan_exact_numbers():
    """The Ea(Tg) of the same triple, negative and reported with its ±."""
    res = rs.fit_series(_triple_series(), "moynihan")
    assert res.ok is True
    assert res.slope == pytest.approx(4413.6190, abs=1e-2)
    assert res.slope_se == pytest.approx(15164.4065, abs=1e-1)
    assert res.Ea_kJ_mol == pytest.approx(-36.6969, abs=1e-3)
    assert res.Ea_se_kJ_mol == pytest.approx(126.0839, abs=1e-3)
    assert res.r2 == pytest.approx(0.078095, abs=1e-6)
    lo, hi = res.Ea_ci95_kJ_mol
    assert lo == pytest.approx(-1638.7, abs=0.5)
    assert hi == pytest.approx(1565.4, abs=0.5)
    assert res.temp_basis == "K (converted)"
    assert "negative_ea" in res.codes


# ===========================================================================
# 2) THE FLAGSHIP — warn, never block (SPEC §9.2 test 21)
# ===========================================================================
def test_bm_like_series_returns_a_fit_and_five_critical_warnings():
    """The whole contract in one function: a real number IS produced, AND
    every objection to it is attached to the same result.

    This is the test that encodes the user's instruction "ok but still make
    this possible to process such data". The BM series is confounded in every
    way a series can be — one specimen measured four times, each heat
    annealing the material the next heat measures, a failed baseline, 0.7 pp
    of mass loss, a rate span of only 2x and a Tg that is not monotonic in
    rate — and none of that stops the arithmetic. It all travels WITH the
    answer instead of in place of it.
    """
    series = _bm_like_series()
    res = rs.fit_series(series, "lasocka")
    codes = set(res.codes)

    # --- half one: the number exists -----------------------------------
    assert res.ok is True
    assert np.isfinite(res.lasocka_B)
    assert res.lasocka_B == pytest.approx(-14.2053, abs=1e-4)
    assert res.n_fitted == 3
    assert res.n_total == 4

    # --- half two: every warning is there -------------------------------
    assert {"no_value", "thermal_history", "virgin_mixed", "non_monotonic",
            "negative_lasocka_b"} <= codes
    assert {"single_specimen_cycles", "narrow_rate_span", "mass_loss_spread",
            "mass_loss_high", "low_r2"} <= codes
    assert {"dof_one", "rows_not_fitted"} <= codes
    assert dk.max_severity(res.all_warnings) == "critical"

    # the failed row is named, with its verbatim message, and is still a row
    no_value = next(w for w in res.all_warnings if w.code == "no_value")
    assert HEAT2_ERROR in no_value.message
    assert len(series.rows) == 4
    assert series.rows[1].value_source == "failed"
    assert series.rows[1].error == HEAT2_ERROR

    # and nothing in any message reads as a refusal
    for w in res.all_warnings:
        assert not any(bad in w.message.lower() for bad in BANNED_WORDS), w.code


# ===========================================================================
# 3) Scope rule, provenance and the rate ladder (SPEC §9.2 tests 22-33)
# ===========================================================================
def test_duplicate_rate_does_not_fire_when_the_duplicate_row_has_no_value():
    """The scope rule, stated once and tested both ways.

    Checks about the FIT evaluate the fitted set. The second 20 °C/min row is
    the one whose baseline failed, so it is not in the fit and claiming a
    duplicated rate would be false. Give that row a value and the code starts
    firing — correctly.
    """
    series = _bm_like_series()
    assert "duplicate_rate" not in set(w.code for w in rs.check_series(series))

    series.rows[1].set_manual_value(683.5)
    codes = set(w.code for w in rs.check_series(series))
    assert "duplicate_rate" in codes
    dup = next(w for w in rs.check_series(series) if w.code == "duplicate_rate")
    assert dup.detail["n_rows"] == 2
    assert dup.severity == "warn"


def test_rate_mismatch_silent_on_bm_measured_vs_programmed():
    """19.96/20, 19.95/20, 14.99/15, 9.99/10 are all inside 5 %: the program
    clock agreeing with the furnace is a CHECKED fact, not an assumption."""
    series = rs.RateSeries(quantity_kind="Tg")
    for meas, prog, tg in ((19.96, 20.0, 620.8), (19.95, 20.0, 683.5),
                           (14.99, 15.0, 657.1), (9.99, 10.0, 634.1)):
        row = rs.manual_row(sample="BM", value_C=tg, specimen_key=BM_SPECIMEN)
        row.q_measured_C_per_min = meas
        row.q_programmed_C_per_min = prog
        series.add(row)
    codes = set(w.code for w in rs.check_series(series))
    assert "rate_mismatch" not in codes
    assert all(r.q_used_source == "measured" for r in series.rows)

    # and it DOES fire when the furnace really did not follow the program
    series.rows[0].q_measured_C_per_min = 12.0
    warns = rs.check_series(series)
    mismatch = next(w for w in warns if w.code == "rate_mismatch")
    assert mismatch.severity == "warn"
    assert mismatch.detail["worst_pct"] == pytest.approx(40.0, abs=0.5)


def test_five_separate_specimens_produce_no_warn_only_info():
    """The intended design must be silent. Anything louder than `info` here
    would be an alarm on a clean measurement, which trains users to ignore
    the banner."""
    res = rs.fit_series(_clean_series(), "lasocka")
    codes = set(res.codes)
    assert res.ok is True
    assert res.lasocka_B == pytest.approx(8.0, abs=1e-9)
    assert "separate_specimens" in codes
    assert dk.max_severity(res.all_warnings) == "info"
    assert res.warnings == []          # nothing above info at all
    assert all(w.glyph == "ℹ" for w in res.all_warnings)


def test_mixed_design_warns_when_one_specimen_contributes_two_rows():
    """One specimen giving two fitted rows next to separate specimens is the
    confound `mixed_design` keys on."""
    series = _clean_series(rates=(5.0, 10.0, 20.0))
    extra = rs.manual_row(sample="G5", q_C_per_min=30.0, value_C=530.0,
                          specimen_key=series.rows[0].specimen_key, prior_heats=1,
                          state="after-cycle-1")
    series.add(extra)
    codes = set(w.code for w in rs.check_series(series))
    assert "mixed_design" in codes
    assert "single_specimen_cycles" not in codes
    assert "separate_specimens" not in codes


def test_mixed_sources_is_not_a_warning_code():
    """Deliberate omission, asserted so it stays deliberate.

    Mixing a single-ramp file with a segment of a cyclic run is harmless in
    itself; alarming on the COUNT of `source_kind` values would fire on this
    feature's own intended use.
    """
    series = _clean_series(rates=(5.0, 10.0, 20.0))
    series.rows[0].source_kind = "file"
    series.rows[1].source_kind = "segment"
    series.rows[2].source_kind = "library"
    codes = set(w.code for w in rs.check_series(series))
    assert "mixed_sources" not in codes
    # no emitter anywhere: the only mention in the module is the paragraph
    # explaining the omission, which never quotes it as a code string
    assert inspect.getsource(rs).count('"mixed_sources"') == 0


def test_single_rate_and_too_few_points_return_ok_false_without_raising():
    """`fit_series` never raises; arithmetic impossibility is a reported
    result, so a batch loop keeps its other series and the UI can print the
    reason in the panel that would have shown the fit."""
    empty = rs.RateSeries(quantity_kind="Tg")
    res = rs.fit_series(empty, "lasocka")
    assert res.ok is False
    assert res.message
    assert "too_few_points" in res.codes

    one_rate = rs.RateSeries(quantity_kind="Tg")
    for tg in (600.0, 610.0, 620.0):
        one_rate.add(rs.manual_row(sample="x", q_C_per_min=10.0, value_C=tg,
                                   specimen_key=f"s{tg}|1|d"))
    res = rs.fit_series(one_rate, "lasocka")
    assert res.ok is False
    assert "single_rate" in res.codes
    assert np.isnan(res.slope)

    # fit_all does not raise either — and it shows why `single_rate` has to
    # be a CRITICAL warning rather than a gate. Lasocka's x is ln q, which is
    # constant here, so its arithmetic really is impossible. Moynihan's and
    # Kissinger-on-Tg's x is 1/Tg, which is NOT constant, so they return a
    # number: Moynihan a slope of exactly 0 (y = ln q is the constant), and
    # kissinger_tg an R² of 0.99999 on three points that carry no rate
    # information at all. Measured, and the reason R² is never the gate.
    fits = {f.method: f for f in rs.fit_all(one_rate)}
    assert fits["lasocka"].ok is False
    assert fits["moynihan"].ok is True and fits["moynihan"].slope == 0.0
    assert fits["kissinger_tg"].ok is True
    assert fits["kissinger_tg"].r2 > 0.999
    for fit in fits.values():
        assert "single_rate" in fit.codes
        assert dk.max_severity(fit.all_warnings) == "critical"


def test_resolve_rate_ladder_prefers_measured_then_programmed_then_filename():
    """measured -> programmed -> filename -> manual, in that order."""
    row = rs.RateRow()
    value, source = rs.resolve_rate(row)
    assert np.isnan(value) and source == "none"
    row.q_manual_C_per_min = 4.0
    assert rs.resolve_rate(row) == (4.0, "manual")
    row.q_filename_C_per_min = 3.0
    assert rs.resolve_rate(row) == (3.0, "filename")
    row.q_programmed_C_per_min = 2.0
    assert rs.resolve_rate(row) == (2.0, "programmed")
    row.q_measured_C_per_min = 1.0
    assert rs.resolve_rate(row) == (1.0, "measured")
    assert row.q_C_per_min == 1.0
    assert row.q_used_source == "measured"


def test_resolve_rate_reports_the_source_actually_used():
    """Pinning a candidate that is not there falls back down the ladder AND
    says so — the same honesty rule `TgParallelTangentResult.low_mode`
    follows. Returning "programmed" for a measured number would silence
    `rate_mismatch` forever, because it would compare a value with itself."""
    row = rs.RateRow(q_measured_C_per_min=19.95, rate_source="programmed")
    value, source = rs.resolve_rate(row)
    assert value == 19.95
    assert source == "measured"

    row.q_programmed_C_per_min = 20.0
    assert rs.resolve_rate(row) == (20.0, "programmed")


def test_rows_are_never_sorted_in_place():
    """`fitted_rows()` is a sorted COPY. The live list stays in table order,
    so "Row 2" in a warning still means the second line on screen."""
    series = _bm_like_series()
    before = [r.uid for r in series.rows]
    rs.check_series(series)
    rs.fit_series(series, "lasocka")
    series.rows[0].include = False
    rs.fit_series(series, "lasocka")
    assert [r.uid for r in series.rows] == before

    fitted = series.fitted_rows()
    assert [abs(r.q_C_per_min) for r in fitted] == sorted(abs(r.q_C_per_min)
                                                          for r in fitted)
    assert fitted is not series.rows
    fitted.reverse()
    assert [r.uid for r in series.rows] == before


def test_cooling_row_is_fitted_with_abs_rate_and_warns():
    """A cooling segment is added and FLAGGED, never silently dropped and
    never silently kept: the magnitude is a perfectly good lever arm and the
    fact that it was a cool is a warning."""
    series = rs.RateSeries(quantity_kind="Tg")
    for q, tg in ((-10.0, 600.0), (-15.0, 606.0), (-20.0, 610.0)):
        row = rs.manual_row(sample="cool", q_C_per_min=q, value_C=tg,
                            specimen_key=f"c{abs(q)}|1|d")
        row.segment_kind = ds.KIND_COOL
        row.role_in_cycle = ds.ROLE_COOL
        series.add(row)
    res = rs.fit_series(series, "lasocka")
    assert res.ok is True
    assert "cooling_row" in res.codes
    assert np.all(res.x > 0)                      # ln|q|, never ln of a negative
    assert res.x == pytest.approx(np.log([10.0, 15.0, 20.0]))
    assert res.lasocka_B > 0                      # fitted, with the magnitudes
    assert "cooling" in series.rows[0].flags


def test_unit_suspect_catches_the_273_15_fingerprint():
    """The runtime net under the structural "Celsius only" rule: a value
    converted to kelvin twice still looks like a plausible Tg on a plot, and
    makes Ea wrong by roughly a factor of three."""
    series = _clean_series(rates=(5.0, 10.0, 20.0))
    series.rows[1].value_C = series.rows[0].value_C + 273.15
    warns = rs.check_series(series)
    suspect = next(w for w in warns if w.code == "unit_suspect")
    assert suspect.severity == "critical"
    assert "273" in suspect.message or "273.1" in suspect.message
    # still fitted
    assert rs.fit_series(series, "lasocka").ok is True

    # and an implausible rate is caught too
    other = _clean_series(rates=(5.0, 10.0, 20.0))
    other.rows[0].q_manual_C_per_min = 5000.0
    assert "unit_suspect" in set(w.code for w in rs.check_series(other))


def test_mixed_window_warns():
    """A window moves Tg by 10-20 K on this data, larger than the whole
    rate-induced shift over a 2x rate range — a series with inconsistent
    windows measures the window, not the rate."""
    series = _clean_series(rates=(5.0, 10.0, 20.0))
    series.rows[0].window_C = (350.0, 700.0)
    series.rows[1].window_C = (350.0, 700.0)
    series.rows[2].window_C = (550.0, 720.0)
    warns = rs.check_series(series)
    mixed = next(w for w in warns if w.code == "mixed_window")
    assert mixed.severity == "warn"
    assert rs.fit_series(series, "lasocka").ok is True


def test_mixed_method_warns():
    """Two constructions of one quantity scatter systematically, and that
    scatter enters the rate slope."""
    series = _clean_series(rates=(5.0, 10.0, 20.0))
    series.rows[0].method = "parallel"
    series.rows[1].method = "double"
    series.rows[2].method = "parallel"
    codes = set(w.code for w in rs.check_series(series))
    assert "mixed_method" in codes
    assert "mixed_quantity" not in codes      # orthogonal fields, orthogonal codes


def test_mixed_quantity_is_critical():
    """A Tg and a Tp are different temperatures; one slope across them has no
    physical meaning — and it is still computed."""
    series = _clean_series(rates=(5.0, 10.0, 20.0))
    series.rows[2].quantity_kind = "Tp"
    warns = rs.check_series(series)
    mixed = next(w for w in warns if w.code == "mixed_quantity")
    assert mixed.severity == "critical"
    res = rs.fit_series(series, "lasocka")
    assert res.ok is True
    assert np.isfinite(res.lasocka_B)


def test_model_quantity_mismatch_warns_and_still_fits():
    """Kissinger asked for on a Tg series: the code is present AND the number
    is produced."""
    series = _clean_series()
    res = rs.fit_series(series, "kissinger")
    assert "model_quantity_mismatch" in res.codes
    assert res.ok is True
    assert np.isfinite(res.Ea_kJ_mol)


def test_series_to_dict_from_dict_round_trip_is_value_identical():
    """Including uid, created_at, extras, settings, window tuple and
    tolerances — a project reopened must be the same series, not a similar
    one."""
    series = _bm_like_series()
    series.rows[0].extras = {"HF_max_mW": 32.15}
    series.rows[0].settings = {"tg_method": "parallel", "low_mode": "auto"}
    series.notes = "the real run"
    payload = series.to_dict()
    clone = rs.RateSeries.from_dict(payload)

    assert clone.to_dict() == payload
    assert clone.series_id == series.series_id
    assert clone.created_at == series.created_at
    assert [r.uid for r in clone.rows] == [r.uid for r in series.rows]
    assert clone.rows[0].extras == {"HF_max_mW": 32.15}
    assert clone.rows[0].settings == {"tg_method": "parallel", "low_mode": "auto"}
    assert clone.rows[0].window_C == (350.0, 700.0)
    assert isinstance(clone.rows[0].window_C, tuple)
    assert clone.tolerances == series.tolerances
    assert payload["schema_version"] == rs.SERIES_SCHEMA_VERSION


# ===========================================================================
# 4) CSV (SPEC §9.2 tests 35-40)
# ===========================================================================
def test_rows_csv_has_value_C_and_value_K_and_no_temp_unit_column(tmp_path):
    """Both units present, both NAMED, and no `temp_unit` column to misread.
    A deliberate divergence from `_current_result_row()`."""
    series = _bm_like_series()
    warns = rs.check_series(series)
    path = rs.write_series_csv(series, warns, tmp_path / "s_rows.csv")
    frame = pd.read_csv(path, encoding="utf-8-sig")

    assert "value_C" in frame.columns and "value_K" in frame.columns
    assert "temp_unit" not in frame.columns
    assert "value_unit" not in frame.columns
    assert list(frame.columns) == list(rs.ROWS_CSV_COLUMNS)
    assert frame["value_C"][0] == pytest.approx(620.8)
    assert frame["value_K"][0] == pytest.approx(620.8 + 273.15)
    # an absent value is blank, never 0 and never the text "nan"
    assert pd.isna(frame["value_C"][1])
    assert frame["error"][1] == HEAT2_ERROR


def test_every_rows_csv_row_carries_series_wide_warning_codes(tmp_path):
    """A single-row extract from an export must never look innocent.

    Two halves, because there are two ways a code reaches a row's cell: a
    warning that names the row, and a warning that names no row at all.
    """
    series = _bm_like_series()
    warns = rs.check_series(series)
    table = rs.rows_table(series, warns)
    design_codes = {"thermal_history", "non_monotonic", "narrow_rate_span",
                    "single_specimen_cycles"}
    for rec, row in zip(table, series.rows):
        codes = set(str(rec["warning_codes"]).split(";"))
        assert rec["max_severity"] in ("critical", "warn", "info")
        if row.fittable:
            assert design_codes <= codes, rec["row_index"]
        else:
            assert "no_value" in codes        # the blank row is named, by uid
        assert rec["warning_messages"]

    # a genuinely series-wide warning (no row_uids at all) lands on EVERY row
    lonely = rs.RateSeries(quantity_kind="Tg")
    lonely.add(rs.manual_row(sample="only", q_C_per_min=10.0, value_C=600.0))
    lonely.add(rs.manual_row(sample="blank", q_C_per_min=20.0))
    lonely_warns = rs.check_series(lonely)
    assert any(w.code == "too_few_points" and not w.row_uids for w in lonely_warns)
    for rec in rs.rows_table(lonely, lonely_warns):
        assert "too_few_points" in str(rec["warning_codes"]).split(";")

    path = rs.write_series_csv(series, warns, tmp_path / "s_rows.csv")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    fitted = frame[frame["fitted"]]
    assert len(fitted) == 3
    assert all("thermal_history" in str(v) for v in fitted["warning_codes"])
    assert all("non_monotonic" in str(v) for v in fitted["warning_codes"])


def test_series_csv_round_trip_restores_values_include_flags_and_rates(tmp_path):
    """Lossless on everything that matters, with FRESH uids: loading one CSV
    twice must not make two rows every uid-keyed lookup confuses."""
    series = _bm_like_series()
    series.rows[3].include = False
    series.rows[0].note = "keep me"
    warns = rs.check_series(series)
    path = rs.write_series_csv(series, warns, tmp_path / "s_rows.csv")

    loaded, load_warns = rs.read_series_csv(path)
    assert len(loaded.rows) == 4
    assert [r.include for r in loaded.rows] == [True, True, True, False]
    assert loaded.rows[0].value_C == pytest.approx(620.8)
    assert not np.isfinite(loaded.rows[1].value_C)
    assert loaded.rows[1].error == HEAT2_ERROR
    assert [round(abs(r.q_C_per_min), 3) for r in loaded.rows] == [20.0, 20.0, 15.0, 10.0]
    assert loaded.rows[0].specimen_key == BM_SPECIMEN
    assert loaded.rows[0].note == "keep me"
    assert loaded.rows[0].prior_heats == 0 and loaded.rows[3].prior_heats == 3
    assert {r.uid for r in loaded.rows}.isdisjoint({r.uid for r in series.rows})
    assert "loaded_from_csv" in {w.code for w in load_warns}

    before = rs.fit_series(series, "lasocka")
    after = rs.fit_series(loaded, "lasocka")
    assert after.lasocka_B == pytest.approx(before.lasocka_B, abs=1e-12)


def test_read_series_csv_converts_a_value_K_column_to_celsius_and_drops_the_unit(tmp_path):
    """Kelvin can enter only here, and leaves immediately: nothing downstream
    ever sees a stored kelvin value."""
    path = tmp_path / "k_rows.csv"
    path.write_text(
        "sample,q_manual_C_per_min,value_K,include\n"
        "A,10,893.95,True\n"
        "B,20,903.95,True\n", encoding="utf-8-sig")
    loaded, _warns = rs.read_series_csv(path)
    assert [r.value_C for r in loaded.rows] == pytest.approx([620.8, 630.8], abs=1e-9)
    assert all(r.value_source in rs.VALUE_SOURCES for r in loaded.rows)
    # the unit itself is gone — there is no field to put it in
    assert not hasattr(loaded.rows[0], "value_unit")
    assert not hasattr(loaded.rows[0], "temp_unit")


def test_read_series_csv_flags_a_missing_source_file_but_keeps_the_value(tmp_path):
    """A stored Tg is a MEASUREMENT, not a cache."""
    path = tmp_path / "m_rows.csv"
    path.write_text(
        "sample,file,q_manual_C_per_min,value_C,include\n"
        f"A,{tmp_path / 'gone.txt'},10,620.8,True\n", encoding="utf-8-sig")
    loaded, warns = rs.read_series_csv(path)
    assert loaded.rows[0].value_C == pytest.approx(620.8)
    assert "missing_source_file" in {w.code for w in warns}
    missing = next(w for w in warns if w.code == "missing_source_file")
    assert missing.severity == "warn"
    assert "gone.txt" in missing.message


def test_fit_csv_has_one_row_per_evaluable_method_with_max_severity_first(tmp_path):
    """One file, several models side by side, model-irrelevant columns blank."""
    series = _bm_like_series()
    warns = rs.check_series(series)
    fits = rs.fit_all(series)
    assert [f.method for f in fits] == ["lasocka", "moynihan", "kissinger_tg"]

    table = rs.fit_table(series, fits, warns)
    assert len(table) == 3
    assert list(table[0].keys()) == list(rs.FIT_CSV_LEAD)
    cols = list(rs.FIT_CSV_LEAD)
    for later in ("n_warnings", "warning_codes", "warning_messages"):
        assert cols.index("max_severity") < cols.index(later)

    lasocka, moynihan = table[0], table[1]
    assert lasocka["Ea_kJ_mol"] == ""            # blank, not 0, not "nan"
    assert moynihan["lasocka_B"] == ""
    assert lasocka["lasocka_B"] == pytest.approx(-14.2053, abs=1e-4)
    assert lasocka["max_severity"] == "critical"
    assert lasocka["monotonic"] is False
    assert lasocka["specimen_keys"] == BM_SPECIMEN
    assert lasocka["prior_heats_used"] == "0;2;3"
    assert lasocka["generated_by"] == rs.GENERATED_BY

    path = rs.write_fit_csv(series, fits, warns, tmp_path / "s_fit.csv")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    assert len(frame) == 3
    assert list(frame.columns) == list(rs.FIT_CSV_LEAD)


def test_warnings_csv_detail_json_carries_the_threshold_that_fired(tmp_path):
    """A reviewer must be able to tell a tripped DEFAULT from a tripped
    physical limit without reading the source."""
    series = _bm_like_series()
    warns = rs.check_series(series)
    table = rs.warnings_table(series, warns)
    assert list(table[0].keys()) == list(rs.WARNINGS_CSV_COLUMNS)
    span = next(r for r in table if r["code"] == "narrow_rate_span")
    detail = json.loads(span["detail_json"])
    assert detail["span_ratio"] == pytest.approx(2.0, abs=1e-6)
    assert detail["span_warn"] == 4.0
    assert detail["span_critical"] == 2.0
    assert span["row_indices"]                   # 1-based, matches the table
    path = rs.write_warnings_csv(series, warns, tmp_path / "s_warnings.csv")
    assert Path(path).stat().st_size > 0


# ===========================================================================
# 5) The never-block contract, stated as tests
# ===========================================================================
def _warning_battery():
    """Every warning this module can emit, from a handful of series.

    Built once and reused so the "no refusal wording" and "ends Fitted
    anyway" laws are checked against the real message strings rather than
    against a list someone kept up to date by hand.
    """
    out = []
    out += rs.check_series(_bm_like_series())
    out += rs.check_series(_clean_series())

    cool = rs.RateSeries(quantity_kind="Tg")
    for q, tg in ((-10.0, 600.0), (-20.0, 610.0)):
        row = rs.manual_row(sample="c", q_C_per_min=q, value_C=tg,
                            specimen_key=f"c{abs(q)}|1|d", window_C=(350.0, 400.0))
        row.segment_kind = ds.KIND_COOL
        row.is_partial = True
        row.setpoint_reached = False
        row.q_measured_C_per_min = q
        row.q_programmed_C_per_min = -q * 2.0
        row.signal = "Heat Flow (mW)" if q == -10.0 else "Temperature Difference (°C)"
        row.method = "parallel" if q == -10.0 else "deriv"
        cool.add(row)
    cool.rows[1].quantity_kind = "Tp"
    cool.rows[1].window_C = (500.0, 900.0)
    out += rs.check_series(cool)

    nothing = rs.RateSeries(quantity_kind="Tg")
    nothing.add(rs.manual_row(sample="n"))
    out += rs.check_series(nothing)

    for series in (_bm_like_series(), _clean_series(), cool):
        for fit in rs.fit_all(series):
            out += fit.all_warnings
    out += rs.fit_series(_clean_series(), "kissinger").all_warnings
    return out


def test_no_warning_text_reads_as_a_refusal():
    """The wording law. "cannot" / "invalid" / "refusing" are banned because
    the user explicitly asked to be able to process this data."""
    battery = _warning_battery()
    assert len(battery) > 20
    for w in battery:
        low = w.message.lower()
        for bad in BANNED_WORDS:
            assert bad not in low, f"{w.code}: {w.message}"


def test_every_refusable_warning_says_fitted_anyway():
    """Any warning a reader could mistake for a refusal ends "Fitted anyway.".

    The exempt codes and why: `too_few_points` and `single_rate` really are
    the no-fit cases (the arithmetic did not happen, so saying it did would
    be a lie); `no_value`, `no_rate` and `no_heating_segment` say in their own
    words that the remaining rows are fitted; every `info` code is not a
    refusal to begin with.
    """
    seen = set()
    for w in _warning_battery():
        seen.add(w.code)
        if w.code in MUST_SAY_FITTED_ANYWAY:
            assert w.message.rstrip().endswith("Fitted anyway."), \
                f"{w.code}: {w.message}"
    # the battery really did exercise most of the table
    assert len(seen & MUST_SAY_FITTED_ANYWAY) >= 12


def test_check_series_never_raises_and_never_filters():
    """Degenerate input produces warnings, not exceptions, and the row list
    comes out exactly as it went in."""
    series = rs.RateSeries(quantity_kind="Tg")
    series.add(rs.manual_row(sample="nan row"))
    series.add(rs.manual_row(sample="huge", q_C_per_min=1e9, value_C=1e9))
    series.add(rs.manual_row(sample="zero", q_C_per_min=0.0, value_C=600.0))
    series.add(rs.manual_row(sample="cold", q_C_per_min=10.0, value_C=-400.0))
    before = list(series.rows)
    warns = rs.check_series(series)
    assert isinstance(warns, list)
    assert series.rows == before
    assert {"no_value", "no_rate", "unit_suspect"} <= {w.code for w in warns}
    # sorted worst first
    order = {"critical": 0, "warn": 1, "info": 2}
    levels = [order[w.severity] for w in warns]
    assert levels == sorted(levels)


def test_fit_series_never_raises_on_anything():
    """Including None, an empty series, a broken enum and a nonsense model."""
    assert rs.fit_series(None).ok is False
    assert rs.fit_series(rs.RateSeries()).ok is False
    weird = _clean_series(rates=(5.0, 10.0))
    weird.quantity_kind = "not a quantity"
    assert rs.fit_series(weird).ok is True          # falls back to lasocka
    res = rs.fit_series(_clean_series(), "no_such_model")
    assert res.ok is False
    assert "no_such_model" in res.message


def test_fit_series_never_consults_warnings_before_computing():
    """Thirty critical warnings must not change one digit of the answer.

    The guard against a future "if warnings: return" creeping into the fit
    path, which is exactly the refusal the user ruled out.
    """
    series = _clean_series()
    baseline = rs.fit_series(series, "lasocka")
    noise = [dk.SeriesWarning(code=f"fake_{i}", severity="critical",
                              message=f"Fake objection {i}. Fitted anyway.")
             for i in range(30)]
    loud = rs.fit_series(series, "lasocka", warnings_in=noise)
    assert loud.ok is True
    assert loud.lasocka_B == pytest.approx(baseline.lasocka_B, abs=0.0)
    assert loud.n_fitted == baseline.n_fitted
    assert dk.max_severity(loud.all_warnings) == "critical"


def test_stubbed_warning_codes_are_named_not_silently_absent():
    """A check that was not built must say so. `hidden_excluded` depends on a
    Qt "Show excluded" toggle that has no field in this model."""
    assert rs.STUBBED_WARNING_CODES == ("hidden_excluded",)
    source = inspect.getsource(rs)
    for code in rs.STUBBED_WARNING_CODES:
        assert source.count(f'"{code}"') == 1      # only the declaration


# ===========================================================================
# 6) Structural guards (MERGED_DECISIONS M3 and M4)
# ===========================================================================
def test_seg_fields_is_the_only_place_dta_segments_field_names_appear():
    """MERGED_DECISIONS M3 / SPEC Risk R1, as a source assertion.

    `Segment.rate_mode` is a MEASURED estimator whose name invites
    `q_programmed_C_per_min = seg.rate_mode`, which would put the wrong x on
    every Arrhenius plot and permanently silence `rate_mismatch` and
    `setpoint_not_reached`. Keeping every read in one adapter is what makes
    that mistake a one-line review rather than a hunt.
    """
    source = inspect.getsource(rs)
    adapter = inspect.getsource(rs._seg_fields)
    for name in ('"rate_mode"', '"rate_nominal_C_min"', '"programmed_rate"',
                 '"rate_mean"'):
        assert source.count(name) == 1, name
        assert name in adapter, name
    assert source.count('"target_gap_C"') == 2
    assert adapter.count('"target_gap_C"') == 2


def test_the_rate_series_never_builds_its_own_arrays():
    """MERGED_DECISIONS M4 / SPEC Risk R2: no `argsort` anywhere in this
    module. A whole-file sort on a cyclic record interleaves four heating
    ramps and turns a 620 °C Tg into 432 °C."""
    source = inspect.getsource(rs)
    assert "argsort(" not in source                # no call, anywhere
    assert "np.sort(" not in source
    assert "run.xy(" in source                     # the one array path


def test_q_programmed_is_nan_without_a_program(cycles_fixture):
    """The R1 guard, retargeted per MERGED_DECISIONS M3: `rate_mode` is
    finite, and `q_programmed_C_per_min` is still NaN, because nothing
    programmed was supplied."""
    _header, df, _run, path = cycles_fixture
    rows, _warns = rs.rows_from_file(path, use_program=False)
    assert len(rows) == 4
    assert all(np.isfinite(r.q_measured_C_per_min) for r in rows)
    assert all(not np.isfinite(r.q_programmed_C_per_min) for r in rows)
    assert all(r.setpoint_reached is None for r in rows)
    assert all(r.q_used_source == "measured" for r in rows)
    assert "rate_mismatch" not in {w.code for w in rs.check_series(
        _series_of(rows))}


def _series_of(rows):
    """A one-line series wrapper, so a builder's output can be checked."""
    series = rs.RateSeries(quantity_kind="Tg")
    series.extend(rows)
    return series


def test_row_start_comes_from_row_indices_not_from_segment_row_start(cycles_fixture):
    """MERGED_DECISIONS M3's real teeth, tested where the two differ.

    `Segment.row_start` indexes the CLEANED arrays. On every shipped file the
    cleaning drops nothing, so the confusion is invisible; inject five
    non-finite temperatures and the two indexings separate, and only
    `row_indices` still points at the right DataFrame rows.
    """
    _header, df, _run, path = cycles_fixture
    header, frame = _load(path)
    frame = frame.copy()
    tcol = frame.columns[1]
    for pos in (500, 1000, 1500, 2000, 2500):
        frame.loc[pos, tcol] = np.nan

    run = rs._segment_dataframe(frame, program=_program(header))
    assert run.n_rows == 4749
    assert run.keep_index.size == 4744

    rows, _warns = rs.rows_from_segments(run, frame, header=header, file=str(path))
    assert len(rows) == 4
    shifted = 0
    for row, seg in zip(rows, run.by_kind(ds.KIND_HEAT)):
        idx = run.row_indices(seg)
        assert row.row_start == int(idx[0])
        assert row.row_stop == int(idx[-1]) + 1
        if row.row_start != seg.row_start:
            shifted += 1
    assert shifted >= 1, "the injection must actually separate the two indexings"


def test_measured_rows_come_from_prepare_segment_xy(cycles_fixture):
    """SPEC test 66, retargeted per MERGED_DECISIONS M4: the arrays the
    measurement sees are this segment's own rows of the DataFrame, in row
    order, never a globally sorted view."""
    header, df, run, path = cycles_fixture
    seen = {}

    def _capture(x, y, window):
        """Record the arrays the measurement was handed, then measure."""
        seen["x"] = np.array(x)
        seen["y"] = np.array(y)
        return _midpoint_measure(x, y, window)

    rows, _warns = rs.rows_from_segments(run, df, header=header, file=str(path),
                                         measure=_capture, window_C=(350.0, 700.0))
    last = rows[-1]
    seg = run.by_kind(ds.KIND_HEAT)[-1]
    idx = run.row_indices(seg)
    expected = df.iloc[idx]
    tcol, ycol = df.columns[1], df.columns[4]
    assert seen["x"].size == len(expected)
    assert np.allclose(seen["x"], pd.to_numeric(expected[tcol]).to_numpy(float))
    assert np.allclose(seen["y"], pd.to_numeric(expected[ycol]).to_numpy(float))
    # a heating ramp is monotone in T, so the per-segment sort is a no-op and
    # the prepared x is also in original ROW order
    assert np.all(np.diff(seen["x"]) > 0)
    assert last.value_source == "computed"
    assert 350.0 <= last.value_C <= 700.0


# ===========================================================================
# 7) The four add paths
# ===========================================================================
def test_five_separate_files_become_five_rows_with_the_rates_read_for_you(tmp_path):
    """The headline of the feature request: five files at 5/10/15/20/30
    °C/min, five rows, no typing."""
    series = rs.RateSeries(name="five files", quantity_kind="Tg")
    add_warnings = []
    for path in _five_files(tmp_path):
        rows, warns = rs.rows_from_file(path, filename_template="GL_q???.txt",
                                        measure=_midpoint_measure,
                                        window_C=(500.0, 700.0))
        series.extend(rows)
        add_warnings += warns

    assert len(series.rows) == 5
    assert add_warnings == []
    assert [round(r.q_C_per_min, 6) for r in series.rows] == [5.0, 10.0, 15.0, 20.0, 30.0]
    assert all(r.q_used_source == "measured" for r in series.rows)
    assert all(r.source_kind == "file" for r in series.rows)
    assert all(r.n_heating_segments_in_source == 1 for r in series.rows)
    assert all(r.include for r in series.rows)
    # the programmed rate agrees, and the filename template also read it
    assert [r.q_programmed_C_per_min for r in series.rows] == [5.0, 10.0, 15.0, 20.0, 30.0]
    assert [r.q_filename_C_per_min for r in series.rows] == [5.0, 10.0, 15.0, 20.0, 30.0]
    assert len(series.specimen_keys()) == 5
    assert series.rate_span_ratio() == pytest.approx(6.0, rel=1e-6)

    res = rs.fit_series(series, "lasocka")
    assert res.ok is True
    assert res.n_fitted == 5
    assert res.lasocka_B > 0
    codes = set(res.codes)
    assert "separate_specimens" in codes
    assert "single_specimen_cycles" not in codes
    assert "thermal_history" not in codes
    assert "narrow_rate_span" not in codes


def test_add_from_a_library_record_uses_the_in_memory_dataframe(tmp_path):
    """Source (c). A record already in the library must not be re-read from
    disk — it still works when the share it came from is offline."""
    path = _five_files(tmp_path)[1]
    header, df = _load(path)
    rec = {"id": "spectrum-123", "title": "GL-10", "path": str(path), "df": df,
           "meta": {"raw_header": header}}
    path.unlink()                                  # the share is gone
    rows, warns = rs.rows_from_record(rec)
    assert len(rows) == 1
    assert warns == []
    assert rows[0].source_kind == "library"
    assert rows[0].record_id == "spectrum-123"
    assert rows[0].sample == "GL-10"
    assert rows[0].q_C_per_min == pytest.approx(10.0, abs=1e-6)


def test_add_from_a_record_with_neither_df_nor_path_is_a_failed_row():
    """One failed row, never an exception and never a silent skip."""
    rows, warns = rs.rows_from_record({"id": "x", "title": "orphan"})
    assert len(rows) == 1
    assert rows[0].value_source == "failed"
    assert rows[0].include is False
    assert "no data" in rows[0].error
    assert {w.code for w in warns} == {"no_heating_segment"}


def test_an_unparseable_file_is_a_failed_row_not_an_exception(tmp_path):
    """A batch of twenty files must not be lost to one bad file."""
    bad = tmp_path / "garbage.txt"
    bad.write_text("this is not a TA export\nnor is this\n", encoding="utf-8")
    rows, warns = rs.rows_from_file(bad)
    assert len(rows) == 1
    assert rows[0].value_source == "failed"
    assert rows[0].error
    assert rows[0].include is False
    assert warns and warns[0].severity == "critical"


def test_manual_rows_need_no_file_at_all():
    """Source (d), and the reason every fit/warning/export test above runs
    with no fixture."""
    row = rs.manual_row(sample="literature", q_C_per_min=10.0, value_C=612.0)
    assert row.source_kind == "manual"
    assert row.value_source == "manual"
    assert row.method == "manual"
    assert row.fittable is True
    assert row.q_used_source == "manual"
    failed = rs.manual_row(sample="broken", q_C_per_min=20.0, error=HEAT2_ERROR)
    assert failed.value_source == "failed"
    assert failed.fittable is False
    assert failed.error == HEAT2_ERROR


def test_a_cyclic_file_yields_every_heating_ramp_switched_off_and_says_why(cycles_fixture):
    """The auto-detection rule's fourth line: several heating ramps at
    DIFFERENT rates are all added, all switched off, and named — the engine
    declines to guess, it does not decline to load."""
    _header, _df, _run, path = cycles_fixture
    rows, warns = rs.rows_from_file(path)
    assert len(rows) == 4
    assert all(r.include is False for r in rows)
    assert all(r.n_heating_segments_in_source == 4 for r in rows)
    amb = next(w for w in warns if w.code == "ambiguous_file_rate")
    assert amb.severity == "info"
    assert "4 heating ramps" in amb.message
    # turn them on and the same file is a perfectly usable 3-rate series
    for r in rows:
        r.include = True
    series = _series_of(rows)
    assert len(series.distinct_rates()) == 3


def test_cooling_segments_are_added_only_when_asked_for(cycles_fixture):
    """Heating only by default; a cool arrives when requested, flagged."""
    header, df, run, path = cycles_fixture
    heats, _w = rs.rows_from_segments(run, df, header=header, file=str(path))
    assert [r.segment_kind for r in heats] == [ds.KIND_HEAT] * 4

    both, _w = rs.rows_from_segments(run, df, header=header, file=str(path),
                                     roles=(ds.ROLE_HEAT, ds.ROLE_COOL))
    kinds = [r.segment_kind for r in both]
    assert kinds.count(ds.KIND_COOL) == 4
    cool = next(r for r in both if r.segment_kind == ds.KIND_COOL)
    assert cool.q_C_per_min < 0
    assert "cooling" in cool.flags

    # `cooling_row` is a check about the FIT, so it needs fitted rows: give
    # the cools values and switch them on, and the warning appears.
    series = _series_of(both)
    for r in series.rows:
        r.include = True
        if r.segment_kind == ds.KIND_COOL:
            r.set_manual_value(600.0)
    codes = {w.code for w in rs.check_series(series)}
    assert "cooling_row" in codes
    assert rs.fit_series(series, "lasocka").ok is True


# ===========================================================================
# 8) Fixture-file provenance (always runs)
# ===========================================================================
def test_fixture_rows_carry_rate_history_and_mass_loss(cycles_fixture):
    """One pass over the committed fixture pinning everything the thermal
    history walk produces."""
    header, df, run, path = cycles_fixture
    rows, _warns = rs.rows_from_segments(run, df, header=header, file=str(path))
    assert len(rows) == 4
    assert [r.prior_heats for r in rows] == [0, 1, 2, 3]
    assert [r.state for r in rows] == ["virgin", "after-cycle-1", "after-cycle-2",
                                       "after-cycle-3"]
    assert [r.cycle_index for r in rows] == [1, 2, 3, 4]
    assert [r.segment_label for r in rows] == [
        "heat 1 (cycle 1)", "heat 2 (cycle 2)", "heat 3 (cycle 3)", "heat 4 (cycle 4)"]
    assert [r.row_start for r in rows] == list(FIXTURE_HEAT_ROW_STARTS)
    for row, rate in zip(rows, FIXTURE_HEAT_RATES):
        assert row.q_measured_C_per_min == pytest.approx(rate, abs=0.01)
    assert [r.q_programmed_C_per_min for r in rows] == [20.0, 20.0, 15.0, 10.0]
    # Heating ramps REACH 1300 C. Each hands over to its soak 5-8 C short and
    # the soak finishes the approach (the holds top out at 1298.6 C), so the
    # gap is measured at the end of the ramp+soak block and comes out ~1.4 C.
    # Judged at the ramp's own last point this was False on all four rows --
    # i.e. `setpoint_not_reached` fired on 100 % of real heating ramps.
    # See dta_segments._extend_ramp_gaps_through_soak.
    assert all(r.setpoint_reached is True for r in rows)
    losses = [r.cum_mass_loss_pct for r in rows]
    assert losses == pytest.approx([0.2585, 0.6678, 0.8473, 1.0298], abs=1e-3)
    assert all(b > a for a, b in zip(losses, losses[1:]))
    assert all(0.0 < v < 2.0 for v in losses)
    assert all(r.sample_mass_mg == pytest.approx(97.8930, abs=1e-4) for r in rows)
    assert all(r.specimen_key == BM_SPECIMEN for r in rows)
    assert rows[0].prior_max_T_C != rows[0].prior_max_T_C     # NaN before heat 1
    assert rows[1].prior_max_T_C == pytest.approx(1298.6, abs=1.0)


def test_specimen_key_from_header_is_not_the_path(cycles_fixture):
    """Two exports of one run are ONE specimen; two specimens of one
    composition on one day are two."""
    header, _df, _run, path = cycles_fixture
    key = rs.specimen_key_from_header(header, str(path))
    assert key == BM_SPECIMEN
    assert str(path) not in key
    other = rs.specimen_key_from_header(header, r"D:\somewhere\else\copy.txt")
    assert other == key
    assert rs.specimen_key_from_header({}, r"D:\x\only_a_name.txt") == "only_a_name||"


def test_fixture_series_is_non_monotonic_and_still_fits(cycles_fixture):
    """The committed fixture, measured end to end, reproduces the real
    conclusion: the Tg ordering is not a rate effect, and a number comes out
    anyway."""
    header, df, run, path = cycles_fixture
    rows, _warns = rs.rows_from_segments(run, df, header=header, file=str(path),
                                         measure=_tg_measure((350.0, 700.0)),
                                         window_C=(350.0, 700.0))
    for r in rows:
        r.include = True
    series = _series_of(rows)
    series.name = "fixture heats"
    res = rs.fit_series(series, "lasocka")
    codes = set(res.codes)
    assert res.ok is True
    assert np.isfinite(res.lasocka_B)
    assert res.n_fitted == 3                      # heat 2 fails at 350-700
    assert rows[1].value_source == "failed"
    assert rows[1].error == HEAT2_ERROR
    assert {"no_value", "non_monotonic", "thermal_history",
            "single_specimen_cycles"} <= codes
    assert "duplicate_rate" not in codes          # the duplicate is the failed row


# ===========================================================================
# 9) Real full-resolution data (skips without the BM folder)
# ===========================================================================
@requires_dta_bm
def test_bm_file_yields_four_heating_segments_at_three_distinct_rates(bm_cycles):
    """SPEC §9.3 test 41, on the un-decimated original."""
    header, df, run, path = bm_cycles
    rows, _warns = rs.rows_from_segments(run, df, header=header, file=str(path))
    assert len(rows) == 4
    for row, nominal in zip(rows, (20.0, 20.0, 15.0, 10.0)):
        assert abs(row.q_C_per_min - nominal) / nominal < 0.05
    for row, rate in zip(rows, FULL_HEAT_RATES):
        assert row.q_measured_C_per_min == pytest.approx(rate, abs=0.01)
    assert [r.row_start for r in rows] == list(FULL_HEAT_ROW_STARTS)
    series = _series_of(rows)
    assert len(series.distinct_rates()) == 3
    assert series.rate_span_ratio() == pytest.approx(2.0, rel=0.05)


@requires_dta_bm
def test_bm_rows_carry_increasing_prior_heats_and_monotone_mass_loss(bm_cycles):
    """SPEC §9.3 test 42. The absolute mass-loss values are NOT pinned — they
    move by a factor of two with the segment boundaries — only the invariants
    the `mass_loss_spread` warning actually rests on."""
    header, df, run, path = bm_cycles
    rows, _warns = rs.rows_from_segments(run, df, header=header, file=str(path))
    assert [r.prior_heats for r in rows] == [0, 1, 2, 3]
    losses = [r.cum_mass_loss_pct for r in rows]
    assert all(np.isfinite(v) for v in losses)
    assert all(b > a for a, b in zip(losses, losses[1:]))
    assert all(0.0 < v < 2.0 for v in losses)
    # the decimated fixture must not have drifted away from the original
    assert losses == pytest.approx([0.2585, 0.6682, 0.8473, 1.0274], abs=1e-3)


@requires_dta_bm
def test_bm_heat2_fails_with_the_verbatim_baseline_message_at_350_700(bm_cycles):
    """SPEC §9.3 test 43. The failure is stored as DATA with the exact
    message, never swallowed and never a vanished row."""
    header, df, run, path = bm_cycles
    rows, _warns = rs.rows_from_segments(run, df, header=header, file=str(path),
                                         measure=_tg_measure((350.0, 700.0)),
                                         window_C=(350.0, 700.0))
    assert len(rows) == 4
    assert rows[1].value_source == "failed"
    assert rows[1].error == HEAT2_ERROR
    assert not np.isfinite(rows[1].value_C)
    assert [r.value_source for r in rows] == ["computed", "failed", "computed",
                                              "computed"]


@requires_dta_bm
def test_bm_heat2_succeeds_in_a_550_720_window(bm_cycles):
    """SPEC §9.3 test 44 — and then `duplicate_rate` starts firing, which is
    the scope rule seen from the other side."""
    header, df, run, path = bm_cycles
    rows, _warns = rs.rows_from_segments(run, df, header=header, file=str(path),
                                         measure=_tg_measure((550.0, 720.0)),
                                         window_C=(550.0, 720.0))
    assert rows[1].value_source == "computed"
    assert rows[1].value_C == pytest.approx(683.5, abs=2.0)
    for r in rows:
        r.include = True
    codes = {w.code for w in rs.check_series(_series_of(rows))}
    assert "duplicate_rate" in codes


@requires_dta_bm
def test_bm_per_segment_tg_bands(bm_cycles):
    """SPEC §9.3 test 45. Bands, never 0.1 K: §0.C measured the boundary
    drift that makes an exact pin a pin on the segmenter."""
    header, df, run, path = bm_cycles
    rows, _warns = rs.rows_from_segments(run, df, header=header, file=str(path),
                                         measure=_tg_measure((350.0, 700.0)),
                                         window_C=(350.0, 700.0))
    for i, (lo, hi) in TG_BANDS.items():
        value = rows[i - 1].value_C
        assert lo < value < hi, f"heat {i}: {value}"


@requires_dta_bm
def test_bm_segment_tg_is_not_the_whole_file_sorted_tg(bm_cycles):
    """SPEC §9.3 test 46 — the regression guard for the original bug.

    Every per-segment Tg is above 600 °C; the same file globally argsort-ed
    by temperature (four heats and four cools interleaved) collapses to about
    432 °C. That 190 K is the whole reason `prepare_segment_xy` exists.
    """
    from dta import dta_science as sci
    header, df, run, path = bm_cycles
    rows, _warns = rs.rows_from_segments(run, df, header=header, file=str(path),
                                         measure=_tg_measure((550.0, 720.0)),
                                         window_C=(550.0, 720.0))
    values = [r.value_C for r in rows]
    assert all(np.isfinite(v) for v in values)
    assert all(v > 600.0 for v in values), values

    T = pd.to_numeric(df[df.columns[1]], errors="coerce").to_numpy(float)
    y = pd.to_numeric(df[df.columns[4]], errors="coerce").to_numpy(float)
    mask = np.isfinite(T) & np.isfinite(y)
    order = np.argsort(T[mask])
    with np.errstate(divide="ignore", invalid="ignore"):
        whole = sci.compute_tg_parallel_improved(T[mask][order], y[mask][order],
                                                 350.0, 700.0).tg
    assert whole == pytest.approx(432.0, abs=5.0)
    assert min(values) - whole > 150.0


@requires_dta_bm
def test_bm_programmed_rate_is_nan_without_an_alignment(bm_cycles):
    """SPEC §9.3 test 47 / Risk R1 on the real file."""
    _header, _df, _run, path = bm_cycles
    rows, _warns = rs.rows_from_file(path, use_program=False)
    assert len(rows) == 4
    assert all(np.isfinite(r.q_measured_C_per_min) for r in rows)
    assert all(np.isnan(r.q_programmed_C_per_min) for r in rows)
    assert all(r.setpoint_reached is None for r in rows)


@requires_dta_bm
def test_bm_heat1_extras_carry_the_crystallization_exotherm(bm_cycles):
    """SPEC §9.3 test 48 — the "not the same material any more" evidence
    riding into the export.

    Measured inside 300-1300 °C: 32.15 mW at 804.5 °C on the virgin heat,
    then 1.27 / 6.09 / 11.13 mW. Over the WHOLE segment heat 2 peaks at
    15.10 mW at 94.4 °C, which is the start-of-ramp transient; the spec's
    "< 15" holds once the window excludes it, and the window is what makes
    the number mean "exotherm".
    """
    header, df, run, path = bm_cycles
    hf = pd.to_numeric(df[df.columns[3]], errors="coerce").to_numpy(float)

    def _measure(x, y, window):
        """Tg only; the exotherm extras are attached by the caller below."""
        value, _extras = _tg_measure((550.0, 720.0))(x, y, window)
        return value, {}

    rows, _warns = rs.rows_from_segments(run, df, header=header, file=str(path),
                                         measure=_measure, window_C=(550.0, 720.0))
    for row, seg in zip(rows, run.by_kind(ds.KIND_HEAT)):
        prep = run.xy(seg, hf, x_kind=ds.X_TEMPERATURE)
        mask = (prep.x >= HF_WINDOW[0]) & (prep.x <= HF_WINDOW[1])
        peak = float(np.nanmax(prep.y[mask]))
        row.extras = {"HF_max_mW": peak,
                      "HF_max_at_C": float(prep.x[mask][int(np.nanargmax(prep.y[mask]))])}
    assert rows[0].extras["HF_max_mW"] > 25.0
    assert rows[0].extras["HF_max_at_C"] == pytest.approx(804.5, abs=5.0)
    assert all(r.extras["HF_max_mW"] < 15.0 for r in rows[1:])

    table = rs.rows_table(_series_of(rows), [])
    assert table[0]["extras"].startswith("HF_max_at_C=804.5")
    assert "HF_max_mW=32.1" in table[0]["extras"]


@requires_dta_bm
def test_bm_series_is_non_monotonic_and_still_fits(bm_cycles):
    """SPEC §9.3 test 49, end to end on the real file: every critical
    objection, and a finite Lasocka B."""
    header, df, run, path = bm_cycles
    rows, _warns = rs.rows_from_segments(run, df, header=header, file=str(path),
                                         measure=_tg_measure((350.0, 700.0)),
                                         window_C=(350.0, 700.0))
    for r in rows:
        r.include = True
    res = rs.fit_series(_series_of(rows), "lasocka")
    codes = set(res.codes)
    assert res.ok is True
    assert np.isfinite(res.lasocka_B)
    assert res.n_fitted == 3
    assert {"non_monotonic", "thermal_history", "no_value",
            "single_specimen_cycles", "narrow_rate_span"} <= codes
    assert dk.max_severity(res.all_warnings) == "critical"


@requires_dta_bm
def test_bm_single_ramp_file_is_one_row_with_nothing_to_type(dta_bm_dir):
    """The clean case, on real data: one file, one heating ramp, one row, the
    rate already filled in."""
    path = dta_bm_dir / "BM-0.txt"
    if not path.exists():
        pytest.skip(f"{path} is not present")
    rows, warns = rs.rows_from_file(path)
    assert len(rows) == 1
    assert warns == []
    row = rows[0]
    assert row.q_C_per_min == pytest.approx(9.993, abs=0.01)
    assert row.q_programmed_C_per_min == pytest.approx(10.0)
    assert row.q_used_source == "measured"
    assert row.include is True
    assert row.n_heating_segments_in_source == 1
    assert row.prior_heats == 0 and row.state == "virgin"
    assert row.specimen_key == "BM-0|85.6990|2026-09-24"
    assert row.sample_mass_mg == pytest.approx(85.6990, abs=1e-4)
    assert row.row_start == 0


@requires_dta_bm
def test_real_files_of_two_specimens_mix_into_one_series(dta_bm_dir, cycles_fixture):
    """Sources (a) and (b) in one series: the cyclic run's heats plus a
    separate single-ramp specimen. The mixture is allowed AND flagged."""
    cyclic = dta_bm_dir / "BM-0-cycles.txt"
    single = dta_bm_dir / "BM-0.txt"
    if not (cyclic.exists() and single.exists()):
        pytest.skip("both BM files are needed")
    series = rs.RateSeries(name="mixed", quantity_kind="Tg")
    for path in (cyclic, single):
        rows, _warns = rs.rows_from_file(path, measure=_tg_measure((550.0, 720.0)),
                                         window_C=(550.0, 720.0))
        for r in rows:
            r.include = True
        series.extend(rows)
    assert len(series.rows) == 5
    assert len(series.specimen_keys()) == 2
    codes = {w.code for w in rs.check_series(series)}
    assert "mixed_design" in codes
    assert "single_specimen_cycles" not in codes
    assert "mixed_sources" not in codes
    res = rs.fit_series(series, "lasocka")
    assert res.ok is True
    assert res.n_fitted == 5


# ===========================================================================
# 10) Small units
# ===========================================================================
def test_rate_row_flags_and_label_read_like_a_sentence():
    """The two strings a user actually reads: the legend and the flag list."""
    row = rs.manual_row(sample="BM-0-cycles", q_C_per_min=20.0, value_C=620.8)
    row.segment_label = "heat 1 (cycle 1)"
    assert row.label == "BM-0-cycles · heat 1 (cycle 1) @ 20.0 °C/min"
    row.is_partial = True
    row.setpoint_reached = False
    assert "partial" in row.flags and "setpoint not reached" in row.flags


def test_set_manual_value_records_what_it_replaced():
    """A hand-typed value never silently erases the computed one: the number
    it replaced is evidence and goes into the note."""
    row = rs.manual_row(sample="x", q_C_per_min=10.0, value_C=620.8)
    row.value_source = "computed"
    row.method = "parallel"
    row.set_manual_value(618.0)
    assert row.value_C == 618.0
    assert row.value_source == "manual"
    assert row.method == "manual"
    assert "computed was 620.8 °C" in row.note


def test_fittable_needs_a_value_a_rate_and_the_include_flag():
    """One definition of "in the fit", so n_fitted and the fit's own n are
    always the same count."""
    row = rs.manual_row(sample="x", q_C_per_min=10.0, value_C=600.0)
    assert row.fittable is True
    row.include = False
    assert row.fittable is False
    row.include = True
    row.q_manual_C_per_min = 0.0
    assert row.fittable is False            # ln 0 is undefined
    row.q_manual_C_per_min = 10.0
    row.value_C = float("nan")
    assert row.fittable is False


def test_validate_raises_only_on_programming_bugs():
    """A duplicate uid or a bad enum corrupts every uid-keyed lookup and is a
    bug in the caller. Nothing PHYSICAL raises — that is check_series's job."""
    series = _clean_series(rates=(5.0, 10.0))
    series.validate()
    series.rows[1].uid = series.rows[0].uid
    with pytest.raises(ValueError, match="duplicate row uid"):
        series.validate()
    series.rows[1].uid = rs.RateRow.new_uid()
    series.rows[0].rate_source = "telepathy"
    with pytest.raises(ValueError, match="rate_source"):
        series.validate()
    # nothing PHYSICAL raises
    series.rows[0].rate_source = "auto"
    series.rows[0].value_C = -5000.0
    series.rows[0].q_manual_C_per_min = -1.0
    series.validate()


def test_index_of_and_remove_work_by_uid_not_by_position():
    """Removing the middle of three must not renumber the other two out from
    under the warnings that name them."""
    series = _clean_series(rates=(5.0, 10.0, 20.0))
    middle = series.rows[1]
    assert series.index_of(middle.uid) == 2
    assert series.remove(middle.uid) is True
    assert [r.uid for r in series.rows] == [series.rows[0].uid, series.rows[1].uid]
    assert series.get(middle.uid) is None
    assert series.remove("not a uid") is False
    assert series.index_of("not a uid") == 0


def test_relocate_touches_only_the_file_field():
    """A renamed share does not change a measurement."""
    series = _clean_series(rates=(5.0, 10.0))
    series.rows[0].file = r"\\Imr-research\ta\Data\SDT\a.txt"
    series.rows[1].file = r"D:\elsewhere\b.txt"
    before = [r.value_C for r in series.rows]
    n = rs.relocate(series, r"\\imr-research\ta", r"E:\archive")
    assert n == 1
    assert series.rows[0].file == r"E:\archive\Data\SDT\a.txt"
    assert series.rows[1].file == r"D:\elsewhere\b.txt"
    assert [r.value_C for r in series.rows] == before
    assert rs.relocate(series, "", "x") == 0


def test_rate_from_filename_delegates_and_never_raises():
    """Delegated to htxrd_science's template parser, including its
    Windows-basename rule, and None (never an exception) on a miss."""
    assert rs.rate_from_filename("GL_q010.txt", "GL_q???.txt") == 10.0
    assert rs.rate_from_filename(r"D:\a\b\GL_q030.txt", "GL_q???.txt") == 30.0
    assert rs.rate_from_filename("nope.txt", "GL_q???.txt") is None
    assert rs.rate_from_filename("", "GL_q???.txt") is None
    assert rs.rate_from_filename("GL_q010.txt", "") is None


def test_distinct_rates_cluster_within_the_tolerance():
    """9.99, 10.00 and 10.01 °C/min are one rate measured three times."""
    series = rs.RateSeries(quantity_kind="Tg")
    for q in (9.99, 10.0, 10.01, 20.0):
        series.add(rs.manual_row(sample=f"{q}", q_C_per_min=q, value_C=600.0))
    assert len(series.distinct_rates()) == 2
    assert series.rate_span_ratio() == pytest.approx(20.0 / 9.99, rel=1e-9)
    single = rs.RateSeries()
    single.add(rs.manual_row(sample="a", q_C_per_min=10.0, value_C=600.0))
    assert np.isnan(single.rate_span_ratio())


def test_few_rates_fires_at_exactly_two_distinct_rates():
    """Two rates: the slope exists but has no residual degrees of freedom, so
    R² and σ are not evidence about it."""
    series = rs.RateSeries(quantity_kind="Tg")
    for q, tg, key in ((10.0, 600.0, "a"), (10.0, 601.0, "b"),
                       (40.0, 620.0, "c"), (40.0, 621.0, "d")):
        series.add(rs.manual_row(sample=key, q_C_per_min=q, value_C=tg,
                                 specimen_key=f"{key}|1|d"))
    codes = {w.code for w in rs.check_series(series)}
    assert "few_rates" in codes
    assert "single_rate" not in codes
    assert "duplicate_rate" in codes
    assert "narrow_rate_span" not in codes        # 4x exactly is not "under 4x"
    assert rs.fit_series(series, "lasocka").ok is True


def test_partial_and_setpoint_and_outside_window_all_warn_and_still_fit():
    """Three provenance warnings that each describe a real defect of the real
    run, and none of which stops the arithmetic."""
    series = _clean_series(rates=(5.0, 10.0, 20.0))
    series.rows[0].is_partial = True
    series.rows[1].setpoint_reached = False
    series.rows[1].q_measured_C_per_min = 9.5
    series.rows[1].q_programmed_C_per_min = 10.0
    for row in series.rows:
        row.window_C = (350.0, 500.0)             # every value sits above it
    codes = {w.code for w in rs.check_series(series)}
    assert {"partial_segment", "setpoint_not_reached", "value_outside_window"} <= codes
    res = rs.fit_series(series, "lasocka")
    assert res.ok is True and np.isfinite(res.lasocka_B)


def test_a_file_with_no_heating_ramp_still_becomes_a_row(tmp_path):
    """The auto-detection rule's first line: 0 heating segments -> one row
    with no rate, and a critical warning. The row exists; the user may type
    a rate into it."""
    path = tmp_path / "isothermal_only.txt"
    n = 4000
    t = np.arange(n) * 0.05
    T = np.full(n, 800.0) + 0.01 * np.sin(t)
    lines = [_TA_HEADER.format(sample="hold", size="10.0000", date="2026-03-01",
                               rate=0.0, top=800.0)]
    lines += [f"{a:.6f}\t{b:.4f}\t10.00000\t0.000000" for a, b in zip(t, T)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")

    rows, warns = rs.rows_from_file(path)
    assert len(rows) == 1
    assert not np.isfinite(rows[0].q_C_per_min)
    assert rows[0].rate_source == "auto"
    codes = {w.code for w in warns}
    assert "no_heating_segment" in codes
    assert next(w for w in warns if w.code == "no_heating_segment").severity == "critical"
    # and the same condition is re-derived from the stored row state
    assert "no_heating_segment" in {w.code for w in rs.check_series(_series_of(rows))}
    # the user types a rate and the row becomes usable, with no reload
    rows[0].q_manual_C_per_min = 10.0
    rows[0].set_manual_value(612.0)
    assert rows[0].fittable is True
    assert rows[0].q_used_source == "manual"


def test_settings_snapshot_travels_into_the_rows_csv(tmp_path):
    """An exported series is self-describing: the file says which baseline
    mode and which window produced each number, months after the widgets have
    moved."""
    snapshot = {"tg_method": "parallel", "smooth_derivative": True,
                "y_col": "Temperature Difference (°C)", "dy_source": "(same)",
                "invert_y": False, "low_mode": "auto", "low_range": "350-420",
                "low_point_x": "", "high_mode": "manual", "high_range": "650-700",
                "high_point_x": "", "tp_signal": "", "tp_window_min_C": "",
                "tp_window_max_C": ""}
    series = _clean_series(rates=(5.0, 10.0, 20.0))
    for row in series.rows:
        row.settings = dict(snapshot)
    path = rs.write_series_csv(series, rs.check_series(series),
                               tmp_path / "s_rows.csv")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    for key in rs.SETTINGS_CSV_KEYS:
        assert key in frame.columns
    assert frame["tg_method"][0] == "parallel"
    assert frame["high_range"][0] == "650-700"

    loaded, _warns = rs.read_series_csv(path)
    assert loaded.rows[0].settings["tg_method"] == "parallel"
    assert loaded.rows[0].settings["low_range"] == "350-420"


def test_series_to_dict_is_json_safe():
    """NaN renders as None so `json.dumps` stays valid — `core.project_io`
    stores raw dicts and must never import `dta.*` to read them back."""
    series = _bm_like_series()
    payload = series.to_dict()
    text = json.dumps(payload)
    assert "NaN" not in text
    back = rs.RateSeries.from_dict(json.loads(text))
    assert len(back.rows) == 4
    assert not np.isfinite(back.rows[1].value_C)
    assert back.rows[3].cum_mass_loss_pct == pytest.approx(BM_LOSSES[3])


def test_seg_fields_reads_a_step_match_when_the_segment_has_no_alignment():
    """The adapter's fallback path, exercised without a file: a `StepMatch`
    supplies the programmed rate and the target gap when the Segment itself
    was never annotated."""
    seg = ds.Segment(
        index=1, kind=ds.KIND_HEAT, role=ds.ROLE_HEAT, cycle_index=1,
        row_start=0, row_stop=100, n_points=100, t_start=0.0, t_end=60.0,
        T_start=25.0, T_end=1225.0, T_min=25.0, T_max=1225.0,
        rate_mean=20.0, rate_median=19.9, rate_mode=19.95, rate_fit_r2=0.999,
        control_lost_below_C=float("nan"), monotone_fraction=1.0,
        n_reversals=0, n_duplicate_T=0, is_partial=False)
    bare = rs._seg_fields(seg)
    assert bare["q_measured_C_per_min"] == 19.95
    assert np.isnan(bare["q_programmed_C_per_min"])
    assert bare["setpoint_reached"] is None
    assert bare["row_start"] == -1                # no run: no DataFrame rows

    match = ds.StepMatch(segment_index=1, step_indices=(3,), programmed_kind="ramp",
                         programmed_rate=20.0, programmed_target=1300.0,
                         programmed_duration=63.75, measured_rate=19.95,
                         measured_duration=60.0, target_gap_C=75.0)
    with_match = rs._seg_fields(seg, match=match)
    assert with_match["q_programmed_C_per_min"] == 20.0
    assert with_match["setpoint_reached"] is False        # 75 °C short of target

    close = ds.StepMatch(segment_index=1, step_indices=(3,), programmed_kind="ramp",
                         programmed_rate=20.0, programmed_target=1300.0,
                         programmed_duration=63.75, measured_rate=19.95,
                         measured_duration=60.0, target_gap_C=2.0)
    assert rs._seg_fields(seg, match=close)["setpoint_reached"] is True


def test_the_module_pulls_in_no_heavy_dependency_at_import_time():
    """pandas, PySide6 and core.io_universal are all imported lazily, inside
    the functions that need them, so a headless science import of this module
    costs nothing and the layering stays one-way."""
    source = inspect.getsource(rs)
    head = source.split("# =====", 1)[0]
    assert "import pandas" not in head
    assert "PySide6" not in source
    assert "import core.io_universal" not in head
    assert "from dta import dta_kinetics" in head
    assert "from dta import dta_segments" in head


def test_evaluable_methods_gate_on_needs_and_on_T0():
    """`needs` decides which models a series can be fitted with, so the UI
    never re-decides which model suits which quantity."""
    tg_series = _clean_series()
    assert rs.evaluable_methods(tg_series) == ["lasocka", "moynihan", "kissinger_tg"]
    tp_series = _clean_series()
    for r in tp_series.rows:
        r.quantity_kind = "Tp"
    assert rs.evaluable_methods(tp_series) == ["kissinger", "fwo"]
    for r in tp_series.rows:
        r.T0_C = 25.0
    assert "augis_bennett" in rs.evaluable_methods(tp_series)


# ===========================================================================
# 12) Adversarial-review regressions (2026-09-26)
#
# One test per confirmed finding of the three review passes. Each one FAILS
# on the code as it stood before the fix; the failing observation is quoted
# in the docstring so a future reader can tell the test from its intent.
# ===========================================================================

def test_a_flattened_project_header_still_yields_the_programmed_rates(cycles_fixture):
    """The `.prism` round trip must not cost the temperature program.

    `core.project_io._json_safe` flattens the `TAHeader` subclass to a plain
    dict on save, so a reopened record has no `header.lines` and its single
    "OrgMethod" key is the collapsed first line, "1: Select gas 1".
    `core.io_universal` writes `meta["header_lines"]` for exactly this case.
    Before the fix nothing read it and a reopened 4-cycle run came back with
    `q_programmed_C_per_min = [nan, nan, nan, nan]` and `setpoint_reached =
    [None, None, None, None]`, which also silences `rate_mismatch` and
    `setpoint_not_reached` permanently.
    """
    header, df, _run, path = cycles_fixture
    flat = {str(k): str(v) for k, v in dict(header).items()}
    assert not hasattr(flat, "lines")
    assert flat.get("OrgMethod", "").startswith("1:")     # collapsed, as saved

    lines = [[k, v] for k, v in getattr(header, "lines", ())]
    assert sum(1 for k, _v in lines if k == "OrgMethod") == 19

    rec = {"df": df, "path": str(path), "id": "rec1", "title": "reopened",
           "meta": {"raw_header": flat, "header_lines": lines}}
    rows, _warns = rs.rows_from_record(rec, roles=("heat",))
    assert [r.q_programmed_C_per_min for r in rows] == [20.0, 20.0, 15.0, 10.0]
    # True, not False: the gap is judged after the soak completes the approach
    # (see the fixture-rows test and _extend_ramp_gaps_through_soak). What this
    # test is really pinning is that BOTH fields survive the flattening at all.
    assert all(r.setpoint_reached is True for r in rows)

    # and without the field it degrades, which is what makes the field
    # load-bearing rather than incidental
    bare = dict(rec, meta={"raw_header": flat})
    bare_rows, _w = rs.rows_from_record(bare, roles=("heat",))
    assert all(np.isnan(r.q_programmed_C_per_min) for r in bare_rows)


def test_replicate_cycles_at_one_rate_load_switched_on():
    """`_source_warnings` must call "the same rate" what the series calls it.

    Three replicate 20 C/min ramps measure 19.99 / 20.03 / 20.10 on real
    data. Binned at `round(abs(q), 2)` that is three rates, so every row was
    switched off and `ambiguous_file_rate` fired; clustered at
    `rate_tol_pct` (2 %) it is one rate measured three times, which is the
    third line of `_source_warnings`' own table and was unreachable before.
    """
    rows = [rs.RateRow(source_kind="segment", sample="replicates",
                       file="replicates.txt", quantity_kind="Tg",
                       q_measured_C_per_min=q, value_C=500.0 + i,
                       value_source="computed", segment_kind=ds.KIND_HEAT,
                       segment_index=i + 1, n_heating_segments_in_source=3)
            for i, q in enumerate((19.99, 20.03, 20.10))]
    warns = rs._source_warnings(rows, "replicates")

    assert [w.code for w in warns] == []
    assert all(r.include for r in rows)
    series = rs.RateSeries(rows=list(rows), quantity_kind="Tg")
    assert series.distinct_rates() == pytest.approx([19.99], abs=1e-6)
    assert len(series.fitted_rows()) == 3
    # the informative critical now reaches the user instead of too_few_points
    codes = {w.code for w in rs.check_series(series, verify_paths=False)}
    assert "single_rate" in codes
    assert "too_few_points" not in codes


def test_cluster_rates_is_the_one_rule_both_layers_use():
    """The builder and the series must agree on what "one rate" means."""
    assert rs.cluster_rates([20.0, 20.1, 19.95]) == pytest.approx([19.95])
    assert rs.cluster_rates([5.0, 10.0, 20.0]) == pytest.approx([5.0, 10.0, 20.0])
    assert rs.cluster_rates([-20.0, 20.0]) == pytest.approx([20.0])   # magnitudes
    assert rs.cluster_rates([0.0, np.nan, 10.0]) == pytest.approx([10.0])
    assert rs.cluster_rates([]) == []
    tight = rs.cluster_rates([20.0, 20.1, 19.95], rate_tol_pct=0.0)
    assert len(tight) == 3


def test_unit_suspect_tests_the_trend_not_every_pair():
    """The kelvin fingerprint must catch the bug it is named for.

    A doubly-converted row sits 273.15 above ITS OWN correct value, and in a
    Lasocka series every row has a different correct value - so the old
    pairwise "two rows differ by 273.15 +- 1" test missed it unless two rows
    happened to share a temperature, and it false-fired `critical` on any
    legitimate series spanning 273 C. Measured before the fix:
    [500, 506, 784.15, 517] did NOT fire; [200, 210, 473.15, 480] DID.
    """
    def _series(values, rates=(5.0, 10.0, 20.0, 40.0)):
        s = rs.RateSeries(quantity_kind="Tg")
        for q, v in zip(rates, values):
            s.add(rs.manual_row(sample=f"g{q:g}", q_C_per_min=q, value_C=v))
        return s

    # 511.0 C typed in as 784.15 K: the named bug
    bug = _series([500.0, 506.0, 511.0 + dk.K0, 517.0])
    suspect = next(w for w in rs.check_series(bug, verify_paths=False)
                   if w.code == "unit_suspect")
    assert suspect.severity == "critical"
    assert "273" in suspect.message
    assert rs.fit_series(bug, "lasocka").ok is True          # warned, not blocked

    # an ordinary series that happens to span 273 C: no alarm
    legit = _series([200.0, 210.0, 473.15, 480.0])
    assert "unit_suspect" not in {w.code for w in
                                  rs.check_series(legit, verify_paths=False)}

    # a high outlier that is NOT a unit error: no alarm from this rule
    outlier = _series([500.0, 506.0, 900.0, 517.0])
    assert "unit_suspect" not in {w.code for w in
                                  rs.check_series(outlier, verify_paths=False)}


def test_setpoint_warning_does_not_contradict_the_rate_it_prints():
    """`setpoint_not_reached` fires on every real heating ramp, so its prose
    has to be true of them.

    MERGED_DECISIONS M3 sets the flag at `target_gap_C <= 5.0`, and
    BM_FINDINGS section 3 measured why every heat trips it: a heat's setpoint
    arrives during the soak that follows (the ramps end 5.1-8.3 C short and
    the 1300 C hold tops out at 1298.6). Meanwhile measured and programmed
    agree to 0.06-0.26 %, which `rate_mismatch` (5 %) calls a match. The old
    text asserted "the programmed +10.00 C/min is not what the specimen
    saw" - false, and at `warn` next to a matching pair of numbers.
    """
    series = rs.RateSeries(quantity_kind="Tg")
    for q, v in ((10.0, 620.0), (20.0, 630.0)):
        row = rs.manual_row(sample=f"g{q:g}", q_C_per_min=q, value_C=v)
        row.q_measured_C_per_min = q - 0.012
        row.q_programmed_C_per_min = q
        row.segment_kind = ds.KIND_HEAT
        row.setpoint_reached = False
        series.add(row)
    warns = rs.check_series(series, verify_paths=False)
    codes = {w.code for w in warns}
    msg = next(w for w in warns if w.code == "setpoint_not_reached").message

    assert "rate_mismatch" not in codes            # they agree to 0.1 %
    assert "is not what the specimen saw" not in msg
    assert "ended short of the programmed target" in msg
    assert "hold that follows" in msg              # the physics, stated once
    assert msg.rstrip().endswith("Fitted anyway.")


def test_monotonic_column_is_false_on_a_strictly_descending_series():
    """The exported `monotonic` column must answer its own question.

    `_chk_monotonic` needs BOTH an ascending and a descending step to raise
    `non_monotonic`, so a series that falls all the way - the one ordering a
    rate effect cannot produce - raised nothing and the column, computed from
    the absence of that warning, read True.
    """
    series = rs.RateSeries(quantity_kind="Tg")
    for q, v in zip((2.0, 5.0, 10.0, 20.0), (520.0, 512.0, 504.0, 498.0)):
        series.add(rs.manual_row(sample=f"g{q:g}", q_C_per_min=q, value_C=v))
    warns = rs.check_series(series, verify_paths=False)
    assert "non_monotonic" not in {w.code for w in warns}     # literally monotone

    fit = rs.fit_series(series, "lasocka", warnings_in=warns)
    table = rs.fit_table(series, [fit], warns)
    assert table[0]["monotonic"] is False
    assert "negative_lasocka_b" in table[0]["warning_codes"]

    rising = _clean_series(rates=(5.0, 10.0, 20.0))
    rising_fit = rs.fit_series(rising, "lasocka")
    assert rs.fit_table(rising, [rising_fit])[0]["monotonic"] is True


def test_a_measurer_that_raises_typeerror_reports_its_own_fault():
    """`_call_measure` must not turn a fault INSIDE the measurer into a
    fault ABOUT the measurer's signature.

    Probing arity with `try: measure(x, y, window, row) except TypeError:`
    cannot tell "wrong number of arguments" from "TypeError raised in the
    body". Before the fix a four-argument measurer that failed internally
    reported "measure() missing 1 required positional argument: 'row'" as
    the row's verbatim error - the one field BM_FINDINGS section 8 requires
    to name the real cause - and a `*args` measurer ran its body twice.
    """
    calls = []

    def four_arg(x, y, window, row):
        calls.append("four")
        return 1.0 + None                        # TypeError, from the body

    row = rs.manual_row(sample="s")
    with pytest.raises(TypeError) as excinfo:
        rs._call_measure(four_arg, np.arange(5.0), np.arange(5.0), (0.0, 1.0), row)
    assert "NoneType" in str(excinfo.value)
    assert "positional argument" not in str(excinfo.value)
    assert calls == ["four"]

    calls.clear()

    def variadic(*args):
        calls.append("var")
        return 1.0 + None

    with pytest.raises(TypeError):
        rs._call_measure(variadic, np.arange(5.0), np.arange(5.0), (0.0, 1.0), row)
    assert calls == ["var"]                      # run ONCE, not twice

    # the three-argument contract still works, and still receives no row
    seen = {}

    def three_arg(x, y, window):
        seen["n"] = len(x)
        return 42.0

    assert rs._call_measure(three_arg, np.arange(5.0), np.arange(5.0),
                            (0.0, 1.0), row) == 42.0
    assert seen == {"n": 5}


def test_a_measurer_failure_reaches_the_row_verbatim(cycles_fixture):
    """The same fix, end to end: the row's `error` names the real cause."""
    header, df, run, path = cycles_fixture

    def broken(x, y, window, row):
        raise TypeError("unsupported operand type(s) for +: 'float' and 'NoneType'")

    rows, _w = rs.rows_from_segments(run, df, header=header, file=str(path),
                                     measure=broken)
    assert [r.value_source for r in rows] == ["failed"] * 4
    assert all("NoneType" in r.error for r in rows)
    assert all("positional argument" not in r.error for r in rows)


def test_rows_from_segments_reports_a_stale_run_instead_of_raising(cycles_fixture):
    """"Never raises on data" includes a run that belongs to another frame.

    The Qt layer caches one `SegmentedRun` per record id, so a re-import or a
    crop replaces the dataframe under it. The mass walk indexed the raw
    column with original-row positions and raised a bare IndexError
    ("index 1225 is out of bounds for axis 0 with size 1000").
    """
    header, df, run, path = cycles_fixture
    cropped = df.iloc[:1000].copy()
    rows, warns = rs.rows_from_segments(run, cropped, header=header, file=str(path))

    assert len(rows) == 4
    scope = next(w for w in warns if w.code == "segment_scope")
    assert scope.severity == "critical"
    assert scope.detail["n_rows_segmented"] == int(run.n_rows)
    assert scope.detail["n_rows_now"] == 1000
    # the matching dataframe stays silent
    _rows, ok_warns = rs.rows_from_segments(run, df, header=header, file=str(path))
    assert "segment_scope" not in {w.code for w in ok_warns}


def test_ambiguous_file_rate_counts_the_rows_it_names(cycles_fixture):
    """`_chk_sources` must not print a heats-only count beside all-row rates.

    With cooling rows in the series - which MERGED_DECISIONS M1 promotes to
    MUST - the message read "holds 4 heating ramps at 9.98, 9.99, 14.98,
    14.99, 19.95, 19.96, 19.97 C/min": a heats-only count next to eight
    ramps' rates, three of which the furnace never ran as a heating rate.
    """
    header, df, run, path = cycles_fixture
    # source_kind="file": a separate file that turns out to hold several rates
    # is the case this code fires on. A deliberate segment pick raises
    # `multi_rate_source` instead (see
    # test_segment_rows_arrive_switched_on_so_the_series_can_be_fitted), and the
    # count/word/rates consistency asserted below is required of BOTH messages.
    rows, _w = rs.rows_from_segments(run, df, header=header, file=str(path),
                                     source_kind="file",
                                     roles=(ds.ROLE_HEAT, ds.ROLE_COOL))
    for i, r in enumerate(rows):
        r.value_C = 500.0 + i
        r.value_source = "manual"
        r.include = True
    series = rs.RateSeries(rows=list(rows), quantity_kind="Tg")
    warn = next(w for w in rs.check_series(series, verify_paths=False)
                if w.code == "ambiguous_file_rate")

    assert len(rows) == 8
    assert "holds 8 ramps" in warn.message          # not "4 heating ramps"
    assert "heating" not in warn.message            # the kinds are mixed
    assert len(warn.detail["rates"]) == 3           # clustered at 2 %
    assert warn.detail["n_rows"] == 8

    # The same consistency is required of the segment-pick message.
    seg_rows, _w2 = rs.rows_from_segments(run, df, header=header, file=str(path),
                                          roles=(ds.ROLE_HEAT, ds.ROLE_COOL))
    for i, r in enumerate(seg_rows):
        r.value_C = 500.0 + i
        r.value_source = "manual"
    seg_series = rs.RateSeries(rows=list(seg_rows), quantity_kind="Tg")
    seg_warn = next(w for w in rs.check_series(seg_series, verify_paths=False)
                    if w.code == "multi_rate_source")
    assert "8 ramps" in seg_warn.message
    assert "heating" not in seg_warn.message
    assert seg_warn.detail["n_rows"] == 8
    assert len(seg_warn.detail["rates"]) == 3


def test_auto_excluded_rows_are_not_blamed_on_the_user(cycles_fixture):
    """`excluded_rows` said "switched off by hand" about rows the engine
    switched off itself.

    `_source_warnings` sets `include = False` whenever a separate FILE yielded
    several distinct rates (SPEC_rate_series section 7.5), so those rows arrive
    off before the user has touched anything.
    """
    header, df, run, path = cycles_fixture
    rows, _w = rs.rows_from_segments(run, df, header=header, file=str(path),
                                     source_kind="file")
    assert all(r.include is False for r in rows)      # nobody clicked anything

    series = rs.RateSeries(rows=list(rows), quantity_kind="Tg")
    warns = {w.code: w.message for w in rs.check_series(series, verify_paths=False)}
    assert "by hand" not in warns["excluded_rows"]
    assert "ambiguous_file_rate" in warns["excluded_rows"]
    assert "switched off by you" not in warns["rows_not_fitted"]


def test_segment_rows_arrive_switched_on_so_the_series_can_be_fitted(cycles_fixture):
    """Picking the segments of a cyclic run must yield a FITTABLE series.

    `_source_warnings` switched every row off whenever one source gave several
    distinct rates. Applied to a deliberate segment pick that was exactly
    backwards: this program was written to measure one specimen at 20, 20, 15 and
    10 C/min *so that the rates could be compared*, and the result was four good
    rows with `n_fitted == 0` and `too_few_points` on screen -- the Lasocka and
    Arrhenius feature producing nothing on the dataset it exists for, which is
    the opposite of "still make this possible to process such data".

    The rows now arrive on, and the genuine objection -- that successive cycles
    of one specimen confound heating rate with thermal history -- is raised as
    `single_specimen_cycles` and `thermal_history`, which describe the problem
    without suppressing the number.
    """
    header, df, run, path = cycles_fixture
    rows, warns = rs.rows_from_segments(run, df, header=header, file=str(path))
    assert len(rows) == 4
    assert all(r.include is True for r in rows)
    codes = {w.code for w in warns}
    assert "multi_rate_source" in codes
    assert "ambiguous_file_rate" not in codes

    for i, r in enumerate(rows):                      # stand in for a measurement
        r.value_C = 600.0 + 3.0 * i
        r.value_source = "manual"
    series = rs.RateSeries(rows=list(rows), quantity_kind="Tg")
    fit = rs.fit_series(series, "lasocka")
    assert fit.ok is True
    assert fit.n == 4
    assert np.isfinite(fit.lasocka_B)

    warned = {w.code for w in fit.warnings}
    assert "too_few_points" not in warned
    # ... and the real confound is still stated, loudly.
    assert {"single_specimen_cycles", "thermal_history"} <= warned


def test_thermal_history_pairs_each_count_with_its_own_state():
    """Two independently sorted lists printed side by side read as pairs.

    Before the fix: "had seen 0, 1, 2, 3, 4 prior heating ramps
    (after-cycle-1, after-cycle-2, after-cycle-3, after-cycle-4, virgin)",
    i.e. a virgin specimen credited with four prior ramps - the exact
    confound the warning exists to describe.
    """
    series = rs.RateSeries(quantity_kind="Tg")
    for i, (q, v) in enumerate(zip((5.0, 10.0, 20.0, 40.0, 80.0),
                                   (500.0, 505.0, 510.0, 515.0, 520.0))):
        row = rs.manual_row(sample="one specimen", q_C_per_min=q, value_C=v)
        row.prior_heats = i
        row.state = "virgin" if i == 0 else f"after-cycle-{i}"
        series.add(row)
    msg = next(w for w in rs.check_series(series, verify_paths=False)
               if w.code == "thermal_history").message

    assert "0 (virgin)" in msg
    assert "4 (after-cycle-4)" in msg
    assert msg.index("0 (virgin)") < msg.index("4 (after-cycle-4)")


def test_roles_hold_means_every_kind_of_hold(cycles_fixture):
    """`ROLE_HOLD` is exported in `SEGMENT_ROLES` but `assign_roles` emits it
    only for a hold no ramp preceded, which never happens on a real run.

    Asking for it returned a `no_heating_segment` placeholder describing a
    file that contains four holds.
    """
    header, df, run, path = cycles_fixture
    rows, warns = rs.rows_from_segments(run, df, header=header, file=str(path),
                                        roles=(ds.ROLE_HOLD,))
    assert [r.segment_index for r in rows] == [2, 5, 8, 11]
    assert "no_heating_segment" not in {w.code for w in warns}

    hot, _w = rs.rows_from_segments(run, df, header=header, file=str(path),
                                    roles=(ds.ROLE_HOLD_HOT,))
    assert [r.segment_index for r in hot] == [r.segment_index for r in rows]
