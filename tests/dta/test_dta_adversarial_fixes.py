"""Regression guards for the defects three adversarial review passes found.

Every test here FAILS on the code as it stood before the triage pass and
passes after it. Each one names the finding it guards and, where the old
behaviour was a measured wrong number, carries that number in its docstring so
a future reader can tell the guard from a preference.

The passes and what survived triage:

  LENS 1 F1   the rate path stored a row SET as a bounding INTERVAL
  LENS 2 #1   "Load series rows from CSV" destroyed a measured series
  LENS 2 #2   `_rate_loaded_warnings` was written twice and read never
  LENS 2 #3   adding the same segments twice halved the published sigma
  LENS 2 #4   the figure's accounting counts did not partition the rows
  LENS 2 #5   the Calculs tab modalled on a data condition and went stale
  LENS 2 #7   an add-time sentence claimed "all switched on" while three were off
  LENS 3 H1   a coarsely rounded temperature column could yield no Tg at all
  LENS 3 H2   a ramp slower than 0.20 C/min was classified as a hold
  LENS 3 H3   a `Time (sec)` clock was read as minutes, 60x
  LENS 3 M1   an unrelated CSV minted phantom rows
  LENS 3 M2   a non-mapping header raised out of a "never raises" function
  LENS 3 L2   removed rows never released their cached DataFrame
  LENS 3 L3   a non-finite `extras` value wrote a bare NaN into the project

Rejected by a majority of the verification passes and NOT guarded here, with
the reproduction that settled each one in the triage report: LENS 2 #6
(switching add-files rows off is documented intent), LENS 2 #8 (`check_series`
cannot be made to raise without replacing it), LENS 3 L4 (the empty-workspace
dialog is unchanged pre-rework code behind a disabled button), LENS 3 L5.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from PySide6.QtWidgets import QMessageBox

from dta import dta_rate_series as drs
from dta import dta_segments as ds
from dta.dta_program import program_from_header
from dta.qt_dta import (DtaWorkspace, segment_scope_key, _is_temperature_col,
                        _unit_from_col)
from core.qt_shell import _load_spectrum_from_path


# ===========================================================================
# helpers
# ===========================================================================
def _no_dialogs(monkeypatch):
    """Every modal QMessageBox becomes an immediate failure.

    MERGED_DECISIONS M0.1: a warning must never be a dialog, because a dialog
    is the surface a user learns to dismiss without reading.
    """
    def boom(*_a, **_k):
        raise AssertionError("a modal dialog was raised")

    for name in ("critical", "warning", "information", "question", "about"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(boom))


def _workspace(qtbot, path=None):
    records = []
    if path is not None:
        spectrum = _load_spectrum_from_path(str(path))
        records = [{"title": spectrum.title, "path": spectrum.path,
                    "df": spectrum.df, "meta": spectrum.meta}]
    widget = DtaWorkspace(records=records)
    qtbot.addWidget(widget)
    return widget


def _select_scope(widget, key):
    for i in range(widget.segment_combo.count()):
        if widget.segment_combo.itemData(i) == key:
            widget.segment_combo.setCurrentIndex(i)
            return True
    raise AssertionError(f"{key!r} is not offered")


def _write_ta(path, t_min, T_C, *, y=None, sample="SYN", size_mg=10.0,
              org_method=(), time_label="Time (min)",
              temp_label="Temperature (°C)"):
    """A TA SDT text export the real parser accepts.

    Mirrors `EXAMPLES/DTA_cycles_example.txt` line for line: tab-separated
    `Key\\tValue` header, `Sig<n>` column names, `StartOfData`, UTF-8 with a
    BOM, CRLF. Tests here need files on DISK rather than frames assigned onto
    the widget, because two of the three verification passes showed that
    assigning `widget.df` reaches states the loader cannot produce.
    """
    t = np.asarray(t_min, dtype=float)
    T = np.asarray(T_C, dtype=float)
    if y is None:
        y = 1.0 + 0.5 * np.tanh((T - 600.0) / 5.0)
    y = np.asarray(y, dtype=float)
    head = [
        "CLOSED", "Version\t2.0", "Language\tEnglish", "Run\t1", "RunSerial\t1",
        "Instrument\tSDT Q600 V20.9 Build 20", "Module\tDSC-TGA Standard",
        "InstSerial\t0600-", "Operator\tSS", f"File\t{sample}.001",
        f"Sample\t{sample}", f"Size\t{size_mg:.4f}\tmg", "ProcName\tRamp",
        "Method\tRamp", "Comment\tsynthetic", "Exotherm\tUp", "Kcell\t0.864000",
        "Controls\tGas 1  Event OFF  Sampling 0.5 sec/pt", "AutoAnalysis\tOff",
        "MacroFile", "Nsig\t4", f"Sig1\t{time_label}", f"Sig2\t{temp_label}",
        "Sig3\tWeight (mg)", "Sig4\tHeat Flow (mW)", "Date\t2026-01-01",
        "Time\t11:05:29",
    ]
    for i, step in enumerate(org_method, start=1):
        head.append(f"OrgMethod\t{i}: {step}")
    head.append("StartOfData")
    body = [f"{a:.7g}\t{b:.7g}\t{size_mg:.7g}\t{c:.7g}"
            for a, b, c in zip(t, T, y)]
    Path(path).write_bytes(
        b"\xef\xbb\xbf" + ("\r\n".join(head + body) + "\r\n").encode("utf-8"))
    return Path(path)


def _ramp(q, *, T0=300.0, T1=700.0, dt_min=0.5 / 60.0, decimals=None):
    n = int(round(abs(T1 - T0) / abs(q) / dt_min)) + 1
    t = np.arange(n) * dt_min
    T = T0 + q * t
    if decimals is not None:
        T = np.round(T, decimals)
    return t, T


def _stall_the_clock(src: Path, dst: Path, *, n_rows: int, at: int = 120):
    """Copy a TA export with `n_rows` timestamps replaying the previous one.

    A repeated timestamp is the file shape `clean_time_temperature`'s own
    docstring names ("a repeated timestamp makes every derivative infinite"):
    the rows keep a real temperature and a real signal, and only the clock
    rule rejects them. That is what makes `keep_index` stop being `arange`
    without any column going non-finite — and so what the Qt layer cannot see
    by comparing row counts.
    """
    text = src.read_bytes().decode("utf-8-sig")
    lines = text.splitlines()
    start = lines.index("StartOfData") + 1
    head, data = lines[:start], [ln for ln in lines[start:] if ln.strip()]
    rows = [ln.split("\t") for ln in data]
    for k in range(n_rows):
        rows[at + k] = list(rows[at + k])
        rows[at + k][0] = rows[at + k - 1][0]
    out = "\r\n".join(head + ["\t".join(r) for r in rows]) + "\r\n"
    dst.write_bytes(b"\xef\xbb\xbf" + out.encode("utf-8"))
    return dst


# ===========================================================================
# LENS 1 F1 — the rate path must measure the rows the cleaning kept
# ===========================================================================
def test_rate_arrays_come_from_the_segment_row_set_not_its_bounding_interval(
        qtbot, dta_cycles_path, tmp_path):
    """MERGED_DECISIONS M4: the rate path yields `df.iloc[run.row_indices(seg)]`.

    `RateRow.row_start` / `row_stop` store `row_indices(seg)[0]` and `[-1] + 1`
    (M3) — a bounding INTERVAL over a row SET. `_rate_segment_for` rebuilt
    `np.arange(row_start, row_stop)` from the pair, which is the same set only
    while the cleaning dropped nothing. With 40 stalled timestamps inside
    heating segment 1 of the committed fixture the authoritative path prepares
    1341 points and the interval 1381 — 40 of them rows the cleaning had
    rejected — and the Tg that came out was the PRISTINE file's answer at every
    count: measured 622.7934813374297 from the rate tab against 619.9335129062252
    from the Tg tab, with no warning code naming it anywhere.
    """
    path = _stall_the_clock(dta_cycles_path, tmp_path / "stalled.txt", n_rows=10)
    widget = _workspace(qtbot, path)
    run = widget.run
    assert run is not None and run.n_dropped_rows == 10
    seg = run.segments[0]
    rows_truth = run.row_indices(seg)
    assert not bool(np.all(np.diff(rows_truth) == 1)), "the set must be gappy"

    widget.rate_add_segments()
    row = widget.rate_series.rows[0]
    assert (row.row_start, row.row_stop) == (int(rows_truth[0]),
                                             int(rows_truth[-1]) + 1)

    y_col = widget.y_combo.currentText()
    y_all = pd.to_numeric(widget.df[y_col], errors="coerce").to_numpy(dtype=float)
    truth = run.xy(seg, y_all)
    x, y, _dy, _col = widget._rate_arrays_for(row, signal=y_col)

    assert x.size == truth.x.size == rows_truth.size == 336
    assert np.array_equal(x, truth.x)
    assert np.array_equal(y, truth.y)
    # As a SET: this file's first two rows are 19.03072 and 19.02074 °C, so the
    # ascending-temperature rule swaps them. The claim is "these rows and no
    # others", not "in acquisition order".
    assert sorted(truth.source_index) == sorted(rows_truth)


def test_rate_value_and_tg_tab_agree_after_the_cleaning_drops_rows(
        qtbot, dta_cycles_path, tmp_path, monkeypatch):
    """Two tabs of one workspace, one segment, one window: one number.

    Before: Tg tab 619.9335129062252, Rate series 622.7934813374297, and the
    rate tab's is the one that enters the Lasocka fit and `rows.csv`.
    """
    _no_dialogs(monkeypatch)
    path = _stall_the_clock(dta_cycles_path, tmp_path / "stalled2.txt", n_rows=10)
    widget = _workspace(qtbot, path)
    _select_scope(widget, segment_scope_key(1))
    widget.xmin_edit.setText("350")
    widget.xmax_edit.setText("700")
    widget._compute()
    tab = widget._current_result().res_parallel.tg

    widget.rate_add_segments()
    widget.rate_compute_all()
    assert widget.rate_series.rows[0].value_C == pytest.approx(tab, abs=1e-9)
    # Measured: the Tg tab moved to 623.1852 while the rate row stayed on
    # 622.8145044552292 — the PRISTINE fixture's answer, and the number the
    # pinned `CYCLE_TG_C[0]` in test_qt_dta_rate_compute.py was measured at.
    assert tab != pytest.approx(622.8145044552292, abs=1e-6)


def test_a_reversed_row_span_is_recovered_instead_of_becoming_the_whole_record(
        qtbot):
    """LENS 1 F1 reproduction C — the fallback that was the bug it prevents.

    When an appended restart makes the clock non-monotone,
    `clean_time_temperature` stably sorts and `keep_index` becomes a
    PERMUTATION, so a segment's first original row can sit after its last.
    `row_stop <= row_start` then hit `_rate_segment_for`'s "no span recorded"
    guard and the row was measured over the WHOLE frame, stamped with the note
    "whole record; this file was not segmented" — which was false, and is the
    global-scope curve the segmentation rework exists to prevent.

    `_rate_segment_rows` looks `row_stop - 1` UP instead of comparing it, so a
    reversed span resolves itself.
    """
    widget = _workspace(qtbot)
    t = np.array([0.0, 1.0, 2.0, 0.5, 1.5, 2.5])
    T = np.array([10.0, 20.0, 30.0, 15.0, 25.0, 35.0])
    _tc, _Tc, keep = ds.clean_time_temperature(t, T)
    assert list(keep) == [0, 3, 1, 4, 2, 5]           # a permutation, 0 dropped

    row = drs.RateRow(source_kind="segment", segment_index=2,
                      segment_kind=ds.KIND_HEAT, role_in_cycle=ds.ROLE_HEAT,
                      cycle_index=1)
    row.row_start = int(keep[1])          # 3
    row.row_stop = int(keep[2]) + 1       # 2  -> reversed in POSITION
    assert row.row_stop < row.row_start

    seg, got_keep, problem = widget._rate_segment_rows(row, time_min=t, temp_C=T)
    assert problem == ""
    assert got_keep is not None
    assert seg.index == 2
    assert (seg.row_start, seg.row_stop) == (1, 3)
    assert seg.n_points == 2
    assert "not segmented" not in seg.note
    assert list(np.asarray(got_keep)[seg.slice]) == [3, 1]


def test_an_unreproducible_row_span_is_reported_on_the_row(qtbot, dta_cycles_path,
                                                           tmp_path, monkeypatch):
    """A span that is not in this file any more is recorded, never refused.

    The fix re-derives the row set from the frame, so a frame that has changed
    under the row can leave the stored span unfindable. That is reported in
    `extras` as `row_span_not_reproducible` and the value is still produced —
    MERGED_DECISIONS M0.1 forbids refusing it.
    """
    _no_dialogs(monkeypatch)
    # The stall sits ABOVE the 350-700 °C window (row 300 of this segment is
    # ~1100 °C), so the fallback span still covers the window and a value can
    # be produced — which is the point: the span is unreproducible, the value
    # is not refused.
    path = _stall_the_clock(dta_cycles_path, tmp_path / "stalled3.txt",
                            n_rows=10, at=300)
    widget = _workspace(qtbot, path)
    widget.rate_add_segments()
    row = widget.rate_series.rows[0]
    row.row_stop = 306             # row 305 is one the stall dropped
    widget.rate_compute_all()
    assert row.extras.get("row_span_not_reproducible") == 1.0
    assert np.isfinite(row.value_C)
    assert row.value_source == "computed"


# ===========================================================================
# LENS 3 H1 — a coarse abscissa must not stop a file yielding a value
# ===========================================================================
def test_a_coarse_temperature_column_still_yields_a_tg(qtbot, tmp_path,
                                                       monkeypatch):
    """A NOISELESS 1 C/min ramp written to two decimals is 16.7 % duplicate.

    `prepare_segment_xy`'s 1 % duplicate guard refused it, so the Tg tab
    produced nothing, a modal dialog said "the wrong abscissa or the wrong
    scope" when both were right, and the file could contribute no point to a
    Lasocka or Kissinger series at all. The duplicates had ALREADY been
    averaged one line above the raise; the planted Tg is 600.0 C and the
    averaged curve returns it to within 0.03 C.
    """
    _no_dialogs(monkeypatch)
    t, T = _ramp(1.0, decimals=2)
    path = _write_ta(tmp_path / "coarse.txt", t, T,
                     org_method=["Ramp 1.00 °C/min to 700.00 °C",
                                 "End of method"], sample="COARSE")
    widget = _workspace(qtbot, path)
    assert [s.kind for s in widget.run.segments] == [ds.KIND_HEAT]
    assert widget.run.segments[0].n_duplicate_T > 0.01 * len(widget.df)

    widget.xmin_edit.setText("500")
    widget.xmax_edit.setText("650")
    widget._compute()
    res = widget._current_result()
    assert res.res_parallel is not None
    assert res.res_parallel.tg == pytest.approx(600.0, abs=0.5)
    assert any("shared an x value" in w for w in res.warnings)


def test_coarse_abscissa_travels_onto_the_row_the_banner_and_the_csv(
        qtbot, tmp_path, monkeypatch):
    """The refusal became a warning, so the warning has to be impossible to miss."""
    _no_dialogs(monkeypatch)
    t, T = _ramp(1.0, decimals=2)
    path = _write_ta(tmp_path / "coarse2.txt", t, T,
                     org_method=["Ramp 1.00 °C/min to 700.00 °C",
                                 "End of method"], sample="COARSE")
    widget = _workspace(qtbot, path)
    widget.xmin_edit.setText("500")
    widget.xmax_edit.setText("650")
    widget.rate_add_segments()
    widget.rate_compute_all()
    row = widget.rate_series.rows[0]
    assert np.isfinite(row.value_C)
    assert row.extras["duplicate_fraction"] == pytest.approx(1.0 / 6.0, abs=0.01)
    assert "coarse_abscissa" in {w.code for w in widget.rate_warnings}

    out = tmp_path / "rows.csv"
    widget.rate_export_rows_csv(str(out))
    assert "coarse_abscissa" in out.read_text(encoding="utf-8-sig")


def test_prepare_segment_xy_reports_a_duplicate_flood_when_asked_to(
        dta_synthetic_cycles):
    """`max_duplicate_frac=None` reports; the float default still refuses.

    The library guardrail is kept for a caller that wants it — the interactive
    paths pass None, which is what removes the hard block.
    """
    t, T = dta_synthetic_cycles
    run = ds.segment_run(t, T)
    seg = run.segments[0]
    y = np.sin(t)
    T_bad = T.copy()
    rows = run.row_indices(seg)
    T_bad[rows[0:200:2]] = T_bad[rows[1:200:2]]

    with pytest.raises(ds.SegmentScopeError):
        ds.prepare_segment_xy(t, T_bad, y, segment=seg, keep_index=run.keep_index)

    prepared = ds.prepare_segment_xy(t, T_bad, y, segment=seg,
                                     keep_index=run.keep_index,
                                     max_duplicate_frac=None)
    assert prepared.n_duplicates_collapsed > 0
    assert 0.0 < prepared.duplicate_fraction < 1.0
    assert bool(np.all(np.diff(prepared.x) > 0))


# ===========================================================================
# LENS 3 H2 — a programmed rate is not noise, so the floor must not override it
# ===========================================================================
@pytest.mark.parametrize("q", [0.19, 0.10, 0.05])
def test_a_programmed_slow_ramp_is_a_ramp(q):
    """`_resolve_threshold`'s floor applied to the programmed path too.

    A 0.10 C/min ramp whose header reads `Ramp 0.10 C/min to 75.00 C` came out
    `isothermal`: no rate, the time axis forced, no Tg on any abscissa, a
    banner asserting the file "contains no ramp of the kind asked for" while
    `program.ramp_rates == (0.1,)` sat in the same object, and none of the
    three knobs `apply_detection_settings` exposes could reach it. The
    boundary was exactly 0.20 C/min, which is the floor's value. 0.1-1 K/min is
    the standard scan range for enthalpy-relaxation work, i.e. the domain the
    Lasocka feature exists for (MERGED_DECISIONS M0.1).
    """
    t, T = _ramp(q, T0=50.0, T1=75.0, decimals=5)
    program = program_from_header(
        {}, [["OrgMethod", f"1: Ramp {q:.2f} °C/min to 75.00 °C"],
             ["OrgMethod", "2: End of method"]], name="slow")
    assert program.ramp_rates == (pytest.approx(q),)

    run = ds.segment_run(t, T, program=program)
    assert [s.kind for s in run.segments] == [ds.KIND_HEAT]
    assert run.segments[0].rate_mode == pytest.approx(q, rel=1e-3)


def test_the_noise_floor_still_guards_the_p95_path(dta_synthetic_cycles):
    """The floor exists for "a file that is one long hold", and keeps that job.

    Its stated reason is that p95(|dT/dt|) on such a file is noise and a
    fraction of noise is still noise. Only the PROGRAMMED path was exempted.
    """
    t, _T = dta_synthetic_cycles
    rng = np.random.default_rng(3)
    T = 500.0 + rng.normal(0.0, 0.05, t.size)
    run = ds.segment_run(t, T)
    assert all(s.kind != ds.KIND_HEAT for s in run.segments)


# ===========================================================================
# LENS 3 H3 — the clock's unit is written on the column
# ===========================================================================
def test_a_clock_column_is_never_read_as_a_temperature():
    """The root of H3, fixed where it starts instead of at each call site.

    `_unit_from_col` ends with `"c)" in cl -> "degC"`, and `Time (sec)` contains
    `c)`. So a seconds clock answered "degC". Two consequences, both measured on
    the code as it stood after the first H3 fix:

      * with the display switch on K, the clock axis had 273.15 ADDED to it;
      * `_is_temperature_col("Time (sec)")` was True, so selecting a hold
        segment refused the clock -- the one abscissa that IS valid on a hold,
        since a hold has no temperature scan to put on an axis.

    The call sites ask `_looks_like_time` first where they must scale a clock,
    and that is why the end-to-end rate came out right. But correctness must not
    rest on every caller remembering the order: a future call site that asked
    this function alone would reproduce the bug. Hence the guard lives in the
    function, once, and this test pins it there.
    """
    for col in ("Time (sec)", "Time (s)", "Time (min)", "time [h]", "Time"):
        assert _unit_from_col(col) == "", col
        assert _is_temperature_col(col) is False, col

    # ... and no temperature column was sacrificed to get there.
    assert _unit_from_col("Temperature (°C)") == "°C"
    assert _unit_from_col("Temperature (K)") == "K"
    assert _unit_from_col("Temp (C)") == "°C"
    assert _unit_from_col("Temperature Difference (°C)") == "°C"
    for col in ("Temperature (°C)", "Temperature (K)", "Temp (C)"):
        assert _is_temperature_col(col) is True, col
    assert _unit_from_col("Heat Flow (mW)") == ""


def test_clock_scale_reads_the_unit_off_the_column_label():
    assert ds.clock_scale_to_min("Time (min)") == (1.0, "min")
    assert ds.clock_scale_to_min("Time (sec)") == (pytest.approx(1 / 60.0), "s")
    assert ds.clock_scale_to_min("Time (s)") == (pytest.approx(1 / 60.0), "s")
    assert ds.clock_scale_to_min("time [h]") == (60.0, "h")
    assert ds.clock_scale_to_min("Time") == (1.0, "")
    assert ds.clock_scale_to_min(None) == (1.0, "")


def test_a_seconds_clock_gives_the_right_rate_and_says_so(tmp_path):
    """`Time (sec)` read as minutes reported 0.3333 C/min for a 20 C/min ramp.

    Zero warnings fired. On a two-file series of one specimen differing only in
    the clock column that came out as a Lasocka B wrong by 4.9x AND of the
    wrong sign, under five warnings none of which named the clock — while the
    column was literally labelled `Time (sec)`.
    """
    t, T = _ramp(20.0, dt_min=1.0 / 60.0, decimals=5)
    path = _write_ta(tmp_path / "sec.txt", t * 60.0, T, sample="GL",
                     time_label="Time (sec)")
    rows, warns = drs.rows_from_file(path)
    assert len(rows) == 1
    assert rows[0].q_C_per_min == pytest.approx(20.0, rel=1e-3)
    clock = [w for w in warns if w.code == "clock_unit"]
    assert len(clock) == 1
    assert clock[0].row_uids == (rows[0].uid,)
    assert "Time (sec)" in clock[0].message


def test_a_seconds_clock_is_displayed_and_windowed_in_minutes(qtbot, tmp_path,
                                                              monkeypatch):
    """The segmented clock and the displayed clock have to be the same clock.

    `Segment.t_start` / `t_end` are minutes whatever the column's label says,
    and `_apply_default_window` writes them straight into the Tg and Calculs
    window fields on a time abscissa. Converting the clock for the SEGMENTER
    alone would therefore have put a minutes window on a seconds axis — 60x
    out — on any multi-segment `Time (sec)` export. Both sides convert.
    """
    _no_dialogs(monkeypatch)
    # heat 300->600 at 20, hold 10 min, cool 600->300 at 20: three segments,
    # so `_apply_default_window` is not a no-op.
    t_up, T_up = _ramp(20.0, T0=300.0, T1=600.0, dt_min=0.01)
    t_hold = t_up[-1] + np.arange(1, 1001) * 0.01
    T_hold = np.full(t_hold.size, 600.0)
    t_dn, T_dn = _ramp(-20.0, T0=600.0, T1=300.0, dt_min=0.01)
    t = np.concatenate([t_up, t_hold, t_hold[-1] + t_dn[1:] + 0.01])
    T = np.concatenate([T_up, T_hold, T_dn[1:]])
    path = _write_ta(tmp_path / "cyc_sec.txt", t * 60.0, T, sample="SEC",
                     time_label="Time (sec)")
    widget = _workspace(qtbot, path)
    assert [s.kind for s in widget.run.segments] == [
        ds.KIND_HEAT, ds.KIND_ISOTHERMAL, ds.KIND_COOL]
    hold = widget.run.segments[1]
    assert hold.t_end - hold.t_start == pytest.approx(10.0, abs=0.2)

    _select_scope(widget, segment_scope_key(2))
    assert widget.x_combo.currentText() == "Time (sec)"
    lo = float(widget.xmin_edit.text())
    hi = float(widget.xmax_edit.text())
    assert hold.t_start <= lo < hi <= hold.t_end, (lo, hi)

    x, _y, _dy, x_col, _yc, _dc = widget._get_xy()
    assert x_col == "Time (sec)"
    # scoped to the hold, so this is the hold's own end IN MINUTES; read as
    # seconds it would be 1494.6.
    assert float(x.max()) == pytest.approx(hold.t_end, abs=0.02)
    assert float(x.min()) == pytest.approx(hold.t_start, abs=0.02)
    assert widget._display_x_label(x_col) == "Time (min)"
    assert widget._calc_x_unit() == "min"


def test_a_minutes_clock_raises_no_clock_warning(tmp_path):
    t, T = _ramp(20.0, decimals=5)
    path = _write_ta(tmp_path / "min.txt", t, T, sample="GL")
    rows, warns = drs.rows_from_file(path)
    assert rows[0].q_C_per_min == pytest.approx(20.0, rel=1e-3)
    assert "clock_unit" not in {w.code for w in warns}


# ===========================================================================
# LENS 2 #1 / LENS 3 H4 / M1 — the CSV load must not delete a measured series
# ===========================================================================
def test_loading_a_non_csv_leaves_the_measured_series_alone(qtbot, dta_cycles_path,
                                                            tmp_path, monkeypatch):
    """`pd.read_csv` on a .txt export does not raise: it returns 0 rows.

    `rate_load_rows_csv` assigned the empty series over the live one before
    anything looked at it, and the status line then read "values are as saved".
    Four measurements, a hand-typed value and a hand-written note gone, with no
    dialog and no undo — and the file most likely to be mis-picked is the .txt
    export in the same folder as the export the user just wrote.
    """
    _no_dialogs(monkeypatch)
    widget = _workspace(qtbot, dta_cycles_path)
    widget.rate_add_segments()
    widget.rate_compute_all()
    widget.rate_set_value(widget.rate_series.rows[1].uid, 601.25)
    widget.rate_set_note(widget.rate_series.rows[1].uid, "measured by hand")
    before = [(r.value_C, r.value_source, r.note) for r in widget.rate_series.rows]
    assert len(before) == 4

    junk = tmp_path / "not_a_series.bin"
    junk.write_bytes(bytes(range(256)) * 3)
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")

    for pick in (dta_cycles_path, junk, empty):
        n, warns = widget.rate_load_rows_csv(str(pick))
        assert n == 0
        assert [(r.value_C, r.value_source, r.note)
                for r in widget.rate_series.rows] == before
        assert "as saved" not in widget.rate_io_note.text()
        assert "Nothing was loaded" in widget.rate_io_note.text()
        assert max(w.severity for w in warns) == "critical"


def test_an_unrelated_csv_is_refused_by_the_schema_floor(tmp_path):
    """`item,qty / milk,2 / eggs,12` minted two included rows at INFO severity.

    "Unknown columns are ignored, so a file written by a newer PRISM opens" is
    the right rule; it needed a floor.
    """
    shopping = tmp_path / "shopping_list.csv"
    shopping.write_text("item,qty\nmilk,2\neggs,12\n", encoding="utf-8")
    series, warns = drs.read_series_csv(shopping)
    assert series.rows == []
    assert [w.severity for w in warns] == ["critical"]
    assert "does not look like a rate-series rows CSV" in warns[0].message


def test_a_real_rows_csv_still_round_trips(qtbot, dta_cycles_path, tmp_path,
                                           monkeypatch):
    """The floor must not shut the door it was put on."""
    _no_dialogs(monkeypatch)
    widget = _workspace(qtbot, dta_cycles_path)
    widget.rate_add_segments()
    widget.rate_compute_all()
    out = tmp_path / "rows.csv"
    widget.rate_export_rows_csv(str(out))
    slope = widget._rate_refit().slope

    other = _workspace(qtbot, dta_cycles_path)
    n, _warns = other.rate_load_rows_csv(str(out))
    assert n == 4
    assert other._rate_refit().slope == pytest.approx(slope, abs=0.0)


def test_an_empty_series_may_still_be_replaced_by_an_empty_one(qtbot, tmp_path,
                                                              monkeypatch):
    """Nothing is refused: the guard only protects rows that exist."""
    _no_dialogs(monkeypatch)
    widget = _workspace(qtbot)
    header_only = tmp_path / "header_only.csv"
    header_only.write_text("uid,value_C\n", encoding="utf-8")
    n, _warns = widget.rate_load_rows_csv(str(header_only))
    assert n == 0
    assert "as saved" in widget.rate_io_note.text()


# ===========================================================================
# LENS 2 #2 — a computed warning nobody renders is not a warning
# ===========================================================================
def test_missing_source_file_reaches_the_banner(qtbot, dta_cycles_path, tmp_path,
                                                monkeypatch):
    """`read_series_csv` is the only caller that computes it, with
    `verify_paths=True`, and it threw the result away.

    `_rate_loaded_warnings` was assigned at two lines and read at none, so
    `missing_source_file` could not reach the user by any route in the whole
    application. The suite was green because the science test asserts the
    warning is RETURNED, never that a renderer sees it.
    """
    _no_dialogs(monkeypatch)
    widget = _workspace(qtbot, dta_cycles_path)
    widget.rate_add_segments()
    widget.rate_compute_all()
    out = tmp_path / "rows.csv"
    widget.rate_export_rows_csv(str(out))
    moved = str(tmp_path / "gone.txt")
    out.write_text(out.read_text(encoding="utf-8-sig").replace(
        str(dta_cycles_path), moved), encoding="utf-8")

    n, warns = widget.rate_load_rows_csv(str(out))
    assert n == 4
    assert "missing_source_file" in {w.code for w in warns}
    assert "missing_source_file" in {w.code for w in widget.rate_warnings}
    assert "no longer there" in widget.rate_warn_banner.text()


# ===========================================================================
# LENS 2 #3 — the same segment added twice
# ===========================================================================
def test_adding_the_same_segments_twice_is_reported_and_not_refused(
        qtbot, dta_cycles_path, tmp_path, monkeypatch):
    """Measured on the fixture: B bit-identical, sigma 62.72 -> 31.36,
    dof 1 -> 4, and the `dof_one` critical ("R^2 is not evidence here") gone.

    The fit improves in exactly the two places a reader checks. Only
    `duplicate_rate` fired, and its wording ("Both rows are kept, unweighted")
    reads as a legitimate replicate design. Identity is
    `(source_kind, file, segment_index)` and not the uid, which is minted
    fresh on every add — which is also why `RateSeries.validate()` could never
    have caught it. Nothing is refused: M0.1.
    """
    _no_dialogs(monkeypatch)
    widget = _workspace(qtbot, dta_cycles_path)
    widget.rate_add_segments()
    widget.rate_compute_all()
    first = widget._rate_refit()
    assert "duplicate_source_row" not in {w.code for w in first.all_warnings}

    widget.rate_add_segments()
    widget.rate_compute_all()
    second = widget._rate_refit()
    assert len(widget.rate_series.rows) == 8
    assert second.ok and second.n == 2 * first.n
    assert second.slope_se < first.slope_se       # the defect, now narrated
    codes = {w.code for w in second.all_warnings}
    assert "duplicate_source_row" in codes
    warn = next(w for w in second.all_warnings
                if w.code == "duplicate_source_row")
    assert warn.severity == "critical"
    assert len(warn.row_uids) == 8

    out = tmp_path / "rows.csv"
    widget.rate_export_rows_csv(str(out))
    assert "duplicate_source_row" in out.read_text(encoding="utf-8-sig")


def test_two_different_segments_of_one_file_are_not_duplicates(
        qtbot, dta_cycles_path, monkeypatch):
    """The four heats of a cyclic run are the feature, not the defect."""
    _no_dialogs(monkeypatch)
    widget = _workspace(qtbot, dta_cycles_path)
    widget.rate_add_segments()
    widget.rate_compute_all()
    fit = widget._rate_refit()
    assert "duplicate_source_row" not in {w.code for w in fit.all_warnings}


# ===========================================================================
# LENS 2 #4 — the accounting line must partition the rows
# ===========================================================================
def test_the_accounting_counts_add_up_to_the_row_total(qtbot, dta_cycles_path,
                                                       monkeypatch):
    """Measured before: "n = 3 of 8 rows (0 excluded, 3 no value)" — 6 of 8,
    with five markers drawn and the line through three.

    The old line mixed `fit.n` (model level) with counts over `r.include`
    (series level) and its parenthetical knew only two causes, so a model-level
    drop (no T0 under Augis-Bennett) or a missing rate left the sum short. The
    counts now partition: fitted + switched off + no value + dropped by the
    model = every row.
    """
    _no_dialogs(monkeypatch)
    widget = _workspace(qtbot, dta_cycles_path)
    widget.rate_add_segments(roles=("heat", "cool"))
    widget.rate_compute_all()
    model = next(widget.rate_model_combo.itemText(i)
                 for i in range(widget.rate_model_combo.count())
                 if "Augis" in widget.rate_model_combo.itemText(i))
    widget.rate_model_combo.setCurrentText(model)
    fit = widget._rate_refit()

    line = widget._rate_accounting_line(fit)
    got = re.search(r"n = (\d+) of (\d+) rows\s+\((\d+) switched off, "
                    r"(\d+) no value, (\d+) dropped by the model\)", line)
    assert got, line
    n_fit, n_tot, n_off, n_blank, n_dropped = (int(g) for g in got.groups())
    assert n_tot == len(widget.rate_series.rows) == 8
    assert n_fit + n_off + n_blank + n_dropped == n_tot
    assert n_dropped > 0, "the Augis-Bennett rows with no T0 must be counted"
    # every unfitted row is named, not hidden behind "+1 more"
    missing = widget._rate_not_fitted_rows(fit)
    assert len(missing) <= widget.RATE_ACCOUNTING_MAX_NAMES
    assert "+1 more" not in line


def test_the_no_value_count_matches_the_banner_after_a_failed_row_is_excluded(
        qtbot, dta_cycles_path, monkeypatch):
    """The figure printed "0 no value" three lines above "3 row(s) have no value".

    `n_no_value` filtered on `r.include` and `check_series`' `no_value` does
    not, so switching a FAILED row off relabelled it "excluded" and the two
    statements contradicted each other on one screen. A reader of the figure
    alone concluded three rows were deliberately switched off and none failed.
    """
    _no_dialogs(monkeypatch)
    widget = _workspace(qtbot, dta_cycles_path)
    widget.rate_add_segments(roles=("heat", "cool"))
    widget.rate_compute_all()
    blank = [r for r in widget.rate_series.rows
             if not np.isfinite(r.value_C)]
    assert blank, "the fixture must have at least one failed row"
    for r in blank:
        widget.rate_set_include(r.uid, False)
    fit = widget._rate_refit()

    line = widget._rate_accounting_line(fit)
    assert f"{len(blank)} no value" in line
    banner = next(w for w in widget.rate_warnings if w.code == "no_value")
    assert banner.detail["n_no_value"] == len(blank)


def test_the_accounting_line_says_what_to_do_when_every_row_is_switched_off(
        qtbot, dta_cycles_path, monkeypatch):
    """`rate_add_files` on a multi-rate file adds every row switched off — the
    documented design. The remedy the fit prints comes from `dta_kinetics`,
    which has no `include` concept, so the figure read "Compute a Tg for more
    rows, or type values in by hand" when three of four rows already had
    values and the only thing that helps is ticking them on.
    """
    _no_dialogs(monkeypatch)
    widget = _workspace(qtbot, dta_cycles_path)
    widget.rate_add_files([str(dta_cycles_path)])
    assert all(not r.include for r in widget.rate_series.rows)
    widget.rate_compute_all()
    line = widget._rate_accounting_line(widget._rate_refit())
    assert "Every row is switched off" in line


# ===========================================================================
# LENS 2 #5 — the Calculs tab
# ===========================================================================
def test_calculs_reports_a_short_range_inline_with_the_axis_and_its_unit(
        qtbot, dta_cycles_path, monkeypatch):
    """The scope switch moves this axis from C to MINUTES under fixed boxes.

    With the default 350-700 boxes, three of the fifteen scope entries of the
    real cyclic file raised `QMessageBox.critical("Range too small for
    integration.")` — a modal about a fact concerning the data in the window,
    naming neither the curve's range nor the axis change, where the Tg tab
    answers the identical condition inline. And `calc_result_label` was never
    cleared, so after dismissing it the panel still showed segment #1's
    integral while the combo and the plot showed #2.
    """
    _no_dialogs(monkeypatch)
    widget = _workspace(qtbot, dta_cycles_path)
    _select_scope(widget, segment_scope_key(1))
    widget.calc_xmin_edit.setText("350")
    widget.calc_xmax_edit.setText("700")
    widget._calc_integrate()
    on_ramp = widget.calc_result_label.text()
    assert "Integrate" in on_ramp
    assert "350 to 700 °C" in on_ramp, on_ramp

    _select_scope(widget, segment_scope_key(2))
    assert widget.x_combo.currentText().lower().startswith("time")
    widget._calc_integrate()
    on_hold = widget.calc_result_label.text()
    assert "319" not in on_hold and "Integrate (" not in on_hold
    assert "min" in on_hold
    assert "point(s) of this scope" in on_hold
    assert "outside this curve" in on_hold

    widget._calc_find_max()
    assert "point(s) of this scope" in widget.calc_result_label.text()


def test_calculs_labels_an_in_range_hold_integral_with_minutes(
        qtbot, dta_cycles_path, monkeypatch):
    """A hold's time span can overlap 350-700, so the number came out with no
    complaint at all — a mW*min integral in the same unit-less sentence as a
    mW*degC one."""
    _no_dialogs(monkeypatch)
    widget = _workspace(qtbot, dta_cycles_path)
    _select_scope(widget, segment_scope_key(2))
    seg = widget.run.segments[1]
    widget.calc_xmin_edit.setText(f"{seg.t_start + 0.5:.4f}")
    widget.calc_xmax_edit.setText(f"{seg.t_end - 0.5:.4f}")
    widget._calc_integrate()
    text = widget.calc_result_label.text()
    assert "Integrate" in text and " min = " in text, text


# ===========================================================================
# LENS 2 #7 — an add-time sentence must not claim something it cannot know
# ===========================================================================
def test_the_multi_rate_sentence_does_not_claim_all_rows_are_switched_on():
    """It is printed at add time and rendered for as long as the rows live, so
    "all switched on" was false the moment three were switched off — printed
    on the same banner as the line saying three were."""
    rows = [drs.RateRow(source_kind="segment", file="f.txt", segment_index=i,
                        segment_kind=ds.KIND_HEAT, q_measured_C_per_min=q)
            for i, q in enumerate((20.0, 15.0, 10.0), start=1)]
    warns = drs._source_warnings(rows, "cyc", rates_were_asked_for=True)
    assert [w.code for w in warns] == ["multi_rate_source"]
    assert "all switched on" not in warns[0].message
    assert "added switched on" in warns[0].message


def test_the_excluded_rows_sentence_names_both_emitters():
    """It pointed the reader at `ambiguous_file_rate`, which is not on the
    banner when the rows came through the segment door — that raises
    `multi_rate_source`."""
    series = drs.RateSeries(quantity_kind="Tg")
    for i in range(2):
        row = drs.RateRow(source_kind="segment", file="f.txt", segment_index=i,
                          q_measured_C_per_min=10.0 * (i + 1), value_C=500.0)
        row.include = False
        series.add(row)
    warn = next(w for w in drs.check_series(series) if w.code == "excluded_rows")
    assert "ambiguous_file_rate / multi_rate_source" in warn.message


# ===========================================================================
# LENS 3 M2 — "Never raises on data" has to be true
# ===========================================================================
@pytest.mark.parametrize("header", ["a string header", ["a", "list"], 7, 3.5,
                                    object()])
def test_rows_from_segments_never_raises_on_a_non_mapping_header(
        dta_cycles_path, header):
    """`dict(header or {})` raised ValueError on a string and TypeError on an
    int, from a function documented "Never raises on data".

    Reachable: `io_universal.parse_saxs_edf_ascii` stores `meta["raw_header"]`
    as a single joined STRING and `rows_from_record` forwards it unchecked, so
    a scripted batch over a mixed library crashed where the GUI only showed a
    failed row whose stated reason was "dictionary update sequence element #0
    has length 1; 2 is required".
    """
    spectrum = _load_spectrum_from_path(str(dta_cycles_path))
    df = spectrum.df
    run = ds.segment_run(
        pd.to_numeric(df[df.columns[0]], errors="coerce").to_numpy(float),
        pd.to_numeric(df[df.columns[1]], errors="coerce").to_numpy(float))
    rows, _warns = drs.rows_from_segments(run, df, header=header,
                                          file=str(dta_cycles_path))
    assert len(rows) == 4


def test_rows_from_record_never_raises_on_a_string_raw_header(dta_cycles_path):
    spectrum = _load_spectrum_from_path(str(dta_cycles_path))
    rec = {"title": "x", "path": str(dta_cycles_path), "df": spectrum.df,
           "meta": {"raw_header": "Sample\nSize\nnot a mapping"}}
    rows, _warns = drs.rows_from_record(rec)
    assert len(rows) == 4


# ===========================================================================
# LENS 3 L2 / L3 — housekeeping that bites later
# ===========================================================================
def test_removing_every_row_releases_its_cached_frame(qtbot, dta_cycles_path,
                                                      monkeypatch):
    """Only `rate_clear()` freed anything, so a session working through fifty
    files held every frame it had measured (~0.85 MiB per 2 MB export)."""
    _no_dialogs(monkeypatch)
    widget = _workspace(qtbot, dta_cycles_path)
    widget.rate_add_segments()
    widget.rate_compute_all()
    assert widget._rate_frames

    uids = [r.uid for r in widget.rate_series.rows]
    widget.rate_remove(uids[:1])
    assert widget._rate_frames, "the other three rows still need that frame"
    widget.rate_remove(uids[1:])
    assert widget.rate_series.rows == []
    assert widget._rate_frames == {}
    assert widget._rate_canonical == {}


def test_a_non_finite_extra_is_json_safe(qtbot, dta_cycles_path, monkeypatch):
    """`to_dict`'s "NaN becomes None so `json.dumps` stays valid" was false for
    `extras`, and two of the eight rows the fixture produces carry a NaN there
    — a row whose parallel tangent failed. A bare `NaN` token went into
    `dta_rate_series.json`, which PRISM reads back and a strict JSON reader
    refuses.
    """
    _no_dialogs(monkeypatch)
    widget = _workspace(qtbot, dta_cycles_path)
    widget.rate_add_segments(roles=("heat", "cool"))
    widget.rate_compute_all()
    assert any(not np.isfinite(v)
               for r in widget.rate_series.rows for v in r.extras.values())

    records = widget.rate_series_records()
    blob = json.dumps(records, allow_nan=False)        # used to raise
    restored = drs.RateSeries.from_dict(json.loads(blob)[0])
    assert len(restored.rows) == len(widget.rate_series.rows)
    for before, after in zip(widget.rate_series.rows, restored.rows):
        for key, value in before.extras.items():
            got = after.extras.get(key, float("nan"))
            if np.isfinite(value):
                assert got == pytest.approx(value, abs=0.0)
            else:
                assert not np.isfinite(got)


# ===========================================================================
# LENS 1 F2 — the documented accuracy bound has to be the measured one
# ===========================================================================
def test_estimate_rate_meets_the_bound_its_docstring_states():
    """The docstring claimed 4e-6 C/min "from 2 000 to 1 000 000 points, at any
    time offset". Measured worst case 3.7e-5, on a million points at
    0.001 min/pt: the residual is prefix-sum cancellation and grows with the
    point count and with 1/dt. It changes no segmentation, no rate and no fit —
    3.7e-5 is 1.9 ppm of a 20 C/min ramp — but it is a contract the next editor
    will trust, so the number in it is the measured one now.
    """
    doc = ds.estimate_rate.__doc__ or ""
    found = re.search(r"accurate to ([0-9.]+e-[0-9]+) C/min", doc)
    assert found, "the accuracy claim must stay greppable"
    bound = float(found.group(1))

    worst = 0.0
    for n, dt, t0 in ((2221, 0.5 / 60.0, 1e7),
                      (100_000, 0.01, 1e5),
                      (1_000_000, 0.001, 1e7)):
        t = t0 + np.arange(n) * dt
        rate = ds.estimate_rate(t, 20.0 * t, window_min=max(0.5, 20.0 * dt))
        worst = max(worst, float(np.nanmax(np.abs(rate - 20.0))))
    assert worst <= bound, f"worst {worst:.3e} exceeds the documented {bound:.0e}"
# ===========================================================================
# ROBUST-L6 (deferred, now fixed) — the cross-package helper must not vanish
#                                   in silence
# ===========================================================================
def test_the_filename_rate_helper_this_module_borrows_still_exists():
    """`rate_from_filename` delegates to `xrd.htxrd_science`, inside a bare
    `except Exception: return None`.

    Delegating is right -- that function already handles the Windows-basename
    rule, case insensitivity and the comma decimal separator, and a second copy
    is a second place for them to drift. Swallowing the ImportError is also
    right, because this module must import without the XRD package. Together
    they mean that renaming or moving the upstream function turns the feature
    off with no error anywhere: `rate_from_filename` simply answers None for
    every input, which is indistinguishable from "this name does not match the
    template".

    So the contract gets a test instead of a silence. If the XRD package
    renames `ramp_value_from_template`, this fails and names the dependency,
    rather than a user noticing months later that filename rates stopped
    being read.
    """
    from xrd.htxrd_science import ramp_value_from_template

    # The upstream behaviour this module relies on, not merely its existence.
    assert ramp_value_from_template("sample_010.txt", "sample_???.txt") == 10.0
    assert ramp_value_from_template("SAMPLE_010.TXT", "sample_???.txt") == 10.0
    assert ramp_value_from_template(r"C:\\data\\sample_010.txt",
                                    "sample_???.txt") == 10.0
    assert ramp_value_from_template("sample_010.txt", "other_???.txt") is None

    # ... and that the delegation actually carries it through.
    assert drs.rate_from_filename("glass_020.txt", "glass_???.txt") == 20.0
    assert drs.rate_from_filename("glass_020.txt", "nope_???.txt") is None
    assert drs.rate_from_filename("", "glass_???.txt") is None
    assert drs.rate_from_filename("glass_020.txt", "") is None


# ===========================================================================
# ROBUST-H2 residue (deferred, now fixed) — a slow ramp with NO method file
# ===========================================================================
def test_a_slow_ramp_with_no_method_file_is_still_a_ramp():
    """The 0.20 C/min noise floor used to veto a genuine slow ramp.

    The floor exists because p95(|dT/dt|) on a file that is one long hold is
    noise, and a fraction of noise is still noise. The programmed path stopped
    consulting it when H2 was fixed, but a file with NO method still hit it: a
    0.1 C/min scan -- the standard range for enthalpy-relaxation and
    fictive-temperature work, which is exactly what the rate feature is for --
    came back as a single `isothermal` with no rate, no temperature abscissa
    and no Tg.

    Noise is recognisable because it goes nowhere, so the floor is now skipped
    when the record's own temperature moves at least `min_ramp_span_C` end to
    end. Both halves are pinned here: the slow ramp reads as a ramp, and a real
    soak does NOT fragment.
    """
    t = np.arange(0.0, 5000.0, 1.0)
    T = 25.0 + 0.1 * t                       # 25 -> 525 C, 500 C of excursion
    run = ds.segment_run(t, T)
    assert len(run.segments) == 1
    assert run.segments[0].kind == ds.KIND_HEAT
    assert run.segments[0].rate_mode == pytest.approx(0.1, abs=5e-3)

    # The control: a genuine soak, whose total excursion is a third of a degree.
    rng = np.random.default_rng(7)
    t_hold = np.arange(0.0, 600.0, 0.2)
    T_hold = 800.0 + rng.normal(0.0, 0.05, t_hold.size)
    assert float(T_hold.max() - T_hold.min()) < ds.SegmentationConfig().min_ramp_span_C
    hold = ds.segment_run(t_hold, T_hold)
    assert len(hold.segments) == 1
    assert hold.segments[0].kind == ds.KIND_ISOTHERMAL


def test_a_hold_after_a_big_ramp_is_still_a_hold():
    """The span test is about the WHOLE record, so a 1000 C ramp turns the
    floor off for the soak that follows it too. The flat-core test is what
    keeps that soak a hold, and this pins that the two rules together still
    give heat-then-hold rather than one long ramp."""
    rng = np.random.default_rng(7)
    t = np.concatenate([np.arange(0.0, 100.0, 0.1),
                        100.0 + np.arange(0.0, 60.0, 0.1)])
    T = np.concatenate([25.0 + 10.0 * np.arange(0.0, 100.0, 0.1),
                        1025.0 + rng.normal(0.0, 0.05, 600)])
    run = ds.segment_run(t, T)
    assert [s.kind for s in run.segments] == [ds.KIND_HEAT, ds.KIND_ISOTHERMAL]
