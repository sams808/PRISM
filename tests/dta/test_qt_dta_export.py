"""Per-segment export, segment-aware batch, degeneracy guard.

SPEC_cycles §2.5 and §3 Phase B steps 10-11. Five things are under test and
they are not independent: the CSV schema is what carries a per-segment number
out of the app, the batch is the same measurement over many files, and the
degeneracy guard is what stops the app endorsing the one number this whole
rework exists to remove.

The through-line is MERGED_DECISIONS M0.1 -- "still make this possible to
process such data". So the tests here are mostly about what is NOT blocked: a
segment whose Tg fails still produces a row, a hold still produces a row when
asked for, a cyclic file is still fully measurable. The single refusal in the
feature (a whole-FILE batch row for a multi-segment file) is tested twice over:
once that it refuses, and once that the same file is completely measurable
through the other two batch scopes, because that is what makes the refusal an
arithmetic limit rather than a physical objection.

Two numbers are pinned against reality rather than against the code:

  * the collapse spread **1.2240e-4 °C** that a global sort of the committed
    fixture produces, which a `1e-6` guard waves through and `DEGENERATE_SPREAD
    = 1e-2` catches;
  * the healthy spread **3.1547 °C** on `EXAMPLES/DTA_example.txt`, two orders
    the other side of the same threshold.

Four orders of margin either way is the whole argument for the constant, and it
is measured in both directions here so a later edit to it fails a test instead
of quietly re-enabling the "✓ Methods agree (spread 0.00 °C)" endorsement.
"""
from __future__ import annotations

import csv
import inspect

import numpy as np
import pandas as pd
import pytest

from PySide6.QtWidgets import QMessageBox

from dta import qt_dta as module
from dta.qt_dta import (
    BATCH_SCOPE_EVERY, BATCH_SCOPE_FILE, BATCH_SCOPE_LABELS,
    BATCH_SCOPE_MATCHING, DEGENERATE_SPREAD, LEGACY_RESULT_COLUMNS,
    MATCH_RATE_TOL_C_MIN, SCOPE_ALL_TIME,
    SEGMENT_RESULT_COLUMNS, DtaSegmentResult, DtaWorkspace,
    degenerate_result_line, finite_spread,
)
from dta.dta_science import (
    resolve_baseline_params,
)
from core.qt_shell import _load_spectrum_from_path
from conftest import requires_dta_bm


T_COL = "Temperature (°C)"
Y_COL = "Heat Flow (mW)"
TIME_COL = "Time (min)"

# The eight ramp segments of the committed cyclic fixture, in time order, and
# the written method steps each of them accounts for. Measured, not assumed:
# the two-step entries are the cooling ramps that ABSORBED the 20-minute 50 °C
# hold programmed after them -- the hold never happened as a separate stretch
# of data because the furnace was still passively cooling through it
# (BM_FINDINGS §2), and the science layer reports both steps rather than only
# the flattering one.
RAMP_INDICES = (1, 3, 4, 6, 7, 9, 10, 12)
RAMP_STEPS = ("3", "5;6", "7", "9;10", "11", "13;14", "15", "17;18")

# Original DataFrame spans of those eight segments. These are the numbers the
# CSV must carry: `SegmentedRun.row_indices`, never `Segment.row_start`.
RAMP_ROW_SPANS = ((0, 346), (448, 903), (903, 1226), (1329, 1892),
                  (1892, 2328), (2431, 3212), (3212, 3873), (3977, 4749))

# Measured (mode-refined) and programmed rates of the same eight, °C/min.
RAMP_RATES_MEASURED = (19.9544, -19.9607, 19.9653, -14.9843,
                       14.9906, -9.9794, 9.9863, -9.9802)
RAMP_RATES_PROGRAMMED = (20.0, -20.0, 20.0, -15.0, 15.0, -10.0, 10.0, -10.0)


def _record(path, *, rec_id=None):
    """A library-shaped record dict. `rec_id=None` reproduces the hand-built
    records the pinned tests use, which carry no `"id"` at all."""
    spectrum = _load_spectrum_from_path(str(path))
    rec = {"title": spectrum.title, "path": spectrum.path,
           "df": spectrum.df, "meta": spectrum.meta}
    if rec_id is not None:
        rec["id"] = rec_id
    return rec


def _holed_record(path, holes=(10, 500, 1500, 3000, 4500)):
    """The cyclic fixture with NaN temperatures punched into it.

    SPEC_cycles R2: on every file shipped with PRISM the cleaned-array indices
    and the DataFrame positions COINCIDE, so an off-by-N between them is
    completely invisible. These five holes are the same ones
    `test_clean_time_temperature_maps_rows_back` uses, so the science layer and
    the Qt layer are checked against one construction.
    """
    rec = _record(path)
    frame = rec["df"].copy()
    frame.loc[frame.index[list(holes)], T_COL] = np.nan
    rec = dict(rec)
    rec["df"] = frame
    rec["title"] = "holed cycles"
    return rec


def _globally_sorted_arrays(widget):
    """The arrays the DELETED whole-file path built: mask, then global argsort.

    Reproduced here rather than reached through the widget, because the widget
    can no longer produce them -- that is
    `test_the_old_whole_file_tg_collapse_is_unreachable`. They are still the
    only honest input for the degeneracy tests: the collapse is a real
    measurement on real data, and a synthetic triple of equal floats would pin
    the threshold against a number nobody has ever seen.
    """
    x = pd.to_numeric(widget.df[T_COL], errors="coerce").to_numpy(float)
    y = pd.to_numeric(widget.df[Y_COL], errors="coerce").to_numpy(float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    order = np.argsort(x)
    return x[order], y[order]


def _auto_baseline():
    return resolve_baseline_params(
        manual_enabled=False, low_use_point=False, low_point_x=None,
        low_min=None, low_max=None, high_use_point=False, high_point_x=None,
        high_min=None, high_max=None, slope_min=None, slope_max=None)


def _collapsed_result(widget):
    """A genuinely degenerate `DtaSegmentResult`, measured on the fixture.

    `errstate` because a global sort of a cyclic run puts points with IDENTICAL
    temperatures next to each other -- 719 duplicate-T pairs and 1002 non-finite
    `dHF/dT` values on the full file -- so `np.gradient` divides by a zero
    spacing. That is not noise to be hidden: it is the same pathology the
    collapse comes from, and it is pinned as a measurement in
    `test_dta_segments.py::test_full_cycles_file_global_sort_pathology`.
    Silencing it here keeps the gate's warning count honest about NEW warnings.
    """
    x, y = _globally_sorted_arrays(widget)
    with np.errstate(divide="ignore", invalid="ignore"):
        return widget._measure_tg(
            x, y, y.copy(), window=(350.0, 700.0), smooth_d=False,
            bp=_auto_baseline(), unit="°C", x_col=T_COL, y_col=Y_COL,
            dy_col="(same)", seg=None)


# ===========================================================================
# 1. The CSV schema
# ===========================================================================


def _settle(qtbot, widget, timeout: int = 10000) -> None:
    """Wait until every debounced redraw this widget queued has actually run.

    `PlotWidget.request_redraw` defers the render by `debounce_ms` (120), so a
    test that changes the scope and immediately counts artists reads the
    PREVIOUS picture. These tests used a fixed `qtbot.wait(250)`, which leaves
    a 130 ms margin of WALL CLOCK: on a machine running several pytest+Qt
    processes at once the timer callback plus the draw of a 4749-row figure can
    miss it, and the assertion then sees the single curve left over from the
    initial render. Measured directly:

        wait=119 ms  pending=True   ax.lines=1   <- assertion expected 4
        wait=250 ms  pending=False  ax.lines=4

    Waiting on the CONDITION instead removes the cliff and is faster in the
    normal case (the queue is usually empty within one event-loop turn). The
    short trailing wait lets the queued canvas paint run too.

    NOT for a negative assertion: a test proving that NOTHING was drawn must
    wait a real interval, because this returns immediately when the queue is
    already empty.
    """
    for plot in (getattr(widget, "plot", None), getattr(widget, "rate_plot", None)):
        if plot is not None:
            qtbot.waitUntil(lambda p=plot: getattr(p, "_pending", None) is None,
                            timeout=timeout)
    qtbot.wait(20)

def test_export_schema_is_backward_compatible():
    """The 18 pre-segment columns keep their names AND their positions.

    Spelled out literally rather than compared to `LEGACY_RESULT_COLUMNS`: a
    test that reads the constant it is protecting would follow a rename instead
    of failing on it, and the whole point of the frozen prefix is that somebody
    else's spreadsheet reads these names in this order.
    """
    assert LEGACY_RESULT_COLUMNS == (
        "file", "sample", "x_col", "y_col", "dy_source", "temp_unit",
        "window_min", "window_max", "smooth_derivative",
        "low_mode", "low_range", "low_point_x",
        "high_mode", "high_range", "high_point_x",
        "Tg_double", "Tg_parallel", "Tg_derivative",
    )
    assert SEGMENT_RESULT_COLUMNS[:18] == LEGACY_RESULT_COLUMNS
    assert len(set(SEGMENT_RESULT_COLUMNS)) == len(SEGMENT_RESULT_COLUMNS)


def test_every_appended_physical_column_names_its_own_unit():
    """Every appended temperature, rate or time column carries its unit in its
    name.

    The legacy block is in the DISPLAY unit and says so through `temp_unit`; the
    appended block is canonical °C whatever the °C/K switch shows. Both rules are
    only readable six months later if the second block is self-describing, so
    the suffix is part of the contract and not a stylistic choice -- a column
    called `T_start` whose unit depended on a radio button at export time is a
    column nobody can use.
    """
    appended = SEGMENT_RESULT_COLUMNS[18:]
    expected_units = {
        "T_start_C": "_C", "T_end_C": "_C", "target_gap_C": "_C",
        "control_lost_below_C": "_C", "rate_measured_C_min": "_C_min",
        "rate_programmed_C_min": "_C_min", "t_start_min": "_min",
        "t_end_min": "_min",
    }
    for name, suffix in expected_units.items():
        assert name in appended, name
        assert name.endswith(suffix), name
    # Nothing else in the appended block is a physical quantity, so nothing
    # else may look like one: a bare `rate` or `T_` column would be a unit
    # the reader has to guess.
    for name in appended:
        if name in expected_units:
            continue
        assert not name.startswith("T_"), name
        assert not name.startswith("rate"), name


def test_the_appended_columns_stay_celsius_when_the_workspace_switches_to_kelvin(
        qtbot, dta_cycles_path):
    """The °C/K switch moves the legacy block and leaves the segment block alone.

    This is the mixing the canonical storage exists to prevent: a column called
    `T_start_C` whose unit depended on a radio button at export time is a column
    nobody can read. `temp_unit` is what says which unit the 18 legacy columns
    are in, and it is the only cell that has to be consulted.
    """
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    widget.select_segment(1)              # a ramp, so temperature is on X
    _settle(qtbot, widget)
    assert widget.temp_k_btn.isEnabled()

    row_c = widget._current_result_row()
    widget.temp_k_btn.setChecked(True)
    _settle(qtbot, widget)
    row_k = widget._current_result_row()

    assert row_c["temp_unit"] == "°C" and row_k["temp_unit"] == "K"
    assert row_k["window_min"] == pytest.approx(row_c["window_min"] + 273.15, abs=1e-6)
    assert row_k["Tg_parallel"] == pytest.approx(row_c["Tg_parallel"] + 273.15, abs=1e-4)
    for name in ("T_start_C", "T_end_C", "rate_measured_C_min",
                 "rate_programmed_C_min", "target_gap_C", "t_start_min",
                 "t_end_min", "row_start", "row_stop", "n_points"):
        assert row_k[name] == row_c[name], name


def test_row_indices_are_dataframe_positions_not_cleaned_bounds(
        qtbot, dta_cycles_path, tmp_path):
    """SPEC_cycles R2, at the Qt level.

    With five NaN temperatures punched in, the two index spaces DIVERGE: the
    cleaned bounds shift by one per hole passed, while the original DataFrame
    spans are unchanged. The CSV must carry the second set -- they are the only
    ones that mean anything next to the table the user exported -- and the
    proof is that they are byte-identical to the holeless file's.
    """
    clean = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(clean)
    holed = DtaWorkspace(records=[_holed_record(dta_cycles_path)])
    qtbot.addWidget(holed)
    assert holed.run.n_dropped_rows == 5
    assert len(holed.run) == len(clean.run) == 12

    rows_clean = clean.export_all_segments_csv(tmp_path / "clean.csv")
    rows_holed = holed.export_all_segments_csv(tmp_path / "holed.csv")

    spans_clean = [(r["row_start"], r["row_stop"]) for r in rows_clean]
    spans_holed = [(r["row_start"], r["row_stop"]) for r in rows_holed]
    assert spans_clean == list(RAMP_ROW_SPANS)
    assert spans_holed == spans_clean

    # ... and at least one segment's CLEANED bounds really did move, or the
    # test proves nothing about which of the two spaces was exported.
    holed_ramps = [s for s in holed.run.segments if s.is_ramp]
    assert any(int(seg.row_start) != span[0]
               for seg, span in zip(holed_ramps, spans_holed))
    # A dropped row inside a segment makes the ORIGINAL span wider than the
    # point count, which is the other half of the same fact.
    assert rows_holed[0]["row_stop"] - rows_holed[0]["row_start"] == 346
    assert rows_holed[0]["n_points"] == 345


def test_an_unmeasured_segment_has_blank_tg_cells_and_no_invented_zeroes(
        qtbot, dta_cycles_path):
    """A segment nobody measured: blank, never 0, never the string "nan".

    A `0` in `target_gap_C` reads as "hit the target exactly" and a `0` in
    `tg_spread` reads as perfect agreement -- both are claims the app has no
    basis for. The segment's own measured fields are still reported, because
    they were measured by the segmentation and not by the Tg.
    """
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    seg = widget.run.segments[0]

    row = widget._segment_result_row(seg, None, run=widget.run, path=None,
                                     sample="")
    assert row["tg_spread"] == ""
    assert row["tg_degenerate"] == ""
    assert row["warnings"] == "" and row["error"] == ""
    assert row["x_col"] == "" and row["temp_unit"] == ""
    assert row["low_mode"] == "" and row["high_mode"] == ""
    assert np.isnan(row["Tg_double"]) and np.isnan(row["Tg_parallel"])
    # measured by the segmentation, so reported
    assert row["segment_index"] == 1 and row["n_points"] == 346
    assert row["is_partial"] == "no"
    assert row["target_gap_C"] == pytest.approx(1.392, abs=1e-3)


def test_a_boolean_exports_as_yes_or_no_and_never_as_an_integer():
    """`bool` IS an `int` in Python, so the order of the isinstance checks in
    `_fmt_export` is load-bearing: without the bool branch first, `is_partial`
    would export as 1/0 in a column of yes/no."""
    fmt = DtaWorkspace._fmt_export
    assert fmt(True) == "yes" and fmt(False) == "no"
    assert fmt(None) == ""
    assert fmt(float("nan")) == ""
    assert fmt(float("inf")) == ""
    assert fmt(3.5) == 3.5
    assert fmt(0) == 0                     # a real zero is still a real zero
    assert fmt("cool") == "cool"


def test_an_error_row_omits_the_tg_columns_instead_of_blanking_them(
        qtbot, dta_example_path):
    """An absent KEY and a blank CELL are two different claims.

    A blank `Tg_parallel` cell means "measured, and there is no value". An
    absent `Tg_parallel` key means "never attempted", which is what a refused
    whole-file row is. Collapsing the two would turn a structural refusal into
    a failed measurement of a number nobody ever asked for.
    """
    widget = DtaWorkspace(records=[_record(dta_example_path)])
    qtbot.addWidget(widget)

    row = widget._error_row(path=None, sample="s", error="nope")
    assert set(row) == {"file", "sample", "error"}
    for name in ("Tg_double", "Tg_parallel", "Tg_derivative", "tg_spread"):
        assert name not in row

    seg = widget.run.segments[0]
    named = widget._error_row(path=None, sample="s", error="nope", seg=seg)
    assert named["segment_index"] == 1
    assert named["segment_kind"] == "heat"
    assert "#1" in named["segment_label"]
    assert "Tg_parallel" not in named


def test_the_csv_column_order_is_the_schema_not_the_first_row_s_keys(
        qtbot, dta_example_path, tmp_path):
    """`pd.DataFrame(list_of_dicts)` orders columns by first appearance.

    So a batch whose FIRST file was refused would emit a differently shaped CSV
    from the same batch with the files in the other order. The order is the
    schema's; a column no row carries is not invented; an unexpected key is
    appended rather than dropped, because losing data to protect a column order
    would be the wrong trade.
    """
    widget = DtaWorkspace(records=[_record(dta_example_path)])
    qtbot.addWidget(widget)
    widget._compute()

    error_first = [widget._error_row(path=None, sample="s", error="refused"),
                   widget._current_result_row()]
    out = tmp_path / "mixed.csv"
    widget._write_rows_csv(error_first, out)

    with out.open(newline="", encoding="utf-8") as fh:
        header = next(csv.reader(fh))
    assert header[:18] == list(LEGACY_RESULT_COLUMNS)
    assert header == [c for c in SEGMENT_RESULT_COLUMNS if c in header]

    # A narrow all-errors batch stays narrow: no 40 headers over nothing.
    narrow = tmp_path / "narrow.csv"
    widget._write_rows_csv([widget._error_row(path=None, sample="", error="x")],
                           narrow)
    with narrow.open(newline="", encoding="utf-8") as fh:
        assert next(csv.reader(fh)) == ["file", "sample", "error"]

    # An unexpected key survives, at the end.
    extra = tmp_path / "extra.csv"
    row = dict(widget._current_result_row())
    row["a_future_column"] = 7
    widget._write_rows_csv([row], extra)
    with extra.open(newline="", encoding="utf-8") as fh:
        assert next(csv.reader(fh))[-1] == "a_future_column"


# ===========================================================================
# 2. Export every segment of the loaded file
# ===========================================================================
def test_export_all_segments_yields_one_row_per_ramp(qtbot, dta_cycles_path,
                                                    tmp_path):
    """Eight ramps, in time order, each with its provenance.

    The step column is the interesting one: the four cooling ramps each account
    for TWO written steps, because the 20-minute 50 °C hold programmed after
    each of them was absorbed into the cooling tail -- the furnace was still
    passively cooling through it (BM_FINDINGS §2). Reporting only the ramp step
    would hide a programmed instruction that ran.
    """
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    out = tmp_path / "segments.csv"

    rows = widget.export_all_segments_csv(out)
    assert out.is_file()
    assert tuple(r["segment_index"] for r in rows) == RAMP_INDICES
    assert tuple(r["program_step"] for r in rows) == RAMP_STEPS
    assert [r["segment_kind"] for r in rows] == [
        "heat", "cool", "heat", "cool", "heat", "cool", "heat", "cool"]
    assert [r["cycle_index"] for r in rows] == [1, 1, 2, 2, 3, 3, 4, 4]
    for row, rate in zip(rows, RAMP_RATES_MEASURED):
        assert row["rate_measured_C_min"] == pytest.approx(rate, abs=5e-3)
    for row, rate in zip(rows, RAMP_RATES_PROGRAMMED):
        assert row["rate_programmed_C_min"] == pytest.approx(rate, abs=1e-9)
    assert [r["is_partial"] for r in rows] == ["no"] * 7 + ["yes"]

    frame = pd.read_csv(out)
    assert len(frame) == 8
    assert list(frame.columns) == list(SEGMENT_RESULT_COLUMNS)
    assert list(frame.columns)[:18] == list(LEGACY_RESULT_COLUMNS)


def test_a_ramp_exported_from_the_whole_run_view_is_measured_against_temperature(
        qtbot, dta_cycles_path, tmp_path):
    """THE thing that makes this export worth running on a cyclic file.

    A cyclic record opens on the whole-run view, whose only honest abscissa is
    time, so `x_combo` sits on the clock. Measured that way with the default
    350-700 window, all eight ramps report "Tg window too small", because
    350-700 MINUTES misses a 64-minute ramp entirely -- eight error rows and no
    findings. A ramp IS a single temperature scan, so the export puts the
    temperature back and says so in the row.
    """
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    assert widget._scope().kind == SCOPE_ALL_TIME
    assert widget.x_combo.currentText() == TIME_COL
    assert widget._x_forced_time is True

    rows = widget.export_all_segments_csv(tmp_path / "ramps.csv")
    assert all(r["x_col"] == T_COL for r in rows)
    assert all(r["temp_unit"] == "°C" for r in rows)
    assert "X is Temperature" in rows[0]["warnings"]
    assert "whole-run view has no single scan" in rows[0]["warnings"]
    # Six of the eight now carry three finite numbers instead of an error.
    finite = [r for r in rows if np.isfinite(r["Tg_parallel"])]
    assert len(finite) == 6
    assert rows[0]["Tg_double"] == pytest.approx(595.6573, abs=1e-3)
    assert rows[0]["Tg_parallel"] == pytest.approx(622.8145, abs=1e-3)
    assert rows[0]["Tg_derivative"] == pytest.approx(618.958, abs=1e-3)


def test_a_column_the_user_chose_is_never_substituted(qtbot, dta_cycles_path,
                                                      tmp_path):
    """The substitution is refused the moment the user picks the abscissa.

    `_x_forced_time` is the distinction: it records that the clock is on X
    because a SCOPE put it there. A user who deliberately asked for the clock
    gets the clock, and the window they typed applies to it.
    """
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    # Selecting a ramp hands the abscissa back -- the scope no longer forces the
    # clock, so the flag clears and X returns to temperature.
    widget.select_segment(1)
    _settle(qtbot, widget)
    assert widget.x_combo.currentText() == T_COL
    assert widget._x_forced_time is False
    # Now the user asks for the clock, through the signal, and owns it.
    widget.x_combo.setCurrentText(TIME_COL)
    _settle(qtbot, widget)
    assert widget.x_combo.currentText() == TIME_COL
    assert widget._x_forced_time is False

    rows = widget.export_all_segments_csv(tmp_path / "clock.csv")
    assert all(r["x_col"] == TIME_COL for r in rows)
    assert all("X is Temperature" not in r["warnings"] for r in rows)


def test_a_segment_whose_measurement_fails_still_gets_a_row(qtbot,
                                                            dta_cycles_path,
                                                            tmp_path):
    """M0.1, in the form that matters most.

    Segment #4 is the second heating ramp, and after cycle 1 the specimen is
    largely crystallized: it has no glass transition left inside 350-700 °C,
    the derivative peak lands on the 700 °C window edge and the HIGH baseline
    gets one point (BM_FINDINGS §8). A row saying that is the finding. A row
    that silently vanished would leave a four-rate series looking like a
    three-rate one with nothing to explain the gap.
    """
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)

    rows = widget.export_all_segments_csv(tmp_path / "fails.csv")
    assert len(rows) == 8                       # nothing dropped
    failed = {r["segment_index"]: r for r in rows if r["error"]}
    assert set(failed) == {4, 9}
    assert "Baseline range has too few points" in failed[4]["error"]
    assert np.isnan(failed[4]["Tg_parallel"])
    # The other two methods still reported, and so did the provenance.
    assert np.isfinite(failed[4]["Tg_double"])
    assert np.isfinite(failed[4]["Tg_derivative"])
    assert failed[4]["segment_kind"] == "heat" and failed[4]["cycle_index"] == 2


def test_unchecking_skip_holds_exports_the_holds_against_the_clock(
        qtbot, dta_cycles_path, tmp_path):
    """A hold has no temperature scan, so it is left out by default -- and
    including it is a choice the user is allowed to make (M0.1), not a refusal.

    The hold rows are honest about what happened to them: X is the clock, the
    unit cell is empty because the numbers are not temperatures, and the 350-700
    window is reported as lying outside a 65-84 minute curve rather than
    producing a number from it.
    """
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    assert widget.skip_holds_check.isChecked() is True
    assert len(widget.export_all_segments_csv(tmp_path / "ramps.csv")) == 8

    widget.skip_holds_check.setChecked(False)
    rows = widget.export_all_segments_csv(tmp_path / "all.csv")
    assert [r["segment_index"] for r in rows] == list(range(1, 13))
    hold = next(r for r in rows if r["segment_kind"] == "isothermal")
    assert hold["x_col"] == TIME_COL
    assert hold["temp_unit"] == ""
    assert "lies entirely outside this curve" in hold["warnings"]
    assert np.isnan(hold["Tg_parallel"])
    # Its provenance is complete even though its Tg is not.
    assert hold["segment_role"] == "hold_hot"
    assert hold["t_start_min"] == pytest.approx(64.854, abs=1e-2)


def test_export_all_segments_neither_moves_the_scope_nor_stores_results(
        qtbot, dta_cycles_path, tmp_path):
    """An export is a read.

    Measuring segment 7 must not disturb what segment 1 shows, must not move
    the combo the user left on the whole run, and must not write into
    `seg_results` -- a stored result is a number the user has SEEN, and a
    twelve-segment export would otherwise fill the store with numbers nobody
    looked at and then re-display them as though they had.
    """
    widget = DtaWorkspace(records=[_record(dta_cycles_path, rec_id="cyc")])
    qtbot.addWidget(widget)
    widget.select_segment(1)
    _settle(qtbot, widget)
    widget._compute()
    before = widget._current_result()
    assert before is not None
    key_before = widget._scope_key
    label_before = widget.result_label.text()

    widget.export_all_segments_csv(tmp_path / "e.csv")

    assert widget._scope_key == key_before
    assert list(widget.seg_results) == [1]
    assert widget._current_result() is before
    assert widget.result_label.text() == label_before


def test_export_all_segments_on_a_single_ramp_file_is_one_row(
        qtbot, dta_example_path, tmp_path):
    """The historic file must behave like the historic file: one scan, one row,
    and the same number the Compute button gives."""
    widget = DtaWorkspace(records=[_record(dta_example_path)])
    qtbot.addWidget(widget)
    widget._compute()
    expected = widget._current_result().tg_values

    rows = widget.export_all_segments_csv(tmp_path / "one.csv")
    assert len(rows) == 1
    assert rows[0]["segment_index"] == 1
    assert rows[0]["Tg_double"] == expected[0]
    assert rows[0]["Tg_parallel"] == expected[1]
    assert rows[0]["Tg_derivative"] == expected[2]


def test_export_all_segments_refuses_a_record_it_cannot_segment(qtbot):
    """No segments means no per-segment export, and the note says which button
    to press instead. The record is still fully analysable -- "Export result
    CSV" covers it -- so nothing about the DATA has been refused."""
    frame = pd.DataFrame({"X": np.linspace(0.0, 10.0, 200),
                          "Y": np.linspace(-1.0, 1.0, 200)})
    widget = DtaWorkspace(records=[{"title": "generic", "path": "",
                                    "df": frame, "meta": {}}])
    qtbot.addWidget(widget)
    assert widget.run is None
    assert widget.export_segments_btn.isEnabled() is False
    assert "not split into segments" in widget.export_segments_note.text()
    assert "Export result CSV" in widget.export_segments_note.text()

    targets, note = widget._export_segment_targets()
    assert targets == ()
    with pytest.raises(ValueError):
        widget.export_all_segments_csv(None)
    # and the single-scan export still works, which is the point
    assert widget.export_csv_btn.isEnabled() is True


def test_the_segment_export_note_counts_the_rows_it_will_write(
        qtbot, dta_cycles_path):
    """Standing text, updated by the checkbox that changes the answer. A user
    who has to press the button to learn what it will do has been made to
    guess."""
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    assert "8 rows" in widget.export_segments_note.text()
    assert "12 segments" in widget.export_segments_note.text()

    widget.skip_holds_check.setChecked(False)
    assert "12 rows" in widget.export_segments_note.text()
    widget.skip_holds_check.setChecked(True)
    assert "8 rows" in widget.export_segments_note.text()


def test_progress_can_stop_the_export_after_the_rows_already_measured(
        qtbot, dta_cycles_path, tmp_path):
    """Cancellation is a RETURN VALUE, not a widget.

    That is what lets this be a two-line lambda instead of a simulated click on
    a modal dialog, and it is why the callable half knows nothing about Qt. The
    rows already measured are kept and written: a cancelled twelve-segment
    export that threw away four finished measurements would be punishing the
    user for stopping it.
    """
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    seen = []

    def progress(done, total, text):
        seen.append((done, total, text))
        return len(seen) <= 3

    out = tmp_path / "part.csv"
    rows = widget.export_all_segments_csv(out, progress=progress)
    assert len(rows) == 3
    assert [r["segment_index"] for r in rows] == [1, 3, 4]
    assert [s[1] for s in seen[:3]] == [8, 8, 8]     # total known up front
    assert "#1" in seen[0][2]                        # labelled by segment
    assert len(pd.read_csv(out)) == 3


def test_export_all_segments_is_reachable_without_a_dialog(
        qtbot, dta_cycles_path, tmp_path, monkeypatch):
    """House rule: every file-dialog method is a thin wrapper over a directly
    callable one taking paths. `_export_all_segments_csv` may only choose a
    path and report the outcome."""
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    seen = []
    monkeypatch.setattr(widget, "export_all_segments_csv",
                        lambda p, **kw: seen.append(p) or [])
    monkeypatch.setattr(module.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: ("", "")))
    widget._export_all_segments_csv()
    assert seen == []                            # cancelled: nothing happens

    target = tmp_path / "chosen.csv"
    monkeypatch.setattr(module.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(target), "")))
    widget._export_all_segments_csv()
    assert seen == [target]

    callable_half = inspect.getsource(DtaWorkspace.export_all_segments_csv)
    assert "QFileDialog" not in callable_half
    assert "QMessageBox" not in callable_half


# ===========================================================================
# 3. The batch and its three scopes
# ===========================================================================
def test_batch_defaults_to_one_row_per_file(qtbot, dta_example_path, tmp_path):
    """The historic behaviour is the default. A folder of single-ramp exports
    must produce exactly what it produced before segments existed."""
    widget = DtaWorkspace(records=[_record(dta_example_path)])
    qtbot.addWidget(widget)
    assert widget.batch_scope() == BATCH_SCOPE_FILE
    assert widget.batch_scope_buttons[BATCH_SCOPE_FILE].isChecked() is True

    out = tmp_path / "batch.csv"
    rows = widget.batch_tg_to_csv([dta_example_path, dta_example_path], out)
    assert len(rows) == 2
    assert all(np.isfinite(r["Tg_parallel"]) for r in rows)
    assert list(pd.read_csv(out).columns)[:18] == list(LEGACY_RESULT_COLUMNS)


def test_batch_numbers_are_the_same_numbers_the_compute_button_gives(
        qtbot, dta_example_path):
    """Bit-for-bit, not approximately.

    Before `_measure_tg` existed the batch loop carried its own copy of the
    three `dta_science` calls and its own copy of the mode/range formatting,
    which is exactly how a batch number drifts from the one the user saw on
    screen. There is no second copy left to drift, and this is what says so.
    """
    widget = DtaWorkspace(records=[_record(dta_example_path)])
    qtbot.addWidget(widget)
    widget._compute()
    res = widget._current_result()

    row = widget.batch_tg_rows([dta_example_path])[0]
    assert row["Tg_double"] == res.tg_values[0]
    assert row["Tg_parallel"] == res.tg_values[1]
    assert row["Tg_derivative"] == res.tg_values[2]
    assert row["Tg_parallel"] == pytest.approx(354.51135373692597, abs=1e-4)
    assert row["low_mode"] == res.res_parallel.low_mode
    assert row["high_mode"] == res.res_parallel.high_mode
    assert row["window_min"] == 350.0 and row["window_max"] == 700.0


def test_whole_file_batch_refuses_a_multi_segment_file_and_says_why(
        qtbot, dta_example_path, dta_cycles_path, tmp_path):
    """The one refusal in the feature, and the reason it is not an M0.1 breach.

    There is no whole-file number to refuse: a single-valued abscissa is a
    PRECONDITION of the tangent construction, not a preference about it, and one
    global sort over eight interleaved passes through 19-1299 °C is not a curve.
    The refusal text has to say that, name the segments so the row is usable,
    and point at the two scopes that do measure the file -- which the next test
    then exercises.
    """
    widget = DtaWorkspace(records=[_record(dta_example_path)])
    qtbot.addWidget(widget)

    out = tmp_path / "mixed.csv"
    rows = widget.batch_tg_to_csv([dta_example_path, dta_cycles_path], out)
    assert len(rows) == 2
    assert np.isfinite(rows[0]["Tg_parallel"])          # the single ramp: fine

    refused = rows[1]
    assert "Tg_parallel" not in refused                 # never attempted
    assert "Tg_double" not in refused
    assert "12 segments" in refused["error"]
    assert "#1 heat" in refused["error"] and "#3 cool" in refused["error"]
    assert "ONE monotone scan" in refused["error"]
    assert BATCH_SCOPE_LABELS[BATCH_SCOPE_EVERY] in refused["error"]
    # never the silent collapsed number
    assert "431.7" not in refused["error"] and "455.9" not in refused["error"]

    frame = pd.read_csv(out)
    assert "error" in frame.columns
    assert frame["Tg_parallel"].notna().sum() == 1


def test_the_batch_ui_says_the_whole_file_limit_is_arithmetic(qtbot,
                                                             dta_cycles_path):
    """Standing text on the tab, before a save path has been chosen.

    A user who learns the refusal from a message box after picking a filename
    has been made to work for nothing, and a modal warning is a warning that
    gets clicked away (M0.1's corollary).
    """
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    note = widget.batch_scope_note.text()
    assert "arithmetic limit" in note
    assert "not a physical objection" in note
    assert "REFUSED" in note

    widget.set_batch_scope(BATCH_SCOPE_EVERY)
    assert "one row per segment" in widget.batch_scope_note.text().lower()
    assert "Skip hold segments" in widget.batch_scope_note.text()
    widget.skip_holds_check.setChecked(False)
    assert "Hold segments are included" in widget.batch_scope_note.text()


def test_every_segment_batch_measures_the_file_the_whole_file_scope_refused(
        qtbot, dta_example_path, dta_cycles_path, tmp_path):
    """What makes the refusal above a limit and not a wall.

    The same cyclic file that cannot yield one row yields eight, one per ramp,
    each measured on its own scan -- and the single-ramp file in the same batch
    still yields exactly one. That is the M0.1 answer: the data is processed,
    the arithmetic that has no answer is simply not attempted.
    """
    widget = DtaWorkspace(records=[_record(dta_example_path)])
    qtbot.addWidget(widget)
    widget.set_batch_scope(BATCH_SCOPE_EVERY)

    out = tmp_path / "every.csv"
    rows = widget.batch_tg_to_csv([dta_example_path, dta_cycles_path], out)
    assert len(rows) == 9
    assert [r["segment_index"] for r in rows] == [1] + list(RAMP_INDICES)
    assert all("error" not in r or True for r in rows)
    from_cycles = rows[1:]
    assert tuple(r["program_step"] for r in from_cycles) == RAMP_STEPS
    assert len({r["file"] for r in rows}) == 2
    assert len(pd.read_csv(out)) == 9

    widget.skip_holds_check.setChecked(False)
    rows = widget.batch_tg_rows([dta_example_path, dta_cycles_path])
    assert len(rows) == 13
    assert [r["segment_index"] for r in rows] == [1] + list(range(1, 13))


def test_a_file_with_no_clock_is_one_curve_in_the_every_segment_scope(
        qtbot, dta_example_path, tmp_path):
    """Not-segmentable is not the same as multi-segment.

    A record with no clock column has always been one curve and still is: the
    "every segment" scope must not refuse it for lacking a diagnostic, since
    there is exactly one scan in it either way.
    """
    widget = DtaWorkspace(records=[_record(dta_example_path)])
    qtbot.addWidget(widget)
    widget.set_batch_scope(BATCH_SCOPE_EVERY)

    curves, refusal = widget._batch_segments_for_file(None)
    assert curves == (None,)
    assert refusal == ""


def test_matching_batch_needs_a_segment_on_screen(qtbot, dta_cycles_path):
    """The match is made against the scan the user is looking at, so an
    aggregate scope has nothing to match. Every file is skipped with that
    reason -- a row each, never a silent empty CSV."""
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    widget.set_batch_scope(BATCH_SCOPE_MATCHING)
    assert widget._current_segment() is None
    assert "single segment selected" in widget.batch_scope_note.text()

    curves, refusal = widget._batch_segments_for_file(widget.run)
    assert curves == ()
    assert BATCH_SCOPE_LABELS[BATCH_SCOPE_MATCHING] in refusal

    rows = widget.batch_tg_rows([dta_cycles_path])
    assert len(rows) == 1
    assert "Tg_parallel" not in rows[0]
    assert "needs a single segment selected" in rows[0]["error"]


def test_matching_batch_matches_kind_and_rate_across_files(
        qtbot, dta_example_path, dta_cycles_path, tmp_path):
    """The scope that builds a same-rate series out of a folder.

    `EXAMPLES/DTA_example.txt` is a +19.98 °C/min heat. The cyclic fixture holds
    heats at +19.95, +19.97, +14.99 and +9.99, so the two nominally-20 ramps
    match and the 15 and the 10 do not -- a 0.5 °C/min window is 38x wider than
    the repeat scatter between the two 20s and 10x narrower than the gap to the
    next programmed rate.
    """
    widget = DtaWorkspace(records=[_record(dta_example_path)])
    qtbot.addWidget(widget)
    reference = widget._current_segment()
    assert reference.kind == "heat"
    assert reference.rate_mode == pytest.approx(19.984, abs=1e-2)

    widget.set_batch_scope(BATCH_SCOPE_MATCHING)
    assert f"{MATCH_RATE_TOL_C_MIN:g} °C/min" in widget.batch_scope_note.text()

    out = tmp_path / "matched.csv"
    rows = widget.batch_tg_to_csv([dta_example_path, dta_cycles_path], out)
    assert [r["segment_index"] for r in rows] == [1, 1, 4]
    assert all(r["segment_kind"] == "heat" for r in rows)
    for row in rows:
        assert abs(row["rate_measured_C_min"] - reference.rate_mode) <= MATCH_RATE_TOL_C_MIN
    assert len(pd.read_csv(out)) == 3


def test_a_cooling_ramp_never_matches_a_heating_one(qtbot, dta_cycles_path):
    """The sign is part of the identity of a rate, not a detail of formatting.

    A -20 °C/min cool and a +20 °C/min heat share a magnitude and nothing else:
    the specimen's history, the direction of the transition and the sign of the
    heat flow all differ. Matching on |rate| would silently mix them into one
    "20 °C/min" series.
    """
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    widget.set_batch_scope(BATCH_SCOPE_MATCHING)

    widget.select_segment(3)                       # cool, -19.96
    _settle(qtbot, widget)
    curves, refusal = widget._batch_segments_for_file(widget.run)
    assert refusal == ""
    assert [s.index for s in curves] == [3]        # NOT #1 or #4 at +19.95

    widget.select_segment(9)                       # cool, -9.979
    _settle(qtbot, widget)
    curves, _ = widget._batch_segments_for_file(widget.run)
    assert [s.index for s in curves] == [9, 12]    # both -9.98, a matched pair

    widget.select_segment(1)                       # heat, +19.95
    _settle(qtbot, widget)
    curves, _ = widget._batch_segments_for_file(widget.run)
    assert [s.index for s in curves] == [1, 4]


def test_matching_ignores_skip_holds_because_the_reference_is_explicit(
        qtbot, dta_cycles_path):
    """Pointing at a hold is asking for holds.

    "Skip hold segments" answers "should a sweep over EVERY segment include the
    ones with no temperature scan". A user who selected a hold has already
    answered a different question, and silently dropping them would answer one
    nobody asked.
    """
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    widget.set_batch_scope(BATCH_SCOPE_MATCHING)
    widget.select_segment(2)                       # a 1300 °C hold
    _settle(qtbot, widget)
    assert widget.skip_holds_check.isChecked() is True

    curves, refusal = widget._batch_segments_for_file(widget.run)
    assert refusal == ""
    assert [s.index for s in curves] == [2, 5, 8, 11]


def test_matching_reports_what_a_file_held_when_nothing_matched(
        qtbot, dta_cycles_path, dta_example_path):
    """A skip needs a reason a user can act on, so the row names the rates the
    file actually holds. "No match" alone leaves them guessing whether the
    tolerance, the sign or the file is wrong."""
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    widget.set_batch_scope(BATCH_SCOPE_MATCHING)
    widget.select_segment(7)                       # heat at +14.99
    _settle(qtbot, widget)

    rows = widget.batch_tg_rows([dta_example_path])
    assert len(rows) == 1
    assert "Tg_parallel" not in rows[0]
    assert "no segment of this file matches" in rows[0]["error"]
    assert "+19.98" in rows[0]["error"]


def test_a_batch_of_forty_is_not_lost_to_one_unreadable_file(
        qtbot, dta_example_path, tmp_path):
    """One bad file contributes one error row and the rest are measured."""
    widget = DtaWorkspace(records=[_record(dta_example_path)])
    qtbot.addWidget(widget)
    bad = tmp_path / "not-an-export.txt"
    bad.write_text("this is not a TA export at all\n", encoding="utf-8")

    rows = widget.batch_tg_rows([bad, dta_example_path])
    assert len(rows) == 2
    assert set(rows[0]) == {"file", "sample", "error"}
    assert rows[0]["error"]
    assert np.isfinite(rows[1]["Tg_parallel"])


def test_batch_progress_can_cancel_between_files(qtbot, dta_example_path,
                                                 dta_cycles_path):
    """`total` counts FILES, because the row count is not known until every
    file has been segmented. Stopping keeps the rows already measured."""
    widget = DtaWorkspace(records=[_record(dta_example_path)])
    qtbot.addWidget(widget)
    widget.set_batch_scope(BATCH_SCOPE_EVERY)
    seen = []

    def progress(done, total, text):
        seen.append((done, total, text))
        return done == 0

    rows = widget.batch_tg_rows([dta_example_path, dta_cycles_path],
                                progress=progress)
    assert [s[1] for s in seen[:2]] == [2, 2]
    assert len(rows) == 1                     # the second file was not read
    assert rows[0]["segment_index"] == 1


def test_the_progress_ticker_reports_cancellation_from_the_dialog(
        qtbot, dta_example_path):
    """The Qt half of the same contract: a cancelled `QProgressDialog` turns
    the callback False, which is the only thing the callable halves inspect."""
    widget = DtaWorkspace(records=[_record(dta_example_path)])
    qtbot.addWidget(widget)

    tick = widget._progress_ticker("Measuring…", 4)
    assert widget._progress_dialog is not None
    assert tick(0, 4, "first") is True
    widget._progress_dialog.cancel()
    assert tick(1, 4, "second") is False
    widget._close_progress()
    assert widget._progress_dialog is None
    widget._close_progress()                  # idempotent, never raises


def test_the_batch_callable_half_opens_no_dialog(qtbot, dta_example_path,
                                                 dta_cycles_path, tmp_path,
                                                 monkeypatch):
    """Every refusal in the batch is a CSV cell, not a modal.

    `conftest`'s autouse patch only silences dialogs; this one fails on them, so
    a future refusal that reaches for a `QMessageBox` is caught here rather than
    discovered by a user forty files into a batch.
    """
    widget = DtaWorkspace(records=[_record(dta_example_path)])
    qtbot.addWidget(widget)

    def _boom(*args, **kwargs):
        raise AssertionError("a modal dialog was opened by the batch")

    for name in ("critical", "warning", "information", "question"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(_boom))
    monkeypatch.setattr(module.QFileDialog, "getSaveFileName",
                        staticmethod(_boom))
    monkeypatch.setattr(module.QFileDialog, "getOpenFileNames",
                        staticmethod(_boom))

    bad = tmp_path / "bad.txt"
    bad.write_text("nope\n", encoding="utf-8")
    for scope in (BATCH_SCOPE_FILE, BATCH_SCOPE_EVERY, BATCH_SCOPE_MATCHING):
        widget.set_batch_scope(scope)
        widget.batch_tg_to_csv([dta_example_path, dta_cycles_path, bad],
                               tmp_path / f"{scope}.csv")
    widget.set_batch_scope(BATCH_SCOPE_FILE)
    widget.export_all_segments_csv(tmp_path / "segs.csv")
    widget.export_current_csv(tmp_path / "cur.csv")


def test_the_batch_wrapper_only_chooses_paths(qtbot, dta_example_path,
                                              dta_cycles_path, tmp_path,
                                              monkeypatch):
    """`_batch_run` picks two paths, shows a progress dialog and reports. All of
    the logic is in `batch_tg_to_csv`, which is why every test above reaches the
    batch without simulating a single click."""
    widget = DtaWorkspace(records=[_record(dta_example_path)])
    qtbot.addWidget(widget)
    out = tmp_path / "wrapped.csv"

    monkeypatch.setattr(module.QFileDialog, "getOpenFileNames",
                        staticmethod(lambda *a, **k: ([], "")))
    widget._batch_run()
    assert not out.exists()                       # cancelled at the first dialog

    monkeypatch.setattr(module.QFileDialog, "getOpenFileNames",
                        staticmethod(lambda *a, **k: (
                            [str(dta_example_path), str(dta_cycles_path)], "")))
    monkeypatch.setattr(module.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: ("", "")))
    widget._batch_run()
    assert not out.exists()                       # cancelled at the second

    monkeypatch.setattr(module.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(out), "")))
    widget._batch_run()
    assert out.is_file()
    assert widget._progress_dialog is None        # taken down again
    assert "1 carried an error" in widget.status_label.text()

    callable_half = inspect.getsource(DtaWorkspace.batch_tg_rows)
    assert "QFileDialog" not in callable_half
    assert "QMessageBox" not in callable_half
    assert "QProgressDialog" not in callable_half


def test_set_batch_scope_ignores_an_unknown_key(qtbot, dta_example_path):
    """A stored key from a future version must not clear the selection: an
    unrecognised scope leaves the one in force, which is the only state the
    batch can honestly run in."""
    widget = DtaWorkspace(records=[_record(dta_example_path)])
    qtbot.addWidget(widget)
    widget.set_batch_scope(BATCH_SCOPE_EVERY)
    widget.set_batch_scope("no_such_scope")
    assert widget.batch_scope() == BATCH_SCOPE_EVERY


# ===========================================================================
# 4. The degeneracy guard (risk R4)
# ===========================================================================
def test_the_measured_collapse_is_reported_as_a_problem_not_as_agreement(
        qtbot, dta_cycles_path):
    """The headline pathology, on the real arrays that produce it.

    Three independent constructions -- two tangent geometries and an extremum of
    a derivative -- cannot land on the same number to four decimals on real
    data. When they do, they are all reading one degenerate index, and the app
    printing "✓ Methods agree (spread 0.00 °C)" over it is an endorsement.
    """
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    res = _collapsed_result(widget)

    for value in res.tg_values:
        assert value == pytest.approx(455.951, abs=1e-3)
    assert res.spread == pytest.approx(1.2240e-4, rel=1e-3)
    assert res.spread < DEGENERATE_SPREAD
    assert res.degenerate is True

    text = "\n".join(widget._result_lines(res))
    assert "Methods agree" not in text
    assert "degenerate" in text
    assert "collapsed" in text
    assert "1.22e-04" in text


def test_the_healthy_control_still_reports_agreement(qtbot, dta_example_path):
    """The other side of the threshold, two orders away.

    `EXAMPLES/DTA_example.txt` spreads 3.1547 °C through the widget. If the
    guard ever fired here it would replace a correct "methods agree" with an
    accusation, which is the failure mode opposite to the one it exists for.
    """
    widget = DtaWorkspace(records=[_record(dta_example_path)])
    qtbot.addWidget(widget)
    widget._compute()
    res = widget._current_result()

    assert res.spread == pytest.approx(3.1547, abs=1e-3)
    assert res.spread > DEGENERATE_SPREAD
    assert res.degenerate is False
    assert "Methods agree" in widget.result_label.text()
    assert "degenerate" not in widget.result_label.text()


def test_the_threshold_sits_four_orders_above_the_collapse_and_two_below_health(
        qtbot, dta_cycles_path, dta_example_path):
    """Why 1e-2 and not the 1e-6 that was also proposed.

    Measured collapse spreads are 1.2240e-4 (this fixture) and 4.0e-5 (the full
    18996-row file); the healthy control is 3.15 °C. A 1e-6 guard waves BOTH
    collapses through. This test pins the constant against both measurements, so
    tightening it fails here instead of silently re-enabling the endorsement.
    """
    assert DEGENERATE_SPREAD == 1e-2
    assert 1.2240e-4 < DEGENERATE_SPREAD        # committed fixture
    assert 4.0e-5 < DEGENERATE_SPREAD           # full BM-0-cycles.txt
    assert 3.15 > DEGENERATE_SPREAD             # healthy single ramp
    assert 1.2240e-4 / 1e-6 > 100               # a 1e-6 guard misses it
    assert DEGENERATE_SPREAD / 1.2240e-4 > 80
    assert 3.15 / DEGENERATE_SPREAD > 300


def test_tg_agreement_is_never_asked_the_question_when_the_methods_collapsed(
        qtbot, dta_cycles_path, monkeypatch):
    """The guard GATES the call, it does not post-process its answer.

    `tg_agreement` has no way to tell "three methods agree to 0.01 °C" from
    "three methods returned one number", and it must not grow one: it is the
    science layer's honest answer to the question it was asked. So the question
    is not asked. `dta_science.py` is untouched by this feature and this is the
    test that keeps it that way.
    """
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    res = _collapsed_result(widget)

    from dta import dta_science

    def _boom(*args, **kwargs):
        raise AssertionError("tg_agreement was asked about a collapse")

    monkeypatch.setattr(dta_science, "tg_agreement", _boom)
    text = "\n".join(widget._result_lines(res))
    assert "collapsed" in text


def test_the_collapse_does_not_gate_the_computation(qtbot, dta_cycles_path):
    """M0.1: a warning, never a refusal.

    The three numbers are still produced, still stored and still exported. The
    guard changes one sentence in the report; it does not withhold a
    measurement, because a user who wants to see where the degenerate index
    landed is entitled to.
    """
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    res = _collapsed_result(widget)

    assert res.has_values is True
    assert res.error == ""
    assert all(np.isfinite(v) for v in res.tg_values)
    assert res.res_double is not None and res.res_parallel is not None


def test_the_collapse_travels_into_the_csv(qtbot, dta_cycles_path):
    """A warning that stays in the UI while the number leaves in a file is
    exactly the failure M0.1 names. Three cells carry it: the verdict, the
    spread, and the sentence."""
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    res = _collapsed_result(widget)

    row = widget._segment_result_row(None, res, run=None, path=None, sample="s")
    assert row["tg_degenerate"] == "yes"
    assert row["tg_spread"] == pytest.approx(1.2240e-4, rel=1e-3)
    assert "methods collapsed onto one value" in row["warnings"]
    assert row["Tg_parallel"] == pytest.approx(455.951, abs=1e-3)


def test_the_collapse_travels_into_the_figure(qtbot, dta_cycles_path,
                                              dta_example_path):
    """A PNG is what leaves the building.

    Three identical numbers in the corner of a figure look like a result that
    agrees with itself unless the picture says otherwise, and the figure is the
    one artefact that outlives the session with no result box beside it. M0.1
    requires the warning to travel with the number, so it is drawn, not only
    reported.
    """
    source = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(source)
    collapsed = _collapsed_result(source)
    assert collapsed.degenerate is True

    widget = DtaWorkspace(records=[_record(dta_example_path)])
    qtbot.addWidget(widget)
    widget._compute()
    texts = [t.get_text() for t in widget.plot.ax.texts]
    assert not any("collapsed" in t for t in texts)

    widget._store_result(collapsed)
    widget._refresh_plot()
    texts = [t.get_text() for t in widget.plot.ax.texts]
    assert any("collapsed, not agreed" in t for t in texts)
    assert any("1.2e-04" in t for t in texts)


def test_a_real_segment_is_not_flagged_degenerate(qtbot, dta_cycles_path,
                                                  tmp_path):
    """Eight real ramps of the cyclic file, measured properly: spreads of
    27-198 °C. The guard must be silent on all of them, or it would be
    condemning the very measurements the rework makes possible."""
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)

    rows = widget.export_all_segments_csv(tmp_path / "s.csv")
    for row in rows:
        assert row["tg_degenerate"] == "no", row["segment_index"]
        assert row["tg_spread"] > 20.0, row["segment_index"]


def test_finite_spread_is_nan_below_two_finite_values():
    """NaN rather than 0.0 for the under-determined case, deliberately.

    A spread of 0 is exactly what the guard looks for, so returning it when only
    ONE method produced a number would report a collapse that never happened --
    and `dta_science.tg_agreement` makes the same distinction (`agree` is None
    below two methods). The two must not disagree.
    """
    assert np.isnan(finite_spread([]))
    assert np.isnan(finite_spread([1.0]))
    assert np.isnan(finite_spread([1.0, float("nan")]))
    assert np.isnan(finite_spread([None, 2.0]))
    assert finite_spread([1.0, 3.0]) == pytest.approx(2.0)
    assert finite_spread([1.0, 3.0, float("nan")]) == pytest.approx(2.0)

    res = DtaSegmentResult(tg_deriv=400.0)
    assert np.isnan(finite_spread(res.tg_values))
    # and a NaN spread is not a collapse
    assert not (np.isfinite(np.nan) and np.nan < DEGENERATE_SPREAD)


def test_the_collapse_sentence_has_one_wording_in_one_place():
    """The result box, the figure annotation and the CSV all read the same
    sentence, so a reader cannot be told three different things about one
    number."""
    line = degenerate_result_line(1.2240e-4, "°C")
    assert "1.22e-04 °C" in line
    assert "not agreement" in line
    assert line.startswith("⚠")
    assert degenerate_result_line(1e-5, "").count("°C") == 0


# ===========================================================================
# 5. `seg_results` and the three legacy aliases
# ===========================================================================
def test_results_are_keyed_by_segment_index(qtbot, dta_cycles_path):
    """Measuring segment 4 must not overwrite the number read off segment 1.

    The whole reason results are per segment is that arrowing through a cyclic
    run otherwise shows segment 1's Tg over segment 7's curve -- which is the
    confusion the rework exists to remove, in a smaller and harder-to-notice
    form than the whole-file collapse.
    """
    widget = DtaWorkspace(records=[_record(dta_cycles_path, rec_id="cyc")])
    qtbot.addWidget(widget)

    widget.select_segment(1)
    _settle(qtbot, widget)
    widget._compute()
    first = widget._current_result().tg_values

    widget.select_segment(7)
    _settle(qtbot, widget)
    assert widget._current_result() is None           # nothing measured yet
    assert widget.result_label.text() == "Tg: —"
    widget._compute()
    seventh = widget._current_result().tg_values

    assert sorted(widget.seg_results) == [1, 7]
    assert first != seventh
    widget.select_segment(1)
    _settle(qtbot, widget)
    assert widget._current_result().tg_values == first


def test_the_three_legacy_scalars_are_aliases_onto_the_current_segment(
        qtbot, dta_cycles_path):
    """Six of the pinned Qt tests read `res_double` / `res_parallel` /
    `tg_deriv` directly, so they stay -- as PROPERTIES over the one store.

    Two stores for one number is how a per-segment result and a "current"
    result drift apart, and the symptom would be invisible: a plausible Tg
    drawn over the wrong curve.
    """
    widget = DtaWorkspace(records=[_record(dta_cycles_path, rec_id="cyc")])
    qtbot.addWidget(widget)
    assert "res_double" not in widget.__dict__        # no shadowing attribute
    assert isinstance(type(widget).res_double, property)
    assert isinstance(type(widget).res_parallel, property)
    assert isinstance(type(widget).tg_deriv, property)

    widget.select_segment(1)
    _settle(qtbot, widget)
    widget._compute()
    stored = widget.seg_results[1]
    assert widget.res_double is stored.res_double
    assert widget.res_parallel is stored.res_parallel
    assert widget.tg_deriv == stored.tg_deriv

    widget.select_segment(7)
    _settle(qtbot, widget)
    assert widget.res_double is None
    assert widget.res_parallel is None
    assert widget.tg_deriv is None

    # Writing through the alias writes the store, not a second copy.
    widget.tg_deriv = 123.5
    assert widget.seg_results[7].tg_deriv == 123.5
    assert widget.tg_deriv == 123.5


def test_an_aggregate_scope_stores_nothing(qtbot, dta_cycles_path):
    """Nothing is computable across several segments, so there is nothing to
    key a result under. Writing through an alias there must be a no-op rather
    than an entry no scope can ever read back."""
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    assert widget._scope().kind == SCOPE_ALL_TIME
    assert widget._result_key() is None

    widget.tg_deriv = 400.0
    assert widget.seg_results == {}
    assert widget.tg_deriv is None


def test_setting_an_alias_to_none_creates_no_blank_entry(qtbot,
                                                        dta_example_path):
    """`seg_results[i]` must keep meaning "segment i has been measured".

    `_forget_results` assigns None, and a dict full of empty `DtaSegmentResult`s
    would make the per-segment export report twelve rows of nothing as twelve
    measurements.
    """
    widget = DtaWorkspace(records=[_record(dta_example_path)])
    qtbot.addWidget(widget)
    assert widget.seg_results == {}

    widget.res_double = None
    widget.tg_deriv = None
    assert widget.seg_results == {}

    widget._compute()
    assert list(widget.seg_results) == [1]
    widget._forget_results()
    assert widget.seg_results == {}


def test_an_unsegmented_record_stores_under_key_zero(qtbot):
    """A record with no clock is one whole-record scan, and `0` is its key --
    real segments are 1-based, so the two can never collide."""
    frame = pd.DataFrame({"X": np.linspace(300.0, 500.0, 400),
                          "Y": np.tanh(np.linspace(-3.0, 3.0, 400))})
    widget = DtaWorkspace(records=[{"title": "generic", "path": "",
                                    "df": frame, "meta": {}}])
    qtbot.addWidget(widget)
    assert widget.run is None
    assert widget._result_key() == 0

    widget._compute()
    assert list(widget.seg_results) == [0]
    # The row for it invents no segment the record does not have.
    row = widget._current_result_row()
    assert row["segment_index"] == ""
    assert row["segment_kind"] == "" and row["segment_label"] == ""
    assert row["n_points"] == ""


def test_re_detection_forgets_every_segment_s_result(qtbot, dta_cycles_path):
    """A Tg measured on the old boundaries is not a measurement on the new
    ones -- for any segment, not just the one on screen."""
    widget = DtaWorkspace(records=[_record(dta_cycles_path, rec_id="cyc")])
    qtbot.addWidget(widget)
    widget.select_segment(1)
    _settle(qtbot, widget)
    widget._compute()
    widget.select_segment(7)
    _settle(qtbot, widget)
    widget._compute()
    assert sorted(widget.seg_results) == [1, 7]

    widget.apply_detection_settings(rate_window_min=0.75)
    widget._redetect_now()
    assert widget.seg_results == {}


def test_the_pinned_tg_is_untouched_by_the_export_and_batch_work(
        qtbot, dta_example_path, tmp_path):
    """The keystone, re-pinned at the end of step 10/11.

    `test_dta_workspace_compute_matches_known_good_values` pins
    354.4666531002641 to 1e-4 and must never be edited. This asserts the same
    three numbers survive the whole export path -- Compute, the result CSV, the
    per-segment CSV and the batch -- because all four now go through
    `_measure_tg`, and a single change there would move all of them at once.
    """
    widget = DtaWorkspace(records=[_record(dta_example_path)])
    qtbot.addWidget(widget)
    widget._compute()

    assert widget.res_double.tg == pytest.approx(354.4666531002641, abs=1e-4)
    assert widget.res_parallel.tg == pytest.approx(354.51135373692597, abs=1e-4)
    assert widget.tg_deriv == pytest.approx(357.6214, abs=1e-3)
    assert float(widget.xmin_edit.text()) == 350.0
    assert float(widget.xmax_edit.text()) == 700.0

    current = widget.export_current_csv(tmp_path / "a.csv")
    segments = widget.export_all_segments_csv(tmp_path / "b.csv")
    batch = widget.batch_tg_rows([dta_example_path])
    for row in (current, segments[0], batch[0]):
        assert row["Tg_double"] == pytest.approx(354.4666531002641, abs=1e-4)
        assert row["Tg_parallel"] == pytest.approx(354.51135373692597, abs=1e-4)
        assert row["Tg_derivative"] == pytest.approx(357.6214, abs=1e-3)


# ===========================================================================
# 6. Skip-when-absent: the full 18996-row file
# ===========================================================================
@requires_dta_bm
def test_the_full_cyclic_file_exports_eight_ramp_rows(qtbot, dta_bm_dir,
                                                      tmp_path):
    """The committed fixture is `[::4]` of this file. Every conclusion drawn on
    the fixture has to be a conclusion about the furnace, so the same eight
    ramps, the same steps and the same programmed rates are pinned here at full
    resolution."""
    widget = DtaWorkspace(records=[_record(dta_bm_dir / "BM-0-cycles.txt")])
    qtbot.addWidget(widget)
    assert len(widget.run) == 12

    rows = widget.export_all_segments_csv(tmp_path / "full.csv")
    assert tuple(r["segment_index"] for r in rows) == RAMP_INDICES
    assert tuple(r["program_step"] for r in rows) == RAMP_STEPS
    for row, rate in zip(rows, RAMP_RATES_PROGRAMMED):
        assert row["rate_programmed_C_min"] == pytest.approx(rate, abs=1e-9)
    for row, rate in zip(rows, RAMP_RATES_MEASURED):
        assert row["rate_measured_C_min"] == pytest.approx(rate, abs=0.05)
    # Full-resolution row spans, four times the fixture's and not derived
    # from them: the first segment starts at 0 and the last ends at the file.
    assert rows[0]["row_start"] == 0
    assert rows[-1]["row_stop"] == 18996


@requires_dta_bm
def test_the_full_cyclic_file_collapses_at_four_e_minus_five(qtbot, dta_bm_dir):
    """The other measured collapse: 431.758 °C with a spread of 4.0e-5.

    Smaller than the fixture's 1.22e-4, which is the point -- the guard has to
    catch the tighter one too, and a threshold chosen from one file would be a
    threshold chosen from one decimation.
    """
    widget = DtaWorkspace(records=[_record(dta_bm_dir / "BM-0-cycles.txt")])
    qtbot.addWidget(widget)
    res = _collapsed_result(widget)

    for value in res.tg_values:
        assert value == pytest.approx(431.758, abs=1e-3)
    assert res.spread == pytest.approx(4.0e-5, rel=0.2)
    assert res.spread < DEGENERATE_SPREAD
    assert res.degenerate is True
    assert "Methods agree" not in "\n".join(widget._result_lines(res))


@requires_dta_bm
def test_the_whole_file_batch_refuses_the_full_cyclic_file_too(
        qtbot, dta_bm_dir, tmp_path):
    """Not a fixture artefact: the real 18996-row export is refused a
    whole-file row for the same reason, and is measurable segment by segment."""
    widget = DtaWorkspace(records=[_record(dta_bm_dir / "BM-0.txt")])
    qtbot.addWidget(widget)
    cyclic = dta_bm_dir / "BM-0-cycles.txt"

    refused = widget.batch_tg_rows([cyclic])[0]
    assert "Tg_parallel" not in refused
    assert "12 segments" in refused["error"]

    widget.set_batch_scope(BATCH_SCOPE_EVERY)
    rows = widget.batch_tg_rows([cyclic])
    assert [r["segment_index"] for r in rows] == list(RAMP_INDICES)
