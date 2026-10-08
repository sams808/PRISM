"""The Rate series tab: the plot stack, the row table, the four add paths.

SPEC_rate_series §6 / §8 step 5 / §9.4, MERGED_DECISIONS M0.2 and M5. This is
the feature the user asked for by name — "add a feature to load data taken at
different rates to make the arrhenius and lasocka plots" — and the headline
test is `test_five_files_at_five_rates_become_five_rows_with_nothing_typed`:
five separate exports at 5, 10, 15, 20 and 30 °C/min become five rows whose
rates were read from each file's own programmed method and its own data.

Four things are load-bearing here and each is tested from the outside, through
the real widget:

  * **The plot stack costs the pinned tests nothing.** `self.plot` keeps its
    name, its identity and its type; it is reparented into a `QStackedWidget`
    at index 0 and nothing else. The 13 pinned Qt tests reach `widget.plot.ax`
    and `widget.plot.figure`.
  * **Adding never computes** (SPEC_rate_series §1 conflict 4). On the real
    cyclic data the second heating ramp has no glass transition left inside
    350-700 °C, so a compute-on-add would draw three filled rows, one blank,
    and a healthy-looking line through them before anyone had looked at a
    baseline. The test proves it by making every `dta_science` entry point
    raise and adding rows anyway.
  * **Nothing is hard-blocked** (MERGED_DECISIONS M0.1). A file that cannot be
    read becomes one visible failed row, not an exception and not a silently
    missing row; four cyclic ramps at three different rates arrive switched OFF
    with an ℹ line saying so, because the engine declines to GUESS which ramp
    was meant — not to load the file. No add path opens a modal dialog.
  * **Identity is the uid, never the row position** (Risk R3). Every mutation
    resolves through `Qt.UserRole` of column 0, every table rebuild runs with
    `itemChanged` blocked, and both are pinned — the re-entrancy test counts
    handler calls across a rebuild and requires zero.

Numbers pinned here were MEASURED on the committed fixtures, not copied from a
spec: the four heating rates of `DTA_cycles_example.txt` (19.9544 / 19.9653 /
14.9906 / 9.9863 °C/min), their original DataFrame spans, their cumulative mass
losses (0.2585 / 0.6678 / 0.8473 / 1.0298 %), and `DTA_example.txt`'s single
19.9844 °C/min ramp.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFileDialog, QMessageBox

from dta import qt_dta as module
from dta import dta_rate_series as drs
from dta.dta_segments import ROLE_COOL
from dta.qt_dta import (
    PLOT_PAGE_MAIN, PLOT_PAGE_RATE, RATE_COL_FLAGS, RATE_COL_FROM,
    RATE_COL_HISTORY, RATE_COL_MASS, RATE_COL_NOTE, RATE_COL_QUANTITY,
    RATE_COL_RATE, RATE_COL_SAMPLE, RATE_COL_SEGMENT, RATE_COL_SOURCE,
    RATE_COL_USE, RATE_COL_VALUE, RATE_DEFAULT_WINDOW_C, RATE_EDITABLE_COLUMNS,
    RATE_TAB_TITLE, RATE_TABLE_COLUMNS, DtaWorkspace, rate_history_text,
    rate_note_display, rate_quantity_label,
)
from core.qt_widgets import PlotWidget
from core.qt_shell import _load_spectrum_from_path


# The four heating ramps of EXAMPLES/DTA_cycles_example.txt, measured.
CYCLE_LABELS = ("heat 1 (cycle 1)", "heat 2 (cycle 2)",
                "heat 3 (cycle 3)", "heat 4 (cycle 4)")
CYCLE_SEGMENTS = (1, 4, 7, 10)
CYCLE_RATES = (19.9544, 19.9653, 14.9906, 9.9863)
CYCLE_PROGRAMMED = (20.0, 20.0, 15.0, 10.0)
CYCLE_SPANS = ((0, 346), (903, 1226), (1892, 2328), (3212, 3873))
CYCLE_MASS_LOSS = (0.2585, 0.6678, 0.8473, 1.0298)
CYCLE_STATES = ("virgin", "after-cycle-1", "after-cycle-2", "after-cycle-3")

# The single ramp of EXAMPLES/DTA_example.txt.
EXAMPLE_RATE = 19.9844

# A minimal but genuine TA SDT export: the same key order, the same tab
# separators, the same "OrgMethod" step line and the same UTF-8-BOM + CRLF
# encoding as the committed fixtures, so `parse_ta_sdt_txt`,
# `program_from_header` and `segment_run` all do their real work on it. Written
# rather than committed because the whole point is to vary the RATE across five
# files, which is exactly what no single committed fixture can be.
_SYNTHETIC_HEADER = """CLOSED
Version\t2.0
Instrument\tSDT Q600 V20.9 Build 20
Module\tDSC-TGA Standard
Operator\tSS
Sample\t{sample}
Size\t{mass:.4f}\tmg
ProcName\tRamp
Method\tRamp
Exotherm\tUp
Nsig\t5
Sig1\tTime (min)
Sig2\tTemperature (°C)
Sig3\tWeight (mg)
Sig4\tHeat Flow (mW)
Sig5\tTemperature Difference (°C)
Date\t2026-09-27
Time\t10:00:00
OrgMethod\t1: Ramp {rate:.2f} °C/min to {target:.2f} °C
StartOfData
"""


def _ramp_file(directory, rate, *, t_start_C=30.0, target_C=700.0,
               mass=20.0, dt=0.1):
    """One synthetic single-ramp export at `rate` °C/min. Returns its path."""
    path = directory / f"glass_{rate:g}Cmin.txt"
    n = int(round((target_C - t_start_C) / rate / dt)) + 1
    t = np.arange(n) * dt
    T = t_start_C + rate * t
    weight = mass - 0.01 * t / max(float(t[-1]), 1e-9)
    heat_flow = 5.0 + 0.001 * T
    dta = -1.5 + 0.0002 * T
    lines = [_SYNTHETIC_HEADER.format(sample=path.stem, mass=mass, rate=rate,
                                      target=target_C)]
    for i in range(n):
        lines.append(f"{t[i]:.6f}\t{T[i]:.5f}\t{weight[i]:.5f}\t"
                     f"{heat_flow[i]:.6f}\t{dta[i]:.6f}\n")
    path.write_bytes("".join(lines).replace("\n", "\r\n").encode("utf-8-sig"))
    return path


def _record(path, *, rec_id=None):
    """A library-shaped record dict. `rec_id=None` reproduces the hand-built
    records the pinned tests use, which carry no `"id"` at all."""
    spectrum = _load_spectrum_from_path(str(path))
    rec = {"title": spectrum.title, "path": spectrum.path,
           "df": spectrum.df, "meta": spectrum.meta}
    if rec_id is not None:
        rec["id"] = rec_id
    return rec


def _workspace(qtbot, path=None, *, rec_id=None):
    records = [] if path is None else [_record(path, rec_id=rec_id)]
    widget = DtaWorkspace(records=records)
    qtbot.addWidget(widget)
    return widget


def _tab_index(widget, title):
    return [widget.tabs.tabText(i) for i in range(widget.tabs.count())].index(title)


def _cells(widget, table_row):
    table = widget.rate_table
    return [table.item(table_row, c).text() for c in range(table.columnCount())]


def _codes(widget):
    return [w.code for w in widget.rate_warnings]


# ===========================================================================
# Construction and placement — the plot stack must cost the pinned tests
# nothing
# ===========================================================================

def test_the_tab_order_is_the_merged_decision(qtbot, dta_example_path):
    """M5 spells the order out: Tg | Program | Calculs | Rate series |
    Export / Batch. Pinned as a whole tuple rather than "Rate series is at
    index 3", because the thing that matters is that the three existing tabs
    did not move."""
    widget = _workspace(qtbot, dta_example_path)
    titles = tuple(widget.tabs.tabText(i) for i in range(widget.tabs.count()))
    assert titles == ("Tg", "Program", "Calculs", "Rate series", "Export / Batch")
    assert widget.tabs.widget(3) is widget._rate_tab


def test_the_tg_canvas_is_page_zero_and_keeps_its_name_and_identity(qtbot, dta_example_path):
    """`self.plot` is REPARENTED, not replaced. The 13 pinned Qt tests reach
    `widget.plot.ax` and `widget.plot.figure`; if the stack had wrapped them in
    a new object, or moved them to page 1, they would all fail at once."""
    widget = _workspace(qtbot, dta_example_path)
    assert widget.plot_stack.count() == 2
    assert widget.plot_stack.widget(PLOT_PAGE_MAIN) is widget.plot
    assert widget.plot_stack.widget(PLOT_PAGE_RATE) is widget.rate_plot
    assert isinstance(widget.plot, PlotWidget) and isinstance(widget.rate_plot, PlotWidget)
    # Two canvases, not two views of one: an Arrhenius fit and a Tg overlay
    # share no axes, and drawing one must not clear the other.
    assert widget.rate_plot is not widget.plot
    assert widget.rate_plot.figure is not widget.plot.figure
    assert widget.rate_plot.ax is not widget.plot.ax
    assert widget.plot.ax.figure is widget.plot.figure


def test_tab_switch_shows_the_rate_canvas_and_switches_back(qtbot, dta_example_path):
    widget = _workspace(qtbot, dta_example_path)
    assert widget.plot_stack.currentIndex() == PLOT_PAGE_MAIN
    widget.tabs.setCurrentIndex(_tab_index(widget, RATE_TAB_TITLE))
    assert widget.plot_stack.currentIndex() == PLOT_PAGE_RATE
    widget.tabs.setCurrentIndex(0)
    assert widget.plot_stack.currentIndex() == PLOT_PAGE_MAIN
    widget.tabs.setCurrentIndex(_tab_index(widget, "Export / Batch"))
    assert widget.plot_stack.currentIndex() == PLOT_PAGE_MAIN


def test_the_rate_tab_is_live_on_a_workspace_with_no_records(qtbot):
    """`test_dta_workspace_constructs_standalone` pins that a bare workspace
    builds. The rate tab must therefore read no column list and no dataframe
    while it is being constructed, and its add path must answer rather than
    raise when asked with nothing loaded."""
    widget = _workspace(qtbot)
    assert widget.rate_table.columnCount() == len(RATE_TABLE_COLUMNS)
    assert widget.rate_table.isSortingEnabled() is False
    assert widget.rate_rows == []
    assert widget.rate_add_segments() == 0
    assert "No record" in widget.rate_rows_note.text()
    assert widget.rate_warn_banner.objectName() == "WarningBannerOk"
    assert "No rows yet" in widget.rate_warn_banner.text()


def test_the_rate_tab_switch_leaves_the_tg_result_and_its_figure_alone(qtbot, dta_example_path):
    """The guard for the whole stack idea: computing a Tg, visiting the rate
    tab and coming back must leave both the number and the drawn overlay in
    place. A shared canvas would have cleared the axes."""
    widget = _workspace(qtbot, dta_example_path)
    widget._compute()
    tg = widget.res_parallel.tg
    texts_before = [t.get_text() for t in widget.plot.ax.texts]
    widget.tabs.setCurrentIndex(_tab_index(widget, RATE_TAB_TITLE))
    widget.rate_add_segments()
    widget.tabs.setCurrentIndex(0)
    assert widget.res_parallel.tg == tg
    assert [t.get_text() for t in widget.plot.ax.texts] == texts_before
    assert len(widget.plot.ax.lines) > 0


# ===========================================================================
# Add path (a): the segments of the record on screen
# ===========================================================================

def test_rate_add_segments_makes_one_row_per_heating_ramp_with_rates_filled(
        qtbot, dta_cycles_path):
    """The cyclic case: four heating ramps, four rows, every rate already
    there. Both rate candidates are kept — measured AND programmed — because
    `rate_mismatch` compares them, and a builder that stored one in both fields
    would silence that check permanently (SPEC_rate_series R1)."""
    widget = _workspace(qtbot, dta_cycles_path, rec_id="cyc-1")
    assert widget.rate_add_segments() == 4
    rows = widget.rate_rows
    assert [r.segment_label for r in rows] == list(CYCLE_LABELS)
    assert [r.segment_index for r in rows] == list(CYCLE_SEGMENTS)
    for row, q, prog in zip(rows, CYCLE_RATES, CYCLE_PROGRAMMED):
        assert row.q_measured_C_per_min == pytest.approx(q, abs=1e-3)
        assert row.q_programmed_C_per_min == pytest.approx(prog, abs=1e-9)
        assert row.q_C_per_min == pytest.approx(q, abs=1e-3)
        assert row.q_used_source == "measured"
    # Provenance: the ORIGINAL DataFrame spans, from
    # `SegmentedRun.row_indices`, never `Segment.row_start` (M3).
    assert [(r.row_start, r.row_stop) for r in rows] == list(CYCLE_SPANS)
    assert [r.record_id for r in rows] == ["cyc-1"] * 4
    assert [r.source_kind for r in rows] == ["segment"] * 4
    assert all(r.n_heating_segments_in_source == 4 for r in rows)


def test_rate_add_segments_fills_the_thermal_history_and_the_mass_loss(
        qtbot, dta_cycles_path):
    """The walk that makes heat 1 and heat 3 distinguishable as two different
    materials: prior heats, prior maximum temperature and cumulative mass loss
    are properties of everything that happened BEFORE a row.

    The mass losses are pinned as measured (0.2585 -> 1.0298 %). They are read
    only across heating segments: every cooling segment of this run shows an
    apparent mass GAIN, which is buoyancy, and summing per-segment deltas would
    accumulate one buoyancy offset per cycle (MERGED_DECISIONS M1)."""
    widget = _workspace(qtbot, dta_cycles_path)
    widget.rate_add_segments()
    rows = widget.rate_rows
    assert [r.prior_heats for r in rows] == [0, 1, 2, 3]
    assert [r.state for r in rows] == list(CYCLE_STATES)
    assert [r.virgin for r in rows] == [True, False, False, False]
    assert not np.isfinite(rows[0].prior_max_T_C)
    for row in rows[1:]:
        assert row.prior_max_T_C == pytest.approx(1298.6, abs=0.5)
    for row, loss in zip(rows, CYCLE_MASS_LOSS):
        assert row.cum_mass_loss_pct == pytest.approx(loss, abs=1e-3)
    assert all(r.sample_mass_mg == pytest.approx(97.893, abs=1e-3) for r in rows)
    # The History and Δm cells are what a reader sees of all this.
    assert _cells(widget, 0)[RATE_COL_HISTORY] == "virgin"
    assert _cells(widget, 2)[RATE_COL_HISTORY] == "cycle 3 · 2 prior heats"
    assert _cells(widget, 3)[RATE_COL_MASS] == "1.030"


def test_rate_add_segments_never_computes_a_value(qtbot, dta_cycles_path, monkeypatch):
    """SPEC_rate_series §1 conflict 4, proved from the outside: every
    `dta_science` entry point raises, and adding four rows still works.

    The reason is on this very file. Its second heating ramp has no glass
    transition left inside 350-700 °C — after cycle 1 the specimen is largely
    crystallized — so a compute-on-add would fill three rows, blank the fourth
    and draw a healthy-looking line through the three before the user had
    inspected a single baseline."""
    def explode(*_a, **_k):
        raise AssertionError("adding a row must not measure anything")

    monkeypatch.setattr(module, "compute_tg_parallel_improved", explode)
    monkeypatch.setattr(module, "compute_tg_double_tangent", explode)
    monkeypatch.setattr(module, "compute_tg_derivative", explode)

    widget = _workspace(qtbot, dta_cycles_path)
    assert widget.rate_add_segments() == 4
    for row in widget.rate_rows:
        assert row.value_source == "—"
        assert not np.isfinite(row.value_C)
        assert row.error == ""
    assert [_cells(widget, i)[RATE_COL_VALUE] for i in range(4)] == ["", "", "", ""]
    assert [_cells(widget, i)[RATE_COL_FROM] for i in range(4)] == ["—"] * 4
    # And the note says so, permanently, on screen rather than in a dialog.
    assert "never computes a value" in widget.rate_rows_note.text()


def test_the_cyclic_rows_arrive_switched_on_with_an_information_line_saying_why(
        qtbot, dta_cycles_path):
    """Four ramps at three rates, added deliberately as SEGMENTS: all four on.

    These rows used to arrive switched off, on the reasoning that the engine
    should not guess which ramp was meant. That reasoning belongs to the FILE
    path (where the rate really is a guess) and is exactly wrong here: pressing
    "Add segments from current file" on a cyclic run is asking for its several
    rates, and switching them off left four good rows with nothing fitted.

    The banner stays in its QUIET tone, deliberately: "this run holds four
    heating ramps" is information about the data, not a fault in it, and painting
    it amber spends the colour a user has learnt to read as "something is wrong
    here". The genuine objection arrives separately, once values exist, as the
    thermal-history warning.
    """
    widget = _workspace(qtbot, dta_cycles_path)
    widget.rate_add_segments()
    assert [r.include for r in widget.rate_rows] == [True] * 4
    codes = _codes(widget)
    assert "multi_rate_source" in codes
    assert "ambiguous_file_rate" not in codes
    assert widget.rate_warn_banner.objectName() == "WarningBannerOk"
    text = widget.rate_warn_banner.text()
    assert "ℹ" in text and "rate series is for" in text
    assert "9.99" in text and "14.99" in text and "19.95" in text
    # Nothing is hidden: four rows in the table, each switched on and visible.
    assert widget.rate_table.rowCount() == 4
    for i in range(4):
        assert widget.rate_table.item(i, RATE_COL_USE).checkState() == Qt.Checked


def test_rate_add_segments_takes_the_cooling_ramps_when_asked_and_flags_them(
        qtbot, dta_cycles_path):
    """A cooling row is ADDED AND FLAGGED when asked for, never silently
    dropped and never silently kept. On this dataset the cooling exotherms give
    a physically impossible negative activation energy; MERGED_DECISIONS M1
    turns that from a refusal into a warning, and the flag is the first half of
    it."""
    widget = _workspace(qtbot, dta_cycles_path)
    assert widget.rate_add_segments(roles=(ROLE_COOL,)) == 4
    rows = widget.rate_rows
    assert [r.segment_index for r in rows] == [3, 6, 9, 12]
    assert all(r.q_C_per_min < 0 for r in rows)
    for i, row in enumerate(rows):
        assert "cooling" in row.flags
        assert "setpoint not reached" in row.flags
        assert "cooling" in _cells(widget, i)[RATE_COL_FLAGS]
    # The last cool is the run's truncated tail and says so.
    assert "partial" in rows[-1].flags
    assert rows[-1].is_partial is True


def test_an_unsegmentable_record_is_reported_and_adds_nothing(qtbot):
    """A record with no clock and no furnace temperature cannot be segmented —
    a rate is °C per MINUTE and substituting the row number for the clock would
    change what every detection window means. The answer is a sentence in the
    note, not an exception and not a dialog."""
    df = pd.DataFrame({"Signal A (mW)": np.linspace(0.0, 1.0, 50),
                       "Signal B (mW)": np.linspace(1.0, 2.0, 50)})
    widget = DtaWorkspace(records=[{"title": "no clock", "df": df, "meta": {}}])
    qtbot.addWidget(widget)
    assert widget.run is None
    assert widget.rate_add_segments() == 0
    assert widget.rate_rows == []
    assert "no detected segments" in widget.rate_rows_note.text()


# ===========================================================================
# Add path (b): separate files — the headline
# ===========================================================================

def test_five_files_at_five_rates_become_five_rows_with_nothing_typed(qtbot, tmp_path):
    """THE headline of this feature, end to end through the real widget.

    Five separate exports at 5, 10, 15, 20 and 30 °C/min — the textbook
    one-specimen-per-rate design — become five rows whose rates were read from
    each file's own programmed method and its own data. Nothing is typed, every
    row is switched ON (one rate per file, so there is nothing to disambiguate)
    and no value is computed."""
    paths = [_ramp_file(tmp_path, q) for q in (5.0, 10.0, 15.0, 20.0, 30.0)]
    widget = _workspace(qtbot)
    assert widget.rate_add_files(paths) == 5
    rows = widget.rate_rows
    assert [r.source_kind for r in rows] == ["file"] * 5
    for row, q in zip(rows, (5.0, 10.0, 15.0, 20.0, 30.0)):
        assert row.q_measured_C_per_min == pytest.approx(q, abs=1e-4)
        assert row.q_programmed_C_per_min == pytest.approx(q, abs=1e-9)
        assert row.q_C_per_min == pytest.approx(q, abs=1e-4)
        assert row.q_used_source == "measured"
        assert row.q_manual_C_per_min != row.q_manual_C_per_min   # NaN: not typed
        assert row.include is True
        assert row.value_source == "—"
        assert row.segment_label == "heat 1 (cycle 1)"
        assert row.n_heating_segments_in_source == 1
    assert [r.sample for r in rows] == [p.stem for p in paths]
    # Five distinct rates spanning 6x: the lever arm a Lasocka or Arrhenius fit
    # needs, present before a single value has been measured.
    assert len(widget.rate_series.distinct_rates()) == 5
    assert widget.rate_series.rate_span_ratio() == pytest.approx(6.0, abs=1e-3)
    assert not _codes(widget)
    assert widget.rate_warn_banner.objectName() == "WarningBannerOk"
    assert "5 rows, 5 distinct rates, 0 values" in widget.rate_warn_banner.text()


def test_an_unreadable_file_becomes_one_visible_failed_row(qtbot, tmp_path):
    """Never an exception (a batch of twenty files must not be lost to one bad
    file) and never a silently skipped file (a series that is quietly one row
    short is worse than one that says so). The verbatim reason travels in the
    row's own `error` field and into its cell."""
    bad = tmp_path / "not_a_dta_file.txt"
    bad.write_text("this is not a TA export at all\n", encoding="utf-8")
    widget = _workspace(qtbot)
    assert widget.rate_add_files([bad]) == 1
    row = widget.rate_rows[0]
    assert row.source_kind == "file"
    assert row.value_source == "failed"
    assert row.include is False
    assert "StartOfData" in row.error
    assert widget.rate_table.rowCount() == 1
    assert _cells(widget, 0)[RATE_COL_FROM] == "failed"
    assert row.error in _cells(widget, 0)[RATE_COL_NOTE]
    assert "no_heating_segment" in _codes(widget)
    assert widget.rate_warn_banner.objectName() == "WarningBannerCritical"


def test_one_bad_file_does_not_cost_the_good_ones(qtbot, tmp_path):
    """Six sources in, six rows out: five measured and one failed, in the order
    they were given, so "which of my files failed?" is answered by looking."""
    paths = [_ramp_file(tmp_path, q) for q in (5.0, 10.0, 15.0, 20.0, 30.0)]
    bad = tmp_path / "broken.txt"
    bad.write_text("nope\n", encoding="utf-8")
    widget = _workspace(qtbot)
    assert widget.rate_add_files(paths[:2] + [bad] + paths[2:]) == 6
    assert [r.value_source for r in widget.rate_rows] == [
        "—", "—", "failed", "—", "—", "—"]
    assert widget.rate_table.rowCount() == 6
    assert len(widget.rate_series.fitted_rows()) == 0     # no values yet
    assert len(widget.rate_series.failed_rows()) == 1


def test_the_failed_row_is_styled_as_a_failure_and_the_others_are_not(qtbot, tmp_path):
    good = _ramp_file(tmp_path, 10.0)
    bad = tmp_path / "broken.txt"
    bad.write_text("nope\n", encoding="utf-8")
    widget = _workspace(qtbot)
    widget.rate_add_files([good, bad])
    table = widget.rate_table
    for col in (RATE_COL_VALUE, RATE_COL_FROM):
        assert table.item(1, col).font().bold() is True
        assert table.item(1, col).foreground().color().name() == module._ROW_WARN_FG
        assert table.item(0, col).font().bold() is False
    # Excluded -> italic for the whole row; the good row is upright.
    assert table.item(1, RATE_COL_SAMPLE).font().italic() is True
    assert table.item(0, RATE_COL_SAMPLE).font().italic() is False
    # And a row that stops failing stops being red: items are REUSED, so the
    # styling has to be cleared, not merely set when it applies.
    widget.rate_set_value(widget.rate_rows[1].uid, 500.0)
    widget._rate_refresh_table()
    assert table.item(1, RATE_COL_FROM).font().bold() is False
    assert table.item(1, RATE_COL_FROM).foreground().color().name() != module._ROW_WARN_FG


def test_a_file_holding_several_ramps_adds_them_all_switched_off(qtbot, dta_cycles_path):
    """The cyclic file added AS A FILE behaves exactly as it does added as
    segments: four rows, all off, one ℹ line. Same builder, same rule — the
    file path is not a second code path."""
    widget = _workspace(qtbot)
    assert widget.rate_add_files([dta_cycles_path]) == 4
    assert [r.include for r in widget.rate_rows] == [False] * 4
    assert [r.source_kind for r in widget.rate_rows] == ["file"] * 4
    assert "ambiguous_file_rate" in _codes(widget)
    rates = [r.q_C_per_min for r in widget.rate_rows]
    assert rates == pytest.approx(list(CYCLE_RATES), abs=1e-3)


def test_add_files_is_callable_with_no_dialog_and_the_wrapper_is_a_thin_shell(
        qtbot, tmp_path, monkeypatch):
    """Every file-dialog method is a thin wrapper over a directly callable
    method that takes paths, so tests never monkeypatch a dialog to exercise
    logic. Here the dialogs are made to RAISE, and the public path still
    works."""
    paths = [_ramp_file(tmp_path, q) for q in (10.0, 20.0)]
    widget = _workspace(qtbot)

    def forbidden(*_a, **_k):
        raise AssertionError("no dialog may be opened by the public add path")

    monkeypatch.setattr(QFileDialog, "getOpenFileNames", staticmethod(forbidden))
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(forbidden))
    for name in ("critical", "warning", "information", "question"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(forbidden))
    assert widget.rate_add_files(paths) == 2

    # The wrapper: nothing but "get the paths -> return if empty -> call the
    # public method".
    monkeypatch.setattr(QFileDialog, "getOpenFileNames",
                        staticmethod(lambda *a, **k: ([], "")))
    widget._on_rate_add_files()
    assert len(widget.rate_rows) == 2
    monkeypatch.setattr(QFileDialog, "getOpenFileNames",
                        staticmethod(lambda *a, **k: ([str(paths[0])], "")))
    widget._on_rate_add_files()
    assert len(widget.rate_rows) == 3


def test_no_add_path_raises_a_modal_dialog_even_when_everything_fails(
        qtbot, dta_cycles_path, tmp_path, monkeypatch):
    """M0.1: warnings must be impossible to miss and must never be modal. A
    dialog is the surface a user learns to dismiss without reading, and this
    feature has none — every add path, including the ones that fail, is driven
    here with all four QMessageBox statics set to raise."""
    def forbidden(*_a, **_k):
        raise AssertionError("this feature raises no modal dialog")

    for name in ("critical", "warning", "information", "question"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(forbidden))
    bad = tmp_path / "broken.txt"
    bad.write_text("nope\n", encoding="utf-8")

    widget = _workspace(qtbot, dta_cycles_path, rec_id="cyc-1")
    widget.rate_add_segments()
    widget.rate_add_files([bad, _ramp_file(tmp_path, 10.0)])
    widget.rate_add_library_records(["cyc-1", "a record that is not there"])
    widget.rate_add_manual_row(sample="literature", q_C_per_min=10.0, value_C=612.0)
    widget.rate_add_segments(roles=(ROLE_COOL,))
    widget.rate_clear()
    assert widget.rate_rows == []


# ===========================================================================
# Add path (c): library records
# ===========================================================================

def test_library_records_add_by_display_key_or_by_spectrum_id(qtbot, dta_example_path):
    """The combo hands back display keys; a script, a test and a project file
    hand back ids. Both resolve, and the row carries the bare `Spectrum.id` —
    not the prefixed cache key, and not the title, which two imports of one
    file share."""
    widget = _workspace(qtbot, dta_example_path, rec_id="rec-99")
    keys = list(widget._record_lookup.keys())
    assert widget.rate_add_library_records(keys) == 1
    assert widget.rate_add_library_records(["rec-99"]) == 1
    rows = widget.rate_rows
    assert [r.source_kind for r in rows] == ["library", "library"]
    assert [r.record_id for r in rows] == ["rec-99", "rec-99"]
    assert rows[0].q_C_per_min == pytest.approx(EXAMPLE_RATE, abs=1e-3)
    assert rows[0].value_source == "—"
    # The frame already in memory is cached, so a compute does not re-read a
    # file the user has already opened.
    assert str(dta_example_path) in widget._rate_frames


def test_a_library_key_that_no_longer_resolves_adds_nothing_and_says_so(
        qtbot, dta_example_path):
    """No row, because there is no source to make a row about — but the gap is
    named in the note and in the status line rather than passed over."""
    widget = _workspace(qtbot, dta_example_path, rec_id="rec-99")
    assert widget.rate_add_library_records(["ghost-id"]) == 0
    assert widget.rate_rows == []
    assert "could not be found in the library" in widget.rate_rows_note.text()
    assert "could not be found in the library" in widget.status_label.text()


def test_the_library_check_list_follows_set_records(qtbot, dta_example_path,
                                                    dta_cycles_path):
    """The shell calls `set_records` whenever the DTA page becomes active —
    including when the Rate series tab is the one showing, where no tab change
    will ever fire. The list must not sit stale."""
    widget = _workspace(qtbot, dta_example_path, rec_id="a")
    assert len(widget.rate_library_combo._actions) == 1
    widget.set_records([_record(dta_example_path, rec_id="a"),
                        _record(dta_cycles_path, rec_id="b")])
    assert len(widget.rate_library_combo._actions) == 2


# ===========================================================================
# Add path (d): typed rows
# ===========================================================================

def test_rate_add_manual_row_returns_a_uid_and_appends_in_order(qtbot):
    widget = _workspace(qtbot)
    first = widget.rate_add_manual_row(sample="a", q_C_per_min=5.0, value_C=500.0)
    second = widget.rate_add_manual_row(sample="b", q_C_per_min=10.0, value_C=510.0)
    assert first != second
    assert [r.uid for r in widget.rate_rows] == [first, second]
    assert [r.sample for r in widget.rate_rows] == ["a", "b"]
    assert widget.rate_rows[0].source_kind == "manual"
    assert widget.rate_rows[0].value_source == "manual"
    assert widget.rate_rows[0].q_C_per_min == 5.0
    assert widget.rate_rows[0].q_used_source == "manual"
    assert widget._rate_row_by_uid(first) is widget.rate_rows[0]


def test_a_typed_row_claims_no_window_it_never_had(qtbot, dta_example_path):
    """Nobody knows what window a literature Tg was read under. A typed row
    that carried this session's window would make the consistency checks report
    a series that agrees with itself when it does not."""
    widget = _workspace(qtbot, dta_example_path)
    uid = widget.rate_add_manual_row(sample="from a paper", q_C_per_min=10.0,
                                     value_C=612.0)
    row = widget._rate_row_by_uid(uid)
    assert not np.isfinite(row.window_C[0]) and not np.isfinite(row.window_C[1])
    assert row.method == "manual"


# ===========================================================================
# The table is a pure view, keyed by uid (Risk R3)
# ===========================================================================

def test_the_table_is_a_pure_view_of_the_series(qtbot, dta_cycles_path):
    """Mutate the model directly, refresh, and every cell follows. The table
    holds no state of its own, which is what makes "the table disagrees with
    the fit" structurally impossible rather than merely unlikely."""
    widget = _workspace(qtbot, dta_cycles_path)
    widget.rate_add_segments()
    rows = widget.rate_rows
    rows[0].sample = "renamed by hand"
    rows[1].include = True
    rows[2].value_C = 621.5
    rows[2].value_source = "computed"
    rows[3].error = "Baseline range has too few points."
    rows[3].value_source = "failed"
    rows[3].note = "checked the baseline by eye"
    widget._rate_refresh_table()
    assert _cells(widget, 0)[RATE_COL_SAMPLE] == "renamed by hand"
    assert widget.rate_table.item(1, RATE_COL_USE).checkState() == Qt.Checked
    assert _cells(widget, 2)[RATE_COL_VALUE] == "621.5"
    assert _cells(widget, 2)[RATE_COL_FROM] == "computed"
    assert _cells(widget, 3)[RATE_COL_NOTE] == (
        "Baseline range has too few points. | checked the baseline by eye")
    for i, row in enumerate(rows):
        cells = _cells(widget, i)
        assert cells[RATE_COL_SOURCE] == row.source_kind
        assert cells[RATE_COL_SEGMENT] == row.segment_label
        assert cells[RATE_COL_QUANTITY] == rate_quantity_label(row)
        assert cells[RATE_COL_HISTORY] == rate_history_text(row)
        assert cells[RATE_COL_NOTE] == rate_note_display(row)


def test_refreshing_the_table_does_not_re_enter_the_item_handler(qtbot, dta_cycles_path):
    """Risk R3, pinned with a counter. Every `setText` in a rebuild would
    otherwise fire `itemChanged`, whose handler parses the cell it has just
    been handed back into the model — silently rewriting rates and values, and
    on a stale uid writing into the wrong row entirely."""
    widget = _workspace(qtbot, dta_cycles_path)
    widget.rate_add_segments()
    calls = []
    original = widget._rate_on_item_changed

    def counting(item):
        calls.append(item)
        return original(item)

    widget._rate_on_item_changed = counting
    widget.rate_table.itemChanged.disconnect()
    widget.rate_table.itemChanged.connect(counting)

    widget._rate_refresh_table()
    assert calls == []
    widget._rate_refresh_row(0)
    assert calls == []
    # The handler is still connected and still works: the guard blocks the
    # rebuild, not the user.
    widget.rate_table.item(0, RATE_COL_SAMPLE).setText("typed")
    assert len(calls) == 1
    assert widget.rate_rows[0].sample == "typed"


def test_every_mutation_is_resolved_by_uid_not_by_row_position(qtbot):
    """Remove the middle of three and then edit what moved up. A positional
    lookup would now write into the wrong row — which is exactly the bug that
    cannot be seen on screen, because the value lands in a cell that looks
    plausible."""
    widget = _workspace(qtbot)
    a = widget.rate_add_manual_row(sample="a", q_C_per_min=5.0, value_C=500.0)
    b = widget.rate_add_manual_row(sample="b", q_C_per_min=10.0, value_C=510.0)
    c = widget.rate_add_manual_row(sample="c", q_C_per_min=20.0, value_C=520.0)
    assert widget.rate_remove([b]) == 1
    assert [r.uid for r in widget.rate_rows] == [a, c]
    assert [_cells(widget, i)[RATE_COL_SAMPLE] for i in range(2)] == ["a", "c"]
    assert widget.rate_table.item(1, RATE_COL_USE).data(Qt.UserRole) == c
    widget.rate_table.item(1, RATE_COL_VALUE).setText("777")
    assert widget._rate_row_by_uid(c).value_C == 777.0
    assert widget._rate_row_by_uid(a).value_C == 500.0
    # Removing something that is not there is not an error and changes nothing.
    assert widget.rate_remove([b]) == 0
    assert len(widget.rate_rows) == 2


def test_sorting_is_disabled_and_the_order_is_insertion_order(qtbot):
    """A sorted table renumbers every row while the user is reading a warning
    that says "row 3", and Qt's sort moves items between rows, so an
    `itemChanged` arriving after a sort carries a row index that means
    nothing. Rows are added at 20, 5, 10 °C/min and must stay in that order."""
    widget = _workspace(qtbot)
    for sample, q in (("first", 20.0), ("second", 5.0), ("third", 10.0)):
        widget.rate_add_manual_row(sample=sample, q_C_per_min=q, value_C=600.0)
    assert widget.rate_table.isSortingEnabled() is False
    assert [_cells(widget, i)[RATE_COL_SAMPLE] for i in range(3)] == [
        "first", "second", "third"]
    assert [r.sample for r in widget.rate_rows] == ["first", "second", "third"]
    # `index_of` is 1-based and matches the screen, which is what "row 2" in a
    # warning means.
    assert widget.rate_series.index_of(widget.rate_rows[1].uid) == 2
    # The FIT sees them sorted by |rate|; the table never is.
    assert [r.sample for r in widget.rate_series.fitted_rows()] == [
        "second", "third", "first"]


def test_the_uid_is_in_user_role_of_column_zero_and_the_columns_are_the_spec(qtbot):
    widget = _workspace(qtbot)
    uid = widget.rate_add_manual_row(sample="a", q_C_per_min=5.0)
    item = widget.rate_table.item(0, RATE_COL_USE)
    assert item.data(Qt.UserRole) == uid
    assert widget._rate_uid_at_row(0) == uid
    headers = [widget.rate_table.horizontalHeaderItem(c).text()
               for c in range(widget.rate_table.columnCount())]
    assert headers == ["Use", "Sample", "Source", "Segment", "q (°C/min)",
                       "Quantity", "Value (°C)", "From", "History", "Δm (%)",
                       "Flags", "Note / error"]


def test_only_the_five_user_fields_are_editable(qtbot, dta_cycles_path):
    """Everything else is provenance: a measured rate, a segment label, a cycle
    number and a mass loss are facts about the source, and an editable copy of a
    measured number is a number nobody can trace."""
    widget = _workspace(qtbot, dta_cycles_path)
    widget.rate_add_segments()
    use = widget.rate_table.item(0, RATE_COL_USE)
    assert bool(use.flags() & Qt.ItemIsUserCheckable) is True
    assert bool(use.flags() & Qt.ItemIsEditable) is False
    for col in range(widget.rate_table.columnCount()):
        item = widget.rate_table.item(0, col)
        editable = bool(item.flags() & Qt.ItemIsEditable)
        assert editable == (col in RATE_EDITABLE_COLUMNS), col


def test_editing_the_note_of_a_failed_row_cannot_erase_the_error(qtbot, tmp_path):
    """`error` and `note` are separate fields and the cell shows both. A user
    who annotates a failed row must not be able to delete the sentence that
    says why it failed — it is the one thing the next reader needs in order to
    tell a bad specimen from a bad window."""
    bad = tmp_path / "broken.txt"
    bad.write_text("nope\n", encoding="utf-8")
    widget = _workspace(qtbot)
    widget.rate_add_files([bad])
    row = widget.rate_rows[0]
    error = row.error
    assert error

    cell = widget.rate_table.item(0, RATE_COL_NOTE)
    cell.setText(error + " | re-ran it, same result")
    assert widget.rate_rows[0].error == error
    assert widget.rate_rows[0].note == "re-ran it, same result"
    # Clearing the cell clears the NOTE and leaves the error standing.
    widget.rate_table.item(0, RATE_COL_NOTE).setText("")
    assert widget.rate_rows[0].note == ""
    assert widget.rate_rows[0].error == error
    assert error in _cells(widget, 0)[RATE_COL_NOTE]


def test_typing_a_rate_pins_it_and_leaves_the_measured_candidate_alone(
        qtbot, dta_cycles_path):
    """The typed number goes into `q_manual_C_per_min`; measured and programmed
    are untouched, so the tooltip and the CSV still show what the instrument
    reported next to what the user preferred. Overwriting the measured field
    would erase the evidence that they differed."""
    widget = _workspace(qtbot, dta_cycles_path)
    widget.rate_add_segments()
    row = widget.rate_rows[0]
    measured = row.q_measured_C_per_min

    widget.rate_table.item(0, RATE_COL_RATE).setText("21")
    assert row.q_manual_C_per_min == 21.0
    assert row.rate_source == "manual"
    assert row.q_C_per_min == 21.0
    assert row.q_used_source == "manual"
    assert row.q_measured_C_per_min == measured
    assert row.q_programmed_C_per_min == 20.0
    tip = widget.rate_table.item(0, RATE_COL_RATE).toolTip()
    assert "measured" in tip and "programmed" in tip and "used: manual" in tip

    # Clearing the cell drops the override and returns to the ladder.
    widget.rate_table.item(0, RATE_COL_RATE).setText("")
    assert not np.isfinite(row.q_manual_C_per_min)
    assert row.rate_source == "auto"
    assert row.q_C_per_min == pytest.approx(measured)
    assert row.q_used_source == "measured"


def test_pinning_a_candidate_that_does_not_exist_reports_what_was_really_used(qtbot):
    """`resolve_rate`'s honesty rule, through the widget: an explicit source
    that is non-finite does not silently become zero — the ladder is walked and
    the source ACTUALLY used is what the table and the export show."""
    widget = _workspace(qtbot)
    uid = widget.rate_add_manual_row(sample="typed", q_C_per_min=12.0)
    assert widget.rate_set_rate_source(uid, "programmed") is True
    row = widget._rate_row_by_uid(uid)
    assert row.rate_source == "programmed"
    assert row.q_C_per_min == 12.0
    assert row.q_used_source == "manual"
    assert widget.rate_set_rate_source(uid, "not a source") is False
    assert row.rate_source == "programmed"


def test_the_context_menu_actions_are_reachable_without_a_menu(qtbot, dta_cycles_path):
    """The menu wrapper does nothing but map a chosen QAction onto
    `rate_apply_row_action`, so every entry is exercisable from a test — and
    from a script — without a menu."""
    widget = _workspace(qtbot, dta_cycles_path)
    widget.rate_add_segments()
    uids = [r.uid for r in widget.rate_rows]
    assert widget.rate_apply_row_action(uids, "include") == 4
    assert all(r.include for r in widget.rate_rows)
    assert widget.rate_apply_row_action(uids[:2], "exclude") == 2
    assert [r.include for r in widget.rate_rows] == [False, False, True, True]
    assert widget.rate_apply_row_action(uids[:1], "programmed") == 1
    assert widget.rate_rows[0].rate_source == "programmed"
    assert widget.rate_rows[0].q_C_per_min == 20.0
    assert widget.rate_apply_row_action(uids[:1], "remove") == 1
    assert len(widget.rate_rows) == 3
    assert widget.rate_apply_row_action([], "remove") == 0
    # An unknown action changes nothing rather than guessing.
    assert widget.rate_apply_row_action([r.uid for r in widget.rate_rows], "wat") == 0


def test_clearing_forgets_the_rows_the_frames_and_their_warnings(qtbot, dta_cycles_path):
    widget = _workspace(qtbot, dta_cycles_path)
    widget.rate_add_segments()
    assert _codes(widget) and widget._rate_frames
    series_id = widget.rate_series.series_id
    assert widget.rate_clear() == 4
    assert widget.rate_rows == []
    assert widget._rate_frames == {}
    assert widget.rate_warnings == []
    assert widget.rate_table.rowCount() == 0
    assert widget.rate_warn_banner.objectName() == "WarningBannerOk"
    # The series keeps its identity: clearing rows is not starting a new series.
    assert widget.rate_series.series_id == series_id


def test_removing_a_row_removes_the_warning_that_was_about_it(qtbot, dta_cycles_path):
    """The banner is not a log of everything that ever happened. A warning
    whose every row is gone goes with them; one that still implicates a live row
    stays."""
    widget = _workspace(qtbot, dta_cycles_path)
    widget.rate_add_segments()
    assert "multi_rate_source" in _codes(widget)
    widget.rate_remove([r.uid for r in widget.rate_rows[:3]])
    assert "multi_rate_source" in _codes(widget)       # one row still implicated
    widget.rate_remove([widget.rate_rows[0].uid])
    assert _codes(widget) == []
    assert widget.rate_warn_banner.objectName() == "WarningBannerOk"


# ===========================================================================
# The °C / K switch (Risk R4)
# ===========================================================================

def test_the_value_column_follows_the_unit_switch_and_the_stored_value_does_not(
        qtbot, dta_example_path):
    """R4. A display switch relabels a header and re-renders a cell; it can
    never convert a stored number. `value_C` must be BIT-IDENTICAL after
    °C -> K -> °C, because a value converted twice still looks like a plausible
    Tg and makes an activation energy wrong by roughly a factor of three."""
    widget = _workspace(qtbot, dta_example_path)
    uid = widget.rate_add_manual_row(sample="a", q_C_per_min=10.0, value_C=354.4666531002641)
    stored = widget._rate_row_by_uid(uid).value_C

    def header():
        return widget.rate_table.horizontalHeaderItem(RATE_COL_VALUE).text()

    assert header() == "Value (°C)"
    assert _cells(widget, 0)[RATE_COL_VALUE] == "354.467"

    widget.temp_k_btn.setChecked(True)
    widget._rate_refresh_table()
    assert header() == "Value (K)"
    assert _cells(widget, 0)[RATE_COL_VALUE] == "627.617"
    assert widget._rate_row_by_uid(uid).value_C == stored

    widget.temp_c_btn.setChecked(True)
    widget._rate_refresh_table()
    assert header() == "Value (°C)"
    assert _cells(widget, 0)[RATE_COL_VALUE] == "354.467"
    assert widget._rate_row_by_uid(uid).value_C == stored


def test_a_value_typed_in_kelvin_is_stored_as_celsius(qtbot, dta_example_path):
    """The inverse of the display rule, and the only other place this feature
    converts a temperature: what is stored never depends on what was on
    screen."""
    widget = _workspace(qtbot, dta_example_path)
    uid = widget.rate_add_manual_row(sample="a", q_C_per_min=10.0)
    widget.temp_k_btn.setChecked(True)
    widget._rate_refresh_table()
    widget.rate_table.item(0, RATE_COL_VALUE).setText("900")
    row = widget._rate_row_by_uid(uid)
    assert row.value_C == pytest.approx(626.85, abs=1e-9)
    assert row.value_K == pytest.approx(900.0, abs=1e-9)
    assert row.value_source == "manual"
    widget.temp_c_btn.setChecked(True)
    widget._rate_refresh_table()
    assert _cells(widget, 0)[RATE_COL_VALUE] == "626.85"


def test_clearing_a_value_restores_not_measured_rather_than_a_manual_blank(qtbot):
    """"—" and a blank number are two different claims everywhere else in this
    feature — "nothing measured yet" against "measured, no value" — and they
    must not become the same thing in the one place a user can type."""
    widget = _workspace(qtbot)
    uid = widget.rate_add_manual_row(sample="a", q_C_per_min=10.0, value_C=600.0)
    assert widget.rate_rows[0].value_source == "manual"
    widget.rate_table.item(0, RATE_COL_VALUE).setText("")
    row = widget._rate_row_by_uid(uid)
    assert row.value_source == "—"
    assert not np.isfinite(row.value_C)
    assert _cells(widget, 0)[RATE_COL_FROM] == "—"


def test_clearing_a_value_restores_the_rows_own_construction_not_the_combo(
        qtbot, dta_cycles_path):
    """A cleared cell must not relabel the row's METHOD from whatever the combo
    happens to show now. The row was added as "Tg · parallel"; if the combo has
    since moved to "Tp · maximum", stamping the combo's method onto it would
    produce a "Tp · parallel tangent" pairing that no method in `dta_science`
    can make, purely because someone emptied a cell."""
    widget = _workspace(qtbot, dta_cycles_path)
    widget.rate_add_segments()
    uid = widget.rate_rows[0].uid
    widget.rate_set_value(uid, 620.8)
    assert widget._rate_row_by_uid(uid).method == "manual"

    widget.rate_quantity_combo.setCurrentIndex(3)          # Tp · maximum
    widget.rate_set_value(uid, float("nan"))
    row = widget._rate_row_by_uid(uid)
    assert row.value_source == "—"
    assert row.quantity_kind == "Tg"
    assert row.method == "parallel"                        # its own snapshot
    assert rate_quantity_label(row) == "Tg · parallel"

    # A typed row has no snapshot and no construction, so it stays "manual".
    typed = widget.rate_add_manual_row(sample="paper", q_C_per_min=10.0, value_C=600.0)
    widget.rate_set_value(typed, float("nan"))
    assert widget._rate_row_by_uid(typed).method == "manual"


def test_the_tp_window_fields_are_celsius_and_never_follow_the_switch(
        qtbot, dta_example_path):
    """R4 again, from the other side: a temperature FIELD that converts is a
    second Kelvin conversion site. These two say °C in their label and stay
    °C."""
    widget = _workspace(qtbot, dta_example_path)
    widget.rate_quantity_combo.setCurrentIndex(3)      # Tp · maximum
    assert widget._rate_quantity() == "Tp"
    assert widget.rate_tp_min_edit.isEnabled() is True
    assert widget._rate_window_C() == (350.0, 700.0)
    widget.temp_k_btn.setChecked(True)
    assert widget.rate_tp_min_edit.text() == "350"
    assert widget._rate_window_C() == (350.0, 700.0)
    widget.temp_c_btn.setChecked(True)
    assert widget._rate_window_C() == (350.0, 700.0)
    # Reversed entries are ordered, not rejected.
    widget.rate_tp_min_edit.setText("700")
    widget.rate_tp_max_edit.setText("350")
    assert widget._rate_window_C() == (350.0, 700.0)


def test_the_tp_widgets_are_only_live_for_tp(qtbot, dta_example_path):
    widget = _workspace(qtbot, dta_example_path)
    assert widget._rate_quantity() == "Tg"
    assert widget.rate_tp_signal_combo.isEnabled() is False
    widget.rate_quantity_combo.setCurrentIndex(3)
    assert widget.rate_tp_signal_combo.isEnabled() is True
    widget.rate_quantity_combo.setCurrentIndex(0)
    assert widget.rate_tp_signal_combo.isEnabled() is False


# ===========================================================================
# What a row records about the settings it was added under
# ===========================================================================

def test_rows_carry_the_settings_snapshot_under_the_csv_key_names(
        qtbot, dta_example_path):
    """Snapshotted onto each row rather than pointed at a shared store: the
    exported file must say which baseline mode and which window produced each
    number, months after the widgets have moved. The keys are
    `dta_rate_series.SETTINGS_CSV_KEYS`, so they become CSV columns with no
    translation layer to drift — checked here by driving the real exporter."""
    widget = _workspace(qtbot, dta_example_path)
    widget.manual_compute_check.setChecked(True)
    widget.low_min_edit.setText("340")
    widget.low_max_edit.setText("348")
    widget.high_use_point_check.setChecked(True)
    widget.high_point_edit.setText("380")
    widget.rate_add_segments()
    row = widget.rate_rows[0]
    assert set(row.settings) == set(drs.SETTINGS_CSV_KEYS)
    assert row.settings["tg_method"] == "parallel"
    assert row.settings["low_mode"] == "range"
    assert row.settings["low_range"] == "340..348"
    assert row.settings["high_mode"] == "point"
    assert row.settings["high_point_x"] == 380.0
    assert row.settings["y_col"] == widget.y_combo.currentText()
    assert row.settings["dy_source"] == widget.y_combo.currentText()
    assert row.settings["tp_window_min_C"] == 350.0

    record = drs.rows_table(widget.rate_series)[0]
    for key in drs.SETTINGS_CSV_KEYS:
        assert record[key] == row.settings[key], key
    assert record["row_index"] == 1
    assert record["value_C"] == ""            # blank, never 0 and never "nan"
    assert record["q_C_per_min"] == pytest.approx(EXAMPLE_RATE, abs=1e-3)


def test_the_recorded_window_is_celsius_even_while_the_display_shows_kelvin(
        qtbot, dta_example_path):
    """The Tg fields are in the DISPLAY unit; `window_C` is not. One conversion,
    through the single offset site in this file."""
    widget = _workspace(qtbot, dta_example_path)
    widget.temp_k_btn.setChecked(True)
    assert widget._get_window() == (623.15, 973.15)
    assert widget._rate_window_C() == pytest.approx((350.0, 700.0), abs=1e-9)
    widget.rate_add_segments()
    assert widget.rate_rows[0].window_C == pytest.approx((350.0, 700.0), abs=1e-9)


def test_a_clock_abscissa_records_the_default_window_not_minutes(
        qtbot, dta_cycles_path):
    """The normal case on a cyclic record: the whole-run view forces the clock
    onto X, so the Tg window fields hold 350-700 MINUTES. Copying those into a
    `window_min_C` cell would be a unit error inside an exported file, so the
    app default is recorded instead and the note says exactly that."""
    widget = _workspace(qtbot, dta_cycles_path)
    assert "Time" in widget.x_combo.currentText()
    assert widget._rate_window_is_default() is True
    assert widget._rate_window_C() == RATE_DEFAULT_WINDOW_C
    widget.rate_add_segments()
    assert all(r.window_C == RATE_DEFAULT_WINDOW_C for r in widget.rate_rows)
    note = widget.rate_settings_note.text()
    assert "not a temperature on this view" in note
    assert "350–700 °C" in note


def test_the_rows_note_says_what_add_segments_would_add(qtbot, dta_cycles_path):
    """Standing information, not a dialog: how many ramps this record has and
    at what rates, before anything is added."""
    widget = _workspace(qtbot, dta_cycles_path)
    note = widget.rate_rows_note.text()
    assert "4 heating ramps" in note
    assert "19.95" in note and "9.986" in note


# ===========================================================================
# The figure: every row visible, even without a value
# ===========================================================================

def test_every_row_is_on_the_figure_even_before_it_has_a_value(qtbot, tmp_path):
    """A row that cannot be seen is a row that gets forgotten in a fit it was
    never in. Five files at 5-30 °C/min show their lever arm as rug ticks
    before a single value exists; a failed row's tick is in the warning tone,
    a not-yet-computed one's is neutral — an unknown fact is not an alarm."""
    paths = [_ramp_file(tmp_path, q) for q in (5.0, 10.0, 15.0, 20.0, 30.0)]
    bad = tmp_path / "broken.txt"
    bad.write_text("nope\n", encoding="utf-8")
    widget = _workspace(qtbot)
    widget.rate_add_files(paths + [bad])
    ax = widget.rate_plot.ax
    rug = [ln for ln in ax.lines if ln.get_marker() == "|"]
    assert len(rug) == 5                      # the bad row has no rate to sit at
    assert all(ln.get_color() == "0.55" for ln in rug)
    assert [ln.get_xdata()[0] for ln in rug] == pytest.approx(
        [5.0, 10.0, 15.0, 20.0, 30.0], abs=1e-4)
    assert ax.get_xscale() == "log"
    # The row with no rate is named on the axes rather than dropped in silence.
    assert any("no rate yet" in t.get_text() for t in ax.texts)

    # Give two of them values: they become points, and the remaining three stay
    # as ticks.
    widget.rate_set_value(widget.rate_rows[0].uid, 600.0)
    widget.rate_set_value(widget.rate_rows[1].uid, 610.0)
    widget._rate_draw()
    ax = widget.rate_plot.ax
    points = [ln for ln in ax.lines if ln.get_marker() == "o"]
    assert len(points) == 1 and len(points[0].get_xdata()) == 2
    assert len([ln for ln in ax.lines if ln.get_marker() == "|"]) == 3

    # An excluded row with a value is greyed, never hidden.
    widget.rate_set_include(widget.rate_rows[0].uid, False)
    widget._rate_draw()
    crosses = [ln for ln in widget.rate_plot.ax.lines if ln.get_marker() == "x"]
    assert len(crosses) == 1
    assert crosses[0].get_xdata()[0] == pytest.approx(5.0, abs=1e-4)


def test_a_failed_row_is_marked_on_the_figure_in_the_warning_tone(qtbot, tmp_path):
    """A failed row has no y, so it cannot be a point — it is still visibly
    THERE, at its own rate, once it has one."""
    bad = tmp_path / "broken.txt"
    bad.write_text("nope\n", encoding="utf-8")
    widget = _workspace(qtbot)
    widget.rate_add_files([bad])
    widget.rate_set_rate(widget.rate_rows[0].uid, 10.0)
    widget._rate_draw()
    rug = [ln for ln in widget.rate_plot.ax.lines if ln.get_marker() == "|"]
    assert len(rug) == 1
    assert rug[0].get_color() == module._ROW_WARN_FG
    assert rug[0].get_xdata()[0] == pytest.approx(10.0)


def test_the_empty_figure_says_what_to_do_instead_of_nothing(qtbot):
    widget = _workspace(qtbot)
    widget._rate_draw()
    ax = widget.rate_plot.ax
    assert len(ax.lines) == 0
    assert any("No rows yet" in t.get_text() for t in ax.texts)
    assert "heating rate" in ax.get_xlabel()


def test_the_rate_figure_survives_a_pending_debounced_render(qtbot):
    """`PlotWidget.request_redraw` fires up to 120 ms later and would overwrite
    a direct draw, so `_rate_draw` cancels the queue first (the canvas's own
    docstring). Without the cancel this test loses the points 200 ms later."""
    widget = _workspace(qtbot)
    widget.rate_add_manual_row(sample="a", q_C_per_min=10.0, value_C=600.0)
    widget.rate_plot.request_redraw(lambda: widget.rate_plot.ax.clear())
    widget._rate_draw()
    qtbot.wait(200)
    points = [ln for ln in widget.rate_plot.ax.lines if ln.get_marker() == "o"]
    assert len(points) == 1


# ===========================================================================
# Cell rendering, widget-free
# ===========================================================================

def test_history_is_blank_when_nothing_is_known_rather_than_claiming_virgin():
    """No alarms — and no assertions — about unknown facts. A hand-typed
    literature value did not say the specimen was virgin, so the cell does not
    say it either."""
    typed = drs.manual_row(sample="from a paper", q_C_per_min=10.0, value_C=600.0)
    assert rate_history_text(typed) == ""
    virgin = drs.manual_row(sample="ours", q_C_per_min=10.0, value_C=600.0,
                            state="virgin")
    assert rate_history_text(virgin) == "virgin"
    reheated = drs.manual_row(sample="ours", q_C_per_min=10.0, value_C=600.0,
                              state="after-cycle-1", prior_heats=1)
    assert rate_history_text(reheated) == "1 prior heat"


def test_the_note_cell_shows_the_error_first_and_survives_a_round_trip():
    row = drs.manual_row(sample="x", q_C_per_min=10.0,
                         error="Baseline range has too few points.")
    assert rate_note_display(row) == "Baseline range has too few points."
    row.note = "looked at it by hand"
    assert rate_note_display(row) == (
        "Baseline range has too few points. | looked at it by hand")
    assert module._rate_note_from_cell(rate_note_display(row), row) == "looked at it by hand"
    assert module._rate_note_from_cell("only a note", row) == "only a note"
    assert module._rate_note_from_cell("", row) == ""


def test_the_quantity_cell_reads_the_row_not_the_combo():
    """A series assembled over two sittings can hold rows of both kinds, and
    `mixed_quantity` is the critical warning that says so. A cell showing the
    CURRENT combo would hide exactly that."""
    row = drs.manual_row(sample="x", q_C_per_min=10.0, value_C=600.0,
                         quantity_kind="Tp", method="peak_max")
    assert rate_quantity_label(row) == "Tp · peak_max"


# ===========================================================================
# The pinned Tg
# ===========================================================================

def test_the_pinned_tg_is_untouched_by_the_rate_series_work(qtbot, dta_example_path):
    """354.4666531002641 / 354.51135373692597 / 357.6214 — the values the
    original Tk TgGuiApp produced for this file. Re-measured here AFTER the plot
    stack, the new tab and a full rate series have been built and drawn, because
    the whole risk of this step is a construction-order change that moves the
    Tg tab's settings."""
    widget = _workspace(qtbot, dta_example_path)
    widget.tabs.setCurrentIndex(_tab_index(widget, RATE_TAB_TITLE))
    widget.rate_add_segments()
    widget.rate_add_manual_row(sample="typed", q_C_per_min=5.0, value_C=350.0)
    widget.tabs.setCurrentIndex(0)
    widget._compute()
    assert widget.res_double.tg == pytest.approx(354.4666531002641, abs=1e-4)
    assert widget.res_parallel.tg == pytest.approx(354.51135373692597, abs=1e-4)
    assert widget.tg_deriv == pytest.approx(357.6214, abs=1e-3)
    assert widget.xmin_edit.text() == "350"
    assert widget.xmax_edit.text() == "700"
