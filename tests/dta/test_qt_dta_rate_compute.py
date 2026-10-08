"""Rate series, step 6: computing the values, the fit, the figure, the banner.

SPEC_rate_series §6 / §8 step 6 / §9.4, MERGED_DECISIONS M0.1 and M4. Step 5
built the rows; this is where they acquire numbers and a fitted line, and the
whole design pressure is on ONE question: what happens to a row that cannot be
measured.

On the committed cyclic fixture that is not hypothetical. With the default
350-700 °C window the second heating ramp raises "Baseline range has too few
points." — after cycle 1 the specimen is largely crystallized, there is no glass
transition left in that window, and the derivative peak lands on the 700 °C edge
with one point above it (BM_FINDINGS §8). So the measured answer to
`rate_compute_all()` on this file is (3, 1), and every test here about
visibility is about that fourth row:

  * it keeps its place in the table, styled as a failure, with the verbatim
    message in its own cell;
  * the banner names it;
  * the annotation says "n = 3 of 4 rows" and "Not fitted: heat 2 (cycle 2)";
  * the figure puts a rug tick in the warning tone at its own rate.

A row that silently vanished from a four-point Lasocka fit would leave the user
believing they had fitted four points. Nothing here blocks: the fit is produced,
with `negative_lasocka_b` (critical) and eleven other objections attached.

Numbers pinned here were MEASURED through the real widget on the committed
fixtures, not copied from a spec. In particular the spec's illustrative
B = -14.2 came from a spike that read the DTA difference signal; the workspace's
own default Y column on this file is `Heat Flow (mW)`, and the values that
follow are what it produces.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from PySide6.QtWidgets import QMessageBox

from dta import qt_dta as module
from dta import dta_kinetics as dk
from dta import dta_rate_series as drs
from dta.dta_segments import ROLE_COOL
from dta.qt_dta import (
    RATE_ANNOTATION_MAX_WARNINGS, RATE_COL_FROM, RATE_COL_NOTE, RATE_COL_VALUE,
    RATE_MODEL_Q_OF_T, RATE_TAB_TITLE, DtaWorkspace, rate_method_error,
)
from core.qt_shell import _load_spectrum_from_path


# Measured through the widget on EXAMPLES/DTA_cycles_example.txt with the
# workspace's own defaults (Y = Heat Flow (mW), window 350-700 °C, parallel
# tangent). Heat 2 has no value: see the module docstring.
CYCLE_TG_C = (622.8145044552292, None, 676.270732638762, 660.278462658904)
CYCLE_RATES = (19.9544, 19.9653, 14.9906, 9.9863)
LASOCKA_B = -47.83387057956676
LASOCKA_A_C = 780.7114470674568
LASOCKA_R2 = 0.36776875966353173
MOYNIHAN_EA = -55.135449260263144

# The single ramp of EXAMPLES/DTA_example.txt, as a Tp-max on Heat Flow inside
# the default 350-700 °C Tp window.
EXAMPLE_TP_C = 350.1128


def _record(path, *, rec_id=None):
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


def _on_rate_tab(widget):
    widget.tabs.setCurrentIndex(_tab_index(widget, RATE_TAB_TITLE))
    return widget


def _cycles(qtbot, path, *, include_all=True):
    """The cyclic fixture with its four heating rows added and switched on.

    A segment pick arrives switched ON (`multi_rate_source`): asking for the
    segments of a cyclic run IS asking for its several rates. The explicit
    `rate_set_include` loop below is therefore a no-op on the default path and is
    kept only so a test that passes `include_all=False` has one place to differ.
    """
    widget = _on_rate_tab(_workspace(qtbot, path, rec_id="cyc-1"))
    widget.rate_add_segments()
    if include_all:
        for row in widget.rate_rows:
            widget.rate_set_include(row.uid, True)
        widget._rate_refresh_table()
        widget._rate_refit()
    return widget


def _codes(widget):
    return [w.code for w in widget.rate_warnings]


def _markers(widget, marker):
    return [ln for ln in widget.rate_plot.ax.lines if ln.get_marker() == marker]


def _fit_lines(widget):
    """The drawn fit curve(s): a solid line with no marker."""
    return [ln for ln in widget.rate_plot.ax.lines
            if ln.get_marker() in ("None", "", None) and ln.get_linestyle() == "-"]


def _texts(widget):
    return [t.get_text() for t in widget.rate_plot.ax.texts]


def _no_dialogs(monkeypatch):
    """Make every modal QMessageBox an immediate test failure.

    MERGED_DECISIONS M0.1: warnings must be impossible to miss and must never be
    a dialog, because a dialog is the surface a user learns to dismiss without
    reading. `QMessageBox` is allowed in this feature only for "nothing to
    export" and "no file loaded at all", neither of which any path here reaches.
    """
    def boom(*_a, **_k):
        raise AssertionError("a modal dialog was raised")

    for name in ("critical", "warning", "information", "question", "about"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(boom))


# ===========================================================================
# Compute all: the measured answer on the real cyclic file
# ===========================================================================

def test_compute_all_measures_every_row_with_one_shared_set_of_settings(
        qtbot, dta_cycles_path):
    """Four heating ramps in, three numbers and one honest failure out.

    Every row is measured under the SAME window, method, baselines and signal —
    the comparison between the rows is the measurement, so a series whose halves
    were measured under different settings is not a series. Each row records what
    it was measured under, which is what makes the exported file self-describing.
    """
    widget = _cycles(qtbot, dta_cycles_path)
    assert widget.rate_compute_all() == (3, 1)
    rows = widget.rate_rows
    assert [r.value_source for r in rows] == ["computed", "failed", "computed",
                                              "computed"]
    for row, expected in zip(rows, CYCLE_TG_C):
        if expected is None:
            assert not np.isfinite(row.value_C)
        else:
            assert row.value_C == pytest.approx(expected, abs=1e-4)
    assert {r.window_C for r in rows} == {(350.0, 700.0)}
    assert {r.method for r in rows} == {"parallel"}
    assert {r.quantity_kind for r in rows} == {"Tg"}
    assert {r.signal for r in rows} == {widget.y_combo.currentText()}


def test_the_row_that_cannot_be_measured_keeps_its_place_and_says_why(
        qtbot, dta_cycles_path):
    """The headline of this step. The failure this feature must not have is a row
    that disappears from the table and from the fit, leaving four rows' worth of
    confidence behind three points."""
    widget = _cycles(qtbot, dta_cycles_path)
    widget.rate_compute_all()
    bad = widget.rate_rows[1]
    assert bad.segment_label == "heat 2 (cycle 2)"
    assert bad.value_source == "failed"
    assert "too few points" in bad.error
    # Still there, in its own position, with the message in its own cell.
    assert widget.rate_table.rowCount() == 4
    assert widget.rate_table.item(1, RATE_COL_FROM).text() == "failed"
    assert bad.error in widget.rate_table.item(1, RATE_COL_NOTE).text()
    assert widget.rate_table.item(1, RATE_COL_VALUE).text() == ""
    # and styled as a failure rather than as an empty cell.
    for col in (RATE_COL_VALUE, RATE_COL_FROM):
        item = widget.rate_table.item(1, col)
        assert item.font().bold() is True
        assert item.foreground().color().name() == module._ROW_WARN_FG


def test_the_failed_row_is_named_in_the_banner_the_annotation_and_the_rug(
        qtbot, dta_cycles_path):
    """Four surfaces, none of which hides it (SPEC_rate_series §6).

    The rug tick is the one that matters most on a log axis: a failed row has no
    y and so cannot be a point, but it is still visibly THERE, at its own rate,
    next to the three rows that did produce numbers.
    """
    widget = _cycles(qtbot, dta_cycles_path)
    widget.rate_compute_all()
    bad = widget.rate_rows[1]

    # (1) the banner
    assert "no_value" in _codes(widget)
    assert "too few points" in widget.rate_warn_banner.text()
    # (2) the annotation
    assert "n = 3 of 4 rows" in widget.rate_annotation_text
    assert "Not fitted: heat 2 (cycle 2)" in widget.rate_annotation_text
    # (3) drawn on the figure, verbatim
    assert widget.rate_annotation_text in _texts(widget)
    # (4) a rug tick in the warning tone at its own rate
    rug = [ln for ln in _markers(widget, "|")
           if ln.get_color() == module._ROW_WARN_FG]
    assert len(rug) == 1
    assert rug[0].get_xdata()[0] == pytest.approx(abs(bad.q_C_per_min), abs=1e-4)
    # and it is NOT one of the three fitted points
    points = _markers(widget, "o")
    assert len(points) == 1 and len(points[0].get_xdata()) == 3


def test_a_rate_series_value_equals_a_tg_tab_compute_on_the_same_segment(
        qtbot, dta_cycles_path):
    """Bit-identical, not merely close (SPEC_rate_series §6).

    Both go through `_measure_tg` on arrays both got from
    `dta_segments.prepare_segment_xy`, so they are the same number by
    construction. Two independent measurement paths would agree today and drift
    the first time either one's column resolution changed.
    """
    widget = _cycles(qtbot, dta_cycles_path)
    widget.rate_compute_all()
    from_series = widget.rate_rows[0].value_C

    widget.tabs.setCurrentIndex(0)
    assert widget.select_segment(1) is True
    widget.xmin_edit.setText("350")
    widget.xmax_edit.setText("700")
    widget._compute()
    assert widget.res_parallel.tg == from_series


def test_compute_all_never_overwrites_a_typed_value(qtbot, dta_cycles_path):
    """A manual override wins and survives every later Compute all.

    The user overrode the computation on purpose. A button labelled "compute"
    quietly undoing that is what makes people keep their numbers in a
    spreadsheet — and the note keeps the computed value it replaced, so a
    reviewer months later can see both.
    """
    widget = _cycles(qtbot, dta_cycles_path)
    widget.rate_compute_all()
    uid = widget.rate_rows[0].uid
    widget.rate_set_value(uid, 599.0)
    row = widget._rate_row_by_uid(uid)
    assert (row.value_C, row.value_source, row.method) == (599.0, "manual", "manual")
    assert "computed was 622.8 °C" in row.note

    measured, failed = widget.rate_compute_all()
    assert (row.value_C, row.value_source) == (599.0, "manual")
    assert measured == 2 and failed == 1        # the other three, minus heat 2
    assert "1 typed value left untouched" in widget.status_label.text()


def test_retry_failed_touches_only_the_failed_rows(qtbot, dta_cycles_path):
    """"I widened the window, try that one again" — and nothing else.

    Re-measuring the whole series under settings chosen to rescue one row would
    quietly change three numbers the user had already read.
    """
    widget = _cycles(qtbot, dta_cycles_path)
    widget.rate_compute_all()
    before = [r.value_C for r in widget.rate_rows]

    # A window only heat 2 will be measured under. The Tg fields hold minutes
    # while the whole-run view is showing, so a segment is selected first — that
    # is `_rate_window_C`'s third case, and it is why it exists.
    widget.select_segment(4)
    widget.xmin_edit.setText("300")
    widget.xmax_edit.setText("660")
    assert widget._rate_window_C() == (300.0, 660.0)

    measured, failed = widget.rate_retry_failed()
    assert (measured, failed) == (0, 1)         # 660 °C still cuts the step off
    after = [r.value_C for r in widget.rate_rows]
    for i in (0, 2, 3):
        assert after[i] == before[i]
    assert [r.window_C for r in widget.rate_rows] == [
        (350.0, 700.0), (300.0, 660.0), (350.0, 700.0), (350.0, 700.0)]
    assert "Retried" in widget.status_label.text()


def test_a_typed_row_with_no_source_fails_loudly_instead_of_staying_blank(qtbot):
    """A blank typed row is not a row nobody tried; after Compute all it is a row
    that says there is nothing to measure. Left blank it looks like the third
    state ("not computed yet") and gets waited for."""
    widget = _on_rate_tab(_workspace(qtbot))
    uid = widget.rate_add_manual_row(sample="from a paper", q_C_per_min=10.0)
    assert widget.rate_compute_all() == (0, 1)
    row = widget._rate_row_by_uid(uid)
    assert row.value_source == "failed"
    assert "typed in" in row.error and "type its value" in row.error


def test_compute_all_records_what_it_measured_under_on_every_row(
        qtbot, dta_cycles_path):
    """Including the row that failed. A failure that does not say under which
    window it failed is a failure a user can only wonder about."""
    widget = _cycles(qtbot, dta_cycles_path)
    widget.manual_compute_check.setChecked(True)
    widget.low_min_edit.setText("360")
    widget.low_max_edit.setText("420")
    widget.rate_compute_all()
    for row in widget.rate_rows:
        assert set(row.settings) == set(drs.SETTINGS_CSV_KEYS)
        assert row.settings["low_mode"] == "range"
        assert row.settings["low_range"] == "360..420"
        assert row.settings["tg_method"] == "parallel"
        assert row.window_C == (350.0, 700.0)
    # and the provenance survives into the exported table, under those key names
    record = drs.rows_table(widget.rate_series)[1]
    assert record["low_range"] == "360..420"
    assert record["value_C"] == ""              # the failed row: blank, not 0
    assert "too few points" in str(record["error"])


def test_the_other_two_constructions_and_the_spread_travel_in_extras(
        qtbot, dta_cycles_path):
    """All three Tg constructions run even though one is wanted, because the
    spread is the only thing that tells a collapse from an agreement, and without
    it in the file there is no way to audit a row after the fact."""
    widget = _cycles(qtbot, dta_cycles_path)
    widget.rate_compute_all()
    row = widget.rate_rows[0]
    assert set(row.extras) == {"tg_double_C", "tg_parallel_C", "tg_deriv_C",
                               "spread_C"}
    assert row.extras["tg_parallel_C"] == pytest.approx(row.value_C, abs=1e-9)
    assert row.extras["spread_C"] == pytest.approx(27.1572, abs=1e-3)
    assert row.extras["spread_C"] > module.DEGENERATE_SPREAD
    # The failed row keeps the two constructions that DID produce a number.
    bad = widget.rate_rows[1]
    assert not np.isfinite(bad.extras["tg_parallel_C"])
    assert bad.extras["tg_double_C"] == pytest.approx(576.8749, abs=1e-3)


def test_the_chosen_constructions_own_message_becomes_the_rows_error():
    """`_measure_tg` joins all three failures; a row measured with one of them
    must quote that one. A parallel-tangent row showing the double tangent's
    complaint sends the user to adjust the wrong baseline."""
    joined = ("|dY| max: no peak found | Double: slope window empty | "
              "Parallel: Baseline range has too few points.")
    assert rate_method_error(joined, "parallel") == "Baseline range has too few points."
    assert rate_method_error(joined, "double") == "slope window empty"
    assert rate_method_error(joined, "deriv") == "no peak found"
    # No prefix match, and no error at all: the whole string, and nothing.
    assert rate_method_error("something else entirely", "parallel") == \
        "something else entirely"
    assert rate_method_error("", "parallel") == ""


def test_no_compute_or_warning_path_raises_a_modal_dialog(
        qtbot, dta_cycles_path, tmp_path, monkeypatch):
    """Every path that can produce a warning, driven with QMessageBox armed."""
    _no_dialogs(monkeypatch)
    bad = tmp_path / "broken.txt"
    bad.write_text("not a TA export\n", encoding="utf-8")
    widget = _cycles(qtbot, dta_cycles_path)
    widget.rate_add_files([bad])
    widget.rate_add_manual_row(sample="typed", q_C_per_min=10.0)
    widget.rate_compute_all()                    # failures, critical warnings
    widget.rate_retry_failed()
    widget.rate_set_model("kissinger")           # quantity mismatch
    widget.rate_set_model("lasocka")
    widget.rate_show_excluded_check.setChecked(False)   # hidden_excluded
    widget.rate_set_include(widget.rate_rows[0].uid, False)
    widget._rate_refit()
    widget.temp_k_btn.setChecked(True)
    widget.rate_clear()
    widget.rate_compute_all()                    # nothing to compute
    assert widget.rate_rows == []


def test_compute_all_on_an_empty_series_says_so_and_does_not_raise(qtbot):
    widget = _on_rate_tab(_workspace(qtbot))
    assert widget.rate_compute_all() == (0, 0)
    assert widget.rate_retry_failed() == (0, 0)
    assert "nothing" in widget.status_label.text()
    assert widget.rate_last_fit is None


# ===========================================================================
# The arrays (MERGED_DECISIONS M4, Risk R2)
# ===========================================================================

def test_the_arrays_are_the_rows_own_segment_in_prepared_row_order(
        qtbot, dta_cycles_path):
    """Risk R2, measured. The row's `row_start`/`row_stop` are ORIGINAL DataFrame
    positions, so the arrays hold exactly that slice — nothing dropped, nothing
    from the eleven other segments — and `prepare_segment_xy`'s stable ascending
    sort is the only thing that reorders it.

    On this fixture that sort moves exactly ONE point: the first two rows of
    heat 1 read 19.03 then 19.02 °C. That single reversal is the whole difference
    between the prepared arrays and the raw slice, which is what makes the
    identity claim testable rather than vague.
    """
    widget = _cycles(qtbot, dta_cycles_path)
    row = widget.rate_rows[0]
    x, y, y_dy, x_col = widget._rate_arrays_for(row, signal=row.signal,
                                                dy_sel="(same)")
    df = widget.df
    raw_x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(float)[
        row.row_start:row.row_stop]
    raw_y = pd.to_numeric(df[row.signal], errors="coerce").to_numpy(float)[
        row.row_start:row.row_stop]
    assert x.size == row.row_stop - row.row_start == raw_x.size
    order = np.argsort(raw_x, kind="stable")
    assert np.array_equal(x, raw_x[order])
    assert np.array_equal(y, raw_y[order])
    assert int(np.count_nonzero(np.diff(raw_x) <= 0)) == 1
    assert bool(np.all(np.diff(x) > 0))
    assert y_dy is not y and np.array_equal(y_dy, y)     # separate, never aliased
    # The scoping is what keeps this off the whole file: the record spans
    # 19.0-1298.6 °C over eight passes, and every per-segment Tg lands above 600.
    all_x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(float)
    assert float(np.nanmax(all_x)) == pytest.approx(1298.642, abs=1e-3)
    assert x.max() == pytest.approx(1292.173, abs=1e-3)
    widget.rate_compute_all()
    assert all(v > 600.0 for v in (r.value_C for r in widget.rate_rows)
               if np.isfinite(v))


def test_a_cooling_row_is_prepared_ascending_from_a_descending_ramp(
        qtbot, dta_cycles_path):
    """A cooling segment is measured, not refused (MERGED_DECISIONS M1): its
    temperature simply runs the other way, and `prepare_segment_xy` gives the
    tangent constructions the ascending abscissa they need."""
    widget = _on_rate_tab(_workspace(qtbot, dta_cycles_path, rec_id="cyc-1"))
    widget.rate_add_segments(roles=(ROLE_COOL,))
    cool = widget.rate_rows[0]
    x, _y, _dy, x_col = widget._rate_arrays_for(cool, signal=cool.signal,
                                               dy_sel="(same)")
    raw = pd.to_numeric(widget.df[x_col], errors="coerce").to_numpy(float)[
        cool.row_start:cool.row_stop]
    assert bool(np.all(np.diff(raw) < 0))       # the ramp descends
    assert bool(np.all(np.diff(x) > 0))         # the prepared abscissa ascends
    assert np.array_equal(x, raw[::-1])


def test_the_arrays_never_come_from_get_xy(qtbot, dta_cycles_path, monkeypatch):
    """Risk R2 as a guard rather than as a measurement.

    `_get_xy` is the convenient existing method and the wrong one: its abscissa
    is whatever the X combo shows and its scope is whatever the workspace is
    looking at, so on a cyclic record it hands back four interleaved heating
    ramps. Made to raise, every row must still be measured — and
    `prepare_segment_xy` must be the thing that was called instead.
    """
    widget = _cycles(qtbot, dta_cycles_path)
    calls = []
    real = module.prepare_segment_xy

    def counted(*a, **k):
        calls.append(k.get("segment"))
        return real(*a, **k)

    monkeypatch.setattr(module, "prepare_segment_xy", counted)
    monkeypatch.setattr(widget, "_get_xy",
                        lambda: (_ for _ in ()).throw(AssertionError("_get_xy")))
    assert widget.rate_compute_all() == (3, 1)
    assert len(calls) == 4
    assert [int(s.row_start) for s in calls] == [r.row_start
                                                 for r in widget.rate_rows]


def test_a_file_added_from_disk_is_read_once_and_then_cached(
        qtbot, dta_example_path):
    """`rate_add_files` deliberately leaves files on disk — a user may add twenty
    and compute two — so the first compute reads and caches. The second must not
    touch the disk again."""
    widget = _on_rate_tab(_workspace(qtbot))
    assert widget.rate_add_files([dta_example_path]) == 1
    assert widget._rate_frames == {}
    reads = []
    real = module.parse_ta_sdt_txt
    widget_module_parse = real

    def counted(path):
        reads.append(str(path))
        return widget_module_parse(path)

    module.parse_ta_sdt_txt = counted
    try:
        widget.rate_compute_all()
        assert len(reads) == 1
        assert str(dta_example_path) in widget._rate_frames
        widget.rate_compute_all()
        assert len(reads) == 1
    finally:
        module.parse_ta_sdt_txt = real


def test_a_row_whose_source_is_gone_fails_with_a_sentence_not_a_traceback(qtbot):
    """A library record that is no longer in memory and carried no path. The row
    stays, and its error says what to do about it."""
    widget = _on_rate_tab(_workspace(qtbot))
    uid = widget.rate_add_manual_row(sample="ghost", q_C_per_min=10.0)
    row = widget._rate_row_by_uid(uid)
    row.record_id = "rec-gone"
    row.row_start, row.row_stop = 0, 100
    assert widget.rate_compute_all() == (0, 1)
    assert "no longer in memory" in row.error
    assert widget.rate_table.rowCount() == 1


# ===========================================================================
# The model combo: built from the registry, gated by `needs`
# ===========================================================================

def test_the_model_combo_is_built_from_the_registry_in_registry_order(qtbot):
    """Six entries, the registry's own keys and labels, its own order. A
    hand-written list here would be a second opinion about which models exist."""
    widget = _on_rate_tab(_workspace(qtbot))
    combo = widget.rate_model_combo
    assert combo.count() == len(dk.KINETIC_METHODS)
    assert [combo.itemData(i) for i in range(combo.count())] == \
        list(dk.KINETIC_METHODS)
    assert [combo.itemText(i) for i in range(combo.count())] == \
        [m["label"] for m in dk.KINETIC_METHODS.values()]


def test_needs_gates_enablement_against_the_series_quantity(qtbot, dta_example_path):
    """SPEC_rate_series test 74. Kissinger is greyed on a Tg series and Lasocka
    on a Tp one, and the reason is in the tooltip — the decision is
    `evaluable_methods`', so the UI never re-decides which model suits which
    quantity."""
    widget = _on_rate_tab(_workspace(qtbot, dta_example_path))
    combo = widget.rate_model_combo

    def enabled():
        return {combo.itemData(i): combo.model().item(i).isEnabled()
                for i in range(combo.count())}

    assert widget.rate_series.quantity_kind == "Tg"
    assert enabled() == {"lasocka": True, "moynihan": True, "kissinger_tg": True,
                         "kissinger": False, "fwo": False, "augis_bennett": False}
    kissinger = list(dk.KINETIC_METHODS).index("kissinger")
    tip = combo.itemData(kissinger, module.Qt.ToolTipRole)
    assert "Defined for Tp" in tip and "this series holds Tg" in tip
    # augis_bennett is Tp AND needs a T0 on two rows; the tooltip names the
    # binding reason rather than both.
    widget.rate_quantity_combo.setCurrentIndex(3)        # Tp · maximum
    assert widget.rate_series.quantity_kind == "Tp"
    assert enabled() == {"lasocka": False, "moynihan": False,
                         "kissinger_tg": False, "kissinger": True, "fwo": True,
                         "augis_bennett": False}
    ab = list(dk.KINETIC_METHODS).index("augis_bennett")
    assert "T0" in combo.itemData(ab, module.Qt.ToolTipRole)
    assert widget._rate_model() == "kissinger"          # the Tp default


def test_a_model_the_series_does_not_satisfy_is_still_selectable_by_name(
        qtbot, dta_cycles_path):
    """M0.1 again: "Kissinger on Tg data" is a judgement about the experiment,
    not an arithmetic impossibility. Greying it in the combo is guidance; asking
    for it by name gets the number with `model_quantity_mismatch` attached."""
    widget = _cycles(qtbot, dta_cycles_path)
    widget.rate_compute_all()
    assert widget.rate_set_model("kissinger") is True
    assert widget._rate_model() == "kissinger"
    fit = widget.rate_last_fit
    assert fit.method == "kissinger"
    assert fit.ok is True
    assert "model_quantity_mismatch" in _codes(widget)
    assert widget.rate_set_model("not_a_model") is False
    assert widget._rate_model() == "kissinger"


def test_changing_the_model_refits_and_never_re_measures(qtbot, dta_cycles_path):
    """A Tg is a Tg whether it is plotted against ln q or 1/T. If the model could
    touch a row, the number in the table would depend on which fit was on screen
    when it was computed."""
    widget = _cycles(qtbot, dta_cycles_path)
    widget.rate_compute_all()
    before = [(r.value_C, r.value_source, r.method, r.window_C)
              for r in widget.rate_rows]
    assert widget.rate_last_fit.lasocka_B == pytest.approx(LASOCKA_B, rel=1e-9)

    widget.rate_set_model("moynihan")
    assert widget.rate_last_fit.method == "moynihan"
    assert widget.rate_last_fit.Ea_kJ_mol == pytest.approx(MOYNIHAN_EA, rel=1e-9)
    after = [(r.value_C, r.value_source, r.method, r.window_C)
             for r in widget.rate_rows]
    assert [b[1:] for b in before] == [a[1:] for a in after]
    for (bv, *_r), (av, *_s) in zip(before, after):
        assert (np.isnan(bv) and np.isnan(av)) or bv == av


def test_the_default_model_follows_the_quantity_only_while_the_series_is_empty(
        qtbot, dta_example_path):
    """An empty series has no measurement to protect. With rows in it the user's
    model choice is a choice, and quietly replacing it would change the number on
    screen without anybody asking."""
    widget = _on_rate_tab(_workspace(qtbot, dta_example_path))
    assert widget._rate_model() == "lasocka"
    widget.rate_quantity_combo.setCurrentIndex(3)        # Tp
    assert widget._rate_model() == "kissinger"
    widget.rate_quantity_combo.setCurrentIndex(0)        # back to Tg
    assert widget._rate_model() == "lasocka"

    widget.rate_add_manual_row(sample="a", q_C_per_min=10.0, value_C=600.0)
    widget.rate_set_model("moynihan")
    widget.rate_quantity_combo.setCurrentIndex(3)
    assert widget._rate_model() == "moynihan"            # untouched


# ===========================================================================
# The fit, the curve and the annotation
# ===========================================================================

def test_the_lasocka_fit_is_produced_with_its_objections_not_refused(
        qtbot, dta_cycles_path):
    """The whole of MERGED_DECISIONS M0.1 in one assertion pair: the number is
    handed over AND the twelve reasons to distrust it are attached. B is
    NEGATIVE, which is physically impossible for a rate dependence, and that is
    a critical warning rather than a refusal."""
    widget = _cycles(qtbot, dta_cycles_path)
    widget.rate_compute_all()
    fit = widget.rate_last_fit
    assert fit.ok is True
    assert fit.n == 3 and fit.dof == 1
    assert fit.lasocka_B == pytest.approx(LASOCKA_B, rel=1e-9)
    assert fit.lasocka_A_C == pytest.approx(LASOCKA_A_C, rel=1e-9)
    assert fit.r2 == pytest.approx(LASOCKA_R2, rel=1e-9)
    codes = set(fit.codes)
    for expected in ("negative_lasocka_b", "non_monotonic", "thermal_history",
                     "single_specimen_cycles", "narrow_rate_span", "no_value",
                     "mass_loss_spread", "virgin_mixed", "low_r2", "dof_one"):
        assert expected in codes
    assert fit.max_severity == "critical"
    # Every message that a user might read as a refusal says otherwise.
    assert "Fitted anyway." in " ".join(w.message for w in fit.all_warnings)


def test_the_annotation_string_is_actually_drawn_on_the_figure(
        qtbot, dta_cycles_path):
    """SPEC_rate_series test 71. Store-and-draw must not drift: a saved PNG has to
    carry the same caveats this attribute reports, so the attribute is what is
    drawn rather than a second formatting of the same fit."""
    widget = _cycles(qtbot, dta_cycles_path)
    widget.rate_compute_all()
    assert widget.rate_annotation_text
    assert widget.rate_annotation_text in _texts(widget)
    # ... and it survives every redraw path, not only the one that made it
    widget.rate_set_model("moynihan")
    assert widget.rate_annotation_text in _texts(widget)
    widget._rate_draw()
    assert widget.rate_annotation_text in _texts(widget)
    # The figure elides; the panel does not.
    assert widget.rate_annotation_text.count("⚠") <= RATE_ANNOTATION_MAX_WARNINGS
    assert "more warning" in widget.rate_annotation_text
    panel = widget.rate_fit_label.text()
    for w in widget.rate_last_fit.all_warnings:
        assert w.message in panel


def test_the_annotation_accounts_for_every_row_not_in_the_fit(qtbot, dta_cycles_path):
    """The counts PARTITION the rows, and move when the user switches one off.

    RETARGETED. This test used to pin the old wording verbatim —
    `"n = 3 of 4 rows   (0 excluded, 1 no value)"` and
    `"n = 2 of 4 rows   (1 excluded, 1 no value)"` — and those assertions were
    wrong, not merely differently worded. The old line mixed `fit.n` (a
    model-level count) with counts taken over `r.include` (series level) and
    its parenthetical knew only two causes, so the three numbers did not add up
    to the row total on real data: an adversarial pass measured
    `"n = 3 of 8 rows   (0 excluded, 3 no value)"` — six of eight — on this
    same fixture with the cooling rows added under Augis-Bennett, with five
    markers drawn and the line going through three.

    Worse, `n_no_value` filtered on `r.include`, so switching a FAILED row off
    moved it into "excluded" and the figure printed "0 no value" three lines
    above a banner reading "3 row(s) have no value". That is what the second
    assertion here pinned: the four-row case happens to balance only because
    the row switched off is one that HAS a value.

    The line now reports four buckets that partition the rows — fitted,
    switched off (and carrying a value), no value (over ALL rows, exactly as
    `check_series` counts them), and dropped by the model — and the assertions
    below check the arithmetic rather than the prose.
    """
    import re
    widget = _cycles(qtbot, dta_cycles_path)
    widget.rate_compute_all()

    def accounting():
        got = re.search(r"n = (\d+) of (\d+) rows\s+\((\d+) switched off, "
                        r"(\d+) no value, (\d+) dropped by the model\)",
                        widget.rate_annotation_text)
        assert got, widget.rate_annotation_text
        return tuple(int(g) for g in got.groups())

    n_fit, n_tot, n_off, n_blank, n_dropped = accounting()
    assert (n_fit, n_tot, n_off, n_blank, n_dropped) == (3, 4, 0, 1, 0)
    assert n_fit + n_off + n_blank + n_dropped == n_tot

    widget.rate_set_include(widget.rate_rows[0].uid, False)
    widget._rate_refit()
    n_fit, n_tot, n_off, n_blank, n_dropped = accounting()
    assert (n_fit, n_tot, n_off, n_blank, n_dropped) == (2, 4, 1, 1, 0)
    assert n_fit + n_off + n_blank + n_dropped == n_tot
    assert "Not fitted: heat 2 (cycle 2)" in widget.rate_annotation_text
    assert "heat 1" not in widget.rate_annotation_text   # excluded, not "not fitted"
    assert "quantity: Tg · parallel · window 350-700 °C" in widget.rate_annotation_text


def test_the_annotation_says_nothing_has_been_measured_before_it_has(
        qtbot, dta_cycles_path):
    """The figure is honest in the intermediate state too: four rows on the axis
    at their own rates, and a line telling the user what to press."""
    widget = _cycles(qtbot, dta_cycles_path)
    assert "No values yet" in widget.rate_annotation_text
    assert widget.rate_annotation_text in _texts(widget)
    assert _fit_lines(widget) == []
    assert len(_markers(widget, "|")) == 4
    assert "No values yet" in widget.rate_fit_label.text()


@pytest.mark.parametrize("method", list(dk.KINETIC_METHODS))
def test_the_drawn_curve_reproduces_each_models_own_fitted_line(method):
    """`RATE_MODEL_Q_OF_T` is the only model-specific arithmetic in the
    presentation layer, so it is checked against `dta_kinetics` for all six
    models on data lying EXACTLY on each model's own line: the curve must
    reproduce the rates it was fitted from to floating-point precision.

    Parameterised by temperature rather than by rate because that direction is
    closed form for every model — three of them are transcendental in T, and a
    second numeric inversion next to `dta_kinetics._solve_T_at_q` would be a
    second place to get a sign wrong.
    """
    kind = "Tp" if method in ("kissinger", "fwo", "augis_bennett") else "Tg"
    T0_C = 30.0
    T_K = np.linspace(900.0, 1100.0, 5)
    T_C = T_K - dk.K0
    slope, intercept = -34000.0, 25.0
    if method == "lasocka":
        # Lasocka's own space is (ln q, T in °C): pick B and A, not m and b.
        q = np.exp((T_C - 500.0) / 14.0)
    elif method in ("moynihan", "fwo"):
        q = np.exp(intercept + slope / T_K)
    elif method == "augis_bennett":
        q = (T_C - T0_C) * np.exp(intercept + slope / T_K)
    else:
        q = T_K ** 2 * np.exp(intercept + slope / T_K)

    points = [dk.RatePoint(q_C_per_min=float(qq), T_C=float(tt),
                           quantity_kind=kind, T0_C=T0_C,
                           label=f"r{i}", uid=f"u{i}")
              for i, (qq, tt) in enumerate(zip(q, T_C))]
    fit = dk.fit_points(points, method)
    assert fit.ok and fit.r2 == pytest.approx(1.0, abs=1e-9)
    q_of_T = RATE_MODEL_Q_OF_T[method]
    predicted = q_of_T(T_C, float(fit.slope), float(fit.intercept),
                       float(fit.extras.get("T0_mean_C", float("nan"))))
    assert np.allclose(predicted, q, rtol=1e-9)


def test_the_curve_is_drawn_through_the_points_and_spans_their_rates(qtbot):
    """One synthetic Lasocka series with a real positive B, so the drawn line is
    the one a user would publish: it runs across the measured rates and passes
    through the points rather than near them."""
    widget = _on_rate_tab(_workspace(qtbot))
    rates = (2.5, 5.0, 10.0, 20.0)
    for q in rates:
        widget.rate_add_manual_row(sample=f"{q:g}", q_C_per_min=q,
                                   value_C=500.0 + 14.0 * np.log(q))
    fit = widget.rate_last_fit
    assert fit.method == "lasocka"
    assert fit.lasocka_B == pytest.approx(14.0, rel=1e-9)
    curve = widget._rate_fit_curve(fit)
    assert curve is not None
    qs, ts = curve
    assert qs.size >= 2 and bool(np.all(np.diff(qs) > 0))
    assert qs.min() <= min(rates) and qs.max() >= max(rates)
    for q in rates:
        assert float(np.interp(q, qs, ts)) == pytest.approx(
            500.0 + 14.0 * np.log(q), abs=0.5)
    lines = _fit_lines(widget)
    assert len(lines) == 1
    assert "Lasocka" in lines[0].get_label() and "R²" in lines[0].get_label()


def test_no_curve_is_drawn_rather_than_a_wrong_one(qtbot):
    """A missing line is a visible absence; a line through one point, or drawn
    off the end of the axis, is a claim about a fit that does not exist."""
    widget = _on_rate_tab(_workspace(qtbot))
    widget.rate_add_manual_row(sample="only one", q_C_per_min=10.0, value_C=600.0)
    assert widget.rate_last_fit.ok is False
    assert widget._rate_fit_curve(widget.rate_last_fit) is None
    assert _fit_lines(widget) == []
    assert "No fit" in widget.rate_fit_label.text()
    # Two rows at the SAME rate: still no x-spread, still no line.
    widget.rate_add_manual_row(sample="same rate", q_C_per_min=10.0, value_C=610.0)
    assert widget.rate_last_fit.ok is False
    assert _fit_lines(widget) == []
    assert "duplicate_rate" in _codes(widget) or "single_rate" in _codes(widget)


def test_an_excluded_row_is_greyed_and_leaves_the_fit(qtbot):
    """SPEC_rate_series test 72: the row survives, `n` drops by one, and an "x"
    marker appears where the point was. Excluding is not deleting."""
    widget = _on_rate_tab(_workspace(qtbot))
    uids = [widget.rate_add_manual_row(sample=f"{q:g}", q_C_per_min=q,
                                       value_C=500.0 + 14.0 * np.log(q))
            for q in (2.5, 5.0, 10.0, 20.0)]
    assert widget.rate_last_fit.n == 4
    assert _markers(widget, "x") == []

    widget.rate_set_include(uids[0], False)
    widget._rate_refit()
    assert len(widget.rate_rows) == 4
    assert widget.rate_last_fit.n == 3
    crosses = _markers(widget, "x")
    assert len(crosses) == 1
    assert crosses[0].get_xdata()[0] == pytest.approx(2.5)
    assert crosses[0].get_color() == "0.55"
    assert "excluded_rows" in _codes(widget)


def test_hiding_excluded_rows_raises_hidden_excluded_and_is_reversible(qtbot):
    """`hidden_excluded` is the one warning code `dta_rate_series` leaves to the
    Qt layer (`STUBBED_WARNING_CODES`), because it depends on a toggle only this
    layer has. A figure silently not showing two of six rows is the figure that
    gets published."""
    widget = _on_rate_tab(_workspace(qtbot))
    uids = [widget.rate_add_manual_row(sample=f"{q:g}", q_C_per_min=q,
                                       value_C=500.0 + 14.0 * np.log(q))
            for q in (2.5, 5.0, 10.0, 20.0)]
    widget.rate_set_include(uids[0], False)
    widget._rate_refit()
    assert "hidden_excluded" not in _codes(widget)

    widget.rate_show_excluded_check.setChecked(False)
    assert "hidden_excluded" in _codes(widget)
    assert _markers(widget, "x") == []
    assert len(widget.rate_rows) == 4                 # the table is untouched
    assert widget.rate_table.rowCount() == 4
    assert widget.rate_last_fit.n == 3                # and so is the fit

    widget.rate_show_excluded_check.setChecked(True)
    assert "hidden_excluded" not in _codes(widget)
    assert len(_markers(widget, "x")) == 1
    # With nothing excluded the toggle raises nothing either way.
    widget.rate_set_include(uids[0], True)
    widget.rate_show_excluded_check.setChecked(False)
    assert "hidden_excluded" not in _codes(widget)


def test_point_labels_can_be_switched_off_without_losing_the_points(qtbot):
    widget = _on_rate_tab(_workspace(qtbot))
    for q in (5.0, 10.0):
        widget.rate_add_manual_row(sample=f"specimen {q:g}", q_C_per_min=q,
                                   value_C=500.0 + 14.0 * np.log(q))
    assert "specimen 5" in _texts(widget)
    widget.rate_show_labels_check.setChecked(False)
    assert "specimen 5" not in _texts(widget)
    assert len(_markers(widget, "o")) == 1
    assert widget.rate_annotation_text in _texts(widget)


def test_q_ref_changes_only_the_back_prediction(qtbot):
    """q_ref is a question put to a fitted line, not an input to the fit."""
    widget = _on_rate_tab(_workspace(qtbot))
    for q in (2.5, 5.0, 10.0, 20.0):
        widget.rate_add_manual_row(sample=f"{q:g}", q_C_per_min=q,
                                   value_C=500.0 + 14.0 * np.log(q))
    first = widget.rate_last_fit
    assert first.q_ref_C_per_min == pytest.approx(10.0)
    assert first.T_at_q_ref_C == pytest.approx(500.0 + 14.0 * np.log(10.0), abs=1e-6)

    widget.rate_qref_edit.setText("1")
    widget._on_rate_view_changed()
    second = widget.rate_last_fit
    assert second.slope == first.slope and second.intercept == first.intercept
    assert second.T_at_q_ref_C == pytest.approx(500.0, abs=1e-6)
    # Junk is the documented default, not a dialog and not a crash.
    widget.rate_qref_edit.setText("not a number")
    widget._on_rate_view_changed()
    assert widget.rate_last_fit.q_ref_C_per_min == pytest.approx(dk.DEFAULT_Q_REF)


def test_the_row_label_distinguishes_files_across_sources_and_segments_within_one(
        qtbot, dta_cycles_path, dta_example_path):
    """Inside one cyclic file every row shares the file name and differs by
    segment; across separate files every row is "heat 1" and differs by file. The
    label has to pick whichever half is the distinguishing one, or the annotation
    names four rows identically."""
    widget = _cycles(qtbot, dta_cycles_path)
    assert [widget._rate_row_label(r) for r in widget.rate_rows] == [
        "heat 1 (cycle 1)", "heat 2 (cycle 2)", "heat 3 (cycle 3)",
        "heat 4 (cycle 4)"]
    widget.rate_clear()
    widget.rate_add_files([dta_example_path])
    # The Sample header, not the file stem: "NaBiO3-400C-N2" is what the
    # operator called the specimen and says more than whatever the export was
    # filed as. The stem is only the fallback when there is no Sample header.
    assert widget._rate_row_label(widget.rate_rows[0]) == "NaBiO3-400C-N2"
    uid = widget.rate_add_manual_row(sample="", q_C_per_min=3.0)
    assert widget._rate_row_label(widget._rate_row_by_uid(uid)) == "row 2"


def test_two_rows_that_would_share_a_label_are_told_apart(qtbot, dta_example_path):
    """Picking the distinguishing half is not the same as being distinct.

    Re-running one specimen name in two exports gives both rows the same Sample
    header, and an annotation that names two rows identically leaves the reader
    unable to tell which point a warning is about. Only the colliding rows pay
    for the disambiguation -- a label that is already unique is left alone.
    """
    widget = DtaWorkspace()
    qtbot.addWidget(widget)
    a = widget.rate_add_manual_row(sample="BM-0", q_C_per_min=10.0, value_C=600.0)
    b = widget.rate_add_manual_row(sample="BM-0", q_C_per_min=20.0, value_C=610.0)
    c = widget.rate_add_manual_row(sample="BM-9", q_C_per_min=15.0, value_C=605.0)
    labels = {u: widget._rate_row_label(widget._rate_row_by_uid(u))
              for u in (a, b, c)}
    assert labels[c] == "BM-9"                       # unique, left alone
    assert labels[a] != labels[b]                    # the collision is broken
    assert len(set(labels.values())) == 3
    for uid in (a, b):
        assert "BM-0" in labels[uid]                 # still recognisable


# ===========================================================================
# The banner, and the invariant that warnings never gate
# ===========================================================================

def test_refit_never_consults_the_warnings(qtbot, dta_cycles_path, monkeypatch):
    """SPEC_rate_series test 86. Thirty critical warnings must change the fit by
    exactly nothing: `check_series` describes a result, it never decides whether
    to produce one (MERGED_DECISIONS M0.1)."""
    widget = _cycles(qtbot, dta_cycles_path)
    widget.rate_compute_all()
    fit = widget.rate_last_fit
    before = (fit.ok, fit.slope, fit.intercept, fit.lasocka_B, fit.n, fit.r2)

    monkeypatch.setattr(drs, "check_series", lambda *a, **k: [
        dk.SeriesWarning(code=f"invented_{i}", severity="critical",
                         message=f"objection {i}") for i in range(30)])
    widget._rate_refit()
    after = widget.rate_last_fit
    assert (after.ok, after.slope, after.intercept, after.lasocka_B, after.n,
            after.r2) == before
    assert sum(1 for c in _codes(widget) if c.startswith("invented_")) == 30
    assert widget.rate_warn_banner.objectName() == "WarningBannerCritical"
    # The drawn line is unchanged too: the figure is not gated either.
    assert len(_fit_lines(widget)) == 1


def test_compute_all_never_consults_the_warnings(qtbot, dta_cycles_path, monkeypatch):
    """The same invariant on the measurement side. Thirty critical objections
    must not stop a single row from being measured."""
    widget = _cycles(qtbot, dta_cycles_path)
    monkeypatch.setattr(drs, "check_series", lambda *a, **k: [
        dk.SeriesWarning(code=f"invented_{i}", severity="critical",
                         message=f"objection {i}") for i in range(30)])
    assert widget.rate_compute_all() == (3, 1)
    assert [r.value_C for r in widget.rate_rows][0] == pytest.approx(
        CYCLE_TG_C[0], abs=1e-4)


def test_the_banner_is_quiet_until_a_value_exists_and_then_carries_the_fit(
        qtbot, dta_cycles_path):
    """Before anything is measured the only objections possible are "too few
    points" and "no value" — which say what the summary line already says, in the
    warning tone, about the normal state two clicks after adding files. Spending
    the colour there teaches a user that amber means "not finished yet", after
    which it stops meaning "something is wrong with your data".

    Nothing is hidden by the rule: the moment any row carries a number, every
    objection to the fit is on the banner.
    """
    widget = _cycles(qtbot, dta_cycles_path)
    assert _codes(widget) == ["multi_rate_source"]
    assert widget.rate_warn_banner.objectName() == "WarningBannerOk"
    assert "too_few_points" not in _codes(widget)

    widget.rate_compute_all()
    codes = _codes(widget)
    assert "negative_lasocka_b" in codes and "no_value" in codes
    assert widget.rate_warn_banner.objectName() == "WarningBannerCritical"
    assert "multi_rate_source" in codes          # the add-time line is still there


def test_the_same_observation_from_two_layers_is_shown_once(qtbot, dta_cycles_path):
    """The add path and `check_series` both raise `multi_rate_source` about the
    same four rows, in their own words. The banner is a list of facts a user has
    to read, so it states that fact once — the add-time wording, which names the
    run and its three rates."""
    widget = _cycles(qtbot, dta_cycles_path)
    widget.rate_compute_all()
    assert _codes(widget).count("multi_rate_source") == 1
    line = next(w for w in widget.rate_warnings
                if w.code == "multi_rate_source").message
    assert "9.99" in line and "14.99" in line and "19.95" in line
    assert widget.rate_warn_banner.text().count("4 heating ramps") == 1


def test_the_banner_object_name_follows_the_worst_severity(qtbot):
    """Three states, and info alone stays in the QUIET tone: "this file holds
    four heating ramps" is information about the data, not a fault in it."""
    widget = _on_rate_tab(_workspace(qtbot))
    assert widget.rate_warn_banner.objectName() == "WarningBannerOk"

    for q in (9.0, 10.0):                       # a 1.11x span -> narrow_rate_span
        widget.rate_add_manual_row(sample=f"{q:g}", q_C_per_min=q,
                                   value_C=500.0 + 14.0 * np.log(q))
    assert widget.rate_last_fit.ok is True
    assert widget.rate_warn_banner.objectName() in ("WarningBanner",
                                                    "WarningBannerCritical")
    assert "narrow_rate_span" in _codes(widget)

    widget.rate_clear()
    assert widget.rate_warnings == []
    assert widget.rate_warn_banner.objectName() == "WarningBannerOk"
    assert "No rows yet" in widget.rate_warn_banner.text()


def test_a_negative_activation_energy_is_reported_and_handed_over(qtbot):
    """MERGED_DECISIONS M1's worked example: the cooling exotherms of the real
    file give a physically impossible Ea. The spec's original answer was to
    refuse; the user's answer is to produce it and say what is wrong with it."""
    widget = _on_rate_tab(_workspace(qtbot))
    widget.rate_quantity_combo.setCurrentIndex(3)               # Tp
    for q, tp in ((5.0, 700.0), (10.0, 680.0), (20.0, 660.0)):  # Tp FALLS with q
        widget.rate_add_manual_row(sample=f"{q:g}", q_C_per_min=q, value_C=tp,
                                   quantity_kind="Tp")
    fit = widget.rate_last_fit
    assert fit.method == "kissinger"
    assert fit.ok is True
    assert fit.Ea_kJ_mol is not None and fit.Ea_kJ_mol < 0.0
    assert "negative_ea" in fit.codes or "positive_slope_kissinger" in fit.codes
    assert fit.max_severity == "critical"
    assert widget.rate_warn_banner.objectName() == "WarningBannerCritical"
    assert len(_fit_lines(widget)) == 1                          # drawn anyway


# ===========================================================================
# Risk R4 — the °C / K display switch never converts a stored value
# ===========================================================================

def test_computed_values_are_bit_identical_across_a_celsius_kelvin_round_trip(
        qtbot, dta_cycles_path):
    """R4. The switch relabels a header, re-renders a cell and redraws an axis. A
    value converted to kelvin twice still looks like a plausible Tg and makes an
    activation energy wrong by roughly a factor of three, so the stored number
    must not move by one bit."""
    widget = _cycles(qtbot, dta_cycles_path)
    widget.rate_compute_all()
    stored = [r.value_C for r in widget.rate_rows]

    def same_as_stored():
        return all((np.isnan(a) and np.isnan(b)) or a == b
                   for a, b in zip(stored, (r.value_C for r in widget.rate_rows)))

    def header():
        return widget.rate_table.horizontalHeaderItem(RATE_COL_VALUE).text()

    assert header() == "Value (°C)"
    assert widget.rate_plot.ax.get_ylabel() == "Tg   (°C)"
    assert widget.rate_table.item(0, RATE_COL_VALUE).text() == "622.815"

    widget.temp_k_btn.setChecked(True)
    assert header() == "Value (K)"
    assert widget.rate_plot.ax.get_ylabel() == "Tg   (K)"
    assert widget.rate_table.item(0, RATE_COL_VALUE).text() == "895.965"
    assert same_as_stored()

    widget.temp_c_btn.setChecked(True)
    assert header() == "Value (°C)"
    assert widget.rate_plot.ax.get_ylabel() == "Tg   (°C)"
    assert widget.rate_table.item(0, RATE_COL_VALUE).text() == "622.815"
    assert same_as_stored()


def test_a_value_computed_while_the_display_shows_kelvin_is_stored_as_celsius(
        qtbot, dta_cycles_path):
    """What is stored never depends on what was on screen. The measurement path
    never sees the display unit at all — it reads the furnace temperature column
    in °C — which is what makes this exact rather than approximate."""
    reference = _cycles(qtbot, dta_cycles_path)
    reference.rate_compute_all()
    expected = [r.value_C for r in reference.rate_rows]

    widget = _cycles(qtbot, dta_cycles_path)
    widget.temp_k_btn.setChecked(True)
    assert widget._temp_unit() == "K"
    assert widget.rate_compute_all() == (3, 1)
    for got, want in zip((r.value_C for r in widget.rate_rows), expected):
        assert (np.isnan(got) and np.isnan(want)) or got == want
    assert {r.window_C for r in widget.rate_rows} == {(350.0, 700.0)}


def test_the_activation_energy_is_identical_in_both_display_units(
        qtbot, dta_cycles_path):
    """SPEC_rate_series test 69. Ea is not offset-invariant, so a display switch
    that reached the fit would change it — by about a factor of three. It does
    not reach the fit: `temp_basis` says the fit did its own conversion, once."""
    widget = _cycles(qtbot, dta_cycles_path)
    widget.rate_compute_all()
    widget.rate_set_model("moynihan")
    ea_c = widget.rate_last_fit.Ea_kJ_mol
    assert ea_c == pytest.approx(MOYNIHAN_EA, rel=1e-9)
    assert widget.rate_last_fit.temp_basis == "K (converted)"

    widget.temp_k_btn.setChecked(True)
    assert widget.rate_last_fit.Ea_kJ_mol == ea_c
    assert widget.rate_last_fit.temp_basis == "K (converted)"
    widget.temp_c_btn.setChecked(True)
    assert widget.rate_last_fit.Ea_kJ_mol == ea_c


# ===========================================================================
# Risk R6 — a queued render must not overwrite the fit
# ===========================================================================

def test_the_fitted_figure_survives_a_pending_debounced_render(qtbot):
    """`PlotWidget.request_redraw` fires up to 120 ms later and would overwrite a
    direct draw (the canvas's own docstring), so `_rate_draw` cancels the queue
    first. Without the cancel this test loses the fit line and the annotation 200
    ms after they were drawn."""
    widget = _on_rate_tab(_workspace(qtbot))
    for q in (5.0, 10.0, 20.0):
        widget.rate_add_manual_row(sample=f"{q:g}", q_C_per_min=q,
                                   value_C=500.0 + 14.0 * np.log(q))
    widget.rate_plot.request_redraw(lambda: widget.rate_plot.ax.clear())
    widget._rate_refit()
    qtbot.wait(200)
    assert len(_markers(widget, "o")) == 1
    assert len(_fit_lines(widget)) == 1
    assert widget.rate_annotation_text in _texts(widget)


# ===========================================================================
# The pinned Tg, once more, on the final code
# ===========================================================================

def test_the_pinned_tg_survives_a_full_compute_and_fit(qtbot, dta_example_path):
    """354.4666531002641 / 354.51135373692597 / 357.6214 — the values the
    original Tk TgGuiApp produced. Re-measured after a rate series has been
    built, measured, fitted, drawn, switched to kelvin and back, because the risk
    of this step is a shared setting or a construction order that moves the Tg
    tab underneath itself."""
    widget = _on_rate_tab(_workspace(qtbot, dta_example_path, rec_id="ex-1"))
    widget.rate_add_segments()
    widget.rate_add_manual_row(sample="typed", q_C_per_min=5.0, value_C=350.0)
    widget.rate_compute_all()
    widget.rate_set_model("moynihan")
    widget.temp_k_btn.setChecked(True)
    widget.temp_c_btn.setChecked(True)
    widget.tabs.setCurrentIndex(0)
    widget._compute()
    assert widget.res_double.tg == pytest.approx(354.4666531002641, abs=1e-4)
    assert widget.res_parallel.tg == pytest.approx(354.51135373692597, abs=1e-4)
    assert widget.tg_deriv == pytest.approx(357.6214, abs=1e-3)
    assert widget.xmin_edit.text() == "350"
    assert widget.xmax_edit.text() == "700"
