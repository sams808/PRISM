"""Segment-scoped arrays in the DTA workspace (SPEC_cycles 2.5, Phase B step 6).

Two halves, and the first is the more important one:

* the BACKWARD-COMPATIBILITY half -- on a single-ramp file the reworked
  `_prepare_xy` must return arrays that are *bit-identical* to the pre-rework
  `mask -> np.argsort` ones, or the pinned Tg 354.4666531002641 moves. Risk R3
  names the four ways that can break (a non-stable sort, the auto window firing
  on a single-segment file, the finite mask applied in the wrong order relative
  to segment selection, and dedup averaging where the old code kept the first
  y), so the tests here compare the arrays themselves rather than the Tg alone;

* the SCOPE half -- a cyclic record is detected as 12 segments, its scope is the
  whole run and is not computable, and any call site that tries to reduce it to
  one sorted curve fails loudly (G2) instead of quietly reproducing the
  whole-file collapse.

Sections 4-9 cover Phase B steps 7 and 8: the scope selector and its memory,
the G1 guardrails (what is COMPUTABLE, said in the UI and never in a dialog),
the three render modes, the direction-safety fixes in `_draw_overlays`, and
the debounce discipline that keeps a 15-entry combo from redrawing a 19k-point
record once per arrow key.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from PySide6.QtWidgets import QMessageBox

from dta.qt_dta import (
    SCOPE_ALL_COOLING, SCOPE_ALL_HEATING, SCOPE_ALL_TIME, SCOPE_SEGMENT,
    DtaWorkspace, _interp_ascending, _scope_for_run,
)
from dta.dta_segments import KIND_HEAT, SegmentScopeError
from dta.dta_science import (
    compute_tg_derivative, compute_tg_double_tangent,
    compute_tg_parallel_improved, resolve_baseline_params,
)
from core.qt_shell import _load_spectrum_from_path
from conftest import requires_dta_bm


def _record(path, *, rec_id=None):
    """A library-shaped record dict. `rec_id=None` reproduces the hand-built
    records the pinned tests use, which carry no `"id"` at all."""
    spectrum = _load_spectrum_from_path(str(path))
    rec = {"title": spectrum.title, "path": spectrum.path,
           "df": spectrum.df, "meta": spectrum.meta}
    if rec_id is not None:
        rec["id"] = rec_id
    return rec


# ===========================================================================
# 1. The backward-compatibility half
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

def test_prepare_xy_reproduces_the_old_masked_argsort_arrays(qtbot, dta_example_path):
    """THE keystone of step 6. `EXAMPLES/DTA_example.txt` is one heating ramp,
    `np.argsort(T)` on it is the identity permutation and it holds no duplicate
    temperatures, so slice + stable sort + duplicate collapse are all no-ops and
    the two code paths must agree byte for byte."""
    widget = DtaWorkspace(records=[_record(dta_example_path)])
    qtbot.addWidget(widget)

    x_col = widget.x_combo.currentText()
    y_col = widget.y_combo.currentText()
    df = widget.df
    x_raw = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
    y_raw = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)

    # Exactly what dta/qt_dta.py::_get_xy did before the rework.
    mask = np.isfinite(x_raw) & np.isfinite(y_raw)
    x_old, y_old = x_raw[mask], y_raw[mask]
    order = np.argsort(x_old)
    assert np.array_equal(order, np.arange(order.size))      # the identity
    x_old, y_old = x_old[order], y_old[order]

    x_new, y_new, y_dy = widget._get_xy()[:3]
    assert np.array_equal(x_new, x_old)
    assert np.array_equal(y_new, y_old)
    # "(same)" dY must still be a distinct array, as the old `y.copy()` was.
    assert np.array_equal(y_dy, y_old)
    assert y_dy is not y_new


def test_pinned_tg_is_bit_identical_to_the_pre_rework_path(qtbot, dta_example_path):
    """`test_dta_workspace_compute_matches_known_good_values` pins the three Tg
    values to 1e-4. This asserts the stronger and more useful thing: the
    reworked widget returns the SAME FLOAT BITS as the deleted
    mask-then-argsort path, recomputed here from the DataFrame.

    Equality is exact (`==`, not `approx`) on purpose. Identical input arrays
    through identical `dta_science` calls can only give identical output, so
    any difference at all means the rework moved a number, and a tolerance
    would only decide how long it took to notice. The literal pins stay too, at
    their own tolerance: they are the contract with the original Tk tool, and
    the last two digits of the recorded 354.4666531002641 differ from what this
    machine's numpy produces by 4e-13 -- which predates this rework, as the
    recomputation in this test shows.
    """
    widget = DtaWorkspace(records=[_record(dta_example_path)])
    qtbot.addWidget(widget)
    widget._compute()

    x_col = widget.x_combo.currentText()
    y_col = widget.y_combo.currentText()
    x = pd.to_numeric(widget.df[x_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(widget.df[y_col], errors="coerce").to_numpy(dtype=float)
    y_dy = y.copy()
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(y_dy)
    x, y, y_dy = x[mask], y[mask], y_dy[mask]
    order = np.argsort(x)
    x, y, y_dy = x[order], y[order], y_dy[order]

    bp = resolve_baseline_params(
        manual_enabled=False, low_use_point=False, low_point_x=None,
        low_min=None, low_max=None, high_use_point=False, high_point_x=None,
        high_min=None, high_max=None, slope_min=None, slope_max=None)
    old_double = compute_tg_double_tangent(
        x, y, 350.0, 700.0, threshold=0.20, guard_frac=0.0,
        smooth_derivative=False, manual_low=bp.low_range,
        manual_slope=bp.manual_slope).tg
    old_parallel = compute_tg_parallel_improved(
        x, y, 350.0, 700.0, smooth_derivative=False,
        manual_low_range=bp.low_range, manual_high_range=bp.high_range,
        manual_low_point=bp.low_point, manual_high_point=bp.high_point).tg
    old_deriv = compute_tg_derivative(x, y_dy, 350.0, 700.0,
                                      smooth_derivative=False, restrict_range=None)

    assert widget.res_double.tg == old_double
    assert widget.res_parallel.tg == old_parallel
    assert widget.tg_deriv == old_deriv

    assert widget.res_double.tg == pytest.approx(354.4666531002641, abs=1e-4)
    assert widget.res_parallel.tg == pytest.approx(354.51135373692597, abs=1e-4)
    assert widget.tg_deriv == pytest.approx(357.6214, abs=1e-3)

    # The window is untouched: the segment-derived default must not fire on a
    # single-segment file, and nothing has typed in the fields.
    assert (widget.xmin_edit.text(), widget.xmax_edit.text()) == ("350", "700")
    assert widget._window_user_edited is False


def test_single_ramp_example_is_one_computable_segment(qtbot, dta_example_path):
    widget = DtaWorkspace(records=[_record(dta_example_path)])
    qtbot.addWidget(widget)

    assert len(widget.run) == 1
    scope = widget._scope()
    assert scope.kind == SCOPE_SEGMENT
    assert scope.computable is True
    assert scope.x_kind == ""
    seg = widget._current_segment()
    assert seg is not None and seg.kind == KIND_HEAT
    assert (seg.row_start, seg.row_stop) == (0, len(widget.df))


def test_calc_derivative_against_time_still_works_on_the_example(qtbot, dta_example_path):
    """The time abscissa takes the no-sort branch and must stay strictly
    increasing, so np.gradient cannot divide by a zero step -- the job the
    deleted `np.unique` dedup used to do in this one branch alone."""
    widget = DtaWorkspace(records=[_record(dta_example_path)])
    qtbot.addWidget(widget)

    widget.calc_use_deriv_check.setChecked(True)
    widget.calc_deriv_x_combo.setCurrentText("Time (min)")
    x, dy, label = widget._get_calc_y()
    assert bool(np.all(np.diff(x) > 0))
    assert np.isfinite(dy).all()
    assert label.startswith("d(")


# ===========================================================================
# 2. The scope half
# ===========================================================================
def test_cycles_example_detects_twelve_segments_on_load(qtbot, dta_cycles_path):
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)

    assert widget.run is not None
    assert len(widget.run) == 12
    assert widget.run.n_cycles == 4
    scope = widget._scope()
    assert scope.kind == SCOPE_ALL_TIME
    assert scope.computable is False
    assert scope.segment is None
    assert "12 segments" in scope.label


def test_compute_on_a_multi_segment_scope_raises_scope_error(qtbot, dta_cycles_path):
    """G2. A call site that forgets to check `scope.computable` fails loudly
    instead of silently reproducing the whole-file collapse (all three methods
    on 431.758 C with a 4e-5 spread, reported as agreement)."""
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)

    x_col = widget.x_combo.currentText()
    x = pd.to_numeric(widget.df[x_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(widget.df[widget.y_combo.currentText()],
                      errors="coerce").to_numpy(dtype=float)
    with pytest.raises(SegmentScopeError) as err:
        widget._prepare_xy(x, y, x_col=x_col)
    assert "12 segments" in str(err.value)


def test_an_explicit_run_of_none_means_that_file_not_the_loaded_one(qtbot, dta_cycles_path):
    """`run=None` is a VALUE -- "this file could not be segmented, treat it as
    one whole record" -- and must not be read as "argument omitted". If it
    were, the batch loop would prepare one file's columns against the
    workspace record's segment boundaries."""
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    assert len(widget.run) == 12                 # the loaded record's own run

    t = pd.to_numeric(widget.df["Time (min)"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(widget.df[widget.y_combo.currentText()],
                      errors="coerce").to_numpy(dtype=float)
    prepared = widget._prepare_xy(t, y, x_col="Time (min)", run=None)
    assert prepared.x.size == len(widget.df)
    assert bool(np.all(np.diff(prepared.x) > 0))

    # ... while the default really does use the loaded record, and refuses.
    with pytest.raises(SegmentScopeError):
        widget._prepare_xy(t, y, x_col="Time (min)")


def test_loading_a_cyclic_record_carries_every_orgmethod_line(qtbot, dta_cycles_path):
    """The header fold used to keep only `1: Select gas 1`. Without all 19
    lines the programmed method is destroyed before anything can read it."""
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)

    org = [v for k, v in widget.header_lines if k == "OrgMethod"]
    assert len(org) == 19
    assert widget.program is not None
    assert len(widget.program.steps) == 19
    assert bool(widget.program) is True


def test_switching_records_does_not_reuse_a_stale_segment_index(
        qtbot, dta_cycles_path, dta_example_path):
    """A 12-segment record followed by a 1-segment one: the scope must fall
    back to that record's own segment, with no IndexError and no leftover."""
    widget = DtaWorkspace(records=[_record(dta_cycles_path), _record(dta_example_path)])
    qtbot.addWidget(widget)
    assert len(widget.run) == 12

    widget.record_combo.setCurrentIndex(1)
    assert len(widget.run) == 1
    assert widget._scope().kind == SCOPE_SEGMENT
    assert widget._current_segment().row_stop == len(widget.df)

    widget.record_combo.setCurrentIndex(0)
    assert len(widget.run) == 12
    assert widget._scope().kind == SCOPE_ALL_TIME


def test_export_result_csv_on_an_aggregate_scope_raises_instead_of_crashing(
        qtbot, dta_cycles_path, tmp_path):
    """`_current_result_row` goes through `_get_xy`, which now refuses an
    aggregate scope. The callable half must raise so the dialog half can show
    a message -- an exception escaping a Qt slot is a crash, not a warning."""
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)

    with pytest.raises(SegmentScopeError):
        widget.export_current_csv(tmp_path / "never.csv")
    assert not (tmp_path / "never.csv").exists()


def test_export_result_csv_writes_the_row_on_a_single_segment(
        qtbot, dta_example_path, tmp_path):
    widget = DtaWorkspace(records=[_record(dta_example_path)])
    qtbot.addWidget(widget)
    widget._compute()

    out = tmp_path / "result.csv"
    row = widget.export_current_csv(out)
    assert out.is_file()
    assert row["Tg_parallel"] == pytest.approx(354.51135373692597, abs=1e-4)


def test_batch_whole_file_refuses_a_cyclic_file(qtbot, dta_example_path,
                                                dta_cycles_path, tmp_path):
    """Whole-file batch over a cyclic run: an error row naming the segments,
    never the silent whole-file number."""
    widget = DtaWorkspace(records=[_record(dta_example_path)])
    qtbot.addWidget(widget)

    out = tmp_path / "batch.csv"
    rows = widget.batch_tg_to_csv([dta_example_path, dta_cycles_path], out)
    assert out.is_file()
    assert len(rows) == 2
    assert np.isfinite(rows[0]["Tg_parallel"])
    assert "Tg_parallel" not in rows[1]
    assert "12 segments" in rows[1]["error"]


# ===========================================================================
# 3. Record identity -- the cache and the title fallback
# ===========================================================================
def test_a_record_without_an_id_is_not_cached(qtbot, dta_example_path):
    """The pinned tests build records by hand and carry no `"id"`. Those must
    load, and must never be written to the run cache: a display title is not an
    identity two imports of one file can be told apart by."""
    widget = DtaWorkspace(records=[_record(dta_example_path)])
    qtbot.addWidget(widget)

    assert widget._record_id.startswith("title:")
    assert widget._key_is_stable(widget._record_id) is False
    assert widget._run_cache == {}
    assert widget.run is not None          # detection still ran


def test_a_record_with_an_id_is_cached_and_reused(qtbot, dta_example_path):
    widget = DtaWorkspace(records=[_record(dta_example_path, rec_id="spec-1")])
    qtbot.addWidget(widget)

    assert widget._record_id == "id:spec-1"
    assert list(widget._run_cache) == ["id:spec-1"]
    first = widget.run
    widget._load_selected_record()
    assert widget.run is first             # the same object, not a re-detection


def test_scope_for_run_without_a_segmentation_is_the_whole_record():
    """A record with no clock or no temperature column is not segmented, and
    must stay computable: refusing to analyse a file the app has always
    analysed would be a regression, not a safeguard."""
    scope = _scope_for_run(None)
    assert scope.kind == SCOPE_ALL_TIME
    assert scope.computable is True
    assert scope.segments == ()
    assert scope.segment is None


def test_a_record_with_no_clock_or_temperature_still_plots_and_computes(qtbot):
    """End to end on the unsegmented fallback: a generic two-column record has
    no rate to measure, so it gets no segmentation at all -- and must behave
    exactly as it did before this rework. The whole-record stand-in segment
    carries the array rule and nothing else."""
    n = 500
    df = pd.DataFrame({"X": np.linspace(0.0, 100.0, n),
                       "Signal": np.sin(np.linspace(0.0, 6.0, n))})
    widget = DtaWorkspace(records=[{"title": "generic", "df": df, "meta": {}}])
    qtbot.addWidget(widget)

    assert widget.run is None
    assert widget._scope().computable is True
    assert widget._current_segment() is None

    x, y = widget._get_xy()[:2]
    assert x.size == n and y.size == n
    assert bool(np.all(np.diff(x) > 0))
    assert len(widget.plot.ax.lines) == 1

    widget.calc_xmin_edit.setText("10")
    widget.calc_xmax_edit.setText("50")
    widget._calc_integrate()
    assert "Integrate" in widget.calc_result_label.text()

    widget.calc_use_deriv_check.setChecked(True)
    widget.calc_deriv_x_combo.setCurrentText("X")
    x_d, dy, _label = widget._get_calc_y()
    assert x_d.size == n
    assert np.isfinite(dy).all()


# ===========================================================================
# 4. The scope selector  (SPEC_cycles Phase B step 7)
# ===========================================================================
def _scope_index(widget, key):
    """Combo row carrying this scope key. Fails loudly if it is not offered."""
    for i in range(widget.segment_combo.count()):
        if widget.segment_combo.itemData(i) == key:
            return i
    raise AssertionError(f"{key!r} is not in the scope list: "
                         + repr([widget.segment_combo.itemData(i)
                                 for i in range(widget.segment_combo.count())]))


def _select_scope(widget, key, qtbot):
    """Pick a scope the way the user does, and let the debounced redraw fire."""
    widget.segment_combo.setCurrentIndex(_scope_index(widget, key))
    _settle(qtbot, widget)


def _x_index(widget, name):
    return [widget.x_combo.itemText(i) for i in range(widget.x_combo.count())].index(name)


def test_cycles_example_offers_three_aggregates_and_twelve_segments(qtbot, dta_cycles_path):
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)

    keys = [widget.segment_combo.itemData(i)
            for i in range(widget.segment_combo.count())]
    assert keys == [SCOPE_ALL_TIME, SCOPE_ALL_HEATING, SCOPE_ALL_COOLING] + [
        f"segment:{i}" for i in range(1, 13)]
    assert len(widget.run) == 12
    assert widget.segment_combo.isEnabled() is True
    # The item text is `dta_segments.segment_label`, not a second formatter:
    # index, kind, signed rate, T span, cycle and the matched step.
    text = widget.segment_combo.itemText(_scope_index(widget, "segment:3"))
    for part in ("#3", "cool", "-20.0 °C/min", "1299 -> 94 °C", "cycle 1", "steps 5, 6"):
        assert part in text
    assert "(partial)" in widget.segment_combo.itemText(_scope_index(widget, "segment:12"))


def test_multi_segment_file_defaults_to_whole_run_vs_time(qtbot, dta_cycles_path):
    """A cyclic file opens on the whole run, not on segment #1: the first
    thing to know about it is that it IS cyclic."""
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)

    assert widget._scope().kind == SCOPE_ALL_TIME
    assert widget._scope_key == SCOPE_ALL_TIME
    assert widget.x_combo.currentText() == "Time (min)"
    note = widget.scope_note_label.text()
    assert "12 segments" in note and "4 cycle(s)" in note and "890.2 min" in note
    assert "single segment" in note


def test_single_ramp_file_gets_one_scope_entry_and_no_choice(qtbot, dta_example_path):
    """One segment is a statement, not a choice. The combo names it and is
    disabled, and no aggregate is offered as a synonym for "the only one"."""
    widget = DtaWorkspace(records=[_record(dta_example_path)])
    qtbot.addWidget(widget)

    assert widget.segment_combo.count() == 1
    assert widget.segment_combo.itemData(0) == "segment:1"
    assert widget.segment_combo.isEnabled() is False
    assert widget._scope().kind == SCOPE_SEGMENT


def test_segment_choice_is_remembered_per_record(qtbot, dta_cycles_path, dta_example_path):
    widget = DtaWorkspace(records=[_record(dta_cycles_path, rec_id="cyc"),
                                   _record(dta_example_path, rec_id="ramp")])
    qtbot.addWidget(widget)
    _select_scope(widget, "segment:6", qtbot)
    assert widget._scope().segment.index == 6

    widget.record_combo.setCurrentIndex(1)
    _settle(qtbot, widget)
    assert widget._scope().segment.index == 1      # its own and only segment

    widget.record_combo.setCurrentIndex(0)
    _settle(qtbot, widget)
    assert widget._scope_key == "segment:6"
    assert widget.segment_combo.currentData() == "segment:6"


def test_a_remembered_scope_that_this_record_cannot_honour_is_dropped(qtbot, dta_cycles_path):
    """The memory is re-validated, never trusted. A key naming segment #99 is
    meaningless against this record, and the failure mode to avoid is not an
    IndexError -- it is quietly analysing the wrong rows."""
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    widget._scope_memory[widget._record_id] = "segment:99"

    assert widget._remembered_scope_key() == SCOPE_ALL_TIME
    widget._scope_key = "segment:99"
    assert widget._scope().kind == SCOPE_ALL_TIME
    assert widget._resolved_scope_key() == SCOPE_ALL_TIME


# ===========================================================================
# 5. Guardrails -- what is COMPUTABLE, and why it is never a dialog
# ===========================================================================
def _action_buttons(widget):
    return (widget.compute_btn, widget.integrate_btn, widget.calc_max_btn,
            widget.calc_min_btn, widget.export_csv_btn)


def test_tg_buttons_are_disabled_outside_a_single_segment(qtbot, dta_cycles_path):
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)

    for key in (SCOPE_ALL_TIME, SCOPE_ALL_HEATING, SCOPE_ALL_COOLING):
        _select_scope(widget, key, qtbot)
        assert [b.isEnabled() for b in _action_buttons(widget)] == [False] * 5, key
        # ... and the reason is on the button and on the note, not in a dialog.
        assert "single segment" in widget.compute_btn.toolTip()

    for key in ("segment:1", "segment:2", "segment:3"):
        _select_scope(widget, key, qtbot)
        assert [b.isEnabled() for b in _action_buttons(widget)] == [True] * 5, key


def test_the_old_whole_file_tg_collapse_is_unreachable(qtbot, dta_cycles_path):
    """THE headline regression.

    A whole-file compute on this record used to return Double == Parallel ==
    |dY|max == 455.951 with a spread of 1.2e-4, and the app printed "Methods
    agree" over it. The number is an artefact of sorting eight passes over
    19-1299 C into one curve: the three methods agree because they are all
    looking at the same sorted mush, not because they are right.

    Both halves are pinned -- that the deleted path really did produce it, and
    that the reworked widget cannot.
    """
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)

    # (a) what the deleted global-argsort path produced, recomputed here.
    x = pd.to_numeric(widget.df["Temperature (°C)"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(widget.df["Heat Flow (mW)"], errors="coerce").to_numpy(float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    order = np.argsort(x)
    x, y = x[order], y[order]
    bp = resolve_baseline_params(
        manual_enabled=False, low_use_point=False, low_point_x=None,
        low_min=None, low_max=None, high_use_point=False, high_point_x=None,
        high_min=None, high_max=None, slope_min=None, slope_max=None)
    collapsed = [
        compute_tg_double_tangent(x, y, 350.0, 700.0, threshold=0.20, guard_frac=0.0,
                                  smooth_derivative=False, manual_low=bp.low_range,
                                  manual_slope=bp.manual_slope).tg,
        compute_tg_parallel_improved(x, y, 350.0, 700.0, smooth_derivative=False,
                                     manual_low_range=bp.low_range,
                                     manual_high_range=bp.high_range,
                                     manual_low_point=bp.low_point,
                                     manual_high_point=bp.high_point).tg,
        compute_tg_derivative(x, y, 350.0, 700.0, smooth_derivative=False,
                              restrict_range=None),
    ]
    for value in collapsed:
        assert value == pytest.approx(455.951, abs=1e-3)
    assert max(collapsed) - min(collapsed) == pytest.approx(1.2240e-4, rel=1e-3)

    # (b) the reworked widget cannot produce it.
    assert widget.compute_btn.isEnabled() is False
    with pytest.raises(SegmentScopeError):
        widget._get_xy()
    widget._compute()
    assert widget.res_double is None and widget.res_parallel is None
    assert widget.tg_deriv is None
    assert "455.95" not in widget.result_label.text()
    assert "single segment" in widget.result_label.text()


def test_an_uncomputable_scope_never_opens_a_dialog(qtbot, dta_cycles_path, monkeypatch):
    """"Warnings, never refusals" has a corollary: a warning that has to be
    clicked away is a warning that gets clicked away. Every guardrail message
    is standing text in the UI, never a modal."""
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)

    def _boom(*args, **kwargs):
        raise AssertionError("a modal dialog was opened by a guardrail")

    for name in ("critical", "warning", "information", "question"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(_boom))

    for key in (SCOPE_ALL_TIME, SCOPE_ALL_HEATING, SCOPE_ALL_COOLING):
        _select_scope(widget, key, qtbot)
        widget._compute()
        widget._calc_integrate()
        widget._calc_find_max()
        widget._calc_find_min()
        assert "⚠" in widget.result_label.text()
        assert "⚠" in widget.calc_result_label.text()


def test_selecting_a_hold_forces_the_time_axis(qtbot, dta_cycles_path):
    """Temperature is not an abscissa on a hold. `prepare_segment_xy` refuses
    the combination; the UI's job is to stop the user reaching that refusal."""
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    _select_scope(widget, "segment:2", qtbot)

    seg = widget._current_segment()
    assert seg.is_hold
    assert widget._scope().x_kind == "t"
    assert widget.x_combo.currentText() == "Time (min)"

    model = widget.x_combo.model()
    blocked = {widget.x_combo.itemText(i): model.item(i).isEnabled()
               for i in range(widget.x_combo.count())}
    assert blocked["Temperature (°C)"] is False
    assert blocked["Time (min)"] is True
    assert blocked["Heat Flow (mW)"] is True
    tip = model.item(_x_index(widget, "Temperature (°C)")).toolTip()
    assert "hold" in tip and "abscissa" in tip

    # Nothing is BLOCKED from being processed: the hold still computes, on the
    # only abscissa it has.
    assert widget.compute_btn.isEnabled() is True
    x, y = widget._get_xy()[:2]
    assert x.size > 10 and bool(np.all(np.diff(x) > 0))
    assert "not an abscissa on a hold" in widget.scope_note_label.text()


def test_leaving_an_aggregate_hands_the_temperature_abscissa_back(qtbot, dta_cycles_path):
    """The forced clock must not be a one-way door: every ramp selected after
    the whole-run view would otherwise be plotted against time, which is not
    the abscissa a Tg is read on."""
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    assert widget.x_combo.currentText() == "Time (min)"

    _select_scope(widget, "segment:3", qtbot)
    assert widget.x_combo.currentText() == "Temperature (°C)"
    model = widget.x_combo.model()
    assert all(model.item(i).isEnabled() for i in range(widget.x_combo.count()))


def test_a_user_chosen_x_column_survives_a_scope_change(qtbot, dta_cycles_path):
    """A column the user picked is theirs. A later scope change may force the
    clock where the maths demands it, but it must not quietly undo a choice."""
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    _select_scope(widget, "segment:1", qtbot)

    widget.x_combo.setCurrentText("Time (min)")     # the user's own choice
    _settle(qtbot, widget)
    assert widget._x_forced_time is False

    _select_scope(widget, "segment:4", qtbot)
    assert widget.x_combo.currentText() == "Time (min)"


def test_the_default_window_lands_inside_the_selected_segment(qtbot, dta_cycles_path):
    """The historic 350-700 box is meaningless on a branch that never went
    above 300; a window outside the data is what produces "Baseline range has
    too few points" three dialogs later."""
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    _select_scope(widget, "segment:3", qtbot)

    seg = widget._current_segment()
    lo, hi = float(widget.xmin_edit.text()), float(widget.xmax_edit.text())
    assert seg.T_min <= lo < hi <= seg.T_max
    assert hi - lo == pytest.approx(0.60 * (seg.T_max - seg.T_min), rel=1e-5)
    assert widget._window_user_edited is False


def test_a_typed_window_is_never_overwritten_by_a_segment_default(qtbot, dta_cycles_path):
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    _select_scope(widget, "segment:1", qtbot)

    widget.xmin_edit.setText("420")
    widget.xmin_edit.textEdited.emit("420")          # what typing emits
    widget.xmax_edit.setText("560")
    widget.xmax_edit.textEdited.emit("560")
    assert widget._window_user_edited is True

    _select_scope(widget, "segment:4", qtbot)
    assert (widget.xmin_edit.text(), widget.xmax_edit.text()) == ("420", "560")


def test_the_default_window_never_fires_on_a_single_ramp_file(qtbot, dta_example_path):
    """R3's second failure mode, pinned on its own. The auto window would
    clamp 700 to 388.71 on this file and move the pinned Tg."""
    widget = DtaWorkspace(records=[_record(dta_example_path)])
    qtbot.addWidget(widget)
    assert (widget.xmin_edit.text(), widget.xmax_edit.text()) == ("350", "700")

    widget._apply_default_window()                   # called directly: still a no-op
    assert (widget.xmin_edit.text(), widget.xmax_edit.text()) == ("350", "700")


# ===========================================================================
# 6. Computing inside one segment
# ===========================================================================
def test_single_segment_scope_computes_finite_tg(qtbot, dta_cycles_path):
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    _select_scope(widget, "segment:1", qtbot)
    widget._compute()

    seg = widget._current_segment()
    values = [widget.res_double.tg, widget.res_parallel.tg, widget.tg_deriv]
    assert all(np.isfinite(v) for v in values)
    assert all(seg.T_min <= v <= seg.T_max for v in values)
    # Not the collapsed whole-file answer, and not a degenerate agreement.
    assert max(values) - min(values) > 0.01
    assert all(abs(v - 455.951) > 1.0 for v in values)


def test_find_max_is_attributed_to_its_segment(qtbot, dta_cycles_path):
    """The number today's whole-file path reports with no indication that it
    came off a COOLING branch of the first cycle."""
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    _select_scope(widget, "segment:3", qtbot)

    widget.calc_xmin_edit.setText("350")
    widget.calc_xmax_edit.setText("700")
    widget._calc_find_max()

    text = widget.calc_result_label.text()
    assert "35.136" in text and "434.328" in text
    assert "seg #3" in text and "cool" in text and "cycle 1" in text


def test_a_single_ramp_result_is_not_cluttered_with_an_attribution(qtbot, dta_example_path):
    """Nothing to attribute on a one-segment record, so nothing is said."""
    widget = DtaWorkspace(records=[_record(dta_example_path)])
    qtbot.addWidget(widget)
    widget._compute()
    assert widget._scope_attribution() == ""
    assert "seg #" not in widget.result_label.text()


# ===========================================================================
# 7. Render modes  (SPEC_cycles Phase B step 8)
# ===========================================================================
def test_whole_run_view_is_temperature_against_time_with_a_span_per_segment(
        qtbot, dta_cycles_path):
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    ax = widget.plot.ax

    assert ax.get_xlabel() == "Time (min)"
    assert ax.get_ylabel() == "Temperature (°C)"
    assert len(ax.patches) == 12                      # one shaded span per segment
    texts = " ".join(t.get_text() for t in ax.texts)
    for i in range(1, 13):
        assert f"#{i}" in texts
    assert widget.ax2 is not None                     # the signal is still visible
    # No prepared curve exists here, so none is published: a leftover pair
    # would let `_draw_overlays` mark another segment's Tg onto this picture.
    assert widget._x is None and widget._y is None


def test_all_heating_view_draws_one_curve_per_heating_segment(qtbot, dta_cycles_path):
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    _select_scope(widget, SCOPE_ALL_HEATING, qtbot)
    ax = widget.plot.ax

    assert len(ax.lines) == 4
    labels = [ln.get_label() for ln in ax.lines]
    assert [lab.split()[0] for lab in labels] == ["#1", "#4", "#7", "#10"]
    # Each curve is sorted WITHIN itself and none across: that is the whole
    # difference between this picture and the global argsort.
    for line, seg in zip(ax.lines, widget.run.by_kind("heat")):
        xd = np.asarray(line.get_xdata(), dtype=float)
        assert bool(np.all(np.diff(xd) > 0))
        assert seg.T_min - 1e-6 <= xd.min() and xd.max() <= seg.T_max + 1e-6
    assert widget._x is None and widget._y is None


def test_all_cooling_view_draws_one_curve_per_cooling_segment(qtbot, dta_cycles_path):
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    _select_scope(widget, SCOPE_ALL_COOLING, qtbot)

    assert len(widget.plot.ax.lines) == 4
    assert [ln.get_label().split()[0] for ln in widget.plot.ax.lines] == \
        ["#3", "#6", "#9", "#12"]


def test_single_segment_view_is_one_curve_against_the_chosen_x(qtbot, dta_cycles_path):
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    _select_scope(widget, "segment:6", qtbot)
    ax = widget.plot.ax

    assert len(ax.lines) == 1
    assert ax.get_xlabel() == "Temperature (°C)"
    xd = np.asarray(ax.lines[0].get_xdata(), dtype=float)
    seg = widget._current_segment()
    assert bool(np.all(np.diff(xd) > 0))
    assert seg.T_min - 1e-6 <= xd.min() and xd.max() <= seg.T_max + 1e-6
    assert widget._x is not None and widget._x.size == xd.size


def test_a_multi_segment_record_draws_a_curve_on_every_scope(qtbot, dta_cycles_path):
    """No scope may leave the canvas blank. Step 6 left every multi-segment
    scope drawing nothing at all; that is the window this step closes."""
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    for i in range(widget.segment_combo.count()):
        widget.segment_combo.setCurrentIndex(i)
        _settle(qtbot, widget)
        key = widget.segment_combo.itemData(i)
        assert len(widget.plot.ax.lines) >= 1, f"no curve drawn for {key}"


# ===========================================================================
# 8. Direction safety in the overlays
# ===========================================================================
def test_interp_ascending_survives_a_descending_abscissa():
    """`np.interp` REQUIRES ascending xp and does not check: handed a
    descending array it returns an endpoint silently, so a Tg marker would sit
    at the end of the curve looking like a measurement."""
    x = np.array([900.0, 700.0, 500.0, 300.0, 100.0])
    y = np.array([9.0, 7.0, 5.0, 3.0, 1.0])

    assert _interp_ascending(400.0, x, y) == pytest.approx(4.0)
    assert np.interp(400.0, x, y) != pytest.approx(4.0)     # the old call
    assert np.isnan(_interp_ascending(400.0, np.array([]), np.array([])))
    # Ascending input is left exactly as np.interp would have it.
    assert _interp_ascending(400.0, x[::-1], y[::-1]) == pytest.approx(4.0)


def test_draw_overlays_falls_back_to_nanmin_nanmax_not_first_and_last(
        qtbot, dta_example_path):
    """With no usable window the tangent lines span the data. `x[0], x[-1]`
    inverts that span the moment the view runs backwards -- a bug this rework
    EXPOSES (the deleted global argsort made every view ascending by
    accident), not one it adds."""
    widget = DtaWorkspace(records=[_record(dta_example_path)])
    qtbot.addWidget(widget)
    widget._compute()
    widget.method_double.setChecked(True)
    widget.explicit_check.setChecked(True)
    _settle(qtbot, widget)

    x_asc, y_asc = widget._x.copy(), widget._y.copy()
    tg = widget.res_double.tg
    widget._x, widget._y = x_asc[::-1].copy(), y_asc[::-1].copy()
    widget.xmin_edit.setText("")                     # force the fallback branch

    widget.plot.ax.clear()
    widget._draw_overlays(dy_col="(same)")

    # Two-point lines that are not vertical: the three tangent constructions.
    # `axvline` also draws two points, at one x, and is not what is being
    # pinned here.
    spans = [xd for xd in (np.asarray(ln.get_xdata(), dtype=float)
                           for ln in widget.plot.ax.lines)
             if xd.size == 2 and xd[0] != xd[1]]
    assert spans, "the explicit tangent lines were not drawn"
    for xd in spans:
        assert xd[0] == pytest.approx(float(np.nanmin(x_asc)))
        assert xd[1] == pytest.approx(float(np.nanmax(x_asc)))
        assert xd[0] < xd[1]

    markers = [ln for ln in widget.plot.ax.lines if len(ln.get_xdata()) == 1]
    assert markers, "the Tg marker was not drawn"
    assert float(markers[-1].get_ydata()[0]) == pytest.approx(
        float(np.interp(tg, x_asc, y_asc)))


# ===========================================================================
# 9. Signal storms and debounce discipline  (Risk R5)
# ===========================================================================
def test_arrowing_through_the_scope_list_coalesces_into_one_render(
        qtbot, dta_cycles_path):
    """Holding the arrow key down through a 15-entry list must not redraw a
    4749-point record fifteen times. Every scope change goes through
    `plot.request_redraw`, so the count follows the debounce FLUSHES, not the
    index changes."""
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    _settle(qtbot, widget)

    calls = []
    widget._render = lambda: calls.append(1)

    for i in range(1, 6):
        widget.segment_combo.setCurrentIndex(i)
    assert calls == []               # five index changes, nothing drawn yet
    _settle(qtbot, widget)
    assert len(calls) == 1           # one flush

    for i in range(6, 9):
        widget.segment_combo.setCurrentIndex(i)
        _settle(qtbot, widget)              # each one allowed to settle
    assert len(calls) == 4


def test_rebuilding_the_scope_combo_does_not_fire_the_handler(qtbot, dta_cycles_path):
    """`clear()` and `addItem()` each emit `currentIndexChanged`, and the
    handler rewrites the window, re-runs the guardrails and redraws."""
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    _select_scope(widget, "segment:5", qtbot)

    calls = []
    widget._render = lambda: calls.append(1)
    widget._refresh_segment_combo()
    # A fixed wait ON PURPOSE: this asserts that NOTHING was drawn, and
    # `_settle` returns immediately when the queue is already empty, which
    # would make the assertion vacuous.
    qtbot.wait(300)

    assert calls == []
    assert widget._scope_key == "segment:5"
    assert widget.segment_combo.currentData() == "segment:5"
    assert widget._suspend_scope is False


def test_save_figure_png_flushes_a_queued_render(qtbot, dta_cycles_path, tmp_path):
    """A render queued up to 120 ms ago has not run yet. Saved without
    flushing it, the PNG shows the previous scope -- and the file, unlike the
    screen, is what leaves the building."""
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    _settle(qtbot, widget)

    widget.segment_combo.setCurrentIndex(_scope_index(widget, "segment:3"))
    assert widget.plot._pending is not None          # queued, not drawn

    out = tmp_path / "seg3.png"
    widget.save_figure_png(out)

    assert widget.plot._pending is None
    assert out.is_file() and out.stat().st_size > 0
    assert any("#3 cool" in t.get_text() for t in widget.plot.ax.texts)


def test_png_records_its_segment(qtbot, dta_cycles_path, dta_example_path, tmp_path):
    """A PNG of segment #3 of a four-cycle run is indistinguishable from a PNG
    of the whole file unless the figure says so."""
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    _select_scope(widget, "segment:3", qtbot)
    widget.save_figure_png(tmp_path / "cyc.png")

    stamp = [t.get_text() for t in widget.plot.ax.texts if "#3" in t.get_text()]
    assert stamp, [t.get_text() for t in widget.plot.ax.texts]
    assert "#3 cool" in stamp[0] and "cycle 1" in stamp[0]
    assert "target missed" in stamp[0]               # the warning travels too

    # A single-segment record has nothing to disambiguate, so it is not
    # stamped -- the watermark must mean something when it appears.
    single = DtaWorkspace(records=[_record(dta_example_path)])
    qtbot.addWidget(single)
    single.save_figure_png(tmp_path / "ramp.png")
    assert not any("#1" in t.get_text() for t in single.plot.ax.texts)


def test_changing_the_scope_drops_the_previous_segment_s_result(qtbot, dta_cycles_path):
    """A Tg measured on segment 1 must not stay on screen over segment 4.

    Per-segment RESULTS are Phase B step 11; until then the conservative half
    is in force, because no number beats another segment's number shown
    without a word.
    """
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    _select_scope(widget, "segment:1", qtbot)
    widget._compute()
    first = widget.res_parallel.tg
    assert np.isfinite(first)

    _select_scope(widget, "segment:4", qtbot)
    assert widget.res_double is None and widget.res_parallel is None
    assert widget.tg_deriv is None
    assert widget.result_label.text() == "Tg: —"
    overlay = " ".join(t.get_text() for t in widget.plot.ax.texts)
    assert f"{first:.2f}" not in overlay


# ===========================================================================
# 10. The full-resolution real files  (skipped where the folder is absent)
# ===========================================================================
@requires_dta_bm
def test_bm0_two_segment_file_offers_no_singleton_aggregate(qtbot, dta_bm_dir):
    """`BM-0.txt` is one heating ramp plus a trailing hold: 2 segments, so it
    gets the whole-run view and two segment rows -- and NOT "All heating
    segments (1)", which would be segment #1 listed twice under two names.
    """
    widget = DtaWorkspace(records=[_record(dta_bm_dir / "BM-0.txt")])
    qtbot.addWidget(widget)

    assert [s.kind for s in widget.run.segments] == ["heat", "isothermal"]
    assert [widget.segment_combo.itemData(i)
            for i in range(widget.segment_combo.count())] == [
        SCOPE_ALL_TIME, "segment:1", "segment:2"]
    assert widget._scope().kind == SCOPE_ALL_TIME
    assert widget.x_combo.currentText() == "Time (min)"
    assert len(widget.plot.ax.patches) == 2

    _select_scope(widget, "segment:1", qtbot)
    assert widget.x_combo.currentText() == "Temperature (°C)"
    assert widget.compute_btn.isEnabled() is True
    widget._compute()
    seg = widget._current_segment()
    for value in (widget.res_double.tg, widget.res_parallel.tg, widget.tg_deriv):
        assert seg.T_min <= value <= seg.T_max


@requires_dta_bm
def test_the_full_19k_row_cyclic_run_renders_every_scope(qtbot, dta_bm_dir):
    """The committed fixture is the real run decimated 4x. This is the real
    run: 18996 rows, and the one place the render modes meet the point count
    the debounce exists for."""
    widget = DtaWorkspace(records=[_record(dta_bm_dir / "BM-0-cycles.txt")])
    qtbot.addWidget(widget)

    assert len(widget.run) == 12 and widget.run.n_cycles == 4
    assert "".join(s.kind[0].upper() for s in widget.run.segments) == "HICHICHICHIC"
    assert len(widget.plot.ax.patches) == 12

    for i in range(widget.segment_combo.count()):
        widget.segment_combo.setCurrentIndex(i)
        _settle(qtbot, widget)
        key = widget.segment_combo.itemData(i)
        assert len(widget.plot.ax.lines) >= 1, f"nothing drawn for {key}"

    _select_scope(widget, SCOPE_ALL_HEATING, qtbot)
    assert len(widget.plot.ax.lines) == 4
    for line in widget.plot.ax.lines:
        assert bool(np.all(np.diff(np.asarray(line.get_xdata(), float)) > 0))
