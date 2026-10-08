"""Rate-series export, reload and `.prism` persistence (build step D7).

Four channels carry a rate series out of the widget and two bring it back:

  * ``rate_export_rows_csv``  one line per row, warnings included
  * ``rate_export_fit_csv``   one line per model the present rows can evaluate
  * ``rate_save_plot_png``    the figure, with its annotation drawn INTO it
  * ``rate_load_rows_csv``    the rows CSV, read back
  * ``rate_series_records`` / ``restore_rate_series``  the ``.prism`` v4 section

Every one is a method taking a path, with the dialog in a separate thin
wrapper, so nothing here monkeypatches ``QFileDialog`` to exercise the writing.

The rule these tests exist to defend: a stored value is a MEASUREMENT, not a
cache. Neither a CSV reload nor a project reopen may re-measure it, because
re-measuring with whatever settings happen to be current would silently change
a number somebody has already written down.
"""
from __future__ import annotations

import numpy as np
import pytest

from PySide6.QtWidgets import QMessageBox

import core.project_io as project_io
from dta.qt_dta import DtaWorkspace, RATE_TAB_TITLE
from core.qt_shell import _load_spectrum_from_path


def _record(path, *, rec_id=None):
    spectrum = _load_spectrum_from_path(str(path))
    rec = {"title": spectrum.title, "path": spectrum.path,
           "df": spectrum.df, "meta": spectrum.meta}
    if rec_id is not None:
        rec["id"] = rec_id
    return rec


def _tab_index(widget, title):
    return [widget.tabs.tabText(i) for i in range(widget.tabs.count())].index(title)


def _cycles(qtbot, path):
    """The cyclic fixture with its four heating rows added and measured."""
    widget = DtaWorkspace(records=[_record(path, rec_id="cyc-1")])
    qtbot.addWidget(widget)
    widget.tabs.setCurrentIndex(_tab_index(widget, RATE_TAB_TITLE))
    widget.rate_add_segments()
    widget.rate_compute_all()
    return widget


def _state(widget):
    """The parts of each row a round trip must preserve exactly."""
    return [(r.q_C_per_min, r.value_C, r.include, r.value_source, r.method,
             r.quantity_kind, r.window_C) for r in widget.rate_rows]


def _same(a, b):
    assert len(a) == len(b), (len(a), len(b))
    for x, y in zip(a, b):
        for i, (u, v) in enumerate(zip(x, y)):
            if isinstance(u, float) and isinstance(v, float):
                assert u == v or (u != u and v != v), (i, x, y)
            else:
                assert u == v, (i, x, y)


# ===========================================================================
# The series a scientist actually has
# ===========================================================================

def test_the_cyclic_series_measures_three_of_four_rows_and_still_fits(
        qtbot, dta_cycles_path):
    """The state every test below starts from, asserted once.

    Three heats measure, heat 2 fails because its derivative peak lands on the
    700 °C window edge (after cycle 1 the specimen is largely crystallized and
    has no glass transition there), and a fit IS produced from the other three.
    """
    widget = _cycles(qtbot, dta_cycles_path)
    assert len(widget.rate_rows) == 4
    assert [r.include for r in widget.rate_rows] == [True] * 4
    assert sum(1 for r in widget.rate_rows if r.value_source == "computed") == 3
    failed = [r for r in widget.rate_rows if r.value_source == "failed"]
    assert len(failed) == 1
    assert failed[0].error == "Baseline range has too few points."

    fit = widget.rate_last_fit
    assert fit is not None and fit.ok is True
    assert fit.n == 3
    assert np.isfinite(fit.lasocka_B)
    codes = {w.code for w in widget.rate_warnings}
    assert "too_few_points" not in codes
    assert {"thermal_history", "single_specimen_cycles", "no_value"} <= codes


# ===========================================================================
# Rows CSV
# ===========================================================================

def test_the_rows_csv_names_its_units_in_its_column_names(qtbot, dta_cycles_path,
                                                         tmp_path):
    """`value_C` and `value_K` both, and no `temp_unit` column.

    The older per-segment export carried a `temp_unit` column whose meaning
    depended on a radio button at export time, so a file read a year later could
    not be trusted without knowing which way that button had been set. A unit in
    the column NAME cannot be misread.
    """
    widget = _cycles(qtbot, dta_cycles_path)
    out = tmp_path / "rows.csv"
    assert widget.rate_export_rows_csv(out) == 4
    head = out.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert "value_C" in head and "value_K" in head
    assert "temp_unit" not in head and "value_unit" not in head


def test_every_row_of_the_csv_carries_warnings_and_none_looks_innocent(
        qtbot, dta_cycles_path, tmp_path):
    """A single-row extract must not look clean.

    Someone pulling one line out of this file into a table for a paper has to
    carry its caveats with it, so every row names the codes that implicate it
    plus the series-level ones, rather than leaving them in a header comment
    nobody copies. Row-scoped codes stay row-scoped -- `thermal_history` names
    the rows in the fit, and the failed row is not one of them -- but no row
    comes out with an empty code list or a clean severity.
    """
    widget = _cycles(qtbot, dta_cycles_path)
    out = tmp_path / "rows.csv"
    widget.rate_export_rows_csv(out)
    import csv
    with out.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 4
    for rec in rows:
        codes = [c for c in rec["warning_codes"].split(";") if c]
        assert codes, rec
        assert rec["max_severity"] == "critical"
        # The series-level facts reach every row, fitted or not.
        assert "multi_rate_source" in codes and "low_r2" in codes
        assert rec["warning_messages"].strip()

    fitted = [r for r in rows if r["fitted"] in ("True", "1", "true")]
    assert len(fitted) == 3
    for rec in fitted:
        assert "thermal_history" in rec["warning_codes"]

    failed = [r for r in rows if r["value_source"] == "failed"]
    assert len(failed) == 1
    # The reason travels verbatim, not only as a code.
    assert failed[0]["error"] == "Baseline range has too few points."
    assert "no_value" in failed[0]["warning_codes"]


def test_the_failed_row_is_in_the_csv_not_omitted_from_it(qtbot, dta_cycles_path,
                                                          tmp_path):
    """Four rows in the table, four lines in the file.

    A row that silently vanished on export would let a reader conclude the
    series was three points by design rather than four with one that failed.
    """
    widget = _cycles(qtbot, dta_cycles_path)
    out = tmp_path / "rows.csv"
    widget.rate_export_rows_csv(out)
    body = [ln for ln in out.read_text(encoding="utf-8").splitlines()[1:] if ln.strip()]
    assert len(body) == 4


# ===========================================================================
# Fit CSV and PNG
# ===========================================================================

def test_the_fit_csv_holds_one_line_per_evaluable_model(qtbot, dta_cycles_path,
                                                        tmp_path):
    """Lasocka, Moynihan and the Kissinger form on Tg — side by side.

    They share ONE `check_series` pass, so a reviewer comparing the lines is
    comparing models rather than two independently recomputed opinions about the
    same rows.
    """
    widget = _cycles(qtbot, dta_cycles_path)
    out = tmp_path / "fit.csv"
    n = widget.rate_export_fit_csv(out)
    assert n >= 3
    import csv
    with out.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == n
    methods = {r["method"] for r in rows}
    assert "lasocka" in methods
    for rec in rows:
        assert rec["warning_codes"], rec["method"]


def test_the_saved_png_is_written_and_carries_the_annotation(qtbot,
                                                             dta_cycles_path,
                                                             tmp_path):
    """The annotation is drawn into the axes, which is the only reason a saved
    figure carries its own caveats."""
    widget = _cycles(qtbot, dta_cycles_path)
    out = tmp_path / "plot.png"
    widget.rate_save_plot_png(out)
    assert out.stat().st_size > 5000
    drawn = [t.get_text() for t in widget.rate_plot.ax.texts]
    assert widget.rate_annotation_text in drawn


# ===========================================================================
# Reload
# ===========================================================================

def test_a_csv_round_trip_restores_the_rows_without_re_measuring(qtbot,
                                                                 dta_cycles_path,
                                                                 tmp_path):
    """Rates, values, include flags and `value_source` all identical.

    `value_source` in particular: a reloaded row must still say whether its
    number was computed or typed, and the row that FAILED must come back failed
    rather than blank, or the reader loses the distinction between "no value
    here" and "this one could not be measured".
    """
    widget = _cycles(qtbot, dta_cycles_path)
    before = _state(widget)
    out = tmp_path / "rows.csv"
    widget.rate_export_rows_csv(out)

    n, _warns = widget.rate_load_rows_csv(out)
    assert n == 4
    _same(before, _state(widget))
    assert any(r.value_source == "failed" for r in widget.rate_rows)


def test_reloading_mints_fresh_uids(qtbot, dta_cycles_path, tmp_path):
    """Loading one CSV twice must not produce rows every uid-keyed lookup
    confuses with each other."""
    widget = _cycles(qtbot, dta_cycles_path)
    out = tmp_path / "rows.csv"
    widget.rate_export_rows_csv(out)
    first = {r.uid for r in widget.rate_rows}
    widget.rate_load_rows_csv(out)
    second = {r.uid for r in widget.rate_rows}
    assert len(second) == 4
    assert not (first & second)


def test_loading_a_csv_says_the_values_are_as_saved(qtbot, dta_cycles_path,
                                                    tmp_path):
    """The user has to be told the numbers were not re-measured, or they will
    assume the current window produced them."""
    widget = _cycles(qtbot, dta_cycles_path)
    out = tmp_path / "rows.csv"
    widget.rate_export_rows_csv(out)
    widget.rate_load_rows_csv(out)
    note = widget.rate_io_note.text()
    assert "as saved" in note
    assert "Compute all values" in note


def test_no_export_or_reload_path_opens_a_modal_dialog(qtbot, dta_cycles_path,
                                                       tmp_path, monkeypatch):
    """The direct methods are dialog-free; only the `_on_*` wrappers may talk."""
    widget = _cycles(qtbot, dta_cycles_path)
    for name in ("critical", "warning", "information", "question"):
        monkeypatch.setattr(QMessageBox, name,
                            lambda *a, **k: pytest.fail("a dialog was opened"))
    widget.rate_export_rows_csv(tmp_path / "a.csv")
    widget.rate_export_fit_csv(tmp_path / "b.csv")
    widget.rate_save_plot_png(tmp_path / "c.png")
    widget.rate_load_rows_csv(tmp_path / "a.csv")


# ===========================================================================
# .prism v4
# ===========================================================================

def test_the_project_format_is_v4(qtbot):
    """The bump happens once, and carries only the rate series."""
    assert project_io.PROJECT_VERSION == 4


def test_a_project_round_trip_restores_the_series_without_re_measuring(
        qtbot, dta_cycles_path, tmp_path):
    """Rates, values, `value_source`, method and window all survive.

    The window and method are stored per row precisely so a reopened project
    shows the numbers that were saved rather than whatever the current settings
    would now produce.
    """
    widget = _cycles(qtbot, dta_cycles_path)
    before = _state(widget)
    recs = widget.rate_series_records()
    assert len(recs) == 1 and len(recs[0]["rows"]) == 4

    path = tmp_path / "p.prism"
    project_io.save_project(str(path), [], {}, dta_rate_series=recs)
    loaded = project_io.load_project(str(path))
    assert len(loaded.dta_rate_series) == 1

    other = DtaWorkspace()
    qtbot.addWidget(other)
    assert other.restore_rate_series(loaded.dta_rate_series) == 4
    _same(before, _state(other))


def test_an_empty_series_writes_no_section_at_all(qtbot, tmp_path):
    """A project made without ever opening the tab must not grow a section."""
    widget = DtaWorkspace()
    qtbot.addWidget(widget)
    assert widget.rate_series_records() == []
    path = tmp_path / "empty.prism"
    project_io.save_project(str(path), [], {},
                            dta_rate_series=widget.rate_series_records())
    import zipfile
    with zipfile.ZipFile(path) as zf:
        assert "dta_rate_series.json" not in zf.namelist()


def test_a_v3_project_still_opens_and_leaves_the_tab_empty(qtbot, tmp_path):
    """Every project saved before this feature existed has no such section, and
    a missing section must read as "no series" rather than failing the load."""
    path = tmp_path / "old.prism"
    project_io.save_project(str(path), [], {})
    loaded = project_io.load_project(str(path))
    assert loaded.dta_rate_series == []
    widget = DtaWorkspace()
    qtbot.addWidget(widget)
    assert widget.restore_rate_series(loaded.dta_rate_series) == 0
    assert widget.rate_rows == []


def test_restoring_clears_a_series_left_over_from_another_project(
        qtbot, dta_cycles_path, tmp_path):
    """Opening a project with no series must not carry the previous one's rows
    into the newly opened work."""
    widget = _cycles(qtbot, dta_cycles_path)
    assert len(widget.rate_rows) == 4
    assert widget.restore_rate_series([]) == 0
    assert widget.rate_rows == []


@pytest.mark.parametrize("junk", [
    [{"rows": "not a list"}],
    ["garbage"],
    [42],
    [None],
])
def test_a_malformed_saved_series_is_absorbed_not_raised(qtbot, junk):
    """Every other workspace has already been restored by the time this runs, so
    a bad section must not take the whole project load down with it."""
    widget = DtaWorkspace()
    qtbot.addWidget(widget)
    assert widget.restore_rate_series(junk) == 0
    assert widget.rate_rows == []


def test_a_row_of_unrecognised_fields_survives_as_an_empty_row(qtbot):
    """Tolerant, not lossy: an unreadable row is kept blank, not dropped.

    A row whose fields this version does not recognise (a project written by a
    newer PRISM, say) comes back with no rate and no value. That is deliberately
    NOT an error and deliberately NOT a silent deletion: the row is visible in
    the table for the user to complete or remove, and it cannot reach a fit,
    because a blank row is not fittable.
    """
    widget = DtaWorkspace()
    qtbot.addWidget(widget)
    n = widget.restore_rate_series(
        [{"schema_version": 9999, "rows": [{"nonsense": True}]}])
    assert n == 1
    assert widget.rate_series.fitted_rows() == []
    row = widget.rate_rows[0]
    assert not np.isfinite(row.q_C_per_min)
    assert not np.isfinite(row.value_C)
