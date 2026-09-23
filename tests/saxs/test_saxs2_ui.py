"""UI wiring tests for the SAXS 2.0 tabs (Frames, Unified fit, Series figures) of SaxsWorkspace.
qt_worker runs synchronously under tests (conftest), dialogs are patched, draws are flushed with
qtbot.wait(20). Synthetic frames and curves only."""
from __future__ import annotations

import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from saxs.core import edf_sectors as es
from saxs.qt_saxs import SaxsWorkspace

import saxs2_synthetic as syn  # noqa: E402  (tests/saxs is on sys.path under pytest)
from test_unified_levels import poisson_curve, true_curve  # noqa: E402

pytestmark = pytest.mark.skipif(not es.HAVE_FABIO, reason="fabio not installed")
FAST = {"rg1_max": "1500", "rg2_max": "200"}


def _fast_options(widget):
    widget.uf_edits["q_max"].setText("0.2")


def test_saxs2_tabs_registered_and_documented(qtbot):
    from core.qt_help import HELP_HTML, MODULE_GUIDES
    widget = SaxsWorkspace()
    qtbot.addWidget(widget)
    labels = [widget.tabs.tabText(i) for i in range(widget.tabs.count())]
    for name in ("Frames", "Unified fit", "Series figures"):
        assert name in labels
        assert name in HELP_HTML and name in MODULE_GUIDES["SAXS/WAXS"]
    assert labels.index("Composite fit") < labels.index("Frames") < labels.index("Unified fit") < labels.index("Series figures")
    # the composite presets now offer the two-level family
    presets = [widget.comp_preset_combo.itemText(i) for i in range(widget.comp_preset_combo.count())]
    assert "L1_L2_T" in presets


@pytest.fixture
def campaign(tmp_path):
    root = str(tmp_path / "frames")
    syn.make_campaign(root)
    pd.DataFrame({"sample": ["S1", "S2", "S3_0p5mm", "S4"], "mu_mm": [10.0, 10.0, 20.0, 5.0]}).to_csv(tmp_path / "samples.csv", index=False)
    return root, str(tmp_path / "samples.csv")


def test_frames_tab_scans_pairs_reduces_and_exports(qtbot, campaign, tmp_path):
    root, samples = campaign
    widget = SaxsWorkspace()
    qtbot.addWidget(widget)
    with patch("saxs.qt_saxs2.QMessageBox.warning") as warn:
        widget.fr_scan()                                   # no root yet
    warn.assert_called_once()
    widget.fr_roots_edit.setText(root)
    widget.fr_scan()
    qtbot.wait(20)
    assert widget.fr_inventory_table.rowCount() == 8
    roles = {widget.fr_inventory_table.item(r, 1).text() for r in range(8)}
    assert roles == {"sample", "empty", "calibrant"}
    assert "4 pair(s)" in widget.fr_status.text() and widget.fr_reduce_btn.isEnabled()
    assert widget.fr_progress.value() == 100
    # selecting a row draws the diagnostics (crop + profiles)
    widget.fr_inventory_table.selectRow(2)
    qtbot.wait(200)
    assert len(widget.fr_diag_plot.figure.get_axes()) == 2
    # reduce with a samples table and an output folder
    out = str(tmp_path / "curves")
    widget.fr_samples_edit.setText(samples)
    widget.fr_outdir_edit.setText(out)
    widget.fr_reduce()
    qtbot.wait(20)
    assert widget.fr_results_table.rowCount() == 4
    classes = {widget.fr_results_table.item(r, 0).text(): widget.fr_results_table.item(r, 6).text() for r in range(4)}
    assert classes["S2"] == "opaque" and classes["S1"] == "measurable"
    assert sorted(os.listdir(out)) == ["S1__sectors.dat", "S2__sectors.dat", "S3_0p5mm__sectors.dat", "S4__sectors.dat"]
    names = [c.name for c in widget.curves]
    assert names == ["S1", "S2", "S3_0p5mm", "S4"]
    assert widget.curves[0].metadata["per_mm"] is True and widget.uf_combo.count() == 4
    # reducing again replaces the curves instead of duplicating them
    widget.fr_reduce()
    qtbot.wait(20)
    assert [c.name for c in widget.curves] == ["S1", "S2", "S3_0p5mm", "S4"]
    summary = str(tmp_path / "summary.csv")
    with patch("saxs.qt_saxs2.QFileDialog.getSaveFileName", return_value=(summary, "")):
        widget.fr_export_summary()
    df = pd.read_csv(summary)
    assert len(df) == 4 and "class_mu_t" in df.columns
    # a bad samples table is reported, not raised
    widget.fr_samples_edit.setText(str(tmp_path / "nope.csv"))
    with patch("saxs.qt_saxs2.QMessageBox.warning") as warn:
        widget.fr_reduce()
    warn.assert_called_once()


@pytest.fixture
def fitted_widget(qtbot):
    widget = SaxsWorkspace()
    qtbot.addWidget(widget)
    widget.add_curve(poisson_curve(true_curve(), name="two", seed=1))
    widget.add_curve(poisson_curve(true_curve(l2=False, tail=False), name="one", seed=2))
    _fast_options(widget)
    return widget


def test_unified_tab_fit_robustness_batch_and_files(qtbot, fitted_widget, tmp_path):
    widget = fitted_widget
    with patch("saxs.qt_saxs2.QMessageBox.warning") as warn:
        widget.uf_combo.setCurrentIndex(-1)
        widget.uf_fit()
    warn.assert_called_once()
    widget.uf_combo.setCurrentText("two")
    widget.uf_fit()
    qtbot.wait(20)
    r = widget._uf_last
    assert r is not None and r.fitted and r.l2
    assert widget.uf_param_table.rowCount() == len(r.params) >= 9
    assert "model L1_L2" in widget.uf_report.toPlainText() and "Level 2" in widget.uf_report.toPlainText()
    axes = widget.uf_plot.figure.get_axes()
    assert len(axes) == 2 and len(axes[0].get_lines()) >= 4              # data, model, L1, L2, background
    widget.uf_robustness()
    qtbot.wait(20)
    assert widget._uf_last.relaxed is not None and "Relaxed bounds" in widget.uf_report.toPlainText()
    assert widget._uf_last.resolved is True
    # batch over both curves, with robustness, fills the table
    widget.uf_batch_list.selectAll()
    widget.uf_batch()
    qtbot.wait(20)
    assert widget.uf_batch_table.rowCount() == 2 and widget.uf_progress.value() == 100
    rows = {widget.uf_batch_table.item(i, 0).text(): widget.uf_batch_table.item(i, 2).text() for i in range(2)}
    assert rows["two"].startswith("L1_L2") and rows["one"].startswith("L1")
    assert set(widget._uf_results) == {"two", "one"}
    # exports and the JSON round trip
    csv_path, xlsx_path, json_path = (str(tmp_path / f"fits.{ext}") for ext in ("csv", "xlsx", "json"))
    with patch("saxs.qt_saxs2.QFileDialog.getSaveFileName", return_value=(csv_path, "")):
        widget.uf_export_csv()
    with patch("saxs.qt_saxs2.QFileDialog.getSaveFileName", return_value=(xlsx_path, "")):
        widget.uf_export_xlsx()
    with patch("saxs.qt_saxs2.QFileDialog.getSaveFileName", return_value=(json_path, "")):
        widget.uf_save_json()
    assert pd.read_csv(csv_path).shape[0] == 2 and os.path.getsize(xlsx_path) > 1000 and os.path.getsize(json_path) > 1000
    widget._uf_results.clear()
    widget.uf_batch_table.setRowCount(0)
    with patch("saxs.qt_saxs2.QFileDialog.getOpenFileName", return_value=(json_path, "")):
        widget.uf_load_json()
    qtbot.wait(20)
    assert set(widget._uf_results) == {"two", "one"} and widget.uf_batch_table.rowCount() == 2
    with patch("saxs.qt_saxs2.QMessageBox.critical") as crit, patch("saxs.qt_saxs2.QFileDialog.getOpenFileName", return_value=(csv_path, "")):
        widget.uf_load_json()
    crit.assert_called_once()


def test_series_tab_figures_and_tables(qtbot, fitted_widget, tmp_path, campaign):
    widget = fitted_widget
    widget.uf_batch_list.selectAll()
    widget.uf_batch()
    qtbot.wait(20)
    # fill from curves, then import an edited table
    widget.sr_fill_from_curves()
    assert widget.sr_table.rowCount() == 2 and widget.sr_table.item(0, 0).text() == "two"
    series = tmp_path / "series.csv"
    pd.DataFrame({"sample": ["two", "one"], "panel_row": [0, 0], "panel_col": [0, 1], "stack_index": [0, 0], "label": ["T", "O"],
                  "x_value": [1.0, 2.0], "group": ["g", "g"], "quantified": [True, False], "family_label": ["fam", "fam"],
                  "col_title": ["c0", "c1"]}).to_csv(series, index=False)
    with patch("saxs.qt_saxs2.QFileDialog.getOpenFileName", return_value=(str(series), "")):
        widget.sr_import_csv()
    assert widget.sr_table.rowCount() == 2 and "2 row(s)" in widget.sr_status.text()
    assert [widget.sr_group_combo.itemText(i) for i in range(widget.sr_group_combo.count())] == ["", "g"]
    widget.sr_grid_figure()
    qtbot.wait(20)
    assert widget.sr_preview.figure is not None and len(widget.sr_preview.figure.get_axes()) == 2
    assert "Grid figure" in widget.sr_status.text()
    widget.sr_group_combo.setCurrentText("g")
    widget.sr_fit_figure()
    qtbot.wait(20)
    assert len(widget.sr_preview.figure.get_axes()) == 4                    # 2 samples x (main + residual)
    widget.sr_trends_figure()
    qtbot.wait(20)
    assert len(widget.sr_preview.figure.get_axes()) == 4                    # the four default trend panels
    # the 2D frame figure from two frames of the synthetic campaign
    root, _ = campaign
    frames = iter([(os.path.join(root, "sessionA", "S2.edf"), ""), (os.path.join(root, "sessionA", "Empty.edf"), "")])
    with patch("saxs.qt_saxs2.QFileDialog.getOpenFileName", side_effect=lambda *a, **k: next(frames)):
        widget.sr_frame_figure()
    qtbot.wait(20)
    assert len(widget.sr_preview.figure.get_axes()) == 3
    # export the current figure (png only, low dpi) and the tables
    widget.sr_fmt_checks["pdf"].setChecked(False)
    widget.sr_fmt_checks["svg"].setChecked(False)
    widget.sr_dpi_edit.setText("50")
    with patch("saxs.qt_saxs2.QFileDialog.getSaveFileName", return_value=(str(tmp_path / "fig"), "")):
        widget.sr_export_figure()
    assert os.path.exists(tmp_path / "fig.png") and "fig.png" in widget.sr_status.text()
    with patch("saxs.qt_saxs2.QFileDialog.getSaveFileName", return_value=(str(tmp_path / "tables"), "")):
        widget.sr_export_tables()
    for ext in ("xlsx", "csv", "md"):
        assert os.path.exists(tmp_path / f"tables.{ext}"), ext
    fits = pd.read_csv(tmp_path / "tables.csv", encoding="utf-8-sig")
    assert list(fits["Sample"]) == ["two", "one"] and "x" in fits.columns
    # save the table back to CSV
    with patch("saxs.qt_saxs2.QFileDialog.getSaveFileName", return_value=(str(tmp_path / "series_out.csv"), "")):
        widget.sr_save_csv()
    assert pd.read_csv(tmp_path / "series_out.csv").shape[0] == 2
    # an empty table is reported, not raised
    widget.sr_table.setRowCount(0)
    with patch("saxs.qt_saxs2.QMessageBox.critical") as crit:
        widget.sr_grid_figure()
    crit.assert_called_once()


def test_worker_progress_reaches_the_callback():
    from core.qt_worker import run_in_thread
    seen = []
    done = []
    run_in_thread(lambda progress: [progress(i, 3, f"step {i}") for i in range(3)] and "ok", done.append, on_progress=lambda i, n, m: seen.append((i, n, m)))
    assert done == ["ok"] and seen == [(0, 3, "step 0"), (1, 3, "step 1"), (2, 3, "step 2")]
