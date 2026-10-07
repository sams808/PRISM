"""UI wiring tests for the SAXS tabs of SaxsWorkspace (Frames, Fit, Series figures). qt_worker runs synchronously
under tests (conftest), dialogs are patched, draws are flushed with qtbot.wait(20). Synthetic frames and curves only."""
from __future__ import annotations

import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from saxs.core import edf_sectors as es
from saxs.qt_saxs import SaxsWorkspace

import fit_synthetic as fsyn  # noqa: E402  (tests/saxs is on sys.path under pytest)
import saxs2_synthetic as syn  # noqa: E402

pytestmark = pytest.mark.skipif(not es.HAVE_FABIO, reason="fabio not installed")


def _fast_options(widget):
    for key, check in widget.fit_family_checks.items():
        check.setChecked(key in ("teubner_strey", "sphere"))
    widget.fit_second_check.setChecked(False)


def test_saxs_tabs_registered_and_documented(qtbot):
    from core.qt_help import HELP_HTML, MODULE_GUIDES
    widget = SaxsWorkspace()
    qtbot.addWidget(widget)
    labels = [widget.tabs.tabText(i) for i in range(widget.tabs.count())]
    assert labels == ["Curves", "Reduction", "Analysis", "WAXS", "Frames", "Fit", "Series figures"]
    for name in ("Frames", "Fit", "Series figures"):
        assert f"<b>{name}</b>" in HELP_HTML and f"<b>{name}</b>" in MODULE_GUIDES["SAXS/WAXS"]
    for gone in ("Composite fit", "Unified fit"):
        assert gone not in labels and gone not in MODULE_GUIDES["SAXS/WAXS"]


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
    assert sorted(os.listdir(out)) == ["S1__sectors.dat", "S2__sectors.dat", "S3_0p5mm__sectors.dat", "S4__sectors.dat", "sasview"]
    assert sorted(os.listdir(os.path.join(out, "sasview"))) == ["S1__sasview.dat", "S2__sasview.dat", "S3_0p5mm__sasview.dat", "S4__sasview.dat"]
    sv_dir = str(tmp_path / "sv_export")
    with patch("saxs.qt_saxs2.QFileDialog.getExistingDirectory", return_value=sv_dir):
        widget.fr_export_sasview()
    assert len(os.listdir(sv_dir)) == 4 and "SasView-ready" in widget.fr_status.text()
    names = [c.name for c in widget.curves]
    assert names == ["S1", "S2", "S3_0p5mm", "S4"]
    assert widget.curves[0].metadata["per_mm"] is True and widget.fit_curve_list.count() == 4
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
    for c in fsyn.curves():
        widget.add_curve(c)
    _fast_options(widget)
    widget.fit_curve_list.selectAll()
    widget.fit_rule_pattern.setText(r"^(A|B)_.*$")
    widget.fit_rule_replace.setText(r"ref\1")
    widget.fit_apply_rule()
    for i in range(widget.fit_curve_list.count()):
        it = widget.fit_curve_list.item(i)
        it.setSelected(it.text() == "A_sph")
    widget.fit_contrast_edit.setText(str(fsyn.CONTRAST))
    widget.fit_set_contrast()
    widget.fit_curve_list.selectAll()
    widget.fit_run()
    qtbot.wait(20)
    return widget


def _plotted(widget):
    """Name of the curve drawn in the Fit tab plot (the label of its data points), or None."""
    axes = widget.fit_plot.figure.get_axes()
    return axes[0].get_lines()[0].get_label() if axes and axes[0].get_lines() else None


def test_fit_tab_runs_the_workflow_and_shows_every_candidate(qtbot, fitted_widget, tmp_path):
    widget = fitted_widget
    assert "3 reference(s) set" not in widget.fit_status.text()            # status now reports the fit
    res = widget._fit_result
    assert res is not None and set(res.curves) == {"refA", "refB", "A_plain", "B_struct", "A_sph"}
    assert res.curves["A_plain"].reference == "refA" and res.curves["B_struct"].reference == "refB"
    assert res.curves["B_struct"].retained.startswith("TS") and res.curves["A_sph"].retained.startswith("SPH")
    assert res.curves["A_sph"].contrast == fsyn.CONTRAST and res.curves["A_plain"].retained in ("B0", "B1")
    assert widget.fit_table.rowCount() == 5 and widget.fit_progress.value() == 100
    rows = {widget.fit_table.item(i, 0).text(): i for i in range(5)}
    widget.fit_table.selectRow(rows["B_struct"])
    assert widget.fit_model_combo.currentText().startswith("TS") and widget.fit_model_combo.count() > 3
    assert widget.fit_param_table.rowCount() >= 6                           # filled at once, not with the plot
    assert "Gate (ratio to reference): PASS" in widget.fit_report.toPlainText()
    qtbot.waitUntil(lambda: _plotted(widget) == "B_struct", timeout=5000)   # the redraw is debounced
    axes = widget.fit_plot.figure.get_axes()
    assert len(axes) == 2 and len(axes[0].get_lines()) >= 5                 # data, model, grain, high-q, ts
    widget.fit_iq4_check.setChecked(True)
    widget.fit_model_combo.setCurrentIndex(widget.fit_model_combo.count() - 1)
    qtbot.waitUntil(lambda: _plotted(widget) == "B_struct"
                    and widget.fit_plot.figure.get_axes()[0].get_ylabel().startswith("I q"), timeout=5000)
    # robustness, then save / load and the exports
    widget.fit_robustness()
    qtbot.wait(20)
    assert widget._fit_result.robustness is not None and len(widget._fit_result.robustness)
    assert widget.fit_table.item(rows["B_struct"], 6).text() in ("yes", "no")
    json_path = str(tmp_path / "fits.json")
    with patch("saxs.qt_saxs_fit.QFileDialog.getSaveFileName", return_value=(json_path, "")):
        widget.fit_save_json()
    widget._fit_result = None
    with patch("saxs.qt_saxs_fit.QFileDialog.getOpenFileName", return_value=(json_path, "")):
        widget.fit_load_json()
    assert widget._fit_result is not None and widget.fit_table.rowCount() == 5
    with patch("saxs.qt_saxs_fit.QFileDialog.getSaveFileName", return_value=(str(tmp_path / "tabs"), "")):
        widget.fit_export_tables()
    assert os.path.exists(tmp_path / "tabs.xlsx") and os.path.exists(tmp_path / "tabs_models.csv")
    with patch("saxs.qt_saxs_fit.QFileDialog.getExistingDirectory", return_value=str(tmp_path)):
        widget.fit_export_curves()
    assert os.path.exists(tmp_path / "B_struct_fit.csv")
    with patch("saxs.qt_saxs_fit.QMessageBox.critical") as crit, \
            patch("saxs.qt_saxs_fit.QFileDialog.getOpenFileName", return_value=(str(tmp_path / "tabs.csv"), "")):
        widget.fit_load_json()
    crit.assert_called_once()


def test_fit_tab_guards(qtbot):
    widget = SaxsWorkspace()
    qtbot.addWidget(widget)
    with patch("saxs.qt_saxs_fit.QMessageBox.warning") as warn:
        widget.fit_run()                                    # nothing selected
    warn.assert_called_once()
    with patch("saxs.qt_saxs_fit.QMessageBox.information") as info:
        widget.fit_robustness()
        widget.fit_export_tables()
    assert info.call_count == 2
    for c in fsyn.curves():
        widget.add_curve(c)
    widget.fit_rule_pattern.setText("([")                  # invalid regex is reported
    with patch("saxs.qt_saxs_fit.QMessageBox.warning") as warn:
        widget.fit_apply_rule()
    warn.assert_called_once()
    o = widget._fit_options()
    assert o.families == ("sphere", "sphere_py", "teubner_strey", "debye", "unified") and o.gate == "reference"
    widget.fit_gate_combo.setCurrentIndex(1)
    widget.fit_edits["dbic_min"].setText("6")
    o = widget._fit_options()
    assert o.gate == "none" and o.dbic_min == 6.0


def test_series_tab_figures_and_tables(qtbot, fitted_widget, tmp_path, campaign):
    widget = fitted_widget
    # fill from curves, then import an edited table
    widget.sr_fill_from_curves()
    assert widget.sr_table.rowCount() == 5 and widget.sr_table.item(0, 0).text() == "refA"
    series = tmp_path / "series.csv"
    pd.DataFrame({"sample": ["B_struct", "A_sph"], "panel_row": [0, 0], "panel_col": [0, 1], "stack_index": [0, 0], "label": ["T", "O"],
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
    assert list(fits["curve"])[:2] == ["B_struct", "A_sph"] and "x" in fits.columns
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
