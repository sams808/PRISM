"""Tests for qt_htxrd.py (M20) — the HTXRD series-processing workspace.

Run separately from the default Tk-focused suite (see pytest.ini / conftest.py
for why): `pytest tests/test_qt_htxrd.py --override-ini="addopts="`
"""
from __future__ import annotations

import numpy as np
import rampy as rp

from qt_htxrd import HtxrdWorkspace
from qt_shell import DataappMainWindow


def _write_series(tmp_path, temps=(100, 200, 300), center_by_temp=None):
    x = np.linspace(20, 40, 600)
    paths = []
    for t in temps:
        center = center_by_temp(t) if center_by_temp else 30.0
        y = rp.gaussian(x, 500.0, center, 0.3) + 50.0
        p = tmp_path / f"scan_{t}.xy"
        np.savetxt(p, np.column_stack([x, y]))
        paths.append(str(p))
    return paths


def test_workspace_constructs_empty(qtbot):
    widget = HtxrdWorkspace()
    qtbot.addWidget(widget)
    assert widget.series == []
    assert widget.series_list.count() == 0


def test_import_files_with_template_orders_series_by_ramp(qtbot, tmp_path, monkeypatch):
    paths = _write_series(tmp_path, temps=(300, 100, 200))
    widget = HtxrdWorkspace()
    qtbot.addWidget(widget)
    widget.template_edit.setText("scan_???.xy")

    monkeypatch.setattr("qt_htxrd.QFileDialog.getOpenFileNames", staticmethod(lambda *a, **k: (paths, "")))
    widget.import_files()
    qtbot.wait(200)  # waterfall render is debounced via request_redraw

    assert len(widget.series) == 3
    assert [p.ramp_value for p in widget.series] == [100.0, 200.0, 300.0]
    assert widget.series_list.count() == 3
    assert len(widget.plot.figure.get_axes()) == 1
    assert len(widget.plot.figure.get_axes()[0].lines) == 3


def test_track_peak_populates_table_plot_and_flags(qtbot, tmp_path, monkeypatch):
    def center_by_temp(t):
        return 30.0 - 0.0005 * t  # linear thermal shift

    paths = _write_series(tmp_path, temps=(100, 200, 300, 400), center_by_temp=center_by_temp)
    widget = HtxrdWorkspace()
    qtbot.addWidget(widget)
    widget.template_edit.setText("scan_???.xy")
    monkeypatch.setattr("qt_htxrd.QFileDialog.getOpenFileNames", staticmethod(lambda *a, **k: (paths, "")))
    widget.import_files()
    qtbot.wait(200)

    widget.window_lo_edit.setText("29.0")
    widget.window_hi_edit.setText("31.0")
    widget.run_track_peak()
    qtbot.wait(20)

    assert len(widget.track_results) == 4
    assert all(r.error is None for r in widget.track_results)
    assert widget.track_table.rowCount() == 4
    # Centers should decrease with temperature (built-in linear shift).
    centers = [r.center for r in widget.track_results]
    assert centers == sorted(centers, reverse=True)
    # Three stacked subplots: center/FWHM/area vs ramp.
    assert len(widget.track_plot.figure.get_axes()) == 3
    assert "anomalies" in widget.transition_text.toPlainText().lower() or "transition" in widget.transition_text.toPlainText().lower()


def test_track_peak_without_series_warns(qtbot):
    widget = HtxrdWorkspace()
    qtbot.addWidget(widget)
    widget.run_track_peak()  # QMessageBox.warning neutralized by conftest fixture
    assert widget.track_results == []


def test_track_peak_invalid_window_warns(qtbot, tmp_path, monkeypatch):
    paths = _write_series(tmp_path)
    widget = HtxrdWorkspace()
    qtbot.addWidget(widget)
    monkeypatch.setattr("qt_htxrd.QFileDialog.getOpenFileNames", staticmethod(lambda *a, **k: (paths, "")))
    widget.import_files()
    qtbot.wait(200)

    widget.window_lo_edit.setText("31.0")
    widget.window_hi_edit.setText("29.0")  # inverted
    widget.run_track_peak()
    assert widget.track_results == []


def test_export_track_csv_writes_file(qtbot, tmp_path, monkeypatch):
    paths = _write_series(tmp_path)
    widget = HtxrdWorkspace()
    qtbot.addWidget(widget)
    widget.template_edit.setText("scan_???.xy")
    monkeypatch.setattr("qt_htxrd.QFileDialog.getOpenFileNames", staticmethod(lambda *a, **k: (paths, "")))
    widget.import_files()
    qtbot.wait(200)
    widget.window_lo_edit.setText("29.0")
    widget.window_hi_edit.setText("31.0")
    widget.run_track_peak()
    qtbot.wait(20)

    out_path = tmp_path / "track.csv"
    monkeypatch.setattr("qt_htxrd.QFileDialog.getSaveFileName", staticmethod(lambda *a, **k: (str(out_path), "")))
    widget.export_track_csv()

    assert out_path.exists()
    content = out_path.read_text()
    assert "center_2theta" in content
    assert "scan_100" in content


def test_real_rasx_example_loads_with_metadata_temperature(qtbot, monkeypatch, examples_dir):
    rasx = examples_dir / "HTXRD_example.rasx"
    widget = HtxrdWorkspace()
    qtbot.addWidget(widget)
    monkeypatch.setattr("qt_htxrd.QFileDialog.getOpenFileNames", staticmethod(lambda *a, **k: ([str(rasx)], "")))
    widget.import_files()
    qtbot.wait(200)

    assert len(widget.series) == 1
    assert widget.series[0].ramp_source == "metadata"
    assert "metadata" in widget.status_label.text()


def test_shell_htxrd_page_is_htxrd_workspace(qtbot):
    window = DataappMainWindow()
    qtbot.addWidget(window)
    qtbot.wait(20)
    assert isinstance(window.htxrd_page, HtxrdWorkspace)
    window.nav.setCurrentRow(7)  # HT-XRD workspace
    qtbot.wait(20)
    assert window.stack.currentWidget() is window.htxrd_page
