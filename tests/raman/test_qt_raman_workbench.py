"""The Raman workbench: visibility vs highlight, operation panels (preview,
apply, treat all, hide originals), analyse tables, the sequence, sorting,
properties, colour by property, and the export helpers."""
from __future__ import annotations


import numpy as np
import pytest

from core.qt_models import Spectrum, SpectrumLibrary
from raman.qt_raman import RamanWorkspace
import raman.raman_ops as ro


def _spectrum(title, center=520.0, amp=1000.0, slope=0.0):
    x = np.linspace(100.0, 1500.0, 1401)
    y = 50.0 + amp * np.exp(-0.5 * ((x - center) / 4.0) ** 2) + 300.0 * np.exp(-0.5 * ((x - 1050.0) / 8.0) ** 2) + slope * (x - 100)
    return Spectrum(id=Spectrum.new_id(), title=title, path="", kind="raman_xy", x=x, y=y, meta={"laser_nm": 532.0})


def _workspace(qtbot, *spectra, **kw):
    lib = SpectrumLibrary()
    for s in spectra:
        lib.add(s)
    ws = RamanWorkspace(library=lib, **kw)
    qtbot.addWidget(ws)
    ws.set_spectra([s.id for s in lib.all()])
    return ws, lib


def test_list_visibility_and_highlight_semantics(qtbot):
    a, b, c = _spectrum("a"), _spectrum("b", 700), _spectrum("c", 900)
    ws, lib = _workspace(qtbot, a, b, c)
    # arriving: the first spectrum is shown and highlighted
    assert [s.title for s in ws._visible_spectra()] == ["a"]
    assert [s.title for s in ws._target_spectra()] == ["a"]
    # highlighting b shows it too; the target is the highlight only
    ws.file_list.item(1).setSelected(True)
    assert [s.title for s in ws._visible_spectra()] == ["a", "b"]
    assert sorted(s.title for s in ws._target_spectra()) == ["a", "b"]
    ws.file_list.clearSelection()
    ws.file_list.item(2).setSelected(True)
    assert [s.title for s in ws._target_spectra()] == ["c"]
    assert [s.title for s in ws._target_spectra(treat_all=True)] == ["a", "b", "c"]
    ws.show_only_selected()
    assert [s.title for s in ws._visible_spectra()] == ["c"]
    ws._set_all_visible(True)
    assert len(ws._visible_spectra()) == 3
    ws.render()
    ax = ws.plot.figure.get_axes()[0]
    assert len(ax.lines) == 3
    assert ws.visible_count_label.text() == "3 shown"


def test_operation_preview_apply_treat_all_and_hide_originals(qtbot):
    added, undo = [], []
    a, b = _spectrum("a", slope=0.2), _spectrum("b", 700, slope=0.2)
    ws, lib = _workspace(qtbot, a, b, on_derived_added=added.append, on_undo_push=undo.append)
    ws.file_list.item(1).setSelected(True)      # both shown, b highlighted
    ws.open_op(ro.OPS_BY_KEY["baseline_simple"])
    assert ws.op_panel.isVisibleTo(ws) and ws.op_title.text() == "Simple baseline"
    ws._param_widgets["range"].setText("1300-1500")
    ws.run_current_op(preview=True)
    assert ws._preview and len(ws._preview) == 2          # a and b are both highlighted (clicking shows + highlights)
    ws.render()
    assert any("preview" in ln.get_label() for ln in ws.plot.figure.get_axes()[0].lines)
    ws.file_list.clearSelection()
    ws.file_list.item(1).setSelected(True)
    ws.op_replace.setChecked(True)
    ws.run_current_op(preview=False)
    titles = [s.title for s in lib.all()]
    assert titles == ["a", "b", "b_offset"]
    assert added == [[lib.all()[2].id]]
    assert ws._preview is None
    visible = [s.title for s in ws._visible_spectra()]
    assert "b" not in visible and "b_offset" in visible          # originals hidden, result shown
    out = lib.all()[2]
    assert abs(np.mean(out.y[(out.x >= 1300) & (out.x <= 1500)])) < 1e-9
    # treat all shown: a and b_offset
    ws.op_replace.setChecked(False)
    ws.op_treat_all.setChecked(True)
    ws.open_op(ro.OPS_BY_KEY["normalize"])
    ws._param_widgets["mode"].setCurrentText("peak (max = 1)")
    ws.op_treat_all.setChecked(True)
    ws.run_current_op(preview=False)
    assert sorted(s.title for s in lib.all()) == sorted(["a", "b", "b_offset", "a_norm", "b_offset_norm"])
    assert all(np.isclose(np.max(s.y), 1.0) for s in lib.all() if s.title.endswith("_norm"))


def test_reference_operation_uses_the_reference_combo(qtbot):
    a, ref = _spectrum("a"), _spectrum("ref", amp=500.0)
    ws, lib = _workspace(qtbot, a, ref)
    ws.open_op(ro.OPS_BY_KEY["scaled_subtraction"])
    assert ws.op_ref_combo.isVisibleTo(ws)
    ws.op_ref_combo.setCurrentIndex(ws.op_ref_combo.findText("ref"))
    ws._param_widgets["k"].setText("2")
    ws.run_current_op(preview=False)
    out = lib.all()[-1]
    assert out.title == "a_sub"
    assert out.y[np.argmin(np.abs(out.x - 520))] == pytest.approx(1050 - 2 * 550, abs=1.0)


def test_analyze_tables_land_in_the_results_pane(qtbot):
    a, b = _spectrum("a"), _spectrum("b", 700)
    ws, lib = _workspace(qtbot, a, b)
    ws._set_all_visible(True)
    ws.op_treat_all.setChecked(True)
    ws.open_op(ro.OPS_BY_KEY["peak_table"])
    ws.op_treat_all.setChecked(True)
    ws.run_current_op(preview=False)
    t = ws.results_table
    assert t.columnCount() == 7 and t.rowCount() == 4
    assert t.horizontalHeaderItem(1).text() == "Position"
    assert [float(t.item(r, 1).text()) for r in range(4)] == pytest.approx([520, 1050, 700, 1050], abs=0.5)
    assert ws._results_text(",").splitlines()[0] == "Spectrum,Position,Height,FWHM,Area,Prominence,Source"
    ws.open_op(ro.OPS_BY_KEY["integrate"])
    ws.op_treat_all.setChecked(True)
    ws._param_widgets["range"].setText("500-540")
    ws._param_widgets["baseline"].setCurrentText("linear")
    ws.run_current_op(preview=False)
    assert ws.results_table.rowCount() == 2
    assert float(ws.results_table.item(0, 4).text()) == pytest.approx(1000 * 4 * np.sqrt(2 * np.pi), rel=0.02)


def test_sequence_record_save_load_run(qtbot, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog
    a, b = _spectrum("a", slope=0.3), _spectrum("b", 700, slope=0.3)
    ws, lib = _workspace(qtbot, a, b)
    ws.open_op(ro.OPS_BY_KEY["baseline_simple"]); ws._param_widgets["range"].setText("1300-1500"); ws.add_current_op_to_sequence()
    ws.open_op(ro.OPS_BY_KEY["smooth"]); ws.add_current_op_to_sequence()
    ws.open_op(ro.OPS_BY_KEY["peak_table"]); ws.add_current_op_to_sequence()
    assert ws.seq_list.count() == 3 and ws.seq_list.item(0).text().startswith("1. Simple baseline")
    p = str(tmp_path / "seq.json")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (p, "")))
    ws.seq_save()
    ws.seq_clear()
    assert ws.seq_list.count() == 0
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (p, "")))
    ws.seq_load()
    assert len(ws.sequence) == 3
    ws._set_all_visible(True)
    ws.seq_run(mode="visible")
    titles = [s.title for s in lib.all()]
    assert "a_offset_sm" in titles and "b_offset_sm" in titles
    assert ws.results_table.rowCount() == 4                      # the peak table of the final spectra
    assert "sequence finished on 2" in ws.log.toPlainText()
    # run on a folder: files imported, then processed
    folder = tmp_path / "batch"
    folder.mkdir()
    for name in ("f1", "f2"):
        np.savetxt(str(folder / f"{name}.txt"), np.column_stack([a.x, a.y]), delimiter="\t")
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: str(folder)))
    ws.seq_run(mode="folder")
    titles = [s.title for s in lib.all()]
    assert "f1" in titles and "f1_offset_sm" in titles and "f2_offset_sm" in titles


def test_sort_properties_colour_by_property(qtbot):
    a, b, c = _spectrum("b", amp=300.0), _spectrum("a", amp=900.0), _spectrum("c", amp=600.0)
    ws, lib = _workspace(qtbot, a, b, c)
    ws.sort_combo.setCurrentText("A → Z")
    ws.sort_spectra()
    assert [s.title for s in lib.all()] == ["a", "b", "c"]
    ws.sort_combo.setCurrentText("By maximum")
    ws.sort_spectra()
    assert [s.title for s in lib.all()] == ["b", "c", "a"]
    ws._set_all_visible(True)
    ws.colour_prop_combo.setCurrentText("maximum")
    ws.colour_by_property()
    cols = [ws.styles[s.id]["color"] for s in lib.all()]
    assert len(set(cols)) == 3
    ws.reset_colours()
    assert all("color" not in ws.styles[s.id] for s in lib.all())
    # properties dialog applied programmatically
    from raman.qt_raman_dialogs import SpectrumPropertiesDialog
    dlg = SpectrumPropertiesDialog(ws, [lib.all()[0]], ws.styles)
    dlg.offset_edit.setText("100"); dlg.laser_edit.setText("785"); dlg.ls_combo.setCurrentText("dashed")
    dlg.apply(ws.styles)
    st = ws.styles[lib.all()[0].id]
    assert st["offset"] == 100.0 and st["ls"] == "dashed" and lib.all()[0].meta["laser_nm"] == 785.0
    ws.render()
    ax = ws.plot.figure.get_axes()[0]
    assert ax.lines[0].get_linestyle() == "--"
    assert ax.lines[0].get_ydata()[0] == pytest.approx(lib.all()[0].y[0] + 100.0)


def test_display_unit_conversion_and_flip(qtbot):
    a = _spectrum("a")
    ws, lib = _workspace(qtbot, a)
    ws.xunit_combo.setCurrentText("Wavelength (nm)")
    ws.laser_edit.setText("532")
    ws.render()
    ax = ws.plot.figure.get_axes()[0]
    xd = ax.lines[0].get_xdata()
    assert xd.min() == pytest.approx(1e7 / (1e7 / 532 - 100), abs=1e-6)
    assert ax.get_xlabel() == "Wavelength (nm)"
    ws.xunit_combo.setCurrentText("as measured")
    ws.flipx_check.setChecked(True)
    ws.render()
    ax = ws.plot.figure.get_axes()[0]
    assert ax.xaxis_inverted()


def test_export_helpers(qtbot, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog
    import core.io_universal as iou
    a, b = _spectrum("a"), _spectrum("b", 700)
    ws, lib = _workspace(qtbot, a, b)
    ws._set_all_visible(True)
    folder = tmp_path / "out"
    folder.mkdir()
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: str(folder)))
    ws.file_list.selectAll()
    ws.export_jcamp()
    back, meta = iou.load_any(str(folder / "a.jdx"), return_meta=True)
    assert meta["selected_parser"] == "jcamp" and np.allclose(back["y"], a.y, rtol=1e-5) and np.allclose(back["x"], a.x, atol=1e-3)
    ws.export_text_files()
    assert (folder / "b.txt").is_file()
    csv = tmp_path / "all.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(csv), "")))
    ws.export_combined_csv()
    head = csv.read_text().splitlines()[0]
    assert head == "x,a,b"


def _tab_index(ws, name):
    return [ws.ribbon.tabText(i) for i in range(ws.ribbon.count())].index(name)


def test_operation_form_is_rebuilt_without_leftovers(qtbot):
    """Switching operations must not leave widgets of the previous form behind
    (cells are sub-layouts: their widgets were leaking and stayed visible)."""
    from PySide6.QtWidgets import QComboBox, QPushButton
    ws, lib = _workspace(qtbot, _spectrum("a"))
    ws.open_op(ro.OPS_BY_KEY["baseline_advanced"])          # two choice params + a Pick button
    assert len(ws.op_panel.findChildren(QComboBox)) == 4     # method, anchors joined by, output + the reference combo
    assert any(b.text() == "Pick…" for b in ws.op_panel.findChildren(QPushButton))
    ws.open_op(ro.OPS_BY_KEY["statistics"])                  # no choice param, no Pick
    assert len(ws.op_panel.findChildren(QComboBox)) == 1     # only the reference combo survives
    assert not any(b.text() == "Pick…" for b in ws.op_panel.findChildren(QPushButton))
    assert set(ws._param_widgets) == {p.key for p in ro.OPS_BY_KEY["statistics"].params}


def test_detail_panels_follow_the_ribbon(qtbot):
    """One panel under the ribbon at a time: the operation form, the plot
    settings or the sequence steps; the Sequence tab shows its steps, an
    operation reopens its panel when its tab is selected again."""
    ws, lib = _workspace(qtbot, _spectrum("a"))
    ws.ribbon.setCurrentIndex(_tab_index(ws, "Sequence"))
    assert ws.seq_panel.isVisibleTo(ws) and not ws.op_panel.isVisibleTo(ws)
    ws.ribbon.setCurrentIndex(_tab_index(ws, "Process"))
    assert not ws.seq_panel.isVisibleTo(ws)
    ws.open_op(ro.OPS_BY_KEY["normalize"])
    assert ws.op_panel.isVisibleTo(ws) and ws.ribbon.tabText(ws.ribbon.currentIndex()) == "Process"
    ws.open_plot_settings()
    assert ws.plot_panel.isVisibleTo(ws) and not ws.op_panel.isVisibleTo(ws)
    ws.ribbon.setCurrentIndex(_tab_index(ws, "Display"))
    assert ws.plot_panel.isVisibleTo(ws)                     # the plot settings stay while styling
    ws.ribbon.setCurrentIndex(_tab_index(ws, "Process"))
    assert ws.op_panel.isVisibleTo(ws) and not ws.plot_panel.isVisibleTo(ws)
    ws.close_op_panel()
    assert not any(p.isVisibleTo(ws) for p in (ws.op_panel, ws.plot_panel, ws.seq_panel))
    # opening an operation from elsewhere switches the ribbon to its tab
    ws.open_op(ro.OPS_BY_KEY["derivative"])
    assert ws.ribbon.tabText(ws.ribbon.currentIndex()) == "Transform"


def test_sequence_steps_reopen_with_their_parameters(qtbot):
    ws, lib = _workspace(qtbot, _spectrum("a"))
    ws.open_op(ro.OPS_BY_KEY["baseline_simple"])
    ws._param_widgets["range"].setText("1300-1500")
    ws.op_suffix.setText("flat")
    ws.add_current_op_to_sequence()
    assert ws.seq_count_label.text() == "1 step"
    ws.open_op(ro.OPS_BY_KEY["smooth"])
    ws._open_seq_step(ws.seq_list.item(0))
    assert ws.op_title.text() == "Simple baseline"
    assert ws._param_widgets["range"].text() == "1300-1500" and ws.op_suffix.text() == "flat"


def test_ribbon_buttons_escape_ampersands_and_every_tab_fits(qtbot):
    from PySide6.QtWidgets import QPushButton
    ws, lib = _workspace(qtbot, _spectrum("a"))
    analyze = ws.ribbon.widget(_tab_index(ws, "Analyze"))
    assert "Peaks && FWHM" in [b.text() for b in analyze.findChildren(QPushButton)]
    # the ribbon is sized for its tallest tab: no tab content is clipped
    tallest = max(ws.ribbon.widget(i).widget().sizeHint().height() for i in range(ws.ribbon.count()))
    assert ws.ribbon.height() >= tallest + ws.ribbon.tabBar().sizeHint().height()


def test_pick_sessions_append_ranges_and_points_and_draw_them(qtbot):
    ws, lib = _workspace(qtbot, _spectrum("a"))
    ws.open_op(ro.OPS_BY_KEY["baseline_advanced"])
    regions = ws._param_widgets["regions"]
    ws._start_pick(regions, "range", None)
    assert ws._handle_pick(300.0) and ws._handle_pick(300.0, dblclick=True) and ws._handle_pick(300.0)
    assert regions.text() == "" and ws._pick_target is not None       # one point so far; the repeats were ignored
    ws._handle_pick(100.0)                                              # wrong order: swapped
    assert regions.text() == "100.0-300.0"
    ws._handle_pick(900.0)
    ws._handle_pick(700.0)
    assert regions.text() == "100.0-300.0; 700.0-900.0" and ws._pick_target is not None   # still picking
    ws._handle_pick(0.0, button=3)                                      # right-click ends the session
    assert ws._pick_target is None
    points = ws._param_widgets["points"]
    ws._start_pick(points, "list", None)
    ws._handle_pick(200.0)
    ws._handle_pick(400.0)
    assert points.text() == "200.00, 400.00" and ws._pick_target is not None
    ws._on_plot_key(type("E", (), {"key": "escape"})())
    assert ws._pick_target is None
    ws.render()
    ax = ws.plot.figure.get_axes()[0]
    assert len(ax.patches) == 2                                         # the two ranges, shaded
    assert len(ax.lines) == 1 + 2                                       # the spectrum + the two anchor marks
    ws.close_op_panel()
    ws.render()
    ax = ws.plot.figure.get_axes()[0]
    assert len(ax.patches) == 0 and len(ax.lines) == 1


def test_remove_all_and_the_name_filter(qtbot, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    undo = []
    ws, lib = _workspace(qtbot, _spectrum("alpha"), _spectrum("beta", 700), _spectrum("gamma", 900), on_undo_push=undo.append)
    ws.filter_edit.setText("be")
    assert [ws.file_list.item(i).isHidden() for i in range(3)] == [True, False, True]
    ws._set_all_visible(True)                                           # acts on the listed entries only
    assert [s.title for s in ws._visible_spectra()] == ["alpha", "beta"]
    ws.filter_edit.clear()
    assert not any(ws.file_list.item(i).isHidden() for i in range(3))
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes))
    ws.remove_all()
    assert len(lib) == 0 and ws.file_list.count() == 0
    assert undo and undo[-1][0] == "delete" and len(undo[-1][1]) == 3


def test_many_spectra_render_without_legend_or_peak_labels(qtbot):
    specs = [_spectrum(f"s{i}", 300.0 + 30.0 * i) for i in range(30)]
    ws, lib = _workspace(qtbot, *specs)
    ws._set_all_visible(True)
    ws.peak_labels_check.setChecked(True)
    ws.render()
    ax = ws.plot.figure.get_axes()[0]
    assert ax.get_legend() is None and len(ax.lines) == 30
    notes = [t.get_text() for t in ax.texts]
    assert any("legend hidden" in n for n in notes) and any("peak labels hidden" in n for n in notes)


def test_found_peaks_are_marked_on_the_plot_with_a_toggle(qtbot):
    ws, lib = _workspace(qtbot, _spectrum("a"))
    sid = lib.all()[0].id
    ws.open_op(ro.OPS_BY_KEY["peak_table"])
    ws._param_widgets["manual"].setText("300")
    ws._param_widgets["snap"].setText("0")
    ws.run_current_op(preview=False)
    assert set(ws.found_peaks) == {sid} and [src for _p, _h, src in ws.found_peaks[sid]] == ["manual", "auto", "auto"]
    ws.close_op_panel()
    ws.render()
    ax = ws.plot.figure.get_axes()[0]
    assert len(ax.lines) == 1 + 3 and len(ax.texts) == 3            # the spectrum + three marks with their positions
    ws.found_peaks_check.setChecked(False)
    ws.render()
    ax = ws.plot.figure.get_axes()[0]
    assert len(ax.lines) == 1 and len(ax.texts) == 0
