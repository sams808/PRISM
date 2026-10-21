"""The Raman block's Spectragryph-parity additions (2026-10-21): peak labels
and table, cursor readout, line styles, clipboard copy, and the Raman ID
similarity search backed by the user's reference library."""
from __future__ import annotations

import json

import numpy as np
import pytest

from core.qt_models import Spectrum, SpectrumLibrary


def _spectrum(title, center=520.0, n=1401, width=4.0, slope=0.0):
    x = np.linspace(100.0, 1500.0, n)
    y = 50.0 + 1000.0 * np.exp(-0.5 * ((x - center) / width) ** 2) + 300.0 * np.exp(-0.5 * ((x - 1050.0) / 8.0) ** 2) + slope * (x - 100)
    return Spectrum(id=Spectrum.new_id(), title=title, path="", kind="raman_xy", x=x, y=y)


def _library_with(*spectra):
    lib = SpectrumLibrary()
    for s in spectra:
        lib.add(s)
    return lib


def test_raman_peak_labels_table_cursor_and_clipboard(qtbot):
    from PySide6.QtWidgets import QApplication
    from raman.qt_raman import RamanWorkspace
    s1, s2 = _spectrum("a"), _spectrum("b", center=700.0)
    ws = RamanWorkspace(library=_library_with(s1, s2))
    qtbot.addWidget(ws)
    ws.set_spectra([s1.id, s2.id])
    ws.file_list.selectAll()
    ws.mode_stacked.setChecked(True)
    ws.peak_labels_check.setChecked(True)
    ws.render()
    ax = ws.plot.figure.get_axes()[0]
    labels = sorted(float(t.get_text()) for t in ax.texts if t.get_text().replace(".", "", 1).isdigit())
    assert labels == pytest.approx([520.0, 700.0, 1050.0, 1050.0], abs=1.0)

    # peak table dialog
    ws.open_peak_table()
    dlg = ws._peak_table_dialog
    assert dlg.table.rowCount() == 4
    assert [dlg.table.item(r, 0).text() for r in range(4)] == ["a", "a", "b", "b"]
    dlg.copy_text()
    assert QApplication.clipboard().text().startswith("Spectrum\tPosition")

    # cursor readout: a synthetic motion event at x = 520 reports both spectra
    ws.cursor_check.setChecked(True)

    class _Ev:
        inaxes = ax
        xdata = 520.0
        ydata = 0.0
    ws._on_cursor_move(_Ev())
    text = ws.cursor_label.text()
    assert text.startswith("x = 520.00") and "a:" in text and "b:" in text
    assert len(ws._cursor_artists) >= 3  # vline + a marker per curve
    ws.cursor_check.setChecked(False)
    assert ws._cursor_artists == [] and ws.cursor_label.text() == ""

    # line style / legend placement
    ws.marker_combo.setCurrentText("Points")
    ws.legend_combo.setCurrentText("No legend")
    ws.render()
    ax = ws.plot.figure.get_axes()[0]
    assert ax.get_legend() is None
    assert all(line.get_linestyle() == "None" for line in ax.get_lines()[:2])

    # copy data: same grid -> one x column + one column per spectrum
    ws.copy_data_to_clipboard()
    lines = QApplication.clipboard().text().splitlines()
    assert lines[0] == "x\ta\tb" and len(lines) == 1 + len(s1.x)
    ws.copy_plot_to_clipboard()
    assert not QApplication.clipboard().image().isNull()


def test_raman_id_similarity_search_uses_the_reference_library(qtbot, tmp_path, monkeypatch):
    from raman.qt_rruff import RruffMatchWorkspace
    import raman.raman_library as rl
    from raman import qt_raman_dialogs as dlgs

    unknown = _spectrum("unknown", center=520.0, slope=0.2)
    ws = RruffMatchWorkspace(library=_library_with(unknown), cache_dir=str(tmp_path / "no_rruff"),
                             library_path=str(tmp_path / "lib.json"))
    qtbot.addWidget(ws)
    ws.set_spectra([unknown.id])

    # add two references through the dialog (accepted with typed values)
    monkeypatch.setattr(dlgs.AddToLibraryDialog, "exec", lambda self: True)
    monkeypatch.setattr(dlgs.AddToLibraryDialog, "values",
                        lambda self: {"name": "standard 520", "laser_nm": 532.0, "source": "test", "notes": ""})
    ws.add_query_to_library()
    entries = rl.load_library(str(tmp_path / "lib.json"))
    assert [e.name for e in entries] == ["standard 520"] and entries[0].meta["laser_nm"] == 532.0
    other = _spectrum("other", center=900.0)
    entries.append(rl.entry_from_spectrum("standard 900", other.x, other.y))
    rl.save_library(entries, str(tmp_path / "lib.json"))

    # similarity search over the library only (no RRUFF cache here), 1st derivative beats the slope
    ws.match_mode_combo.setCurrentIndex(1)  # Pearson
    assert ws.preprocess_combo.isEnabled()
    ws.source_rruff_check.setChecked(False)
    ws.preprocess_combo.setCurrentIndex(2)  # d1
    ws.find_matches()
    assert ws.results_table.rowCount() == 2
    assert ws.results_table.item(0, 0).text() == "standard 520"
    assert ws.results_table.item(0, 5).text() == "my library"
    assert float(ws.results_table.item(0, 6).text()) > 0.99
    assert float(ws.results_table.item(1, 6).text()) < 0.5
    # the hit's curve is overlaid and accepting records a score
    ax = ws.plot.figure.get_axes()[0]
    assert any("my library" in line.get_label() for line in ax.get_lines())
    ws.results_table.selectRow(0)
    ws.accept_selected_candidate()
    rec = unknown.meta["rruff_match"]
    assert rec["mineral"] == "standard 520" and rec["rruff_id"] == "library" and rec["score"] > 0.99

    # peak mode again disables the similarity widgets and the sources
    ws.match_mode_combo.setCurrentIndex(0)
    assert not ws.preprocess_combo.isEnabled()


def test_raman_id_similarity_over_a_fake_rruff_cache(qtbot, tmp_path):
    from raman.qt_rruff import RruffMatchWorkspace
    cache = tmp_path / "rruff"; raw = cache / "raw"; raw.mkdir(parents=True)
    index = []
    for name, rid, center in (("Quartzlike", "R000001", 520.0), ("Farlike", "R000002", 900.0)):
        s = _spectrum(name, center=center)
        fn = f"{name}__{rid}__Raman__532__0__unoriented__Raman_Data_Processed__1.txt"
        body = [f"##NAMES={name}", f"##RRUFFID={rid}", "##LASER_WAVELENGTH=532"]
        body += [f"{x:.2f}, {y:.3f}" for x, y in zip(s.x, s.y)]
        (raw / fn).write_text("\n".join(body), encoding="utf-8")
        index.append({"mineral": name, "rruff_id": rid, "wavelength_nm": 532.0, "peaks": [center],
                      "raw_path": str(raw / fn), "category": "excellent_unoriented", "data_kind": "Processed",
                      "scan_type": "Raman", "orientation_deg": None})
    (cache / "index.json").write_text(json.dumps(index), encoding="utf-8")
    unknown = _spectrum("unknown", center=520.0)
    ws = RruffMatchWorkspace(library=_library_with(unknown), cache_dir=str(cache), library_path=str(tmp_path / "lib.json"))
    qtbot.addWidget(ws)
    ws.set_spectra([unknown.id])
    ws.match_mode_combo.setCurrentIndex(1)
    ws.source_library_check.setChecked(False)
    ws.find_matches()
    assert ws.results_table.rowCount() == 2
    assert ws.results_table.item(0, 0).text() == "Quartzlike"
    assert float(ws.results_table.item(0, 6).text()) > 0.99
