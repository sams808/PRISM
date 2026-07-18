"""Tests for qt_xrd.py — the QualX-style XRD phase-ID workspace.
Uses the same tiny synthetic QualX-format database as test_xrd_id_science."""
from __future__ import annotations

import numpy as np
import pytest

import xrd_id_science as xid
from qt_models import Spectrum, SpectrumLibrary
from qt_shell import NAV_ITEMS, DataappMainWindow
from qt_xrd import XrdIdWorkspace
from test_xrd_id_science import QUARTZ_D, QUARTZ_I, _make_source_sq


@pytest.fixture()
def db_path(tmp_path):
    src = tmp_path / "src.sq"
    _make_source_sq(src, [
        (1010, "Quartz low", "Quartz", "Si O2", "P 32 2 1", "A", QUARTZ_D, QUARTZ_I, ["Si", "O"]),
        (1020, "Calcite", "Calcite", "C Ca O3", "R -3 c", "A",
         [3.035, 2.285, 2.095, 1.913, 1.875, 3.86], [100.0, 18.0, 18.0, 17.0, 17.0, 12.0], ["C", "Ca", "O"]),
    ])
    out = tmp_path / "unified.sq"
    xid.build_xrd_database([(str(src), "TESTDB")], out_path=str(out), log=lambda *a: None)
    return str(out)


def _quartz_pattern() -> Spectrum:
    """Synthetic diffractogram with quartz's lines as narrow Gaussians."""
    tt = xid.d_to_two_theta(np.array(QUARTZ_D))
    x = np.linspace(10, 80, 4000)
    y = np.full_like(x, 50.0)
    for c, i in zip(tt, QUARTZ_I):
        y += 10.0 * i * np.exp(-0.5 * ((x - c) / 0.05) ** 2)
    return Spectrum(id=Spectrum.new_id(), title="my_xrd", path="", kind="xrd_xy", x=x, y=y)


def _workspace(qtbot, db_path, on_accept=None):
    library = SpectrumLibrary()
    sp = _quartz_pattern()
    library.add(sp)
    widget = XrdIdWorkspace(library=library, db_path=db_path, on_accept=on_accept)
    qtbot.addWidget(widget)
    widget.set_spectra([sp.id])
    return widget, library, sp


def test_db_status_and_source_checkboxes(qtbot, db_path):
    widget, _, _ = _workspace(qtbot, db_path)
    assert "2 cards" in widget.db_status_label.text()
    assert set(widget.source_checks.keys()) == {"TESTDB"}


def test_auto_find_then_search_ranks_quartz_first(qtbot, db_path):
    widget, _, _ = _workspace(qtbot, db_path)
    widget.auto_find_peaks()
    assert widget.peaks_edit.text() != ""
    widget.run_search()
    qtbot.wait(20)
    assert widget.results_table.rowCount() >= 1
    assert widget.results_table.item(0, 1).text() == "Quartz"
    assert int(widget.results_table.item(0, 0).text()) > 80  # FoM %
    # provenance visible: source tag + original code
    assert widget.results_table.item(0, 3).text() == "TESTDB"
    assert widget.results_table.item(0, 4).text() == "1010"


def test_accept_records_phase_and_is_iterative(qtbot, db_path):
    calls = []
    widget, _, sp = _workspace(qtbot, db_path, on_accept=lambda sid, old: calls.append((sid, old)))
    widget.auto_find_peaks()
    widget.run_search()
    qtbot.wait(20)
    widget.results_table.selectRow(0)
    widget.accept_selected()
    qtbot.wait(20)

    assert sp.meta["xrd_match"]["mineral"] == "Quartz"
    assert sp.meta["xrd_match"]["source_code"] == "1010"
    assert [m["mineral"] for m in sp.meta["xrd_matches"]] == ["Quartz"]
    assert calls and calls[0][0] == sp.id
    assert calls[0][1] == {"xrd_match": None, "xrd_matches": None}


def test_check_raman_phases_overlays_reference_lines(qtbot, db_path):
    widget, library, sp = _workspace(qtbot, db_path)
    # a Raman spectrum of the same sample, with an accepted Mineral ID phase
    raman = Spectrum(id=Spectrum.new_id(), title="raman_sample", path="", kind="raman_xy",
                     x=np.linspace(100, 1200, 500), y=np.ones(500))
    raman.meta["rruff_matches"] = [{"mineral": "Quartz", "rruff_id": "R040031"}]
    library.add(raman)

    widget.check_raman_phases()
    qtbot.wait(20)
    assert widget.results_table.rowCount() == 1
    assert widget.results_table.item(0, 1).text() == "Quartz"
    assert "Quartz" in widget.db_status_label.text()


def test_card_browser_lookup(qtbot, db_path):
    widget, _, _ = _workspace(qtbot, db_path)
    widget.browse_edit.setText("calcite")
    widget.browse_cards()
    assert widget.browse_list.count() == 1
    assert "Calcite" in widget.browse_list.item(0).text()
    assert "TESTDB 1020" in widget.browse_list.item(0).text()


def test_shell_has_xrd_id_page(qtbot):
    window = DataappMainWindow()
    qtbot.addWidget(window)
    window.nav.setCurrentRow(NAV_ITEMS.index("XRD ID"))
    qtbot.wait(20)
    assert window.stack.currentWidget() is window.xrd_id_page


def test_xrd_accept_is_undoable_through_shell(qtbot, db_path):
    window = DataappMainWindow()
    qtbot.addWidget(window)
    sp = _quartz_pattern()
    window.library.add(sp)
    # simulate what the workspace's on_accept wiring records
    window.library_page.push_undo(("xrd_ident", sp.id, {"xrd_match": None, "xrd_matches": None}))
    sp.meta["xrd_match"] = {"mineral": "Quartz"}
    sp.meta["xrd_matches"] = [{"mineral": "Quartz"}]

    window.library_page._undo()
    assert "xrd_match" not in sp.meta
    assert "xrd_matches" not in sp.meta
