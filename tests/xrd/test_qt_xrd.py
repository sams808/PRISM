"""Tests for qt_xrd.py — the QualX-style XRD phase-ID workspace.
Uses the same tiny synthetic QualX-format database as test_xrd_id_science."""
from __future__ import annotations

import numpy as np
import pytest

from PySide6.QtWidgets import QFileDialog, QMessageBox

import xrd.xrd_id_science as xid
from core.qt_models import Spectrum, SpectrumLibrary
from core.qt_shell import NAV_ITEMS, PrismMainWindow
from xrd.qt_xrd import XrdIdWorkspace
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


CALCITE_LINES = [3.035, 2.095, 1.913, 3.86]


def _mixed_pattern() -> Spectrum:
    """Quartz + calcite, so the iterative two-phase workflow can be driven
    end to end (a single-phase pattern cannot surface the second card)."""
    x = np.linspace(10, 80, 4000)
    y = np.full_like(x, 50.0)
    for d_list, ints in ((QUARTZ_D, QUARTZ_I), (CALCITE_LINES, [100.0, 60.0, 55.0, 40.0])):
        for c, i in zip(xid.d_to_two_theta(np.array(d_list)), ints):
            y += 10.0 * i * np.exp(-0.5 * ((x - c) / 0.05) ** 2)
    return Spectrum(id=Spectrum.new_id(), title="mixture", path="", kind="xrd_xy", x=x, y=y)


def _workspace(qtbot, db_path, on_accept=None):
    library = SpectrumLibrary()
    sp = _quartz_pattern()
    library.add(sp)
    widget = XrdIdWorkspace(library=library, db_path=db_path, on_accept=on_accept)
    qtbot.addWidget(widget)
    widget.set_spectra([sp.id])
    # set_spectra queues the entry preview through the 120ms debounce —
    # let it complete INSIDE the test, or matplotlib's own singleShot(0)
    # idle-draw lands on a torn-down canvas in the next test.
    qtbot.wait(200)
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


def test_database_list_shows_and_toggles(qtbot, db_path):
    from PySide6.QtCore import Qt
    widget, _, _ = _workspace(qtbot, db_path)
    assert widget.db_list.count() == 1
    assert widget.db_list.item(0).checkState() == Qt.Checked
    assert widget._enabled_paths() == [db_path]
    widget.db_list.item(0).setCheckState(Qt.Unchecked)
    assert widget._enabled_paths() == []  # unchecked databases are not probed
    qtbot.wait(200)  # flush the deferred list rebuild + debounced render before teardown


def test_add_database_and_search_probes_both(qtbot, db_path, tmp_path):
    """Register a second database at runtime; one search must probe both
    (the user's 'allow them to use multiple if wanted')."""
    src = tmp_path / "extra_src.sq"
    _make_source_sq(src, [
        (9, "Cristobalite", "Cristobalite", "Si O2", "P 41 21 2", "A",
         [4.05, 2.485, 2.841, 3.135], [100.0, 20.0, 13.0, 11.0], ["Si", "O"]),
    ])
    db2 = tmp_path / "extra.sq"
    xid.build_xrd_database([(str(src), "EXTRA")], out_path=str(db2), log=lambda *a: None)

    widget, _, _ = _workspace(qtbot, db_path)
    widget._register_paths([str(db2)])
    qtbot.wait(20)
    assert widget.db_list.count() == 2
    assert len(widget._enabled_paths()) == 2
    assert "Added: extra" in widget.db_status_label.text()

    # a quartz+cristobalite mixture: each phase only exists in one database.
    # No spectrum selected -> equal peak intensities, so the strong-line
    # prefilter probes both phases' lines (with a spectrum, weak second-phase
    # lines are found via the iterative Accept workflow instead).
    tt_q = xid.d_to_two_theta(np.array(QUARTZ_D))
    tt_c = xid.d_to_two_theta(np.array([4.05, 2.485, 2.841, 3.135]))
    widget.peaks_edit.setText(", ".join(f"{v:.3f}" for v in np.concatenate([tt_q, tt_c])))
    widget.spec_combo.setCurrentIndex(-1)
    widget.run_search()
    qtbot.wait(20)
    found = {widget.results_table.item(r, 1).text() for r in range(widget.results_table.rowCount())}
    assert {"Quartz", "Cristobalite"} <= found
    qtbot.wait(200)  # flush the debounced preview render before teardown


def test_add_rejects_non_database_file(qtbot, db_path, tmp_path):
    junk = tmp_path / "junk.sq"
    junk.write_text("not sqlite at all")
    widget, _, _ = _workspace(qtbot, db_path)
    widget._register_paths([str(junk)])
    qtbot.wait(20)
    assert widget.db_list.count() == 1  # nothing added
    assert widget.add_db_btn.isEnabled()  # buttons re-enabled after the error


def test_entry_preview_shows_diffractogram(qtbot, db_path):
    """Arriving on the page renders the query pattern immediately (user
    report: 'where tf is my diffractogram')."""
    widget, _, _ = _workspace(qtbot, db_path)
    axes = widget.plot.figure.get_axes()
    assert axes and axes[0].lines


def test_accept_keeps_overlay_and_clear_resets(qtbot, db_path):
    widget, _, sp = _workspace(qtbot, db_path)
    widget.auto_find_peaks()
    widget.run_search()
    qtbot.wait(20)
    widget.results_table.selectRow(0)
    widget.accept_selected()
    qtbot.wait(20)
    state = widget._state()
    assert len(state["accepted"]) == 1   # the accepted phase stays overlaid (QualX-style)
    assert state["explained"]            # its peaks turn gray instead of vanishing

    widget.clear_session()
    qtbot.wait(250)
    assert widget._state() == {"accepted": [], "explained": []}
    assert widget.peaks_edit.text() == ""
    assert widget.results_table.rowCount() == 0


def test_crystal_system_filter_narrows_results(qtbot, db_path):
    widget, _, _ = _workspace(qtbot, db_path)
    widget.auto_find_peaks()
    widget.system_filter._actions["cubic"].setChecked(True)  # quartz is trigonal
    widget.run_search()
    qtbot.wait(20)
    assert widget.results_table.rowCount() == 0
    widget.system_filter._actions["cubic"].setChecked(False)
    widget.system_filter._actions["trigonal"].setChecked(True)
    widget.run_search()
    qtbot.wait(20)
    assert widget.results_table.rowCount() >= 1


def test_quality_filter_populated_from_databases(qtbot, db_path):
    widget, _, _ = _workspace(qtbot, db_path)
    assert set(widget.quality_filter._actions) == {"A"}  # the fixture DB's only quality code


def test_zoom_preserved_when_testing_other_cards(qtbot, db_path):
    """User request: a zoom on some peaks must survive re-renders (e.g.
    selecting another candidate card); Clear autoscales again."""
    widget, _, _ = _workspace(qtbot, db_path)
    widget.auto_find_peaks()
    widget.run_search()
    qtbot.wait(200)  # results preview rendered
    ax = widget.plot.figure.get_axes()[0]
    widget.plot.toolbar.push_current()  # what a real toolbar zoom does first
    ax.set_xlim(25.0, 28.0)
    ax.set_ylim(-20.0, 60.0)

    widget.plot.request_redraw(widget.render_preview)  # another card clicked
    qtbot.wait(200)
    ax2 = widget.plot.figure.get_axes()[0]
    assert ax2.get_xlim() == (25.0, 28.0)
    assert ax2.get_ylim() == (-20.0, 60.0)

    widget.clear_session()  # starting over zooms back out
    qtbot.wait(250)
    assert widget.plot.figure.get_axes()[0].get_xlim() != (25.0, 28.0)


def test_shell_has_xrd_id_page(qtbot):
    window = PrismMainWindow()
    qtbot.addWidget(window)
    window.nav.setCurrentRow(NAV_ITEMS.index("XRD ID"))
    qtbot.wait(20)
    assert window.stack.currentWidget() is window.xrd_id_page


def test_xrd_accept_is_undoable_through_shell(qtbot, db_path):
    window = PrismMainWindow()
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


# ---------------------------------------------------------------------------
# Peak finding in the workspace
# ---------------------------------------------------------------------------

def test_auto_find_uses_the_xrd_finder_and_reports_what_it_did(qtbot, db_path):
    widget, _, _ = _workspace(qtbot, db_path)
    widget.auto_find_peaks()
    peaks = widget._parse_peaks()
    assert peaks == sorted(peaks)
    # every quartz line in range is found, at its true position
    for d in QUARTZ_D:
        tt = float(xid.d_to_two_theta(np.array([d]))[0])
        if 10.0 < tt < 80.0:
            assert any(abs(p - tt) < 0.05 for p in peaks), f"missed the line at {tt:.2f}"
    assert "peaks at" in widget.peak_status_label.text()
    assert "background removed" in widget.peak_status_label.text()
    qtbot.wait(200)


def test_peak_intensities_are_background_corrected_in_the_workspace(qtbot, db_path):
    """The query intensities the search is ranked on must be the peaks',
    not the peaks plus the background sitting under them."""
    widget, library, sp = _workspace(qtbot, db_path)
    sp.y = sp.y + 4000.0 * np.exp(-0.5 * ((sp.x - 20.0) / 5.0) ** 2)  # an amorphous halo
    widget.auto_find_peaks()
    peaks = widget._parse_peaks()
    corrected = widget._peak_intensities(peaks)
    widget.background_check.setChecked(False)
    raw = widget._peak_intensities(peaks)
    # the strongest quartz line is at 26.6 deg, on the halo's flank; with the
    # halo counted in, low-angle peaks are inflated relative to high-angle ones
    lo = min(range(len(peaks)), key=lambda k: peaks[k])
    hi = max(range(len(peaks)), key=lambda k: peaks[k])
    assert raw[lo] / raw[hi] > corrected[lo] / corrected[hi]
    qtbot.wait(200)


def test_shift_tolerance_is_passed_to_the_search(qtbot, db_path):
    """A displaced pattern (every line off by the same amount) must still
    be identified, and the offset reported in the results table."""
    widget, _, sp = _workspace(qtbot, db_path)
    tt = xid.d_to_two_theta(np.array(QUARTZ_D)) + 0.12
    widget.peaks_edit.setText(", ".join(f"{v:.3f}" for v in sorted(tt)))
    widget.shift_edit.setText("0.15")
    widget.run_search()
    qtbot.wait(50)
    assert widget.results_table.item(0, 1).text() == "Quartz"
    assert int(widget.results_table.item(0, 0).text()) > 85
    assert float(widget.results_table.item(0, 8).text()) == pytest.approx(0.12, abs=0.03)
    qtbot.wait(200)


# ---------------------------------------------------------------------------
# Managing accepted phases
# ---------------------------------------------------------------------------

def _accept_first(qtbot, widget):
    widget.auto_find_peaks()
    widget.run_search()
    qtbot.wait(30)
    widget.results_table.selectRow(0)
    widget.accept_selected()
    qtbot.wait(30)


def test_accepted_phase_appears_in_the_ledger(qtbot, db_path):
    widget, _, _ = _workspace(qtbot, db_path)
    _accept_first(qtbot, widget)
    assert widget.accepted_table.rowCount() == 1
    assert widget.accepted_table.item(0, 0).text() == "Quartz"
    assert widget.accepted_table.item(0, 3).text() == "1010"
    assert int(widget.accepted_table.item(0, 5).text()) >= 1  # peaks explained
    qtbot.wait(200)


def test_removing_an_accepted_phase_returns_its_peaks_and_its_record(qtbot, db_path):
    calls = []
    widget, _, sp = _workspace(qtbot, db_path, on_accept=lambda sid, old: calls.append((sid, old)))
    widget.auto_find_peaks()
    before = set(widget._parse_peaks())
    widget.run_search()
    qtbot.wait(30)
    widget.results_table.selectRow(0)
    widget.accept_selected()
    qtbot.wait(30)
    assert sp.meta["xrd_match"]["mineral"] == "Quartz"
    after_accept = set(widget._parse_peaks())
    assert after_accept < before  # the explained peaks left the search

    widget.accepted_table.selectRow(0)
    widget.remove_accepted_phase()
    qtbot.wait(30)

    assert widget.accepted_table.rowCount() == 0
    assert widget._state()["accepted"] == []
    assert widget._state()["explained"] == []
    assert "xrd_match" not in sp.meta and "xrd_matches" not in sp.meta
    # the peaks it explained are searchable again
    assert {round(v, 3) for v in widget._parse_peaks()} == {round(v, 3) for v in before}
    # and the removal is undoable, like the accept
    assert len(calls) == 2
    assert calls[1][1]["xrd_matches"] is not None
    qtbot.wait(200)


def test_removing_one_of_two_phases_keeps_the_other(qtbot, db_path):
    """A two-phase mixture worked the way the workspace intends it: accept,
    re-search the remainder, accept again - then withdraw ONE of them."""
    library = SpectrumLibrary()
    sp = _mixed_pattern()
    library.add(sp)
    widget = XrdIdWorkspace(library=library, db_path=db_path)
    qtbot.addWidget(widget)
    widget.set_spectra([sp.id])
    qtbot.wait(200)
    tt_c = xid.d_to_two_theta(np.array(CALCITE_LINES))
    widget.auto_find_peaks()
    widget.run_search()
    qtbot.wait(30)
    for name in ("Quartz", "Calcite"):
        row = next(r for r in range(widget.results_table.rowCount())
                   if widget.results_table.item(r, 1).text() == name)
        widget.results_table.selectRow(row)
        widget.accept_selected()
        qtbot.wait(30)
    assert widget.accepted_table.rowCount() == 2
    assert [m["mineral"] for m in sp.meta["xrd_matches"]] == ["Quartz", "Calcite"]

    quartz_row = next(r for r in range(widget.accepted_table.rowCount())
                      if widget.accepted_table.item(r, 0).text() == "Quartz")
    widget.accepted_table.selectRow(quartz_row)
    widget.remove_accepted_phase()
    qtbot.wait(30)

    assert [widget.accepted_table.item(r, 0).text()
            for r in range(widget.accepted_table.rowCount())] == ["Calcite"]
    assert [m["mineral"] for m in sp.meta["xrd_matches"]] == ["Calcite"]
    # calcite's peaks stay explained; only quartz's came back
    explained = widget._state()["explained"]
    assert explained and all(any(abs(p - v) < 0.2 for v in tt_c) for p in explained)
    qtbot.wait(200)


def test_clear_withdraws_the_phases_it_stops_showing(qtbot, db_path):
    """Clearing used to drop the overlays while leaving the phases recorded
    on the spectrum — a pattern that looked untouched but still carried
    identifications into a report."""
    widget, _, sp = _workspace(qtbot, db_path)
    _accept_first(qtbot, widget)
    assert sp.meta.get("xrd_matches")

    widget.clear_session()
    qtbot.wait(250)
    assert widget._state() == {"accepted": [], "explained": []}
    assert widget.accepted_table.rowCount() == 0
    assert "xrd_matches" not in sp.meta and "xrd_match" not in sp.meta


def test_accepted_phases_are_rebuilt_from_metadata_in_a_new_session(qtbot, db_path):
    """A saved project reopened, or PRISM restarted: the identifications
    live in the spectrum's metadata, and the ledger (and its stick
    overlay) must come back from them."""
    library = SpectrumLibrary()
    sp = _quartz_pattern()
    sp.meta["xrd_matches"] = [{"mineral": "Quartz", "name": "Quartz low", "formula": "Si O2",
                               "source": "TESTDB", "source_code": "1010",
                               "spacegroup": "P 32 2 1", "fom": 0.94, "n_matched": 7}]
    sp.meta["xrd_match"] = sp.meta["xrd_matches"][0]
    library.add(sp)
    widget = XrdIdWorkspace(library=library, db_path=db_path)
    qtbot.addWidget(widget)
    widget.set_spectra([sp.id])
    qtbot.wait(200)

    assert widget.accepted_table.rowCount() == 1
    assert widget.accepted_table.item(0, 0).text() == "Quartz"
    entry = widget._state()["accepted"][0]
    assert len(entry["match"].d) == len(QUARTZ_D)  # the card was looked up again
    # and it can be withdrawn like any other
    widget.accepted_table.selectRow(0)
    widget.remove_accepted_phase()
    qtbot.wait(30)
    assert "xrd_matches" not in sp.meta
    qtbot.wait(200)


# ---------------------------------------------------------------------------
# CIF export
# ---------------------------------------------------------------------------

def test_export_writes_cifs_for_cod_phases_and_reports_the_rest(qtbot, db_path, tmp_path, monkeypatch):
    widget, _, sp = _workspace(qtbot, db_path)
    _accept_first(qtbot, widget)
    # re-tag the accepted card as a COD entry with a structure in a local
    # mirror, plus a PDF-2 card that cannot have one
    entry = widget._state()["accepted"][0]
    entry["match"].source, entry["match"].source_code = "COD2205", "1011172"
    entry["match"].mineral = "Quartz"
    widget._state()["accepted"].append({
        "match": xid.XrdMatch(card_id=-1, source="PDF2", source_code="500493",
                              name="", mineral="Lead Bismuth Vanadium Oxide",
                              formula="Pb4 Bi V O8", spacegroup="", quality="", rir=None,
                              fom=0.4, cov_card=0.0, cov_query=0.0, n_matched=3),
        "peaks": [], "record": None})

    mirror = tmp_path / "mirror"
    mirror.mkdir()
    (mirror / "1011172.cif").write_text("data_quartz\n_cell_length_a 4.913\n")
    out = tmp_path / "rietveld"
    out.mkdir()
    widget.cif_folder_edit.setText(str(mirror))
    widget.cif_download_check.setChecked(False)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(out)))
    shown = []
    monkeypatch.setattr(QMessageBox, "information",
                        staticmethod(lambda *a, **k: shown.append(a[-1])))

    widget._refresh_accepted_table()
    assert widget.accepted_table.item(1, 6).text().startswith("no structure")
    widget.export_accepted_cifs()
    qtbot.wait(300)

    assert (out / "Quartz_COD1011172.cif").is_file()
    assert (out / "phases.csv").is_file()
    assert shown and "1 structure file(s) written" in shown[0]
    assert "no structure available" in shown[0]  # the PDF-2 card is named, not hidden
    qtbot.wait(200)


def test_export_without_accepted_phases_says_so(qtbot, db_path, monkeypatch):
    widget, _, _ = _workspace(qtbot, db_path)
    shown = []
    monkeypatch.setattr(QMessageBox, "information",
                        staticmethod(lambda *a, **k: shown.append(a[-1])))
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("must not ask for a folder with nothing to export"))))
    widget.export_accepted_cifs()
    assert shown and "No accepted phases" in shown[0]
    qtbot.wait(200)


def test_add_cif_folder_makes_the_structures_searchable(qtbot, db_path, tmp_path):
    """The user's own structure files become a database like any other:
    calculated once, then probed alongside the card databases."""
    folder = tmp_path / "structures"
    folder.mkdir()
    (folder / "halite.cif").write_text("\n".join([
        "data_x", "_chemical_name_mineral 'Halite'", "_chemical_formula_sum 'Na Cl'",
        "_symmetry_space_group_name_H-M 'F m -3 m'",
        "_cell_length_a 5.64", "_cell_length_b 5.64", "_cell_length_c 5.64",
        "_cell_angle_alpha 90", "_cell_angle_beta 90", "_cell_angle_gamma 90",
        "loop_", "_symmetry_equiv_pos_as_xyz",
        "x,y,z", "-x,-y,z", "-x,y,-z", "x,-y,-z",
        "x,y+1/2,z+1/2", "-x,-y+1/2,z+1/2", "-x,y+1/2,-z+1/2", "x,-y+1/2,-z+1/2",
        "x+1/2,y,z+1/2", "-x+1/2,-y,z+1/2", "-x+1/2,y,-z+1/2", "x+1/2,-y,-z+1/2",
        "x+1/2,y+1/2,z", "-x+1/2,-y+1/2,z", "-x+1/2,y+1/2,-z", "x+1/2,-y+1/2,-z",
        "loop_", "_atom_site_label", "_atom_site_type_symbol",
        "_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z",
        "Na1 Na 0 0 0", "Cl1 Cl 0.5 0.5 0.5",
    ]) + "\n", encoding="utf-8")

    widget, _, _ = _workspace(qtbot, db_path)
    widget._register_paths([str(folder)], cif_mode=True)
    qtbot.wait(400)
    assert widget.db_list.count() == 2
    assert "structures" in widget.db_status_label.text()
    assert widget.add_cif_folder_btn.isEnabled()

    # halite's own calculated pattern, searched against everything enabled
    from xrd import cif_patterns as cifp
    pat = cifp.powder_pattern_from_cif(str(folder / "halite.cif"), two_theta_max=80)
    widget.peaks_edit.setText(", ".join(f"{v:.3f}" for v in pat.two_theta))
    widget.spec_combo.setCurrentIndex(-1)
    widget.run_search()
    qtbot.wait(60)
    names = [widget.results_table.item(r, 1).text()
             for r in range(widget.results_table.rowCount())]
    assert "Halite" in names
    qtbot.wait(200)


# ---------------------------------------------------------------------------
# Peak search options
# ---------------------------------------------------------------------------

def test_peak_options_dialog_reads_and_writes_the_whole_setup(qtbot, db_path):
    widget, _, _ = _workspace(qtbot, db_path)
    widget.open_peak_options()
    dlg = widget._peak_dialog
    qtbot.addWidget(dlg)
    assert dlg is not None
    # it opens showing what the search is actually using
    assert float(dlg._fields["min_prominence_sigma"].text()) == \
        widget.peak_options.min_prominence_sigma
    assert dlg.local_noise_check.isChecked() is widget.peak_options.local_noise
    # an optional setting left blank stays "automatic", not zero
    assert dlg._fields["background_width"].text() == ""
    assert dlg.options().background_width is None

    dlg._fields["min_prominence_sigma"].setText("6")
    dlg._fields["min_separation"].setText("0.2")
    dlg._fields["two_theta_max"].setText("55")
    dlg._fields["max_peaks"].setText("15")
    dlg.local_noise_check.setChecked(False)
    dlg.ka2_check_dlg.setChecked(False)
    opts = dlg.options()
    assert opts.min_prominence_sigma == 6.0
    assert opts.min_separation == 0.2
    assert opts.two_theta_max == 55.0
    assert opts.max_peaks == 15
    assert opts.local_noise is False and opts.flag_ka2 is False
    qtbot.wait(200)


def test_applying_options_re_runs_the_search_and_updates_the_panel(qtbot, db_path):
    widget, _, _ = _workspace(qtbot, db_path)
    widget.auto_find_peaks()
    before = len(widget._parse_peaks())
    assert before > 0

    widget.open_peak_options()
    dlg = widget._peak_dialog
    qtbot.addWidget(dlg)
    dlg._fields["min_prominence_sigma"].setText("50")   # absurdly strict
    dlg._apply()
    qtbot.wait(50)
    # the search really re-ran, and the left panel now agrees with it
    assert len(widget._parse_peaks()) < before
    assert widget.detection_edit.text() == "50"
    assert widget.peak_options.min_prominence_sigma == 50.0
    assert "50σ" in widget.peak_status_label.text()

    dlg._restore_defaults()
    qtbot.wait(50)
    assert widget.peak_options.min_prominence_sigma == 4.0
    assert widget.detection_edit.text() == "4"
    qtbot.wait(200)


def test_search_range_option_restricts_the_reported_peaks(qtbot, db_path):
    widget, _, _ = _workspace(qtbot, db_path)
    widget.auto_find_peaks()
    everything = widget._parse_peaks()
    assert any(p < 30.0 for p in everything)

    widget._apply_peak_options(xid.PeakSearchOptions(two_theta_min=30.0, two_theta_max=60.0))
    qtbot.wait(50)
    windowed = widget._parse_peaks()
    assert windowed and all(30.0 <= p <= 60.0 for p in windowed)
    assert windowed == [p for p in everything if 30.0 <= p <= 60.0]
    assert "range 30–60°" in widget.peak_status_label.text()
    qtbot.wait(200)


def test_panel_controls_stay_authoritative_over_the_dialog(qtbot, db_path):
    """The three settings shown in both places must never disagree."""
    widget, _, _ = _workspace(qtbot, db_path)
    widget.detection_edit.setText("7")
    widget.ka2_check.setChecked(False)
    widget.open_peak_options()
    dlg = widget._peak_dialog
    qtbot.addWidget(dlg)
    assert dlg._fields["min_prominence_sigma"].text() == "7"
    assert dlg.ka2_check_dlg.isChecked() is False
    qtbot.wait(200)


def test_show_threshold_draws_the_background_and_the_limit(qtbot, db_path):
    """The overlay is what makes the settings tunable: it answers 'why was
    that not a peak?' by drawing the level it had to clear."""
    widget, _, _ = _workspace(qtbot, db_path)
    widget.auto_find_peaks()
    qtbot.wait(50)
    widget.render_preview()
    labels_off = {t.get_label() for t in widget.plot.figure.get_axes()[0].get_lines()}
    assert "background" not in labels_off

    widget.show_trace_check.setChecked(True)
    qtbot.wait(250)
    widget.render_preview()
    ax = widget.plot.figure.get_axes()[0]
    labels = {t.get_label() for t in ax.get_lines()}
    assert "background" in labels
    assert any(lbl.startswith("detection limit") for lbl in labels)
    # the drawn limit sits above the drawn background everywhere
    lines = {t.get_label(): t for t in ax.get_lines()}
    bg = lines["background"].get_ydata()
    lim = next(t for lbl, t in lines.items() if lbl.startswith("detection limit")).get_ydata()
    assert np.all(np.asarray(lim) >= np.asarray(bg))
    qtbot.wait(200)


def test_peak_options_survive_a_new_workspace(qtbot, db_path):
    """Tuned once, not re-tuned on every pattern: the setup is stored and
    restored (conftest swaps QSettings for an in-memory store, so this
    exercises the save/load path without touching the real registry)."""
    library = SpectrumLibrary()
    sp = _quartz_pattern()
    library.add(sp)
    first = XrdIdWorkspace(library=library, db_path=db_path)
    qtbot.addWidget(first)
    first.set_spectra([sp.id])
    qtbot.wait(200)
    first._apply_peak_options(xid.PeakSearchOptions(min_prominence_sigma=8.5,
                                                    min_separation=0.33,
                                                    flag_ka2=False))
    qtbot.wait(50)

    second = XrdIdWorkspace(library=library, db_path=db_path)
    qtbot.addWidget(second)
    second.set_spectra([sp.id])
    qtbot.wait(200)
    assert second.peak_options.min_prominence_sigma == 8.5
    assert second.peak_options.min_separation == 0.33
    assert second.peak_options.flag_ka2 is False
    assert second.detection_edit.text() == "8.5"
    assert second.ka2_check.isChecked() is False
    qtbot.wait(200)
