"""Thermo OMNIC readers (the group's DXR Raman microscope): the .spa binary
against OMNIC's own text export of the same spectrum, and the .mapx map
container (both examples are real files of the instrument)."""
from __future__ import annotations

import numpy as np
import pytest

import core.io_universal as iou
import raman.thermo_omnic as om


def test_spa_matches_the_text_export(examples_dir):
    spa = str(examples_dir / "Raman_example_omnic.spa")
    txt = np.loadtxt(str(examples_dir / "Raman_example_omnic.txt"))
    assert om.is_spa(spa)
    x, y, meta = om.read_spa(spa)
    assert x.shape == y.shape == (txt.shape[0],)
    assert np.all(np.diff(x) > 0)
    assert np.allclose(x, txt[:, 0], atol=2e-3)                       # 1.9285 cm-1 grid, float32 endpoints
    # OMNIC's text export differs from the stored float32 values by up to
    # 0.13 % (median ratio 1.00017): a final processing/rounding step of
    # the export, not a reading error — the shape, order and scale agree.
    assert np.allclose(y, txt[:, 1], rtol=2e-3)
    assert np.median(y / txt[:, 1]) == pytest.approx(1.0, abs=5e-4)
    assert meta["n_points"] == txt.shape[0]
    assert meta["stage_x"] == 1266 and meta["stage_y"] == 7183 and meta["snr"] == pytest.approx(90.9)
    assert meta["exposure_s"] == pytest.approx(0.5)


def test_spa_goes_through_the_importer(examples_dir):
    df, meta = iou.load_any(str(examples_dir / "Raman_example_omnic.spa"), return_meta=True)
    assert meta["selected_parser"] == "omnic_spa"
    c = meta["canonical_map"]
    assert c["X"] == "shift_cm-1" and len(df) > 1000 and meta["snr"] == pytest.approx(90.9)


def test_spa_rejects_other_files(examples_dir, tmp_path):
    p = tmp_path / "x.spa"
    p.write_bytes(b"not a spectral data file" * 40)
    assert not om.is_spa(str(p))
    with pytest.raises(ValueError):
        om.read_spa(str(p))


def test_mapx_regions_axis_and_metadata(examples_dir):
    p = str(examples_dir / "Raman_map_example.mapx")
    assert om.is_mapx(p)
    m = om.read_mapx(p)
    assert [r.name for r in m.regions] == ["Region6", "Region7"]
    assert all(r.data.shape == (2, 2, 1738) for r in m.regions)
    assert m.n_spectra == 8
    assert m.x[0] == pytest.approx(49.9967, abs=1e-3) and m.x[-1] == pytest.approx(3399.74, abs=1e-2)
    assert np.median(np.diff(m.x)) == pytest.approx(1.9285, abs=1e-3)    # the step of the text exports
    r = m.regions[0]
    assert r.stage_box == pytest.approx((2472.076, -5118.555, 2572.076, -5018.555), abs=1e-3)  # nm -> um
    assert r.x_positions[0] == pytest.approx(2472.076, abs=1e-3) and r.x_positions[-1] == pytest.approx(2572.076, abs=1e-3)
    assert np.allclose(r.mean_spectrum(), r.data.reshape(-1, 1738).mean(axis=0))
    assert m.meta["laser_mW"] == 9 and m.meta["n_scans"] == 50 and m.meta["exposure_s"] == pytest.approx(0.1)
    assert m.meta["objective"] == "50x LMPlan" and m.meta["grating_lines_mm"] == 900
    assert m.meta["spectral_type"] == "Raman Shift"
    assert "2 region(s), 8 spectra" in om.mapx_summary(m)


def test_mapx_through_the_importer_and_the_library(examples_dir, monkeypatch):
    p = str(examples_dir / "Raman_map_example.mapx")
    df, meta = iou.load_any(p, return_meta=True)
    assert meta["selected_parser"] == "omnic_mapx" and meta["n_spectra"] == 8 and len(df) == 1738
    import core.qt_library as ql
    spectra = ql.load_spectra_from_path(p)
    titles = [s.title for s in spectra]
    assert len(spectra) == 2 * (1 + 4)
    assert titles[0] == "Raman_map_example_Region6_mean" and "Raman_map_example_Region6_x1_y0" in titles
    px = next(s for s in spectra if s.title.endswith("_x1_y1"))
    assert px.meta["stage_um"] == pytest.approx([2572.076, -5018.555], abs=1e-3)
    assert px.meta["omnic"]["laser_mW"] == 9
    # above the cap only the region means are imported, and the metadata says so
    monkeypatch.setattr(ql, "MAP_PIXEL_IMPORT_CAP", 4)
    capped = ql.load_spectra_from_path(p)
    assert [s.title for s in capped] == ["Raman_map_example_Region6_mean", "Raman_map_example_Region7_mean"]
    assert all(s.meta["pixels_imported"] is False for s in capped)


def test_library_page_imports_a_map(qtbot, examples_dir, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    from core.qt_library import LibraryPage
    from core.qt_models import SpectrumLibrary
    shown = []
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: shown.append(a[2]) or QMessageBox.Ok))
    page = LibraryPage(SpectrumLibrary())
    qtbot.addWidget(page)
    added, errors = page.import_paths([str(examples_dir / "Raman_map_example.mapx"), str(examples_dir / "Raman_example_omnic.spa")])
    assert added == 11 and errors == []
    assert page.table.rowCount() == 11
    assert shown and "8 spectra" in shown[0] and "10 spectra imported" in shown[0]
    qtbot.wait(20)  # let the preview canvas's deferred draw_idle() finish before teardown
