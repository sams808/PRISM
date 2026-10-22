"""The Home page: module cards, presets, the database inventory and the
pack export / import flows (on temp caches — see conftest's hermetic
fixtures)."""
from __future__ import annotations

import json
import os

import numpy as np

import core.db_packs as dbp


def _fake_caches(tmp_path):
    import raman.rruff_science as rs
    import raman.raman_library as rl
    raw = os.path.join(rs.RRUFF_CACHE_DIR, "raw")
    os.makedirs(raw, exist_ok=True)
    fn = os.path.join(raw, "Quartz__R000001__Raman__532__0__unoriented__Raman_Data_Processed__1.txt")
    with open(fn, "w", encoding="utf-8") as f:
        f.write("##NAMES=Quartz\n##RRUFFID=R000001\n" + "\n".join(f"{100 + i}, {10 if i != 50 else 500}" for i in range(200)))
    with open(os.path.join(rs.RRUFF_CACHE_DIR, "index.json"), "w", encoding="utf-8") as f:
        json.dump([{"mineral": "Quartz", "rruff_id": "R000001", "wavelength_nm": 532.0, "peaks": [150.0],
                    "raw_path": fn, "category": "excellent_unoriented"}], f)
    x = np.linspace(100, 1000, 91)
    rl.save_library([rl.entry_from_spectrum("std", x, np.exp(-0.5 * ((x - 500) / 20) ** 2))])


def test_home_lists_the_installed_databases(qtbot, tmp_path):
    from core.qt_shell import PrismMainWindow
    _fake_caches(tmp_path)
    window = PrismMainWindow()
    qtbot.addWidget(window)
    home = window.open_settings("Reference").db_panel
    home.refresh_databases(force=True)  # workers run inline in tests
    assert set(home.db_rows) == set(dbp.ALL_KEYS)
    assert "1 spectra" in home.db_rows["rruff"][1].text()
    assert "installed" in home.db_rows["raman_library"][1].text()
    assert "not installed" in home.db_rows["amcsd"][1].text()
    assert "2 of 5" in home.db_status.text()
    window.home_page.refresh_databases()
    assert "RRUFF" in window.home_page.db_line.text()


def test_home_export_then_import_round_trip(qtbot, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog, QMessageBox
    from core.qt_shell import PrismMainWindow
    import raman.rruff_science as rs
    import raman.raman_library as rl
    _fake_caches(tmp_path)
    window = PrismMainWindow()
    qtbot.addWidget(window)
    home = window.open_settings("Reference").db_panel
    home.refresh_databases(force=True)

    pack = tmp_path / "pack"
    pack.mkdir()
    shown = []
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: str(pack)))
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes))
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: shown.append(a[2]) or QMessageBox.Ok))
    home.export_databases()
    assert (pack / dbp.MANIFEST_NAME).is_file() and (pack / "rruff_pack.sq").is_file() and (pack / "raman_library.json").is_file()
    assert shown and "Pack written" in shown[-1]

    # wipe the caches, import the pack back
    import shutil
    shutil.rmtree(rs.RRUFF_CACHE_DIR)
    os.remove(rl.LIBRARY_PATH)
    home.refresh_databases(force=True)
    assert "not installed" in home.db_rows["rruff"][1].text()
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(pack / dbp.MANIFEST_NAME), "")))
    home.import_databases()
    assert "Installed" in shown[-1]
    assert os.path.isfile(os.path.join(rs.RRUFF_CACHE_DIR, "index.json"))
    assert len(rl.load_library()) == 1
    assert "1 spectra" in home.db_rows["rruff"][1].text()
    assert not home._db_busy and home.db_export_btn.isEnabled()
