"""Tests for the shell's project save/load wiring (M14)."""
from __future__ import annotations

import numpy as np
import pytest
import rampy as rp

from qt_models import Spectrum
from qt_shell import DataappMainWindow


def _spectrum(title="s1"):
    x = np.linspace(400, 600, 200)
    y = rp.gaussian(x, 80.0, 505.0, 25.0)
    return Spectrum(id=Spectrum.new_id(), title=title, path="", kind="raman_xy", x=x, y=y)


def test_shell_project_round_trip(qtbot, tmp_path, monkeypatch):
    window = DataappMainWindow()
    qtbot.addWidget(window)
    qtbot.wait(20)

    sp = _spectrum("session_spectrum")
    window.library.add(sp)
    window.fit_param_memory.set(sp.id, [{"shape": "G", "shift_val": 505.0}])

    project_path = tmp_path / "roundtrip.dataapp"
    monkeypatch.setattr("qt_shell.QFileDialog.getSaveFileName", staticmethod(lambda *a, **k: (str(project_path), "")))
    window.save_project()
    assert project_path.exists()

    # Fresh window, load the project back.
    window2 = DataappMainWindow()
    qtbot.addWidget(window2)
    qtbot.wait(20)
    monkeypatch.setattr("qt_shell.QFileDialog.getOpenFileName", staticmethod(lambda *a, **k: (str(project_path), "")))
    window2.open_project()

    loaded = window2.library.all()
    assert len(loaded) == 1
    assert loaded[0].title == "session_spectrum"
    assert loaded[0].id == sp.id
    assert np.allclose(loaded[0].x, sp.x)
    assert window2.fit_param_memory.get(sp.id)[0]["shift_val"] == 505.0
    # Library table refreshed too.
    assert window2.library_page.table.rowCount() == 1


def test_save_project_appends_extension(qtbot, tmp_path, monkeypatch):
    window = DataappMainWindow()
    qtbot.addWidget(window)
    qtbot.wait(20)
    window.library.add(_spectrum())

    bare_path = tmp_path / "noext"
    monkeypatch.setattr("qt_shell.QFileDialog.getSaveFileName", staticmethod(lambda *a, **k: (str(bare_path), "")))
    window.save_project()
    assert (tmp_path / "noext.dataapp").exists()


def test_open_project_replaces_existing_library(qtbot, tmp_path, monkeypatch):
    window = DataappMainWindow()
    qtbot.addWidget(window)
    qtbot.wait(20)
    keeper = _spectrum("in_project")
    window.library.add(keeper)

    project_path = tmp_path / "p.dataapp"
    monkeypatch.setattr("qt_shell.QFileDialog.getSaveFileName", staticmethod(lambda *a, **k: (str(project_path), "")))
    window.save_project()

    # Add a second spectrum, then open the project — the pre-open library
    # contents must be replaced (question dialog auto-answers Yes via
    # conftest's autouse fixture).
    window.library.add(_spectrum("volatile"))
    assert len(window.library) == 2
    monkeypatch.setattr("qt_shell.QFileDialog.getOpenFileName", staticmethod(lambda *a, **k: (str(project_path), "")))
    window.open_project()

    titles = [s.title for s in window.library.all()]
    assert titles == ["in_project"]
