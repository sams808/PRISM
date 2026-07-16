"""Tests for the Library management tools (rename/duplicate/reorder/
delete-with-undo) and the Combine/scale dialog — Tk-parity features the
first Qt pass dropped, restored on user request."""
from __future__ import annotations

import numpy as np
import pytest

from qt_models import Spectrum
from qt_shell import CombineDialog, DataappMainWindow


def _spectrum(title, value=1.0, n=50):
    x = np.linspace(0, 100, n)
    return Spectrum(id=Spectrum.new_id(), title=title, path="", kind="raman_xy",
                    x=x, y=np.full(n, float(value)))


def _window_with(qtbot, titles_values):
    window = DataappMainWindow()
    qtbot.addWidget(window)
    for title, value in titles_values:
        window.library.add(_spectrum(title, value))
    window.library_page._refresh_table()
    qtbot.wait(20)
    return window


def test_rename_updates_library(qtbot, monkeypatch):
    window = _window_with(qtbot, [("old_name", 1.0)])
    page = window.library_page
    page.table.selectRow(0)
    monkeypatch.setattr("PySide6.QtWidgets.QInputDialog.getText",
                        staticmethod(lambda *a, **k: ("new_name", True)))
    page._rename_selected()
    assert window.library.all()[0].title == "new_name"
    assert page.table.item(0, 0).text() == "new_name"


def test_duplicate_creates_derived_copy(qtbot):
    window = _window_with(qtbot, [("orig", 5.0)])
    page = window.library_page
    page.table.selectRow(0)
    page._duplicate_selected()
    items = window.library.all()
    assert len(items) == 2
    assert items[1].title == "orig_copy"
    assert items[1].status == "derived"
    assert items[1].id != items[0].id
    items[1].y[0] = 999.0
    assert items[0].y[0] == 5.0  # deep copy, not a view


def test_move_up_down_reorders(qtbot):
    window = _window_with(qtbot, [("a", 1), ("b", 2), ("c", 3)])
    page = window.library_page
    page.table.selectRow(2)
    page._move_selected(-1)
    assert [s.title for s in window.library.all()] == ["a", "c", "b"]
    page._move_selected(+1)
    assert [s.title for s in window.library.all()] == ["a", "b", "c"]


def test_delete_and_undo_restores_order(qtbot):
    window = _window_with(qtbot, [("a", 1), ("b", 2), ("c", 3)])
    page = window.library_page
    page.table.selectRow(1)
    page._delete_selected()
    assert [s.title for s in window.library.all()] == ["a", "c"]
    assert page.undo_btn.isEnabled()

    page._undo_delete()
    assert [s.title for s in window.library.all()] == ["a", "b", "c"]
    assert not page.undo_btn.isEnabled()


def test_combine_dialog_sum(qtbot):
    a = _spectrum("a", 2.0)
    b = _spectrum("b", 3.0)
    dlg = CombineDialog(None, [a, b])
    qtbot.addWidget(dlg)
    dlg.op_combo.setCurrentText("Sum")
    dlg._on_create()
    result = dlg.result_spectrum
    assert result is not None
    assert np.allclose(result.y, 5.0)
    assert result.status == "derived"
    assert result.meta["sources"] == ["a", "b"]


def test_combine_dialog_subtract_with_weights(qtbot):
    a = _spectrum("a", 10.0)
    b = _spectrum("b", 4.0)
    dlg = CombineDialog(None, [a, b])
    qtbot.addWidget(dlg)
    dlg.op_combo.setCurrentText("Subtract (1st − rest)")
    dlg.weights_edit.setText("1, 0.5")
    dlg._on_create()
    assert np.allclose(dlg.result_spectrum.y, 8.0)


def test_combine_dialog_scale_single(qtbot):
    a = _spectrum("a", 3.0)
    dlg = CombineDialog(None, [a])
    qtbot.addWidget(dlg)
    dlg.factor_edit.setText("2")
    dlg.offset_edit.setText("1")
    dlg._on_create()
    assert np.allclose(dlg.result_spectrum.y, 7.0)
    assert dlg.result_spectrum.title == "a_scaled"
