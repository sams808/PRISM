"""Tests for the Qt shell (M5) — window construction, real-file import via
io_universal, and selection-triggered plotting. Uses pytest-qt's qtbot to
manage the QApplication/event loop.
"""
from __future__ import annotations

import pytest

from qt_shell import NAV_ITEMS, DataappMainWindow, _load_spectrum_from_path
from qt_widgets import PlotWidget


def test_plot_widget_constructs_and_clears(qtbot):
    widget = PlotWidget()
    qtbot.addWidget(widget)
    widget.clear("Empty")
    qtbot.wait(20)  # let matplotlib's deferred draw_idle() complete before teardown
    assert widget.ax.get_title() == "Empty"


def test_load_spectrum_from_path_dta_example(dta_example_path):
    spectrum = _load_spectrum_from_path(str(dta_example_path))
    assert spectrum.kind == "ta_sdt"
    assert len(spectrum.x) > 100
    assert spectrum.title == "DTA_example"


def test_main_window_constructs_with_nav_and_library_page(qtbot):
    window = DataappMainWindow()
    qtbot.addWidget(window)
    qtbot.wait(20)
    # Tied to NAV_ITEMS rather than a hardcoded count — this assertion broke
    # (correctly, but noisily) at M8, M9, and M12 as each milestone appended
    # a new nav entry. What it actually guards is nav/stack consistency:
    assert window.nav.count() == len(NAV_ITEMS)
    assert window.stack.count() == len(NAV_ITEMS)
    assert window.nav.currentRow() == 0
    assert window.stack.currentWidget() is window.library_page


def test_import_and_select_populates_plot(qtbot, dta_example_path):
    window = DataappMainWindow()
    qtbot.addWidget(window)

    spectrum = _load_spectrum_from_path(str(dta_example_path))
    window.library.add(spectrum)
    window.library_page._refresh_table()

    assert window.library_page.table.rowCount() == 1
    window.library_page.table.selectRow(0)
    qtbot.wait(50)

    assert len(window.library_page.plot.ax.lines) == 1


def test_two_imports_with_same_stem_do_not_collide(qtbot, raman_example_path, tmp_path):
    """Regression guard for the identity bug this data model fixes: import
    the same file twice (simulating two files that would produce the same
    display title) and confirm both survive as distinct library entries."""
    window = DataappMainWindow()
    qtbot.addWidget(window)

    copy_path = tmp_path / raman_example_path.name
    copy_path.write_bytes(raman_example_path.read_bytes())

    s1 = _load_spectrum_from_path(str(raman_example_path))
    s2 = _load_spectrum_from_path(str(copy_path))
    assert s1.title == s2.title  # same stem -> same display title
    assert s1.id != s2.id  # but distinct identity

    window.library.add(s1)
    window.library.add(s2)
    window.library_page._refresh_table()
    qtbot.wait(20)
    assert window.library_page.table.rowCount() == 2
