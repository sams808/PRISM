"""
qt_shell.py — the Qt application shell (M5).

Not a mechanical port of main.py's layout: that app is a flat wall of
buttons regardless of what's loaded. This organizes by technique/workflow
instead — a left rail for Library / Raman / XAS / DTA workspaces, so (once
M6-M11 fill in the technique pages) a DTA user only ever sees DTA-relevant
tools. For now the technique pages are placeholders; Library is fully
functional (import via io_universal, select, plot) so this milestone is a
real, demoable slice rather than inert scaffolding.
"""
from __future__ import annotations

import logging
import os
import traceback
from pathlib import Path
from typing import Optional

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QMainWindow, QMessageBox, QPushButton, QSplitter, QStackedWidget,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

import io_universal
from qt_dta import DtaWorkspace
from qt_models import Spectrum, SpectrumLibrary
from qt_multi_fit import MultiFitWorkspace
from qt_settings_store import PerItemSettingsStore
from qt_simple_plot import SimplePlotWorkspace
from qt_single_fit import SingleFitWorkspace
from qt_htxrd import HtxrdWorkspace
from qt_rruff import RruffMatchWorkspace
from qt_widgets import PlotWidget
from qt_xas import XasWorkspace

logger = logging.getLogger("dataapp")

NAV_LIBRARY = "Library"
NAV_RAMAN = "Raman"
NAV_XAS = "XAS"
NAV_DTA = "DTA / Thermal"
NAV_FITTING = "Peak Fitting"
NAV_MULTIFIT = "Multi-Fit"
NAV_RRUFF = "Mineral ID"
NAV_HTXRD = "HT-XRD"
# NAV_FITTING/NAV_MULTIFIT/NAV_RRUFF/NAV_HTXRD are appended at the end (not
# inserted after Raman) so the DTA page keeps nav row 3 —
# test_qt_dta.py's test_shell_dta_page_picks_up_library_records hardcodes
# setCurrentRow(3), and there's no reason to reorder the rail just to churn
# that index.
NAV_ITEMS = [NAV_LIBRARY, NAV_RAMAN, NAV_XAS, NAV_DTA, NAV_FITTING, NAV_MULTIFIT, NAV_RRUFF, NAV_HTXRD]
DTA_KINDS = {"ta_sdt", "dta_table"}


def _load_spectrum_from_path(path: str) -> Spectrum:
    """Generic import via io_universal's parser registry, picking X/Y from
    each parser's canonical_map (every parser sets canonical_map["X"]/["Y"]
    as a fallback pair even when it can't infer richer canonical keys)."""
    df, meta = io_universal.load_any(path, return_meta=True)
    canon = meta.get("canonical_map", {}) or {}
    x_col = canon.get("X") or df.columns[0]
    y_col = canon.get("Y") or df.columns[1]
    x = df[x_col].astype(float).to_numpy()
    y = df[y_col].astype(float).to_numpy()
    order = np.argsort(x, kind="mergesort")
    return Spectrum(
        id=Spectrum.new_id(),
        title=Path(path).stem,
        path=str(path),
        kind=meta.get("selected_parser", "generic_xy"),
        x=x[order], y=y[order],
        df=df, meta=meta, status="imported",
    )


class LibraryPage(QWidget):
    """Data hub: import, browse, and plot imported spectra."""

    def __init__(self, library: SpectrumLibrary, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.library = library

        root = QHBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        left = QWidget()
        left.setObjectName("Card")
        left.setFixedWidth(320)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(12, 12, 12, 12)

        title = QLabel("Imported files")
        title.setObjectName("SectionTitle")
        left_layout.addWidget(title)

        note = QLabel("Bring data in, then select a row to preview it.")
        note.setObjectName("SectionNote")
        note.setWordWrap(True)
        left_layout.addWidget(note)

        import_btn = QPushButton("Import files…")
        import_btn.setObjectName("Primary")
        import_btn.clicked.connect(self._on_import_clicked)
        left_layout.addWidget(import_btn)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Title", "Kind"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        left_layout.addWidget(self.table, 1)

        root.addWidget(left)

        right = QWidget()
        right.setObjectName("Card")
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 8, 8, 8)
        self.plot = PlotWidget()
        self.plot.clear("Select a spectrum to preview")
        right_layout.addWidget(self.plot)
        root.addWidget(right, 1)

    def _on_import_clicked(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Import files", "",
            "Data files (*.txt *.dat *.csv *.xy *.asc);;All files (*.*)",
        )
        if not paths:
            return
        added = 0
        errors = []
        for path in paths:
            try:
                spectrum = _load_spectrum_from_path(path)
                self.library.add(spectrum)
                added += 1
            except Exception as exc:
                errors.append(f"{os.path.basename(path)}: {exc}")
                logger.warning("Import failed for %s", path, exc_info=True)
        if added:
            self._refresh_table()
        if errors:
            QMessageBox.warning(self, "Import", "Some files could not be imported:\n" + "\n".join(errors))

    def _refresh_table(self) -> None:
        items = self.library.all()
        self.table.setRowCount(len(items))
        for row, spectrum in enumerate(items):
            title_item = QTableWidgetItem(spectrum.title)
            title_item.setData(Qt.UserRole, spectrum.id)
            self.table.setItem(row, 0, title_item)
            self.table.setItem(row, 1, QTableWidgetItem(spectrum.kind))

    def _on_selection_changed(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        item = self.table.item(rows[0].row(), 0)
        spectrum_id = item.data(Qt.UserRole)
        spectrum = self.library.get(spectrum_id)
        if spectrum is None:
            return
        self.plot.ax.clear()
        self.plot.ax.plot(spectrum.x, spectrum.y, color="#3c6e71", lw=1.2)
        self.plot.ax.set_title(spectrum.title)
        self.plot.ax.grid(alpha=0.25)
        self.plot.figure.tight_layout()
        self.plot.canvas.draw_idle()


class PlaceholderPage(QWidget):
    """A technique workspace not yet ported (filled in by M6/M11/M7/M8)."""

    def __init__(self, name: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        label = QLabel(f"{name} workspace — coming in a later milestone.")
        label.setObjectName("SectionNote")
        layout.addWidget(label)


class DataappMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dataapp")
        self.resize(1280, 820)

        self.library = SpectrumLibrary()
        # Shared by Peak Fitting (M8) and Multi-Fit (M9): a batch write-back
        # must be immediately visible in Peak Fitting and vice versa, so
        # there's exactly one PerItemSettingsStore, not one per workspace.
        self.fit_param_memory = PerItemSettingsStore(list)

        central = QWidget()
        self.setCentralWidget(central)
        outer = QHBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        sidebar = QWidget()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(180)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 12, 0, 0)

        self.nav = QListWidget()
        self.nav.setObjectName("NavList")
        for name in NAV_ITEMS:
            QListWidgetItem(name, self.nav)
        self.nav.currentRowChanged.connect(self._on_nav_changed)
        sidebar_layout.addWidget(self.nav)
        outer.addWidget(sidebar)

        self.stack = QStackedWidget()
        self.library_page = LibraryPage(self.library)
        self.stack.addWidget(self.library_page)
        self.raman_page = SimplePlotWorkspace(library=self.library)
        self.stack.addWidget(self.raman_page)
        self.xas_page = XasWorkspace()
        self.stack.addWidget(self.xas_page)
        self.dta_page = DtaWorkspace()
        self.stack.addWidget(self.dta_page)
        self.fitting_page = SingleFitWorkspace(library=self.library, fit_param_memory=self.fit_param_memory)
        self.stack.addWidget(self.fitting_page)
        self.multifit_page = MultiFitWorkspace(library=self.library, fit_param_memory=self.fit_param_memory)
        self.stack.addWidget(self.multifit_page)
        self.rruff_page = RruffMatchWorkspace(library=self.library)
        self.stack.addWidget(self.rruff_page)
        self.htxrd_page = HtxrdWorkspace()
        self.stack.addWidget(self.htxrd_page)
        outer.addWidget(self.stack, 1)

        self.nav.setCurrentRow(0)

        self.statusBar().showMessage("Ready.")

    def _dta_records_from_library(self):
        records = []
        for spectrum in self.library.by_kind(DTA_KINDS):
            records.append({
                "title": spectrum.title,
                "path": spectrum.path,
                "df": spectrum.df,
                "meta": spectrum.meta,
            })
        return records

    def _on_nav_changed(self, row: int) -> None:
        self.stack.setCurrentIndex(row)
        if self.stack.widget(row) is self.dta_page:
            self.dta_page.set_records(self._dta_records_from_library())
        elif self.stack.widget(row) is self.raman_page:
            self.raman_page.set_spectra([s.id for s in self.library.all()])
        elif self.stack.widget(row) is self.fitting_page:
            self.fitting_page.set_spectra([s.id for s in self.library.all()])
        elif self.stack.widget(row) is self.multifit_page:
            self.multifit_page.set_spectra([s.id for s in self.library.all()])
            # A recipe saved via Peak Fitting's "Save as model..." while the
            # user was on a different tab wouldn't otherwise show up here
            # until they clicked the page's own manual Refresh button —
            # found via a manual smoke test that saved a recipe after the
            # shell was already constructed, the realistic order of events,
            # rather than the unit tests' write-file-then-construct-widget
            # order, which structurally couldn't hit this gap.
            self.multifit_page._refresh_recipe_list()
        elif self.stack.widget(row) is self.rruff_page:
            self.rruff_page.set_spectra([s.id for s in self.library.all()])
