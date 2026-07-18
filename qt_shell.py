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
from pathlib import Path
from typing import Optional

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPushButton,
    QStackedWidget, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QWidget,
)

import io_universal
from qt_dta import DtaWorkspace
from qt_models import Spectrum, SpectrumLibrary
from qt_multi_fit import MultiFitWorkspace
from qt_settings_store import PerItemSettingsStore
from qt_simple_plot import SimplePlotWorkspace
from qt_single_fit import SingleFitWorkspace
from qt_baseline import BaselineWorkspace
from qt_calc import CalcWorkspace
from qt_cluster import ClusterWorkspace
from qt_xrd import XrdIdWorkspace
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
NAV_CLUSTER = "Clustering"
NAV_BASELINE = "Baseline"
# NAV_FITTING/NAV_MULTIFIT/NAV_RRUFF/NAV_HTXRD are appended at the end (not
# inserted after Raman) so the DTA page keeps nav row 3 —
# test_qt_dta.py's test_shell_dta_page_picks_up_library_records hardcodes
# setCurrentRow(3), and there's no reason to reorder the rail just to churn
# that index.
NAV_CALC = "Calculations"
NAV_XRD_ID = "XRD ID"
NAV_ITEMS = [NAV_LIBRARY, NAV_RAMAN, NAV_XAS, NAV_DTA, NAV_FITTING, NAV_MULTIFIT, NAV_RRUFF, NAV_HTXRD, NAV_CLUSTER, NAV_BASELINE, NAV_CALC, NAV_XRD_ID]
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


class CombineDialog(QDialog):
    """Sum / average / weighted-subtract multiple spectra, or scale one —
    the generalized successor of the old Tk SpectralSumWindow. The result
    is added to the Library as a derived spectrum."""

    def __init__(self, parent, spectra: list):
        super().__init__(parent)
        self.setWindowTitle("Combine / scale spectra")
        self.spectra = spectra
        self.result_spectrum: Optional[Spectrum] = None

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Inputs: " + ", ".join(s.title for s in spectra)))

        from PySide6.QtWidgets import QComboBox
        op_row = QHBoxLayout()
        op_row.addWidget(QLabel("Operation"))
        self.op_combo = QComboBox()
        ops = ["Scale (single spectrum)"] if len(spectra) == 1 else [
            "Sum", "Average", "Subtract (1st − rest)",
        ]
        self.op_combo.addItems(ops)
        op_row.addWidget(self.op_combo, 1)
        layout.addLayout(op_row)

        self.weights_edit = QLineEdit()
        self.weights_edit.setPlaceholderText("optional weights, comma-separated (e.g. 1, 0.5)")
        self.factor_edit = QLineEdit("1.0")
        self.offset_edit = QLineEdit("0.0")
        if len(spectra) == 1:
            scale_row = QHBoxLayout()
            scale_row.addWidget(QLabel("Factor"))
            scale_row.addWidget(self.factor_edit)
            scale_row.addWidget(QLabel("Offset"))
            scale_row.addWidget(self.offset_edit)
            layout.addLayout(scale_row)
        else:
            layout.addWidget(self.weights_edit)
            self.normalize_check = QCheckBox("Area-normalize each spectrum first (area → 100)")
            layout.addWidget(self.normalize_check)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Result name"))
        self.name_edit = QLineEdit(self._suggest_name())
        name_row.addWidget(self.name_edit, 1)
        layout.addLayout(name_row)

        buttons = QHBoxLayout()
        ok_btn = QPushButton("Create")
        ok_btn.setObjectName("Primary")
        ok_btn.clicked.connect(self._on_create)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        buttons.addStretch(1)
        buttons.addWidget(cancel_btn)
        buttons.addWidget(ok_btn)
        layout.addLayout(buttons)

    def _suggest_name(self) -> str:
        if len(self.spectra) == 1:
            return f"{self.spectra[0].title}_scaled"
        return f"{self.spectra[0].title}_combined{len(self.spectra)}"

    def _on_create(self) -> None:
        import spectrum_math as sm
        try:
            if len(self.spectra) == 1:
                sp = self.spectra[0]
                factor = float(self.factor_edit.text() or "1")
                offset = float(self.offset_edit.text() or "0")
                x, y = sm.scale_spectrum(sp.x, sp.y, factor=factor, offset=offset)
                op_desc = f"scale×{factor:g}+{offset:g}"
            else:
                op_ui = self.op_combo.currentText()
                op = {"Sum": "sum", "Average": "average"}.get(op_ui, "subtract")
                weights = None
                wtext = self.weights_edit.text().strip()
                if wtext:
                    weights = [float(w) for w in wtext.split(",")]
                x, y = sm.combine_spectra(
                    [(s.x, s.y) for s in self.spectra], op=op, weights=weights,
                    normalize_first=self.normalize_check.isChecked(),
                )
                op_desc = op
        except (ValueError, TypeError) as exc:
            QMessageBox.critical(self, "Combine error", str(exc))
            return

        title = self.name_edit.text().strip() or self._suggest_name()
        self.result_spectrum = Spectrum(
            id=Spectrum.new_id(), title=title, path="", kind=self.spectra[0].kind,
            x=x, y=y, df=None,
            meta={"derived": op_desc, "sources": [s.title for s in self.spectra]},
            status="derived",
        )
        self.accept()


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

        custom_import_btn = QPushButton("Custom import…")
        custom_import_btn.setToolTip("Pick the parser and X/Y columns manually — for files the auto-detection guesses wrong on.")
        custom_import_btn.clicked.connect(self._on_custom_import_clicked)
        left_layout.addWidget(custom_import_btn)

        combine_btn = QPushButton("Combine / scale selected…")
        combine_btn.clicked.connect(self._on_combine_clicked)
        left_layout.addWidget(combine_btn)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Title", "Kind"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.ExtendedSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._on_context_menu)
        left_layout.addWidget(self.table, 1)

        self.undo_btn = QPushButton("Undo")
        self.undo_btn.setToolTip("Undo the last library change: delete, rename, duplicate, combine result, applied baseline, or accepted mineral ID.")
        self.undo_btn.setEnabled(False)
        self.undo_btn.clicked.connect(self._undo)
        left_layout.addWidget(self.undo_btn)
        # Typed actions, most recent last:
        #   ("delete", [(position, Spectrum), ...])   undo re-adds at position
        #   ("add", [spectrum_id, ...])               undo removes (derived spectra)
        #   ("rename", spectrum_id, old_title)        undo restores the title
        #   ("ident", spectrum_id, old_match|None)    undo restores meta["rruff_match"]
        self._undo_stack: list = []

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

    # ------------------------------------------------------------------
    # Library management: rename / duplicate / reorder / delete-with-undo
    # (parity with the old Tk app's list management, which the first Qt
    # pass dropped) plus Combine/scale (the old SpectralSumWindow,
    # generalized).
    # ------------------------------------------------------------------
    def _selected_spectra(self) -> list:
        rows = sorted({i.row() for i in self.table.selectionModel().selectedRows()})
        out = []
        for row in rows:
            item = self.table.item(row, 0)
            sp = self.library.get(item.data(Qt.UserRole)) if item else None
            if sp is not None:
                out.append(sp)
        return out

    def _on_context_menu(self, pos) -> None:
        from PySide6.QtWidgets import QMenu
        if self.table.itemAt(pos) is None:
            return
        selected = self._selected_spectra()
        menu = QMenu(self)
        if len(selected) == 1:
            menu.addAction("Rename…", self._rename_selected)
            menu.addAction("Duplicate", self._duplicate_selected)
            menu.addSeparator()
            menu.addAction("Move up", lambda: self._move_selected(-1))
            menu.addAction("Move down", lambda: self._move_selected(+1))
            menu.addSeparator()
        if len(selected) >= 2:
            menu.addAction("Combine / scale…", self._on_combine_clicked)
            menu.addSeparator()
        menu.addAction(f"Export {len(selected)} item(s) as text…", self._export_selected_txt)
        menu.addSeparator()
        menu.addAction(f"Delete {len(selected)} item(s)", self._delete_selected)
        menu.exec(self.table.viewport().mapToGlobal(pos))

    def _export_selected_txt(self) -> None:
        """Write each selected spectrum as a two-column tab-separated .txt —
        the way derived spectra (baseline-subtracted, combined, …) get back
        OUT of the app for use elsewhere (Origin, notebooks, colleagues)."""
        selected = self._selected_spectra()
        if not selected:
            return
        if len(selected) == 1:
            sp = selected[0]
            path, _ = QFileDialog.getSaveFileName(self, "Export spectrum as…", f"{sp.title}.txt", "Text (*.txt);;CSV (*.csv)")
            if not path:
                return
            targets = [(sp, path)]
        else:
            folder = QFileDialog.getExistingDirectory(self, "Export selected spectra into folder…")
            if not folder:
                return
            targets = [(sp, os.path.join(folder, f"{sp.title}.txt")) for sp in selected]

        written, errors = 0, []
        for sp, path in targets:
            try:
                sep = "," if path.lower().endswith(".csv") else "\t"
                data = np.column_stack([np.asarray(sp.x, float), np.asarray(sp.y, float)])
                np.savetxt(path, data, delimiter=sep, header=f"{sp.title} (exported from Dataapp)", comments="# ")
                written += 1
            except OSError as exc:
                errors.append(f"{sp.title}: {exc}")
        msg = f"Exported {written} file(s)."
        if errors:
            msg += "\nFailed: " + "; ".join(errors)
            QMessageBox.warning(self, "Export", msg)

    def clear_all(self) -> None:
        """Clear the whole library (the old app's 'Clear imports') — done
        through the same delete path, so it's undoable."""
        if len(self.library) == 0:
            return
        resp = QMessageBox.question(self, "Clear imports", f"Remove all {len(self.library)} spectra from the library? (Undo is available.)")
        if resp != QMessageBox.Yes:
            return
        batch = [(i, sp) for i, sp in enumerate(self.library.all())]
        for _, sp in batch:
            self.library.remove(sp.id)
        self.push_undo(("delete", batch))
        self._refresh_table()

    def push_undo(self, action: tuple) -> None:
        """Record an undoable library action (see _undo_stack's format).
        Also the entry point for other workspaces' undoable effects on the
        library (applied baselines, accepted mineral IDs), via the shell."""
        self._undo_stack.append(action)
        self.undo_btn.setEnabled(True)

    def _rename_selected(self) -> None:
        selected = self._selected_spectra()
        if len(selected) != 1:
            return
        from PySide6.QtWidgets import QInputDialog
        sp = selected[0]
        new_title, ok = QInputDialog.getText(self, "Rename", "New name:", text=sp.title)
        if ok and new_title.strip() and new_title.strip() != sp.title:
            self.push_undo(("rename", sp.id, sp.title))
            sp.title = new_title.strip()
            self._refresh_table()

    def _duplicate_selected(self) -> None:
        selected = self._selected_spectra()
        if len(selected) != 1:
            return
        sp = selected[0]
        copy_sp = Spectrum(
            id=Spectrum.new_id(), title=f"{sp.title}_copy", path=sp.path, kind=sp.kind,
            x=np.array(sp.x, float).copy(), y=np.array(sp.y, float).copy(),
            df=sp.df, meta=dict(sp.meta), status="derived",
        )
        self.library.add(copy_sp)
        self.push_undo(("add", [copy_sp.id]))
        self._refresh_table()

    def _move_selected(self, delta: int) -> None:
        selected = self._selected_spectra()
        if len(selected) != 1:
            return
        order = [s.id for s in self.library.all()]
        i = order.index(selected[0].id)
        j = i + delta
        if not (0 <= j < len(order)):
            return
        order[i], order[j] = order[j], order[i]
        self.library.reorder(order)
        self._refresh_table()
        self.table.selectRow(j)

    def _delete_selected(self) -> None:
        selected = self._selected_spectra()
        if not selected:
            return
        order = [s.id for s in self.library.all()]
        batch = [(order.index(sp.id), sp) for sp in selected]
        for _, sp in batch:
            self.library.remove(sp.id)
        self.push_undo(("delete", batch))
        self._refresh_table()

    def _undo(self) -> None:
        if not self._undo_stack:
            return
        action = self._undo_stack.pop()
        kind = action[0]
        if kind == "delete":
            batch = action[1]
            for position, sp in sorted(batch, key=lambda t: t[0]):
                self.library.add(sp)
            # Restore the original ordering as closely as possible.
            order = [s.id for s in self.library.all()]
            for position, sp in sorted(batch, key=lambda t: t[0]):
                order.remove(sp.id)
                order.insert(min(position, len(order)), sp.id)
            self.library.reorder(order)
        elif kind == "add":
            for sid in action[1]:
                if self.library.get(sid) is not None:
                    self.library.remove(sid)
        elif kind == "rename":
            sp = self.library.get(action[1])
            if sp is not None:
                sp.title = action[2]
        elif kind == "ident":
            sp = self.library.get(action[1])
            if sp is not None:
                old = action[2]
                if isinstance(old, dict) and "rruff_match" in old and "rruff_matches" in old:
                    # multi-phase envelope (iterative accept): restore both keys
                    for key in ("rruff_match", "rruff_matches"):
                        if old.get(key) is None:
                            sp.meta.pop(key, None)
                        else:
                            sp.meta[key] = old[key]
                elif old is None:
                    sp.meta.pop("rruff_match", None)
                    sp.meta.pop("rruff_matches", None)
                else:  # legacy single-match record
                    sp.meta["rruff_match"] = old
        elif kind == "xrd_ident":
            sp = self.library.get(action[1])
            if sp is not None:
                old = action[2] or {}
                for key in ("xrd_match", "xrd_matches"):
                    if old.get(key) is None:
                        sp.meta.pop(key, None)
                    else:
                        sp.meta[key] = old[key]
        self.undo_btn.setEnabled(bool(self._undo_stack))
        self._refresh_table()

    # Kept as an alias: the File-menu wiring and older tests used this name
    # when deletion was the only undoable action.
    _undo_delete = _undo

    def _on_custom_import_clicked(self) -> None:
        from qt_custom_import import CustomImportDialog
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Custom import", "", "All files (*.*)",
        )
        added = 0
        for path in paths:
            dlg = CustomImportDialog(self, path)
            if dlg.exec() and dlg.spectrum is not None:
                self.library.add(dlg.spectrum)
                added += 1
        if added:
            self._refresh_table()

    def _on_combine_clicked(self) -> None:
        selected = self._selected_spectra()
        if not selected:
            QMessageBox.information(self, "Combine", "Select one spectrum (to scale) or several (to sum/average/subtract).")
            return
        dlg = CombineDialog(self, selected)
        if dlg.exec():
            result = dlg.result_spectrum
            if result is not None:
                self.library.add(result)
                self.push_undo(("add", [result.id]))
                self._refresh_table()
                self.table.selectRow(self.table.rowCount() - 1)

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
        from qt_help import APP_VERSION
        self.setWindowTitle(f"Dataapp {APP_VERSION}")
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
        self.rruff_page = RruffMatchWorkspace(
            library=self.library, on_send_cifs=self._on_rruff_send_cifs,
            on_accept=lambda sid, old: self.library_page.push_undo(("ident", sid, old)),
        )
        self.stack.addWidget(self.rruff_page)
        self.htxrd_page = HtxrdWorkspace()
        self.stack.addWidget(self.htxrd_page)
        self.cluster_page = ClusterWorkspace(library=self.library)
        self.stack.addWidget(self.cluster_page)
        self.baseline_page = BaselineWorkspace(
            library=self.library,
            on_derived_added=lambda ids: self.library_page.push_undo(("add", list(ids))),
        )
        self.stack.addWidget(self.baseline_page)
        self.calc_page = CalcWorkspace(
            library=self.library,
            on_derived_added=lambda ids: self.library_page.push_undo(("add", list(ids))),
        )
        self.stack.addWidget(self.calc_page)
        self.xrd_id_page = XrdIdWorkspace(
            library=self.library,
            on_accept=lambda sid, old: self.library_page.push_undo(("xrd_ident", sid, old)),
        )
        self.stack.addWidget(self.xrd_id_page)
        outer.addWidget(self.stack, 1)

        self.nav.setCurrentRow(0)

        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction("Import files…", self.library_page._on_import_clicked, "Ctrl+O")
        file_menu.addAction("Custom import…", self.library_page._on_custom_import_clicked, "Ctrl+I")
        file_menu.addAction("Export selected as text…", self.library_page._export_selected_txt, "Ctrl+E")
        file_menu.addSeparator()
        file_menu.addAction("Open project…", self.open_project, "Ctrl+Shift+O")
        file_menu.addAction("Save project as…", self.save_project, "Ctrl+S")
        file_menu.addSeparator()
        file_menu.addAction("Clear imports…", self.library_page.clear_all)
        file_menu.addAction("Undo", self.library_page._undo, "Ctrl+Z")
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close, "Ctrl+Q")

        view_menu = self.menuBar().addMenu("&View")
        self.dark_mode_action = view_menu.addAction("Dark mode")
        self.dark_mode_action.setCheckable(True)
        self.dark_mode_action.toggled.connect(self._on_dark_mode_toggled)

        self.console_action = view_menu.addAction("Python console")
        self.console_action.setCheckable(True)
        self.console_action.toggled.connect(self._on_console_toggled)
        self._console_dock = None  # created lazily on first open

        help_menu = self.menuBar().addMenu("&Help")
        help_menu.addAction("Quick-start guide", self.show_help, "F1")
        help_menu.addAction("About Dataapp", self.show_about)

        self.statusBar().showMessage("Ready.")

        # Restore window geometry + last-used workspace from the previous
        # session (QSettings, per-user registry on Windows).
        from PySide6.QtCore import QSettings
        settings = QSettings("Dataapp", "Dataapp")
        geometry = settings.value("geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        nav_row = settings.value("nav_row", 0, type=int)
        if 0 <= nav_row < self.nav.count():
            self.nav.setCurrentRow(nav_row)
            # setCurrentRow doesn't fire currentRowChanged when the row is
            # unchanged, and the per-workspace refresh hook must still run
            # for whatever page we restored into.
            self._on_nav_changed(self.nav.currentRow())

    def closeEvent(self, event) -> None:
        from PySide6.QtCore import QSettings
        settings = QSettings("Dataapp", "Dataapp")
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("nav_row", self.nav.currentRow())
        super().closeEvent(event)

    def show_help(self) -> None:
        from qt_help import HelpDialog
        HelpDialog(self).exec()

    def show_about(self) -> None:
        from qt_help import ABOUT_HTML, HelpDialog
        HelpDialog(self, html=ABOUT_HTML, title="About Dataapp").exec()

    def _on_rruff_send_cifs(self, cif_paths) -> None:
        """RRUFF→CIF handoff target: add the structures to the Raman
        workspace's CIF overlay and switch to it so the result is visible."""
        added = self.raman_page.add_cif_files(list(cif_paths))
        self.nav.setCurrentRow(NAV_ITEMS.index(NAV_RAMAN))
        self.statusBar().showMessage(f"Added {added} CIF(s) to the Raman CIF overlay.")

    def _on_console_toggled(self, visible: bool) -> None:
        if self._console_dock is None:
            import numpy as np
            import pandas as pd
            from qt_console import ConsoleDock
            self._console_dock = ConsoleDock({
                "window": self,
                "library": self.library,
                "xas_store": self.xas_page.store,
                "htxrd_series": self.htxrd_page.series,
                "fit_params": self.fit_param_memory,
                "np": np,
                "pd": pd,
            }, parent=self)
            self.addDockWidget(Qt.BottomDockWidgetArea, self._console_dock)
            # Keep the menu checkbox honest when the user closes the dock
            # via its own title-bar X instead of the menu.
            self._console_dock.visibilityChanged.connect(self.console_action.setChecked)
        self._console_dock.setVisible(visible)

    def _on_dark_mode_toggled(self, enabled: bool) -> None:
        """Dark mode restyles the Qt chrome only — matplotlib plot areas
        stay white so what's on screen always matches PNG/SVG/PDF export."""
        from PySide6.QtWidgets import QApplication
        from qt_theme import apply_theme
        app = QApplication.instance()
        if app is not None:
            apply_theme(app, dark=enabled)

    # ------------------------------------------------------------------
    # Project persistence (M14): everything in the shared Library plus the
    # shared fit-parameter store, in one .dataapp file. The XAS and HT-XRD
    # workspaces keep their own session state (different data models) and
    # are not yet included — the file format is versioned so they can be
    # added later without breaking old projects.
    # ------------------------------------------------------------------
    def save_project(self) -> None:
        import project_io
        path, _ = QFileDialog.getSaveFileName(self, "Save project as…", "", "Dataapp project (*.dataapp)")
        if not path:
            return
        if not path.lower().endswith(".dataapp"):
            path += ".dataapp"
        fit_params = {sid: params for sid, params in self.fit_param_memory.items()}
        try:
            cif_overlays = [
                {k: s.get(k) for k in ("path", "label", "plot_label", "visible", "color", "pad")}
                for s in self.raman_page.cif_series
            ]
            baseline_settings = {sid: dict(val) for sid, val in self.baseline_page.settings.items()}
            project_io.save_project(
                path, self.library.all(), fit_params,
                xas_spectra=self.xas_page.store.all(),
                htxrd_patterns=self.htxrd_page.series,
                cif_overlays=cif_overlays,
                baseline_settings=baseline_settings,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Save project error", str(exc))
            return
        self.statusBar().showMessage(f"Project saved: {path}")

    def open_project(self) -> None:
        import project_io
        path, _ = QFileDialog.getOpenFileName(self, "Open project", "", "Dataapp project (*.dataapp);;All files (*.*)")
        if not path:
            return
        if len(self.library) > 0:
            resp = QMessageBox.question(
                self, "Open project",
                "Opening a project replaces the current library contents. Continue?",
            )
            if resp != QMessageBox.Yes:
                return
        try:
            project = project_io.load_project(path)
        except Exception as exc:
            QMessageBox.critical(self, "Open project error", str(exc))
            return

        self.library.clear()
        self.fit_param_memory.clear()
        for sp in project.spectra:
            self.library.add(sp)
        for sid, params in project.fit_params.items():
            self.fit_param_memory.set(sid, params)

        self.xas_page.store.clear()
        for sp in project.xas_spectra:
            self.xas_page.store.add(sp)
        self.xas_page.selected_sid = None
        self.xas_page._refresh_all()

        if project.htxrd_patterns:
            self.htxrd_page.set_series(project.htxrd_patterns)

        if project.cif_overlays:
            self.raman_page.restore_cif_overlays(project.cif_overlays)
        self.baseline_page.settings.clear()
        for sid, val in project.baseline_settings.items():
            self.baseline_page.settings.set(sid, val)

        self.library_page._refresh_table()
        # Re-sync whichever workspace is currently visible.
        self._on_nav_changed(self.nav.currentRow())
        self.statusBar().showMessage(
            f"Project loaded: {len(project.spectra)} spectra, {len(project.xas_spectra)} XAS objects, "
            f"{len(project.htxrd_patterns)} HT-XRD patterns from {path}"
        )

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
        elif self.stack.widget(row) is self.cluster_page:
            self.cluster_page.set_spectra([s.id for s in self.library.all()])
        elif self.stack.widget(row) is self.baseline_page:
            self.baseline_page.set_spectra([s.id for s in self.library.all()])
        elif self.stack.widget(row) is self.calc_page:
            self.calc_page.set_spectra([s.id for s in self.library.all()])
        elif self.stack.widget(row) is self.xrd_id_page:
            self.xrd_id_page.set_spectra([s.id for s in self.library.all()])
