"""
qt_raman_dialogs.py — small dialogs of the Raman block: the peak table
(positions, heights, FWHM, areas of the selected spectra, copy as text),
the "add to reference library" form and the reference-library manager.
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton,
    QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget,
)

from core.qt_widgets import to_float as _to_float

PEAK_COLUMNS = ["Spectrum", "Position", "Height", "FWHM", "Area", "Prominence"]


class PeakTableDialog(QDialog):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Peak table")
        self.resize(640, 420)
        self.rows: list = []
        lay = QVBoxLayout(self)
        self.note = QLabel("")
        self.note.setObjectName("SectionNote")
        self.note.setWordWrap(True)
        lay.addWidget(self.note)
        self.table = QTableWidget(0, len(PEAK_COLUMNS))
        self.table.setHorizontalHeaderLabels(PEAK_COLUMNS)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        lay.addWidget(self.table, 1)
        btns = QHBoxLayout()
        copy_btn = QPushButton("Copy as text")
        copy_btn.clicked.connect(self.copy_text)
        btns.addWidget(copy_btn)
        save_btn = QPushButton("Save CSV…")
        save_btn.clicked.connect(self.save_csv)
        btns.addWidget(save_btn)
        btns.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        btns.addWidget(close_btn)
        lay.addLayout(btns)

    def set_spectra(self, spectra, threshold_frac: float) -> None:
        from raman.raman_tools import peak_table
        self.rows = []
        for s in spectra:
            try:
                for r in peak_table(s.x, s.y, threshold_frac=threshold_frac):
                    self.rows.append((s.title, r))
            except Exception as exc:
                self.note.setText(f"{s.title}: {exc}")
        self.table.setRowCount(len(self.rows))
        for i, (title, r) in enumerate(self.rows):
            vals = [title, f"{r.position:.2f}", f"{r.height:.5g}", f"{r.fwhm:.2f}", f"{r.area:.5g}", f"{r.prominence:.5g}"]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if c:
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(i, c, it)
        self.table.resizeColumnsToContents()
        self.note.setText(f"{len(self.rows)} peak(s) above {threshold_frac * 100:g} % of the range "
                          f"in {len(spectra)} spectrum/spectra. FWHM is the full width at half prominence; "
                          "area is above the local base between the half-width points.")

    def _text(self, sep: str = "\t") -> str:
        lines = [sep.join(PEAK_COLUMNS)]
        for title, r in self.rows:
            lines.append(sep.join([title, f"{r.position:.3f}", f"{r.height:.6g}", f"{r.fwhm:.3f}", f"{r.area:.6g}", f"{r.prominence:.6g}"]))
        return "\n".join(lines)

    def copy_text(self) -> None:
        from PySide6.QtWidgets import QApplication
        QApplication.clipboard().setText(self._text())

    def save_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save peak table", "peaks.csv", "CSV (*.csv)")
        if not path:
            return
        with open(path, "w", encoding="utf-8-sig") as f:
            f.write(self._text(","))


class AddToLibraryDialog(QDialog):
    """Name, laser wavelength and notes for spectra being added to the
    user's Raman reference library."""

    def __init__(self, parent: Optional[QWidget], titles: List[str]):
        super().__init__(parent)
        self.setWindowTitle("Add to the Raman reference library")
        self.titles = titles
        form = QFormLayout(self)
        self.name_edit = QLineEdit(titles[0] if len(titles) == 1 else "")
        self.name_edit.setPlaceholderText("leave empty to keep each spectrum's own title" if len(titles) > 1 else "")
        form.addRow("Name", self.name_edit)
        self.laser_edit = QLineEdit("")
        self.laser_edit.setPlaceholderText("e.g. 532")
        form.addRow("Laser λ (nm)", self.laser_edit)
        self.source_edit = QLineEdit("")
        self.source_edit.setPlaceholderText("instrument, sample id, person…")
        form.addRow("Source", self.source_edit)
        self.notes_edit = QTextEdit()
        self.notes_edit.setMaximumHeight(80)
        form.addRow("Notes", self.notes_edit)
        form.addRow(QLabel(f"{len(titles)} spectrum/spectra will be added as measured (no processing)."))
        btns = QHBoxLayout()
        ok = QPushButton("Add")
        ok.setObjectName("Primary")
        ok.clicked.connect(self.accept)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        btns.addStretch(1)
        btns.addWidget(cancel)
        btns.addWidget(ok)
        form.addRow(btns)

    def values(self) -> dict:
        return {"name": self.name_edit.text().strip(), "laser_nm": _to_float(self.laser_edit.text()),
                "source": self.source_edit.text().strip(), "notes": self.notes_edit.toPlainText().strip()}


class LibraryManagerDialog(QDialog):
    """List, rename, remove, export and import the reference library."""

    def __init__(self, parent: Optional[QWidget], library_path: Optional[str] = None, on_changed=None):
        super().__init__(parent)
        import raman.raman_library as rl
        self.rl = rl
        self.path = library_path or rl.LIBRARY_PATH
        self.on_changed = on_changed
        self.setWindowTitle("Raman reference library")
        self.resize(640, 420)
        lay = QVBoxLayout(self)
        self.note = QLabel(f"File: {self.path}")
        self.note.setObjectName("SectionNote")
        self.note.setWordWrap(True)
        lay.addWidget(self.note)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Name", "λ (nm)", "Points", "Range", "Added"])
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        lay.addWidget(self.table, 1)
        btns = QHBoxLayout()
        rename_btn = QPushButton("Rename…")
        rename_btn.clicked.connect(self.rename_selected)
        btns.addWidget(rename_btn)
        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(self.remove_selected)
        btns.addWidget(remove_btn)
        export_btn = QPushButton("Export library…")
        export_btn.clicked.connect(self.export_library)
        btns.addWidget(export_btn)
        import_btn = QPushButton("Import library…")
        import_btn.clicked.connect(self.import_library)
        btns.addWidget(import_btn)
        btns.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        btns.addWidget(close_btn)
        lay.addLayout(btns)
        self.entries = []
        self.reload()

    def reload(self) -> None:
        try:
            self.entries = self.rl.load_library(self.path)
        except Exception as exc:
            QMessageBox.critical(self, "Reference library", str(exc))
            self.entries = []
        self.table.setRowCount(len(self.entries))
        for i, e in enumerate(self.entries):
            x = np.asarray(e.x, float)
            vals = [e.name, f"{e.meta.get('laser_nm') or ''}", str(len(e.x)),
                    f"{x.min():.0f}–{x.max():.0f}" if x.size else "", str(e.meta.get("added_at", ""))]
            for c, v in enumerate(vals):
                self.table.setItem(i, c, QTableWidgetItem(v))
        self.table.resizeColumnsToContents()
        self.note.setText(f"File: {self.path} — {len(self.entries)} reference spectra")

    def _selected(self) -> List[int]:
        return sorted({i.row() for i in self.table.selectionModel().selectedRows()})

    def _save(self) -> None:
        self.rl.save_library(self.entries, self.path)
        self.reload()
        if self.on_changed is not None:
            self.on_changed()

    def rename_selected(self) -> None:
        from PySide6.QtWidgets import QInputDialog
        rows = self._selected()
        if len(rows) != 1:
            return
        e = self.entries[rows[0]]
        name, ok = QInputDialog.getText(self, "Rename", "New name:", text=e.name)
        if ok and name.strip():
            e.name = name.strip()
            self._save()

    def remove_selected(self) -> None:
        rows = self._selected()
        if not rows:
            return
        if QMessageBox.question(self, "Remove", f"Remove {len(rows)} reference spectrum/spectra from the library?") != QMessageBox.Yes:
            return
        self.entries = [e for i, e in enumerate(self.entries) if i not in rows]
        self._save()

    def export_library(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export the reference library", "raman_library.json", "PRISM Raman library (*.json)")
        if not path:
            return
        self.rl.save_library(self.entries, path)
        QMessageBox.information(self, "Reference library", f"Exported {len(self.entries)} spectra to {os.path.basename(path)}.")

    def import_library(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import a reference library", "", "PRISM Raman library (*.json)")
        if not path:
            return
        try:
            incoming = self.rl.load_library(path)
        except Exception as exc:
            QMessageBox.critical(self, "Reference library", str(exc))
            return
        self.entries, n_new = self.rl.merge_entries(self.entries, incoming)
        self._save()
        QMessageBox.information(self, "Reference library", f"{n_new} new spectra added ({len(self.entries)} total).")
