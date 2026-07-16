"""
qt_rruff.py — RRUFF match-assist UI (M12), built on M10's local RRUFF cache
and M7's CIF-Bragg-overlay rendering pattern (reused here for candidate
peak markers).

This is an ASSIST tool, never an auto-labeler — the user's own explicit
original requirement, since RRUFF spectra span many laser excitation
wavelengths (affecting relative peak intensities/fluorescence even though
Raman shift in cm^-1 is nominally wavelength-independent): every candidate
row shows its own wavelength, and nothing is written back to a spectrum's
metadata until the user clicks "Accept as identification" for a specific,
selected candidate.

The RRUFF index (~28k spectra, ~19MB JSON) is loaded lazily on first use,
not at workspace construction — confirmed to take ~0.25s for the real
corpus, which is fine for an on-demand click but not worth paying at every
app startup regardless of whether this workspace is ever opened.
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import numpy as np

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPushButton, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QWidget,
)

from fitting_science import find_peak_candidates
from qt_models import SpectrumLibrary
from qt_widgets import PlotWidget
from rruff_science import (
    RRUFF_ATTRIBUTION_NOTE,
    RRUFF_CACHE_DIR,
    RRUFF_CITATION,
    find_cifs_for_mineral,
    index_summary,
    load_index,
    parse_rruff_txt,
    rank_rruff_matches,
)

RESULT_COLUMNS = ["Mineral", "RRUFF ID", "λ (nm)", "Matched", "Match %", "Category"]


def _to_float(text: str, default: Optional[float] = None) -> Optional[float]:
    try:
        return float((text or "").strip())
    except (TypeError, ValueError):
        return default


class RruffMatchWorkspace(QWidget):
    def __init__(
        self, parent: Optional[QWidget] = None, library: Optional[SpectrumLibrary] = None,
        cache_dir: Optional[str] = None, amcsd_cache_dir: Optional[str] = None,
        on_send_cifs=None, on_accept=None,
    ):
        super().__init__(parent)
        self.library = library if library is not None else SpectrumLibrary()
        # Shell-provided callback (spectrum_id, previous_match_or_None) fired
        # when an identification is accepted, so Accept joins the undo stack.
        self.on_accept = on_accept
        self.cache_dir = cache_dir or RRUFF_CACHE_DIR
        self.amcsd_cache_dir = amcsd_cache_dir  # None -> rruff_science default
        # Shell-provided callback taking a list of CIF paths (the RRUFF→CIF
        # overlay handoff into the Raman workspace).
        self.on_send_cifs = on_send_cifs
        self._index: Optional[List[Dict[str, Any]]] = None
        self._query_peaks: List[float] = []
        self._results: List[Dict[str, Any]] = []
        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        splitter = QSplitter()
        root.addWidget(splitter)

        left = QWidget()
        left.setObjectName("Card")
        left.setMaximumWidth(340)
        left_layout = QVBoxLayout(left)

        left_layout.addWidget(QLabel("Query spectrum"))
        self.spec_combo = QComboBox()
        left_layout.addWidget(self.spec_combo)

        auto_btn = QPushButton("Auto-find peaks from spectrum")
        auto_btn.clicked.connect(self.auto_find_peaks)
        left_layout.addWidget(auto_btn)

        left_layout.addWidget(QLabel("Peak positions (cm⁻¹, comma-separated — editable)"))
        self.peaks_edit = QLineEdit()
        left_layout.addWidget(self.peaks_edit)

        tol_row = QHBoxLayout()
        tol_row.addWidget(QLabel("Tolerance (cm⁻¹)"))
        self.tolerance_edit = QLineEdit("10.0")
        tol_row.addWidget(self.tolerance_edit)
        left_layout.addLayout(tol_row)

        find_btn = QPushButton("Find matches")
        find_btn.setObjectName("Primary")
        find_btn.clicked.connect(self.find_matches)
        left_layout.addWidget(find_btn)

        self.db_status_label = QLabel("RRUFF database not loaded yet.")
        self.db_status_label.setWordWrap(True)
        left_layout.addWidget(self.db_status_label)

        self.overlay_raw_check = QCheckBox("Overlay candidate's measured spectrum (not just peaks)")
        self.overlay_raw_check.setChecked(True)
        left_layout.addWidget(self.overlay_raw_check)

        accept_btn = QPushButton("Accept selected candidate as identification")
        accept_btn.clicked.connect(self.accept_selected_candidate)
        left_layout.addWidget(accept_btn)

        send_cif_btn = QPushButton("Overlay candidate's XRD (CIF) in Raman workspace")
        send_cif_btn.clicked.connect(self.send_candidate_cifs)
        left_layout.addWidget(send_cif_btn)

        citation_label = QLabel(f"{RRUFF_CITATION}\n\n{RRUFF_ATTRIBUTION_NOTE}")
        citation_label.setWordWrap(True)
        citation_label.setObjectName("SectionNote")
        left_layout.addWidget(citation_label)
        left_layout.addStretch(1)
        splitter.addWidget(left)

        right = QWidget()
        right.setObjectName("Card")
        right_layout = QVBoxLayout(right)

        self.results_table = QTableWidget(0, len(RESULT_COLUMNS))
        self.results_table.setHorizontalHeaderLabels(RESULT_COLUMNS)
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.results_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.results_table.itemSelectionChanged.connect(self._on_result_selected)
        right_layout.addWidget(self.results_table, 1)

        self.plot = PlotWidget(figsize=(7, 4.5))
        right_layout.addWidget(self.plot, 1)

        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)

    # ------------------------------------------------------------------
    def set_spectra(self, spectrum_ids: List[str]) -> None:
        current = self.spec_combo.currentData()
        self.spec_combo.blockSignals(True)
        self.spec_combo.clear()
        for sid in spectrum_ids:
            spectrum = self.library.get(sid)
            if spectrum is not None:
                self.spec_combo.addItem(spectrum.title, sid)
        if current is not None:
            idx = self.spec_combo.findData(current)
            if idx >= 0:
                self.spec_combo.setCurrentIndex(idx)
        self.spec_combo.blockSignals(False)

    def _current_spectrum(self):
        sid = self.spec_combo.currentData()
        return self.library.get(sid) if sid else None

    # ------------------------------------------------------------------
    def _ensure_index_loaded(self) -> bool:
        if self._index is not None:
            return True
        t0 = time.time()
        self._index = load_index(cache_dir=self.cache_dir)
        elapsed = time.time() - t0
        if not self._index:
            self.db_status_label.setText(
                f"No RRUFF database found at {self.cache_dir}. Run rruff_science.build_index() first."
            )
            return False
        summary = index_summary(self._index)
        self.db_status_label.setText(
            f"RRUFF database: {summary['n_spectra']} spectra, {summary['n_minerals']} minerals, "
            f"{len(summary['wavelengths_nm'])} wavelengths (loaded in {elapsed:.2f}s)."
        )
        return True

    # ------------------------------------------------------------------
    def auto_find_peaks(self) -> None:
        spectrum = self._current_spectrum()
        if spectrum is None:
            QMessageBox.warning(self, "Auto-find peaks", "Select a query spectrum first.")
            return
        peaks = find_peak_candidates(spectrum.x, spectrum.y, max_peaks=15)
        if not peaks:
            QMessageBox.information(self, "Auto-find peaks", "No clear peak candidates were found.")
            return
        self.peaks_edit.setText(", ".join(f"{p:.1f}" for p in sorted(peaks)))

    def _parse_peaks(self) -> List[float]:
        text = self.peaks_edit.text().strip()
        if not text:
            return []
        out = []
        for part in text.split(","):
            v = _to_float(part)
            if v is not None:
                out.append(v)
        return out

    def find_matches(self) -> None:
        if not self._ensure_index_loaded():
            return
        peaks = self._parse_peaks()
        if not peaks:
            QMessageBox.warning(self, "Find matches", "Enter or auto-find at least one peak position.")
            return
        tolerance = _to_float(self.tolerance_edit.text(), 10.0)
        self._query_peaks = peaks
        self._results = rank_rruff_matches(peaks, self._index, tolerance=tolerance, top_n=25)
        self._populate_results_table()
        self._render_preview(candidate=self._results[0] if self._results else None)

    def _populate_results_table(self) -> None:
        self.results_table.setRowCount(len(self._results))
        for row, rec in enumerate(self._results):
            values = [
                rec.get("mineral", ""), rec.get("rruff_id", ""),
                f"{rec.get('wavelength_nm', ''):.0f}" if rec.get("wavelength_nm") else "",
                str(rec.get("matched_peaks", "")),
                f"{rec.get('match_fraction', 0.0) * 100:.0f}%",
                rec.get("category", ""),
            ]
            for col, val in enumerate(values):
                self.results_table.setItem(row, col, QTableWidgetItem(str(val)))
        self.results_table.resizeColumnsToContents()
        if self._results:
            self.results_table.selectRow(0)

    def _on_result_selected(self) -> None:
        rows = self.results_table.selectionModel().selectedRows()
        if not rows or not self._results:
            return
        self._render_preview(candidate=self._results[rows[0].row()])

    # ------------------------------------------------------------------
    def _render_preview(self, candidate: Optional[Dict[str, Any]]) -> None:
        fig = self.plot.figure
        fig.clf()
        ax = fig.add_subplot(111)

        spectrum = self._current_spectrum()
        if spectrum is not None:
            y = spectrum.y
            y_norm = y / np.nanmax(np.abs(y)) if np.nanmax(np.abs(y)) > 0 else y
            ax.plot(spectrum.x, y_norm, color="black", lw=1.1, label=f"query: {spectrum.title}")

        if candidate is not None:
            label = f"{candidate.get('mineral', '?')} ({candidate.get('rruff_id', '?')}, {candidate.get('wavelength_nm', '?')} nm)"
            if self.overlay_raw_check.isChecked() and candidate.get("raw_path") and os.path.isfile(candidate["raw_path"]):
                try:
                    with open(candidate["raw_path"], "r", encoding="utf-8", errors="replace") as f:
                        text = f.read()
                    ref = parse_rruff_txt(text, source_filename=os.path.basename(candidate["raw_path"]))
                    if len(ref.y):
                        ref_norm = ref.y / np.nanmax(np.abs(ref.y)) if np.nanmax(np.abs(ref.y)) > 0 else ref.y
                        ax.plot(ref.x, ref_norm, color="crimson", lw=1.0, alpha=0.8, label=label)
                except OSError:
                    pass
            for peak in candidate.get("peaks", []):
                ax.axvline(peak, color="crimson", ls="--", lw=0.7, alpha=0.5)

        ax.set_xlabel("Raman shift (cm⁻¹)")
        ax.set_ylabel("Normalized intensity")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        self.plot.canvas.draw_idle()

    # ------------------------------------------------------------------
    def send_candidate_cifs(self) -> None:
        """RRUFF→CIF overlay handoff: look up the selected candidate's
        mineral in the local AMCSD CIF cache and send its structure(s) to
        the Raman workspace's CIF overlay — Raman identification to
        predicted-XRD verification in one click."""
        rows = self.results_table.selectionModel().selectedRows()
        if not rows or not self._results:
            QMessageBox.warning(self, "CIF overlay", "Select a candidate row first.")
            return
        candidate = self._results[rows[0].row()]
        mineral = candidate.get("mineral", "")
        kwargs = {"cache_dir": self.amcsd_cache_dir} if self.amcsd_cache_dir else {}
        cif_paths = find_cifs_for_mineral(mineral, **kwargs)
        if not cif_paths:
            QMessageBox.information(
                self, "CIF overlay",
                f"No AMCSD structure found for '{mineral}' in the local cache.\n"
                "Build it once with rruff_science.ingest_amcsd_cif_zip() "
                "(cif.zip from https://www.rruff.net/AMS/zipped_files/).",
            )
            return
        if self.on_send_cifs is None:
            QMessageBox.information(self, "CIF overlay", "No Raman workspace is wired to receive CIFs here.")
            return
        self.on_send_cifs(cif_paths)
        self.db_status_label.setText(
            f"Sent {len(cif_paths)} AMCSD structure(s) for '{mineral}' to the Raman workspace CIF overlay."
        )

    def accept_selected_candidate(self) -> None:
        rows = self.results_table.selectionModel().selectedRows()
        spectrum = self._current_spectrum()
        if spectrum is None or not rows or not self._results:
            QMessageBox.warning(self, "Accept identification", "Select a query spectrum and a candidate row first.")
            return
        candidate = self._results[rows[0].row()]
        previous_match = spectrum.meta.get("rruff_match")
        spectrum.meta["rruff_match"] = {
            "mineral": candidate.get("mineral"),
            "rruff_id": candidate.get("rruff_id"),
            "wavelength_nm": candidate.get("wavelength_nm"),
            "matched_peaks": candidate.get("matched_peaks"),
            "match_fraction": candidate.get("match_fraction"),
            "accepted_at": time.time(),
        }
        if self.on_accept is not None:
            self.on_accept(spectrum.id, previous_match)
        QMessageBox.information(
            self, "Accepted",
            f"Recorded '{candidate.get('mineral')}' ({candidate.get('rruff_id')}) as the accepted identification "
            f"for '{spectrum.title}'.\n\n{RRUFF_CITATION}",
        )
