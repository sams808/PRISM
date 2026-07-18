"""
qt_xrd.py — the XRD phase-identification workspace: the QualX search-match
rebuilt inside Dataapp (xrd_id_science engine over the unified
COD1906-INO + COD2205 + PDF2 database), plus a reference-card browser and
the Raman↔XRD bridge.

Same philosophy as Mineral ID: candidates are RANKED with their figure of
merit, sources, and codes — identification only happens on an explicit
Accept, which works iteratively for mixtures (matched peaks removed, the
remaining ones re-searched), mirroring the Raman workflow.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QMessageBox, QPushButton, QSplitter, QTabWidget, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

import xrd_id_science as xid
from fitting_science import find_peak_candidates
from qt_models import SpectrumLibrary
from qt_widgets import PlotWidget

RESULT_COLUMNS = ["FoM %", "Mineral / name", "Formula", "Source", "Code", "Q", "Space group", "Matched"]
CARD_COLORS = ["crimson", "royalblue", "seagreen", "darkorange", "purple", "teal"]


def _to_float(text: str, default: Optional[float] = None) -> Optional[float]:
    try:
        return float((text or "").strip())
    except (TypeError, ValueError):
        return default


def _parse_elements(text: str) -> List[str]:
    return [t.strip() for t in (text or "").replace(";", ",").split(",") if t.strip()]


class XrdIdWorkspace(QWidget):
    def __init__(self, parent: Optional[QWidget] = None, library: Optional[SpectrumLibrary] = None,
                 db_path: Optional[str] = None, on_accept=None):
        super().__init__(parent)
        self.library = library if library is not None else SpectrumLibrary()
        self.db_path = db_path or xid.XRD_ID_DB_PATH
        # Shell callback (spectrum_id, previous_state) — accepted phases join
        # the Library undo stack, same as Mineral ID.
        self.on_accept = on_accept
        self._results: List[xid.XrdMatch] = []
        self._query_peaks: List[float] = []
        self._query_int: List[float] = []
        self._build_ui()
        self._refresh_db_status()

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

        left_layout.addWidget(QLabel("Query pattern"))
        self.spec_combo = QComboBox()
        left_layout.addWidget(self.spec_combo)

        auto_row = QHBoxLayout()
        auto_btn = QPushButton("Auto-find peaks")
        auto_btn.clicked.connect(self.auto_find_peaks)
        auto_row.addWidget(auto_btn)
        auto_row.addWidget(QLabel("limit ×σ"))
        self.detection_edit = QLineEdit("3.0")
        self.detection_edit.setMaximumWidth(45)
        auto_row.addWidget(self.detection_edit)
        left_layout.addLayout(auto_row)

        left_layout.addWidget(QLabel("Peaks 2θ (comma-sep — editable)"))
        self.peaks_edit = QLineEdit()
        left_layout.addWidget(self.peaks_edit)

        row = QHBoxLayout()
        row.addWidget(QLabel("λ (Å)"))
        self.wavelength_edit = QLineEdit(f"{xid.CU_KA1}")
        self.wavelength_edit.setMaximumWidth(60)
        row.addWidget(self.wavelength_edit)
        row.addWidget(QLabel("tol (°2θ)"))
        self.tol_edit = QLineEdit("0.2")
        self.tol_edit.setMaximumWidth(45)
        row.addWidget(self.tol_edit)
        left_layout.addLayout(row)

        left_layout.addWidget(QLabel("Chemistry filter"))
        el_row = QHBoxLayout()
        el_row.addWidget(QLabel("contains all:"))
        self.elements_all_edit = QLineEdit()
        self.elements_all_edit.setPlaceholderText("e.g. Si, O")
        el_row.addWidget(self.elements_all_edit)
        left_layout.addLayout(el_row)
        el_row2 = QHBoxLayout()
        el_row2.addWidget(QLabel("excludes:"))
        self.elements_none_edit = QLineEdit()
        self.elements_none_edit.setPlaceholderText("e.g. Pb")
        el_row2.addWidget(self.elements_none_edit)
        left_layout.addLayout(el_row2)

        self.source_checks: Dict[str, QCheckBox] = {}
        self.sources_holder = QHBoxLayout()
        left_layout.addLayout(self.sources_holder)

        self._search_btn = QPushButton("Search match")
        self._search_btn.setObjectName("Primary")
        self._search_btn.clicked.connect(self.run_search)
        left_layout.addWidget(self._search_btn)

        accept_btn = QPushButton("Accept selected phase")
        accept_btn.clicked.connect(self.accept_selected)
        left_layout.addWidget(accept_btn)

        self.raman_link_btn = QPushButton("Check Raman-identified phases here")
        self.raman_link_btn.setToolTip(
            "For each phase accepted in Mineral ID on this spectrum's sample, look the mineral up in "
            "the XRD database and overlay its reference lines on the query pattern."
        )
        self.raman_link_btn.clicked.connect(self.check_raman_phases)
        left_layout.addWidget(self.raman_link_btn)

        self.db_status_label = QLabel("")
        self.db_status_label.setWordWrap(True)
        left_layout.addWidget(self.db_status_label)
        left_layout.addStretch(1)
        splitter.addWidget(left)

        right = QWidget()
        right.setObjectName("Card")
        right_layout = QVBoxLayout(right)
        self.tabs = QTabWidget()

        id_tab = QWidget()
        idl = QVBoxLayout(id_tab)
        self.results_table = QTableWidget(0, len(RESULT_COLUMNS))
        self.results_table.setHorizontalHeaderLabels(RESULT_COLUMNS)
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.results_table.setSelectionMode(QTableWidget.ExtendedSelection)
        self.results_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.results_table.itemSelectionChanged.connect(self._on_result_selected)
        idl.addWidget(self.results_table, 1)
        self.plot = PlotWidget(figsize=(7, 4))
        idl.addWidget(self.plot, 1)
        self.tabs.addTab(id_tab, "Phase ID")

        browse_tab = QWidget()
        bl = QVBoxLayout(browse_tab)
        browse_row = QHBoxLayout()
        browse_row.addWidget(QLabel("Name / mineral / formula"))
        self.browse_edit = QLineEdit()
        self.browse_edit.returnPressed.connect(self.browse_cards)
        browse_row.addWidget(self.browse_edit, 1)
        browse_btn = QPushButton("Look up")
        browse_btn.clicked.connect(self.browse_cards)
        browse_row.addWidget(browse_btn)
        bl.addLayout(browse_row)
        self.browse_list = QListWidget()
        self.browse_list.itemSelectionChanged.connect(lambda: self.browse_plot.request_redraw(self.render_browse))
        bl.addWidget(self.browse_list, 1)
        self.browse_plot = PlotWidget(figsize=(7, 4))
        bl.addWidget(self.browse_plot, 1)
        self.tabs.addTab(browse_tab, "Card browser")

        right_layout.addWidget(self.tabs)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        self._browse_hits: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    def _refresh_db_status(self) -> None:
        summary = xid.database_summary(self.db_path)
        if summary is None:
            self.db_status_label.setText(
                "No unified XRD database yet. Build it once from your QualX .sq sources with "
                "xrd_id_science.build_xrd_database() — see F1 help."
            )
            return
        parts = ", ".join(f"{k}: {v}" for k, v in sorted(summary["by_source"].items()))
        self.db_status_label.setText(f"XRD database: {summary['total_cards']} cards ({parts}).")
        for tag in sorted(summary["by_source"]):
            if tag not in self.source_checks:
                cb = QCheckBox(tag)
                cb.setChecked(True)
                self.source_checks[tag] = cb
                self.sources_holder.addWidget(cb)

    def set_spectra(self, spectrum_ids: List[str]) -> None:
        current = self.spec_combo.currentData()
        self.spec_combo.blockSignals(True)
        self.spec_combo.clear()
        for sid in spectrum_ids:
            sp = self.library.get(sid)
            if sp is not None:
                self.spec_combo.addItem(sp.title, sid)
        if current is not None:
            idx = self.spec_combo.findData(current)
            if idx >= 0:
                self.spec_combo.setCurrentIndex(idx)
        self.spec_combo.blockSignals(False)
        self._refresh_db_status()

    def _current_spectrum(self):
        sid = self.spec_combo.currentData()
        return self.library.get(sid) if sid else None

    # ------------------------------------------------------------------
    def auto_find_peaks(self) -> None:
        sp = self._current_spectrum()
        if sp is None:
            QMessageBox.warning(self, "Auto-find peaks", "Select a query pattern first.")
            return
        limit = _to_float(self.detection_edit.text(), 3.0) or 3.0
        centers = find_peak_candidates(sp.x, sp.y, max_peaks=25, min_prominence_sigma=limit)
        if not centers:
            QMessageBox.information(self, "Auto-find peaks", "No clear peaks — lower the detection limit.")
            return
        self.peaks_edit.setText(", ".join(f"{c:.3f}" for c in sorted(centers)))

    def _parse_peaks(self):
        vals = []
        for tok in (self.peaks_edit.text() or "").split(","):
            v = _to_float(tok)
            if v is not None:
                vals.append(v)
        return vals

    def _peak_intensities(self, peaks: List[float]) -> List[float]:
        sp = self._current_spectrum()
        if sp is None:
            return [1.0] * len(peaks)
        x, y = np.asarray(sp.x, float), np.asarray(sp.y, float)
        out = []
        for p in peaks:
            near = np.abs(x - p) <= max(0.15, float(np.mean(np.diff(x))) * 3)
            out.append(float(np.nanmax(y[near])) if np.any(near) else 1.0)
        floor = min(out) if out else 0.0
        return [max(v - floor, 1.0) for v in out]

    def run_search(self) -> None:
        peaks = self._parse_peaks()
        if not peaks:
            QMessageBox.warning(self, "Search match", "Enter or auto-find at least one 2θ peak.")
            return
        if xid.database_summary(self.db_path) is None:
            QMessageBox.information(self, "Search match", self.db_status_label.text())
            return
        self._query_peaks = peaks
        self._query_int = self._peak_intensities(peaks)
        sp = self._current_spectrum()
        tt_range = (float(np.nanmin(sp.x)), float(np.nanmax(sp.x))) if sp is not None else None
        sources = [tag for tag, cb in self.source_checks.items() if cb.isChecked()]

        self._search_btn.setEnabled(False)
        self._search_btn.setText("Searching…")
        kwargs = dict(
            wavelength=_to_float(self.wavelength_edit.text(), xid.CU_KA1),
            tol_two_theta=_to_float(self.tol_edit.text(), 0.2),
            elements_all=_parse_elements(self.elements_all_edit.text()),
            elements_none=_parse_elements(self.elements_none_edit.text()),
            sources=sources if sources and len(sources) < len(self.source_checks) else (),
            two_theta_range=tt_range, db_path=self.db_path,
        )
        from qt_worker import run_in_thread
        run_in_thread(
            lambda: xid.search_match(list(peaks), list(self._query_int), **kwargs),
            self._on_search_done, self._on_search_error,
        )

    def _on_search_error(self, traceback_text: str) -> None:
        self._search_btn.setEnabled(True)
        self._search_btn.setText("Search match")
        QMessageBox.critical(self, "Search match error", traceback_text)

    def _on_search_done(self, results: List[xid.XrdMatch]) -> None:
        self._search_btn.setEnabled(True)
        self._search_btn.setText("Search match")
        self._results = results
        self.results_table.setRowCount(len(results))
        for row, r in enumerate(results):
            display = r.mineral or r.name or r.formula
            values = [f"{100 * r.fom:.0f}", display, r.formula, r.source, r.source_code,
                      r.quality, r.spacegroup, str(r.n_matched)]
            for col, val in enumerate(values):
                self.results_table.setItem(row, col, QTableWidgetItem(str(val)))
        self.results_table.resizeColumnsToContents()
        if results:
            self.results_table.selectRow(0)
        else:
            self.plot.request_redraw(self.render_preview)

    def _selected_matches(self) -> List[xid.XrdMatch]:
        rows = sorted({i.row() for i in self.results_table.selectionModel().selectedRows()})
        return [self._results[r] for r in rows if 0 <= r < len(self._results)]

    def _on_result_selected(self) -> None:
        self.plot.request_redraw(self.render_preview)

    def render_preview(self) -> None:
        fig = self.plot.figure
        fig.clf()
        ax = fig.add_subplot(111)
        sp = self._current_spectrum()
        ymax = 1.0
        if sp is not None:
            y = np.asarray(sp.y, float)
            ymax = float(np.nanmax(y)) or 1.0
            ax.plot(sp.x, y / ymax * 100.0, color="black", lw=1.0, label=f"query: {sp.title}")
        wavelength = _to_float(self.wavelength_edit.text(), xid.CU_KA1)
        for k, r in enumerate(self._selected_matches()):
            color = CARD_COLORS[k % len(CARD_COLORS)]
            tt = xid.d_to_two_theta(r.d, wavelength)
            ok = np.isfinite(tt)
            label = f"{r.mineral or r.name or r.formula} [{r.source} {r.source_code}] FoM {100 * r.fom:.0f}%"
            ax.vlines(tt[ok], 0, -r.i[ok], color=color, lw=1.2, label=label)
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_xlabel("2θ (deg)")
        ax.set_ylabel("query (up) / card lines (down)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        self.plot.canvas.draw_idle()

    # ------------------------------------------------------------------
    def accept_selected(self) -> None:
        """Iterative accept, mirroring Mineral ID: record the phase, drop
        the peaks it explains, re-search the remainder."""
        sp = self._current_spectrum()
        matches = self._selected_matches()
        if sp is None or not matches:
            QMessageBox.warning(self, "Accept phase", "Select a query pattern and a result row first.")
            return
        r = matches[0]
        previous_state = {
            "xrd_match": sp.meta.get("xrd_match"),
            "xrd_matches": list(sp.meta["xrd_matches"]) if sp.meta.get("xrd_matches") else None,
        }
        record = {
            "mineral": r.mineral, "name": r.name, "formula": r.formula,
            "source": r.source, "source_code": r.source_code, "spacegroup": r.spacegroup,
            "fom": r.fom, "n_matched": r.n_matched, "accepted_at": time.time(),
        }
        sp.meta["xrd_match"] = record
        accepted = list(sp.meta.get("xrd_matches", []))
        accepted.append(record)
        sp.meta["xrd_matches"] = accepted
        if self.on_accept is not None:
            self.on_accept(sp.id, previous_state)

        matched_q = {round(q, 4) for q, _ in r.matched_pairs}
        remaining = [p for p in self._query_peaks if round(p, 4) not in matched_q]
        phase_list = ", ".join(f"{m['mineral'] or m['name'] or m['formula']} [{m['source']} {m['source_code']}]"
                               for m in accepted)
        if remaining:
            self.peaks_edit.setText(", ".join(f"{p:.3f}" for p in sorted(remaining)))
            QMessageBox.information(
                self, "Phase accepted",
                f"Recorded '{record['mineral'] or record['name'] or record['formula']}' "
                f"[{r.source} {r.source_code}] for '{sp.title}'.\n"
                f"{len(remaining)} peak(s) remain unexplained — searching those now.\n"
                f"Accepted so far: {phase_list}",
            )
            self.run_search()
        else:
            self.peaks_edit.setText("")
            self._results = []
            self.results_table.setRowCount(0)
            self.plot.request_redraw(self.render_preview)
            QMessageBox.information(
                self, "All peaks explained",
                f"Every query peak of '{sp.title}' is explained by {len(accepted)} phase(s): {phase_list}.",
            )

    # ------------------------------------------------------------------
    # Raman ↔ XRD bridge
    # ------------------------------------------------------------------
    def check_raman_phases(self) -> None:
        """Overlay the XRD reference lines of every phase accepted in
        Mineral ID (meta['rruff_matches'] on any library spectrum sharing…
        for now: on the selected query spectrum itself or any spectrum in
        the library) against the current pattern."""
        sp = self._current_spectrum()
        if sp is None:
            QMessageBox.warning(self, "Raman link", "Select a query pattern first.")
            return
        # accepted Raman phases anywhere in the library (typically the Raman
        # spectrum of the same sample)
        minerals: List[str] = []
        for s in self.library.all():
            for m in s.meta.get("rruff_matches", []) or ([s.meta["rruff_match"]] if s.meta.get("rruff_match") else []):
                name = (m or {}).get("mineral")
                if name and name not in minerals:
                    minerals.append(name)
        if not minerals:
            QMessageBox.information(self, "Raman link",
                                    "No accepted Raman identifications found in the Library (Mineral ID → Accept).")
            return
        found: List[xid.XrdMatch] = []
        missing = []
        for name in minerals:
            hits = xid.find_cards_by_text(name, limit=3, db_path=self.db_path)
            if not hits:
                missing.append(name)
                continue
            h = hits[0]
            found.append(xid.XrdMatch(
                card_id=h["card_id"], source=h["source"], source_code=h["source_code"],
                name=h["name"], mineral=h["mineral"] or name, formula=h["formula"],
                spacegroup=h["spacegroup"], quality=h["quality"], rir=None,
                fom=0.0, cov_card=0.0, cov_query=0.0, n_matched=0, d=h["d"], i=h["i"],
            ))
        self._results = found
        self._on_search_done(found)
        if found:
            self.results_table.selectAll()
        msg = f"Overlaid XRD reference lines for {len(found)} Raman-identified phase(s): " \
              f"{', '.join(r.mineral or r.name for r in found)}."
        if missing:
            msg += f"\nNot in the XRD database: {', '.join(missing)}."
        self.db_status_label.setText(msg)

    # ------------------------------------------------------------------
    def browse_cards(self) -> None:
        text = self.browse_edit.text().strip()
        if not text:
            return
        self._browse_hits = xid.find_cards_by_text(text, limit=50, db_path=self.db_path)
        self.browse_list.clear()
        for h in self._browse_hits:
            label = h["mineral"] or h["name"] or h["formula"]
            self.browse_list.addItem(f"{label}  |  {h['formula']}  [{h['source']} {h['source_code']}]  {h['spacegroup']}")
        if not self._browse_hits:
            self.browse_list.addItem("(no cards found)")

    def render_browse(self) -> None:
        fig = self.browse_plot.figure
        fig.clf()
        ax = fig.add_subplot(111)
        rows = sorted({i.row() for i in self.browse_list.selectedIndexes()})
        wavelength = _to_float(self.wavelength_edit.text(), xid.CU_KA1)
        for k, row in enumerate(rows):
            if 0 <= row < len(self._browse_hits):
                h = self._browse_hits[row]
                tt = xid.d_to_two_theta(h["d"], wavelength)
                ok = np.isfinite(tt)
                ax.vlines(tt[ok], 0, h["i"][ok], color=CARD_COLORS[k % len(CARD_COLORS)], lw=1.2,
                          label=f"{h['mineral'] or h['name']} [{h['source']} {h['source_code']}]")
        if rows:
            ax.legend(fontsize=7)
        ax.set_xlabel("2θ (deg)")
        ax.set_ylabel("relative intensity")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        self.browse_plot.canvas.draw_idle()
