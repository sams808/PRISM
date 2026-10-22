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

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QMenu, QMessageBox,
    QPushButton, QScrollArea, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QWidget,
)

from core.fitting_science import find_peak_candidates
from core.qt_models import SpectrumLibrary
from core.qt_widgets import PlotWidget, to_float as _to_float
from raman.rruff_science import (
    RRUFF_ATTRIBUTION_NOTE,
    RRUFF_CACHE_DIR,
    RRUFF_CITATION,
    filter_rruff_index,
    find_cifs_for_mineral,
    format_chemistry,
    index_summary,
    load_index,
    parse_rruff_txt,
    rank_rruff_matches,
    record_details,
    rruff_page_url,
)

RESULT_COLUMNS = ["Mineral", "Formula", "RRUFF ID", "λ (nm)", "Matched", "Match %", "Quality", "Score"]


class RruffMatchWorkspace(QWidget):
    def __init__(
        self, parent: Optional[QWidget] = None, library: Optional[SpectrumLibrary] = None,
        cache_dir: Optional[str] = None, amcsd_cache_dir: Optional[str] = None,
        on_send_cifs=None, on_accept=None, library_path: Optional[str] = None,
    ):
        super().__init__(parent)
        self.library = library if library is not None else SpectrumLibrary()
        # The user's own Raman reference library file (None -> raman_library's default)
        self.library_path = library_path
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
        # Per-spectrum session state (QualX-style, same as XRD ID): accepted
        # phases stay overlaid across iterative searches, and the query
        # peaks they explain turn gray instead of vanishing.
        self._session: Dict[str, Dict[str, Any]] = {}
        # Background-download progress plumbing (queue.Queue is thread-safe;
        # the worker thread only ever puts() into it — a QTimer on the main
        # thread drains it, so no Qt widget is ever touched off-thread).
        import queue
        self._dl_queue: "queue.Queue[str]" = queue.Queue()
        self._dl_timer = QTimer(self)
        self._dl_timer.setInterval(200)
        self._dl_timer.timeout.connect(self._drain_download_log)
        self._build_ui()

    def _state(self) -> Dict[str, Any]:
        sid = self.spec_combo.currentData() or "__none__"
        return self._session.setdefault(sid, {"accepted": [], "explained": []})

    def _selected_candidates(self) -> List[Dict[str, Any]]:
        rows = self.results_table.selectionModel().selectedRows()
        return [self._results[r.row()] for r in sorted(rows, key=lambda r: r.row())
                if 0 <= r.row() < len(self._results)]

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        splitter = QSplitter()
        root.addWidget(splitter)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_scroll.setMaximumWidth(372)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left = QWidget()
        left.setObjectName("Card")
        left_layout = QVBoxLayout(left)
        left_layout.setSpacing(6)

        def section(text: str) -> None:
            lab = QLabel(text)
            lab.setObjectName("GroupLabel")
            left_layout.addWidget(lab)

        # ---- Query
        section("Query spectrum")
        self.spec_combo = QComboBox()
        self.spec_combo.currentIndexChanged.connect(
            lambda _=None: self.plot.request_redraw(lambda: self._render_preview(self._selected_candidates())))
        left_layout.addWidget(self.spec_combo)

        auto_btn = QPushButton("Auto-find peaks from spectrum")
        auto_btn.setToolTip("Fill the peak list from the spectrum's own maxima (peak-position search only).")
        auto_btn.clicked.connect(self.auto_find_peaks)
        left_layout.addWidget(auto_btn)

        left_layout.addWidget(QLabel("Peak positions (cm⁻¹, comma-separated — editable)"))
        self.peaks_edit = QLineEdit()
        left_layout.addWidget(self.peaks_edit)

        tol_row = QHBoxLayout()
        tol_row.addWidget(QLabel("Tolerance (cm⁻¹)"))
        self.tolerance_edit = QLineEdit("3.0")
        self.tolerance_edit.setToolTip("How far a reference peak may sit from a query peak and still count as matched.")
        tol_row.addWidget(self.tolerance_edit)
        left_layout.addLayout(tol_row)

        # ---- Search: peak overlap, or full-spectrum similarity over RRUFF
        # and the user's own reference library (Spectragryph parity).
        section("Search")
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Match by"))
        self.match_mode_combo = QComboBox()
        self.match_mode_combo.addItem("Peak positions", "peaks")
        self.match_mode_combo.addItem("Spectrum similarity (Pearson)", "pearson")
        self.match_mode_combo.addItem("Spectrum similarity (Euclidean)", "euclid")
        self.match_mode_combo.setToolTip("Peak positions: how many of your peaks each reference explains.\n"
                                         "Spectrum similarity: correlation of the whole spectrum shape (1 = identical).")
        self.match_mode_combo.currentIndexChanged.connect(self._on_match_mode_changed)
        mode_row.addWidget(self.match_mode_combo, 1)
        left_layout.addLayout(mode_row)
        sim_row = QHBoxLayout()
        sim_row.addWidget(QLabel("Compare"))
        self.preprocess_combo = QComboBox()
        for label, value in (("raw spectra", "raw"), ("smoothed", "smooth"), ("1st derivative", "d1"), ("2nd derivative", "d2")):
            self.preprocess_combo.addItem(label, value)
        self.preprocess_combo.setToolTip("Derivatives remove sloping baselines (fluorescence) before the correlation.")
        sim_row.addWidget(self.preprocess_combo, 1)
        sim_row.addWidget(QLabel("x range"))
        self.sim_range_edit = QLineEdit("")
        self.sim_range_edit.setPlaceholderText("e.g. 200-1200")
        self.sim_range_edit.setMaximumWidth(90)
        sim_row.addWidget(self.sim_range_edit)
        left_layout.addLayout(sim_row)
        src_row = QHBoxLayout()
        src_row.addWidget(QLabel("Sources"))
        self.source_rruff_check = QCheckBox("RRUFF")
        self.source_rruff_check.setChecked(True)
        src_row.addWidget(self.source_rruff_check)
        self.source_library_check = QCheckBox("My library")
        self.source_library_check.setToolTip("Your own reference library (Add query to my library…)")
        self.source_library_check.setChecked(True)
        src_row.addWidget(self.source_library_check)
        src_row.addStretch(1)
        left_layout.addLayout(src_row)
        self._sim_widgets = [self.preprocess_combo, self.sim_range_edit, self.source_rruff_check, self.source_library_check]
        self._on_match_mode_changed()

        find_btn = QPushButton("Find matches")
        find_btn.setToolTip("Search RRUFF (and your library) with the settings above; hits appear in the table on the right")
        find_btn.setObjectName("Primary")
        find_btn.clicked.connect(self.find_matches)
        left_layout.addWidget(find_btn)

        # ---- Database filters (user request): restrict candidates by laser
        # wavelength / orientation / scan type / quality before ranking.
        section("Database filters (RRUFF)")
        grid = QGridLayout()
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(4)
        self.wavelength_combo = QComboBox()
        self.wavelength_combo.addItem("Any", None)  # real values fill in once the index loads
        self.wavelength_combo.setToolTip("Keep only references measured with this laser (±2 nm): relative intensities depend on it.")
        self.orientation_combo = QComboBox()
        for label, value in (("Any", None), ("Oriented", True), ("Unoriented", False)):
            self.orientation_combo.addItem(label, value)
        self.orientation_combo.setToolTip("Oriented single-crystal spectra or unoriented (powder-like) ones")
        self.scan_type_combo = QComboBox()
        for label, value in (("Any", None), ("Raman (high-res)", "Raman"), ("Broad scan (LR survey)", "Broad_Scan")):
            self.scan_type_combo.addItem(label, value)
        self.scan_type_combo.setToolTip("High-resolution scans, or the low-resolution broad survey scans")
        self.quality_combo = QComboBox()
        for label, value in (("Any", None), ("Excellent", "excellent"), ("Fair", "fair"), ("Poor", "poor"), ("Unrated", "unrated")):
            self.quality_combo.addItem(label, value)
        self.quality_combo.setToolTip("RRUFF's own quality rating of the reference spectrum")
        for combo in (self.wavelength_combo, self.orientation_combo, self.scan_type_combo, self.quality_combo):
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(8)
        grid.addWidget(QLabel("Laser λ"), 0, 0)
        grid.addWidget(self.wavelength_combo, 0, 1)
        grid.addWidget(QLabel("Orientation"), 0, 2)
        grid.addWidget(self.orientation_combo, 0, 3)
        grid.addWidget(QLabel("Scan"), 1, 0)
        grid.addWidget(self.scan_type_combo, 1, 1)
        grid.addWidget(QLabel("Quality"), 1, 2)
        grid.addWidget(self.quality_combo, 1, 3)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        left_layout.addLayout(grid)

        # ---- Results
        section("Results")
        self.overlay_raw_check = QCheckBox("Overlay the candidate's measured spectrum")
        self.overlay_raw_check.setToolTip("Draw the reference spectrum itself, not only its peak positions.")
        self.overlay_raw_check.setChecked(True)
        left_layout.addWidget(self.overlay_raw_check)
        self.show_query_peaks_check = QCheckBox("Show the query peaks")
        self.show_query_peaks_check.setToolTip("The peak positions of the query (found or typed) as bars on the plot")
        self.show_candidate_peaks_check = QCheckBox("Show the candidates' peaks")
        self.show_candidate_peaks_check.setToolTip("The peak positions of the selected and accepted references as dashed lines")
        for cb in (self.show_query_peaks_check, self.show_candidate_peaks_check):
            cb.setChecked(True)
            cb.toggled.connect(lambda *_: self.plot.request_redraw(lambda: self._render_preview(self._selected_candidates())))
            left_layout.addWidget(cb)
        card_btn = QPushButton("Card / RRUFF page…")
        card_btn.setToolTip("Composition, measured chemistry, locality, source and the other header fields of the selected hit, "
                            "with a button to its page on rruff.info (also: double-click the hit)")
        card_btn.clicked.connect(self.show_candidate_card)

        accept_btn = QPushButton("Accept selected candidate as identification")
        accept_btn.setToolTip("Record the selected reference on the spectrum; for mixtures the search continues over the remaining peaks.")
        accept_btn.clicked.connect(self.accept_selected_candidate)
        left_layout.addWidget(accept_btn)

        send_cif_btn = QPushButton("Overlay its XRD pattern (CIF) in Raman")
        send_cif_btn.setToolTip("Look the mineral up in the AMCSD structures and send its predicted XRD lines to the Raman workspace.")
        send_cif_btn.clicked.connect(self.send_candidate_cifs)
        left_layout.addWidget(send_cif_btn)

        clear_row = QHBoxLayout()
        clear_row.addWidget(card_btn)
        update_btn = QPushButton("Update plot")
        update_btn.setToolTip("Redraw the query with the selected hits")
        update_btn.clicked.connect(
            lambda: self.plot.request_redraw(lambda: self._render_preview(self._selected_candidates())))
        clear_row.addWidget(update_btn)
        clear_btn = QPushButton("Clear")
        clear_btn.setToolTip("Start over on this spectrum: clears peaks, results, and the accepted-phase overlays (accepted identifications stay recorded; Ctrl+Z in the Library to undo those).")
        clear_btn.clicked.connect(self.clear_session)
        clear_row.addWidget(clear_btn)
        left_layout.addLayout(clear_row)

        # ---- the user's own reference library
        section("My reference library")
        lib_row = QHBoxLayout()
        add_lib_btn = QPushButton("Add query to my library…")
        add_lib_btn.setToolTip("Store the query spectrum, as measured, in your own Raman reference library (used by the similarity search and shareable through Home > Export databases).")
        add_lib_btn.clicked.connect(self.add_query_to_library)
        lib_row.addWidget(add_lib_btn)
        manage_lib_btn = QPushButton("Manage library…")
        manage_lib_btn.setToolTip("Rename, remove, export or import the spectra of your own reference library")
        manage_lib_btn.clicked.connect(self.manage_library)
        lib_row.addWidget(manage_lib_btn)
        left_layout.addLayout(lib_row)

        # ---- Databases: no Python needed, downloads + indexes RRUFF/AMCSD
        # from inside the app (what the shipped Download-*.bat scripts do).
        section("Databases on this computer")
        self.db_status_label = QLabel("RRUFF database not loaded yet.")
        self.db_status_label.setWordWrap(True)
        self.db_status_label.setObjectName("SectionNote")
        left_layout.addWidget(self.db_status_label)
        self.download_db_btn = QPushButton("Download RRUFF database…")
        self.download_db_btn.setToolTip(
            "Downloads ~400-500 MB from rruff.net and builds the local search index "
            "— no Python needed. Takes several minutes; safe to re-run if interrupted."
        )
        self.download_db_btn.clicked.connect(self.download_database)
        left_layout.addWidget(self.download_db_btn)
        self.download_amcsd_btn = QPushButton("Download AMCSD structures…")
        self.download_amcsd_btn.setToolTip(
            "Downloads ~66 MB of crystal structures, needed only for "
            "'Overlay its XRD pattern (CIF)'."
        )
        self.download_amcsd_btn.clicked.connect(self.download_amcsd)
        left_layout.addWidget(self.download_amcsd_btn)

        citation_label = QLabel(f"{RRUFF_CITATION}\n\n{RRUFF_ATTRIBUTION_NOTE}")
        citation_label.setWordWrap(True)
        citation_label.setObjectName("SectionNote")
        left_layout.addWidget(citation_label)
        left_layout.addStretch(1)
        left_scroll.setWidget(left)
        splitter.addWidget(left_scroll)

        right = QWidget()
        right.setObjectName("Card")
        right_layout = QVBoxLayout(right)
        self.results_label = QLabel("Hits appear here after Find matches. Click a hit to overlay it on the query; "
                                    "Shift/Ctrl-click to compare several; double-click for its card.")
        self.results_label.setObjectName("SectionNote")
        self.results_label.setWordWrap(True)
        right_layout.addWidget(self.results_label)

        self.results_table = QTableWidget(0, len(RESULT_COLUMNS))
        self.results_table.setHorizontalHeaderLabels(RESULT_COLUMNS)
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.results_table.setSelectionMode(QTableWidget.ExtendedSelection)  # shift/ctrl-click overlays several candidates
        self.results_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.results_table.itemSelectionChanged.connect(self._on_result_selected)
        self.results_table.itemDoubleClicked.connect(lambda *_: self.show_candidate_card())
        self.results_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.results_table.customContextMenuRequested.connect(self._results_menu)
        self.results_table.setToolTip("Double-click a hit for its card (composition, locality, source, RRUFF page); right-click for more.\n"
                                      "Match % = share of the query's peak intensity the reference explains; Score = geometric mean of "
                                      "that and of the share of the reference's own peaks found in the query (1 = perfect).")
        right_layout.addWidget(self.results_table, 1)

        self.plot = PlotWidget(figsize=(7, 4.5))
        right_layout.addWidget(self.plot, 1)

        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)

    # ------------------------------------------------------------------
    # No-Python-needed database setup (downloads from rruff.net)
    # ------------------------------------------------------------------
    def download_database(self) -> None:
        resp = QMessageBox.question(
            self, "Download RRUFF database",
            "This downloads roughly 400-500 MB from rruff.net and builds the local search "
            "index — no Python needed, but it takes several minutes and needs an internet "
            "connection. Continue?",
        )
        if resp != QMessageBox.Yes:
            return
        import raman.rruff_science as rs
        self._start_download(
            self.download_db_btn, "Downloading…",
            lambda: rs.download_and_build_rruff_cache(log=self._dl_queue.put),
            self._on_database_downloaded,
        )

    def _on_database_downloaded(self, n_spectra: int) -> None:
        self._index = None  # force a reload of the freshly-built index
        self._index_load_attempted = False
        self._ensure_index_loaded()
        self.db_status_label.setText(f"Downloaded and indexed {n_spectra} spectra. " + self.db_status_label.text())

    def download_amcsd(self) -> None:
        resp = QMessageBox.question(
            self, "Download AMCSD structures",
            "This downloads roughly 66 MB from rruff.net for the 'Overlay candidate's XRD (CIF)' "
            "button — no Python needed. Continue?",
        )
        if resp != QMessageBox.Yes:
            return
        import raman.rruff_science as rs
        self._start_download(
            self.download_amcsd_btn, "Downloading…",
            lambda: rs.download_and_build_amcsd_cache(log=self._dl_queue.put),
            lambda n: self.db_status_label.setText(f"Downloaded and indexed {n} AMCSD structures."),
        )

    def _start_download(self, button: QPushButton, busy_text: str, work, on_done) -> None:
        """Both download buttons disable together — the RRUFF and AMCSD
        downloads share one progress queue/status label, so running them
        concurrently would interleave their log lines."""
        self.download_db_btn.setEnabled(False)
        self.download_amcsd_btn.setEnabled(False)
        button.setText(busy_text)
        self.db_status_label.setText("Starting download…")
        self._dl_timer.start()
        from core.qt_worker import run_in_thread
        run_in_thread(work, lambda result: self._on_download_done(on_done, result),
                     self._on_download_error)

    def _drain_download_log(self) -> None:
        """Runs on the main thread every 200ms while a download is active;
        the worker thread only ever puts() into the queue (thread-safe),
        never touches a widget directly."""
        last = None
        while not self._dl_queue.empty():
            last = self._dl_queue.get_nowait()
        if last is not None:
            self.db_status_label.setText(last)

    def _reset_download_buttons(self) -> None:
        self._dl_timer.stop()
        self.download_db_btn.setEnabled(True)
        self.download_db_btn.setText("Download RRUFF database…")
        self.download_amcsd_btn.setEnabled(True)
        self.download_amcsd_btn.setText("Download AMCSD structures…")

    def _on_download_done(self, on_done, result) -> None:
        self._reset_download_buttons()
        on_done(result)

    def _on_download_error(self, traceback_text: str) -> None:
        self._reset_download_buttons()
        QMessageBox.critical(self, "Download failed", traceback_text)

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
        # Eager index load on page entry (user report: the λ filter was
        # empty until the first search, because the index only loaded then).
        if self._index is None and not getattr(self, "_index_load_attempted", False):
            self._index_load_attempted = True
            self._ensure_index_loaded()
        # Arriving on the page must show the spectrum immediately, not an
        # empty axes (user request — same rule on every workspace).
        self.plot.request_redraw(lambda: self._render_preview(self._selected_candidates()))

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
                "No RRUFF database on this computer yet: use 'Download RRUFF database…' below "
                "(internet, ~450 MB, a few minutes), or Home › Import a database pack from a "
                "colleague. Your own reference library works without it."
            )
            return False
        summary = index_summary(self._index)
        self.db_status_label.setText(
            f"RRUFF database: {summary['n_spectra']} spectra, {summary['n_minerals']} minerals, "
            f"{len(summary['wavelengths_nm'])} wavelengths (loaded in {elapsed:.2f}s)."
        )
        # Fill the λ filter with the wavelengths actually present, keeping
        # any selection the user already made.
        current = self.wavelength_combo.currentData()
        self.wavelength_combo.blockSignals(True)
        self.wavelength_combo.clear()
        self.wavelength_combo.addItem("Any", None)
        for wl in summary["wavelengths_nm"]:
            self.wavelength_combo.addItem(f"{wl:g} nm", float(wl))
        if current is not None:
            idx = self.wavelength_combo.findData(current)
            if idx >= 0:
                self.wavelength_combo.setCurrentIndex(idx)
        self.wavelength_combo.blockSignals(False)
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
        self.peaks_edit.setCursorPosition(0)       # show the list from its first value, not scrolled to the end

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

    # ------------------------------------------------------------------
    # Similarity search + the user's reference library
    # ------------------------------------------------------------------
    def _on_match_mode_changed(self, *_) -> None:
        sim = self.match_mode_combo.currentData() != "peaks"
        for w in self._sim_widgets:
            w.setEnabled(sim)

    def _library_path(self) -> Optional[str]:
        return getattr(self, "library_path", None)  # tests point this at a temp file

    def _sim_x_range(self):
        text = (self.sim_range_edit.text() or "").strip()
        if not text:
            return None
        import re
        nums = re.findall(r"[-+]?\d+\.?\d*", text)
        if len(nums) != 2:
            raise ValueError("x range: give 'min-max' (e.g. 200-1200) or leave it empty.")
        return (float(nums[0]), float(nums[1]))

    def find_by_similarity(self) -> None:
        import raman.raman_library as rl
        spectrum = self._current_spectrum()
        if spectrum is None:
            QMessageBox.warning(self, "Find matches", "Select a query spectrum first.")
            return
        try:
            x_range = self._sim_x_range()
        except ValueError as exc:
            QMessageBox.warning(self, "Find matches", str(exc))
            return
        method = self.match_mode_combo.currentData()
        how = self.preprocess_combo.currentData()
        use_rruff = self.source_rruff_check.isChecked()
        use_lib = self.source_library_check.isChecked()
        if not (use_rruff or use_lib):
            QMessageBox.warning(self, "Find matches", "Tick at least one source (RRUFF or your library).")
            return
        records = []
        if use_rruff:
            if self._ensure_index_loaded():
                records = filter_rruff_index(
                    self._index,
                    wavelength_nm=self.wavelength_combo.currentData(),
                    oriented=self.orientation_combo.currentData(),
                    scan_type=self.scan_type_combo.currentData(),
                    quality=self.quality_combo.currentData(),
                )
                accepted_ids = {m.get("rruff_id") for m in spectrum.meta.get("rruff_matches", [])}
                records = [r for r in records if r.get("rruff_id") not in accepted_ids]
        entries = rl.load_library(self._library_path()) if use_lib else []
        if not records and not entries:
            QMessageBox.information(self, "Find matches", "Nothing to search: no RRUFF spectra pass the filters and the reference library is empty.")
            return
        qx, qy = np.asarray(spectrum.x, float), np.asarray(spectrum.y, float)
        self._query_peaks = self._parse_peaks()
        self.db_status_label.setText(f"Similarity search over {len(records)} RRUFF + {len(entries)} library spectra…")

        def job(progress=None):
            hits = []
            if entries:
                for r in rl.rank_library(qx, qy, entries, method=method, how=how, x_range=x_range, top_n=50, progress=progress):
                    e = r["entry"]
                    hits.append({"mineral": e.name, "rruff_id": "library", "wavelength_nm": e.meta.get("laser_nm"),
                                 "category": "my library", "peaks": [], "entry": e, "score": r["score"],
                                 "library_id": e.id, "source": e.meta.get("source", "")})
            if records:
                hits.extend(rl.rank_rruff_by_similarity(qx, qy, records, method=method, how=how, x_range=x_range,
                                                        top_n=50, progress=progress))
            hits.sort(key=lambda h: h["score"], reverse=True)
            return hits[:25]

        from core.qt_worker import run_in_thread
        run_in_thread(job, self._on_similarity_done, self._on_similarity_error,
                      on_progress=lambda i, n, what: self.db_status_label.setText(f"Similarity search: {what} {i}/{n}…"))

    def _on_similarity_done(self, hits) -> None:
        self._results = list(hits)
        self._populate_results_table()
        summary = f"{len(hits)} hit(s) by {self.match_mode_combo.currentText().lower()}; score 1 = identical shape."
        self.db_status_label.setText(summary)
        self.results_label.setText(summary + " Click a hit to overlay it; double-click for its card.")
        self._render_preview([self._results[0]] if self._results else [])

    def _on_similarity_error(self, traceback_text: str) -> None:
        QMessageBox.critical(self, "Similarity search failed", traceback_text)

    def add_query_to_library(self) -> None:
        import raman.raman_library as rl
        from raman.qt_raman_dialogs import AddToLibraryDialog
        spectrum = self._current_spectrum()
        if spectrum is None:
            QMessageBox.warning(self, "Reference library", "Select a query spectrum first.")
            return
        dlg = AddToLibraryDialog(self, [spectrum.title])
        if not dlg.exec():
            return
        v = dlg.values()
        path = self._library_path()
        entries = rl.load_library(path)
        entries.append(rl.entry_from_spectrum(v["name"] or spectrum.title, spectrum.x, spectrum.y,
                                              laser_nm=v["laser_nm"], source=v["source"], notes=v["notes"],
                                              extra={"origin_path": spectrum.path}))
        rl.save_library(entries, path)
        self.db_status_label.setText(f"Added '{v['name'] or spectrum.title}' to the reference library ({len(entries)} entries).")

    def manage_library(self) -> None:
        from raman.qt_raman_dialogs import LibraryManagerDialog
        LibraryManagerDialog(self, library_path=self._library_path()).exec()

    def find_matches(self) -> None:
        if self.match_mode_combo.currentData() != "peaks":
            self.find_by_similarity()
            return
        if not self._ensure_index_loaded():
            return
        peaks = self._parse_peaks()
        if not peaks:
            QMessageBox.warning(self, "Find matches", "Enter or auto-find at least one peak position.")
            return
        tolerance = _to_float(self.tolerance_edit.text(), 3.0)
        self._query_peaks = peaks
        searchable = filter_rruff_index(
            self._index,
            wavelength_nm=self.wavelength_combo.currentData(),
            oriented=self.orientation_combo.currentData(),
            scan_type=self.scan_type_combo.currentData(),
            quality=self.quality_combo.currentData(),
        )
        if not searchable:
            QMessageBox.information(self, "Find matches", "No database spectra pass the current filters — relax them and retry.")
            self._results = []
            self._populate_results_table()
            return
        # Already-accepted references for this spectrum drop out of further
        # searches — accept means "this phase is identified", so the search
        # continues over the remaining candidates/peaks.
        spectrum = self._current_spectrum()
        accepted_ids = {m.get("rruff_id") for m in (spectrum.meta.get("rruff_matches", []) if spectrum else [])}
        if accepted_ids:
            searchable = [r for r in searchable if r.get("rruff_id") not in accepted_ids]
        filter_note = ""
        if len(searchable) < len(self._index):
            filter_note = (f" Filters keep {len(searchable)} of {len(self._index)} database spectra"
                           + (f" ({len(accepted_ids)} already-accepted excluded)." if accepted_ids else "."))
        qxy = (np.asarray(spectrum.x, float), np.asarray(spectrum.y, float)) if spectrum is not None else None
        n_searchable = len(searchable)
        self.db_status_label.setText(f"Peak search over {n_searchable} spectra…")

        def job(progress=None):
            return rank_rruff_matches(peaks, searchable, tolerance=tolerance, top_n=25, query_xy=qxy, progress=progress)

        def done(hits):
            self._results = list(hits)
            self._populate_results_table()
            best = float(hits[0]["score"]) if hits else 0.0
            verdict = ("convincing" if best >= 0.6 else "possible: judge the overlay" if best >= 0.4
                       else "weak: no convincing match in the database")
            summary = f"{len(hits)} hit(s) over {n_searchable} spectra; best score {best:.2f} ({verdict})."
            self.db_status_label.setText(summary + filter_note)
            self.results_label.setText(summary + " Click a hit to overlay it; double-click for its card.")
            self._render_preview([self._results[0]] if self._results else [])

        from core.qt_worker import run_in_thread
        run_in_thread(job, done, self._on_search_error,
                      on_progress=lambda i, n, what: self.db_status_label.setText(f"Peak search: {what} {i}/{n}…"))

    def _on_search_error(self, traceback_text: str) -> None:
        QMessageBox.critical(self, "Search failed", traceback_text)

    def _populate_results_table(self) -> None:
        self.results_table.setRowCount(len(self._results))
        for row, rec in enumerate(self._results):
            wl = rec.get("wavelength_nm")
            try:
                wl_text = f"{float(wl):.0f}" if wl not in (None, "") else ""
            except (TypeError, ValueError):
                wl_text = str(wl)
            values = [
                rec.get("mineral", ""), format_chemistry(self._formula_of(rec)), rec.get("rruff_id", ""), wl_text,
                str(rec.get("matched_peaks", "")) if "matched_peaks" in rec else "",
                f"{rec.get('match_fraction', 0.0) * 100:.0f}%" if "match_fraction" in rec else "",
                str(rec.get("category", "")).split("_")[0],
                f"{rec['score']:.3f}" if "score" in rec else "",
            ]
            for col, val in enumerate(values):
                self.results_table.setItem(row, col, QTableWidgetItem(str(val)))
            self.results_table.item(row, 6).setToolTip(str(rec.get("category", "")))
        self.results_table.resizeColumnsToContents()
        if self.results_table.columnWidth(1) > 220:
            self.results_table.setColumnWidth(1, 220)      # a long formula must not push the scores off screen
        self.results_table.horizontalHeader().setStretchLastSection(True)
        if self._results:
            self.results_table.selectRow(0)

    def _on_result_selected(self) -> None:
        rows = self.results_table.selectionModel().selectedRows()
        if not rows or not self._results:
            return
        self._render_preview(self._selected_candidates())

    # ------------------------------------------------------------------
    CANDIDATE_COLORS = ["crimson", "royalblue", "seagreen", "darkorange", "purple", "teal"]
    # Muted palette for phases already accepted (they stay overlaid)
    ACCEPTED_COLORS = ["#9c6b74", "#6b7d9c", "#6b9c7d", "#9c8a6b", "#856b9c", "#6b969c"]

    def _render_preview(self, candidates: List[Dict[str, Any]]) -> None:
        """Overlay the query against one or several candidates (shift/ctrl-
        click rows to compare references side by side)."""
        self.plot.cancel_pending()  # direct draws supersede any queued entry render
        # keep the user's zoom across candidate re-renders (same spectrum)
        self.plot.preserve_zoom(("raman_id", self.spec_combo.currentData()))
        fig = self.plot.figure
        fig.clf()
        ax = fig.add_subplot(111)

        spectrum = self._current_spectrum()
        if spectrum is not None:
            y = spectrum.y
            y_norm = y / np.nanmax(np.abs(y)) if np.nanmax(np.abs(y)) > 0 else y
            ax.plot(spectrum.x, y_norm, color="black", lw=1.1, label=f"query: {spectrum.title}")

        # Query peak bars (same convention as XRD ID): current unexplained
        # peaks in black, peaks explained by accepted phases in gray.
        state = self._state()
        peaks = self._parse_peaks()
        show_query = self.show_query_peaks_check.isChecked()
        show_cand = self.show_candidate_peaks_check.isChecked()
        if peaks and show_query:
            ax.vlines(peaks, 0, 1.04, color="black", lw=0.8, alpha=0.30,
                      label="query peaks")
        if state["explained"] and show_query:
            ax.vlines(state["explained"], 0, 1.04, color="gray", lw=0.8,
                      alpha=0.45, ls="--", label="explained by accepted phases")

        # Accepted phases stay overlaid across iterative searches (muted).
        for k, acc in enumerate(state["accepted"]):
            color = self.ACCEPTED_COLORS[k % len(self.ACCEPTED_COLORS)]
            acc_label = f"accepted: {acc.get('mineral', '?')} ({acc.get('rruff_id', '?')})"
            for j, peak in enumerate(acc.get("peaks", []) if show_cand else []):
                ax.axvline(peak, color=color, ls=":", lw=0.9, alpha=0.6,
                           label=acc_label if j == 0 else None)

        for k, candidate in enumerate(candidates):
            color = self.CANDIDATE_COLORS[k % len(self.CANDIDATE_COLORS)]
            formula = format_chemistry(self._formula_of(candidate))
            label = (f"{candidate.get('mineral', '?')}" + (f" {formula}" if formula else "")
                     + f" ({candidate.get('rruff_id', '?')}, {candidate.get('wavelength_nm', '?')} nm)")
            drew_curve = False
            if candidate.get("entry") is not None:  # a hit from the user's reference library
                ex, ey = candidate["entry"].arrays()
                if ey.size and np.nanmax(np.abs(ey)) > 0:
                    ax.plot(ex, ey / np.nanmax(np.abs(ey)), color=color, lw=1.0, alpha=0.8,
                            label=f"{candidate.get('mineral', '?')} (my library, score {candidate.get('score', float('nan')):.3f})")
                    drew_curve = True
            elif self.overlay_raw_check.isChecked() and candidate.get("raw_path") and os.path.isfile(candidate["raw_path"]):
                try:
                    with open(candidate["raw_path"], "r", encoding="utf-8", errors="replace") as f:
                        text = f.read()
                    ref = parse_rruff_txt(text, source_filename=os.path.basename(candidate["raw_path"]))
                    if len(ref.y):
                        ref_norm = ref.y / np.nanmax(np.abs(ref.y)) if np.nanmax(np.abs(ref.y)) > 0 else ref.y
                        ax.plot(ref.x, ref_norm, color=color, lw=1.0, alpha=0.8, label=label)
                        drew_curve = True
                except OSError:
                    pass
            for j, peak in enumerate(candidate.get("peaks", []) if show_cand else []):
                ax.axvline(peak, color=color, ls="--", lw=0.7, alpha=0.5,
                           label=label if (not drew_curve and j == 0) else None)

        ax.set_xlabel("Raman shift (cm⁻¹)")
        ax.set_ylabel("Normalized intensity")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        self.plot.restore_zoom(ax)
        fig.tight_layout()
        self.plot.canvas.draw_idle()

    # ------------------------------------------------------------------
    def _formula_of(self, rec: Dict[str, Any]) -> str:
        """Ideal chemistry of a hit: from the index, else read once from the
        header of its data file (older indexes have no chemistry field)."""
        if rec.get("entry") is not None:
            return str(rec["entry"].meta.get("formula") or "")
        f = rec.get("ideal_chemistry")
        if f is None:
            f = ""
            p = rec.get("raw_path")
            if p and os.path.isfile(p):
                try:
                    with open(p, "r", encoding="utf-8", errors="replace") as fh:
                        for _ in range(40):
                            line = fh.readline()
                            if not line:
                                break
                            if line.startswith("##IDEAL CHEMISTRY="):
                                f = line.partition("=")[2].strip()
                                break
                except OSError:
                    f = ""
            rec["ideal_chemistry"] = f
        return str(f or "")

    def _results_menu(self, pos) -> None:
        if not self._selected_candidates():
            return
        menu = QMenu(self)
        menu.addAction("Card…", self.show_candidate_card)
        menu.addAction("Open the RRUFF page", self.open_candidate_page)
        menu.exec(self.results_table.viewport().mapToGlobal(pos))

    def open_candidate_page(self) -> None:
        cands = self._selected_candidates()
        if not cands:
            return
        if cands[0].get("entry") is not None:
            QMessageBox.information(self, "RRUFF page", "This hit is a spectrum of your own reference library; it has no RRUFF page.")
            return
        QDesktopServices.openUrl(QUrl(rruff_page_url(record_details(cands[0]))))

    def show_candidate_card(self) -> None:
        cands = self._selected_candidates()
        if not cands:
            QMessageBox.information(self, "Card", "Select a hit in the table first.")
            return
        from raman.qt_raman_dialogs import RruffCardDialog
        RruffCardDialog(self, cands[0]).exec()

    def clear_session(self) -> None:
        """Start over on the current spectrum: peaks, results, and the
        accepted-overlay/explained-peak session state (identifications
        recorded on the spectrum itself are untouched — undo those from
        the Library)."""
        sid = self.spec_combo.currentData() or "__none__"
        self._session.pop(sid, None)
        self.peaks_edit.setText("")
        self._query_peaks = []
        self._results = []
        self._populate_results_table()
        self.plot.reset_zoom_memory()  # starting over autoscales again
        self.plot.request_redraw(lambda: self._render_preview([]))

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
        """Accept = 'this phase is identified': record it, remove the peaks
        it explains from the query, and immediately re-search the REMAINING
        peaks for further phases (user request — mixtures are the norm, one
        spectrum can hold several minerals). Accepted references are
        excluded from the follow-up search; every accept lands on the undo
        stack (Ctrl+Z in the Library)."""
        rows = self.results_table.selectionModel().selectedRows()
        spectrum = self._current_spectrum()
        if spectrum is None or not rows or not self._results:
            QMessageBox.warning(self, "Accept identification", "Select a query spectrum and a candidate row first.")
            return
        candidate = self._results[rows[0].row()]
        tolerance = _to_float(self.tolerance_edit.text(), 3.0)

        previous_state = {
            "rruff_match": spectrum.meta.get("rruff_match"),
            "rruff_matches": list(spectrum.meta["rruff_matches"]) if spectrum.meta.get("rruff_matches") else None,
        }
        match_record = {
            "mineral": candidate.get("mineral"),
            "rruff_id": candidate.get("rruff_id"),
            "wavelength_nm": candidate.get("wavelength_nm"),
            "matched_peaks": candidate.get("matched_peaks"),
            "match_fraction": candidate.get("match_fraction"),
            "score": candidate.get("score"),
            "library_id": candidate.get("library_id"),
            "accepted_at": time.time(),
        }
        spectrum.meta["rruff_match"] = match_record  # latest accept (back-compat: reports/projects read this)
        accepted = list(spectrum.meta.get("rruff_matches", []))
        accepted.append(match_record)
        spectrum.meta["rruff_matches"] = accepted
        if self.on_accept is not None:
            self.on_accept(spectrum.id, previous_state)

        cand_peaks = candidate.get("peaks", []) or []
        remaining = [p for p in self._query_peaks
                     if not any(abs(p - cp) <= tolerance for cp in cand_peaks)]
        # session overlays: the accepted phase's bars stay, its peaks gray out
        state = self._state()
        state["accepted"].append(dict(candidate))
        state["explained"].extend(p for p in self._query_peaks if p not in remaining)
        phase_list = ", ".join(f"{m['mineral']} ({m['rruff_id']})" for m in accepted)

        if remaining:
            self.peaks_edit.setText(", ".join(f"{p:.1f}" for p in sorted(remaining)))
            QMessageBox.information(
                self, "Phase accepted",
                f"Recorded '{candidate.get('mineral')}' ({candidate.get('rruff_id')}) for '{spectrum.title}'.\n"
                f"{len(remaining)} peak(s) remain unexplained — the table now shows matches for those.\n"
                f"Accepted so far: {phase_list}\n\n{RRUFF_CITATION}",
            )
            self.find_matches()
        else:
            self.peaks_edit.setText("")
            self._results = []
            self._populate_results_table()
            self._render_preview([])
            QMessageBox.information(
                self, "All peaks explained",
                f"Every query peak of '{spectrum.title}' is now explained by {len(accepted)} accepted phase(s): "
                f"{phase_list}.\n\n{RRUFF_CITATION}",
            )
