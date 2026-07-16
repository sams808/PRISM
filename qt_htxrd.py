"""
qt_htxrd.py — HTXRD series-processing workspace (M20).

Thin Qt layer over htxrd_science.py: import a whole folder (or file
selection) of diffraction patterns as one temperature-ordered series
(.rasx Temp metadata -> Jana-style filename template -> sequence index),
view it as a stacked/waterfall plot ordered by temperature, track one peak
across the series inside a user-chosen 2theta window (center/FWHM/area vs
temperature — the tractable analog of Jana's cell-parameter Graph tool),
and flag candidate phase-transition windows from fit-quality jumps.
"""
from __future__ import annotations

import csv
from typing import List, Optional

import numpy as np

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QMessageBox, QPushButton, QSplitter, QTabWidget,
    QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget,
)

from htxrd_science import (
    HtxrdPattern,
    PeakTrackResult,
    find_series_files,
    flag_transition_candidates,
    load_series,
    track_peak,
)
from qt_widgets import PlotWidget


def _to_float(text: str, default: Optional[float] = None) -> Optional[float]:
    try:
        return float((text or "").strip())
    except (TypeError, ValueError):
        return default


class HtxrdWorkspace(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.series: List[HtxrdPattern] = []
        self.track_results: List[PeakTrackResult] = []
        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        splitter = QSplitter()
        root.addWidget(splitter)

        left = QWidget()
        left.setObjectName("Card")
        left.setMaximumWidth(330)
        left_layout = QVBoxLayout(left)

        import_row = QHBoxLayout()
        folder_btn = QPushButton("Import folder…")
        folder_btn.clicked.connect(self.import_folder)
        files_btn = QPushButton("Import files…")
        files_btn.clicked.connect(self.import_files)
        import_row.addWidget(folder_btn)
        import_row.addWidget(files_btn)
        left_layout.addLayout(import_row)

        left_layout.addWidget(QLabel("Filename template (Jana-style, e.g. scan_???.xy)\nUsed only when a file has no temperature metadata"))
        self.template_edit = QLineEdit()
        left_layout.addWidget(self.template_edit)

        left_layout.addWidget(QLabel("Series (ordered by ramp value)"))
        self.series_list = QListWidget()
        self.series_list.itemSelectionChanged.connect(lambda: self.plot.request_redraw(self.render_waterfall))
        left_layout.addWidget(self.series_list, 1)

        shift_row = QHBoxLayout()
        shift_row.addWidget(QLabel("Waterfall shift ×"))
        self.shift_edit = QLineEdit("1.0")
        self.shift_edit.setMaximumWidth(60)
        self.shift_edit.textChanged.connect(lambda: self.plot.request_redraw(self.render_waterfall))
        shift_row.addWidget(self.shift_edit)
        self.normalize_check = QCheckBox("Normalize")
        self.normalize_check.toggled.connect(lambda: self.plot.request_redraw(self.render_waterfall))
        shift_row.addWidget(self.normalize_check)
        shift_row.addStretch(1)
        left_layout.addLayout(shift_row)

        left_layout.addWidget(QLabel("Peak tracking — 2θ window"))
        window_row = QHBoxLayout()
        self.window_lo_edit = QLineEdit()
        self.window_lo_edit.setPlaceholderText("2θ min")
        self.window_hi_edit = QLineEdit()
        self.window_hi_edit.setPlaceholderText("2θ max")
        window_row.addWidget(self.window_lo_edit)
        window_row.addWidget(self.window_hi_edit)
        left_layout.addLayout(window_row)

        shape_row = QHBoxLayout()
        shape_row.addWidget(QLabel("Peak shape"))
        self.shape_combo = QComboBox()
        self.shape_combo.addItems(["G", "GL"])
        shape_row.addWidget(self.shape_combo)
        left_layout.addLayout(shape_row)

        self._track_btn = QPushButton("Track peak across series")
        self._track_btn.setObjectName("Primary")
        self._track_btn.clicked.connect(self.run_track_peak)
        left_layout.addWidget(self._track_btn)

        export_btn = QPushButton("Export tracking results to CSV…")
        export_btn.clicked.connect(self.export_track_csv)
        left_layout.addWidget(export_btn)

        self.status_label = QLabel("Import a series to begin.")
        self.status_label.setWordWrap(True)
        left_layout.addWidget(self.status_label)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()

        waterfall_tab = QWidget()
        wl = QVBoxLayout(waterfall_tab)
        self.plot = PlotWidget(figsize=(7, 5.5))
        wl.addWidget(self.plot)
        self.tabs.addTab(waterfall_tab, "Waterfall")

        track_tab = QWidget()
        tl = QVBoxLayout(track_tab)
        self.track_plot = PlotWidget(figsize=(7, 5))
        tl.addWidget(self.track_plot, 2)
        self.track_table = QTableWidget(0, 7)
        self.track_table.setHorizontalHeaderLabels(["Pattern", "Ramp", "Center", "FWHM", "Amplitude", "Area", "chi2_red"])
        self.track_table.setEditTriggers(QTableWidget.NoEditTriggers)
        tl.addWidget(self.track_table, 1)
        self.transition_text = QTextEdit()
        self.transition_text.setReadOnly(True)
        self.transition_text.setMaximumHeight(90)
        tl.addWidget(self.transition_text)
        self.tabs.addTab(track_tab, "Peak tracking")

        right_layout.addWidget(self.tabs)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)

    # ------------------------------------------------------------------
    def import_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select folder of diffraction patterns")
        if not folder:
            return
        paths = find_series_files(folder)
        if not paths:
            QMessageBox.information(self, "Import series", "No pattern files (.rasx/.xy/.xrdml/.txt/.dat) found in that folder.")
            return
        self._load(paths)

    def import_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select diffraction patterns", "",
            "Diffraction patterns (*.rasx *.xy *.xrdml *.txt *.dat);;All files (*.*)",
        )
        if not paths:
            return
        self._load(list(paths))

    def _load(self, paths: List[str]) -> None:
        template = self.template_edit.text().strip()
        try:
            series = load_series(paths, filename_template=template)
        except Exception as exc:
            QMessageBox.critical(self, "Import series error", str(exc))
            return
        self.set_series(series)

    def set_series(self, series: List[HtxrdPattern]) -> None:
        """Adopt an already-loaded series (fresh import or project restore)
        and refresh the list + waterfall."""
        self.series = list(series)
        self.series_list.clear()
        sources = set()
        for pat in self.series:
            self.series_list.addItem(f"{pat.name}   [{pat.ramp_value:g}{'°C' if pat.ramp_source == 'metadata' else ''}, {pat.ramp_source}]")
            sources.add(pat.ramp_source)
        self.series_list.selectAll()
        self.status_label.setText(
            f"Loaded {len(self.series)} patterns. Ramp source(s): {', '.join(sorted(sources))}."
            + (" WARNING: some ramp values are just sequence indices — add a filename template or use files with temperature metadata."
               if "index" in sources else "")
        )
        self.render_waterfall()
        self.plot.canvas.draw_idle()

    def _selected_patterns(self) -> List[HtxrdPattern]:
        selected_rows = {i.row() for i in self.series_list.selectedIndexes()}
        if not selected_rows:
            return list(self.series)
        return [p for i, p in enumerate(self.series) if i in selected_rows]

    # ------------------------------------------------------------------
    def render_waterfall(self) -> None:
        patterns = self._selected_patterns()
        fig = self.plot.figure
        fig.clf()
        if not patterns:
            self.plot.canvas.draw_idle()
            return
        ax = fig.add_subplot(111)

        shift_factor = _to_float(self.shift_edit.text(), 1.0) or 1.0
        max_dy = max(((np.nanmax(p.y) - np.nanmin(p.y)) for p in patterns), default=1.0)
        delta = 0.5 * shift_factor * max_dy

        cmap = None
        try:
            import matplotlib
            import matplotlib.colors as mcolors
            ramps = [p.ramp_value for p in patterns]
            norm = mcolors.Normalize(vmin=min(ramps), vmax=max(ramps) or 1.0)
            cmap = matplotlib.colormaps["inferno"]
        except Exception:
            norm = None

        for i, pat in enumerate(patterns):
            y = pat.y.astype(float)
            if self.normalize_check.isChecked():
                rng = np.nanmax(y) - np.nanmin(y)
                if rng > 0:
                    y = (y - np.nanmin(y)) / rng * max_dy
            color = cmap(norm(pat.ramp_value)) if (cmap is not None and norm is not None) else None
            label = f"{pat.ramp_value:g}" + ("°C" if pat.ramp_source == "metadata" else "")
            ax.plot(pat.x, y + i * delta, lw=0.8, color=color, label=label if len(patterns) <= 15 else None)

        ax.set_xlabel("2θ (deg)")
        ax.set_ylabel("Intensity (offset)")
        ax.set_title(f"HTXRD series — {len(patterns)} patterns (bottom = lowest ramp)")
        if len(patterns) <= 15:
            ax.legend(fontsize=7, title="Ramp")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        self.plot.canvas.draw_idle()

    # ------------------------------------------------------------------
    def run_track_peak(self) -> None:
        if not self.series:
            QMessageBox.warning(self, "Peak tracking", "Import a series first.")
            return
        lo = _to_float(self.window_lo_edit.text())
        hi = _to_float(self.window_hi_edit.text())
        if lo is None or hi is None or hi <= lo:
            QMessageBox.warning(self, "Peak tracking", "Enter a valid 2θ window (min < max).")
            return

        patterns = self._selected_patterns()
        shape = self.shape_combo.currentText()

        self._track_btn.setEnabled(False)
        self._track_btn.setText("Tracking…")

        from qt_worker import run_in_thread
        run_in_thread(
            lambda: track_peak(patterns, window_lo=lo, window_hi=hi, shape=shape),
            self._on_track_done, self._on_track_error,
        )

    def _on_track_error(self, traceback_text: str) -> None:
        self._track_btn.setEnabled(True)
        self._track_btn.setText("Track peak across series")
        QMessageBox.critical(self, "Peak tracking error", traceback_text)

    def _on_track_done(self, results) -> None:
        self._track_btn.setEnabled(True)
        self._track_btn.setText("Track peak across series")
        self.track_results = results
        self._populate_track_table()
        self._render_track_plot()
        flags = flag_transition_candidates(self.track_results)
        if flags:
            lines = ["Candidate phase-transition signatures (fit-quality anomalies):"]
            for ramp, chi2, reason in flags:
                lines.append(f"  ramp={ramp:g}: {reason}")
            self.transition_text.setPlainText("\n".join(lines))
        else:
            self.transition_text.setPlainText("No fit-quality anomalies flagged across the series.")
        self.tabs.setCurrentIndex(1)

    def _populate_track_table(self) -> None:
        self.track_table.setRowCount(len(self.track_results))
        for row, r in enumerate(self.track_results):
            values = [r.pattern_name, f"{r.ramp_value:g}"]
            if r.error is None:
                values += [f"{r.center:.4f}", f"{r.fwhm:.4f}", f"{r.amplitude:.4g}", f"{r.area:.4g}", f"{r.chi2_red:.4g}"]
            else:
                values += ["—", "—", "—", "—", f"ERROR: {r.error}"]
            for col, val in enumerate(values):
                self.track_table.setItem(row, col, QTableWidgetItem(str(val)))
        self.track_table.resizeColumnsToContents()

    def _render_track_plot(self) -> None:
        fig = self.track_plot.figure
        fig.clf()
        ok = [r for r in self.track_results if r.error is None]
        if not ok:
            self.track_plot.canvas.draw_idle()
            return
        ramps = [r.ramp_value for r in ok]
        ax_c, ax_w, ax_a = fig.subplots(3, 1, sharex=True)
        ax_c.plot(ramps, [r.center for r in ok], "o-", ms=4, lw=1)
        ax_c.set_ylabel("Center (°2θ)")
        ax_c.grid(alpha=0.25)
        ax_w.plot(ramps, [r.fwhm for r in ok], "s-", ms=4, lw=1, color="seagreen")
        ax_w.set_ylabel("FWHM (°2θ)")
        ax_w.grid(alpha=0.25)
        ax_a.plot(ramps, [r.area for r in ok], "^-", ms=4, lw=1, color="darkorange")
        ax_a.set_ylabel("Area")
        ax_a.set_xlabel("Ramp value (temperature)")
        ax_a.grid(alpha=0.25)
        fig.suptitle("Tracked peak vs. ramp", fontsize=10)
        fig.tight_layout()
        self.track_plot.canvas.draw_idle()

    # ------------------------------------------------------------------
    def export_track_csv(self) -> None:
        if not self.track_results:
            QMessageBox.information(self, "Export", "Run peak tracking first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export tracking results", "", "CSV (*.csv)")
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["pattern", "ramp_value", "center_2theta", "fwhm_2theta", "amplitude", "area", "chi2_red", "error"])
                for r in self.track_results:
                    writer.writerow([r.pattern_name, r.ramp_value, r.center, r.fwhm, r.amplitude, r.area, r.chi2_red, r.error or ""])
        except OSError as exc:
            QMessageBox.critical(self, "Export error", str(exc))
            return
        QMessageBox.information(self, "Export", f"Tracking results exported to:\n{path}")
