"""qt_saxs2.py - the Frames and Series figures tabs of the SAXS/WAXS workspace, mixed into SaxsWorkspace (the Fit
tab between them comes from qt_saxs_fit.py):

* **Frames**: inventory of raw 2D frames (roles, sessions, refined beam centres, lobe / streak
  diagnostics), pairing with the session empties, batch sector reduction with loading classes and
  per-mm normalisation; the reduced curves join the workspace list.
* **Series figures**: a series table (import / edit) drives the grid figure, the fit-panel figures (from the Fit tab
  results), the trends figure and the 2D frame figure; export as pdf / svg / png; tables as xlsx / csv / md.

Every long step runs through core.qt_worker.run_in_thread with progress. The science lives in
saxs.core (frames, sector_batch, descriptors, fit_workflow, series_figures, tables); nothing here
knows a sample.
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import pandas as pd

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QProgressBar, QPushButton, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from core.qt_widgets import PlotWidget, to_float as _to_float

SERIES_COLUMNS = ["sample", "panel_row", "panel_col", "stack_index", "label", "colour", "marker", "linestyle", "x_value",
                  "family_label", "col_title", "quantified", "note_symbol", "group", "extra_panels", "fit_order"]


def _item(text) -> QTableWidgetItem:
    it = QTableWidgetItem("" if text is None else str(text))
    it.setFlags(it.flags() & ~Qt.ItemIsEditable)
    return it


def _fmt(v, nd: int = 4) -> str:
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (float, np.floating)):
        return "" if not np.isfinite(v) else f"{v:.{nd}g}"
    return str(v)


class FigurePreview(QWidget):
    """A canvas + toolbar that can take any matplotlib Figure (the series figures are built as
    standalone Figure objects)."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = None
        self.toolbar = None
        self.figure = None
        self.placeholder = QLabel("Build a figure with the buttons on the left.")
        self.placeholder.setAlignment(Qt.AlignCenter)
        self._layout.addWidget(self.placeholder)

    def set_figure(self, fig) -> None:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT

        for w in (self.toolbar, self.canvas, self.placeholder):
            if w is not None:
                self._layout.removeWidget(w)
                w.setParent(None)
                w.deleteLater()
        self.placeholder = None
        self.figure = fig
        self.canvas = FigureCanvasQTAgg(fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self._layout.addWidget(self.toolbar)
        self._layout.addWidget(self.canvas, 1)
        self.canvas.draw_idle()


class Saxs2Mixin:
    """Builds the Frames / Fit / Series figures tabs onto a SaxsWorkspace (the Fit tab via SaxsFitMixin)."""

    # ================================================================== construction
    def _build_saxs2_tabs(self) -> None:
        self._fr_records = []
        self._fr_pairing = None
        self._fr_results = []
        self._sr_fig = None
        self._build_frames_tab()
        self._build_fit_tab()
        self._build_series_tab()

    def _refresh_saxs2_combos(self) -> None:
        self._refresh_fit_lists()

    # ------------------------------------------------------------------ Frames
    def _build_frames_tab(self) -> None:
        tab = QWidget()
        root = QHBoxLayout(tab)
        splitter = QSplitter()
        root.addWidget(splitter)
        left = QWidget()
        left.setMaximumWidth(360)
        ll = QVBoxLayout(left)

        box = QGroupBox("Raw frames")
        g = QGridLayout(box)
        self.fr_roots_edit = QLineEdit()
        self.fr_roots_edit.setPlaceholderText("folder; folder; …")
        add_btn = QPushButton("Add folder…")
        add_btn.clicked.connect(self._fr_add_root)
        g.addWidget(QLabel("Roots"), 0, 0)
        g.addWidget(self.fr_roots_edit, 0, 1)
        g.addWidget(add_btn, 0, 2)
        self.fr_empty_edit = QLineEdit(r"(?i)empty")
        self.fr_calib_edit = QLineEdit(r"(?i)agbeh")
        self.fr_exclude_edit = QLineEdit()
        self.fr_wavelength_edit = QLineEdit()
        self.fr_wavelength_edit.setPlaceholderText("Å, if the header has none")
        for r_, (lab, w) in enumerate((("empty regex", self.fr_empty_edit), ("calibrant regex", self.fr_calib_edit),
                                       ("exclude regex", self.fr_exclude_edit), ("wavelength", self.fr_wavelength_edit)), start=1):
            g.addWidget(QLabel(lab), r_, 0)
            g.addWidget(w, r_, 1, 1, 2)
        self.fr_name_combo = QComboBox()
        self.fr_name_combo.addItems(["stem", "comment"])
        self.fr_pairing_combo = QComboBox()
        self.fr_pairing_combo.addItems(["same_folder", "nearest_time"])
        g.addWidget(QLabel("name from"), 5, 0)
        g.addWidget(self.fr_name_combo, 5, 1)
        g.addWidget(QLabel("pairing"), 6, 0)
        g.addWidget(self.fr_pairing_combo, 6, 1)
        self.fr_scan_btn = QPushButton("Scan frames")
        self.fr_scan_btn.setObjectName("Primary")
        self.fr_scan_btn.clicked.connect(self.fr_scan)
        g.addWidget(self.fr_scan_btn, 7, 0, 1, 3)
        ll.addWidget(box)

        box2 = QGroupBox("Reduction")
        g2 = QGridLayout(box2)
        self.fr_samples_edit = QLineEdit()
        self.fr_samples_edit.setPlaceholderText("samples.csv (mu_mm or composition + density)")
        b_samples = QPushButton("…")
        b_samples.setMaximumWidth(28)
        b_samples.clicked.connect(lambda: self._pick_file(self.fr_samples_edit, "Samples table", "CSV (*.csv)"))
        g2.addWidget(QLabel("samples"), 0, 0)
        g2.addWidget(self.fr_samples_edit, 0, 1)
        g2.addWidget(b_samples, 0, 2)
        self.fr_outdir_edit = QLineEdit()
        self.fr_outdir_edit.setPlaceholderText("optional: folder for the .dat curves")
        b_out = QPushButton("…")
        b_out.setMaximumWidth(28)
        b_out.clicked.connect(lambda: self._pick_dir(self.fr_outdir_edit))
        g2.addWidget(QLabel("output"), 1, 0)
        g2.addWidget(self.fr_outdir_edit, 1, 1)
        g2.addWidget(b_out, 1, 2)
        self.fr_axis_edit = QLineEdit("20")
        self.fr_bar_dx_edit = QLineEdit("40")
        self.fr_bar_dy_edit = QLineEdit("14")
        self.fr_foot_edit = QLineEdit("16")
        self.fr_refine_check = QCheckBox("refine the beam centre from the direct beam")
        self.fr_refine_check.setChecked(True)
        row = QHBoxLayout()
        for lab, w in (("axis excl. (°)", self.fr_axis_edit), ("bar |dx|", self.fr_bar_dx_edit), ("|dy| px", self.fr_bar_dy_edit),
                       ("footprint px", self.fr_foot_edit)):
            row.addWidget(QLabel(lab))
            w.setMaximumWidth(40)
            row.addWidget(w)
        g2.addLayout(row, 2, 0, 1, 3)
        g2.addWidget(self.fr_refine_check, 3, 0, 1, 3)
        self.fr_reduce_btn = QPushButton("Reduce all pairs")
        self.fr_reduce_btn.setObjectName("Primary")
        self.fr_reduce_btn.setEnabled(False)
        self.fr_reduce_btn.clicked.connect(self.fr_reduce)
        g2.addWidget(self.fr_reduce_btn, 4, 0, 1, 3)
        self.fr_export_btn = QPushButton("Export summary CSV…")
        self.fr_export_btn.clicked.connect(self.fr_export_summary)
        g2.addWidget(self.fr_export_btn, 5, 0, 1, 3)
        self.fr_sasview_btn = QPushButton("Export for SasView…")
        self.fr_sasview_btn.setToolTip("Three-column Q / I / dI files (usable range, per mm when mu is known) that SasView "
                                       "and other fit engines open; the output folder above also gets a sasview\\ copy.")
        self.fr_sasview_btn.clicked.connect(self.fr_export_sasview)
        g2.addWidget(self.fr_sasview_btn, 6, 0, 1, 3)
        ll.addWidget(box2)
        self.fr_progress = QProgressBar()
        self.fr_progress.setRange(0, 100)
        self.fr_progress.setValue(0)
        ll.addWidget(self.fr_progress)
        self.fr_status = QLabel("Choose the folders of raw 2D frames and scan them.")
        self.fr_status.setWordWrap(True)
        ll.addWidget(self.fr_status)
        ll.addStretch(1)
        splitter.addWidget(left)

        right = QSplitter(Qt.Vertical)
        self.fr_inventory_table = QTableWidget(0, 11)
        self.fr_inventory_table.setHorizontalHeaderLabels(["frame", "role", "session", "config", "exposure (s)", "centre offset (px)",
                                                           "centre source", "beam (cps)", "lobe q (1/Å)", "streak H", "saturated"])
        self.fr_inventory_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.fr_inventory_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.fr_inventory_table.itemSelectionChanged.connect(self._fr_show_diagnostics)
        right.addWidget(self.fr_inventory_table)
        self.fr_diag_plot = PlotWidget(figsize=(7, 3.2))
        right.addWidget(self.fr_diag_plot)
        self.fr_results_table = QTableWidget(0, 10)
        self.fr_results_table.setHorizontalHeaderLabels(["sample", "empty", "T", "T4/T", "mu t", "t_eff (mm)", "class", "q min use",
                                                         "q max use", "centre"])
        self.fr_results_table.setEditTriggers(QTableWidget.NoEditTriggers)
        right.addWidget(self.fr_results_table)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        self.tabs.addTab(tab, "Frames")

    # ------------------------------------------------------------------ Series figures
    def _build_series_tab(self) -> None:
        tab = QWidget()
        root = QHBoxLayout(tab)
        splitter = QSplitter()
        root.addWidget(splitter)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.addWidget(QLabel("Series table (one row per sample; grouping, order, colours, x-values)"))
        self.sr_table = QTableWidget(0, len(SERIES_COLUMNS))
        self.sr_table.setHorizontalHeaderLabels(SERIES_COLUMNS)
        ll.addWidget(self.sr_table, 1)
        row = QHBoxLayout()
        for lab, fn in (("Import CSV…", self.sr_import_csv), ("Save CSV…", self.sr_save_csv), ("Fill from curves", self.sr_fill_from_curves),
                        ("Add row", lambda: self.sr_table.insertRow(self.sr_table.rowCount())), ("Remove row", self._sr_remove_row)):
            b = QPushButton(lab)
            b.clicked.connect(fn)
            row.addWidget(b)
        ll.addLayout(row)
        box = QGroupBox("Figures")
        g = QGridLayout(box)
        self.sr_grid_btn = QPushButton("Grid of curves")
        self.sr_grid_btn.clicked.connect(self.sr_grid_figure)
        self.sr_fits_btn = QPushButton("Fit panels (group)")
        self.sr_fits_btn.clicked.connect(self.sr_fit_figure)
        self.sr_iq4_check = QCheckBox("fit panels in I q⁴")
        self.sr_group_combo = QComboBox()
        self.sr_group_combo.setEditable(True)
        self.sr_group_combo.setPlaceholderText("group (blank = all)")
        self.sr_trends_btn = QPushButton("Trends vs x")
        self.sr_trends_btn.clicked.connect(self.sr_trends_figure)
        self.sr_frame_btn = QPushButton("2D frame figure…")
        self.sr_frame_btn.clicked.connect(self.sr_frame_figure)
        g.addWidget(self.sr_grid_btn, 0, 0)
        g.addWidget(self.sr_trends_btn, 0, 1)
        g.addWidget(self.sr_fits_btn, 1, 0)
        g.addWidget(self.sr_group_combo, 1, 1)
        g.addWidget(self.sr_iq4_check, 2, 0, 1, 2)
        g.addWidget(self.sr_frame_btn, 6, 0, 1, 2)
        self.sr_shift_edit = QLineEdit("1.6")
        self.sr_shift_edit.setMaximumWidth(50)
        self.sr_xlabel_edit = QLineEdit("x")
        self.sr_ylabel_edit = QLineEdit("Intensity (normalized, shifted)")
        g.addWidget(QLabel("shift (decades)"), 3, 0)
        g.addWidget(self.sr_shift_edit, 3, 1)
        g.addWidget(QLabel("x label (trends)"), 4, 0)
        g.addWidget(self.sr_xlabel_edit, 4, 1)
        g.addWidget(QLabel("y label (grid)"), 5, 0)
        g.addWidget(self.sr_ylabel_edit, 5, 1)
        ll.addWidget(box)
        box2 = QGroupBox("Export")
        g2 = QGridLayout(box2)
        self.sr_fmt_checks = {}
        for i, fmt in enumerate(("pdf", "svg", "png")):
            c = QCheckBox(fmt)
            c.setChecked(True)
            self.sr_fmt_checks[fmt] = c
            g2.addWidget(c, 0, i)
        self.sr_dpi_edit = QLineEdit("600")
        self.sr_dpi_edit.setMaximumWidth(50)
        g2.addWidget(QLabel("dpi"), 0, 3)
        g2.addWidget(self.sr_dpi_edit, 0, 4)
        self.sr_export_fig_btn = QPushButton("Export figure…")
        self.sr_export_fig_btn.clicked.connect(self.sr_export_figure)
        g2.addWidget(self.sr_export_fig_btn, 1, 0, 1, 3)
        self.sr_export_tables_btn = QPushButton("Export tables (xlsx / csv / md)…")
        self.sr_export_tables_btn.clicked.connect(self.sr_export_tables)
        g2.addWidget(self.sr_export_tables_btn, 1, 3, 1, 2)
        ll.addWidget(box2)
        self.sr_status = QLabel("Import a series table or fill it from the loaded curves.")
        self.sr_status.setWordWrap(True)
        ll.addWidget(self.sr_status)
        splitter.addWidget(left)
        self.sr_preview = FigurePreview()
        splitter.addWidget(self.sr_preview)
        splitter.setStretchFactor(1, 1)
        self.tabs.addTab(tab, "Series figures")

    # ================================================================== helpers
    def _pick_file(self, edit: QLineEdit, title: str, filt: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, title, "", filt)
        if path:
            edit.setText(path)

    def _pick_dir(self, edit: QLineEdit) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose a folder", "")
        if path:
            edit.setText(path)

    def _fr_add_root(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Folder of raw frames", "")
        if path:
            cur = self.fr_roots_edit.text().strip()
            self.fr_roots_edit.setText(f"{cur}; {path}" if cur else path)

    def _fr_roots(self) -> List[str]:
        return [p.strip() for p in self.fr_roots_edit.text().replace("\n", ";").split(";") if p.strip()]

    def _set_progress(self, bar: QProgressBar, i: int, n: int, msg: str, label: Optional[QLabel] = None) -> None:
        bar.setValue(int(100 * i / n) if n else 0)
        if label is not None:
            label.setText(msg)

    # ================================================================== Frames actions
    def fr_scan(self) -> None:
        from core.qt_worker import run_in_thread
        from saxs.core import frames as fr

        roots = self._fr_roots()
        if not roots:
            QMessageBox.warning(self, "Frames", "Give at least one folder of raw frames.")
            return
        kw = dict(empty_pattern=self.fr_empty_edit.text() or r"(?i)empty", calibrant_pattern=self.fr_calib_edit.text() or r"(?i)agbeh",
                  exclude_pattern=self.fr_exclude_edit.text() or None, name_from=self.fr_name_combo.currentText(),
                  default_wavelength_A=_to_float(self.fr_wavelength_edit.text()))
        rule = self.fr_pairing_combo.currentText()
        self.fr_scan_btn.setEnabled(False)
        self.fr_progress.setValue(0)

        def job(progress):
            errors = []
            records = fr.scan_frames(roots, diagnostics=True, progress=progress, on_error=lambda p, e: errors.append(f"{p}: {e}"), **kw)
            return records, fr.pair_frames(records, rule=rule), errors

        run_in_thread(job, self._fr_on_scanned, self._fr_on_error, on_progress=lambda i, n, m: self._set_progress(self.fr_progress, i, n, m, self.fr_status))

    def _fr_on_error(self, traceback_text: str) -> None:
        self.fr_scan_btn.setEnabled(True)
        self.fr_reduce_btn.setEnabled(bool(self._fr_pairing and self._fr_pairing.pairs))
        QMessageBox.critical(self, "Frames", traceback_text)

    def _fr_on_scanned(self, payload) -> None:
        records, pairing, errors = payload
        self._fr_records, self._fr_pairing = records, pairing
        self.fr_scan_btn.setEnabled(True)
        self.fr_reduce_btn.setEnabled(bool(pairing.pairs))
        t = self.fr_inventory_table
        t.setRowCount(len(records))
        for r, rec in enumerate(records):
            off = "" if rec.centre_offset_px is None else f"{rec.centre_offset_px[0]:+.1f}, {rec.centre_offset_px[1]:+.1f}"
            for c, text in enumerate((rec.rel, rec.role, rec.session, rec.configuration, _fmt(rec.exposure_s), off, rec.centre_source,
                                      _fmt(rec.direct_beam_cps, 3), _fmt(rec.lobe_q, 3), _fmt(rec.streak_h, 3), _fmt(rec.saturated))):
                t.setItem(r, c, _item(text))
        t.resizeColumnsToContents()
        msg = f"{len(records)} frame(s): {len(pairing.pairs)} pair(s), {len(pairing.unpaired)} unpaired sample(s)."
        if errors:
            msg += f" {len(errors)} frame(s) skipped."
        self.fr_status.setText(msg)
        self.fr_progress.setValue(100)

    def _fr_show_diagnostics(self) -> None:
        rows = sorted({i.row() for i in self.fr_inventory_table.selectedIndexes()})
        if not rows or rows[0] >= len(self._fr_records):
            return
        rec = self._fr_records[rows[0]]
        self.fr_diag_plot.request_redraw(self._fr_render_diagnostics, rec)

    def _fr_render_diagnostics(self, rec) -> None:
        from saxs.core import edf_sectors as es

        fig = self.fr_diag_plot.figure
        fig.clf()
        try:
            data, _h, geom = es.read_frame(rec.path, rec.wavelength_A)
        except Exception as exc:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f"cannot read {rec.rel}: {exc}", ha="center", va="center", transform=ax.transAxes)
            self.fr_diag_plot.canvas.draw_idle()
            return
        g = rec.geometry(refined=True)
        ax1 = fig.add_subplot(121)
        w = 45
        x0, y0 = int(round(g.cx)), int(round(g.cy))
        crop = data[max(y0 - w, 0):y0 + w, max(x0 - w, 0):x0 + w] / max(g.exposure_s, 1e-9)
        ax1.imshow(np.where(crop > 0, np.log10(np.maximum(crop, 1e-4)), np.nan), origin="lower", cmap="viridis",
                   extent=[-w, w, -w, w])
        ax1.set_title(f"{rec.name} ({rec.centre_source})", fontsize=8)
        ax1.set_xlabel("x - centre (px)")
        ax2 = fig.add_subplot(122)
        an = es.sector_anisotropy(data, g)
        ax2.loglog(an.q[2:], an.horizontal[2:], color="crimson", lw=1, label="horizontal ±25°")
        ax2.loglog(an.q[2:], an.vertical[2:], color="steelblue", lw=1, label="vertical ±25°")
        ax2.loglog(an.q[2:], an.diagonal[2:], color="0.4", lw=0.8, ls="--", label="diagonal")
        if an.lobe_q:
            ax2.axvline(an.lobe_q, color="k", ls=":", lw=0.8)
        ax2.set_xlabel("q (Å⁻¹)")
        ax2.set_ylabel("counts / s / pixel")
        ax2.legend(fontsize=7)
        ax2.grid(alpha=0.25, which="both")
        fig.tight_layout()
        self.fr_diag_plot.canvas.draw_idle()

    def fr_reduce(self) -> None:
        from core.qt_worker import run_in_thread
        from saxs.core import sector_batch as sb

        if not self._fr_pairing or not self._fr_pairing.pairs:
            QMessageBox.warning(self, "Frames", "Scan frames first: no pair to reduce.")
            return
        opts = sb.ReductionOptions(axis_exclusion_deg=_to_float(self.fr_axis_edit.text(), 20.0) or 20.0,
                                   bar_rect_px=(_to_float(self.fr_bar_dx_edit.text(), 40.0) or 40.0, _to_float(self.fr_bar_dy_edit.text(), 14.0) or 14.0),
                                   footprint_r_px=_to_float(self.fr_foot_edit.text(), 16.0) or 16.0, refine_centre=self.fr_refine_check.isChecked())
        samples = self.fr_samples_edit.text().strip()
        mu = None
        if samples:
            try:
                mu = sb.MuSource.from_samples_table(samples)
            except Exception as exc:
                QMessageBox.warning(self, "Frames", f"Samples table not usable: {exc}")
                return
        out_dir = self.fr_outdir_edit.text().strip() or None
        pairs = list(self._fr_pairing.pairs)
        self.fr_reduce_btn.setEnabled(False)
        self.fr_progress.setValue(0)

        sasview_dir = os.path.join(out_dir, "sasview") if out_dir else None

        def job(progress):
            errors = []
            res = sb.reduce_pairs(pairs, opts, mu, out_dir=out_dir, progress=progress, on_error=lambda p, e: errors.append(f"{p.sample.name}: {e}"),
                                  sasview_dir=sasview_dir)
            return res, errors

        run_in_thread(job, self._fr_on_reduced, self._fr_on_error, on_progress=lambda i, n, m: self._set_progress(self.fr_progress, i, n, m, self.fr_status))

    def _fr_on_reduced(self, payload) -> None:
        results, errors = payload
        self._fr_results = results
        self.fr_reduce_btn.setEnabled(True)
        t = self.fr_results_table
        t.setRowCount(len(results))
        for r, res in enumerate(results):
            for c, text in enumerate((res.name, os.path.basename(res.empty_path), _fmt(res.transmission.T, 3), _fmt(res.heterogeneity, 3),
                                      _fmt(res.mu_t, 3), _fmt(res.t_eff_mm, 3), res.loading_class, _fmt(res.q_min_use, 3), _fmt(res.q_max_use, 3),
                                      res.centre_source)):
                t.setItem(r, c, _item(text))
            existing = [c for c in self.curves if c.name == res.name]
            for c in existing:
                self.curves.remove(c)
            self.add_curve(res.usable_curve())
        t.resizeColumnsToContents()
        classes = {}
        for res in results:
            classes[res.loading_class] = classes.get(res.loading_class, 0) + 1
        msg = f"{len(results)} curve(s) reduced and added to the workspace: " + ", ".join(f"{v} {k}" for k, v in classes.items()) + "."
        if errors:
            msg += f" {len(errors)} failed."
        self.fr_status.setText(msg)
        self.fr_progress.setValue(100)
        if errors:
            QMessageBox.warning(self, "Frames", "\n".join(errors))

    def fr_export_sasview(self) -> None:
        from saxs.core import sector_batch as sb

        if not self._fr_results:
            QMessageBox.information(self, "Frames", "Reduce pairs first.")
            return
        folder = QFileDialog.getExistingDirectory(self, "Folder for the SasView-ready curves", "")
        if not folder:
            return
        paths = sb.export_sasview(self._fr_results, folder)
        self.fr_status.setText(f"{len(paths)} SasView-ready curve(s) (Q, I, dI; usable range) -> {folder}")

    def fr_export_summary(self) -> None:
        from saxs.core import sector_batch as sb

        if not self._fr_results:
            QMessageBox.information(self, "Frames", "Reduce pairs first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export the reduction summary", "reduction_summary.csv", "CSV (*.csv)")
        if not path:
            return
        sb.summary_table(self._fr_results).to_csv(path, index=False, encoding="utf-8-sig")
        self.fr_status.setText(f"Summary exported to {path}")

    # ================================================================== Series actions
    def _sr_dataframe(self) -> pd.DataFrame:
        rows = []
        for r in range(self.sr_table.rowCount()):
            row = {}
            for c, col in enumerate(SERIES_COLUMNS):
                it = self.sr_table.item(r, c)
                row[col] = it.text().strip() if it is not None else ""
            if row["sample"]:
                rows.append(row)
        df = pd.DataFrame(rows, columns=SERIES_COLUMNS)
        for col in ("panel_row", "panel_col", "stack_index", "x_value", "fit_order"):
            df[col] = pd.to_numeric(df[col], errors="coerce")          # blank cells become NaN
        for col in ("panel_row", "panel_col", "stack_index"):
            df[col] = df[col].fillna(0)
        df["quantified"] = df["quantified"].replace("", "yes")
        return df

    def _sr_fill(self, df: pd.DataFrame) -> None:
        self.sr_table.setRowCount(len(df))
        for r, (_, row) in enumerate(df.iterrows()):
            for c, col in enumerate(SERIES_COLUMNS):
                v = row.get(col, "")
                text = "" if (v is None or (isinstance(v, float) and np.isnan(v))) else (("yes" if v else "no") if isinstance(v, (bool, np.bool_)) else str(v))
                self.sr_table.setItem(r, c, QTableWidgetItem(text))
        self.sr_table.resizeColumnsToContents()
        groups = sorted({str(g) for g in df.get("group", pd.Series(dtype=str)) if str(g) not in ("", "nan")})
        self.sr_group_combo.clear()
        self.sr_group_combo.addItems([""] + groups)

    def _sr_series_table(self):
        from saxs.core.series_figures import SeriesTable

        df = self._sr_dataframe()
        if df.empty:
            raise ValueError("the series table is empty")
        return SeriesTable.from_dataframe(df)

    def _sr_remove_row(self) -> None:
        rows = sorted({i.row() for i in self.sr_table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.sr_table.removeRow(r)

    def sr_import_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import a series table", "", "CSV (*.csv)")
        if not path:
            return
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except Exception as exc:
            QMessageBox.critical(self, "Series figures", str(exc))
            return
        self._sr_fill(df)
        self.sr_status.setText(f"{len(df)} row(s) imported from {os.path.basename(path)}.")

    def sr_save_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save the series table", "series.csv", "CSV (*.csv)")
        if path:
            self._sr_dataframe().to_csv(path, index=False, encoding="utf-8-sig")

    def sr_fill_from_curves(self) -> None:
        if not self.curves:
            QMessageBox.information(self, "Series figures", "Load or reduce curves first.")
            return
        df = pd.DataFrame([{"sample": c.name, "panel_row": 0, "panel_col": 0, "stack_index": i, "label": c.name, "quantified": "yes",
                            "group": "all"} for i, c in enumerate(self.curves)], columns=SERIES_COLUMNS)
        self._sr_fill(df)
        self.sr_status.setText(f"{len(df)} row(s) from the loaded curves (edit panels, labels, x-values).")

    def _sr_show(self, fig, what: str) -> None:
        self._sr_fig = fig
        self.sr_preview.set_figure(fig)
        self.sr_status.setText(f"{what} built ({len(fig.get_axes())} axes). Export it with the buttons below.")

    def sr_grid_figure(self) -> None:
        from saxs.core.series_figures import grid_figure

        try:
            table = self._sr_series_table()
            curves = {c.name: c for c in self.curves}
            fig = grid_figure(curves, table, shift_decades=_to_float(self.sr_shift_edit.text(), 1.6) or 1.6, ylabel=self.sr_ylabel_edit.text() or "Intensity")
        except Exception as exc:
            QMessageBox.critical(self, "Series figures", str(exc))
            return
        self._sr_show(fig, "Grid figure")

    def sr_fit_figure(self) -> None:
        from saxs.core.series_figures import fit_panels_figure

        res = getattr(self, "_fit_result", None)
        if res is None:
            QMessageBox.information(self, "Series figures", "Fit curves first (Fit tab).")
            return
        try:
            table = self._sr_series_table()
            group = self.sr_group_combo.currentText().strip()
            samples = table.samples_in_group(group) if group else table.samples
            fig = fit_panels_figure(res, [s for s in samples if s in res.curves], table, ncol=3, title=group or None,
                                    iq4=self.sr_iq4_check.isChecked())
        except Exception as exc:
            QMessageBox.critical(self, "Series figures", str(exc))
            return
        self._sr_show(fig, "Fit panels")

    def sr_trends_figure(self) -> None:
        from saxs.core.descriptors import describe_batch
        from saxs.core.fit_workflow import summary_table
        from saxs.core.series_figures import trends_figure

        try:
            table = self._sr_series_table()
            curves = [c for c in self.curves if table.has(c.name)]
            _res, desc = describe_batch(curves) if curves else (None, None)
            res = getattr(self, "_fit_result", None)
            fits_df = summary_table(res) if res is not None else None
            fig = trends_figure(table, desc, fits_df, x_label=self.sr_xlabel_edit.text() or "x")
        except Exception as exc:
            QMessageBox.critical(self, "Series figures", str(exc))
            return
        self._sr_show(fig, "Trends figure")

    def sr_frame_figure(self) -> None:
        from saxs.core.series_figures import frame_figure

        sample, _ = QFileDialog.getOpenFileName(self, "Sample frame", "", "EDF (*.edf);;All files (*.*)")
        if not sample:
            return
        empty, _ = QFileDialog.getOpenFileName(self, "Empty frame", "", "EDF (*.edf);;All files (*.*)")
        if not empty:
            return
        try:
            fig = frame_figure(sample, empty, titles=(os.path.splitext(os.path.basename(sample))[0], "empty"),
                               default_wavelength_A=_to_float(self.fr_wavelength_edit.text()))
        except Exception as exc:
            QMessageBox.critical(self, "Series figures", str(exc))
            return
        self._sr_show(fig, "Frame figure")

    def sr_export_figure(self) -> None:
        from saxs.core.series_figures import export_figure

        if self._sr_fig is None:
            QMessageBox.information(self, "Series figures", "Build a figure first.")
            return
        formats = tuple(f for f, c in self.sr_fmt_checks.items() if c.isChecked())
        if not formats:
            QMessageBox.warning(self, "Series figures", "Tick at least one format.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export the figure (base name, extensions added)", "figure", "All files (*.*)")
        if not path:
            return
        base = os.path.splitext(path)[0] if os.path.splitext(path)[1].lower().lstrip(".") in ("pdf", "svg", "png") else path
        paths = export_figure(self._sr_fig, base, formats, int(_to_float(self.sr_dpi_edit.text(), 600.0) or 600))
        self.sr_status.setText("Exported: " + ", ".join(os.path.basename(p) for p in paths))

    def sr_export_tables(self) -> None:
        from saxs.core import tables as tb
        from saxs.core.descriptors import describe_batch

        res = getattr(self, "_fit_result", None)
        if res is None:
            QMessageBox.information(self, "Series figures", "Fit curves first (Fit tab).")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export the tables (base name)", "tables", "All files (*.*)")
        if not path:
            return
        base = os.path.splitext(path)[0] if os.path.splitext(path)[1].lower().lstrip(".") in ("xlsx", "csv", "md") else path
        try:
            table = self._sr_series_table()
            order = [n for n in table.samples if n in res.curves] + [n for n in res.curves if not table.has(n)]
            res.curves = {n: res.curves[n] for n in order}
            extra = ({n: {"x": table.get(n, "x_value")} for n in order if table.has(n)}
                     if table.df["x_value"].notna().any() else None)
            curves = [c for c in self.curves if c.name in set(order)]
            desc = describe_batch(curves)[1] if curves else None
            paths = tb.write_workflow_tables(res, base, extra=extra, descriptors=desc)
        except Exception as exc:
            QMessageBox.critical(self, "Series figures", str(exc))
            return
        self.sr_status.setText("Tables exported: " + ", ".join(os.path.basename(p) for p in paths.values()))
