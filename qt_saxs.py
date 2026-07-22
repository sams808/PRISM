"""
qt_saxs.py — the SAXS/WAXS workspace: PRISM's port of the author's 'pomme'
suite (saxs_core engine). Tabs: Curves (Xenocs-style 1D ASCII import,
log-log view), Reduction (empty/background subtraction with manual or auto
scaling), Analysis (Guinier / generalized Porod / pseudo-Bragg peak with
auto region detection), WAXS (pseudo-Voigt multi-peak fit, d-spacings,
crystallinity index). Model fitting via sasmodels stays in the pomme repo
(optional heavy dependency); publication plotting lives in Figures.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QMessageBox, QPlainTextEdit, QPushButton, QSplitter, QTableWidget,
    QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from qt_models import Spectrum, SpectrumLibrary
from qt_widgets import PlotWidget
from saxs_core.analysis import (
    auto_detect_guinier_region, auto_detect_peak_window, auto_detect_porod_region,
    fit_guinier, fit_porod_general, fit_pseudo_bragg_peak,
)
from saxs_core.chemistry import CapillaryConfig, SamplePhysicsConfig
from saxs_core.composite_batch import BatchItem, batch_to_csv_rows, run_batch, write_batch_csv
from saxs_core.composite_fit import PRESETS, CompositeModel, build_composite, build_preset
from saxs_core.composite_staged import _build_derived, apply_hygiene, fit_staged, propose_windows
from saxs_core.curve import Curve
from saxs_core.loader import load_curve
from saxs_core.reduction import CorrectionSettings, correct_sample
from saxs_core.waxs import auto_find_peaks, fit_waxs_peaks

COMPOSITE_PRESET_CHOICES = ["Auto (BIC ladder)"] + list(PRESETS.keys())
COMPOSITE_WINDOW_KEYS = ["W_peak", "W_hiq", "W_loq"]


def _model_for_preset_name(name: str) -> CompositeModel:
    if name in PRESETS:
        return build_preset(name)
    return build_composite(name.split("+"))


def _to_float(text: str, default: Optional[float] = None) -> Optional[float]:
    try:
        return float((text or "").strip())
    except (TypeError, ValueError):
        return default


class SaxsWorkspace(QWidget):
    def __init__(self, parent: Optional[QWidget] = None, library: Optional[SpectrumLibrary] = None,
                 on_derived_added=None):
        super().__init__(parent)
        self.library = library if library is not None else SpectrumLibrary()
        self.on_derived_added = on_derived_added
        self.curves: List[Curve] = []
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
        ll = QVBoxLayout(left)
        import_btn = QPushButton("Import curve(s)…")
        import_btn.clicked.connect(self.import_curves)
        ll.addWidget(import_btn)
        ll.addWidget(QLabel("Curves (q, I)"))
        self.curve_list = QListWidget()
        self.curve_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.curve_list.itemSelectionChanged.connect(lambda: self.plot.request_redraw(self.render_curves))
        ll.addWidget(self.curve_list, 1)
        send_btn = QPushButton("Send selected to Library")
        send_btn.clicked.connect(self.send_to_library)
        ll.addWidget(send_btn)
        self.status_label = QLabel("Import Xenocs-style 1D ASCII exports (.dat/.txt) to begin.")
        self.status_label.setWordWrap(True)
        ll.addWidget(self.status_label)
        splitter.addWidget(left)

        right = QWidget()
        right.setObjectName("Card")
        rl = QVBoxLayout(right)
        self.tabs = QTabWidget()

        curves_tab = QWidget()
        cl = QVBoxLayout(curves_tab)
        self.plot = PlotWidget(figsize=(7, 5))
        cl.addWidget(self.plot)
        self.tabs.addTab(curves_tab, "Curves")

        red_tab = QWidget()
        rl2 = QVBoxLayout(red_tab)
        row = QHBoxLayout()
        row.addWidget(QLabel("Sample"))
        self.red_sample_combo = QComboBox()
        row.addWidget(self.red_sample_combo, 1)
        row.addWidget(QLabel("Empty"))
        self.red_empty_combo = QComboBox()
        row.addWidget(self.red_empty_combo, 1)
        rl2.addLayout(row)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Scale mode"))
        self.red_mode_combo = QComboBox()
        self.red_mode_combo.addItems(["auto", "manual", "transmission", "physics"])
        row2.addWidget(self.red_mode_combo)
        row2.addWidget(QLabel("manual factor"))
        self.red_factor_edit = QLineEdit("1.0")
        self.red_factor_edit.setMaximumWidth(60)
        row2.addWidget(self.red_factor_edit)
        row2.addWidget(QLabel("E (eV)"))
        self.red_energy_edit = QLineEdit("8047")
        self.red_energy_edit.setMaximumWidth(60)
        row2.addWidget(self.red_energy_edit)
        reduce_btn = QPushButton("Subtract empty → corrected curve")
        reduce_btn.setObjectName("Primary")
        reduce_btn.clicked.connect(self.run_reduction)
        row2.addWidget(reduce_btn)
        row2.addStretch(1)
        rl2.addLayout(row2)
        self.red_plot = PlotWidget(figsize=(7, 4.4))
        rl2.addWidget(self.red_plot, 1)
        self.tabs.addTab(red_tab, "Reduction")

        ana_tab = QWidget()
        al = QVBoxLayout(ana_tab)
        arow = QHBoxLayout()
        arow.addWidget(QLabel("Curve"))
        self.ana_combo = QComboBox()
        arow.addWidget(self.ana_combo, 1)
        arow.addWidget(QLabel("q range"))
        self.ana_qmin_edit = QLineEdit()
        self.ana_qmin_edit.setMaximumWidth(65)
        self.ana_qmax_edit = QLineEdit()
        self.ana_qmax_edit.setMaximumWidth(65)
        arow.addWidget(self.ana_qmin_edit)
        arow.addWidget(self.ana_qmax_edit)
        for label, fn in (("Guinier", self.run_guinier), ("Porod", self.run_porod), ("Peak", self.run_peak)):
            btn = QPushButton(label)
            btn.clicked.connect(fn)
            arow.addWidget(btn)
        al.addLayout(arow)
        self.ana_plot = PlotWidget(figsize=(7, 3.8))
        al.addWidget(self.ana_plot, 2)
        self.ana_report = QPlainTextEdit()
        self.ana_report.setReadOnly(True)
        self.ana_report.setMaximumHeight(120)
        al.addWidget(self.ana_report)
        self.tabs.addTab(ana_tab, "Analysis")

        waxs_tab = QWidget()
        wl = QVBoxLayout(waxs_tab)
        wrow = QHBoxLayout()
        wrow.addWidget(QLabel("Curve"))
        self.waxs_combo = QComboBox()
        wrow.addWidget(self.waxs_combo, 1)
        wrow.addWidget(QLabel("λ (Å)"))
        self.waxs_wl_edit = QLineEdit("1.5406")
        self.waxs_wl_edit.setMaximumWidth(60)
        wrow.addWidget(self.waxs_wl_edit)
        waxs_btn = QPushButton("Auto-find + fit peaks")
        waxs_btn.setObjectName("Primary")
        waxs_btn.clicked.connect(self.run_waxs)
        wrow.addWidget(waxs_btn)
        wrow.addStretch(1)
        wl.addLayout(wrow)
        self.waxs_plot = PlotWidget(figsize=(7, 3.8))
        wl.addWidget(self.waxs_plot, 2)
        self.waxs_report = QPlainTextEdit()
        self.waxs_report.setReadOnly(True)
        self.waxs_report.setMaximumHeight(120)
        wl.addWidget(self.waxs_report)
        self.tabs.addTab(waxs_tab, "WAXS")

        self._build_composite_tab()

        rl.addWidget(self.tabs)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)

    def _build_composite_tab(self) -> None:
        """Composite SAXS model fitting (spec-driven staged pipeline):
        window selectors, a preset picker (auto BIC-ladder or a manual
        composite), single-fit + batch, a component-overlay plot, and a
        report panel. General-purpose: batch here processes whichever
        curves the user selects, in list order — no sample-naming
        assumptions, matching every other saxs_core module in this app."""
        comp_tab = QWidget()
        comp_root = QHBoxLayout(comp_tab)
        comp_splitter = QSplitter()
        comp_root.addWidget(comp_splitter)

        comp_left = QWidget()
        comp_left.setMaximumWidth(320)
        cll = QVBoxLayout(comp_left)

        cll.addWidget(QLabel("Curve (single fit)"))
        self.comp_combo = QComboBox()
        self.comp_combo.currentTextChanged.connect(self.on_comp_curve_changed)
        cll.addWidget(self.comp_combo)

        cll.addWidget(QLabel("Preset"))
        self.comp_preset_combo = QComboBox()
        self.comp_preset_combo.addItems(COMPOSITE_PRESET_CHOICES)
        cll.addWidget(self.comp_preset_combo)

        ms_row = QHBoxLayout()
        ms_row.addWidget(QLabel("multistart_n"))
        self.comp_multistart_edit = QLineEdit("8")
        self.comp_multistart_edit.setMaximumWidth(50)
        ms_row.addWidget(self.comp_multistart_edit)
        ms_row.addStretch(1)
        cll.addLayout(ms_row)

        cll.addWidget(QLabel("Windows (q lo / q hi) — auto-proposed, editable"))
        self.comp_window_edits = {}
        for key in COMPOSITE_WINDOW_KEYS:
            row = QHBoxLayout()
            row.addWidget(QLabel(key))
            lo_edit = QLineEdit()
            lo_edit.setMaximumWidth(70)
            hi_edit = QLineEdit()
            hi_edit.setMaximumWidth(70)
            row.addWidget(lo_edit)
            row.addWidget(hi_edit)
            cll.addLayout(row)
            self.comp_window_edits[key] = (lo_edit, hi_edit)
        auto_win_btn = QPushButton("Auto-detect windows")
        auto_win_btn.clicked.connect(self.auto_detect_comp_windows)
        cll.addWidget(auto_win_btn)

        self._comp_fit_btn = QPushButton("Fit selected curve")
        self._comp_fit_btn.setObjectName("Primary")
        self._comp_fit_btn.clicked.connect(self.run_composite_fit)
        cll.addWidget(self._comp_fit_btn)

        cll.addWidget(QLabel("Batch (curves selected below)"))
        self.comp_batch_list = QListWidget()
        self.comp_batch_list.setSelectionMode(QListWidget.ExtendedSelection)
        cll.addWidget(self.comp_batch_list, 1)
        self._comp_batch_btn = QPushButton("Run batch")
        self._comp_batch_btn.clicked.connect(self.run_composite_batch)
        cll.addWidget(self._comp_batch_btn)
        export_btn = QPushButton("Export batch CSV…")
        export_btn.clicked.connect(self.export_composite_batch_csv)
        cll.addWidget(export_btn)

        comp_splitter.addWidget(comp_left)

        comp_right = QWidget()
        crl = QVBoxLayout(comp_right)
        self.comp_plot = PlotWidget(figsize=(7, 4.2))
        crl.addWidget(self.comp_plot, 2)
        self.comp_report = QPlainTextEdit()
        self.comp_report.setReadOnly(True)
        self.comp_report.setMaximumHeight(140)
        crl.addWidget(self.comp_report)
        crl.addWidget(QLabel("Batch results"))
        self.comp_batch_table = QTableWidget(0, 7)
        self.comp_batch_table.setHorizontalHeaderLabels(
            ["sample_id", "preset_chosen", "d (Å)", "xi (Å)", "fa", "chi2red", "d_TS/d_gauss"])
        self.comp_batch_table.setEditTriggers(QTableWidget.NoEditTriggers)
        crl.addWidget(self.comp_batch_table, 1)
        comp_splitter.addWidget(comp_right)
        comp_splitter.setStretchFactor(1, 1)

        self._comp_last = None
        self._comp_batch_result = None
        self._comp_batch_curves = {}
        self.tabs.addTab(comp_tab, "Composite fit")

    # ------------------------------------------------------------------
    def set_spectra(self, spectrum_ids: List[str]) -> None:  # nav-enter hook (self-contained workspace)
        pass

    def _refresh_combos(self) -> None:
        names = [c.name for c in self.curves]
        for combo in (self.red_sample_combo, self.red_empty_combo, self.ana_combo, self.waxs_combo,
                      self.comp_combo):
            current = combo.currentText()
            combo.clear()
            combo.addItems(names)
            idx = combo.findText(current)
            if idx >= 0:
                combo.setCurrentIndex(idx)
        self.curve_list.clear()
        for c in self.curves:
            self.curve_list.addItem(c.name)
        self.curve_list.selectAll()
        selected_batch = {i.text() for i in self.comp_batch_list.selectedItems()}
        self.comp_batch_list.clear()
        for c in self.curves:
            self.comp_batch_list.addItem(c.name)
        for i in range(self.comp_batch_list.count()):
            item = self.comp_batch_list.item(i)
            if item.text() in selected_batch:
                item.setSelected(True)

    def add_curve(self, curve: Curve) -> None:
        self.curves.append(curve)
        self._refresh_combos()

    def import_curves(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Import 1D scattering curves", "",
            "1D curves (*.dat *.txt *.csv);;All files (*.*)")
        errors = []
        for p in paths:
            try:
                self.add_curve(load_curve(p))
            except Exception as exc:
                errors.append(f"{p}: {exc}")
        self.status_label.setText(f"{len(self.curves)} curve(s) loaded.")
        if errors:
            QMessageBox.warning(self, "Import", "\n".join(errors))
        self.plot.request_redraw(self.render_curves)

    def send_to_library(self) -> None:
        rows = sorted({i.row() for i in self.curve_list.selectedIndexes()})
        new_ids = []
        for r in rows:
            c = self.curves[r]
            sp = Spectrum(id=Spectrum.new_id(), title=c.name, path=getattr(c, "path", "") or "",
                          kind="saxs_1d", x=np.asarray(c.q, float), y=np.asarray(c.intensity, float),
                          df=None, meta={"derived": "saxs_curve"}, status="derived")
            self.library.add(sp)
            new_ids.append(sp.id)
        if new_ids and self.on_derived_added is not None:
            self.on_derived_added(new_ids)
        self.status_label.setText(f"Sent {len(new_ids)} curve(s) to the Library.")

    def _curve_by_name(self, name: str) -> Optional[Curve]:
        for c in self.curves:
            if c.name == name:
                return c
        return None

    # ------------------------------------------------------------------
    def render_curves(self) -> None:
        fig = self.plot.figure
        fig.clf()
        ax = fig.add_subplot(111)
        rows = {i.row() for i in self.curve_list.selectedIndexes()}
        for r in sorted(rows):
            if 0 <= r < len(self.curves):
                c = self.curves[r]
                ax.loglog(c.q, np.clip(c.intensity, 1e-12, None), lw=1.0, label=c.name)
        if rows:
            ax.legend(fontsize=7)
        ax.set_xlabel("q (Å⁻¹)")
        ax.set_ylabel("I (a.u.)")
        ax.grid(alpha=0.25, which="both")
        fig.tight_layout()
        self.plot.canvas.draw_idle()

    def run_reduction(self) -> None:
        sample = self._curve_by_name(self.red_sample_combo.currentText())
        empty = self._curve_by_name(self.red_empty_combo.currentText())
        if sample is None or empty is None:
            QMessageBox.warning(self, "Reduction", "Import and pick a sample and an empty curve.")
            return
        settings = CorrectionSettings(scale_mode=self.red_mode_combo.currentText(),
                                      manual_scale=_to_float(self.red_factor_edit.text(), 1.0) or 1.0)
        try:
            result = correct_sample(sample, empty, SamplePhysicsConfig(), CapillaryConfig(),
                                    settings, _to_float(self.red_energy_edit.text(), 8047.0) or 8047.0)
        except Exception as exc:
            QMessageBox.critical(self, "Reduction error", str(exc))
            return
        corrected = Curve(q=result.q, intensity=result.corrected, sigma=result.sigma_corrected,
                          name=f"{sample.name}_corr")
        self.add_curve(corrected)
        fig = self.red_plot.figure
        fig.clf()
        ax = fig.add_subplot(111)
        ax.loglog(sample.q, np.clip(sample.intensity, 1e-12, None), lw=0.8, alpha=0.5, label="sample")
        ax.loglog(empty.q, np.clip(empty.intensity, 1e-12, None), lw=0.8, alpha=0.5, label="empty")
        ax.loglog(corrected.q, np.clip(corrected.intensity, 1e-12, None), lw=1.4, color="black",
                  label=f"corrected (scale {result.scale_factor:.4g})")
        ax.legend(fontsize=8)
        ax.set_xlabel("q (Å⁻¹)")
        ax.set_ylabel("I (a.u.)")
        ax.grid(alpha=0.25, which="both")
        fig.tight_layout()
        self.red_plot.canvas.draw_idle()
        self.status_label.setText(f"Reduction done (scale factor {result.scale_factor:.5g}).")

    # ------------------------------------------------------------------
    def _ana_curve_and_range(self, auto_fn):
        c = self._curve_by_name(self.ana_combo.currentText())
        if c is None:
            raise ValueError("Import and pick a curve first.")
        q, intensity = np.asarray(c.q, float), np.asarray(c.intensity, float)
        qmin = _to_float(self.ana_qmin_edit.text())
        qmax = _to_float(self.ana_qmax_edit.text())
        if qmin is None or qmax is None or qmax <= qmin:
            qmin, qmax = auto_fn(q, intensity)
            self.ana_qmin_edit.setText(f"{qmin:.4g}")
            self.ana_qmax_edit.setText(f"{qmax:.4g}")
        return c, q, intensity, float(qmin), float(qmax)

    def _render_ana(self, c, extra=None) -> None:
        fig = self.ana_plot.figure
        fig.clf()
        ax = fig.add_subplot(111)
        ax.loglog(c.q, np.clip(c.intensity, 1e-12, None), lw=1.0, color="black", label=c.name)
        if extra is not None:
            xq, xi, xlabel = extra
            ax.loglog(xq, np.clip(xi, 1e-12, None), lw=1.6, color="crimson", label=xlabel)
        ax.legend(fontsize=8)
        ax.set_xlabel("q (Å⁻¹)")
        ax.set_ylabel("I")
        ax.grid(alpha=0.25, which="both")
        fig.tight_layout()
        self.ana_plot.canvas.draw_idle()

    def run_guinier(self) -> None:
        try:
            c, q, intensity, qmin, qmax = self._ana_curve_and_range(auto_detect_guinier_region)
            r = fit_guinier(q, intensity, qmin, qmax)
        except Exception as exc:
            QMessageBox.warning(self, "Guinier", str(exc))
            return
        self.ana_report.setPlainText(
            f"Guinier [{qmin:.4g}-{qmax:.4g}]: Rg = {r.Rg:.3g} Å (I0 = {r.I0:.4g}, R² = {r.r2:.4f}, "
            f"qRg max = {qmax * r.Rg:.2f})\n"
            f"Sphere-equivalent diameter ≈ {2 * r.Rg * np.sqrt(5.0 / 3.0):.3g} Å"
        )
        m = (q >= qmin) & (q <= qmax)
        self._render_ana(c, (q[m], r.I0 * np.exp(-(q[m] * r.Rg) ** 2 / 3.0), "Guinier fit"))

    def run_porod(self) -> None:
        try:
            c, q, intensity, qmin, qmax = self._ana_curve_and_range(auto_detect_porod_region)
            r = fit_porod_general(q, intensity, qmin, qmax)
        except Exception as exc:
            QMessageBox.warning(self, "Porod", str(exc))
            return
        self.ana_report.setPlainText(
            f"Generalized Porod [{qmin:.4g}-{qmax:.4g}]: slope m = {r.m:.3f} (A = {r.A:.4g}, "
            f"background = {r.B:.4g}, R²(log) = {r.r2_log:.4f})"
        )
        m = (q >= qmin) & (q <= qmax)
        self._render_ana(c, (q[m], r.A * q[m] ** (-r.m) + r.B, "Porod fit"))

    def run_peak(self) -> None:
        try:
            c, q, intensity, qmin, qmax = self._ana_curve_and_range(auto_detect_peak_window)
            r = fit_pseudo_bragg_peak(q, intensity, qmin, qmax)
        except Exception as exc:
            QMessageBox.warning(self, "Peak", str(exc))
            return
        self.ana_report.setPlainText(
            f"Pseudo-Bragg peak [{qmin:.4g}-{qmax:.4g}]: q0 = {r.q0:.4g} Å⁻¹ → d = {r.d_spacing:.3g} Å, "
            f"FWHM = {r.fwhm:.4g}, apparent correlation length ≈ {r.xi_app:.3g} Å, area = {r.area:.4g}"
        )
        m = (q >= qmin) & (q <= qmax)
        model = (r.amp * np.exp(-((q[m] - r.q0) ** 2) / (2.0 * r.sigma ** 2))
                 + r.baseline_c0 + r.baseline_c1 * q[m])
        self._render_ana(c, (q[m], model, "peak fit"))

    # ------------------------------------------------------------------
    def run_waxs(self) -> None:
        c = self._curve_by_name(self.waxs_combo.currentText())
        if c is None:
            QMessageBox.warning(self, "WAXS", "Import and pick a curve first.")
            return
        q, intensity = np.asarray(c.q, float), np.asarray(c.intensity, float)
        try:
            specs = auto_find_peaks(q, intensity)
            if not specs:
                QMessageBox.information(self, "WAXS", "No peaks found.")
                return
            result = fit_waxs_peaks(q, intensity, specs)
        except Exception as exc:
            QMessageBox.critical(self, "WAXS error", str(exc))
            return
        lines = [f"WAXS fit: {len(result.peaks)} peak(s), R² = {result.r2:.4f}"]
        for p in result.peaks:
            lines.append(f"  q = {p.center:.4g} Å⁻¹  d = {p.d_spacing:.3g} Å  FWHM = {p.fwhm:.4g}  area = {p.area:.4g}"
                         + ("  (amorphous)" if p.is_amorphous else ""))
        if result.crystallinity_index is not None:
            lines.append(f"Crystallinity index ≈ {100 * result.crystallinity_index:.1f}%")
        self.waxs_report.setPlainText("\n".join(lines))
        fig = self.waxs_plot.figure
        fig.clf()
        ax = fig.add_subplot(111)
        ax.plot(q, intensity, lw=0.9, color="black", label=c.name)
        m = (q >= result.xmin) & (q <= result.xmax)
        ax.plot(q[m][:len(result.total_fit)], result.total_fit, lw=1.4, color="crimson", label="fit")
        ax.legend(fontsize=8)
        ax.set_xlabel("q (Å⁻¹)")
        ax.set_ylabel("I")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        self.waxs_plot.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Composite fit (staged pipeline, spec §4)
    # ------------------------------------------------------------------
    def _fill_window_fields(self, windows) -> None:
        for key, (lo_edit, hi_edit) in self.comp_window_edits.items():
            if key in windows:
                lo, hi = windows[key]
                lo_edit.setText(f"{lo:.5g}")
                hi_edit.setText(f"{hi:.5g}")

    def _read_window_fields(self):
        windows = {}
        for key, (lo_edit, hi_edit) in self.comp_window_edits.items():
            lo = _to_float(lo_edit.text())
            hi = _to_float(hi_edit.text())
            if lo is not None and hi is not None and hi > lo:
                windows[key] = (lo, hi)
        return windows or None

    def auto_detect_comp_windows(self) -> None:
        c = self._curve_by_name(self.comp_combo.currentText())
        if c is None:
            QMessageBox.warning(self, "Composite fit", "Import and pick a curve first.")
            return
        try:
            windows = propose_windows(np.asarray(c.q, float), np.asarray(c.intensity, float))
        except Exception as exc:
            QMessageBox.critical(self, "Composite fit", str(exc))
            return
        self._fill_window_fields(windows)

    def on_comp_curve_changed(self, _text: str = "") -> None:
        c = self._curve_by_name(self.comp_combo.currentText())
        if c is None:
            return
        try:
            windows = propose_windows(np.asarray(c.q, float), np.asarray(c.intensity, float))
            self._fill_window_fields(windows)
        except Exception:
            pass
        self._comp_last = None
        self.comp_plot.request_redraw(lambda c=c: self._render_comp_preview(c))
        self.comp_report.setPlainText("")

    def _render_comp_preview(self, c: Curve) -> None:
        fig = self.comp_plot.figure
        fig.clf()
        ax = fig.add_subplot(111)
        ax.loglog(c.q, np.clip(c.intensity, 1e-12, None), ".", ms=2, color="black", label=c.name)
        ax.legend(fontsize=8)
        ax.set_xlabel("q (Å⁻¹)")
        ax.set_ylabel("I (a.u.)")
        ax.grid(alpha=0.25, which="both")
        fig.tight_layout()
        self.comp_plot.canvas.draw_idle()

    # ------------------------------------------------------------------
    @staticmethod
    def _compute_composite_fit(curve: Curve, sample_id: str, windows, multistart_n: int, preset_name: str):
        if preset_name == "Auto (BIC ladder)":
            result = fit_staged(curve, sample_id=sample_id, windows=windows, multistart_n=multistart_n)
            model = _model_for_preset_name(result.preset_chosen)
            params = {k: v["value"] for k, v in result.params.items()}
            return {"model": model, "params": params, "derived": result.derived, "gof": result.gof,
                    "preset": result.preset_chosen, "flags": result.flags, "curve": curve}

        hygiene = apply_hygiene(curve)
        q = np.asarray(hygiene.curve.q, dtype=float)
        I = np.asarray(hygiene.curve.intensity, dtype=float)
        sigma = np.asarray(hygiene.curve.sigma, dtype=float)
        model = _model_for_preset_name(preset_name)
        seeds = model.seed(q, I, windows)
        params_in = model.to_lmfit_parameters(seed_values=seeds)
        lm_result = model.fit(q, I, sigma=sigma, params=params_in)
        params = {k: float(v.value) for k, v in lm_result.params.items()}
        derived = _build_derived(model, lm_result.params)
        gof = {"chi2red": float(lm_result.redchi), "aic": float(lm_result.aic), "bic": float(lm_result.bic),
               "n_points": int(len(q))}
        fitted_curve = Curve(q=q, intensity=I, sigma=sigma, name=curve.name)
        return {"model": model, "params": params, "derived": derived, "gof": gof,
                "preset": preset_name, "flags": [], "curve": fitted_curve}

    def run_composite_fit(self) -> None:
        from qt_worker import run_in_thread
        c = self._curve_by_name(self.comp_combo.currentText())
        if c is None:
            QMessageBox.warning(self, "Composite fit", "Import and pick a curve first.")
            return
        windows = self._read_window_fields()
        preset_name = self.comp_preset_combo.currentText()
        multistart_n = int(_to_float(self.comp_multistart_edit.text(), 8.0) or 8)
        curve_copy = Curve(q=np.asarray(c.q, float), intensity=np.asarray(c.intensity, float),
                           sigma=np.asarray(c.sigma, float) if c.sigma is not None else None, name=c.name)
        sample_id = c.name

        self._comp_fit_btn.setEnabled(False)
        self._comp_fit_btn.setText("Fitting…")

        def compute(curve_copy=curve_copy, sample_id=sample_id, windows=windows,
                    multistart_n=multistart_n, preset_name=preset_name):
            return self._compute_composite_fit(curve_copy, sample_id, windows, multistart_n, preset_name)

        run_in_thread(compute, self._on_comp_fit_done, self._on_comp_fit_error)

    def _on_comp_fit_error(self, traceback_text: str) -> None:
        self._comp_fit_btn.setEnabled(True)
        self._comp_fit_btn.setText("Fit selected curve")
        QMessageBox.critical(self, "Composite fit error", traceback_text)

    def _on_comp_fit_done(self, payload) -> None:
        self._comp_fit_btn.setEnabled(True)
        self._comp_fit_btn.setText("Fit selected curve")
        self._comp_last = payload
        self.comp_plot.cancel_pending()
        self._render_comp_result()
        self.comp_report.setPlainText(self._composite_report_text(payload))

    def _render_comp_result(self) -> None:
        payload = self._comp_last
        if payload is None:
            return
        curve, model, params = payload["curve"], payload["model"], payload["params"]
        self.comp_plot.preserve_zoom((curve.name, payload["preset"]))
        fig = self.comp_plot.figure
        fig.clf()
        ax = fig.add_subplot(111)
        q = np.asarray(curve.q, float)
        I = np.asarray(curve.intensity, float)
        ax.loglog(q, np.clip(I, 1e-12, None), ".", ms=2, color="black", label=curve.name)
        total = model.eval(q, params)
        ax.loglog(q, np.clip(total, 1e-12, None), lw=1.6, color="crimson", label=f"total ({payload['preset']})")
        for name, comp_curve in model.eval_components(q, params).items():
            ax.loglog(q, np.clip(comp_curve, 1e-12, None), lw=1.0, alpha=0.7, label=name)
        ax.legend(fontsize=7)
        ax.set_xlabel("q (Å⁻¹)")
        ax.set_ylabel("I (a.u.)")
        ax.grid(alpha=0.25, which="both")
        self.comp_plot.restore_zoom(ax)
        fig.tight_layout()
        self.comp_plot.canvas.draw_idle()

    @staticmethod
    def _composite_report_text(payload) -> str:
        lines = [f"Preset: {payload['preset']}"]
        if payload["flags"]:
            lines.append("Flags: " + "; ".join(payload["flags"]))
        lines.append("")
        lines.append("Derived:")
        for k, v in payload["derived"].items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                lines.append(f"  {k} = {v:.5g}")
        lines.append("")
        lines.append("Goodness of fit:")
        for k, v in payload["gof"].items():
            lines.append(f"  {k} = {v}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def run_composite_batch(self) -> None:
        from qt_worker import run_in_thread
        rows = sorted({i.row() for i in self.comp_batch_list.selectedIndexes()})
        if not rows:
            QMessageBox.warning(self, "Composite batch", "Select at least one curve in the batch list.")
            return
        items = [BatchItem(sample_id=self.curves[r].name, curve=self.curves[r],
                           group="default", order_hint=float(i))
                 for i, r in enumerate(rows)]
        curves_by_id = {it.sample_id: it.curve for it in items}
        # Deliberately NOT reading the single-curve window fields here: those
        # are scoped to whichever one curve is selected in the combo above,
        # auto-proposed for that curve specifically. A batch spans curves
        # with different peak positions in general, so each one must get its
        # own auto-proposed windows (run_batch's own windows=None default) —
        # blanket-applying one curve's windows across an unrelated batch is
        # silently wrong the moment peak positions differ between samples.
        multistart_n = int(_to_float(self.comp_multistart_edit.text(), 8.0) or 8)

        self._comp_batch_btn.setEnabled(False)
        self._comp_batch_btn.setText("Running…")

        def compute(items=items, multistart_n=multistart_n):
            return run_batch(items, windows=None, multistart_n=multistart_n)

        run_in_thread(compute, lambda batch: self._on_comp_batch_done(batch, curves_by_id),
                      self._on_comp_batch_error)

    def _on_comp_batch_error(self, traceback_text: str) -> None:
        self._comp_batch_btn.setEnabled(True)
        self._comp_batch_btn.setText("Run batch")
        QMessageBox.critical(self, "Composite batch error", traceback_text)

    def _on_comp_batch_done(self, batch, curves_by_id) -> None:
        self._comp_batch_btn.setEnabled(True)
        self._comp_batch_btn.setText("Run batch")
        self._comp_batch_result = batch
        self._comp_batch_curves = curves_by_id
        rows = batch_to_csv_rows(batch, curves_by_id)
        cols = ["sample_id", "preset_chosen", "derived_d", "derived_xi", "derived_fa",
                "gof_chi2red", "d_ts_over_d_gauss"]
        self.comp_batch_table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, key in enumerate(cols):
                val = row.get(key, "")
                text = "" if val is None else (f"{val:.5g}" if isinstance(val, float) else str(val))
                self.comp_batch_table.setItem(r, c, QTableWidgetItem(text))
        self.comp_batch_table.resizeColumnsToContents()
        if batch.errors:
            QMessageBox.warning(self, "Composite batch",
                                "Some samples failed to fit:\n" +
                                "\n".join(f"{k}: {v}" for k, v in batch.errors.items()))

    def export_composite_batch_csv(self) -> None:
        if self._comp_batch_result is None:
            QMessageBox.information(self, "Export", "Run a batch fit first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export batch results as…", "", "CSV (*.csv)")
        if not path:
            return
        try:
            write_batch_csv(path, self._comp_batch_result, self._comp_batch_curves)
        except OSError as exc:
            QMessageBox.critical(self, "Export error", str(exc))
            return
        QMessageBox.information(self, "Export", f"Batch results exported to:\n{path}")
