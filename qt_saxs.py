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
    QMessageBox, QPlainTextEdit, QPushButton, QSplitter, QTabWidget,
    QVBoxLayout, QWidget,
)

from qt_models import Spectrum, SpectrumLibrary
from qt_widgets import PlotWidget
from saxs_core.analysis import (
    auto_detect_guinier_region, auto_detect_peak_window, auto_detect_porod_region,
    fit_guinier, fit_porod_general, fit_pseudo_bragg_peak,
)
from saxs_core.chemistry import CapillaryConfig, SamplePhysicsConfig
from saxs_core.curve import Curve
from saxs_core.loader import load_curve
from saxs_core.reduction import CorrectionSettings, correct_sample
from saxs_core.waxs import auto_find_peaks, fit_waxs_peaks


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

        rl.addWidget(self.tabs)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)

    # ------------------------------------------------------------------
    def set_spectra(self, spectrum_ids: List[str]) -> None:  # nav-enter hook (self-contained workspace)
        pass

    def _refresh_combos(self) -> None:
        names = [c.name for c in self.curves]
        for combo in (self.red_sample_combo, self.red_empty_combo, self.ana_combo, self.waxs_combo):
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
