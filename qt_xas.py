"""
qt_xas.py — XAS/XANES/EXAFS processing ported to Qt (M11), built on
xas_science.py's SpectrumStore/Spectrum/Operation object model (already
identity-stable via `Spectrum.sid` — no redesign needed here, unlike
qt_models.SpectrumLibrary which replaced main.py's four-parallel-list
anti-pattern from scratch).

Core slice ported faithfully from xas_processing_v10.py's XASUltimateApp:
object list (import ZIP/CSV/.prj, rename/duplicate/delete/export), Preview,
mu(E) Builder, Normalization + EXAFS/FT (Larch), Tools (edge definer),
Export (Athena .dat/.prj).

Two real bugs found and fixed in xas_science.py while building this (see
its own comments): (1) larch_normalize/larch_exafs_pipeline set pre1/pre2/
norm1/norm2/nnorm as Group attributes but never passed them as explicit
kwargs to pre_edge(), which only reads e0 that way — Larch silently used
its own auto-computed defaults instead for every call until now; (2)
compute_mu()'s deglitch/deglitch_z/deglitch_window parameters were dead
(accepted, never referenced) — exactly the gap the plan asked M11 to
"confirm." Fixed there; exposed here as real deglitch controls in the
mu(E) Builder tab (applied as an optional post-step after build_mu(), since
the interactive builder aligns two possibly-different-grid I0/It spectra
via build_mu()'s own interpolation — compute_mu() assumes a shared grid
already and isn't a drop-in replacement for that).

Athena-inspired additions (new "Analysis" tab, cheap/self-contained ones
only — PCA needs a new scikit-learn dependency better co-scoped with M16;
self-absorption correction is a substantial standalone physics feature):
merge/average repeat scans, difference spectra, linear combination fitting
(NNLS-based, 2+ references).

Deliberately deferred to a documented follow-up (not half-implemented):
the Pre-processing tab (smoothing preview/apply, Bragg angle/energy
correction, Mode C interactive click-based feature alignment — all fairly
beamline/angle-dispersive-specific), the separate I0 baseline-"Fit" tab,
PCA, self-absorption correction, the CSV Builder export tool.
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMenu, QMessageBox, QPushButton,
    QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QTextEdit,
    QVBoxLayout, QWidget,
)

from qt_widgets import PlotWidget
from xas_science import (
    LARCH_AVAILABLE,
    Operation,
    Spectrum,
    SpectrumStore,
    _classify_kind_from_name,
    _extract_energy_angle_signal,
    _interp_to_grid,
    _periodic_table_symbols,
    _uid,
    build_mu,
    deglitch_mu,
    edge_text,
    export_athena_column,
    export_athena_prj_best_effort,
    infer_edge_label_from_roi_scaled,
    larch_exafs_pipeline,
    larch_normalize,
    read_athena_prj,
    read_csv_dataset,
    read_easyxafs_zip,
    require_larch,
)

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]


def _to_float(text: str, default: Optional[float] = None) -> Optional[float]:
    try:
        return float((text or "").strip())
    except (TypeError, ValueError):
        return default


def _to_int(text: str, default: int) -> int:
    try:
        return int(float((text or "").strip()))
    except (TypeError, ValueError):
        return default


class XasWorkspace(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.store = SpectrumStore()
        self.selected_sid: Optional[str] = None
        self._build_ui()
        self._refresh_all()

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

        import_row = QHBoxLayout()
        for label, handler in [
            ("ZIP…", self.import_zips), ("CSV…", self.import_csvs),
            (".prj…", self.import_prj), ("Clear", self.clear_all),
        ]:
            btn = QPushButton(label)
            btn.clicked.connect(handler)
            import_row.addWidget(btn)
        left_layout.addLayout(import_row)

        left_layout.addWidget(QLabel("Imported spectra (objects)"))
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Name", "Kind", "Edge", "E0", "E range"])
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._on_table_context_menu)
        left_layout.addWidget(self.table, 1)

        self.status_label = QLabel("Ready.")
        self.status_label.setWordWrap(True)
        left_layout.addWidget(self.status_label)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_preview_tab(), "Preview")
        self.tabs.addTab(self._build_mu_tab(), "μ(E) Builder")
        self.tabs.addTab(self._build_norm_tab(), "Normalization / EXAFS")
        self.tabs.addTab(self._build_analysis_tab(), "Analysis")
        self.tabs.addTab(self._build_tools_tab(), "Tools")
        self.tabs.addTab(self._build_export_tab(), "Export")
        right_layout.addWidget(self.tabs)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)

    def _set_status(self, msg: str) -> None:
        self.status_label.setText(msg)

    # ------------------------------------------------------------------
    # Preview tab
    # ------------------------------------------------------------------
    def _build_preview_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        self.preview_plot = PlotWidget(figsize=(7, 5))
        layout.addWidget(self.preview_plot)
        return w

    def _plot_selected_preview(self) -> None:
        if self.selected_sid is None:
            self.preview_plot.clear("Preview")
            return
        sp = self.store.get(self.selected_sid)
        ax = self.preview_plot.ax
        ax.clear()
        ax.plot(sp.energy, sp.y, lw=1.2, color=COLORS[0], label=sp.name)
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel(sp.units)
        ax.set_title(f"{sp.label} — {sp.name} [{sp.kind}]")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        self.preview_plot.figure.tight_layout()
        self.preview_plot.canvas.draw_idle()

    # ------------------------------------------------------------------
    # mu(E) Builder tab
    # ------------------------------------------------------------------
    def _build_mu_tab(self) -> QWidget:
        w = QWidget()
        layout = QHBoxLayout(w)

        ctrl = QWidget()
        ctrl.setMaximumWidth(320)
        ctrl_layout = QVBoxLayout(ctrl)

        ctrl_layout.addWidget(QLabel("I0 selection"))
        self.mu_i0_combo = QComboBox()
        self.mu_i0_combo.currentIndexChanged.connect(self._preview_mu)
        ctrl_layout.addWidget(self.mu_i0_combo)

        ctrl_layout.addWidget(QLabel("It spectra (multi-select)"))
        self.mu_it_list = QListWidget()
        self.mu_it_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.mu_it_list.itemSelectionChanged.connect(self._preview_mu)
        ctrl_layout.addWidget(self.mu_it_list, 1)

        log_row = QHBoxLayout()
        log_row.addWidget(QLabel("log"))
        self.mu_log_combo = QComboBox()
        self.mu_log_combo.addItems(["ln", "log10"])
        self.mu_log_combo.currentTextChanged.connect(self._preview_mu)
        log_row.addWidget(self.mu_log_combo)
        ctrl_layout.addLayout(log_row)

        self.mu_deglitch_check = QCheckBox("Deglitch")
        ctrl_layout.addWidget(self.mu_deglitch_check)
        degl_row = QHBoxLayout()
        degl_row.addWidget(QLabel("z"))
        self.mu_deglitch_z_edit = QLineEdit("6.0")
        self.mu_deglitch_z_edit.setMaximumWidth(50)
        degl_row.addWidget(self.mu_deglitch_z_edit)
        degl_row.addWidget(QLabel("window"))
        self.mu_deglitch_window_edit = QLineEdit("21")
        self.mu_deglitch_window_edit.setMaximumWidth(50)
        degl_row.addWidget(self.mu_deglitch_window_edit)
        ctrl_layout.addLayout(degl_row)

        preview_btn = QPushButton("Preview μ")
        preview_btn.clicked.connect(self._preview_mu)
        ctrl_layout.addWidget(preview_btn)
        compute_btn = QPushButton("Compute μ → new objects")
        compute_btn.setObjectName("Primary")
        compute_btn.clicked.connect(self.compute_mu_selected)
        ctrl_layout.addWidget(compute_btn)
        ctrl_layout.addStretch(1)
        layout.addWidget(ctrl)

        self.mu_plot = PlotWidget(figsize=(7, 5))
        layout.addWidget(self.mu_plot, 1)
        return w

    def _mu_deglitch_params(self):
        enabled = self.mu_deglitch_check.isChecked()
        z = _to_float(self.mu_deglitch_z_edit.text(), 6.0)
        window = _to_int(self.mu_deglitch_window_edit.text(), 21)
        return enabled, z, window

    def _build_mu(self, i0: Spectrum, it: Spectrum) -> np.ndarray:
        mu = build_mu(i0_energy=i0.energy, i0=i0.y, it_energy=it.energy, it=it.y, log_mode=self.mu_log_combo.currentText())
        enabled, z, window = self._mu_deglitch_params()
        if enabled:
            mu = deglitch_mu(mu, z=z, window=window)
        return mu

    def _preview_mu(self) -> None:
        i0_name = self.mu_i0_combo.currentText()
        if not i0_name:
            self.mu_plot.clear("μ preview")
            return
        i0 = self.store.find_by_name(i0_name)
        it_items = self.mu_it_list.selectedItems()
        if i0 is None or not it_items:
            self.mu_plot.clear("μ preview")
            return
        it = self.store.find_by_name(it_items[0].text())
        if it is None:
            return
        try:
            mu = self._build_mu(i0, it)
        except Exception as exc:
            QMessageBox.critical(self, "μ preview error", str(exc))
            return
        ax = self.mu_plot.ax
        ax.clear()
        ax.plot(i0.energy, mu, lw=1.2, color=COLORS[0], label=f"μ from {it.name}")
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("arb.")
        ax.set_title(f"μ preview — I0={i0.name}")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        self.mu_plot.figure.tight_layout()
        self.mu_plot.canvas.draw_idle()

    def compute_mu_selected(self) -> None:
        i0_name = self.mu_i0_combo.currentText()
        if not i0_name:
            QMessageBox.warning(self, "μ builder", "Select an I0 spectrum.")
            return
        i0 = self.store.find_by_name(i0_name)
        if i0 is None:
            return
        it_items = self.mu_it_list.selectedItems()
        if not it_items:
            QMessageBox.warning(self, "μ builder", "Select at least one It spectrum.")
            return

        last = None
        for item in it_items:
            it = self.store.find_by_name(item.text())
            if it is None:
                continue
            try:
                mu = self._build_mu(i0, it)
            except Exception as exc:
                QMessageBox.critical(self, "μ builder error", str(exc))
                continue
            sp_mu = it.copy(new_name=f"{it.name}_mu", new_kind="mu")
            sp_mu.energy = np.asarray(i0.energy, float)
            sp_mu.y = np.asarray(mu, float)
            sp_mu.label = it.label
            sp_mu.e0 = it.e0
            enabled, z, window = self._mu_deglitch_params()
            sp_mu.history.append(Operation("mu_builder", {"I0": i0.name, "It": it.name, "log": self.mu_log_combo.currentText(), "deglitch": enabled, "deglitch_z": z, "deglitch_window": window}))
            self.store.add(sp_mu)
            last = sp_mu

        self._refresh_all()
        if last is not None:
            self._set_status(f"Computed μ for {len(it_items)} It spectrum/spectra.")

    # ------------------------------------------------------------------
    # Normalization / EXAFS tab
    # ------------------------------------------------------------------
    def _build_norm_tab(self) -> QWidget:
        w = QWidget()
        layout = QHBoxLayout(w)

        ctrl = QWidget()
        ctrl.setMaximumWidth(360)
        ctrl_layout = QVBoxLayout(ctrl)

        e0_row = QHBoxLayout()
        e0_row.addWidget(QLabel("E0 method"))
        self.norm_e0_combo = QComboBox()
        self.norm_e0_combo.addItems(["larch", "deriv", "manual"])
        e0_row.addWidget(self.norm_e0_combo)
        e0_row.addWidget(QLabel("manual"))
        self.norm_e0_manual_edit = QLineEdit()
        self.norm_e0_manual_edit.setMaximumWidth(70)
        e0_row.addWidget(self.norm_e0_manual_edit)
        ctrl_layout.addLayout(e0_row)

        grid_row1 = QHBoxLayout()
        grid_row1.addWidget(QLabel("pre1"))
        self.norm_pre1_edit = QLineEdit("-150")
        grid_row1.addWidget(self.norm_pre1_edit)
        grid_row1.addWidget(QLabel("pre2"))
        self.norm_pre2_edit = QLineEdit("-50")
        grid_row1.addWidget(self.norm_pre2_edit)
        ctrl_layout.addLayout(grid_row1)

        grid_row2 = QHBoxLayout()
        grid_row2.addWidget(QLabel("norm1"))
        self.norm_norm1_edit = QLineEdit("30")
        grid_row2.addWidget(self.norm_norm1_edit)
        grid_row2.addWidget(QLabel("norm2"))
        self.norm_norm2_edit = QLineEdit("150")
        grid_row2.addWidget(self.norm_norm2_edit)
        grid_row2.addWidget(QLabel("nnorm"))
        self.norm_nnorm_edit = QLineEdit("1")
        grid_row2.addWidget(self.norm_nnorm_edit)
        ctrl_layout.addLayout(grid_row2)

        self.norm_smooth_check = QCheckBox("Smooth for E0/derivative only")
        self.norm_smooth_check.setChecked(True)
        ctrl_layout.addWidget(self.norm_smooth_check)

        normalize_btn = QPushButton("Normalize selected μ → new objects")
        normalize_btn.setObjectName("Primary")
        normalize_btn.clicked.connect(self.normalize_selected)
        ctrl_layout.addWidget(normalize_btn)

        ctrl_layout.addWidget(QLabel("EXAFS / FT (Larch autobk + xftf)"))
        ex_row1 = QHBoxLayout()
        ex_row1.addWidget(QLabel("rbkg"))
        self.exafs_rbkg_edit = QLineEdit("1.0")
        ex_row1.addWidget(self.exafs_rbkg_edit)
        ex_row1.addWidget(QLabel("kmin"))
        self.exafs_kmin_edit = QLineEdit("0")
        ex_row1.addWidget(self.exafs_kmin_edit)
        ex_row1.addWidget(QLabel("kmax"))
        self.exafs_kmax_edit = QLineEdit("15")
        ex_row1.addWidget(self.exafs_kmax_edit)
        ctrl_layout.addLayout(ex_row1)

        ex_row2 = QHBoxLayout()
        ex_row2.addWidget(QLabel("dk"))
        self.exafs_dk_edit = QLineEdit("0.1")
        ex_row2.addWidget(self.exafs_dk_edit)
        ex_row2.addWidget(QLabel("k-weight"))
        self.exafs_kweight_edit = QLineEdit("2")
        ex_row2.addWidget(self.exafs_kweight_edit)
        ctrl_layout.addLayout(ex_row2)

        ex_row3 = QHBoxLayout()
        ex_row3.addWidget(QLabel("window"))
        self.exafs_window_combo = QComboBox()
        self.exafs_window_combo.addItems(["hanning", "kaiser", "parzen", "welch", "sine", "gaussian"])
        ex_row3.addWidget(self.exafs_window_combo)
        ex_row3.addWidget(QLabel("rmax_out"))
        self.exafs_rmax_edit = QLineEdit("10")
        ex_row3.addWidget(self.exafs_rmax_edit)
        ctrl_layout.addLayout(ex_row3)

        exafs_btn = QPushButton("Compute χ(k) + FT → new objects")
        exafs_btn.clicked.connect(self.exafs_selected)
        ctrl_layout.addWidget(exafs_btn)

        ctrl_layout.addWidget(QLabel("Select μ spectra"))
        self.norm_mu_list = QListWidget()
        self.norm_mu_list.setSelectionMode(QListWidget.ExtendedSelection)
        ctrl_layout.addWidget(self.norm_mu_list, 1)

        layout.addWidget(ctrl)
        self.norm_plot = PlotWidget(figsize=(7, 5))
        layout.addWidget(self.norm_plot, 1)
        return w

    def _norm_params(self):
        e0_method = self.norm_e0_combo.currentText()
        e0_manual = _to_float(self.norm_e0_manual_edit.text()) if e0_method == "manual" else None
        pre1 = _to_float(self.norm_pre1_edit.text(), -150.0)
        pre2 = _to_float(self.norm_pre2_edit.text(), -50.0)
        norm1 = _to_float(self.norm_norm1_edit.text(), 30.0)
        norm2 = _to_float(self.norm_norm2_edit.text(), 150.0)
        nnorm = _to_int(self.norm_nnorm_edit.text(), 1)
        smooth_for_e0 = ("savitzky-golay", {"window": 11, "poly": 3}) if self.norm_smooth_check.isChecked() else None
        return e0_method, e0_manual, pre1, pre2, norm1, norm2, nnorm, smooth_for_e0

    def normalize_selected(self) -> None:
        items = self.norm_mu_list.selectedItems()
        if not items:
            QMessageBox.warning(self, "Normalization", "Select μ spectra.")
            return
        e0_method, e0_manual, pre1, pre2, norm1, norm2, nnorm, smooth_for_e0 = self._norm_params()

        last = None
        for item in items:
            sp = self.store.find_by_name(item.text())
            if sp is None:
                continue
            try:
                out = larch_normalize(sp.energy, sp.y, e0_method=e0_method, e0_manual=e0_manual, pre1=pre1, pre2=pre2, norm1=norm1, norm2=norm2, nnorm=nnorm, smooth_for_e0=smooth_for_e0)
            except Exception as exc:
                QMessageBox.critical(self, "Normalization error", str(exc))
                continue
            sp_norm = sp.copy(new_name=f"{sp.name}_norm", new_kind="norm")
            sp_norm.y = out["norm"]; sp_norm.e0 = out["e0"]
            sp_norm.history.append(Operation("normalize", {"e0_method": e0_method, "e0": out["e0"]}))
            sp_flat = sp.copy(new_name=f"{sp.name}_flat", new_kind="flat")
            sp_flat.y = out["flat"]; sp_flat.e0 = out["e0"]
            sp_flat.history.append(Operation("normalize_flat", {"e0": out["e0"]}))
            self.store.add(sp_norm); self.store.add(sp_flat)
            last = (sp, out)

        self._refresh_all()
        if last is not None:
            sp, out = last
            ax = self.norm_plot.ax
            ax.clear()
            ax.plot(sp.energy, sp.y, lw=1.1, label="μ(E)")
            ax.plot(sp.energy, out["norm"], lw=1.1, label="norm")
            ax.plot(sp.energy, out["deriv"], lw=1.0, label="dμ/dE", alpha=0.6)
            for val in (out["anchors"].get(k) for k in ("pre1", "pre2", "norm1", "norm2")):
                if val is not None:
                    ax.axvline(float(val), ls="--", lw=1.0, alpha=0.7)
            ax.set_xlabel("Energy (eV)"); ax.set_ylabel("arb.")
            ax.set_title(f"{sp.label} — E0={out['e0']:.2f}")
            ax.legend(fontsize=8); ax.grid(alpha=0.25)
            self.norm_plot.figure.tight_layout()
            self.norm_plot.canvas.draw_idle()

    def exafs_selected(self) -> None:
        items = self.norm_mu_list.selectedItems()
        if not items:
            QMessageBox.warning(self, "EXAFS/FT", "Select μ spectra.")
            return
        e0_method, e0_manual, pre1, pre2, norm1, norm2, nnorm, smooth_for_e0 = self._norm_params()
        rbkg = _to_float(self.exafs_rbkg_edit.text(), 1.0)
        kmin = _to_float(self.exafs_kmin_edit.text(), 0.0)
        kmax = _to_float(self.exafs_kmax_edit.text(), 15.0)
        dk = _to_float(self.exafs_dk_edit.text(), 0.1)
        kweight = _to_int(self.exafs_kweight_edit.text(), 2)
        window = self.exafs_window_combo.currentText()
        rmax_out = _to_float(self.exafs_rmax_edit.text(), 10.0)

        last = None
        for item in items:
            sp = self.store.find_by_name(item.text())
            if sp is None:
                continue
            try:
                out = larch_exafs_pipeline(
                    sp.energy, sp.y, e0_method=e0_method, e0_manual=e0_manual, pre1=pre1, pre2=pre2,
                    norm1=norm1, norm2=norm2, nnorm=nnorm, rbkg=rbkg, kmin=kmin, kmax=kmax, dk=dk,
                    kweight=kweight, window=window, rmax_out=rmax_out, smooth_for_e0=smooth_for_e0,
                )
            except Exception as exc:
                QMessageBox.critical(self, "EXAFS/FT error", str(exc))
                continue

            sp_norm = sp.copy(new_name=f"{sp.name}_norm", new_kind="norm"); sp_norm.y = out["norm"]; sp_norm.e0 = out["e0"]
            sp_norm.history.append(Operation("normalize", {"e0_method": e0_method}))
            sp_flat = sp.copy(new_name=f"{sp.name}_flat", new_kind="flat"); sp_flat.y = out["flat"]; sp_flat.e0 = out["e0"]
            sp_flat.history.append(Operation("normalize_flat", {"e0": out["e0"]}))
            self.store.add(sp_norm); self.store.add(sp_flat)

            sp_chi = sp.copy(new_name=f"{sp.name}_chi", new_kind="chi(k)"); sp_chi.energy = out["k"]; sp_chi.y = out["chi"]; sp_chi.e0 = out["e0"]
            sp_chi.history.append(Operation("autobk", {"rbkg": rbkg, "kmin": kmin, "kmax": kmax, "dk": dk}))
            self.store.add(sp_chi)

            sp_chikw = sp.copy(new_name=f"{sp.name}_chi_k{kweight}", new_kind=f"chi(k)*k^{kweight}")
            sp_chikw.energy = out["k"]; sp_chikw.y = out["chi_kw"]; sp_chikw.e0 = out["e0"]
            sp_chikw.history.append(Operation("kweight", {"kweight": kweight}))
            self.store.add(sp_chikw)

            sp_ft = sp.copy(new_name=f"{sp.name}_FTmag", new_kind="FT|chi|"); sp_ft.energy = out["r"]; sp_ft.y = out["chir_mag"]; sp_ft.e0 = out["e0"]
            sp_ft.history.append(Operation("xftf", {"kmin": kmin, "kmax": kmax, "dk": dk, "kweight": kweight, "window": window, "rmax_out": rmax_out}))
            self.store.add(sp_ft)
            last = (sp, out, kweight)

        self._refresh_all()
        if last is not None:
            sp, out, kweight = last
            ax = self.norm_plot.ax
            ax.clear()
            ax.plot(out["k"], out["chi"], lw=1.1, label="χ(k)")
            ax.plot(out["k"], out["chi_kw"], lw=1.1, label=f"χ(k)*k^{kweight}")
            ax.plot(out["r"], out["chir_mag"], lw=1.1, label="|FT|")
            ax.set_xlabel("k (1/Å) / R (Å)"); ax.set_ylabel("arb.")
            ax.set_title(f"{sp.label} — E0={out['e0']:.2f}")
            ax.legend(fontsize=8); ax.grid(alpha=0.25)
            self.norm_plot.figure.tight_layout()
            self.norm_plot.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Analysis tab (Athena-inspired additions)
    # ------------------------------------------------------------------
    def _build_analysis_tab(self) -> QWidget:
        w = QWidget()
        layout = QHBoxLayout(w)

        ctrl = QWidget()
        ctrl.setMaximumWidth(340)
        ctrl_layout = QVBoxLayout(ctrl)

        ctrl_layout.addWidget(QLabel("Select spectra (multi-select)"))
        self.analysis_list = QListWidget()
        self.analysis_list.setSelectionMode(QListWidget.ExtendedSelection)
        ctrl_layout.addWidget(self.analysis_list, 1)

        merge_btn = QPushButton("Merge / average selected → new object")
        merge_btn.clicked.connect(self.merge_average_selected)
        ctrl_layout.addWidget(merge_btn)

        diff_btn = QPushButton("Difference (1st − 2nd selected) → new object")
        diff_btn.clicked.connect(self.difference_selected)
        ctrl_layout.addWidget(diff_btn)

        ctrl_layout.addWidget(QLabel("Linear combination fit: last selected = target,\nothers = references"))
        lcf_btn = QPushButton("Fit linear combination")
        lcf_btn.clicked.connect(self.linear_combination_fit_selected)
        ctrl_layout.addWidget(lcf_btn)

        self.analysis_result_text = QTextEdit()
        self.analysis_result_text.setReadOnly(True)
        self.analysis_result_text.setMaximumHeight(140)
        ctrl_layout.addWidget(self.analysis_result_text)
        ctrl_layout.addStretch(1)
        layout.addWidget(ctrl)

        self.analysis_plot = PlotWidget(figsize=(7, 5))
        layout.addWidget(self.analysis_plot, 1)
        return w

    def _analysis_selected_spectra(self) -> List[Spectrum]:
        out = []
        for item in self.analysis_list.selectedItems():
            sp = self.store.find_by_name(item.text())
            if sp is not None:
                out.append(sp)
        return out

    def merge_average_selected(self) -> None:
        specs = self._analysis_selected_spectra()
        if len(specs) < 2:
            QMessageBox.warning(self, "Merge / average", "Select at least 2 spectra to merge.")
            return
        ref = specs[0]
        stacked = [ref.y]
        for sp in specs[1:]:
            stacked.append(_interp_to_grid(sp.energy, sp.y, ref.energy))
        avg_y = np.mean(np.vstack(stacked), axis=0)

        sp_avg = ref.copy(new_name=f"{ref.name}_avg{len(specs)}", new_kind=ref.kind)
        sp_avg.y = avg_y
        sp_avg.history.append(Operation("merge_average", {"members": [s.name for s in specs]}))
        self.store.add(sp_avg)
        self._refresh_all()

        ax = self.analysis_plot.ax
        ax.clear()
        for i, sp in enumerate(specs):
            ax.plot(sp.energy, sp.y, lw=0.9, alpha=0.5, color=COLORS[i % len(COLORS)], label=sp.name)
        ax.plot(ref.energy, avg_y, lw=1.8, color="black", label="average")
        ax.set_xlabel("Energy (eV)"); ax.set_ylabel(ref.units)
        ax.set_title(f"Merged average of {len(specs)} spectra")
        ax.legend(fontsize=7); ax.grid(alpha=0.25)
        self.analysis_plot.figure.tight_layout()
        self.analysis_plot.canvas.draw_idle()
        self._set_status(f"Averaged {len(specs)} spectra → {sp_avg.name}")

    def difference_selected(self) -> None:
        specs = self._analysis_selected_spectra()
        if len(specs) != 2:
            QMessageBox.warning(self, "Difference", "Select exactly 2 spectra (A then B; result is A − B).")
            return
        a, b = specs
        b_interp = _interp_to_grid(b.energy, b.y, a.energy)
        diff_y = a.y - b_interp

        sp_diff = a.copy(new_name=f"{a.name}_minus_{b.name}", new_kind=f"diff({a.kind})")
        sp_diff.y = diff_y
        sp_diff.history.append(Operation("difference", {"a": a.name, "b": b.name}))
        self.store.add(sp_diff)
        self._refresh_all()

        ax = self.analysis_plot.ax
        ax.clear()
        ax.plot(a.energy, a.y, lw=1.1, label=a.name)
        ax.plot(a.energy, b_interp, lw=1.1, label=b.name)
        ax.plot(a.energy, diff_y, lw=1.4, color="black", label="difference (A − B)")
        ax.set_xlabel("Energy (eV)"); ax.set_ylabel(a.units)
        ax.set_title(f"Difference: {a.name} − {b.name}")
        ax.legend(fontsize=8); ax.grid(alpha=0.25)
        self.analysis_plot.figure.tight_layout()
        self.analysis_plot.canvas.draw_idle()
        self._set_status(f"Difference spectrum → {sp_diff.name}")

    def linear_combination_fit_selected(self) -> None:
        """Athena-style linear combination fitting: fit the LAST selected
        spectrum (target) as a non-negative weighted sum of the other
        selected spectra (references), via NNLS."""
        from scipy.optimize import nnls

        specs = self._analysis_selected_spectra()
        if len(specs) < 3:
            QMessageBox.warning(self, "Linear combination fit", "Select 2+ references and 1 target (3+ total; target = last selected).")
            return
        target = specs[-1]
        refs = specs[:-1]

        A = np.column_stack([_interp_to_grid(r.energy, r.y, target.energy) for r in refs])
        weights, residual_norm = nnls(A, target.y)
        fit_y = A @ weights
        ss_res = float(np.sum((target.y - fit_y) ** 2))
        ss_tot = float(np.sum((target.y - np.mean(target.y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else float("nan")

        lines = [f"Target: {target.name}", f"R² = {r2:.5f}", ""]
        for r, w_ in zip(refs, weights):
            lines.append(f"  {r.name}: weight = {w_:.4f}")
        lines.append(f"\nSum of weights = {sum(weights):.4f} (Athena-style LCF is often constrained to sum to 1; not enforced here, shown for reference)")
        self.analysis_result_text.setPlainText("\n".join(lines))

        sp_fit = target.copy(new_name=f"{target.name}_LCFfit", new_kind="lcf_fit")
        sp_fit.y = fit_y
        sp_fit.history.append(Operation("linear_combination_fit", {"target": target.name, "refs": [r.name for r in refs], "weights": list(map(float, weights)), "r2": r2}))
        self.store.add(sp_fit)
        self._refresh_all()

        ax = self.analysis_plot.ax
        ax.clear()
        ax.plot(target.energy, target.y, lw=1.3, color="black", label=f"target: {target.name}")
        ax.plot(target.energy, fit_y, lw=1.5, ls="--", color="red", label="LCF fit")
        for i, r in enumerate(refs):
            ax.plot(r.energy, r.y, lw=0.8, alpha=0.4, color=COLORS[i % len(COLORS)], label=r.name)
        ax.set_xlabel("Energy (eV)"); ax.set_ylabel(target.units)
        ax.set_title(f"Linear combination fit — R²={r2:.4f}")
        ax.legend(fontsize=7); ax.grid(alpha=0.25)
        self.analysis_plot.figure.tight_layout()
        self.analysis_plot.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Tools tab (edge definer)
    # ------------------------------------------------------------------
    def _build_tools_tab(self) -> QWidget:
        w = QWidget()
        layout = QHBoxLayout(w)

        ctrl = QWidget()
        ctrl.setMaximumWidth(320)
        ctrl_layout = QVBoxLayout(ctrl)

        ctrl_layout.addWidget(QLabel("Select spectra (multi-select)"))
        self.edge_list = QListWidget()
        self.edge_list.setSelectionMode(QListWidget.ExtendedSelection)
        ctrl_layout.addWidget(self.edge_list, 1)

        elem_row = QHBoxLayout()
        elem_row.addWidget(QLabel("Element"))
        self.edge_elem_combo = QComboBox()
        self.edge_elem_combo.addItems(_periodic_table_symbols())
        self.edge_elem_combo.setCurrentText("Fe")
        elem_row.addWidget(self.edge_elem_combo)
        ctrl_layout.addLayout(elem_row)

        edge_row = QHBoxLayout()
        edge_row.addWidget(QLabel("Edge"))
        self.edge_line_combo = QComboBox()
        self.edge_line_combo.addItems(["K", "L1", "L2", "L3", "M1", "M2", "M3", "M4", "M5"])
        edge_row.addWidget(self.edge_line_combo)
        ctrl_layout.addLayout(edge_row)

        self.edge_set_e0_check = QCheckBox("Also set E0 to tabulated edge energy (xraydb)")
        ctrl_layout.addWidget(self.edge_set_e0_check)

        preview_btn = QPushButton("Preview")
        preview_btn.clicked.connect(self.preview_edge_definer)
        ctrl_layout.addWidget(preview_btn)
        apply_btn = QPushButton("Apply to selected spectra")
        apply_btn.setObjectName("Primary")
        apply_btn.clicked.connect(self.apply_edge_definer)
        ctrl_layout.addWidget(apply_btn)
        ctrl_layout.addStretch(1)
        layout.addWidget(ctrl)

        self.tools_plot = PlotWidget(figsize=(7, 5))
        layout.addWidget(self.tools_plot, 1)
        return w

    def _edge_selected_spectra(self) -> List[Spectrum]:
        out = []
        for item in self.edge_list.selectedItems():
            sp = self.store.find_by_name(item.text())
            if sp is not None:
                out.append(sp)
        return out

    def preview_edge_definer(self) -> None:
        specs = self._edge_selected_spectra()
        if not specs:
            self.tools_plot.clear("Edge definer preview")
            return
        elem = self.edge_elem_combo.currentText()
        edge = self.edge_line_combo.currentText()
        e_edge = None
        try:
            Group, xraydb, find_e0, pre_edge, autobk, xftf = require_larch()
            if xraydb is not None:
                edges = xraydb.xray_edges(elem)
                if edge in edges and getattr(edges[edge], "energy", None) is not None:
                    e_edge = float(edges[edge].energy)
        except Exception:
            pass

        ax = self.tools_plot.ax
        ax.clear()
        for i, sp in enumerate(specs):
            ax.plot(sp.energy, sp.y, lw=1.1, color=COLORS[i % len(COLORS)], label=sp.name)
        ax.set_xlabel("Energy (eV)"); ax.set_ylabel("arb.")
        ax.set_title(f"Edge definer preview — {elem} {edge}")
        ax.legend(fontsize=8); ax.grid(alpha=0.25)
        if e_edge is not None and np.isfinite(e_edge):
            ax.axvline(e_edge, ls="--", lw=1.2)
        self.tools_plot.figure.tight_layout()
        self.tools_plot.canvas.draw_idle()

    def apply_edge_definer(self) -> None:
        specs = self._edge_selected_spectra()
        if not specs:
            QMessageBox.warning(self, "Edge definer", "Select at least one spectrum.")
            return
        elem = self.edge_elem_combo.currentText()
        edge = self.edge_line_combo.currentText()
        e_edge = None
        if self.edge_set_e0_check.isChecked():
            try:
                Group, xraydb, find_e0, pre_edge, autobk, xftf = require_larch()
                if xraydb is None:
                    raise ValueError("xraydb not available in this Larch install; cannot set tabulated E0.")
                edges = xraydb.xray_edges(elem)
                if edge not in edges or getattr(edges[edge], "energy", None) is None:
                    raise ValueError("Unknown element/edge in xraydb.")
                e_edge = float(edges[edge].energy)
            except Exception as exc:
                QMessageBox.critical(self, "Edge definer error", str(exc))
                return

        for sp in specs:
            sp.label = f"XAS({elem} {edge})"
            if e_edge is not None:
                sp.e0 = e_edge
            sp.history.append(Operation("edge_definer", {"element": elem, "edge": edge, "set_e0": e_edge is not None}))

        self._refresh_all()
        self.preview_edge_definer()
        self._set_status(f"Applied manual edge label to {len(specs)} spectrum/spectra.")

    # ------------------------------------------------------------------
    # Export tab
    # ------------------------------------------------------------------
    def _build_export_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.addWidget(QLabel("Athena export"))
        row = QHBoxLayout()
        dat_btn = QPushButton("Export selected as Athena column (.dat)")
        dat_btn.clicked.connect(self.export_athena_dat)
        row.addWidget(dat_btn)
        prj_btn = QPushButton("Export ALL mu/norm/flat as Athena project (.prj)")
        prj_btn.clicked.connect(self.export_athena_prj)
        row.addWidget(prj_btn)
        layout.addLayout(row)
        layout.addStretch(1)
        return w

    def export_athena_dat(self) -> None:
        if self.selected_sid is None:
            QMessageBox.information(self, "Export", "Select a spectrum first.")
            return
        sp = self.store.get(self.selected_sid)
        path, _ = QFileDialog.getSaveFileName(self, "Save Athena column file", "", "Athena column file (*.dat)")
        if not path:
            return
        header = ["# Athena column file exported from Dataapp", f"# name = {sp.name}", f"# kind = {sp.kind}", f"# label = {sp.label}"]
        if sp.e0 is not None and np.isfinite(sp.e0):
            header.append(f"# e0 = {sp.e0:.6f}")
        try:
            export_athena_column(path, sp.energy, sp.y, header)
            self._set_status(f"Saved: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export .dat error", str(exc))

    def export_athena_prj(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save Athena project (.prj)", "", "Athena project (*.prj)")
        if not path:
            return
        try:
            ok = export_athena_prj_best_effort(path, [s for s in self.store.all() if s.kind in ("mu", "norm", "flat")])
            if not ok:
                QMessageBox.warning(self, "Export .prj", "Could not find write_athena in this Larch install. Export .dat instead.")
            else:
                self._set_status(f"Saved .prj: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export .prj error", str(exc))

    # ------------------------------------------------------------------
    # Import / object management
    # ------------------------------------------------------------------
    def _spectrum_from_record(self, rec: Dict[str, Any]) -> Spectrum:
        angle, energy, signal, cols = _extract_energy_angle_signal(rec["df"])
        kind = _classify_kind_from_name(rec["name"])
        scan_def = rec.get("scan_def", {}) or {}
        label, e0 = infer_edge_label_from_roi_scaled(energy, signal, scan_def)
        return Spectrum(
            sid=_uid("sp"), name=rec["name"], kind=kind, energy=energy, y=signal,
            angle=angle if np.isfinite(angle).any() else None, units="counts/s", label=label, e0=e0,
            meta={"source": rec.get("source", ""), "columns": cols, "scan_def": scan_def, "metadata": rec.get("metadata", {})},
        )

    def import_zips(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Select EasyXAFS ZIP(s)", "", "ZIP files (*.zip);;All files (*.*)")
        if not paths:
            return
        try:
            n = 0
            for zp in paths:
                for rec in read_easyxafs_zip(zp):
                    sp = self._spectrum_from_record(rec)
                    sp.history.append(Operation("import", {"source": rec.get("source", "")}))
                    self.store.add(sp)
                    n += 1
            self._refresh_all()
            self._set_status(f"Imported {n} dataset(s) from ZIP(s).")
        except Exception as exc:
            QMessageBox.critical(self, "Import ZIP error", str(exc))

    def import_csvs(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Select CSV(s)", "", "CSV files (*.csv);;All files (*.*)")
        if not paths:
            return
        try:
            n = 0
            for p in paths:
                rec = read_csv_dataset(p)
                sp = self._spectrum_from_record(rec)
                sp.history.append(Operation("import", {"source": rec.get("source", "")}))
                self.store.add(sp)
                n += 1
            self._refresh_all()
            self._set_status(f"Imported {n} CSV dataset(s).")
        except Exception as exc:
            QMessageBox.critical(self, "Import CSV error", str(exc))

    def import_prj(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select Athena project (.prj)", "", "Athena project (*.prj);;All files (*.*)")
        if not path:
            return
        try:
            specs = read_athena_prj(path)
            for sp in specs:
                self.store.add(sp)
            self._refresh_all()
            self._set_status(f"Imported {len(specs)} group(s) from .prj.")
        except Exception as exc:
            QMessageBox.critical(self, "Import .prj error", str(exc))

    def clear_all(self) -> None:
        self.store.clear()
        self.selected_sid = None
        self._refresh_all()
        self._set_status("Cleared all spectra.")

    def load_initial_spectra(self, specs: List[Spectrum]) -> None:
        for sp in specs:
            self.store.add(sp)
        self._refresh_all()

    # ------------------------------------------------------------------
    def _refresh_all(self) -> None:
        self._refresh_table()
        self._refresh_lists()

    def _refresh_table(self) -> None:
        specs = self.store.all()
        self.table.setRowCount(len(specs))
        for row, sp in enumerate(specs):
            e0_txt = "" if sp.e0 is None or not np.isfinite(sp.e0) else f"{sp.e0:.1f}"
            er_txt = f"{np.nanmin(sp.energy):.1f}–{np.nanmax(sp.energy):.1f}" if sp.energy.size else ""
            name_item = QTableWidgetItem(sp.name)
            name_item.setData(Qt.UserRole, sp.sid)
            self.table.setItem(row, 0, name_item)
            self.table.setItem(row, 1, QTableWidgetItem(sp.kind))
            self.table.setItem(row, 2, QTableWidgetItem(edge_text(sp.label)))
            self.table.setItem(row, 3, QTableWidgetItem(e0_txt))
            self.table.setItem(row, 4, QTableWidgetItem(er_txt))
        self.table.resizeColumnsToContents()

    def _refresh_lists(self) -> None:
        all_names = [s.name for s in self.store.all()]
        i0_names = [s.name for s in self.store.all() if s.kind in ("I0", "fit", "I0_fit")]
        it_names = [s.name for s in self.store.all() if s.kind == "It"]
        mu_names = [s.name for s in self.store.all() if s.kind == "mu"]

        self.mu_i0_combo.blockSignals(True)
        current_i0 = self.mu_i0_combo.currentText()
        self.mu_i0_combo.clear()
        self.mu_i0_combo.addItems(i0_names)
        if current_i0 in i0_names:
            self.mu_i0_combo.setCurrentText(current_i0)
        self.mu_i0_combo.blockSignals(False)

        self._fill_list(self.mu_it_list, it_names)
        self._fill_list(self.norm_mu_list, mu_names)
        self._fill_list(self.edge_list, all_names)
        self._fill_list(self.analysis_list, all_names)

    @staticmethod
    def _fill_list(listwidget: QListWidget, names: List[str]) -> None:
        selected = {listwidget.item(i).text() for i in range(listwidget.count()) if listwidget.item(i).isSelected()}
        listwidget.clear()
        for n in names:
            item = QListWidgetItem(n)
            listwidget.addItem(item)
            if n in selected:
                item.setSelected(True)

    def _on_selection_changed(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            self.selected_sid = None
            return
        item = self.table.item(rows[0].row(), 0)
        self.selected_sid = item.data(Qt.UserRole)
        self._plot_selected_preview()

    def _on_table_context_menu(self, pos) -> None:
        item = self.table.itemAt(pos)
        if item is None:
            return
        row = item.row()
        sid = self.table.item(row, 0).data(Qt.UserRole)
        self.table.selectRow(row)
        self.selected_sid = sid

        menu = QMenu(self)
        menu.addAction("Rename…", self.rename_selected)
        menu.addAction("Duplicate", self.duplicate_selected)
        menu.addSeparator()
        menu.addAction("Delete", self.delete_selected)
        menu.addSeparator()
        menu.addAction("Export selected as .dat", self.export_athena_dat)
        menu.exec(self.table.viewport().mapToGlobal(pos))

    def rename_selected(self) -> None:
        if self.selected_sid is None:
            return
        sp = self.store.get(self.selected_sid)
        from PySide6.QtWidgets import QInputDialog
        new_name, ok = QInputDialog.getText(self, "Rename", "New name:", text=sp.name)
        if ok and new_name.strip():
            sp.name = new_name.strip()
            self._refresh_all()

    def duplicate_selected(self) -> None:
        if self.selected_sid is None:
            return
        sp = self.store.get(self.selected_sid)
        sp2 = sp.copy(new_name=f"{sp.name}_copy")
        sp2.history.append(Operation("duplicate", {"from": sp.sid}))
        self.store.add(sp2)
        self._refresh_all()

    def delete_selected(self) -> None:
        if self.selected_sid is None:
            return
        sp = self.store.get(self.selected_sid)
        resp = QMessageBox.question(self, "Delete", f"Delete '{sp.name}'?")
        if resp == QMessageBox.Yes:
            self.store.remove(self.selected_sid)
            self.selected_sid = None
            self._refresh_all()
