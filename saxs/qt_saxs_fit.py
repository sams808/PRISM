"""qt_saxs_fit.py - the Fit tab of the SAXS/WAXS workspace (mixed into SaxsWorkspace).

One engine: saxs.core.fit_workflow. Pick the curves, give each its reference (the same material without the minority
component; a regex rule fills many at once) and, for an absolute particle model, its contrast (typed, or computed from
a phase formula + density against the sample composition + density). "Fit selected" runs the workflow (base fits,
model-free reference gate, structural candidates, selection) in a worker thread; the results table lists the retained
model of every curve, the plot shows any candidate with its components and normalised residuals (I or I q^4), the
report gives the gate, every candidate's verdict and the derived physics. Robustness, JSON save / load and the table /
curve exports use the same result object as the headless pipeline (scripts/saxs_pipeline.py).
"""
from __future__ import annotations

import os
import re
from typing import Dict, List, Optional

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSplitter, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from core.qt_widgets import PlotWidget, to_float as _to_float

FAMILY_CHOICES = [("sphere", "spheres (absolute with a contrast)"), ("sphere_py", "spheres + hard-sphere S(q)"),
                  ("teubner_strey", "Teubner-Strey"), ("debye", "Debye (DAB)"), ("unified", "unified level")]
NO_REF = "(none)"
COMP_COLOURS = {"grain": ("0.45", "--"), "flat": ("0.6", ":"), "oz": ("0.6", ":"), "sph": ("#0072B2", "-."),
                "ts": ("#D55E00", "-."), "dab": ("#009E73", "-."), "uni": ("#CC79A7", "-."), "uni2": ("#56B4E9", ":")}


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


class ContrastDialog(QDialog):
    """Particle contrast and stoichiometric ceiling of a phase in a sample (saxs.core.contrast)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Particle contrast")
        lay = QFormLayout(self)
        self.phase_edit = QLineEdit("")
        self.phase_edit.setPlaceholderText("phase formula, e.g. Al2O3")
        self.phase_rho = QLineEdit("")
        self.sample_edit = QLineEdit("")
        self.sample_edit.setPlaceholderText("sample composition, e.g. 50(SiO2)25(Na2O)25(B2O3)")
        self.sample_rho = QLineEdit("")
        lay.addRow("Phase formula", self.phase_edit)
        lay.addRow("Phase density (g/cm³)", self.phase_rho)
        lay.addRow("Sample composition", self.sample_edit)
        lay.addRow("Sample density (g/cm³)", self.sample_rho)
        self.result_label = QLabel("")
        self.result_label.setWordWrap(True)
        lay.addRow(self.result_label)
        calc = QPushButton("Compute")
        calc.clicked.connect(self.compute)
        lay.addRow(calc)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addRow(bb)
        self.contrast: Optional[float] = None
        self.phi_max: Optional[float] = None

    def compute(self) -> None:
        from saxs.core.contrast import phase_contrast_1e10cm2, stoichiometric_ceiling

        try:
            args = (self.phase_edit.text().strip(), float(self.phase_rho.text()), self.sample_edit.text().strip(),
                    float(self.sample_rho.text()))
            self.contrast = phase_contrast_1e10cm2(*args)
            self.phi_max = stoichiometric_ceiling(*args)
        except Exception as exc:
            self.result_label.setText(f"error: {exc}")
            return
        self.result_label.setText(f"contrast = {self.contrast:.3f} x 1e10 cm⁻²; stoichiometric ceiling "
                                  f"phi_max = {self.phi_max:.3f}")


class SaxsFitMixin:
    """Builds the Fit tab onto a SaxsWorkspace (needs ``self.tabs``, ``self.curves``, ``self._curve_by_name``)."""

    # ================================================================== construction
    def _build_fit_tab(self) -> None:
        from saxs.core.fit_workflow import WorkflowOptions

        self._fit_result = None
        self._fit_refs: Dict[str, Optional[str]] = {}
        self._fit_contrast: Dict[str, Optional[float]] = {}
        self._fit_phimax: Dict[str, Optional[float]] = {}
        o = WorkflowOptions()
        tab = QWidget()
        root = QHBoxLayout(tab)
        splitter = QSplitter()
        root.addWidget(splitter)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumWidth(400)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.addWidget(QLabel("Curves (select the ones to fit; include their references)"))
        self.fit_curve_list = QListWidget()
        self.fit_curve_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.fit_curve_list.itemSelectionChanged.connect(self._fit_show_assignments)
        ll.addWidget(self.fit_curve_list, 1)

        box = QGroupBox("Reference of the selected curves")
        g = QGridLayout(box)
        self.fit_ref_combo = QComboBox()
        b = QPushButton("Set")
        b.clicked.connect(self.fit_set_reference)
        g.addWidget(self.fit_ref_combo, 0, 0, 1, 2)
        g.addWidget(b, 0, 2)
        self.fit_rule_pattern = QLineEdit("")
        self.fit_rule_pattern.setPlaceholderText(r"regex on names, e.g. ^(.*)_x\d+$")
        self.fit_rule_replace = QLineEdit("")
        self.fit_rule_replace.setPlaceholderText(r"reference, e.g. \1_x0")
        b2 = QPushButton("Apply rule")
        b2.clicked.connect(self.fit_apply_rule)
        g.addWidget(self.fit_rule_pattern, 1, 0)
        g.addWidget(self.fit_rule_replace, 1, 1)
        g.addWidget(b2, 1, 2)
        ll.addWidget(box)

        box = QGroupBox("Particle contrast (spheres on an absolute scale)")
        g = QGridLayout(box)
        self.fit_contrast_edit = QLineEdit("")
        self.fit_contrast_edit.setPlaceholderText("Δρ (1e10 cm⁻²), blank = relative")
        b = QPushButton("Set")
        b.clicked.connect(self.fit_set_contrast)
        b2 = QPushButton("Compute…")
        b2.clicked.connect(self.fit_compute_contrast)
        g.addWidget(self.fit_contrast_edit, 0, 0)
        g.addWidget(b, 0, 1)
        g.addWidget(b2, 0, 2)
        ll.addWidget(box)
        self.fit_assign_label = QLabel("")
        self.fit_assign_label.setWordWrap(True)
        ll.addWidget(self.fit_assign_label)

        box = QGroupBox("Options")
        g = QGridLayout(box)
        self.fit_edits: Dict[str, QLineEdit] = {}
        fields = [("floor", "weight floor", o.floor), ("dbic_min", "dBIC min", o.dbic_min),
                  ("p_lo", "grain p min", o.grain_p_bounds[0]), ("p_hi", "grain p max", o.grain_p_bounds[1]),
                  ("oz_hi", "OZ ξ max (Å)", o.oz_xi_bounds[1]), ("vis_frac", "visibility fraction", o.visibility_fraction),
                  ("gw_lo", "gate q min (1/Å)", o.gate_window[0]), ("gw_hi", "gate q max (1/Å)", o.gate_window[1]),
                  ("vis_q", "visibility q max", o.visibility_qmax), ("null", "gate peak min (blank = auto)", None)]
        for i, (key, lab, val) in enumerate(fields):
            e = QLineEdit("" if val is None else f"{val:g}")
            e.setMaximumWidth(70)
            self.fit_edits[key] = e
            g.addWidget(QLabel(lab), i // 2, 2 * (i % 2))
            g.addWidget(e, i // 2, 2 * (i % 2) + 1)
        r = len(fields) // 2 + 1
        self.fit_gate_combo = QComboBox()
        self.fit_gate_combo.addItems(["reference gate", "no gate (always offer structures)"])
        g.addWidget(QLabel("gate"), r, 0)
        g.addWidget(self.fit_gate_combo, r, 1, 1, 3)
        self.fit_family_checks: Dict[str, QCheckBox] = {}
        for i, (key, lab) in enumerate(FAMILY_CHOICES):
            c = QCheckBox(lab)
            c.setChecked(True)
            self.fit_family_checks[key] = c
            g.addWidget(c, r + 1 + i // 2, 2 * (i % 2), 1, 2)
        rr = r + 1 + (len(FAMILY_CHOICES) + 1) // 2
        self.fit_second_check = QCheckBox("second (small) population")
        self.fit_second_check.setChecked(True)
        self.fit_smear_check = QCheckBox("resolution smearing (dQ)")
        self.fit_smear_check.setChecked(True)
        g.addWidget(self.fit_second_check, rr, 0, 1, 2)
        g.addWidget(self.fit_smear_check, rr, 2, 1, 2)
        ll.addWidget(box)

        self.fit_run_btn = QPushButton("Fit selected curves")
        self.fit_run_btn.setObjectName("Primary")
        self.fit_run_btn.clicked.connect(self.fit_run)
        ll.addWidget(self.fit_run_btn)
        self.fit_robust_btn = QPushButton("Robustness of the retained models")
        self.fit_robust_btn.clicked.connect(self.fit_robustness)
        ll.addWidget(self.fit_robust_btn)
        row = QHBoxLayout()
        for lab, fn in (("Save…", self.fit_save_json), ("Load…", self.fit_load_json), ("Tables…", self.fit_export_tables),
                        ("Curves…", self.fit_export_curves)):
            bb = QPushButton(lab)
            bb.clicked.connect(fn)
            row.addWidget(bb)
        ll.addLayout(row)
        self.fit_progress = QProgressBar()
        self.fit_progress.setRange(0, 100)
        ll.addWidget(self.fit_progress)
        self.fit_status = QLabel("Select curves, set references, then Fit.")
        self.fit_status.setWordWrap(True)
        ll.addWidget(self.fit_status)
        scroll.setWidget(left)
        splitter.addWidget(scroll)

        right = QSplitter(Qt.Vertical)
        self.fit_table = QTableWidget(0, 7)
        self.fit_table.setHorizontalHeaderLabels(["curve", "reference", "gate", "retained", "chi2_r", "result", "robust"])
        self.fit_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.fit_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.fit_table.itemSelectionChanged.connect(self._fit_on_table_select)
        right.addWidget(self.fit_table)
        mid = QWidget()
        ml = QVBoxLayout(mid)
        crow = QHBoxLayout()
        crow.addWidget(QLabel("model"))
        self.fit_model_combo = QComboBox()
        self.fit_model_combo.currentTextChanged.connect(lambda _t: self._fit_on_model_change())
        crow.addWidget(self.fit_model_combo, 1)
        self.fit_iq4_check = QCheckBox("I q⁴")
        self.fit_iq4_check.toggled.connect(lambda _b: self.fit_plot.request_redraw(self._fit_render))
        crow.addWidget(self.fit_iq4_check)
        ml.addLayout(crow)
        self.fit_plot = PlotWidget(figsize=(7, 4.6))
        ml.addWidget(self.fit_plot, 1)
        right.addWidget(mid)
        lower = QWidget()
        lw = QHBoxLayout(lower)
        self.fit_param_table = QTableWidget(0, 6)
        self.fit_param_table.setHorizontalHeaderLabels(["parameter", "value", "stderr", "min", "max", "at bound"])
        self.fit_param_table.setEditTriggers(QTableWidget.NoEditTriggers)
        lw.addWidget(self.fit_param_table, 1)
        self.fit_report = QPlainTextEdit()
        self.fit_report.setReadOnly(True)
        lw.addWidget(self.fit_report, 1)
        right.addWidget(lower)
        right.setStretchFactor(1, 2)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        self.tabs.addTab(tab, "Fit")

    def _refresh_fit_lists(self) -> None:
        names = [c.name for c in self.curves]
        selected = {i.text() for i in self.fit_curve_list.selectedItems()}
        self.fit_curve_list.clear()
        for n in names:
            self.fit_curve_list.addItem(n)
        for i in range(self.fit_curve_list.count()):
            if self.fit_curve_list.item(i).text() in selected:
                self.fit_curve_list.item(i).setSelected(True)
        current = self.fit_ref_combo.currentText()
        self.fit_ref_combo.clear()
        self.fit_ref_combo.addItems([NO_REF] + names)
        idx = self.fit_ref_combo.findText(current)
        if idx >= 0:
            self.fit_ref_combo.setCurrentIndex(idx)

    # ================================================================== assignments
    def _fit_selected_names(self) -> List[str]:
        return [i.text() for i in self.fit_curve_list.selectedItems()]

    def fit_set_reference(self) -> None:
        ref = self.fit_ref_combo.currentText()
        for n in self._fit_selected_names():
            self._fit_refs[n] = None if (ref == NO_REF or ref == n) else ref
        self._fit_show_assignments()

    def fit_apply_rule(self) -> None:
        pat, rep = self.fit_rule_pattern.text().strip(), self.fit_rule_replace.text()
        if not pat:
            QMessageBox.information(self, "Fit", "Type a regex (and its replacement) first.")
            return
        try:
            rx = re.compile(pat)
        except re.error as exc:
            QMessageBox.warning(self, "Fit", f"Invalid regex: {exc}")
            return
        names = {c.name for c in self.curves}
        n_set = 0
        for n in names:
            if rx.fullmatch(n):
                ref = rx.sub(rep, n)
                if ref != n and ref in names:
                    self._fit_refs[n] = ref
                    n_set += 1
        self.fit_status.setText(f"Rule applied: {n_set} reference(s) set.")
        self._fit_show_assignments()

    def fit_set_contrast(self) -> None:
        v = _to_float(self.fit_contrast_edit.text())
        for n in self._fit_selected_names():
            self._fit_contrast[n] = v
        self._fit_show_assignments()

    def fit_compute_contrast(self) -> None:
        dlg = ContrastDialog(self)
        if dlg.exec() and dlg.contrast is not None:
            self.fit_contrast_edit.setText(f"{dlg.contrast:.4g}")
            for n in self._fit_selected_names():
                self._fit_contrast[n] = dlg.contrast
                self._fit_phimax[n] = dlg.phi_max
            self._fit_show_assignments()

    def _fit_show_assignments(self) -> None:
        names = self._fit_selected_names()
        if not names:
            self.fit_assign_label.setText("")
            return
        lines = [f"{n} -> ref {self._fit_refs.get(n) or '-'}, contrast {_fmt(self._fit_contrast.get(n), 4) or '-'}" for n in names[:8]]
        if len(names) > 8:
            lines.append(f"... {len(names) - 8} more")
        self.fit_assign_label.setText("\n".join(lines))

    # ================================================================== options and inputs
    def _fit_options(self):
        from saxs.core.fit_workflow import WorkflowOptions

        o = WorkflowOptions()
        v = {k: _to_float(e.text()) for k, e in self.fit_edits.items()}
        pick = lambda key, default: default if v.get(key) is None else v[key]      # noqa: E731
        fams = tuple(k for k, c in self.fit_family_checks.items() if c.isChecked())
        return WorkflowOptions(
            floor=pick("floor", o.floor), dbic_min=pick("dbic_min", o.dbic_min),
            grain_p_bounds=(pick("p_lo", o.grain_p_bounds[0]), pick("p_hi", o.grain_p_bounds[1])),
            oz_xi_bounds=(o.oz_xi_bounds[0], pick("oz_hi", o.oz_xi_bounds[1])),
            visibility_fraction=pick("vis_frac", o.visibility_fraction),
            gate_window=(pick("gw_lo", o.gate_window[0]), pick("gw_hi", o.gate_window[1])),
            visibility_qmax=pick("vis_q", o.visibility_qmax), gate_null_max=v.get("null"),
            gate="reference" if self.fit_gate_combo.currentIndex() == 0 else "none",
            families=fams or o.families, second_population=self.fit_second_check.isChecked(),
            smear=self.fit_smear_check.isChecked())

    def _fit_inputs(self, names: List[str]):
        """CurveInputs of the named curves plus their references (added automatically when missing)."""
        from saxs.core.fit_workflow import CurveInput

        wanted = list(names)
        for n in names:
            r = self._fit_refs.get(n)
            if r and r not in wanted:
                wanted.append(r)
        inputs = []
        for n in wanted:
            c = self._curve_by_name(n)
            if c is None:
                raise ValueError(f"curve {n!r} is not loaded")
            ref = self._fit_refs.get(n)
            inputs.append(CurveInput.from_curve(c, name=n, reference=ref if ref in wanted else None,
                                                contrast=self._fit_contrast.get(n), phi_max=self._fit_phimax.get(n)))
        return inputs

    # ================================================================== actions
    def fit_run(self) -> None:
        from core.qt_worker import run_in_thread
        from saxs.core.fit_workflow import run_workflow

        names = self._fit_selected_names()
        if not names:
            QMessageBox.warning(self, "Fit", "Select at least one curve.")
            return
        try:
            inputs = self._fit_inputs(names)
            opts = self._fit_options()
        except Exception as exc:
            QMessageBox.warning(self, "Fit", str(exc))
            return
        self._fit_last_inputs = inputs
        self.fit_run_btn.setEnabled(False)
        self.fit_progress.setValue(0)
        run_in_thread(lambda progress: run_workflow(inputs, opts, progress=progress), self._fit_on_done, self._fit_on_error,
                      on_progress=lambda i, n, m: self._fit_set_progress(i, n, m))

    def _fit_set_progress(self, i: int, n: int, msg: str) -> None:
        self.fit_progress.setValue(int(100 * i / max(n, 1)))
        self.fit_status.setText(msg)

    def _fit_on_error(self, traceback_text: str) -> None:
        self.fit_run_btn.setEnabled(True)
        self.fit_robust_btn.setEnabled(True)
        QMessageBox.critical(self, "Fit", traceback_text)

    def _fit_on_done(self, result) -> None:
        self.fit_run_btn.setEnabled(True)
        self.fit_robust_btn.setEnabled(True)
        self.fit_progress.setValue(100)
        self._fit_result = result
        n_s = sum(c.structural for c in result.curves.values())
        nm = f"; reference/reference null {result.null_max:.3g}" if np.isfinite(result.null_max) else ""
        self.fit_status.setText(f"{len(result.curves)} curve(s) fitted, {n_s} with a retained structure{nm}.")
        self._fit_fill_table()
        if self.fit_table.rowCount():
            self.fit_table.selectRow(0)

    def fit_robustness(self) -> None:
        from core.qt_worker import run_in_thread
        from saxs.core.fit_workflow import robustness

        if self._fit_result is None:
            QMessageBox.information(self, "Fit", "Fit curves first.")
            return
        try:
            inputs = self._fit_inputs(list(self._fit_result.curves))
        except Exception as exc:
            QMessageBox.warning(self, "Fit", str(exc))
            return
        res = self._fit_result
        self.fit_robust_btn.setEnabled(False)
        run_in_thread(lambda progress: (robustness(res, inputs, progress=progress), res)[1], self._fit_on_done, self._fit_on_error,
                      on_progress=lambda i, n, m: self._fit_set_progress(i, n, m))

    # ================================================================== results display
    @staticmethod
    def _fit_result_text(cr) -> str:
        d = cr.derived
        parts = []
        if "ts_d_nm" in d:
            parts.append(f"TS d {d['ts_d_nm']:.1f} nm, ξ {d['ts_xi_nm']:.1f} nm")
        if "sph_Dn_nm" in d:
            parts.append(f"spheres D {d['sph_Dn_nm']:.1f} nm" + (f", φ {d['sph_phi']:.3g}" if "sph_phi" in d else ""))
        if "dab_xi_nm" in d:
            parts.append(f"Debye ξ {d['dab_xi_nm']:.2f} nm")
        if "uni_Rg_nm" in d:
            parts.append(f"unified Rg {d['uni_Rg_nm']:.1f} nm, d {d['uni_d']:.1f}")
        if "uni2_Rg_nm" in d:
            parts.append(f"small pop. Rg {d['uni2_Rg_nm']:.1f} nm")
        if not parts:
            parts.append(f"grain p {d.get('grain_p', np.nan):.2f}, high-q {d.get('highq_C', np.nan):.3g}")
        return "; ".join(parts)

    def _fit_fill_table(self) -> None:
        from saxs.core.fit_workflow import robust_flags

        res = self._fit_result
        t = self.fit_table
        t.setRowCount(len(res.curves))
        for i, (name, cr) in enumerate(res.curves.items()):
            g = cr.gate
            gate = g.method + (" PASS" if g.passed and g.method == "reference" else "")
            if np.isfinite(g.peak):
                gate += f" ({g.peak:.2f})"
            fl = robust_flags(res.robustness, name)
            rob = "" if not fl else ("yes" if fl.get("robust_primary") else "no")
            for c, text in enumerate((name, cr.reference or "", gate, cr.retained, _fmt(cr.retained_fit.redchi, 3),
                                      self._fit_result_text(cr), rob)):
                t.setItem(i, c, _item(text))
        t.resizeColumnsToContents()

    def _fit_current(self):
        if self._fit_result is None:
            return None
        rows = sorted({i.row() for i in self.fit_table.selectedIndexes()})
        if not rows:
            return None
        it = self.fit_table.item(rows[0], 0)
        return self._fit_result.curves.get(it.text()) if it is not None else None

    def _fit_on_table_select(self) -> None:
        cr = self._fit_current()
        if cr is None:
            return
        self.fit_model_combo.blockSignals(True)
        self.fit_model_combo.clear()
        keys = [cr.retained] + [k for k in cr.fits if k != cr.retained and not isinstance(cr.fits[k], str)]
        self.fit_model_combo.addItems(keys)
        self.fit_model_combo.blockSignals(False)
        self.fit_report.setPlainText(self._fit_report_text(cr))
        self._fit_on_model_change()

    def _fit_selected_fit(self):
        """(curve result, model key, fit) of the selected row and model; the fit is None for a failed candidate."""
        cr = self._fit_current()
        if cr is None:
            return None, "", None
        key = self.fit_model_combo.currentText() or cr.retained
        f = cr.fits.get(key)
        return cr, key, (None if f is None or isinstance(f, str) else f)

    def _fit_on_model_change(self) -> None:
        _cr, _key, f = self._fit_selected_fit()
        if f is None:
            self.fit_param_table.setRowCount(0)
        else:
            self._fit_fill_params(f)
        self.fit_plot.request_redraw(self._fit_render)

    def _fit_render(self) -> None:
        cr, key, f = self._fit_selected_fit()
        fig = self.fit_plot.figure
        for a in list(fig.axes):       # clearing shared log axes one by one resets the sibling to a non-positive range
            fig.delaxes(a)
        fig.clf()
        if f is None:
            self.fit_plot.canvas.draw_idle()
            return
        iq4 = self.fit_iq4_check.isChecked()
        w = f.q ** 4 if iq4 else np.ones_like(f.q)
        ax = fig.add_subplot(211)
        axr = fig.add_subplot(212, sharex=ax)
        ax.loglog(f.q, f.I * w, ".", ms=3, color="black", label=cr.name)
        ax.loglog(f.q, f.model * w, color="crimson", lw=1.5, label=f"{key}  χ²r {f.redchi:.2f}" + ("  (retained)" if key == cr.retained else ""))
        from saxs.core.series_figures import component_label
        for ck, cv in f.components.items():
            col, ls = COMP_COLOURS.get(ck, ("0.5", ":"))
            ax.loglog(f.q, np.clip(cv * w, 1e-300, None), color=col, ls=ls, lw=1.0, label=component_label(f, ck))
        yy = f.I * w
        ax.set_ylim(np.nanmin(yy[yy > 0]) / 5, np.nanmax(yy) * 3)
        ax.legend(fontsize=7)
        ax.set_ylabel("I q⁴" if iq4 else "I")
        ax.grid(alpha=0.25, which="both")
        ax.tick_params(labelbottom=False)
        axr.axhspan(-1, 1, color="0.92", lw=0)
        axr.semilogx(f.q, f.residual, ".", ms=3, color="steelblue")
        axr.axhline(0, color="0.5", lw=0.8)
        axr.set_ylim(-5, 5)
        axr.set_xlabel("q (Å⁻¹)")
        axr.set_ylabel("residual (σ)")
        axr.grid(alpha=0.25, which="both")
        fig.tight_layout()
        self.fit_plot.canvas.draw_idle()

    def _fit_fill_params(self, f) -> None:
        t = self.fit_param_table
        rows = [(k, p) for k, p in f.params.items()]
        t.setRowCount(len(rows))
        for i, (k, p) in enumerate(rows):
            cells = (k, _fmt(p.value, 5), _fmt(p.stderr, 3), _fmt(p.lo, 4) if p.vary else "fixed", _fmt(p.hi, 4) if p.vary else "",
                     "yes" if (p.vary and p.at_bound) else "")
            for c, text in enumerate(cells):
                t.setItem(i, c, _item(text))
        t.resizeColumnsToContents()

    @staticmethod
    def _fit_report_text(cr) -> str:
        g = cr.gate
        lines = [f"{cr.name}: retained {cr.retained} on base {cr.base}; reference {cr.reference or '-'}"]
        if g.method == "reference":
            run = f"run x{g.run.span:.2f} at q {g.run.q_lo:.4g}-{g.run.q_hi:.4g}" if g.run else "no excess run"
            lines.append(f"Gate (ratio to reference): {'PASS' if g.passed else 'no structure'} - peak {g.peak:.3g} at q {g.q_peak:.4g} "
                         f"(threshold {g.null_max:.3g}), {run}" + (f"; {g.note}" if g.note else ""))
        else:
            lines.append(f"Gate: {g.method} ({g.note})")
        lines.append(f"Residual run of the base (diagnostic only): x{cr.fit_gate_span:.2f}")
        if cr.reasons:
            lines.append("Candidates:")
            for k, why in cr.reasons.items():
                f = cr.fits.get(k)
                bic = f" BIC {f.bic:.1f}" if f is not None and not isinstance(f, str) else ""
                lines.append(f"  {k:14s}{bic}: {'; '.join(why)}")
        lines.append("Derived:")
        for k, v in cr.derived.items():
            if isinstance(v, (float, int, np.floating)) and not isinstance(v, bool) and np.isfinite(v):
                lines.append(f"  {k} = {v:.4g}")
        if cr.phi_max:
            lines.append(f"  stoichiometric ceiling phi_max = {cr.phi_max:.3g}")
        return "\n".join(lines)

    # ================================================================== io
    def fit_save_json(self) -> None:
        if self._fit_result is None:
            QMessageBox.information(self, "Fit", "Fit curves first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save the fit results", "fits.json", "JSON (*.json)")
        if path:
            self._fit_result.save(path)
            self.fit_status.setText(f"Saved {os.path.basename(path)}.")

    def fit_load_json(self) -> None:
        from saxs.core.fit_workflow import WorkflowResult

        path, _ = QFileDialog.getOpenFileName(self, "Load fit results", "", "JSON (*.json)")
        if not path:
            return
        try:
            res = WorkflowResult.load(path)
        except Exception as exc:
            QMessageBox.critical(self, "Fit", str(exc))
            return
        for name, cr in res.curves.items():
            self._fit_refs[name] = cr.reference
            self._fit_contrast[name] = cr.contrast
            self._fit_phimax[name] = cr.phi_max
        self._fit_on_done(res)

    def fit_export_tables(self) -> None:
        from saxs.core.tables import write_workflow_tables

        if self._fit_result is None:
            QMessageBox.information(self, "Fit", "Fit curves first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export the tables (base name)", "fit_tables", "All files (*.*)")
        if not path:
            return
        base = os.path.splitext(path)[0] if os.path.splitext(path)[1].lower().lstrip(".") in ("xlsx", "csv", "md") else path
        paths = write_workflow_tables(self._fit_result, base)
        self.fit_status.setText("Tables: " + ", ".join(os.path.basename(p) for p in paths.values()))

    def fit_export_curves(self) -> None:
        from saxs.core.fit_workflow import curve_table
        from saxs.core.tables import transliterate_frame

        if self._fit_result is None:
            QMessageBox.information(self, "Fit", "Fit curves first.")
            return
        folder = QFileDialog.getExistingDirectory(self, "Folder for the per-curve fit files")
        if not folder:
            return
        for name, cr in self._fit_result.curves.items():
            transliterate_frame(curve_table(cr)).to_csv(os.path.join(folder, f"{name}_fit.csv"), index=False, encoding="utf-8-sig")
        self.fit_status.setText(f"{len(self._fit_result.curves)} curve file(s) -> {folder}")
