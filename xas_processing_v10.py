"""
xas_gui_ultimate.py

Standalone, single-file GUI for processing EasyXAFS XAS/XANES/EXAFS data with Athena interoperability.
All processing/science logic lives in xas_science.py (framework-agnostic, tested) — this file is
purely the Tkinter presentation layer.

Run:
    python xas_processing_v10.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    matplotlib.use("Agg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from xas_science import (
    _SCIPY_AVAILABLE,
    _interp_to_grid,
    smooth_spectrum,
    fit_chebyshev,
    fit_spline,
    whittaker_smooth,
    mu_from_transmission,
    build_mu,
    _periodic_table_symbols,
    angle_energy_correction_bragg,
    TiePoint,
    apply_alignment_mode_c,
    infer_edge_label_from_roi_scaled,
    edge_text,
    require_larch,
    larch_normalize,
    larch_exafs_pipeline,
    Operation,
    Spectrum,
    SpectrumStore,
    _uid,
    _extract_energy_angle_signal,
    _classify_kind_from_name,
    read_easyxafs_zip,
    read_csv_dataset,
    read_athena_prj,
    export_athena_column,
    export_athena_prj_best_effort,
)


# ---------------------------- UI Components ----------------------------

class PlotPanel:
    def __init__(self, parent):
        self.frame = ttk.Frame(parent)
        self.fig = Figure(figsize=(7.2, 5.0), dpi=110)
        self.ax = self.fig.add_subplot(111); self.ax.grid(alpha=0.25)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.frame); self.toolbar.update()

    def clear(self, title: str = ""):
        self.ax.clear(); self.ax.grid(alpha=0.25)
        if title: self.ax.set_title(title)
        self.canvas.draw_idle()

    def plot(self, xs: List[np.ndarray], ys: List[np.ndarray], labels: List[str], xlabel: str, ylabel: str, title: str):
        self.ax.clear()
        for x, y, lab in zip(xs, ys, labels): self.ax.plot(x, y, lw=1.3, label=lab)
        self.ax.set_xlabel(xlabel); self.ax.set_ylabel(ylabel); self.ax.set_title(title)
        self.ax.grid(alpha=0.25)
        if any(labels): self.ax.legend(loc="best")
        self.fig.tight_layout(); self.canvas.draw_idle()

def simple_input(parent, title: str, prompt: str, default: str = "") -> Optional[str]:
    win = tk.Toplevel(parent); win.title(title); win.grab_set()
    ttk.Label(win, text=prompt).pack(padx=10, pady=(10,4))
    var = tk.StringVar(value=default); ent = ttk.Entry(win, textvariable=var, width=40)
    ent.pack(padx=10, pady=(0,10)); ent.focus_set()
    out = {"val": None}
    def ok(): out["val"] = var.get().strip(); win.destroy()
    def cancel(): out["val"] = None; win.destroy()
    btn = ttk.Frame(win); btn.pack(padx=10, pady=(0,10), fill="x")
    ttk.Button(btn, text="OK", command=ok).pack(side="left", expand=True, fill="x")
    ttk.Button(btn, text="Cancel", command=cancel).pack(side="left", expand=True, fill="x", padx=(6,0))
    win.bind("<Return>", lambda e: ok()); win.bind("<Escape>", lambda e: cancel())
    parent.wait_window(win)
    return out["val"]

def show_text_window(parent, title: str, text: str):
    win = tk.Toplevel(parent); win.title(title); win.geometry("900x650")
    txt = tk.Text(win, wrap="none"); txt.insert("1.0", text); txt.configure(state="disabled")
    txt.pack(fill="both", expand=True)
    xscroll = ttk.Scrollbar(win, orient="horizontal", command=txt.xview)
    yscroll = ttk.Scrollbar(win, orient="vertical", command=txt.yview)
    txt.configure(xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
    xscroll.pack(side="bottom", fill="x"); yscroll.pack(side="right", fill="y")


# ---------------------------- Main App ----------------------------

class XASUltimateApp(tk.Tk):
    def __init__(self, *, initial_records: Optional[List[Dict[str, Any]]] = None, allow_import: bool = False):
        super().__init__()
        self.title("XAS Ultimate GUI (EasyXAFS ↔ Athena)")
        self.geometry("1500x900")
        self.store = SpectrumStore()
        self.allow_import = bool(allow_import)
        self.selected_sid: Optional[str] = None
        self.status_var = tk.StringVar(value="Ready.")
        self.tiepoints: List[TiePoint] = []
        self._pick_state = {"active": False, "waiting": "before", "before_line": None, "after_line": None, "e_before": None}
        self._pick_line_map = {}
        self._pick_markers = []
        self._mode_c_click_state = {"waiting": "before", "last_before": None}
        self._mode_c_markers: List[Tuple[str, float, float]] = []
        self._fit_last_preview = None

        self._build_ui()
        if initial_records:
            self.load_initial_records(initial_records)

    def _build_ui(self):
        root = ttk.Frame(self, padding=8); root.pack(fill="both", expand=True)
        left = ttk.Frame(root); left.pack(side="left", fill="y", padx=(0,8))
        right = ttk.Frame(root); right.pack(side="left", fill="both", expand=True)

        if self.allow_import:
            lf_import = ttk.LabelFrame(left, text="Import", padding=8); lf_import.pack(fill="x", pady=(0,8))
            ttk.Button(lf_import, text="Import ZIP(s)...", command=self.ui_import_zips).pack(fill="x")
            ttk.Button(lf_import, text="Import CSV(s)...", command=self.ui_import_csvs).pack(fill="x", pady=(6,0))
            ttk.Button(lf_import, text="Import Athena .prj...", command=self.ui_import_prj).pack(fill="x", pady=(6,0))
            ttk.Button(lf_import, text="Clear", command=self.ui_clear).pack(fill="x", pady=(6,0))

        lf_list = ttk.LabelFrame(left, text="Imported spectra (objects)", padding=8); lf_list.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(lf_list, columns=("name","kind","label","e0","erange","source"), show="headings", height=24)
        for c, t, w in [("name","Name",170),("kind","Type",70),("label","Edge",130),("e0","E0",70),("erange","E range",120),("source","Source",180)]:
            self.tree.heading(c, text=t); self.tree.column(c, width=w, anchor="w" if c in ("name","label","source") else "e")
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select); self.tree.bind("<Button-3>", self.on_tree_right_click)

        ttk.Label(left, textvariable=self.status_var, wraplength=330, justify="left").pack(fill="x", pady=(8,0))

        self.nb = ttk.Notebook(right); self.nb.pack(fill="both", expand=True)
        for tab_name, attr in [("Preview", "tab_preview"), ("Pre-processing", "tab_preproc"), ("Fit (I0 etc.)", "tab_fit"),
                               ("μ(E) Builder", "tab_mu"), ("Normalization (Larch)", "tab_norm"), ("Tools", "tab_tools"), ("Export", "tab_export")]:
            frame = ttk.Frame(self.nb); setattr(self, attr, frame); self.nb.add(frame, text=tab_name)

        self._build_preview_tab(); self._build_preproc_tab(); self._build_fit_tab()
        self._build_mu_tab(); self._build_norm_tab(); self._build_tools_tab(); self._build_export_tab()

    def _build_preview_tab(self):
        self.preview_plot = PlotPanel(self.tab_preview)
        self.preview_plot.frame.pack(fill="both", expand=True)

    def _build_preproc_tab(self):
        parent = self.tab_preproc
        top = ttk.Frame(parent); top.pack(fill="x", padx=8, pady=8)
        mid = ttk.Frame(parent); mid.pack(fill="both", expand=True, padx=8, pady=(0,8))

        ctrl = ttk.LabelFrame(top, text="Pre-processing", padding=8); ctrl.pack(side="left", fill="x", expand=True)

        ttk.Label(ctrl, text="Smoothing target").grid(row=0, column=0, sticky="w")
        self.var_sm_target = tk.StringVar(value="")
        self.cb_sm_target = ttk.Combobox(ctrl, textvariable=self.var_sm_target, values=[], state="readonly", width=34)
        self.cb_sm_target.grid(row=0, column=1, sticky="w", padx=(6,0)); self.cb_sm_target.bind("<<ComboboxSelected>>", lambda e: self.ui_preview_smoothing())

        ttk.Label(ctrl, text="Smoothing method").grid(row=1, column=0, sticky="w", pady=(6,0))
        self.var_sm_method = tk.StringVar(value="Savitzky-Golay")
        self.cb_sm = ttk.Combobox(ctrl, textvariable=self.var_sm_method, values=["Savitzky-Golay","Median+SG","Whittaker"] + (["Spline"] if _SCIPY_AVAILABLE else []), state="readonly", width=18)
        self.cb_sm.grid(row=1, column=1, sticky="w", padx=(6,0), pady=(6,0)); self.cb_sm.bind("<<ComboboxSelected>>", lambda e: (self._build_sm_params(), self.ui_preview_smoothing()))

        self.var_sm_autopreview = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, text="Auto preview smoothing", variable=self.var_sm_autopreview, command=self.ui_preview_smoothing).grid(row=2, column=0, columnspan=2, sticky="w", pady=(6,0))

        self.sm_params_frame = ttk.Frame(ctrl); self.sm_params_frame.grid(row=3, column=0, columnspan=2, sticky="we", pady=(6,0))
        self._build_sm_params()

        btns_sm = ttk.Frame(ctrl); btns_sm.grid(row=4, column=0, columnspan=2, sticky="we", pady=(8,0))
        ttk.Button(btns_sm, text="Preview smoothing", command=self.ui_preview_smoothing).pack(side="left", fill="x", expand=True)
        ttk.Button(btns_sm, text="Apply smoothing -> new object", command=self.ui_apply_smoothing).pack(side="left", fill="x", expand=True, padx=(8,0))

        ttk.Separator(ctrl, orient="horizontal").grid(row=5, column=0, columnspan=2, sticky="we", pady=10)

        ttk.Label(ctrl, text="Angle/E correction mode").grid(row=6, column=0, sticky="w")
        self.var_ang_mode = tk.StringVar(value="A: Bragg+Linear")
        self.cb_ang = ttk.Combobox(ctrl, textvariable=self.var_ang_mode, values=["A: Bragg+Linear","B: Bragg only","C: Feature alignment (click)"], state="readonly", width=24)
        self.cb_ang.grid(row=6, column=1, sticky="w", padx=(6,0))

        ttk.Label(ctrl, text="Before spectrum").grid(row=7, column=0, sticky="w", pady=(6,0))
        self.var_ang_before = tk.StringVar(value="")
        self.cb_ang_before = ttk.Combobox(ctrl, textvariable=self.var_ang_before, values=[], state="readonly", width=34)
        self.cb_ang_before.grid(row=7, column=1, sticky="w", padx=(6,0), pady=(6,0)); self.cb_ang_before.bind("<<ComboboxSelected>>", lambda e: self.ui_plot_mode_c_overlay())

        ttk.Label(ctrl, text="After spectrum").grid(row=8, column=0, sticky="w", pady=(6,0))
        self.var_ang_after = tk.StringVar(value="")
        self.cb_ang_after = ttk.Combobox(ctrl, textvariable=self.var_ang_after, values=[], state="readonly", width=34)
        self.cb_ang_after.grid(row=8, column=1, sticky="w", padx=(6,0), pady=(6,0)); self.cb_ang_after.bind("<<ComboboxSelected>>", lambda e: self.ui_plot_mode_c_overlay())

        self.var_fit_linear = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, text="Fit linear calibration (Mode A)", variable=self.var_fit_linear).grid(row=9, column=0, columnspan=2, sticky="w", pady=(6,0))

        ttk.Label(ctrl, text="Mode C model").grid(row=10, column=0, sticky="w", pady=(6,0))
        self.var_c_model = tk.StringVar(value="shift")
        ttk.Combobox(ctrl, textvariable=self.var_c_model, values=["shift","affine"], state="readonly", width=10).grid(row=10, column=1, sticky="w", padx=(6,0), pady=(6,0))

        btns_ang = ttk.Frame(ctrl); btns_ang.grid(row=11, column=0, columnspan=2, sticky="we", pady=(8,0))
        ttk.Button(btns_ang, text="Plot overlay", command=self.ui_plot_mode_c_overlay).pack(side="left", fill="x", expand=True)
        ttk.Button(btns_ang, text="Start picking (1 pair)", command=self.ui_start_picking_pair).pack(side="left", fill="x", expand=True, padx=(8,0))
        ttk.Button(btns_ang, text="Apply correction -> new object", command=self.ui_apply_angle_correction).pack(side="left", fill="x", expand=True, padx=(8,0))

        right = ttk.Frame(top); right.pack(side="left", fill="both", expand=True, padx=(8,0))
        self.preproc_plot = PlotPanel(right); self.preproc_plot.frame.pack(fill="both", expand=True)
        self.preproc_plot.canvas.mpl_connect("button_press_event", self.on_preproc_click)

        tp_frame = ttk.LabelFrame(mid, text="Mode C: feature alignment tie points", padding=8); tp_frame.pack(fill="both", expand=True)
        ttk.Label(tp_frame, text="Athena-like selection: click near a feature. We snap to nearest point.\nWorkflow: plot overlay → start picking → click BEFORE then AFTER.", justify="left").pack(anchor="w")
        ttk.Button(tp_frame, text="Clear tie points", command=self.ui_clear_tiepoints).pack(anchor="w", pady=(6,0))

        self.tp_tree = ttk.Treeview(tp_frame, columns=("e_before","e_after","dE"), show="headings", height=7)
        for c, t, w in [("e_before","E before (eV)",120),("e_after","E after (eV)",120),("dE","ΔE (eV)",90)]:
            self.tp_tree.heading(c, text=t); self.tp_tree.column(c, width=w, anchor="e")
        self.tp_tree.pack(fill="x", pady=(6,0)); self.tp_tree.bind("<Button-3>", self.on_tp_right_click)

    def _build_fit_tab(self):
        parent = self.tab_fit
        top = ttk.Frame(parent); top.pack(fill="x", padx=8, pady=8)
        mid = ttk.Frame(parent); mid.pack(fill="both", expand=True, padx=8, pady=(0,8))

        ctrl = ttk.LabelFrame(top, text="Fit settings", padding=8); ctrl.pack(side="left", fill="x", expand=True)

        ttk.Label(ctrl, text="Fit method").grid(row=0, column=0, sticky="w")
        self.var_fit_method = tk.StringVar(value="Chebyshev")
        self.cb_fit = ttk.Combobox(ctrl, textvariable=self.var_fit_method, values=["Chebyshev","Whittaker"] + (["Spline"] if _SCIPY_AVAILABLE else []), state="readonly", width=14)
        self.cb_fit.grid(row=0, column=1, sticky="w", padx=(6,0)); self.cb_fit.bind("<<ComboboxSelected>>", lambda e: self._build_fit_params())

        self.fit_params_frame = ttk.Frame(ctrl); self.fit_params_frame.grid(row=1, column=0, columnspan=2, sticky="we", pady=(6,0))

        self.var_fit_help = tk.StringVar(value="")
        ttk.Label(ctrl, textvariable=self.var_fit_help, wraplength=360, justify="left").grid(row=2, column=0, columnspan=2, sticky="w", pady=(10,0))

        self._build_fit_params()

        ttk.Label(ctrl, text="Mask ranges (e.g. '7100,7120')").grid(row=4, column=0, sticky="nw", pady=(10,0))
        self.txt_fit_mask = tk.Text(ctrl, width=20, height=3)
        self.txt_fit_mask.grid(row=4, column=1, sticky="w", pady=(10,0))

        ttk.Button(ctrl, text="Preview fit", command=self.ui_preview_fit).grid(row=5, column=0, columnspan=2, sticky="we", pady=(10,0))
        ttk.Button(ctrl, text="Save fit -> new object", command=self.ui_save_fit).grid(row=6, column=0, columnspan=2, sticky="we", pady=(6,0))

        self.fit_plot = PlotPanel(top); self.fit_plot.frame.pack(side="left", fill="both", expand=True, padx=(8,0))
        self.fit_resid_plot = PlotPanel(mid); self.fit_resid_plot.frame.pack(fill="both", expand=True)

    def _build_mu_tab(self):
        parent = self.tab_mu
        top = ttk.Frame(parent); top.pack(fill="x", padx=8, pady=8)
        bottom = ttk.Frame(parent); bottom.pack(fill="both", expand=True, padx=8, pady=(0,8))

        ctrl = ttk.LabelFrame(top, text="μ(E) builder", padding=8); ctrl.pack(side="left", fill="x", expand=True)

        ttk.Label(ctrl, text="I0 selection").grid(row=0, column=0, sticky="w")
        self.var_i0_single = tk.StringVar(value="")
        self.cb_i0_single = ttk.Combobox(ctrl, textvariable=self.var_i0_single, values=[], state="readonly", width=34)
        self.cb_i0_single.grid(row=0, column=1, sticky="w", pady=(0,0), padx=(6,0)); self.cb_i0_single.bind("<<ComboboxSelected>>", lambda e: self.ui_preview_mu())

        ttk.Label(ctrl, text="log").grid(row=2, column=0, sticky="w", pady=(6,0))
        self.var_log = tk.StringVar(value="ln")
        ttk.Combobox(ctrl, textvariable=self.var_log, values=["ln","log10"], state="readonly", width=8).grid(row=2, column=1, sticky="w", pady=(6,0), padx=(6,0))

        self.var_mu_autopreview = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, text="Auto preview μ", variable=self.var_mu_autopreview, command=self.ui_preview_mu).grid(row=3, column=0, columnspan=2, sticky="w", pady=(10,0))

        btns = ttk.Frame(ctrl); btns.grid(row=4, column=0, columnspan=2, sticky="we", pady=(10,0))
        ttk.Button(btns, text="Preview μ", command=self.ui_preview_mu).pack(side="left", fill="x", expand=True)
        ttk.Button(btns, text="Compute μ -> new objects", command=self.ui_compute_mu).pack(side="left", fill="x", expand=True, padx=(8,0))

        it_frame = ttk.LabelFrame(top, text="Select It spectra", padding=8); it_frame.pack(side="left", fill="both", expand=True, padx=(8,0))
        self.it_listbox = tk.Listbox(it_frame, selectmode=tk.EXTENDED, height=10); self.it_listbox.pack(fill="both", expand=True)
        self.it_listbox.bind("<<ListboxSelect>>", lambda e: self._maybe_autopreview_mu())

        self.mu_plot = PlotPanel(bottom); self.mu_plot.frame.pack(fill="both", expand=True)

    def _build_norm_tab(self):
        parent = self.tab_norm
        top = ttk.Frame(parent); top.pack(fill="x", padx=8, pady=8)
        bottom = ttk.Frame(parent); bottom.pack(fill="both", expand=True, padx=8, pady=(0,8))

        ctrl = ttk.LabelFrame(top, text="Normalization settings (Larch pre_edge)", padding=8); ctrl.pack(side="left", fill="x", expand=True)

        ttk.Label(ctrl, text="E0 method").grid(row=0, column=0, sticky="w")
        self.var_e0_method = tk.StringVar(value="larch")
        ttk.Combobox(ctrl, textvariable=self.var_e0_method, values=["larch","deriv","manual"], state="readonly", width=10).grid(row=0, column=1, sticky="w", padx=(6,0))
        ttk.Label(ctrl, text="E0 manual").grid(row=0, column=2, sticky="w", padx=(12,0))
        self.ent_e0_manual = ttk.Entry(ctrl, width=10); self.ent_e0_manual.grid(row=0, column=3, sticky="w", padx=(6,0))

        ttk.Label(ctrl, text="pre1 (ΔeV)").grid(row=1, column=0, sticky="w", pady=(6,0))
        self.ent_pre1 = ttk.Entry(ctrl, width=8); self.ent_pre1.insert(0, "-150"); self.ent_pre1.grid(row=1, column=1, sticky="w", pady=(6,0), padx=(6,0))
        ttk.Label(ctrl, text="pre2 (ΔeV)").grid(row=1, column=2, sticky="w", pady=(6,0), padx=(12,0))
        self.ent_pre2 = ttk.Entry(ctrl, width=8); self.ent_pre2.insert(0, "-50"); self.ent_pre2.grid(row=1, column=3, sticky="w", pady=(6,0), padx=(6,0))

        ttk.Label(ctrl, text="norm1 (ΔeV)").grid(row=2, column=0, sticky="w", pady=(6,0))
        self.ent_norm1 = ttk.Entry(ctrl, width=8); self.ent_norm1.insert(0, "30"); self.ent_norm1.grid(row=2, column=1, sticky="w", pady=(6,0), padx=(6,0))
        ttk.Label(ctrl, text="norm2 (ΔeV)").grid(row=2, column=2, sticky="w", pady=(6,0), padx=(12,0))
        self.ent_norm2 = ttk.Entry(ctrl, width=8); self.ent_norm2.insert(0, "150"); self.ent_norm2.grid(row=2, column=3, sticky="w", pady=(6,0), padx=(6,0))

        ttk.Label(ctrl, text="nnorm").grid(row=3, column=0, sticky="w", pady=(6,0))
        self.ent_nnorm = ttk.Entry(ctrl, width=8); self.ent_nnorm.insert(0, "0"); self.ent_nnorm.grid(row=3, column=1, sticky="w", pady=(6,0), padx=(6,0))

        self.var_norm_smooth = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, text="Smooth for E0/derivative only", variable=self.var_norm_smooth).grid(row=4, column=0, columnspan=2, sticky="w", pady=(10,0))
        ttk.Label(ctrl, text="Smooth method").grid(row=5, column=0, sticky="w", pady=(6,0))
        self.var_norm_sm = tk.StringVar(value="Savitzky-Golay")
        ttk.Combobox(ctrl, textvariable=self.var_norm_sm, values=["Savitzky-Golay","Median+SG","Whittaker"], state="readonly", width=18).grid(row=5, column=1, sticky="w", pady=(6,0), padx=(6,0))

        self.var_show_norm_anchors = tk.BooleanVar(value=True); self.var_show_norm_baselines = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, text="Show anchor points", variable=self.var_show_norm_anchors).grid(row=5, column=2, sticky="w", padx=(12,0), pady=(6,0))
        ttk.Checkbutton(ctrl, text="Show baselines", variable=self.var_show_norm_baselines).grid(row=5, column=3, sticky="w", padx=(12,0), pady=(6,0))

        ttk.Button(ctrl, text="Normalize selected μ spectra -> new objects", command=self.ui_normalize_selected).grid(row=6, column=0, columnspan=4, sticky="we", pady=(12,0))
        ex = ttk.LabelFrame(ctrl, text="EXAFS / FT (Larch autobk + xftf)", padding=6); ex.grid(row=7, column=0, columnspan=4, sticky="we", pady=(12,0))

        ttk.Label(ex, text="rbkg").grid(row=0, column=0, sticky="w")
        self.ent_rbkg = ttk.Entry(ex, width=8); self.ent_rbkg.insert(0, "1.0"); self.ent_rbkg.grid(row=0, column=1, sticky="w", padx=(6,0))
        ttk.Label(ex, text="kmin").grid(row=0, column=2, sticky="w", padx=(12,0))
        self.ent_kmin = ttk.Entry(ex, width=8); self.ent_kmin.insert(0, "0"); self.ent_kmin.grid(row=0, column=3, sticky="w", padx=(6,0))
        ttk.Label(ex, text="kmax").grid(row=0, column=4, sticky="w", padx=(12,0))
        self.ent_kmax = ttk.Entry(ex, width=8); self.ent_kmax.insert(0, "15"); self.ent_kmax.grid(row=0, column=5, sticky="w", padx=(6,0))

        ttk.Label(ex, text="dk").grid(row=1, column=0, sticky="w", pady=(6,0))
        self.ent_dk = ttk.Entry(ex, width=8); self.ent_dk.insert(0, "0.1"); self.ent_dk.grid(row=1, column=1, sticky="w", pady=(6,0), padx=(6,0))
        ttk.Label(ex, text="k-weight").grid(row=1, column=2, sticky="w", pady=(6,0), padx=(12,0))
        self.ent_kweight = ttk.Entry(ex, width=8); self.ent_kweight.insert(0, "2"); self.ent_kweight.grid(row=1, column=3, sticky="w", pady=(6,0), padx=(6,0))
        ttk.Label(ex, text="window").grid(row=1, column=4, sticky="w", pady=(6,0), padx=(12,0))
        self.var_ft_window = tk.StringVar(value="hanning")
        ttk.Combobox(ex, textvariable=self.var_ft_window, values=["hanning","kaiser","parzen","welch","sine","gaussian"], state="readonly", width=10).grid(row=1, column=5, sticky="w", pady=(6,0), padx=(6,0))

        ttk.Label(ex, text="rmax_out").grid(row=2, column=0, sticky="w", pady=(6,0))
        self.ent_rmax = ttk.Entry(ex, width=8); self.ent_rmax.insert(0, "10"); self.ent_rmax.grid(row=2, column=1, sticky="w", pady=(6,0), padx=(6,0))

        ttk.Button(ex, text="Compute χ(k) + FT for selected μ spectra -> new objects", command=self.ui_exafs_selected).grid(row=3, column=0, columnspan=6, sticky="we", pady=(10,0))

        mu_frame = ttk.LabelFrame(top, text="Select μ spectra", padding=8); mu_frame.pack(side="left", fill="both", expand=True, padx=(8,0))
        self.mu_listbox = tk.Listbox(mu_frame, selectmode=tk.EXTENDED, height=10); self.mu_listbox.pack(fill="both", expand=True)

        self.norm_plot = PlotPanel(bottom); self.norm_plot.frame.pack(fill="both", expand=True)

    def _build_tools_tab(self):
        parent = self.tab_tools
        top = ttk.Frame(parent); top.pack(fill="x", padx=8, pady=8)
        bottom = ttk.Frame(parent); bottom.pack(fill="both", expand=True, padx=8, pady=(0,8))

        lf_meta = ttk.LabelFrame(top, text="Metadata / History", padding=8); lf_meta.pack(side="left", fill="x")
        ttk.Button(lf_meta, text="Show metadata of selected", command=self.ui_show_metadata).pack(fill="x")
        ttk.Button(lf_meta, text="Show pipeline history of selected", command=self.ui_show_history).pack(fill="x", pady=(6,0))

        lf_edge = ttk.LabelFrame(top, text="Edge definer (manual label override)", padding=8); lf_edge.pack(side="left", fill="both", expand=True, padx=(8,0))

        ttk.Label(lf_edge, text="Select spectra (multi-select)").grid(row=0, column=0, columnspan=3, sticky="w")
        self.edge_listbox = tk.Listbox(lf_edge, selectmode=tk.EXTENDED, height=6); self.edge_listbox.grid(row=1, column=0, columnspan=3, sticky="we", pady=(4,8))
        lf_edge.grid_columnconfigure(2, weight=1)

        ttk.Label(lf_edge, text="Element").grid(row=2, column=0, sticky="w")
        self.var_edge_elem = tk.StringVar(value="Fe")
        self.cb_edge_elem = ttk.Combobox(lf_edge, textvariable=self.var_edge_elem, values=_periodic_table_symbols(), state="readonly", width=8)
        self.cb_edge_elem.grid(row=2, column=1, sticky="w", padx=(6,0))

        ttk.Label(lf_edge, text="Edge").grid(row=3, column=0, sticky="w", pady=(6,0))
        self.var_edge_line = tk.StringVar(value="K")
        self.cb_edge_line = ttk.Combobox(lf_edge, textvariable=self.var_edge_line, values=["K","L1","L2","L3","M1","M2","M3","M4","M5"], state="readonly", width=8)
        self.cb_edge_line.grid(row=3, column=1, sticky="w", padx=(6,0), pady=(6,0))

        self.var_edge_set_e0 = tk.BooleanVar(value=False)
        ttk.Checkbutton(lf_edge, text="Also set E0 to tabulated edge energy (xraydb)", variable=self.var_edge_set_e0).grid(row=4, column=0, columnspan=3, sticky="w", pady=(6,0))

        ttk.Button(lf_edge, text="Preview", command=self.ui_preview_edge_definer).grid(row=5, column=0, sticky="we", pady=(8,0))
        ttk.Button(lf_edge, text="Apply to selected spectra", command=self.ui_apply_edge_definer).grid(row=5, column=1, columnspan=2, sticky="we", pady=(8,0), padx=(8,0))

        self.tools_plot = PlotPanel(bottom); self.tools_plot.frame.pack(fill="both", expand=True)

    def _build_export_tab(self):
        parent = self.tab_export
        top = ttk.Frame(parent); top.pack(fill="x", padx=8, pady=8)
        bottom = ttk.Frame(parent); bottom.pack(fill="both", expand=True, padx=8, pady=(0,8))

        lf_ath = ttk.LabelFrame(top, text="Athena export", padding=8); lf_ath.pack(side="left", fill="x", expand=True)
        ttk.Button(lf_ath, text="Export selected as Athena column (.dat)", command=self.ui_export_athena_dat).pack(fill="x")
        ttk.Button(lf_ath, text="Export ALL mu/norm/flat as Athena project (.prj) (best effort)", command=self.ui_export_athena_prj).pack(fill="x", pady=(6,0))

        lf_csv = ttk.LabelFrame(top, text="CSV Builder (energy/angle/I0 + multiple It)", padding=8); lf_csv.pack(side="left", fill="both", expand=True, padx=(8,0))

        ttk.Label(lf_csv, text="Pick I0 spectrum").grid(row=0, column=0, sticky="w")
        self.var_csv_i0 = tk.StringVar(value="")
        self.cb_csv_i0 = ttk.Combobox(lf_csv, textvariable=self.var_csv_i0, values=[], state="readonly", width=30)
        self.cb_csv_i0.grid(row=0, column=1, sticky="w", padx=(6,0))

        ttk.Label(lf_csv, text="Select It spectra (multi-select)").grid(row=1, column=0, columnspan=2, sticky="w", pady=(10,0))
        self.csv_it_list = tk.Listbox(lf_csv, selectmode=tk.EXTENDED, height=6); self.csv_it_list.grid(row=2, column=0, columnspan=2, sticky="we", pady=(4,0))

        self.var_csv_include_angle = tk.BooleanVar(value=True)
        ttk.Checkbutton(lf_csv, text="Include angle column (from I0)", variable=self.var_csv_include_angle).grid(row=3, column=0, columnspan=2, sticky="w", pady=(8,0))

        ttk.Button(lf_csv, text="Build CSV...", command=self.ui_build_csv).grid(row=4, column=0, columnspan=2, sticky="we", pady=(10,0))

        self.export_plot = PlotPanel(bottom); self.export_plot.frame.pack(fill="both", expand=True)

    def _build_sm_params(self):
        for w in self.sm_params_frame.winfo_children(): w.destroy()
        m = self.var_sm_method.get()
        if m == "Savitzky-Golay":
            ttk.Label(self.sm_params_frame, text="window").grid(row=0, column=0, sticky="w")
            self.ent_sg_w = ttk.Entry(self.sm_params_frame, width=8); self.ent_sg_w.insert(0, "11"); self.ent_sg_w.bind('<KeyRelease>', lambda e: self._maybe_autopreview_smoothing()); self.ent_sg_w.grid(row=0, column=1, padx=(6,12))
            ttk.Label(self.sm_params_frame, text="poly").grid(row=0, column=2, sticky="w")
            self.ent_sg_p = ttk.Entry(self.sm_params_frame, width=8); self.ent_sg_p.insert(0, "3"); self.ent_sg_p.bind('<KeyRelease>', lambda e: self._maybe_autopreview_smoothing()); self.ent_sg_p.grid(row=0, column=3, padx=(6,0))
        elif m == "Median+SG":
            ttk.Label(self.sm_params_frame, text="median").grid(row=0, column=0, sticky="w")
            self.ent_m_w = ttk.Entry(self.sm_params_frame, width=8); self.ent_m_w.insert(0, "9"); self.ent_m_w.bind('<KeyRelease>', lambda e: self._maybe_autopreview_smoothing()); self.ent_m_w.grid(row=0, column=1, padx=(6,12))
            ttk.Label(self.sm_params_frame, text="sg window").grid(row=0, column=2, sticky="w")
            self.ent_m_sgw = ttk.Entry(self.sm_params_frame, width=8); self.ent_m_sgw.insert(0, "11"); self.ent_m_sgw.bind('<KeyRelease>', lambda e: self._maybe_autopreview_smoothing()); self.ent_m_sgw.grid(row=0, column=3, padx=(6,12))
            ttk.Label(self.sm_params_frame, text="sg poly").grid(row=0, column=4, sticky="w")
            self.ent_m_sgp = ttk.Entry(self.sm_params_frame, width=8); self.ent_m_sgp.insert(0, "3"); self.ent_m_sgp.bind('<KeyRelease>', lambda e: self._maybe_autopreview_smoothing()); self.ent_m_sgp.grid(row=0, column=5, padx=(6,0))
        elif m == "Whittaker":
            ttk.Label(self.sm_params_frame, text="λ").grid(row=0, column=0, sticky="w")
            self.ent_w_lam = ttk.Entry(self.sm_params_frame, width=10); self.ent_w_lam.insert(0, "1e5"); self.ent_w_lam.bind('<KeyRelease>', lambda e: self._maybe_autopreview_smoothing()); self.ent_w_lam.grid(row=0, column=1, padx=(6,12))
            ttk.Label(self.sm_params_frame, text="d").grid(row=0, column=2, sticky="w")
            self.ent_w_d = ttk.Entry(self.sm_params_frame, width=8); self.ent_w_d.insert(0, "2"); self.ent_w_d.bind('<KeyRelease>', lambda e: self._maybe_autopreview_smoothing()); self.ent_w_d.grid(row=0, column=3, padx=(6,0))
        else:
            ttk.Label(self.sm_params_frame, text="s").grid(row=0, column=0, sticky="w")
            self.ent_spl_s = ttk.Entry(self.sm_params_frame, width=10); self.ent_spl_s.insert(0, "0.0"); self.ent_spl_s.bind('<KeyRelease>', lambda e: self._maybe_autopreview_smoothing()); self.ent_spl_s.grid(row=0, column=1, padx=(6,0))

    def _build_fit_params(self):
        for w in self.fit_params_frame.winfo_children(): w.destroy()
        m = self.var_fit_method.get()
        help_txt = ""
        if m == "Chebyshev":
            ttk.Label(self.fit_params_frame, text="degree").grid(row=0, column=0, sticky="w")
            self.ent_cheb_deg = ttk.Entry(self.fit_params_frame, width=10); self.ent_cheb_deg.insert(0, "40"); self.ent_cheb_deg.grid(row=0, column=1, padx=(6,0))
            self.ent_cheb_deg.bind("<KeyRelease>", lambda e: self.ui_preview_fit() if getattr(self, "_fit_last_preview", None) else None)
            help_txt = "Chebyshev polynomial baseline fit.\ndegree: polynomial degree (integer). Typical: 10–80."
        elif m == "Whittaker":
            ttk.Label(self.fit_params_frame, text="λ").grid(row=0, column=0, sticky="w")
            self.ent_fit_lam = ttk.Entry(self.fit_params_frame, width=12); self.ent_fit_lam.insert(0, "1e5"); self.ent_fit_lam.grid(row=0, column=1, padx=(6,12))
            self.ent_fit_lam.bind("<KeyRelease>", lambda e: self.ui_preview_fit() if getattr(self, "_fit_last_preview", None) else None)
            ttk.Label(self.fit_params_frame, text="d").grid(row=0, column=2, sticky="w")
            self.ent_fit_d = ttk.Entry(self.fit_params_frame, width=8); self.ent_fit_d.insert(0, "2"); self.ent_fit_d.grid(row=0, column=3, padx=(6,0))
            self.ent_fit_d.bind("<KeyRelease>", lambda e: self.ui_preview_fit() if getattr(self, "_fit_last_preview", None) else None)
            help_txt = "Whittaker smoother baseline.\nλ: smoothness (float, >= 0). Typical: 1e3–1e7.\nd: difference order (int). Typical: 2."
        else:
            ttk.Label(self.fit_params_frame, text="s").grid(row=0, column=0, sticky="w")
            self.ent_fit_s = ttk.Entry(self.fit_params_frame, width=12); self.ent_fit_s.insert(0, "0.0"); self.ent_fit_s.grid(row=0, column=1, padx=(6,0))
            self.ent_fit_s.bind("<KeyRelease>", lambda e: self.ui_preview_fit() if getattr(self, "_fit_last_preview", None) else None)
            help_txt = "Spline baseline fit (SciPy).\ns: smoothing factor (float, >= 0). 0 = interpolate exactly."
        if getattr(self, "var_fit_help", None) is not None: self.var_fit_help.set(help_txt)

    def refresh_tree(self):
        for item in self.tree.get_children(): self.tree.delete(item)
        for sp in self.store.all():
            e0 = "" if sp.e0 is None or not np.isfinite(sp.e0) else f"{sp.e0:.1f}"
            er = f"{np.nanmin(sp.energy):.1f}–{np.nanmax(sp.energy):.1f}" if sp.energy.size else ""
            self.tree.insert("", "end", iid=sp.sid, values=(sp.name, sp.kind, edge_text(sp.label), e0, er, sp.meta.get("source","")))
        self.refresh_dropdowns()

    def refresh_dropdowns(self):
        all_names = [s.name for s in self.store.all()]
        if hasattr(self, 'cb_sm_target'): self.cb_sm_target['values'] = all_names
        if hasattr(self, 'cb_ang_before'): self.cb_ang_before['values'] = all_names
        if hasattr(self, 'cb_ang_after'): self.cb_ang_after['values'] = all_names

        i0_names = [s.name for s in self.store.all() if s.kind in ("I0","fit","I0_fit")]
        it_names = [s.name for s in self.store.all() if s.kind == "It"]
        mu_names = [s.name for s in self.store.all() if s.kind == "mu"]

        self.cb_i0_single["values"] = i0_names
        self.it_listbox.delete(0, tk.END)
        for n in it_names: self.it_listbox.insert(tk.END, n)
        self.mu_listbox.delete(0, tk.END)
        for n in mu_names: self.mu_listbox.insert(tk.END, n)
        self.cb_csv_i0["values"] = i0_names
        self.csv_it_list.delete(0, tk.END)
        for n in it_names: self.csv_it_list.insert(tk.END, n)
        if hasattr(self, 'edge_listbox'):
            self.edge_listbox.delete(0, tk.END)
            for n in all_names: self.edge_listbox.insert(tk.END, n)

    def on_tree_select(self, event=None):
        sel = self.tree.selection()
        if not sel: return
        self.selected_sid = sel[0]; self.plot_selected_preview()

    def on_tree_right_click(self, event):
        iid = self.tree.identify_row(event.y)
        if iid: self.tree.selection_set(iid); self.selected_sid = iid
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="Rename...", command=self.ui_rename_selected)
        menu.add_command(label="Duplicate", command=self.ui_duplicate_selected)
        menu.add_separator(); menu.add_command(label="Delete", command=self.ui_delete_selected); menu.add_separator()
        menu.add_command(label="Export selected as .dat", command=self.ui_export_athena_dat)
        menu.tk_popup(event.x_root, event.y_root)

    def on_tp_right_click(self, event):
        row = self.tp_tree.identify_row(event.y)
        if not row: return
        tk.Menu(self, tearoff=0).add_command(label="Delete tie point", command=lambda: self._delete_tiepoint(row)).tk_popup(event.x_root, event.y_root)

    def _delete_tiepoint(self, row_iid: str):
        try:
            idx = int(row_iid.replace("tp",""))
            if 0 <= idx < len(self.tiepoints):
                self.tiepoints.pop(idx); self._refresh_tiepoints_table()
        except Exception: pass

    def plot_selected_preview(self):
        if self.selected_sid is None: self.preview_plot.clear("Preview"); return
        sp = self.store.get(self.selected_sid)
        self.preview_plot.plot([sp.energy], [sp.y], [sp.name], "Energy (eV)", sp.units, f"{sp.label} — {sp.name} [{sp.kind}]")

    def ui_import_zips(self):
        paths = filedialog.askopenfilenames(title="Select EasyXAFS ZIP(s)", filetypes=[("ZIP files","*.zip"),("All files","*.*")])
        if not paths: return
        try:
            n = 0
            for zp in paths:
                for rec in read_easyxafs_zip(zp):
                    sp = self._spectrum_from_record(rec); sp.history.append(Operation("import", {"source": rec.get("source","")}))
                    self.store.add(sp); n += 1
            self.refresh_tree(); self.status_var.set(f"Imported {n} dataset(s) from ZIP(s).")
        except Exception as exc: messagebox.showerror("Import ZIP error", str(exc), parent=self)

    def ui_import_csvs(self):
        paths = filedialog.askopenfilenames(title="Select CSV(s)", filetypes=[("CSV files","*.csv"),("All files","*.*")])
        if not paths: return
        try:
            n = 0
            for p in paths:
                rec = read_csv_dataset(p); sp = self._spectrum_from_record(rec); sp.history.append(Operation("import", {"source": rec.get("source","")}))
                self.store.add(sp); n += 1
            self.refresh_tree(); self.status_var.set(f"Imported {n} CSV dataset(s).")
        except Exception as exc: messagebox.showerror("Import CSV error", str(exc), parent=self)

    def ui_import_prj(self):
        p = filedialog.askopenfilename(title="Select Athena project (.prj)", filetypes=[("Athena project","*.prj"),("All files","*.*")])
        if not p: return
        try:
            specs = read_athena_prj(p)
            for sp in specs: self.store.add(sp)
            self.refresh_tree(); self.status_var.set(f"Imported {len(specs)} group(s) from .prj.")
        except Exception as exc: messagebox.showerror("Import .prj error", str(exc), parent=self)

    def ui_clear(self):
        self.store.clear(); self.selected_sid = None; self.tiepoints.clear(); self._refresh_tiepoints_table()
        self.refresh_tree(); self.preview_plot.clear("Preview"); self.status_var.set("Cleared all spectra.")

    def load_initial_records(self, records: List[Dict[str, Any]]):
        loaded = 0
        for rec in records:
            xas = rec.get("xas")
            if xas is None:
                continue
            energy = np.asarray(getattr(xas, "energy", []), dtype=float)
            i0 = np.asarray(getattr(xas, "i0", []), dtype=float)
            it = np.asarray(getattr(xas, "it", []), dtype=float)
            if energy.size == 0 or i0.size == 0 or it.size == 0:
                continue
            mu = mu_from_transmission(i0, it, logbase="ln")
            inferred = (rec.get("meta") or {}).get("inferred_edge") or {}
            label = inferred.get("label") or "?"
            sp = Spectrum(
                sid=_uid("sp"),
                name=rec.get("title") or getattr(xas, "path", "Imported XAS"),
                kind="mu",
                energy=energy,
                y=mu,
                angle=None,
                units="μ(E)",
                label=label,
                e0=inferred.get("e0"),
                meta={
                    "source": getattr(xas, "path", ""),
                    "columns": {
                        "energy_col": getattr(xas, "energy_col", "Energy"),
                        "i0_col": getattr(xas, "i0_col", "I0"),
                        "it_col": getattr(xas, "it_col", "It"),
                    },
                    "scan_def": getattr(xas, "scan_def", {}),
                    "metadata": getattr(xas, "metadata", {}),
                },
            )
            sp.history.append(Operation("import", {"source": sp.meta.get("source", "")}))
            self.store.add(sp)
            loaded += 1

        self.refresh_tree()
        if loaded:
            self.status_var.set(f"Loaded {loaded} dataset(s) from the main app.")

    def _spectrum_from_record(self, rec: Dict[str, Any]) -> Spectrum:
        angle, energy, signal, cols = _extract_energy_angle_signal(rec["df"])
        kind = _classify_kind_from_name(rec["name"])
        scan_def = rec.get("scan_def", {}) or {}
        label, e0 = infer_edge_label_from_roi_scaled(energy, signal, scan_def)
        return Spectrum(
            sid=_uid("sp"), name=rec["name"], kind=kind, energy=energy, y=signal,
            angle=angle if np.isfinite(angle).any() else None, units="counts/s", label=label, e0=e0,
            meta={"source": rec.get("source",""), "columns": cols, "scan_def": scan_def, "metadata": rec.get("metadata", {})},
        )

    def ui_rename_selected(self):
        if self.selected_sid is None: return
        sp = self.store.get(self.selected_sid)
        new = simple_input(self, "Rename", "New name:", sp.name)
        if new: sp.name = new; self.refresh_tree()

    def ui_duplicate_selected(self):
        if self.selected_sid is None: return
        sp = self.store.get(self.selected_sid); sp2 = sp.copy(new_name=f"{sp.name}_copy")
        sp2.history.append(Operation("duplicate", {"from": sp.sid})); self.store.add(sp2); self.refresh_tree()

    def ui_delete_selected(self):
        if self.selected_sid is None: return
        if messagebox.askyesno("Delete", f"Delete '{self.store.get(self.selected_sid).name}'?", parent=self):
            self.store.remove(self.selected_sid); self.selected_sid = None; self.refresh_tree()

    def _maybe_autopreview_smoothing(self):
        if getattr(self, "var_sm_autopreview", None) is not None and bool(self.var_sm_autopreview.get()): self.ui_preview_smoothing()

    def _maybe_autopreview_mu(self):
        if getattr(self, "var_mu_autopreview", None) is not None and bool(self.var_mu_autopreview.get()): self.ui_preview_mu()

    def ui_preview_smoothing(self):
        try:
            name = getattr(self, "var_sm_target", None).get().strip() if getattr(self, "var_sm_target", None) is not None else ""
            sp = self.store.find_by_name(name) if name else (self.store.get(self.selected_sid) if self.selected_sid else None)
            if sp is None: self.preproc_plot.clear("Smoothing preview"); return

            method_ui = self.var_sm_method.get()
            if method_ui == "Savitzky-Golay": method, params = "savitzky-golay", {"window": int(self.ent_sg_w.get()), "poly": int(self.ent_sg_p.get())}
            elif method_ui == "Median+SG": method, params = "median+sg", {"median_window": int(self.ent_m_w.get()), "sg_window": int(self.ent_m_sgw.get()), "sg_poly": int(self.ent_m_sgp.get())}
            elif method_ui == "Whittaker": method, params = "whittaker", {"lam": float(self.ent_w_lam.get()), "d": int(self.ent_w_d.get())}
            else: method, params = "spline", {"s": float(self.ent_spl_s.get())}

            self.preproc_plot.plot([sp.energy, sp.energy], [sp.y, smooth_spectrum(sp.y, method, params)], ["raw", "smoothed"], "Energy (eV)", sp.units, f"Smoothing preview — {sp.name} ({method_ui})")
        except Exception as exc: messagebox.showerror("Smoothing preview error", str(exc), parent=self)

    def ui_apply_smoothing(self):
        name = getattr(self, "var_sm_target", None).get().strip() if getattr(self, "var_sm_target", None) is not None else ""
        sp = self.store.find_by_name(name) if name else (self.store.get(self.selected_sid) if self.selected_sid else None)
        if sp is None: messagebox.showinfo("Smoothing", "Select a spectrum.", parent=self); return

        method_ui = self.var_sm_method.get()
        if method_ui == "Savitzky-Golay": method, params = "savitzky-golay", {"window": int(self.ent_sg_w.get()), "poly": int(self.ent_sg_p.get())}
        elif method_ui == "Median+SG": method, params = "median+sg", {"median_window": int(self.ent_m_w.get()), "sg_window": int(self.ent_m_sgw.get()), "sg_poly": int(self.ent_m_sgp.get())}
        elif method_ui == "Whittaker": method, params = "whittaker", {"lam": float(self.ent_w_lam.get()), "d": int(self.ent_w_d.get())}
        else: method, params = "spline", {"s": float(self.ent_spl_s.get())}

        try:
            sp2 = sp.copy(new_name=f"{sp.name}_sm", new_kind=sp.kind)
            sp2.y = smooth_spectrum(sp.y, method, params)
            sp2.history.append(Operation("smooth", {"method": method, **params}))
            self.store.add(sp2); self.refresh_tree()
            self.preproc_plot.plot([sp.energy, sp2.energy], [sp.y, sp2.y], ["raw", "smoothed"], "Energy (eV)", sp.units, f"Smoothing ({method_ui})")
        except Exception as exc: messagebox.showerror("Smoothing error", str(exc), parent=self)

    def ui_apply_angle_correction(self):
        bname = getattr(self, "var_ang_before", None).get().strip() if getattr(self, "var_ang_before", None) is not None else ""
        aname = getattr(self, "var_ang_after", None).get().strip() if getattr(self, "var_ang_after", None) is not None else ""

        sp_before = self.store.find_by_name(bname) if bname else (self.store.get(self.selected_sid) if self.selected_sid else None)
        if sp_before is None: messagebox.showinfo("Angle/E correction", "Select a BEFORE spectrum.", parent=self); return
        sp_after = self.store.find_by_name(aname) if aname else None

        if self.var_ang_mode.get().startswith("C"):
            if sp_after is None: messagebox.showinfo("Mode C", "Select an AFTER spectrum for alignment.", parent=self); return
            if len(self.tiepoints) < 1: messagebox.showinfo("Mode C", "Add at least one tie point first.", parent=self); return
            try:
                e_corr, diag = apply_alignment_mode_c(sp_after.energy, self.tiepoints, model=self.var_c_model.get())
                sp2 = sp_after.copy(new_name=f"{sp_after.name}_Ealign", new_kind=f"corrected_{sp_after.kind}")
                sp2.energy = np.asarray(e_corr, float)
                sp2.history.append(Operation("align_mode_c", {"before": sp_before.name, "after": sp_after.name, **diag}))
                sp2.label, sp2.e0 = infer_edge_label_from_roi_scaled(sp2.energy, sp2.y, sp2.meta.get("scan_def", {}) or {})
                self.store.add(sp2); self.refresh_tree()
                self.preproc_plot.plot([sp_before.energy, sp_after.energy, sp2.energy], [sp_before.y, sp_after.y, sp2.y], ["before", "after", "after (aligned)"], "Energy (eV)", sp_after.units, "Mode C feature alignment")
            except Exception as exc: messagebox.showerror("Mode C error", str(exc), parent=self)
            return

        if sp_before.angle is None or not np.isfinite(sp_before.angle).any(): messagebox.showerror("Angle/E correction", "BEFORE spectrum has no valid angle column.", parent=self); return
        mode = "A" if self.var_ang_mode.get().startswith("A") else "B"
        try:
            e_corr, diag = angle_energy_correction_bragg(sp_before.angle, sp_before.energy, sp_before.meta.get("scan_def", {}) or {}, mode=mode, fit_linear=bool(self.var_fit_linear.get()))
            sp2 = sp_before.copy(new_name=f"{sp_before.name}_Ebragg{mode}", new_kind=f"corrected_{sp_before.kind}")
            sp2.energy = np.asarray(e_corr, float)
            sp2.history.append(Operation("angle_energy_correction", {"mode": mode, **diag}))
            sp2.label, sp2.e0 = infer_edge_label_from_roi_scaled(sp2.energy, sp2.y, sp_before.meta.get("scan_def", {}) or {})
            self.store.add(sp2); self.refresh_tree()
            self.preproc_plot.plot([sp_before.energy, sp2.energy], [sp_before.y, sp2.y], ["raw axis","corrected axis"], "Energy (eV)", sp_before.units, f"Bragg correction Mode {mode}")
        except Exception as exc: messagebox.showerror("Angle/E correction error", str(exc), parent=self)

    def on_preproc_click(self, event):
        if event is None or event.xdata is None or event.ydata is None: return
        st = getattr(self, "_mode_c_click_state", None)
        if not st or not st.get("active", False): return

        role = st.get("waiting", "before")
        line = self._pick_line_map.get(role)
        if line is None: self.status_var.set("Picking: plot overlay first."); return

        x = np.asarray(line.get_xdata(), float); y = np.asarray(line.get_ydata(), float)
        if x.size == 0: return

        idx = int(np.nanargmin(np.abs(x - float(event.xdata))))
        ex, ey = float(x[idx]), float(y[idx])

        mk, = self.preproc_plot.ax.plot([ex], [ey], marker="o", ms=6, linestyle="None")
        self._pick_markers.append(mk); self.preproc_plot.canvas.draw_idle()

        if role == "before":
            st["last_before"] = ex; st["waiting"] = "after"; self.status_var.set(f"Picked BEFORE at {ex:.3f} eV. Now click AFTER.")
        else:
            eb = st.get("last_before", None)
            if eb is None: st["waiting"] = "before"; self.status_var.set("Picking reset: missing BEFORE point. Click BEFORE."); return
            self.tiepoints.append(TiePoint(e_before=float(eb), e_after=ex)); self._refresh_tiepoints_table()
            st["waiting"] = "before"; st["active"] = False; self.status_var.set(f"Added tie point: {eb:.3f} → {ex:.3f} (ΔE={ex-eb:.3f} eV).")

    def ui_plot_mode_c_overlay(self):
        bname = getattr(self, "var_ang_before", None).get().strip() if getattr(self, "var_ang_before", None) is not None else ""
        aname = getattr(self, "var_ang_after", None).get().strip() if getattr(self, "var_ang_after", None) is not None else ""
        if not bname or not aname: messagebox.showinfo("Mode C", "Select BEFORE and AFTER spectra.", parent=self); return
        b, a = self.store.find_by_name(bname), self.store.find_by_name(aname)
        if b is None or a is None: messagebox.showerror("Mode C", "BEFORE/AFTER spectrum not found.", parent=self); return

        self.preproc_plot.ax.clear(); self.preproc_plot.ax.grid(alpha=0.25)
        self._pick_line_map["before"], = self.preproc_plot.ax.plot(b.energy, b.y, lw=1.3, label=f"before: {b.name}")
        self._pick_line_map["after"],  = self.preproc_plot.ax.plot(a.energy, a.y, lw=1.3, label=f"after: {a.name}")
        self.preproc_plot.ax.set_xlabel("Energy (eV)"); self.preproc_plot.ax.set_ylabel(a.units); self.preproc_plot.ax.set_title("Mode C overlay")
        self.preproc_plot.ax.legend(loc="best"); self.preproc_plot.fig.tight_layout(); self.preproc_plot.canvas.draw_idle()

    def ui_start_picking_pair(self):
        if not self._pick_line_map.get("before") or not self._pick_line_map.get("after"): messagebox.showinfo("Mode C", "Plot overlay first.", parent=self); return
        self._mode_c_click_state.update({"active": True, "waiting": "before", "last_before": None})
        self.status_var.set("Picking: click BEFORE feature point.")

    def ui_clear_tiepoints(self):
        self.tiepoints.clear(); self._refresh_tiepoints_table()
        try:
            for mk in getattr(self, "_pick_markers", []): mk.remove()
            self._pick_markers = []; self.preproc_plot.canvas.draw_idle()
        except Exception: pass
        self.status_var.set("Cleared tie points.")

    def _refresh_tiepoints_table(self):
        for item in self.tp_tree.get_children(): self.tp_tree.delete(item)
        for i, tp in enumerate(self.tiepoints): self.tp_tree.insert("", "end", iid=f"tp{i}", values=(f"{tp.e_before:.3f}", f"{tp.e_after:.3f}", f"{tp.e_before - tp.e_after:+.3f}"))

    def _parse_mask_ranges(self) -> List[Tuple[float,float]]:
        txt = self.txt_fit_mask.get("1.0","end").strip()
        if not txt: return []
        out = []
        for line in txt.splitlines():
            line = line.strip()
            if not line: continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 2: raise ValueError(f"Bad mask line: {line}")
            lo, hi = float(parts[0]), float(parts[1])
            if hi <= lo: raise ValueError(f"Bad mask range: {lo},{hi}")
            out.append((lo,hi))
        return out

    def _make_fit_mask(self, sp: Spectrum) -> np.ndarray:
        mask = np.isfinite(sp.energy) & np.isfinite(sp.y)
        for lo, hi in self._parse_mask_ranges(): mask &= ~((sp.energy >= lo) & (sp.energy <= hi))
        return mask

    def ui_preview_fit(self):
        try:
            if self.selected_sid is None: raise ValueError("Select a spectrum to fit.")
            sp = self.store.get(self.selected_sid)
            method_ui = self.var_fit_method.get()
            mask = self._make_fit_mask(sp)

            if method_ui == "Chebyshev":
                deg = int(self.ent_cheb_deg.get())
                yfit = fit_chebyshev(sp.energy, sp.y, deg, mask)
                diag = {"method": "chebyshev", "degree": deg}
            elif method_ui == "Whittaker":
                lam, d = float(self.ent_fit_lam.get()), int(self.ent_fit_d.get())
                yfit = whittaker_smooth(sp.y, lam=lam, d=d)
                diag = {"method": "whittaker", "lam": lam, "d": d}
            else:
                if not _SCIPY_AVAILABLE: raise RuntimeError("Spline method requires SciPy.")
                s = float(self.ent_fit_s.get())
                yfit = fit_spline(sp.energy, sp.y, s=s, mask=mask)
                diag = {"method": "spline", "s": s}

            resid = sp.y - yfit
            self._fit_last_preview = (sp, yfit, resid, diag)

            self.fit_plot.plot([sp.energy, sp.energy], [sp.y, yfit], ["data", "fit"], "Energy (eV)", sp.units, f"Fit preview — {sp.name} ({method_ui})")
            self.fit_resid_plot.plot([sp.energy], [resid], ["residual"], "Energy (eV)", sp.units, "Residual (data - fit)")
            x0, x1 = float(np.nanmin(sp.energy)), float(np.nanmax(sp.energy))
            self.fit_plot.ax.set_xlim(x0, x1); self.fit_resid_plot.ax.set_xlim(x0, x1)
            self.fit_plot.canvas.draw_idle(); self.fit_resid_plot.canvas.draw_idle()
        except Exception as exc: messagebox.showerror("Fit preview error", str(exc), parent=self)

    def ui_save_fit(self):
        try:
            if self._fit_last_preview is None: self.ui_preview_fit()
            if self._fit_last_preview is None: raise ValueError("No fit preview available.")
            sp, yfit, resid, diag = self._fit_last_preview
            sp2 = sp.copy(new_name=f"{sp.name}_fit", new_kind="fit"); sp2.y = np.asarray(yfit, float)
            sp2.history.append(Operation("fit_baseline", diag))
            self.store.add(sp2); self.refresh_tree(); self.status_var.set(f"Saved fit as new object: {sp2.name}")
        except Exception as exc: messagebox.showerror("Save fit error", str(exc), parent=self)

    def ui_preview_mu(self):
        try:
            i0_name = self.var_i0_single.get().strip()
            if not i0_name: self.mu_plot.clear("μ preview"); return
            i0 = self.store.find_by_name(i0_name)
            if i0 is None: self.mu_plot.clear("μ preview"); return

            it_sel = [self.it_listbox.get(i) for i in self.it_listbox.curselection()]
            if not it_sel: self.mu_plot.clear("μ preview"); return
            it = self.store.find_by_name(it_sel[0])
            if it is None: self.mu_plot.clear("μ preview"); return

            mu = build_mu(i0_energy=i0.energy, i0=i0.y, it_energy=it.energy, it=it.y, log_mode=self.var_log.get())
            self.mu_plot.plot([i0.energy], [mu], [f"μ from {it.name}"], "Energy (eV)", "arb.", f"μ preview — I0={i0.name}")
        except Exception as exc: messagebox.showerror("μ preview error", str(exc), parent=self)

    def ui_compute_mu(self):
        try:
            i0_name = self.var_i0_single.get().strip()
            if not i0_name: raise ValueError("Select I0 spectrum")
            i0 = self.store.find_by_name(i0_name)
            if i0 is None: raise ValueError("I0 not found")

            it_sel = [self.it_listbox.get(i) for i in self.it_listbox.curselection()]
            if not it_sel: raise ValueError("Select at least one It spectrum")
            its = [s for s in [self.store.find_by_name(n) for n in it_sel] if s is not None]

            last = None
            for it in its:
                mu = build_mu(i0_energy=i0.energy, i0=i0.y, it_energy=it.energy, it=it.y, log_mode=self.var_log.get())
                sp_mu = it.copy(new_name=f"{it.name}_mu", new_kind="mu")
                sp_mu.energy = np.asarray(i0.energy, float); sp_mu.y = np.asarray(mu, float); sp_mu.label = it.label; sp_mu.e0 = it.e0
                sp_mu.history.append(Operation("mu_builder", {"I0": i0.name, "It": it.name, "log": self.var_log.get()}))
                self.store.add(sp_mu); last = (it, sp_mu)

            self.refresh_tree()
            if last is not None: self.mu_plot.plot([last[1].energy], [last[1].y], [last[1].name], "Energy (eV)", "arb.", "μ(E) computed")
        except Exception as exc: messagebox.showerror("μ builder error", str(exc), parent=self)

    def ui_normalize_selected(self):
        try:
            sel = [self.mu_listbox.get(i) for i in self.mu_listbox.curselection()]
            if not sel: raise ValueError("Select μ spectra")
            mu_specs = [s for s in [self.store.find_by_name(n) for n in sel] if s is not None]
            if not mu_specs: raise ValueError("Selected μ not found")

            e0_method = self.var_e0_method.get()
            t = self.ent_e0_manual.get().strip()
            e0_manual = float(t) if t and e0_method == "manual" else None
            pre1, pre2 = float(self.ent_pre1.get()), float(self.ent_pre2.get())
            norm1, norm2 = float(self.ent_norm1.get()), float(self.ent_norm2.get())
            nnorm = int(float(self.ent_nnorm.get()))

            smooth_for_e0 = None
            if self.var_norm_smooth.get():
                sm_ui = self.var_norm_sm.get()
                if sm_ui == "Savitzky-Golay": smooth_for_e0 = ("savitzky-golay", {"window": 11, "poly": 3})
                elif sm_ui == "Median+SG": smooth_for_e0 = ("median+sg", {"median_window": 9, "sg_window": 11, "sg_poly": 3})
                else: smooth_for_e0 = ("whittaker", {"lam": 1e5, "d": 2})

            last = None
            for sp in mu_specs:
                out = larch_normalize(sp.energy, sp.y, e0_method=e0_method, e0_manual=e0_manual, pre1=pre1, pre2=pre2, norm1=norm1, norm2=norm2, nnorm=nnorm, smooth_for_e0=smooth_for_e0)
                sp_norm = sp.copy(new_name=f"{sp.name}_norm", new_kind="norm")
                sp_norm.y = out["norm"]; sp_norm.e0 = out["e0"]; sp_norm.history.append(Operation("normalize", {"e0_method": e0_method, "e0": out["e0"]}))
                sp_flat = sp.copy(new_name=f"{sp.name}_flat", new_kind="flat")
                sp_flat.y = out["flat"]; sp_flat.e0 = out["e0"]; sp_flat.history.append(Operation("normalize_flat", {"e0": out["e0"]}))
                self.store.add(sp_norm); self.store.add(sp_flat); last = (sp, out)

            self.refresh_tree()
            if last is not None:
                sp, out = last
                xs, ys, labs = [sp.energy, sp.energy, sp.energy], [sp.y, out["norm"], out["deriv"]], ["μ(E)", "norm", "dμ/dE"]
                if getattr(self, "var_show_norm_baselines", None) is not None and bool(self.var_show_norm_baselines.get()):
                    xs.extend([sp.energy, sp.energy])
                    ys.extend([out.get("pre_edge_line", np.nan*sp.y), out.get("post_edge_line", np.nan*sp.y)])
                    labs.extend(["pre-edge baseline", "post-edge baseline"])
                self.norm_plot.plot(xs, ys, labs, "Energy (eV)", "arb.", f"{sp.label} — E0={out['e0']:.2f}")

                if getattr(self, "var_show_norm_anchors", None) is not None and bool(self.var_show_norm_anchors.get()):
                    for val in [out.get("anchors", {}).get(key) for key in ("pre1","pre2","norm1","norm2")]:
                        if val is not None: self.norm_plot.ax.axvline(float(val), ls="--", lw=1.0, alpha=0.8)
                    self.norm_plot.canvas.draw_idle()
        except Exception as exc: messagebox.showerror("Normalization error", str(exc), parent=self)

    def ui_exafs_selected(self):
        try:
            sel = [self.mu_listbox.get(i) for i in self.mu_listbox.curselection()]
            if not sel: raise ValueError("Select μ spectra")
            mu_specs = [s for s in [self.store.find_by_name(n) for n in sel] if s is not None]
            if not mu_specs: raise ValueError("Selected μ not found")

            e0_method = self.var_e0_method.get()
            t = self.ent_e0_manual.get().strip()
            e0_manual = float(t) if t and e0_method == "manual" else None
            pre1, pre2 = float(self.ent_pre1.get()), float(self.ent_pre2.get())
            norm1, norm2 = float(self.ent_norm1.get()), float(self.ent_norm2.get())
            nnorm = int(float(self.ent_nnorm.get()))

            smooth_for_e0 = None
            if self.var_norm_smooth.get():
                sm_ui = self.var_norm_sm.get()
                if sm_ui == "Savitzky-Golay": smooth_for_e0 = ("savitzky-golay", {"window": 11, "poly": 3})
                elif sm_ui == "Median+SG": smooth_for_e0 = ("median+sg", {"median_window": 9, "sg_window": 11, "sg_poly": 3})
                else: smooth_for_e0 = ("whittaker", {"lam": 1e5, "d": 2})

            rbkg, kmin, kmax, dk = float(self.ent_rbkg.get()), float(self.ent_kmin.get()), float(self.ent_kmax.get()), float(self.ent_dk.get())
            kweight, window, rmax_out = int(float(self.ent_kweight.get())), self.var_ft_window.get(), float(self.ent_rmax.get())

            last = None
            for sp in mu_specs:
                out = larch_exafs_pipeline(
                    sp.energy, sp.y, e0_method=e0_method, e0_manual=e0_manual, pre1=pre1, pre2=pre2, norm1=norm1, norm2=norm2, nnorm=nnorm,
                    rbkg=rbkg, kmin=kmin, kmax=kmax, dk=dk, kweight=kweight, window=window, rmax_out=rmax_out, smooth_for_e0=smooth_for_e0
                )

                sp_norm = sp.copy(new_name=f"{sp.name}_norm", new_kind="norm"); sp_norm.y = out["norm"]; sp_norm.e0 = out["e0"]; sp_norm.history.append(Operation("normalize", {"e0_method": e0_method}))
                sp_flat = sp.copy(new_name=f"{sp.name}_flat", new_kind="flat"); sp_flat.y = out["flat"]; sp_flat.e0 = out["e0"]; sp_flat.history.append(Operation("normalize_flat", {"e0": out["e0"]}))
                self.store.add(sp_norm); self.store.add(sp_flat)

                sp_chi = sp.copy(new_name=f"{sp.name}_chi", new_kind="chi(k)"); sp_chi.energy = out["k"]; sp_chi.y = out["chi"]; sp_chi.e0 = out["e0"]; sp_chi.history.append(Operation("autobk", {"rbkg": rbkg, "kmin": kmin, "kmax": kmax, "dk": dk}))
                self.store.add(sp_chi)

                sp_chikw = sp.copy(new_name=f"{sp.name}_chi_k{kweight}", new_kind=f"chi(k)*k^{kweight}"); sp_chikw.energy = out["k"]; sp_chikw.y = out["chi_kw"]; sp_chikw.e0 = out["e0"]; sp_chikw.history.append(Operation("kweight", {"kweight": kweight}))
                self.store.add(sp_chikw)

                sp_ft = sp.copy(new_name=f"{sp.name}_FTmag", new_kind="FT|chi|"); sp_ft.energy = out["r"]; sp_ft.y = out["chir_mag"]; sp_ft.e0 = out["e0"]; sp_ft.history.append(Operation("xftf", {"kmin": kmin, "kmax": kmax, "dk": dk, "kweight": kweight, "window": window, "rmax_out": rmax_out}))
                self.store.add(sp_ft)

                last = (sp, out)

            self.refresh_tree()
            if last is not None:
                sp, out = last
                self.norm_plot.plot([out["k"], out["k"], out["r"]], [out["chi"], out["chi_kw"], out["chir_mag"]], ["chi(k)", f"chi(k)*k^{kweight}", "|FT|"], "k (1/Å) / R (Å)", "arb.", f"{sp.label} — E0={out['e0']:.2f}")
        except Exception as exc: messagebox.showerror("EXAFS/FT error", str(exc), parent=self)

    def ui_show_metadata(self):
        if self.selected_sid is None: return
        sp = self.store.get(self.selected_sid); show_text_window(self, f"Metadata — {sp.name}", json.dumps(sp.meta, indent=2, default=str))

    def ui_show_history(self):
        if self.selected_sid is None: return
        sp = self.store.get(self.selected_sid)
        lines = [f"Name: {sp.name}", f"Kind: {sp.kind}", f"Label: {sp.label}", f"Parents: {sp.parents}", ""]
        for op in sp.history:
            when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(op.when))
            lines.append(f"- {when} | {op.name} | {json.dumps(op.params, default=str)}")
        show_text_window(self, f"History — {sp.name}", "\n".join(lines))

    def ui_preview_edge_definer(self):
        try:
            names = [self.edge_listbox.get(i) for i in self.edge_listbox.curselection()] if hasattr(self, "edge_listbox") else []
            specs = [s for s in [self.store.find_by_name(n) for n in names] if s is not None]
            if not specs: self.tools_plot.clear("Edge definer preview"); return

            elem = self.var_edge_elem.get().strip(); edge = self.var_edge_line.get().strip(); e_edge = None
            try:
                Group, xraydb, find_e0, pre_edge, autobk, xftf = require_larch()
                if xraydb is not None:
                    edges = xraydb.xray_edges(elem)
                    if edge in edges and getattr(edges[edge], "energy", None) is not None: e_edge = float(edges[edge].energy)
            except Exception: pass

            self.tools_plot.plot([sp.energy for sp in specs], [sp.y for sp in specs], [sp.name for sp in specs], "Energy (eV)", "arb.", f"Edge definer preview — {elem} {edge}")
            if e_edge is not None and np.isfinite(e_edge): self.tools_plot.ax.axvline(e_edge, ls="--", lw=1.2); self.tools_plot.canvas.draw_idle()
        except Exception as exc: messagebox.showerror("Edge definer preview error", str(exc), parent=self)

    def ui_apply_edge_definer(self):
        try:
            names = [self.edge_listbox.get(i) for i in self.edge_listbox.curselection()] if hasattr(self, "edge_listbox") else []
            specs = [s for s in [self.store.find_by_name(n) for n in names] if s is not None]
            if not specs: raise ValueError("Select at least one spectrum in the Edge Definer list.")

            elem = self.var_edge_elem.get().strip(); edge = self.var_edge_line.get().strip(); e_edge = None
            if bool(self.var_edge_set_e0.get()):
                Group, xraydb, find_e0, pre_edge, autobk, xftf = require_larch()
                if xraydb is None: raise ValueError("xraydb not available in this Larch install; cannot set tabulated E0.")
                edges = xraydb.xray_edges(elem)
                if edge not in edges or getattr(edges[edge], "energy", None) is None: raise ValueError("Unknown element/edge in xraydb.")
                e_edge = float(edges[edge].energy)

            for sp in specs:
                sp.label = f"XAS({elem} {edge})"
                if e_edge is not None: sp.e0 = e_edge
                sp.history.append(Operation("edge_definer", {"element": elem, "edge": edge, "set_e0": bool(e_edge is not None)}))

            self.refresh_tree(); self.ui_preview_edge_definer(); self.status_var.set(f"Applied manual edge label to {len(specs)} spectrum/spectra.")
        except Exception as exc: messagebox.showerror("Edge definer error", str(exc), parent=self)

    def ui_export_athena_dat(self):
        if self.selected_sid is None: messagebox.showinfo("Export", "Select a spectrum first.", parent=self); return
        sp = self.store.get(self.selected_sid)
        p = filedialog.asksaveasfilename(title="Save Athena column file", defaultextension=".dat", filetypes=[("Athena column file","*.dat"),("All files","*.*")])
        if not p: return
        header = [f"# Athena column file exported from XAS Ultimate GUI", f"# name = {sp.name}", f"# kind = {sp.kind}", f"# label = {sp.label}"]
        if sp.e0 is not None and np.isfinite(sp.e0): header.append(f"# e0 = {sp.e0:.6f}")
        try:
            export_athena_column(p, sp.energy, sp.y, header); self.status_var.set(f"Saved: {p}")
        except Exception as exc: messagebox.showerror("Export .dat error", str(exc), parent=self)

    def ui_export_athena_prj(self):
        out = filedialog.asksaveasfilename(title="Save Athena project (.prj)", defaultextension=".prj", filetypes=[("Athena project","*.prj"),("All files","*.*")])
        if not out: return
        try:
            ok = export_athena_prj_best_effort(out, [s for s in self.store.all() if s.kind in ("mu","norm","flat")])
            if not ok: messagebox.showwarning("Export .prj", "Could not find write_athena in your larch install. Export .dat instead.", parent=self)
            else: self.status_var.set(f"Saved .prj: {out}")
        except Exception as exc: messagebox.showerror("Export .prj error", str(exc), parent=self)

    def ui_build_csv(self):
        try:
            i0_name = self.var_csv_i0.get().strip()
            if not i0_name: raise ValueError("Select I0")
            i0 = self.store.find_by_name(i0_name)
            if i0 is None: raise ValueError("I0 not found")
            it_sel = [self.csv_it_list.get(i) for i in self.csv_it_list.curselection()]
            if not it_sel: raise ValueError("Select at least one It")
            its = [s for s in [self.store.find_by_name(n) for n in it_sel] if s is not None]
            if not its: raise ValueError("Selected It spectra not found")

            E = np.asarray(i0.energy, float)
            if E.size < 10: raise ValueError("I0 energy grid is too small")

            for sp in its:
                if i0.e0 is not None and sp.e0 is not None and abs(float(i0.e0) - float(sp.e0)) > 25.0:
                    if not messagebox.askyesno("Edge mismatch", f"Edge/E0 seems different between I0 ({i0.name}, E0={i0.e0:.1f}) and {sp.name} (E0={sp.e0:.1f}).\n\nContinue anyway?", parent=self): return
                    break

            if any((np.asarray(sp.energy).shape != E.shape) or (np.nanmax(np.abs(np.asarray(sp.energy, float) - E)) > 1e-6) for sp in its):
                if not messagebox.askyesno("Energy grid mismatch", "Selected It spectra have different energy grids than I0.\nRebin (interpolate) It spectra to the I0 energy grid?", parent=self): return

            out = filedialog.asksaveasfilename(title="Save CSV", defaultextension=".csv", filetypes=[("CSV","*.csv"),("All files","*.*")])
            if not out: return

            df = pd.DataFrame({"energy_eV": E})
            if self.var_csv_include_angle.get(): df["angle_deg"] = i0.angle if i0.angle is not None and len(i0.angle)==len(E) else np.nan
            df["I0"] = np.asarray(i0.y, float)

            for sp in its: df[f"It_{sp.name}"] = _interp_to_grid(np.asarray(sp.energy, float), np.asarray(sp.y, float), E)

            df.to_csv(out, index=False)
            self.export_plot.plot([E], [df["I0"].to_numpy(float)], ["I0"], "Energy (eV)", "counts/s", "CSV builder preview (I0)")
        except Exception as exc: messagebox.showerror("CSV Builder error", str(exc), parent=self)

# ---------------------------- Entrypoint ----------------------------

def main() -> None:
    XASUltimateApp().mainloop()

if __name__ == "__main__":
    try: main()
    except Exception as exc:
        import traceback; traceback.print_exc(); raise
