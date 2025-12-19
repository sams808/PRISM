# ui_simple_plot.py
# --------------------------------------------------------------------------------------
# SimplePlotWindow (V40) — standalone UI for quick multi-plot + CIF overlays
# - File-type agnostic: relies on injected loader callables.
# - CIF utilities are factored out into cif_tools.py.
# --------------------------------------------------------------------------------------
from __future__ import annotations

import os, colorsys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from typing import Callable, List, Tuple, Dict, Any

from cif_tools import (
    bragg_peaks_from_cif_generic,
    list_cif_files_case_insensitive,
)

def _apply_modern_style(widget):
    palette = {
        "bg": "#f8f4ea",          # bone white
        "card": "#ffffff",
        "card_alt": "#f2ede3",
        "accent": "#a9cff5",      # light blue for buttons
        "accent_alt": "#a9cff5",
        "muted": "#657080",
        "success": "#a9cff5",
    }
    style = ttk.Style(widget)
    try:
        style.theme_use("clam")
    except Exception:
        pass
    style.configure(".", background=palette["bg"], foreground="#1c2733", fieldbackground=palette["card"])
    style.configure("Card.TFrame", background=palette["card"], borderwidth=1, relief="flat")
    style.configure("CardAlt.TFrame", background=palette["card_alt"], borderwidth=1, relief="flat")
    style.configure("Card.TLabelframe", background=palette["card"], relief="flat", borderwidth=1)
    style.configure("Card.TLabelframe.Label", background=palette["card"], foreground="#1c2733")
    style.configure("Section.TLabel", background=palette["bg"], foreground="#1c2733", font=("Segoe UI", 13, "bold"))
    style.configure("Muted.TLabel", background=palette["bg"], foreground=palette["muted"])
    style.configure(
        "TCombobox",
        fieldbackground=palette["card"],
        background=palette["card"],
        foreground="#1c2733",
        arrowcolor="#1c2733",
        borderwidth=1,
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", palette["card"])],
        foreground=[("!disabled", "#1c2733")],
    )

    def _btn(name, color):
        style.configure(name, background=color, foreground="#1c2733", padding=(10, 7), borderwidth=1, relief="solid")
        style.map(name, background=[("active", color)], bordercolor=[("focus", "#d7d1c8")])

    _btn("Primary.TButton", palette["accent"])
    _btn("Alt.TButton", palette["accent_alt"])
    _btn("Success.TButton", palette["success"])
    style.configure(
        "TButton",
        background=palette["accent"],
        foreground="#1c2733",
        padding=(10, 7),
        borderwidth=1,
        relief="solid",
    )
    style.map(
        "TButton",
        background=[("active", palette["accent"]), ("pressed", palette["accent"])],
        bordercolor=[("focus", "#d7d1c8")],
    )
    try:
        widget.configure(bg=palette["bg"])
    except Exception:
        pass
    return palette

_APP_CFG_PATH = Path.home() / ".raman_app.json"


def _cfg_load():
    try:
        with open(_APP_CFG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _cfg_save(d: dict):
    try:
        with open(_APP_CFG_PATH, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)
    except Exception:
        pass


class SimplePlotWindow(tk.Toplevel):
    """
    Quick visualisation of imported spectra (multi-format via injected loader).
    - Multi-selection, normalisation, spectral-difference mode
    - Axis titles/limits, stacked or separate
    - CIF overlay with a small manager (toggle/color/label/position)
    - Smoothing (SG or MA(5)), dark/light, grid, annotations, export
    """

    COLORS = ["navy", "darkred", "seagreen", "darkorange", "purple", "teal", "brown", "indigo"]

    def __init__(
        self,
        master,
        file_paths: List[str],
        file_titles: List[str],
        *,
        load_any_func: Callable[[str], Dict[str, Any]],
        pick_ta_xy_func: Callable[[pd.DataFrame], Tuple[np.ndarray, np.ndarray, Dict[str, str]]],
    ):
        super().__init__(master)
        self.title("Simple plot")
        self.geometry("1350x750")
        self.palette = _apply_modern_style(self)

        # ------------------- injected data access -------------------
        self._load_any = load_any_func
        self._pick_ta_xy = pick_ta_xy_func

        # ------------------- file list -------------------
        self.file_paths = list(file_paths)
        self.file_titles = list(file_titles)

        # ------------------- runtime plot state -------------------
        self.current_data: List[Tuple[np.ndarray, np.ndarray]] = []
        self._lines = []              # Line2D on current figure
        self._debouncers = {}         # tk.after debounce
        self._dta_state: Dict[str, Dict[str, str]] = {}
        self._dta_active_path: str | None = None

        # ------------------- CIF state -------------------
        self._cif_bragg_series = []   # dicts: {"path","label","plot_label","peaks","visible","color","pad"}
        self._cif_colors = ["crimson", "royalblue", "seagreen", "darkorange", "purple", "goldenrod"]
        self._cif_manager_win = None
        self._cif_manager_inner = None
        self._cif_label_pos = tk.StringVar(value="right_out")  # right_out | left_out | right_in | left_in

        # persistent config
        self._cfg = _cfg_load()
        self.cif_base_dir = self._cfg.get("cif_base_dir")

        # Bragg vertical line height multiplier (ONLY affects vline height)
        try:
            init_bh = float(self._cfg.get("bragg_height_scale", 1.0))
        except Exception:
            init_bh = 1.0
        if init_bh <= 0.1:
            init_bh = 0.1
        self.var_bragg_h_scale = tk.DoubleVar(value=init_bh)

        # preload CIFs if a folder is memorized
        if self.cif_base_dir and os.path.isdir(self.cif_base_dir):
            try:
                self.reload_cifs_from_folder(self.cif_base_dir)
            except Exception:
                pass

        # ------------------- smoothing (optional SG) -------------------
        try:
            from scipy.signal import savgol_filter as _sg
            self._savgol = _sg
        except Exception:
            self._savgol = None
        self._warned_sg = False

        # ------------------- layout -------------------
        main = ttk.Frame(self, style="CardAlt.TFrame")
        main.pack(fill="both", expand=True, padx=8, pady=8)
        main.columnconfigure(0, weight=0)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        # left panel: list + options
        frame_left = ttk.Frame(main, style="Card.TFrame", padding=8)
        frame_left.grid(row=0, column=0, sticky="ns", padx=(0, 15))
        ttk.Label(frame_left, text="Select from imported", style="Section.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 6))

        self.listbox = tk.Listbox(
            frame_left,
            height=10,
            selectmode="extended",
            exportselection=0,
            bg=self.palette["card"],
            fg="#1c2733",
            bd=0,
            highlightthickness=0,
            selectbackground=self.palette["accent"],
            selectforeground="#1c2733"
        )
        for title in self.file_titles:
            self.listbox.insert(tk.END, title)
        self.listbox.grid(row=1, column=0, sticky="nsew")
        self.listbox.bind("<<ListboxSelect>>", self._on_listbox_select)

        self.var_all_imported = tk.BooleanVar()
        ttk.Checkbutton(frame_left, text="All imported",
                        variable=self.var_all_imported,
                        command=self.toggle_all_imported).grid(row=2, column=0, sticky="w", pady=(10, 0))

        self.var_norm = tk.BooleanVar()
        ttk.Checkbutton(frame_left, text="Norm.",
                        variable=self.var_norm,
                        command=self.plot_selected_spectrum).grid(row=3, column=0, sticky="w", pady=(2, 0))

        # spectral difference
        self.var_spectral_diff = tk.BooleanVar()
        self.chk_spectral_diff = ttk.Checkbutton(frame_left, text="Spectral difference",
                                                 variable=self.var_spectral_diff,
                                                 command=self.on_toggle_spectral_diff)
        self.chk_spectral_diff.grid(row=4, column=0, sticky="w", pady=(6, 0))
        self.var_diff_ref = tk.StringVar()
        self.combo_diff_ref = ttk.Combobox(frame_left, textvariable=self.var_diff_ref, state="readonly", width=24)
        self.combo_diff_ref.grid(row=5, column=0, sticky="w", padx=(12, 0), pady=(2, 0))
        self.combo_diff_ref.grid_remove()

        self.var_zero_line = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame_left, text="Zero line (y=0)",
                        variable=self.var_zero_line,
                        command=self.plot_selected_spectrum).grid(row=6, column=0, sticky="w", pady=(6, 0))

        ttk.Label(frame_left, text="Smoothing").grid(row=7, column=0, sticky="w", pady=(10, 0))
        row_smooth = ttk.Frame(frame_left)
        row_smooth.grid(row=8, column=0, sticky="w")
        self.smooth_method = tk.StringVar(value="None")
        ttk.Label(row_smooth, text="Method:").pack(side="left")
        self.cb_smooth = ttk.Combobox(row_smooth, textvariable=self.smooth_method, state="readonly", width=10,
                                      values=["None", "SG", "MA(5)"])
        self.cb_smooth.pack(side="left", padx=(4, 6))
        self.cb_smooth.bind("<<ComboboxSelected>>", lambda e: self.plot_selected_spectrum())

        # DTA-specific controls (shown only when a DTA/STA trace is selected)
        self.var_dta_x = tk.StringVar()
        self.var_dta_y = tk.StringVar()
        self.var_dta_deriv = tk.StringVar(value="None")
        self.var_dta_scale = tk.StringVar(value="1.0")
        self.var_dta_trace_mode = tk.StringVar(value="deriv_only")
        self.var_dta_time = tk.StringVar()
        self.var_dta_temp = tk.StringVar()

        dta_frame = ttk.LabelFrame(frame_left, text="DTA / STA", style="Card.TLabelframe")
        self.dta_frame = dta_frame
        dta_frame.grid(row=9, column=0, sticky="ew", pady=(10, 0))
        ttk.Label(dta_frame, text="X").grid(row=0, column=0, sticky="e", padx=(6, 4), pady=2)
        ttk.Label(dta_frame, text="Y").grid(row=1, column=0, sticky="e", padx=(6, 4), pady=2)
        self.cb_dta_x = ttk.Combobox(dta_frame, textvariable=self.var_dta_x, state="readonly", width=30)
        self.cb_dta_y = ttk.Combobox(dta_frame, textvariable=self.var_dta_y, state="readonly", width=30)
        self.cb_dta_x.grid(row=0, column=1, sticky="w", padx=(0, 6), pady=2)
        self.cb_dta_y.grid(row=1, column=1, sticky="w", padx=(0, 6), pady=2)
        self.cb_dta_x.bind("<<ComboboxSelected>>", lambda e: self._on_dta_option_changed())
        self.cb_dta_y.bind("<<ComboboxSelected>>", lambda e: self._on_dta_option_changed())

        ttk.Label(dta_frame, text="Y transform").grid(row=2, column=0, sticky="e", padx=(6, 4), pady=2)
        deriv_row = ttk.Frame(dta_frame)
        deriv_row.grid(row=2, column=1, sticky="w", padx=(0, 6), pady=2)
        self.cb_dta_deriv = ttk.Combobox(
            deriv_row, textvariable=self.var_dta_deriv, state="readonly", width=12,
            values=["None", "dY/dt", "dY/dT"]
        )
        self.cb_dta_deriv.pack(side="left")
        ttk.Label(deriv_row, text="×").pack(side="left", padx=(6, 4))
        self.ent_dta_scale = ttk.Entry(deriv_row, textvariable=self.var_dta_scale, width=6)
        self.ent_dta_scale.pack(side="left")
        self.cb_dta_deriv.bind("<<ComboboxSelected>>", lambda e: self._on_dta_option_changed())
        self.ent_dta_scale.bind("<KeyRelease>", lambda e: self._on_dta_option_changed())

        trace_mode_row = ttk.Frame(dta_frame)
        trace_mode_row.grid(row=3, column=0, columnspan=2, sticky="w", padx=(4, 0), pady=(4, 2))
        ttk.Label(trace_mode_row, text="Plot").pack(side="left", padx=(0, 6))
        ttk.Radiobutton(trace_mode_row, text="Derivative only", value="deriv_only",
                        variable=self.var_dta_trace_mode,
                        command=self._on_dta_option_changed).pack(side="left")
        ttk.Radiobutton(trace_mode_row, text="+ original", value="with_base",
                        variable=self.var_dta_trace_mode,
                        command=self._on_dta_option_changed).pack(side="left", padx=(8, 0))
        self.trace_mode_row = trace_mode_row

        ttk.Label(dta_frame, text="Time column").grid(row=4, column=0, sticky="e", padx=(6, 4), pady=2)
        self.cb_dta_time = ttk.Combobox(dta_frame, textvariable=self.var_dta_time, state="readonly", width=30)
        self.cb_dta_time.grid(row=4, column=1, sticky="w", padx=(0, 6), pady=2)
        self.cb_dta_time.bind("<<ComboboxSelected>>", lambda e: self._on_dta_option_changed())
        ttk.Label(dta_frame, text="Temp. column").grid(row=5, column=0, sticky="e", padx=(6, 4), pady=2)
        self.cb_dta_temp = ttk.Combobox(dta_frame, textvariable=self.var_dta_temp, state="readonly", width=30)
        self.cb_dta_temp.grid(row=5, column=1, sticky="w", padx=(0, 6), pady=2)
        self.cb_dta_temp.bind("<<ComboboxSelected>>", lambda e: self._on_dta_option_changed())
        self.cb_dta_time.state(["disabled"])
        self.cb_dta_temp.state(["disabled"])
        self.dta_frame.grid_remove()

        # right panel
        right = ttk.Frame(main)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        self.fig = plt.Figure(figsize=(8, 5), dpi=100)
        self.fig.set_facecolor("white")
        try:
            self.fig.set_layout_engine("none")
        except Exception:
            self.fig.set_constrained_layout(False)
        self.fig.subplots_adjust(left=0.10, right=0.98, bottom=0.18, top=0.90, wspace=0.12, hspace=0.25)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
        self.canvas.get_tk_widget().configure(width=950, height=490)

        # top bar (modes + CIF)
        frame_modes = ttk.Frame(right)
        frame_modes.grid(row=0, column=0, sticky="ew", pady=(0, 2))
        frame_modes.columnconfigure(0, weight=1)

        left_modes = ttk.Frame(frame_modes)
        left_modes.grid(row=0, column=0, sticky="w")
        self.plot_mode = tk.StringVar(value="separate")
        ttk.Label(left_modes, text="Sep. plots").pack(side="left")
        ttk.Radiobutton(left_modes, variable=self.plot_mode, value="separate", text="",
                        command=self.plot_selected_spectrum).pack(side="left", padx=(0, 6))
        ttk.Label(left_modes, text="Stacked").pack(side="left")
        ttk.Radiobutton(left_modes, variable=self.plot_mode, value="stacked", text="",
                        command=self.plot_selected_spectrum).pack(side="left", padx=(0, 8))
        ttk.Label(left_modes, text="Shift:").pack(side="left")
        self.shift_factor = tk.StringVar(value="0.5")
        self.shift_entry = ttk.Entry(left_modes, textvariable=self.shift_factor, width=5)
        self.shift_entry.pack(side="left", padx=(3, 10))
        self.shift_entry.bind("<KeyRelease>", lambda e: self.plot_selected_spectrum())

        ttk.Label(left_modes, text="Colors:").pack(side="left")
        self.color_scheme = tk.StringVar(value="Distinct")
        self.cb_colors = ttk.Combobox(left_modes, textvariable=self.color_scheme, width=14, state="readonly",
                                      values=["Matplotlib cycle", "Distinct", "Hash by name", "Monochrome"])
        self.cb_colors.pack(side="left", padx=(3, 4))
        self.cb_colors.bind("<<ComboboxSelected>>", lambda e: self.plot_selected_spectrum())
        self.base_color = "#3b82f6"

        right_modes = ttk.Frame(frame_modes)
        right_modes.grid(row=0, column=1, sticky="e")
        ttk.Button(right_modes, text="Import CIF…", command=self.import_cif_and_plot_bragg).pack(side="right", padx=(6, 0))
        ttk.Button(right_modes, text="CIF util.", command=self.open_cif_manager).pack(side="right", padx=(6, 0))
        self.var_dark_mode = tk.BooleanVar(value=False)
        ttk.Checkbutton(right_modes, text="Dark mode", variable=self.var_dark_mode,
                        command=self.plot_selected_spectrum).pack(side="right", padx=(6, 6))

        # axis editor
        frame_axes = ttk.Frame(right)
        frame_axes.grid(row=2, column=0, sticky="ew", pady=(5, 0))
        self.title_global = tk.StringVar(value="")
        ttk.Label(frame_axes, text="Title").grid(row=0, column=0)
        ttk.Entry(frame_axes, textvariable=self.title_global, width=22).grid(row=0, column=1, columnspan=2)

        self.x_title = tk.StringVar(value="Raman Shift (cm⁻¹)")
        self.y_title = tk.StringVar(value="Intensity (a.u.)")
        self.xmin = tk.StringVar(value=""); self.xmax = tk.StringVar(value="")
        self.ymin = tk.StringVar(value=""); self.ymax = tk.StringVar(value="")
        ttk.Label(frame_axes, text="X title").grid(row=1, column=0)
        ttk.Entry(frame_axes, textvariable=self.x_title, width=14).grid(row=1, column=1)
        ttk.Label(frame_axes, text="Y title").grid(row=1, column=2)
        ttk.Entry(frame_axes, textvariable=self.y_title, width=14).grid(row=1, column=3)

        ttk.Label(frame_axes, text="X limits").grid(row=2, column=0)
        ttk.Entry(frame_axes, textvariable=self.xmin, width=6).grid(row=2, column=1)
        ttk.Entry(frame_axes, textvariable=self.xmax, width=6).grid(row=2, column=2)
        ttk.Label(frame_axes, text="Y limits").grid(row=2, column=3)
        ttk.Entry(frame_axes, textvariable=self.ymin, width=6).grid(row=2, column=4)
        ttk.Entry(frame_axes, textvariable=self.ymax, width=6).grid(row=2, column=5)
        ttk.Button(frame_axes, text="Auto scale Y", command=self.autoscale_y).grid(row=2, column=7, padx=6)

        self.var_no_title = tk.BooleanVar()
        ttk.Checkbutton(frame_axes, text="No title", variable=self.var_no_title,
                        command=self.plot_selected_spectrum).grid(row=0, column=7)
        self.var_hide_yticks = tk.BooleanVar()
        ttk.Checkbutton(frame_axes, text="No Y ticks", variable=self.var_hide_yticks,
                        command=self.plot_selected_spectrum).grid(row=1, column=7)
        self.var_grid = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame_axes, text="Grid", variable=self.var_grid,
                        command=self.plot_selected_spectrum).grid(row=2, column=8, padx=(10, 0))

        self.title_global.trace_add("write", lambda *_: self.plot_selected_spectrum())
        self.x_title.trace_add("write", lambda *_: self.plot_selected_spectrum())
        self.y_title.trace_add("write", lambda *_: self.plot_selected_spectrum())
        self.xmin.trace_add("write", lambda *_: self.update_plot_axes(apply_only=True))
        self.xmax.trace_add("write", lambda *_: self.update_plot_axes(apply_only=True))
        self.ymin.trace_add("write", lambda *_: self.update_plot_axes(apply_only=True))
        self.ymax.trace_add("write", lambda *_: self.update_plot_axes(apply_only=True))

        # annotations + export
        row3 = ttk.Frame(right); row3.grid(row=3, column=0, sticky="ew", pady=(6, 0))
        row3.columnconfigure(0, weight=1)
        frame_anno = ttk.Frame(row3); frame_anno.grid(row=0, column=0, sticky="w")
        self._annotations = []; self._cid_click = None
        self.var_anno_on = tk.BooleanVar(value=False)
        self.var_anno_text = tk.StringVar(value="")
        self.var_anno_style = tk.StringVar(value="label")
        self.var_anno_color = tk.StringVar(value="auto")
        self.var_anno_size = tk.StringVar(value="10")

        ttk.Checkbutton(frame_anno, text="Annotate",
                        variable=self.var_anno_on,
                        command=self._toggle_annotate).pack(side="left", padx=(0, 8))
        ttk.Label(frame_anno, text="Text").pack(side="left")
        ttk.Entry(frame_anno, textvariable=self.var_anno_text, width=16).pack(side="left", padx=(4, 10))
        ttk.Combobox(frame_anno, state="readonly", width=8,
                     values=["label", "arrow", "Varrow"], textvariable=self.var_anno_style).pack(side="left", padx=(0, 10))
        ttk.Label(frame_anno, text="Color").pack(side="left")
        ttk.Combobox(frame_anno, state="readonly", width=10,
                     values=["auto","black","white","navy","crimson","seagreen","darkorange","purple","teal","brown","indigo"],
                     textvariable=self.var_anno_color).pack(side="left", padx=(4, 10))
        ttk.Label(frame_anno, text="Size").pack(side="left")
        ttk.Entry(frame_anno, textvariable=self.var_anno_size, width=4).pack(side="left", padx=(4, 10))
        ttk.Button(frame_anno, text="Undo", width=6, command=self._anno_undo).pack(side="left", padx=(6, 2))
        ttk.Button(frame_anno, text="Clear", width=6, command=self._anno_clear).pack(side="left", padx=(2, 8))

        ttk.Label(row3, text="|").grid(row=0, column=1, padx=8)
        frame_export = ttk.Frame(row3); frame_export.grid(row=0, column=2, sticky="e")
        self.var_export_png = tk.BooleanVar(value=True)
        self.var_export_svg = tk.BooleanVar(value=False)
        self.var_export_pdf = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame_export, text="PNG", variable=self.var_export_png).pack(side="left", padx=2)
        ttk.Checkbutton(frame_export, text="SVG", variable=self.var_export_svg).pack(side="left", padx=2)
        ttk.Checkbutton(frame_export, text="PDF", variable=self.var_export_pdf).pack(side="left", padx=2)
        ttk.Button(frame_export, text="Export", command=self.export_all_formats).pack(side="left", padx=8)

        self.status = tk.StringVar(value="x: —   y: —")
        ttk.Label(right, textvariable=self.status).grid(row=4, column=0, sticky="w", pady=(2, 0))
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self.canvas.mpl_connect("key_press_event", self._on_key_press)

        self.plot_selected_spectrum()

    # ------------------------------- helpers -------------------------------

    def _apply_smoothing(self, x, y):
        m = self.smooth_method.get()
        if m == "None":
            return y
        if m.startswith("MA"):
            w = 5
            kernel = np.ones(w, dtype=float) / float(w)
            return np.convolve(y, kernel, mode="same")
        if m == "SG":
            w, p = 7, 2
            if self._savgol is None:
                if not self._warned_sg:
                    messagebox.showinfo("Smoothing", "SciPy unavailable → using MA(5) instead of SG.", parent=self)
                    self._warned_sg = True
                kernel = np.ones(5, dtype=float) / 5.0
                return np.convolve(y, kernel, mode="same")
            return self._savgol(y, window_length=w, polyorder=p, mode="interp")
        return y

    def _style_axes(self, ax):
        dark = self.var_dark_mode.get()
        fg = "white" if dark else "black"
        bg = "#1e1e1e" if dark else "white"
        gridc = "#444" if dark else "#bbb"
        ax.set_facecolor(bg)
        for sp in ax.spines.values():
            sp.set_color(fg)
        ax.tick_params(colors=fg, labelsize=9, length=3, width=1)
        ax.xaxis.label.set_color(fg)
        ax.yaxis.label.set_color(fg)
        ax.title.set_color(fg)
        if self.var_grid.get():
            ax.grid(True, which="both", color=gridc, alpha=0.5)
        else:
            ax.grid(False)

    def _color_from_scheme(self, i, label, total):
        scheme = self.color_scheme.get()
        if scheme == "Matplotlib cycle":
            return None
        if scheme == "Distinct":
            hue = i / max(total, 1)
            r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.9)
            return (r, g, b)
        if scheme == "Hash by name":
            h = (hash(label) % 360) / 360.0
            r, g, b = colorsys.hsv_to_rgb(h, 0.70, 0.9)
            return (r, g, b)
        if scheme == "Monochrome":
            return self.base_color
        return self.COLORS[i % len(self.COLORS)]

    def _num(self, s: str):
        s = (s or "").strip().replace(",", ".")
        if s in {"", "-", "+", ".", "+.", "-."}:
            return None
        try:
            return float(s)
        except Exception:
            return None

    def _effective_xlim(self, ax):
        xmin_ax, xmax_ax = ax.get_xlim()

        def _pf(t):
            try:
                return float(t)
            except Exception:
                return None

        xmin_in = _pf(self.xmin.get()) if hasattr(self, "xmin") else None
        xmax_in = _pf(self.xmax.get()) if hasattr(self, "xmax") else None
        if xmin_in is not None and xmax_in is not None and xmin_in < xmax_in:
            return xmin_in, xmax_in
        if xmin_in is not None:
            return xmin_in, xmax_ax
        if xmax_in is not None:
            return xmin_ax, xmax_in
        return xmin_ax, xmax_ax

    # ------------------------------- DTA helpers -------------------------------

    def _find_best_column(self, columns: List[str], keyword: str) -> str | None:
        for col in columns:
            if keyword.lower() in col.lower():
                return col
        return None

    def _ensure_dta_defaults(self, path: str, payload: Dict[str, Any]) -> Dict[str, str]:
        if path in self._dta_state:
            state = self._dta_state[path]
            if "scale" not in state:
                state["scale"] = 1.0
            return state
        df = payload.get("df")
        meta = payload.get("meta") or {}
        cols = list(df.columns) if df is not None else []
        canonical = meta.get("canonical_map", {}) or {}

        x_col = canonical.get("T_C") or canonical.get("time_min") or self._find_best_column(cols, "temp") \
            or self._find_best_column(cols, "time") or (cols[0] if cols else "")
        y_col = (canonical.get("DSC_mW_mg") or canonical.get("HF_mW") or canonical.get("TG_pct")
                 or canonical.get("DTG_pct_min") or (cols[1] if len(cols) > 1 else (cols[0] if cols else "")))

        state = {
            "x": x_col,
            "y": y_col,
            "deriv": "none",
            "trace_mode": "deriv_only",
            "time_col": canonical.get("time_min") or self._find_best_column(cols, "time") or "",
            "temp_col": canonical.get("T_C") or self._find_best_column(cols, "temp") or "",
            "scale": 1.0,
        }
        self._dta_state[path] = state
        return state

    def _refresh_dta_controls(self):
        self._dta_active_path = None
        idxs = self.listbox.curselection()
        for idx in idxs:
            path = self.file_paths[idx]
            try:
                payload = self._load_any(path)
            except Exception:
                continue
            if payload.get("kind") not in {"DTA", "TA_SDT"}:
                continue
            df = payload.get("df")
            cols = list(df.columns) if df is not None else []
            state = self._ensure_dta_defaults(path, payload)
            self._dta_active_path = path
            for cb in (self.cb_dta_x, self.cb_dta_y, self.cb_dta_time, self.cb_dta_temp):
                cb["values"] = cols
            self.var_dta_x.set(state.get("x", ""))
            self.var_dta_y.set(state.get("y", ""))
            deriv_mode = state.get("deriv", "none")
            mode_to_label = {"none": "None", "time": "dY/dt", "temp": "dY/dT"}
            self.var_dta_deriv.set(mode_to_label.get(deriv_mode, "None"))
            self.var_dta_trace_mode.set(state.get("trace_mode", "deriv_only"))
            self.var_dta_time.set(state.get("time_col", ""))
            self.var_dta_temp.set(state.get("temp_col", ""))
            self.var_dta_scale.set(str(state.get("scale", 1.0)))
            self._update_dta_basis_state(deriv_mode)
            self._update_dta_trace_visibility(deriv_mode)
            self.dta_frame.grid()
            return
        self.dta_frame.grid_remove()

    def _update_dta_basis_state(self, mode: str):
        if mode == "time":
            self.cb_dta_time.state(["!disabled"])
            self.cb_dta_temp.state(["disabled"])
        elif mode == "temp":
            self.cb_dta_time.state(["disabled"])
            self.cb_dta_temp.state(["!disabled"])
        else:
            self.cb_dta_time.state(["disabled"])
            self.cb_dta_temp.state(["disabled"])

    def _update_dta_trace_visibility(self, mode: str):
        if mode == "none":
            if self.trace_mode_row.winfo_ismapped():
                self.trace_mode_row.grid_remove()
        else:
            if not self.trace_mode_row.winfo_ismapped():
                self.trace_mode_row.grid()

    def _propagate_dta_state(self, source_state: Dict[str, str | float]):
        idxs = self.listbox.curselection()
        for idx in idxs:
            path = self.file_paths[idx]
            try:
                payload = self._load_any(path)
            except Exception:
                continue
            if payload.get("kind") not in {"DTA", "TA_SDT"}:
                continue
            df = payload.get("df")
            if df is None:
                continue
            state = self._ensure_dta_defaults(path, payload)
            cols = list(df.columns)
            if source_state.get("x") in cols:
                state["x"] = source_state.get("x", state.get("x", ""))
            if source_state.get("y") in cols:
                state["y"] = source_state.get("y", state.get("y", ""))
            state["deriv"] = source_state.get("deriv", state.get("deriv", "none"))
            if source_state.get("trace_mode"):
                state["trace_mode"] = source_state.get("trace_mode", state.get("trace_mode", "deriv_only"))
            time_col = source_state.get("time_col")
            temp_col = source_state.get("temp_col")
            if time_col in cols:
                state["time_col"] = time_col
            if temp_col in cols:
                state["temp_col"] = temp_col
            try:
                state["scale"] = float(source_state.get("scale", 1.0))
            except Exception:
                state["scale"] = 1.0

    def _on_dta_option_changed(self):
        if not self._dta_active_path or self._dta_active_path not in self._dta_state:
            return
        state = self._dta_state[self._dta_active_path]
        state["x"] = self.var_dta_x.get() or state.get("x", "")
        state["y"] = self.var_dta_y.get() or state.get("y", "")
        label_to_mode = {"None": "none", "dY/dt": "time", "dY/dT": "temp"}
        deriv_mode = label_to_mode.get(self.var_dta_deriv.get(), "none")
        state["deriv"] = deriv_mode
        if deriv_mode != "none":
            state["trace_mode"] = self.var_dta_trace_mode.get() or state.get("trace_mode", "deriv_only")
        else:
            self.var_dta_trace_mode.set(state.get("trace_mode", "deriv_only"))
        state["time_col"] = self.var_dta_time.get() or state.get("time_col", "")
        state["temp_col"] = self.var_dta_temp.get() or state.get("temp_col", "")
        try:
            state["scale"] = float(self.var_dta_scale.get())
        except Exception:
            pass
        self._update_dta_basis_state(deriv_mode)
        self._update_dta_trace_visibility(deriv_mode)
        self._propagate_dta_state(state)
        self.plot_selected_spectrum()

    def _compute_derivative(self, y: np.ndarray, x: np.ndarray) -> np.ndarray:
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 3:
            return np.full_like(y, np.nan, dtype=float)
        out = np.full_like(y, np.nan, dtype=float)
        out[mask] = np.gradient(y[mask], x[mask])
        return out

    def _resolve_payload_traces(
        self,
        path: str,
        payload: Dict[str, Any],
        fallback_label: str,
        *,
        include_dta_base: bool = False,
    ) -> List[Tuple[np.ndarray, np.ndarray, str]]:
        kind = payload.get("kind")
        if kind in {"DTA", "TA_SDT"}:
            df = payload.get("df")
            if df is None:
                return [(np.array([]), np.array([]), fallback_label)]
            state = self._ensure_dta_defaults(path, payload)
            cols = list(df.columns)
            x_col = state.get("x") or (cols[0] if cols else "")
            y_col = state.get("y") or ((cols[1] if len(cols) > 1 else cols[0]) if cols else "")
            deriv_mode = state.get("deriv", "none")
            time_col = state.get("time_col") or self._find_best_column(cols, "time")
            temp_col = state.get("temp_col") or self._find_best_column(cols, "temp")
            try:
                scale = float(state.get("scale", 1.0))
            except Exception:
                scale = 1.0

            x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
            y_base = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
            base_label = fallback_label

            traces: List[Tuple[np.ndarray, np.ndarray, str]] = []
            if deriv_mode == "time":
                base_col = time_col or x_col
                base = pd.to_numeric(df[base_col], errors="coerce").to_numpy(dtype=float)
                y = self._compute_derivative(y_base, base)
                label = f"{fallback_label} · d({y_col})/d({base_col})"
                state["time_col"] = base_col
                traces.append((x, y * scale, label))
                if include_dta_base and state.get("trace_mode", "deriv_only") == "with_base":
                    traces.append((x, y_base, f"{base_label} (orig {y_col})"))
            elif deriv_mode == "temp":
                base_col = temp_col or x_col
                base = pd.to_numeric(df[base_col], errors="coerce").to_numpy(dtype=float)
                y = self._compute_derivative(y_base, base)
                label = f"{fallback_label} · d({y_col})/d({base_col})"
                state["temp_col"] = base_col
                traces.append((x, y * scale, label))
                if include_dta_base and state.get("trace_mode", "deriv_only") == "with_base":
                    traces.append((x, y_base, f"{base_label} (orig {y_col})"))
            else:
                traces.append((x, y_base, base_label))
            state["x"] = x_col
            state["y"] = y_col
            state["scale"] = scale
            return traces

        if kind == "XY":
            return [(payload.get("x", np.array([])), payload.get("y", np.array([])), fallback_label)]

        df = payload.get("df")
        if df is not None:
            x_arr, y_arr, info = self._pick_ta_xy(df)
            return [(x_arr, y_arr, info.get("label", fallback_label))]
        return [(payload.get("x", np.array([])), payload.get("y", np.array([])), fallback_label)]

    # ------------------------------- CIF UI -------------------------------

    def import_cif_and_plot_bragg(self):
        path = filedialog.askopenfilename(title="Select CIF file",
                                          filetypes=[("CIF files", "*.cif"), ("All files", "*.*")])
        if not path:
            return
        apath = os.path.abspath(path)
        for s in self._cif_bragg_series:
            if os.path.abspath(s["path"]) == apath:
                s["visible"] = True
                self.plot_selected_spectrum()
                return
        try:
            peaks = bragg_peaks_from_cif_generic(path, two_theta_max=80.0, hkl_max=6)
        except Exception as e:
            messagebox.showerror("CIF import error", f"Could not read CIF:\n{e}", parent=self)
            return
        if not peaks:
            messagebox.showinfo("CIF import", "No Bragg peak found below 80° 2θ.", parent=self)
            return
        label = os.path.basename(path)
        color = self._cif_colors[len(self._cif_bragg_series) % len(self._cif_colors)]
        self._cif_bragg_series.append({
            "path": path, "label": label, "plot_label": "",
            "peaks": peaks, "visible": True, "color": color, "pad": 0.03
        })
        self.plot_selected_spectrum()

    def open_cif_manager(self):
        if self._cif_manager_win is not None and self._cif_manager_win.winfo_exists():
            self._cif_manager_win.lift()
            return
        win = tk.Toplevel(self)
        self._cif_manager_win = win
        win.title("CIF manager")
        win.geometry("680x280")
        win.transient(self)

        # --- First row: folder, label pos, Bragg height × ---
        top = ttk.Frame(win); top.pack(fill="x", padx=8, pady=6)

        ttk.Button(top, text="Set CIF folder…",
                   command=self._choose_cif_folder_and_reload).pack(side="left")

        pos_frame = ttk.Frame(top); pos_frame.pack(side="left", padx=12)
        ttk.Label(pos_frame, text="Label pos.").pack(side="left")
        cb_pos = ttk.Combobox(pos_frame, textvariable=self._cif_label_pos, state="readonly", width=11,
                              values=["right_out", "left_out", "right_in", "left_in"])
        cb_pos.pack(side="left", padx=4)
        cb_pos.bind("<<ComboboxSelected>>",
                    lambda e: self._debounce(("cif_pos", "all"), 120, self.plot_selected_spectrum))

        ttk.Label(pos_frame, text="Bragg height ×").pack(side="left", padx=(10, 2))
        ent_bh = ttk.Entry(pos_frame, textvariable=self.var_bragg_h_scale, width=6)
        ent_bh.pack(side="left")
        ent_bh.bind("<KeyRelease>", lambda e: (self._save_cfg_cif(), self.plot_selected_spectrum()))

        # --- Scrollable list of CIF entries (toggle/color/label/pad) ---
        list_frame = ttk.Frame(win); list_frame.pack(fill="both", expand=True, padx=8, pady=4)
        canvas = tk.Canvas(list_frame, borderwidth=0); inner = ttk.Frame(canvas)
        vsb = ttk.Scrollbar(list_frame, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y"); canvas.pack(side="left", fill="both", expand=True)
        canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        self._cif_manager_inner = inner
        self._rebuild_cif_manager_list()

    # optional alias
    def _cif_choose_folder(self):
        self._choose_cif_folder_and_reload()

    def _save_cfg_cif(self):
        self._cfg["cif_base_dir"] = self.cif_base_dir
        try:
            self._cfg["bragg_height_scale"] = float(self.var_bragg_h_scale.get())
        except Exception:
            self._cfg["bragg_height_scale"] = 1.0
        _cfg_save(self._cfg)

    def _choose_cif_folder_and_reload(self):
        initdir = self.cif_base_dir if (self.cif_base_dir and os.path.isdir(self.cif_base_dir)) else os.getcwd()
        d = filedialog.askdirectory(initialdir=initdir, title="Select CIF folder")
        if not d:
            return
        self.cif_base_dir = d
        self._save_cfg_cif()
        try:
            self.reload_cifs_from_folder(d)
        except Exception as e:
            print("reload_cifs_from_folder failed:", e)
        self.plot_selected_spectrum()

    def reload_cifs_from_folder(self, folder_path: str):
        folder = folder_path
        if not folder:
            return
        cif_paths = list_cif_files_case_insensitive(folder)
        if not cif_paths:
            messagebox.showinfo("CIF manager", "No CIF files found in the selected folder.", parent=self)
            self._rebuild_cif_manager_list()
            return
        added = 0
        for p in cif_paths:
            ap = os.path.abspath(p)
            if any(os.path.abspath(s.get("path", "")) == ap for s in self._cif_bragg_series):
                continue
            try:
                peaks = bragg_peaks_from_cif_generic(p, two_theta_max=80.0, hkl_max=6)
            except Exception:
                continue
            color = self._cif_colors[len(self._cif_bragg_series) % len(self._cif_colors)]
            self._cif_bragg_series.append({
                "path": p, "label": os.path.basename(p), "plot_label": "",
                "peaks": peaks, "visible": False, "color": color, "pad": 0.03
            })
            added += 1
        self._rebuild_cif_manager_list()
        self.plot_selected_spectrum()
        messagebox.showinfo("CIF manager", f"Found {len(cif_paths)} CIF files.\nAdded {added} new entries.", parent=self)

    def _rebuild_cif_manager_list(self):
        inner = self._cif_manager_inner
        if inner is None or not inner.winfo_exists():
            return
        for c in inner.winfo_children():
            c.destroy()
        for serie in self._cif_bragg_series:
            row = ttk.Frame(inner); row.pack(fill="x", pady=2)
            var_show = tk.BooleanVar(value=serie.get("visible", False))
            ttk.Checkbutton(row, text=serie.get("label", os.path.basename(serie.get("path", ""))),
                            variable=var_show,
                            command=lambda s=serie, v=var_show: self._cif_toggle_visible(s, v.get())).pack(side="left")
            name_var = tk.StringVar(value=serie.get("plot_label", ""))
            ent_name = ttk.Entry(row, textvariable=name_var, width=18); ent_name.pack(side="left", padx=4)
            name_var.trace_add("write", lambda *_nv, s=serie, v=name_var: self._on_cif_field(s, "plot_label", v.get()))
            col_var = tk.StringVar(value=serie.get("color", "crimson"))
            ent_col = ttk.Entry(row, textvariable=col_var, width=9); ent_col.pack(side="left", padx=4)
            col_var.trace_add("write", lambda *_nv, s=serie, v=col_var: self._on_cif_field(s, "color", v.get() or "crimson"))
            pad_var = tk.StringVar(value=str(serie.get("pad", "0.03")))
            ent_pad = ttk.Entry(row, textvariable=pad_var, width=5); ent_pad.pack(side="left", padx=4)

            # capture 'serie' and 'pad_var' to avoid late-binding bug
            def _pad_change_for(s, pv):
                txt = (pv.get() or "").strip()
                try:
                    s["pad"] = float(txt)
                except Exception:
                    s["pad"] = 0.03
                self.plot_selected_spectrum()

            pad_var.trace_add("write", lambda *_i, s=serie, pv=pad_var: _pad_change_for(s, pv))

    def _on_cif_field(self, serie: dict, key: str, value: str):
        serie[key] = value
        self._debounce(("cif_field", id(serie), key), 80, self.plot_selected_spectrum)

    def _cif_toggle_visible(self, serie, visible):
        serie["visible"] = bool(visible)
        self._debounce(("cif_vis", id(serie)), 80, self.plot_selected_spectrum)

    def _draw_cif_bragg_markers(self):
        """Draw Bragg vlines on **each** axes. Fix: avoid double-offset; respect per-axis xlim."""
        if not self._cif_bragg_series:
            return
        axes = self.fig.get_axes()
        if not axes:
            return

        # Only series toggled visible
        visible_series = [c for c in self._cif_bragg_series if c.get("visible", True)]
        if not visible_series:
            return

        # user control: height of Bragg vlines only
        try:
            hmul = float(self.var_bragg_h_scale.get())
        except Exception:
            hmul = 1.0
        hmul = max(0.1, hmul)

        for ax in axes:
            xmin, xmax = self._effective_xlim(ax)
            ymin_orig, ymax_orig = ax.get_ylim()
            height = (ymax_orig - ymin_orig) or 1.0

            # ----- density-aware spacing, computed for THIS axis window -----
            n_in_view = []
            for serie in visible_series:
                pks = [tt for (tt, _hkl, _d) in serie["peaks"] if xmin <= tt <= xmax]
                n_in_view.append(len(pks))
            max_n = max(n_in_view) if n_in_view else 0
            row_space_factor = 1.0 + min(2.0, 0.02 * max_n)  # up to ×3

            # ----- geometry base (based on ORIGINAL y-lims of this axis) -----
            base_band_h = 0.03 * height
            base_spacing = 1.8 * base_band_h
            base_offset = 0.02 * height

            band_h   = base_band_h * hmul                  # ONLY vline height scaling
            spacing  = base_spacing * row_space_factor     # row gap scales with density
            off_down = base_offset * row_space_factor

            # extend y-range downward ONCE (no double-subtraction later)
            extra_down = off_down + len(visible_series) * spacing
            ax.set_ylim(ymin_orig - extra_down, ymax_orig)

            # adjust x-limits if labels are outside (pad per series)
            pos = self._cif_label_pos.get()
            xspan = (xmax - xmin) or 1.0
            if pos in ("right_out", "left_out"):
                max_pad = max(float(c.get("pad", 0.03)) for c in visible_series)
                pad_abs = max_pad * xspan
                if pos == "right_out":
                    ax.set_xlim(xmin, xmax + pad_abs)
                    xmin, xmax = ax.get_xlim()
                    xspan = (xmax - xmin) or 1.0
                else:
                    ax.set_xlim(xmin - pad_abs, xmax)
                    xmin, xmax = ax.get_xlim()
                    xspan = (xmax - xmin) or 1.0

            ymin_new, ymax_new = ax.get_ylim()

            # draw each visible CIF row
            for i, serie in enumerate(visible_series):
                peaks = serie["peaks"]
                color = serie.get("color", "crimson")
                disp_label = serie.get("plot_label") or serie.get("label") or f"CIF {i+1}"
                cif_pad = float(serie.get("pad", 0.03))

                # FIX: place rows relative to ORIGINAL ymin, minus ONE offset
                y0 = (ymin_orig - off_down) - i * spacing
                y1 = y0 + band_h * 0.6

                # draw Bragg vlines (constant linewidth)
                for tt, hkl, d in peaks:
                    if tt < xmin or tt > xmax:
                        continue
                    ax.vlines(tt, y0, y1, colors=color, linewidth=1.2)

                # label placement
                if pos == "right_out":
                    x_text, dytext, ha = xmax - cif_pad * xspan, band_h * 0.75, "right"
                elif pos == "left_out":
                    x_text, dytext, ha = xmin + cif_pad * xspan, band_h * 0.75, "left"
                elif pos == "right_in":
                    x_text, dytext, ha = xmax - cif_pad * xspan, 0, "right"
                else:
                    x_text, dytext, ha = xmin + cif_pad * xspan, 0, "left"

                ax.text(
                    x_text, y0 + dytext, disp_label,
                    va="bottom", ha=ha, fontsize=7, color=color,
                    bbox=dict(boxstyle="round,pad=0.1", edgecolor="none", fc="white", lw=0.6, alpha=0.9),
                    clip_on=False,
                )

    # ------------------------------- selection & export -------------------------------

    def _debounce(self, key, ms, func):
        try:
            h = self._debouncers.get(key)
            if h is not None:
                self.after_cancel(h)
        except Exception:
            pass
        self._debouncers[key] = self.after(ms, func)

    def _on_listbox_select(self, event=None):
        selected = set(self.listbox.curselection())
        all_selected = set(range(self.listbox.size()))
        self.var_all_imported.set(selected == all_selected)
        self.on_selection_change()

    def toggle_all_imported(self):
        if self.var_all_imported.get():
            self.listbox.selection_set(0, tk.END)
        else:
            self.listbox.selection_clear(0, tk.END)
        # Reuse the same selection-change hook so that spectral-diff UI state
        # (combobox visibility, reference list, etc.) stays in sync when this
        # shortcut checkbox is toggled.
        self.on_selection_change()

    def export_all_formats(self):
        basename = filedialog.asksaveasfilename(defaultextension=".png",
                    filetypes=[("PNG", "*.png"), ("SVG", "*.svg"), ("PDF", "*.pdf"), ("All files", "*.*")],
                    title="Export plot as...")
        if not basename:
            return
        basename = os.path.splitext(basename)[0]
        errors = []
        self.canvas.draw()
        face = self.fig.get_facecolor()
        if self.var_export_png.get():
            try:
                self.fig.savefig(basename + ".png", dpi=300, bbox_inches="tight", facecolor=face)
            except Exception as e:
                errors.append(f"PNG: {e}")
        if self.var_export_svg.get():
            try:
                self.fig.savefig(basename + ".svg", bbox_inches="tight", facecolor=face)
            except Exception as e:
                errors.append(f"SVG: {e}")
        if self.var_export_pdf.get():
            try:
                self.fig.savefig(basename + ".pdf", bbox_inches="tight", facecolor=face)
            except Exception as e:
                errors.append(f"PDF: {e}")
        if errors:
            messagebox.showerror("Export", "Some exports failed:\n" + "\n".join(errors), parent=self)
        else:
            messagebox.showinfo("Export", "Plot exported successfully.", parent=self)

    def on_selection_change(self, event=None):
        if self.var_spectral_diff.get():
            idxs = self.listbox.curselection()
            if len(idxs) < 2:
                self.var_spectral_diff.set(False)
                self.combo_diff_ref.grid_remove()
            else:
                ref_titles = [self.file_titles[idx] for idx in idxs]
                self.combo_diff_ref["values"] = ref_titles
                if self.var_diff_ref.get() not in ref_titles:
                    self.var_diff_ref.set(ref_titles[0])
        self._refresh_dta_controls()
        self.plot_selected_spectrum()

    def on_toggle_spectral_diff(self):
        idxs = self.listbox.curselection()
        if self.var_spectral_diff.get():
            if len(idxs) < 2:
                messagebox.showinfo("Info", "Select at least two spectra to enable spectral difference.", parent=self)
                self.var_spectral_diff.set(False)
                self.combo_diff_ref.grid_remove()
                self.plot_selected_spectrum()
                return
            self.var_norm.set(True)
            ref_titles = [self.file_titles[idx] for idx in idxs]
            self.combo_diff_ref["values"] = ref_titles
            if self.var_diff_ref.get() not in ref_titles:
                self.var_diff_ref.set(ref_titles[0])
            self.combo_diff_ref.grid()
            self.combo_diff_ref.bind("<<ComboboxSelected>>", lambda e: self.plot_selected_spectrum())
        else:
            self.combo_diff_ref.grid_remove()
        self.plot_selected_spectrum()

    # ------------------------------- annotate -------------------------------

    def _toggle_annotate(self):
        if self.var_anno_on.get():
            if self._cid_click is None:
                self._cid_click = self.canvas.mpl_connect("button_press_event", self._on_annotate_click)
        else:
            if self._cid_click is not None:
                self.canvas.mpl_disconnect(self._cid_click)
                self._cid_click = None

    def _on_annotate_click(self, event):
        if event.inaxes is None:
            return
        if event.button == 3:  # right-click → delete nearest
            target = self._find_nearest_annotation(event, tol_px=15)
            if target is not None:
                try:
                    target.remove()
                except Exception:
                    pass
                self._annotations = [a for a in self._annotations if a is not target]
                self.canvas.draw_idle()
            return
        if event.button != 1:
            return

        txt = self.var_anno_text.get().strip() or f"{event.xdata:.0f}"
        is_dark = bool(self.var_dark_mode.get())
        color = self.var_anno_color.get()
        if color == "auto":
            color = "white" if is_dark else "black"
        try:
            fz = max(6, int(float(self.var_anno_size.get())))
        except Exception:
            fz = 10

        style = self.var_anno_style.get()
        ax = event.inaxes
        if style == "Varrow":
            artist = ax.annotate(txt, xy=(event.xdata, event.ydata),
                                 xytext=(0, 18), textcoords="offset points",
                                 ha="center", va="bottom", fontsize=fz, color=color,
                                 arrowprops=dict(arrowstyle="->", color=color, lw=1))
        elif style == "arrow":
            artist = ax.annotate(txt, xy=(event.xdata, event.ydata),
                                 xytext=(10, 10), textcoords="offset points",
                                 fontsize=fz, color=color,
                                 arrowprops=dict(arrowstyle="->", color=color, lw=1))
        else:
            artist = ax.text(event.xdata, event.ydata, txt,
                             fontsize=fz, color=color,
                             bbox=dict(boxstyle="round,pad=0.2", ec=color, fc="none", lw=0.6, alpha=0.9))
        self._annotations.append(artist)
        self.canvas.draw_idle()

    def _anno_undo(self):
        if self._annotations:
            try:
                self._annotations.pop().remove()
            except Exception:
                pass
            self.canvas.draw_idle()

    def _anno_clear(self):
        changed = False
        while self._annotations:
            try:
                self._annotations.pop().remove()
                changed = True
            except Exception:
                break
        if changed:
            self.canvas.draw_idle()

    def _find_nearest_annotation(self, event, tol_px=15):
        ax = event.inaxes
        if ax is None:
            return None
        ex, ey = event.x, event.y
        to_px = ax.transData.transform
        best, best_d = None, float("inf")
        for artist in list(self._annotations):
            if artist.axes is not ax:
                continue
            if hasattr(artist, "xy"):
                x, y = artist.xy
            else:
                x, y = artist.get_position()
            px, py = to_px((x, y))
            d = ((px - ex) ** 2 + (py - ey) ** 2) ** 0.5
            if d < best_d:
                best_d, best = d, artist
        return best if best_d <= tol_px else None

    def _on_key_press(self, event):
        if event.key in ("backspace", "delete"):
            self._anno_undo()
        elif event.key == "escape":
            if self.var_anno_on.get():
                self.var_anno_on.set(False)
                self._toggle_annotate()

    # ------------------------------- plotting core -------------------------------

    def plot_selected_spectrum(self, event=None):
        idxs = self.listbox.curselection()
        if not idxs:
            self.fig.clf()
            self.canvas.draw()
            return

        self._annotations.clear()
        try:
            self.fig.set_layout_engine("none")
        except Exception:
            self.fig.set_constrained_layout(False)
        self.fig.set_facecolor("#1e1e1e" if self.var_dark_mode.get() else "white")
        self.fig.subplots_adjust(left=0.10, right=0.98, bottom=0.18, top=0.90, wspace=0.12, hspace=0.25)
        self._lines = []

        # spectral difference reference
        use_diff = self.var_spectral_diff.get() and len(idxs) > 1
        x_ref = y_ref = None
        ref_title = None
        if use_diff:
            titles = [self.file_titles[i] for i in idxs]
            if self.var_diff_ref.get() not in titles:
                self.var_diff_ref.set(titles[0])
            ref_title = self.var_diff_ref.get()
            ref_idx = self.file_titles.index(ref_title)
            payload_ref = self._load_any(self.file_paths[ref_idx])
            ref_traces = self._resolve_payload_traces(
                self.file_paths[ref_idx], payload_ref, ref_title, include_dta_base=False
            )
            if not ref_traces:
                return
            x_ref, y_ref, _ = ref_traces[0]
            # normalise ref on selected X-range
            try:
                norm_xmin = float(self.xmin.get()) if self.xmin.get() else None
                norm_xmax = float(self.xmax.get()) if self.xmax.get() else None
            except Exception:
                norm_xmin = norm_xmax = None
            mask = np.ones_like(x_ref, dtype=bool)
            if norm_xmin is not None:
                mask &= (x_ref >= norm_xmin)
            if norm_xmax is not None:
                mask &= (x_ref <= norm_xmax)
            finite_mask = mask & np.isfinite(x_ref) & np.isfinite(y_ref)
            if finite_mask.sum() >= 2:
                area_ref = np.trapz(y_ref[finite_mask], x_ref[finite_mask])
                if abs(area_ref) > 1e-12:
                    y_ref = y_ref / area_ref * 100

        # load & prep each selected spectrum
        spectra, labels = [], []
        try:
            norm_xmin = float(self.xmin.get()) if self.xmin.get() else None
            norm_xmax = float(self.xmax.get()) if self.xmax.get() else None
        except Exception:
            norm_xmin = norm_xmax = None

        for idx in idxs:
            title = self.file_titles[idx]
            if use_diff and title == ref_title:
                continue
            payload = self._load_any(self.file_paths[idx])
            traces = self._resolve_payload_traces(
                self.file_paths[idx], payload, title, include_dta_base=True
            )

            for x, y, title_used in traces:
                # normalisation
                if self.var_norm.get() or use_diff:
                    mask = np.ones_like(x, dtype=bool)
                    if norm_xmin is not None:
                        mask &= (x >= norm_xmin)
                    if norm_xmax is not None:
                        mask &= (x <= norm_xmax)
                    finite_mask = mask & np.isfinite(x) & np.isfinite(y)
                    x_for_norm = x[finite_mask]
                    y_for_norm = y[finite_mask]
                    if len(x_for_norm) > 1:
                        area = np.trapz(y_for_norm, x_for_norm)
                    else:
                        finite_full = np.isfinite(x) & np.isfinite(y)
                        area = np.trapz(y[finite_full], x[finite_full]) if finite_full.sum() > 1 else 0.0
                    if abs(area) > 1e-12:
                        y = y / area * 100

                # difference
                if use_diff and x_ref is not None:
                    ref_mask = np.isfinite(x_ref) & np.isfinite(y_ref)
                    if ref_mask.sum() >= 2:
                        y = y - np.interp(x, x_ref[ref_mask], y_ref[ref_mask])

                y = self._apply_smoothing(x, y)
                spectra.append((x, y))
                labels.append(title_used)

        self.current_data = spectra

        # draw
        self.fig.clf()
        self.fig.set_facecolor("#1e1e1e" if self.var_dark_mode.get() else "white")
        mode = self.plot_mode.get()
        try:
            shift_factor = float(self.shift_factor.get())
        except Exception:
            shift_factor = 0.5
        global_title = self.title_global.get().strip()

        if mode == "separate":
            num = len(spectra)
            axes = np.atleast_1d(self.fig.subplots(1, num, sharey=True)).ravel()
            for i, (x, y) in enumerate(spectra):
                ax = axes[i]
                color = self._color_from_scheme(i, labels[i], num)
                ln = (ax.plot(x, y, lw=1, label=labels[i])[0] if color is None
                      else ax.plot(x, y, color=color, lw=1, label=labels[i])[0])
                self._lines.append(ln)
                ax.set_xlabel(self.x_title.get())
                if not self.var_no_title.get():
                    ax.set_title(global_title if global_title else labels[i])
                if self.var_hide_yticks.get():
                    if i == 0:
                        ax.set_ylabel(self.y_title.get())
                    else:
                        ax.set_ylabel("")
                    ax.tick_params(left=False, labelleft=False)
                else:
                    if i == 0:
                        ax.set_ylabel(self.y_title.get())
                    else:
                        ax.set_ylabel("")
                        ax.tick_params(labelleft=False)
                if self.var_zero_line.get():
                    ax.axhline(0.0, lw=0.8, alpha=0.6, color="0.6")
                self._style_axes(ax)
        else:
            ax = self.fig.add_subplot(111)
            # vertical shift
            max_dy = 0.0
            for (x, y) in spectra:
                mask = np.ones_like(x, dtype=bool)
                try:
                    norm_xmin = float(self.xmin.get()) if self.xmin.get() else None
                    norm_xmax = float(self.xmax.get()) if self.xmax.get() else None
                except Exception:
                    norm_xmin = norm_xmax = None
                if norm_xmin is not None:
                    mask &= (x >= norm_xmin)
                if norm_xmax is not None:
                    mask &= (x <= norm_xmax)
                yv = y[mask] if np.any(mask) else y
                dy = (yv.max() - yv.min()) if len(yv) else 1.0
                max_dy = max(max_dy, dy * 0.5)
            delta = shift_factor * max_dy

            for i, (x, y) in enumerate(spectra):
                y_to_plot = y + i * delta
                color = self._color_from_scheme(i, labels[i], len(spectra))
                ln = (ax.plot(x, y_to_plot, lw=1, label=labels[i])[0] if color is None
                      else ax.plot(x, y_to_plot, color=color, lw=1, label=labels[i])[0])
                self._lines.append(ln)

            if len(spectra) > 1:
                ax.legend(fontsize=9)
            ax.set_xlabel(self.x_title.get())
            ax.set_ylabel(self.y_title.get())
            if self.var_hide_yticks.get():
                ax.tick_params(left=False, labelleft=False)
            else:
                ax.tick_params(left=True, labelleft=True)
            if self.var_zero_line.get():
                ax.axhline(0.0, lw=0.8, alpha=0.6, color="0.6")
            if not self.var_no_title.get():
                ax.set_title(global_title if global_title else labels[-1])
            self._style_axes(ax)

        if self.var_anno_on.get():
            self.var_anno_on.set(False)
            self._toggle_annotate()

        self._draw_cif_bragg_markers()
        self.canvas.draw_idle()
        self.fig.subplots_adjust(left=0.10, right=0.98, bottom=0.20, top=0.90, wspace=0.08, hspace=0.12)
        self.update_plot_axes(apply_only=True)

    def _on_mouse_move(self, event):
        if event.inaxes is None:
            self.status.set("x: —   y: —")
            return
        ax = event.inaxes
        candidates = [ln for ln in self._lines if ln.axes is ax]
        if not candidates:
            self.status.set("x: —   y: —")
            return
        x0 = event.xdata
        best, best_dx = None, float("inf")
        for ln in candidates:
            xd, yd = ln.get_xdata(), ln.get_ydata()
            if len(xd) == 0:
                continue
            idx = np.searchsorted(xd, x0)
            idx = max(0, min(len(xd) - 1, idx))
            if idx > 0 and abs(xd[idx - 1] - x0) < abs(xd[idx] - x0):
                idx -= 1
            dx = abs(xd[idx] - x0)
            if dx < best_dx:
                best_dx = dx
                best = (xd[idx], yd[idx])
        if best is None:
            self.status.set("x: —   y: —")
        else:
            xv, yv = best
            self.status.set(f"x: {xv:.3f}   y: {yv:.3f}")

    def update_plot_axes(self, apply_only=False):
        xmin = self._num(self.xmin.get())
        xmax = self._num(self.xmax.get())
        ymin = self._num(self.ymin.get())
        ymax = self._num(self.ymax.get())
        for ax in self.fig.get_axes():
            try:
                if xmin is not None or xmax is not None:
                    lo, hi = ax.get_xlim()
                    ax.set_xlim(xmin if xmin is not None else lo, xmax if xmax is not None else hi)
                if ymin is not None or ymax is not None:
                    lo, hi = ax.get_ylim()
                    ax.set_ylim(ymin if ymin is not None else lo, ymax if ymax is not None else hi)
            except Exception:
                pass
            if not apply_only:
                ax.set_xlabel(self.x_title.get())
                ax.set_ylabel(self.y_title.get())
        self.canvas.draw_idle()
        self.canvas.get_tk_widget().update_idletasks()

    def autoscale_y(self):
        if not self.current_data:
            return
        try:
            xmin = float(self.xmin.get()) if self.xmin.get() else None
            xmax = float(self.xmax.get()) if self.xmax.get() else None
        except Exception:
            xmin = xmax = None
        mode = self.plot_mode.get()
        if mode == "separate":
            axes = self.fig.get_axes()
            for i, (x, y) in enumerate(self.current_data):
                ax = axes[i] if i < len(axes) else axes[0]
                mask = np.ones_like(x, dtype=bool)
                if xmin is not None:
                    mask &= (x >= xmin)
                if xmax is not None:
                    mask &= (x <= xmax)
                yv = y[mask] if np.any(mask) else y
                if len(yv) == 0:
                    continue
                ax.set_ylim(yv.min(), yv.max() * 1.1)
            self.ymin.set("")
            self.ymax.set("")
            self.canvas.draw()
        else:
            all_yv = []
            for (x, y) in self.current_data:
                mask = np.ones_like(x, dtype=bool)
                if xmin is not None:
                    mask &= (x >= xmin)
                if xmax is not None:
                    mask &= (x <= xmax)
                yv = y[mask] if np.any(mask) else y
                if len(yv):
                    all_yv.append(yv)
            if not all_yv:
                messagebox.showinfo("Auto scale", "No data in selected X range.", parent=self)
                return
            y_min = min(y.min() for y in all_yv)
            y_max = max(y.max() for y in all_yv)
            for ax in self.fig.get_axes():
                ax.set_ylim(y_min, y_max * 1.1)
            self.ymin.set("")
            self.ymax.set("")
            self.canvas.draw()

    def destroy(self):
        if hasattr(self, "fig"):
            plt.close(self.fig)
        super().destroy()
