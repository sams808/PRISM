# plot_dta.py
"""
DTA / DSC / TGA text export reader + interactive Tg tool.

This script is designed for TA Instruments-style text exports (SDT/Q-series-like),
but it is intentionally tolerant to format variations:

- Robust text decoding (UTF-16/UTF-8/Latin-1 fallbacks)
- Header end detection using "StartOfData" OR first numeric row fallback
- Column name extraction from "Sig1 ... / Sig2 ..." header lines, else fallback names

GUI features (Tkinter + Matplotlib):
- Select X and Y columns (the selected Y is used for plotting AND for all tangent-based Tg calculations)
- Optional "dY source" column used ONLY for derivative-based Tg (max|dY/dX|) and derivative overlay
- Matplotlib toolbar (pan/zoom/home/save)
- Stable view while tuning Tg parameters; auto-rescale when you change X or Y
- Minimal derivative smoothing (MA10) ON/OFF (applies ONLY to derivative arrays)

Tg methods implemented:
1) Double tangent: LOW baseline tangent ∩ Tg-slope tangent
   - LOW baseline can be AUTO or manual x-range
   - Slope tangent can be AUTO (tangent at max|dY/dX|) or manual x-range (linear fit)
2) Parallel tangents: force low/high baselines parallel; Tg = (midline between baselines) ∩ curve
   - LOW and HIGH baselines can be AUTO or manual x-ranges (shared LOW range with double tangent)
   - Reference x used to choose the correct crossing is from derivative peak (AUTO) or within manual slope-range
3) Derivative Tg: Tg = argmax_x |dY/dX| computed from "dY source" (default = same as Y)

Batch processing:
- Run the three Tg methods for many files using the CURRENT GUI settings
- Optional PNG snapshots per file for QC before trusting the CSV

Dependencies: numpy, pandas, matplotlib, tkinter
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

def _apply_modern_style(widget):
    palette = {
        "bg": "#f4f4f4",
        "card": "#f4f4f4",
        "card_alt": "#f4f4f4",
        "accent": "#e0e0e0",      # light grey for buttons
        "accent_alt": "#e0e0e0",
        "accent_warn": "#e0e0e0",
        "muted": "#657080",
        "success": "#e0e0e0",
    }
    style = None
    try:
        from tkinter import ttk as _ttk
        style = _ttk.Style(widget)
        style.theme_use("clam")
        style.configure(".", background=palette["bg"], foreground="#1c2733", fieldbackground=palette["card"])
        style.configure("Card.TFrame", background=palette["card"], borderwidth=1, relief="flat")
        style.configure("CardAlt.TFrame", background=palette["card_alt"], borderwidth=1, relief="flat")
        style.configure("Card.TLabelframe", background=palette["card"], relief="flat", borderwidth=1)
        style.configure("Card.TLabelframe.Label", background=palette["card"], foreground="#1c2733", font=("Segoe UI", 11, "underline"))
        style.configure("Section.TLabel", background=palette["bg"], foreground="#1c2733", font=("Segoe UI", 13, "bold", "underline"))
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
        _btn("Alt.TButton", palette["accent"])
        _btn("Warn.TButton", palette["accent"])
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
        style.configure("TNotebook", background=palette["bg"], borderwidth=0)
        style.configure("TNotebook.Tab", background=palette["card"], foreground="#1c2733", padding=(10, 6))
        style.map("TNotebook.Tab", background=[("selected", palette["card_alt"])])
    except Exception:
        pass

    try:
        widget.configure(bg=palette["bg"])
    except Exception:
        pass
    return palette


# =============================================================================
# 1) File I/O (robust TA-style export parsing)
# =============================================================================

def _pick_file_gui() -> Optional[Path]:
    """Open a native file dialog and return a selected file path."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        p = filedialog.askopenfilename(
            title="Select a TA export text file",
            filetypes=[("Text files", "*.txt *.dat *.asc *.csv"), ("All files", "*.*")]
        )
        return Path(p) if p else None
    except Exception:
        return None


def read_text_autodetect(path: Path) -> str:
    """Read bytes and decode with common encodings seen in TA exports."""
    b = path.read_bytes()

    # UTF-16 BOM
    if b.startswith(b"\xff\xfe") or b.startswith(b"\xfe\xff"):
        return b.decode("utf-16", errors="replace")

    # Heuristic: many NUL bytes suggests UTF-16
    head = b[:4000]
    if head and head.count(b"\x00") > len(head) * 0.1:
        try:
            return b.decode("utf-16", errors="replace")
        except Exception:
            pass

    # UTF-8 (with BOM support)
    try:
        return b.decode("utf-8-sig", errors="replace")
    except Exception:
        return b.decode("latin-1", errors="replace")


def parse_ta_sdt_txt(path: Path) -> Tuple[Dict[str, str], List[str], pd.DataFrame]:
    """
    Parse TA Instruments SDT/Q-series style export.
    - Header ends at 'StartOfData' (case-insensitive) OR fallback to first numeric row.
    - Column names often appear as "SigN  <Name>" in header.

    Returns: (header_dict, colnames, df)
    """
    text = read_text_autodetect(path)
    lines = text.splitlines()

    start_idx = None
    for i, line in enumerate(lines):
        norm = re.sub(r"\s+", "", line).lower()
        if "startofdata" in norm:
            start_idx = i
            break

    # Fallback: first numeric line
    if start_idx is None:
        for i, line in enumerate(lines):
            s = line.strip()
            if not s:
                continue
            if s[0].isdigit() or s[0] in "+-.":
                start_idx = i - 1
                break

    if start_idx is None:
        raise ValueError("Could not find data start (no StartOfData and no numeric rows).")

    header_lines = lines[:start_idx]
    data_lines = lines[start_idx + 1:]

    header: Dict[str, str] = {}
    for line in header_lines:
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            key = parts[0].strip()
            val = "\t".join(parts[1:]).strip()
            header.setdefault(key, val)

    # Extract SigN names
    sig_map: Dict[int, str] = {}
    sig_re = re.compile(r"^Sig(\d+)\s+(.*)$")
    for line in header_lines:
        m = sig_re.match(line.strip())
        if m:
            idx = int(m.group(1))
            name = m.group(2).strip()
            sig_map[idx] = name

    if sig_map:
        colnames = [sig_map[i] for i in sorted(sig_map.keys())]
    else:
        # Guess from first numeric row
        first_data = next((ln for ln in data_lines if ln.strip()), "")
        ncols = len(re.split(r"\s+", first_data.strip()))
        colnames = [f"col{i+1}" for i in range(ncols)]

    numeric_rows = []
    for ln in data_lines:
        s = ln.strip()
        if not s:
            continue
        if s[0].isdigit() or s[0] in "+-.":
            numeric_rows.append(s)

    if not numeric_rows:
        raise ValueError("No numeric data rows found after header.")

    df = pd.read_csv(
        StringIO("\n".join(numeric_rows)),
        sep=r"\s+",
        engine="python",
        names=colnames,
    )

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(how="all").reset_index(drop=True)
    return header, colnames, df


# =============================================================================
# 2-4) Smoothing, derivatives, and Tg methods now live in dta_science.py
# (framework-agnostic, pytest-covered) — this file only wires the GUI to it.
# =============================================================================

from dta_science import (
    moving_average_10,
    compute_derivative,
    _fit_line,
    _line_y,
    _intersect_lines,
    _root_on_grid,
    TransitionInfo,
    _transition_from_derivative,
    _peak_index_in_range,
    TgDoubleTangentResult,
    compute_tg_double_tangent,
    TgParallelTangentResult,
    compute_tg_parallel_improved,
    compute_tg_derivative,
    BaselineParams,
    resolve_baseline_params,
)


# =============================================================================
# 5) GUI (Tkinter + Matplotlib)
# =============================================================================

def _make_scrollable_frame_class(tk, ttk, *, background="#f8f4ea"):
    """
    ttk.Frame that provides a vertical scrollbar via a Canvas.
    Intended for dense left-panel controls.
    """
    class ScrollableFrame(ttk.Frame):
        def __init__(self, master, **kwargs):
            super().__init__(master, **kwargs)

            self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0, background=background)
            self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
            self.canvas.configure(yscrollcommand=self.vsb.set)

            self.vsb.pack(side="right", fill="y")
            self.canvas.pack(side="left", fill="both", expand=True)

            self.inner = ttk.Frame(self.canvas)
            self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

            def _on_inner_configure(_evt=None):
                self.canvas.configure(scrollregion=self.canvas.bbox("all"))

            self.inner.bind("<Configure>", _on_inner_configure)

            def _on_canvas_configure(evt):
                self.canvas.itemconfigure(self._win, width=evt.width)

            self.canvas.bind("<Configure>", _on_canvas_configure)

            # Mousewheel scrolling
            def _on_mousewheel(event):
                if getattr(event, "delta", 0):
                    step = int(-1 * (event.delta / 120))
                    self.canvas.yview_scroll(step, "units")

            def _on_button4(_event):
                self.canvas.yview_scroll(-1, "units")

            def _on_button5(_event):
                self.canvas.yview_scroll(1, "units")

            def _bind(_evt=None):
                self.canvas.bind_all("<MouseWheel>", _on_mousewheel)
                self.canvas.bind_all("<Button-4>", _on_button4)
                self.canvas.bind_all("<Button-5>", _on_button5)

            def _unbind(_evt=None):
                self.canvas.unbind_all("<MouseWheel>")
                self.canvas.unbind_all("<Button-4>")
                self.canvas.unbind_all("<Button-5>")

            self.canvas.bind("<Enter>", _bind)
            self.canvas.bind("<Leave>", _unbind)

    return ScrollableFrame


class TgGuiApp:
    def __init__(self, root, path: Optional[Path] = None, records: Optional[List[Dict[str, Any]]] = None):
        import tkinter as tk
        from tkinter import ttk

        self.root = root
        root.title("DTA Tg Tool")
        root.geometry("1280x800")
        try:
            root.minsize(1100, 700)
        except Exception:
            pass

        self.palette = _apply_modern_style(root)

        ScrollableFrame = _make_scrollable_frame_class(tk, ttk, background=self.palette["bg"])

        # Imported records (from main app)
        self.records: List[Dict[str, Any]] = records or []
        self._current_record: Optional[Dict[str, Any]] = None
        self._record_lookup: Dict[str, Dict[str, Any]] = {}
        for i, rec in enumerate(self.records):
            key = rec.get("title") or f"Record {i+1}"
            base = key
            suffix = 2
            while key in self._record_lookup:
                key = f"{base} ({suffix})"
                suffix += 1
            self._record_lookup[key] = rec

        self.path: Optional[Path] = None
        self.header: Dict[str, str] = {}
        self.colnames: List[str] = []
        self.df: Optional[pd.DataFrame] = None

        # Cached arrays
        self._x: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None

        # Results
        self.res_double: Optional[TgDoubleTangentResult] = None
        self.res_parallel: Optional[TgParallelTangentResult] = None
        self.tg_deriv: Optional[float] = None

        # Plot view locking: stable view when tweaking Tg parameters
        self._view_initialized = False
        self._lock_view = True
        self._last_x_col = None
        self._last_y_col = None

        self._overlay_artists: List[Any] = []
        self._calc_artists: List[Any] = []

        outer = ttk.Frame(root, padding=10, style="CardAlt.TFrame")
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=0)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(0, weight=1)

        self.left = ttk.Frame(outer, width=450, style="Card.TFrame", padding=6)
        self.left.grid(row=0, column=0, sticky="nsw", padx=(0, 10))
        self.right = ttk.Frame(outer, style="CardAlt.TFrame")
        self.right.grid(row=0, column=1, sticky="nsew")

        self.nb = ttk.Notebook(self.left)
        self.nb.pack(fill="both", expand=True)

        self.tab_tg = ttk.Frame(self.nb)
        self.tab_calc = ttk.Frame(self.nb)
        self.tab_batch = ttk.Frame(self.nb)
        self.nb.add(self.tab_tg, text="Tg")
        self.nb.add(self.tab_calc, text="Calculs")
        self.nb.add(self.tab_batch, text="Export / Batch")

        # Scrollable tab bodies
        self.tg_scroll = ScrollableFrame(self.tab_tg)
        self.tg_scroll.pack(fill="both", expand=True)
        self.tab_tg_body = self.tg_scroll.inner

        self.calc_scroll = ScrollableFrame(self.tab_calc)
        self.calc_scroll.pack(fill="both", expand=True)
        self.tab_calc_body = self.calc_scroll.inner

        self.batch_scroll = ScrollableFrame(self.tab_batch)
        self.batch_scroll.pack(fill="both", expand=True)
        self.tab_batch_body = self.batch_scroll.inner

        self._build_tg_tab()
        self._build_calc_tab()
        self._build_batch_tab()
        self._build_mpl()

        if path is not None:
            self.load_file(path)
        elif self.records:
            first_title = next(iter(self._record_lookup.keys()))
            try:
                self.record_var.set(first_title)
                self._load_selected_record()
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # UI construction
    # -------------------------------------------------------------------------

    
    def _build_tg_tab(self):
        import tkinter as tk
        from tkinter import ttk

        parent = self.tab_tg_body

        # --- Data ---
        data_box = ttk.LabelFrame(parent, text="Data", padding=8, style="Card.TLabelframe")
        data_box.pack(fill="x", pady=(0, 6))

        self.file_label = ttk.Label(data_box, text="No file loaded", wraplength=430, style="Card.TLabel")
        self.file_label.pack(fill="x", pady=(0, 4))

        # Imported record selector (if provided by host app)
        self.record_var = tk.StringVar(value="")
        record_row = ttk.Frame(data_box, style="Card.TFrame")
        record_row.pack(fill="x", pady=(0, 4))
        ttk.Label(record_row, text="Imported", style="Card.TLabel").pack(side="left")
        self.record_combo = ttk.Combobox(
            record_row,
            textvariable=self.record_var,
            values=list(self._record_lookup.keys()),
            state="readonly",
            width=28
        )
        self.record_combo.pack(side="left", padx=(6, 6), fill="x", expand=True)
        self.record_combo.bind("<<ComboboxSelected>>", lambda _e=None: self._load_selected_record())
        ttk.Button(record_row, text="Load", command=self._load_selected_record, style="Primary.TButton").pack(side="left")

        btns = ttk.Frame(data_box, style="Card.TFrame")
        btns.pack(fill="x")
        ttk.Button(btns, text="Open…", command=self._on_open, style="Alt.TButton").pack(side="left")
        ttk.Button(btns, text="Reload", command=self._on_reload, style="Ghost.TButton").pack(side="left", padx=(6, 0))

        # --- Plot selections ---
        plot_box = ttk.LabelFrame(parent, text="Plot", padding=8, style="Card.TLabelframe")
        plot_box.pack(fill="x", pady=(0, 6))
        plot_box.columnconfigure(1, weight=1)

        self.x_var = tk.StringVar(value="")
        self.y_var = tk.StringVar(value="")
        self.dy_src_var = tk.StringVar(value="(same)")  # derivative source, default same as Y

        ttk.Label(plot_box, text="X").grid(row=0, column=0, sticky="w")
        ttk.Label(plot_box, text="Y").grid(row=1, column=0, sticky="w")
        ttk.Label(plot_box, text="dY source").grid(row=2, column=0, sticky="w")

        self.x_combo = ttk.Combobox(plot_box, textvariable=self.x_var, state="readonly", width=30)
        self.y_combo = ttk.Combobox(plot_box, textvariable=self.y_var, state="readonly", width=30)
        self.dy_combo = ttk.Combobox(plot_box, textvariable=self.dy_src_var, state="readonly", width=30)

        self.x_combo.grid(row=0, column=1, sticky="ew", padx=(6, 0))
        self.y_combo.grid(row=1, column=1, sticky="ew", padx=(6, 0), pady=(2, 0))
        self.dy_combo.grid(row=2, column=1, sticky="ew", padx=(6, 0), pady=(2, 0))

        def _on_x_change(_evt=None):
            self._view_initialized = False
            self._refresh_plot()

        def _on_y_change(_evt=None):
            self._view_initialized = False
            # keep calc tab Y in sync (if present)
            try:
                if hasattr(self, "calc_y_var") and self.calc_y_var.get() in ("", None):
                    self.calc_y_var.set(self.y_var.get())
            except Exception:
                pass
            self._refresh_plot()

        def _on_dy_change(_evt=None):
            self._refresh_plot()

        self.x_combo.bind("<<ComboboxSelected>>", _on_x_change)
        self.y_combo.bind("<<ComboboxSelected>>", _on_y_change)
        self.dy_combo.bind("<<ComboboxSelected>>", _on_dy_change)

        opt = ttk.Frame(plot_box)
        opt.grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self.invert_y_var = tk.BooleanVar(value=False)
        self.show_deriv_var = tk.BooleanVar(value=False)
        self.explicit_var = tk.BooleanVar(value=True)

        ttk.Checkbutton(opt, text="Invert Y", variable=self.invert_y_var, command=self._refresh_plot).pack(side="left")
        ttk.Checkbutton(opt, text="dY overlay", variable=self.show_deriv_var, command=self._refresh_plot).pack(side="left", padx=(8, 0))
        ttk.Checkbutton(opt, text="Explicit", variable=self.explicit_var, command=self._refresh_plot).pack(side="left", padx=(8, 0))

        sm = ttk.Frame(plot_box)
        sm.grid(row=4, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Label(sm, text="Smooth dY (MA10)").pack(side="left")
        self.smooth_var = tk.StringVar(value="off")
        ttk.Radiobutton(sm, text="Off", value="off", variable=self.smooth_var, command=self._refresh_plot).pack(side="left", padx=(8, 0))
        ttk.Radiobutton(sm, text="On", value="on", variable=self.smooth_var, command=self._refresh_plot).pack(side="left", padx=(6, 0))

        # --- Tg setup ---
        tg_box = ttk.LabelFrame(parent, text="Tg", padding=8, style="Card.TLabelframe")
        tg_box.pack(fill="x", pady=(0, 6))
        tg_box.columnconfigure(1, weight=1)

        self.method_var = tk.StringVar(value="parallel")  # double / parallel / deriv
        mrow = ttk.Frame(tg_box)
        mrow.grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Radiobutton(mrow, text="Double", value="double", variable=self.method_var, command=self._refresh_plot).pack(side="left")
        ttk.Radiobutton(mrow, text="Parallel (improved)", value="parallel", variable=self.method_var, command=self._refresh_plot).pack(side="left", padx=(8, 0))
        ttk.Radiobutton(mrow, text="|dY| max", value="deriv", variable=self.method_var, command=self._refresh_plot).pack(side="left", padx=(8, 0))

        # Window (only user-controlled numeric parameter here)
        self.xmin_var = tk.StringVar(value="350")
        self.xmax_var = tk.StringVar(value="700")

        wrow = ttk.Frame(tg_box)
        wrow.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Label(wrow, text="Window").pack(side="left")
        ttk.Entry(wrow, textvariable=self.xmin_var, width=9).pack(side="left", padx=(6, 4))
        ttk.Label(wrow, text="→").pack(side="left")
        ttk.Entry(wrow, textvariable=self.xmax_var, width=9).pack(side="left", padx=(4, 0))

        # Manual baselines (shared)
        ranges = ttk.LabelFrame(tg_box, text="Manual ranges (shared)", padding=8, style="Card.TLabelframe")
        ranges.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ranges.columnconfigure(1, weight=1)

        # LOW controls
        self.low_min_var = tk.StringVar(value="")
        self.low_max_var = tk.StringVar(value="")
        self.low_use_point_var = tk.BooleanVar(value=False)
        self.low_point_var = tk.StringVar(value="")

        # HIGH controls
        self.high_min_var = tk.StringVar(value="")
        self.high_max_var = tk.StringVar(value="")
        self.high_use_point_var = tk.BooleanVar(value=False)
        self.high_point_var = tk.StringVar(value="")

        # Manual slope range for Double Tangent method
        self.slope_min_var = tk.StringVar(value="")
        self.slope_max_var = tk.StringVar(value="")

        def _toggle_low_point():
            usep = bool(self.low_use_point_var.get())
            state_rng = "disabled" if usep else "normal"
            state_pt = "normal" if usep else "disabled"
            self.low_min_entry.configure(state=state_rng)
            self.low_max_entry.configure(state=state_rng)
            self.low_point_entry.configure(state=state_pt)
            self._refresh_plot()

        def _toggle_high_point():
            usep = bool(self.high_use_point_var.get())
            state_rng = "disabled" if usep else "normal"
            state_pt = "normal" if usep else "disabled"
            self.high_min_entry.configure(state=state_rng)
            self.high_max_entry.configure(state=state_rng)
            self.high_point_entry.configure(state=state_pt)
            self._refresh_plot()

        # Row builder
        def _baseline_row(row: int, label: str,
                          vmin: tk.StringVar, vmax: tk.StringVar,
                          use_point: tk.BooleanVar, vpt: tk.StringVar,
                          toggle_cmd):
            fr = ttk.Frame(ranges)
            fr.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(2 if row else 0, 0))
            ttk.Label(fr, text=f"{label}:").pack(side="left")

            e1 = ttk.Entry(fr, textvariable=vmin, width=9)
            e2 = ttk.Entry(fr, textvariable=vmax, width=9)
            e1.pack(side="left", padx=(6, 4))
            ttk.Label(fr, text="→").pack(side="left")
            e2.pack(side="left", padx=(4, 8))

            cb = ttk.Checkbutton(fr, text="Use point", variable=use_point, command=toggle_cmd)
            cb.pack(side="left")

            ttk.Label(fr, text="Point:").pack(side="left", padx=(8, 4))
            ep = ttk.Entry(fr, textvariable=vpt, width=9)
            ep.pack(side="left")

            return e1, e2, ep

        self.low_min_entry, self.low_max_entry, self.low_point_entry = _baseline_row(
            0, "LOW", self.low_min_var, self.low_max_var, self.low_use_point_var, self.low_point_var, _toggle_low_point
        )
        self.high_min_entry, self.high_max_entry, self.high_point_entry = _baseline_row(
            2, "HIGH", self.high_min_var, self.high_max_var, self.high_use_point_var, self.high_point_var, _toggle_high_point
        )

        # SLOPE range (used by Double Tangent method only)
        slope_fr = ttk.Frame(ranges, style="Card.TFrame")
        slope_fr.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Label(slope_fr, text="SLOPE:").pack(side="left")
        self.slope_min_entry = ttk.Entry(slope_fr, textvariable=self.slope_min_var, width=9)
        self.slope_max_entry = ttk.Entry(slope_fr, textvariable=self.slope_max_var, width=9)
        self.slope_min_entry.pack(side="left", padx=(6, 4))
        ttk.Label(slope_fr, text="→").pack(side="left")
        self.slope_max_entry.pack(side="left", padx=(4, 0))
        ttk.Label(slope_fr, text="(for Double Tangent only)").pack(side="left", padx=(8, 0))

        # init states
        _toggle_low_point()
        _toggle_high_point()

        # Action buttons
        brow = ttk.Frame(tg_box, style="Card.TFrame")
        brow.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        # Auto/Manual compute toggle
        self.manual_compute_var = tk.BooleanVar(value=False)

        def _on_manual_toggle():
            # Manual flag only affects which parameters are used; button label stays constant.
            return

        ttk.Checkbutton(brow, text="Manual", variable=self.manual_compute_var, command=_on_manual_toggle).pack(side="left", padx=(0, 8))
        self.compute_btn = ttk.Button(brow, text="Compute", command=self._compute, style="Primary.TButton")
        self.compute_btn.pack(side="left", fill="x", expand=True)
        ttk.Button(brow, text="Clear", command=self._clear_results, style="Warn.TButton").pack(side="left", padx=(6, 0), fill="x", expand=True)
        ttk.Button(brow, text="Redraw", command=self._refresh_plot, style="Ghost.TButton").pack(side="left", padx=(6, 0), fill="x", expand=True)

        self.result_label = ttk.Label(tg_box, text="Tg: —", font=("Segoe UI", 10, "bold"), wraplength=430, justify="left", style="Card.TLabel")
        self.result_label.grid(row=4, column=0, columnspan=2, sticky="w", pady=(8, 0))

        self.status = ttk.Label(parent, text="", wraplength=430, justify="left", style="Muted.TLabel")
        self.status.pack(fill="x", pady=(0, 0))

        self._set_status(
            "Parallel (improved): each baseline is either a range (linear fit) or a point "
            "(a line parallel to the other baseline passing through that point). "
            "Thr/Guard controls were removed on purpose."
        )

    def _build_calc_tab(self):
        import tkinter as tk
        from tkinter import ttk

        parent = self.tab_calc_body

        box = ttk.LabelFrame(parent, text="Calculs", padding=8, style="Card.TLabelframe")
        box.pack(fill="x", pady=(0, 6))
        box.columnconfigure(1, weight=1)

        self.calc_y_var = tk.StringVar(value="")
        self.calc_xmin_var = tk.StringVar(value="")
        self.calc_xmax_var = tk.StringVar(value="")
        self.calc_result_var = tk.StringVar(value="")

        ttk.Label(box, text="Y").grid(row=0, column=0, sticky="w")
        self.calc_y_combo = ttk.Combobox(box, textvariable=self.calc_y_var, state="readonly", width=28)
        self.calc_y_combo.grid(row=0, column=1, sticky="ew", padx=(6, 0))

        # Derivative option for Calculs
        self.calc_use_deriv_var = tk.BooleanVar(value=False)
        self.calc_deriv_x_var = tk.StringVar(value="")

        drow = ttk.Frame(box)
        drow.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Checkbutton(drow, text="Use derivative", variable=self.calc_use_deriv_var, command=self._calc_on_deriv_toggle).pack(side="left")
        ttk.Label(drow, text="d/dX:").pack(side="left", padx=(10, 4))
        self.calc_deriv_x_combo = ttk.Combobox(drow, textvariable=self.calc_deriv_x_var, state="readonly", width=22)
        self.calc_deriv_x_combo.pack(side="left", fill="x", expand=True)

        row = ttk.Frame(box)
        row.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Label(row, text="Xmin").pack(side="left")
        ttk.Entry(row, textvariable=self.calc_xmin_var, width=10).pack(side="left", padx=(6, 10))
        ttk.Label(row, text="Xmax").pack(side="left")
        ttk.Entry(row, textvariable=self.calc_xmax_var, width=10).pack(side="left", padx=(6, 0))

        btnrow = ttk.Frame(box, style="Card.TFrame")
        btnrow.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        ttk.Button(btnrow, text="Integrate", command=self._calc_integrate, style="Success.TButton").pack(side="left", fill="x", expand=True)
        ttk.Button(btnrow, text="Find Max", command=self._calc_find_max, style="Primary.TButton").pack(side="left", padx=(6, 0), fill="x", expand=True)
        ttk.Button(btnrow, text="Find Min", command=self._calc_find_min, style="Primary.TButton").pack(side="left", padx=(6, 0), fill="x", expand=True)
        ttk.Button(btnrow, text="Clear", command=self._calc_clear, style="Warn.TButton").pack(side="left", padx=(6, 0), fill="x", expand=True)

        res = ttk.Label(box, textvariable=self.calc_result_var, wraplength=430, justify="left")
        res.grid(row=4, column=0, columnspan=2, sticky="w", pady=(10, 0))

        self._set_status("Calculs: choose Y, set Xmin/Xmax, then integrate or find extrema.")


    def _build_batch_tab(self):
        import tkinter as tk
        from tkinter import ttk

        parent = self.tab_batch_body

        export_box = ttk.LabelFrame(parent, text="Export current", padding=8, style="Card.TLabelframe")
        export_box.pack(fill="x", pady=(0, 6))

        ttk.Button(export_box, text="Export result CSV…", command=self._export_current_csv, style="Primary.TButton").pack(fill="x", pady=(0, 4))
        ttk.Button(export_box, text="Save figure PNG…", command=self._save_figure_png, style="Alt.TButton").pack(fill="x")

        batch_box = ttk.LabelFrame(parent, text="Batch", padding=8, style="Card.TLabelframe")
        batch_box.pack(fill="x")

        ttk.Label(
            batch_box,
            text="Batch uses CURRENT GUI settings (window, manual ranges, thr/guard, smoothing).",
            wraplength=430,
            justify="left"
        ).pack(fill="x", pady=(0, 6))

        self.batch_snapshot_var = tk.StringVar(value="no")  # no/png
        ttk.Label(batch_box, text="Snapshots:").pack(anchor="w")
        r = ttk.Frame(batch_box)
        r.pack(anchor="w")
        ttk.Radiobutton(r, text="No", value="no", variable=self.batch_snapshot_var).pack(side="left")
        ttk.Radiobutton(r, text="PNG", value="png", variable=self.batch_snapshot_var).pack(side="left", padx=(8, 0))

        ttk.Button(batch_box, text="Run batch…", command=self._batch_run).pack(fill="x", pady=(6, 0))

    def _build_mpl(self):
        """Matplotlib toolbar (pan/zoom/home/save) + plot."""
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        from matplotlib.figure import Figure
        from tkinter import ttk

        toolbar_frame = ttk.Frame(self.right, style="Card.TFrame")
        toolbar_frame.pack(side="top", fill="x")

        plot_frame = ttk.Frame(self.right, style="CardAlt.TFrame")
        plot_frame.pack(side="top", fill="both", expand=True)

        self.fig = Figure(figsize=(7.8, 5.6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax2 = None

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

    # -------------------------------------------------------------------------
    # UI actions / helpers
    # -------------------------------------------------------------------------

    def _load_selected_record(self):
        import tkinter.messagebox as mb

        title = (self.record_var.get() or "").strip()
        if not title or title not in self._record_lookup:
            return
        rec = self._record_lookup[title]
        self._current_record = rec

        df = rec.get("df")
        meta = rec.get("meta") or {}
        path = rec.get("path")

        if df is None and path:
            try:
                hdr, cols, df = parse_ta_sdt_txt(Path(path))
                meta = {**meta, "raw_header": hdr, "signals": cols}
            except Exception as e:
                mb.showerror("Load error", f"Could not reload {path}:\n{e}")
                return

        if df is None:
            mb.showerror("Load error", "Selected record has no data attached.")
            return

        display_title = rec.get("title") or (Path(path).stem if path else "DTA data")
        self._apply_dataframe(df, meta=meta, path=path, display_title=display_title)

    def _suggest_default_columns(self, meta: Optional[Dict[str, Any]]) -> Tuple[str, str]:
        canonical = (meta or {}).get("canonical_map") or {}

        def pick_from_canon(keys: List[str]) -> Optional[str]:
            for key in keys:
                col = canonical.get(key)
                if col in self.colnames:
                    return col
            return None

        def pick_by_keyword(keywords: List[str]) -> Optional[str]:
            for kw in keywords:
                for c in self.colnames:
                    if kw.lower() in str(c).lower():
                        return c
            return None

        xdef = pick_from_canon(["T_C", "time_min", "X"])
        ydef = pick_from_canon(["DSC_mW_mg", "HF_mW", "TG_pct", "mass_mg", "Y"])

        if xdef is None:
            xdef = pick_by_keyword(["temperature", "temp"]) or (self.colnames[0] if self.colnames else "")
        if ydef is None:
            ydef = pick_by_keyword(["heat flow", "dsc", "dta", "heat", "signal", "tg"]) or (
                self.colnames[1] if len(self.colnames) > 1 else (self.colnames[0] if self.colnames else "")
            )

        return xdef or "", ydef or ""

    def _apply_dataframe(self, df: pd.DataFrame, *, meta: Optional[Dict[str, Any]], path: Optional[str], display_title: str):
        self.path = Path(path) if path else None
        self.header = (meta or {}).get("raw_header", {}) or {}
        self.colnames = list(df.columns)
        self.df = df.copy()

        # Populate combobox values
        self.x_combo["values"] = self.colnames
        self.y_combo["values"] = self.colnames
        self.dy_combo["values"] = ["(same)"] + self.colnames
        if hasattr(self, "calc_y_combo"):
            self.calc_y_combo["values"] = self.colnames
        if hasattr(self, "calc_deriv_x_combo"):
            self.calc_deriv_x_combo["values"] = self.colnames

        xdef, ydef = self._suggest_default_columns(meta)

        self.x_var.set(xdef)
        self.y_var.set(ydef)
        self.dy_src_var.set("(same)")
        if hasattr(self, "calc_y_var"):
            self.calc_y_var.set(ydef)
        if hasattr(self, "calc_deriv_x_var"):
            self.calc_deriv_x_var.set(xdef)

        # Initialize Tg window defaults
        if xdef and xdef in self.df.columns:
            xvals = pd.to_numeric(self.df[xdef], errors="coerce").to_numpy()
            xvals = xvals[np.isfinite(xvals)]
            if len(xvals) > 0:
                self.xmin_var.set("350")
                self.xmax_var.set("700")
                if hasattr(self, "calc_xmin_var"):
                    self.calc_xmin_var.set("350")
                    self.calc_xmax_var.set("700")

        sample = self.header.get("Sample", self.path.stem if self.path else display_title)
        suffix = f"{self.path.name}" if self.path else display_title
        self.file_label.configure(text=f"{suffix}\nSample: {sample}")

        self._clear_results()
        self._view_initialized = False
        self._refresh_plot()

    def _on_open(self):
        p = _pick_file_gui()
        if p:
            self.load_file(p)

    def _on_reload(self):
        if self._current_record is not None:
            self._load_selected_record()
        elif self.path:
            self.load_file(self.path)

    def load_file(self, path: Path):
        self._current_record = None
        self.path = Path(path)
        self.header, self.colnames, self.df = parse_ta_sdt_txt(self.path)

        sample = self.header.get("Sample", self.path.stem)
        self.file_label.configure(text=f"{self.path.name}\nSample: {sample}")

        # Populate combobox values
        self.x_combo["values"] = self.colnames
        self.y_combo["values"] = self.colnames
        self.dy_combo["values"] = ["(same)"] + self.colnames
        if hasattr(self, "calc_y_combo"):
            self.calc_y_combo["values"] = self.colnames
        if hasattr(self, "calc_deriv_x_combo"):
            self.calc_deriv_x_combo["values"] = self.colnames

        def pick_default(keywords: List[str]) -> Optional[str]:
            for kw in keywords:
                for c in self.colnames:
                    if kw.lower() in c.lower():
                        return c
            return None

        xdef = pick_default(["temperature", "temp"]) or (self.colnames[0] if self.colnames else "")
        ydef = pick_default(["heat flow", "dsc", "dta", "heat", "signal"]) or (self.colnames[1] if len(self.colnames) > 1 else "")

        self._apply_dataframe(self.df, meta={"raw_header": self.header}, path=str(self.path), display_title=self.path.name)

    def _unit_from_col(self, col: str) -> str:
        cl = col.lower()
        if "°c" in cl or "temperature" in cl or "temp" in cl or "c)" in cl:
            return "°C"
        if "kelvin" in cl or ("temp" in cl and "k" in cl):
            return "K"
        return ""

    def _smooth_derivative_enabled(self) -> bool:
        return self.smooth_var.get() == "on"

    def _get_window(self) -> Tuple[float, float]:
        a = float(self.xmin_var.get())
        b = float(self.xmax_var.get())
        return (min(a, b), max(a, b))

    def _get_manual_range(self, enabled: bool, vmin, vmax) -> Optional[Tuple[float, float]]:
        if not enabled:
            return None
        try:
            a = float(vmin.get())
            b = float(vmax.get())
            return tuple(sorted((a, b)))
        except Exception:
            return None

    def _get_xy(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, str, str]:
        """
        Return arrays for:
          x, y (for plot+tangents), y_dy (source for derivative overlay and derivative Tg)
        """
        if self.df is None:
            raise ValueError("No data loaded.")

        x_col = self.x_var.get()
        y_col = self.y_var.get()
        dy_sel = self.dy_src_var.get()

        if not x_col or not y_col:
            raise ValueError("Pick X and Y columns first.")

        x = pd.to_numeric(self.df[x_col], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(self.df[y_col], errors="coerce").to_numpy(dtype=float)

        if dy_sel and dy_sel != "(same)" and dy_sel in self.df.columns:
            y_dy = pd.to_numeric(self.df[dy_sel], errors="coerce").to_numpy(dtype=float)
            dy_col = dy_sel
        else:
            y_dy = y.copy()
            dy_col = "(same)"

        m = np.isfinite(x) & np.isfinite(y) & np.isfinite(y_dy)
        x = x[m]
        y = y[m]
        y_dy = y_dy[m]

        order = np.argsort(x)
        x = x[order]
        y = y[order]
        y_dy = y_dy[order]

        if self.invert_y_var.get():
            y = -y
            y_dy = -y_dy

        return x, y, y_dy, x_col, y_col, dy_col

    def _clear_overlay(self):
        for a in self._overlay_artists:
            try:
                a.remove()
            except Exception:
                pass
        self._overlay_artists = []

    
    def _clear_calc_overlay(self):
        # Remove text/lines/patches created by Calculs tools without changing view.
        for a in [getattr(self, '_calc_alt_line', None), getattr(self, '_calc_text', None)]:
            if a is not None:
                try:
                    a.remove()
                except Exception:
                    pass
        self._calc_alt_line = None
        self._calc_text = None
        # Remove artists created by Calculs tools (primary axis)
        for a in getattr(self, "_calc_artists", []):
            try:
                a.remove()
            except Exception:
                pass
        self._calc_artists = []

        # Remove derivative-axis artists if present
        for a in getattr(self, "_calc_artists2", []):
            try:
                a.remove()
            except Exception:
                pass
        self._calc_artists2 = []

        # Hide secondary axis if we created one for Calculs
        ax2 = getattr(self, "_calc_ax2", None)
        if ax2 is not None:
            try:
                ax2.set_visible(False)
            except Exception:
                pass

    def _calc_get_axis(self, x=None, y=None, label=""):
        """Return the axis used for Calculs drawing, with smart secondary-axis behavior.

        Rules:
        - If Calculs Y matches Tg Y (or derivative of it), use the main axis (or a standard right axis for derivative).
        - If Calculs Y differs from Tg Y, use an inside-right red axis and also plot that series there
          so it remains readable (separate scale).
        """
        use_deriv = bool(getattr(self, "calc_use_deriv_var", None) and self.calc_use_deriv_var.get())
        tg_y = (getattr(self, "y_var", None).get() if getattr(self, "y_var", None) else "") or ""
        calc_y = (getattr(self, "calc_y_var", None).get() if getattr(self, "calc_y_var", None) else "") or ""

        is_alt = bool(calc_y and tg_y and (calc_y != tg_y))

        # No derivative + same Y => primary axis
        if (not use_deriv) and (not is_alt):
            return self.ax

        # Create or reuse a secondary axis sharing the same x
        ax2 = getattr(self, "_calc_ax2", None)
        if ax2 is None:
            ax2 = self.ax.twinx()
            self._calc_ax2 = ax2

        # Configure the axis style
        try:
            ax2.set_visible(True)
        except Exception:
            pass

        if is_alt:
            # Keep the right spine slightly offset so ticks/labels stay outside the data
            try:
                offset = 1.10 if getattr(self, "ax2", None) else 1.04
                ax2.spines["right"].set_position(("axes", offset))
            except Exception:
                pass
            try:
                ax2.spines["right"].set_color("red")
                ax2.yaxis.label.set_color("red")
                ax2.tick_params(axis="y", colors="red", labelright=True, pad=8, direction="out")
            except Exception:
                pass
            try:
                ax2.set_ylabel(label or calc_y or "Calculs")
            except Exception:
                pass
            try:
                # place label just outside the plot on the right
                ax2.yaxis.set_label_coords(1.08, 0.5)
            except Exception:
                pass

            # Plot the selected Calculs series on that axis (full trace) for readability
            if (x is not None) and (y is not None):
                try:
                    # remove previous line if any
                    if getattr(self, "_calc_alt_line", None) is not None:
                        try:
                            self._calc_alt_line.remove()
                        except Exception:
                            pass
                        self._calc_alt_line = None
                    self._calc_alt_line, = ax2.plot(x, y, linewidth=1.0)
                    try:
                        self._calc_alt_line.set_color("red")
                    except Exception:
                        pass

                    # smart vertical scaling based on the visible (or calc) x-window to avoid "flat line" look
                    try:
                        try:
                            rx0, rx1 = self._get_calc_range()
                            m = (x >= min(rx0, rx1)) & (x <= max(rx0, rx1))
                        except Exception:
                            x0, x1 = self.ax.get_xlim()
                            m = (x >= min(x0, x1)) & (x <= max(x0, x1))
                        yy = y[m] if (m is not None and np.any(m)) else y
                        yy = yy[np.isfinite(yy)]
                        if yy.size >= 2:
                            ymin = float(np.nanmin(yy)); ymax = float(np.nanmax(yy))
                            pad = 0.05 * (ymax - ymin) if ymax > ymin else (abs(ymin) * 0.05 + 1e-9)
                            ax2.set_ylim(ymin - pad, ymax + pad)
                    except Exception:
                        pass
                except Exception:
                    pass

            return ax2

        # Derivative of same Y => standard secondary axis (right side)
        try:
            ax2.spines["right"].set_position(("axes", 1.0))
        except Exception:
            pass
        try:
            ax2.spines["right"].set_color("black")
            ax2.yaxis.label.set_color("black")
            ax2.tick_params(axis="y", colors="black", labelright=True, pad=6)
        except Exception:
            pass
        try:
            ax2.set_ylabel(label or "Derivative")
        except Exception:
            pass
        try:
            if (x is not None) and (y is not None):
                try:
                    try:
                        rx0, rx1 = self._get_calc_range()
                        m = (x >= min(rx0, rx1)) & (x <= max(rx0, rx1))
                    except Exception:
                        x0, x1 = self.ax.get_xlim()
                        m = (x >= min(x0, x1)) & (x <= max(x0, x1))
                    yy = y[m] if (m is not None and np.any(m)) else y
                    yy = yy[np.isfinite(yy)]
                    if yy.size >= 2:
                        ymin = float(np.nanmin(yy)); ymax = float(np.nanmax(yy))
                        pad = 0.05 * (ymax - ymin) if ymax > ymin else (abs(ymin) * 0.05 + 1e-9)
                        ax2.set_ylim(ymin - pad, ymax + pad)
                except Exception:
                    pass
        except Exception:
            pass
        return ax2


    def _calc_clear(self):
        """Clear Calculs graphics overlays without changing current axes limits."""
        try:
            self._clear_calc_overlay()
            # keep current view (x/y limits) as-is
            self.canvas.draw_idle()
            self.calc_result_var.set("")
            self._set_status("Calculs cleared (view kept).")
        except Exception:
            pass


    
    def _calc_on_deriv_toggle(self):
        """When derivative is toggled, refresh Calculs overlays and keep view."""
        try:
            # set default derivative X to current X if empty
            if hasattr(self, "calc_deriv_x_var") and not self.calc_deriv_x_var.get():
                self.calc_deriv_x_var.set(self.x_var.get())
            self._calc_clear()
        except Exception:
            pass

    def _get_calc_range(self) -> Tuple[float, float]:
        try:
            a = float(self.calc_xmin_var.get())
            b = float(self.calc_xmax_var.get())
            return (min(a, b), max(a, b))
        except Exception:
            raise ValueError("Provide numeric Xmin / Xmax for Calculs.")

    def _get_calc_y(self) -> Tuple[np.ndarray, np.ndarray, str]:
        if self.df is None:
            raise ValueError("No data loaded.")
        y_col = self.calc_y_var.get() or self.y_var.get()
        if not y_col:
            raise ValueError("Select a Y for Calculs.")

        use_deriv = bool(getattr(self, "calc_use_deriv_var", None) and self.calc_use_deriv_var.get())
        x_col = self.x_var.get()
        if use_deriv:
            x_col = self.calc_deriv_x_var.get() or self.x_var.get()

        x = pd.to_numeric(self.df[x_col], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(self.df[y_col], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        x = x[m]; y = y[m]
        order = np.argsort(x)
        x = x[order]; y = y[order]

        if self.invert_y_var.get():
            y = -y

        if use_deriv:
            # robust numerical derivative dy/dx
            if x.size < 3:
                raise ValueError("Not enough points for derivative.")
            # avoid repeated x (np.gradient hates zeros)
            dx = np.diff(x)
            if np.any(dx == 0):
                # keep unique x by small jitter-free aggregation: take first occurrence
                _, idx = np.unique(x, return_index=True)
                idx = np.sort(idx)
                x = x[idx]; y = y[idx]
            dydx = np.gradient(y, x)
            return x, dydx, f"d({y_col})/d({x_col})"

        return x, y, y_col


    def _calc_integrate(self):
        import tkinter.messagebox as mb
        try:
            x, y, y_col = self._get_calc_y()
            xmin, xmax = self._get_calc_range()
            w = (x >= xmin) & (x <= xmax)
            if w.sum() < 2:
                raise ValueError("Range too small for integration.")
            area = float(np.trapz(y[w], x[w]))
            self.calc_result_var.set(f"Integrate ({y_col}) from {xmin:.4g} to {xmax:.4g} = {area:.6g}")

            # highlight on graph
            self._clear_calc_overlay()
            ax = self._calc_get_axis(x=x, y=y, label=y_col)
            # always show selected x-range on primary axis
            self._calc_artists.append(self.ax.axvspan(xmin, xmax, alpha=0.08))
            artists_list = self._calc_artists2 if ax is not self.ax else self._calc_artists
            artists_list.append(ax.fill_between(x[w], y[w], 0.0, alpha=0.12))
            # annotate value bottom-left
            try:
                if getattr(self, '_calc_text', None) is not None:
                    self._calc_text.remove()
            except Exception:
                pass
            try:
                self._calc_text = self.ax.text(0.01, 0.01, f"Intg = {area:.6g}", transform=self.ax.transAxes, va="bottom", ha="left")
                self._calc_artists.append(self._calc_text)
            except Exception:
                pass
            self.canvas.draw_idle()
        except Exception as e:
            mb.showerror("Integrate failed", str(e))

    def _calc_find_max(self):
        import tkinter.messagebox as mb
        try:
            x, y, y_col = self._get_calc_y()
            xmin, xmax = self._get_calc_range()
            w = (x >= xmin) & (x <= xmax)
            if w.sum() < 2:
                raise ValueError("Range too small.")
            j = int(np.nanargmax(y[w]))
            xw = x[w]; yw = y[w]
            xv = float(xw[j]); yv = float(yw[j])
            self.calc_result_var.set(f"Max ({y_col}) in [{xmin:.4g}, {xmax:.4g}] = {yv:.6g} at x={xv:.6g}")

            self._clear_calc_overlay()
            ax = self._calc_get_axis(x=x, y=y, label=y_col)
            self._calc_artists.append(self.ax.axvspan(xmin, xmax, alpha=0.06))
            artists_list = self._calc_artists2 if ax is not self.ax else self._calc_artists
            artists_list.append(ax.plot([xv], [yv], marker="o")[0])
            artists_list.append(ax.axvline(xv, linestyle="--", linewidth=1.5))

            # annotate value bottom-left
            try:
                if getattr(self, '_calc_text', None) is not None:
                    self._calc_text.remove()
            except Exception:
                pass
            try:
                self._calc_text = self.ax.text(0.01, 0.01, f"Max = {yv:.6g} @ x={xv:.6g}", transform=self.ax.transAxes, va="bottom", ha="left")
                self._calc_artists.append(self._calc_text)
            except Exception:
                pass
            self.canvas.draw_idle()
        except Exception as e:
            mb.showerror("Find Max failed", str(e))

    def _calc_find_min(self):
        import tkinter.messagebox as mb
        try:
            x, y, y_col = self._get_calc_y()
            xmin, xmax = self._get_calc_range()
            w = (x >= xmin) & (x <= xmax)
            if w.sum() < 2:
                raise ValueError("Range too small.")
            j = int(np.nanargmin(y[w]))
            xw = x[w]; yw = y[w]
            xv = float(xw[j]); yv = float(yw[j])
            self.calc_result_var.set(f"Min ({y_col}) in [{xmin:.4g}, {xmax:.4g}] = {yv:.6g} at x={xv:.6g}")

            self._clear_calc_overlay()
            ax = self._calc_get_axis(x=x, y=y, label=y_col)
            self._calc_artists.append(self.ax.axvspan(xmin, xmax, alpha=0.06))
            artists_list = self._calc_artists2 if ax is not self.ax else self._calc_artists
            artists_list.append(ax.plot([xv], [yv], marker="o")[0])
            artists_list.append(ax.axvline(xv, linestyle="--", linewidth=1.5))

            # annotate value bottom-left
            try:
                if getattr(self, '_calc_text', None) is not None:
                    self._calc_text.remove()
            except Exception:
                pass
            try:
                self._calc_text = self.ax.text(0.01, 0.01, f"Min = {yv:.6g} @ x={xv:.6g}", transform=self.ax.transAxes, va="bottom", ha="left")
                self._calc_artists.append(self._calc_text)
            except Exception:
                pass
            self.canvas.draw_idle()
        except Exception as e:
            mb.showerror("Find Min failed", str(e))

    def _refresh_plot(self):
        if self.df is None:
            return

        # View lock logic:
        # - If X or Y changed: reset autoscale
        # - If only parameters changed: keep current view stable
        prev_xlim = prev_ylim = None
        try:
            x_col = self.x_var.get()
            y_col = self.y_var.get()
        except Exception:
            x_col = y_col = None

        columns_changed = (x_col != self._last_x_col) or (y_col != self._last_y_col)
        if columns_changed:
            self._view_initialized = False
            self._last_x_col = x_col
            self._last_y_col = y_col

        if self._lock_view and self._view_initialized:
            prev_xlim = self.ax.get_xlim()
            prev_ylim = self.ax.get_ylim()

        # Clear
        if self.ax2 is not None:
            try:
                self.ax2.remove()
            except Exception:
                pass
            self.ax2 = None

        self.ax.clear()
        self._clear_overlay()
        self._clear_calc_overlay()

        # Plot data
        x, y, y_dy, x_col, y_col, dy_col = self._get_xy()
        self._x, self._y = x, y

        self.ax.plot(x, y, linewidth=2.0) #, label=y_col
        self.ax.set_xlabel(x_col)
        self.ax.set_ylabel(y_col + (" (inv)" if self.invert_y_var.get() else ""))
        sample = self.header.get("Sample", self.path.stem if self.path else "")
        self.ax.set_title(sample)
        self.ax.grid(True, alpha=0.25)
        #self.ax.legend(loc="best", fontsize=9)

        # Tg window shading
        try:
            xmin, xmax = self._get_window()
            self.ax.axvspan(xmin, xmax, alpha=0.04)
        except Exception:
            pass

        # Initialize view (autoscale) once after column changes / load
        if self._lock_view and not self._view_initialized:
            self.fig.canvas.draw_idle()
            self._view_initialized = True
            prev_xlim = self.ax.get_xlim()
            prev_ylim = self.ax.get_ylim()

        # Derivative overlay (from y_dy)
        if self.show_deriv_var.get():
            try:
                xmin, xmax = self._get_window()
                w = (x >= xmin) & (x <= xmax)
                if w.sum() >= 20:
                    dy = compute_derivative(y_dy[w], x[w])
                    if self._smooth_derivative_enabled():
                        dy = moving_average_10(dy)
                    self.ax2 = self.ax.twinx()
                    self.ax2.plot(x[w], dy, linewidth=1.2, alpha=0.85)
                    self.ax2.set_ylabel(f"dY/d{x_col}")
                    self.ax2.grid(False)
            except Exception:
                pass

        # Constructions
        self._draw_overlays(y_dy=y_dy, dy_col=dy_col)

        # Restore view if needed (keep stable when tweaking params)
        if self._lock_view and prev_xlim is not None and prev_ylim is not None and not columns_changed:
            self.ax.set_xlim(prev_xlim)
            self.ax.set_ylim(prev_ylim)
            self.ax.set_autoscale_on(False)
        else:
            self.ax.set_autoscale_on(True)

        self.canvas.draw_idle()

    def _draw_overlays(self, y_dy: np.ndarray, dy_col: str):
        """Draw the selected method construction + a compact results box."""
        if self._x is None or self._y is None:
            return

        x = self._x
        y = self._y
        explicit = bool(self.explicit_var.get())
        unit = self._unit_from_col(self.x_var.get())
        method = self.method_var.get()

        # Values summary (always shown)
        td = self.res_double.tg if self.res_double is not None else float("nan")
        tp = self.res_parallel.tg if self.res_parallel is not None else float("nan")
        tx = self.tg_deriv if self.tg_deriv is not None else float("nan")

        info_lines = [
            f"Double: {td:.2f}{unit}" if np.isfinite(td) else "Double: —",
            f"Parallel: {tp:.2f}{unit}" if np.isfinite(tp) else "Parallel: —",
            f"|dY| max: {tx:.2f}{unit}" if np.isfinite(tx) else "|dY| max: —",
        ]
        if dy_col != "(same)":
            info_lines.append(f"dY src: {dy_col}")

        def draw_tg_marker(tg: float, label: str):
            if not np.isfinite(tg):
                return
            self._overlay_artists.append(self.ax.axvline(tg, linestyle="--", linewidth=2.0))
            yy = float(np.interp(tg, x, y))
            self._overlay_artists.append(self.ax.plot([tg], [yy], marker="o")[0])
            ytop = self.ax.get_ylim()[1]
            #self._overlay_artists.append(self.ax.text(
            #    tg, ytop,
            #    f" {label}={tg:.2f}{unit}",
            #    va="top", ha="left", fontsize=9,
            #    bbox=dict(boxstyle="round,pad=0.2", alpha=0.12),
            #))

        # Draw method-specific constructions
        try:
            xmin, xmax = self._get_window()
        except Exception:
            xmin, xmax = float(x[0]), float(x[-1])
        xx = np.array([xmin, xmax], dtype=float)

        if method == "double" and self.res_double is not None:
            r = self.res_double
            if explicit:
                self._overlay_artists.append(self.ax.plot(xx, _line_y(r.m_low, r.b_low, xx), linestyle="--", linewidth=1.3)[0])
                self._overlay_artists.append(self.ax.plot(xx, _line_y(r.m_slope, r.b_slope, xx), linestyle="-.", linewidth=1.3)[0])
                # Show auto bounds lightly
                self._overlay_artists.append(self.ax.axvline(r.x_left, alpha=0.15, linestyle=":"))
                self._overlay_artists.append(self.ax.axvline(r.x_right, alpha=0.15, linestyle=":"))
            draw_tg_marker(r.tg, "Tg")

        elif method == "parallel" and self.res_parallel is not None:
            r = self.res_parallel
            if explicit:
                self._overlay_artists.append(self.ax.plot(xx, _line_y(r.m_par, r.b_low, xx), linestyle="--", linewidth=1.2)[0])
                self._overlay_artists.append(self.ax.plot(xx, _line_y(r.m_par, r.b_high, xx), linestyle="--", linewidth=1.2)[0])
                self._overlay_artists.append(self.ax.plot(xx, _line_y(r.m_par, r.b_mid, xx), linestyle="-", linewidth=1.3)[0])
                self._overlay_artists.append(self.ax.axvline(r.x_left, alpha=0.15, linestyle=":"))
                self._overlay_artists.append(self.ax.axvline(r.x_right, alpha=0.15, linestyle=":"))
            draw_tg_marker(r.tg, "Tg")

        elif method == "deriv" and np.isfinite(tx):
            draw_tg_marker(tx, "Tg")

        # Results box
        self._overlay_artists.append(self.ax.text(
            0.02, 0.98, "\n".join(info_lines),
            transform=self.ax.transAxes,
            va="top", ha="left",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.35", alpha=0.15),
        ))

    
    def _raw_baseline_inputs(self) -> Dict[str, Any]:
        """Read raw widget values UNCONDITIONALLY (regardless of the Manual
        checkbox) for feeding into dta_science.resolve_baseline_params(),
        which is the only place that decides whether they actually get used.
        """
        def _f(var):
            try:
                return float(var.get())
            except Exception:
                return None

        return dict(
            low_use_point=bool(getattr(self, "low_use_point_var", False).get()),
            low_point_x=_f(self.low_point_var) if hasattr(self, "low_point_var") else None,
            low_min=_f(self.low_min_var) if hasattr(self, "low_min_var") else None,
            low_max=_f(self.low_max_var) if hasattr(self, "low_max_var") else None,
            high_use_point=bool(getattr(self, "high_use_point_var", False).get()),
            high_point_x=_f(self.high_point_var) if hasattr(self, "high_point_var") else None,
            high_min=_f(self.high_min_var) if hasattr(self, "high_min_var") else None,
            high_max=_f(self.high_max_var) if hasattr(self, "high_max_var") else None,
            slope_min=_f(self.slope_min_var) if hasattr(self, "slope_min_var") else None,
            slope_max=_f(self.slope_max_var) if hasattr(self, "slope_max_var") else None,
        )

    def _resolved_baseline_params(self) -> BaselineParams:
        """The ONE path both interactive Compute and batch processing use to
        turn 'Manual' checkbox state + typed fields into actual values —
        calling this identically from both is what makes it structurally
        impossible for batch to silently apply leftover manual values when
        Manual is unchecked (the bug this replaces).
        """
        manual_enabled = bool(getattr(self, "manual_compute_var", None) and self.manual_compute_var.get())
        return resolve_baseline_params(manual_enabled=manual_enabled, **self._raw_baseline_inputs())

    def _compute(self):
        import tkinter.messagebox as mb
        try:
            x, y, y_dy, x_col, y_col, dy_col = self._get_xy()
            xmin, xmax = self._get_window()
            smooth_d = self._smooth_derivative_enabled()

            bp = self._resolved_baseline_params()
            low_range, low_point = bp.low_range, bp.low_point
            high_range, high_point = bp.high_range, bp.high_point
            manual_slope = bp.manual_slope

            # Derivative Tg uses y_dy
            self.tg_deriv = compute_tg_derivative(
                x, y_dy, xmin, xmax,
                smooth_derivative=smooth_d,
                restrict_range=None,
            )

            # Double tangent: keep as before (AUTO low baseline + slope at derivative peak),
            # but allow LOW range if provided (point mode ignored for double).
            try:
                self.res_double = compute_tg_double_tangent(
                    x, y, xmin, xmax,
                    threshold=0.20, guard_frac=0.0,
                    smooth_derivative=smooth_d,
                    manual_low=low_range,
                    manual_slope=manual_slope,
                )
            except Exception:
                self.res_double = None

            # Parallel improved
            try:
                self.res_parallel = compute_tg_parallel_improved(
                    x, y, xmin, xmax,
                    smooth_derivative=smooth_d,
                    manual_low_range=low_range,
                    manual_high_range=high_range,
                    manual_low_point=low_point,
                    manual_high_point=high_point,
                )
            except Exception:
                self.res_parallel = None

            unit = self._unit_from_col(x_col)
            td = self.res_double.tg if self.res_double is not None else float("nan")
            tp = self.res_parallel.tg if self.res_parallel is not None else float("nan")
            tx = self.tg_deriv if self.tg_deriv is not None else float("nan")

            lines = [
                f"Y: {y_col}",
                f"Double: {td:.2f} {unit}" if np.isfinite(td) else "Double: —",
                f"Parallel: {tp:.2f} {unit}" if np.isfinite(tp) else "Parallel: —",
                f"|dY| max: {tx:.2f} {unit}" if np.isfinite(tx) else "|dY| max: —",
            ]
            if dy_col != "(same)":
                lines.append(f"dY src: {dy_col}")

            # Describe the baseline mode ACTUALLY used by the parallel-tangent
            # result (res_parallel.low_mode/high_mode), not the raw checkbox
            # state — the point+point case silently falls back to AUTO ranges
            # internally, and the two must not be reported inconsistently.
            if self.res_parallel is not None:
                lp = self.res_parallel.low_used
                if self.res_parallel.low_mode == "point":
                    lines.append(f"LOW: point x={lp[0]:.6g}")
                else:
                    lines.append(f"LOW: {lp[0]:.6g}..{lp[1]:.6g}")
                hp = self.res_parallel.high_used
                if self.res_parallel.high_mode == "point":
                    lines.append(f"HIGH: point x={hp[0]:.6g}")
                else:
                    lines.append(f"HIGH: {hp[0]:.6g}..{hp[1]:.6g}")

            self.result_label.configure(text="\n".join(lines))
            self._refresh_plot()

        except Exception as e:
            mb.showerror("Tg computation failed", str(e))

    def _clear_results(self):
        self.res_double = None
        self.res_parallel = None
        self.tg_deriv = None
        self.result_label.configure(text="Tg: —")
        self._set_status("Cleared results.")
        if self.df is not None:
            self._refresh_plot()

    def _set_status(self, msg: str):
        self.status.configure(text=msg)

    # -------------------------------------------------------------------------
    # Export / Save / Batch
    # -------------------------------------------------------------------------

    def _export_current_csv(self):
        from tkinter import filedialog, messagebox
        if self.df is None:
            messagebox.showerror("Export", "No data loaded.")
            return

        out_csv = filedialog.asksaveasfilename(
            title="Save result as CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")]
        )
        if not out_csv:
            return

        row = self._current_result_row()
        pd.DataFrame([row]).to_csv(out_csv, index=False)
        messagebox.showinfo("Export", f"Saved:\n{out_csv}")

    def _save_figure_png(self):
        from tkinter import filedialog, messagebox
        if self.df is None:
            messagebox.showerror("Save figure", "No data loaded.")
            return

        out_png = filedialog.asksaveasfilename(
            title="Save figure as PNG",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("All files", "*.*")]
        )
        if not out_png:
            return

        self.fig.savefig(out_png, dpi=200, bbox_inches="tight")
        messagebox.showinfo("Save figure", f"Saved:\n{out_png}")

    
    def _current_result_row(self) -> Dict[str, Any]:
        x, y, y_dy, x_col, y_col, dy_col = self._get_xy()
        xmin, xmax = self._get_window()
        smooth_d = self._smooth_derivative_enabled()

        bp = self._resolved_baseline_params()
        low_range, low_point = bp.low_range, bp.low_point
        high_range, high_point = bp.high_range, bp.high_point

        def _fmt_rng(rng):
            if rng is None:
                return ""
            return f"{rng[0]:.6g}..{rng[1]:.6g}"

        sample = self.header.get("Sample", self.path.stem if self.path else "")
        td = self.res_double.tg if self.res_double is not None else np.nan
        tp = self.res_parallel.tg if self.res_parallel is not None else np.nan
        tx = self.tg_deriv if self.tg_deriv is not None else np.nan

        # Report the mode ACTUALLY used by the last computed parallel-tangent
        # result (accounts for the point+point -> AUTO fallback), not the raw
        # checkbox/typed-field state, which can diverge from it.
        if self.res_parallel is not None:
            low_mode = self.res_parallel.low_mode
            high_mode = self.res_parallel.high_mode
        else:
            low_mode = "point" if low_point is not None else ("range" if low_range is not None else "auto")
            high_mode = "point" if high_point is not None else ("range" if high_range is not None else "auto")

        return dict(
            file=str(self.path) if self.path else "",
            sample=sample,
            x_col=x_col,
            y_col=y_col,
            dy_source=(y_col if dy_col == "(same)" else dy_col),
            window_min=xmin,
            window_max=xmax,
            smooth_derivative=int(smooth_d),
            low_mode=low_mode,
            low_range=_fmt_rng(low_range),
            low_point_x=("" if low_point is None else float(low_point)),
            high_mode=high_mode,
            high_range=_fmt_rng(high_range),
            high_point_x=("" if high_point is None else float(high_point)),
            Tg_double=td,
            Tg_parallel=tp,
            Tg_derivative=tx,
        )

    
    def _batch_run(self):
        import tkinter.messagebox as mb
        from tkinter import filedialog

        files = filedialog.askopenfilenames(
            title="Select files for batch Tg",
            filetypes=[("Text files", "*.txt *.dat *.asc *.csv"), ("All files", "*.*")]
        )
        if not files:
            return

        out_csv = filedialog.asksaveasfilename(
            title="Save batch results as CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")]
        )
        if not out_csv:
            return

        make_snaps = (self.batch_snapshot_var.get() == "png")
        snap_dir = None
        if make_snaps:
            outp = Path(out_csv)
            snap_dir = outp.with_suffix("").as_posix() + "_snapshots"
            Path(snap_dir).mkdir(parents=True, exist_ok=True)

        # Freeze current GUI settings
        x_name = self.x_var.get()
        y_name = self.y_var.get()
        dy_sel = self.dy_src_var.get()

        xmin, xmax = self._get_window()
        smooth_d = self._smooth_derivative_enabled()
        invert = bool(self.invert_y_var.get())

        # Same resolver interactive Compute uses — if "Manual" is unchecked,
        # this discards any leftover typed values instead of batch silently
        # applying them (the bug this replaces).
        bp = self._resolved_baseline_params()
        low_range, low_point = bp.low_range, bp.low_point
        high_range, high_point = bp.high_range, bp.high_point
        manual_slope = bp.manual_slope

        default_low_mode = "point" if low_point is not None else ("range" if low_range is not None else "auto")
        default_high_mode = "point" if high_point is not None else ("range" if high_range is not None else "auto")

        rows: List[Dict[str, Any]] = []

        for f in files:
            p = Path(f)
            try:
                header, cols, df = parse_ta_sdt_txt(p)

                def _find(name: str) -> Optional[str]:
                    for c in df.columns:
                        if c.lower() == name.lower():
                            return c
                    return None

                x_col = x_name if x_name in df.columns else (_find(x_name) or x_name)
                y_col = y_name if y_name in df.columns else (_find(y_name) or y_name)

                if x_col not in df.columns or y_col not in df.columns:
                    raise ValueError(f"Missing columns: {x_name} or {y_name}")

                x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
                y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)

                if dy_sel and dy_sel != "(same)" and dy_sel in df.columns:
                    y_dy = pd.to_numeric(df[dy_sel], errors="coerce").to_numpy(dtype=float)
                    dy_col = dy_sel
                else:
                    y_dy = y.copy()
                    dy_col = "(same)"

                m = np.isfinite(x) & np.isfinite(y) & np.isfinite(y_dy)
                x = x[m]; y = y[m]; y_dy = y_dy[m]
                order = np.argsort(x)
                x = x[order]; y = y[order]; y_dy = y_dy[order]
                if invert:
                    y = -y; y_dy = -y_dy

                tg_der = compute_tg_derivative(
                    x, y_dy, xmin, xmax,
                    smooth_derivative=smooth_d,
                    restrict_range=None,
                )

                try:
                    rd = compute_tg_double_tangent(
                        x, y, xmin, xmax,
                        threshold=0.20, guard_frac=0.0,
                        smooth_derivative=smooth_d,
                        manual_low=low_range,
                        manual_slope=manual_slope,
                    )
                    tg_double = rd.tg
                except Exception:
                    rd = None
                    tg_double = np.nan

                try:
                    rp = compute_tg_parallel_improved(
                        x, y, xmin, xmax,
                        smooth_derivative=smooth_d,
                        manual_low_range=low_range,
                        manual_high_range=high_range,
                        manual_low_point=low_point,
                        manual_high_point=high_point,
                    )
                    tg_par = rp.tg
                except Exception:
                    rp = None
                    tg_par = np.nan

                sample = header.get("Sample", p.stem)

                def _fmt_rng(rng):
                    if rng is None:
                        return ""
                    return f"{rng[0]:.6g}..{rng[1]:.6g}"

                rows.append(dict(
                    file=str(p),
                    sample=sample,
                    x_col=x_col,
                    y_col=y_col,
                    dy_source=(y_col if dy_col == "(same)" else dy_col),
                    window_min=xmin,
                    window_max=xmax,
                    smooth_derivative=int(smooth_d),
                    # Report the mode ACTUALLY used by this file's parallel-
                    # tangent result (rp), not the requested/typed mode — the
                    # point+point case silently falls back to AUTO ranges.
                    low_mode=(rp.low_mode if rp is not None else default_low_mode),
                    low_range=_fmt_rng(low_range),
                    low_point_x=("" if low_point is None else float(low_point)),
                    high_mode=(rp.high_mode if rp is not None else default_high_mode),
                    high_range=_fmt_rng(high_range),
                    high_point_x=("" if high_point is None else float(high_point)),
                    Tg_double=tg_double,
                    Tg_parallel=tg_par,
                    Tg_derivative=tg_der,
                ))

                if make_snaps and snap_dir is not None:
                    out_png = Path(snap_dir) / f"{p.stem}_tg.png"
                    self._save_snapshot_png(
                        out_png=out_png,
                        header=header,
                        x=x, y=y, y_dy=y_dy,
                        x_label=x_col, y_label=y_col,
                        tg_double=rd, tg_parallel=rp, tg_derivative=tg_der,
                        xmin=xmin, xmax=xmax,
                        show_derivative=True,
                        smooth_derivative=smooth_d,
                        dy_label=(y_col if dy_col == "(same)" else dy_col),
                    )

            except Exception as e:
                rows.append(dict(file=str(p), sample="", error=str(e)))

        pd.DataFrame(rows).to_csv(out_csv, index=False)
        msg = f"Saved batch CSV:\n{out_csv}"
        if make_snaps and snap_dir is not None:
            msg += f"\n\nSnapshots folder:\n{snap_dir}"
        mb.showinfo("Batch complete", msg)


    def _save_snapshot_png(
        self,
        out_png: Path,
        header: Dict[str, str],
        x: np.ndarray,
        y: np.ndarray,
        y_dy: np.ndarray,
        x_label: str,
        y_label: str,
        tg_double: Optional[TgDoubleTangentResult],
        tg_parallel: Optional[TgParallelTangentResult],
        tg_derivative: float,
        xmin: float,
        xmax: float,
        show_derivative: bool,
        smooth_derivative: bool,
        dy_label: str,
    ):
        """Create a clean snapshot figure for batch QC."""
        fig = plt.figure(figsize=(9.0, 6.0), dpi=140)
        ax = fig.add_subplot(111)
        ax.plot(x, y, linewidth=2.0, label=y_label)
        ax.axvspan(xmin, xmax, alpha=0.04)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=9)

        ax2 = None
        if show_derivative:
            w = (x >= xmin) & (x <= xmax)
            if w.sum() >= 20:
                dy = compute_derivative(y_dy[w], x[w])
                if smooth_derivative:
                    dy = moving_average_10(dy)
                ax2 = ax.twinx()
                ax2.plot(x[w], dy, linewidth=1.2, alpha=0.85)
                ax2.set_ylabel(f"d({dy_label})/d({x_label})")

        unit = self._unit_from_col(x_label)

        def marker(tg: float, lab: str):
            if not np.isfinite(tg):
                return
            ax.axvline(tg, linestyle="--", linewidth=2.0)
            yy = float(np.interp(tg, x, y))
            ax.plot([tg], [yy], marker="o")
            ytop = ax.get_ylim()[1]
            ax.text(tg, ytop, f" {lab}={tg:.2f}{unit}", va="top", ha="left",
                    fontsize=9, bbox=dict(boxstyle="round,pad=0.2", alpha=0.12))

        if tg_double is not None:
            if True:  # explicit always for snapshots
                xx = np.array([xmin, xmax], dtype=float)
                ax.plot(xx, _line_y(tg_double.m_low, tg_double.b_low, xx), linestyle="--", linewidth=1.3)
                ax.plot(xx, _line_y(tg_double.m_slope, tg_double.b_slope, xx), linestyle="-.", linewidth=1.3)
            marker(tg_double.tg, "Tg")

        if tg_parallel is not None:
            xx = np.array([xmin, xmax], dtype=float)
            ax.plot(xx, _line_y(tg_parallel.m_par, tg_parallel.b_low, xx), linestyle="--", linewidth=1.2)
            ax.plot(xx, _line_y(tg_parallel.m_par, tg_parallel.b_high, xx), linestyle="--", linewidth=1.2)
            ax.plot(xx, _line_y(tg_parallel.m_par, tg_parallel.b_mid, xx), linestyle="-", linewidth=1.3)
            marker(tg_parallel.tg, "Tg")

        marker(tg_derivative, "Tg|dY|")

        sample = header.get("Sample", out_png.stem)
        fig.suptitle(sample)

        info_lines = [
            f"Double: {tg_double.tg:.2f}{unit}" if tg_double is not None and np.isfinite(tg_double.tg) else "Double: —",
            f"Parallel: {tg_parallel.tg:.2f}{unit}" if tg_parallel is not None and np.isfinite(tg_parallel.tg) else "Parallel: —",
            f"|dY| max: {tg_derivative:.2f}{unit}" if np.isfinite(tg_derivative) else "|dY| max: —",
        ]
        ax.text(0.02, 0.98, "\n".join(info_lines), transform=ax.transAxes,
                va="top", ha="left", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.35", alpha=0.15))

        fig.tight_layout()
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)


class DtaProcessingWindow(TgGuiApp):
    """
    Convenience alias for embedding the Tg GUI inside the main application.
    Expects a Tk master (can be a Toplevel) and a list of imported DTA records.
    """

    def __init__(self, master, records: List[Dict[str, Any]]):
        super().__init__(master, path=None, records=records)


# =============================================================================
# 6) CLI plot
# =============================================================================

def choose_column(colnames: List[str], prompt: str) -> str:
    print("\nAvailable columns:")
    for i, name in enumerate(colnames):
        print(f"  [{i}] {name}")
    while True:
        raw = input(prompt).strip()
        if raw.isdigit():
            idx = int(raw)
            if 0 <= idx < len(colnames):
                return colnames[idx]
        matches = [c for c in colnames if raw.lower() in c.lower()]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            print(f"Ambiguous: matches {matches}. Please enter an index.")
        else:
            print("Invalid choice. Enter an index (e.g. 0) or a unique substring.")


def find_best_column(colnames: List[str], keyword: str) -> Optional[str]:
    hits = [c for c in colnames if keyword.lower() in c.lower()]
    return hits[0] if hits else None


def cli_plot(path: Path):
    header, colnames, df = parse_ta_sdt_txt(path)

    sample = header.get("Sample", path.stem)
    print(f"\nLoaded: {path.name}")
    print(f"Sample: {sample}")
    print("\nPreview:")
    print(df.head(8).to_string(index=False))

    x_col = choose_column(colnames, "\nChoose X column (index or substring): ")
    y_col = choose_column(colnames, "Choose Y column (index or substring): ")

    print("\nDerivative options:")
    print("  [0] None (plot Y)")
    print("  [1] dY/dt  (derivative vs Time)")
    print("  [2] dY/dT  (derivative vs Temperature)")
    while True:
        dopt = input("Select 0/1/2: ").strip()
        if dopt in {"0", "1", "2"}:
            break

    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)

    y_to_plot = y
    y_label = y_col

    if dopt == "1":
        t_col = find_best_column(colnames, "Time") or choose_column(colnames, "Pick the Time column for dY/dt: ")
        t = df[t_col].to_numpy(dtype=float)
        y_to_plot = compute_derivative(y, t)
        y_label = f"d({y_col})/d({t_col})"
        print(f"Using '{t_col}' for derivative basis.")
    elif dopt == "2":
        T_col = find_best_column(colnames, "Temperature") or choose_column(colnames, "Pick the Temperature column for dY/dT: ")
        T = df[T_col].to_numpy(dtype=float)
        y_to_plot = compute_derivative(y, T)
        y_label = f"d({y_col})/d({T_col})"
        print(f"Using '{T_col}' for derivative basis.")

    plt.figure()
    plt.plot(x, y_to_plot)
    plt.xlabel(x_col)
    plt.ylabel(y_label)
    plt.title(sample)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# =============================================================================
# 7) Entrypoint
# =============================================================================

def launch_tg_gui(file: Optional[str] = None):
    import tkinter as tk
    root = tk.Tk()
    TgGuiApp(root, Path(file) if file else None)
    root.mainloop()


def main():
    ap = argparse.ArgumentParser(description="Read & plot TA-style DTA/DSC/TGA exports.")
    ap.add_argument("file", nargs="?", help="Path to export file (optional).")
    ap.add_argument("--tg", action="store_true", help="Launch Tg GUI tool.")
    args = ap.parse_args()

    path: Optional[Path] = Path(args.file) if args.file else _pick_file_gui()
    if not path:
        raise SystemExit("No file selected/provided.")

    if args.tg:
        launch_tg_gui(str(path))
    else:
        cli_plot(path)


if __name__ == "__main__":
    main()
