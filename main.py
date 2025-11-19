# =====================================================================
# 1. Imports and Global Settings  —  V40 (patched for io_importers)
# =====================================================================

# =====================================================================
# Raman Processing GUI — main (generic, retro-compatible)
# =====================================================================

from __future__ import annotations

# --- standard libs
import os, io, re, json, math, copy, time, warnings, hashlib, datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

# --- UI
import tkinter as tk
from tkinter import ttk, filedialog, simpledialog, messagebox, colorchooser
from ttkthemes import ThemedTk

# --- numerics / plotting
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# --- domain libs you already use
import rampy as rp
import lmfit
from scipy.signal import find_peaks
import colorsys

# CUSTOM IMPORTS (patched: use io_universal universal loader)
from io_universal import load_any as _io_load_any
from ui_fit_params import FitParamWindow
from ui_simple_plot import SimplePlotWindow
# cif_tools.py est importé par ui_simple_plot.py

# Ignore matplotlib layout warnings (Tkinter UI redraw)
warnings.filterwarnings("ignore", message=".*The figure layout has changed to tight.*")

# Model parameter directory (for saving/loading fit models)
MODEL_DIR = "param_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# Heuristics for default X/Y by parser (used before we show a popup)
# We only keep column names; we do NOT persist instrument parameters.
# ---------------------------------------------------------------------
DEFAULT_RULES = {
    # TA SDT (DSC/TGA): prefer Temperature vs Heat Flow if both are available
    # NOTE: we map our adapter canonical keys to those expected here via meta['canonical_map']
    "ta_sdt": [
        (("T_C", "HF_mW"), "Temperature vs Heat Flow"),
        (("time_min", "mass_mg"), "Time vs Mass"),
        (("time_min", "dT_C"), "Time vs dT(°C)"),
        (("T_C", "m_mg"), "Temperature vs Mass"),
    ],
    # SAXS EDF: prefer q vs I
    "saxs_edf_ascii": [
        (("q_A^-1", "I"), "q vs I(q)"),
    ],
    # Fallback for generic XY: take first two columns
    "generic_xy": []
}
# --- Retro-compat alias (au cas où d'autres modules utilisaient l'ancien nom) ---
V40_DEFAULT_RULES = DEFAULT_RULES  # alias conservé pour compatibilité

# When we cannot map canonical keys, we’ll fall back to the first two numeric columns.

# =====================================================================
# 2. Utility Functions (file save/load, helpers)
# =====================================================================

def safe_save_json(filepath: str, data, ask_overwrite: bool = True, parent=None) -> bool:
    """Save JSON with optional overwrite prompt."""
    if ask_overwrite and os.path.exists(filepath):
        if not messagebox.askyesno("Overwrite?",
                                   f"File '{os.path.basename(filepath)}' already exists. Overwrite?",
                                   parent=parent):
            return False
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        messagebox.showerror("Save error", f"Error saving file:\n{e}", parent=parent)
        return False


def safe_save_txt(filepath: str, arr: np.ndarray, ask_overwrite: bool = True, parent=None) -> bool:
    """Save 2-col array to .txt (tab-separated)."""
    if ask_overwrite and os.path.exists(filepath):
        if not messagebox.askyesno("Overwrite?",
                                   f"File '{os.path.basename(filepath)}' already exists. Overwrite?",
                                   parent=parent):
            return False
    try:
        np.savetxt(filepath, arr, delimiter="\t", comments='', fmt="%.6g")
        return True
    except Exception as e:
        messagebox.showerror("Save error", f"Error saving file:\n{e}", parent=parent)
        return False


def list_model_files() -> List[str]:
    """List saved fit parameter models (without extension)."""
    return [f[:-5] for f in os.listdir(MODEL_DIR) if f.endswith(".json")]

def _safe_trapz(y: np.ndarray, x: np.ndarray) -> float:
    try:
        return float(np.trapz(y, x))
    except Exception:
        return 0.0

# ------------------------------ loading helpers ---------------------------

# ==== Adapter over io_universal =================================================
# io_universal.load_any(path, return_meta=True) -> returns (df, meta).
# We keep a tiny wrapper so the rest of this module can remain unchanged.
def _build_canonical_map_for_ta(df: pd.DataFrame) -> Dict[str, str]:
    cols = {c.lower(): c for c in df.columns}
    cmap = {}

    # Temperature
    if "temp_c" in cols: cmap["T_C"] = cols["temp_c"]
    elif "temperature (°c)" in cols: cmap["T_C"] = cols["temperature (°c)"]
    elif "temperature" in cols: cmap["T_C"] = cols["temperature"]

    # Time
    if "time_min" in cols: cmap["time_min"] = cols["time_min"]
    elif "time (min)" in cols: cmap["time_min"] = cols["time (min)"]
    elif "time" in cols: cmap["time_min"] = cols["time"]

    # Heat flow
    if "dsc_mwmg" in cols: cmap["HF_mW"] = cols["dsc_mwmg"]
    elif "dsc" in cols: cmap["HF_mW"] = cols["dsc"]
    elif "heat flow" in cols: cmap["HF_mW"] = cols["heat flow"]

    # Mass (mg)
    if "mass_mg" in cols: cmap["m_mg"] = cols["mass_mg"]

    # dT (°C/min)
    if "dt_c" in cols: cmap["dT_C"] = cols["dt_c"]

    return cmap

def load_any(file_path: str, return_meta: bool = False):
    """Adapter returning (df, meta) compatible with the rest of this file."""
    df, meta = _io_load_any(file_path, return_meta=True)
    meta = (meta or {}).copy()

    # Older code expects canonical_map for TA datasets.
    if meta.get("selected_parser") == "ta_sdt" and "canonical_map" not in meta:
        meta["canonical_map"] = _build_canonical_map_for_ta(df)

    if return_meta:
        return df, meta
    return df
# ===============================================================================
@dataclass
class LoadedXY:
    """Normalized return for any data file → something we can plot."""
    x: np.ndarray
    y: np.ndarray
    df: pd.DataFrame                # full table (kept for later)
    x_col: str                      # chosen X column name
    y_col: str                      # chosen Y column name
    kind: str                       # 'ta_sdt' | 'saxs_edf_ascii' | 'generic_xy' (parser name)
    meta: Dict                      # meta from parser (includes 'canonical_map', 'selected_parser', etc.)
    source_path: str


def _friendly_kind(meta: Dict) -> str:
    name = (meta or {}).get("selected_parser") or ""
    if name == "ta_sdt":
        return "TA SDT (DSC/TGA/DTA)"
    if name == "saxs_edf_ascii":
        return "SAXS (EDF ASCII)"
    return "XY (generic)"


def _auto_pick_xy(df: pd.DataFrame, meta: Dict) -> Tuple[str, str, str, bool]:
    """
    Decide best (x_col, y_col) and return:
        (x_col, y_col, reason_string, is_confident: bool)
    Strategy:
      - Use parser-specific canonical_map when possible (adapter provides it).
      - Apply DEFAULT_RULES for the selected parser.
      - Fall back to first two columns.
    """
    parser = (meta or {}).get("selected_parser", "generic_xy")
    c = (meta or {}).get("canonical_map", {}) or {}
    columns = list(df.columns)

    # 1) Try rule table for this parser with canonical keys
    for (want_keys, reason) in DEFAULT_RULES.get(parser, []):
        kx, ky = want_keys
        cx = c.get(kx)
        cy = c.get(ky)
        if cx in columns and cy in columns:
            return cx, cy, f"{_friendly_kind(meta)} — {reason}", True

    # 2) Heuristics if canonical_map incomplete (e.g. alternative labels)
    if parser == "ta_sdt":
        cx = _guess_col(columns, patterns=("temp",)) or _guess_col(columns, patterns=("temperature",))
        cy = _guess_col(columns, patterns=("heat", "flow")) or _guess_col(columns, patterns=("dsc",))
        if cx and cy:
            return cx, cy, "Heuristic: Temperature vs Heat Flow", False
        cx = _guess_col(columns, patterns=("time",))
        cy = _guess_col(columns, patterns=("mass",)) or _guess_col(columns, patterns=("tg",))
        if cx and cy:
            return cx, cy, "Heuristic: Time vs Mass/TG", False

    if parser == "saxs_edf_ascii":
        cx = _guess_col(columns, patterns=("q(",)) or _guess_col(columns, patterns=(" q ",))
        cy = _guess_col(columns, patterns=("i(q",)) or _guess_col(columns, patterns=("intensity",))
        if cx and cy:
            return cx, cy, "Heuristic: q vs I", False

    # 3) Fallback → first two columns
    if len(columns) >= 2:
        return columns[0], columns[1], "First two columns", False

    raise ValueError("Not enough columns to choose X/Y.")


def _guess_col(columns: List[str], patterns: Tuple[str, ...]) -> Optional[str]:
    """Find first column whose lowercase name contains all given tokens/patterns (loose match)."""
    for col in columns:
        low = col.lower()
        if all(pat.strip().lower() in low for pat in patterns):
            return col
    return None


def _show_xy_selector_dialog(parent, df: pd.DataFrame, meta: Dict,
                             suggested: Tuple[str, str] | None = None,
                             force_type: Optional[str] = None) -> Tuple[str, str, str] | None:
    """
    Popup for user to pick data type + X/Y columns.
    Returns (kind_name, x_col, y_col) or None if canceled.
    """
    top = tk.Toplevel(parent)
    top.title("Select data type and columns")
    top.transient(parent)
    top.grab_set()
    top.resizable(False, False)

    # ------------------ layout ------------------
    frm = ttk.Frame(top, padding=12)
    frm.grid(row=0, column=0, sticky="nsew")

    # Data type selector
    detected_kind = (meta or {}).get("selected_parser", "generic_xy")
    kind_var = tk.StringVar(value=detected_kind)
    ttk.Label(frm, text="Data type:").grid(row=0, column=0, sticky="w")
    kind_cb = ttk.Combobox(frm, textvariable=kind_var, state="readonly", width=24,
                           values=("ta_sdt", "saxs_edf_ascii", "generic_xy"))
    kind_cb.grid(row=0, column=1, sticky="we", padx=(8,0))
    if force_type:
        kind_var.set(force_type)
        kind_cb.configure(state="disabled")

    # X / Y comboboxes
    cols = list(df.columns)
    x_var = tk.StringVar()
    y_var = tk.StringVar()
    if suggested:
        x_var.set(suggested[0])
        y_var.set(suggested[1])

    ttk.Label(frm, text="X column:").grid(row=1, column=0, sticky="w", pady=(8,0))
    x_cb = ttk.Combobox(frm, textvariable=x_var, values=cols, width=40, state="readonly")
    x_cb.grid(row=1, column=1, sticky="we", padx=(8,0), pady=(8,0))

    ttk.Label(frm, text="Y column:").grid(row=2, column=0, sticky="w", pady=(6,0))
    y_cb = ttk.Combobox(frm, textvariable=y_var, values=cols, width=40, state="readonly")
    y_cb.grid(row=2, column=1, sticky="we", padx=(8,0), pady=(6,0))

    # Hint / auto-detect info
    hint_txt = tk.Text(frm, height=3, width=56)
    hint_txt.grid(row=3, column=0, columnspan=2, sticky="we", pady=(10,0))
    hint_msg = (
        f"Detected: {_friendly_kind(meta)}\n"
        f"Columns: {', '.join(cols[:6])}{'…' if len(cols)>6 else ''}\n"
        "Pick X/Y or accept the suggestion."
    )
    hint_txt.insert("1.0", hint_msg)
    hint_txt.configure(state="disabled")

    # Buttons
    btns = ttk.Frame(frm)
    btns.grid(row=4, column=0, columnspan=2, pady=(12,0))
    res = {"ok": False}
    def on_ok():
        if not x_var.get() or not y_var.get():
            messagebox.showwarning("Missing", "Please select both X and Y.", parent=top)
            return
        res["ok"] = True
        top.destroy()

    def on_cancel():
        res["ok"] = False
        top.destroy()

    ttk.Button(btns, text="OK", command=on_ok).grid(row=0, column=0, padx=6)
    ttk.Button(btns, text="Cancel", command=on_cancel).grid(row=0, column=1, padx=6)

    # Key bindings
    top.bind("<Return>", lambda e: on_ok())
    top.bind("<Escape>", lambda e: on_cancel())

    top.wait_window()
    if not res["ok"]:
        return None
    return kind_var.get(), x_var.get(), y_var.get()

def load_file_as_xy(file_path: str, parent=None, *, force_popup: bool = False) -> LoadedXY:
    """
    High-level helper used by your 'Open file' action.
    - Uses universal loader to parse file to a DataFrame (+meta).
    - Auto-picks sensible X/Y mapping when obvious.
    - If ambiguous or force_popup=True: shows a UI to let the user pick data type and columns.
    Returns a LoadedXY object (x, y, df, x_col, y_col, kind, meta, source_path).
    """
    # Parse with universal loader (adapter)
    df, meta = load_any(file_path, return_meta=True)
    kind = (meta or {}).get("selected_parser", "generic_xy")

    # Auto-pick X/Y
    x_col, y_col, reason, confident = _auto_pick_xy(df, meta)

    # If not confident or user wants control → popup
    if force_popup or not confident:
        picked = _show_xy_selector_dialog(parent, df, meta, suggested=(x_col, y_col))
        if picked is None:
            raise RuntimeError("User canceled column selection.")
        kind, x_col, y_col = picked

    # Extract arrays
    x = df[x_col].astype(float).to_numpy()
    y = df[y_col].astype(float).to_numpy()

    # Ensure strictly increasing X (stable mergesort)
    order = np.argsort(x, kind="mergesort")
    x = x[order]; y = y[order]

    return LoadedXY(
        x=x, y=y, df=df,
        x_col=x_col, y_col=y_col,
        kind=kind, meta=meta or {},
        source_path=str(file_path)
    )

def load_file_as_xy_noprompt(file_path: str) -> LoadedXY:
    """Comme load_file_as_xy, mais ne déclenche JAMAIS de popup (accepte l'auto-pick même peu sûr)."""
    df, meta = load_any(file_path, return_meta=True)
    kind = meta.get("selected_parser", "generic_xy")

    # Auto-pick (accepte même si 'not confident')
    x_col, y_col, _reason, _confident = _auto_pick_xy(df, meta)

    # Fallback s'il n'y a vraiment pas 2 colonnes
    if x_col is None or y_col is None:
        cols = list(df.columns)
        if len(cols) >= 2:
            x_col, y_col = cols[0], cols[1]
        else:
            raise ValueError("Not enough numeric columns to choose X/Y.")

    x = df[x_col].astype(float).to_numpy()
    y = df[y_col].astype(float).to_numpy()
    order = np.argsort(x, kind="mergesort")
    x, y = x[order], y[order]

    return LoadedXY(
        x=x, y=y, df=df,
        x_col=x_col, y_col=y_col,
        kind=kind, meta=meta,
        source_path=str(file_path)
    )

def _map_parser_to_simple_kind(parser_name: str) -> str:
    """Mappe les noms de parseur vers les tags attendus par ui_simple_plot."""
    if parser_name == "ta_sdt":
        return "TA_SDT"
    # tout le reste → XY
    return "XY"

def simpleplot_unified_loader(path: str):
    """
    Adapteur pour ui_simple_plot : renvoie un dict avec .['kind'] ('XY'/'TA_SDT'), x, y, df, meta, x_col, y_col.
    AUCUN popup.
    """
    payload = load_file_as_xy_noprompt(path)
    simple_kind = _map_parser_to_simple_kind((payload.meta or {}).get("selected_parser", "generic_xy"))
    return {
        "kind": simple_kind,
        "x": payload.x,
        "y": payload.y,
        "df": payload.df,
        "meta": payload.meta,
        "x_col": payload.x_col,
        "y_col": payload.y_col,
    }

def pick_default_xy_for_ta_sdt(df: pd.DataFrame):
    """
    Choose a sensible default XY for TA SDT data.
    Priority X: temp_C → time_min
    Priority Y: mass_pct/mass_mg → DSC (dsc_mWmg/dsc) → dT_C/dT_uV → first two numeric cols
    Returns (x, y, info_dict)
    """
    # Accept both the adapter-friendly and legacy names
    x_candidates = ["temp_C", "time_min", "Temperature", "Time (min)", "Time"]
    y_candidates = ["mass_pct", "mass_mg", "dsc_mWmg", "dsc", "heatflow_mW", "dT_C", "dT_uV", "tg_pct", "dtg_pct_min"]

    # Normalize column name lookup (case-insensitive)
    cols_map = {c.lower(): c for c in df.columns}

    def pick(names):
        for n in names:
            c = cols_map.get(n.lower())
            if c is not None:
                return c
        return None

    x_col = pick(x_candidates)
    y_col = pick(y_candidates)

    if x_col is None or y_col is None:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(num_cols) >= 2:
            x_col, y_col = num_cols[0], num_cols[1]
        else:
            raise ValueError("TA SDT file loaded, but insufficient numeric columns to plot.")

    x = df[x_col].to_numpy()
    y = df[y_col].to_numpy()
    return x, y, {"x_col": x_col, "y_col": y_col, "label": f"{y_col} vs {x_col}"}

# ------------------------------ Convenience shims ------------------------------

def open_and_plot_file(app, parent=None, *, force_popup: bool = False):
    """
    Example glue you can call from your 'Open…' menu action:
      - Shows file dialog, loads, adds a trace to your plot frame.
    Expect `app.plot_frame.add_trace(x, y, label=...)` to exist (if you keep that pattern).
    """
    path = filedialog.askopenfilename(parent=parent, title="Open data file")
    if not path:
        return
    try:
        payload = load_file_as_xy(path, parent=parent, force_popup=force_popup)
    except Exception as e:
        messagebox.showerror("Load error", f"{e}", parent=parent)
        return

    label = os.path.basename(path)
    nice = _friendly_kind(payload.meta)
    label = f"{label}  [{nice}: {payload.x_col} vs {payload.y_col}]"

    if hasattr(app, "plot_frame") and hasattr(app.plot_frame, "add_trace"):
        app.plot_frame.add_trace(payload.x, payload.y, label=label)

    # Optionally stash
    if not hasattr(app, "loaded_tables"):
        app.loaded_tables = []
    app.loaded_tables.append(payload)

def choose_xy_for_current_table(app, parent=None):
    """
    Allow user to re-map X/Y for the last loaded table and replot it,
    without reloading the file.
    """
    if not getattr(app, "loaded_tables", None):
        messagebox.showinfo("Info", "No table loaded yet.", parent=parent); return
    payload = app.loaded_tables[-1]
    picked = _show_xy_selector_dialog(parent, payload.df, payload.meta,
                                      suggested=(payload.x_col, payload.y_col),
                                      force_type=payload.kind)
    if picked is None:
        return
    kind, x_col, y_col = picked
    x = payload.df[x_col].astype(float).to_numpy()
    y = payload.df[y_col].astype(float).to_numpy()
    order = np.argsort(x, kind="mergesort")
    x, y = x[order], y[order]

    payload.x, payload.y = x, y
    payload.x_col, payload.y_col = x_col, y_col
    if hasattr(app, "plot_frame") and hasattr(app.plot_frame, "add_trace"):
        app.plot_frame.add_trace(x, y, label=f"{Path(payload.source_path).name}  [{_friendly_kind(payload.meta)}: {x_col} vs {y_col}]")

# =====================================================================
# 3. GUI Classes
# =====================================================================

def on_open_fit_params(self):
    spectra_names = [tr.label for tr in getattr(self.plot_frame, "traces", [])] or ["Current"]
    current = spectra_names[0]

    if not hasattr(self, "fit_param_memory"):
        self.fit_param_memory = {}

    FitParamWindow(
        master=self.root,
        spectra_names=spectra_names,
        fit_param_memory=self.fit_param_memory,
        current_spectrum=current,
        model_dir=MODEL_DIR,
        callback=self.on_fit_params_updated
    )

def on_fit_params_updated(self, memory: Dict[str, list]):
    self.fit_param_memory = memory
    safe_save_json(os.path.join(MODEL_DIR, "_last_session_params.json"), memory, ask_overwrite=False, parent=self.root)

###########################################################################
# 3.1 FitParamWindow: (imported)
###########################################################################

###########################################################################
# 3.2 SimplePlotWindow: (imported)
###########################################################################

#########################################################################
# 3.3 RamanApp: Main Application GUI
#########################################################################

class RamanApp:
    """
    Main application class for the Raman spectra processing GUI.
    Handles all the top-level controls, file imports, and launches sub-windows for plotting,
    baseline settings, parameter editing, summing, and fitting.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Raman processing v1")
        self.root.geometry("650x380")
        self.file_paths = []
        self.file_titles = []
        self.file_statuses = []
        self.fit_param_memory = {}
        # NEW: cache des mappings X/Y choisis (payloads LoadedXY)
        self.xy_by_path: Dict[str, LoadedXY] = {}
        # Layout
        self._setup_layout()
        self.update_file_listbox()

    def _setup_layout(self):
        style = ttk.Style(self.root)

        # Frame centrale fond doux
        CENTER_BG = "#f4f2fa"
        style.configure('Center.TFrame', background=CENTER_BG)
        # Text widget, police, et fond
        TEXT_BG = "#f4f2fa"
        TEXT_FG = "#222"

        """Initialises all UI frames and buttons."""
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=2)
        self.root.columnconfigure(2, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Left frame: general parameter and file ops
        frame_left = ttk.Frame(self.root, padding=12)
        frame_left.grid(row=0, column=0, sticky="ns")
        frame_left.grid_propagate(False)
        ttk.Label(frame_left, text="Parameters", font=("Arial", 12, "bold")).pack(pady=(0,12))
        ttk.Button(frame_left, text="Quick import", width=18, command=self.import_files_quick).pack(pady=4)
        ttk.Button(frame_left, text="Custom import", width=18, command=self.import_files_custom).pack(pady=3)
        ttk.Button(frame_left, text="Rename", width=18, command=self.rename).pack(pady=3)
        ttk.Button(frame_left, text="Reorder", width=18, command=self.reorder).pack(pady=3)
        ttk.Button(frame_left, text="Clear imports", width=18, command=self.clear_imports).pack(pady=3)
        ttk.Button(frame_left, text="Baseline param.", width=18, command=self.baseline_param).pack(pady=3)
        ttk.Button(frame_left, text="Fit param.", width=18, command=self.fit_param).pack(pady=3)
        ttk.Button(frame_left, text="Exit", width=18, command=self.exit_app).pack(side="bottom", pady=(15,3))

        # Center frame: imported files display
        frame_center = ttk.Frame(self.root, padding=12, style='Center.TFrame')
        frame_center.grid(row=0, column=1, sticky="nsew")
        frame_center.grid_propagate(False)
        ttk.Label(frame_center, text="Imported files list", font=("Arial", 12, "bold"), background=CENTER_BG).pack(pady=(0,8), anchor='w')
        self.text_files = tk.Text(
            frame_center, width=45, height=15,
            state="disabled",
            bg=TEXT_BG, fg=TEXT_FG,
            bd=0, highlightthickness=0, font=("Consolas", 11)
        )
        self.text_files.pack(fill="both", expand=True, padx=2, pady=(0,2))
        self.text_files.tag_configure("filename", foreground="#666", font=("Consolas", 10, "italic"))
        self.text_files.tag_configure("title", foreground="#333", font=("Consolas", 11, "bold"))

        # Right frame: spectrum processing tools
        frame_right = ttk.Frame(self.root, padding=12)
        frame_right.grid(row=0, column=2, sticky="ns")
        ttk.Label(frame_right, text="Processing", font=("Arial", 12, "bold")).pack(pady=(0,12))
        ttk.Button(frame_right, text="Simple plot", width=16, command=self.simple_plot).pack(pady=4)
        ttk.Button(frame_right, text="Sum spectra", width=16, command=self.sum_spectra).pack(pady=3)
        ttk.Button(frame_right, text="1 fit", width=16, command=self.one_fit).pack(pady=3)
        ttk.Button(frame_right, text="Multi fit", width=16, command=self.multi_fit).pack(pady=3)

    # ========== App controls ==========

    def exit_app(self):
        """Exit the app and close all Matplotlib figures."""
        import matplotlib.pyplot as plt
        plt.close('all')
        self.root.destroy()

    # ========== File Import/Management ==========

    def import_files_quick(self):
        """Batch import silencieux : auto-détection type + X/Y ; jamais de popup."""
        paths = filedialog.askopenfilenames(title="Select files")
        if not paths:
            return
        if not hasattr(self, "xy_by_path"):
            self.xy_by_path = {}
        added = False
        for path in paths:
            if path in self.file_paths:
                continue
            try:
                payload = load_file_as_xy_noprompt(path)  # <<< no prompt
            except Exception as e:
                messagebox.showerror("Import error", f"{os.path.basename(path)}\n{e}", parent=self.root)
                continue

            # Mémorise un enregistrement normalisé que toutes les fenêtres peuvent relire
            simple_kind = _map_parser_to_simple_kind((payload.meta or {}).get("selected_parser", "generic_xy"))
            self.xy_by_path[path] = {
                "kind": simple_kind,
                "x": payload.x,
                "y": payload.y,
                "df": payload.df,
                "meta": payload.meta,
                "x_col": payload.x_col,
                "y_col": payload.y_col,
            }

            self.file_paths.append(path)
            self.file_titles.append(Path(path).stem)
            self.file_statuses.append("imported")
            added = True

        if added:
            self.update_file_listbox()
        else:
            messagebox.showinfo("Import", "No new files were added.", parent=self.root)

    def import_files_custom(self):
        """Import wizard : on laisse l’utilisateur choisir type et colonnes (quand il le souhaite)."""
        paths = filedialog.askopenfilenames(title="Select files")
        if not paths:
            return
        if not hasattr(self, "xy_by_path"):
            self.xy_by_path = {}
        added = False
        for path in paths:
            if path in self.file_paths:
                continue
            try:
                payload = load_file_as_xy(path, parent=self.root, force_popup=True)
            except RuntimeError as e:
                # User canceled the selector dialog → silently skip
                if "User canceled" in str(e):
                    continue
                messagebox.showerror("Import error", f"{os.path.basename(path)}\n{e}", parent=self.root)
                continue
            except Exception as e:
                messagebox.showerror("Import error", f"{os.path.basename(path)}\n{e}", parent=self.root)
                continue

            simple_kind = _map_parser_to_simple_kind((payload.meta or {}).get("selected_parser", "generic_xy"))
            self.xy_by_path[path] = {
                "kind": simple_kind,
                "x": payload.x,
                "y": payload.y,
                "df": payload.df,
                "meta": payload.meta,
                "x_col": payload.x_col,
                "y_col": payload.y_col,
            }

            self.file_paths.append(path)
            self.file_titles.append(Path(path).stem)
            self.file_statuses.append("imported")
            added = True

        if added:
            self.update_file_listbox()
        else:
            messagebox.showinfo("Import", "No new files were added.", parent=self.root)

    def clear_imports(self):
        """Remove all imported files after confirmation."""
        resp = messagebox.askyesno("Clear imports", "Are you sure you want to remove all imported files?")
        if resp:
            self.file_paths = []
            self.file_titles = []
            self.file_statuses = []
            self.update_file_listbox()

    def update_file_listbox(self):
        self.text_files.config(state="normal", bg="#f4f2fa", fg="#222")
        self.text_files.delete("1.0", tk.END)
        for title, path, status in zip(self.file_titles, self.file_paths, self.file_statuses):
            tag = "title"
            if status == "sum":
                tag = "sum"
            elif status == "baseline":
                tag = "baseline"
            self.text_files.insert(tk.END, title, tag)
            self.text_files.insert(tk.END, f"  ({os.path.basename(path)})\n", "filename")
        self.text_files.tag_configure("sum", foreground="#219150", font=("Consolas", 11, "bold"))
        self.text_files.tag_configure("baseline", foreground="#b61b37", font=("Consolas", 11, "bold"))
        self.text_files.tag_configure("title", foreground="#333", font=("Consolas", 11, "bold"))
        self.text_files.config(state="disabled")

    # ========== File Renaming/Reordering ==========

    def rename(self):
        """Popup window to rename imported files."""
        if not self.file_paths:
            messagebox.showinfo("Info", "No files to rename.")
            return
        win = tk.Toplevel(self.root)
        win.title("Rename files")
        vars_titles = []
        for i, title in enumerate(self.file_titles):
            ttk.Label(win, text=f"File {i+1} :").grid(row=i, column=0, padx=5, pady=2, sticky='e')
            var = tk.StringVar(value=title)
            vars_titles.append(var)
            ttk.Entry(win, textvariable=var, width=40).grid(row=i, column=1, padx=5, pady=2)
        def apply_rename():
            self.file_titles = [var.get() for var in vars_titles]
            self.update_file_listbox()
            win.destroy()
        ttk.Button(win, text="OK", command=apply_rename).grid(row=len(self.file_titles), column=0, columnspan=2, pady=10)

    def reorder(self):
        """Popup window to change file order via combo boxes."""
        if not self.file_paths:
            messagebox.showinfo("Info", "No files to reorder.")
            return
        win = tk.Toplevel(self.root)
        win.title("Reorder files")
        vars_order = [tk.StringVar(value=title) for title in self.file_titles]
        all_titles = list(self.file_titles)
        ttk.Label(win, text="Choose order:").grid(row=0, column=0, columnspan=2)
        for i in range(len(self.file_titles)):
            ttk.Label(win, text=f"Position {i+1}").grid(row=i+1, column=0, padx=5, pady=2, sticky="e")
            cb = ttk.Combobox(win, textvariable=vars_order[i], values=all_titles, state="readonly", width=35)
            cb.grid(row=i+1, column=1, padx=5, pady=2)
        def apply_reorder():
            selected_titles = [var.get() for var in vars_order]
            if len(set(selected_titles)) != len(selected_titles):
                messagebox.showerror("Error", "Each file must be selected only once.")
                return
            new_paths = [self.file_paths[self.file_titles.index(title)] for title in selected_titles]
            new_statuses = [self.file_statuses[self.file_titles.index(title)] for title in selected_titles]
            self.file_paths = new_paths
            self.file_titles = selected_titles
            self.file_statuses = new_statuses
            self.update_file_listbox()
            win.destroy()
        ttk.Button(win, text="OK", command=apply_reorder).grid(row=len(self.file_titles)+2, column=0, columnspan=2, pady=10)

    # ========== Baseline & Fitting Params ==========

    def baseline_param(self):
        if not self.file_paths:
            messagebox.showinfo("Info", "No files imported.")
            return

        # cache local
        if not hasattr(self, "xy_by_path"):
            self.xy_by_path = {}

        spectra_xy_list = []
        for fp in self.file_paths:
            rec = self.xy_by_path.get(fp)
            if rec is None:
                # charge SANS popup → objet LoadedXY
                payload = load_file_as_xy_noprompt(fp)
                # mémorise en dict homogène pour tout l'app
                rec = {
                    "kind": _map_parser_to_simple_kind((payload.meta or {}).get("selected_parser","generic_xy")),
                    "x": payload.x, "y": payload.y,
                    "df": payload.df, "meta": payload.meta,
                    "x_col": payload.x_col, "y_col": payload.y_col
                }
                self.xy_by_path[fp] = rec

            # rec est un dict → always use keys
            x = np.asarray(rec["x"], dtype=float)
            y = np.asarray(rec["y"], dtype=float)
            spectra_xy_list.append((x, y))

        BaselineParamWindow(
            self.root,
            spectra_titles=self.file_titles,
            spectra_xy_list=spectra_xy_list,
            current_index=0,
            roi_n=6,
            baseline_type="poly5",
            callback=None,
            file_callback=self.on_new_file_exported
        )


    def fit_param(self):
        """Open parameter window for fitting a spectrum."""
        if not self.file_paths:
            messagebox.showinfo("Info", "No files imported.")
            return
        current_spectrum = self.file_titles[0]
        FitParamWindow(self.root, self.file_titles, self.fit_param_memory, current_spectrum=current_spectrum)

    # ========== Spectrum Processing Windows ==========

    def sum_spectra(self):
        """Open window to compute the sum of selected spectra."""
        if not self.file_paths:
            messagebox.showinfo("Info", "No files to sum.")
            return
        SpectralSumWindow(
            self.root,
            self.file_paths,
            self.file_titles,
            callback=self.on_new_file_exported,
            xy_cache=getattr(self, "xy_by_path", {})
        )

    def one_fit(self):
        if not self.file_paths:
            messagebox.showinfo("Info", "No files to fit.")
            return
        SingleFitWindow(self.root, self.file_paths, self.file_titles, self.fit_param_memory, xy_cache=getattr(self, "xy_by_path", {}))

    def multi_fit(self):
        """Stub for multi-spectrum fit. To implement."""
        pass

    def simple_plot(self):
        if not self.file_paths:
            messagebox.showinfo("Info", "No files to plot.")
            return
        SimplePlotWindow(
            self.root,
            self.file_paths,
            self.file_titles,
            load_any_func=simpleplot_unified_loader,
            pick_ta_xy_func=pick_default_xy_for_ta_sdt
        )

    def on_new_file_exported(self, path, status="sum"):
        """
        When a new file (sum or baseline-corrected) is exported,
        add it to the imported file list.
        """
        if isinstance(path, str) and path and os.path.exists(path):
            self.file_paths.append(path)
            self.file_titles.append(os.path.splitext(os.path.basename(path))[0])
            self.file_statuses.append(status)
            self.update_file_listbox()

#########################################################################
# 3.4 SpectralSumWindow: Spectral summing and signal/noise window
#########################################################################

class SpectralSumWindow(tk.Toplevel):
    """
    Window for summing selected Raman spectra.
    Allows S/N inspection, optional baseline subtraction and normalization, and export of the summed spectrum.
    """
    COLORS = ["navy", "darkred", "seagreen", "darkorange", "purple", "teal", "brown", "indigo"]

    def __init__(self, master, file_paths, file_titles, callback=None, xy_cache=None):
        super().__init__(master)
        self.title("Spectral sum")
        self.geometry("1200x750")
        self.file_paths = file_paths
        self.file_titles = file_titles
        self.callback = callback
        self.baseline_params = None
        self.xy_cache = xy_cache or {}
        # Load spectra (x,y) quietly — AFTER xy_cache is set
        self.spectra = [self._xy_from_cache(p) for p in file_paths]

        # Track selection
        self.list_vars = []
        self.sn_labels = []
        self.selected_idxs = list(range(len(self.file_paths)))  # all by default

        # ---- UI Layout ----
        self._build_left_panel()
        self._build_right_panel()
        self.update()

    # ==== UI Layout Methods ====

    def _build_left_panel(self):
        """Build the left panel for spectrum selection, baseline and normalization options."""
        frame_left = ttk.Frame(self)
        frame_left.grid(row=0, column=0, sticky="ns", padx=14, pady=10)

        ttk.Label(frame_left, text="Spectra").grid(row=0, column=0, sticky="w")
        ttk.Label(frame_left, text="s/n").grid(row=0, column=1, sticky="w")

        # Scrollable spectrum list with checkboxes
        self.frame_list = ttk.Frame(frame_left)
        self.frame_list.grid(row=1, column=0, columnspan=2, sticky="nsew")
        canvas = tk.Canvas(self.frame_list, width=185, height=180)
        scrollbar = ttk.Scrollbar(self.frame_list, orient="vertical", command=canvas.yview)
        self.listbox_frame = ttk.Frame(canvas)
        self.listbox_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.listbox_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.sn_method = tk.StringVar(value="classic")
        sn_row = ttk.Frame(frame_left)
        sn_row.grid(row=6, column=0, columnspan=2, pady=(4,0), sticky="w")
        ttk.Label(sn_row, text="S/N method:").pack(side="left")
        ttk.Radiobutton(sn_row, text="Classic", variable=self.sn_method, value="classic", command=self.update).pack(side="left", padx=(3,0))
        ttk.Radiobutton(sn_row, text="S-G", variable=self.sn_method, value="sg", command=self.update).pack(side="left", padx=(3,0))
        ttk.Radiobutton(sn_row, text="Poly10", variable=self.sn_method, value="poly10", command=self.update).pack(side="left", padx=(3,0))

        # Each spectrum gets a checkbox and S/N label
        for i, title in enumerate(self.file_titles):
            var = tk.BooleanVar(value=True)
            chk = ttk.Checkbutton(self.listbox_frame, text=title, variable=var, command=self.update)
            chk.grid(row=i, column=0, sticky="w")
            self.list_vars.append(var)
            sn = ttk.Label(self.listbox_frame, text="", background="#a68aff", width=7, anchor="center", relief="ridge")
            sn.grid(row=i, column=1, sticky="e", padx=(10,2))
            self.sn_labels.append(sn)

        # Options for selection and processing
        self.var_all = tk.BooleanVar(value=True)
        self.var_baseline = tk.BooleanVar()
        self.var_norm = tk.BooleanVar()
        ttk.Checkbutton(frame_left, text="All imported", variable=self.var_all, command=self.toggle_all).grid(row=2, column=0, sticky="w", pady=(5,0))
        ttk.Checkbutton(frame_left, text="Baseline subtracted", variable=self.var_baseline, command=self.update).grid(row=3, column=0, sticky="w")
        ttk.Checkbutton(frame_left, text="Norm.", variable=self.var_norm, command=self.update).grid(row=5, column=0, sticky="w", pady=(5,0))

        # Baseline param and reset
        btns_row = ttk.Frame(frame_left)
        btns_row.grid(row=4, column=0, columnspan=2, sticky="w")
        ttk.Button(btns_row, text="Baseline param.", command=self.set_baseline_param).pack(side="left")
        ttk.Button(btns_row, text="Reset ROI", command=self.reset_baseline_params).pack(side="left", padx=8)

    def _build_right_panel(self):
        """Build the right panel with S/N for sum, plot, and export button."""
        frame_right = ttk.Frame(self)
        frame_right.grid(row=0, column=1, padx=8, pady=8, sticky="nsew")
        self.sn_sum_label = ttk.Label(frame_right, text="Summed S/N: ---", background="#a68aff", width=18, anchor="center", relief="ridge")
        self.sn_sum_label.grid(row=0, column=0, sticky="w", padx=(0,12))
        self.fig = plt.Figure(figsize=(5.2,3.7))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame_right)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", pady=3)
        ttk.Button(frame_right, text="Export as txt", command=self.export_sum).grid(row=2, column=0, sticky="e", pady=(5,0))

        self.last_sum_x = None
        self.last_sum_y = None

    def _on_manual_select(self, event=None):
        checked = all(var.get() for var in self.list_vars)
        self.var_all.set(checked)
        self.update()

    def _xy_from_cache(self, path):
        rec = self.xy_cache.get(path)
        if rec is None:
            # fallback: silent loader
            payload = load_file_as_xy_noprompt(path)
            rec = {
                "kind": _map_parser_to_simple_kind((payload.meta or {}).get("selected_parser","generic_xy")),
                "x": payload.x, "y": payload.y,
                "df": payload.df, "meta": payload.meta,
                "x_col": payload.x_col, "y_col": payload.y_col
            }
            self.xy_cache[path] = rec
        x = np.asarray(rec["x"], dtype=float)
        y = np.asarray(rec["y"], dtype=float)
        return (x, y)

    # ==== Utility Methods ====

    def toggle_all(self):
        """Select or unselect all spectra checkboxes."""
        state = self.var_all.get()
        for var in self.list_vars:
            var.set(state)
        self.update()

    def reset_baseline_params(self):
        """Remove all baseline parameters."""
        self.baseline_params = None
        self.update()

    def set_baseline_param(self):
        """Open a baseline parameter window for the first spectrum (can be improved)."""
        idx = 0
        x, y = self.spectra[idx]
        roi_init = [[110.0, 120.0], [160.0, 190.0], [845.0, 870.0], [1230.0, 1270.0], [1720.0, 1900.0]]
        baseline_type = "poly5"
        if self.baseline_params:
            roi_init, baseline_type = self.baseline_params
        def callback(rois, method):
            self.baseline_params = (rois, method)
            self.update()
        BaselineParamWindow(
            self,
            spectra_titles=[self.file_titles[idx] if idx < len(self.file_titles) else 'Summed spectrum'],
            spectra_xy_list=[(x, y)],
            current_index=0,
            roi_n=6,
            baseline_type=baseline_type,
            callback=callback,
            roi_init=roi_init,
            simple_mode=True
        )

    def compute_sn(self, y, window=5, x=None):
        method = self.sn_method.get() if hasattr(self, "sn_method") else "classic"
        if method == "classic" or x is None:
            if len(y) < window + 2:
                return np.nan
            smooth = np.convolve(y, np.ones(window)/window, mode='same')
            noise = y - smooth
            noise_std = np.std(noise)
            signal = y.max() - y.min()
            return np.nan if noise_std == 0 else signal / noise_std
        elif method == "sg":
            from scipy.signal import savgol_filter
            win = min(21, len(y)-2 if len(y)%2==1 else len(y)-3)
            win = win if win > 5 else 5
            try:
                smooth = savgol_filter(y, window_length=win, polyorder=3)
            except Exception:
                return np.nan
            noise = y - smooth
            noise_std = np.std(noise)
            signal = y.max() - y.min()
            return np.nan if noise_std == 0 else signal / noise_std
        elif method == "poly10" and x is not None:
            try:
                p = np.polyfit(x, y, deg=min(10, len(y)-1))
                smooth = np.polyval(p, x)
            except Exception:
                return np.nan
            noise = y - smooth
            noise_std = np.std(noise)
            signal = y.max() - y.min()
            return np.nan if noise_std == 0 else signal / noise_std
        else:
            return np.nan

    def baseline_subtract(self, x, y):
        """Apply the selected baseline subtraction method. Fallback: linear baseline."""
        if not self.baseline_params:
            coef = np.polyfit(x, y, 1)
            baseline = np.polyval(coef, x)
            return y - baseline

        rois, method = self.baseline_params
        try:
            if method.startswith("poly"):
                degree = int(method[4:]) if method[4:].isdigit() else 5
                ycorr, _ = rp.baseline(x, y, rois, "poly", polynomial_order=degree)
            elif method == "unispline":
                ycorr, _ = rp.baseline(x, y, rois, "unispline", s=1e0)
            elif method == "als":
                ycorr, _ = rp.baseline(x, y, rois, "als", lam=1e5, p=0.05)
            elif method == "arPLS":
                ycorr, _ = rp.baseline(x, y, rois, "arPLS", lam=1e6, ratio=0.001)
            elif method == "drPLS":
                ycorr, _ = rp.baseline(x, y, rois, "drPLS")
            elif method == "rubberband":
                ycorr, _ = rp.baseline(x, y, rois, "rubberband")
            else:
                messagebox.showerror("Error", f"Unknown method: {method}")
                return y
            return ycorr
        except Exception as e:
            messagebox.showerror("Baseline error", f"Baseline subtraction failed:\n{e}")
            return y

    # ==== Main Logic ====

    def update(self):
        """Update the summed spectrum plot, S/Ns, and UI according to selection and options."""
        selected = [i for i, v in enumerate(self.list_vars) if v.get()]
        self.var_all.set(len(selected) == len(self.list_vars))
        sn_list = []
        spectra_proc = []

        for i, (x, y) in enumerate(self.spectra):
            y_proc = np.copy(y)
            if self.var_baseline.get() and x.size and y.size:
                y_proc = self.baseline_subtract(x, y_proc)
            sn = self.compute_sn(y_proc, x=x) if x.size and y.size else np.nan
            sn_list.append(sn)
            if i in selected and x.size and y.size:
                spectra_proc.append((x, y_proc))
            self.sn_labels[i].config(text=("" if np.isnan(sn) else f"{sn:.1f}"))

        # Compute and display sum
        if spectra_proc:
            x_ref = spectra_proc[0][0]
            Y = []
            for (x, y) in spectra_proc:
                if np.array_equal(x, x_ref):
                    Y.append(y)
                else:
                    Y.append(np.interp(x_ref, x, y))
            y_sum = np.mean(Y, axis=0)

            # --- NORMALISATION (remplace l'ancien bloc) ---
            if self.var_norm.get():
                area = float(np.trapz(y_sum, x_ref)) if len(x_ref) and len(y_sum) else 0.0
                if abs(area) > 1e-10:
                    y_sum = y_sum / area * 100.0
                else:
                    messagebox.showwarning("Normalization",
                                           "Normalization skipped: near-zero area for the current sum.",
                                           parent=self)
            # ----------------------------------------------

            sn_sum = self.compute_sn(y_sum, x=x_ref)
            self.sn_sum_label.config(text=f"Summed S/N: {'' if np.isnan(sn_sum) else f'{sn_sum:.1f}'}", background="#a68aff")
            self.ax.clear()
            self.ax.plot(x_ref, y_sum, color="navy")
            self.ax.set_title("Summed spectrum")
            self.ax.set_xlabel("Raman shift (cm⁻¹)")
            self.ax.set_ylabel("Intensity (a.u.)")
            self.ax.grid(alpha=0.14)
            self.fig.tight_layout()
            self.canvas.draw()
            self.last_sum_x = x_ref
            self.last_sum_y = y_sum
        else:
            self.sn_sum_label.config(text=f"Summed S/N: ---", background="#a68aff")
            self.ax.clear()
            self.ax.set_title("Summed spectrum")
            self.fig.tight_layout()
            self.canvas.draw()
            self.last_sum_x = None
            self.last_sum_y = None

    def export_sum(self):
        """Export the summed spectrum as a .txt file (tab-separated)."""
        if getattr(self, "last_sum_x", None) is None or getattr(self, "last_sum_y", None) is None:
            messagebox.showinfo("Export", "No summed spectrum to export.", parent=self)
            return
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt", filetypes=[("Text", "*.txt"), ("All files", "*.*")],
            title="Export summed spectrum as txt")
        if not filename:
            return
        arr = np.column_stack((self.last_sum_x, self.last_sum_y))
        if not safe_save_txt(filename, arr, ask_overwrite=True, parent=self):
            return
        messagebox.showinfo("Export", f"Summed spectrum exported to:\n{filename}", parent=self)
        if self.callback:
            self.callback(filename, status="sum")

    def destroy(self):
        """Ensure matplotlib figures are closed on window close."""
        if hasattr(self, 'fig'):
            plt.close(self.fig)
        super().destroy()

#########################################################################
# 3.5 BaselineParamWindow: Baseline regions and method selection dialog
#########################################################################

class BaselineParamWindow(tk.Toplevel):
    """
    Popup window for interactive selection of baseline ROIs and method,
    preview of the correction, and export of the baseline or corrected spectrum.
    Mémorise les paramètres pour chaque spectre dans la session.
    """

    BASELINE_TYPES = [
        ("Polynomial-4", "poly4"),
        ("Polynomial-5", "poly5"),
        ("Polynomial-6", "poly6"),
        ("Unispline", "unispline"),
        ("ALS", "als"),
        ("arPLS", "arPLS"),
        ("drPLS", "drPLS"),
        ("Rubberband", "rubberband")
    ]

    def __init__(
        self, master, spectra_titles, spectra_xy_list, current_index=0,
        roi_n=6, baseline_type="poly5", callback=None, file_callback=None, roi_init=None,
        simple_mode=False, memory=None
    ):
        super().__init__(master)
        self.spec_states = {}
        self.title("Baseline parameters")
        self.resizable(False, False)
        self.spectra_titles = spectra_titles
        self.spectra_xy_list = spectra_xy_list
        self.callback = callback
        self.file_callback = file_callback
        self.simple_mode = simple_mode
        self.baseline_type = baseline_type
        self.memory = memory if memory is not None else {}

        # State by spectrum (to be restored when switching)
        self.spec_var = tk.StringVar(value=self.spectra_titles[current_index])
        self.current_index = current_index

        # Top combobox to select spectrum
        top = ttk.Frame(self)
        top.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=5)
        ttk.Label(top, text="Select spectrum:").pack(side="left")
        self.combo = ttk.Combobox(top, values=self.spectra_titles, textvariable=self.spec_var, state="readonly", width=32)
        self.combo.pack(side="left", padx=6)
        self.combo.bind("<<ComboboxSelected>>", self.on_spectrum_changed)

        # State per spectrum (default/init)
        self._init_spec_state(self.spec_var.get(), roi_n, baseline_type, roi_init)

        # Entry vars
        self.entries_min = []
        self.entries_max = []
        self.xmin_crop = tk.StringVar(value=self.state['xmin'])
        self.xmax_crop = tk.StringVar(value=self.state['xmax'])

        # Layout
        self._build_left_panel()
        self._build_right_panel()
        self.update_roi_entries()

    def _init_spec_state(self, spec, roi_n, baseline_type, roi_init=None):
        # Each spectrum has its own parameters for ROIs, type, crop, etc.
        if spec not in self.memory:
            self.memory[spec] = {
                'roi_n': roi_n,
                'roi_values': [list(pair) for pair in (roi_init if roi_init else self._auto_rois(roi_n))],
                'type': baseline_type,
                'xmin': '100',
                'xmax': '1900'
            }
        self.state = self.memory[spec]
        self.current_index = self.spectra_titles.index(spec)
        self.x, self.y = self.spectra_xy_list[self.current_index]

    def on_spectrum_changed(self, event=None):
        # 1. Sauvegarder l’état précédent
        if hasattr(self, "current_index"):
            old_spec = self.spectra_titles[self.current_index]
            self._save_current_entries(old_spec)
        # 2. Mettre à jour l’index courant
        spec = self.spec_var.get()
        self.current_index = self.spectra_titles.index(spec)
        self.x, self.y = self.spectra_xy_list[self.current_index]
        # 3. Restaurer l’état du spectre (si existe), sinon init
        state = self.spec_states.get(spec, None)
        if state is None:
            self.var_nroi.set(6)
            self.roi_values = self._auto_rois(6)
            self.var_type.set("poly5")
            self.xmin_crop.set("100")
            self.xmax_crop.set("1900")
        else:
            self.var_nroi.set(state["roi_n"])
            self.roi_values = state["roi_values"]
            self.var_type.set(state["type"])
            self.xmin_crop.set(state.get("xmin", ""))
            self.xmax_crop.set(state.get("xmax", ""))
        self.update_roi_entries()

    def _save_current_entries(self, spec=None):
        if spec is None:
            spec = self.spectra_titles[self.current_index]
        n = self.var_nroi.get()
        roi_values = []
        for i in range(n):
            try:
                a = float(self.entries_min[i].get())
                b = float(self.entries_max[i].get())
                roi_values.append([a, b])
            except Exception:
                roi_values.append([None, None])
        state = {
            'roi_n': n,
            'roi_values': roi_values,
            'type': self.var_type.get(),
            'xmin': self.xmin_crop.get(),
            'xmax': self.xmax_crop.get()
        }
        self.spec_states[spec] = state

    # ==== UI Layout ====

    def _build_left_panel(self):
        """Build left control panel: ROI, method, actions."""
        left = ttk.Frame(self)
        left.grid(row=1, column=0, padx=10, pady=8, sticky="ns")

        ttk.Label(left, text="Baseline parameters", font=("Arial",13)).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(left, text="Number of ROI").grid(row=1, column=0, sticky="w", pady=(8,0))
        self.var_nroi = tk.IntVar(value=self.state['roi_n'])
        sp = ttk.Spinbox(left, from_=1, to=8, textvariable=self.var_nroi, width=4, command=self.update_roi_entries)
        sp.grid(row=1, column=1, sticky="w")
        sp.bind('<KeyRelease>', lambda e: self.update_roi_entries())

        # Table for ROI min/max values
        self.roi_table = ttk.Frame(left)
        self.roi_table.grid(row=2, column=0, columnspan=2, pady=(6,0))

        # Crop region (Xmin/Xmax)
        crop_row_num = 4
        if not self.simple_mode:
            row_crop = ttk.Frame(left)
            row_crop.grid(row=crop_row_num, column=0, columnspan=2, pady=(7,0), sticky="w")
            ttk.Label(row_crop, text="Crop Xmin:").pack(side="left")
            ttk.Entry(row_crop, width=8, textvariable=self.xmin_crop).pack(side="left", padx=(2,8))
            ttk.Label(row_crop, text="Xmax:").pack(side="left")
            ttk.Entry(row_crop, width=8, textvariable=self.xmax_crop).pack(side="left")
            self.xmin_crop.trace_add("write", lambda *args: self.update_plot())
            self.xmax_crop.trace_add("write", lambda *args: self.update_plot())

        # Baseline type selector
        ttk.Label(left, text="Type of baseline").grid(row=crop_row_num+1, column=0, sticky="w", pady=(10,0))
        self.var_type = tk.StringVar(value=self.state['type'])
        cb_type = ttk.Combobox(
            left,
            values=[label for label, _ in self.BASELINE_TYPES],
            textvariable=tk.StringVar(value=self._label_from_type(self.state['type'])),
            state="readonly", width=16
        )
        cb_type.grid(row=crop_row_num+1, column=1, sticky="w")
        cb_type.bind("<<ComboboxSelected>>", self._update_type_from_label)
        cb_type.bind("<KeyRelease>", lambda e: self._update_type_from_label())
        self.cb_type = cb_type

        # Action buttons (export/save only in full mode)
        btn_row = crop_row_num + 2
        if not self.simple_mode:
            ttk.Button(left, text="Save as new spectrum", command=self.save_new_spectrum).grid(row=btn_row, column=0, columnspan=2, pady=8)
            self.var_export_header = tk.BooleanVar(value=True)
            ttk.Checkbutton(left, text="Include header (for baseline)", variable=self.var_export_header).grid(row=btn_row+1, column=0, columnspan=2, sticky="w")
            ttk.Button(left, text="Save baseline as txt", command=self.save_baseline_as_txt).grid(row=btn_row+2, column=0, columnspan=2, pady=4)
            btn_row += 3

        # Save/Cancel/Reset
        footer = ttk.Frame(left)
        footer.grid(row=btn_row, column=0, columnspan=2, pady=(7,3))
        ttk.Button(footer, text="Save", command=self.save_and_close).pack(side="left", padx=(0,10))
        ttk.Button(footer, text="Cancel", command=self.destroy).pack(side="left")
        ttk.Button(footer, text="Reset ROI", command=self.reset_rois).pack(side="left", padx=(12,0))

    def _build_right_panel(self):
        """Right: plot preview panel."""
        right = ttk.Frame(self)
        right.grid(row=1, column=1, padx=10, pady=8)
        self.fig, self.ax = plt.subplots(figsize=(4.3,3.3), dpi=110)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack()

    # ==== ROI and Baseline Handling ====
    def _auto_rois(self, n):
        default = [
            [110.0, 120.0],
            [160.0, 190.0],
            [815.0, 840.0],
            [1210.0, 1250.0],
            [1620.0, 1650.0],
            [1760.0, 1900.0]
        ]
        return [list(map(float, x)) for x in default[:n]]

    def reset_rois(self):
        n = self.var_nroi.get()
        self.state['roi_values'] = self._auto_rois(n)
        self.update_roi_entries()

    def update_roi_entries(self):
        try:
            n = self.var_nroi.get()
        except Exception:
            n = 6
        if (not hasattr(self, 'roi_values')) or (not isinstance(self.roi_values, list)) or (len(self.roi_values) != n):
            self.roi_values = self._auto_rois(n)
        else:
            # sanity
            fixed = []
            for pair in self.roi_values:
                try:
                    a, b = float(pair[0]), float(pair[1])
                    fixed.append([a, b])
                except Exception:
                    fixed.append([0.0, 0.0])
            self.roi_values = fixed

        # Save current before updating UI
        self._save_current_entries()
        n = self.var_nroi.get()
        roi_values = self.state.get('roi_values', self._auto_rois(n))
        for w in self.roi_table.winfo_children():
            w.destroy()
        self.entries_min = []
        self.entries_max = []
        for i in range(n):
            ttk.Label(self.roi_table, text=f"ROI #{i+1}").grid(row=i, column=0, sticky="e")
            ent_min = ttk.Entry(self.roi_table, width=7)
            ent_max = ttk.Entry(self.roi_table, width=7)
            ent_min.grid(row=i, column=1)
            ent_max.grid(row=i, column=2)
            ent_min.insert(0, str(roi_values[i][0]) if i < len(roi_values) else "0")
            ent_max.insert(0, str(roi_values[i][1]) if i < len(roi_values) else "0")
            ent_min.bind("<KeyRelease>", lambda e: self.update_plot())
            ent_max.bind("<KeyRelease>", lambda e: self.update_plot())
            self.entries_min.append(ent_min)
            self.entries_max.append(ent_max)
        self.update_plot()

    def get_rois(self):
        rois = []
        for ent_min, ent_max in zip(self.entries_min, self.entries_max):
            try:
                a = float(ent_min.get())
                b = float(ent_max.get())
                if a < b:
                    rois.append([a,b])
            except Exception:
                pass
        return np.array(rois)

    def get_crop_indices(self):
        xmin = self.xmin_crop.get()
        xmax = self.xmax_crop.get()
        x = self.x
        mask = np.ones_like(x, dtype=bool)
        try:
            if xmin:
                mask &= (x >= float(xmin))
            if xmax:
                mask &= (x <= float(xmax))
        except Exception:
            pass
        return mask

    # ==== Baseline Type Selection ====
    def _label_from_type(self, type_str):
        for label, value in self.BASELINE_TYPES:
            if value == type_str:
                return label
        return self.BASELINE_TYPES[0][0]

    def _type_from_label(self, label):
        for lbl, value in self.BASELINE_TYPES:
            if lbl == label:
                return value
        return self.BASELINE_TYPES[0][1]

    def _update_type_from_label(self, event=None):
        label = self.cb_type.get()
        self.var_type.set(self._type_from_label(label))
        self.state['type'] = self.var_type.get()
        self.update_plot()

    # ==== Plotting and Preview ====
    def update_plot(self):
        self.ax.clear()
        x, y = (self.x, self.y) if self.simple_mode else (self.x[self.get_crop_indices()], self.y[self.get_crop_indices()])
        if x.size and y.size:
            self.ax.plot(x, y, 'k-', lw=1, label="Raw")

        rois = self.get_rois()
        for xmin, xmax in rois:
            self.ax.axvspan(xmin, xmax, color="purple", alpha=0.2)

        base = None
        method = self.var_type.get()
        try:
            if len(rois) >= 1 and x.size and y.size:
                if method.startswith("poly"):
                    degree = int(method[4:]) if method[4:].isdigit() else 5
                    ycorr, base = rp.baseline(x, y, rois, "poly", polynomial_order=degree)
                elif method == "unispline":
                    ycorr, base = rp.baseline(x, y, rois, "unispline", s=1e0)
                elif method == "als":
                    ycorr, base = rp.baseline(x, y, rois, "als", lam=1e5, p=0.05)
                elif method == "arPLS":
                    ycorr, base = rp.baseline(x, y, rois, "arPLS", lam=1e6, ratio=0.001)
                elif method == "drPLS":
                    ycorr, base = rp.baseline(x, y, rois, "drPLS")
                elif method == "rubberband":
                    ycorr, base = rp.baseline(x, y, rois, "rubberband")
            if base is not None and hasattr(base, 'shape') and len(base) == len(x):
                self.ax.plot(x, base, lw=1.5, label=f"{method} baseline")
                self.ax.legend(fontsize=9)
        except Exception as e:
            self.ax.set_title("Error: "+str(e), fontsize=9)

        self.ax.set_xlabel("Raman shift (cm$^{-1}$)")
        self.ax.set_ylabel("Intensity (a.u.)")
        self.fig.tight_layout()
        self.canvas.draw()

    # ==== Export / Save ====

    def save_baseline_as_txt(self):
        rois = self.get_rois()
        method = self.var_type.get()
        mask = self.get_crop_indices()
        x, y = self.x[mask], self.y[mask]
        if len(rois) < 1:
            messagebox.showerror("Error", "At least one ROI required.")
            return
        try:
            if method.startswith("poly"):
                degree = int(method[4:]) if method[4:].isdigit() else 5
                _, base = rp.baseline(x, y, rois, "poly", polynomial_order=degree)
            elif method == "unispline":
                _, base = rp.baseline(x, y, rois, "unispline", s=1e0)
            elif method == "als":
                _, base = rp.baseline(x, y, rois, "als", lam=1e5, p=0.05)
            elif method == "arPLS":
                _, base = rp.baseline(x, y, rois, "arPLS", lam=1e6, ratio=0.001)
            elif method == "drPLS":
                _, base = rp.baseline(x, y, rois, "drPLS")
            elif method == "rubberband":
                _, base = rp.baseline(x, y, rois, "rubberband")
            else:
                messagebox.showerror("Error", f"Unknown method: {method}")
                return
        except Exception as e:
            messagebox.showerror("Error", f"Failed to compute baseline:\n{e}")
            return

        filename = filedialog.asksaveasfilename(defaultextension=".txt",
                                                filetypes=[("Text", "*.txt"), ("All files", "*.*")],
                                                title="Save baseline as txt")
        if not filename:
            return
        arr = np.column_stack((x, base))
        header = ""
        if getattr(self, "var_export_header", None) and self.var_export_header.get():
            header += f"# Baseline method: {method}\n"
            header += "# ROIs: " + json.dumps(rois.tolist()) + "\n"
            header += f"# X range: [{x.min():.3f}, {x.max():.3f}]\n"
        try:
            np.savetxt(filename, arr, delimiter="\t", header=header, comments="")
            messagebox.showinfo("Export", f"Baseline saved:\n{filename}", parent=self)
        except Exception as e:
            messagebox.showerror("Export", f"Failed to save baseline:\n{e}", parent=self)

    def save_new_spectrum(self):
        rois = self.get_rois()
        method = self.var_type.get()
        mask = self.get_crop_indices()
        x, y = self.x[mask], self.y[mask]
        if len(rois) < 1:
            messagebox.showerror("Error", "At least one ROI required.")
            return
        try:
            if method.startswith("poly"):
                degree = int(method[4:]) if method[4:].isdigit() else 5
                ycorr, _ = rp.baseline(x, y, rois, "poly", polynomial_order=degree)
            elif method == "unispline":
                ycorr, _ = rp.baseline(x, y, rois, "unispline", s=1e0)
            elif method == "als":
                ycorr, _ = rp.baseline(x, y, rois, "als", lam=1e5, p=0.05)
            elif method == "arPLS":
                ycorr, _ = rp.baseline(x, y, rois, "arPLS", lam=1e6, ratio=0.001)
            elif method == "drPLS":
                ycorr, _ = rp.baseline(x, y, rois, "drPLS")
            elif method == "rubberband":
                ycorr, _ = rp.baseline(x, y, rois, "rubberband")
            else:
                messagebox.showerror("Error", f"Unknown method: {method}")
                return
        except Exception as e:
            messagebox.showerror("Error", f"Failed to compute baseline:\n{e}")
            return

        filename = filedialog.asksaveasfilename(defaultextension=".txt",
                        filetypes=[("Text", "*.txt"), ("All files", "*.*")],
                        title="Save corrected spectrum as...")
        if not filename:
            return
        arr = np.column_stack((x, ycorr))
        try:
            np.savetxt(filename, arr, delimiter="\t")
        except Exception as e:
            messagebox.showerror("Export", f"Failed to save spectrum:\n{e}", parent=self)
            return
        messagebox.showinfo("Export", f"Baseline-corrected spectrum saved:\n{filename}", parent=self)
        if self.file_callback:
            try:
                self.file_callback(filename, status="baseline")
            except TypeError:
                self.file_callback(filename)

    # ==== Save and close ====

    def save_and_close(self):
        _sync_state_to_memory(self)
        self._save_current_entries()
        rois = self.get_rois()
        method = self.var_type.get()
        if self.callback:
            self.callback(
                rois, method, self.xmin_crop.get(), self.xmax_crop.get(), self.spec_var.get()
            )
        self.destroy()

#########################################################################
# 3.6 SingleFitWindow: Fit processing and chi2 calculation
#########################################################################

def _sync_state_to_memory(self):
    """Keep spec_states and memory consistent (minimal fix)."""
    try:
        if hasattr(self, "spec_states") and isinstance(self.spec_states, dict):
            for key, val in self.spec_states.items():
                self.memory[key] = val
    except Exception:
        pass

class SingleFitWindow(tk.Toplevel):
    """
    SingleFitWindow
    ----------------
    GUI window to fit a single Raman spectrum with two modes:
      - "Classic": one-shot Levenberg-Marquardt fit (leastsq).
      - "Origin-like": stepwise blocks (LM) with optional soft-penalty and UI-driven stepping until convergence.
    """
    COLORS = ["black", "red", "seagreen", "royalblue", "orange", "purple", "brown", "indigo"]

    def __init__(self, master, file_paths, file_titles, fit_param_memory, xy_cache=None):
        super().__init__(master)
        self.title("Single fit")
        self.geometry("1200x750")

        # Data & parameter stores
        self.file_paths = file_paths
        self.file_titles = file_titles
        self.fit_param_memory = fit_param_memory
        self._last_snapshot_by_spec = {}
        self.xy_cache = xy_cache or {}

        # State for latest fit/plot
        self._current_fit = None         # lmfit MinimizerResult
        self._current_yfit = None        # composite model (sum of components)
        self._current_peaks = None       # list of individual peak arrays
        self._current_x = None
        self._current_y = None

        # ===== Left panel =====
        frame_left = ttk.Frame(self, padding=10)
        frame_left.grid(row=0, column=0, sticky="nsw")

        ttk.Label(frame_left, text="Select from imported").pack(anchor="w")
        self.spec_var = tk.StringVar(value=file_titles[0])
        self.combo = ttk.Combobox(frame_left, textvariable=self.spec_var,
                                  values=file_titles, state="readonly", width=25)
        self.combo.pack(anchor="w", pady=(0, 6))
        self.combo.bind("<<ComboboxSelected>>", self.on_spec_change)

        self.norm_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame_left, text="Norm.", variable=self.norm_var,
                        command=self.update_plot).pack(anchor="w", pady=(2, 4))

        # Plot X range (view window)
        frm_plot = ttk.Frame(frame_left); frm_plot.pack(anchor="w", pady=(3, 0))
        ttk.Label(frm_plot, text="Plot X").grid(row=0, column=0)
        self.xmin_var = tk.StringVar(value="")
        self.xmax_var = tk.StringVar(value="")
        ttk.Entry(frm_plot, textvariable=self.xmin_var, width=6).grid(row=1, column=0, padx=(0, 2))
        ttk.Entry(frm_plot, textvariable=self.xmax_var, width=6).grid(row=1, column=1, padx=(0, 2))
        self.xmin_var.trace_add("write", lambda *_: self.update_plot())
        self.xmax_var.trace_add("write", lambda *_: self.update_plot())

        # Parameters & Fit buttons
        ttk.Button(frame_left, text="Fit param.", command=self.open_param_window).pack(anchor="w", pady=(10, 4))
        ttk.Button(frame_left, text="Fit !", command=self.run_fit).pack(anchor="w", pady=(4, 8))
        ttk.Button(frame_left, text="Reset params to snapshot", command=self.reset_params_to_snapshot)\
           .pack(anchor="w", pady=(0, 8))

        # Fit mode selector
        frm_mode = ttk.Frame(frame_left); frm_mode.pack(anchor="w", pady=(2, 2))
        ttk.Label(frm_mode, text="Fit mode:").pack(side="left")
        self.fit_mode = tk.StringVar(value="classic")  # "classic" | "origin"
        ttk.Radiobutton(frm_mode, text="Classic (one-shot LM)", variable=self.fit_mode, value="classic",
                        command=self._toggle_origin_controls).pack(side="left", padx=(6, 0))
        ttk.Radiobutton(frm_mode, text="Origin-like (stepwise)", variable=self.fit_mode, value="origin",
                        command=self._toggle_origin_controls).pack(side="left", padx=(6, 0))

        # Origin-like panel (hidden by default)
        self.frm_origin = ttk.Frame(frame_left)

        row1 = ttk.Frame(self.frm_origin); row1.pack(anchor="w", pady=(6, 2))
        ttk.Label(row1, text="Δχ² tol:").pack(side="left")
        self.origin_tol = tk.StringVar(value="1e-6")
        ttk.Entry(row1, textvariable=self.origin_tol, width=10).pack(side="left", padx=(4, 10))

        row2 = ttk.Frame(self.frm_origin); row2.pack(anchor="w", pady=(4, 2))
        ttk.Label(row2, text="Step iters:").pack(side="left")
        ttk.Button(row2, text="1",  width=3, command=lambda: self.run_fit_origin_stepwise(1)).pack(side="left", padx=2)
        ttk.Button(row2, text="2",  width=3, command=lambda: self.run_fit_origin_stepwise(2)).pack(side="left", padx=2)
        ttk.Button(row2, text="5",  width=3, command=lambda: self.run_fit_origin_stepwise(5)).pack(side="left", padx=2)
        ttk.Button(row2, text="10", width=3, command=lambda: self.run_fit_origin_stepwise(10)).pack(side="left", padx=2)

        ttk.Button(self.frm_origin, text="Fit full (until converge)",
                   command=self.run_fit_origin_full).pack(anchor="w", pady=(6, 2))

        row_alpha = ttk.Frame(self.frm_origin); row_alpha.pack(anchor="w", pady=(2, 2))
        ttk.Label(row_alpha, text="Step blend α:").pack(side="left")
        self.origin_alpha = tk.StringVar(value="0.25")
        ttk.Entry(row_alpha, textvariable=self.origin_alpha, width=6).pack(side="left", padx=(4, 0))

        self.frm_origin.pack_forget()  # hidden initially

        # ===== Main plot panel =====
        frame_main = ttk.Frame(self, padding=4)
        frame_main.grid(row=0, column=1, sticky="nsew")
        frame_main.columnconfigure(0, weight=1)
        frame_main.rowconfigure(1, weight=1)

        self.fig = plt.Figure(figsize=(6, 4.5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame_main)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")

        # Axes titles
        frm_axes = ttk.Frame(frame_main); frm_axes.grid(row=2, column=0, pady=(7, 2))
        self.x_title = tk.StringVar(value="Raman Shift (cm$^{-1}$)")
        self.y_title = tk.StringVar(value="Intensity (a.u.)")
        ttk.Label(frm_axes, text="X title").pack(side="left")
        ttk.Entry(frm_axes, textvariable=self.x_title, width=18).pack(side="left", padx=4)
        ttk.Label(frm_axes, text="Y title").pack(side="left")
        ttk.Entry(frm_axes, textvariable=self.y_title, width=18).pack(side="left", padx=4)
        self.x_title.trace_add("write", lambda *_: self.update_plot())
        self.y_title.trace_add("write", lambda *_: self.update_plot())

        # Chi-square + Report + Export
        frm_chi2 = ttk.Frame(frame_main); frm_chi2.grid(row=3, column=0, pady=(6, 2), sticky="w")
        ttk.Label(frm_chi2, text="Chi²:", font=("Arial", 11, "bold")).pack(side="left")
        self.chi2_label = tk.Label(frm_chi2, text=" -- ", font=("Consolas", 11, "bold"),
                                   bg="#662c91", fg="white", width=8, anchor="center")
        self.chi2_label.pack(side="left", padx=6)
        ttk.Button(frm_chi2, text="Generate report", command=self.generate_report)\
           .pack(side="left", padx=(15, 8))

        export_bar = ttk.Frame(frm_chi2)
        export_bar.pack(side="left", padx=(10, 0))
        ttk.Label(export_bar, text="|").pack(side="left", padx=(0, 6))
        self.var_export_png = tk.BooleanVar(value=True)
        self.var_export_svg = tk.BooleanVar(value=False)
        self.var_export_pdf = tk.BooleanVar(value=False)
        ttk.Checkbutton(export_bar, text="PNG", variable=self.var_export_png).pack(side="left", padx=(0, 4))
        ttk.Checkbutton(export_bar, text="SVG", variable=self.var_export_svg).pack(side="left", padx=(0, 4))
        ttk.Checkbutton(export_bar, text="PDF", variable=self.var_export_pdf).pack(side="left", padx=(0, 8))
        ttk.Button(export_bar, text="Export plot", command=self.export_plot).pack(side="left")

        # ===== Global log panel =====
        self.rowconfigure(1, weight=0)
        frame_log = ttk.Frame(self, padding=(10, 4))
        frame_log.grid(row=1, column=0, columnspan=2, sticky="nsew")
        ttk.Label(frame_log, text="Log").pack(anchor="w")
        self.txt_global_log = tk.Text(frame_log, height=7)
        self.txt_global_log.pack(fill="both", expand=True)

        # First draw
        self.update_plot()

    # --------------------------
    # Helpers: data & parameters
    # --------------------------
    def get_current_spec_idx(self):
        return self.file_titles.index(self.spec_var.get())

    def get_current_spec_path(self):
        return self.file_paths[self.get_current_spec_idx()]

    def get_current_params(self):
        """Return the CURRENT (live) parameter structure for the selected spectrum."""
        return self.fit_param_memory.get(self.spec_var.get(), None)

    def reset_params_to_snapshot(self):
        spec = self.spec_var.get()
        snap = self._last_snapshot_by_spec.get(spec)
        if snap is None:
            messagebox.showinfo("Reset", "No pre-fit snapshot found for this spectrum yet.")
            return
        self.fit_param_memory[spec] = copy.deepcopy(snap)
        self._append_log(f"[Reset] Restored last pre-fit snapshot for '{spec}'.")
        self.update_plot()

    def _snapshot_current_params(self):
        spec = self.spec_var.get()
        cur = self.fit_param_memory.get(spec)
        if cur is not None:
            self._last_snapshot_by_spec[spec] = copy.deepcopy(cur)
            self._append_log(f"[Snapshot] Saved params for '{spec}' before fit.")

    def get_xy(self):
        """Charge X/Y sans aucun popup; utilise le cache si possible."""
        path = self.get_current_spec_path()

        rec = self.xy_cache.get(path)
        if rec is None:
            # Pas dans le cache ? on charge en no-prompt et on stocke
            payload = load_file_as_xy_noprompt(path)
            rec = {
                "kind": _map_parser_to_simple_kind((payload.meta or {}).get("selected_parser", "generic_xy")),
                "x": payload.x, "y": payload.y,
                "df": payload.df, "meta": payload.meta,
                "x_col": payload.x_col, "y_col": payload.y_col
            }
            self.xy_cache[path] = rec

        x = np.asarray(rec["x"], dtype=float)
        y = np.asarray(rec["y"], dtype=float)

        # Plot window (facultatif)
        try:
            xmin = float(self.xmin_var.get()); xmax = float(self.xmax_var.get())
            mask = (x >= xmin) & (x <= xmax)
            x, y = x[mask], y[mask]
        except Exception:
            pass

        # Normalisation (aire→100) si cochée
        if self.norm_var.get() and len(x) and len(y):
            area = float(np.trapz(y, x))
            if abs(area) > 1e-10:
                y = y / area * 100.0

        return x, y

    def _relax_params(self, old_params, new_params, alpha=0.25):
        """Blend parameter values: old <- old + alpha * (new - old)."""
        blended = old_params.copy()
        for name, par in blended.items():
            if name in new_params and par.vary:
                try:
                    old_v = float(par.value)
                    new_v = float(new_params[name].value)
                    par.set(value=old_v + alpha * (new_v - old_v))
                except Exception:
                    pass
        return blended

    # --------------------------
    # UI callbacks
    # --------------------------
    def _toggle_origin_controls(self):
        if self.fit_mode.get() == "origin":
            self.frm_origin.pack(anchor="w", pady=(6, 8))
        else:
            self.frm_origin.pack_forget()

    def on_spec_change(self, event=None):
        self.update_plot()

    def open_param_window(self):
        spec = self.spec_var.get()
        FitParamWindow(self, self.file_titles, self.fit_param_memory,
                       current_spectrum=spec, callback=self.on_param_saved)

    def on_param_saved(self, mem):
        self.fit_param_memory = mem
        self.update_plot()

    def _append_log(self, msg: str):
        try:
            self.txt_global_log.insert("end", msg.rstrip() + "\n")
            self.txt_global_log.see("end")
        except Exception:
            pass

    # --------------------------
    # Plotting
    # --------------------------
    def update_plot(self):
        x, y = self.get_xy()
        self.ax.clear()
        self.ax.plot(x, y, color="black", lw=1.2, label="Data")

        params_struct = self.get_current_params()
        show_components = params_struct is not None and len(params_struct) > 0

        fit_result = None
        peaks = None

        if show_components:
            lm_params = self.build_lmfit_parameters(params_struct)
            y_fit, peaks = self.compute_model(x, lm_params, params_struct)
            fit_result = y_fit
            self._current_yfit = y_fit
            self._current_peaks = peaks
        else:
            self._current_yfit = None
            self._current_peaks = None

        if fit_result is not None:
            self.ax.plot(x, fit_result, color="red", lw=2, label="Fit")
            for i, pk in enumerate(peaks):
                self.ax.plot(x, pk, lw=1.3, color=self.COLORS[(i + 2) % len(self.COLORS)], alpha=0.7)
            chi2 = self.compute_chi2(y, fit_result, lm_params)
            self.chi2_label.config(text=f"{chi2:.3g}")
        else:
            self.chi2_label.config(text="--")

        self.ax.set_xlabel(self.x_title.get())
        self.ax.set_ylabel(self.y_title.get())
        self.ax.set_title("Fit preview")
        self.ax.legend(fontsize=8)
        self.fig.tight_layout()
        self.canvas.draw()
        self._current_x, self._current_y = x, y

    def compute_model(self, x, lm_params, params_struct):
        total = np.zeros_like(x)
        peaks = []
        for i, d in enumerate(params_struct):
            a = lm_params[f"a{i}"].value
            f = lm_params[f"f{i}"].value
            l = lm_params[f"l{i}"].value
            if d.get("shape", "G") == "G":
                pk = rp.gaussian(x, a, f, l)
            else:
                eta = lm_params[f"eta{i}"].value if f"eta{i}" in lm_params else 0.5
                pk = rp.pseudovoigt(x, a, f, l, eta)
            peaks.append(pk)
            total += pk
        return total, peaks

    # --------------------------
    # Parameter building (robust)
    # --------------------------
    def build_lmfit_parameters(self, params_struct):
        p = lmfit.Parameters()
        eps = 1e-9

        for i, d in enumerate(params_struct):
            # ---- Amplitude ----
            fit_amp = bool(d.get("fit_amp", True))
            a0 = d.get("amp_val", None)
            try:
                a0 = float(a0)
            except Exception:
                a0 = None
            if a0 is None or a0 <= 0.0:
                a0 = 1.0
            p.add(f"a{i}", value=a0, min=0.0, vary=fit_amp)

            # ---- Center (shift) ----
            fmin = float(d["shift_min"]); fmax = float(d["shift_max"])
            fval = float(d["shift_val"])
            if fmin > fmax:
                fmin, fmax = fmax, fmin
            if fval <= fmin: fval = fmin + eps
            if fval >= fmax: fval = fmax - eps
            p.add(f"f{i}", value=fval, min=fmin, max=fmax, vary=bool(d.get("fit_shift", True)))

            # ---- FWHM ----
            lmin = float(d.get("fwhm_min", 1e-9))
            lmax = float(d.get("fwhm_max", max(lmin*1.000001, 1e-6)))
            lval = float(d.get("fwhm_val", max(lmin*1.0005, 1.0)))

            if lmin > lmax:
                lmin, lmax = lmax, lmin

            # garantir min < max (lmfit l'exige)
            if abs(lmax - lmin) < 1e-12:
                lmax = lmin + 1e-6

            # ne pas démarrer sur une borne
            eps = 1e-9
            if lval <= lmin: lval = lmin + eps
            if lval >= lmax: lval = lmax - eps

            p.add(f"l{i}", value=lval, min=lmin, max=lmax, vary=bool(d.get("fit_fwhm", True)))

            # ---- Pseudo-Voigt (GL): eta in [0,1] ----
            if d.get("shape", "G") == "GL":
                try:
                    eta_min = float(d.get("eta_min", 0.0))
                    eta_max = float(d.get("eta_max", 1.0))
                except Exception:
                    eta_min, eta_max = 0.0, 1.0
                eta_min = max(0.0, min(eta_min, 1.0))
                eta_max = max(0.0, min(eta_max, 1.0))
                if eta_min >= eta_max:
                    eta_min, eta_max = 0.0, 1.0
                try:
                    eta_val = float(d.get("eta_val", 0.5))
                except Exception:
                    eta_val = 0.5
                if not (eta_min < eta_val < eta_max):
                    eta_val = 0.5
                p.add(f"eta{i}", value=eta_val, min=eta_min, max=eta_max, vary=bool(d.get("fit_eta", True)))

        return p

    # --------------------------
    # Classic fit (one-shot LM)
    # --------------------------
    def run_fit(self):
        self._snapshot_current_params()
        x, y = self.get_xy()
        params_struct = self.get_current_params()
        if params_struct is None:
            messagebox.showwarning("No parameters", "No fit parameters found for this spectrum.")
            return

        lm_params = self.build_lmfit_parameters(params_struct)

        try:
            result, y_fit, peaks = self.fit_with_params(x, y, lm_params, params_struct)
            self._current_fit = result
            self._current_yfit = y_fit
            self._current_peaks = peaks
            self._current_x = x
            self._current_y = y

            # Draw
            self.ax.clear()
            self.ax.plot(x, y, color="black", lw=1.2, label="Data")
            self.ax.plot(x, y_fit, color="red", lw=2, label="Fit")
            for i, pk in enumerate(peaks):
                self.ax.plot(x, pk, lw=1.3, color=self.COLORS[(i + 2) % len(self.COLORS)], alpha=0.7)
            self.ax.set_xlabel(self.x_title.get()); self.ax.set_ylabel(self.y_title.get())
            self.ax.set_title("Fit preview")
            self.ax.legend(fontsize=8)
            self.fig.tight_layout()
            self.canvas.draw()

            # Reduced chi² from result params
            chi2 = self.compute_chi2(y, y_fit, result.params)
            self.chi2_label.config(text=f"{chi2:.3g}")
            self._append_log(f"[Classic] chi2_red={chi2:.6g} spec='{self.spec_var.get()}'")

            # Write back
            self._writeback_params_from_result(result, params_struct)

        except Exception as e:
            self.chi2_label.config(text="FAIL")
            messagebox.showerror("Fit error", str(e))

    def fit_with_params(self, x, y, lm_params, params_struct):
        import numpy as np

        # Keep only finite points
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]; y = y[mask]

        def model_func(params, x):
            total = np.zeros_like(x)
            peaks = []
            for i, d in enumerate(params_struct):
                a = params[f"a{i}"].value
                f = params[f"f{i}"].value
                l = params[f"l{i}"].value
                if d.get("shape", "G") == "G":
                    pk = rp.gaussian(x, a, f, l)
                else:
                    eta = params[f"eta{i}"].value if f"eta{i}" in params else 0.5
                    pk = rp.pseudovoigt(x, a, f, l, eta)
                peaks.append(pk)
                total += pk
            return total, peaks

        def residual(params, x, y):
            model, _ = model_func(params, x)
            return model - y

        result = lmfit.minimize(residual, lm_params, args=(x, y), method='leastsq',
                                ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=10000)
        y_fit, peaks = model_func(result.params, x)
        return result, y_fit, peaks

    def compute_chi2(self, y, y_fit, lm_params):
        resid = y - y_fit
        n = len(y)
        p = sum([param.vary for param in lm_params.values()])
        dof = max(n - p, 1)
        chi2 = float(np.sum(resid ** 2) / dof)
        return chi2

    # --------------------------
    # Origin-like (stepwise) mode
    # --------------------------
    def _origin_model_func(self, params, x, params_struct):
        total = np.zeros_like(x)
        peaks = []
        for i, d in enumerate(params_struct):
            a = params[f"a{i}"].value
            f = params[f"f{i}"].value
            l = params[f"l{i}"].value
            if d.get("shape", "G") == "G":
                pk = rp.gaussian(x, a, f, l)
            else:
                eta = params[f"eta{i}"].value if f"eta{i}" in params else 0.5
                pk = rp.pseudovoigt(x, a, f, l, eta)
            peaks.append(pk)
            total += pk
        return total, peaks

    def _origin_residual(self, params, x, y, params_struct, soft_penalty=False):
        model, _ = self._origin_model_func(params, x, params_struct)
        res = model - y
        if soft_penalty:
            pen = []
            for name, par in params.items():
                if not par.vary:
                    continue
                if (par.min is not None) and (par.value < par.min):
                    pen.append((par.min - par.value) * 1e4)
                if (par.max is not None) and (par.value > par.max):
                    pen.append((par.value - par.max) * 1e4)
            if pen:
                res = np.r_[res, np.array(pen)]
        return res

    def _origin_common(self):
        x, y = self.get_xy()
        params_struct = self.get_current_params()
        if params_struct is None:
            raise RuntimeError("No fit parameters defined for this spectrum.")
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]; y = y[mask]
        lm_params = self.build_lmfit_parameters(params_struct)
        return x, y, params_struct, lm_params

    def _append_origin_log(self, tag, step, chisq, params, chi2_red=None, drel=None):
        parts = [f"[Origin {tag}]"]
        if step is not None: parts.append(f"step={step}")
        parts.append(f"chisq={chisq:.6g}")
        if chi2_red is not None: parts.append(f"chi2_red={chi2_red:.6g}")
        if drel is not None: parts.append(f"Δrel={drel:.3g}")
        self._append_log(" ".join(parts))

    def run_fit_origin_stepwise(self, step_iters=1):
        if self.fit_mode.get() != "origin":
            messagebox.showinfo("Info", "Select 'Origin-like' mode to use stepwise buttons.")
            return

        self._snapshot_current_params()

        try:
            tol = float(self.origin_tol.get())
        except Exception:
            tol = 1e-6

        x, y, params_struct, lm_params = self._origin_common()

        method = 'leastsq'
        fit_kws = dict(ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=100)
        soft_penalty = False

        minimizer = lmfit.Minimizer(
            self._origin_residual,
            lm_params,
            fcn_args=(x, y, params_struct),
            fcn_kws={'soft_penalty': soft_penalty},
        )

        prev_chisq = getattr(self._current_fit, "chisqr", None)
        last_result = None

        try:
            alpha = float(self.origin_alpha.get())
        except Exception:
            alpha = 0.25

        for s in range(step_iters):
            result = minimizer.minimize(method=method, **fit_kws)
            last_result = result
            relaxed = self._relax_params(minimizer.params, result.params, alpha=alpha)
            minimizer.params = relaxed

            y_fit, peaks = self._origin_model_func(relaxed, x, params_struct)
            try:
                chi2_red = self.compute_chi2(y, y_fit, relaxed)
            except Exception:
                chi2_red = float('nan')

            chisq = result.chisqr
            rel_drop = None if prev_chisq is None else (prev_chisq - chisq) / max(prev_chisq, 1e-30)

            self._append_origin_log("step", s + 1, chisq, relaxed, chi2_red=chi2_red, drel=rel_drop)

            self._current_fit = result
            self._current_yfit = y_fit
            self._current_peaks = peaks
            self._current_x = x
            self._current_y = y

            self.ax.clear()
            self.ax.plot(x, y, color="black", lw=1.2, label="Data")
            self.ax.plot(x, y_fit, color="red", lw=2, label="Fit")
            for i, pk in enumerate(peaks):
                self.ax.plot(x, pk, lw=1.3, color=self.COLORS[(i + 2) % len(self.COLORS)], alpha=0.7)
            self.ax.set_xlabel(self.x_title.get()); self.ax.set_ylabel(self.y_title.get())
            self.ax.set_title("Fit preview (Origin-like)")
            self.ax.legend(fontsize=8)
            self.canvas.draw()
            self.chi2_label.config(text=f"{chi2_red:.3g}")

            if rel_drop is not None and rel_drop < tol:
                self._append_origin_log("converge", None, chisq, result.params, chi2_red=chi2_red, drel=rel_drop)
                break

            prev_chisq = chisq

        if last_result is not None:
            self._writeback_params_from_result(last_result, params_struct)

    def run_fit_origin_full(self):
        if self.fit_mode.get() != "origin":
            messagebox.showinfo("Info", "Select 'Origin-like' mode first.")
            return

        self._snapshot_current_params()

        try:
            tol = float(self.origin_tol.get())
        except Exception:
            tol = 1e-6

        for b in range(100):
            before = None if self._current_fit is None else self._current_fit.chisqr
            self.run_fit_origin_stepwise(10)
            after = None if self._current_fit is None else self._current_fit.chisqr
            if before is not None and after is not None:
                rel_drop = (before - after) / max(before, 1e-30)
                self._append_origin_log("block", b + 1, after, self._current_fit.params, drel=rel_drop)
                if rel_drop < tol:
                    break

    # --------------------------
    # Write-back & Export/Report
    # --------------------------
    def _writeback_params_from_result(self, result, params_struct):
        spec = self.spec_var.get()
        for i, d in enumerate(params_struct):
            d["shift_val"] = float(result.params[f"f{i}"].value)
            d["fwhm_val"]  = float(result.params[f"l{i}"].value)
            if f"a{i}" in result.params:
                d["amp_val"] = float(result.params[f"a{i}"].value)
            if d.get("shape", "G") == "GL" and f"eta{i}" in result.params:
                d["eta_val"] = float(result.params[f"eta{i}"].value)
        self.fit_param_memory[spec] = params_struct

    def export_plot(self):
        from tkinter import filedialog
        basename = filedialog.asksaveasfilename(defaultextension=".png",
                        filetypes=[("PNG", "*.png"), ("SVG", "*.svg"), ("PDF", "*.pdf"), ("All files", "*.*")],
                        title="Export plot as...")
        if not basename:
            return
        basename = os.path.splitext(basename)[0]
        errors = []
        self.canvas.draw()
        if self.var_export_png.get():
            try:
                self.fig.savefig(basename + ".png")
            except Exception as e:
                errors.append(f"PNG: {e}")
        if self.var_export_svg.get():
            try:
                self.fig.savefig(basename + ".svg")
            except Exception as e:
                errors.append(f"SVG: {e}")
        if self.var_export_pdf.get():
            try:
                self.fig.savefig(basename + ".pdf")
            except Exception as e:
                errors.append(f"PDF: {e}")
        if errors:
            messagebox.showerror("Export", "Some exports failed:\n" + "\n".join(errors))
        else:
            messagebox.showinfo("Export", "Plot exported successfully.")

    def generate_report(self):
        from tkinter import filedialog
        x = self._current_x
        y = self._current_y
        params_struct = self.get_current_params()
        y_fit = self._current_yfit
        peaks = self._current_peaks

        if x is None or y is None or params_struct is None or peaks is None:
            messagebox.showinfo("Report", "No fit available. Run fit or set parameters first.")
            return

        areas = [np.trapz(pk, x) for pk in peaks]

        try:
            lm_params_for_chi2 = self._current_fit.params if self._current_fit else self.build_lmfit_parameters(params_struct)
            chi2 = self.compute_chi2(y, y_fit, lm_params_for_chi2)
        except Exception:
            chi2 = np.nan

        now = datetime.datetime.now()
        datestr = now.strftime("%Y%m%d-%H%M%S")
        specname = self.spec_var.get()
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            initialfile=f"{specname}_fit{datestr}.txt",
            filetypes=[("Text", "*.txt"), ("All files", "*.*")]
        )
        if not filename:
            return

        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"# Raman Fit Report\n")
            f.write(f"# Spectrum: {specname}\n")
            f.write(f"# Date: {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Chi²: {chi2:.5g}\n")
            f.write("#\n")
            f.write("# Component\tCenter\tFWHM\tAmplitude\tArea\tShape\tEta\n")
            for i, d in enumerate(params_struct):
                center = d["shift_val"]
                fwhm = d["fwhm_val"]
                try:
                    amp = float(self._current_fit.params[f"a{i}"].value) if self._current_fit else float(d.get("amp_val", 1.0))
                except Exception:
                    amp = "--"
                area = areas[i] if i < len(areas) else "--"
                shape = d.get("shape", "G")
                eta = "--"
                if shape == "GL":
                    try:
                        eta = float(self._current_fit.params[f"eta{i}"].value)
                    except Exception:
                        eta = float(d.get("eta_val", 0.5)) if "eta_val" in d else "--"
                f.write(f"{i+1}\t{center:.2f}\t{fwhm:.2f}\t{amp}\t{area:.2f}\t{shape}\t{eta}\n")
            f.write("#\n# End of report\n")

        messagebox.showinfo("Report", f"Report exported to:\n{filename}")

# =====================================================================
# 4. Main loop (application entry point)
# =====================================================================

def main():
    # Fenêtre racine
    try:
        root = ThemedTk(theme="arc")
    except Exception:
        root = tk.Tk()  # fallback si ttkthemes n'est pas dispo

    app = RamanApp(root)

    # Menu: File -> Simple plot / Exit
    menubar = tk.Menu(root)
    filemenu = tk.Menu(menubar, tearoff=0)
    filemenu.add_command(label="Quick import", command=app.import_files_quick)
    filemenu.add_command(label="Custom import", command=app.import_files_custom)
    filemenu.add_separator()
    filemenu.add_command(label="Simple plot\tCtrl+P", command=app.simple_plot)
    filemenu.add_separator()
    filemenu.add_command(label="Exit", command=app.exit_app)
    menubar.add_cascade(label="File", menu=filemenu)
    root.config(menu=menubar)

    # Raccourcis clavier (patched: Ctrl+O -> Custom import)
    root.bind("<Control-p>", lambda e: app.simple_plot())
    root.bind("<Control-o>", lambda e: app.import_files_custom())
    root.bind("<Control-q>", lambda e: app.exit_app())

    # Quit propre si on ferme la croix
    root.protocol("WM_DELETE_WINDOW", app.exit_app)
    root.mainloop()


if __name__ == "__main__":
    main()
