# ui_fit_params.py
# --------------------------------------------------------------------------------------
# Standalone FitParamWindow (V40)
# - No dependency on main.py globals.
# - You pass the model_dir path in __init__.
# - Self-contained JSON save/load with overwrite prompts.
# --------------------------------------------------------------------------------------

from __future__ import annotations

import os
import json
import copy
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from typing import Dict, List, Optional


def _apply_modern_style(widget):
    """Apply a futuristic yet sober style to ttk widgets and return palette."""
    palette = {
        "bg": "#0c1427",
        "card": "#111d33",
        "accent": "#64e7ff",
        "accent_alt": "#d08bff",
        "muted": "#9db2ce",
    }
    style = ttk.Style(widget)
    try:
        style.theme_use("clam")
    except Exception:
        pass
    style.configure(".", background=palette["bg"], foreground="#e8edf7", fieldbackground=palette["card"])
    style.configure("Card.TFrame", background=palette["card"])
    style.configure("Section.TLabel", background=palette["bg"], foreground="#e8edf7", font=("Segoe UI", 11, "bold"))
    style.configure("Card.TLabel", background=palette["card"], foreground="#e8edf7")

    def _btn(name, color):
        style.configure(name, background=color, foreground="#0c1427", padding=(10, 7), borderwidth=0)
        style.map(name, background=[("active", color)])

    _btn("Primary.TButton", palette["accent"])
    _btn("Alt.TButton", palette["accent_alt"])
    try:
        widget.configure(bg=palette["bg"])
    except Exception:
        pass
    return palette


def _ensure_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def _list_model_files(model_dir: str) -> List[str]:
    """Return model names (filenames without .json) present in model_dir."""
    if not os.path.isdir(model_dir):
        return []
    out = []
    for f in os.listdir(model_dir):
        if f.lower().endswith(".json"):
            out.append(f[:-5])
    out.sort()
    return out


def _save_json_with_prompt(parent, filepath: str, data) -> bool:
    """Save JSON to filepath with overwrite prompt; returns True on success."""
    if os.path.exists(filepath):
        if not messagebox.askyesno(
            "Overwrite?",
            f"File '{os.path.basename(filepath)}' already exists.\nOverwrite?",
            parent=parent,
        ):
            return False
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        messagebox.showerror("Save error", f"Error saving file:\n{e}", parent=parent)
        return False


class FitParamWindow(tk.Toplevel):
    """
    Toplevel window to edit, save, and load fit parameters for a spectrum.

    Users can configure multiple components (Gaussian or pseudo-Voigt "GL"):
      - Center (Raman shift) with min/value/max and 'vary' checkbox
      - FWHM with min/value/max and 'vary' checkbox
      - Shape: G (Gaussian) or GL (pseudo-Voigt)
      - Eta range/min/value/max + 'vary' (only meaningful for GL; ignored for G)
      - Amplitude 'Amp (value)' + 'vary' checkbox

    Parameters are stored per spectrum inside the provided memory dict:
      memory[spectrum_name] = [
        {
          "shift_min": float, "shift_val": float, "shift_max": float, "fit_shift": bool,
          "fwhm_min": float,  "fwhm_val": float,  "fwhm_max": float,  "fit_fwhm": bool,
          "eta_min": float,   "eta_val": float,   "eta_max": float,   "fit_eta": bool,
          "shape": "G" | "GL",
          "amp_val": float, "fit_amp": bool
        },
        ...
      ]

    NOTE:
      - 'eta_*' is ignored at runtime if shape == "G".
      - 'amp_val' will be used as initial amplitude; if 'fit_amp' is False, it will be fixed.
    """

    def __init__(
        self,
        master,
        spectra_names: List[str],
        fit_param_memory: Dict[str, list],
        current_spectrum: str,
        *,
        model_dir: str = "param_models",
        callback=None,
    ):
        super().__init__(master)
        self.title("Fit parameters")
        self.palette = _apply_modern_style(self)
        self.spectra_names = list(spectra_names)
        self.memory = fit_param_memory
        self.current_spectrum = current_spectrum
        self.callback = callback
        self.model_dir = model_dir

        _ensure_dir(self.model_dir)

        self.n_components = 5
        self.vars = []       # list[ dict[str, tk.Variable] ] per row
        self.frame = None

        # Gentle guardrail: too many windows can slow things down
        try:
            num_toplevels = len([w for w in master.winfo_children() if isinstance(w, tk.Toplevel)])
            if num_toplevels > 8:
                messagebox.showwarning(
                    "Warning",
                    "You have many windows open. Close unused ones to avoid slowing down.",
                    parent=self
                )
        except Exception:
            pass

        # ===== Top bar: spectrum chooser + add/remove component =====
        bar = ttk.Frame(self, padding=6, style="Card.TFrame")
        bar.pack(fill="x")
        ttk.Label(bar, text="Spectrum:", style="Card.TLabel").pack(side="left")
        self.spec_var = tk.StringVar(value=self.current_spectrum)
        spec_cb = ttk.Combobox(
            bar, textvariable=self.spec_var, values=self.spectra_names,
            state="readonly", width=28
        )
        spec_cb.pack(side="left", padx=6)
        spec_cb.bind("<<ComboboxSelected>>", self.on_spectrum_changed)

        ttk.Button(bar, text="+", width=2, command=self.add_component, style="Alt.TButton").pack(side="right", padx=(3, 0))
        ttk.Button(bar, text="–", width=2, command=self.remove_component, style="Alt.TButton").pack(side="right", padx=(3, 0))

        # Main area
        self.main_frame = ttk.Frame(self, padding=8, style="Card.TFrame")
        self.main_frame.pack(fill="both", expand=True)

        self.n_components = self.get_initial_ncomp()
        self.redraw_table()

        # Accept button (write back into memory and notify parent)
        ttk.Button(self, text="Accept", command=self.accept, style="Primary.TButton").pack(side="bottom", pady=8, fill="x", padx=10)

        # ===== Model save/load bar =====
        model_frame = ttk.Frame(self, padding=6, style="Card.TFrame")
        model_frame.pack(pady=8)
        ttk.Button(model_frame, text="Save as model", command=self.save_model, style="Alt.TButton").pack(side="left", padx=4)
        ttk.Button(model_frame, text="Load model", command=self.load_model, style="Alt.TButton").pack(side="left", padx=4)

        # Close on ESC
        self.bind("<Escape>", lambda e: self.destroy())

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------
    def get_initial_ncomp(self) -> int:
        """Return number of components for the initial spectrum (from memory if available)."""
        name = self.spec_var.get()
        if name in self.memory and self.memory[name]:
            return len(self.memory[name])
        return 5

    # ------------------------------------------------------------------
    # Table rendering
    # ------------------------------------------------------------------
    def redraw_table(self):
        """Redraw the table of fit parameters for the current number of components."""
        if self.frame:
            self.frame.destroy()
        self.frame = ttk.Frame(self.main_frame)
        self.frame.pack()
        self.vars = []

        # Column layout (indices):
        #  0 : label
        #  1-3 : shift (min, val, max)
        #  4   : vary shift (?)
        #  5-7 : fwhm  (min, val, max)
        #  8   : vary fwhm (?)
        #  9-11: eta   (min, val, max)
        #  12  : vary eta (?)
        #  13-14: shape (G / GL radios)
        #  15  : Amp (value)
        #  16  : vary amplitude (?)

        header = [
            "", "Raman shift", "", "", "?", "FWHM", "", "", "?", "η (G/GL)", "", "", "?",
            "G", "GL", "Amp", "?"
        ]
        for j, txt in enumerate(header):
            span = 1
            if j == 1: span = 3      # shift block
            if j == 5: span = 3      # fwhm block
            if j == 9: span = 3      # eta block
            if txt:
                ttk.Label(self.frame, text=txt).grid(row=1, column=j, columnspan=span)
        subheader = [
            "", "min", "value", "max", "", "min", "value", "max", "",
            "min", "value", "max", "", "", "", "value", ""
        ]
        for j, txt in enumerate(subheader):
            if txt:
                ttk.Label(self.frame, text=txt).grid(row=2, column=j)

        params = self.memory.get(self.spec_var.get(), None)
        n = self.n_components

        # Rows (one per component)
        for i in range(n):
            row_vars = {}
            ttk.Label(self.frame, text=f"Comp #{i+1}").grid(row=3 + i, column=0, sticky="e")

            # ---- Shift (center) ----
            for k, param in enumerate(["min", "val", "max"]):
                key = f"shift_{param}"
                default = 1000.0 if param != "min" else 900.0
                v = tk.DoubleVar(
                    value=params[i][key] if params and i < len(params) and key in params[i] else default
                )
                ttk.Entry(self.frame, textvariable=v, width=8).grid(row=3 + i, column=1 + k)
                row_vars[key] = v

            var_fit_shift = tk.BooleanVar(
                value=(params[i]["fit_shift"] if params and i < len(params) and "fit_shift" in params[i] else True)
            )
            ttk.Checkbutton(self.frame, variable=var_fit_shift).grid(row=3 + i, column=4)
            row_vars["fit_shift"] = var_fit_shift

            # ---- FWHM ----
            for k, param in enumerate(["min", "val", "max"]):
                key = f"fwhm_{param}"
                default = 50.0
                v = tk.DoubleVar(
                    value=params[i][key] if params and i < len(params) and key in params[i] else default
                )
                ttk.Entry(self.frame, textvariable=v, width=8).grid(row=3 + i, column=5 + k)
                row_vars[key] = v

            var_fit_fwhm = tk.BooleanVar(
                value=(params[i]["fit_fwhm"] if params and i < len(params) and "fit_fwhm" in params[i] else True)
            )
            ttk.Checkbutton(self.frame, variable=var_fit_fwhm).grid(row=3 + i, column=8)
            row_vars["fit_fwhm"] = var_fit_fwhm

            # ---- Eta (G/GL ratio) ----
            for k, param in enumerate(["min", "val", "max"]):
                key = f"eta_{param}"
                default = 0.0 if param == "min" else (0.5 if param == "val" else 1.0)
                v = tk.DoubleVar(
                    value=params[i][key] if params and i < len(params) and key in params[i] else default
                )
                ttk.Entry(self.frame, textvariable=v, width=8).grid(row=3 + i, column=9 + k)
                row_vars[key] = v

            var_fit_eta = tk.BooleanVar(
                value=(params[i]["fit_eta"] if params and i < len(params) and "fit_eta" in params[i] else True)
            )
            ttk.Checkbutton(self.frame, variable=var_fit_eta).grid(row=3 + i, column=12)
            row_vars["fit_eta"] = var_fit_eta

            # ---- Shape (G / GL) ----
            var_shape = tk.StringVar(
                value=(params[i]["shape"] if params and i < len(params) and "shape" in params[i] else "G")
            )
            ttk.Radiobutton(self.frame, variable=var_shape, value="G").grid(row=3 + i, column=13)
            ttk.Radiobutton(self.frame, variable=var_shape, value="GL").grid(row=3 + i, column=14)
            row_vars["shape"] = var_shape

            # ---- Amplitude value + vary ----
            amp_default = 1.0
            amp_val = tk.DoubleVar(
                value=(params[i]["amp_val"] if params and i < len(params) and "amp_val" in params[i] else amp_default)
            )
            ttk.Entry(self.frame, textvariable=amp_val, width=8).grid(row=3 + i, column=15)
            row_vars["amp_val"] = amp_val

            var_fit_amp = tk.BooleanVar(
                value=(params[i]["fit_amp"] if params and i < len(params) and "fit_amp" in params[i] else True)
            )
            ttk.Checkbutton(self.frame, variable=var_fit_amp).grid(row=3 + i, column=16)
            row_vars["fit_amp"] = var_fit_amp

            self.vars.append(row_vars)

    # ------------------------------------------------------------------
    # Row count management
    # ------------------------------------------------------------------
    def add_component(self):
        self.n_components += 1
        self.redraw_table()

    def remove_component(self):
        if self.n_components > 1:
            self.n_components -= 1
            self.redraw_table()

    # ------------------------------------------------------------------
    # Save/load current set for the selected spectrum
    # ------------------------------------------------------------------
    def save_current_set(self):
        """Save current fit parameter set to memory (per spectrum), warning before overwrite."""
        name = self.spec_var.get()
        if name in self.memory:
            resp = messagebox.askyesno("Overwrite?", f"Parameters for '{name}' already exist. Overwrite?", parent=self)
            if not resp:
                return
        self.memory[name] = self.collect_current_params()
        messagebox.showinfo("Saved", f"Parameters saved for {name}", parent=self)

    def load_current_set(self):
        """Load saved parameter set for the current spectrum, if it exists."""
        name = self.spec_var.get()
        if name in self.memory:
            self.n_components = len(self.memory[name])
            self.redraw_table()
        else:
            messagebox.showinfo("No saved set", f"No parameters found for {name}", parent=self)

    # ------------------------------------------------------------------
    # Spectrum change
    # ------------------------------------------------------------------
    def on_spectrum_changed(self, event=None):
        self._sync_state_to_memory()
        name = self.spec_var.get()
        if name in self.memory:
            self.n_components = len(self.memory[name])
        else:
            self.n_components = 5
        self.redraw_table()

    # ------------------------------------------------------------------
    # Collect, accept, models I/O
    # ------------------------------------------------------------------
    def collect_current_params(self) -> list:
        """Collect all current parameter values from the table."""
        params = []
        for row in self.vars:
            d = {
                "shift_min": row["shift_min"].get(),
                "shift_val": row["shift_val"].get(),
                "shift_max": row["shift_max"].get(),
                "fit_shift": row["fit_shift"].get(),

                "fwhm_min": row["fwhm_min"].get(),
                "fwhm_val": row["fwhm_val"].get(),
                "fwhm_max": row["fwhm_max"].get(),
                "fit_fwhm": row["fit_fwhm"].get(),

                "eta_min":  row["eta_min"].get(),
                "eta_val":  row["eta_val"].get(),
                "eta_max":  row["eta_max"].get(),
                "fit_eta":  row["fit_eta"].get(),

                "shape": row["shape"].get(),

                "amp_val": row["amp_val"].get(),
                "fit_amp": row["fit_amp"].get(),
            }
            params.append(d)
        return params

    def accept(self):
        """Write current table values back to memory for the selected spectrum, notify caller, close."""
        name = self.spec_var.get()
        params = self.collect_current_params()
        self.memory[name] = params
        if self.callback:
            try:
                self.callback(self.memory)
            except Exception:
                pass
        self.destroy()

    def save_model(self):
        """Save current parameters as a model with user-chosen name, confirm overwrite if needed."""
        params = self.collect_current_params()
        name = simpledialog.askstring("Model name", "Name for this parameter model:", parent=self)
        if not name:
            return
        _ensure_dir(self.model_dir)
        filename = os.path.join(self.model_dir, name + ".json")
        if not _save_json_with_prompt(self, filename, params):
            return
        messagebox.showinfo("Success", f"Model '{name}' saved.", parent=self)
        self.redraw_table()

    def load_model(self):
        """Load a parameter model from model_dir via a dropdown combobox."""
        files = _list_model_files(self.model_dir)
        if not files:
            messagebox.showinfo("Info", "No saved models found.", parent=self)
            return

        win = tk.Toplevel(self)
        win.title("Load model")
        ttk.Label(win, text="Choose a model to load:").pack(padx=10, pady=(10, 2))
        model_var = tk.StringVar(value=files[0])
        cb = ttk.Combobox(win, textvariable=model_var, values=files, state="readonly", width=35)
        cb.pack(padx=12, pady=8)

        def do_load():
            selected = model_var.get()
            filename = os.path.join(self.model_dir, selected + ".json")
            try:
                with open(filename, "r", encoding="utf-8") as f:
                    params = json.load(f)
                self.set_params(params)
                win.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Error loading model:\n{e}", parent=win)

        ttk.Button(win, text="OK", command=do_load).pack(pady=(0, 10))
        ttk.Button(win, text="Cancel", command=win.destroy).pack()

    def set_params(self, params: list):
        """Replace parameters for the current spectrum and redraw."""
        self.n_components = len(params)
        self.memory[self.spec_var.get()] = copy.deepcopy(params)
        self.redraw_table()

    # ------------------------------------------------------------------
    # Internal state sync
    # ------------------------------------------------------------------
    def _sync_state_to_memory(self):
        """Store current editable table into memory for the currently selected spectrum."""
        try:
            name = self.spec_var.get()
            self.memory[name] = self.collect_current_params()
        except Exception:
            pass
