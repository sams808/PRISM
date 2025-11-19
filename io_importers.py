# io_importers.py
from __future__ import annotations
import re, os, io
from pathlib import Path
import numpy as np
import pandas as pd

# --------------------------- utils ---------------------------

_NUM = re.compile(r"[+-]?(?:\d+\.?\d*|\d*\.?\d+)(?:[eE][+-]?\d+)?")

def _read_head(path, n=80, encodings=("utf-8-sig","latin-1","cp1252")):
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, errors="strict") as f:
                return f.read().splitlines()[:n], enc
        except Exception:
            continue
    # last resort
    with open(path, "rb") as f:
        raw = f.read(3000)
    try:
        text = raw.decode("utf-8", "ignore")
    except Exception:
        text = raw.decode("latin-1", "ignore")
    return text.splitlines()[:n], "utf-8"

def _guess_decimal_and_sep(lines):
    """Very robust: looks for ; , \t, and decimal comma usage."""
    sample = "\n".join(lines[:40])
    dec_comma_hits = len(re.findall(r"\d+,\d", sample))
    dec_dot_hits   = len(re.findall(r"\d+\.\d", sample))
    decimal = "," if dec_comma_hits > dec_dot_hits else "."
    # Prioritize ; then , then \t for TA/Netzsch
    if ";" in sample:
        sep = ";"
    elif "\t" in sample:
        sep = "\t"
    elif "," in sample and decimal == ".":
        sep = ","
    else:
        # Let pandas guess later
        sep = None
    return decimal, sep

def _first_numeric_row(lines, min_nums=2):
    for idx, ln in enumerate(lines):
        # remove obvious header junk
        if ln.strip().startswith("#"): 
            continue
        toks = re.split(r"[;,\t ]+", ln.strip())
        nums = sum(1 for t in toks if _NUM.fullmatch(t.replace(",",".") ))
        if nums >= min_nums:
            return idx
    return None

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase + strip + unify whitespaces."""
    new = {}
    for c in df.columns:
        s = str(c).strip()
        s = re.sub(r"\s+", " ", s)
        new[c] = s
    df = df.rename(columns=new)
    return df

def _canon_name_map(col: str):
    c = col.lower()
    # temperature
    if re.search(r"temp|t\W*set", c) and "°c" in c or " c" in c or "° c" in c:
        return "temp_C"
    if re.search(r"temp", c) and ("k" in c or " °k" in c):
        return "temp_K"
    # time
    if re.search(r"\btime\b", c) and ("min" in c or "m" == c.split()[-1]):
        return "time_min"
    if re.search(r"\btime\b", c) and ("s" in c or "sec" in c):
        return "time_s"
    # DSC / Heat flow
    if "dsc" in c or "heat flow" in c or "heatflow" in c or "mW" in c:
        return "dsc_mWmg" if "mw" in c else "dsc"
    # TG / mass / weight
    if "tg" in c or "mass %" in c or "weight %" in c or re.search(r"\bwt\W*%\b", c):
        return "tg_pct"
    # DTG
    if "dtg" in c or "%/min" in c:
        return "dtg_pct_min"
    # generic x/y
    if c in ("x", "x axis", "raman shift (cm-1)", "raman shift (cm⁻¹)"):
        return "x"
    if c in ("y", "intensity", "counts", "a.u.", "intensity (a.u.)"):
        return "y"
    return None

def _rename_to_canon(df: pd.DataFrame):
    canon = {}
    for c in df.columns:
        cc = _canon_name_map(c)
        if cc:
            # don't overwrite if already present
            if cc in canon.values():
                continue
            canon[c] = cc
    return df.rename(columns=canon)

# --------------------------- DTA detection & loading ---------------------------

def _looks_like_dta(lines):
    head = "\n".join(lines[:25]).lower()
    if "netzsch" in head or "ta instruments" in head or "universal analysis" in head:
        return True
    # heuristics: presence of dsc/tg/dtg labels
    if re.search(r"\b(dsc|heat\s*flow|tg|dtg)\b", head):
        return True
    return False

def _read_table(path, header_row=None, sep=None, decimal="."):
    if header_row is None:
        # let pandas try to guess header, fallback to no header
        try:
            df = pd.read_csv(path, sep=sep, decimal=decimal, engine="python")
        except Exception:
            df = pd.read_csv(path, sep=sep, decimal=decimal, engine="python", header=None)
    else:
        df = pd.read_csv(path, sep=sep, decimal=decimal, engine="python", header=header_row)
    return _normalize_cols(df)

def load_dta(path: str):
    lines, enc = _read_head(path)
    decimal, sep = _guess_decimal_and_sep(lines)
    # find first numeric row to use as header if the row above looks like headers
    header_row = None
    numrow = _first_numeric_row(lines, min_nums=2)
    if numrow is None:
        # try blind read, user will choose later if needed
        df = _read_table(path, header_row=None, sep=sep, decimal=decimal)
    else:
        # try to put header as the row just before numeric if that row is texty
        if numrow > 0 and re.search(r"[A-Za-z]", lines[numrow-1]):
            header_row = numrow - 1
        df = _read_table(path, header_row=header_row, sep=sep, decimal=decimal)
    df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)
    df = _rename_to_canon(df)

    meta = {"source": "NETZSCH/TA (heuristic)", "encoding": enc, "decimal": decimal, "sep": sep}
    return {"kind": "TA", "df": df, "meta": meta}

# --------------------------- Generic XY loading ---------------------------

def load_xy(path: str):
    # robust read; let pandas sniff, then reduce to first two numeric-like columns
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        # try common seps
        for sep in [",", ";", "\t", " "]:
            try:
                df = pd.read_csv(path, sep=sep, engine="python")
                break
            except Exception:
                df = None
        if df is None:
            raise
    df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)
    # find first 2 numeric columns
    numeric_cols = [c for c in df.columns if pd.to_numeric(df[c], errors="coerce").notna().sum() > max(5, int(len(df)*0.25))]
    if len(numeric_cols) < 2:
        # fallback: try to coerce everything
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(how="all", axis=1)
        numeric_cols = [c for c in df.columns if df[c].notna().sum() > 5]
        if len(numeric_cols) < 2:
            raise ValueError("Could not find two numeric columns for XY data.")
    x = df[numeric_cols[0]].to_numpy(dtype=float)
    y = df[numeric_cols[1]].to_numpy(dtype=float)
    return {"kind": "XY", "x": x, "y": y, "label": Path(path).stem}

# --------------------------- Public API ---------------------------

def load_any(path: str):
    ext = Path(path).suffix.lower()
    lines, _ = _read_head(path)
    if ext in (".dta", ".sta", ".dat") or _looks_like_dta(lines):
        return load_dta(path)
    # otherwise try XY first; if it fails but file looks DTA → try DTA
    try:
        return load_xy(path)
    except Exception:
        if _looks_like_dta(lines):
            return load_dta(path)
        raise

def pick_ta_xy(df: pd.DataFrame):
    """
    Heuristic: prefer Temperature vs DSC, else Time vs DSC, else TG, else DTG, else first numeric pair.
    Returns (x, y, info_dict).
    """
    # ensure canonical names exist when possible
    df = _rename_to_canon(_normalize_cols(df))

    col_x = None
    for cand in ("temp_C","temp_K","time_min","time_s"):
        if cand in df.columns:
            col_x = cand; break
    col_y = None
    for cand in ("dsc_mWmg","dsc","tg_pct","dtg_pct_min"):
        if cand in df.columns:
            col_y = cand; break

    # if still none, pick first two numeric
    if col_x is None or col_y is None:
        numeric_cols = [c for c in df.columns if pd.to_numeric(df[c], errors="coerce").notna().sum() > max(5, int(len(df)*0.25))]
        if len(numeric_cols) >= 2:
            if col_x is None: col_x = numeric_cols[0]
            if col_y is None: col_y = numeric_cols[1]
        else:
            raise ValueError("Could not choose X/Y columns automatically for TA data.")

    x = pd.to_numeric(df[col_x], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[col_y], errors="coerce").to_numpy(dtype=float)

    label = f"{col_y} vs {col_x}"
    return x, y, {"label": label, "x_col": col_x, "y_col": col_y}

# --------------------------- Optional mini-wizard ---------------------------

import tkinter as tk
from tkinter import ttk, messagebox

class ColumnChooser(tk.Toplevel):
    """
    Minimal dialog used only when we can't auto-pick TA X/Y.
    """
    def __init__(self, master, df: pd.DataFrame, title="Choose columns for TA / DTA"):
        super().__init__(master)
        self.title(title); self.geometry("420x180"); self.resizable(False, False); self.transient(master)
        self.result = None
        cols = list(df.columns)
        pad = {"padx": 8, "pady": 6}
        ttk.Label(self, text="X column").grid(row=0, column=0, **pad, sticky="e")
        ttk.Label(self, text="Y column").grid(row=1, column=0, **pad, sticky="e")
        self.cb_x = ttk.Combobox(self, values=cols, state="readonly", width=32)
        self.cb_y = ttk.Combobox(self, values=cols, state="readonly", width=32)
        self.cb_x.grid(row=0, column=1, **pad, sticky="w")
        self.cb_y.grid(row=1, column=1, **pad, sticky="w")
        btns = ttk.Frame(self); btns.grid(row=2, column=0, columnspan=2, pady=(10,6))
        ttk.Button(btns, text="Cancel", command=self.destroy).pack(side="right", padx=4)
        ttk.Button(btns, text="OK", command=self._ok).pack(side="right", padx=4)
        self.grab_set(); self.cb_x.focus_set()

    def _ok(self):
        x = self.cb_x.get().strip()
        y = self.cb_y.get().strip()
        if not x or not y or x == y:
            messagebox.showwarning("Columns", "Pick two different columns.", parent=self); return
        self.result = (x, y)
        self.destroy()
