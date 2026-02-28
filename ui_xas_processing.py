
"""
xas_processing_all.py

Single-file module combining:
- Reading EasyXAFS/easyEXAFS bundles from folders or ZIPs (including selecting multiple ZIPs)
- Parsing data to XASData
- Computing mu(E) with optional deglitching/smoothing
- Athena-equivalent processing via xraylarch (pre-edge, normalization, autobk, FFT)
- Robust element/edge inference for labeling, using ONLY scan_def["ROI_Scaled"] energy window + Larch/xraydb
  (Never uses scan_def["element"] for labeling)
- Minimal Tkinter UI window to select dataset and plot mu(E)

Notes:
- This module does NOT assume the ZIP contains a single file. It expects bundle ZIPs containing:
    *_exd.csv, *_mcas.npz (optional), scan_def.json, metadata.json
  and it supports selecting MULTIPLE ZIPs in one go.
- If a ZIP contains multiple datasets (multiple *_exd.csv), all are loaded.
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from functools import lru_cache
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# matplotlib optional; UI requires it
try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
except Exception:  # pragma: no cover
    matplotlib = None
    FigureCanvasTkAgg = None
    Figure = None

# Tk UI optional (only used if you instantiate XASProcessingWindow)
try:
    import tkinter as tk
    from tkinter import ttk, messagebox
except Exception:  # pragma: no cover
    tk = None
    ttk = None
    messagebox = None


# -----------------------------
# Data containers
# -----------------------------

@dataclass
class XASData:
    path: str
    df: pd.DataFrame
    energy_col: str
    i0_col: str
    it_col: str
    energy: np.ndarray
    i0: np.ndarray
    it: np.ndarray
    # Optional: carry scan_def / metadata (useful for labeling)
    scan_def: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


@dataclass
class Bundle:
    name: str
    df: pd.DataFrame
    scan_def: dict
    metadata: dict
    path: str
    npz_bytes: Optional[bytes] = None


# -----------------------------
# JSON helpers
# -----------------------------

def _safe_json_load_path(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))

def _safe_json_load_bytes(b: bytes) -> dict:
    return json.loads(b.decode("utf-8", errors="ignore"))


# -----------------------------
# ZIP bundle grouping (multiple datasets per ZIP supported)
# -----------------------------

def _group_zip_members_by_dataset(members: List[str]) -> Dict[str, Dict[str, str]]:
    """
    Group members into dataset groups keyed by the folder containing *_exd.csv.
    Works when files are at root OR inside subfolders.

    Returns:
      {group_key: {"csv": <member>, "npz": <member>, "scan_def": <member>, "metadata": <member>}}
    """
    csvs = [m for m in members if re.search(r"_exd\.csv$", m, flags=re.IGNORECASE)]
    if not csvs:
        return {}

    groups: Dict[str, Dict[str, str]] = {}
    for csv in csvs:
        key = str(Path(csv).parent).replace("\\", "/")
        if key == ".":
            key = ""
        groups.setdefault(key, {})
        groups[key]["csv"] = csv

    def find_in_group(key: str, pattern: str) -> Optional[str]:
        prefix = (key.rstrip("/") + "/") if key else ""
        in_same = [m for m in members if m.startswith(prefix) and re.search(pattern, m, re.IGNORECASE)]
        if in_same:
            return in_same[0]
        in_root = [m for m in members if "/" not in m.strip("/") and re.search(pattern, m, re.IGNORECASE)]
        return in_root[0] if in_root else None

    for key in list(groups.keys()):
        groups[key]["npz"] = find_in_group(key, r"_mcas\.npz$")
        groups[key]["scan_def"] = find_in_group(key, r"(?:^|/)scan_def\.json$")
        groups[key]["metadata"] = find_in_group(key, r"(?:^|/)metadata\.json$")

    return {k: v for k, v in groups.items() if "csv" in v}


def read_bundles_from_zip(zip_path: Union[str, Path]) -> List[Bundle]:
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path, "r") as z:
        members = z.namelist()
        groups = _group_zip_members_by_dataset(members)
        if not groups:
            raise FileNotFoundError(f"No '*_exd.csv' found in zip: {zip_path}")

        bundles: List[Bundle] = []
        for key, files in groups.items():
            csv_m = files["csv"]
            df = pd.read_csv(io.BytesIO(z.read(csv_m)))
            df.attrs["source_csv"] = csv_m

            scan_def = _safe_json_load_bytes(z.read(files["scan_def"])) if files.get("scan_def") else {}
            metadata = _safe_json_load_bytes(z.read(files["metadata"])) if files.get("metadata") else {}
            npz_bytes = z.read(files["npz"]) if files.get("npz") else None

            # Name: zip stem + dataset folder (or csv stem)
            csv_stem = Path(csv_m).stem
            ds_name = Path(key).name if key else csv_stem
            name = f"{zip_path.stem}__{ds_name}" if len(groups) > 1 else zip_path.stem

            bundles.append(
                Bundle(
                    name=name,
                    df=df,
                    scan_def=scan_def,
                    metadata=metadata,
                    path=str(zip_path),
                    npz_bytes=npz_bytes,
                )
            )
        return bundles


def read_bundle(path: Union[str, Path]) -> Bundle:
    """
    Reads a SINGLE dataset bundle from a directory or a ZIP.
    If ZIP contains multiple datasets, raise with guidance.
    """
    path = Path(path)
    if path.is_dir():
        csvs = list(path.glob("*_exd.csv"))
        if not csvs:
            raise FileNotFoundError(f"No '*_exd.csv' found in {path}")
        if len(csvs) > 1:
            raise ValueError(f"Multiple '*_exd.csv' files found in {path}. Use read_bundles(...) instead.")

        npzs = list(path.glob("*_mcas.npz"))
        if len(npzs) > 1:
            raise ValueError(f"Multiple '*_mcas.npz' files found in {path}. Use read_bundles(...) instead.")

        scan_p = path / "scan_def.json"
        meta_p = path / "metadata.json"

        df = pd.read_csv(csvs[0])
        df.attrs["source_csv"] = csvs[0].name

        return Bundle(
            name=path.name,
            df=df,
            scan_def=_safe_json_load_path(scan_p) if scan_p.exists() else {},
            metadata=_safe_json_load_path(meta_p) if meta_p.exists() else {},
            path=str(path),
            npz_bytes=npzs[0].read_bytes() if npzs else None,
        )

    if path.is_file() and path.suffix.lower() == ".zip":
        bundles = read_bundles_from_zip(path)
        if len(bundles) == 1:
            return bundles[0]
        raise ValueError(f"Zip contains multiple datasets; use read_bundles([...]) for {path}")

    raise ValueError("Path must be a bundle directory or a .zip bundle.")


def read_bundles(paths: Union[str, Path, List[Union[str, Path]]]) -> List[Bundle]:
    """
    Accepts one path or a list of paths. Each can be:
      - bundle directory
      - bundle ZIP (may contain 1+ datasets)
    Returns a flat list of Bundle objects.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]

    out: List[Bundle] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            out.append(read_bundle(p))
        elif p.is_file() and p.suffix.lower() == ".zip":
            out.extend(read_bundles_from_zip(p))
        else:
            raise ValueError(f"Unsupported path: {p}")
    return out


# -----------------------------
# Column detection and parsing
# -----------------------------

def _regex_find_col(columns: List[str], regexes: List[str]) -> Optional[str]:
    for rgx in regexes:
        r = re.compile(rgx, flags=re.IGNORECASE)
        for c in columns:
            if r.search(c):
                return c
    return None

def _xasdata_from_df(df: pd.DataFrame, path: str, *, scan_def: Optional[dict] = None, metadata: Optional[dict] = None) -> XASData:
    """
    Defensive parser:
    - prefers 'Energy(eV)' for energy, NOT Angle
    - tries to find I0 and It columns; if not found, falls back to numeric columns
    """
    df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)
    cols = [str(c).strip() for c in df.columns]
    df.columns = cols

    # ENERGY: prefer explicit energy columns (Energy(eV), Energy, ...), never "Angle"
    energy_col = _regex_find_col(cols, [r"^energy\s*\(.*ev.*\)$", r"\benergy\b.*\bev\b", r"^energy$"])
    if energy_col is None:
        # last resort: numeric column that is monotonic-ish (energy usually increases)
        numeric_cols = [c for c in cols if pd.to_numeric(df[c], errors="coerce").notna().sum() > max(8, int(len(df) * 0.25))]
        best = None
        for c in numeric_cols:
            arr = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
            arr = arr[np.isfinite(arr)]
            if arr.size < 10:
                continue
            d = np.diff(arr)
            score = float((d > 0).mean())
            if best is None or score > best[0]:
                best = (score, c)
        energy_col = best[1] if best else None

    i0_col = _regex_find_col(cols, [r"^i0\b", r"\bincident\b", r"\bi0\b"])
    it_col = _regex_find_col(cols, [r"^it\b", r"\btransmitted\b", r"\btrans\b", r"\bfluor\b", r"^if\b"])

    numeric_cols = [c for c in cols if pd.to_numeric(df[c], errors="coerce").notna().sum() > max(8, int(len(df) * 0.25))]
    if energy_col is None and numeric_cols:
        energy_col = numeric_cols[0]
    numeric_wo_energy = [c for c in numeric_cols if c != energy_col]

    if i0_col is None and len(numeric_wo_energy) >= 1:
        i0_col = numeric_wo_energy[0]
    if it_col is None and len(numeric_wo_energy) >= 2:
        it_col = numeric_wo_energy[1]

    if not energy_col or not i0_col or not it_col:
        raise ValueError(
            "Could not detect Energy/I0/It columns. "
            "Expected columns like 'Energy(eV)', 'I0', 'It' (or at least 3 numeric columns)."
        )

    energy = pd.to_numeric(df[energy_col], errors="coerce").to_numpy(float)
    i0 = pd.to_numeric(df[i0_col], errors="coerce").to_numpy(float)
    it = pd.to_numeric(df[it_col], errors="coerce").to_numpy(float)

    mask = np.isfinite(energy) & np.isfinite(i0) & np.isfinite(it)
    energy, i0, it = energy[mask], i0[mask], it[mask]

    order = np.argsort(energy, kind="mergesort")

    return XASData(
        path=path,
        df=df,
        energy_col=energy_col,
        i0_col=i0_col,
        it_col=it_col,
        energy=energy[order],
        i0=i0[order],
        it=it[order],
        scan_def=scan_def or {},
        metadata=metadata or {},
    )

def parse_xas_file(path: Union[str, Path]) -> XASData:
    path = Path(path)
    df = pd.read_csv(path, sep=None, engine="python", comment="#")
    return _xasdata_from_df(df, str(path))

def parse_xas_bundle(bundle: Bundle) -> XASData:
    return _xasdata_from_df(bundle.df.copy(), f"{bundle.path}::{bundle.name}", scan_def=bundle.scan_def, metadata=bundle.metadata)


# -----------------------------
# Simple mu(E)
# -----------------------------

def moving_average(y: np.ndarray, window: int = 7) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    window = int(max(1, window))
    if window <= 1:
        return y.copy()
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(y, kernel, mode="same")

def deglitch_robust(energy: np.ndarray, y: np.ndarray, z_thresh: float = 6.0, window: int = 21) -> np.ndarray:
    energy = np.asarray(energy, dtype=float)
    y = np.asarray(y, dtype=float)
    window = int(max(5, window))
    if window % 2 == 0:
        window += 1
    n = y.size
    if n < window:
        return y.copy()

    half = window // 2
    med = np.empty_like(y)
    mad = np.empty_like(y)
    for idx in range(n):
        lo = max(0, idx - half)
        hi = min(n, idx + half + 1)
        seg = y[lo:hi]
        m0 = np.nanmedian(seg)
        med[idx] = m0
        mad[idx] = np.nanmedian(np.abs(seg - m0)) + 1e-12

    robust_z = 0.6745 * (y - med) / mad
    flagged = np.abs(robust_z) > float(z_thresh)
    if flagged.any():
        y_fixed = y.copy()
        good = ~flagged
        y_fixed[flagged] = np.interp(energy[flagged], energy[good], y[good])
        return y_fixed
    return y.copy()

def transmission_to_mu(i0: np.ndarray, it: np.ndarray, log_base: str = "ln") -> np.ndarray:
    i0 = np.asarray(i0, dtype=float)
    it = np.asarray(it, dtype=float)
    eps = 1e-12
    ratio = np.maximum(i0, eps) / np.maximum(it, eps)
    if log_base == "log10":
        return np.log10(ratio)
    return np.log(ratio)

def compute_mu(
    xas: XASData,
    *,
    log_base: str = "ln",
    deglitch: bool = False,
    deglitch_z: float = 6.0,
    deglitch_window: int = 21,
    smooth_window: int = 1,
) -> np.ndarray:
    mu = transmission_to_mu(xas.i0, xas.it, log_base=log_base)
    if deglitch:
        mu = deglitch_robust(xas.energy, mu, z_thresh=deglitch_z, window=deglitch_window)
    if int(smooth_window) > 1:
        mu = moving_average(mu, window=int(smooth_window))
    return mu


# -----------------------------
# Larch + edge inference (STRICT: ROI_Scaled only; never scan_def["element"])
# -----------------------------

def _require_larch():
    try:
        from larch import Group, xraydb
        from larch.xafs import find_e0, pre_edge, autobk, xftf
        return Group, xraydb, find_e0, pre_edge, autobk, xftf
    except Exception as exc:
        raise ImportError("xraylarch is required for Athena-like processing. Install with: pip install xraylarch") from exc

def _roi_scaled_energy_window(scan_def: dict) -> Optional[Tuple[float, float]]:
    roi_s = scan_def.get("ROI_Scaled") or {}
    e_min = roi_s.get("roi_min")
    e_max = roi_s.get("roi_max")
    if e_min is None or e_max is None:
        return None
    try:
        e_min = float(e_min); e_max = float(e_max)
    except Exception:
        return None
    if not np.isfinite(e_min) or not np.isfinite(e_max) or e_max <= e_min:
        return None
    return (e_min, e_max)

def infer_xas_edge_from_roi_scaled(
    energy_ev: np.ndarray,
    mu: np.ndarray,
    scan_def: dict,
    *,
    max_delta_ev: float = 80.0,
) -> Dict[str, Any]:
    """
    Infer element+edge using ONLY:
      - scan_def["ROI_Scaled"]["roi_min"/"roi_max"] to constrain candidate edges
      - Larch find_e0 on mu(E) (or fallback to max derivative)
      - Larch xraydb edge table

    Returns {} if Larch/xraydb isn't available or no robust match.
    """
    win = _roi_scaled_energy_window(scan_def)
    if win is None:
        return {}

    e_win_min, e_win_max = win
    e = np.asarray(energy_ev, float)
    m = np.asarray(mu, float)
    mask = np.isfinite(e) & np.isfinite(m)
    if mask.sum() < 8:
        return {}
    e = e[mask]; m = m[mask]
    order = np.argsort(e, kind="mergesort")
    e = e[order]; m = m[order]

    wmask = (e >= e_win_min) & (e <= e_win_max)
    if wmask.sum() < 8:
        return {}

    ew = e[wmask]
    mw = m[wmask]

    # Find E0
    try:
        _, xraydb, find_e0, *_ = _require_larch()
        e0 = float(find_e0(energy=ew, mu=mw))
    except Exception:
        try:
            d = np.gradient(mw, ew)
            e0 = float(ew[int(np.nanargmax(d))])
        except Exception:
            return {}

    pad = max(25.0, 0.01 * (e_win_max - e_win_min))
    cmin = e_win_min - pad
    cmax = e_win_max + pad

    best = None
    for sym, edge_name, ee in _cached_edge_energy_table():
        if ee < cmin or ee > cmax:
            continue
        delta = abs(ee - e0)
        if delta > max_delta_ev:
            continue
        cand = {
            "element": sym,
            "edge": str(edge_name),
            "label": f"XAS({sym} {edge_name})",
            "e0": float(e0),
            "edge_energy": float(ee),
            "delta_e0": float(delta),
            "roi_scaled_min": float(e_win_min),
            "roi_scaled_max": float(e_win_max),
        }
        if best is None or cand["delta_e0"] < best["delta_e0"]:
            best = cand

    return best or {}


@lru_cache(maxsize=1)
def _cached_edge_energy_table() -> Tuple[Tuple[str, str, float], ...]:
    """Cache all tabulated edge energies to avoid repeated xraydb scans."""
    try:
        _, xraydb, *_ = _require_larch()
    except Exception:
        return tuple()

    out: List[Tuple[str, str, float]] = []
    for z in range(1, 99):
        sym = xraydb.atomic_symbol(z)
        if not sym:
            continue
        try:
            edges = xraydb.xray_edges(sym)
        except Exception:
            continue
        if not edges:
            continue

        for edge_name, edge_obj in edges.items():
            ee = getattr(edge_obj, "energy", None)
            if ee is None:
                continue
            try:
                ee = float(ee)
            except Exception:
                continue
            if np.isfinite(ee):
                out.append((sym, str(edge_name), ee))
    return tuple(out)


def infer_xas_edge_from_spectrum(energy_ev: np.ndarray, mu: np.ndarray, *, max_delta_ev: float = 80.0) -> Dict[str, Any]:
    """Infer XAS edge from spectrum only (no scan_def required)."""
    try:
        _, _, find_e0, *_ = _require_larch()
        e0 = float(find_e0(energy=energy_ev, mu=mu))
    except Exception:
        e = np.asarray(energy_ev, dtype=float)
        m = np.asarray(mu, dtype=float)
        mask = np.isfinite(e) & np.isfinite(m)
        if mask.sum() < 8:
            return {}
        e = e[mask]
        m = m[mask]
        order = np.argsort(e, kind="mergesort")
        e = e[order]
        m = m[order]
        try:
            d = np.gradient(m, e)
            e0 = float(e[int(np.nanargmax(d))])
        except Exception:
            return {}

    best = None
    for sym, edge_name, edge_energy in _cached_edge_energy_table():
        delta = abs(edge_energy - e0)
        if delta > max_delta_ev:
            continue
        cand = {
            "element": sym,
            "edge": edge_name,
            "label": f"{sym} {edge_name}",
            "e0": e0,
            "edge_energy": edge_energy,
            "delta_e0": delta,
        }
        if best is None or cand["delta_e0"] < best["delta_e0"]:
            best = cand
    return best or {}


# -----------------------------
# Record loading for UI / app
# -----------------------------

def load_records_from_paths(paths: Union[str, Path, List[Union[str, Path]]]) -> List[Dict[str, Any]]:
    """
    High-level loader: given multiple ZIPs/folders, returns records for UI:
      {"title": "...", "xas": XASData, "bundle": Bundle|None, "edge": dict}

    Titles are generated as:
      XAS(Element Edge) — <dataset name>
    where Element/Edge come ONLY from ROI_Scaled + larch inference.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]

    bundles: List[Bundle] = []
    files: List[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_dir() or (p.is_file() and p.suffix.lower() == ".zip"):
            bundles.extend(read_bundles(p))
        elif p.is_file():
            files.append(p)
        else:
            raise ValueError(f"Unsupported path: {p}")

    records: List[Dict[str, Any]] = []

    for b in bundles:
        xas = parse_xas_bundle(b)
        mu = compute_mu(xas)
        edge = infer_xas_edge_from_roi_scaled(xas.energy, mu, xas.scan_def)
        label = edge.get("label", "XAS(Unknown)")
        title = f"{label} — {b.name}"
        records.append({"title": title, "xas": xas, "bundle": b, "edge": edge})

    for f in files:
        xas = parse_xas_file(f)
        mu = compute_mu(xas)
        edge = infer_xas_edge_from_roi_scaled(xas.energy, mu, xas.scan_def)
        label = edge.get("label", "XAS(Unknown)")
        title = f"{label} — {f.stem}"
        records.append({"title": title, "xas": xas, "bundle": None, "edge": edge})

    if not records:
        raise FileNotFoundError("No datasets found.")
    return records


# -----------------------------
# Minimal UI (updated)
# -----------------------------

class XASProcessingWindow:
    def __init__(self, master, records: List[Dict[str, Any]]):
        if tk is None or ttk is None or Figure is None:
            raise RuntimeError("Tkinter/Matplotlib not available in this environment.")

        self.master = master
        self.records = records
        self.master.title("XAS processing")
        self.master.geometry("1100x700")

        left = ttk.Frame(master, padding=10)
        left.pack(side="left", fill="y")
        right = ttk.Frame(master, padding=10)
        right.pack(side="left", fill="both", expand=True)

        ttk.Label(left, text="XAS dataset").pack(anchor="w")
        self.var_dataset = tk.StringVar(value=records[0]["title"])
        self.cb_dataset = ttk.Combobox(
            left,
            textvariable=self.var_dataset,
            state="readonly",
            values=[r["title"] for r in records],
            width=48,
        )
        self.cb_dataset.pack(fill="x", pady=(0, 10))

        self.var_log = tk.StringVar(value="ln")
        ttk.Label(left, text="log base").pack(anchor="w")
        ttk.Combobox(left, textvariable=self.var_log, state="readonly", values=["ln", "log10"], width=10).pack(anchor="w", pady=(0, 8))

        self.var_deglitch = tk.BooleanVar(value=False)
        ttk.Checkbutton(left, text="Deglitch", variable=self.var_deglitch).pack(anchor="w")

        ttk.Label(left, text="Deglitch z").pack(anchor="w")
        self.ent_deg_z = ttk.Entry(left, width=10)
        self.ent_deg_z.insert(0, "6.0")
        self.ent_deg_z.pack(anchor="w", pady=(0, 6))

        ttk.Label(left, text="Deglitch window").pack(anchor="w")
        self.ent_deg_win = ttk.Entry(left, width=10)
        self.ent_deg_win.insert(0, "21")
        self.ent_deg_win.pack(anchor="w", pady=(0, 6))

        ttk.Label(left, text="Smooth window").pack(anchor="w")
        self.ent_smooth = ttk.Entry(left, width=10)
        self.ent_smooth.insert(0, "1")
        self.ent_smooth.pack(anchor="w", pady=(0, 12))

        ttk.Button(left, text="Compute and plot μ(E)", command=self.compute_and_plot).pack(fill="x")

        self.fig = Figure(figsize=(7, 5), dpi=110)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.compute_and_plot()

    def _get_selected_record(self) -> Dict[str, Any]:
        name = self.var_dataset.get()
        for rec in self.records:
            if rec.get("title") == name:
                return rec
        return self.records[0]

    def compute_and_plot(self):
        rec = self._get_selected_record()
        try:
            z = float(self.ent_deg_z.get())
            win = int(self.ent_deg_win.get())
            smooth = int(self.ent_smooth.get())
            mu = compute_mu(
                rec["xas"],
                log_base=self.var_log.get(),
                deglitch=self.var_deglitch.get(),
                deglitch_z=z,
                deglitch_window=win,
                smooth_window=smooth,
            )
        except Exception as exc:
            if messagebox:
                messagebox.showerror("XAS processing", str(exc), parent=self.master)
            else:
                raise
            return

        self.ax.clear()
        self.ax.plot(rec["xas"].energy, mu, lw=1.5, label="μ(E)")
        self.ax.set_xlabel("Energy (eV)")
        self.ax.set_ylabel("μ(E)")
        self.ax.set_title(rec.get("title", "μ(E)"))
        self.ax.grid(alpha=0.25)
        self.ax.legend(loc="best")
        self.fig.tight_layout()
        self.canvas.draw_idle()
