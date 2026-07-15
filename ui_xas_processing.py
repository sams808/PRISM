"""
Compatibility XAS module backed by xas_processing_v10.

This module keeps the API expected by main.py while delegating core XAS
processing logic to the v10 implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import tkinter as tk
    from tkinter import ttk
except Exception:  # pragma: no cover
    tk = None
    ttk = None

from xas_processing_v10 import (
    XASUltimateApp,
    _extract_energy_angle_signal,
    infer_edge_label_from_roi_scaled,
    mu_from_transmission,
    read_athena_prj,
    read_csv_dataset,
    read_easyxafs_zip,
)


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


def _xasdata_from_df(df: pd.DataFrame, path: str, *, scan_def: Optional[dict] = None, metadata: Optional[dict] = None) -> XASData:
    df = df.dropna(how="all", axis=0).dropna(how="all", axis=1).copy()

    angle, energy, signal, names = _extract_energy_angle_signal(df)
    energy_col = names.get("energy_col", "Energy")

    i0_col = None
    it_col = None
    cols = [str(c) for c in df.columns]
    # infer I0/It with direct column patterns first
    import re

    for c in cols:
        cl = c.lower()
        if i0_col is None and re.search(r"^i0\b|\bincident\b", cl):
            i0_col = c
        if it_col is None and re.search(r"^it\b|\btransmitted\b|\btrans\b|^if\b|\bfluor\b", cl):
            it_col = c

    # strict fallback to classic 3-column numeric convention: Energy + two numeric channels
    if i0_col is None or it_col is None:
        numeric_cols = [
            c for c in cols
            if pd.to_numeric(df[c], errors="coerce").notna().sum() > max(8, int(len(df) * 0.25))
        ]
        numeric_wo_energy = [c for c in numeric_cols if c != energy_col]
        if i0_col is None and numeric_wo_energy:
            i0_col = numeric_wo_energy[0]
        if it_col is None and len(numeric_wo_energy) > 1:
            it_col = numeric_wo_energy[1]

    if not i0_col or not it_col:
        # If only one signal exists (v10 style), synthesize It from signal and I0=1 for compatibility
        s_col = names.get("signal_col")
        if s_col and s_col in df.columns:
            i0_col = "_i0_synth"
            it_col = "_it_from_signal"
            df[i0_col] = 1.0
            sig = pd.to_numeric(df[s_col], errors="coerce").to_numpy(float)
            sig = np.clip(sig, 1e-12, np.inf)
            df[it_col] = 1.0 / sig
        else:
            raise ValueError("Could not infer I0/It columns from dataset.")

    e = pd.to_numeric(df[energy_col], errors="coerce").to_numpy(float)
    i0 = pd.to_numeric(df[i0_col], errors="coerce").to_numpy(float)
    it = pd.to_numeric(df[it_col], errors="coerce").to_numpy(float)

    mask = np.isfinite(e) & np.isfinite(i0) & np.isfinite(it)
    e = e[mask]
    i0 = i0[mask]
    it = it[mask]
    order = np.argsort(e, kind="mergesort")

    return XASData(
        path=path,
        df=df,
        energy_col=energy_col,
        i0_col=i0_col,
        it_col=it_col,
        energy=e[order],
        i0=i0[order],
        it=it[order],
        scan_def=scan_def or {},
        metadata=metadata or {},
    )


def parse_xas_file(path: Union[str, Path]) -> XASData:
    rec = read_csv_dataset(path)
    return _xasdata_from_df(rec["df"], str(path), scan_def=rec.get("scan_def") or {}, metadata=rec.get("metadata") or {})


def read_athena_project(path: Union[str, Path]) -> List[XASData]:
    """Load Athena .prj groups into XASData-compatible datasets."""
    spectra = read_athena_prj(path)
    out: List[XASData] = []
    for idx, sp in enumerate(spectra):
        energy = np.asarray(getattr(sp, "energy", []), dtype=float)
        mu = np.asarray(getattr(sp, "y", []), dtype=float)
        mask = np.isfinite(energy) & np.isfinite(mu)
        energy = energy[mask]
        mu = mu[mask]
        if energy.size == 0:
            continue

        # Build synthetic transmission channels so compute_mu() remains compatible.
        i0 = np.ones_like(mu, dtype=float)
        mu_clipped = np.clip(mu, -700.0, 700.0)
        it = np.exp(-mu_clipped)

        name = getattr(sp, "name", "Athena") or f"Athena_{idx+1}"
        df = pd.DataFrame({"Energy": energy, "I0_synth": i0, "It_from_mu": it, "mu_imported": mu})
        out.append(
            XASData(
                path=f"{path}::{name}::{idx}",
                df=df,
                energy_col="Energy",
                i0_col="I0_synth",
                it_col="It_from_mu",
                energy=energy,
                i0=i0,
                it=it,
                scan_def={},
                metadata={"source": str(path), "athena_name": name},
            )
        )
    return out


def read_bundles_from_zip(zip_path: Union[str, Path]) -> List[Bundle]:
    records = read_easyxafs_zip(zip_path)
    out: List[Bundle] = []
    for rec in records:
        out.append(
            Bundle(
                name=str(rec.get("name", Path(zip_path).stem)),
                df=rec["df"],
                scan_def=rec.get("scan_def") or {},
                metadata=rec.get("metadata") or {},
                path=str(zip_path),
            )
        )
    return out


def read_bundle(path: Union[str, Path]) -> Bundle:
    p = Path(path)
    if p.is_file() and p.suffix.lower() == ".zip":
        bundles = read_bundles_from_zip(p)
        if len(bundles) != 1:
            raise ValueError(f"Zip contains {len(bundles)} datasets; use read_bundles().")
        return bundles[0]
    raise ValueError("Path must be a .zip EasyXAFS bundle.")


def read_bundles(paths: Union[str, Path, List[Union[str, Path]]]) -> List[Bundle]:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    out: List[Bundle] = []
    for p in paths:
        p = Path(p)
        if p.is_file() and p.suffix.lower() == ".zip":
            out.extend(read_bundles_from_zip(p))
        else:
            raise ValueError(f"Unsupported path: {p} (expected .zip)")
    return out


def parse_xas_bundle(bundle: Bundle) -> XASData:
    return _xasdata_from_df(bundle.df.copy(), f"{bundle.path}::{bundle.name}", scan_def=bundle.scan_def, metadata=bundle.metadata)


def compute_mu(
    xas: XASData,
    *,
    log_base: str = "ln",
    deglitch: bool = False,
    deglitch_z: float = 6.0,
    deglitch_window: int = 21,
    smooth_window: int = 1,
) -> np.ndarray:
    mu = mu_from_transmission(xas.i0, xas.it, logbase=log_base)
    # Keep compatibility flags; v10-based processing is available in dedicated app.
    if int(smooth_window) > 1:
        k = int(smooth_window)
        if k % 2 == 0:
            k += 1
        kernel = np.ones(k, dtype=float) / k
        mu = np.convolve(mu, kernel, mode="same")
    return mu


def infer_xas_edge_from_roi_scaled(
    energy_ev: np.ndarray,
    mu: np.ndarray,
    scan_def: dict,
    *,
    max_delta_ev: float = 80.0,
) -> Dict[str, Any]:
    label, e0 = infer_edge_label_from_roi_scaled(energy_ev, mu, scan_def, max_delta=float(max_delta_ev))
    out: Dict[str, Any] = {"label": "?"}
    import re
    m = re.match(r"^XAS\(([^\s\)\?]+)\s+([^\s\)\?]+)\)$", str(label))
    if m:
        out["element"] = m.group(1)
        out["edge"] = m.group(2)
        out["label"] = f"{m.group(1)} {m.group(2)}"
    if e0 is not None and np.isfinite(e0):
        out["e0"] = float(e0)
    return out


def infer_xas_edge_from_spectrum(energy_ev: np.ndarray, mu: np.ndarray, *, max_delta_ev: float = 80.0) -> Dict[str, Any]:
    label, e0 = infer_edge_label_from_roi_scaled(energy_ev, mu, {}, max_delta=float(max_delta_ev))
    out: Dict[str, Any] = {"label": "?"}
    import re
    m = re.match(r"^XAS\(([^\s\)\?]+)\s+([^\s\)\?]+)\)$", str(label))
    if m:
        out["element"] = m.group(1)
        out["edge"] = m.group(2)
        out["label"] = f"{m.group(1)} {m.group(2)}"
    if e0 is not None and np.isfinite(e0):
        out["e0"] = float(e0)
    return out


def load_records_from_paths(paths: Union[str, Path, List[Union[str, Path]]]) -> List[Dict[str, Any]]:
    if isinstance(paths, (str, Path)):
        paths = [paths]

    records: List[Dict[str, Any]] = []
    for p in paths:
        pp = Path(p)
        if pp.is_file() and pp.suffix.lower() == ".zip":
            for b in read_bundles_from_zip(pp):
                xas = parse_xas_bundle(b)
                mu = compute_mu(xas)
                edge = infer_xas_edge_from_roi_scaled(xas.energy, mu, xas.scan_def)
                records.append({"title": f"{edge.get('label', 'XAS(Unknown)')} — {b.name}", "xas": xas, "bundle": b, "edge": edge})
        elif pp.is_file():
            xas = parse_xas_file(pp)
            mu = compute_mu(xas)
            edge = infer_xas_edge_from_roi_scaled(xas.energy, mu, xas.scan_def)
            records.append({"title": f"{edge.get('label', 'XAS(Unknown)')} — {pp.stem}", "xas": xas, "bundle": None, "edge": edge})
        else:
            raise ValueError(f"Unsupported path: {p}")

    if not records:
        raise FileNotFoundError("No datasets found.")
    return records


class XASProcessingWindow:
    """Adapter that opens the full v10 XAS GUI."""

    def __init__(self, master, records: List[Dict[str, Any]]):
        if tk is None or ttk is None:
            raise RuntimeError("Tkinter not available in this environment.")
        self.master = master
        self.records = records
        self.master.title("XAS processing")
        ttk.Label(master, text="Launching XAS v10 processor...", padding=16).pack(fill="both", expand=True)
        self.master.after(100, self._launch)

    def _launch(self):
        # Launch v10 app in-process as requested and close placeholder Toplevel.
        self.master.destroy()
        app = XASUltimateApp(initial_records=self.records, allow_import=False)
        app.mainloop()
