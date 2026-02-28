from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


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


def _find_col(columns: list[str], patterns: list[str]) -> Optional[str]:
    for pattern in patterns:
        for col in columns:
            if pattern in col.lower():
                return col
    return None


def parse_xas_file(path: str) -> XASData:
    df = pd.read_csv(path, sep=None, engine="python", comment="#")
    df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)
    cols = [str(c).strip() for c in df.columns]
    df.columns = cols

    energy_col = _find_col(cols, ["energy", "ev"])
    i0_col = _find_col(cols, ["i0", "incident"])
    it_col = _find_col(cols, ["it", "trans", "transmitted"])

    numeric_cols = [c for c in cols if pd.to_numeric(df[c], errors="coerce").notna().sum() > max(8, int(len(df) * 0.25))]
    if energy_col is None and numeric_cols:
        energy_col = numeric_cols[0]
    if i0_col is None and len(numeric_cols) >= 2:
        i0_col = numeric_cols[1]
    if it_col is None and len(numeric_cols) >= 3:
        it_col = numeric_cols[2]

    if not energy_col or not i0_col or not it_col:
        raise ValueError(
            "Could not detect Energy/I0/It columns. Expected columns like Energy, I0, It (or at least 3 numeric columns)."
        )

    energy = pd.to_numeric(df[energy_col], errors="coerce").to_numpy(float)
    i0 = pd.to_numeric(df[i0_col], errors="coerce").to_numpy(float)
    it = pd.to_numeric(df[it_col], errors="coerce").to_numpy(float)

    mask = np.isfinite(energy) & np.isfinite(i0) & np.isfinite(it)
    energy = energy[mask]
    i0 = i0[mask]
    it = it[mask]
    order = np.argsort(energy, kind="mergesort")

    return XASData(
        path=str(Path(path)),
        df=df,
        energy_col=energy_col,
        i0_col=i0_col,
        it_col=it_col,
        energy=energy[order],
        i0=i0[order],
        it=it[order],
    )


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
