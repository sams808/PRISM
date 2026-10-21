"""
raman_library.py — the user's own Raman reference library and full-spectrum
similarity search (framework-agnostic).

Spectragryph's Identify workflow in PRISM terms: build a library from
spectra you measured yourself (standards, previous samples), then rank
unknowns against it by full-spectrum similarity instead of peak lists. The
same ranking runs over the RRUFF cache (its measured spectra), so the two
sources answer the same question with the same number.

Similarity ("hit quality") is the Pearson correlation coefficient of the
two spectra on a common grid, after the chosen preprocessing (raw, smoothed,
first or second derivative), inside the overlap of their x-ranges or a
user range. The correlation ignores scale and offset, so intensity units
and a flat background never matter; derivatives remove sloping baselines
(fluorescence) at the price of noise sensitivity, which the smoothing
window controls.

The library is one JSON file (default ~/.raman_cache/raman_library/
library.json): arrays as lists, metadata as a dict. Packs (core.db_packs)
copy that file.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

LIBRARY_DIR = os.path.join(os.path.expanduser("~"), ".raman_cache", "raman_library")
LIBRARY_PATH = os.path.join(LIBRARY_DIR, "library.json")
LIBRARY_FORMAT = "prism-raman-library"
LIBRARY_VERSION = 1

PREPROCESS_CHOICES = ("raw", "smooth", "d1", "d2")
METHOD_CHOICES = ("pearson", "euclid")


@dataclass
class LibraryEntry:
    id: str
    name: str
    x: List[float]
    y: List[float]
    meta: Dict[str, Any] = field(default_factory=dict)  # laser_nm, source, notes, added_at, …

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex

    def arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.asarray(self.x, float), np.asarray(self.y, float)


def entry_from_spectrum(name: str, x: Sequence[float], y: Sequence[float], *,
                        laser_nm: Optional[float] = None, source: str = "", notes: str = "",
                        extra: Optional[Dict[str, Any]] = None) -> LibraryEntry:
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    order = np.argsort(x[m], kind="mergesort")
    meta = {"laser_nm": laser_nm, "source": source, "notes": notes,
            "added_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    if extra:
        meta.update({k: v for k, v in extra.items() if k not in meta or v})
    return LibraryEntry(id=LibraryEntry.new_id(), name=str(name),
                        x=[float(v) for v in x[m][order]], y=[float(v) for v in y[m][order]], meta=meta)


def load_library(path: Optional[str] = None) -> List[LibraryEntry]:
    path = path or LIBRARY_PATH
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if data.get("format") != LIBRARY_FORMAT:
        raise ValueError(f"{path} is not a PRISM Raman library.")
    out = []
    for rec in data.get("entries", []):
        try:
            out.append(LibraryEntry(id=str(rec["id"]), name=str(rec.get("name", "")),
                                    x=list(rec["x"]), y=list(rec["y"]), meta=dict(rec.get("meta", {}) or {})))
        except (KeyError, TypeError):
            continue
    return out


def save_library(entries: Sequence[LibraryEntry], path: Optional[str] = None) -> str:
    path = path or LIBRARY_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = {"format": LIBRARY_FORMAT, "version": LIBRARY_VERSION,
            "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
            "entries": [asdict(e) for e in entries]}
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, path)
    return path


def merge_entries(existing: Sequence[LibraryEntry], incoming: Sequence[LibraryEntry]) -> Tuple[List[LibraryEntry], int]:
    """Union by entry id (a re-imported pack never duplicates)."""
    out = list(existing)
    ids = {e.id for e in out}
    n_new = 0
    for e in incoming:
        if e.id not in ids:
            out.append(e); ids.add(e.id); n_new += 1
    return out, n_new


# ----------------------------------------------------------------------
# Similarity
# ----------------------------------------------------------------------

def preprocess(x: np.ndarray, y: np.ndarray, how: str = "raw", smooth_window: int = 9) -> np.ndarray:
    """raw | smooth (Savitzky-Golay, order 2) | d1 | d2 (SG derivatives)."""
    y = np.asarray(y, float)
    if how == "raw" or len(y) < 7:
        return y
    from scipy.signal import savgol_filter
    win = max(5, int(smooth_window) | 1)
    win = min(win, len(y) - (1 - len(y) % 2))
    if win < 5:
        return y
    dx = float(np.median(np.diff(x))) if len(x) > 1 else 1.0
    if how == "smooth":
        return savgol_filter(y, win, 2, mode="interp")
    if how == "d1":
        return savgol_filter(y, win, 2, deriv=1, delta=dx, mode="interp")
    if how == "d2":
        return savgol_filter(y, win, 3, deriv=2, delta=dx, mode="interp")
    raise ValueError(f"unknown preprocessing {how!r} (expected one of {PREPROCESS_CHOICES})")


def similarity(qx: np.ndarray, qy: np.ndarray, rx: np.ndarray, ry: np.ndarray, *,
               method: str = "pearson", how: str = "raw", smooth_window: int = 9,
               x_range: Optional[Tuple[float, float]] = None, min_overlap: float = 0.5) -> float:
    """Hit quality in [−1, 1] for Pearson (1 = identical shape) or in [0, 1]
    for the normalised Euclidean distance. NaN when the overlap of the two
    x-ranges (inside x_range) is shorter than `min_overlap` of the query's
    range, so a reference measured over a different window never scores."""
    qx = np.asarray(qx, float); qy = np.asarray(qy, float)
    rx = np.asarray(rx, float); ry = np.asarray(ry, float)
    mq = np.isfinite(qx) & np.isfinite(qy); mr = np.isfinite(rx) & np.isfinite(ry)
    qx, qy, rx, ry = qx[mq], qy[mq], rx[mr], ry[mr]
    if qx.size < 7 or rx.size < 7:
        return float("nan")
    oq = np.argsort(qx, kind="mergesort"); orr = np.argsort(rx, kind="mergesort")
    qx, qy, rx, ry = qx[oq], qy[oq], rx[orr], ry[orr]
    lo = max(qx[0], rx[0]); hi = min(qx[-1], rx[-1])
    if x_range is not None:
        lo = max(lo, min(x_range)); hi = min(hi, max(x_range))
    q_span = (min(qx[-1], max(x_range)) - max(qx[0], min(x_range))) if x_range is not None else (qx[-1] - qx[0])
    if hi <= lo or q_span <= 0 or (hi - lo) < min_overlap * q_span:
        return float("nan")
    grid = qx[(qx >= lo) & (qx <= hi)]
    if grid.size < 7:
        return float("nan")
    a = preprocess(grid, np.interp(grid, qx, qy), how, smooth_window)
    b = preprocess(grid, np.interp(grid, rx, ry), how, smooth_window)
    if method == "pearson":
        a = a - a.mean(); b = b - b.mean()
        na = float(np.sqrt(np.dot(a, a))); nb = float(np.sqrt(np.dot(b, b)))
        if na <= 0 or nb <= 0:
            return float("nan")
        return float(np.dot(a, b) / (na * nb))
    if method == "euclid":
        na = float(np.sqrt(np.dot(a, a))); nb = float(np.sqrt(np.dot(b, b)))
        if na <= 0 or nb <= 0:
            return float("nan")
        d = float(np.sqrt(np.sum((a / na - b / nb) ** 2)))
        return float(max(0.0, 1.0 - d / np.sqrt(2.0)))
    raise ValueError(f"unknown method {method!r} (expected one of {METHOD_CHOICES})")


def rank_library(qx: np.ndarray, qy: np.ndarray, entries: Sequence[LibraryEntry], *,
                 method: str = "pearson", how: str = "raw", smooth_window: int = 9,
                 x_range: Optional[Tuple[float, float]] = None, top_n: int = 25,
                 progress=None) -> List[Dict[str, Any]]:
    """Rank the library against a query: [{'entry', 'score'}, …], best
    first; entries with no usable overlap are left out."""
    out = []
    n = len(entries)
    for i, e in enumerate(entries):
        rx, ry = e.arrays()
        s = similarity(qx, qy, rx, ry, method=method, how=how, smooth_window=smooth_window, x_range=x_range)
        if np.isfinite(s):
            out.append({"entry": e, "score": float(s), "name": e.name, "id": e.id, "meta": e.meta})
        if progress and (i % 200 == 0 or i == n - 1):
            progress(i + 1, n, "library")
    out.sort(key=lambda r: r["score"], reverse=True)
    return out[:top_n]


def rank_rruff_by_similarity(qx: np.ndarray, qy: np.ndarray, records: Sequence[Dict[str, Any]], *,
                             method: str = "pearson", how: str = "raw", smooth_window: int = 9,
                             x_range: Optional[Tuple[float, float]] = None, top_n: int = 25,
                             progress=None) -> List[Dict[str, Any]]:
    """The same ranking over RRUFF index records (their raw spectra are read
    from `raw_path`). Records without a readable raw file are skipped."""
    from raman.rruff_science import parse_rruff_txt
    out = []
    n = len(records)
    for i, rec in enumerate(records):
        p = rec.get("raw_path")
        if not p or not os.path.isfile(p):
            continue
        try:
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                ref = parse_rruff_txt(f.read(), source_filename=os.path.basename(p))
        except Exception:
            continue
        if not len(ref.x):
            continue
        s = similarity(qx, qy, ref.x, ref.y, method=method, how=how, smooth_window=smooth_window, x_range=x_range)
        if np.isfinite(s):
            out.append({**rec, "score": float(s)})
        if progress and (i % 200 == 0 or i == n - 1):
            progress(i + 1, n, "RRUFF")
    out.sort(key=lambda r: r["score"], reverse=True)
    return out[:top_n]
