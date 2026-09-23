"""Model-free descriptors of a reduced SAXS curve (per mm of material, usable range).

Everything a curve says before any model is fitted: log-rebinned profile, local log-log slopes,
the low-q power-law exponent and Porod constant, the high-q plateau and its ratio to the scaled
empty, the excess invariant over the grain baseline, Porod-plot shoulders, a Kratky maximum and an
explicit test for a maximum in I, I q^2 and I q^4 with a significance against the propagated noise.
The numbers and defaults follow the September 2026 rethink (``phase2_modelfree.py``); the maximum
test is new. Pure numpy / scipy; sample identities are strings carried in the curve metadata.

Input: a :class:`~saxs.core.curve.Curve` on its usable range (``ReducedSample.usable_curve``), whose
``metadata["columns"]["ratio"]`` (sample / scaled empty) feeds the plateau robustness check.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .curve import Curve

Progress = Optional[Callable[[int, int, str], None]]
REPRESENTATIONS = {"I": 0, "Iq2": 2, "Iq4": 4}


@dataclass
class DescriptorOptions:
    per_decade: int = 40
    slope_window_dec: float = 0.15
    slopes_at: Tuple[float, ...] = (6e-3, 1e-2, 2e-2, 4e-2)
    lowq: Tuple[float, float] = (4.5e-3, 1.2e-2)
    hiq: Tuple[float, float] = (0.15, 0.20)
    shoulder_range: Tuple[float, float] = (7e-3, 8e-2)
    excess_qmax: float = 0.1
    probes_at: Tuple[float, ...] = (0.01, 0.02, 0.05)
    min_points: int = 20
    shoulder_prominence: float = 0.04       # log10 units, on the 3-point smoothed Porod plot
    kratky_prominence: float = 0.05         # log10 units
    maximum_smooth_pts: int = 7
    maximum_min_prominence_sigma: float = 3.0
    maximum_min_relative_prominence: float = 0.05
    maximum_weights: Tuple[str, ...] = ("I", "Iq2", "Iq4")

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, object]]) -> "DescriptorOptions":
        out = cls()
        for k, v in dict(d or {}).items():
            if not hasattr(out, k):
                raise KeyError(f"unknown descriptor option {k!r}")
            if isinstance(getattr(out, k), tuple):
                v = tuple(v)
            setattr(out, k, v)
        return out


# ------------------------------------------------------------------------------------ primitives
def rebin_log(q: np.ndarray, I: np.ndarray, sigma: Optional[np.ndarray], per_decade: int = 40) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Logarithmic rebinning: ``per_decade`` bins per decade, geometric mean q, mean I, sigma summed in
    quadrature over the bin (the rethink's ``rebin_log``)."""
    q = np.asarray(q, float)
    I = np.asarray(I, float)
    sigma = np.zeros_like(I) if sigma is None else np.asarray(sigma, float)
    if q.size == 0:
        return q, I, sigma
    edges = np.geomspace(q.min(), q.max() * 1.0001, int(np.log10(q.max() / q.min()) * per_decade) + 2)
    idx = np.digitize(q, edges) - 1
    qb, Ib, sb = [], [], []
    for k in range(len(edges) - 1):
        m = idx == k
        n = int(m.sum())
        if n == 0:
            continue
        qb.append(np.exp(np.mean(np.log(q[m]))))
        Ib.append(np.mean(I[m]))
        sb.append(np.sqrt(np.sum(sigma[m] ** 2)) / n)
    return np.array(qb), np.array(Ib), np.array(sb)


def local_slopes(q: np.ndarray, I: np.ndarray, window_dec: float = 0.15) -> np.ndarray:
    """d log10 I / d log10 q by a sliding regression over ``window_dec`` decades (negative for a
    decaying curve; NaN where fewer than 4 positive points fall in the window)."""
    q = np.asarray(q, float)
    I = np.asarray(I, float)
    lq, lI = np.log10(q), np.log10(np.maximum(I, 1e-300))
    n = np.full(len(q), np.nan)
    for i in range(len(q)):
        m = np.abs(lq - lq[i]) <= window_dec / 2
        if m.sum() >= 4 and np.all(I[m] > 0):
            n[i] = np.polyfit(lq[m], lI[m], 1)[0]
    return n


def slope_at(q: np.ndarray, slopes: np.ndarray, q0: float, tol_dec: float = 0.1) -> float:
    if len(q) == 0:
        return float("nan")
    j = int(np.argmin(np.abs(np.log(q / q0))))
    return float(slopes[j]) if abs(np.log(q[j] / q0)) < tol_dec * np.log(10) else float("nan")


def value_at(q: np.ndarray, y: np.ndarray, q0: float, tol_dec: float = 0.1) -> float:
    if len(q) == 0:
        return float("nan")
    j = int(np.argmin(np.abs(np.log(q / q0))))
    return float(y[j]) if abs(np.log(q[j] / q0)) < tol_dec * np.log(10) else float("nan")


def lowq_exponent(q: np.ndarray, I: np.ndarray, sigma: np.ndarray, qrange: Tuple[float, float],
                  min_points: int = 6) -> Tuple[float, float, float, int]:
    """Weighted power law I = K q^-n over ``qrange`` (weights 1/sigma_log): (n, n_err, K, n_points)."""
    q, I, sigma = (np.asarray(x, float) for x in (q, I, sigma))
    lo, hi = qrange
    m = (q >= lo) & (q <= hi) & (I > 0) & np.isfinite(sigma)
    if m.sum() < min_points:
        return float("nan"), float("nan"), float("nan"), int(m.sum())
    x, y, w = np.log10(q[m]), np.log10(I[m]), I[m] / np.maximum(sigma[m], 1e-300)
    coef, cov = np.polyfit(x, y, 1, w=w, cov="unscaled")
    return float(-coef[0]), float(np.sqrt(cov[0, 0])), float(10 ** coef[1]), int(m.sum())


def band_median(q: np.ndarray, y: np.ndarray, qrange: Tuple[float, float], min_points: int = 3) -> float:
    q, y = np.asarray(q, float), np.asarray(y, float)
    m = (q >= qrange[0]) & (q <= qrange[1]) & np.isfinite(y)
    return float(np.median(y[m])) if m.sum() >= min_points else float("nan")


def plateau(q: np.ndarray, I: np.ndarray, qrange: Tuple[float, float], min_points: int = 3) -> Tuple[float, float]:
    """(median, 1.4826 MAD) of I inside ``qrange``: the flat high-q level."""
    C = band_median(q, I, qrange, min_points)
    if not np.isfinite(C):
        return C, float("nan")
    q, I = np.asarray(q, float), np.asarray(I, float)
    m = (q >= qrange[0]) & (q <= qrange[1]) & np.isfinite(I)
    return C, float(1.4826 * np.median(np.abs(I[m] - C)))


def porod_constant(q: np.ndarray, I: np.ndarray, qrange: Tuple[float, float], min_points: int = 4) -> float:
    """K4 = <I q^4> over ``qrange`` (the fixed-exponent Porod constant)."""
    q, I = np.asarray(q, float), np.asarray(I, float)
    m = (q >= qrange[0]) & (q <= qrange[1]) & np.isfinite(I)
    return float(np.mean(I[m] * q[m] ** 4)) if m.sum() >= min_points else float("nan")


def invariant(q: np.ndarray, I: np.ndarray, qrange: Optional[Tuple[float, float]] = None, C: float = 0.0,
              min_points: int = 8) -> float:
    """Q = int max(I - C, 0) q^2 dq over ``qrange`` (trapezoid)."""
    q, I = np.asarray(q, float), np.asarray(I, float)
    m = np.isfinite(I)
    if qrange is not None:
        m &= (q >= qrange[0]) & (q <= qrange[1])
    if m.sum() < min_points:
        return float("nan")
    return float(np.trapezoid(np.maximum(I[m] - C, 0) * q[m] ** 2, q[m]))


def excess_over_grain(q: np.ndarray, I: np.ndarray, K4: float, C: float, qmax: float, min_points: int = 8) -> Tuple[float, float, float]:
    """Excess over the grain baseline K4 q^-4 + C up to ``qmax``: (Q_exc of the positive excess,
    signed Q_exc, maximum excess fraction)."""
    q, I = np.asarray(q, float), np.asarray(I, float)
    m = (q <= qmax) & np.isfinite(I)
    if not np.isfinite(K4) or m.sum() < min_points:
        return float("nan"), float("nan"), float("nan")
    base = K4 * q ** -4 + C
    E = I - base
    return (float(np.trapezoid(np.maximum(E[m], 0) * q[m] ** 2, q[m])), float(np.trapezoid(E[m] * q[m] ** 2, q[m])),
            float(np.nanmax(E[m] / base[m])))


def porod_shoulders(q: np.ndarray, I: np.ndarray, C: float, qrange: Tuple[float, float], prominence: float = 0.04,
                    smooth_pts: int = 3, min_points: int = 8) -> List[Tuple[float, float, float]]:
    """Local maxima of the Porod plot (I - C) q^4 (log10, ``smooth_pts`` moving average) inside
    ``qrange``: [(q, prominence, size_nm = 2 pi / q / 10)], most prominent first."""
    from scipy.signal import find_peaks

    q, I = np.asarray(q, float), np.asarray(I, float)
    P4 = np.maximum(I - C, 1e-300) * q ** 4
    ms = (q >= qrange[0]) & (q <= qrange[1]) & (I - C > 0)
    if ms.sum() < min_points:
        return []
    lp = np.log10(P4[ms])
    if smooth_pts and smooth_pts > 1:
        lp = np.convolve(lp, np.ones(smooth_pts) / smooth_pts, mode="same")
    pk, pr = find_peaks(lp, prominence=prominence)
    out = [(float(q[ms][k]), float(p), float(2 * np.pi / q[ms][k] / 10)) for k, p in zip(pk, pr["prominences"])]
    return sorted(out, key=lambda t: -t[1])


def kratky_maximum(q: np.ndarray, I: np.ndarray, C: float, qrange: Tuple[float, float], prominence: float = 0.05,
                   min_points: int = 8) -> float:
    """q of the most prominent local maximum of (I - C) q^2 (log10) inside ``qrange``, NaN when none."""
    from scipy.signal import find_peaks

    q, I = np.asarray(q, float), np.asarray(I, float)
    ms = (q >= qrange[0]) & (q <= qrange[1]) & (I - C > 0)
    if ms.sum() < min_points:
        return float("nan")
    K2 = np.maximum(I[ms] - C, 1e-300) * q[ms] ** 2
    pk, pr = find_peaks(np.log10(K2), prominence=prominence)
    return float(q[ms][pk[int(np.argmax(pr["prominences"]))]]) if len(pk) else float("nan")


def maximum_test(q: np.ndarray, I: np.ndarray, sigma: Optional[np.ndarray], weights: Sequence[str] = ("I", "Iq2", "Iq4"),
                 smooth_pts: int = 7, min_prominence_sigma: float = 3.0, min_relative_prominence: float = 0.05,
                 qrange: Optional[Tuple[float, float]] = None) -> Dict[str, Dict[str, object]]:
    """Explicit test for a maximum in I q^w (w = 0, 2, 4). Each representation is smoothed with a
    ``smooth_pts`` moving average; a local maximum (interior, at least ``smooth_pts`` points from
    either end) is significant when its prominence exceeds ``min_prominence_sigma`` times the
    propagated noise of the smoothed value and ``min_relative_prominence`` of the value itself.
    Returns ``{name: {"q_max", "prominence", "prominence_sigma", "relative_prominence", "significant"}}``
    (the most prominent significant maximum, else the most prominent one with ``significant=False``,
    else ``q_max=None``)."""
    from scipy.signal import find_peaks

    q, I = np.asarray(q, float), np.asarray(I, float)
    sigma = np.zeros_like(I) if sigma is None else np.asarray(sigma, float)
    m = np.isfinite(I) & (q > 0)
    if qrange is not None:
        m &= (q >= qrange[0]) & (q <= qrange[1])
    q, I, sigma = q[m], I[m], sigma[m]
    k = max(int(smooth_pts) | 1, 1)                     # odd window
    kernel = np.ones(k) / k
    out: Dict[str, Dict[str, object]] = {}
    for name in weights:
        w = REPRESENTATIONS[name]
        empty = {"q_max": None, "prominence": 0.0, "prominence_sigma": float("nan"), "relative_prominence": 0.0, "significant": False}
        if len(q) < 3 * k:
            out[name] = dict(empty)
            continue
        y = I * q ** w
        ys = np.convolve(y, kernel, mode="valid")           # length n - k + 1, centred
        ss = np.sqrt(np.convolve((sigma * q ** w) ** 2, kernel ** 2, mode="valid"))
        qs = q[k // 2: len(q) - k // 2]
        pk, pr = find_peaks(ys, prominence=0.0)
        best_sig, best_any = None, None
        for j, prom in zip(pk, pr["prominences"]):
            if j < k or j > len(ys) - 1 - k:                 # too close to an edge to be a maximum
                continue
            nsig = prom / ss[j] if ss[j] > 0 else float("inf")
            rel = prom / ys[j] if ys[j] > 0 else 0.0
            rec = {"q_max": float(qs[j]), "prominence": float(prom), "prominence_sigma": float(nsig),
                   "relative_prominence": float(rel), "significant": bool(nsig >= min_prominence_sigma and rel >= min_relative_prominence)}
            if best_any is None or prom > best_any["prominence"]:
                best_any = rec
            if rec["significant"] and (best_sig is None or prom > best_sig["prominence"]):
                best_sig = rec
        out[name] = best_sig or best_any or dict(empty)
    return out


def has_maximum(maxima: Dict[str, Dict[str, object]]) -> bool:
    """A curve has a maximum when the test is significant in I or in I q^2."""
    return bool(maxima.get("I", {}).get("significant") or maxima.get("Iq2", {}).get("significant"))


# ------------------------------------------------------------------------------------ results
VALUE_COLUMNS = ["n_low", "n_low_err", "K_low", "n_low_pts", "K4", "C_hi", "C_hi_mad", "ratio_hi", "ratio_low", "q_shoulder",
                 "shoulder_prom", "shoulder_size_nm", "q_kratky_max", "Q_exc", "Q_exc_signed", "excess_frac_max", "Q_tot",
                 "rho_e", "K4_over_rhoe2"]
META_COLUMNS = ["T_used", "t_eff_mm", "mu_t", "cls", "q_min_use", "q_max_use", "n_use"]


@dataclass
class DescriptorResult:
    sample: str
    cls: str
    n_use: int
    values: Dict[str, float] = field(default_factory=dict)
    maxima: Dict[str, Dict[str, object]] = field(default_factory=dict)
    has_maximum: bool = False
    shoulders: List[Tuple[float, float, float]] = field(default_factory=list)
    rebinned: Dict[str, np.ndarray] = field(default_factory=dict)      # q, I, sigma, slopes, C
    meta: Dict[str, object] = field(default_factory=dict)
    options: Dict[str, object] = field(default_factory=dict)

    def row(self) -> Dict[str, object]:
        r: Dict[str, object] = {"sample": self.sample}
        r.update({k: self.meta.get(k, float("nan")) for k in META_COLUMNS})
        r["cls"] = self.cls
        r["n_use"] = self.n_use
        r.update(self.values)
        r["has_maximum"] = self.has_maximum
        for name, d in self.maxima.items():
            r[f"max_q_{name}"] = d.get("q_max") if d.get("q_max") is not None else float("nan")
            r[f"max_prom_sigma_{name}"] = d.get("prominence_sigma", float("nan"))
            r[f"max_significant_{name}"] = bool(d.get("significant", False))
        return r


def _meta_float(md: Dict[str, object], *keys: str) -> float:
    for k in keys:
        v = md.get(k)
        if v is None:
            continue
        try:
            return float(str(v).split("+-")[0].split()[0])
        except (ValueError, IndexError):
            continue
    return float("nan")


def describe_curve(curve: Curve, options: Optional[DescriptorOptions] = None, rho_e: Optional[float] = None) -> DescriptorResult:
    """Descriptors of one curve (already restricted to its usable range; a ``usable`` column in
    ``metadata['columns']`` is honoured when present)."""
    o = options or DescriptorOptions()
    md = curve.metadata or {}
    cols = {k: np.asarray(v, float) for k, v in (md.get("columns") or {}).items()}
    name = str(md.get("sample") or curve.name)
    cls = str(md.get("class_mu_t", ""))
    meta = {"T_used": _meta_float(md, "transmission_T", "transmission_T_used (footprint r<16px, vertical +-30 deg)"),
            "t_eff_mm": _meta_float(md, "t_eff_mm"), "mu_t": _meta_float(md, "mu_t_eff"), "cls": cls,
            "q_min_use": _meta_float(md, "q_min_use"), "q_max_use": _meta_float(md, "q_max_use")}
    q = np.asarray(curve.q, float)
    I = np.asarray(curve.intensity, float)
    sig = np.asarray(curve.sigma, float) if curve.sigma is not None else np.zeros_like(I)
    keep = np.isfinite(I) & np.isfinite(q) & (q > 0)
    if "usable" in cols and len(cols["usable"]) == len(q):
        keep &= cols["usable"].astype(bool)
    ratio = cols.get("ratio")
    if ratio is not None and len(ratio) != len(q):
        ratio = None
    q, I, sig = q[keep], I[keep], sig[keep]
    ratio = ratio[keep] if ratio is not None else None
    meta["n_use"] = int(len(q))
    res = DescriptorResult(sample=name, cls=cls, n_use=int(len(q)), meta=meta, options=o.to_dict())
    if len(q) < o.min_points:
        res.values = {k: float("nan") for k in VALUE_COLUMNS}
        return res
    qb, Ib, sb = rebin_log(q, I, sig, o.per_decade)
    slopes = local_slopes(qb, Ib, o.slope_window_dec)
    v: Dict[str, float] = {}
    for qq in o.slopes_at:
        v[f"n_at_{qq:g}"] = slope_at(qb, slopes, qq)
    n_low, n_err, K, npts = lowq_exponent(qb, Ib, sb, o.lowq)
    v.update(n_low=n_low, n_low_err=n_err, K_low=K, n_low_pts=npts)
    v["K4"] = porod_constant(qb, Ib, o.lowq)
    v["C_hi"], v["C_hi_mad"] = plateau(qb, Ib, o.hiq)
    v["ratio_hi"] = band_median(q, ratio, o.hiq) if ratio is not None else float("nan")
    v["ratio_low"] = band_median(q, ratio, o.lowq) if ratio is not None else float("nan")
    C = v["C_hi"] if np.isfinite(v["C_hi"]) else 0.0
    res.shoulders = porod_shoulders(qb, Ib, C, o.shoulder_range, o.shoulder_prominence)
    if res.shoulders:
        v["q_shoulder"], v["shoulder_prom"], v["shoulder_size_nm"] = res.shoulders[0]
    else:
        v["q_shoulder"] = v["shoulder_prom"] = v["shoulder_size_nm"] = float("nan")
    v["q_kratky_max"] = kratky_maximum(qb, Ib, C, o.shoulder_range, o.kratky_prominence)
    v["Q_exc"], v["Q_exc_signed"], v["excess_frac_max"] = excess_over_grain(qb, Ib, v["K4"], C, o.excess_qmax)
    v["Q_tot"] = invariant(qb, Ib, (qb.min(), o.excess_qmax), C)
    for qq in o.probes_at:
        v[f"I_mm_at_{qq:g}"] = value_at(qb, Ib, qq)
    v["rho_e"] = float(rho_e) if rho_e is not None else float("nan")
    v["K4_over_rhoe2"] = v["K4"] / rho_e ** 2 if (rho_e is not None and rho_e > 0) else float("nan")
    res.values = v
    res.maxima = maximum_test(qb, Ib, sb, o.maximum_weights, o.maximum_smooth_pts, o.maximum_min_prominence_sigma,
                              o.maximum_min_relative_prominence)
    res.has_maximum = has_maximum(res.maxima)
    res.rebinned = {"q": qb, "I": Ib, "sigma": sb, "slopes": slopes, "C": C}
    return res


def describe_batch(curves: Iterable[Curve], options: Optional[DescriptorOptions] = None,
                   rho_e: Optional[Union[Dict[str, float], Callable[[str], Optional[float]]]] = None,
                   progress: Progress = None) -> Tuple[List[DescriptorResult], pd.DataFrame]:
    """Descriptors of many curves -> (results, table with one row per curve, phase2 column names)."""
    curves = list(curves)
    results: List[DescriptorResult] = []
    n = len(curves)
    for i, c in enumerate(curves):
        name = str((c.metadata or {}).get("sample") or c.name)
        if progress is not None:
            progress(i, n, name)
        if callable(rho_e):
            re_ = rho_e(name)
        elif isinstance(rho_e, dict):
            re_ = rho_e.get(name)
        else:
            re_ = None
        results.append(describe_curve(c, options, re_))
    if progress is not None:
        progress(n, n, "done")
    return results, descriptors_table(results)


def descriptors_table(results: Iterable[DescriptorResult]) -> pd.DataFrame:
    rows = [r.row() for r in results]
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["sample"] + META_COLUMNS + VALUE_COLUMNS)
