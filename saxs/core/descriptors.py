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

from core.np_compat import trapezoid
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
    return float(trapezoid(np.maximum(I[m] - C, 0) * q[m] ** 2, q[m]))


def excess_over_grain(q: np.ndarray, I: np.ndarray, K4: float, C: float, qmax: float, min_points: int = 8) -> Tuple[float, float, float]:
    """Excess over the grain baseline K4 q^-4 + C up to ``qmax``: (Q_exc of the positive excess,
    signed Q_exc, maximum excess fraction)."""
    q, I = np.asarray(q, float), np.asarray(I, float)
    m = (q <= qmax) & np.isfinite(I)
    if not np.isfinite(K4) or m.sum() < min_points:
        return float("nan"), float("nan"), float("nan")
    base = K4 * q ** -4 + C
    E = I - base
    return (float(trapezoid(np.maximum(E[m], 0) * q[m] ** 2, q[m])), float(trapezoid(E[m] * q[m] ** 2, q[m])),
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


def moving_average(y: np.ndarray, pts: int) -> np.ndarray:
    """``pts``-point moving average that keeps its normalisation at the ends: each output point is
    the mean of the samples that actually fall in the window. ``np.convolve(..., "same")`` divides
    by the full window everywhere, which drags the first and last points towards zero and fakes a
    rise at the start of a steeply decaying curve."""
    y = np.asarray(y, float)
    k = max(int(pts) | 1, 1)
    if k <= 1 or len(y) < 3:
        return y.copy()
    k = min(k, len(y) if len(y) % 2 else len(y) - 1)
    kernel = np.ones(k)
    num = np.convolve(y, kernel, mode="same")
    den = np.convolve(np.ones_like(y), kernel, mode="same")
    return num / den


def interior_peaks(q: np.ndarray, y: np.ndarray, qrange: Tuple[float, float], edge_pts: int = 3,
                   smooth_pts: int = 3) -> List[Tuple[float, float, int]]:
    """Local maxima of ``y`` inside ``qrange``, most prominent first: ``[(q, prominence, index)]``.

    "Interior" means at least ``edge_pts`` points from either end of the array: a monotone rise or
    fall that only turns over at the edge of the measured range is not a peak of the data, it is the
    edge. ``y`` is smoothed with a ``smooth_pts`` moving average first (the smoothing is applied to
    the whole array so the window is not truncated inside ``qrange``)."""
    from scipy.signal import find_peaks

    q, y = np.asarray(q, float), np.asarray(y, float)
    y = moving_average(y, smooth_pts)
    finite = np.isfinite(y)
    if finite.sum() < 3:
        return []
    ys = np.where(finite, y, np.nanmin(y[finite]) - 1.0)
    pk, pr = find_peaks(ys, prominence=0.0)
    edge = max(int(edge_pts), 1)
    out = [(float(q[i]), float(p), int(i)) for i, p in zip(pk, pr["prominences"])
           if edge <= i < len(y) - edge and qrange[0] <= q[i] <= qrange[1] and finite[i]]
    return sorted(out, key=lambda t: -t[1])


def local_slopes_err(q: np.ndarray, I: np.ndarray, window_dec: float = 0.15) -> np.ndarray:
    """Standard error of :func:`local_slopes`, from the scatter of the points about each local
    regression: s / (sqrt(N) sd(log q)). NaN wherever the slope itself is NaN."""
    q, I = np.asarray(q, float), np.asarray(I, float)
    lq, lI = np.log10(q), np.log10(np.maximum(I, 1e-300))
    e = np.full(len(q), np.nan)
    for i in range(len(q)):
        m = np.abs(lq - lq[i]) <= window_dec / 2
        if m.sum() >= 4 and np.all(I[m] > 0):
            x, y = lq[m], lI[m]
            a, b = np.polyfit(x, y, 1)
            resid = y - (a * x + b)
            sxx = np.sum((x - x.mean()) ** 2)
            if sxx > 0:
                e[i] = float(np.sqrt(np.sum(resid ** 2) / max(len(x) - 2, 1) / sxx))
    return e


def slope_recovery(q: np.ndarray, I: np.ndarray, knee_range: Tuple[float, float], q_limit: float,
                   window_dec: float = 0.30, edge_pts: int = 2) -> Dict[str, float]:
    """Does the local log-log slope shallow and then STEEPEN again? The model-free fingerprint of a
    population of finite size, and the one thing a background cannot imitate.

A power law, a flat background and a rising tail can only make d log I / d log q *rise* with q
    (towards 0 and beyond): their sum is monotone in slope, whatever their exponents and amplitudes.
    A population of radius of gyration R_g does the opposite around its own Guinier knee: it
    flattens the slope at q ~ 1/R_g and steepens it again to its Porod exponent just above, so the
    slope has a local maximum followed by a decrease. The recovery is the size of that decrease, in
    slope units; for a level that dominates locally it approaches the level's own exponent (2 to
    4.5), and it is 0 within the noise when there is no level.

    Every level has such a transition, the coarse one included, so what makes this specific is
    *where* the knee is looked for: only inside ``knee_range``, and only at maxima that are interior
    to the measured range. A grain level of tens of nanometres has its knee one or two decades below
    that window, and inside the window it contributes a monotone Porod decay.

    ``knee_range`` is where a level of the allowed radii has its knee (typically
    ``[1/Rg_max, 1/Rg_min]``) and ``q_limit`` how far above it the steepening is looked for (the
    level's Porod region, typically ``3/Rg_min``). Candidate knees are INTERIOR local maxima of the
    slope, at least ``edge_pts`` points from either end: this is what keeps the grain level's own
    Guinier transition at the low-q edge of the measured range from counting as a knee.

    Returns ``{"recovery", "recovery_err", "q_knee", "q_steep", "knee_prominence", "n_knee",
    "n_steep"}``; the recovery is 0 and the q NaN when the slope has no interior maximum in the
    window. ``recovery_err`` combines the standard errors of the two slopes in quadrature."""
    q, I = np.asarray(q, float), np.asarray(I, float)
    out = {"recovery": 0.0, "recovery_err": float("nan"), "q_knee": float("nan"), "q_steep": float("nan"),
           "knee_prominence": 0.0, "n_knee": float("nan"), "n_steep": float("nan")}
    n = local_slopes(q, I, window_dec)
    if not np.isfinite(n).any():
        return out
    err = local_slopes_err(q, I, window_dec)
    guarded = np.where(np.isfinite(n), n, -np.inf)
    for q_k, prom, i in interior_peaks(q, guarded, knee_range, edge_pts=edge_pts, smooth_pts=1):
        after = np.where((np.arange(len(q)) > i) & (q <= q_limit) & np.isfinite(n))[0]
        if not len(after):
            continue
        j = int(after[int(np.nanargmin(n[after]))])
        rec = float(n[i] - n[j])
        if rec > out["recovery"]:
            e = float(np.sqrt(np.nansum([err[i] ** 2, err[j] ** 2]))) if np.isfinite(err[i]) or np.isfinite(err[j]) else float("nan")
            out.update(recovery=rec, recovery_err=e, q_knee=float(q_k), q_steep=float(q[j]),
                       knee_prominence=float(prom), n_knee=float(n[i]), n_steep=float(n[j]))
    return out


def largest_rise(q: np.ndarray, I: np.ndarray, background: Union[float, np.ndarray], q_limit: float,
                 bg_max: float = 0.25, smooth_pts: int = 3) -> Dict[str, float]:
    """Largest cumulative UPWARD excursion of the background-subtracted intensity, in log10 units.

    A sum of positive Beaucage levels over a flat background is monotone decreasing: the scattered
    intensity of a disordered material cannot grow with q below the amorphous halo. A rise therefore
    means the reduction failed there - a parasitic capillary lobe, or an empty subtraction amplified
    by a transmission of 1e-5 - and no size may be read from that curve, however large the feature.

    Measured from the first point (an artefact usually sits at the low-q edge, which is exactly where
    an interior-peak test cannot see it) up to ``q_limit`` or to the point where the background
    reaches ``bg_max`` of the intensity, whichever comes first. Returns ``{"rise", "q_start",
    "q_end", "n_points"}``; ``rise`` is 0 for a monotone curve."""
    q, I = np.asarray(q, float), np.asarray(I, float)
    exc = I - np.asarray(background, float)
    bg = np.asarray(background, float) * np.ones_like(q)
    ok = np.isfinite(exc) & (exc > 0) & (q <= q_limit) & (bg <= bg_max * np.maximum(I, 1e-300))
    out = {"rise": 0.0, "q_start": float("nan"), "q_end": float("nan"), "n_points": int(ok.sum())}
    if ok.sum() < 4:
        return out
    y = moving_average(np.log10(exc[ok]), smooth_pts)
    qq = q[ok]
    run = 0.0
    start = 0
    for i in range(1, len(y)):
        step = y[i] - y[i - 1]
        if step > 0:
            if run == 0.0:
                start = i - 1
            run += step
            if run > out["rise"]:
                out.update(rise=float(run), q_start=float(qq[start]), q_end=float(qq[i]))
        else:
            run = 0.0
    return out


def knee_signature(q: np.ndarray, I: np.ndarray, background: Union[float, np.ndarray], qrange: Tuple[float, float],
                   edge_pts: int = 3, smooth_pts: int = 3) -> Dict[str, float]:
    """Guinier-knee signature of a population of finite size in the background-subtracted curve.

    A population of radius of gyration R_g turns the profile over at q ~ 1/R_g: below the knee the
    intensity is flat (Guinier), above it decays as a power law. In the Kratky representation
    (I - background) q^2 that turn-over IS a local maximum, and in the Porod representation
    (I - background) q^4 it is a shoulder. Either is the model-free fingerprint of a level, and
    neither can be produced by a power law of any exponent, which is what distinguishes a real
    population from a fit absorbing a slight mismatch of another term.

    Returns the most prominent interior maximum (log10 units) of each representation inside
    ``qrange`` (``kratky_q``, ``kratky_prominence``, ``porod_q``, ``porod_prominence``); the
    prominence is 0 and the q NaN when there is none."""
    q, I = np.asarray(q, float), np.asarray(I, float)
    exc = I - np.asarray(background, float)
    ok = np.isfinite(exc) & (exc > 0)
    out = {"kratky_q": float("nan"), "kratky_prominence": 0.0, "porod_q": float("nan"), "porod_prominence": 0.0}
    if ok.sum() < 2 * max(int(edge_pts), 1) + 3:
        return out
    for key, w in (("kratky", 2), ("porod", 4)):
        y = np.where(ok, np.log10(np.maximum(exc, 1e-300) * q ** w), np.nan)
        pk = interior_peaks(q, y, qrange, edge_pts=edge_pts, smooth_pts=smooth_pts)
        if pk:
            out[f"{key}_q"], out[f"{key}_prominence"] = pk[0][0], pk[0][1]
    return out


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
    either end) is significant when its prominence exceeds ``min_prominence_sigma`` times the noise
    of the smoothed value and ``min_relative_prominence`` of the value itself. The noise is the
    larger of the propagated sigma and the local scatter of the data about the smoothed curve
    (robust MAD over three windows), so an underestimated sigma (opaque loadings) cannot promote
    noise to a maximum. Returns ``{name: {"q_max", "prominence", "prominence_sigma",
    "relative_prominence", "significant"}}`` (the most prominent significant maximum, else the most
    prominent one with ``significant=False``, else ``q_max=None``).

    Reading: a maximum of I itself is what a characteristic spacing produces (:func:`has_maximum`);
    a maximum of I q^2 or I q^4 without one in I is the Guinier knee of a level, not a spacing
    (:func:`has_kratky_maximum`)."""
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
        # data-driven noise floor: local scatter about the smoothed curve (MAD over 3 k points), per smoothed value
        dev = np.abs(y[k // 2: len(q) - k // 2] - ys)
        wide = 3 * k
        scatter = np.array([1.4826 * np.median(dev[max(0, i - wide // 2): i + wide // 2 + 1]) for i in range(len(dev))]) / np.sqrt(k)
        ss = np.maximum(ss, scatter)
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
    """A maximum of I itself (what a characteristic spacing produces)."""
    return bool(maxima.get("I", {}).get("significant"))


def has_kratky_maximum(maxima: Dict[str, Dict[str, object]]) -> bool:
    """A maximum of I q^2: with none in I it is the Guinier knee of a level, not a spacing."""
    return bool(maxima.get("Iq2", {}).get("significant"))


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
    has_kratky_maximum: bool = False
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
        r["has_kratky_maximum"] = self.has_kratky_maximum
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
    res.has_kratky_maximum = has_kratky_maximum(res.maxima)
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
