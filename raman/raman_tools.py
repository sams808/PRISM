"""
raman_tools.py — Raman-side spectrum tools (framework-agnostic): the peak
table with labels, x-axis conversions between wavelength / absolute
wavenumber / Raman shift / photon energy, x-axis calibration against
reference lines, and the single-spectrum operations the Calculations
workspace exposes for Raman work (normalise to a value, remove a peak,
subtract a range mean, SNV, MSC).

All of these existed in Spectragryph; they are here so a Raman user never
needs that program again. Conventions:
  - Raman shift Δν (cm⁻¹) = 1e7/λ_exc(nm) − 1e7/λ(nm); the intensity axis
    is left as measured (no Jacobian) unless asked for.
  - "FWHM" means the full width at half maximum, measured on the data.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.np_compat import trapezoid

HC_EV_NM = 1239.841984  # h·c in eV·nm
X_UNITS = ("Raman shift (cm⁻¹)", "Wavelength (nm)", "Wavenumber (cm⁻¹)", "Photon energy (eV)")


# ----------------------------------------------------------------------
# Peak table
# ----------------------------------------------------------------------

@dataclass
class PeakRow:
    position: float      # refined apex (parabolic through the three top points)
    height: float        # y at the apex, as measured
    prominence: float
    fwhm: float          # full width at half prominence, interpolated
    area: float          # trapezoid area above the local half-prominence base, between the width ends
    left: float
    right: float
    index: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _parabolic_apex(x: np.ndarray, y: np.ndarray, i: int) -> Tuple[float, float]:
    if i <= 0 or i >= len(x) - 1:
        return float(x[i]), float(y[i])
    y0, y1, y2 = float(y[i - 1]), float(y[i]), float(y[i + 1])
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-30:
        return float(x[i]), y1
    delta = 0.5 * (y0 - y2) / denom
    delta = float(np.clip(delta, -1.0, 1.0))
    dx = float(x[i + 1] - x[i]) if delta >= 0 else float(x[i] - x[i - 1])
    return float(x[i] + delta * dx), float(y1 - 0.25 * (y0 - y2) * delta)


def peak_table(x: Sequence[float], y: Sequence[float], *, threshold_frac: float = 0.05,
               min_prominence_frac: float = 0.02, min_distance: float = 0.0,
               max_peaks: int = 50, x_range: Optional[Tuple[float, float]] = None,
               inverted: bool = False) -> List[PeakRow]:
    """Peaks of a spectrum as a table, Spectragryph's "Peaks & FWHM".

    threshold_frac: a peak must rise above min(y) + threshold_frac·(max−min)
    (the default 5 % keeps noise ripples out of the table);
    min_prominence_frac: minimum prominence as a fraction of the range;
    min_distance: minimum separation in x units; inverted finds minima
    (transmission-style data). Sorted by position."""
    from scipy.signal import find_peaks, peak_widths
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 5:
        return []
    order = np.argsort(x, kind="mergesort")
    x, y = x[order], y[order]
    if x_range is not None:
        lo, hi = min(x_range), max(x_range)
        sel = (x >= lo) & (x <= hi)
        if sel.sum() < 5:
            return []
        x, y = x[sel], y[sel]
    yy = -y if inverted else y
    rng = float(np.max(yy) - np.min(yy))
    if rng <= 0:
        return []
    height = float(np.min(yy)) + float(threshold_frac) * rng
    step = float(np.median(np.diff(x))) if x.size > 1 else 1.0
    distance = max(1, int(round(min_distance / step))) if min_distance > 0 and step > 0 else 1
    idx, props = find_peaks(yy, height=height, prominence=max(1e-12, float(min_prominence_frac) * rng),
                            distance=distance)
    if idx.size == 0:
        return []
    widths, _h, lips, rips = peak_widths(yy, idx, rel_height=0.5)
    rows: List[PeakRow] = []
    for k, i in enumerate(idx):
        pos, apex = _parabolic_apex(x, yy, int(i))
        li, ri = float(lips[k]), float(rips[k])
        left = float(np.interp(li, np.arange(x.size), x))
        right = float(np.interp(ri, np.arange(x.size), x))
        base = float(yy[i] - props["prominences"][k])
        sel = (x >= left) & (x <= right)
        area = float(trapezoid(np.clip(yy[sel] - base, 0, None), x[sel])) if sel.sum() >= 2 else 0.0
        rows.append(PeakRow(position=pos, height=float(-apex if inverted else apex),
                            prominence=float(props["prominences"][k]), fwhm=abs(right - left),
                            area=area, left=left, right=right, index=int(i)))
    rows.sort(key=lambda r: r.prominence, reverse=True)
    rows = rows[:max_peaks]
    rows.sort(key=lambda r: r.position)
    return rows


def peak_table_text(rows: Sequence[PeakRow], *, digits: int = 1, sep: str = "\t") -> str:
    head = sep.join(["position", "height", "FWHM", "area", "prominence"])
    body = [sep.join([f"{r.position:.{digits}f}", f"{r.height:.4g}", f"{r.fwhm:.{digits}f}",
                      f"{r.area:.4g}", f"{r.prominence:.4g}"]) for r in rows]
    return "\n".join([head] + body)


# ----------------------------------------------------------------------
# x-axis conversions
# ----------------------------------------------------------------------

def wavelength_to_raman_shift(lambda_nm: np.ndarray, laser_nm: float) -> np.ndarray:
    lam = np.asarray(lambda_nm, float)
    with np.errstate(divide="ignore"):
        return 1e7 / float(laser_nm) - 1e7 / lam


def raman_shift_to_wavelength(shift_cm: np.ndarray, laser_nm: float) -> np.ndarray:
    s = np.asarray(shift_cm, float)
    with np.errstate(divide="ignore"):
        return 1e7 / (1e7 / float(laser_nm) - s)


def convert_x_axis(x: Sequence[float], y: Sequence[float], from_unit: str, to_unit: str, *,
                   laser_nm: Optional[float] = None, jacobian: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Convert the x axis between the units of X_UNITS. Raman shift needs
    the excitation wavelength. With jacobian=True the intensity is scaled
    by |dx_old/dx_new| so the integral is preserved (off by default: an
    instrument's counts-per-pixel are what people compare). The result is
    sorted ascending in the new unit."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if from_unit not in X_UNITS or to_unit not in X_UNITS:
        raise ValueError(f"units must be one of {X_UNITS}")
    if from_unit == to_unit:
        return x.copy(), y.copy()
    needs_laser = "Raman" in from_unit or "Raman" in to_unit
    if needs_laser and (laser_nm is None or laser_nm <= 0):
        raise ValueError("The excitation wavelength (nm) is needed to convert to or from Raman shift.")
    # to wavelength first
    with np.errstate(divide="ignore", invalid="ignore"):
        if from_unit == "Wavelength (nm)":
            lam = x
        elif from_unit == "Wavenumber (cm⁻¹)":
            lam = 1e7 / x
        elif from_unit == "Photon energy (eV)":
            lam = HC_EV_NM / x
        else:
            lam = raman_shift_to_wavelength(x, laser_nm)
        if to_unit == "Wavelength (nm)":
            xn = lam
        elif to_unit == "Wavenumber (cm⁻¹)":
            xn = 1e7 / lam
        elif to_unit == "Photon energy (eV)":
            xn = HC_EV_NM / lam
        else:
            xn = wavelength_to_raman_shift(lam, laser_nm)
    yn = y.copy()
    if jacobian:
        with np.errstate(divide="ignore", invalid="ignore"):
            dxo = np.gradient(x); dxn = np.gradient(xn)
            yn = y * np.abs(dxo / dxn)
    m = np.isfinite(xn) & np.isfinite(yn)
    xn, yn = xn[m], yn[m]
    order = np.argsort(xn, kind="mergesort")
    return xn[order], yn[order]


# ----------------------------------------------------------------------
# x-axis calibration against reference lines
# ----------------------------------------------------------------------

@dataclass
class CalibrationResult:
    coefficients: List[float]     # polynomial, highest power first (np.polyval order)
    order: int
    measured: List[float]
    reference: List[float]
    residuals: List[float]        # reference − corrected(measured)
    rms: float

    def apply(self, x: Sequence[float]) -> np.ndarray:
        return np.polyval(self.coefficients, np.asarray(x, float))


def calibrate_x_axis(measured: Sequence[float], reference: Sequence[float], order: int = 1) -> CalibrationResult:
    """Polynomial x' = P(x) mapping measured line positions onto their
    reference values (silicon 520.7 cm⁻¹, neon lines, a calibrated
    standard). One pair gives a shift, two a linear scale, more a curve;
    the order is capped at n_pairs − 1 and at 4. Residuals tell whether
    the chosen order is honest."""
    mx = np.asarray(list(measured), float); rx = np.asarray(list(reference), float)
    if mx.size != rx.size or mx.size == 0:
        raise ValueError("Give the same number of measured and reference positions (at least one pair).")
    order = int(max(0, min(order, 4, mx.size - 1)))
    if mx.size == 1:
        coef = np.array([1.0, float(rx[0] - mx[0])])
    else:
        coef = np.polyfit(mx, rx, order)
    corrected = np.polyval(coef, mx)
    res = rx - corrected
    return CalibrationResult(coefficients=[float(c) for c in coef], order=order,
                             measured=[float(v) for v in mx], reference=[float(v) for v in rx],
                             residuals=[float(v) for v in res], rms=float(np.sqrt(np.mean(res ** 2))))


def parse_pairs(text: str) -> Tuple[List[float], List[float]]:
    """'520.3 -> 520.7; 1001.9 -> 1001.4' (also '520.3=520.7', '520.3,520.7'
    or '520.3 520.7' per pair; pairs separated by ';' or newlines)."""
    import re
    measured, reference = [], []
    for chunk in re.split(r"[;\n]+", text or ""):
        chunk = chunk.strip()
        if not chunk:
            continue
        nums = re.findall(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?", chunk)
        if len(nums) != 2:
            raise ValueError(f"Cannot read the pair {chunk!r} (expected 'measured -> reference').")
        measured.append(float(nums[0])); reference.append(float(nums[1]))
    return measured, reference


# ----------------------------------------------------------------------
# Single-spectrum operations (Calculations workspace)
# ----------------------------------------------------------------------

def normalize_to_value(x: Sequence[float], y: Sequence[float], *, x0: float, target: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Scale so that y(x0) (linearly interpolated) equals `target`."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    v = float(np.interp(float(x0), x, y))
    if abs(v) < 1e-30:
        raise ValueError(f"The spectrum is zero at x = {x0:g}; cannot normalise there.")
    return x.copy(), y * (float(target) / v)


def subtract_range_mean(x: Sequence[float], y: Sequence[float], *, xmin: float, xmax: float) -> Tuple[np.ndarray, np.ndarray]:
    """Spectragryph's 'simple baseline': subtract the mean of y over a
    signal-free range (an offset correction, nothing more)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    lo, hi = min(xmin, xmax), max(xmin, xmax)
    sel = (x >= lo) & (x <= hi)
    if sel.sum() < 1:
        raise ValueError(f"No data in [{lo:g}, {hi:g}].")
    return x.copy(), y - float(np.mean(y[sel]))


def remove_peak(x: Sequence[float], y: Sequence[float], *, center: float, fwhm: float, span_factor: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
    """Replace the peak at `center` (± span_factor·FWHM) by a straight
    line between the signal just outside that interval."""
    x = np.asarray(x, float); y = np.asarray(y, float).copy()
    half = float(span_factor) * abs(float(fwhm))
    if half <= 0:
        raise ValueError("FWHM must be positive.")
    inside = (x > center - half) & (x < center + half)
    if inside.sum() == 0:
        raise ValueError("No data points inside the peak interval.")
    i0 = int(np.argmax(inside)); i1 = int(len(x) - 1 - np.argmax(inside[::-1]))
    xa = x[max(0, i0 - 1)]; xb = x[min(len(x) - 1, i1 + 1)]
    ya = y[max(0, i0 - 1)]; yb = y[min(len(x) - 1, i1 + 1)]
    y[inside] = np.interp(x[inside], [xa, xb], [ya, yb])
    return x.copy(), y


def snv(y: Sequence[float]) -> np.ndarray:
    """Standard normal variate: (y − mean) / std of the whole spectrum."""
    y = np.asarray(y, float)
    s = float(np.std(y))
    if s <= 0:
        raise ValueError("Flat spectrum: SNV undefined.")
    return (y - float(np.mean(y))) / s


def msc(y: Sequence[float], reference: Sequence[float]) -> np.ndarray:
    """Multiplicative scatter correction against a reference on the same
    grid: regress y = a + b·ref, return (y − a) / b."""
    y = np.asarray(y, float); r = np.asarray(reference, float)
    if y.shape != r.shape:
        raise ValueError("MSC needs the spectrum and the reference on the same grid.")
    b, a = np.polyfit(r, y, 1)
    if abs(b) < 1e-30:
        raise ValueError("MSC: zero slope against the reference.")
    return (y - a) / b
