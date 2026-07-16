"""
htxrd_science.py — high-temperature XRD series processing (M20),
framework-agnostic.

Scope per the plan (informed by Jana2020's "cyclic refinement" workflow and
GSAS-II's sequential refinement, deliberately scoped DOWN from full
Rietveld — see the M20 milestone text for the reasoning):
  - Directory import of an ordered pattern series, ramp variable
    (temperature) derived from each file's own metadata (.rasx Temp axis,
    already parsed by io_universal.parse_rasx) or from a Jana-style
    filename template ("NB-LM01MO_???.XRDML" — rewrite the varying
    substring with ?'s).
  - Peak tracking: fit one peak (fitting_science.fit_spectrum — the same
    single entry point everything else uses) independently per pattern
    inside a user-chosen 2theta window; report center/FWHM/amplitude/area
    + fit quality per pattern.
  - Phase-transition-window flagging: a fit-quality (chi2_red) series
    diagnostic — patterns where the single-peak model abruptly degrades
    are flagged, the lightweight analog of Jana's GOF-vs-temperature plot
    (literally how the CaTeO3 tutorial example located its dehydration
    transition windows).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import io_universal


# =============================================================================
# Series loading
# =============================================================================

@dataclass
class HtxrdPattern:
    path: str
    name: str
    x: np.ndarray  # 2theta (deg)
    y: np.ndarray  # intensity
    ramp_value: Optional[float] = None  # temperature (deg C) or template-derived number
    ramp_source: str = "none"  # "metadata" | "filename" | "index" | "none"
    meta: Dict[str, Any] = field(default_factory=dict)


def ramp_value_from_template(filename: str, template: str) -> Optional[float]:
    """Jana-style filename template: the varying substring is written as
    one or more '?' characters (e.g. 'NB-LM01MO_???.XRDML' matches
    'NB-LM01MO_100.XRDML' -> 100.0). Matching is case-insensitive and only
    against the basename. Returns None when the filename doesn't fit the
    template or the varying part isn't numeric."""
    base = os.path.basename(filename)
    m = re.search(r"\?+", template)
    if not m:
        return None
    n_wild = m.end() - m.start()
    pattern = re.escape(template[: m.start()]) + rf"(.{{{n_wild}}})" + re.escape(template[m.end():]) + "$"
    match = re.match(pattern, base, flags=re.IGNORECASE)
    if not match:
        return None
    token = match.group(1).strip()
    token = token.replace(",", ".")
    try:
        return float(token)
    except ValueError:
        digits = re.sub(r"[^\d.+-]", "", token)
        try:
            return float(digits) if digits else None
        except ValueError:
            return None


def load_pattern(path: str) -> HtxrdPattern:
    """Load one diffraction pattern via io_universal's parser registry
    (so .rasx, .xy, generic text all work), picking up the .rasx Temp-axis
    midpoint as the ramp value when present. Each HTXRD scan is collected
    over a temperature RANGE while the sample keeps heating (Position ->
    EndPosition), so the midpoint is the single representative value."""
    df, meta = io_universal.load_any(path, return_meta=True)
    canon = meta.get("canonical_map", {}) or {}
    x_col = canon.get("X") or df.columns[0]
    y_col = canon.get("Y") or df.columns[1]
    x = df[x_col].astype(float).to_numpy()
    y = df[y_col].astype(float).to_numpy()
    order = np.argsort(x, kind="mergesort")
    x, y = x[order], y[order]

    ramp_value = None
    ramp_source = "none"
    t0, t1 = meta.get("temp_start_C"), meta.get("temp_end_C")
    if t0 is not None:
        ramp_value = (float(t0) + float(t1)) / 2.0 if t1 is not None else float(t0)
        ramp_source = "metadata"

    return HtxrdPattern(
        path=str(path), name=os.path.splitext(os.path.basename(path))[0],
        x=x, y=y, ramp_value=ramp_value, ramp_source=ramp_source, meta=meta,
    )


def load_series(
    paths: List[str], *, filename_template: str = "",
) -> List[HtxrdPattern]:
    """Load a whole ordered series. Ramp-value resolution per pattern, in
    priority order: file metadata (.rasx Temp axis) -> filename template
    (if provided) -> sequence index (fallback, so the series is still
    plottable/trackable even with no temperature information at all).
    Returned sorted by ramp value (stable for ties)."""
    patterns: List[HtxrdPattern] = []
    for i, p in enumerate(paths):
        pat = load_pattern(p)
        if pat.ramp_value is None and filename_template:
            v = ramp_value_from_template(p, filename_template)
            if v is not None:
                pat.ramp_value = v
                pat.ramp_source = "filename"
        if pat.ramp_value is None:
            pat.ramp_value = float(i)
            pat.ramp_source = "index"
        patterns.append(pat)

    patterns.sort(key=lambda p: p.ramp_value)
    return patterns


def find_series_files(folder: str, *, extensions: Tuple[str, ...] = (".rasx", ".xy", ".xrdml", ".txt", ".dat")) -> List[str]:
    """All candidate pattern files in a folder (non-recursive), name-sorted."""
    out = []
    try:
        for entry in os.scandir(folder):
            if entry.is_file() and entry.name.lower().endswith(tuple(e.lower() for e in extensions)):
                out.append(entry.path)
    except OSError:
        return []
    return sorted(out)


# =============================================================================
# Peak tracking across the series
# =============================================================================

@dataclass
class PeakTrackResult:
    pattern_name: str
    ramp_value: float
    center: float
    fwhm: float
    amplitude: float
    area: float
    chi2_red: float
    error: Optional[str] = None


def track_peak(
    patterns: List[HtxrdPattern], *, window_lo: float, window_hi: float,
    shape: str = "G",
) -> List[PeakTrackResult]:
    """Fit ONE peak (fitting_science.fit_spectrum, mode='classic')
    independently per pattern within [window_lo, window_hi] 2theta, and
    report center/FWHM/amplitude/area + chi2_red per pattern — the
    tractable analog of Jana's cell-parameter-vs-temperature Graph tool at
    the single-peak level. Per-pattern failures are reported as rows with
    an error, never abort the rest of the series (a peak legitimately
    vanishing above a phase transition is an expected outcome, not an
    exceptional one)."""
    from fitting_science import fit_spectrum

    if window_hi <= window_lo:
        raise ValueError("window_hi must be greater than window_lo")

    results: List[PeakTrackResult] = []
    for pat in patterns:
        mask = (pat.x >= window_lo) & (pat.x <= window_hi)
        x_win, y_win = pat.x[mask], pat.y[mask]
        if len(x_win) < 8:
            results.append(PeakTrackResult(pat.name, pat.ramp_value, np.nan, np.nan, np.nan, np.nan, np.nan,
                                           error=f"only {len(x_win)} points in window"))
            continue

        baseline = float(np.nanmin(y_win))
        y_fit_input = y_win - baseline  # simple constant-baseline removal for the local window
        peak_idx = int(np.nanargmax(y_fit_input))
        center_guess = float(x_win[peak_idx])
        amp_guess = max(float(y_fit_input[peak_idx]), 1e-9)
        span = window_hi - window_lo

        comp = {
            "shape": shape,
            "shift_val": center_guess, "shift_min": window_lo, "shift_max": window_hi, "fit_shift": True,
            "fwhm_val": span * 0.1, "fwhm_min": span * 0.005, "fwhm_max": span, "fit_fwhm": True,
            "eta_val": 0.5, "eta_min": 0.0, "eta_max": 1.0, "fit_eta": True,
            "amp_val": amp_guess, "fit_amp": True,
        }
        try:
            fr = fit_spectrum(x_win, y_fit_input, [comp], mode="classic")
            params = fr.lmfit_result.params
            center = float(params["f0"].value)
            fwhm = float(params["l0"].value)
            amplitude = float(params["a0"].value)
            area = float(np.trapz(fr.peaks[0], x_win))
            results.append(PeakTrackResult(pat.name, pat.ramp_value, center, fwhm, amplitude, area, fr.chi2_red))
        except Exception as exc:
            results.append(PeakTrackResult(pat.name, pat.ramp_value, np.nan, np.nan, np.nan, np.nan, np.nan,
                                           error=str(exc)))
    return results


def flag_transition_candidates(
    results: List[PeakTrackResult], *, z: float = 3.0,
) -> List[Tuple[float, float, str]]:
    """Flag ramp values where the tracked fit quality (chi2_red) jumps
    anomalously versus the series median — the lightweight analog of
    Jana's GOF-vs-temperature plot for locating phase-transition windows.
    A fit that failed outright is flagged too (a vanished peak IS a
    transition signature). Returns (ramp_value, chi2_red, reason) tuples."""
    flags: List[Tuple[float, float, str]] = []
    valid = [r for r in results if r.error is None and np.isfinite(r.chi2_red)]
    if len(valid) >= 3:
        chi2s = np.array([r.chi2_red for r in valid], float)
        med = float(np.median(chi2s))
        mad = float(np.median(np.abs(chi2s - med)))
        robust_sigma = 1.4826 * mad if mad > 1e-30 else (float(np.std(chi2s)) or 1.0)
        for r in valid:
            if (r.chi2_red - med) > z * robust_sigma:
                flags.append((r.ramp_value, r.chi2_red, f"chi2_red {r.chi2_red:.4g} >> series median {med:.4g}"))
    for r in results:
        if r.error is not None:
            flags.append((r.ramp_value, np.nan, f"fit failed: {r.error}"))
    flags.sort(key=lambda t: t[0])
    return flags
