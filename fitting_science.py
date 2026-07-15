"""
fitting_science.py — framework-agnostic single-spectrum peak-fitting math
(Gaussian/pseudo-Voigt/GL components via lmfit + rampy).

Consolidates what used to be three near-identical inline copies of the same
peak-model function in main.py's SingleFitWindow (compute_model, a local
model_func closure inside fit_with_params, and _origin_model_func) into one.

One entry point, fit_spectrum(), covers both fitting modes so a future
multi-spectrum batch-fit feature can call it per-spectrum instead of forking
the math the way ui_dta_processing.py's batch/interactive split once did.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import lmfit
import rampy as rp


# =============================================================================
# Peak model — the ONE place peak shapes are computed from lmfit Parameters.
# =============================================================================

def compute_model(x: np.ndarray, lm_params: "lmfit.Parameters", params_struct: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Sum of per-component Gaussian/pseudo-Voigt(GL) peaks.

    params_struct is a list of per-component dicts (one per peak), each with
    at least a "shape" key ("G" or "GL"); lm_params must contain matching
    a{i}/f{i}/l{i}(/eta{i} for GL) entries, as built by build_lmfit_parameters.
    """
    x = np.asarray(x, dtype=float)
    total = np.zeros_like(x)
    peaks: List[np.ndarray] = []
    for i, d in enumerate(params_struct):
        a = lm_params[f"a{i}"].value
        f = lm_params[f"f{i}"].value
        l = lm_params[f"l{i}"].value
        if d.get("shape", "G") == "G":
            pk = rp.gaussian(x, a, f, l)
        else:
            eta = lm_params[f"eta{i}"].value if f"eta{i}" in lm_params else 0.5
            pk = rp.pseudovoigt(x, a, f, l, eta)
        peaks.append(pk)
        total += pk
    return total, peaks


def build_lmfit_parameters(params_struct: List[Dict[str, Any]]) -> "lmfit.Parameters":
    """Build an lmfit.Parameters set from the params_struct list (identical
    logic/tolerances to the original main.py implementation — bounds
    clamping, avoiding starting exactly on a bound, etc.)."""
    p = lmfit.Parameters()
    eps = 1e-9

    for i, d in enumerate(params_struct):
        # ---- Amplitude ----
        fit_amp = bool(d.get("fit_amp", True))
        a0 = d.get("amp_val", None)
        try:
            a0 = float(a0)
        except Exception:
            a0 = None
        if a0 is None or a0 <= 0.0:
            a0 = 1.0
        p.add(f"a{i}", value=a0, min=0.0, vary=fit_amp)

        # ---- Center (shift) ----
        fmin = float(d["shift_min"]); fmax = float(d["shift_max"])
        fval = float(d["shift_val"])
        if fmin > fmax:
            fmin, fmax = fmax, fmin
        if fval <= fmin:
            fval = fmin + eps
        if fval >= fmax:
            fval = fmax - eps
        p.add(f"f{i}", value=fval, min=fmin, max=fmax, vary=bool(d.get("fit_shift", True)))

        # ---- FWHM ----
        lmin = float(d.get("fwhm_min", 1e-9))
        lmax = float(d.get("fwhm_max", max(lmin * 1.000001, 1e-6)))
        lval = float(d.get("fwhm_val", max(lmin * 1.0005, 1.0)))

        if lmin > lmax:
            lmin, lmax = lmax, lmin
        if abs(lmax - lmin) < 1e-12:
            lmax = lmin + 1e-6

        eps2 = 1e-9
        if lval <= lmin:
            lval = lmin + eps2
        if lval >= lmax:
            lval = lmax - eps2

        p.add(f"l{i}", value=lval, min=lmin, max=lmax, vary=bool(d.get("fit_fwhm", True)))

        # ---- Pseudo-Voigt (GL): eta in [0,1] ----
        if d.get("shape", "G") == "GL":
            try:
                eta_min = float(d.get("eta_min", 0.0))
                eta_max = float(d.get("eta_max", 1.0))
            except Exception:
                eta_min, eta_max = 0.0, 1.0
            eta_min = max(0.0, min(eta_min, 1.0))
            eta_max = max(0.0, min(eta_max, 1.0))
            if eta_min >= eta_max:
                eta_min, eta_max = 0.0, 1.0
            try:
                eta_val = float(d.get("eta_val", 0.5))
            except Exception:
                eta_val = 0.5
            if not (eta_min < eta_val < eta_max):
                eta_val = 0.5
            p.add(f"eta{i}", value=eta_val, min=eta_min, max=eta_max, vary=bool(d.get("fit_eta", True)))

    return p


def compute_chi2(y: np.ndarray, y_fit: np.ndarray, lm_params: "lmfit.Parameters") -> float:
    resid = np.asarray(y, dtype=float) - np.asarray(y_fit, dtype=float)
    n = len(y)
    p = sum(param.vary for param in lm_params.values())
    dof = max(n - p, 1)
    return float(np.sum(resid ** 2) / dof)


def compute_r_squared(y: np.ndarray, y_fit: np.ndarray) -> float:
    """Coefficient of determination for the overall fit (M8 report enhancement)."""
    y = np.asarray(y, dtype=float)
    y_fit = np.asarray(y_fit, dtype=float)
    ss_res = float(np.sum((y - y_fit) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot < 1e-30:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def peak_centroid(x: np.ndarray, y_component: np.ndarray) -> float:
    """Intensity-weighted centroid of one fitted peak component (M8 A FAIRE
    item 10). Numerical (trapz-weighted), not a closed-form formula, so it
    stays valid for any shape (G, GL, and future asymmetric shapes) without
    per-shape special-casing."""
    x = np.asarray(x, dtype=float)
    y_component = np.asarray(y_component, dtype=float)
    denom = np.trapz(y_component, x)
    if abs(denom) < 1e-30:
        return float("nan")
    return float(np.trapz(x * y_component, x) / denom)


def find_peak_candidates(
    x: np.ndarray, y: np.ndarray, *, max_peaks: int = 10, smooth_window: int = 9,
    edge_margin_frac: float = 0.02,
) -> List[float]:
    """2nd-derivative peak-finder (M8 layer-6 item, GSAS-II/PeakFit-style
    automated initial-guess placement): smooth y, take the discrete 2nd
    derivative, and report x-positions of its local minima (strong negative
    curvature = a peak apex) as candidate component centers, strongest first.

    A lighter-weight complement to scipy.signal.find_peaks on the raw signal:
    the 2nd-derivative minimum is comparatively robust to a sloping/curved
    baseline that would otherwise bias plain amplitude-based peak-picking.

    Candidates within `edge_margin_frac` of either end of the x-range are
    dropped (default: outer 2% on each side). Confirmed directly against
    EXAMPLES/Raman_example.txt: raw, non-baseline-corrected Raman data
    routinely has a huge, sharp intensity rise at the very edge of the
    recorded window (the Rayleigh-line tail at low Raman shift, or a
    detector-edge artifact) that a curvature-based finder reports as by far
    the "strongest" candidate — real, but not a molecular peak, and a poor
    first result for an auto-find button. Every real peak-fitting tool
    (GSAS-II, Origin, PeakFit) has the same edge-effect risk; excluding a
    margin is the standard mitigation.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 5:
        return []

    order = np.argsort(x, kind="mergesort")
    x, y = x[order], y[order]

    window = max(5, smooth_window | 1)  # force odd
    window = min(window, len(y) - (1 - len(y) % 2))
    if window < 5:
        return []

    try:
        from scipy.signal import savgol_filter, find_peaks
        y_smooth = savgol_filter(y, window_length=window, polyorder=3, mode="interp")
        d2 = savgol_filter(y, window_length=window, polyorder=3, deriv=2, mode="interp")
    except Exception:
        y_smooth = y
        d2 = np.gradient(np.gradient(y, x), x)

    neg_d2 = -d2
    span = float(x[-1] - x[0])
    margin = span * edge_margin_frac
    interior = (x >= x[0] + margin) & (x <= x[-1] - margin)

    # Derive the detection threshold from the interior region only. Using the
    # full array's std would let one extreme edge artifact (see the module
    # docstring above) inflate the threshold and silently suppress genuine
    # interior peaks that would otherwise have qualified — caught directly by
    # test_find_peak_candidates_excludes_edge_artifact.
    threshold_source = neg_d2[interior] if np.any(interior) else neg_d2
    prominence = float(np.std(threshold_source)) * 0.5 if len(threshold_source) else None
    peak_idx, props = find_peaks(neg_d2, prominence=prominence if prominence and prominence > 0 else None)
    if len(peak_idx) == 0:
        return []

    if "prominences" in props and len(props["prominences"]):
        strength = props["prominences"]
    else:
        strength = y_smooth[peak_idx]

    in_bounds = (x[peak_idx] >= x[0] + margin) & (x[peak_idx] <= x[-1] - margin)
    peak_idx, strength = peak_idx[in_bounds], strength[in_bounds]
    if len(peak_idx) == 0:
        return []

    order_by_strength = np.argsort(strength)[::-1]
    centers = [float(x[peak_idx[i]]) for i in order_by_strength[:max_peaks]]
    return centers


def relax_params(old_params: "lmfit.Parameters", new_params: "lmfit.Parameters", alpha: float = 0.25) -> "lmfit.Parameters":
    """Blend parameter values: old <- old + alpha * (new - old)."""
    blended = old_params.copy()
    for name, par in blended.items():
        if name in new_params and par.vary:
            try:
                old_v = float(par.value)
                new_v = float(new_params[name].value)
                par.set(value=old_v + alpha * (new_v - old_v))
            except Exception:
                pass
    return blended


def origin_residual(params: "lmfit.Parameters", x: np.ndarray, y: np.ndarray, params_struct: List[Dict[str, Any]], soft_penalty: bool = False) -> np.ndarray:
    model, _ = compute_model(x, params, params_struct)
    res = model - y
    if soft_penalty:
        pen = []
        for name, par in params.items():
            if not par.vary:
                continue
            if (par.min is not None) and (par.value < par.min):
                pen.append((par.min - par.value) * 1e4)
            if (par.max is not None) and (par.value > par.max):
                pen.append((par.value - par.max) * 1e4)
        if pen:
            res = np.r_[res, np.array(pen)]
    return res


# =============================================================================
# One entry point — used by single-spectrum fitting today, and designed for
# a future multi-spectrum batch feature to call per-spectrum without forking.
# =============================================================================

@dataclass
class FitResult:
    lmfit_result: Any        # lmfit.minimizer.MinimizerResult
    params: "lmfit.Parameters"  # the parameter set actually producing y_fit/peaks
    y_fit: np.ndarray
    peaks: List[np.ndarray]
    chi2_red: float


def _ensure_numeric(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Coerce/validate x, y as numeric float arrays.

    The one existing GUI call site (main.py's SingleFitWindow.get_xy) already
    coerces via np.asarray(..., dtype=float) before this is ever called, so
    this mostly can't fire from today's single call site — but fit_spectrum
    is designed to be reusable (e.g. a future multi-fit batch feature calling
    it directly on freshly-loaded data), so it must be safe standalone too,
    raising a clear error instead of a low-level TypeError deep inside
    lmfit/np.trapz for non-numeric input.
    """
    try:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"x and y must be numeric (float-convertible) arrays: {exc}") from exc
    return x, y


def fit_spectrum(
    x: np.ndarray,
    y: np.ndarray,
    params_struct: List[Dict[str, Any]],
    *,
    mode: str = "classic",
    lm_params: Optional["lmfit.Parameters"] = None,
    alpha: float = 0.25,
    soft_penalty: bool = False,
) -> FitResult:
    """Fit (or take one relaxation step toward fitting) a set of peak
    components to (x, y).

    mode="classic": one-shot Levenberg-Marquardt fit (leastsq), starting from
        build_lmfit_parameters(params_struct) unless lm_params is given.
    mode="origin_step": ONE stepwise iteration of the "Origin-like" mode —
        fit, then relax old params toward the new fit by `alpha`, returning
        the relaxed parameters as the result. lm_params (the current
        parameter state) is required; call this repeatedly in a loop,
        checking FitResult.chi2_red / lmfit_result.chisqr for convergence
        between calls — that looping/convergence-check responsibility stays
        with the caller (the GUI draws and logs between steps).
    """
    x, y = _ensure_numeric(x, y)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if mode == "classic":
        params = lm_params if lm_params is not None else build_lmfit_parameters(params_struct)

        def residual(params, x, y):
            model, _ = compute_model(x, params, params_struct)
            return model - y

        result = lmfit.minimize(
            residual, params, args=(x, y), method="leastsq",
            ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=10000,
        )
        y_fit, peaks = compute_model(x, result.params, params_struct)
        chi2 = compute_chi2(y, y_fit, result.params)
        return FitResult(lmfit_result=result, params=result.params, y_fit=y_fit, peaks=peaks, chi2_red=chi2)

    if mode == "origin_step":
        if lm_params is None:
            raise ValueError("mode='origin_step' requires lm_params (the current parameter state).")
        minimizer = lmfit.Minimizer(
            origin_residual, lm_params, fcn_args=(x, y, params_struct), fcn_kws={"soft_penalty": soft_penalty},
        )
        result = minimizer.minimize(method="leastsq", ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=100)
        relaxed = relax_params(lm_params, result.params, alpha=alpha)
        y_fit, peaks = compute_model(x, relaxed, params_struct)
        chi2 = compute_chi2(y, y_fit, relaxed)
        return FitResult(lmfit_result=result, params=relaxed, y_fit=y_fit, peaks=peaks, chi2_red=chi2)

    raise ValueError(f"Unknown fit mode: {mode!r} (expected 'classic' or 'origin_step')")
