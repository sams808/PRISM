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

# NOTE on the width convention: rampy's gaussian/pseudovoigt third
# parameter is HWHM (half-width at half-maximum), even though this app's
# params_struct keys and UI labels have historically said "FWHM" — the
# original main.py passed fwhm_val straight into rampy's HWHM slot, so
# every saved model and every fit ever made uses the HWHM interpretation.
# Discovered while adding the V/EMG shapes (their first drafts assumed the
# label was literal and produced peaks half as wide as G/GL for the same
# `l`). The new shapes below deliberately take `hwhm` to match the
# established behavior rather than the label — changing the semantics now
# would silently re-interpret every existing saved model.

def voigt_peak(x: np.ndarray, amplitude: float, center: float, hwhm: float, eta: float) -> np.ndarray:
    """TRUE Voigt profile (Gaussian⊗Lorentzian convolution via
    scipy.special.voigt_profile), height-normalized so `amplitude` is the
    peak height — the same conventions as rampy's gaussian/pseudovoigt used
    by the G/GL shapes (width = HWHM, see module note above). `eta` (0..1)
    splits the width between the two components: eta=0 is a pure Gaussian
    of HWHM=hwhm and eta=1 a pure Lorentzian of HWHM=hwhm. (For
    intermediate eta the total width is close to, not exactly, `hwhm` —
    the same interpretation the pseudo-Voigt already carries.)"""
    from scipy.special import voigt_profile
    fwhm_total = 2.0 * max(float(hwhm), 1e-12)
    eta = min(max(float(eta), 0.0), 1.0)
    f_l = eta * fwhm_total
    f_g = (1.0 - eta) * fwhm_total
    sigma = f_g / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    gamma = f_l / 2.0
    if sigma < 1e-12 and gamma < 1e-12:
        sigma = 1e-12
    profile = voigt_profile(np.asarray(x, float) - float(center), sigma, gamma)
    peak_val = voigt_profile(0.0, sigma, gamma)
    if peak_val <= 0:
        return np.zeros_like(np.asarray(x, float))
    return float(amplitude) * profile / peak_val


def emg_peak(x: np.ndarray, amplitude: float, center: float, hwhm: float, skew: float) -> np.ndarray:
    """Exponentially modified Gaussian with SIGNED skew, height-normalized
    so `amplitude` is the peak height (width = HWHM of the underlying
    Gaussian, matching the G/GL convention — see module note above).
    skew > 0 tails to high x, skew < 0 tails to low x (mirror), |skew| is
    the exponential decay constant in x-units. |skew| below ~1% of sigma
    degenerates numerically toward a plain Gaussian, which is what's
    returned in that limit.

    Uses the erfcx-stable formulation (exp(-(x-mu)^2/2sigma^2) * erfcx(z))
    rather than the naive exp(...)*erfc(...) kernel, which overflows for
    small tau."""
    from scipy.special import erfcx
    x = np.asarray(x, dtype=float)
    sigma = 2.0 * max(float(hwhm), 1e-12) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    tau = float(skew)
    if abs(tau) < 1e-2 * sigma:
        return rp.gaussian(x, amplitude, center, hwhm)

    sign = 1.0 if tau > 0 else -1.0
    tau_abs = abs(tau)
    dx = sign * (x - float(center))  # mirror for negative skew
    z = sigma / (np.sqrt(2.0) * tau_abs) - dx / (np.sqrt(2.0) * sigma)
    profile = np.exp(-dx * dx / (2.0 * sigma * sigma)) * erfcx(z)
    peak_val = float(np.nanmax(profile))
    if peak_val <= 0 or not np.isfinite(peak_val):
        return np.zeros_like(x)
    return float(amplitude) * profile / peak_val


def compute_model(x: np.ndarray, lm_params: "lmfit.Parameters", params_struct: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Sum of per-component peaks. Shapes: "G" (Gaussian), "GL"
    (pseudo-Voigt), "V" (true Voigt), "EMG" (exponentially modified
    Gaussian, signed skew — the asymmetric-peak shape).

    params_struct is a list of per-component dicts (one per peak), each with
    at least a "shape" key; lm_params must contain matching a{i}/f{i}/l{i}
    (/eta{i} for GL and V, /s{i} for EMG) entries, as built by
    build_lmfit_parameters.
    """
    x = np.asarray(x, dtype=float)
    total = np.zeros_like(x)
    peaks: List[np.ndarray] = []
    for i, d in enumerate(params_struct):
        a = lm_params[f"a{i}"].value
        f = lm_params[f"f{i}"].value
        l = lm_params[f"l{i}"].value
        shape = d.get("shape", "G")
        if shape == "G":
            pk = rp.gaussian(x, a, f, l)
        elif shape == "V":
            eta = lm_params[f"eta{i}"].value if f"eta{i}" in lm_params else 0.5
            pk = voigt_peak(x, a, f, l, eta)
        elif shape == "EMG":
            skew = lm_params[f"s{i}"].value if f"s{i}" in lm_params else 0.0
            pk = emg_peak(x, a, f, l, skew)
        else:  # "GL" pseudo-Voigt (historic default for any unknown shape)
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

        # ---- Pseudo-Voigt (GL) and true Voigt (V): eta in [0,1] ----
        if d.get("shape", "G") in ("GL", "V"):
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

        # ---- EMG: signed skew (exponential decay constant, x-units) ----
        if d.get("shape", "G") == "EMG":
            try:
                s_min = float(d.get("skew_min", -100.0))
                s_max = float(d.get("skew_max", 100.0))
            except Exception:
                s_min, s_max = -100.0, 100.0
            if s_min > s_max:
                s_min, s_max = s_max, s_min
            try:
                s_val = float(d.get("skew_val", 1.0))
            except Exception:
                s_val = 1.0
            if not (s_min < s_val < s_max):
                s_val = (s_min + s_max) / 2.0
            p.add(f"s{i}", value=s_val, min=s_min, max=s_max, vary=bool(d.get("fit_skew", True)))

    # ---- Parameter linking (Origin-style "share this FWHM with peak N") ----
    # Second pass, after every base parameter exists: a component with
    # "link_fwhm": j takes its width from component j via an lmfit
    # constraint expression (l{i} = l{j}); same for "link_eta". Self-links
    # and out-of-range indices are ignored rather than erroring — a linked
    # recipe applied to a smaller model shouldn't explode.
    n = len(params_struct)
    for i, d in enumerate(params_struct):
        for key, pname in (("link_fwhm", "l"), ("link_eta", "eta")):
            j = d.get(key)
            if j is None:
                continue
            try:
                j = int(j)
            except (TypeError, ValueError):
                continue
            if j == i or not (0 <= j < n):
                continue
            if f"{pname}{i}" in p and f"{pname}{j}" in p:
                p[f"{pname}{i}"].expr = f"{pname}{j}"

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
    minimizer: Any = None    # the lmfit.Minimizer (classic mode) — needed by conf_interval()


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

        minimizer = lmfit.Minimizer(residual, params, fcn_args=(x, y))
        result = minimizer.minimize(
            method="leastsq",
            ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=10000,
        )
        y_fit, peaks = compute_model(x, result.params, params_struct)
        chi2 = compute_chi2(y, y_fit, result.params)
        return FitResult(lmfit_result=result, params=result.params, y_fit=y_fit, peaks=peaks, chi2_red=chi2,
                         minimizer=minimizer)

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


def compute_confidence_intervals(fit_result: FitResult, sigmas=(1, 2)) -> str:
    """F-test confidence intervals via lmfit.conf_interval() (the rigorous
    profiling method, complementing the covariance-based ±1σ standard
    errors already in reports). Returns lmfit's formatted ci_report text.

    Raises ValueError with a readable message when profiling isn't
    applicable (needs a classic-mode FitResult, ≥2 varying parameters, and
    parameters not pinned at their bounds — all common lmfit limitations)."""
    if fit_result.minimizer is None:
        raise ValueError("Confidence intervals need a classic-mode fit (run 'Fit !' first).")
    n_vary = sum(p.vary for p in fit_result.lmfit_result.params.values())
    if n_vary < 2:
        raise ValueError("Confidence-interval profiling needs at least 2 varying parameters.")
    try:
        ci = lmfit.conf_interval(fit_result.minimizer, fit_result.lmfit_result, sigmas=list(sigmas))
    except Exception as exc:
        raise ValueError(
            f"lmfit could not profile confidence intervals: {exc}\n"
            "(Common causes: a parameter stuck at its min/max bound, or a "
            "degenerate/underdetermined fit.)"
        ) from exc
    return lmfit.printfuncs.ci_report(ci)
