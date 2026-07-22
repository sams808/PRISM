"""
saxs_core/composite_staged.py — the staged, reproducible fitting pipeline
(spec §4, stages 0-4 in this file; stages 5-6 diagnostics/model-selection
follow in a later pass). General-purpose: works on ANY 1D SAXS Curve, not
tied to a specific sample series or naming scheme — `sample_id` is just an
arbitrary string used to seed the deterministic multistart RNG and to label
the result; callers can pass a filename, a UUID, or anything else.

Stage sequence (spec §4.2):
  0  hygiene: trim, sigma model, auto-window proposal, class guess (a/b/c)
  1  fit BG (flat_background + power_law) on W_hiq only; freeze pl_B/pl_p
  2  add teubner_strey, seeded from the peak window; fit TS+bg_C on
     W_peak ∪ W_hiq (pl_B/pl_p stay frozen); a class-a guardrail (pulled
     forward from spec's own Stage 6 rule) rejects a TS fit that isn't
     actually significant, falling back to BG alone for that sample
  3  add guinier_porod for a low-q upturn; fit GP+bg_C on W_loq with
     TS/pl frozen
  4  global: release ALL parameters with widened bounds around the
     stage 1-3 best-fit values, multistart (deterministic, seeded from
     sample_id), keep the lowest reduced chi-square

The pipeline never raises on a shoulder-only or featureless profile — a
failed later stage simply falls back to the best composite assembled so
far, and the function still returns a valid FitResult.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from saxs_core.composite_fit import CompositeModel, build_composite
from saxs_core.curve import Curve

Windows = Dict[str, Tuple[float, float]]

CODE_VERSION = "composite_staged-v1"


# =============================================================================
# Stage 0 — hygiene, sigma model, windows, class guess
# =============================================================================

def estimate_sigma_model(q: np.ndarray, I: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Poisson-like sigma model when the curve carries no measured sigma
    (spec §3): sigma_i = max(eps, c*sqrt(max(I_i,0) + I_bg_est)), with c
    calibrated on the high-q plateau's scatter via a rolling MAD."""
    q = np.asarray(q, dtype=float)
    I = np.asarray(I, dtype=float)
    n = len(I)
    tail = I[int(0.85 * n):]
    if tail.size < 5:
        tail = I[-max(5, n // 10):] if n else I
    I_bg_est = float(np.median(tail)) if tail.size else 0.0
    resid = tail - I_bg_est
    med = float(np.median(resid)) if resid.size else 0.0
    mad = float(np.median(np.abs(resid - med))) * 1.4826 if resid.size >= 3 else float(np.std(resid))
    denom = float(np.sqrt(max(np.median(np.clip(tail, 0, None)), 0.0) + I_bg_est)) or 1.0
    c = max(mad / denom, 1e-6)
    sigma = c * np.sqrt(np.clip(I, 0, None) + max(I_bg_est, 0.0))
    return np.maximum(sigma, eps)


def _log_rebin(q: np.ndarray, I: np.ndarray, sigma: np.ndarray, per_decade: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Log-spaced rebinning for SPEED/preview only — never used for the
    archived fit unless explicitly requested (spec §3)."""
    positive = q > 0
    q, I, sigma = q[positive], I[positive], sigma[positive]
    qmin, qmax = float(np.min(q)), float(np.max(q))
    n_decades = max(math.log10(qmax / qmin), 1e-6)
    n_bins = max(int(n_decades * per_decade), 10)
    edges = np.geomspace(qmin, qmax * 1.0000001, n_bins + 1)
    idx = np.clip(np.digitize(q, edges) - 1, 0, n_bins - 1)
    q_out, I_out, s_out = [], [], []
    for b in range(n_bins):
        m = idx == b
        if not np.any(m):
            continue
        q_out.append(float(np.mean(q[m])))
        I_out.append(float(np.mean(I[m])))
        s_out.append(float(np.sqrt(np.sum(sigma[m] ** 2)) / np.sum(m)))
    return np.array(q_out), np.array(I_out), np.array(s_out)


@dataclass
class HygieneResult:
    curve: Curve
    n_trimmed_edge: int
    n_dropped_nonfinite: int
    sigma_model: str  # "measured" | "poisson_like_estimated"


def apply_hygiene(curve: Curve, *, trim_n: int = 3, log_rebin: bool = False,
                  rebin_per_decade: int = 150) -> HygieneResult:
    """Trim first/last `trim_n` points, drop non-finite/negative-I points,
    attach a sigma model if the curve doesn't carry one. `log_rebin` is
    OFF by default and should stay off for the archived/final fit — it
    exists only for fast interactive previews of very dense curves."""
    q = np.asarray(curve.q, dtype=float)
    I = np.asarray(curve.intensity, dtype=float)
    sigma = None if curve.sigma is None else np.asarray(curve.sigma, dtype=float)

    finite = np.isfinite(q) & np.isfinite(I) & (I >= 0)
    if sigma is not None:
        finite = finite & np.isfinite(sigma)
    n_dropped = int((~finite).sum())
    q, I = q[finite], I[finite]
    sigma = sigma[finite] if sigma is not None else None

    order = np.argsort(q)
    q, I = q[order], I[order]
    sigma = sigma[order] if sigma is not None else None

    n_edge = 0
    if trim_n > 0 and len(q) > 2 * trim_n:
        q, I = q[trim_n:-trim_n], I[trim_n:-trim_n]
        sigma = sigma[trim_n:-trim_n] if sigma is not None else None
        n_edge = 2 * trim_n

    sigma_model = "measured"
    if sigma is None:
        sigma = estimate_sigma_model(q, I)
        sigma_model = "poisson_like_estimated"

    if log_rebin:
        q, I, sigma = _log_rebin(q, I, sigma, rebin_per_decade)

    new_curve = curve.copy_with(
        q=q, intensity=I, sigma=sigma, step="composite_hygiene",
        trim_n=trim_n, n_dropped_nonfinite=n_dropped, log_rebin=log_rebin,
        sigma_model=sigma_model,
    )
    return HygieneResult(curve=new_curve, n_trimmed_edge=n_edge,
                         n_dropped_nonfinite=n_dropped, sigma_model=sigma_model)


def guess_class(q: np.ndarray, I: np.ndarray) -> Tuple[str, float]:
    """Cheap heuristic guess ('a'|'b'|'c') from peak prominence in a
    Kratky-like q^2*I representation. This only guides window proposals,
    logging, and a guardrail against noise-driven false positives in
    stages 0-4; the rigorous arbiter is the BIC-based model-selection
    ladder (spec's Stage 6, a later pass).

    Uses scipy.signal.find_peaks' own prominence metric (height relative
    to the surrounding valleys) rather than a hand-rolled comparison —
    genuinely more robust to noise than checking "is this point higher
    than one some fixed distance away", which a random fluctuation on an
    otherwise-monotonic (background-dominated) trend can satisfy by
    chance; find_peaks never returns a boundary/monotonic-trend point as
    a peak at all.

    Smoothing uses scipy.ndimage.uniform_filter1d with mode="nearest"
    (extends the boundary value outward) rather than saxs_core.analysis'
    moving_average (an implicit-zero-pad "same"-mode convolution): on a
    monotonically-rising, background-dominated Kratky trend, zero-padding
    produces an artificial dip-then-recovery right at the high-q edge that
    find_peaks mistakes for a real local max. A narrower window
    (n // 150 vs. that function's n // 35) also matters here specifically:
    this application's real peaks (large xi => narrow in q) sit close to
    q_min, so a wide window would smear the very feature being sought —
    verified empirically across both a real measured profile and multiple
    noise realizations of a synthetic curve before adopting these values."""
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import find_peaks
    q = np.asarray(q, dtype=float)
    I = np.asarray(I, dtype=float)
    n = len(q)
    kratky = (q ** 2) * I
    win = max(5, n // 150)
    smoothed = uniform_filter1d(kratky, size=win, mode="nearest")
    baseline = float(np.median(smoothed))
    prominence_floor = max(baseline * 0.3, 1e-300)
    peaks, props = find_peaks(smoothed, prominence=prominence_floor)
    if peaks.size == 0:
        return "a", 0.0
    best = int(peaks[int(np.argmax(props["prominences"]))])
    peak_val = float(smoothed[best])
    prominence = (peak_val / baseline) if baseline > 0 else (float("inf") if peak_val > 0 else 0.0)
    if prominence < 1.3:
        return "a", prominence
    if prominence < 3.0:
        return "b", prominence
    return "c", prominence


def _locate_peak(q: np.ndarray, I: np.ndarray, smooth_frac: int = 200) -> Tuple[float, float, float]:
    """Locate the strongest finite-q feature via a Kratky-like q^2*I
    representation, bracketed by half-max descent — the same technique as
    saxs_core.analysis.auto_detect_peak_window, but with a FINER smoothing
    window (len(q)//200 rather than that function's //35).

    Why a dedicated detector rather than reusing auto_detect_peak_window
    as-is: that function's coarser window suits the broad globular-particle
    features it was built for, but washes out a genuinely narrow
    Teubner-Strey peak — this application's xi (2500-5000 Å per the spec)
    implies a peak only a small fraction of the full instrument q-range
    wide. Verified empirically against both a synthetic TS curve and a
    real measured profile before adopting; still a general Kratky-based
    detector, not tuned to any particular sample's expected q*."""
    q = np.asarray(q, dtype=float)
    I = np.asarray(I, dtype=float)
    from saxs_core.analysis import moving_average
    win = max(5, len(q) // smooth_frac)
    smoothed = moving_average((q ** 2) * I, win)
    i_peak = int(np.argmax(smoothed))
    peak_val = float(smoothed[i_peak])
    half = 0.6 * peak_val
    left = i_peak
    while left > 0 and smoothed[left] > half:
        left -= 1
    right = i_peak
    while right < len(q) - 1 and smoothed[right] > half:
        right += 1
    pad = max(2, win // 2)
    left = max(0, left - pad)
    right = min(len(q) - 1, right + pad)
    return float(q[i_peak]), float(q[left]), float(q[right])


def propose_windows(q: np.ndarray, I: np.ndarray) -> Windows:
    """Auto-propose W_hiq/W_peak/W_loq (spec §4.1). Always visible/editable
    by the caller — this is a starting point, not a hard requirement."""
    q = np.asarray(q, dtype=float)
    I = np.asarray(I, dtype=float)
    qmin, qmax = float(np.min(q)), float(np.max(q))
    q_star, _peak_lo, peak_hi = _locate_peak(q, I)

    w_peak = (max(q_star / 2.5, qmin), min(q_star * 2.5, qmax))
    hiq_lo = min(3.0 * peak_hi, 0.95 * qmax)
    if hiq_lo >= qmax:
        hiq_lo = 0.8 * qmax
    w_hiq = (hiq_lo, qmax)
    w_loq = (qmin, max(w_peak[0], qmin * 1.0001))  # tied to W_peak's own start, self-consistent
    return {"W_peak": w_peak, "W_hiq": w_hiq, "W_loq": w_loq}


def _mask_for(q: np.ndarray, windows: Windows, keys: Tuple[str, ...]) -> np.ndarray:
    """Union of the named windows (points inside ANY of them)."""
    mask = np.zeros_like(q, dtype=bool)
    for key in keys:
        if key not in windows:
            continue
        lo, hi = sorted(windows[key])
        mask = mask | ((q >= lo) & (q <= hi))
    return mask


def _seed_from_sample_id(sample_id: str) -> int:
    """Stable (process/run independent) seed from an arbitrary string —
    Python's built-in hash() is randomized per-process, so this uses a
    fixed hash instead (spec §4.5: 'deterministic, seeded from sample_id
    hash')."""
    digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


# =============================================================================
# FitResult (spec §4.3)
# =============================================================================

@dataclass
class FitResult:
    sample_id: str
    preset_chosen: str
    residual_mode: str
    loss: str
    windows: Windows
    sigma_model: str
    params: Dict[str, Dict[str, Any]]
    derived: Dict[str, Any]
    gof: Dict[str, float]
    flags: List[str] = field(default_factory=list)
    seeds_used: Dict[str, float] = field(default_factory=dict)
    multistart_n: int = 0
    code_version: str = CODE_VERSION
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    no_peak: bool = False
    stages: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id, "preset_chosen": self.preset_chosen,
            "residual_mode": self.residual_mode, "loss": self.loss,
            "windows": {k: list(v) for k, v in self.windows.items()},
            "sigma_model": self.sigma_model, "params": self.params,
            "derived": self.derived, "gof": self.gof, "flags": list(self.flags),
            "seeds_used": self.seeds_used, "multistart_n": self.multistart_n,
            "code_version": self.code_version, "timestamp": self.timestamp,
            "no_peak": self.no_peak, "stages": self.stages,
        }

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> "FitResult":
        payload = dict(payload)
        payload["windows"] = {k: tuple(v) for k, v in payload.get("windows", {}).items()}
        return cls(**payload)

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, indent=2, default=str)

    @classmethod
    def load_json(cls, path: str) -> "FitResult":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_json(json.load(f))

    def to_csv_row(self) -> Dict[str, Any]:
        """One flat row for the batch CSV (Phase 5)."""
        row: Dict[str, Any] = {
            "sample_id": self.sample_id, "preset_chosen": self.preset_chosen,
            "residual_mode": self.residual_mode, "loss": self.loss,
            "sigma_model": self.sigma_model, "no_peak": self.no_peak,
            "flags": ";".join(self.flags), "code_version": self.code_version,
            "timestamp": self.timestamp,
        }
        row.update({f"gof_{k}": v for k, v in self.gof.items()})
        row.update({f"derived_{k}": v for k, v in self.derived.items() if not isinstance(v, dict)})
        return row


def _params_to_dict(lmfit_params: Any) -> Dict[str, Dict[str, Any]]:
    out = {}
    for name in lmfit_params:
        p = lmfit_params[name]
        out[name] = {
            "value": float(p.value), "stderr": None if p.stderr is None else float(p.stderr),
            "min": float(p.min), "max": float(p.max), "vary": bool(p.vary),
        }
    return out


def _build_derived(model: CompositeModel, result_params: Any) -> Dict[str, Any]:
    """Per-component derived() (nested) PLUS the spec's flat, named
    top-level aliases (d, xi, fa, q_max, a2, c1, c2, Rg, p_pl, p_gp) —
    whichever of those are actually present in this composite."""
    nested = model.derived(result_params)
    flat: Dict[str, Any] = {"components": nested}
    prefixes = {prefix.rstrip("_") or comp.name: (prefix, comp.name) for prefix, comp in model.components}
    if "ts" in prefixes:
        prefix, _ = prefixes["ts"]
        flat["d"] = result_params[prefix + "d"].value
        flat["xi"] = result_params[prefix + "xi"].value
        ts_derived = nested.get("ts", {})
        flat["fa"] = ts_derived.get("fa")
        flat["q_max"] = ts_derived.get("q_max")
        flat["a2"] = ts_derived.get("a2")
        flat["c1"] = ts_derived.get("c1")
        flat["c2"] = ts_derived.get("c2")
    if "pl" in prefixes:
        prefix, _ = prefixes["pl"]
        flat["p_pl"] = result_params[prefix + "p"].value
    if "gp" in prefixes:
        prefix, _ = prefixes["gp"]
        flat["Rg"] = result_params[prefix + "Rg"].value
        flat["p_gp"] = result_params[prefix + "p"].value
    return flat


def _gof(result: Any) -> Dict[str, float]:
    return {
        "chi2red": float(result.redchi), "aic": float(result.aic), "bic": float(result.bic),
        "n_points": int(result.ndata),
    }


# =============================================================================
# Stages 1-4
# =============================================================================

def _stage1_bg(q: np.ndarray, I: np.ndarray, sigma: np.ndarray, windows: Windows) -> Dict[str, Any]:
    model = build_composite(["flat_background", "power_law"])
    mask = _mask_for(q, windows, ("W_hiq",))
    if int(mask.sum()) < 5:
        mask = np.ones_like(q, dtype=bool)  # degenerate window: fall back to everything
    seeds = model.seed(q[mask], I[mask], windows)
    params = model.to_lmfit_parameters(seed_values=seeds)
    result = model.fit(q[mask], I[mask], sigma=sigma[mask], params=params)
    return {"model": model, "result": result, "mask": mask, "seeds": seeds}


def _stage2_add_ts(q: np.ndarray, I: np.ndarray, sigma: np.ndarray, windows: Windows,
                   stage1: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    bg_params = stage1["result"].params
    model = build_composite(["flat_background", "power_law", "teubner_strey"])
    mask = _mask_for(q, windows, ("W_peak", "W_hiq"))
    if int(mask.sum()) < 8:
        return None
    peak_mask = _mask_for(q, windows, ("W_peak",))
    q_win, I_win = (q[peak_mask], I[peak_mask]) if np.any(peak_mask) else (q[mask], I[mask])
    ts_seed = model.components[-1][1].seed(q_win, I_win, windows)
    # S seeded as I(q*) minus the Stage-1 background+power-law level there (spec §4.2)
    bg_at_qstar = (bg_params["bg_C"].value
                   + bg_params["pl_B"].value * max(ts_seed["d"] and (2 * math.pi / ts_seed["d"]), 1e-8) ** (-bg_params["pl_p"].value))
    ts_seed["S"] = max(ts_seed["S"] - bg_at_qstar, ts_seed["S"] * 0.1)

    seed_values = {"bg_C": bg_params["bg_C"].value, "pl_B": bg_params["pl_B"].value,
                  "pl_p": bg_params["pl_p"].value, **{f"ts_{k}": v for k, v in ts_seed.items()}}
    params = model.to_lmfit_parameters(seed_values=seed_values)
    model.fix(params, "pl_B", bg_params["pl_B"].value)
    model.fix(params, "pl_p", bg_params["pl_p"].value)
    # narrow ts_d's bounds to the active window (spec §1.7): d in [2pi/q_hi, 2pi/q_lo]
    q_lo_win, q_hi_win = float(np.min(q[mask])), float(np.max(q[mask]))
    d_lo, d_hi = sorted([2 * math.pi / q_hi_win, 2 * math.pi / max(q_lo_win, 1e-8)])
    params["ts_d"].set(min=max(d_lo, 10.0), max=min(d_hi, 1e6))
    if not (params["ts_d"].min < params["ts_d"].value < params["ts_d"].max):
        params["ts_d"].set(value=(params["ts_d"].min + params["ts_d"].max) / 2.0)

    result = model.fit(q[mask], I[mask], sigma=sigma[mask], params=params)
    return {"model": model, "result": result, "mask": mask, "seeds": seed_values}


def ts_guardrail_ok(result: Any, sigma_local: np.ndarray, windows: Windows) -> Tuple[bool, str]:
    """Pulled forward from spec's own Stage 6 class-a guardrail: refuse a
    TS fit whose height isn't actually significant, or whose peak sits
    outside the peak window entirely — used here in Stage 2 (Phase 3) so
    a nonsense peak never contaminates the buildup even before the full
    BIC ladder (a later pass) exists."""
    S = result.params["ts_S"].value
    d = result.params["ts_d"].value
    xi = result.params["ts_xi"].value
    k, kappa = 2 * math.pi / d, 1.0 / xi
    disc = k ** 2 - kappa ** 2
    if disc <= 0:
        return False, "ts_no_finite_q_max"
    q_max = math.sqrt(disc)
    lo, hi = windows.get("W_peak", (0.0, np.inf))
    if not (lo <= q_max <= hi):
        return False, "ts_q_max_outside_w_peak"
    sigma_typ = float(np.median(sigma_local)) if sigma_local.size else 0.0
    if sigma_typ > 0 and S < 3.0 * sigma_typ:
        return False, "ts_not_significant"
    return True, ""


def _stage3_add_gp(q: np.ndarray, I: np.ndarray, sigma: np.ndarray, windows: Windows,
                   prev: Dict[str, Any], had_ts: bool) -> Optional[Dict[str, Any]]:
    prev_names = ["flat_background", "power_law"] + (["teubner_strey"] if had_ts else [])
    model = build_composite(prev_names + ["guinier_porod"])
    mask = _mask_for(q, windows, ("W_loq",))
    if int(mask.sum()) < 5:
        return None
    frozen = prev["result"].params
    seed_values = {name: frozen[name].value for name in frozen}
    gp_seed = model.components[-1][1].seed(q[mask], I[mask], windows)
    seed_values.update({f"gp_{k}": v for k, v in gp_seed.items()})
    params = model.to_lmfit_parameters(seed_values=seed_values)
    for name in frozen:
        if name != "bg_C":
            model.fix(params, name, frozen[name].value)
    params["gp_p"].set(min=2.5, max=4.3)
    if not (params["gp_p"].min < params["gp_p"].value < params["gp_p"].max):
        params["gp_p"].set(value=4.0)
    result = model.fit(q[mask], I[mask], sigma=sigma[mask], params=params)
    return {"model": model, "result": result, "mask": mask, "seeds": seed_values}


_SCALE_PARAM_SUFFIXES = ("_C", "_B", "_S", "_G", "_A", "_C_lorentz")
_LENGTH_PARAM_SUFFIXES = ("_d", "_xi", "_Rg")


def _widen_bounds_for_global(model: CompositeModel, best_values: Dict[str, float]) -> Dict[str, Tuple[float, float]]:
    """Spec §4.2 Stage 4: bounds = best ± (x/÷3 for scales, ±40% for
    d/xi/Rg); p stays within its component's own [1,4.5] bound."""
    overrides: Dict[str, Tuple[float, float]] = {}
    for prefix, comp in model.components:
        for p in comp.params():
            full = prefix + p.name
            best = best_values.get(full, p.value)
            if p.name.endswith("p") and not p.name.endswith(("_d", "xi")):
                overrides[full] = (p.min, p.max)
            elif any(full.endswith(suf) for suf in _LENGTH_PARAM_SUFFIXES):
                lo, hi = best * 0.6, best * 1.4
                overrides[full] = (max(min(lo, hi), p.min), min(max(lo, hi), p.max))
            elif any(full.endswith(suf) for suf in _SCALE_PARAM_SUFFIXES) or p.name in ("S", "B", "C", "G", "A", "C_lorentz"):
                # x/÷3 only means anything numerically when `best` is well
                # clear of zero; lmfit's own min==max safeguard uses an
                # absolute tolerance (1e-13), so a tiny-but-nonzero best
                # (pl_B is often ~1e-11 by design here) produces a min/max
                # pair BOTH below that floor, which lmfit then treats as
                # degenerate and raises. Fall back to the component's own
                # default bound in that regime instead.
                if best > 1e-8:
                    lo, hi = best / 3.0, best * 3.0
                    overrides[full] = (max(min(lo, hi), p.min), min(max(lo, hi), p.max))
                else:
                    overrides[full] = (p.min, p.max)
            else:
                overrides[full] = (p.min, p.max)
    return overrides


def _stage4_global(q: np.ndarray, I: np.ndarray, sigma: np.ndarray, model: CompositeModel,
                   best_values: Dict[str, float], sample_id: str, multistart_n: int) -> Dict[str, Any]:
    bound_overrides = _widen_bounds_for_global(model, best_values)
    rng = np.random.default_rng(_seed_from_sample_id(sample_id))
    best_result = None
    for _ in range(max(multistart_n, 1)):
        perturbed = {name: v * math.exp(rng.uniform(-0.2, 0.2)) if v > 0 else v
                    for name, v in best_values.items()}
        params = model.to_lmfit_parameters(seed_values=perturbed, bound_overrides=bound_overrides)
        try:
            result = model.fit(q, I, sigma=sigma, params=params)
        except Exception:
            continue
        if best_result is None or result.redchi < best_result.redchi:
            best_result = result
    if best_result is None:
        # last resort: fit once from the un-perturbed best values
        params = model.to_lmfit_parameters(seed_values=best_values, bound_overrides=bound_overrides)
        best_result = model.fit(q, I, sigma=sigma, params=params)
    return {"model": model, "result": best_result}


# =============================================================================
# Stage 5 — diagnostics
# =============================================================================

def _durbin_watson(residual_normalized: np.ndarray) -> float:
    """Durbin-Watson statistic on sigma-normalized residuals: ~2 means no
    autocorrelation; the spec flags DW < 1.3 (residuals trending, usually
    a sign the model shape is wrong somewhere)."""
    r = np.asarray(residual_normalized, dtype=float)
    if r.size < 2:
        return float("nan")
    denom = float(np.sum(r ** 2))
    if denom <= 0:
        return float("nan")
    return float(np.sum(np.diff(r) ** 2) / denom)


def _correlation_flags(result: Any, threshold: float = 0.95) -> List[str]:
    """Flag any pair of varying parameters with |correlation| > threshold
    (lmfit computes result.params[name].correl automatically once stderrs
    are available)."""
    flags: List[str] = []
    seen = set()
    for name, par in result.params.items():
        if not par.vary or not getattr(par, "correl", None):
            continue
        for other, rho in par.correl.items():
            key = tuple(sorted((name, other)))
            if key in seen:
                continue
            seen.add(key)
            if rho is not None and np.isfinite(rho) and abs(rho) > threshold:
                flags.append(f"high_correlation:{key[0]}~{key[1]}:{rho:.3f}")
    return flags


def compute_diagnostics(model: CompositeModel, result: Any, q: np.ndarray, windows: Windows) -> Dict[str, Any]:
    """Spec §4.2 Stage 5: chi2red/AIC/BIC (lmfit computes these already),
    Durbin-Watson, parameter-correlation flags, and physicality flags
    (q_max inside W_peak; xi vs d/2pi sanity; Rg vs 2pi/q_min warning)."""
    dw = _durbin_watson(np.asarray(result.residual, dtype=float))
    flags: List[str] = []
    if np.isfinite(dw) and dw < 1.3:
        flags.append(f"low_durbin_watson:{dw:.2f}")
    flags.extend(_correlation_flags(result))

    prefixes = {prefix.rstrip("_") or comp.name: prefix for prefix, comp in model.components}
    if "ts" in prefixes:
        prefix = prefixes["ts"]
        d = result.params[prefix + "d"].value
        xi = result.params[prefix + "xi"].value
        k, kappa = 2 * math.pi / d, 1.0 / xi
        disc = k ** 2 - kappa ** 2
        q_max = math.sqrt(disc) if disc > 0 else None
        lo, hi = windows.get("W_peak", (0.0, float("inf")))
        if q_max is None or not (lo <= q_max <= hi):
            flags.append("ts_q_max_outside_w_peak")
        if not (xi > d / (2 * math.pi)):
            flags.append("ts_xi_not_greater_than_d_over_2pi")
    if "gp" in prefixes:
        prefix = prefixes["gp"]
        Rg = result.params[prefix + "Rg"].value
        qmin = float(np.min(q)) if q.size else 1e-8
        if Rg > 0.8 * (2 * math.pi / max(qmin, 1e-12)):
            flags.append("gp_rg_poorly_constrained_vs_qmin")

    gof = {"chi2red": float(result.redchi), "aic": float(result.aic), "bic": float(result.bic),
          "n_points": int(result.ndata), "durbin_watson": dw}
    return {"gof": gof, "flags": flags}


# =============================================================================
# Stage 6 — model-selection ladder
# =============================================================================

def _fit_full_range(component_names: List[str], q: np.ndarray, I: np.ndarray, sigma: np.ndarray,
                    sample_id: str, multistart_n: int) -> Dict[str, Any]:
    model = build_composite(component_names)
    seeds = model.seed(q, I)
    return _stage4_global(q, I, sigma, model, seeds, sample_id + ":" + "_".join(component_names), multistart_n)


def _walk_ladder(order: List[str], bics: Dict[str, float], aics: Dict[str, float]) -> Tuple[str, List[Dict[str, Any]]]:
    """Pure decision logic (no fitting): walk `order` left-to-right,
    replacing the current pick whenever the next candidate clears
    Delta-BIC > 10 (current's BIC minus candidate's BIC). Any disagreement
    with the Delta-AIC > 10 verdict is recorded, but BIC always decides —
    the spec's own explicit tiebreak rule."""
    current = order[0]
    disagreements: List[Dict[str, Any]] = []
    for candidate in order[1:]:
        d_bic = bics[current] - bics[candidate]
        d_aic = aics[current] - aics[candidate]
        prefer_bic = d_bic > 10.0
        prefer_aic = d_aic > 10.0
        if prefer_bic != prefer_aic:
            disagreements.append({"pair": [current, candidate], "d_bic": d_bic, "d_aic": d_aic})
        if prefer_bic:
            current = candidate
    return current, disagreements


def select_best_preset(
    q: np.ndarray, I: np.ndarray, sigma: np.ndarray, assembled_name: str,
    assembled_model: CompositeModel, assembled_result: Any,
    sample_id: str, multistart_n: int,
) -> Dict[str, Any]:
    """Spec §4.2 Stage 6: the ladder BG -> BG_DAB -> (whatever stages 1-4
    assembled). Primary criterion is ΔBIC > 10 (lower BIC wins); ΔAIC is
    cross-checked and any disagreement between the two is recorded, but
    BIC always decides ties (spec's own explicit tiebreak). Simplification
    (noted for the record): when the assembled composite is BG_TS_GP, this
    compares it directly against the BG/BG_DAB baseline in one shot rather
    than also separately re-testing BG_TS as an intermediate rung — the
    guardrail already in stages 1-4 gates whether GP was added at all."""
    candidates: Dict[str, Tuple[CompositeModel, Any]] = {}
    ladder: Dict[str, Any] = {}

    bg_fit = _fit_full_range(["flat_background", "power_law"], q, I, sigma, sample_id, multistart_n)
    candidates["BG"] = (bg_fit["model"], bg_fit["result"])
    ladder["BG"] = {"bic": float(bg_fit["result"].bic), "aic": float(bg_fit["result"].aic)}

    dab_fit = _fit_full_range(["flat_background", "power_law", "dab"], q, I, sigma, sample_id, multistart_n)
    candidates["BG_DAB"] = (dab_fit["model"], dab_fit["result"])
    ladder["BG_DAB"] = {"bic": float(dab_fit["result"].bic), "aic": float(dab_fit["result"].aic)}

    if assembled_name not in candidates:
        candidates[assembled_name] = (assembled_model, assembled_result)
        ladder[assembled_name] = {"bic": float(assembled_result.bic), "aic": float(assembled_result.aic)}

    order = ["BG", "BG_DAB"] + ([assembled_name] if assembled_name not in ("BG", "BG_DAB") else [])
    bics = {name: candidates[name][1].bic for name in order}
    aics = {name: candidates[name][1].aic for name in order}
    current_name, disagreements = _walk_ladder(order, bics, aics)
    if disagreements:
        ladder["disagreements"] = disagreements

    final_model, final_result = candidates[current_name]
    return {"chosen": current_name, "model": final_model, "result": final_result, "ladder": ladder}


# =============================================================================
# Orchestrator
# =============================================================================

def fit_staged(
    curve: Curve,
    *,
    sample_id: Optional[str] = None,
    windows: Optional[Windows] = None,
    trim_n: int = 3,
    residual_mode: str = "linear_sigma",
    loss: str = "linear",
    multistart_n: int = 8,
    log: Callable[[str], None] = lambda *_: None,
) -> FitResult:
    """Run stages 0-4 on one profile. Never raises: a later stage that
    can't be fit (too few points in its window, a non-significant/
    nonsensical TS peak, an lmfit exception) simply falls back to the
    best composite assembled so far — the returned FitResult always
    reflects SOME valid fit, down to BG alone in the worst case."""
    sample_id = sample_id or curve.name
    flags: List[str] = []

    hygiene = apply_hygiene(curve, trim_n=trim_n)
    q = np.asarray(hygiene.curve.q, dtype=float)
    I = np.asarray(hygiene.curve.intensity, dtype=float)
    sigma = np.asarray(hygiene.curve.sigma, dtype=float)

    cls_guess, prominence = guess_class(q, I)
    active_windows = dict(propose_windows(q, I))
    if windows:
        active_windows.update(windows)  # user overrides win

    stages: Dict[str, Any] = {
        "stage0": {"class_guess": cls_guess, "prominence": prominence,
                  "n_trimmed_edge": hygiene.n_trimmed_edge,
                  "n_dropped_nonfinite": hygiene.n_dropped_nonfinite,
                  "n_points": int(q.size)},
    }

    stage1 = _stage1_bg(q, I, sigma, active_windows)
    stages["stage1"] = {"redchi": float(stage1["result"].redchi), "mask_n": int(stage1["mask"].sum())}
    current_model, current_result = stage1["model"], stage1["result"]
    preset_names = ["flat_background", "power_law"]
    had_ts = False

    stage2 = _stage2_add_ts(q, I, sigma, active_windows, stage1)
    if stage2 is not None:
        ok, reason = ts_guardrail_ok(stage2["result"], sigma[stage2["mask"]], active_windows)
        if ok and cls_guess == "a":
            # Stage 0's class guess is itself now prominence-based via
            # scipy.signal.find_peaks (robust to noise-driven false
            # positives) — an extra backstop alongside the guardrail's own
            # significance/q_max-in-window checks, not a replacement.
            ok, reason = False, "class_guess_featureless"
        stages["stage2"] = {"redchi": float(stage2["result"].redchi), "mask_n": int(stage2["mask"].sum()),
                            "guardrail_ok": ok, "guardrail_reason": reason}
        if ok:
            current_model, current_result = stage2["model"], stage2["result"]
            preset_names = ["flat_background", "power_law", "teubner_strey"]
            had_ts = True
        else:
            flags.append(f"ts_rejected:{reason}")
    else:
        stages["stage2"] = {"skipped": "insufficient_points_in_window"}
        flags.append("ts_skipped_insufficient_window")

    stage3 = _stage3_add_gp(q, I, sigma, active_windows, {"result": current_result}, had_ts)
    if stage3 is not None:
        stages["stage3"] = {"redchi": float(stage3["result"].redchi), "mask_n": int(stage3["mask"].sum())}
        current_model, current_result = stage3["model"], stage3["result"]
        preset_names = preset_names + ["guinier_porod"]
    else:
        stages["stage3"] = {"skipped": "insufficient_points_in_window"}
        flags.append("gp_skipped_insufficient_window")

    best_values = {name: current_result.params[name].value for name in current_result.params}
    stage4 = _stage4_global(q, I, sigma, current_model, best_values, sample_id, multistart_n)
    stages["stage4"] = {"redchi": float(stage4["result"].redchi), "n_multistart": multistart_n}
    assembled_model, assembled_result = stage4["model"], stage4["result"]

    assembled_name = {
        ("flat_background", "power_law"): "BG",
        ("flat_background", "power_law", "teubner_strey"): "BG_TS",
        ("flat_background", "power_law", "teubner_strey", "guinier_porod"): "BG_TS_GP",
    }.get(tuple(preset_names), "+".join(preset_names))

    stage6 = select_best_preset(q, I, sigma, assembled_name, assembled_model, assembled_result,
                                sample_id, multistart_n)
    stages["stage6"] = stage6["ladder"]
    preset_chosen = stage6["chosen"]
    final_model, final_result = stage6["model"], stage6["result"]
    if preset_chosen != assembled_name:
        flags.append(f"ladder_demoted:{assembled_name}->{preset_chosen}")

    no_peak = "teubner_strey" not in {comp.name for _, comp in final_model.components}
    if no_peak and had_ts:
        flags.append("no_peak")  # TS was fit through stages 1-4 but the ladder rejected it on BIC

    diagnostics = compute_diagnostics(final_model, final_result, q, active_windows)
    stages["stage5"] = diagnostics
    flags.extend(diagnostics["flags"])

    return FitResult(
        sample_id=sample_id, preset_chosen=preset_chosen, residual_mode=residual_mode, loss=loss,
        windows=active_windows, sigma_model=hygiene.sigma_model,
        params=_params_to_dict(final_result.params),
        derived=_build_derived(final_model, final_result.params),
        gof=diagnostics["gof"], flags=flags, seeds_used=best_values,
        multistart_n=multistart_n, no_peak=no_peak, stages=stages,
    )
